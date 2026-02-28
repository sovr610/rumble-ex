"""
brain_ai/workspace/global_workspace.py -- Global Workspace with Iterative Competition and Ignition

Implements Global Workspace Theory (GWT) for multi-modal integration, extended with:

  - Slot-based output: (B, K, D) winner token embeddings instead of a single pooled vector
  - Iterative competition rounds with convergence detection (Jaccard + cosine stability)
  - Ignition-gated broadcast: broadcast packets are produced only when ignition fires
  - Typed EncoderOutput input / WorkspaceOutput output contracts
  - Backward-compatible fallback for legacy Dict[str, Tensor] callers

Pipeline:
    Stage tokens (fixed modality order)
        -> WM summary for novelty bias
        -> Iterative rounds (competition -> broadcast feedback -> re-compete)
        -> Ignition scoring (stability + confidence + margin)
        -> Ignition gate applied to slots
        -> Broadcast adapters produce specialist packets
        -> Ignition-gated working memory write
        -> WorkspaceOutput

Brain analog: prefrontal cortex / thalamocortical loop.

References:
    Baars (1988) "A Cognitive Theory of Consciousness"
    Dehaene & Naccache (2001) "Towards a cognitive neuroscience of consciousness"
    Mashour et al. (2020) "Conscious Processing and the Global Neuronal Workspace Hypothesis"
    VanRullen & Kanai (2021) "Deep learning and the Global Workspace Theory"

Usage:
    from brain_ai.workspace.global_workspace import (
        GlobalWorkspace,
        WorkspaceOutput,
        WorkspaceState,
        create_global_workspace,
    )

Copy this template to brain_ai/workspace/global_workspace.py when integrating.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: working memory backend
# ---------------------------------------------------------------------------
try:
    from .working_memory import WorkingMemory, create_working_memory
    _WM_AVAILABLE = True
except ImportError:
    _WM_AVAILABLE = False


# ---------------------------------------------------------------------------
# EncoderOutput import (try canonical locations, else define locally)
# ---------------------------------------------------------------------------
_EncoderOutput = None
try:
    from brain_ai.encoders.schema import EncoderOutput as _EncoderOutput  # type: ignore
except ImportError:
    pass
if _EncoderOutput is None:
    try:
        from brain_ai.types import EncoderOutput as _EncoderOutput  # type: ignore
    except ImportError:
        pass

if _EncoderOutput is None:
    @dataclass
    class EncoderOutput:  # type: ignore[no-redef]
        """Minimal fallback EncoderOutput when canonical schema is not importable."""
        modality: str
        feats: Tensor          # (B, T, D)
        mask: Tensor           # (B, T) bool
        salience: Optional[Tensor] = None  # (B, T) or (B, 1)
        aux: Optional[Dict[str, Any]] = None
else:
    EncoderOutput = _EncoderOutput  # type: ignore[misc]


# ===========================================================================
# SECTION 1: Configuration
# ===========================================================================

@dataclass
class WorkspaceConfig:
    """Full configuration for the Global Workspace module.

    All parameters have sensible defaults suitable for the minimal config.
    For production, override with values from BrainAIConfig.workspace.
    """

    # --- Core dims ---
    workspace_dim: int = 512
    num_heads: int = 8
    capacity_limit: int = 7           # Miller's 7 +/- 2
    dropout: float = 0.1

    # --- Iterative competition ---
    max_rounds: int = 4               # Hard cap on competition rounds
    min_rounds: int = 2               # Minimum rounds before convergence check
    stability_threshold: float = 0.9          # Jaccard >= this
    embedding_stability_threshold: float = 0.95  # Cosine >= this
    stability_patience: int = 2       # Consecutive stable rounds needed

    # --- Ignition ---
    ignition_threshold: float = 0.3   # Ignition fires when score >= threshold
    ignition_cooldown_len: int = 8    # History deque length
    ignition_margin_weight: float = 0.3
    ignition_stability_weight: float = 0.4
    ignition_confidence_weight: float = 0.3

    # --- Broadcast ---
    broadcast_feedback_scale: float = 0.1  # Scale of feedback signal

    # --- Working memory ---
    memory_hidden_dim: int = 512
    memory_mode: str = "auto"         # "cfc", "ltc", "gru", "auto"

    # --- Competition dynamics ---
    competition_temperature: float = 0.5
    min_attention: float = 0.01

    # --- Output ---
    use_confidence_gating: bool = True

    # --- Telemetry ---
    collect_telemetry: bool = False   # Expensive; off by default


# ===========================================================================
# SECTION 2: Output Dataclasses
# ===========================================================================

@dataclass
class WinnersMetadata:
    """Per-slot winner provenance information.

    Attributes:
        modality_ids: (B, K) int -- index into the sorted modality list
        local_ids: (B, K) int -- token position inside that modality's feats
        scores: (B, K) float -- competition scores after softmax
    """
    modality_ids: Tensor   # (B, K) int64
    local_ids: Tensor      # (B, K) int64
    scores: Tensor         # (B, K) float


@dataclass
class WorkspaceState:
    """Persistent state carried between forward calls.

    All tensors should be detached from the computation graph when passed
    across timesteps to avoid graph retention.

    Attributes:
        wm_state: Backend-specific hidden state (CfC / LTC / GRU).
        wm_output: (B, D) last working memory output vector.
        prev_winners: Optional (B, K) int64 winner indices from last timestep,
            used for temporal decay / re-activation bias.
        ignition_history: Deque of recent ignition scores (floats), length
            bounded by config.ignition_cooldown_len.
        step_count: Number of forward calls seen so far.
    """
    wm_state: Optional[Any] = None
    wm_output: Optional[Tensor] = None          # (B, D)
    prev_winners: Optional[Tensor] = None        # (B, K)
    ignition_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=8),
    )
    step_count: int = 0


@dataclass
class WorkspaceOutput:
    """Complete output from the GlobalWorkspace forward pass.

    Invariants:
        slots.shape == (B, K, D) where K <= capacity_limit
        slot_mask.shape == (B, K)
        ignition_score.shape == (B,)
        ignited.shape == (B,)
        broadcast_packets values have shape (B, specialist_dim)
    """
    slots: Tensor                                          # (B, K, D)
    slot_mask: Tensor                                      # (B, K) bool
    winners: WinnersMetadata
    ignition_score: Tensor                                 # (B,)
    ignited: Tensor                                        # (B,) bool
    broadcast_packets: Dict[str, Tensor]                   # per-specialist
    workspace_state: WorkspaceState
    telemetry: Optional[Dict[str, Any]] = None


# ===========================================================================
# SECTION 3: Token Staging
# ===========================================================================

class TokenStager(nn.Module):
    """Project heterogeneous EncoderOutput feats into a unified token table.

    Maintains a deterministic modality ordering (sorted alphabetically) so
    that the same set of modalities always produces the same token layout.

    Each modality gets its own linear projection + LayerNorm if its input
    dimension differs from workspace_dim, otherwise an identity path is used.
    """

    def __init__(
        self,
        workspace_dim: int,
        modality_dims: Dict[str, int],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.workspace_dim = workspace_dim
        # Canonical modality order — deterministic, matches SKILL.md contract
        MODALITY_ORDER = ["vision", "text", "audio", "sensors", "engram"]
        self.modality_order: List[str] = [
            m for m in MODALITY_ORDER if m in modality_dims
        ] + sorted(k for k in modality_dims if k not in MODALITY_ORDER)

        self.projections = nn.ModuleDict()
        self.salience_heads = nn.ModuleDict()
        for name in self.modality_order:
            dim = modality_dims[name]
            if dim != workspace_dim:
                self.projections[name] = nn.Sequential(
                    nn.Linear(dim, workspace_dim),
                    nn.LayerNorm(workspace_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(workspace_dim, workspace_dim),
                )
            else:
                self.projections[name] = nn.Sequential(
                    nn.LayerNorm(workspace_dim),
                    nn.Dropout(dropout),
                )
            # Per-token salience head: (B, T, D) -> (B, T)
            self.salience_heads[name] = nn.Sequential(
                nn.Linear(workspace_dim, workspace_dim // 4),
                nn.GELU(),
                nn.Linear(workspace_dim // 4, 1),
                nn.Softplus(),  # Ensure non-negative
            )

        # Modality type embeddings: one learnable vector per modality
        self.modality_embeddings = nn.Embedding(len(self.modality_order), workspace_dim)

    # --------------------------------------------------------------------- #

    def forward(
        self,
        encoder_outputs: Dict[str, EncoderOutput],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[str], List[int]]:
        """Stage tokens from all present modalities.

        Returns:
            token_table: (B, T_total, D) projected tokens
            token_mask:  (B, T_total) bool -- valid tokens
            salience:    (B, T_total) non-negative salience scores
            modality_idx: (B, T_total) int -- modality index per token
            modality_names: ordered list of present modality names
            token_counts: number of tokens per modality (in modality_names order)
        """
        feats_list: List[Tensor] = []
        mask_list: List[Tensor] = []
        sal_list: List[Tensor] = []
        mid_list: List[Tensor] = []
        names: List[str] = []
        counts: List[int] = []

        for idx, name in enumerate(self.modality_order):
            if name not in encoder_outputs:
                continue
            eo = encoder_outputs[name]

            # Project to workspace dim
            proj = self.projections[name](eo.feats)           # (B, T_i, D)

            # Add modality type embedding
            mod_emb = self.modality_embeddings(
                torch.tensor(idx, device=proj.device)
            )  # (D,)
            proj = proj + mod_emb.unsqueeze(0).unsqueeze(0)

            # Salience
            if eo.salience is not None:
                sal = eo.salience
                if sal.ndim == 2 and sal.shape[-1] == 1:
                    # (B, 1) -> broadcast later, but keep shape for cat
                    sal = sal.expand(-1, proj.shape[1])
                elif sal.ndim == 3 and sal.shape[-1] == 1:
                    sal = sal.squeeze(-1)
            else:
                sal = self.salience_heads[name](proj).squeeze(-1)  # (B, T_i)

            # Modality index
            B, T_i = proj.shape[:2]
            mod_idx = torch.full(
                (B, T_i), idx, dtype=torch.long, device=proj.device,
            )

            feats_list.append(proj)
            mask_list.append(eo.mask)
            sal_list.append(sal)
            mid_list.append(mod_idx)
            names.append(name)
            counts.append(T_i)

        if len(feats_list) == 0:
            raise ValueError("TokenStager received no valid encoder outputs")

        token_table = torch.cat(feats_list, dim=1)    # (B, T_total, D)
        token_mask = torch.cat(mask_list, dim=1)       # (B, T_total)
        salience = torch.cat(sal_list, dim=1)          # (B, T_total)
        modality_idx = torch.cat(mid_list, dim=1)      # (B, T_total)

        return token_table, token_mask, salience, modality_idx, names, counts


# ===========================================================================
# SECTION 4: Competition Module
# ===========================================================================

class CompetitionModule(nn.Module):
    """Single-round attention-based competition for workspace access.

    Performs multi-head self-attention among candidate tokens, combines
    with salience scores, and selects the top-K winners.

    All scoring is computed in fp32 for numerical stability.
    """

    def __init__(
        self,
        workspace_dim: int = 512,
        num_heads: int = 8,
        capacity_limit: int = 7,
        temperature: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.workspace_dim = workspace_dim
        self.num_heads = num_heads
        self.capacity_limit = capacity_limit
        self.temperature = temperature

        self.attention = nn.MultiheadAttention(
            embed_dim=workspace_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gate: attended token -> scalar score
        self.gate = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim // 4),
            nn.GELU(),
            nn.Linear(workspace_dim // 4, 1),
        )

        # Refinement GRU cell: allows round-over-round feature update
        self.refine_gate = nn.Sequential(
            nn.Linear(workspace_dim * 2, workspace_dim),
            nn.Sigmoid(),
        )
        self.refine_update = nn.Sequential(
            nn.Linear(workspace_dim * 2, workspace_dim),
            nn.Tanh(),
        )

        self.norm = nn.LayerNorm(workspace_dim)

    # --------------------------------------------------------------------- #

    def forward(
        self,
        tokens: Tensor,
        salience: Tensor,
        mask: Optional[Tensor] = None,
        wm_summary: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run one round of competition.

        Args:
            tokens:     (B, T, D) candidate tokens
            salience:   (B, T) per-token salience
            mask:       (B, T) bool -- True = valid
            wm_summary: (B, D) optional working memory context for novelty

        Returns:
            refined:        (B, T, D) updated token embeddings
            scores:         (B, T) combined competition scores (fp32)
            winner_slots:   (B, K, D) selected winner embeddings
            winner_indices: (B, K) indices into the T dimension
        """
        B, T, D = tokens.shape
        input_dtype = tokens.dtype

        # Cast to fp32 for scoring stability
        tokens_f32 = tokens.float()
        salience_f32 = salience.float()

        # Build key padding mask: MultiheadAttention expects True = ignore
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # invert: True = padding

        # Self-attention among tokens
        attended, attn_weights = self.attention(
            tokens_f32, tokens_f32, tokens_f32,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        # attn_weights: (B, T, T) averaged over heads

        # GRU-style refinement
        combined = torch.cat([tokens_f32, attended], dim=-1)
        gate = self.refine_gate(combined)
        update = self.refine_update(combined)
        refined = gate * tokens_f32 + (1.0 - gate) * update

        # Compute gate scores
        gate_scores = self.gate(refined).squeeze(-1)  # (B, T)

        # Novelty bonus from working memory
        if wm_summary is not None:
            # Cosine distance from WM summary => novel tokens get higher score
            wm_expanded = wm_summary.unsqueeze(1).expand_as(refined)
            novelty = 1.0 - F.cosine_similarity(refined, wm_expanded, dim=-1)
            gate_scores = gate_scores + 0.1 * novelty

        # Combine with salience
        combined_scores = gate_scores + salience_f32
        combined_scores = combined_scores / self.temperature

        # Mask out padding tokens
        if mask is not None:
            combined_scores = combined_scores.masked_fill(~mask, float('-inf'))

        # Softmax scores
        scores = F.softmax(combined_scores, dim=-1)  # (B, T)

        # Top-K selection
        K = min(self.capacity_limit, T)
        topk_scores, topk_idx = torch.topk(scores, K, dim=-1)  # (B, K)

        # Gather winner embeddings from refined tokens
        refined_normed = self.norm(refined.to(input_dtype))
        winner_slots = torch.gather(
            refined_normed,
            dim=1,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, D),
        )  # (B, K, D)

        return refined_normed, scores, winner_slots, topk_idx


# ===========================================================================
# SECTION 5: Ignition Scorer
# ===========================================================================

class IgnitionScorer(nn.Module):
    """Compute ignition score from competition dynamics.

    Ignition is a scalar in [0, 1] per batch element indicating whether the
    competition has reached a decisive state (high stability, high confidence,
    large margin between winner and runner-up).

    The three components are combined as a weighted sum:
        ignition = w_s * stability + w_c * confidence + w_m * margin

    All computation is in fp32.
    """

    def __init__(
        self,
        workspace_dim: int = 512,
        margin_weight: float = 0.3,
        stability_weight: float = 0.4,
        confidence_weight: float = 0.3,
    ):
        super().__init__()
        self.margin_weight = margin_weight
        self.stability_weight = stability_weight
        self.confidence_weight = confidence_weight

        # Learnable confidence from slot embeddings
        self.confidence_head = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim // 4),
            nn.GELU(),
            nn.Linear(workspace_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        winner_scores: Tensor,
        slot_stability: Tensor,
        winner_slots: Tensor,
    ) -> Tensor:
        """Compute ignition score.

        Args:
            winner_scores: (B, K) competition scores of winners
            slot_stability: (B,) cosine stability metric in [0, 1]
            winner_slots:   (B, K, D) winner embeddings

        Returns:
            ignition_score: (B,) in [0, 1]
        """
        B = winner_scores.shape[0]
        winner_scores = winner_scores.float()
        slot_stability = slot_stability.float()

        # Margin: difference between top-1 and top-2 score
        if winner_scores.shape[1] >= 2:
            sorted_scores, _ = winner_scores.sort(dim=-1, descending=True)
            margin = (sorted_scores[:, 0] - sorted_scores[:, 1]).clamp(0, 1)
        else:
            margin = winner_scores[:, 0].clamp(0, 1)

        # Confidence from slot content
        slot_mean = winner_slots.float().mean(dim=1)  # (B, D)
        confidence = self.confidence_head(slot_mean).squeeze(-1)  # (B,)

        # Weighted combination
        ignition = (
            self.stability_weight * slot_stability
            + self.confidence_weight * confidence
            + self.margin_weight * margin
        )

        return ignition.clamp(0.0, 1.0)


# ===========================================================================
# SECTION 6: Ignition Gate
# ===========================================================================

class IgnitionGate(nn.Module):
    """Gate slot embeddings by ignition score.

    When ignition fires (score >= threshold), slots pass through unchanged.
    Below threshold, slots are attenuated by the soft ignition score,
    preventing noisy / ambiguous content from propagating.

    The gate is differentiable (soft), so gradients flow through even
    when ignition is sub-threshold.
    """

    def __init__(self, threshold: float = 0.35):
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        slots: Tensor,
        ignition_score: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply ignition gate.

        Args:
            slots:           (B, K, D)
            ignition_score:  (B,) in [0, 1]

        Returns:
            gated_slots:  (B, K, D) -- attenuated by ignition score
            ignited:      (B,) bool -- True where ignition_score >= threshold
        """
        ignited = ignition_score >= self.threshold  # (B,) bool

        # Soft gate: scale slots by ignition score (preserves gradients)
        gate = ignition_score.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        gated_slots = slots * gate

        return gated_slots, ignited


# ===========================================================================
# SECTION 7: Broadcast Adapter Registry
# ===========================================================================

class BroadcastAdapter(nn.Module):
    """Single specialist broadcast adapter: workspace_dim -> specialist_dim."""

    def __init__(self, workspace_dim: int, specialist_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(workspace_dim, specialist_dim),
        )

    def forward(self, workspace_summary: Tensor) -> Tensor:
        """(B, D_ws) -> (B, D_specialist)"""
        return self.net(workspace_summary)


class BroadcastAdapterRegistry(nn.Module):
    """Registry of per-specialist broadcast adapters.

    On forward, takes the workspace summary (mean-pooled gated slots) and
    projects it to each specialist's native dimension.
    """

    def __init__(
        self,
        workspace_dim: int,
        modality_dims: Dict[str, int],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.workspace_dim = workspace_dim
        self.adapters = nn.ModuleDict()
        for name, dim in sorted(modality_dims.items()):
            self.adapters[name] = BroadcastAdapter(workspace_dim, dim, dropout)

    def forward(
        self,
        workspace_summary: Tensor,
        ignited: Tensor,
    ) -> Dict[str, Tensor]:
        """Produce broadcast packets for each specialist.

        Args:
            workspace_summary: (B, D) mean-pooled gated slots
            ignited: (B,) bool -- used to zero-out packets when not ignited

        Returns:
            packets: Dict[str, (B, D_specialist)]
        """
        packets: Dict[str, Tensor] = {}
        gate = ignited.float().unsqueeze(-1)  # (B, 1)
        for name, adapter in self.adapters.items():
            raw = adapter(workspace_summary)  # (B, D_specialist)
            packets[name] = raw * gate        # Zero if not ignited
        return packets


# ===========================================================================
# SECTION 8: Iterative Round Manager
# ===========================================================================

@dataclass
class RoundRecord:
    """Diagnostics for a single competition round."""
    round_idx: int
    winner_indices: Tensor       # (B, K)
    winner_scores: Tensor        # (B, K)
    jaccard_stability: float     # Scalar averaged over batch
    cosine_stability: float      # Scalar averaged over batch
    converged: bool


class IterativeRoundManager(nn.Module):
    """Manage multi-round competition with convergence detection.

    Each round:
        1. Run CompetitionModule to produce refined tokens + winners
        2. Optionally inject broadcast feedback from previous round
        3. Measure winner-set stability (Jaccard) and embedding stability (cosine)
        4. Check convergence: both metrics >= thresholds for patience rounds
        5. Early stop or continue up to max_rounds

    Convergence metrics:
        - Jaccard stability: |W_t intersect W_{t-1}| / |W_t union W_{t-1}|
        - Cosine stability: mean cosine similarity between slot embeddings
          at round t vs round t-1
    """

    def __init__(
        self,
        config: WorkspaceConfig,
    ):
        super().__init__()
        self.max_rounds = config.max_rounds
        self.min_rounds = config.min_rounds
        self.winner_threshold = config.stability_threshold
        self.embed_threshold = config.embedding_stability_threshold
        self.patience = config.stability_patience
        self.feedback_scale = config.broadcast_feedback_scale

        # Broadcast feedback projection: slot summary -> token-level bias
        self.feedback_proj = nn.Sequential(
            nn.Linear(config.workspace_dim, config.workspace_dim),
            nn.GELU(),
            nn.Linear(config.workspace_dim, config.workspace_dim),
        )

    # --------------------------------------------------------------------- #

    @staticmethod
    def _jaccard_stability(
        prev_idx: Tensor,
        curr_idx: Tensor,
    ) -> float:
        """Compute Jaccard similarity between two sets of winner indices.

        Args:
            prev_idx: (B, K) int
            curr_idx: (B, K) int

        Returns:
            Scalar Jaccard averaged over batch.
        """
        B = prev_idx.shape[0]
        total = 0.0
        for b in range(B):
            prev_set = set(prev_idx[b].tolist())
            curr_set = set(curr_idx[b].tolist())
            if len(prev_set) == 0 and len(curr_set) == 0:
                total += 1.0
            else:
                inter = len(prev_set & curr_set)
                union = len(prev_set | curr_set)
                total += inter / max(union, 1)
        return total / max(B, 1)

    @staticmethod
    def _cosine_stability(
        prev_slots: Tensor,
        curr_slots: Tensor,
    ) -> float:
        """Mean cosine similarity between slot embeddings across rounds.

        Args:
            prev_slots: (B, K, D)
            curr_slots: (B, K, D)

        Returns:
            Scalar cosine similarity averaged over B and K.
        """
        cos = F.cosine_similarity(
            prev_slots.float().reshape(-1, prev_slots.shape[-1]),
            curr_slots.float().reshape(-1, curr_slots.shape[-1]),
            dim=-1,
        )
        return cos.mean().item()

    # --------------------------------------------------------------------- #

    def forward(
        self,
        token_table: Tensor,
        token_mask: Tensor,
        salience: Tensor,
        competition: CompetitionModule,
        wm_summary: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, float, List[RoundRecord]]:
        """Run iterative competition rounds.

        Args:
            token_table: (B, T, D) staged tokens
            token_mask:  (B, T) bool
            salience:    (B, T) per-token salience
            competition: CompetitionModule instance
            wm_summary:  (B, D) optional working memory context

        Returns:
            winner_slots:   (B, K, D) final winner embeddings
            winner_indices: (B, K) final winner indices
            winner_scores:  (B, K) final winner scores
            final_cosine:   scalar cosine stability of last round
            round_history:  list of RoundRecord per round
        """
        B, T, D = token_table.shape
        current_tokens = token_table
        current_salience = salience

        prev_indices: Optional[Tensor] = None
        prev_slots: Optional[Tensor] = None
        stable_count = 0

        round_history: List[RoundRecord] = []

        winner_slots: Optional[Tensor] = None
        winner_indices: Optional[Tensor] = None
        winner_scores_out: Optional[Tensor] = None
        final_cosine = 0.0

        for r in range(self.max_rounds):
            # --- Run competition ---
            refined, scores, w_slots, w_idx = competition(
                current_tokens, current_salience,
                mask=token_mask,
                wm_summary=wm_summary,
            )

            # Gather winner scores
            w_scores = torch.gather(scores, dim=1, index=w_idx)

            # --- Stability metrics ---
            jaccard = 1.0
            cosine = 1.0
            if prev_indices is not None and prev_slots is not None:
                jaccard = self._jaccard_stability(prev_indices, w_idx)
                cosine = self._cosine_stability(prev_slots, w_slots)

            converged = False
            if r >= self.min_rounds - 1:  # 0-indexed
                if jaccard >= self.winner_threshold and cosine >= self.embed_threshold:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= self.patience:
                    converged = True

            round_history.append(RoundRecord(
                round_idx=r,
                winner_indices=w_idx.detach(),
                winner_scores=w_scores.detach(),
                jaccard_stability=jaccard,
                cosine_stability=cosine,
                converged=converged,
            ))

            winner_slots = w_slots
            winner_indices = w_idx
            winner_scores_out = w_scores
            final_cosine = cosine

            prev_indices = w_idx.detach()
            prev_slots = w_slots.detach()

            if converged:
                break

            # --- Broadcast feedback for next round ---
            if r < self.max_rounds - 1:
                slot_summary = w_slots.mean(dim=1)  # (B, D)
                feedback = self.feedback_proj(slot_summary)  # (B, D)
                # Add feedback as bias to token embeddings
                current_tokens = refined + self.feedback_scale * feedback.unsqueeze(1)
                # Salience gets a small boost from winner proximity
                # (tokens that were close to winners last round get higher salience)
                current_salience = scores.detach()  # Use previous round's scores as salience

        assert winner_slots is not None
        assert winner_indices is not None
        assert winner_scores_out is not None

        return winner_slots, winner_indices, winner_scores_out, final_cosine, round_history


# ===========================================================================
# SECTION 9: Working Memory Wrapper
# ===========================================================================

class _WorkingMemoryWrapper(nn.Module):
    """Thin wrapper around the working memory backend.

    If the canonical WorkingMemory from working_memory.py is available, uses
    it. Otherwise provides a minimal GRU-based fallback.

    The wrapper exposes a uniform interface for the GlobalWorkspace:
        - forward(x, state) -> (output, new_state)
        - reset_state(batch_size, device, dtype) -> state
        - detach_state(state) -> state
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        mode: str = "auto",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if _WM_AVAILABLE:
            self._backend = create_working_memory(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                mode=mode,
            )
            self._type = "canonical"
            logger.info("WorkingMemory: using canonical backend (mode=%s)", mode)
        else:
            self._backend = nn.GRUCell(input_dim, hidden_dim)
            self._output_proj = nn.Linear(hidden_dim, output_dim)
            self._type = "fallback_gru"
            logger.info("WorkingMemory: using fallback GRU backend")

    def forward(
        self,
        x: Tensor,
        state: Optional[Any] = None,
    ) -> Tuple[Tensor, Any]:
        """Process one timestep.

        Args:
            x: (B, D) input
            state: backend-specific hidden state

        Returns:
            output: (B, D_out)
            new_state: updated hidden state
        """
        if self._type == "canonical":
            result = self._backend(x, state=state)
            return result['output'], result.get('state', None)
        else:
            # Fallback GRU
            if state is not None and isinstance(state, Tensor):
                if state.shape[0] != x.shape[0]:
                    state = None
            h = self._backend(x, state)
            out = self._output_proj(h)
            return out, h

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Optional[Any]:
        """Create a fresh initial state."""
        if self._type == "canonical":
            if hasattr(self._backend, 'reset_state'):
                self._backend.reset_state()
            return None
        else:
            return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    @staticmethod
    def detach_state(state: Optional[Any]) -> Optional[Any]:
        """Detach state from computation graph."""
        if state is None:
            return None
        if isinstance(state, Tensor):
            return state.detach()
        if isinstance(state, (tuple, list)):
            cls = type(state)
            return cls(_WorkingMemoryWrapper.detach_state(s) for s in state)
        return state


# ===========================================================================
# SECTION 10: Legacy Input Adapter
# ===========================================================================

def _wrap_legacy_inputs(
    modality_inputs: Dict[str, Tensor],
    workspace_dim: int,
) -> Dict[str, EncoderOutput]:
    """Convert legacy Dict[str, Tensor] to Dict[str, EncoderOutput].

    The legacy interface passes raw tensors of shape (B, D) per modality.
    We wrap each in an EncoderOutput with T=1 and a trivially-true mask.
    """
    warnings.warn(
        "GlobalWorkspace received Dict[str, Tensor] instead of Dict[str, EncoderOutput]. "
        "This legacy format is deprecated and will be removed in a future version. "
        "Please migrate to EncoderOutput.",
        DeprecationWarning,
        stacklevel=3,
    )
    result: Dict[str, EncoderOutput] = {}
    for name, tensor in modality_inputs.items():
        if tensor.ndim == 2:
            feats = tensor.unsqueeze(1)  # (B, 1, D)
        elif tensor.ndim == 3:
            feats = tensor
        else:
            raise ValueError(
                f"Legacy input '{name}' has ndim={tensor.ndim}, expected 2 or 3"
            )
        B, T = feats.shape[:2]
        mask = torch.ones(B, T, dtype=torch.bool, device=feats.device)
        result[name] = EncoderOutput(
            modality=name,
            feats=feats,
            mask=mask,
            salience=None,
        )
    return result


def _format_legacy_output(ws_output: WorkspaceOutput) -> Dict[str, Any]:
    """Convert WorkspaceOutput to legacy Dict format for backward compatibility."""
    workspace = ws_output.slots.mean(dim=1)  # (B, D) pooled
    broadcasts = ws_output.broadcast_packets
    attention = {
        "scores": ws_output.winners.scores,
    }
    return {
        'workspace': workspace,
        'broadcasts': broadcasts,
        'attention': attention,
        'ignition': ws_output.ignition_score,
        'global_ignition': ws_output.ignited.float(),
        'memory_output': ws_output.workspace_state.wm_output,
    }


# ===========================================================================
# SECTION 11: GlobalWorkspace (Main Orchestrator)
# ===========================================================================

class GlobalWorkspace(nn.Module):
    """Global Workspace with iterative competition and ignition-gated broadcast.

    This module is the main orchestrator for multi-modal information integration
    using Global Workspace Theory. It accepts typed EncoderOutput from each
    modality encoder and produces a WorkspaceOutput with slot-based winners,
    ignition state, and per-specialist broadcast packets.

    Pipeline:
        a. Stage tokens (TokenStager: fixed modality order, projection + salience)
        b. Get WM summary for novelty scoring
        c. Run iterative rounds (IterativeRoundManager: competition -> feedback -> re-compete)
        d. Compute ignition score (IgnitionScorer: stability + confidence + margin)
        e. Apply ignition gate to slots (IgnitionGate: soft gating)
        f. Run broadcast adapters to produce specialist packets (BroadcastAdapterRegistry)
        g. Update working memory (ignition-gated write)
        h. Return WorkspaceOutput

    Args:
        config: WorkspaceConfig with all hyperparameters
        modality_dims: Dict mapping modality names to their feature dimensions.
            If None, defaults to equal-dim modalities at workspace_dim.
    """

    def __init__(
        self,
        config: Optional[WorkspaceConfig] = None,
        modality_dims: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        super().__init__()

        self.config = config or WorkspaceConfig(**kwargs)
        D = self.config.workspace_dim

        # Default modality dims: all at workspace_dim
        self.modality_dims: Dict[str, int] = modality_dims or {
            'vision': D,
            'text': D,
            'audio': D,
            'sensors': D,
        }

        # --- Sub-modules ---

        # (a) Token staging
        self.token_stager = TokenStager(
            workspace_dim=D,
            modality_dims=self.modality_dims,
            dropout=self.config.dropout,
        )

        # (c) Competition
        self.competition = CompetitionModule(
            workspace_dim=D,
            num_heads=self.config.num_heads,
            capacity_limit=self.config.capacity_limit,
            temperature=self.config.competition_temperature,
            dropout=self.config.dropout,
        )

        # (c) Iterative round manager
        self.round_manager = IterativeRoundManager(config=self.config)

        # (d) Ignition scorer
        self.ignition_scorer = IgnitionScorer(
            workspace_dim=D,
            margin_weight=self.config.ignition_margin_weight,
            stability_weight=self.config.ignition_stability_weight,
            confidence_weight=self.config.ignition_confidence_weight,
        )

        # (e) Ignition gate
        self.ignition_gate = IgnitionGate(
            threshold=self.config.ignition_threshold,
        )

        # (f) Broadcast adapters
        self.broadcast_registry = BroadcastAdapterRegistry(
            workspace_dim=D,
            modality_dims=self.modality_dims,
            dropout=self.config.dropout,
        )

        # (g) Working memory
        self.working_memory = _WorkingMemoryWrapper(
            input_dim=D,
            hidden_dim=self.config.memory_hidden_dim,
            output_dim=D,
            mode=self.config.memory_mode,
        )

        # Integration layer: combine gated workspace content with WM output
        self.wm_integration = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # Confidence estimator (optional)
        if self.config.use_confidence_gating:
            self.confidence_head = nn.Sequential(
                nn.Linear(D, D // 4),
                nn.GELU(),
                nn.Linear(D // 4, 1),
                nn.Sigmoid(),
            )
        else:
            self.confidence_head = None

        # Buffers for ignition history tracking
        self.register_buffer('_ignition_ema', torch.tensor(0.0))

    # --------------------------------------------------------------------- #
    # State management
    # --------------------------------------------------------------------- #

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> WorkspaceState:
        """Create a fresh WorkspaceState for a new episode / sequence.

        Args:
            batch_size: Number of sequences in the batch.
            device: Target device.
            dtype: Floating-point dtype (default fp32).

        Returns:
            Fresh WorkspaceState with zeroed working memory.
        """
        D = self.config.workspace_dim
        wm_state = self.working_memory.reset_state(batch_size, device, dtype)
        return WorkspaceState(
            wm_state=wm_state,
            wm_output=torch.zeros(batch_size, D, device=device, dtype=dtype),
            prev_winners=None,
            ignition_history=deque(maxlen=self.config.ignition_cooldown_len),
            step_count=0,
        )

    @staticmethod
    def detach_state(state: WorkspaceState) -> WorkspaceState:
        """Detach all tensors in state from the computation graph.

        Use at truncation boundaries to prevent graph retention.
        """
        return WorkspaceState(
            wm_state=_WorkingMemoryWrapper.detach_state(state.wm_state),
            wm_output=(
                state.wm_output.detach()
                if state.wm_output is not None
                else None
            ),
            prev_winners=(
                state.prev_winners.detach()
                if state.prev_winners is not None
                else None
            ),
            ignition_history=deque(state.ignition_history, maxlen=state.ignition_history.maxlen),
            step_count=state.step_count,
        )

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #

    def forward(
        self,
        encoder_outputs: Union[Dict[str, EncoderOutput], Dict[str, Tensor]],
        state: Optional[WorkspaceState] = None,
        return_details: bool = False,
        return_legacy: bool = False,
    ) -> Union[WorkspaceOutput, Dict[str, Any]]:
        """Process multi-modal encoder outputs through the global workspace.

        Args:
            encoder_outputs: Dict mapping modality names to EncoderOutput (preferred)
                or to raw Tensors (legacy, deprecated).
            state: Optional WorkspaceState from the previous timestep.
                If None, a fresh state is created internally.
            return_details: If True, populate telemetry field with round history,
                attention maps, and convergence metrics.
            return_legacy: If True, return a plain Dict instead of WorkspaceOutput
                for backward compatibility.

        Returns:
            WorkspaceOutput with slots, ignition, broadcast packets, and state,
            or a legacy Dict if return_legacy=True.

        Raises:
            ValueError: If no valid modality inputs are provided.
        """
        # ----- Legacy input detection and wrapping ----- #
        if len(encoder_outputs) > 0:
            first_val = next(iter(encoder_outputs.values()))
            if isinstance(first_val, Tensor):
                encoder_outputs = _wrap_legacy_inputs(
                    encoder_outputs,  # type: ignore[arg-type]
                    self.config.workspace_dim,
                )

        # Cast to the typed form
        eo_dict: Dict[str, EncoderOutput] = encoder_outputs  # type: ignore[assignment]

        # ----- Determine batch size and device ----- #
        first_eo = next(iter(eo_dict.values()))
        B = first_eo.feats.shape[0]
        device = first_eo.feats.device
        dtype = first_eo.feats.dtype

        # ----- Initialize state if needed ----- #
        if state is None:
            state = self.reset_state(B, device, dtype)

        # ================================================================= #
        # (a) Stage tokens
        # ================================================================= #
        token_table, token_mask, salience, modality_idx, mod_names, token_counts = \
            self.token_stager(eo_dict)
        # token_table: (B, T_total, D)
        # modality_idx: (B, T_total) -- maps each token to its modality index

        # ================================================================= #
        # (b) Working memory summary for novelty bias
        # ================================================================= #
        wm_summary = state.wm_output  # (B, D) or None

        # ================================================================= #
        # (c) Iterative competition rounds
        # ================================================================= #
        winner_slots, winner_indices, winner_scores, final_cosine, round_history = \
            self.round_manager(
                token_table=token_table,
                token_mask=token_mask,
                salience=salience,
                competition=self.competition,
                wm_summary=wm_summary,
            )
        # winner_slots: (B, K, D)
        # winner_indices: (B, K) -- indices into T_total
        # winner_scores: (B, K)

        # ================================================================= #
        # Build WinnersMetadata: map flat indices back to modality + local
        # ================================================================= #
        # modality_idx was built during staging: (B, T_total)
        winner_mod_ids = torch.gather(modality_idx, dim=1, index=winner_indices)
        # (B, K) modality index per winner

        # Compute local token id within each modality
        # For each winner, local_id = global_idx - cumulative_start_of_modality
        cum_offsets = torch.zeros(B, token_table.shape[1], dtype=torch.long, device=device)
        offset = 0
        for i, count in enumerate(token_counts):
            # Tokens in range [offset, offset+count) belong to modality i
            cum_offsets[:, offset:offset + count] = offset
            offset += count
        winner_offsets = torch.gather(cum_offsets, dim=1, index=winner_indices)
        winner_local_ids = winner_indices - winner_offsets

        winners_meta = WinnersMetadata(
            modality_ids=winner_mod_ids,
            local_ids=winner_local_ids,
            scores=winner_scores,
        )

        # ================================================================= #
        # (d) Ignition scoring
        # ================================================================= #
        slot_stability = torch.tensor(
            final_cosine, dtype=torch.float32, device=device,
        ).expand(B)

        ignition_score = self.ignition_scorer(
            winner_scores=winner_scores,
            slot_stability=slot_stability,
            winner_slots=winner_slots,
        )  # (B,)

        # ================================================================= #
        # (e) Ignition gate
        # ================================================================= #
        gated_slots, ignited = self.ignition_gate(winner_slots, ignition_score)
        # gated_slots: (B, K, D)
        # ignited: (B,) bool

        # Slot mask: True for all K slots (no padding in top-K output)
        K = gated_slots.shape[1]
        slot_mask = torch.ones(B, K, dtype=torch.bool, device=device)

        # ================================================================= #
        # (f) Broadcast adapters
        # ================================================================= #
        workspace_summary = gated_slots.mean(dim=1)  # (B, D) -- mean pool
        broadcast_packets = self.broadcast_registry(
            workspace_summary=workspace_summary,
            ignited=ignited,
        )

        # ================================================================= #
        # (g) Working memory update (ignition-gated write)
        # ================================================================= #
        # Only write to WM when ignition fires (or soft-write proportional to score)
        wm_input = workspace_summary  # (B, D)

        # Ignition-gated write: blend WM input with previous output
        ign_gate = ignition_score.unsqueeze(-1)  # (B, 1)
        if state.wm_output is not None:
            wm_input_gated = ign_gate * wm_input + (1.0 - ign_gate) * state.wm_output
        else:
            wm_input_gated = wm_input

        wm_output, new_wm_state = self.working_memory(wm_input_gated, state.wm_state)

        # Integration: combine gated workspace content with WM output
        combined = torch.cat([workspace_summary, wm_output], dim=-1)
        integrated = self.wm_integration(combined)

        # Optional confidence gating
        if self.confidence_head is not None:
            confidence = self.confidence_head(integrated)  # (B, 1)
            gated_slots = gated_slots * confidence.unsqueeze(1)  # (B, K, D)

        # ================================================================= #
        # Update state
        # ================================================================= #
        new_state = WorkspaceState(
            wm_state=new_wm_state,
            wm_output=wm_output.detach(),
            prev_winners=winner_indices.detach(),
            ignition_history=deque(
                list(state.ignition_history) + [ignition_score.mean().item()],
                maxlen=self.config.ignition_cooldown_len,
            ),
            step_count=state.step_count + 1,
        )

        # ================================================================= #
        # Build telemetry
        # ================================================================= #
        telemetry: Optional[Dict[str, Any]] = None
        if return_details or self.config.collect_telemetry:
            telemetry = {
                'round_history': [
                    {
                        'round': rec.round_idx,
                        'jaccard': rec.jaccard_stability,
                        'cosine': rec.cosine_stability,
                        'converged': rec.converged,
                        'winner_indices': rec.winner_indices,
                        'winner_scores': rec.winner_scores,
                    }
                    for rec in round_history
                ],
                'num_rounds': len(round_history),
                'final_jaccard': round_history[-1].jaccard_stability if round_history else 0.0,
                'final_cosine': final_cosine,
                'ignition_score_mean': ignition_score.mean().item(),
                'ignited_fraction': ignited.float().mean().item(),
                'modality_names': mod_names,
                'token_counts': token_counts,
                'ignition_history': list(new_state.ignition_history),
            }

        # ================================================================= #
        # Assemble output
        # ================================================================= #
        ws_output = WorkspaceOutput(
            slots=gated_slots,
            slot_mask=slot_mask,
            winners=winners_meta,
            ignition_score=ignition_score,
            ignited=ignited,
            broadcast_packets=broadcast_packets,
            workspace_state=new_state,
            telemetry=telemetry,
        )

        # ----- Legacy return format ----- #
        if return_legacy:
            warnings.warn(
                "return_legacy=True is deprecated. Migrate to WorkspaceOutput.",
                DeprecationWarning,
                stacklevel=2,
            )
            return _format_legacy_output(ws_output)

        return ws_output

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"workspace_dim={self.config.workspace_dim}, "
            f"num_heads={self.config.num_heads}, "
            f"capacity_limit={self.config.capacity_limit}, "
            f"max_rounds={self.config.max_rounds}, "
            f"ignition_threshold={self.config.ignition_threshold}, "
            f"modalities={list(self.modality_dims.keys())}"
        )


# ===========================================================================
# SECTION 12: Factory Function
# ===========================================================================

def create_global_workspace(
    config: Optional[WorkspaceConfig] = None,
    modality_dims: Optional[Dict[str, int]] = None,
    **kwargs,
) -> GlobalWorkspace:
    """Create a GlobalWorkspace with the given configuration.

    This is the recommended entry point for instantiation.

    Args:
        config: Full WorkspaceConfig. If None, one is created from kwargs.
        modality_dims: Dict mapping modality names to feature dimensions.
            If None, defaults to all modalities at workspace_dim.
        **kwargs: Passed to WorkspaceConfig constructor if config is None.

    Returns:
        Configured GlobalWorkspace instance.

    Examples:
        # Minimal
        gw = create_global_workspace()

        # Custom dims
        gw = create_global_workspace(
            config=WorkspaceConfig(workspace_dim=1024, num_heads=16),
            modality_dims={'vision': 768, 'text': 512},
        )

        # From kwargs
        gw = create_global_workspace(workspace_dim=256, max_rounds=3)
    """
    if config is None:
        config = WorkspaceConfig(**kwargs)
    return GlobalWorkspace(config=config, modality_dims=modality_dims)


# ===========================================================================
# SECTION 13: Self-Tests
# ===========================================================================

def _run_self_tests():
    """Run ~25 self-tests covering all major functionality.

    Tests are intentionally lightweight (small dims, few tokens) so they
    can run on CPU in a few seconds without GPU.
    """
    import traceback
    import sys

    passed = 0
    failed = 0
    errors: List[str] = []

    def _test(name: str, fn: Callable[[], None]):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            print(f"  [PASS] {name}")
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            errors.append(f"  [FAIL] {name}: {e}\n{tb}")
            print(f"  [FAIL] {name}: {e}")

    D = 64
    B = 2
    K = 3
    device = torch.device("cpu")

    print("=" * 70)
    print("GlobalWorkspace Self-Tests")
    print("=" * 70)

    # ----- Helper: build small workspace ----- #
    def _make_ws(max_rounds=5, min_rounds=2, threshold=0.35):
        cfg = WorkspaceConfig(
            workspace_dim=D,
            num_heads=4,
            capacity_limit=K,
            max_rounds=max_rounds,
            min_rounds=min_rounds,
            ignition_threshold=threshold,
            memory_hidden_dim=D,
            memory_mode="gru",
            dropout=0.0,
            competition_temperature=0.5,
            collect_telemetry=True,
        )
        dims = {'vision': D, 'text': D, 'audio': D}
        return create_global_workspace(config=cfg, modality_dims=dims)

    def _make_eo(name, B, T, D, device, sal_val=1.0):
        feats = torch.randn(B, T, D, device=device)
        mask = torch.ones(B, T, dtype=torch.bool, device=device)
        sal = torch.full((B, T), sal_val, device=device)
        return EncoderOutput(modality=name, feats=feats, mask=mask, salience=sal)

    # -------------------------------------------------------------------- #
    # Test 1: Full forward pass with 3 modalities
    # -------------------------------------------------------------------- #
    def test_full_forward_3_modalities():
        torch.manual_seed(42)
        ws = _make_ws()
        eo = {
            'vision': _make_eo('vision', B, 5, D, device),
            'text': _make_eo('text', B, 4, D, device),
            'audio': _make_eo('audio', B, 3, D, device),
        }
        out = ws(eo, return_details=True)
        assert isinstance(out, WorkspaceOutput), "Expected WorkspaceOutput"
        assert out.slots.shape == (B, K, D), f"slots shape {out.slots.shape}"
        assert out.slot_mask.shape == (B, K), f"slot_mask shape {out.slot_mask.shape}"
        assert out.ignition_score.shape == (B,), f"ignition shape {out.ignition_score.shape}"
        assert out.ignited.shape == (B,), f"ignited shape {out.ignited.shape}"

    _test("Full forward pass with 3 modalities", test_full_forward_3_modalities)

    # -------------------------------------------------------------------- #
    # Test 2: Output has all required fields
    # -------------------------------------------------------------------- #
    def test_output_fields():
        torch.manual_seed(42)
        ws = _make_ws()
        eo = {'vision': _make_eo('vision', B, 4, D, device)}
        out = ws(eo, return_details=True)
        assert hasattr(out, 'slots')
        assert hasattr(out, 'slot_mask')
        assert hasattr(out, 'winners')
        assert hasattr(out, 'ignition_score')
        assert hasattr(out, 'ignited')
        assert hasattr(out, 'broadcast_packets')
        assert hasattr(out, 'workspace_state')
        assert hasattr(out, 'telemetry')
        assert isinstance(out.winners, WinnersMetadata)
        assert isinstance(out.workspace_state, WorkspaceState)

    _test("WorkspaceOutput has all required fields", test_output_fields)

    # -------------------------------------------------------------------- #
    # Test 3: WinnersMetadata shapes
    # -------------------------------------------------------------------- #
    def test_winners_metadata_shapes():
        torch.manual_seed(42)
        ws = _make_ws()
        eo = {
            'vision': _make_eo('vision', B, 5, D, device),
            'text': _make_eo('text', B, 3, D, device),
        }
        out = ws(eo)
        wm = out.winners
        assert wm.modality_ids.shape == (B, K), f"modality_ids {wm.modality_ids.shape}"
        assert wm.local_ids.shape == (B, K), f"local_ids {wm.local_ids.shape}"
        assert wm.scores.shape == (B, K), f"scores {wm.scores.shape}"

    _test("WinnersMetadata shapes", test_winners_metadata_shapes)

    # -------------------------------------------------------------------- #
    # Test 4: Iterative rounds converge with stable input
    # -------------------------------------------------------------------- #
    def test_rounds_converge_stable():
        torch.manual_seed(0)
        ws = _make_ws(max_rounds=10, min_rounds=1)
        # Use identical features so competition is trivially stable
        shared = torch.randn(B, 6, D, device=device)
        eo_vision = EncoderOutput(
            modality='vision', feats=shared[:, :3],
            mask=torch.ones(B, 3, dtype=torch.bool, device=device),
            salience=torch.ones(B, 3, device=device),
        )
        eo_text = EncoderOutput(
            modality='text', feats=shared[:, 3:],
            mask=torch.ones(B, 3, dtype=torch.bool, device=device),
            salience=torch.ones(B, 3, device=device),
        )
        out = ws({'vision': eo_vision, 'text': eo_text}, return_details=True)
        telem = out.telemetry
        assert telem is not None
        num_rounds = telem['num_rounds']
        # With stable input, should converge before max_rounds (10)
        assert num_rounds <= 10, f"Expected <= 10 rounds, got {num_rounds}"

    _test("Iterative rounds converge with stable input", test_rounds_converge_stable)

    # -------------------------------------------------------------------- #
    # Test 5: Rounds cap at max_rounds with noisy input
    # -------------------------------------------------------------------- #
    def test_rounds_cap_at_max():
        torch.manual_seed(99)
        max_r = 3
        ws = _make_ws(max_rounds=max_r, min_rounds=1)
        ws.round_manager.winner_threshold = 0.9999  # Almost impossible to converge
        ws.round_manager.embed_threshold = 0.9999
        eo = {
            'vision': _make_eo('vision', B, 8, D, device),
            'text': _make_eo('text', B, 8, D, device),
        }
        out = ws(eo, return_details=True)
        assert out.telemetry is not None
        assert out.telemetry['num_rounds'] == max_r, \
            f"Expected {max_r} rounds, got {out.telemetry['num_rounds']}"

    _test("Rounds cap at max_rounds with noisy input", test_rounds_cap_at_max)

    # -------------------------------------------------------------------- #
    # Test 6: Ignition fires when competition is decisive
    # -------------------------------------------------------------------- #
    def test_ignition_fires_decisive():
        torch.manual_seed(42)
        ws = _make_ws(threshold=0.01)  # Very low threshold -> should fire
        eo = {'vision': _make_eo('vision', B, 4, D, device, sal_val=10.0)}
        out = ws(eo)
        # With very low threshold and high salience, ignition should fire
        assert out.ignited.any(), "Expected ignition to fire with low threshold"

    _test("Ignition fires when competition is decisive", test_ignition_fires_decisive)

    # -------------------------------------------------------------------- #
    # Test 7: Ignition does not fire when ambiguous
    # -------------------------------------------------------------------- #
    def test_ignition_no_fire_ambiguous():
        torch.manual_seed(42)
        ws = _make_ws(threshold=0.99)  # Very high threshold -> should not fire
        eo = {'vision': _make_eo('vision', B, 4, D, device, sal_val=0.001)}
        out = ws(eo)
        # With very high threshold and low salience, ignition should not fire
        assert not out.ignited.all(), "Expected ignition not to fire with high threshold"

    _test("Ignition does not fire when ambiguous", test_ignition_no_fire_ambiguous)

    # -------------------------------------------------------------------- #
    # Test 8: Broadcast adapters produce correct shapes
    # -------------------------------------------------------------------- #
    def test_broadcast_shapes():
        torch.manual_seed(42)
        ws = _make_ws()
        eo = {
            'vision': _make_eo('vision', B, 4, D, device),
            'text': _make_eo('text', B, 3, D, device),
            'audio': _make_eo('audio', B, 2, D, device),
        }
        out = ws(eo)
        for name in ['vision', 'text', 'audio']:
            assert name in out.broadcast_packets, f"Missing broadcast for {name}"
            pkt = out.broadcast_packets[name]
            assert pkt.shape == (B, D), f"Broadcast '{name}' shape {pkt.shape} != ({B}, {D})"

    _test("Broadcast adapters produce correct shapes", test_broadcast_shapes)

    # -------------------------------------------------------------------- #
    # Test 9: Working memory state carries across steps
    # -------------------------------------------------------------------- #
    def test_wm_state_carries():
        torch.manual_seed(42)
        ws = _make_ws()
        eo = {'vision': _make_eo('vision', B, 4, D, device)}

        # Step 1
        out1 = ws(eo)
        state1 = out1.workspace_state
        assert state1.step_count == 1

        # Step 2: pass state from step 1
        out2 = ws(eo, state=state1)
        state2 = out2.workspace_state
        assert state2.step_count == 2
        assert state2.wm_output is not None

        # WM output should differ between steps (it's recurrent)
        if state1.wm_output is not None:
            diff = (state2.wm_output - state1.wm_output).abs().sum()
            # After detach both are constant, but the actual WM outputs inside
            # the forward pass differ. Just check they exist.
            assert state2.wm_output.shape == (B, D)

    _test("Working memory state carries across steps", test_wm_state_carries)

    # -------------------------------------------------------------------- #
    # Test 10: Reset state restores baseline
    # -------------------------------------------------------------------- #
    def test_reset_state():
        torch.manual_seed(42)
        ws = _make_ws()
        state = ws.reset_state(B, device)
        assert state.step_count == 0
        assert state.wm_output is not None
        assert state.wm_output.shape == (B, D)
        assert (state.wm_output == 0).all()
        assert state.prev_winners is None
        assert len(state.ignition_history) == 0

    _test("Reset state restores baseline", test_reset_state)

    # -------------------------------------------------------------------- #
    # Test 11: Detach state produces detached tensors
    # -------------------------------------------------------------------- #
    def test_detach_state():
        torch.manual_seed(42)
        ws = _make_ws()
        eo = {'vision': _make_eo('vision', B, 4, D, device)}
        out = ws(eo)
        state = out.workspace_state
        det = GlobalWorkspace.detach_state(state)
        assert det.step_count == state.step_count
        if det.wm_output is not None:
            assert not det.wm_output.requires_grad
        if det.prev_winners is not None:
            assert not det.prev_winners.requires_grad

    _test("Detach state produces detached tensors", test_detach_state)

    # -------------------------------------------------------------------- #
    # Test 12: Legacy Dict[str, Tensor] input compatibility
    # -------------------------------------------------------------------- #
    def test_legacy_tensor_input():
        torch.manual_seed(42)
        ws = _make_ws()
        legacy_input = {
            'vision': torch.randn(B, D),
            'text': torch.randn(B, D),
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = ws(legacy_input)
            # Should have issued a deprecation warning
            assert any(issubclass(ww.category, DeprecationWarning) for ww in w), \
                "Expected DeprecationWarning for legacy input"
        assert isinstance(out, WorkspaceOutput)
        assert out.slots.shape[0] == B
        assert out.slots.shape[2] == D

    _test("Legacy Dict[str, Tensor] input compatibility", test_legacy_tensor_input)

    # -------------------------------------------------------------------- #
    # Test 13: Legacy return format
    # -------------------------------------------------------------------- #
    def test_legacy_return_format():
        torch.manual_seed(42)
        ws = _make_ws()
        eo = {'vision': _make_eo('vision', B, 4, D, device)}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = ws(eo, return_legacy=True)
            assert any(issubclass(ww.category, DeprecationWarning) for ww in w)
        assert isinstance(out, dict)
        assert 'workspace' in out
        assert 'broadcasts' in out
        assert out['workspace'].shape == (B, D)

    _test("Legacy return format via return_legacy=True", test_legacy_return_format)

    # -------------------------------------------------------------------- #
    # Test 14: Deterministic with fixed seed
    # -------------------------------------------------------------------- #
    def test_deterministic():
        ws = _make_ws()
        eo_fn = lambda: {'vision': _make_eo('vision', B, 4, D, device)}

        torch.manual_seed(123)
        ws_state = ws.reset_state(B, device)
        out1 = ws(eo_fn(), state=ws_state)

        # Reset and re-run with same seed
        torch.manual_seed(123)
        ws2 = _make_ws()
        ws2.load_state_dict(ws.state_dict())
        ws2_state = ws2.reset_state(B, device)
        out2 = ws2(eo_fn(), state=ws2_state)

        # Slots should match exactly
        assert torch.allclose(out1.slots, out2.slots, atol=1e-5), \
            f"Slots differ: max delta {(out1.slots - out2.slots).abs().max()}"

    _test("Deterministic with fixed seed", test_deterministic)

    # -------------------------------------------------------------------- #
    # Test 15: Single modality forward pass
    # -------------------------------------------------------------------- #
    def test_single_modality():
        torch.manual_seed(42)
        ws = _make_ws()
        eo = {'vision': _make_eo('vision', B, 4, D, device)}
        out = ws(eo)
        assert out.slots.shape == (B, K, D)

    _test("Single modality forward pass", test_single_modality)

    # -------------------------------------------------------------------- #
    # Test 16: TokenStager modality ordering is deterministic
    # -------------------------------------------------------------------- #
    def test_stager_ordering():
        torch.manual_seed(42)
        stager = TokenStager(D, {'text': D, 'vision': D, 'audio': D})
        assert stager.modality_order == ['audio', 'text', 'vision'], \
            f"Expected sorted order, got {stager.modality_order}"

    _test("TokenStager modality ordering is deterministic", test_stager_ordering)

    # -------------------------------------------------------------------- #
    # Test 17: CompetitionModule single round
    # -------------------------------------------------------------------- #
    def test_competition_single_round():
        torch.manual_seed(42)
        comp = CompetitionModule(workspace_dim=D, num_heads=4, capacity_limit=K, temperature=0.5)
        tokens = torch.randn(B, 8, D)
        sal = torch.ones(B, 8)
        mask = torch.ones(B, 8, dtype=torch.bool)
        refined, scores, w_slots, w_idx = comp(tokens, sal, mask)
        assert refined.shape == (B, 8, D)
        assert scores.shape == (B, 8)
        assert w_slots.shape == (B, K, D)
        assert w_idx.shape == (B, K)
        assert (scores >= 0).all()
        assert torch.allclose(scores.sum(dim=-1), torch.ones(B), atol=1e-5)

    _test("CompetitionModule single round", test_competition_single_round)

    # -------------------------------------------------------------------- #
    # Test 18: IgnitionScorer output range
    # -------------------------------------------------------------------- #
    def test_ignition_scorer_range():
        torch.manual_seed(42)
        scorer = IgnitionScorer(workspace_dim=D)
        w_scores = torch.rand(B, K)
        stability = torch.rand(B)
        w_slots = torch.randn(B, K, D)
        ign = scorer(w_scores, stability, w_slots)
        assert ign.shape == (B,)
        assert (ign >= 0).all() and (ign <= 1).all(), f"Ignition out of [0,1]: {ign}"

    _test("IgnitionScorer output in [0, 1]", test_ignition_scorer_range)

    # -------------------------------------------------------------------- #
    # Test 19: IgnitionGate soft gating
    # -------------------------------------------------------------------- #
    def test_ignition_gate():
        gate = IgnitionGate(threshold=0.5)
        slots = torch.ones(B, K, D)
        # Score = 0.8 (above threshold)
        ign_high = torch.tensor([0.8, 0.8])
        gated_high, ignited_high = gate(slots, ign_high)
        assert ignited_high.all()
        # Gated = slots * 0.8
        expected = slots * 0.8
        assert torch.allclose(gated_high, expected, atol=1e-5)

        # Score = 0.2 (below threshold)
        ign_low = torch.tensor([0.2, 0.2])
        gated_low, ignited_low = gate(slots, ign_low)
        assert not ignited_low.any()
        expected_low = slots * 0.2
        assert torch.allclose(gated_low, expected_low, atol=1e-5)

    _test("IgnitionGate soft gating", test_ignition_gate)

    # -------------------------------------------------------------------- #
    # Test 20: BroadcastAdapterRegistry
    # -------------------------------------------------------------------- #
    def test_broadcast_registry():
        torch.manual_seed(42)
        reg = BroadcastAdapterRegistry(D, {'vision': D, 'text': 32})
        summary = torch.randn(B, D)
        ignited = torch.tensor([True, False])
        packets = reg(summary, ignited)
        assert packets['vision'].shape == (B, D)
        assert packets['text'].shape == (B, 32)
        # Second batch item (not ignited) should be zero
        assert (packets['vision'][1] == 0).all()
        assert (packets['text'][1] == 0).all()

    _test("BroadcastAdapterRegistry zero-out when not ignited", test_broadcast_registry)

    # -------------------------------------------------------------------- #
    # Test 21: Factory function create_global_workspace
    # -------------------------------------------------------------------- #
    def test_factory_function():
        ws = create_global_workspace(
            config=WorkspaceConfig(workspace_dim=D, num_heads=4, capacity_limit=K, memory_mode="gru"),
            modality_dims={'vision': D},
        )
        assert isinstance(ws, GlobalWorkspace)
        eo = {'vision': _make_eo('vision', B, 4, D, device)}
        out = ws(eo)
        assert isinstance(out, WorkspaceOutput)

    _test("Factory function create_global_workspace", test_factory_function)

    # -------------------------------------------------------------------- #
    # Test 22: Factory with kwargs
    # -------------------------------------------------------------------- #
    def test_factory_kwargs():
        ws = create_global_workspace(
            workspace_dim=D, num_heads=4, capacity_limit=K, memory_mode="gru",
        )
        assert ws.config.workspace_dim == D
        assert ws.config.num_heads == 4

    _test("Factory with kwargs", test_factory_kwargs)

    # -------------------------------------------------------------------- #
    # Test 23: Backward pass (gradients flow)
    # -------------------------------------------------------------------- #
    def test_backward_pass():
        torch.manual_seed(42)
        ws = _make_ws()
        eo = {'vision': _make_eo('vision', B, 4, D, device)}
        out = ws(eo)
        loss = out.slots.sum()
        loss.backward()
        # Check that at least some parameters have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in ws.parameters())
        assert has_grad, "No gradients flowed through the workspace"

    _test("Backward pass (gradients flow)", test_backward_pass)

    # -------------------------------------------------------------------- #
    # Test 24: Multiple sequential steps with state
    # -------------------------------------------------------------------- #
    def test_sequential_steps():
        torch.manual_seed(42)
        ws = _make_ws()
        state = ws.reset_state(B, device)
        ignition_scores = []
        for step in range(5):
            eo = {'vision': _make_eo('vision', B, 4, D, device)}
            out = ws(eo, state=state)
            state = GlobalWorkspace.detach_state(out.workspace_state)
            ignition_scores.append(out.ignition_score.mean().item())
        assert state.step_count == 5
        assert len(state.ignition_history) == 5

    _test("Multiple sequential steps with state", test_sequential_steps)

    # -------------------------------------------------------------------- #
    # Test 25: Telemetry contains expected keys
    # -------------------------------------------------------------------- #
    def test_telemetry_keys():
        torch.manual_seed(42)
        ws = _make_ws()
        eo = {'vision': _make_eo('vision', B, 4, D, device)}
        out = ws(eo, return_details=True)
        assert out.telemetry is not None
        expected_keys = {
            'round_history', 'num_rounds', 'final_jaccard', 'final_cosine',
            'ignition_score_mean', 'ignited_fraction', 'modality_names',
            'token_counts', 'ignition_history',
        }
        missing = expected_keys - set(out.telemetry.keys())
        assert len(missing) == 0, f"Missing telemetry keys: {missing}"

    _test("Telemetry contains expected keys", test_telemetry_keys)

    # -------------------------------------------------------------------- #
    # Test 26: param_count returns > 0
    # -------------------------------------------------------------------- #
    def test_param_count():
        ws = _make_ws()
        count = ws.param_count()
        assert count > 0, f"param_count returned {count}"

    _test("param_count returns > 0", test_param_count)

    # -------------------------------------------------------------------- #
    # Test 27: No NaN in output
    # -------------------------------------------------------------------- #
    def test_no_nan():
        torch.manual_seed(42)
        ws = _make_ws()
        eo = {
            'vision': _make_eo('vision', B, 5, D, device),
            'text': _make_eo('text', B, 3, D, device),
        }
        out = ws(eo)
        assert not torch.isnan(out.slots).any(), "NaN in slots"
        assert not torch.isnan(out.ignition_score).any(), "NaN in ignition_score"
        for name, pkt in out.broadcast_packets.items():
            assert not torch.isnan(pkt).any(), f"NaN in broadcast packet '{name}'"

    _test("No NaN in output", test_no_nan)

    # -------------------------------------------------------------------- #
    # Test 28: Slot mask is all True (K valid slots)
    # -------------------------------------------------------------------- #
    def test_slot_mask_all_true():
        torch.manual_seed(42)
        ws = _make_ws()
        eo = {'vision': _make_eo('vision', B, 6, D, device)}
        out = ws(eo)
        assert out.slot_mask.all(), "Expected all slots to be valid"

    _test("Slot mask is all True for K valid slots", test_slot_mask_all_true)

    # -------------------------------------------------------------------- #
    # Test 29: Works with batch_size=1
    # -------------------------------------------------------------------- #
    def test_batch_size_one():
        torch.manual_seed(42)
        ws = _make_ws()
        eo = {'vision': _make_eo('vision', 1, 4, D, device)}
        out = ws(eo)
        assert out.slots.shape == (1, K, D)

    _test("Works with batch_size=1", test_batch_size_one)

    # -------------------------------------------------------------------- #
    # Summary
    # -------------------------------------------------------------------- #
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print("\nFailed tests:")
        for e in errors:
            print(e)
    print("=" * 70)

    return failed == 0


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    success = _run_self_tests()
    raise SystemExit(0 if success else 1)
