"""Competition subsystem for the Global Workspace module.

This module implements the full competition pipeline that determines which
encoder tokens gain access to the limited-capacity global workspace.  It
replaces the single-round ``AttentionCompetition`` with a deterministic,
multi-term scoring pipeline that supports explicit tie-breaking, task-biased
routing, and novelty-based attention modulation.

Pipeline overview::

    EncoderOutputs  -->  TokenStager  -->  CompetitionScorer  -->  DeterministicTopK
                                                                        |
                                          CompetitionResult  <--  SlotConstructor

Typical import (once integrated into brain_ai)::

    from brain_ai.workspace.competition import (
        CompetitionConfig,
        CompetitionModule,
        CompetitionResult,
        TokenTable,
        WinnersMetadata,
    )

Copy this file to ``brain_ai/workspace/competition.py`` when integrating
into the main package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# EncoderOutput -- inline minimal copy for self-containment
# ---------------------------------------------------------------------------
# When this template is copied into the main package, replace this with:
#     from brain_ai.encoders.schema import EncoderOutput


@dataclass
class EncoderOutput:
    """Minimal inline copy of the canonical encoder output contract.

    Defined here so the template is self-contained and runnable without
    importing the full ``brain_ai`` package.

    Shape rules:
        feats    -- (B, T, D)   float, always 3-D even if T=1
        mask     -- (B, T)      bool, True=valid, False=padding
        salience -- (B, T) or (B, 1) float, non-negative competition weight
        time     -- (B, T)      float32, optional real-valued timestamps
    """

    modality: str
    feats: Tensor                                   # (B, T, D)
    mask: Tensor                                    # (B, T) bool
    salience: Optional[Tensor] = None               # (B, T) or (B, 1)
    pos_ids: Optional[Tensor] = None                # (B, T) int64
    time: Optional[Tensor] = None                   # (B, T) float32
    spike: Optional[Tensor] = None                  # (B, T, *)
    aux: Dict[str, Any] = field(default_factory=dict)


# ===================================================================== #
#                        Configuration                                  #
# ===================================================================== #

@dataclass
class CompetitionConfig:
    """All configuration fields for the competition subsystem.

    Attributes:
        capacity_limit:
            Maximum number of tokens (slots) that can win workspace access.
            Inspired by Miller's Law (7 +/- 2).
        num_heads:
            Number of attention heads used internally by scoring networks.
            Not directly used by the deterministic scorer but reserved for
            any attention-based variants.
        w_content:
            Weight for the learned content-relevance score.
        w_salience:
            Weight for the encoder-provided salience score.
        w_novelty:
            Weight for novelty relative to working memory.
        w_task:
            Weight for the optional external task-bias signal.
        eps_tie_modality:
            Epsilon multiplier for modality-based tie-breaking.
            Ensures deterministic ordering among tokens with equal scores.
        eps_tie_token:
            Epsilon multiplier for within-modality token tie-breaking.
        use_slot_mixer:
            Whether to apply a residual MLP to refine winner slots.
        slot_mixer_expansion:
            Expansion ratio for the hidden layer of the slot mixer.
        score_temperature:
            Temperature scaling applied to final scores before top-K.
            Lower values produce sharper selection.
        workspace_dim:
            Dimension of workspace token embeddings.  Must match the
            encoder output dimension (``config.encoder.output_dim``).
    """

    capacity_limit: int = 7
    num_heads: int = 16
    w_content: float = 1.0
    w_salience: float = 0.5
    w_novelty: float = 0.3
    w_task: float = 0.0
    eps_tie_modality: float = 1e-6
    eps_tie_token: float = 1e-8
    use_slot_mixer: bool = True
    slot_mixer_expansion: float = 2.0
    score_temperature: float = 1.0
    workspace_dim: int = 512


# ===================================================================== #
#                     Result / intermediate dataclasses                 #
# ===================================================================== #

@dataclass
class TokenTable:
    """Flat table produced by :class:`TokenStager`.

    All tensors share the same batch dimension ``B`` and concatenated time
    dimension ``T_total``.  Modalities are always concatenated in
    :data:`TokenStager.MODALITY_ORDER`.

    Attributes:
        tokens:
            (B, T_total, D) -- concatenated encoder features.
        mask:
            (B, T_total) bool -- True where the token is valid.
        salience:
            (B, T_total) -- per-token salience scores.
        time:
            (B, T_total) or ``None`` -- optional timestamps.
        modality_ids:
            (T_total,) int64 -- integer id for each token's source modality
            (index into :data:`MODALITY_ORDER`).
        local_ids:
            (T_total,) int64 -- token index within its source modality
            (0-based per modality segment).
        modality_boundaries:
            ``Dict[str, Tuple[int, int]]`` mapping modality name to the
            ``[start, end)`` slice in the concatenated time axis.
    """

    tokens: Tensor                                  # (B, T_total, D)
    mask: Tensor                                    # (B, T_total)
    salience: Tensor                                # (B, T_total)
    time: Optional[Tensor]                          # (B, T_total) or None
    modality_ids: Tensor                            # (T_total,)
    local_ids: Tensor                               # (T_total,)
    modality_boundaries: Dict[str, Tuple[int, int]]


@dataclass
class WinnersMetadata:
    """Metadata about the tokens that won workspace access.

    Each tensor has shape ``(B, K)`` where ``K`` is the effective number
    of winners (may be less than ``capacity_limit`` if fewer valid tokens
    exist).

    Attributes:
        modality_ids:
            (B, K) int64 -- modality id for each winning token.
        local_ids:
            (B, K) int64 -- local token index within its modality.
        scores:
            (B, K) float -- final score of each winning token.
    """

    modality_ids: Tensor                            # (B, K)
    local_ids: Tensor                               # (B, K)
    scores: Tensor                                  # (B, K)


@dataclass
class CompetitionResult:
    """Final output of :class:`CompetitionModule`.

    Attributes:
        slots:
            (B, K, D) -- workspace slot embeddings for winning tokens.
        slot_mask:
            (B, K) bool -- True for real winners, False for padding slots
            (when fewer than K valid tokens were available).
        winners:
            :class:`WinnersMetadata` describing which tokens won.
        scores:
            (B, T_total) -- raw scores for *every* token before selection.
    """

    slots: Tensor                                   # (B, K, D)
    slot_mask: Tensor                               # (B, K) bool
    winners: WinnersMetadata
    scores: Tensor                                  # (B, T_total)


# ===================================================================== #
#                          1. TokenStager                               #
# ===================================================================== #

class TokenStager:
    """Deterministic concatenation of multi-modal encoder outputs.

    Encoder outputs are always concatenated in a fixed order defined by
    :data:`MODALITY_ORDER`.  Missing modalities are silently skipped; the
    resulting :class:`TokenTable` records boundaries so downstream code
    can map winners back to their source modality.

    This is a plain Python class (not ``nn.Module``) because it has no
    learnable parameters.

    Class Constants:
        MODALITY_ORDER:
            The canonical ordering used for concatenation.  Modalities not
            in this list are appended alphabetically after the canonical
            entries.
    """

    MODALITY_ORDER: List[str] = ["vision", "text", "audio", "sensors", "engram"]

    def stage(
        self,
        encoder_outputs: Dict[str, EncoderOutput],
    ) -> TokenTable:
        """Concatenate encoder outputs into a single flat token table.

        Parameters
        ----------
        encoder_outputs:
            Mapping from modality name to its ``EncoderOutput``.  The dict
            need not contain all modalities -- missing ones are skipped.

        Returns
        -------
        TokenTable
            Flat table with all tokens, masks, salience, and bookkeeping
            tensors.

        Raises
        ------
        ValueError
            If ``encoder_outputs`` is empty or contains inconsistent batch
            sizes / embedding dimensions.
        """
        if not encoder_outputs:
            raise ValueError("encoder_outputs must contain at least one modality")

        # Determine ordering: canonical first, then extras alphabetically
        ordered_names: List[str] = []
        for name in self.MODALITY_ORDER:
            if name in encoder_outputs:
                ordered_names.append(name)
        for name in sorted(encoder_outputs.keys()):
            if name not in ordered_names:
                ordered_names.append(name)

        # Collect tensors
        all_feats: List[Tensor] = []
        all_masks: List[Tensor] = []
        all_salience: List[Tensor] = []
        all_time: List[Tensor] = []
        modality_id_parts: List[Tensor] = []
        local_id_parts: List[Tensor] = []
        boundaries: Dict[str, Tuple[int, int]] = {}
        has_any_time: bool = False

        # Build a modality-name -> integer-id mapping
        name_to_id: Dict[str, int] = {}
        for idx, name in enumerate(self.MODALITY_ORDER):
            name_to_id[name] = idx
        next_id = len(self.MODALITY_ORDER)
        for name in ordered_names:
            if name not in name_to_id:
                name_to_id[name] = next_id
                next_id += 1

        offset = 0
        ref_batch: Optional[int] = None
        ref_dim: Optional[int] = None
        ref_device: Optional[torch.device] = None

        for name in ordered_names:
            eo = encoder_outputs[name]
            B, T, D = eo.feats.shape

            # Consistency checks
            if ref_batch is None:
                ref_batch = B
                ref_dim = D
                ref_device = eo.feats.device
            else:
                if B != ref_batch:
                    raise ValueError(
                        f"Batch size mismatch: {name} has B={B}, "
                        f"expected B={ref_batch}"
                    )
                if D != ref_dim:
                    raise ValueError(
                        f"Embedding dim mismatch: {name} has D={D}, "
                        f"expected D={ref_dim}"
                    )

            all_feats.append(eo.feats)
            all_masks.append(eo.mask)

            # Salience: default to 1.0 when not provided
            if eo.salience is not None:
                sal = eo.salience
                # Broadcast (B, 1) -> (B, T)
                if sal.ndim == 2 and sal.shape[1] == 1:
                    sal = sal.expand(B, T)
                all_salience.append(sal)
            else:
                all_salience.append(
                    torch.ones(B, T, device=ref_device, dtype=torch.float32)
                )

            # Time: track whether any modality provides it
            if eo.time is not None:
                has_any_time = True
                all_time.append(eo.time)
            else:
                all_time.append(
                    torch.zeros(B, T, device=ref_device, dtype=torch.float32)
                )

            # Modality / local ids
            mod_id = name_to_id[name]
            modality_id_parts.append(
                torch.full((T,), mod_id, dtype=torch.long, device=ref_device)
            )
            local_id_parts.append(
                torch.arange(T, dtype=torch.long, device=ref_device)
            )

            boundaries[name] = (offset, offset + T)
            offset += T

        tokens = torch.cat(all_feats, dim=1)       # (B, T_total, D)
        mask = torch.cat(all_masks, dim=1)          # (B, T_total)
        salience = torch.cat(all_salience, dim=1)   # (B, T_total)
        time_tensor: Optional[Tensor] = None
        if has_any_time:
            time_tensor = torch.cat(all_time, dim=1)  # (B, T_total)
        modality_ids = torch.cat(modality_id_parts) # (T_total,)
        local_ids = torch.cat(local_id_parts)       # (T_total,)

        return TokenTable(
            tokens=tokens,
            mask=mask,
            salience=salience,
            time=time_tensor,
            modality_ids=modality_ids,
            local_ids=local_ids,
            modality_boundaries=boundaries,
        )


# ===================================================================== #
#                     2. CompetitionScorer                              #
# ===================================================================== #

class CompetitionScorer(nn.Module):
    """Four-term scoring function for workspace competition.

    The final score for each token is a weighted sum of four components:

    1. **Content** -- learned relevance score from a small MLP applied
       to each token embedding.
    2. **Salience** -- encoder-provided importance weight (from the
       ``EncoderOutput.salience`` field).
    3. **Novelty** -- ``1 - cosine_similarity(token, wm_summary)`` where
       ``wm_summary`` is the mean of the working memory buffer.  Tokens
       that are dissimilar to recent memory contents score higher.
    4. **Task bias** -- an optional external signal ``(B, T_total)``
       allowing top-down attention control.

    The four component weights (``w_content``, ``w_salience``, ``w_novelty``,
    ``w_task``) come from :class:`CompetitionConfig` and are **not**
    learned.  This keeps the scoring interpretable and avoids degenerate
    solutions during early training.

    Parameters
    ----------
    config:
        :class:`CompetitionConfig` controlling weights and dimensions.
    """

    def __init__(self, config: CompetitionConfig) -> None:
        super().__init__()
        self.config = config
        D = config.workspace_dim

        # f_content: Linear -> LayerNorm -> GELU -> Linear -> squeeze
        self.f_content = nn.Sequential(
            nn.Linear(D, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Linear(D, 1),
        )

        # Fixed weights (not nn.Parameter -- not learned)
        self.w_content = config.w_content
        self.w_salience = config.w_salience
        self.w_novelty = config.w_novelty
        self.w_task = config.w_task

    def forward(
        self,
        token_table: TokenTable,
        wm_summary: Optional[Tensor] = None,
        task_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute scores for all tokens.

        Parameters
        ----------
        token_table:
            Output of :class:`TokenStager` containing tokens, mask,
            and salience.
        wm_summary:
            (B, D) mean embedding from the working memory buffer.  When
            ``None``, novelty is set to 1.0 for all tokens (maximally
            novel).
        task_bias:
            (B, T_total) external task-priority signal.  When ``None``,
            the task term contributes zero.

        Returns
        -------
        scores:
            (B, T_total) float32 score for each token.  Higher is better.
        """
        tokens = token_table.tokens           # (B, T, D)
        salience = token_table.salience       # (B, T)
        B, T, D = tokens.shape

        # --- Force fp32 for all score computation ---
        # Use autocast(enabled=False) to ensure fp32 even inside AMP regions.
        with torch.amp.autocast("cuda", enabled=False), \
             torch.amp.autocast("cpu", enabled=False):
            tokens_f32 = tokens.float()       # (B, T, D) fp32
            salience_f32 = salience.float()   # (B, T)    fp32

            # 1. Content score
            content = self.f_content(tokens_f32).squeeze(-1)  # (B, T)

            # 2. Salience (already fp32)
            sal = salience_f32

            # 3. Novelty
            if wm_summary is not None:
                wm_f32 = wm_summary.float()   # (B, D)
                # Expand for broadcasting: (B, 1, D)
                wm_expanded = wm_f32.unsqueeze(1).expand_as(tokens_f32)
                cos_sim = F.cosine_similarity(
                    tokens_f32, wm_expanded, dim=-1
                )  # (B, T)
                novelty = 1.0 - cos_sim
            else:
                novelty = torch.ones(
                    B, T, device=tokens.device, dtype=torch.float32
                )

            # 4. Task bias
            if task_bias is not None:
                tb = task_bias.float()
            else:
                tb = torch.zeros(
                    B, T, device=tokens.device, dtype=torch.float32
                )

            # Weighted combination
            score = (
                self.w_content * content
                + self.w_salience * sal
                + self.w_novelty * novelty
                + self.w_task * tb
            )

        return score


# ===================================================================== #
#                      3. DeterministicTopK                             #
# ===================================================================== #

class DeterministicTopK(nn.Module):
    """Top-K selection with explicit, reproducible tie-breaking.

    When two tokens have identical scores (up to floating-point equality),
    a tiny deterministic adjustment is added based on the token's modality
    index and local position index.  This guarantees that the selected set
    is identical across runs with the same inputs, regardless of GPU
    non-determinism in sorting algorithms.

    Tie-breaking formula::

        score_adj = score
                    + eps1 * (-modality_id / max_modality)
                    + eps2 * (-local_id   / max_local)

    The negative signs ensure that, among tied tokens, those from
    earlier modalities (and earlier positions within a modality) are
    preferred.  The eps values are small enough that they never override
    a genuine score difference.

    All computations are performed in fp32.

    Parameters
    ----------
    eps_modality:
        Scale for modality-based tie-breaking.
    eps_token:
        Scale for within-modality tie-breaking.
    """

    def __init__(
        self,
        eps_modality: float = 1e-6,
        eps_token: float = 1e-8,
    ) -> None:
        super().__init__()
        self.eps_modality = eps_modality
        self.eps_token = eps_token

    def forward(
        self,
        scores: Tensor,
        mask: Tensor,
        K: int,
        modality_ids: Tensor,
        local_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Select the top-K tokens with deterministic tie-breaking.

        Parameters
        ----------
        scores:
            (B, T_total) raw scores for every token.
        mask:
            (B, T_total) bool -- True for valid tokens.
        K:
            Number of winners to select.
        modality_ids:
            (T_total,) integer modality index per token position.
        local_ids:
            (T_total,) integer local index per token position.

        Returns
        -------
        winner_indices:
            (B, K) int64 -- indices into the T_total dimension for each
            winning token.  Padded with ``-1`` when fewer than K valid
            tokens exist.
        winner_scores:
            (B, K) float32 -- score for each winning token.  Padded
            entries have score ``-inf``.
        """
        B, T = scores.shape
        device = scores.device

        with torch.amp.autocast("cuda", enabled=False), \
             torch.amp.autocast("cpu", enabled=False):
            scores_f32 = scores.float()

            # Compute tie-breaking adjustment (broadcast over batch)
            max_mod = max(modality_ids.max().item(), 1)
            max_local = max(local_ids.max().item(), 1)

            # (T_total,) tie-break offsets
            tie_break = (
                self.eps_modality * (-modality_ids.float() / max_mod)
                + self.eps_token * (-local_ids.float() / max_local)
            )
            # Expand to (1, T_total) for broadcasting
            tie_break = tie_break.unsqueeze(0)

            # Adjusted scores
            scores_adj = scores_f32 + tie_break

            # Mask invalid tokens with -inf
            scores_adj = scores_adj.masked_fill(~mask, float("-inf"))

            # Count valid tokens per batch element
            valid_counts = mask.sum(dim=-1)  # (B,)

            # Effective K: clamp to T to avoid torch.topk errors
            effective_K = min(K, T)

            # Top-K selection
            topk_scores, topk_indices = torch.topk(
                scores_adj, effective_K, dim=-1, largest=True, sorted=True
            )

            # Handle case where K > valid_tokens for some batch elements:
            # Mark padded winners with index=-1 and score=-inf
            if effective_K > 0:
                # Build a mask for positions beyond valid count
                rank_positions = torch.arange(
                    effective_K, device=device
                ).unsqueeze(0).expand(B, -1)  # (B, K)
                valid_mask = rank_positions < valid_counts.unsqueeze(1)  # (B, K)

                # Pad invalid winners
                topk_indices = topk_indices.where(
                    valid_mask,
                    torch.tensor(-1, dtype=torch.long, device=device),
                )
                topk_scores = topk_scores.where(
                    valid_mask,
                    torch.tensor(
                        float("-inf"), dtype=torch.float32, device=device
                    ),
                )

            # If K > T, pad with extra -1 columns
            if K > effective_K:
                pad_size = K - effective_K
                pad_indices = torch.full(
                    (B, pad_size), -1, dtype=torch.long, device=device
                )
                pad_scores = torch.full(
                    (B, pad_size),
                    float("-inf"),
                    dtype=torch.float32,
                    device=device,
                )
                topk_indices = torch.cat(
                    [topk_indices, pad_indices], dim=-1
                )
                topk_scores = torch.cat(
                    [topk_scores, pad_scores], dim=-1
                )

        return topk_indices, topk_scores


# ===================================================================== #
#                       4. SlotConstructor                              #
# ===================================================================== #

class SlotMixer(nn.Module):
    """Residual MLP for refining winner slot embeddings.

    Architecture: ``LayerNorm -> Linear -> GELU -> Linear`` with a
    residual connection.  The hidden dimension is
    ``int(D * expansion_ratio)``.

    Parameters
    ----------
    dim:
        Input and output embedding dimension.
    expansion_ratio:
        Multiplier for the hidden layer size.
    """

    def __init__(self, dim: int, expansion_ratio: float = 2.0) -> None:
        super().__init__()
        hidden = int(dim * expansion_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

        # Zero-init output projection for stable residual at init
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual MLP.

        Parameters
        ----------
        x:
            (..., D) input tensor of any leading dimensions.

        Returns
        -------
        out:
            (..., D) refined tensor with residual connection.
        """
        residual = x
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return residual + h


class SlotConstructor(nn.Module):
    """Build ``(B, K, D)`` workspace slots from winner indices.

    Gathers the embeddings for winning tokens and optionally refines
    them with a :class:`SlotMixer` residual MLP.

    Parameters
    ----------
    config:
        :class:`CompetitionConfig` controlling mixer settings.
    """

    def __init__(self, config: CompetitionConfig) -> None:
        super().__init__()
        self.config = config
        self.mixer: Optional[SlotMixer] = None
        if config.use_slot_mixer:
            self.mixer = SlotMixer(
                dim=config.workspace_dim,
                expansion_ratio=config.slot_mixer_expansion,
            )

    def forward(
        self,
        tokens: Tensor,
        winner_indices: Tensor,
        token_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Construct slots from winner indices.

        Parameters
        ----------
        tokens:
            (B, T_total, D) all token embeddings.
        winner_indices:
            (B, K) indices into the T_total dimension.  ``-1`` denotes
            a padding slot (no valid winner).
        token_mask:
            (B, T_total) bool mask for valid tokens (used for
            constructing the slot_mask).

        Returns
        -------
        slots:
            (B, K, D) workspace slot embeddings.  Padding slots are
            filled with zeros.
        slot_mask:
            (B, K) bool -- True for real winners, False for padding.
        """
        B, T, D = tokens.shape
        K = winner_indices.shape[1]
        device = tokens.device

        # Build slot mask: True where winner_indices != -1
        slot_mask = winner_indices >= 0  # (B, K)

        # Replace -1 indices with 0 for safe gather, then zero-out later
        safe_indices = winner_indices.clamp(min=0)  # (B, K)

        # Gather: (B, K, D)
        expanded_indices = safe_indices.unsqueeze(-1).expand(
            B, K, D
        )  # (B, K, D)
        slots = torch.gather(
            tokens, dim=1, index=expanded_indices
        )  # (B, K, D)

        # Zero out padding slots
        slots = slots * slot_mask.unsqueeze(-1).float()

        # Optional refinement
        if self.mixer is not None:
            # Only apply mixer to real slots; keep padded slots as zeros
            mixed = self.mixer(slots)
            slots = mixed * slot_mask.unsqueeze(-1).float()

        return slots, slot_mask


# ===================================================================== #
#                     5. CompetitionModule                              #
# ===================================================================== #

class CompetitionModule(nn.Module):
    """Orchestrates the full competition pipeline.

    Wires together :class:`TokenStager`, :class:`CompetitionScorer`,
    :class:`DeterministicTopK`, and :class:`SlotConstructor` into a
    single ``nn.Module`` with a clean ``forward`` interface.

    Parameters
    ----------
    config:
        :class:`CompetitionConfig` with all hyperparameters.

    Example
    -------
    ::

        config = CompetitionConfig(capacity_limit=7, workspace_dim=512)
        comp = CompetitionModule(config)

        enc_outs = {
            "vision": EncoderOutput(
                modality="vision", feats=..., mask=...,
            ),
            "text": EncoderOutput(
                modality="text", feats=..., mask=...,
            ),
        }
        result: CompetitionResult = comp(enc_outs)
        # result.slots  -> (B, 7, 512)
        # result.slot_mask -> (B, 7) bool
    """

    def __init__(self, config: CompetitionConfig) -> None:
        super().__init__()
        self.config = config
        self.stager = TokenStager()
        self.scorer = CompetitionScorer(config)
        self.topk = DeterministicTopK(
            eps_modality=config.eps_tie_modality,
            eps_token=config.eps_tie_token,
        )
        self.slot_constructor = SlotConstructor(config)

        # Register a persistent buffer for the modality order lookup
        # (purely for device tracking, no gradient)
        self.register_buffer(
            "_modality_order_ids",
            torch.arange(
                len(TokenStager.MODALITY_ORDER), dtype=torch.long
            ),
            persistent=False,
        )

    def forward(
        self,
        encoder_outputs: Dict[str, EncoderOutput],
        wm_summary: Optional[Tensor] = None,
        task_bias: Optional[Tensor] = None,
    ) -> CompetitionResult:
        """Run the full competition pipeline.

        Parameters
        ----------
        encoder_outputs:
            Mapping from modality name to :class:`EncoderOutput`.
        wm_summary:
            (B, D) mean of working memory buffer for novelty scoring.
        task_bias:
            (B, T_total) external task-priority signal.

        Returns
        -------
        CompetitionResult
            Containing slots, slot_mask, winners metadata, and raw scores.
        """
        # 1. Stage tokens into flat table
        table = self.stager.stage(encoder_outputs)

        # 2. Score all tokens
        scores = self.scorer(
            table, wm_summary=wm_summary, task_bias=task_bias
        )

        # 3. Apply temperature
        if self.config.score_temperature != 1.0:
            scores = scores / self.config.score_temperature

        # 4. Top-K selection
        K = self.config.capacity_limit
        winner_indices, winner_scores = self.topk(
            scores=scores,
            mask=table.mask,
            K=K,
            modality_ids=table.modality_ids,
            local_ids=table.local_ids,
        )

        # 5. Construct slots
        slots, slot_mask = self.slot_constructor(
            tokens=table.tokens,
            winner_indices=winner_indices,
            token_mask=table.mask,
        )

        # 6. Build winners metadata
        # Gather modality_ids and local_ids for winners
        B = winner_indices.shape[0]
        T_total = table.modality_ids.shape[0]
        mod_ids_expanded = table.modality_ids.unsqueeze(0).expand(
            B, T_total
        )
        loc_ids_expanded = table.local_ids.unsqueeze(0).expand(
            B, T_total
        )

        safe_idx = winner_indices.clamp(min=0)
        winner_mod_ids = torch.gather(
            mod_ids_expanded, dim=1, index=safe_idx
        )
        winner_loc_ids = torch.gather(
            loc_ids_expanded, dim=1, index=safe_idx
        )

        # Zero out metadata for padding winners
        pad_mask = winner_indices < 0
        winner_mod_ids = winner_mod_ids.masked_fill(pad_mask, -1)
        winner_loc_ids = winner_loc_ids.masked_fill(pad_mask, -1)

        winners = WinnersMetadata(
            modality_ids=winner_mod_ids,
            local_ids=winner_loc_ids,
            scores=winner_scores,
        )

        return CompetitionResult(
            slots=slots,
            slot_mask=slot_mask,
            winners=winners,
            scores=scores,
        )


# ===================================================================== #
#                          Factory function                             #
# ===================================================================== #

def create_competition_module(
    capacity_limit: int = 7,
    workspace_dim: int = 512,
    w_content: float = 1.0,
    w_salience: float = 0.5,
    w_novelty: float = 0.3,
    w_task: float = 0.0,
    use_slot_mixer: bool = True,
    slot_mixer_expansion: float = 2.0,
    score_temperature: float = 1.0,
    **kwargs: Any,
) -> CompetitionModule:
    """Create a :class:`CompetitionModule` with the given settings.

    This is the primary entry point for constructing the competition
    subsystem.  All keyword arguments are forwarded to
    :class:`CompetitionConfig`.

    Returns
    -------
    CompetitionModule
        Ready-to-use competition pipeline.
    """
    config = CompetitionConfig(
        capacity_limit=capacity_limit,
        workspace_dim=workspace_dim,
        w_content=w_content,
        w_salience=w_salience,
        w_novelty=w_novelty,
        w_task=w_task,
        use_slot_mixer=use_slot_mixer,
        slot_mixer_expansion=slot_mixer_expansion,
        score_temperature=score_temperature,
        **kwargs,
    )
    return CompetitionModule(config)


# ===================================================================== #
#                         Self-test suite                               #
# ===================================================================== #

def _self_test() -> None:
    """Comprehensive self-test (~30 tests) for the competition subsystem.

    Run with::

        python competition_template.py

    All tests use CPU, tiny dimensions, and deterministic seeds so they
    complete in a few seconds without a GPU.
    """
    import sys

    torch.manual_seed(42)
    device = torch.device("cpu")
    passed = 0
    failed = 0
    errors: List[str] = []

    def _check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            msg = f"  [FAIL] {name}"
            if detail:
                msg += f" -- {detail}"
            print(msg)
            errors.append(msg)

    def _make_encoder_output(
        modality: str,
        B: int = 2,
        T: int = 5,
        D: int = 32,
        with_salience: bool = True,
        salience_val: float = 1.0,
        with_time: bool = False,
    ) -> EncoderOutput:
        """Helper to build a synthetic EncoderOutput."""
        feats = torch.randn(B, T, D)
        mask = torch.ones(B, T, dtype=torch.bool)
        sal = (
            torch.full((B, T), salience_val) if with_salience else None
        )
        time = (
            torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
            if with_time
            else None
        )
        return EncoderOutput(
            modality=modality,
            feats=feats,
            mask=mask,
            salience=sal,
            time=time,
        )

    B, D = 2, 32
    K = 3
    config = CompetitionConfig(
        capacity_limit=K,
        workspace_dim=D,
        w_content=1.0,
        w_salience=0.5,
        w_novelty=0.3,
        w_task=0.0,
        use_slot_mixer=True,
        slot_mixer_expansion=2.0,
        score_temperature=1.0,
    )

    print("=" * 70)
    print("CompetitionModule self-test")
    print("=" * 70)

    # ------------------------------------------------------------------
    # TokenStager tests
    # ------------------------------------------------------------------
    print("\n--- TokenStager ---")

    stager = TokenStager()

    # Test 1: Single modality staging
    eo_vis = _make_encoder_output("vision", B=B, T=4, D=D)
    table = stager.stage({"vision": eo_vis})
    _check(
        "T01 single modality shape",
        table.tokens.shape == (B, 4, D),
        f"got {table.tokens.shape}",
    )
    _check(
        "T01b single modality mask shape",
        table.mask.shape == (B, 4),
        f"got {table.mask.shape}",
    )

    # Test 2: Multiple modalities in MODALITY_ORDER
    eo_text = _make_encoder_output("text", B=B, T=3, D=D)
    eo_audio = _make_encoder_output("audio", B=B, T=6, D=D)
    table_multi = stager.stage({
        "audio": eo_audio,
        "vision": eo_vis,
        "text": eo_text,
    })
    _check(
        "T02 multi modality concat length",
        table_multi.tokens.shape == (B, 4 + 3 + 6, D),
        f"got {table_multi.tokens.shape}",
    )

    # Test 3: Fixed ordering (vision before text before audio)
    boundaries = table_multi.modality_boundaries
    _check(
        "T03a fixed order: vision first",
        boundaries["vision"] == (0, 4),
        f"got {boundaries['vision']}",
    )
    _check(
        "T03b fixed order: text second",
        boundaries["text"] == (4, 7),
        f"got {boundaries['text']}",
    )
    _check(
        "T03c fixed order: audio third",
        boundaries["audio"] == (7, 13),
        f"got {boundaries['audio']}",
    )

    # Test 4: Missing modalities are skipped
    table_partial = stager.stage({"text": eo_text, "audio": eo_audio})
    _check(
        "T04a missing modalities skipped",
        "vision" not in table_partial.modality_boundaries,
        f"keys = {list(table_partial.modality_boundaries.keys())}",
    )
    _check(
        "T04b missing mod total tokens",
        table_partial.tokens.shape[1] == 3 + 6,
        f"got T={table_partial.tokens.shape[1]}",
    )

    # Test 5: Default salience when not provided
    eo_no_sal = _make_encoder_output(
        "sensors", B=B, T=2, D=D, with_salience=False
    )
    table_sal = stager.stage({"sensors": eo_no_sal})
    _check(
        "T05 default salience = 1.0",
        torch.allclose(table_sal.salience, torch.ones(B, 2)),
        f"got salience = {table_sal.salience}",
    )

    # Test 6: Salience broadcast from (B, 1) to (B, T)
    eo_sal_b1 = EncoderOutput(
        modality="vision",
        feats=torch.randn(B, 5, D),
        mask=torch.ones(B, 5, dtype=torch.bool),
        salience=torch.full((B, 1), 0.7),
    )
    table_b1 = stager.stage({"vision": eo_sal_b1})
    _check(
        "T06 salience broadcast (B,1) -> (B,T)",
        table_b1.salience.shape == (B, 5)
        and torch.allclose(
            table_b1.salience, torch.full((B, 5), 0.7)
        ),
        f"shape={table_b1.salience.shape}, vals={table_b1.salience[0]}",
    )

    # Test 7: Modality IDs correctness
    _check(
        "T07a modality_ids dtype",
        table_multi.modality_ids.dtype == torch.long,
        f"got {table_multi.modality_ids.dtype}",
    )
    # Vision = 0, Text = 1, Audio = 2 (by MODALITY_ORDER)
    vis_ids = table_multi.modality_ids[:4]
    _check(
        "T07b modality_ids vision=0",
        (vis_ids == 0).all().item(),
        f"got {vis_ids.tolist()}",
    )
    text_ids = table_multi.modality_ids[4:7]
    _check(
        "T07c modality_ids text=1",
        (text_ids == 1).all().item(),
        f"got {text_ids.tolist()}",
    )

    # Test 8: Local IDs reset per modality
    audio_local_ids = table_multi.local_ids[7:13]
    _check(
        "T08 local_ids reset per modality",
        audio_local_ids.tolist() == [0, 1, 2, 3, 4, 5],
        f"got {audio_local_ids.tolist()}",
    )

    # Test 9: Time field -- provided vs. None
    eo_with_time = _make_encoder_output(
        "vision", B=B, T=3, D=D, with_time=True
    )
    table_time = stager.stage({"vision": eo_with_time})
    _check(
        "T09a time present when provided",
        table_time.time is not None
        and table_time.time.shape == (B, 3),
        f"time={table_time.time is not None}",
    )
    table_no_time = stager.stage({"vision": eo_vis})
    _check(
        "T09b time is None when no modality provides it",
        table_no_time.time is None,
        f"time={table_no_time.time}",
    )

    # Test 10: Empty encoder_outputs raises ValueError
    try:
        stager.stage({})
        _check(
            "T10 empty encoder_outputs raises", False,
            "no exception raised",
        )
    except ValueError:
        _check("T10 empty encoder_outputs raises", True)

    # ------------------------------------------------------------------
    # CompetitionScorer tests
    # ------------------------------------------------------------------
    print("\n--- CompetitionScorer ---")

    scorer = CompetitionScorer(config)

    # Test 11: Output shape
    table_score = stager.stage({"vision": eo_vis, "text": eo_text})
    scores = scorer(table_score)
    T_total = table_score.tokens.shape[1]
    _check(
        "T11 scorer output shape",
        scores.shape == (B, T_total),
        f"got {scores.shape}",
    )

    # Test 12: Scores are fp32
    _check(
        "T12 scorer output dtype fp32",
        scores.dtype == torch.float32,
        f"got {scores.dtype}",
    )

    # Test 13: No NaN in scores
    _check(
        "T13 scorer no NaN",
        not torch.isnan(scores).any().item(),
        "found NaN in scores",
    )

    # Test 14: Novelty with wm_summary
    wm_summary = torch.randn(B, D)
    scores_with_wm = scorer(table_score, wm_summary=wm_summary)
    _check(
        "T14a scorer with wm_summary shape",
        scores_with_wm.shape == (B, T_total),
        f"got {scores_with_wm.shape}",
    )
    # Scores should differ when novelty is considered
    _check(
        "T14b scores differ with novelty",
        not torch.allclose(scores, scores_with_wm, atol=1e-6),
        "scores identical with and without wm_summary",
    )

    # Test 15: Task bias
    task_bias_t = torch.randn(B, T_total)
    config_task = CompetitionConfig(
        capacity_limit=K,
        workspace_dim=D,
        w_task=1.0,
        w_content=0.0,
        w_salience=0.0,
        w_novelty=0.0,
    )
    scorer_task = CompetitionScorer(config_task)
    scores_task = scorer_task(table_score, task_bias=task_bias_t)
    _check(
        "T15 task-only scoring shape",
        scores_task.shape == (B, T_total),
        f"got {scores_task.shape}",
    )

    # ------------------------------------------------------------------
    # DeterministicTopK tests
    # ------------------------------------------------------------------
    print("\n--- DeterministicTopK ---")

    topk = DeterministicTopK(eps_modality=1e-6, eps_token=1e-8)

    # Test 16: Basic top-K
    scores_basic = torch.tensor(
        [[5.0, 3.0, 1.0, 4.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0]]
    )
    mask_basic = torch.ones(2, 5, dtype=torch.bool)
    mod_ids_basic = torch.tensor([0, 0, 1, 1, 2])
    loc_ids_basic = torch.tensor([0, 1, 0, 1, 0])
    win_idx, win_scores = topk(
        scores_basic, mask_basic, 3, mod_ids_basic, loc_ids_basic
    )
    _check(
        "T16a topk basic shape",
        win_idx.shape == (2, 3) and win_scores.shape == (2, 3),
        f"idx={win_idx.shape}, scores={win_scores.shape}",
    )
    # Batch 0: top-3 should be indices 0 (5.0), 3 (4.0), 1 (3.0)
    _check(
        "T16b topk batch 0 correct",
        set(win_idx[0].tolist()) == {0, 3, 1},
        f"got {win_idx[0].tolist()}",
    )

    # Test 17: Exact ties with deterministic breaking
    scores_tied = torch.tensor([[3.0, 3.0, 3.0, 3.0]])
    mask_tied = torch.ones(1, 4, dtype=torch.bool)
    mod_ids_tied = torch.tensor([0, 1, 2, 3])
    loc_ids_tied = torch.tensor([0, 0, 0, 0])
    win_tied, _ = topk(
        scores_tied, mask_tied, 2, mod_ids_tied, loc_ids_tied
    )
    # Earlier modality should win (mod 0, then mod 1)
    _check(
        "T17 tie-breaking prefers earlier modality",
        win_tied[0, 0].item() == 0 and win_tied[0, 1].item() == 1,
        f"got {win_tied[0].tolist()}",
    )

    # Test 18: Same modality tie-breaking by local_id
    scores_same_mod = torch.tensor([[3.0, 3.0, 3.0]])
    mask_same_mod = torch.ones(1, 3, dtype=torch.bool)
    mod_ids_same = torch.tensor([0, 0, 0])
    loc_ids_same = torch.tensor([0, 1, 2])
    win_same, _ = topk(
        scores_same_mod, mask_same_mod, 2, mod_ids_same, loc_ids_same
    )
    _check(
        "T18 tie-breaking same mod prefers earlier token",
        win_same[0, 0].item() == 0 and win_same[0, 1].item() == 1,
        f"got {win_same[0].tolist()}",
    )

    # Test 19: Masked tokens excluded
    scores_masked = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    mask_masked = torch.tensor(
        [[True, False, True, True, False]]
    )
    mod_ids_m = torch.tensor([0, 0, 1, 1, 2])
    loc_ids_m = torch.tensor([0, 1, 0, 1, 0])
    win_m, sc_m = topk(
        scores_masked, mask_masked, 2, mod_ids_m, loc_ids_m
    )
    # Valid tokens: 0 (5.0), 2 (3.0), 3 (2.0) -> top 2: [0, 2]
    _check(
        "T19 masked tokens excluded",
        set(win_m[0].tolist()) == {0, 2},
        f"got {win_m[0].tolist()}",
    )

    # Test 20: K > valid_tokens pads with -1
    scores_few = torch.tensor([[5.0, 3.0]])
    mask_few = torch.tensor([[True, True]])
    mod_ids_few = torch.tensor([0, 1])
    loc_ids_few = torch.tensor([0, 0])
    win_few, sc_few = topk(
        scores_few, mask_few, 5, mod_ids_few, loc_ids_few
    )
    _check(
        "T20a K > valid pads with -1",
        win_few.shape == (1, 5),
        f"shape={win_few.shape}",
    )
    _check(
        "T20b first 2 real, rest -1",
        (win_few[0, :2] >= 0).all().item()
        and (win_few[0, 2:] == -1).all().item(),
        f"got {win_few[0].tolist()}",
    )
    _check(
        "T20c padded scores are -inf",
        all(
            sc_few[0, i].item() == float("-inf") for i in range(2, 5)
        ),
        f"got {sc_few[0].tolist()}",
    )

    # Test 21: fp32 enforcement in topk
    scores_half = torch.tensor(
        [[1.0, 2.0, 3.0]], dtype=torch.float16
    )
    mask_half = torch.ones(1, 3, dtype=torch.bool)
    mod_half = torch.tensor([0, 0, 0])
    loc_half = torch.tensor([0, 1, 2])
    _, sc_half = topk(
        scores_half, mask_half, 2, mod_half, loc_half
    )
    _check(
        "T21 topk output fp32 even with fp16 input",
        sc_half.dtype == torch.float32,
        f"got {sc_half.dtype}",
    )

    # ------------------------------------------------------------------
    # SlotConstructor tests
    # ------------------------------------------------------------------
    print("\n--- SlotConstructor ---")

    slot_ctor = SlotConstructor(config)

    # Test 22: Slot shape
    tokens_test = torch.randn(B, 10, D)
    indices_test = torch.tensor([[0, 3, 7], [1, 4, 9]])
    mask_test = torch.ones(B, 10, dtype=torch.bool)
    slots, slot_mask = slot_ctor(tokens_test, indices_test, mask_test)
    _check(
        "T22a slot shape",
        slots.shape == (B, 3, D),
        f"got {slots.shape}",
    )
    _check(
        "T22b slot mask shape",
        slot_mask.shape == (B, 3),
        f"got {slot_mask.shape}",
    )
    _check(
        "T22c slot mask all True for valid indices",
        slot_mask.all().item(),
        f"got {slot_mask}",
    )

    # Test 23: Padded indices produce zeros
    indices_padded = torch.tensor([[0, 2, -1], [1, -1, -1]])
    slots_p, slot_mask_p = slot_ctor(
        tokens_test, indices_padded, mask_test
    )
    _check(
        "T23a padded slots are zero",
        (slots_p[0, 2] == 0).all().item()
        and (slots_p[1, 1] == 0).all().item(),
        f"norm[0,2]={slots_p[0, 2].norm().item():.6f}",
    )
    _check(
        "T23b padded slot_mask is False",
        not slot_mask_p[0, 2].item()
        and not slot_mask_p[1, 1].item(),
        f"mask = {slot_mask_p}",
    )

    # Test 24: SlotMixer residual property (at init, output ~ input)
    config_no_mixer = CompetitionConfig(
        capacity_limit=K,
        workspace_dim=D,
        use_slot_mixer=False,
    )
    slot_ctor_no_mixer = SlotConstructor(config_no_mixer)
    slots_no_mix, _ = slot_ctor_no_mixer(
        tokens_test, indices_test, mask_test
    )
    # With zero-init mixer, slots should be close to non-mixed
    _check(
        "T24 slot mixer residual (zero-init ~ identity)",
        torch.allclose(slots, slots_no_mix, atol=1e-4),
        f"max diff = {(slots - slots_no_mix).abs().max().item():.6f}",
    )

    # ------------------------------------------------------------------
    # Full CompetitionModule tests
    # ------------------------------------------------------------------
    print("\n--- CompetitionModule ---")

    comp = CompetitionModule(config)

    # Test 25: Full pipeline end-to-end
    enc_outs = {
        "vision": _make_encoder_output("vision", B=B, T=5, D=D),
        "text": _make_encoder_output("text", B=B, T=4, D=D),
    }
    result = comp(enc_outs)
    _check(
        "T25a CompetitionResult slots shape",
        result.slots.shape == (B, K, D),
        f"got {result.slots.shape}",
    )
    _check(
        "T25b CompetitionResult slot_mask shape",
        result.slot_mask.shape == (B, K),
        f"got {result.slot_mask.shape}",
    )
    _check(
        "T25c CompetitionResult scores shape",
        result.scores.shape == (B, 5 + 4),
        f"got {result.scores.shape}",
    )

    # Test 26: WinnersMetadata populated
    _check(
        "T26a winners modality_ids shape",
        result.winners.modality_ids.shape == (B, K),
        f"got {result.winners.modality_ids.shape}",
    )
    _check(
        "T26b winners local_ids shape",
        result.winners.local_ids.shape == (B, K),
        f"got {result.winners.local_ids.shape}",
    )
    _check(
        "T26c winners scores shape",
        result.winners.scores.shape == (B, K),
        f"got {result.winners.scores.shape}",
    )

    # Test 27: Pipeline with wm_summary
    wm = torch.randn(B, D)
    result_wm = comp(enc_outs, wm_summary=wm)
    _check(
        "T27 pipeline with wm_summary valid",
        result_wm.slots.shape == (B, K, D)
        and not torch.isnan(result_wm.slots).any(),
        "NaN or shape mismatch",
    )

    # Test 28: Pipeline with task_bias
    T_total_full = 5 + 4
    tb = torch.randn(B, T_total_full)
    result_tb = comp(enc_outs, task_bias=tb)
    _check(
        "T28 pipeline with task_bias valid",
        result_tb.slots.shape == (B, K, D),
        f"got {result_tb.slots.shape}",
    )

    # Test 29: All five modalities
    enc_all = {
        "vision": _make_encoder_output("vision", B=B, T=2, D=D),
        "text": _make_encoder_output("text", B=B, T=3, D=D),
        "audio": _make_encoder_output("audio", B=B, T=4, D=D),
        "sensors": _make_encoder_output("sensors", B=B, T=1, D=D),
        "engram": _make_encoder_output("engram", B=B, T=2, D=D),
    }
    result_all = comp(enc_all)
    T_expected = 2 + 3 + 4 + 1 + 2
    _check(
        "T29a all 5 modalities: scores shape",
        result_all.scores.shape == (B, T_expected),
        f"got {result_all.scores.shape}",
    )
    _check(
        "T29b all 5 modalities: slots valid",
        result_all.slots.shape == (B, K, D)
        and result_all.slot_mask.any().item(),
        f"slots shape = {result_all.slots.shape}",
    )

    # Test 30: score_temperature != 1.0
    config_temp = CompetitionConfig(
        capacity_limit=K,
        workspace_dim=D,
        score_temperature=0.5,
    )
    comp_temp = CompetitionModule(config_temp)
    result_temp = comp_temp(enc_outs)
    _check(
        "T30 temperature changes scores",
        not torch.allclose(
            result_temp.scores, result.scores, atol=1e-6
        ),
        "scores identical despite different temperature",
    )

    # Test 31: Gradient flows through competition module
    comp_grad = CompetitionModule(config)
    enc_grad = {
        "vision": EncoderOutput(
            modality="vision",
            feats=torch.randn(B, 3, D, requires_grad=True),
            mask=torch.ones(B, 3, dtype=torch.bool),
            salience=torch.ones(B, 3),
        ),
    }
    result_grad = comp_grad(enc_grad)
    loss = result_grad.slots.sum()
    loss.backward()
    _check(
        "T31 gradient flows to input feats",
        enc_grad["vision"].feats.grad is not None
        and enc_grad["vision"].feats.grad.abs().sum().item() > 0,
        "no gradient on input feats",
    )

    # Test 32: Determinism (same input -> same output)
    torch.manual_seed(123)
    comp_det = CompetitionModule(config)
    comp_det.eval()
    enc_det = {
        "vision": _make_encoder_output("vision", B=B, T=4, D=D),
        "text": _make_encoder_output("text", B=B, T=3, D=D),
    }
    with torch.no_grad():
        r1 = comp_det(enc_det)
        r2 = comp_det(enc_det)
    _check(
        "T32 deterministic (same input -> same scores)",
        torch.allclose(r1.scores, r2.scores, atol=1e-7),
        f"max diff = {(r1.scores - r2.scores).abs().max():.8f}",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for e in errors:
            print(f"  {e}")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    _self_test()
