"""
brain_ai/workspace/ignition.py — Ignition Dynamics for the Global Workspace

This module provides the ignition subsystem for the Global Workspace Theory (GWT)
implementation. Ignition is the critical phase transition in GWT where competing
specialist modules suddenly achieve global broadcast — a winner-take-most event
analogous to the neural ignition observed in prefrontal cortex during conscious
access.

The existing IterativeCompetition in global_workspace.py uses a monolithic MLP
ignition_detector that maps mean workspace features to a single scalar. This works
for simple cases but lacks:

1. **Interpretability** — No breakdown of WHY ignition occurred.
2. **Structured components** — Stability, confidence, margin, and coherence are
   conflated into a single opaque scalar.
3. **Ignition gating** — No smooth gain control; only early stopping on a hard
   threshold.
4. **Lock-in prevention** — No mechanism to prevent the same coalition of
   specialists from dominating workspace access indefinitely.

This module replaces the monolithic detector with a structured, interpretable
ignition pipeline:

    Iterative competition rounds
           |
           v
    StabilityComputer   ConfidenceComputer   CoherenceComputer
           \\                   |                    /
            \\                  |                   /
             v                 v                  v
                     IgnitionScorer
                          |
                          v
                     IgnitionGate
                          |
                          v
                   LockInPrevention
                          |
                          v
                    IgnitionResult

Key classes:
    IgnitionConfig          — All hyperparameters for the ignition subsystem.
    IgnitionComponents      — Dataclass holding the 4 interpretable component scores.
    StabilityComputer       — Winner-set and embedding convergence metrics.
    ConfidenceComputer      — Softmax margin and mean score gap metrics.
    CoherenceComputer       — Cross-modal diversity and mutual consistency metrics.
    IgnitionScorer          — Combines components into final ignition score (nn.Module).
    IgnitionGate            — Applies smooth/hard gating to slots (nn.Module).
    LockInPrevention        — Novelty scoring, winner decay, slot dropout (nn.Module).
    IgnitionModule          — Orchestrates all ignition logic (nn.Module).
    IgnitionResult          — Dataclass holding all outputs of IgnitionModule.

Integration with existing code:
    # In IterativeCompetition.forward(), replace:
    #     ignition = self.ignition_detector(current_features.mean(dim=1))
    # with:
    #     result = self.ignition_module(
    #         slots=current_features,
    #         all_scores=saliences,
    #         winners_metadata=metadata,
    #         round_history=history,
    #         prev_state=prev_state,
    #     )
    #     if result.ignited.any():
    #         break  # ignition occurred, stop iterating

Design principles:
    - All computation in fp32 to avoid half-precision artefacts in thresholds.
    - No in-place operations on gradient-carrying tensors.
    - Interpretable mode is the DEFAULT; learned mode is opt-in.
    - Training uses smooth sigmoid gating; inference uses hard threshold for latency.
    - Every component score is bounded to [0, 1] for stable weighted combination.

References:
    Dehaene et al. (2003) "A neuronal model of a global workspace in effortful
        cognitive tasks." PNAS.
    Baars (1988) "A Cognitive Theory of Consciousness." Cambridge UP.
    Mashour et al. (2020) "Conscious Processing and the Global Neuronal Workspace
        Hypothesis." Neuron.
    Doerig et al. (2021) "Hard criteria for empirical theories of consciousness."
        Cognitive Neuroscience.
    VanRullen & Kanai (2021) "Deep learning and the Global Workspace Theory."
        Trends in Neurosciences.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Numerical stability
_EPS: float = 1e-8

# Clamping bounds for component scores
_SCORE_MIN: float = 0.0
_SCORE_MAX: float = 1.0

# Default workspace dimension (must match SelectionBroadcastConfig)
_DEFAULT_WORKSPACE_DIM: int = 1024


# ===========================================================================
# SECTION 1: IgnitionConfig
# ===========================================================================

@dataclass
class IgnitionConfig:
    """Configuration for the ignition dynamics subsystem.

    This dataclass aggregates all hyperparameters controlling ignition
    detection, gating, and lock-in prevention. Defaults produce a
    balanced interpretable scorer where all four component weights sum
    to 1.0 (0.3 + 0.3 + 0.2 + 0.2).

    Attributes:
        ignition_threshold: Score above which ignition is declared.
            Lower values make ignition easier to trigger. Range: (0, 1).
        w_stability: Weight for the stability component in interpretable mode.
        w_confidence: Weight for the confidence component in interpretable mode.
        w_margin: Weight for the margin component in interpretable mode.
        w_coherence: Weight for the coherence component in interpretable mode.
        weak_gain: Minimum gain applied when ignition has NOT occurred.
            Prevents complete suppression of workspace content. Range: [0, 1).
        gate_temperature: Temperature for the sigmoid gating function.
            Lower values produce a sharper transition. Must be > 0.
        use_learned_ignition: If True, use a learned MLP scorer instead of
            the interpretable weighted sum.
        learned_ignition_hidden: Hidden dimension for the learned MLP scorer.
        slot_dropout: Probability of dropping a winning slot during training.
            Used by LockInPrevention to encourage diversity.
        winner_decay: Decay factor applied to previous winners' scores.
            Higher values penalise recently-winning slots more strongly.
        cooldown_steps: Number of steps after ignition during which the
            effective threshold is boosted (0 = no cooldown).
        cooldown_boost: Additive boost to the threshold during cooldown.
            effective_threshold = ignition_threshold + cooldown_boost.
    """

    ignition_threshold: float = 0.3
    w_stability: float = 0.3
    w_confidence: float = 0.3
    w_margin: float = 0.2
    w_coherence: float = 0.2
    weak_gain: float = 0.3
    gate_temperature: float = 0.1
    use_learned_ignition: bool = False
    learned_ignition_hidden: int = 128
    slot_dropout: float = 0.1
    winner_decay: float = 0.1
    cooldown_steps: int = 0
    cooldown_boost: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not (0.0 < self.ignition_threshold < 1.0):
            raise ValueError(
                f"ignition_threshold must be in (0, 1), got {self.ignition_threshold}"
            )
        if self.gate_temperature <= 0.0:
            raise ValueError(
                f"gate_temperature must be > 0, got {self.gate_temperature}"
            )
        if not (0.0 <= self.weak_gain < 1.0):
            raise ValueError(
                f"weak_gain must be in [0, 1), got {self.weak_gain}"
            )
        if not (0.0 <= self.slot_dropout < 1.0):
            raise ValueError(
                f"slot_dropout must be in [0, 1), got {self.slot_dropout}"
            )
        if not (0.0 <= self.winner_decay <= 1.0):
            raise ValueError(
                f"winner_decay must be in [0, 1], got {self.winner_decay}"
            )
        if self.cooldown_steps < 0:
            raise ValueError(
                f"cooldown_steps must be >= 0, got {self.cooldown_steps}"
            )
        if self.cooldown_boost < 0.0:
            raise ValueError(
                f"cooldown_boost must be >= 0, got {self.cooldown_boost}"
            )
        # Weights must be non-negative
        for w_name in ("w_stability", "w_confidence", "w_margin", "w_coherence"):
            w_val = getattr(self, w_name)
            if w_val < 0.0:
                raise ValueError(f"{w_name} must be >= 0, got {w_val}")


# ===========================================================================
# SECTION 2: IgnitionComponents dataclass
# ===========================================================================

@dataclass
class IgnitionComponents:
    """Interpretable component scores for ignition detection.

    Each field is a (B,) tensor of float values in [0, 1], where B is the
    batch size. These components are combined by IgnitionScorer to produce
    the final ignition score.

    Attributes:
        stability: Convergence of the winner set across competition rounds.
            High stability (near 1.0) means the same slots keep winning,
            indicating the competition has settled.
        confidence: Softmax margin between the K-th and (K+1)-th ranked
            scores, measuring how decisively the top-K winners are selected.
        margin: Mean gap between winner scores and non-winner scores.
            Large margin indicates clear separation of winners from losers.
        coherence: Cross-modal coherence of the winning slot coalition.
            High coherence means winners span multiple modalities AND
            are mutually consistent in embedding space.
    """

    stability: Tensor    # (B,) float in [0, 1]
    confidence: Tensor   # (B,) float in [0, 1]
    margin: Tensor       # (B,) float in [0, 1]
    coherence: Tensor    # (B,) float in [0, 1]

    def to_tensor(self) -> Tensor:
        """Stack all components into a (B, 4) tensor for learned scoring."""
        return torch.stack(
            [self.stability, self.confidence, self.margin, self.coherence],
            dim=-1,
        )

    def detach(self) -> "IgnitionComponents":
        """Return a detached copy of all components."""
        return IgnitionComponents(
            stability=self.stability.detach(),
            confidence=self.confidence.detach(),
            margin=self.margin.detach(),
            coherence=self.coherence.detach(),
        )


# ===========================================================================
# SECTION 3: IgnitionResult dataclass
# ===========================================================================

@dataclass
class IgnitionResult:
    """Complete output of the IgnitionModule.

    Attributes:
        ignition_score: (B,) float in [0, 1] — the raw ignition score
            before thresholding.
        ignited: (B,) bool — whether ignition occurred for each sample.
        effective_gain: (B,) float — the gain applied to slots. In training
            this is a smooth sigmoid value; at test time it is 0 or 1.
        gated_slots: (B, K, D) float — slots after gain modulation.
        components: The interpretable component scores.
        cooldown_active: (B,) bool — whether the cooldown period is active.
    """

    ignition_score: Tensor          # (B,)
    ignited: Tensor                 # (B,) bool
    effective_gain: Tensor          # (B,)
    gated_slots: Tensor             # (B, K, D)
    components: IgnitionComponents
    cooldown_active: Tensor         # (B,) bool


# ===========================================================================
# SECTION 4: StabilityComputer
# ===========================================================================

class StabilityComputer:
    """Computes stability metrics for the competition winner set.

    Stability measures how settled the competition is. If the same slots
    keep winning across rounds, the competition has converged and ignition
    is more likely.

    Two complementary metrics are combined:
    1. **Winner stability** — Jaccard similarity between the current and
       previous winner sets.
    2. **Embedding stability** — Mean cosine similarity between current
       and previous slot embeddings across all slots.

    These are combined with equal weight by default.
    """

    @staticmethod
    def compute_winner_stability(
        winners_current: Tensor,
        winners_previous: Tensor,
    ) -> Tensor:
        """Compute Jaccard similarity between winner sets.

        Args:
            winners_current: (B, N) binary mask of current round winners.
            winners_previous: (B, N) binary mask of previous round winners.

        Returns:
            (B,) Jaccard similarity in [0, 1]. Returns 0 if both sets are empty.
        """
        # Ensure fp32
        wc = winners_current.float()
        wp = winners_previous.float()

        intersection = (wc * wp).sum(dim=-1)           # (B,)
        union = ((wc + wp).clamp(max=1.0)).sum(dim=-1)  # (B,)

        jaccard = intersection / (union + _EPS)
        return jaccard.clamp(_SCORE_MIN, _SCORE_MAX)

    @staticmethod
    def compute_embedding_stability(
        slots_current: Tensor,
        slots_previous: Tensor,
    ) -> Tensor:
        """Compute mean cosine similarity between slot embeddings.

        Args:
            slots_current: (B, K, D) current slot embeddings.
            slots_previous: (B, K, D) previous round slot embeddings.

        Returns:
            (B,) mean cosine similarity in [0, 1] (clamped from [-1, 1]).
        """
        # Normalise along embedding dimension
        sc_norm = F.normalize(slots_current.float(), dim=-1)
        sp_norm = F.normalize(slots_previous.float(), dim=-1)

        # Per-slot cosine similarity: (B, K)
        cos_sim = (sc_norm * sp_norm).sum(dim=-1)

        # Mean over slots, shift from [-1, 1] to [0, 1]
        mean_cos = cos_sim.mean(dim=-1)
        stability = (mean_cos + 1.0) / 2.0

        return stability.clamp(_SCORE_MIN, _SCORE_MAX)

    @staticmethod
    def combine(
        winner_stab: Tensor,
        embed_stab: Tensor,
        alpha: float = 0.5,
    ) -> Tensor:
        """Combine winner and embedding stability.

        Args:
            winner_stab: (B,) Jaccard similarity.
            embed_stab: (B,) embedding cosine stability.
            alpha: Weight for winner_stab; (1-alpha) for embed_stab.

        Returns:
            (B,) overall stability in [0, 1].
        """
        combined = alpha * winner_stab + (1.0 - alpha) * embed_stab
        return combined.clamp(_SCORE_MIN, _SCORE_MAX)


# ===========================================================================
# SECTION 5: ConfidenceComputer
# ===========================================================================

class ConfidenceComputer:
    """Computes confidence metrics for the top-K selection.

    Confidence measures how decisive the winner selection is. If the top-K
    scores are well-separated from the rest, ignition should be easier.

    Two metrics:
    1. **Margin** — Softmax margin between the K-th and (K+1)-th ranked
       scores (i.e., the gap at the selection boundary).
    2. **Mean gap** — Average score difference between winners and losers.
    """

    @staticmethod
    def compute_margin(
        all_scores: Tensor,
        K: int,
    ) -> Tensor:
        """Compute softmax margin between K-th and (K+1)-th scores.

        Args:
            all_scores: (B, N) raw scores for all N candidate slots.
            K: Number of winners to select.

        Returns:
            (B,) softmax margin in [0, 1]. If N <= K, returns ones.
        """
        B, N = all_scores.shape

        if N <= K:
            return torch.ones(B, device=all_scores.device, dtype=torch.float32)

        # Softmax over all candidates
        probs = F.softmax(all_scores.float(), dim=-1)  # (B, N)

        # Sort descending
        sorted_probs, _ = probs.sort(dim=-1, descending=True)

        # K-th winner (index K-1) vs (K+1)-th loser (index K)
        winner_boundary = sorted_probs[:, K - 1]   # (B,)
        loser_boundary = sorted_probs[:, K]         # (B,)

        margin = (winner_boundary - loser_boundary).clamp(_SCORE_MIN, _SCORE_MAX)
        return margin

    @staticmethod
    def compute_mean_gap(
        winner_scores: Tensor,
        loser_scores: Tensor,
    ) -> Tensor:
        """Compute mean score gap between winners and losers.

        Args:
            winner_scores: (B, K) scores of winning slots.
            loser_scores: (B, M) scores of losing slots.

        Returns:
            (B,) mean gap, sigmoid-normalised to [0, 1].
        """
        # Mean winner score minus mean loser score
        w_mean = winner_scores.float().mean(dim=-1)  # (B,)
        l_mean = loser_scores.float().mean(dim=-1)   # (B,)

        raw_gap = w_mean - l_mean  # (B,)

        # Sigmoid normalisation to [0, 1]
        normalised = torch.sigmoid(raw_gap)
        return normalised


# ===========================================================================
# SECTION 6: CoherenceComputer
# ===========================================================================

class CoherenceComputer:
    """Computes coherence metrics for the winning slot coalition.

    Coherence measures whether the winning slots form a meaningful coalition.
    A coalition that spans multiple modalities AND has mutually consistent
    embeddings is more likely to represent a genuine conscious percept.

    Two metrics:
    1. **Cross-modal** — Fraction of unique modalities among winners.
       Perfect cross-modal coherence = all slots from different modalities.
    2. **Mutual consistency** — Mean pairwise cosine similarity among
       winning slot embeddings.
    """

    @staticmethod
    def compute_cross_modal(
        winners_metadata: Dict[str, Tensor],
    ) -> Tensor:
        """Compute cross-modal diversity of winners.

        Args:
            winners_metadata: Dictionary with at least:
                - "modality_ids": (B, K) int tensor where each value is a
                  modality index (0, 1, 2, ...). Different integers = different
                  modalities.

        Returns:
            (B,) fraction of unique modalities in [0, 1].
            If K=0 or modality_ids not present, returns zeros.
        """
        if "modality_ids" not in winners_metadata:
            # Fallback: assume all from same modality
            # Need batch size from some other key or return a default
            for key, val in winners_metadata.items():
                if isinstance(val, Tensor) and val.dim() >= 1:
                    B = val.shape[0]
                    return torch.zeros(B, device=val.device, dtype=torch.float32)
            return torch.zeros(1, dtype=torch.float32)

        modality_ids = winners_metadata["modality_ids"]  # (B, K)
        B, K = modality_ids.shape

        if K == 0:
            return torch.zeros(B, device=modality_ids.device, dtype=torch.float32)

        # Count unique modalities per batch element
        # Use a loop over batch (small B typically) for clarity
        device = modality_ids.device
        diversity = torch.zeros(B, device=device, dtype=torch.float32)

        for b in range(B):
            num_unique = modality_ids[b].unique().numel()
            diversity[b] = float(num_unique) / float(K)

        return diversity.clamp(_SCORE_MIN, _SCORE_MAX)

    @staticmethod
    def compute_mutual_consistency(
        slots: Tensor,
    ) -> Tensor:
        """Compute mean pairwise cosine similarity among winning slots.

        Args:
            slots: (B, K, D) winning slot embeddings.

        Returns:
            (B,) mean pairwise cosine similarity in [0, 1].
            If K <= 1, returns ones (a single slot is trivially consistent).
        """
        B, K, D = slots.shape

        if K <= 1:
            return torch.ones(B, device=slots.device, dtype=torch.float32)

        # Normalise
        slots_norm = F.normalize(slots.float(), dim=-1)  # (B, K, D)

        # Pairwise cosine similarity: (B, K, K)
        sim_matrix = torch.bmm(slots_norm, slots_norm.transpose(1, 2))

        # Extract upper triangle (exclude diagonal)
        # Create mask for upper triangle
        mask = torch.triu(torch.ones(K, K, device=slots.device, dtype=torch.bool), diagonal=1)
        mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B, K, K)

        # Mean of upper triangle entries
        num_pairs = K * (K - 1) // 2
        pairwise_sum = (sim_matrix * mask.float()).sum(dim=(-2, -1))  # (B,)
        mean_sim = pairwise_sum / (float(num_pairs) + _EPS)

        # Shift from [-1, 1] to [0, 1]
        normalised = (mean_sim + 1.0) / 2.0
        return normalised.clamp(_SCORE_MIN, _SCORE_MAX)

    @staticmethod
    def combine(
        cross_modal: Tensor,
        consistency: Tensor,
        alpha: float = 0.5,
    ) -> Tensor:
        """Combine cross-modal diversity and mutual consistency.

        Args:
            cross_modal: (B,) cross-modal diversity.
            consistency: (B,) mutual consistency.
            alpha: Weight for cross_modal; (1-alpha) for consistency.

        Returns:
            (B,) overall coherence in [0, 1].
        """
        combined = alpha * cross_modal + (1.0 - alpha) * consistency
        return combined.clamp(_SCORE_MIN, _SCORE_MAX)


# ===========================================================================
# SECTION 7: IgnitionScorer (nn.Module)
# ===========================================================================

class IgnitionScorer(nn.Module):
    """Combines the four ignition components into a single score.

    Two modes:
    1. **Interpretable** (default): Weighted sum of the four component scores,
       clamped to [0, 1]. Weights are specified in IgnitionConfig and are NOT
       learnable. This mode is fully transparent: you can read off exactly how
       much each component contributed.

    2. **Learned**: An MLP that takes the 4 component scalars as input and
       outputs a single ignition score. This mode can capture nonlinear
       interactions between components but sacrifices interpretability.

    All computation is performed in fp32 regardless of input dtype.

    Args:
        config: IgnitionConfig with component weights and learned mode settings.
    """

    def __init__(self, config: IgnitionConfig) -> None:
        super().__init__()
        self.config = config

        if config.use_learned_ignition:
            hidden = config.learned_ignition_hidden
            self.mlp = nn.Sequential(
                nn.Linear(4, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.mlp = None

        # Register weights as buffer for easy serialisation (non-learnable)
        self.register_buffer(
            "_weights",
            torch.tensor(
                [config.w_stability, config.w_confidence,
                 config.w_margin, config.w_coherence],
                dtype=torch.float32,
            ),
        )

    def forward(self, components: IgnitionComponents) -> Tensor:
        """Compute ignition score from components.

        Args:
            components: IgnitionComponents with (B,) tensors.

        Returns:
            (B,) ignition score in [0, 1].
        """
        if self.mlp is not None:
            # Learned mode
            x = components.to_tensor().float()  # (B, 4)
            score = self.mlp(x).squeeze(-1)     # (B,)
        else:
            # Interpretable mode: weighted sum
            stacked = components.to_tensor().float()  # (B, 4)
            weights = self._weights.to(stacked.device)  # (4,)

            # Normalise weights to sum to 1
            w_sum = weights.sum() + _EPS
            normalised_weights = weights / w_sum

            score = (stacked * normalised_weights.unsqueeze(0)).sum(dim=-1)  # (B,)

        return score.clamp(_SCORE_MIN, _SCORE_MAX)


# ===========================================================================
# SECTION 8: IgnitionGate (nn.Module)
# ===========================================================================

class IgnitionGate(nn.Module):
    """Applies gain modulation to slots based on ignition score.

    During training, a smooth sigmoid gate is used so that gradients can
    flow through the gating decision. During testing, a hard threshold
    is applied for deterministic, low-latency behaviour.

    The gain is computed as:
        Training: gain = sigmoid((score - threshold) / temperature)
        Testing:  gain = (score >= threshold).float()

    Slots below the threshold still receive a weak_gain to prevent
    complete information loss:
        effective_gain = weak_gain + (1 - weak_gain) * gain

    Gated slots are:
        gated_slots = slots * effective_gain[:, None, None]

    Args:
        config: IgnitionConfig with threshold, temperature, and weak_gain.
    """

    def __init__(self, config: IgnitionConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        slots: Tensor,
        ignition_score: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply ignition gating to slots.

        Args:
            slots: (B, K, D) slot embeddings.
            ignition_score: (B,) ignition scores in [0, 1].

        Returns:
            gated_slots: (B, K, D) gain-modulated slots.
            effective_gain: (B,) the gain actually applied.
        """
        threshold = self.config.ignition_threshold
        temperature = self.config.gate_temperature
        weak_gain = self.config.weak_gain

        score = ignition_score.float()  # (B,)

        if self.training:
            # Smooth sigmoid gating for gradient flow
            raw_gain = torch.sigmoid(
                (score - threshold) / temperature
            )  # (B,)
        else:
            # Hard threshold for deterministic testing
            raw_gain = (score >= threshold).float()  # (B,)

        # Apply weak gain floor
        effective_gain = weak_gain + (1.0 - weak_gain) * raw_gain  # (B,)

        # Modulate slots
        gated_slots = slots.float() * effective_gain.unsqueeze(-1).unsqueeze(-1)

        return gated_slots, effective_gain


# ===========================================================================
# SECTION 9: LockInPrevention (nn.Module)
# ===========================================================================

class LockInPrevention(nn.Module):
    """Mechanisms to prevent the same coalition from dominating the workspace.

    In a healthy Global Workspace, different coalitions should win access
    at different times depending on stimulus relevance. Without explicit
    prevention, a strong coalition can "lock in" and suppress novel inputs.

    Three mechanisms:
    1. **Novelty scoring** — Tokens dissimilar to the current working memory
       summary receive a novelty bonus, biasing competition toward fresh input.
    2. **Winner decay** — Previous winners' scores are decayed, reducing their
       competitive advantage in the next round.
    3. **Slot dropout** — During training, random winning slots are dropped,
       forcing the network to build robust coalitions.
    4. **Cooldown** — After ignition, the threshold is temporarily boosted,
       making re-ignition harder and encouraging new coalitions to form.

    Args:
        config: IgnitionConfig with decay, dropout, and cooldown parameters.
    """

    def __init__(self, config: IgnitionConfig) -> None:
        super().__init__()
        self.config = config

    def novelty_score(
        self,
        tokens: Tensor,
        wm_summary: Tensor,
    ) -> Tensor:
        """Compute per-token novelty relative to working memory.

        Novelty = 1 - cosine_similarity(token, wm_summary).
        Tokens very different from current working memory content are
        considered novel and should receive a competition bonus.

        Args:
            tokens: (B, T, D) all candidate tokens.
            wm_summary: (B, D) current working memory summary vector.

        Returns:
            (B, T) novelty scores in [0, 2] (since cosine sim is in [-1, 1]).
        """
        tokens_f = tokens.float()
        wm_f = wm_summary.float()

        # Expand wm_summary for broadcasting: (B, 1, D)
        wm_expanded = wm_f.unsqueeze(1)

        # Cosine similarity per token: (B, T)
        cos_sim = F.cosine_similarity(tokens_f, wm_expanded, dim=-1)

        # Novelty = 1 - similarity
        novelty = 1.0 - cos_sim

        return novelty

    def apply_winner_decay(
        self,
        scores: Tensor,
        prev_winner_mask: Tensor,
        decay_factor: Optional[float] = None,
    ) -> Tensor:
        """Decay scores of previous round winners.

        Args:
            scores: (B, N) current competition scores.
            prev_winner_mask: (B, N) binary mask of previous winners.
            decay_factor: Override for config.winner_decay.

        Returns:
            (B, N) adjusted scores with winners decayed.
        """
        if decay_factor is None:
            decay_factor = self.config.winner_decay

        scores_f = scores.float()
        mask_f = prev_winner_mask.float()

        # Decay: winners get their scores reduced
        # adjusted = scores * (1 - decay * mask)
        adjustment = 1.0 - decay_factor * mask_f
        adjusted = scores_f * adjustment

        return adjusted

    def apply_slot_dropout(
        self,
        slots: Tensor,
        slot_mask: Tensor,
        p: Optional[float] = None,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Apply dropout to winning slots during training.

        Args:
            slots: (B, K, D) winning slot embeddings.
            slot_mask: (B, K) binary mask indicating active slots.
            p: Override for config.slot_dropout.
            training: Override for self.training.

        Returns:
            dropped_slots: (B, K, D) with some slots zeroed out.
            dropped_mask: (B, K) updated mask after dropout.
        """
        if p is None:
            p = self.config.slot_dropout
        if training is None:
            training = self.training

        if not training or p <= 0.0:
            return slots, slot_mask

        B, K, D = slots.shape

        # Generate dropout mask: 1 = keep, 0 = drop
        keep_prob = 1.0 - p
        dropout_mask = torch.bernoulli(
            torch.full((B, K), keep_prob, device=slots.device)
        )  # (B, K)

        # Ensure at least one slot survives per batch element
        # If all would be dropped, keep the first one
        all_dropped = (dropout_mask.sum(dim=-1) == 0)  # (B,)
        if all_dropped.any():
            dropout_mask[all_dropped, 0] = 1.0

        # Apply
        dropped_slots = slots * dropout_mask.unsqueeze(-1)
        dropped_mask = slot_mask * dropout_mask

        return dropped_slots, dropped_mask

    def check_cooldown(
        self,
        ignition_history: List[bool],
        cooldown_steps: Optional[int] = None,
        cooldown_boost: Optional[float] = None,
    ) -> Tensor:
        """Check if cooldown is active and compute effective threshold.

        After ignition occurs, the threshold is temporarily boosted for
        `cooldown_steps` steps, making re-ignition harder. This encourages
        the workspace to explore new coalitions.

        Args:
            ignition_history: List of booleans, most recent last.
                True = ignition occurred at that step.
            cooldown_steps: Override for config.cooldown_steps.
            cooldown_boost: Override for config.cooldown_boost.

        Returns:
            Scalar tensor: effective threshold (possibly boosted).
        """
        if cooldown_steps is None:
            cooldown_steps = self.config.cooldown_steps
        if cooldown_boost is None:
            cooldown_boost = self.config.cooldown_boost

        base_threshold = self.config.ignition_threshold

        if cooldown_steps <= 0 or len(ignition_history) == 0:
            return torch.tensor(base_threshold, dtype=torch.float32)

        # Check if any ignition occurred in the last cooldown_steps entries
        recent = ignition_history[-cooldown_steps:]
        cooldown_active = any(recent)

        if cooldown_active:
            effective = base_threshold + cooldown_boost
            # Clamp to valid range
            effective = min(effective, 0.99)
        else:
            effective = base_threshold

        return torch.tensor(effective, dtype=torch.float32)


# ===========================================================================
# SECTION 10: IgnitionModule (nn.Module) — Orchestrator
# ===========================================================================

class IgnitionModule(nn.Module):
    """Orchestrates all ignition dynamics logic.

    This is the main entry point for the ignition subsystem. It computes
    all four component scores, combines them into an ignition score, applies
    gating, and handles lock-in prevention.

    Components:
        - StabilityComputer: winner set and embedding convergence
        - ConfidenceComputer: softmax margin and score gap
        - CoherenceComputer: cross-modal diversity and mutual consistency
        - IgnitionScorer: combines components into final score
        - IgnitionGate: applies gain modulation to slots
        - LockInPrevention: novelty, decay, dropout, cooldown

    Usage:
        config = IgnitionConfig()
        module = IgnitionModule(config)

        result = module(
            slots=current_features,        # (B, K, D) winning slot embeddings
            all_scores=salience_scores,    # (B, N) all candidate scores
            winners_metadata={             # metadata about winners
                "modality_ids": ...,       # (B, K) int modality indices
                "winner_mask": ...,        # (B, N) binary winner mask
            },
            round_history={                # history from previous rounds
                "prev_slots": ...,         # (B, K, D) previous round slots
                "prev_winner_mask": ...,   # (B, N) previous winner mask
            },
            prev_state={                   # state from previous timesteps
                "ignition_history": [...], # list of bools
                "wm_summary": ...,         # (B, D) working memory summary
            },
        )

    Args:
        config: IgnitionConfig with all hyperparameters.
        workspace_dim: Dimension of slot embeddings. Only needed if
            use_learned_ignition is True (for proper MLP initialisation).
    """

    def __init__(
        self,
        config: Optional[IgnitionConfig] = None,
        workspace_dim: int = _DEFAULT_WORKSPACE_DIM,
    ) -> None:
        super().__init__()

        self.config = config or IgnitionConfig()
        self.workspace_dim = workspace_dim

        # Sub-components (non-nn.Module, stateless)
        self.stability_computer = StabilityComputer()
        self.confidence_computer = ConfidenceComputer()
        self.coherence_computer = CoherenceComputer()

        # nn.Module components
        self.scorer = IgnitionScorer(self.config)
        self.gate = IgnitionGate(self.config)
        self.lock_in = LockInPrevention(self.config)

    def _compute_stability(
        self,
        slots: Tensor,
        round_history: Optional[Dict[str, Tensor]],
        winners_metadata: Dict[str, Tensor],
    ) -> Tensor:
        """Compute stability component.

        If no previous round data is available, returns zeros (no stability
        information yet).
        """
        B = slots.shape[0]
        device = slots.device

        if round_history is None:
            return torch.zeros(B, device=device, dtype=torch.float32)

        prev_slots = round_history.get("prev_slots")
        prev_winner_mask = round_history.get("prev_winner_mask")
        current_winner_mask = winners_metadata.get("winner_mask")

        # Winner stability
        if prev_winner_mask is not None and current_winner_mask is not None:
            winner_stab = self.stability_computer.compute_winner_stability(
                current_winner_mask, prev_winner_mask,
            )
        else:
            winner_stab = torch.zeros(B, device=device, dtype=torch.float32)

        # Embedding stability
        if prev_slots is not None and prev_slots.shape == slots.shape:
            embed_stab = self.stability_computer.compute_embedding_stability(
                slots, prev_slots,
            )
        else:
            embed_stab = torch.zeros(B, device=device, dtype=torch.float32)

        return self.stability_computer.combine(winner_stab, embed_stab)

    def _compute_confidence(
        self,
        all_scores: Tensor,
        winners_metadata: Dict[str, Tensor],
    ) -> Tensor:
        """Compute confidence component."""
        K = winners_metadata.get("K", None)
        if K is None:
            # Infer K from modality_ids shape
            mid = winners_metadata.get("modality_ids")
            if mid is not None:
                K = mid.shape[-1]
            else:
                # Fallback: assume half the candidates
                K = max(1, all_scores.shape[-1] // 2)

        return self.confidence_computer.compute_margin(all_scores, K)

    def _compute_margin(
        self,
        all_scores: Tensor,
        winners_metadata: Dict[str, Tensor],
    ) -> Tensor:
        """Compute margin component."""
        winner_mask = winners_metadata.get("winner_mask")

        if winner_mask is None:
            # Cannot compute margin without knowing which are winners
            B = all_scores.shape[0]
            return torch.full(
                (B,), 0.5, device=all_scores.device, dtype=torch.float32
            )

        # Separate winner and loser scores
        mask_bool = winner_mask.bool()
        B, N = all_scores.shape

        # Gather winner and loser scores
        winner_scores_list = []
        loser_scores_list = []

        for b in range(B):
            w_scores = all_scores[b][mask_bool[b]]
            l_scores = all_scores[b][~mask_bool[b]]
            winner_scores_list.append(w_scores)
            loser_scores_list.append(l_scores)

        # Pad to equal length for batching
        max_w = max(w.numel() for w in winner_scores_list)
        max_l = max(l.numel() for l in loser_scores_list)

        if max_w == 0 or max_l == 0:
            return torch.full(
                (B,), 0.5, device=all_scores.device, dtype=torch.float32
            )

        winner_scores = torch.zeros(B, max_w, device=all_scores.device)
        loser_scores = torch.zeros(B, max_l, device=all_scores.device)

        for b in range(B):
            wn = winner_scores_list[b].numel()
            ln = loser_scores_list[b].numel()
            if wn > 0:
                winner_scores[b, :wn] = winner_scores_list[b]
            if ln > 0:
                loser_scores[b, :ln] = loser_scores_list[b]

        return self.confidence_computer.compute_mean_gap(winner_scores, loser_scores)

    def _compute_coherence(
        self,
        slots: Tensor,
        winners_metadata: Dict[str, Tensor],
    ) -> Tensor:
        """Compute coherence component."""
        cross_modal = self.coherence_computer.compute_cross_modal(winners_metadata)
        consistency = self.coherence_computer.compute_mutual_consistency(slots)

        # Ensure matching devices
        cross_modal = cross_modal.to(slots.device)
        consistency = consistency.to(slots.device)

        return self.coherence_computer.combine(cross_modal, consistency)

    def forward(
        self,
        slots: Tensor,
        all_scores: Tensor,
        winners_metadata: Dict[str, Tensor],
        round_history: Optional[Dict[str, Tensor]] = None,
        prev_state: Optional[Dict[str, object]] = None,
    ) -> IgnitionResult:
        """Run the full ignition pipeline.

        Args:
            slots: (B, K, D) winning slot embeddings after competition.
            all_scores: (B, N) scores for all N candidate slots.
            winners_metadata: Dictionary containing:
                - "modality_ids": (B, K) int modality indices per winner.
                - "winner_mask": (B, N) binary mask of winners among all N.
                - "K": int, number of winners (optional, inferred if absent).
            round_history: Optional dictionary from previous competition round:
                - "prev_slots": (B, K, D) previous round slot embeddings.
                - "prev_winner_mask": (B, N) previous round winner mask.
            prev_state: Optional state from previous timesteps:
                - "ignition_history": List[bool] of past ignition outcomes.
                - "wm_summary": (B, D) working memory summary for novelty.

        Returns:
            IgnitionResult with all outputs.
        """
        if prev_state is None:
            prev_state = {}

        B, K, D = slots.shape
        device = slots.device

        # --- Compute components (all in fp32) ---
        stability = self._compute_stability(slots, round_history, winners_metadata)
        confidence = self._compute_confidence(all_scores, winners_metadata)
        margin = self._compute_margin(all_scores, winners_metadata)
        coherence = self._compute_coherence(slots, winners_metadata)

        components = IgnitionComponents(
            stability=stability,
            confidence=confidence,
            margin=margin,
            coherence=coherence,
        )

        # --- Score ---
        ignition_score = self.scorer(components)  # (B,)

        # --- Cooldown ---
        ignition_history = prev_state.get("ignition_history", [])
        effective_threshold_scalar = self.lock_in.check_cooldown(
            ignition_history,
        )
        effective_threshold = effective_threshold_scalar.to(device)
        cooldown_active_bool = (
            effective_threshold.item() > self.config.ignition_threshold
        )
        cooldown_active = torch.full(
            (B,), cooldown_active_bool, device=device, dtype=torch.bool,
        )

        # If cooldown is active, temporarily override gate config threshold
        if cooldown_active_bool:
            # Create a temporary config with boosted threshold
            original_threshold = self.gate.config.ignition_threshold
            self.gate.config.ignition_threshold = effective_threshold.item()

        # --- Gate ---
        gated_slots, effective_gain = self.gate(slots, ignition_score)

        # Restore original threshold if we changed it
        if cooldown_active_bool:
            self.gate.config.ignition_threshold = original_threshold

        # --- Determine ignited ---
        ignited = (ignition_score >= effective_threshold).to(torch.bool)

        # --- Lock-in: slot dropout (only during training) ---
        if self.training and self.config.slot_dropout > 0.0:
            slot_mask = torch.ones(B, K, device=device)
            gated_slots, _ = self.lock_in.apply_slot_dropout(
                gated_slots, slot_mask,
            )

        return IgnitionResult(
            ignition_score=ignition_score,
            ignited=ignited,
            effective_gain=effective_gain,
            gated_slots=gated_slots,
            components=components,
            cooldown_active=cooldown_active,
        )


# ===========================================================================
# SECTION 11: Utility functions
# ===========================================================================

def create_ignition_module(
    config: Optional[IgnitionConfig] = None,
    workspace_dim: int = _DEFAULT_WORKSPACE_DIM,
    **kwargs,
) -> IgnitionModule:
    """Factory function for IgnitionModule.

    Args:
        config: IgnitionConfig. If None, a default config is created.
        workspace_dim: Dimension of workspace slot embeddings.
        **kwargs: Overrides for IgnitionConfig if config is None.

    Returns:
        Configured IgnitionModule.
    """
    if config is None:
        config = IgnitionConfig(**kwargs)
    return IgnitionModule(config=config, workspace_dim=workspace_dim)


def compute_ignition_components_from_rounds(
    round_slots: List[Tensor],
    round_scores: List[Tensor],
    round_winner_masks: List[Tensor],
    modality_ids: Tensor,
    K: int,
) -> Tuple[IgnitionComponents, Dict[str, Tensor]]:
    """Convenience function: compute IgnitionComponents from round history.

    Given the full history of iterative competition rounds, compute all
    four component scores for the FINAL round.

    Args:
        round_slots: List of (B, K, D) slot tensors, one per round.
        round_scores: List of (B, N) score tensors, one per round.
        round_winner_masks: List of (B, N) binary winner masks, one per round.
        modality_ids: (B, K) modality indices for winners.
        K: Number of winners.

    Returns:
        components: IgnitionComponents for the final round.
        metadata: Dict suitable for passing to IgnitionModule.
    """
    assert len(round_slots) >= 1, "Need at least one round"

    final_slots = round_slots[-1]
    final_scores = round_scores[-1]
    final_mask = round_winner_masks[-1]

    # Stability: compare last two rounds (or zero if only one round)
    if len(round_slots) >= 2:
        prev_slots = round_slots[-2]
        prev_mask = round_winner_masks[-2]

        winner_stab = StabilityComputer.compute_winner_stability(
            final_mask, prev_mask,
        )
        embed_stab = StabilityComputer.compute_embedding_stability(
            final_slots, prev_slots,
        )
        stability = StabilityComputer.combine(winner_stab, embed_stab)
    else:
        B = final_slots.shape[0]
        stability = torch.zeros(B, device=final_slots.device, dtype=torch.float32)

    # Confidence
    confidence = ConfidenceComputer.compute_margin(final_scores, K)

    # Margin
    B, N = final_scores.shape
    mask_bool = final_mask.bool()
    # Simple batch-level computation
    winner_scores_all = []
    loser_scores_all = []
    for b in range(B):
        ws = final_scores[b][mask_bool[b]]
        ls = final_scores[b][~mask_bool[b]]
        winner_scores_all.append(ws)
        loser_scores_all.append(ls)

    # Pad for batching
    max_w = max(w.numel() for w in winner_scores_all) if winner_scores_all else 1
    max_l = max(l.numel() for l in loser_scores_all) if loser_scores_all else 1
    winner_batch = torch.zeros(B, max(max_w, 1), device=final_scores.device)
    loser_batch = torch.zeros(B, max(max_l, 1), device=final_scores.device)
    for b in range(B):
        wn = winner_scores_all[b].numel()
        ln = loser_scores_all[b].numel()
        if wn > 0:
            winner_batch[b, :wn] = winner_scores_all[b]
        if ln > 0:
            loser_batch[b, :ln] = loser_scores_all[b]

    margin = ConfidenceComputer.compute_mean_gap(winner_batch, loser_batch)

    # Coherence
    metadata_dict = {"modality_ids": modality_ids}
    cross_modal = CoherenceComputer.compute_cross_modal(metadata_dict)
    consistency = CoherenceComputer.compute_mutual_consistency(final_slots)
    cross_modal = cross_modal.to(final_slots.device)
    coherence = CoherenceComputer.combine(cross_modal, consistency)

    components = IgnitionComponents(
        stability=stability,
        confidence=confidence,
        margin=margin,
        coherence=coherence,
    )

    metadata = {
        "modality_ids": modality_ids,
        "winner_mask": final_mask,
        "K": K,
    }

    return components, metadata


# ===========================================================================
# SECTION 12: __all__ exports
# ===========================================================================

__all__ = [
    # Config
    "IgnitionConfig",
    # Data structures
    "IgnitionComponents",
    "IgnitionResult",
    # Computers (stateless)
    "StabilityComputer",
    "ConfidenceComputer",
    "CoherenceComputer",
    # nn.Module components
    "IgnitionScorer",
    "IgnitionGate",
    "LockInPrevention",
    # Orchestrator
    "IgnitionModule",
    # Factory
    "create_ignition_module",
    # Utilities
    "compute_ignition_components_from_rounds",
    # Constants
    "_EPS",
    "_SCORE_MIN",
    "_SCORE_MAX",
    "_DEFAULT_WORKSPACE_DIM",
]


# ===========================================================================
# SECTION 13: Self-test — run with: python ignition_template.py
# ===========================================================================

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  ignition_template.py — Self-test suite")
    print("=" * 70)

    passed = 0
    failed = 0

    def _check(name: str, condition: bool, msg: str = "") -> None:
        global passed, failed
        status = "PASS" if condition else "FAIL"
        label = f"  [{status}] {name}"
        if not condition and msg:
            label += f"\n         Reason: {msg}"
        print(label)
        if condition:
            passed += 1
        else:
            failed += 1

    device = torch.device("cpu")
    B = 4       # batch size
    K = 5       # number of winner slots
    N = 12      # total candidates
    D = 64      # workspace dim

    torch.manual_seed(42)

    # -------------------------------------------------------------------
    # Test 1: StabilityComputer — identical winners produce Jaccard = 1.0
    # -------------------------------------------------------------------
    print("\n--- Test 1: Stability with identical winners ---")

    winners = torch.zeros(B, N)
    winners[:, :K] = 1.0
    winner_stab = StabilityComputer.compute_winner_stability(winners, winners)
    _check(
        "identical winners: Jaccard == 1.0",
        torch.allclose(winner_stab, torch.ones(B), atol=1e-5),
        f"got {winner_stab}",
    )

    # -------------------------------------------------------------------
    # Test 2: StabilityComputer — completely different winners produce Jaccard = 0.0
    # -------------------------------------------------------------------
    print("\n--- Test 2: Stability with disjoint winners ---")

    w_current = torch.zeros(B, N)
    w_current[:, :K] = 1.0
    w_previous = torch.zeros(B, N)
    w_previous[:, K:2*K] = 1.0  # non-overlapping (requires N >= 2*K)

    # For safety, ensure N >= 2*K
    if N >= 2 * K:
        disjoint_stab = StabilityComputer.compute_winner_stability(w_current, w_previous)
        _check(
            "disjoint winners: Jaccard == 0.0",
            torch.allclose(disjoint_stab, torch.zeros(B), atol=1e-5),
            f"got {disjoint_stab}",
        )
    else:
        _check(
            "disjoint winners: skipped (N < 2*K)",
            True,
        )

    # -------------------------------------------------------------------
    # Test 3: StabilityComputer — embedding stability with identical slots
    # -------------------------------------------------------------------
    print("\n--- Test 3: Embedding stability — identical slots ---")

    slots_a = torch.randn(B, K, D)
    embed_stab = StabilityComputer.compute_embedding_stability(slots_a, slots_a)
    _check(
        "identical slots: embed stability == 1.0",
        torch.allclose(embed_stab, torch.ones(B), atol=1e-4),
        f"got {embed_stab}",
    )

    # -------------------------------------------------------------------
    # Test 4: StabilityComputer — embedding stability with negated slots
    # -------------------------------------------------------------------
    print("\n--- Test 4: Embedding stability — negated slots ---")

    slots_neg = -slots_a
    embed_stab_neg = StabilityComputer.compute_embedding_stability(slots_a, slots_neg)
    _check(
        "negated slots: embed stability == 0.0",
        torch.allclose(embed_stab_neg, torch.zeros(B), atol=1e-4),
        f"got {embed_stab_neg}",
    )

    # -------------------------------------------------------------------
    # Test 5: ConfidenceComputer — margin with clear winner
    # -------------------------------------------------------------------
    print("\n--- Test 5: Confidence margin — clear winner ---")

    # Make first K scores much larger than rest
    scores_clear = torch.zeros(B, N)
    scores_clear[:, :K] = 10.0
    scores_clear[:, K:] = -10.0

    margin_clear = ConfidenceComputer.compute_margin(scores_clear, K)
    _check(
        "clear winner: margin close to 1.0",
        (margin_clear > 0.9).all().item(),
        f"got {margin_clear}",
    )

    # -------------------------------------------------------------------
    # Test 6: ConfidenceComputer — margin with uniform scores
    # -------------------------------------------------------------------
    print("\n--- Test 6: Confidence margin — uniform scores ---")

    scores_uniform = torch.zeros(B, N)
    margin_uniform = ConfidenceComputer.compute_margin(scores_uniform, K)
    _check(
        "uniform scores: margin close to 0.0",
        (margin_uniform < 0.1).all().item(),
        f"got {margin_uniform}",
    )

    # -------------------------------------------------------------------
    # Test 7: ConfidenceComputer — mean gap
    # -------------------------------------------------------------------
    print("\n--- Test 7: Confidence mean gap ---")

    w_scores = torch.ones(B, K) * 5.0
    l_scores = torch.ones(B, N - K) * -5.0
    gap = ConfidenceComputer.compute_mean_gap(w_scores, l_scores)
    _check(
        "large gap: sigmoid(10) close to 1.0",
        (gap > 0.99).all().item(),
        f"got {gap}",
    )

    # -------------------------------------------------------------------
    # Test 8: CoherenceComputer — single modality vs multi-modality
    # -------------------------------------------------------------------
    print("\n--- Test 8: Cross-modal coherence ---")

    # All same modality
    same_mod = torch.zeros(B, K, dtype=torch.long)
    meta_same = {"modality_ids": same_mod}
    cross_same = CoherenceComputer.compute_cross_modal(meta_same)
    _check(
        "single modality: cross-modal == 1/K",
        torch.allclose(cross_same, torch.full((B,), 1.0 / K), atol=1e-5),
        f"got {cross_same}",
    )

    # All different modalities
    diff_mod = torch.arange(K).unsqueeze(0).expand(B, -1)
    meta_diff = {"modality_ids": diff_mod}
    cross_diff = CoherenceComputer.compute_cross_modal(meta_diff)
    _check(
        "all different modalities: cross-modal == 1.0",
        torch.allclose(cross_diff, torch.ones(B), atol=1e-5),
        f"got {cross_diff}",
    )

    # -------------------------------------------------------------------
    # Test 9: CoherenceComputer — mutual consistency
    # -------------------------------------------------------------------
    print("\n--- Test 9: Mutual consistency ---")

    # Identical slots produce perfect consistency
    identical_slots = torch.randn(1, 1, D).expand(B, K, D)
    consistency_high = CoherenceComputer.compute_mutual_consistency(identical_slots)
    _check(
        "identical slots: mutual consistency == 1.0",
        torch.allclose(consistency_high, torch.ones(B), atol=1e-4),
        f"got {consistency_high}",
    )

    # K=1 is trivially consistent
    single_slot = torch.randn(B, 1, D)
    consistency_single = CoherenceComputer.compute_mutual_consistency(single_slot)
    _check(
        "single slot: mutual consistency == 1.0",
        torch.allclose(consistency_single, torch.ones(B), atol=1e-5),
        f"got {consistency_single}",
    )

    # -------------------------------------------------------------------
    # Test 10: IgnitionScorer — interpretable weighted combination
    # -------------------------------------------------------------------
    print("\n--- Test 10: IgnitionScorer — interpretable mode ---")

    cfg_interp = IgnitionConfig(
        w_stability=0.3, w_confidence=0.3,
        w_margin=0.2, w_coherence=0.2,
    )
    scorer_interp = IgnitionScorer(cfg_interp)

    components_high = IgnitionComponents(
        stability=torch.ones(B),
        confidence=torch.ones(B),
        margin=torch.ones(B),
        coherence=torch.ones(B),
    )
    score_high = scorer_interp(components_high)
    _check(
        "all-ones components: score == 1.0",
        torch.allclose(score_high, torch.ones(B), atol=1e-5),
        f"got {score_high}",
    )

    components_zero = IgnitionComponents(
        stability=torch.zeros(B),
        confidence=torch.zeros(B),
        margin=torch.zeros(B),
        coherence=torch.zeros(B),
    )
    score_zero = scorer_interp(components_zero)
    _check(
        "all-zeros components: score == 0.0",
        torch.allclose(score_zero, torch.zeros(B), atol=1e-5),
        f"got {score_zero}",
    )

    # -------------------------------------------------------------------
    # Test 11: IgnitionScorer — learned mode
    # -------------------------------------------------------------------
    print("\n--- Test 11: IgnitionScorer — learned mode ---")

    cfg_learned = IgnitionConfig(use_learned_ignition=True, learned_ignition_hidden=32)
    scorer_learned = IgnitionScorer(cfg_learned)

    score_learned = scorer_learned(components_high)
    _check(
        "learned mode: output shape == (B,)",
        score_learned.shape == (B,),
        f"got shape {score_learned.shape}",
    )
    _check(
        "learned mode: output in [0, 1]",
        bool((score_learned >= 0.0).all() and (score_learned <= 1.0).all()),
        f"range [{score_learned.min().item():.4f}, {score_learned.max().item():.4f}]",
    )

    # Gradient flows through learned scorer
    scorer_learned.zero_grad()
    loss = score_learned.sum()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in scorer_learned.parameters()
    )
    _check(
        "learned mode: gradient flows to MLP parameters",
        has_grad,
    )

    # -------------------------------------------------------------------
    # Test 12: IgnitionGate — smooth vs hard
    # -------------------------------------------------------------------
    print("\n--- Test 12: IgnitionGate — smooth vs hard ---")

    cfg_gate = IgnitionConfig(
        ignition_threshold=0.5, gate_temperature=0.1, weak_gain=0.0,
    )
    gate = IgnitionGate(cfg_gate)

    test_slots = torch.randn(B, K, D)
    high_score = torch.ones(B) * 0.9   # well above threshold
    low_score = torch.ones(B) * 0.1    # well below threshold

    # Training mode: smooth sigmoid
    gate.train()
    gated_high_train, gain_high_train = gate(test_slots, high_score)
    gated_low_train, gain_low_train = gate(test_slots, low_score)
    _check(
        "train mode: high score gain > 0.9",
        (gain_high_train > 0.9).all().item(),
        f"got {gain_high_train}",
    )
    _check(
        "train mode: low score gain < 0.1",
        (gain_low_train < 0.1).all().item(),
        f"got {gain_low_train}",
    )

    # Non-training mode: hard threshold
    gate.train(False)
    gated_high_hard, gain_high_hard = gate(test_slots, high_score)
    gated_low_hard, gain_low_hard = gate(test_slots, low_score)
    _check(
        "non-training mode: high score gain == 1.0",
        torch.allclose(gain_high_hard, torch.ones(B), atol=1e-5),
        f"got {gain_high_hard}",
    )
    _check(
        "non-training mode: low score gain == 0.0",
        torch.allclose(gain_low_hard, torch.zeros(B), atol=1e-5),
        f"got {gain_low_hard}",
    )

    # -------------------------------------------------------------------
    # Test 13: IgnitionGate — weak gain floor
    # -------------------------------------------------------------------
    print("\n--- Test 13: IgnitionGate — weak gain floor ---")

    cfg_weak = IgnitionConfig(
        ignition_threshold=0.5, gate_temperature=0.1, weak_gain=0.3,
    )
    gate_weak = IgnitionGate(cfg_weak)
    gate_weak.train(False)

    _, gain_low_weak = gate_weak(test_slots, low_score)
    _check(
        "weak gain: low score gain >= weak_gain (0.3)",
        (gain_low_weak >= 0.3 - 1e-5).all().item(),
        f"got {gain_low_weak}",
    )

    # -------------------------------------------------------------------
    # Test 14: LockInPrevention — novelty score
    # -------------------------------------------------------------------
    print("\n--- Test 14: LockInPrevention — novelty score ---")

    cfg_lock = IgnitionConfig()
    lock_in = LockInPrevention(cfg_lock)

    tokens = torch.randn(B, N, D)
    wm_summary = torch.randn(B, D)

    novelty = lock_in.novelty_score(tokens, wm_summary)
    _check(
        "novelty shape: (B, N)",
        novelty.shape == (B, N),
        f"got {novelty.shape}",
    )

    # Identical token and wm_summary produce novelty near 0
    wm_single = torch.randn(1, D).expand(B, D)
    tokens_same = wm_single.unsqueeze(1).expand(B, N, D)
    novelty_same = lock_in.novelty_score(tokens_same, wm_single)
    _check(
        "identical tokens: novelty close to 0",
        (novelty_same.abs() < 1e-4).all().item(),
        f"max novelty = {novelty_same.abs().max().item():.6f}",
    )

    # -------------------------------------------------------------------
    # Test 15: LockInPrevention — winner decay
    # -------------------------------------------------------------------
    print("\n--- Test 15: LockInPrevention — winner decay ---")

    scores_orig = torch.ones(B, N)
    prev_mask = torch.zeros(B, N)
    prev_mask[:, :K] = 1.0

    adjusted = lock_in.apply_winner_decay(scores_orig, prev_mask, decay_factor=0.5)
    _check(
        "winners decayed to 0.5",
        torch.allclose(adjusted[:, :K], torch.full((B, K), 0.5), atol=1e-5),
        f"got winners={adjusted[0, :K]}",
    )
    _check(
        "non-winners unchanged at 1.0",
        torch.allclose(adjusted[:, K:], torch.ones(B, N - K), atol=1e-5),
        f"got losers={adjusted[0, K:]}",
    )

    # -------------------------------------------------------------------
    # Test 16: LockInPrevention — slot dropout only during training
    # -------------------------------------------------------------------
    print("\n--- Test 16: Slot dropout — training vs non-training ---")

    lock_in.train()
    slots_full = torch.ones(B, K, D)
    mask_full = torch.ones(B, K)

    # Run many times with high dropout to check some slots get dropped
    any_dropped = False
    for _ in range(20):
        dropped, dmask = lock_in.apply_slot_dropout(
            slots_full, mask_full, p=0.9, training=True,
        )
        if (dmask == 0.0).any().item():
            any_dropped = True
            break
    _check(
        "training: some slots are dropped (p=0.9)",
        any_dropped,
    )

    # Non-training: no dropout
    lock_in.train(False)
    dropped_notr, dmask_notr = lock_in.apply_slot_dropout(
        slots_full, mask_full, p=0.9, training=False,
    )
    _check(
        "non-training: no slots dropped",
        torch.allclose(dropped_notr, slots_full),
        f"got {(dmask_notr == 0).sum().item()} dropped slots",
    )

    # -------------------------------------------------------------------
    # Test 17: LockInPrevention — at least one slot survives dropout
    # -------------------------------------------------------------------
    print("\n--- Test 17: Slot dropout — at least one survives ---")

    lock_in.train()
    survived_all = True
    for _ in range(50):
        _, dmask_safe = lock_in.apply_slot_dropout(
            slots_full, mask_full, p=0.99, training=True,
        )
        if (dmask_safe.sum(dim=-1) == 0).any().item():
            survived_all = False
            break
    _check(
        "high dropout: at least 1 slot always survives",
        survived_all,
    )

    # -------------------------------------------------------------------
    # Test 18: LockInPrevention — cooldown after ignition
    # -------------------------------------------------------------------
    print("\n--- Test 18: Cooldown mechanism ---")

    cfg_cd = IgnitionConfig(
        ignition_threshold=0.3, cooldown_steps=3, cooldown_boost=0.5,
    )
    lock_cd = LockInPrevention(cfg_cd)

    # No ignition history means base threshold
    thr_no_hist = lock_cd.check_cooldown([])
    _check(
        "no history: threshold == 0.3",
        abs(thr_no_hist.item() - 0.3) < 1e-5,
        f"got {thr_no_hist.item():.4f}",
    )

    # Ignition 1 step ago means boosted
    thr_recent = lock_cd.check_cooldown([False, False, True])
    _check(
        "recent ignition: threshold == 0.8",
        abs(thr_recent.item() - 0.8) < 1e-5,
        f"got {thr_recent.item():.4f}",
    )

    # Ignition 4 steps ago (beyond cooldown window of 3) means base
    thr_old = lock_cd.check_cooldown([True, False, False, False])
    _check(
        "old ignition: threshold == 0.3",
        abs(thr_old.item() - 0.3) < 1e-5,
        f"got {thr_old.item():.4f}",
    )

    # -------------------------------------------------------------------
    # Test 19: All component scores are in fp32
    # -------------------------------------------------------------------
    print("\n--- Test 19: fp32 enforcement ---")

    # Create fp16 inputs
    slots_fp16 = torch.randn(B, K, D, dtype=torch.float16)
    scores_fp16 = torch.randn(B, N, dtype=torch.float16)

    # Stability
    w_mask = torch.zeros(B, N)
    w_mask[:, :K] = 1.0
    stab = StabilityComputer.compute_winner_stability(w_mask, w_mask)
    _check(
        "stability dtype == float32",
        stab.dtype == torch.float32,
        f"got {stab.dtype}",
    )

    # Embedding stability with fp16 input
    embed_s = StabilityComputer.compute_embedding_stability(slots_fp16, slots_fp16)
    _check(
        "embedding stability from fp16 input: dtype == float32",
        embed_s.dtype == torch.float32,
        f"got {embed_s.dtype}",
    )

    # Confidence
    conf = ConfidenceComputer.compute_margin(scores_fp16, K)
    _check(
        "confidence margin from fp16 input: dtype == float32",
        conf.dtype == torch.float32,
        f"got {conf.dtype}",
    )

    # Coherence
    cons = CoherenceComputer.compute_mutual_consistency(slots_fp16)
    _check(
        "mutual consistency from fp16 input: dtype == float32",
        cons.dtype == torch.float32,
        f"got {cons.dtype}",
    )

    # -------------------------------------------------------------------
    # Test 20: IgnitionConfig validation
    # -------------------------------------------------------------------
    print("\n--- Test 20: IgnitionConfig validation ---")

    # Valid config
    try:
        _ = IgnitionConfig()
        _check("default config: valid", True)
    except Exception as e:
        _check("default config: valid", False, str(e))

    # Invalid threshold
    caught = False
    try:
        _ = IgnitionConfig(ignition_threshold=1.5)
    except ValueError:
        caught = True
    _check("threshold=1.5: raises ValueError", caught)

    caught = False
    try:
        _ = IgnitionConfig(ignition_threshold=0.0)
    except ValueError:
        caught = True
    _check("threshold=0.0: raises ValueError", caught)

    # Invalid gate_temperature
    caught = False
    try:
        _ = IgnitionConfig(gate_temperature=-0.1)
    except ValueError:
        caught = True
    _check("gate_temperature=-0.1: raises ValueError", caught)

    # Invalid negative weight
    caught = False
    try:
        _ = IgnitionConfig(w_stability=-0.1)
    except ValueError:
        caught = True
    _check("w_stability=-0.1: raises ValueError", caught)

    # -------------------------------------------------------------------
    # Test 21: IgnitionModule — full forward pass
    # -------------------------------------------------------------------
    print("\n--- Test 21: IgnitionModule — full forward pass ---")

    cfg_full = IgnitionConfig(ignition_threshold=0.3)
    module = IgnitionModule(cfg_full, workspace_dim=D)
    module.train()

    test_slots_mod = torch.randn(B, K, D)
    test_all_scores = torch.randn(B, N)
    test_metadata = {
        "modality_ids": torch.randint(0, 3, (B, K)),
        "winner_mask": torch.zeros(B, N),
        "K": K,
    }
    test_metadata["winner_mask"][:, :K] = 1.0

    test_round_history = {
        "prev_slots": torch.randn(B, K, D),
        "prev_winner_mask": torch.zeros(B, N),
    }
    test_round_history["prev_winner_mask"][:, :K] = 1.0

    result = module(
        slots=test_slots_mod,
        all_scores=test_all_scores,
        winners_metadata=test_metadata,
        round_history=test_round_history,
    )

    _check(
        "result.ignition_score shape == (B,)",
        result.ignition_score.shape == (B,),
        f"got {result.ignition_score.shape}",
    )
    _check(
        "result.ignited shape == (B,)",
        result.ignited.shape == (B,),
        f"got {result.ignited.shape}",
    )
    _check(
        "result.ignited dtype == bool",
        result.ignited.dtype == torch.bool,
        f"got {result.ignited.dtype}",
    )
    _check(
        "result.effective_gain shape == (B,)",
        result.effective_gain.shape == (B,),
        f"got {result.effective_gain.shape}",
    )
    _check(
        "result.gated_slots shape == (B, K, D)",
        result.gated_slots.shape == (B, K, D),
        f"got {result.gated_slots.shape}",
    )
    _check(
        "result.cooldown_active shape == (B,)",
        result.cooldown_active.shape == (B,),
        f"got {result.cooldown_active.shape}",
    )

    # -------------------------------------------------------------------
    # Test 22: IgnitionModule — no round_history (first round)
    # -------------------------------------------------------------------
    print("\n--- Test 22: IgnitionModule — first round (no history) ---")

    result_first = module(
        slots=test_slots_mod,
        all_scores=test_all_scores,
        winners_metadata=test_metadata,
        round_history=None,
    )
    _check(
        "first round: stability component == 0.0 (no history)",
        torch.allclose(
            result_first.components.stability,
            torch.zeros(B),
            atol=1e-5,
        ),
        f"got {result_first.components.stability}",
    )

    # -------------------------------------------------------------------
    # Test 23: IgnitionModule — cooldown integration
    # -------------------------------------------------------------------
    print("\n--- Test 23: IgnitionModule — cooldown integration ---")

    cfg_cooldown = IgnitionConfig(
        ignition_threshold=0.3, cooldown_steps=2, cooldown_boost=0.5,
    )
    module_cd = IgnitionModule(cfg_cooldown, workspace_dim=D)
    module_cd.train(False)

    # With recent ignition in history
    prev_state_cd = {
        "ignition_history": [True],
    }
    result_cd = module_cd(
        slots=test_slots_mod,
        all_scores=test_all_scores,
        winners_metadata=test_metadata,
        round_history=test_round_history,
        prev_state=prev_state_cd,
    )
    _check(
        "cooldown active after recent ignition",
        result_cd.cooldown_active.all().item(),
        f"got {result_cd.cooldown_active}",
    )

    # Without ignition history means no cooldown
    result_no_cd = module_cd(
        slots=test_slots_mod,
        all_scores=test_all_scores,
        winners_metadata=test_metadata,
        round_history=test_round_history,
        prev_state={},
    )
    _check(
        "no cooldown without ignition history",
        not result_no_cd.cooldown_active.any().item(),
        f"got {result_no_cd.cooldown_active}",
    )

    # -------------------------------------------------------------------
    # Test 24: compute_ignition_components_from_rounds utility
    # -------------------------------------------------------------------
    print("\n--- Test 24: compute_ignition_components_from_rounds ---")

    round_slots = [torch.randn(B, K, D), torch.randn(B, K, D)]
    round_scores = [torch.randn(B, N), torch.randn(B, N)]
    round_masks = [torch.zeros(B, N), torch.zeros(B, N)]
    round_masks[0][:, :K] = 1.0
    round_masks[1][:, :K] = 1.0
    mod_ids = torch.randint(0, 3, (B, K))

    comps, meta = compute_ignition_components_from_rounds(
        round_slots, round_scores, round_masks, mod_ids, K,
    )
    _check(
        "utility: stability shape == (B,)",
        comps.stability.shape == (B,),
        f"got {comps.stability.shape}",
    )
    _check(
        "utility: confidence shape == (B,)",
        comps.confidence.shape == (B,),
        f"got {comps.confidence.shape}",
    )
    _check(
        "utility: margin shape == (B,)",
        comps.margin.shape == (B,),
        f"got {comps.margin.shape}",
    )
    _check(
        "utility: coherence shape == (B,)",
        comps.coherence.shape == (B,),
        f"got {comps.coherence.shape}",
    )
    _check(
        "utility: metadata has K",
        meta.get("K") == K,
        f"got K={meta.get('K')}",
    )

    # -------------------------------------------------------------------
    # Test 25: IgnitionGate — gradient flow through smooth gate
    # -------------------------------------------------------------------
    print("\n--- Test 25: Gradient flow through IgnitionGate ---")

    gate_grad = IgnitionGate(IgnitionConfig(
        ignition_threshold=0.5, gate_temperature=0.1, weak_gain=0.0,
    ))
    gate_grad.train()

    slots_grad = torch.randn(B, K, D, requires_grad=True)
    score_grad = torch.tensor([0.6, 0.4, 0.7, 0.3], requires_grad=True)

    gated, gain = gate_grad(slots_grad, score_grad)
    loss_gate = gated.sum()
    loss_gate.backward()

    _check(
        "gradient flows to slots through gate",
        slots_grad.grad is not None and slots_grad.grad.abs().sum().item() > 0,
    )
    _check(
        "gradient flows to score through gate",
        score_grad.grad is not None and score_grad.grad.abs().sum().item() > 0,
    )

    # -------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------
    total = passed + failed
    print("\n" + "=" * 70)
    print(f"  Results: {passed}/{total} tests passed")
    if failed > 0:
        print(f"  FAILED:  {failed} test(s)")
    else:
        print("  All tests PASSED.")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
