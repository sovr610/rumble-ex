"""
Metacognitive routing module for the dual-process reasoning system.

Determines whether a given batch item should be handled entirely by System 1
(fast path) or escalated to System 2 (slow, iterative deliberation) based on
a deterministic linear combination of four signals:

    route_score = w_conf * (1 - conf)
               + w_novelty * novelty
               + w_anomaly * anomaly
               - w_budget  * budget_penalty

When ``route_score >= route_threshold`` the item is sent to System 2 with a
per-item step budget proportional to the score.  A hard skip gate
(``conf >= min_conf_to_skip_s2``) forces S1 regardless of other signals.

Brain analog: anterior cingulate cortex (conflict monitoring), basal ganglia
(action selection / routing), neuromodulatory arousal (novelty and anomaly
signals).

Design constraints:
  - All routing arithmetic in fp32 -- no sampling, no nondeterministic ops.
  - Deterministic: same inputs always yield the same routing decision.
  - Batch-independent: reordering the batch does not change per-item decisions.
  - Compatible with System1Result from system1_template.py, plain dicts, and
    raw confidence Tensors.

Dependencies: torch (no external packages).
Python: >= 3.9

Typical usage::

    from metacognition_template import (
        MetacognitionConfig,
        MetacognitiveRouter,
        NoveltyScorerFactory,
        BudgetTracker,
        split_batch_by_routing,
        merge_routed_results,
    )

    cfg = MetacognitionConfig()
    cfg.validate()
    router = MetacognitiveRouter(cfg)
    scorer = NoveltyScorerFactory.create(cfg)

    # Inside forward loop:
    novelty = scorer(hidden)
    decision = router(s1_result, novelty=novelty)
    s1_items, s2_items, s1_idx, s2_idx = split_batch_by_routing(x, decision)
"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# SECTION 1: Configuration
# ============================================================================


VALID_NOVELTY_METHODS: Tuple[str, ...] = (
    "prototype",
    "htm_anomaly",
    "engram_miss",
    "none",
)


@dataclass
class MetacognitionConfig:
    """Configuration for the metacognitive routing module.

    Controls the routing decision between System 1 (fast) and System 2
    (slow iterative) processing paths.

    Attributes:
        route_threshold: Score threshold above which items are sent to S2.
        w_conf: Weight for the confidence-deficit term ``(1 - conf)``.
        w_novelty: Weight for the novelty signal.
        w_anomaly: Weight for the anomaly signal (e.g. from HTM).
        w_budget: Weight for the remaining-budget penalty.
        min_conf_to_skip_s2: Hard confidence gate -- items at or above
            this confidence are forced to S1 regardless of other signals.
        base_steps: Minimum number of S2 deliberation steps assigned
            when an item is routed to System 2.
        step_scale_alpha: Scaling coefficient that maps route_score to
            additional S2 steps beyond ``base_steps``.
        max_steps: Hard ceiling on S2 steps.
        always_run_s2: Debug flag -- force all items through S2.
        novelty_method: Which novelty scoring strategy to use.
            One of ``"prototype"``, ``"htm_anomaly"``, ``"engram_miss"``,
            ``"none"``.
        num_prototypes: Number of prototype vectors for the prototype
            novelty scorer.
        prototype_dim: Dimensionality of each prototype vector.
        prototype_ema_decay: Exponential moving average decay rate for
            updating prototypes toward incoming representations.
    """

    # --- routing threshold ---
    route_threshold: float = 0.5

    # --- score weights ---
    w_conf: float = 1.0
    w_novelty: float = 0.5
    w_anomaly: float = 0.3
    w_budget: float = 0.1

    # --- shortcut ---
    min_conf_to_skip_s2: float = 0.95

    # --- adaptive step budgeting ---
    base_steps: int = 3
    step_scale_alpha: float = 5.0
    max_steps: int = 10

    # --- override ---
    always_run_s2: bool = False

    # --- novelty detection ---
    novelty_method: str = "prototype"
    num_prototypes: int = 64
    prototype_dim: int = 512
    prototype_ema_decay: float = 0.99

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate all configuration fields.

        Returns a list of error/warning strings.  An empty list means the
        configuration is valid.
        """
        errors: List[str] = []

        # route_threshold in [0, 1]
        if not isinstance(self.route_threshold, (int, float)):
            errors.append(
                f"route_threshold must be a float, got {type(self.route_threshold).__name__}"
            )
        elif not (0.0 <= self.route_threshold <= 1.0):
            errors.append(
                f"route_threshold must be in [0, 1], got {self.route_threshold}"
            )

        # min_conf_to_skip_s2 in [0, 1]
        if not isinstance(self.min_conf_to_skip_s2, (int, float)):
            errors.append(
                f"min_conf_to_skip_s2 must be a float, got "
                f"{type(self.min_conf_to_skip_s2).__name__}"
            )
        elif not (0.0 <= self.min_conf_to_skip_s2 <= 1.0):
            errors.append(
                f"min_conf_to_skip_s2 must be in [0, 1], got "
                f"{self.min_conf_to_skip_s2}"
            )

        # Non-negative weights
        for attr_name in ("w_conf", "w_novelty", "w_anomaly", "w_budget"):
            val = getattr(self, attr_name)
            if not isinstance(val, (int, float)):
                errors.append(f"{attr_name} must be numeric, got {type(val).__name__}")
            elif val < 0.0:
                errors.append(f"{attr_name} must be >= 0.0, got {val}")

        # Positive integers
        if not isinstance(self.base_steps, int) or self.base_steps < 1:
            errors.append(f"base_steps must be a positive integer, got {self.base_steps}")
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            errors.append(f"max_steps must be a positive integer, got {self.max_steps}")

        # step_scale_alpha non-negative
        if not isinstance(self.step_scale_alpha, (int, float)):
            errors.append(
                f"step_scale_alpha must be numeric, got "
                f"{type(self.step_scale_alpha).__name__}"
            )
        elif self.step_scale_alpha < 0.0:
            errors.append(f"step_scale_alpha must be >= 0.0, got {self.step_scale_alpha}")

        # novelty_method
        if self.novelty_method not in VALID_NOVELTY_METHODS:
            errors.append(
                f"novelty_method must be one of {VALID_NOVELTY_METHODS}, "
                f"got {self.novelty_method!r}"
            )

        # Prototype parameters
        if not isinstance(self.num_prototypes, int) or self.num_prototypes < 1:
            errors.append(
                f"num_prototypes must be a positive integer, got {self.num_prototypes}"
            )
        if not isinstance(self.prototype_dim, int) or self.prototype_dim < 1:
            errors.append(
                f"prototype_dim must be a positive integer, got {self.prototype_dim}"
            )
        if not isinstance(self.prototype_ema_decay, (int, float)):
            errors.append(
                f"prototype_ema_decay must be numeric, got "
                f"{type(self.prototype_ema_decay).__name__}"
            )
        elif not (0.0 < self.prototype_ema_decay <= 1.0):
            errors.append(
                f"prototype_ema_decay must be in (0, 1], got {self.prototype_ema_decay}"
            )

        # base_steps should not exceed max_steps
        if (
            isinstance(self.base_steps, int)
            and isinstance(self.max_steps, int)
            and self.base_steps > self.max_steps
        ):
            errors.append(
                f"base_steps ({self.base_steps}) must be <= max_steps ({self.max_steps})"
            )

        # Budget overflow warning
        if (
            isinstance(self.step_scale_alpha, (int, float))
            and isinstance(self.base_steps, int)
            and isinstance(self.max_steps, int)
        ):
            effective_max = self.step_scale_alpha * 1.0 + self.base_steps
            if effective_max > self.max_steps:
                errors.append(
                    f"Budget overflow: step_scale_alpha * 1.0 + base_steps = "
                    f"{effective_max:.1f} > max_steps = {self.max_steps}. "
                    f"Consider increasing max_steps or decreasing step_scale_alpha."
                )

        return errors


# ============================================================================
# SECTION 2: Routing decision container
# ============================================================================


@dataclass
class RoutingDecision:
    """Output of the metacognitive routing decision.

    All tensors have the batch dimension as their first axis.

    Attributes:
        used_system2: Boolean mask of shape ``(B,)`` indicating which items
            are routed to System 2.
        route_score: Float tensor of shape ``(B,)`` -- the computed routing
            score.  Higher values indicate greater need for System 2.
        steps_budget: Integer tensor of shape ``(B,)`` -- the number of
            System 2 deliberation steps allocated.  Zero for S1-only items.
        novelty: Float tensor of shape ``(B,)`` -- the novelty signal used
            in the routing computation.
        debug: Dictionary of additional debug information.
    """

    used_system2: Tensor   # (B,) bool
    route_score: Tensor    # (B,) float32
    steps_budget: Tensor   # (B,) int64
    novelty: Tensor        # (B,) float32
    debug: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# SECTION 3: MetacognitiveRouter
# ============================================================================


class MetacognitiveRouter(nn.Module):
    """Deterministic router that decides whether each batch item should be
    handled by System 1 (fast) or System 2 (slow iterative refinement).

    The routing score is a linear combination of four signals::

        score = w_conf * (1 - conf)
              + w_novelty * novelty
              + w_anomaly * anomaly
              - w_budget  * budget_penalty

    Items with ``score >= route_threshold`` are routed to System 2.

    A hard skip gate overrides the score: if ``conf >= min_conf_to_skip_s2``
    the item is forced to System 1 regardless of other signals.

    The ``always_run_s2`` debug flag does the opposite: all items are forced
    to System 2.

    All arithmetic is performed in fp32.  No sampling or nondeterministic
    operations are used -- identical inputs always produce identical outputs.

    Args:
        config: :class:`MetacognitionConfig` controlling thresholds,
            weights, and step budget parameters.
    """

    def __init__(self, config: MetacognitionConfig) -> None:
        super().__init__()
        self.config: MetacognitionConfig = config

        # Store weights as plain Python floats -- no learnable parameters
        # in the router itself (routing is a deterministic policy).
        self._w_conf: float = float(config.w_conf)
        self._w_novelty: float = float(config.w_novelty)
        self._w_anomaly: float = float(config.w_anomaly)
        self._w_budget: float = float(config.w_budget)
        self._route_threshold: float = float(config.route_threshold)
        self._min_conf_to_skip_s2: float = float(config.min_conf_to_skip_s2)
        self._always_run_s2: bool = bool(config.always_run_s2)
        self._base_steps: int = int(config.base_steps)
        self._step_scale_alpha: float = float(config.step_scale_alpha)
        self._max_steps: int = int(config.max_steps)

    # ------------------------------------------------------------------
    # Confidence extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_confidence(s1_result: Any) -> Tensor:
        """Extract a ``(B,)`` confidence tensor from various input types.

        Supports:
          - Object with ``.conf_calibrated`` attribute (System1Result)
          - Dict with ``"confidence"`` key
          - Plain ``Tensor`` of shape ``(B,)`` or ``(B, 1)``

        Args:
            s1_result: System 1 output in any of the supported formats.

        Returns:
            Confidence tensor of shape ``(B,)`` in fp32.

        Raises:
            TypeError: If the input format is not recognized.
        """
        if hasattr(s1_result, "conf_calibrated"):
            conf = s1_result.conf_calibrated
        elif isinstance(s1_result, dict) and "confidence" in s1_result:
            conf = s1_result["confidence"]
        elif isinstance(s1_result, Tensor):
            conf = s1_result
        else:
            raise TypeError(
                f"Cannot extract confidence from {type(s1_result).__name__}. "
                f"Expected a System1Result-like object with .conf_calibrated, "
                f"a dict with 'confidence' key, or a plain Tensor."
            )

        # Ensure (B,) shape and fp32
        conf = conf.float()
        if conf.ndim == 2 and conf.shape[-1] == 1:
            conf = conf.squeeze(-1)
        if conf.ndim == 0:
            conf = conf.unsqueeze(0)
        return conf

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        s1_result: Any,
        novelty: Optional[Tensor] = None,
        anomaly: Optional[Tensor] = None,
        ignition: Optional[Tensor] = None,
        remaining_budget: Optional[float] = None,
    ) -> RoutingDecision:
        """Compute the routing decision for a batch.

        Args:
            s1_result: System 1 output.  Accepted formats:
                - Object with ``.conf_calibrated`` tensor ``(B,)``
                - Dict with ``"confidence"`` tensor ``(B,)``
                - Plain ``Tensor`` of shape ``(B,)``
            novelty: Optional novelty signal ``(B,)`` in ``[0, 1]``.
                If ``None``, defaults to zeros.
            anomaly: Optional anomaly signal ``(B,)`` in ``[0, 1]``.
                If ``None``, defaults to zeros.
            ignition: Reserved for future global workspace ignition
                signal.  Currently unused.
            remaining_budget: Optional remaining compute budget as a
                fraction in ``[0, 1]``.  When low, the budget penalty
                term discourages routing to System 2.

        Returns:
            :class:`RoutingDecision` containing the boolean routing mask,
            float routing scores, integer step budgets, and debug info.
        """
        # -- Extract and validate confidence --
        conf: Tensor = self._extract_confidence(s1_result)
        B: int = conf.shape[0]
        device: torch.device = conf.device

        # -- Default signals --
        if novelty is None:
            novelty_t: Tensor = torch.zeros(B, dtype=torch.float32, device=device)
        else:
            novelty_t = novelty.float()

        if anomaly is None:
            anomaly_t: Tensor = torch.zeros(B, dtype=torch.float32, device=device)
        else:
            anomaly_t = anomaly.float()

        # -- Budget penalty --
        if remaining_budget is not None:
            # Penalty increases as budget decreases: penalty = 1 - remaining
            budget_penalty: Tensor = torch.full(
                (B,), 1.0 - float(remaining_budget),
                dtype=torch.float32, device=device,
            )
        else:
            budget_penalty = torch.zeros(B, dtype=torch.float32, device=device)

        # -- Compute route score (fp32) --
        score: Tensor = (
            self._w_conf * (1.0 - conf)
            + self._w_novelty * novelty_t
            + self._w_anomaly * anomaly_t
            - self._w_budget * budget_penalty
        )

        # -- Hard skip gate: high-confidence items forced to S1 --
        hard_skip: Tensor = conf >= self._min_conf_to_skip_s2  # (B,) bool

        # -- Routing decision --
        if self._always_run_s2:
            used_s2: Tensor = torch.ones(B, dtype=torch.bool, device=device)
        else:
            used_s2 = (score >= self._route_threshold) & (~hard_skip)

        # -- Compute per-item step budget --
        # steps = clamp(round(base_steps + alpha * score), 1, max_steps)
        raw_steps: Tensor = self._base_steps + self._step_scale_alpha * score
        steps_budget: Tensor = raw_steps.round().long().clamp(min=1, max=self._max_steps)

        # Zero out steps for items staying in S1
        steps_budget = torch.where(used_s2, steps_budget, torch.zeros_like(steps_budget))

        # -- Debug info --
        debug: Dict[str, Any] = {
            "conf": conf.detach(),
            "novelty": novelty_t.detach(),
            "anomaly": anomaly_t.detach(),
            "budget_penalty": budget_penalty.detach(),
            "hard_skip": hard_skip.detach(),
            "raw_steps": raw_steps.detach(),
        }

        return RoutingDecision(
            used_system2=used_s2,
            route_score=score,
            steps_budget=steps_budget,
            novelty=novelty_t,
            debug=debug,
        )

    def extra_repr(self) -> str:
        return (
            f"threshold={self._route_threshold}, "
            f"w_conf={self._w_conf}, w_novelty={self._w_novelty}, "
            f"w_anomaly={self._w_anomaly}, w_budget={self._w_budget}, "
            f"min_conf_skip={self._min_conf_to_skip_s2}, "
            f"base_steps={self._base_steps}, alpha={self._step_scale_alpha}, "
            f"max_steps={self._max_steps}, always_s2={self._always_run_s2}"
        )


# ============================================================================
# SECTION 4: Novelty scoring base class and implementations
# ============================================================================


class NoveltyScorer(nn.Module):
    """Abstract base class for novelty scoring modules.

    A novelty scorer takes a batch of feature representations and produces
    a per-item novelty signal in ``[0, 1]`` where higher values indicate
    greater novelty (more distant from known prototypes / patterns).

    Subclasses must implement :meth:`forward` and optionally
    :meth:`update_prototypes`.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Compute per-item novelty scores.

        Args:
            x: Input features of shape ``(B, D)``.

        Returns:
            Novelty scores of shape ``(B,)`` in ``[0, 1]``.
        """
        raise NotImplementedError

    def update_prototypes(self, x: Tensor) -> None:
        """Update internal prototypes / state from new data.

        Default implementation is a no-op.

        Args:
            x: Input features of shape ``(B, D)``.
        """
        pass


class PrototypeNoveltyScorer(NoveltyScorer):
    """Novelty scorer based on minimum distance to learned prototypes.

    Maintains ``K`` prototype vectors in a buffer (non-learnable, updated
    via exponential moving average).  Novelty for a given input is the
    minimum L2 distance to any prototype, normalized to ``[0, 1]`` via
    a sigmoid transformation.

    The normalization uses::

        novelty = sigmoid(alpha * (min_dist - median_dist))

    where ``alpha`` is a sensitivity parameter and ``median_dist`` is
    a running estimate of the median minimum distance.

    Args:
        num_prototypes: Number of prototype vectors ``K``.
        prototype_dim: Dimensionality ``D`` of each prototype.
        ema_decay: Exponential moving average decay for prototype updates.
        sigma: Sensitivity scaling for the sigmoid normalization.
    """

    def __init__(
        self,
        num_prototypes: int = 64,
        prototype_dim: int = 512,
        ema_decay: float = 0.99,
        sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_prototypes: int = num_prototypes
        self.prototype_dim: int = prototype_dim
        self.ema_decay: float = ema_decay
        self.sigma: float = sigma

        # Prototype memory (not learnable -- updated via EMA)
        self.register_buffer(
            "prototypes",
            torch.randn(num_prototypes, prototype_dim),
        )
        # Running median distance estimate for normalization
        self.register_buffer(
            "running_median_dist",
            torch.tensor(1.0, dtype=torch.float32),
        )
        # Track whether prototypes have been initialized from real data
        self.register_buffer(
            "_initialized",
            torch.tensor(False, dtype=torch.bool),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute novelty as normalized min-distance to prototypes.

        Args:
            x: Input features ``(B, D)`` where ``D == prototype_dim``.

        Returns:
            Novelty scores ``(B,)`` in ``[0, 1]``.
        """
        x_f: Tensor = x.float()  # (B, D)
        protos: Tensor = self.prototypes.float()  # (K, D)

        # Pairwise L2 distances: (B, K)
        # ||x - p||^2 = ||x||^2 + ||p||^2 - 2 x p^T
        x_sq: Tensor = (x_f * x_f).sum(dim=-1, keepdim=True)  # (B, 1)
        p_sq: Tensor = (protos * protos).sum(dim=-1, keepdim=True).t()  # (1, K)
        cross: Tensor = torch.mm(x_f, protos.t())  # (B, K)
        dists_sq: Tensor = (x_sq + p_sq - 2.0 * cross).clamp(min=0.0)
        dists: Tensor = dists_sq.sqrt()  # (B, K)

        # Minimum distance to any prototype
        min_dist: Tensor = dists.min(dim=-1).values  # (B,)

        # Normalize via sigmoid centered on running median
        median: float = float(self.running_median_dist.item())
        novelty: Tensor = torch.sigmoid(
            self.sigma * (min_dist - median)
        )  # (B,) in [0, 1]

        return novelty

    @torch.no_grad()
    def update_prototypes(self, x: Tensor) -> None:
        """Update prototypes toward the incoming batch via EMA.

        Each input sample is assigned to its nearest prototype, and
        that prototype is updated::

            proto_k = decay * proto_k + (1 - decay) * x_assigned

        Also updates the running median distance estimate.

        Args:
            x: Input features ``(B, D)``.
        """
        x_f: Tensor = x.float()
        protos: Tensor = self.prototypes.float()

        # Initialize prototypes from the first batch if not yet done
        if not self._initialized.item():
            n_init: int = min(x_f.shape[0], self.num_prototypes)
            self.prototypes[:n_init] = x_f[:n_init].clone()
            self._initialized.fill_(True)
            return

        # Pairwise distances
        x_sq: Tensor = (x_f * x_f).sum(dim=-1, keepdim=True)
        p_sq: Tensor = (protos * protos).sum(dim=-1, keepdim=True).t()
        cross: Tensor = torch.mm(x_f, protos.t())
        dists_sq: Tensor = (x_sq + p_sq - 2.0 * cross).clamp(min=0.0)
        dists: Tensor = dists_sq.sqrt()

        # Assign each input to its nearest prototype
        min_dists, assignments = dists.min(dim=-1)  # (B,), (B,)

        # Update running median distance
        if min_dists.numel() > 0:
            new_median: Tensor = min_dists.median()
            self.running_median_dist.mul_(self.ema_decay).add_(
                new_median * (1.0 - self.ema_decay)
            )

        # EMA update: for each prototype, average the assigned samples
        for k in range(self.num_prototypes):
            mask: Tensor = assignments == k
            if mask.any():
                centroid: Tensor = x_f[mask].mean(dim=0)
                self.prototypes[k] = (
                    self.ema_decay * self.prototypes[k]
                    + (1.0 - self.ema_decay) * centroid
                )


class HTMAnomalyProxy(NoveltyScorer):
    """Pass-through novelty scorer for HTM anomaly signals.

    Assumes the anomaly signal is already computed by the HTM temporal
    layer and is in ``[0, 1]``.  This module normalizes raw values
    via clamping and optional sigmoid rescaling.

    Args:
        scale: Scaling factor applied before sigmoid.  Set to ``0.0``
            to skip sigmoid and use raw clamped values.
    """

    def __init__(self, scale: float = 0.0) -> None:
        super().__init__()
        self.scale: float = scale

    def forward(self, x: Tensor) -> Tensor:
        """Normalize and return the HTM anomaly signal.

        Args:
            x: Raw anomaly scores ``(B,)`` or ``(B, 1)``.

        Returns:
            Normalized anomaly in ``[0, 1]`` shape ``(B,)``.
        """
        x_f: Tensor = x.float()
        if x_f.ndim == 2 and x_f.shape[-1] == 1:
            x_f = x_f.squeeze(-1)
        if x_f.ndim == 0:
            x_f = x_f.unsqueeze(0)

        if self.scale > 0.0:
            return torch.sigmoid(self.scale * x_f)
        else:
            return x_f.clamp(0.0, 1.0)


class EngramMissProxy(NoveltyScorer):
    """Pass-through novelty scorer for engram memory miss rate.

    The engram memory layer reports how many of its N-gram hash lookups
    resulted in a miss.  This module normalizes the miss rate to
    ``[0, 1]``.

    Args:
        scale: Scaling factor applied before sigmoid.  Set to ``0.0``
            to skip sigmoid and use raw clamped values.
    """

    def __init__(self, scale: float = 0.0) -> None:
        super().__init__()
        self.scale: float = scale

    def forward(self, x: Tensor) -> Tensor:
        """Normalize and return the engram miss rate.

        Args:
            x: Raw miss rates ``(B,)`` or ``(B, 1)``.

        Returns:
            Normalized miss rate in ``[0, 1]`` shape ``(B,)``.
        """
        x_f: Tensor = x.float()
        if x_f.ndim == 2 and x_f.shape[-1] == 1:
            x_f = x_f.squeeze(-1)
        if x_f.ndim == 0:
            x_f = x_f.unsqueeze(0)

        if self.scale > 0.0:
            return torch.sigmoid(self.scale * x_f)
        else:
            return x_f.clamp(0.0, 1.0)


class NullNoveltyScorer(NoveltyScorer):
    """Novelty scorer that always returns zeros.

    Used as a fallback when novelty detection is disabled
    (``novelty_method="none"``).
    """

    def forward(self, x: Tensor) -> Tensor:
        """Return zero novelty for all items.

        Args:
            x: Input features ``(B, D)`` or ``(B,)``.

        Returns:
            Zeros of shape ``(B,)``.
        """
        B: int = x.shape[0]
        return torch.zeros(B, dtype=torch.float32, device=x.device)


# ============================================================================
# SECTION 5: Novelty scorer factory
# ============================================================================


class NoveltyScorerFactory:
    """Factory that creates the appropriate :class:`NoveltyScorer` from
    a :class:`MetacognitionConfig`.

    Dispatches on ``config.novelty_method``:

    * ``"prototype"`` -> :class:`PrototypeNoveltyScorer`
    * ``"htm_anomaly"`` -> :class:`HTMAnomalyProxy`
    * ``"engram_miss"`` -> :class:`EngramMissProxy`
    * ``"none"`` -> :class:`NullNoveltyScorer`
    """

    @staticmethod
    def create(config: MetacognitionConfig) -> NoveltyScorer:
        """Create a novelty scorer based on the config.

        Args:
            config: Configuration specifying novelty method and parameters.

        Returns:
            An initialized :class:`NoveltyScorer` instance.

        Raises:
            ValueError: If ``config.novelty_method`` is not recognized.
        """
        method: str = config.novelty_method

        if method == "prototype":
            return PrototypeNoveltyScorer(
                num_prototypes=config.num_prototypes,
                prototype_dim=config.prototype_dim,
                ema_decay=config.prototype_ema_decay,
            )
        elif method == "htm_anomaly":
            return HTMAnomalyProxy()
        elif method == "engram_miss":
            return EngramMissProxy()
        elif method == "none":
            return NullNoveltyScorer()
        else:
            raise ValueError(
                f"Unknown novelty_method {method!r}. "
                f"Expected one of {VALID_NOVELTY_METHODS}."
            )


# ============================================================================
# SECTION 6: Budget tracker
# ============================================================================


class BudgetTracker:
    """Tracks the total and remaining compute budget for System 2
    deliberation across a sequence of batches.

    Usage::

        tracker = BudgetTracker(total_budget=100)
        # ... inside forward loop ...
        tracker.allocate(decision.steps_budget)  # deducts allocated steps
        penalty = tracker.budget_penalty  # rising as budget depletes
        tracker.reset()  # at the start of each epoch

    Attributes:
        total_budget: Total number of S2 steps available for the sequence.
        remaining: Remaining budget.
    """

    def __init__(self, total_budget: int) -> None:
        if total_budget < 0:
            raise ValueError(f"total_budget must be >= 0, got {total_budget}")
        self.total_budget: int = total_budget
        self.remaining: int = total_budget

    def allocate(self, steps_budget: Tensor) -> Tensor:
        """Allocate steps from the remaining budget.

        Items whose requested budget exceeds the remaining total are
        clamped.  The remaining budget is decremented accordingly.

        Args:
            steps_budget: Per-item step budgets ``(B,)`` int64.

        Returns:
            Actual allocated steps ``(B,)`` int64 (possibly clamped).
        """
        requested: int = int(steps_budget.sum().item())

        if requested <= self.remaining:
            self.remaining -= requested
            return steps_budget
        else:
            # Distribute remaining budget proportionally, rounding down
            if requested > 0:
                scale: float = self.remaining / max(requested, 1)
            else:
                scale = 0.0
            allocated: Tensor = (steps_budget.float() * scale).floor().long()
            allocated = allocated.clamp(min=0)
            actual_total: int = int(allocated.sum().item())
            self.remaining = max(0, self.remaining - actual_total)
            return allocated

    def reset(self) -> None:
        """Reset the remaining budget to the total."""
        self.remaining = self.total_budget

    @property
    def budget_penalty(self) -> float:
        """Return the budget penalty in ``[0, 1]``.

        ``0.0`` when full budget remains, ``1.0`` when budget is exhausted.
        """
        if self.total_budget == 0:
            return 1.0
        return 1.0 - (self.remaining / self.total_budget)

    @property
    def fraction_remaining(self) -> float:
        """Return the fraction of budget remaining in ``[0, 1]``."""
        if self.total_budget == 0:
            return 0.0
        return self.remaining / self.total_budget

    def __repr__(self) -> str:
        return (
            f"BudgetTracker(total={self.total_budget}, "
            f"remaining={self.remaining}, "
            f"penalty={self.budget_penalty:.3f})"
        )


# ============================================================================
# SECTION 7: Routing metrics accumulator
# ============================================================================


class RoutingMetricsAccumulator:
    """Accumulates routing statistics over multiple batches for logging
    and monitoring.

    Tracks:
    - ``s2_fraction``: fraction of items routed to System 2
    - ``mean_route_score``: average routing score across all items
    - ``mean_steps_budget``: average S2 step budget (S2 items only)
    - ``mean_novelty``: average novelty signal
    - ``hard_skip_fraction``: fraction of items that hit the hard skip gate
    - ``total_items``: total number of items processed

    Usage::

        acc = RoutingMetricsAccumulator()
        for batch in dataloader:
            decision = router(s1_result)
            acc.update(decision)
        metrics = acc.compute()
        acc.reset()
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        self._total_items: int = 0
        self._total_s2: int = 0
        self._sum_route_score: float = 0.0
        self._sum_steps_budget: float = 0.0
        self._s2_items_for_steps: int = 0
        self._sum_novelty: float = 0.0
        self._total_hard_skip: int = 0

    def update(self, decision: RoutingDecision) -> None:
        """Accumulate statistics from a single routing decision.

        Args:
            decision: Routing decision from :class:`MetacognitiveRouter`.
        """
        B: int = decision.used_system2.shape[0]
        n_s2: int = int(decision.used_system2.sum().item())

        self._total_items += B
        self._total_s2 += n_s2
        self._sum_route_score += float(decision.route_score.sum().item())
        self._sum_novelty += float(decision.novelty.sum().item())

        # Steps budget for S2 items only
        if n_s2 > 0:
            s2_mask = decision.used_system2
            s2_steps = decision.steps_budget[s2_mask]
            self._sum_steps_budget += float(s2_steps.sum().item())
            self._s2_items_for_steps += n_s2

        # Hard skip count
        if "hard_skip" in decision.debug:
            self._total_hard_skip += int(decision.debug["hard_skip"].sum().item())

    def compute(self) -> Dict[str, float]:
        """Compute aggregate metrics.

        Returns:
            Dictionary with the following keys:
            - ``s2_fraction``: fraction of items routed to S2
            - ``mean_route_score``: average routing score
            - ``mean_steps_budget``: average S2 steps (S2 items only)
            - ``mean_novelty``: average novelty
            - ``hard_skip_fraction``: fraction hitting hard skip
            - ``total_items``: total items processed
        """
        total: int = max(self._total_items, 1)
        s2_total: int = max(self._s2_items_for_steps, 1)

        return {
            "s2_fraction": self._total_s2 / total,
            "mean_route_score": self._sum_route_score / total,
            "mean_steps_budget": self._sum_steps_budget / s2_total,
            "mean_novelty": self._sum_novelty / total,
            "hard_skip_fraction": self._total_hard_skip / total,
            "total_items": float(self._total_items),
        }

    def __repr__(self) -> str:
        m = self.compute()
        return (
            f"RoutingMetrics(s2_frac={m['s2_fraction']:.3f}, "
            f"mean_score={m['mean_route_score']:.3f}, "
            f"mean_steps={m['mean_steps_budget']:.1f}, "
            f"items={int(m['total_items'])})"
        )


# ============================================================================
# SECTION 8: Batch splitting and merging utilities
# ============================================================================


def split_batch_by_routing(
    data: Tensor,
    decision: RoutingDecision,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split a batch into S1-only and S2-routed sub-batches.

    This is used to scatter data before running System 2 only on the
    items that need it, avoiding wasted computation.

    Args:
        data: Full batch tensor of shape ``(B, ...)``.
        decision: Routing decision from :class:`MetacognitiveRouter`.

    Returns:
        Tuple of:
        - ``s1_items``: Tensor of items handled by S1 only ``(N1, ...)``.
        - ``s2_items``: Tensor of items routed to S2 ``(N2, ...)``.
        - ``s1_idx``: Index tensor ``(N1,)`` mapping back to original batch.
        - ``s2_idx``: Index tensor ``(N2,)`` mapping back to original batch.

    Note:
        ``N1 + N2 == B``.  Either ``N1`` or ``N2`` may be zero.
    """
    s2_mask: Tensor = decision.used_system2  # (B,) bool
    s1_mask: Tensor = ~s2_mask

    s1_idx: Tensor = s1_mask.nonzero(as_tuple=False).view(-1)
    s2_idx: Tensor = s2_mask.nonzero(as_tuple=False).view(-1)

    s1_items: Tensor = data[s1_idx] if s1_idx.numel() > 0 else data[:0]
    s2_items: Tensor = data[s2_idx] if s2_idx.numel() > 0 else data[:0]

    return s1_items, s2_items, s1_idx, s2_idx


def merge_routed_results(
    s1_out: Tensor,
    s2_out: Tensor,
    s1_idx: Tensor,
    s2_idx: Tensor,
    B: int,
) -> Tensor:
    """Merge System 1 and System 2 outputs back into a full batch.

    This is the gather operation complementary to
    :func:`split_batch_by_routing`.

    Args:
        s1_out: System 1 outputs ``(N1, D)`` or ``(N1, ...)``.
        s2_out: System 2 outputs ``(N2, D)`` or ``(N2, ...)``.
        s1_idx: Original batch indices for S1 items ``(N1,)``.
        s2_idx: Original batch indices for S2 items ``(N2,)``.
        B: Original batch size.

    Returns:
        Merged tensor ``(B, D)`` with S1 and S2 results in their
        original positions.
    """
    # Determine output shape
    if s1_out.numel() > 0:
        trailing_shape = s1_out.shape[1:]
        dtype = s1_out.dtype
        device = s1_out.device
    elif s2_out.numel() > 0:
        trailing_shape = s2_out.shape[1:]
        dtype = s2_out.dtype
        device = s2_out.device
    else:
        raise ValueError("Both s1_out and s2_out are empty -- cannot determine shape.")

    merged: Tensor = torch.zeros(
        (B,) + trailing_shape, dtype=dtype, device=device,
    )

    if s1_idx.numel() > 0:
        merged[s1_idx] = s1_out
    if s2_idx.numel() > 0:
        merged[s2_idx] = s2_out

    return merged


# ============================================================================
# SECTION 9: Self-tests
# ============================================================================


def _run_self_tests() -> None:
    """Run comprehensive self-tests and print results.

    Tests cover:
      1. Confident skip: conf=0.99 -> S1
      2. Uncertain route: conf=0.3 -> S2
      3. Novelty trigger: conf=0.9, novelty=0.9 -> S2
      4. always_run_s2 forces all to S2
      5. Steps scaling: lower conf -> more steps
      6. Determinism: 10 runs produce identical results
      7. Batch independence: reorder -> same per-item decisions
      8. Budget tracker: allocate until exhausted
      9. All novelty methods instantiate and produce valid output
     10. Prototype EMA update changes prototypes
     11. Hard skip overrides high novelty
     12. Edge cases: batch size 1, all confident, all uncertain
     13. Config validation catches errors
    """
    torch.manual_seed(42)
    device: torch.device = torch.device("cpu")

    passed: int = 0
    total: int = 13
    failures: List[str] = []

    def _report(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed
        status: str = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failures.append(f"{name}: {detail}")
        msg: str = f"  [{status}] {name}"
        if detail and not ok:
            msg += f" -- {detail}"
        print(msg)

    print("=" * 70)
    print("Metacognitive Router -- Self Tests")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Helper: default config
    # ------------------------------------------------------------------
    def _default_cfg() -> MetacognitionConfig:
        return MetacognitionConfig(
            route_threshold=0.5,
            w_conf=1.0,
            w_novelty=0.5,
            w_anomaly=0.3,
            w_budget=0.1,
            min_conf_to_skip_s2=0.95,
            base_steps=3,
            step_scale_alpha=5.0,
            max_steps=10,
            always_run_s2=False,
            novelty_method="prototype",
            num_prototypes=8,
            prototype_dim=32,
            prototype_ema_decay=0.99,
        )

    # ==================================================================
    # Test 1: Confident skip -- conf=0.99 -> forced S1
    # ==================================================================
    try:
        cfg = _default_cfg()
        router = MetacognitiveRouter(cfg)

        conf = torch.tensor([0.99], dtype=torch.float32, device=device)
        decision = router(conf)

        ok = (
            not decision.used_system2[0].item()
            and decision.steps_budget[0].item() == 0
        )
        _report(
            "1. Confident skip (conf=0.99 -> S1)",
            ok,
            f"used_s2={decision.used_system2.tolist()}, "
            f"steps={decision.steps_budget.tolist()}",
        )
    except Exception as e:
        _report("1. Confident skip", False, str(e))

    # ==================================================================
    # Test 2: Uncertain route -- conf=0.3 -> S2
    # ==================================================================
    try:
        cfg = _default_cfg()
        router = MetacognitiveRouter(cfg)

        conf = torch.tensor([0.3], dtype=torch.float32, device=device)
        decision = router(conf)

        # score = 1.0*(1-0.3) + 0 + 0 - 0 = 0.7 >= 0.5 threshold
        ok = (
            decision.used_system2[0].item()
            and decision.steps_budget[0].item() > 0
        )
        _report(
            "2. Uncertain route (conf=0.3 -> S2)",
            ok,
            f"used_s2={decision.used_system2.tolist()}, "
            f"score={decision.route_score.tolist()}, "
            f"steps={decision.steps_budget.tolist()}",
        )
    except Exception as e:
        _report("2. Uncertain route", False, str(e))

    # ==================================================================
    # Test 3: Novelty trigger -- conf=0.9, novelty=0.9 -> S2
    # ==================================================================
    try:
        cfg = _default_cfg()
        router = MetacognitiveRouter(cfg)

        conf = torch.tensor([0.9], dtype=torch.float32, device=device)
        novelty = torch.tensor([0.9], dtype=torch.float32, device=device)
        decision = router(conf, novelty=novelty)

        # score = 1.0*(1-0.9) + 0.5*0.9 + 0 - 0 = 0.1 + 0.45 = 0.55 >= 0.5
        # conf=0.9 < 0.95 so no hard skip
        ok = decision.used_system2[0].item()
        _report(
            "3. Novelty trigger (conf=0.9, nov=0.9 -> S2)",
            ok,
            f"used_s2={decision.used_system2.tolist()}, "
            f"score={decision.route_score.tolist()}",
        )
    except Exception as e:
        _report("3. Novelty trigger", False, str(e))

    # ==================================================================
    # Test 4: always_run_s2 forces all to S2
    # ==================================================================
    try:
        cfg = _default_cfg()
        cfg.always_run_s2 = True
        router = MetacognitiveRouter(cfg)

        # Even very confident items should be routed to S2
        conf = torch.tensor([0.99, 0.99, 0.99], dtype=torch.float32, device=device)
        decision = router(conf)

        ok = decision.used_system2.all().item()
        _report(
            "4. always_run_s2 forces all to S2",
            ok,
            f"used_s2={decision.used_system2.tolist()}",
        )
    except Exception as e:
        _report("4. always_run_s2", False, str(e))

    # ==================================================================
    # Test 5: Steps scaling -- lower conf -> more steps
    # ==================================================================
    try:
        cfg = _default_cfg()
        router = MetacognitiveRouter(cfg)

        # Two items: one low conf, one moderate
        conf = torch.tensor([0.1, 0.5], dtype=torch.float32, device=device)
        decision = router(conf)

        # Both should be routed to S2
        # Lower conf -> higher route_score -> more steps
        steps = decision.steps_budget
        if decision.used_system2.all().item():
            ok = steps[0].item() >= steps[1].item()
        else:
            ok = False

        _report(
            "5. Steps scaling (lower conf -> more steps)",
            ok,
            f"conf={conf.tolist()}, steps={steps.tolist()}, "
            f"score={decision.route_score.tolist()}",
        )
    except Exception as e:
        _report("5. Steps scaling", False, str(e))

    # ==================================================================
    # Test 6: Determinism -- 10 runs produce identical results
    # ==================================================================
    try:
        cfg = _default_cfg()
        router = MetacognitiveRouter(cfg)

        conf = torch.tensor([0.2, 0.5, 0.8, 0.99], dtype=torch.float32, device=device)
        novelty = torch.tensor([0.3, 0.7, 0.1, 0.0], dtype=torch.float32, device=device)
        anomaly = torch.tensor([0.1, 0.2, 0.3, 0.0], dtype=torch.float32, device=device)

        ref_decision = router(conf, novelty=novelty, anomaly=anomaly)
        all_match: bool = True

        for _ in range(10):
            d = router(conf, novelty=novelty, anomaly=anomaly)
            if not torch.equal(d.used_system2, ref_decision.used_system2):
                all_match = False
                break
            if not torch.allclose(d.route_score, ref_decision.route_score, atol=1e-7):
                all_match = False
                break
            if not torch.equal(d.steps_budget, ref_decision.steps_budget):
                all_match = False
                break

        _report("6. Determinism (10 runs identical)", all_match)
    except Exception as e:
        _report("6. Determinism", False, str(e))

    # ==================================================================
    # Test 7: Batch independence -- reorder -> same per-item decisions
    # ==================================================================
    try:
        cfg = _default_cfg()
        router = MetacognitiveRouter(cfg)

        conf = torch.tensor([0.2, 0.5, 0.8, 0.99], dtype=torch.float32, device=device)
        novelty = torch.tensor([0.3, 0.7, 0.1, 0.0], dtype=torch.float32, device=device)

        d_original = router(conf, novelty=novelty)

        # Reverse the batch order
        perm = torch.tensor([3, 2, 1, 0], dtype=torch.long, device=device)
        conf_perm = conf[perm]
        novelty_perm = novelty[perm]

        d_perm = router(conf_perm, novelty=novelty_perm)

        # Undo permutation and compare
        inv_perm = torch.tensor([3, 2, 1, 0], dtype=torch.long, device=device)
        ok = (
            torch.equal(d_original.used_system2, d_perm.used_system2[inv_perm])
            and torch.allclose(
                d_original.route_score, d_perm.route_score[inv_perm], atol=1e-7
            )
            and torch.equal(d_original.steps_budget, d_perm.steps_budget[inv_perm])
        )

        _report("7. Batch independence (reorder -> same decisions)", ok)
    except Exception as e:
        _report("7. Batch independence", False, str(e))

    # ==================================================================
    # Test 8: Budget tracker -- allocate until exhausted
    # ==================================================================
    try:
        tracker = BudgetTracker(total_budget=20)
        ok_initial: bool = (
            tracker.remaining == 20
            and abs(tracker.budget_penalty - 0.0) < 1e-6
            and abs(tracker.fraction_remaining - 1.0) < 1e-6
        )

        # First allocation: 10 steps
        alloc1 = tracker.allocate(torch.tensor([5, 5], dtype=torch.long))
        ok_after_1: bool = (
            tracker.remaining == 10
            and int(alloc1.sum().item()) == 10
        )

        # Second allocation: request 15 (exceeds remaining 10)
        alloc2 = tracker.allocate(torch.tensor([8, 7], dtype=torch.long))
        ok_after_2: bool = (
            tracker.remaining >= 0
            and int(alloc2.sum().item()) <= 10
        )

        # After exhaustion, penalty should be high
        ok_penalty: bool = tracker.budget_penalty > 0.0

        # Reset
        tracker.reset()
        ok_reset: bool = tracker.remaining == 20

        ok = ok_initial and ok_after_1 and ok_after_2 and ok_penalty and ok_reset
        _report(
            "8. Budget tracker (allocate until exhausted)",
            ok,
            f"initial={ok_initial}, after1={ok_after_1}, "
            f"after2={ok_after_2}, penalty={ok_penalty}, reset={ok_reset}",
        )
    except Exception as e:
        _report("8. Budget tracker", False, str(e))

    # ==================================================================
    # Test 9: All novelty methods instantiate and produce valid output
    # ==================================================================
    try:
        all_methods_ok: bool = True
        details: List[str] = []

        for method in VALID_NOVELTY_METHODS:
            cfg_nm = _default_cfg()
            cfg_nm.novelty_method = method
            scorer = NoveltyScorerFactory.create(cfg_nm)

            if method == "prototype":
                x = torch.randn(4, 32, device=device)
            else:
                x = torch.rand(4, device=device)

            out = scorer(x)
            valid: bool = (
                out.shape == (4,)
                and not torch.isnan(out).any().item()
                and (out >= 0.0).all().item()
                and (out <= 1.0).all().item()
            )
            details.append(f"{method}: valid={valid}, shape={out.shape}")
            if not valid:
                all_methods_ok = False

        _report(
            "9. All novelty methods produce valid output",
            all_methods_ok,
            "; ".join(details),
        )
    except Exception as e:
        _report("9. All novelty methods", False, str(e))

    # ==================================================================
    # Test 10: Prototype EMA update changes prototypes
    # ==================================================================
    try:
        scorer_p = PrototypeNoveltyScorer(
            num_prototypes=4, prototype_dim=8, ema_decay=0.9,
        )

        # Initialize prototypes
        x_init = torch.randn(4, 8, device=device)
        scorer_p.update_prototypes(x_init)
        protos_before = scorer_p.prototypes.clone()

        # Update with new data
        x_new = torch.randn(8, 8, device=device) * 5.0  # distant points
        scorer_p.update_prototypes(x_new)
        protos_after = scorer_p.prototypes.clone()

        # Prototypes should have changed
        ok = not torch.allclose(protos_before, protos_after, atol=1e-6)
        _report(
            "10. Prototype EMA update changes prototypes",
            ok,
            f"max_diff={float((protos_before - protos_after).abs().max().item()):.6f}",
        )
    except Exception as e:
        _report("10. Prototype EMA update", False, str(e))

    # ==================================================================
    # Test 11: Hard skip overrides high novelty
    # ==================================================================
    try:
        cfg = _default_cfg()
        router = MetacognitiveRouter(cfg)

        # Very high confidence (above skip threshold) with very high novelty
        conf = torch.tensor([0.96], dtype=torch.float32, device=device)
        novelty = torch.tensor([1.0], dtype=torch.float32, device=device)
        anomaly = torch.tensor([1.0], dtype=torch.float32, device=device)

        decision = router(conf, novelty=novelty, anomaly=anomaly)

        # Hard skip should force S1 even though novelty and anomaly are max
        ok = not decision.used_system2[0].item()
        _report(
            "11. Hard skip overrides high novelty",
            ok,
            f"conf={conf.tolist()}, used_s2={decision.used_system2.tolist()}, "
            f"score={decision.route_score.tolist()}",
        )
    except Exception as e:
        _report("11. Hard skip overrides high novelty", False, str(e))

    # ==================================================================
    # Test 12: Edge cases -- B=1, all confident, all uncertain
    # ==================================================================
    try:
        cfg = _default_cfg()
        router = MetacognitiveRouter(cfg)

        # Sub-test A: Batch size 1
        conf_b1 = torch.tensor([0.5], dtype=torch.float32, device=device)
        d_b1 = router(conf_b1)
        b1_ok: bool = (
            d_b1.used_system2.shape == (1,)
            and d_b1.route_score.shape == (1,)
            and d_b1.steps_budget.shape == (1,)
        )

        # Sub-test B: All confident (above skip threshold)
        conf_all_high = torch.full((8,), 0.99, dtype=torch.float32, device=device)
        d_all_high = router(conf_all_high)
        all_high_ok: bool = not d_all_high.used_system2.any().item()

        # Sub-test C: All uncertain
        conf_all_low = torch.full((8,), 0.1, dtype=torch.float32, device=device)
        d_all_low = router(conf_all_low)
        all_low_ok: bool = d_all_low.used_system2.all().item()

        # Sub-test D: Dict input
        d_dict = router({"confidence": torch.tensor([0.5], device=device)})
        dict_ok: bool = d_dict.route_score.shape == (1,)

        # Sub-test E: Object with conf_calibrated
        class _FakeResult:
            conf_calibrated = torch.tensor([0.5], dtype=torch.float32, device=device)

        d_obj = router(_FakeResult())
        obj_ok: bool = d_obj.route_score.shape == (1,)

        ok = b1_ok and all_high_ok and all_low_ok and dict_ok and obj_ok
        _report(
            "12. Edge cases (B=1, all conf, all uncert, dict, obj)",
            ok,
            f"b1={b1_ok}, all_high={all_high_ok}, all_low={all_low_ok}, "
            f"dict={dict_ok}, obj={obj_ok}",
        )
    except Exception as e:
        _report("12. Edge cases", False, str(e))

    # ==================================================================
    # Test 13: Config validation catches errors
    # ==================================================================
    try:
        # Valid config should have no errors
        valid_cfg = _default_cfg()
        valid_errors = valid_cfg.validate()
        no_errors_on_valid: bool = len(valid_errors) == 0

        # Invalid configs
        bad_cfg1 = _default_cfg()
        bad_cfg1.route_threshold = -0.5
        errs1 = bad_cfg1.validate()
        catches_bad_threshold: bool = len(errs1) > 0 and any(
            "route_threshold" in e for e in errs1
        )

        bad_cfg2 = _default_cfg()
        bad_cfg2.w_conf = -1.0
        errs2 = bad_cfg2.validate()
        catches_neg_weight: bool = len(errs2) > 0 and any(
            "w_conf" in e for e in errs2
        )

        bad_cfg3 = _default_cfg()
        bad_cfg3.novelty_method = "nonexistent"
        errs3 = bad_cfg3.validate()
        catches_bad_method: bool = len(errs3) > 0 and any(
            "novelty_method" in e for e in errs3
        )

        bad_cfg4 = _default_cfg()
        bad_cfg4.max_steps = -5
        errs4 = bad_cfg4.validate()
        catches_bad_steps: bool = len(errs4) > 0 and any(
            "max_steps" in e for e in errs4
        )

        bad_cfg5 = _default_cfg()
        bad_cfg5.prototype_ema_decay = 1.5
        errs5 = bad_cfg5.validate()
        catches_bad_decay: bool = len(errs5) > 0 and any(
            "prototype_ema_decay" in e for e in errs5
        )

        ok = (
            no_errors_on_valid
            and catches_bad_threshold
            and catches_neg_weight
            and catches_bad_method
            and catches_bad_steps
            and catches_bad_decay
        )
        _report(
            "13. Config validation catches errors",
            ok,
            f"valid={no_errors_on_valid}, threshold={catches_bad_threshold}, "
            f"weight={catches_neg_weight}, method={catches_bad_method}, "
            f"steps={catches_bad_steps}, decay={catches_bad_decay}",
        )
    except Exception as e:
        _report("13. Config validation", False, str(e))

    # ==================================================================
    # Additional tests: split/merge and metrics accumulator
    # ==================================================================
    # These are bonus tests beyond the 13 core tests.
    print("\n  --- Bonus tests ---")

    # Bonus A: split_batch_by_routing and merge_routed_results
    try:
        cfg = _default_cfg()
        router = MetacognitiveRouter(cfg)

        B_test: int = 6
        data = torch.randn(B_test, 16, device=device)
        conf = torch.tensor(
            [0.99, 0.1, 0.99, 0.2, 0.99, 0.3],
            dtype=torch.float32, device=device,
        )
        decision = router(conf)

        s1_items, s2_items, s1_idx, s2_idx = split_batch_by_routing(data, decision)
        bonus_a_split: bool = (
            s1_items.shape[0] + s2_items.shape[0] == B_test
            and s1_idx.shape[0] + s2_idx.shape[0] == B_test
        )

        # Create fake outputs
        s1_out = torch.ones(s1_idx.shape[0], 16, device=device) * 1.0
        s2_out = torch.ones(s2_idx.shape[0], 16, device=device) * 2.0
        merged = merge_routed_results(s1_out, s2_out, s1_idx, s2_idx, B_test)
        bonus_a_merge: bool = merged.shape == (B_test, 16)

        # Check correctness: S1 items should be 1.0, S2 items should be 2.0
        for i in range(B_test):
            if i in s1_idx.tolist():
                if not torch.allclose(merged[i], torch.tensor(1.0)):
                    bonus_a_merge = False
            else:
                if not torch.allclose(merged[i], torch.tensor(2.0)):
                    bonus_a_merge = False

        ok_bonus_a: bool = bonus_a_split and bonus_a_merge
        _report(
            "Bonus A: split/merge batch routing",
            ok_bonus_a,
            f"split={bonus_a_split}, merge={bonus_a_merge}",
        )
    except Exception as e:
        _report("Bonus A: split/merge", False, str(e))

    # Bonus B: RoutingMetricsAccumulator
    try:
        acc = RoutingMetricsAccumulator()
        cfg = _default_cfg()
        router = MetacognitiveRouter(cfg)

        for _ in range(5):
            conf = torch.rand(8, device=device)
            d = router(conf)
            acc.update(d)

        metrics = acc.compute()
        ok_bonus_b: bool = (
            "s2_fraction" in metrics
            and "mean_route_score" in metrics
            and "mean_steps_budget" in metrics
            and "mean_novelty" in metrics
            and "hard_skip_fraction" in metrics
            and "total_items" in metrics
            and metrics["total_items"] == 40.0
            and 0.0 <= metrics["s2_fraction"] <= 1.0
        )

        acc.reset()
        metrics_after_reset = acc.compute()
        ok_bonus_b = ok_bonus_b and metrics_after_reset["total_items"] == 0.0

        _report(
            "Bonus B: RoutingMetricsAccumulator",
            ok_bonus_b,
            f"metrics={metrics}",
        )
    except Exception as e:
        _report("Bonus B: RoutingMetricsAccumulator", False, str(e))

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"{passed}/{total} self-tests passed")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
    print("=" * 70)

    if passed < total:
        sys.exit(1)


# ============================================================================
# Entry point
# ============================================================================


if __name__ == "__main__":
    _run_self_tests()
