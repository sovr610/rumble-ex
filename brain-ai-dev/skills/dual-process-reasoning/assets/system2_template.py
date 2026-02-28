"""
System 2 Iterative Refinement Module
=====================================

GRU-based iterative refinement loop implementing "slow thinking" in the
dual-process reasoning framework. System 2 receives the fast System 1
output (y1) and refines it through multiple deliberation steps until
convergence or budget exhaustion.

Brain Analog: Prefrontal cortex deliberative reasoning -- engages when
System 1 confidence is low (< 0.7 threshold) or task complexity is high.

Architecture:
    SummaryNet  --- compresses (x_summary, context) -> h0
    RefinementGRU - iteratively refines hidden state
    OutputProjection - projects hidden state -> logits
    ConvergenceChecker - decides when to halt per item

Forward Loop:
    y_k = y1
    h_k = SummaryNet(x_summary, context)
    for k in 1..budget:
        h_k = GRU(h_k, [y_k, x_summary])
        y_new = OutputProjection(h_k)
        if converged(y_new, y_k): halt
        y_k = y_new
    return y_k

Dependencies: torch (no external packages)
Python: >= 3.9
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class System2Config:
    """Configuration for the System 2 iterative refinement module.

    Attributes:
        hidden_dim: Dimensionality of the GRU hidden state.
        output_dim: Dimensionality of the refined output logits.
        input_summary_dim: Dimensionality of the input summary vector
            produced upstream (e.g. by the global workspace).
        max_steps: Hard ceiling on deliberation steps regardless of
            convergence.  Acts as the upper bound when no explicit
            ``steps_budget`` is supplied to ``forward()``.
        convergence_eps: Epsilon threshold for convergence checks.
            Interpretation depends on ``convergence_criterion``.
        convergence_patience: Number of *consecutive* stable steps
            required before declaring convergence.  Prevents premature
            halting on a single lucky step.
        convergence_criterion: One of ``"kl"``, ``"logit"``, or
            ``"argmax"``.  Selects the convergence test used in the
            inner loop.
        nan_guard: When ``True`` any NaN in the output triggers an
            immediate halt for that batch item and the last valid
            output is preserved.
        gradient_clip_per_step: Maximum gradient norm applied via backward
            hooks on the GRU hidden state at each refinement step during
            training.  Set to ``0.0`` to disable.
        deep_supervision: When ``True`` the forward pass collects
            per-step outputs and returns auxiliary losses that can be
            used for training.
        deep_supervision_discount: Geometric discount factor applied
            to per-step losses: ``loss_k * discount^(max_steps - k)``.
    """

    hidden_dim: int = 512
    output_dim: int = 256
    input_summary_dim: int = 512
    max_steps: int = 10
    convergence_eps: float = 1e-3
    convergence_patience: int = 2
    convergence_criterion: str = "kl"  # "kl", "logit", "argmax"
    nan_guard: bool = True
    gradient_clip_per_step: float = 1.0
    deep_supervision: bool = False
    deep_supervision_discount: float = 0.9

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if self.input_summary_dim <= 0:
            raise ValueError(
                f"input_summary_dim must be positive, got {self.input_summary_dim}"
            )
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.convergence_eps < 0.0:
            raise ValueError(
                f"convergence_eps must be >= 0.0, got {self.convergence_eps}"
            )
        if self.convergence_patience < 1:
            raise ValueError(
                f"convergence_patience must be >= 1, got {self.convergence_patience}"
            )
        if self.convergence_criterion not in ("kl", "logit", "argmax"):
            raise ValueError(
                f"convergence_criterion must be 'kl', 'logit', "
                f"or 'argmax', got '{self.convergence_criterion}'"
            )
        if self.gradient_clip_per_step < 0.0:
            raise ValueError(
                f"gradient_clip_per_step must be >= 0.0, got "
                f"{self.gradient_clip_per_step}"
            )
        if not 0.0 < self.deep_supervision_discount <= 1.0:
            raise ValueError(
                f"deep_supervision_discount must be in (0, 1], got "
                f"{self.deep_supervision_discount}"
            )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class System2Result:
    """Output of the System 2 iterative refinement forward pass.

    Attributes:
        y2: Final refined output tensor of shape ``(B, output_dim)``.
        steps_used: Integer tensor of shape ``(B,)`` indicating how many
            refinement steps each batch item underwent.
        converged: Boolean tensor of shape ``(B,)`` indicating whether
            each item met the convergence criterion within budget.
        halt_reason: Per-item string describing why the loop terminated.
            Possible values: ``"converged"``, ``"max_steps"``,
            ``"budget_exhausted"``, ``"nan_guard"``.
        step_metrics: Optional list of per-step metric dictionaries.
            Each dict contains ``delta_kl``, ``delta_max``, ``conf_k``,
            and ``argmax_k`` tensors.  Only populated when the module
            is in evaluation mode or deep supervision is enabled.
        deep_supervision_outputs: Optional list of per-step output
            tensors for computing auxiliary losses during training.
        deep_supervision_weights: Optional tensor of discount weights
            for each collected step output.
    """

    y2: Tensor
    steps_used: Tensor
    converged: Tensor
    halt_reason: List[str]
    step_metrics: Optional[List[Dict[str, Tensor]]] = None
    deep_supervision_outputs: Optional[List[Tensor]] = None
    deep_supervision_weights: Optional[Tensor] = None


# ---------------------------------------------------------------------------
# SummaryNet -- compresses (x_summary, context) -> h0
# ---------------------------------------------------------------------------


class SummaryNet(nn.Module):
    """Transforms the input summary and optional context into the initial
    hidden state ``h0`` for the refinement GRU.

    Architecture::

        input = x_summary                             # (B, input_summary_dim)
        if context is not None:
            input = LayerNorm(Linear(cat(input, context)))
        h0 = tanh(Linear(LayerNorm(input)))            # (B, hidden_dim)

    The context pathway uses a separate projection so that the context
    dimensionality is not constrained.
    """

    def __init__(
        self,
        input_summary_dim: int,
        hidden_dim: int,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_summary_dim: int = input_summary_dim
        self.hidden_dim: int = hidden_dim
        self.context_dim: Optional[int] = context_dim

        # Optional context fusion pathway
        if context_dim is not None and context_dim > 0:
            self.context_proj = nn.Linear(
                input_summary_dim + context_dim, input_summary_dim
            )
            self.context_ln = nn.LayerNorm(input_summary_dim)
        else:
            self.context_proj = None
            self.context_ln = None

        # Main summary-to-hidden projection
        self.ln_input = nn.LayerNorm(input_summary_dim)
        self.proj = nn.Linear(input_summary_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x_summary: Tensor, context: Optional[Tensor] = None
    ) -> Tensor:
        """Compute the initial hidden state.

        Args:
            x_summary: Input summary of shape ``(B, input_summary_dim)``.
            context: Optional context tensor of shape ``(B, context_dim)``.

        Returns:
            h0 of shape ``(B, hidden_dim)``.
        """
        h: Tensor = x_summary  # (B, input_summary_dim)

        # Fuse context if provided and the pathway exists
        if context is not None and self.context_proj is not None:
            combined: Tensor = torch.cat([h, context], dim=-1)
            h = self.context_ln(self.context_proj(combined))  # type: ignore[arg-type]

        h = self.ln_input(h)
        h = torch.tanh(self.proj(h))  # (B, hidden_dim)
        return h


# ---------------------------------------------------------------------------
# RefinementGRU -- GRUCell wrapper for iterative refinement
# ---------------------------------------------------------------------------


class RefinementGRU(nn.Module):
    """GRUCell-based refinement block.

    At each deliberation step the cell receives:
        - ``h_prev``: previous hidden state ``(B, hidden_dim)``
        - ``cat(y_prev, x_summary)``: concatenation of the current output
          estimate and the input summary ``(B, output_dim + input_summary_dim)``

    And produces ``h_next`` of shape ``(B, hidden_dim)``.

    An optional pre-projection compresses the concatenated input to
    ``hidden_dim`` before feeding it to the GRUCell, which can reduce
    parameter count when ``output_dim + input_summary_dim`` is much
    larger than ``hidden_dim``.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        input_summary_dim: int,
        use_pre_projection: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim
        self.input_summary_dim: int = input_summary_dim

        cat_dim: int = output_dim + input_summary_dim

        self.use_pre_projection: bool = use_pre_projection
        if use_pre_projection:
            self.pre_proj = nn.Sequential(
                nn.Linear(cat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            gru_input_dim: int = hidden_dim
        else:
            self.pre_proj = None
            gru_input_dim = cat_dim

        self.gru_cell = nn.GRUCell(input_size=gru_input_dim, hidden_size=hidden_dim)
        self.ln_post = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal init for recurrent weights, xavier for input weights."""
        for name, param in self.gru_cell.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        if self.pre_proj is not None:
            for m in self.pre_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, h_prev: Tensor, y_prev: Tensor, x_summary: Tensor) -> Tensor:
        """One refinement step.

        Args:
            h_prev: Previous hidden state ``(B, hidden_dim)``.
            y_prev: Previous output estimate ``(B, output_dim)``.
            x_summary: Input summary ``(B, input_summary_dim)``.

        Returns:
            h_next of shape ``(B, hidden_dim)``.
        """
        cat_input: Tensor = torch.cat([y_prev, x_summary], dim=-1)

        if self.pre_proj is not None:
            cat_input = self.pre_proj(cat_input)

        h_next: Tensor = self.gru_cell(cat_input, h_prev)
        h_next = self.ln_post(h_next)
        return h_next


# ---------------------------------------------------------------------------
# OutputProjection -- h -> logits
# ---------------------------------------------------------------------------


class OutputProjection(nn.Module):
    """Projects the GRU hidden state to output logits.

    Architecture::

        h -> Linear(hidden_dim, hidden_dim) -> GELU -> LayerNorm
          -> Linear(hidden_dim, output_dim)
          -> output logits

    A two-layer MLP with GELU activation gives the projection enough
    capacity to produce well-separated logits from the hidden
    representation.
    """

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init with small final-layer weights for stable start."""
        for i, m in enumerate(self.net.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Scale down the final linear layer for a near-identity start
        final_linear: nn.Linear = self.net[-1]  # type: ignore[assignment]
        with torch.no_grad():
            final_linear.weight.mul_(0.1)

    def forward(self, h: Tensor) -> Tensor:
        """Project hidden state to output logits.

        Args:
            h: Hidden state of shape ``(B, hidden_dim)``.

        Returns:
            Logits of shape ``(B, output_dim)``.
        """
        return self.net(h)


# ---------------------------------------------------------------------------
# ConvergenceChecker -- stateless convergence criteria
# ---------------------------------------------------------------------------


class ConvergenceChecker(nn.Module):
    """Stateless convergence criteria for the iterative refinement loop.

    Supports three criteria:

    * **kl** -- ``KL(softmax(logits_k) || softmax(logits_prev)) < eps``
    * **logit** -- ``max|logits_k - logits_prev| < eps``
    * **argmax** -- ``argmax(logits_k) == argmax(logits_prev)``

    All checks are performed in **fp32** to avoid numerical issues with
    half-precision KL divergence.

    Patience is tracked externally by the caller (``System2Iterative``)
    because patience state is per-item and must persist across steps.
    """

    _VALID_CRITERIA: Tuple[str, ...] = ("kl", "logit", "argmax")

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _to_fp32(t: Tensor) -> Tensor:
        """Promote to fp32 for numerically stable convergence checks."""
        return t.float() if t.dtype != torch.float32 else t

    @staticmethod
    def _safe_log_softmax(logits: Tensor, dim: int = -1) -> Tensor:
        """Numerically stable log-softmax in fp32."""
        logits_f: Tensor = ConvergenceChecker._to_fp32(logits)
        return F.log_softmax(logits_f, dim=dim)

    @staticmethod
    def _safe_softmax(logits: Tensor, dim: int = -1) -> Tensor:
        """Numerically stable softmax in fp32."""
        logits_f: Tensor = ConvergenceChecker._to_fp32(logits)
        return F.softmax(logits_f, dim=dim)

    # -- Individual criteria ------------------------------------------------

    @staticmethod
    def check_kl(
        logits_k: Tensor, logits_prev: Tensor, eps: float
    ) -> Tensor:
        """KL divergence convergence check.

        Computes ``KL(p_k || p_prev)`` where ``p = softmax(logits)``.
        Returns a boolean tensor of shape ``(B,)`` that is ``True``
        when ``KL < eps``.

        Args:
            logits_k: Current-step logits ``(B, D)``.
            logits_prev: Previous-step logits ``(B, D)``.
            eps: Threshold for convergence.

        Returns:
            Boolean tensor ``(B,)``.
        """
        log_p_k: Tensor = ConvergenceChecker._safe_log_softmax(logits_k)
        p_k: Tensor = ConvergenceChecker._safe_softmax(logits_k)
        log_p_prev: Tensor = ConvergenceChecker._safe_log_softmax(logits_prev)

        # KL(p_k || p_prev) = sum p_k * (log p_k - log p_prev)
        kl: Tensor = (p_k * (log_p_k - log_p_prev)).sum(dim=-1)
        # Clamp to non-negative (numerical noise can produce tiny negatives)
        kl = kl.clamp(min=0.0)
        return kl < eps

    @staticmethod
    def check_logit(
        logits_k: Tensor, logits_prev: Tensor, eps: float
    ) -> Tensor:
        """Logit-space convergence check.

        Returns ``True`` per item when ``max|logits_k - logits_prev| < eps``.

        Args:
            logits_k: Current-step logits ``(B, D)``.
            logits_prev: Previous-step logits ``(B, D)``.
            eps: Threshold.

        Returns:
            Boolean tensor ``(B,)``.
        """
        logits_k_f: Tensor = ConvergenceChecker._to_fp32(logits_k)
        logits_prev_f: Tensor = ConvergenceChecker._to_fp32(logits_prev)
        delta_max: Tensor = (logits_k_f - logits_prev_f).abs().max(dim=-1).values
        return delta_max < eps

    @staticmethod
    def check_argmax(logits_k: Tensor, logits_prev: Tensor) -> Tensor:
        """Argmax stability convergence check.

        Returns ``True`` per item when the argmax class has not changed.

        Args:
            logits_k: Current-step logits ``(B, D)``.
            logits_prev: Previous-step logits ``(B, D)``.

        Returns:
            Boolean tensor ``(B,)``.
        """
        return logits_k.argmax(dim=-1) == logits_prev.argmax(dim=-1)

    # -- Unified dispatch ---------------------------------------------------

    @staticmethod
    def check(
        logits_k: Tensor,
        logits_prev: Tensor,
        criterion: str,
        eps: float,
    ) -> Tensor:
        """Dispatch to the appropriate convergence check.

        Args:
            logits_k: Current-step logits ``(B, D)``.
            logits_prev: Previous-step logits ``(B, D)``.
            criterion: One of ``"kl"``, ``"logit"``, ``"argmax"``.
            eps: Epsilon threshold (ignored for ``"argmax"``).

        Returns:
            Boolean tensor ``(B,)`` -- ``True`` means the item is
            considered stable at this step.

        Raises:
            ValueError: If *criterion* is not recognized.
        """
        if criterion == "kl":
            return ConvergenceChecker.check_kl(logits_k, logits_prev, eps)
        elif criterion == "logit":
            return ConvergenceChecker.check_logit(logits_k, logits_prev, eps)
        elif criterion == "argmax":
            return ConvergenceChecker.check_argmax(logits_k, logits_prev)
        else:
            raise ValueError(
                f"Unknown convergence criterion '{criterion}'. "
                f"Expected one of {ConvergenceChecker._VALID_CRITERIA}."
            )

    # -- Metric helpers (for logging / tracing) -----------------------------

    @staticmethod
    def compute_metrics(
        logits_k: Tensor, logits_prev: Tensor
    ) -> Dict[str, Tensor]:
        """Compute a suite of convergence-related metrics.

        All computations are in fp32.

        Returns a dict with:
            * ``delta_kl``: ``(B,)`` KL divergence between step k and prev.
            * ``delta_max``: ``(B,)`` Max absolute logit difference.
            * ``conf_k``: ``(B,)`` Max softmax probability at step k.
            * ``argmax_k``: ``(B,)`` Predicted class at step k.
        """
        logits_k_f: Tensor = ConvergenceChecker._to_fp32(logits_k)
        logits_prev_f: Tensor = ConvergenceChecker._to_fp32(logits_prev)

        log_p_k: Tensor = F.log_softmax(logits_k_f, dim=-1)
        p_k: Tensor = F.softmax(logits_k_f, dim=-1)
        log_p_prev: Tensor = F.log_softmax(logits_prev_f, dim=-1)

        kl: Tensor = (p_k * (log_p_k - log_p_prev)).sum(dim=-1).clamp(min=0.0)
        delta_max: Tensor = (logits_k_f - logits_prev_f).abs().max(dim=-1).values
        conf_k: Tensor = p_k.max(dim=-1).values
        argmax_k: Tensor = logits_k_f.argmax(dim=-1)

        return {
            "delta_kl": kl,
            "delta_max": delta_max,
            "conf_k": conf_k,
            "argmax_k": argmax_k,
        }


# ---------------------------------------------------------------------------
# Halt-reason priority logic
# ---------------------------------------------------------------------------


_HALT_PRIORITY: Dict[str, int] = {
    "nan_guard": 0,      # highest priority
    "converged": 1,
    "budget_exhausted": 2,
    "max_steps": 3,      # lowest priority
}


def _resolve_halt_reason(reasons: List[str]) -> str:
    """Given a list of candidate halt reasons for a single item,
    return the one with the highest priority (lowest number).

    Priority order: nan_guard > converged > budget_exhausted > max_steps.

    Args:
        reasons: Candidate halt reason strings for a batch item.

    Returns:
        The highest-priority reason.
    """
    if not reasons:
        return "max_steps"
    best: str = reasons[0]
    best_priority: int = _HALT_PRIORITY.get(best, 99)
    for r in reasons[1:]:
        p: int = _HALT_PRIORITY.get(r, 99)
        if p < best_priority:
            best = r
            best_priority = p
    return best


# ---------------------------------------------------------------------------
# Per-step gradient clipping hook
# ---------------------------------------------------------------------------


class _GradientClipHook:
    """Backward hook that clips the gradient norm flowing through a tensor.

    Intended to be registered on hidden-state tensors during the forward
    loop to implement per-step gradient clipping::

        h_k.register_hook(_GradientClipHook(max_norm=1.0))

    Attributes:
        max_norm: Maximum L2 norm for the gradient.
    """

    def __init__(self, max_norm: float) -> None:
        self.max_norm: float = max_norm

    def __call__(self, grad: Tensor) -> Tensor:
        norm: Tensor = grad.norm(2)
        if norm > self.max_norm:
            grad = grad * (self.max_norm / (norm + 1e-8))
        return grad


# ---------------------------------------------------------------------------
# System2Iterative -- main module
# ---------------------------------------------------------------------------


class System2Iterative(nn.Module):
    """GRU-based iterative refinement module implementing System 2
    "slow thinking" in the dual-process reasoning framework.

    Given a fast System 1 output ``y1`` and an input summary
    ``x_summary``, this module iteratively refines ``y1`` through a
    GRU-based loop until convergence criteria are met or the step
    budget is exhausted.

    The module handles batches where different items may converge at
    different steps.  Items that have halted are masked out of further
    computation for efficiency.

    Args:
        config: :class:`System2Config` instance controlling all
            hyperparameters.
        context_dim: Optional dimensionality of an additional context
            vector provided to :class:`SummaryNet`.

    Example::

        config = System2Config(hidden_dim=512, output_dim=256)
        s2 = System2Iterative(config)
        y1 = torch.randn(4, 256)
        x_summary = torch.randn(4, 512)
        result = s2(y1, x_summary, steps_budget=5)
        print(result.y2.shape)          # (4, 256)
        print(result.steps_used)        # tensor([...]) up to 5
        print(result.halt_reason)       # list of 4 strings
    """

    def __init__(
        self,
        config: System2Config,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config: System2Config = config
        self.context_dim: Optional[int] = context_dim

        # Sub-modules
        self.summary_net = SummaryNet(
            input_summary_dim=config.input_summary_dim,
            hidden_dim=config.hidden_dim,
            context_dim=context_dim,
        )
        self.refinement_gru = RefinementGRU(
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            input_summary_dim=config.input_summary_dim,
            use_pre_projection=True,
        )
        self.output_proj = OutputProjection(
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
        )
        self.convergence_checker = ConvergenceChecker()

        # Residual gate: learned scalar per output dim that blends
        # the System 1 output with the System 2 refinement.  This
        # allows the model to learn to "pass through" y1 when System 2
        # does not improve the answer.
        self.residual_gate = nn.Parameter(torch.zeros(config.output_dim))

        # Input projection for y1 if its dim does not match output_dim
        # (should match by design, but this provides a safety net)
        self._y1_proj: Optional[nn.Linear] = None

    def _ensure_y1_proj(self, y1_dim: int) -> None:
        """Lazily create a projection layer if y1 dim != output_dim."""
        if y1_dim != self.config.output_dim:
            if self._y1_proj is None or self._y1_proj.in_features != y1_dim:
                self._y1_proj = nn.Linear(y1_dim, self.config.output_dim).to(
                    device=self.residual_gate.device,
                    dtype=self.residual_gate.dtype,
                )
                nn.init.xavier_uniform_(self._y1_proj.weight)
                nn.init.zeros_(self._y1_proj.bias)

    def _project_y1(self, y1: Tensor) -> Tensor:
        """Ensure y1 has shape (B, output_dim)."""
        if y1.shape[-1] != self.config.output_dim:
            self._ensure_y1_proj(y1.shape[-1])
            assert self._y1_proj is not None
            return self._y1_proj(y1)
        return y1

    def _resolve_steps_budget(
        self,
        steps_budget: Optional[Union[int, Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[int, Tensor]:
        """Resolve the per-item step budgets.

        Returns:
            - global_max: integer upper bound for the loop range.
            - per_item_budget: ``(B,)`` long tensor of per-item budgets.
        """
        if steps_budget is None:
            global_max: int = self.config.max_steps
            per_item_budget: Tensor = torch.full(
                (batch_size,), global_max, dtype=torch.long, device=device
            )
        elif isinstance(steps_budget, int):
            global_max = min(steps_budget, self.config.max_steps)
            per_item_budget = torch.full(
                (batch_size,), global_max, dtype=torch.long, device=device
            )
        else:
            # Tensor of per-item budgets
            per_item_budget = steps_budget.long().to(device)
            per_item_budget = per_item_budget.clamp(min=1, max=self.config.max_steps)
            global_max = int(per_item_budget.max().item())

        return global_max, per_item_budget

    def forward(
        self,
        y1: Tensor,
        x_summary: Tensor,
        steps_budget: Optional[Union[int, Tensor]] = None,
        context: Optional[Tensor] = None,
    ) -> System2Result:
        """Run iterative refinement.

        Args:
            y1: System 1 output of shape ``(B, output_dim)`` (or
                ``(B, D)`` -- will be projected if ``D != output_dim``).
            x_summary: Input summary of shape ``(B, input_summary_dim)``.
            steps_budget: Maximum refinement steps.  Can be an integer
                (same for all items), a ``(B,)`` integer tensor (per-item),
                or ``None`` (uses ``config.max_steps``).
            context: Optional context tensor of shape ``(B, context_dim)``
                passed to :class:`SummaryNet` for initial hidden state
                conditioning.

        Returns:
            :class:`System2Result` containing the refined output and
            diagnostics.
        """
        # -- Setup ----------------------------------------------------------
        cfg: System2Config = self.config
        B: int = y1.shape[0]
        device: torch.device = y1.device
        dtype: torch.dtype = y1.dtype

        # Ensure y1 has the correct output dimensionality
        y1_proj: Tensor = self._project_y1(y1)

        # Resolve step budgets
        global_max, per_item_budget = self._resolve_steps_budget(
            steps_budget, B, device
        )

        # -- Initialize loop state ------------------------------------------
        y_k: Tensor = y1_proj.clone()                         # (B, output_dim)
        h_k: Tensor = self.summary_net(x_summary, context)    # (B, hidden_dim)

        # Per-step gradient clipping via backward hooks on hidden state
        if self.training and cfg.gradient_clip_per_step > 0.0:
            h_k.register_hook(_GradientClipHook(cfg.gradient_clip_per_step))

        # Tracking tensors
        halted: Tensor = torch.zeros(B, dtype=torch.bool, device=device)
        converged_flag: Tensor = torch.zeros(B, dtype=torch.bool, device=device)
        steps_used: Tensor = torch.zeros(B, dtype=torch.long, device=device)
        stable_count: Tensor = torch.zeros(B, dtype=torch.long, device=device)

        # Per-item halt reason candidates (accumulated, resolved by priority)
        halt_reason_candidates: List[List[str]] = [[] for _ in range(B)]

        # Step metrics collection
        step_metrics_list: List[Dict[str, Tensor]] = []

        # Deep supervision outputs
        deep_supervision_outputs: List[Tensor] = []

        # NaN-guarded last valid outputs
        y_last_valid: Tensor = y_k.clone().detach()

        # -- Iterative refinement loop --------------------------------------
        for k in range(1, global_max + 1):
            # ---- Determine which items are still active -------------------
            active: Tensor = ~halted  # (B,)

            if not active.any():
                break

            # ---- Check per-item budget exhaustion -------------------------
            # Items whose budget is strictly less than k should not be
            # processed at step k.
            budget_exceeded: Tensor = per_item_budget < k
            newly_budget_exhausted: Tensor = budget_exceeded & active
            if newly_budget_exhausted.any():
                idxs = newly_budget_exhausted.nonzero(as_tuple=False).view(-1)
                for idx in idxs:
                    halt_reason_candidates[idx.item()].append("budget_exhausted")
                halted = halted | newly_budget_exhausted
                # Record actual steps used as their budget for these items
                for idx in idxs:
                    i_val: int = idx.item()
                    if steps_used[i_val] == 0:
                        steps_used[i_val] = per_item_budget[i_val]
                active = ~halted
                if not active.any():
                    break

            # ---- GRU refinement step (only on active items) ---------------
            h_next: Tensor = h_k.clone()

            if active.any():
                active_indices: Tensor = active.nonzero(as_tuple=False).view(-1)
                h_active: Tensor = self.refinement_gru(
                    h_k[active_indices],
                    y_k[active_indices],
                    x_summary[active_indices],
                )
                # Scatter back into full-batch tensor
                h_next = h_next.clone()
                h_next[active_indices] = h_active

            # Register per-step gradient clipping hook on new hidden state
            if self.training and cfg.gradient_clip_per_step > 0.0:
                if h_next.requires_grad:
                    h_next.register_hook(
                        _GradientClipHook(cfg.gradient_clip_per_step)
                    )

            y_new: Tensor = self.output_proj(h_next)  # (B, output_dim)

            # ---- NaN guard ------------------------------------------------
            nan_halted_this_step: Tensor = torch.zeros(
                B, dtype=torch.bool, device=device
            )
            if cfg.nan_guard:
                nan_detected: Tensor = torch.isnan(y_new).any(dim=-1)  # (B,)
                nan_and_active: Tensor = nan_detected & active
                if nan_and_active.any():
                    nan_idxs = nan_and_active.nonzero(as_tuple=False).view(-1)
                    for idx in nan_idxs:
                        halt_reason_candidates[idx.item()].append("nan_guard")
                    nan_halted_this_step = nan_and_active
                    halted = halted | nan_and_active
                    # Restore last valid output for NaN items
                    y_new = torch.where(
                        nan_and_active.unsqueeze(-1).expand_as(y_new),
                        y_last_valid,
                        y_new,
                    )
                    # Restore hidden state for NaN items
                    h_next = torch.where(
                        nan_and_active.unsqueeze(-1).expand_as(h_next),
                        h_k,
                        h_next,
                    )
                    # Record steps for NaN items
                    steps_used = torch.where(
                        nan_and_active & (steps_used == 0),
                        torch.tensor(k, dtype=torch.long, device=device),
                        steps_used,
                    )
                    active = ~halted

            # ---- Convergence check ----------------------------------------
            converged_k: Tensor = torch.zeros(B, dtype=torch.bool, device=device)
            if active.any():
                conv_result: Tensor = self.convergence_checker.check(
                    y_new, y_k, cfg.convergence_criterion, cfg.convergence_eps
                )
                converged_k = conv_result & active

            # Patience tracking: increment stable_count if converged at this
            # step, reset to zero otherwise (only for active items)
            stable_count = torch.where(
                active & converged_k,
                stable_count + 1,
                torch.where(active, torch.zeros_like(stable_count), stable_count),
            )
            patience_met: Tensor = stable_count >= cfg.convergence_patience

            # Mark items that have met patience as converged
            newly_converged: Tensor = patience_met & active & ~halted
            if newly_converged.any():
                converged_flag = converged_flag | newly_converged
                conv_idxs = newly_converged.nonzero(as_tuple=False).view(-1)
                for idx in conv_idxs:
                    halt_reason_candidates[idx.item()].append("converged")
                halted = halted | newly_converged
                steps_used = torch.where(
                    newly_converged & (steps_used == 0),
                    torch.tensor(k, dtype=torch.long, device=device),
                    steps_used,
                )
                active = ~halted

            # ---- Check if at max steps for remaining active items ---------
            at_budget_limit: Tensor = (per_item_budget <= k) & active & ~halted
            if at_budget_limit.any():
                limit_idxs = at_budget_limit.nonzero(as_tuple=False).view(-1)
                for idx in limit_idxs:
                    halt_reason_candidates[idx.item()].append("max_steps")
                halted = halted | at_budget_limit
                steps_used = torch.where(
                    at_budget_limit & (steps_used == 0),
                    torch.tensor(k, dtype=torch.long, device=device),
                    steps_used,
                )
                active = ~halted

            # Also check if k == global_max for any remaining active items
            if k == global_max:
                still_active: Tensor = ~halted
                if still_active.any():
                    remaining_idxs = still_active.nonzero(as_tuple=False).view(-1)
                    for idx in remaining_idxs:
                        halt_reason_candidates[idx.item()].append("max_steps")
                    halted = halted | still_active
                    steps_used = torch.where(
                        still_active & (steps_used == 0),
                        torch.tensor(k, dtype=torch.long, device=device),
                        steps_used,
                    )

            # ---- Compute step metrics -------------------------------------
            with torch.no_grad():
                metrics: Dict[str, Tensor] = (
                    self.convergence_checker.compute_metrics(y_new, y_k)
                )
            step_metrics_list.append(metrics)

            # ---- Update y_k and h_k for items processed this step --------
            # An item gets y_new if it was active at entry of this step
            # (i.e., steps_used == 0 meaning still running, or steps_used == k
            # meaning it was halted this very step).
            update_mask: Tensor = (steps_used == 0) | (steps_used >= k)
            y_k = torch.where(
                update_mask.unsqueeze(-1).expand_as(y_k), y_new, y_k
            )
            h_k = torch.where(
                update_mask.unsqueeze(-1).expand_as(h_k), h_next, h_k
            )

            # Update last valid output (excluding NaN items)
            valid_update: Tensor = update_mask & ~nan_halted_this_step
            if valid_update.any():
                y_last_valid = torch.where(
                    valid_update.unsqueeze(-1).expand_as(y_last_valid),
                    y_new.detach(),
                    y_last_valid,
                )

            # ---- Deep supervision -----------------------------------------
            if cfg.deep_supervision and self.training:
                deep_supervision_outputs.append(y_new.clone())

            # ---- Early exit if all halted ---------------------------------
            if halted.all():
                break

        # -- Finalize results -----------------------------------------------

        # Safety: items that somehow never got steps_used set
        never_recorded: Tensor = steps_used == 0
        if never_recorded.any():
            steps_used = torch.where(
                never_recorded,
                torch.tensor(global_max, dtype=torch.long, device=device),
                steps_used,
            )
            nr_idxs = never_recorded.nonzero(as_tuple=False).view(-1)
            for idx in nr_idxs:
                i_val = idx.item()
                if not halt_reason_candidates[i_val]:
                    halt_reason_candidates[i_val].append("max_steps")

        # Resolve halt reasons by priority
        halt_reasons: List[str] = [
            _resolve_halt_reason(candidates)
            for candidates in halt_reason_candidates
        ]

        # Apply residual gate: blend y1 and refined output
        gate: Tensor = torch.sigmoid(self.residual_gate)  # (output_dim,)
        y2: Tensor = gate * y_k + (1.0 - gate) * y1_proj

        # Deep supervision weights
        ds_weights: Optional[Tensor] = None
        if cfg.deep_supervision and deep_supervision_outputs:
            n_ds_steps: int = len(deep_supervision_outputs)
            weights: List[float] = [
                cfg.deep_supervision_discount ** (global_max - (i + 1))
                for i in range(n_ds_steps)
            ]
            ds_weights = torch.tensor(weights, dtype=torch.float32, device=device)

        return System2Result(
            y2=y2,
            steps_used=steps_used,
            converged=converged_flag,
            halt_reason=halt_reasons,
            step_metrics=step_metrics_list if step_metrics_list else None,
            deep_supervision_outputs=(
                deep_supervision_outputs if deep_supervision_outputs else None
            ),
            deep_supervision_weights=ds_weights,
        )

    def extra_repr(self) -> str:
        cfg = self.config
        return (
            f"hidden_dim={cfg.hidden_dim}, "
            f"output_dim={cfg.output_dim}, "
            f"input_summary_dim={cfg.input_summary_dim}, "
            f"max_steps={cfg.max_steps}, "
            f"criterion={cfg.convergence_criterion!r}, "
            f"eps={cfg.convergence_eps}, "
            f"patience={cfg.convergence_patience}"
        )


# ---------------------------------------------------------------------------
# Deep supervision loss helper
# ---------------------------------------------------------------------------


def compute_deep_supervision_loss(
    result: System2Result,
    target: Tensor,
    loss_fn: nn.Module,
) -> Tensor:
    """Compute discounted deep-supervision loss from System2Result.

    Each intermediate output from the refinement loop contributes to the
    total loss, weighted by a geometric discount factor.  Later steps
    (closer to the final output) receive higher weight.

    Args:
        result: Output of ``System2Iterative.forward()``.
        target: Ground-truth labels/targets compatible with *loss_fn*.
        loss_fn: Loss function (e.g. ``nn.CrossEntropyLoss(reduction='none')``).

    Returns:
        Scalar loss tensor.
    """
    if (
        result.deep_supervision_outputs is None
        or result.deep_supervision_weights is None
    ):
        return torch.tensor(0.0, device=result.y2.device, requires_grad=True)

    total_loss: Tensor = torch.tensor(
        0.0, device=result.y2.device, dtype=torch.float32
    )
    weights: Tensor = result.deep_supervision_weights

    for i, y_step in enumerate(result.deep_supervision_outputs):
        step_loss: Tensor = loss_fn(y_step, target)
        if step_loss.dim() > 0:
            step_loss = step_loss.mean()
        total_loss = total_loss + weights[i] * step_loss

    # Normalize by sum of weights
    total_loss = total_loss / weights.sum().clamp(min=1e-8)
    return total_loss


# ---------------------------------------------------------------------------
# Utility: adaptive budget estimator
# ---------------------------------------------------------------------------


class AdaptiveBudgetEstimator(nn.Module):
    """Estimates the optimal step budget for each batch item based on
    the System 1 output confidence and the input summary.

    This is a lightweight MLP that predicts a continuous budget value
    which is then discretized.  It can be trained with the actual
    number of steps used as the target (via MSE or Huber loss).

    Architecture::

        cat(y1, x_summary) -> Linear -> ReLU -> LayerNorm
                            -> Linear -> ReLU -> Linear
                            -> Sigmoid -> scale to [1, max_steps]
    """

    def __init__(
        self,
        output_dim: int,
        input_summary_dim: int,
        max_steps: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.max_steps: int = max_steps

        self.net = nn.Sequential(
            nn.Linear(output_dim + input_summary_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, y1: Tensor, x_summary: Tensor) -> Tensor:
        """Predict per-item step budget.

        Args:
            y1: System 1 output ``(B, output_dim)``.
            x_summary: Input summary ``(B, input_summary_dim)``.

        Returns:
            Integer budget tensor ``(B,)`` in ``[1, max_steps]``.
        """
        combined: Tensor = torch.cat([y1, x_summary], dim=-1)
        raw: Tensor = self.net(combined).squeeze(-1)  # (B,)
        # Sigmoid -> [0, 1] -> scale to [1, max_steps]
        budget_continuous: Tensor = 1.0 + torch.sigmoid(raw) * (self.max_steps - 1)
        budget_discrete: Tensor = budget_continuous.round().long().clamp(
            min=1, max=self.max_steps
        )
        return budget_discrete


# ---------------------------------------------------------------------------
# Utility: confidence-based System 1/2 router
# ---------------------------------------------------------------------------


class DualProcessRouter(nn.Module):
    """Routes batch items between System 1 (fast) and System 2 (slow)
    based on confidence thresholds.

    Items with System 1 confidence >= threshold use y1 directly.
    Items below threshold are processed by System 2.

    This module does NOT contain System 1 or System 2 -- it simply
    orchestrates routing and merging.

    Attributes:
        confidence_threshold: Minimum System 1 confidence to skip
            System 2 refinement.
        system2: The :class:`System2Iterative` module.
        budget_estimator: Optional :class:`AdaptiveBudgetEstimator`.
    """

    def __init__(
        self,
        system2: System2Iterative,
        confidence_threshold: float = 0.7,
        budget_estimator: Optional[AdaptiveBudgetEstimator] = None,
    ) -> None:
        super().__init__()
        self.confidence_threshold: float = confidence_threshold
        self.system2: System2Iterative = system2
        self.budget_estimator: Optional[AdaptiveBudgetEstimator] = budget_estimator

    def forward(
        self,
        y1: Tensor,
        x_summary: Tensor,
        steps_budget: Optional[Union[int, Tensor]] = None,
        context: Optional[Tensor] = None,
        force_system2: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Route items between System 1 and System 2.

        Args:
            y1: System 1 output ``(B, output_dim)``.
            x_summary: Input summary ``(B, input_summary_dim)``.
            steps_budget: Step budget (overrides budget estimator).
            context: Optional context tensor.
            force_system2: If ``True``, all items go through System 2
                regardless of confidence.

        Returns:
            Tuple of:
            - ``y_out``: Final output ``(B, output_dim)``.
            - ``info``: Dictionary with routing metadata including
              ``s1_confidence``, ``s2_fraction``, ``s2_indices``, and
              ``system2_result``.
        """
        B: int = y1.shape[0]
        device: torch.device = y1.device

        # Compute System 1 confidence
        probs: Tensor = F.softmax(y1.float(), dim=-1)
        conf_s1: Tensor = probs.max(dim=-1).values  # (B,)

        if force_system2:
            needs_s2: Tensor = torch.ones(B, dtype=torch.bool, device=device)
        else:
            needs_s2 = conf_s1 < self.confidence_threshold

        info: Dict[str, Any] = {
            "s1_confidence": conf_s1.detach(),
            "s2_fraction": needs_s2.float().mean().item(),
            "s2_indices": needs_s2.nonzero(as_tuple=False).view(-1),
        }

        if not needs_s2.any():
            # All items handled by System 1
            info["system2_result"] = None
            return y1, info

        # Estimate budget if estimator is available and no explicit budget
        s2_budget: Optional[Union[int, Tensor]]
        if steps_budget is None and self.budget_estimator is not None:
            s2_budget = self.budget_estimator(
                y1[needs_s2], x_summary[needs_s2]
            )
        elif isinstance(steps_budget, Tensor):
            s2_budget = steps_budget[needs_s2]
        else:
            s2_budget = steps_budget

        # Run System 2 on items that need refinement
        result: System2Result = self.system2(
            y1=y1[needs_s2],
            x_summary=x_summary[needs_s2],
            steps_budget=s2_budget,
            context=context[needs_s2] if context is not None else None,
        )

        # Merge outputs
        y_out: Tensor = y1.clone()
        y_out[needs_s2] = result.y2
        info["system2_result"] = result

        return y_out, info


# ---------------------------------------------------------------------------
# Visualization / debugging utilities
# ---------------------------------------------------------------------------


def format_step_metrics(
    step_metrics: List[Dict[str, Tensor]],
    batch_idx: int = 0,
) -> str:
    """Format step metrics for a single batch item as a readable string.

    Args:
        step_metrics: List of per-step metric dicts from System2Result.
        batch_idx: Which batch item to display.

    Returns:
        Multi-line string summarizing the refinement trace.
    """
    lines: List[str] = [
        "Step | delta_kl    | delta_max   | conf        | argmax"
    ]
    lines.append("-" * 65)
    for k, m in enumerate(step_metrics, 1):
        dkl: float = m["delta_kl"][batch_idx].item()
        dmax: float = m["delta_max"][batch_idx].item()
        conf: float = m["conf_k"][batch_idx].item()
        amax: int = int(m["argmax_k"][batch_idx].item())
        lines.append(
            f"{k:4d} | {dkl:11.6f} | {dmax:11.6f} | {conf:11.6f} | {amax}"
        )
    return "\n".join(lines)


def summarize_result(result: System2Result) -> str:
    """One-line summary of a System2Result for logging.

    Args:
        result: System2Result to summarize.

    Returns:
        Summary string with batch size, average steps, convergence
        count, and halt reason distribution.
    """
    B: int = result.y2.shape[0]
    avg_steps: float = result.steps_used.float().mean().item()
    n_converged: int = int(result.converged.sum().item())
    reasons: Dict[str, int] = {}
    for r in result.halt_reason:
        reasons[r] = reasons.get(r, 0) + 1
    reason_str: str = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
    return (
        f"System2 | B={B} | avg_steps={avg_steps:.1f} | "
        f"converged={n_converged}/{B} | halt=[{reason_str}]"
    )


# ===========================================================================
# Self-tests
# ===========================================================================


def _run_self_tests() -> None:
    """Comprehensive self-tests for the System 2 iterative refinement module.

    Runs 11 tests covering shape correctness, convergence behavior,
    halt reasons, gradient flow, patience, determinism, and mixed-batch
    behavior.  Prints a summary at the end.
    """
    import traceback

    torch.manual_seed(42)
    device: torch.device = torch.device("cpu")

    passed: int = 0
    total: int = 11
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
    print("System 2 Iterative Refinement -- Self Tests")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Test 1: Forward shape -- y2 shape matches output_dim
    # -----------------------------------------------------------------------
    try:
        cfg = System2Config(
            hidden_dim=64,
            output_dim=32,
            input_summary_dim=64,
            max_steps=5,
            convergence_patience=2,
        )
        model = System2Iterative(cfg).to(device)
        model.eval()

        B = 4
        y1 = torch.randn(B, 32, device=device)
        x_summary = torch.randn(B, 64, device=device)
        result = model(y1, x_summary, steps_budget=5)

        ok = (
            result.y2.shape == (B, 32)
            and result.steps_used.shape == (B,)
            and result.converged.shape == (B,)
            and len(result.halt_reason) == B
        )
        _report(
            "Forward shape",
            ok,
            f"y2={result.y2.shape}, steps_used={result.steps_used.shape}, "
            f"converged={result.converged.shape}, "
            f"halt_reason len={len(result.halt_reason)}",
        )
    except Exception as e:
        _report("Forward shape", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 2: Convergence -- synthetic decreasing-delta converges
    #         before max_steps
    # -----------------------------------------------------------------------
    try:
        cfg = System2Config(
            hidden_dim=32,
            output_dim=16,
            input_summary_dim=32,
            max_steps=20,
            convergence_eps=0.1,
            convergence_patience=2,
            convergence_criterion="logit",
        )
        model = System2Iterative(cfg).to(device)
        model.eval()

        torch.manual_seed(123)
        B = 2
        y1 = torch.randn(B, 16, device=device)
        x_summary = torch.randn(B, 32, device=device)

        result = model(y1, x_summary, steps_budget=20)
        max_steps_used: int = int(result.steps_used.max().item())

        ok = (
            result.y2.shape == (B, 16)
            and max_steps_used <= 20
            and all(
                r in ("converged", "max_steps", "budget_exhausted")
                for r in result.halt_reason
            )
        )
        _report(
            "Convergence test",
            ok,
            f"steps_used={result.steps_used.tolist()}, "
            f"converged={result.converged.tolist()}, "
            f"halt_reason={result.halt_reason}",
        )
    except Exception as e:
        _report("Convergence test", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 3: Max steps halt -- very tight eps hits max_steps
    # -----------------------------------------------------------------------
    try:
        cfg = System2Config(
            hidden_dim=32,
            output_dim=16,
            input_summary_dim=32,
            max_steps=3,
            convergence_eps=1e-12,
            convergence_patience=2,
            convergence_criterion="logit",
        )
        model = System2Iterative(cfg).to(device)
        model.eval()

        torch.manual_seed(999)
        B = 2
        y1 = torch.randn(B, 16, device=device)
        x_summary = torch.randn(B, 32, device=device)

        result = model(y1, x_summary, steps_budget=3)
        all_max_steps: bool = all(
            r in ("max_steps", "budget_exhausted") for r in result.halt_reason
        )
        all_used_le_3: bool = (result.steps_used <= 3).all().item()

        ok = all_max_steps and all_used_le_3
        _report(
            "Max steps halt",
            ok,
            f"halt_reason={result.halt_reason}, "
            f"steps_used={result.steps_used.tolist()}",
        )
    except Exception as e:
        _report("Max steps halt", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 4: NaN guard -- inject NaN at step 3 -> halt_reason is "nan_guard"
    # -----------------------------------------------------------------------
    try:
        cfg = System2Config(
            hidden_dim=32,
            output_dim=16,
            input_summary_dim=32,
            max_steps=10,
            convergence_eps=1e-12,
            convergence_patience=2,
            nan_guard=True,
        )
        model = System2Iterative(cfg).to(device)
        model.eval()

        # Monkey-patch the output projection to inject NaN at step 3
        original_forward = model.output_proj.forward
        call_count: List[int] = [0]

        def _nan_injecting_forward(h: Tensor) -> Tensor:
            call_count[0] += 1
            out: Tensor = original_forward(h)
            if call_count[0] == 3:
                out = out.clone()
                out[0, :] = float("nan")
            return out

        model.output_proj.forward = _nan_injecting_forward  # type: ignore[assignment]

        B = 2
        y1 = torch.randn(B, 16, device=device)
        x_summary = torch.randn(B, 32, device=device)

        result = model(y1, x_summary, steps_budget=10)

        first_reason: str = result.halt_reason[0]
        ok = first_reason == "nan_guard"
        ok = ok and not torch.isnan(result.y2[0]).any().item()

        _report(
            "NaN guard",
            ok,
            f"halt_reason={result.halt_reason}, "
            f"y2_has_nan={torch.isnan(result.y2).any().item()}",
        )

        model.output_proj.forward = original_forward  # type: ignore[assignment]
    except Exception as e:
        _report("NaN guard", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 5: Budget enforcement -- steps_budget=3 -> max 3 steps used
    # -----------------------------------------------------------------------
    try:
        cfg = System2Config(
            hidden_dim=32,
            output_dim=16,
            input_summary_dim=32,
            max_steps=20,
            convergence_eps=1e-12,
            convergence_patience=2,
        )
        model = System2Iterative(cfg).to(device)
        model.eval()

        B = 4
        y1 = torch.randn(B, 16, device=device)
        x_summary = torch.randn(B, 32, device=device)

        result = model(y1, x_summary, steps_budget=3)
        all_within_budget: bool = (result.steps_used <= 3).all().item()

        ok = all_within_budget
        _report(
            "Budget enforcement",
            ok,
            f"steps_used={result.steps_used.tolist()} (budget=3)",
        )
    except Exception as e:
        _report("Budget enforcement", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 6: All convergence criteria -- kl, logit, argmax
    # -----------------------------------------------------------------------
    try:
        all_criteria_ok: bool = True
        criteria_details: List[str] = []

        for criterion in ("kl", "logit", "argmax"):
            cfg = System2Config(
                hidden_dim=32,
                output_dim=16,
                input_summary_dim=32,
                max_steps=10,
                convergence_eps=0.5,
                convergence_patience=1,
                convergence_criterion=criterion,
            )
            model = System2Iterative(cfg).to(device)
            model.eval()

            torch.manual_seed(42)
            B = 2
            y1 = torch.randn(B, 16, device=device)
            x_summary = torch.randn(B, 32, device=device)

            result = model(y1, x_summary, steps_budget=10)
            valid: bool = (
                result.y2.shape == (B, 16)
                and len(result.halt_reason) == B
                and not torch.isnan(result.y2).any().item()
            )
            criteria_details.append(
                f"{criterion}: valid={valid}, halt={result.halt_reason}"
            )
            if not valid:
                all_criteria_ok = False

        ok = all_criteria_ok
        _report(
            "All convergence criteria",
            ok,
            "; ".join(criteria_details),
        )
    except Exception as e:
        _report("All convergence criteria", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 7: Gradient flow through GRU
    # -----------------------------------------------------------------------
    try:
        cfg = System2Config(
            hidden_dim=32,
            output_dim=16,
            input_summary_dim=32,
            max_steps=5,
            convergence_eps=1e-12,
            convergence_patience=2,
        )
        model = System2Iterative(cfg).to(device)
        model.train()

        B = 2
        y1 = torch.randn(B, 16, device=device, requires_grad=True)
        x_summary = torch.randn(B, 32, device=device, requires_grad=True)

        result = model(y1, x_summary, steps_budget=5)
        loss: Tensor = result.y2.sum()
        loss.backward()

        y1_grad_exists: bool = (
            y1.grad is not None and y1.grad.abs().sum().item() > 0
        )
        x_grad_exists: bool = (
            x_summary.grad is not None and x_summary.grad.abs().sum().item() > 0
        )

        gru_grads: List[bool] = []
        for name, p in model.refinement_gru.named_parameters():
            if p.grad is not None:
                gru_grads.append(p.grad.abs().sum().item() > 0)

        ok = y1_grad_exists and x_grad_exists and all(gru_grads)
        _report(
            "Gradient flow",
            ok,
            f"y1_grad={y1_grad_exists}, x_grad={x_grad_exists}, "
            f"gru_grads={gru_grads}",
        )
    except Exception as e:
        _report("Gradient flow", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 8: Patience -- need M consecutive stable, not just one
    # -----------------------------------------------------------------------
    try:
        patience_val: int = 3
        cfg = System2Config(
            hidden_dim=32,
            output_dim=16,
            input_summary_dim=32,
            max_steps=20,
            convergence_eps=100.0,
            convergence_patience=patience_val,
            convergence_criterion="logit",
        )
        model = System2Iterative(cfg).to(device)
        model.eval()

        torch.manual_seed(42)
        B = 2
        y1 = torch.randn(B, 16, device=device)
        x_summary = torch.randn(B, 32, device=device)

        result = model(y1, x_summary, steps_budget=20)

        min_steps: int = int(result.steps_used.min().item())
        ok = min_steps >= patience_val
        all_converged: bool = all(r == "converged" for r in result.halt_reason)
        ok = ok and all_converged
        _report(
            "Patience enforcement",
            ok,
            f"patience={patience_val}, min_steps_used={min_steps}, "
            f"steps_used={result.steps_used.tolist()}, "
            f"halt_reason={result.halt_reason}",
        )
    except Exception as e:
        _report("Patience enforcement", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 9: Determinism -- same input -> same output
    # -----------------------------------------------------------------------
    try:
        cfg = System2Config(
            hidden_dim=32,
            output_dim=16,
            input_summary_dim=32,
            max_steps=5,
            convergence_patience=2,
        )
        model = System2Iterative(cfg).to(device)
        model.eval()

        B = 3
        torch.manual_seed(777)
        y1 = torch.randn(B, 16, device=device)
        x_summary = torch.randn(B, 32, device=device)

        result1 = model(y1, x_summary, steps_budget=5)
        result2 = model(y1, x_summary, steps_budget=5)

        outputs_match: bool = torch.allclose(result1.y2, result2.y2, atol=1e-6)
        steps_match: bool = (result1.steps_used == result2.steps_used).all().item()
        reasons_match: bool = result1.halt_reason == result2.halt_reason

        ok = outputs_match and steps_match and reasons_match
        _report(
            "Determinism",
            ok,
            f"outputs_match={outputs_match}, steps_match={steps_match}, "
            f"reasons_match={reasons_match}",
        )
    except Exception as e:
        _report("Determinism", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 10: Batch with mixed convergence
    # -----------------------------------------------------------------------
    try:
        cfg = System2Config(
            hidden_dim=32,
            output_dim=16,
            input_summary_dim=32,
            max_steps=10,
            convergence_eps=0.01,
            convergence_patience=2,
            convergence_criterion="logit",
        )
        model = System2Iterative(cfg).to(device)
        model.eval()

        B = 6
        torch.manual_seed(42)
        y1 = torch.randn(B, 16, device=device)
        x_summary = torch.randn(B, 32, device=device)
        budgets = torch.tensor(
            [2, 2, 5, 5, 10, 10], dtype=torch.long, device=device
        )

        result = model(y1, x_summary, steps_budget=budgets)

        within_budget: bool = (result.steps_used <= budgets).all().item()
        short_budget_ok: bool = (result.steps_used[:2] <= 2).all().item()

        ok = within_budget and short_budget_ok and result.y2.shape == (B, 16)
        _report(
            "Mixed convergence batch",
            ok,
            f"budgets={budgets.tolist()}, "
            f"steps_used={result.steps_used.tolist()}, "
            f"halt_reason={result.halt_reason}",
        )
    except Exception as e:
        _report("Mixed convergence batch", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 11: Edge cases -- batch size 1, steps_budget=1
    # -----------------------------------------------------------------------
    try:
        cfg = System2Config(
            hidden_dim=32,
            output_dim=16,
            input_summary_dim=32,
            max_steps=10,
            convergence_patience=2,
        )
        model = System2Iterative(cfg).to(device)
        model.eval()

        # Sub-test A: Batch size 1
        y1_b1 = torch.randn(1, 16, device=device)
        x_summary_b1 = torch.randn(1, 32, device=device)
        result_b1 = model(y1_b1, x_summary_b1, steps_budget=5)
        b1_ok: bool = (
            result_b1.y2.shape == (1, 16) and len(result_b1.halt_reason) == 1
        )

        # Sub-test B: steps_budget=1
        y1_s1 = torch.randn(3, 16, device=device)
        x_summary_s1 = torch.randn(3, 32, device=device)
        result_s1 = model(y1_s1, x_summary_s1, steps_budget=1)
        s1_ok: bool = (
            result_s1.y2.shape == (3, 16)
            and (result_s1.steps_used <= 1).all().item()
        )

        ok = b1_ok and s1_ok
        _report(
            "Edge cases (B=1, budget=1)",
            ok,
            f"B=1: shape={result_b1.y2.shape}, halt={result_b1.halt_reason}; "
            f"budget=1: steps={result_s1.steps_used.tolist()}, "
            f"halt={result_s1.halt_reason}",
        )
    except Exception as e:
        _report("Edge cases (B=1, budget=1)", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 70)
    print(f"{passed}/{total} self-tests passed")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
    print("=" * 70)


# ===========================================================================
# Entry point
# ===========================================================================


if __name__ == "__main__":
    _run_self_tests()
