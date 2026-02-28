"""
Planning Modules for Active Inference.

Implements rollout-based planners that simulate candidate action sequences
through a learned transition model in latent space, score each sequence by
Expected Free Energy (EFE), and select the action (or sequence) that
minimises EFE.

Two planner families are provided:

1. **Random Shooting** -- sample N action sequences from a fixed prior
   (uniform/Gaussian), evaluate via a single rollout pass, and select the
   best.  Simple, embarrassingly parallel, zero hyperparameters beyond N.

2. **Cross-Entropy Method (CEM)** -- iteratively sample, evaluate, keep the
   elite fraction, and refit the sampling distribution.  Converges to
   higher-quality plans than random shooting for the same total budget.

Both planners delegate the heavy lifting to a shared ``RolloutEngine`` that
parallelises N candidate sequences over the batch dimension by reshaping
``(B, N, ...)`` into ``(B*N, ...)``, loops over H horizon steps through the
transition model, accumulates discounted per-step EFE, and returns a
structured ``RolloutResult`` with full trajectory information.

Tensor shape conventions:
    B   -- batch size
    N   -- number of candidate rollout sequences
    H   -- planning horizon (number of imagined time-steps)
    S   -- state_dim (latent space dimensionality)
    A   -- action_dim (continuous width or discrete cardinality)
    O   -- obs_dim (observation space dimensionality)

Hard invariants:
    - ``RolloutResult.total_efe`` has shape ``(B, N)``.
    - ``PlanResult.best_action`` has shape ``(B, A)`` for continuous actions
      or ``(B,)`` for discrete actions.
    - All EFE computations are performed in fp32 regardless of input dtype.
    - The planner never mutates ``initial_state`` in place.

References:
    - Chua, K. et al. (2018). Deep Reinforcement Learning in a Handful of
      Trials using Probabilistic Dynamics Models (PETS).
    - Botvinick, M. & Toussaint, M. (2012). Planning as inference.
    - Fountas, Z. et al. (2020). Deep Active Inference Agents Using
      Monte-Carlo Methods. NeurIPS.
    - De Boer, P.-T. et al. (2005). A Tutorial on the Cross-Entropy Method.
    - Friston, K. et al. (2017). Active Inference: A Process Theory.
    - Pinneri, C. et al. (2021). Sample-efficient cross-entropy method for
      real-time planning. CoRL.
    - Mazzaglia, P. et al. (2022). The Free Energy Principle for Perception
      and Action: A Deep Active Inference Agent. Neural Computation.

This module is part of the brain-inspired AI system described in CLAUDE.md.
The decision layer (basal ganglia analog) uses EFE to select actions that
jointly minimise prediction error, resolve uncertainty, and maintain agency.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

_EFE_SUM_INVARIANT_TOL: float = 1e-5
_MIN_STD_CLAMP: float = 1e-4
_DEFAULT_ACTION_BOUND: float = 1.0
_LOG_2PI: float = math.log(2.0 * math.pi)


# ============================================================================
# Protocol interfaces -- what the planner expects from external modules
# ============================================================================


class TransitionModelProtocol(Protocol):
    """Protocol for transition models used by the rollout engine.

    A transition model predicts the next latent state distribution given the
    current latent state and an action.  It must return a mean and log-variance
    parameterising a diagonal Gaussian in state space.
    """

    def __call__(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state distribution.

        Args:
            state: Current latent state, shape ``(B, state_dim)``.
            action: Action vector, shape ``(B, action_dim)``.

        Returns:
            next_state_mean: Predicted mean, shape ``(B, state_dim)``.
            next_state_log_var: Predicted log-variance, shape ``(B, state_dim)``.
        """
        ...


class LikelihoodModelProtocol(Protocol):
    """Protocol for likelihood (decoder) models.

    Maps latent states to predicted observation distributions.
    """

    def __call__(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode state to observation distribution.

        Args:
            state: Latent state, shape ``(B, state_dim)``.

        Returns:
            obs_mean: Predicted observation mean, shape ``(B, obs_dim)``.
            obs_log_var: Predicted observation log-variance, shape ``(B, obs_dim)``.
        """
        ...


class EFEComputerProtocol(Protocol):
    """Protocol for EFE computation modules.

    Given transition outputs for a single step, returns the scalar EFE and a
    breakdown dictionary.
    """

    def compute_single_step(
        self,
        predicted_obs_params: Tuple[torch.Tensor, torch.Tensor],
        posterior_params: Tuple[torch.Tensor, torch.Tensor],
        prior_params: Tuple[torch.Tensor, torch.Tensor],
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        preference_params: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute EFE for a single planning step.

        Args:
            predicted_obs_params: ``(pred_mean, pred_log_var)`` each ``(B, obs_dim)``.
            posterior_params: ``(post_mean, post_log_var)`` each ``(B, state_dim)``.
            prior_params: ``(prior_mean, prior_log_var)`` each ``(B, state_dim)``.
            state: Current state, ``(B, state_dim)``.
            action: Action taken, ``(B,)`` or ``(B, action_dim)``.
            next_state: Predicted next state, ``(B, state_dim)``.
            preference_params: ``(pref_mean, pref_log_var)`` each ``(obs_dim,)``
                or ``(B, obs_dim)``.

        Returns:
            efe_step: Total EFE, shape ``(B,)``.
            terms_dict: Breakdown with keys ``"pragmatic"``, ``"epistemic"``,
                ``"instrumental"``, ``"total"``.
        """
        ...


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class PlannerConfig:
    """Configuration for the planning / rollout sub-system.

    Controls the planner type, horizon, number of rollouts, CEM-specific
    hyperparameters, action space bounds, and sampling temperatures.

    Attributes:
        planner_type: Which planner to use.  One of ``"random_shooting"`` or
            ``"cem"``.  Additional planner types can be registered via
            :func:`create_planner`.
        planning_horizon: Number of imagined time-steps H.  Longer horizons
            give better long-term plans at the cost of compounding model error
            and compute.
        num_rollouts: Number of candidate action sequences N evaluated per
            planning call.  Higher values improve plan quality at the cost
            of memory and compute.
        discount_factor: Temporal discount gamma applied to EFE at each step.
            Values < 1.0 make the agent prefer near-term rewards; 1.0 is no
            discounting.
        normalize_by_horizon: If True, divide the accumulated EFE by the
            effective number of discounted steps so that the total magnitude
            is comparable across different horizons.

        cem_iterations: Number of refinement rounds for the CEM planner.
            More iterations give better convergence but cost proportionally
            more compute.
        cem_elite_fraction: Fraction of samples kept as elites per CEM
            iteration.  Typical values are 0.05--0.2.  Smaller values give
            sharper but noisier updates.
        cem_temperature: Softmax temperature used when sampling from the CEM
            distribution.  Higher temperatures produce more exploration; lower
            temperatures sharpen the distribution.
        cem_momentum: Exponential momentum for blending old and new CEM
            distribution parameters.  0.0 means no momentum (standard CEM);
            0.9 means strong smoothing.  Helps prevent distribution collapse
            in early iterations.
        cem_min_std: Minimum standard deviation for the CEM Gaussian.  Prevents
            premature convergence by keeping some exploration alive.

        action_bounds: Optional tuple ``(lower, upper)`` for continuous action
            clamping.  When set, sampled continuous actions are clamped to this
            range after generation.  Ignored for discrete actions.
        action_temperature: Softmax temperature for final action selection
            from the softmax over rollout EFEs.  Lower values make selection
            more greedy; higher values add stochasticity.
        discrete_actions: If True, the action space is discrete (integer
            indices / one-hot).  If False, actions are continuous vectors.
        action_dim: Dimensionality of the action space.  For discrete actions,
            this is the number of categories.  Must be set before planning.
        state_dim: Dimensionality of the latent state space.  Must match the
            transition model's state representation.
        obs_dim: Dimensionality of the observation space.  Must match the
            likelihood model's output.

        action_noise_std: Standard deviation of Gaussian noise injected into
            continuous action samples.  Only used by random shooting.
        seed: Optional RNG seed for reproducibility.  ``None`` means
            non-deterministic sampling.
        use_warm_start: If True, CEM warm-starts from the previous call's
            distribution (shifted by one time-step).  Reduces compute by
            starting from a better initial guess.
        log_level: Verbosity level for planner diagnostics.  0 = silent,
            1 = summary per call, 2 = per-iteration detail.
    """

    planner_type: str = "cem"
    planning_horizon: int = 8
    num_rollouts: int = 128
    discount_factor: float = 0.99
    normalize_by_horizon: bool = True

    # CEM-specific
    cem_iterations: int = 5
    cem_elite_fraction: float = 0.1
    cem_temperature: float = 1.0
    cem_momentum: float = 0.0
    cem_min_std: float = 0.01

    # Action space
    action_bounds: Optional[Tuple[float, float]] = None
    action_temperature: float = 1.0
    discrete_actions: bool = False
    action_dim: int = 10
    state_dim: int = 256
    obs_dim: int = 4096

    # Noise / exploration
    action_noise_std: float = 0.3
    seed: Optional[int] = None
    use_warm_start: bool = True
    log_level: int = 0


# ============================================================================
# Result dataclasses
# ============================================================================


@dataclass
class RolloutResult:
    """Structured output of the RolloutEngine.

    Contains the full trajectory of latent states, predicted observations,
    and per-step / total EFE for every candidate sequence.

    Attributes:
        total_efe: Total (discounted, optionally horizon-normalised) EFE for
            each candidate sequence.  Shape ``(B, N)``.  Lower is better.
        per_step_efe: Per-step EFE before discounting, shape ``(B, N, H)``.
            Useful for diagnostics and for identifying which horizon steps
            contribute most to the total.
        per_step_terms: Dictionary mapping term names (``"pragmatic"``,
            ``"epistemic"``, ``"instrumental"``) to tensors of shape
            ``(B, N, H)`` giving the per-step, per-term breakdown.
        trajectories: Full latent state trajectories, shape
            ``(B, N, H, state_dim)``.  Index ``[b, n, t]`` is the predicted
            state at step t for rollout n in batch element b.
        predicted_obs: Predicted observation means along each trajectory,
            shape ``(B, N, H, obs_dim)``.  Useful for visualisation and
            debugging of the generative model.
        discount_factors: Discount factor applied at each step, shape ``(H,)``.
    """

    total_efe: torch.Tensor                          # (B, N)
    per_step_efe: torch.Tensor                       # (B, N, H)
    per_step_terms: Dict[str, torch.Tensor]          # each (B, N, H)
    trajectories: torch.Tensor                       # (B, N, H, state_dim)
    predicted_obs: torch.Tensor                      # (B, N, H, obs_dim)
    discount_factors: torch.Tensor                   # (H,)

    def best_indices(self) -> torch.Tensor:
        """Return the index of the best (lowest EFE) sequence per batch.

        Returns:
            Indices of shape ``(B,)``.
        """
        return self.total_efe.argmin(dim=-1)

    def best_efe(self) -> torch.Tensor:
        """Return the EFE of the best sequence per batch element.

        Returns:
            Shape ``(B,)``.
        """
        idx = self.best_indices()
        B = self.total_efe.shape[0]
        return self.total_efe[torch.arange(B, device=self.total_efe.device), idx]

    def best_trajectories(self) -> torch.Tensor:
        """Return the latent trajectory of the best sequence per batch.

        Returns:
            Shape ``(B, H, state_dim)``.
        """
        idx = self.best_indices()
        B = self.trajectories.shape[0]
        return self.trajectories[torch.arange(B, device=self.trajectories.device), idx]


@dataclass
class PlanResult:
    """Structured output of a planner's ``plan()`` call.

    Contains the selected action, the full best sequence, EFE information,
    and a debug dictionary for logging / analysis.

    Attributes:
        best_action: The action to execute at the current time-step.  Shape
            ``(B, action_dim)`` for continuous actions or ``(B,)`` for
            discrete actions.
        best_action_sequence: Full planned action sequence of the best
            rollout.  Shape ``(B, H, action_dim)`` for continuous or
            ``(B, H)`` for discrete.  Useful for warm-starting the next
            planning call.
        efe_total: Total EFE of the selected sequence, shape ``(B,)``.
        efe_terms: Dictionary mapping term names to ``(B,)`` tensors giving
            the breakdown of the selected sequence's EFE.
        all_efe: EFE of all N candidate sequences, shape ``(B, N)``.  Useful
            for analysis of the planning landscape.
        debug: Free-form dictionary for planner-specific diagnostics.
            May include keys such as ``"elapsed_ms"``, ``"cem_iteration_efe"``,
            ``"elite_mean_efe"``, ``"cem_final_std"``, ``"n_rollouts"``,
            ``"planner_type"``, etc.
    """

    best_action: torch.Tensor
    best_action_sequence: torch.Tensor
    efe_total: torch.Tensor                          # (B,)
    efe_terms: Dict[str, torch.Tensor]               # each (B,)
    all_efe: torch.Tensor                            # (B, N)
    debug: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Run basic shape and consistency checks.

        Raises:
            ValueError: If any invariant is violated.
        """
        B = self.efe_total.shape[0]
        if self.efe_total.dim() != 1:
            raise ValueError(
                f"efe_total should be 1-D (B,), got shape {self.efe_total.shape}"
            )
        if self.all_efe.dim() != 2 or self.all_efe.shape[0] != B:
            raise ValueError(
                f"all_efe should be (B, N), got shape {self.all_efe.shape}"
            )
        if self.best_action.shape[0] != B:
            raise ValueError(
                f"best_action batch dim {self.best_action.shape[0]} != {B}"
            )
        if self.best_action_sequence.shape[0] != B:
            raise ValueError(
                f"best_action_sequence batch dim "
                f"{self.best_action_sequence.shape[0]} != {B}"
            )
        for name, term in self.efe_terms.items():
            if term.shape != (B,):
                raise ValueError(
                    f"efe_terms['{name}'] should be ({B},), got {term.shape}"
                )


# ============================================================================
# Rollout Engine
# ============================================================================


class RolloutEngine(nn.Module):
    """Parallel rollout engine for latent imagination.

    Evaluates N candidate action sequences by simulating them through a
    learned transition model in latent space and scoring each trajectory
    by Expected Free Energy.

    The engine parallelises over N by reshaping ``(B, N, ...)`` into
    ``(B*N, ...)`` before the step loop, then reshaping back.  This is
    transparent to the transition, likelihood, and EFE modules which only
    see a ``(B_eff, ...)`` batch.

    The rollout procedure at each step t:
        1. Extract action_t from the candidate sequence.
        2. Predict next_state distribution via transition_model(state, action_t).
        3. Sample next_state from the predicted distribution.
        4. Decode predicted observations via likelihood_model(next_state).
        5. Compute per-step EFE via efe_computer.compute_single_step(...).
        6. Accumulate discounted EFE.
        7. Advance state <- next_state.

    After the loop, optionally normalise by the effective discounted horizon.

    Args:
        transition_model: A module implementing :class:`TransitionModelProtocol`.
        likelihood_model: A module implementing :class:`LikelihoodModelProtocol`.
        efe_computer: A module implementing :class:`EFEComputerProtocol`.
        config: :class:`PlannerConfig` with horizon, discount, and normalisation
            settings.

    Example::

        engine = RolloutEngine(transition, likelihood, efe_computer, config)
        result = engine.rollout(
            initial_state=torch.randn(4, 256),
            action_sequences=torch.randn(4, 128, 8, 10),
            preferences=(pref_mean, pref_lv),
        )
        print(result.total_efe.shape)  # (4, 128)
    """

    def __init__(
        self,
        transition_model: nn.Module,
        likelihood_model: nn.Module,
        efe_computer: nn.Module,
        config: PlannerConfig,
    ) -> None:
        super().__init__()
        self.transition_model = transition_model
        self.likelihood_model = likelihood_model
        self.efe_computer = efe_computer
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_discount_factors(self, horizon: int) -> torch.Tensor:
        """Pre-compute the discount factor for each step in the horizon.

        Args:
            horizon: Number of planning steps H.

        Returns:
            Tensor of shape ``(H,)`` with values
            ``[1, gamma, gamma^2, ..., gamma^(H-1)]``.
        """
        gamma = self.config.discount_factor
        return torch.tensor(
            [gamma ** t for t in range(horizon)], dtype=torch.float32
        )

    def _effective_horizon(self, horizon: int) -> float:
        """Compute the sum of discounted steps for normalisation.

        Args:
            horizon: Number of planning steps H.

        Returns:
            ``sum_{t=0}^{H-1} gamma^t``.  This is the geometric series
            ``(1 - gamma^H) / (1 - gamma)`` when gamma != 1, else H.
        """
        gamma = self.config.discount_factor
        if abs(gamma - 1.0) < 1e-8:
            return float(horizon)
        return (1.0 - gamma ** horizon) / (1.0 - gamma)

    def _flatten_batch_rollouts(
        self,
        tensor: torch.Tensor,
        B: int,
        N: int,
    ) -> torch.Tensor:
        """Reshape ``(B, N, ...)`` to ``(B*N, ...)``.

        Args:
            tensor: Input tensor with first two dims ``(B, N)``.
            B: Batch size.
            N: Number of rollouts.

        Returns:
            Reshaped tensor with first dim ``B*N``.
        """
        trailing = tensor.shape[2:]
        return tensor.reshape(B * N, *trailing)

    def _unflatten_batch_rollouts(
        self,
        tensor: torch.Tensor,
        B: int,
        N: int,
    ) -> torch.Tensor:
        """Reshape ``(B*N, ...)`` back to ``(B, N, ...)``.

        Args:
            tensor: Input tensor with first dim ``B*N``.
            B: Batch size.
            N: Number of rollouts.

        Returns:
            Reshaped tensor with first two dims ``(B, N)``.
        """
        trailing = tensor.shape[1:]
        return tensor.reshape(B, N, *trailing)

    # ------------------------------------------------------------------
    # Main rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        initial_state: torch.Tensor,
        action_sequences: torch.Tensor,
        preferences: Tuple[torch.Tensor, torch.Tensor],
        *,
        return_trajectories: bool = True,
    ) -> RolloutResult:
        """Roll out candidate action sequences and evaluate by EFE.

        Args:
            initial_state: Current latent state, shape ``(B, state_dim)``.
            action_sequences: Candidate action sequences, shape
                ``(B, N, H, action_dim)``.  For discrete actions, these should
                be one-hot encoded or continuous logits that will be converted
                to one-hot internally.
            preferences: Tuple of ``(pref_mean, pref_log_var)`` for the
                preference distribution.  Each has shape ``(obs_dim,)`` or
                ``(B, obs_dim)``.
            return_trajectories: If True, store and return the full latent
                trajectories and predicted observations.  Set to False to
                save memory when only EFE scores are needed.

        Returns:
            :class:`RolloutResult` containing total EFE, per-step EFE,
            per-step term breakdowns, and optionally full trajectories.
        """
        B, N, H, A = action_sequences.shape
        S = initial_state.shape[-1]
        device = initial_state.device
        dtype = torch.float32  # EFE always in fp32

        # -- Expand initial state for all N rollouts: (B, S) -> (B*N, S) --
        state = (
            initial_state
            .unsqueeze(1)
            .expand(B, N, S)
            .reshape(B * N, S)
            .to(dtype)
        )

        # -- Expand preferences for (B*N, ...) if they are (B, ...) --
        pref_mean, pref_lv = preferences
        if pref_mean.dim() == 2 and pref_mean.shape[0] == B:
            pref_mean = (
                pref_mean
                .unsqueeze(1)
                .expand(B, N, -1)
                .reshape(B * N, -1)
            )
            pref_lv = (
                pref_lv
                .unsqueeze(1)
                .expand(B, N, -1)
                .reshape(B * N, -1)
            )
        elif pref_mean.dim() == 1:
            # (obs_dim,) -- broadcast is handled by the EFE computer
            pass
        pref_mean = pref_mean.to(dtype)
        pref_lv = pref_lv.to(dtype)

        # -- Pre-compute discount factors --
        discount_factors = self._compute_discount_factors(H).to(device)

        # -- Allocators for per-step accumulators --
        per_step_efe = torch.zeros(B * N, H, device=device, dtype=dtype)
        per_step_pragmatic = torch.zeros(B * N, H, device=device, dtype=dtype)
        per_step_epistemic = torch.zeros(B * N, H, device=device, dtype=dtype)
        per_step_instrumental = torch.zeros(B * N, H, device=device, dtype=dtype)

        if return_trajectories:
            O = self.config.obs_dim
            trajectories = torch.zeros(B * N, H, S, device=device, dtype=dtype)
            predicted_obs_all = torch.zeros(B * N, H, O, device=device, dtype=dtype)
        else:
            trajectories = torch.empty(0)
            predicted_obs_all = torch.empty(0)

        # -- Rollout loop over horizon --
        for t in range(H):
            # Extract action for this step: (B, N, A) -> (B*N, A)
            action_t = action_sequences[:, :, t, :].reshape(B * N, A).to(dtype)

            # 1. Transition: predict next state distribution
            prior_mean, prior_log_var = self.transition_model(state, action_t)
            prior_mean = prior_mean.float()
            prior_log_var = prior_log_var.float()

            # 2. Sample next state via reparameterisation
            std = torch.exp(0.5 * prior_log_var)
            eps = torch.randn_like(std)
            next_state = prior_mean + std * eps

            # 3. Decode to observation space
            obs_mean, obs_log_var = self.likelihood_model(next_state)
            obs_mean = obs_mean.float()
            obs_log_var = obs_log_var.float()

            # 4. Posterior approximation
            # In a full implementation the encoder would re-encode the
            # predicted observation.  Here we use the prior as a proxy
            # (consistent with the agent template convention).
            posterior_mean = prior_mean
            posterior_log_var = prior_log_var

            # 5. Compute per-step EFE and term breakdown
            # Prepare action for EFE computer (may expect integer indices)
            if self.config.discrete_actions:
                action_for_efe = action_t.argmax(dim=-1)  # (B*N,)
            else:
                action_for_efe = action_t

            efe_step, terms = self.efe_computer.compute_single_step(
                predicted_obs_params=(obs_mean, obs_log_var),
                posterior_params=(posterior_mean, posterior_log_var),
                prior_params=(prior_mean, prior_log_var),
                state=state,
                action=action_for_efe,
                next_state=next_state,
                preference_params=(pref_mean, pref_lv),
            )

            # 6. Store per-step values
            per_step_efe[:, t] = efe_step.float()
            per_step_pragmatic[:, t] = terms.get("pragmatic", torch.zeros_like(efe_step)).float()
            per_step_epistemic[:, t] = terms.get("epistemic", torch.zeros_like(efe_step)).float()
            per_step_instrumental[:, t] = terms.get("instrumental", torch.zeros_like(efe_step)).float()

            if return_trajectories:
                trajectories[:, t, :] = next_state
                predicted_obs_all[:, t, :] = obs_mean

            # 7. Advance state
            state = next_state

        # -- Compute total discounted EFE per sequence --
        # per_step_efe: (B*N, H), discount_factors: (H,)
        discounted_efe = per_step_efe * discount_factors.unsqueeze(0)
        total_efe = discounted_efe.sum(dim=-1)  # (B*N,)

        # -- Horizon normalisation --
        if self.config.normalize_by_horizon and H > 0:
            eff_horizon = self._effective_horizon(H)
            total_efe = total_efe / eff_horizon

        # -- Reshape everything back to (B, N, ...) --
        total_efe_bn = total_efe.view(B, N)
        per_step_efe_bnh = per_step_efe.view(B, N, H)

        per_step_terms = {
            "pragmatic": per_step_pragmatic.view(B, N, H),
            "epistemic": per_step_epistemic.view(B, N, H),
            "instrumental": per_step_instrumental.view(B, N, H),
        }

        if return_trajectories:
            traj_out = trajectories.view(B, N, H, S)
            obs_out = predicted_obs_all.view(B, N, H, self.config.obs_dim)
        else:
            traj_out = torch.empty(B, N, 0, S, device=device)
            obs_out = torch.empty(B, N, 0, self.config.obs_dim, device=device)

        return RolloutResult(
            total_efe=total_efe_bn,
            per_step_efe=per_step_efe_bnh,
            per_step_terms=per_step_terms,
            trajectories=traj_out,
            predicted_obs=obs_out,
            discount_factors=discount_factors,
        )

    def forward(
        self,
        initial_state: torch.Tensor,
        action_sequences: torch.Tensor,
        preferences: Tuple[torch.Tensor, torch.Tensor],
    ) -> RolloutResult:
        """Standard nn.Module forward, delegates to :meth:`rollout`.

        Provided so that the engine can participate in ``nn.Module`` graphs
        (e.g. for parameter iteration and device placement).

        Args:
            Same as :meth:`rollout`.

        Returns:
            Same as :meth:`rollout`.
        """
        return self.rollout(initial_state, action_sequences, preferences)


# ============================================================================
# Abstract base planner
# ============================================================================


class Planner(ABC):
    """Abstract base class for planning modules.

    A planner selects actions by generating candidate action sequences,
    evaluating them via a :class:`RolloutEngine`, and returning the best
    action (or a distribution over actions) as a :class:`PlanResult`.

    Subclasses must implement :meth:`plan` and :meth:`reset`.
    """

    @abstractmethod
    def plan(
        self,
        initial_state: torch.Tensor,
        observation: torch.Tensor,
        preferences: Tuple[torch.Tensor, torch.Tensor],
        ctx: Optional[Dict[str, Any]] = None,
    ) -> PlanResult:
        """Select an action by planning from the current state.

        Args:
            initial_state: Current latent state, shape ``(B, state_dim)``.
            observation: Current raw observation, shape ``(B, obs_dim)``.
                May be used by planners that condition proposals on the
                observation (e.g. amortised proposals).
            preferences: Tuple ``(pref_mean, pref_log_var)`` for the
                preference distribution.
            ctx: Optional context dictionary.  May contain keys such as
                ``"step"`` (global step counter), ``"temperature"`` (override
                for action_temperature), ``"generator"`` (torch.Generator
                for reproducibility), etc.

        Returns:
            :class:`PlanResult` with the selected action, full sequence,
            EFE breakdown, and diagnostics.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset planner state between episodes.

        Should clear any warm-start buffers, CEM distributions, or other
        persistent state.
        """
        ...


# ============================================================================
# Action sampling utilities
# ============================================================================


def _sample_continuous_actions(
    batch_size: int,
    num_rollouts: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    noise_std: float = 1.0,
    action_bounds: Optional[Tuple[float, float]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample continuous action sequences from a Gaussian prior.

    Args:
        batch_size: B.
        num_rollouts: N.
        horizon: H.
        action_dim: A.
        device: Target device.
        dtype: Target dtype.
        noise_std: Standard deviation of the sampling distribution.
        action_bounds: Optional ``(lower, upper)`` for clamping.
        generator: Optional RNG for reproducibility.

    Returns:
        Action sequences of shape ``(B, N, H, A)``, optionally clamped.
    """
    actions = torch.randn(
        batch_size, num_rollouts, horizon, action_dim,
        device=device, dtype=dtype, generator=generator,
    ) * noise_std

    if action_bounds is not None:
        lower, upper = action_bounds
        actions = actions.clamp(lower, upper)
    else:
        # Default: tanh squash to [-1, 1]
        actions = torch.tanh(actions)

    return actions


def _sample_discrete_actions(
    batch_size: int,
    num_rollouts: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample discrete action sequences as one-hot tensors.

    Args:
        batch_size: B.
        num_rollouts: N.
        horizon: H.
        action_dim: A (number of discrete categories).
        device: Target device.
        generator: Optional RNG for reproducibility.

    Returns:
        One-hot action sequences of shape ``(B, N, H, A)``, dtype float32.
    """
    indices = torch.randint(
        0, action_dim, (batch_size, num_rollouts, horizon),
        device=device, generator=generator,
    )
    return F.one_hot(indices, action_dim).float()


def _select_best_from_efe(
    efe_per_sequence: torch.Tensor,
    action_sequences: torch.Tensor,
    temperature: float = 1.0,
    discrete_actions: bool = False,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the best action and sequence given EFE scores.

    With ``deterministic=True``, selects the argmin.  Otherwise, samples
    from a softmax over negative EFE (so lower EFE = higher probability).

    Args:
        efe_per_sequence: ``(B, N)`` EFE scores.  Lower is better.
        action_sequences: ``(B, N, H, A)`` candidate sequences.
        temperature: Softmax temperature.
        discrete_actions: If True, convert one-hot to integer index for
            the returned best_action.
        deterministic: If True, use argmin instead of sampling.

    Returns:
        best_action: ``(B, A)`` or ``(B,)`` first action of the selected seq.
        best_sequence: ``(B, H, A)`` or ``(B, H)`` full selected sequence.
        selected_idx: ``(B,)`` index of selected sequence in dim 1.
    """
    B, N = efe_per_sequence.shape
    device = efe_per_sequence.device

    if deterministic:
        selected_idx = efe_per_sequence.argmin(dim=-1)  # (B,)
    else:
        logits = -efe_per_sequence / max(temperature, 1e-8)
        probs = F.softmax(logits, dim=-1)  # (B, N)
        selected_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)

    batch_idx = torch.arange(B, device=device)

    # Extract best sequence: (B, H, A)
    best_sequence = action_sequences[batch_idx, selected_idx]

    # Extract first action: (B, A)
    best_action = best_sequence[:, 0, :]

    if discrete_actions:
        best_action = best_action.argmax(dim=-1)  # (B,)
        best_sequence = best_sequence.argmax(dim=-1)  # (B, H)

    return best_action, best_sequence, selected_idx


def _gather_efe_terms_for_selected(
    rollout_result: RolloutResult,
    selected_idx: torch.Tensor,
    config: PlannerConfig,
) -> Dict[str, torch.Tensor]:
    """Extract per-term EFE for the selected rollout indices.

    Sums the per-step terms (with discount) for each selected sequence
    and returns a ``{term_name: (B,)}`` dictionary.

    Args:
        rollout_result: The full rollout result.
        selected_idx: ``(B,)`` indices of selected sequences.
        config: Planner configuration (for discount/normalisation).

    Returns:
        Dictionary with ``"pragmatic"``, ``"epistemic"``, ``"instrumental"``
        keys, each mapping to a ``(B,)`` tensor.
    """
    B = selected_idx.shape[0]
    device = selected_idx.device
    batch_idx = torch.arange(B, device=device)

    H = rollout_result.per_step_efe.shape[2]
    discounts = rollout_result.discount_factors[:H].to(device)  # (H,)

    terms: Dict[str, torch.Tensor] = {}
    for name, val in rollout_result.per_step_terms.items():
        # val: (B, N, H) -> select: (B, H)
        selected_steps = val[batch_idx, selected_idx]  # (B, H)
        # Apply discounting and sum over horizon
        discounted = (selected_steps * discounts.unsqueeze(0)).sum(dim=-1)  # (B,)
        if config.normalize_by_horizon and H > 0:
            gamma = config.discount_factor
            if abs(gamma - 1.0) < 1e-8:
                eff = float(H)
            else:
                eff = (1.0 - gamma ** H) / (1.0 - gamma)
            discounted = discounted / eff
        terms[name] = discounted

    return terms


# ============================================================================
# Random Shooting Planner
# ============================================================================


class RandomShootingPlanner(Planner, nn.Module):
    """Random shooting planner for active inference.

    Generates N random action sequences from a fixed prior distribution
    (Gaussian for continuous, uniform-categorical for discrete), evaluates
    each via a single pass through the :class:`RolloutEngine`, and selects
    the sequence with the lowest total EFE.

    This is the simplest planning algorithm.  It has no iterative refinement
    and its quality scales linearly with N.  For low-dimensional action
    spaces and short horizons, it can be surprisingly effective and is
    useful as a baseline.

    Args:
        rollout_engine: :class:`RolloutEngine` for evaluating sequences.
        config: :class:`PlannerConfig` with ``planner_type="random_shooting"``.

    Example::

        planner = RandomShootingPlanner(engine, config)
        result = planner.plan(
            initial_state=torch.randn(4, 256),
            observation=torch.randn(4, 4096),
            preferences=(pref_mean, pref_lv),
        )
        print(result.best_action.shape)  # (4, 10) or (4,)
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        config: PlannerConfig,
    ) -> None:
        nn.Module.__init__(self)
        self.rollout_engine = rollout_engine
        self.config = config
        self._generator: Optional[torch.Generator] = None

        if config.seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(config.seed)

    def _sample_action_sequences(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate N random action sequences.

        Args:
            batch_size: B.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Action sequences of shape ``(B, N, H, A)``.
        """
        N = self.config.num_rollouts
        H = self.config.planning_horizon
        A = self.config.action_dim

        if self.config.discrete_actions:
            return _sample_discrete_actions(
                batch_size, N, H, A, device,
                generator=self._generator,
            )
        else:
            return _sample_continuous_actions(
                batch_size, N, H, A, device, dtype,
                noise_std=self.config.action_noise_std,
                action_bounds=self.config.action_bounds,
                generator=self._generator,
            )

    def plan(
        self,
        initial_state: torch.Tensor,
        observation: torch.Tensor,
        preferences: Tuple[torch.Tensor, torch.Tensor],
        ctx: Optional[Dict[str, Any]] = None,
    ) -> PlanResult:
        """Plan via random shooting.

        Samples N action sequences, rolls them out, and selects the one
        with the lowest total EFE.

        Args:
            initial_state: ``(B, state_dim)``.
            observation: ``(B, obs_dim)`` -- not used by random shooting
                but accepted for interface compliance.
            preferences: ``(pref_mean, pref_log_var)`` tuple.
            ctx: Optional context.  Recognised keys:
                - ``"deterministic"`` (bool): If True, select argmin instead
                  of sampling.
                - ``"generator"`` (torch.Generator): Override the internal RNG.

        Returns:
            :class:`PlanResult`.
        """
        ctx = ctx or {}
        t_start = time.monotonic()

        B = initial_state.shape[0]
        device = initial_state.device
        dtype = initial_state.dtype
        deterministic = ctx.get("deterministic", False)

        # Override generator if provided
        gen_backup = self._generator
        if "generator" in ctx:
            self._generator = ctx["generator"]

        # 1. Sample action sequences
        action_sequences = self._sample_action_sequences(B, device, dtype)

        # Restore generator
        self._generator = gen_backup

        # 2. Rollout and evaluate
        rollout_result = self.rollout_engine.rollout(
            initial_state=initial_state,
            action_sequences=action_sequences,
            preferences=preferences,
        )

        # 3. Select best
        temperature = ctx.get("temperature", self.config.action_temperature)
        best_action, best_sequence, selected_idx = _select_best_from_efe(
            efe_per_sequence=rollout_result.total_efe,
            action_sequences=action_sequences,
            temperature=temperature,
            discrete_actions=self.config.discrete_actions,
            deterministic=deterministic,
        )

        # 4. Extract per-term breakdown for selected sequence
        efe_terms = _gather_efe_terms_for_selected(
            rollout_result, selected_idx, self.config,
        )

        # 5. Extract total EFE for selected
        batch_idx = torch.arange(B, device=device)
        efe_total = rollout_result.total_efe[batch_idx, selected_idx]

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        debug = {
            "elapsed_ms": elapsed_ms,
            "n_rollouts": self.config.num_rollouts,
            "planner_type": "random_shooting",
            "mean_efe": rollout_result.total_efe.mean().item(),
            "std_efe": rollout_result.total_efe.std().item(),
            "best_idx": selected_idx.tolist(),
        }

        if self.config.log_level >= 1:
            logger.info(
                "RandomShooting: N=%d H=%d best_efe=%.4f mean_efe=%.4f (%.1f ms)",
                self.config.num_rollouts,
                self.config.planning_horizon,
                efe_total.mean().item(),
                debug["mean_efe"],
                elapsed_ms,
            )

        return PlanResult(
            best_action=best_action,
            best_action_sequence=best_sequence,
            efe_total=efe_total,
            efe_terms=efe_terms,
            all_efe=rollout_result.total_efe,
            debug=debug,
        )

    def reset(self) -> None:
        """Reset planner state.  Random shooting is stateless."""
        if self.config.seed is not None and self._generator is not None:
            self._generator.manual_seed(self.config.seed)


# ============================================================================
# Cross-Entropy Method (CEM) Planner
# ============================================================================


class CEMPlanner(Planner, nn.Module):
    """Cross-Entropy Method planner for active inference.

    Iteratively refines a Gaussian sampling distribution over action
    sequences.  Each iteration:
        1. Sample N sequences from ``N(mu, sigma^2)``.
        2. Evaluate via rollout and compute EFE.
        3. Select the elite fraction (lowest EFE).
        4. Refit ``(mu, sigma)`` to the elite subset.
        5. Optionally blend with the previous distribution (momentum).
        6. Optionally anneal the temperature.

    After the final iteration, select the best sequence and return its
    first action.

    CEM converges to higher-quality plans than random shooting for the
    same total evaluation budget (N * cem_iterations), especially in
    high-dimensional continuous action spaces and long horizons.

    Warm-starting: When ``config.use_warm_start`` is True, the planner
    retains the CEM distribution from the previous call and shifts it
    by one time-step (dropping the first step, appending a fresh prior
    for the last step).  This dramatically reduces the number of
    iterations needed in sequential decision-making.

    Args:
        rollout_engine: :class:`RolloutEngine` for evaluating sequences.
        config: :class:`PlannerConfig` with ``planner_type="cem"``.

    Example::

        planner = CEMPlanner(engine, config)
        result = planner.plan(
            initial_state=torch.randn(4, 256),
            observation=torch.randn(4, 4096),
            preferences=(pref_mean, pref_lv),
        )
        print(result.efe_total.shape)  # (4,)
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        config: PlannerConfig,
    ) -> None:
        nn.Module.__init__(self)
        self.rollout_engine = rollout_engine
        self.config = config
        self._generator: Optional[torch.Generator] = None

        if config.seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(config.seed)

        # Warm-start buffers
        self._cem_mu: Optional[torch.Tensor] = None   # (B, H, A)
        self._cem_std: Optional[torch.Tensor] = None   # (B, H, A)
        self._prev_batch_size: int = 0

    # ------------------------------------------------------------------
    # Distribution management
    # ------------------------------------------------------------------

    def _init_distribution(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialise or warm-start the CEM distribution.

        When warm-starting, shifts the previous distribution by one step:
        the first row is dropped, and a fresh prior row is appended.

        Args:
            batch_size: B.
            device: Target device.
            dtype: Target dtype.

        Returns:
            mu: ``(B, H, A)`` mean of the sampling distribution.
            std: ``(B, H, A)`` std of the sampling distribution.
        """
        H = self.config.planning_horizon
        A = self.config.action_dim

        # Check warm-start feasibility
        can_warm_start = (
            self.config.use_warm_start
            and self._cem_mu is not None
            and self._cem_std is not None
            and self._prev_batch_size == batch_size
            and self._cem_mu.shape == (batch_size, H, A)
        )

        if can_warm_start:
            # Shift by one time-step: drop first, append fresh
            mu = torch.cat([
                self._cem_mu[:, 1:, :],
                torch.zeros(batch_size, 1, A, device=device, dtype=dtype),
            ], dim=1)
            std = torch.cat([
                self._cem_std[:, 1:, :],
                torch.ones(batch_size, 1, A, device=device, dtype=dtype)
                * self.config.action_noise_std,
            ], dim=1)
            if self.config.log_level >= 2:
                logger.debug("CEM: warm-starting from previous distribution")
        else:
            mu = torch.zeros(batch_size, H, A, device=device, dtype=dtype)
            std = (
                torch.ones(batch_size, H, A, device=device, dtype=dtype)
                * self.config.action_noise_std
            )

        return mu, std

    def _sample_from_distribution(
        self,
        mu: torch.Tensor,
        std: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Sample action sequences from the CEM Gaussian distribution.

        Args:
            mu: Distribution mean, ``(B, H, A)``.
            std: Distribution std, ``(B, H, A)``.
            num_samples: N (number of sequences to sample).

        Returns:
            Action sequences of shape ``(B, N, H, A)``.
        """
        B, H, A = mu.shape
        device = mu.device
        dtype = mu.dtype

        # Expand for sampling: (B, 1, H, A) + (B, 1, H, A) * (B, N, H, A)
        noise = torch.randn(
            B, num_samples, H, A,
            device=device, dtype=dtype,
            generator=self._generator,
        )

        samples = mu.unsqueeze(1) + std.unsqueeze(1) * noise

        # Apply action bounds
        if self.config.discrete_actions:
            # Convert to one-hot for discrete actions
            samples = F.one_hot(samples.argmax(dim=-1), A).float()
        elif self.config.action_bounds is not None:
            lower, upper = self.config.action_bounds
            samples = samples.clamp(lower, upper)
        else:
            samples = torch.tanh(samples)

        return samples

    def _refit_distribution(
        self,
        elite_sequences: torch.Tensor,
        old_mu: torch.Tensor,
        old_std: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refit the CEM distribution to the elite subset.

        Applies momentum blending between the old and new distribution
        parameters, and enforces a minimum standard deviation.

        When ``n_elite == 1``, the standard deviation cannot be estimated
        from the sample, so the old standard deviation (scaled down) is
        used as a fallback.

        Args:
            elite_sequences: ``(B, n_elite, H, A)`` elite action sequences.
            old_mu: Previous distribution mean, ``(B, H, A)``.
            old_std: Previous distribution std, ``(B, H, A)``.

        Returns:
            new_mu: Updated mean, ``(B, H, A)``.
            new_std: Updated std, ``(B, H, A)``.
        """
        n_elite = elite_sequences.shape[1]

        # Compute elite statistics
        new_mu = elite_sequences.mean(dim=1)    # (B, H, A)

        if n_elite > 1:
            new_std = elite_sequences.std(dim=1)    # (B, H, A)
        else:
            # Cannot compute std from a single sample; shrink old std
            new_std = old_std * 0.5

        # Enforce minimum std
        new_std = new_std.clamp(min=self.config.cem_min_std)

        # Momentum blending
        alpha = self.config.cem_momentum
        if alpha > 0.0:
            new_mu = alpha * old_mu + (1.0 - alpha) * new_mu
            new_std = alpha * old_std + (1.0 - alpha) * new_std

        return new_mu, new_std

    # ------------------------------------------------------------------
    # Main planning loop
    # ------------------------------------------------------------------

    def plan(
        self,
        initial_state: torch.Tensor,
        observation: torch.Tensor,
        preferences: Tuple[torch.Tensor, torch.Tensor],
        ctx: Optional[Dict[str, Any]] = None,
    ) -> PlanResult:
        """Plan via the Cross-Entropy Method.

        Runs ``cem_iterations`` rounds of sample-evaluate-refit, then
        selects the best action from the final iteration.

        Args:
            initial_state: ``(B, state_dim)``.
            observation: ``(B, obs_dim)`` -- available for future extensions
                (e.g. observation-conditioned proposals).
            preferences: ``(pref_mean, pref_log_var)`` tuple.
            ctx: Optional context.  Recognised keys:
                - ``"deterministic"`` (bool): If True, select argmin.
                - ``"cem_iterations"`` (int): Override number of iterations.
                - ``"generator"`` (torch.Generator): Override RNG.
                - ``"temperature"`` (float): Override action temperature.

        Returns:
            :class:`PlanResult` with the selected action, EFE breakdown,
            and CEM-specific diagnostics in the debug dict.
        """
        ctx = ctx or {}
        t_start = time.monotonic()

        B = initial_state.shape[0]
        N = self.config.num_rollouts
        H = self.config.planning_horizon
        A = self.config.action_dim
        device = initial_state.device
        dtype = initial_state.dtype
        deterministic = ctx.get("deterministic", False)
        n_iterations = ctx.get("cem_iterations", self.config.cem_iterations)
        n_elite = max(1, int(N * self.config.cem_elite_fraction))

        # Override generator if provided
        gen_backup = self._generator
        if "generator" in ctx:
            self._generator = ctx["generator"]

        # Initialise CEM distribution
        mu, std = self._init_distribution(B, device, dtype)

        # Track iteration-level diagnostics
        iteration_efe_means: List[float] = []
        iteration_elite_means: List[float] = []
        iteration_stds: List[float] = []

        best_efe_global: Optional[torch.Tensor] = None
        best_action_global: Optional[torch.Tensor] = None
        best_sequence_global: Optional[torch.Tensor] = None
        best_terms_global: Optional[Dict[str, torch.Tensor]] = None
        best_all_efe: Optional[torch.Tensor] = None
        best_selected_idx: Optional[torch.Tensor] = None

        for iteration in range(n_iterations):
            # -- 1. Sample action sequences --
            action_sequences = self._sample_from_distribution(mu, std, N)

            # -- 2. Rollout and evaluate --
            rollout_result = self.rollout_engine.rollout(
                initial_state=initial_state,
                action_sequences=action_sequences,
                preferences=preferences,
            )

            efe_scores = rollout_result.total_efe  # (B, N)

            # -- 3. Select elite --
            _, elite_idx = efe_scores.topk(n_elite, dim=-1, largest=False)

            # Gather elite sequences: (B, n_elite, H, A)
            elite_seqs = torch.gather(
                action_sequences,
                dim=1,
                index=elite_idx.unsqueeze(-1).unsqueeze(-1).expand(
                    B, n_elite, H, A
                ),
            )

            # Elite EFE for diagnostics
            elite_efe = torch.gather(efe_scores, dim=1, index=elite_idx)  # (B, n_elite)

            # -- 4. Refit distribution --
            mu, std = self._refit_distribution(elite_seqs, mu, std)

            # -- 5. Temperature annealing --
            # Linearly decay temperature over iterations
            if n_iterations > 1:
                progress = iteration / (n_iterations - 1)
                current_temp = self.config.cem_temperature * (1.0 - 0.5 * progress)
            else:
                current_temp = self.config.cem_temperature

            # -- Diagnostics --
            iteration_efe_means.append(efe_scores.mean().item())
            iteration_elite_means.append(elite_efe.mean().item())
            iteration_stds.append(std.mean().item())

            if self.config.log_level >= 2:
                logger.debug(
                    "CEM iter %d/%d: mean_efe=%.4f elite_efe=%.4f mean_std=%.4f",
                    iteration + 1,
                    n_iterations,
                    iteration_efe_means[-1],
                    iteration_elite_means[-1],
                    iteration_stds[-1],
                )

            # -- Track global best across iterations --
            temperature = ctx.get("temperature", self.config.action_temperature)

            _, _, selected_idx = _select_best_from_efe(
                efe_per_sequence=efe_scores,
                action_sequences=action_sequences,
                temperature=temperature,
                discrete_actions=self.config.discrete_actions,
                deterministic=True,  # For tracking, always use argmin
            )

            batch_idx = torch.arange(B, device=device)
            current_best_efe = efe_scores[batch_idx, selected_idx]

            if best_efe_global is None or (current_best_efe < best_efe_global).all():
                best_efe_global = current_best_efe
                best_all_efe = efe_scores
                best_selected_idx = selected_idx

                # Extract action and sequence for best
                best_action_tmp = action_sequences[batch_idx, selected_idx, 0, :]
                best_sequence_tmp = action_sequences[batch_idx, selected_idx]

                if self.config.discrete_actions:
                    best_action_global = best_action_tmp.argmax(dim=-1)
                    best_sequence_global = best_sequence_tmp.argmax(dim=-1)
                else:
                    best_action_global = best_action_tmp
                    best_sequence_global = best_sequence_tmp

                # Extract terms for selected
                best_terms_global = _gather_efe_terms_for_selected(
                    rollout_result, selected_idx, self.config,
                )

        # -- Final selection (from last iteration, may use stochastic) --
        # We re-do selection on the last iteration with the user's temperature
        if not deterministic and best_all_efe is not None:
            temperature = ctx.get("temperature", self.config.action_temperature)
            # Rebuild action_sequences from final distribution for final selection
            final_actions = self._sample_from_distribution(mu, std, N)
            final_result = self.rollout_engine.rollout(
                initial_state=initial_state,
                action_sequences=final_actions,
                preferences=preferences,
            )

            best_action, best_sequence, selected_idx = _select_best_from_efe(
                efe_per_sequence=final_result.total_efe,
                action_sequences=final_actions,
                temperature=temperature,
                discrete_actions=self.config.discrete_actions,
                deterministic=deterministic,
            )

            batch_idx = torch.arange(B, device=device)
            efe_total = final_result.total_efe[batch_idx, selected_idx]
            efe_terms = _gather_efe_terms_for_selected(
                final_result, selected_idx, self.config,
            )
            all_efe = final_result.total_efe

            best_action_global = best_action
            best_sequence_global = best_sequence
            best_efe_global = efe_total
            best_terms_global = efe_terms
            best_all_efe = all_efe

        # -- Save CEM distribution for warm-starting --
        self._cem_mu = mu.detach().clone()
        self._cem_std = std.detach().clone()
        self._prev_batch_size = B

        # Restore generator
        self._generator = gen_backup

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        debug: Dict[str, Any] = {
            "elapsed_ms": elapsed_ms,
            "n_rollouts": N,
            "n_iterations": n_iterations,
            "n_elite": n_elite,
            "planner_type": "cem",
            "iteration_efe_means": iteration_efe_means,
            "iteration_elite_means": iteration_elite_means,
            "iteration_stds": iteration_stds,
            "cem_final_mu_norm": mu.norm().item(),
            "cem_final_std_mean": std.mean().item(),
            "cem_final_std_min": std.min().item(),
            "cem_final_std_max": std.max().item(),
        }

        if self.config.log_level >= 1:
            logger.info(
                "CEM: N=%d H=%d iters=%d best_efe=%.4f final_std=%.4f (%.1f ms)",
                N,
                H,
                n_iterations,
                best_efe_global.mean().item() if best_efe_global is not None else float("nan"),
                std.mean().item(),
                elapsed_ms,
            )

        # Ensure we have valid outputs (handles edge case of 0 iterations)
        if best_action_global is None:
            best_action_global = torch.zeros(
                B, A, device=device, dtype=dtype
            )
        if best_sequence_global is None:
            best_sequence_global = torch.zeros(
                B, H, A, device=device, dtype=dtype
            )
        if best_efe_global is None:
            best_efe_global = torch.zeros(B, device=device, dtype=dtype)
        if best_terms_global is None:
            best_terms_global = {
                "pragmatic": torch.zeros(B, device=device),
                "epistemic": torch.zeros(B, device=device),
                "instrumental": torch.zeros(B, device=device),
            }
        if best_all_efe is None:
            best_all_efe = torch.zeros(B, N, device=device, dtype=dtype)

        return PlanResult(
            best_action=best_action_global,
            best_action_sequence=best_sequence_global,
            efe_total=best_efe_global,
            efe_terms=best_terms_global,
            all_efe=best_all_efe,
            debug=debug,
        )

    def reset(self) -> None:
        """Reset CEM warm-start buffers and RNG."""
        self._cem_mu = None
        self._cem_std = None
        self._prev_batch_size = 0

        if self.config.seed is not None and self._generator is not None:
            self._generator.manual_seed(self.config.seed)


# ============================================================================
# Planner registry and factory
# ============================================================================

_PLANNER_REGISTRY: Dict[str, type] = {
    "random_shooting": RandomShootingPlanner,
    "cem": CEMPlanner,
}


def register_planner(name: str, planner_class: type) -> None:
    """Register a custom planner type in the global registry.

    Args:
        name: String identifier for the planner (used in PlannerConfig.planner_type).
        planner_class: A class that inherits from :class:`Planner`.

    Raises:
        TypeError: If ``planner_class`` does not inherit from :class:`Planner`.
    """
    if not (isinstance(planner_class, type) and issubclass(planner_class, Planner)):
        raise TypeError(
            f"planner_class must inherit from Planner, got {planner_class}"
        )
    _PLANNER_REGISTRY[name] = planner_class
    logger.debug("Registered planner '%s' -> %s", name, planner_class.__name__)


def create_planner(
    config: PlannerConfig,
    transition_model: nn.Module,
    likelihood_model: nn.Module,
    efe_computer: nn.Module,
    **kwargs: Any,
) -> Planner:
    """Factory function to create a planner from configuration.

    Builds a :class:`RolloutEngine` and wraps it in the appropriate planner
    class based on ``config.planner_type``.

    Args:
        config: :class:`PlannerConfig` specifying the planner type and
            hyperparameters.
        transition_model: Module implementing :class:`TransitionModelProtocol`.
        likelihood_model: Module implementing :class:`LikelihoodModelProtocol`.
        efe_computer: Module implementing :class:`EFEComputerProtocol`.
        **kwargs: Additional keyword arguments passed to the planner constructor.

    Returns:
        A :class:`Planner` instance (either :class:`RandomShootingPlanner`
        or :class:`CEMPlanner`, depending on ``config.planner_type``).

    Raises:
        ValueError: If ``config.planner_type`` is not registered.
        ValueError: If ``config.action_dim`` or ``config.state_dim`` are
            invalid.

    Example::

        config = PlannerConfig(planner_type="cem", num_rollouts=256)
        planner = create_planner(config, transition, likelihood, efe)
        result = planner.plan(state, obs, preferences)
    """
    if config.action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {config.action_dim}")
    if config.state_dim <= 0:
        raise ValueError(f"state_dim must be positive, got {config.state_dim}")
    if config.planning_horizon <= 0:
        raise ValueError(
            f"planning_horizon must be positive, got {config.planning_horizon}"
        )
    if config.num_rollouts <= 0:
        raise ValueError(f"num_rollouts must be positive, got {config.num_rollouts}")

    planner_type = config.planner_type.lower().strip()

    if planner_type not in _PLANNER_REGISTRY:
        available = ", ".join(sorted(_PLANNER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown planner_type '{planner_type}'. "
            f"Available: {available}. "
            f"Use register_planner() to add custom types."
        )

    # Build the rollout engine
    engine = RolloutEngine(
        transition_model=transition_model,
        likelihood_model=likelihood_model,
        efe_computer=efe_computer,
        config=config,
    )

    # Instantiate the planner
    planner_cls = _PLANNER_REGISTRY[planner_type]
    planner = planner_cls(rollout_engine=engine, config=config, **kwargs)

    logger.info(
        "Created planner '%s' with N=%d, H=%d, A=%d, S=%d",
        planner_type,
        config.num_rollouts,
        config.planning_horizon,
        config.action_dim,
        config.state_dim,
    )

    return planner


# ============================================================================
# Convenience presets
# ============================================================================


def minimal_config() -> PlannerConfig:
    """Tiny planner configuration for unit tests (~minimal compute).

    Returns:
        :class:`PlannerConfig` with small dimensions and few rollouts.
    """
    return PlannerConfig(
        planner_type="random_shooting",
        planning_horizon=2,
        num_rollouts=8,
        action_dim=4,
        state_dim=16,
        obs_dim=32,
        discrete_actions=False,
        action_noise_std=0.5,
        seed=42,
    )


def standard_config() -> PlannerConfig:
    """Standard planner configuration for development.

    Returns:
        :class:`PlannerConfig` with CEM, moderate dimensions.
    """
    return PlannerConfig(
        planner_type="cem",
        planning_horizon=8,
        num_rollouts=128,
        cem_iterations=5,
        cem_elite_fraction=0.1,
        cem_temperature=1.0,
        cem_momentum=0.1,
        action_dim=10,
        state_dim=256,
        obs_dim=4096,
        discrete_actions=False,
        seed=None,
    )


def production_config() -> PlannerConfig:
    """Production planner configuration for deployment.

    Returns:
        :class:`PlannerConfig` with CEM, large N and more iterations.
    """
    return PlannerConfig(
        planner_type="cem",
        planning_horizon=12,
        num_rollouts=512,
        cem_iterations=8,
        cem_elite_fraction=0.05,
        cem_temperature=0.5,
        cem_momentum=0.2,
        cem_min_std=0.005,
        action_dim=128,
        state_dim=256,
        obs_dim=4096,
        discrete_actions=False,
        action_bounds=(-1.0, 1.0),
        seed=None,
    )


# ============================================================================
# Stub models for testing
# ============================================================================


class _StubTransitionModel(nn.Module):
    """Minimal transition model for self-tests.

    Implements a simple linear transition: next_state = W * [state; action] + b,
    with a fixed log-variance.  Deterministic given weights.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(state_dim + action_dim, state_dim)
        self.log_var = nn.Parameter(torch.zeros(state_dim) - 2.0)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([state, action], dim=-1)
        mean = self.net(combined)
        log_var = self.log_var.expand_as(mean)
        return mean, log_var


class _StubLikelihoodModel(nn.Module):
    """Minimal likelihood model for self-tests.

    Projects latent state to observation space with a fixed log-variance.
    """

    def __init__(self, state_dim: int, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(state_dim, obs_dim)
        self.log_var = nn.Parameter(torch.zeros(obs_dim) - 1.0)

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(state)
        log_var = self.log_var.expand_as(mean)
        return mean, log_var


class _StubEFEComputer(nn.Module):
    """Minimal EFE computer for self-tests.

    Returns simple squared-error between predicted observations and
    preferences as the pragmatic term, and constant-zero epistemic and
    instrumental terms.  This is sufficient to verify planner logic
    without requiring the full EFE decomposition.
    """

    def __init__(self) -> None:
        super().__init__()

    def compute_single_step(
        self,
        predicted_obs_params: Tuple[torch.Tensor, torch.Tensor],
        posterior_params: Tuple[torch.Tensor, torch.Tensor],
        prior_params: Tuple[torch.Tensor, torch.Tensor],
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        preference_params: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute a simplified EFE for testing purposes.

        Returns:
            efe: ``(B,)`` total EFE (pragmatic only).
            terms: dict with ``"pragmatic"``, ``"epistemic"``, ``"instrumental"``.
        """
        pred_mean, pred_log_var = predicted_obs_params
        pref_mean, pref_lv = preference_params

        # Broadcast preference mean
        if pref_mean.dim() == 1:
            pref_mean = pref_mean.unsqueeze(0).expand_as(pred_mean)
        if pref_lv.dim() == 1:
            pref_lv = pref_lv.unsqueeze(0).expand_as(pred_mean)

        # Simple pragmatic: precision-weighted squared error (precision = exp(-log_var))
        pref_precision = torch.exp(-pref_lv)
        pragmatic = (pref_precision * (pred_mean - pref_mean).pow(2)).sum(dim=-1)  # (B,)

        # Epistemic: KL proxy (simplified)
        post_mean, post_log_var = posterior_params
        prior_mean, prior_log_var = prior_params
        epistemic = 0.5 * (
            (prior_log_var - post_log_var)
            + (post_log_var.exp() + (post_mean - prior_mean).pow(2))
            / prior_log_var.exp().clamp(min=1e-8)
            - 1.0
        ).sum(dim=-1)

        instrumental = torch.zeros_like(pragmatic)

        total = pragmatic + epistemic - instrumental

        return total, {
            "pragmatic": pragmatic,
            "epistemic": epistemic,
            "instrumental": instrumental,
            "total": total,
        }


def _build_stub_models(
    state_dim: int = 16,
    action_dim: int = 4,
    obs_dim: int = 32,
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Build stub models for testing.

    Args:
        state_dim: Latent state dimensionality.
        action_dim: Action dimensionality.
        obs_dim: Observation dimensionality.

    Returns:
        Tuple of (transition_model, likelihood_model, efe_computer).
    """
    transition = _StubTransitionModel(state_dim, action_dim)
    likelihood = _StubLikelihoodModel(state_dim, obs_dim)
    efe = _StubEFEComputer()
    return transition, likelihood, efe


# ============================================================================
# Self-test block
# ============================================================================


def _run_self_tests() -> None:
    """Run ~25 self-tests to validate all planner components.

    Tests cover:
        - RolloutEngine shape correctness and determinism
        - RandomShootingPlanner best-selection logic and output formats
        - CEMPlanner iterative improvement, elite selection, action bounds
        - Interface compliance (Planner ABC, PlanResult validation)
        - PlanResult dataclass validation
        - Factory function and error handling
        - Configuration presets
    """
    import traceback

    passed = 0
    failed = 0
    errors: List[str] = []

    def _test(name: str, fn: Callable[[], None]) -> None:
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            errors.append(f"  FAIL  {name}: {e}\n{tb}")
            print(f"  FAIL  {name}: {e}")

    print("=" * 72)
    print("Planner Module Self-Tests")
    print("=" * 72)

    torch.manual_seed(12345)

    # Shared test fixtures
    S, A, O = 16, 4, 32
    B, N, H = 2, 8, 3
    device = torch.device("cpu")

    transition, likelihood, efe_computer = _build_stub_models(S, A, O)

    config = PlannerConfig(
        planner_type="random_shooting",
        planning_horizon=H,
        num_rollouts=N,
        action_dim=A,
        state_dim=S,
        obs_dim=O,
        discount_factor=0.99,
        normalize_by_horizon=True,
        discrete_actions=False,
        action_noise_std=0.5,
        seed=42,
    )

    engine = RolloutEngine(transition, likelihood, efe_computer, config)

    pref_mean = torch.zeros(O)
    pref_lv = torch.zeros(O)  # log_var=0 means variance=1, precision=1
    preferences = (pref_mean, pref_lv)

    # ---------------------------------------------------------------
    # Test 1: RolloutEngine output shapes
    # ---------------------------------------------------------------
    def test_rollout_shapes():
        state = torch.randn(B, S)
        actions = torch.randn(B, N, H, A)
        result = engine.rollout(state, actions, preferences)
        assert result.total_efe.shape == (B, N), \
            f"total_efe shape {result.total_efe.shape} != ({B}, {N})"
        assert result.per_step_efe.shape == (B, N, H), \
            f"per_step_efe shape {result.per_step_efe.shape} != ({B}, {N}, {H})"
        assert result.trajectories.shape == (B, N, H, S), \
            f"trajectories shape {result.trajectories.shape} != ({B}, {N}, {H}, {S})"
        assert result.predicted_obs.shape == (B, N, H, O), \
            f"predicted_obs shape {result.predicted_obs.shape} != ({B}, {N}, {H}, {O})"
        assert result.discount_factors.shape == (H,), \
            f"discount_factors shape {result.discount_factors.shape} != ({H},)"
    _test("01_rollout_shapes", test_rollout_shapes)

    # ---------------------------------------------------------------
    # Test 2: RolloutEngine determinism with same seed
    # ---------------------------------------------------------------
    def test_rollout_determinism():
        torch.manual_seed(999)
        state = torch.randn(B, S)
        actions = torch.randn(B, N, H, A)
        torch.manual_seed(111)
        r1 = engine.rollout(state, actions, preferences)
        torch.manual_seed(111)
        r2 = engine.rollout(state, actions, preferences)
        assert torch.allclose(r1.total_efe, r2.total_efe, atol=1e-5), \
            "Rollout not deterministic with same seed"
    _test("02_rollout_determinism", test_rollout_determinism)

    # ---------------------------------------------------------------
    # Test 3: RolloutEngine per-step terms keys
    # ---------------------------------------------------------------
    def test_rollout_per_step_keys():
        state = torch.randn(B, S)
        actions = torch.randn(B, N, H, A)
        result = engine.rollout(state, actions, preferences)
        expected_keys = {"pragmatic", "epistemic", "instrumental"}
        assert set(result.per_step_terms.keys()) == expected_keys, \
            f"per_step_terms keys {set(result.per_step_terms.keys())} != {expected_keys}"
        for k, v in result.per_step_terms.items():
            assert v.shape == (B, N, H), f"per_step_terms['{k}'] shape {v.shape}"
    _test("03_rollout_per_step_keys", test_rollout_per_step_keys)

    # ---------------------------------------------------------------
    # Test 4: RolloutEngine discount factors correct
    # ---------------------------------------------------------------
    def test_discount_factors():
        factors = engine._compute_discount_factors(H)
        expected = torch.tensor([0.99**t for t in range(H)])
        assert torch.allclose(factors, expected, atol=1e-6), \
            "Discount factors incorrect"
    _test("04_discount_factors", test_discount_factors)

    # ---------------------------------------------------------------
    # Test 5: RolloutEngine effective horizon
    # ---------------------------------------------------------------
    def test_effective_horizon():
        eff = engine._effective_horizon(H)
        expected = sum(0.99**t for t in range(H))
        assert abs(eff - expected) < 1e-6, \
            f"Effective horizon {eff} != {expected}"
    _test("05_effective_horizon", test_effective_horizon)

    # ---------------------------------------------------------------
    # Test 6: RolloutEngine no-trajectory mode
    # ---------------------------------------------------------------
    def test_rollout_no_trajectories():
        state = torch.randn(B, S)
        actions = torch.randn(B, N, H, A)
        result = engine.rollout(state, actions, preferences, return_trajectories=False)
        assert result.total_efe.shape == (B, N)
        assert result.trajectories.shape[2] == 0, \
            "Trajectories should be empty when return_trajectories=False"
    _test("06_rollout_no_trajectories", test_rollout_no_trajectories)

    # ---------------------------------------------------------------
    # Test 7: RolloutResult best_indices
    # ---------------------------------------------------------------
    def test_rollout_result_best_indices():
        state = torch.randn(B, S)
        actions = torch.randn(B, N, H, A)
        result = engine.rollout(state, actions, preferences)
        best_idx = result.best_indices()
        assert best_idx.shape == (B,), f"best_indices shape {best_idx.shape}"
        for b in range(B):
            assert result.total_efe[b, best_idx[b]] == result.total_efe[b].min()
    _test("07_rollout_result_best_indices", test_rollout_result_best_indices)

    # ---------------------------------------------------------------
    # Test 8: RolloutResult best_efe
    # ---------------------------------------------------------------
    def test_rollout_result_best_efe():
        state = torch.randn(B, S)
        actions = torch.randn(B, N, H, A)
        result = engine.rollout(state, actions, preferences)
        best = result.best_efe()
        assert best.shape == (B,)
        for b in range(B):
            assert abs(best[b].item() - result.total_efe[b].min().item()) < 1e-6
    _test("08_rollout_result_best_efe", test_rollout_result_best_efe)

    # ---------------------------------------------------------------
    # Test 9: RandomShooting output shapes (continuous)
    # ---------------------------------------------------------------
    def test_random_shooting_shapes():
        planner = RandomShootingPlanner(engine, config)
        state = torch.randn(B, S)
        obs = torch.randn(B, O)
        result = planner.plan(state, obs, preferences, ctx={"deterministic": True})
        assert result.best_action.shape == (B, A), \
            f"best_action shape {result.best_action.shape}"
        assert result.best_action_sequence.shape == (B, H, A), \
            f"best_action_sequence shape {result.best_action_sequence.shape}"
        assert result.efe_total.shape == (B,), \
            f"efe_total shape {result.efe_total.shape}"
        assert result.all_efe.shape == (B, N), \
            f"all_efe shape {result.all_efe.shape}"
    _test("09_random_shooting_shapes", test_random_shooting_shapes)

    # ---------------------------------------------------------------
    # Test 10: RandomShooting best selection (argmin)
    # ---------------------------------------------------------------
    def test_random_shooting_best_selection():
        planner = RandomShootingPlanner(engine, config)
        state = torch.randn(B, S)
        obs = torch.randn(B, O)
        result = planner.plan(state, obs, preferences, ctx={"deterministic": True})
        for b in range(B):
            assert result.efe_total[b] <= result.all_efe[b].min() + 1e-5, \
                f"Selected EFE {result.efe_total[b]} > min {result.all_efe[b].min()}"
    _test("10_random_shooting_best_selection", test_random_shooting_best_selection)

    # ---------------------------------------------------------------
    # Test 11: RandomShooting discrete actions
    # ---------------------------------------------------------------
    def test_random_shooting_discrete():
        disc_config = PlannerConfig(
            planner_type="random_shooting",
            planning_horizon=H,
            num_rollouts=N,
            action_dim=A,
            state_dim=S,
            obs_dim=O,
            discrete_actions=True,
            seed=42,
        )
        disc_engine = RolloutEngine(transition, likelihood, efe_computer, disc_config)
        planner = RandomShootingPlanner(disc_engine, disc_config)
        state = torch.randn(B, S)
        obs = torch.randn(B, O)
        result = planner.plan(state, obs, preferences, ctx={"deterministic": True})
        assert result.best_action.shape == (B,), \
            f"Discrete best_action shape {result.best_action.shape} != ({B},)"
        assert result.best_action.dtype in (torch.int64, torch.long), \
            f"Discrete best_action dtype {result.best_action.dtype}"
        assert (result.best_action >= 0).all() and (result.best_action < A).all(), \
            "Discrete best_action out of range"
    _test("11_random_shooting_discrete", test_random_shooting_discrete)

    # ---------------------------------------------------------------
    # Test 12: RandomShooting PlanResult validation passes
    # ---------------------------------------------------------------
    def test_random_shooting_plan_result_valid():
        planner = RandomShootingPlanner(engine, config)
        state = torch.randn(B, S)
        obs = torch.randn(B, O)
        result = planner.plan(state, obs, preferences, ctx={"deterministic": True})
        result.validate()  # Should not raise
    _test("12_random_shooting_plan_result_valid", test_random_shooting_plan_result_valid)

    # ---------------------------------------------------------------
    # Test 13: RandomShooting efe_terms keys
    # ---------------------------------------------------------------
    def test_random_shooting_efe_terms():
        planner = RandomShootingPlanner(engine, config)
        state = torch.randn(B, S)
        obs = torch.randn(B, O)
        result = planner.plan(state, obs, preferences, ctx={"deterministic": True})
        assert "pragmatic" in result.efe_terms
        assert "epistemic" in result.efe_terms
        assert "instrumental" in result.efe_terms
        for k, v in result.efe_terms.items():
            assert v.shape == (B,), f"efe_terms['{k}'] shape {v.shape}"
    _test("13_random_shooting_efe_terms", test_random_shooting_efe_terms)

    # ---------------------------------------------------------------
    # Test 14: RandomShooting debug info
    # ---------------------------------------------------------------
    def test_random_shooting_debug():
        planner = RandomShootingPlanner(engine, config)
        state = torch.randn(B, S)
        obs = torch.randn(B, O)
        result = planner.plan(state, obs, preferences)
        assert "elapsed_ms" in result.debug
        assert result.debug["planner_type"] == "random_shooting"
        assert result.debug["n_rollouts"] == N
    _test("14_random_shooting_debug", test_random_shooting_debug)

    # ---------------------------------------------------------------
    # Test 15: CEM output shapes
    # ---------------------------------------------------------------
    def test_cem_shapes():
        cem_config = PlannerConfig(
            planner_type="cem",
            planning_horizon=H,
            num_rollouts=N,
            cem_iterations=3,
            cem_elite_fraction=0.25,
            action_dim=A,
            state_dim=S,
            obs_dim=O,
            seed=42,
            use_warm_start=False,
        )
        cem_engine = RolloutEngine(transition, likelihood, efe_computer, cem_config)
        planner = CEMPlanner(cem_engine, cem_config)
        state = torch.randn(B, S)
        obs = torch.randn(B, O)
        result = planner.plan(state, obs, preferences, ctx={"deterministic": True})
        assert result.best_action.shape == (B, A), \
            f"CEM best_action shape {result.best_action.shape}"
        assert result.efe_total.shape == (B,)
        assert result.all_efe.shape[0] == B
    _test("15_cem_shapes", test_cem_shapes)

    # ---------------------------------------------------------------
    # Test 16: CEM iterative improvement
    # ---------------------------------------------------------------
    def test_cem_improvement():
        cem_config = PlannerConfig(
            planner_type="cem",
            planning_horizon=H,
            num_rollouts=16,
            cem_iterations=5,
            cem_elite_fraction=0.25,
            action_dim=A,
            state_dim=S,
            obs_dim=O,
            seed=42,
            use_warm_start=False,
            cem_momentum=0.0,
        )
        cem_engine = RolloutEngine(transition, likelihood, efe_computer, cem_config)
        planner = CEMPlanner(cem_engine, cem_config)
        state = torch.randn(B, S)
        obs = torch.randn(B, O)
        result = planner.plan(state, obs, preferences, ctx={"deterministic": True})
        # Iteration EFE means should generally decrease (or at least not increase drastically)
        means = result.debug.get("iteration_efe_means", [])
        assert len(means) == 5, f"Expected 5 iteration means, got {len(means)}"
        # Check that the elite means decrease (more reliable than full means)
        elite_means = result.debug.get("iteration_elite_means", [])
        if len(elite_means) >= 2:
            # At least the last should be <= the first (with tolerance for noise)
            assert elite_means[-1] <= elite_means[0] + 1.0, \
                f"CEM elite mean did not improve: {elite_means[0]:.4f} -> {elite_means[-1]:.4f}"
    _test("16_cem_improvement", test_cem_improvement)

    # ---------------------------------------------------------------
    # Test 17: CEM elite count
    # ---------------------------------------------------------------
    def test_cem_elite_count():
        cem_config = PlannerConfig(
            planner_type="cem",
            planning_horizon=H,
            num_rollouts=20,
            cem_iterations=2,
            cem_elite_fraction=0.1,
            action_dim=A,
            state_dim=S,
            obs_dim=O,
            seed=42,
            use_warm_start=False,
        )
        cem_engine = RolloutEngine(transition, likelihood, efe_computer, cem_config)
        planner = CEMPlanner(cem_engine, cem_config)
        state = torch.randn(B, S)
        obs = torch.randn(B, O)
        result = planner.plan(state, obs, preferences, ctx={"deterministic": True})
        n_elite = result.debug.get("n_elite", -1)
        expected_elite = max(1, int(20 * 0.1))  # 2
        assert n_elite == expected_elite, \
            f"n_elite {n_elite} != {expected_elite}"
    _test("17_cem_elite_count", test_cem_elite_count)

    # ---------------------------------------------------------------
    # Test 18: CEM action bounds respected
    # ---------------------------------------------------------------
    def test_cem_action_bounds():
        cem_config = PlannerConfig(
            planner_type="cem",
            planning_horizon=H,
            num_rollouts=N,
            cem_iterations=2,
            action_dim=A,
            state_dim=S,
            obs_dim=O,
            action_bounds=(-0.5, 0.5),
            seed=42,
            use_warm_start=False,
        )
        cem_engine = RolloutEngine(transition, likelihood, efe_computer, cem_config)
        planner = CEMPlanner(cem_engine, cem_config)
        state = torch.randn(B, S)
        obs = torch.randn(B, O)
        result = planner.plan(state, obs, preferences, ctx={"deterministic": True})
        assert result.best_action.min() >= -0.5 - 1e-6, \
            f"Action below lower bound: {result.best_action.min()}"
        assert result.best_action.max() <= 0.5 + 1e-6, \
            f"Action above upper bound: {result.best_action.max()}"
    _test("18_cem_action_bounds", test_cem_action_bounds)

    # ---------------------------------------------------------------
    # Test 19: CEM warm-start
    # ---------------------------------------------------------------
    def test_cem_warm_start():
        cem_config = PlannerConfig(
            planner_type="cem",
            planning_horizon=H,
            num_rollouts=N,
            cem_iterations=2,
            action_dim=A,
            state_dim=S,
            obs_dim=O,
            seed=42,
            use_warm_start=True,
        )
        cem_engine = RolloutEngine(transition, likelihood, efe_computer, cem_config)
        planner = CEMPlanner(cem_engine, cem_config)
        state = torch.randn(B, S)
        obs = torch.randn(B, O)

        # First call: cold start
        result1 = planner.plan(state, obs, preferences, ctx={"deterministic": True})
        assert planner._cem_mu is not None, "CEM mu not saved after first plan"
        assert planner._cem_std is not None, "CEM std not saved after first plan"

        # Second call: should warm-start
        result2 = planner.plan(state, obs, preferences, ctx={"deterministic": True})
        # After warm-start, the initial distribution should differ from cold start
        assert planner._prev_batch_size == B
    _test("19_cem_warm_start", test_cem_warm_start)

    # ---------------------------------------------------------------
    # Test 20: CEM reset clears warm-start
    # ---------------------------------------------------------------
    def test_cem_reset():
        cem_config = PlannerConfig(
            planner_type="cem",
            planning_horizon=H,
            num_rollouts=N,
            cem_iterations=2,
            action_dim=A,
            state_dim=S,
            obs_dim=O,
            seed=42,
            use_warm_start=True,
        )
        cem_engine = RolloutEngine(transition, likelihood, efe_computer, cem_config)
        planner = CEMPlanner(cem_engine, cem_config)
        state = torch.randn(B, S)
        obs = torch.randn(B, O)

        planner.plan(state, obs, preferences, ctx={"deterministic": True})
        assert planner._cem_mu is not None
        planner.reset()
        assert planner._cem_mu is None, "CEM mu not cleared after reset"
        assert planner._cem_std is None, "CEM std not cleared after reset"
        assert planner._prev_batch_size == 0
    _test("20_cem_reset", test_cem_reset)

    # ---------------------------------------------------------------
    # Test 21: Interface compliance (ABC)
    # ---------------------------------------------------------------
    def test_interface_compliance():
        assert issubclass(RandomShootingPlanner, Planner)
        assert issubclass(CEMPlanner, Planner)
        # Check abstract methods are implemented
        for cls in [RandomShootingPlanner, CEMPlanner]:
            assert hasattr(cls, "plan")
            assert hasattr(cls, "reset")
    _test("21_interface_compliance", test_interface_compliance)

    # ---------------------------------------------------------------
    # Test 22: PlanResult validation catches bad shapes
    # ---------------------------------------------------------------
    def test_plan_result_validation_catches_errors():
        bad_result = PlanResult(
            best_action=torch.zeros(B, A),
            best_action_sequence=torch.zeros(B, H, A),
            efe_total=torch.zeros(B, 2),  # Wrong shape!
            efe_terms={"pragmatic": torch.zeros(B)},
            all_efe=torch.zeros(B, N),
        )
        try:
            bad_result.validate()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
    _test("22_plan_result_validation_catches_errors", test_plan_result_validation_catches_errors)

    # ---------------------------------------------------------------
    # Test 23: Factory function creates correct planner types
    # ---------------------------------------------------------------
    def test_factory_random_shooting():
        cfg = PlannerConfig(
            planner_type="random_shooting",
            planning_horizon=H,
            num_rollouts=N,
            action_dim=A,
            state_dim=S,
            obs_dim=O,
        )
        planner = create_planner(cfg, transition, likelihood, efe_computer)
        assert isinstance(planner, RandomShootingPlanner), \
            f"Expected RandomShootingPlanner, got {type(planner)}"
    _test("23_factory_random_shooting", test_factory_random_shooting)

    # ---------------------------------------------------------------
    # Test 24: Factory function creates CEM planner
    # ---------------------------------------------------------------
    def test_factory_cem():
        cfg = PlannerConfig(
            planner_type="cem",
            planning_horizon=H,
            num_rollouts=N,
            action_dim=A,
            state_dim=S,
            obs_dim=O,
        )
        planner = create_planner(cfg, transition, likelihood, efe_computer)
        assert isinstance(planner, CEMPlanner), \
            f"Expected CEMPlanner, got {type(planner)}"
    _test("24_factory_cem", test_factory_cem)

    # ---------------------------------------------------------------
    # Test 25: Factory function rejects unknown planner type
    # ---------------------------------------------------------------
    def test_factory_unknown_type():
        cfg = PlannerConfig(
            planner_type="nonexistent",
            action_dim=A,
            state_dim=S,
            obs_dim=O,
        )
        try:
            create_planner(cfg, transition, likelihood, efe_computer)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "nonexistent" in str(e).lower()
    _test("25_factory_unknown_type", test_factory_unknown_type)

    # ---------------------------------------------------------------
    # Test 26: Factory rejects invalid config
    # ---------------------------------------------------------------
    def test_factory_invalid_config():
        cfg = PlannerConfig(
            planner_type="cem",
            action_dim=0,  # Invalid
            state_dim=S,
            obs_dim=O,
        )
        try:
            create_planner(cfg, transition, likelihood, efe_computer)
            assert False, "Should have raised ValueError for action_dim=0"
        except ValueError:
            pass
    _test("26_factory_invalid_config", test_factory_invalid_config)

    # ---------------------------------------------------------------
    # Test 27: Configuration presets
    # ---------------------------------------------------------------
    def test_config_presets():
        mc = minimal_config()
        assert mc.planner_type == "random_shooting"
        assert mc.num_rollouts == 8
        assert mc.seed == 42

        sc = standard_config()
        assert sc.planner_type == "cem"
        assert sc.num_rollouts == 128

        pc = production_config()
        assert pc.planner_type == "cem"
        assert pc.num_rollouts == 512
        assert pc.action_bounds == (-1.0, 1.0)
    _test("27_config_presets", test_config_presets)

    # ---------------------------------------------------------------
    # Test 28: CEM momentum blending
    # ---------------------------------------------------------------
    def test_cem_momentum():
        cem_config = PlannerConfig(
            planner_type="cem",
            planning_horizon=H,
            num_rollouts=16,
            cem_iterations=3,
            cem_elite_fraction=0.25,
            cem_momentum=0.5,
            action_dim=A,
            state_dim=S,
            obs_dim=O,
            seed=42,
            use_warm_start=False,
        )
        cem_engine = RolloutEngine(transition, likelihood, efe_computer, cem_config)
        planner = CEMPlanner(cem_engine, cem_config)
        state = torch.randn(B, S)
        obs = torch.randn(B, O)
        result = planner.plan(state, obs, preferences, ctx={"deterministic": True})
        # With momentum, the final std should not collapse to zero
        final_std = result.debug.get("cem_final_std_mean", 0.0)
        assert final_std > 0.0, "CEM std collapsed to zero with momentum"
    _test("28_cem_momentum", test_cem_momentum)

    # ---------------------------------------------------------------
    # Test 29: Register custom planner
    # ---------------------------------------------------------------
    def test_register_custom_planner():
        class DummyPlanner(Planner, nn.Module):
            def __init__(self, rollout_engine, config):
                nn.Module.__init__(self)
                self.config = config
            def plan(self, initial_state, observation, preferences, ctx=None):
                B = initial_state.shape[0]
                return PlanResult(
                    best_action=torch.zeros(B, self.config.action_dim),
                    best_action_sequence=torch.zeros(B, self.config.planning_horizon, self.config.action_dim),
                    efe_total=torch.zeros(B),
                    efe_terms={"pragmatic": torch.zeros(B)},
                    all_efe=torch.zeros(B, self.config.num_rollouts),
                )
            def reset(self):
                pass

        register_planner("dummy", DummyPlanner)
        assert "dummy" in _PLANNER_REGISTRY
        cfg = PlannerConfig(
            planner_type="dummy",
            action_dim=A, state_dim=S, obs_dim=O,
        )
        p = create_planner(cfg, transition, likelihood, efe_computer)
        assert isinstance(p, DummyPlanner)
        # Clean up
        del _PLANNER_REGISTRY["dummy"]
    _test("29_register_custom_planner", test_register_custom_planner)

    # ---------------------------------------------------------------
    # Test 30: RolloutEngine initial state not mutated
    # ---------------------------------------------------------------
    def test_rollout_no_state_mutation():
        state = torch.randn(B, S)
        state_copy = state.clone()
        actions = torch.randn(B, N, H, A)
        engine.rollout(state, actions, preferences)
        assert torch.equal(state, state_copy), \
            "RolloutEngine mutated initial_state"
    _test("30_rollout_no_state_mutation", test_rollout_no_state_mutation)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("=" * 72)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if errors:
        print("\nFailures:")
        for e in errors:
            print(e)
    print("=" * 72)


# ============================================================================
# Module entry point
# ============================================================================


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _run_self_tests()
