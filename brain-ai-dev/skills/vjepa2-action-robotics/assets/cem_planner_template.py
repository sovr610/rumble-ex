"""
CEM Planner Template: Cross-Entropy Method for Robot Action Planning

Implements the CEMPlanner that uses a world model to plan action sequences
via the Cross-Entropy Method (CEM). The planner:
  1. Initializes a Gaussian distribution over action sequences
  2. Samples N sequences per iteration
  3. Clips to action space constraints
  4. Rolls out each sequence through the world model
  5. Selects top-k (elite) sequences by L1 distance to goal representation
  6. Updates distribution with momentum
  7. Returns mu[0]: the optimal first action

Action space (7-DOF):
  [0:3] xyz translation -- clipped to ±maxnorm (meters)
  [3:6] orientation     -- zeroed (not planned by CEM)
  [6]   gripper         -- clipped to [-0.75, 0.75]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CEMConfig:
    horizon: int = 10               # Steps to plan ahead
    num_samples: int = 512          # Number of action sequences per iteration
    num_elites: int = 64            # Top-k sequences retained as elites
    num_iterations: int = 5         # CEM refinement iterations
    momentum_xyz: float = 0.1       # Momentum for xyz components (low = fast adapt)
    momentum_gripper: float = 0.3   # Momentum for gripper (higher = slower adapt)
    maxnorm: float = 0.02           # Max xyz translation per step (meters)
    gripper_min: float = -0.75      # Minimum gripper opening
    gripper_max: float = 0.75       # Maximum gripper opening
    sigma_init: float = 1.0         # Initial standard deviation
    sigma_min: float = 1e-6         # Minimum sigma (prevents collapse to zero)

    @property
    def gripper_range(self) -> Tuple[float, float]:
        return (self.gripper_min, self.gripper_max)


# ---------------------------------------------------------------------------
# World Model Protocol
# ---------------------------------------------------------------------------

class WorldModelProtocol(Protocol):
    """Protocol interface that CEMPlanner expects from a world model."""

    def predict_next(
        self,
        current_repr: Tensor,
        action: Tensor,
        state: Tensor,
    ) -> Tensor:
        """Predict next representation: [B, repr_dim] -> [B, repr_dim]."""
        ...


# ---------------------------------------------------------------------------
# Action Clipping
# ---------------------------------------------------------------------------

def clip_actions(
    sequences: Tensor,
    maxnorm: float = 0.02,
    gripper_min: float = -0.75,
    gripper_max: float = 0.75,
) -> Tensor:
    """
    Enforce action space constraints on sampled sequences.

    Constraints:
      - XYZ (indices 0:3): clamp each component to [-maxnorm, +maxnorm]
      - Orientation (indices 3:6): set to 0 (not planned by CEM)
      - Gripper (index 6): clamp to [gripper_min, gripper_max]

    Args:
        sequences: [N, horizon, 7] or [horizon, 7] or [7]

    Returns:
        clipped: same shape
    """
    sequences = sequences.clone()

    if sequences.dim() == 1:
        # Single action vector [7]
        sequences[0:3] = sequences[0:3].clamp(-maxnorm, maxnorm)
        sequences[3:6] = 0.0
        sequences[6]   = sequences[6].clamp(gripper_min, gripper_max)
    elif sequences.dim() == 2:
        # [horizon, 7] or [N, 7]
        sequences[:, 0:3] = sequences[:, 0:3].clamp(-maxnorm, maxnorm)
        sequences[:, 3:6] = 0.0
        sequences[:, 6]   = sequences[:, 6].clamp(gripper_min, gripper_max)
    elif sequences.dim() == 3:
        # [N, horizon, 7]
        sequences[..., 0:3] = sequences[..., 0:3].clamp(-maxnorm, maxnorm)
        sequences[..., 3:6] = 0.0
        sequences[..., 6]   = sequences[..., 6].clamp(gripper_min, gripper_max)
    else:
        raise ValueError(f"Unexpected sequences shape: {sequences.shape}")

    return sequences


# ---------------------------------------------------------------------------
# Batched Rollout
# ---------------------------------------------------------------------------

def batched_rollout(
    world_model: WorldModelProtocol,
    current_repr: Tensor,
    sequences: Tensor,
    current_state: Tensor,
) -> Tensor:
    """
    Roll out N action sequences in parallel through the world model.

    Args:
        world_model:   Supports predict_next(repr[B,D], action[B,7], state[B,7]) -> repr[B,D]
        current_repr:  [repr_dim] or [1, repr_dim] -- single starting state
        sequences:     [N, horizon, 7] -- N action sequences to evaluate
        current_state: [7] or [1, 7] -- current robot state (open-loop: stays fixed)

    Returns:
        final_reprs: [N, repr_dim]
    """
    N, horizon, _ = sequences.shape
    device = sequences.device

    # Expand starting state to batch dimension
    if current_repr.dim() == 1:
        repr_batch = current_repr.unsqueeze(0).expand(N, -1).contiguous()
    else:
        repr_batch = current_repr.expand(N, -1).contiguous()

    if current_state.dim() == 1:
        state_batch = current_state.unsqueeze(0).expand(N, -1).contiguous()
    else:
        state_batch = current_state.expand(N, -1).contiguous()

    # Autoregressive rollout
    for t in range(horizon):
        actions_t = sequences[:, t, :]    # [N, 7]
        repr_batch = world_model.predict_next(repr_batch, actions_t, state_batch)
        # State is fixed (open-loop planning)

    return repr_batch  # [N, repr_dim]


# ---------------------------------------------------------------------------
# CEM Planner
# ---------------------------------------------------------------------------

class CEMPlanner:
    """
    Cross-Entropy Method optimizer for robot action planning.

    Given a current world state representation and a goal representation,
    finds the action sequence that best reaches the goal by iteratively
    refining a Gaussian distribution over action sequences.

    Usage:
        planner = CEMPlanner(world_model, config)
        action = planner.plan(current_repr, goal_repr, current_state)
    """

    def __init__(self, world_model: WorldModelProtocol, config: CEMConfig):
        self.world_model = world_model
        self.config = config

    def plan(
        self,
        current_repr: Tensor,
        goal_repr: Tensor,
        current_state: Tensor,
        return_diagnostics: bool = False,
    ) -> Tensor:
        """
        Plan the optimal first action to reach the goal representation.

        Args:
            current_repr:  [repr_dim] -- current world state representation
            goal_repr:     [repr_dim] -- target world state representation
            current_state: [7] -- current robot proprioceptive state
            return_diagnostics: if True, also return cost history

        Returns:
            action: [7] -- optimal first action (clipped to action space)
            diagnostics: (optional) dict with 'cost_history', 'final_mu', 'final_sigma'
        """
        cfg = self.config
        device = current_repr.device

        # Initialize distribution
        mu    = torch.zeros(cfg.horizon, 7, device=device)
        sigma = torch.full((cfg.horizon, 7), cfg.sigma_init, device=device)

        cost_history: List[float] = []

        for iteration in range(cfg.num_iterations):
            # 1. Sample N action sequences from Normal(mu, sigma^2)
            noise = torch.randn(cfg.num_samples, cfg.horizon, 7, device=device)
            sequences = (
                mu.unsqueeze(0).expand(cfg.num_samples, -1, -1) +
                sigma.unsqueeze(0).expand(cfg.num_samples, -1, -1) * noise
            )

            # 2. Clip to action space constraints
            sequences = clip_actions(sequences, cfg.maxnorm, cfg.gripper_min, cfg.gripper_max)

            # 3. Roll out each sequence through world model
            with torch.no_grad():
                final_reprs = batched_rollout(
                    self.world_model,
                    current_repr,
                    sequences,
                    current_state,
                )  # [N, repr_dim]

            # 4. Score: L1 distance to goal representation
            goal_expanded = goal_repr.unsqueeze(0).expand(cfg.num_samples, -1)
            costs = torch.mean(torch.abs(final_reprs - goal_expanded), dim=-1)  # [N]

            best_cost = costs.min().item()
            cost_history.append(best_cost)

            # 5. Select top-k elites by lowest cost
            elite_idxs = torch.argsort(costs)[:cfg.num_elites]
            elites = sequences[elite_idxs]  # [k, horizon, 7]

            # 6. Update distribution with per-DOF momentum
            new_mu    = elites.mean(dim=0)   # [horizon, 7]
            new_sigma = elites.std(dim=0).clamp(min=cfg.sigma_min)  # [horizon, 7]

            # XYZ components: fast adaptation (low momentum)
            mu_next = mu.clone()
            mu_next[..., 0:3] = (
                cfg.momentum_xyz * mu[..., 0:3] +
                (1.0 - cfg.momentum_xyz) * new_mu[..., 0:3]
            )

            # Orientation: always zero (not planned by CEM)
            mu_next[..., 3:6] = 0.0

            # Gripper: slower adaptation
            mu_next[..., 6] = (
                cfg.momentum_gripper * mu[..., 6] +
                (1.0 - cfg.momentum_gripper) * new_mu[..., 6]
            )

            # Sigma: smooth update (uniform momentum for variance)
            sigma_next = (
                cfg.momentum_xyz * sigma +
                (1.0 - cfg.momentum_xyz) * new_sigma
            ).clamp(min=cfg.sigma_min)

            mu    = mu_next
            sigma = sigma_next

        # Return the optimal first action, clipped to action space
        first_action = mu[0].clone()  # [7]
        first_action = clip_actions(first_action, cfg.maxnorm, cfg.gripper_min, cfg.gripper_max)

        if return_diagnostics:
            diagnostics = {
                'cost_history': cost_history,
                'final_mu': mu.detach().cpu(),
                'final_sigma': sigma.detach().cpu(),
            }
            return first_action, diagnostics

        return first_action

    def plan_sequence(
        self,
        current_repr: Tensor,
        goal_repr: Tensor,
        current_state: Tensor,
    ) -> Tensor:
        """
        Return the full planned action sequence (horizon steps), not just step 0.

        Returns:
            mu: [horizon, 7] -- planned action sequence
        """
        cfg = self.config
        device = current_repr.device

        mu    = torch.zeros(cfg.horizon, 7, device=device)
        sigma = torch.full((cfg.horizon, 7), cfg.sigma_init, device=device)

        for _ in range(cfg.num_iterations):
            noise = torch.randn(cfg.num_samples, cfg.horizon, 7, device=device)
            sequences = (
                mu.unsqueeze(0).expand(cfg.num_samples, -1, -1) +
                sigma.unsqueeze(0).expand(cfg.num_samples, -1, -1) * noise
            )
            sequences = clip_actions(sequences, cfg.maxnorm, cfg.gripper_min, cfg.gripper_max)

            with torch.no_grad():
                final_reprs = batched_rollout(
                    self.world_model, current_repr, sequences, current_state
                )

            goal_expanded = goal_repr.unsqueeze(0).expand(cfg.num_samples, -1)
            costs = torch.mean(torch.abs(final_reprs - goal_expanded), dim=-1)

            elite_idxs = torch.argsort(costs)[:cfg.num_elites]
            elites = sequences[elite_idxs]

            new_mu    = elites.mean(dim=0)
            new_sigma = elites.std(dim=0).clamp(min=cfg.sigma_min)

            mu_next = mu.clone()
            mu_next[..., 0:3] = cfg.momentum_xyz * mu[..., 0:3] + (1 - cfg.momentum_xyz) * new_mu[..., 0:3]
            mu_next[..., 3:6] = 0.0
            mu_next[..., 6]   = cfg.momentum_gripper * mu[..., 6] + (1 - cfg.momentum_gripper) * new_mu[..., 6]
            sigma = (cfg.momentum_xyz * sigma + (1 - cfg.momentum_xyz) * new_sigma).clamp(min=cfg.sigma_min)
            mu = mu_next

        return clip_actions(mu, cfg.maxnorm, cfg.gripper_min, cfg.gripper_max)


# ---------------------------------------------------------------------------
# Synthetic World Model for Testing
# ---------------------------------------------------------------------------

class LinearSyntheticWorldModel:
    """
    Synthetic world model for testing CEM convergence.

    predict_next: repr -> repr + alpha * action[:repr_dim]

    The optimal action to reach goal_repr from current_repr is:
        action[:repr_dim] = (goal_repr - current_repr) / (horizon * alpha)
    """

    def __init__(self, repr_dim: int = 64, alpha: float = 0.1):
        self.repr_dim = repr_dim
        self.alpha = alpha

    def predict_next(self, repr_: Tensor, action: Tensor, state: Tensor) -> Tensor:
        """repr + alpha * action (using only action[:repr_dim])."""
        action_component = action[:, :self.repr_dim] if action.shape[-1] >= self.repr_dim else action
        # Pad if needed
        if action_component.shape[-1] < self.repr_dim:
            pad = torch.zeros(
                action_component.shape[0],
                self.repr_dim - action_component.shape[-1],
                device=action.device
            )
            action_component = torch.cat([action_component, pad], dim=-1)
        return repr_ + self.alpha * action_component


class IdentityWorldModel:
    """World model that always returns the same representation (no change)."""

    def predict_next(self, repr_: Tensor, action: Tensor, state: Tensor) -> Tensor:
        return repr_


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_plan_returns_shape():
    """T30: plan() returns shape [7]."""
    print("[TEST] plan() returns shape [7]...")

    repr_dim = 32
    wm = IdentityWorldModel()
    cfg = CEMConfig(horizon=3, num_samples=32, num_elites=8, num_iterations=2)
    planner = CEMPlanner(wm, cfg)

    current_repr = torch.randn(repr_dim)
    goal_repr    = torch.randn(repr_dim)
    state        = torch.zeros(7)

    action = planner.plan(current_repr, goal_repr, state)
    assert action.shape == (7,), f"Action shape: {action.shape} != (7,)"
    print(f"  Action shape: {action.shape} -- PASS")


def _test_cem_reduces_cost():
    """T31 / T32: CEM reduces L1 distance to goal over iterations."""
    print("[TEST] CEM reduces distance to goal over iterations...")

    repr_dim = 7  # Match action dim for simple test
    alpha = 1.0   # Strong signal

    class StrongLinearWM:
        def predict_next(self, repr_: Tensor, action: Tensor, state: Tensor) -> Tensor:
            return repr_ + alpha * action

    wm = StrongLinearWM()
    cfg = CEMConfig(
        horizon=1,
        num_samples=256,
        num_elites=32,
        num_iterations=10,
        momentum_xyz=0.0,
        momentum_gripper=0.0,
        maxnorm=2.0,   # Allow larger actions for this test
    )
    planner = CEMPlanner(wm, cfg)

    torch.manual_seed(42)
    current_repr = torch.zeros(repr_dim)
    goal_repr = torch.tensor([0.5, -0.5, 0.3, 0.0, 0.0, 0.0, 0.4])

    action, diag = planner.plan(current_repr, goal_repr, torch.zeros(7),
                                 return_diagnostics=True)

    costs = diag['cost_history']
    initial_cost = costs[0]
    final_cost   = costs[-1]

    print(f"  Initial cost: {initial_cost:.4f}, Final cost: {final_cost:.4f}")
    assert final_cost < initial_cost, (
        f"CEM did not reduce cost: {final_cost:.4f} >= {initial_cost:.4f}"
    )
    print(f"  Cost reduced by {(1 - final_cost/initial_cost)*100:.1f}% -- PASS")


def _test_action_clipping():
    """T37 / T38 / T39: Action space constraints are enforced."""
    print("[TEST] Action clipping...")

    cfg = CEMConfig(maxnorm=0.02, gripper_min=-0.75, gripper_max=0.75)

    # Test with extreme values
    action = torch.tensor([5.0, -5.0, 3.0, 1.5, -2.0, 0.8, 2.0])
    clipped = clip_actions(action, cfg.maxnorm, cfg.gripper_min, cfg.gripper_max)

    assert clipped[0:3].abs().max().item() <= cfg.maxnorm + 1e-6, (
        f"XYZ not clipped: {clipped[0:3]}"
    )
    assert (clipped[3:6] == 0.0).all(), f"Orientation not zeroed: {clipped[3:6]}"
    assert clipped[6].item() <= cfg.gripper_max + 1e-6, f"Gripper not clamped high: {clipped[6]}"
    assert clipped[6].item() >= cfg.gripper_min - 1e-6, f"Gripper not clamped low: {clipped[6]}"

    print(f"  XYZ in [-{cfg.maxnorm}, {cfg.maxnorm}]: {clipped[0:3].tolist()} -- PASS")
    print(f"  Orientation zeroed: {clipped[3:6].tolist()} -- PASS")
    print(f"  Gripper in [{cfg.gripper_min}, {cfg.gripper_max}]: {clipped[6].item():.3f} -- PASS")

    # Test with already-valid values (should be no-op)
    valid = torch.tensor([0.01, -0.01, 0.01, 0.0, 0.0, 0.0, 0.5])
    clipped_valid = clip_actions(valid, cfg.maxnorm, cfg.gripper_min, cfg.gripper_max)
    assert torch.allclose(valid, clipped_valid), "Valid action should not be modified"
    print(f"  Valid action unchanged -- PASS")


def _test_sigma_does_not_collapse():
    """T35: Sigma never reaches 0 due to clamp."""
    print("[TEST] Sigma stays above minimum...")

    repr_dim = 16
    wm = IdentityWorldModel()
    cfg = CEMConfig(
        horizon=5, num_samples=64, num_elites=8, num_iterations=20,
        sigma_min=1e-6
    )
    planner = CEMPlanner(wm, cfg)

    current_repr = torch.zeros(repr_dim)
    goal_repr = torch.ones(repr_dim)
    state = torch.zeros(7)

    _, diag = planner.plan(current_repr, goal_repr, state, return_diagnostics=True)
    sigma = diag['final_sigma']

    assert sigma.min().item() >= cfg.sigma_min - 1e-9, (
        f"Sigma collapsed below minimum: {sigma.min().item()}"
    )
    print(f"  Min sigma: {sigma.min().item():.2e} >= {cfg.sigma_min:.2e} -- PASS")


def _test_cem_convergence_synthetic():
    """T43 / T44: CEM converges on synthetic world model."""
    print("[TEST] CEM convergence on synthetic linear world model...")

    repr_dim = 7
    wm = LinearSyntheticWorldModel(repr_dim=repr_dim, alpha=0.1)
    cfg = CEMConfig(
        horizon=10,
        num_samples=512,
        num_elites=64,
        num_iterations=5,
        momentum_xyz=0.1,
        momentum_gripper=0.3,
        maxnorm=0.02,
    )
    planner = CEMPlanner(wm, cfg)

    torch.manual_seed(0)
    current_repr = torch.zeros(repr_dim)
    goal_repr    = torch.tensor([0.1, -0.05, 0.08, 0.0, 0.0, 0.0, 0.3])
    state        = torch.zeros(7)

    action, diag = planner.plan(current_repr, goal_repr, state, return_diagnostics=True)

    costs = diag['cost_history']
    print(f"  Cost history: {[f'{c:.4f}' for c in costs]}")
    assert costs[-1] < costs[0], "Cost did not decrease"
    print(f"  Cost decreased from {costs[0]:.4f} to {costs[-1]:.4f} -- PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("CEMPlanner Self-Tests")
    print("=" * 60)

    _test_plan_returns_shape()
    _test_cem_reduces_cost()
    _test_action_clipping()
    _test_sigma_does_not_collapse()
    _test_cem_convergence_synthetic()

    print("=" * 60)
    print("All self-tests PASSED")
    print("=" * 60)
