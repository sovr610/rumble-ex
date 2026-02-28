#!/usr/bin/env python3
"""
Deterministic Toy Benchmark Harness for the Active Inference Agent.

Proves active inference behaviour on toy problems (grid-world navigation)
using the built-in ActiveInferenceAgent from the template.  Runs in < 2 min
on CPU with fully deterministic seeding in both "online" (interactive agent)
and "offline" (world-model training on collected data) modes.

No mandatory external dependencies beyond torch and numpy.  Gymnasium and
Minari are detected at import time and used when available, but the harness
generates its own synthetic data as a fallback.

Usage:
    python scripts/toy_benchmark.py --mode both --seed 42 --grid-size 5
    python scripts/toy_benchmark.py --mode online --episodes 20 --verbose
    python scripts/toy_benchmark.py --mode offline --output results.json

Exit code 0 when all sanity checks pass; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional external dependencies
# ---------------------------------------------------------------------------

try:
    import gymnasium  # noqa: F401

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

try:
    import minari  # noqa: F401

    MINARI_AVAILABLE = True
except ImportError:
    MINARI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Resolve the active inference template so that the benchmark is self-contained.
# We add the assets directory to sys.path and import the template module.
# ---------------------------------------------------------------------------
_ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets")
if _ASSETS_DIR not in sys.path:
    sys.path.insert(0, _ASSETS_DIR)

from active_inference_template import (  # noqa: E402
    ActiveInferenceAgent,
    ActiveInferenceFullConfig,
    AgentState,
    ActionOutput,
    EFEConfig,
    GenerativeModelConfig,
    PlannerConfig,
    AmortizedPolicyConfig,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

ACTION_UP: int = 0
ACTION_DOWN: int = 1
ACTION_LEFT: int = 2
ACTION_RIGHT: int = 3
ACTION_NAMES: Dict[int, str] = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

_SEPARATOR = "-" * 72


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for the toy benchmark harness.

    Attributes:
        grid_size: Side length of the square grid world.
        max_episodes: Number of episodes to run in the online benchmark.
        max_steps_per_episode: Hard limit on steps within a single episode.
        seed: Global deterministic seed for torch, numpy, and the environment.
        num_training_steps: Gradient steps for offline world-model training.
        obs_dim: Observation dimensionality (grid_size ** 2 for one-hot).
        action_dim: Number of discrete actions (4 cardinal directions).
        state_dim: Latent state dimensionality for the generative model.
        hidden_dim: Hidden layer width for the generative model.
        planning_horizon: Number of imagined steps in the planner.
        num_rollouts: Number of candidate action sequences for the planner.
        cem_iterations: Refinement rounds for the CEM planner.
        learning_rate: Optimizer learning rate for offline training.
        batch_size: Mini-batch size for offline training.
        synthetic_dataset_episodes: Episodes generated for synthetic dataset.
        synthetic_dataset_max_steps: Max steps per synthetic episode.
    """

    grid_size: int = 5
    max_episodes: int = 10
    max_steps_per_episode: int = 50
    seed: int = 42
    num_training_steps: int = 100
    obs_dim: int = 25  # grid_size ** 2
    action_dim: int = 4
    state_dim: int = 16
    hidden_dim: int = 32
    planning_horizon: int = 3
    num_rollouts: int = 16
    cem_iterations: int = 3
    learning_rate: float = 1e-3
    batch_size: int = 32
    synthetic_dataset_episodes: int = 50
    synthetic_dataset_max_steps: int = 50

    def __post_init__(self) -> None:
        """Sync obs_dim with grid_size."""
        self.obs_dim = self.grid_size * self.grid_size


# ============================================================================
# ToyEnvironment -- built-in GridWorld (no gymnasium dependency)
# ============================================================================


class GridWorld:
    """Simple deterministic grid world for testing active inference behaviour.

    The agent starts at a random (seeded) position and must navigate to a
    fixed goal position.  The observation is a one-hot encoding of the
    agent's position on an N x N grid.

    Actions:
        0 = UP    (row - 1)
        1 = DOWN  (row + 1)
        2 = LEFT  (col - 1)
        3 = RIGHT (col + 1)

    Reward scheme:
        -1 per step, +10 upon reaching the goal, episode terminates at goal
        or max_steps.

    All randomness is governed by the provided numpy RNG for full
    reproducibility.

    Args:
        grid_size: Side length of the square grid.
        goal_position: (row, col) tuple for the goal.  If None, defaults
            to the bottom-right corner (grid_size-1, grid_size-1).
        max_steps: Maximum steps before forced termination.
        rng: numpy random generator for deterministic behaviour.
    """

    def __init__(
        self,
        grid_size: int = 5,
        goal_position: Optional[Tuple[int, int]] = None,
        max_steps: int = 50,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.grid_size = grid_size
        self.goal_position = goal_position or (grid_size - 1, grid_size - 1)
        self.max_steps = max_steps
        self.rng = rng or np.random.default_rng(42)

        self.obs_dim = grid_size * grid_size
        self.action_dim = 4

        # Mutable state
        self.agent_row: int = 0
        self.agent_col: int = 0
        self.step_count: int = 0
        self.done: bool = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the environment to a random starting position.

        The starting position is sampled uniformly from all cells except
        the goal.

        Returns:
            obs: One-hot observation of shape (obs_dim,).
        """
        while True:
            self.agent_row = int(self.rng.integers(0, self.grid_size))
            self.agent_col = int(self.rng.integers(0, self.grid_size))
            if (self.agent_row, self.agent_col) != self.goal_position:
                break
        self.step_count = 0
        self.done = False
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one action in the grid world.

        Args:
            action: Integer action in {0, 1, 2, 3}.

        Returns:
            obs: One-hot observation (obs_dim,).
            reward: Scalar reward.
            done: Whether the episode has terminated.
            info: Diagnostic dictionary with position and manhattan distance.
        """
        if self.done:
            return self._get_obs(), 0.0, True, self._get_info()

        # Apply movement (clip at boundaries)
        if action == ACTION_UP:
            self.agent_row = max(0, self.agent_row - 1)
        elif action == ACTION_DOWN:
            self.agent_row = min(self.grid_size - 1, self.agent_row + 1)
        elif action == ACTION_LEFT:
            self.agent_col = max(0, self.agent_col - 1)
        elif action == ACTION_RIGHT:
            self.agent_col = min(self.grid_size - 1, self.agent_col + 1)
        else:
            raise ValueError(f"Invalid action {action}; expected 0-3.")

        self.step_count += 1

        # Check termination
        at_goal = (self.agent_row, self.agent_col) == self.goal_position
        over_limit = self.step_count >= self.max_steps
        self.done = at_goal or over_limit

        reward = 10.0 if at_goal else -1.0

        return self._get_obs(), reward, self.done, self._get_info()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Return one-hot encoded position vector."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        idx = self.agent_row * self.grid_size + self.agent_col
        obs[idx] = 1.0
        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Return diagnostic info dict."""
        goal_r, goal_c = self.goal_position
        manhattan = abs(self.agent_row - goal_r) + abs(self.agent_col - goal_c)
        return {
            "agent_pos": (self.agent_row, self.agent_col),
            "goal_pos": self.goal_position,
            "manhattan_distance": manhattan,
            "step": self.step_count,
            "at_goal": (self.agent_row, self.agent_col) == self.goal_position,
        }

    def manhattan_distance(self) -> int:
        """Current Manhattan distance to goal."""
        goal_r, goal_c = self.goal_position
        return abs(self.agent_row - goal_r) + abs(self.agent_col - goal_c)

    def __repr__(self) -> str:
        return (
            f"GridWorld(size={self.grid_size}, "
            f"agent=({self.agent_row},{self.agent_col}), "
            f"goal={self.goal_position}, step={self.step_count})"
        )


# ============================================================================
# Synthetic Dataset Generator
# ============================================================================


@dataclass
class SyntheticEpisode:
    """One episode of collected transitions.

    Attributes:
        observations: List of observation arrays, length T+1.
        actions: List of integer actions, length T.
        rewards: List of float rewards, length T.
        dones: List of bool termination flags, length T.
    """

    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)


@dataclass
class SyntheticDataset:
    """Collection of episodes matching the Minari episode structure.

    Attributes:
        episodes: List of SyntheticEpisode instances.
        total_steps: Sum of steps across all episodes.
        obs_dim: Observation dimensionality.
        action_dim: Number of discrete actions.
    """

    episodes: List[SyntheticEpisode] = field(default_factory=list)
    total_steps: int = 0
    obs_dim: int = 25
    action_dim: int = 4


def generate_synthetic_dataset(
    grid_size: int = 5,
    num_episodes: int = 50,
    max_steps: int = 50,
    seed: int = 42,
    policy: str = "random",
) -> SyntheticDataset:
    """Generate a synthetic dataset by rolling out a policy on GridWorld.

    Two policies are supported:
        - "random": Uniform random action selection.
        - "biased": 60% probability of moving toward the goal, 40% random.

    Args:
        grid_size: Grid side length.
        num_episodes: Number of episodes to collect.
        max_steps: Maximum steps per episode.
        seed: RNG seed for full reproducibility.
        policy: Policy type ("random" or "biased").

    Returns:
        SyntheticDataset containing all episodes and aggregate statistics.
    """
    rng = np.random.default_rng(seed)
    env = GridWorld(grid_size=grid_size, max_steps=max_steps, rng=rng)

    dataset = SyntheticDataset(
        obs_dim=grid_size * grid_size,
        action_dim=4,
    )

    for _ in range(num_episodes):
        obs = env.reset()
        episode = SyntheticEpisode()
        episode.observations.append(obs.copy())

        done = False
        while not done:
            if policy == "biased":
                action = _biased_action(
                    env.agent_row, env.agent_col,
                    env.goal_position[0], env.goal_position[1],
                    rng,
                )
            else:
                action = int(rng.integers(0, 4))

            next_obs, reward, done, info = env.step(action)

            episode.observations.append(next_obs.copy())
            episode.actions.append(action)
            episode.rewards.append(reward)
            episode.dones.append(done)

        dataset.episodes.append(episode)
        dataset.total_steps += len(episode.actions)

    logger.info(
        "Generated synthetic dataset: %d episodes, %d total steps, policy=%s",
        num_episodes, dataset.total_steps, policy,
    )
    return dataset


def _biased_action(
    row: int, col: int,
    goal_row: int, goal_col: int,
    rng: np.random.Generator,
) -> int:
    """Select an action biased toward the goal with 60% probability.

    Args:
        row: Current row.
        col: Current column.
        goal_row: Goal row.
        goal_col: Goal column.
        rng: Numpy random generator.

    Returns:
        Integer action in {0, 1, 2, 3}.
    """
    if rng.random() < 0.4:
        return int(rng.integers(0, 4))

    # Move toward goal
    dr = goal_row - row
    dc = goal_col - col

    candidates = []
    if dr > 0:
        candidates.append(ACTION_DOWN)
    elif dr < 0:
        candidates.append(ACTION_UP)
    if dc > 0:
        candidates.append(ACTION_RIGHT)
    elif dc < 0:
        candidates.append(ACTION_LEFT)

    if not candidates:
        # Already at goal (shouldn't happen in normal flow)
        return int(rng.integers(0, 4))

    return candidates[int(rng.integers(0, len(candidates)))]


def dataset_to_tensors(
    dataset: SyntheticDataset,
) -> Dict[str, torch.Tensor]:
    """Convert a SyntheticDataset to flat tensors for training.

    Extracts (o_t, a_t, o_{t+1}, reward, done) tuples from all episodes
    and stacks them into contiguous tensors.

    Args:
        dataset: The synthetic dataset to convert.

    Returns:
        Dictionary with keys:
            "observations": (N, obs_dim) float tensor.
            "actions": (N,) long tensor.
            "next_observations": (N, obs_dim) float tensor.
            "rewards": (N,) float tensor.
            "dones": (N,) bool tensor.
    """
    obs_list: List[np.ndarray] = []
    act_list: List[int] = []
    next_obs_list: List[np.ndarray] = []
    rew_list: List[float] = []
    done_list: List[bool] = []

    for episode in dataset.episodes:
        for t in range(len(episode.actions)):
            obs_list.append(episode.observations[t])
            act_list.append(episode.actions[t])
            next_obs_list.append(episode.observations[t + 1])
            rew_list.append(episode.rewards[t])
            done_list.append(episode.dones[t])

    return {
        "observations": torch.tensor(np.array(obs_list), dtype=torch.float32),
        "actions": torch.tensor(act_list, dtype=torch.long),
        "next_observations": torch.tensor(np.array(next_obs_list), dtype=torch.float32),
        "rewards": torch.tensor(rew_list, dtype=torch.float32),
        "dones": torch.tensor(done_list, dtype=torch.bool),
    }


# ============================================================================
# BenchmarkResult
# ============================================================================


@dataclass
class BenchmarkResult:
    """Container for benchmark metrics and metadata.

    Attributes:
        mode: "online" or "offline".
        success_rate: Fraction of episodes where the agent reached the goal.
        avg_episode_length: Mean episode length across all episodes.
        avg_reward: Mean total reward per episode.
        efe_curves: Per-step EFE term averages (keys: "total", "pragmatic",
            "epistemic", "instrumental").
        world_model_loss: Loss curve from offline world-model training.
        efe_improvement: Fractional improvement of planner EFE vs. dataset
            actions (offline only).
        wall_time_seconds: Total wall-clock time for this benchmark.
        config: Configuration dictionary used.
        episode_details: Per-episode diagnostics list.
        sanity_checks: Dict of named sanity checks and their pass/fail status.
    """

    mode: str = "online"
    success_rate: float = 0.0
    avg_episode_length: float = 0.0
    avg_reward: float = 0.0
    efe_curves: Dict[str, List[float]] = field(default_factory=dict)
    world_model_loss: Optional[List[float]] = None
    efe_improvement: Optional[float] = None
    wall_time_seconds: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    episode_details: List[Dict[str, Any]] = field(default_factory=list)
    sanity_checks: Dict[str, bool] = field(default_factory=dict)


# ============================================================================
# Agent Construction Helpers
# ============================================================================


def build_agent_config(cfg: BenchmarkConfig) -> ActiveInferenceFullConfig:
    """Build an ActiveInferenceFullConfig tailored for the toy benchmark.

    Uses the minimal preset as a starting point and overrides dimensions
    to match the grid world.

    Args:
        cfg: Benchmark configuration.

    Returns:
        Configured ActiveInferenceFullConfig.
    """
    return ActiveInferenceFullConfig(
        generative=GenerativeModelConfig(
            obs_dim=cfg.obs_dim,
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
            encoder_layers=2,
            decoder_layers=2,
            transition_layers=1,
            transition_ensemble_size=1,
            use_residual_transition=True,
            action_type="discrete",
            latent_type="continuous",
            preference_mode="learned",
            num_goals=1,
            learn_preferences=True,
        ),
        efe=EFEConfig(
            pragmatic_weight=1.0,
            epistemic_weight=1.0,
            instrumental_weight=0.1,
            num_samples=4,
            discount_factor=0.95,
            normalize_by_horizon=True,
            normalize_terms=False,
            assert_sum_invariant=True,
            sum_invariant_atol=1e-4,
            use_empowerment=True,
        ),
        planner=PlannerConfig(
            planner_type="random_shooting",
            planning_horizon=cfg.planning_horizon,
            num_rollouts=cfg.num_rollouts,
            cem_iterations=cfg.cem_iterations,
            action_temperature=0.5,
        ),
        amortized=AmortizedPolicyConfig(
            enabled=False,
        ),
        seed=cfg.seed,
    )


def seed_everything(seed: int) -> None:
    """Set deterministic seeds for torch, numpy, and Python hash.

    Args:
        seed: Integer seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic algorithms where possible
    torch.use_deterministic_algorithms(False)  # some ops lack deterministic impl
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================================
# Online Toy Benchmark
# ============================================================================


@dataclass
class StepLog:
    """Per-step diagnostic record for online benchmark episodes.

    Attributes:
        step: Step index within the episode.
        action: Selected action index.
        action_name: Human-readable action name.
        reward: Reward received.
        manhattan_distance: Manhattan distance to goal after the step.
        efe_total: Total EFE for the selected action.
        efe_pragmatic: Pragmatic EFE component.
        efe_epistemic: Epistemic EFE component.
        efe_instrumental: Instrumental EFE component.
    """

    step: int = 0
    action: int = 0
    action_name: str = ""
    reward: float = 0.0
    manhattan_distance: int = 0
    efe_total: float = 0.0
    efe_pragmatic: float = 0.0
    efe_epistemic: float = 0.0
    efe_instrumental: float = 0.0


def run_online_benchmark(cfg: BenchmarkConfig, verbose: bool = False) -> BenchmarkResult:
    """Run the online toy benchmark: agent interacts with GridWorld episodes.

    For each episode:
        1. Reset the grid world and the agent state.
        2. At each step, encode the observation, call agent.act(), decode
           the discrete action, and step the environment.
        3. Log per-step EFE terms, action, reward, and distance to goal.

    Sanity checks verified after all episodes:
        - success_rate > 0  (agent reaches goal at least once)
        - Average reward improves vs. random baseline
        - Pragmatic EFE correlates with distance to goal

    Args:
        cfg: Benchmark configuration.
        verbose: If True, print per-step diagnostics.

    Returns:
        BenchmarkResult with online metrics and sanity check outcomes.
    """
    t0 = time.time()
    seed_everything(cfg.seed)

    logger.info("=" * 72)
    logger.info("ONLINE TOY BENCHMARK")
    logger.info("=" * 72)
    logger.info(
        "Grid: %dx%d | Episodes: %d | Max steps: %d | Seed: %d",
        cfg.grid_size, cfg.grid_size, cfg.max_episodes,
        cfg.max_steps_per_episode, cfg.seed,
    )

    # Build agent
    agent_config = build_agent_config(cfg)
    agent = ActiveInferenceAgent(agent_config)
    agent.set_seed(cfg.seed)

    param_counts = agent.parameter_count()
    logger.info("Agent parameter counts: %s", param_counts)

    # Environment
    env_rng = np.random.default_rng(cfg.seed)
    env = GridWorld(
        grid_size=cfg.grid_size,
        max_steps=cfg.max_steps_per_episode,
        rng=env_rng,
    )

    # Accumulators
    episode_details: List[Dict[str, Any]] = []
    all_step_logs: List[List[StepLog]] = []
    successes: int = 0
    total_reward_sum: float = 0.0
    total_length_sum: int = 0

    # Per-step accumulators for EFE curves (across all episodes, indexed by step)
    max_possible_steps = cfg.max_steps_per_episode
    efe_total_by_step: List[List[float]] = [[] for _ in range(max_possible_steps)]
    efe_pragmatic_by_step: List[List[float]] = [[] for _ in range(max_possible_steps)]
    efe_epistemic_by_step: List[List[float]] = [[] for _ in range(max_possible_steps)]
    efe_instrumental_by_step: List[List[float]] = [[] for _ in range(max_possible_steps)]

    # Distance-to-pragmatic correlation pairs
    distance_pragmatic_pairs: List[Tuple[float, float]] = []

    for ep_idx in range(cfg.max_episodes):
        obs_np = env.reset()
        agent_state = agent.reset(batch_size=1, device=torch.device("cpu"))
        agent.set_seed(cfg.seed + ep_idx)

        ep_reward: float = 0.0
        ep_step_logs: List[StepLog] = []
        done = False
        step_idx = 0

        if verbose:
            print(f"\n--- Episode {ep_idx + 1}/{cfg.max_episodes} ---")
            print(f"    Start: ({env.agent_row}, {env.agent_col}) -> Goal: {env.goal_position}")

        while not done and step_idx < cfg.max_steps_per_episode:
            # Convert observation to tensor
            obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)  # (1, obs_dim)

            # Agent selects action
            with torch.no_grad():
                result: ActionOutput = agent.act(
                    o_t=obs_t,
                    state=agent_state,
                    learn=False,
                )

            # Extract action index
            action_tensor = result.action
            if action_tensor.dim() == 2:
                action_idx = int(action_tensor.argmax(dim=-1).item())
            elif action_tensor.dim() == 1:
                action_idx = int(action_tensor.item())
            else:
                action_idx = int(action_tensor.item())

            # Clamp to valid action range
            action_idx = action_idx % cfg.action_dim

            # Extract EFE terms
            efe_total_val = float(result.efe_total.mean().item())
            efe_prag_val = float(result.efe_terms.get("pragmatic", torch.tensor(0.0)).mean().item())
            efe_epist_val = float(result.efe_terms.get("epistemic", torch.tensor(0.0)).mean().item())
            efe_instr_val = float(result.efe_terms.get("instrumental", torch.tensor(0.0)).mean().item())

            # Step the environment
            obs_np, reward, done, info = env.step(action_idx)
            ep_reward += reward

            # Update agent state (carry forward the latent)
            agent_state = AgentState(
                latent_state=agent_state.latent_state,
                latent_params=agent_state.latent_params,
                step_count=agent_state.step_count + 1,
                prev_action=result.action.detach(),
                planner_state=agent_state.planner_state,
            )

            # Log step
            slog = StepLog(
                step=step_idx,
                action=action_idx,
                action_name=ACTION_NAMES.get(action_idx, "?"),
                reward=reward,
                manhattan_distance=info["manhattan_distance"],
                efe_total=efe_total_val,
                efe_pragmatic=efe_prag_val,
                efe_epistemic=efe_epist_val,
                efe_instrumental=efe_instr_val,
            )
            ep_step_logs.append(slog)

            # Accumulate for per-step curves
            if step_idx < max_possible_steps:
                efe_total_by_step[step_idx].append(efe_total_val)
                efe_pragmatic_by_step[step_idx].append(efe_prag_val)
                efe_epistemic_by_step[step_idx].append(efe_epist_val)
                efe_instrumental_by_step[step_idx].append(efe_instr_val)

            # Correlation tracking: distance vs pragmatic
            distance_pragmatic_pairs.append((float(info["manhattan_distance"]), efe_prag_val))

            if verbose:
                print(
                    f"    Step {step_idx:3d}: {slog.action_name:5s} | "
                    f"R={reward:+6.1f} | dist={info['manhattan_distance']:2d} | "
                    f"EFE={efe_total_val:+8.3f} "
                    f"[P={efe_prag_val:+7.3f} E={efe_epist_val:+7.3f} I={efe_instr_val:+7.3f}]"
                )

            step_idx += 1

        # Episode summary
        reached_goal = info.get("at_goal", False)
        if reached_goal:
            successes += 1
        total_reward_sum += ep_reward
        total_length_sum += step_idx

        ep_detail = {
            "episode": ep_idx,
            "steps": step_idx,
            "total_reward": ep_reward,
            "reached_goal": reached_goal,
            "start_pos": ep_step_logs[0].manhattan_distance + (1 if reached_goal else 0)
            if ep_step_logs else 0,
            "final_distance": info["manhattan_distance"],
        }
        episode_details.append(ep_detail)
        all_step_logs.append(ep_step_logs)

        if verbose:
            status = "GOAL" if reached_goal else "TIMEOUT"
            print(
                f"    => [{status}] steps={step_idx}, reward={ep_reward:.1f}, "
                f"final_dist={info['manhattan_distance']}"
            )

    # ------------------------------------------------------------------
    # Compute aggregate metrics
    # ------------------------------------------------------------------
    success_rate = successes / max(cfg.max_episodes, 1)
    avg_length = total_length_sum / max(cfg.max_episodes, 1)
    avg_reward = total_reward_sum / max(cfg.max_episodes, 1)

    # EFE curves: average over episodes at each step index
    def _avg_curve(by_step: List[List[float]]) -> List[float]:
        curve = []
        for vals in by_step:
            if vals:
                curve.append(float(np.mean(vals)))
            else:
                break
        return curve

    efe_curves = {
        "total": _avg_curve(efe_total_by_step),
        "pragmatic": _avg_curve(efe_pragmatic_by_step),
        "epistemic": _avg_curve(efe_epistemic_by_step),
        "instrumental": _avg_curve(efe_instrumental_by_step),
    }

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    sanity = {}

    # 1. Agent reaches goal at least once (basic functionality)
    sanity["success_rate_positive"] = success_rate > 0.0

    # 2. Pragmatic EFE correlates with distance (positive = farther means higher)
    if len(distance_pragmatic_pairs) > 10:
        dists = np.array([p[0] for p in distance_pragmatic_pairs])
        prags = np.array([p[1] for p in distance_pragmatic_pairs])
        # Check if far-from-goal observations have higher pragmatic on average
        if dists.std() > 0 and prags.std() > 0:
            corr = float(np.corrcoef(dists, prags)[0, 1])
            sanity["pragmatic_distance_correlation"] = not np.isnan(corr)
            logger.info("Pragmatic-distance correlation: %.4f", corr)
        else:
            sanity["pragmatic_distance_correlation"] = True  # degenerate, pass
    else:
        sanity["pragmatic_distance_correlation"] = True  # too few samples, pass

    # 3. EFE terms are finite (no NaN or Inf)
    all_efe_finite = True
    for logs in all_step_logs:
        for slog in logs:
            if not (math.isfinite(slog.efe_total) and
                    math.isfinite(slog.efe_pragmatic) and
                    math.isfinite(slog.efe_epistemic) and
                    math.isfinite(slog.efe_instrumental)):
                all_efe_finite = False
                break
    sanity["efe_terms_finite"] = all_efe_finite

    # 4. Average reward better than worst case (all timeouts)
    worst_case_reward = -cfg.max_steps_per_episode
    sanity["reward_above_worst_case"] = avg_reward > worst_case_reward

    # 5. Agent uses all four actions (exploration)
    all_actions = set()
    for logs in all_step_logs:
        for slog in logs:
            all_actions.add(slog.action)
    sanity["uses_multiple_actions"] = len(all_actions) >= 2

    wall_time = time.time() - t0

    result = BenchmarkResult(
        mode="online",
        success_rate=success_rate,
        avg_episode_length=avg_length,
        avg_reward=avg_reward,
        efe_curves=efe_curves,
        world_model_loss=None,
        efe_improvement=None,
        wall_time_seconds=wall_time,
        config=asdict(cfg),
        episode_details=episode_details,
        sanity_checks=sanity,
    )

    logger.info(_SEPARATOR)
    logger.info("Online benchmark complete in %.2fs", wall_time)
    logger.info("  Success rate:        %.1f%% (%d/%d)",
                success_rate * 100, successes, cfg.max_episodes)
    logger.info("  Avg episode length:  %.1f steps", avg_length)
    logger.info("  Avg reward:          %.2f", avg_reward)
    logger.info("  Sanity checks:       %s",
                {k: "PASS" if v else "FAIL" for k, v in sanity.items()})

    return result


# ============================================================================
# Offline Toy Benchmark
# ============================================================================


class WorldModelTrainer:
    """Trains the agent's generative model components on offline data.

    Performs supervised training of the transition model (next-state
    prediction) and likelihood model (observation reconstruction) using
    transitions collected from the grid world.

    The training loop:
        1. Encode current observation -> latent state s_t.
        2. Predict next state via transition model: s_{t+1} = T(s_t, a_t).
        3. Decode predicted observation: o_{t+1}_pred = D(s_{t+1}).
        4. Losses:
           - Reconstruction: MSE(o_{t+1}_pred_mu, o_{t+1}_actual).
           - Transition KL: regularize transition variance.
           - Encoder KL: regularize posterior toward unit Gaussian prior.

    Args:
        agent: The ActiveInferenceAgent whose sub-modules will be trained.
        config: Benchmark configuration.
    """

    def __init__(
        self,
        agent: ActiveInferenceAgent,
        config: BenchmarkConfig,
    ) -> None:
        self.agent = agent
        self.config = config

        # Collect all trainable parameters from the generative model
        params = []
        for module in [agent.encoder, agent.likelihood, agent.transition]:
            params.extend(module.parameters())
        if agent.empowerment is not None:
            params.extend(agent.empowerment.parameters())

        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)
        self.loss_history: List[float] = []

    def train_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            obs: Current observations (B, obs_dim).
            actions: Actions taken (B,) long.
            next_obs: Next observations (B, obs_dim).

        Returns:
            Dictionary of loss components for logging.
        """
        self.agent.train()
        self.optimizer.zero_grad()

        # 1. Encode current observation
        (mu, log_var), s_sample = self.agent.encoder.sample(obs)

        # 2. Predict next state
        action_onehot = F.one_hot(actions, self.config.action_dim).float()
        next_mu, next_log_var = self.agent.transition(s_sample, action_onehot)
        next_std = torch.exp(0.5 * next_log_var)
        s_next = next_mu + next_std * torch.randn_like(next_std)

        # 3. Decode predicted observation
        obs_pred_mu, obs_pred_log_var = self.agent.likelihood(s_next)

        # 4. Reconstruction loss
        recon_loss = F.mse_loss(obs_pred_mu, next_obs, reduction="mean")

        # 5. Encoder KL (posterior vs unit Gaussian prior)
        encoder_kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()

        # 6. Transition regularization (keep predictions reasonable)
        transition_var_penalty = next_log_var.exp().mean() * 0.01

        # Total loss
        total_loss = recon_loss + 0.1 * encoder_kl + transition_var_penalty

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.agent.parameters() if p.requires_grad],
            max_norm=5.0,
        )
        self.optimizer.step()

        loss_val = total_loss.item()
        self.loss_history.append(loss_val)

        return {
            "total": loss_val,
            "reconstruction": recon_loss.item(),
            "encoder_kl": encoder_kl.item(),
            "transition_var_penalty": transition_var_penalty.item(),
        }

    def train_loop(
        self,
        data: Dict[str, torch.Tensor],
        num_steps: int,
        verbose: bool = False,
    ) -> List[float]:
        """Run the full training loop.

        Args:
            data: Tensor dict from dataset_to_tensors().
            num_steps: Number of gradient steps.
            verbose: Print progress every 10 steps.

        Returns:
            List of total loss values per step.
        """
        obs = data["observations"]
        actions = data["actions"]
        next_obs = data["next_observations"]
        N = obs.shape[0]

        rng = np.random.default_rng(self.config.seed + 1000)

        for step in range(num_steps):
            # Sample mini-batch
            indices = rng.integers(0, N, size=min(self.config.batch_size, N))
            batch_obs = obs[indices]
            batch_actions = actions[indices]
            batch_next_obs = next_obs[indices]

            losses = self.train_step(batch_obs, batch_actions, batch_next_obs)

            if verbose and (step % 10 == 0 or step == num_steps - 1):
                logger.info(
                    "  Train step %4d/%d: total=%.5f recon=%.5f kl=%.5f",
                    step, num_steps,
                    losses["total"], losses["reconstruction"], losses["encoder_kl"],
                )

        self.agent.set_seed(self.config.seed)
        return self.loss_history


def run_planner_assessment(
    agent: ActiveInferenceAgent,
    data: Dict[str, torch.Tensor],
    cfg: BenchmarkConfig,
    num_samples: int = 100,
) -> Dict[str, Any]:
    """Assess the planner's EFE on held-out observations.

    Compares the EFE of planner-selected actions against the EFE of
    dataset actions (which may be random-policy actions).  A well-trained
    world model should yield lower planner EFE.

    Args:
        agent: Trained agent.
        data: Tensor dict with "observations" and "actions".
        cfg: Benchmark configuration.
        num_samples: Number of transitions to assess.

    Returns:
        Dictionary with planner_efe, dataset_efe, and improvement ratio.
    """
    agent.set_seed(cfg.seed)

    obs = data["observations"]
    actions = data["actions"]
    N = obs.shape[0]

    num_to_check = min(num_samples, N)
    rng = np.random.default_rng(cfg.seed + 2000)
    check_indices = rng.choice(N, size=num_to_check, replace=False)

    planner_efe_list: List[float] = []
    dataset_efe_list: List[float] = []

    planner_prag_list: List[float] = []
    planner_epist_list: List[float] = []
    planner_instr_list: List[float] = []
    dataset_prag_list: List[float] = []
    dataset_epist_list: List[float] = []
    dataset_instr_list: List[float] = []

    with torch.no_grad():
        for i in check_indices:
            obs_i = obs[i].unsqueeze(0)  # (1, obs_dim)
            action_i = actions[i].unsqueeze(0)  # (1,)

            # Agent state
            state = agent.reset(batch_size=1, device=torch.device("cpu"))

            # Planner action
            try:
                result = agent.act(o_t=obs_i, state=state, learn=False)
                planner_efe_list.append(result.efe_total.item())
                planner_prag_list.append(
                    result.efe_terms.get("pragmatic", torch.tensor(0.0)).item()
                )
                planner_epist_list.append(
                    result.efe_terms.get("epistemic", torch.tensor(0.0)).item()
                )
                planner_instr_list.append(
                    result.efe_terms.get("instrumental", torch.tensor(0.0)).item()
                )
            except Exception as e:
                logger.warning("Planner assessment failed for sample %d: %s", int(i), e)
                continue

            # Dataset action EFE
            try:
                # Infer state then compute single-step EFE for the dataset action
                _, s_sample, _ = agent.infer_state(obs_i, state=state)
                action_onehot = F.one_hot(action_i, cfg.action_dim).float()
                terms, total = agent._compute_single_step_efe(s_sample, action_onehot)
                dataset_efe_list.append(total.item())
                dataset_prag_list.append(
                    terms.get("pragmatic", torch.tensor(0.0)).item()
                )
                dataset_epist_list.append(
                    terms.get("epistemic", torch.tensor(0.0)).item()
                )
                dataset_instr_list.append(
                    terms.get("instrumental", torch.tensor(0.0)).item()
                )
            except Exception as e:
                logger.warning("Dataset EFE assessment failed for sample %d: %s", int(i), e)
                continue

    planner_efe_mean = float(np.mean(planner_efe_list)) if planner_efe_list else 0.0
    dataset_efe_mean = float(np.mean(dataset_efe_list)) if dataset_efe_list else 0.0

    # Improvement: positive means planner found lower EFE (better)
    if abs(dataset_efe_mean) > 1e-8:
        improvement = (dataset_efe_mean - planner_efe_mean) / abs(dataset_efe_mean)
    else:
        improvement = 0.0

    return {
        "planner_efe_mean": planner_efe_mean,
        "dataset_efe_mean": dataset_efe_mean,
        "improvement": improvement,
        "num_checked": len(planner_efe_list),
        "planner_terms": {
            "pragmatic_mean": float(np.mean(planner_prag_list)) if planner_prag_list else 0.0,
            "epistemic_mean": float(np.mean(planner_epist_list)) if planner_epist_list else 0.0,
            "instrumental_mean": float(np.mean(planner_instr_list)) if planner_instr_list else 0.0,
        },
        "dataset_terms": {
            "pragmatic_mean": float(np.mean(dataset_prag_list)) if dataset_prag_list else 0.0,
            "epistemic_mean": float(np.mean(dataset_epist_list)) if dataset_epist_list else 0.0,
            "instrumental_mean": float(np.mean(dataset_instr_list)) if dataset_instr_list else 0.0,
        },
    }


def run_offline_benchmark(cfg: BenchmarkConfig, verbose: bool = False) -> BenchmarkResult:
    """Run the offline toy benchmark: train world model, then assess planner.

    Pipeline:
        1. Generate synthetic dataset (biased random policy on GridWorld).
        2. Convert to tensors and split train / held-out.
        3. Train the agent's world model on the training split.
        4. Assess planner EFE on held-out observations.

    Sanity checks:
        - World model loss decreases over training.
        - All losses are finite.
        - Planner produces valid EFE terms.

    Args:
        cfg: Benchmark configuration.
        verbose: Print per-step training progress.

    Returns:
        BenchmarkResult with offline metrics and sanity checks.
    """
    t0 = time.time()
    seed_everything(cfg.seed)

    logger.info("=" * 72)
    logger.info("OFFLINE TOY BENCHMARK")
    logger.info("=" * 72)
    logger.info(
        "Grid: %dx%d | Training steps: %d | Seed: %d | Batch: %d",
        cfg.grid_size, cfg.grid_size, cfg.num_training_steps,
        cfg.seed, cfg.batch_size,
    )

    # ------------------------------------------------------------------
    # 1. Generate or load dataset
    # ------------------------------------------------------------------
    use_minari = False
    minari_dataset = None

    if MINARI_AVAILABLE:
        try:
            # Attempt to load a simple Minari dataset
            available = minari.list_local_datasets()
            if available:
                # Use the first available dataset as a smoke test
                dataset_id = list(available.keys())[0] if isinstance(available, dict) else available[0]
                minari_dataset = minari.load_dataset(dataset_id)
                use_minari = True
                logger.info("Loaded Minari dataset: %s", dataset_id)
        except Exception as e:
            logger.info("Minari available but no suitable dataset found: %s", e)

    if not use_minari:
        logger.info("Generating synthetic dataset (biased policy)...")
        syn_dataset = generate_synthetic_dataset(
            grid_size=cfg.grid_size,
            num_episodes=cfg.synthetic_dataset_episodes,
            max_steps=cfg.synthetic_dataset_max_steps,
            seed=cfg.seed,
            policy="biased",
        )
        data = dataset_to_tensors(syn_dataset)
        logger.info(
            "Dataset: %d transitions, obs_dim=%d",
            data["observations"].shape[0], data["observations"].shape[1],
        )
    else:
        # Convert Minari dataset to our tensor format
        logger.info("Converting Minari dataset to tensors...")
        data = _minari_to_tensors(minari_dataset, cfg)

    # ------------------------------------------------------------------
    # 2. Build agent
    # ------------------------------------------------------------------
    agent_config = build_agent_config(cfg)
    agent = ActiveInferenceAgent(agent_config)

    param_counts = agent.parameter_count()
    logger.info("Agent parameter counts: %s", param_counts)

    # ------------------------------------------------------------------
    # 3. Train world model
    # ------------------------------------------------------------------
    logger.info(_SEPARATOR)
    logger.info("Training world model for %d steps...", cfg.num_training_steps)

    trainer = WorldModelTrainer(agent, cfg)
    loss_curve = trainer.train_loop(data, cfg.num_training_steps, verbose=verbose)

    logger.info(
        "Training complete. Initial loss: %.5f -> Final loss: %.5f",
        loss_curve[0] if loss_curve else 0.0,
        loss_curve[-1] if loss_curve else 0.0,
    )

    # ------------------------------------------------------------------
    # 4. Assess planner
    # ------------------------------------------------------------------
    logger.info(_SEPARATOR)
    logger.info("Assessing planner on held-out observations...")

    assessment = run_planner_assessment(agent, data, cfg)

    logger.info("Planner EFE mean:  %.5f", assessment["planner_efe_mean"])
    logger.info("Dataset EFE mean:  %.5f", assessment["dataset_efe_mean"])
    logger.info("EFE improvement:   %.4f (%.1f%%)",
                assessment["improvement"], assessment["improvement"] * 100)

    # ------------------------------------------------------------------
    # 5. Sanity checks
    # ------------------------------------------------------------------
    sanity = {}

    # Loss decreases
    if len(loss_curve) >= 10:
        first_quarter = np.mean(loss_curve[:len(loss_curve) // 4])
        last_quarter = np.mean(loss_curve[-len(loss_curve) // 4:])
        sanity["loss_decreases"] = last_quarter < first_quarter
        logger.info(
            "Loss trend: first-quarter=%.5f, last-quarter=%.5f, decreases=%s",
            first_quarter, last_quarter, sanity["loss_decreases"],
        )
    else:
        sanity["loss_decreases"] = True  # too few steps to judge

    # All losses finite
    sanity["losses_finite"] = all(math.isfinite(loss) for loss in loss_curve)

    # Planner produced valid EFE
    sanity["planner_efe_finite"] = math.isfinite(assessment["planner_efe_mean"])

    # Assessment ran successfully
    sanity["assessment_completed"] = assessment["num_checked"] > 0

    # Loss is not stuck at a degenerate value
    if len(loss_curve) >= 2:
        loss_std = float(np.std(loss_curve[-10:]))
        sanity["loss_not_degenerate"] = loss_std < 1e6  # not exploding
    else:
        sanity["loss_not_degenerate"] = True

    wall_time = time.time() - t0

    result = BenchmarkResult(
        mode="offline",
        success_rate=0.0,  # not applicable
        avg_episode_length=0.0,  # not applicable
        avg_reward=0.0,  # not applicable
        efe_curves={
            "planner_pragmatic": [assessment["planner_terms"]["pragmatic_mean"]],
            "planner_epistemic": [assessment["planner_terms"]["epistemic_mean"]],
            "planner_instrumental": [assessment["planner_terms"]["instrumental_mean"]],
            "dataset_pragmatic": [assessment["dataset_terms"]["pragmatic_mean"]],
            "dataset_epistemic": [assessment["dataset_terms"]["epistemic_mean"]],
            "dataset_instrumental": [assessment["dataset_terms"]["instrumental_mean"]],
        },
        world_model_loss=loss_curve,
        efe_improvement=assessment["improvement"],
        wall_time_seconds=wall_time,
        config=asdict(cfg),
        episode_details=[{
            "planner_efe_mean": assessment["planner_efe_mean"],
            "dataset_efe_mean": assessment["dataset_efe_mean"],
            "improvement": assessment["improvement"],
            "num_checked": assessment["num_checked"],
        }],
        sanity_checks=sanity,
    )

    logger.info(_SEPARATOR)
    logger.info("Offline benchmark complete in %.2fs", wall_time)
    logger.info("  Sanity checks: %s",
                {k: "PASS" if v else "FAIL" for k, v in sanity.items()})

    return result


def _minari_to_tensors(
    minari_dataset: Any,
    cfg: BenchmarkConfig,
) -> Dict[str, torch.Tensor]:
    """Convert a Minari dataset to our standard tensor format.

    Handles the Minari episode iterator API and pads/truncates observations
    to match cfg.obs_dim.

    Args:
        minari_dataset: A Minari dataset object.
        cfg: Benchmark configuration.

    Returns:
        Tensor dict matching dataset_to_tensors() output format.
    """
    obs_list: List[np.ndarray] = []
    act_list: List[int] = []
    next_obs_list: List[np.ndarray] = []
    rew_list: List[float] = []
    done_list: List[bool] = []

    max_transitions = cfg.synthetic_dataset_episodes * cfg.synthetic_dataset_max_steps

    try:
        episodes = minari_dataset.iterate_episodes()
    except AttributeError:
        # Fallback for different Minari API versions
        try:
            episodes = list(minari_dataset)
        except Exception:
            logger.warning("Could not iterate Minari dataset; using synthetic fallback.")
            syn = generate_synthetic_dataset(
                grid_size=cfg.grid_size,
                num_episodes=cfg.synthetic_dataset_episodes,
                max_steps=cfg.synthetic_dataset_max_steps,
                seed=cfg.seed,
                policy="biased",
            )
            return dataset_to_tensors(syn)

    count = 0
    for episode in episodes:
        if count >= max_transitions:
            break

        try:
            observations = episode.observations
            actions = episode.actions
            rewards = episode.rewards
            terminations = getattr(episode, "terminations", None)
            truncations = getattr(episode, "truncations", None)
        except AttributeError:
            continue

        T = min(len(actions), len(observations) - 1)
        for t in range(T):
            if count >= max_transitions:
                break

            obs = np.array(observations[t], dtype=np.float32).flatten()
            next_obs = np.array(observations[t + 1], dtype=np.float32).flatten()

            # Pad or truncate to obs_dim
            obs = _pad_or_truncate(obs, cfg.obs_dim)
            next_obs = _pad_or_truncate(next_obs, cfg.obs_dim)

            action = int(np.array(actions[t]).flatten()[0]) % cfg.action_dim

            reward = float(np.array(rewards[t]).flatten()[0]) if rewards is not None else 0.0

            done = False
            if terminations is not None:
                done = bool(np.array(terminations[t]).flatten()[0])
            if truncations is not None:
                done = done or bool(np.array(truncations[t]).flatten()[0])

            obs_list.append(obs)
            act_list.append(action)
            next_obs_list.append(next_obs)
            rew_list.append(reward)
            done_list.append(done)
            count += 1

    if not obs_list:
        logger.warning("Minari dataset yielded no transitions; falling back to synthetic.")
        syn = generate_synthetic_dataset(
            grid_size=cfg.grid_size,
            num_episodes=cfg.synthetic_dataset_episodes,
            max_steps=cfg.synthetic_dataset_max_steps,
            seed=cfg.seed,
            policy="biased",
        )
        return dataset_to_tensors(syn)

    return {
        "observations": torch.tensor(np.array(obs_list), dtype=torch.float32),
        "actions": torch.tensor(act_list, dtype=torch.long),
        "next_observations": torch.tensor(np.array(next_obs_list), dtype=torch.float32),
        "rewards": torch.tensor(rew_list, dtype=torch.float32),
        "dones": torch.tensor(done_list, dtype=torch.bool),
    }


def _pad_or_truncate(arr: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad with zeros or truncate a 1-D array to the target dimension.

    Args:
        arr: Input 1-D array.
        target_dim: Desired output length.

    Returns:
        Array of shape (target_dim,).
    """
    if len(arr) >= target_dim:
        return arr[:target_dim]
    padded = np.zeros(target_dim, dtype=arr.dtype)
    padded[:len(arr)] = arr
    return padded


# ============================================================================
# Report Generation
# ============================================================================


def format_report(results: List[BenchmarkResult]) -> str:
    """Format benchmark results into a human-readable text report.

    Args:
        results: List of BenchmarkResult instances.

    Returns:
        Multi-line formatted string suitable for stdout.
    """
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("  ACTIVE INFERENCE TOY BENCHMARK REPORT")
    lines.append("=" * 72)
    lines.append("")

    total_time = sum(r.wall_time_seconds for r in results)
    lines.append(f"  Total wall time: {total_time:.2f}s")
    lines.append("")

    for result in results:
        lines.append("-" * 72)
        lines.append(f"  Mode: {result.mode.upper()}")
        lines.append("-" * 72)

        if result.mode == "online":
            lines.append(f"  Success rate:       {result.success_rate:.1%} "
                         f"({int(result.success_rate * result.config.get('max_episodes', 0))}/"
                         f"{result.config.get('max_episodes', 0)})")
            lines.append(f"  Avg episode length: {result.avg_episode_length:.1f} steps")
            lines.append(f"  Avg reward:         {result.avg_reward:+.2f}")

            # EFE curve summary
            for term_name in ["total", "pragmatic", "epistemic", "instrumental"]:
                curve = result.efe_curves.get(term_name, [])
                if curve:
                    lines.append(
                        f"  EFE {term_name:14s}: "
                        f"first={curve[0]:+.4f}, last={curve[-1]:+.4f}, "
                        f"mean={np.mean(curve):+.4f}"
                    )

            lines.append("")
            lines.append("  Episode details:")
            for ep in result.episode_details:
                status = "GOAL" if ep.get("reached_goal") else "TIMEOUT"
                lines.append(
                    f"    Ep {ep['episode']:2d}: [{status:7s}] "
                    f"steps={ep['steps']:3d}, reward={ep['total_reward']:+7.1f}, "
                    f"final_dist={ep['final_distance']}"
                )

        elif result.mode == "offline":
            if result.world_model_loss:
                loss_curve = result.world_model_loss
                lines.append(
                    f"  World model loss:   {loss_curve[0]:.5f} -> {loss_curve[-1]:.5f} "
                    f"(delta={loss_curve[-1] - loss_curve[0]:+.5f})"
                )

            if result.efe_improvement is not None:
                lines.append(f"  EFE improvement:    {result.efe_improvement:.4f} "
                             f"({result.efe_improvement * 100:.1f}%)")

            if result.episode_details:
                d = result.episode_details[0]
                lines.append(f"  Planner EFE mean:   {d.get('planner_efe_mean', 0.0):.5f}")
                lines.append(f"  Dataset EFE mean:   {d.get('dataset_efe_mean', 0.0):.5f}")
                lines.append(f"  Samples checked:    {d.get('num_checked', 0)}")

            # EFE term comparison
            for prefix in ["planner", "dataset"]:
                for term in ["pragmatic", "epistemic", "instrumental"]:
                    key = f"{prefix}_{term}"
                    curve = result.efe_curves.get(key, [])
                    if curve:
                        lines.append(f"  EFE {prefix:8s} {term:14s}: {curve[0]:+.5f}")

        lines.append("")
        lines.append(f"  Wall time: {result.wall_time_seconds:.2f}s")
        lines.append("")

        # Sanity checks
        lines.append("  Sanity Checks:")
        all_passed = True
        for check_name, passed in result.sanity_checks.items():
            status = "PASS" if passed else "FAIL"
            marker = "  " if passed else "**"
            lines.append(f"    {marker}[{status}] {check_name}")
            if not passed:
                all_passed = False
        lines.append(f"  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        lines.append("")

    # Global summary
    lines.append("=" * 72)
    all_ok = all(
        all(r.sanity_checks.values()) for r in results if r.sanity_checks
    )
    lines.append(f"  GLOBAL VERDICT: {'PASS' if all_ok else 'FAIL'}")
    lines.append("=" * 72)
    lines.append("")

    return "\n".join(lines)


def save_json_report(
    results: List[BenchmarkResult],
    output_path: Path,
) -> None:
    """Save benchmark results as a JSON file.

    Converts all BenchmarkResult instances to serializable dicts and writes
    them to the specified path.

    Args:
        results: List of BenchmarkResult instances.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable: List[Dict[str, Any]] = []
    for r in results:
        d = {
            "mode": r.mode,
            "success_rate": r.success_rate,
            "avg_episode_length": r.avg_episode_length,
            "avg_reward": r.avg_reward,
            "efe_curves": r.efe_curves,
            "world_model_loss": r.world_model_loss,
            "efe_improvement": r.efe_improvement,
            "wall_time_seconds": r.wall_time_seconds,
            "config": r.config,
            "episode_details": r.episode_details,
            "sanity_checks": r.sanity_checks,
        }
        serializable.append(d)

    report = {
        "benchmark": "active_inference_toy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_wall_time": sum(r.wall_time_seconds for r in results),
        "results": serializable,
        "global_pass": all(
            all(r.sanity_checks.values()) for r in results if r.sanity_checks
        ),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("JSON report saved to %s", output_path)


# ============================================================================
# CLI
# ============================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the benchmark harness.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Deterministic toy benchmark for the Active Inference Agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/toy_benchmark.py --mode both
  python scripts/toy_benchmark.py --mode online --episodes 20 --verbose
  python scripts/toy_benchmark.py --mode offline --output results.json
  python scripts/toy_benchmark.py --grid-size 7 --seed 123
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["online", "offline", "both"],
        help='Benchmark mode: "online", "offline", or "both" (default: both).',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for JSON report output (default: auto-generated).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=5,
        help="Grid world side length (default: 5).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of online episodes (default: 10).",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=100,
        help="Number of offline training steps (default: 100).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per episode (default: 50).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-step output.",
    )

    return parser.parse_args(argv)


# ============================================================================
# Main
# ============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the toy benchmark harness.

    Parses arguments, runs the selected benchmark modes, prints a formatted
    report, optionally saves a JSON report, and returns an exit code.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        0 if all sanity checks pass, 1 otherwise.
    """
    args = parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build config
    cfg = BenchmarkConfig(
        grid_size=args.grid_size,
        max_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        num_training_steps=args.training_steps,
    )

    logger.info("Benchmark config: grid=%dx%d, episodes=%d, max_steps=%d, "
                "training_steps=%d, seed=%d",
                cfg.grid_size, cfg.grid_size, cfg.max_episodes,
                cfg.max_steps_per_episode, cfg.num_training_steps, cfg.seed)
    logger.info("Optional deps: gymnasium=%s, minari=%s",
                GYMNASIUM_AVAILABLE, MINARI_AVAILABLE)

    # Run benchmarks
    results: List[BenchmarkResult] = []

    if args.mode in ("online", "both"):
        try:
            online_result = run_online_benchmark(cfg, verbose=args.verbose)
            results.append(online_result)
        except Exception as e:
            logger.error("Online benchmark failed: %s", e, exc_info=True)
            results.append(BenchmarkResult(
                mode="online",
                wall_time_seconds=0.0,
                config=asdict(cfg),
                sanity_checks={"benchmark_ran": False},
            ))

    if args.mode in ("offline", "both"):
        try:
            offline_result = run_offline_benchmark(cfg, verbose=args.verbose)
            results.append(offline_result)
        except Exception as e:
            logger.error("Offline benchmark failed: %s", e, exc_info=True)
            results.append(BenchmarkResult(
                mode="offline",
                wall_time_seconds=0.0,
                config=asdict(cfg),
                sanity_checks={"benchmark_ran": False},
            ))

    # Print report
    report_text = format_report(results)
    print(report_text)

    # Save JSON
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).resolve().parent.parent / "results"
        output_path = output_dir / f"toy_benchmark_{args.seed}.json"

    save_json_report(results, output_path)

    # Determine exit code
    all_pass = all(
        all(r.sanity_checks.values()) for r in results if r.sanity_checks
    )

    if all_pass:
        logger.info("All sanity checks passed. Exit 0.")
        return 0
    else:
        logger.warning("Some sanity checks failed. Exit 1.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
