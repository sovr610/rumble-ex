"""
mpc_demo.py -- Model Predictive Control Demo with CEM Planning

Demonstrates V-JEPA 2-AC MPC using a synthetic world model:
  1. Creates a synthetic "goal-reaching" world model
  2. Runs CEM planning for a sequence of steps (receding horizon)
  3. Visualizes convergence of the CEM distribution over iterations
  4. Plots the action sequence and trajectory toward the goal

The synthetic world model implements:
  next_repr = current_repr + 0.1 * action[:repr_dim]

This allows testing the full MPC loop without a trained model.

Usage:
    python mpc_demo.py                       # Run with default config
    python mpc_demo.py --steps 10            # 10 MPC steps
    python mpc_demo.py --horizon 5           # Planning horizon
    python mpc_demo.py --samples 256         # CEM sample count
    python mpc_demo.py --no-plot             # Skip matplotlib visualization
    python mpc_demo.py --repr-dim 7          # Representation dimension
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# Ensure assets directory is importable
SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(SKILL_DIR, "assets")
if ASSETS_DIR not in sys.path:
    sys.path.insert(0, ASSETS_DIR)

from cem_planner_template import (
    CEMPlanner,
    CEMConfig,
    clip_actions,
    batched_rollout,
)


# ---------------------------------------------------------------------------
# Synthetic World Models
# ---------------------------------------------------------------------------

class LinearWorldModel:
    """
    Linear world model: next = current + alpha * action[:repr_dim]

    This is the simplest possible model for testing CEM convergence.
    The optimal action is directly proportional to (goal - current).
    """

    def __init__(self, repr_dim: int = 16, alpha: float = 0.1):
        self.repr_dim = repr_dim
        self.alpha = alpha
        self._call_count = 0

    def predict_next(self, repr_: Tensor, action: Tensor, state: Tensor) -> Tensor:
        self._call_count += 1
        B = repr_.shape[0]
        # Use action dims modulo repr_dim
        action_clipped = action[:, :self.repr_dim].clone()
        if action_clipped.shape[1] < self.repr_dim:
            pad = torch.zeros(B, self.repr_dim - action_clipped.shape[1], device=action.device)
            action_clipped = torch.cat([action_clipped, pad], dim=1)
        return repr_ + self.alpha * action_clipped


class NonLinearWorldModel:
    """
    Nonlinear world model with damped dynamics:
      next = current + alpha * tanh(action[:repr_dim]) + 0.01 * noise

    Tests CEM on a harder, more realistic dynamics model.
    """

    def __init__(self, repr_dim: int = 7, alpha: float = 0.15, noise: float = 0.0):
        self.repr_dim = repr_dim
        self.alpha = alpha
        self.noise = noise

    def predict_next(self, repr_: Tensor, action: Tensor, state: Tensor) -> Tensor:
        B = repr_.shape[0]
        a = action[:, :self.repr_dim]
        if a.shape[1] < self.repr_dim:
            pad = torch.zeros(B, self.repr_dim - a.shape[1], device=action.device)
            a = torch.cat([a, pad], dim=1)
        delta = self.alpha * torch.tanh(a)
        if self.noise > 0:
            delta = delta + self.noise * torch.randn_like(delta)
        return repr_ + delta


# ---------------------------------------------------------------------------
# MPC Loop
# ---------------------------------------------------------------------------

@dataclass
class MPCConfig:
    mpc_steps: int = 20             # Number of execute-plan cycles
    repr_dim: int = 7               # World state representation dimension
    cem_horizon: int = 10           # Planning horizon
    cem_samples: int = 512          # CEM sample count
    cem_elites: int = 64            # CEM elite count
    cem_iterations: int = 5         # CEM iterations per planning step
    world_model_alpha: float = 0.1  # World model step size
    use_nonlinear: bool = False     # Use nonlinear world model


def run_mpc(config: MPCConfig, verbose: bool = True) -> Dict:
    """
    Run the full MPC loop: plan -> execute -> observe -> repeat.

    Returns:
        results dict with:
            'trajectory':       List of repr states
            'actions':          List of executed actions
            'distances':        List of L1 distances to goal
            'cem_cost_history': List[List] of per-step CEM cost histories
            'elapsed_s':        Total time
    """
    device = torch.device("cpu")

    # Create world model
    if config.use_nonlinear:
        world_model = NonLinearWorldModel(repr_dim=config.repr_dim, alpha=config.world_model_alpha)
        model_name = "NonLinear"
    else:
        world_model = LinearWorldModel(repr_dim=config.repr_dim, alpha=config.world_model_alpha)
        model_name = "Linear"

    if verbose:
        print(f"\nWorld Model: {model_name} (alpha={config.world_model_alpha}, dim={config.repr_dim})")

    # CEM config
    cem_cfg = CEMConfig(
        horizon=config.cem_horizon,
        num_samples=config.cem_samples,
        num_elites=config.cem_elites,
        num_iterations=config.cem_iterations,
        momentum_xyz=0.1,
        momentum_gripper=0.3,
        maxnorm=2.0,     # Relaxed for synthetic demo (real robot: 0.02m)
    )

    planner = CEMPlanner(world_model, cem_cfg)

    # Initial and goal states
    torch.manual_seed(42)
    current_repr = torch.zeros(config.repr_dim, device=device)
    goal_repr = torch.tensor(
        [0.5, -0.3, 0.8, 0.0, 0.0, 0.0, 0.4][:config.repr_dim],
        dtype=torch.float32, device=device
    )
    # Pad goal if repr_dim > 7
    if config.repr_dim > 7:
        padding = torch.randn(config.repr_dim - 7, device=device) * 0.3
        goal_repr = torch.cat([goal_repr, padding])

    current_state = torch.zeros(7, device=device)

    # Initial distance
    initial_dist = torch.mean(torch.abs(current_repr - goal_repr)).item()

    if verbose:
        print(f"Initial distance to goal: {initial_dist:.4f}")
        print(f"Goal repr: {goal_repr.cpu().numpy().round(3)}")
        print(f"\nRunning {config.mpc_steps} MPC steps...")
        print(f"{'Step':>5} {'Action':>25} {'Distance':>10} {'Time (ms)':>10}")
        print("-" * 55)

    # MPC loop
    trajectory = [current_repr.clone()]
    actions_executed = []
    distances = [initial_dist]
    cem_cost_histories = []

    t_total = time.time()

    for step in range(config.mpc_steps):
        t_step = time.time()

        # Plan action using CEM
        action, diagnostics = planner.plan(
            current_repr, goal_repr, current_state,
            return_diagnostics=True
        )
        cem_cost_histories.append(diagnostics['cost_history'])

        # Execute action in world model (single step)
        current_repr = world_model.predict_next(
            current_repr.unsqueeze(0),
            action.unsqueeze(0),
            current_state.unsqueeze(0)
        ).squeeze(0)

        # Measure progress
        dist = torch.mean(torch.abs(current_repr - goal_repr)).item()

        # Store results
        trajectory.append(current_repr.clone())
        actions_executed.append(action.clone())
        distances.append(dist)

        step_ms = (time.time() - t_step) * 1000
        if verbose:
            a_xyz = action[:3].numpy().round(4)
            print(f"{step+1:>5} [{a_xyz[0]:+.4f}, {a_xyz[1]:+.4f}, {a_xyz[2]:+.4f}] "
                  f"{dist:>10.4f} {step_ms:>9.1f}")

    elapsed = time.time() - t_total
    final_dist = distances[-1]

    if verbose:
        print(f"\nFinal distance: {final_dist:.4f} (initial: {initial_dist:.4f})")
        improvement = (1 - final_dist / initial_dist) * 100 if initial_dist > 0 else 0
        print(f"Distance reduction: {improvement:.1f}%")
        print(f"Total time: {elapsed:.2f}s ({elapsed*1000/config.mpc_steps:.0f}ms/step)")

    return {
        'trajectory':        trajectory,
        'actions':           actions_executed,
        'distances':         distances,
        'cem_cost_histories': cem_cost_histories,
        'elapsed_s':         elapsed,
        'initial_dist':      initial_dist,
        'final_dist':        final_dist,
        'goal_repr':         goal_repr,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_results(results: Dict, config: MPCConfig, save_path: Optional[str] = None):
    """
    Create a multi-panel visualization of MPC results.

    Panel 1: Distance to goal over MPC steps
    Panel 2: CEM cost convergence per planning step (last 3 steps)
    Panel 3: Executed action components over steps
    Panel 4: First 3 state dimensions vs goal
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[WARNING] matplotlib not installed. Skipping visualization.")
        print("  Install with: pip install matplotlib")
        return

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    steps = list(range(len(results['distances'])))
    goal_repr = results['goal_repr'].numpy()

    # --- Panel 1: Distance to goal ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, results['distances'], 'b-o', markersize=3, linewidth=1.5)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Perfect goal')
    ax1.set_xlabel('MPC Step')
    ax1.set_ylabel('L1 Distance to Goal')
    ax1.set_title('Distance to Goal over MPC Steps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Annotate start and end
    ax1.annotate(f'Start: {results["initial_dist"]:.3f}',
                 xy=(0, results['distances'][0]),
                 xytext=(3, results['distances'][0] * 0.9),
                 fontsize=8)
    ax1.annotate(f'End: {results["final_dist"]:.3f}',
                 xy=(len(steps)-1, results['distances'][-1]),
                 xytext=(len(steps)-5, results['distances'][-1] * 1.1),
                 fontsize=8)

    # --- Panel 2: CEM convergence per planning step ---
    ax2 = fig.add_subplot(gs[0, 1])
    n_to_show = min(5, len(results['cem_cost_histories']))
    # Show every (total/5)-th step to spread across the run
    step_interval = max(1, len(results['cem_cost_histories']) // n_to_show)
    for i in range(0, len(results['cem_cost_histories']), step_interval):
        hist = results['cem_cost_histories'][i]
        alpha_val = 0.4 + 0.6 * (i / max(len(results['cem_cost_histories']) - 1, 1))
        ax2.plot(hist, '-o', markersize=4, alpha=alpha_val, label=f'MPC step {i+1}')
    ax2.set_xlabel('CEM Iteration')
    ax2.set_ylabel('Best Cost (L1)')
    ax2.set_title('CEM Cost Convergence per Planning Step')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Action components over steps ---
    ax3 = fig.add_subplot(gs[1, 0])
    if results['actions']:
        actions_np = torch.stack(results['actions']).numpy()
        action_steps = list(range(1, len(results['actions']) + 1))
        labels = ['dx', 'dy', 'dz', 'roll', 'pitch', 'yaw', 'gripper']
        colors = ['r', 'g', 'b', 'orange', 'purple', 'brown', 'black']

        for i in range(min(3, actions_np.shape[1])):  # Show xyz only
            ax3.plot(action_steps, actions_np[:, i], '-', color=colors[i],
                     alpha=0.7, linewidth=1.5, label=labels[i])
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('MPC Step')
        ax3.set_ylabel('Action Value')
        ax3.set_title('Executed Action Components (XYZ)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # --- Panel 4: State trajectory vs goal ---
    ax4 = fig.add_subplot(gs[1, 1])
    if results['trajectory']:
        traj_np = torch.stack(results['trajectory']).numpy()
        n_dims = min(3, traj_np.shape[1])
        traj_steps = list(range(len(results['trajectory'])))
        dim_colors = ['r', 'g', 'b']

        for i in range(n_dims):
            ax4.plot(traj_steps, traj_np[:, i], '-', color=dim_colors[i],
                     alpha=0.7, linewidth=1.5, label=f'dim {i}')
            ax4.axhline(y=goal_repr[i], color=dim_colors[i],
                        linestyle='--', alpha=0.4, label=f'goal dim {i}')

        ax4.set_xlabel('MPC Step')
        ax4.set_ylabel('State Value')
        ax4.set_title('State Trajectory vs Goal (first 3 dims)')
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)

    # Overall title
    improvement = (1 - results['final_dist'] / results['initial_dist']) * 100
    fig.suptitle(
        f"V-JEPA 2-AC MPC Demo | "
        f"{config.mpc_steps} steps, horizon={config.cem_horizon}, "
        f"samples={config.cem_samples} | "
        f"Improvement: {improvement:.1f}%",
        fontsize=11, fontweight='bold'
    )

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CEM Sigma Visualization
# ---------------------------------------------------------------------------

def visualize_cem_sigma_collapse(config: MPCConfig, save_path: Optional[str] = None):
    """
    Show sigma collapsing over CEM iterations on a single planning step.
    This illustrates how CEM concentrates probability mass.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    world_model = LinearWorldModel(repr_dim=config.repr_dim, alpha=config.world_model_alpha)
    cem_cfg = CEMConfig(
        horizon=config.cem_horizon,
        num_samples=config.cem_samples,
        num_elites=config.cem_elites,
        num_iterations=config.cem_iterations,
        momentum_xyz=0.0,  # No momentum for clean visualization
        momentum_gripper=0.0,
        maxnorm=2.0,
    )

    # Track sigma per iteration by monkey-patching
    sigma_history = []
    original_plan = planner_plan  # local copy
    class TrackingPlanner(CEMPlanner):
        def plan(self, current_repr, goal_repr, current_state, return_diagnostics=False):
            cfg = self.config
            device = current_repr.device
            mu    = torch.zeros(cfg.horizon, 7, device=device)
            sigma = torch.full((cfg.horizon, 7), cfg.sigma_init, device=device)
            cost_history = []
            for iteration in range(cfg.num_iterations):
                noise = torch.randn(cfg.num_samples, cfg.horizon, 7, device=device)
                sequences = mu.unsqueeze(0).expand(cfg.num_samples, -1, -1) + \
                            sigma.unsqueeze(0).expand(cfg.num_samples, -1, -1) * noise
                sequences = clip_actions(sequences, cfg.maxnorm, cfg.gripper_min, cfg.gripper_max)
                with torch.no_grad():
                    final_reprs = batched_rollout(self.world_model, current_repr, sequences, current_state)
                goal_expanded = goal_repr.unsqueeze(0).expand(cfg.num_samples, -1)
                costs = torch.mean(torch.abs(final_reprs - goal_expanded), dim=-1)
                cost_history.append(costs.min().item())
                elite_idxs = torch.argsort(costs)[:cfg.num_elites]
                elites = sequences[elite_idxs]
                new_mu    = elites.mean(dim=0)
                new_sigma = elites.std(dim=0).clamp(min=cfg.sigma_min)
                mu_next = mu.clone()
                mu_next[..., 0:3] = cfg.momentum_xyz * mu[..., 0:3] + (1 - cfg.momentum_xyz) * new_mu[..., 0:3]
                mu_next[..., 3:6] = 0.0
                mu_next[..., 6] = cfg.momentum_gripper * mu[..., 6] + (1 - cfg.momentum_gripper) * new_mu[..., 6]
                sigma = (cfg.momentum_xyz * sigma + (1 - cfg.momentum_xyz) * new_sigma).clamp(min=cfg.sigma_min)
                mu = mu_next
                sigma_history.append(sigma.mean().item())
            first_action = clip_actions(mu[0].clone(), cfg.maxnorm, cfg.gripper_min, cfg.gripper_max)
            if return_diagnostics:
                return first_action, {'cost_history': cost_history, 'final_mu': mu, 'final_sigma': sigma}
            return first_action

    # We don't need the monkey-patch reference
    del original_plan

    planner = TrackingPlanner(world_model, cem_cfg)
    torch.manual_seed(42)
    current = torch.zeros(config.repr_dim)
    goal = torch.tensor([0.5, -0.3, 0.8, 0.0, 0.0, 0.0, 0.4][:config.repr_dim], dtype=torch.float32)
    if config.repr_dim > 7:
        goal = torch.cat([goal, torch.randn(config.repr_dim - 7) * 0.3])

    _, diag = planner.plan(current, goal, torch.zeros(7), return_diagnostics=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Sigma collapse
    axes[0].plot(range(1, len(sigma_history) + 1), sigma_history, 'b-o', linewidth=2)
    axes[0].set_xlabel('CEM Iteration')
    axes[0].set_ylabel('Mean Sigma (exploration spread)')
    axes[0].set_title('CEM Sigma Collapse Over Iterations')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Cost convergence
    axes[1].plot(range(1, len(diag['cost_history']) + 1), diag['cost_history'], 'r-s', linewidth=2)
    axes[1].set_xlabel('CEM Iteration')
    axes[1].set_ylabel('Best Cost (L1 to goal)')
    axes[1].set_title('CEM Cost Convergence')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('CEM Distribution Evolution (Single Planning Step)', fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Sigma collapse figure saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Numerical Summary
# ---------------------------------------------------------------------------

def print_summary(results: Dict, config: MPCConfig):
    """Print a concise numerical summary of MPC results."""
    print("\n" + "=" * 55)
    print("MPC DEMO SUMMARY")
    print("=" * 55)
    print(f"  MPC steps:            {config.mpc_steps}")
    print(f"  CEM horizon:          {config.cem_horizon}")
    print(f"  CEM samples:          {config.cem_samples}")
    print(f"  CEM iterations:       {config.cem_iterations}")
    print(f"  Repr dim:             {config.repr_dim}")
    print(f"  World model:          {'NonLinear' if config.use_nonlinear else 'Linear'}")
    print()
    print(f"  Initial distance:     {results['initial_dist']:.4f}")
    print(f"  Final distance:       {results['final_dist']:.4f}")
    dist_init = results['initial_dist']
    dist_final = results['final_dist']
    improvement = (1 - dist_final / dist_init) * 100 if dist_init > 0 else 0
    print(f"  Improvement:          {improvement:.1f}%")
    print()
    print(f"  Total time:           {results['elapsed_s']:.2f}s")
    ms_per_step = results['elapsed_s'] * 1000 / config.mpc_steps
    print(f"  Time per step:        {ms_per_step:.1f}ms")
    print()

    # CEM stats
    all_costs = results['cem_cost_histories']
    first_iterations_init = [h[0] for h in all_costs]
    first_iterations_final = [h[-1] for h in all_costs]
    avg_cem_improvement = np.mean([
        (1 - f / i) * 100 if i > 0 else 0
        for i, f in zip(first_iterations_init, first_iterations_final)
    ])
    print(f"  Avg CEM improvement/step: {avg_cem_improvement:.1f}%")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="V-JEPA 2-AC MPC Demo with CEM Planning"
    )
    parser.add_argument("--steps", type=int, default=15, help="Number of MPC steps")
    parser.add_argument("--horizon", type=int, default=10, help="CEM planning horizon")
    parser.add_argument("--samples", type=int, default=256, help="CEM samples")
    parser.add_argument("--elites", type=int, default=32, help="CEM elites")
    parser.add_argument("--iterations", type=int, default=5, help="CEM iterations")
    parser.add_argument("--repr-dim", type=int, default=7, help="Representation dimension")
    parser.add_argument("--alpha", type=float, default=0.15, help="World model alpha")
    parser.add_argument("--nonlinear", action="store_true", help="Use nonlinear world model")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plots")
    parser.add_argument("--save-fig", type=str, default=None,
                        help="Save figure to this path (e.g. mpc_results.png)")
    parser.add_argument("--sigma-fig", type=str, default=None,
                        help="Save sigma collapse figure to this path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    return parser.parse_args()


def planner_plan(self, *args, **kwargs):
    """Placeholder for monkey-patch reference. Not used."""
    pass


if __name__ == "__main__":
    args = parse_args()

    config = MPCConfig(
        mpc_steps=args.steps,
        repr_dim=args.repr_dim,
        cem_horizon=args.horizon,
        cem_samples=args.samples,
        cem_elites=args.elites,
        cem_iterations=args.iterations,
        world_model_alpha=args.alpha,
        use_nonlinear=args.nonlinear,
    )

    print("=" * 55)
    print("V-JEPA 2-AC MPC Demo")
    print("=" * 55)

    # Run MPC
    results = run_mpc(config, verbose=not args.quiet)

    # Print summary
    print_summary(results, config)

    # Visualize
    if not args.no_plot:
        save_path = args.save_fig
        visualize_results(results, config, save_path=save_path)

        if args.sigma_fig:
            visualize_cem_sigma_collapse(config, save_path=args.sigma_fig)

    # Verify CEM is actually converging
    dist_final = results['final_dist']
    dist_init  = results['initial_dist']
    if dist_init > 1e-6:
        improvement = (1 - dist_final / dist_init) * 100
        if improvement >= 50:
            print(f"\nDEMO PASSED: {improvement:.1f}% improvement achieved.")
        else:
            print(f"\nDEMO WARNING: only {improvement:.1f}% improvement. "
                  f"Try --samples 512 or --iterations 8 for better convergence.")
    else:
        print("\nDEMO PASSED: Initial distance was near zero.")
