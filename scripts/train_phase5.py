#!/usr/bin/env python3
"""
Phase 5 Training Script: Active Inference Decision System

Trains the Active Inference agent for goal-directed decision making.
Tests the balance between pragmatic (goal-seeking) and epistemic (exploration) behavior.

Target: Agent learns optimal policy in grid-world navigation task

Usage:
    python scripts/train_phase5.py
    python scripts/train_phase5.py --episodes 1000 --planning-horizon 5
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_ai.decision.active_inference import (
    ActiveInferenceAgent,
    ActiveInferenceConfig,
    StateEncoder,
    GenerativeModel,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Active Inference Agent")
    parser.add_argument("--mode", type=str, default="dev",
                        choices=["dev", "production", "production_3b", "production_1b"],
                        help="Training mode")
    parser.add_argument("--episodes", type=int, default=None, help="Training episodes")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per episode")
    parser.add_argument("--grid-size", type=int, default=None, help="Grid world size")
    parser.add_argument("--planning-horizon", type=int, default=None, help="Planning horizon")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epistemic-weight", type=float, default=None, help="Exploration weight")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--save-path", type=str, default=None, help="Save path")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision")
    return parser.parse_args()


def get_mode_config(mode: str) -> dict:
    """Get configuration based on training mode."""
    configs = {
        "dev": {
            "episodes": 1000,
            "max_steps": 30,
            "grid_size": 5,
            "planning_horizon": 3,
            "lr": 3e-3,
            "epistemic_weight": 0.5,
            "hidden_dim": 256,
            "save_path": "checkpoints/active_inference_dev.pth",
        },
        "production_1b": {
            "episodes": 5000,
            "max_steps": 100,
            "grid_size": 10,
            "planning_horizon": 5,
            "lr": 1e-3,
            "epistemic_weight": 0.3,
            "hidden_dim": 1024,
            "save_path": "checkpoints/active_inference_1b.pth",
        },
        "production_3b": {
            "episodes": 10000,
            "max_steps": 200,
            "grid_size": 15,
            "planning_horizon": 8,
            "lr": 5e-4,
            "epistemic_weight": 0.2,
            "hidden_dim": 2048,
            "save_path": "checkpoints/active_inference_3b.pth",
        },
        "production": {  # 7B scale
            "episodes": 20000,
            "max_steps": 500,
            "grid_size": 20,
            "planning_horizon": 10,
            "lr": 1e-4,
            "epistemic_weight": 0.1,
            "hidden_dim": 4096,
            "save_path": "checkpoints/active_inference_7b.pth",
        },
    }
    return configs.get(mode, configs["dev"])


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class GridWorldEnv:
    """
    Simple grid-world environment for testing Active Inference.
    
    Features:
    - Navigation task with goal locations
    - Partially observable (limited view)
    - Stochastic transitions
    - Reward at goal, small penalty for movement
    """
    
    def __init__(
        self,
        size: int = 8,
        num_goals: int = 1,
        obs_dim: int = 64,
        stochastic: bool = True,
    ):
        self.size = size
        self.num_goals = num_goals
        self.obs_dim = obs_dim
        self.stochastic = stochastic
        
        # Actions: 0=up, 1=right, 2=down, 3=left, 4=stay
        self.num_actions = 5
        
        self.reset()
    
    def reset(self) -> torch.Tensor:
        """Reset environment and return initial observation."""
        # Random agent position
        self.agent_pos = np.array([
            np.random.randint(0, self.size),
            np.random.randint(0, self.size),
        ])
        
        # Random goal position (not at agent)
        while True:
            self.goal_pos = np.array([
                np.random.randint(0, self.size),
                np.random.randint(0, self.size),
            ])
            if not np.array_equal(self.goal_pos, self.agent_pos):
                break
        
        self.steps = 0
        return self._get_observation()
    
    def _get_observation(self) -> torch.Tensor:
        """Get current observation (partial, egocentric)."""
        obs = np.zeros(self.obs_dim)
        
        # Encode agent position (normalized)
        obs[0] = self.agent_pos[0] / self.size
        obs[1] = self.agent_pos[1] / self.size
        
        # Encode relative goal direction
        rel_goal = self.goal_pos - self.agent_pos
        obs[2] = rel_goal[0] / self.size
        obs[3] = rel_goal[1] / self.size
        
        # Distance to goal
        dist = np.sqrt(np.sum(rel_goal ** 2))
        obs[4] = dist / (self.size * np.sqrt(2))
        
        # Local observation (3x3 grid around agent)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                pos = self.agent_pos + np.array([dx, dy])
                idx = 5 + (dx + 1) * 3 + (dy + 1)
                
                if 0 <= pos[0] < self.size and 0 <= pos[1] < self.size:
                    # Check if goal nearby
                    if np.array_equal(pos, self.goal_pos):
                        obs[idx] = 1.0
                    else:
                        obs[idx] = 0.5  # Valid position
                else:
                    obs[idx] = -1.0  # Wall/boundary
        
        # Add noise for partial observability
        if self.stochastic:
            obs += np.random.randn(self.obs_dim) * 0.05
        
        return torch.tensor(obs, dtype=torch.float32)
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Take action in environment.
        
        Returns: observation, reward, done, info
        """
        # Action effects (with stochasticity)
        movements = {
            0: np.array([-1, 0]),  # up
            1: np.array([0, 1]),   # right
            2: np.array([1, 0]),   # down
            3: np.array([0, -1]),  # left
            4: np.array([0, 0]),   # stay
        }
        
        # Stochastic transition
        if self.stochastic and np.random.rand() < 0.1:
            action = np.random.randint(0, self.num_actions)
        
        # Apply movement
        new_pos = self.agent_pos + movements[action]
        
        # Check boundaries
        new_pos = np.clip(new_pos, 0, self.size - 1)
        self.agent_pos = new_pos
        
        # Check goal
        at_goal = np.array_equal(self.agent_pos, self.goal_pos)
        
        # Compute reward
        if at_goal:
            reward = 10.0
        else:
            # Small penalty for movement (encourages efficiency)
            reward = -0.1
            # Distance-based shaping
            dist = np.sqrt(np.sum((self.goal_pos - self.agent_pos) ** 2))
            reward += (1.0 - dist / (self.size * np.sqrt(2))) * 0.1
        
        self.steps += 1
        done = at_goal or self.steps >= 100
        
        return self._get_observation(), reward, done, {'at_goal': at_goal}


class NeuralActiveInferenceAgent(nn.Module):
    """
    Neural network-based Active Inference agent.
    
    Learns to minimize expected free energy by:
    1. Encoding observations into latent states
    2. Learning a generative model (transition + likelihood)
    3. Computing expected free energy for action selection
    """
    
    def __init__(
        self,
        obs_dim: int = 64,
        state_dim: int = 32,
        action_dim: int = 5,
        hidden_dim: int = 128,
        planning_horizon: int = 3,
        pragmatic_weight: float = 1.0,
        epistemic_weight: float = 1.0,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.planning_horizon = planning_horizon
        self.pragmatic_weight = pragmatic_weight
        self.epistemic_weight = epistemic_weight
        
        # State encoder q(s|o)
        self.encoder = StateEncoder(obs_dim, state_dim, hidden_dim)
        
        # Generative model
        self.generative_model = GenerativeModel(
            obs_dim, state_dim, action_dim, hidden_dim
        )
        
        # Preference model (learned goals)
        self.preference = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Policy network for action selection
        self.policy = nn.Sequential(
            nn.Linear(state_dim + obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def encode_state(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to latent state distribution."""
        return self.encoder(obs)
    
    def compute_efe(
        self,
        state: torch.Tensor,
        action_onehot: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Expected Free Energy for an action.
        
        EFE = -E[log P(o|pi)] - E[KL[q(s'|pi)||p(s'|o,pi)]]
            = Pragmatic value (goal achievement) + Epistemic value (info gain)
        """
        batch_size = state.shape[0]
        
        # Predict next state p(s'|s,a)
        next_state_mu, next_state_logvar = self.generative_model.predict_next_state(
            state, action_onehot
        )
        
        # Sample next state
        next_state_std = torch.exp(0.5 * next_state_logvar)
        next_state = next_state_mu + torch.randn_like(next_state_std) * next_state_std
        
        # Predict observation p(o|s')
        predicted_obs = self.generative_model.predict_obs(next_state)
        
        # Pragmatic value: preference for predicted observations
        pragmatic = self.preference(predicted_obs).squeeze(-1)
        
        # Epistemic value: state uncertainty (entropy)
        state_entropy = 0.5 * next_state_logvar.sum(dim=-1)
        epistemic = state_entropy  # Higher entropy = more to learn
        
        # EFE (negate because we want to minimize it)
        efe = -(self.pragmatic_weight * pragmatic + self.epistemic_weight * epistemic)
        
        return efe
    
    def select_action(
        self,
        obs: torch.Tensor,
        temperature: float = 1.0,
        explore: bool = True,
    ) -> Tuple[int, dict]:
        """Select action using learned policy."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Encode state
        state_mu, state_logvar = self.encode_state(obs)
        state = state_mu  # Use mean for action selection
        
        # Get action probabilities from policy
        policy_input = torch.cat([state, obs], dim=-1)
        logits = self.policy(policy_input)
        
        # Compute EFE for each action (for analysis)
        efes = []
        for a in range(self.action_dim):
            action_onehot = F.one_hot(
                torch.tensor([a], device=obs.device),
                self.action_dim
            ).float()
            efe = self.compute_efe(state, action_onehot, obs)
            efes.append(efe)
        efes = torch.stack(efes, dim=-1).squeeze(0)
        
        # Combine policy logits with EFE (negative EFE = good)
        combined = logits - efes.unsqueeze(0) * 0.5
        
        # Apply temperature and sample
        probs = F.softmax(combined / temperature, dim=-1)
        
        if explore:
            action = torch.multinomial(probs, 1).item()
        else:
            action = probs.argmax(dim=-1).item()
        
        return action, {
            'probs': probs.detach(),
            'efes': efes.detach(),
            'state': state.detach(),
        }
    
    def compute_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        
        Trains:
        1. State encoder (reconstruction)
        2. Generative model (prediction accuracy)
        3. Policy (maximize reward, minimize EFE)
        """
        batch_size = obs.shape[0]
        
        # Encode states
        state_mu, state_logvar = self.encode_state(obs)
        next_state_mu, next_state_logvar = self.encode_state(next_obs)
        
        # Sample states
        state = state_mu + torch.exp(0.5 * state_logvar) * torch.randn_like(state_mu)
        
        # Reconstruction loss
        reconstructed_obs = self.generative_model.predict_obs(state)
        recon_loss = F.mse_loss(reconstructed_obs, obs)
        
        # Transition prediction loss
        action_onehot = F.one_hot(action.long(), self.action_dim).float()
        pred_next_mu, pred_next_logvar = self.generative_model.predict_next_state(
            state, action_onehot
        )
        trans_loss = F.mse_loss(pred_next_mu, next_state_mu.detach())
        
        # KL divergence for VAE regularization
        kl_loss = -0.5 * torch.mean(
            1 + state_logvar - state_mu.pow(2) - state_logvar.exp()
        )
        
        # Policy loss (maximize reward)
        policy_input = torch.cat([state.detach(), obs], dim=-1)
        logits = self.policy(policy_input)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, action.long().unsqueeze(1)).squeeze(1)
        policy_loss = -(selected_log_probs * reward).mean()
        
        # Preference learning
        pref_pred = self.preference(obs).squeeze(-1)
        pref_loss = F.mse_loss(pref_pred, reward)
        
        # Total loss
        total_loss = recon_loss + trans_loss + 0.1 * kl_loss + policy_loss + 0.5 * pref_loss
        
        return {
            'total': total_loss,
            'recon': recon_loss,
            'trans': trans_loss,
            'kl': kl_loss,
            'policy': policy_loss,
            'pref': pref_loss,
        }


def train_episode(
    agent: NeuralActiveInferenceAgent,
    env: GridWorldEnv,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    explore: bool = True,
) -> Tuple[float, int, bool]:
    """Run one training episode."""
    obs = env.reset().to(device)
    
    episode_reward = 0.0
    transitions = []
    
    for step in range(100):
        # Select action
        action, info = agent.select_action(obs, temperature=1.0, explore=explore)
        
        # Take step
        next_obs, reward, done, env_info = env.step(action)
        next_obs = next_obs.to(device)
        
        # Store transition
        transitions.append({
            'obs': obs,
            'action': torch.tensor([action], device=device, dtype=torch.long),
            'reward': torch.tensor([reward], device=device, dtype=torch.float32),
            'next_obs': next_obs,
            'done': torch.tensor([float(done)], device=device, dtype=torch.float32),
        })
        
        episode_reward += reward
        obs = next_obs
        
        if done:
            break
    
    # Train on episode transitions
    if len(transitions) > 0:
        # Batch transitions
        batch_obs = torch.stack([t['obs'] for t in transitions])
        batch_action = torch.cat([t['action'] for t in transitions])
        batch_reward = torch.cat([t['reward'] for t in transitions])
        batch_next_obs = torch.stack([t['next_obs'] for t in transitions])
        batch_done = torch.cat([t['done'] for t in transitions])
        
        # Compute loss and update
        optimizer.zero_grad()
        losses = agent.compute_loss(
            batch_obs, batch_action, batch_reward,
            batch_next_obs, batch_done
        )
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()
    
    return episode_reward, step + 1, env_info.get('at_goal', False)


def evaluate(
    agent: NeuralActiveInferenceAgent,
    env: GridWorldEnv,
    device: torch.device,
    num_episodes: int = 50,
) -> Tuple[float, float, float]:
    """Evaluate agent without exploration."""
    total_reward = 0.0
    total_steps = 0
    success_count = 0
    
    for _ in range(num_episodes):
        obs = env.reset().to(device)
        episode_reward = 0.0
        
        for step in range(100):
            action, _ = agent.select_action(obs, temperature=0.1, explore=False)
            next_obs, reward, done, info = env.step(action)
            next_obs = next_obs.to(device)
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                if info.get('at_goal', False):
                    success_count += 1
                break
        
        total_reward += episode_reward
        total_steps += step + 1
    
    return (
        total_reward / num_episodes,
        total_steps / num_episodes,
        success_count / num_episodes,
    )


def main():
    args = parse_args()
    device = get_device(args.device)
    
    # Get mode-specific configuration
    mode_config = get_mode_config(args.mode)
    
    # Override with command-line arguments if provided
    episodes = args.episodes or mode_config["episodes"]
    max_steps = args.max_steps or mode_config["max_steps"]
    grid_size = args.grid_size or mode_config["grid_size"]
    planning_horizon = args.planning_horizon or mode_config["planning_horizon"]
    lr = args.lr or mode_config["lr"]
    epistemic_weight = args.epistemic_weight or mode_config["epistemic_weight"]
    hidden_dim = mode_config["hidden_dim"]
    save_path = args.save_path or mode_config["save_path"]
    
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    print(f"  - Episodes: {episodes}")
    print(f"  - Max steps: {max_steps}")
    print(f"  - Grid size: {grid_size}")
    print(f"  - Planning horizon: {planning_horizon}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Epistemic weight: {epistemic_weight}")
    print(f"  - Hidden dim: {hidden_dim}")
    
    # Create environment (smaller grid = easier to learn)
    env = GridWorldEnv(size=grid_size, stochastic=False)  # Deterministic for easier learning
    print(f"\nCreated GridWorld environment: {env.size}x{env.size} (deterministic)")
    
    # Create agent with mode-specific dimensions
    agent = NeuralActiveInferenceAgent(
        obs_dim=64 if args.mode == "dev" else 256,
        state_dim=32 if args.mode == "dev" else 128,
        action_dim=5,
        hidden_dim=hidden_dim,
        planning_horizon=planning_horizon,
        epistemic_weight=epistemic_weight,
    ).to(device)
    
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Active Inference Agent parameters: {total_params:,}")
    
    # Optimizer with mode-specific settings
    optimizer = torch.optim.AdamW(
        agent.parameters(), 
        lr=lr,
        weight_decay=0.01 if args.mode != "dev" else 0.0,
        betas=(0.9, 0.95)
    )
    
    best_success_rate = 0.0
    
    print(f"\nTraining for {episodes} episodes...")
    print("=" * 60)
    
    episode_rewards = []
    
    for episode in range(1, episodes + 1):
        # Train episode
        reward, steps, success = train_episode(
            agent, env, optimizer, device, explore=True
        )
        episode_rewards.append(reward)
        
        # Log progress
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            eval_reward, eval_steps, eval_success = evaluate(
                agent, env, device, num_episodes=30
            )
            
            print(f"\nEpisode {episode}/{episodes}")
            print(f"  Train Avg Reward (last 50): {avg_reward:.2f}")
            print(f"  Eval Avg Reward: {eval_reward:.2f}")
            print(f"  Eval Avg Steps: {eval_steps:.1f}")
            print(f"  Eval Success Rate: {eval_success*100:.1f}%")
            
            if eval_success > best_success_rate:
                best_success_rate = eval_success
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "episode": episode,
                    "mode": args.mode,
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "success_rate": eval_success,
                    "config": mode_config,
                }, save_path)
                print(f"  New best! Saved to {save_path}")
            
            print("-" * 60)
    
    # Final evaluation
    final_reward, final_steps, final_success = evaluate(
        agent, env, device, num_episodes=100
    )
    
    print("\n" + "=" * 60)
    print(f"Training complete.")
    print(f"  Final Success Rate: {final_success*100:.1f}%")
    print(f"  Final Avg Reward: {final_reward:.2f}")
    print(f"  Final Avg Steps: {final_steps:.1f}")
    print(f"  Best Success Rate: {best_success_rate*100:.1f}%")
    print(f"Mode: {args.mode} | Final checkpoint: {save_path}")
    
    # Validation gate (adjusted for mode)
    target_success = 0.80 if args.mode == "dev" else 0.90
    if final_success >= target_success:
        print(f"\n[PASS] PHASE 5 VALIDATION PASSED: Achieved {target_success*100}%+ success rate")
    elif final_success >= target_success - 0.30:
        print(f"\n[PARTIAL] PHASE 5 PARTIAL: Achieved {final_success*100:.1f}% (target: {target_success*100}%)")
    else:
        print(f"\n[FAIL] PHASE 5 NOT PASSED: {final_success*100:.1f}% < {target_success*100}% target")


if __name__ == "__main__":
    main()
