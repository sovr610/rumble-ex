"""
Active Inference Decision System

Implements action selection using the Active Inference framework.

Active Inference is based on the Free Energy Principle:
- Agents minimize expected free energy (EFE) when selecting actions
- EFE balances pragmatic (goal-directed) and epistemic (information-seeking) value
- Actions are selected to both achieve goals AND reduce uncertainty

This provides a unified framework for decision-making that naturally
balances exploration vs exploitation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import math

# Try to import pymdp
try:
    import pymdp
    from pymdp import utils as pymdp_utils
    from pymdp.agent import Agent as PyMDPAgent
    PYMDP_AVAILABLE = True
except ImportError:
    PYMDP_AVAILABLE = False


@dataclass
class ActiveInferenceConfig:
    """Configuration for Active Inference agent."""
    obs_dim: int = 512  # Observation dimension (from workspace)
    state_dim: int = 64  # Latent state dimension
    action_dim: int = 10  # Number of possible actions
    hidden_dim: int = 256  # Hidden layer dimension

    planning_horizon: int = 3  # How far to plan ahead
    num_samples: int = 100  # Monte Carlo samples for EFE

    # EFE weights
    pragmatic_weight: float = 1.0  # Weight for goal-directed value
    epistemic_weight: float = 1.0  # Weight for information gain

    # Temperature for action selection
    action_temperature: float = 1.0

    # Preference learning
    learn_preferences: bool = True
    preference_prior_strength: float = 0.1


class StateEncoder(nn.Module):
    """
    Encodes observations into latent state distribution.

    Implements q(s|o) - approximate posterior over states given observations.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output mean and log variance for Gaussian
        self.mu = nn.Linear(hidden_dim, state_dim)
        self.log_var = nn.Linear(hidden_dim, state_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation to state distribution.

        Returns:
            mu: Mean of state distribution
            log_var: Log variance of state distribution
        """
        h = self.encoder(obs)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var

    def sample(
        self,
        obs: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """Sample states from the posterior."""
        mu, log_var = self.forward(obs)

        if num_samples > 1:
            mu = mu.unsqueeze(1).expand(-1, num_samples, -1)
            log_var = log_var.unsqueeze(1).expand(-1, num_samples, -1)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class GenerativeModel(nn.Module):
    """
    Generative model for Active Inference.

    Implements:
    - P(o|s): Likelihood model (how states generate observations)
    - P(s'|s,a): Transition model (how actions change states)
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Likelihood model P(o|s)
        self.likelihood = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Transition model P(s'|s,a)
        self.transition = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.trans_mu = nn.Linear(hidden_dim, state_dim)
        self.trans_log_var = nn.Linear(hidden_dim, state_dim)

    def predict_obs(self, state: torch.Tensor) -> torch.Tensor:
        """Predict observation from state."""
        return self.likelihood(state)

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state distribution given current state and action.

        Args:
            state: Current state (batch, state_dim)
            action: Action taken (batch, action_dim) or one-hot

        Returns:
            mu: Mean of next state
            log_var: Log variance of next state
        """
        if action.dim() == 1:
            action = F.one_hot(action.long(), self.action_dim).float()

        combined = torch.cat([state, action], dim=-1)
        h = self.transition(combined)
        mu = self.trans_mu(h)
        log_var = self.trans_log_var(h)
        return mu, log_var

    def sample_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Sample next state."""
        mu, log_var = self.predict_next_state(state, action)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class Preferences(nn.Module):
    """
    Learnable preferences over observations.

    Represents C in Active Inference - the desired observations.
    Can be fixed (goal specification) or learned (reward shaping).
    """

    def __init__(
        self,
        obs_dim: int,
        num_goals: int = 1,
        learnable: bool = True,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.num_goals = num_goals

        # Preferred observation patterns
        if learnable:
            self.preferences = nn.Parameter(torch.randn(num_goals, obs_dim) * 0.1)
        else:
            self.register_buffer('preferences', torch.zeros(num_goals, obs_dim))

        # Goal weights (for multi-goal scenarios)
        self.goal_weights = nn.Parameter(torch.ones(num_goals) / num_goals)

    def set_preference(self, goal_idx: int, preference: torch.Tensor):
        """Manually set a preference vector."""
        with torch.no_grad():
            self.preferences[goal_idx] = preference

    def get_preference(self) -> torch.Tensor:
        """Get weighted combination of preferences."""
        weights = F.softmax(self.goal_weights, dim=0)
        return (weights.unsqueeze(-1) * self.preferences).sum(dim=0)

    def compute_pragmatic_value(self, predicted_obs: torch.Tensor) -> torch.Tensor:
        """
        Compute pragmatic value (goal-directedness) of predicted observations.

        Higher value = closer to preferred observations.
        """
        preference = self.get_preference()

        # Negative squared distance (higher is better)
        value = -torch.sum((predicted_obs - preference) ** 2, dim=-1)

        return value


class ActiveInferenceAgent(nn.Module):
    """
    Active Inference Agent for decision-making.

    Selects actions by minimizing Expected Free Energy (EFE):
    EFE = -pragmatic_value - epistemic_value

    Where:
    - Pragmatic value: How well actions achieve goals
    - Epistemic value: How much uncertainty is reduced
    """

    def __init__(
        self,
        config: Optional[ActiveInferenceConfig] = None,
        **kwargs,
    ):
        super().__init__()

        self.config = config or ActiveInferenceConfig(**kwargs)

        # State encoder q(s|o)
        self.encoder = StateEncoder(
            obs_dim=self.config.obs_dim,
            state_dim=self.config.state_dim,
            hidden_dim=self.config.hidden_dim,
        )

        # Generative model
        self.generative = GenerativeModel(
            obs_dim=self.config.obs_dim,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        )

        # Preferences
        self.preferences = Preferences(
            obs_dim=self.config.obs_dim,
            learnable=self.config.learn_preferences,
        )

        # Action prior (habitual tendencies)
        self.action_prior = nn.Parameter(
            torch.zeros(self.config.action_dim)
        )

    def compute_efe(
        self,
        state: torch.Tensor,
        action_idx: int,
    ) -> torch.Tensor:
        """
        Compute Expected Free Energy for an action.

        EFE(a) = E_q(s')[D_KL[q(o|s') || p(o)]] - E_q(s')[H[p(o|s')]]

        Simplified as:
        EFE(a) ≈ -pragmatic_value - epistemic_value
        """
        batch_size = state.shape[0]
        device = state.device

        # Create action tensor
        action = torch.zeros(batch_size, self.config.action_dim, device=device)
        action[:, action_idx] = 1.0

        # Predict next state
        next_state_mu, next_state_log_var = self.generative.predict_next_state(
            state, action
        )

        # Sample future states for Monte Carlo estimation
        next_state_std = torch.exp(0.5 * next_state_log_var)
        next_states = next_state_mu.unsqueeze(1) + next_state_std.unsqueeze(1) * \
            torch.randn(batch_size, self.config.num_samples, self.config.state_dim, device=device)

        # Predict observations from future states
        predicted_obs = self.generative.predict_obs(
            next_states.view(-1, self.config.state_dim)
        ).view(batch_size, self.config.num_samples, -1)

        # Pragmatic value: preference alignment
        pragmatic = self.preferences.compute_pragmatic_value(predicted_obs)
        pragmatic = pragmatic.mean(dim=1)  # Average over samples

        # Epistemic value: information gain (entropy reduction)
        # Approximate as variance of predicted observations
        obs_variance = predicted_obs.var(dim=1).sum(dim=-1)
        epistemic = -obs_variance  # Negative because we want to reduce uncertainty

        # Combine
        efe = -(
            self.config.pragmatic_weight * pragmatic +
            self.config.epistemic_weight * epistemic
        )

        return efe

    def select_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Select action by minimizing Expected Free Energy.

        Args:
            observation: Current observation (batch, obs_dim)
            deterministic: If True, select argmax action

        Returns:
            action: Selected action indices (batch,)
            info: Dict with EFE values and probabilities
        """
        batch_size = observation.shape[0]
        device = observation.device

        # Encode observation to state
        state_mu, state_log_var = self.encoder(observation)

        # Sample state
        state_std = torch.exp(0.5 * state_log_var)
        state = state_mu + state_std * torch.randn_like(state_std)

        # Compute EFE for each action
        efes = []
        for a in range(self.config.action_dim):
            efe = self.compute_efe(state, a)
            efes.append(efe)

        efes = torch.stack(efes, dim=1)  # (batch, action_dim)

        # Add action prior (habitual tendencies)
        prior = F.log_softmax(self.action_prior, dim=0)
        efes = efes - prior.unsqueeze(0)

        # Convert to action probabilities
        # Lower EFE = higher probability
        action_logits = -efes / self.config.action_temperature
        action_probs = F.softmax(action_logits, dim=-1)

        # Select action
        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = torch.multinomial(action_probs, num_samples=1).squeeze(-1)

        info = {
            'efe': efes,
            'action_probs': action_probs,
            'state_mu': state_mu,
            'state_log_var': state_log_var,
        }

        return action, info

    def forward(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass - select action given observation."""
        return self.select_action(observation, deterministic)

    def update_preferences(
        self,
        target_obs: torch.Tensor,
        lr: float = 0.01,
    ):
        """Update preferences toward target observation."""
        if self.config.learn_preferences:
            current_pref = self.preferences.get_preference()
            new_pref = current_pref + lr * (target_obs - current_pref)
            self.preferences.set_preference(0, new_pref)


class ContinuousActiveInference(ActiveInferenceAgent):
    """
    Active Inference for continuous action spaces.

    Instead of discrete actions, outputs continuous control signals
    as Gaussian distribution parameters.
    """

    def __init__(
        self,
        config: Optional[ActiveInferenceConfig] = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        # Replace discrete action components with continuous
        self.policy_mu = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.action_dim),
        )

        self.policy_log_std = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.action_dim),
        )

    def select_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Select continuous action."""
        # Encode to state
        state_mu, state_log_var = self.encoder(observation)
        state_std = torch.exp(0.5 * state_log_var)
        state = state_mu + state_std * torch.randn_like(state_std)

        # Get action distribution
        action_mu = self.policy_mu(state)
        action_log_std = self.policy_log_std(state)
        action_std = torch.exp(action_log_std.clamp(-5, 2))

        # Sample action
        if deterministic:
            action = action_mu
        else:
            action = action_mu + action_std * torch.randn_like(action_std)

        info = {
            'action_mu': action_mu,
            'action_std': action_std,
            'state_mu': state_mu,
            'state_log_var': state_log_var,
        }

        return action, info


# Factory function
def create_active_inference_agent(
    obs_dim: int = 512,
    state_dim: int = 64,
    action_dim: int = 10,
    continuous: bool = False,
    **kwargs,
) -> ActiveInferenceAgent:
    """
    Create Active Inference agent.

    Args:
        obs_dim: Observation dimension
        state_dim: Latent state dimension
        action_dim: Action dimension
        continuous: If True, use continuous action space
        **kwargs: Additional config parameters
    """
    config = ActiveInferenceConfig(
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        **kwargs,
    )

    if continuous:
        return ContinuousActiveInference(config)
    else:
        return ActiveInferenceAgent(config)


# =============================================================================
# IMPROVED EFE: Enhanced Expected Free Energy with Empowerment (2025)
# =============================================================================

class EmpowermentEstimator(nn.Module):
    """
    Estimates empowerment (instrumental value) for Active Inference.
    
    Empowerment measures how many future options an action enables.
    High empowerment = action keeps many future possibilities open.
    
    Based on variational empowerment: I(a; s' | s)
    The mutual information between actions and future states.
    
    This provides an intrinsic motivation beyond just goal-seeking,
    encouraging the agent to maintain "option value".
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Source network: q(a|s) - what actions are possible from this state
        self.source = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Planning network: q(a|s, s') - infer action from state transition
        self.planning = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def compute_empowerment(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute variational lower bound on empowerment.
        
        Empowerment ≈ log q(a|s,s') - log q(a|s)
        
        Higher value = more control over future states.
        """
        # Source distribution q(a|s)
        source_logits = self.source(state)
        source_log_prob = F.log_softmax(source_logits, dim=-1)
        
        # Planning distribution q(a|s,s')
        combined = torch.cat([state, next_state], dim=-1)
        planning_logits = self.planning(combined)
        planning_log_prob = F.log_softmax(planning_logits, dim=-1)
        
        # Get log probs for taken action
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        
        source_log_p = source_log_prob.gather(-1, action.long())
        planning_log_p = planning_log_prob.gather(-1, action.long())
        
        # Empowerment lower bound
        empowerment = planning_log_p - source_log_p
        
        return empowerment.squeeze(-1)
    
    def estimate_state_empowerment(
        self,
        state: torch.Tensor,
        forward_model: nn.Module,
        num_samples: int = 32,
    ) -> torch.Tensor:
        """
        Estimate empowerment of a state by sampling actions.
        
        This gives a measure of "how much control" the agent has from this state.
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Get action distribution from source
        source_logits = self.source(state)
        action_probs = F.softmax(source_logits, dim=-1)
        
        # Sample actions
        actions = torch.multinomial(action_probs, num_samples, replacement=True)
        
        total_empowerment = 0.0
        for i in range(num_samples):
            action = actions[:, i]
            action_onehot = F.one_hot(action, self.action_dim).float()
            
            # Predict next state
            with torch.no_grad():
                next_state_mu, _ = forward_model.predict_next_state(state, action_onehot)
            
            # Compute empowerment for this action
            emp = self.compute_empowerment(state, next_state_mu, action)
            total_empowerment = total_empowerment + emp
        
        return total_empowerment / num_samples


class ImprovedEFEComputation(nn.Module):
    """
    Improved Expected Free Energy computation with three components.
    
    EFE = Pragmatic Value + Epistemic Value + Instrumental Value
    
    Where:
    - Pragmatic: How well does action achieve preferences? (goal-directed)
    - Epistemic: How much uncertainty is reduced? (information gain)
    - Instrumental: How many future options enabled? (empowerment)
    
    This extends standard Active Inference with intrinsic motivation
    from empowerment, leading to more robust exploration.
    
    Based on 2025 research on deep active inference.
    """
    
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        use_empowerment: bool = True,
        empowerment_weight: float = 0.1,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_empowerment = use_empowerment
        self.empowerment_weight = empowerment_weight
        
        # Preference model P(o) - log probability of preferred observations
        self.preference_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.preference_mu = nn.Linear(hidden_dim, obs_dim)
        self.preference_log_var = nn.Linear(hidden_dim, obs_dim)
        
        # Forward model P(s'|s,a)
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.forward_mu = nn.Linear(hidden_dim, state_dim)
        self.forward_logvar = nn.Linear(hidden_dim, state_dim)
        
        # Observation model P(o|s)
        self.observation_model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        
        # Empowerment estimator
        if use_empowerment:
            self.empowerment = EmpowermentEstimator(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
            )
        else:
            self.empowerment = None
        
        # Learnable component weights
        self.pragmatic_weight = nn.Parameter(torch.tensor(1.0))
        self.epistemic_weight = nn.Parameter(torch.tensor(1.0))
        self.instrumental_weight = nn.Parameter(torch.tensor(empowerment_weight))
    
    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state distribution."""
        if action.dim() == 1:
            action = F.one_hot(action.long(), self.action_dim).float()
        
        combined = torch.cat([state, action], dim=-1)
        h = self.forward_model(combined)
        mu = self.forward_mu(h)
        log_var = self.forward_logvar(h)
        return mu, log_var
    
    def compute_efe(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        num_samples: int = 32,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute Enhanced Expected Free Energy for action.
        
        Args:
            state: Current state (batch, state_dim)
            action: Action to evaluate (batch,) or (batch, action_dim)
            num_samples: Monte Carlo samples for expectations
            
        Returns:
            efe: Expected Free Energy (lower is better for action selection)
            components: Breakdown of EFE terms for analysis
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Handle action encoding
        if action.dim() == 1:
            action_idx = action
            action_onehot = F.one_hot(action.long(), self.action_dim).float()
        else:
            action_onehot = action
            action_idx = action.argmax(dim=-1)
        
        # Predict next state distribution
        next_mu, next_logvar = self.predict_next_state(state, action_onehot)
        
        # Sample next states for Monte Carlo
        std = torch.exp(0.5 * next_logvar)
        eps = torch.randn(num_samples, batch_size, self.state_dim, device=device)
        next_states = next_mu.unsqueeze(0) + std.unsqueeze(0) * eps  # (samples, batch, state_dim)
        
        # Predict observations from next states
        pred_obs = self.observation_model(next_states.view(-1, self.state_dim))
        pred_obs = pred_obs.view(num_samples, batch_size, self.obs_dim)
        
        # ==== 1. PRAGMATIC VALUE ====
        # KL divergence from predicted observations to preferences
        # Lower KL = closer to preferences = higher pragmatic value
        
        # Encode preferences (using first observation as anchor)
        pref_h = self.preference_encoder(pred_obs.mean(dim=0))  # Use mean prediction
        pref_mu = self.preference_mu(pref_h)
        pref_logvar = self.preference_log_var(pref_h)
        
        # Compute expected distance from preferences
        pragmatic_value = -((pred_obs.mean(dim=0) - pref_mu) ** 2).sum(dim=-1)
        
        # ==== 2. EPISTEMIC VALUE ====
        # Entropy of next state distribution (uncertainty about outcome)
        # We want to REDUCE uncertainty, so lower entropy is better
        
        # Differential entropy of Gaussian: 0.5 * log(2πe * σ²)
        epistemic_entropy = 0.5 * (next_logvar + math.log(2 * math.pi * math.e)).sum(dim=-1)
        epistemic_value = -epistemic_entropy  # Negative because we want to reduce
        
        # Alternative: Variance of predictions (sample-based)
        obs_variance = pred_obs.var(dim=0).sum(dim=-1)
        epistemic_value_alt = -obs_variance
        
        # Combine both measures
        epistemic_value = 0.5 * epistemic_value + 0.5 * epistemic_value_alt
        
        # ==== 3. INSTRUMENTAL VALUE (Empowerment) ====
        if self.use_empowerment and self.empowerment is not None:
            # Estimate empowerment from expected next state
            instrumental = self.empowerment.estimate_state_empowerment(
                next_mu,
                self,
                num_samples=8,  # Fewer samples for inner loop
            )
        else:
            instrumental = torch.zeros(batch_size, device=device)
        
        # ==== COMBINE EFE ====
        # Lower EFE = better action
        # Negate positive values (pragmatic, instrumental) since we minimize EFE
        efe = -(
            F.softplus(self.pragmatic_weight) * pragmatic_value +
            F.softplus(self.epistemic_weight) * epistemic_value +
            F.softplus(self.instrumental_weight) * instrumental
        )
        
        components = {
            'pragmatic': pragmatic_value.detach(),
            'epistemic': epistemic_value.detach(),
            'instrumental': instrumental.detach(),
            'epistemic_entropy': epistemic_entropy.detach(),
            'obs_variance': obs_variance.detach(),
            'weights': {
                'pragmatic': F.softplus(self.pragmatic_weight).item(),
                'epistemic': F.softplus(self.epistemic_weight).item(),
                'instrumental': F.softplus(self.instrumental_weight).item(),
            }
        }
        
        return efe, components


class ImprovedActiveInferenceAgent(ActiveInferenceAgent):
    """
    Active Inference Agent with improved EFE computation.
    
    Extends base agent with:
    - Three-component EFE (pragmatic + epistemic + instrumental)
    - Amortized action selection (faster inference)
    - Multi-step planning with tree search option
    """
    
    def __init__(
        self,
        config: Optional[ActiveInferenceConfig] = None,
        use_empowerment: bool = True,
        use_amortized: bool = True,
        planning_depth: int = 1,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        
        self.use_empowerment = use_empowerment
        self.use_amortized = use_amortized
        self.planning_depth = planning_depth
        
        # Improved EFE computation
        self.improved_efe = ImprovedEFEComputation(
            obs_dim=self.config.obs_dim,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            use_empowerment=use_empowerment,
        )
        
        # Amortized policy (direct action selection, trained to match EFE-optimal)
        if use_amortized:
            self.amortized_policy = nn.Sequential(
                nn.Linear(self.config.state_dim, self.config.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim, self.config.action_dim),
            )
    
    def select_action_improved(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        use_amortized: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Select action using improved EFE.
        
        Args:
            observation: Current observation (batch, obs_dim)
            deterministic: If True, select argmax
            use_amortized: If True, use amortized policy for speed
            
        Returns:
            action: Selected action
            info: Detailed information about selection
        """
        # Encode to state
        state_mu, state_log_var = self.encoder(observation)
        state = state_mu  # Use mean for action selection
        
        # Fast path: amortized policy
        if use_amortized and self.use_amortized and hasattr(self, 'amortized_policy'):
            logits = self.amortized_policy(state)
            action_probs = F.softmax(logits / self.config.action_temperature, dim=-1)
            
            if deterministic:
                action = action_probs.argmax(dim=-1)
            else:
                action = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
            
            return action, {
                'action_probs': action_probs,
                'amortized': True,
            }
        
        # Slow path: compute EFE for all actions
        batch_size = observation.shape[0]
        device = observation.device
        
        efes = []
        all_components = []
        
        for a in range(self.config.action_dim):
            action = torch.full((batch_size,), a, device=device)
            efe, components = self.improved_efe.compute_efe(state, action)
            efes.append(efe)
            all_components.append(components)
        
        efes = torch.stack(efes, dim=1)  # (batch, action_dim)
        
        # Convert to probabilities
        action_logits = -efes / self.config.action_temperature
        action_probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
        
        return action, {
            'efe': efes,
            'action_probs': action_probs,
            'components': all_components,
            'amortized': False,
        }
    
    def train_amortized_policy(
        self,
        observations: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Train amortized policy to match EFE-optimal actions.
        
        This allows fast inference at test time while maintaining
        EFE-based training.
        """
        if not self.use_amortized:
            return 0.0
        
        # Get EFE-optimal actions (no grad for target)
        with torch.no_grad():
            optimal_action, info = self.select_action_improved(
                observations, deterministic=False, use_amortized=False
            )
            target_probs = info['action_probs']
        
        # Get amortized policy predictions
        state_mu, _ = self.encoder(observations)
        logits = self.amortized_policy(state_mu)
        pred_probs = F.softmax(logits, dim=-1)
        
        # KL divergence loss
        loss = F.kl_div(
            pred_probs.log(),
            target_probs,
            reduction='batchmean',
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
