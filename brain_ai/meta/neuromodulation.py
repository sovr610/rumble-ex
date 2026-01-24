"""
Neuromodulation Module

Implements neural analogs of neuromodulatory systems that control
learning and plasticity in the brain.

Four modulators corresponding to major neurotransmitter systems:
- Dopamine (DA): Reward prediction error, motivation
- Acetylcholine (ACh): Attention, learning rate in novel situations
- Norepinephrine (NE): Arousal, global gain modulation
- Serotonin (5-HT): Mood, exploration/exploitation balance

These modulators control WHEN and HOW MUCH the network learns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class NeuromodulationConfig:
    """Configuration for neuromodulation."""
    input_dim: int = 512
    hidden_dim: int = 128
    num_modulators: int = 4

    # Learning rate bounds
    min_lr_multiplier: float = 0.0
    max_lr_multiplier: float = 2.0

    # Modulator-specific settings
    baseline_activity: float = 0.5
    adaptation_rate: float = 0.1


class ModulatorNetwork(nn.Module):
    """
    Single neuromodulator network.

    Maps system state to modulator activity level.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        baseline: float = 0.5,
    ):
        super().__init__()

        self.baseline = baseline

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Activity history for adaptation
        self.register_buffer('activity_history', torch.zeros(100))
        self.register_buffer('history_idx', torch.tensor(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute modulator activity."""
        activity = torch.sigmoid(self.network(x))

        # Center around baseline
        activity = self.baseline + (activity - 0.5)

        return activity.squeeze(-1)

    def update_history(self, activity: torch.Tensor):
        """Track activity for adaptation."""
        idx = self.history_idx.item() % 100
        self.activity_history[idx] = activity.mean().detach()
        self.history_idx += 1


class DopamineSystem(ModulatorNetwork):
    """
    Dopamine-like modulator.

    Signals reward prediction error (RPE):
    - Positive RPE: Better than expected -> increase learning
    - Negative RPE: Worse than expected -> decrease learning
    - Zero RPE: As expected -> maintain

    Controls reinforcement learning and motivation.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim, baseline=0.5)

        # RPE computation
        self.value_predictor = nn.Linear(input_dim, 1)

    def compute_rpe(
        self,
        x: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reward prediction error.

        Args:
            x: Current state representation
            reward: Actual reward received

        Returns:
            rpe: Reward prediction error
            predicted_value: Expected reward
        """
        predicted_value = self.value_predictor(x).squeeze(-1)

        if reward is not None:
            rpe = reward - predicted_value
        else:
            # Without explicit reward, use surprise as proxy
            rpe = torch.zeros_like(predicted_value)

        return rpe, predicted_value


class AcetylcholineSystem(ModulatorNetwork):
    """
    Acetylcholine-like modulator.

    Signals uncertainty and novelty:
    - High ACh: Novel situation, increase attention and learning
    - Low ACh: Familiar situation, rely on learned responses

    Controls attention and learning rate modulation.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim, baseline=0.3)

        # Familiarity detector
        self.familiarity = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def compute_novelty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute novelty signal (inverse of familiarity)."""
        familiarity = self.familiarity(x).squeeze(-1)
        novelty = 1 - familiarity
        return novelty


class NorepinephrineSystem(ModulatorNetwork):
    """
    Norepinephrine-like modulator.

    Signals arousal and urgency:
    - High NE: High arousal, global gain increase
    - Low NE: Low arousal, reduced responsiveness

    Controls global gain and fight-or-flight responses.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim, baseline=0.4)

        # Arousal detector
        self.arousal = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )


class SerotoninSystem(ModulatorNetwork):
    """
    Serotonin-like modulator.

    Signals mood and temporal discounting:
    - High 5-HT: Patient, exploitation mode
    - Low 5-HT: Impulsive, exploration mode

    Controls exploration/exploitation balance and patience.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim, baseline=0.5)

        # Exploration tendency
        self.exploration = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )


class NeuromodulatoryGate(nn.Module):
    """
    Complete neuromodulatory gating system.

    Combines all modulators to produce learning rate multiplier
    and other control signals for the rest of the system.

    Args:
        config: NeuromodulationConfig with all parameters
    """

    def __init__(
        self,
        config: Optional[NeuromodulationConfig] = None,
        **kwargs,
    ):
        super().__init__()

        self.config = config or NeuromodulationConfig(**kwargs)

        # Individual modulators
        self.dopamine = DopamineSystem(
            self.config.input_dim,
            self.config.hidden_dim,
        )
        self.acetylcholine = AcetylcholineSystem(
            self.config.input_dim,
            self.config.hidden_dim,
        )
        self.norepinephrine = NorepinephrineSystem(
            self.config.input_dim,
            self.config.hidden_dim,
        )
        self.serotonin = SerotoninSystem(
            self.config.input_dim,
            self.config.hidden_dim,
        )

        # Integration network
        self.integration = nn.Sequential(
            nn.Linear(4, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

        # Exploration bonus network
        self.exploration_bonus = nn.Sequential(
            nn.Linear(4, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        x: torch.Tensor,
        anomaly_score: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        prediction_error: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute neuromodulatory signals.

        Args:
            x: Current representation (batch, input_dim)
            anomaly_score: How anomalous is current input [0, 1]
            confidence: Current confidence level [0, 1]
            prediction_error: How wrong were predictions
            reward: Actual reward (for RPE)

        Returns:
            Dict with:
                - lr_multiplier: Learning rate modifier
                - modulators: Individual modulator activities
                - exploration_bonus: Bonus for exploration
                - global_gain: Overall responsiveness
        """
        batch_size = x.shape[0]
        device = x.device

        # Compute individual modulator activities
        da = self.dopamine(x)
        ach = self.acetylcholine(x)
        ne = self.norepinephrine(x)
        sht = self.serotonin(x)

        # Incorporate external signals
        if anomaly_score is not None:
            # High anomaly -> increase ACh (novelty)
            ach = ach + 0.3 * anomaly_score.squeeze(-1)
            ach = torch.clamp(ach, 0, 1)

        if confidence is not None:
            # Low confidence -> increase NE (arousal)
            ne = ne + 0.2 * (1 - confidence.squeeze(-1))
            ne = torch.clamp(ne, 0, 1)

        if prediction_error is not None:
            # High prediction error -> increase DA
            pe_signal = torch.sigmoid(prediction_error.squeeze(-1))
            da = da + 0.2 * pe_signal
            da = torch.clamp(da, 0, 1)

        # Stack modulator activities
        modulators = torch.stack([da, ach, ne, sht], dim=-1)

        # Compute learning rate multiplier
        # DA and ACh increase learning, 5-HT moderates it
        raw_lr = self.integration(modulators).squeeze(-1)
        lr_multiplier = torch.sigmoid(raw_lr) * (
            self.config.max_lr_multiplier - self.config.min_lr_multiplier
        ) + self.config.min_lr_multiplier

        # Exploration bonus (from low 5-HT and high NE)
        exploration = self.exploration_bonus(modulators).squeeze(-1)

        # Global gain (from NE)
        global_gain = 0.5 + ne

        return {
            'lr_multiplier': lr_multiplier,
            'modulators': {
                'dopamine': da,
                'acetylcholine': ach,
                'norepinephrine': ne,
                'serotonin': sht,
            },
            'exploration_bonus': exploration,
            'global_gain': global_gain,
        }

    def modulate_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        lr_multiplier: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Modulate gradients based on neuromodulatory state.

        Args:
            gradients: Dict of parameter gradients
            lr_multiplier: Learning rate multiplier

        Returns:
            Modulated gradients
        """
        modulated = {}
        for name, grad in gradients.items():
            if grad is not None:
                # Apply modulation (broadcast across batch)
                modulated[name] = grad * lr_multiplier.mean()
        return modulated


class PlasticityController(nn.Module):
    """
    High-level plasticity controller.

    Determines learning mode based on system state:
    - Static: Inference only, no learning
    - Online: Continuous adaptation
    - Few-shot: Rapid task adaptation (MAML-style)
    """

    def __init__(
        self,
        config: Optional[NeuromodulationConfig] = None,
        **kwargs,
    ):
        super().__init__()

        self.config = config or NeuromodulationConfig(**kwargs)

        self.gate = NeuromodulatoryGate(self.config)

        # Mode predictor
        self.mode_predictor = nn.Sequential(
            nn.Linear(self.config.input_dim + 4, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 3),  # static, online, few-shot
        )

        # Thresholds
        self.static_threshold = 0.8  # High confidence -> static
        self.fewshot_threshold = 0.3  # Low confidence + novel -> few-shot

    def forward(
        self,
        x: torch.Tensor,
        anomaly_score: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Determine plasticity mode and learning parameters.

        Returns:
            Dict with mode, lr_multiplier, and detailed modulator info
        """
        # Get neuromodulatory signals
        gate_output = self.gate(
            x,
            anomaly_score=anomaly_score,
            confidence=confidence,
        )

        # Prepare features for mode prediction
        modulators = gate_output['modulators']
        mod_tensor = torch.stack([
            modulators['dopamine'],
            modulators['acetylcholine'],
            modulators['norepinephrine'],
            modulators['serotonin'],
        ], dim=-1)

        features = torch.cat([x, mod_tensor], dim=-1)
        mode_logits = self.mode_predictor(features)
        mode_probs = F.softmax(mode_logits, dim=-1)

        # Determine mode
        # 0: static, 1: online, 2: few-shot
        mode = mode_probs.argmax(dim=-1)

        return {
            'mode': mode,
            'mode_probs': mode_probs,
            'lr_multiplier': gate_output['lr_multiplier'],
            'exploration_bonus': gate_output['exploration_bonus'],
            'modulators': gate_output['modulators'],
        }


# Factory function
def create_neuromodulatory_gate(
    input_dim: int = 512,
    hidden_dim: int = 128,
    **kwargs,
) -> NeuromodulatoryGate:
    """Create neuromodulatory gate with specified configuration."""
    config = NeuromodulationConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        **kwargs,
    )
    return NeuromodulatoryGate(config)
