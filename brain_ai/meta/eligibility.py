"""
Eligibility Traces Module

Implements eligibility traces for biologically-plausible online learning.

Eligibility traces bridge the temporal gap between actions and outcomes:
- When a synapse fires, it becomes "eligible" for modification
- Eligibility decays exponentially over time
- When a reward/error signal arrives, eligible synapses are updated

This enables learning from delayed feedback without backpropagation
through time, matching how biological neurons learn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass


@dataclass
class EligibilityConfig:
    """Configuration for eligibility traces."""
    trace_decay: float = 0.95  # λ parameter
    learning_rate: float = 0.01
    max_trace_value: float = 1.0  # Prevent trace explosion
    trace_type: str = "accumulating"  # accumulating, replacing, dutch


class EligibilityTrace:
    """
    Single eligibility trace for a parameter tensor.

    Maintains running trace that marks recently active synapses
    as eligible for modification.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        decay: float = 0.95,
        device: torch.device = None,
        trace_type: str = "accumulating",
    ):
        self.shape = shape
        self.decay = decay
        self.trace_type = trace_type
        self.device = device or torch.device('cpu')

        # Initialize trace to zeros
        self.trace = torch.zeros(shape, device=self.device)

    def reset(self):
        """Reset trace to zeros."""
        self.trace.zero_()

    def update(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
    ):
        """
        Update trace based on pre/post synaptic activity.

        For a weight matrix W[i,j]:
        trace[i,j] = decay * trace[i,j] + post[i] * pre[j]
        """
        # Compute activity product
        if pre_activity.dim() == 1 and post_activity.dim() == 1:
            # Outer product for weight matrix
            activity = torch.outer(post_activity, pre_activity)
        else:
            # Batch outer product
            activity = torch.bmm(
                post_activity.unsqueeze(-1),
                pre_activity.unsqueeze(-2)
            ).mean(dim=0)  # Average across batch

        # Update trace based on type
        if self.trace_type == "accumulating":
            # Standard: decay + new
            self.trace = self.decay * self.trace + activity
        elif self.trace_type == "replacing":
            # Replace: max of decayed and new
            self.trace = torch.max(self.decay * self.trace, activity)
        elif self.trace_type == "dutch":
            # Dutch trace: (1 - activity) * decay * trace + activity
            self.trace = (1 - activity) * self.decay * self.trace + activity

    def get_update(
        self,
        reward_signal: torch.Tensor,
        learning_rate: float,
    ) -> torch.Tensor:
        """
        Compute weight update from trace and reward signal.

        ΔW = lr * reward * trace
        """
        return learning_rate * reward_signal * self.trace

    def clamp(self, max_value: float = 1.0):
        """Prevent trace values from exploding."""
        self.trace = torch.clamp(self.trace, -max_value, max_value)


class EligibilityNetwork(nn.Module):
    """
    Neural network layer with built-in eligibility traces.

    Enables three-factor Hebbian learning:
    ΔW = η * neuromodulator * eligibility_trace

    Where eligibility combines pre and post synaptic activity.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: Optional[EligibilityConfig] = None,
        bias: bool = True,
    ):
        super().__init__()

        self.config = config or EligibilityConfig()

        # Standard linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Eligibility traces for weights (and bias)
        self.weight_trace = EligibilityTrace(
            shape=(out_features, in_features),
            decay=self.config.trace_decay,
            trace_type=self.config.trace_type,
        )

        if bias:
            self.bias_trace = EligibilityTrace(
                shape=(out_features,),
                decay=self.config.trace_decay,
                trace_type=self.config.trace_type,
            )
        else:
            self.bias_trace = None

        # Store activities for trace update
        self.pre_activity = None
        self.post_activity = None

    def reset_traces(self):
        """Reset all eligibility traces."""
        self.weight_trace.reset()
        if self.bias_trace is not None:
            self.bias_trace.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that records activities for trace update.

        Args:
            x: Input (batch, in_features)

        Returns:
            Output (batch, out_features)
        """
        # Store pre-synaptic activity
        self.pre_activity = x.detach()

        # Linear transformation
        output = self.linear(x)

        # Store post-synaptic activity
        self.post_activity = output.detach()

        return output

    def update_traces(self):
        """Update eligibility traces based on stored activities."""
        if self.pre_activity is None or self.post_activity is None:
            return

        # Average across batch
        pre = self.pre_activity.mean(dim=0)
        post = self.post_activity.mean(dim=0)

        # Update weight trace
        self.weight_trace.update(pre, post)
        self.weight_trace.clamp(self.config.max_trace_value)

        # Update bias trace
        if self.bias_trace is not None:
            # Bias trace is just post activity
            self.bias_trace.trace = (
                self.config.trace_decay * self.bias_trace.trace + post
            )
            self.bias_trace.clamp(self.config.max_trace_value)

    def apply_learning(
        self,
        reward_signal: torch.Tensor,
        learning_rate: Optional[float] = None,
    ):
        """
        Apply three-factor learning rule.

        ΔW = lr * reward * trace
        """
        lr = learning_rate or self.config.learning_rate

        # Weight update
        weight_update = self.weight_trace.get_update(reward_signal, lr)
        with torch.no_grad():
            self.linear.weight.data += weight_update

        # Bias update
        if self.bias_trace is not None:
            bias_update = self.bias_trace.get_update(reward_signal, lr)
            with torch.no_grad():
                self.linear.bias.data += bias_update


class EligibilityMLP(nn.Module):
    """
    Multi-layer perceptron with eligibility traces.

    All layers use three-factor learning for online adaptation.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        config: Optional[EligibilityConfig] = None,
        activation: str = "relu",
    ):
        super().__init__()

        self.config = config or EligibilityConfig()

        # Build layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                EligibilityNetwork(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    config=self.config,
                )
            )

        # Activation
        self.activation = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
        }[activation]

    def reset_traces(self):
        """Reset all traces."""
        for layer in self.layers:
            layer.reset_traces()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation except on last layer
            if i < len(self.layers) - 1:
                x = self.activation(x)

        return x

    def update_traces(self):
        """Update traces in all layers."""
        for layer in self.layers:
            layer.update_traces()

    def apply_learning(
        self,
        reward_signal: torch.Tensor,
        learning_rate: Optional[float] = None,
    ):
        """Apply learning to all layers."""
        for layer in self.layers:
            layer.apply_learning(reward_signal, learning_rate)


class OnlineLearner(nn.Module):
    """
    Online learning wrapper using eligibility traces.

    Provides interface for continuous online learning with:
    1. Forward pass that updates traces
    2. Reward signal that triggers learning
    3. Optional neuromodulation of learning rate
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[EligibilityConfig] = None,
    ):
        super().__init__()

        self.config = config or EligibilityConfig()
        self.model = model

        # Create eligibility traces for all linear layers
        self.traces = {}
        self._create_traces()

        # Pre/post activity storage
        self.activities = {}

    def _create_traces(self):
        """Create traces for all linear layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.traces[name] = EligibilityTrace(
                    shape=module.weight.shape,
                    decay=self.config.trace_decay,
                    trace_type=self.config.trace_type,
                )

    def reset_traces(self):
        """Reset all traces."""
        for trace in self.traces.values():
            trace.reset()
        self.activities.clear()

    def _register_hooks(self):
        """Register hooks to capture activations."""
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                self.activities[name] = {
                    'pre': input[0].detach(),
                    'post': output.detach(),
                }
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))

        return hooks

    def forward(
        self,
        x: torch.Tensor,
        update_traces: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with trace updates.

        Args:
            x: Input tensor
            update_traces: Whether to update eligibility traces

        Returns:
            Model output
        """
        # Register hooks for this forward pass
        hooks = self._register_hooks()

        try:
            output = self.model(x)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        # Update traces
        if update_traces:
            self._update_traces()

        return output

    def _update_traces(self):
        """Update all traces from captured activities."""
        for name, trace in self.traces.items():
            if name in self.activities:
                acts = self.activities[name]
                pre = acts['pre'].mean(dim=0)
                post = acts['post'].mean(dim=0)
                trace.update(pre, post)
                trace.clamp(self.config.max_trace_value)

    def apply_reward(
        self,
        reward: torch.Tensor,
        learning_rate: Optional[float] = None,
    ):
        """
        Apply reward signal to trigger learning.

        Args:
            reward: Scalar reward/error signal
            learning_rate: Optional override for learning rate
        """
        lr = learning_rate or self.config.learning_rate

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.traces:
                trace = self.traces[name]
                update = trace.get_update(reward, lr)

                with torch.no_grad():
                    module.weight.data += update


class TemporalDifferenceTrace(nn.Module):
    """
    TD-style eligibility traces for reinforcement learning.

    Implements TD(λ) style traces where:
    - Trace accumulates as states are visited
    - TD error propagates back through trace
    """

    def __init__(
        self,
        state_dim: int,
        value_hidden: int = 128,
        trace_decay: float = 0.95,
        td_lambda: float = 0.9,
    ):
        super().__init__()

        self.trace_decay = trace_decay
        self.td_lambda = td_lambda

        # Value function
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

        # Create traces for value network
        self.traces = {}
        for name, param in self.value_net.named_parameters():
            self.traces[name] = torch.zeros_like(param)

    def reset_traces(self):
        """Reset all traces."""
        for name in self.traces:
            self.traces[name].zero_()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute value estimate."""
        return self.value_net(state)

    def td_update(
        self,
        state: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        gamma: float = 0.99,
        learning_rate: float = 0.01,
    ) -> torch.Tensor:
        """
        Perform TD(λ) update.

        Args:
            state: Current state
            reward: Received reward
            next_state: Next state
            done: Episode done flag
            gamma: Discount factor
            learning_rate: Learning rate

        Returns:
            TD error
        """
        # Compute values
        value = self.value_net(state)
        with torch.no_grad():
            next_value = self.value_net(next_state)
            target = reward + gamma * next_value * (1 - done.float())

        # TD error
        td_error = target - value

        # Compute gradients
        value.backward(torch.ones_like(value))

        # Update traces and apply learning
        with torch.no_grad():
            for name, param in self.value_net.named_parameters():
                if param.grad is not None:
                    # Update trace: e = γλe + ∇V
                    self.traces[name] = (
                        gamma * self.td_lambda * self.traces[name] +
                        param.grad
                    )

                    # Apply TD update: ΔW = α * δ * e
                    param.data += learning_rate * td_error.mean() * self.traces[name]

                    # Clear gradient
                    param.grad.zero_()

        return td_error


# Factory function
def create_eligibility_network(
    layer_sizes: List[int],
    trace_decay: float = 0.95,
    trace_type: str = "accumulating",
    **kwargs,
) -> EligibilityMLP:
    """
    Create eligibility-trace network.

    Args:
        layer_sizes: List of layer dimensions
        trace_decay: Eligibility trace decay rate
        trace_type: 'accumulating', 'replacing', or 'dutch'
    """
    config = EligibilityConfig(
        trace_decay=trace_decay,
        trace_type=trace_type,
        **kwargs,
    )
    return EligibilityMLP(layer_sizes, config)
