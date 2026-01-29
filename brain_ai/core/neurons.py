"""
Spiking Neuron Models

Implements Leaky Integrate-and-Fire (LIF) neurons with various surrogate
gradient functions for backpropagation through spikes.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable


class SurrogateGradient(torch.autograd.Function):
    """
    Base surrogate gradient for non-differentiable spike function.

    Forward: Heaviside step function (0 if x < 0, 1 if x >= 0)
    Backward: Smooth approximation for gradient flow
    """

    scale = 1.0  # Will be set by subclasses

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement backward")


class ATanSurrogate(torch.autograd.Function):
    """
    Arctangent surrogate gradient.

    Smooth approximation: (1/π) * arctan(πx) + 0.5
    Gradient: α / (2 * (1 + (παx)²))
    """

    alpha = 2.0

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        alpha = ATanSurrogate.alpha
        grad = alpha / (2 * (1 + (torch.pi * alpha * x) ** 2))
        return grad_output * grad


class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Fast sigmoid surrogate gradient.

    Gradient: slope / (2 * (1 + slope * |x|)²)
    """

    slope = 25.0

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        slope = FastSigmoidSurrogate.slope
        grad = slope / (2 * (1 + slope * x.abs()) ** 2)
        return grad_output * grad


class StraightThroughSurrogate(torch.autograd.Function):
    """
    Straight-through estimator.

    Simply passes gradients through unchanged (gradient = 1).
    Simplest but less accurate.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def get_surrogate(name: str, **kwargs) -> Callable:
    """
    Get surrogate gradient function by name.

    Args:
        name: One of 'atan', 'fast_sigmoid', 'straight_through'
        **kwargs: Parameters like alpha, slope

    Returns:
        Surrogate gradient apply function
    """
    if name == "atan":
        if "alpha" in kwargs:
            ATanSurrogate.alpha = kwargs["alpha"]
        return ATanSurrogate.apply
    elif name == "fast_sigmoid":
        if "slope" in kwargs:
            FastSigmoidSurrogate.slope = kwargs["slope"]
        return FastSigmoidSurrogate.apply
    elif name == "straight_through":
        return StraightThroughSurrogate.apply
    else:
        raise ValueError(f"Unknown surrogate: {name}. Use 'atan', 'fast_sigmoid', or 'straight_through'")


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire Neuron.

    Membrane dynamics:
        U[t+1] = β * U[t] + I[t] - S[t] * V_thresh

    Spike generation:
        S[t] = 1 if U[t] > V_thresh else 0

    Args:
        beta: Membrane potential decay factor (0 < β < 1)
        threshold: Spike threshold voltage
        reset_mechanism: 'subtract' (soft reset) or 'zero' (hard reset)
        surrogate: Surrogate gradient type
        surrogate_alpha: Parameter for surrogate gradient
        learn_beta: If True, beta becomes a learnable parameter
        learn_threshold: If True, threshold becomes learnable
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        surrogate: str = "atan",
        surrogate_alpha: float = 2.0,
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()

        self.reset_mechanism = reset_mechanism
        self.spike_fn = get_surrogate(surrogate, alpha=surrogate_alpha)

        # Learnable or fixed parameters
        if learn_beta:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("beta", torch.tensor(beta))

        if learn_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold))
        else:
            self.register_buffer("threshold", torch.tensor(threshold))

        # Membrane potential state
        self.mem = None

    def init_mem(self, batch_size: int, *shape, device: torch.device = None) -> torch.Tensor:
        """Initialize membrane potential to zeros."""
        self.mem = torch.zeros(batch_size, *shape, device=device)
        return self.mem

    def reset_mem(self):
        """Reset membrane potential state."""
        self.mem = None

    def forward(
        self,
        x: torch.Tensor,
        mem: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single timestep forward pass.

        Args:
            x: Input current (batch, *features)
            mem: Optional membrane potential override

        Returns:
            spk: Output spikes (binary)
            mem: Updated membrane potential
        """
        if mem is not None:
            self.mem = mem

        if self.mem is None:
            self.mem = torch.zeros_like(x)

        # Integrate input
        self.mem = self.beta * self.mem + x

        # Generate spike
        mem_shifted = self.mem - self.threshold
        spk = self.spike_fn(mem_shifted)

        # Reset mechanism
        if self.reset_mechanism == "subtract":
            # Soft reset: subtract threshold
            self.mem = self.mem - spk * self.threshold
        else:
            # Hard reset: set to zero
            self.mem = self.mem * (1 - spk)

        return spk, self.mem


class AdaptiveLIFNeuron(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire Neuron.

    Adds spike-frequency adaptation via a dynamic threshold:
        θ[t+1] = θ_base + β_adapt * a[t]
        a[t+1] = ρ * a[t] + S[t]

    This makes the neuron fire less frequently after recent spikes,
    matching biological adaptation behavior.
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        adaptation_beta: float = 0.1,
        adaptation_decay: float = 0.95,
        surrogate: str = "atan",
    ):
        super().__init__()

        self.beta = beta
        self.threshold_base = threshold
        self.adaptation_beta = adaptation_beta
        self.adaptation_decay = adaptation_decay
        self.spike_fn = get_surrogate(surrogate)

        self.mem = None
        self.adaptation = None

    def reset_mem(self):
        self.mem = None
        self.adaptation = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mem is None:
            self.mem = torch.zeros_like(x)
            self.adaptation = torch.zeros_like(x)

        # Adaptive threshold
        threshold = self.threshold_base + self.adaptation_beta * self.adaptation

        # Integrate
        self.mem = self.beta * self.mem + x

        # Spike
        spk = self.spike_fn(self.mem - threshold)

        # Reset
        self.mem = self.mem - spk * threshold

        # Update adaptation
        self.adaptation = self.adaptation_decay * self.adaptation + spk

        return spk, self.mem


class RecurrentLIFNeuron(nn.Module):
    """
    LIF Neuron with recurrent connections.

    Adds lateral connections between neurons in the same layer:
        U[t+1] = β * U[t] + W_in * x[t] + W_rec * S[t-1]
    """

    def __init__(
        self,
        size: int,
        beta: float = 0.9,
        threshold: float = 1.0,
        recurrent_weight_scale: float = 0.1,
        surrogate: str = "atan",
    ):
        super().__init__()

        self.beta = beta
        self.threshold = threshold
        self.spike_fn = get_surrogate(surrogate)

        # Recurrent weights
        self.recurrent = nn.Linear(size, size, bias=False)
        nn.init.normal_(self.recurrent.weight, std=recurrent_weight_scale)

        self.mem = None
        self.prev_spk = None

    def reset_mem(self):
        self.mem = None
        self.prev_spk = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mem is None:
            self.mem = torch.zeros_like(x)
            self.prev_spk = torch.zeros_like(x)

        # Recurrent input from previous spikes
        recurrent_input = self.recurrent(self.prev_spk)

        # Integrate
        self.mem = self.beta * self.mem + x + recurrent_input

        # Spike
        spk = self.spike_fn(self.mem - self.threshold)

        # Reset and update
        self.mem = self.mem - spk * self.threshold
        self.prev_spk = spk

        return spk, self.mem


class AdvancedLIFNeuron(nn.Module):
    """
    Advanced LIF Neuron with learnable delays and heterogeneous time constants.
    
    Based on latest research (2025) showing that:
    - Learnable synaptic delays improve temporal coding
    - Per-neuron heterogeneous time constants aid learning
    - Adaptive surrogate gradients improve convergence
    
    References:
    - Yu et al. (2025) "Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning"
    - Hammouamri et al. (2024) "Learning delays in SNNs"
    
    Args:
        size: Number of neurons in the layer
        beta_init: Initial membrane decay factor
        threshold: Spike threshold
        learnable_beta: If True, learn per-neuron time constants
        max_delay: Maximum learnable delay (in timesteps)
        use_delays: Enable learnable synaptic delays
        use_adaptive_threshold: Enable activity-dependent threshold
        surrogate: Surrogate gradient type
    """
    
    def __init__(
        self,
        size: int,
        beta_init: float = 0.9,
        threshold: float = 1.0,
        learnable_beta: bool = True,
        max_delay: int = 10,
        use_delays: bool = True,
        use_adaptive_threshold: bool = False,
        surrogate: str = "atan",
    ):
        super().__init__()
        
        self.size = size
        self.max_delay = max_delay
        self.use_delays = use_delays
        self.use_adaptive_threshold = use_adaptive_threshold
        self.spike_fn = get_surrogate(surrogate)
        
        # Heterogeneous time constants (per-neuron learnable beta)
        # Use logit parameterization for unconstrained optimization
        if learnable_beta:
            # Initialize around beta_init
            logit_beta = torch.log(torch.tensor(beta_init) / (1 - beta_init))
            self.log_beta = nn.Parameter(
                torch.full((size,), logit_beta.item()) + torch.randn(size) * 0.1
            )
        else:
            self.register_buffer(
                'log_beta',
                torch.full((size,), torch.log(torch.tensor(beta_init) / (1 - beta_init)))
            )
        
        # Learnable threshold
        self.register_buffer('threshold_base', torch.tensor(threshold))
        
        # Learnable synaptic delays
        if use_delays:
            # Soft attention over delay taps
            self.delay_weights = nn.Parameter(torch.zeros(size, max_delay))
            # Initialize with slight preference for small delays
            nn.init.normal_(self.delay_weights, mean=0, std=0.1)
        else:
            self.delay_weights = None
        
        # Adaptive threshold parameters
        if use_adaptive_threshold:
            self.adaptation_weight = nn.Parameter(torch.tensor(0.1))
            self.adaptation_decay = nn.Parameter(torch.tensor(0.95))
        
        # State
        self.mem = None
        self.spike_history = None  # For delay mechanism
        self.adaptation = None  # For adaptive threshold
    
    @property
    def beta(self) -> torch.Tensor:
        """Get per-neuron beta values constrained to (0, 1)."""
        return torch.sigmoid(self.log_beta)
    
    def reset_mem(self):
        """Reset all neuron states."""
        self.mem = None
        self.spike_history = None
        self.adaptation = None
    
    def apply_delays(self, spike_history: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable delays to spike history.
        
        Uses soft attention over delay taps to allow gradient flow.
        
        Args:
            spike_history: (batch, time, neurons) spike tensor
            
        Returns:
            Delayed input (batch, neurons)
        """
        if self.delay_weights is None or spike_history.shape[1] < self.max_delay:
            # No delays or not enough history
            return spike_history[:, -1]
        
        # Get last max_delay timesteps
        recent_spikes = spike_history[:, -self.max_delay:]  # (batch, max_delay, neurons)
        
        # Soft attention over delay taps
        delay_attn = torch.softmax(self.delay_weights, dim=-1)  # (neurons, max_delay)
        
        # Weighted sum over delay taps
        # recent_spikes: (batch, max_delay, neurons)
        # delay_attn: (neurons, max_delay)
        # We need: (batch, neurons)
        delayed = torch.einsum('bdn,nd->bn', recent_spikes, delay_attn)
        
        return delayed
    
    def forward(
        self,
        x: torch.Tensor,
        mem: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single timestep forward pass.
        
        Args:
            x: Input current (batch, neurons)
            mem: Optional membrane potential override
            
        Returns:
            spk: Output spikes (batch, neurons)
            mem: Updated membrane potential (batch, neurons)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize state if needed
        if mem is not None:
            self.mem = mem
        
        if self.mem is None:
            self.mem = torch.zeros(batch_size, self.size, device=device)
        
        if self.spike_history is None and self.use_delays:
            self.spike_history = torch.zeros(
                batch_size, self.max_delay, self.size, device=device
            )
        
        if self.adaptation is None and self.use_adaptive_threshold:
            self.adaptation = torch.zeros(batch_size, self.size, device=device)
        
        # Apply delays if enabled
        if self.use_delays and self.spike_history is not None:
            delayed_input = self.apply_delays(self.spike_history)
            x = x + 0.1 * delayed_input  # Recurrent delayed input
        
        # Per-neuron membrane dynamics with heterogeneous time constants
        beta = self.beta  # (neurons,)
        self.mem = beta.unsqueeze(0) * self.mem + x
        
        # Compute effective threshold (with adaptation if enabled)
        if self.use_adaptive_threshold and self.adaptation is not None:
            threshold = self.threshold_base + self.adaptation_weight * self.adaptation
        else:
            threshold = self.threshold_base
        
        # Generate spike
        mem_shifted = self.mem - threshold
        spk = self.spike_fn(mem_shifted)
        
        # Soft reset (subtract threshold where spiked)
        self.mem = self.mem - spk * threshold
        
        # Update adaptation (moving average of spike activity)
        if self.use_adaptive_threshold:
            self.adaptation = self.adaptation_decay * self.adaptation + spk
        
        # Update spike history for delays
        if self.use_delays and self.spike_history is not None:
            self.spike_history = torch.cat([
                self.spike_history[:, 1:],  # Remove oldest
                spk.unsqueeze(1),  # Add newest
            ], dim=1)
        
        return spk, self.mem
    
    def get_delay_distribution(self) -> torch.Tensor:
        """Get the learned delay distribution for visualization."""
        if self.delay_weights is None:
            return None
        return torch.softmax(self.delay_weights, dim=-1)
    
    def get_effective_delays(self) -> torch.Tensor:
        """Get effective delay per neuron (expected delay)."""
        if self.delay_weights is None:
            return None
        delay_dist = self.get_delay_distribution()
        delays = torch.arange(self.max_delay, device=delay_dist.device).float()
        return (delay_dist * delays).sum(dim=-1)
