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
