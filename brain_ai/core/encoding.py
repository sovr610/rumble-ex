"""
Spike Encoding Schemes

Converts continuous values to spike trains and vice versa.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal


class RateEncoder(nn.Module):
    """
    Rate Coding Encoder.

    Encodes continuous values as spike probabilities.
    Higher values = higher firing rate.

    Methods:
        - 'bernoulli': Stochastic spikes with P(spike) = value
        - 'poisson': Poisson process with rate = value
        - 'deterministic': Fixed threshold comparison
    """

    def __init__(
        self,
        num_steps: int = 25,
        method: Literal["bernoulli", "poisson", "deterministic"] = "bernoulli",
        gain: float = 1.0,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.method = method
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tensor as spike train.

        Args:
            x: Input values in [0, 1] or normalized, shape (batch, *features)

        Returns:
            spikes: Spike train, shape (time, batch, *features)
        """
        # Normalize to [0, 1] probability range
        x_norm = torch.clamp(x * self.gain, 0, 1)

        if self.method == "bernoulli":
            # Each timestep, spike with probability = value
            spikes = torch.rand(self.num_steps, *x_norm.shape, device=x.device) < x_norm
            return spikes.float()

        elif self.method == "poisson":
            # Poisson-distributed spikes
            # Lambda = value * scaling factor
            rate = x_norm * 0.3  # Scale to reasonable firing rate
            spikes = torch.rand(self.num_steps, *x_norm.shape, device=x.device) < rate
            return spikes.float()

        else:  # deterministic
            # Spike when cumulative sum crosses threshold
            spikes = []
            accumulator = torch.zeros_like(x_norm)
            for t in range(self.num_steps):
                accumulator = accumulator + x_norm
                spike = (accumulator >= 1.0).float()
                accumulator = accumulator - spike
                spikes.append(spike)
            return torch.stack(spikes)


class TemporalEncoder(nn.Module):
    """
    Temporal (Time-to-First-Spike) Encoder.

    Encodes values as spike timing: higher values spike earlier.
    More efficient than rate coding but harder to train.
    """

    def __init__(
        self,
        num_steps: int = 25,
        tau: float = 5.0,  # Time constant
    ):
        super().__init__()
        self.num_steps = num_steps
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input as time-to-first-spike.

        Higher values spike at earlier timesteps.
        """
        # Compute spike time: t = tau * log(1 / x)
        # Higher x = lower t = earlier spike
        x_clamp = torch.clamp(x, 0.01, 1.0)  # Avoid log(0)
        spike_times = self.tau * torch.log(1 / x_clamp)
        spike_times = torch.clamp(spike_times, 0, self.num_steps - 1)

        # Create spike train
        spikes = torch.zeros(self.num_steps, *x.shape, device=x.device)

        for t in range(self.num_steps):
            # Spike at computed time
            spikes[t] = (spike_times.int() == t).float()

        return spikes


class LatencyEncoder(nn.Module):
    """
    Latency Encoder with linear relationship.

    Maps value linearly to spike latency.
    """

    def __init__(
        self,
        num_steps: int = 25,
        normalize: bool = True,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x_min = x.min()
            x_max = x.max()
            x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        else:
            x_norm = torch.clamp(x, 0, 1)

        # Higher value = earlier spike (lower latency)
        # Latency = (1 - x) * num_steps
        latencies = ((1 - x_norm) * (self.num_steps - 1)).int()

        spikes = torch.zeros(self.num_steps, *x.shape, device=x.device)
        for t in range(self.num_steps):
            spikes[t] = (latencies == t).float()

        return spikes


class PopulationEncoder(nn.Module):
    """
    Population Coding Encoder.

    Uses multiple neurons with overlapping receptive fields.
    Each value activates a subset of neurons based on Gaussian tuning curves.
    """

    def __init__(
        self,
        input_dim: int,
        num_neurons_per_dim: int = 10,
        num_steps: int = 25,
        sigma: float = 0.2,
    ):
        super().__init__()
        self.num_neurons = num_neurons_per_dim
        self.num_steps = num_steps
        self.sigma = sigma

        # Tuning curve centers spread across [0, 1]
        centers = torch.linspace(0, 1, num_neurons_per_dim)
        self.register_buffer("centers", centers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode each input dimension across multiple neurons.

        Args:
            x: Shape (batch, input_dim)

        Returns:
            spikes: Shape (time, batch, input_dim * num_neurons)
        """
        batch_size = x.shape[0]

        # Compute activation for each neuron (Gaussian tuning)
        # Shape: (batch, input_dim, num_neurons)
        x_expanded = x.unsqueeze(-1)  # (batch, input_dim, 1)
        activations = torch.exp(-((x_expanded - self.centers) ** 2) / (2 * self.sigma ** 2))

        # Flatten: (batch, input_dim * num_neurons)
        activations = activations.flatten(start_dim=1)

        # Rate encode the activations
        spikes = torch.rand(self.num_steps, *activations.shape, device=x.device) < activations
        return spikes.float()


class SpikeDecoder(nn.Module):
    """
    Decode spike trains back to continuous values.
    """

    def __init__(
        self,
        method: Literal["rate", "first_spike", "membrane"] = "rate",
    ):
        super().__init__()
        self.method = method

    def forward(
        self,
        spikes: torch.Tensor,
        membrane: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode spike train to continuous output.

        Args:
            spikes: Spike train (time, batch, features) or (batch, features)
            membrane: Final membrane potential (for 'membrane' method)
        """
        if self.method == "rate":
            # Sum spikes over time and normalize
            if spikes.dim() > 2:
                return spikes.sum(dim=0) / spikes.shape[0]
            return spikes

        elif self.method == "first_spike":
            # Time of first spike (inverse latency)
            if spikes.dim() <= 2:
                return spikes

            first_spike_time = torch.argmax(spikes, dim=0).float()
            # Invert: earlier spike = higher value
            max_time = spikes.shape[0]
            return 1 - (first_spike_time / max_time)

        elif self.method == "membrane":
            # Use membrane potential directly
            if membrane is not None:
                return membrane
            # Fallback to rate coding
            return spikes.sum(dim=0) / spikes.shape[0] if spikes.dim() > 2 else spikes


class DeltaEncoder(nn.Module):
    """
    Delta (Change-based) Encoder.

    Generates spikes only when input changes significantly.
    Mimics event-driven sensors like DVS cameras.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        num_steps: int = 25,
    ):
        super().__init__()
        self.threshold = threshold
        self.num_steps = num_steps
        self.prev_input = None

    def reset(self):
        self.prev_input = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate ON/OFF spikes based on input changes.

        Returns:
            spikes: Shape (time, batch, features * 2) - ON and OFF channels
        """
        if self.prev_input is None:
            self.prev_input = torch.zeros_like(x)

        # Compute change
        delta = x - self.prev_input

        # ON spikes (positive change)
        on_spikes = (delta > self.threshold).float()

        # OFF spikes (negative change)
        off_spikes = (delta < -self.threshold).float()

        # Update state
        self.prev_input = x.clone()

        # Combine channels
        spikes = torch.cat([on_spikes, off_spikes], dim=-1)

        # Repeat for num_steps (event persists briefly)
        return spikes.unsqueeze(0).repeat(self.num_steps, *([1] * spikes.dim()))
