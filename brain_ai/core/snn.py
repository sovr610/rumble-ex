"""
Spiking Neural Network Core

Main SNN architectures for feedforward and convolutional processing.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Union
from .neurons import LIFNeuron, get_surrogate
from .encoding import RateEncoder, SpikeDecoder


class SNNLinear(nn.Module):
    """
    Single spiking linear layer: Linear + LIF neuron.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate: str = "atan",
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lif = LIFNeuron(beta=beta, threshold=threshold, surrogate=surrogate)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def reset_mem(self):
        self.lif.reset_mem()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single timestep forward."""
        cur = self.linear(x)
        spk, mem = self.lif(cur)
        spk = self.dropout(spk)
        return spk, mem


class SNNCore(nn.Module):
    """
    Feedforward Spiking Neural Network.

    Processes input over multiple timesteps, accumulating spikes
    for rate-coded output.

    Args:
        input_size: Input feature dimension
        hidden_sizes: List of hidden layer sizes
        output_size: Output feature dimension
        beta: Membrane decay factor
        num_steps: Number of simulation timesteps
        surrogate: Surrogate gradient type
        dropout: Dropout probability
        output_membrane: If True, output final membrane potential instead of spikes
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        beta: float = 0.9,
        num_steps: int = 25,
        surrogate: str = "atan",
        dropout: float = 0.2,
        output_membrane: bool = False,
    ):
        super().__init__()

        self.num_steps = num_steps
        self.output_membrane = output_membrane

        # Build layers
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()

        for i in range(len(sizes) - 1):
            # Last layer has no dropout
            layer_dropout = dropout if i < len(sizes) - 2 else 0.0
            self.layers.append(
                SNNLinear(
                    sizes[i], sizes[i + 1],
                    beta=beta,
                    surrogate=surrogate,
                    dropout=layer_dropout,
                )
            )

        # Optional input encoder
        self.encoder = RateEncoder(num_steps=num_steps)
        self.decoder = SpikeDecoder(method="rate")

    def reset_mem(self):
        """Reset all neuron states."""
        for layer in self.layers:
            layer.reset_mem()

    def forward_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single timestep through all layers."""
        for layer in self.layers[:-1]:
            x, _ = layer(x)

        spk, mem = self.layers[-1](x)
        return spk, mem

    def forward(
        self,
        x: torch.Tensor,
        encode_input: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass over num_steps timesteps.

        Args:
            x: Input tensor
               - (batch, features) for static input repeated each timestep
               - (time, batch, features) for temporal input
            encode_input: If True, apply rate encoding to input

        Returns:
            spike_record: Spikes over time (time, batch, output)
            mem: Final membrane potential (batch, output)
        """
        self.reset_mem()

        # Handle input shape
        if x.dim() == 2:
            if encode_input:
                # Rate encode: (batch, features) -> (time, batch, features)
                x = self.encoder(x)
            else:
                # Repeat static input
                x = x.unsqueeze(0).repeat(self.num_steps, 1, 1)

        spike_record = []
        mem_record = []

        for t in range(x.shape[0]):
            spk, mem = self.forward_step(x[t])
            spike_record.append(spk)
            mem_record.append(mem)

        spike_record = torch.stack(spike_record)  # (time, batch, output)

        return spike_record, mem_record[-1]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions via spike rate."""
        spikes, mem = self.forward(x)

        if self.output_membrane:
            return mem.argmax(dim=-1)
        else:
            return spikes.sum(dim=0).argmax(dim=-1)

    def get_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get decoded output (rate or membrane)."""
        spikes, mem = self.forward(x)

        if self.output_membrane:
            return mem
        else:
            return self.decoder(spikes)


class SNNConv2d(nn.Module):
    """
    Spiking 2D convolutional layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        beta: float = 0.9,
        surrogate: str = "atan",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.lif = LIFNeuron(beta=beta, surrogate=surrogate)
        self.bn = nn.BatchNorm2d(out_channels)

    def reset_mem(self):
        self.lif.reset_mem()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cur = self.bn(self.conv(x))
        spk, mem = self.lif(cur)
        return spk, mem


class ConvSNN(nn.Module):
    """
    Convolutional Spiking Neural Network for vision.

    Architecture:
        Conv -> LIF -> Pool -> Conv -> LIF -> Pool -> ... -> FC -> LIF

    Args:
        input_channels: Number of input channels (3 for RGB)
        channels: List of conv channel sizes
        fc_sizes: List of fully-connected sizes
        num_classes: Output classes
        beta: Membrane decay
        num_steps: Simulation timesteps
    """

    def __init__(
        self,
        input_channels: int = 3,
        channels: List[int] = [32, 64, 128],
        fc_sizes: List[int] = [256],
        num_classes: int = 10,
        beta: float = 0.9,
        num_steps: int = 25,
        surrogate: str = "atan",
        input_size: Tuple[int, int] = (32, 32),
    ):
        super().__init__()

        self.num_steps = num_steps

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_ch = input_channels
        for out_ch in channels:
            self.conv_layers.append(
                SNNConv2d(in_ch, out_ch, beta=beta, surrogate=surrogate)
            )
            in_ch = out_ch

        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Calculate FC input size
        fc_input = channels[-1] * 4 * 4

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        fc_in = fc_input
        for fc_out in fc_sizes:
            self.fc_layers.append(
                SNNLinear(fc_in, fc_out, beta=beta, surrogate=surrogate, dropout=0.2)
            )
            fc_in = fc_out

        # Output layer
        self.output_layer = SNNLinear(fc_in, num_classes, beta=beta, surrogate=surrogate)

    def reset_mem(self):
        for layer in self.conv_layers:
            layer.reset_mem()
        for layer in self.fc_layers:
            layer.reset_mem()
        self.output_layer.reset_mem()

    def forward_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single timestep forward."""
        # Conv layers with pooling
        for i, conv in enumerate(self.conv_layers):
            x, _ = conv(x)
            if i < len(self.conv_layers) - 1:
                x = self.pool(x)

        # Adaptive pooling and flatten
        x = self.adaptive_pool(x)
        x = x.flatten(start_dim=1)

        # FC layers
        for fc in self.fc_layers:
            x, _ = fc(x)

        # Output
        spk, mem = self.output_layer(x)
        return spk, mem

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x: Input images (batch, channels, height, width)

        Returns:
            spike_record: (time, batch, num_classes)
            mem: Final membrane (batch, num_classes)
        """
        self.reset_mem()

        spike_record = []

        for t in range(self.num_steps):
            spk, mem = self.forward_step(x)
            spike_record.append(spk)

        return torch.stack(spike_record), mem

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        spikes, _ = self.forward(x)
        return spikes.sum(dim=0).argmax(dim=-1)


class ResidualSNNBlock(nn.Module):
    """
    Spiking Residual Block (SEW-ResNet style).

    Uses spike-element-wise (SEW) addition for residual connection.
    """

    def __init__(
        self,
        channels: int,
        beta: float = 0.9,
        surrogate: str = "atan",
        connect_fn: str = "ADD",  # ADD, AND, IAND
    ):
        super().__init__()

        self.conv1 = SNNConv2d(channels, channels, beta=beta, surrogate=surrogate)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.lif2 = LIFNeuron(beta=beta, surrogate=surrogate)
        self.connect_fn = connect_fn

    def reset_mem(self):
        self.conv1.reset_mem()
        self.lif2.reset_mem()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        identity = x

        out, _ = self.conv1(x)
        out = self.bn2(self.conv2(out))

        # Spike-Element-Wise connection
        if self.connect_fn == "ADD":
            out = out + identity
        elif self.connect_fn == "AND":
            out = out * identity
        elif self.connect_fn == "IAND":
            out = out * (1 - identity)

        spk, mem = self.lif2(out)
        return spk, mem


# Factory function
def create_snn(
    architecture: str,
    input_size: Union[int, Tuple[int, int, int]],
    output_size: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create SNN architectures.

    Args:
        architecture: 'feedforward', 'conv', or 'residual'
        input_size: Input dimension(s)
        output_size: Output dimension
        **kwargs: Additional arguments
    """
    if architecture == "feedforward":
        return SNNCore(
            input_size=input_size,
            hidden_sizes=kwargs.get("hidden_sizes", [256, 128]),
            output_size=output_size,
            **{k: v for k, v in kwargs.items() if k != "hidden_sizes"}
        )

    elif architecture == "conv":
        return ConvSNN(
            input_channels=input_size[0] if isinstance(input_size, tuple) else 3,
            num_classes=output_size,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown architecture: {architecture}")
