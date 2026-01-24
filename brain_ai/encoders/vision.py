"""
Vision Encoder

Spiking convolutional encoder for images and event-based vision data.
Outputs fixed 512-dim representation for the Global Workspace.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from ..core.neurons import LIFNeuron, get_surrogate
from ..core.encoding import RateEncoder, DeltaEncoder


class SNNConvBlock(nn.Module):
    """Convolutional block with batch norm and spiking neuron."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        beta: float = 0.9,
        surrogate: str = "atan",
        pool: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = LIFNeuron(beta=beta, surrogate=surrogate)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def reset_mem(self):
        self.lif.reset_mem()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        spk, _ = self.lif(x)
        return self.pool(spk)


class VisionEncoder(nn.Module):
    """
    Spiking Vision Encoder.

    Processes images through spiking convolutional layers,
    outputting a fixed-size representation for the Global Workspace.

    Supports:
    - Static images (RGB or grayscale)
    - Event streams (DVS cameras)
    - Video frames

    Args:
        input_channels: Number of input channels (1 for grayscale, 3 for RGB, 2 for DVS)
        output_dim: Output feature dimension (default 512 for workspace)
        channels: List of conv channel sizes
        beta: Membrane decay factor
        num_steps: Simulation timesteps for static images
        input_size: Expected input spatial size (height, width)
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 512,
        channels: List[int] = [32, 64, 128],
        beta: float = 0.9,
        num_steps: int = 25,
        surrogate: str = "atan",
        input_size: Tuple[int, int] = (224, 224),
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_dim = output_dim
        self.num_steps = num_steps

        # Build convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_ch = input_channels

        for i, out_ch in enumerate(channels):
            # Pool on all but last layer
            pool = i < len(channels) - 1
            self.conv_blocks.append(
                SNNConvBlock(
                    in_ch, out_ch,
                    beta=beta,
                    surrogate=surrogate,
                    pool=pool,
                )
            )
            in_ch = out_ch

        # Adaptive pooling to fixed spatial size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Calculate flattened size
        self.flat_size = channels[-1] * 4 * 4

        # Projection to output dimension
        self.projection = nn.Sequential(
            nn.Linear(self.flat_size, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        # Optional: rate encoder for static images
        self.rate_encoder = RateEncoder(num_steps=num_steps)

    def reset_mem(self):
        """Reset all neuron states."""
        for block in self.conv_blocks:
            block.reset_mem()

    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """Single timestep through conv layers."""
        for block in self.conv_blocks:
            x = block(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        temporal_input: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
               - Static: (batch, channels, height, width)
               - Temporal: (time, batch, channels, height, width)
            temporal_input: If True, x is already a temporal sequence

        Returns:
            features: (batch, output_dim)
        """
        self.reset_mem()

        if temporal_input:
            # Already temporal: (time, batch, C, H, W)
            num_steps = x.shape[0]
            spike_sum = torch.zeros(
                x.shape[1], self.flat_size, device=x.device
            )

            for t in range(num_steps):
                feat = self.forward_step(x[t])
                feat = self.adaptive_pool(feat)
                feat = feat.flatten(start_dim=1)
                spike_sum += feat

            # Average over time
            features = spike_sum / num_steps

        else:
            # Static image: repeat for num_steps
            spike_sum = torch.zeros(
                x.shape[0], self.flat_size, device=x.device
            )

            for _ in range(self.num_steps):
                feat = self.forward_step(x)
                feat = self.adaptive_pool(feat)
                feat = feat.flatten(start_dim=1)
                spike_sum += feat

            features = spike_sum / self.num_steps

        # Project to output dimension
        return self.projection(features)


class EventVisionEncoder(nn.Module):
    """
    Event-based Vision Encoder for DVS cameras.

    Processes event streams (x, y, t, polarity) directly
    without converting to frames first.

    Uses voxel grid representation for efficient processing.
    """

    def __init__(
        self,
        output_dim: int = 512,
        height: int = 128,
        width: int = 128,
        num_bins: int = 16,
        channels: List[int] = [32, 64, 128],
        beta: float = 0.9,
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.output_dim = output_dim

        # Input: voxel grid (num_bins, H, W)
        self.encoder = VisionEncoder(
            input_channels=num_bins,
            output_dim=output_dim,
            channels=channels,
            beta=beta,
            num_steps=1,  # Single pass through voxel grid
            input_size=(height, width),
        )

    def events_to_voxel(
        self,
        events: dict,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert events to voxel grid representation.

        Args:
            events: Dict with 't', 'x', 'y', 'p' arrays
            device: Target device

        Returns:
            voxel: (1, num_bins, height, width)
        """
        t = events['t'].astype('float32')
        x = events['x'].astype('int64')
        y = events['y'].astype('int64')
        p = events['p'].astype('float32') * 2 - 1  # -1 or +1

        # Normalize timestamps
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = (t - t_min) / (t_max - t_min) * (self.num_bins - 1)
        else:
            t_norm = torch.zeros_like(torch.from_numpy(t))

        # Build voxel grid with bilinear interpolation
        voxel = torch.zeros(self.num_bins, self.height, self.width, device=device)

        t_norm = torch.from_numpy(t_norm).to(device)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        p = torch.from_numpy(p).to(device)

        t_floor = t_norm.long().clamp(0, self.num_bins - 1)
        t_ceil = (t_floor + 1).clamp(0, self.num_bins - 1)
        dt = t_norm - t_floor.float()

        # Accumulate events
        voxel.index_put_(
            (t_floor, y, x),
            p * (1 - dt),
            accumulate=True
        )
        voxel.index_put_(
            (t_ceil, y, x),
            p * dt,
            accumulate=True
        )

        return voxel.unsqueeze(0)  # Add batch dim

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Voxel grid (batch, num_bins, height, width)
               or pre-converted event representation

        Returns:
            features: (batch, output_dim)
        """
        return self.encoder(x, temporal_input=False)


class MultiScaleVisionEncoder(nn.Module):
    """
    Multi-scale vision encoder using pyramid pooling.

    Extracts features at multiple spatial scales for robust
    representation across different object sizes.
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 512,
        beta: float = 0.9,
        num_steps: int = 25,
    ):
        super().__init__()

        self.num_steps = num_steps

        # Shared backbone
        self.backbone = nn.ModuleList([
            SNNConvBlock(input_channels, 32, beta=beta, pool=True),
            SNNConvBlock(32, 64, beta=beta, pool=True),
            SNNConvBlock(64, 128, beta=beta, pool=False),
        ])

        # Multi-scale pooling
        self.pool_scales = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),  # Global
            nn.AdaptiveAvgPool2d((2, 2)),  # 2x2
            nn.AdaptiveAvgPool2d((4, 4)),  # 4x4
        ])

        # Projection (128 * (1 + 4 + 16) = 128 * 21 = 2688)
        self.projection = nn.Linear(128 * 21, output_dim)

    def reset_mem(self):
        for block in self.backbone:
            block.reset_mem()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.reset_mem()

        features_sum = None

        for _ in range(self.num_steps):
            feat = x
            for block in self.backbone:
                feat = block(feat)

            # Multi-scale pooling
            pooled = []
            for pool in self.pool_scales:
                pooled.append(pool(feat).flatten(start_dim=1))

            step_features = torch.cat(pooled, dim=1)

            if features_sum is None:
                features_sum = step_features
            else:
                features_sum += step_features

        return self.projection(features_sum / self.num_steps)


# Factory function
def create_vision_encoder(
    encoder_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Create vision encoder by type.

    Args:
        encoder_type: 'standard', 'event', or 'multiscale'
        **kwargs: Encoder-specific arguments
    """
    if encoder_type == "standard":
        return VisionEncoder(**kwargs)
    elif encoder_type == "event":
        return EventVisionEncoder(**kwargs)
    elif encoder_type == "multiscale":
        return MultiScaleVisionEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
