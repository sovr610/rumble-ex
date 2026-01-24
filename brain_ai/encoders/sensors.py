"""
Sensor Encoder

Liquid Time-Constant Networks for processing temporal sensor data
(IMU, proprioception, environmental sensors, time-series).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

from ..core.neurons import LIFNeuron


class LiquidTimeConstant(nn.Module):
    """
    Liquid Time-Constant (LTC) cell.

    Implements continuous-time RNN with input-dependent time constants:
        dx/dt = (-x + f(x, I)) / τ(x, I)

    Where τ adapts based on input, enabling flexible temporal dynamics.

    This is a simplified implementation. For full features, use ncps library.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        tau_min: float = 0.1,
        tau_max: float = 10.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Input transformation
        self.input_layer = nn.Linear(input_size + hidden_size, hidden_size)

        # Time constant network
        self.tau_layer = nn.Linear(input_size + hidden_size, hidden_size)

        # Output transformation
        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        dt: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single timestep forward.

        Args:
            x: Input (batch, input_size)
            h: Hidden state (batch, hidden_size)
            dt: Time step size

        Returns:
            output: (batch, hidden_size)
            h_new: Updated hidden state
        """
        batch_size = x.shape[0]

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=-1)

        # Compute input transformation
        f_xh = torch.tanh(self.input_layer(combined))

        # Compute adaptive time constant
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(
            self.tau_layer(combined)
        )

        # Update hidden state: h_new = h + dt * (-h + f(x,h)) / tau
        dh = (-h + f_xh) / tau
        h_new = h + dt * dh

        # Output
        output = self.output_layer(h_new)

        return output, h_new


class ClosedFormContinuous(nn.Module):
    """
    Closed-form Continuous-time (CfC) cell.

    Faster than LTC - uses closed-form solution instead of ODE integration.
    Provides similar expressiveness with better computational efficiency.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # Backbone network
        self.backbone = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )

        # Time constant (fixed, learned)
        self.tau = nn.Parameter(torch.ones(hidden_size))

        # Interpolation network
        self.ff1 = nn.Linear(input_size + hidden_size, hidden_size)
        self.ff2 = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single timestep forward using closed-form solution.
        """
        batch_size = x.shape[0]

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        combined = torch.cat([x, h], dim=-1)

        # Compute new state candidate
        ff = torch.tanh(self.ff1(combined))
        ff = self.ff2(ff)

        # Sigmoid gate based on time constant
        t_interp = torch.sigmoid(-dt * torch.abs(self.tau))

        # Interpolate between old and new state
        h_new = h * t_interp + ff * (1 - t_interp)

        return h_new, h_new


class SensorEncoder(nn.Module):
    """
    Liquid Neural Network Sensor Encoder.

    Processes temporal sensor streams (IMU, proprioception, etc.)
    using liquid time-constant dynamics.

    Supports:
    - IMU data (accelerometer, gyroscope)
    - Proprioceptive signals (joint angles, velocities)
    - Environmental sensors (temperature, pressure, etc.)
    - Generic time-series data

    Args:
        input_dim: Number of sensor channels
        output_dim: Output feature dimension (default 512)
        hidden_dim: Hidden state dimension
        num_layers: Number of liquid layers
        cell_type: 'ltc' or 'cfc'
        use_spike_output: Convert output to spikes
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        cell_type: str = "cfc",
        use_spike_output: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_spike_output = use_spike_output

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Liquid layers
        self.layers = nn.ModuleList()
        CellClass = ClosedFormContinuous if cell_type == "cfc" else LiquidTimeConstant

        for i in range(num_layers):
            in_size = hidden_dim
            self.layers.append(CellClass(in_size, hidden_dim))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

        # Optional spike output
        if use_spike_output:
            self.spike_lif = LIFNeuron(beta=0.9)

        # Hidden states
        self.hidden_states = None

    def reset_state(self):
        """Reset hidden states for new sequence."""
        self.hidden_states = None
        if self.use_spike_output:
            self.spike_lif.reset_mem()

    def forward(
        self,
        x: torch.Tensor,
        dt: float = 1.0,
        return_sequence: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Sensor input
               - (batch, seq_len, input_dim) for sequences
               - (batch, input_dim) for single timestep
            dt: Time step size
            return_sequence: If True, return all timesteps

        Returns:
            features: (batch, output_dim) or (batch, seq_len, output_dim)
        """
        # Handle single timestep input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        batch_size, seq_len, _ = x.shape

        # Initialize hidden states if needed
        if self.hidden_states is None:
            self.hidden_states = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]

        outputs = []

        for t in range(seq_len):
            # Input projection
            h = self.input_proj(x[:, t, :])

            # Process through liquid layers
            for i, layer in enumerate(self.layers):
                h, self.hidden_states[i] = layer(h, self.hidden_states[i], dt)

            # Output projection
            out = self.output_proj(h)

            # Optional spike conversion
            if self.use_spike_output:
                out, _ = self.spike_lif(out)

            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, output_dim)

        if return_sequence:
            return outputs
        else:
            return outputs[:, -1, :]  # Last timestep


class IMUEncoder(nn.Module):
    """
    Specialized encoder for IMU (Inertial Measurement Unit) data.

    Handles:
    - Accelerometer (3-axis)
    - Gyroscope (3-axis)
    - Magnetometer (3-axis, optional)
    """

    def __init__(
        self,
        output_dim: int = 512,
        include_magnetometer: bool = False,
        hidden_dim: int = 128,
    ):
        super().__init__()

        input_dim = 9 if include_magnetometer else 6

        # Separate processing for accel and gyro
        self.accel_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(),
        )
        self.gyro_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(),
        )

        if include_magnetometer:
            self.mag_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),
                nn.ReLU(),
            )
            fusion_dim = hidden_dim + hidden_dim // 2
        else:
            fusion_dim = hidden_dim

        # Temporal processing
        self.temporal = SensorEncoder(
            input_dim=fusion_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        )

        self.include_magnetometer = include_magnetometer

    def forward(
        self,
        accel: torch.Tensor,
        gyro: torch.Tensor,
        mag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            accel: Accelerometer data (batch, seq, 3)
            gyro: Gyroscope data (batch, seq, 3)
            mag: Magnetometer data (batch, seq, 3), optional

        Returns:
            features: (batch, output_dim)
        """
        batch_size, seq_len, _ = accel.shape

        # Encode each modality
        accel_feat = self.accel_encoder(accel.reshape(-1, 3)).reshape(batch_size, seq_len, -1)
        gyro_feat = self.gyro_encoder(gyro.reshape(-1, 3)).reshape(batch_size, seq_len, -1)

        if self.include_magnetometer and mag is not None:
            mag_feat = self.mag_encoder(mag.reshape(-1, 3)).reshape(batch_size, seq_len, -1)
            combined = torch.cat([accel_feat, gyro_feat, mag_feat], dim=-1)
        else:
            combined = torch.cat([accel_feat, gyro_feat], dim=-1)

        return self.temporal(combined)


class MultiSensorEncoder(nn.Module):
    """
    Encoder for multiple heterogeneous sensor streams.

    Handles sensors with different sampling rates and dimensionalities.
    """

    def __init__(
        self,
        sensor_configs: dict,  # {name: {'dim': int, 'rate': float}}
        output_dim: int = 512,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.sensor_configs = sensor_configs

        # Per-sensor encoders
        self.encoders = nn.ModuleDict()
        total_dim = 0

        for name, config in sensor_configs.items():
            self.encoders[name] = nn.Sequential(
                nn.Linear(config['dim'], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            total_dim += hidden_dim

        # Fusion temporal encoder
        self.fusion = SensorEncoder(
            input_dim=total_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim * 2,
        )

    def forward(
        self,
        sensor_data: dict,  # {name: tensor}
    ) -> torch.Tensor:
        """
        Args:
            sensor_data: Dict mapping sensor names to data tensors
                        Each tensor: (batch, seq_len, sensor_dim)

        Returns:
            features: (batch, output_dim)
        """
        encoded = []

        for name, data in sensor_data.items():
            if name in self.encoders:
                batch_size, seq_len, _ = data.shape
                enc = self.encoders[name](data.reshape(-1, data.shape[-1]))
                enc = enc.reshape(batch_size, seq_len, -1)
                encoded.append(enc)

        # Concatenate and fuse
        combined = torch.cat(encoded, dim=-1)
        return self.fusion(combined)


# Try to import ncps for enhanced liquid networks
try:
    from ncps.torch import CfC, LTC
    from ncps.wirings import AutoNCP

    class NCPSensorEncoder(nn.Module):
        """
        Sensor encoder using official ncps library.

        Uses Neural Circuit Policies for biologically-inspired
        temporal processing.
        """

        def __init__(
            self,
            input_dim: int,
            output_dim: int = 512,
            hidden_dim: int = 64,
            mode: str = "cfc",
        ):
            super().__init__()

            # Auto-wired NCP
            wiring = AutoNCP(hidden_dim, output_dim)

            if mode == "cfc":
                self.rnn = CfC(input_dim, wiring, batch_first=True)
            else:
                self.rnn = LTC(input_dim, wiring, batch_first=True)

            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            output, _ = self.rnn(x)
            return output[:, -1, :]

    NCPS_AVAILABLE = True
except ImportError:
    NCPS_AVAILABLE = False


# Factory function
def create_sensor_encoder(
    encoder_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Create sensor encoder by type.

    Args:
        encoder_type: 'standard', 'imu', 'multi', or 'ncp'
    """
    if encoder_type == "standard":
        return SensorEncoder(**kwargs)
    elif encoder_type == "imu":
        return IMUEncoder(**kwargs)
    elif encoder_type == "multi":
        return MultiSensorEncoder(**kwargs)
    elif encoder_type == "ncp":
        if not NCPS_AVAILABLE:
            raise ImportError("ncps library not installed. Use: pip install ncps")
        return NCPSensorEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
