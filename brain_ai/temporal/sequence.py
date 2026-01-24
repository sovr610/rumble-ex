"""
Sequence Prediction Fallback

LSTM-based sequence predictor as fallback when HTM is not needed
or htm.core is unavailable. Provides similar interface for easy swapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class SequenceConfig:
    """Configuration for sequence predictor."""
    input_size: int = 512
    hidden_size: int = 256
    num_layers: int = 2
    output_size: Optional[int] = None  # Defaults to input_size
    dropout: float = 0.1
    bidirectional: bool = False


class LSTMSequencePredictor(nn.Module):
    """
    LSTM-based sequence predictor.

    Provides next-step prediction and anomaly detection
    similar to HTM but using standard LSTM architecture.
    """

    def __init__(self, config: Optional[SequenceConfig] = None):
        super().__init__()

        self.config = config or SequenceConfig()
        cfg = self.config

        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers
        output_size = cfg.output_size or cfg.input_size

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
            bidirectional=cfg.bidirectional,
            batch_first=True,
        )

        lstm_output_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(lstm_output_size, cfg.hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, output_size),
        )

        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(lstm_output_size, cfg.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Hidden state
        self.hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Prediction history for anomaly baseline
        self.prediction_errors: List[float] = []
        self.error_window = 100

    def reset(self):
        """Reset hidden state for new sequence."""
        self.hidden = None
        self.prediction_errors = []

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        num_directions = 2 if self.config.bidirectional else 1
        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        c0 = torch.zeros_like(h0)
        return h0, c0

    def compute_anomaly(
        self,
        prediction: torch.Tensor,
        actual: torch.Tensor,
        lstm_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute anomaly score based on prediction error.

        Uses both reconstruction error and learned anomaly detector.
        """
        # Reconstruction error
        mse = F.mse_loss(prediction, actual, reduction='none').mean(dim=-1)

        # Learned anomaly score
        learned_anomaly = self.anomaly_head(lstm_output).squeeze(-1)

        # Combine
        anomaly = 0.5 * torch.sigmoid(mse) + 0.5 * learned_anomaly

        return anomaly

    def forward(
        self,
        x: torch.Tensor,
        learn: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Process input sequence.

        Args:
            x: Input (batch, seq_len, input_size) or (batch, input_size)
            learn: Whether this is training mode

        Returns:
            Dict with features, prediction, anomaly, etc.
        """
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]
        device = x.device

        # Initialize hidden state if needed
        if self.hidden is None:
            self.hidden = self.init_hidden(batch_size, device)

        # Ensure hidden state matches batch size
        if self.hidden[0].shape[1] != batch_size:
            self.hidden = self.init_hidden(batch_size, device)

        # LSTM forward
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        # Detach hidden state to prevent backprop through time
        if not learn:
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        # Get last timestep output
        last_output = lstm_out[:, -1, :]  # (batch, hidden)

        # Predict next input
        prediction = self.predictor(last_output)

        # Compute anomaly (comparing prediction to actual last input)
        actual = x[:, -1, :]
        anomaly = self.compute_anomaly(prediction, actual, last_output)

        # Compute anomaly likelihood
        anomaly_np = anomaly.mean().item()
        self.prediction_errors.append(anomaly_np)
        if len(self.prediction_errors) > self.error_window:
            self.prediction_errors.pop(0)

        if len(self.prediction_errors) > 10:
            import numpy as np
            mean_error = np.mean(self.prediction_errors)
            std_error = np.std(self.prediction_errors) + 1e-6
            likelihood = 1 - np.exp(-(anomaly_np - mean_error) / std_error)
            anomaly_likelihood = torch.tensor(
                np.clip(likelihood, 0, 1),
                device=device
            ).expand(batch_size)
        else:
            anomaly_likelihood = anomaly

        return {
            'features': last_output,
            'prediction': prediction,
            'anomaly': anomaly,
            'anomaly_likelihood': anomaly_likelihood,
            'active_cells': last_output,  # Compatibility with HTM interface
            'predictive_cells': prediction,  # Compatibility
        }


class GRUSequencePredictor(nn.Module):
    """
    GRU-based sequence predictor.

    Lighter weight alternative to LSTM with similar capabilities.
    """

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.predictor = nn.Linear(hidden_size, input_size)
        self.hidden = None

    def reset(self):
        self.hidden = None

    def forward(self, x: torch.Tensor, learn: bool = True) -> Dict[str, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]

        if self.hidden is None or self.hidden.shape[1] != batch_size:
            self.hidden = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device
            )

        output, self.hidden = self.gru(x, self.hidden)

        if not learn:
            self.hidden = self.hidden.detach()

        last_output = output[:, -1, :]
        prediction = self.predictor(last_output)

        # Simple anomaly: prediction error
        actual = x[:, -1, :]
        anomaly = F.mse_loss(prediction, actual, reduction='none').mean(dim=-1)
        anomaly = torch.sigmoid(anomaly)

        return {
            'features': last_output,
            'prediction': prediction,
            'anomaly': anomaly,
            'anomaly_likelihood': anomaly,
            'active_cells': last_output,
            'predictive_cells': prediction,
        }


class TransformerSequencePredictor(nn.Module):
    """
    Transformer-based sequence predictor.

    Uses self-attention for capturing long-range dependencies.
    """

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_size) * 0.1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Prediction head
        self.predictor = nn.Linear(hidden_size, input_size)

        # Sequence buffer
        self.sequence_buffer: Optional[torch.Tensor] = None

    def reset(self):
        self.sequence_buffer = None

    def forward(self, x: torch.Tensor, learn: bool = True) -> Dict[str, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape
        device = x.device

        # Add to buffer
        if self.sequence_buffer is None:
            self.sequence_buffer = x
        else:
            self.sequence_buffer = torch.cat([self.sequence_buffer, x], dim=1)
            # Truncate to max length
            if self.sequence_buffer.shape[1] > self.max_seq_len:
                self.sequence_buffer = self.sequence_buffer[:, -self.max_seq_len:, :]

        # Project and add positional encoding
        projected = self.input_proj(self.sequence_buffer)
        seq_len = projected.shape[1]
        projected = projected + self.pos_encoding[:, :seq_len, :]

        # Transformer forward
        output = self.transformer(projected)

        # Get last timestep
        last_output = output[:, -1, :]
        prediction = self.predictor(last_output)

        # Anomaly
        actual = x[:, -1, :]
        anomaly = F.mse_loss(prediction, actual, reduction='none').mean(dim=-1)
        anomaly = torch.sigmoid(anomaly)

        return {
            'features': last_output,
            'prediction': prediction,
            'anomaly': anomaly,
            'anomaly_likelihood': anomaly,
            'active_cells': last_output,
            'predictive_cells': prediction,
        }


class TemporalLayer(nn.Module):
    """
    Unified temporal layer that can use HTM or fallback predictors.

    Provides consistent interface regardless of backend.
    """

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        backend: str = "auto",  # "htm", "lstm", "gru", "transformer", "auto"
        **kwargs
    ):
        super().__init__()

        self.backend = backend

        if backend == "auto":
            # Try HTM first, fall back to LSTM
            try:
                from .htm import HTMLayer, HTMConfig, HTM_CORE_AVAILABLE
                if HTM_CORE_AVAILABLE:
                    config = HTMConfig(input_size=input_size, **kwargs)
                    self.layer = HTMLayer(config, use_htm_core=True)
                    self.backend = "htm_core"
                else:
                    config = HTMConfig(input_size=input_size, **kwargs)
                    self.layer = HTMLayer(config, use_htm_core=False)
                    self.backend = "htm_pytorch"
            except Exception:
                config = SequenceConfig(input_size=input_size, hidden_size=hidden_size)
                self.layer = LSTMSequencePredictor(config)
                self.backend = "lstm"

        elif backend == "htm":
            from .htm import HTMLayer, HTMConfig
            config = HTMConfig(input_size=input_size, **kwargs)
            self.layer = HTMLayer(config)

        elif backend == "lstm":
            config = SequenceConfig(input_size=input_size, hidden_size=hidden_size)
            self.layer = LSTMSequencePredictor(config)

        elif backend == "gru":
            self.layer = GRUSequencePredictor(input_size=input_size, hidden_size=hidden_size)

        elif backend == "transformer":
            self.layer = TransformerSequencePredictor(
                input_size=input_size, hidden_size=hidden_size, **kwargs
            )

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def reset(self):
        self.layer.reset()

    def forward(self, x: torch.Tensor, learn: bool = True) -> Dict[str, torch.Tensor]:
        return self.layer(x, learn=learn)


# Factory function
def create_temporal_layer(
    backend: str = "auto",
    input_size: int = 512,
    **kwargs
) -> nn.Module:
    """
    Create temporal layer with specified backend.

    Args:
        backend: 'auto', 'htm', 'lstm', 'gru', 'transformer'
        input_size: Input feature dimension
        **kwargs: Backend-specific arguments
    """
    return TemporalLayer(input_size=input_size, backend=backend, **kwargs)
