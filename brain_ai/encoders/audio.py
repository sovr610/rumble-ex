"""
Audio Encoder

Spiking audio encoder using mel spectrograms and 1D convolutions.
Supports speech, music, and environmental sounds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

from ..core.neurons import LIFNeuron


class MelSpectrogramFrontend(nn.Module):
    """
    Mel spectrogram feature extraction frontend.

    Converts raw audio waveforms to mel-frequency spectrograms.
    Uses torchaudio if available, otherwise falls back to manual implementation.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2

        # Try to use torchaudio
        try:
            import torchaudio
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=self.f_max,
            )
            self.use_torchaudio = True
        except ImportError:
            self.use_torchaudio = False
            # Create mel filterbank manually
            self.register_buffer(
                'mel_fb',
                self._create_mel_filterbank(n_fft // 2 + 1, n_mels, sample_rate, f_min, self.f_max)
            )

    def _create_mel_filterbank(
        self,
        n_freqs: int,
        n_mels: int,
        sample_rate: int,
        f_min: float,
        f_max: float,
    ) -> torch.Tensor:
        """Create mel filterbank matrix."""
        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * math.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)

        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = torch.tensor([mel_to_hz(m) for m in mel_points])
        bin_points = (hz_points / sample_rate * (n_freqs - 1) * 2).long()

        filterbank = torch.zeros(n_mels, n_freqs)
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            for j in range(left, center):
                filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to mel spectrogram.

        Args:
            waveform: (batch, samples) or (batch, 1, samples)

        Returns:
            mel_spec: (batch, n_mels, time_frames)
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        if self.use_torchaudio:
            mel_spec = self.mel_transform(waveform)
        else:
            # Manual STFT + mel filterbank
            # Add small epsilon to avoid log(0)
            stft = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True,
            )
            power_spec = stft.abs() ** 2
            mel_spec = torch.matmul(self.mel_fb, power_spec.transpose(-1, -2)).transpose(-1, -2)

        # Log mel spectrogram
        mel_spec = torch.log(mel_spec + 1e-9)

        return mel_spec


class SNNConv1dBlock(nn.Module):
    """1D Convolutional block with spiking neuron."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        beta: float = 0.9,
        surrogate: str = "atan",
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.lif = LIFNeuron(beta=beta, surrogate=surrogate)

    def reset_mem(self):
        self.lif.reset_mem()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        spk, _ = self.lif(x)
        return spk


class AudioEncoder(nn.Module):
    """
    Spiking Audio Encoder.

    Processes audio through mel spectrogram extraction followed
    by spiking 1D convolutions.

    Args:
        output_dim: Output feature dimension (default 512)
        sample_rate: Audio sample rate
        n_mels: Number of mel frequency bins
        channels: List of conv channel sizes
        beta: Membrane decay for LIF neurons
        num_steps: Simulation timesteps
    """

    def __init__(
        self,
        output_dim: int = 512,
        sample_rate: int = 16000,
        n_mels: int = 80,
        channels: List[int] = [128, 256],
        beta: float = 0.9,
        num_steps: int = 25,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.num_steps = num_steps

        # Mel spectrogram frontend
        self.mel_frontend = MelSpectrogramFrontend(
            sample_rate=sample_rate,
            n_mels=n_mels,
        )

        # Spiking conv layers
        self.conv_blocks = nn.ModuleList()
        in_ch = n_mels

        for out_ch in channels:
            self.conv_blocks.append(
                SNNConv1dBlock(in_ch, out_ch, beta=beta)
            )
            in_ch = out_ch

        # Temporal pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Projection
        self.projection = nn.Sequential(
            nn.Linear(channels[-1], output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def reset_mem(self):
        for block in self.conv_blocks:
            block.reset_mem()

    def forward(
        self,
        x: torch.Tensor,
        precomputed_mel: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Audio waveform (batch, samples) or
               mel spectrogram if precomputed_mel=True (batch, n_mels, time)
            precomputed_mel: If True, skip mel extraction

        Returns:
            features: (batch, output_dim)
        """
        # Extract mel spectrogram if needed
        if not precomputed_mel:
            x = self.mel_frontend(x)

        # x shape: (batch, n_mels, time_frames)
        self.reset_mem()

        # Process through spiking conv layers
        spike_sum = None

        for _ in range(self.num_steps):
            feat = x
            for block in self.conv_blocks:
                feat = block(feat)

            # Pool over time
            pooled = self.pool(feat).squeeze(-1)  # (batch, channels)

            if spike_sum is None:
                spike_sum = pooled
            else:
                spike_sum += pooled

        # Average and project
        features = spike_sum / self.num_steps
        return self.projection(features)


class StreamingAudioEncoder(nn.Module):
    """
    Streaming audio encoder for real-time processing.

    Processes audio in chunks while maintaining state
    across chunks for continuous processing.
    """

    def __init__(
        self,
        output_dim: int = 512,
        chunk_size: int = 1600,  # 100ms at 16kHz
        n_mels: int = 80,
        hidden_dim: int = 256,
        beta: float = 0.9,
    ):
        super().__init__()

        self.chunk_size = chunk_size
        self.output_dim = output_dim

        self.mel_frontend = MelSpectrogramFrontend(n_mels=n_mels)

        # GRU for streaming state
        self.rnn = nn.GRU(n_mels, hidden_dim, batch_first=True)

        # Spiking output layer
        self.lif = LIFNeuron(beta=beta)
        self.projection = nn.Linear(hidden_dim, output_dim)

        self.hidden_state = None

    def reset_state(self):
        self.hidden_state = None
        self.lif.reset_mem()

    def forward(
        self,
        chunk: torch.Tensor,
        reset: bool = False,
    ) -> torch.Tensor:
        """
        Process audio chunk.

        Args:
            chunk: Audio chunk (batch, samples)
            reset: Reset streaming state

        Returns:
            features: (batch, output_dim)
        """
        if reset:
            self.reset_state()

        # Extract mel features
        mel = self.mel_frontend(chunk)  # (batch, n_mels, time)
        mel = mel.transpose(1, 2)  # (batch, time, n_mels)

        # Process through RNN
        output, self.hidden_state = self.rnn(mel, self.hidden_state)

        # Take last timestep and apply spiking
        last_output = output[:, -1, :]
        projected = self.projection(last_output)
        spk, _ = self.lif(projected)

        return spk


class MultiTaskAudioEncoder(nn.Module):
    """
    Audio encoder with multiple output heads for different tasks.

    Supports speech recognition, speaker identification, and
    emotion recognition simultaneously.
    """

    def __init__(
        self,
        output_dim: int = 512,
        num_classes_speech: int = 1000,  # Phoneme/word classes
        num_speakers: int = 100,
        num_emotions: int = 8,
        **kwargs
    ):
        super().__init__()

        self.backbone = AudioEncoder(output_dim=output_dim, **kwargs)

        # Task-specific heads
        self.speech_head = nn.Linear(output_dim, num_classes_speech)
        self.speaker_head = nn.Linear(output_dim, num_speakers)
        self.emotion_head = nn.Linear(output_dim, num_emotions)

    def forward(
        self,
        x: torch.Tensor,
        task: Optional[str] = None,
    ):
        """
        Forward pass.

        Args:
            x: Audio input
            task: 'speech', 'speaker', 'emotion', or None for features only
        """
        features = self.backbone(x)

        if task == "speech":
            return self.speech_head(features)
        elif task == "speaker":
            return self.speaker_head(features)
        elif task == "emotion":
            return self.emotion_head(features)
        else:
            return features


# Factory function
def create_audio_encoder(
    encoder_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Create audio encoder by type.

    Args:
        encoder_type: 'standard', 'streaming', or 'multitask'
    """
    if encoder_type == "standard":
        return AudioEncoder(**kwargs)
    elif encoder_type == "streaming":
        return StreamingAudioEncoder(**kwargs)
    elif encoder_type == "multitask":
        return MultiTaskAudioEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
