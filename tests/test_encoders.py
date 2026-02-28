"""
Tests for Modality Encoders

Validates all four modality encoders:
    - VisionEncoder: spiking convolutional encoder for images
    - TextEncoder: transformer-based text encoder
    - AudioEncoder: spiking audio encoder with mel spectrogram
    - SensorEncoder: liquid time-constant encoder for sensor data

Key properties tested:
    - Creation via class and factory function
    - Forward pass output shape
    - Output dimension consistency (all produce same output_dim)
    - Output finiteness (no NaN/Inf)

Run:
    pytest tests/test_encoders.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys

# Path fixup so tests can run from repo root
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from brain_ai.encoders.vision import VisionEncoder, create_vision_encoder
from brain_ai.encoders.text import TextEncoder, create_text_encoder
from brain_ai.encoders.audio import AudioEncoder, create_audio_encoder
from brain_ai.encoders.sensors import SensorEncoder, create_sensor_encoder


# ---------------------------------------------------------------------------
# Constants -- keep small for fast execution
# ---------------------------------------------------------------------------
OUTPUT_DIM = 64
BATCH_SIZE = 2
SEED = 42


def _seed(s: int = SEED):
    torch.manual_seed(s)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vision_encoder():
    """Create a small VisionEncoder."""
    _seed()
    enc = VisionEncoder(
        input_channels=3,
        output_dim=OUTPUT_DIM,
        channels=[8, 16],
        beta=0.9,
        num_steps=2,  # Very few steps for speed
        input_size=(32, 32),
        dropout=0.0,
    )
    enc.train(False)
    return enc


@pytest.fixture
def text_encoder():
    """Create a small TextEncoder."""
    _seed()
    enc = TextEncoder(
        vocab_size=1000,
        embed_dim=32,
        output_dim=OUTPUT_DIM,
        num_layers=1,
        num_heads=2,
        ff_dim=64,
        max_seq_len=32,
        dropout=0.0,
        pooling="mean",
    )
    enc.train(False)
    return enc


@pytest.fixture
def audio_encoder():
    """Create a small AudioEncoder using precomputed mel spectrograms."""
    _seed()
    enc = AudioEncoder(
        output_dim=OUTPUT_DIM,
        n_mels=16,
        channels=[16, 32],
        beta=0.9,
        num_steps=2,
        dropout=0.0,
    )
    enc.train(False)
    return enc


@pytest.fixture
def sensor_encoder():
    """Create a small SensorEncoder."""
    _seed()
    enc = SensorEncoder(
        input_dim=8,
        output_dim=OUTPUT_DIM,
        hidden_dim=32,
        num_layers=1,
        cell_type="cfc",
        use_spike_output=False,
        dropout=0.0,
    )
    enc.train(False)
    return enc


@pytest.fixture
def vision_input():
    """Small image tensor."""
    _seed()
    return torch.randn(BATCH_SIZE, 3, 32, 32)


@pytest.fixture
def text_input():
    """Small token ID tensor."""
    _seed()
    return torch.randint(1, 1000, (BATCH_SIZE, 16))


@pytest.fixture
def audio_mel_input():
    """Precomputed mel spectrogram tensor."""
    _seed()
    # (batch, n_mels, time_frames)
    return torch.randn(BATCH_SIZE, 16, 20)


@pytest.fixture
def sensor_input():
    """Sensor time-series tensor."""
    _seed()
    # (batch, seq_len, input_dim)
    return torch.randn(BATCH_SIZE, 10, 8)


# ===========================================================================
# Tests: VisionEncoder
# ===========================================================================

class TestVisionEncoder:
    """Tests for the VisionEncoder."""

    def test_creation(self):
        """VisionEncoder should create without error."""
        enc = VisionEncoder(
            input_channels=3,
            output_dim=OUTPUT_DIM,
            channels=[8, 16],
            num_steps=2,
        )
        assert enc.output_dim == OUTPUT_DIM
        assert enc.input_channels == 3

    def test_factory_creation(self):
        """create_vision_encoder should produce a VisionEncoder."""
        enc = create_vision_encoder(
            encoder_type="standard",
            input_channels=3,
            output_dim=OUTPUT_DIM,
            channels=[8, 16],
            num_steps=2,
        )
        assert isinstance(enc, VisionEncoder)

    def test_forward_shape(self, vision_encoder, vision_input):
        """Forward output should have shape (batch, output_dim)."""
        with torch.no_grad():
            output = vision_encoder(vision_input)

        assert output.shape == (BATCH_SIZE, OUTPUT_DIM), (
            f"Expected ({BATCH_SIZE}, {OUTPUT_DIM}), got {output.shape}"
        )

    def test_forward_finite(self, vision_encoder, vision_input):
        """Forward output should be finite."""
        with torch.no_grad():
            output = vision_encoder(vision_input)

        assert not torch.isnan(output).any(), "NaN in VisionEncoder output"
        assert not torch.isinf(output).any(), "Inf in VisionEncoder output"

    def test_reset_mem(self, vision_encoder, vision_input):
        """reset_mem should not raise."""
        vision_encoder.reset_mem()
        with torch.no_grad():
            output = vision_encoder(vision_input)
        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_temporal_input(self, vision_encoder):
        """VisionEncoder should handle temporal input (time, batch, C, H, W)."""
        _seed()
        T = 3
        x = torch.randn(T, BATCH_SIZE, 3, 32, 32)
        with torch.no_grad():
            output = vision_encoder(x, temporal_input=True)

        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)

    @pytest.mark.parametrize("channels_cfg", [1, 3])
    def test_varying_input_channels(self, channels_cfg):
        """VisionEncoder should work with different input channel counts."""
        _seed()
        enc = VisionEncoder(
            input_channels=channels_cfg,
            output_dim=OUTPUT_DIM,
            channels=[8, 16],
            num_steps=2,
        )
        enc.train(False)
        x = torch.randn(BATCH_SIZE, channels_cfg, 32, 32)
        with torch.no_grad():
            output = enc(x)

        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)


# ===========================================================================
# Tests: TextEncoder
# ===========================================================================

class TestTextEncoder:
    """Tests for the TextEncoder."""

    def test_creation(self):
        """TextEncoder should create without error."""
        enc = TextEncoder(
            vocab_size=1000,
            embed_dim=32,
            output_dim=OUTPUT_DIM,
            num_layers=1,
            num_heads=2,
        )
        assert enc.output_dim == OUTPUT_DIM

    def test_factory_creation(self):
        """create_text_encoder should produce a TextEncoder."""
        enc = create_text_encoder(
            encoder_type="standard",
            vocab_size=1000,
            embed_dim=32,
            output_dim=OUTPUT_DIM,
            num_layers=1,
            num_heads=2,
        )
        assert isinstance(enc, TextEncoder)

    def test_forward_shape(self, text_encoder, text_input):
        """Forward output should have shape (batch, output_dim)."""
        with torch.no_grad():
            output = text_encoder(text_input)

        assert output.shape == (BATCH_SIZE, OUTPUT_DIM), (
            f"Expected ({BATCH_SIZE}, {OUTPUT_DIM}), got {output.shape}"
        )

    def test_forward_finite(self, text_encoder, text_input):
        """Forward output should be finite."""
        with torch.no_grad():
            output = text_encoder(text_input)

        assert not torch.isnan(output).any(), "NaN in TextEncoder output"
        assert not torch.isinf(output).any(), "Inf in TextEncoder output"

    def test_with_attention_mask(self, text_encoder, text_input):
        """TextEncoder should accept an attention mask."""
        mask = torch.ones(BATCH_SIZE, 16)
        mask[:, 8:] = 0  # Mask out second half
        with torch.no_grad():
            output = text_encoder(text_input, attention_mask=mask)

        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_deterministic(self, text_encoder, text_input):
        """Same input should produce same output in inference mode."""
        with torch.no_grad():
            out1 = text_encoder(text_input)
            out2 = text_encoder(text_input)

        assert torch.allclose(out1, out2, atol=1e-6), (
            f"Non-deterministic output. Max diff: "
            f"{(out1 - out2).abs().max().item()}"
        )

    @pytest.mark.parametrize("pooling", ["mean", "max"])
    def test_pooling_strategies(self, text_input, pooling):
        """TextEncoder should support different pooling strategies."""
        _seed()
        enc = TextEncoder(
            vocab_size=1000,
            embed_dim=32,
            output_dim=OUTPUT_DIM,
            num_layers=1,
            num_heads=2,
            pooling=pooling,
        )
        enc.train(False)
        with torch.no_grad():
            output = enc(text_input)

        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)

    @pytest.mark.parametrize("seq_len", [4, 16, 32])
    def test_varying_sequence_lengths(self, seq_len):
        """TextEncoder should handle different sequence lengths."""
        _seed()
        enc = TextEncoder(
            vocab_size=1000,
            embed_dim=32,
            output_dim=OUTPUT_DIM,
            num_layers=1,
            num_heads=2,
            max_seq_len=64,
        )
        enc.train(False)
        tokens = torch.randint(1, 1000, (BATCH_SIZE, seq_len))
        with torch.no_grad():
            output = enc(tokens)

        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)


# ===========================================================================
# Tests: AudioEncoder
# ===========================================================================

class TestAudioEncoder:
    """Tests for the AudioEncoder."""

    def test_creation(self):
        """AudioEncoder should create without error."""
        enc = AudioEncoder(
            output_dim=OUTPUT_DIM,
            n_mels=16,
            channels=[16, 32],
            num_steps=2,
        )
        assert enc.output_dim == OUTPUT_DIM

    def test_factory_creation(self):
        """create_audio_encoder should produce an AudioEncoder."""
        enc = create_audio_encoder(
            encoder_type="standard",
            output_dim=OUTPUT_DIM,
            n_mels=16,
            channels=[16, 32],
            num_steps=2,
        )
        assert isinstance(enc, AudioEncoder)

    def test_forward_shape_precomputed_mel(self, audio_encoder, audio_mel_input):
        """Forward with precomputed mel should output (batch, output_dim)."""
        with torch.no_grad():
            output = audio_encoder(audio_mel_input, precomputed_mel=True)

        assert output.shape == (BATCH_SIZE, OUTPUT_DIM), (
            f"Expected ({BATCH_SIZE}, {OUTPUT_DIM}), got {output.shape}"
        )

    def test_forward_finite(self, audio_encoder, audio_mel_input):
        """Forward output should be finite."""
        with torch.no_grad():
            output = audio_encoder(audio_mel_input, precomputed_mel=True)

        assert not torch.isnan(output).any(), "NaN in AudioEncoder output"
        assert not torch.isinf(output).any(), "Inf in AudioEncoder output"

    def test_reset_mem(self, audio_encoder, audio_mel_input):
        """reset_mem should not raise."""
        audio_encoder.reset_mem()
        with torch.no_grad():
            output = audio_encoder(audio_mel_input, precomputed_mel=True)
        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)

    @pytest.mark.parametrize("n_mels", [16, 32])
    def test_varying_mel_bins(self, n_mels):
        """AudioEncoder should handle different numbers of mel bins."""
        _seed()
        enc = AudioEncoder(
            output_dim=OUTPUT_DIM,
            n_mels=n_mels,
            channels=[16, 32],
            num_steps=2,
            dropout=0.0,
        )
        enc.train(False)
        mel = torch.randn(BATCH_SIZE, n_mels, 20)
        with torch.no_grad():
            output = enc(mel, precomputed_mel=True)

        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)

    @pytest.mark.parametrize("time_frames", [10, 20, 40])
    def test_varying_time_frames(self, time_frames):
        """AudioEncoder should handle different temporal lengths."""
        _seed()
        enc = AudioEncoder(
            output_dim=OUTPUT_DIM,
            n_mels=16,
            channels=[16, 32],
            num_steps=2,
            dropout=0.0,
        )
        enc.train(False)
        mel = torch.randn(BATCH_SIZE, 16, time_frames)
        with torch.no_grad():
            output = enc(mel, precomputed_mel=True)

        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)


# ===========================================================================
# Tests: SensorEncoder
# ===========================================================================

class TestSensorEncoder:
    """Tests for the SensorEncoder."""

    def test_creation(self):
        """SensorEncoder should create without error."""
        enc = SensorEncoder(
            input_dim=8,
            output_dim=OUTPUT_DIM,
            hidden_dim=32,
            num_layers=1,
        )
        assert enc.output_dim == OUTPUT_DIM
        assert enc.input_dim == 8

    def test_factory_creation(self):
        """create_sensor_encoder should produce a SensorEncoder."""
        enc = create_sensor_encoder(
            encoder_type="standard",
            input_dim=8,
            output_dim=OUTPUT_DIM,
            hidden_dim=32,
            num_layers=1,
        )
        assert isinstance(enc, SensorEncoder)

    def test_forward_shape_sequence(self, sensor_encoder, sensor_input):
        """Forward with sequence input should output (batch, output_dim)."""
        with torch.no_grad():
            output = sensor_encoder(sensor_input)

        assert output.shape == (BATCH_SIZE, OUTPUT_DIM), (
            f"Expected ({BATCH_SIZE}, {OUTPUT_DIM}), got {output.shape}"
        )

    def test_forward_shape_single_step(self, sensor_encoder):
        """Forward with single timestep input (batch, input_dim)."""
        _seed()
        sensor_encoder.reset_state()
        x = torch.randn(BATCH_SIZE, 8)
        with torch.no_grad():
            output = sensor_encoder(x)

        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_forward_finite(self, sensor_encoder, sensor_input):
        """Forward output should be finite."""
        with torch.no_grad():
            output = sensor_encoder(sensor_input)

        assert not torch.isnan(output).any(), "NaN in SensorEncoder output"
        assert not torch.isinf(output).any(), "Inf in SensorEncoder output"

    def test_reset_state(self, sensor_encoder, sensor_input):
        """reset_state should clear hidden states."""
        with torch.no_grad():
            sensor_encoder(sensor_input)

        assert sensor_encoder.hidden_states is not None
        sensor_encoder.reset_state()
        assert sensor_encoder.hidden_states is None

    def test_return_sequence(self, sensor_encoder, sensor_input):
        """return_sequence=True should output (batch, seq_len, output_dim)."""
        sensor_encoder.reset_state()
        with torch.no_grad():
            output = sensor_encoder(sensor_input, return_sequence=True)

        seq_len = sensor_input.shape[1]
        assert output.shape == (BATCH_SIZE, seq_len, OUTPUT_DIM), (
            f"Expected ({BATCH_SIZE}, {seq_len}, {OUTPUT_DIM}), got {output.shape}"
        )

    @pytest.mark.parametrize("cell_type", ["cfc", "ltc"])
    def test_cell_types(self, cell_type):
        """SensorEncoder should work with both CfC and LTC cells."""
        _seed()
        enc = SensorEncoder(
            input_dim=8,
            output_dim=OUTPUT_DIM,
            hidden_dim=32,
            num_layers=1,
            cell_type=cell_type,
            dropout=0.0,
        )
        enc.train(False)
        x = torch.randn(BATCH_SIZE, 10, 8)
        with torch.no_grad():
            output = enc(x)

        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)

    @pytest.mark.parametrize("input_dim", [4, 8, 16])
    def test_varying_input_dims(self, input_dim):
        """SensorEncoder should handle different input dimensions."""
        _seed()
        enc = SensorEncoder(
            input_dim=input_dim,
            output_dim=OUTPUT_DIM,
            hidden_dim=32,
            num_layers=1,
            dropout=0.0,
        )
        enc.train(False)
        x = torch.randn(BATCH_SIZE, 10, input_dim)
        with torch.no_grad():
            output = enc(x)

        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)


# ===========================================================================
# Tests: Output dimension consistency
# ===========================================================================

class TestOutputDimensionConsistency:
    """Verify all encoders produce the same output dimension."""

    def test_all_encoders_same_output_dim(
        self, vision_encoder, text_encoder, audio_encoder, sensor_encoder,
        vision_input, text_input, audio_mel_input, sensor_input,
    ):
        """All four encoders should produce tensors with output_dim=64."""
        with torch.no_grad():
            vision_out = vision_encoder(vision_input)
            text_out = text_encoder(text_input)
            audio_out = audio_encoder(audio_mel_input, precomputed_mel=True)
            sensor_encoder.reset_state()
            sensor_out = sensor_encoder(sensor_input)

        # All should have the same last dimension
        assert vision_out.shape[-1] == OUTPUT_DIM
        assert text_out.shape[-1] == OUTPUT_DIM
        assert audio_out.shape[-1] == OUTPUT_DIM
        assert sensor_out.shape[-1] == OUTPUT_DIM

    def test_all_encoders_same_batch_dim(
        self, vision_encoder, text_encoder, audio_encoder, sensor_encoder,
        vision_input, text_input, audio_mel_input, sensor_input,
    ):
        """All encoders should preserve the batch dimension."""
        with torch.no_grad():
            vision_out = vision_encoder(vision_input)
            text_out = text_encoder(text_input)
            audio_out = audio_encoder(audio_mel_input, precomputed_mel=True)
            sensor_encoder.reset_state()
            sensor_out = sensor_encoder(sensor_input)

        assert vision_out.shape[0] == BATCH_SIZE
        assert text_out.shape[0] == BATCH_SIZE
        assert audio_out.shape[0] == BATCH_SIZE
        assert sensor_out.shape[0] == BATCH_SIZE

    def test_all_encoders_2d_output(
        self, vision_encoder, text_encoder, audio_encoder, sensor_encoder,
        vision_input, text_input, audio_mel_input, sensor_input,
    ):
        """All encoders should produce 2D output (batch, output_dim)."""
        with torch.no_grad():
            vision_out = vision_encoder(vision_input)
            text_out = text_encoder(text_input)
            audio_out = audio_encoder(audio_mel_input, precomputed_mel=True)
            sensor_encoder.reset_state()
            sensor_out = sensor_encoder(sensor_input)

        assert vision_out.dim() == 2, f"Vision output dim: {vision_out.dim()}"
        assert text_out.dim() == 2, f"Text output dim: {text_out.dim()}"
        assert audio_out.dim() == 2, f"Audio output dim: {audio_out.dim()}"
        assert sensor_out.dim() == 2, f"Sensor output dim: {sensor_out.dim()}"

    @pytest.mark.parametrize("output_dim", [32, 64, 128])
    def test_configurable_output_dim(self, output_dim):
        """All encoders should respect the configured output_dim."""
        _seed()

        vision = VisionEncoder(
            input_channels=3, output_dim=output_dim,
            channels=[8, 16], num_steps=2,
        )
        text = TextEncoder(
            vocab_size=1000, embed_dim=32, output_dim=output_dim,
            num_layers=1, num_heads=2,
        )
        audio = AudioEncoder(
            output_dim=output_dim, n_mels=16,
            channels=[16, 32], num_steps=2,
        )
        sensor = SensorEncoder(
            input_dim=8, output_dim=output_dim,
            hidden_dim=32, num_layers=1,
        )

        vision.train(False)
        text.train(False)
        audio.train(False)
        sensor.train(False)

        with torch.no_grad():
            v_out = vision(torch.randn(1, 3, 32, 32))
            t_out = text(torch.randint(1, 1000, (1, 16)))
            a_out = audio(torch.randn(1, 16, 20), precomputed_mel=True)
            sensor.reset_state()
            s_out = sensor(torch.randn(1, 10, 8))

        assert v_out.shape[-1] == output_dim, f"Vision: {v_out.shape[-1]}"
        assert t_out.shape[-1] == output_dim, f"Text: {t_out.shape[-1]}"
        assert a_out.shape[-1] == output_dim, f"Audio: {a_out.shape[-1]}"
        assert s_out.shape[-1] == output_dim, f"Sensor: {s_out.shape[-1]}"


# ===========================================================================
# Tests: Factory function error handling
# ===========================================================================

class TestFactoryErrors:
    """Tests for factory function error handling."""

    def test_vision_unknown_type(self):
        """create_vision_encoder should raise on unknown type."""
        with pytest.raises(ValueError, match="Unknown encoder type"):
            create_vision_encoder(encoder_type="unknown")

    def test_text_unknown_type(self):
        """create_text_encoder should raise on unknown type."""
        with pytest.raises(ValueError, match="Unknown encoder type"):
            create_text_encoder(encoder_type="unknown")

    def test_audio_unknown_type(self):
        """create_audio_encoder should raise on unknown type."""
        with pytest.raises(ValueError, match="Unknown encoder type"):
            create_audio_encoder(encoder_type="unknown")

    def test_sensor_unknown_type(self):
        """create_sensor_encoder should raise on unknown type."""
        with pytest.raises(ValueError, match="Unknown encoder type"):
            create_sensor_encoder(encoder_type="unknown")


# ===========================================================================
# Tests: Batch size parameterized
# ===========================================================================

class TestBatchSizes:
    """Verify encoders handle various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_vision_batch_sizes(self, batch_size):
        """VisionEncoder should handle various batch sizes."""
        _seed()
        enc = VisionEncoder(
            input_channels=3, output_dim=OUTPUT_DIM,
            channels=[8, 16], num_steps=2,
        )
        enc.train(False)
        x = torch.randn(batch_size, 3, 32, 32)
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (batch_size, OUTPUT_DIM)

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_text_batch_sizes(self, batch_size):
        """TextEncoder should handle various batch sizes."""
        _seed()
        enc = TextEncoder(
            vocab_size=1000, embed_dim=32, output_dim=OUTPUT_DIM,
            num_layers=1, num_heads=2,
        )
        enc.train(False)
        x = torch.randint(1, 1000, (batch_size, 16))
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (batch_size, OUTPUT_DIM)

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_audio_batch_sizes(self, batch_size):
        """AudioEncoder should handle various batch sizes."""
        _seed()
        enc = AudioEncoder(
            output_dim=OUTPUT_DIM, n_mels=16,
            channels=[16, 32], num_steps=2, dropout=0.0,
        )
        enc.train(False)
        x = torch.randn(batch_size, 16, 20)
        with torch.no_grad():
            out = enc(x, precomputed_mel=True)
        assert out.shape == (batch_size, OUTPUT_DIM)

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_sensor_batch_sizes(self, batch_size):
        """SensorEncoder should handle various batch sizes."""
        _seed()
        enc = SensorEncoder(
            input_dim=8, output_dim=OUTPUT_DIM,
            hidden_dim=32, num_layers=1, dropout=0.0,
        )
        enc.train(False)
        x = torch.randn(batch_size, 10, 8)
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (batch_size, OUTPUT_DIM)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
