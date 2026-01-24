"""
Tests for SNN Core Module

Validates neurons, encoding, and network architectures.
"""

import pytest
import torch
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from brain_ai.core import (
    LIFNeuron,
    AdaptiveLIFNeuron,
    RecurrentLIFNeuron,
    RateEncoder,
    TemporalEncoder,
    SpikeDecoder,
    SNNCore,
    ConvSNN,
    get_surrogate,
)


class TestLIFNeuron:
    """Tests for Leaky Integrate-and-Fire neuron."""

    def test_init(self):
        neuron = LIFNeuron(beta=0.9, threshold=1.0)
        assert neuron.beta == 0.9
        assert neuron.threshold == 1.0

    def test_forward_shape(self):
        neuron = LIFNeuron()
        x = torch.randn(32, 128)  # batch=32, features=128
        spk, mem = neuron(x)
        assert spk.shape == (32, 128)
        assert mem.shape == (32, 128)

    def test_spike_binary(self):
        neuron = LIFNeuron()
        x = torch.randn(32, 128) * 2  # Strong input
        for _ in range(10):
            spk, mem = neuron(x)
        # Spikes should be binary
        assert torch.all((spk == 0) | (spk == 1))

    def test_membrane_decay(self):
        neuron = LIFNeuron(beta=0.9)
        x = torch.ones(1, 10)

        # First step: membrane = input
        spk1, mem1 = neuron(x)

        # Second step with zero input: membrane should decay
        spk2, mem2 = neuron(torch.zeros(1, 10))

        # Membrane should decay by beta (approximately, accounting for possible spike)
        assert torch.all(mem2 <= mem1 * 0.9 + 0.01)

    def test_reset_mechanism_subtract(self):
        neuron = LIFNeuron(threshold=1.0, reset_mechanism="subtract")
        x = torch.ones(1, 1) * 1.5  # Above threshold

        spk, mem = neuron(x)
        assert spk.item() == 1.0
        # After spike, membrane should be 1.5 - 1.0 = 0.5
        assert abs(mem.item() - 0.5) < 0.01

    def test_reset_mechanism_zero(self):
        neuron = LIFNeuron(threshold=1.0, reset_mechanism="zero")
        x = torch.ones(1, 1) * 1.5

        spk, mem = neuron(x)
        assert spk.item() == 1.0
        # After spike, membrane should be reset to 0
        assert abs(mem.item()) < 0.01

    def test_reset_mem(self):
        neuron = LIFNeuron()
        x = torch.randn(32, 128)
        neuron(x)
        assert neuron.mem is not None

        neuron.reset_mem()
        assert neuron.mem is None


class TestAdaptiveLIF:
    """Tests for Adaptive LIF neuron."""

    def test_adaptation(self):
        neuron = AdaptiveLIFNeuron(adaptation_beta=0.5)
        x = torch.ones(1, 10) * 2  # Strong input

        # Fire repeatedly
        for _ in range(20):
            spk, mem = neuron(x)

        # After many spikes, adaptation should be high
        assert neuron.adaptation.mean() > 0


class TestSurrogateGradients:
    """Tests for surrogate gradient functions."""

    def test_atan_forward(self):
        fn = get_surrogate("atan")
        x = torch.tensor([-1.0, 0.0, 1.0])
        y = fn(x)
        assert y.tolist() == [0.0, 1.0, 1.0]

    def test_atan_backward(self):
        fn = get_surrogate("atan")
        x = torch.tensor([0.0], requires_grad=True)
        y = fn(x)
        y.backward()
        assert x.grad is not None
        assert x.grad.item() > 0

    def test_fast_sigmoid(self):
        fn = get_surrogate("fast_sigmoid")
        x = torch.tensor([0.0], requires_grad=True)
        y = fn(x)
        y.backward()
        assert x.grad is not None

    def test_straight_through(self):
        fn = get_surrogate("straight_through")
        x = torch.tensor([0.5], requires_grad=True)
        y = fn(x)
        y.backward()
        assert x.grad.item() == 1.0


class TestEncoding:
    """Tests for spike encoding schemes."""

    def test_rate_encoder_shape(self):
        encoder = RateEncoder(num_steps=25)
        x = torch.rand(32, 784)  # MNIST-like
        spikes = encoder(x)
        assert spikes.shape == (25, 32, 784)

    def test_rate_encoder_binary(self):
        encoder = RateEncoder(num_steps=25)
        x = torch.rand(32, 784)
        spikes = encoder(x)
        assert torch.all((spikes == 0) | (spikes == 1))

    def test_temporal_encoder(self):
        encoder = TemporalEncoder(num_steps=25)
        x = torch.rand(32, 100)
        spikes = encoder(x)
        assert spikes.shape == (25, 32, 100)
        # Each position should have exactly one spike
        assert torch.all(spikes.sum(dim=0) <= 1)

    def test_spike_decoder_rate(self):
        decoder = SpikeDecoder(method="rate")
        spikes = torch.rand(25, 32, 10) > 0.5
        output = decoder(spikes.float())
        assert output.shape == (32, 10)
        assert torch.all(output >= 0) and torch.all(output <= 1)


class TestSNNCore:
    """Tests for feedforward SNN."""

    def test_init(self):
        model = SNNCore(
            input_size=784,
            hidden_sizes=[256, 128],
            output_size=10,
        )
        assert len(model.layers) == 3

    def test_forward_shape(self):
        model = SNNCore(
            input_size=784,
            hidden_sizes=[256, 128],
            output_size=10,
            num_steps=25,
        )
        x = torch.randn(32, 784)
        spikes, mem = model(x)
        assert spikes.shape == (25, 32, 10)
        assert mem.shape == (32, 10)

    def test_predict(self):
        model = SNNCore(
            input_size=784,
            hidden_sizes=[256, 128],
            output_size=10,
        )
        x = torch.randn(32, 784)
        preds = model.predict(x)
        assert preds.shape == (32,)
        assert torch.all(preds >= 0) and torch.all(preds < 10)

    def test_temporal_input(self):
        model = SNNCore(
            input_size=784,
            hidden_sizes=[256],
            output_size=10,
            num_steps=25,
        )
        # Temporal input: (time, batch, features)
        x = torch.randn(25, 32, 784)
        spikes, mem = model(x)
        assert spikes.shape == (25, 32, 10)

    def test_gradient_flow(self):
        model = SNNCore(
            input_size=100,
            hidden_sizes=[50],
            output_size=10,
        )
        x = torch.randn(8, 100, requires_grad=True)
        spikes, mem = model(x)
        loss = spikes.sum()
        loss.backward()
        assert x.grad is not None


class TestConvSNN:
    """Tests for convolutional SNN."""

    def test_init(self):
        model = ConvSNN(
            input_channels=3,
            channels=[32, 64],
            num_classes=10,
        )
        assert len(model.conv_layers) == 2

    def test_forward_shape(self):
        model = ConvSNN(
            input_channels=3,
            channels=[32, 64],
            fc_sizes=[128],
            num_classes=10,
            num_steps=10,
        )
        x = torch.randn(8, 3, 32, 32)  # CIFAR-like
        spikes, mem = model(x)
        assert spikes.shape == (10, 8, 10)
        assert mem.shape == (8, 10)

    def test_predict(self):
        model = ConvSNN(
            input_channels=1,
            channels=[16, 32],
            num_classes=10,
            num_steps=5,
        )
        x = torch.randn(4, 1, 28, 28)  # MNIST-like
        preds = model.predict(x)
        assert preds.shape == (4,)


class TestIntegration:
    """Integration tests."""

    def test_snn_with_encoding(self):
        encoder = RateEncoder(num_steps=25)
        model = SNNCore(
            input_size=784,
            hidden_sizes=[128],
            output_size=10,
            num_steps=25,
        )

        x = torch.rand(16, 784)
        spikes, mem = model(x, encode_input=True)
        assert spikes.shape == (25, 16, 10)

    def test_training_step(self):
        model = SNNCore(
            input_size=784,
            hidden_sizes=[256],
            output_size=10,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))

        optimizer.zero_grad()
        spikes, mem = model(x)
        output = spikes.sum(dim=0)  # Rate coding
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
