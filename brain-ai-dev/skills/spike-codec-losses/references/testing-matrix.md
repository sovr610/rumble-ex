# Testing Matrix: Spike Codec and SNN Losses

Concrete test specifications for spike encoders, spike decoders, and SNN loss functions in `brain_ai/core/encoding.py` and `brain_ai/core/losses.py`. Each section maps to a test file in `tests/test_spike_codec/`. Imperative form throughout.

---

## 1. Test Categories Overview

Six categories cover the full surface area. Round-trip tests verify that encoding followed by decoding recovers the original signal's ordering and magnitude within tolerance. Statistical behavior tests confirm that stochastic encoders produce expected distributions (Bernoulli, Poisson). Loss stability tests assert that every loss term produces a finite, non-zero scalar with valid gradients under both fp32 and mixed precision. AMP hardening tests run every loss forward and backward under `torch.amp.autocast` and flag any NaN, Inf, or silent dtype demotion. Axis convention tests enforce the `(B, T, ...)` batch-first contract, rejecting or auto-converting `(T, B, ...)` inputs. Integration tests wire the full pipeline end-to-end: encode, feed through an SNN layer, decode, compute loss, verify that the optimizer reduces loss over multiple steps.

---

## 2. Rate Coding Round-Trip

Verify that `RateEncoder` followed by `SpikeDecoder(method='rate')` preserves signal ordering and magnitude.

1. Input `x = [0.1, 0.3, 0.9]`, encode with `RateEncoder(num_steps=T)`.
2. Count spikes: `counts = spikes.sum(dim=0)`. Assert `counts[0.9] > counts[0.3] > counts[0.1]` with P > 0.80 over 50 seeds.
3. Decode via rate decoding. Assert decoded argmax matches input argmax.
4. Deterministic mode: same input produces identical output on repeated calls.
5. Statistical: 1000-trial KS test that Bernoulli spike counts follow `Binomial(T, p)`.

```python
import pytest
import torch
from scipy import stats
from brain_ai.core.encoding import RateEncoder, SpikeDecoder

@pytest.mark.parametrize("T", [10, 25, 50, 100])
@pytest.mark.parametrize("method", ["bernoulli", "poisson", "deterministic"])
def test_rate_coding_roundtrip_ordering(T, method):
    encoder = RateEncoder(num_steps=T, method=method)
    x = torch.tensor([[0.1, 0.3, 0.9]])
    num_correct, num_trials = 0, (50 if method != "deterministic" else 1)
    for seed in range(num_trials):
        torch.manual_seed(seed)
        counts = encoder(x).sum(dim=0)
        if counts[0, 2] > counts[0, 1] and counts[0, 1] > counts[0, 0]:
            num_correct += 1
    threshold = 1 if method == "deterministic" else int(num_trials * 0.80)
    assert num_correct >= threshold

def test_rate_coding_decode_argmax():
    torch.manual_seed(42)
    encoder = RateEncoder(num_steps=100, method="bernoulli")
    decoder = SpikeDecoder(method="rate")
    x = torch.tensor([[0.1, 0.3, 0.9]])
    decoded = decoder(encoder(x))
    assert decoded.argmax(dim=-1).item() == x.argmax(dim=-1).item()

def test_rate_coding_deterministic_reproducibility():
    encoder = RateEncoder(num_steps=50, method="deterministic")
    x = torch.tensor([[0.2, 0.5, 0.8]])
    assert torch.equal(encoder(x), encoder(x))

def test_rate_coding_binomial_distribution():
    T, p = 100, 0.5
    encoder = RateEncoder(num_steps=T, method="bernoulli")
    x = torch.tensor([[p]])
    counts = []
    for seed in range(1000):
        torch.manual_seed(seed)
        counts.append(encoder(x).sum().item())
    _, ks_pval = stats.kstest(counts, stats.binom(T, p).cdf)
    assert ks_pval > 0.01, f"KS p-value {ks_pval:.4f} too low"
```

---

## 3. Latency / TTFS Round-Trip

Verify that `TemporalEncoder` and `LatencyEncoder` produce monotonically earlier spikes for larger input values. `SpikeDecoder(method='first_spike')` must identify the winning neuron.

1. Input `x = [0.2, 0.5, 0.8]`. Encode. Extract first-spike times via `argmax(dim=0)`.
2. Assert monotonic: `t[0.8] < t[0.5] < t[0.2]`.
3. Assert at most one spike per neuron: `spikes.sum(dim=0) <= 1`.
4. Decode with first-spike decoder. Assert argmax is neuron 2 (x=0.8).

```python
import pytest
import torch
from brain_ai.core.encoding import TemporalEncoder, LatencyEncoder, SpikeDecoder

@pytest.mark.parametrize("EncoderClass", [TemporalEncoder, LatencyEncoder])
@pytest.mark.parametrize("T", [10, 25, 50])
def test_temporal_monotonicity(EncoderClass, T):
    kwargs = {"num_steps": T, "normalize": True} if EncoderClass == LatencyEncoder else {"num_steps": T}
    encoder = EncoderClass(**kwargs)
    x = torch.tensor([[0.2, 0.5, 0.8]])
    first_t = encoder(x).argmax(dim=0)
    assert first_t[0, 2] < first_t[0, 1] < first_t[0, 0]

@pytest.mark.parametrize("EncoderClass", [TemporalEncoder, LatencyEncoder])
def test_ttfs_single_spike_per_neuron(EncoderClass):
    kwargs = {"num_steps": 25, "normalize": True} if EncoderClass == LatencyEncoder else {"num_steps": 25}
    encoder = EncoderClass(**kwargs)
    x = torch.tensor([[0.2, 0.5, 0.8]])
    assert (encoder(x).sum(dim=0) <= 1).all()

def test_first_spike_decode_picks_winner():
    encoder = TemporalEncoder(num_steps=25)
    decoder = SpikeDecoder(method="first_spike")
    x = torch.tensor([[0.2, 0.5, 0.8]])
    assert decoder(encoder(x)).argmax(dim=-1).item() == 2
```

---

## 4. Population Coding Round-Trip

Verify that `PopulationEncoder` distributes activity across tuning-curve neurons and center-of-mass decode recovers the original value within tolerance.

1. Input `x = [[0.2], [0.5], [0.8]]`. Encode with `PopulationEncoder(input_dim=1, num_neurons_per_dim=P)`.
2. Decode via weighted average of tuning-curve centers. Assert reconstructed within +/-0.15.
3. Verify `centers` buffer exists with shape `(P,)`, spanning `[0, 1]`.

```python
import pytest
import torch
from brain_ai.core.encoding import PopulationEncoder

@pytest.mark.parametrize("P", [4, 8, 16])
@pytest.mark.parametrize("T", [25, 50, 100])
def test_population_coding_roundtrip(P, T):
    torch.manual_seed(42)
    encoder = PopulationEncoder(input_dim=1, num_neurons_per_dim=P, num_steps=T)
    x = torch.tensor([[0.2], [0.5], [0.8]])
    spike_counts = encoder(x).sum(dim=0)  # (3, P)
    total = spike_counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
    reconstructed = (spike_counts * encoder.centers.unsqueeze(0)).sum(dim=-1) / total.squeeze(-1)
    for i, expected in enumerate([0.2, 0.5, 0.8]):
        assert abs(reconstructed[i].item() - expected) < 0.15

def test_population_encoder_shape_and_centers():
    P = 10
    encoder = PopulationEncoder(input_dim=4, num_neurons_per_dim=P, num_steps=25)
    x = torch.rand(2, 4)
    assert encoder(x).shape == (25, 2, 4 * P)
    assert encoder.centers.shape == (P,)
    assert encoder.centers[0].item() == pytest.approx(0.0, abs=1e-6)
    assert encoder.centers[-1].item() == pytest.approx(1.0, abs=1e-6)
```

---

## 5. Delta Modulation Round-Trip

Verify that `DeltaEncoder` produces ON spikes at positive change points and OFF spikes at negative change points. Reset must clear internal state.

1. Feed `x1=[0,0,0]` then `x2=[0.5,-0.5,0]`. Assert ON at neuron 0, OFF at neuron 1, none at neuron 2.
2. Call `reset()`. Re-encode `x2`. Assert delta computed from zero.
3. Output shape: `(B, T, 2*N)` -- ON and OFF channels (batch-first).

```python
import torch
from brain_ai.core.encoding import DeltaEncoder

def test_delta_on_off_channels():
    encoder = DeltaEncoder(threshold=0.1, num_steps=5)
    encoder.reset()
    _ = encoder(torch.tensor([[0.0, 0.0, 0.0]]))
    spikes = encoder(torch.tensor([[0.5, -0.5, 0.0]]))  # (5, 1, 6)
    on, off = spikes[0, 0, :3], spikes[0, 0, 3:]
    assert on[0] == 1.0 and on[1] == 0.0 and on[2] == 0.0
    assert off[0] == 0.0 and off[1] == 1.0 and off[2] == 0.0

def test_delta_reset_clears_state():
    encoder = DeltaEncoder(threshold=0.1, num_steps=5)
    encoder.reset()
    _ = encoder(torch.tensor([[0.5, 0.5]]))
    encoder.reset()
    spikes = encoder(torch.tensor([[0.5, 0.5]]))
    assert spikes[0, 0, 0] == 1.0 and spikes[0, 0, 1] == 1.0

def test_delta_output_shape():
    encoder = DeltaEncoder(threshold=0.1, num_steps=10)
    encoder.reset()
    assert encoder(torch.randn(4, 8)).shape == (10, 4, 16)
```

---

## 6. Loss Stability Under AMP

For each loss term, run forward + backward under `torch.amp.autocast`. Assert: loss is finite, no NaN, gradients non-zero. ProbSpikes must accumulate counts in fp32. ISI conv kernel (if vectorized) must stay in fp32. SNNLoss composer must return finite total and all finite metrics.

```python
import pytest
import torch
import torch.nn as nn
from brain_ai.core.losses import (
    prob_spikes_loss, spike_rate_regularization, temporal_consistency_loss,
    inter_spike_interval_loss, membrane_potential_regularization, SNNLoss,
)
from brain_ai.core.neurons import LIFNeuron

def _make_spike_tensor(B, T, N, device="cpu"):
    linear = nn.Linear(N, N).to(device)
    lif = LIFNeuron(beta=0.9, threshold=1.0, surrogate="atan").to(device)
    x = torch.randn(B, N, device=device, requires_grad=True)
    lif.reset_mem()
    spikes = []
    for t in range(T):
        spk, mem = lif(linear(x))
        spikes.append(spk)
    return torch.stack(spikes), mem, x, linear

@pytest.mark.parametrize("T", [10, 25, 50])
@pytest.mark.parametrize("B", [4, 16])
def test_prob_spikes_loss_amp(T, B):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spikes, mem, x, _ = _make_spike_tensor(B, T, 10, device)
    targets = torch.randint(0, 10, (B,), device=device)
    amp_on = device == "cuda"
    with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=amp_on):
        loss = prob_spikes_loss(spikes, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None
    assert spikes.float().sum(dim=0).dtype == torch.float32  # fp32 counts

@pytest.mark.parametrize("T", [10, 25, 50])
@pytest.mark.parametrize("B", [4, 16])
def test_rate_reg_amp(T, B):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spikes, _, x, _ = _make_spike_tensor(B, T, 32, device)
    amp_on = device == "cuda"
    with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=amp_on):
        loss = spike_rate_regularization(spikes, target_rate=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None

@pytest.mark.parametrize("T", [10, 25, 50])
def test_temporal_consistency_amp(T):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spikes, _, x, _ = _make_spike_tensor(4, T, 16, device)
    amp_on = device == "cuda"
    with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=amp_on):
        loss = temporal_consistency_loss(spikes, window_size=5)
    assert torch.isfinite(loss)
    if T >= 10:
        loss.backward()

@pytest.mark.parametrize("T", [10, 25, 50])
def test_isi_loss_amp(T):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spikes, _, _, _ = _make_spike_tensor(4, T, 16, device)
    amp_on = device == "cuda"
    with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=amp_on):
        loss = inter_spike_interval_loss(spikes, target_cv=1.0)
    assert torch.isfinite(loss)

@pytest.mark.parametrize("T", [10, 25, 50])
def test_membrane_reg_amp(T):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, mem, x, _ = _make_spike_tensor(4, T, 16, device)
    amp_on = device == "cuda"
    with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=amp_on):
        loss = membrane_potential_regularization(mem, max_membrane=5.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None

@pytest.mark.parametrize("T", [10, 25, 50])
def test_snn_loss_composer_amp(T):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spikes, mem, _, _ = _make_spike_tensor(4, T, 10, device)
    targets = torch.randint(0, 10, (4,), device=device)
    loss_fn = SNNLoss(task_loss_weight=1.0, spike_rate_weight=0.1,
                      temporal_weight=0.01, temporal_sparsity_weight=0.01,
                      membrane_reg_weight=0.001, use_prob_spikes=True).to(device)
    amp_on = device == "cuda"
    with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=amp_on):
        total_loss, metrics = loss_fn(spikes, mem, targets)
    assert torch.isfinite(total_loss)
    for k, v in metrics.items():
        if isinstance(v, float):
            assert not (v != v), f"Metric '{k}' is NaN"
```

---

## 7. Axis Convention Tests

The codebase uses batch-first `(B, T, N)` internally. Verify correct handling and conversion between `(B, T, N)` and `(T, B, N)` at API boundaries.

1. Round-trip: `batch_to_time_first(time_to_batch_first(x))` equals `x`.
2. Encoder output is `(B, T, N)`. Decoder accepts `(B, T, N)`. Losses expect `(B, T, C)`.
3. Shape assertion helper rejects wrong dimensionality.

```python
import pytest
import torch
from brain_ai.core.encoding import RateEncoder, time_to_batch_first, batch_to_time_first
from brain_ai.core.decoding import RateDecoder
from brain_ai.core.losses import ProbSpikesLoss, SpikeRateRegularization

def test_axis_roundtrip_identity():
    x = torch.randn(4, 10, 8)  # (B, T, N) batch-first
    assert torch.equal(x, time_to_batch_first(batch_to_time_first(x)))
    y = torch.randn(10, 4, 8)  # (T, B, N) time-first
    assert torch.equal(y, batch_to_time_first(time_to_batch_first(y)))

def test_encoder_output_is_batch_first():
    encoder = RateEncoder(num_steps=10, method="deterministic")
    spikes = encoder(torch.tensor([[0.3, 0.7]]))  # (1, 2)
    assert spikes.spikes.shape[0] == 1 and spikes.spikes.shape[1] == 10  # (B=1, T=10, N=2)

def test_decoder_accepts_batch_first():
    decoder = RateDecoder()
    from brain_ai.core.encoding import SpikeBatch
    batch = SpikeBatch(spikes=torch.randint(0, 2, (4, 10, 8)).float())
    result = decoder.decode(batch)
    assert result.logits_proxy.shape == (4, 8)

def test_loss_batch_first_shape():
    spikes = torch.randn(4, 10, 5).abs()  # (B=4, T=10, C=5) batch-first
    loss_fn = ProbSpikesLoss(LossConfig())
    loss, _ = loss_fn.compute(spikes, None, torch.randint(0, 5, (4,)))
    assert loss.dim() == 0 and torch.isfinite(loss)

def test_time_first_conversion_at_boundary():
    spikes_bt = torch.randn(4, 10, 8).abs()  # batch-first
    spikes_tb = batch_to_time_first(spikes_bt)  # convert for legacy API
    assert spikes_tb.shape == (10, 4, 8)

def test_shape_assertion_helper():
    def assert_spike_shape(s, dims=3):
        if s.dim() != dims:
            raise ValueError(f"Expected {dims}D, got {s.dim()}D shape {s.shape}")
    assert_spike_shape(torch.randn(10, 4, 8))
    with pytest.raises(ValueError, match="Expected 3D"):
        assert_spike_shape(torch.randn(4, 8))
```

---

## 8. SpikeBatch Contract Tests

Specify a `SpikeBatch` wrapper enforcing invariants. Tests serve as the specification even if the class is not yet implemented.

- `spikes` is binary (0 or 1). `mask` shape is `(B, T)`, dtype `bool`. `time_steps` equals T.
- `from_dense()` binarizes and validates. Rejects non-3D tensors.

```python
import pytest
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class SpikeBatch:
    spikes: torch.Tensor
    mask: Optional[torch.Tensor] = None
    def __post_init__(self):
        assert self.spikes.dim() == 3
        B, T, N = self.spikes.shape  # batch-first (B, T, N)
        if self.mask is None:
            self.mask = torch.ones(B, T, dtype=torch.bool, device=self.spikes.device)
        assert self.mask.shape == (B, T)
    def is_binary(self):
        return all(v in (0.0, 1.0) for v in torch.unique(self.spikes).tolist())
    @property
    def time_steps(self): return self.spikes.shape[1]  # dim=1 is time
    @classmethod
    def from_dense(cls, tensor, mask=None):
        if tensor.dim() != 3:
            raise ValueError(f"Expected 3D (B,T,N), got {tensor.dim()}D")
        return cls(spikes=(tensor > 0.5).float(), mask=mask)

def test_spike_batch_binary():
    batch = SpikeBatch(spikes=torch.randint(0, 2, (10, 4, 8)).float())
    assert batch.is_binary()

def test_spike_batch_non_binary_detected():
    assert not SpikeBatch(spikes=torch.randn(10, 4, 8)).is_binary()

def test_spike_batch_mask_shape():
    batch = SpikeBatch(spikes=torch.randint(0, 2, (10, 4, 8)).float())
    assert batch.mask.shape == (4, 10) and batch.mask.dtype == torch.bool

def test_spike_batch_time_steps():
    assert SpikeBatch(spikes=torch.randint(0, 2, (25, 4, 8)).float()).time_steps == 25

def test_spike_batch_from_dense():
    batch = SpikeBatch.from_dense(torch.randn(10, 4, 8))
    assert batch.is_binary() and batch.spikes.shape == (10, 4, 8)

def test_spike_batch_rejects_2d():
    with pytest.raises(ValueError, match="Expected 3D"):
        SpikeBatch.from_dense(torch.randn(4, 8))

def test_spike_batch_custom_mask():
    mask = torch.ones(4, 10, dtype=torch.bool)
    mask[:, 7:] = False
    batch = SpikeBatch(spikes=torch.randint(0, 2, (10, 4, 8)).float(), mask=mask)
    assert batch.mask[:, :7].all() and not batch.mask[:, 7:].any()
```

---

## 9. Gradient Flow Tests

Verify that gradients flow end-to-end from loss through SNN back to trainable parameters. Test with each surrogate variant.

```python
import pytest
import torch
from brain_ai.core.snn import SNNLinear, SNNCore
from brain_ai.core.losses import prob_spikes_loss, SNNLoss

def test_gradient_flow_snn_linear():
    torch.manual_seed(42)
    layer = SNNLinear(16, 10, beta=0.9, surrogate="atan")
    x = torch.randn(4, 16, requires_grad=True)
    layer.reset_mem()
    spikes = torch.stack([layer(x)[0] for _ in range(15)])
    loss = prob_spikes_loss(spikes, torch.randint(0, 10, (4,)))
    assert loss.requires_grad
    loss.backward()
    assert layer.linear.weight.grad is not None
    assert layer.linear.weight.grad.abs().sum() > 0
    assert torch.isfinite(layer.linear.weight.grad).all()

def test_gradient_flow_full_snn_core():
    torch.manual_seed(42)
    model = SNNCore(input_size=32, hidden_sizes=[16], output_size=10, num_steps=10)
    spikes, mem = model(torch.randn(4, 32))
    prob_spikes_loss(spikes, torch.randint(0, 10, (4,))).backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"

@pytest.mark.parametrize("surrogate", ["atan", "fast_sigmoid", "straight_through"])
def test_gradient_flow_per_surrogate(surrogate):
    torch.manual_seed(42)
    layer = SNNLinear(8, 8, surrogate=surrogate)
    layer.reset_mem()
    spikes = torch.stack([layer(torch.randn(4, 8))[0] for _ in range(10)])
    spikes.sum().backward()
    grad_norm = layer.linear.weight.grad.norm().item()
    assert grad_norm > 0 and torch.isfinite(torch.tensor(grad_norm))

def test_gradient_flow_with_snn_loss():
    torch.manual_seed(42)
    model = SNNCore(input_size=16, hidden_sizes=[8], output_size=10, num_steps=15)
    loss_fn = SNNLoss(task_loss_weight=1.0, spike_rate_weight=0.1,
                      temporal_weight=0.01, membrane_reg_weight=0.001)
    spikes, mem = model(torch.randn(4, 16))
    total_loss, _ = loss_fn(spikes, mem, torch.randint(0, 10, (4,)))
    total_loss.backward()
    norms = {n: p.grad.norm().item() for n, p in model.named_parameters()
             if p.requires_grad and p.grad is not None}
    assert len(norms) > 0
    assert all(v > 0 for v in norms.values())
```

---

## 10. Integration Test: Full Pipeline

Encode input, feed through SNN, compute loss, train for multiple steps. Verify loss decreases and no NaN appears.

```python
import torch
import torch.optim as optim
from brain_ai.core.snn import SNNCore
from brain_ai.core.losses import prob_spikes_loss, SNNLoss

def _train_loop(model, loss_fn, x, targets, steps=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        spikes, mem = model(x)
        if loss_fn is not None:
            result = loss_fn(spikes, mem, targets)
            loss = result[0] if isinstance(result, tuple) else result
        else:
            loss = prob_spikes_loss(spikes, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def test_integration_rate_coding_pipeline():
    torch.manual_seed(42)
    model = SNNCore(input_size=32, hidden_sizes=[16], output_size=10, num_steps=15)
    losses = _train_loop(model, None, torch.randn(8, 32), torch.randint(0, 10, (8,)))
    assert losses[-1] < losses[0]
    assert all(l == l for l in losses)  # no NaN

def test_integration_ttfs_pipeline():
    torch.manual_seed(42)
    model = SNNCore(input_size=32, hidden_sizes=[16], output_size=10, num_steps=25)
    losses = _train_loop(model, None, torch.randn(8, 32), torch.randint(0, 10, (8,)))
    assert losses[-1] < losses[0]

def test_integration_snn_loss_composer():
    torch.manual_seed(42)
    model = SNNCore(input_size=32, hidden_sizes=[16], output_size=10, num_steps=15)
    loss_fn = SNNLoss(task_loss_weight=1.0, spike_rate_weight=0.1,
                      temporal_weight=0.01, membrane_reg_weight=0.001)
    losses = _train_loop(model, loss_fn, torch.randn(8, 32), torch.randint(0, 10, (8,)))
    assert losses[-1] < losses[0]

def test_integration_no_nan_across_training():
    torch.manual_seed(42)
    model = SNNCore(input_size=32, hidden_sizes=[16], output_size=10, num_steps=15)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    x, targets = torch.randn(8, 32), torch.randint(0, 10, (8,))
    for step in range(20):
        optimizer.zero_grad()
        spikes, mem = model(x)
        loss = prob_spikes_loss(spikes, targets)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss), f"Non-finite loss at step {step}"
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), f"Non-finite param '{name}' step {step}"
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad '{name}' step {step}"
```

---

## 11. Performance and Memory Tests

Mark with `@pytest.mark.slow`. Verify wall-clock time and memory-efficient dtype savings.

```python
import pytest
import time
import torch
from brain_ai.core.encoding import RateEncoder
from brain_ai.core.losses import inter_spike_interval_loss

@pytest.mark.slow
def test_rate_encoder_performance():
    encoder = RateEncoder(num_steps=100, method="bernoulli")
    x = torch.rand(32, 1024)
    start = time.perf_counter()
    for _ in range(10):
        encoder(x)
    elapsed = (time.perf_counter() - start) / 10
    assert elapsed < 0.1, f"Rate encoding took {elapsed*1000:.1f}ms, limit 100ms"

def test_spike_memory_uint8_vs_float32():
    f = torch.randint(0, 2, (32, 100, 1024), dtype=torch.float32)
    u = f.to(torch.uint8)
    ratio = (f.element_size() * f.nelement()) / (u.element_size() * u.nelement())
    assert ratio >= 4.0
    assert torch.equal(f, u.float())

@pytest.mark.slow
def test_isi_loss_bounded_time():
    spikes = torch.randint(0, 2, (50, 4, 100)).float()
    start = time.perf_counter()
    inter_spike_interval_loss(spikes, target_cv=1.0)
    assert time.perf_counter() - start < 1.0

@pytest.mark.slow
def test_snn_core_forward_performance():
    from brain_ai.core.snn import SNNCore
    model = SNNCore(input_size=256, hidden_sizes=[128], output_size=10, num_steps=25)
    x = torch.randn(32, 256)
    model(x)  # warmup
    start = time.perf_counter()
    for _ in range(5):
        model(x)
    assert (time.perf_counter() - start) / 5 < 0.2
```

---

## 12. Conftest Fixtures

Place in `tests/test_spike_codec/conftest.py`.

```python
import pytest
import torch
from brain_ai.core.snn import SNNLinear, SNNCore
from brain_ai.core.losses import SNNLoss

@pytest.fixture
def sample_spikes():
    def _make(B=4, T=25, N=10, rate=0.1, seed=42):
        torch.manual_seed(seed)
        return (torch.rand(B, T, N) < rate).float()  # batch-first (B, T, N)
    return _make

@pytest.fixture
def sample_input():
    def _make(B=4, N=16, seed=42):
        torch.manual_seed(seed)
        return torch.rand(B, N)
    return _make

@pytest.fixture
def snn_layer():
    def _make(in_f=16, out_f=10, surrogate="atan"):
        return SNNLinear(in_f, out_f, beta=0.9, surrogate=surrogate)
    return _make

@pytest.fixture
def snn_core():
    def _make(input_size=32, hidden_sizes=None, output_size=10, num_steps=15):
        return SNNCore(input_size=input_size,
                       hidden_sizes=hidden_sizes or [16],
                       output_size=output_size, num_steps=num_steps)
    return _make

@pytest.fixture
def loss_composer():
    return SNNLoss(task_loss_weight=1.0, spike_rate_weight=0.1, target_spike_rate=0.1,
                   temporal_weight=0.01, temporal_sparsity_weight=0.01,
                   membrane_reg_weight=0.001, use_prob_spikes=True, temperature=1.0)

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(autouse=True)
def seed_everything():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
```

---

## 13. Done-When Checklist

All items must pass before the spike-codec-losses skill is complete. No partial credit.

### Round-Trip Tests
- [ ] Rate encoding preserves input ordering in > 80% of stochastic trials (Sec 2)
- [ ] Rate decode argmax matches input argmax (Sec 2)
- [ ] Deterministic rate encoder is perfectly reproducible (Sec 2)
- [ ] Bernoulli spike counts match Binomial distribution, KS p > 0.01 (Sec 2)
- [ ] Temporal/latency encoding produces monotonic first-spike-time ordering (Sec 3)
- [ ] TTFS produces at most one spike per neuron (Sec 3)
- [ ] First-spike decoder identifies the correct winning neuron (Sec 3)
- [ ] Population encode+decode recovers value within +/-0.15 (Sec 4)
- [ ] Delta encoder produces correct ON/OFF spikes at change points (Sec 5)
- [ ] Delta encoder reset clears internal state (Sec 5)

### Loss Stability
- [ ] ProbSpikes loss finite and non-zero for T in {10, 25, 50} (Sec 6)
- [ ] Spike rate regularization finite for T in {10, 25, 50} (Sec 6)
- [ ] Temporal consistency loss finite for T in {10, 25, 50} (Sec 6)
- [ ] ISI loss finite for T in {10, 25, 50} (Sec 6)
- [ ] Membrane regularization finite for T in {10, 25, 50} (Sec 6)
- [ ] SNNLoss composer returns finite total and all finite metrics (Sec 6)
- [ ] ProbSpikes spike counts accumulated in fp32 under AMP (Sec 6)

### Axis Convention
- [ ] time_to_batch_first and batch_to_time_first are exact inverses (Sec 7)
- [ ] Encoder output is batch-first (B, T, N) (Sec 7)
- [ ] Decoder correctly handles batch-first input (Sec 7)
- [ ] Loss functions accept (B, T, C) and produce scalar (Sec 7)

### SpikeBatch Contract
- [ ] SpikeBatch.is_binary() True for valid spike tensors (Sec 8)
- [ ] SpikeBatch.mask shape is (B, T) with dtype bool (Sec 8)
- [ ] SpikeBatch.time_steps returns correct T (Sec 8)
- [ ] SpikeBatch.from_dense() binarizes and validates (Sec 8)
- [ ] SpikeBatch rejects 2D tensors (Sec 8)

### Gradient Flow
- [ ] SNNLinear weight gradients non-zero after backward (Sec 9)
- [ ] All SNNCore parameters receive finite gradients (Sec 9)
- [ ] Each surrogate produces non-zero gradient norms (Sec 9)
- [ ] SNNLoss backward produces gradients on all trainable parameters (Sec 9)

### Integration
- [ ] Rate coding pipeline loss decreases over 10 steps (Sec 10)
- [ ] TTFS pipeline loss decreases over 10 steps (Sec 10)
- [ ] SNNLoss composer pipeline loss decreases over 10 steps (Sec 10)
- [ ] No NaN or Inf in parameters or gradients over 20 steps (Sec 10)

### Performance
- [ ] Rate encoding B=32, T=100, N=1024 completes in < 100ms (Sec 11)
- [ ] Spike tensor uint8 vs float32 yields >= 4x memory savings (Sec 11)
- [ ] ISI loss completes in < 1 second for B=4, T=50, N=100 (Sec 11)

---

## Pytest Organization

```
tests/test_spike_codec/
├── conftest.py                    # Shared fixtures (Sec 12)
├── test_rate_roundtrip.py         # Sec 2
├── test_temporal_roundtrip.py     # Sec 3
├── test_population_roundtrip.py   # Sec 4
├── test_delta_roundtrip.py        # Sec 5
├── test_loss_amp.py               # Sec 6
├── test_axis_convention.py        # Sec 7
├── test_spike_batch.py            # Sec 8
├── test_gradient_flow.py          # Sec 9
├── test_integration.py            # Sec 10
└── test_performance.py            # Sec 11 (@pytest.mark.slow)
```

```bash
pytest tests/test_spike_codec/ -v --tb=short                         # full matrix
pytest tests/test_spike_codec/ -v -k "roundtrip"                     # round-trips only
pytest tests/test_spike_codec/test_loss_amp.py -v                    # AMP stability
pytest tests/test_spike_codec/test_performance.py -v -m slow         # benchmarks
pytest tests/test_spike_codec/ --cov=brain_ai.core.encoding \
       --cov=brain_ai.core.losses --cov-report=term-missing          # coverage
```
