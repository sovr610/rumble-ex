# Testing Matrix: Learnable Delays and Heterogeneous Tau

Reference for the `delays-heterogeneous-tau` skill in BrainAI. Defines concrete test
specifications for DCLS-style learnable delays and heterogeneous/learnable membrane
time constants (tau) in spiking neural networks.

---

## 1. Test Categories Overview

Six categories cover the full contract surface of both features.

| # | Category | File | Focus |
|---|---|---|---|
| 1 | Delay Module Contract | `test_delay_module.py` | Shape, correctness, granularity, memory |
| 2 | Tau/Beta Contract | `test_heterogeneous_tau.py` | Shape, constraints, mapping, init |
| 3 | Gradient Flow | both files | Autograd correctness |
| 4 | Sigma Schedule | `test_delay_module.py` | Annealing schedule behavior |
| 5 | Checkpoint Round-Trip | `test_delay_tau_checkpoint.py` | Save/load fidelity |
| 6 | Integration | `test_delay_tau_integration.py` | End-to-end with LIF/SNN |

---

## 2. Category 1: Delay Module Contract Tests

### 1.1 Forward Shape — DelayLinear

Input tensor `(B, T, in_features)` must produce output `(B, T, out_features)`.
State buffer `spike_history` must have shape `(B, Td, in_features)`.

```python
def test_delay_linear_shapes():
    """DelayLinear produces correct output shape for (B, T, in_f) input."""
    B, T, in_f, out_f, Td = 4, 20, 64, 128, 16
    layer = DelayLinear(in_f, out_f, max_delay=Td)
    x = torch.randn(B, T, in_f)
    state = layer.reset_state(B, x.device, x.dtype)
    out, new_state = layer(x, state)
    assert out.shape == (B, T, out_f), f"expected {(B, T, out_f)}, got {out.shape}"
    assert new_state.spike_history.shape == (B, Td, in_f)
```

### 1.2 Forward Shape — DelayConv1d

Input tensor `(B, T, C_in, L)` must produce output `(B, T, C_out, L')` where `L'`
follows standard convolution rules.

```python
def test_delay_conv1d_shapes():
    """DelayConv1d preserves batch/time dims and applies conv along spatial dim."""
    B, T, C_in, C_out, L, Td = 2, 10, 8, 16, 32, 8
    layer = DelayConv1d(C_in, C_out, kernel_size=3, max_delay=Td, padding=1)
    x = torch.randn(B, T, C_in, L)
    state = layer.reset_state(B, x.device, x.dtype)
    out, _ = layer(x, state)
    assert out.shape == (B, T, C_out, L), \
        f"spatial length changed unexpectedly: {out.shape}"
```

### 1.3 Gaussian Interpolation Bins

For each delay position `d`, the Gaussian kernel weights across `Td` bins must sum
to approximately 1.0 (they define a probability distribution over delay bins).

```python
def test_gaussian_bins_sum_to_one():
    """Gaussian interpolation weights sum to 1.0 per delay position."""
    in_f, out_f, Td = 8, 8, 16
    layer = DelayLinear(in_f, out_f, max_delay=Td, sigma=1.0)
    weights = layer._gaussian_weights()   # shape: (out_f, in_f, Td) or (Td,)
    bin_sums = weights.sum(dim=-1)
    assert torch.allclose(bin_sums, torch.ones_like(bin_sums), atol=1e-4), \
        f"bins do not sum to 1; max deviation {(bin_sums - 1).abs().max()}"
```

### 1.4 Discretization Correctness at Small Sigma

When sigma approaches zero, the smoothed output must match a hard discrete delay
selection within `atol=1e-3`. Validates that the continuous relaxation collapses
correctly at inference time. The discretized forward path selects the integer bin
nearest to `d_raw`; the smoothed path applies Gaussian-weighted interpolation.
With sigma near zero the two paths must agree.

```python
def test_discretization_matches_hard_selection():
    """Near-zero sigma: smoothed output matches hard discrete delay selection."""
    B, T, in_f, out_f, Td = 2, 10, 8, 8, 8
    layer = DelayLinear(in_f, out_f, max_delay=Td, sigma=1e-5)
    layer.train(False)
    x = torch.randn(B, T, in_f)
    state = layer.reset_state(B, x.device, x.dtype)
    out_smooth, _ = layer(x, state)
    out_discrete, _ = layer.forward_discrete(x, state)
    assert torch.allclose(out_smooth, out_discrete, atol=1e-3), \
        f"max diff {(out_smooth - out_discrete).abs().max()}"
```

### 1.5 Granularity Parameter Shapes

The learnable delay parameter `d` must have a shape that matches the configured
granularity mode.

| Mode | Expected `d.shape` |
|---|---|
| `per_synapse` | `(out_features, in_features)` |
| `per_output` | `(out_features,)` |
| `per_input` | `(in_features,)` |
| `per_block` | `(ceil(out_features/block_size), ceil(in_features/block_size))` |

```python
import math
@pytest.mark.parametrize("granularity,expected_shape", [
    ("per_synapse", (out_f, in_f)),
    ("per_output",  (out_f,)),
    ("per_input",   (in_f,)),
    ("per_block",   (math.ceil(out_f / block_size), math.ceil(in_f / block_size))),
])
def test_delay_granularity_shapes(granularity, expected_shape):
    """d_raw shape matches the configured granularity mode."""
    in_f, out_f, Td = 8, 16, 8
    layer = DelayLinear(in_f, out_f, max_delay=Td, granularity=granularity)
    assert layer.d_raw.shape == expected_shape, \
        f"granularity={granularity}: expected {expected_shape}, got {layer.d_raw.shape}"
```

### 1.6 Memory Guard

Peak memory for a known `(out, in, Td)` configuration must not exceed the computed
threshold. Prevents silent regressions in buffer allocation.

```python
def test_delay_memory_within_budget(tiny_delay_config):
    """Peak memory stays below threshold for a small known config."""
    out_f, in_f, Td = 8, 8, 4
    threshold_mb = _delay_memory_budget_mb(B=4, T=20, out_f=out_f,
                                           in_f=in_f, Td=Td, safety=1.5)
    layer = DelayLinear(in_f, out_f, max_delay=Td)
    x = torch.randn(4, 20, in_f)
    state = layer.reset_state(4, x.device, x.dtype)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _, _ = layer(x, state)
    peak = _get_peak_memory_mb()
    assert peak < threshold_mb, f"peak {peak:.1f} MB > budget {threshold_mb:.1f} MB"
```

---

## 3. Category 2: Tau/Beta Contract Tests

### 2.1 Output Shape

`HeterogeneousTau` must return a beta tensor whose shape exactly matches the
neuron layer shape.

```python
def test_heterogeneous_tau_output_shape():
    """HeterogeneousTau beta has correct shape for per_neuron granularity."""
    N = 64
    tau_mod = HeterogeneousTau(N, granularity="per_neuron",
                               tau_mode="heterogeneous_learnable")
    beta = tau_mod()
    assert beta.shape == (N,), f"expected ({N},), got {beta.shape}"
```

### 2.2 Beta Strictly in (0, 1)

All computed beta values must satisfy `0 < beta < 1`. This is a hard biological
constraint: decay factor must be strictly bounded.

```python
@pytest.mark.parametrize("tau_init", [
    "homogeneous", "heterogeneous_gamma",
    "heterogeneous_loguniform", "preset_bank"
])
def test_beta_strictly_bounded(tau_init):
    """beta = exp(-dt/tau) is strictly in (0, 1) for all init strategies."""
    tau_mod = HeterogeneousTau(N=128, tau_init=tau_init, dt=1e-3,
                               tau_mode="heterogeneous_learnable")
    beta = tau_mod()
    assert (beta > 0).all(), "beta contains values <= 0"
    assert (beta < 1).all(), "beta contains values >= 1"
```

### 2.3 Tau Strictly Positive

Raw parameter `tau_raw` passes through `softplus` so that effective tau is always
positive. Verify the constraint holds after initialization.

```python
def test_tau_always_positive():
    """tau = tau_min + softplus(tau_raw) is strictly positive."""
    tau_mod = HeterogeneousTau(N=128, tau_mode="heterogeneous_learnable")
    tau = tau_mod.get_tau()
    assert (tau > 0).all(), f"tau contains non-positive values: min={tau.min()}"
```

### 2.4 Softplus Mapping Correctness

Verify that the effective tau matches the expected formula
`tau_min + softplus(tau_raw)` using known values.

```python
def test_softplus_mapping():
    """tau = tau_min + softplus(tau_raw) matches manual computation."""
    tau_min = 1e-3
    tau_mod = HeterogeneousTau(N=4, tau_min=tau_min,
                               tau_mode="heterogeneous_learnable")
    raw = tau_mod.tau_raw.detach().clone()
    expected = tau_min + F.softplus(raw)
    actual = tau_mod.get_tau()
    assert torch.allclose(actual, expected, atol=1e-7), \
        f"softplus mapping mismatch: max diff {(actual - expected).abs().max()}"
```

### 2.5 Exp Mapping Correctness

Verify that `beta = exp(-dt / tau)` matches manual computation.

```python
def test_exp_mapping():
    """beta = exp(-dt/tau) matches manual exp computation."""
    dt = 1e-3
    tau_mod = HeterogeneousTau(N=4, dt=dt, tau_mode="heterogeneous_learnable")
    tau = tau_mod.get_tau()
    expected_beta = torch.exp(-dt / tau)
    actual_beta = tau_mod()
    assert torch.allclose(actual_beta, expected_beta, atol=1e-7)
```

### 2.6 Granularity Shapes

| Mode | Expected `tau_raw.shape` | Broadcast target |
|---|---|---|
| `per_neuron` | `(N,)` | `(N,)` |
| `per_channel` | `(C, 1, 1)` | feature maps |
| `per_layer` | `()` or `(1,)` | scalar |

```python
@pytest.mark.parametrize("granularity,N,C,expected", [
    ("per_neuron",  64, None, (64,)),
    ("per_channel", None, 8,  (8, 1, 1)),
    ("per_layer",   None, None, ()),
])
def test_tau_granularity_shapes(granularity, N, C, expected):
    """tau_raw.shape matches the configured tau granularity."""
    tau_mod = HeterogeneousTau(N=N, C=C, granularity=granularity,
                               tau_mode="heterogeneous_learnable")
    assert tau_mod.tau_raw.shape == expected, \
        f"granularity={granularity}: expected {expected}, got {tau_mod.tau_raw.shape}"
```

### 2.7 Initialization Distributions

All four init strategies must produce tau values in a physiologically plausible
range `[tau_min, tau_max]` without NaN or Inf.

```python
@pytest.mark.parametrize("tau_init", [
    "homogeneous", "heterogeneous_gamma",
    "heterogeneous_loguniform", "preset_bank"
])
def test_tau_init_produces_valid_distribution(tau_init):
    """All init strategies produce finite tau values within plausible range."""
    tau_mod = HeterogeneousTau(N=256, tau_init=tau_init,
                               tau_mode="heterogeneous_learnable",
                               tau_min=1e-4, tau_max=0.5)
    tau = tau_mod.get_tau()
    assert torch.isfinite(tau).all(), f"{tau_init}: tau contains NaN/Inf"
    assert (tau >= 1e-4).all(), f"{tau_init}: tau below tau_min"
```

---

## 4. Category 3: Gradient Flow Tests

### 4.1 Gradient Through Delays

Loss backpropagation must write a non-None, non-zero gradient into `d_raw`.

```python
def test_gradient_flows_through_delays():
    """d_raw.grad is not None and has nonzero mean after backward."""
    B, T, in_f, out_f, Td = 2, 10, 8, 8, 4
    layer = DelayLinear(in_f, out_f, max_delay=Td,
                        delays_mode="learned")
    x = torch.randn(B, T, in_f)
    state = layer.reset_state(B, x.device, x.dtype)
    out, _ = layer(x, state)
    out.sum().backward()
    assert layer.d_raw.grad is not None, "d_raw.grad is None"
    assert layer.d_raw.grad.abs().mean() > 1e-8, "d_raw.grad is effectively zero"
```

### 4.2 Gradient Through Tau

Loss backpropagation must write a non-None, non-zero gradient into `tau_raw`.

```python
def test_gradient_flows_through_tau():
    """tau_raw.grad is not None and has nonzero mean after backward."""
    N = 64
    tau_mod = HeterogeneousTau(N=N, tau_mode="heterogeneous_learnable")
    beta = tau_mod()
    membrane = torch.randn(N, requires_grad=False)
    v = beta * membrane
    v.sum().backward()
    assert tau_mod.tau_raw.grad is not None, "tau_raw.grad is None"
    assert tau_mod.tau_raw.grad.abs().mean() > 1e-8, "tau_raw.grad is zero"
```

### 4.3 No Gradient When Fixed — Delays

When `delays_mode="fixed_random"`, `d_raw` must not appear in the computation
graph and its grad must remain None.

```python
def test_no_gradient_when_delays_fixed():
    """d_raw.grad stays None when delays_mode is fixed_random."""
    B, T, in_f, out_f, Td = 2, 5, 8, 8, 4
    layer = DelayLinear(in_f, out_f, max_delay=Td,
                        delays_mode="fixed_random")
    x = torch.randn(B, T, in_f)
    state = layer.reset_state(B, x.device, x.dtype)
    out, _ = layer(x, state)
    out.sum().backward()
    assert layer.d_raw.grad is None, \
        "d_raw should not receive gradient in fixed_random mode"
```

### 4.4 No Gradient When Tau Fixed

When `tau_mode="heterogeneous_fixed"`, `tau_raw` must not receive gradients.

```python
def test_no_gradient_when_tau_fixed():
    """tau_raw.grad stays None when tau_mode is heterogeneous_fixed."""
    N = 32
    tau_mod = HeterogeneousTau(N=N, tau_mode="heterogeneous_fixed")
    beta = tau_mod()
    beta.sum().backward()
    assert tau_mod.tau_raw.grad is None, \
        "tau_raw should not receive gradient in fixed mode"
```

### 4.5 Finite Difference Check — Delays

Autograd gradient of `d_raw` must match a numerical finite-difference estimate
within `rtol=1e-2`.

```python
def test_finite_difference_delays(tiny_delay_config):
    """Autograd gradient of d_raw agrees with finite difference (rtol=1e-2)."""
    layer = DelayLinear(**tiny_delay_config, delays_mode="learned")
    x = torch.randn(2, 5, tiny_delay_config["in_features"])
    state = layer.reset_state(2, x.device, x.dtype)

    def f(d):
        layer.d_raw.data.copy_(d)
        out, _ = layer(x, state)
        return out.sum()

    torch.autograd.gradcheck(f, layer.d_raw, eps=1e-4, rtol=1e-2, atol=1e-4)
```

---

## 5. Category 4: Sigma Schedule Tests

### 5.1 Monotone Decrease

Call `update_sigma(epoch)` for epochs 0 through 99 and assert each value is
less than or equal to the previous.

```python
def test_sigma_monotonically_decreasing():
    """sigma(epoch+1) <= sigma(epoch) for all epochs."""
    scheduler = SigmaScheduler(sigma_start=1.0, sigma_end=0.1, T_anneal=100)
    prev = scheduler.update_sigma(0)
    for epoch in range(1, 100):
        current = scheduler.update_sigma(epoch)
        assert current <= prev + 1e-9, \
            f"sigma increased at epoch {epoch}: {prev} -> {current}"
        prev = current
```

### 5.2 Convergence to sigma_end

After the annealing period, sigma must be within `1e-4` of `sigma_end`.

```python
def test_sigma_converges_to_end():
    """sigma reaches sigma_end after T_anneal epochs."""
    sigma_end = 0.05
    scheduler = SigmaScheduler(sigma_start=2.0, sigma_end=sigma_end, T_anneal=50)
    final = scheduler.update_sigma(200)
    assert abs(final - sigma_end) < 1e-4, \
        f"|sigma - sigma_end| = {abs(final - sigma_end)}"
```

### 5.3 Constant Schedule

When `schedule="constant"`, sigma must not change across any epoch.

```python
def test_sigma_constant_schedule():
    """sigma stays constant when schedule=constant."""
    sigma_val = 0.5
    scheduler = SigmaScheduler(sigma_start=sigma_val, sigma_end=sigma_val,
                               schedule="constant", T_anneal=100)
    values = [scheduler.update_sigma(e) for e in range(100)]
    assert all(abs(v - sigma_val) < 1e-9 for v in values), \
        f"sigma changed under constant schedule: {set(values)}"
```

### 5.4 Determinism

Calling `update_sigma` with the same epoch twice must return the same value.

```python
def test_sigma_update_is_deterministic():
    """update_sigma(epoch) returns identical value on repeated calls."""
    scheduler = SigmaScheduler(sigma_start=1.0, sigma_end=0.1, T_anneal=80)
    assert scheduler.update_sigma(42) == scheduler.update_sigma(42)
```

### 5.5 Checkpoint Round-Trip — Sigma

Save sigma state, load it, and assert the sigma value at the same epoch matches.

```python
def test_sigma_checkpoint_roundtrip(tmp_path):
    """Save/load sigma state -> same sigma at same epoch."""
    scheduler = SigmaScheduler(sigma_start=1.0, sigma_end=0.1, T_anneal=100)
    sigma_at_50 = scheduler.update_sigma(50)
    torch.save(scheduler.state_dict(), tmp_path / "sigma.pt")

    scheduler2 = SigmaScheduler(sigma_start=1.0, sigma_end=0.1, T_anneal=100)
    scheduler2.load_state_dict(torch.load(tmp_path / "sigma.pt"))
    assert abs(scheduler2.current_sigma - sigma_at_50) < 1e-9
```

---

## 6. Category 5: Checkpoint Round-Trip Tests

### 6.1 Identical Forward Output After Reload

Save `state_dict`, construct a fresh module, load the state, and confirm that
the forward pass produces byte-identical output.

```python
def test_checkpoint_identical_output(tmp_path):
    """Reload from state_dict -> identical forward output (atol=1e-6)."""
    B, T, in_f, out_f, Td = 2, 10, 8, 8, 4
    layer = DelayLinear(in_f, out_f, max_delay=Td)
    x = torch.randn(B, T, in_f)
    state = layer.reset_state(B, x.device, x.dtype)
    out1, _ = layer(x, state)

    torch.save(layer.state_dict(), tmp_path / "layer.pt")
    layer2 = DelayLinear(in_f, out_f, max_delay=Td)
    layer2.load_state_dict(torch.load(tmp_path / "layer.pt"))
    out2, _ = layer2(x, state)

    assert torch.allclose(out1, out2, atol=1e-6), \
        f"max diff after reload: {(out1 - out2).abs().max()}"
```

### 6.2 Required Keys Present in state_dict

The saved `state_dict` must contain all keys needed to fully reconstruct
the delay and tau state.

```python
def test_state_dict_required_keys():
    """state_dict contains d_raw, tau_raw, sigma buffer, sigma_epoch buffer."""
    layer = DelayLinear(8, 8, max_delay=4)
    sd = layer.state_dict()
    required = {"d_raw", "sigma", "sigma_epoch"}
    missing = required - set(sd.keys())
    assert not missing, f"Missing keys: {missing}"

    tau_mod = HeterogeneousTau(N=8, tau_mode="heterogeneous_learnable")
    sd_tau = tau_mod.state_dict()
    assert "tau_raw" in sd_tau, "tau_raw missing from HeterogeneousTau state_dict"
```

### 6.3 Training Resume Consistency

Training for 100 epochs continuously must produce the same final parameters as
stopping at epoch 50, saving, reloading, and continuing to epoch 100.

```python
def test_resume_training_consistency(tmp_path):
    """Checkpoint at epoch 50 -> resume -> matches continuous run (rtol=1e-4)."""
    # 1. Run 100 steps continuously; record final d_raw.
    # 2. Run 50 steps; save state_dict and optimizer state.
    # 3. Load into fresh layer; run 50 more steps.
    # 4. Assert torch.allclose on final d_raw (rtol=1e-4).
    ...  # Implementation follows standard PyTorch checkpoint pattern.
```

### 6.4 Cross-Device Checkpoint Load

Save on CUDA, load on CPU, and assert values match.

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cross_device_checkpoint(tmp_path):
    """Save on CUDA -> load on CPU -> values match (atol=1e-5)."""
    layer_gpu = DelayLinear(8, 8, max_delay=4).cuda()
    torch.save(layer_gpu.state_dict(), tmp_path / "gpu_layer.pt")

    layer_cpu = DelayLinear(8, 8, max_delay=4)
    layer_cpu.load_state_dict(
        torch.load(tmp_path / "gpu_layer.pt", map_location="cpu")
    )
    assert torch.allclose(
        layer_gpu.d_raw.cpu(), layer_cpu.d_raw, atol=1e-5
    )
```

---

## 7. Category 6: Integration Tests

### 7.1 DelayLinear + LIFNeuron End-to-End

A full forward pass through `DelayLinear` followed by `LIFNeuron` must produce
a valid spike tensor with values in `{0, 1}` (or approximately so under
surrogate gradients).

```python
def test_delay_linear_lif_end_to_end():
    """DelayLinear -> LIFNeuron produces valid spike output."""
    B, T, in_f, out_f, Td = 2, 20, 64, 64, 8
    delay_layer = DelayLinear(in_f, out_f, max_delay=Td)
    lif = LIFNeuron(out_f)
    x = torch.randn(B, T, in_f)
    d_state = delay_layer.reset_state(B, x.device, x.dtype)
    v = torch.zeros(B, out_f)
    spikes_out = []
    for t in range(T):
        x_t = x[:, t, :]           # (B, in_f)
        delayed, d_state = delay_layer.step(x_t, d_state)
        spikes, v = lif(delayed, v)
        spikes_out.append(spikes)
    out = torch.stack(spikes_out, dim=1)  # (B, T, out_f)
    assert out.shape == (B, T, out_f)
    assert torch.isfinite(out).all(), "spike output contains NaN/Inf"
```

### 7.2 HeterogeneousTau + LIFNeuron — Different Decay Rates

Different tau values per neuron must produce measurably different membrane
potential time constants. Test by checking variance in final membrane potentials.

```python
def test_heterogeneous_tau_produces_different_decays():
    """Different tau per neuron -> different membrane potential decays."""
    B, T, N = 1, 50, 32
    tau_mod = HeterogeneousTau(N=N, tau_init="heterogeneous_loguniform",
                               tau_mode="heterogeneous_learnable")
    beta = tau_mod().detach()
    v = torch.zeros(B, N)
    input_current = torch.ones(B, N) * 0.5
    for _ in range(T):
        v = beta * v + input_current
    assert v.std() > 1e-3, \
        "All neurons decayed identically — heterogeneous tau had no effect"
```

### 7.3 Both Features Together — No Shape Conflicts

Running `DelayLinear` and `HeterogeneousTau` together in a single `LIFLayer`
must not raise any shape errors for `(B, T, in_f=32, N=64)` input.

```python
def test_delays_and_hetero_tau_no_shape_conflicts():
    """DelayLinear + HeterogeneousTau together: no shape mismatch."""
    B, T, in_f, N, Td = 2, 15, 32, 64, 6
    delay_layer = DelayLinear(in_f, N, max_delay=Td)
    tau_mod = HeterogeneousTau(N=N, tau_mode="heterogeneous_learnable")
    lif = LIFNeuron(N, tau_module=tau_mod)

    x = torch.randn(B, T, in_f)
    d_state = delay_layer.reset_state(B, x.device, x.dtype)
    v = torch.zeros(B, N)
    for t in range(T):
        delayed, d_state = delay_layer.step(x[:, t], d_state)
        spikes, v = lif(delayed, v)
    assert spikes.shape == (B, N)
```

### 7.4 All Four Ablation Configs Produce Valid Output

Run all four combinations of (delays on/off) x (hetero-tau on/off) and
confirm each produces a finite output tensor.

```python
@pytest.mark.parametrize("use_delays,use_hetero_tau", [
    (False, False),  # baseline
    (True,  False),  # delays only
    (False, True),   # hetero tau only
    (True,  True),   # both
])
def test_ablation_configs_all_valid(use_delays, use_hetero_tau):
    """All 4 ablation configs produce finite output without errors."""
    B, T, N = 2, 10, 32
    model = SNNBlock(N, use_delays=use_delays, use_hetero_tau=use_hetero_tau)
    x = torch.randn(B, T, N)
    out = model(x)
    assert out.shape == (B, T, N), f"shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "non-finite values in output"
```

---

## 8. Memory Budget Tests

### Budget Formula

```
max_memory_mb = safety * (
    4 * B * Td * in_features           # spike history buffer
    + 4 * out_features * in_features   # weight matrix
    + 4 * out_features * in_features   # d_raw (per_synapse granularity)
    + 4 * B * T * out_features         # output activations
) / (1024 ** 2)
```

With `safety = 1.5`.

### Budget Test — Parametrized

```python
@pytest.mark.parametrize("B,T,out_f,in_f,Td", [
    (4,  20,  64,  64,  16),
    (8,  50, 128, 128,  32),
    (2,  10,  32,  32,   8),
])
def test_delay_memory_budget(B, T, out_f, in_f, Td):
    """Peak memory stays below 1.5x theoretical minimum."""
    safety = 1.5
    budget_bytes = safety * (
        4 * B * Td * in_f
        + 4 * out_f * in_f
        + 4 * out_f * in_f
        + 4 * B * T * out_f
    )
    budget_mb = budget_bytes / (1024 ** 2)

    layer = DelayLinear(in_f, out_f, max_delay=Td)
    x = torch.randn(B, T, in_f)
    state = layer.reset_state(B, x.device, x.dtype)
    mem_before = _get_process_memory_mb()
    _, _ = layer(x, state)
    mem_after = _get_process_memory_mb()
    used_mb = mem_after - mem_before

    assert used_mb <= budget_mb, \
        f"used {used_mb:.2f} MB > budget {budget_mb:.2f} MB " \
        f"for (B={B}, T={T}, out={out_f}, in={in_f}, Td={Td})"
```

---

## 9. Performance Regression Tests

### 9.1 Timing Helper

```python
import time

def _time_forward(layer, x, state, n=50):
    """Return mean forward pass time in milliseconds over n iterations."""
    for _ in range(5):          # warmup
        if state is not None:
            layer(x, state)
        else:
            layer(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        if state is not None:
            layer(x, state)
        else:
            layer(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n * 1000
```

### 9.2 Delay Overhead Factor

Overhead from `DelayLinear` relative to `nn.Linear` must stay below `2.0x`.

```python
def test_delay_overhead_factor():
    """DelayLinear forward time < 2.0x nn.Linear baseline."""
    B, T, in_f, out_f, Td = 4, 20, 128, 128, 16
    baseline = nn.Linear(in_f, out_f)
    delayed  = DelayLinear(in_f, out_f, max_delay=Td)
    x_flat = torch.randn(B * T, in_f)
    x_seq  = torch.randn(B, T, in_f)
    state  = delayed.reset_state(B, x_seq.device, x_seq.dtype)

    t_base  = _time_forward(baseline, x_flat, state=None)
    t_delay = _time_forward(delayed, x_seq, state=state)
    overhead = t_delay / t_base

    print(f"[PERF] baseline={t_base:.3f}ms delay={t_delay:.3f}ms factor={overhead:.2f}")
    assert overhead < 2.0, f"delay overhead {overhead:.2f}x exceeds 2.0x limit"
```

### 9.3 Tau Overhead Factor

Computing beta from `HeterogeneousTau` must add less than `1.2x` overhead
compared to a fixed scalar beta.

```python
def test_tau_overhead_factor():
    """HeterogeneousTau() call overhead < 1.2x fixed scalar beta."""
    N = 512
    tau_mod = HeterogeneousTau(N=N, tau_mode="heterogeneous_learnable")
    n_iter = 1000

    start = time.perf_counter()
    for _ in range(n_iter):
        _ = torch.tensor(0.9)   # fixed scalar baseline
    t_fixed = (time.perf_counter() - start) / n_iter * 1e6

    start = time.perf_counter()
    for _ in range(n_iter):
        _ = tau_mod()
    t_hetero = (time.perf_counter() - start) / n_iter * 1e6

    overhead = t_hetero / max(t_fixed, 1e-9)
    print(f"[PERF] fixed={t_fixed:.2f}us hetero={t_hetero:.2f}us factor={overhead:.2f}")
    assert overhead < 1.2, f"tau overhead {overhead:.2f}x exceeds 1.2x limit"
```

---

## 10. Pytest Configuration

### Directory Layout

```
tests/
├── conftest.py                    # Shared fixtures
├── test_delay_module.py           # Categories 1, 3, 4 for delays
├── test_heterogeneous_tau.py      # Categories 2, 3 for tau
├── test_delay_tau_checkpoint.py   # Category 5
└── test_delay_tau_integration.py  # Category 6
```

### conftest.py — Shared Fixtures

```python
import pytest
import torch


@pytest.fixture
def tiny_delay_config():
    """Minimal DelayLinear config for fast unit tests."""
    return dict(in_features=8, out_features=8, max_delay=4,
                granularity="per_synapse", delays_mode="learned", sigma=0.5)


@pytest.fixture
def tiny_tau_config():
    """Minimal HeterogeneousTau config for fast unit tests."""
    return dict(N=8, granularity="per_neuron",
                tau_mode="heterogeneous_learnable",
                tau_init="heterogeneous_gamma", dt=1e-3)


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    """Parametrize tests over available devices."""
    return torch.device(request.param)


@pytest.fixture(params=["per_synapse", "per_output", "per_input", "per_block"])
def granularity(request):
    """Parametrize tests over all delay granularity modes."""
    return request.param


@pytest.fixture(params=[
    "homogeneous",
    "heterogeneous_gamma",
    "heterogeneous_loguniform",
    "preset_bank",
])
def tau_init(request):
    """Parametrize tests over all tau initialization strategies."""
    return request.param


def _get_process_memory_mb():
    """Return current process RSS in MB (cross-platform)."""
    import psutil
    import os
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _get_peak_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return _get_process_memory_mb()


def _delay_memory_budget_mb(B, T, out_f, in_f, Td, safety=1.5):
    bytes_ = safety * (
        4 * B * Td * in_f
        + 4 * out_f * in_f
        + 4 * out_f * in_f
        + 4 * B * T * out_f
    )
    return bytes_ / (1024 ** 2)
```

---

## 11. "Done When" Checklist

All 26 items below must pass in CI before the feature is considered complete.

### Delay Module (10 items)

- [ ] 1. DelayLinear forward shapes correct for all granularity modes
- [ ] 2. DelayConv1d forward shapes correct (spatial dims preserved with padding=1)
- [ ] 3. Gaussian interpolation bins sum to ~1.0 (atol=1e-4) for each delay position
- [ ] 4. Discretization output matches smoothed output (atol=1e-3) when sigma approaches zero
- [ ] 5. Gradient flows through d_raw: grad is not None and mean > 1e-8
- [ ] 6. No gradient when delays_mode="fixed_random": d_raw.grad is None
- [ ] 7. Sigma schedule decreases monotonically across all annealing epochs
- [ ] 8. Sigma checkpoint round-trip: current_sigma matches exactly after reload
- [ ] 9. Memory stays below 1.5x theoretical budget for all parametrized configs
- [ ] 10. Forward overhead < 2.0x nn.Linear baseline on same hardware

### Tau Module (8 items)

- [ ] 11. Beta strictly in (0, 1) for all four initialization strategies
- [ ] 12. Tau strictly positive for all four initialization strategies
- [ ] 13. Gradient flows through tau_raw when tau_mode="heterogeneous_learnable"
- [ ] 14. No gradient when tau_mode="heterogeneous_fixed": tau_raw.grad is None
- [ ] 15. All four init strategies (homogeneous, gamma, loguniform, preset_bank) produce finite values
- [ ] 16. Per-neuron, per-channel, per-layer granularity produce correct tau_raw shapes
- [ ] 17. Forward overhead < 1.2x fixed scalar beta baseline
- [ ] 18. ConvSNN spatial broadcast: per-channel tau broadcasts correctly over (C, H, W)

### Integration (8 items)

- [ ] 19. DelayLinear + LIFNeuron end-to-end: output shape (B, T, N), all finite
- [ ] 20. HeterogeneousTau + LIFNeuron: measurably different decay rates (std > 1e-3)
- [ ] 21. Both features together: no shape mismatch for (B, T, in_f=32, N=64) input
- [ ] 22. snn_unroll with delay module: unroll over T timesteps accumulates state correctly
- [ ] 23. All four ablation configs (baseline/delays/tau/both) produce valid finite output
- [ ] 24. state_dict contains all required keys: d_raw, tau_raw, sigma, sigma_epoch
- [ ] 25. Checkpoint reload produces identical forward output (atol=1e-6)
- [ ] 26. Cross-device checkpoint load (CUDA save, CPU load) produces matching values (atol=1e-5)
