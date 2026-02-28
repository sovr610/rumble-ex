# Testing Matrix: What "Done When" Means as Concrete Tests

This document defines the concrete test specifications that must pass before the spiking neural network core is considered production-ready. Each category maps directly to a test file in `tests/test_spiking_core/`. Tests are written in imperative form — implement exactly as described.

---

## Category 1: Gradient Checks

### 1a) Gradient Presence — Sanity Check

For each neuron variant (`LIF`, `AdaptiveLIF`, `RecurrentLIF`, `AdvancedLIF`):

1. Create a tiny network with `input=8`, `hidden=4`, `output=2`.
2. Forward a random batch (e.g. `batch=4`) through `T=10` timesteps.
3. Compute `loss = output_spikes.sum()`.
4. Call `loss.backward()`.
5. Assert: `W_in.grad` is not `None` and `W_in.grad.abs().sum() > 0`.
6. Assert: if the threshold is learnable (`v_th` registered as a parameter), `v_th.grad.abs().sum() > 0`.
7. Assert: if `beta` is learnable, `beta.grad.abs().sum() > 0`.
8. Assert: if recurrent weights `W_rec` are present (`RecurrentLIF`, `AdvancedLIF`), `W_rec.grad.abs().sum() > 0`.

This test must run independently for each of the four variants. Failure on any variant is a blocking issue.

---

### 1b) Surrogate Swap Test

Run the following procedure three times, once per surrogate (`ATan`, `FastSigmoid`, `StraightThrough`):

1. Instantiate the same tiny network (identical architecture and weight init, fixed seed).
2. Forward the same random batch through `T=10` timesteps.
3. Compute `loss = output_spikes.sum()`.
4. Call `loss.backward()`.
5. Record `grad_norm = W_in.grad.norm().item()`.

After all three runs:

- Assert: not all three `grad_norm` values are equal (proves the surrogate function is actually wired into the backward pass and is not a dead code path).
- Assert: all three `grad_norm` values are finite and non-zero.

---

### 1c) Finite Difference Spot-Check

Use a tiny network with `input=4`, `hidden=2`, `output=1`, `T=3`:

1. Fix a random input batch and a random seed.
2. Select 5 weight scalar entries at random from `W_in`.
3. For each selected scalar `w_i`:
   - Compute `f(w_i + ε)` and `f(w_i - ε)` where `ε = 1e-4` and `f` is the scalar loss (`output_spikes.sum()`).
   - Numeric gradient: `g_numeric = (f(w_i + ε) - f(w_i - ε)) / (2 * ε)`.
   - Autograd gradient: run forward + backward normally, extract `W_in.grad` at the same scalar position.
   - Relative error: `|g_numeric - g_autograd| / (|g_autograd| + 1e-8)`.
   - Assert: `relative_error < 0.05`.

**Important note on surrogate mismatch**: The forward pass uses the Heaviside step function; the backward pass uses the surrogate approximation. Finite difference measures the true forward Heaviside gradient (which is zero almost everywhere), so agreement is not algebraically guaranteed. This test is valid only around operating points where the surrogate closely approximates the finite difference. Use the `ATan` surrogate for this test and interpret failure as a debugging signal, not a pass/fail threshold for all surrogates.

---

### 1d) Gradient Magnitude Test

Build a 3-layer network (`input=32`, `hidden=[64, 32]`, `output=16`, `T=10`):

1. Forward a random batch.
2. Compute loss and backward.
3. Collect gradient norms per layer: `[W_layer1.grad.norm(), W_layer2.grad.norm(), W_layer3.grad.norm()]`.

**With max-gradient normalization enabled** (clip gradients to norm=1 before stepping):

- Assert: the ratio `grad_norm_layer1 / grad_norm_layer3 < 3.0` (gradient does not explode going backward through the network).

**Without normalization** (raw gradients):

- Record and log the growth factor as a diagnostic value. Do not assert a bound here; instead, document the actual ratio in the test output. This establishes a baseline for regression tracking.

---

## Category 2: Convergence Checks (Dev Mode)

Mark all tests in this category with `@pytest.mark.slow`. These tests are excluded from the default `pytest` run and require explicit invocation with `-m slow`.

---

### 2a) MNIST Rate-Coded Classification

1. Instantiate `SNNCore(784, [256, 128], 10, T=25)` with the `ATan` surrogate.
2. Load MNIST training and test sets (standard torchvision, 60k/10k split).
3. Encode inputs as rate-coded Poisson spike trains over `T=25` timesteps.
4. Train for 5 epochs with `Adam(lr=1e-3)`, batch size 256.
5. At the end of each epoch, record training loss and test accuracy.
6. Assert: final test accuracy `> 0.90` (90%).
7. Assert: training loss at epoch 5 is strictly less than training loss at epoch 1 (loss decreased overall).
8. Assert: no `NaN` in loss at any epoch.

Rate-coded MNIST with surrogate gradients routinely achieves 95%+ in the literature. A result below 90% indicates a bug in the surrogate wiring, reset logic, or loss computation.

---

### 2b) CIFAR-10 Dev Mode

1. Instantiate a small `ConvSNN` with 2 convolutional blocks followed by a spiking linear classifier.
2. Load CIFAR-10 (standard 50k/10k split).
3. Train for 10 epochs with `Adam(lr=1e-3)`, batch size 128.
4. Record test accuracy at each epoch.
5. Assert: final test accuracy `> 0.40` (40%).
6. Assert: accuracy at epoch 10 is strictly greater than accuracy at epoch 1 (improving, not stuck).
7. Assert: no `NaN` in loss at any epoch.

The 40% threshold is deliberately low — SOTA is above 70% with full hyperparameter tuning. The goal here is proving the training loop is functional, not achieving SOTA.

---

### 2c) Training Stability Over Long Runs

1. Train the MNIST setup from 2a for 20 epochs (same architecture and optimizer).
2. After each epoch, scan all model parameters and gradients for `NaN` and `Inf`.
3. At the end of the 20-epoch run, compute the mean firing rate per layer averaged over the test set.
4. Assert: no `NaN` or `Inf` in any parameter at any epoch.
5. Assert: mean firing rate per layer `> 0.05` (no dead layers — more than 5% neurons are active on average).
6. Assert: mean firing rate per layer `< 0.90` (no saturated layers — less than 90% neurons are continuously spiking).

A dead-neuron rate below 5% indicates the threshold is too high or the surrogate gradient has vanished. A saturation rate above 90% indicates the threshold is too low or the learning rate is too aggressive.

---

## Category 3: State Reset and Carry Tests

---

### 3a) Reset Determinism

For each neuron variant:

1. Fix a random seed. Create an input batch `x` with shape `(batch=4, T=10, input_dim=8)`.
2. Forward the batch, resetting state before the call. Save output `y1`.
3. Reset state explicitly.
4. Forward the same batch `x` again with a fresh reset. Save output `y2`.
5. Assert: `torch.equal(y1, y2)` on CPU (bitwise identical).
6. On CUDA: assert `torch.allclose(y1, y2, atol=1e-6)` (minor floating-point accumulation allowed).

This test enforces that `reset_state()` is fully deterministic and leaves no residual state between calls.

---

### 3b) Carry Changes Output

For each neuron variant:

1. Create input batch `x`.
2. Forward with fresh reset → save `y1`.
3. Forward `x` again **without** resetting (carry state from run 1) → save `y2`.
4. Reset state and forward `x` again → save `y3`.
5. Assert: `torch.equal(y1, y3)` (reset recovers identical output).
6. Assert: `not torch.equal(y1, y2)` (carried state changes the output).

If assertion 6 fails, the state carry mechanism is not connected to the forward computation.

---

### 3c) Truncated BPTT Numerical Match

1. Build a network with `input=16`, `hidden=32`, `output=8`.
2. Fix a random batch `x` with `T=20`.

**Full BPTT pass:**

3. Forward the entire batch in one call (`T=20`) with reset before. Save spike outputs `spikes_full`.

**Truncated BPTT pass (chunk size 10):**

4. Reset state. Forward timesteps `0:10`, do **not** detach state. Forward timesteps `10:20` with carried state. Concatenate spike outputs → `spikes_chunked`.

5. Assert: `torch.equal(spikes_full, spikes_chunked)` (chunking does not alter the forward computation).
6. Compute gradient norms from a full-BPTT backward pass → `grad_norm_full`.
7. Compute gradient norms from a truncated backward pass (backward only through chunk 2) → `grad_norm_trunc`.
8. Assert: `grad_norm_trunc <= grad_norm_full` (truncation reduces or preserves gradient magnitude; it never amplifies it).

---

### 3d) Detach Cuts Gradient Flow

1. Build a two-chunk setup: chunk 1 is timesteps `0:10`, chunk 2 is timesteps `10:20`.
2. Create a learnable input projection `W_proj_1` applied only to chunk 1 and `W_proj_2` applied only to chunk 2.
3. Forward chunk 1 through the SNN. Detach the resulting state (`state = state.detach()`).
4. Forward chunk 2 through the SNN with the detached state.
5. Compute loss from chunk 2 outputs only and call `backward()`.
6. Assert: `W_proj_1.grad is None` or `W_proj_1.grad.abs().sum() == 0` (detach blocked the gradient from flowing into chunk 1).
7. Assert: `W_proj_2.grad.abs().sum() > 0` (chunk 2 input trains normally).

---

### 3e) Partial Reset

1. Instantiate `AdaptiveLIF` which carries state `(v, adaptation)`.
2. Forward a batch to populate both state components with non-zero values.
3. Confirm: `state.v.abs().sum() > 0` and `state.adaptation.abs().sum() > 0`.
4. Call a partial reset that zeros only `v`, leaving `adaptation` untouched.
5. Assert: `state.v.abs().sum() == 0`.
6. Assert: `state.adaptation.abs().sum() > 0` (adaptation component preserved).

If `AdaptiveLIF` does not expose a partial reset API, add one. The API signature should be `reset_state(components=['v'])` where `components` is a list of state variable names to zero.

---

## Category 4: CPU vs CUDA Parity

Skip all tests in this category if `torch.cuda.is_available()` returns `False`. Use `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")`.

---

### 4a) Output Parity

For each neuron variant:

1. Set `torch.manual_seed(42)` and `torch.use_deterministic_algorithms(True)`.
2. Forward a fixed batch on CPU → save `spikes_cpu`, `membrane_cpu`.
3. Move model and batch to CUDA. Forward the same batch → save `spikes_cuda`, `membrane_cuda`.
4. Assert: `torch.equal(spikes_cpu, spikes_cuda.cpu())` (binary spikes must match exactly; the Heaviside is deterministic).
5. Assert: `torch.allclose(membrane_cpu, membrane_cuda.cpu(), atol=1e-5, rtol=1e-4)` (membrane values may differ slightly due to floating-point accumulation order).

---

### 4b) Gradient Parity

1. Forward + backward on CPU with fixed seed. Record `{param_name: grad.norm().item()}` for all parameters.
2. Forward + backward on CUDA with same fixed seed. Record the same.
3. For each parameter, assert: `abs(grad_norm_cpu - grad_norm_cuda) < atol=1e-4 + rtol=1e-3 * grad_norm_cpu`.

---

### 4c) Mixed Precision Parity (CUDA Only)

1. Forward in `fp32` on CUDA → save `spikes_fp32`.
2. Cast model weights and inputs to `bfloat16`. Forward with `bfloat16` activations but accumulate membrane potential in `fp32` (use `autocast` with `dtype=torch.bfloat16` and `fp32` state accumulation via explicit cast before threshold comparison).
3. Save `spikes_bf16`.
4. Assert: `torch.equal(spikes_fp32, spikes_bf16)` in most cases (binary spike outputs should be identical because threshold comparison is done in fp32).
5. Assert: `torch.allclose(membrane_fp32, membrane_bf16, atol=1e-2)` (bf16 introduces ~1% error in membrane traces, which is acceptable).

If the model does not yet implement fp32 state accumulation in mixed precision mode, add it before writing this test. The pattern is: cast input to bf16, cast state to fp32 for the integration step, apply threshold in fp32, spike output is binary.

---

## Category 5: Debug Surface Tests

These tests verify that the diagnostic instrumentation built into the network is accurate and useful.

---

### 5a) Firing Rate Probe

1. Forward a batch through a multi-layer network with `return_details=True` (or equivalent mechanism that exposes per-layer spike statistics).
2. Extract `firing_rate[layer]` for each layer — the fraction of spikes that fired, averaged over batch and time.
3. Assert: `0.0 < firing_rate[layer] < 1.0` for all layers.
4. Assert: `0.01 <= firing_rate[layer] <= 0.50` for typical inputs (rates outside this range indicate pathological behavior, not necessarily a bug, but log a warning and document).

---

### 5b) Membrane Statistics

From the same `return_details=True` forward pass:

1. Extract `membrane_mean[layer]`, `membrane_var[layer]`, `membrane_max[layer]` for each layer.
2. Assert: no `NaN` or `Inf` in any of these values.
3. Assert: `membrane_max[layer] < 10 * v_th` for all layers (membrane is not running away to arbitrarily large values).
4. Assert: `membrane_var[layer] > 0` (there is actual variation in membrane potentials; all-identical membranes indicate a dead or constant-input pathway).

---

### 5c) Spike Sparsity

1. Forward a random batch through the network.
2. Compute per-layer sparsity: `sparsity[layer] = (spikes[layer] == 0).float().mean()`.
3. Assert: `sparsity[layer] > 0.50` for all layers (spikes should be sparse — the SNN design assumption).
4. Assert: `sparsity[layer] < 0.99` for all layers (not all neurons are dead).

A layer with `sparsity > 0.99` is a dead layer. A layer with `sparsity < 0.50` has a firing rate problem (threshold too low, or input too large).

---

### 5d) Gradient Statistics

After a backward pass:

1. Collect gradient norms for `W_in`, `W_rec` (if present), and `v_th` (if learnable) for each layer.
2. Assert: no `NaN` in any gradient.
3. Assert: all gradient norms are finite (`torch.isfinite(grad.norm())`).
4. Assert: all gradient norms are `> 0` (no complete gradient vanishing).

Log the gradient norms as part of the test output (use `print` or `capsys`) so they are visible in `pytest -s` runs. This creates a reference baseline for future regression tracking.

---

## Category 6: Numerical Stability Tests

---

### 6a) Long Unroll Stability

1. Instantiate a 2-layer network with `input=32`, `hidden=64`, `output=16`.
2. Forward a batch through `T=200` timesteps.
3. Assert: no `NaN` in the spike output tensor.
4. Assert: `membrane_max < 100 * v_th` at every timestep (membrane does not diverge).
5. Call `backward()` on `loss = output_spikes.sum()`.
6. Assert: all parameter gradients are finite after the full 200-step unroll.

If gradient norms explode at `T=200`, gradient clipping must be applied before this test can pass. Document the required clip value.

---

### 6b) Beta Boundary Test

Run forward + backward for each of the following `β` values using a tiny network:

- `β = 0.999`: Near-unity leak. State decays very slowly. Assert: forward completes, no `NaN`.
- `β = 0.001`: Near-zero leak. State decays almost instantly each timestep. Assert: forward completes, no `NaN`.
- `β = 0.0`: Exact zero leak. No membrane memory; purely feedforward within each timestep. Assert: forward completes, output is a function of current input only (carry state test: assert `y_carry == y_reset` for this case since β=0 erases history).

For all three: assert that the backward pass completes and gradients are finite.

---

### 6c) Threshold at Zero

1. Set `v_th = 0.0`.
2. Forward any positive-input batch.
3. Assert: no `NaN` in outputs.
4. Assert: no division-by-zero or `log(0)` in any intermediate computation (inspect by scanning for `NaN` in membrane before the threshold step).
5. Assert: spike output is all-ones or near-all-ones (every neuron with positive membrane fires immediately when `v_th=0`).

---

### 6d) Very Large and Very Small Inputs

**Large input (`x = 100.0`):**

1. Create a constant input tensor filled with `100.0`.
2. Forward through the network.
3. Assert: no `Inf` in membrane potential (confirm the integration `v = β * v + x` with large `x` does not produce `Inf` within the first few timesteps).
4. Assert: spike output is valid binary tensor.

**Small input (`x = 1e-6`):**

1. Create a constant input tensor filled with `1e-6`.
2. Forward through the network.
3. Assert: no `NaN` in membrane.
4. Compute loss and backward.
5. Assert: `W_in.grad.abs().sum() > 0` (the surrogate still provides gradient signal even when inputs are tiny and spikes are rare).

---

## "Done When" Checklist

All items in this checklist must pass before the spiking core is considered production-ready. No partial credit.

- [ ] All 4 neuron variants (`LIF`, `AdaptiveLIF`, `RecurrentLIF`, `AdvancedLIF`) produce non-zero gradients for all learnable parameters (1a)
- [ ] Surrogate swap (`ATan`, `FastSigmoid`, `STE`) produces measurably different gradient norms (1b)
- [ ] Finite difference matches autograd within 5% relative error for tiny networks using `ATan` surrogate (1c)
- [ ] Gradient magnitude growth from output to input layer is less than 3x when normalization is enabled (1d)
- [ ] MNIST rate-coded classification converges above 90% test accuracy within 5 epochs (2a)
- [ ] CIFAR-10 dev mode achieves above 40% and improves monotonically over 10 epochs (2b)
- [ ] No `NaN` or `Inf` in any parameter across a 20-epoch MNIST training run (2c)
- [ ] No dead layers (mean firing rate > 5%) and no saturated layers (mean firing rate < 90%) after 20 epochs (2c)
- [ ] Reset state produces bitwise-identical outputs for the same input on CPU (3a)
- [ ] Carried state produces measurably different outputs than reset state (3b)
- [ ] Truncated BPTT forward pass matches full BPTT forward pass exactly (3c)
- [ ] Truncated BPTT gradient norms are less than or equal to full BPTT gradient norms (3c)
- [ ] `detach()` provably blocks gradient flow from chunk 2 backward into chunk 1 inputs (3d)
- [ ] Partial reset API on `AdaptiveLIF` zeros only the specified state component (3e)
- [ ] CPU and CUDA spike outputs match exactly; membrane values match within `atol=1e-5` (4a)
- [ ] CPU and CUDA gradient norms match within `atol=1e-4, rtol=1e-3` (4b)
- [ ] Mixed precision (`bf16` activations + `fp32` state) produces valid binary spikes identical to `fp32` baseline (4c)
- [ ] Debug surfaces report per-layer firing rates in `(0, 1)` with no `NaN` (5a)
- [ ] Membrane statistics (mean, var, max) are finite and `max < 10 * v_th` (5b)
- [ ] Spike sparsity is between 50% and 99% for all layers (5c)
- [ ] All parameter gradients are finite and non-zero after backward (5d)
- [ ] `T=200` unroll completes with no `NaN` or `Inf` in outputs or gradients (6a)
- [ ] `β` boundary values `{0.0, 0.001, 0.999}` all complete forward and backward without error (6b)
- [ ] `v_th=0.0` produces no `NaN` and no division-by-zero (6c)
- [ ] `x=100.0` produces no `Inf`; `x=1e-6` still produces non-zero gradients (6d)

---

## Pytest Organization

```
tests/test_spiking_core/
├── test_gradient_checks.py       # Category 1
├── test_convergence.py           # Category 2 (slow, @pytest.mark.slow)
├── test_state_management.py      # Category 3
├── test_device_parity.py         # Category 4 (skip if no CUDA)
├── test_debug_surfaces.py        # Category 5
├── test_numerical_stability.py   # Category 6
└── conftest.py                   # Shared fixtures
```

Run only fast tests (excludes Category 2):

```bash
pytest tests/test_spiking_core/ -v
```

Run all tests including slow convergence checks:

```bash
pytest tests/test_spiking_core/ -v -m "slow or not slow"
```

Run only CUDA parity tests:

```bash
pytest tests/test_spiking_core/test_device_parity.py -v
```

---

## Shared Fixtures (conftest.py)

```python
import pytest
import torch
from brain_ai.core.neurons import LIF, AdaptiveLIF, RecurrentLIF, AdvancedLIF
from brain_ai.core.surrogates import ATan, FastSigmoid, StraightThrough
from brain_ai.core.snn import SNNCore


@pytest.fixture(params=["lif", "adaptive_lif", "recurrent_lif", "advanced_lif"])
def neuron_variant(request):
    """Parametrize over all four neuron types with small dimensions."""
    input_dim = 8
    hidden_dim = 4
    variants = {
        "lif": lambda: LIF(input_dim, hidden_dim),
        "adaptive_lif": lambda: AdaptiveLIF(input_dim, hidden_dim),
        "recurrent_lif": lambda: RecurrentLIF(input_dim, hidden_dim),
        "advanced_lif": lambda: AdvancedLIF(input_dim, hidden_dim),
    }
    return variants[request.param]()


@pytest.fixture(params=["atan", "fast_sigmoid", "straight_through"])
def surrogate(request):
    """Parametrize over all three surrogate gradient functions."""
    surrogates = {
        "atan": ATan(),
        "fast_sigmoid": FastSigmoid(),
        "straight_through": StraightThrough(),
    }
    return surrogates[request.param]


@pytest.fixture
def tiny_snn():
    """Minimal SNNCore for fast unit tests: input=8, hidden=[4], output=2, T=5."""
    return SNNCore(input_dim=8, hidden_dims=[4], output_dim=2, T=5)


@pytest.fixture
def random_batch():
    """Fixed random batch for reproducible tests: (batch=4, T=10, input_dim=8)."""
    torch.manual_seed(42)
    return torch.randn(4, 10, 8)


@pytest.fixture
def device():
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## Implementation Notes for Test Authors

**On surrogate finite difference mismatch**: Do not attempt to make the finite difference test pass with `StraightThrough`. STE has zero derivative almost everywhere by definition. Use `ATan` only for Category 1c and document this constraint in the test file header.

**On CUDA determinism**: `torch.use_deterministic_algorithms(True)` may cause errors with certain CUDA operations. Catch `RuntimeError` around the CUDA forward pass and skip the determinism flag if the operation does not support it, then fall back to tolerance-based comparison only.

**On dead neuron detection**: A "dead neuron" is one that never fires across an entire epoch. The Category 2c test measures this per-layer, not per-neuron, to keep the test tractable. A layer is considered dead if its mean firing rate across all neurons and all test inputs is below 5%.

**On truncated BPTT forward equivalence**: The assertion that truncated forward matches full forward is strictly about the spike tensor values, not about memory usage or computation graph structure. The two are mathematically equivalent because detaching state only affects the backward pass, not the recurrent integration formula.

**On the 14-item checklist**: The checklist is a gate, not a goal. All 14 items must pass before the spiking core is declared ready for integration into the full brain-ai system pipeline.
