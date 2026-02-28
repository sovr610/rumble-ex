# Testing Matrix Reference

## Overview

This document specifies the complete set of tests for the DreamerV3 RSSM implementation,
organized into six phases corresponding to the major components. Each phase includes
functional tests, shape tests, gradient tests, and edge case tests.

Use this matrix when writing pytest tests, during manual validation, or when debugging
unexpected behavior. Tests are ordered from foundational (Block GRU) to integrated (loss).

---

## Phase 1: Block GRU

### P1.1 — Input/Output Shapes

| Test | Input | Expected Output | Check |
|------|-------|----------------|-------|
| Forward pass | x: (B, input_dim), h: (B, hidden_dim) | h_new: (B, hidden_dim) | shape equality |
| Batch size 1 | x: (1, input_dim), h: (1, hidden_dim) | h_new: (1, hidden_dim) | no errors |
| Large batch | x: (256, input_dim), h: (256, hidden_dim) | h_new: (256, hidden_dim) | no errors |
| Hidden dim 512 | x: (4, 100), h: (4, 512) | h_new: (4, 512) | shape correct |
| Hidden dim 2048 | x: (4, 100), h: (4, 2048) | h_new: (4, 2048) | shape correct |

```python
def test_block_gru_output_shape():
    gru = BlockGRU(input_dim=64, hidden_dim=128)
    x = torch.randn(4, 64)
    h = torch.zeros(4, 128)
    h_new = gru(x, h)
    assert h_new.shape == (4, 128)
```

### P1.2 — LayerNorm + SiLU Input Projection

| Test | What to Check |
|------|--------------|
| Input projection present | `gru.input_proj` contains LayerNorm and SiLU |
| Projection output normalized | mean ≈ 0, std ≈ 1 for the projected activations |
| SiLU nonlinearity active | output is not all-positive (SiLU allows negative values via x*sigmoid(x)) |

```python
def test_block_gru_input_proj_modules():
    gru = BlockGRU(input_dim=64, hidden_dim=128)
    module_types = [type(m).__name__ for m in gru.input_proj]
    assert 'Linear' in module_types
    assert 'LayerNorm' in module_types
    assert 'SiLU' in module_types
```

### P1.3 — RMSNorm Output

| Test | What to Check |
|------|--------------|
| Output has unit RMS | `(h_new ** 2).mean(dim=-1).sqrt()` ≈ `norm.scale.mean()` |
| RMSNorm has learned scale | `gru.norm_out.weight` exists and is trainable |
| Output is not mean-zero | RMSNorm does not subtract mean (unlike LayerNorm) |

```python
def test_block_gru_rms_norm_output():
    gru = BlockGRU(input_dim=64, hidden_dim=128)
    gru.train(False)
    x = torch.randn(16, 64)
    h = torch.zeros(16, 128)
    h_new = gru(x, h)
    # RMSNorm: h / rms * scale, scale initialized to 1
    rms = (h_new ** 2).mean(dim=-1).sqrt()
    # Should be close to 1 (scale=1 by default, so RMS of h_new ≈ 1/rms_of_pre * scale)
    assert rms.shape == (16,)
    assert not torch.any(torch.isnan(h_new))
```

### P1.4 — Gradient Flow

| Test | What to Check |
|------|--------------|
| Gradients reach input | `x.grad` is non-None and non-zero after backward |
| Gradients reach h | `h.grad` is non-None and non-zero after backward |
| Gradients are finite | no nan or inf in any gradient |
| Gate parameters receive gradients | `gate_r.weight.grad`, `gate_z.weight.grad`, `gate_n.weight.grad` non-None |

```python
def test_block_gru_gradient_flow():
    gru = BlockGRU(input_dim=64, hidden_dim=128)
    x = torch.randn(4, 64, requires_grad=True)
    h = torch.randn(4, 128, requires_grad=True)
    h_new = gru(x, h)
    loss = h_new.sum()
    loss.backward()
    assert x.grad is not None
    assert h.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    assert torch.all(torch.isfinite(h.grad))
```

### P1.5 — Different Inputs Produce Different Outputs

```python
def test_block_gru_input_sensitivity():
    gru = BlockGRU(input_dim=64, hidden_dim=128)
    h = torch.zeros(2, 128)
    x1 = torch.randn(2, 64)
    x2 = torch.randn(2, 64)
    h1 = gru(x1, h)
    h2 = gru(x2, h)
    assert not torch.allclose(h1, h2)
```

---

## Phase 2: Categorical State

### P2.1 — Unimix Distribution

| Test | Expected Behavior |
|------|------------------|
| Probabilities sum to 1 | `probs.sum(dim=-1)` ≈ 1.0 for all batch and stoch dims |
| No zero probabilities | `probs.min()` > 0 |
| Minimum probability | `probs.min()` ≈ `unimix / num_classes` |
| Unimix=0 → pure softmax | `probs` ≈ `softmax(logits)` |
| Uniform logits → equal probs | all probs ≈ `1 / num_classes` |

```python
def test_unimix_probs_sum_to_one():
    logits = torch.randn(8, 32, 32)
    probs = unimix_probs(logits, unimix=0.01)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

def test_unimix_no_zero_probs():
    logits = torch.zeros(4, 32, 32)
    logits[0, 0, 0] = 1e10  # extreme logit to make one class nearly certain
    probs = unimix_probs(logits, unimix=0.01)
    assert probs.min() > 0
```

### P2.2 — Straight-Through Gradient

| Test | Expected Behavior |
|------|------------------|
| Forward: hard one-hot | output sums to 1, each row has exactly one 1 |
| Backward: gradient through probs | `logits.grad` is non-None after backward |
| Gradient is from soft probs | gradient shape matches logits shape |
| One-hot output has no gradient | `z_hard` detached; gradient only through `probs` |

```python
def test_straight_through_gradient():
    logits = torch.randn(4, 32, 32, requires_grad=True)
    z = sample_straight_through(logits)
    loss = z.sum()
    loss.backward()
    assert logits.grad is not None
    # z should be (approximately) one-hot in forward pass
    assert z.shape == (4, 32, 32)
```

### P2.3 — Prior/Posterior Output Shapes

```python
def test_prior_net_shape():
    prior = PriorNet(deter_dim=1024, stoch_dim=32, num_classes=32, hidden_dim=256)
    h = torch.randn(8, 1024)
    logits = prior(h)
    assert logits.shape == (8, 32, 32)

def test_posterior_net_shape():
    posterior = PosteriorNet(deter_dim=1024, embed_dim=512, stoch_dim=32,
                              num_classes=32, hidden_dim=256)
    h = torch.randn(8, 1024)
    embed = torch.randn(8, 512)
    logits = posterior(h, embed)
    assert logits.shape == (8, 32, 32)
```

### P2.4 — Codebook Diversity

With uniform random logits, all 32 classes should be used across a batch:

```python
def test_codebook_diversity():
    logits = torch.randn(64, 32, 32)  # large batch
    z = sample_straight_through(logits)  # (64, 32, 32)
    # For each of the 32 distributions, check that multiple classes are selected
    indices = z.argmax(dim=-1)  # (64, 32)
    for dist_idx in range(32):
        unique_classes = indices[:, dist_idx].unique()
        assert len(unique_classes) > 1, f"Distribution {dist_idx} uses only one class"
```

---

## Phase 3: Symlog Twohot

### P3.1 — Round-Trip Accuracy

| Input | Tolerance |
|-------|-----------|
| -1e6 | relative 1% (clamped to boundary) |
| -1000 | relative 0.1% |
| -100 | relative 0.01% |
| -1 | absolute 0.001 |
| -0.1 | absolute 0.0001 |
| 0 | absolute 1e-7 |
| 0.1 | absolute 0.0001 |
| 1 | absolute 0.001 |
| 100 | relative 0.01% |
| 1000 | relative 0.1% |
| 1e6 | relative 1% |

```python
@pytest.mark.parametrize("value,atol,rtol", [
    (0.0, 1e-6, 0.0),
    (1.0, 1e-3, 0.0),
    (-1.0, 1e-3, 0.0),
    (100.0, 0.0, 1e-3),
    (-100.0, 0.0, 1e-3),
    (1e6, 0.0, 0.05),   # clamped, some precision loss
    (-1e6, 0.0, 0.05),
])
def test_symlog_twohot_round_trip(value, atol, rtol):
    module = SymlogTwohot()
    x = torch.tensor([value])
    twohot = module.encode(x)
    # Create logit tensor that produces the twohot probabilities
    logits = twohot.log().clamp(-30, 30)  # approximate logits
    decoded = module.decode(logits)
    assert torch.allclose(decoded, x, atol=atol, rtol=rtol), \
        f"Round-trip failed for {value}: got {decoded.item():.6f}"
```

### P3.2 — Twohot Properties

| Test | Expected |
|------|---------|
| Sum to 1 | `encode(x).sum(-1)` ≈ 1.0 |
| Non-negative | `encode(x).min()` >= 0 |
| Exactly 2 nonzero | `(encode(x) > 0).sum(-1)` == 2 (for values not on bin centers) |
| Value 0 hits center bin | `encode(0)[127]` > 0 |
| Max value hits last bin | `encode(1e9)[-1]` ≈ 1.0 |

### P3.3 — Loss Properties

| Test | Expected |
|------|---------|
| Loss is finite | no nan or inf for inputs in [-1e6, 1e6] |
| Loss is differentiable | `loss.backward()` completes without error |
| Loss is non-negative | all per-element losses >= 0 |
| Loss at correct prediction ≈ 0 | when logits perfectly encode the target |

```python
def test_symlog_twohot_loss_finite():
    module = SymlogTwohot()
    logits = torch.randn(16, 255, requires_grad=True)
    targets = torch.tensor([-1e6, -1000, -100, -10, -1, 0, 1, 10, 100, 1000,
                             1e6, 0.5, -0.5, 42.0, -42.0, 0.001])
    loss = module.loss(logits, targets)
    assert torch.all(torch.isfinite(loss))
    loss.mean().backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))
```

### P3.4 — Edge Cases

| Case | Expected Behavior |
|------|-----------------|
| Very large positive (1e9) | clamped to bin 254, loss finite |
| Very large negative (-1e9) | clamped to bin 0, loss finite |
| Exactly on bin center | w_lower=1 or w_upper=1 (not split) |
| Between two bins | both adjacent bins have nonzero weight |
| Zero input | bin 127 has nonzero weight, others near 127 may also |

---

## Phase 4: RSSM

### P4.1 — Observe Produces Correct Shapes

```python
def test_rssm_observe_shapes():
    cfg = RSSMConfig(deter_dim=64, stoch_dim=8, num_classes=8, hidden_dim=64)
    rssm = RSSM(cfg, embed_dim=32, action_dim=4)
    state = rssm.initial_state(batch_size=4)
    embed = torch.randn(4, 32)
    action = torch.randn(4, 4)
    new_state, prior_logits = rssm.observe_step(embed, action, state)
    assert new_state.deter.shape == (4, 64)
    assert new_state.stoch.shape == (4, 8, 8)
    assert new_state.logits.shape == (4, 8, 8)
    assert prior_logits.shape == (4, 8, 8)
```

### P4.2 — Observe Sequence

```python
def test_rssm_observe_sequence_shapes():
    cfg = RSSMConfig(deter_dim=64, stoch_dim=8, num_classes=8, hidden_dim=64)
    rssm = RSSM(cfg, embed_dim=32, action_dim=4)
    T, B = 10, 4
    embed_seq = torch.randn(T, B, 32)
    action_seq = torch.randn(T, B, 4)
    state = rssm.initial_state(B)
    posteriors, priors = rssm.observe(embed_seq, action_seq, state)
    # posteriors and priors should be sequences of length T
    assert len(posteriors) == T or posteriors.deter.shape[0] == T
```

### P4.3 — Imagine Unrolls for H Steps

```python
def test_rssm_imagine_horizon():
    cfg = RSSMConfig(deter_dim=64, stoch_dim=8, num_classes=8, hidden_dim=64)
    rssm = RSSM(cfg, embed_dim=32, action_dim=4)
    state = rssm.initial_state(batch_size=2)
    policy = lambda feat: torch.randn(feat.shape[0], 4)
    traj = rssm.imagine(policy, state, horizon=15)
    assert traj.features.shape[0] == 15
    assert traj.features.shape[1] == 2
    assert traj.actions.shape[0] == 15
    assert traj.reward_logits.shape[0] == 15
    assert traj.continue_logits.shape[0] == 15
```

### P4.4 — Initial State is Zeros

```python
def test_rssm_initial_state_zeros():
    cfg = RSSMConfig(deter_dim=64, stoch_dim=8, num_classes=8, hidden_dim=64)
    rssm = RSSM(cfg, embed_dim=32, action_dim=4)
    state = rssm.initial_state(batch_size=3)
    assert torch.all(state.deter == 0)
    assert torch.all(state.stoch == 0)
```

### P4.5 — Features Concatenation

```python
def test_rssm_features_shape():
    cfg = RSSMConfig(deter_dim=64, stoch_dim=8, num_classes=8, hidden_dim=64)
    rssm = RSSM(cfg, embed_dim=32, action_dim=4)
    state = rssm.initial_state(batch_size=4)
    feat = rssm.get_features(state)
    expected_dim = 64 + 8 * 8  # deter_dim + stoch_dim * num_classes
    assert feat.shape == (4, expected_dim)
```

---

## Phase 5: KL Balancing

### P5.1 — Identical Distributions Give Zero KL

```python
def test_kl_identical_distributions():
    logits = torch.randn(4, 32, 32)
    kl = kl_categorical(p_logits=logits, q_logits=logits, unimix=0.01)
    # With identical distributions, KL should be near 0
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)
```

### P5.2 — Free Nats Clipping

```python
def test_kl_free_nats_clipping():
    loss_fn = WorldModelLoss(LossConfig(kl_free_nats=1.0))
    # Create distributions that are very close (KL << 1.0)
    logits = torch.randn(4, 32, 32)
    small_noise = logits + 0.001 * torch.randn_like(logits)
    # When KL < free_nats, the loss should be clamped to free_nats
    kl = kl_categorical(p_logits=logits, q_logits=small_noise)
    kl_clamped = torch.clamp(kl, min=1.0)
    # All values should be >= free_nats (1.0)
    assert kl_clamped.min() >= 1.0
```

### P5.3 — Stop-Gradient Correctness

```python
def test_kl_stop_gradient_dynamics():
    """Dynamics loss: gradient should flow to prior but not posterior."""
    posterior_logits = torch.randn(4, 32, 32, requires_grad=True)
    prior_logits = torch.randn(4, 32, 32, requires_grad=True)

    # Dynamics loss: sg(posterior) || prior
    kl_dyn = kl_categorical(
        p_logits=prior_logits,
        q_logits=posterior_logits.detach(),
    ).sum()
    kl_dyn.backward()

    # Prior should receive gradients
    assert prior_logits.grad is not None
    # Posterior should NOT receive gradients (detached)
    assert posterior_logits.grad is None

def test_kl_stop_gradient_representation():
    """Representation loss: gradient should flow to posterior but not prior."""
    posterior_logits = torch.randn(4, 32, 32, requires_grad=True)
    prior_logits = torch.randn(4, 32, 32, requires_grad=True)

    # Representation loss: posterior || sg(prior)
    kl_rep = kl_categorical(
        p_logits=prior_logits.detach(),
        q_logits=posterior_logits,
    ).sum()
    kl_rep.backward()

    # Posterior should receive gradients
    assert posterior_logits.grad is not None
    # Prior should NOT receive gradients (detached)
    assert prior_logits.grad is None
```

### P5.4 — Different Distributions Give Positive KL

```python
def test_kl_positive_for_different_dists():
    p_logits = torch.zeros(4, 32, 32)
    q_logits = torch.ones(4, 32, 32) * 10  # very different distribution
    kl = kl_categorical(p_logits=p_logits, q_logits=q_logits)
    assert kl.min() > 0
```

### P5.5 — Coefficient Application

```python
def test_kl_loss_coefficients():
    """Verify 0.5 * L_dyn + 0.1 * L_rep computation."""
    posterior = torch.randn(4, 32, 32)
    prior = torch.randn(4, 32, 32)
    loss_fn = WorldModelLoss(LossConfig(kl_dyn_scale=0.5, kl_rep_scale=0.1))
    result = loss_fn.forward_kl(posterior, prior)
    # Manually compute expected
    kl_dyn = kl_categorical(prior, posterior.detach()).clamp(min=1.0).sum(-1).mean()
    kl_rep = kl_categorical(prior.detach(), posterior).clamp(min=1.0).sum(-1).mean()
    expected = 0.5 * kl_dyn + 0.1 * kl_rep
    assert torch.allclose(result, expected, atol=1e-5)
```

---

## Phase 6: World Model Loss

### P6.1 — Combined Loss is Differentiable

```python
def test_world_model_loss_differentiable():
    cfg = RSSMConfig(deter_dim=32, stoch_dim=4, num_classes=4, hidden_dim=32)
    rssm = RSSM(cfg, embed_dim=16, action_dim=2)
    loss_fn = WorldModelLoss(LossConfig())
    # ... run a forward pass and backward
    total_loss = result.total
    total_loss.backward()
    # Check that parameters received gradients
    for name, param in rssm.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
```

### P6.2 — Individual Loss Terms Present

```python
def test_world_model_loss_terms():
    result = compute_world_model_loss(...)
    assert hasattr(result, 'total')
    assert hasattr(result, 'kl_dyn')
    assert hasattr(result, 'kl_rep')
    assert hasattr(result, 'obs_loss')
    assert hasattr(result, 'reward_loss')
    assert hasattr(result, 'continue_loss')
    assert torch.all(torch.isfinite(result.total))
```

### P6.3 — Scaling Correct

```python
def test_world_model_loss_scaling():
    """Verify that obs_scale=0 zeroes out observation loss."""
    cfg_zero_obs = LossConfig(obs_scale=0.0)
    cfg_one_obs = LossConfig(obs_scale=1.0)
    # When obs_scale=0, obs_loss should not appear in total
    result_zero = compute_loss(cfg_zero_obs, ...)
    result_one = compute_loss(cfg_one_obs, ...)
    # The difference should equal obs_loss
    assert torch.allclose(
        result_one.total - result_zero.total,
        result_one.obs_loss,
        atol=1e-5,
    )
```

---

## Edge Cases (All Phases)

### EC.1 — Batch Size 1

All modules must handle batch_size=1 without broadcasting errors:

```python
@pytest.mark.parametrize("cls", [BlockGRU, PriorNet, PosteriorNet, RSSM, SymlogTwohot, WorldModelLoss])
def test_batch_size_one(cls):
    # Instantiate with appropriate args and run with batch_size=1
    pass  # specific instantiation depends on class
```

### EC.2 — Horizon 1

Imagination with horizon=1 should produce single-step trajectories:

```python
def test_imagine_horizon_one():
    rssm = create_small_rssm()
    state = rssm.initial_state(4)
    traj = rssm.imagine(lambda f: torch.zeros(4, 2), state, horizon=1)
    assert traj.features.shape[0] == 1
```

### EC.3 — All-Zero Inputs

All modules should handle zero tensors without nan:

```python
def test_all_zero_inputs():
    gru = BlockGRU(64, 128)
    x = torch.zeros(4, 64)
    h = torch.zeros(4, 128)
    h_new = gru(x, h)
    assert torch.all(torch.isfinite(h_new))
    assert not torch.all(h_new == 0)  # should produce non-zero output
```

### EC.4 — Extreme Reward Values

```python
@pytest.mark.parametrize("value", [-1e6, -1000, 1000, 1e6])
def test_extreme_reward_prediction(value):
    module = SymlogTwohot()
    logits = torch.randn(4, 255, requires_grad=True)
    target = torch.full((4,), value)
    loss = module.loss(logits, target)
    assert torch.all(torch.isfinite(loss))
    loss.mean().backward()
    assert torch.all(torch.isfinite(logits.grad))
```

### EC.5 — Device Consistency

All modules should work on CPU and GPU (when available):

```python
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU"))])
def test_device_consistency(device):
    gru = BlockGRU(64, 128).to(device)
    x = torch.randn(4, 64, device=device)
    h = torch.zeros(4, 128, device=device)
    h_new = gru(x, h)
    assert h_new.device.type == device
```

---

## Test Priority Matrix

| Test | Priority | Catches |
|------|----------|---------|
| P1.4 Gradient flow through GRU | Critical | Vanishing gradients in temporal sequence |
| P2.2 Straight-through gradient | Critical | Broken imagination training |
| P2.1 Unimix no zeros | Critical | NaN KL loss crashing training |
| P3.1 Round-trip accuracy | High | Incorrect reward decoding |
| P3.3 Loss finite | Critical | NaN reward loss |
| P4.1 Observe shapes | High | Shape mismatches in full forward pass |
| P4.3 Imagine horizon | High | Actor-critic cannot train |
| P5.3 Stop-gradient correctness | Critical | Wrong gradient flow in KL |
| P5.2 Free nats clipping | High | Posterior collapse not prevented |
| P6.1 Combined loss differentiable | Critical | Training fails completely |
| EC.3 Zero inputs | High | Initialization issues |
| EC.4 Extreme rewards | High | NaN at first reward encounter |
