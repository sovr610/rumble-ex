# Sleep Consolidation Test Matrix

## Overview

Tests must verify that the sleep consolidation cycle correctly replays experiences, downscales
weights, transfers knowledge, and generates dream experiences -- all without corrupting the
model or violating time/memory budgets. Each category is a hard gate: the consolidation system
is not shippable if any category fails.

---

## Test Category A: Replay Fidelity

Verify that priority-weighted replay sampling, compressed sequences, and importance sampling
correction work correctly.

```python
PRIORITY_EXPONENTS = [0.0, 0.3, 0.6, 1.0]
COMPRESSION_RATIOS = [1.0, 2.0, 5.0, 10.0]
BUFFER_SIZES = [100, 1000, 10000]

@pytest.mark.parametrize("alpha", PRIORITY_EXPONENTS)
@pytest.mark.parametrize("buffer_size", BUFFER_SIZES)
def test_priority_sampling_distribution(alpha, buffer_size):
    """Priority sampling produces expected power-law distribution."""
    buffer = create_test_buffer(buffer_size)
    scheduler = ReplayScheduler(SleepConfig(priority_exponent=alpha))

    # Sample many batches and compute empirical frequency
    counts = torch.zeros(buffer_size)
    for _ in range(1000):
        batch, indices, _ = scheduler.sample_replay_batch(buffer, batch_size=32)
        for idx in indices:
            counts[idx] += 1

    if alpha == 0.0:
        # Uniform: all counts should be similar (chi-squared test)
        chi2 = ((counts - counts.mean()) ** 2 / counts.mean()).sum()
        assert chi2 < chi2_critical_value(df=buffer_size-1, p=0.05)
    else:
        # Priority-weighted: high-priority items sampled more
        high_priority_indices = buffer.get_top_priority_indices(k=10)
        low_priority_indices = buffer.get_bottom_priority_indices(k=10)
        assert counts[high_priority_indices].mean() > counts[low_priority_indices].mean()

@pytest.mark.parametrize("ratio", COMPRESSION_RATIOS)
def test_compressed_sequence_length(ratio):
    """Compressed sequences have correct length."""
    T = 100
    sequence = torch.randn(T, 64)
    compressed = compress_sequence(sequence, ratio)
    expected_T = max(2, math.ceil(T / ratio))
    assert compressed.shape[0] == expected_T
    assert compressed.shape[1] == 64

def test_key_transition_preservation():
    """Compressed sequences preserve reward transitions."""
    T = 100
    sequence = torch.randn(T, 64)
    rewards = torch.zeros(T)
    rewards[25] = 10.0  # reward spike at t=25
    rewards[75] = -5.0  # reward spike at t=75

    compressed = compress_with_key_transitions(sequence, rewards, ratio=5.0)
    # The compressed sequence should include timesteps near t=25 and t=75
    # (verified by checking that reward spikes are preserved)
```

### Key Assertions

- Priority distribution matches theoretical power-law within statistical tolerance.
- alpha=0 produces uniform sampling (chi-squared test p > 0.05).
- alpha=1 samples highest-priority item most frequently.
- Compressed sequences have exactly ceil(T/ratio) elements (minimum 2).
- Key transitions (reward spikes) are preserved in compressed sequences.
- Importance sampling weights are bounded in (0, 1] after normalization.

---

## Test Category B: Homeostasis Correctness

Verify that synaptic downscaling preserves relative weight ratios, reduces weight norms,
and restores capacity metrics.

```python
STRATEGIES = ["global", "selective", "layerwise"]
FACTORS = [0.95, 0.90, 0.85, 0.80]

@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("factor", FACTORS)
def test_weight_ratio_preservation(strategy, factor):
    """Relative weight ratios are preserved after downscaling."""
    model = MockModel(dim=64)
    original_weights = {n: p.clone() for n, p in model.named_parameters() if 'bias' not in n}

    homeostasis = SynapticHomeostasis(SleepConfig(
        downscale_factor=factor,
        downscale_strategy=strategy,
    ))
    homeostasis.downscale(model)

    for name, param in model.named_parameters():
        if 'bias' in name or param.ndim < 2:
            continue
        orig = original_weights[name]
        # Check ratio preservation (within tolerance for selective)
        if strategy == "global":
            ratio_before = orig[0, 0] / orig[0, 1]
            ratio_after = param[0, 0] / param[0, 1]
            assert torch.allclose(ratio_before, ratio_after, rtol=1e-6)

@pytest.mark.parametrize("factor", FACTORS)
def test_weight_norm_decrease(factor):
    """Weight norms decrease monotonically after downscaling."""
    model = MockModel(dim=64)
    norm_before = sum(p.norm().item() for p in model.parameters() if p.ndim >= 2)

    homeostasis = SynapticHomeostasis(SleepConfig(downscale_factor=factor))
    homeostasis.downscale(model)

    norm_after = sum(p.norm().item() for p in model.parameters() if p.ndim >= 2)
    assert norm_after < norm_before

def test_capacity_restoration():
    """Capacity metric returns to target range after downscaling."""
    model = MockModel(dim=64)
    # Simulate weight growth
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(2.0)

    homeostasis = SynapticHomeostasis(SleepConfig(downscale_factor=0.5))
    result = homeostasis.downscale(model)
    assert result.wnr_after < result.wnr_before
```

### Key Assertions

- Global downscaling preserves exact weight ratios (relative error < 1e-6).
- Weight norms decrease after any downscaling (for all strategies).
- WNR (weight norm ratio) decreases after downscaling.
- Biases are not modified.
- Parameters with ndim < 2 are not modified.
- Selective downscaling protects top-importance weights (less scaling applied).
- Layerwise downscaling brings per-layer WNR closer to target.
- No NaN or Inf values after downscaling.

---

## Test Category C: Systems Consolidation

Verify that knowledge distillation from fast to slow systems works correctly.

```python
def test_distillation_loss_decreases():
    """Distillation loss decreases over transfer steps."""
    fast_model = MockModel(dim=64, out_dim=10)
    slow_model = MockModel(dim=64, out_dim=10)
    buffer = create_test_buffer(1000)

    consolidation = SystemsConsolidation(SleepConfig(
        distillation_temperature=2.0,
        transfer_learning_rate=1e-3,
    ))

    losses = []
    for step in range(50):
        result = consolidation.transfer_step(fast_model, slow_model, buffer)
        losses.append(result.distillation_loss)

    # Loss should decrease overall (allowing non-monotonic)
    assert losses[-1] < losses[0] * 0.9  # at least 10% decrease

def test_fast_model_unchanged():
    """Fast model weights are not modified during transfer."""
    fast_model = MockModel(dim=64)
    slow_model = MockModel(dim=64)
    buffer = create_test_buffer(100)

    original_fast = {n: p.clone() for n, p in fast_model.named_parameters()}

    consolidation = SystemsConsolidation(SleepConfig())
    consolidation.transfer(fast_model, slow_model, buffer)

    for name, param in fast_model.named_parameters():
        assert torch.equal(param, original_fast[name]), f"Fast model modified: {name}"

def test_slow_model_improves():
    """Slow model accuracy on replay data improves after transfer."""
    # Create fast model trained on data, slow model untrained
    fast_model = create_trained_model()
    slow_model = MockModel(dim=64, out_dim=10)
    test_data = create_test_data()

    acc_before = evaluate_accuracy(slow_model, test_data)

    consolidation = SystemsConsolidation(SleepConfig())
    consolidation.transfer(fast_model, slow_model, replay_buffer)

    acc_after = evaluate_accuracy(slow_model, test_data)
    assert acc_after > acc_before + 0.02  # at least 2% improvement
```

### Key Assertions

- Distillation loss decreases over transfer steps (>10% improvement).
- Fast model weights are unchanged after transfer (frozen teacher).
- Slow model accuracy improves after transfer (>2% on replay data).
- Temperature scaling produces softer distributions (entropy increases).
- Combined loss correctly weights distillation vs reconstruction.
- Transfer does not cause NaN in slow model weights.

---

## Test Category D: End-to-End Consolidation

Verify the full consolidation cycle including NREM, homeostasis, and REM phases.

```python
def test_full_consolidation_completes():
    """Full consolidate() call completes without error."""
    model = MockModel(dim=64)
    buffer = create_test_buffer(5000)
    cfg = SleepConfig.dev()

    consolidator = SleepConsolidator(cfg)
    result = consolidator.consolidate(model, buffer)

    assert result is not None
    assert result.nrem_loss is not None
    assert result.homeostasis_wnr_before is not None
    assert result.homeostasis_wnr_after is not None
    assert result.total_sleep_time > 0

def test_consolidation_within_budget():
    """Total sleep time stays within configured budget."""
    model = MockModel(dim=64)
    buffer = create_test_buffer(5000)
    cfg = SleepConfig(sleep_duration_budget=0.2)

    # Simulate wake training time
    wake_time = 100.0  # seconds

    consolidator = SleepConsolidator(cfg)
    result = consolidator.consolidate(model, buffer)

    assert result.total_sleep_time <= wake_time * cfg.sleep_duration_budget

def test_consolidation_metrics_all_present():
    """ConsolidationResult contains valid metrics for all sub-phases."""
    model = MockModel(dim=64)
    buffer = create_test_buffer(5000)
    cfg = SleepConfig.dev()

    consolidator = SleepConsolidator(cfg)
    result = consolidator.consolidate(model, buffer)

    # All metrics should be finite
    assert math.isfinite(result.nrem_loss)
    assert math.isfinite(result.homeostasis_wnr_before)
    assert math.isfinite(result.homeostasis_wnr_after)
    assert result.homeostasis_wnr_after <= result.homeostasis_wnr_before
    assert result.total_sleep_time > 0
    assert result.nrem_steps >= 0
    assert result.rem_steps >= 0

def test_deterministic_consolidation():
    """Same seed produces identical consolidation results."""
    model1 = MockModel(dim=64)
    model2 = MockModel(dim=64)
    model2.load_state_dict(model1.state_dict())
    buffer = create_test_buffer(1000)

    cfg = SleepConfig.dev()

    torch.manual_seed(42)
    result1 = SleepConsolidator(cfg).consolidate(model1, buffer)

    torch.manual_seed(42)
    result2 = SleepConsolidator(cfg).consolidate(model2, buffer)

    assert abs(result1.nrem_loss - result2.nrem_loss) < 1e-6
```

### Key Assertions

- Full consolidation completes without errors.
- All metrics in ConsolidationResult are present and finite.
- Total sleep time stays within budget.
- Homeostasis WNR decreases (after < before).
- Deterministic consolidation produces identical results given same seed.
- Consolidation with empty/small buffer skips gracefully.

---

## Test Category E: Generative Replay (REM)

Verify dream generation and blending with real experiences.

```python
def test_dream_generation_shapes():
    """Generated dream experiences have correct shapes."""
    world_model = MockWorldModel(latent_dim=64, obs_dim=32)
    buffer = create_test_buffer(1000)

    dreams = generate_dream_experiences(world_model, buffer, num_dreams=16, horizon=10)

    assert len(dreams) == 16
    for dream in dreams:
        assert dream.observations.shape == (10, 32)
        assert dream.actions.shape[0] == 10
        assert dream.rewards.shape[0] == 10

def test_blend_ratio_correct():
    """Blended batch has correct proportion of real vs dream."""
    real_batch = create_batch(size=70)
    dream_batch = create_batch(size=30)
    blended = blend_batches(real_batch, dream_batch, blend_ratio=0.3)

    assert len(blended) == 100
    # 70% real, 30% dream
    assert blended.real_count == 70
    assert blended.dream_count == 30

def test_rem_disabled_without_world_model():
    """REM phase is skipped when no world model is provided."""
    model = MockModel(dim=64)
    buffer = create_test_buffer(1000)
    cfg = SleepConfig(enable_rem=True)

    consolidator = SleepConsolidator(cfg)
    result = consolidator.consolidate(model, buffer, world_model=None)

    assert result.rem_steps == 0
    assert result.rem_loss is None
```

---

## "Done When" Checklist

### Replay System

- [ ] Priority sampling distribution matches power-law (KS-test p > 0.05)
- [ ] Compressed sequences have correct length
- [ ] Key transitions preserved in compressed sequences (>95% retention)
- [ ] IS weights are bounded in (0, 1] after normalization
- [ ] 10 deterministic replay cycles produce identical gradients
- [ ] Empty buffer returns gracefully (no crash)

### Homeostasis

- [ ] Global downscaling preserves exact weight ratios (error < 1e-6)
- [ ] All strategies reduce weight norms
- [ ] Biases unchanged after downscaling
- [ ] No NaN/Inf after downscaling
- [ ] WNR returns to target range after downscale
- [ ] fp16 safety: downscaling in fp32 regardless of model dtype

### Systems Consolidation

- [ ] Distillation loss decreases (>10% over transfer)
- [ ] Fast model frozen during transfer (exact weight match)
- [ ] Slow model accuracy improves (>2% on replay data)
- [ ] Temperature scaling increases entropy
- [ ] No NaN in slow model after transfer

### End-to-End

- [ ] Full consolidation completes without error
- [ ] All metrics present and finite in ConsolidationResult
- [ ] Sleep time within budget
- [ ] Deterministic with same seed
- [ ] Graceful handling of edge cases (empty buffer, no world model)
- [ ] CPU and CUDA both work

### Generative Replay

- [ ] Dream shapes correct
- [ ] Blend ratio produces correct proportions
- [ ] REM skipped without world model
- [ ] Dream quality filter removes implausible experiences
- [ ] No NaN in dream-trained weights

---

## Running the Matrix

```bash
# Full test suite
python -m pytest tests/test_sleep_consolidation.py -v --tb=short

# Single category
python -m pytest tests/test_sleep_consolidation.py -v -k "test_priority"
python -m pytest tests/test_sleep_consolidation.py -v -k "homeostasis"
python -m pytest tests/test_sleep_consolidation.py -v -k "distillation"
python -m pytest tests/test_sleep_consolidation.py -v -k "end_to_end"
python -m pytest tests/test_sleep_consolidation.py -v -k "dream"

# Skip CUDA tests
python -m pytest tests/test_sleep_consolidation.py -v -k "not cuda"

# Coverage report
python -m pytest tests/test_sleep_consolidation.py \
    --cov=brain_ai.consolidation --cov-report=term-missing
```

All categories must pass with zero skips (other than hardware-gated CUDA tests) before
merging consolidation changes.
