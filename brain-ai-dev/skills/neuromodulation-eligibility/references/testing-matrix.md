# Testing Matrix: Neuromodulation + Eligibility Traces (Three-Factor Learning)

This document defines every test case for the neuromodulation and eligibility traces skill.
Tests are organized by class, with each entry specifying the scenario, assertion pattern,
tolerance, and any fixtures or skip conditions. The generated test suite lives at
`tests/test_neuromodulation.py` and targets approximately 80 test cases across seven classes.

---

## Shared Fixtures and Conventions

### Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `eligibility_config` | function | `EligibilityConfig()` with defaults (accumulating, tau_e=20, rate kernel) |
| `neuromod_config` | function | `NeuromodConfig()` with defaults (weighted_sum combination) |
| `three_factor_config` | function | `ThreeFactorConfig()` with defaults (online mode, lr=0.001) |
| `trace_module` | function | `EligibilityTraceModule(N_pre=32, N_post=16, config=eligibility_config)` |
| `neuromod_gate` | function | `NeuromodulatoryGate(config=neuromod_config)` |
| `three_factor` | function | `ThreeFactorUpdate(config=three_factor_config)` |
| `pre_spikes` | function | `torch.bernoulli(torch.full((B, N_pre), 0.3))` with B=4, N_pre=32 |
| `post_spikes` | function | `torch.bernoulli(torch.full((B, N_post), 0.3))` with B=4, N_post=16 |
| `pre_rates` | function | `torch.rand(B, N_pre)` with B=4, N_pre=32 |
| `post_rates` | function | `torch.rand(B, N_post)` with B=4, N_post=16 |
| `random_weights` | function | `torch.randn(N_post, N_pre) * 0.1` |
| `device` | session | `torch.device("cuda" if torch.cuda.is_available() else "cpu")` |

All fixtures use `torch.manual_seed(42)` for reproducibility. Batch size B=4 unless stated otherwise.

### Skip Conditions

| Condition | Decorator | Rationale |
|---|---|---|
| No CUDA available | `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")` | AMP autocast tests require GPU |
| Optional dependency | `@pytest.mark.skipif(...)` per module | htm.core, ncps, learn2learn may not be installed |

### Standard Tolerances

| Comparison | Tolerance | Usage |
|---|---|---|
| Exact zero | `torch.all(tensor == 0)` | Third-factor gating, reset, zero-input tests |
| Approximate equality | `atol=1e-6, rtol=1e-5` | fp32 determinism checks |
| Loose float comparison | `atol=1e-4, rtol=1e-3` | AMP/mixed-precision results |
| Range checks | Strict inequality or `<=` / `>=` | Modulator output bounds, clamp checks |

### Common Assertion Patterns

- **Exact zero**: `assert torch.all(delta_w == 0)` -- used for third-factor gating, zero-input, reset
- **Non-zero**: `assert torch.any(delta_w != 0)` -- used to confirm updates occur
- **Bounded**: `assert torch.all(output >= lo) and torch.all(output <= hi)`
- **Shape**: `assert tensor.shape == expected_shape`
- **Determinism**: run N times, `assert torch.allclose(run_i, run_0, atol=1e-6)`
- **Monotonic**: `assert values[-1] > values[0]` or `assert loss_final < loss_initial`

---

## 1. TestEligibilityTraces (~18 tests)

Tests the `EligibilityTraceModule` covering trace accumulation, decay, reset/carry, STDP kernels,
rate-based correlation, clamping, numerical stability, and determinism.

**Fixtures**: `trace_module`, `eligibility_config`, `pre_spikes`, `post_spikes`, `pre_rates`,
`post_rates`, `device`.

| # | Test Name | Description | Setup | Assertion | Tolerance |
|---|---|---|---|---|---|
| 1.1 | `test_accumulating_trace_increases` | Repeated pre/post activity with accumulating trace type causes monotonic increase in trace norm. | Config: `trace_type="accumulating"`. Call `update(pre, post)` 5 times with the same nonzero pre/post. | `torch.norm(e_t5) > torch.norm(e_t1)` | N/A (strict inequality) |
| 1.2 | `test_replacing_trace_bounded` | Replacing trace never exceeds the maximum single-step correlation value regardless of repetition count. | Config: `trace_type="replacing"`. Call `update(pre, post)` 20 times. Compute single-step max: `f_max = max(f(pre, post))`. | `torch.all(e <= f_max + atol)` | atol=1e-6 |
| 1.3 | `test_dutch_trace_hybrid` | Dutch trace grows faster than replacing but slower than pure accumulating. | Run all three trace types for 10 steps with identical pre/post. | `norm(e_replacing) <= norm(e_dutch) <= norm(e_accumulating)` for steps > 3 | N/A (strict ordering) |
| 1.4 | `test_decay_toward_zero` | Without new pre/post activity, traces decay toward zero. | Call `update(pre, post)` once, then call `update(zeros, zeros)` for 100 steps. | `torch.norm(e_t100) < 0.01 * torch.norm(e_t1)` | Relative threshold 1% |
| 1.5 | `test_tau_e_effect` | Larger tau_e produces slower decay. | Two modules: tau_e=10 and tau_e=50. Same initial trace, then 20 zero-input steps. | `torch.norm(e_slow_tau) > torch.norm(e_fast_tau)` at each step | N/A (strict inequality) |
| 1.6 | `test_reset_clears_traces` | `reset()` sets all trace values to exactly zero. | Call `update(pre, post)` 5 times to build up traces, then call `reset(B, device)`. | `torch.all(e == 0)` | Exact zero |
| 1.7 | `test_carry_across_calls` | Traces persist across sequential `update()` calls when carry mode is active. | Call `update(pre1, post1)`, record `e1`. Call `update(zeros, zeros)` once. | `torch.norm(e_after) > 0` and `e_after` reflects decayed `e1` | atol=1e-6 vs manual decay |
| 1.8 | `test_batch_independence` | Modifying one batch element does not affect others. | B=4. Set `pre[0] = 1.0`, `pre[1:] = 0.0`, `post[0] = 1.0`, `post[1:] = 0.0`. | `torch.all(e[1:] == 0)` and `torch.any(e[0] != 0)` | Exact zero for inactive batches |
| 1.9 | `test_stdp_causal_positive` | Causal spike timing (pre before post, dt > 0) produces positive eligibility. | Config: `kernel="stdp_pair"`. `dt = +5.0` (pre fires 5 ms before post). | `torch.all(e[pre & post mask] > 0)` | Strict positive |
| 1.10 | `test_stdp_anticausal_negative` | Anti-causal spike timing (post before pre, dt < 0) produces negative eligibility. | Config: `kernel="stdp_pair"`. `dt = -5.0`. | `torch.all(e[pre & post mask] < 0)` | Strict negative |
| 1.11 | `test_rate_outer_product_shape` | Rate-based kernel produces outer product with correct shape (B, N_post, N_pre). | Config: `kernel="rate"`. `pre: (B, N_pre)`, `post: (B, N_post)`. | `e.shape == (B, N_post, N_pre)` | Shape match |
| 1.12 | `test_rate_diagonal_elementwise` | When N_pre == N_post, diagonal variant produces element-wise product shape (B, N). | Config: `kernel="rate"`, `diagonal=True`, `N_pre=N_post=32`. | `e.shape == (B, 32)` and `torch.allclose(e, pre * post)` after first step | atol=1e-6 |
| 1.13 | `test_zero_pre_zero_trace_update` | Zero pre-synaptic activity produces zero trace update regardless of post. | `pre = torch.zeros(B, N_pre)`, `post = torch.rand(B, N_post)`. | Trace update component is exactly zero; `e` changes only due to decay of existing trace. | Exact zero for update term |
| 1.14 | `test_zero_post_zero_trace_update` | Zero post-synaptic activity produces zero trace update regardless of pre. | `pre = torch.rand(B, N_pre)`, `post = torch.zeros(B, N_post)`. | Trace update component is exactly zero. | Exact zero for update term |
| 1.15 | `test_clamp_range_enforced` | Trace values never exceed configured clamp bounds. | Config: `clamp_range=[-2.0, 2.0]`. Run 50 update steps with large pre/post values (all ones). | `torch.all(e >= -2.0) and torch.all(e <= 2.0)` | Exact bounds |
| 1.16 | `test_fp32_under_amp` | Traces are computed in fp32 even when AMP autocast is active. | Wrap `update()` call in `torch.cuda.amp.autocast()`. | `e.dtype == torch.float32` | Exact dtype check |
| 1.17 | `test_nan_guard` | NaN inputs trigger trace reset rather than propagating NaN. | Pass `pre = torch.full((B, N_pre), float('nan'))`. | `torch.all(torch.isfinite(e))` | No NaN, no Inf |
| 1.18 | `test_determinism_across_runs` | Identical inputs produce identical traces across 10 independent runs. | `torch.manual_seed(42)` before each run. Fixed pre/post/dt sequence for 20 steps. | `torch.allclose(e_run_i, e_run_0, atol=1e-6)` for i in 1..9 | atol=1e-6 |

**Skip**: Test 1.16 skipped without CUDA (AMP autocast requires GPU).

---

## 2. TestNeuromodulatoryGate (~15 tests)

Tests the `NeuromodulatoryGate` module covering individual modulator computation, range guarantees,
combination functions, state updates, and determinism.

**Fixtures**: `neuromod_gate`, `neuromod_config`, `device`.

| # | Test Name | Description | Setup | Assertion | Tolerance |
|---|---|---|---|---|---|
| 2.1 | `test_da_positive_reward` | Positive reward signal produces positive DA output. | `signals = {"reward": torch.tensor([1.0])}`. | `modulators["DA"] > 0` | Strict positive |
| 2.2 | `test_da_negative_reward` | Negative reward signal produces negative DA output. | `signals = {"reward": torch.tensor([-1.0])}`. | `modulators["DA"] < 0` | Strict negative |
| 2.3 | `test_da_range` | DA output is bounded in [-1, 1] via tanh activation. | Sweep `reward` over `linspace(-10, 10, 100)`. | `torch.all(DA >= -1.0) and torch.all(DA <= 1.0)` | Exact bounds |
| 2.4 | `test_da_baseline_tracking` | DA baseline updates via exponential moving average after each call. | Call forward 10 times with reward=1.0, record baseline after each. | Baseline monotonically increases toward 1.0; `baseline[-1] > baseline[0]` | N/A (monotonic) |
| 2.5 | `test_ach_high_novelty` | High novelty input produces high ACh value. | `signals = {"novelty": torch.tensor([5.0])}`. | `modulators["ACh"] > 0.8` | Threshold 0.8 |
| 2.6 | `test_ach_range` | ACh output is bounded in [0, 1] via sigmoid. | Sweep `novelty` over `linspace(-10, 10, 100)`. | `torch.all(ACh >= 0.0) and torch.all(ACh <= 1.0)` | Exact bounds |
| 2.7 | `test_ne_high_urgency` | High urgency input produces high NE value. | `signals = {"urgency": torch.tensor([5.0])}`. | `modulators["NE"] > 0.8` | Threshold 0.8 |
| 2.8 | `test_ne_range` | NE output is bounded in [0, 1] via sigmoid. | Sweep `urgency` over `linspace(-10, 10, 100)`. | `torch.all(NE >= 0.0) and torch.all(NE <= 1.0)` | Exact bounds |
| 2.9 | `test_5ht_high_patience` | High patience input produces high 5-HT value. | `signals = {"patience": torch.tensor([5.0])}`. | `modulators["5HT"] > 0.8` | Threshold 0.8 |
| 2.10 | `test_5ht_range` | 5-HT output is bounded in [0, 1] via sigmoid. | Sweep `patience` over `linspace(-10, 10, 100)`. | `torch.all(sht >= 0.0) and torch.all(sht <= 1.0)` | Exact bounds |
| 2.11 | `test_all_zero_neutral_state` | All modulators are zero (or at neutral baseline) when all input signals are zero. | `signals = {"reward": 0, "novelty": 0, "urgency": 0, "patience": 0}`. All tensors zero. | `DA == 0`, `ACh == sigmoid(0) = 0.5` (or near-zero depending on bias), `NE == 0.5`, `5HT == 0.5`. For tanh-based: `DA == 0`. Verify `global_plasticity` is at neutral. | atol=1e-6 |
| 2.12 | `test_weighted_sum_combination` | Weighted sum produces correct linear combination of modulators. | Config: `combination_fn="weighted_sum"`. Known modulator values. | `global_plasticity == sum(w_i * m_i)` within tolerance. | atol=1e-5 |
| 2.13 | `test_gated_product_combination` | DA gates the contribution of other modulators in product mode. | Config: `combination_fn="gated_product"`. Set DA=0. | `global_plasticity == 0` regardless of ACh/NE/5HT values. | Exact zero |
| 2.14 | `test_mlp_combination_bounded` | MLP combination function produces bounded scalar output. | Config: `combination_fn="mlp"`. Random modulator values. | `global_plasticity` is finite and bounded (check no NaN/Inf and value within reasonable range). | `torch.isfinite(global_plasticity)` |
| 2.15 | `test_determinism` | Same inputs and initial state produce identical outputs across 10 runs. | `torch.manual_seed(42)` before each run. Same signals dict. | `torch.allclose(modulators_i, modulators_0, atol=1e-6)` and `torch.allclose(gp_i, gp_0, atol=1e-6)` | atol=1e-6 |

---

## 3. TestThreeFactorUpdate (~15 tests)

Tests the `ThreeFactorUpdate` module covering the fundamental gating property, weight clamping,
update modes, per-layer configuration, update frequency, and numerical properties.

**Fixtures**: `three_factor`, `three_factor_config`, `random_weights`, `trace_module`, `device`.

| # | Test Name | Description | Setup | Assertion | Tolerance |
|---|---|---|---|---|---|
| 3.1 | `test_zero_modulator_zero_update` | **THE fundamental test.** When `mod_signal=0`, `delta_w` is exactly zero regardless of eligibility trace values. | Build nonzero eligibility via 5 update steps. Set `mod_signal = torch.tensor(0.0)`. | `torch.all(delta_w == 0)` | Exact zero |
| 3.2 | `test_nonzero_modulator_nonzero_update` | When `mod_signal != 0` and eligibility is nonzero, `delta_w` is nonzero. | Build nonzero eligibility. Set `mod_signal = torch.tensor(1.0)`. | `torch.any(delta_w != 0)` | Strict non-zero |
| 3.3 | `test_positive_modulator_strengthens` | Positive modulator with positive eligibility produces positive weight update direction. | `e > 0`, `mod_signal = 1.0`. | `torch.all(delta_w[e > 0] > 0)` | Strict positive |
| 3.4 | `test_negative_modulator_weakens` | Negative modulator reverses the sign of weight updates relative to eligibility. | `e > 0`, `mod_signal = -1.0`. | `torch.all(delta_w[e > 0] < 0)` | Strict negative |
| 3.5 | `test_weight_clamp` | Updated weights stay within configured `[w_min, w_max]` bounds. | Config: `weight_clamp=[-0.5, 0.5]`. Apply large mod_signal (10.0) with large eligibility. | `torch.all(w_new >= -0.5) and torch.all(w_new <= 0.5)` | Exact bounds |
| 3.6 | `test_delta_clamp` | Per-step update magnitude is bounded by `max_delta` if configured. | Config: `max_delta=0.01`. Large mod_signal and eligibility. | `torch.all(torch.abs(delta_w) <= 0.01 + atol)` | atol=1e-7 |
| 3.7 | `test_online_mode_inplace` | In online mode, weights are modified in-place on the parameter tensor. | Config: `mode="online"`. Record `id(weights.data)` before and after. | `weights` tensor contains updated values; original tensor object is the same. | Verify `w_before != w_after` element-wise and `data_ptr` unchanged. |
| 3.8 | `test_hybrid_mode_auxiliary_loss` | In hybrid mode, an auxiliary loss scalar is computed and returned alongside the standard forward pass. | Config: `mode="hybrid"`. | Return value includes `aux_loss` that is a scalar tensor with `requires_grad` (if attached to graph) or a detached float. `aux_loss >= 0`. | N/A (type and sign check) |
| 3.9 | `test_per_layer_config` | Different layers can receive different update rules (e.g., different lr, different clamp). | Register two layers with different `ThreeFactorConfig` instances (lr=0.01 vs lr=0.001). Same eligibility and mod_signal. | `torch.norm(delta_w_layer1) / torch.norm(delta_w_layer2)` approximately equals `lr1 / lr2 = 10.0`. | rtol=0.1 |
| 3.10 | `test_eligible_layer_registry` | Only layers registered as "eligible" receive three-factor updates. | Register layer_a as eligible, layer_b as not eligible. Run update. | `delta_w_a` is nonzero; `layer_b` weights unchanged from initial values. | Exact match for non-eligible |
| 3.11 | `test_non_eligible_layers_frozen` | Weights of non-eligible layers are bitwise identical before and after a three-factor update cycle. | Clone weights before update. Run full update cycle. | `torch.equal(w_before, w_after)` for non-eligible layers. | Exact (bitwise) |
| 3.12 | `test_update_frequency` | When `update_frequency=N`, weight updates occur only every N-th call to `apply_update`. | Config: `update_frequency=5`. Call `apply_update` 10 times, track which calls produce nonzero delta_w. | Nonzero delta_w only at steps 5 and 10 (1-indexed). All other steps produce exact zero delta_w. | Exact zero for non-update steps |
| 3.13 | `test_lr_scaling` | Learning rate scales the magnitude of delta_w linearly. | Run with lr=0.01 and lr=0.001, same eligibility and mod_signal. | `torch.allclose(delta_w_high / delta_w_low, torch.tensor(10.0), rtol=0.01)` element-wise where both are nonzero. | rtol=0.01 |
| 3.14 | `test_detached_computation` | Three-factor weight updates do not create a computational graph (no gradient flow through the update). | Call `apply_update`, inspect `delta_w`. | `delta_w.requires_grad == False` and `delta_w.grad_fn is None`. | Exact property check |
| 3.15 | `test_fp32_weight_updates` | Weight updates are computed in fp32 even if model weights are fp16. | Create fp16 weights. Run update. | Intermediate `delta_w` computation uses fp32 (check inside the function or verify result precision). | `delta_w.dtype == torch.float32` before casting back |

---

## 4. TestPlasticityDiagnostics (~10 tests)

Tests the `PlasticityTrace` diagnostics object covering per-modulator statistics, per-layer norms,
effective learning rate computation, serialization, and overhead control.

**Fixtures**: `trace_module`, `neuromod_gate`, `three_factor`, `device`.

| # | Test Name | Description | Setup | Assertion | Tolerance |
|---|---|---|---|---|---|
| 4.1 | `test_per_modulator_stats` | Mean and std of each modulator are logged correctly in the diagnostics object. | Run forward pass with `return_details=True`. Extract `trace_log.modulator_stats`. | `"DA"` in stats; `stats["DA"]["mean"]` is a float; `stats["DA"]["std"] >= 0`. Check all four modulators present. | N/A (type and key checks) |
| 4.2 | `test_eligibility_norm_per_layer` | Eligibility trace L2 norm is logged per registered layer. | Register two eligible layers. Run update with `return_details=True`. | `trace_log.eligibility_norms` is a dict with two keys matching layer names; values are non-negative floats. | Values >= 0 |
| 4.3 | `test_effective_learning_rate` | Effective learning rate equals `lr * global_plasticity` and is logged. | `lr=0.001`, `global_plasticity=0.5`. | `trace_log.effective_lr` approximately equals `0.0005`. | atol=1e-6 |
| 4.4 | `test_update_norm_per_layer` | L2 norm of delta_w is logged per layer. | Run update with `return_details=True`. | `trace_log.update_norms` is a dict; values are non-negative floats; at least one is > 0 if mod_signal != 0. | Values >= 0 |
| 4.5 | `test_clamp_hit_count` | Number of times weight or trace clamps activate is tracked. | Config: `weight_clamp=[-0.1, 0.1]`. Apply large updates. | `trace_log.clamp_hits > 0`. | Strict positive |
| 4.6 | `test_plasticity_trace_object` | PlasticityTrace stores modulator values at each timestep as a list. | Run 5 update steps with `return_details=True`, collect traces. | `len(trace_log.modulator_history) == 5`; each entry contains DA/ACh/NE/5HT values. | Length match |
| 4.7 | `test_top_updated_synapses` | Top-K synapses by absolute delta_w magnitude are identified. | Run update, request top-5 synapses. | `trace_log.top_synapses` is a list of length 5; each entry contains `(layer, index, abs_delta_w)` sorted descending. | Sorted order check |
| 4.8 | `test_eligibility_decay_curves` | Compressed summary of eligibility decay matches raw trace data at sampled time points. | Run 50 steps, record raw trace norms. Request decay curve summary. | Summary values at sampled time indices match raw norms within tolerance. | atol=1e-4 |
| 4.9 | `test_json_serializable` | All diagnostics fields can be serialized to JSON without error. | Run update with `return_details=True`. Call `json.dumps(trace_log.to_dict())`. | No `TypeError` raised; output is a valid JSON string. | N/A (no exception) |
| 4.10 | `test_return_details_false_no_overhead` | When `return_details=False`, the trace log is None and no extra computation is performed. | Run update with `return_details=False`. | Return value for trace_log is `None`. Verify via timing that `return_details=False` is faster than `True` (or simply that the object is None). | `trace_log is None` |

---

## 5. TestConfig (~8 tests)

Tests configuration dataclasses, presets, validation, and serialization round-trips.

**Fixtures**: None (all tests construct configs directly).

| # | Test Name | Description | Setup | Assertion | Tolerance |
|---|---|---|---|---|---|
| 5.1 | `test_default_instantiation` | All config dataclasses instantiate with default values without error. | `EligibilityConfig()`, `NeuromodConfig()`, `ThreeFactorConfig()`, `PlasticityFullConfig()`. | No exception raised; all fields have non-None values. | N/A |
| 5.2 | `test_presets` | `minimal()`, `dev()`, and `production()` factory methods create valid configs. | `PlasticityFullConfig.minimal()`, `.dev()`, `.production()`. | Each returns a `PlasticityFullConfig` instance; `production` has larger dims than `minimal`. | Type check + field comparison |
| 5.3 | `test_eligibility_trace_type_values` | `trace_type` only accepts valid values. | Attempt `EligibilityConfig(trace_type="accumulating")`, `"replacing"`, `"dutch"`. Then attempt `"invalid"`. | Valid values succeed; `"invalid"` raises `ValueError`. | Exception check |
| 5.4 | `test_neuromod_combination_fn_values` | `combination_fn` only accepts valid values. | Attempt `"weighted_sum"`, `"gated_product"`, `"mlp"`. Then attempt `"invalid"`. | Valid values succeed; `"invalid"` raises `ValueError`. | Exception check |
| 5.5 | `test_three_factor_mode_values` | `mode` only accepts valid values. | Attempt `"online"`, `"hybrid"`, `"auxiliary_loss"`. Then attempt `"invalid"`. | Valid values succeed; `"invalid"` raises `ValueError`. | Exception check |
| 5.6 | `test_serialization_round_trip` | `to_dict()` followed by `from_dict()` produces an identical config object. | `cfg = PlasticityFullConfig.dev()`. `d = cfg.to_dict()`. `cfg2 = PlasticityFullConfig.from_dict(d)`. | `cfg == cfg2` (all fields match). | Exact equality |
| 5.7 | `test_field_validation_positive` | `tau_e > 0` and `lr > 0` are enforced. | Attempt `EligibilityConfig(tau_e=0)`, `EligibilityConfig(tau_e=-1)`, `ThreeFactorConfig(lr=0)`. | Each raises `ValueError`. | Exception check |
| 5.8 | `test_clamp_range_order` | `clamp_range[0] < clamp_range[1]` is enforced. | Attempt `EligibilityConfig(clamp_range=[5.0, -5.0])`. | Raises `ValueError` with message indicating min must be less than max. | Exception check |

---

## 6. TestDelayedRewardAssociation (~6 tests)

End-to-end learning tests on a toy task: a stimulus is presented, then after a configurable delay
of N steps with no input, a reward arrives. The model must learn the stimulus-reward association.
These tests validate that eligibility traces bridge the temporal credit assignment gap.

**Fixtures**: `device`. Each test constructs its own small model internally.

**Model setup**: A two-layer linear network (input_dim=8, hidden=32, output=2) with eligibility
traces on the hidden-to-output weights. Neuromodulatory gate receives the reward signal as DA.
Training loop: 200 episodes, delay=5 steps (unless stated otherwise).

**Skip**: These tests are marked `@pytest.mark.slow` (each runs ~200 training episodes).

| # | Test Name | Description | Setup | Assertion | Tolerance |
|---|---|---|---|---|---|
| 6.1 | `test_toy_task_setup` | The toy delayed reward task is correctly configured: stimulus at t=0, reward at t=delay, no input in between. | Instantiate task with delay=5. Step through one episode. | Stimulus nonzero at t=0; inputs zero for t in 1..4; reward nonzero at t=5. | Exact structure |
| 6.2 | `test_without_eligibility_fails` | Without eligibility traces, the model cannot learn when reward is delayed by N steps (baseline control). | Train model with three-factor disabled (standard backprop only, reward at t=delay). 200 episodes. | Final accuracy is at or near chance level (50% for binary, so accuracy < 0.6). Loss does not decrease significantly. | accuracy < 0.6 |
| 6.3 | `test_with_eligibility_learns` | With eligibility traces active, the model learns the delayed stimulus-reward association. | Train model with three-factor enabled, same task. 200 episodes, delay=5. | Final accuracy > 0.8. | accuracy > 0.8 |
| 6.4 | `test_loss_decreases` | Training loss decreases over the course of training with eligibility traces. | Record loss at episodes 1-10 (early) and episodes 191-200 (late). | `mean(loss_late) < mean(loss_early)`. | Strict inequality |
| 6.5 | `test_accuracy_above_chance` | Final accuracy is statistically above chance (p < 0.05 via binomial test or threshold). | Evaluate on 100 test episodes after training. | Accuracy > 0.65 (well above 0.5 chance for binary task). | Threshold 0.65 |
| 6.6 | `test_longer_delay_still_works` | With longer delay (N=10, N=20), learning still occurs but converges more slowly. | Train with delay=10 for 400 episodes; delay=20 for 800 episodes. | Both achieve accuracy > 0.7. `epochs_to_70pct[delay=20] > epochs_to_70pct[delay=10] > epochs_to_70pct[delay=5]`. | accuracy > 0.7; monotonic epoch ordering |

---

## 7. TestIntegration (~8 tests)

End-to-end integration tests verifying the full pipeline from pre/post activity through eligibility
computation, neuromodulator gating, weight update, and interaction with other brain_ai modules.

**Fixtures**: `trace_module`, `neuromod_gate`, `three_factor`, `random_weights`, `device`.

**Skip**: SNN integration test skipped if `brain_ai.core` not importable. AMP test skipped without CUDA.

| # | Test Name | Description | Setup | Assertion | Tolerance |
|---|---|---|---|---|---|
| 7.1 | `test_full_pipeline` | Pre/post activity flows through eligibility traces, then through neuromodulator gate, then produces weight update via three-factor rule. | Wire all three modules. `pre -> post -> trace_module.update() -> neuromod_gate.forward() -> three_factor.apply_update()`. | `delta_w` has correct shape matching weights; `delta_w` is nonzero when mod_signal is nonzero; pipeline completes without error. | Shape check + non-zero check |
| 7.2 | `test_snn_integration` | Spike-based eligibility traces operate correctly on SNN layer weights. | Import `brain_ai.core.SNN` (or equivalent). Create SNN layer, register its weights as eligible. Use spike outputs as pre/post. | Traces are computed from actual spike trains; weight updates are applied to SNN synapses; SNN forward pass still works after update. | Functional correctness (no error, shapes match) |
| 7.3 | `test_fast_memory_adapter` | Adapter weights are updated by three-factor rule while main (backbone) weights remain frozen. | Create a linear layer (main) with a small adapter layer. Register only adapter as eligible. Run update. | Adapter weights change; main layer weights are bitwise identical to initial values. | `torch.equal` for main weights; `not torch.equal` for adapter weights |
| 7.4 | `test_multiple_eligible_layers` | Multiple eligible layers are each updated independently with their own traces and potentially different configs. | Register 3 layers as eligible, each with different lr. Run single update cycle. | Each layer has its own nonzero delta_w; magnitudes differ according to lr ratios. | Shape checks + lr ratio within rtol=0.2 |
| 7.5 | `test_checkpoint_round_trip` | Full state (traces, modulator baselines, step counters) can be saved and restored. | Build up state over 10 steps. Save via `state_dict()`. Create fresh modules. Load via `load_state_dict()`. Run one more step on both. | Outputs from original and restored modules are identical. | atol=1e-6 |
| 7.6 | `test_episode_boundary_reset` | Calling `reset()` at episode boundaries clears all traces, preventing cross-episode contamination. | Run 10 steps of episode 1 (building up traces). Call `reset()`. Check traces. Run step 1 of episode 2. | After reset: `torch.all(e == 0)`. Episode 2 traces start fresh (identical to a module that never saw episode 1). | Exact zero after reset; atol=1e-6 for fresh-vs-reset comparison |
| 7.7 | `test_streaming_carry_mode` | In streaming mode (no reset between calls), traces carry forward and accumulate correctly. | Call `update()` with chunk_1 data. Then `update()` with chunk_2 data (no reset in between). Compare to single-pass with concatenated data. | Traces after chunk_2 reflect history from chunk_1 (nonzero initial state). Results are consistent with manual sequential computation. | atol=1e-5 |
| 7.8 | `test_amp_no_nan` | Full pipeline produces no NaN values under AMP mixed precision. | Wrap entire pipeline in `torch.cuda.amp.autocast()`. Run 20 steps with random inputs. | All outputs (traces, modulators, delta_w, updated weights) pass `torch.isfinite()`. | No NaN, no Inf |

**Skip**: Test 7.2 skipped if `brain_ai.core` is not importable. Test 7.8 skipped without CUDA.

---

## Summary

| Test Class | Count | Key Invariant Tested |
|---|---|---|
| TestEligibilityTraces | 18 | Trace dynamics: accumulation, decay, STDP, clamping, determinism |
| TestNeuromodulatoryGate | 15 | Modulator computation: ranges, combination functions, state tracking |
| TestThreeFactorUpdate | 15 | Gating semantics: zero mod = zero update, clamping, mode variants |
| TestPlasticityDiagnostics | 10 | Observability: logging, serialization, overhead control |
| TestConfig | 8 | Configuration: validation, presets, serialization round-trip |
| TestDelayedRewardAssociation | 6 | End-to-end learning: temporal credit assignment via traces |
| TestIntegration | 8 | Pipeline correctness: multi-module wiring, checkpointing, AMP |
| **Total** | **80** | |

### Running the Tests

```bash
# All neuromodulation tests
python -m pytest tests/test_neuromodulation.py -v

# Skip slow delayed-reward tests
python -m pytest tests/test_neuromodulation.py -v -m "not slow"

# Only the three done-when gate tests
python -m pytest tests/test_neuromodulation.py -v -k "test_zero_modulator_zero_update or test_determinism_across_runs or test_with_eligibility_learns"

# With coverage
python -m pytest tests/test_neuromodulation.py --cov=brain_ai.meta --cov-report=term-missing
```

### Test Generation

The test file is generated by `scripts/gen_neuromod_tests.py`, which reads this matrix and produces
`tests/test_neuromodulation.py` with all fixtures, parametrization, and skip decorators. Manual edits
to the generated file are overwritten on re-generation -- modify this matrix instead.
