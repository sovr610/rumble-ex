---
name: Affective State Estimator (Valence-Arousal + Appraisal + Modulation)
description: >-
  This skill should be used when the user asks to "add emotion layer",
  "affective state estimation", "valence-arousal model",
  "emotional modulation of learning", "appraisal theory computation",
  "mood-congruent processing", "amygdala-inspired module",
  "somatic marker", "intrinsic motivation signal",
  "curiosity-driven exploration", "frustration detection",
  "reward prediction affect", "homeostatic regulation",
  "implement affective modulator", "add valence arousal space",
  "implement appraisal module", "add emotional bias to decisions",
  "implement somatic marker hypothesis", "add intrinsic reward signal",
  "implement dimensional emotion model", "add affect-modulated learning rate",
  "implement curiosity-anxiety tradeoff", "add frustration-driven exploration",
  or mentions affective state, valence-arousal, appraisal theory, emotional modulation,
  somatic markers, intrinsic motivation, mood-congruent processing,
  or homeostatic regulation in the cognitive pipeline.
version: 0.1.0
---

# Affective State Estimator (Valence-Arousal + Appraisal + Modulation)

## Purpose

This skill standardizes the "emotional coloring" layer of the cognitive architecture:
estimate the system's affective state from internal signals (prediction error, uncertainty,
reward, novelty, homeostatic deviation) and use that state to modulate learning rates,
attention gain, exploration temperature, memory consolidation priority, and risk sensitivity
in downstream modules. Biological brains do not process information neutrally -- emotions
shape what is learned, what is attended to, what is remembered, and what actions are taken.
This module provides the computational analog.

The affective estimator does NOT generate subjective experience. It computes a
low-dimensional affective vector (valence, arousal, dominance) from observable internal
signals and exposes it as a modulation surface for other modules. The non-negotiable goals
are deterministic state estimation, bounded outputs, and correct modulation directionality
(positive valence should increase exploitation, high arousal should increase learning rate).

## Key Files

| Target Module | Template Asset | Purpose |
|---|---|---|
| `brain_ai/affect/estimator.py` | `assets/affective_estimator_template.py` | AffectiveEstimator: signal intake, dimensional mapping, state output |
| `brain_ai/affect/appraisal.py` | `assets/appraisal_module_template.py` | AppraisalModule: relevance, congruence, coping potential computation |
| `brain_ai/affect/modulator.py` | `assets/affective_modulator_template.py` | AffectiveModulator: learning rate, attention, exploration, memory modulation |
| `brain_ai/affect/space.py` | `assets/valence_arousal_template.py` | ValenceArousalSpace: dimensional model math, circumplex geometry |
| `brain_ai/config.py` (extend) | `assets/affective_config_template.py` | AffectiveConfig, AppraisalConfig, ModulationConfig |

## Public Contract

```python
# AffectiveEstimator
reset(batch_size: int, device: torch.device) -> AffectiveState
estimate(signals: InternalSignals, state: AffectiveState = None) -> AffectiveState
get_modulation(state: AffectiveState) -> ModulationVector

# AppraisalModule
appraise(signals: InternalSignals) -> AppraisalResult

# AffectiveModulator
modulate_learning_rate(base_lr: float, state: AffectiveState) -> float
modulate_attention(attention_logits: Tensor, state: AffectiveState) -> Tensor
modulate_exploration(temperature: float, state: AffectiveState) -> float
modulate_memory_priority(priority: Tensor, state: AffectiveState) -> Tensor
```

Input `signals` is an `InternalSignals` dataclass carrying reward prediction error, epistemic
uncertainty, prediction error magnitude, novelty score, and homeostatic deviation, each `(B,)`
or `(B, 1)`. Output `AffectiveState` carries valence `(B,)`, arousal `(B,)`, and dominance
`(B,)`, all in `[-1, 1]`.

## Core Output Contract

| Field | Shape / Type | Description |
|---|---|---|
| `valence` | `(B,)` | Pleasure-displeasure axis, `[-1, 1]` |
| `arousal` | `(B,)` | Activation-deactivation axis, `[-1, 1]` |
| `dominance` | `(B,)` | Control-submission axis, `[-1, 1]` |
| `appraisal` | `AppraisalResult` | Relevance, congruence, coping scores |
| `modulation` | `ModulationVector` | Learning rate scale, attention gain, exploration temp, memory priority |
| `mood` | `(B, 3)` | Exponentially smoothed (V, A, D) for mood-congruent effects |
| `step_count` | `int` | Number of estimation steps since reset |

**Hard invariants**:
- All affective dimensions are bounded: `|valence| <= 1`, `|arousal| <= 1`, `|dominance| <= 1`.
- Modulation factors are positive: `learning_rate_scale > 0`, `attention_gain > 0`, `exploration_temp > 0`.
- Zero input signals produce neutral affect: `valence = 0, arousal = 0, dominance = 0`.
- Deterministic: same `InternalSignals` sequence produces identical `AffectiveState` sequence.
- All computation in fp32 for numerical stability under AMP.

## Dimensional Model (Russell's Circumplex + PAD)

The affective space uses three dimensions from the PAD (Pleasure-Arousal-Dominance) model:

| Dimension | Biological Correlate | Internal Signal Source |
|---|---|---|
| Valence (V) | DA reward prediction error | `reward_prediction_error` via tanh mapping |
| Arousal (A) | NE / LC tonic firing rate | `prediction_error_magnitude + uncertainty` via sigmoid |
| Dominance (D) | PFC engagement / coping | `coping_potential - challenge_level` via tanh |

The three dimensions span a unit cube `[-1, 1]^3`. Named emotional states can be identified
as regions in this space (Russell's circumplex is the V-A plane projection):

| Region | V | A | D | Cognitive Effect |
|---|---|---|---|---|
| Happy/excited | + | + | + | Exploit, consolidate, broaden attention |
| Calm/content | + | - | + | Maintain, low learning rate, narrow focus |
| Anxious/fearful | - | + | - | Explore, high learning rate, vigilant attention |
| Sad/fatigued | - | - | - | Disengage, low learning rate, perseverate |
| Frustrated | - | + | + | Increase exploration, try new strategies |
| Curious | 0 | + | + | Epistemic exploration, high learning rate |

## Appraisal Theory (Scherer's Component Process Model)

The appraisal module computes four appraisal checks that determine the affective response:

| Check | Computation | Maps To |
|---|---|---|
| Relevance | `sigmoid(abs(prediction_error) + novelty)` | Arousal magnitude |
| Congruence | `tanh(reward_prediction_error)` | Valence sign and magnitude |
| Coping potential | `sigmoid(1.0 - uncertainty)` | Dominance |
| Norm compatibility | `sigmoid(homeostatic_deviation)` | Valence modulation |

The appraisal results feed directly into the dimensional model:
- `valence = w_c * congruence + w_n * (1 - norm_deviation)`
- `arousal = w_r * relevance`
- `dominance = w_p * coping_potential`

All weights are configurable with sensible defaults summing to 1.0 per dimension.

## Internal Signal Sources

| Signal | Source Module | Range | Biological Analog |
|---|---|---|---|
| `reward_prediction_error` | Active inference EFE / TD error | `(-inf, inf)` | DA phasic signal |
| `epistemic_uncertainty` | Transition model ensemble disagreement | `[0, inf)` | ACh / uncertainty |
| `prediction_error_magnitude` | Sensory prediction error norm | `[0, inf)` | NE / surprise |
| `novelty_score` | HTM anomaly score or embedding distance | `[0, 1]` | Hippocampal novelty |
| `homeostatic_deviation` | Distance from homeostatic setpoints | `[0, inf)` | Hypothalamic signals |

## Modulation Targets

The affective state modulates five downstream systems:

| Target | Modulation Rule | Directionality |
|---|---|---|
| Learning rate | `lr_scale = base * (1 + alpha * arousal)` | High arousal -> faster learning |
| Attention gain | `gain = base * (1 + beta * abs(valence))` | Strong affect -> sharper attention |
| Exploration temperature | `temp = base * (1 + gamma * (1 - dominance))` | Low dominance -> more exploration |
| Memory consolidation | `priority = base * (1 + delta * arousal * abs(valence))` | Emotional events prioritized |
| Risk sensitivity | `risk_scale = base * (1 - epsilon * valence)` | Negative valence -> risk averse |

Each modulation has configurable gain, floor, and ceiling to prevent runaway amplification.

## Somatic Marker Hypothesis (Damasio)

The somatic marker mechanism biases decision-making through accumulated affect associations.
When the active inference agent evaluates candidate action sequences, the affective modulator
adds a "somatic marker" bonus/penalty to EFE scores based on the predicted affective
consequences of each trajectory:

```
efe_adjusted = efe_total + somatic_weight * predicted_valence(trajectory)
```

This creates a fast, affect-based pruning of action space before full EFE evaluation,
analogous to how gut feelings guide human decision-making.

## Integration with Neuromodulation System

The affective estimator maps bidirectionally to the existing neuromodulatory signals:

| Neuromodulator | Affective Dimension | Direction |
|---|---|---|
| DA (Dopamine) | Valence | DA -> V (positive RPE -> positive valence) |
| NE (Norepinephrine) | Arousal | NE -> A (high NE -> high arousal) |
| 5-HT (Serotonin) | Dominance (inverse) | 5-HT -> D (high 5-HT -> patience -> high dominance) |
| ACh (Acetylcholine) | Arousal (epistemic) | ACh -> A (novelty-driven arousal component) |

The affective estimator consumes neuromodulatory signals as inputs and produces modulation
targets that in turn influence neuromodulator dynamics, creating a feedback loop that
stabilizes at an emotional equilibrium.

## Mood and Temporal Dynamics

Mood is distinguished from momentary affect by its temporal scale:

- **Affect**: instantaneous response to current signals, changes every step
- **Mood**: exponentially weighted moving average of affect, `tau_mood` steps time constant

```
mood(t) = (1 - 1/tau_mood) * mood(t-1) + (1/tau_mood) * affect(t)
```

Mood provides mood-congruent processing: current mood biases appraisal of ambiguous signals.
When mood is negative, ambiguous prediction errors are appraised more negatively (congruence
bias). This creates self-reinforcing mood episodes that eventually decay back to neutral.

## Configuration Surface

### AffectiveConfig

| Field | Default | Purpose |
|---|---|---|
| `valence_scale` | 1.0 | Scaling for valence computation |
| `arousal_scale` | 1.0 | Scaling for arousal computation |
| `dominance_scale` | 1.0 | Scaling for dominance computation |
| `tau_mood` | 100 | Mood EMA time constant (steps) |
| `mood_congruence_bias` | 0.1 | Strength of mood-congruent appraisal bias |
| `output_clamp` | 1.0 | Hard clamp on all affective dimensions |
| `use_somatic_markers` | True | Enable somatic marker decision bias |
| `somatic_marker_weight` | 0.1 | Weight of somatic marker in EFE adjustment |
| `hidden_dim` | 64 | Hidden layer width in learned mappings |

### AppraisalConfig

| Field | Default | Purpose |
|---|---|---|
| `relevance_weight` | 1.0 | Weight of relevance in arousal |
| `congruence_weight` | 1.0 | Weight of congruence in valence |
| `coping_weight` | 1.0 | Weight of coping potential in dominance |
| `norm_weight` | 0.5 | Weight of norm compatibility in valence |
| `use_learned_appraisal` | False | Use MLP instead of analytic appraisal |

### ModulationConfig

| Field | Default | Purpose |
|---|---|---|
| `lr_alpha` | 0.5 | Learning rate arousal sensitivity |
| `lr_floor` | 0.1 | Minimum learning rate scale |
| `lr_ceiling` | 3.0 | Maximum learning rate scale |
| `attention_beta` | 0.3 | Attention valence sensitivity |
| `attention_floor` | 0.5 | Minimum attention gain |
| `attention_ceiling` | 2.0 | Maximum attention gain |
| `exploration_gamma` | 0.5 | Exploration dominance sensitivity |
| `exploration_floor` | 0.1 | Minimum exploration temperature scale |
| `exploration_ceiling` | 3.0 | Maximum exploration temperature scale |
| `memory_delta` | 0.5 | Memory consolidation affect sensitivity |
| `risk_epsilon` | 0.3 | Risk sensitivity valence scale |

Presets: `AffectiveFullConfig.minimal()`, `.dev()`, `.production()`.

## Done-When Gates

| Gate | Test | Threshold |
|---|---|---|
| **(a) Bounded outputs** | Feed extreme signals (1e6 magnitude); assert all affective dims in `[-1, 1]` and all modulation factors positive | Exact bound compliance |
| **(b) Neutral on zero input** | Feed zero signals; assert `valence == 0, arousal == 0, dominance == 0` within `atol=1e-6` | Exact neutral |
| **(c) Modulation directionality** | Verify high arousal -> higher learning rate, negative valence -> higher risk sensitivity, low dominance -> more exploration | Monotonic relationship |

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Valence saturates at +1 or -1 | No clamping or tanh on raw RPE | Apply tanh with configurable scale |
| Arousal always maximal | Prediction error not normalized | Normalize PE by running statistics |
| Mood never returns to neutral | tau_mood too large or no decay | Verify EMA decay is applied, reduce tau |
| Learning rate explodes | No ceiling on lr_scale | Enforce lr_ceiling clamp |
| Somatic markers dominate EFE | somatic_marker_weight too high | Reduce weight, verify scale relative to EFE |
| Mood-congruent bias creates stuck loops | Congruence bias too strong | Reduce mood_congruence_bias, add decay |
| NaN under AMP | fp16 accumulation in mood EMA | Force fp32 for all affect computation |
| Curiosity and anxiety conflated | Both from uncertainty without sign | Separate epistemic (controllable) from aleatoric uncertainty |

## Anti-Patterns

- **Unbounded affective dimensions** -- always clamp to `[-1, 1]`; unbounded affect creates cascading instability
- **fp16 mood accumulation** -- mood EMA needs fp32 precision over long horizons
- **Hardcoded modulation gains** -- use ModulationConfig, not magic numbers in downstream modules
- **Ignoring dominance dimension** -- dominance (coping/control) is critical for exploration-exploitation balance
- **Symmetric arousal effects** -- arousal should increase learning rate regardless of valence sign
- **No floor on modulation factors** -- zero learning rate or zero attention gain kills the system
- **Testing only with positive rewards** -- negative valence paths are where bugs hide
- **Storing mood on module** -- pass mood explicitly in AffectiveState, never on `self`

## Additional Resources

### Reference Files

- **`references/dimensional-models.md`** -- Russell's circumplex, PAD model, discrete vs dimensional debate, mathematical formalization
- **`references/appraisal-computation.md`** -- Scherer's appraisal theory, computational implementation, signal mapping
- **`references/modulation-targets.md`** -- How affect modulates learning, attention, exploration, memory, risk sensitivity
- **`references/neuroscience-grounding.md`** -- Amygdala, insula, OFC, ACC roles in affect, somatic marker hypothesis
- **`references/testing-matrix.md`** -- All test cases: bounded outputs, neutral on zero, directionality, dynamics, integration

### Asset Templates

- **`assets/affective_estimator_template.py`** -- AffectiveEstimator: signal intake, dimensional mapping, state output, self-test
- **`assets/appraisal_module_template.py`** -- AppraisalModule: relevance, congruence, coping, norm compatibility, self-test
- **`assets/affective_modulator_template.py`** -- AffectiveModulator: learning rate, attention, exploration, memory, risk, self-test
- **`assets/valence_arousal_template.py`** -- ValenceArousalSpace: dimensional model math, circumplex geometry, self-test
- **`assets/affective_config_template.py`** -- All configs, presets, serialization, validation, self-test

### Scripts

- **`scripts/validate_affect.py`** -- Runtime contract validation (bounded outputs, neutral on zero, directionality)
- **`scripts/gen_affect_tests.py`** -- Generates `tests/test_affective_estimator.py` (~70+ test cases)
- **`scripts/affect_dynamics_benchmark.py`** -- Benchmark affect estimation throughput, mood dynamics, modulation speed
