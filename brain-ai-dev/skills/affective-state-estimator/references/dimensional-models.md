# Dimensional Models of Affect

This document covers the theoretical foundations for the dimensional representation of
affective states used in the AffectiveEstimator. The core claim: emotions are not discrete
categories but positions in a continuous low-dimensional space.

---

## 1. Russell's Circumplex Model (1980)

James Russell proposed that all affective states can be mapped onto a two-dimensional
circular space defined by:

- **Valence** (horizontal axis): pleasure vs displeasure
- **Arousal** (vertical axis): activation vs deactivation

Key properties:
- Emotions are distributed around the circle, not clustered at discrete points
- Opposite emotions are at opposite poles (happy vs sad, excited vs calm)
- The center represents neutral affect
- Angle from origin = emotion type, distance from origin = intensity

### Circumplex Coordinates

| Emotion | Valence | Arousal | Angle (deg) |
|---|---|---|---|
| Excited | +0.7 | +0.7 | 45 |
| Happy | +0.9 | +0.3 | 18 |
| Content | +0.7 | -0.3 | -23 |
| Calm | +0.3 | -0.7 | -67 |
| Bored | -0.3 | -0.7 | -113 |
| Sad | -0.7 | -0.3 | -157 |
| Angry | -0.7 | +0.7 | 135 |
| Afraid | -0.5 | +0.9 | 119 |
| Surprised | +0.1 | +0.9 | 84 |

### Mathematical Formalization

```
affect_angle = atan2(arousal, valence)
affect_intensity = sqrt(valence^2 + arousal^2)
affect_intensity = clamp(affect_intensity, 0, 1)
```

For the circumplex projection from 3D PAD space:

```
circumplex_x = valence
circumplex_y = arousal
circumplex_r = sqrt(valence^2 + arousal^2)
circumplex_theta = atan2(arousal, valence)
```

---

## 2. PAD Model (Mehrabian & Russell, 1974)

The PAD (Pleasure-Arousal-Dominance) model extends the circumplex with a third dimension:

- **Pleasure (P)**: identical to valence
- **Arousal (A)**: identical to arousal
- **Dominance (D)**: sense of control vs being controlled

The dominance dimension distinguishes emotions that the circumplex conflates:
- Fear (V-, A+, D-) vs Anger (V-, A+, D+) -- both negative/aroused, but fear is submissive, anger is dominant
- Awe (V+, A+, D-) vs Pride (V+, A+, D+) -- both positive/aroused, different control

### PAD Space Geometry

The PAD space is a unit cube `[-1, 1]^3`. The volume is divided into 8 octants,
each corresponding to a broad emotional category:

| Octant | P | A | D | Category |
|---|---|---|---|---|
| +++ | + | + | + | Exuberant |
| ++- | + | + | - | Dependent (awe) |
| +-+ | + | - | + | Relaxed |
| +-- | + | - | - | Docile |
| -++ | - | + | + | Hostile (anger) |
| -+- | - | + | - | Anxious (fear) |
| --+ | - | - | + | Disdainful |
| --- | - | - | - | Bored/depressed |

---

## 3. Discrete vs Dimensional Debate

### Discrete (Basic Emotions) Approach

Ekman's basic emotions: happiness, sadness, fear, anger, disgust, surprise.
- Pro: Categorical labels are intuitive and easy to threshold
- Con: Categories are culturally biased, fuzzy boundaries, miss blends
- Con: Hard to compute gradients through discrete categories

### Dimensional Approach (Our Choice)

- Pro: Continuous, differentiable, gradient-friendly
- Pro: Naturally represents blended emotions (anxious excitement = V+, A+, D-)
- Pro: Low-dimensional (3D) but expressive
- Pro: Maps cleanly to neuromodulatory signals (DA -> V, NE -> A, 5-HT -> D)
- Con: Named emotions must be reconstructed from coordinates

### Resolution for This Architecture

We use the **dimensional model** as the internal representation because:
1. Continuous values are differentiable and can participate in gradient-based learning
2. The 3D space maps directly to neuromodulatory systems already in the architecture
3. Modulation targets (learning rate, attention, etc.) are continuous-valued
4. Named emotions can be post-hoc identified as regions in the space for interpretability

---

## 4. Mapping from Internal Signals to PAD

The computational challenge is mapping raw internal signals (which are
domain-specific scalars) to the three affective dimensions. We use a
combination of analytic mappings and optional learned projections.

### Analytic Mapping (Default)

```python
# Valence: primarily from reward prediction error
valence_raw = tanh(congruence_weight * reward_prediction_error)
valence_norm = valence_raw + norm_weight * (1 - tanh(homeostatic_deviation))
valence = clamp(valence_scale * valence_norm, -1, 1)

# Arousal: from prediction error magnitude and uncertainty
arousal_raw = sigmoid(prediction_error_magnitude + epistemic_uncertainty) * 2 - 1
arousal = clamp(arousal_scale * arousal_raw, -1, 1)

# Dominance: from coping potential (inverse of uncontrollable uncertainty)
dominance_raw = tanh(coping_potential - challenge_level)
dominance = clamp(dominance_scale * dominance_raw, -1, 1)
```

### Learned Mapping (Optional)

When `use_learned_appraisal=True`, an MLP replaces the analytic mapping:

```python
signal_vector = cat([rpe, uncertainty, pe_mag, novelty, homeostatic_dev])
pad_raw = mlp(signal_vector)  # (B, 3)
pad = tanh(pad_raw)  # bounded to [-1, 1]
```

The learned mapping is initialized to approximate the analytic mapping and
fine-tuned during training.

---

## 5. Temporal Smoothing and Mood

Instantaneous affect is noisy. Mood is a temporally smoothed version:

```
mood_t = (1 - alpha) * mood_{t-1} + alpha * affect_t
where alpha = 1 / tau_mood
```

Mood serves two purposes:
1. **Stability**: prevents rapid oscillation in modulation targets
2. **Congruence**: biases appraisal of ambiguous signals toward the current mood

The mood-congruent bias is implemented as an additive offset to the congruence
appraisal check:

```
congruence_biased = congruence + mood_congruence_bias * mood_valence
```

This creates self-reinforcing mood episodes: negative mood makes ambiguous events
seem more negative, prolonging the negative mood. The effect is bounded by the
decay constant and bias strength.

---

## 6. Key References

- Russell, J. A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology.
- Mehrabian, A. & Russell, J. A. (1974). An Approach to Environmental Psychology. MIT Press.
- Barrett, L. F. (2006). Are Emotions Natural Kinds? Perspectives on Psychological Science.
- Posner, J., Russell, J. A., & Peterson, B. S. (2005). The circumplex model of affect. Development and Psychopathology.
- Lindquist, K. A. & Barrett, L. F. (2012). A functional architecture of the human brain. Behavioral and Brain Sciences.
