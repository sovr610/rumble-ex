# Appraisal Computation

This document details the computational implementation of appraisal theory for the
AffectiveEstimator. Appraisal theory posits that emotions arise from cognitive evaluations
(appraisals) of events, not from the events themselves.

---

## 1. Scherer's Component Process Model

Klaus Scherer's model defines a sequence of appraisal checks that an organism performs
on incoming stimuli. Each check evaluates a different aspect of the stimulus and
contributes to the resulting affective state. We implement four checks:

| Check | Question | Internal Signal | Output |
|---|---|---|---|
| Relevance | Is this significant? | `abs(prediction_error) + novelty` | Arousal magnitude |
| Congruence | Does this match goals? | `reward_prediction_error` | Valence sign/magnitude |
| Coping potential | Can I handle this? | `1 - epistemic_uncertainty` | Dominance |
| Norm compatibility | Is this within expected bounds? | `homeostatic_deviation` | Valence modulation |

### Check Ordering

Scherer proposes these checks occur in sequence, with earlier checks gating later ones.
In our implementation, we compute all checks in parallel (for GPU efficiency) but apply
a relevance gate: if relevance is below threshold, arousal and downstream modulations
are suppressed.

```python
relevance = sigmoid(relevance_scale * (abs(pe) + novelty))
if relevance < relevance_threshold:
    arousal *= relevance / relevance_threshold  # soft gate
```

---

## 2. Signal Mapping Details

### 2a. Relevance Check

**Purpose**: Determine if the current situation demands attention.

**Inputs**: prediction error magnitude, novelty score

**Computation**:
```python
def compute_relevance(
    prediction_error_magnitude: Tensor,  # (B,), [0, inf)
    novelty_score: Tensor,               # (B,), [0, 1]
    scale: float = 1.0,
) -> Tensor:
    """Relevance check: how significant is the current situation?

    Returns:
        (B,) tensor in [0, 1]. Higher values indicate greater relevance.
    """
    raw = prediction_error_magnitude + novelty_score
    return torch.sigmoid(scale * raw)
```

**Properties**:
- Output in `[0, 1]`: 0 = irrelevant, 1 = maximally relevant
- Monotonically increasing with both PE magnitude and novelty
- Sigmoid provides saturation for extreme inputs
- The scale parameter controls sensitivity

### 2b. Congruence Check

**Purpose**: Determine if the situation is goal-congruent (good) or goal-incongruent (bad).

**Inputs**: reward prediction error

**Computation**:
```python
def compute_congruence(
    reward_prediction_error: Tensor,  # (B,), (-inf, inf)
    scale: float = 1.0,
    mood_bias: float = 0.0,           # mood-congruent bias term
) -> Tensor:
    """Congruence check: does this match goals?

    Returns:
        (B,) tensor in [-1, 1]. Positive = congruent, negative = incongruent.
    """
    biased_rpe = reward_prediction_error + mood_bias
    return torch.tanh(scale * biased_rpe)
```

**Properties**:
- Output in `[-1, 1]`: sign indicates direction, magnitude indicates certainty
- Positive RPE -> positive congruence -> positive valence
- The mood_bias term implements mood-congruent appraisal
- tanh provides symmetric saturation

### 2c. Coping Potential Check

**Purpose**: Estimate the system's ability to deal with the current situation.

**Inputs**: epistemic uncertainty (inverse of coping)

**Computation**:
```python
def compute_coping_potential(
    epistemic_uncertainty: Tensor,  # (B,), [0, inf)
    scale: float = 1.0,
) -> Tensor:
    """Coping potential: can the system handle this?

    Returns:
        (B,) tensor in [0, 1]. Higher values indicate greater coping ability.
    """
    # Low uncertainty -> high coping potential
    return torch.sigmoid(scale * (1.0 - epistemic_uncertainty))
```

**Properties**:
- Output in `[0, 1]`: 0 = no coping ability, 1 = full control
- Inverse relationship with uncertainty: high uncertainty = low coping
- Maps to dominance dimension: high coping -> high dominance

### 2d. Norm Compatibility Check

**Purpose**: Evaluate whether the situation deviates from homeostatic norms.

**Inputs**: homeostatic deviation

**Computation**:
```python
def compute_norm_compatibility(
    homeostatic_deviation: Tensor,  # (B,), [0, inf)
    scale: float = 1.0,
) -> Tensor:
    """Norm compatibility: is the situation within expected bounds?

    Returns:
        (B,) tensor in [0, 1]. Higher values indicate greater compatibility.
    """
    return torch.sigmoid(-scale * homeostatic_deviation + 2.0)
```

**Properties**:
- Output in `[0, 1]`: 1 = fully compatible, 0 = extreme deviation
- Modulates valence: large deviations push valence negative
- The +2.0 bias ensures that zero deviation maps to ~0.88 (mildly positive)

---

## 3. Appraisal to Affect Mapping

The four appraisal outputs combine to produce the three affective dimensions:

```python
def appraisal_to_affect(
    relevance: Tensor,         # (B,), [0, 1]
    congruence: Tensor,        # (B,), [-1, 1]
    coping_potential: Tensor,  # (B,), [0, 1]
    norm_compatibility: Tensor, # (B,), [0, 1]
    weights: AppraisalConfig,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Map appraisal results to PAD dimensions.

    Returns:
        (valence, arousal, dominance) each (B,) in [-1, 1].
    """
    # Valence: congruence + norm compatibility
    valence = (
        weights.congruence_weight * congruence
        + weights.norm_weight * (2.0 * norm_compatibility - 1.0)
    )
    valence = torch.clamp(valence, -1.0, 1.0)

    # Arousal: relevance (shifted to [-1, 1])
    arousal = weights.relevance_weight * (2.0 * relevance - 1.0)
    arousal = torch.clamp(arousal, -1.0, 1.0)

    # Dominance: coping potential (shifted to [-1, 1])
    dominance = weights.coping_weight * (2.0 * coping_potential - 1.0)
    dominance = torch.clamp(dominance, -1.0, 1.0)

    return valence, arousal, dominance
```

---

## 4. Learned Appraisal (Optional)

When `use_learned_appraisal=True`, an MLP replaces the analytic checks:

```python
class LearnedAppraisalModule(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Initialize to approximate analytic mapping
        self._init_weights()

    def forward(self, signals: Tensor) -> Tuple[Tensor, ...]:
        out = self.net(signals)
        relevance = torch.sigmoid(out[:, 0])
        congruence = torch.tanh(out[:, 1])
        coping = torch.sigmoid(out[:, 2])
        norm_compat = torch.sigmoid(out[:, 3])
        return relevance, congruence, coping, norm_compat
```

The learned module can capture nonlinear interactions between signals that the
analytic mapping misses, but requires training data (from the system's own
experience) to calibrate.

---

## 5. Computational Properties

### Differentiability

All appraisal functions use smooth, differentiable activations (sigmoid, tanh).
Gradients flow through the entire appraisal -> affect -> modulation chain,
allowing end-to-end optimization when desired.

### Determinism

Given identical inputs, the appraisal module produces identical outputs.
No randomness is used in appraisal computation. The learned variant is also
deterministic (no dropout during inference).

### Numerical Stability

- All intermediate computations in fp32
- Sigmoid and tanh are numerically stable for all input ranges
- The scale parameters prevent signal magnitudes from reaching extreme values

---

## 6. Key References

- Scherer, K. R. (2009). The dynamic architecture of emotion. Cognition and Emotion.
- Scherer, K. R. (2001). Appraisal considered as a process of multilevel sequential checking. In Appraisal Processes in Emotion.
- Lazarus, R. S. (1991). Emotion and Adaptation. Oxford University Press.
- Moors, A., Ellsworth, P. C., Scherer, K. R., & Frijda, N. H. (2013). Appraisal theories of emotion. Cognition and Emotion.
- Broekens, J., Jacobs, E., & Jonker, C. M. (2015). A reinforcement learning model of joy, distress, hope and fear. Connection Science.
