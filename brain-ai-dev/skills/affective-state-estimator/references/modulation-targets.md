# Modulation Targets

This document details how affective state modulates five downstream cognitive systems:
learning rate, attention gain, exploration temperature, memory consolidation priority,
and risk sensitivity.

---

## 1. Learning Rate Modulation

### Biological Basis

High arousal (NE release) enhances synaptic plasticity in biological brains.
Emotionally salient events are learned faster. The locus coeruleus -- norepinephrine
system modulates gain on cortical processing, effectively increasing learning rate
during arousing events.

### Computational Rule

```python
def modulate_learning_rate(
    base_lr: float,
    arousal: Tensor,      # (B,), [-1, 1]
    alpha: float = 0.5,   # sensitivity
    floor: float = 0.1,   # minimum scale
    ceiling: float = 3.0, # maximum scale
) -> Tensor:
    """Scale learning rate by arousal level.

    High arousal -> faster learning. The relationship is monotonically
    increasing with arousal, using absolute arousal (both positive and
    negative arousal increase learning).

    Returns:
        (B,) tensor of learning rate scales, in [floor, ceiling].
    """
    scale = 1.0 + alpha * arousal.abs()
    scale = torch.clamp(scale, floor, ceiling)
    return base_lr * scale
```

### Properties

- **Directionality**: Higher absolute arousal -> higher learning rate
- **Symmetry**: Both positive arousal (excitement) and negative arousal (fear) increase learning
- **Bounds**: Scale is clamped to `[floor, ceiling]`, preventing zero or runaway learning
- **Neutral**: At arousal=0, scale=1.0 (no modulation)

### Interaction with Three-Factor Learning

When combined with the neuromodulation/eligibility traces skill:
- Arousal modulates the global plasticity gain `g`
- High arousal -> higher `g` -> larger weight updates from eligibility traces
- This amplifies the three-factor learning signal during emotionally significant events

---

## 2. Attention Gain Modulation

### Biological Basis

Emotional stimuli capture attention. The amygdala projects to visual and
prefrontal cortex, enhancing processing of emotionally relevant stimuli.
Both positive and negative affect increase attentional gain, but with
different spatial profiles (negative narrows, positive broadens).

### Computational Rule

```python
def modulate_attention(
    attention_logits: Tensor,  # (B, N) or (B, H, N, N)
    valence: Tensor,           # (B,), [-1, 1]
    arousal: Tensor,           # (B,), [-1, 1]
    beta: float = 0.3,        # sensitivity
    floor: float = 0.5,       # minimum gain
    ceiling: float = 2.0,     # maximum gain
) -> Tensor:
    """Modulate attention logits by affective state.

    Strong affect (high |valence|) increases attention sharpness.
    The gain scales the logits before softmax, effectively changing
    the temperature of the attention distribution.

    Returns:
        Modulated attention logits, same shape as input.
    """
    affect_magnitude = valence.abs() * arousal.abs()  # (B,)
    gain = 1.0 + beta * affect_magnitude
    gain = torch.clamp(gain, floor, ceiling)

    # Reshape gain for broadcasting
    while gain.dim() < attention_logits.dim():
        gain = gain.unsqueeze(-1)

    return attention_logits * gain
```

### Properties

- **Directionality**: Strong affect (positive or negative) sharpens attention
- **Interaction**: Valence and arousal multiply -- high arousal alone or high valence alone has moderate effect; both together have strong effect
- **Spatial profile**: This implementation uniformly scales all attention weights. A more sophisticated version could narrow attention (higher gain on top-k items) for negative affect
- **Bounds**: Gain clamped to prevent attention collapse or explosion

### Integration with Global Workspace

In the global workspace (broadcast) module:
- Attention gain modulates the competition for workspace access
- High emotional salience increases the probability that emotionally tagged representations win the competition
- This implements the "emotional attention capture" effect

---

## 3. Exploration Temperature Modulation

### Biological Basis

Fear and uncertainty drive exploration in novel environments (fight-or-flight
broadens behavioral repertoire). Conversely, contentment reduces exploration.
The dominance dimension is key: low control (low dominance) -> try new strategies.

### Computational Rule

```python
def modulate_exploration(
    base_temperature: float,
    dominance: Tensor,         # (B,), [-1, 1]
    valence: Tensor,           # (B,), [-1, 1]
    gamma: float = 0.5,       # dominance sensitivity
    gamma_v: float = 0.2,     # valence sensitivity (frustration)
    floor: float = 0.1,       # minimum temperature scale
    ceiling: float = 3.0,     # maximum temperature scale
) -> Tensor:
    """Scale exploration temperature by affective state.

    Low dominance (low coping) -> more exploration (higher temperature).
    Negative valence with high dominance (frustration) also increases
    exploration to try new strategies.

    Returns:
        (B,) tensor of temperature scales, in [floor, ceiling].
    """
    # Low dominance -> explore more
    dominance_factor = 1.0 + gamma * (1.0 - dominance) / 2.0

    # Frustration (negative valence + high dominance) -> explore more
    frustration = torch.clamp(-valence * dominance, 0.0, 1.0)
    frustration_factor = 1.0 + gamma_v * frustration

    scale = dominance_factor * frustration_factor
    scale = torch.clamp(scale, floor, ceiling)
    return base_temperature * scale
```

### Properties

- **Dominance pathway**: Low dominance (unable to cope) -> more random exploration
- **Frustration pathway**: High dominance but negative valence (frustrated, in control but failing) -> try different strategies
- **Content pathway**: High dominance + positive valence -> low exploration (exploit)
- **Bounds**: Temperature scale clamped, preventing zero exploration or pure noise

### Integration with Active Inference

In the active inference planning module:
- Exploration temperature modulates the `action_temperature` in the CEM planner
- Higher temperature -> broader search over action sequences
- This creates an affect-driven exploration-exploitation tradeoff that supplements the epistemic EFE term

---

## 4. Memory Consolidation Priority Modulation

### Biological Basis

Emotional events are better remembered. The amygdala modulates hippocampal
consolidation through noradrenergic and glucocorticoid pathways. Both positive
and negative emotional events receive priority encoding.

### Computational Rule

```python
def modulate_memory_priority(
    base_priority: Tensor,   # (B,) or (B, N)
    valence: Tensor,         # (B,), [-1, 1]
    arousal: Tensor,         # (B,), [-1, 1]
    delta: float = 0.5,     # sensitivity
    floor: float = 0.1,     # minimum priority
    ceiling: float = 3.0,   # maximum priority
) -> Tensor:
    """Scale memory consolidation priority by emotional significance.

    Emotionally significant events (high arousal * |valence|) get
    higher priority for consolidation in the engram memory system.

    Returns:
        Modulated priority, same shape as base_priority.
    """
    emotional_significance = arousal.abs() * valence.abs()  # (B,)
    scale = 1.0 + delta * emotional_significance
    scale = torch.clamp(scale, floor, ceiling)

    # Reshape for broadcasting
    while scale.dim() < base_priority.dim():
        scale = scale.unsqueeze(-1)

    return base_priority * scale
```

### Properties

- **Directionality**: High emotional significance -> higher memory priority
- **Symmetry**: Both positive and negative emotional events get priority
- **Arousal gating**: Arousal acts as a gate -- even strong valence without arousal does not boost priority
- **Bounds**: Priority clamped to prevent memory monopolization

### Integration with Engram Memory

In the engram conditional memory module:
- Memory priority modulates the `write_strength` when storing new engrams
- Higher priority events are written with stronger traces
- During retrieval, emotionally encoded memories have higher activation baseline

---

## 5. Risk Sensitivity Modulation

### Biological Basis

Negative affect increases risk aversion (loss aversion), while positive affect
promotes risk-seeking behavior. This asymmetry is documented in prospect theory
and the somatic marker hypothesis. The anterior insula and OFC mediate risk
evaluation, receiving projections from the amygdala.

### Computational Rule

```python
def modulate_risk_sensitivity(
    base_risk_scale: float,
    valence: Tensor,          # (B,), [-1, 1]
    epsilon: float = 0.3,    # sensitivity
    floor: float = 0.3,      # minimum risk scale
    ceiling: float = 2.0,    # maximum risk scale
) -> Tensor:
    """Scale risk sensitivity by valence.

    Negative valence -> more risk averse (higher risk penalty).
    Positive valence -> more risk tolerant (lower risk penalty).

    Returns:
        (B,) tensor of risk scales, in [floor, ceiling].
    """
    # Negative valence -> risk_scale > 1 (risk averse)
    # Positive valence -> risk_scale < 1 (risk tolerant)
    scale = 1.0 - epsilon * valence
    scale = torch.clamp(scale, floor, ceiling)
    return base_risk_scale * scale
```

### Properties

- **Directionality**: Negative valence -> higher risk penalty -> risk averse
- **Asymmetry**: Loss aversion is captured by the negative valence path
- **Neutral**: At valence=0, risk sensitivity is unmodified
- **Bounds**: Clamped to prevent negative risk sensitivity or extreme risk seeking

### Integration with Active Inference

In the active inference EFE computation:
- Risk sensitivity modulates the pragmatic term weight
- Higher risk sensitivity -> more weight on avoiding bad outcomes (higher pragmatic weight)
- This creates an affect-dependent preference asymmetry: when feeling bad, the agent avoids risk; when feeling good, it takes more chances

---

## 6. Modulation Vector Summary

All five modulations are combined into a `ModulationVector` dataclass:

```python
@dataclass
class ModulationVector:
    lr_scale: Tensor       # (B,), learning rate multiplier
    attention_gain: Tensor # (B,), attention logit multiplier
    exploration_temp: Tensor  # (B,), exploration temperature multiplier
    memory_priority: Tensor   # (B,), memory consolidation priority multiplier
    risk_scale: Tensor     # (B,), risk sensitivity multiplier
```

**Invariants**:
- All fields are positive (> 0)
- All fields default to 1.0 when affect is neutral
- Each field is independently clamped to its configured `[floor, ceiling]`
- The vector is computed deterministically from the affective state

---

## 7. Key References

- McGaugh, J. L. (2004). The amygdala modulates the consolidation of memories of emotionally arousing experiences. Annual Review of Neuroscience.
- Aston-Jones, G. & Cohen, J. D. (2005). An integrative theory of locus coeruleus-norepinephrine function. Annual Review of Neuroscience.
- Damasio, A. R. (1994). Descartes' Error: Emotion, Reason, and the Human Brain.
- Kahneman, D. & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. Econometrica.
- Vuilleumier, P. (2005). How brains beware: neural mechanisms of emotional attention. Trends in Cognitive Sciences.
