# Neuroscience Grounding

This document provides the neuroscience basis for the affective state estimator,
mapping brain structures and circuits to computational components.

---

## 1. Amygdala: Relevance Detection and Emotional Learning

### Biological Role

The amygdala is a bilateral structure in the medial temporal lobe. It is the
primary hub for:
- **Relevance detection**: rapid evaluation of sensory stimuli for biological significance
- **Fear conditioning**: associating stimuli with aversive outcomes
- **Emotional memory**: modulating hippocampal consolidation of emotional events
- **Attention capture**: projections to visual cortex and prefrontal cortex

### Computational Analog

The **relevance check** in the AppraisalModule maps to amygdala function:

```
Amygdala input: sensory prediction error + novelty signal
Amygdala output: relevance score -> arousal dimension

Basolateral amygdala (BLA) -> learned associations (appraisal learning)
Central amygdala (CeA) -> autonomic/arousal output (arousal dimension)
BLA -> hippocampus -> memory priority modulation
BLA -> PFC -> attention gain modulation
```

### Key Properties

- **Speed**: the amygdala receives fast, coarse sensory input via the thalamic pathway
  (bypassing cortex), enabling rapid relevance detection. Our relevance check is
  similarly computed before full appraisal.
- **Plasticity**: BLA synapses undergo LTP during fear conditioning. Our learned
  appraisal module captures this with gradient-based updates.
- **Bidirectional PFC interaction**: PFC can suppress amygdala responses (emotion
  regulation). Our dominance dimension (PFC engagement) inversely modulates arousal
  effects.

---

## 2. Insula: Interoceptive Awareness and Somatic Markers

### Biological Role

The insular cortex (insula) processes:
- **Interoception**: awareness of internal body states (heartbeat, gut feelings)
- **Disgust**: primary gustatory and visceral disgust processing
- **Pain**: both physical and social pain processing
- **Homeostatic signals**: deviations from metabolic/physiological setpoints

### Computational Analog

The **homeostatic deviation** signal and **norm compatibility check** map to insular function:

```
Insula input: homeostatic deviation signals
Insula output: norm compatibility -> valence modulation

Anterior insula -> subjective feeling states (the "felt" quality of affect)
Posterior insula -> raw interoceptive signals (homeostatic deviation)
Anterior insula -> ACC -> decision-making bias (somatic markers)
```

### Somatic Marker Hypothesis (Damasio)

Antonio Damasio proposed that decision-making relies on "somatic markers" --
bodily feeling states associated with prior outcomes. When evaluating options,
the brain reactivates somatic markers from similar past situations, creating
a "gut feeling" that biases the decision before conscious deliberation.

**Computational implementation**:

```python
# During action evaluation in active inference:
for each candidate_trajectory:
    # Standard EFE computation
    efe = compute_efe(trajectory)

    # Somatic marker: predict affective consequence
    predicted_affect = estimator.predict_affect(trajectory)
    somatic_bonus = somatic_weight * predicted_affect.valence

    # Somatic marker biases EFE (lower EFE = preferred)
    efe_adjusted = efe - somatic_bonus
```

This creates fast, affect-based pruning of the action space. Trajectories
predicted to cause positive affect get a bonus (lower EFE), while those
predicted to cause negative affect get a penalty. This supplements the
explicit EFE computation with an implicit, experience-based preference.

---

## 3. Orbitofrontal Cortex (OFC): Value Representation and Emotion Regulation

### Biological Role

The OFC is critical for:
- **Subjective value**: encoding the subjective desirability of outcomes
- **Expectation updating**: revising reward expectations based on experience
- **Emotion regulation**: top-down modulation of amygdala responses
- **Reversal learning**: rapidly updating stimulus-outcome associations when contingencies change

### Computational Analog

The **congruence check** and **reward prediction error processing** map to OFC function:

```
OFC input: reward prediction error (from DA system)
OFC output: congruence assessment -> valence dimension

Medial OFC -> reward value representation (positive valence)
Lateral OFC -> punishment/loss representation (negative valence)
OFC -> amygdala -> emotion regulation (dominance modulation of arousal)
OFC -> ventral striatum -> action value updating
```

### Integration with Active Inference

The OFC is the biological substrate for the "pragmatic" term in EFE:
preferences about desired observations. The affective estimator's valence
dimension, driven by congruence with reward predictions, provides the
signal that shapes preference learning in the active inference module.

---

## 4. Anterior Cingulate Cortex (ACC): Conflict and Coping

### Biological Role

The ACC monitors:
- **Conflict detection**: incompatible response tendencies
- **Error monitoring**: discrepancy between expected and actual outcomes
- **Effort allocation**: deciding how much cognitive effort to invest
- **Pain processing**: both physical and social pain

### Computational Analog

The **coping potential check** and **dominance dimension** map to ACC function:

```
ACC input: conflict signals, prediction errors, effort costs
ACC output: coping assessment -> dominance dimension

Dorsal ACC (dACC) -> conflict/error monitoring (prediction error magnitude)
Subgenual ACC (sgACC) -> mood regulation (mood dynamics, tau_mood)
ACC -> dlPFC -> cognitive control engagement (dominance)
ACC -> LC -> norepinephrine release (arousal modulation)
```

### Effort-Based Decision Making

The ACC's role in effort allocation maps to our exploration temperature modulation:
- High conflict (low dominance) -> disengage automatic processing -> explore alternatives
- Low conflict (high dominance) -> maintain current strategy -> exploit

---

## 5. Neuromodulatory System Mapping

The four neuromodulators already implemented in the architecture map directly
to affective dimensions:

### Dopamine (DA) -> Valence

| DA Signal | Affective Effect | Mechanism |
|---|---|---|
| Positive RPE (DA burst) | Positive valence | Unexpected reward -> pleasure |
| Negative RPE (DA dip) | Negative valence | Reward omission -> displeasure |
| Tonic DA level | Mood baseline | Sustained DA -> positive mood bias |

**Bidirectional mapping**:
- DA signals drive valence (forward path)
- Valence modulates DA receptor sensitivity (feedback path via risk sensitivity)

### Norepinephrine (NE) -> Arousal

| NE Signal | Affective Effect | Mechanism |
|---|---|---|
| Phasic NE (LC burst) | High arousal | Surprise/urgency -> activation |
| Tonic NE (LC baseline) | Baseline arousal | Vigilance state |
| NE network reset | Exploration mode | High tonic NE -> broad attention |

**Bidirectional mapping**:
- NE signals drive arousal (forward path)
- Arousal modulates learning rate and attention gain (feedback path)

### Serotonin (5-HT) -> Dominance (Inverse)

| 5-HT Signal | Affective Effect | Mechanism |
|---|---|---|
| High 5-HT | High dominance | Patience, long-term planning |
| Low 5-HT | Low dominance | Impulsivity, short-term focus |
| 5-HT depletion | Anxiety/aggression | Loss of coping ability |

**Bidirectional mapping**:
- 5-HT signals drive dominance (forward path)
- Dominance modulates exploration temperature (feedback path)

### Acetylcholine (ACh) -> Arousal (Epistemic Component)

| ACh Signal | Affective Effect | Mechanism |
|---|---|---|
| High ACh | Epistemic arousal | Novelty-driven curiosity |
| ACh + high uncertainty | Anxiety | Uncontrollable uncertainty |
| ACh + low uncertainty | Curiosity | Controllable uncertainty |

**The curiosity-anxiety distinction**: Same signal (ACh/uncertainty), different
affective outcomes depending on dominance. High uncertainty + high dominance
(controllable) -> curiosity. High uncertainty + low dominance (uncontrollable)
-> anxiety. This emerges naturally from the PAD model without special-casing.

---

## 6. Circuit Summary

```
                    Sensory Input
                         |
                    [AMYGDALA] ---> Relevance ---> Arousal
                    /    |    \
                   /     |     \
              [OFC]   [ACC]   [INSULA]
              /          |          \
        Congruence   Coping    Norm Compat.
             \         |          /
              \        |         /
               [AFFECTIVE STATE]
              (V, A, D) = (Valence, Arousal, Dominance)
                    |
              [MODULATION]
             /    |    |    \
           LR   Attn  Expl  Memory  Risk
```

---

## 7. Key References

- LeDoux, J. E. (2000). Emotion circuits in the brain. Annual Review of Neuroscience.
- Damasio, A. R. (1996). The somatic marker hypothesis and the possible functions of the prefrontal cortex. Philosophical Transactions of the Royal Society B.
- Craig, A. D. (2009). How do you feel -- now? The anterior insula and human awareness. Nature Reviews Neuroscience.
- Rolls, E. T. (2019). The Orbitofrontal Cortex. Oxford University Press.
- Shenhav, A., Botvinick, M. M., & Cohen, J. D. (2013). The expected value of control. Neuron.
- Sara, S. J. (2009). The locus coeruleus and noradrenergic modulation of cognition. Nature Reviews Neuroscience.
- Dayan, P. & Huys, Q. J. M. (2009). Serotonin in affective control. Annual Review of Neuroscience.
- Yu, A. J. & Dayan, P. (2005). Uncertainty, neuromodulation, and attention. Neuron.
