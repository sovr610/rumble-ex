# Reasoning Trace Visualization

## Overview

The brain_ai system implements dual-process reasoning inspired by Kahneman's System 1/System 2 framework. Visualizing reasoning traces is essential for understanding:

- **Routing decisions**: When does the system engage slow, deliberative reasoning (System 2) vs. fast, automatic processing (System 1)?
- **System 2 iteration traces**: How does the iterative GRU-based reasoning refine its representation over multiple steps?
- **Rule activation patterns**: Which symbolic rules fire and with what strength?
- **Confidence evolution**: How does the system's confidence change through the reasoning pipeline?

The key configuration parameters from `ReasoningConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 4096 | Reasoning hidden dimension |
| `num_reasoning_steps` | 16 | Maximum System 2 iterations |
| `confidence_threshold` | 0.8 | Threshold for System 1 vs. 2 routing |
| `num_entities` | 10000 | Knowledge base entity count |
| `num_predicates` | 1000 | Predicate space size |
| `logic_type` | "product" | Differentiable logic t-norm |
| `system2_layers` | 8 | System 2 GRU depth |
| `system2_heads` | 16 | System 2 attention heads |

---

## Dual-Process Routing Visualization

### The Routing Decision

At the entry point of the reasoning module (`DualProcessReasoner`), the system evaluates its confidence in the current workspace representation. If confidence exceeds the threshold (0.8), System 1 handles the input directly. Otherwise, System 2 engages for deeper processing.

### Routing Gauge Plot

Visualize the routing decision as a gauge or dial:

```
         System 1          |  System 2
    (fast, automatic)      | (slow, deliberative)
                           |
  0.0   0.2   0.4   0.6  0.8   1.0
  [=====|=====|=====|=====|=====|====]
                        ^
                     conf=0.65
                   -> SYSTEM 2
```

Implementation:

```python
fig, ax = plt.subplots(figsize=(10, 3))
# Draw gauge bar
ax.barh(0, confidence, height=0.4, color='#00bcd4', label='Confidence')
ax.barh(0, 1.0 - confidence, left=confidence, height=0.4,
        color='#e0e0e0')
ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
           label=f'Threshold ({threshold})')
# Annotate
route = 'System 1' if confidence >= threshold else 'System 2'
color = '#00bcd4' if confidence >= threshold else '#e91e63'
ax.text(confidence, 0, f' {confidence:.2f}', va='center',
        fontsize=12, fontweight='bold', color=color)
ax.set_xlim(0, 1)
ax.set_title(f'Routing Decision: {route}')
```

### Routing Distribution Over a Batch

For a batch of inputs, show the distribution of confidence values and routing decisions:

```
Count
  |  ####
  |  ####  ####
  |  ####  ####  ####
  |  ####  ####  ####  ####  ####
  +--+--+--+--+--+--+--+--+--+--+-->
  0.0  0.2  0.4  0.6  0.8  1.0
                        ^
                   Threshold
```

Color the histogram bars based on routing:
- **Cyan** (`#00bcd4`): System 1 (confidence >= threshold)
- **Magenta** (`#e91e63`): System 2 (confidence < threshold)

Annotate the fraction of inputs routed to each system:
```
System 1: 65% (high confidence)
System 2: 35% (requires deliberation)
```

### Routing Over Time

For sequential inputs, plot confidence as a time series with the threshold as a horizontal line:

```
Confidence
1.0 |    *  *     *  *  *
0.8 |---*---*---------*---*--- threshold
0.6 |  *         *
0.4 |       *  *    *
0.2 |
0.0 +-------------------------> Time
```

Color the background regions:
- Light cyan above threshold (System 1 active)
- Light magenta below threshold (System 2 active)

This reveals patterns: does the system become more confident over a sequence? Does it oscillate between systems?

---

## System 2 Iteration Trace Trees

### Step-by-Step Reasoning

When System 2 engages, it performs up to `num_reasoning_steps` (16) iterative refinements. At each step, the representation is updated via a GRU with multi-head attention. Key outputs per step:

- **Hidden state**: The evolving representation (4096-dim)
- **Step confidence**: Confidence at each iteration
- **Attention pattern**: What the GRU attends to
- **Residual magnitude**: How much the representation changes

### Confidence Evolution Plot

Plot confidence as a function of System 2 iteration:

```
Confidence
1.0 |                    *--*--*
0.9 |                *--*
0.8 |------------ threshold --------
0.7 |            *
0.6 |        *
0.5 |    *
0.4 | *
    +--+--+--+--+--+--+--+--+--+-->
    S0  S1  S2  S3  S4  S5  S6  S7
         System 2 Iteration
```

Mark the step where confidence first exceeds the threshold (early stopping point). If confidence never exceeds the threshold after all steps, flag this as a "hard case" with a warning annotation.

### Representation Drift

Track how the hidden representation changes across iterations using cosine similarity with the initial state:

```python
cos_sim[t] = F.cosine_similarity(h[t], h[0], dim=-1)
```

Plot:

```
Cosine Sim
1.0 | *
0.9 |  *
0.8 |   *
0.7 |    *
0.6 |     *  *
0.5 |        *  *  *
    +--+--+--+--+--+--+-->
    S0  S1  S2  S3  S4
```

A steep decline indicates the reasoning is making substantial changes. A plateau suggests convergence.

### Step Residual Norms

Plot the L2 norm of the update at each step:

```
||delta_h||
3.0 | *
2.0 |  *
1.0 |   *  *
0.5 |        *  *  *
0.0 +--+--+--+--+--+--+-->
    S0  S1  S2  S3  S4
```

Declining residual norms indicate convergence. A sudden spike suggests the reasoning encountered a difficulty requiring a large correction.

### Trace Tree Visualization

For complex reasoning, visualize the trace as a tree structure:

```
Root (conf=0.4)
  |
  +-- Step 1 (conf=0.5, delta=2.1)
  |     |
  |     +-- Attention: [WM_slot_0: 0.6, WM_slot_2: 0.3, ...]
  |
  +-- Step 2 (conf=0.6, delta=1.3)
  |     |
  |     +-- Attention: [WM_slot_1: 0.5, WM_slot_0: 0.4, ...]
  |
  +-- Step 3 (conf=0.82, delta=0.5) <-- CONVERGED
```

Render this as a vertical tree using matplotlib with:
- Nodes: Circles with confidence value
- Edges: Lines with residual norm annotation
- Color: Gradient from red (low confidence) to green (high confidence)

---

## Rule Activation Bar Charts

### Symbolic Rules in brain_ai

The neuro-symbolic engine uses differentiable logic (product t-norm by default). Rules are represented as predicate combinations with continuous truth values. When `ReasoningConfig.use_ltn = True`, Logic Tensor Networks provide grounded predicates.

### Rule Activation Display

For a given input, display the activation (truth value) of the top-K most active rules:

```
Rule                          Activation
is_animal(x) AND has_fur(x)     0.92  |===================|
is_moving(x) AND is_fast(x)    0.78  |===============     |
is_large(x)                     0.65  |============        |
is_dangerous(x)                 0.31  |======              |
is_domestic(x)                  0.12  |==                  |
```

Implementation:

```python
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(rules))
bars = ax.barh(y_pos, activations, color='steelblue')

# Color bars by activation level
for bar, val in zip(bars, activations):
    if val > 0.8:
        bar.set_color('#2ca02c')    # High: green
    elif val > 0.5:
        bar.set_color('#ff7f0e')    # Medium: orange
    else:
        bar.set_color('#d62728')    # Low: red

ax.set_yticks(y_pos)
ax.set_yticklabels(rules)
ax.set_xlabel('Activation (Truth Value)')
ax.set_title('Rule Activations')
ax.set_xlim(0, 1)
```

### Rule Activation Over Time

For sequential inputs, show how rule activations evolve:

```
Activation
1.0 |  ****      ****
0.8 | *    *    *    *
0.6 |       *  *      *
0.4 |        **
0.2 |
0.0 +--+--+--+--+--+--+--+-->
    t0  t1  t2  t3  t4  t5
```

Use one line per rule, with a legend. Limit to top-10 most variable rules for readability.

### Rule Co-Activation Matrix

Compute the correlation between rule activations across a batch:

```
          Rule A  Rule B  Rule C  Rule D
Rule A     1.0    0.8     0.2    -0.1
Rule B     0.8    1.0     0.3     0.0
Rule C     0.2    0.3     1.0     0.7
Rule D    -0.1    0.0     0.7     1.0
```

Display as a correlation heatmap. Clustered rules suggest conceptual groupings in the learned knowledge base.

---

## Confidence Evolution Plots

### Full Pipeline Confidence Tracking

Track confidence at multiple points in the brain_ai pipeline:

1. **Encoder confidence**: Entropy of encoder output distribution
2. **Workspace confidence**: Competition winner score
3. **Pre-reasoning confidence**: Before dual-process routing
4. **Post-reasoning confidence**: After System 1 or System 2
5. **Output confidence**: Final prediction confidence (max softmax)

Plot as a waterfall or step chart:

```
Confidence
1.0 |                        *
0.9 |                   *---*
0.8 |------------- threshold --------
0.7 |              *
0.6 |         *
0.5 |    *
0.4 | *
    +----+----+----+----+----+----+
    Enc   WS  Pre-R  S2-1  S2-2  Out
         Pipeline Stage
```

### Confidence Calibration

Compare predicted confidence with actual accuracy:

```
Accuracy
1.0 |                    *
0.8 |               *
0.6 |          *
0.4 |     *
0.2 | *
0.0 +--+--+--+--+--+--+-->
   0.0  0.2  0.4  0.6  0.8  1.0
         Predicted Confidence
```

The diagonal line represents perfect calibration. Points above the line indicate underconfidence; below indicates overconfidence. This is a reliability diagram.

---

## Logic Tensor Network Visualization

### Grounded Predicate Networks

When `use_ltn = True`, predicates are implemented as neural networks. Visualize:

1. **Predicate response surface**: For a 2D embedding space, plot the predicate's truth value as a contour map.
2. **Entity embeddings**: Project entity embeddings to 2D (via PCA or t-SNE) and color by predicate truth value.

### Quantifier Visualization

LTN uses generalized quantifiers with p-norms:
- **Universal** (for-all): p = 2.0 (strict)
- **Existential** (exists): p = 0.5 (lenient)

Visualize the aggregation behavior:

```
Aggregated Truth Value
1.0 |  *                  *
0.8 |    *              *
0.6 |      *          *
0.4 |        *      *
0.2 |          *  *
0.0 |            *
    +---+---+---+---+---+--->
    0%  20%  40%  60%  80% 100%
     Fraction of True Groundings
```

Compare curves for different p values.

---

## Color Conventions

### System Colors

| System | Color | Hex |
|--------|-------|-----|
| System 1 (fast) | Cyan | `#00bcd4` |
| System 2 (slow) | Magenta | `#e91e63` |
| Threshold line | Red dashed | `#f44336` |
| Converged step | Green | `#4caf50` |
| Non-converged | Orange warning | `#ff9800` |

### Confidence Gradient

Map confidence to a continuous color gradient:
- 0.0: Deep red (`#d32f2f`)
- 0.5: Orange (`#ff9800`)
- 0.8: Yellow-green (`#cddc39`)
- 1.0: Green (`#4caf50`)

Use `matplotlib.colors.LinearSegmentedColormap.from_list()` for custom gradient.

### Rule Activation Colors

| Activation Level | Color | Hex | Meaning |
|-----------------|-------|-----|---------|
| > 0.8 | Green | `#2ca02c` | High confidence rule |
| 0.5 - 0.8 | Orange | `#ff7f0e` | Moderate activation |
| 0.2 - 0.5 | Light red | `#e57373` | Weak activation |
| < 0.2 | Gray | `#bdbdbd` | Effectively inactive |

---

## Integration Points

### From Workspace

The reasoning module receives the workspace representation as input. The pre-reasoning confidence comes from the workspace competition winner score. Visualize the handoff:

```
Workspace Winner Score: 0.72
  |
  V
Routing Decision: System 2 (0.72 < 0.80)
  |
  V
System 2 Iterations: 5 steps to convergence
  |
  V
Post-Reasoning Confidence: 0.91
```

### To Decision System

After reasoning, the refined representation feeds into the decision system. The active inference agent uses the confidence for Expected Free Energy computation:
- High confidence -> low epistemic value -> exploit
- Low confidence -> high epistemic value -> explore

Visualize this as a decision landscape plot.

### Meta-Learning Feedback

The neuromodulatory system receives the reasoning confidence as input. Low confidence triggers increased norepinephrine (exploration) and acetylcholine (attention). Visualize the neuromodulator levels alongside confidence:

```
Level
1.0 |  NE
0.8 |  NE  ACh
0.6 |  NE  ACh
0.4 |  NE  ACh   DA
0.2 |  NE  ACh   DA   5-HT
    +----+----+----+----+
     NE   ACh   DA   5-HT
```

---

## Best Practices

1. **Always show the confidence threshold** as a reference in routing plots.
2. **Color-code System 1 vs. System 2** consistently (cyan vs. magenta).
3. **Mark convergence points** in iteration traces.
4. **Limit rule display** to top-K most active or most variable rules.
5. **Show early stopping** in System 2 traces (where confidence first exceeds threshold).
6. **Include the routing fraction** (% System 1 vs. System 2) in batch-level plots.
7. **Use horizontal bar charts** for rule activations (rule names are long).
8. **Annotate hard cases** where System 2 fails to converge.
9. **Save all figures** to file, use headless backend.
10. **Close figures** after saving to prevent memory accumulation.

---

## Example: Complete Reasoning Trace

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Routing gauge
conf = 0.65
threshold = 0.8
axes[0].barh(0, conf, height=0.5, color='#e91e63')
axes[0].barh(0, 1.0 - conf, left=conf, height=0.5, color='#e0e0e0')
axes[0].axvline(x=threshold, color='red', linestyle='--', linewidth=2)
axes[0].set_xlim(0, 1)
axes[0].set_title(f'Routing: System 2 (conf={conf})')

# 2. System 2 confidence evolution
steps = np.arange(8)
confidences = [0.65, 0.68, 0.72, 0.77, 0.82, 0.85, 0.87, 0.88]
axes[1].plot(steps, confidences, '-o', color='#e91e63', linewidth=2)
axes[1].axhline(y=threshold, color='red', linestyle='--')
axes[1].fill_between(steps, 0, confidences, alpha=0.1, color='#e91e63')
conv_step = next(i for i, c in enumerate(confidences) if c >= threshold)
axes[1].axvline(x=conv_step, color='green', linestyle=':', linewidth=2)
axes[1].set_xlabel('System 2 Step')
axes[1].set_ylabel('Confidence')
axes[1].set_title('Confidence Evolution')

# 3. Rule activations
rules = ['is_animal AND has_fur', 'is_moving AND fast',
         'is_large', 'is_dangerous', 'is_domestic']
activations = [0.92, 0.78, 0.65, 0.31, 0.12]
colors = ['#2ca02c', '#ff7f0e', '#ff7f0e', '#e57373', '#bdbdbd']
axes[2].barh(range(len(rules)), activations, color=colors)
axes[2].set_yticks(range(len(rules)))
axes[2].set_yticklabels(rules)
axes[2].set_xlabel('Activation')
axes[2].set_title('Rule Activations')
axes[2].set_xlim(0, 1)

fig.tight_layout()
fig.savefig('reasoning_trace.png', dpi=150)
plt.close(fig)
```

---

## References

- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Badreddine, S. et al. (2022). "Logic Tensor Networks." *Artificial Intelligence*.
- brain_ai `ReasoningConfig` in `brain_ai/config.py`: confidence_threshold, num_reasoning_steps, use_ltn.
- brain_ai `DualProcessReasoner` in `brain_ai/reasoning/system2.py`.
- brain_ai `SystemOutput.reasoning_trace` in `brain_ai/system.py`.
