# Attention Heatmap Visualization

## Overview

Attention heatmaps visualize the weight matrices that determine how information flows between components in the brain_ai system. The primary attention mechanisms are:

1. **Global Workspace Competition**: Multi-head attention (`WorkspaceConfig.num_heads = 32`) that selects which modality representations gain access to the workspace.
2. **Cross-Modal Attention**: Attention between different modality encoders (`WorkspaceConfig.cross_modal_heads = 16`) enabling information sharing across vision, text, audio, and sensor streams.
3. **Selection-Broadcast Attention**: The iterative competition rounds (`WorkspaceConfig.selection_rounds = 3`) that determine workspace access and broadcast patterns.
4. **Self-Attention within Encoders**: Attention heads within the vision (ViT), text (transformer), and audio encoders.

This reference covers conventions for rendering these attention patterns as interpretable heatmaps.

---

## Single-Head Attention Heatmap

### Layout

A single attention head produces a weight matrix of shape `(query_len, key_len)` where values sum to 1 along the key dimension (after softmax):

```
         k_0   k_1   k_2   k_3   k_4
q_0  [  0.8   0.1   0.05  0.03  0.02 ]
q_1  [  0.1   0.7   0.1   0.05  0.05 ]
q_2  [  0.05  0.1   0.6   0.15  0.1  ]
q_3  [  0.02  0.05  0.15  0.68  0.1  ]
q_4  [  0.03  0.05  0.1   0.09  0.73 ]
```

### Rendering

Use `matplotlib.pyplot.imshow()` or `pcolormesh()`:

```python
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(weights, cmap='viridis', aspect='auto', vmin=0, vmax=1)
ax.set_xlabel('Key Position')
ax.set_ylabel('Query Position')
ax.set_title('Attention Weights (Head 0)')
fig.colorbar(im, ax=ax, label='Attention Weight')
```

### Axis Labels

- **Positional indices**: Default for encoder self-attention (0, 1, 2, ...).
- **Token labels**: If available, use actual tokens or token abbreviations on the axes.
- **Modality labels**: For workspace competition, label keys with modality names (vision, text, audio, sensor, engram).

### Value Annotations

For small matrices (< 10x10), overlay numerical values on each cell:

```python
for i in range(rows):
    for j in range(cols):
        ax.text(j, i, f'{weights[i,j]:.2f}',
                ha='center', va='center',
                color='white' if weights[i,j] > 0.5 else 'black',
                fontsize=8)
```

For larger matrices, omit annotations and rely on the colorbar.

---

## Multi-Head Attention

### Grid Layout

The brain_ai workspace uses 32 attention heads. Displaying all heads simultaneously:

1. **Grid of subplots**: Arrange heads in a grid (e.g., 4x8 for 32 heads). Each subplot shows one head's weight matrix as a small heatmap. Omit individual colorbars; use a single shared colorbar.

2. **Head summary**: Compute statistics per head (entropy, max weight, sparsity) and display as a bar chart to identify "sharp" vs. "diffuse" heads.

### Head Entropy

Attention head entropy measures how focused or distributed the attention is:

```
H(head_h) = -sum_k(w_k * log(w_k + eps))
```

- **Low entropy**: Focused attention on few keys (sharp head).
- **High entropy**: Uniform attention across all keys (diffuse head).

Plot entropy per head as a bar chart, sorted from lowest to highest, to identify specialized vs. generic heads.

### Head Pruning Diagnostics

Heads with consistently high entropy (near uniform) may be redundant. Visualize:
- Entropy over training steps (line plot per head)
- Pairwise cosine similarity between head outputs (correlation matrix)

---

## Cross-Modal Attention Matrices

### Concept

Cross-modal attention in brain_ai enables information flow between modality-specific representations before workspace competition. With 4 modalities (vision, text, audio, sensors) plus optional engram, this produces a set of attention matrices.

### Layout for Cross-Modal Pairs

For each pair of modalities (source -> target), display an attention heatmap:

```
                Vision    Text    Audio   Sensor
Vision          [self]   [v->t]  [v->a]  [v->s]
Text            [t->v]   [self]  [t->a]  [t->s]
Audio           [a->v]   [a->t]  [self]  [a->s]
Sensor          [s->v]   [s->t]  [s->a]  [self]
```

This produces a grid of heatmaps. Self-attention blocks are along the diagonal.

### Summary Cross-Modal Flow

Compute the mean attention weight between each pair of modalities and display as a simpler matrix:

```
            Vision  Text  Audio  Sensor
Vision       --     0.35  0.15   0.05
Text         0.40    --   0.10   0.02
Audio        0.20   0.12   --    0.03
Sensor       0.08   0.03  0.04    --
```

Render this as a heatmap with modality labels, annotated with values. This reveals which modalities attend most to each other.

### Color Mapping for Cross-Modal

Use a colormap that distinguishes strength levels clearly:

| Weight Range | Visual Meaning |
|-------------|---------------|
| 0.0 - 0.1 | Minimal attention (pale/white) |
| 0.1 - 0.3 | Moderate attention (light color) |
| 0.3 - 0.5 | Strong attention (medium color) |
| 0.5 - 1.0 | Dominant attention (dark/saturated) |

Default colormap: `viridis` (perceptually uniform). Alternative: `YlOrRd` for a fire-like intensity scale.

---

## Workspace Competition Scores

### Selection-Broadcast Cycle

The brain_ai global workspace uses an iterative selection-broadcast cycle (`WorkspaceConfig.selection_rounds = 3`):

1. **Selection**: Modality representations compete for workspace access via attention-based scoring.
2. **Ignition**: If a representation exceeds `ignition_threshold` (0.3), it gains broadcast access.
3. **Broadcast**: The winning representation is broadcast to all modules.

### Competition Score Visualization

At each selection round, each modality receives a competition score in `[0, 1]`. Visualize as:

#### Bar Chart (Single Timestep)

```
Score
1.0 |
0.8 |  ####
0.6 |  ####  ####
0.4 |  ####  ####
0.2 |  ####  ####  ####  ####
0.0 +------+------+------+------+
    Vision  Text  Audio  Sensor
```

Highlight the winner (highest score) with a distinct color or border. Mark the ignition threshold as a horizontal dashed line.

#### Evolution Over Rounds

For multi-round competition, show score evolution as a line plot:

```
Score
1.0 |        /----*
0.8 |   /---/
0.6 |  /
0.4 | *        *----*
0.2 |     \---/
0.0 +----+----+----+
    R1    R2    R3
```

Each line represents a modality. This reveals how competition dynamics converge (or oscillate).

### Winner Identification

Mark the winner at each round with a filled marker. If no modality exceeds the ignition threshold (failed ignition), annotate this condition with a red background or warning text.

### Temporal Competition History

Over a sequence of inputs (e.g., frames in a video or tokens in a sentence), plot which modality wins the competition at each time step:

```
Winner
Sensor |                    ****
Audio  |         ****
Text   |    ****
Vision | ***          ******
       +---------------------------> Time
```

This reveals modality switching patterns and attentional shifts.

---

## Color Mapping Conventions

### Standard Colormaps

| Visualization Type | Colormap | Rationale |
|-------------------|----------|-----------|
| Attention weights | `viridis` | Perceptually uniform, colorblind-safe |
| Diverging (positive/negative) | `RdBu_r` | Clear +/- distinction |
| Competition scores | `YlOrRd` | Intuitive intensity |
| Cross-modal flow | `Blues` | Clean, professional |
| Head entropy | `plasma` | Sequential, distinct from viridis |

### Modality-Specific Colors

Consistent with the brain_ai standard:

| Modality | Color | Hex Code |
|----------|-------|----------|
| Vision | Blue | `#1f77b4` |
| Text | Green | `#2ca02c` |
| Audio | Orange | `#ff7f0e` |
| Sensor | Purple | `#9467bd` |
| Engram | Brown | `#8c564b` |

Use these colors for:
- Bar chart bars in competition plots
- Line colors in evolution plots
- Border colors in multi-modal attention grids

### Dark Mode

When `VizConfig.dark_mode = True`:
- Background: `#1a1a2e` (dark navy)
- Text: `#e0e0e0` (light gray)
- Grid lines: `#333366` (muted blue-gray)
- Colormaps: Use `_r` (reversed) versions if the standard version has a light background issue.

---

## Normalization and Scaling

### Softmax Attention Weights

Standard attention weights are already normalized (sum to 1 along key dimension). Display directly without further normalization. Colorbar range: `[0, 1]` or `[0, max_value]`.

### Pre-Softmax Logits

Sometimes visualizing raw attention logits (before softmax) is more informative for debugging. These can be negative and are not bounded. Use diverging colormap (`RdBu_r`) centered at 0.

### Temperature-Scaled Attention

brain_ai uses confidence gating (`WorkspaceConfig.use_confidence_gating = True`). Temperature-scaled attention weights may be sharper or flatter than standard softmax. Always note the temperature in the plot title.

### Layer Normalization Effects

Attention patterns can change dramatically after layer normalization. When comparing attention before and after LN, use side-by-side plots with identical colorbars.

---

## Aggregation Strategies

### Mean Attention

Average attention weights across all heads:

```python
mean_attn = attention_weights.mean(dim=0)  # (query, key)
```

Useful as a summary view but loses head-specific patterns.

### Max Attention

Take the maximum weight across heads at each (query, key) position:

```python
max_attn = attention_weights.max(dim=0).values
```

Highlights the strongest attention signal regardless of which head produces it.

### Attention Rollout

For multi-layer attention (e.g., text encoder with 32 layers), multiply attention matrices layer by layer (with residual connections accounted for):

```python
rollout = torch.eye(seq_len)
for layer_attn in layer_attentions:
    rollout = torch.matmul(rollout, 0.5 * layer_attn + 0.5 * torch.eye(seq_len))
```

This provides an end-to-end view of information flow.

### Gradient-Weighted Attention

Multiply attention weights by the gradient of the output with respect to the attention matrix. This highlights which attention patterns are most influential for the final prediction.

---

## Interactive Features (Optional)

When `VizConfig.backend = 'plotly'`, provide interactive features:

1. **Hover tooltips**: Show exact weight values on hover.
2. **Zoom**: Zoom into regions of the attention matrix.
3. **Head selection**: Dropdown to switch between heads.
4. **Layer selection**: Slider to navigate between layers.

These features require plotly and are optional. Always fall back to static matplotlib if plotly is unavailable.

---

## Best Practices

1. **Always include a colorbar** with a descriptive label.
2. **Label axes** with meaningful names (modality names, token positions, neuron indices).
3. **Annotate small matrices** (<10x10) with numerical values.
4. **Use consistent colormaps** across all attention visualizations in a single report.
5. **Show the winner** clearly in competition plots (bold, highlight, or arrow).
6. **Close figures** after saving to free memory.
7. **Subsample** for large attention matrices (>100x100). Show a representative region.
8. **Include entropy** or sparsity metrics alongside the heatmap for quick interpretation.
9. **Use `aspect='auto'`** for non-square matrices so all cells are visible.
10. **Set `vmin=0`** for attention weights (they are non-negative after softmax).

---

## Example: Workspace Competition Heatmap

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Synthetic: 4 modalities, 3 selection rounds
modalities = ['Vision', 'Text', 'Audio', 'Sensor']
scores = np.array([
    [0.3, 0.4, 0.2, 0.1],  # Round 1
    [0.5, 0.3, 0.15, 0.05], # Round 2
    [0.7, 0.2, 0.08, 0.02], # Round 3
])

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(scores.T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(3))
ax.set_xticklabels(['Round 1', 'Round 2', 'Round 3'])
ax.set_yticks(range(4))
ax.set_yticklabels(modalities)
ax.set_xlabel('Selection Round')
ax.set_ylabel('Modality')
ax.set_title('Workspace Competition Scores')
fig.colorbar(im, ax=ax, label='Competition Score')

for i in range(4):
    for j in range(3):
        ax.text(j, i, f'{scores[j,i]:.2f}', ha='center', va='center',
                color='white' if scores[j,i] > 0.5 else 'black', fontsize=10)

fig.tight_layout()
fig.savefig('competition_heatmap.png', dpi=150)
plt.close(fig)
```

---

## References

- Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS*.
- Baars, B.J. (1988). *A Cognitive Theory of Consciousness* (Global Workspace Theory).
- Vig, J. (2019). "A Multiscale Visualization of Attention in the Transformer Model." *ACL Demo*.
- brain_ai `WorkspaceConfig` in `brain_ai/config.py`: num_heads, selection_rounds, ignition_threshold.
- brain_ai `SystemOutput` in `brain_ai/system.py`: attention field containing workspace attention weights.
