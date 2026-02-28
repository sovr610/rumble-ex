# Workspace Visualization

## Overview

The Global Workspace in brain_ai implements Global Workspace Theory (GWT), a cognitive architecture where specialized processors (modality encoders, HTM, reasoning modules) compete for access to a shared workspace. The workspace serves as the integration bottleneck, funneling multi-modal information through a capacity-limited channel (Miller's Law: `WorkspaceConfig.capacity_limit = 7` slots).

Visualizing workspace dynamics is critical for understanding:
- Which modalities dominate workspace access and when
- How broadcast patterns distribute integrated information back to modules
- What working memory slots contain at any given time
- How the ignition threshold affects information flow

This reference covers visualization conventions for competition dynamics, broadcast patterns, and working memory contents.

---

## Competition Dynamics Over Time

### The Selection-Broadcast Cycle

The brain_ai workspace implements a selection-broadcast cycle with configurable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `selection_rounds` | 3 | Number of iterative competition rounds |
| `ignition_threshold` | 0.3 | Minimum score for workspace access |
| `broadcast_iterations` | 2 | Number of broadcast refinement steps |
| `broadcast_decay` | 0.9 | Temporal decay of broadcast signal |
| `use_confidence_gating` | True | Gate output by confidence level |

### Competition Timeline Plot

A timeline plot shows competition scores for all modalities across selection rounds for a single input:

```
Score
1.0 |          ___________
    |         /
0.8 |        /       .....
    |       /       .
0.6 |      /       .
    |     /       .
0.4 |    /    ---*--------
    |   / ---/
0.2 |  /--/
    | *
0.0 +----+----+----+
    R0   R1   R2   R3
```

- **Solid lines**: Each modality's competition score
- **Dashed line**: Ignition threshold (0.3)
- **Filled markers**: Round where a modality achieves ignition
- **Star marker**: Winner at final round

#### Implementation

```python
fig, ax = plt.subplots(figsize=(10, 6))
for i, name in enumerate(modality_names):
    ax.plot(rounds, scores[:, i], '-o', label=name,
            color=MODALITY_COLORS[name], linewidth=2)
ax.axhline(y=threshold, color='red', linestyle='--',
           alpha=0.7, label=f'Ignition Threshold ({threshold})')
ax.set_xlabel('Selection Round')
ax.set_ylabel('Competition Score')
ax.set_title('Workspace Competition Dynamics')
ax.legend(loc='upper left')
ax.set_ylim(0, 1.05)
```

### Batch-Averaged Competition

For a batch of inputs, show the mean and standard deviation of competition scores:

```python
mean_scores = all_scores.mean(dim=0)  # (rounds, modalities)
std_scores = all_scores.std(dim=0)
ax.fill_between(rounds, mean - std, mean + std, alpha=0.2)
```

This reveals whether competition patterns are consistent across inputs or highly variable.

### Competition Convergence

Track how quickly scores converge (stop changing significantly between rounds):

```
delta[r] = ||scores[r] - scores[r-1]||_2
```

Plot delta vs. round number. A steep decline indicates fast convergence; slow decline suggests more rounds are needed. This is a key diagnostic for tuning `selection_rounds`.

### Temporal Competition Across a Sequence

For sequential inputs (e.g., video frames or text tokens), create a stacked area chart showing which modality wins the workspace at each time step:

```
Proportion
1.0 |########@@@@@@######@@@@@@
0.8 |########@@@@@@######@@@@@@
0.6 |::::::::@@@@@@::::::@@@@@@
0.4 |::::::::@@@@@@::::::@@@@@@
0.2 |........@@@@@@......@@@@@@
0.0 +----------------------------> Time
    Frame 1  Frame 2  Frame 3
```

Where each pattern/color represents a modality. This shows attention switching over time.

---

## Broadcast Pattern Maps

### What Is Broadcast?

After a representation wins workspace access (achieves ignition), it is broadcast to all receiving modules. The broadcast signal strength to each module can vary:

```
Broadcast Source: Vision (winner)
  -> SNN Core:    0.8 (high integration)
  -> HTM:         0.6 (moderate)
  -> Reasoning:   0.9 (strong)
  -> Decision:    0.7 (moderate-high)
  -> Meta:        0.3 (low)
```

### Broadcast Map Visualization

#### Circular/Radial Layout

Place the workspace at the center, modules around the perimeter. Draw directed arrows (edges) from workspace to each module, with arrow width and color proportional to broadcast strength:

```
              HTM
              ^
             /|\
            / | \
           /  |  \
    SNN <--  WS  --> Reasoning
           \  |  /
            \ | /
             \|/
              v
           Decision
```

Use `matplotlib.patches.FancyArrowPatch` for curved arrows, with `linewidth` proportional to broadcast strength. Color the arrows using the sequential colormap (stronger = darker).

#### Matrix Layout

For simplicity, a heatmap matrix with broadcast source on one axis and target modules on the other:

```
           SNN   HTM   Reasoning  Decision  Meta
Vision    [0.8   0.6   0.9        0.7       0.3]
Text      [0.4   0.7   0.5        0.6       0.2]
Audio     [0.2   0.3   0.1        0.2       0.1]
Sensor    [0.1   0.1   0.1        0.5       0.8]
```

This shows all broadcast patterns simultaneously.

#### Temporal Broadcast Evolution

Over broadcast iterations (`broadcast_iterations = 2`), the signal decays by `broadcast_decay = 0.9`. Visualize the decay:

```
Strength
1.0 | *
0.9 |    *
0.8 |       *
0.7 |          *
    +----+----+----+
    Iter0 Iter1 Iter2
```

Plot one line per module to show differential decay rates (some modules may amplify the signal via feedback).

---

## Working Memory Slot Contents

### Slot Architecture

The global workspace maintains `capacity_limit = 7` working memory slots (following Miller's Law). Each slot contains a `workspace_dim = 4096`-dimensional vector. The working memory is implemented with a Liquid NN (CfC) or LSTM fallback.

### Slot Content Visualization

#### Slot Activation Barcode

Reduce each 4096-dim slot vector to a compact visual representation:

1. **Mini heatmap**: Reshape the 4096-dim vector into a 64x64 grid and display as a small heatmap. Seven side-by-side mini heatmaps show all slots at once.

2. **Top-K feature bars**: For each slot, show the top-K (e.g., 10) most activated dimensions as a horizontal bar chart. This highlights what the slot is "attending to."

3. **Slot similarity matrix**: Compute pairwise cosine similarity between all 7 slots and display as a 7x7 heatmap. This reveals:
   - Redundant slots (high similarity)
   - Diverse representations (low similarity)
   - Empty slots (near-zero norm)

#### Slot Norms

Plot the L2 norm of each slot as a bar chart:

```
Norm
3.0 |  ###
2.5 |  ###  ###
2.0 |  ###  ###  ###
1.5 |  ###  ###  ###  ###
1.0 |  ###  ###  ###  ###  ###
0.5 |  ###  ###  ###  ###  ###  ###
0.0 |  ###  ###  ###  ###  ###  ###  ###
    +----+----+----+----+----+----+----+
    Slot0 Slot1 Slot2 Slot3 Slot4 Slot5 Slot6
```

Low-norm slots may be underutilized; high-norm slots carry the most information.

#### Slot Labels

If available, annotate slots with their likely content based on the modality that most recently wrote to them. Color-code slot borders using modality colors.

### Temporal Slot Dynamics

Over a sequence of time steps, working memory slots update. Visualize the trajectory:

#### Slot Content Over Time (Heatmap)

Create a 2D heatmap with:
- **X-axis**: Time step
- **Y-axis**: Slot index (0-6)
- **Color**: Slot norm or a scalar summary (e.g., first principal component)

This shows which slots are active at which times.

#### Slot Write/Read Patterns

Track which module writes to and reads from each slot at each time step:

```
Slot 0: [W:Vision] -> [R:Reasoning, R:Decision]
Slot 1: [W:Text]   -> [R:Reasoning]
Slot 2: [W:HTM]    -> [R:Decision, R:Meta]
...
```

Visualize as a bipartite graph or Sankey diagram.

---

## Information Flow Diagrams

### Full Pipeline Flow

The brain_ai forward pass follows this path:

```
Input -> Encoders -> SNN Core -> HTM -> Global Workspace -> Reasoning -> Active Inference -> Output
                                              ^                                    |
                                        Neuromodulation <-- Meta-Learning <--------+
```

Annotate each arrow with:
- Tensor dimensionality (e.g., "4096-dim")
- Information throughput (bits, if computed)
- Module activation level (mean activation value)

### Bottleneck Identification

The workspace is the information bottleneck. Measure:

1. **Input information**: Entropy of modality representations before workspace.
2. **Workspace information**: Entropy of workspace representation.
3. **Information compression ratio**: Input entropy / workspace entropy.

Plot these as a funnel diagram:

```
Input (high entropy)
    |
    V
Encoders (moderate)
    |
    V
Workspace (low) <- BOTTLENECK
    |
    V
Reasoning (expanded)
    |
    V
Output (task-specific)
```

### Mutual Information Between Modules

Compute pairwise mutual information (or linear CKA similarity) between module representations. Display as a correlation matrix heatmap:

```
           Encoder  SNN   HTM   WS    Reason  Decision
Encoder     1.0    0.8   0.5   0.6    0.4     0.3
SNN         0.8    1.0   0.6   0.7    0.5     0.4
HTM         0.5    0.6   1.0   0.5    0.3     0.2
WS          0.6    0.7   0.5   1.0    0.8     0.7
Reason      0.4    0.5   0.3   0.8    1.0     0.6
Decision    0.3    0.4   0.2   0.7    0.6     1.0
```

This reveals the information pathways in the system.

---

## Color Conventions

### Workspace-Specific Colors

| Element | Color | Hex |
|---------|-------|-----|
| Workspace center | Gold | `#FFD700` |
| Active slot | Green | `#2ca02c` |
| Inactive slot | Gray | `#808080` |
| Ignition event | Red flash | `#FF4500` |
| Broadcast arrow | Orange gradient | `#ff7f0e` to `#ffbb78` |
| Failed ignition | Gray dashed | `#a0a0a0` |

### Module Colors (for flow diagrams)

| Module | Color | Hex |
|--------|-------|-----|
| Encoders | Teal | `#17becf` |
| SNN Core | Steel Blue | `#4682B4` |
| HTM | Sea Green | `#2E8B57` |
| Workspace | Gold | `#FFD700` |
| Reasoning | Orchid | `#DA70D6` |
| Decision | Tomato | `#FF6347` |
| Meta | Slate Blue | `#6A5ACD` |

### Slot Colors

Use a qualitative palette for 7 slots: `tab10` first 7 colors, or a custom palette:

| Slot | Color | Hex |
|------|-------|-----|
| 0 | Blue | `#1f77b4` |
| 1 | Orange | `#ff7f0e` |
| 2 | Green | `#2ca02c` |
| 3 | Red | `#d62728` |
| 4 | Purple | `#9467bd` |
| 5 | Brown | `#8c564b` |
| 6 | Pink | `#e377c2` |

---

## Scaling Considerations

### Large Workspace Dim

With `workspace_dim = 4096`, direct visualization of the full vector is impractical. Strategies:
- **PCA reduction**: Project to 2D or 3D for scatter/trajectory plots.
- **Heatmap reshape**: Reshape to 64x64 grid.
- **Statistics only**: Plot mean, std, min, max, norm of each slot.
- **Top-K dimensions**: Show only the K most active dimensions.

### Many Time Steps

For long sequences, subsample time steps or use aggregate statistics:
- **Sliding window average**: Smooth competition scores over a window.
- **Event-based**: Only show time steps where the winner changes.
- **Summary statistics**: Mean and variance of competition scores per modality over the full sequence.

### Multiple Batches

- Plot competition dynamics for a representative single batch element.
- For batch statistics, show mean +/- std across the batch.
- For diversity analysis, show competition patterns for several batch elements as small multiples.

---

## Integration with Other Visualizations

The workspace visualization connects naturally with other visualization modules:

1. **From Attention Heatmaps**: The attention weights driving workspace competition feed directly into the competition score plots.

2. **To Reasoning Traces**: The workspace representation that enters the reasoning module can be tracked to see how dual-process routing decisions depend on workspace content.

3. **From SNN Core**: Spike patterns in the SNN core determine the encoder outputs that compete for workspace access. Overlay spike rate summary with competition scores.

4. **To Embedding Projections**: Working memory slot contents can be projected via t-SNE/PCA to understand clustering and separation.

---

## Best Practices

1. **Always show the ignition threshold** as a reference line in competition plots.
2. **Annotate the winner** at each round with a distinct marker or highlight.
3. **Use consistent modality colors** across all workspace visualizations.
4. **Show slot norms** alongside slot content for quick assessment of utilization.
5. **Indicate empty/unused slots** with grayed-out or hatched backgrounds.
6. **Use temporal context**: Never show a single frame in isolation; always indicate where in the sequence the snapshot comes from.
7. **Close figures** after saving (`plt.close(fig)`) to prevent memory leaks.
8. **Headless backend**: Always use `matplotlib.use('Agg')` for server/CI environments.
9. **Subsample** long sequences for readability; note the subsampling in the title.
10. **Validate** that workspace tensors are non-zero before plotting; plot a warning if all values are zero.

---

## Example: Competition Dynamics with Working Memory

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

modalities = ['Vision', 'Text', 'Audio', 'Sensor']
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
rounds = np.arange(4)
# Synthetic scores: (4 rounds, 4 modalities)
scores = np.array([
    [0.25, 0.25, 0.25, 0.25],
    [0.40, 0.30, 0.20, 0.10],
    [0.55, 0.25, 0.12, 0.08],
    [0.70, 0.18, 0.08, 0.04],
])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Competition dynamics
for i, name in enumerate(modalities):
    axes[0].plot(rounds, scores[:, i], '-o', label=name,
                 color=colors[i], linewidth=2, markersize=8)
axes[0].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Threshold')
axes[0].set_xlabel('Selection Round')
axes[0].set_ylabel('Competition Score')
axes[0].set_title('Workspace Competition Dynamics')
axes[0].legend()
axes[0].set_ylim(0, 1.0)

# Working memory slot norms
slot_norms = np.array([2.8, 2.1, 1.5, 0.9, 0.4, 0.1, 0.05])
slot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
               '#9467bd', '#8c564b', '#e377c2']
axes[1].bar(range(7), slot_norms, color=slot_colors)
axes[1].set_xlabel('Memory Slot')
axes[1].set_ylabel('L2 Norm')
axes[1].set_title('Working Memory Slot Utilization')
axes[1].set_xticks(range(7))

fig.tight_layout()
fig.savefig('workspace_dynamics.png', dpi=150)
plt.close(fig)
```

---

## References

- Baars, B.J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
- Dehaene, S. & Naccache, L. (2001). "Towards a cognitive neuroscience of consciousness." *Cognition*.
- Miller, G.A. (1956). "The magical number seven, plus or minus two." *Psychological Review*.
- brain_ai `WorkspaceConfig` in `brain_ai/config.py`: workspace_dim, capacity_limit, selection_rounds, ignition_threshold.
- brain_ai `GlobalWorkspace` in `brain_ai/workspace/global_workspace.py`.
