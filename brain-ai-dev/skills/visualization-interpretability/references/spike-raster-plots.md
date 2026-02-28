# Spike Raster Plot Conventions

## Overview

Spike raster plots are the foundational visualization tool for spiking neural networks (SNNs). In the brain_ai system, the SNN core (analogous to cortical columns) processes inputs through LIF (Leaky Integrate-and-Fire) neurons with surrogate gradients. Visualizing the spike patterns is essential for understanding temporal coding, debugging training dynamics, and verifying that population-level statistics match biological plausibility constraints (e.g., target 10% spike rate defined in `SNNConfig.spike_rate_target`).

This reference covers:
- Raster plot layout and conventions
- Neuron indexing and subsampling
- Time axis conventions
- Firing rate heatmaps
- Membrane potential traces
- Population coding visualization
- Color conventions

---

## Raster Plot Layout

### Standard Raster Plot

A raster plot displays individual spike events as dots or short vertical lines on a 2D plane:

- **X-axis**: Time steps (discrete, integer-valued). The brain_ai system uses `SNNConfig.num_timesteps` (default 50 for production, 10 for minimal config).
- **Y-axis**: Neuron index. Each row represents one neuron.
- **Markers**: A filled dot or short vertical bar at `(t, n)` indicates neuron `n` fired at time step `t`.

```
Neuron 99  |                  .     .
Neuron 98  |        .              .   .
Neuron 97  |   .        .
   ...     |
Neuron  1  |      .        .    .
Neuron  0  |  .       .           .
           +----------------------------->
            0   10   20   30   40   50
                   Time Step
```

### Axis Conventions

| Axis | Label | Units | Range |
|------|-------|-------|-------|
| X | Time Step | discrete steps | `[0, num_timesteps)` |
| Y | Neuron Index | integer | `[0, num_neurons)` |

When displaying subsets of neurons, always indicate the subsampling in the title or annotation, e.g., "Spike Raster (100 of 4096 neurons, random sample)".

### Marker Style

- **Default**: Small filled circles (`matplotlib marker='|'` or `'.'`), size 1-2 points.
- **Dense regimes** (>1000 spikes): Use rasterized rendering (`rasterized=True`) to avoid slow vector rendering.
- **Sparse regimes** (<100 spikes): Use slightly larger markers (size 3-4) for visibility.

---

## Neuron Indexing

### Layer-Based Indexing

The brain_ai SNN core has multiple hidden layers defined in `SNNConfig.hidden_sizes` (default `[4096, 4096, 2048, 2048]`). Neuron indices should reflect this layered structure:

```
Layer 0: neurons   0 - 4095    (4096 neurons)
Layer 1: neurons 4096 - 8191   (4096 neurons)
Layer 2: neurons 8192 - 10239  (2048 neurons)
Layer 3: neurons 10240 - 12287 (2048 neurons)
```

When plotting all layers, use horizontal separator lines or alternating background colors to distinguish layers. When plotting a single layer, label the y-axis as "Neuron (Layer N)".

### Subsampling Strategy

For production models with thousands of neurons per layer, always subsample to `VizConfig.max_neurons` (default 100) for readability:

1. **Uniform subsampling**: Select every k-th neuron. Preserves spatial ordering.
2. **Random subsampling**: Random selection without replacement. Better statistical representation but loses ordering. Set a fixed random seed for reproducibility.
3. **Activity-based subsampling**: Select the top-N most active neurons. Useful for identifying hot spots but introduces selection bias. Always annotate this.

Recommendation: Default to uniform subsampling. Provide `neuron_ids` parameter to override.

### Batch Dimension

Spike tensors from brain_ai have shape `(batch, timesteps, neurons)` or `(batch, neurons, timesteps)`. The raster plotter must handle both orderings. Convention:
- Plot a single batch element at a time.
- Provide a `batch_idx` parameter defaulting to 0.
- For multi-batch comparison, use a grid of subplots.

---

## Time Axis

### Discrete Time Steps

The SNN core operates in discrete time steps. Each step corresponds to one forward pass through the LIF neuron dynamics:

```
V[t+1] = beta * V[t] + I[t] - spike[t] * threshold
spike[t+1] = Heaviside(V[t+1] - threshold)
```

The time axis should show integer step indices. For production models with 50 time steps, tick marks every 10 steps are appropriate.

### Real-Time Mapping

If a real-time mapping is known (e.g., 1 step = 1 ms for biological fidelity), provide a secondary x-axis with physical time units. This is optional and depends on the simulation context.

### Sliding Window for Firing Rates

When computing firing rates, use a sliding window of width `window` (default 10 steps):

```
rate[t, n] = sum(spikes[t-window//2 : t+window//2, n]) / window
```

The window size should be annotated on the plot. Smaller windows reveal fast transients; larger windows smooth noise.

---

## Firing Rate Heatmaps

### Layout

A firing rate heatmap is a 2D color-coded image:
- **X-axis**: Time step (or time bin center)
- **Y-axis**: Neuron index
- **Color**: Firing rate (spikes per step within window)

Use `matplotlib.pyplot.imshow()` with `aspect='auto'` and `origin='lower'` so neuron 0 is at the bottom.

### Color Mapping

| Metric | Colormap | Range | Rationale |
|--------|----------|-------|-----------|
| Firing rate | `viridis` | [0, max_rate] | Perceptually uniform, colorblind-safe |
| Rate deviation from target | `RdBu_r` | [-target, +target] | Diverging: blue = below target, red = above |
| Inter-spike interval | `plasma` | [1, max_isi] | Sequential, distinct from viridis |

The target firing rate is `SNNConfig.spike_rate_target` (default 0.1, i.e., 10%). Any deviation from this target is relevant for training diagnostics.

### Colorbar

Always include a colorbar with:
- Label: "Firing Rate (spikes/step)" or "Rate Deviation"
- Tick marks at meaningful values (0, target, max)
- Consistent range across subplots for comparison

### Aggregated Rate Histogram

Alongside the heatmap, a marginal histogram showing the distribution of mean firing rates across neurons helps identify:
- **Dead neurons**: rate near 0 (common early in training)
- **Saturated neurons**: rate near 1.0 (all-spike regime, indicates training instability)
- **Target compliance**: peak near 0.1

---

## Membrane Potential Traces

### Single-Neuron Trace

Plot membrane potential V[t] over time for one or a few selected neurons:

```
V (mV)
  ^
  |        /\        /\
  |  /\   /  \      /  \__
  | /  \ /    \____/      \
  |/    \/                 \____
  +------------------------------> time
  0    10    20    30    40    50
```

Key elements:
- **Membrane potential line**: Solid line (blue by default)
- **Threshold line**: Dashed horizontal line (red, at `threshold` value, typically 1.0)
- **Spike markers**: Vertical dashed lines or triangles at spike times
- **Reset**: After a spike, V resets to 0 (hard reset) or decreases by threshold (soft reset). This should be visible as a sharp drop.

### Multi-Neuron Overlay

When overlaying multiple neurons, use different colors from a qualitative colormap (`tab10`, `Set2`). Limit to 5-8 neurons for readability. Include a legend with neuron indices.

### Subthreshold Dynamics

The subthreshold dynamics reveal:
- **Leak rate** (`beta` parameter): Higher beta means slower leak (longer memory).
- **Input strength**: Steeper rise indicates stronger input.
- **Near-miss events**: V approaches threshold but does not cross. These are important for understanding network dynamics.

Annotate the plot with `beta` value and threshold in the title or legend.

---

## Population Coding Visualization

### Population Histogram

A histogram of total spike counts per neuron over all time steps:

```
Count
  ^
  |  ####
  |  ######
  |  ########
  |  ############
  |  ##############
  +--------------------> Neuron Index
```

This reveals which neurons are most active and whether the population is well-distributed.

### Spike Count Distribution

Plot the distribution of spike counts across all neurons:

```
P(count)
  ^
  |     *
  |    * *
  |   *   *
  |  *     *
  | *       *
  +-----------> Spike Count
```

For a healthy SNN with target rate 0.1 and 50 time steps, the expected spike count per neuron is 5 with Poisson-like variance.

### Temporal Population Vector

At each time step, compute a population vector (the binary spike pattern across all neurons). Visualize similarity between consecutive population vectors using cosine similarity:

```
cos_sim(pop[t], pop[t+1])
```

This reveals temporal coding stability vs. variability.

### Cross-Layer Activity Correlation

For multi-layer SNNs, compute the cross-correlation between mean firing rates of adjacent layers. High correlation indicates good information flow; near-zero correlation suggests a bottleneck.

---

## Color Conventions

Consistent with the brain_ai visualization standard:

### Layer Colors

| SNN Layer | Color | Hex |
|-----------|-------|-----|
| Layer 0 (input) | Steel Blue | `#4682B4` |
| Layer 1 | Medium Sea Green | `#3CB371` |
| Layer 2 | Coral | `#FF7F50` |
| Layer 3 (output) | Medium Purple | `#9370DB` |

### Spike Event Colors

| Context | Color | Hex |
|---------|-------|-----|
| Excitatory spike | Black | `#000000` |
| Inhibitory spike | Red | `#FF0000` |
| Threshold | Red dashed | `#FF0000` |
| Membrane potential | Blue | `#4169E1` |
| Reset level | Gray dashed | `#808080` |

### Modality Colors (when showing modality-specific SNN layers)

| Modality | Color | Hex |
|----------|-------|-----|
| Vision | Blue | `#1f77b4` |
| Text | Green | `#2ca02c` |
| Audio | Orange | `#ff7f0e` |
| Sensor | Purple | `#9467bd` |
| Engram | Brown | `#8c564b` |

---

## Implementation Notes for brain_ai

### Tensor Shapes

The SNN core (`brain_ai/core/snn.py`) produces spike tensors of shape `(batch, timesteps, hidden_size)` where values are binary (0 or 1) after the surrogate gradient step. Membrane potentials have the same shape but continuous values.

### Surrogate Gradient Visualization

The surrogate gradient functions (ATan, FastSigmoid, StraightThrough) can be visualized as a function of membrane potential minus threshold. This is useful for understanding the effective gradient signal:

```python
import torch
x = torch.linspace(-3, 3, 1000)
atan_grad = (1 / (1 + (alpha * x) ** 2)) * alpha / (2 * torch.pi)
```

Plot these curves for each surrogate type to compare sharpness and support.

### Learnable Delays Visualization

When `SNNConfig.use_learnable_delays` is enabled, each synapse has a learnable delay in `[0, max_delay]`. Visualize the delay distribution as a histogram to check:
- Are delays clustered (redundant) or spread (diverse)?
- Does the distribution shift during training?

### Heterogeneous Tau Visualization

When `SNNConfig.use_heterogeneous_tau` is enabled, each neuron has its own time constant. Visualize:
- Distribution of tau values (histogram)
- Correlation between tau and firing rate (scatter plot)
- Tau values sorted and colored by layer

---

## Best Practices

1. **Always label axes** with units (time steps, neuron index, spikes/step).
2. **Always include a title** with the layer name, batch index, and any subsampling note.
3. **Use headless backend** (`matplotlib.use('Agg')`) for CI/CD and server environments.
4. **Save figures** to files rather than displaying them interactively.
5. **Use tight_layout()** or `constrained_layout=True` to avoid label clipping.
6. **Set random seed** for reproducible subsampling.
7. **Subsample** large populations to `max_neurons` (default 100).
8. **Use rasterized=True** for scatter plots with >10,000 points.
9. **Close figures** after saving with `plt.close(fig)` to free memory.
10. **Log scale** for firing rate histograms if the distribution is heavy-tailed.

---

## Example: Minimal Raster Plot

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# Synthetic spike data: 1 batch, 50 timesteps, 100 neurons
spikes = (torch.rand(1, 50, 100) > 0.9).float()

fig, ax = plt.subplots(figsize=(12, 6))
times, neurons = torch.where(spikes[0])
ax.scatter(times.numpy(), neurons.numpy(), s=1, c='black', marker='|')
ax.set_xlabel('Time Step')
ax.set_ylabel('Neuron Index')
ax.set_title('Spike Raster Plot (Synthetic, 100 neurons, 50 steps)')
ax.set_xlim(0, 50)
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig('raster_example.png', dpi=150)
plt.close(fig)
```

---

## References

- Dayan, P. & Abbott, L.F. (2001). *Theoretical Neuroscience*. Chapter 1: Neural Encoding.
- Gerstner, W. et al. (2014). *Neuronal Dynamics*. Chapter 7: Spike Trains.
- Neftci, E.O. et al. (2019). "Surrogate Gradient Learning in Spiking Neural Networks." *IEEE Signal Processing Magazine*.
- brain_ai `SNNConfig` in `brain_ai/config.py`: spike_rate_target, num_timesteps, hidden_sizes.
