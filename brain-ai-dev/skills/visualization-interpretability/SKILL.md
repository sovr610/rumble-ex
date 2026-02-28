---
name: Visualization & Interpretability
description: >
  This skill should be used when the user asks to "visualize activations",
  "plot spike rasters", "show attention heatmaps", "visualize workspace",
  "plot reasoning traces", "create t-SNE embeddings", "interpret model decisions",
  "visualize HTM patterns", "show neuromodulator levels", "plot training curves",
  "create activation maps", "visualize feature maps", "explain predictions",
  "debug model behavior visually", or needs guidance on visualization,
  interpretability, or explainability tooling for the brain_ai system.
version: 0.1.0
---

# Visualization & Interpretability

## Overview

Guide implementation of visualization and interpretability infrastructure for all cognitive layers. The brain_ai system produces rich intermediate representations — spike trains, SDR patterns, workspace competition scores, reasoning traces, neuromodulator levels — that are invisible without proper visualization. Cover spike raster plots, attention heatmaps, workspace activation maps, reasoning trace trees, embedding projections, and training curve dashboards.

## Public Contract

### SpikeRasterPlotter

Visualize SNN spike patterns across neurons and time steps.

```python
class SpikeRasterPlotter:
    def __init__(self, config: VizConfig): ...
    def plot_raster(self, spikes: Tensor, neuron_ids: Optional[List] = None) -> Figure: ...
    def plot_firing_rates(self, spikes: Tensor, window: int = 10) -> Figure: ...
    def plot_membrane_potential(self, membrane: Tensor, threshold: float) -> Figure: ...
    def animate_spikes(self, spikes: Tensor, fps: int = 10) -> Animation: ...
```

### AttentionHeatmapper

Visualize attention weights from workspace competition and cross-modal attention.

```python
class AttentionHeatmapper:
    def __init__(self, config: VizConfig): ...
    def plot_attention(self, weights: Tensor, labels: Optional[List[str]] = None) -> Figure: ...
    def plot_cross_modal_attention(self, weights: Dict[str, Tensor]) -> Figure: ...
    def plot_workspace_competition(self, scores: Tensor, winner_idx: int) -> Figure: ...
```

### WorkspaceVisualizer

Visualize global workspace state, broadcast patterns, and working memory.

```python
class WorkspaceVisualizer:
    def __init__(self, config: VizConfig): ...
    def plot_competition_dynamics(self, history: List[Tensor]) -> Figure: ...
    def plot_broadcast_map(self, broadcast: Tensor, module_names: List[str]) -> Figure: ...
    def plot_working_memory_slots(self, memory: Tensor, labels: Optional[List] = None) -> Figure: ...
```

### ReasoningTraceVisualizer

Render dual-process reasoning traces as trees/graphs.

```python
class ReasoningTraceVisualizer:
    def __init__(self, config: VizConfig): ...
    def plot_routing_decision(self, confidence: float, threshold: float) -> Figure: ...
    def plot_system2_steps(self, trace: List[Dict]) -> Figure: ...
    def plot_rule_activation(self, rules: List[str], activations: Tensor) -> Figure: ...
```

### EmbeddingProjector

Dimensionality reduction and visualization for workspace representations.

```python
class EmbeddingProjector:
    def __init__(self, config: VizConfig): ...
    def plot_tsne(self, embeddings: Tensor, labels: Optional[Tensor] = None) -> Figure: ...
    def plot_umap(self, embeddings: Tensor, labels: Optional[Tensor] = None) -> Figure: ...
    def plot_pca(self, embeddings: Tensor, n_components: int = 2) -> Figure: ...
    def interactive_plot(self, embeddings: Tensor, metadata: Dict) -> None: ...
```

### TrainingDashboard

Aggregate training curves and module-specific diagnostics.

```python
class TrainingDashboard:
    def __init__(self, log_dir: str, config: VizConfig): ...
    def plot_loss_curves(self, metrics: Dict[str, List[float]]) -> Figure: ...
    def plot_neuromodulator_levels(self, levels: Dict[str, List[float]]) -> Figure: ...
    def plot_phase_transitions(self, phase_boundaries: List[int]) -> Figure: ...
    def generate_report(self, output_dir: str) -> str: ...
```

## Key Concepts

### Visualization by Cognitive Layer

| Layer | What to Visualize | Key Insight |
|-------|-------------------|-------------|
| SNN Core | Spike rasters, membrane potentials, firing rates | Temporal coding patterns, population dynamics |
| Encoders | Feature maps, activation distributions | Modality-specific representations |
| HTM | SDR overlap, column activations, anomaly scores | Sequence learning, prediction accuracy |
| Workspace | Competition scores, broadcast patterns, WM slots | Information integration, bottleneck behavior |
| Reasoning | Routing decisions, System 2 traces, rule activations | Confidence calibration, reasoning depth |
| Active Inference | EFE landscapes, belief updates, policy distributions | Goal-directed behavior, uncertainty reduction |
| Meta-Learning | Adaptation speed, neuromodulator trajectories | Learning-to-learn dynamics |

### Output Formats

- **Static**: PNG/SVG via matplotlib (default, always available)
- **Interactive**: HTML via plotly (optional, for embedding projections)
- **Animated**: GIF/MP4 via matplotlib.animation (for temporal dynamics)
- **Dashboard**: Integrated HTML report combining multiple visualizations

### Color Schemes

Consistent color mapping across all visualizations:
- Modalities: vision=blue, text=green, audio=orange, sensor=purple
- Neuromodulators: DA=red, ACh=green, NE=blue, 5-HT=yellow
- Systems: System1=cyan, System2=magenta
- Phases: gradient from blue (P1) to red (P7)

## Configuration Surface

```python
@dataclass
class VizConfig:
    backend: str = "matplotlib"          # matplotlib | plotly
    output_format: str = "png"           # png | svg | html | gif
    dpi: int = 150
    figsize: Tuple[int, int] = (12, 8)
    colormap: str = "viridis"
    dark_mode: bool = False
    save_dir: str = "viz/"
    # Spike raster
    max_neurons: int = 100               # Subsample if more
    max_timesteps: int = 500
    # Embeddings
    tsne_perplexity: int = 30
    umap_n_neighbors: int = 15
    max_points: int = 5000               # Subsample for performance
```

## Done-When Gates

1. **Spike Raster** — `SpikeRasterPlotter.plot_raster()` produces correct raster from synthetic spike tensor; neurons on y-axis, time on x-axis; exported PNG is non-empty.
2. **Attention Heatmap** — `AttentionHeatmapper.plot_attention()` renders correct heatmap from synthetic weight matrix; colorbar present; labels correct.
3. **Dashboard Report** — `TrainingDashboard.generate_report()` produces HTML file with loss curves, neuromodulator levels, and phase boundaries from synthetic metrics data.

## Failure Modes

| Mode | Symptom | Fix |
|------|---------|-----|
| Display backend missing | "no display" error | Use Agg backend for headless; save to file only |
| OOM on large embeddings | Crash during t-SNE | Subsample to max_points before projection |
| Blank figures | Empty PNG files | Check tensor shapes; verify data is non-zero |
| Slow animation | Minutes to render | Reduce frames, lower fps, subsample neurons |
| plotly not available | ImportError | Fall back to matplotlib; warn user |

## Anti-Patterns

- Plotting all neurons — subsample to max_neurons for readability
- Blocking on display — always save to file, use non-interactive backend for CI
- Hardcoding colors — use consistent color scheme from VizConfig
- Missing axis labels — always label axes with units
- Generating viz in training loop — visualize post-hoc from saved activations

## Resources

### Reference Files
- **`references/spike-raster-plots.md`** — Raster plot conventions, population coding visualization
- **`references/attention-heatmaps.md`** — Attention visualization, cross-modal patterns
- **`references/workspace-visualization.md`** — Competition dynamics, broadcast maps, WM slots
- **`references/reasoning-traces.md`** — Dual-process trace trees, rule activation graphs
- **`references/testing-matrix.md`** — Test scenarios for visualization infrastructure

### Asset Files
- **`assets/spike_raster_template.py`** — SpikeRasterPlotter with raster, rates, membrane, self-tests
- **`assets/attention_heatmap_template.py`** — AttentionHeatmapper with cross-modal support
- **`assets/workspace_viz_template.py`** — WorkspaceVisualizer with competition and broadcast
- **`assets/reasoning_trace_template.py`** — ReasoningTraceVisualizer with routing and rules
- **`assets/embedding_projector_template.py`** — EmbeddingProjector with t-SNE, UMAP, PCA
- **`assets/viz_config_template.py`** — VizConfig + TrainingDashboard + report generation

### Scripts
- **`scripts/validate_viz.py`** — Validates visualization infrastructure against done-when gates
- **`scripts/gen_viz_tests.py`** — Generates 100+ pytest test cases for visualization modules
- **`scripts/viz_demo.py`** — Demo script producing sample visualizations from synthetic data
