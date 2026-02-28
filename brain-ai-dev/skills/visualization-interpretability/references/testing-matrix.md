# Testing Matrix for Visualization Infrastructure

## Overview

This document defines the comprehensive testing strategy for all visualization modules in the brain_ai visualization-interpretability skill. Each visualization class must pass a suite of tests validating correctness, determinism, format compliance, and robustness.

The three done-when gates from the SKILL.md are:

1. **Spike Raster**: `SpikeRasterPlotter.plot_raster()` produces correct raster from synthetic spike tensor; neurons on y-axis, time on x-axis; exported PNG is non-empty.
2. **Attention Heatmap**: `AttentionHeatmapper.plot_attention()` renders correct heatmap from synthetic weight matrix; colorbar present; labels correct.
3. **Dashboard Report**: `TrainingDashboard.generate_report()` produces HTML file with loss curves, neuromodulator levels, and phase boundaries from synthetic metrics data.

---

## Test Categories

### Category 1: Figure Non-Empty Tests

Every `plot_*` method must produce a figure that, when saved to a file, results in a non-empty file (size > 0 bytes). This catches:

- Methods that return without drawing
- Broken figure creation
- Silent exceptions that produce blank canvases

**Test pattern:**

```python
def test_figure_non_empty(plotter, synthetic_data):
    fig = plotter.plot_raster(synthetic_data)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=72)
        plt.close(fig)
        assert os.path.getsize(f.name) > 0, "Saved PNG is empty"
    os.unlink(f.name)
```

**Coverage:**

| Module | Methods to Test |
|--------|----------------|
| SpikeRasterPlotter | `plot_raster`, `plot_firing_rates`, `plot_membrane_potential` |
| AttentionHeatmapper | `plot_attention`, `plot_cross_modal_attention`, `plot_workspace_competition` |
| WorkspaceVisualizer | `plot_competition_dynamics`, `plot_broadcast_map`, `plot_working_memory_slots` |
| ReasoningTraceVisualizer | `plot_routing_decision`, `plot_system2_steps`, `plot_rule_activation` |
| EmbeddingProjector | `plot_tsne`, `plot_pca`, `plot_umap` (if available) |
| TrainingDashboard | `plot_loss_curves`, `plot_neuromodulator_levels`, `generate_report` |

**Minimum tests per method**: 2 (default parameters, custom parameters)

---

### Category 2: Correct Axes Tests

Every plot must have properly labeled axes with correct orientation:

- **Spike rasters**: X = Time Step, Y = Neuron Index
- **Attention heatmaps**: X = Key/Target, Y = Query/Source
- **Competition plots**: X = Selection Round, Y = Competition Score
- **Rule activations**: X = Activation, Y = Rule Name
- **Embeddings**: X = Component 1, Y = Component 2

**Test pattern:**

```python
def test_axes_labels(plotter, data):
    fig = plotter.plot_raster(data)
    ax = fig.axes[0]
    assert 'Time' in ax.get_xlabel() or 'Step' in ax.get_xlabel()
    assert 'Neuron' in ax.get_ylabel()
    plt.close(fig)
```

**Axis validation matrix:**

| Plot Type | X Label Contains | Y Label Contains | Additional Checks |
|-----------|-----------------|-----------------|-------------------|
| Spike raster | "Time" or "Step" | "Neuron" | xlim >= 0, ylim >= 0 |
| Firing rate heatmap | "Time" or "Step" | "Neuron" | Colorbar present |
| Membrane potential | "Time" or "Step" | "Membrane" or "Potential" | Threshold line present |
| Attention heatmap | "Key" or "Position" | "Query" or "Position" | Colorbar present |
| Cross-modal attention | modality names | modality names | Grid layout |
| Competition scores | "Round" | "Score" or modality names | Threshold line |
| Broadcast map | module names | "Strength" or module names | -- |
| Working memory | "Slot" | "Dimension" or "Norm" | 7 slots max |
| Routing decision | "Confidence" | -- | Threshold marker |
| System 2 steps | "Step" or "Iteration" | "Confidence" | Convergence marker |
| Rule activation | "Activation" | rule names | Range [0,1] |
| t-SNE | "t-SNE 1" or "Component 1" | "t-SNE 2" or "Component 2" | Points visible |
| PCA | "PC 1" or "Component 1" | "PC 2" or "Component 2" | Variance explained |
| Loss curves | "Step" or "Epoch" | "Loss" | Multiple lines for multiple metrics |
| Neuromodulator levels | "Step" or "Time" | "Level" | 4 neuromodulators |

---

### Category 3: Deterministic Rendering Tests

Given the same input data and the same random seed, plots must be identical (bitwise or within tolerance). This ensures reproducibility.

**Test pattern:**

```python
def test_deterministic(plotter, data):
    import hashlib
    hashes = []
    for _ in range(2):
        torch.manual_seed(42)
        np.random.seed(42)
        fig = plotter.plot_raster(data)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=72)
            plt.close(fig)
            with open(f.name, 'rb') as fp:
                hashes.append(hashlib.md5(fp.read()).hexdigest())
            os.unlink(f.name)
    assert hashes[0] == hashes[1], "Non-deterministic rendering"
```

**Notes:**
- t-SNE is inherently non-deterministic unless a fixed seed is used. The `EmbeddingProjector` must accept a `random_state` parameter and pass it through.
- PCA is deterministic by construction.
- UMAP can be non-deterministic; fix the seed.

---

### Category 4: Headless Backend Tests

All visualization code must work without a display server (no X11, no Wayland). This is validated by:

1. Ensuring `matplotlib.use('Agg')` is called before any pyplot import.
2. Running all tests in a headless environment (e.g., CI container).
3. No calls to `plt.show()` in any production code.

**Test pattern:**

```python
def test_headless_backend():
    import matplotlib
    assert matplotlib.get_backend().lower() == 'agg', \
        f"Expected Agg backend, got {matplotlib.get_backend()}"
```

**Checklist:**
- [ ] `matplotlib.use('Agg')` at top of every asset file
- [ ] No `plt.show()` calls anywhere
- [ ] No `plt.ion()` calls
- [ ] No interactive widget usage
- [ ] All output via `fig.savefig()` or returned `Figure` objects

---

### Category 5: File Export Tests

Every visualization must support export to at least PNG format. Test that exported files are valid images.

**Test pattern:**

```python
def test_png_export(plotter, data, tmp_path):
    fig = plotter.plot_raster(data)
    path = os.path.join(tmp_path, 'test_raster.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    # Verify it's a valid PNG
    with open(path, 'rb') as f:
        header = f.read(8)
    assert header[:4] == b'\x89PNG', "Invalid PNG header"
    assert os.path.getsize(path) > 1000, "PNG suspiciously small"
```

**Format matrix:**

| Format | Extension | Validation Method | Required |
|--------|-----------|-------------------|----------|
| PNG | `.png` | Check PNG magic bytes (`\x89PNG`) | Yes |
| SVG | `.svg` | Check `<svg` tag in content | Optional |
| HTML | `.html` | Check `<html` or `<!DOCTYPE` tag | Dashboard only |

---

### Category 6: Edge Case Tests

Test behavior with unusual or extreme inputs:

| Edge Case | Expected Behavior | Modules Affected |
|-----------|-------------------|------------------|
| Empty tensor (all zeros) | Plot with annotation "No spikes detected" or empty heatmap | All |
| Single neuron | Plot with single row | SpikeRaster |
| Single timestep | Plot with single column | SpikeRaster, Firing Rates |
| 1x1 attention matrix | Single-cell heatmap with annotation | AttentionHeatmapper |
| All-ones attention | Uniform heatmap, entropy = max | AttentionHeatmapper |
| Single modality competition | Bar chart with single bar, auto-win | WorkspaceVisualizer |
| Confidence = 0.0 | Always System 2, clear routing | ReasoningTrace |
| Confidence = 1.0 | Always System 1, clear routing | ReasoningTrace |
| 1 embedding point | Single dot on plot | EmbeddingProjector |
| 10000+ embeddings | Subsampled to max_points | EmbeddingProjector |
| All-zero embeddings | Warning annotation, zero-centered plot | EmbeddingProjector |
| NaN in data | Handle gracefully (skip or warn) | All |
| Very large values | Clip or scale without overflow | All |
| Mixed positive/negative | Appropriate colormap (diverging) | Membrane potential, logits |

**Test pattern:**

```python
def test_empty_spikes(plotter):
    spikes = torch.zeros(1, 50, 100)
    fig = plotter.plot_raster(spikes)
    assert fig is not None
    plt.close(fig)

def test_nan_handling(plotter):
    data = torch.randn(1, 50, 100)
    data[0, 10, 20] = float('nan')
    fig = plotter.plot_raster(data)
    assert fig is not None
    plt.close(fig)
```

---

### Category 7: Configuration Tests

Validate that `VizConfig` parameters are respected:

| Parameter | Test | Expected |
|-----------|------|----------|
| `dpi` | Set to 72, check file size < set to 300 | Lower DPI = smaller file |
| `figsize` | Set to (6,4), check figure size | `fig.get_size_inches() == (6, 4)` |
| `colormap` | Set to 'plasma', check colormap used | Colormap name in axes images |
| `dark_mode` | Set True, check background color | Dark background (< 0.3 luminance) |
| `max_neurons` | Set to 50, provide 200 neurons | Only 50 neurons plotted |
| `max_timesteps` | Set to 100, provide 500 steps | Only 100 steps shown |
| `save_dir` | Set custom dir | Files saved to correct dir |
| `output_format` | Set to 'svg' | SVG file produced |

---

### Category 8: Integration Tests

Test that visualization modules work together:

1. **Spike raster + firing rate side-by-side**: Create a figure with two subplots.
2. **Workspace competition + broadcast map**: Link competition winner to broadcast source.
3. **Reasoning routing + confidence evolution**: Verify confidence evolution starts at routing confidence.
4. **Full dashboard report**: All components rendered into a single HTML report.

**Test pattern for dashboard:**

```python
def test_dashboard_report(tmp_path):
    config = VizConfig(save_dir=str(tmp_path))
    dashboard = TrainingDashboard(log_dir=str(tmp_path), config=config)
    metrics = {
        'loss': [1.0, 0.8, 0.6, 0.4, 0.3],
        'accuracy': [0.2, 0.4, 0.6, 0.7, 0.8],
    }
    neuro = {
        'DA': [0.5, 0.6, 0.7, 0.8, 0.9],
        'ACh': [0.3, 0.4, 0.5, 0.6, 0.7],
        'NE': [0.8, 0.7, 0.6, 0.5, 0.4],
        '5-HT': [0.4, 0.4, 0.5, 0.5, 0.6],
    }
    report_path = dashboard.generate_report(
        output_dir=str(tmp_path),
        metrics=metrics,
        neuromodulator_levels=neuro,
        phase_boundaries=[2, 4],
    )
    assert os.path.exists(report_path)
    with open(report_path) as f:
        content = f.read()
    assert '<html' in content.lower() or '<!doctype' in content.lower()
    assert 'loss' in content.lower() or 'Loss' in content
    assert len(content) > 500
```

---

## Test Count Summary

| Module | Cat 1 | Cat 2 | Cat 3 | Cat 4 | Cat 5 | Cat 6 | Cat 7 | Cat 8 | Total |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| SpikeRasterPlotter | 6 | 3 | 2 | 1 | 2 | 4 | 3 | 1 | 22 |
| AttentionHeatmapper | 6 | 3 | 2 | 1 | 2 | 4 | 3 | 1 | 22 |
| WorkspaceVisualizer | 6 | 3 | 1 | 1 | 2 | 3 | 2 | 1 | 19 |
| ReasoningTraceVisualizer | 6 | 3 | 1 | 1 | 2 | 3 | 2 | 1 | 19 |
| EmbeddingProjector | 4 | 2 | 2 | 1 | 2 | 4 | 2 | 0 | 17 |
| TrainingDashboard | 4 | 2 | 1 | 1 | 2 | 2 | 2 | 2 | 16 |
| **Total** | **32** | **16** | **9** | **6** | **12** | **20** | **14** | **6** | **115** |

The `gen_viz_tests.py` script generates these 115+ test cases as pytest-compatible functions.

---

## CI/CD Integration

### Environment Requirements

```
matplotlib>=3.5.0
numpy>=1.21.0
torch>=1.12.0
scikit-learn>=1.0.0  # For t-SNE, PCA
pytest>=7.0.0
```

### Headless Configuration

```bash
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
```

### Test Execution

```bash
# Run all visualization tests
python -m pytest tests/test_visualization/ -v

# Run with coverage
python -m pytest tests/test_visualization/ --cov=brain_ai.viz --cov-report=html

# Run specific category
python -m pytest tests/test_visualization/ -k "non_empty" -v
python -m pytest tests/test_visualization/ -k "deterministic" -v
```

### Performance Constraints

| Test Category | Max Duration per Test | Rationale |
|--------------|----------------------|-----------|
| Figure non-empty | 2 seconds | Simple rendering |
| Axes validation | 1 second | Just checking labels |
| Deterministic | 5 seconds | Two renders + hash |
| File export | 3 seconds | Save + validate |
| Edge cases | 2 seconds | Should not hang |
| Integration | 10 seconds | Multi-component |

Total test suite should complete in under 3 minutes.

---

## Failure Triage

| Failure Pattern | Likely Cause | Fix |
|----------------|-------------|-----|
| "no display" error | Missing Agg backend | Add `matplotlib.use('Agg')` before imports |
| Empty PNG (0 bytes) | Figure not drawn to | Check that plotting code actually calls draw methods |
| Wrong axis labels | Label not set | Add `ax.set_xlabel()` / `ax.set_ylabel()` |
| Non-deterministic hash | Random state not fixed | Pass `random_state` to t-SNE/UMAP |
| MemoryError on large data | No subsampling | Enforce `max_neurons`, `max_points` |
| TypeError: NoneType | Missing data check | Add input validation at top of each method |
| ValueError: shape mismatch | Wrong tensor layout | Document and validate expected shapes |
| ImportError: sklearn | Optional dependency | Graceful fallback with warning |

---

## References

- pytest documentation: https://docs.pytest.org/
- matplotlib testing guide: https://matplotlib.org/stable/devel/testing.html
- brain_ai SKILL.md done-when gates
- VizConfig in `assets/viz_config_template.py`
