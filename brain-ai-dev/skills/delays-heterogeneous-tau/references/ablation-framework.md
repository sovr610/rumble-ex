# Ablation Framework: Learnable Delays and Heterogeneous Tau

Reference for the `delays-heterogeneous-tau` skill. Covers the full ablation methodology
for isolating and comparing the contribution of DCLS-style learnable synaptic delays and
per-neuron heterogeneous time constants (tau / beta) in the BrainAI spiking neural network core.

---

## 1. Config Flag Architecture

### 1.1 Primary Feature Flags in SNNConfig

Two master switches in `brain_ai/config.py::SNNConfig` gate the entire subsystem:

```python
use_learnable_delays: bool = True   # Master switch — delay modules
use_heterogeneous_tau: bool = True  # Master switch — heterogeneous tau/beta
```

Setting either master flag to `False` forces the subsystem into its baseline behavior
regardless of any sub-flag values. This makes it safe to run the baseline ablation config
without touching the sub-flag tree at all.

Existing fields on `SNNConfig` that interact with delays:

```python
beta: float = 0.95      # Fallback scalar beta when tau_mode == "homogeneous_fixed"
max_delay: int = 16     # Maximum delay in timesteps (delays_max_td alias)
```

### 1.2 DelayConfig Dataclass

Nest `DelayConfig` inside `SNNConfig` to hold all delay sub-flags:

```python
@dataclass
class DelayConfig:
    """Fine-grained control over the delay subsystem."""

    # Operational mode
    mode: str = "learnable_dcls"
    # "off"            — no delay module; tensor passes through unchanged
    # "fixed_random"   — random delays initialized at startup, frozen (no grad)
    # "learnable_dcls" — DCLS-style Gaussian interpolation + sigma annealing

    # Delay resolution
    granularity: str = "per_output"
    # "per_synapse" — one delay scalar d_{i,j} per input-output pair (densest)
    # "per_output"  — one delay per output neuron, shared across inputs (default)
    # "per_input"   — one delay per input channel, shared across outputs
    # "per_block"   — single scalar per entire linear block (fewest params)

    # Gaussian kernel schedule
    sigma_schedule: str = "decreasing"
    # "constant"   — sigma stays at sigma_init throughout training
    # "decreasing" — sigma anneals from sigma_init toward sigma_final by final epoch

    sigma_init: float = 1.0      # Starting sigma for Gaussian interpolation
    sigma_final: float = 0.1     # Target sigma after annealing
    sigma_epochs: int = 200      # Epochs over which sigma decays

    # Discretization at evaluation
    eval_discretize: bool = True
    # When True, round d_raw to nearest integer timestep at eval/inference.
    # When False, use continuous Gaussian-weighted interpolation at all times.

    # Delay range
    max_td: int = 16   # Maximum delay in integer timesteps (replaces max_delay)
    num_bins: int = 3  # K — number of integer bins in Gaussian support window
```

### 1.3 TauConfig Dataclass

Nest `TauConfig` inside `SNNConfig` to hold all tau sub-flags:

```python
@dataclass
class TauConfig:
    """Fine-grained control over the heterogeneous time-constant subsystem."""

    # Operational mode
    mode: str = "heterogeneous_learnable"
    # "homogeneous_fixed"       — scalar beta shared by all neurons, no grad (baseline)
    # "heterogeneous_fixed"     — diverse tau per neuron, initialized but frozen
    # "heterogeneous_learnable" — diverse tau per neuron, gradients enabled

    # Resolution of the tau assignment
    granularity: str = "per_neuron"
    # "per_neuron"   — independent tau for every membrane (highest expressivity)
    # "per_channel"  — one tau per feature channel, shared across spatial positions
    # "per_layer"    — single tau per LIF layer (effectively homogeneous but learnable)

    # Initialization distribution
    init: str = "heterogeneous_loguniform"
    # "homogeneous"              — all tau initialized to beta_scalar equivalent
    # "heterogeneous_gamma"      — Gamma(shape=4, scale=5) clipped to [tau_min, tau_max]
    # "heterogeneous_loguniform" — log-uniform draw in [tau_min, tau_max] (default)
    # "preset_bank"              — fixed biologically-motivated bank
    #                              (fast: 5ms, medium: 20ms, slow: 100ms columns)

    # Clipping range (in milliseconds equivalent; converted to beta internally)
    tau_min: float = 1.0    # Minimum tau value
    tau_max: float = 100.0  # Maximum tau value
```

### 1.4 Integration into SNNConfig

```python
@dataclass
class SNNConfig:
    # ... existing fields ...
    use_learnable_delays: bool = True
    use_heterogeneous_tau: bool = True
    max_delay: int = 16
    beta: float = 0.95

    # Nested sub-configs (constructed with defaults if not supplied)
    delay: DelayConfig = field(default_factory=DelayConfig)
    tau: TauConfig = field(default_factory=TauConfig)
```

During `BrainAI.__init__`, the SNN core reads `config.snn.delay` and `config.snn.tau`
after checking the master flags. If `use_learnable_delays` is `False`, `delay.mode` is
overridden to `"off"` programmatically. If `use_heterogeneous_tau` is `False`,
`tau.mode` is overridden to `"homogeneous_fixed"`.

---

## 2. Ablation Matrix

Four canonical configurations define the comparison space:

| Config Name   | delays_mode    | tau_mode                | Purpose                              |
|---------------|----------------|-------------------------|--------------------------------------|
| baseline      | off            | homogeneous_fixed       | Control — no temporal features       |
| delays_only   | learnable_dcls | homogeneous_fixed       | Isolate delay contribution           |
| hetero_only   | off            | heterogeneous_learnable | Isolate tau contribution             |
| both          | learnable_dcls | heterogeneous_learnable | Full temporal expressivity           |

### 2.1 Config Instantiation Helpers

Define factory functions that produce exact `SNNConfig` objects for each cell:

```python
def make_baseline_config(base: SNNConfig) -> SNNConfig:
    cfg = copy.deepcopy(base)
    cfg.use_learnable_delays = False
    cfg.use_heterogeneous_tau = False
    cfg.delay.mode = "off"
    cfg.tau.mode = "homogeneous_fixed"
    return cfg

def make_delays_only_config(base: SNNConfig) -> SNNConfig:
    cfg = copy.deepcopy(base)
    cfg.use_learnable_delays = True
    cfg.use_heterogeneous_tau = False
    cfg.delay.mode = "learnable_dcls"
    cfg.tau.mode = "homogeneous_fixed"
    return cfg

def make_hetero_only_config(base: SNNConfig) -> SNNConfig:
    cfg = copy.deepcopy(base)
    cfg.use_learnable_delays = False
    cfg.use_heterogeneous_tau = True
    cfg.delay.mode = "off"
    cfg.tau.mode = "heterogeneous_learnable"
    return cfg

def make_both_config(base: SNNConfig) -> SNNConfig:
    cfg = copy.deepcopy(base)
    cfg.use_learnable_delays = True
    cfg.use_heterogeneous_tau = True
    cfg.delay.mode = "learnable_dcls"
    cfg.tau.mode = "heterogeneous_learnable"
    return cfg
```

### 2.2 Controlled Variables

The following variables must be identical across all four configs in every ablation run:

- **Random seeds**: use the same seed set for weight initialization, data shuffling, and
  augmentation. Recommended set: `[42, 123, 456]`. Each seed produces one independent trial.
- **Optimizer class and hyperparameters**: same `AdamW`, same `lr`, same `weight_decay`,
  same `grad_clip`, same `beta1`/`beta2`.
- **Learning rate schedule**: same cosine decay or linear warmup — do not let one config
  benefit from a different schedule.
- **Architecture width**: same `hidden_sizes` list. Delays and tau add parameters on top of
  the shared backbone; the backbone itself must be identical.
- **Dataset split**: use a fixed split file or fixed seed for train/val/test assignment.
  Regenerating splits across configs contaminates comparisons.
- **Number of epochs and batch size**: same everywhere.
- **Surrogate gradient**: same function and `surrogate_alpha`.
- **Number of timesteps**: same `num_timesteps`.

### 2.3 Reporting Requirements

Every ablation run must produce the following artifacts before a result is considered valid:

- Accuracy and loss curves for all four configs on a single overlaid plot per benchmark.
- Final metrics table: mean +/- std across seeds for accuracy, loss, and overhead factor.
- Parameter count comparison table: count trainable parameters per config; note the delta
  introduced by delay parameters and tau parameters relative to baseline.
- Training time comparison: wall-clock seconds per epoch per config; express as
  overhead factor = `time_config / time_baseline`.
- Delay histograms (at epoch 0, mid-training, final epoch) for `delays_only` and `both`.
- Tau histograms (at epoch 0, mid-training, final epoch) for `hetero_only` and `both`.

---

## 3. Benchmark Selection

### 3.1 Primary Benchmarks — Temporal Pattern Detection

Use at least one primary benchmark in every ablation run. These datasets exhibit strong
temporal structure that delays and multi-timescale tau are expected to exploit.

#### SHD — Spiking Heidelberg Digits

- Spoken digit classification (0-9) encoded as spike trains.
- 700 input channels, approximately 10,000 training samples.
- Strong temporal structure: digit identity is carried in spike timing, not just rate.
- Well-studied in SNN temporal benchmark literature; use the standard train/test split.
- Expected sensitivity to delays: high. Expected sensitivity to tau diversity: high.

#### SSC — Spiking Speech Commands

- 35-class spoken command classification in spike-encoded format.
- Longer sequences than SHD; more challenging temporal patterns.
- Complements SHD by testing generalization across a larger label set.
- Use when a single-benchmark run has been completed on SHD and a cross-dataset check
  is required.

### 3.2 Secondary Benchmarks — Sequential Tasks

Use secondary benchmarks when the primary benchmark results are inconclusive or when
testing long-range dependency processing specifically:

#### Sequential MNIST (sMNIST)

- MNIST pixels fed one-by-one as a 784-timestep sequence.
- Baseline SNN should already pass this; ablation tests whether delays or tau change
  the learning curve shape or final accuracy.

#### Permuted Sequential MNIST (psMNIST)

- Same as sMNIST but with a fixed random permutation applied to pixel order.
- Destroys local spatial correlation; forces the model to rely on sequence memory.
- Particularly sensitive to tau diversity because different timescales must be integrated.

### 3.3 Sanity-Check Tasks

Run these on every ablation to confirm that enabling temporal features does not regress
on simpler tasks:

- **Standard MNIST** (non-sequential): accuracy must not drop by more than 0.5 percentage
  points relative to baseline. Regression here indicates a bug in the feature flag path.
- **Small time-series anomaly task**: validates that the delay/tau machinery does not break
  HTM-compatible temporal processing elsewhere in the pipeline.

### 3.4 Benchmark-Level Requirements

- Run every ablation config on at least SHD (primary) and standard MNIST (sanity check).
- Log wall-clock time per epoch for every config on every benchmark.
- Log delay and tau distributions at three checkpoints: epoch 0 (after first forward pass),
  mid-training (epoch = total_epochs // 2), and final epoch.

---

## 4. Logging Contract

### 4.1 Per-Epoch Metric Dictionary

Every training loop must emit a dictionary conforming to the following schema at the end
of each epoch. Keys are conditional on which features are active.

```python
{
    # --- Always present ---
    "epoch": int,
    "config_name": str,          # "baseline" | "delays_only" | "hetero_only" | "both"
    "seed": int,
    "benchmark": str,

    "train_loss": float,
    "val_loss": float,
    "train_acc": float,
    "val_acc": float,

    # --- Present when delays_mode != "off" ---
    "delay_mean": float,         # Mean of d_raw across all delay parameters
    "delay_std": float,          # Std of d_raw across all delay parameters
    "delay_entropy": float,      # Entropy of discretized delay histogram
    "delay_boundary_pct": float, # Fraction of delays at or beyond max_td - 1
    "sigma": float,              # Current sigma value (annealing schedule position)

    # --- Present when tau_mode != "homogeneous_fixed" ---
    "tau_mean": float,           # Mean tau in milliseconds equivalent
    "tau_std": float,            # Std tau across all neurons
    "tau_min": float,            # Minimum tau observed this epoch
    "tau_max": float,            # Maximum tau observed this epoch
    "tau_drift": float,          # Mean absolute change in tau since previous epoch
    "tau_firing_rate_corr": float, # Pearson correlation between tau and per-neuron
                                   # mean firing rate over the validation set

    # --- Always present (performance budget) ---
    "forward_time_ms": float,    # Mean wall-clock time per forward pass (ms)
    "backward_time_ms": float,   # Mean wall-clock time per backward pass (ms)
    "peak_memory_mb": float,     # Peak GPU memory in MB for one training batch
    "overhead_factor": float,    # forward_time_ms / baseline_forward_time_ms
                                 # (baseline reference measured at epoch 0 with frozen model)
}
```

Store each epoch dictionary as a line in a newline-delimited JSON file
(`metrics.jsonl`) for easy streaming and post-processing.

### 4.2 Delay Diagnostics

`delay_boundary_pct` detects saturation: if a large fraction of learned delays accumulate
at `max_td - 1` or at `0`, the capacity is either too small or the initialization is
degenerate. Log a warning when `delay_boundary_pct > 0.20`.

`delay_entropy` measures diversity of the delay distribution. Near-zero entropy indicates
collapse to a single delay value, which means the delay module has learned no timing
structure. Log a warning when `delay_entropy < 0.5` beyond epoch 10.

### 4.3 Tau Diagnostics

`tau_drift` detects rapid oscillation or instability in the tau parameters. Sustained drift
greater than 5 ms per epoch beyond epoch 20 warrants inspection of the learning rate or
clipping range.

`tau_firing_rate_corr` measures whether neurons with longer tau tend to fire less (the
expected biological relationship). A strong negative correlation (below -0.3) is expected
in healthy heterogeneous-tau runs. Positive correlation may indicate mode collapse.

### 4.4 Checkpoint Contract

Every checkpoint `state_dict` must include the following keys in addition to all standard
model parameters:

| Key | Description |
|-----|-------------|
| `d_raw` | Raw (unconstrained) delay parameter tensor |
| `tau_raw` | Raw tau parameter tensor (present only when `tau_mode` is learnable) |
| `sigma` | Buffer holding current sigma scalar for delay annealing schedule |
| `sigma_epoch` | Buffer holding the epoch index used to compute sigma position |
| `config_name` | String tag identifying the ablation config |
| `seed` | Integer seed used for this run |

Reload test requirement: for every checkpoint saved during training, perform a forward pass
with a fixed random input before saving and record the output tensor. After reloading the
checkpoint into a fresh model instance, repeat the same forward pass. The maximum absolute
difference between the two output tensors must be below `1e-5` (float32 tolerance).
Fail loudly and log the checkpoint path if this invariant is violated.

---

## 5. Performance Budget

Delay modules add a history buffer and a Gaussian interpolation step to every forward pass.
Heterogeneous tau adds per-neuron multiply operations. Both must stay within budget to remain
viable for production training.

### 5.1 Targets

| Metric | Target | Action on Breach |
|--------|--------|-----------------|
| Delay forward overhead factor | < 2.0x baseline | Warn; log to metrics; do not abort |
| Tau forward overhead factor | < 1.3x baseline | Warn; log to metrics; do not abort |
| Combined overhead factor | < 2.5x baseline | Warn; recommend granularity reduction |
| Memory overhead (peak GPU MB) | < 1.5x baseline | Warn; recommend reducing `num_bins` or `max_td` |

### 5.2 Profiling Hooks

Attach lightweight hooks at the start of training to measure the baseline forward and
backward time with all features disabled (`config_name == "baseline"`). Store this
reference value in a module-level variable so that `overhead_factor` can be computed
consistently across all subsequent configs in the same process.

```python
# Pseudocode — attach in ablation_runner.py before the training loop
_baseline_forward_ms: float = 0.0

def profile_baseline(model, dummy_input, n_warmup=10, n_measure=50):
    model.train(False)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy_input)
        t0 = time.perf_counter()
        for _ in range(n_measure):
            model(dummy_input)
        elapsed = (time.perf_counter() - t0) / n_measure * 1000  # ms
    return elapsed
```

Hardware-specific notes: on machines without a GPU, forward time is dominated by memory
bandwidth rather than compute. Overhead factors on CPU are typically higher than on GPU.
Always record hardware details (GPU model, VRAM, CPU) alongside overhead metrics.

---

## 6. Ablation Script Contract

### 6.1 Command Interface

```
scripts/ablation_runner.py --configs baseline,delays_only,hetero_only,both \
    --seeds 42,123,456 \
    --benchmark shd \
    --epochs 100 \
    --output-dir results/ablation_YYYYMMDD
```

Required arguments:

| Argument | Type | Description |
|----------|------|-------------|
| `--configs` | comma-separated list | One or more of: `baseline`, `delays_only`, `hetero_only`, `both` |
| `--seeds` | comma-separated ints | Random seeds; each produces one independent trial per config |
| `--benchmark` | str | `shd`, `ssc`, `smnist`, `psmnist`, or `mnist` |
| `--epochs` | int | Number of training epochs per trial |
| `--output-dir` | path | Root directory for all outputs |

Optional arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--base-config` | SNNConfig() | Path to a JSON/YAML base SNNConfig to inherit from |
| `--batch-size` | 128 | Batch size for all configs |
| `--lr` | 3e-4 | Peak learning rate |
| `--no-sanity-check` | False | Skip the standard MNIST sanity run |
| `--skip-plots` | False | Skip plot generation (for headless CI runs) |

### 6.2 Output Directory Structure

```
results/ablation_YYYYMMDD/
├── run_manifest.json                     # All CLI args, timestamps, hardware info
├── config_baseline_seed42/
│   ├── metrics.jsonl                     # Per-epoch metric dicts (newline-delimited JSON)
│   ├── checkpoints/
│   │   ├── epoch_000.pt
│   │   ├── epoch_050.pt
│   │   └── epoch_100.pt
│   └── logs/
│       └── train.log
├── config_baseline_seed123/
│   └── ...
├── config_baseline_seed456/
│   └── ...
├── config_delays_only_seed42/
│   └── ...
├── config_delays_only_seed123/
│   └── ...
├── config_delays_only_seed456/
│   └── ...
├── config_hetero_only_seed42/
│   └── ...
├── config_hetero_only_seed123/
│   └── ...
├── config_hetero_only_seed456/
│   └── ...
├── config_both_seed42/
│   └── ...
├── config_both_seed123/
│   └── ...
├── config_both_seed456/
│   └── ...
├── comparison_report.md                  # Auto-generated summary with tables and key findings
└── plots/
    ├── accuracy_curves.png               # All 4 configs, all seeds, val accuracy vs epoch
    ├── loss_curves.png                   # All 4 configs, all seeds, val loss vs epoch
    ├── delay_histograms.png              # Delay distributions at epochs 0, mid, final
    ├── tau_histograms.png                # Tau distributions at epochs 0, mid, final
    ├── overhead_factor.png               # Bar chart of overhead per config
    └── param_count_comparison.png        # Stacked bar: backbone + delay params + tau params
```

### 6.3 Comparison Report Format

`comparison_report.md` must be auto-generated by `ablation_runner.py` after all trials
complete. Minimum sections:

1. Run summary (date, hardware, benchmark, total wall-clock time).
2. Final metrics table: config x seed with mean +/- std row at the bottom.
3. Parameter count table: absolute counts and deltas relative to baseline.
4. Overhead factor table: forward, backward, and peak memory per config.
5. Key findings section: three to five bullet points auto-derived from the metrics
   (for example: "delays_only improved val_acc by X.X% relative to baseline").
6. Paths to all generated plots.

---

## 7. Anti-Patterns

The following practices invalidate ablation results and must be avoided:

- Do not run ablation configs with different random seeds. If `baseline` uses seed 42 and
  `delays_only` uses seed 99, the weight initialization difference confounds the comparison.
  Use the same seed set for every config.

- Do not compare configs with significantly different parameter counts without explicitly
  noting the delta. A model with 10% more parameters may outperform the baseline for reasons
  unrelated to delays or tau. Always log parameter counts and acknowledge the delta in the
  comparison report.

- Do not skip the baseline config. Without a control measurement, there is no reference
  point for evaluating the contribution of delays or tau.

- Do not use different optimizer settings across ablation configs. Adjusting the learning
  rate or weight decay for one config but not others conflates optimizer sensitivity with
  feature contribution.

- Do not report only final accuracy. Learning curve shape carries diagnostic information:
  a config that converges faster but plateaus at the same final accuracy still demonstrates
  a meaningful difference in training dynamics.

- Do not ignore the overhead factor. A config that achieves 1% higher accuracy at 10x the
  training cost is not a fair comparison for production use. Always include the overhead
  factor in the comparison report and interpret results in light of it.

- Do not disable `eval_discretize` when reporting inference results. Production inference
  will use discretized delays; reporting continuous-delay accuracy overstates real-world
  performance.

- Do not mix `tau_granularity` values across configs that are meant to represent the same
  condition. If `hetero_only` uses `per_neuron` and `both` uses `per_channel`, the tau
  capacity difference is a confound.

- Do not run fewer than two seeds per config. Single-seed results cannot distinguish
  feature contribution from initialization luck. Three seeds (42, 123, 456) is the
  minimum for reporting mean +/- std.

---

## 8. Quick Reference — Flag Combinations by Config

```
baseline:
    SNNConfig.use_learnable_delays = False
    SNNConfig.use_heterogeneous_tau = False
    SNNConfig.delay.mode = "off"
    SNNConfig.tau.mode = "homogeneous_fixed"

delays_only:
    SNNConfig.use_learnable_delays = True
    SNNConfig.use_heterogeneous_tau = False
    SNNConfig.delay.mode = "learnable_dcls"
    SNNConfig.delay.granularity = "per_output"
    SNNConfig.delay.sigma_schedule = "decreasing"
    SNNConfig.tau.mode = "homogeneous_fixed"

hetero_only:
    SNNConfig.use_learnable_delays = False
    SNNConfig.use_heterogeneous_tau = True
    SNNConfig.delay.mode = "off"
    SNNConfig.tau.mode = "heterogeneous_learnable"
    SNNConfig.tau.granularity = "per_neuron"
    SNNConfig.tau.init = "heterogeneous_loguniform"

both:
    SNNConfig.use_learnable_delays = True
    SNNConfig.use_heterogeneous_tau = True
    SNNConfig.delay.mode = "learnable_dcls"
    SNNConfig.delay.granularity = "per_output"
    SNNConfig.delay.sigma_schedule = "decreasing"
    SNNConfig.tau.mode = "heterogeneous_learnable"
    SNNConfig.tau.granularity = "per_neuron"
    SNNConfig.tau.init = "heterogeneous_loguniform"
```

---

## 9. Relationship to Existing Feature Flags

The delay and tau subsystems sit entirely within `SNNConfig` and are independent of the
system-level feature flags (`use_snn`, `use_htm`, `use_workspace`, `use_symbolic`,
`use_meta`, `use_engram`). The ablation matrix assumes `use_snn = True`; all other
system-level flags can be set to `False` during ablation to isolate the SNN contribution
without interference from HTM, workspace, or reasoning layers.

Recommended ablation model configuration:

```python
BrainAIConfig(
    use_snn=True,
    use_htm=False,
    use_workspace=False,
    use_symbolic=False,
    use_meta=False,
    use_engram=False,
    snn=<config_for_ablation_cell>,
)
```

This produces the smallest trainable model that exercises the delay and tau machinery,
minimizing confounds from other cognitive layers and reducing wall-clock time per trial.
