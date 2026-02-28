---
name: Spike Codec & Loss Pack
description: >-
  This skill should be used when the user asks to "encode spikes", "decode spike trains",
  "add spike encoding", "implement rate coding", "implement latency coding", "implement TTFS",
  "add population coding", "add delta modulation", "decode spike counts", "first-spike decoding",
  "population decoding", "add SNN loss", "implement ProbSpikes", "spike rate regularization",
  "ISI regularization", "temporal consistency loss", "membrane regularization",
  "SNN loss pack", "loss composition", "spike-count cross-entropy",
  "AMP hardening for SNN", "mixed precision SNN", "spike tensor convention",
  "axis convention batch-first", "SpikeBatch", "deterministic inference for encoders",
  "round-trip encoding test", or mentions spike I/O semantics, SNN training objectives,
  or encoding/decoding pipelines for spiking neural networks.
version: 0.1.0
---

# Spike Codec & Loss Pack

## Purpose

This skill standardizes how continuous inputs become spike trains (encoding), how output
spike trains become decisions/logits (decoding), and how SNN training losses/regularizers
are computed. It enforces a single axis convention, AMP-safe patterns, and composable loss
terms with per-component logging.

## Key Files

| Target Module | Template Asset | Purpose |
|---|---|---|
| `brain_ai/core/encoding.py` | `assets/encoding_template.py` | SpikeBatch, axis helpers, 5 encoder classes (7 type variants) |
| `brain_ai/core/decoding.py` | `assets/decoding_template.py` | DecoderOutput, 4 decoders, ensemble |
| `brain_ai/core/losses.py` | `assets/losses_template.py` | 5 loss terms, SNNLossComposer |
| `brain_ai/config.py` (extend) | `assets/codec_config_template.py` | EncodingConfig, DecodingConfig, LossConfig |

## Axis Convention

All internal spike tensors are **batch-first**: `(B, T, ...)`.

```python
BATCH_DIM = 0
TIME_DIM  = 1
# spikes.shape = (B, T, N) or (B, T, C, H, W)
```

Two helpers handle boundary conversion with third-party (time-first) code:

```python
def time_to_batch_first(x: Tensor) -> Tensor:
    return x.transpose(0, 1)   # (T,B,...) -> (B,T,...)

def batch_to_time_first(x: Tensor) -> Tensor:
    return x.transpose(0, 1)   # (B,T,...) -> (T,B,...)
```

All encoders, decoders, and losses call these helpers — never roll custom permutations.

**Migration note**: existing `brain_ai/core/encoding.py` uses `(T, B, D)`. Wrap existing
encoders with `time_to_batch_first()` at the boundary during migration.

## SpikeBatch Contract

Every encoder returns a `SpikeBatch` dataclass:

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `spikes` | `(B, T, N)` or `(B, T, C, H, W)` | bool / uint8 / float{0,1} | Spike tensor |
| `mask` | `(B, T)` | bool | True where valid (all-True unless ragged) |
| `aux` | dict | — | rates, spike_times, population_map, normalization stats |

Properties: `time_steps`, `batch_size`, `feature_shape`, `is_binary()`, `device`.

## Encoders

### A) Rate Coding (Bernoulli)

Normalize input x to rate p, sample per timestep: `spike = (U < p)`.

| Option | Default | Purpose |
|---|---|---|
| `normalization` | `"minmax"` | none / minmax / sigmoid / clamp |
| `deterministic_eval` | `True` | Threshold instead of sampling at inference |
| `rate_gain` / `rate_bias` | 1.0 / 0.0 | p = clamp(gain * x + bias) |
| `seed` | `None` | torch.Generator for reproducibility |

**AMP rule**: all `torch.rand` calls in fp32, cast spikes to bool after.

### B) Latency / TTFS

Map intensity to spike timing: stronger input fires earlier.

- **Latency** (multi-spike): emit spike(s) at computed time over `[0, T-1]`
- **TTFS** (strict): exactly one spike per neuron per window

Three mappings: `linear`, `exponential` (t = tau * log(1/x)), `log`.

Options: `t_min`, `t_max`, `allow_no_spike`, `jitter` (training only).
Returns `spike_times: (B, N)` with -1 for no-spike in aux.

### C) Population Coding

Expand each input dim into P neurons with Gaussian tuning curves:

```
activation = exp(-(x - center)^2 / (2 * sigma^2))
```

Options: `population_size`, `sigma`, `centers` (linear/learned/uniform).
Returns `population_map` in aux for downstream decoding.

### D) Delta Modulation

Emit spikes when signal changes exceed threshold. ON/OFF channels (2x feature expansion).
Stateful — call `reset()` between sequences.

## Decoders

Every decoder returns a `DecoderOutput`:

| Field | Shape | Purpose |
|---|---|---|
| `logits_proxy` | `(B, C)` | Float tensor for loss computation |
| `prediction` | `(B,)` | argmax / argmin result |
| `confidence` | `(B,)` | Normalized confidence in [0, 1] |
| `aux` | dict | count_histogram, first_spike_times, margin |

### A) Rate / Spike-Count Decoding

```python
counts = spikes.float().sum(dim=1)  # (B, N) — accumulate in fp32
logits = counts / temperature
prediction = counts.argmax(dim=-1)
```

### B) First-Spike Decoding

```python
cumsum = spikes.cumsum(dim=1)
first_mask = (cumsum == 1) & spikes.bool()
t_first = first_mask.float().argmax(dim=1)  # (B, N)
prediction = t_first.argmin(dim=-1)          # earliest wins
logits = -t_first.float() / T               # earlier = higher
```

### C) Population Decoding

Sum counts per group from `population_map`. Classification: argmax over groups.
Regression: expectation over centers weighted by group rates.

### D) Membrane Decoding

Use final membrane `membrane[:, -1, :]` or max over time. Useful when spikes are sparse.

## Loss Pack

### Architecture

`SNNLossComposer` sums weighted `LossTerm` instances. Each term returns
`(scalar_loss, diagnostics_dict)`. The composer returns:

```python
(total_loss, {
    "total": float,
    "components": {"probspikes": float, "rate_reg": float, ...},
    "diagnostics": {"probspikes": {...}, "rate_reg": {...}},
    "weights": {"probspikes": 1.0, "rate_reg": 0.1, ...}
})
```

### A) ProbSpikes (Spike-Count Cross-Entropy)

```python
counts = spikes.float().sum(dim=1)           # (B, C) in fp32 ALWAYS
loss = F.cross_entropy(counts / temperature, targets)
```

Hardening: fp32 accumulation, clamp + eps before log, temperature before softmax.

### B) Spike-Rate Regularization

```python
rate = spikes.float().mean(dim=1)            # (B, N) in fp32
loss = ((rate - target_rate) ** 2).mean()    # + range penalty
```

Per-layer targets supported. Diagnostics: dead/saturated neuron fractions.

### C) Temporal Consistency

Penalize firing-rate flicker: `|r[t] - r[t-1]|` or windowed variance.

Config: `window_size`, `penalty_type` (l1 / l2 / variance).

### D) ISI Regularization (Vectorized)

**Must use `F.conv1d`** — no Python loops:

```python
kernel = exp(-t / tau) for t in [1..W]       # (1, 1, W) fp32
spikes_flat = spikes.reshape(B*N, 1, T)
refractory = F.conv1d(spikes_flat.float(), kernel, padding=W-1)[:, :, :T]
loss = (spikes_flat.float() * refractory).mean()
```

### E) Membrane Potential Regularization

`excess = ReLU(|membrane| - max_membrane)`, loss = mean(excess^2).

## AMP Hardening Rules

| Rule | Applies To |
|---|---|
| All sampling ops (`torch.rand`) in fp32 | Encoders |
| Cast spikes to bool/uint8 after generation | Encoders |
| Count accumulation `.float().sum(dim=1)` in fp32 | Decoders, ProbSpikes |
| `softmax` / `log_softmax` in fp32 | ProbSpikes, Decoders |
| Clamp + eps before division/log | ProbSpikes, Rate reg |
| Temperature scaling before softmax | ProbSpikes |
| ISI conv kernel in fp32 | ISI reg |
| Normalize counts by T to prevent scaling | All losses |

## Config Surface

### EncodingConfig

| Field | Default | Options |
|---|---|---|
| `type` | `"rate_bernoulli"` | rate_bernoulli, rate_poisson, rate_deterministic, latency, ttfs, population, delta |
| `num_steps` | 25 | int > 0 |
| `normalization` | `"minmax"` | none, minmax, sigmoid, clamp |
| `deterministic_eval` | `True` | bool |
| `population_size` | 8 | int > 0 |
| `latency_mapping` | `"exponential"` | linear, exponential, log |

### DecodingConfig

| Field | Default | Options |
|---|---|---|
| `type` | `"rate"` | rate, first_spike, population, membrane, ensemble |
| `temperature` | 1.0 | float > 0 |
| `eps` | 1e-7 | float > 0 |
| `population_mode` | `"classification"` | classification, regression |

### LossConfig

| Field | Default | Purpose |
|---|---|---|
| `w_probspikes` | 1.0 | ProbSpikes weight |
| `w_rate` | 0.1 | Rate regularization weight |
| `w_temporal` | 0.01 | Temporal consistency weight |
| `w_isi` | 0.01 | ISI regularization weight |
| `w_membrane` | 0.001 | Membrane regularization weight |
| `target_rate` | 0.1 | Target firing rate |
| `probspikes_temperature` | 1.0 | Temperature for count softmax |
| `isi_refractory_window` | 3 | ISI kernel width |

Presets: `rate_classification_preset()`, `ttfs_classification_preset()`,
`population_classification_preset()`, `regression_preset()`.

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| NaN loss under AMP | Count sum in float16 underflows | Force `.float()` before `.sum()` |
| Loss scales with T | Counts not normalized | Divide counts by T or use temperature |
| ISI timeout on large N | Python loop over neurons | Use `F.conv1d` with reshaping |
| Degenerate TTFS (all fire at t=0) | Missing jitter / bad normalization | Add jitter in training, check input range |
| Dead population neurons | Sigma too narrow | Widen sigma or use learned centers |
| Zero gradients through encoder | Thresholding kills gradients | Use surrogate gradient in downstream SNN |

## Anti-Patterns

- **Hardcoding `x.permute(1,0,...)`** — use `time_to_batch_first()` / `batch_to_time_first()`
- **Sampling in float16** — all stochastic operations must use fp32 probabilities
- **ISI with Python for-loops** — must be vectorized with `F.conv1d`
- **Forgetting DeltaEncoder `.reset()`** — causes cross-sequence contamination
- **Population coding without `population_map`** — decoder cannot reconstruct groups
- **Missing `seed` / `Generator`** — non-reproducible experiments
- **`log(prob)` without eps** — NaN when any count is zero

## Additional Resources

### Reference Files

- **`references/spike-encoders.md`** — Full encoder specifications, axis helpers, AMP rules, migration guide
- **`references/spike-decoders.md`** — Full decoder specifications, DecoderOutput contract, ensemble patterns
- **`references/loss-pack.md`** — All loss terms, composition architecture, logging contract, normalization
- **`references/testing-matrix.md`** — Round-trip tests, AMP tests, axis tests, integration tests, checklist

### Asset Templates

- **`assets/encoding_template.py`** — SpikeBatch, axis helpers, 5 encoder classes, factory, self-test
- **`assets/decoding_template.py`** — DecoderOutput, 4 decoder classes, ensemble, factory, self-test
- **`assets/losses_template.py`** — 5 LossTerm classes, SNNLossComposer, metrics, self-test
- **`assets/codec_config_template.py`** — EncodingConfig, DecodingConfig, LossConfig, presets, serialization

### Scripts

- **`scripts/validate_codec_losses.py`** — Runtime contract validation (encoding/decoding/loss checks)
- **`scripts/gen_codec_tests.py`** — Generates `tests/test_codec_losses.py` (~70+ test cases)
- **`scripts/amp_stress_test.py`** — AMP stress test across encoders/decoders/losses with scaling checks
