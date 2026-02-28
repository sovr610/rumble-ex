---
name: HTM Spatial Pooler + Temporal Memory + Reflex Acceleration
description: >-
  This skill should be used when the user asks to "implement spatial pooler", "add temporal memory",
  "implement HTM", "add sequence learning", "create SDR utilities", "implement SDR hashing",
  "add anomaly detection", "implement reflex memory", "add reflex acceleration",
  "implement online learning", "add Hebbian learning", "implement permanence updates",
  "add boosting", "implement duty cycle", "add inhibition", "implement segment matching",
  "add dendritic segments", "implement CSR segment store", "add synapse pruning",
  "implement anomaly likelihood", "add sequence prediction", "implement TM",
  "add SP column encoding", "implement fallback predictor", "add LSTM/GRU fallback",
  "implement SequenceOutput contract", "add SDR overlap", "implement SDR Jaccard",
  "add reflex promotion", "implement LSH lookup", "add pattern caching",
  or mentions HTM column-cell architecture, online sequence memory, sparse distributed
  representations, or distribution shift detection in spiking/temporal pipelines.
version: 0.1.0
---

# HTM Spatial Pooler + Temporal Memory + Reflex Acceleration

## Purpose

This skill standardizes the online sequence learning subsystem: converting dense features
into Sparse Distributed Representations (SDRs), learning temporal transitions without
backprop, predicting next patterns, emitting anomaly signals, and accelerating frequent
patterns via Reflex Memory caching. It enforces a unified SequenceOutput contract across
HTM and all fallback predictors.

## Key Files

| Target Module | Template Asset | Purpose |
|---|---|---|
| `brain_ai/temporal/sdr_utils.py` | `assets/sdr_utils_template.py` | SDR index form, conversions, overlap, Jaccard, hashing |
| `brain_ai/temporal/spatial_pooler.py` | `assets/spatial_pooler_template.py` | SP with deterministic tie-breaking, binarization, boosting |
| `brain_ai/temporal/temporal_memory.py` | `assets/temporal_memory_template.py` | TM with CSR segment store, vectorized matching, SequenceOutput |
| `brain_ai/temporal/reflex_memory.py` | `assets/reflex_memory_template.py` | Reflex with LSH, promotion/demotion, state_dict |
| `brain_ai/temporal/fallback_predictors.py` | `assets/fallback_predictors_template.py` | LSTM/GRU/Transformer returning SequenceOutput |
| `brain_ai/config.py` (extend) | `assets/htm_config_template.py` | SPConfig, TMConfig, ReflexConfig, FallbackConfig, presets |

## Public Contract

All temporal predictors (HTM and fallbacks) implement the same interface:

```python
reset(batch_size: int, device: torch.device)
step(x_t: Tensor, learn: bool = True) -> SequenceOutput
predict(x_seq: Tensor, learn: bool = True) -> List[SequenceOutput]
```

Input `x_t` accepts dense features `(B, D)` float OR SDR indices `(B, K)` int.

## SequenceOutput Contract

Every step returns:

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `sdr` | `(B, K)` | int | Active column indices (current SDR) |
| `pred_sdr` | `(B, K_pred)` | int | Predicted next column indices |
| `anomaly_score` | `(B,)` | float | Anomaly in [0, 1] — fraction of unpredicted columns |
| `aux` | dict | — | burst_count, overlap, duty_cycles, reflex_hits, segment_stats |

## SDR Representation

All SDRs use **fixed-sparsity index form**: `(B, K)` int tensors.

```python
K = int(column_count * sparsity)  # e.g., 2% of 2048 = 41
```

Core utilities:

```python
indices_to_dense(indices, N) -> (B, N) bool
dense_to_indices(dense, K) -> (B, K) int
sdr_overlap(a, b) -> (B,) int       # shared indices count
sdr_jaccard(a, b) -> (B,) float     # similarity in [0, 1]
sdr_hash(indices) -> (B,) int64     # order-independent hash
```

## Spatial Pooler

Converts inputs into stable SDRs with controlled sparsity.

| Step | Operation | Output |
|---|---|---|
| Binarize | topk / threshold / learned_gate | `(B, N_input)` bool |
| Overlap | count connected synapses matching active bits | `(B, N_col)` int |
| Boost | `overlap * boost_factor` | `(B, N_col)` float |
| Inhibit | top-K with deterministic tie-breaking | `(B, K)` int |
| Learn | Hebbian permanence ± for active columns | permanences updated |

Key rules:
- Overlap counts in int32/fp32 (never fp16)
- Permanence updates in fp32 (even if stored fp16)
- Deterministic tie-breaking via stable sort on `(overlap, column_id)`
- Duty cycle boosting: under-utilized columns get `boost > 1`

## Temporal Memory

Learns sequence transitions via cells-in-columns and dendritic segments.

### Activation Logic

For each active column:
- **Predicted** (has predictive cells): activate only predicted cells
- **Unpredicted** (burst): activate ALL cells, pick winner by best matching segment

### Segment Matching

A segment matches when `active_synapse_count >= activation_threshold`.

### CSR Segment Store

Segments and synapses stored in flat CSR-style arrays:

```python
seg_cell: (S,) int32       # owning cell
seg_start: (S,) int32      # index into synapse arrays
seg_len: (S,) int16        # synapse count
syn_src_cell: (M,) int32   # presynaptic cell
syn_perm: (M,) float32     # permanence
```

Vectorized matching via `segment_reduce()` — no Python loops over segments.

### Learning Rules (Hebbian, No Backprop)

| Condition | Action |
|---|---|
| Correctly predicted column | Reinforce winning segment (+inc active, -dec inactive) |
| Bursting column | Create/grow segment on winner cell → prior winner_cells |
| False prediction | Punish segment (-predicted_dec) |

## Anomaly Scoring

```python
anomaly = 1 - |predicted_columns ∩ active_columns| / |active_columns|
```

Smoothed via rolling Gaussian anomaly likelihood (window of last 1000 scores).

## Reflex Memory Acceleration

Caches frequent SDR→prediction mappings for O(1) retrieval.

| Phase | Behavior |
|---|---|
| Observation | Store TM predictions, track hit_count and confidence |
| Promotion | After N stable observations → promote to fast path |
| Fast path | LSH lookup → Jaccard verify → return cached prediction |
| Demotion | If accuracy drops below threshold → demote back |

Guardrails: LRU eviction, memory caps, periodic access count decay, state_dict support.

## Fallback Predictors

LSTM/GRU/Transformer predictors returning identical SequenceOutput:

| Backend | Parameters | Learning | Use Case |
|---|---|---|---|
| LSTM | Trainable (gradient) | Backprop | General sequence prediction |
| GRU | Trainable (gradient) | Backprop | Lighter weight alternative |
| Transformer | Trainable (gradient) | Backprop | Long-range dependencies |

Anomaly computed identically to HTM: overlap-based comparison of predicted vs actual SDRs.

## Configuration Surface

### SPConfig

| Field | Default | Purpose |
|---|---|---|
| `column_count` | 2048 | Number of SP columns |
| `sparsity` | 0.02 | Target active fraction |
| `binarization_mode` | `"topk"` | passthrough, topk, threshold, learned_gate |
| `boost_strength` | 3.0 | Boosting aggressiveness |
| `inhibition_mode` | `"global"` | global, local |

### TMConfig

| Field | Default | Purpose |
|---|---|---|
| `cells_per_column` | 32 | Cells per column (context depth) |
| `activation_threshold` | 13 | Min synapses for segment match |
| `max_new_synapses` | 20 | Synapses per new segment |
| `max_total_segments` | 100000 | Segment capacity |
| `max_total_synapses` | 2000000 | Synapse capacity |

### ReflexConfig

| Field | Default | Purpose |
|---|---|---|
| `promotion_threshold` | 5 | Observations before promotion |
| `similarity_threshold` | 0.9 | Jaccard threshold for match |
| `num_tables` | 8 | LSH hash tables |
| `max_promoted` | 10000 | Promoted entry capacity |

Presets: `htm_minimal_preset()`, `htm_dev_preset()`, `htm_production_1b_preset()`,
`htm_production_3b_preset()`, `htm_production_7b_preset()`.

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Non-deterministic SDRs | Unstable tie-breaking in top-k | Use stable sort with column_id offset |
| Permanence drift under AMP | Update math in fp16 | Force fp32 for permanence updates |
| Dead columns (never activate) | Missing or weak boosting | Increase boost_strength, check duty cycles |
| TM segment explosion | No synapse pruning | Enable prune_threshold, cap segments/cell |
| Slow segment matching | Dict-based storage + Python loops | Use CSR store + scatter_add |
| Reflex caching garbage | Promoting after 1 observation | Set promotion_threshold >= 3 |
| Stale Reflex predictions | No demotion mechanism | Enable demotion with accuracy tracking |
| Fallback output mismatch | Different return types | Ensure all return SequenceOutput |

## Anti-Patterns

- **Dense (N_col × N_input) permanence matrix** — use sparse potential pools
- **Dict-based segment storage** — use CSR flat arrays for vectorized ops
- **Gradient-based TM learning** — TM uses pure Hebbian rules, no optimizer
- **Python loops over segments** — vectorize with segment_reduce / scatter_add
- **Sampling in fp16** — all stochastic and count operations in fp32
- **Reflex without demotion** — stale predictions persist forever
- **Forgetting winner_cells between steps** — breaks TM learning
- **Per-batch segment store** — segments are shared (learned over time)

## Additional Resources

### Reference Files

- **`references/spatial-pooler.md`** — Full SP spec: binarization, overlap, inhibition, boosting, permanence learning
- **`references/temporal-memory.md`** — Full TM spec: cells/columns, CSR store, activation, prediction, learning rules
- **`references/reflex-memory.md`** — Reflex spec: LSH lookup, promotion/demotion, guardrails, state_dict
- **`references/testing-matrix.md`** — All test cases: online learning, anomaly, reflex equivalence, storage invariants

### Asset Templates

- **`assets/sdr_utils_template.py`** — SDR index form, conversions, overlap, Jaccard, hashing, self-test
- **`assets/spatial_pooler_template.py`** — SP with binarization, inhibition, boosting, learning, self-test
- **`assets/temporal_memory_template.py`** — TM with CSR store, SequenceOutput, learning, anomaly, self-test
- **`assets/reflex_memory_template.py`** — Reflex with LSH, promotion, demotion, state_dict, self-test
- **`assets/fallback_predictors_template.py`** — LSTM/GRU/Transformer with SequenceOutput, factory, self-test
- **`assets/htm_config_template.py`** — SPConfig, TMConfig, ReflexConfig, FallbackConfig, presets, upgrade

### Scripts

- **`scripts/validate_htm.py`** — Runtime contract validation (SP/TM/Reflex/Fallback checks)
- **`scripts/gen_htm_tests.py`** — Generates `tests/test_htm_temporal.py` (~100+ test cases)
- **`scripts/htm_benchmark.py`** — Performance benchmarking (throughput, latency, memory, scaling)
