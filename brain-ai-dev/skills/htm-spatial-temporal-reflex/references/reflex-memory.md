# Reflex Memory Reference

Detailed implementation reference for the Reflex Memory acceleration system
in the brain-ai-dev HTM skill. Load this when Claude needs deep Reflex
implementation details beyond the high-level skill instructions.

---

## 1. Overview

Reflex Memory makes frequent patterns O(1) to predict by caching
"current SDR + context" to "predicted next SDR" mappings. It sits in front
of the Temporal Memory (TM) as a fast lookup layer:

```
Input SDR ──► Reflex Lookup (O(1))
                ├── HIT  → return cached prediction
                └── MISS → run full TM (O(segments * synapses)), observe for promotion
```

Core properties:

- Use Locality-Sensitive Hashing (LSH) for fast approximate matching of
  sparse distributed representations.
- Only promote patterns after repeated stable observations — never cache
  noise or one-shot inputs.
- Preserve baseline TM behavior exactly. Reflex is a pure acceleration
  layer, not a replacement. Any promoted pattern must reproduce the same
  prediction the TM would have produced.
- Maintain correctness guarantees via baseline snapshots and periodic
  verification.
- Support full `state_dict` / `load_state_dict` for checkpointing.

### Relationship to Existing Code

The existing implementation lives in `brain_ai/temporal/htm.py` at class
`ReflexMemory` (lines 783-988) and `AcceleratedHTM` (lines 990-1198). The
skill upgrade adds:

| Capability | Existing | Upgraded |
|---|---|---|
| Promotion policy | Implicit (access count) | Explicit (count + confidence + stability) |
| Confidence tracking | None | Per-entry confidence score |
| Demotion | None | Automatic when accuracy drops |
| Context hashing | None | Winner-cells context hash in key |
| Baseline verification | None | Snapshot at promotion, periodic check |
| State dict | Partial (buffers only) | Full serialize/deserialize |
| Observation vs promoted | Single table | Separate observation and promoted tables |
| Statistics | Basic hits/misses | Full metrics (evictions, confidence, utilization) |

---

## 2. What Reflex Stores

Each entry contains all information needed to serve a cached prediction and
verify correctness:

```python
@dataclass
class ReflexEntry:
    key_hash: int           # 64-bit SDR hash (order-independent)
    key_indices: Tensor     # (K,) int — original SDR active indices for exact verification
    context_hash: int       # 64-bit hash of winner_cells (optional context disambiguation)
    pred_indices: Tensor    # (K_pred,) int — predicted next active columns
    hit_count: int          # total access frequency (subject to decay)
    last_seen_step: int     # global step counter at last access (for LRU eviction)
    confidence: float       # stability of prediction across observations (0.0 to 1.0)
    promoted: bool          # whether this entry serves the fast path
    baseline_snapshot: Tensor  # TM prediction at promotion time (for correctness verification)
```

### Field Semantics

- `key_hash`: Use for fast bucket lookup. Compute via the order-independent
  SDR hashing function (Section 3). Two SDRs with the same active indices
  in any order must produce the same hash.
- `key_indices`: Keep the raw sorted indices for exact Jaccard verification
  after hash-based candidate retrieval. Store sorted ascending.
- `context_hash`: Hash of the TM's `winner_cells` state at the time of
  observation. Include this in the lookup key so that the same SDR in
  different sequence contexts maps to different predictions. Set to 0 when
  context is unavailable.
- `pred_indices`: The predicted next SDR (active column indices). Store
  sorted ascending. This is what the fast path returns on a hit.
- `hit_count`: Increment on every access (lookup or observation). Subject
  to periodic multiplicative decay (Section 7.3).
- `last_seen_step`: Update to the current global step on every access.
  Used for LRU eviction ordering.
- `confidence`: Fraction of observations where the TM produced the same
  prediction as the most common prediction for this entry. Compute as
  `agreement_count / total_observations`. Only promote when this exceeds
  `min_confidence`.
- `promoted`: Boolean flag. Observation entries have `promoted=False`.
  Entries on the fast path have `promoted=True`.
- `baseline_snapshot`: Snapshot of the TM's prediction at promotion time.
  Use for periodic correctness verification (Section 10).

---

## 3. SDR Hashing (Order-Independent)

SDR hashing must produce the same hash regardless of index ordering.
Two approaches:

### 3.1 XOR with Per-Index Mixing (Preferred)

Use an order-independent combine via XOR with golden-ratio hash mixing
per index:

```python
def sdr_hash(indices: Tensor) -> Tensor:
    """
    Compute order-independent hash of SDR indices.

    Args:
        indices: (B, K) int tensor of active indices

    Returns:
        (B,) uint64 hash values
    """
    GOLDEN = 0x9E3779B97F4A7C15
    B, K = indices.shape
    h = torch.zeros(B, dtype=torch.long, device=indices.device)
    for k in range(K):
        mixed = indices[:, k] * GOLDEN
        mixed = (mixed >> 32) ^ mixed   # avalanche mixing
        h ^= mixed
    return h
```

Properties:
- Order-independent: XOR is commutative and associative.
- Avalanche: The golden-ratio multiply plus shift-XOR ensures single-bit
  input changes propagate to many output bits.
- Batch-friendly: Operates on (B, K) tensors.
- Collision rate: Acceptable for approximate matching; exact verification
  follows (Section 4.3).

### 3.2 Sort-Then-Hash (Alternative)

Sort the indices first, then hash the sorted tuple:

```python
def sdr_hash_sorted(indices: Tensor) -> Tensor:
    sorted_idx, _ = indices.sort(dim=-1)
    # Hash the byte representation
    return torch.tensor([
        hash(row.cpu().numpy().tobytes()) for row in sorted_idx
    ], device=indices.device)
```

This is deterministic but slower due to the sort and Python-level loop.
Prefer the XOR approach for GPU-resident computation.

### 3.3 Context Hashing

Apply the same `sdr_hash` function to the `winner_cells` indices to
produce `context_hash`. Combine the SDR hash and context hash:

```python
def combined_hash(sdr_indices: Tensor, context_indices: Tensor) -> Tensor:
    h1 = sdr_hash(sdr_indices)
    h2 = sdr_hash(context_indices)
    # Combine with rotation to avoid symmetry
    return h1 ^ ((h2 << 17) | (h2 >> 47))
```

---

## 4. LSH Lookup Strategy

### 4.1 Hash Tables

Maintain `num_tables` independent hash functions (default 8). Each table
maps a hash bucket to a list of entry indices:

```python
# Random projection matrices: (num_tables, N_columns, bits_per_hash)
# Initialize once at construction, persist across state_dict
projections = torch.randn(num_tables, N_columns, bits_per_hash)

def compute_lsh(sdr_dense: Tensor) -> Tensor:
    """
    Compute LSH codes for dense SDR representation.

    Args:
        sdr_dense: (B, N_columns) bool or float tensor

    Returns:
        (B, num_tables, bits_per_hash) binary hash codes
    """
    proj = sdr_dense.float() @ projections  # (B, num_tables, bits_per_hash)
    return (proj > 0).long()
```

Each hash code is a `bits_per_hash`-bit integer per table. Store entries in
`num_tables` separate hash-table dicts keyed by the integer code.

### 4.2 Candidate Retrieval

For a query SDR:

1. Compute LSH codes across all `num_tables` tables.
2. For each table, look up the corresponding bucket to get candidate entry
   indices.
3. Take the union of candidates across all tables (multi-probe).
4. Cap the candidate list at `max_candidates` (default 32) to bound
   verification cost. If more candidates found, keep those with the
   highest hash-match count (number of tables that agree).

```python
def retrieve_candidates(self, query_dense: Tensor) -> List[int]:
    codes = self.compute_lsh(query_dense)  # (1, num_tables, bits_per_hash)
    candidate_counts = {}  # entry_idx -> num_tables_matched

    for t in range(self.num_tables):
        bucket_key = codes[0, t].item()  # or tuple for multi-bit
        for entry_idx in self.hash_tables[t].get(bucket_key, []):
            candidate_counts[entry_idx] = candidate_counts.get(entry_idx, 0) + 1

    # Sort by number of matching tables (descending), take top max_candidates
    sorted_candidates = sorted(candidate_counts.keys(),
                                key=lambda i: candidate_counts[i],
                                reverse=True)
    return sorted_candidates[:self.max_candidates]
```

### 4.3 Exact Verification

For each candidate, compute Jaccard similarity with the query SDR:

```python
def jaccard_similarity(query_indices: Tensor, candidate_indices: Tensor) -> float:
    """
    Compute Jaccard similarity between two sets of indices.

    Args:
        query_indices: (K_q,) sorted int tensor
        candidate_indices: (K_c,) sorted int tensor

    Returns:
        Jaccard coefficient in [0, 1]
    """
    query_set = set(query_indices.tolist())
    cand_set = set(candidate_indices.tolist())
    intersection = len(query_set & cand_set)
    union = len(query_set | cand_set)
    return intersection / max(union, 1)
```

Accept the candidate if `jaccard >= similarity_threshold` (default 0.9).
Return the best match (highest Jaccard). If context hashing is enabled,
also verify `context_hash` matches.

---

## 5. Promotion Policy

### 5.1 Observation Phase

When the TM produces a prediction for a pattern:

1. Compute `sdr_hash` of the current `active_columns`.
2. Optionally compute `context_hash` of the current `winner_cells`.
3. Look up the combined hash in the observation table.
4. If found: increment `hit_count`, record the TM's prediction, update
   `confidence` (see below).
5. If not found: create a new observation entry with `hit_count=1`,
   `confidence=0.0`, `promoted=False`.

### 5.2 Confidence Tracking

Track prediction stability across observations:

```python
def update_confidence(entry: ReflexEntry, tm_prediction: Tensor):
    """
    Update confidence based on whether TM prediction matches stored prediction.

    Call on every observation of an existing entry.
    """
    current_pred_set = set(entry.pred_indices.tolist())
    tm_pred_set = set(tm_prediction.tolist())

    # Jaccard between stored prediction and current TM prediction
    overlap = len(current_pred_set & tm_pred_set)
    union = len(current_pred_set | tm_pred_set)
    agreement = overlap / max(union, 1)

    # Exponential moving average of agreement
    alpha = 2.0 / (entry.hit_count + 1)  # adaptive smoothing
    entry.confidence = (1 - alpha) * entry.confidence + alpha * agreement
```

Confidence reflects how stable the TM's prediction is for this pattern.
A confidence of 1.0 means every observation produced the exact same
prediction. A confidence below `min_confidence` indicates the pattern is
still unstable and should not be promoted.

### 5.3 Promotion Criteria

Promote an observation entry to the fast path when ALL of:

- `hit_count >= promotion_threshold` (default 5)
- `confidence >= min_confidence` (default 0.8)
- The pattern has been observed at least `promotion_threshold` times with
  consistent predictions

```python
def check_promotion(entry: ReflexEntry, config: ReflexConfig) -> bool:
    return (
        not entry.promoted
        and entry.hit_count >= config.promotion_threshold
        and entry.confidence >= config.min_confidence
    )
```

### 5.4 Promotion Action

When promoting an entry:

1. Store a `baseline_snapshot` of the TM's current prediction for this
   pattern. This is the ground truth for correctness verification.
2. Set `entry.promoted = True`.
3. Insert the entry into the promoted hash tables for fast-path lookup.
4. Remove the entry from the observation table (it now lives in the
   promoted table).

```python
def promote_entry(self, entry: ReflexEntry, tm_prediction: Tensor):
    entry.promoted = True
    entry.baseline_snapshot = tm_prediction.clone()
    self._insert_into_promoted_tables(entry)
    self._remove_from_observation_tables(entry)
```

### 5.5 Demotion

If a promoted entry starts producing wrong predictions, demote it:

1. Track post-promotion accuracy over a rolling window of the last N
   lookups (default N=20).
2. On each hit, compare the cached `pred_indices` against the TM's actual
   prediction (run TM in verification mode periodically, not on every hit).
3. If accuracy drops below `demotion_threshold` (default 0.7): demote.
4. On demotion: reset `hit_count`, set `promoted=False`, move back to the
   observation table, clear `baseline_snapshot`.

```python
def check_demotion(entry: ReflexEntry, recent_accuracy: float,
                   config: ReflexConfig) -> bool:
    return (
        entry.promoted
        and recent_accuracy < config.demotion_threshold
    )

def demote_entry(self, entry: ReflexEntry):
    entry.promoted = False
    entry.hit_count = 0
    entry.confidence = 0.0
    entry.baseline_snapshot = None
    self._remove_from_promoted_tables(entry)
    self._insert_into_observation_tables(entry)
```

---

## 6. Fast Path vs Slow Path

The forward pass dispatches based on Reflex lookup result:

```
Input SDR ──► LSH lookup in promoted entries
  |
  ├── HIT (jaccard >= threshold):
  │     Return cached pred_indices
  │     Set anomaly = 0 (known pattern, no surprise)
  │     Increment reflex_hits counter
  │     Update entry.last_seen_step
  │     Update entry.hit_count
  │
  └── MISS:
        Run full TM computation (slow path)
        Increment reflex_misses counter
        Store/update observation entry for potential future promotion
        Return TM's prediction and anomaly score
```

### Integration with AcceleratedHTM

The `AcceleratedHTM.forward()` method orchestrates this:

```python
def _forward_single(self, x: Tensor, learn: bool, device) -> Dict[str, Tensor]:
    # 1. Try fast path
    reflex_result = self.reflex.lookup_promoted(x)

    if reflex_result is not None:
        pred, confidence, entry_idx = reflex_result
        self.stats.reflex_hits += 1
        return {
            'features': pred,
            'anomaly': torch.tensor(0.0, device=device),
            'from_reflex': torch.tensor(True, device=device),
            'confidence': torch.tensor(confidence, device=device),
        }

    # 2. Slow path: full TM
    self.stats.reflex_misses += 1
    htm_result = self.htm(x, learn=learn)

    # 3. Observe for promotion
    if learn:
        self.reflex.observe(x, htm_result['features'],
                           context=htm_result.get('winner_cells'))

    return {
        'features': htm_result['features'],
        'anomaly': htm_result['anomaly'],
        'from_reflex': torch.tensor(False, device=device),
        'confidence': torch.tensor(1.0 - htm_result['anomaly'].item(), device=device),
    }
```

### Background TM Update on Hit

Optionally, run the TM in the background even on Reflex hits to keep the
TM's state up to date. This prevents the TM from falling behind when
Reflex serves most requests:

```python
if reflex_hit and learn and self.training:
    with torch.no_grad():
        self.htm(x, learn=True)  # keep TM synapses current
```

Enable this via `update_tm_on_hit=True` in configuration. Default is True
during training, False during inference.

---

## 7. Guardrails

### 7.1 Memory Limits

| Parameter | Default | Purpose |
|---|---|---|
| `max_observations` | 50000 | Cap on observation table entries |
| `max_promoted` | 10000 | Cap on promoted (fast-path) entries |
| `max_candidates_per_bucket` | 64 | Prevent hash collision blowup in any single bucket |
| `max_candidates` | 32 | Cap candidates per query for verification |

When a table reaches its limit, apply LRU eviction before inserting a new
entry. Never exceed the configured maximum.

### 7.2 LRU Eviction

Track `last_seen_step` for each entry. When the table is full:

```python
def evict_lru(self, table: str):
    """
    Evict the least-recently-used entry from the specified table.

    Args:
        table: 'observations' or 'promoted'
    """
    entries = self.observations if table == 'observations' else self.promoted
    if not entries:
        return

    # Find entry with the smallest last_seen_step
    oldest_idx = min(range(len(entries)),
                     key=lambda i: entries[i].last_seen_step)

    entry = entries[oldest_idx]
    self._remove_from_hash_tables(entry, table)
    del entries[oldest_idx]
    self.stats.eviction_count += 1
```

Eviction priority rules:
- Evict observation entries before promoted entries (promoted entries have
  proven their value).
- Within a table, evict the entry with the oldest `last_seen_step`.
- Tie-break by lowest `hit_count`.

### 7.3 Periodic Decay

Apply multiplicative decay to all `hit_count` values to prevent old
patterns from persisting indefinitely:

```python
def decay_access_counts(self, decay_rate: float = 0.99):
    """
    Apply multiplicative decay to all hit counts.

    Call every `decay_interval` steps (default 1000).
    Demote entries whose decayed hit_count falls below promotion_threshold.
    """
    for entry in self.observations:
        entry.hit_count = int(entry.hit_count * decay_rate)

    for entry in self.promoted:
        entry.hit_count = int(entry.hit_count * decay_rate)
        # Check if decayed count drops below threshold
        if entry.hit_count < self.config.promotion_threshold:
            self.demote_entry(entry)
```

Decay ensures that patterns which were frequent in the past but are no
longer relevant get gradually evicted.

### 7.4 Bucket Size Limits

Prevent any single LSH bucket from growing too large:

```python
def _insert_into_bucket(self, table_idx: int, bucket_key: int, entry_idx: int):
    bucket = self.hash_tables[table_idx].setdefault(bucket_key, [])
    if len(bucket) >= self.config.max_candidates_per_bucket:
        # Evict oldest entry from this bucket
        oldest = min(bucket, key=lambda idx: self.entries[idx].last_seen_step)
        bucket.remove(oldest)
    bucket.append(entry_idx)
```

---

## 8. State Dict / Checkpointing

Reflex state must be fully saveable and loadable for training checkpoints
and model deployment. All entries, hash tables, projections, and statistics
must round-trip through `state_dict()` / `load_state_dict()`.

### 8.1 Serialization

```python
def state_dict(self) -> Dict[str, Any]:
    return {
        'observations': self._serialize_entries(self.observations),
        'promoted': self._serialize_entries(self.promoted),
        'projections': self.projections.clone(),
        'step_counter': self.step_counter,
        'statistics': {
            'reflex_hits': self.stats.reflex_hits,
            'reflex_misses': self.stats.reflex_misses,
            'eviction_count': self.stats.eviction_count,
        },
        'config': asdict(self.config),
    }

def _serialize_entries(self, entries: List[ReflexEntry]) -> List[Dict]:
    return [
        {
            'key_hash': e.key_hash,
            'key_indices': e.key_indices.cpu(),
            'context_hash': e.context_hash,
            'pred_indices': e.pred_indices.cpu(),
            'hit_count': e.hit_count,
            'last_seen_step': e.last_seen_step,
            'confidence': e.confidence,
            'promoted': e.promoted,
            'baseline_snapshot': e.baseline_snapshot.cpu() if e.baseline_snapshot is not None else None,
        }
        for e in entries
    ]
```

### 8.2 Deserialization

```python
def load_state_dict(self, state: Dict[str, Any]):
    self.observations = self._deserialize_entries(state['observations'])
    self.promoted = self._deserialize_entries(state['promoted'])
    self.projections = state['projections'].to(self.device)
    self.step_counter = state['step_counter']
    self.stats.reflex_hits = state['statistics']['reflex_hits']
    self.stats.reflex_misses = state['statistics']['reflex_misses']
    self.stats.eviction_count = state['statistics']['eviction_count']
    # Rebuild hash tables from deserialized entries
    self._rebuild_hash_tables()

def _deserialize_entries(self, data: List[Dict]) -> List[ReflexEntry]:
    return [
        ReflexEntry(
            key_hash=d['key_hash'],
            key_indices=d['key_indices'].to(self.device),
            context_hash=d['context_hash'],
            pred_indices=d['pred_indices'].to(self.device),
            hit_count=d['hit_count'],
            last_seen_step=d['last_seen_step'],
            confidence=d['confidence'],
            promoted=d['promoted'],
            baseline_snapshot=(d['baseline_snapshot'].to(self.device)
                               if d['baseline_snapshot'] is not None else None),
        )
        for d in data
    ]

def _rebuild_hash_tables(self):
    """Rebuild all LSH hash tables from current entries."""
    for t in range(self.config.num_tables):
        self.obs_hash_tables[t] = {}
        self.promo_hash_tables[t] = {}

    for idx, entry in enumerate(self.observations):
        codes = self._compute_entry_lsh(entry)
        for t in range(self.config.num_tables):
            self._insert_into_bucket(self.obs_hash_tables, t, codes[t], idx)

    for idx, entry in enumerate(self.promoted):
        codes = self._compute_entry_lsh(entry)
        for t in range(self.config.num_tables):
            self._insert_into_bucket(self.promo_hash_tables, t, codes[t], idx)
```

### 8.3 Compatibility with nn.Module state_dict

Register `projections` as a buffer so it is automatically included in the
parent module's `state_dict()`. The entry tables require custom logic
because they contain variable-length lists of dataclass instances. Override
`state_dict()` and `load_state_dict()` at the `ReflexMemory` module level
and merge with `super().state_dict()`.

---

## 9. Statistics and Logging

Track per-run metrics for monitoring and debugging:

```python
@dataclass
class ReflexStatistics:
    reflex_hits: int = 0          # fast-path hits
    reflex_misses: int = 0        # slow-path fallbacks
    promoted_count: int = 0       # current number of promoted entries
    observation_count: int = 0    # current number of observation entries
    eviction_count: int = 0       # total LRU evictions performed
    demotion_count: int = 0       # total demotions performed
    promotion_count: int = 0      # total promotions performed

    @property
    def hit_rate(self) -> float:
        total = self.reflex_hits + self.reflex_misses
        return self.reflex_hits / max(total, 1)

    @property
    def memory_utilization(self) -> float:
        """Fraction of total capacity in use."""
        used = self.observation_count + self.promoted_count
        capacity = max_observations + max_promoted  # from config
        return used / max(capacity, 1)

    @property
    def mean_confidence(self) -> float:
        """Average confidence of promoted entries."""
        # Compute from promoted entries list
        ...

    def to_dict(self) -> Dict[str, float]:
        return {
            'reflex_hits': self.reflex_hits,
            'reflex_misses': self.reflex_misses,
            'hit_rate': self.hit_rate,
            'promoted_count': self.promoted_count,
            'observation_count': self.observation_count,
            'memory_utilization': self.memory_utilization,
            'mean_confidence': self.mean_confidence,
            'eviction_count': self.eviction_count,
            'demotion_count': self.demotion_count,
            'promotion_count': self.promotion_count,
        }
```

### Logging Integration

Log statistics periodically (e.g., every 1000 steps) at INFO level:

```python
import logging
logger = logging.getLogger('brain_ai.temporal.reflex')

def log_statistics(self):
    stats = self.stats.to_dict()
    logger.info(
        f"Reflex: hit_rate={stats['hit_rate']:.3f} "
        f"promoted={stats['promoted_count']} "
        f"observations={stats['observation_count']} "
        f"evictions={stats['eviction_count']} "
        f"demotions={stats['demotion_count']}"
    )
```

---

## 10. Correctness Requirement

"Reflex matches baseline for promoted patterns" — this is the central
invariant. Violating it means the acceleration layer changes model
behavior.

### 10.1 Baseline Snapshot

At promotion time, store the TM's prediction as `baseline_snapshot`:

```python
entry.baseline_snapshot = tm_prediction.clone().detach()
```

This snapshot represents the ground truth: what the TM would predict for
this input pattern in this context.

### 10.2 Verification Protocol

Periodically (every `verify_interval` steps, default 5000), sample a
subset of promoted entries and verify:

```python
def verify_promoted_entries(self, htm: HTMLayer, sample_size: int = 100):
    """
    Verify that promoted entries still match TM predictions.

    Run the TM on the stored input pattern and compare against
    the cached prediction.
    """
    if not self.promoted:
        return

    sample = random.sample(self.promoted, min(sample_size, len(self.promoted)))

    for entry in sample:
        # Reconstruct input SDR as dense tensor
        sdr_dense = torch.zeros(htm.config.column_count, device=self.device)
        sdr_dense[entry.key_indices] = 1.0

        # Run TM
        tm_result = htm(sdr_dense, learn=False)
        tm_pred = torch.where(tm_result['features'] > 0)[0]

        # Compare
        cached_set = set(entry.pred_indices.tolist())
        tm_set = set(tm_pred.tolist())

        if cached_set != tm_set:
            jaccard = len(cached_set & tm_set) / max(len(cached_set | tm_set), 1)
            if jaccard < self.config.demotion_threshold:
                logger.warning(
                    f"Reflex entry {entry.key_hash:#x} drifted: "
                    f"jaccard={jaccard:.3f}, demoting"
                )
                self.demote_entry(entry)
```

### 10.3 Test Assertion

In unit tests, run baseline TM and Reflex TM side-by-side on repeated
sequences and assert exact match for promoted patterns:

```python
def test_reflex_matches_baseline():
    """Reflex predictions must exactly match baseline TM for promoted patterns."""
    htm = create_htm_layer(input_size=512, column_count=2048)
    reflex = ReflexMemory(config=ReflexConfig(promotion_threshold=3))

    # Train on a repeating sequence
    sequence = [generate_random_sdr(2048, sparsity=0.02) for _ in range(5)]
    for epoch in range(10):
        htm.reset()
        for sdr in sequence:
            result = htm(sdr, learn=True)
            reflex.observe(sdr, result['features'])

    # After promotion, verify all promoted entries
    for entry in reflex.promoted:
        sdr_dense = torch.zeros(2048)
        sdr_dense[entry.key_indices] = 1.0

        baseline = htm(sdr_dense, learn=False)
        baseline_pred = set(torch.where(baseline['features'] > 0)[0].tolist())
        reflex_pred = set(entry.pred_indices.tolist())

        assert reflex_pred == baseline_pred, (
            f"Mismatch for entry {entry.key_hash:#x}: "
            f"reflex={reflex_pred}, baseline={baseline_pred}"
        )
```

---

## 11. Configuration Surface

All Reflex parameters are grouped in a single dataclass:

```python
@dataclass
class ReflexConfig:
    # Feature flag
    enabled: bool = True

    # Memory capacity
    max_observations: int = 50000    # observation table capacity
    max_promoted: int = 10000        # promoted table capacity

    # Promotion / demotion
    promotion_threshold: int = 5     # min hit_count for promotion
    min_confidence: float = 0.8      # min confidence for promotion
    demotion_threshold: float = 0.7  # accuracy below this triggers demotion
    demotion_window: int = 20        # rolling window size for post-promotion accuracy

    # LSH parameters
    similarity_threshold: float = 0.9  # Jaccard threshold for accepting a match
    num_tables: int = 8                # number of independent LSH tables
    bits_per_hash: int = 12            # bits per hash code per table
    max_candidates: int = 32           # max candidates per query
    max_candidates_per_bucket: int = 64  # prevent bucket blowup

    # Decay
    decay_rate: float = 0.99           # multiplicative decay factor
    decay_interval: int = 1000         # steps between decay applications

    # Verification
    verify_interval: int = 5000        # steps between promoted entry verification
    verify_sample_size: int = 100      # entries to verify per check

    # Background TM update
    update_tm_on_hit: bool = True      # keep TM current even on Reflex hits
```

### Integration with HTMConfig

Embed `ReflexConfig` inside the existing `HTMConfig`:

```python
@dataclass
class HTMConfig:
    # ... existing fields ...
    reflex: ReflexConfig = field(default_factory=ReflexConfig)
```

Access as `config.reflex.enabled`, `config.reflex.max_promoted`, etc.

### Scale Presets

| Preset | max_observations | max_promoted | num_tables | promotion_threshold |
|---|---|---|---|---|
| minimal (tests) | 100 | 50 | 4 | 3 |
| production_1b | 50000 | 10000 | 8 | 5 |
| production_7b | 200000 | 50000 | 16 | 5 |

---

## 12. Migration from Existing Code

### 12.1 Current Implementation

The existing `ReflexMemory` in `brain_ai/temporal/htm.py` (lines 783-988)
uses:

- Dense tensor buffers for `patterns`, `predictions`, `access_counts`,
  `timestamps`, and `valid_mask`.
- A single LSH projection matrix (`hash_projections`) with Hamming
  similarity for lookup.
- LRU eviction based on importance score (access_count / recency).
- Moving-average update of predictions on re-observation.
- Basic hit/miss statistics.
- No promotion/demotion distinction — all stored patterns are on the fast
  path.
- No confidence tracking.
- No baseline snapshot or correctness verification.
- No context hashing (winner_cells not included in key).

The `AcceleratedHTM` (lines 990-1198) wraps `HTMLayer` + `ReflexMemory`
with:

- Fast-path lookup via `self.rm.lookup(x)`.
- Slow-path fallback to full HTM on miss.
- Promotion based on `htm_call_counts` dict (raw `hash(numpy.tobytes())`).
- Optional background TM update on hit.

### 12.2 Changes Required

1. **Add explicit observation vs promoted tables.** Replace the single
   `patterns` / `predictions` buffer pair with two separate data
   structures: `self.observations: List[ReflexEntry]` and
   `self.promoted: List[ReflexEntry]`.

2. **Add confidence tracking.** Implement `update_confidence()` as
   described in Section 5.2. Track per-entry confidence as an EMA of
   prediction agreement.

3. **Add baseline snapshot.** Store `baseline_snapshot` on promotion.
   Implement `verify_promoted_entries()` as described in Section 10.2.

4. **Add demotion mechanism.** Implement `check_demotion()` and
   `demote_entry()` as described in Section 5.5. Track rolling accuracy
   for promoted entries.

5. **Add context hashing.** Include `winner_cells` hash in the lookup key
   via `combined_hash()`. Pass context from TM through `AcceleratedHTM`.

6. **Add proper state_dict / load_state_dict.** Implement full
   serialization as described in Section 8. Rebuild hash tables on load.

7. **Add statistics tracking.** Replace basic `hits`/`misses` with full
   `ReflexStatistics` dataclass. Add eviction, demotion, promotion
   counters.

8. **Replace dense buffer storage with entry list.** The current approach
   stores patterns as `(max_patterns, pattern_dim)` tensors. For the
   upgraded version, use `List[ReflexEntry]` for flexibility (variable-
   size SDRs, optional fields). Keep LSH hash tables as dicts for O(1)
   bucket lookup.

9. **Replace hash(numpy.tobytes()) with sdr_hash().** The current
   `AcceleratedHTM` uses `hash(x.detach().cpu().numpy().tobytes())` for
   promotion counting. Replace with the order-independent `sdr_hash()`
   from Section 3.

### 12.3 Migration Steps

1. Define `ReflexEntry` dataclass and `ReflexConfig` dataclass.
2. Implement `sdr_hash()` and `combined_hash()` functions.
3. Rewrite `ReflexMemory.__init__()` to create observation and promoted
   tables, LSH projections, and statistics.
4. Implement `observe()` method (observation phase).
5. Implement `lookup_promoted()` method (fast path).
6. Implement `promote_entry()` and `demote_entry()`.
7. Implement `state_dict()` and `load_state_dict()`.
8. Update `AcceleratedHTM._forward_single()` to use new API.
9. Add verification logic.
10. Write unit tests covering all invariants.

### 12.4 Backward Compatibility

Keep the existing `ReflexMemory` class available as `ReflexMemoryV1` for
backward compatibility with saved checkpoints. The new implementation
should be `ReflexMemory` (replacing the old one). Provide a migration
utility:

```python
def migrate_v1_to_v2(v1_state: Dict) -> Dict:
    """Convert ReflexMemoryV1 state_dict to v2 format."""
    # All V1 entries become observations (unpromoted) in V2
    ...
```

---

## 13. Anti-Patterns

Avoid these common mistakes when implementing or extending Reflex Memory:

### 13.1 Exact SDR as dict key

Do not use `tuple(sdr_indices.tolist())` or `hash(tensor.tobytes())` as
a dictionary key for matching. SDRs require approximate matching because
slight variations in the active column set (due to noise or SP learning)
should still match. Use LSH with Jaccard verification instead.

### 13.2 Promoting after 1 observation

Do not promote a pattern after a single observation. One-shot patterns may
be noise, transient, or part of a rare sequence. Require at least
`promotion_threshold` (default 5) observations with consistent predictions
(`confidence >= min_confidence`) before promoting.

### 13.3 No eviction policy

Do not allow unbounded memory growth. Always enforce `max_observations` and
`max_promoted` limits with LRU eviction. Without eviction, memory usage
grows linearly with the number of unique patterns seen.

### 13.4 No demotion mechanism

Do not allow promoted entries to persist forever without verification. The
TM's internal state evolves as it learns new sequences, so a prediction
that was correct at promotion time may become stale. Implement demotion
when post-promotion accuracy drops below `demotion_threshold`.

### 13.5 Reflex without baseline verification

Do not assume cached predictions remain correct indefinitely. Always store
`baseline_snapshot` at promotion time and periodically verify against the
TM. Silent prediction drift is the most dangerous failure mode because it
changes model behavior without any error signal.

### 13.6 Context-free hashing

Do not hash only the active columns without including sequence context.
The same SDR in different positions within different sequences should
produce different predictions. Include `winner_cells` (or equivalent
context state) in the hash key via `context_hash`.

### 13.7 Not decaying access counts

Do not keep access counts growing monotonically. Without decay, a pattern
that was very frequent early in training will never be evicted even if it
becomes completely irrelevant later. Apply multiplicative decay every
`decay_interval` steps.

### 13.8 Storing dense prediction tensors

Do not store full `(N_columns,)` dense tensors as predictions. SDRs are
sparse by definition (~2% active). Store only the sorted active indices
`(K_pred,)` to reduce memory by ~50x.

### 13.9 Modifying TM state on fast path

Do not update TM segment permanences or cell states using Reflex
predictions. The Reflex path skips the TM entirely. If the TM needs to
stay current, run it in the background with `torch.no_grad()` as a
separate operation (controlled by `update_tm_on_hit`).

---

## Appendix A: Performance Characteristics

### Lookup Complexity

| Operation | Time | Space |
|---|---|---|
| LSH hash computation | O(N * bits_per_hash * num_tables) | O(num_tables * bits_per_hash) |
| Candidate retrieval | O(num_tables * bucket_size) | O(max_candidates) |
| Jaccard verification | O(max_candidates * K) | O(K) |
| Total fast-path | O(N * bits + num_tables * bucket + max_cand * K) | O(max_candidates) |
| Full TM (slow path) | O(num_segments * synapses_per_segment) | O(num_cells) |

Where:
- N = column_count (2048)
- K = active columns (~40 at 2% sparsity)
- bits = bits_per_hash (12)
- bucket = average bucket size
- max_cand = max_candidates (32)

For typical parameters, the fast path is 10-100x faster than full TM.

### Memory Footprint

Per entry (approximate):

| Field | Size |
|---|---|
| key_hash | 8 bytes |
| key_indices | K * 8 bytes (~320 bytes) |
| context_hash | 8 bytes |
| pred_indices | K_pred * 8 bytes (~320 bytes) |
| hit_count | 4 bytes |
| last_seen_step | 8 bytes |
| confidence | 4 bytes |
| promoted | 1 byte |
| baseline_snapshot | K_pred * 8 bytes (~320 bytes) |
| **Total per entry** | **~1 KB** |

At max capacity (50000 observations + 10000 promoted): ~60 MB.

### Throughput Targets

- Fast-path hit: < 50 microseconds per pattern (GPU), < 200 microseconds (CPU)
- Full TM fallback: 1-10 milliseconds per pattern (depending on segment count)
- Target hit rate after warmup: > 60% for repetitive workloads

---

## Appendix B: Interaction with Training Phases

Reflex Memory is relevant during:

- **Phase 3 (HTM training):** Enable observation but not promotion.
  Let the TM learn stable sequences first.
- **Phase 4+ (downstream training):** Enable promotion. Patterns that
  survive Phase 3 training are stable enough to cache.
- **Inference:** Enable full fast path. Freeze promotions (no new
  observations) for deterministic behavior, or keep learning for
  continual adaptation.

Configure via training phase scripts:

```python
# Phase 3: observe only
config.reflex.enabled = True
config.reflex.promotion_threshold = 999999  # effectively disable promotion

# Phase 4+: enable promotion
config.reflex.promotion_threshold = 5

# Inference: freeze
config.reflex.enabled = True
# No new observations, only fast-path lookups
```
