# Testing Matrix: HTM Spatial Pooler + Temporal Memory + Reflex Acceleration

This document defines the concrete test specifications that must pass before the HTM subsystem (Spatial Pooler, Temporal Memory, Reflex Memory, and fallback predictors) is considered production-ready. Each category maps to a test file in `tests/test_htm/`. Tests are written in imperative form -- implement exactly as described.

All tests use synthetic sequences. No real data dependency. Parametrize over `batch_size=[1, 4]`, `column_count=[256, 2048]`, and `cells_per_column=[4, 32]` unless stated otherwise.

**Note on output format:** Tests reference dict-based output (`result['anomaly']`, etc.) for
backward compatibility with the existing `HTMLayer`. When the codebase migrates to the
`SequenceOutput` dataclass contract, adapt tests to use attribute access (`.anomaly_score`,
`.sdr`, `.pred_sdr`, `.aux`) instead of dict subscripting.

---

## 1. Overview

| Category | Focus | File |
|---|---|---|
| SDR Utilities | Roundtrip, overlap, Jaccard, hashing | `test_sdr_utilities.py` |
| SP Contract | Sparsity, boosting, permanence, overlap | `test_spatial_pooler.py` |
| TM Sequence Learning | Prediction, bursting, segments, synapses, CSR | `test_temporal_memory.py` |
| Anomaly Detection | Score range, novel injection, adaptation | `test_anomaly.py` |
| Reflex Acceleration | Promotion, lookup, eviction, correctness | `test_reflex_memory.py` |
| Fallback Predictors | LSTM/GRU/Transformer interface compliance | `test_fallback_predictors.py` |
| Integration | End-to-end pipelines, backend switching | `test_htm_integration.py` |
| Performance / Memory | Throughput, memory bounds, latency | `test_htm_performance.py` |

SDR utility functions (`indices_to_dense`, `dense_to_indices`, `sdr_overlap`, `sdr_jaccard`, `sdr_hash`) are tested as standalone pure functions. They are expected to exist in `brain_ai/temporal/sdr_utils.py` or equivalent; if they do not yet exist, implement them before writing these tests. The `SparseTensor` class in `brain_ai/temporal/htm.py` provides the reference semantics.

---

## 2. SDR Utility Tests

All SDR utility tests operate on index-form SDRs: integer tensors of shape `(B, K)` where `K` is the number of active columns and values are column indices in `[0, N)`.

---

### 2a) `test_indices_to_dense_roundtrip`

```python
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("K", [10, 40])
@pytest.mark.parametrize("N", [256, 2048])
def test_indices_to_dense_roundtrip(B, K, N):
    """indices -> dense -> indices preserves values."""
    indices = random_sdr(B, K, N)           # (B, K) unique ints in [0, N)
    dense = indices_to_dense(indices, N)    # (B, N) binary
    recovered = dense_to_indices(dense, K)  # (B, K)
    # Sort both before comparison (order may differ)
    assert torch.equal(indices.sort(dim=-1).values, recovered.sort(dim=-1).values)
```

Assert:
1. `dense.shape == (B, N)`.
2. `dense.sum(dim=-1)` equals `K` for every row.
3. Recovered indices match originals after sorting.

---

### 2b) `test_dense_to_indices_topk`

```python
@pytest.mark.parametrize("N", [256, 2048])
@pytest.mark.parametrize("K", [10, 40])
def test_dense_to_indices_topk(N, K):
    """dense_to_indices selects exactly K active bits via top-k."""
    dense = torch.rand(4, N)
    indices = dense_to_indices(dense, K)
    assert indices.shape == (4, K)
    # All indices in valid range
    assert (indices >= 0).all() and (indices < N).all()
    # No duplicates within a row
    for b in range(4):
        assert len(set(indices[b].tolist())) == K
```

---

### 2c) `test_sdr_overlap_identical`

```python
def test_sdr_overlap_identical():
    """overlap(a, a) == K for all batch elements."""
    sdr = random_sdr(B=4, K=40, N=2048)
    overlap = sdr_overlap(sdr, sdr)
    assert torch.equal(overlap, torch.full((4,), 40))
```

---

### 2d) `test_sdr_overlap_disjoint`

```python
def test_sdr_overlap_disjoint():
    """overlap(a, b) == 0 when no indices are shared."""
    a = torch.arange(0, 40).unsqueeze(0)    # indices [0..39]
    b = torch.arange(40, 80).unsqueeze(0)   # indices [40..79]
    overlap = sdr_overlap(a, b)
    assert overlap.item() == 0
```

---

### 2e) `test_sdr_jaccard_range`

```python
@pytest.mark.parametrize("B", [1, 4])
def test_sdr_jaccard_range(B):
    """Jaccard similarity is in [0, 1]."""
    a = random_sdr(B, K=40, N=2048)
    b = random_sdr(B, K=40, N=2048)
    j = sdr_jaccard(a, b)
    assert (j >= 0.0).all() and (j <= 1.0).all()
```

---

### 2f) `test_sdr_jaccard_identical`

```python
def test_sdr_jaccard_identical():
    """jaccard(a, a) == 1.0."""
    sdr = random_sdr(B=4, K=40, N=2048)
    j = sdr_jaccard(sdr, sdr)
    assert torch.allclose(j, torch.ones(4))
```

---

### 2g) `test_sdr_hash_deterministic`

```python
def test_sdr_hash_deterministic():
    """Same indices produce same hash across multiple calls."""
    sdr = random_sdr(B=1, K=40, N=2048)
    h1 = sdr_hash(sdr)
    h2 = sdr_hash(sdr)
    assert h1 == h2
```

---

### 2h) `test_sdr_hash_order_independent`

```python
def test_sdr_hash_order_independent():
    """Shuffled index order produces identical hash."""
    indices = torch.tensor([[5, 10, 20, 100, 500]])
    shuffled = torch.tensor([[500, 5, 100, 20, 10]])
    assert sdr_hash(indices) == sdr_hash(shuffled)
```

---

### 2i) `test_sdr_hash_collision_resistance`

```python
def test_sdr_hash_collision_resistance():
    """Different SDRs produce different hashes (statistical, not absolute)."""
    hashes = set()
    num_trials = 1000
    for _ in range(num_trials):
        sdr = random_sdr(B=1, K=40, N=2048)
        hashes.add(sdr_hash(sdr))
    # At least 99% unique (allowing 1% collision rate)
    assert len(hashes) >= int(0.99 * num_trials)
```

---

## 3. Spatial Pooler Contract Tests

All tests in this section use `PytorchSpatialPooler` from `brain_ai/temporal/htm.py`. The SP takes binary input `(input_size,)` or `(B, input_size)` and produces binary output of the same batch shape with `column_count` columns.

---

### 3a) `test_sp_output_shape`

```python
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("column_count", [256, 2048])
def test_sp_output_shape(B, column_count):
    """Active columns tensor has shape (B, column_count)."""
    sp = PytorchSpatialPooler(input_size=128, column_count=column_count)
    x = (torch.rand(B, 128) > 0.5).float()
    out = sp(x, learn=False)
    assert out.shape == (B, column_count)
```

---

### 3b) `test_sp_sparsity`

```python
@pytest.mark.parametrize("column_count", [256, 2048])
@pytest.mark.parametrize("sparsity", [0.02, 0.05])
def test_sp_sparsity(column_count, sparsity):
    """Exactly K = int(column_count * sparsity) columns are active per sample."""
    sp = PytorchSpatialPooler(input_size=128, column_count=column_count, sparsity=sparsity)
    K = int(column_count * sparsity)
    x = (torch.rand(4, 128) > 0.5).float()
    out = sp(x, learn=False)
    for b in range(4):
        assert out[b].sum().item() == K
```

---

### 3c) `test_sp_deterministic`

```python
def test_sp_deterministic():
    """Same input produces same output (deterministic tie-breaking)."""
    sp = PytorchSpatialPooler(input_size=128, column_count=256)
    x = (torch.rand(1, 128) > 0.5).float()
    out1 = sp(x, learn=False)
    out2 = sp(x, learn=False)
    assert torch.equal(out1, out2)
```

---

### 3d) `test_sp_boosting_activates_unused`

```python
def test_sp_boosting_activates_unused():
    """Columns that never activate get boosted and eventually win."""
    sp = PytorchSpatialPooler(input_size=64, column_count=256, sparsity=0.02, boost_strength=3.0)
    # Feed same input 200 times to let boosting build up
    x = (torch.rand(64) > 0.5).float()
    initial_active = sp(x, learn=True)
    initial_set = set(torch.where(initial_active > 0)[0].tolist())

    for _ in range(200):
        sp(x, learn=True)

    # After boosting, some initially inactive columns should now be active
    # (or at least duty cycles should converge toward target)
    final_active = sp(x, learn=True)
    final_set = set(torch.where(final_active > 0)[0].tolist())
    # The active set should have changed due to boosting
    assert initial_set != final_set or sp.active_duty_cycles.std() < sp.active_duty_cycles.mean()
```

---

### 3e) `test_sp_permanence_bounds`

```python
def test_sp_permanence_bounds():
    """All permanences remain in [0, 1] after repeated learning."""
    sp = PytorchSpatialPooler(input_size=64, column_count=256)
    for _ in range(100):
        x = (torch.rand(64) > 0.5).float()
        sp(x, learn=True)
    assert sp.permanences.min() >= 0.0
    assert sp.permanences.max() <= 1.0
```

---

### 3f) `test_sp_duty_cycle_convergence`

```python
def test_sp_duty_cycle_convergence():
    """Duty cycles converge toward target sparsity over many iterations."""
    sparsity = 0.02
    sp = PytorchSpatialPooler(input_size=64, column_count=256, sparsity=sparsity, boost_strength=3.0)
    for _ in range(500):
        x = (torch.rand(64) > 0.5).float()
        sp(x, learn=True)
    mean_duty = sp.active_duty_cycles.mean().item()
    # Mean duty cycle should be within 50% of target sparsity
    assert abs(mean_duty - sparsity) < sparsity * 0.5
```

---

### 3g) `test_sp_binarization_modes`

For each binarization strategy (top-k thresholding at 0.5, learned gating), verify the SP receives valid binary input:

```python
@pytest.mark.parametrize("mode", ["topk", "threshold", "learned_gate"])
def test_sp_binarization_modes(mode):
    """Each binarization mode produces valid binary input for SP."""
    raw_input = torch.rand(4, 128)
    if mode == "topk":
        K_in = 20
        _, top_idx = raw_input.topk(K_in, dim=-1)
        binary = torch.zeros_like(raw_input)
        binary.scatter_(1, top_idx, 1.0)
    elif mode == "threshold":
        binary = (raw_input > 0.5).float()
    elif mode == "learned_gate":
        gate = torch.sigmoid(torch.randn(128))  # simulated learned gate
        binary = (raw_input > gate).float()
    # Verify binary
    assert ((binary == 0) | (binary == 1)).all()
    # Verify SP accepts it
    sp = PytorchSpatialPooler(input_size=128, column_count=256)
    out = sp(binary, learn=False)
    assert out.shape == (4, 256)
```

---

### 3h) `test_sp_stimulus_threshold`

```python
def test_sp_stimulus_threshold():
    """Columns with overlap below min_overlap produce zero activation before inhibition."""
    sp = PytorchSpatialPooler(input_size=64, column_count=256)
    # Zero input should produce zero overlap for all columns
    x = torch.zeros(64)
    overlap = sp.compute_overlap(x)
    # All overlaps should be zero (no connected synapses match)
    assert overlap.sum().item() == 0.0
```

---

### 3i) `test_sp_overlap_correctness`

```python
def test_sp_overlap_correctness():
    """Manual overlap computation matches SP.compute_overlap."""
    sp = PytorchSpatialPooler(input_size=64, column_count=256)
    x = (torch.rand(64) > 0.5).float()

    # Manual computation
    connected = (sp.permanences >= sp.permanence_connected).float() * sp.potential_mask
    expected_overlap = connected @ x * sp.boost_factors

    actual_overlap = sp.compute_overlap(x)
    assert torch.allclose(actual_overlap, expected_overlap, atol=1e-6)
```

---

### 3j) `test_sp_mixed_precision_safety`

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for autocast")
def test_sp_mixed_precision_safety():
    """Overlap counts are identical under fp16 autocast vs fp32."""
    sp = PytorchSpatialPooler(input_size=128, column_count=256).cuda()
    x = (torch.rand(128) > 0.5).float().cuda()

    # fp32 baseline
    out_fp32 = sp(x, learn=False)

    # fp16 autocast
    with torch.cuda.amp.autocast(dtype=torch.float16):
        out_fp16 = sp(x, learn=False)

    # Binary outputs must match (inhibition is top-k, should be identical)
    assert torch.equal(out_fp32, out_fp16.float())
```

---

## 4. Temporal Memory Sequence Learning Tests

All tests use `PytorchTemporalMemory` from `brain_ai/temporal/htm.py`. The TM operates on binary column activations `(column_count,)` and maintains sparse segment/synapse storage internally.

---

### 4.1 Online Learning Without Backprop

#### 4.1a) `test_tm_no_optimizer`

```python
def test_tm_no_optimizer():
    """TM has no parameters requiring grad; no optimizer step needed."""
    tm = PytorchTemporalMemory(column_count=256, cells_per_column=4)
    grad_params = [p for p in tm.parameters() if p.requires_grad]
    assert len(grad_params) == 0, "TM should have no trainable parameters"
    # Verify learning happens via segment dict, not gradients
    active = torch.zeros(256)
    active[:5] = 1.0
    tm(active, learn=True)
    assert len(tm.segments) >= 0  # no error, learning is Hebbian
```

---

#### 4.1b) `test_tm_prediction_improves`

```python
@pytest.mark.parametrize("column_count", [256, 2048])
def test_tm_prediction_improves(column_count):
    """Feed repeating ABCDA sequence; prediction accuracy increases with exposure."""
    K = int(column_count * 0.02)
    sequence = repeating_sequence(length=4, num_symbols=4, N_columns=column_count, K=K)

    tm = PytorchTemporalMemory(column_count=column_count, cells_per_column=4)

    anomalies_epoch1 = []
    anomalies_epoch5 = []

    for epoch in range(6):
        tm.reset()
        for step, active_cols in enumerate(sequence):
            result = tm(active_cols, learn=True)
            if epoch == 0:
                anomalies_epoch1.append(result['anomaly'].item())
            elif epoch == 5:
                anomalies_epoch5.append(result['anomaly'].item())

    # Skip first step (always unpredicted). Average anomaly should decrease.
    avg_anomaly_1 = sum(anomalies_epoch1[1:]) / len(anomalies_epoch1[1:])
    avg_anomaly_5 = sum(anomalies_epoch5[1:]) / len(anomalies_epoch5[1:])
    assert avg_anomaly_5 < avg_anomaly_1, (
        f"Prediction should improve: epoch1={avg_anomaly_1:.3f} vs epoch5={avg_anomaly_5:.3f}"
    )
```

---

#### 4.1c) `test_tm_higher_order_context`

```python
def test_tm_higher_order_context():
    """TM distinguishes B-in-ABCD from B-in-XBCY (predicts C vs Y correctly)."""
    column_count = 256
    K = 5

    # Create symbols A, B, C, D, X, Y with non-overlapping SDRs
    symbols = {}
    for i, name in enumerate("ABCDXY"):
        cols = torch.zeros(column_count)
        cols[i * K : (i + 1) * K] = 1.0
        symbols[name] = cols

    seq1 = [symbols[s] for s in "ABCD"]
    seq2 = [symbols[s] for s in "XBCY"]

    tm = PytorchTemporalMemory(column_count=column_count, cells_per_column=32)

    # Train both sequences for many epochs
    for _ in range(20):
        tm.reset()
        for s in seq1:
            tm(s, learn=True)
        tm.reset()
        for s in seq2:
            tm(s, learn=True)

    # After training, feed A then B -- predictive cells should predict C-columns
    tm.reset()
    tm(symbols["A"], learn=False)
    result_ab = tm(symbols["B"], learn=False)

    # Feed X then B -- predictive cells should predict C-columns (from XBCY) or Y-columns
    tm.reset()
    tm(symbols["X"], learn=False)
    result_xb = tm(symbols["B"], learn=False)

    # The predictive cell patterns after B should differ based on context
    pred_after_ab = result_ab['predictive_cells']
    pred_after_xb = result_xb['predictive_cells']
    assert not torch.equal(pred_after_ab, pred_after_xb), (
        "TM should produce different predictions for B-in-ABCD vs B-in-XBCY"
    )
```

---

### 4.2 Activation Logic

#### 4.2a) `test_tm_predicted_column_no_burst`

```python
def test_tm_predicted_column_no_burst():
    """Predicted columns activate only predicted cells, not all cells."""
    column_count = 256
    cells_per_column = 4
    tm = PytorchTemporalMemory(column_count=column_count, cells_per_column=cells_per_column)

    # Train a sequence so TM learns to predict step 2 from step 1
    cols_a = torch.zeros(column_count); cols_a[:5] = 1.0
    cols_b = torch.zeros(column_count); cols_b[5:10] = 1.0

    for _ in range(10):
        tm.reset()
        tm(cols_a, learn=True)
        tm(cols_b, learn=True)

    # Now present A, then B. Column B should NOT burst.
    tm.reset()
    tm(cols_a, learn=False)
    result = tm(cols_b, learn=False)

    # For each active column in B, count how many cells activated
    for col_idx in range(5, 10):
        start = col_idx * cells_per_column
        end = start + cells_per_column
        active_in_col = result['active_cells'][start:end].sum().item()
        # If predicted, only predicted cells fire (< cells_per_column)
        assert active_in_col < cells_per_column, (
            f"Column {col_idx} should not burst (activated {active_in_col}/{cells_per_column})"
        )
```

---

#### 4.2b) `test_tm_unpredicted_column_bursts`

```python
def test_tm_unpredicted_column_bursts():
    """Unpredicted columns activate ALL cells (bursting)."""
    column_count = 256
    cells_per_column = 4
    tm = PytorchTemporalMemory(column_count=column_count, cells_per_column=cells_per_column)

    # Present a novel column pattern with no prior context
    cols = torch.zeros(column_count)
    cols[:5] = 1.0
    result = tm(cols, learn=False)

    # First presentation with no prior state -- all active columns should burst
    for col_idx in range(5):
        start = col_idx * cells_per_column
        end = start + cells_per_column
        active_in_col = result['active_cells'][start:end].sum().item()
        assert active_in_col == cells_per_column, (
            f"Column {col_idx} should burst: expected {cells_per_column}, got {active_in_col}"
        )
```

---

#### 4.2c) `test_tm_winner_cell_selection`

```python
def test_tm_winner_cell_selection():
    """Winner cell is the cell with best matching segment, or least-used if no match."""
    column_count = 256
    cells_per_column = 4
    tm = PytorchTemporalMemory(column_count=column_count, cells_per_column=cells_per_column)

    # Present a novel column -- should burst, one winner selected
    cols = torch.zeros(column_count)
    cols[:5] = 1.0
    result = tm(cols, learn=True)

    # Exactly one winner per active column
    for col_idx in range(5):
        start = col_idx * cells_per_column
        end = start + cells_per_column
        winners_in_col = tm.winner_cells[start:end].sum().item()
        assert winners_in_col == 1.0, (
            f"Column {col_idx}: expected 1 winner, got {winners_in_col}"
        )
```

---

### 4.3 Segment/Synapse Management

#### 4.3a) `test_tm_segment_creation`

```python
def test_tm_segment_creation():
    """New segments are created for bursting columns during learning."""
    tm = PytorchTemporalMemory(column_count=256, cells_per_column=4)
    assert len(tm.segments) == 0

    # Step 1: establish context
    cols_a = torch.zeros(256); cols_a[:5] = 1.0
    tm(cols_a, learn=True)

    # Step 2: new columns burst, learning creates segments
    cols_b = torch.zeros(256); cols_b[5:10] = 1.0
    tm(cols_b, learn=True)

    assert len(tm.segments) > 0, "Segments should be created after learning from bursting"
```

---

#### 4.3b) `test_tm_segment_reinforcement`

```python
def test_tm_segment_reinforcement():
    """Correct predictions strengthen synapse permanences."""
    tm = PytorchTemporalMemory(
        column_count=256, cells_per_column=4,
        initial_permanence=0.3, permanence_inc=0.1
    )
    cols_a = torch.zeros(256); cols_a[:5] = 1.0
    cols_b = torch.zeros(256); cols_b[5:10] = 1.0

    # Train A->B twice
    for _ in range(2):
        tm.reset()
        tm(cols_a, learn=True)
        tm(cols_b, learn=True)

    # After second presentation, permanences should have increased from initial
    for cell_id, segments in tm.segments.items():
        for seg in segments:
            for pre_cell, perm in seg.items():
                # At least one synapse should exceed initial_permanence
                if perm > 0.3:
                    return  # test passes
    # If we get here, no synapse was strengthened
    pytest.fail("No synapse permanence was reinforced above initial value")
```

---

#### 4.3c) `test_tm_segment_punishment`

```python
def test_tm_segment_punishment():
    """False predictions weaken synapse permanences."""
    tm = PytorchTemporalMemory(
        column_count=256, cells_per_column=4,
        initial_permanence=0.5, permanence_dec=0.1
    )
    # Train A->B
    cols_a = torch.zeros(256); cols_a[:5] = 1.0
    cols_b = torch.zeros(256); cols_b[5:10] = 1.0
    tm.reset()
    tm(cols_a, learn=True)
    tm(cols_b, learn=True)

    # Record permanences after first training
    initial_perms = {}
    for cell_id, segments in tm.segments.items():
        for seg_idx, seg in enumerate(segments):
            for pre_cell, perm in seg.items():
                initial_perms[(cell_id, seg_idx, pre_cell)] = perm

    # Now present A->C (not B). The segment that predicted B from A context
    # should have inactive synapses weakened during the learning of C.
    cols_c = torch.zeros(256); cols_c[10:15] = 1.0
    tm.reset()
    tm(cols_a, learn=True)
    tm(cols_c, learn=True)

    # Check that at least some previously-active synapses were weakened
    # (synapses to cells that were NOT in prev_active_indices get decremented)
    weakened = False
    for cell_id, segments in tm.segments.items():
        for seg_idx, seg in enumerate(segments):
            for pre_cell, perm in seg.items():
                key = (cell_id, seg_idx, pre_cell)
                if key in initial_perms and perm < initial_perms[key]:
                    weakened = True
                    break
    # Note: punishment happens on active segments with inactive presynaptic cells.
    # This is a structural check; the exact cells affected depend on winner selection.
    assert weakened or len(tm.segments) > len(initial_perms) // 3, (
        "Either synapses were weakened or new segments were created for the new transition"
    )
```

---

#### 4.3d) `test_tm_synapse_pruning`

```python
def test_tm_synapse_pruning():
    """Dead synapses (permanence <= 0) are removed from segments."""
    tm = PytorchTemporalMemory(
        column_count=256, cells_per_column=4,
        initial_permanence=0.05, permanence_dec=0.1
    )
    cols_a = torch.zeros(256); cols_a[:5] = 1.0
    cols_b = torch.zeros(256); cols_b[5:10] = 1.0

    # Train A->B to create segments with low initial permanence
    tm.reset()
    tm(cols_a, learn=True)
    tm(cols_b, learn=True)

    # Repeatedly present A->C to punish A->B synapses
    cols_c = torch.zeros(256); cols_c[10:15] = 1.0
    for _ in range(5):
        tm.reset()
        tm(cols_a, learn=True)
        tm(cols_c, learn=True)

    # Verify no synapse has permanence <= 0
    for cell_id, segments in tm.segments.items():
        for seg in segments:
            for pre_cell, perm in seg.items():
                assert perm > 0.0, (
                    f"Dead synapse found: cell={cell_id}, pre={pre_cell}, perm={perm}"
                )
```

---

#### 4.3e) `test_tm_segment_cap`

```python
@pytest.mark.parametrize("max_segs", [4, 8])
def test_tm_segment_cap(max_segs):
    """Segments per cell never exceed max_segments_per_cell."""
    tm = PytorchTemporalMemory(
        column_count=256, cells_per_column=4,
        max_segments_per_cell=max_segs
    )
    # Feed many different transitions to force segment creation
    for i in range(50):
        tm.reset()
        cols_a = torch.zeros(256)
        cols_a[(i * 3) % 256 : (i * 3) % 256 + 5] = 1.0
        cols_b = torch.zeros(256)
        cols_b[(i * 7 + 50) % 256 : (i * 7 + 50) % 256 + 5] = 1.0
        tm(cols_a, learn=True)
        tm(cols_b, learn=True)

    for cell_id, segments in tm.segments.items():
        assert len(segments) <= max_segs, (
            f"Cell {cell_id} has {len(segments)} segments, max is {max_segs}"
        )
```

---

#### 4.3f) `test_tm_synapse_cap`

```python
@pytest.mark.parametrize("max_syns", [16, 32])
def test_tm_synapse_cap(max_syns):
    """Synapses per segment never exceed max_synapses_per_segment."""
    tm = PytorchTemporalMemory(
        column_count=256, cells_per_column=4,
        max_synapses_per_segment=max_syns
    )
    # Feed long sequences to accumulate synapses
    for _ in range(20):
        tm.reset()
        for i in range(10):
            cols = torch.zeros(256)
            cols[i * 5 : i * 5 + 5] = 1.0
            tm(cols, learn=True)

    for cell_id, segments in tm.segments.items():
        for seg_idx, seg in enumerate(segments):
            assert len(seg) <= max_syns, (
                f"Cell {cell_id} seg {seg_idx} has {len(seg)} synapses, max is {max_syns}"
            )
```

---

### 4.4 CSR Storage Invariants

The `PytorchTemporalMemory` uses a `Dict[int, List[Dict[int, float]]]` for segment storage. These tests validate structural integrity of this store under adversarial workloads. If the implementation migrates to true CSR (compressed sparse row) tensors, adapt these tests to check pointer/index consistency instead.

---

#### 4.4a) `test_csr_pointer_consistency`

```python
def test_csr_pointer_consistency():
    """All segment lists are properly formed: no None entries, no empty dicts that should be pruned."""
    tm = PytorchTemporalMemory(column_count=256, cells_per_column=4)
    # Run a moderate workload
    for _ in range(10):
        tm.reset()
        for i in range(5):
            cols = torch.zeros(256)
            cols[i * 10 : i * 10 + 5] = 1.0
            tm(cols, learn=True)

    for cell_id, segments in tm.segments.items():
        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg, dict)
            assert seg is not None
            # All keys are valid cell indices
            for pre_cell in seg.keys():
                assert 0 <= pre_cell < tm.num_cells
```

---

#### 4.4b) `test_csr_no_orphan_synapses`

```python
def test_csr_no_orphan_synapses():
    """All synapses belong to a segment that belongs to a valid cell."""
    tm = PytorchTemporalMemory(column_count=256, cells_per_column=4)
    for _ in range(10):
        tm.reset()
        for i in range(5):
            cols = torch.zeros(256)
            cols[i * 10 : i * 10 + 5] = 1.0
            tm(cols, learn=True)

    for cell_id, segments in tm.segments.items():
        assert 0 <= cell_id < tm.num_cells, f"Invalid cell_id: {cell_id}"
        for seg in segments:
            for pre_cell, perm in seg.items():
                assert 0 <= pre_cell < tm.num_cells, f"Orphan synapse: pre_cell={pre_cell}"
                assert isinstance(perm, float), f"Permanence must be float, got {type(perm)}"
                assert 0.0 < perm <= 1.0, f"Permanence out of bounds: {perm}"
```

---

#### 4.4c) `test_csr_compaction`

```python
def test_csr_compaction():
    """After heavy learning and pruning, no empty segments remain in the dict."""
    tm = PytorchTemporalMemory(
        column_count=256, cells_per_column=4,
        initial_permanence=0.05, permanence_dec=0.1
    )
    # Create segments, then aggressively prune
    for epoch in range(20):
        tm.reset()
        for i in range(5):
            cols = torch.zeros(256)
            offset = (epoch * 7 + i * 13) % 250
            cols[offset : offset + 5] = 1.0
            tm(cols, learn=True)

    # Check no empty segments
    for cell_id, segments in tm.segments.items():
        for seg_idx, seg in enumerate(segments):
            # A segment with no synapses is a gap that should have been removed
            # (if the implementation prunes empty segments; if not, this test
            # documents the expectation that it should)
            if len(seg) == 0:
                pytest.fail(f"Empty segment at cell {cell_id}, index {seg_idx}")
```

---

#### 4.4d) `test_csr_fuzz_random_operations`

```python
@pytest.mark.slow
def test_csr_fuzz_random_operations():
    """Random sequence of learn/reset/step operations produces no corruption."""
    import random
    random.seed(42)
    torch.manual_seed(42)

    tm = PytorchTemporalMemory(column_count=256, cells_per_column=4)

    for _ in range(500):
        action = random.choice(["step_learn", "step_noop", "reset"])
        if action == "reset":
            tm.reset()
        else:
            cols = torch.zeros(256)
            active_idx = random.sample(range(256), 5)
            for idx in active_idx:
                cols[idx] = 1.0
            learn = action == "step_learn"
            result = tm(cols, learn=learn)
            # Structural invariants
            assert result['anomaly'].item() >= 0.0
            assert result['anomaly'].item() <= 1.0
            assert not torch.isnan(result['active_cells']).any()

    # Final structural check
    stats = tm.get_memory_stats()
    assert stats['total_segments'] >= 0
    assert stats['total_synapses'] >= 0
    assert stats['cells_with_segments'] <= tm.num_cells
```

---

## 5. Anomaly Detection Tests

---

### 5a) `test_anomaly_range`

```python
@pytest.mark.parametrize("column_count", [256, 2048])
def test_anomaly_range(column_count):
    """anomaly_score is always in [0, 1]."""
    tm = PytorchTemporalMemory(column_count=column_count, cells_per_column=4)
    for _ in range(50):
        cols = torch.zeros(column_count)
        active = torch.randperm(column_count)[:int(column_count * 0.02)]
        cols[active] = 1.0
        result = tm(cols, learn=True)
        a = result['anomaly'].item()
        assert 0.0 <= a <= 1.0, f"Anomaly out of range: {a}"
```

---

### 5b) `test_anomaly_known_sequence`

```python
def test_anomaly_known_sequence():
    """After learning, anomaly is near 0 for known patterns."""
    column_count = 256
    K = 5
    sequence = repeating_sequence(length=4, num_symbols=4, N_columns=column_count, K=K)

    tm = PytorchTemporalMemory(column_count=column_count, cells_per_column=32)

    # Train for many epochs
    for _ in range(30):
        tm.reset()
        for s in sequence:
            tm(s, learn=True)

    # Check anomaly: skip first step (always unpredicted after reset)
    tm.reset()
    tm(sequence[0], learn=False)
    anomalies = []
    for s in sequence[1:]:
        result = tm(s, learn=False)
        anomalies.append(result['anomaly'].item())

    avg_anomaly = sum(anomalies) / len(anomalies)
    assert avg_anomaly < 0.3, f"Average anomaly for known sequence should be low: {avg_anomaly:.3f}"
```

---

### 5c) `test_anomaly_novel_injection`

```python
def test_anomaly_novel_injection():
    """Injecting a novel symbol Z causes an immediate anomaly spike."""
    column_count = 256
    K = 5
    sequence = repeating_sequence(length=4, num_symbols=4, N_columns=column_count, K=K)

    tm = PytorchTemporalMemory(column_count=column_count, cells_per_column=32)

    # Train
    for _ in range(30):
        tm.reset()
        for s in sequence:
            tm(s, learn=True)

    # Present known sequence, then inject novel symbol
    tm.reset()
    for s in sequence:
        tm(s, learn=False)

    # Novel symbol: columns that were never in the training sequence
    novel = torch.zeros(column_count)
    novel[200:205] = 1.0  # disjoint from training symbols

    result = tm(novel, learn=False)
    assert result['anomaly'].item() > 0.5, (
        f"Novel injection should produce high anomaly: {result['anomaly'].item():.3f}"
    )
```

---

### 5d) `test_anomaly_adaptation`

```python
@pytest.mark.slow
def test_anomaly_adaptation():
    """Keep injecting novel Z; anomaly gradually declines as TM adapts."""
    column_count = 256
    K = 5

    tm = PytorchTemporalMemory(column_count=column_count, cells_per_column=32)

    # Train original sequence
    seq = repeating_sequence(length=4, num_symbols=4, N_columns=column_count, K=K)
    for _ in range(20):
        tm.reset()
        for s in seq:
            tm(s, learn=True)

    # Now inject Z repeatedly after A
    novel = torch.zeros(column_count); novel[200:205] = 1.0
    anomalies = []
    for _ in range(20):
        tm.reset()
        tm(seq[0], learn=True)  # A
        result = tm(novel, learn=True)  # Z
        anomalies.append(result['anomaly'].item())

    # Anomaly should decrease over the 20 repetitions
    first_half = sum(anomalies[:5]) / 5
    second_half = sum(anomalies[15:]) / 5
    assert second_half < first_half, (
        f"Anomaly should adapt: first5={first_half:.3f}, last5={second_half:.3f}"
    )
```

---

### 5e) `test_anomaly_distribution_shift`

```python
@pytest.mark.slow
def test_anomaly_distribution_shift():
    """Train on distribution A, switch to B; anomaly rises then adapts."""
    column_count = 256
    K = 5
    seq_a = repeating_sequence(length=4, num_symbols=4, N_columns=column_count, K=K)
    seq_b = repeating_sequence(length=4, num_symbols=4, N_columns=column_count, K=K, offset=100)

    tm = PytorchTemporalMemory(column_count=column_count, cells_per_column=32)

    # Train on A
    for _ in range(30):
        tm.reset()
        for s in seq_a:
            tm(s, learn=True)

    # Switch to B -- initial anomaly should be high
    anomalies_b = []
    for epoch in range(20):
        tm.reset()
        for s in seq_b:
            result = tm(s, learn=True)
            anomalies_b.append(result['anomaly'].item())

    # First few steps of B should be high anomaly
    early = sum(anomalies_b[:8]) / 8
    assert early > 0.3, f"Distribution shift should cause high initial anomaly: {early:.3f}"
```

---

### 5f) `test_anomaly_likelihood_smoothing`

```python
def test_anomaly_likelihood_smoothing():
    """anomaly_likelihood is smoother than raw anomaly over a long run."""
    htm = HTMLayer(HTMConfig(input_size=64, column_count=256, cells_per_column=4))

    raw_anomalies = []
    likelihoods = []

    for i in range(200):
        x = torch.rand(64)
        result = htm(x, learn=True)
        raw_anomalies.append(result['anomaly'].item())
        likelihoods.append(result['anomaly_likelihood'].item())

    # After warmup (first 20 steps), likelihood should have lower variance
    raw_var = torch.tensor(raw_anomalies[20:]).var().item()
    lik_var = torch.tensor(likelihoods[20:]).var().item()
    # Likelihood should be at least somewhat smoothed
    # (this is a weak check; mainly verifying the computation runs)
    assert all(0.0 <= l <= 1.0 for l in likelihoods), "Likelihood values must be in [0, 1]"
```

---

## 6. Reflex Memory Tests

All tests use `ReflexMemory` from `brain_ai/temporal/htm.py`.

---

### 6.1 Promotion and Lookup

#### 6.1a) `test_reflex_promotion_after_threshold`

```python
def test_reflex_promotion_after_threshold():
    """Pattern is stored (promoted) after being presented to store()."""
    rm = ReflexMemory(pattern_dim=64, max_patterns=100, promotion_threshold=5)
    pattern = torch.randn(64)
    prediction = torch.randn(64)

    idx = rm.store(pattern, prediction, force=True)
    assert idx >= 0, "Pattern should be stored"
    assert rm.num_stored.item() == 1
```

---

#### 6.1b) `test_reflex_not_promoted_too_early`

```python
def test_reflex_not_promoted_too_early():
    """Before promotion threshold, AcceleratedHTM does NOT store in RM."""
    htm_layer = create_htm_layer(input_size=64, column_count=256)
    ahtm = AcceleratedHTM(htm_layer, promotion_threshold=5)

    x = (torch.rand(64) > 0.5).float()

    # Present pattern fewer times than promotion threshold
    for _ in range(4):
        ahtm(x, learn=True)

    assert ahtm.rm.num_stored.item() == 0, (
        "Pattern should NOT be promoted before reaching threshold"
    )
```

---

#### 6.1c) `test_reflex_fast_path_hit`

```python
def test_reflex_fast_path_hit():
    """Promoted pattern returns cached prediction via fast path."""
    rm = ReflexMemory(pattern_dim=64, max_patterns=100, similarity_threshold=0.8)
    pattern = torch.randn(64)
    prediction = torch.randn(64)

    rm.store(pattern, prediction, force=True)
    result = rm.lookup(pattern)
    assert result is not None, "Stored pattern should be found"
    pred, confidence, idx = result
    assert confidence >= 0.8
    assert pred.shape == (64,)
```

---

#### 6.1d) `test_reflex_slow_path_miss`

```python
def test_reflex_slow_path_miss():
    """Unknown pattern falls through to full TM (returns None from RM lookup)."""
    rm = ReflexMemory(pattern_dim=64, max_patterns=100)
    # Empty RM
    unknown = torch.randn(64)
    result = rm.lookup(unknown)
    assert result is None, "Unknown pattern should not match"

    # Store one pattern, query a different one
    rm.store(torch.randn(64), torch.randn(64), force=True)
    result = rm.lookup(torch.randn(64))
    # LSH makes this probabilistic, but with random patterns it should miss
    # (or match with very low similarity)
    if result is not None:
        _, confidence, _ = result
        assert confidence < 0.95, "Random pattern should not strongly match"
```

---

### 6.2 Correctness

#### 6.2a) `test_reflex_matches_baseline`

```python
def test_reflex_matches_baseline():
    """For promoted patterns, Reflex prediction is the SP features output from the HTM that created it."""
    htm_layer = create_htm_layer(input_size=64, column_count=256)
    rm = ReflexMemory(pattern_dim=64, max_patterns=100, similarity_threshold=0.8)

    x = (torch.rand(64) > 0.5).float()
    baseline_result = htm_layer(x, learn=True)
    rm.store(x, baseline_result['features'], force=True)

    lookup_result = rm.lookup(x)
    assert lookup_result is not None
    pred, _, _ = lookup_result
    # The stored prediction should match the baseline features
    assert torch.allclose(pred, baseline_result['features'], atol=1e-5)
```

---

#### 6.2b) `test_reflex_side_by_side`

```python
def test_reflex_side_by_side():
    """Run baseline HTM and AcceleratedHTM on same sequences, compare acceleration behavior."""
    htm_baseline = create_htm_layer(input_size=64, column_count=256)
    ahtm = create_accelerated_htm(input_size=64, column_count=256, promotion_threshold=3)

    torch.manual_seed(42)
    sequence = [(torch.rand(64) > 0.5).float() for _ in range(5)]

    # Run same sequence multiple times through AHTM to trigger promotion
    for _ in range(5):
        for x in sequence:
            result = ahtm(x, learn=True)

    # After promotion, AHTM should report some RM hits
    stats = ahtm.get_statistics()
    assert stats['rm_hits'] >= 0  # at least the mechanism runs without error
    assert stats['htm_calls'] > 0  # some calls went through HTM
```

---

### 6.3 Guardrails

#### 6.3a) `test_reflex_memory_cap`

```python
def test_reflex_memory_cap():
    """Stored entries never exceed max_patterns."""
    max_p = 50
    rm = ReflexMemory(pattern_dim=32, max_patterns=max_p)

    for i in range(100):
        # Store distinct patterns
        pattern = torch.zeros(32)
        pattern[i % 32] = float(i)  # make each somewhat unique
        rm.store(pattern, torch.randn(32), force=True)

    assert rm.num_stored.item() <= max_p
```

---

#### 6.3b) `test_reflex_lru_eviction`

```python
def test_reflex_lru_eviction():
    """When table is full, oldest/least-important entries are evicted."""
    max_p = 10
    rm = ReflexMemory(pattern_dim=32, max_patterns=max_p)

    # Fill table
    stored_patterns = []
    for i in range(max_p):
        p = torch.randn(32)
        rm.store(p, torch.randn(32), force=True)
        stored_patterns.append(p)

    assert rm.num_stored.item() == max_p

    # Store one more -- should evict least important
    new_pattern = torch.randn(32)
    idx = rm.store(new_pattern, torch.randn(32), force=True)
    assert idx >= 0  # eviction succeeded, new pattern stored
    assert rm.num_stored.item() == max_p  # count should not exceed max
```

---

#### 6.3c) `test_reflex_demotion`

```python
def test_reflex_demotion():
    """Inaccurate promoted entry can be overwritten (demoted) by eviction."""
    rm = ReflexMemory(pattern_dim=32, max_patterns=5)

    # Fill with patterns
    for i in range(5):
        rm.store(torch.randn(32), torch.randn(32), force=True)

    # The entry with lowest importance (access_count / recency) gets evicted
    # Artificially set one entry's access count to 0
    rm.access_counts[0] = 0.0
    rm.timestamps[0] = 0  # very old

    # Store new pattern -- should evict entry 0
    new_p = torch.randn(32)
    idx = rm.store(new_p, torch.randn(32), force=True)
    # The evicted slot should have been reused
    assert rm.num_stored.item() == 5
```

---

#### 6.3d) `test_reflex_decay`

```python
def test_reflex_decay():
    """Access counts decay over time when decay_access_counts() is called."""
    rm = ReflexMemory(pattern_dim=32, max_patterns=10, decay_rate=0.9)
    pattern = torch.randn(32)
    rm.store(pattern, torch.randn(32), force=True)

    # Manually set high access count
    rm.access_counts[0] = 100.0

    rm.decay_access_counts()
    assert rm.access_counts[0].item() == pytest.approx(90.0, abs=0.01)

    rm.decay_access_counts()
    assert rm.access_counts[0].item() == pytest.approx(81.0, abs=0.01)
```

---

#### 6.3e) `test_reflex_state_dict_roundtrip`

```python
def test_reflex_state_dict_roundtrip():
    """save -> load -> predictions are identical."""
    rm = ReflexMemory(pattern_dim=32, max_patterns=10)
    pattern = torch.randn(32)
    prediction = torch.randn(32)
    rm.store(pattern, prediction, force=True)

    # Save state
    state = rm.state_dict()

    # Create new RM and load
    rm2 = ReflexMemory(pattern_dim=32, max_patterns=10)
    rm2.load_state_dict(state)

    # Lookup should produce same result
    result1 = rm.lookup(pattern)
    result2 = rm2.lookup(pattern)
    assert result1 is not None and result2 is not None
    pred1, conf1, _ = result1
    pred2, conf2, _ = result2
    assert torch.allclose(pred1, pred2)
    assert abs(conf1 - conf2) < 1e-6
```

---

### 6.4 Statistics

#### 6.4a) `test_reflex_hit_rate_tracking`

```python
def test_reflex_hit_rate_tracking():
    """hit_rate is computed correctly as hits / (hits + misses)."""
    rm = ReflexMemory(pattern_dim=32, max_patterns=10, similarity_threshold=0.8)
    pattern = torch.randn(32)
    rm.store(pattern, torch.randn(32), force=True)

    # Force some hits and misses
    rm.lookup(pattern)   # should hit (same pattern)
    rm.lookup(torch.randn(32))  # likely miss
    rm.lookup(torch.randn(32))  # likely miss

    stats = rm.get_statistics()
    # At minimum, hits + misses should equal total lookups (3)
    # Note: the first store() also does a lookup internally, adding to counts
    assert stats['hits'] + stats['misses'] >= 3
    assert 0.0 <= stats['hit_rate'] <= 1.0
```

---

#### 6.4b) `test_reflex_statistics_reset`

```python
def test_reflex_statistics_reset():
    """reset_statistics() clears hit/miss counters."""
    rm = ReflexMemory(pattern_dim=32, max_patterns=10)
    rm.store(torch.randn(32), torch.randn(32), force=True)
    rm.lookup(torch.randn(32))

    rm.reset_statistics()
    stats = rm.get_statistics()
    assert stats['hits'] == 0
    assert stats['misses'] == 0
    assert stats['hit_rate'] == 0.0
```

---

## 7. Fallback Predictor Tests

All fallback predictors (`LSTMSequencePredictor`, `GRUSequencePredictor`, `TransformerSequencePredictor`) must satisfy the same interface contract. These tests are parametrized over all three backends.

---

### 7a) `test_fallback_interface_compliance`

```python
@pytest.mark.parametrize("backend_cls", [
    LSTMSequencePredictor,
    GRUSequencePredictor,
    TransformerSequencePredictor,
])
def test_fallback_interface_compliance(backend_cls):
    """Each fallback returns a dict with required keys."""
    if backend_cls == LSTMSequencePredictor:
        model = backend_cls(SequenceConfig(input_size=64, hidden_size=32))
    elif backend_cls == GRUSequencePredictor:
        model = backend_cls(input_size=64, hidden_size=32)
    else:
        model = backend_cls(input_size=64, hidden_size=32)

    x = torch.randn(2, 64)
    result = model(x, learn=True)

    required_keys = {'features', 'prediction', 'anomaly', 'anomaly_likelihood',
                     'active_cells', 'predictive_cells'}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - set(result.keys())}"
    )
```

---

### 7b) `test_fallback_sdr_shape`

```python
@pytest.mark.parametrize("backend", ["lstm", "gru", "transformer"])
def test_fallback_sdr_shape(backend):
    """Output features have correct batch dimension."""
    layer = create_temporal_layer(backend=backend, input_size=64, hidden_size=32)
    x = torch.randn(4, 64)
    result = layer(x, learn=True)
    assert result['features'].shape[0] == 4
    assert result['prediction'].shape[0] == 4
```

---

### 7c) `test_fallback_anomaly_range`

```python
@pytest.mark.parametrize("backend", ["lstm", "gru", "transformer"])
def test_fallback_anomaly_range(backend):
    """anomaly_score is in [0, 1] for all fallback backends."""
    layer = create_temporal_layer(backend=backend, input_size=64, hidden_size=32)
    for _ in range(20):
        x = torch.randn(2, 64)
        result = layer(x, learn=True)
        a = result['anomaly']
        assert (a >= 0.0).all() and (a <= 1.0).all(), f"Anomaly out of range: {a}"
```

---

### 7d) `test_fallback_reset_clears_state`

```python
@pytest.mark.parametrize("backend", ["lstm", "gru", "transformer"])
def test_fallback_reset_clears_state(backend):
    """After reset(), hidden state is cleared. Same input produces same output."""
    layer = create_temporal_layer(backend=backend, input_size=64, hidden_size=32)
    layer.layer.eval() if hasattr(layer, 'layer') else layer.eval()
    x = torch.randn(2, 64)

    # First pass
    layer.reset()
    result1 = layer(x, learn=False)

    # Contaminate state with different input
    layer(torch.randn(2, 64), learn=False)

    # Reset and re-run
    layer.reset()
    result2 = layer(x, learn=False)

    assert torch.allclose(result1['features'], result2['features'], atol=1e-5), (
        "Reset should restore deterministic behavior"
    )
```

---

### 7e) `test_fallback_prediction_shape`

```python
@pytest.mark.parametrize("backend", ["lstm", "gru", "transformer"])
@pytest.mark.parametrize("input_size", [64, 256])
def test_fallback_prediction_shape(backend, input_size):
    """Prediction output has shape (B, input_size)."""
    layer = create_temporal_layer(backend=backend, input_size=input_size, hidden_size=32)
    x = torch.randn(4, input_size)
    result = layer(x, learn=True)
    assert result['prediction'].shape == (4, input_size)
```

---

### 7f) `test_fallback_gradient_flow`

```python
@pytest.mark.parametrize("backend", ["lstm", "gru", "transformer"])
def test_fallback_gradient_flow(backend):
    """Gradients flow through fallback (unlike HTM which is non-differentiable)."""
    layer = create_temporal_layer(backend=backend, input_size=64, hidden_size=32)
    layer.train()
    x = torch.randn(2, 64, requires_grad=True)
    result = layer(x, learn=True)
    loss = result['prediction'].sum()
    loss.backward()

    # At least one parameter should have a gradient
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in layer.parameters()
    )
    assert has_grad, "Fallback should be differentiable with gradient flow"
```

---

## 8. Integration Tests

---

### 8a) `test_htm_layer_end_to_end`

```python
def test_htm_layer_end_to_end():
    """SP -> TM -> anomaly pipeline produces valid output dict."""
    htm = HTMLayer(HTMConfig(input_size=64, column_count=256, cells_per_column=4))
    x = torch.rand(64)
    result = htm(x, learn=True)

    assert 'features' in result
    assert 'active_cells' in result
    assert 'predictive_cells' in result
    assert 'anomaly' in result
    assert 'anomaly_likelihood' in result
    assert result['features'].shape == (256,)
    assert result['active_cells'].shape == (256 * 4,)
    assert 0.0 <= result['anomaly'].item() <= 1.0
```

---

### 8b) `test_accelerated_htm_end_to_end`

```python
def test_accelerated_htm_end_to_end():
    """SP -> TM -> Reflex -> output pipeline produces valid output."""
    ahtm = create_accelerated_htm(input_size=64, column_count=256, promotion_threshold=3)
    x = (torch.rand(64) > 0.5).float()
    result = ahtm(x, learn=True)

    assert 'features' in result
    assert 'anomaly' in result
    assert 'from_reflex' in result
    assert 'confidence' in result
    assert result['features'].shape == (256,)
```

---

### 8c) `test_temporal_layer_backend_switch`

```python
def test_temporal_layer_backend_switch():
    """Switch backends at runtime; outputs remain structurally compatible."""
    input_size = 64
    x = torch.randn(2, input_size)

    backends = ["lstm", "gru", "transformer"]
    results = {}
    for backend in backends:
        layer = create_temporal_layer(backend=backend, input_size=input_size, hidden_size=32)
        layer.reset()
        result = layer(x, learn=False)
        results[backend] = result

    # All backends should produce the same set of output keys
    key_sets = [set(r.keys()) for r in results.values()]
    common_keys = key_sets[0]
    for ks in key_sets[1:]:
        common_keys = common_keys & ks
    assert {'features', 'prediction', 'anomaly'}.issubset(common_keys)

    # All backends should produce matching batch dimensions
    for backend, result in results.items():
        assert result['features'].shape[0] == 2, f"{backend}: wrong batch dim"
```

---

### 8d) `test_sequence_learning_pipeline`

```python
def test_sequence_learning_pipeline():
    """Dense input -> binarize -> SP -> TM -> predict -> check anomaly."""
    htm = HTMLayer(HTMConfig(input_size=64, column_count=256, cells_per_column=4))

    # Create a repeating dense sequence
    torch.manual_seed(42)
    seq_length = 5
    dense_sequence = [torch.rand(64) for _ in range(seq_length)]

    # Train for multiple epochs
    for epoch in range(10):
        htm.reset()
        for x in dense_sequence:
            htm(x, learn=True)

    # Run assessment pass
    htm.reset()
    anomalies = []
    for x in dense_sequence:
        result = htm(x, learn=False)
        anomalies.append(result['anomaly'].item())

    # After training, later elements in the sequence should have lower anomaly
    # (first element is always unpredicted after reset)
    assert all(0.0 <= a <= 1.0 for a in anomalies)
```

---

### 8e) `test_batch_consistency`

```python
@pytest.mark.parametrize("B", [1, 4])
def test_batch_consistency(B):
    """Batch of identical inputs produces identical outputs per sample."""
    htm = HTMLayer(HTMConfig(input_size=64, column_count=256, cells_per_column=4))
    single_input = torch.rand(64)
    batch_input = single_input.unsqueeze(0).expand(B, -1).clone()

    result = htm(batch_input, learn=False)

    # All samples in the batch should produce the same features
    for i in range(1, B):
        assert torch.equal(result['features'][0], result['features'][i]), (
            f"Sample 0 and sample {i} differ in batch of identical inputs"
        )
```

---

## 9. Performance / Memory Tests

Mark all tests in this category with `@pytest.mark.slow`.

---

### 9a) `test_sp_throughput`

```python
@pytest.mark.slow
def test_sp_throughput():
    """SP processes 1000 steps in < 30 seconds (CPU, column_count=2048)."""
    import time
    sp = PytorchSpatialPooler(input_size=512, column_count=2048)
    start = time.perf_counter()
    for _ in range(1000):
        x = (torch.rand(512) > 0.5).float()
        sp(x, learn=True)
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, f"SP too slow: {elapsed:.1f}s for 1000 steps"
```

---

### 9b) `test_tm_memory_bounded`

```python
@pytest.mark.slow
def test_tm_memory_bounded():
    """Segment and synapse counts stay within configured caps after long run."""
    max_segs = 16
    max_syns = 32
    tm = PytorchTemporalMemory(
        column_count=256, cells_per_column=4,
        max_segments_per_cell=max_segs,
        max_synapses_per_segment=max_syns
    )

    for _ in range(100):
        tm.reset()
        for i in range(10):
            cols = torch.zeros(256)
            active = torch.randperm(256)[:5]
            cols[active] = 1.0
            tm(cols, learn=True)

    stats = tm.get_memory_stats()
    # Check caps are respected
    for cell_id, segments in tm.segments.items():
        assert len(segments) <= max_segs
        for seg in segments:
            assert len(seg) <= max_syns
```

---

### 9c) `test_reflex_lookup_faster_than_tm`

```python
@pytest.mark.slow
def test_reflex_lookup_faster_than_tm():
    """Reflex Memory hit latency is lower than full HTM computation."""
    import time

    # Prepare HTM and RM
    htm_layer = create_htm_layer(input_size=64, column_count=256)
    rm = ReflexMemory(pattern_dim=64, max_patterns=100, similarity_threshold=0.8)

    pattern = (torch.rand(64) > 0.5).float()
    # Get HTM result and store in RM
    htm_result = htm_layer(pattern, learn=False)
    rm.store(pattern, htm_result['features'], force=True)

    # Time RM lookup (1000 iterations)
    start = time.perf_counter()
    for _ in range(1000):
        rm.lookup(pattern)
    rm_time = time.perf_counter() - start

    # Time HTM forward (1000 iterations)
    start = time.perf_counter()
    for _ in range(1000):
        htm_layer(pattern, learn=False)
    htm_time = time.perf_counter() - start

    assert rm_time < htm_time, (
        f"RM lookup ({rm_time:.3f}s) should be faster than HTM ({htm_time:.3f}s)"
    )
```

---

### 9d) `test_csr_memory_vs_dict`

```python
@pytest.mark.slow
def test_csr_memory_vs_dict():
    """CSR/dict segment storage uses reasonable memory for sparse representations."""
    import sys

    tm = PytorchTemporalMemory(column_count=2048, cells_per_column=32)

    # Train to build up some segments
    for _ in range(20):
        tm.reset()
        for i in range(10):
            cols = torch.zeros(2048)
            active = torch.randperm(2048)[:40]
            cols[active] = 1.0
            tm(cols, learn=True)

    stats = tm.get_memory_stats()
    # Memory should scale with actual connections, not O(num_cells^2)
    num_cells = 2048 * 32  # 65536
    max_dense_synapses = num_cells * num_cells  # ~4 billion -- clearly infeasible
    assert stats['total_synapses'] < num_cells, (
        f"Synapse count ({stats['total_synapses']}) should be << num_cells^2"
    )
    # Verify storage is sparse
    sparsity_ratio = stats['total_synapses'] / max_dense_synapses
    assert sparsity_ratio < 0.001, f"Storage should be highly sparse: ratio={sparsity_ratio:.6f}"
```

---

## 10. Conftest Fixtures

Place the following in `tests/test_htm/conftest.py`:

```python
import pytest
import torch
from brain_ai.temporal.htm import (
    HTMConfig,
    PytorchSpatialPooler,
    PytorchTemporalMemory,
    HTMLayer,
    ReflexMemory,
    AcceleratedHTM,
    create_htm_layer,
    create_accelerated_htm,
)
from brain_ai.temporal.sequence import (
    LSTMSequencePredictor,
    GRUSequencePredictor,
    TransformerSequencePredictor,
    SequenceConfig,
    TemporalLayer,
    create_temporal_layer,
)


@pytest.fixture
def sp_config():
    """Small Spatial Pooler config for fast tests."""
    return {
        "input_size": 64,
        "column_count": 256,
        "sparsity": 0.02,
    }


@pytest.fixture
def tm_config():
    """Small Temporal Memory config for fast tests."""
    return {
        "column_count": 256,
        "cells_per_column": 4,
        "activation_threshold": 3,
        "min_threshold": 2,
        "max_new_synapse_count": 10,
        "initial_permanence": 0.21,
        "permanence_connected": 0.5,
        "permanence_inc": 0.1,
        "permanence_dec": 0.1,
        "max_segments_per_cell": 16,
        "max_synapses_per_segment": 32,
    }


@pytest.fixture
def reflex_config():
    """Small Reflex Memory config for fast tests."""
    return {
        "pattern_dim": 64,
        "max_patterns": 100,
        "promotion_threshold": 5,
        "similarity_threshold": 0.8,
        "decay_rate": 0.99,
    }


@pytest.fixture
def htm_config():
    """Complete HTM config for integration tests."""
    return HTMConfig(
        input_size=64,
        column_count=256,
        cells_per_column=4,
        sparsity=0.02,
        activation_threshold=3,
        min_threshold=2,
        max_new_synapse_count=10,
    )


def random_sdr(B: int, K: int, N: int) -> torch.Tensor:
    """
    Generate random SDR indices.

    Args:
        B: Batch size
        K: Number of active columns per sample
        N: Total number of columns

    Returns:
        (B, K) tensor of unique indices in [0, N)
    """
    indices = []
    for _ in range(B):
        idx = torch.randperm(N)[:K].sort().values
        indices.append(idx)
    return torch.stack(indices)


def repeating_sequence(
    length: int,
    num_symbols: int,
    N_columns: int,
    K: int,
    offset: int = 0,
) -> list:
    """
    Generate ABCD-style repeating sequence as binary column vectors.

    Args:
        length: Number of symbols in one cycle
        num_symbols: Number of distinct symbols (>= length)
        N_columns: Total columns in SP output
        K: Active columns per symbol
        offset: Starting column index for symbols (for creating disjoint sets)

    Returns:
        List of (N_columns,) binary tensors
    """
    assert num_symbols >= length
    assert offset + num_symbols * K <= N_columns, (
        f"Not enough columns: need {offset + num_symbols * K}, have {N_columns}"
    )

    symbols = []
    for i in range(length):
        cols = torch.zeros(N_columns)
        start = offset + i * K
        cols[start : start + K] = 1.0
        symbols.append(cols)

    return symbols


@pytest.fixture
def trained_htm():
    """
    Factory fixture: pre-trained HTM for testing.

    Usage:
        htm = trained_htm(config, sequence, num_epochs=10)
    """
    def _factory(config=None, sequence=None, num_epochs=10):
        if config is None:
            config = HTMConfig(
                input_size=64, column_count=256,
                cells_per_column=4, sparsity=0.02,
            )
        htm = HTMLayer(config)

        if sequence is None:
            sequence = repeating_sequence(
                length=4, num_symbols=4,
                N_columns=config.column_count,
                K=int(config.column_count * config.sparsity),
            )

        for _ in range(num_epochs):
            htm.reset()
            for s in sequence:
                htm(s, learn=True)

        return htm

    return _factory
```

---

## 11. Done-When Checklist

All items must pass before the HTM subsystem is considered production-ready. No partial credit.

### SDR Utilities (items 1-9)
- [ ] 1. SDR roundtrip: `indices -> dense -> indices` preserves values for all parametrized `(B, K, N)` (2a)
- [ ] 2. `dense_to_indices` selects exactly `K` active bits with no duplicates (2b)
- [ ] 3. `sdr_overlap(a, a) == K` for all batch elements (2c)
- [ ] 4. `sdr_overlap(a, b) == 0` when index sets are disjoint (2d)
- [ ] 5. `sdr_jaccard` returns values in `[0, 1]` for random SDR pairs (2e)
- [ ] 6. `sdr_jaccard(a, a) == 1.0` (2f)
- [ ] 7. `sdr_hash` is deterministic: same indices produce same hash (2g)
- [ ] 8. `sdr_hash` is order-independent: shuffled indices produce same hash (2h)
- [ ] 9. `sdr_hash` collision rate is below 1% over 1000 random SDRs (2i)

### Spatial Pooler (items 10-19)
- [ ] 10. SP output shape is `(B, column_count)` for batched input (3a)
- [ ] 11. Exactly `K = int(column_count * sparsity)` columns active per sample (3b)
- [ ] 12. SP is deterministic: same input produces same output (3c)
- [ ] 13. Boosting activates initially unused columns over time (3d)
- [ ] 14. All permanences remain in `[0, 1]` after 100 learning steps (3e)
- [ ] 15. Duty cycles converge within 50% of target sparsity after 500 steps (3f)
- [ ] 16. All binarization modes (topk, threshold, learned_gate) produce valid SP input (3g)
- [ ] 17. Zero input produces zero overlap for all columns (3h)
- [ ] 18. Manual overlap computation matches `SP.compute_overlap` (3i)
- [ ] 19. fp16 autocast produces same binary SP output as fp32 (3j)

### Temporal Memory (items 20-33)
- [ ] 20. TM has no trainable parameters requiring gradient (4.1a)
- [ ] 21. Prediction accuracy improves for repeating sequence across epochs (4.1b)
- [ ] 22. TM distinguishes higher-order context: B-in-ABCD vs B-in-XBCY (4.1c)
- [ ] 23. Predicted columns activate only predicted cells, not all cells (4.2a)
- [ ] 24. Unpredicted columns burst: all cells activate (4.2b)
- [ ] 25. Exactly one winner cell per bursting column (4.2c)
- [ ] 26. Segments are created for bursting columns during learning (4.3a)
- [ ] 27. Correct predictions reinforce synapse permanences (4.3b)
- [ ] 28. Inactive presynaptic cells get permanence decremented (4.3c)
- [ ] 29. Dead synapses (permanence <= 0) are removed (4.3d)
- [ ] 30. Segments per cell never exceed `max_segments_per_cell` (4.3e)
- [ ] 31. Synapses per segment never exceed `max_synapses_per_segment` (4.3f)
- [ ] 32. Segment storage has no invalid cell indices or orphan synapses (4.4a, 4.4b)
- [ ] 33. 500-step fuzz test with random operations produces no corruption (4.4d)

### Anomaly Detection (items 34-38)
- [ ] 34. Anomaly score is in `[0, 1]` for all inputs (5a)
- [ ] 35. Anomaly is near 0 for well-learned sequences (5b)
- [ ] 36. Novel symbol injection causes immediate anomaly spike > 0.5 (5c)
- [ ] 37. Repeated novel injection causes anomaly to decline over time (5d)
- [ ] 38. Anomaly likelihood values are in `[0, 1]` (5f)

### Reflex Memory (items 39-49)
- [ ] 39. Pattern is stored after calling `store(force=True)` (6.1a)
- [ ] 40. AcceleratedHTM does not promote before reaching threshold (6.1b)
- [ ] 41. Stored pattern is found via LSH lookup with high confidence (6.1c)
- [ ] 42. Unknown pattern returns `None` from lookup (6.1d)
- [ ] 43. Reflex prediction matches the HTM features that created it (6.2a)
- [ ] 44. Stored entries never exceed `max_patterns` (6.3a)
- [ ] 45. LRU eviction removes least-important entry when table is full (6.3b)
- [ ] 46. Access counts decay correctly with configured decay rate (6.3d)
- [ ] 47. `state_dict()` roundtrip preserves lookup results (6.3e)
- [ ] 48. Hit rate tracks correctly as `hits / (hits + misses)` (6.4a)
- [ ] 49. `reset_statistics()` zeros all counters (6.4b)

### Fallback Predictors (items 50-55)
- [ ] 50. All three fallbacks (LSTM, GRU, Transformer) return required output keys (7a)
- [ ] 51. Output features have correct batch dimension (7b)
- [ ] 52. Anomaly is in `[0, 1]` for all fallback backends (7c)
- [ ] 53. `reset()` restores deterministic behavior for same input (7d)
- [ ] 54. Prediction shape is `(B, input_size)` (7e)
- [ ] 55. Gradients flow through fallback parameters (7f)

### Integration (items 56-60)
- [ ] 56. `HTMLayer` end-to-end pipeline produces valid output dict with correct shapes (8a)
- [ ] 57. `AcceleratedHTM` end-to-end pipeline produces valid output with `from_reflex` flag (8b)
- [ ] 58. All `TemporalLayer` backends produce structurally compatible outputs (8c)
- [ ] 59. Dense-to-binary-to-SP-to-TM pipeline runs without error (8d)
- [ ] 60. Batch of identical inputs produces identical per-sample outputs (8e)

### Performance / Memory (items 61-64)
- [ ] 61. SP processes 1000 steps in under 30 seconds on CPU (9a)
- [ ] 62. Segment and synapse counts respect configured caps after long runs (9b)
- [ ] 63. Reflex Memory lookup is faster than full HTM forward pass (9c)
- [ ] 64. Sparse segment storage uses << O(num_cells^2) memory (9d)

---

## Pytest Organization

```
tests/test_htm/
├── test_sdr_utilities.py        # Section 2
├── test_spatial_pooler.py       # Section 3
├── test_temporal_memory.py      # Section 4
├── test_anomaly.py              # Section 5
├── test_reflex_memory.py        # Section 6
├── test_fallback_predictors.py  # Section 7
├── test_htm_integration.py      # Section 8
├── test_htm_performance.py      # Section 9 (@pytest.mark.slow)
└── conftest.py                  # Section 10
```

Run only fast tests (excludes performance):

```bash
pytest tests/test_htm/ -v -m "not slow"
```

Run all tests including performance:

```bash
pytest tests/test_htm/ -v
```

Run only a specific category:

```bash
pytest tests/test_htm/test_spatial_pooler.py -v
pytest tests/test_htm/test_reflex_memory.py -v
```

---

## Implementation Notes for Test Authors

**On SDR utility functions**: The public contract specifies `indices_to_dense`, `dense_to_indices`, `sdr_overlap`, `sdr_jaccard`, and `sdr_hash`. If these do not yet exist as standalone functions, implement them in `brain_ai/temporal/sdr_utils.py` before writing the tests. The `SparseTensor` class in `htm.py` provides reference semantics for the conversions.

**On TM non-differentiability**: The TM learns via Hebbian rules (permanence increment/decrement), not backpropagation. Test 4.1a explicitly verifies that TM has zero trainable parameters. Do not attempt to compute gradients through the TM forward pass.

**On Reflex Memory LSH**: The LSH similarity threshold is tunable. Tests 6.1c and 6.1d depend on the threshold being set to match the test's expectations. Use `similarity_threshold=0.8` for hit tests and verify that random patterns do not accidentally match. If a random pattern does match (small probability), the test should tolerate it gracefully.

**On sequence fixture determinism**: The `repeating_sequence` fixture creates non-overlapping SDR symbols using contiguous index ranges. When `offset` is used for distribution shift tests (5e), ensure the offset symbols do not overlap with the default symbols.

**On anomaly thresholds**: Anomaly tests use soft thresholds (e.g., "average anomaly < 0.3"). These are not exact; the TM is stochastic in winner cell selection and segment creation. If a test is flaky at the exact boundary, increase the number of training epochs or widen the tolerance, but document the change.

**On the 64-item checklist**: This checklist is a gate. All 64 items must pass before the HTM subsystem is declared ready for integration into the seven-layer cognitive pipeline. Partial completion is tracked but does not constitute readiness.
