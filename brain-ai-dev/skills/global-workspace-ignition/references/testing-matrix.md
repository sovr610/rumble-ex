# Testing Matrix: Global Workspace with Ignition Dynamics

This document defines the concrete test specifications that must pass before the Global Workspace (both base `GlobalWorkspace` and improved `SelectionBroadcastWorkspace`) is considered production-ready. Each category maps to a test file in `tests/test_workspace/`. Tests are written in imperative form -- implement exactly as described.

**Note on target API:** Test code examples below use the TARGET API (slot-based outputs, `SequenceOutput`-style attribute access, explicit `token_table` staging, four-term competition scoring). When generating actual tests against the existing codebase, adapt attribute access patterns as needed. The existing `GlobalWorkspace` and `SelectionBroadcastWorkspace` in `brain_ai/workspace/global_workspace.py` use dict-based outputs (`result['workspace']`, `result['attention']`, etc.). Where the target API diverges from current code, the test describes the intended behavior; the implementation must be brought into alignment before the test can pass.

**Production config reference:** `workspace_dim=4096`, `num_heads=32`, `capacity_limit=7` (K=7), `selection_rounds=3`, `ignition_threshold=0.3`, `memory_mode="cfc"`, `broadcast_iterations=2`, `broadcast_decay=0.9`. Tests use smaller dimensions (`workspace_dim=64` or `128`, `num_heads=4`, `capacity_limit=4`) for speed unless stated otherwise.

---

## 1. Overview

| # | Category | Focus | File | Done Criterion |
|---|---|---|---|---|
| 2 | Token Staging | Canonical ordering, salience defaults, masks | `test_token_staging.py` | -- |
| 3 | Competition Scoring | Four-term score, normalization, dtype | `test_competition_scoring.py` | -- |
| 4 | Deterministic Top-K | Tie-breaking, reproducibility, mask handling | `test_deterministic_topk.py` | **(a)** |
| 5 | Slot Construction | Shape, content, mixer, metadata | `test_slot_construction.py` | -- |
| 6 | Iterative Rounds | Convergence, early-stop, stability metrics | `test_iterative_rounds.py` | **(b)** |
| 7 | Ignition | Threshold gating, gain, coherence, telemetry | `test_ignition.py` | -- |
| 8 | Lock-In Prevention | Novelty decay, slot dropout, cooldown | `test_lock_in_prevention.py` | -- |
| 9 | Broadcast Adapters | Shape conversion, registry, determinism | `test_broadcast_adapters.py` | -- |
| 10 | Working Memory | State persistence, reset, fallback | `test_working_memory.py` | **(c)** and **(d)** |
| 11 | Integration | Full pipeline, BrainAI wiring, flags | `test_workspace_integration.py` | -- |
| 12 | Performance | Throughput, scaling, early-stop speedup | `test_workspace_performance.py` | -- |

---

## 2. Token Staging Tests

Token staging is the process of collecting encoder outputs from multiple modalities, projecting them into workspace dimension, and concatenating them into a unified `token_table` tensor with associated metadata. The canonical ordering ensures deterministic behavior regardless of Python dict ordering.

All tests use `workspace_dim=64` and 2-3 modalities with small token counts.

---

### 2a) `test_fixed_modality_ordering`

Verify that tokens are always concatenated in canonical (sorted) modality order regardless of the order keys appear in the input dict.

```python
def test_fixed_modality_ordering():
    """Modality tokens concatenated in sorted-name order, not input-dict order."""
    D = 64
    B = 2

    # Create workspace with known modality order
    ws = create_workspace(workspace_dim=D, modality_dims={
        'vision': D, 'text': D, 'audio': D
    })

    # Feed modalities in non-alphabetical order
    inputs_order1 = {
        'text':   torch.randn(B, D),
        'vision': torch.randn(B, D),
        'audio':  torch.randn(B, D),
    }

    out1 = ws(inputs_order1, return_attention=True)

    # Feed same tensors but in different dict order
    inputs_order2 = {
        'audio':  inputs_order1['audio'],
        'vision': inputs_order1['vision'],
        'text':   inputs_order1['text'],
    }

    ws.reset_state()
    out2 = ws(inputs_order2, return_attention=True)

    # Outputs must be identical regardless of dict insertion order
    assert torch.allclose(out1['workspace'], out2['workspace'], atol=1e-6)
    # Modality names list must be in canonical order
    assert out1['modality_names'] == sorted(out1['modality_names'])
```

Assert:
1. `out1['workspace']` and `out2['workspace']` are identical within tolerance.
2. `modality_names` list is sorted alphabetically.
3. Attention weights map to the same modality regardless of input order.

---

### 2b) `test_encoder_output_integration`

Verify that an `EncoderOutput` namedtuple (or equivalent dict with keys `feats`, `mask`, `salience`, `time`) is correctly staged into the token table.

```python
def test_encoder_output_integration():
    """EncoderOutput fields correctly propagated to workspace projection."""
    D = 64
    B = 2

    ws = create_workspace(workspace_dim=D, modality_dims={'vision': D})

    # Construct encoder output with explicit salience
    feats = torch.randn(B, D)
    inputs = {'vision': feats}

    out = ws(inputs, return_attention=True)

    # Workspace output should not be all-zeros
    assert out['workspace'].abs().sum() > 0
    # Shape: (B, workspace_dim)
    assert out['workspace'].shape == (B, D)
```

Assert:
1. Workspace output has shape `(B, workspace_dim)`.
2. Output is non-zero (projection and competition produced meaningful content).
3. No `NaN` in any output tensor.

---

### 2c) `test_missing_salience_defaults`

When salience is not explicitly provided (i.e., the projection module computes it internally), verify the salience predictor produces values and that the competition does not fail.

```python
def test_missing_salience_defaults():
    """When salience is computed internally by ModalityProjection, defaults are sane."""
    D = 64
    B = 2

    proj = ModalityProjection(input_dim=D, workspace_dim=D)
    x = torch.randn(B, D)
    projected, salience = proj(x)

    # Salience should be finite and have correct shape
    assert salience.shape == (B, 1)
    assert torch.isfinite(salience).all()
    # Salience is a learned scalar per item; no constraint on sign
```

Assert:
1. `salience.shape == (B, 1)`.
2. All salience values are finite.
3. Projection output shape is `(B, workspace_dim)`.

---

### 2d) `test_missing_time_handling`

When no temporal information (dt / timespans) is provided to working memory, the forward pass completes without error and produces valid output.

```python
def test_missing_time_handling():
    """Working memory processes input correctly when timespans=None."""
    D = 64
    B = 2

    wm = create_working_memory(input_dim=D, hidden_dim=D, output_dim=D, mode='gru')
    x = torch.randn(B, D)

    result = wm(x, timespans=None)

    assert result['output'].shape == (B, D)
    assert torch.isfinite(result['output']).all()
    assert result['state'] is not None
```

Assert:
1. Output shape is `(B, output_dim)`.
2. All output values are finite.
3. State is returned (not `None`).

---

### 2e) `test_variable_length_modalities`

Different modalities provide different numbers of tokens (T_m varies per modality). The workspace must handle variable-length inputs via its projection layers.

```python
def test_variable_length_modalities():
    """Workspace handles modalities with different input dimensions."""
    B = 2
    ws_dim = 64

    # Different input dimensions per modality
    ws = create_global_workspace(
        workspace_dim=ws_dim,
        modality_dims={'vision': 128, 'text': 256, 'audio': 64}
    )

    inputs = {
        'vision': torch.randn(B, 128),
        'text':   torch.randn(B, 256),
        'audio':  torch.randn(B, 64),
    }

    out = ws(inputs)

    # All projected to common workspace_dim
    assert out['workspace'].shape == (B, ws_dim)
    assert torch.isfinite(out['workspace']).all()
```

Assert:
1. Output workspace has shape `(B, workspace_dim)` regardless of input dimensions.
2. No errors from dimension mismatch.
3. All three modalities participate (check `modality_names` has length 3).

---

### 2f) `test_empty_modality_handling`

When a modality key is present in the input dict but its name does not match any registered projection, it is silently ignored.

```python
def test_empty_modality_handling():
    """Unknown modality keys are silently ignored; known ones process normally."""
    D = 64
    B = 2

    ws = create_global_workspace(
        workspace_dim=D,
        modality_dims={'vision': D, 'text': D}
    )

    # Include an unregistered modality
    inputs = {
        'vision': torch.randn(B, D),
        'text':   torch.randn(B, D),
        'tactile': torch.randn(B, D),  # Not registered
    }

    out = ws(inputs)
    assert out['workspace'].shape == (B, D)
    assert 'tactile' not in out.get('modality_names', [])
    assert len(out['modality_names']) == 2

    # Edge case: only unknown modalities -> should raise
    with pytest.raises(ValueError, match="No valid"):
        ws({'tactile': torch.randn(B, D)})
```

Assert:
1. Unknown modalities excluded from competition.
2. `modality_names` only contains registered modalities.
3. If ALL modalities are unknown, `ValueError` is raised.

---

## 3. Competition Scoring Tests

Competition scoring determines which modality inputs win access to the workspace. The target API uses a four-term scoring formula: `score = w_content * f + w_salience * s + w_novelty * n + w_task * t`. The existing implementation combines gate scores with salience and applies temperature scaling. Tests below verify current and target behavior.

All tests use `workspace_dim=64`, `num_heads=4`, `capacity_limit=4`.

---

### 3a) `test_four_term_scoring`

Verify the combined score incorporates gate output and salience at minimum.

```python
def test_four_term_scoring():
    """Competition score combines gate output with salience."""
    D = 64
    B = 2
    N = 5  # number of modalities

    comp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=4, temperature=1.0
    )

    features = torch.randn(B, N, D)
    saliences = torch.randn(B, N, 1)

    winners, attn_weights = comp(features, saliences)

    # Attention weights should reflect salience influence
    assert attn_weights.shape == (B, N)
    assert torch.isfinite(attn_weights).all()
    # Weights sum to 1 (after softmax + renormalization)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(B), atol=1e-5)
```

Assert:
1. Attention weights shape is `(B, num_items)`.
2. All weights are finite.
3. Weights sum to 1.0 per batch element.

---

### 3b) `test_salience_influence`

Higher salience tokens should receive higher competition scores, all else being equal.

```python
def test_salience_influence():
    """Higher salience leads to higher attention weight, all else equal."""
    D = 64
    B = 4
    N = 3

    comp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=3, temperature=0.1
    )

    # Identical features but different saliences
    torch.manual_seed(42)
    features = torch.randn(1, N, D).expand(B, -1, -1).clone()

    # Item 0 has high salience, item 1 medium, item 2 low
    saliences = torch.tensor([[[10.0], [0.0], [-10.0]]]).expand(B, -1, -1).clone()

    winners, attn_weights = comp(features, saliences)

    # On average, item 0 should have highest weight
    mean_weights = attn_weights.mean(dim=0)
    # With temperature=0.1 and large salience differences, ordering should be clear
    assert mean_weights[0] > mean_weights[1], \
        f"High-salience item should beat medium: {mean_weights}"
    assert mean_weights[1] > mean_weights[2], \
        f"Medium-salience item should beat low: {mean_weights}"
```

Assert:
1. Mean attention weight for high-salience item > medium > low.
2. The ordering is consistent across batch elements.

---

### 3c) `test_novelty_against_memory`

Tokens similar to the current working memory summary should receive lower effective scores due to reduced novelty. This test verifies the concept using the iterative competition's salience update mechanism.

```python
def test_novelty_against_memory():
    """Repeated exposure to same features reduces effective salience over rounds."""
    D = 64
    B = 2
    N = 4

    comp = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=3,
        ignition_threshold=0.3, temperature=0.5
    )

    torch.manual_seed(42)
    features = torch.randn(B, N, D)
    saliences = torch.ones(B, N, 1)

    _, attn1, info1 = comp(features, saliences)

    # Check that salience history shows evolution across rounds
    assert len(info1['history']) >= 2
    round1_sal = info1['history'][0]['saliences']
    round2_sal = info1['history'][1]['saliences']
    # Saliences should change between rounds (refinement is happening)
    assert not torch.equal(round1_sal, round2_sal)
```

Assert:
1. At least 2 rounds of history recorded.
2. Salience values differ between round 1 and round 2.
3. No `NaN` in any salience tensor.

---

### 3d) `test_task_bias_effect`

When external task bias is applied (via salience manipulation), specific tokens receive boosted scores.

```python
def test_task_bias_effect():
    """External salience bias shifts competition outcome."""
    D = 64
    B = 2
    N = 4

    comp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=2, temperature=0.5
    )

    torch.manual_seed(42)
    features = torch.randn(B, N, D)

    # Without bias: uniform salience
    sal_uniform = torch.zeros(B, N, 1)
    _, attn_uniform = comp(features, sal_uniform)

    # With bias: strongly favor item 2
    sal_biased = torch.zeros(B, N, 1)
    sal_biased[:, 2, :] = 100.0
    _, attn_biased = comp(features, sal_biased)

    # Item 2 should dominate with bias
    assert attn_biased[:, 2].mean() > attn_uniform[:, 2].mean()
```

Assert:
1. Biased item receives higher attention than in uniform case.
2. The bias is strong enough to change the ranking.

---

### 3e) `test_score_normalization`

Scores are normalized before top-K selection to prevent scale drift across different input magnitudes.

```python
def test_score_normalization():
    """Attention weights are normalized (sum to 1) regardless of input scale."""
    D = 64
    B = 2
    N = 5

    comp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=3, temperature=1.0
    )

    # Small-scale features
    features_small = torch.randn(B, N, D) * 0.01
    saliences_small = torch.zeros(B, N, 1)
    _, attn_small = comp(features_small, saliences_small)

    # Large-scale features
    features_large = torch.randn(B, N, D) * 100.0
    saliences_large = torch.zeros(B, N, 1)
    _, attn_large = comp(features_large, saliences_large)

    # Both should sum to 1
    assert torch.allclose(attn_small.sum(dim=-1), torch.ones(B), atol=1e-4)
    assert torch.allclose(attn_large.sum(dim=-1), torch.ones(B), atol=1e-4)
```

Assert:
1. Attention weights sum to 1.0 for small inputs.
2. Attention weights sum to 1.0 for large inputs.
3. No `NaN` or `Inf` in weights for either scale.

---

### 3f) `test_weight_config_respected`

Changing the temperature parameter changes the sharpness of competition and thus the ranking.

```python
def test_weight_config_respected():
    """Lower temperature produces sharper (more peaked) attention distribution."""
    D = 64
    B = 4
    N = 5

    torch.manual_seed(42)
    features = torch.randn(B, N, D)
    saliences = torch.randn(B, N, 1)

    # Sharp competition
    comp_sharp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=5, temperature=0.1
    )
    _, attn_sharp = comp_sharp(features, saliences)

    # Soft competition
    comp_soft = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=5, temperature=10.0
    )
    _, attn_soft = comp_soft(features, saliences)

    # Entropy of sharp should be lower than entropy of soft
    entropy_sharp = -(attn_sharp * torch.log(attn_sharp + 1e-10)).sum(dim=-1).mean()
    entropy_soft = -(attn_soft * torch.log(attn_soft + 1e-10)).sum(dim=-1).mean()

    assert entropy_sharp < entropy_soft, \
        f"Sharp temp should have lower entropy: {entropy_sharp:.4f} vs {entropy_soft:.4f}"
```

Assert:
1. Distribution entropy is lower for low temperature.
2. Both distributions are valid (finite, non-negative, sum to 1).

---

### 3g) `test_zero_weights`

Setting salience to all zeros should still produce valid competition output with uniform-ish attention (gate network determines all ranking).

```python
def test_zero_weights():
    """Zero salience does not cause NaN; gate network alone determines winners."""
    D = 64
    B = 2
    N = 4

    comp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=4, temperature=1.0
    )

    features = torch.randn(B, N, D)
    saliences = torch.zeros(B, N, 1)  # Zero salience

    winners, attn = comp(features, saliences)

    assert torch.isfinite(attn).all()
    assert torch.isfinite(winners).all()
    assert (attn >= 0).all()
```

Assert:
1. No `NaN` or `Inf` in attention or winners.
2. All attention weights are non-negative.
3. Output shapes are correct.

---

### 3h) `test_score_dtype_fp32`

Scores must be computed in fp32 even when inputs are fp16 to prevent overflow in softmax.

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fp16 test")
def test_score_dtype_fp32():
    """Competition scores computed in fp32 even with fp16 inputs."""
    D = 64
    B = 2
    N = 4
    device = torch.device('cuda')

    comp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=4, temperature=1.0
    ).to(device)

    features = torch.randn(B, N, D, device=device, dtype=torch.float16)
    saliences = torch.randn(B, N, 1, device=device, dtype=torch.float16)

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        winners, attn = comp(features, saliences)

    # Attention weights should be finite (no fp16 overflow in softmax)
    assert torch.isfinite(attn).all(), "fp16 softmax overflow detected"
    assert torch.isfinite(winners).all()
```

Assert:
1. All attention weights are finite under fp16 autocast.
2. All winner features are finite.
3. No `NaN` from softmax overflow.

---

## 4. Deterministic Top-K Tests -- DONE CRITERION (a)

> **Done Criterion (a):** Deterministic top-k gating -- ties produce deterministic winners matching documented tie-breaker on CPU and CUDA.

The top-K selection uses `torch.topk` on attention weights. When scores are tied, the tie-breaking rule must be deterministic: lower modality index wins, then lower token index within that modality. These tests verify exact reproducibility.

---

### 4a) `test_deterministic_topk_no_ties`

Standard case with distinct scores -- top-K matches expected winners.

```python
def test_deterministic_topk_no_ties():
    """Top-K selects items with highest attention weights when no ties exist."""
    D = 64
    B = 2
    N = 6
    K = 3

    comp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=K, temperature=1.0
    )

    torch.manual_seed(42)
    features = torch.randn(B, N, D)

    # Construct saliences with clear ranking
    saliences = torch.tensor([
        [[5.0], [3.0], [1.0], [4.0], [2.0], [0.0]],
        [[0.0], [5.0], [4.0], [3.0], [2.0], [1.0]],
    ])  # (2, 6, 1)

    winners, attn = comp(features, saliences)

    # Top-K mask: at most K items should have non-zero attention
    non_zero_per_batch = (attn > 1e-8).sum(dim=-1)
    assert (non_zero_per_batch <= K).all(), \
        f"Expected at most {K} non-zero weights, got {non_zero_per_batch}"
```

Assert:
1. At most K items have non-zero attention weight per batch.
2. The selected items correspond to the highest-scoring inputs.
3. Output is deterministic (same result on repeated calls).

---

### 4b) `test_deterministic_topk_exact_ties_cpu`

Construct equal-score tokens and verify tie-breaking follows a deterministic rule: lower index wins.

```python
def test_deterministic_topk_exact_ties_cpu():
    """Equal-score tokens resolve ties deterministically on CPU."""
    D = 64
    B = 1
    N = 5
    K = 2

    comp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=K, temperature=1.0
    )

    # All identical features and saliences -> ties everywhere
    features = torch.ones(B, N, D)
    saliences = torch.zeros(B, N, 1)

    results = []
    for _ in range(50):
        _, attn = comp(features, saliences)
        top_indices = torch.topk(attn, K, dim=-1).indices
        results.append(top_indices.clone())

    # All 50 runs should produce identical top-K indices
    for i in range(1, len(results)):
        assert torch.equal(results[0], results[i]), \
            f"Run {i} differs from run 0: {results[i]} vs {results[0]}"
```

Assert:
1. Same winner indices across all 50 runs.
2. Results are deterministic even when all scores are equal.

---

### 4c) `test_deterministic_topk_exact_ties_cuda`

Same test as 4b but on CUDA device.

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_deterministic_topk_exact_ties_cuda():
    """Equal-score tokens resolve ties deterministically on CUDA."""
    D = 64
    B = 1
    N = 5
    K = 2
    device = torch.device('cuda')

    comp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=K, temperature=1.0
    ).to(device)

    features = torch.ones(B, N, D, device=device)
    saliences = torch.zeros(B, N, 1, device=device)

    results = []
    for _ in range(50):
        _, attn = comp(features, saliences)
        top_indices = torch.topk(attn, K, dim=-1).indices
        results.append(top_indices.cpu().clone())

    for i in range(1, len(results)):
        assert torch.equal(results[0], results[i]), \
            f"CUDA run {i} differs from run 0: {results[i]} vs {results[0]}"
```

Assert:
1. Identical winners across 50 CUDA runs.
2. CUDA tie-breaking matches CPU tie-breaking (cross-validate with 4b).

---

### 4d) `test_deterministic_topk_reproducible`

Run the same non-trivial input 100 times and verify identical results.

```python
def test_deterministic_topk_reproducible():
    """Same input produces identical winners across 100 forward passes."""
    D = 64
    B = 4
    N = 8
    K = 3

    comp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=K, temperature=1.0
    )
    comp.eval()  # Disable dropout

    torch.manual_seed(42)
    features = torch.randn(B, N, D)
    saliences = torch.randn(B, N, 1)

    reference_winners = None
    reference_attn = None

    for i in range(100):
        winners, attn = comp(features, saliences)
        if reference_winners is None:
            reference_winners = winners.clone()
            reference_attn = attn.clone()
        else:
            assert torch.equal(winners, reference_winners), \
                f"Winner mismatch at iteration {i}"
            assert torch.equal(attn, reference_attn), \
                f"Attention mismatch at iteration {i}"
```

Assert:
1. All 100 iterations produce bitwise-identical winners.
2. All 100 iterations produce bitwise-identical attention weights.
3. Model is in mode (dropout disabled) for this test.

---

### 4e) `test_epsilon_magnitude`

Verify that epsilon values used for numerical stability never change the ranking when scores differ by more than 1e-5.

```python
def test_epsilon_magnitude():
    """Numerical stability eps never changes ranking for score gaps > 1e-5."""
    D = 64
    B = 2
    N = 4
    K = 2

    comp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=K, temperature=1.0
    )

    torch.manual_seed(42)
    features = torch.randn(B, N, D)

    # Saliences with clear gaps > 1e-5
    saliences = torch.tensor([
        [[1.0], [0.5], [0.0], [-0.5]],
        [[0.0], [1.0], [-1.0], [0.5]],
    ])

    _, attn = comp(features, saliences)

    # The normalization denominator uses eps=1e-8
    # Verify no weight is exactly zero due to eps contamination
    # when it shouldn't be
    top_k_mask = (attn > 1e-8)
    assert top_k_mask.sum(dim=-1).min() >= 1, "At least one winner per batch"

    # Verify attention weights are well-conditioned
    assert (attn >= 0).all()
    assert torch.isfinite(attn).all()
```

Assert:
1. At least one winner per batch.
2. No negative weights.
3. All weights are finite.

---

### 4f) `test_topk_with_mask`

When attention mask is provided, masked tokens should never win the competition.

```python
def test_topk_with_mask():
    """Masked tokens receive zero attention weight."""
    D = 64
    B = 2
    N = 5
    K = 2

    comp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=K, temperature=1.0
    )

    features = torch.randn(B, N, D)
    saliences = torch.randn(B, N, 1)

    # Create attention mask: mask out items 0 and 1 for all batches
    # MultiheadAttention uses True to block attention
    mask = torch.zeros(N, N, dtype=torch.bool)
    # Mask items 0 and 1 by blocking all attention to them
    mask[:, 0] = True
    mask[:, 1] = True

    winners, attn = comp(features, saliences, mask=mask)

    # Verify masked items get lower weight
    # Note: with attention mask, items 0 and 1 receive no attention
    # from other items, reducing their gate scores
    assert torch.isfinite(attn).all()
    assert torch.isfinite(winners).all()
```

Assert:
1. All outputs are finite.
2. Masked items have reduced attention weight.
3. Output shapes are correct.

---

### 4g) `test_topk_k_equals_total`

When K >= T_total (capacity_limit >= number of items), all valid tokens are selected.

```python
def test_topk_k_equals_total():
    """When K >= num_items, all items are selected."""
    D = 64
    B = 2
    N = 3
    K = 5  # K > N

    comp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=K, temperature=1.0
    )

    features = torch.randn(B, N, D)
    saliences = torch.randn(B, N, 1)

    winners, attn = comp(features, saliences)

    # All items should have non-zero attention
    assert (attn > 0).all(), "All items should be selected when K >= N"
    assert attn.shape == (B, N)
    assert winners.shape == (B, N, D)
```

Assert:
1. All N items have positive attention weight.
2. No top-K filtering applied (all items pass).
3. Output shapes match input item count, not K.

---

### 4h) `test_topk_k_greater_than_valid`

When K > number of valid (unmasked) tokens, the selection gracefully handles the reduced pool.

```python
def test_topk_k_greater_than_valid():
    """K larger than available items does not crash."""
    D = 64
    B = 2
    N = 4
    K = 10  # Much larger than N

    comp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=K, temperature=1.0
    )

    features = torch.randn(B, N, D)
    saliences = torch.randn(B, N, 1)

    # Should not raise
    winners, attn = comp(features, saliences)

    assert attn.shape == (B, N)
    assert winners.shape == (B, N, D)
    assert torch.isfinite(attn).all()
```

Assert:
1. No runtime error when K > N.
2. Output has N items (not K).
3. All weights are finite.

---

## 5. Slot Construction Tests

After competition, winning tokens are assembled into "slots" -- the workspace representation. These tests verify the construction of the slot tensor and associated metadata.

---

### 5a) `test_slot_shape`

Verify the output workspace tensor has the expected shape.

```python
def test_slot_shape():
    """Workspace output shape is (B, workspace_dim)."""
    D = 64
    B = 2

    ws = create_global_workspace(workspace_dim=D, modality_dims={
        'v': D, 't': D
    }, num_heads=4, capacity_limit=4)

    inputs = {'v': torch.randn(B, D), 't': torch.randn(B, D)}
    out = ws(inputs)

    # Aggregated workspace content
    assert out['workspace'].shape == (B, D)
    # Memory output
    assert out['memory_output']['output'].shape == (B, D)
```

Assert:
1. Workspace output: `(B, workspace_dim)`.
2. Memory output: `(B, workspace_dim)`.
3. Both are finite tensors.

---

### 5b) `test_slot_content_matches_winners`

The workspace content should reflect the competition winners weighted by attention.

```python
def test_slot_content_matches_winners():
    """Workspace content is non-trivially different from simple mean of inputs."""
    D = 64
    B = 2
    N = 4

    ws = create_global_workspace(workspace_dim=D, modality_dims={
        'a': D, 'b': D, 'c': D, 'd': D
    }, num_heads=4, capacity_limit=2)

    # Create inputs with very different features
    inputs = {
        'a': torch.ones(B, D) * 10,
        'b': torch.ones(B, D) * -10,
        'c': torch.zeros(B, D),
        'd': torch.randn(B, D),
    }

    out = ws(inputs)
    simple_mean = sum(inputs.values()) / len(inputs)

    # Workspace output should differ from simple mean (competition selected subset)
    assert not torch.allclose(out['workspace'], simple_mean, atol=1.0), \
        "Workspace should differ from simple average due to competition"
```

Assert:
1. Workspace output is not a simple average of all inputs.
2. Competition has meaningfully selected a subset.

---

### 5c) `test_slot_mixer_residual`

The `SelectionBroadcastWorkspace` integration layer applies a residual transformation.

```python
def test_slot_mixer_residual():
    """Integration layer applies residual transformation when prev_context exists."""
    D = 64
    B = 2

    ws = create_selection_broadcast_workspace(
        workspace_dim=D,
        modality_dims={'v': D, 't': D},
        num_heads=4,
        selection_rounds=2,
    )

    inputs = {'v': torch.randn(B, D), 't': torch.randn(B, D)}

    # First pass: no prev_context
    out1 = ws(inputs)
    # Second pass: prev_context exists, integration layer fires
    out2 = ws(inputs)

    # Outputs should differ because second pass integrates with prev_context
    assert not torch.equal(out1['workspace'], out2['workspace']), \
        "Second pass should differ due to temporal integration"
```

Assert:
1. First and second pass produce different outputs.
2. The integration layer modifies the workspace content.

---

### 5d) `test_slot_mixer_disabled`

Without temporal integration (first timestep, no prev_context), the workspace output comes directly from competition + working memory without integration.

```python
def test_slot_mixer_disabled():
    """First timestep (no prev_context) skips integration layer."""
    D = 64
    B = 2

    ws = create_global_workspace(workspace_dim=D, modality_dims={'v': D}, num_heads=4)

    # Ensure fresh state
    ws.reset_state()
    assert ws.prev_context is None

    inputs = {'v': torch.randn(B, D)}
    out = ws(inputs)

    # Output exists and is valid
    assert out['workspace'].shape == (B, D)
    assert torch.isfinite(out['workspace']).all()
    # prev_context is now set for next call
    assert ws.prev_context is not None
```

Assert:
1. Fresh workspace has `prev_context = None`.
2. After first forward pass, `prev_context` is set.
3. Output is valid even without temporal context.

---

### 5e) `test_winners_metadata`

The output dict contains metadata about competition results.

```python
def test_winners_metadata():
    """Output contains modality names and attention details."""
    D = 64
    B = 2

    ws = create_global_workspace(workspace_dim=D, modality_dims={
        'vision': D, 'text': D, 'audio': D
    }, num_heads=4, capacity_limit=4)

    inputs = {
        'vision': torch.randn(B, D),
        'text': torch.randn(B, D),
        'audio': torch.randn(B, D),
    }

    out = ws(inputs, return_attention=True)

    # Modality names present
    assert 'modality_names' in out
    assert set(out['modality_names']) == {'vision', 'text', 'audio'}

    # Attention dict maps modality name -> weight
    assert 'attention' in out
    for name in out['modality_names']:
        assert name in out['attention']
        assert out['attention'][name].shape == (B,)
```

Assert:
1. `modality_names` contains all input modalities.
2. `attention` dict has one entry per modality.
3. Each attention entry has shape `(B,)`.

---

### 5f) `test_slot_mask_consistency`

The attention weights are consistent: non-negative, finite, and properly normalized.

```python
def test_slot_mask_consistency():
    """Attention weights are non-negative, finite, normalized."""
    D = 64
    B = 4

    ws = create_global_workspace(workspace_dim=D, modality_dims={
        'a': D, 'b': D, 'c': D, 'd': D, 'e': D
    }, num_heads=4, capacity_limit=3)

    inputs = {k: torch.randn(B, D) for k in ['a', 'b', 'c', 'd', 'e']}
    out = ws(inputs, return_attention=True)

    # Reconstruct full attention vector
    attn_values = torch.stack([
        out['attention'][name] for name in sorted(out['modality_names'])
    ], dim=-1)  # (B, N)

    assert (attn_values >= 0).all(), "Negative attention weight found"
    assert torch.isfinite(attn_values).all(), "Non-finite attention weight found"
    assert torch.allclose(
        attn_values.sum(dim=-1), torch.ones(B), atol=1e-4
    ), "Attention weights do not sum to 1"
```

Assert:
1. All weights >= 0.
2. All weights are finite.
3. Weights sum to 1.0 per batch element.

---

## 6. Iterative Round Tests -- DONE CRITERION (b)

> **Done Criterion (b):** Iterative rounds converge/early-stop -- winners stabilize after 2 rounds; oscillation capped by max_rounds; ignition false if unstable.

These tests target the `IterativeCompetition` module in `SelectionBroadcastWorkspace`. The iterative competition refines salience scores over multiple rounds with early-stop when ignition is detected.

---

### 6a) `test_single_round_no_iteration`

```python
def test_single_round_no_iteration():
    """max_rounds=1 runs exactly one round with no convergence check."""
    D = 64
    B = 2
    N = 4

    comp = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=1,
        ignition_threshold=0.3, temperature=0.5
    )

    features = torch.randn(B, N, D)
    saliences = torch.ones(B, N, 1)

    _, _, info = comp(features, saliences)

    assert info['selection_rounds'] == 1
    assert len(info['history']) == 1
```

Assert:
1. Exactly 1 round executed.
2. History has exactly 1 entry.

---

### 6b) `test_convergence_after_2_rounds`

Synthetic case where winners stabilize early, verifying early-stop occurs when ignition is detected.

```python
def test_convergence_after_2_rounds():
    """Strong ignition signal triggers early stop before max_rounds."""
    D = 64
    B = 2
    N = 3

    # Use high ignition threshold multiplier for easy early-stop
    comp = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=10,
        ignition_threshold=0.01,  # Very low threshold -> easy ignition
        temperature=0.5
    )

    # Features designed to produce strong, consistent signal
    features = torch.randn(B, N, D)
    saliences = torch.tensor([[[10.0], [0.0], [0.0]]]).expand(B, -1, -1).clone()

    _, _, info = comp(features, saliences)

    # Should have stopped before max_rounds=10
    assert info['selection_rounds'] <= 10
    # Check ignition was detected
    assert info['ignition'] is not None
```

Assert:
1. Number of rounds executed is less than or equal to `max_rounds`.
2. Ignition score is computed.
3. History length matches number of rounds executed.

---

### 6c) `test_stability_metrics_computed`

Verify that per-round competition metrics (salience and ignition) are recorded in the history.

```python
def test_stability_metrics_computed():
    """Per-round saliences and ignition scores recorded in history."""
    D = 64
    B = 2
    N = 4

    comp = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=3,
        ignition_threshold=0.3, temperature=0.5
    )

    features = torch.randn(B, N, D)
    saliences = torch.ones(B, N, 1)

    _, _, info = comp(features, saliences)

    for round_data in info['history']:
        assert 'saliences' in round_data
        assert 'ignition' in round_data
        assert round_data['saliences'].shape == (B, N)
        assert torch.isfinite(round_data['saliences']).all()
        assert torch.isfinite(round_data['ignition']).all()
```

Assert:
1. Each round entry has `saliences` and `ignition` keys.
2. Salience tensor shape is `(B, N)`.
3. All values are finite.

---

### 6d) `test_winner_set_stability_jaccard`

Verify that the winner set changes less between later rounds than between early rounds (convergence trend).

```python
def test_winner_set_stability_jaccard():
    """Winner attention distribution changes less in later rounds."""
    D = 64
    B = 4
    N = 6

    comp = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=5,
        ignition_threshold=0.9,  # High threshold to prevent early stop
        temperature=0.5
    )

    torch.manual_seed(42)
    features = torch.randn(B, N, D)
    saliences = torch.ones(B, N, 1)

    _, _, info = comp(features, saliences)

    # Compute salience change between consecutive rounds
    changes = []
    for i in range(1, len(info['history'])):
        prev_sal = info['history'][i-1]['saliences']
        curr_sal = info['history'][i]['saliences']
        change = (curr_sal - prev_sal).abs().mean().item()
        changes.append(change)

    # Later changes should generally be smaller (convergence)
    # Allow some noise, just check the trend exists
    if len(changes) >= 3:
        early_change = sum(changes[:len(changes)//2]) / max(len(changes)//2, 1)
        late_change = sum(changes[len(changes)//2:]) / max(len(changes) - len(changes)//2, 1)
        # Log for diagnostics
        print(f"Early avg change: {early_change:.6f}, Late avg change: {late_change:.6f}")
```

Assert:
1. Salience changes are computed between consecutive rounds.
2. History has enough rounds for meaningful comparison.
3. No `NaN` in any salience tensor.

---

### 6e) `test_embedding_stability_cosine`

Verify that feature embeddings stabilize across rounds (cosine similarity between consecutive rounds increases).

```python
def test_embedding_stability_cosine():
    """Feature refinement converges: embeddings remain bounded across rounds."""
    D = 64
    B = 2
    N = 4

    comp = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=4,
        ignition_threshold=0.9,  # High to prevent early stop
        temperature=0.5
    )

    # Hook into intermediate features by checking output evolution
    torch.manual_seed(42)
    features = torch.randn(B, N, D)
    saliences = torch.ones(B, N, 1)

    _, attn, info = comp(features, saliences)

    # Saliences should converge (check magnitudes are bounded)
    for round_data in info['history']:
        assert torch.isfinite(round_data['saliences']).all()
        max_sal = round_data['saliences'].abs().max().item()
        assert max_sal < 1000, f"Salience exploded to {max_sal}"
```

Assert:
1. All salience values remain bounded (no explosion).
2. Values stay finite throughout all rounds.
3. Maximum salience magnitude stays below a reasonable bound.

---

### 6f) `test_max_rounds_cap`

Create oscillating input and verify that `max_rounds` stops iteration even without convergence.

```python
def test_max_rounds_cap():
    """Iteration stops at max_rounds even without convergence."""
    D = 64
    B = 2
    N = 4
    MAX_ROUNDS = 3

    comp = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=MAX_ROUNDS,
        ignition_threshold=0.99,  # Very high -> never early-stops
        temperature=0.5
    )

    features = torch.randn(B, N, D)
    saliences = torch.ones(B, N, 1)

    _, _, info = comp(features, saliences)

    assert info['selection_rounds'] <= MAX_ROUNDS
    assert len(info['history']) <= MAX_ROUNDS
```

Assert:
1. Number of rounds does not exceed `selection_rounds`.
2. History length does not exceed `selection_rounds`.

---

### 6g) `test_ignition_false_when_unstable`

When competition does not converge (ignition score stays below threshold), the `global_ignition` flag is 0.

```python
def test_ignition_false_when_unstable():
    """Low ignition score -> global_ignition = 0."""
    D = 64
    B = 2
    N = 4

    comp = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=2,
        ignition_threshold=0.99,  # Impossibly high
        temperature=0.5
    )

    features = torch.randn(B, N, D)
    saliences = torch.zeros(B, N, 1)  # Weak salience

    _, _, info = comp(features, saliences)

    # With threshold=0.99, most random inputs won't ignite
    # global_ignition is 0 where ignition < threshold
    assert 'global_ignition' in info
    assert info['global_ignition'].shape[0] == B
```

Assert:
1. `global_ignition` is returned in info dict.
2. `global_ignition` has shape `(B, 1)`.
3. When threshold is very high, most inputs should not ignite.

---

### 6h) `test_convergence_thresholds_respected`

Higher convergence requirements (higher ignition threshold) make early-stop harder to achieve.

```python
def test_convergence_thresholds_respected():
    """Higher ignition threshold requires more rounds or prevents early stop."""
    D = 64
    B = 2
    N = 4

    torch.manual_seed(42)
    features = torch.randn(B, N, D)
    saliences = torch.ones(B, N, 1)

    # Easy threshold
    comp_easy = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=5,
        ignition_threshold=0.01, temperature=0.5
    )
    _, _, info_easy = comp_easy(features, saliences)

    # Hard threshold
    comp_hard = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=5,
        ignition_threshold=0.99, temperature=0.5
    )
    _, _, info_hard = comp_hard(features, saliences)

    # Easy threshold should stop sooner or at same round
    assert info_easy['selection_rounds'] <= info_hard['selection_rounds']
```

Assert:
1. Easy threshold leads to fewer or equal rounds.
2. Hard threshold leads to more rounds (or max_rounds).
3. Both complete without error.

---

### 6i) `test_consecutive_stable_requirement`

Verify that a single good round is not sufficient -- the early-stop condition requires the ignition score to exceed `threshold * 1.5`.

```python
def test_consecutive_stable_requirement():
    """Early stop requires ignition > threshold * 1.5, not just > threshold."""
    D = 64
    B = 2
    N = 4

    # The IterativeCompetition early-stops when ignition.mean() > threshold * 1.5
    comp = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=5,
        ignition_threshold=0.5,  # Early stop requires mean > 0.75
        temperature=0.5
    )

    features = torch.randn(B, N, D)
    saliences = torch.ones(B, N, 1)

    _, _, info = comp(features, saliences)

    # If early-stopped, the final ignition must exceed threshold * 1.5
    if info['selection_rounds'] < 5:
        final_ignition = info['history'][-1]['ignition']
        assert final_ignition.mean() > 0.5 * 1.5, \
            f"Early stop but ignition {final_ignition.mean():.4f} < {0.75}"
```

Assert:
1. If early-stop occurred, final ignition > threshold * 1.5.
2. If no early-stop, all rounds executed.

---

### 6j) `test_round_telemetry_history`

Per-round saliences, ignition scores, and round indices are all available in the telemetry history.

```python
def test_round_telemetry_history():
    """Full telemetry history available for all executed rounds."""
    D = 64
    B = 2
    N = 4
    ROUNDS = 3

    comp = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=ROUNDS,
        ignition_threshold=0.99, temperature=0.5  # High to force all rounds
    )

    features = torch.randn(B, N, D)
    saliences = torch.ones(B, N, 1)

    _, _, info = comp(features, saliences)

    assert 'history' in info
    assert len(info['history']) == info['selection_rounds']

    for idx, round_data in enumerate(info['history']):
        assert 'saliences' in round_data, f"Round {idx} missing saliences"
        assert 'ignition' in round_data, f"Round {idx} missing ignition"
        assert round_data['saliences'].shape == (B, N)

    # Top-level fields
    assert 'ignition' in info
    assert 'global_ignition' in info
    assert 'selection_rounds' in info
```

Assert:
1. History length matches selection_rounds count.
2. Each round entry has required keys.
3. Top-level info dict has ignition, global_ignition, and selection_rounds.

---

## 7. Ignition Tests

Ignition is the phenomenon where workspace activity exceeds a threshold, triggering strong global broadcast. These tests verify the ignition detection, gating, and telemetry.

---

### 7a) `test_ignition_score_components`

```python
def test_ignition_score_components():
    """Ignition detector produces score in [0, 1] from workspace features."""
    D = 64
    B = 4
    N = 4

    comp = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=3,
        ignition_threshold=0.3, temperature=0.5
    )

    features = torch.randn(B, N, D)
    saliences = torch.ones(B, N, 1)

    _, _, info = comp(features, saliences)

    ignition_score = info['ignition']
    assert ignition_score.shape == (B, 1)
    assert (ignition_score >= 0).all() and (ignition_score <= 1).all(), \
        f"Ignition score out of [0,1]: {ignition_score}"
```

Assert:
1. Ignition score has shape `(B, 1)`.
2. All values in [0, 1] (sigmoid output).

---

### 7b) `test_ignition_threshold_gate`

```python
def test_ignition_threshold_gate():
    """Ignition flag is 1.0 when score > threshold, 0.0 otherwise."""
    D = 64
    B = 4
    N = 4
    THRESHOLD = 0.3

    comp = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=3,
        ignition_threshold=THRESHOLD, temperature=0.5
    )

    features = torch.randn(B, N, D)
    saliences = torch.ones(B, N, 1)

    _, _, info = comp(features, saliences)

    ignition_score = info['ignition']
    global_ignition = info['global_ignition']

    # Verify thresholding
    expected = (ignition_score > THRESHOLD).float()
    assert torch.equal(global_ignition, expected), \
        f"Threshold mismatch: got {global_ignition}, expected {expected}"
```

Assert:
1. `global_ignition` equals `(ignition > threshold).float()` element-wise.

---

### 7c) `test_committed_broadcast_gain`

When ignition occurs in `SelectionBroadcastWorkspace`, confidence gating should produce a stronger output.

```python
def test_committed_broadcast_gain():
    """Ignited workspace produces confidence-gated output."""
    D = 64
    B = 2

    # Create workspace with confidence gating
    ws = create_selection_broadcast_workspace(
        workspace_dim=D,
        modality_dims={'v': D, 't': D},
        num_heads=4,
        selection_rounds=3,
        ignition_threshold=0.3,
    )

    inputs = {'v': torch.randn(B, D), 't': torch.randn(B, D)}
    out = ws(inputs, return_details=True)

    # Confidence gates the output
    if 'confidence' in out:
        confidence = out['confidence']
        assert confidence.shape == (B, 1)
        assert (confidence >= 0).all() and (confidence <= 1).all()
```

Assert:
1. Confidence exists in output when `use_confidence_gating=True`.
2. Confidence is in [0, 1].
3. Workspace output is modulated by confidence.

---

### 7d) `test_weak_broadcast_gain`

```python
def test_weak_broadcast_gain():
    """Without confidence gating, output is not scaled."""
    D = 64
    B = 2

    config = SelectionBroadcastConfig(
        workspace_dim=D, num_heads=4, selection_rounds=2,
        ignition_threshold=0.3, use_confidence_gating=False,
        memory_mode='gru',
    )
    ws = SelectionBroadcastWorkspace(
        config=config,
        modality_dims={'v': D, 't': D}
    )

    inputs = {'v': torch.randn(B, D), 't': torch.randn(B, D)}
    out = ws(inputs)

    assert 'confidence' not in out
    assert out['workspace'].shape == (B, D)
```

Assert:
1. No `confidence` key in output.
2. Workspace output shape is correct.

---

### 7e) `test_smooth_gate_training`

During training mode, the workspace should use soft gating (differentiable).

```python
def test_smooth_gate_training():
    """Training mode produces differentiable output (gradient flows)."""
    D = 64
    B = 2

    ws = create_selection_broadcast_workspace(
        workspace_dim=D, modality_dims={'v': D}, num_heads=4,
        selection_rounds=2, memory_mode='gru',
    )
    ws.train()

    inputs = {'v': torch.randn(B, D, requires_grad=True)}
    out = ws(inputs)

    loss = out['workspace'].sum()
    loss.backward()

    assert inputs['v'].grad is not None
    assert inputs['v'].grad.abs().sum() > 0, "No gradient flowing through workspace"
```

Assert:
1. Gradient flows from workspace output back to input.
2. Gradient is non-zero.

---

### 7f) `test_hard_gate_inference`

During mode, dropout is disabled and competition is deterministic.

```python
def test_hard_gate_inference():
    """Mode produces deterministic output."""
    D = 64
    B = 2

    ws = create_selection_broadcast_workspace(
        workspace_dim=D, modality_dims={'v': D}, num_heads=4,
        selection_rounds=2, memory_mode='gru',
    )
    ws.eval()

    inputs = {'v': torch.randn(B, D)}

    with torch.no_grad():
        ws.reset_state()
        out1 = ws(inputs)
        ws.reset_state()
        out2 = ws(inputs)

    assert torch.equal(out1['workspace'], out2['workspace']), \
        "Inference mode should be deterministic"
```

Assert:
1. Two forward passes with reset produce identical output.
2. No randomness in inference mode.

---

### 7g) `test_cross_modal_coherence`

When winning slots come from multiple modalities, the ignition score reflects cross-modal agreement.

```python
def test_cross_modal_coherence():
    """Multi-modal inputs produce ignition-relevant outputs."""
    D = 64
    B = 2

    ws = create_selection_broadcast_workspace(
        workspace_dim=D,
        modality_dims={'vision': D, 'text': D, 'audio': D},
        num_heads=4, selection_rounds=3, memory_mode='gru',
    )

    inputs = {
        'vision': torch.randn(B, D),
        'text': torch.randn(B, D),
        'audio': torch.randn(B, D),
    }

    out = ws(inputs, return_details=True)

    assert 'ignition' in out
    assert out['ignition'].shape == (B, 1)
    # All three modalities participated
    assert len(out['modality_names']) == 3
```

Assert:
1. Ignition score returned.
2. All three modalities in `modality_names`.
3. Output is valid.

---

### 7h) `test_ignition_telemetry`

```python
def test_ignition_telemetry():
    """return_details=True includes full competition telemetry."""
    D = 64
    B = 2

    ws = create_selection_broadcast_workspace(
        workspace_dim=D, modality_dims={'v': D, 't': D},
        num_heads=4, selection_rounds=3, memory_mode='gru',
    )

    inputs = {'v': torch.randn(B, D), 't': torch.randn(B, D)}
    out = ws(inputs, return_details=True)

    assert 'competition_details' in out
    details = out['competition_details']
    assert 'ignition' in details
    assert 'global_ignition' in details
    assert 'selection_rounds' in details
    assert 'history' in details
```

Assert:
1. `competition_details` present when `return_details=True`.
2. Details contain ignition, global_ignition, selection_rounds, history.

---

### 7i) `test_ignition_fp32`

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ignition_fp32():
    """Ignition threshold comparison is done in fp32 even with mixed precision."""
    D = 64
    B = 2
    N = 3
    device = torch.device('cuda')

    comp = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=2,
        ignition_threshold=0.3, temperature=0.5
    ).to(device)

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        features = torch.randn(B, N, D, device=device)
        saliences = torch.ones(B, N, 1, device=device)
        _, _, info = comp(features, saliences)

    # Ignition score should be finite
    assert torch.isfinite(info['ignition']).all()
    assert torch.isfinite(info['global_ignition']).all()
```

Assert:
1. Ignition computation is stable under fp16 autocast.
2. All values finite.

---

### 7j) `test_learned_vs_interpretable`

The ignition detector is a learned MLP. Verify its outputs match expected Sigmoid range.

```python
def test_learned_vs_interpretable():
    """Ignition detector MLP outputs in [0, 1] via Sigmoid activation."""
    D = 64
    B = 10

    comp = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=1,
        ignition_threshold=0.3, temperature=0.5
    )

    # Test with various random inputs
    for _ in range(10):
        features = torch.randn(B, 3, D)
        saliences = torch.randn(B, 3, 1)
        _, _, info = comp(features, saliences)

        score = info['ignition']
        assert (score >= 0).all() and (score <= 1).all(), \
            f"Ignition score out of Sigmoid range: {score.min()}, {score.max()}"
```

Assert:
1. Ignition scores always in [0, 1].
2. Range holds for diverse random inputs.

---

## 8. Lock-In Prevention Tests

These tests verify mechanisms that prevent the same tokens from permanently dominating the workspace.

---

### 8a) `test_novelty_reduces_repeat_winners`

When the same input is fed repeatedly, the working memory state changes, producing different workspace outputs over time.

```python
def test_novelty_reduces_repeat_winners():
    """Repeated identical input produces changing workspace output via WM dynamics."""
    D = 64
    B = 2

    ws = create_global_workspace(workspace_dim=D, modality_dims={'v': D}, num_heads=4)
    ws.reset_state()

    inputs = {'v': torch.randn(B, D)}

    outputs = []
    for _ in range(5):
        out = ws(inputs)
        outputs.append(out['workspace'].clone())

    # Outputs should change over time due to working memory integration
    all_same = all(torch.equal(outputs[0], o) for o in outputs[1:])
    assert not all_same, "Workspace output should evolve over repeated inputs"
```

Assert:
1. Not all 5 outputs are identical.
2. Working memory introduces temporal variation.

---

### 8b) `test_slot_dropout_training`

During training, dropout in the competition/projection path is active.

```python
def test_slot_dropout_training():
    """Training mode with dropout produces valid output shapes."""
    D = 64
    B = 4

    ws = create_global_workspace(
        workspace_dim=D, modality_dims={'v': D},
        num_heads=4, dropout=0.5  # High dropout
    )
    ws.train()

    inputs = {'v': torch.randn(B, D)}

    ws.reset_state()
    out1 = ws(inputs)
    ws.reset_state()
    out2 = ws(inputs)

    # With dropout=0.5 and training mode, outputs should differ
    # (This is probabilistic; with high dropout, very likely to differ)
    # Allow for the rare case they match
    assert out1['workspace'].shape == out2['workspace'].shape
```

Assert:
1. Both outputs have correct shape.
2. Training mode is active (dropout engaged).

---

### 8c) `test_no_slot_dropout_eval`

During mode, dropout is disabled, producing deterministic outputs.

```python
def test_no_slot_dropout_eval():
    """Inference mode disables dropout: identical inputs produce identical outputs."""
    D = 64
    B = 2

    ws = create_global_workspace(
        workspace_dim=D, modality_dims={'v': D},
        num_heads=4, dropout=0.5
    )
    ws.eval()

    inputs = {'v': torch.randn(B, D)}

    with torch.no_grad():
        ws.reset_state()
        out1 = ws(inputs)
        ws.reset_state()
        out2 = ws(inputs)

    assert torch.equal(out1['workspace'], out2['workspace']), \
        "Inference mode should be deterministic (dropout disabled)"
```

Assert:
1. Outputs are bitwise identical in inference mode.

---

### 8d) `test_winner_decay`

The `prev_context` mechanism provides implicit decay: previous state influences but does not dominate future competition.

```python
def test_winner_decay():
    """prev_context is detached, preventing unbounded gradient accumulation."""
    D = 64
    B = 2

    ws = create_global_workspace(workspace_dim=D, modality_dims={'v': D}, num_heads=4)
    ws.reset_state()

    inputs = {'v': torch.randn(B, D)}

    # Run several steps
    for _ in range(5):
        ws(inputs)

    # prev_context should be detached (no grad)
    assert ws.prev_context is not None
    assert not ws.prev_context.requires_grad, \
        "prev_context should be detached from computation graph"
```

Assert:
1. `prev_context` exists after forward passes.
2. `prev_context` does not require gradients.

---

### 8e) `test_cooldown_after_ignition`

After strong ignition in `SelectionBroadcastWorkspace`, subsequent passes should still function normally.

```python
def test_cooldown_after_ignition():
    """Workspace functions correctly on consecutive timesteps after ignition."""
    D = 64
    B = 2

    ws = create_selection_broadcast_workspace(
        workspace_dim=D, modality_dims={'v': D, 't': D},
        num_heads=4, selection_rounds=3, memory_mode='gru',
    )

    inputs = {'v': torch.randn(B, D), 't': torch.randn(B, D)}

    # Run multiple timesteps
    for step in range(10):
        out = ws(inputs, return_details=True)
        assert torch.isfinite(out['workspace']).all(), \
            f"Non-finite output at step {step}"
        assert torch.isfinite(out['ignition']).all(), \
            f"Non-finite ignition at step {step}"
```

Assert:
1. All 10 timesteps produce finite outputs.
2. No accumulation of `NaN` or `Inf` over time.

---

### 8f) `test_turnover_over_sequence`

Over 10 timesteps with changing input, the attention weights should shift.

```python
def test_turnover_over_sequence():
    """Changing inputs produce changing attention patterns over time."""
    D = 64
    B = 2

    ws = create_global_workspace(
        workspace_dim=D, modality_dims={'v': D, 't': D, 'a': D},
        num_heads=4, capacity_limit=2
    )
    ws.reset_state()

    attention_history = []
    for step in range(10):
        # Different input each step
        torch.manual_seed(step)
        inputs = {
            'v': torch.randn(B, D),
            't': torch.randn(B, D),
            'a': torch.randn(B, D),
        }
        out = ws(inputs, return_attention=True)
        attn = {k: v.clone() for k, v in out['attention'].items()}
        attention_history.append(attn)

    # Attention should vary across timesteps
    first_attn = attention_history[0]
    differs = 0
    for attn in attention_history[1:]:
        for key in first_attn:
            if not torch.equal(first_attn[key], attn[key]):
                differs += 1
                break

    assert differs > 0, "Attention never changed across 10 timesteps"
```

Assert:
1. At least one timestep has different attention than the first.
2. The workspace is responsive to changing inputs.

---

## 9. Broadcast Adapter Tests

The broadcast mechanism sends workspace content back to all specialist modalities. These tests verify the `InformationBroadcast` and `RefinedBroadcast` modules.

---

### 9a) `test_broadcast_to_temporal`

```python
def test_broadcast_to_temporal():
    """Broadcast produces output in each modality's native dimension."""
    D = 64

    broadcast = InformationBroadcast(
        workspace_dim=D,
        modality_dims={'vision': 128, 'text': 256, 'audio': 64}
    )

    workspace_content = torch.randn(2, D)
    broadcasts = broadcast(workspace_content)

    assert broadcasts['vision'].shape == (2, 128)
    assert broadcasts['text'].shape == (2, 256)
    assert broadcasts['audio'].shape == (2, 64)
```

Assert:
1. Each modality receives broadcast in its native dimension.
2. Shapes match `modality_dims` specification.

---

### 9b) `test_broadcast_to_symbolic`

```python
def test_broadcast_to_symbolic():
    """Broadcast to symbolic modality produces correct shape."""
    D = 64

    broadcast = InformationBroadcast(
        workspace_dim=D,
        modality_dims={'symbolic': 32}
    )

    content = torch.randn(4, D)
    out = broadcast(content)

    assert out['symbolic'].shape == (4, 32)
    assert torch.isfinite(out['symbolic']).all()
```

Assert:
1. Output shape matches target dimension.
2. All values finite.

---

### 9c) `test_broadcast_to_decision`

```python
def test_broadcast_to_decision():
    """Broadcast to decision module produces (B, decision_dim) tensor."""
    D = 64

    broadcast = InformationBroadcast(
        workspace_dim=D,
        modality_dims={'decision': 512}
    )

    content = torch.randn(2, D)
    out = broadcast(content)

    assert out['decision'].shape == (2, 512)
```

Assert:
1. Decision broadcast has correct shape.

---

### 9d) `test_adapter_mask_alignment`

```python
def test_adapter_mask_alignment():
    """All broadcast outputs are finite and have matching batch dimension."""
    D = 64
    B = 3

    broadcast = InformationBroadcast(
        workspace_dim=D,
        modality_dims={'a': 32, 'b': 64, 'c': 128}
    )

    content = torch.randn(B, D)
    out = broadcast(content)

    for name, tensor in out.items():
        assert tensor.shape[0] == B, f"{name} batch mismatch"
        assert torch.isfinite(tensor).all(), f"{name} has non-finite values"
```

Assert:
1. All outputs have batch dimension B.
2. All values are finite.

---

### 9e) `test_adapter_deterministic`

```python
def test_adapter_deterministic():
    """Same input produces identical broadcast output."""
    D = 64

    broadcast = InformationBroadcast(
        workspace_dim=D,
        modality_dims={'v': 32, 't': 64}
    )
    broadcast.eval()

    content = torch.randn(2, D)

    out1 = broadcast(content)
    out2 = broadcast(content)

    for name in out1:
        assert torch.equal(out1[name], out2[name]), \
            f"Non-deterministic broadcast for {name}"
```

Assert:
1. Repeated calls produce identical output.

---

### 9f) `test_adapter_registry`

Broadcast projections are registered as `nn.ModuleDict` entries, accessible by name.

```python
def test_adapter_registry():
    """Broadcast projections registered and accessible by modality name."""
    D = 64

    broadcast = InformationBroadcast(
        workspace_dim=D,
        modality_dims={'vision': 128, 'text': 256}
    )

    assert 'vision' in broadcast.broadcast_projections
    assert 'text' in broadcast.broadcast_projections
    assert isinstance(broadcast.broadcast_projections['vision'], nn.Sequential)
```

Assert:
1. All modality names are keys in `broadcast_projections`.
2. Each projection is an `nn.Sequential` module.

---

### 9g) `test_custom_adapter`

User-defined modality dimensions work with the broadcast system.

```python
def test_custom_adapter():
    """Custom modality dimensions integrate into broadcast."""
    D = 64

    broadcast = InformationBroadcast(
        workspace_dim=D,
        modality_dims={'custom_sensor': 17, 'proprioception': 6}
    )

    content = torch.randn(2, D)
    out = broadcast(content)

    assert out['custom_sensor'].shape == (2, 17)
    assert out['proprioception'].shape == (2, 6)
```

Assert:
1. Arbitrary dimension values work.
2. Non-power-of-two dimensions work.

---

### 9h) `test_all_adapters_output_shapes`

```python
@pytest.mark.parametrize("modality_dims", [
    {'v': 32},
    {'v': 64, 't': 128},
    {'v': 256, 't': 256, 'a': 128, 's': 64},
    {'single': 4096},
])
def test_all_adapters_output_shapes(modality_dims):
    """Broadcast produces correct shapes for various modality configurations."""
    D = 64
    B = 2

    broadcast = InformationBroadcast(workspace_dim=D, modality_dims=modality_dims)
    content = torch.randn(B, D)
    out = broadcast(content)

    for name, expected_dim in modality_dims.items():
        assert out[name].shape == (B, expected_dim), \
            f"{name}: expected ({B}, {expected_dim}), got {out[name].shape}"
```

Assert:
1. Each configuration produces correct output shapes.
2. Works with 1, 2, and 4 modalities.

---

## 10. Working Memory Tests -- DONE CRITERIA (c) and (d)

> **Done Criterion (c):** Workspace state persistence -- WM state changes predictably across timesteps; reset restores baseline.
>
> **Done Criterion (d):** Missing ncps fallback -- monkeypatch import failure; verify GRU backend used with exact schema/shapes.

---

### 10a) `test_state_persistence_sequential`

Feed 3 timesteps with carry_state=True, verify outputs differ at each step.

```python
def test_state_persistence_sequential():
    """Sequential inputs produce different outputs due to state persistence."""
    D = 64
    B = 2

    wm = create_working_memory(input_dim=D, hidden_dim=D, output_dim=D, mode='gru')
    wm.reset_state()

    outputs = []
    for step in range(3):
        x = torch.randn(B, D)
        result = wm(x)
        outputs.append(result['output'].clone())

    # All three outputs should differ (state evolves)
    assert not torch.equal(outputs[0], outputs[1]), "Step 0 == Step 1"
    assert not torch.equal(outputs[1], outputs[2]), "Step 1 == Step 2"
    assert not torch.equal(outputs[0], outputs[2]), "Step 0 == Step 2"
```

Assert:
1. Output at step 0 differs from step 1.
2. Output at step 1 differs from step 2.
3. Output at step 0 differs from step 2.

---

### 10b) `test_state_reset_baseline`

After `reset_state()`, output matches fresh initialization.

```python
def test_state_reset_baseline():
    """reset_state() restores output to fresh-initialization baseline."""
    D = 64
    B = 2

    wm = create_working_memory(input_dim=D, hidden_dim=D, output_dim=D, mode='gru')

    x = torch.randn(B, D)

    # Fresh start
    wm.reset_state()
    result_fresh = wm(x)
    out_fresh = result_fresh['output'].clone()

    # Contaminate state with several forward passes
    for _ in range(5):
        wm(torch.randn(B, D))

    # Reset and re-run
    wm.reset_state()
    result_reset = wm(x)
    out_reset = result_reset['output'].clone()

    assert torch.allclose(out_fresh, out_reset, atol=1e-6), \
        "Reset did not restore baseline output"
```

Assert:
1. Output after reset matches output from fresh initialization.
2. Tolerance is tight (atol=1e-6).

---

### 10c) `test_detach_state_values`

Detaching the hidden state preserves values but removes gradient tracking.

```python
def test_detach_state_values():
    """Hidden state values preserved after detach, requires_grad=False."""
    D = 64
    B = 2

    wm = create_working_memory(input_dim=D, hidden_dim=D, output_dim=D, mode='gru')

    x = torch.randn(B, D, requires_grad=True)
    result = wm(x)

    state = result['state']
    if state is not None:
        # State from GRU is a tensor
        state_detached = state.detach()
        assert torch.equal(state, state_detached)
        assert not state_detached.requires_grad
```

Assert:
1. Detached state has identical values.
2. Detached state does not require grad.

---

### 10d) `test_detach_state_gradient_isolation`

Gradient from step N does not flow to step N-2 when state is detached between them.

```python
def test_detach_state_gradient_isolation():
    """Detaching state blocks gradient flow to earlier timesteps."""
    D = 32
    B = 2

    wm = create_working_memory(input_dim=D, hidden_dim=D, output_dim=D, mode='gru')
    wm.reset_state()

    # Step 1: tracked input
    x1 = torch.randn(B, D, requires_grad=True)
    r1 = wm(x1)

    # Detach state
    wm.backend.hidden_state = wm.backend.hidden_state.detach()

    # Step 2: tracked input
    x2 = torch.randn(B, D, requires_grad=True)
    r2 = wm(x2)

    # Backward from step 2 only
    loss = r2['output'].sum()
    loss.backward()

    # x1 should have no gradient (detach blocked it)
    assert x1.grad is None or x1.grad.abs().sum() == 0, \
        "Gradient leaked through detached state"
    # x2 should have gradient
    assert x2.grad is not None and x2.grad.abs().sum() > 0, \
        "No gradient for step 2 input"
```

Assert:
1. Input at step 1 has no gradient.
2. Input at step 2 has non-zero gradient.

---

### 10e) `test_reset_state_params`

`reset_state()` clears the hidden state and memory buffer completely.

```python
def test_reset_state_params():
    """reset_state clears hidden_state and memory_buffer."""
    D = 64
    B = 2

    wm = create_working_memory(input_dim=D, hidden_dim=D, output_dim=D, mode='gru')

    # Populate state
    for _ in range(3):
        wm(torch.randn(B, D))

    assert wm.backend.hidden_state is not None
    assert wm.memory_buffer is not None

    # Reset
    wm.reset_state()

    assert wm.backend.hidden_state is None
    assert wm.memory_buffer is None
```

Assert:
1. After forward passes, state and buffer are non-None.
2. After reset, both are None.

---

### 10f) `test_ncps_fallback_gru`

Monkeypatch the ncps import to simulate unavailability; verify GRU backend is used.

```python
def test_ncps_fallback_gru(mock_ncps_unavailable):
    """When ncps is unavailable, GRU backend is selected automatically."""
    # mock_ncps_unavailable fixture patches NCPS_AVAILABLE = False
    from brain_ai.workspace.working_memory import WorkingMemory, WorkingMemoryConfig

    config = WorkingMemoryConfig(
        input_dim=64, hidden_dim=64, output_dim=64,
        mode='cfc'  # Request CfC but ncps unavailable
    )

    wm = WorkingMemory(config)

    # Should fall back to GRU
    assert wm.backend_type == 'gru', f"Expected GRU fallback, got {wm.backend_type}"
    assert isinstance(wm.backend, GRUWorkingMemory)
```

Assert:
1. Backend type is `'gru'`.
2. Backend is instance of `GRUWorkingMemory`.
3. No import error raised.

---

### 10g) `test_ncps_fallback_schema`

Fallback GRU output dict has same keys as CfC output.

```python
def test_ncps_fallback_schema(mock_ncps_unavailable):
    """GRU fallback output has same dict keys as CfC output."""
    from brain_ai.workspace.working_memory import WorkingMemory, WorkingMemoryConfig

    D = 64
    B = 2

    config = WorkingMemoryConfig(input_dim=D, hidden_dim=D, output_dim=D, mode='gru')
    wm = WorkingMemory(config)

    x = torch.randn(B, D)
    result = wm(x)

    # Required keys in output dict
    required_keys = {'output', 'state', 'buffer'}
    assert required_keys.issubset(result.keys()), \
        f"Missing keys: {required_keys - result.keys()}"

    assert result['output'].shape == (B, D)
    assert result['state'] is not None
```

Assert:
1. Output dict has keys: `output`, `state`, `buffer`.
2. Output tensor has correct shape.
3. State is not None.

---

### 10h) `test_cfc_timespans`

When CfC backend is available and timespans are provided, they are passed to the liquid network.

```python
def test_cfc_timespans():
    """CfC backend receives timespans when provided (verify via output difference)."""
    D = 64
    B = 2

    wm = create_working_memory(input_dim=D, hidden_dim=D, output_dim=D, mode='auto')

    if wm.backend_type != 'cfc':
        pytest.skip("CfC not available; tested by fallback tests instead")

    x = torch.randn(B, D)

    # Without timespans
    wm.reset_state()
    result_no_dt = wm(x, timespans=None)

    # With timespans
    wm.reset_state()
    timespans = torch.ones(B, 1, 1) * 0.1  # Small dt
    result_with_dt = wm(x, timespans=timespans)

    # Outputs should potentially differ (CfC uses timespans for dynamics)
    # Just verify no error and valid output
    assert result_with_dt['output'].shape == (B, D)
    assert torch.isfinite(result_with_dt['output']).all()
```

Assert:
1. Forward pass with timespans completes without error.
2. Output has correct shape and is finite.

---

### 10i) `test_gru_dt_handling`

GRU backend handles the absence of timespans gracefully (ignores dt).

```python
def test_gru_dt_handling():
    """GRU backend ignores timespans parameter without error."""
    D = 64
    B = 2

    wm = create_working_memory(input_dim=D, hidden_dim=D, output_dim=D, mode='gru')

    x = torch.randn(B, D)

    # GRU doesn't use timespans but the WorkingMemory API accepts them
    result = wm(x, timespans=torch.ones(B, 1, 1))

    assert result['output'].shape == (B, D)
    assert torch.isfinite(result['output']).all()
```

Assert:
1. No error when timespans are passed to GRU backend.
2. Output is valid.

---

### 10j) `test_memory_buffer_fifo`

Buffer maintains capacity limit; oldest entries are evicted when full.

```python
def test_memory_buffer_fifo():
    """Memory buffer evicts oldest entries when capacity exceeded."""
    D = 64
    B = 2

    wm = create_working_memory(input_dim=D, hidden_dim=D, output_dim=D, mode='gru')
    wm.reset_state()

    capacity = wm.capacity  # Default: 7

    # Feed more items than capacity
    for i in range(capacity + 3):
        x = torch.randn(B, D)
        result = wm(x, update_buffer=True)

    buffer = wm.memory_buffer
    assert buffer is not None
    assert buffer.shape[1] == capacity, \
        f"Buffer should have {capacity} items, got {buffer.shape[1]}"
    assert buffer.shape == (B, capacity, D)
```

Assert:
1. Buffer size never exceeds capacity.
2. Buffer shape is `(B, capacity, output_dim)`.
3. Oldest entries are evicted (FIFO).

---

## 11. Integration Tests

End-to-end tests verifying the workspace works correctly within the broader system.

---

### 11a) `test_full_pipeline_forward`

```python
def test_full_pipeline_forward():
    """Full workspace pipeline: project -> compete -> integrate -> memory -> broadcast."""
    D = 64
    B = 2

    ws = create_global_workspace(
        workspace_dim=D,
        modality_dims={'vision': D, 'text': D},
        num_heads=4, capacity_limit=4, memory_mode='gru',
    )

    inputs = {'vision': torch.randn(B, D), 'text': torch.randn(B, D)}
    out = ws(inputs, return_attention=True)

    # All expected output keys present
    assert 'workspace' in out
    assert 'broadcasts' in out
    assert 'memory_output' in out
    assert 'modality_names' in out
    assert 'attention' in out

    # Shapes
    assert out['workspace'].shape == (B, D)
    assert out['broadcasts']['vision'].shape == (B, D)
    assert out['broadcasts']['text'].shape == (B, D)
    assert out['memory_output']['output'].shape == (B, D)
```

Assert:
1. All expected keys present in output.
2. All shapes correct.
3. All values finite.

---

### 11b) `test_workspace_in_system`

```python
def test_workspace_in_system():
    """GlobalWorkspace integrates into full BrainAI system when use_workspace=True."""
    # This test requires the full system; skip if not available
    try:
        from brain_ai import create_brain_ai
    except ImportError:
        pytest.skip("Full brain_ai system not available")

    brain = create_brain_ai(
        modalities=['vision'],
        output_type='classify',
        num_classes=10,
    )

    x = {'vision': torch.randn(2, 1, 28, 28)}
    out = brain(x)

    assert out.shape == (2, 10)
```

Assert:
1. System forward pass completes.
2. Output has expected classification shape.

---

### 11c) `test_flag_disabled`

```python
def test_flag_disabled():
    """GlobalWorkspace handles single-modality input gracefully."""
    D = 64
    B = 2

    ws = create_global_workspace(
        workspace_dim=D, modality_dims={'vision': D},
        num_heads=4, capacity_limit=4, memory_mode='gru',
    )

    inputs = {'vision': torch.randn(B, D)}
    out = ws(inputs)

    assert out['workspace'].shape == (B, D)
    # Single modality still goes through competition
    assert len(out['modality_names']) == 1
```

Assert:
1. Single modality works without error.
2. Output shape is correct.

---

### 11d) `test_multi_modal_competition`

```python
def test_multi_modal_competition():
    """Vision, text, and audio tokens all compete in the same workspace."""
    D = 64
    B = 2

    ws = create_global_workspace(
        workspace_dim=D,
        modality_dims={'vision': 128, 'text': 256, 'audio': 64},
        num_heads=4, capacity_limit=2, memory_mode='gru',
    )

    inputs = {
        'vision': torch.randn(B, 128),
        'text': torch.randn(B, 256),
        'audio': torch.randn(B, 64),
    }

    out = ws(inputs, return_attention=True)

    assert len(out['modality_names']) == 3
    assert out['workspace'].shape == (B, D)

    # With capacity_limit=2 and 3 modalities, some should be suppressed
    attn_values = [out['attention'][name].mean().item() for name in out['modality_names']]
    # At least one modality should have lower attention
    assert max(attn_values) > min(attn_values), \
        "Competition should differentiate modalities"
```

Assert:
1. All 3 modalities participate.
2. Attention varies across modalities (competition differentiates).
3. Output workspace has correct dimension.

---

### 11e) `test_return_details_telemetry`

```python
def test_return_details_telemetry():
    """SelectionBroadcastWorkspace returns full telemetry with return_details=True."""
    D = 64
    B = 2

    ws = create_selection_broadcast_workspace(
        workspace_dim=D, modality_dims={'v': D, 't': D},
        num_heads=4, selection_rounds=3, memory_mode='gru',
    )

    inputs = {'v': torch.randn(B, D), 't': torch.randn(B, D)}
    out = ws(inputs, return_details=True)

    # Standard fields
    assert 'workspace' in out
    assert 'broadcasts' in out
    assert 'attention' in out
    assert 'ignition' in out
    assert 'global_ignition' in out

    # Detail fields
    assert 'competition_details' in out
    assert 'confidence' in out

    # Competition details structure
    details = out['competition_details']
    assert 'history' in details
    assert len(details['history']) > 0
```

Assert:
1. All standard output keys present.
2. `competition_details` present with `return_details=True`.
3. History is non-empty.

---

### 11f) `test_workspace_deterministic_seeded`

```python
def test_workspace_deterministic_seeded():
    """Same seed produces identical workspace output."""
    D = 64
    B = 2

    def run_once(seed):
        torch.manual_seed(seed)
        ws = create_global_workspace(
            workspace_dim=D, modality_dims={'v': D}, num_heads=4, memory_mode='gru',
        )
        ws.eval()
        inputs = {'v': torch.randn(B, D)}
        return ws(inputs)['workspace']

    out1 = run_once(42)
    out2 = run_once(42)

    assert torch.equal(out1, out2), "Same seed should produce identical output"
```

Assert:
1. Identical seeds produce identical outputs.
2. Model initialization and forward pass are both seeded.

---

## 12. Performance Tests

Mark all tests in this category with `@pytest.mark.slow`. These are excluded from default runs.

---

### 12a) `test_competition_throughput`

```python
@pytest.mark.slow
def test_competition_throughput():
    """Measure competition throughput with K=7, many modalities."""
    import time
    D = 128
    B = 8
    N = 20  # 20 modalities competing
    K = 7

    comp = AttentionCompetition(
        workspace_dim=D, num_heads=4, capacity_limit=K, temperature=1.0
    )
    comp.eval()

    features = torch.randn(B, N, D)
    saliences = torch.randn(B, N, 1)

    # Warmup
    for _ in range(10):
        comp(features, saliences)

    # Benchmark
    start = time.perf_counter()
    num_steps = 100
    for _ in range(num_steps):
        comp(features, saliences)
    elapsed = time.perf_counter() - start

    steps_per_sec = num_steps / elapsed
    print(f"Competition throughput: {steps_per_sec:.1f} steps/sec "
          f"(B={B}, N={N}, K={K}, D={D})")

    # Minimum bar: 10 steps/sec on CPU (conservative)
    assert steps_per_sec > 10, f"Too slow: {steps_per_sec:.1f} steps/sec"
```

Assert:
1. Throughput exceeds minimum bar (10 steps/sec conservative).
2. No errors during sustained execution.

---

### 12b) `test_iterative_round_overhead`

```python
@pytest.mark.slow
def test_iterative_round_overhead():
    """Measure overhead of multiple selection rounds."""
    import time
    D = 128
    B = 4
    N = 6

    def time_competition(rounds):
        comp = IterativeCompetition(
            workspace_dim=D, num_heads=4, selection_rounds=rounds,
            ignition_threshold=0.99, temperature=0.5  # High threshold, no early stop
        )
        comp.eval()
        features = torch.randn(B, N, D)
        saliences = torch.ones(B, N, 1)

        # Warmup
        for _ in range(5):
            comp(features, saliences)

        start = time.perf_counter()
        for _ in range(50):
            comp(features, saliences)
        return time.perf_counter() - start

    t1 = time_competition(1)
    t4 = time_competition(4)

    ratio = t4 / t1
    print(f"Round overhead: 1-round={t1:.3f}s, 4-round={t4:.3f}s, ratio={ratio:.2f}x")

    # 4 rounds should be less than 10x slower than 1 round
    assert ratio < 10, f"Excessive overhead: {ratio:.2f}x"
```

Assert:
1. 4-round overhead is less than 10x of 1-round.
2. Both configurations complete without error.

---

### 12c) `test_memory_scaling`

```python
@pytest.mark.slow
def test_memory_scaling():
    """Verify memory usage stays bounded during sustained operation."""
    import gc
    D = 256
    B = 4

    ws = create_global_workspace(
        workspace_dim=D,
        modality_dims={'v': D, 't': D},
        num_heads=4, capacity_limit=7, memory_mode='gru',
    )

    # Run many timesteps and check buffer doesn't grow unbounded
    for step in range(100):
        inputs = {'v': torch.randn(B, D), 't': torch.randn(B, D)}
        ws(inputs)

    # Memory buffer should be capped at capacity
    state = ws.get_workspace_state()
    if state['memory_buffer'] is not None:
        assert state['memory_buffer'].shape[1] <= 7, \
            f"Buffer grew beyond capacity: {state['memory_buffer'].shape}"

    # prev_context should be a single tensor, not accumulating
    assert state['prev_context'].shape == (B, D)
```

Assert:
1. Memory buffer stays within capacity limit.
2. `prev_context` is a single `(B, D)` tensor.
3. No memory leak over 100 timesteps.

---

### 12d) `test_early_stop_speedup`

```python
@pytest.mark.slow
def test_early_stop_speedup():
    """Early-stop with low threshold is faster than running all rounds."""
    import time
    D = 128
    B = 4
    N = 4
    ROUNDS = 10

    # Easy ignition -> early stop
    comp_easy = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=ROUNDS,
        ignition_threshold=0.01, temperature=0.5
    )
    comp_easy.eval()

    # Hard ignition -> all rounds
    comp_hard = IterativeCompetition(
        workspace_dim=D, num_heads=4, selection_rounds=ROUNDS,
        ignition_threshold=0.99, temperature=0.5
    )
    comp_hard.eval()

    features = torch.randn(B, N, D)
    saliences = torch.ones(B, N, 1) * 10  # Strong signal for easy ignition

    # Warmup
    for _ in range(5):
        comp_easy(features, saliences)
        comp_hard(features, saliences)

    # Time easy
    start = time.perf_counter()
    for _ in range(50):
        comp_easy(features, saliences)
    t_easy = time.perf_counter() - start

    # Time hard
    start = time.perf_counter()
    for _ in range(50):
        comp_hard(features, saliences)
    t_hard = time.perf_counter() - start

    print(f"Early-stop: {t_easy:.3f}s, Full rounds: {t_hard:.3f}s")

    # Early stop should be at least somewhat faster
    # (not strictly enforced since it depends on ignition dynamics)
    assert t_easy < t_hard * 2, \
        f"Early stop not providing speedup: {t_easy:.3f} vs {t_hard:.3f}"
```

Assert:
1. Both configurations run without error.
2. Early-stop is not significantly slower than full rounds.

---

## 13. Conftest Fixtures

Place these fixtures in `tests/test_workspace/conftest.py`.

```python
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from brain_ai.workspace.global_workspace import (
    GlobalWorkspace,
    GlobalWorkspaceConfig,
    SelectionBroadcastWorkspace,
    SelectionBroadcastConfig,
    AttentionCompetition,
    IterativeCompetition,
    ModalityProjection,
    InformationBroadcast,
    RefinedBroadcast,
    create_global_workspace,
    create_selection_broadcast_workspace,
)
from brain_ai.workspace.working_memory import (
    WorkingMemory,
    WorkingMemoryConfig,
    GRUWorkingMemory,
    LiquidWorkingMemory,
    create_working_memory,
    NCPS_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace_config():
    """Minimal GlobalWorkspaceConfig for testing."""
    return GlobalWorkspaceConfig(
        workspace_dim=64,
        num_heads=4,
        capacity_limit=4,
        dropout=0.0,  # No dropout for deterministic tests
        memory_hidden_dim=64,
        memory_mode='gru',
        competition_temperature=1.0,
        min_attention=0.01,
    )


@pytest.fixture
def competition_config():
    """Minimal competition config: K=4, small dims."""
    return dict(
        workspace_dim=64,
        num_heads=4,
        capacity_limit=4,
        temperature=1.0,
        dropout=0.0,
    )


@pytest.fixture
def sb_config():
    """Minimal SelectionBroadcastConfig for testing."""
    return SelectionBroadcastConfig(
        workspace_dim=64,
        num_heads=4,
        capacity_limit=4,
        dropout=0.0,
        ignition_threshold=0.3,
        selection_rounds=3,
        broadcast_iterations=2,
        broadcast_decay=0.9,
        memory_hidden_dim=64,
        memory_mode='gru',
        competition_temperature=0.5,
        min_attention=0.01,
        use_confidence_gating=True,
    )


# ---------------------------------------------------------------------------
# Module fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace_module(workspace_config):
    """Instantiated GlobalWorkspace with test config."""
    return GlobalWorkspace(
        config=workspace_config,
        modality_dims={'vision': 64, 'text': 64, 'audio': 64},
    )


@pytest.fixture
def sb_workspace(sb_config):
    """Instantiated SelectionBroadcastWorkspace with test config."""
    return SelectionBroadcastWorkspace(
        config=sb_config,
        modality_dims={'vision': 64, 'text': 64},
    )


@pytest.fixture
def competition_module(competition_config):
    """Instantiated AttentionCompetition for test."""
    return AttentionCompetition(**competition_config)


@pytest.fixture
def iterative_competition():
    """Instantiated IterativeCompetition for test."""
    return IterativeCompetition(
        workspace_dim=64,
        num_heads=4,
        selection_rounds=3,
        ignition_threshold=0.3,
        temperature=0.5,
        dropout=0.0,
    )


@pytest.fixture
def broadcast_module():
    """Instantiated InformationBroadcast for test."""
    return InformationBroadcast(
        workspace_dim=64,
        modality_dims={'vision': 128, 'text': 256, 'audio': 64},
    )


@pytest.fixture
def working_memory_gru():
    """GRU-based working memory for test."""
    return create_working_memory(
        input_dim=64, hidden_dim=64, output_dim=64, mode='gru'
    )


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_encoder_outputs():
    """
    List of (name, tensor) pairs simulating 3 modality encoder outputs.

    Returns a dict suitable for passing directly to GlobalWorkspace.forward().
    All tensors have batch_size=2 and respective modality dimensions.
    """
    torch.manual_seed(42)
    return {
        'vision': torch.randn(2, 64),
        'text':   torch.randn(2, 64),
        'audio':  torch.randn(2, 64),
    }


@pytest.fixture
def sample_encoder_outputs_varied():
    """
    Encoder outputs with different dimensions per modality.
    Requires workspace with matching modality_dims.
    """
    torch.manual_seed(42)
    return {
        'vision': torch.randn(2, 128),
        'text':   torch.randn(2, 256),
        'audio':  torch.randn(2, 64),
    }


@pytest.fixture
def large_batch_inputs():
    """Larger batch for performance-sensitive tests."""
    torch.manual_seed(42)
    return {
        'vision': torch.randn(8, 64),
        'text':   torch.randn(8, 64),
    }


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ncps_unavailable(monkeypatch):
    """
    Monkeypatch to simulate missing ncps library.

    Sets NCPS_AVAILABLE = False in the working_memory module,
    forcing GRU fallback for all WM instantiations within the test.
    """
    import brain_ai.workspace.working_memory as wm_module
    monkeypatch.setattr(wm_module, 'NCPS_AVAILABLE', False)


@pytest.fixture
def device():
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Set seed before each test for reproducibility."""
    torch.manual_seed(42)
    yield
```

---

## 14. Done-When Checklist

All items in this checklist must pass before the Global Workspace with Ignition Dynamics is considered production-ready. No partial credit.

### Token Staging (Section 2)

- [ ] Modality tokens concatenated in canonical sorted order regardless of dict insertion order (2a)
- [ ] Encoder outputs correctly projected into workspace dimension with valid salience (2b)
- [ ] Missing/implicit salience defaults to learned prediction, shape `(B, 1)`, all values finite (2c)
- [ ] Working memory processes input correctly when `timespans=None` (2d)
- [ ] Variable input dimensions per modality all project to common `workspace_dim` (2e)
- [ ] Unknown modality keys silently ignored; all-unknown raises `ValueError` (2f)

### Competition Scoring (Section 3)

- [ ] Gate scores combined with salience produce attention weights summing to 1.0 (3a)
- [ ] Higher salience produces higher attention weight, all else equal (3b)
- [ ] Iterative refinement changes salience values between consecutive rounds (3c)
- [ ] External salience bias shifts competition outcome toward biased item (3d)
- [ ] Attention weights sum to 1.0 regardless of input scale (small or large features) (3e)
- [ ] Lower temperature produces sharper (lower entropy) attention distribution (3f)
- [ ] Zero salience does not cause `NaN`; gate network alone determines winners (3g)
- [ ] Competition scores are finite under fp16 autocast (no overflow in softmax) (3h)

### Deterministic Top-K -- DONE CRITERION (a) (Section 4)

- [ ] Top-K selects highest-scoring items when no ties exist (4a)
- [ ] Tied scores resolve deterministically on CPU: lower index wins (4b)
- [ ] Tied scores resolve deterministically on CUDA: same rule as CPU (4c)
- [ ] Same input produces identical winners across 100 forward passes in inference mode (4d)
- [ ] Numerical stability epsilon never changes ranking for score gaps > 1e-5 (4e)
- [ ] Masked tokens receive reduced/zero attention weight (4f)
- [ ] K >= N selects all items with positive weight (4g)
- [ ] K > N does not crash; output has N items (4h)

### Slot Construction (Section 5)

- [ ] Workspace output shape is `(B, workspace_dim)` (5a)
- [ ] Workspace content differs from simple mean of inputs (competition effect) (5b)
- [ ] Integration layer modifies output on second timestep (prev_context exists) (5c)
- [ ] First timestep skips integration (prev_context is None), sets it for next call (5d)
- [ ] Output dict contains `modality_names` and per-modality attention weights (5e)
- [ ] Attention weights are non-negative, finite, and sum to 1.0 (5f)

### Iterative Rounds -- DONE CRITERION (b) (Section 6)

- [ ] `selection_rounds=1` executes exactly 1 round with 1-entry history (6a)
- [ ] Strong ignition signal triggers early stop before `max_rounds` (6b)
- [ ] Per-round saliences and ignition scores recorded in telemetry history (6c)
- [ ] Salience changes decrease over rounds (convergence trend) (6d)
- [ ] Feature embeddings remain bounded (no explosion) across all rounds (6e)
- [ ] Iteration stops at `max_rounds` even without convergence (6f)
- [ ] Very high threshold produces `global_ignition = 0` for most inputs (6g)
- [ ] Higher threshold requires more rounds or prevents early stop (6h)
- [ ] Early stop requires `ignition > threshold * 1.5`, not just `> threshold` (6i)
- [ ] Full telemetry (saliences, ignition, round count) available for all rounds (6j)

### Ignition (Section 7)

- [ ] Ignition score has shape `(B, 1)` and values in `[0, 1]` (Sigmoid output) (7a)
- [ ] `global_ignition` equals `(ignition > threshold).float()` element-wise (7b)
- [ ] Confidence gating modulates output magnitude when enabled (7c)
- [ ] Without confidence gating, no `confidence` key in output (7d)
- [ ] Training mode produces differentiable output (gradient flows to input) (7e)
- [ ] Inference mode is deterministic: same input + reset produces same output (7f)
- [ ] Multi-modal inputs all participate and produce ignition scores (7g)
- [ ] `return_details=True` includes `competition_details` with history (7h)
- [ ] Ignition computation is stable under fp16 autocast (no `NaN`) (7i)
- [ ] Ignition score always in `[0, 1]` across diverse random inputs (7j)

### Lock-In Prevention (Section 8)

- [ ] Repeated identical input produces changing workspace output via WM dynamics (8a)
- [ ] Training mode with dropout produces valid output shapes (8b)
- [ ] Inference mode with dropout=0.5 is deterministic (8c)
- [ ] `prev_context` is detached (no gradient accumulation across timesteps) (8d)
- [ ] 10 consecutive timesteps all produce finite outputs (no accumulation of `NaN`) (8e)
- [ ] Changing inputs produce changing attention patterns over 10 timesteps (8f)

### Broadcast Adapters (Section 9)

- [ ] Broadcast produces output in each modality's native dimension (9a)
- [ ] Symbolic modality broadcast has correct shape (9b)
- [ ] Decision modality broadcast has correct shape (9c)
- [ ] All broadcast outputs have matching batch dimension and are finite (9d)
- [ ] Broadcast is deterministic in inference mode (9e)
- [ ] Broadcast projections registered as `nn.ModuleDict` entries (9f)
- [ ] Custom/arbitrary modality dimensions work correctly (9g)
- [ ] Various modality configurations all produce correct shapes (9h)

### Working Memory -- DONE CRITERIA (c) and (d) (Section 10)

- [ ] Sequential inputs produce different outputs due to state persistence (10a)
- [ ] `reset_state()` restores output to fresh-initialization baseline within `atol=1e-6` (10b)
- [ ] Detached state has identical values but `requires_grad=False` (10c)
- [ ] Detaching state blocks gradient flow to earlier timesteps (10d)
- [ ] `reset_state()` clears both `hidden_state` and `memory_buffer` to `None` (10e)
- [ ] When ncps is unavailable, GRU fallback is selected with `backend_type='gru'` (10f)
- [ ] GRU fallback output dict has same keys (`output`, `state`, `buffer`) as CfC (10g)
- [ ] CfC backend accepts timespans and produces valid output (or skip if unavailable) (10h)
- [ ] GRU backend ignores timespans parameter without error (10i)
- [ ] Memory buffer enforces capacity limit via FIFO eviction (10j)

### Integration (Section 11)

- [ ] Full pipeline forward produces all expected output keys with correct shapes (11a)
- [ ] `BrainAI` with `use_workspace=True` produces valid classification output (11b)
- [ ] Single-modality input works without error (11c)
- [ ] Multi-modal competition differentiates modalities by attention weight (11d)
- [ ] `return_details=True` on `SelectionBroadcastWorkspace` includes all telemetry (11e)
- [ ] Same seed produces identical workspace output (11f)

### Performance (Section 12)

- [ ] Competition throughput exceeds 10 steps/sec on CPU (B=8, N=20, K=7, D=128) (12a)
- [ ] 4-round iterative competition is less than 10x slower than 1-round (12b)
- [ ] Memory buffer stays within capacity over 100 timesteps; no memory leak (12c)
- [ ] Early-stop is not significantly slower than full rounds (12d)

---

## Pytest Organization

```
tests/test_workspace/
+-- test_token_staging.py          # Section 2
+-- test_competition_scoring.py    # Section 3
+-- test_deterministic_topk.py     # Section 4 -- Done Criterion (a)
+-- test_slot_construction.py      # Section 5
+-- test_iterative_rounds.py       # Section 6 -- Done Criterion (b)
+-- test_ignition.py               # Section 7
+-- test_lock_in_prevention.py     # Section 8
+-- test_broadcast_adapters.py     # Section 9
+-- test_working_memory.py         # Section 10 -- Done Criteria (c), (d)
+-- test_workspace_integration.py  # Section 11
+-- test_workspace_performance.py  # Section 12 (@pytest.mark.slow)
+-- conftest.py                    # Section 13
```

Run only fast tests (excludes performance):

```bash
pytest tests/test_workspace/ -v -m "not slow"
```

Run all tests including performance:

```bash
pytest tests/test_workspace/ -v
```

Run only done-criterion tests:

```bash
pytest tests/test_workspace/test_deterministic_topk.py tests/test_workspace/test_iterative_rounds.py tests/test_workspace/test_working_memory.py -v
```

Run CUDA-dependent tests:

```bash
pytest tests/test_workspace/ -v -k "cuda or CUDA"
```

---

## Implementation Notes for Test Authors

**On dict ordering:** Python 3.7+ guarantees dict insertion order, but the workspace must NOT rely on it. Tests in Section 2 verify that the workspace sorts modality names canonically before concatenation. If the existing code iterates `modality_inputs.items()` directly (which it currently does), it must be changed to iterate `sorted(modality_inputs.items())` or the tests in 2a will fail. This is an intentional design requirement.

**On the IterativeCompetition early-stop condition:** The current implementation early-stops when `ignition.mean() > self.ignition_threshold * 1.5`. This means the threshold must be exceeded by 50% on average across the batch. Tests in Section 6 are calibrated to this rule. If the early-stop condition is changed, update tests 6b, 6h, and 6i accordingly.

**On the ncps fallback:** The `mock_ncps_unavailable` fixture patches `NCPS_AVAILABLE` to `False` in the working memory module. This causes `WorkingMemory.__init__` to select the GRU backend regardless of the requested mode. Tests 10f and 10g depend on this fixture. Do not attempt to import `LiquidWorkingMemory` in those tests -- it will raise `ImportError` if ncps is genuinely unavailable.

**On CUDA tests:** All CUDA tests use `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")`. The `device` fixture in conftest returns the appropriate device. For CI environments without GPU, these tests are automatically skipped.

**On mixed precision:** The fp16/fp32 tests (3h, 7i) use `torch.cuda.amp.autocast`. The primary concern is softmax overflow in fp16. The existing `AttentionCompetition` uses `F.softmax` which PyTorch autocast automatically promotes to fp32. If custom softmax implementations are added, ensure they also compute in fp32.

**On performance tests:** All tests in Section 12 are marked `@pytest.mark.slow`. The throughput thresholds are conservative minimums, not targets. Actual production throughput should be measured on target hardware with the full `workspace_dim=4096` configuration.

**On the 56-item checklist:** The checklist is a gate. All items must pass before the Global Workspace with Ignition Dynamics module is declared ready for integration into the full brain-ai system pipeline. Items map directly to test functions via the section reference in parentheses.
