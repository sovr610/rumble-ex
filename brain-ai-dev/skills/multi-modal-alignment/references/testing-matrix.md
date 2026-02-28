# Testing Matrix for Multi-Modal Alignment

> Complete test specification for the multi-modal alignment skill.
> Covers projection heads, contrastive losses, retrieval, metrics,
> configuration, and integration with the Global Workspace pipeline.

---

## 1. Test Categories

| Category | Class | Tests | Focus |
|----------|-------|-------|-------|
| Projection Heads | `TestProjectionHeads` | 8 | Shape, norm, gradient, modality isolation |
| Contrastive Losses | `TestContrastiveLosses` | 10 | InfoNCE, SigLIP, numerical stability, temperature |
| Uniformity & Gap | `TestUniformityGap` | 8 | Uniformity loss, gap regularization, centering |
| Cross-Modal Retrieval | `TestRetrieval` | 8 | Recall@K, batch retrieval, ranking |
| Alignment Metrics | `TestAlignmentMetrics` | 8 | Gap measurement, SVD health, uniformity score |
| Configuration | `TestConfig` | 10 | Validation, presets, serialization |
| ModalityAligner | `TestModalityAligner` | 10 | End-to-end, multi-modality, gradient flow |
| Integration | `TestIntegration` | 8 | Workspace compatibility, binding, competition |
| **Total** | **8 classes** | **~70** | |

---

## 2. Projection Head Tests

### TestProjectionHeads

| ID | Name | Description | Assert |
|----|------|-------------|--------|
| P01 | linear_output_shape | Linear projector: (B, D_enc) -> (B, D_align) | shape == (B, D_align) |
| P02 | mlp1_output_shape | MLP-1 projector: (B, D_enc) -> (B, D_align) | shape == (B, D_align) |
| P03 | mlp2_output_shape | MLP-2 projector: (B, D_enc) -> (B, D_align) | shape == (B, D_align) |
| P04 | output_l2_normalized | All projectors output unit-norm vectors | `norm(z) ~= 1.0` within eps |
| P05 | gradient_flows | Gradient from loss reaches projector inputs | `x.grad is not None` |
| P06 | different_modalities_separate | Separate projectors give different outputs for same input | `not allclose(z_a, z_b)` |
| P07 | dropout_active_training | Dropout changes output in training mode | outputs differ across calls |
| P08 | dropout_inactive_not_training | No randomness when not training | outputs identical across calls |

---

## 3. Contrastive Loss Tests

### TestContrastiveLosses

| ID | Name | Description | Assert |
|----|------|-------------|--------|
| L01 | infonce_positive_loss | InfoNCE with random embeddings > 0 | loss > 0 |
| L02 | infonce_perfect_alignment | InfoNCE with identical pairs = 0 (modulo temp) | loss < 0.01 |
| L03 | infonce_symmetric | L(a,b) == L(b,a) | allclose |
| L04 | infonce_batch_scaling | Loss changes with batch size | loss_B32 != loss_B64 |
| L05 | siglip_positive_loss | SigLIP with random embeddings > 0 | loss > 0 |
| L06 | siglip_perfect_alignment | SigLIP with matched pairs near 0 | loss < 0.1 |
| L07 | temperature_effect | Lower temperature increases loss magnitude | loss_low_t > loss_high_t |
| L08 | learnable_temperature_gradient | Gradient flows to temperature parameter | temp.grad is not None |
| L09 | fp32_enforcement | Loss computed in fp32 with fp16 inputs | loss.dtype == float32 |
| L10 | no_nan_gradient | No NaN in gradients after backward | not isnan(grad) |

---

## 4. Uniformity & Gap Tests

### TestUniformityGap

| ID | Name | Description | Assert |
|----|------|-------------|--------|
| U01 | uniformity_loss_negative | Uniformity loss is negative for spread embeddings | loss < 0 |
| U02 | uniformity_worse_for_collapsed | Collapsed embeddings have higher uniformity loss | loss_collapsed > loss_spread |
| U03 | gap_reg_zero_same_dist | Gap reg = 0 when centroids identical | loss ~= 0 |
| U04 | gap_reg_positive_different | Gap reg > 0 when centroids differ | loss > 0 |
| U05 | centering_reduces_mean | After centering, mean is near zero | `mean(centered) ~= 0` |
| U06 | centering_ema_updates | EMA centroid updates across batches | centroid changes |
| U07 | combined_loss_gradient | Combined loss gradients are non-zero | all grads nonzero |
| U08 | gap_reg_weight_scales | Larger weight = larger gap reg contribution | proportional scaling |

---

## 5. Cross-Modal Retrieval Tests

### TestRetrieval

| ID | Name | Description | Assert |
|----|------|-------------|--------|
| R01 | recall_at_1_perfect | Identical embeddings -> R@1 = 1.0 | recall == 1.0 |
| R02 | recall_at_1_random | Random embeddings -> R@1 ~= 1/B | recall ~= 1/B |
| R03 | recall_at_k_monotonic | R@1 <= R@5 <= R@10 | monotonically non-decreasing |
| R04 | bidirectional_retrieval | a->b and b->a computed separately | both returned |
| R05 | batch_size_invariance | Retrieval works for B=1 to B=128 | no errors |
| R06 | similarity_matrix_shape | Similarity matrix is (B, B) | shape == (B, B) |
| R07 | similarity_matrix_symmetric | sim(a,b).T ~= sim(b,a) | allclose with transpose |
| R08 | retrieval_with_masks | Masked tokens handled correctly | no errors with partial masks |

---

## 6. Alignment Metrics Tests

### TestAlignmentMetrics

| ID | Name | Description | Assert |
|----|------|-------------|--------|
| M01 | gap_measurement_zero | Same distribution -> gap ~= 0 | gap < eps |
| M02 | gap_measurement_positive | Different distributions -> gap > 0 | gap > 0 |
| M03 | uniformity_score_range | Uniformity in expected range | -10 < uniformity < 0 |
| M04 | svd_ratio_healthy | Well-spread embeddings -> ratio > 0.01 | ratio > 0.01 |
| M05 | svd_ratio_collapsed | Collapsed embeddings -> ratio ~= 0 | ratio < 0.01 |
| M06 | retrieval_recall_range | Recall in [0, 1] | 0 <= recall <= 1 |
| M07 | all_metrics_computed | compute_all returns all expected keys | all keys present |
| M08 | metrics_no_nan | No NaN in any metric | not isnan |

---

## 7. Configuration Tests

### TestConfig

| ID | Name | Description | Assert |
|----|------|-------------|--------|
| C01 | default_creation | AlignmentConfig() creates valid config | no error |
| C02 | minimal_preset | AlignmentConfig.minimal() validates | no error |
| C03 | dev_preset | AlignmentConfig.dev() validates | no error |
| C04 | production_preset | AlignmentConfig.production() validates | no error |
| C05 | rejects_bad_temperature | temperature <= 0 raises ValueError | ValueError |
| C06 | rejects_bad_loss_type | Unknown loss_type raises ValueError | ValueError |
| C07 | rejects_bad_projector | Unknown projector_type raises ValueError | ValueError |
| C08 | serialization_roundtrip | to_dict -> from_dict preserves all fields | dict equality |
| C09 | alignment_dim_positive | alignment_dim <= 0 raises ValueError | ValueError |
| C10 | weight_non_negative | Negative loss weights raise ValueError | ValueError |

---

## 8. ModalityAligner End-to-End Tests

### TestModalityAligner

| ID | Name | Description | Assert |
|----|------|-------------|--------|
| A01 | forward_basic | Two modalities produce AlignmentOutput | output has all fields |
| A02 | proj_shapes | proj_a and proj_b have correct shape | (B, D_align) |
| A03 | similarity_shape | Similarity matrix is (B, B) | shape check |
| A04 | loss_scalar | Loss is scalar tensor | loss.ndim == 0 |
| A05 | loss_positive | Loss > 0 for random inputs | loss > 0 |
| A06 | gradient_to_inputs | Gradients flow to input features | feats.grad is not None |
| A07 | multiple_modalities | Register and align 5 modality pairs | no error |
| A08 | not_training_mode_deterministic | Same input -> same output when not training | allclose |
| A09 | train_mode_with_module_flag | Uses module.train(False) not .eval() for deterministic mode | self-test pattern |
| A10 | masked_inputs | Handles variable-length masks | no error, valid output |

---

## 9. Integration Tests

### TestIntegration

| ID | Name | Description | Assert |
|----|------|-------------|--------|
| I01 | workspace_compatible_output | Aligned features have workspace_dim | shape[-1] == workspace_dim |
| I02 | encoder_output_integration | AlignmentOutput feeds into workspace | no error |
| I03 | multi_modality_alignment | 3+ modalities aligned simultaneously | all pairs have loss |
| I04 | binding_map_produced | Cross-attention produces binding maps | maps is not None |
| I05 | alignment_before_competition | Aligned tokens score differently | scores differ |
| I06 | end_to_end_gradient | Gradient from workspace loss reaches projectors | grad is not None |
| I07 | batch_size_one | Handles B=1 without error | no error |
| I08 | empty_mask_handling | All-False mask handled gracefully | no error or NaN |

---

## 10. Done-When Checklist

- [ ] All 70+ tests pass with `pytest -x`
- [ ] Self-tests in all 5 asset templates pass (`python *_template.py`)
- [ ] `scripts/validate_alignment.py` reports all contracts satisfied
- [ ] Contrastive loss < 2.0 for B >= 64 after 1K synthetic steps
- [ ] Modality gap < 0.3 after convergence
- [ ] Cross-modal R@1 > 50% on synthetic matched pairs
- [ ] No NaN or Inf in any metric or gradient
- [ ] All projection heads output L2-normalized vectors
- [ ] Temperature stays in [0.001, 1.0] when learnable
- [ ] fp32 precision used for loss computation under AMP
- [ ] Configuration validation rejects all invalid inputs
- [ ] Serialization roundtrip preserves all config fields

---

## 11. Pytest Conventions

```python
# Marker for slow tests
@pytest.mark.slow
def test_large_batch_alignment():
    ...

# Parametrize across projector types
@pytest.mark.parametrize("proj_type", ["linear", "mlp_1", "mlp_2"])
def test_projector_output_shape(proj_type):
    ...

# Parametrize across loss types
@pytest.mark.parametrize("loss_type", ["infonce", "siglip"])
def test_loss_positive(loss_type):
    ...

# Test device handling
@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return torch.device(request.param)
```

---

## 12. Self-Test Pattern

Every asset template includes a `_self_test()` function runnable as:

```bash
python modality_aligner_template.py
python contrastive_losses_template.py
python cross_modal_retriever_template.py
python alignment_metrics_template.py
python alignment_config_template.py
```

Self-tests use:
- `module.train(False)` instead of `.eval()` for deterministic mode
- CPU-only, tiny dimensions (D=32 to 64)
- `torch.manual_seed(42)` for reproducibility
- `_check(name, condition, detail)` helper for consistent output
