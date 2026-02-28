# Testing Matrix — Reference for Data Pipeline & Loaders Skill

This document specifies the test scenarios, verification strategies, and expected outcomes for all data pipeline components. Use this as the canonical reference when writing or reviewing tests for loaders, augmentation, preprocessing, splits, and the dataset registry.

---

## 1. Loader Contract Compliance Tests

These tests verify that every phase-specific loader correctly implements the `BasePhaseLoader` interface.

### 1.1 Interface Implementation

| Test ID | Test Name | Description | Pass Criteria |
|---|---|---|---|
| LC-01 | `test_inherits_base` | Each phase loader inherits from `BasePhaseLoader` | `issubclass(PhaseLoader, BasePhaseLoader)` is True |
| LC-02 | `test_has_abstract_methods` | All abstract methods are implemented | Instantiation does not raise `TypeError` |
| LC-03 | `test_constructor_accepts_config` | Constructor accepts `DataConfig` and `mode` | No exception on `PhaseLoader(config, "dev")` |
| LC-04 | `test_constructor_rejects_bad_mode` | Constructor raises on invalid mode | `ValueError` on `PhaseLoader(config, "invalid")` |
| LC-05 | `test_get_train_loader_returns_dataloader` | `get_train_loader()` returns a DataLoader | `isinstance(result, DataLoader)` |
| LC-06 | `test_get_val_loader_returns_dataloader` | `get_val_loader()` returns a DataLoader | `isinstance(result, DataLoader)` |
| LC-07 | `test_get_test_loader_returns_dataloader` | `get_test_loader()` returns a DataLoader | `isinstance(result, DataLoader)` |
| LC-08 | `test_get_dataset_info_returns_info` | `get_dataset_info()` returns DatasetInfo | `isinstance(result, DatasetInfo)` |
| LC-09 | `test_get_sample_shape_returns_dict` | `get_sample_shape()` returns dict of tuples | All values are tuples of ints |
| LC-10 | `test_train_loader_shuffles` | Train loader shuffles data | Different iteration orders across two epochs |
| LC-11 | `test_val_loader_no_shuffle` | Val loader does not shuffle | Identical iteration order across two passes |
| LC-12 | `test_train_loader_drop_last` | Train loader drops incomplete last batch | All batches have same size |
| LC-13 | `test_val_loader_no_drop_last` | Val loader keeps all samples | Total samples across batches equals dataset size |

### 1.2 All-Phase Parametrized Tests

These tests run for each of the 7 phase loaders (parametrized with `@pytest.mark.parametrize`):

| Test ID | Test Name | Pass Criteria |
|---|---|---|
| LC-20 | `test_phase_N_train_batch_shape` | First batch shape matches get_sample_shape() |
| LC-21 | `test_phase_N_val_batch_shape` | Val batch shape matches train batch shape |
| LC-22 | `test_phase_N_test_batch_shape` | Test batch shape matches train batch shape |
| LC-23 | `test_phase_N_batch_size` | Batch dimension equals config.batch_size |
| LC-24 | `test_phase_N_dtypes` | All tensors have correct dtypes |
| LC-25 | `test_phase_N_no_nan` | No NaN values in any batch tensor |
| LC-26 | `test_phase_N_no_inf` | No Inf values in any batch tensor |
| LC-27 | `test_phase_N_dev_mode_fast` | Dev mode instantiation takes < 5 seconds |
| LC-28 | `test_phase_N_info_complete` | DatasetInfo has all required fields non-None |

---

## 2. Shape Verification Tests

These tests verify that output tensors match the per-modality shape contracts defined in `loader-contracts.md`.

### 2.1 Vision Shape Tests

| Test ID | Test Name | Expected Shape | dtype |
|---|---|---|---|
| SV-01 | `test_vision_dev_input_shape` | `[B, 1, 28, 28]` | `float32` |
| SV-02 | `test_vision_dev_target_shape` | `[B]` | `int64` |
| SV-03 | `test_vision_dev_value_range` | All values in `[0, 1]` or `[-1, 1]` | - |
| SV-04 | `test_vision_prod_input_shape` | `[B, 3, H, H]` where H in {224, 384} | `float32` |
| SV-05 | `test_vision_channels_first` | `dim 1` is channel dim (1 or 3) | - |

### 2.2 Text Shape Tests

| Test ID | Test Name | Expected Shape | dtype |
|---|---|---|---|
| ST-01 | `test_text_dev_input_shape` | `[B, 128]` | `int64` |
| ST-02 | `test_text_dev_mask_shape` | `[B, 128]` | `bool` |
| ST-03 | `test_text_dev_value_range` | All token IDs in `[0, vocab_size)` | - |
| ST-04 | `test_text_padding_convention` | Padding uses token ID 0 | - |
| ST-05 | `test_text_mask_consistency` | Mask is True where tokens are non-zero | - |

### 2.3 Audio Shape Tests

| Test ID | Test Name | Expected Shape | dtype |
|---|---|---|---|
| SA-01 | `test_audio_dev_input_shape` | `[B, 64, T]` where T > 0 | `float32` |
| SA-02 | `test_audio_dev_target_shape` | `[B]` | `int64` |
| SA-03 | `test_audio_mel_bins` | dim 1 equals n_mels config value | - |

### 2.4 Sequence Shape Tests

| Test ID | Test Name | Expected Shape | dtype |
|---|---|---|---|
| SS-01 | `test_sequence_dev_input_shape` | `[B, T, D]` | `float32` |
| SS-02 | `test_sequence_dev_mask_shape` | `[B, T]` | `bool` |
| SS-03 | `test_sequence_mask_valid` | At least one True per sample | - |
| SS-04 | `test_sequence_target_shape` | Matches documented contract | - |

### 2.5 RL Episode Shape Tests

| Test ID | Test Name | Expected Shape | dtype |
|---|---|---|---|
| SR-01 | `test_rl_state_shape` | `[B, state_dim]` | `float32` |
| SR-02 | `test_rl_action_shape` | `[B]` or `[B, action_dim]` | `int64` or `float32` |
| SR-03 | `test_rl_reward_shape` | `[B]` | `float32` |
| SR-04 | `test_rl_next_state_shape` | `[B, state_dim]` | `float32` |
| SR-05 | `test_rl_done_shape` | `[B]` | `float32` |
| SR-06 | `test_rl_done_values` | All values in `{0.0, 1.0}` | - |

### 2.6 Meta-Learning Episode Shape Tests

| Test ID | Test Name | Expected Shape | dtype |
|---|---|---|---|
| SM-01 | `test_meta_support_x_shape` | `[B, N*K, C, H, W]` | `float32` |
| SM-02 | `test_meta_support_y_shape` | `[B, N*K]` | `int64` |
| SM-03 | `test_meta_query_x_shape` | `[B, N*Q, C, H, W]` | `float32` |
| SM-04 | `test_meta_query_y_shape` | `[B, N*Q]` | `int64` |
| SM-05 | `test_meta_class_range` | All labels in `[0, N)` | - |

---

## 3. Augmentation Determinism Tests

These tests verify that augmentation produces identical results when seeded, and that evaluation transforms are always deterministic.

### 3.1 Seeded Reproducibility

| Test ID | Test Name | Description | Pass Criteria |
|---|---|---|---|
| AD-01 | `test_vision_aug_seed_determinism` | Same seed, same input yields same output | `torch.allclose(y1, y2)` |
| AD-02 | `test_text_aug_seed_determinism` | Same seed, same tokens yields same output | `torch.equal(y1, y2)` |
| AD-03 | `test_audio_aug_seed_determinism` | Same seed, same spectrogram yields same output | `torch.allclose(y1, y2)` |
| AD-04 | `test_sequence_aug_seed_determinism` | Same seed, same sequence yields same output | `torch.allclose(y1, y2)` |
| AD-05 | `test_different_seeds_different_output` | Different seeds produce different output | `not torch.equal(y1, y2)` |

### 3.2 Eval Transform Determinism

| Test ID | Test Name | Description | Pass Criteria |
|---|---|---|---|
| AD-10 | `test_eval_vision_deterministic` | Eval transform is deterministic (no seed needed) | Identical output across 10 calls |
| AD-11 | `test_eval_text_deterministic` | Eval transform is deterministic | Identical output across 10 calls |
| AD-12 | `test_eval_audio_deterministic` | Eval transform is deterministic | Identical output across 10 calls |
| AD-13 | `test_eval_sequence_deterministic` | Eval transform is deterministic | Identical output across 10 calls |

### 3.3 Augmentation Properties

| Test ID | Test Name | Description | Pass Criteria |
|---|---|---|---|
| AD-20 | `test_aug_preserves_shape` | Augmented tensor has same shape as input | Shape equality |
| AD-21 | `test_aug_preserves_dtype` | Augmented tensor has same dtype | dtype equality |
| AD-22 | `test_none_strength_no_change` | Strength "none" produces identity transform | `torch.allclose(input, output)` |
| AD-23 | `test_heavy_strength_modifies` | Strength "heavy" changes most inputs | Output differs from input on most samples |
| AD-24 | `test_text_aug_preserves_padding` | Text augmentation does not modify padding tokens | Padded positions remain 0 |
| AD-25 | `test_text_aug_valid_token_range` | All augmented tokens in valid range | `0 <= tokens < vocab_size` |

---

## 4. Split Non-Overlap Tests

These tests verify that train/val/test splits have zero index overlap and cover the full dataset.

### 4.1 Basic Split Properties

| Test ID | Test Name | Description | Pass Criteria |
|---|---|---|---|
| SN-01 | `test_split_no_overlap_train_val` | Train and val indices do not overlap | Empty intersection |
| SN-02 | `test_split_no_overlap_train_test` | Train and test indices do not overlap | Empty intersection |
| SN-03 | `test_split_no_overlap_val_test` | Val and test indices do not overlap | Empty intersection |
| SN-04 | `test_split_covers_all` | Union of all splits equals full dataset | Union size equals dataset size |
| SN-05 | `test_split_ratios_approximate` | Split sizes approximately match requested ratios | Within 5% of requested ratios |

### 4.2 Stratified Split Properties

| Test ID | Test Name | Description | Pass Criteria |
|---|---|---|---|
| SN-10 | `test_stratified_class_coverage` | All classes appear in all splits | Every class has >= 1 sample per split |
| SN-11 | `test_stratified_balance` | Class proportions are similar across splits | Chi-squared test p > 0.05 |
| SN-12 | `test_stratified_reproducibility` | Same seed produces same splits | Index equality |

### 4.3 Persistence Tests

| Test ID | Test Name | Description | Pass Criteria |
|---|---|---|---|
| SN-20 | `test_save_load_roundtrip` | Saved splits can be loaded identically | Loaded indices equal original |
| SN-21 | `test_load_detects_corruption` | Loading corrupted file raises error | Exception raised |
| SN-22 | `test_save_creates_file` | Save creates a file at the specified path | File exists after save |

---

## 5. Registry Round-Trip Tests

These tests verify the DatasetRegistry's registration, lookup, and lifecycle.

### 5.1 Registration Tests

| Test ID | Test Name | Description | Pass Criteria |
|---|---|---|---|
| RR-01 | `test_register_new_dataset` | Registering a new dataset succeeds | No exception |
| RR-02 | `test_register_duplicate_raises` | Duplicate registration raises error | `ValueError` raised |
| RR-03 | `test_register_invalid_phase_raises` | Phase outside [1,7] raises error | `ValueError` raised |
| RR-04 | `test_register_non_loader_raises` | Non-BasePhaseLoader class raises error | `TypeError` raised |

### 5.2 Lookup Tests

| Test ID | Test Name | Description | Pass Criteria |
|---|---|---|---|
| RR-10 | `test_get_loader_returns_instance` | `get_loader()` returns a BasePhaseLoader instance | `isinstance` check |
| RR-11 | `test_get_loader_unknown_raises` | Unknown name raises KeyError | `KeyError` raised |
| RR-12 | `test_get_loader_dev_mode` | Dev mode loader instantiates correctly | No exception, returns loader |
| RR-13 | `test_get_loader_creates_dataloader` | Returned loader produces working DataLoaders | Can iterate one batch |

### 5.3 Listing Tests

| Test ID | Test Name | Description | Pass Criteria |
|---|---|---|---|
| RR-20 | `test_list_all_datasets` | List without filter returns all datasets | Length equals total registered |
| RR-21 | `test_list_by_phase` | Phase filter returns correct subset | All results have matching phase |
| RR-22 | `test_list_by_modality` | Modality filter returns correct subset | All results have matching modality |
| RR-23 | `test_list_empty_phase` | Querying empty phase returns empty list | Empty list |

### 5.4 End-to-End Round-Trip

| Test ID | Test Name | Description | Pass Criteria |
|---|---|---|---|
| RR-30 | `test_register_get_iterate` | Register, get_loader, iterate one batch | Successfully yields tensors |
| RR-31 | `test_all_phases_have_dev_data` | Every phase (1-7) has at least one dev dataset | `len(list_datasets(phase=i)) >= 1` for i in 1..7 |
| RR-32 | `test_get_loader_mnist_dev` | Specific test: get "synthetic_snn" in dev mode | Returns working loader |

---

## 6. Preprocessing Tests

| Test ID | Test Name | Description | Pass Criteria |
|---|---|---|---|
| PP-01 | `test_normalize_image_uint8` | uint8 input normalized to [0,1] float32 | Value range and dtype check |
| PP-02 | `test_normalize_image_float` | float32 input in [0,1] unchanged | Identity transform |
| PP-03 | `test_normalize_image_shape_preserved` | Shape unchanged after normalization | Shape equality |
| PP-04 | `test_tokenize_text_output_dtype` | Tokenized text is int64 | dtype check |
| PP-05 | `test_tokenize_text_max_len` | Output length equals max_len | Shape check |
| PP-06 | `test_tokenize_text_padding` | Short text is padded with zeros | Trailing zeros |
| PP-07 | `test_compute_spectrogram_shape` | Output has [n_mels, T] shape | Shape check |
| PP-08 | `test_compute_spectrogram_no_nan` | No NaN in spectrogram output | `not torch.isnan().any()` |
| PP-09 | `test_normalize_sequence_zero_mean` | Normalized sequence has ~zero mean | `abs(mean) < 0.01` |
| PP-10 | `test_normalize_sequence_unit_std` | Normalized sequence has ~unit std | `abs(std - 1.0) < 0.01` |

---

## 7. Performance Tests

These tests verify throughput and latency constraints.

| Test ID | Test Name | Description | Pass Criteria |
|---|---|---|---|
| PF-01 | `test_dev_loader_startup_time` | Dev loader creation takes < 5 seconds | Wall time check |
| PF-02 | `test_batch_iteration_speed` | Can iterate 10 batches in < 10 seconds | Wall time check |
| PF-03 | `test_augmentation_overhead` | Augmented batch < 2x slower than no-aug | Ratio check |
| PF-04 | `test_memory_no_leak` | 100 iterations do not grow memory by > 10% | Memory check |

---

## 8. Integration Tests

| Test ID | Test Name | Description | Pass Criteria |
|---|---|---|---|
| IT-01 | `test_loader_to_model_forward` | Batch from loader feeds into brain_ai model | No shape error |
| IT-02 | `test_all_phases_sequential` | Create loaders for all 7 phases sequentially | All succeed |
| IT-03 | `test_config_propagation` | DataConfig values propagate to DataLoader | batch_size, num_workers match |

---

## 9. Test Execution Summary

| Category | Count | Priority |
|---|---|---|
| Loader Contract Compliance | ~25 tests | P0 (must pass) |
| Shape Verification | ~25 tests | P0 (must pass) |
| Augmentation Determinism | ~15 tests | P0 (must pass) |
| Split Non-Overlap | ~10 tests | P0 (must pass) |
| Registry Round-Trip | ~15 tests | P0 (must pass) |
| Preprocessing | ~10 tests | P1 (should pass) |
| Performance | ~4 tests | P2 (informational) |
| Integration | ~3 tests | P1 (should pass) |
| **Total** | **~107 tests** | |

All P0 tests must pass before any training phase can begin. P1 tests should pass for production readiness. P2 tests are informational and may have environment-dependent results.
