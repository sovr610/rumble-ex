# Testing Matrix: Task2Vec Task Embeddings + Curriculum/Clustering

This document specifies every test case for the Task2Vec skill. Tests are organized by class, each corresponding to a module in the skill. The generated test file is `tests/test_task2vec.py`, produced by `scripts/gen_task2vec_tests.py`. All tests run under `pytest` with markers for optional hardware and dependencies.

## Global Fixtures and Conventions

### Shared Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `synthetic_episode` | `function` | Returns a `TaskEpisode` with `x_support` of shape `(5, 3, 32, 32)`, `y_support` of shape `(5,)`, 5-way classification, fixed class IDs `[0,1,2,3,4]`, fixed seed 42. Deterministic random data via `torch.manual_seed`. |
| `second_episode` | `function` | Same structure as `synthetic_episode` but with seed 99 and different class IDs `[5,6,7,8,9]`. Used for cross-episode comparison tests. |
| `minimal_config` | `session` | `Task2VecFullConfig.minimal()` with `embedding_dim=64`, `probe_model="conv4"`, `n_clusters=3`. Small enough for CPU-only CI. |
| `dev_config` | `session` | `Task2VecFullConfig.dev()` with `embedding_dim=256`, `probe_model="conv4"`, `n_clusters=5`. |
| `extractor` | `function` | `Task2VecExtractor(minimal_config.task2vec)` constructed fresh per test to avoid state leakage. |
| `populated_registry` | `function` | `TaskEmbeddingRegistry` with 20 pre-inserted synthetic embeddings (10 from dataset "A", 10 from dataset "B"), each L2-normalized, dimension matching `minimal_config`. |
| `embedding_matrix` | `function` | `np.ndarray` of shape `(50, 64)` with L2-normalized rows drawn from 5 planted Gaussian clusters. Used for clustering and curriculum tests. |
| `task_ids` | `function` | List of 50 deterministic task ID strings corresponding to rows of `embedding_matrix`. |
| `tmp_registry_dir` | `function` | `tmp_path` directory for registry save/load tests. Cleaned up automatically by pytest. |

### Skip Conditions

```python
import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

try:
    import hdbscan  # noqa: F401
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

requires_hdbscan = pytest.mark.skipif(
    not HAS_HDBSCAN,
    reason="hdbscan not installed"
)
```

### Common Assertion Helpers

```python
def assert_l2_normalized(vec, tol=1e-5):
    """Assert that a vector has unit L2 norm."""
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < tol, f"L2 norm {norm} not within {tol} of 1.0"

def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## 1. TestTask2VecExtractor (~15 tests)

Tests the `Task2VecExtractor` class in `brain_ai/meta/task2vec.py`. Each test constructs a fresh extractor from `minimal_config` to prevent cross-test contamination.

### Fixtures

- `extractor`, `synthetic_episode`, `second_episode`, `minimal_config`

### Tests

| # | Test Name | Description | Assertion Pattern |
|---|---|---|---|
| 1 | `test_embedding_shape` | Extract embedding from `synthetic_episode`. Verify output `TaskEmbedding.embedding` is a 1-D numpy array with length equal to `config.embedding_dim` (64). | `assert emb.embedding.shape == (64,)` |
| 2 | `test_embedding_l2_normalized` | Extract embedding; verify L2 norm equals 1.0 within tolerance 1e-5. | `assert_l2_normalized(emb.embedding)` |
| 3 | `test_determinism_same_input` | Extract twice from the same episode with the same seed. Assert element-wise difference is below 1e-7. | `np.testing.assert_allclose(emb1.embedding, emb2.embedding, atol=1e-7)` |
| 4 | `test_cross_device_similarity` | **`@requires_cuda`**. Extract on CPU, then on CUDA. Compute cosine similarity. Assert > 0.9999. This accounts for floating-point differences across devices without requiring exact equality. | `assert cosine_sim(emb_cpu, emb_cuda) > 0.9999` |
| 5 | `test_probe_frozen` | Snapshot all probe parameter values before extraction. Run extraction. Assert no parameter has changed (compare via `torch.equal`). | `for p_before, p_after in zip(snap, probe.parameters()): assert torch.equal(p_before, p_after)` |
| 6 | `test_probe_eval_mode` | During extraction, assert `extractor.probe.training == False`. Verify dropout layers have `p=0` effective behavior and BatchNorm layers use running stats (`.training == False`). Tested by monkey-patching the forward call to capture module states. | `assert not extractor.probe.training` and all BN modules `assert not bn.training` |
| 7 | `test_different_episodes_different_embeddings` | Extract from `synthetic_episode` and `second_episode`. Assert cosine similarity is less than 0.99 (they should be distinguishable). | `assert cosine_sim(emb1, emb2) < 0.99` |
| 8 | `test_fisher_values_non_negative` | Access raw Fisher diagonal values before projection (via extractor internals or diagnostic output). Assert all values >= 0, since Fisher entries are squared gradients. | `assert (fisher_diag >= 0).all()` |
| 9 | `test_log1p_normalization` | After log1p transform of Fisher values, assert all values >= 0. `log1p(x)` for `x >= 0` is always >= 0. | `assert (log1p_fisher >= 0).all()` |
| 10 | `test_parameter_subset_last_block` | Set `layer_subset="last_block"`. Extract. Verify the raw Fisher dimension matches only the last block's parameter count, not the full model. | `assert fisher_dim == expected_last_block_params` |
| 11 | `test_parameter_subset_per_stage` | Set `layer_subset="per_stage"`. Extract. Verify Fisher dimension equals sum of per-stage parameter counts. | `assert fisher_dim == sum(stage_param_counts)` |
| 12 | `test_aggregation_per_channel` | Set `aggregation="per_channel"`. Verify final embedding dim matches config. The per-channel aggregation averages Fisher values within each output channel. | `assert emb.embedding.shape == (config.embedding_dim,)` |
| 13 | `test_ephemeral_head_sized_to_nway` | Create a 5-way episode and a 10-way episode. Verify the ephemeral classification head has output dimension matching N (5 or 10 respectively). | `assert head.out_features == n_way` |
| 14 | `test_task_id_deterministic` | Construct a task ID from `(dataset="mnist", split="train", classes=[0,1,2], indices=[10,20,30])`. Construct again with same inputs. Assert identical strings. Change one field, assert different. | `assert id1 == id2` and `assert id1 != id3` |
| 15 | `test_diagnostics_keys` | Extract an embedding and inspect `emb.diagnostics`. Assert it contains keys: `"fisher_norm"`, `"sparsity"`, `"probe_loss"`, `"extraction_time_sec"`. Assert all values are finite floats. | `for key in required_keys: assert key in diag and np.isfinite(diag[key])` |
| 16 | `test_fp32_enforcement_under_amp` | **`@requires_cuda`**. Wrap extraction in `torch.cuda.amp.autocast()`. Assert that Fisher computation still runs in fp32 (check tensor dtypes inside the extraction path). The extractor must force fp32 to maintain determinism. | `assert fisher_tensor.dtype == torch.float32` |

---

## 2. TestEmbeddingRegistry (~12 tests)

Tests the `TaskEmbeddingRegistry` class in `brain_ai/meta/embedding_registry.py`. Registry persistence uses JSONL for metadata and NPZ for embedding arrays.

### Fixtures

- `populated_registry`, `tmp_registry_dir`, `minimal_config`

### Tests

| # | Test Name | Description | Assertion Pattern |
|---|---|---|---|
| 1 | `test_save_load_roundtrip` | Save `populated_registry` to `tmp_registry_dir`. Create new registry, load from same path. Assert all 20 embeddings match element-wise within 1e-7. | `np.testing.assert_allclose(orig, loaded, atol=1e-7)` for each entry |
| 2 | `test_jsonl_format` | Save registry. Read the `.jsonl` file line by line. Assert each line is valid JSON (via `json.loads`). Assert every parsed dict contains required keys: `task_id`, `dataset`, `split`, `row_index`. | `json.loads(line)` succeeds for every line; required keys present |
| 3 | `test_npz_embeddings_key` | Save registry. Load `.npz` file. Assert `"embeddings"` key exists. Assert shape is `(20, 64)` matching 20 entries with embedding dim 64. | `assert data["embeddings"].shape == (20, 64)` |
| 4 | `test_update_registry_new_entry` | Add a new entry with a novel `task_id`. Assert `len(registry)` increases by 1. Retrieve the entry and verify embedding matches. | `assert len(registry) == 21` and retrieved embedding matches |
| 5 | `test_update_registry_overwrite` | Add an entry, then add again with the same `task_id` but different embedding. Assert `len(registry)` unchanged. Verify the stored embedding is the new one. | `assert len(registry) == 20` and `np.allclose(stored, new_emb)` |
| 6 | `test_get_embedding` | Retrieve a specific `task_id` from `populated_registry`. Assert the returned vector matches the original inserted vector exactly. | `np.testing.assert_array_equal(retrieved, original)` |
| 7 | `test_get_embeddings_batch` | Request 5 task IDs at once via `get_embeddings`. Assert returned array shape is `(5, 64)`. Assert each row matches individual `get_embedding` calls. | `assert batch.shape == (5, 64)` and per-row equality |
| 8 | `test_query_filter_dataset` | Query with `dataset="A"`. Assert exactly 10 results (matching the 10 "A" entries in `populated_registry`). | `assert len(results) == 10` and all have `dataset == "A"` |
| 9 | `test_query_filter_split` | Insert entries with `split="train"` and `split="val"`. Query with `split="val"`. Assert only val entries returned. | `all(r.split == "val" for r in results)` |
| 10 | `test_version_header` | Save registry. Read first line of JSONL. Assert it contains a `"version"` field with a semver-compatible string. | `assert "version" in first_line` and matches `r"\d+\.\d+\.\d+"` |
| 11 | `test_probe_signature_mismatch_warning` | Create registry with probe signature "conv4_last_block_v1". Attempt to add entry with probe signature "resnet12_per_stage_v1". Assert a warning is emitted (via `pytest.warns` or logging capture). | `with pytest.warns(UserWarning, match="probe_signature")` |
| 12 | `test_merge_registries` | Create two registries with 10 entries each, 5 overlapping task IDs. Merge them. Assert merged registry has 15 entries (10 + 10 - 5 deduped). Assert no duplicate task IDs. | `assert len(merged) == 15` and `len(set(ids)) == 15` |
| 13 | `test_empty_registry` | Create empty registry. Save and load. Assert `len == 0`. Query returns empty list. `get_embedding` for nonexistent ID raises `KeyError`. | `assert len(reg) == 0` and `pytest.raises(KeyError)` |
| 14 | `test_row_index_consistency` | Save registry. Parse JSONL to get `row_index` for each task. Load NPZ. For each task, assert `npz["embeddings"][row_index]` matches the embedding retrieved by `get_embedding(task_id)`. | `np.testing.assert_array_equal(npz_row, get_result)` for each entry |

---

## 3. TestClustering (~12 tests)

Tests the clustering module in `brain_ai/meta/task_clustering.py`. Uses `embedding_matrix` fixture with 50 embeddings from 5 planted clusters.

### Fixtures

- `embedding_matrix`, `task_ids`, `minimal_config`

### Tests

| # | Test Name | Description | Assertion Pattern |
|---|---|---|---|
| 1 | `test_kmeans_produces_k_clusters` | Run k-means with `n_clusters=5`, seed 42. Assert exactly 5 unique labels returned. Assert label array length equals 50. | `assert len(set(labels)) == 5` and `len(labels) == 50` |
| 2 | `test_agglomerative_deterministic` | Run agglomerative clustering twice with `linkage="ward"` and `n_clusters=5`. Assert label arrays are identical. Agglomerative with fixed linkage is inherently deterministic. | `np.testing.assert_array_equal(labels1, labels2)` |
| 3 | `test_hdbscan_optional_import` | **`@requires_hdbscan`**. Run HDBSCAN on `embedding_matrix`. Assert it returns labels (possibly including -1 for noise). If HDBSCAN is unavailable, test is skipped. | `assert isinstance(labels, np.ndarray)` and `labels.shape == (50,)` |
| 4 | `test_cosine_distance_matrix_properties` | Compute cosine distance matrix from `embedding_matrix`. Assert: (a) symmetric within 1e-7, (b) diagonal entries are zero within 1e-7, (c) all values in `[0, 2]`. | `np.testing.assert_allclose(D, D.T, atol=1e-7)` and `assert D.diagonal().max() < 1e-7` and `assert D.min() >= -1e-7 and D.max() <= 2.0 + 1e-7` |
| 5 | `test_stability_ari` | Cluster with k-means (k=5) using seed 42, then seed 123. Compute adjusted Rand index between the two label sets. Assert ARI >= 0.9. The planted cluster structure should yield high stability. | `assert adjusted_rand_score(labels1, labels2) >= 0.9` |
| 6 | `test_stability_nmi` | Same as above but compute normalized mutual information. Assert NMI >= 0.85. | `assert normalized_mutual_info_score(labels1, labels2) >= 0.85` |
| 7 | `test_silhouette_score_range` | Compute silhouette score for k-means labels on `embedding_matrix` with cosine metric. Assert value in `[-1, 1]`. For planted clusters, assert > 0 (positive indicates meaningful structure). | `assert -1 <= sil <= 1` and `assert sil > 0` |
| 8 | `test_within_vs_between_cluster_distance` | Compute mean within-cluster pairwise cosine distance. Compute mean between-cluster pairwise cosine distance. Assert within < between. | `assert mean_within < mean_between` |
| 9 | `test_representative_tasks` | For each cluster, find the task whose embedding is closest to the cluster centroid. Assert the returned task ID is valid and its embedding has the minimum distance to the centroid among cluster members. | `for cluster_id in range(k): assert rep_task in cluster_members[cluster_id]` and distance is minimal |
| 10 | `test_degenerate_single_cluster` | Run k-means with `n_clusters=1`. Assert all labels are 0. Assert no error is raised. | `assert set(labels) == {0}` |
| 11 | `test_degenerate_identical_embeddings` | Create 10 identical embeddings. Run k-means with `n_clusters=3`. Assert it completes without error. Labels may all be the same (since data is identical). | No exception raised; `labels.shape == (10,)` |
| 12 | `test_no_empty_clusters` | Run k-means with `n_clusters=5` on 50 embeddings from 5 planted clusters. Assert every cluster has at least one member. | `for c in range(5): assert (labels == c).sum() >= 1` |
| 13 | `test_diagnostics_dict` | Run clustering and retrieve diagnostics. Assert dict contains keys: `"algorithm"`, `"n_clusters"`, `"silhouette"`, `"inertia"` (for k-means), `"label_counts"`. | `for key in required: assert key in diagnostics` |

---

## 4. TestCurriculum (~15 tests)

Tests the curriculum module in `brain_ai/meta/curriculum.py`. Uses `embedding_matrix`, `task_ids`, and clustered labels from the clustering module.

### Fixtures

- `embedding_matrix`, `task_ids`, `minimal_config`
- `cluster_labels`: fixture that runs k-means on `embedding_matrix` with k=5, seed 42, returning labels

### Tests

| # | Test Name | Description | Assertion Pattern |
|---|---|---|---|
| 1 | `test_easy_to_hard_ordering` | Compute difficulty proxy (centroid distance) for each task. Get curriculum order with `strategy="easy_to_hard"`. Assert the returned order matches argsort of difficulty proxy (ascending). | `assert order == sorted(task_ids, key=lambda t: difficulty[t])` |
| 2 | `test_diversity_batch_min_distance` | Build a diversity-constrained batch of size 10 with `diversity_min_distance=0.3`. Compute all pairwise cosine distances in the batch. Assert minimum pairwise distance >= 0.3. | `assert min_pairwise_cosine_dist >= 0.3` |
| 3 | `test_anti_curriculum` | Get curriculum with `strategy="anti_curriculum"`. Get easy-to-hard order separately. Assert anti-curriculum order is the reverse of easy-to-hard. | `assert anti_order == list(reversed(easy_order))` |
| 4 | `test_mixed_schedule_p_hard` | Configure mixed schedule with `schedule_fn="linear"`, total epochs 100. At epoch 0, assert `p_hard` is near 0. At epoch 50, assert `p_hard` is near 0.5. At epoch 99, assert `p_hard` is near 1.0. Tolerance of 0.05. | `assert abs(p_hard_at_epoch(0) - 0.0) < 0.05` etc. |
| 5 | `test_kendall_tau_significance` | Generate random ordering of 50 tasks. Generate curriculum ordering (easy-to-hard). Compute Kendall tau between them. Assert `|tau| > 0` with p-value < 0.05 (curriculum is statistically different from random). | `tau, p = kendalltau(curriculum_ranks, random_ranks); assert p < 0.05` |
| 6 | `test_logging_epoch_info` | Run one epoch of curriculum ordering with logging enabled. Capture log output. Assert log contains: `"strategy"`, `"task_order_hash"`, `"cluster_histogram"`. | `assert "strategy" in log_entry` and `"task_order_hash" in log_entry` and `"cluster_histogram" in log_entry` |
| 7 | `test_determinism_same_seed` | Get curriculum order with seed 42 twice. Assert orders are identical. | `assert order1 == order2` |
| 8 | `test_different_seeds_different_order` | Get curriculum order with seed 42 and seed 99. Assert orders differ. With 50 tasks, the probability of identical orderings by chance is negligible. | `assert order_42 != order_99` |
| 9 | `test_schedule_linear` | Evaluate linear schedule at 11 evenly-spaced epochs (0 through 100). Assert `p_hard` increases linearly from 0.0 to 1.0. Check interpolated values within tolerance 0.02. | `for epoch in range(0, 101, 10): assert abs(p_hard(epoch) - epoch/100) < 0.02` |
| 10 | `test_schedule_cosine` | Evaluate cosine schedule at epoch boundaries. Assert `p_hard` follows `0.5 * (1 - cos(pi * epoch / total))`. Check at 0, 25, 50, 75, 100 percent of total epochs. | `assert abs(p_hard(epoch) - expected_cosine(epoch)) < 0.02` |
| 11 | `test_schedule_step` | Evaluate step schedule. Assert `p_hard` is 0 before the step point and 1.0 at and after the step point. Default step at 50% of total epochs. | `assert p_hard(49) == 0.0` and `assert p_hard(50) == 1.0` |
| 12 | `test_batch_size_enforcement` | Request a meta-batch of size 8. Assert exactly 8 task IDs are returned, regardless of curriculum strategy. | `assert len(batch) == 8` |
| 13 | `test_diversity_fallback_relaxation` | Set `diversity_min_distance=0.95` (very high threshold that cannot be satisfied for 10 tasks). Request batch of 10. Assert the function still returns 10 tasks (falls back to relaxed threshold rather than failing). | `assert len(batch) == 10` |
| 14 | `test_stratified_sampling` | Request stratified batch with 5 clusters. Assert at least one task from each cluster appears in the batch (batch size >= 5). | `for c in range(5): assert any(cluster_labels[t] == c for t in batch)` |
| 15 | `test_get_curriculum_order_no_drops` | Call `get_curriculum_order` with all 50 task IDs. Assert the returned list contains exactly the same 50 task IDs (no duplicates, no drops), just reordered. | `assert set(order) == set(task_ids)` and `len(order) == 50` |
| 16 | `test_build_meta_batch_size` | Call `build_meta_batch` with batch size 12. Assert exactly 12 task IDs returned. Repeat with batch size 1 and batch size 50. | `assert len(batch) == requested_size` for each |
| 17 | `test_empty_task_list` | Call `get_curriculum_order` with empty list. Assert returns empty list without error. Call `build_meta_batch` with empty list. Assert returns empty list. | `assert get_curriculum_order([]) == []` and `assert build_meta_batch([]) == []` |

---

## 5. TestConfig (~8 tests)

Tests the configuration dataclasses defined in `brain_ai/config.py` (the Task2Vec-related additions) and `assets/task2vec_config_template.py`.

### Fixtures

- None required (configs are lightweight dataclasses).

### Tests

| # | Test Name | Description | Assertion Pattern |
|---|---|---|---|
| 1 | `test_default_instantiation` | Instantiate `Task2VecConfig()`, `ClusterConfig()`, `CurriculumConfig()`, `RegistryConfig()` with no arguments. Assert no exception. Assert all fields have their documented default values. | `cfg = Task2VecConfig(); assert cfg.embedding_dim == 512` etc. |
| 2 | `test_preset_minimal` | Call `Task2VecFullConfig.minimal()`. Assert `embedding_dim` is small (e.g., 64), `probe_model` is `"conv4"`, `n_clusters` is set. Assert all sub-configs are valid. | `cfg = Task2VecFullConfig.minimal(); assert cfg.task2vec.embedding_dim == 64` |
| 3 | `test_preset_dev` | Call `Task2VecFullConfig.dev()`. Assert intermediate values suitable for development iteration. | `cfg = Task2VecFullConfig.dev(); assert cfg.task2vec.embedding_dim == 256` |
| 4 | `test_preset_production` | Call `Task2VecFullConfig.production()`. Assert production-scale values: `embedding_dim == 512`, `probe_model` appropriate for scale. | `cfg = Task2VecFullConfig.production(); assert cfg.task2vec.embedding_dim == 512` |
| 5 | `test_serialization_roundtrip` | Create config, call `to_dict()`, then `from_dict()` on the result. Assert all fields are equal to the original. Tests JSON-safe serialization for checkpoint storage. | `assert Config.from_dict(cfg.to_dict()) == cfg` |
| 6 | `test_invalid_embedding_dim` | Attempt to create `Task2VecConfig(embedding_dim=0)` or `Task2VecConfig(embedding_dim=-1)`. Assert `ValueError` is raised during validation. | `pytest.raises(ValueError)` |
| 7 | `test_invalid_n_clusters` | Attempt to create `ClusterConfig(n_clusters=0)` or `ClusterConfig(n_clusters=-5)`. Assert `ValueError`. | `pytest.raises(ValueError)` |
| 8 | `test_invalid_strategy` | Attempt to create `CurriculumConfig(strategy="nonexistent")`. Assert `ValueError`. Valid strategies are: `"none"`, `"easy_to_hard"`, `"diversity"`, `"anti_curriculum"`, `"mixed"`. | `pytest.raises(ValueError)` |
| 9 | `test_invalid_registry_format` | Attempt to create `RegistryConfig(format="xml")`. Assert `ValueError`. Valid formats are: `"jsonl_npz"` (or whatever the valid set is). | `pytest.raises(ValueError)` |

---

## 6. TestIntegration (~8 tests)

End-to-end tests that exercise the full pipeline: extraction, registration, clustering, and curriculum ordering together. These tests verify that the modules compose correctly.

### Fixtures

- `minimal_config`, `synthetic_episode`, `tmp_registry_dir`
- `multi_episode_generator`: fixture that yields 20 distinct `TaskEpisode` objects with different seeds

### Tests

| # | Test Name | Description | Assertion Pattern |
|---|---|---|---|
| 1 | `test_end_to_end_pipeline` | Extract embedding from 5 episodes. Register each in a fresh registry. Run k-means clustering on the registry. Generate curriculum ordering. Assert: embeddings have correct shape, registry has 5 entries, clustering produces labels, curriculum returns all 5 task IDs in some order. | `assert reg_len == 5` and `len(set(labels)) >= 1` and `set(order) == set(task_ids)` |
| 2 | `test_registry_survives_checkpoint` | Create registry with 10 entries. Save to disk (simulating checkpoint). Load into new registry object. Assert all 10 embeddings match originals within 1e-7. Assert metadata (task IDs, datasets, splits) match exactly. | `np.testing.assert_allclose` for embeddings and `assert` for metadata fields |
| 3 | `test_curriculum_changes_across_epochs` | Generate curriculum ordering for epoch 0 and epoch 50 using mixed schedule. Assert the orderings differ (the mixed schedule changes `p_hard` over epochs, so task selection should shift). | `assert order_epoch0 != order_epoch50` |
| 4 | `test_diversity_affects_composition` | Build two meta-batches: one with `diversity_min_distance=0.0` (unconstrained) and one with `diversity_min_distance=0.3`. Compute mean pairwise distance in each batch. Assert the diversity-constrained batch has higher mean pairwise distance. | `assert mean_dist_constrained >= mean_dist_unconstrained` |
| 5 | `test_config_presets_produce_working_pipelines` | For each preset (`minimal`, `dev`, `production`), construct the extractor, extract one embedding, register it, run clustering (trivially on 1 embedding), generate curriculum. Assert no exceptions are raised throughout. | No exception raised for any preset |
| 6 | `test_multiple_episodes_registry` | Extract embeddings from 20 episodes via `multi_episode_generator`. Register all. Assert registry length is 20. Assert all task IDs are unique. Assert all embeddings are L2-normalized. | `assert len(reg) == 20` and `len(set(ids)) == 20` and all norms within 1e-5 of 1.0 |
| 7 | `test_cluster_assignments_stable` | Extract 20 embeddings, register, cluster twice with different random seeds but same algorithm and `n_clusters`. Compute ARI between runs. Assert ARI >= 0.9 (matching the done-when gate from SKILL.md). | `assert adjusted_rand_score(labels_run1, labels_run2) >= 0.9` |
| 8 | `test_logging_valid_json` | Run full pipeline with logging enabled. Capture all log lines emitted by the curriculum module. Parse each as JSON. Assert all parse successfully and contain expected fields (`strategy`, `task_order_hash`, `epoch`). | `json.loads(line)` succeeds for all logged lines |

---

## Summary Statistics

| Test Class | Test Count | CUDA Required | Optional Deps |
|---|---|---|---|
| TestTask2VecExtractor | 16 | 2 tests | None |
| TestEmbeddingRegistry | 14 | 0 | None |
| TestClustering | 13 | 0 | HDBSCAN (1 test) |
| TestCurriculum | 17 | 0 | None |
| TestConfig | 9 | 0 | None |
| TestIntegration | 8 | 0 | None |
| **Total** | **77** | **2** | **1** |

## Running the Tests

```bash
# All tests (CPU only, skips CUDA and HDBSCAN tests)
python -m pytest tests/test_task2vec.py -v

# Include CUDA tests (requires GPU)
python -m pytest tests/test_task2vec.py -v --run-cuda

# Include all optional dependency tests
pip install hdbscan
python -m pytest tests/test_task2vec.py -v

# With coverage
python -m pytest tests/test_task2vec.py -v --cov=brain_ai.meta --cov-report=html

# Single test class
python -m pytest tests/test_task2vec.py::TestTask2VecExtractor -v

# Single test
python -m pytest tests/test_task2vec.py::TestClustering::test_stability_ari -v
```

## Generating the Test File

The test file can be generated from this matrix specification:

```bash
python scripts/gen_task2vec_tests.py --output tests/test_task2vec.py
```

The generator reads this matrix document and produces a fully runnable pytest module with all fixtures, skip markers, and assertion patterns described above.
