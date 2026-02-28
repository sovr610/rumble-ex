# Testing Matrix

Test scenarios across all 6 phases of the Data Loader Throughput + Sequence Packing skill.

## Phase 1: Pipeline Measurement

| Test ID | Description | Validation |
|---------|-------------|------------|
| P1.1 | Timer accuracy: mark_data_start/end produces positive t_data | `t_data > 0` |
| P1.2 | Timer accuracy: mark_compute_start/end produces positive t_fwd_bwd_opt | `t_fwd_bwd_opt > 0` |
| P1.3 | t_total_step >= t_data + t_fwd_bwd_opt (no negative gaps) | `t_total >= t_data + t_compute` (within tolerance) |
| P1.4 | data_stall_ratio in [0, 1] | `0 <= ratio <= 1` |
| P1.5 | Stall ratio computation: known t_data / t_total matches | Inject known values, verify ratio |
| P1.6 | PipelineMetrics p50/p90 aggregation: 100 samples, verify percentiles | `abs(p50 - np.percentile(data, 50)) < epsilon` |
| P1.7 | data_metrics.json schema: all required fields present | JSON schema validation |
| P1.8 | to_json() produces valid JSON file readable by json.load | File exists, json.load succeeds |
| P1.9 | CUDA event timers work on GPU (if available) | GPU timing > 0 when work is done |
| P1.10 | Host-to-device transfer attribution: t_h2d > 0 for non-trivial tensor | Measured t_h2d is positive |

## Phase 2: Streaming Reads + Caching

| Test ID | Description | Validation |
|---------|-------------|------------|
| P2.1 | HF streaming backend: yields batches in sequence | Batches are non-empty dicts |
| P2.2 | Memmap backend: loads .bin data, yields token blocks | Block length == target_seq_len |
| P2.3 | Shard caching: first read caches, second read hits cache | Cache directory has shard files |
| P2.4 | Checksum validation: cached shard checksum matches | compute_checksum(cached) == stored_checksum |
| P2.5 | Corruption recovery: corrupted cache evicted and refetched | After corruption, data still loads correctly |
| P2.6 | set_epoch changes iteration order for streaming backend | Samples from epoch 0 != samples from epoch 1 |
| P2.7 | ShardedStreamDataset yields data without error for 100 iterations | No exceptions, data is tensors |
| P2.8 | Cache LRU eviction: when cache full, oldest shard evicted | Oldest shard file removed |

## Phase 3: DataLoader Tuning

| Test ID | Description | Validation |
|---------|-------------|------------|
| P3.1 | DataLoader with num_workers=0 yields batches | Batches are non-empty |
| P3.2 | DataLoader with num_workers=2 yields batches | Batches are non-empty |
| P3.3 | DataLoader with prefetch_factor=4 yields batches | Batches are non-empty |
| P3.4 | persistent_workers=True: second epoch starts without worker respawn delay | Time-to-first-batch in epoch 2 < epoch 1 |
| P3.5 | pin_memory=True: batch tensors are pinned | `batch.is_pinned()` is True |
| P3.6 | DistributedSampler set_epoch: epoch 0 != epoch 1 ordering | Sample indices differ |
| P3.7 | Tuning grid runner completes without error | Returns best config dict |
| P3.8 | WorkerWatchdog: no stall within timeout | check() returns True |
| P3.9 | WorkerWatchdog: stall detected after timeout | check() returns False or raises |
| P3.10 | create_dataloader() respects all config parameters | Loader attributes match config |

## Phase 4: Sequence Packing

| Test ID | Description | Validation |
|---------|-------------|------------|
| P4.1 | Pretrain block builder: output blocks are target_seq_len | `len(block.input_ids) == target_seq_len` |
| P4.2 | Pretrain blocks: near-zero padding (only last block may have padding) | Total padding < target_seq_len |
| P4.3 | Pretrain blocks: doc_boundaries tracked correctly | Boundaries match input document lengths |
| P4.4 | SFT packing: cu_seqlens starts at 0 | `cu_seqlens[0] == 0` |
| P4.5 | SFT packing: cu_seqlens is monotonically increasing | `all(cu_seqlens[i] < cu_seqlens[i+1])` |
| P4.6 | SFT packing: cu_seqlens[-1] == total tokens | `cu_seqlens[-1] == len(input_ids)` |
| P4.7 | SFT packing: position_ids reset to 0 at each cu_seqlens boundary | Check position_ids at each boundary |
| P4.8 | SFT packing: labels preserved per example | Labels match input examples |
| P4.9 | Bucketing: assigns samples to correct buckets | 100-token sample -> 256 bucket |
| P4.10 | Bucketing: reduces number of unique sequence lengths | Fewer unique lengths after bucketing |
| P4.11 | padding_ratio(): returns 0.0 for perfectly packed batch | `ratio == 0.0` |
| P4.12 | padding_ratio(): returns correct value for known padding | `ratio == expected` |
| P4.13 | Packing collator: produces valid cu_seqlens from variable-length inputs | Schema validation |
| P4.14 | Padding collator: pads to max length in batch | All sequences same length |

## Phase 5: Deterministic Sharding

| Test ID | Description | Validation |
|---------|-------------|------------|
| P5.1 | DistributedSampler: rank 0 and rank 1 get different indices | `set(rank0_indices) & set(rank1_indices) == empty` |
| P5.2 | DistributedSampler: all indices covered (union = full dataset minus dropped) | `len(union) == expected` |
| P5.3 | set_epoch changes sample ordering | `order_epoch0 != order_epoch1` |
| P5.4 | Same (seed, epoch, rank) produces identical ordering | Two runs produce same order |
| P5.5 | Stream sharding: different ranks get different data | First 100 samples differ |
| P5.6 | Stream sharding: set_epoch changes stream order | Epoch 0 samples != epoch 1 samples |
| P5.7 | Sample-ID dedup: no overlap between ranks | Hash intersection is empty |
| P5.8 | Shard-file assignment: no shard assigned to two ranks | Shard lists are disjoint |
| P5.9 | Seed reproducibility: same seed+epoch gives same shard assignment | Two calls produce identical assignment |
| P5.10 | drop_last=True: all ranks get same batch count | `count_rank0 == count_rank1` |

## Phase 6: Metrics Integration

| Test ID | Description | Validation |
|---------|-------------|------------|
| P6.1 | data_metrics.json has all required fields from schema | JSON schema validation |
| P6.2 | data_stall_ratio_p50 in [0, 1] | Bounds check |
| P6.3 | data_stall_ratio_p90 >= data_stall_ratio_p50 | Monotonicity of percentiles |
| P6.4 | effective_tokens_per_sec computed correctly | `effective == raw * (1 - padding_ratio)` |
| P6.5 | padding_ratio in [0, 1) | Bounds check |
| P6.6 | raw_tokens_per_sec > 0 | Positive throughput |
| P6.7 | packing_mode recorded in metrics | Field present and valid enum |
| P6.8 | sharding_mode recorded in metrics | Field present and valid enum |
| P6.9 | DataLoader settings captured in metrics | num_workers, prefetch_factor present |
| P6.10 | Metrics round-trip: write JSON, read back, values match | Serialization fidelity |

## Cross-Phase Integration Tests

| Test ID | Description | Validation |
|---------|-------------|------------|
| X.1 | Full pipeline: stream -> pack -> dataloader -> audit | All components work together |
| X.2 | Packing reduces padding_ratio vs no packing | `packed_ratio < unpacked_ratio` |
| X.3 | Effective tokens/sec improves with packing | `packed_effective > unpacked_effective` |
| X.4 | Distributed sharding + packing: ranks get unique packed batches | No sample overlap |
| X.5 | Metrics reflect actual pipeline configuration | Config matches metrics output |

## Test Execution Notes

- **GPU tests** (P1.9, P1.10): Skip if CUDA not available. Use `pytest.mark.skipif`.
- **Distributed tests** (P5.x): Simulate with `rank` and `world_size` parameters, no actual multi-process needed for unit tests.
- **Streaming tests** (P2.1): Use synthetic data to avoid network dependencies.
- **Timing tests** (P1.x): Use generous tolerances (timers are noisy).
- **All tests**: Should complete in < 30 seconds total (no heavy model loading).
