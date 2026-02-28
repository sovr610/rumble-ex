---
name: Data Loader Throughput + Sequence Packing
description: >
  This skill should be used when the user asks to "audit dataloader throughput",
  "measure data pipeline stalls", "implement sequence packing",
  "reduce padding waste in training", "add cu_seqlens packing for SFT",
  "boundary-aware packing FlashAttention", "DataCollatorWithFlattening",
  "configure DataLoader num_workers prefetch_factor", "tune DataLoader settings",
  "streaming dataset pipeline", "WebDataset tar shards",
  "memmap pretraining dataset", "HF datasets streaming IterableDataset",
  "deterministic sharding per rank", "DistributedSampler set_epoch",
  "padding ratio metrics", "effective tokens per second",
  "pretraining block builder", "data stall ratio measurement",
  "persistent_workers pin_memory tuning", "shard cache policy",
  "packed sequences with position_ids reset", "varlen_attn cu_seqlens",
  or needs guidance on turning the input pipeline into a first-class
  performance target with measurement, streaming I/O, packing, and
  deterministic distributed sharding.
version: 0.1.0
---

# Data Loader Throughput + Sequence Packing

## Overview

Turn the input pipeline into a first-class performance target. Measure where step time goes (GPU idle waiting on data), upgrade to streaming + cache-friendly formats, implement sequence packing to eliminate padding waste, and enforce deterministic per-rank sharding for distributed training.

Design principle: **measure first, stream sequentially, pack tightly, shard deterministically.**

## Pipeline Performance Model

GPU step time decomposes into two regions:

```
t_total_step = t_data + t_fwd_bwd_opt
data_stall_ratio = t_data / t_total_step
```

On real LLM runs, `data_stall_ratio` of 20-40% is common when the pipeline is untuned. The goal is single-digit percent.

## Public Contract

### PipelineAuditor

Instruments the training loop to measure data stalls and emit actionable metrics.

```python
class PipelineAuditor:
    def __init__(self, device: torch.device): ...
    def mark_data_start(self) -> None: ...
    def mark_data_end(self) -> None: ...
    def mark_compute_start(self) -> None: ...
    def mark_compute_end(self) -> None: ...
    def report(self) -> PipelineMetrics: ...
    def to_json(self, path: str) -> None: ...
```

### SequencePacker

Packs variable-length sequences into fixed-length blocks with correct boundary tracking.

```python
class SequencePacker:
    def __init__(self, cfg: PackingConfig): ...
    def pack_pretrain(self, token_streams: Iterator[List[int]]) -> Iterator[PackedBlock]: ...
    def pack_sft(self, examples: List[TokenizedExample]) -> PackedBatch: ...
    def padding_ratio(self, batch: PackedBatch) -> float: ...
```

### ShardedStreamDataset

Deterministic per-rank streaming dataset with caching.

```python
class ShardedStreamDataset(torch.utils.data.IterableDataset):
    def __init__(self, cfg: StreamConfig, rank: int, world_size: int): ...
    def __iter__(self) -> Iterator[Dict[str, Tensor]]: ...
    def set_epoch(self, epoch: int) -> None: ...
```

## Key Concepts

### Phase 1 — Measure Pipeline Utilization

Instrument the step with three timers: `t_data` (need-batch to batch-on-GPU), `t_fwd_bwd_opt` (forward/backward/optimizer), `t_total_step`. Compute `data_stall_ratio = t_data / t_total_step` and `gpu_busy_ratio = 1 - data_stall_ratio`.

Record CUDA transfer attribution: time for `batch.to(device, non_blocking=True)` and whether `pin_memory=True` is enabled. Emit `data_metrics.json` with p50/p90 stall ratios, tokens/sec (raw and effective), padding ratio, and DataLoader settings.

### Phase 2 — Streaming Reads + Caching

Three I/O backend options based on data format:

| Backend | Best for | Key API |
|---------|----------|---------|
| HF Streaming | Massive corpora, quick adoption | `load_dataset(..., streaming=True)` |
| WebDataset | Sequential I/O, networked storage | `wds.WebDataset(shard_pattern)` |
| Token memmap | Text pretraining throughput ceiling | `np.memmap(path, dtype=np.uint16)` |

**HF Streaming**: Use `shuffle(seed, buffer_size=...)` and `set_epoch(epoch)` for reshuffling. Deterministic sharding: `dataset.shard(num_shards=world_size, index=rank)`.

**WebDataset**: Store samples in `.tar` shards (`data-{000000..012345}.tar`). Shard-level shuffle. Assign shard ranges per-rank for deterministic distribution.

**Token memmap**: Offline tokenize into contiguous `.bin` + `.idx` files. Runtime is near-zero CPU: slice token blocks directly. This is the Megatron-LM approach.

**Caching**: Cache whole shards (not individual samples) for better locality. LRU by shard, validate checksums, evict and re-fetch on corruption.

### Phase 3 — DataLoader Tuning

Key knobs on `torch.utils.data.DataLoader`:

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `num_workers` | 0 | Start at 4 per GPU, increase until no gain |
| `prefetch_factor` | 2 | Batches prefetched per worker; 2-8 range |
| `persistent_workers` | False | Set True if epochs repeat |
| `pin_memory` | False | Set True for GPU training |

Enforce `sampler.set_epoch(epoch)` when using `DistributedSampler`. Add a watchdog: if a worker dies or stalls, fail loudly with the last shard/sample ID.

### Phase 4 — Sequence Packing

Two distinct packing modes:

**A) Pretraining Block Builder** — Concatenate tokenized documents into fixed-length blocks of `target_seq_len` tokens. Maintain a rolling buffer, emit full blocks, track document boundaries for optional loss masking. Padding approaches zero.

**B) SFT Boundary-Aware Packing** — Pack variable-length examples while preserving attention isolation via `cu_seqlens`. Uses `flash_attn_varlen_func` or PyTorch's `torch.nn.attention.varlen.varlen_attn`.

Output for SFT packing:
- `input_ids`: flattened concatenated tokens
- `cu_seqlens`: `[0, len1, len1+len2, ...]` (int32)
- `position_ids`: reset to 0 at each boundary
- `labels`: per-example masking preserved

**Bucketing**: Group samples by length (e.g., boundaries `[256, 512, 1024, 2048]`) before packing. Reduces fragmentation and shape churn for `torch.compile`.

**Throughput**: HF `DataCollatorWithFlattening` reports 2x throughput on high-variance datasets (FLAN), 1.4x on low-variance (OrcaMath), with 20% peak memory reduction.

### Phase 5 — Deterministic Sharding

**Map-style datasets**: Use `DistributedSampler(dataset, shuffle=True, seed=seed, drop_last=True)`. Call `sampler.set_epoch(epoch)` each epoch. All ranks share the same seed.

**Iterable/streaming datasets**: Shard at the shard-file level (best) or sample-index level. HF streaming: `dataset.shard(num_shards=world_size, index=rank)`.

Log sample-ID hashes for a small subset per rank as a sanity check against accidental duplication.

### Phase 6 — Metrics Integration

Extend benchmark harness metrics with:

| Metric | Key |
|--------|-----|
| Data stall ratio | `data_stall_ratio_p50`, `_p90` |
| Padding ratio | `padding_ratio` |
| Effective tokens/sec | `effective_tokens_per_sec_p50` |
| Raw tokens/sec | `raw_tokens_per_sec_p50` |
| Packing mode | `packing_mode` |
| Sharding mode | `sharding_mode` |

`effective_tokens_per_sec = real_nonpad_tokens / step_time` — gate decisions on this metric, not raw tokens.

## Configuration Surface

```python
@dataclass
class DataPipelineConfig:
    # I/O format
    format: str = "hf_streaming"         # hf_streaming | webdataset_tar | token_memmap
    streaming: bool = True
    cache_dir: Optional[str] = None
    # Shuffle
    shuffle_buffer_size: int = 10000
    # DataLoader
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    # Packing
    pack_enabled: bool = False
    pack_mode: str = "none"              # none | pretrain_blocks | sft_boundary_aware
    pack_target_seq_len: int = 2048
    pack_bucket_boundaries: Tuple[int, ...] = (256, 512, 1024, 2048)
    pack_boundary_aware: bool = True
    # Sharding
    shard_deterministic: bool = True
    shard_seed: int = 42
```

## Done-When Gates

1. **Pipeline Measured** — `data_stall_ratio` is computed and logged per step. `data_metrics.json` contains p50/p90 stall ratios, raw and effective tokens/sec, padding ratio, and DataLoader settings.
2. **Streaming Works** — At least one sequential I/O backend (HF streaming, WebDataset, or memmap) loads data without random I/O. Caching stores whole shards with checksum validation.
3. **Packing Eliminates Padding** — `padding_ratio` drops materially with packing enabled. SFT packing produces correct `cu_seqlens`, `position_ids` (reset at boundaries), and `labels`. Pretraining block builder emits fixed-length blocks with near-zero padding.
4. **Deterministic Sharding** — In distributed runs, ranks do not duplicate samples within an epoch. Given `(seed, epoch, rank)`, shard order is stable.

## Resources

### Reference Files
- **`references/pipeline-measurement.md`** — Timer instrumentation, CUDA transfer attribution, data_metrics.json schema, stall ratio interpretation, watchdog implementation
- **`references/streaming-formats.md`** — HF streaming API, WebDataset tar shards, token memmap format, shard caching policy, backend selection guide
- **`references/sequence-packing.md`** — Pretraining block builder, SFT boundary-aware packing, cu_seqlens construction, position_ids reset, bucketing strategy, DataCollatorWithFlattening, varlen_attn API, padding ratio computation
- **`references/dataloader-tuning.md`** — num_workers/prefetch_factor/persistent_workers/pin_memory tuning grid, DistributedSampler set_epoch, worker watchdog, effective batch availability
- **`references/deterministic-sharding.md`** — Map-style DistributedSampler, iterable stream sharding, shard-file-level assignment, sample-ID dedup verification, epoch reseeding
- **`references/testing-matrix.md`** — Test scenarios for all components across 6 phases

### Asset Files
- **`assets/pipeline_auditor_template.py`** — PipelineAuditor with timers, CUDA attribution, metrics JSON, self-tests
- **`assets/sequence_packer_template.py`** — SequencePacker with pretrain blocks, SFT boundary-aware packing, cu_seqlens, bucketing, self-tests
- **`assets/streaming_dataset_template.py`** — ShardedStreamDataset with HF/WebDataset/memmap backends, caching, self-tests
- **`assets/dataloader_tuner_template.py`** — DataLoader configuration, tuning grid runner, watchdog, self-tests
- **`assets/packing_collators_template.py`** — Padding and packing collators, DataCollatorWithFlattening-style, metrics, self-tests
- **`assets/pipeline_config_template.py`** — DataPipelineConfig with validation, serialization, self-tests

### Scripts
- **`scripts/validate_pipeline.py`** — Validates done-when gates (measurement, streaming, packing, sharding)
- **`scripts/gen_pipeline_tests.py`** — Generates pytest test cases covering all 6 phases
- **`scripts/throughput_benchmark.py`** — Measures tokens/sec, padding ratio, stall ratio across configurations
