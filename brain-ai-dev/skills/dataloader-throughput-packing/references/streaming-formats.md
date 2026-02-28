# Streaming Formats

Three I/O backends for sequential, high-throughput data loading. Choose based on data format, storage system, and throughput requirements.

## 1. HF Streaming (Hugging Face Datasets)

### Overview

The simplest path to streaming. Works with any dataset on the Hugging Face Hub or local files in supported formats (JSON, Parquet, CSV, Arrow).

### Key API

```python
from datasets import load_dataset

# Stream from Hub
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

# Stream from local files
dataset = load_dataset("json", data_files="data/*.jsonl", split="train", streaming=True)
```

### Shuffling

HF streaming uses a **shuffle buffer** approach:

```python
# Buffer-based shuffle: fills buffer, then samples randomly from it
dataset = dataset.shuffle(seed=42, buffer_size=10000)
```

- `buffer_size` controls randomness quality vs memory. Larger = more random, more RAM.
- For pretraining, 10k-100k is typical.
- For fine-tuning with smaller data, buffer can equal dataset size.

### Epoch Management

```python
# CRITICAL: call set_epoch each epoch for different shuffling
dataset.set_epoch(epoch)
```

Without `set_epoch`, the shuffle order repeats identically across epochs.

### Deterministic Sharding

```python
# Shard at the dataset level for distributed training
dataset = dataset.shard(num_shards=world_size, index=rank)
```

This assigns every `world_size`-th example to each rank. Deterministic given the same seed and epoch.

### Converting Map to Iterable

```python
# Convert a map-style dataset to iterable with sharding built in
iterable_ds = dataset.to_iterable_dataset(num_shards=128)
```

Setting `num_shards` > `world_size` allows better load balancing.

## 2. WebDataset

### Overview

WebDataset stores samples in `.tar` archive shards. Designed for sequential I/O from networked or object storage (S3, GCS). Each shard is a standard tar file containing related files for each sample.

### Tar Shard Format

```
data-000000.tar
  sample_000000.txt
  sample_000000.json
  sample_000001.txt
  sample_000001.json
  ...
data-000001.tar
  ...
data-012345.tar
  ...
```

Naming convention: `data-{000000..012345}.tar` where the range defines shard indices.

### Key API

```python
import webdataset as wds

dataset = (
    wds.WebDataset("s3://bucket/data-{000000..012345}.tar")
    .shuffle(1000)              # shuffle buffer within shard
    .decode("pil")              # decode images (or use "torch" for tensors)
    .to_tuple("input.txt", "output.txt")
    .batched(32)
)
```

### Shard-Level Shuffle

WebDataset shuffles at two levels:
1. **Shard order**: Shards are shuffled before reading (set seed for determinism).
2. **Within-shard**: A buffer shuffles samples within each shard.

```python
dataset = (
    wds.WebDataset(urls, shardshuffle=True, seed=42)
    .shuffle(1000)  # within-shard buffer
)
```

### Distributed Shard Assignment

Assign non-overlapping shard ranges to each rank:

```python
all_shards = sorted(glob.glob("data-*.tar"))
rank_shards = all_shards[rank::world_size]  # interleaved assignment
# or: contiguous blocks
chunk = len(all_shards) // world_size
rank_shards = all_shards[rank * chunk : (rank + 1) * chunk]
```

WebDataset also has built-in distributed support:

```python
dataset = wds.WebDataset(urls).shard_selection(
    lambda shards: shards[rank::world_size]
)
```

### Caching Hooks

```python
dataset = wds.WebDataset(urls, cache_dir="/tmp/wds_cache", cache_size=50_000_000_000)
```

Shards are cached locally on first read. Subsequent epochs read from cache.

## 3. Token Memmap (Megatron-LM Style)

### Overview

The highest throughput option for text pretraining. Tokens are stored in contiguous binary files. Runtime cost is near-zero CPU: just slice numpy arrays.

### File Format

Two files per dataset:
- **`.bin`**: Raw token IDs as contiguous uint16 (or uint32 for large vocabs)
- **`.idx`**: Document boundary offsets as int64

```python
import numpy as np

# Reading
tokens = np.memmap("data.bin", dtype=np.uint16, mode="r")
doc_offsets = np.memmap("data.idx", dtype=np.int64, mode="r")

# Document i spans tokens[doc_offsets[i] : doc_offsets[i+1]]
```

### Offline Tokenization Pipeline

```python
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

all_tokens = []
doc_boundaries = [0]

for doc in documents:
    tokens = tokenizer.encode(doc, add_special_tokens=False)
    all_tokens.extend(tokens)
    doc_boundaries.append(len(all_tokens))

# Write binary
token_array = np.array(all_tokens, dtype=np.uint16)
token_array.tofile("data.bin")

boundary_array = np.array(doc_boundaries, dtype=np.int64)
boundary_array.tofile("data.idx")
```

### Document Boundary Tracking

The `.idx` file stores the start offset of each document. This enables:
- Slicing individual documents without scanning the whole file
- Respecting document boundaries in pretraining block construction
- Efficient random access to any document by index

### Runtime Slicing

```python
# Slice a fixed-length block for pretraining
block_start = random_offset
block = tokens[block_start : block_start + seq_len]
# Zero copy! np.memmap returns a view.
```

## Caching Policy

### Shard-Level Caching

Cache entire shards, not individual samples. Benefits:
- Better I/O locality (sequential reads)
- Simpler eviction logic (evict whole shards)
- Matches the streaming access pattern

### LRU Eviction

```python
class ShardCache:
    def __init__(self, cache_dir: str, max_size_bytes: int):
        self.cache_dir = cache_dir
        self.max_size = max_size_bytes
        self.access_order = OrderedDict()  # shard_id -> size

    def get_or_fetch(self, shard_id: str, fetch_fn) -> Path:
        cache_path = Path(self.cache_dir) / shard_id
        if cache_path.exists() and self._validate_checksum(cache_path, shard_id):
            self.access_order.move_to_end(shard_id)
            return cache_path
        # Evict if needed
        while self._total_size() + self._shard_size(shard_id) > self.max_size:
            evicted_id, _ = self.access_order.popitem(last=False)
            (Path(self.cache_dir) / evicted_id).unlink()
        # Fetch and cache
        fetch_fn(shard_id, cache_path)
        self.access_order[shard_id] = cache_path.stat().st_size
        self._write_checksum(cache_path, shard_id)
        return cache_path
```

### Checksum Validation

Store a SHA256 checksum alongside each cached shard:

```python
import hashlib

def compute_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
```

On cache hit, verify the checksum. On mismatch, evict the corrupted shard and re-fetch.

### Corruption Recovery

1. Detect: checksum mismatch or read error
2. Evict: delete the corrupted cached shard and its checksum file
3. Re-fetch: download the shard again from the source
4. Validate: verify the new copy's checksum
5. Log: record the corruption event for monitoring

## Backend Selection Decision Tree

```
Is data already tokenized as contiguous binary?
  YES -> Token memmap (highest throughput)
  NO  ->
    Is data on object storage (S3/GCS) or network filesystem?
      YES -> Is data already in tar shards?
        YES -> WebDataset
        NO  -> Can you resharding offline?
          YES -> WebDataset (reformat then use)
          NO  -> HF Streaming
      NO  ->
        Is data on HuggingFace Hub?
          YES -> HF Streaming
          NO  -> Is dataset small enough to tokenize offline?
            YES -> Token memmap (tokenize offline, use memmap)
            NO  -> HF Streaming (supports local files)
```

### Performance Expectations

| Backend | Typical Throughput | CPU Overhead | Storage Pattern |
|---------|-------------------|--------------|-----------------|
| Token memmap | Highest (memory-mapped) | Near-zero | Sequential block reads |
| WebDataset | High (sequential tar reads) | Low (tar extraction) | Sequential shard reads |
| HF Streaming | Moderate (depends on format) | Moderate (decoding) | Sequential with buffer |

All three backends are sequential I/O. The key differences are in decoding overhead and memory access patterns.
