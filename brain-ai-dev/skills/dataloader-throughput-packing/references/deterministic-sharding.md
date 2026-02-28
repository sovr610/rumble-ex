# Deterministic Sharding

Ensuring every rank in distributed training gets a unique, reproducible subset of data. Given `(seed, epoch, rank)`, the shard order must be fully determined.

## Map-Style Datasets: DistributedSampler

### Standard Pattern

```python
from torch.utils.data import DistributedSampler, DataLoader

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=42,
    drop_last=True       # ensures all ranks get same number of batches
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)   # CRITICAL
    for batch in loader:
        train_step(batch)
```

### How DistributedSampler Works

1. Creates a random permutation of all indices using `seed + epoch` as the random seed.
2. If `drop_last=True`, truncates to a multiple of `world_size`.
3. Splits the permuted indices into `world_size` contiguous chunks.
4. Rank `i` gets chunk `i`.

All ranks use the same seed, so they generate the same permutation and thus agree on the split. Each rank deterministically gets its own non-overlapping subset.

### set_epoch(epoch)

```python
sampler.set_epoch(epoch)
```

This changes the random seed to `self.seed + epoch`, producing a different permutation each epoch. Without calling this, the seed stays at `self.seed + 0` and the same ordering repeats every epoch.

**This is the #1 most common distributed training bug.** It silently produces correct-looking training that converges worse due to reduced data diversity.

### drop_last=True

Ensures all ranks process the same number of batches. Without it:
- Some ranks may get one extra batch.
- This causes hangs at `all_reduce` barriers (rank 0 is at step N+1 while rank 1 is still at step N).
- Always use `drop_last=True` in distributed training.

## Iterable/Streaming Datasets

### Shard-File-Level Assignment (Preferred)

The best approach for streaming data: assign non-overlapping shard files to each rank.

```python
all_shards = sorted(glob.glob("data-*.tar"))

# Interleaved assignment (better load balance if shards vary in size)
rank_shards = [s for i, s in enumerate(all_shards) if i % world_size == rank]

# Contiguous assignment (better for sequential I/O on networked storage)
chunk = len(all_shards) // world_size
rank_shards = all_shards[rank * chunk : (rank + 1) * chunk]
```

**Why shard-level is best:**
- No communication between ranks needed.
- No duplicate samples possible (each shard assigned to exactly one rank).
- Efficient sequential I/O within each rank's shard set.
- Easy to verify: just check shard assignment lists don't overlap.

### Sample-Index-Level Sharding

If shard-level assignment is not possible (e.g., single large file), shard at the sample index level:

```python
class ShardedIterableDataset(IterableDataset):
    def __init__(self, total_samples, rank, world_size, seed, epoch=0):
        self.total_samples = total_samples
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = rng.permutation(self.total_samples)
        # Each rank takes every world_size-th sample
        rank_indices = indices[self.rank::self.world_size]
        for idx in rank_indices:
            yield self.load_sample(idx)
```

### HF Streaming Sharding

```python
from datasets import load_dataset

dataset = load_dataset("allenai/c4", split="train", streaming=True)
dataset = dataset.shard(num_shards=world_size, index=rank)
dataset = dataset.shuffle(seed=42, buffer_size=10000)
```

HF's `shard()` assigns every `world_size`-th example to the rank. Combined with a fixed seed, this is deterministic.

For epoch reshuffling:

```python
dataset.set_epoch(epoch)  # Changes the shuffle seed internally
```

### WebDataset Distributed Sharding

```python
import webdataset as wds

all_urls = [f"data-{i:06d}.tar" for i in range(10000)]
rank_urls = all_urls[rank::world_size]

dataset = wds.WebDataset(rank_urls, shardshuffle=True, seed=42 + epoch)
```

By assigning shard URLs per-rank and using `seed + epoch` for shard-order shuffling, each rank gets deterministic, non-overlapping data.

## Sample-ID Dedup Verification

### Purpose

As a sanity check, verify that ranks are not accidentally receiving duplicate samples. This catches bugs in sharding logic.

### Implementation

```python
import hashlib

def hash_sample(sample_id) -> str:
    return hashlib.md5(str(sample_id).encode()).hexdigest()[:8]

# Each rank logs hashes of first N samples
sample_hashes_per_rank = {}
N = 100

# On each rank:
my_hashes = set()
for i, batch in enumerate(loader):
    if i >= N:
        break
    for sid in batch["sample_id"]:
        my_hashes.add(hash_sample(sid))

# Gather to rank 0 (using dist.all_gather or file-based)
# On rank 0:
all_hashes = [set_from_rank_0, set_from_rank_1, ...]
for i in range(len(all_hashes)):
    for j in range(i + 1, len(all_hashes)):
        overlap = all_hashes[i] & all_hashes[j]
        if overlap:
            logging.error(
                f"DEDUP VIOLATION: rank {i} and rank {j} share "
                f"{len(overlap)} samples: {overlap}"
            )
```

### When to Run

- During development: every run.
- In production: first epoch of each new dataset configuration.
- After changing sharding parameters.

## Epoch Reseeding

### Mechanism

Given `(seed, epoch, rank)`, the data order is fully determined:

```
effective_seed = base_seed + epoch
permutation = random_permutation(total_samples, seed=effective_seed)
rank_samples = permutation[rank::world_size]
```

### Reproducibility Contract

- Same `(seed, epoch, rank)` produces the same sample sequence.
- Different `epoch` produces a different permutation.
- Different `rank` selects a different subset from the same permutation.
- Changing `seed` produces a completely different permutation.

### Verification

```python
def verify_reproducibility(seed, epoch, rank, world_size, num_samples):
    """Run twice with same args, verify identical output."""
    order_1 = get_sample_order(seed, epoch, rank, world_size, num_samples)
    order_2 = get_sample_order(seed, epoch, rank, world_size, num_samples)
    assert order_1 == order_2, "Non-deterministic sharding detected!"

    # Verify different epochs produce different orders
    order_diff = get_sample_order(seed, epoch + 1, rank, world_size, num_samples)
    assert order_1 != order_diff, "set_epoch not changing order!"
```

## Common Pitfalls

### 1. Forgetting set_epoch

**Symptom:** Training converges but validation loss plateaus earlier than expected.
**Cause:** Same data order every epoch reduces effective data diversity.
**Fix:** Always call `sampler.set_epoch(epoch)` or `dataset.set_epoch(epoch)`.

### 2. Inconsistent Seeds Across Ranks

**Symptom:** Some ranks duplicate data, others miss data.
**Cause:** Each rank uses a different base seed, producing different permutations that don't partition cleanly.
**Fix:** All ranks must share the same base seed. Only `rank` should differ.

### 3. drop_last Mismatches

**Symptom:** Training hangs at gradient synchronization (all_reduce).
**Cause:** Some ranks have one more batch than others, so they reach a synchronization point that other ranks never reach.
**Fix:** Always use `drop_last=True` with `DistributedSampler`. For iterable datasets, pad the shorter rank's data or use a barrier.

### 4. Non-Deterministic Worker Ordering

**Symptom:** Different runs with the same seed produce slightly different sample orders.
**Cause:** When `num_workers > 0`, the order batches are returned depends on which worker finishes first.
**Fix:** Use `worker_init_fn` with deterministic seeds, or accept that within-epoch ordering may vary slightly (sample set is still correct).

### 5. Streaming Dataset Epoch Boundaries

**Symptom:** Streaming dataset repeats the same data or skips data at epoch boundaries.
**Cause:** `IterableDataset` doesn't have a built-in epoch concept.
**Fix:** Explicitly implement `set_epoch()` that changes the shuffle seed.
