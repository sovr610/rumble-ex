# Rank-Aware Data Loading — Reference for Distributed Scaling Skill

This document specifies the distributed data loading strategy for `brain_ai` training: DistributedSampler usage, epoch shuffling, worker seeding, duplicate prevention, and memory optimization. Use this as the canonical reference when implementing, auditing, or debugging rank-aware data loading infrastructure.

---

## 1. DistributedSampler

The `DistributedSampler` partitions a dataset across ranks so that each rank processes a unique, non-overlapping subset of samples per epoch. Without it, all ranks would process the same data, wasting compute.

### 1.1 Basic Usage

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=False,
)

loader = DataLoader(
    dataset,
    batch_size=micro_batch_size,
    sampler=sampler,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)
```

### 1.2 How DistributedSampler Partitions

Given a dataset of N samples and W ranks, the sampler:

1. Creates a permutation of indices `[0, 1, ..., N-1]` (if `shuffle=True`) or the identity (if `shuffle=False`).
2. Pads the permutation to length `ceil(N / W) * W` by repeating initial samples. This ensures every rank gets the same number of samples.
3. Splits the padded permutation into W contiguous chunks of size `ceil(N / W)`.
4. Rank `r` receives chunk `r`.

**Example:** N=10 samples, W=3 ranks
```
Permutation:  [7, 2, 9, 0, 5, 3, 8, 1, 6, 4]
Padded to 12: [7, 2, 9, 0, 5, 3, 8, 1, 6, 4, 7, 2]  (repeats first 2)
Rank 0: [7, 2, 9, 0]
Rank 1: [5, 3, 8, 1]
Rank 2: [6, 4, 7, 2]
```

Note that samples 7 and 2 appear twice globally (once in rank 0 and once in rank 2) due to padding. With `drop_last=True` on the sampler, the padded samples would be excluded, but the default is `drop_last=False`.

### 1.3 Padding and drop_last

**Sampler-level `drop_last=False` (default):** Pads with duplicate samples. Every rank processes the same number of batches. Gradient synchronization works because all ranks call backward the same number of times.

**Sampler-level `drop_last=True`:** Drops samples from each rank so that total is exactly `floor(N / W) * W`. No duplicates, but some samples are never seen. This is acceptable for large datasets (ImageNet, RedPajama) where missing a few samples per epoch is insignificant.

**DataLoader-level `drop_last=True`:** Drops the final incomplete batch from each rank. This is important to maintain consistent batch sizes for BatchNorm and for gradient accumulation (which expects consistent micro-batch sizes).

**Recommendation for BrainAI:** Use `DistributedSampler(drop_last=False)` with `DataLoader(drop_last=True)`. This minimizes wasted samples while ensuring all ranks produce the same number of complete batches.

---

## 2. set_epoch() — Critical for Correct Shuffling

### 2.1 Why set_epoch() Is Required

The `DistributedSampler` generates its permutation from a random number generator seeded with `seed + epoch`. If `set_epoch()` is never called, the epoch remains 0 forever, and the sampler produces the same permutation every epoch.

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # CRITICAL: changes shuffle order
    for batch in loader:
        train_step(batch)
```

### 2.2 Consequences of Forgetting set_epoch()

- Every epoch sees the same data in the same order.
- The model memorizes the fixed order, harming generalization.
- Different ranks still see different subsets (the partition is correct), but the partition is always the same.

This is a silent correctness bug — training appears to work but produces a worse model. It is not caught by any runtime error.

### 2.3 Verification

To verify that `set_epoch()` is changing the order:

```python
sampler.set_epoch(0)
indices_epoch0 = list(sampler)

sampler.set_epoch(1)
indices_epoch1 = list(sampler)

assert indices_epoch0 != indices_epoch1, "set_epoch() is not changing shuffle order!"
assert set(indices_epoch0) == set(indices_epoch1), "Different samples across epochs!"
```

---

## 3. Worker Seeding Per Rank

### 3.1 The Problem

Each DataLoader worker process has its own random state. By default, PyTorch seeds workers as `base_seed + worker_id`, where `base_seed` is the same across ranks. This means worker 0 on rank 0 and worker 0 on rank 1 produce identical random augmentations.

For BrainAI training with data augmentation (random crops, color jitter, noise injection for audio), this reduces the effective diversity of augmentations.

### 3.2 Rank-Aware Worker Seeding

```python
def worker_init_fn(worker_id: int):
    """Seed each worker uniquely based on rank and worker ID."""
    worker_seed = torch.initial_seed() % (2**32)
    import numpy as np
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)

loader = DataLoader(
    dataset,
    batch_size=micro_batch_size,
    sampler=sampler,
    num_workers=8,
    worker_init_fn=worker_init_fn,
)
```

PyTorch's DataLoader already sets `torch.initial_seed()` to `base_seed + worker_id` per worker. Adding the rank to the base seed ensures cross-rank uniqueness:

```python
# Set generator with rank-specific seed
g = torch.Generator()
g.manual_seed(42 + rank)

loader = DataLoader(
    dataset,
    sampler=sampler,
    generator=g,
    worker_init_fn=worker_init_fn,
)
```

### 3.3 Deterministic Training

For reproducible experiments, seed everything:

```python
def seed_everything(seed: int, rank: int):
    """Set all random seeds, incorporating rank for uniqueness."""
    effective_seed = seed + rank
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)
    import numpy as np
    np.random.seed(effective_seed)
    import random
    random.seed(effective_seed)
    # For CUDA determinism (at the cost of performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 4. Preventing Duplicate Samples

### 4.1 Verification Strategy

After constructing the sampler and loader, verify that no sample index appears in more than one rank's batch:

```python
def verify_no_duplicates(sampler, world_size):
    """Verify that each sample is assigned to exactly one rank."""
    all_indices = []
    for rank in range(world_size):
        sampler_copy = DistributedSampler(
            sampler.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # Deterministic for verification
            drop_last=True,
        )
        all_indices.extend(list(sampler_copy))

    from collections import Counter
    counts = Counter(all_indices)
    duplicates = {idx: count for idx, count in counts.items() if count > 1}
    assert len(duplicates) == 0, f"Duplicate samples across ranks: {duplicates}"
```

### 4.2 Handling Uneven Dataset Sizes

When `len(dataset)` is not divisible by `world_size`, the sampler must handle the remainder. There are three strategies:

**Padding (default):** Add duplicate samples to make the length divisible. Some samples are seen twice. Acceptable for training (the extra samples are a tiny fraction of the epoch).

**Dropping:** Use `drop_last=True` to skip remainder samples. Some samples are never seen in that epoch. Acceptable for large datasets.

**Per-rank variable length:** Each rank gets a different number of samples. This breaks DDP's assumption that all ranks call backward the same number of times. Do NOT use this with DDP or FSDP.

**Recommendation:** Use the default padding strategy. For BrainAI's production datasets (millions of samples), the padding overhead is negligible.

### 4.3 Multi-Dataset Training

BrainAI trains on multiple datasets simultaneously (vision, text, audio, multimodal). Each dataset needs its own sampler:

```python
samplers = {}
loaders = {}
for modality, dataset in datasets.items():
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=micro_batch_size, sampler=sampler)
    samplers[modality] = sampler
    loaders[modality] = loader

# At each epoch start
for sampler in samplers.values():
    sampler.set_epoch(epoch)
```

For interleaved training (alternating modalities), ensure all samplers advance at the same rate to prevent data starvation.

---

## 5. pin_memory with Multi-GPU

### 5.1 What pin_memory Does

Setting `pin_memory=True` allocates host memory in page-locked (pinned) memory. Transfers from pinned memory to GPU are faster because they can use DMA (direct memory access) without involving the CPU.

### 5.2 When to Enable

Always enable `pin_memory=True` for GPU training. The memory overhead is small (equal to one batch in pinned memory) and the transfer speedup is 2-3x.

```python
loader = DataLoader(
    dataset,
    batch_size=micro_batch_size,
    sampler=sampler,
    num_workers=8,
    pin_memory=True,
    pin_memory_device=f"cuda:{rank}",  # Pin to the correct GPU
)
```

### 5.3 pin_memory_device

On multi-GPU systems, `pin_memory_device` specifies which GPU the pinned memory targets. Without it, pinned memory is prepared for GPU 0 by default. Setting it to the rank's GPU avoids an extra inter-GPU copy.

### 5.4 Transferring Pinned Data

When using pinned memory, transfer data with `non_blocking=True`:

```python
for data, target in loader:
    data = data.to(f"cuda:{rank}", non_blocking=True)
    target = target.to(f"cuda:{rank}", non_blocking=True)
    # The transfer happens asynchronously; CUDA operations will wait for it
    output = model(data)
```

The `non_blocking=True` allows the CPU to continue preparing the next batch while the current batch transfers to GPU. This overlaps data transfer with computation.

---

## 6. num_workers Tuning

### 6.1 Guidelines

The optimal `num_workers` depends on:
- Number of CPU cores per GPU
- Data loading complexity (decoding images vs reading tensors)
- Disk I/O bandwidth

**BrainAI training config defaults to `num_workers=8`.**

### 6.2 Per-Rank Worker Count

On a server with 8 GPUs and 64 CPU cores, each rank gets approximately 8 cores. Setting `num_workers=8` per rank means 64 total worker processes, which matches the core count.

**Over-subscribing** (e.g., `num_workers=16` per rank = 128 total) causes CPU contention and actually slows down data loading. A good rule of thumb:

```python
num_workers_per_rank = total_cpu_cores // world_size
```

### 6.3 prefetch_factor

The `prefetch_factor` controls how many batches each worker prefetches:

```python
loader = DataLoader(
    dataset,
    num_workers=8,
    prefetch_factor=4,  # Each worker prefetches 4 batches
)
```

With 8 workers and `prefetch_factor=4`, up to 32 batches are buffered. This uses approximately `32 * batch_memory` of RAM but ensures the GPU never stalls waiting for data.

For BrainAI's multi-modal data (images + text + audio), each batch can be large. Monitor host RAM usage and reduce `prefetch_factor` if memory pressure is high.

---

## 7. BrainAI-Specific Loading Considerations

### 7.1 Multi-Modal Batch Construction

BrainAI expects inputs as a dictionary: `{'vision': tensor, 'text': tensor, ...}`. The DataLoader must collate multi-modal samples:

```python
def multi_modal_collate(batch):
    """Collate function for multi-modal BrainAI data."""
    collated = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(batch[0][key], (int, float)):
            collated[key] = torch.tensor([sample[key] for sample in batch])
    return collated
```

### 7.2 Variable-Length Sequences

Text and audio inputs have variable lengths. Use padding within the collate function:

```python
def pad_collate(batch):
    """Pad variable-length sequences in a multi-modal batch."""
    collated = {}
    for key in batch[0].keys():
        tensors = [sample[key] for sample in batch]
        if key in ('text', 'token_ids', 'audio'):
            collated[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=0
            )
        else:
            collated[key] = torch.stack(tensors)
    return collated
```

### 7.3 Engram Token Preparation

When `use_engram=True`, the data loader must provide `token_ids` in addition to text embeddings. These are integer indices into the engram hash table:

```python
# In dataset __getitem__:
return {
    'text': text_embedding,       # (seq_len, embed_dim)
    'token_ids': token_ids,       # (seq_len,) — integer indices
    'vision': image_tensor,       # (C, H, W)
    'label': label,               # scalar
}
```

### 7.4 Phase-Specific Data Loading

The 7-phase training pipeline activates different data streams per phase:

| Phase | Primary Data | Modalities |
|-------|-------------|------------|
| 1 (SNN Core) | Vision (MNIST/CIFAR in dev, ImageNet in prod) | vision |
| 2 (Encoders) | All modality datasets | vision, text, audio |
| 3 (HTM) | Sequential data (text, time series) | text, sensors |
| 4 (Workspace) | Multi-modal pairs | vision+text, audio+text |
| 5 (Active Inference) | RL environments | sensors |
| 6 (Reasoning) | Reasoning benchmarks | text |
| 7 (Meta-Learning) | Few-shot episodic data | all |

Each phase should configure its own set of samplers and loaders. Reuse the `RankAwareDataLoader` class with phase-specific dataset configurations.

---

## 8. Debugging Data Loading Issues

### 8.1 Verifying Sample Coverage

After one epoch, check that all samples were seen exactly once across all ranks:

```python
seen_indices = set()
for batch_idx, (data, target) in enumerate(loader):
    # Retrieve the actual indices from the sampler
    start = batch_idx * micro_batch_size
    end = start + len(target)
    batch_indices = list(sampler)[start:end]
    seen_indices.update(batch_indices)

# Gather across ranks
all_seen = [None] * world_size
dist.all_gather_object(all_seen, seen_indices)
if rank == 0:
    total_unique = len(set().union(*all_seen))
    print(f"Unique samples seen: {total_unique} / {len(dataset)}")
```

### 8.2 DataLoader Hangs

If the DataLoader hangs with `num_workers > 0`:
- Set `num_workers=0` to diagnose if it is a worker issue.
- Check for deadlocks in custom dataset `__getitem__` (e.g., file locks).
- Ensure `worker_init_fn` does not block.
- On some systems, `multiprocessing_context='spawn'` is needed.

### 8.3 Memory Leaks

Worker processes can accumulate memory over time. Use `persistent_workers=True` with periodic restarts:

```python
loader = DataLoader(
    dataset,
    num_workers=8,
    persistent_workers=True,  # Keep workers alive between epochs
)
```

If memory grows, set `persistent_workers=False` and accept the per-epoch worker startup cost (typically 1-3 seconds).
