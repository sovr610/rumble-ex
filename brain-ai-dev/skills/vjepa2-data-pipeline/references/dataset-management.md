# Dataset Management Reference

## Multi-Source Dataset Architecture

V-JEPA 2 supports mixing multiple video datasets during training using per-source
weights. This enables curriculum learning, domain balancing, and heterogeneous
dataset combination without rebuilding index files.

### ConcatIndices

`ConcatIndices` provides a two-level indexing scheme that maps a global flat index
to a `(dataset_idx, sample_idx)` pair.

```python
class ConcatIndices:
    """
    Maps global indices to (dataset_index, sample_index) pairs
    for multi-source datasets.

    Example:
        datasets = [DatasetA (1000 samples), DatasetB (500 samples)]
        concat = ConcatIndices([1000, 500])
        concat[0]    -> (0, 0)     # first sample of DatasetA
        concat[999]  -> (0, 999)   # last sample of DatasetA
        concat[1000] -> (1, 0)     # first sample of DatasetB
        concat[1499] -> (1, 499)   # last sample of DatasetB
    """
    def __init__(self, dataset_sizes: List[int]):
        self.sizes = dataset_sizes
        self.cumulative = [0] + list(itertools.accumulate(dataset_sizes))
        self.total = self.cumulative[-1]

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, global_idx: int) -> Tuple[int, int]:
        if global_idx < 0 or global_idx >= self.total:
            raise IndexError(f"Index {global_idx} out of range [0, {self.total})")

        # Binary search for dataset index
        dataset_idx = bisect.bisect_right(self.cumulative, global_idx) - 1
        sample_idx  = global_idx - self.cumulative[dataset_idx]
        return dataset_idx, sample_idx

    def get_dataset_ranges(self) -> List[Tuple[int, int]]:
        """Returns (start, end) index ranges for each dataset."""
        return [(self.cumulative[i], self.cumulative[i+1])
                for i in range(len(self.sizes))]
```

### MultiSourceVideoDataset

```python
class MultiSourceVideoDataset(Dataset):
    """
    Wraps multiple VideoDataset instances behind a unified index space.
    Delegates __getitem__ to the appropriate sub-dataset.
    """
    def __init__(
        self,
        datasets: List[VideoDataset],
        weights: Optional[List[float]] = None,
    ):
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)
        sizes = [len(d) for d in datasets]
        self.index_map = ConcatIndices(sizes)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, global_idx: int) -> Dict[str, Any]:
        dataset_idx, sample_idx = self.index_map[global_idx]
        return self.datasets[dataset_idx][sample_idx]
```

---

## DistributedWeightedSampler

Standard `DistributedSampler` does not support per-sample weights.
`DistributedWeightedSampler` combines weighted sampling with rank-based
partitioning to ensure:

1. Each rank gets a disjoint subset of indices (no duplicates across ranks)
2. Sampling probabilities respect the per-source weight vector
3. Full coverage: every epoch visits all samples exactly once per global pass

### Design

```
Global index space: [0, N)
    |
    v [weighted permutation]
Global permutation: [p_0, p_1, ..., p_{N-1}]
    |
    v [partition by rank]
Rank 0: [p_0, p_{ws}, p_{2*ws}, ...]
Rank 1: [p_1, p_{ws+1}, p_{2*ws+1}, ...]
...
Rank R: [p_R, p_{ws+R}, p_{2*ws+R}, ...]
```

### Weight Normalization

Per-source weights are expanded to per-sample weights:

```python
def expand_weights(
    dataset_sizes: List[int],
    source_weights: List[float],
) -> List[float]:
    """
    Expand per-source weights to per-sample weights.

    Example:
        sizes   = [1000, 500]
        weights = [2.0,  1.0]
        -> sample_weights[0:1000]   = 2.0 / 1000 (each sample in source 0)
        -> sample_weights[1000:1500]= 1.0 / 500  (each sample in source 1)
    """
    sample_weights = []
    for size, weight in zip(dataset_sizes, source_weights):
        per_sample = weight / size if size > 0 else 0.0
        sample_weights.extend([per_sample] * size)
    return sample_weights
```

### Complete Implementation

```python
class DistributedWeightedSampler(Sampler):
    def __init__(
        self,
        weights: List[float],        # Per-sample weights, length N
        num_samples: int,            # Total samples to draw (usually N)
        rank: int,                   # This process rank
        world_size: int,             # Total number of processes
        seed: int = 0,               # For reproducibility across epochs
        replacement: bool = False,   # False = no-duplicate sampling
    ):
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.replacement = replacement
        self.epoch = 0

        # Pad num_samples to be divisible by world_size
        self.num_samples_per_rank = math.ceil(num_samples / world_size)
        self.total_size = self.num_samples_per_rank * world_size

    def set_epoch(self, epoch: int):
        """Call this before each epoch for different shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.replacement:
            # Weighted sampling with replacement
            indices = torch.multinomial(
                self.weights, self.total_size,
                replacement=True, generator=g
            ).tolist()
        else:
            # Weighted shuffle without replacement
            # Use argsort of random scores scaled by weights
            scores = torch.rand(len(self.weights), generator=g) ** (1.0 / self.weights)
            indices = torch.argsort(scores, descending=True).tolist()

            # Pad to total_size by cycling
            while len(indices) < self.total_size:
                indices += indices
            indices = indices[:self.total_size]

        # Partition: rank R takes indices[R::world_size]
        rank_indices = indices[self.rank:self.total_size:self.world_size]
        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples_per_rank
```

---

## Variable FPC Per Source

Different datasets can have different `frames_per_clip` (FPC) values when
constructing a `MultiSourceVideoDataset`. This is useful when:

- Pre-training data: 16 FPC at low resolution
- Fine-tuning data: 32 FPC at higher resolution
- Domain-specific data: custom FPC matching task requirements

```python
datasets = [
    VideoDataset(paths_a, frames_per_clip=16, clip_mode="fps"),
    VideoDataset(paths_b, frames_per_clip=32, clip_mode="duration"),
]
multi = MultiSourceVideoDataset(datasets, weights=[2.0, 1.0])
```

**Collation note:** When mixing FPC, the DataLoader collate function must
handle variable-length temporal dims. Use a mask collator or pad to the
maximum FPC in the batch.

---

## DataLoader Construction

### Standard Configuration

```python
def build_train_loader(
    dataset: Dataset,
    sampler: Sampler,
    batch_size: int,
    num_workers: int,
    mask_collator: Optional[Callable] = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=mask_collator,
        pin_memory=pin_memory,
        drop_last=True,           # Required for distributed training
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
    )
```

### Worker Init Function (LCG Seeding)

LCG (Linear Congruential Generator) provides fast, deterministic, non-correlated
seeds across workers without requiring numpy or torch state manipulation at
startup.

```python
def worker_init_fn(worker_id: int):
    """
    Deterministic LCG-based seeding for DataLoader workers.

    LCG formula: seed_{n+1} = (a * seed_n + c) % m
    Using POSIX parameters: a=1664525, c=1013904223, m=2^32
    """
    worker_info = torch.utils.data.get_worker_info()
    base_seed = worker_info.seed % (2**31)

    # LCG step to decorrelate worker seeds
    LCG_A = 1664525
    LCG_C = 1013904223
    LCG_M = 2**32
    seed = (LCG_A * (base_seed + worker_id) + LCG_C) % LCG_M

    random.seed(seed)
    np.random.seed(seed % (2**31))
    torch.manual_seed(seed)
```

**Reproducibility:** Given the same `DataLoader` configuration and epoch,
worker seeds are deterministic. Set `torch.backends.cudnn.deterministic=True`
for full reproducibility including GPU ops.

---

## Resource Monitoring (Optional)

A background monitoring thread can track CPU and memory usage per worker:

```python
import threading
import psutil
import time

class ResourceMonitor:
    """
    Background thread that logs CPU/memory every `interval` seconds.
    Start in worker_init_fn; stops automatically when worker exits.
    """
    def __init__(self, interval: float = 5.0, worker_id: int = 0):
        self.interval = interval
        self.worker_id = worker_id
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        proc = psutil.Process()
        while not self._stop.wait(self.interval):
            cpu  = proc.cpu_percent()
            mem  = proc.memory_info().rss / 1e6  # MB
            print(f"[Worker {self.worker_id}] CPU={cpu:.1f}% MEM={mem:.0f}MB")
```

---

## Index File Format

For large datasets, pre-computed index files avoid scanning the filesystem
at startup. Each line: `relative/path/to/video.mp4 num_frames label`

```
videos/video_001.mp4 300 0
videos/video_002.mp4 450 1
...
```

```python
def load_video_index(index_path: str) -> List[Tuple[str, int, int]]:
    """Load (path, num_frames, label) triples from index file."""
    records = []
    with open(index_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                path, num_frames = parts[0], int(parts[1])
                label = int(parts[2]) if len(parts) >= 3 else -1
                records.append((path, num_frames, label))
    return records
```

This avoids the expensive `len(VideoReader(path))` call during dataset
initialization when datasets have millions of clips.
