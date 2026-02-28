# PyTorch Lightning DataModule for Video

## Overview

`L.LightningDataModule` provides a clean interface for encapsulating data loading logic in a way that works correctly across single-GPU, multi-GPU DDP, and multi-node distributed training. For video datasets, the DataModule handles the lifecycle mismatch between one-time operations (metadata scanning, manifest building) and per-rank operations (dataset creation, dataloader construction).

The most important distributed training rule: **do not create a DistributedSampler manually.** Lightning's Trainer injects it automatically via `use_distributed_sampler=True` (the default). Manual sampler creation is a common source of bugs — data duplication, uneven distribution across ranks, or gradient sync issues.

---

## DataModule Lifecycle

Lightning calls DataModule methods in a specific order that matches the distributed training lifecycle:

```
__init__()        — always called, on all ranks, before any distributed setup
    ↓
prepare_data()    — called ONCE on rank 0 only (or main process in non-DDP)
    ↓
setup(stage)      — called on ALL ranks, after DDP is initialized
    ↓
*_dataloader()    — called on ALL ranks, per epoch
```

Understanding this lifecycle is critical for avoiding bugs.

### `__init__(cfg)`

Store configuration only. No dataset creation, no file I/O, no VideoReader construction.

```python
def __init__(self, cfg: DataModuleConfig):
    super().__init__()
    self.cfg = cfg
    self.train_dataset = None  # populated in setup()
    self.val_dataset   = None
    self.test_dataset  = None
```

### `prepare_data()`

Run one-time, rank-0-only operations:
- Scanning the video directory to build a manifest JSON
- Downloading data if needed
- Any write operations to shared storage

**Critical**: Do NOT assign `self.*` state here that you expect to be available in `setup()`. The `prepare_data()` call happens on rank 0, but `setup()` runs on all ranks independently. Only rank 0's `prepare_data()` runs; rank 1+ skip it entirely.

```python
def prepare_data(self) -> None:
    """
    Build manifest JSON if it doesn't exist. Runs on rank 0 only.
    Do NOT set self.train_dataset etc. here — not replicated to other ranks.
    """
    manifest_path = os.path.join(self.cfg.data_root, 'manifest.json')
    if not os.path.exists(manifest_path):
        # Scan and write manifest to disk
        scan_and_save_manifest(self.cfg.data_root, manifest_path)
    # After this, manifest.json exists on shared storage for all ranks to read
```

### `setup(stage)`

Run on all ranks after DDP is initialized. Read the manifest, create dataset instances.

```python
def setup(self, stage: Optional[str] = None) -> None:
    """
    Create dataset instances for the requested stage.

    stage: 'fit' (train+val), 'validate', 'test', 'predict', or None (all)
    """
    manifest = load_manifest(os.path.join(self.cfg.data_root, 'manifest.json'))

    train_manifest = [m for m in manifest if m.split == self.cfg.train_split]
    val_manifest   = [m for m in manifest if m.split == self.cfg.val_split]
    test_manifest  = [m for m in manifest if m.split == self.cfg.test_split]

    train_transform = build_train_transform(self.cfg.dataset_cfg)
    eval_transform  = build_eval_transform(self.cfg.dataset_cfg)

    if stage in ('fit', None):
        self.train_dataset = VideoDataset(train_manifest, self.cfg.dataset_cfg, train_transform)
        self.val_dataset   = VideoDataset(val_manifest,   self.cfg.dataset_cfg, eval_transform)

    if stage in ('validate', None):
        self.val_dataset   = VideoDataset(val_manifest,   self.cfg.dataset_cfg, eval_transform)

    if stage in ('test', None):
        self.test_dataset  = VideoDataset(test_manifest,  self.cfg.dataset_cfg, eval_transform)
```

---

## DataLoader Construction

### train_dataloader

```python
def train_dataloader(self) -> DataLoader:
    return DataLoader(
        self.train_dataset,
        batch_size=self.cfg.batch_size,
        shuffle=True,                          # shuffles within the rank's shard
        num_workers=self._get_num_workers(),
        pin_memory=self.cfg.pin_memory,
        persistent_workers=self.cfg.persistent_workers and self._get_num_workers() > 0,
        drop_last=True,                        # even batches for distributed
        prefetch_factor=self.cfg.prefetch_factor if self._get_num_workers() > 0 else None,
    )
```

### val_dataloader and test_dataloader

```python
def val_dataloader(self) -> DataLoader:
    return DataLoader(
        self.val_dataset,
        batch_size=self.cfg.batch_size,
        shuffle=False,
        num_workers=self._get_num_workers(),
        pin_memory=self.cfg.pin_memory,
        persistent_workers=self.cfg.persistent_workers and self._get_num_workers() > 0,
        drop_last=False,                       # evaluate on all examples
        prefetch_factor=self.cfg.prefetch_factor if self._get_num_workers() > 0 else None,
    )
```

---

## Distributed Sampler: Do NOT Add Manually

Lightning's `Trainer` wraps the DataLoader's sampler automatically when `use_distributed_sampler=True` (default). Adding your own `DistributedSampler` will cause:

1. **Data duplication**: each rank processes all data instead of a shard
2. **Double-wrapping**: Lightning's sampler wraps your sampler, causing incorrect behavior
3. **Epoch boundary bugs**: Lightning's sampler handles `set_epoch()` for shuffle correctness; yours won't be called

```python
# WRONG: manual DistributedSampler
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(self.train_dataset)  # Lightning wraps this AGAIN
return DataLoader(self.train_dataset, sampler=sampler, ...)

# CORRECT: no sampler, no shuffle=True in distributed (Lightning handles it)
return DataLoader(
    self.train_dataset,
    shuffle=True,    # Lightning replaces this with DistributedSampler when in DDP
    ...
)
```

You can verify the final sampler in a Lightning callback:

```python
class SamplerInspector(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        sampler = trainer.train_dataloader.sampler
        print(f"Sampler type: {type(sampler).__name__}")
        # Should print: "DistributedSampler" in DDP, "RandomSampler" in single-GPU
```

---

## Worker Scaling Formula

```python
def _get_num_workers(self) -> int:
    """
    Scale workers with number of GPUs.

    Formula: num_workers = num_workers_per_gpu * num_gpus
    """
    num_gpus = max(1, self.trainer.num_devices) if self.trainer else 1
    return self.cfg.num_workers_per_gpu * num_gpus
```

**Reference values for video loading:**

| Setup | Workers | Rationale |
|-------|---------|-----------|
| 1x GPU (dev) | 4 | 4 workers/GPU, low overhead |
| 4x GPU (A100) | 16 | 4 workers/GPU, IO-intensive |
| 8x GPU (A100) | 32 | 4 workers/GPU, max typical |
| CPU-only debug | 0 | Avoid worker overhead in debug |

**Too many workers:** Each worker loads a VideoReader per `__getitem__`. With 64 workers and 100k dataset items, you can exhaust RAM from buffer overhead. Start at 4/GPU and increase if GPU utilization is low.

**Too few workers:** GPU starves waiting for data. Monitor `DataLoader time` in Lightning's `on_train_batch_start` log.

---

## pin_memory

```python
pin_memory=True  # recommended for all GPU training
```

Pin memory (page-locked memory) enables asynchronous host-to-device transfers:
- Without pin_memory: Python copies tensor to staging area, then transfers to GPU. Two sequential copies.
- With pin_memory: DataLoader allocates in pinned memory, GPU can DMA directly. One copy.

Impact: ~15-30% throughput improvement for large batch sizes. Essential for multi-GPU training where each GPU is pulling data simultaneously.

Caveat: Pinned memory uses physical RAM and cannot be swapped to disk. On memory-constrained systems (< 32 GB RAM with large batches), reduce `num_workers` before disabling `pin_memory`.

---

## persistent_workers

```python
persistent_workers=True  # recommended for video loading
```

Without persistent workers, PyTorch respawns worker processes at the start of each epoch. Worker startup cost includes:
- Python interpreter startup
- Import of torch, decord, torchvision
- Module-level `set_bridge('torch')` call
- Initial warmup of the decord reader pool

For video loading, this startup can take 2-5 seconds per worker. With 16 workers and 200 epochs, that's 16 * 200 * 2.5s = ~2.2 hours of wasted startup time.

`persistent_workers=True` keeps worker processes alive between epochs. The per-epoch startup cost drops to near zero.

Note: `persistent_workers=True` requires `num_workers > 0`. With `num_workers=0`, this parameter is ignored.

---

## prefetch_factor

```python
prefetch_factor=2  # default; increase if GPU is idle between batches
```

`prefetch_factor=N` means each worker pre-loads N batches ahead of what the main process is consuming. Total pre-loaded batches = `num_workers * prefetch_factor`.

Increase `prefetch_factor` when:
- GPU utilization is low (< 90%) and data loading is the bottleneck
- IO bandwidth is underutilized (disk reads not saturated)

Decrease `prefetch_factor` when:
- RAM usage is too high (each pre-loaded batch stays in pinned memory)
- Batches are very large

---

## State Management for Training Resumption

To resume training mid-epoch (after crash), save the DataLoader state:

```python
# Lightning 2.0+ supports stateful dataloaders
class VideoDataModule(L.LightningDataModule):
    def state_dict(self) -> dict:
        return {
            'train_consumed': self._train_consumed,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._train_consumed = state_dict.get('train_consumed', 0)
```

For full mid-epoch resumption, Lightning 2.0+ provides `StatefulDataLoader` via `torch.utils.data.StatefulDataLoader` which supports `state_dict()` / `load_state_dict()` natively.

---

## Multi-Node Configuration

For multi-node training (e.g., 4 nodes x 8 GPUs = 32 GPUs total):

```python
# Lightning Trainer handles distributed init automatically
trainer = L.Trainer(
    num_nodes=4,
    devices=8,
    strategy='ddp',
)
trainer.fit(model, datamodule=video_datamodule)
```

The DataModule's `_get_num_workers()` should use `self.trainer.num_devices` which returns per-node GPU count. Total workers per node = `4 * 8 = 32` — each node spawns its own workers independently.

Data must be on shared storage (NFS, GCS) accessible from all nodes.

---

## collate_fn for Variable-Length Clips

The default `collate_fn` expects all tensors in a batch to have the same shape. If all clips have the same `(T, C, H, W)` shape (guaranteed by your config), the default works fine.

If you need variable-length clips (e.g., multi-scale testing), provide a custom collate:

```python
def video_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate a list of sample dicts into a batched dict.
    Pads videos to the maximum length in the batch.
    """
    max_t = max(item['video'].shape[0] for item in batch)
    videos = []
    labels = []
    for item in batch:
        vid = item['video']
        T = vid.shape[0]
        if T < max_t:
            pad = torch.zeros(max_t - T, *vid.shape[1:], dtype=vid.dtype)
            vid = torch.cat([vid, pad], dim=0)
        videos.append(vid)
        labels.append(item['label'])
    return {
        'video': torch.stack(videos, dim=0),
        'label': torch.tensor(labels),
    }

# Usage in train_dataloader:
return DataLoader(dataset, collate_fn=video_collate_fn, ...)
```

---

## Summary of Best Practices

| Setting | Value | Reason |
|---------|-------|--------|
| num_workers | 4 * num_gpus | Balance IO parallelism with memory |
| pin_memory | True | Async host-to-device copy |
| persistent_workers | True | Avoid worker respawn overhead |
| drop_last (train) | True | Even batches for DDP gradient sync |
| drop_last (val/test) | False | Evaluate on all examples |
| prefetch_factor | 2-4 | Pre-load batches ahead of GPU |
| DistributedSampler | Do NOT add | Lightning injects automatically |
| prepare_data state | No self.* assignments | Not propagated to all ranks |
| setup state | OK to assign | Called on all ranks |
