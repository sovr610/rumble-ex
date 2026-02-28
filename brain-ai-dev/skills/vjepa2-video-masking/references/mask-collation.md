# Mask Collator Design

## Overview

`MaskCollator` is a custom DataLoader `collate_fn` that combines standard batch
collation with spatiotemporal mask generation.  It groups batch items by their
frames-per-clip (FPC) value, generates masks per group, and returns a list of
`(collated_batch, masks_enc, masks_pred)` tuples -- one per FPC group.

---

## Why a Custom Collator?

PyTorch's `default_collate` stacks tensors along a new batch dimension, which requires
all tensors to have identical shapes.  When a dataset returns clips of different lengths
(multi-FPC training), naive collation fails.  `MaskCollator` solves this by:

1. Sorting batch items into FPC bins.
2. Running `default_collate` within each bin (same shape per bin).
3. Generating masks sized to match each bin's token grid.

---

## Worker Seed Determinism

DataLoader workers run in separate processes.  If each worker independently
samples a random seed, masks will differ across runs even with `torch.manual_seed`.

**Solution**: share a `multiprocessing.Value('i', 0)` counter across workers.  Each
call to `MaskCollator.__call__` atomically increments the counter and uses it to
derive the RNG seed:

```python
from multiprocessing import Value

class MaskCollator:
    def __init__(self, ...):
        self._itr_counter = Value('i', -1)  # shared across forked workers

    def step(self):
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return self._itr_counter.value

    def __call__(self, batch):
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        # pass g to mask generator for reproducibility
        ...
```

Because the counter is shared via OS-level shared memory, all workers see the same
monotonically increasing step number.  Replaying training with the same dataset order
produces bit-identical masks.

---

## FPC Grouping Algorithm

```python
from torch.utils.data import default_collate
from collections import defaultdict

def _group_by_fpc(batch):
    """Return dict {fpc: [item, ...]} preserving order within each group."""
    groups = defaultdict(list)
    for item in batch:
        fpc = item['frames'].shape[0]   # temporal dimension of raw clip
        groups[fpc].append(item)
    return groups

def collate(batch, mask_generators, seed):
    results = []
    groups  = _group_by_fpc(batch)

    for fpc, items in groups.items():
        collated = default_collate(items)
        B        = len(items)

        # Pick the mask generator that matches this FPC's token grid
        mg   = mask_generators[fpc]
        m_enc, m_pred = mg(B, seed=seed)

        results.append((collated, m_enc, m_pred))

    return results
```

---

## max_context_frames_ratio Integration

The collator passes `max_context_frames_ratio` through to each `MaskGenerator`:

```python
MaskGenerator(
    ...
    max_context_frames_ratio = self.max_context_frames_ratio,
)
```

This ensures that when the encoder sees a reduced temporal context the masking is
consistent regardless of which worker generated the batch.

---

## max_keep Integration

Similarly, `max_keep` is forwarded to the mask generator.  The collator may also
enforce `max_keep` as a post-processing step on the collated encoder masks if
per-sample token counts differ after FPC grouping:

```python
if self.max_keep is not None:
    m_enc = [_cap_keep(m, self.max_keep, g) for m in m_enc]
```

where `_cap_keep` randomly selects at most `max_keep` True elements.

---

## Return Shape Contract

```
List[Tuple[
    collated_batch : Dict[str, Tensor]   # standard collated fields
    masks_enc      : List[Tensor[N]]     # one bool tensor per sample
    masks_pred     : List[Tensor[N]]     # one bool tensor per sample
]]
```

Length of the outer list equals the number of distinct FPC values in the batch.
Length of `masks_enc` / `masks_pred` equals the number of samples in that FPC group.

---

## Integration with DataLoader

```python
from torch.utils.data import DataLoader

mask_collator = MaskCollator(
    mask_generators          = make_mask_generators(config),
    max_context_frames_ratio = config.mask.max_context_frames_ratio,
    max_keep                 = config.mask.max_keep,
)

loader = DataLoader(
    dataset,
    batch_size  = 32,
    num_workers = 8,
    collate_fn  = mask_collator,
    pin_memory  = True,
)

for batch_groups in loader:
    for collated, masks_enc, masks_pred in batch_groups:
        # process each FPC group
        ...
```

---

## Thread Safety Notes

- `multiprocessing.Value` uses a reentrant lock (`get_lock()`).  Always acquire the
  lock before reading **and** writing to avoid TOCTOU races.
- The collator object is created once in the main process before the DataLoader is
  constructed so that all workers share the same underlying `ctypes` array via OS fork.
  Creating new `multiprocessing.Value` objects inside worker processes defeats the
  purpose of the shared counter.
- Setting `persistent_workers=True` keeps workers alive between epochs, preserving
  the shared counter state and improving seed determinism.

---

## Minimal Complete Example

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Value
from torch.utils.data.dataloader import default_collate

class MinimalMaskCollator:
    def __init__(self, grid_size=(8, 14, 14)):
        self._counter = Value('i', -1)
        self.T, self.H, self.W = grid_size

    def _step(self):
        with self._counter.get_lock():
            self._counter.value += 1
            return self._counter.value

    def __call__(self, batch):
        seed   = self._step()
        g      = torch.Generator()
        g.manual_seed(seed)
        N      = self.T * self.H * self.W
        B      = len(batch)
        collated = default_collate(batch)
        # Simple 50% random mask for illustration
        rand   = torch.rand(B, N, generator=g)
        m_pred = rand < 0.5
        m_enc  = ~m_pred
        return [(collated, list(m_enc), list(m_pred))]

dataset = TensorDataset(torch.randn(64, 3, 16, 224, 224))
collator = MinimalMaskCollator()
loader   = DataLoader(dataset, batch_size=4, collate_fn=collator, num_workers=2)
for groups in loader:
    for batch, enc, pred in groups:
        print(batch[0].shape, enc[0].shape)
        break
```
