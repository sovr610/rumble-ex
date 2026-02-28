# Multi-Block 3D Masking Algorithm

## Overview

V-JEPA 2 uses a **multi-block spatiotemporal masking** strategy.  Unlike random
per-token masking (used in BERT-style models), the masks are contiguous 3-D
cuboids in the `(T, H, W)` token grid.  This forces the predictor to reason about
structured spatiotemporal regions rather than interpolating from surrounding tokens.

---

## Block Size Sampling

For each mask block the following procedure is used:

```
temporal_len = round( T_grid * uniform(t_scale_min, t_scale_max) )
spatial_area = H_grid * W_grid * uniform(s_scale_min, s_scale_max)
aspect       = uniform(aspect_min, aspect_max)

block_h = round( sqrt(spatial_area * aspect) )
block_w = round( sqrt(spatial_area / aspect) )

# Clamp to grid bounds
block_h = max(1, min(block_h, H_grid))
block_w = max(1, min(block_w, W_grid))
```

The resulting block covers `temporal_len * block_h * block_w` tokens.

---

## Spatial Configuration Presets

| Preset  | `s_scale_min` | `s_scale_max` | Purpose                          |
|---------|---------------|---------------|----------------------------------|
| Small   | 0.15          | 0.15          | Fine-grained local prediction    |
| Large   | 0.70          | 0.70          | Coarse holistic prediction       |
| Default | 0.15          | 0.70          | Balanced (sample from range)     |

V-JEPA 2 trains with **both** small (npred=8) and large (npred=2) mask generators
simultaneously.  The small masks produce many local targets; the large mask produces
few holistic targets.  Losses are summed across mask types.

---

## Temporal Configuration

```
temporal_scale = (1.0, 1.0)
```

`temporal_scale = (1.0, 1.0)` means every prediction block always spans the full
temporal depth of the token grid.  This forces the predictor to model motion across
the entire clip rather than short snippets.

For ablation experiments use `temporal_scale = (0.5, 1.0)` to allow blocks covering
50–100 % of the temporal extent.

---

## Aspect Ratio

```
aspect_ratio = (0.75, 1.5)
```

The aspect ratio `aspect = block_h / block_w` is sampled uniformly from this range.
Values < 1.0 produce wide, short blocks; values > 1.0 produce tall, narrow blocks.
Uniform sampling prevents the network from exploiting axis-aligned regularities.

---

## Prediction Block Placement

```python
def _place_block(T, H, W, temporal_len, block_h, block_w):
    """Sample a random (t0, h0, w0) such that the block fits within the grid."""
    t0 = random.randint(0, T - temporal_len)
    h0 = random.randint(0, H - block_h)
    w0 = random.randint(0, W - block_w)
    return t0, h0, w0
```

For `npred` blocks per sample:

```python
pred_mask = torch.zeros(T, H, W, dtype=torch.bool)
for _ in range(npred):
    temporal_len, block_h, block_w = sample_block_size(...)
    t0, h0, w0 = _place_block(T, H, W, temporal_len, block_h, block_w)
    pred_mask[t0:t0+temporal_len, h0:h0+block_h, w0:w0+block_w] = True
```

Blocks are placed *with replacement* — overlapping prediction blocks are allowed and
simply union together.  The encoder mask is the strict complement:

```python
enc_mask = ~pred_mask
```

---

## max_context_frames_ratio

```python
max_context_frames_ratio: float = 1.0
```

After computing `enc_mask`, this ratio limits how many temporal frames the encoder
actually sees.  For `max_context_frames_ratio < 1.0`:

```python
max_context_frames = round(T_grid * max_context_frames_ratio)
# Zero out encoder visibility beyond this frame index
enc_mask[max_context_frames:, :, :] = False
```

This creates a **causal masking** variant where the encoder only observes the first
`max_context_frames` frames and must predict future frames — a temporally harder
pretraining objective.

---

## max_keep Token Cap

```python
max_keep: Optional[int] = None
```

After all masks are computed, `max_keep` caps the number of encoder-visible tokens.
If `enc_mask.sum() > max_keep`, a random subset of `max_keep` visible tokens is
selected:

```python
if max_keep is not None:
    visible_indices = enc_mask.flatten().nonzero(as_tuple=False).squeeze(1)
    if len(visible_indices) > max_keep:
        perm = torch.randperm(len(visible_indices))[:max_keep]
        enc_mask.fill_(False)
        enc_mask_flat = enc_mask.flatten()
        enc_mask_flat[visible_indices[perm]] = True
        enc_mask = enc_mask_flat.view(T, H, W)
```

Purpose: bound peak GPU memory when batch size is fixed but clip length varies
(e.g. in multi-FPC batches).

---

## Mask Shapes Returned

The `MaskGenerator.__call__` returns flat boolean tensors indexed over token positions:

```
masks_enc  : List[Tensor]  # length = batch_size; each shape [N]
masks_pred : List[Tensor]  # length = batch_size; each shape [N]
```

where `N = T_grid * H_grid * W_grid`.  Flat indexing allows direct use with
`torch.index_select` or boolean indexing along the sequence dimension.

---

## Correctness Invariant

For every sample `i` in the batch the following must hold:

```python
assert (masks_enc[i] | masks_pred[i]).all(), "Gap: some tokens uncovered"
assert not (masks_enc[i] & masks_pred[i]).any(), "Overlap: some tokens double-counted"
```

Equivalently: `masks_enc[i]` is the exact bitwise complement of `masks_pred[i]`.

Note: the invariant holds before `max_keep` truncation.  After truncation, the encoder
mask is a *subset* of the complement — overlaps are still forbidden but gaps are
permitted (dropped tokens are not predicted).

---

## Algorithm Pseudocode (Full)

```python
def generate_masks(batch_size, T, H, W,
                   s_scale, t_scale, aspect_ratio, npred,
                   max_context_frames_ratio=1.0, max_keep=None):
    N = T * H * W
    masks_enc  = []
    masks_pred = []

    for _ in range(batch_size):
        pred = torch.zeros(T, H, W, dtype=torch.bool)

        for _ in range(npred):
            # 1. Sample block size
            tl   = max(1, round(T * random.uniform(*t_scale)))
            area = H * W * random.uniform(*s_scale)
            asp  = random.uniform(*aspect_ratio)
            bh   = max(1, min(round(math.sqrt(area * asp)), H))
            bw   = max(1, min(round(math.sqrt(area / asp)), W))

            # 2. Place block
            t0 = random.randint(0, T - tl)
            h0 = random.randint(0, H - bh)
            w0 = random.randint(0, W - bw)
            pred[t0:t0+tl, h0:h0+bh, w0:w0+bw] = True

        # 3. Encoder = complement
        enc = ~pred

        # 4. Temporal context cap
        mcf = round(T * max_context_frames_ratio)
        if mcf < T:
            enc[mcf:, :, :] = False

        # 5. max_keep cap
        if max_keep is not None:
            vis = enc.flatten().nonzero(as_tuple=False).squeeze(1)
            if len(vis) > max_keep:
                sel = vis[torch.randperm(len(vis))[:max_keep]]
                enc_flat = torch.zeros(N, dtype=torch.bool)
                enc_flat[sel] = True
                enc = enc_flat.view(T, H, W)

        masks_enc.append(enc.flatten())
        masks_pred.append(pred.flatten())

    return masks_enc, masks_pred
```
