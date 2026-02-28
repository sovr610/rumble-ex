# Multi-Sequence Encoder / Predictor Wrappers

## Overview

When training with heterogeneous clip lengths (multi-FPC mode), a single forward pass
through the encoder must handle groups of clips whose token sequences differ in length.
The `MultiSequenceEncoder` and `MultiSequencePredictor` wrappers manage this by
processing each FPC group independently and returning a list of results.

---

## Why Not Pad to Max Length?

Padding all clips to the longest sequence in the batch wastes compute (attention is
quadratic in sequence length) and distorts mask statistics.  Instead, V-JEPA 2 groups
clips by their token count so that every group can be batched as a regular dense tensor
inside the encoder.

---

## MultiSequenceEncoder

### Architecture

```
Input:  x_groups       -- list of tensors, each [B_g, N_g, D]
        masks_groups   -- list of lists of bool tensors, each [N_g]

For each group g:
    1. apply_masks(x_groups[g], masks_groups[g])  -> [B_g, N_vis_g, D]
    2. encoder.forward(x_visible)                 -> [B_g, N_vis_g, D]
    3. Append result to output list

Output: List[Tensor[B_g, N_vis_g, D]]
```

### Implementation Sketch

```python
class MultiSequenceEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x_groups, masks_groups):
        """
        x_groups     : List[Tensor[B_g, N_g, D]]
        masks_groups : List[List[Tensor[N_g]]]   (bool, True=visible)
        Returns      : List[Tensor[B_g, N_vis_g, D]]
        """
        outputs = []
        for x, masks in zip(x_groups, masks_groups):
            x_vis = apply_masks(x, masks)   # [B_g, N_vis_g, D]
            out   = self.encoder(x_vis)     # [B_g, N_vis_g, D]
            outputs.append(out)
        return outputs
```

---

## MultiSequencePredictor

### Architecture

The predictor receives encoder outputs (`context_groups`) and must produce predictions
at the masked positions (`masks_pred_groups`).  It also needs the encoder mask
(`masks_enc_groups`) to know where in the full sequence the context tokens came from,
so it can construct the correct positional embeddings.

```
Input:  context_groups     -- List[Tensor[B_g, N_vis_g, D]]  (encoder output)
        masks_enc_groups   -- List[List[Tensor[N_g]]]
        masks_pred_groups  -- List[List[Tensor[N_g]]]

For each group g:
    1. predictor.forward(context, masks_enc, masks_pred) -> [B_g, N_pred_g, D]

Output: List[Tensor[B_g, N_pred_g, D]]
```

### Implementation Sketch

```python
class MultiSequencePredictor(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, context_groups, masks_enc_groups, masks_pred_groups):
        """
        context_groups    : List[Tensor[B_g, N_vis, D]]
        masks_enc_groups  : List[List[Tensor[N_g]]]
        masks_pred_groups : List[List[Tensor[N_g]]]
        Returns           : List[Tensor[B_g, N_pred, D]]
        """
        outputs = []
        for ctx, m_enc, m_pred in zip(
                context_groups, masks_enc_groups, masks_pred_groups):
            pred = self.predictor(ctx, m_enc, m_pred)  # [B_g, N_pred, D]
            outputs.append(pred)
        return outputs
```

---

## apply_masks Utility

`apply_masks(x, masks)` gathers only the *visible* tokens from the full token sequence.
This is the core operation enabling masked-encoder efficiency.

### Semantics

```
x     : Tensor[B, N, D]
masks : List[Tensor[N]]  -- one bool mask per sample; True = encoder-visible

Returns: Tensor[B, N_vis, D]  -- ragged if masks differ per sample
```

Because `N_vis` may differ per sample within a group, `apply_masks` assumes all masks
in a group have the same number of True values (enforced by the collator or by using
the same `MaskGenerator` for every sample in the group).

### Implementation

```python
def apply_masks(x, masks):
    """
    x     : [B, N, D]
    masks : List of bool Tensors each of shape [N]
    Returns [B, N_vis, D] where N_vis = masks[0].sum()
    """
    B, N, D = x.shape
    assert len(masks) == B, "One mask per sample required"
    n_vis = masks[0].long().sum().item()

    out = torch.zeros(B, n_vis, D, dtype=x.dtype, device=x.device)
    for i, m in enumerate(masks):
        out[i] = x[i][m]   # boolean index along sequence dim
    return out
```

For large batches a vectorised version using `torch.gather` is faster:

```python
def apply_masks_fast(x, masks):
    """Vectorised apply_masks using torch.gather."""
    B, N, D = x.shape
    # Stack masks: [B, N]
    M   = torch.stack(masks, dim=0).to(x.device)          # [B, N]
    idx = M.nonzero(as_tuple=False)                        # [K, 2]
    n_vis = M[0].sum().item()
    # Each row has same number of True entries (guaranteed by collator)
    vis_idx = idx[:, 1].view(B, n_vis)                    # [B, N_vis]
    vis_idx = vis_idx.unsqueeze(-1).expand(-1, -1, D)     # [B, N_vis, D]
    return torch.gather(x, 1, vis_idx)
```

---

## Variable-Length Batching Flow

Full data flow when `multi_fpc=True`:

```
Dataset.__getitem__
    -> (frames[T_i, H, W, C], label)   # T_i may vary per item

MaskCollator.__call__(batch)
    -> groups = {fpc: [items...]}
    -> for each fpc:
        collated = default_collate(fpc_items)   # [B_g, T_g, H, W, C]
        m_enc, m_pred = mask_gen_fpc(B_g)
    -> return [(collated_g, m_enc_g, m_pred_g) for each g]

Training loop:
    for batch_groups in loader:
        x_groups      = []
        masks_e_groups = []
        masks_p_groups = []
        for collated, m_e, m_p in batch_groups:
            x = PatchEmbed3D(collated['frames'])     # [B_g, N_g, D]
            x_groups.append(x)
            masks_e_groups.append(m_e)
            masks_p_groups.append(m_p)

        ctx_groups  = multi_enc(x_groups, masks_e_groups)
        pred_groups = multi_pred(ctx_groups, masks_e_groups, masks_p_groups)

        loss = 0
        for ctx, pred, x_full, m_p in zip(ctx_groups, pred_groups, x_groups, masks_p_groups):
            target = apply_masks(x_full, m_p)        # ground truth
            loss   += F.mse_loss(pred, target)
        loss.backward()
```

---

## Mixed-Resolution Training

The multi-sequence wrapper naturally supports clips at different spatial resolutions
within the same batch step.  Each FPC group is processed by its own `PatchEmbed3D`
instance configured for that group's `(T, H, W)`.  The resulting embedding sequences
have different `N` but the same `D`, so encoder weights are shared.

This enables curriculum training that starts with low-resolution short clips (cheaper)
and gradually increases resolution, all within a single codebase.

---

## Gradient Accumulation

Because each group is processed in a separate forward call, gradients accumulate
naturally.  No special handling is required:

```python
optimizer.zero_grad()
for batch_groups in loader:
    ctx_groups  = multi_enc(...)
    pred_groups = multi_pred(...)
    loss = compute_loss(pred_groups, ...)
    (loss / num_accum_steps).backward()
optimizer.step()
```

If `num_accum_steps > 1`, call `optimizer.step()` only every N batches, as usual.
