# AC Predictor: Token Interleaving, Block-Causal Mask, AC-RoPE

## Overview

The Action-Conditioned (AC) Predictor extends the standard V-JEPA 2 predictor to accept
robot action, proprioceptive state, and optional camera extrinsics as conditioning signals.
This document specifies the token layout, attention mask structure, and RoPE variant used.

---

## Token Interleaving

### Per-Frame Token Layout

For a sequence of T frames, each frame t produces a block of tokens arranged as:

```
Frame t tokens = [state_token_t, (extrinsics_token_t), action_token_t, visual_token_t_1, ..., visual_token_t_N]
```

Where N = H/patch_size * W/patch_size (number of spatial patches per frame).

**Token roles:**

| Token | Source | Dimensionality | Notes |
|---|---|---|---|
| `state_token` | Robot proprioceptive state | 7-DOF -> embed_dim | Linear projection of [x, y, z, roll, pitch, yaw, gripper] |
| `extrinsics_token` | Camera-to-world matrix | 6-DOF -> embed_dim | Flattened rotation (3) + translation (3), optional |
| `action_token` | Delta action command | 7-DOF -> embed_dim | Linear projection of delta [x, y, z, roll, pitch, yaw, gripper] |
| `visual_token_1..N` | Encoder output patches | embed_dim | From frozen ViT encoder |

**Non-visual tokens per frame (with extrinsics):** 3
**Non-visual tokens per frame (without extrinsics):** 2

### Full Sequence Layout

```
Total tokens = T * (num_non_visual + N)

Where num_non_visual = 3 if use_extrinsics else 2
```

For T=8 frames, N=196 patches, with extrinsics:
- Tokens per frame = 3 + 196 = 199
- Total tokens = 8 * 199 = 1592

### Implementation: Interleaving Logic

```python
def interleave_tokens(visual_tokens, action_tokens, state_tokens, extrinsics_tokens=None):
    """
    Args:
        visual_tokens: [B, T, N, D]  -- encoder output per frame
        action_tokens: [B, T, D]     -- projected action per frame
        state_tokens:  [B, T, D]     -- projected state per frame
        extrinsics_tokens: [B, T, D] or None

    Returns:
        interleaved: [B, T*(num_non_visual+N), D]
        action_token_mask: [B, T*(num_non_visual+N)] bool, True where action/state/extrinsics
    """
    B, T, N, D = visual_tokens.shape
    frame_blocks = []
    mask_blocks = []

    for t in range(T):
        block = [state_tokens[:, t:t+1, :]]     # [B, 1, D]
        mask = [True]

        if extrinsics_tokens is not None:
            block.append(extrinsics_tokens[:, t:t+1, :])
            mask.append(True)

        block.append(action_tokens[:, t:t+1, :])  # [B, 1, D]
        mask.append(True)

        block.append(visual_tokens[:, t, :, :])   # [B, N, D]
        mask.extend([False] * N)

        frame_blocks.append(torch.cat(block, dim=1))
        mask_blocks.extend(mask)

    interleaved = torch.cat(frame_blocks, dim=1)  # [B, T*(num_nv+N), D]
    action_token_mask = torch.tensor(mask_blocks, dtype=torch.bool)
    return interleaved, action_token_mask
```

---

## Block-Causal Attention Mask

### Semantics

The block-causal mask implements **frame-level causality** with **within-frame bidirectionality**:

- Tokens in frame t can attend to all tokens in frames 0 .. t (inclusive)
- Tokens in frame t CANNOT attend to tokens in frames t+1, t+2, ... T-1
- Within a single frame, all tokens attend to each other freely (bidirectional)

This is called "block-causal" because the causal boundary is at the frame level (block),
not at the individual token level.

### Visual Representation

For T=3 frames with tokens_per_frame=K:

```
Query\Key  | Frame 0 (K tokens) | Frame 1 (K tokens) | Frame 2 (K tokens)
-----------+--------------------+--------------------+--------------------
Frame 0    |        YES         |        NO          |        NO
Frame 1    |        YES         |        YES         |        NO
Frame 2    |        YES         |        YES         |        YES
```

Each "YES"/"NO" represents a K×K block of allowed/blocked attention.

### Implementation

```python
def build_block_causal_mask(T: int, tokens_per_frame: int, device) -> torch.Tensor:
    """
    Returns additive attention bias mask.
    -inf where attention is blocked, 0.0 where allowed.

    Shape: [T*tokens_per_frame, T*tokens_per_frame]
    """
    total = T * tokens_per_frame
    mask = torch.zeros(total, total, device=device)

    for t in range(T):
        # Tokens that are in frames > t: block them
        # Query range: [t * K, (t+1) * K)
        q_start = t * tokens_per_frame
        q_end = (t + 1) * tokens_per_frame

        # Key range to block: [(t+1) * K, T * K)
        k_start = (t + 1) * tokens_per_frame
        k_end = T * tokens_per_frame

        if k_start < k_end:
            mask[q_start:q_end, k_start:k_end] = float('-inf')

    return mask  # [total, total], add to attention logits before softmax
```

### PyTorch Integration

```python
# In attention forward pass:
# attn_weights: [B, num_heads, total, total]
attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
attn_weights = F.softmax(attn_weights, dim=-1)
```

### Note on Predictor-Only Causality

The `pred_is_frame_causal` flag controls whether the mask is applied:
- `True` (default for AC): block-causal mask applied -> autoregressive rollout
- `False` (standard JEPA): no causal mask -> bidirectional attention across all frames

---

## AC-RoPE: Action-Aware Rotary Position Embedding

### Motivation

Standard 3D-RoPE assigns spatial positions (depth=time, height, width) to all tokens.
For action/state/extrinsics tokens there is no meaningful spatial position — they represent
abstract signals, not image patches. Applying full 3-axis RoPE would impose a spurious
spatial structure on these tokens.

**AC-RoPE solution:** apply only the depth-axis (temporal) rotation to action/state/extrinsics
tokens, while visual tokens receive the full 3-axis rotation.

### Implementation

```python
def apply_ac_rope(q, k, rope_freqs, action_token_mask):
    """
    Args:
        q, k: [B, num_heads, total_tokens, head_dim]
        rope_freqs: [total_tokens, head_dim] -- precomputed frequencies
        action_token_mask: [total_tokens] bool -- True for action/state/extrinsics tokens

    Returns:
        q_rotated, k_rotated: same shape as input
    """
    # For visual tokens: apply full 3-axis RoPE
    visual_mask = ~action_token_mask
    q[:, :, visual_mask, :] = apply_rope_3d(q[:, :, visual_mask, :], rope_freqs[visual_mask])
    k[:, :, visual_mask, :] = apply_rope_3d(k[:, :, visual_mask, :], rope_freqs[visual_mask])

    # For action/state/extrinsics tokens: apply only depth-axis RoPE
    # Zero out height and width frequency components, keep only depth
    depth_only_freqs = rope_freqs.clone()
    depth_only_freqs[action_token_mask, head_dim//3:] = 0  # zero height + width components

    q[:, :, action_token_mask, :] = apply_rope_3d(q[:, :, action_token_mask, :], depth_only_freqs[action_token_mask])
    k[:, :, action_token_mask, :] = apply_rope_3d(k[:, :, action_token_mask, :], depth_only_freqs[action_token_mask])

    return q, k
```

### Frequency Assignment

For a sequence position p in frame t:
- All tokens in frame t share the same **depth coordinate** = t
- Visual token at patch (h, w): depth=t, height=h, width=w
- Action token at frame t: depth=t, height=0, width=0 (or not applied at all for h/w axes)

### Alternative Implementation: Masking Head Dimensions

Head dimensions are split into thirds for 3D-RoPE:
- dims 0..D/3-1: depth axis rotation
- dims D/3..2D/3-1: height axis rotation
- dims 2D/3..D-1: width axis rotation

For action tokens, only dims 0..D/3-1 are rotated; the rest are left unchanged.

---

## Forward Pass Summary

```
Input:
  context_repr: [B, T, N, D]       -- encoder output per frame
  actions:      [B, T, 7]          -- delta actions
  states:       [B, T, 7]          -- proprioceptive states
  extrinsics:   [B, T, 4, 4] opt  -- camera-to-world matrices

Step 1: Project conditioning signals
  action_emb    = action_embed(actions)     [B, T, D]
  state_emb     = state_embed(states)       [B, T, D]
  extrin_emb    = extrinsics_embed(extrin)  [B, T, D]  (if use_extrinsics)

Step 2: Interleave tokens
  tokens, action_mask = interleave_tokens(context_repr, action_emb, state_emb, extrin_emb)
  # tokens: [B, T*(num_nv+N), D]

Step 3: Project to predictor dimension
  tokens = input_proj(tokens)  [B, T*(num_nv+N), predictor_dim]

Step 4: Build block-causal mask (if pred_is_frame_causal)
  causal_mask = build_block_causal_mask(T, tokens_per_frame, device)

Step 5: Transformer blocks with AC-RoPE
  for block in transformer_blocks:
      tokens = block(tokens, causal_mask, action_mask)

Step 6: Extract visual token predictions
  # Only keep visual token positions, discard action/state/extrinsics tokens
  visual_preds = tokens[:, visual_positions, :]  [B, T*N, predictor_dim]

Output: visual_preds [B, T*N, predictor_dim]
```
