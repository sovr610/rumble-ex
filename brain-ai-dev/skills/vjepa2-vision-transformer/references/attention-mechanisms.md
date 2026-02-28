# Attention Mechanisms — V-JEPA 2

Complete specification for all attention variants used in V-JEPA 2:
Vanilla MHSA, RoPE Attention, AC-RoPE Attention, Cross-Attention, and SDPA integration.

---

## 1. Vanilla Multi-Head Self-Attention (MHSA)

### Mathematical Formulation

```
Q = x * W_q + b_q   ∈ R^[B, N, D]
K = x * W_k + b_k   ∈ R^[B, N, D]
V = x * W_v + b_v   ∈ R^[B, N, D]

Reshape: Q, K, V -> [B, H, N, D/H]  where H = num_heads, D/H = head_dim

Attention(Q, K, V) = softmax(Q * K^T / sqrt(head_dim)) * V

Output = concat(head_1, ..., head_H) * W_proj + b_proj
```

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Attention(nn.Module):
    """
    Vanilla multi-head self-attention.

    Optionally uses PyTorch 2.0 F.scaled_dot_product_attention (SDPA)
    for fused, memory-efficient computation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_sdpa: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_sdpa = use_sdpa

        # Single fused QKV projection (3x more efficient than separate)
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,                    # [B, N, D]
        attn_mask: Optional[torch.Tensor] = None,  # [B, H, N, N] or [N, N]
    ) -> torch.Tensor:
        B, N, D = x.shape

        # Project and reshape to multi-head format
        qkv = self.qkv(x)                   # [B, N, 3*D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)    # [3, B, H, N, head_dim]
        q, k, v = qkv.unbind(0)              # each [B, H, N, head_dim]

        if self.use_sdpa:
            # PyTorch 2.0 fused implementation: handles scaling, softmax, dropout
            # Supports FlashAttention-2 on CUDA when available
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
            )
        else:
            # Manual implementation for debugging / non-CUDA devices
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v                                    # [B, H, N, head_dim]

        # Merge heads and project
        x = x.transpose(1, 2).reshape(B, N, D)             # [B, N, D]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

---

## 2. RoPE Attention

Applies 3-axis rotary embeddings to Q and K before computing attention.
V values are never rotated (they carry content, not position).

### Key Difference from Vanilla

```
Q_rope = apply_3d_rope(Q, grid_depth, grid_h, grid_w)
K_rope = apply_3d_rope(K, grid_depth, grid_h, grid_w)
Attention(Q_rope, K_rope, V)  # same formula as vanilla after rotation
```

### Implementation

```python
class RoPEAttention(nn.Module):
    """
    Multi-head self-attention with 3-axis Rotary Position Embeddings.

    Q and K are rotated per the temporal (depth), height, and width axes.
    Values V are left unchanged.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_sdpa: bool = True,
        theta: float = 10000.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_sdpa = use_sdpa

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # RoPE encoder — generates and caches rotation frequencies
        from .positional_encoding_template import RoPE3D
        self.rope = RoPE3D(head_dim=self.head_dim, theta=theta)

    def forward(
        self,
        x: torch.Tensor,               # [B, N, D]  N = T*H*W
        rope_grid: tuple,              # (grid_depth, grid_h, grid_w)
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        grid_depth, grid_h, grid_w = rope_grid

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # [3, B, H, N, head_dim]
        q, k, v = qkv.unbind(0)

        # Apply RoPE rotation to Q and K
        q, k = self.rope.apply_rope(q, k, grid_depth, grid_h, grid_w)

        if self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

---

## 3. AC-RoPE (Action-Conditioned RoPE Attention)

Used in V-JEPA 2's robotics variant. Action tokens are appended to the video sequence.
They receive only depth-axis rotation (temporal identity, no spatial position).

### Sequence Structure

```
Full sequence: [video_tokens (T*H*W), action_tokens (A)]
              = N_total tokens

Video tokens receive: full 3-axis rotation (d, h, w)
Action tokens receive: depth-axis rotation only (d)
```

### Implementation

```python
class ACRoPEAttention(nn.Module):
    """
    Action-Conditioned RoPE Attention for video + action token sequences.

    Action tokens are assumed to be at the END of the sequence.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_sdpa: bool = True,
        theta: float = 10000.0,
        n_action_tokens: int = 0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_sdpa = use_sdpa
        self.n_action_tokens = n_action_tokens

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        from .positional_encoding_template import RoPE3D
        self.rope = RoPE3D(head_dim=self.head_dim, theta=theta)

    def forward(
        self,
        x: torch.Tensor,        # [B, N_total, D]
        rope_grid: tuple,       # (grid_depth, grid_h, grid_w)
        action_frame_ids: Optional[torch.Tensor] = None,  # [A] frame indices for actions
    ) -> torch.Tensor:
        B, N_total, D = x.shape
        grid_depth, grid_h, grid_w = rope_grid
        N_video = grid_depth * grid_h * grid_w
        N_action = self.n_action_tokens

        qkv = self.qkv(x).reshape(B, N_total, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # each [B, H, N_total, head_dim]

        # Split video and action tokens
        q_vid, q_act = q[..., :N_video, :], q[..., N_video:, :]
        k_vid, k_act = k[..., :N_video, :], k[..., N_video:, :]

        # Full 3-axis RoPE for video tokens
        q_vid, k_vid = self.rope.apply_rope(q_vid, k_vid, grid_depth, grid_h, grid_w)

        # Depth-only RoPE for action tokens
        if N_action > 0:
            if action_frame_ids is None:
                # Default: assign action tokens to last frames
                action_frame_ids = torch.arange(
                    grid_depth - N_action, grid_depth,
                    device=x.device
                )
            ad = self.rope.axis_dim
            freqs = self.rope.freqs.to(x.device)
            ang = torch.outer(action_frame_ids.float(), freqs)  # [A, ad//2]
            ang = ang.unsqueeze(0).unsqueeze(0)                  # [1, 1, A, ad//2]

            from .rope_3axis import apply_rope_1d
            q_act_d = apply_rope_1d(q_act[..., :ad], ang)
            k_act_d = apply_rope_1d(k_act[..., :ad], ang)
            q_act = torch.cat([q_act_d, q_act[..., ad:]], dim=-1)
            k_act = torch.cat([k_act_d, k_act[..., ad:]], dim=-1)

        # Reassemble full sequence
        q = torch.cat([q_vid, q_act], dim=2)
        k = torch.cat([k_vid, k_act], dim=2)

        if self.use_sdpa:
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N_total, D)
        x = self.proj(x)
        return x
```

---

## 4. Cross-Attention

Cross-attention uses separate Q (from decoder / query tokens) and KV (from encoder output).
Used in the AttentivePooler for downstream probing.

### Mathematical Formulation

```
Q = queries * W_q           ∈ R^[B, Nq, D]  (learnable query tokens)
K = encoder_out * W_k       ∈ R^[B, Nkv, D] (ViT encoder output)
V = encoder_out * W_v       ∈ R^[B, Nkv, D]

Attention(Q, K, V) = softmax(Q * K^T / sqrt(head_dim)) * V  ∈ R^[B, Nq, D]
```

### Implementation

```python
class CrossAttention(nn.Module):
    """
    Cross-attention where queries come from one source and keys/values from another.

    Typical use:
        queries  = learnable tokens in AttentivePooler
        encoder  = output of ViT backbone
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_sdpa: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_sdpa = use_sdpa

        # Separate Q projection (queries) vs KV projection (encoder output)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        queries: torch.Tensor,   # [B, Nq, D]  — decoder side
        encoder: torch.Tensor,   # [B, Nkv, D] — encoder output
    ) -> torch.Tensor:
        B, Nq, D = queries.shape
        Nkv = encoder.shape[1]

        # Project queries
        q = self.q(queries)                              # [B, Nq, D]
        q = q.reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # q: [B, H, Nq, head_dim]

        # Project keys and values from encoder
        kv = self.kv(encoder)                            # [B, Nkv, 2*D]
        kv = kv.reshape(B, Nkv, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)                  # [2, B, H, Nkv, head_dim]
        k, v = kv.unbind(0)

        if self.use_sdpa:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, Nq, Nkv]
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v                                   # [B, H, Nq, head_dim]

        out = out.transpose(1, 2).reshape(B, Nq, D)         # [B, Nq, D]
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
```

---

## 5. SDPA Integration Details

`F.scaled_dot_product_attention` (PyTorch 2.0+) automatically selects the best kernel:

| Condition                     | Kernel Used              | Notes                        |
|-------------------------------|--------------------------|------------------------------|
| CUDA + small sequences        | FlashAttention-2         | Fastest, O(N) memory         |
| CUDA + large sequences        | Memory-efficient attn    | Still O(N) memory            |
| CPU                           | Math (reference)         | No speedup vs manual         |
| `attn_mask` is bool tensor    | Applies as additive -inf | Supports causal masks        |
| `attn_mask` is float tensor   | Added to attn logits     | Supports soft masking        |

### Usage Pattern

```python
# Boolean mask (True = keep, False = mask out)
bool_mask = torch.ones(B, 1, N, N, dtype=torch.bool, device=x.device)
# Set future positions to False for causal masking
bool_mask = torch.tril(bool_mask)

# SDPA converts bool mask internally
out = F.scaled_dot_product_attention(q, k, v, attn_mask=bool_mask)

# Additive mask (0 = keep, -inf = mask out)
additive_mask = torch.zeros(B, 1, N, N, device=x.device)
additive_mask[:, :, :, masked_positions] = float('-inf')

out = F.scaled_dot_product_attention(q, k, v, attn_mask=additive_mask)
```

### SDPA vs Manual — When to Use Manual

Use the manual implementation when:
1. Debugging attention patterns (need to inspect the attn matrix)
2. Running on MPS (Apple Silicon) where some SDPA paths may differ
3. Implementing custom attention modifications (relative bias, ALiBi, etc.)
4. Python 3.9 / PyTorch 1.x environments

---

## Attention in the Block Pipeline

```
x ──► LayerNorm ──► Attention ──► DropPath ──► + ──► x
                                               ▲
                                               │ (residual)
                                               x

x ──► LayerNorm ──► FFN ──────► DropPath ──► + ──► x
```

### Pre-norm vs Post-norm

V-JEPA 2 uses **pre-norm** (LayerNorm before attention/FFN).
Post-norm (original Transformer) is less stable at large scale.

```python
# Pre-norm (V-JEPA 2 style)
x = x + drop_path(attn(norm1(x)))
x = x + drop_path(ffn(norm2(x)))

# Post-norm (original, NOT used in V-JEPA 2)
x = norm1(x + drop_path(attn(x)))
x = norm2(x + drop_path(ffn(x)))
```

---

## Masked Attention for Masked Patch Modeling

In V-JEPA 2's pretraining, context tokens (unmasked) attend to each other.
Target tokens are predicted from context via a predictor network.

```python
def apply_masks(x: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
    """
    Select unmasked tokens from the full token sequence.

    Args:
        x: [B, N, D] full token sequence
        masks: List of [B, n_keep] index tensors

    Returns:
        x_masked: [B*len(masks), n_keep, D]
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.shape[-1])  # [B, n_keep, D]
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)
```
