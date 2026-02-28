# Predictor Architecture Reference

## VisionTransformerPredictor

The predictor is a lightweight transformer that takes context representations and mask tokens as input, and outputs predictions for the masked positions. Its design is critical to JEPA: it must be expressive enough to make good predictions, but not so powerful that it can memorize without learning.

### Architecture Diagram

```
Input: context_repr [B, N_vis, D_enc]  +  masks_enc, masks_pred (position indices)

                    ┌────────────────────────────────────────────┐
                    │           VisionTransformerPredictor        │
                    │                                            │
  context_repr ─────┤  input_proj (Linear D_enc -> D_pred)      │
                    │        │                                   │
                    │        ▼                                   │
                    │  context_proj [B, N_vis, D_pred]          │
                    │        │                                   │
  mask_tokens ──────┤  Lookup + add positional embeddings       │
  (learnable)       │        │                                   │
                    │        ▼                                   │
                    │  target_tokens [B, N_pred, D_pred]        │
                    │        │                                   │
                    │  Concatenate context + target tokens       │
                    │        │                                   │
                    │  [B, N_vis + N_pred, D_pred]              │
                    │        │                                   │
                    │  Sort by position index (argsort)         │
                    │        │                                   │
                    │  Transformer Blocks x depth               │
                    │  (MHSA + FFN, pre-norm)                   │
                    │        │                                   │
                    │  Un-sort (inverse permutation)            │
                    │        │                                   │
                    │  Extract target positions [B, N_pred, D_pred]│
                    │        │                                   │
                    │  output_proj (Linear D_pred -> D_enc)     │
                    └────────────────────┬───────────────────────┘
                                         │
                                         ▼
                             pred_repr [B, N_pred, D_enc]
```

---

## Default Hyperparameters

| Parameter | Default Value | Notes |
|-----------|--------------|-------|
| `embed_dim` | 1024 | Must match context encoder output dim |
| `predictor_embed_dim` | 384 | Internal predictor dimension (smaller than encoder) |
| `depth` | 12 | Number of transformer blocks |
| `num_heads` | 12 | Attention heads (384 / 12 = 32 head_dim) |
| `num_targets` | 10 | Number of learnable mask tokens |
| `mlp_ratio` | 4.0 | FFN expansion factor |
| `qkv_bias` | True | QKV projection bias |
| `drop` | 0.0 | Dropout (typically 0 for SSL) |
| `attn_drop` | 0.0 | Attention dropout |

Why is predictor smaller than encoder? The predictor operates in a lower-dimensional space (384 vs 1024) to enforce an information bottleneck. The encoder must pack all useful information into a compact representation; the predictor cannot simply copy features.

---

## Forward Pass: Step-by-Step

### Step 1: Input Projection

```python
# Project context representations from encoder space to predictor space
context_proj = self.input_proj(context_repr)  # [B, N_vis, D_pred]
```

The input projection adapts the encoder's embedding dimension to the predictor's smaller internal dimension. This projection has a weight decay applied during training.

### Step 2: Mask Token Preparation

```python
# self.mask_tokens: nn.Parameter of shape [1, num_targets, D_pred]
# Expand to batch size, then add positional embeddings for target positions
B, N_vis, _ = context_repr.shape
N_pred = masks_pred[0].numel()  # Number of masked positions

# Repeat mask tokens across batch
# Then add positional embeddings indexed by masks_pred
mask_tokens = self.mask_tokens.expand(B, -1, -1)  # [B, num_targets, D_pred]
# Typically: select/expand as needed based on N_pred
```

**Mask tokens are learnable parameters**, not zeros. They provide the predictor with a learned "I don't know" signal at masked positions. The positional embedding for each mask token is set to the positional embedding of the patch it is predicting.

### Step 3: Concatenation

```python
# Combine visible context tokens with mask tokens
# context_proj:  [B, N_vis, D_pred]  — encoded visible patches
# target_tokens: [B, N_pred, D_pred] — mask tokens at target positions
x = torch.cat([context_proj, target_tokens], dim=1)  # [B, N_vis + N_pred, D_pred]
```

All tokens — both context and mask — are concatenated into a single sequence. This allows the transformer's attention mechanism to let mask tokens attend to context tokens and vice versa.

### Step 4: Sort by Position Index

```python
# Combine position indices for context and target
# masks_enc[0]: [N_vis] — original position indices of visible patches
# masks_pred[0]: [N_pred] — original position indices of masked patches
all_positions = torch.cat([masks_enc[0], masks_pred[0]], dim=0)  # [N_vis + N_pred]

# Sort to restore spatial/temporal order
sort_idx = all_positions.argsort()  # ascending sort by position
x_sorted = x[:, sort_idx, :]  # [B, N_vis + N_pred, D_pred]

# Save inverse permutation for un-sorting
unsort_idx = sort_idx.argsort()
```

**Why sort?** Transformers with positional embeddings work best when tokens are presented in their natural order. Sorting ensures that relative positional relationships (patch A is to the left of patch B) are faithfully represented in the attention computation.

Without sorting, a mask token at position 100 could appear at index 5 in the sequence, confusing the positional encoding scheme.

### Step 5: Transformer Processing

```python
for block in self.blocks:
    x_sorted = block(x_sorted)  # Standard ViT block: LayerNorm + MHSA + LayerNorm + FFN
```

Standard pre-norm ViT transformer blocks with:
- `LayerNorm` before self-attention
- Multi-head self-attention (MHSA) with QKV projections
- Residual connection
- `LayerNorm` before FFN
- FFN: `Linear -> GELU -> Linear` with expansion `mlp_ratio`
- Residual connection

All mask tokens can attend to all context tokens (and other mask tokens). This global attention is crucial — the predictor must integrate information from distant context patches.

### Step 6: Un-sort

```python
# Restore original order (context first, then targets)
x_unsorted = x_sorted[:, unsort_idx, :]  # [B, N_vis + N_pred, D_pred]
```

Un-sorting restores the concatenated sequence to `[context_tokens | target_tokens]` order, making it easy to extract target predictions by slicing.

### Step 7: Extract Target Predictions

```python
# Extract only the target (mask token) positions
# Context tokens are at indices [0 : N_vis], target tokens at [N_vis :]
pred_tokens = x_unsorted[:, N_vis:, :]  # [B, N_pred, D_pred]
```

Only the tokens corresponding to masked positions contribute to the loss. Context token outputs are discarded.

### Step 8: Output Projection

```python
# Project back to encoder embedding dimension
pred_repr = self.output_proj(pred_tokens)  # [B, N_pred, D_enc]
```

The output projection maps predictions back to the encoder's space so they can be compared to the target encoder's output. This is where the loss is computed.

---

## Complete Reference Implementation

```python
import torch
import torch.nn as nn
from functools import partial
from typing import List


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention (uses Flash Attention when available)
        try:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )
        except Exception:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int,
                 out_features: int, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden, dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerPredictor(nn.Module):
    """
    Predicts masked patch representations from visible context.

    Args:
        embed_dim: Encoder output dimension (predictor input/output dim)
        predictor_embed_dim: Internal predictor dimension (typically smaller)
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        num_targets: Number of learnable mask tokens
        mlp_ratio: FFN expansion ratio
        qkv_bias: Add bias to QKV projections
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        predictor_embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        num_targets: int = 10,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.num_targets = num_targets

        # Projection into predictor space
        self.input_proj = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Learnable mask tokens (one per target slot)
        self.mask_tokens = nn.Parameter(
            torch.zeros(1, num_targets, predictor_embed_dim)
        )
        nn.init.trunc_normal_(self.mask_tokens, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = norm_layer(predictor_embed_dim)

        # Projection back to encoder space
        self.output_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal (ViT standard)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        context_repr: torch.Tensor,
        masks_enc: List[torch.Tensor],
        masks_pred: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            context_repr: [B, N_vis, embed_dim] — encoder output for visible patches
            masks_enc:    List of [N_vis] tensors — position indices of visible patches
            masks_pred:   List of [N_pred] tensors — position indices of masked patches

        Returns:
            pred_repr: [B, N_pred, embed_dim] — predictions for masked positions
        """
        B, N_vis, D = context_repr.shape
        device = context_repr.device

        # Step 1: Project context to predictor embedding space
        x_context = self.input_proj(context_repr)  # [B, N_vis, D_pred]

        # Step 2: Prepare mask tokens with positional embeddings
        # Use the first mask set (support single mask set; extend for multiple)
        m_enc = masks_enc[0]    # [N_vis]
        m_pred = masks_pred[0]  # [N_pred]
        N_pred = m_pred.shape[0]

        # Expand mask tokens to batch and trim/pad to N_pred
        # mask_tokens: [1, num_targets, D_pred] -> [B, N_pred, D_pred]
        if N_pred <= self.num_targets:
            target_tokens = self.mask_tokens[:, :N_pred, :].expand(B, -1, -1)
        else:
            # More targets than mask token slots — tile
            repeats = (N_pred + self.num_targets - 1) // self.num_targets
            target_tokens = self.mask_tokens.expand(B, -1, -1).repeat(1, repeats, 1)
            target_tokens = target_tokens[:, :N_pred, :]

        target_tokens = target_tokens.contiguous()  # [B, N_pred, D_pred]

        # Step 3: Concatenate context + target tokens
        x = torch.cat([x_context, target_tokens], dim=1)  # [B, N_vis + N_pred, D_pred]

        # Step 4: Sort by original position index
        all_positions = torch.cat([m_enc, m_pred], dim=0)  # [N_vis + N_pred]
        sort_idx = all_positions.argsort()                   # ascending by patch index
        unsort_idx = sort_idx.argsort()                      # inverse permutation

        x = x[:, sort_idx, :]  # [B, N_vis + N_pred, D_pred]

        # Step 5: Process through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Step 6: Un-sort to restore [context | target] ordering
        x = x[:, unsort_idx, :]  # [B, N_vis + N_pred, D_pred]

        # Step 7: Extract target (mask token) predictions
        pred_tokens = x[:, N_vis:, :]  # [B, N_pred, D_pred]

        # Step 8: Project back to encoder embedding space
        pred_repr = self.output_proj(pred_tokens)  # [B, N_pred, D_enc]

        return pred_repr
```

---

## Token Ordering: Why It Matters

### Scenario Without Sorting

```
Patch positions in video: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

masks_enc  = [0, 2, 4, 6, 8]   # visible
masks_pred = [1, 3, 5, 7, 9]   # masked

After concatenation (no sort):
  Sequence = [pos0, pos2, pos4, pos6, pos8, pos1, pos3, pos5, pos7, pos9]
  Positional embeddings do NOT match true spatial layout!
  Mask token at index 5 (sequence position) has embedding for pos1
  but attention treats it as "sequence position 5" → confusion
```

### Scenario With Sorting

```
After sort by position:
  Sequence = [pos0, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9]
  Positional embeddings exactly match spatial positions
  Attention can use spatial/temporal structure correctly
```

Sorting is especially important for spatiotemporal video data where temporal ordering and spatial adjacency carry strong semantic meaning.

---

## Multiple Mask Sets

V-JEPA 2 often uses multiple mask sets (multiple prediction targets per sample for training efficiency):

```python
# Multiple mask sets: each element is one mask configuration
# masks_enc:  [masks_enc_0, masks_enc_1, ...]  — one per target
# masks_pred: [masks_pred_0, masks_pred_1, ...]

# Process each mask set, accumulate loss
total_loss = 0.0
for m_enc, m_pred in zip(masks_enc, masks_pred):
    pred = predictor(context_repr, [m_enc], [m_pred])
    target = gather(target_repr, m_pred)
    total_loss += smooth_l1_loss(pred, target)

loss = total_loss / len(masks_enc)
```

The reference implementation above handles the single-mask case. For multiple masks, loop over the mask list and average losses.

---

## Positional Embedding Interaction

The predictor does not have its own positional embedding table — it inherits position information from:
1. The mask tokens' initial values (which can be indexed by position)
2. The sort-by-position step, which aligns tokens with the standard positional ordering

In the full V-JEPA 2 implementation, the encoder's positional embeddings (RoPE or sinusoidal) are used when assembling mask tokens:
```python
# Add encoder's positional embedding for mask token positions
pos_emb = encoder.pos_embed[:, m_pred, :]  # [B, N_pred, D_enc]
# Project to predictor space
pos_emb_pred = pos_proj(pos_emb)            # [B, N_pred, D_pred]
target_tokens = target_tokens + pos_emb_pred
```

This ensures mask tokens "know" which spatial-temporal location they are trying to predict.

---

## Parameter Count Estimates

| Configuration | Params (approx) |
|--------------|----------------|
| embed_dim=768, pred_dim=384, depth=12, heads=12 | ~22M |
| embed_dim=1024, pred_dim=384, depth=12, heads=12 | ~22M |
| embed_dim=1024, pred_dim=384, depth=6, heads=12  | ~11M |
| embed_dim=1024, pred_dim=768, depth=12, heads=12 | ~85M |

The predictor is intentionally lightweight (10-25M params) compared to the encoder (300M+ params). This asymmetry is a key feature — too powerful a predictor can "short-circuit" learning.
