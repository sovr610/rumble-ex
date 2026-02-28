---
name: V-JEPA 2 Vision Transformer
description: >
  This skill should be used when the user asks to "implement ViT for V-JEPA",
  "create vision transformer", "add RoPE to transformer", "implement SwiGLU",
  "configure ViT variant", "add cross-attention", "implement attentive pooler",
  "weight initialization for ViT", "sinusoidal positional embeddings",
  "3-axis rotary position embeddings", "transformer block with drop path",
  "multi-head self-attention", "patch embedding", or needs guidance on
  Vision Transformer architecture, positional encoding strategies, or
  transformer building blocks for V-JEPA 2.
version: 0.1.0
---

# V-JEPA 2 Vision Transformer

## Overview

Guide implementation of the Vision Transformer (ViT) architecture used in V-JEPA 2, Meta FAIR's self-supervised video foundation model. Cover all ViT variants (Tiny to Gigantic, 192-1664 embed_dim), transformer block implementations (standard, RoPE, AC-RoPE), feed-forward networks (MLP and SwiGLU), attention mechanisms (vanilla, RoPE, cross-attention), positional encodings (sincos and 3-axis RoPE), weight initialization, and the attentive pooler for downstream probing.

## Public Contract

### VisionTransformer

Main encoder processing video patches into latent representations.

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, tubelet_size=2, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0,
                 use_rope=False, use_silu=False, wide_silu=False, use_sdpa=True,
                 drop_path_rate=0.0, use_activation_checkpointing=False): ...
    def forward(self, x: Tensor, masks: Optional[List[Tensor]] = None,
                out_layers: Optional[List[int]] = None) -> Tensor: ...
    def interpolate_pos_encoding(self, x: Tensor, pos_embed: Tensor) -> Tensor: ...
```

### TransformerBlock

Standard transformer block with optional RoPE.

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0,
                 use_rope=False, use_silu=False, wide_silu=False): ...
    def forward(self, x, rope_freqs=None): ...
```

### Attention Variants

```python
class Attention(nn.Module):       # Vanilla multi-head self-attention
class RoPEAttention(nn.Module):   # 3-axis RoPE attention (frame/height/width)
class CrossAttention(nn.Module):  # Separate Q and KV projections for pooling
```

### PositionalEncoding

```python
def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, uniform_power=False): ...
def get_2d_sincos_pos_embed(embed_dim, grid_size): ...
class RoPE3D:  # 3-axis rotary embeddings with (depth, height, width) decomposition
```

## Key Concepts

### ViT Variant Specifications

| Variant | embed_dim | depth | num_heads | mlp_ratio | Params |
|---------|-----------|-------|-----------|-----------|--------|
| ViT-Large | 1024 | 24 | 16 | 4 | ~300M |
| ViT-Huge | 1280 | 32 | 16 | 4 | ~600M |
| ViT-Giant | 1408 | 40 | 16 | 48/11 | ~1B |

### Transformer Block Pipeline

```
Input -> LayerNorm -> Multi-Head Attention -> Residual + DropPath
      -> LayerNorm -> FFN (MLP or SwiGLU) -> Residual + DropPath -> Output
```

### SwiGLU Feed-Forward Network

Gated activation: `SiLU(W1 * x) * W2 * x` with hidden dim aligned to multiples of 8.
When `wide_silu=True`: `hidden = int(2 * hidden / 3)` rounded up to next multiple of 8.

### 3-Axis RoPE Decomposition

Head dimension split into `d_dim`, `h_dim`, `w_dim` (each = `2 * floor(head_dim / 6)`).
Applied independently to depth (temporal), height, and width axes.
Rotation formula: split `[..., D]` into pairs of 2, rotate each pair by position-dependent angle.

### Sinusoidal Positional Embeddings

3D sincos: decomposes embedding into temporal + spatial components.
Default: 50% dims for depth, 25% each for height/width.
`uniform_power=True`: equal 33% allocation.
Stored as `nn.Parameter(requires_grad=False)` — not learnable.

### Weight Initialization

- **Truncated normal**: Custom inverse CDF method, default bounds `[-2, 2]`
- **Block rescaling**: Attention output and MLP output weights divided by `sqrt(2 * layer_id)` to prevent signal explosion
- **Sincos pos embed**: Initialized once and frozen
- **Zero-initialized mask tokens**: Default for predictor target tokens

### Position Embedding Interpolation

- **Video**: `trilinear` interpolation for `(T, H, W)` reshaping
- **Image**: `bicubic` interpolation for `(H, W)` reshaping
- Enables models trained at 16 frames / 256px to evaluate at 64 frames / 384px

### Patch Embedding

- `PatchEmbed`: 2D Conv for images
- `PatchEmbed3D`: 3D Conv with tubelet for video — `Conv3d(kernel=(tubelet_size, patch_size, patch_size))`
- Total tokens: `(T/t) * (H/P) * (W/P)`

### Cross-Attention and Attentive Pooler

Separate Q (learnable query tokens) and KV (encoder output) projections.
Used by `AttentivePooler(num_queries=1)` -> `nn.Linear(embed_dim, num_classes)` for downstream probing.

## Configuration Surface

```python
@dataclass
class ViTConfig:
    model_name: str = "vit_large"        # Factory function name
    img_size: int = 224
    patch_size: int = 16
    tubelet_size: int = 2
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    use_rope: bool = False
    use_silu: bool = False
    wide_silu: bool = False
    use_sdpa: bool = True                # PyTorch 2.0 fused attention
    drop_path_rate: float = 0.0
    use_activation_checkpointing: bool = False
    compile_model: bool = False
```

## Done-When Gates

1. **ViT Forward** — `VisionTransformer.forward()` produces correct output shape `[B, N, D]` for video input `[B, C, T, H, W]`; masking reduces sequence length correctly.
2. **RoPE Correctness** — `RoPE3D` produces rotationally-equivariant attention; outputs differ when positions change.
3. **Interpolation** — Model trained at 256px produces valid outputs at 384px via positional embedding interpolation.

## Failure Modes

| Mode | Symptom | Fix |
|------|---------|-----|
| OOM on Giant model | Crash on forward | Enable activation_checkpointing + bfloat16 |
| RoPE dimension mismatch | Shape error in attention | Verify head_dim divisible by 6 for 3-axis split |
| SwiGLU hidden dim wrong | Unexpected parameter count | Ensure alignment to multiples of 8 |
| Pos embed interpolation artifacts | Quality drop at new resolution | Use trilinear for video, bicubic for images |

## Resources

### Reference Files
- **`references/vit-variants.md`** — Complete variant specs, factory functions, parameter counts
- **`references/rope-3axis.md`** — 3-axis RoPE math, frequency generation, rotation implementation
- **`references/attention-mechanisms.md`** — Vanilla, RoPE, AC-RoPE, Cross-Attention, SDPA
- **`references/weight-initialization.md`** — Truncated normal, block rescaling, sincos generation
- **`references/testing-matrix.md`** — Test scenarios for ViT infrastructure

### Asset Files
- **`assets/vision_transformer_template.py`** — VisionTransformer with patch embedding, masking, self-tests
- **`assets/transformer_block_template.py`** — Block, Attention, RoPEAttention, SwiGLU, DropPath
- **`assets/positional_encoding_template.py`** — Sincos 2D/3D, RoPE3D, interpolation utilities
- **`assets/cross_attention_template.py`** — CrossAttention, AttentivePooler for probing
- **`assets/vit_config_template.py`** — ViTConfig, factory functions, variant presets

### Scripts
- **`scripts/validate_vit.py`** — Validates done-when gates
- **`scripts/gen_vit_tests.py`** — Generates 100+ pytest test cases
- **`scripts/vit_benchmark.py`** — Throughput and memory benchmarks per variant
