# ViT Variant Specifications — V-JEPA 2

Complete specifications for all Vision Transformer variants from Tiny to Gigantic,
including factory functions, parameter counts, and usage guidance.

---

## Variant Summary Table

| Variant     | embed_dim | depth | num_heads | mlp_ratio | head_dim | Approx Params |
|-------------|-----------|-------|-----------|-----------|----------|---------------|
| ViT-Tiny    | 192       | 12    | 3         | 4.0       | 64       | ~6M           |
| ViT-Small   | 384       | 12    | 6         | 4.0       | 64       | ~22M          |
| ViT-Base    | 768       | 12    | 12        | 4.0       | 64       | ~86M          |
| ViT-Large   | 1024      | 24    | 16        | 4.0       | 64       | ~307M         |
| ViT-Huge    | 1280      | 32    | 16        | 4.0       | 80       | ~632M         |
| ViT-Giant   | 1408      | 40    | 16        | 48/11     | 88       | ~1.1B         |
| ViT-Gigantic| 1664      | 48    | 16        | 64/11     | 104      | ~1.9B         |

Notes:
- `head_dim = embed_dim // num_heads`
- mlp_ratio expressed as float; `48/11 ≈ 4.363636...`, `64/11 ≈ 5.818...`
- For RoPE variants, `head_dim` must be divisible by 6 to allow 3-axis split
  - ViT-Tiny (head_dim=64): 64/6 = 10.66 — use padding or reduce to 60
  - ViT-Large (head_dim=64): same consideration; use d_dim=h_dim=w_dim=10, pad remaining 4
  - ViT-Huge (head_dim=80): 80/6 = 13.33 — use 12 per axis, pad 8
  - ViT-Giant (head_dim=88): 88/6 = 14.66 — use 14 per axis, pad 4
  - ViT-Gigantic (head_dim=104): 104/6 = 17.33 — use 16 per axis, pad 8

---

## Detailed Variant Specifications

### ViT-Tiny

```python
embed_dim  = 192
depth      = 12
num_heads  = 3
mlp_ratio  = 4.0
head_dim   = 64   # 192 // 3
mlp_hidden = 768  # 192 * 4.0
```

Use cases:
- Rapid prototyping and ablation studies
- CPU-feasible inference for research
- Smallest footprint in encoder-decoder setups
- Unit tests (use BrainAIConfig.minimal() analog)

Parameter breakdown (patch_size=16, img_size=224, no class token):
- Patch embed: 3*16*16 * 192 = 147,456
- Transformer blocks x12: ~5.5M
- Total: ~6M

---

### ViT-Small

```python
embed_dim  = 384
depth      = 12
num_heads  = 6
mlp_ratio  = 4.0
head_dim   = 64
mlp_hidden = 1536
```

Use cases:
- Balanced speed/quality for downstream fine-tuning
- Suitable for single-GPU training on medium datasets
- Good baseline for V-JEPA 2 pretraining ablations

---

### ViT-Base

```python
embed_dim  = 768
depth      = 12
num_heads  = 12
mlp_ratio  = 4.0
head_dim   = 64
mlp_hidden = 3072
```

Use cases:
- Standard benchmark model (ImageNet, Kinetics-400)
- Widely used V-JEPA-1 baseline size
- Fits in ~4GB VRAM at bfloat16 for typical batch sizes

---

### ViT-Large

```python
embed_dim  = 1024
depth      = 24
num_heads  = 16
mlp_ratio  = 4.0
head_dim   = 64
mlp_hidden = 4096
```

Use cases:
- Primary V-JEPA 2 context encoder variant
- Strong downstream performance on video benchmarks
- ~8-16GB VRAM requirement (bfloat16, batch=8, 16 frames)

---

### ViT-Huge

```python
embed_dim  = 1280
depth      = 32
num_heads  = 16
mlp_ratio  = 4.0
head_dim   = 80
mlp_hidden = 5120
```

Use cases:
- V-JEPA 2 target encoder at production scale
- Robotics control policies (AC-RoPE variant)
- Requires ~40GB VRAM for full precision

---

### ViT-Giant

```python
embed_dim  = 1408
depth      = 40
num_heads  = 16
mlp_ratio  = 48/11   # ≈ 4.3636...
head_dim   = 88
mlp_hidden = int(1408 * 48/11)  # = 6144
```

Notes on mlp_ratio:
- The fractional ratio `48/11` is a deliberate choice to hit a round MLP hidden dim:
  `1408 * 48/11 = 67584/11 ≈ 6144`
- In code: `mlp_hidden = round(embed_dim * mlp_ratio)` or explicitly set to 6144

Use cases:
- Large-scale video pretraining
- Requires gradient checkpointing + bfloat16
- Multi-GPU (DDP or FSDP) recommended

---

### ViT-Gigantic

```python
embed_dim  = 1664
depth      = 48
num_heads  = 16
mlp_ratio  = 64/11   # ≈ 5.8181...
head_dim   = 104
mlp_hidden = int(1664 * 64/11)  # ≈ 9672, round to 9664 for alignment
```

Notes:
- Largest V-JEPA 2 variant; requires 8+ A100 80GB for full batch
- Always use activation checkpointing
- Compile with `torch.compile()` for throughput gains

---

## Factory Functions

### Standard Factory Pattern

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ViTConfig:
    model_name: str
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
    use_sdpa: bool = True
    drop_path_rate: float = 0.0
    use_activation_checkpointing: bool = False
    compile_model: bool = False


def vit_tiny(**kwargs) -> ViTConfig:
    """ViT-Tiny: 192-dim, 12 layers, 3 heads. ~6M params."""
    cfg = dict(model_name="vit_tiny", embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0)
    cfg.update(kwargs)
    return ViTConfig(**cfg)


def vit_small(**kwargs) -> ViTConfig:
    """ViT-Small: 384-dim, 12 layers, 6 heads. ~22M params."""
    cfg = dict(model_name="vit_small", embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0)
    cfg.update(kwargs)
    return ViTConfig(**cfg)


def vit_base(**kwargs) -> ViTConfig:
    """ViT-Base: 768-dim, 12 layers, 12 heads. ~86M params."""
    cfg = dict(model_name="vit_base", embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0)
    cfg.update(kwargs)
    return ViTConfig(**cfg)


def vit_large(**kwargs) -> ViTConfig:
    """ViT-Large: 1024-dim, 24 layers, 16 heads. ~307M params."""
    cfg = dict(model_name="vit_large", embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0)
    cfg.update(kwargs)
    return ViTConfig(**cfg)


def vit_huge(**kwargs) -> ViTConfig:
    """ViT-Huge: 1280-dim, 32 layers, 16 heads. ~632M params."""
    cfg = dict(model_name="vit_huge", embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4.0)
    cfg.update(kwargs)
    return ViTConfig(**cfg)


def vit_giant(**kwargs) -> ViTConfig:
    """ViT-Giant: 1408-dim, 40 layers, 16 heads. ~1.1B params."""
    cfg = dict(
        model_name="vit_giant",
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48/11,       # intentional fractional ratio -> 6144 hidden
    )
    cfg.update(kwargs)
    return ViTConfig(**cfg)


def vit_gigantic(**kwargs) -> ViTConfig:
    """ViT-Gigantic: 1664-dim, 48 layers, 16 heads. ~1.9B params."""
    cfg = dict(
        model_name="vit_gigantic",
        embed_dim=1664,
        depth=48,
        num_heads=16,
        mlp_ratio=64/11,       # intentional fractional ratio
    )
    cfg.update(kwargs)
    return ViTConfig(**cfg)


VARIANT_REGISTRY = {
    "vit_tiny":     vit_tiny,
    "vit_small":    vit_small,
    "vit_base":     vit_base,
    "vit_large":    vit_large,
    "vit_huge":     vit_huge,
    "vit_giant":    vit_giant,
    "vit_gigantic": vit_gigantic,
}
```

---

## Parameter Count Estimation

Use this formula to estimate parameter counts before instantiation:

```
# Patch embedding
patch_embed_params = in_chans * patch_size^2 * embed_dim  (2D)
patch_embed_params = in_chans * tubelet * patch_size^2 * embed_dim  (3D video)

# Single transformer block
qkv_params    = 3 * embed_dim^2
proj_params   = embed_dim^2
mlp_params    = 2 * embed_dim * mlp_hidden  (standard MLP)
               = 3 * embed_dim * mlp_hidden  (SwiGLU: gate/up/down projections)
ln_params     = 4 * embed_dim  (2 LayerNorms, weight + bias each)
block_params  = qkv_params + proj_params + mlp_params + ln_params

# Total
total = patch_embed_params + depth * block_params
```

Example for ViT-Large (embed_dim=1024, depth=24, mlp_hidden=4096):
```
block = 3*1024^2 + 1024^2 + 2*1024*4096 + 4*1024
      = 3,145,728 + 1,048,576 + 8,388,608 + 4,096
      = 12,587,008

total = 24 * 12,587,008 + patch_embed ≈ 302M + 0.6M ≈ 302M
```
(Matches ~307M including bias terms and positional embedding.)

---

## Usage Guidance

### Choosing a Variant

| Task                              | Recommended Variant | Notes                              |
|-----------------------------------|--------------------|------------------------------------|
| Unit tests / CI                   | ViT-Tiny           | Fast, low memory                   |
| Research ablations                | ViT-Small, Base    | Reasonable cost                    |
| Video classification benchmarks   | ViT-Large          | Standard V-JEPA 2 setting          |
| Robotics / embodied AI            | ViT-Huge (AC-RoPE) | Action conditioning support        |
| Production video understanding    | ViT-Giant/Gigantic | Multi-GPU required                 |

### RoPE Compatibility

Not all variants have head_dim divisible by 6 for perfect 3-axis split:
- ViT-Tiny/Small/Base/Large (head_dim=64): Use `d_dim=h_dim=w_dim=10`, remaining 4 dims not rotated
- ViT-Huge (head_dim=80): Use 12 per axis, 8 dims bypass RoPE
- Always use `2 * floor(head_dim / 6)` formula from rope-3axis.md

### SwiGLU with Giant/Gigantic

Giant uses fractional mlp_ratio by design:
```python
# Giant: embed_dim=1408, mlp_ratio=48/11
raw_hidden  = 1408 * (48/11)  # = 6144.0 (exact)
# No rounding needed; use wide_silu=True for SwiGLU variant
# wide_silu: hidden = ceil(int(2 * raw_hidden / 3) / 8) * 8
#          = ceil(4096 / 8) * 8 = 4096
```

### Activation Checkpointing

Enable for Giant and Gigantic by default:
```python
cfg = vit_giant(use_activation_checkpointing=True)
```

All blocks use `torch.utils.checkpoint.checkpoint()` in `forward()` when this flag is set.

### Drop Path Scheduling

Use linearly increasing drop path rates across layers:
```python
import torch
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
# Block i gets dpr[i]
```

Recommended rates:
- ViT-Tiny/Small: 0.0 (no stochastic depth needed)
- ViT-Base/Large: 0.1
- ViT-Huge: 0.2
- ViT-Giant/Gigantic: 0.3
