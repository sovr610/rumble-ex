# Weight Initialization — V-JEPA 2

Complete specification for all weight initialization strategies used in V-JEPA 2,
including truncated normal, block rescaling, sinusoidal embedding generation,
zero-initialization of mask tokens, and patch embedding initialization.

---

## 1. Truncated Normal Distribution

Standard `torch.nn.init.trunc_normal_` clips at `[-2*std, 2*std]`, but V-JEPA 2
uses a custom implementation that clips exactly at `[-2, 2]` in standard units.

### Custom Inverse CDF Method

The approach:
1. Map bounds `[-a, a]` to CDF values via `erf`
2. Sample uniformly in `[cdf(-a), cdf(+a)]`
3. Apply inverse CDF (erfinv) to get samples
4. Scale and shift to achieve desired `mean` and `std`

```python
import torch
import torch.nn as nn
import math


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.02,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """
    Fill tensor with samples from truncated normal distribution.

    Uses the percent-point function (PPF / inverse CDF) method for correctness.
    Bounds [a, b] are in units of the distribution (not normalized), so
    truncation occurs at [mean + a*std, mean + b*std].

    This matches the V-JEPA 2 / timm implementation.

    Args:
        tensor: Tensor to fill in-place
        mean: Distribution mean
        std: Distribution standard deviation
        a: Lower bound (standard units, default -2)
        b: Upper bound (standard units, default +2)

    Returns:
        tensor (modified in-place)
    """
    # Method: inverse CDF of truncated normal
    # For N(0,1) truncated to [a, b]:
    #   P(X <= x) = [Phi(x) - Phi(a)] / [Phi(b) - Phi(a)]
    # Sample u ~ Uniform[Phi(a), Phi(b)], then X = Phi^{-1}(u)

    def norm_cdf(x: float) -> float:
        # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        import warnings
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Compute CDF bounds on the standard normal
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Sample uniformly in [l, u], then apply erfinv to get N(0,1) truncated samples
        tensor.uniform_(2 * l - 1, 2 * u - 1)  # uniform in [-1+2l, -1+2u]
        tensor.erfinv_()

        # Scale to desired mean and std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Hard clamp (numerical safety)
        tensor.clamp_(min=a, max=b)

    return tensor


def trunc_normal_tensor(
    shape: tuple,
    mean: float = 0.0,
    std: float = 0.02,
    a: float = -2.0,
    b: float = 2.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Convenience function that returns a new truncated normal tensor."""
    t = torch.empty(shape, dtype=dtype, device=device)
    return trunc_normal_(t, mean=mean, std=std, a=a, b=b)
```

### Usage in Module `__init__`

```python
def _init_weights(module: nn.Module) -> None:
    """Apply V-JEPA 2 default weight initialization to a module tree."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv3d):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

---

## 2. Block Rescaling (Signal Explosion Prevention)

At large depth (24-48 layers), residual stacking causes activation variance to grow
as `O(depth)`. Block rescaling divides the output projections by `sqrt(2 * layer_id)`
to keep variance bounded across layers.

### Formula

For block at layer index `i` (1-indexed):

```
W_out_attn /= sqrt(2 * i)
W_out_mlp  /= sqrt(2 * i)
```

Applied to:
- `attn.proj.weight` — the final linear in attention
- `mlp.fc2.weight`   — the final linear in the FFN (or `mlp.down.weight` for SwiGLU)

### Implementation

```python
def rescale_block_weights(block: nn.Module, layer_id: int) -> None:
    """
    Apply block rescaling to the output projection weights of a transformer block.

    layer_id: 1-indexed position of this block in the stack.
    """
    scale = math.sqrt(2.0 * layer_id)

    # Attention output projection
    if hasattr(block, 'attn') and hasattr(block.attn, 'proj'):
        with torch.no_grad():
            block.attn.proj.weight.div_(scale)

    # MLP / FFN output projection
    if hasattr(block, 'mlp'):
        mlp = block.mlp
        if hasattr(mlp, 'fc2'):
            # Standard MLP
            with torch.no_grad():
                mlp.fc2.weight.div_(scale)
        elif hasattr(mlp, 'down'):
            # SwiGLU (down projection)
            with torch.no_grad():
                mlp.down.weight.div_(scale)


def apply_block_rescaling(blocks: nn.ModuleList) -> None:
    """Apply rescaling to all blocks in a transformer stack."""
    for layer_id, block in enumerate(blocks, start=1):
        rescale_block_weights(block, layer_id)
```

### Effect on Training Dynamics

Without rescaling:
```
Var(x_L) ≈ L * Var(x_0)  (grows linearly with depth)
```

With rescaling:
```
Var(residual_i) = Var(W_out * h) ≈ Var(h) / (2*i)
Sum_{i=1}^{L} Var(residual_i) ≈ Var(h) * log(L) / 2  (much smaller growth)
```

---

## 3. Sinusoidal Positional Embedding Generation

### 2D Sincos Embedding

```python
import numpy as np


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,          # H = W grid size (square assumed)
    cls_token: bool = False,
) -> np.ndarray:
    """
    Generate 2D sine-cosine positional embeddings for image patches.

    Decomposes embed_dim equally: first half for height, second half for width.

    Args:
        embed_dim: Total embedding dimension (must be even)
        grid_size: Number of patches along each spatial dimension
        cls_token: If True, prepend a zero row for the cls token

    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+G*G, embed_dim] with cls
    """
    assert embed_dim % 2 == 0, "embed_dim must be even for 2D sincos"

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # each [G, G]

    # Flatten to [G*G]
    grid_h = grid_h.reshape(-1)
    grid_w = grid_w.reshape(-1)

    # Generate embeddings for each axis with half the dims
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid_h)  # [G*G, D/2]
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid_w)  # [G*G, D/2]

    # Concatenate along embedding dimension
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # [G*G, D]

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,         # Spatial H = W
    grid_depth: int,        # Temporal frames
    cls_token: bool = False,
    uniform_power: bool = False,
) -> np.ndarray:
    """
    Generate 3D sine-cosine positional embeddings for video tubelets.

    Dimension allocation:
        Default (uniform_power=False):
            depth: embed_dim // 2   (50%)
            height: embed_dim // 4  (25%)
            width:  embed_dim // 4  (25%)

        uniform_power=True:
            depth:  embed_dim // 3  (33.3%)
            height: embed_dim // 3  (33.3%)
            width:  embed_dim // 3  (33.3%)
            (may not sum perfectly; trailing dims zero-padded)

    Args:
        embed_dim: Total embedding dimension
        grid_size: Number of spatial patches per side
        grid_depth: Number of temporal tokens (T / tubelet_size)
        cls_token: Prepend zero row for cls token
        uniform_power: Use equal allocation across all 3 axes

    Returns:
        pos_embed: [T*H*W, embed_dim] or [1+T*H*W, embed_dim] with cls
    """
    if uniform_power:
        d_dim = embed_dim // 3
        h_dim = embed_dim // 3
        w_dim = embed_dim - d_dim - h_dim  # remainder goes to width
    else:
        d_dim = embed_dim // 2
        h_dim = embed_dim // 4
        w_dim = embed_dim // 4

    # 1D grids
    grid_d = np.arange(grid_depth, dtype=np.float32)
    grid_h = np.arange(grid_size,  dtype=np.float32)
    grid_w = np.arange(grid_size,  dtype=np.float32)

    # 1D sincos per axis
    emb_d = _get_1d_sincos_pos_embed(d_dim, grid_d)  # [T, d_dim]
    emb_h = _get_1d_sincos_pos_embed(h_dim, grid_h)  # [H, h_dim]
    emb_w = _get_1d_sincos_pos_embed(w_dim, grid_w)  # [W, w_dim]

    # Broadcast and combine: each token (t, h, w) gets concat of all three
    T, H, W = grid_depth, grid_size, grid_size

    # Expand depth: [T, d_dim] -> [T, 1, 1, d_dim] -> [T, H, W, d_dim]
    emb_d_grid = np.tile(emb_d[:, None, None, :], (1, H, W, 1))  # [T, H, W, d_dim]
    # Expand height: [H, h_dim] -> [1, H, 1, h_dim] -> [T, H, W, h_dim]
    emb_h_grid = np.tile(emb_h[None, :, None, :], (T, 1, W, 1))  # [T, H, W, h_dim]
    # Expand width:  [W, w_dim] -> [1, 1, W, w_dim] -> [T, H, W, w_dim]
    emb_w_grid = np.tile(emb_w[None, None, :, :], (T, H, 1, 1))  # [T, H, W, w_dim]

    # Concatenate and reshape to [T*H*W, D]
    pos_embed = np.concatenate(
        [emb_d_grid, emb_h_grid, emb_w_grid], axis=-1
    ).reshape(T * H * W, -1)  # [N, d_dim+h_dim+w_dim]

    # Pad to embed_dim if dims don't sum perfectly
    if pos_embed.shape[1] < embed_dim:
        padding = np.zeros((pos_embed.shape[0], embed_dim - pos_embed.shape[1]))
        pos_embed = np.concatenate([pos_embed, padding], axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed  # [N, embed_dim]


def _get_1d_sincos_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Generate 1D sinusoidal positional embedding.

    Args:
        embed_dim: Embedding dimension (must be even)
        pos: 1D array of positions [N]

    Returns:
        emb: [N, embed_dim]
             Columns alternate: sin(pos * w_0), cos(pos * w_0), sin(pos * w_1), ...
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0   # [0, 2/D, 4/D, ..., (D-2)/D]
    omega = 1.0 / (10000.0 ** omega)  # [1, 1/10000^(2/D), ...]

    pos = pos.reshape(-1)       # [N]
    out = np.outer(pos, omega)  # [N, D//2]

    # Interleave sin and cos: [sin_0, cos_0, sin_1, cos_1, ...]
    emb_sin = np.sin(out)       # [N, D//2]
    emb_cos = np.cos(out)       # [N, D//2]

    # Concatenate: [sin..., cos...] (V-JEPA 2 convention: all sins first, then all cos)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # [N, D]
    return emb.astype(np.float32)
```

### Registering as Frozen Parameter

```python
# In VisionTransformer.__init__:
pos_embed_np = get_3d_sincos_pos_embed(
    embed_dim=self.embed_dim,
    grid_size=self.grid_size,
    grid_depth=self.grid_depth,
    uniform_power=self.uniform_power,
)
pos_embed = torch.from_numpy(pos_embed_np).float()
# Store as non-learnable parameter
self.pos_embed = nn.Parameter(pos_embed.unsqueeze(0), requires_grad=False)
# Shape: [1, N, D]
```

---

## 4. Zero-Initialization of Mask Tokens

Mask tokens are placeholders used in the predictor for target positions.
They must start at zero to avoid biasing the predictor before training.

```python
# In predictor / decoder __init__:
self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

# OR use trunc_normal after a few warm-up steps:
# nn.init.trunc_normal_(self.mask_token, std=0.02)

# In forward, expand to fill masked positions:
def fill_mask_tokens(
    context_tokens: torch.Tensor,  # [B, N_ctx, D]
    mask_indices: torch.Tensor,    # [B, N_mask] indices into full sequence
    n_total: int,                  # total token count
    mask_token: nn.Parameter,      # [1, 1, D]
) -> torch.Tensor:
    """Create full sequence [B, N_total, D] with mask tokens at masked positions."""
    B, D = context_tokens.shape[0], context_tokens.shape[-1]

    # Start with all mask tokens
    full = mask_token.expand(B, n_total, D).clone()

    # Scatter context tokens back to their positions
    # (inverse of the gather used during masking)
    # This requires the context indices, not mask indices
    # ... see masking implementation
    return full
```

---

## 5. Patch Embedding Initialization

### 2D PatchEmbed (Images)

```python
class PatchEmbed(nn.Module):
    """2D patch embedding via Conv2d."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        # Fan-in: patch_size * patch_size * in_chans
        fan_in = self.proj.kernel_size[0] * self.proj.kernel_size[1] * self.proj.in_channels
        std = math.sqrt(1.0 / fan_in)
        trunc_normal_(self.proj.weight, std=std)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, D]
```

### 3D PatchEmbed (Video)

```python
class PatchEmbed3D(nn.Module):
    """3D patch embedding via Conv3d with temporal tubelet."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        self._init_weights()

    @property
    def num_patches_per_frame(self) -> int:
        return (self.img_size // self.patch_size) ** 2

    def _init_weights(self) -> None:
        k = self.proj.kernel_size
        fan_in = k[0] * k[1] * k[2] * self.proj.in_channels
        std = math.sqrt(1.0 / fan_in)
        trunc_normal_(self.proj.weight, std=std)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x = self.proj(x)           # [B, D, T/t, H/p, W/p]
        x = x.flatten(2)           # [B, D, N]
        x = x.transpose(1, 2)      # [B, N, D]
        return x
```

---

## Initialization Checklist

| Component           | Method                  | std / value        | Notes                          |
|---------------------|-------------------------|--------------------|--------------------------------|
| Linear weight       | trunc_normal_           | 0.02               | Default for all linear layers  |
| Linear bias         | zeros_                  | 0                  |                                |
| LayerNorm weight    | ones_                   | 1                  |                                |
| LayerNorm bias      | zeros_                  | 0                  |                                |
| Conv2d/3d weight    | trunc_normal_           | 1/sqrt(fan_in)     | Fan-in scale for patch embed   |
| Conv2d/3d bias      | zeros_                  | 0                  |                                |
| Positional embed    | sincos (frozen)         | N/A                | Not learnable                  |
| Mask token          | zeros_                  | 0                  | Or trunc_normal_ after warmup  |
| QKV bias            | zeros_                  | 0                  | (if qkv_bias=True)             |
| Proj bias           | zeros_                  | 0                  |                                |
| Attn output weight  | trunc_normal_ then /=   | 0.02 / sqrt(2*L)   | Block rescaling after init     |
| MLP output weight   | trunc_normal_ then /=   | 0.02 / sqrt(2*L)   | Block rescaling after init     |
