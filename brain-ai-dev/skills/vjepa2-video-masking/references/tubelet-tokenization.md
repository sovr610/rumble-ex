# Tubelet Tokenization

## Overview

Tubelet tokenization converts a raw video tensor `[B, C, T, H, W]` into a sequence of
spatiotemporal patch embeddings `[B, N, D]` via a single `Conv3d` operation.  The key
insight is that a 3-D convolution whose kernel covers `(tubelet_size, patch_size, patch_size)`
strides non-overlapping over the temporal and spatial dimensions simultaneously, producing
one embedding per *tubelet* (a short, cube-shaped video region).

---

## Conv3d Patching

```python
import torch.nn as nn

tubelet_size = 2   # temporal kernel depth (frames per tubelet)
patch_size   = 16  # spatial kernel height = width
embed_dim    = 1024

proj = nn.Conv3d(
    in_channels  = 3,
    out_channels = embed_dim,
    kernel_size  = (tubelet_size, patch_size, patch_size),
    stride       = (tubelet_size, patch_size, patch_size),
    padding      = 0,
)
# Input  : [B, 3, T, H, W]
# Output : [B, embed_dim, T/tubelet_size, H/patch_size, W/patch_size]
```

`stride == kernel_size` ensures non-overlapping coverage (no information is repeated
across tokens).  Changing stride independently enables overlapping tubelets at the cost
of more tokens.

---

## Grid Dimensions

After the `Conv3d` projection the spatial and temporal resolutions collapse to integer
grid indices:

| Dimension | Grid size formula | Variable |
|-----------|-------------------|----------|
| Temporal  | `T // tubelet_size` | depth  |
| Height    | `H // patch_size`   | height |
| Width     | `W // patch_size`   | width  |

For the canonical V-JEPA 2 configuration (`T=16, H=W=224, tubelet_size=2, patch_size=16`):

```
depth  = 16 // 2  =  8
height = 224 // 16 = 14
width  = 224 // 16 = 14
```

---

## Token Count Formula

```
N = (T / tubelet_size) * (H / patch_size) * (W / patch_size)
  = depth * height * width
```

Concrete examples:

| Resolution | T  | tubelet | patch | N        |
|------------|----|---------|-------|----------|
| 224 px 16f | 16 | 2       | 16    | 8*14*14 = **1568** |
| 256 px 16f | 16 | 2       | 16    | 8*16*16 = **2048** |
| 224 px 8f  |  8 | 2       | 16    | 4*14*14 = **784**  |
| 384 px 8f  |  8 | 2       | 16    | 4*24*24 = **2304** |

All integer divisions must be exact — partial tubelets are dropped.  Always validate
`T % tubelet_size == 0` and `H % patch_size == 0` at construction time.

---

## PatchEmbed (2-D, images)

```python
class PatchEmbed(nn.Module):
    """Standard 2-D image patch embedding for ViT."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.grid_size  = (img_size // patch_size, img_size // patch_size)
        self._num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    @property
    def num_patches(self):
        return self._num_patches

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)            # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x
```

---

## PatchEmbed3D (video)

```python
class PatchEmbed3D(nn.Module):
    """3-D tubelet embedding for video."""

    def __init__(self,
                 img_size=224,
                 frames=16,
                 patch_size=16,
                 tubelet_size=2,
                 in_chans=3,
                 embed_dim=1024):
        super().__init__()
        assert frames % tubelet_size == 0, "T must be divisible by tubelet_size"
        assert img_size % patch_size == 0, "H/W must be divisible by patch_size"

        self.img_size     = img_size
        self.frames       = frames
        self.patch_size   = patch_size
        self.tubelet_size = tubelet_size
        self.embed_dim    = embed_dim

        self.grid_size = (
            frames   // tubelet_size,   # depth
            img_size // patch_size,     # height
            img_size // patch_size,     # width
        )
        self._num_patches = (
            self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        )

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride     =(tubelet_size, patch_size, patch_size),
        )

    @property
    def num_patches(self):
        return self._num_patches

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x = self.proj(x)            # [B, D, T/t, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x
```

---

## GPU Decoding Considerations

When loading video at high throughput the bottleneck often shifts from tokenization to
decoding.  Recommendations:

1. **Pre-decoded frames** — Store clips as pre-decoded `.npy` / `.pt` tensors in shared
   memory or NVMe to skip CPU JPEG decoding during training.

2. **GPU-accelerated decoding** — Libraries such as `torchvision.io.VideoReader` (NVDEC)
   or `dali` can decode H.264/H.265 directly on GPU and hand off tensors without
   PCIe round-trips.

3. **`tubelet_size` as a throughput knob** — Doubling `tubelet_size` halves the temporal
   token count and proportionally reduces encoder FLOPs, at the cost of coarser temporal
   resolution.  This is useful for high-FPS sources.

4. **Mixed precision** — `autocast(dtype=torch.bfloat16)` applied to the `Conv3d` and
   subsequent ViT layers reduces memory by ~2x and accelerates computation on Ampere+
   GPUs.

5. **Divisibility checks** — Validate `T % tubelet_size == 0` in the dataloader or
   dataset `__getitem__` rather than the model, so misconfigured clips are caught early
   with a clear error message.

---

## Positional Encoding

V-JEPA 2 uses *factorized sinusoidal* positional embeddings: separate 1-D sin/cos
encodings for the temporal (`depth`) axis, the height axis, and the width axis.  These
are summed element-wise before being added to the token sequence:

```python
def get_sinusoidal_3d_pos_embed(embed_dim, grid_t, grid_h, grid_w):
    """Returns [N, D] positional embedding for (grid_t * grid_h * grid_w) tokens."""
    assert embed_dim % 3 == 0, "embed_dim must be divisible by 3"
    d = embed_dim // 3
    pe_t = get_1d_sincos_pos_embed_from_grid(d, grid_t)  # [T, d]
    pe_h = get_1d_sincos_pos_embed_from_grid(d, grid_h)  # [H, d]
    pe_w = get_1d_sincos_pos_embed_from_grid(d, grid_w)  # [W, d]
    # Broadcast and sum: [T*H*W, D]
    pe = (pe_t[:, None, None, :] +
          pe_h[None, :, None, :] +
          pe_w[None, None, :, :]).reshape(-1, embed_dim)
    return pe
```

Factorized encoding allows the model to generalise to different grid sizes at inference
time by simply recomputing the positional embedding for the new `(T', H', W')` grid.
