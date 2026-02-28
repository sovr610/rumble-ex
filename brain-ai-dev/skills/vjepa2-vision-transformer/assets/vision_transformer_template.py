"""
VisionTransformer for V-JEPA 2.

Implements:
- PatchEmbed: 2D conv patch embedding for images
- PatchEmbed3D: 3D conv patch embedding for video (tubelet)
- VisionTransformer: Full encoder with optional masking, intermediate layers,
  positional embedding interpolation, and activation checkpointing

Input/output contract:
    Image: x [B, C, H, W]  -> out [B, N, D]   where N = (H/P)^2
    Video: x [B, C, T, H, W] -> out [B, N, D] where N = (T/t)*(H/P)*(W/P)

    With masking (masks is a List of [B, n_keep] index tensors):
        out [B*len(masks), n_keep, D]

    With out_layers (list of layer indices):
        returns List[Tensor], one per requested layer

Self-tests run when executed directly: python vision_transformer_template.py
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# ---------------------------------------------------------------------------
# Weight initialization utilities (inline to keep module self-contained)
# ---------------------------------------------------------------------------

def _trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.02,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """Fill tensor in-place with truncated normal samples."""
    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


# ---------------------------------------------------------------------------
# Positional Embedding Generators
# ---------------------------------------------------------------------------

def _get_1d_sincos(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """1D sinusoidal positional embedding."""
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    omega = np.arange(half, dtype=np.float64) / half
    omega = 1.0 / (10000.0 ** omega)
    pos = pos.reshape(-1)
    out = np.outer(pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
) -> np.ndarray:
    """2D sincos positional embedding for image patches."""
    grid_w, grid_h = np.meshgrid(
        np.arange(grid_size, dtype=np.float32),
        np.arange(grid_size, dtype=np.float32),
    )
    emb_h = _get_1d_sincos(embed_dim // 2, grid_h.reshape(-1))
    emb_w = _get_1d_sincos(embed_dim // 2, grid_w.reshape(-1))
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    grid_depth: int,
    cls_token: bool = False,
    uniform_power: bool = False,
) -> np.ndarray:
    """3D sincos positional embedding for video tubelets."""
    if uniform_power:
        d_dim = embed_dim // 3
        h_dim = embed_dim // 3
        w_dim = embed_dim - d_dim - h_dim
    else:
        d_dim = embed_dim // 2
        h_dim = embed_dim // 4
        w_dim = embed_dim // 4

    T, H, W = grid_depth, grid_size, grid_size
    emb_d = _get_1d_sincos(d_dim, np.arange(T, dtype=np.float32))   # [T, d_dim]
    emb_h = _get_1d_sincos(h_dim, np.arange(H, dtype=np.float32))   # [H, h_dim]
    emb_w = _get_1d_sincos(w_dim, np.arange(W, dtype=np.float32))   # [W, w_dim]

    d_grid = np.tile(emb_d[:, None, None, :], (1, H, W, 1))
    h_grid = np.tile(emb_h[None, :, None, :], (T, 1, W, 1))
    w_grid = np.tile(emb_w[None, None, :, :], (T, H, 1, 1))

    pos = np.concatenate([d_grid, h_grid, w_grid], axis=-1).reshape(T * H * W, -1)

    total = d_dim + h_dim + w_dim
    if total < embed_dim:
        pos = np.concatenate([pos, np.zeros((pos.shape[0], embed_dim - total), np.float32)], axis=1)
    elif total > embed_dim:
        pos = pos[:, :embed_dim]

    if cls_token:
        pos = np.concatenate([np.zeros([1, embed_dim], np.float32), pos], axis=0)
    return pos.astype(np.float32)


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    2D Patch Embedding via Conv2d.

    Converts image [B, C, H, W] -> flattened patches [B, N, D]
    where N = (H/P) * (W/P), P = patch_size.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.grid_size  = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        fan_in = self.proj.in_channels * self.proj.kernel_size[0] * self.proj.kernel_size[1]
        _trunc_normal_(self.proj.weight, std=math.sqrt(1.0 / fan_in))
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, D]


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding via Conv3d with temporal tubelet.

    Converts video [B, C, T, H, W] -> flattened patches [B, N, D]
    where N = (T/t) * (H/P) * (W/P), t = tubelet_size, P = patch_size.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size   = img_size // patch_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        k = self.proj.kernel_size
        fan_in = self.proj.in_channels * k[0] * k[1] * k[2]
        _trunc_normal_(self.proj.weight, std=math.sqrt(1.0 / fan_in))
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.proj(x)           # [B, D, T/t, H/P, W/P]
        x = x.flatten(2)           # [B, D, N]
        x = x.transpose(1, 2)      # [B, N, D]
        return x


# ---------------------------------------------------------------------------
# Transformer Block (inline, no external import)
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, wide_silu: bool = True):
        super().__init__()
        if wide_silu:
            hd = math.ceil(int(2 * hidden_dim / 3) / 8) * 8
        else:
            hd = math.ceil(hidden_dim / 8) * 8
        self.hidden_dim = hd
        self.gate = nn.Linear(dim, hd)
        self.up   = nn.Linear(dim, hd)
        self.down = nn.Linear(hd, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        mask.div_(keep)
        return x * mask


class _Block(nn.Module):
    """
    Single transformer block: pre-norm -> attention -> residual+droppath
                                -> pre-norm -> FFN -> residual+droppath
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop_path: float,
        use_silu: bool,
        wide_silu: bool,
        use_sdpa: bool,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.use_sdpa  = use_sdpa

        self.qkv  = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)

        hidden = int(dim * mlp_ratio)
        if use_silu:
            self.mlp = _SwiGLU(dim, hidden, wide_silu=wide_silu)
        else:
            self.mlp = _MLP(dim, hidden)

        self.drop_path1 = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.use_sdpa:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = attn @ v

        return self.proj(out.transpose(1, 2).reshape(B, N, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self._attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# apply_masks utility
# ---------------------------------------------------------------------------

def apply_masks(
    x: torch.Tensor,
    masks: List[torch.Tensor],
) -> torch.Tensor:
    """
    Select token subsets specified by mask index tensors.

    Args:
        x: [B, N, D] full token sequence.
        masks: List of [B, n_keep] integer index tensors.

    Returns:
        [B * len(masks), n_keep, D] stacked masked sequences.
    """
    all_x = []
    for m in masks:
        # m: [B, n_keep]
        idx = m.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # [B, n_keep, D]
        all_x.append(torch.gather(x, dim=1, index=idx))
    return torch.cat(all_x, dim=0)   # [B * num_masks, n_keep, D]


# ---------------------------------------------------------------------------
# VisionTransformer
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """
    Vision Transformer encoder for V-JEPA 2.

    Supports both image and video inputs via 2D or 3D patch embedding.
    Positional embeddings are sinusoidal and frozen (not learnable).

    Key features:
    - Masked forward pass: applies_masks() reduces token count before transformer blocks
    - Intermediate layer outputs: out_layers parameter returns list of tensors
    - Positional embedding interpolation: handles resolution changes at inference
    - Activation checkpointing: saves memory at cost of extra compute

    Args:
        img_size: Input spatial resolution (H = W assumed).
        patch_size: Patch size in pixels.
        tubelet_size: Temporal tubelet size for video (3D conv stride).
        in_chans: Input channels (3 for RGB).
        embed_dim: Token embedding dimension D.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: FFN hidden dim multiplier.
        use_rope: Use 3-axis RoPE instead of sincos positional embeddings.
        use_silu: Use SwiGLU FFN instead of GELU MLP.
        wide_silu: Apply 2/3 hidden dim reduction for SwiGLU.
        use_sdpa: Use F.scaled_dot_product_attention.
        drop_path_rate: Maximum stochastic depth rate (linear schedule).
        use_activation_checkpointing: Enable gradient checkpointing per block.
        uniform_power: Equal sincos dim allocation across temporal/spatial axes.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_rope: bool = False,
        use_silu: bool = False,
        wide_silu: bool = False,
        use_sdpa: bool = True,
        drop_path_rate: float = 0.0,
        use_activation_checkpointing: bool = False,
        uniform_power: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
        )

        self.img_size    = img_size
        self.patch_size  = patch_size
        self.tubelet_size = tubelet_size
        self.in_chans    = in_chans
        self.embed_dim   = embed_dim
        self.depth       = depth
        self.num_heads   = num_heads
        self.use_activation_checkpointing = use_activation_checkpointing
        self.uniform_power = uniform_power

        # Spatial grid size at training resolution
        self.grid_size = img_size // patch_size

        # Patch embedding: use 3D for video support by default
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Drop-path rates: linearly increasing across blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            _Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                use_silu=use_silu,
                wide_silu=wide_silu,
                use_sdpa=use_sdpa,
            )
            for i in range(depth)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Positional embeddings: 3D sincos, frozen (non-learnable)
        # We initialize with a dummy grid size (224px, 16 frames, tubelet=2)
        # and interpolate at forward time when needed
        self._init_pos_embed()

        # Weight initialization
        self.apply(self._init_weights)

    def _init_pos_embed(self, grid_depth: int = 8, grid_h: int = None, grid_w: int = None) -> None:
        """Initialize 3D sincos positional embedding for the default grid."""
        if grid_h is None:
            grid_h = self.grid_size
        if grid_w is None:
            grid_w = self.grid_size

        N = grid_depth * grid_h * grid_w
        pos_embed_np = get_3d_sincos_pos_embed(
            embed_dim=self.embed_dim,
            grid_size=grid_h,  # assumes square spatial grid
            grid_depth=grid_depth,
            uniform_power=self.uniform_power,
        )
        # Shapes: [N, D] -> [1, N, D]
        pos_embed = torch.from_numpy(pos_embed_np).float().unsqueeze(0)
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)

        # Store the training grid shape for interpolation detection
        self._pos_embed_grid = (grid_depth, grid_h, grid_w)

    def _init_weights(self, m: nn.Module) -> None:
        """Apply truncated normal initialization to linear and conv layers."""
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            k = m.kernel_size
            fan_in = m.in_channels
            for ki in (k if isinstance(k, tuple) else (k,)):
                fan_in *= ki
            _trunc_normal_(m.weight, std=math.sqrt(1.0 / fan_in))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def interpolate_pos_encoding(
        self,
        x: torch.Tensor,                  # [B, N, D] token sequence after patch embed
        grid_depth: int,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """
        Interpolate stored positional embeddings to match a new (T, H, W) grid.

        Uses trilinear interpolation for 3D video grids.

        Args:
            x: Token sequence (used only for device/dtype info).
            grid_depth: Target temporal grid size.
            grid_h: Target spatial height grid size.
            grid_w: Target spatial width grid size.

        Returns:
            pos_embed: [1, T*H*W, D] interpolated positional embeddings.
        """
        td, th, tw = self._pos_embed_grid

        if (grid_depth, grid_h, grid_w) == (td, th, tw):
            return self.pos_embed  # no interpolation needed

        D = self.embed_dim
        pos = self.pos_embed  # [1, N_train, D]

        # Reshape to 3D spatial tensor for interpolation: [1, D, td, th, tw]
        pos_3d = pos.reshape(1, td, th, tw, D).permute(0, 4, 1, 2, 3).float()

        # Trilinear interpolation to new grid
        pos_interp = F.interpolate(
            pos_3d,
            size=(grid_depth, grid_h, grid_w),
            mode="trilinear",
            align_corners=False,
        )  # [1, D, T, H, W]

        # Reshape back to [1, N_new, D]
        pos_interp = pos_interp.permute(0, 2, 3, 4, 1).reshape(1, -1, D)

        return pos_interp.to(x.dtype)

    def _get_grid_from_input(self, x: torch.Tensor) -> Tuple[int, int, int]:
        """
        Infer (grid_depth, grid_h, grid_w) from the input tensor shape.

        Handles both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs.
        """
        if x.ndim == 4:
            # 2D image: treat as single frame
            B, C, H, W = x.shape
            grid_depth = 1
            grid_h = H // self.patch_size
            grid_w = W // self.patch_size
        elif x.ndim == 5:
            B, C, T, H, W = x.shape
            grid_depth = T // self.tubelet_size
            grid_h = H // self.patch_size
            grid_w = W // self.patch_size
        else:
            raise ValueError(f"Input must be 4D or 5D, got {x.ndim}D shape {x.shape}")
        return grid_depth, grid_h, grid_w

    def _prepare_4d_as_5d(self, x: torch.Tensor) -> torch.Tensor:
        """Expand 4D image [B, C, H, W] to 5D video [B, C, t, H, W] for uniform processing."""
        if x.ndim == 4:
            x = x.unsqueeze(2)  # [B, C, 1, H, W]
            # Repeat tubelet times along temporal axis so Conv3d kernel fits
            x = x.expand(-1, -1, self.tubelet_size, -1, -1)
        return x

    def forward(
        self,
        x: torch.Tensor,                           # [B, C, H, W] or [B, C, T, H, W]
        masks: Optional[List[torch.Tensor]] = None, # List of [B, n_keep] index tensors
        out_layers: Optional[List[int]] = None,     # Layer indices for intermediate outputs
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the ViT encoder.

        Args:
            x: Input tensor. 4D for images, 5D for video.
            masks: If provided, only the selected tokens are processed through
                   the transformer. Each mask is [B, n_keep] indices into the
                   full token sequence. Multiple masks are stacked in the batch dim.
                   Output shape: [B * len(masks), n_keep, D].
            out_layers: If provided, return intermediate features from these block
                        indices (0-indexed). Output is a list of tensors, one per index.
                        When specified, the returned list is ordered by out_layers.

        Returns:
            If out_layers is None: Tensor [B, N, D] (or [B*M, n_keep, D] with masks).
            If out_layers is not None: List of Tensor, one per requested layer.
        """
        # Infer grid dimensions from input
        grid_depth, grid_h, grid_w = self._get_grid_from_input(x)

        # Ensure input is 5D for Conv3d
        x = self._prepare_4d_as_5d(x)

        # Patch embedding: [B, C, T, H, W] -> [B, N, D]
        x = self.patch_embed(x)

        # Add positional embeddings (with interpolation for different resolutions)
        pos = self.interpolate_pos_encoding(x, grid_depth, grid_h, grid_w)  # [1, N, D]
        x = x + pos  # broadcast over batch

        # Apply token masking if provided
        # This reduces the sequence length, dramatically cutting attention cost
        if masks is not None:
            x = apply_masks(x, masks)  # [B*M, n_keep, D]

        # Run through transformer blocks
        if out_layers is not None:
            out_layer_set = set(out_layers)
            layer_outputs = {}

        for i, block in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

            if out_layers is not None and i in out_layer_set:
                layer_outputs[i] = self.norm(x)

        # Final normalization
        x = self.norm(x)

        if out_layers is not None:
            # Also include last layer if it's the final block index
            if (self.depth - 1) in out_layer_set:
                layer_outputs[self.depth - 1] = x
            # Return in the order specified by out_layers
            return [layer_outputs[i] for i in out_layers if i in layer_outputs]

        return x

    def no_weight_decay(self) -> set:
        """Parameters that should not receive weight decay (for optimizer configuration)."""
        return {"pos_embed"}

    def extra_repr(self) -> str:
        return (
            f"img_size={self.img_size}, patch_size={self.patch_size}, "
            f"embed_dim={self.embed_dim}, depth={self.depth}, "
            f"num_heads={self.num_heads}"
        )


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("VisionTransformer Template -- Self-Tests")
    print("=" * 60)

    failures = []

    # ------------------------------------------------------------------
    # Test 1: 2D image input produces correct output shape [B, N, D]
    # ------------------------------------------------------------------
    print("\n[1] 2D image input: output shape [B, N, D]...")
    try:
        model = VisionTransformer(
            img_size=224, patch_size=16,
            embed_dim=192, depth=4, num_heads=3,
        )
        model.training = False
        B, C, H, W = 2, 3, 224, 224
        N_expected = (H // 16) * (W // 16)  # 196
        x = torch.randn(B, C, H, W)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, N_expected, 192), \
            f"Expected {(B, N_expected, 192)}, got {out.shape}"
        print(f"   PASS: output shape {out.shape}")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("2D forward shape")

    # ------------------------------------------------------------------
    # Test 2: Video input produces correct output shape [B, N, D]
    # ------------------------------------------------------------------
    print("\n[2] Video input [B,C,T,H,W]: output shape [B, N, D]...")
    try:
        model = VisionTransformer(
            img_size=224, patch_size=16, tubelet_size=2,
            embed_dim=192, depth=4, num_heads=3,
        )
        model.training = False
        B, C, T, H, W = 2, 3, 8, 224, 224
        grid_T = T // 2  # 4
        grid_S = H // 16  # 14
        N_expected = grid_T * grid_S * grid_S  # 4*14*14 = 784
        x = torch.randn(B, C, T, H, W)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, N_expected, 192), \
            f"Expected {(B, N_expected, 192)}, got {out.shape}"
        print(f"   PASS: output shape {out.shape}")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("Video forward shape")

    # ------------------------------------------------------------------
    # Test 3: Masked forward reduces sequence length
    # ------------------------------------------------------------------
    print("\n[3] Masked forward reduces token count...")
    try:
        model = VisionTransformer(
            img_size=224, patch_size=16,
            embed_dim=192, depth=2, num_heads=3,
        )
        model.training = False
        B, C, H, W = 2, 3, 224, 224
        N_total = 196
        n_keep  = 49  # keep 25% of tokens

        masks = [torch.randperm(N_total)[:n_keep].unsqueeze(0).expand(B, -1)]
        x = torch.randn(B, C, H, W)
        with torch.no_grad():
            out = model(x, masks=masks)

        expected_shape = (B * 1, n_keep, 192)
        assert out.shape == expected_shape, \
            f"Expected {expected_shape}, got {out.shape}"
        print(f"   PASS: masked output shape {out.shape} ({n_keep}/{N_total} tokens kept)")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("Masked forward shape")

    # ------------------------------------------------------------------
    # Test 4: Multiple masks stack correctly in batch dim
    # ------------------------------------------------------------------
    print("\n[4] Multiple masks stack in batch dim...")
    try:
        model = VisionTransformer(embed_dim=192, depth=2, num_heads=3)
        model.training = False
        B, N_total, n_keep = 2, 196, 49
        num_masks = 3
        masks = [torch.randperm(N_total)[:n_keep].unsqueeze(0).expand(B, -1)
                 for _ in range(num_masks)]
        x = torch.randn(B, 3, 224, 224)
        with torch.no_grad():
            out = model(x, masks=masks)
        assert out.shape == (B * num_masks, n_keep, 192), \
            f"Expected ({B*num_masks},{n_keep},192), got {out.shape}"
        print(f"   PASS: stacked shape {out.shape}")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("Multiple masks stacking")

    # ------------------------------------------------------------------
    # Test 5: out_layers returns list of intermediate tensors
    # ------------------------------------------------------------------
    print("\n[5] out_layers returns intermediate features...")
    try:
        model = VisionTransformer(embed_dim=192, depth=6, num_heads=3)
        model.training = False
        x = torch.randn(2, 3, 224, 224)
        out_layers = [1, 3, 5]
        with torch.no_grad():
            outs = model(x, out_layers=out_layers)
        assert isinstance(outs, list), "out_layers should return a list"
        assert len(outs) == len(out_layers), \
            f"Expected {len(out_layers)} tensors, got {len(outs)}"
        for i, o in enumerate(outs):
            assert o.shape == (2, 196, 192), \
                f"Layer {out_layers[i]}: expected (2,196,192), got {o.shape}"
        print(f"   PASS: got {len(outs)} tensors at layers {out_layers}")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("out_layers intermediate features")

    # ------------------------------------------------------------------
    # Test 6: Positional embedding interpolation to larger resolution
    # ------------------------------------------------------------------
    print("\n[6] Positional embedding interpolation to 384px...")
    try:
        # Model with default 224px pos embed
        model = VisionTransformer(
            img_size=224, patch_size=16,
            embed_dim=192, depth=2, num_heads=3,
        )
        model.training = False

        # Infer on larger 384px input
        x_large = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            out = model(x_large)

        N_expected = (384 // 16) ** 2  # 576
        assert out.shape == (1, N_expected, 192), \
            f"Expected (1, {N_expected}, 192), got {out.shape}"
        print(f"   PASS: 384px output shape {out.shape}")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("Pos embed interpolation 384px")

    # ------------------------------------------------------------------
    # Test 7: Positional embedding is non-learnable
    # ------------------------------------------------------------------
    print("\n[7] Positional embedding is non-learnable (requires_grad=False)...")
    try:
        model = VisionTransformer(embed_dim=192, depth=2, num_heads=3)
        assert not model.pos_embed.requires_grad, \
            "pos_embed must be non-learnable"
        print("   PASS")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("Pos embed frozen")

    # ------------------------------------------------------------------
    # Test 8: Output has no NaN or Inf
    # ------------------------------------------------------------------
    print("\n[8] Output has no NaN or Inf...")
    try:
        model = VisionTransformer(embed_dim=192, depth=4, num_heads=3)
        model.training = False
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert not out.isnan().any(), "NaN in output"
        assert not out.isinf().any(), "Inf in output"
        print("   PASS")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("NaN/Inf check")

    # ------------------------------------------------------------------
    # Test 9: Gradient flows through the model
    # ------------------------------------------------------------------
    print("\n[9] Gradient flows through forward pass...")
    try:
        model = VisionTransformer(embed_dim=192, depth=2, num_heads=3)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that patch_embed has gradients
        assert model.patch_embed.proj.weight.grad is not None, \
            "No gradient in patch_embed.proj.weight"
        assert not model.patch_embed.proj.weight.grad.isnan().any(), \
            "NaN gradient in patch_embed"
        print("   PASS: gradients flow correctly")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("Gradient flow")

    # ------------------------------------------------------------------
    # Test 10: PatchEmbed3D token count
    # ------------------------------------------------------------------
    print("\n[10] PatchEmbed3D token count...")
    try:
        pe = PatchEmbed3D(img_size=224, patch_size=16, tubelet_size=2, embed_dim=192)
        for T, H, W, N_expected in [
            (8,  224, 224, 4*14*14),
            (16, 224, 224, 8*14*14),
            (32, 256, 256, 16*16*16),
        ]:
            x = torch.randn(2, 3, T, H, W)
            out = pe(x)
            assert out.shape == (2, N_expected, 192), \
                f"T={T},H={H},W={W}: expected (2,{N_expected},192), got {out.shape}"
        print("   PASS: all token counts correct")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("PatchEmbed3D token count")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} test(s): {failures}")
        sys.exit(1)
    else:
        print("ALL 10 TESTS PASSED")
        sys.exit(0)
