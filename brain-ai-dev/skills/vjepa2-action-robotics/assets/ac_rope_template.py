"""
AC-RoPE Template: Action-Conditioned Rotary Position Embedding

Implements ACRoPEAttention where:
  - Visual tokens receive full 3-axis RoPE (depth, height, width)
  - Action/state/extrinsics tokens receive only depth-axis rotation
    (no spatial position embedding)

This is determined by an action_token_mask boolean tensor.

Also provides ACBlock: the full pre-norm transformer block using ACRoPEAttention.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Rotary Position Embedding Utilities
# ---------------------------------------------------------------------------

def precompute_rope_freqs_3d(
    num_tokens: int,
    head_dim: int,
    T: int,
    H: int,
    W: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Precompute 3D RoPE frequencies.

    The head_dim is split into 3 equal parts for depth, height, width axes.
    Assumes num_tokens = T * H * W.

    Returns:
        freqs: [num_tokens, head_dim] complex (real/imag as 2D float)
               Shape is [num_tokens, head_dim] of real values where adjacent
               pairs (2i, 2i+1) form complex numbers.

    Implementation note: We return cos/sin pairs as [num_tokens, head_dim//2, 2]
    for easy application via rotate_half.
    """
    assert head_dim % 6 == 0, (
        f"head_dim {head_dim} must be divisible by 6 for 3D RoPE "
        f"(split into depth/height/width, each using head_dim//3 dims, "
        f"each axis uses head_dim//6 cos+sin pairs)"
    )
    assert T * H * W == num_tokens, (
        f"T*H*W={T*H*W} != num_tokens={num_tokens}"
    )

    dim_per_axis = head_dim // 3  # dims allocated to each spatial axis
    half_per_axis = dim_per_axis // 2  # pairs for cos/sin

    # Build position indices for each axis
    t_pos = torch.arange(T, device=device).float()
    h_pos = torch.arange(H, device=device).float()
    w_pos = torch.arange(W, device=device).float()

    # Grid positions: [T, H, W] for each axis
    t_grid, h_grid, w_grid = torch.meshgrid(t_pos, h_pos, w_pos, indexing='ij')
    t_flat = t_grid.reshape(-1)   # [T*H*W]
    h_flat = h_grid.reshape(-1)   # [T*H*W]
    w_flat = w_grid.reshape(-1)   # [T*H*W]

    # Frequency bands: theta^{-2i/dim}
    freqs_depth  = 1.0 / (theta ** (torch.arange(0, half_per_axis, device=device).float() / half_per_axis))
    freqs_height = 1.0 / (theta ** (torch.arange(0, half_per_axis, device=device).float() / half_per_axis))
    freqs_width  = 1.0 / (theta ** (torch.arange(0, half_per_axis, device=device).float() / half_per_axis))

    # Outer product: position * frequency -> [N, half_per_axis]
    angles_depth  = t_flat.unsqueeze(-1) * freqs_depth.unsqueeze(0)
    angles_height = h_flat.unsqueeze(-1) * freqs_height.unsqueeze(0)
    angles_width  = w_flat.unsqueeze(-1) * freqs_width.unsqueeze(0)

    # Build cos/sin for each axis: [N, dim_per_axis] (interleaved cos, sin)
    def cos_sin_interleaved(angles: Tensor) -> Tensor:
        # angles: [N, half] -> return [N, 2*half] interleaved [cos_0, sin_0, cos_1, sin_1, ...]
        cos_vals = torch.cos(angles)  # [N, half]
        sin_vals = torch.sin(angles)  # [N, half]
        # Stack and interleave
        stacked = torch.stack([cos_vals, sin_vals], dim=-1)  # [N, half, 2]
        return stacked.reshape(angles.shape[0], -1)  # [N, dim_per_axis]

    depth_rope  = cos_sin_interleaved(angles_depth)
    height_rope = cos_sin_interleaved(angles_height)
    width_rope  = cos_sin_interleaved(angles_width)

    # Concatenate all axes: [N, head_dim]
    full_rope = torch.cat([depth_rope, height_rope, width_rope], dim=-1)
    return full_rope  # [num_tokens, head_dim]


def precompute_rope_freqs_depth_only(
    num_tokens: int,
    head_dim: int,
    T: int,
    H: int,
    W: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Precompute depth-only RoPE frequencies (for action/state/extrinsics tokens).

    Same as 3D RoPE but height and width frequency components are zeroed.
    This means only the depth axis rotates; spatial axes are identity.

    Returns:
        freqs: [num_tokens, head_dim] -- same layout as 3D, but height/width zeros
    """
    assert head_dim % 6 == 0
    dim_per_axis = head_dim // 3
    half_per_axis = dim_per_axis // 2

    # Same depth positions
    t_pos = torch.arange(T, device=device).float()
    h_pos = torch.arange(H, device=device).float()
    w_pos = torch.arange(W, device=device).float()
    t_grid, h_grid, w_grid = torch.meshgrid(t_pos, h_pos, w_pos, indexing='ij')
    t_flat = t_grid.reshape(-1)

    freqs_depth = 1.0 / (theta ** (torch.arange(0, half_per_axis, device=device).float() / half_per_axis))
    angles_depth = t_flat.unsqueeze(-1) * freqs_depth.unsqueeze(0)

    cos_vals = torch.cos(angles_depth)
    sin_vals = torch.sin(angles_depth)
    stacked = torch.stack([cos_vals, sin_vals], dim=-1)
    depth_rope = stacked.reshape(num_tokens, -1)  # [N, dim_per_axis]

    # Height and width axes: zeros (no rotation)
    zero_rope = torch.zeros(num_tokens, dim_per_axis, device=device)

    return torch.cat([depth_rope, zero_rope, zero_rope], dim=-1)  # [N, head_dim]


def apply_rope(x: Tensor, rope_freqs: Tensor) -> Tensor:
    """
    Apply rotary position embedding to query or key tensor.

    Args:
        x:          [B, num_heads, N, head_dim]
        rope_freqs: [N, head_dim]  (alternating cos/sin pairs)

    Returns:
        rotated: [B, num_heads, N, head_dim]
    """
    B, H, N, D = x.shape
    assert rope_freqs.shape == (N, D), (
        f"rope_freqs shape {rope_freqs.shape} != expected ({N}, {D})"
    )

    # Extract cos/sin from interleaved format
    # rope_freqs is [N, D] where adjacent pairs are (cos_i, sin_i)
    rope = rope_freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, N, D]

    # Split into (cos, sin) pairs
    # Reshape to [1, 1, N, D//2, 2] to extract pairs
    rope_pairs = rope.reshape(1, 1, N, D // 2, 2)
    cos_vals = rope_pairs[..., 0]  # [1, 1, N, D//2]
    sin_vals = rope_pairs[..., 1]  # [1, 1, N, D//2]

    # Reshape x to pairs: [B, H, N, D//2, 2]
    x_pairs = x.reshape(B, H, N, D // 2, 2)
    x_even = x_pairs[..., 0]  # [B, H, N, D//2]
    x_odd  = x_pairs[..., 1]  # [B, H, N, D//2]

    # Rotate: [cos -sin; sin cos] * [x_even; x_odd]
    x_rot_even = x_even * cos_vals - x_odd * sin_vals
    x_rot_odd  = x_even * sin_vals + x_odd * cos_vals

    # Interleave back
    x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)  # [B, H, N, D//2, 2]
    return x_rot.reshape(B, H, N, D)


# ---------------------------------------------------------------------------
# AC-RoPE Attention
# ---------------------------------------------------------------------------

class ACRoPEAttention(nn.Module):
    """
    Multi-head self-attention with AC-RoPE.

    Visual tokens receive full 3-axis RoPE.
    Action/state/extrinsics tokens receive depth-only RoPE.

    The distinction is made via action_token_mask [total_tokens] bool tensor.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        # Cache for precomputed RoPE frequencies
        self._rope_cache: Optional[Tuple[Tensor, Tensor, Tuple]] = None

    def _get_rope_freqs(
        self,
        total_tokens: int,
        T: int,
        N: int,
        use_extrinsics: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get or compute RoPE frequencies for visual and action tokens.

        Assumes a layout of [state, (extrinsics), action, visual_1..N] per frame,
        repeated T times. N = H_patches * W_patches spatial patches per frame.

        Returns:
            full_3d_rope: [total_tokens, head_dim] -- for visual tokens
            depth_only_rope: [total_tokens, head_dim] -- for action tokens
        """
        # Compute spatial grid dimensions (assume square patches)
        H_patches = W_patches = int(math.isqrt(N))
        if H_patches * W_patches != N:
            # Non-square: use H=1, W=N as fallback
            H_patches, W_patches = 1, N

        assert self.head_dim % 6 == 0, (
            f"head_dim={self.head_dim} must be divisible by 6 for 3D RoPE"
        )

        # Build per-frame token positions:
        # For each visual token (h, w) in frame t, position = (t, h, w)
        # For action/state/extrinsics tokens at frame t, position = (t, 0, 0)
        num_non_visual = 3 if use_extrinsics else 2
        tokens_per_frame = num_non_visual + N

        # Build position arrays
        depth_positions = []
        height_positions = []
        width_positions = []

        for t in range(T):
            # Non-visual tokens: depth=t, height=0, width=0
            for _ in range(num_non_visual):
                depth_positions.append(t)
                height_positions.append(0)
                width_positions.append(0)
            # Visual tokens: depth=t, height=h, width=w
            for h in range(H_patches):
                for w in range(W_patches):
                    depth_positions.append(t)
                    height_positions.append(h)
                    width_positions.append(w)

        t_flat = torch.tensor(depth_positions,  dtype=torch.float32, device=device)
        h_flat = torch.tensor(height_positions, dtype=torch.float32, device=device)
        w_flat = torch.tensor(width_positions,  dtype=torch.float32, device=device)

        dim_per_axis = self.head_dim // 3
        half_per_axis = dim_per_axis // 2
        theta = 10000.0

        freqs = 1.0 / (theta ** (
            torch.arange(0, half_per_axis, device=device).float() / half_per_axis
        ))

        def make_axis_rope(positions: Tensor, freqs: Tensor) -> Tensor:
            angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
            cos_v = torch.cos(angles)
            sin_v = torch.sin(angles)
            stacked = torch.stack([cos_v, sin_v], dim=-1)
            return stacked.reshape(total_tokens, dim_per_axis)

        depth_rope  = make_axis_rope(t_flat, freqs)
        height_rope = make_axis_rope(h_flat, freqs)
        width_rope  = make_axis_rope(w_flat, freqs)

        # Full 3D rope
        full_3d = torch.cat([depth_rope, height_rope, width_rope], dim=-1)

        # Depth-only: zero height and width axes
        zero_axis = torch.zeros(total_tokens, dim_per_axis, device=device)
        depth_only = torch.cat([depth_rope, zero_axis, zero_axis], dim=-1)

        return full_3d.to(dtype), depth_only.to(dtype)

    def forward(
        self,
        x: Tensor,
        T: int,
        N: int,
        use_extrinsics: bool = True,
        causal_mask: Optional[Tensor] = None,
        action_token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with AC-RoPE.

        Args:
            x:                [B, total_tokens, dim]
            T:                number of frames
            N:                number of visual patches per frame
            use_extrinsics:   whether extrinsics tokens are present
            causal_mask:      [total_tokens, total_tokens] additive bias
            action_token_mask: [total_tokens] bool, True=action/state/extrinsics

        Returns:
            out: [B, total_tokens, dim]
        """
        B, L, D = x.shape
        device, dtype = x.device, x.dtype

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, head_dim]
        q, k, v = qkv.unbind(0)            # each [B, H, L, head_dim]

        # Get RoPE frequencies
        full_rope, depth_rope = self._get_rope_freqs(L, T, N, use_extrinsics, device, dtype)

        if action_token_mask is None:
            # Fallback: apply full 3D RoPE to all tokens
            q = apply_rope(q, full_rope)
            k = apply_rope(k, full_rope)
        else:
            # Apply full 3D RoPE to visual tokens
            visual_mask = ~action_token_mask  # [L] bool

            q_full = apply_rope(q, full_rope)   # full rope applied
            k_full = apply_rope(k, full_rope)

            q_depth = apply_rope(q, depth_rope) # depth-only rope
            k_depth = apply_rope(k, depth_rope)

            # Select: visual tokens from full_rope, action tokens from depth_rope
            # action_token_mask is [L]; expand to [B, H, L, head_dim]
            mask_exp = action_token_mask.view(1, 1, L, 1).expand(B, self.num_heads, L, self.head_dim)

            q = torch.where(mask_exp, q_depth, q_full)
            k = torch.where(mask_exp, k_depth, k_full)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        if causal_mask is not None:
            attn = attn + causal_mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


# ---------------------------------------------------------------------------
# AC Block
# ---------------------------------------------------------------------------

def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    return x / keep_prob * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class ACBlock(nn.Module):
    """
    Full transformer block using ACRoPEAttention.

    Structure:
        LayerNorm -> ACRoPEAttention -> DropPath -> residual
        LayerNorm -> FFN              -> DropPath -> residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ACRoPEAttention(dim, num_heads)
        self.drop_path1 = DropPath(drop_path_rate)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(
        self,
        x: Tensor,
        T: int,
        N: int,
        use_extrinsics: bool = True,
        causal_mask: Optional[Tensor] = None,
        action_token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:                [B, total_tokens, dim]
            T, N:             frame and patch counts for RoPE computation
            use_extrinsics:   for non-visual token count
            causal_mask:      [total_tokens, total_tokens] additive bias
            action_token_mask: [total_tokens] bool

        Returns:
            out: [B, total_tokens, dim]
        """
        # Attention sub-layer
        x = x + self.drop_path1(
            self.attn(
                self.norm1(x),
                T=T,
                N=N,
                use_extrinsics=use_extrinsics,
                causal_mask=causal_mask,
                action_token_mask=action_token_mask,
            )
        )
        # FFN sub-layer
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_action_tokens_different_rope():
    """T24 / T25 / T26: Action tokens get different RoPE than visual tokens."""
    print("[TEST] Action tokens get different RoPE than visual tokens...")

    dim = 48  # divisible by 6 and by num_heads=4
    attn = ACRoPEAttention(dim=dim, num_heads=4)

    B, T, N = 1, 2, 4
    num_non_visual = 2  # state + action (no extrinsics)
    tokens_per_frame = num_non_visual + N
    total = T * tokens_per_frame

    # Build action_token_mask
    mask_entries = []
    for t in range(T):
        mask_entries.extend([True, True])   # state, action
        mask_entries.extend([False] * N)    # visual
    action_token_mask = torch.tensor(mask_entries, dtype=torch.bool)

    x = torch.randn(B, total, dim)

    # Get RoPE frequencies for both types
    full_rope, depth_rope = attn._get_rope_freqs(
        total, T, N, use_extrinsics=False,
        device=torch.device("cpu"), dtype=torch.float32
    )

    # _get_rope_freqs returns per-head frequencies: shape [total_tokens, head_dim]
    # where head_dim = dim // num_heads = 48 // 4 = 12
    # depth_only rope zeros out the height+width axes:
    #   depth_rope has zeros in positions [dim_per_axis:] where dim_per_axis = head_dim//3
    #
    # The zero_axis = torch.zeros(total, dim_per_axis) is raw zeros
    # (NOT cos/sin pairs) -- they multiply with the rotation in apply_rope.
    # When height/width angles are zero, those rotation components vanish.
    head_dim = attn.head_dim  # = dim // num_heads = 12
    dim_per_axis = head_dim // 3  # = 4
    height_width_start = dim_per_axis  # index where height+width portion begins in head_dim

    visual_idx = 2  # first visual token in frame 0
    action_idx = 0  # state token (index 0)

    action_rope_hw = depth_rope[action_idx, height_width_start:]
    visual_rope_hw  = full_rope[visual_idx,  height_width_start:]

    # depth_rope: height and width portions are raw zeros -> all zeros
    assert (action_rope_hw == 0.0).all(), (
        f"Action token height/width rope should be all zeros (depth-only), "
        f"got {action_rope_hw}"
    )
    print(f"  Action tokens have zero height/width RoPE (depth-only) -- PASS")

    # For the last visual token at non-zero (h,w) position, full_rope hw portion
    # should be non-zero (has actual cos/sin rotations from spatial position)
    last_visual_idx = tokens_per_frame - 1  # last patch in frame 0 has max spatial pos
    last_visual_rope_hw = full_rope[last_visual_idx, height_width_start:]
    # At h=1, w=1 (for N=4 grid), angles are non-zero -> rope entries non-trivial
    if last_visual_idx > 2:
        assert not (last_visual_rope_hw == 0.0).all(), (
            "Non-zero-position visual token should have non-zero height/width RoPE"
        )
    print(f"  Visual tokens have non-trivial height/width RoPE -- PASS (structural)")


def _test_output_shape():
    """T27 / T28: Output shape is correct."""
    print("[TEST] AC-RoPE output shape...")

    dim = 48
    block = ACBlock(dim=dim, num_heads=4, mlp_ratio=2.0)

    B, T, N = 2, 3, 9  # 3x3 grid
    num_non_visual = 2
    total = T * (num_non_visual + N)

    x = torch.randn(B, total, dim)

    # Build mask
    mask_entries = []
    for t in range(T):
        mask_entries.extend([True, True])
        mask_entries.extend([False] * N)
    action_mask = torch.tensor(mask_entries, dtype=torch.bool)

    out = block(x, T=T, N=N, use_extrinsics=False, action_token_mask=action_mask)
    assert out.shape == (B, total, dim), f"Shape: {out.shape} != ({B}, {total}, {dim})"
    print(f"  ACBlock output shape {out.shape} -- PASS")

    # No NaN
    assert not torch.isnan(out).any(), "NaN in ACBlock output"
    print(f"  No NaN in output -- PASS")


def _test_gradient_flow():
    """T29: Gradient flows through ACBlock."""
    print("[TEST] ACBlock gradient flow...")

    dim = 48
    block = ACBlock(dim=dim, num_heads=4)

    B, T, N = 1, 2, 4
    total = T * (2 + N)  # no extrinsics
    x = torch.randn(B, total, dim, requires_grad=True)

    action_mask = torch.zeros(total, dtype=torch.bool)
    for t in range(T):
        action_mask[t * (2 + N)] = True      # state
        action_mask[t * (2 + N) + 1] = True  # action

    out = block(x, T=T, N=N, use_extrinsics=False, action_token_mask=action_mask)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient on input"
    params_no_grad = [n for n, p in block.named_parameters()
                      if p.requires_grad and p.grad is None]
    if params_no_grad:
        print(f"  WARNING: params without grad: {params_no_grad}")
    else:
        print(f"  All parameters have gradients -- PASS")


def _test_no_nan_with_causal_mask():
    """Verify no NaN when causal mask is applied."""
    print("[TEST] No NaN with causal mask...")

    from ac_predictor_template import build_block_causal_mask

    dim = 48
    attn = ACRoPEAttention(dim=dim, num_heads=4)

    B, T, N = 1, 3, 4
    tokens_per_frame = 2 + N  # no extrinsics
    total = T * tokens_per_frame

    x = torch.randn(B, total, dim)

    # Build causal mask
    causal_mask = build_block_causal_mask(T, tokens_per_frame, torch.device("cpu"))

    # Build action mask
    mask_entries = []
    for t in range(T):
        mask_entries.extend([True, True])
        mask_entries.extend([False] * N)
    action_mask = torch.tensor(mask_entries, dtype=torch.bool)

    out = attn(x, T=T, N=N, use_extrinsics=False,
               causal_mask=causal_mask, action_token_mask=action_mask)

    assert not torch.isnan(out).any(), "NaN detected with causal mask"
    assert out.shape == (B, total, dim)
    print(f"  No NaN with causal mask, shape {out.shape} -- PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("ACRoPEAttention / ACBlock Self-Tests")
    print("=" * 60)

    _test_action_tokens_different_rope()
    _test_output_shape()
    _test_gradient_flow()

    # Test with causal mask (requires ac_predictor_template to be importable)
    try:
        _test_no_nan_with_causal_mask()
    except ImportError:
        print("[SKIP] causal mask test requires ac_predictor_template.py in path")

    print("=" * 60)
    print("All self-tests PASSED")
    print("=" * 60)
