"""
Positional Encoding Utilities for V-JEPA 2 Vision Transformer.

Includes:
- get_2d_sincos_pos_embed: 2D sine-cosine positional embeddings for image patches
- get_3d_sincos_pos_embed: 3D sine-cosine positional embeddings for video tubelets
- RoPE3D: 3-axis Rotary Position Embeddings (depth/height/width)
- apply_rope_1d: Core rotation function used by RoPE3D

Self-tests run when executed directly: python positional_encoding_template.py
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1D Sinusoidal Helper
# ---------------------------------------------------------------------------

def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Generate 1D sinusoidal positional embedding for an array of positions.

    The standard convention used in V-JEPA 2:
        freq[k] = 1 / 10000^(2k / embed_dim)   for k = 0, 1, ..., embed_dim//2 - 1
        embedding = [sin(pos * freq[0]), ..., sin(pos * freq[D/2-1]),
                     cos(pos * freq[0]), ..., cos(pos * freq[D/2-1])]

    Args:
        embed_dim: Dimension of the embedding (must be even).
        pos: 1D array of positions, shape [N].

    Returns:
        emb: float32 array of shape [N, embed_dim].
    """
    assert embed_dim % 2 == 0, f"embed_dim must be even, got {embed_dim}"
    half = embed_dim // 2
    omega = np.arange(half, dtype=np.float64) / half  # [0, 2/D, 4/D, ...]
    omega = 1.0 / (10000.0 ** omega)                  # [1, 1/10000^(2/D), ...]

    pos = pos.reshape(-1)                              # [N]
    out = np.outer(pos, omega)                         # [N, D//2]  (outer product)

    emb_sin = np.sin(out)                              # [N, D//2]
    emb_cos = np.cos(out)                              # [N, D//2]

    # Concatenate: all sines then all cosines (V-JEPA 2 convention)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # [N, D]
    return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# 2D Sinusoidal Positional Embeddings
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
) -> np.ndarray:
    """
    Generate 2D sine-cosine positional embeddings for a square image grid.

    Splits embed_dim equally: first half encodes height, second half encodes width.

    Args:
        embed_dim: Total embedding dimension (must be divisible by 2).
        grid_size: Number of patches per spatial side (H = W = grid_size).
        cls_token: If True, prepend a zero-vector for the class token.

    Returns:
        pos_embed: numpy array of shape [grid_size**2, embed_dim],
                   or [1 + grid_size**2, embed_dim] if cls_token=True.
    """
    assert embed_dim % 2 == 0, f"embed_dim must be even, got {embed_dim}"

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)

    # Create meshgrid: grid_h[i,j] = row i, grid_w[i,j] = col j
    grid_w_2d, grid_h_2d = np.meshgrid(grid_w, grid_h)  # both [G, G]
    grid_h_flat = grid_h_2d.reshape(-1)                  # [G*G]
    grid_w_flat = grid_w_2d.reshape(-1)                  # [G*G]

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h_flat)  # [G*G, D/2]
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w_flat)  # [G*G, D/2]

    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # [G*G, D]

    if cls_token:
        cls_emb = np.zeros((1, embed_dim), dtype=np.float32)
        pos_embed = np.concatenate([cls_emb, pos_embed], axis=0)  # [1+G*G, D]

    return pos_embed


# ---------------------------------------------------------------------------
# 3D Sinusoidal Positional Embeddings
# ---------------------------------------------------------------------------

def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    grid_depth: int,
    cls_token: bool = False,
    uniform_power: bool = False,
) -> np.ndarray:
    """
    Generate 3D sine-cosine positional embeddings for video tubelets.

    Dimension allocation strategy:
        Default (uniform_power=False):
            depth:  embed_dim // 2   (50%)
            height: embed_dim // 4   (25%)
            width:  embed_dim // 4   (25%)
            This matches V-JEPA 2's asymmetric temporal weighting.

        uniform_power=True:
            depth:  embed_dim // 3   (33%)
            height: embed_dim // 3   (33%)
            width:  embed_dim - 2*(embed_dim//3)  (remainder, ~34%)

    Args:
        embed_dim: Total embedding dimension.
        grid_size: Spatial patches per side (H = W = grid_size).
        grid_depth: Number of temporal tokens (T / tubelet_size).
        cls_token: If True, prepend a zero row for the class token.
        uniform_power: Use equal dimension allocation across all 3 axes.

    Returns:
        pos_embed: float32 array of shape [T*H*W, embed_dim],
                   or [1 + T*H*W, embed_dim] if cls_token=True.
    """
    if uniform_power:
        d_dim = embed_dim // 3
        h_dim = embed_dim // 3
        w_dim = embed_dim - d_dim - h_dim  # absorb rounding remainder
    else:
        d_dim = embed_dim // 2
        h_dim = embed_dim // 4
        w_dim = embed_dim // 4

    # Generate 1D grids for each axis
    grid_d_pos = np.arange(grid_depth, dtype=np.float32)  # [T]
    grid_h_pos = np.arange(grid_size,  dtype=np.float32)  # [H]
    grid_w_pos = np.arange(grid_size,  dtype=np.float32)  # [W]

    # 1D sincos embeddings per axis
    emb_d = _get_1d_sincos_pos_embed_from_grid(d_dim, grid_d_pos)  # [T, d_dim]
    emb_h = _get_1d_sincos_pos_embed_from_grid(h_dim, grid_h_pos)  # [H, h_dim]
    emb_w = _get_1d_sincos_pos_embed_from_grid(w_dim, grid_w_pos)  # [W, w_dim]

    T, H, W = grid_depth, grid_size, grid_size

    # Broadcast each axis over the full (T, H, W) grid
    # depth: [T, d_dim] -> [T, H, W, d_dim]
    emb_d_grid = np.tile(emb_d[:, None, None, :], (1, H, W, 1))
    # height: [H, h_dim] -> [T, H, W, h_dim]
    emb_h_grid = np.tile(emb_h[None, :, None, :], (T, 1, W, 1))
    # width: [W, w_dim] -> [T, H, W, w_dim]
    emb_w_grid = np.tile(emb_w[None, None, :, :], (T, H, 1, 1))

    # Concatenate and reshape to [T*H*W, total_dim]
    pos_embed = np.concatenate(
        [emb_d_grid, emb_h_grid, emb_w_grid], axis=-1
    ).reshape(T * H * W, -1)

    # Zero-pad if the dimension allocations don't sum to embed_dim exactly
    total_dim = d_dim + h_dim + w_dim
    if total_dim < embed_dim:
        pad = np.zeros((pos_embed.shape[0], embed_dim - total_dim), dtype=np.float32)
        pos_embed = np.concatenate([pos_embed, pad], axis=1)
    elif total_dim > embed_dim:
        pos_embed = pos_embed[:, :embed_dim]

    if cls_token:
        cls_emb = np.zeros((1, embed_dim), dtype=np.float32)
        pos_embed = np.concatenate([cls_emb, pos_embed], axis=0)

    return pos_embed.astype(np.float32)


# ---------------------------------------------------------------------------
# RoPE: 1D Rotation Utility
# ---------------------------------------------------------------------------

def apply_rope_1d(
    x: torch.Tensor,       # [..., axis_dim]  (any leading dims)
    angles: torch.Tensor,  # [..., axis_dim // 2]
) -> torch.Tensor:
    """
    Apply RoPE rotation to a single-axis slice using complex multiplication.

    The standard RoPE rotation for a pair (x_{2k}, x_{2k+1}) at position p:
        x_{2k}'   = x_{2k}   * cos(angle_k) - x_{2k+1} * sin(angle_k)
        x_{2k+1}' = x_{2k}   * sin(angle_k) + x_{2k+1} * cos(angle_k)

    This is exactly complex multiplication: (x_{2k} + i*x_{2k+1}) * e^{i*angle_k}.

    Args:
        x:      Tensor with last dim = axis_dim (must be even).
        angles: Tensor with last dim = axis_dim // 2.
                Leading dims must be broadcastable with x's leading dims.

    Returns:
        Tensor of same shape and dtype as x.
    """
    shape = x.shape
    axis_dim = shape[-1]
    assert axis_dim % 2 == 0, f"axis_dim must be even, got {axis_dim}"
    assert angles.shape[-1] == axis_dim // 2, (
        f"angles last dim {angles.shape[-1]} != axis_dim//2 = {axis_dim // 2}"
    )

    # Work in float32 for numerical stability (bfloat16 loses precision in erfinv)
    x_f32 = x.float()
    angles_f32 = angles.float()

    # Reshape into complex pairs: [..., axis_dim//2, 2]
    x_pairs = x_f32.reshape(*shape[:-1], axis_dim // 2, 2).contiguous()
    # View as complex numbers: [..., axis_dim//2]
    x_complex = torch.view_as_complex(x_pairs)

    # Build rotation: e^{i*angle} = cos(angle) + i*sin(angle)
    rot = torch.polar(torch.ones_like(angles_f32), angles_f32)  # [..., axis_dim//2]

    # Complex multiply: rotates each pair by its corresponding angle
    x_rotated = x_complex * rot

    # Back to real: [..., axis_dim//2, 2] -> [..., axis_dim]
    x_out = torch.view_as_real(x_rotated).reshape(shape)

    return x_out.to(x.dtype)


# ---------------------------------------------------------------------------
# RoPE3D: 3-Axis Rotary Position Embeddings
# ---------------------------------------------------------------------------

class RoPE3D(nn.Module):
    """
    3-Axis Rotary Position Embeddings for video vision transformers.

    Decomposes head_dim into three independent rotation axes:
        - Depth  (d): temporal/frame axis
        - Height (h): spatial vertical axis
        - Width  (w): spatial horizontal axis

    Each axis receives axis_dim = 2 * floor(head_dim / 6) dimensions.
    Remaining dimensions (head_dim - 3*axis_dim) are passed through unchanged.

    The RoPE property: dot(q_m, k_n) depends only on (m - n), not on m or n
    individually. This gives the model implicit relative-position reasoning.

    Usage:
        rope = RoPE3D(head_dim=64)
        q_rot, k_rot = rope.apply_rope(q, k, grid_depth=8, grid_h=14, grid_w=14)
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        """
        Args:
            head_dim: Dimension of each attention head (= embed_dim // num_heads).
            theta: Base frequency for RoPE. V-JEPA 2 uses theta=10000.
        """
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

        # Compute per-axis dimension: 2 * floor(head_dim / 6)
        self.axis_dim = 2 * (head_dim // 6)
        self.rotated_dims = 3 * self.axis_dim

        if self.axis_dim == 0:
            raise ValueError(
                f"head_dim={head_dim} is too small for 3-axis RoPE. "
                f"Need head_dim >= 6."
            )

        # Pre-compute and freeze frequency banks (one bank shared by all 3 axes)
        # freqs[k] = 1 / theta^(2k / axis_dim)  for k = 0, ..., axis_dim//2 - 1
        i = torch.arange(0, self.axis_dim, 2, dtype=torch.float32)  # [axis_dim//2]
        freqs = 1.0 / (theta ** (i / self.axis_dim))                 # [axis_dim//2]
        self.register_buffer("freqs", freqs, persistent=True)

        # Angle cache: avoids recomputing the same grid on every forward pass
        self._cache_key: Optional[Tuple] = None
        self._cache_angles: Optional[torch.Tensor] = None

    def _build_grid_angles(
        self,
        grid_depth: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build the angle tensor for all tokens in a (T, H, W) grid.

        Returns:
            angles: shape [N, 3 * (axis_dim // 2)]
                    where N = grid_depth * grid_h * grid_w
                    Layout: [d_angles | h_angles | w_angles]
        """
        cache_key = (grid_depth, grid_h, grid_w, str(device))
        if self._cache_key == cache_key and self._cache_angles is not None:
            return self._cache_angles

        freqs = self.freqs.to(device)   # [axis_dim // 2]
        D_half = self.axis_dim // 2     # number of frequency bands per axis

        # Integer position grids for each axis
        pos_d = torch.arange(grid_depth, device=device, dtype=torch.float32)  # [T]
        pos_h = torch.arange(grid_h,     device=device, dtype=torch.float32)  # [H]
        pos_w = torch.arange(grid_w,     device=device, dtype=torch.float32)  # [W]

        # Outer products: [positions] x [frequencies] -> [N_axis, D_half]
        angles_d = torch.outer(pos_d, freqs)  # [T, D_half]
        angles_h = torch.outer(pos_h, freqs)  # [H, D_half]
        angles_w = torch.outer(pos_w, freqs)  # [W, D_half]

        T, H, W = grid_depth, grid_h, grid_w

        # Broadcast across the full (T, H, W) grid
        # depth: [T, D_half] -> [T, 1, 1, D_half] -> [T, H, W, D_half]
        ang_d = angles_d.view(T, 1, 1, D_half).expand(T, H, W, D_half).reshape(T * H * W, D_half)
        # height: [H, D_half] -> [1, H, 1, D_half] -> [T, H, W, D_half]
        ang_h = angles_h.view(1, H, 1, D_half).expand(T, H, W, D_half).reshape(T * H * W, D_half)
        # width: [W, D_half] -> [1, 1, W, D_half] -> [T, H, W, D_half]
        ang_w = angles_w.view(1, 1, W, D_half).expand(T, H, W, D_half).reshape(T * H * W, D_half)

        # Stack all axis angles: [N, 3 * D_half]
        angles = torch.cat([ang_d, ang_h, ang_w], dim=-1)

        self._cache_key = cache_key
        self._cache_angles = angles
        return angles

    def apply_rope(
        self,
        q: torch.Tensor,   # [B, num_heads, N, head_dim]
        k: torch.Tensor,   # [B, num_heads, N, head_dim]
        grid_depth: int,
        grid_h: int,
        grid_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 3-axis RoPE to queries and keys.

        Only the first `rotated_dims = 3 * axis_dim` dimensions are rotated.
        Remaining dimensions (if any) are passed through unchanged.
        Values are never rotated.

        Args:
            q: Query tensor [B, H, N, head_dim].
            k: Key tensor [B, H, N, head_dim].
            grid_depth: Number of temporal tokens (T / tubelet_size).
            grid_h: Number of vertical spatial tokens (H / patch_size).
            grid_w: Number of horizontal spatial tokens (W / patch_size).

        Returns:
            q_rotated, k_rotated: Same shape as inputs.
        """
        B, num_heads, N, D = q.shape
        assert N == grid_depth * grid_h * grid_w, (
            f"Token count mismatch: N={N} != {grid_depth}*{grid_h}*{grid_w}"
            f"={grid_depth * grid_h * grid_w}"
        )
        assert D == self.head_dim, f"head_dim mismatch: got {D}, expected {self.head_dim}"

        # Get angle tensor [N, 3*(axis_dim//2)]
        angles = self._build_grid_angles(grid_depth, grid_h, grid_w, q.device)
        # Expand for batch and head dims: [1, 1, N, 3*(axis_dim//2)]
        angles = angles.unsqueeze(0).unsqueeze(0)

        # Split queries and keys into rotated and pass-through sections
        ad = self.axis_dim
        D_half = ad // 2

        q_rot  = q[..., :self.rotated_dims]   # [B, H, N, 3*ad]
        q_pass = q[..., self.rotated_dims:]    # [B, H, N, remainder]
        k_rot  = k[..., :self.rotated_dims]
        k_pass = k[..., self.rotated_dims:]

        # Rotate each axis independently
        # Depth axis: dims [0:ad], angles [..., 0:D_half]
        q_d = apply_rope_1d(q_rot[..., 0:ad],       angles[..., 0:D_half])
        k_d = apply_rope_1d(k_rot[..., 0:ad],       angles[..., 0:D_half])

        # Height axis: dims [ad:2*ad], angles [..., D_half:2*D_half]
        q_h = apply_rope_1d(q_rot[..., ad:2*ad],    angles[..., D_half:2*D_half])
        k_h = apply_rope_1d(k_rot[..., ad:2*ad],    angles[..., D_half:2*D_half])

        # Width axis: dims [2*ad:3*ad], angles [..., 2*D_half:3*D_half]
        q_w = apply_rope_1d(q_rot[..., 2*ad:3*ad],  angles[..., 2*D_half:3*D_half])
        k_w = apply_rope_1d(k_rot[..., 2*ad:3*ad],  angles[..., 2*D_half:3*D_half])

        # Reassemble: rotated parts + pass-through
        q_out = torch.cat([q_d, q_h, q_w, q_pass], dim=-1)  # [B, H, N, D]
        k_out = torch.cat([k_d, k_h, k_w, k_pass], dim=-1)

        return q_out, k_out

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, theta={self.theta}, "
            f"axis_dim={self.axis_dim}, rotated_dims={self.rotated_dims}"
        )


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Positional Encoding Template — Self-Tests")
    print("=" * 60)

    failures = []

    # ------------------------------------------------------------------
    # Test 1: get_2d_sincos_pos_embed shape
    # ------------------------------------------------------------------
    print("\n[1] get_2d_sincos_pos_embed shapes...")
    try:
        for embed_dim, grid_size in [(192, 7), (768, 14), (1024, 14), (384, 16)]:
            emb = get_2d_sincos_pos_embed(embed_dim, grid_size)
            expected_shape = (grid_size * grid_size, embed_dim)
            assert emb.shape == expected_shape, (
                f"embed_dim={embed_dim}, grid={grid_size}: "
                f"expected {expected_shape}, got {emb.shape}"
            )
            # Values should be in [-1, 1] (sincos bounded)
            assert emb.min() >= -1.0 and emb.max() <= 1.0, (
                f"Sincos values out of range: [{emb.min():.4f}, {emb.max():.4f}]"
            )
        # Test with cls_token
        emb_cls = get_2d_sincos_pos_embed(768, 14, cls_token=True)
        assert emb_cls.shape == (197, 768), f"cls_token shape wrong: {emb_cls.shape}"
        assert np.allclose(emb_cls[0], 0.0), "First row (cls token) should be zeros"
        print("   PASS: all shapes correct, values in [-1, 1], cls token is zero")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("2D sincos shape")

    # ------------------------------------------------------------------
    # Test 2: get_3d_sincos_pos_embed shape
    # ------------------------------------------------------------------
    print("\n[2] get_3d_sincos_pos_embed shapes...")
    try:
        for embed_dim, grid_size, grid_depth in [
            (768, 14, 8),
            (1024, 14, 16),
            (192, 7, 4),
        ]:
            N = grid_depth * grid_size * grid_size
            emb = get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth)
            assert emb.shape == (N, embed_dim), (
                f"Expected ({N}, {embed_dim}), got {emb.shape}"
            )
        print("   PASS: 3D sincos shapes correct")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("3D sincos shape")

    # ------------------------------------------------------------------
    # Test 3: uniform_power changes the embedding
    # ------------------------------------------------------------------
    print("\n[3] uniform_power=True produces different embedding...")
    try:
        emb_default = get_3d_sincos_pos_embed(768, 14, 8, uniform_power=False)
        emb_uniform  = get_3d_sincos_pos_embed(768, 14, 8, uniform_power=True)
        assert not np.allclose(emb_default, emb_uniform), \
            "uniform_power=True should produce different values"
        print("   PASS: uniform_power changes embedding values")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("uniform_power")

    # ------------------------------------------------------------------
    # Test 4: RoPE3D initialization and axis_dim computation
    # ------------------------------------------------------------------
    print("\n[4] RoPE3D initialization...")
    try:
        for head_dim in [60, 64, 80, 88, 104]:
            rope = RoPE3D(head_dim=head_dim)
            expected_axis_dim = 2 * (head_dim // 6)
            assert rope.axis_dim == expected_axis_dim, (
                f"head_dim={head_dim}: axis_dim={rope.axis_dim} "
                f"!= expected {expected_axis_dim}"
            )
            assert rope.freqs.shape == (expected_axis_dim // 2,), (
                f"freqs shape wrong: {rope.freqs.shape}"
            )
        print("   PASS: axis_dim and freqs correctly initialized for all head_dims")
    except (AssertionError, ValueError) as e:
        print(f"   FAIL: {e}")
        failures.append("RoPE3D init")

    # ------------------------------------------------------------------
    # Test 5: RoPE changes output when positions differ
    # ------------------------------------------------------------------
    print("\n[5] RoPE changes output when positions differ...")
    try:
        rope = RoPE3D(head_dim=64)
        B, H, D = 1, 1, 64
        torch.manual_seed(42)
        q = torch.randn(B, H, 4, D)
        k = torch.randn(B, H, 4, D)

        # Grid (2, 2, 1): 2 frames, 2 rows, 1 column
        q1, k1 = rope.apply_rope(q.clone(), k.clone(), grid_depth=2, grid_h=2, grid_w=1)

        # Permute token order (swap frames): different positional assignment
        q_perm = q[:, :, [2, 3, 0, 1], :]
        k_perm = k[:, :, [2, 3, 0, 1], :]
        q2, k2 = rope.apply_rope(q_perm, k_perm, grid_depth=2, grid_h=2, grid_w=1)

        # The first token should have different rotated values
        assert not torch.allclose(q1[:, :, 0], q2[:, :, 0], atol=1e-5), \
            "RoPE should produce different output for different positions"
        print("   PASS: RoPE output changes with position permutation")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("RoPE position sensitivity")

    # ------------------------------------------------------------------
    # Test 6: RoPE is deterministic (same input -> same output)
    # ------------------------------------------------------------------
    print("\n[6] RoPE is deterministic...")
    try:
        rope = RoPE3D(head_dim=64)
        torch.manual_seed(0)
        q = torch.randn(2, 4, 8, 64)
        k = torch.randn(2, 4, 8, 64)

        q1, k1 = rope.apply_rope(q, k, grid_depth=2, grid_h=2, grid_w=2)
        q2, k2 = rope.apply_rope(q, k, grid_depth=2, grid_h=2, grid_w=2)

        assert torch.allclose(q1, q2), "RoPE not deterministic (q)"
        assert torch.allclose(k1, k2), "RoPE not deterministic (k)"
        print("   PASS: RoPE is deterministic")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("RoPE determinism")

    # ------------------------------------------------------------------
    # Test 7: RoPE output shape matches input shape
    # ------------------------------------------------------------------
    print("\n[7] RoPE output shapes...")
    try:
        for head_dim, grid_shape in [
            (64,  (4, 7, 7)),
            (80,  (8, 14, 14)),
        ]:
            rope = RoPE3D(head_dim=head_dim)
            gd, gh, gw = grid_shape
            N = gd * gh * gw
            B, num_heads = 2, 4
            q = torch.randn(B, num_heads, N, head_dim)
            k = torch.randn(B, num_heads, N, head_dim)
            q_out, k_out = rope.apply_rope(q, k, gd, gh, gw)
            assert q_out.shape == q.shape, f"q shape changed: {q_out.shape} != {q.shape}"
            assert k_out.shape == k.shape, f"k shape changed: {k_out.shape} != {k.shape}"
        print("   PASS: RoPE preserves tensor shapes")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("RoPE output shape")

    # ------------------------------------------------------------------
    # Test 8: Sincos embeddings are frozen (not in optimizer state by nature)
    # ------------------------------------------------------------------
    print("\n[8] Sincos can be stored as non-learnable parameter...")
    try:
        emb = get_2d_sincos_pos_embed(768, 14)
        emb_t = torch.from_numpy(emb).float().unsqueeze(0)

        # Store as non-learnable parameter (V-JEPA 2 pattern)
        param = nn.Parameter(emb_t, requires_grad=False)
        assert not param.requires_grad, "Sincos pos embed should be non-learnable"

        # Verify shape
        assert param.shape == (1, 196, 768), f"Wrong shape: {param.shape}"
        print("   PASS: Sincos stored correctly as frozen parameter [1, 196, 768]")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("Sincos frozen param")

    # ------------------------------------------------------------------
    # Test 9: apply_rope_1d correctness (compare with manual rotation)
    # ------------------------------------------------------------------
    print("\n[9] apply_rope_1d matches manual cos/sin rotation...")
    try:
        axis_dim = 8
        N = 4
        x = torch.randn(N, axis_dim)
        angles = torch.randn(N, axis_dim // 2)

        # Manual rotation
        x_manual = torch.zeros_like(x)
        for n in range(N):
            for k in range(axis_dim // 2):
                c = torch.cos(angles[n, k])
                s = torch.sin(angles[n, k])
                x_manual[n, 2*k]   = x[n, 2*k]   * c - x[n, 2*k+1] * s
                x_manual[n, 2*k+1] = x[n, 2*k]   * s + x[n, 2*k+1] * c

        # Complex implementation
        x_rope = apply_rope_1d(x, angles)

        assert torch.allclose(x_manual, x_rope, atol=1e-5), (
            f"Max diff: {(x_manual - x_rope).abs().max():.2e}"
        )
        print("   PASS: apply_rope_1d matches manual cos/sin computation")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("apply_rope_1d correctness")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} test(s): {failures}")
        sys.exit(1)
    else:
        total_tests = 9
        print(f"ALL {total_tests} TESTS PASSED")
        sys.exit(0)
