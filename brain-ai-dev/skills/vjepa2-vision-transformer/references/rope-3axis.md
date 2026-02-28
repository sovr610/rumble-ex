# 3-Axis Rotary Position Embeddings (RoPE) — V-JEPA 2

Complete mathematical specification and implementation guide for the 3-axis RoPE
used in V-JEPA 2's temporal-spatial attention.

---

## Background

Rotary Position Embeddings (RoPE) encode positional information by rotating query and
key vectors in the complex plane. Unlike additive sinusoidal embeddings, RoPE:

1. Encodes relative position implicitly — the dot product `q_i · k_j` depends only on
   the relative position `i - j`, not absolute positions.
2. Decays with distance — far-apart tokens naturally have lower attention scores.
3. Is compatible with KV-cache — keys at position `j` need only be computed once.

For video, V-JEPA 2 decomposes the head dimension into three independent axes:
- **Depth (d)**: temporal axis — frame index
- **Height (h)**: spatial vertical axis
- **Width (w)**: spatial horizontal axis

---

## Dimension Decomposition

Given `head_dim` (= `embed_dim // num_heads`), each axis receives:

```
axis_dim = 2 * floor(head_dim / 6)
```

Total rotated dimensions = `3 * axis_dim`
Remaining dimensions (if `head_dim % 6 != 0`) are left unrotated (identity pass-through).

### Examples

| Variant      | head_dim | axis_dim | rotated | remaining |
|--------------|----------|----------|---------|-----------|
| ViT-Tiny/Large | 64     | 20       | 60      | 4         |
| ViT-Huge     | 80       | 24       | 72      | 8         |
| ViT-Giant    | 88       | 28       | 84      | 4         |
| ViT-Gigantic | 104      | 32       | 96      | 8         |

---

## Frequency Generation

For each axis of dimension `axis_dim`, generate frequencies:

```python
import torch

def get_rope_frequencies(axis_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Generate RoPE frequency bands for a single axis.

    Args:
        axis_dim: Number of dimensions allocated to this axis (must be even).
        theta: Base frequency. V-JEPA 2 uses theta=10000.

    Returns:
        freqs: shape [axis_dim // 2]
        freqs[i] = 1 / theta^(2i / axis_dim)  for i in 0, 1, ..., axis_dim//2 - 1
    """
    assert axis_dim % 2 == 0, "axis_dim must be even for complex rotation pairs"
    i = torch.arange(0, axis_dim, 2, dtype=torch.float32)  # [0, 2, 4, ..., axis_dim-2]
    freqs = 1.0 / (theta ** (i / axis_dim))                # [axis_dim // 2]
    return freqs
```

### Frequency Intuition

For `axis_dim=20` (head_dim=64 case):
- 10 frequency bands per axis
- `freqs[0] = 1.0` (highest frequency, changes rapidly)
- `freqs[9] = 1/10000^(18/20) = 1/6309.6 ≈ 0.000158` (very slow drift)

---

## Position-to-Angle Mapping

For a sequence of positions along one axis:

```python
def positions_to_angles(
    positions: torch.Tensor,  # shape [N] — integer grid positions
    freqs: torch.Tensor,      # shape [axis_dim // 2]
) -> torch.Tensor:
    """
    Compute rotation angles for each position and frequency band.

    Returns:
        angles: shape [N, axis_dim // 2]
        angles[n, k] = positions[n] * freqs[k]
    """
    # positions: [N], freqs: [D/2]  ->  outer product  ->  [N, D/2]
    angles = torch.outer(positions.float(), freqs)
    return angles
```

---

## Rotation Application (Complex Multiplication)

RoPE rotation applies a 2D rotation to each consecutive pair of dimensions:

```
[x_{2k}, x_{2k+1}] -> [x_{2k} * cos(θ_k) - x_{2k+1} * sin(θ_k),
                        x_{2k} * sin(θ_k) + x_{2k+1} * cos(θ_k)]
```

This is equivalent to complex multiplication: `(x_{2k} + i*x_{2k+1}) * e^{iθ_k}`

### Implementation via view_as_complex

```python
def apply_rope_1d(
    x: torch.Tensor,     # shape [..., axis_dim]
    angles: torch.Tensor # shape [..., axis_dim // 2]
) -> torch.Tensor:
    """
    Apply 1D RoPE rotation to a single axis slice of queries or keys.

    Uses complex multiplication for correctness and efficiency.
    """
    # Reshape x into complex pairs: [..., axis_dim] -> [..., axis_dim//2, 2]
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)  # [..., D/2, 2]

    # View as complex: [..., D/2]
    x_complex = torch.view_as_complex(x_pairs)          # [..., D/2]

    # Build rotation complex number: e^{i*theta}
    # angles shape must match x_complex: [..., D/2]
    rot = torch.polar(torch.ones_like(angles), angles)  # [..., D/2]

    # Complex multiply: rotates each pair
    x_rotated = x_complex * rot                         # [..., D/2]

    # Convert back to real: [..., D/2, 2] -> [..., axis_dim]
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)

    return x_out.type_as(x)  # preserve original dtype
```

---

## Full 3-Axis RoPE Class

```python
import torch
import torch.nn as nn
from typing import Tuple, Optional


class RoPE3D(nn.Module):
    """
    3-Axis Rotary Position Embeddings for video tokens.

    Decomposes head_dim into three axes: depth (temporal), height, width.
    Each axis rotates its allocated dimensions independently.

    The positional grid is specified per-forward call, enabling flexible
    resolution and frame count changes without re-initializing.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

        # Each axis gets axis_dim = 2 * floor(head_dim / 6)
        self.axis_dim = 2 * (head_dim // 6)
        self.rotated_dims = 3 * self.axis_dim

        # Compute and cache frequencies (not learnable)
        freqs = self._get_freqs(self.axis_dim, theta)  # [axis_dim // 2]
        self.register_buffer("freqs", freqs)

        # Frequency cache: maps (grid_depth, grid_h, grid_w) -> cached angles
        # Cleared when grid shape changes
        self._cache_key: Optional[Tuple] = None
        self._cache_angles: Optional[torch.Tensor] = None

    @staticmethod
    def _get_freqs(axis_dim: int, theta: float) -> torch.Tensor:
        i = torch.arange(0, axis_dim, 2, dtype=torch.float32)
        return 1.0 / (theta ** (i / axis_dim))

    def _build_grid_angles(
        self,
        grid_depth: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build angle tensor for all tokens in a (T, H, W) grid.

        Returns:
            angles: shape [N, 3 * (axis_dim // 2)]
                    where N = grid_depth * grid_h * grid_w
                    Columns: [d_angles..., h_angles..., w_angles...]
        """
        cache_key = (grid_depth, grid_h, grid_w, device)
        if self._cache_key == cache_key and self._cache_angles is not None:
            return self._cache_angles

        freqs = self.freqs.to(device)  # [axis_dim // 2]

        # Generate position indices for each axis
        pos_d = torch.arange(grid_depth, device=device, dtype=torch.float32)
        pos_h = torch.arange(grid_h,     device=device, dtype=torch.float32)
        pos_w = torch.arange(grid_w,     device=device, dtype=torch.float32)

        # Angles per axis token: outer product of positions and frequencies
        angles_d = torch.outer(pos_d, freqs)  # [T, axis_dim//2]
        angles_h = torch.outer(pos_h, freqs)  # [H, axis_dim//2]
        angles_w = torch.outer(pos_w, freqs)  # [W, axis_dim//2]

        # Broadcast over full (T, H, W) grid
        # Each token (t, h, w) gets angles from all three axes
        D = self.axis_dim // 2
        T, H, W = grid_depth, grid_h, grid_w

        # Expand depth angles: [T, D] -> [T, 1, 1, D] -> [T, H, W, D]
        ang_d = angles_d.view(T, 1, 1, D).expand(T, H, W, D)
        # Expand height angles: [H, D] -> [1, H, 1, D] -> [T, H, W, D]
        ang_h = angles_h.view(1, H, 1, D).expand(T, H, W, D)
        # Expand width angles: [W, D] -> [1, 1, W, D] -> [T, H, W, D]
        ang_w = angles_w.view(1, 1, W, D).expand(T, H, W, D)

        # Flatten spatial-temporal: [T, H, W, D] -> [T*H*W, D]
        ang_d = ang_d.reshape(T * H * W, D)
        ang_h = ang_h.reshape(T * H * W, D)
        ang_w = ang_w.reshape(T * H * W, D)

        # Concatenate all axis angles: [N, 3*D]
        angles = torch.cat([ang_d, ang_h, ang_w], dim=-1)  # [N, 3*(axis_dim//2)]

        self._cache_key = cache_key
        self._cache_angles = angles
        return angles

    def apply_rope(
        self,
        q: torch.Tensor,  # [B, num_heads, N, head_dim]
        k: torch.Tensor,  # [B, num_heads, N, head_dim]
        grid_depth: int,
        grid_h: int,
        grid_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 3-axis RoPE to queries and keys.

        The first `rotated_dims` dimensions of head_dim get rotated;
        remaining dimensions are passed through unchanged.
        """
        B, H, N, D = q.shape
        assert N == grid_depth * grid_h * grid_w, (
            f"Token count {N} != grid product {grid_depth}*{grid_h}*{grid_w}"
        )
        assert D == self.head_dim, f"head_dim mismatch: {D} vs {self.head_dim}"

        # Build angle tensor: [N, 3*(axis_dim//2)]
        angles = self._build_grid_angles(grid_depth, grid_h, grid_w, q.device)
        # angles: [N, axis_dim_total//2] where axis_dim_total = 3 * axis_dim

        # Expand for batch and heads: [1, 1, N, 3*(axis_dim//2)]
        angles = angles.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 3*(axis_dim//2)]

        # Split rotated vs pass-through dims
        q_rot = q[..., :self.rotated_dims]          # [B, H, N, 3*axis_dim]
        q_pass = q[..., self.rotated_dims:]          # [B, H, N, remainder]
        k_rot = k[..., :self.rotated_dims]
        k_pass = k[..., self.rotated_dims:]

        # Apply rotation axis by axis
        ad = self.axis_dim
        # Depth axis: dims [0:ad], angles [0:ad//2]
        q_d = apply_rope_1d(q_rot[..., 0:ad],       angles[..., 0:ad//2])
        k_d = apply_rope_1d(k_rot[..., 0:ad],       angles[..., 0:ad//2])
        # Height axis: dims [ad:2*ad], angles [ad//2:ad]
        q_h = apply_rope_1d(q_rot[..., ad:2*ad],    angles[..., ad//2:ad])
        k_h = apply_rope_1d(k_rot[..., ad:2*ad],    angles[..., ad//2:ad])
        # Width axis: dims [2*ad:3*ad], angles [ad:3*ad//2]
        q_w = apply_rope_1d(q_rot[..., 2*ad:3*ad],  angles[..., ad:3*ad//2])
        k_w = apply_rope_1d(k_rot[..., 2*ad:3*ad],  angles[..., ad:3*ad//2])

        # Reassemble: rotated + pass-through
        q_out = torch.cat([q_d, q_h, q_w, q_pass], dim=-1)  # [B, H, N, D]
        k_out = torch.cat([k_d, k_h, k_w, k_pass], dim=-1)

        return q_out, k_out
```

---

## apply_rope_1d (standalone utility)

```python
def apply_rope_1d(
    x: torch.Tensor,      # [..., axis_dim]
    angles: torch.Tensor, # [..., axis_dim // 2]
) -> torch.Tensor:
    """
    Apply RoPE rotation via complex multiplication to a single axis slice.
    """
    shape = x.shape
    axis_dim = shape[-1]
    assert axis_dim % 2 == 0

    # Pair up consecutive dims for complex representation
    x_pairs = x.float().reshape(*shape[:-1], axis_dim // 2, 2)
    x_complex = torch.view_as_complex(x_pairs.contiguous())

    # Build rotation: e^{i*angle}
    rot = torch.polar(torch.ones_like(angles.float()), angles.float())

    # Multiply and return to real
    x_rotated = x_complex * rot
    x_real = torch.view_as_real(x_rotated).reshape(shape)
    return x_real.to(x.dtype)
```

---

## Relative Position Property Derivation

For query at position `m` and key at position `n`:

```
q_m = R(m) * q      (R is the rotation matrix for position m)
k_n = R(n) * k

dot(q_m, k_n) = (R(m)*q)^T * (R(n)*k)
              = q^T * R(m)^T * R(n) * k
              = q^T * R(n-m) * k
```

The rotation matrices satisfy `R(m)^T * R(n) = R(n-m)`, so the inner product
depends only on the relative position `n-m`. This is the core property that makes
RoPE work for variable-length sequences.

---

## AC-RoPE (Action-Conditioned RoPE)

In V-JEPA 2's robotics variant, action tokens are concatenated to the sequence.
AC-RoPE applies depth-axis RoPE only to action tokens (giving them temporal identity
without spatial location), while video tokens receive the full 3-axis rotation.

```python
def apply_ac_rope(
    q_video: torch.Tensor,   # [B, H, N_video, D]
    k_video: torch.Tensor,   # [B, H, N_video, D]
    q_action: torch.Tensor,  # [B, H, N_action, D]
    k_action: torch.Tensor,  # [B, H, N_action, D]
    rope3d: RoPE3D,
    action_positions: torch.Tensor,  # [N_action] — integer frame indices
    grid_depth: int, grid_h: int, grid_w: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply 3-axis RoPE to video tokens, depth-only RoPE to action tokens."""
    # Video tokens: full 3-axis rotation
    q_v, k_v = rope3d.apply_rope(q_video, k_video, grid_depth, grid_h, grid_w)

    # Action tokens: depth axis only
    # Build depth angles for action frame positions
    D_per_axis = rope3d.axis_dim // 2
    freqs = rope3d.freqs.to(q_action.device)
    ang_action = torch.outer(action_positions.float(), freqs)  # [N_action, D_per_axis]
    ang_action = ang_action.unsqueeze(0).unsqueeze(0)          # [1, 1, N_action, D_per_axis]

    ad = rope3d.axis_dim
    # Rotate only the depth-axis slice; other dims pass through
    q_a_d = apply_rope_1d(q_action[..., :ad],  ang_action)
    k_a_d = apply_rope_1d(k_action[..., :ad],  ang_action)
    q_a_out = torch.cat([q_a_d, q_action[..., ad:]], dim=-1)
    k_a_out = torch.cat([k_a_d, k_action[..., ad:]], dim=-1)

    return q_v, k_v, q_a_out, k_a_out
```

---

## Common Errors and Fixes

| Error                          | Cause                               | Fix                                              |
|-------------------------------|-------------------------------------|--------------------------------------------------|
| `view_as_complex` shape error  | axis_dim is odd                     | Ensure `head_dim % 6 == 0` or pad to even       |
| Angle broadcast mismatch       | angles not expanded for B, H dims   | `angles.unsqueeze(0).unsqueeze(0)` before use   |
| NaN in rotated output          | bfloat16 precision in complex ops   | Cast to float32 before `view_as_complex`        |
| Cache stale after resolution change | Cache key not including device  | Include device in cache key tuple               |
| Wrong equivariance behavior    | Angles built with wrong grid shape  | Verify `N == T * H * W` assertion              |
