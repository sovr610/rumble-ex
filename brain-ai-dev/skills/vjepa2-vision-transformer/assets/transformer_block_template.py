"""
Transformer Block Components for V-JEPA 2 Vision Transformer.

Includes:
- DropPath: Stochastic depth (sample-wise drop during training)
- MLP: Standard 2-layer feed-forward network with GELU
- SwiGLU: Gated feed-forward network with SiLU activation, hidden dim aligned to 8
- Attention: Vanilla multi-head self-attention (SDPA or manual)
- RoPEAttention: Multi-head self-attention with 3-axis RoPE
- Block: Full transformer block (pre-norm, residual, drop-path)

Self-tests run when executed directly: python transformer_block_template.py
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DropPath (Stochastic Depth)
# ---------------------------------------------------------------------------

def drop_path_fn(
    x: torch.Tensor,
    drop_prob: float,
    training: bool,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    """
    Apply stochastic depth drop to a residual path.

    Per-sample dropping: each sample in the batch is independently zeroed
    with probability drop_prob. During inference (training=False), the
    function is a no-op (returns x unchanged).

    Args:
        x: Input tensor with shape [B, ...].
        drop_prob: Probability of dropping the entire sample contribution.
        training: Whether to apply stochastic depth.
        scale_by_keep: If True, scale surviving samples to maintain expectation.

    Returns:
        Tensor of same shape as x.
    """
    if not training or drop_prob == 0.0:
        return x

    keep_prob = 1.0 - drop_prob

    # When drop_prob=1.0, keep_prob=0 -> all samples dropped -> return zeros directly
    if keep_prob == 0.0:
        return torch.zeros_like(x)

    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)

    if scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor


class DropPath(nn.Module):
    """
    Stochastic depth per-sample drop for residual connections.

    During training, drops entire samples with probability drop_prob.
    During model.eval(), is an identity function.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path_fn(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# ---------------------------------------------------------------------------
# MLP (Standard Feed-Forward Network)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Standard two-layer feed-forward network with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1   = nn.Linear(in_features, hidden_features, bias=bias)
        self.act   = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2   = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

def _align_to_8(n: int) -> int:
    """Round n up to the nearest multiple of 8."""
    return math.ceil(n / 8) * 8


class SwiGLU(nn.Module):
    """
    SwiGLU gated feed-forward network.

    Architecture:
        gate = SiLU(x @ W_gate)
        up   = x @ W_up
        out  = (gate * up) @ W_down

    With wide_silu=True:
        hidden_dim = ceil(int(2 * hidden_features / 3) / 8) * 8

    Reference: "GLU Variants Improve Transformer" (Noam Shazeer, 2020)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        wide_silu: bool = True,
        drop: float = 0.0,
        bias: bool = True,
    ):
        """
        Args:
            in_features: Input embedding dimension.
            hidden_features: Nominal hidden dimension (before 2/3 adjustment).
            out_features: Output dimension (defaults to in_features).
            wide_silu: If True, apply the 2/3 reduction and align to multiples of 8.
            drop: Dropout probability on the output.
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        out_features = out_features or in_features

        if wide_silu:
            adjusted = int(2 * hidden_features / 3)
            self.hidden_dim = _align_to_8(adjusted)
        else:
            self.hidden_dim = _align_to_8(hidden_features)

        # Three projections: gate, up (content), and down (output)
        self.gate = nn.Linear(in_features, self.hidden_dim, bias=bias)
        self.up   = nn.Linear(in_features, self.hidden_dim, bias=bias)
        self.down = nn.Linear(self.hidden_dim, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in [self.gate, self.up, self.down]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate(x)) * self.up(x)
        return self.drop(self.down(gated))

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}"


# ---------------------------------------------------------------------------
# Attention (Vanilla MHSA)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention with optional SDPA (PyTorch 2.0+)."""

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
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.use_sdpa  = use_sdpa

        self.qkv       = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj      = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

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


# ---------------------------------------------------------------------------
# RoPEAttention (MHSA + 3-Axis RoPE)
# ---------------------------------------------------------------------------

class RoPEAttention(nn.Module):
    """Multi-head self-attention with 3-axis Rotary Position Embeddings."""

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
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.use_sdpa  = use_sdpa

        self.qkv       = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj      = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        from positional_encoding_template import RoPE3D
        self.rope = RoPE3D(head_dim=self.head_dim, theta=theta)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        rope_grid: tuple,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        grid_depth, grid_h, grid_w = rope_grid

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

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


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    Transformer block with pre-norm, residual connections, and DropPath.

    Pipeline:
        x = x + DropPath(Attention(LayerNorm(x)))
        x = x + DropPath(FFN(LayerNorm(x)))

    Supports vanilla or RoPE attention; standard MLP or SwiGLU FFN.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
        use_rope: bool = False,
        use_silu: bool = False,
        wide_silu: bool = False,
        use_sdpa: bool = True,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.use_rope = use_rope

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        if use_rope:
            self.attn = RoPEAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                use_sdpa=use_sdpa,
                theta=theta,
            )
        else:
            self.attn = Attention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                use_sdpa=use_sdpa,
            )

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_silu:
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                wide_silu=wide_silu,
            )
        else:
            self.mlp = MLP(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
            )

    def forward(
        self,
        x: torch.Tensor,
        rope_grid: Optional[tuple] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_rope:
            assert rope_grid is not None, \
                "rope_grid=(T,H,W) must be provided for RoPEAttention blocks"
            x = x + self.drop_path1(self.attn(self.norm1(x), rope_grid, attn_mask))
        else:
            x = x + self.drop_path1(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Transformer Block Template -- Self-Tests")
    print("=" * 60)

    failures = []

    # Test 1: DropPath(0) is identity
    print("\n[1] DropPath(0) is identity...")
    try:
        dp = DropPath(drop_prob=0.0).train()
        x = torch.randn(4, 196, 768)
        out = dp(x)
        assert torch.allclose(x, out)
        print("   PASS")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("DropPath identity")

    # Test 2: DropPath in inference mode is identity
    print("\n[2] DropPath inference mode is identity...")
    try:
        dp = DropPath(drop_prob=0.5)
        dp.training = False  # set to inference mode
        x = torch.randn(4, 196, 768)
        out = dp(x)
        assert torch.allclose(x, out)
        print("   PASS")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("DropPath inference identity")

    # Test 3: DropPath(1.0) zeros all samples
    print("\n[3] DropPath(1.0) zeros all outputs...")
    try:
        dp = DropPath(drop_prob=1.0)
        dp.training = True
        x = torch.ones(8, 196, 768)
        out = dp(x)
        assert out.abs().sum().item() == 0.0
        print("   PASS")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("DropPath zeros")

    # Test 4: SwiGLU hidden dim alignment to multiples of 8
    print("\n[4] SwiGLU hidden dim is multiple of 8...")
    try:
        test_cases = [
            (192,  4.0),
            (384,  4.0),
            (768,  4.0),
            (1024, 4.0),
            (1280, 4.0),
            (1408, 48/11),
            (1664, 64/11),
        ]
        for embed_dim, mlp_ratio in test_cases:
            hidden = int(embed_dim * mlp_ratio)
            mlp = SwiGLU(in_features=embed_dim, hidden_features=hidden, wide_silu=True)
            assert mlp.hidden_dim % 8 == 0, (
                f"embed_dim={embed_dim}, ratio={mlp_ratio:.4f}: "
                f"hidden_dim={mlp.hidden_dim} not divisible by 8"
            )
        print("   PASS: all variants have hidden_dim divisible by 8")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("SwiGLU alignment")

    # Test 5: SwiGLU output shape
    print("\n[5] SwiGLU output shape...")
    try:
        B, N, D = 2, 196, 768
        mlp = SwiGLU(in_features=D, hidden_features=D * 4)
        x = torch.randn(B, N, D)
        out = mlp(x)
        assert out.shape == (B, N, D), f"Expected {(B, N, D)}, got {out.shape}"
        print(f"   PASS: output shape {out.shape} (hidden_dim={mlp.hidden_dim})")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("SwiGLU shape")

    # Test 6: Standard MLP output shape
    print("\n[6] Standard MLP output shape...")
    try:
        B, N, D = 2, 196, 768
        mlp = MLP(in_features=D, hidden_features=D * 4)
        x = torch.randn(B, N, D)
        out = mlp(x)
        assert out.shape == (B, N, D), f"Expected {(B, N, D)}, got {out.shape}"
        print(f"   PASS: output shape {out.shape}")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("MLP shape")

    # Test 7: Attention output shape
    print("\n[7] Attention output shape (vanilla)...")
    try:
        for B, N, D, H in [(2, 196, 768, 12), (1, 49, 192, 3)]:
            attn = Attention(dim=D, num_heads=H, use_sdpa=True)
            x = torch.randn(B, N, D)
            out = attn(x)
            assert out.shape == (B, N, D), (
                f"B={B},N={N},D={D},H={H}: expected {(B,N,D)}, got {out.shape}"
            )
        print("   PASS")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("Attention shape")

    # Test 8: SDPA matches manual attention
    print("\n[8] SDPA matches manual attention...")
    try:
        torch.manual_seed(42)
        attn_manual = Attention(dim=256, num_heads=8, use_sdpa=False)
        attn_sdpa   = Attention(dim=256, num_heads=8, use_sdpa=True)
        attn_sdpa.load_state_dict(attn_manual.state_dict())

        attn_manual.training = False
        attn_sdpa.training = False

        x = torch.randn(2, 64, 256)
        with torch.no_grad():
            out_manual = attn_manual(x)
            out_sdpa   = attn_sdpa(x)

        max_diff = (out_manual - out_sdpa).abs().max().item()
        assert max_diff < 1e-4, f"Max diff SDPA vs manual: {max_diff:.2e}"
        print(f"   PASS: max diff = {max_diff:.2e}")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("SDPA vs manual")

    # Test 9: Block output shape (vanilla)
    print("\n[9] Block output shape (vanilla attn + MLP)...")
    try:
        B, N, D = 2, 196, 384
        block = Block(dim=D, num_heads=6, mlp_ratio=4.0, use_rope=False, use_silu=False)
        for m in block.modules():
            m.training = False
        x = torch.randn(B, N, D)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (B, N, D), f"Expected {(B,N,D)}, got {out.shape}"
        print(f"   PASS: output shape {out.shape}")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("Block vanilla shape")

    # Test 10: Block output shape (SwiGLU)
    print("\n[10] Block output shape (vanilla attn + SwiGLU)...")
    try:
        B, N, D = 2, 196, 768
        block = Block(dim=D, num_heads=12, mlp_ratio=4.0, use_silu=True, wide_silu=True)
        for m in block.modules():
            m.training = False
        x = torch.randn(B, N, D)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (B, N, D), f"Expected {(B,N,D)}, got {out.shape}"
        assert block.mlp.hidden_dim % 8 == 0, \
            f"SwiGLU hidden dim {block.mlp.hidden_dim} not divisible by 8"
        print(f"   PASS: shape {out.shape}, SwiGLU hidden={block.mlp.hidden_dim}")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("Block SwiGLU shape")

    # Test 11: Block output has no NaN or Inf
    print("\n[11] Block with drop_path=0, no NaN or Inf in output...")
    try:
        block = Block(dim=384, num_heads=6, drop_path=0.0)
        for m in block.modules():
            m.training = False
        x = torch.randn(2, 196, 384)
        with torch.no_grad():
            out = block(x)
        assert not out.isnan().any(), "NaN in block output"
        assert not out.isinf().any(), "Inf in block output"
        print("   PASS: no NaN or Inf")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("Block NaN/Inf check")

    # Test 12: Block in inference mode is deterministic
    print("\n[12] Block in inference mode is deterministic...")
    try:
        block = Block(dim=192, num_heads=3, drop_path=0.3)
        # Use torch Module.train(False) to set all sub-modules to inference mode
        for m in block.modules():
            m.training = False
        x = torch.randn(2, 49, 192)
        with torch.no_grad():
            out1 = block(x)
            out2 = block(x)
        assert torch.allclose(out1, out2), "Block inference not deterministic"
        print("   PASS")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("Block inference determinism")

    # Summary
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} test(s): {failures}")
        sys.exit(1)
    else:
        print("ALL 12 TESTS PASSED")
        sys.exit(0)
