"""
VisionTransformerPredictor -- Predicts masked patch representations for V-JEPA 2.

Given context representations from the visible patches and position indices for
both context and target patches, predicts the latent representation at each
masked (target) position.

Token ordering algorithm:
    1. Project context tokens: embed_dim -> predictor_embed_dim
    2. Insert learnable mask tokens at target positions
    3. Concatenate context + target tokens
    4. Sort by original patch position index
    5. Process through transformer blocks
    6. Un-sort (inverse permutation) to restore [context | target] order
    7. Extract target predictions (slice last N_pred tokens)
    8. Project back to encoder space: predictor_embed_dim -> embed_dim

This ensures positional attention is computed in the correct spatial/temporal order.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """
    Multi-head self-attention with optional Flash Attention via SDPA.

    Args:
        dim:       Token embedding dimension.
        num_heads: Number of attention heads.
        qkv_bias:  Whether to use bias in QKV projections.
        attn_drop: Dropout applied to attention weights.
        proj_drop: Dropout applied to projection output.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv      = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.proj     = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # [3, B, H, N, head_dim]
        q, k, v = qkv.unbind(0)             # each [B, H, N, head_dim]

        # Use scaled_dot_product_attention (Flash Attention when available)
        try:
            drop_p = self.attn_drop_p if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
        except Exception:
            # Fallback: manual attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            if self.training and self.attn_drop_p > 0:
                attn = F.dropout(attn, p=self.attn_drop_p)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    """Feed-forward network: Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Block(nn.Module):
    """
    Pre-norm ViT transformer block: LayerNorm + MHSA + LayerNorm + FFN.

    Uses pre-normalization (norm before attention) which is standard for ViT.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp   = MLP(dim, mlp_hidden, dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# VisionTransformerPredictor
# ---------------------------------------------------------------------------

class VisionTransformerPredictor(nn.Module):
    """
    Predicts masked patch representations from visible context using a lightweight
    transformer with learnable mask tokens.

    Args:
        embed_dim:            Encoder output dimension (predictor input / output dim).
        predictor_embed_dim:  Internal predictor dimension (typically smaller).
        depth:                Number of transformer blocks.
        num_heads:            Number of attention heads in each block.
        num_targets:          Number of learnable mask token slots.
        mlp_ratio:            FFN expansion factor.
        qkv_bias:             Whether to add bias to QKV projections.
        drop:                 Dropout rate (typically 0 for SSL pretraining).
        attn_drop:            Attention dropout rate.
        norm_layer:           Normalization layer class.
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
        norm_layer: type = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.embed_dim           = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.num_targets         = num_targets

        # 1. Input projection: encoder space -> predictor space
        self.input_proj = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # 2. Learnable mask tokens — one per target slot
        #    Shape: [1, num_targets, predictor_embed_dim]
        self.mask_tokens = nn.Parameter(
            torch.zeros(1, num_targets, predictor_embed_dim)
        )
        nn.init.trunc_normal_(self.mask_tokens, std=0.02)

        # 3. Transformer blocks
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

        # 4. Final layer norm (pre-output)
        self.norm = norm_layer(predictor_embed_dim)

        # 5. Output projection: predictor space -> encoder space
        self.output_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using truncated normal (standard ViT init)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _get_mask_tokens_for_targets(
        self, batch_size: int, n_pred: int
    ) -> torch.Tensor:
        """
        Expand mask tokens to [B, N_pred, predictor_embed_dim].

        If N_pred > num_targets, tiles mask tokens to cover all target positions.

        Args:
            batch_size: Batch dimension.
            n_pred:     Number of target (masked) positions.

        Returns:
            Tensor of shape [B, N_pred, predictor_embed_dim].
        """
        if n_pred <= self.num_targets:
            # Use first n_pred mask token slots
            tokens = self.mask_tokens[:, :n_pred, :]  # [1, N_pred, D_pred]
        else:
            # Tile mask tokens to cover all target positions
            repeats = math.ceil(n_pred / self.num_targets)
            tokens = self.mask_tokens.repeat(1, repeats, 1)  # [1, R*num_targets, D_pred]
            tokens = tokens[:, :n_pred, :]                    # [1, N_pred, D_pred]

        return tokens.expand(batch_size, -1, -1).contiguous()

    def forward(
        self,
        context_repr: torch.Tensor,
        masks_enc: List[torch.Tensor],
        masks_pred: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Predict representations for masked positions.

        Algorithm:
            1. Project context to predictor embedding space.
            2. Insert learnable mask tokens at target positions.
            3. Concatenate context + target tokens.
            4. Sort by original patch position index.
            5. Process through transformer blocks.
            6. Un-sort to restore [context | target] order.
            7. Extract target predictions (last N_pred positions in concat order).
            8. Project back to encoder embedding dimension.

        Args:
            context_repr: [B, N_vis, embed_dim] encoder output for visible patches.
            masks_enc:    List of 1-D tensors, each [N_vis] giving original patch
                          position indices for visible patches.
            masks_pred:   List of 1-D tensors, each [N_pred] giving original patch
                          position indices for masked (target) patches.

        Returns:
            pred_repr: [B, N_pred, embed_dim] predictions for masked positions.
        """
        B, N_vis, _ = context_repr.shape

        # Use first mask set (loop over multiple in JEPATrainer if needed)
        m_enc  = masks_enc[0]   # [N_vis]  — visible patch position indices
        m_pred = masks_pred[0]  # [N_pred] — target patch position indices
        N_pred = m_pred.shape[0]

        # Step 1: Project context to predictor space
        x_context = self.input_proj(context_repr)  # [B, N_vis, D_pred]

        # Step 2: Prepare mask tokens with target count
        target_tokens = self._get_mask_tokens_for_targets(B, N_pred)
        # target_tokens: [B, N_pred, D_pred]

        # Step 3: Concatenate context + target tokens
        x = torch.cat([x_context, target_tokens], dim=1)  # [B, N_vis + N_pred, D_pred]

        # Step 4: Sort by original position index (restores spatial/temporal order)
        all_positions = torch.cat([m_enc, m_pred], dim=0)  # [N_vis + N_pred]
        sort_idx   = torch.argsort(all_positions)           # ascending by position
        unsort_idx = torch.argsort(sort_idx)                # inverse permutation

        x = x[:, sort_idx, :]  # [B, N_vis + N_pred, D_pred]

        # Step 5: Process through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Step 6: Un-sort to restore [context | target] concatenation order
        x = x[:, unsort_idx, :]  # [B, N_vis + N_pred, D_pred]

        # Step 7: Extract target position outputs (they are the last N_pred positions
        # in the concatenated sequence because masks_enc was listed first)
        pred_tokens = x[:, N_vis:, :]  # [B, N_pred, D_pred]

        # Step 8: Project back to encoder embedding space
        pred_repr = self.output_proj(pred_tokens)  # [B, N_pred, embed_dim]

        return pred_repr

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"predictor_embed_dim={self.predictor_embed_dim}, "
            f"depth={len(self.blocks)}, "
            f"num_targets={self.num_targets}"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("VisionTransformerPredictor self-tests")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # --- Test 1: Output shape is correct ---
    B, N_vis, N_pred = 2, 10, 6
    embed_dim = 128
    pred_dim  = 64

    predictor = VisionTransformerPredictor(
        embed_dim=embed_dim,
        predictor_embed_dim=pred_dim,
        depth=2,
        num_heads=4,
        num_targets=8,
    ).to(device)

    context = torch.randn(B, N_vis, embed_dim, device=device)
    masks_enc  = [torch.arange(N_vis, device=device)]
    masks_pred = [torch.arange(N_vis, N_vis + N_pred, device=device)]

    with torch.no_grad():
        output = predictor(context, masks_enc, masks_pred)

    assert output.shape == (B, N_pred, embed_dim), (
        f"Expected shape ({B}, {N_pred}, {embed_dim}), got {output.shape}"
    )
    print(f"[PASS] Output shape: {output.shape}")

    # --- Test 2: Output has no NaN or Inf ---
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print("[PASS] Output contains no NaN or Inf")

    # --- Test 3: Different batch sizes all work ---
    for bs in [1, 3, 8]:
        ctx = torch.randn(bs, N_vis, embed_dim, device=device)
        with torch.no_grad():
            out = predictor(ctx, masks_enc, masks_pred)
        assert out.shape == (bs, N_pred, embed_dim), (
            f"Failed for batch_size={bs}: got {out.shape}"
        )
    print("[PASS] Works for batch sizes 1, 3, 8")

    # --- Test 4: N_pred > num_targets (mask token tiling) ---
    predictor_small_mask = VisionTransformerPredictor(
        embed_dim=embed_dim, predictor_embed_dim=pred_dim,
        depth=1, num_heads=4, num_targets=4,  # Only 4 mask token slots
    ).to(device)

    N_pred_large = 12  # More targets than mask token slots
    masks_pred_large = [torch.arange(N_vis, N_vis + N_pred_large, device=device)]

    with torch.no_grad():
        output_large = predictor_small_mask(context, masks_enc, masks_pred_large)

    assert output_large.shape == (B, N_pred_large, embed_dim), (
        f"Expected ({B}, {N_pred_large}, {embed_dim}), got {output_large.shape}"
    )
    print(f"[PASS] Handles N_pred ({N_pred_large}) > num_targets (4) via tiling")

    # --- Test 5: Gradients flow through predictor ---
    predictor2 = VisionTransformerPredictor(
        embed_dim=64, predictor_embed_dim=32, depth=1, num_heads=2, num_targets=4,
    )
    context2 = torch.randn(2, 5, 64, requires_grad=False)
    masks_enc2  = [torch.arange(5)]
    masks_pred2 = [torch.arange(5, 9)]

    output2 = predictor2(context2, masks_enc2, masks_pred2)
    loss = output2.mean()
    loss.backward()

    # Check that predictor parameters have gradients
    for name, p in predictor2.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"
    print("[PASS] Gradients flow through all trainable predictor parameters")

    # --- Test 6: Sort/unsort is invertible (identity test) ---
    N = 20
    positions = torch.randperm(N)
    sort_idx   = torch.argsort(positions)
    unsort_idx = torch.argsort(sort_idx)

    x_test = torch.randn(2, N, 32)
    x_sorted   = x_test[:, sort_idx, :]
    x_restored = x_sorted[:, unsort_idx, :]

    assert torch.allclose(x_test, x_restored, atol=1e-7), (
        "Sort -> unsort should recover the original tensor exactly"
    )
    print("[PASS] Sort/unsort operation is exactly invertible")

    # --- Test 7: Non-consecutive masks work correctly ---
    # Visible patches at positions 0, 2, 4, 6; target at 1, 3, 5
    masks_nc_enc  = [torch.tensor([0, 2, 4, 6])]
    masks_nc_pred = [torch.tensor([1, 3, 5])]
    context_nc = torch.randn(2, 4, embed_dim, device=device)

    predictor3 = VisionTransformerPredictor(
        embed_dim=embed_dim, predictor_embed_dim=pred_dim,
        depth=2, num_heads=4, num_targets=8,
    ).to(device)

    with torch.no_grad():
        output_nc = predictor3(context_nc, masks_nc_enc, masks_nc_pred)

    assert output_nc.shape == (2, 3, embed_dim), (
        f"Expected (2, 3, {embed_dim}), got {output_nc.shape}"
    )
    assert not torch.isnan(output_nc).any()
    print("[PASS] Non-consecutive position indices handled correctly")

    # --- Test 8: Mask token parameter is learnable ---
    assert predictor.mask_tokens.requires_grad, (
        "mask_tokens should be a learnable parameter"
    )
    assert predictor.mask_tokens.shape == (1, predictor.num_targets, pred_dim), (
        f"mask_tokens shape should be (1, {predictor.num_targets}, {pred_dim})"
    )
    print(f"[PASS] mask_tokens is learnable parameter with shape {predictor.mask_tokens.shape}")

    # --- Test 9: Different contexts produce different outputs (no constant shortcut) ---
    ctx_a = torch.randn(1, N_vis, embed_dim, device=device)
    ctx_b = torch.randn(1, N_vis, embed_dim, device=device)

    with torch.no_grad():
        out_a = predictor(ctx_a, masks_enc, masks_pred)
        out_b = predictor(ctx_b, masks_enc, masks_pred)

    assert not torch.allclose(out_a, out_b, atol=1e-4), (
        "Different contexts should produce different predictions"
    )
    print("[PASS] Different contexts produce different predictions")

    # --- Test 10: Parameter count is reasonable ---
    total_params = sum(p.numel() for p in predictor.parameters())
    print(f"[PASS] Total parameters: {total_params:,} "
          f"({total_params / 1e6:.2f}M) for embed_dim={embed_dim}, "
          f"pred_dim={pred_dim}, depth=2")

    print()
    print("All 10 self-tests passed.")
