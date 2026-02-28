"""
multi_sequence_template.py
==========================
MultiSequenceEncoder and MultiSequencePredictor wrappers for handling
variable-length clip sequences in a single training step.

Also provides:
    apply_masks(x, masks) -- gather visible tokens from a full sequence.
    apply_masks_fast(x, masks) -- vectorised variant using torch.gather.

References:
    - references/multi-sequence.md
    - SKILL.md § MultiSequenceWrapper, § Multi-Sequence Batching

Usage:
    enc_wrapper  = MultiSequenceEncoder(encoder)
    pred_wrapper = MultiSequencePredictor(predictor)

    ctx_groups  = enc_wrapper(x_groups, masks_enc_groups)
    pred_groups = pred_wrapper(ctx_groups, masks_enc_groups, masks_pred_groups)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# apply_masks: token gathering
# ---------------------------------------------------------------------------

def apply_masks(x: Tensor, masks: List[Tensor]) -> Tensor:
    """
    Gather encoder-visible tokens from a full token sequence.

    This is the naive reference implementation.  It is correct but slightly
    slower than apply_masks_fast due to Python-level looping.

    Args:
        x:     Tensor[B, N, D] -- full sequence of token embeddings.
        masks: List[Tensor[N, bool]] -- one mask per sample.
               True = encoder-visible (keep).

    Returns:
        Tensor[B, N_vis, D] -- only the visible tokens.

    Requires:
        len(masks) == B
        All masks must have the same number of True values (N_vis must be
        equal across samples so the output can be a dense tensor).

    Raises:
        ValueError if shapes are inconsistent.
    """
    B, N, D = x.shape
    if len(masks) != B:
        raise ValueError(
            f"len(masks) = {len(masks)} must equal batch size B = {B}."
        )

    n_vis = int(masks[0].long().sum().item())

    # Validate all masks agree on N_vis
    for i, m in enumerate(masks):
        if m.shape != (N,):
            raise ValueError(
                f"masks[{i}].shape = {m.shape}, expected ({N},)."
            )
        n_vis_i = int(m.long().sum().item())
        if n_vis_i != n_vis:
            raise ValueError(
                f"masks[{i}] has {n_vis_i} visible tokens but masks[0] has "
                f"{n_vis}.  All masks must have the same number of True values."
            )

    out = torch.zeros(B, n_vis, D, dtype=x.dtype, device=x.device)
    for i, m in enumerate(masks):
        out[i] = x[i][m]  # boolean index along sequence dim
    return out


def apply_masks_fast(x: Tensor, masks: List[Tensor]) -> Tensor:
    """
    Vectorised version of apply_masks using torch.gather.

    Avoids Python-level looping over batch dimension; substantially faster
    for large B or D.

    Args:
        x:     Tensor[B, N, D]
        masks: List[Tensor[N, bool]]

    Returns:
        Tensor[B, N_vis, D]
    """
    B, N, D = x.shape
    if len(masks) != B:
        raise ValueError(
            f"len(masks) = {len(masks)} must equal batch size B = {B}."
        )

    # Stack into [B, N]
    M = torch.stack(masks, dim=0).to(x.device)   # [B, N] bool

    n_vis = int(M[0].long().sum().item())

    # Gather visible indices: [B, N_vis]
    # nonzero returns a 2-D tensor of (row, col) pairs; reshape to [B, N_vis]
    idx = M.nonzero(as_tuple=False)               # [B*N_vis, 2]
    vis_idx = idx[:, 1].view(B, n_vis)            # [B, N_vis]

    # Expand for gathering along D
    vis_idx_d = vis_idx.unsqueeze(-1).expand(-1, -1, D)  # [B, N_vis, D]
    return torch.gather(x, 1, vis_idx_d)


def scatter_masks(
    x_vis: Tensor,
    masks:  List[Tensor],
    N:      int,
    fill_value: float = 0.0,
) -> Tensor:
    """
    Inverse of apply_masks: re-insert visible tokens at their original
    positions, filling masked positions with fill_value.

    Args:
        x_vis:      Tensor[B, N_vis, D] -- encoder output.
        masks:      List[Tensor[N, bool]] -- same masks used in apply_masks.
        N:          Total sequence length.
        fill_value: Value for masked positions (default 0.0).

    Returns:
        Tensor[B, N, D] -- full sequence with visible tokens restored.
    """
    B, N_vis, D = x_vis.shape
    out = x_vis.new_full((B, N, D), fill_value)
    for i, m in enumerate(masks):
        out[i][m] = x_vis[i]
    return out


# ---------------------------------------------------------------------------
# Stub encoder / predictor interfaces
# ---------------------------------------------------------------------------

class _StubEncoder(nn.Module):
    """
    Minimal transformer encoder stub for testing.
    In real V-JEPA 2 training replace with VisionTransformer.
    """

    def __init__(self, embed_dim: int = 1024, num_heads: int = 8,
                 num_layers: int = 2) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class _StubPredictor(nn.Module):
    """
    Minimal predictor stub for testing.
    In real V-JEPA 2 replace with VisionTransformerPredictor which uses
    learnable mask tokens at prediction positions.
    """

    def __init__(self, embed_dim: int = 1024, num_heads: int = 8,
                 num_layers: int = 2) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(
        self,
        context: Tensor,                    # [B, N_vis, D]
        masks_enc:  List[Tensor],           # enc bool masks [N]
        masks_pred: List[Tensor],           # pred bool masks [N]
    ) -> Tensor:
        """
        Produce predictions at the masked positions.

        Fills prediction positions with the learnable mask token and uses
        context (encoder output) as memory for cross-attention.

        Returns: Tensor[B, N_pred, D]
        """
        B = context.shape[0]
        n_pred = int(masks_pred[0].long().sum().item())
        D      = context.shape[-1]

        # Query: mask tokens at prediction positions
        queries = self.mask_token.expand(B, n_pred, D)
        out     = self.decoder(queries, context)  # [B, N_pred, D]
        return out


# ---------------------------------------------------------------------------
# MultiSequenceEncoder
# ---------------------------------------------------------------------------

class MultiSequenceEncoder(nn.Module):
    """
    Wrapper that applies an encoder to multiple FPC groups in sequence.

    For each group the full token sequence is first reduced to the visible
    (encoder-unmasked) subset via apply_masks, then passed through the
    encoder.  Results are collected into a list.

    Args:
        encoder: Any nn.Module with signature forward(x: Tensor) -> Tensor.
                 Expected input  [B_g, N_vis, D].
                 Expected output [B_g, N_vis, D].
        fast:    If True use apply_masks_fast (vectorised gather).
    """

    def __init__(self, encoder: nn.Module, fast: bool = True) -> None:
        super().__init__()
        self.encoder = encoder
        self.fast    = fast
        self._apply  = apply_masks_fast if fast else apply_masks

    def forward(
        self,
        x_groups:     List[Tensor],      # List[[B_g, N_g, D]]
        masks_groups: List[List[Tensor]], # List[List[[N_g] bool]]
    ) -> List[Tensor]:
        """
        Args:
            x_groups:     One tensor per FPC group: [B_g, N_g, D].
            masks_groups: One list of masks per group, each mask [N_g] bool.

        Returns:
            List[Tensor[B_g, N_vis_g, D]]  -- encoder outputs per group.
        """
        if len(x_groups) != len(masks_groups):
            raise ValueError(
                f"len(x_groups)={len(x_groups)} != "
                f"len(masks_groups)={len(masks_groups)}."
            )

        outputs: List[Tensor] = []
        for x, masks in zip(x_groups, masks_groups):
            x_vis = self._apply(x, masks)   # [B_g, N_vis_g, D]
            out   = self.encoder(x_vis)     # [B_g, N_vis_g, D]
            outputs.append(out)

        return outputs


# ---------------------------------------------------------------------------
# MultiSequencePredictor
# ---------------------------------------------------------------------------

class MultiSequencePredictor(nn.Module):
    """
    Wrapper that applies a predictor to multiple FPC groups.

    For each group the predictor receives:
    - context:    encoder output [B_g, N_vis_g, D]
    - masks_enc:  which positions the context came from
    - masks_pred: which positions to predict

    The predictor returns predictions [B_g, N_pred_g, D].

    Args:
        predictor: nn.Module with signature
                   forward(context, masks_enc, masks_pred) -> Tensor.
    """

    def __init__(self, predictor: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        context_groups:     List[Tensor],      # List[[B_g, N_vis_g, D]]
        masks_enc_groups:   List[List[Tensor]], # List[List[[N_g] bool]]
        masks_pred_groups:  List[List[Tensor]], # List[List[[N_g] bool]]
    ) -> List[Tensor]:
        """
        Args:
            context_groups:    Encoder outputs per group.
            masks_enc_groups:  Encoder masks per group.
            masks_pred_groups: Prediction masks per group.

        Returns:
            List[Tensor[B_g, N_pred_g, D]] -- predictor outputs per group.
        """
        n = len(context_groups)
        if len(masks_enc_groups) != n or len(masks_pred_groups) != n:
            raise ValueError(
                "context_groups, masks_enc_groups, and masks_pred_groups must "
                "all have the same length."
            )

        outputs: List[Tensor] = []
        for ctx, m_enc, m_pred in zip(
                context_groups, masks_enc_groups, masks_pred_groups):
            pred = self.predictor(ctx, m_enc, m_pred)  # [B_g, N_pred_g, D]
            outputs.append(pred)

        return outputs


# ---------------------------------------------------------------------------
# Self-tests  (python multi_sequence_template.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    PASSED = 0
    FAILED = 0

    def check(name: str, condition: bool, msg: str = "") -> None:
        global PASSED, FAILED
        if condition:
            print(f"  PASS  {name}")
            PASSED += 1
        else:
            print(f"  FAIL  {name}  {msg}")
            FAILED += 1

    # -------------------------------------------------------------------
    # apply_masks and apply_masks_fast
    # -------------------------------------------------------------------
    print("=" * 60)
    print("apply_masks -- correctness")
    print("=" * 60)

    B, N, D = 4, 20, 64
    x = torch.randn(B, N, D)

    # Create masks: first 10 True per sample
    masks = [torch.zeros(N, dtype=torch.bool) for _ in range(B)]
    for m in masks:
        m[:10] = True

    out_naive = apply_masks(x, masks)
    out_fast  = apply_masks_fast(x, masks)

    check("apply_masks output shape [B,10,D]",
          out_naive.shape == (B, 10, D), f"got {out_naive.shape}")
    check("apply_masks_fast output shape [B,10,D]",
          out_fast.shape == (B, 10, D), f"got {out_fast.shape}")
    check("apply_masks == apply_masks_fast",
          torch.allclose(out_naive, out_fast))

    # Verify token values match original
    for i, m in enumerate(masks):
        expected = x[i][m]
        check(f"apply_masks[{i}] correct token values",
              torch.allclose(out_naive[i], expected))

    # All-True mask (identity)
    masks_all = [torch.ones(N, dtype=torch.bool) for _ in range(B)]
    out_all = apply_masks(x, masks_all)
    check("apply_masks all-True returns full sequence",
          torch.allclose(out_all, x))

    # -------------------------------------------------------------------
    # scatter_masks
    # -------------------------------------------------------------------
    print()
    print("=" * 60)
    print("scatter_masks -- roundtrip")
    print("=" * 60)

    x_vis = apply_masks(x, masks)                     # [B, 10, D]
    x_rec = scatter_masks(x_vis, masks, N=N)          # [B, 20, D]

    # Visible positions should match
    for i, m in enumerate(masks):
        check(f"scatter_masks visible positions[{i}] correct",
              torch.allclose(x_rec[i][m], x[i][m]))
    # Masked positions should be zero
    inv_mask = [~m for m in masks]
    for i, m in enumerate(inv_mask):
        check(f"scatter_masks masked positions[{i}] == 0",
              (x_rec[i][m] == 0).all().item())

    # -------------------------------------------------------------------
    # MultiSequenceEncoder
    # -------------------------------------------------------------------
    print()
    print("=" * 60)
    print("MultiSequenceEncoder -- multi-FPC forward")
    print("=" * 60)

    embed_dim = 64

    encoder   = _StubEncoder(embed_dim=embed_dim, num_heads=4, num_layers=1)
    enc_multi = MultiSequenceEncoder(encoder, fast=True)

    # Group 0: B=2, N=1568, D=64 (50% visible)
    N0 = 1568
    B0 = 2
    x0 = torch.randn(B0, N0, embed_dim)
    m0_base = torch.zeros(N0, dtype=torch.bool)
    m0_base[:N0 // 2] = True
    masks0 = [m0_base.clone() for _ in range(B0)]

    # Group 1: B=3, N=784, D=64 (25% visible)
    N1 = 784
    B1 = 3
    x1 = torch.randn(B1, N1, embed_dim)
    m1_base = torch.zeros(N1, dtype=torch.bool)
    m1_base[:N1 // 4] = True
    masks1 = [m1_base.clone() for _ in range(B1)]

    x_groups     = [x0, x1]
    masks_groups = [masks0, masks1]

    with torch.no_grad():
        ctx_groups = enc_multi(x_groups, masks_groups)

    check("MultiSequenceEncoder: 2 groups returned",
          len(ctx_groups) == 2)
    check("MultiSequenceEncoder group0 shape",
          ctx_groups[0].shape == (B0, N0 // 2, embed_dim),
          f"got {ctx_groups[0].shape}")
    check("MultiSequenceEncoder group1 shape",
          ctx_groups[1].shape == (B1, N1 // 4, embed_dim),
          f"got {ctx_groups[1].shape}")

    # -------------------------------------------------------------------
    # MultiSequencePredictor
    # -------------------------------------------------------------------
    print()
    print("=" * 60)
    print("MultiSequencePredictor -- multi-FPC forward")
    print("=" * 60)

    predictor   = _StubPredictor(embed_dim=embed_dim, num_heads=4, num_layers=1)
    pred_multi  = MultiSequencePredictor(predictor)

    # Prediction masks = complement of encoder masks
    pred_masks0 = [~m for m in masks0]
    pred_masks1 = [~m for m in masks1]
    masks_pred_groups = [pred_masks0, pred_masks1]

    with torch.no_grad():
        pred_groups = pred_multi(ctx_groups, masks_groups, masks_pred_groups)

    N_pred0 = N0 - N0 // 2   # 784
    N_pred1 = N1 - N1 // 4   # 588

    check("MultiSequencePredictor: 2 groups returned",
          len(pred_groups) == 2)
    check("MultiSequencePredictor group0 shape",
          pred_groups[0].shape == (B0, N_pred0, embed_dim),
          f"got {pred_groups[0].shape}")
    check("MultiSequencePredictor group1 shape",
          pred_groups[1].shape == (B1, N_pred1, embed_dim),
          f"got {pred_groups[1].shape}")

    # -------------------------------------------------------------------
    # Error handling
    # -------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Error handling")
    print("=" * 60)

    # Mismatched len(masks) vs B
    try:
        apply_masks(x, masks[:2])  # B=4 but only 2 masks
        check("apply_masks raises ValueError on mask count mismatch",
              False, "no error raised")
    except ValueError:
        check("apply_masks raises ValueError on mask count mismatch", True)

    # Inconsistent N_vis across masks
    masks_bad = [torch.zeros(N, dtype=torch.bool) for _ in range(B)]
    masks_bad[0][:10] = True
    masks_bad[1][:5]  = True   # different N_vis
    try:
        apply_masks(x, masks_bad)
        check("apply_masks raises ValueError on inconsistent N_vis",
              False, "no error raised")
    except ValueError:
        check("apply_masks raises ValueError on inconsistent N_vis", True)

    print()
    print("=" * 60)
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    sys.exit(0 if FAILED == 0 else 1)
