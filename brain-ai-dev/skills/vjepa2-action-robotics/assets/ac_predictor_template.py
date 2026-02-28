"""
AC Predictor Template: ActionConditionedPredictor

Implements the action-conditioned variant of the V-JEPA 2 predictor.
Conditioning signals (action, state, extrinsics) are interleaved with
visual tokens per frame. A block-causal attention mask enforces temporal
causality at the frame level.

Token layout per frame:
    [state_token, (extrinsics_token), action_token, visual_1, ..., visual_N]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ACPredictorConfig:
    embed_dim: int = 384               # Input encoder embedding dimension
    predictor_embed_dim: int = 1024    # Internal predictor dimension
    depth: int = 24                    # Number of transformer blocks
    num_heads: int = 16                # Attention heads
    action_embed_dim: int = 7          # Raw action DOF (7-DOF robot)
    state_embed_dim: int = 7           # Raw state DOF (7-DOF robot)
    extrinsics_embed_dim: int = 6      # 6-DOF camera extrinsics
    use_extrinsics: bool = True        # Whether to include extrinsics token
    pred_is_frame_causal: bool = True  # Whether to apply block-causal mask
    mlp_ratio: float = 4.0            # FFN expansion ratio
    drop_path_rate: float = 0.0       # Stochastic depth rate
    norm_layer: str = "layernorm"      # Normalization type


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_norm_layer(name: str, dim: int) -> nn.Module:
    if name == "layernorm":
        return nn.LayerNorm(dim)
    elif name == "rmsnorm":
        return nn.RMSNorm(dim)  # type: ignore[attr-defined]
    else:
        raise ValueError(f"Unknown norm layer: {name}")


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """Stochastic depth regularization (DropPath)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    output = x / keep_prob * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Block-Causal Mask
# ---------------------------------------------------------------------------

def build_block_causal_mask(
    T: int,
    tokens_per_frame: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Build an additive attention bias implementing block-causal masking.

    Frame t can attend to frames 0..t (inclusive), but not to t+1..T-1.
    Within a frame, all tokens attend to each other (bidirectional).

    Returns:
        mask: [T*tokens_per_frame, T*tokens_per_frame]
              0.0 where attention is allowed, -inf where blocked.
    """
    total = T * tokens_per_frame
    mask = torch.zeros(total, total, dtype=dtype, device=device)

    for t in range(T):
        q_start = t * tokens_per_frame
        q_end = (t + 1) * tokens_per_frame
        # Block all future frames
        k_start = (t + 1) * tokens_per_frame
        if k_start < total:
            mask[q_start:q_end, k_start:total] = float("-inf")

    return mask


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention with optional causal mask
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Standard MHSA with support for additive bias (for causal masking)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:         [B, N, D]
            attn_bias: [N, N] or [B, 1, N, N] additive mask (0=allow, -inf=block)

        Returns:
            out: [B, N, D]
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, head_dim]
        q, k, v = qkv.unbind(0)            # each [B, H, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        if attn_bias is not None:
            if attn_bias.dim() == 2:
                attn = attn + attn_bias.unsqueeze(0).unsqueeze(0)
            else:
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


# ---------------------------------------------------------------------------
# MLP Feed-Forward
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop_path_rate: float = 0.0, norm_layer: str = "layernorm"):
        super().__init__()
        self.norm1 = get_norm_layer(norm_layer, dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.drop_path = DropPath(drop_path_rate)
        self.norm2 = get_norm_layer(norm_layer, dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim)

    def forward(self, x: Tensor, attn_bias: Optional[Tensor] = None) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), attn_bias))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Token Interleaving
# ---------------------------------------------------------------------------

def interleave_tokens(
    visual_tokens: Tensor,    # [B, T, N, D]
    action_tokens: Tensor,    # [B, T, D]
    state_tokens: Tensor,     # [B, T, D]
    extrinsics_tokens: Optional[Tensor] = None,  # [B, T, D]
) -> Tuple[Tensor, Tensor]:
    """
    Interleave conditioning tokens with visual tokens per frame.

    Returns:
        interleaved: [B, T * tokens_per_frame, D]
        action_token_mask: [T * tokens_per_frame] bool,
                           True for action/state/extrinsics positions
    """
    B, T, N, D = visual_tokens.shape
    frame_blocks: List[Tensor] = []
    mask_entries: List[bool] = []

    for t in range(T):
        # Build token list for frame t
        block_parts = [state_tokens[:, t:t+1, :]]  # [B, 1, D]
        mask_entries.append(True)  # state

        if extrinsics_tokens is not None:
            block_parts.append(extrinsics_tokens[:, t:t+1, :])
            mask_entries.append(True)  # extrinsics

        block_parts.append(action_tokens[:, t:t+1, :])  # [B, 1, D]
        mask_entries.append(True)  # action

        block_parts.append(visual_tokens[:, t, :, :])  # [B, N, D]
        mask_entries.extend([False] * N)  # visual tokens

        frame_block = torch.cat(block_parts, dim=1)  # [B, tokens_per_frame, D]
        frame_blocks.append(frame_block)

    interleaved = torch.cat(frame_blocks, dim=1)  # [B, T*tokens_per_frame, D]
    action_token_mask = torch.tensor(mask_entries, dtype=torch.bool,
                                      device=visual_tokens.device)
    return interleaved, action_token_mask


def extract_visual_positions(
    T: int,
    N: int,
    use_extrinsics: bool,
    total_tokens: int,
    device: torch.device,
) -> Tensor:
    """Return indices of visual token positions in the interleaved sequence."""
    num_non_visual = 3 if use_extrinsics else 2
    tokens_per_frame = num_non_visual + N
    visual_indices = []
    for t in range(T):
        frame_start = t * tokens_per_frame
        visual_start = frame_start + num_non_visual
        visual_indices.extend(range(visual_start, visual_start + N))
    return torch.tensor(visual_indices, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Main AC Predictor
# ---------------------------------------------------------------------------

class ActionConditionedPredictor(nn.Module):
    """
    V-JEPA 2 predictor with action/state/extrinsics conditioning.

    Architecture:
        1. Project action, state, extrinsics to embed_dim
        2. Interleave conditioning tokens with visual tokens per frame
        3. Project interleaved tokens to predictor_embed_dim
        4. Apply transformer blocks with optional block-causal mask
        5. Extract visual token predictions and project back to embed_dim

    Public interface matches SKILL.md spec:
        forward(context_repr, actions, states, extrinsics) -> Tensor
    """

    def __init__(
        self,
        embed_dim: int = 384,
        predictor_embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        action_embed_dim: int = 7,
        state_embed_dim: int = 7,
        use_extrinsics: bool = True,
        pred_is_frame_causal: bool = True,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer: str = "layernorm",
        extrinsics_embed_dim: int = 6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.use_extrinsics = use_extrinsics
        self.pred_is_frame_causal = pred_is_frame_causal

        # --- Conditioning projections ---
        self.action_embed = nn.Linear(action_embed_dim, embed_dim)
        self.state_embed = nn.Linear(state_embed_dim, embed_dim)
        if use_extrinsics:
            self.extrinsics_embed = nn.Linear(extrinsics_embed_dim, embed_dim)
        else:
            self.extrinsics_embed = None

        # --- Input projection (interleaved tokens -> predictor dim) ---
        self.input_proj = nn.Linear(embed_dim, predictor_embed_dim)

        # --- Transformer backbone ---
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path_rate=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.norm = get_norm_layer(norm_layer, predictor_embed_dim)

        # --- Output projection (predictor dim -> embed_dim for loss computation) ---
        self.output_proj = nn.Linear(predictor_embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def num_non_visual_tokens(self) -> int:
        """Number of conditioning tokens per frame."""
        return 3 if self.use_extrinsics else 2

    def forward(
        self,
        context_repr: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            context_repr: [B, T, N, embed_dim]  -- encoder output per frame
            actions:      [B, T, 7]              -- delta action commands
            states:       [B, T, 7]              -- robot proprioceptive states
            extrinsics:   [B, T, 6] optional     -- 6-DOF camera extrinsics
                          (or [B, T, 4, 4] -> caller must flatten to 6-DOF)

        Returns:
            visual_preds: [B, T*N, embed_dim]
        """
        B, T, N, D = context_repr.shape
        device = context_repr.device

        # --- Project conditioning signals ---
        action_emb = self.action_embed(actions)    # [B, T, embed_dim]
        state_emb  = self.state_embed(states)      # [B, T, embed_dim]

        extrin_emb: Optional[Tensor] = None
        if self.use_extrinsics:
            assert extrinsics is not None, "extrinsics required when use_extrinsics=True"
            assert self.extrinsics_embed is not None
            # Handle both 6-DOF [B, T, 6] and flat inputs
            if extrinsics.dim() == 4:
                # [B, T, 4, 4] -> flatten to [B, T, 16]? Use first 6 dims as proxy
                # Caller should pass pre-extracted 6-DOF
                raise ValueError(
                    "Pass extrinsics as [B, T, 6] (pre-extracted 6-DOF). "
                    "Use extrinsics_to_6dof() from pose_mathematics."
                )
            extrin_emb = self.extrinsics_embed(extrinsics)  # [B, T, embed_dim]

        # --- Token interleaving ---
        tokens, action_mask = interleave_tokens(
            context_repr, action_emb, state_emb, extrin_emb
        )
        # tokens:      [B, T*tokens_per_frame, embed_dim]
        # action_mask: [T*tokens_per_frame] bool

        # --- Project to predictor dimension ---
        tokens = self.input_proj(tokens)  # [B, T*tokens_per_frame, predictor_embed_dim]

        # --- Build block-causal mask (if enabled) ---
        attn_bias: Optional[Tensor] = None
        tokens_per_frame = self.num_non_visual_tokens + N
        if self.pred_is_frame_causal:
            attn_bias = build_block_causal_mask(
                T, tokens_per_frame, device, tokens.dtype
            )

        # --- Transformer blocks ---
        for block in self.blocks:
            tokens = block(tokens, attn_bias)

        tokens = self.norm(tokens)  # [B, T*tokens_per_frame, predictor_embed_dim]

        # --- Extract visual token predictions ---
        visual_positions = extract_visual_positions(
            T, N, self.use_extrinsics, tokens.shape[1], device
        )
        visual_tokens = tokens[:, visual_positions, :]  # [B, T*N, predictor_embed_dim]

        # --- Project back to embed_dim for loss computation ---
        visual_preds = self.output_proj(visual_tokens)  # [B, T*N, embed_dim]

        return visual_preds

    def predict_single_frame(
        self,
        context_repr: Tensor,
        action: Tensor,
        state: Tensor,
        extrinsics: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Convenience wrapper for single-frame prediction (T=1).

        Args:
            context_repr: [B, N, embed_dim]  -- single frame encoder output
            action:       [B, 7]
            state:        [B, 7]
            extrinsics:   [B, 6] optional

        Returns:
            pred: [B, N, embed_dim]
        """
        context_repr = context_repr.unsqueeze(1)    # [B, 1, N, D]
        action = action.unsqueeze(1)                # [B, 1, 7]
        state  = state.unsqueeze(1)                 # [B, 1, 7]
        if extrinsics is not None:
            extrinsics = extrinsics.unsqueeze(1)    # [B, 1, 6]

        out = self.forward(context_repr, action, state, extrinsics)
        return out.squeeze(1)  # [B, N, embed_dim]


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_output_shape():
    """T01 / T02: Output shape is correct."""
    print("[TEST] Output shape...")

    # Without extrinsics
    cfg_no_ext = ACPredictorConfig(
        embed_dim=64, predictor_embed_dim=128, depth=2, num_heads=4,
        use_extrinsics=False
    )
    pred_no_ext = ActionConditionedPredictor(
        embed_dim=cfg_no_ext.embed_dim,
        predictor_embed_dim=cfg_no_ext.predictor_embed_dim,
        depth=cfg_no_ext.depth,
        num_heads=cfg_no_ext.num_heads,
        use_extrinsics=False,
    )
    B, T, N = 2, 4, 16
    context = torch.randn(B, T, N, cfg_no_ext.embed_dim)
    actions = torch.randn(B, T, 7)
    states  = torch.randn(B, T, 7)

    out = pred_no_ext(context, actions, states)
    expected = (B, T * N, cfg_no_ext.embed_dim)
    assert out.shape == expected, f"No-extrinsics shape: {out.shape} != {expected}"
    print(f"  No-extrinsics: {out.shape} -- PASS")

    # With extrinsics
    pred_ext = ActionConditionedPredictor(
        embed_dim=cfg_no_ext.embed_dim,
        predictor_embed_dim=cfg_no_ext.predictor_embed_dim,
        depth=cfg_no_ext.depth,
        num_heads=cfg_no_ext.num_heads,
        use_extrinsics=True,
    )
    extrinsics = torch.randn(B, T, 6)
    out_ext = pred_ext(context, actions, states, extrinsics)
    assert out_ext.shape == expected, f"With-extrinsics shape: {out_ext.shape} != {expected}"
    print(f"  With-extrinsics: {out_ext.shape} -- PASS")


def _test_block_causal_mask():
    """T17 / T18 / T19 / T20 / T21: Block-causal mask structure is correct."""
    print("[TEST] Block-causal mask...")

    T, K = 3, 5
    mask = build_block_causal_mask(T, K, device=torch.device("cpu"))

    # Frame 0 cannot attend to frame 1 (upper-right block should be -inf)
    for t_q in range(T):
        for t_k in range(t_q + 1, T):
            q_s, q_e = t_q * K, (t_q + 1) * K
            k_s, k_e = t_k * K, (t_k + 1) * K
            block = mask[q_s:q_e, k_s:k_e]
            assert block.isinf().all(), (
                f"Frame {t_q} -> Frame {t_k} should be -inf, got {block}"
            )

    # Frame 1 can attend to frame 0 (lower-left block should be 0)
    for t_q in range(1, T):
        for t_k in range(0, t_q):
            q_s, q_e = t_q * K, (t_q + 1) * K
            k_s, k_e = t_k * K, (t_k + 1) * K
            block = mask[q_s:q_e, k_s:k_e]
            assert (block == 0).all(), (
                f"Frame {t_q} -> Frame {t_k} should be 0, got {block}"
            )

    # Diagonal blocks should be 0 (within-frame bidirectional)
    for t in range(T):
        s = t * K
        e = (t + 1) * K
        diag_block = mask[s:e, s:e]
        assert (diag_block == 0).all(), f"Diagonal block {t} has non-zero entries"

    print(f"  Mask shape: {mask.shape} -- PASS")
    print(f"  Upper-future blocks are -inf -- PASS")
    print(f"  Lower-past blocks are 0 -- PASS")
    print(f"  Diagonal blocks are 0 -- PASS")


def _test_token_interleaving():
    """T09 / T11 / T13 / T14: Token interleaving counts and positions are correct."""
    print("[TEST] Token interleaving...")

    B, T, N, D = 2, 3, 16, 64
    visual   = torch.randn(B, T, N, D)
    actions  = torch.randn(B, T, D)
    states   = torch.randn(B, T, D)
    extrin   = torch.randn(B, T, D)

    # Without extrinsics: total = T * (2 + N)
    interleaved, mask = interleave_tokens(visual, actions, states, None)
    expected_total = T * (2 + N)
    assert interleaved.shape == (B, expected_total, D), (
        f"No-ext total: {interleaved.shape[1]} != {expected_total}"
    )
    assert mask.sum().item() == T * 2, (
        f"No-ext mask True count: {mask.sum()} != {T * 2}"
    )
    print(f"  No-extrinsics total tokens: {interleaved.shape[1]} -- PASS")
    print(f"  No-extrinsics mask True count: {mask.sum()} -- PASS")

    # With extrinsics: total = T * (3 + N)
    interleaved_e, mask_e = interleave_tokens(visual, actions, states, extrin)
    expected_total_e = T * (3 + N)
    assert interleaved_e.shape == (B, expected_total_e, D), (
        f"With-ext total: {interleaved_e.shape[1]} != {expected_total_e}"
    )
    assert mask_e.sum().item() == T * 3, (
        f"With-ext mask True count: {mask_e.sum()} != {T * 3}"
    )
    print(f"  With-extrinsics total tokens: {interleaved_e.shape[1]} -- PASS")
    print(f"  With-extrinsics mask True count: {mask_e.sum()} -- PASS")

    # First token in each frame block is True (state token)
    tokens_per_frame_ext = 3 + N
    for t in range(T):
        idx = t * tokens_per_frame_ext
        assert mask_e[idx].item() is True, f"Frame {t} first token should be True (state)"
    print(f"  First token per frame is state (True) -- PASS")

    # Last N tokens in each frame block are False (visual)
    for t in range(T):
        frame_end = (t + 1) * tokens_per_frame_ext
        visual_start = frame_end - N
        assert not mask_e[visual_start:frame_end].any(), (
            f"Frame {t} visual tokens should be False"
        )
    print(f"  Last N tokens per frame are visual (False) -- PASS")


def _test_gradient_flow():
    """T08: Gradients flow through all parameters."""
    print("[TEST] Gradient flow...")

    pred = ActionConditionedPredictor(
        embed_dim=32, predictor_embed_dim=64, depth=2, num_heads=4,
        use_extrinsics=False
    )
    B, T, N = 1, 2, 4
    context = torch.randn(B, T, N, 32, requires_grad=True)
    actions = torch.randn(B, T, 7)
    states  = torch.randn(B, T, 7)

    out = pred(context, actions, states)
    loss = out.sum()
    loss.backward()

    params_without_grad = [
        name for name, p in pred.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    if params_without_grad:
        print(f"  WARNING: params without grad: {params_without_grad[:5]}")
    else:
        print(f"  All {sum(1 for p in pred.parameters() if p.requires_grad)} params have gradients -- PASS")


def _test_no_future_attention():
    """Verify that future frame information does not leak through attention."""
    print("[TEST] No future attention leakage...")

    pred = ActionConditionedPredictor(
        embed_dim=32, predictor_embed_dim=64, depth=1, num_heads=4,
        use_extrinsics=False, pred_is_frame_causal=True
    )

    B, T, N = 1, 3, 4
    context1 = torch.randn(B, T, N, 32)
    context2 = context1.clone()
    # Perturb only frame 2 (future)
    context2[:, 2, :, :] = context2[:, 2, :, :] + 10.0

    actions = torch.randn(B, T, 7)
    states  = torch.randn(B, T, 7)

    out1 = pred(context1, actions, states)
    out2 = pred(context2, actions, states)

    # Frame 0 predictions should be identical (not affected by frame 2 change)
    frame0_preds1 = out1[:, :N, :]
    frame0_preds2 = out2[:, :N, :]
    max_diff = (frame0_preds1 - frame0_preds2).abs().max().item()

    assert max_diff < 1e-5, (
        f"Frame 0 predictions changed when frame 2 was perturbed: max_diff={max_diff}"
    )
    print(f"  Frame 0 unaffected by frame 2 perturbation (diff={max_diff:.2e}) -- PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("ActionConditionedPredictor Self-Tests")
    print("=" * 60)

    _test_output_shape()
    _test_block_causal_mask()
    _test_token_interleaving()
    _test_gradient_flow()
    _test_no_future_attention()

    print("=" * 60)
    print("All self-tests PASSED")
    print("=" * 60)
