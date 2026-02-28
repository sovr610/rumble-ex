#!/usr/bin/env python3
"""
gen_vit_tests.py — Generates 100+ pytest test cases for the V-JEPA 2 ViT infrastructure.

Covers:
- Forward shapes per variant (Tiny through Gigantic)
- Video vs image input
- Masking (single mask, multiple masks, no masking)
- out_layers (single, multiple, last equals default)
- RoPE equivariance and relative position property
- SwiGLU hidden dim alignment
- SDPA vs manual attention equivalence
- Positional embedding interpolation
- Cross-attention output shapes
- AttentivePooler (single query, multi-query, with classifier)
- DropPath behavior
- Config factory functions
- Parameter count estimation

Output: Valid Python pytest module, written to stdout.
Run as: python scripts/gen_vit_tests.py > tests/test_generated_vit.py

Usage:
    python scripts/gen_vit_tests.py                   # print to stdout
    python scripts/gen_vit_tests.py --count           # just print count
"""

import sys
import textwrap


# ---------------------------------------------------------------------------
# Test case registry
# ---------------------------------------------------------------------------

TEST_CASES = []


def tc(name: str, code: str, marks: str = "", category: str = "general") -> None:
    """Register a test case."""
    TEST_CASES.append({
        "name": name,
        "code": code,
        "marks": marks,
        "category": category,
    })


# ---------------------------------------------------------------------------
# Forward Shape Tests — Image Input
# ---------------------------------------------------------------------------

VARIANTS_2D = [
    ("tiny",     192,  12,  3),
    ("small",    384,  12,  6),
    ("base",     768,  12, 12),
    ("large",   1024,  24, 16),
    ("huge",    1280,  32, 16),
    ("giant",   1408,  40, 16),
    ("gigantic",1664,  48, 16),
]

for name, embed_dim, depth, num_heads in VARIANTS_2D:
    tc(
        name=f"test_forward_2d_{name}",
        category="forward_shape",
        code=f"""\
def test_forward_2d_{name}():
    model = VisionTransformer(
        img_size=224, patch_size=16,
        embed_dim={embed_dim}, depth={depth}, num_heads={num_heads},
    )
    model.training = False
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 196, {embed_dim}), f"Got {{out.shape}}"
    assert not out.isnan().any(), "NaN in output"
""",
    )

# ---------------------------------------------------------------------------
# Forward Shape Tests — Video Input
# ---------------------------------------------------------------------------

VIDEO_CONFIGS = [
    ("tiny_video_8f",  192, 12,  3,  8, 2, 224, 4*14*14),
    ("base_video_8f",  768, 12, 12,  8, 2, 224, 4*14*14),
    ("base_video_16f", 768, 12, 12, 16, 2, 224, 8*14*14),
    ("large_video_16f",1024, 24, 16, 16, 2, 224, 8*14*14),
    ("base_video_32f", 768, 12, 12, 32, 2, 224, 16*14*14),
]

for cfg_name, embed_dim, depth, num_heads, T, tubelet, img_size, N_expected in VIDEO_CONFIGS:
    tc(
        name=f"test_forward_video_{cfg_name}",
        category="forward_shape",
        code=f"""\
def test_forward_video_{cfg_name}():
    model = VisionTransformer(
        img_size={img_size}, patch_size=16, tubelet_size={tubelet},
        embed_dim={embed_dim}, depth={depth}, num_heads={num_heads},
    )
    model.training = False
    x = torch.randn(2, 3, {T}, {img_size}, {img_size})
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, {N_expected}, {embed_dim}), f"Got {{out.shape}}"
""",
    )

# ---------------------------------------------------------------------------
# Masking Tests
# ---------------------------------------------------------------------------

tc(
    name="test_masking_single_reduces_tokens",
    category="masking",
    code="""\
def test_masking_single_reduces_tokens():
    model = VisionTransformer(embed_dim=192, depth=2, num_heads=3)
    model.training = False
    N_total, n_keep, B = 196, 49, 2
    masks = [torch.randperm(N_total)[:n_keep].unsqueeze(0).expand(B, -1)]
    x = torch.randn(B, 3, 224, 224)
    with torch.no_grad():
        out = model(x, masks=masks)
    assert out.shape == (B, n_keep, 192)
""",
)

tc(
    name="test_masking_multiple_masks_stack",
    category="masking",
    code="""\
def test_masking_multiple_masks_stack():
    model = VisionTransformer(embed_dim=192, depth=2, num_heads=3)
    model.training = False
    N_total, n_keep, B, M = 196, 49, 2, 3
    masks = [torch.randperm(N_total)[:n_keep].unsqueeze(0).expand(B, -1) for _ in range(M)]
    x = torch.randn(B, 3, 224, 224)
    with torch.no_grad():
        out = model(x, masks=masks)
    assert out.shape == (B * M, n_keep, 192)
""",
)

tc(
    name="test_masking_keep_all",
    category="masking",
    code="""\
def test_masking_keep_all():
    "Keeping all tokens should give same result as no masking."
    model = VisionTransformer(embed_dim=192, depth=2, num_heads=3)
    model.training = False
    N_total = 196
    B = 2
    # Keep all tokens in sorted order
    masks = [torch.arange(N_total).unsqueeze(0).expand(B, -1)]
    x = torch.randn(B, 3, 224, 224)
    with torch.no_grad():
        out_masked = model(x, masks=masks)
        out_no_mask = model(x, masks=None)
    assert out_masked.shape == (B, N_total, 192)
    # Values should be same when all tokens kept in original order
    assert torch.allclose(out_masked, out_no_mask, atol=1e-5)
""",
)

for n_keep in [10, 50, 100, 150, 196]:
    tc(
        name=f"test_masking_n_keep_{n_keep}",
        category="masking",
        code=f"""\
def test_masking_n_keep_{n_keep}():
    model = VisionTransformer(embed_dim=192, depth=2, num_heads=3)
    model.training = False
    N_total = 196
    B = 2
    masks = [torch.randperm(N_total)[:{n_keep}].unsqueeze(0).expand(B, -1)]
    x = torch.randn(B, 3, 224, 224)
    with torch.no_grad():
        out = model(x, masks=masks)
    assert out.shape == (B, {n_keep}, 192)
""",
    )

# ---------------------------------------------------------------------------
# out_layers Tests
# ---------------------------------------------------------------------------

tc(
    name="test_out_layers_single",
    category="out_layers",
    code="""\
def test_out_layers_single():
    model = VisionTransformer(embed_dim=192, depth=6, num_heads=3)
    model.training = False
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x, out_layers=[2])
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0].shape == (2, 196, 192)
""",
)

tc(
    name="test_out_layers_multiple",
    category="out_layers",
    code="""\
def test_out_layers_multiple():
    model = VisionTransformer(embed_dim=192, depth=6, num_heads=3)
    model.training = False
    x = torch.randn(2, 3, 224, 224)
    layers = [0, 2, 4, 5]
    with torch.no_grad():
        outs = model(x, out_layers=layers)
    assert len(outs) == len(layers)
    for o in outs:
        assert o.shape == (2, 196, 192)
""",
)

tc(
    name="test_out_layers_last_equals_final",
    category="out_layers",
    code="""\
def test_out_layers_last_equals_final():
    "out_layers=[depth-1] should equal the default forward output."
    D = 6
    model = VisionTransformer(embed_dim=192, depth=D, num_heads=3)
    model.training = False
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out_default = model(x)
        out_last = model(x, out_layers=[D - 1])
    assert torch.allclose(out_default, out_last[0], atol=1e-5)
""",
)

tc(
    name="test_out_layers_returns_list",
    category="out_layers",
    code="""\
def test_out_layers_returns_list():
    model = VisionTransformer(embed_dim=192, depth=4, num_heads=3)
    model.training = False
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x, out_layers=[0])
    assert isinstance(out, list), f"Expected list, got {type(out)}"
""",
)

# ---------------------------------------------------------------------------
# RoPE Tests
# ---------------------------------------------------------------------------

tc(
    name="test_rope_position_sensitivity",
    category="rope",
    code="""\
def test_rope_position_sensitivity():
    rope = RoPE3D(head_dim=60)
    q = torch.randn(1, 1, 4, 60)
    k = torch.randn(1, 1, 4, 60)
    q1, k1 = rope.apply_rope(q.clone(), k.clone(), grid_depth=2, grid_h=2, grid_w=1)
    q_perm = q[:, :, [2, 3, 0, 1], :]
    k_perm = k[:, :, [2, 3, 0, 1], :]
    q2, k2 = rope.apply_rope(q_perm, k_perm, grid_depth=2, grid_h=2, grid_w=1)
    assert not torch.allclose(q1[:, :, 0, :], q2[:, :, 0, :], atol=1e-5)
""",
)

tc(
    name="test_rope_deterministic",
    category="rope",
    code="""\
def test_rope_deterministic():
    rope = RoPE3D(head_dim=60)
    q = torch.randn(2, 4, 6, 60)
    k = torch.randn(2, 4, 6, 60)
    q1, k1 = rope.apply_rope(q, k, 2, 3, 1)
    q2, k2 = rope.apply_rope(q, k, 2, 3, 1)
    assert torch.allclose(q1, q2)
    assert torch.allclose(k1, k2)
""",
)

tc(
    name="test_rope_preserves_shape",
    category="rope",
    code="""\
@pytest.mark.parametrize("head_dim,gd,gh,gw", [
    (60,  2, 3, 1),
    (60,  4, 7, 7),
    (72,  8, 14, 14),
])
def test_rope_preserves_shape(head_dim, gd, gh, gw):
    rope = RoPE3D(head_dim=head_dim)
    N = gd * gh * gw
    q = torch.randn(2, 4, N, head_dim)
    k = torch.randn(2, 4, N, head_dim)
    q_out, k_out = rope.apply_rope(q, k, gd, gh, gw)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
""",
)

tc(
    name="test_rope_relative_position_property",
    category="rope",
    code="""\
def test_rope_relative_position_property():
    rope = RoPE3D(head_dim=60)
    torch.manual_seed(7)
    q = torch.randn(1, 1, 2, 60)
    k = torch.randn(1, 1, 2, 60)
    # Positions 0 and 1 (relative distance = 1)
    q1, k1 = rope.apply_rope(q.clone(), k.clone(), grid_depth=2, grid_h=1, grid_w=1)
    logit1 = (q1[:, :, 0, :] * k1[:, :, 1, :]).sum(-1)
    # Positions 3 and 4 (relative distance = 1, shifted by 3)
    q5 = torch.cat([torch.zeros(1, 1, 3, 60), q[:, :, :2, :]], dim=2)
    k5 = torch.cat([torch.zeros(1, 1, 3, 60), k[:, :, :2, :]], dim=2)
    q5r, k5r = rope.apply_rope(q5, k5, grid_depth=5, grid_h=1, grid_w=1)
    logit2 = (q5r[:, :, 3, :] * k5r[:, :, 4, :]).sum(-1)
    diff = abs(logit1.item() - logit2.item())
    assert diff < 1e-3, f"Relative position property violated: diff={diff:.2e}"
""",
)

tc(
    name="test_rope_axis_dim_computation",
    category="rope",
    code="""\
@pytest.mark.parametrize("head_dim,expected_axis_dim", [
    (60, 20),
    (64, 20),
    (72, 24),
    (80, 24),
    (84, 28),
    (88, 28),
    (96, 32),
    (104, 32),
])
def test_rope_axis_dim_computation(head_dim, expected_axis_dim):
    rope = RoPE3D(head_dim=head_dim)
    assert rope.axis_dim == expected_axis_dim, \\
        f"head_dim={head_dim}: axis_dim={rope.axis_dim} != {expected_axis_dim}"
    assert rope.rotated_dims == 3 * expected_axis_dim
""",
)

tc(
    name="test_rope_freqs_shape",
    category="rope",
    code="""\
def test_rope_freqs_shape():
    for head_dim in [60, 64, 80, 88]:
        rope = RoPE3D(head_dim=head_dim)
        axis_dim = 2 * (head_dim // 6)
        assert rope.freqs.shape == (axis_dim // 2,), \\
            f"head_dim={head_dim}: freqs shape {rope.freqs.shape}"
""",
)

# ---------------------------------------------------------------------------
# SwiGLU Tests
# ---------------------------------------------------------------------------

SWIGLU_CASES = [
    (192,  4.0),
    (384,  4.0),
    (768,  4.0),
    (1024, 4.0),
    (1280, 4.0),
    (1408, 48/11),
    (1664, 64/11),
]

for embed_dim, mlp_ratio in SWIGLU_CASES:
    tc(
        name=f"test_swiglu_dim_aligned_embed{embed_dim}",
        category="swiglu",
        code=f"""\
def test_swiglu_dim_aligned_embed{embed_dim}():
    import math
    hidden = int({embed_dim} * {mlp_ratio})
    mlp = SwiGLU(in_features={embed_dim}, hidden_features=hidden, wide_silu=True)
    assert mlp.hidden_dim % 8 == 0, \\
        f"hidden_dim={{mlp.hidden_dim}} not divisible by 8"
""",
    )

tc(
    name="test_swiglu_output_shape",
    category="swiglu",
    code="""\
@pytest.mark.parametrize("B,N,D", [(2, 196, 768), (1, 784, 1024), (4, 49, 384)])
def test_swiglu_output_shape(B, N, D):
    mlp = SwiGLU(in_features=D, hidden_features=D * 4)
    x = torch.randn(B, N, D)
    out = mlp(x)
    assert out.shape == (B, N, D)
""",
)

tc(
    name="test_swiglu_more_params_than_mlp",
    category="swiglu",
    code="""\
def test_swiglu_more_params_than_mlp():
    D, hidden = 768, 3072
    mlp = MLP(in_features=D, hidden_features=hidden)
    swiglu = SwiGLU(in_features=D, hidden_features=hidden)
    assert sum(p.numel() for p in swiglu.parameters()) > \\
           sum(p.numel() for p in mlp.parameters())
""",
)

# ---------------------------------------------------------------------------
# SDPA Equivalence Tests
# ---------------------------------------------------------------------------

tc(
    name="test_sdpa_matches_manual_attention",
    category="attention",
    code="""\
@pytest.mark.parametrize("D,H,N", [(256, 8, 64), (192, 3, 49), (384, 6, 196)])
def test_sdpa_matches_manual_attention(D, H, N):
    torch.manual_seed(42)
    attn_manual = Attention(dim=D, num_heads=H, use_sdpa=False)
    attn_sdpa   = Attention(dim=D, num_heads=H, use_sdpa=True)
    attn_sdpa.load_state_dict(attn_manual.state_dict())
    attn_manual.training = False
    attn_sdpa.training = False
    x = torch.randn(2, N, D)
    with torch.no_grad():
        out_m = attn_manual(x)
        out_s = attn_sdpa(x)
    assert torch.allclose(out_m, out_s, atol=1e-4), \\
        f"Max diff: {(out_m - out_s).abs().max():.2e}"
""",
)

tc(
    name="test_attention_output_shape",
    category="attention",
    code="""\
@pytest.mark.parametrize("D,H,N", [(768, 12, 196), (192, 3, 49), (1024, 16, 784)])
def test_attention_output_shape(D, H, N):
    attn = Attention(dim=D, num_heads=H)
    x = torch.randn(2, N, D)
    out = attn(x)
    assert out.shape == (2, N, D)
""",
)

# ---------------------------------------------------------------------------
# Positional Embedding Interpolation Tests
# ---------------------------------------------------------------------------

tc(
    name="test_interp_224_to_384",
    category="interpolation",
    code="""\
def test_interp_224_to_384():
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=192, depth=2, num_heads=3)
    model.training = False
    x = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        out = model(x)
    N = (384 // 16) ** 2
    assert out.shape == (1, N, 192)
""",
)

tc(
    name="test_interp_video_temporal",
    category="interpolation",
    code="""\
def test_interp_video_temporal():
    model = VisionTransformer(img_size=224, patch_size=16, tubelet_size=2,
                              embed_dim=192, depth=2, num_heads=3)
    model.training = False
    x = torch.randn(1, 3, 16, 256, 256)
    with torch.no_grad():
        out = model(x)
    N = (16 // 2) * (256 // 16) ** 2
    assert out.shape == (1, N, 192)
    assert not out.isnan().any()
""",
)

tc(
    name="test_interp_pos_embed_frozen",
    category="interpolation",
    code="""\
def test_interp_pos_embed_frozen():
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=192, depth=2, num_heads=3)
    model.training = False
    assert not model.pos_embed.requires_grad
    before = model.pos_embed.clone()
    x = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        _ = model(x)
    assert torch.allclose(model.pos_embed, before)
""",
)

for src_px, dst_px in [(224, 448), (256, 512), (112, 224)]:
    N = (dst_px // 16) ** 2
    tc(
        name=f"test_interp_{src_px}px_to_{dst_px}px",
        category="interpolation",
        code=f"""\
def test_interp_{src_px}px_to_{dst_px}px():
    model = VisionTransformer(img_size={src_px}, patch_size=16, embed_dim=192, depth=2, num_heads=3)
    model.training = False
    x = torch.randn(1, 3, {dst_px}, {dst_px})
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, {N}, 192)
""",
    )

# ---------------------------------------------------------------------------
# Cross-Attention Tests
# ---------------------------------------------------------------------------

tc(
    name="test_cross_attention_shape",
    category="cross_attention",
    code="""\
@pytest.mark.parametrize("B,Nq,Nkv,D,H", [
    (2, 4, 196, 768, 12),
    (1, 1, 784, 1024, 16),
    (4, 8, 49, 384, 6),
])
def test_cross_attention_shape(B, Nq, Nkv, D, H):
    ca = CrossAttention(dim=D, num_heads=H)
    ca.training = False
    q = torch.randn(B, Nq, D)
    kv = torch.randn(B, Nkv, D)
    with torch.no_grad():
        out = ca(q, kv)
    assert out.shape == (B, Nq, D)
""",
)

tc(
    name="test_cross_attention_sdpa_vs_manual",
    category="cross_attention",
    code="""\
def test_cross_attention_sdpa_vs_manual():
    torch.manual_seed(0)
    D, H = 256, 8
    ca_m = CrossAttention(dim=D, num_heads=H, use_sdpa=False)
    ca_s = CrossAttention(dim=D, num_heads=H, use_sdpa=True)
    ca_s.load_state_dict(ca_m.state_dict())
    ca_m.training = False
    ca_s.training = False
    q = torch.randn(2, 4, D)
    kv = torch.randn(2, 32, D)
    with torch.no_grad():
        out_m = ca_m(q, kv)
        out_s = ca_s(q, kv)
    assert torch.allclose(out_m, out_s, atol=1e-4)
""",
)

# ---------------------------------------------------------------------------
# AttentivePooler Tests
# ---------------------------------------------------------------------------

tc(
    name="test_pooler_single_query_shape",
    category="attentive_pooler",
    code="""\
@pytest.mark.parametrize("B,N,D,H", [(2, 196, 768, 12), (1, 784, 1024, 16)])
def test_pooler_single_query_shape(B, N, D, H):
    pooler = AttentivePooler(num_queries=1, embed_dim=D, num_heads=H)
    pooler.training = False
    x = torch.randn(B, N, D)
    with torch.no_grad():
        out = pooler(x)
    assert out.shape == (B, 1, D)
""",
)

tc(
    name="test_pooler_multi_query_shape",
    category="attentive_pooler",
    code="""\
@pytest.mark.parametrize("Nq", [1, 4, 8, 16])
def test_pooler_multi_query_shape(Nq):
    D, H = 768, 12
    pooler = AttentivePooler(num_queries=Nq, embed_dim=D, num_heads=H)
    pooler.training = False
    x = torch.randn(2, 196, D)
    with torch.no_grad():
        out = pooler(x)
    assert out.shape == (2, Nq, D)
""",
)

tc(
    name="test_pooler_classifier_shape",
    category="attentive_pooler",
    code="""\
@pytest.mark.parametrize("num_classes", [10, 400, 1000])
def test_pooler_classifier_shape(num_classes):
    D, H = 768, 12
    pooler = AttentivePooler(num_queries=1, embed_dim=D, num_heads=H, num_classes=num_classes)
    pooler.training = False
    x = torch.randn(2, 196, D)
    with torch.no_grad():
        out = pooler(x)
    assert out.shape == (2, num_classes)
""",
)

tc(
    name="test_pooler_query_tokens_learnable",
    category="attentive_pooler",
    code="""\
def test_pooler_query_tokens_learnable():
    pooler = AttentivePooler(num_queries=1, embed_dim=768, num_heads=12)
    assert pooler.query_tokens.requires_grad
    assert pooler.query_tokens.shape == (1, 1, 768)
""",
)

tc(
    name="test_pooler_gradient_flow",
    category="attentive_pooler",
    code="""\
def test_pooler_gradient_flow():
    pooler = AttentivePooler(num_queries=1, embed_dim=192, num_heads=3, num_classes=10)
    x = torch.randn(2, 196, 192, requires_grad=True)
    logits = pooler(x)
    logits.sum().backward()
    assert x.grad is not None
    assert not x.grad.isnan().any()
""",
)

# ---------------------------------------------------------------------------
# DropPath Tests
# ---------------------------------------------------------------------------

tc(
    name="test_droppath_zero_is_identity",
    category="droppath",
    code="""\
def test_droppath_zero_is_identity():
    dp = DropPath(drop_prob=0.0)
    dp.training = True
    x = torch.randn(4, 196, 768)
    out = dp(x)
    assert torch.allclose(x, out)
""",
)

tc(
    name="test_droppath_inference_is_identity",
    category="droppath",
    code="""\
def test_droppath_inference_is_identity():
    dp = DropPath(drop_prob=0.5)
    dp.training = False
    x = torch.randn(4, 196, 768)
    out = dp(x)
    assert torch.allclose(x, out)
""",
)

tc(
    name="test_droppath_full_drops",
    category="droppath",
    code="""\
def test_droppath_full_drops():
    dp = DropPath(drop_prob=1.0)
    dp.training = True
    x = torch.ones(8, 196, 768)
    out = dp(x)
    assert out.abs().sum().item() == 0.0
""",
)

# ---------------------------------------------------------------------------
# Config Factory Tests
# ---------------------------------------------------------------------------

FACTORY_EXPECTED = [
    ("vit_tiny",     192,  12,  3),
    ("vit_small",    384,  12,  6),
    ("vit_base",     768,  12, 12),
    ("vit_large",   1024,  24, 16),
    ("vit_huge",    1280,  32, 16),
    ("vit_giant",   1408,  40, 16),
    ("vit_gigantic",1664,  48, 16),
]

for fname, embed_dim, depth, num_heads in FACTORY_EXPECTED:
    tc(
        name=f"test_config_factory_{fname}",
        category="config",
        code=f"""\
def test_config_factory_{fname}():
    from vit_config_template import {fname}
    cfg = {fname}()
    assert cfg.embed_dim == {embed_dim}
    assert cfg.depth == {depth}
    assert cfg.num_heads == {num_heads}
    assert cfg.embed_dim % cfg.num_heads == 0
    assert cfg.model_name == "{fname}"
""",
    )

tc(
    name="test_config_registry_complete",
    category="config",
    code="""\
def test_config_registry_complete():
    from vit_config_template import VARIANT_REGISTRY
    expected = {"vit_tiny", "vit_small", "vit_base", "vit_large",
                "vit_huge", "vit_giant", "vit_gigantic"}
    assert set(VARIANT_REGISTRY.keys()) == expected
""",
)

tc(
    name="test_config_kwargs_override",
    category="config",
    code="""\
def test_config_kwargs_override():
    from vit_config_template import vit_base
    cfg = vit_base(img_size=384, use_rope=True, drop_path_rate=0.1)
    assert cfg.img_size == 384
    assert cfg.use_rope is True
    assert cfg.drop_path_rate == 0.1
    assert cfg.embed_dim == 768  # unchanged
""",
)

tc(
    name="test_config_invalid_name_raises",
    category="config",
    code="""\
def test_config_invalid_name_raises():
    from vit_config_template import get_vit_config
    with pytest.raises(ValueError):
        get_vit_config("vit_nonexistent")
""",
)

tc(
    name="test_config_param_count_ranges",
    category="config",
    code="""\
@pytest.mark.parametrize("name,lo_m,hi_m", [
    ("vit_tiny",      4,   12),
    ("vit_small",    15,   30),
    ("vit_base",     60,  110),
    ("vit_large",   250,  380),
    ("vit_huge",    550,  750),
    ("vit_giant",   900, 1400),
    ("vit_gigantic", 1500, 2500),
])
def test_config_param_count_ranges(name, lo_m, hi_m):
    from vit_config_template import VARIANT_REGISTRY, param_count_estimate
    cfg = VARIANT_REGISTRY[name]()
    est = param_count_estimate(cfg) / 1e6
    assert lo_m <= est <= hi_m, \\
        f"{name}: {est:.1f}M not in [{lo_m}M, {hi_m}M]"
""",
)

# ---------------------------------------------------------------------------
# Block-level Tests
# ---------------------------------------------------------------------------

tc(
    name="test_block_output_shape_mlp",
    category="block",
    code="""\
@pytest.mark.parametrize("D,H,N", [(192, 3, 196), (768, 12, 196), (384, 6, 49)])
def test_block_output_shape_mlp(D, H, N):
    block = Block(dim=D, num_heads=H, mlp_ratio=4.0, use_silu=False)
    block.training = False
    x = torch.randn(2, N, D)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (2, N, D)
""",
)

tc(
    name="test_block_output_shape_swiglu",
    category="block",
    code="""\
@pytest.mark.parametrize("D,H", [(192, 3), (768, 12), (1024, 16)])
def test_block_output_shape_swiglu(D, H):
    block = Block(dim=D, num_heads=H, mlp_ratio=4.0, use_silu=True, wide_silu=True)
    block.training = False
    x = torch.randn(2, 196, D)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (2, 196, D)
    assert block.mlp.hidden_dim % 8 == 0
""",
)

tc(
    name="test_block_no_nan",
    category="block",
    code="""\
def test_block_no_nan():
    block = Block(dim=384, num_heads=6, drop_path=0.0)
    block.training = False
    x = torch.randn(2, 196, 384)
    with torch.no_grad():
        out = block(x)
    assert not out.isnan().any()
    assert not out.isinf().any()
""",
)

# ---------------------------------------------------------------------------
# Sincos Embedding Tests
# ---------------------------------------------------------------------------

tc(
    name="test_sincos_2d_shape",
    category="sincos",
    code="""\
@pytest.mark.parametrize("D,G", [(192, 7), (768, 14), (1024, 16)])
def test_sincos_2d_shape(D, G):
    emb = get_2d_sincos_pos_embed(D, G)
    assert emb.shape == (G * G, D)
""",
)

tc(
    name="test_sincos_3d_shape",
    category="sincos",
    code="""\
@pytest.mark.parametrize("D,G,T", [(192, 7, 4), (768, 14, 8), (1024, 14, 16)])
def test_sincos_3d_shape(D, G, T):
    emb = get_3d_sincos_pos_embed(D, G, T)
    assert emb.shape == (T * G * G, D)
""",
)

tc(
    name="test_sincos_uniform_power_differs",
    category="sincos",
    code="""\
def test_sincos_uniform_power_differs():
    e1 = get_3d_sincos_pos_embed(768, 14, 8, uniform_power=False)
    e2 = get_3d_sincos_pos_embed(768, 14, 8, uniform_power=True)
    assert not np.allclose(e1, e2)
""",
)

tc(
    name="test_sincos_values_bounded",
    category="sincos",
    code="""\
def test_sincos_values_bounded():
    emb = get_2d_sincos_pos_embed(768, 14)
    assert emb.min() >= -1.0
    assert emb.max() <= 1.0
""",
)


# ---------------------------------------------------------------------------
# Code Generation
# ---------------------------------------------------------------------------

def generate_test_file() -> str:
    """Generate the complete pytest module as a string."""
    lines = []
    lines.append('"""')
    lines.append("Auto-generated pytest test suite for V-JEPA 2 ViT infrastructure.")
    lines.append(f"Total test functions: {len(TEST_CASES)}")
    lines.append("")
    lines.append("Generated by: scripts/gen_vit_tests.py")
    lines.append("Run with: pytest tests/test_generated_vit.py -v")
    lines.append('"""')
    lines.append("")
    lines.append("import sys")
    lines.append("import os")
    lines.append("import math")
    lines.append("import numpy as np")
    lines.append("import pytest")
    lines.append("import torch")
    lines.append("import torch.nn as nn")
    lines.append("")
    lines.append("# Add assets directory to path")
    lines.append("_SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))")
    lines.append("sys.path.insert(0, os.path.join(_SKILL_DIR, 'assets'))")
    lines.append("")
    lines.append("# Import all components under test")
    lines.append("from vision_transformer_template import (")
    lines.append("    VisionTransformer, PatchEmbed, PatchEmbed3D,")
    lines.append("    get_2d_sincos_pos_embed, get_3d_sincos_pos_embed,")
    lines.append(")")
    lines.append("from transformer_block_template import (")
    lines.append("    DropPath, MLP, SwiGLU, Attention, RoPEAttention, Block,")
    lines.append(")")
    lines.append("from positional_encoding_template import (")
    lines.append("    RoPE3D, apply_rope_1d,")
    lines.append(")")
    lines.append("from cross_attention_template import (")
    lines.append("    CrossAttention, AttentivePooler,")
    lines.append(")")
    lines.append("")
    lines.append("")

    # Group by category with section comments
    categories = {}
    for tc_entry in TEST_CASES:
        cat = tc_entry["category"]
        categories.setdefault(cat, []).append(tc_entry)

    for cat, tcs in categories.items():
        lines.append(f"# {'=' * 60}")
        lines.append(f"# {cat.replace('_', ' ').title()} Tests ({len(tcs)} tests)")
        lines.append(f"# {'=' * 60}")
        lines.append("")
        for tc_entry in tcs:
            code = textwrap.dedent(tc_entry["code"])
            lines.append(code)

    return "\n".join(lines)


def main() -> int:
    args = sys.argv[1:]

    if "--count" in args:
        print(f"Total test cases: {len(TEST_CASES)}")
        by_cat = {}
        for tc_entry in TEST_CASES:
            by_cat.setdefault(tc_entry["category"], 0)
            by_cat[tc_entry["category"]] += 1
        for cat, count in sorted(by_cat.items()):
            print(f"  {cat}: {count}")
        return 0

    output = generate_test_file()
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
