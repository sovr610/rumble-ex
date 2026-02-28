# Testing Matrix — V-JEPA 2 Vision Transformer

Complete test scenarios for the ViT infrastructure, organized by component.
Each scenario includes: what to test, expected behavior, and failure indicators.

---

## 1. Forward Shape Tests (Per Variant)

### 1.1 Image Input — All Variants

For each variant, test that `VisionTransformer.forward(x)` returns `[B, N, D]`
where `N = (H/P)^2` and `D = embed_dim`.

| Variant      | Input shape        | Expected N  | Expected D |
|-------------|---------------------|-------------|-----------|
| ViT-Tiny    | [2, 3, 224, 224]    | 196         | 192       |
| ViT-Small   | [2, 3, 224, 224]    | 196         | 384       |
| ViT-Base    | [2, 3, 224, 224]    | 196         | 768       |
| ViT-Large   | [2, 3, 224, 224]    | 196         | 1024      |
| ViT-Huge    | [2, 3, 224, 224]    | 196         | 1280      |
| ViT-Giant   | [1, 3, 224, 224]    | 196         | 1408      |
| ViT-Gigantic| [1, 3, 224, 224]    | 196         | 1664      |

Patch count: `(224 // 16)^2 = 14^2 = 196`

```python
@pytest.mark.parametrize("variant,embed_dim,depth,num_heads", [
    ("tiny",     192,  12,  3),
    ("small",    384,  12,  6),
    ("base",     768,  12, 12),
    ("large",   1024,  24, 16),
    ("huge",    1280,  32, 16),
    ("giant",   1408,  40, 16),
    ("gigantic",1664,  48, 16),
])
def test_forward_shape_2d(variant, embed_dim, depth, num_heads):
    B, C, H, W = 2, 3, 224, 224
    patch_size = 16
    N_expected = (H // patch_size) * (W // patch_size)  # 196

    model = VisionTransformer(
        img_size=H, patch_size=patch_size,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads,
    )
    model.eval()
    x = torch.randn(B, C, H, W)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, N_expected, embed_dim), \
        f"{variant}: expected {(B, N_expected, embed_dim)}, got {out.shape}"
```

### 1.2 Video Input (3D)

```
Input:  [B, C, T, H, W]
Output: [B, N, D]   where N = (T/t) * (H/P) * (W/P)
```

| Config             | T  | tubelet | H   | P  | N          |
|-------------------|----|---------|-----|----|------------|
| Short clip        | 8  | 2       | 224 | 16 | 4*14*14=784 |
| Standard 16-frame | 16 | 2       | 224 | 16 | 8*14*14=1568 |
| Long 64-frame     | 64 | 2       | 224 | 16 | 32*14*14=6272 |

```python
@pytest.mark.parametrize("T,tubelet,N_expected", [
    (8,  2, 4*14*14),
    (16, 2, 8*14*14),
    (32, 2, 16*14*14),
])
def test_forward_shape_3d_video(T, tubelet, N_expected):
    B, C, H, W = 2, 3, 224, 224
    model = VisionTransformer(
        img_size=H, patch_size=16, tubelet_size=tubelet,
        embed_dim=768, depth=12, num_heads=12,
    )
    x = torch.randn(B, C, T, H, W)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, N_expected, 768)
```

---

## 2. Masking Efficiency Tests

### 2.1 Token Count Reduction

Masking should reduce the number of tokens processed by the encoder.

```python
def test_masking_reduces_token_count():
    B, C, T, H, W = 2, 3, 8, 224, 224
    model = VisionTransformer(embed_dim=768, depth=4, num_heads=12)
    N_total = (T // 2) * (H // 16) * (W // 16)  # 4*14*14 = 784
    n_keep = N_total // 4  # keep 25%

    # Create mask: for each sample, keep n_keep random tokens
    masks = [torch.randperm(N_total)[:n_keep].unsqueeze(0).expand(B, -1)]
    x = torch.randn(B, C, T, H, W)
    out = model(x, masks=masks)
    # Output has B tokens per mask (stacked)
    assert out.shape == (B * len(masks), n_keep, 768), \
        f"Expected {(B * len(masks), n_keep, 768)}, got {out.shape}"
```

### 2.2 Multiple Masks

```python
def test_multiple_masks():
    """Multiple mask tensors produce stacked outputs."""
    B, N_total, D = 2, 196, 768
    n_keep = 49
    model = VisionTransformer(embed_dim=D, depth=4, num_heads=12)
    x = torch.randn(B, 3, 224, 224)

    masks = [
        torch.randperm(N_total)[:n_keep].unsqueeze(0).expand(B, -1),
        torch.randperm(N_total)[:n_keep].unsqueeze(0).expand(B, -1),
    ]
    out = model(x, masks=masks)
    assert out.shape == (B * 2, n_keep, D)
```

### 2.3 No Masking (Identity)

```python
def test_no_masking_passes_all_tokens():
    B, C, H, W = 2, 3, 224, 224
    model = VisionTransformer(embed_dim=768, depth=4, num_heads=12)
    x = torch.randn(B, C, H, W)
    out_no_mask = model(x, masks=None)
    assert out_no_mask.shape == (B, 196, 768)
```

---

## 3. RoPE Equivariance Tests

### 3.1 Output Changes with Position

RoPE must produce different outputs for tokens at different positions.

```python
def test_rope_position_sensitivity():
    """RoPE output must differ when spatial positions differ."""
    head_dim = 64
    from assets.positional_encoding_template import RoPE3D

    rope = RoPE3D(head_dim=head_dim)
    B, H, D = 1, 1, head_dim
    q = torch.randn(B, H, 4, D)
    k = torch.randn(B, H, 4, D)

    # Grid (2, 1, 2) -- 2 frames, 1 row, 2 columns
    q1, k1 = rope.apply_rope(q, k, grid_depth=2, grid_h=1, grid_w=2)

    # Shuffle positions by transposing temporal/spatial layout
    q_shuffled = q[:, :, [2, 3, 0, 1], :]  # swap frame order
    k_shuffled = k[:, :, [2, 3, 0, 1], :]
    q2, k2 = rope.apply_rope(q_shuffled, k_shuffled, grid_depth=2, grid_h=1, grid_w=2)

    # Output should differ from original (positions changed)
    assert not torch.allclose(q1[:, :, 0], q2[:, :, 0], atol=1e-5), \
        "RoPE output should differ when position assignments change"
```

### 3.2 Same Content, Same Position = Same Output

```python
def test_rope_deterministic_same_position():
    head_dim = 64
    from assets.positional_encoding_template import RoPE3D
    rope = RoPE3D(head_dim=head_dim)

    q = torch.randn(1, 1, 6, head_dim)
    k = torch.randn(1, 1, 6, head_dim)

    q1, k1 = rope.apply_rope(q, k, 2, 3, 1)
    q2, k2 = rope.apply_rope(q, k, 2, 3, 1)

    assert torch.allclose(q1, q2), "RoPE must be deterministic"
    assert torch.allclose(k1, k2), "RoPE must be deterministic"
```

### 3.3 Relative Position Property

For two tokens at relative distance `delta`, the attention logit should be the same
regardless of absolute position.

```python
def test_rope_relative_position_property():
    """dot(q_m, k_n) should equal dot(q_{m+c}, k_{n+c}) for any constant c."""
    head_dim = 60  # exactly divisible by 6 for clean test
    from assets.positional_encoding_template import RoPE3D
    rope = RoPE3D(head_dim=head_dim)

    q = torch.randn(1, 1, 2, head_dim)
    k = torch.randn(1, 1, 2, head_dim)

    # Position (0, 0, 0) and (1, 0, 0)
    q1, k1 = rope.apply_rope(q.clone(), k.clone(), grid_depth=2, grid_h=1, grid_w=1)
    logit1 = (q1[:, :, 0, :] * k1[:, :, 1, :]).sum(-1)

    # Build a 5-frame grid and select positions 3 and 4
    q5 = torch.cat([torch.randn(1, 1, 3, head_dim), q[:, :, :2, :]], dim=2)
    k5 = torch.cat([torch.randn(1, 1, 3, head_dim), k[:, :, :2, :]], dim=2)
    q5r, k5r = rope.apply_rope(q5, k5, grid_depth=5, grid_h=1, grid_w=1)
    logit2 = (q5r[:, :, 3, :] * k5r[:, :, 4, :]).sum(-1)

    # Relative distance is still 1; logits should match
    assert torch.allclose(logit1, logit2, atol=1e-4), \
        f"Relative position property violated: {logit1.item():.6f} vs {logit2.item():.6f}"
```

---

## 4. Positional Embedding Interpolation Tests

### 4.1 Basic Interpolation Runs

```python
def test_pos_embed_interpolation_runs():
    """Model trained at 224px should process 384px without error."""
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=2, num_heads=12)
    x_large = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        out = model(x_large)
    N_expected = (384 // 16) ** 2  # 576
    assert out.shape == (1, N_expected, 768)
```

### 4.2 Trilinear Video Interpolation

```python
def test_pos_embed_video_interpolation():
    """Model trained at 8 frames / 224px should evaluate at 16 frames / 256px."""
    model = VisionTransformer(
        img_size=224, patch_size=16, tubelet_size=2,
        embed_dim=768, depth=2, num_heads=12,
    )
    # Original: T=8, H=W=224 -> grid (4, 14, 14) -> N=784
    # New: T=16, H=W=256 -> grid (8, 16, 16) -> N=2048
    x_new = torch.randn(1, 3, 16, 256, 256)
    with torch.no_grad():
        out = model(x_new)
    N_expected = (16 // 2) * (256 // 16) ** 2  # 8*16*16 = 2048
    assert out.shape == (1, N_expected, 768)
```

### 4.3 Same-Resolution Interpolation is Identity

```python
def test_pos_embed_same_resolution_identity():
    """At training resolution, interpolation should not change the pos embed."""
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=2, num_heads=12)
    pos_orig = model.pos_embed.detach().clone()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        out = model(x)

    # pos_embed should be unchanged (non-learnable)
    assert torch.allclose(model.pos_embed, pos_orig), \
        "Positional embedding should not be modified by forward pass"
```

---

## 5. SwiGLU Alignment Tests

### 5.1 Hidden Dimension is Multiple of 8

```python
@pytest.mark.parametrize("embed_dim,mlp_ratio,wide_silu", [
    (192,  4.0,  True),
    (384,  4.0,  True),
    (768,  4.0,  True),
    (1024, 4.0,  True),
    (1408, 48/11, True),
])
def test_swiglu_hidden_dim_divisible_by_8(embed_dim, mlp_ratio, wide_silu):
    from assets.transformer_block_template import SwiGLU
    mlp = SwiGLU(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio))
    assert mlp.hidden_dim % 8 == 0, \
        f"SwiGLU hidden_dim {mlp.hidden_dim} not divisible by 8 for embed_dim={embed_dim}"
```

### 5.2 SwiGLU Output Shape Matches Input

```python
def test_swiglu_output_shape():
    from assets.transformer_block_template import SwiGLU
    B, N, D = 2, 196, 768
    mlp = SwiGLU(in_features=D, hidden_features=D * 4)
    x = torch.randn(B, N, D)
    out = mlp(x)
    assert out.shape == (B, N, D)
```

### 5.3 SwiGLU vs Standard MLP Parameter Count

```python
def test_swiglu_has_more_params_than_mlp():
    """SwiGLU has 3 weight matrices vs MLP's 2, so more params for same hidden_dim."""
    from assets.transformer_block_template import SwiGLU, MLP
    D, mlp_dim = 768, 3072
    mlp = MLP(in_features=D, hidden_features=mlp_dim)
    swiglu = SwiGLU(in_features=D, hidden_features=mlp_dim)

    mlp_params = sum(p.numel() for p in mlp.parameters())
    swiglu_params = sum(p.numel() for p in swiglu.parameters())
    assert swiglu_params > mlp_params, \
        "SwiGLU should have more parameters than MLP for same hidden_dim"
```

---

## 6. SDPA Equivalence Test

SDPA and manual attention must produce the same outputs (within float32 tolerance).

```python
def test_sdpa_matches_manual_attention():
    from assets.transformer_block_template import Attention
    B, N, D, H = 2, 64, 256, 8

    # Manual implementation
    attn_manual = Attention(dim=D, num_heads=H, use_sdpa=False)
    # SDPA implementation (same weights)
    attn_sdpa = Attention(dim=D, num_heads=H, use_sdpa=True)
    attn_sdpa.load_state_dict(attn_manual.state_dict())

    x = torch.randn(B, N, D)
    with torch.no_grad():
        out_manual = attn_manual(x)
        out_sdpa = attn_sdpa(x)

    assert torch.allclose(out_manual, out_sdpa, atol=1e-5), \
        f"SDPA and manual attention differ: max diff = {(out_manual - out_sdpa).abs().max():.2e}"
```

---

## 7. DropPath Tests

### 7.1 DropPath=0 is Identity

```python
def test_drop_path_zero_is_identity():
    from assets.transformer_block_template import DropPath
    dp = DropPath(drop_prob=0.0)
    x = torch.randn(2, 196, 768)
    out = dp(x)
    assert torch.allclose(x, out), "DropPath(0) should be identity"
```

### 7.2 DropPath Eval Mode is Identity

```python
def test_drop_path_eval_is_identity():
    from assets.transformer_block_template import DropPath
    dp = DropPath(drop_prob=0.5).eval()
    x = torch.randn(2, 196, 768)
    out = dp(x)
    assert torch.allclose(x, out), "DropPath in eval mode should be identity"
```

### 7.3 DropPath Drops Samples

```python
def test_drop_path_drops_in_training():
    """With drop_prob=1.0, all residuals should be zeroed out."""
    from assets.transformer_block_template import DropPath
    dp = DropPath(drop_prob=1.0).train()
    x = torch.ones(8, 196, 768)
    out = dp(x)
    assert out.abs().sum() == 0.0, "DropPath(1.0) should zero all outputs"
```

---

## 8. Activation Checkpointing Tests

### 8.1 Checkpointing Reduces Peak Memory

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_activation_checkpointing_reduces_memory():
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    def measure_peak_memory(use_checkpoint):
        torch.cuda.reset_peak_memory_stats()
        model = VisionTransformer(
            embed_dim=768, depth=12, num_heads=12,
            use_activation_checkpointing=use_checkpoint,
        ).cuda()
        x = torch.randn(4, 3, 224, 224, device="cuda")
        out = model(x)
        out.sum().backward()
        peak = torch.cuda.max_memory_allocated() / 1e6  # MB
        del model, x, out
        torch.cuda.empty_cache()
        return peak

    peak_no_ckpt   = measure_peak_memory(False)
    peak_with_ckpt = measure_peak_memory(True)
    assert peak_with_ckpt < peak_no_ckpt, \
        f"Checkpointing should reduce peak memory: {peak_with_ckpt:.1f}MB vs {peak_no_ckpt:.1f}MB"
```

### 8.2 Checkpointing Produces Same Output

```python
def test_activation_checkpointing_same_output():
    model_base = VisionTransformer(embed_dim=192, depth=4, num_heads=3, use_activation_checkpointing=False)
    model_ckpt = VisionTransformer(embed_dim=192, depth=4, num_heads=3, use_activation_checkpointing=True)
    model_ckpt.load_state_dict(model_base.state_dict())

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out_base = model_base(x)
        out_ckpt = model_ckpt(x)
    assert torch.allclose(out_base, out_ckpt, atol=1e-5)
```

---

## 9. out_layers (Intermediate Features) Tests

### 9.1 Single Intermediate Layer

```python
def test_out_layers_single():
    model = VisionTransformer(embed_dim=768, depth=12, num_heads=12)
    x = torch.randn(2, 3, 224, 224)
    out = model(x, out_layers=[5])
    assert isinstance(out, list), "out_layers should return a list"
    assert len(out) == 1
    assert out[0].shape == (2, 196, 768)
```

### 9.2 Multiple Intermediate Layers

```python
def test_out_layers_multiple():
    model = VisionTransformer(embed_dim=768, depth=12, num_heads=12)
    x = torch.randn(2, 3, 224, 224)
    out = model(x, out_layers=[3, 6, 9, 11])
    assert len(out) == 4
    for o in out:
        assert o.shape == (2, 196, 768)
```

### 9.3 out_layers Last Equals Default Output

```python
def test_out_layers_last_equals_default():
    model = VisionTransformer(embed_dim=768, depth=12, num_heads=12)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out_default = model(x)
        out_last = model(x, out_layers=[11])
    assert torch.allclose(out_default, out_last[0])
```

---

## 10. Cross-Attention and Attentive Pooler Tests

### 10.1 Cross-Attention Output Shape

```python
def test_cross_attention_output_shape():
    from assets.cross_attention_template import CrossAttention
    B, Nq, Nkv, D = 2, 8, 196, 768
    ca = CrossAttention(dim=D, num_heads=12)
    queries = torch.randn(B, Nq, D)
    encoder = torch.randn(B, Nkv, D)
    out = ca(queries, encoder)
    assert out.shape == (B, Nq, D)
```

### 10.2 AttentivePooler Single Query

```python
def test_attentive_pooler_single_query():
    from assets.cross_attention_template import AttentivePooler
    B, N, D = 2, 196, 768
    pooler = AttentivePooler(num_queries=1, embed_dim=D, num_heads=12)
    encoder_out = torch.randn(B, N, D)
    out = pooler(encoder_out)
    assert out.shape == (B, 1, D)
```

### 10.3 AttentivePooler with Classification Head

```python
def test_attentive_pooler_with_classifier():
    from assets.cross_attention_template import AttentivePooler
    B, N, D, num_classes = 2, 196, 768, 400
    pooler = AttentivePooler(
        num_queries=1, embed_dim=D, num_heads=12,
        num_classes=num_classes,
    )
    encoder_out = torch.randn(B, N, D)
    logits = pooler(encoder_out)
    assert logits.shape == (B, num_classes)
```

---

## 11. Configuration Factory Tests

```python
@pytest.mark.parametrize("factory_fn,expected_embed_dim,expected_depth", [
    ("vit_tiny",     192,  12),
    ("vit_small",    384,  12),
    ("vit_base",     768,  12),
    ("vit_large",   1024,  24),
    ("vit_huge",    1280,  32),
    ("vit_giant",   1408,  40),
    ("vit_gigantic",1664,  48),
])
def test_factory_produces_correct_config(factory_fn, expected_embed_dim, expected_depth):
    from assets.vit_config_template import VARIANT_REGISTRY
    cfg = VARIANT_REGISTRY[factory_fn]()
    assert cfg.embed_dim == expected_embed_dim
    assert cfg.depth == expected_depth

def test_factory_kwargs_override():
    from assets.vit_config_template import vit_base
    cfg = vit_base(img_size=384, use_rope=True)
    assert cfg.img_size == 384
    assert cfg.use_rope is True
    assert cfg.embed_dim == 768  # unchanged

def test_all_factories_produce_valid_configs():
    from assets.vit_config_template import VARIANT_REGISTRY
    for name, factory in VARIANT_REGISTRY.items():
        cfg = factory()
        assert cfg.embed_dim > 0
        assert cfg.depth > 0
        assert cfg.num_heads > 0
        assert cfg.embed_dim % cfg.num_heads == 0, \
            f"{name}: embed_dim {cfg.embed_dim} not divisible by num_heads {cfg.num_heads}"
```

---

## 12. Sincos Embedding Tests

### 12.1 Shape Correctness

```python
def test_2d_sincos_pos_embed_shape():
    from assets.positional_encoding_template import get_2d_sincos_pos_embed
    embed_dim, grid_size = 768, 14
    emb = get_2d_sincos_pos_embed(embed_dim, grid_size)
    assert emb.shape == (grid_size * grid_size, embed_dim)

def test_3d_sincos_pos_embed_shape():
    from assets.positional_encoding_template import get_3d_sincos_pos_embed
    embed_dim, grid_size, grid_depth = 768, 14, 8
    emb = get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth)
    assert emb.shape == (grid_depth * grid_size * grid_size, embed_dim)
```

### 12.2 Embeddings are Frozen

```python
def test_pos_embed_not_learnable():
    model = VisionTransformer(embed_dim=192, depth=2, num_heads=3)
    assert not model.pos_embed.requires_grad, \
        "Positional embedding must be non-learnable (requires_grad=False)"
```

### 12.3 Uniform Power vs Default Allocation

```python
def test_3d_sincos_uniform_power_changes_embedding():
    from assets.positional_encoding_template import get_3d_sincos_pos_embed
    import numpy as np
    emb_default = get_3d_sincos_pos_embed(768, 14, 8, uniform_power=False)
    emb_uniform = get_3d_sincos_pos_embed(768, 14, 8, uniform_power=True)
    assert not np.allclose(emb_default, emb_uniform), \
        "uniform_power=True should produce different embeddings than default"
```

---

## Test Execution Matrix

| Test Group                    | Fast (no GPU) | GPU Required | Time estimate |
|-------------------------------|--------------|--------------|---------------|
| Forward shape (Tiny-Large)    | Yes          | No           | < 30s         |
| Forward shape (Huge-Gigantic) | Yes*         | No           | 2-5min        |
| Masking tests                 | Yes          | No           | < 10s         |
| RoPE equivariance             | Yes          | No           | < 5s          |
| Pos embed interpolation       | Yes          | No           | < 10s         |
| SwiGLU alignment              | Yes          | No           | < 5s          |
| SDPA equivalence              | Yes          | No           | < 10s         |
| DropPath tests                | Yes          | No           | < 5s          |
| Activation checkpointing      | Yes          | Yes          | < 30s         |
| out_layers tests              | Yes          | No           | < 15s         |
| Cross-attention/pooler        | Yes          | No           | < 10s         |
| Config factory tests          | Yes          | No           | < 2s          |
| Sincos embed tests            | Yes          | No           | < 5s          |

*Tiny model instances only for speed; use `embed_dim` overrides for CI.
