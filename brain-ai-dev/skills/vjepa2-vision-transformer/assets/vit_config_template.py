"""
ViT Configuration and Factory Functions for V-JEPA 2.

Provides:
- ViTConfig: Dataclass capturing all VisionTransformer constructor parameters
- Factory functions: vit_tiny, vit_small, vit_base, vit_large, vit_huge,
                     vit_giant, vit_gigantic
- VARIANT_REGISTRY: Dict mapping name strings to factory functions
- param_count_estimate: Utility to estimate parameter count before instantiation

Self-tests run when executed directly: python vit_config_template.py
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Optional


# ---------------------------------------------------------------------------
# ViTConfig Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ViTConfig:
    """
    Complete configuration for VisionTransformer.

    All fields correspond directly to VisionTransformer.__init__ parameters,
    enabling reproducible model construction from config files or command-line args.

    Dimension hierarchy:
        embed_dim -> head_dim = embed_dim // num_heads
        embed_dim * mlp_ratio -> mlp_hidden_dim (or SwiGLU-adjusted dim)

    Validation:
        embed_dim % num_heads == 0 is required for all variants.
    """

    # Model identity
    model_name: str = "vit_large"

    # Input geometry
    img_size:      int   = 224   # Spatial resolution (H = W assumed)
    patch_size:    int   = 16    # Patch size in pixels (P x P)
    tubelet_size:  int   = 2     # Temporal tubelet for 3D conv (video models)
    in_chans:      int   = 3     # Input channels (3 for RGB)

    # Transformer architecture
    embed_dim:     int   = 1024  # Token embedding dimension D
    depth:         int   = 24    # Number of transformer blocks
    num_heads:     int   = 16    # Number of attention heads
    mlp_ratio:     float = 4.0   # FFN hidden dim = int(embed_dim * mlp_ratio)

    # Attention variants
    use_rope:      bool  = False  # 3-axis Rotary Position Embeddings
    use_sdpa:      bool  = True   # F.scaled_dot_product_attention (PyTorch 2.0+)
    theta:         float = 10000.0  # RoPE base frequency (only if use_rope=True)

    # FFN variants
    use_silu:      bool  = False  # SwiGLU instead of GELU MLP
    wide_silu:     bool  = False  # Apply 2/3 reduction for SwiGLU hidden dim

    # Regularization
    drop_path_rate: float = 0.0  # Max stochastic depth rate (linear schedule per block)

    # Efficiency
    use_activation_checkpointing: bool = False  # Gradient checkpointing per block
    compile_model:                bool = False  # torch.compile() the model

    # Positional embedding
    use_3d_pos_embed: bool  = True   # 3D sincos (video) vs 2D sincos (image)
    uniform_power:    bool  = False  # Equal dim allocation across depth/height/width

    def __post_init__(self) -> None:
        """Validate configuration consistency."""
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by "
                f"num_heads={self.num_heads}. "
                f"Got head_dim={self.embed_dim / self.num_heads:.2f} (not integer)."
            )
        if self.use_rope:
            head_dim = self.embed_dim // self.num_heads
            axis_dim = 2 * (head_dim // 6)
            if axis_dim == 0:
                raise ValueError(
                    f"head_dim={head_dim} is too small for 3-axis RoPE. "
                    f"Need head_dim >= 6. "
                    f"For {self.model_name}: embed_dim={self.embed_dim} // num_heads={self.num_heads}."
                )

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.embed_dim // self.num_heads

    @property
    def mlp_hidden_dim(self) -> int:
        """Hidden dimension of the FFN (before any SwiGLU adjustment)."""
        return int(self.embed_dim * self.mlp_ratio)

    @property
    def num_patches_per_frame(self) -> int:
        """Spatial patches per frame."""
        return (self.img_size // self.patch_size) ** 2

    def to_dict(self) -> dict:
        """Convert config to a plain dict (for serialization)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ViTConfig":
        """Reconstruct config from a plain dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def __repr__(self) -> str:
        return (
            f"ViTConfig({self.model_name}: "
            f"D={self.embed_dim}, L={self.depth}, H={self.num_heads}, "
            f"mlp_ratio={self.mlp_ratio:.3f}, "
            f"rope={self.use_rope}, silu={self.use_silu})"
        )


# ---------------------------------------------------------------------------
# Parameter Count Estimation
# ---------------------------------------------------------------------------

def param_count_estimate(cfg: ViTConfig, video: bool = True) -> int:
    """
    Estimate total parameter count for a ViTConfig without instantiating the model.

    This is an approximation; actual counts may differ slightly due to:
    - Positional embedding buffer size
    - SwiGLU vs MLP hidden dim adjustment
    - Optional classification head

    Args:
        cfg: ViTConfig instance.
        video: If True, use 3D patch embedding (Conv3d); else 2D (Conv2d).

    Returns:
        Estimated parameter count.
    """
    D = cfg.embed_dim
    H = cfg.depth

    # Patch embedding
    if video:
        # Conv3d: (tubelet * patch^2 * in_chans) * D + D (bias)
        patch_params = (cfg.tubelet_size * cfg.patch_size ** 2 * cfg.in_chans) * D + D
    else:
        # Conv2d: (patch^2 * in_chans) * D + D (bias)
        patch_params = (cfg.patch_size ** 2 * cfg.in_chans) * D + D

    # Per-block parameters
    # QKV projection: 3 * D * D + 3*D (bias)
    qkv_params = 3 * D * D + 3 * D
    # Output projection: D * D + D
    proj_params = D * D + D
    # LayerNorm x2: 2 * (D weight + D bias)
    ln_params = 4 * D

    # FFN parameters
    if cfg.use_silu:
        # SwiGLU: gate + up + down; hidden_dim adjusted
        raw_hidden = cfg.mlp_hidden_dim
        adjusted   = int(2 * raw_hidden / 3) if cfg.wide_silu else raw_hidden
        hidden_dim = math.ceil(adjusted / 8) * 8
        # gate: D * hidden + hidden; up: D * hidden + hidden; down: hidden * D + D
        ffn_params = (D * hidden_dim + hidden_dim) * 2 + (hidden_dim * D + D)
    else:
        # Standard MLP: fc1 + fc2
        hidden_dim = cfg.mlp_hidden_dim
        ffn_params = (D * hidden_dim + hidden_dim) + (hidden_dim * D + D)

    block_params = qkv_params + proj_params + ln_params + ffn_params

    # Final LayerNorm after all blocks
    final_ln_params = 2 * D

    total = patch_params + H * block_params + final_ln_params
    return total


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------

def vit_tiny(**kwargs) -> ViTConfig:
    """
    ViT-Tiny: 192-dim, 12 layers, 3 heads. ~6M parameters.

    Use for: unit tests, rapid prototyping, CPU-feasible inference.
    head_dim = 64; RoPE axis_dim = 20 (60/64 dims rotated).
    """
    defaults = dict(
        model_name="vit_tiny",
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
    )
    defaults.update(kwargs)
    return ViTConfig(**defaults)


def vit_small(**kwargs) -> ViTConfig:
    """
    ViT-Small: 384-dim, 12 layers, 6 heads. ~22M parameters.

    Use for: ablation studies, balanced speed/quality experiments.
    head_dim = 64; compatible with RoPE.
    """
    defaults = dict(
        model_name="vit_small",
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
    )
    defaults.update(kwargs)
    return ViTConfig(**defaults)


def vit_base(**kwargs) -> ViTConfig:
    """
    ViT-Base: 768-dim, 12 layers, 12 heads. ~86M parameters.

    Use for: standard ImageNet/Kinetics-400 benchmarks.
    head_dim = 64; fits in ~4GB VRAM at bfloat16 (batch=8).
    """
    defaults = dict(
        model_name="vit_base",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
    )
    defaults.update(kwargs)
    return ViTConfig(**defaults)


def vit_large(**kwargs) -> ViTConfig:
    """
    ViT-Large: 1024-dim, 24 layers, 16 heads. ~307M parameters.

    The primary V-JEPA 2 context encoder variant.
    head_dim = 64; requires ~8-16GB VRAM at bfloat16 (batch=8, 16 frames).
    """
    defaults = dict(
        model_name="vit_large",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
    )
    defaults.update(kwargs)
    return ViTConfig(**defaults)


def vit_huge(**kwargs) -> ViTConfig:
    """
    ViT-Huge: 1280-dim, 32 layers, 16 heads. ~632M parameters.

    V-JEPA 2 target encoder; also used for robotics (AC-RoPE).
    head_dim = 80; RoPE axis_dim = 24 (72/80 dims rotated).
    Requires ~40GB VRAM at full precision; use bfloat16 + activation checkpointing.
    """
    defaults = dict(
        model_name="vit_huge",
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
    )
    defaults.update(kwargs)
    return ViTConfig(**defaults)


def vit_giant(**kwargs) -> ViTConfig:
    """
    ViT-Giant: 1408-dim, 40 layers, 16 heads. ~1.1B parameters.

    mlp_ratio = 48/11 ~ 4.3636 to produce exactly 6144 hidden dim.
    head_dim = 88; RoPE axis_dim = 28 (84/88 dims rotated).
    Requires multi-GPU (DDP or FSDP) + activation checkpointing.
    """
    defaults = dict(
        model_name="vit_giant",
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48/11,   # => hidden = 1408 * (48/11) = 6144
        use_activation_checkpointing=True,
    )
    defaults.update(kwargs)
    return ViTConfig(**defaults)


def vit_gigantic(**kwargs) -> ViTConfig:
    """
    ViT-Gigantic: 1664-dim, 48 layers, 16 heads. ~1.9B parameters.

    mlp_ratio = 64/11 ~ 5.8181 for large hidden dim.
    head_dim = 104; RoPE axis_dim = 32 (96/104 dims rotated).
    Requires 8+ A100 80GB; always use bfloat16 + activation checkpointing + torch.compile.
    """
    defaults = dict(
        model_name="vit_gigantic",
        embed_dim=1664,
        depth=48,
        num_heads=16,
        mlp_ratio=64/11,   # => hidden ~ 9600
        use_activation_checkpointing=True,
    )
    defaults.update(kwargs)
    return ViTConfig(**defaults)


# ---------------------------------------------------------------------------
# Variant Registry
# ---------------------------------------------------------------------------

VARIANT_REGISTRY: Dict[str, Callable[..., ViTConfig]] = {
    "vit_tiny":     vit_tiny,
    "vit_small":    vit_small,
    "vit_base":     vit_base,
    "vit_large":    vit_large,
    "vit_huge":     vit_huge,
    "vit_giant":    vit_giant,
    "vit_gigantic": vit_gigantic,
}


def get_vit_config(model_name: str, **kwargs) -> ViTConfig:
    """
    Look up and instantiate a ViTConfig by name.

    Args:
        model_name: One of the keys in VARIANT_REGISTRY.
        **kwargs: Overrides for any config fields.

    Returns:
        ViTConfig instance.

    Raises:
        ValueError: If model_name is not in the registry.
    """
    if model_name not in VARIANT_REGISTRY:
        available = list(VARIANT_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available}"
        )
    return VARIANT_REGISTRY[model_name](**kwargs)


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ViT Config Template -- Self-Tests")
    print("=" * 60)

    failures = []

    # ------------------------------------------------------------------
    # Test 1: All factory functions produce valid configs
    # ------------------------------------------------------------------
    print("\n[1] All factory functions produce valid configs...")
    try:
        expected = {
            "vit_tiny":     (192,  12,  3),
            "vit_small":    (384,  12,  6),
            "vit_base":     (768,  12, 12),
            "vit_large":   (1024,  24, 16),
            "vit_huge":    (1280,  32, 16),
            "vit_giant":   (1408,  40, 16),
            "vit_gigantic":(1664,  48, 16),
        }
        for name, (embed_dim, depth, num_heads) in expected.items():
            cfg = VARIANT_REGISTRY[name]()
            assert cfg.embed_dim == embed_dim, \
                f"{name}: embed_dim={cfg.embed_dim} != {embed_dim}"
            assert cfg.depth == depth, \
                f"{name}: depth={cfg.depth} != {depth}"
            assert cfg.num_heads == num_heads, \
                f"{name}: num_heads={cfg.num_heads} != {num_heads}"
            assert cfg.model_name == name, \
                f"{name}: model_name='{cfg.model_name}'"
        print("   PASS: all 7 variants have correct dimensions")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("Factory dimensions")

    # ------------------------------------------------------------------
    # Test 2: embed_dim divisible by num_heads for all variants
    # ------------------------------------------------------------------
    print("\n[2] embed_dim divisible by num_heads for all variants...")
    try:
        for name, factory in VARIANT_REGISTRY.items():
            cfg = factory()
            assert cfg.embed_dim % cfg.num_heads == 0, (
                f"{name}: embed_dim={cfg.embed_dim} not divisible by "
                f"num_heads={cfg.num_heads}"
            )
        print("   PASS: all variants have integer head_dim")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("embed_dim / num_heads divisibility")

    # ------------------------------------------------------------------
    # Test 3: kwargs override works
    # ------------------------------------------------------------------
    print("\n[3] kwargs override factory defaults...")
    try:
        cfg = vit_base(img_size=384, use_rope=True, drop_path_rate=0.1)
        assert cfg.img_size == 384
        assert cfg.use_rope is True
        assert cfg.drop_path_rate == 0.1
        assert cfg.embed_dim == 768    # unchanged
        assert cfg.depth == 12         # unchanged
        print(f"   PASS: overrides applied correctly")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("kwargs override")

    # ------------------------------------------------------------------
    # Test 4: get_vit_config lookup by name
    # ------------------------------------------------------------------
    print("\n[4] get_vit_config lookup by string name...")
    try:
        for name in VARIANT_REGISTRY:
            cfg = get_vit_config(name)
            assert cfg.model_name == name
        print("   PASS: all names resolve correctly")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("get_vit_config lookup")

    # ------------------------------------------------------------------
    # Test 5: Invalid name raises ValueError
    # ------------------------------------------------------------------
    print("\n[5] Invalid name raises ValueError...")
    try:
        try:
            get_vit_config("vit_nonexistent")
            print("   FAIL: should have raised ValueError")
            failures.append("Invalid name ValueError")
        except ValueError:
            print("   PASS: ValueError raised correctly")
    except Exception as e:
        print(f"   FAIL: wrong exception: {e}")
        failures.append("Invalid name exception type")

    # ------------------------------------------------------------------
    # Test 6: head_dim property
    # ------------------------------------------------------------------
    print("\n[6] head_dim property...")
    try:
        expected_head_dims = {
            "vit_tiny":  64,   # 192 // 3
            "vit_small": 64,   # 384 // 6
            "vit_base":  64,   # 768 // 12
            "vit_large": 64,   # 1024 // 16
            "vit_huge":  80,   # 1280 // 16
            "vit_giant": 88,   # 1408 // 16
            "vit_gigantic": 104,  # 1664 // 16
        }
        for name, expected_hd in expected_head_dims.items():
            cfg = VARIANT_REGISTRY[name]()
            assert cfg.head_dim == expected_hd, (
                f"{name}: head_dim={cfg.head_dim} != {expected_hd}"
            )
        print("   PASS: all head_dim values correct")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("head_dim property")

    # ------------------------------------------------------------------
    # Test 7: param_count_estimate within expected range
    # ------------------------------------------------------------------
    print("\n[7] param_count_estimate within expected range...")
    try:
        expected_ranges = {
            "vit_tiny":      (4e6,   12e6),
            "vit_small":     (15e6,  30e6),
            "vit_base":      (60e6, 110e6),
            "vit_large":    (250e6, 380e6),
            "vit_huge":     (550e6, 750e6),
            "vit_giant":    (900e6, 1.4e9),
            "vit_gigantic": (1.5e9, 2.5e9),
        }
        for name, (lo, hi) in expected_ranges.items():
            cfg = VARIANT_REGISTRY[name]()
            est = param_count_estimate(cfg)
            assert lo <= est <= hi, (
                f"{name}: estimated {est/1e6:.1f}M params "
                f"not in range [{lo/1e6:.0f}M, {hi/1e6:.0f}M]"
            )
            print(f"   {name}: ~{est/1e6:.1f}M params")
        print("   PASS: all estimates within expected ranges")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("param_count_estimate range")

    # ------------------------------------------------------------------
    # Test 8: RoPE validation catches too-small head_dim
    # ------------------------------------------------------------------
    print("\n[8] RoPE validation rejects head_dim < 6...")
    try:
        try:
            bad_cfg = ViTConfig(model_name="invalid", embed_dim=4, num_heads=1, use_rope=True)
            print("   FAIL: should have raised ValueError for head_dim=4")
            failures.append("RoPE head_dim validation")
        except ValueError:
            print("   PASS: ValueError raised for head_dim < 6")
    except Exception as e:
        print(f"   FAIL: wrong exception: {e}")
        failures.append("RoPE validation exception")

    # ------------------------------------------------------------------
    # Test 9: to_dict / from_dict round-trip
    # ------------------------------------------------------------------
    print("\n[9] to_dict / from_dict round-trip...")
    try:
        original = vit_large(img_size=256, use_rope=True, drop_path_rate=0.2)
        d = original.to_dict()
        restored = ViTConfig.from_dict(d)

        assert restored.embed_dim == original.embed_dim
        assert restored.use_rope  == original.use_rope
        assert restored.drop_path_rate == original.drop_path_rate
        assert restored.img_size  == original.img_size
        print("   PASS: round-trip preserves all fields")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("to_dict/from_dict round-trip")

    # ------------------------------------------------------------------
    # Test 10: Giant/Gigantic have activation checkpointing enabled by default
    # ------------------------------------------------------------------
    print("\n[10] Giant/Gigantic enable activation checkpointing by default...")
    try:
        giant     = vit_giant()
        gigantic  = vit_gigantic()
        assert giant.use_activation_checkpointing, \
            "vit_giant should default to use_activation_checkpointing=True"
        assert gigantic.use_activation_checkpointing, \
            "vit_gigantic should default to use_activation_checkpointing=True"
        print("   PASS")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("Giant activation checkpointing")

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
