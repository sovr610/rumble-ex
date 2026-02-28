#!/usr/bin/env python3
"""
validate_vit.py — Validates the 3 Done-When Gates from SKILL.md.

Gates:
    (a) ViT Forward Shape — VisionTransformer.forward() produces [B, N, D] for
        video input [B, C, T, H, W]; masking reduces sequence length correctly.
    (b) RoPE Correctness — RoPE3D produces rotationally-equivariant attention;
        outputs differ when positions change.
    (c) Interpolation — Model trained at 256px produces valid outputs at 384px
        via positional embedding interpolation.

Usage:
    python scripts/validate_vit.py

Output:
    PASS or FAIL for each gate, with details on failure.
    Returns exit code 0 if all gates pass, 1 if any fail.
"""

import sys
import os
import math

# Allow running from any directory by adding assets to path
_SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_SKILL_DIR, "assets"))

import torch
import torch.nn as nn


def print_gate(label: str, status: str, detail: str = "") -> None:
    mark = "PASS" if status == "pass" else "FAIL"
    msg = f"[{mark}] {label}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)


# ===========================================================================
# Gate (a): ViT Forward Shape
# ===========================================================================

def gate_a_vit_forward_shape() -> bool:
    """
    Validates:
    1. VisionTransformer produces [B, N, D] for video input [B, C, T, H, W]
    2. N = (T/tubelet) * (H/patch) * (W/patch)
    3. Masked forward reduces N to n_keep
    4. Multiple masks stack correctly in batch dim
    """
    from vision_transformer_template import VisionTransformer

    results = {}

    # --- Sub-test: standard video forward ---
    try:
        model = VisionTransformer(
            img_size=224, patch_size=16, tubelet_size=2,
            embed_dim=192, depth=4, num_heads=3,
        )
        model.training = False

        B, C, T, H, W = 2, 3, 8, 224, 224
        N_expected = (T // 2) * (H // 16) * (W // 16)  # 4*14*14 = 784
        D_expected = 192

        x = torch.randn(B, C, T, H, W)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (B, N_expected, D_expected), (
            f"Expected {(B, N_expected, D_expected)}, got {out.shape}"
        )
        results["video_forward"] = True
    except Exception as e:
        results["video_forward"] = f"FAIL: {e}"

    # --- Sub-test: masking reduces token count ---
    try:
        model = VisionTransformer(embed_dim=192, depth=2, num_heads=3)
        model.training = False

        N_total = 196  # 14*14 patches at 224px
        n_keep  = 50
        B = 2

        masks = [torch.randperm(N_total)[:n_keep].unsqueeze(0).expand(B, -1)]
        x = torch.randn(B, 3, 224, 224)
        with torch.no_grad():
            out = model(x, masks=masks)

        assert out.shape == (B, n_keep, 192), (
            f"Expected ({B},{n_keep},192), got {out.shape}"
        )
        results["masking_reduces_tokens"] = True
    except Exception as e:
        results["masking_reduces_tokens"] = f"FAIL: {e}"

    # --- Sub-test: multiple masks stack ---
    try:
        model = VisionTransformer(embed_dim=192, depth=2, num_heads=3)
        model.training = False

        N_total = 196
        n_keep  = 49
        B = 2
        num_masks = 3

        masks = [torch.randperm(N_total)[:n_keep].unsqueeze(0).expand(B, -1)
                 for _ in range(num_masks)]
        x = torch.randn(B, 3, 224, 224)
        with torch.no_grad():
            out = model(x, masks=masks)

        assert out.shape == (B * num_masks, n_keep, 192), (
            f"Expected ({B*num_masks},{n_keep},192), got {out.shape}"
        )
        results["multiple_masks_stack"] = True
    except Exception as e:
        results["multiple_masks_stack"] = f"FAIL: {e}"

    # --- Sub-test: 2D image input ---
    try:
        model = VisionTransformer(embed_dim=192, depth=2, num_heads=3)
        model.training = False
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 196, 192)
        results["image_2d_input"] = True
    except Exception as e:
        results["image_2d_input"] = f"FAIL: {e}"

    # Report
    all_pass = all(v is True for v in results.values())
    failures = {k: v for k, v in results.items() if v is not True}

    if all_pass:
        detail = (
            f"Video [B,C,T,H,W] -> [B,N,D]: PASS | "
            f"Masking reduces N: PASS | "
            f"Multi-mask stacking: PASS | "
            f"Image 2D input: PASS"
        )
        print_gate("Gate (a): ViT Forward Shape", "pass", detail)
    else:
        for name, msg in failures.items():
            print_gate(f"Gate (a): ViT Forward Shape [{name}]", "fail", str(msg))

    return all_pass


# ===========================================================================
# Gate (b): RoPE Correctness
# ===========================================================================

def gate_b_rope_correctness() -> bool:
    """
    Validates:
    1. RoPE3D outputs differ when token positions change
    2. RoPE3D is deterministic for same inputs
    3. Relative position property: dot(q_m, k_n) depends on (m-n) not on absolute positions
    4. Frequency bands are correct (theta=10000, decreasing magnitudes)
    """
    from positional_encoding_template import RoPE3D, apply_rope_1d

    results = {}

    # --- Sub-test: position sensitivity ---
    try:
        rope = RoPE3D(head_dim=60)  # 60 is divisible by 6, perfect for 3-axis
        B, H, D = 1, 1, 60
        torch.manual_seed(42)
        q = torch.randn(B, H, 4, D)
        k = torch.randn(B, H, 4, D)

        # Grid (2, 2, 1)
        q1, k1 = rope.apply_rope(q.clone(), k.clone(), grid_depth=2, grid_h=2, grid_w=1)

        # Permute spatial positions
        q_perm = q[:, :, [2, 3, 0, 1], :]
        k_perm = k[:, :, [2, 3, 0, 1], :]
        q2, k2 = rope.apply_rope(q_perm, k_perm, grid_depth=2, grid_h=2, grid_w=1)

        # First token should be different after permutation
        max_diff = (q1[:, :, 0, :] - q2[:, :, 0, :]).abs().max().item()
        assert max_diff > 1e-4, (
            f"RoPE outputs should differ for different positions; max_diff={max_diff:.2e}"
        )
        results["position_sensitivity"] = True
    except Exception as e:
        results["position_sensitivity"] = f"FAIL: {e}"

    # --- Sub-test: determinism ---
    try:
        rope = RoPE3D(head_dim=60)
        torch.manual_seed(0)
        q = torch.randn(2, 4, 12, 60)
        k = torch.randn(2, 4, 12, 60)

        q1, k1 = rope.apply_rope(q, k, 2, 2, 3)
        q2, k2 = rope.apply_rope(q, k, 2, 2, 3)

        assert torch.allclose(q1, q2), "RoPE not deterministic (q)"
        assert torch.allclose(k1, k2), "RoPE not deterministic (k)"
        results["determinism"] = True
    except Exception as e:
        results["determinism"] = f"FAIL: {e}"

    # --- Sub-test: relative position property ---
    try:
        rope = RoPE3D(head_dim=60)
        torch.manual_seed(7)

        q = torch.randn(1, 1, 2, 60)
        k = torch.randn(1, 1, 2, 60)

        # Grid (2, 1, 1): positions (0,0,0) and (1,0,0)
        q1, k1 = rope.apply_rope(q.clone(), k.clone(), grid_depth=2, grid_h=1, grid_w=1)
        logit1 = (q1[:, :, 0, :] * k1[:, :, 1, :]).sum(-1)

        # Shift: grid (5, 1, 1), pick positions 3 and 4 (relative distance still 1)
        q5 = torch.cat([torch.randn(1, 1, 3, 60), q[:, :, :2, :]], dim=2)
        k5 = torch.cat([torch.randn(1, 1, 3, 60), k[:, :, :2, :]], dim=2)
        q5r, k5r = rope.apply_rope(q5, k5, grid_depth=5, grid_h=1, grid_w=1)
        logit2 = (q5r[:, :, 3, :] * k5r[:, :, 4, :]).sum(-1)

        diff = abs(logit1.item() - logit2.item())
        assert diff < 1e-3, (
            f"Relative position property violated: |logit1-logit2|={diff:.2e}"
        )
        results["relative_position_property"] = True
    except Exception as e:
        results["relative_position_property"] = f"FAIL: {e}"

    # --- Sub-test: frequency bands are ordered correctly ---
    try:
        rope = RoPE3D(head_dim=60, theta=10000.0)
        freqs = rope.freqs  # should be [1.0, ..., small]
        assert freqs[0].item() > freqs[-1].item(), \
            "Frequencies should decrease (first=1.0, last=small)"
        assert abs(freqs[0].item() - 1.0) < 1e-5, \
            f"First frequency should be 1.0, got {freqs[0].item()}"
        results["frequency_ordering"] = True
    except Exception as e:
        results["frequency_ordering"] = f"FAIL: {e}"

    # --- Sub-test: apply_rope_1d matches manual rotation ---
    try:
        torch.manual_seed(99)
        axis_dim = 8
        N = 3
        x = torch.randn(N, axis_dim)
        angles = torch.randn(N, axis_dim // 2)

        x_manual = torch.zeros_like(x)
        for n in range(N):
            for ki in range(axis_dim // 2):
                c = math.cos(angles[n, ki].item())
                s = math.sin(angles[n, ki].item())
                x_manual[n, 2*ki]   = x[n, 2*ki]   * c - x[n, 2*ki+1] * s
                x_manual[n, 2*ki+1] = x[n, 2*ki]   * s + x[n, 2*ki+1] * c

        x_rope = apply_rope_1d(x, angles)
        max_diff = (x_manual - x_rope).abs().max().item()
        assert max_diff < 1e-5, f"apply_rope_1d max_diff={max_diff:.2e}"
        results["apply_rope_1d_correctness"] = True
    except Exception as e:
        results["apply_rope_1d_correctness"] = f"FAIL: {e}"

    # Report
    all_pass = all(v is True for v in results.values())
    failures = {k: v for k, v in results.items() if v is not True}

    if all_pass:
        n = len(results)
        detail = f"All {n} sub-tests passed: position sensitivity, determinism, relative property, freq ordering, 1d rotation"
        print_gate("Gate (b): RoPE Correctness", "pass", detail)
    else:
        for name, msg in failures.items():
            print_gate(f"Gate (b): RoPE Correctness [{name}]", "fail", str(msg))

    return all_pass


# ===========================================================================
# Gate (c): Positional Embedding Interpolation
# ===========================================================================

def gate_c_interpolation() -> bool:
    """
    Validates:
    1. Model trained at 256px runs at 384px via interpolation
    2. Model trained at 8 frames / 224px runs at 16 frames / 256px
    3. Positional embedding remains unchanged by forward pass (frozen)
    4. Interpolated output has no NaN or Inf
    """
    from vision_transformer_template import VisionTransformer

    results = {}

    # --- Sub-test: 256px -> 384px spatial interpolation ---
    try:
        model = VisionTransformer(
            img_size=256, patch_size=16,
            embed_dim=192, depth=2, num_heads=3,
        )
        model.training = False

        x_384 = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            out = model(x_384)

        N_expected = (384 // 16) ** 2  # 576
        assert out.shape == (1, N_expected, 192), (
            f"Expected (1,{N_expected},192), got {out.shape}"
        )
        assert not out.isnan().any(), "NaN in interpolated output"
        results["spatial_interp_256_384"] = True
    except Exception as e:
        results["spatial_interp_256_384"] = f"FAIL: {e}"

    # --- Sub-test: 224px training, 384px inference ---
    try:
        model = VisionTransformer(
            img_size=224, patch_size=16,
            embed_dim=192, depth=2, num_heads=3,
        )
        model.training = False

        x = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            out = model(x)

        N_expected = (384 // 16) ** 2  # 576
        assert out.shape == (1, N_expected, 192)
        results["spatial_interp_224_384"] = True
    except Exception as e:
        results["spatial_interp_224_384"] = f"FAIL: {e}"

    # --- Sub-test: temporal + spatial interpolation (video) ---
    try:
        model = VisionTransformer(
            img_size=224, patch_size=16, tubelet_size=2,
            embed_dim=192, depth=2, num_heads=3,
        )
        model.training = False

        # Model was built for 224px; now run 16-frame 256px video
        x = torch.randn(1, 3, 16, 256, 256)
        with torch.no_grad():
            out = model(x)

        # Expected: (16/2) * (256/16)^2 = 8 * 256 = 2048
        N_expected = (16 // 2) * (256 // 16) ** 2
        assert out.shape == (1, N_expected, 192), (
            f"Expected (1,{N_expected},192), got {out.shape}"
        )
        results["temporal_spatial_interp"] = True
    except Exception as e:
        results["temporal_spatial_interp"] = f"FAIL: {e}"

    # --- Sub-test: pos_embed is unchanged by forward pass ---
    try:
        model = VisionTransformer(
            img_size=224, patch_size=16,
            embed_dim=192, depth=2, num_heads=3,
        )
        model.training = False
        pos_before = model.pos_embed.clone()

        x = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            _ = model(x)

        pos_after = model.pos_embed
        assert torch.allclose(pos_before, pos_after), \
            "pos_embed should not change during forward pass"
        results["pos_embed_unchanged"] = True
    except Exception as e:
        results["pos_embed_unchanged"] = f"FAIL: {e}"

    # --- Sub-test: same resolution does not trigger interpolation (same output) ---
    try:
        model = VisionTransformer(
            img_size=224, patch_size=16,
            embed_dim=192, depth=2, num_heads=3,
        )
        model.training = False

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2), "Same-resolution outputs should match"
        assert out1.shape == (2, 196, 192)
        results["same_resolution_deterministic"] = True
    except Exception as e:
        results["same_resolution_deterministic"] = f"FAIL: {e}"

    # Report
    all_pass = all(v is True for v in results.values())
    failures = {k: v for k, v in results.items() if v is not True}

    if all_pass:
        n = len(results)
        detail = (
            f"All {n} sub-tests passed: "
            f"256->384 spatial, 224->384 spatial, temporal+spatial, "
            f"pos_embed frozen, same-res deterministic"
        )
        print_gate("Gate (c): Positional Embedding Interpolation", "pass", detail)
    else:
        for name, msg in failures.items():
            print_gate(
                f"Gate (c): Interpolation [{name}]", "fail", str(msg)
            )

    return all_pass


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    print("=" * 65)
    print("V-JEPA 2 ViT — Done-When Gate Validation")
    print("=" * 65)
    print()

    gate_results = {}

    print("Running Gate (a): ViT Forward Shape...")
    try:
        gate_results["a"] = gate_a_vit_forward_shape()
    except Exception as exc:
        print_gate("Gate (a): ViT Forward Shape", "fail", f"Uncaught exception: {exc}")
        gate_results["a"] = False
    print()

    print("Running Gate (b): RoPE Correctness...")
    try:
        gate_results["b"] = gate_b_rope_correctness()
    except Exception as exc:
        print_gate("Gate (b): RoPE Correctness", "fail", f"Uncaught exception: {exc}")
        gate_results["b"] = False
    print()

    print("Running Gate (c): Positional Embedding Interpolation...")
    try:
        gate_results["c"] = gate_c_interpolation()
    except Exception as exc:
        print_gate("Gate (c): Interpolation", "fail", f"Uncaught exception: {exc}")
        gate_results["c"] = False
    print()

    # Summary
    print("=" * 65)
    passed = sum(1 for v in gate_results.values() if v)
    total  = len(gate_results)
    print(f"Summary: {passed}/{total} gates passed")

    for gate_id, ok in gate_results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  Gate ({gate_id}): {status}")

    print()
    if passed == total:
        print("ALL GATES PASSED — Implementation is ready.")
        return 0
    else:
        failed = [g for g, v in gate_results.items() if not v]
        print(f"GATES FAILED: {failed}")
        print("Fix the failing components before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
