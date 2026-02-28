#!/usr/bin/env python3
"""
validate_masking.py
===================
Validates the three done-when gates for V-JEPA 2 Video & Masking:

    Gate 1 -- Tokenization
        PatchEmbed3D converts [B, 3, 16, 224, 224] -> [B, 1568, D].
        Token count matches (T/t) * (H/P) * (W/P).

    Gate 2 -- Mask Coverage
        MaskGenerator produces masks where encoder+prediction masks together
        cover all tokens exactly once; no overlaps, no gaps.

    Gate 3 -- Collator Grouping
        MaskCollator correctly groups mixed-FPC batches.
        Each group has a consistent sequence length.

Usage:
    cd /path/to/vjepa2-video-masking/assets
    python ../scripts/validate_masking.py

Exit code 0 = all gates passed.  Non-zero = at least one gate failed.
"""

from __future__ import annotations

import sys
import os
import textwrap
from typing import List

# Allow imports from assets/ directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(_SCRIPT_DIR, "..", "assets")
sys.path.insert(0, os.path.abspath(_ASSETS_DIR))

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class GateResult:
    def __init__(self, gate_name: str) -> None:
        self.gate_name = gate_name
        self.checks: List[tuple] = []  # (name, passed, detail)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append((name, passed, detail))

    @property
    def passed(self) -> bool:
        return all(p for _, p, _ in self.checks)

    def report(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines  = [f"\n{'='*60}", f"Gate: {self.gate_name}  [{status}]", "="*60]
        for name, passed, detail in self.checks:
            marker = "  OK  " if passed else " FAIL "
            line = f"[{marker}] {name}"
            if detail:
                line += f"  -- {detail}"
            lines.append(line)
        return "\n".join(lines)


def run_gate(result: GateResult, name: str, condition: bool, detail: str = "") -> None:
    result.add(name, condition, detail)


# ---------------------------------------------------------------------------
# Gate 1: Tokenization
# ---------------------------------------------------------------------------

def validate_tokenization() -> GateResult:
    """
    Verify PatchEmbed3D output shapes and token count formulas.
    """
    from patch_embed_template import PatchEmbed3D, PatchEmbed

    r = GateResult("Gate 1 -- Tokenization")

    # --- 3-D primary case: the done-when spec ---
    pe3d = PatchEmbed3D(img_size=224, frames=16, patch_size=16,
                        tubelet_size=2, embed_dim=1024)
    x    = torch.randn(2, 3, 16, 224, 224)
    out  = pe3d(x)

    expected_shape = (2, 1568, 1024)
    run_gate(r, "PatchEmbed3D output shape [2,1568,1024]",
             out.shape == expected_shape,
             f"got {tuple(out.shape)}")

    run_gate(r, "Token count formula: (16/2)*(224/16)*(224/16) == 1568",
             pe3d.num_patches == 1568,
             f"got {pe3d.num_patches}")

    run_gate(r, "Grid size == (8,14,14)",
             pe3d.grid_size == (8, 14, 14),
             f"got {pe3d.grid_size}")

    # --- Additional shapes ---
    cases = [
        dict(img_size=256, frames=16, patch_size=16, tubelet_size=2,
             embed_dim=1024, B=1, expected=(1, 2048, 1024), label="256px 2048 tokens"),
        dict(img_size=224, frames=8, patch_size=16, tubelet_size=2,
             embed_dim=768, B=4, expected=(4, 784, 768),   label="8f 784 tokens"),
        dict(img_size=224, frames=16, patch_size=16, tubelet_size=4,
             embed_dim=1024, B=1, expected=(1, 784, 1024), label="tubelet=4 784 tokens"),
    ]
    for c in cases:
        label = c.pop("label")
        B     = c.pop("B")
        exp   = c.pop("expected")
        pe    = PatchEmbed3D(**c)
        o     = pe(torch.randn(B, 3, c["frames"], c["img_size"], c["img_size"]))
        run_gate(r, f"PatchEmbed3D {label}",
                 tuple(o.shape) == exp,
                 f"got {tuple(o.shape)}, expected {exp}")

    # --- 2-D PatchEmbed sanity ---
    pe2d = PatchEmbed(img_size=224, patch_size=16, embed_dim=768)
    o2d  = pe2d(torch.randn(4, 3, 224, 224))
    run_gate(r, "PatchEmbed 2-D shape [4,196,768]",
             tuple(o2d.shape) == (4, 196, 768),
             f"got {tuple(o2d.shape)}")

    # --- Divisibility errors ---
    try:
        PatchEmbed3D(frames=15, tubelet_size=2)
        run_gate(r, "PatchEmbed3D raises on frames % tubelet != 0", False,
                 "no error raised")
    except ValueError:
        run_gate(r, "PatchEmbed3D raises on frames % tubelet != 0", True)

    try:
        PatchEmbed3D(img_size=225, patch_size=16)
        run_gate(r, "PatchEmbed3D raises on img_size % patch != 0", False,
                 "no error raised")
    except ValueError:
        run_gate(r, "PatchEmbed3D raises on img_size % patch != 0", True)

    return r


# ---------------------------------------------------------------------------
# Gate 2: Mask Coverage
# ---------------------------------------------------------------------------

def validate_mask_coverage() -> GateResult:
    """
    Verify MaskGenerator produces complete, non-overlapping masks.
    """
    from mask_generator_template import MaskGenerator

    r = GateResult("Gate 2 -- Mask Coverage")

    GRID  = (8, 14, 14)
    N     = 8 * 14 * 14  # 1568
    BATCH = 8

    # --- Small generator ---
    mg_small = MaskGenerator(
        spatial_scale=(0.15, 0.15), temporal_scale=(1.0, 1.0),
        aspect_ratio=(0.75, 1.5), npred=8, grid_size=GRID,
    )
    enc_s, pred_s = mg_small(BATCH, seed=0)

    run_gate(r, "Small gen: correct number of enc masks",
             len(enc_s) == BATCH, f"got {len(enc_s)}")
    run_gate(r, "Small gen: enc mask shape [N]",
             enc_s[0].shape == (N,), f"got {enc_s[0].shape}")
    run_gate(r, "Small gen: enc dtype bool",
             enc_s[0].dtype == torch.bool)

    all_covered   = True
    no_overlap    = True
    correct_count = True
    exact_compl   = True

    for i in range(BATCH):
        enc  = enc_s[i]
        pred = pred_s[i]
        if not (enc | pred).all().item():
            all_covered = False
        if (enc & pred).any().item():
            no_overlap = False
        if enc.long().sum().item() + pred.long().sum().item() != N:
            correct_count = False
        if not torch.equal(enc, ~pred):
            exact_compl = False

    run_gate(r, "Small gen: no gaps  (enc|pred == all True)", all_covered)
    run_gate(r, "Small gen: no overlaps (enc&pred == all False)", no_overlap)
    run_gate(r, "Small gen: enc.sum + pred.sum == N", correct_count)
    run_gate(r, "Small gen: enc == ~pred (exact complement)", exact_compl)

    # --- Large generator ---
    mg_large = MaskGenerator(
        spatial_scale=(0.7, 0.7), temporal_scale=(1.0, 1.0),
        aspect_ratio=(0.75, 1.5), npred=2, grid_size=GRID,
    )
    enc_l, pred_l = mg_large(BATCH, seed=7)

    all_covered_l = True
    no_overlap_l  = True
    for i in range(BATCH):
        if not (enc_l[i] | pred_l[i]).all().item():
            all_covered_l = False
        if (enc_l[i] & pred_l[i]).any().item():
            no_overlap_l = False

    run_gate(r, "Large gen: no gaps", all_covered_l)
    run_gate(r, "Large gen: no overlaps", no_overlap_l)

    # Large blocks should dominate
    mean_pred = sum(p.float().mean().item() for p in pred_l) / BATCH
    run_gate(r, "Large gen: prediction area >= 50% of tokens",
             mean_pred >= 0.50,
             f"mean pred ratio = {mean_pred:.3f}")

    # --- max_context_frames_ratio ---
    T, H, W = GRID
    mg_half = MaskGenerator(
        spatial_scale=(0.1, 0.1), temporal_scale=(1.0, 1.0),
        aspect_ratio=(1.0, 1.0), npred=0, grid_size=GRID,
        max_context_frames_ratio=0.5,
    )
    enc_h, _ = mg_half(1, seed=0)
    enc_3d   = enc_h[0].view(T, H, W)
    visible_frames = enc_3d.any(dim=(1, 2))
    last_vis  = visible_frames.nonzero(as_tuple=False)
    if last_vis.numel() > 0:
        lv = last_vis[-1, 0].item()
        run_gate(r, "max_context_frames_ratio=0.5 caps visible frames",
                 lv < T,
                 f"last visible frame = {lv}, T = {T}")
    else:
        run_gate(r, "max_context_frames_ratio=0.5 some frames visible",
                 False, "no visible frames at all")

    # --- max_keep ---
    MAX_KEEP = 300
    mg_mk = MaskGenerator(
        spatial_scale=(0.1, 0.1), temporal_scale=(1.0, 1.0),
        aspect_ratio=(1.0, 1.0), npred=1, grid_size=GRID,
        max_keep=MAX_KEEP,
    )
    enc_mk, _ = mg_mk(BATCH, seed=5)
    all_capped = all(e.long().sum().item() <= MAX_KEEP for e in enc_mk)
    run_gate(r, f"max_keep={MAX_KEEP} respected for all samples",
             all_capped,
             f"enc token counts: {[e.long().sum().item() for e in enc_mk]}")

    # --- Seed determinism ---
    mg_det = MaskGenerator(
        spatial_scale=(0.15, 0.15), temporal_scale=(1.0, 1.0),
        aspect_ratio=(0.75, 1.5), npred=4, grid_size=GRID,
    )
    e_a, _ = mg_det(4, seed=42)
    e_b, _ = mg_det(4, seed=42)
    e_c, _ = mg_det(4, seed=99)

    same = all(torch.equal(a, b) for a, b in zip(e_a, e_b))
    diff = any(not torch.equal(a, c) for a, c in zip(e_a, e_c))
    run_gate(r, "Seed determinism: same seed -> identical masks", same)
    run_gate(r, "Seed determinism: diff seeds -> different masks", diff)

    return r


# ---------------------------------------------------------------------------
# Gate 3: Collator Grouping
# ---------------------------------------------------------------------------

def validate_collator_grouping() -> GateResult:
    """
    Verify MaskCollator groups mixed-FPC batches correctly.
    """
    from mask_generator_template import MaskGenerator
    from mask_collator_template  import MaskCollator

    r = GateResult("Gate 3 -- Collator Grouping")

    def make_item(fpc: int) -> dict:
        """Minimal dataset item with fpc key."""
        return {
            "fpc":    fpc,
            "frames": torch.randn(fpc, 3, 224 // 16, 224 // 16),  # placeholder
            "label":  torch.tensor(0),
        }

    # Build generators for each FPC
    mg16 = MaskGenerator(
        spatial_scale=(0.15, 0.15), temporal_scale=(1.0, 1.0),
        aspect_ratio=(0.75, 1.5), npred=4, grid_size=(8, 14, 14),
    )
    mg8  = MaskGenerator(
        spatial_scale=(0.15, 0.15), temporal_scale=(1.0, 1.0),
        aspect_ratio=(0.75, 1.5), npred=4, grid_size=(4, 14, 14),
    )
    mg4  = MaskGenerator(
        spatial_scale=(0.15, 0.15), temporal_scale=(1.0, 1.0),
        aspect_ratio=(0.75, 1.5), npred=4, grid_size=(2, 14, 14),
    )

    N16 = 8  * 14 * 14  # 1568
    N8  = 4  * 14 * 14  # 784
    N4  = 2  * 14 * 14  # 392

    mg_dict  = {16: mg16, 8: mg8, 4: mg4}
    collator = MaskCollator(mg_dict)

    # --- Single-FPC batch ---
    batch_single = [make_item(16) for _ in range(4)]
    result_single = collator(batch_single)

    run_gate(r, "Single-FPC: returns list", isinstance(result_single, list))
    run_gate(r, "Single-FPC: 1 group",
             len(result_single) == 1,
             f"got {len(result_single)}")

    _, enc_ss, pred_ss = result_single[0]
    run_gate(r, "Single-FPC: 4 masks",
             len(enc_ss) == 4, f"got {len(enc_ss)}")
    run_gate(r, "Single-FPC: mask shape [N16]",
             enc_ss[0].shape == (N16,), f"got {enc_ss[0].shape}")
    run_gate(r, "Single-FPC: mask dtype bool",
             enc_ss[0].dtype == torch.bool)

    # --- Two-FPC batch: [16, 16, 8, 8] ---
    batch_2fpc = [make_item(16), make_item(16), make_item(8), make_item(8)]
    result_2fpc = collator(batch_2fpc)

    run_gate(r, "Two-FPC batch: 2 groups",
             len(result_2fpc) == 2,
             f"got {len(result_2fpc)}")

    groups_by_n = {}
    for col, enc, pred in result_2fpc:
        n = enc[0].shape[0]
        groups_by_n[n] = (col, enc, pred)

    run_gate(r, "Two-FPC: group N=1568 present",
             N16 in groups_by_n,
             f"keys present: {list(groups_by_n.keys())}")
    run_gate(r, "Two-FPC: group N=784 present",
             N8 in groups_by_n,
             f"keys present: {list(groups_by_n.keys())}")

    if N16 in groups_by_n:
        _, enc16, pred16 = groups_by_n[N16]
        run_gate(r, "Two-FPC: FPC=16 group has 2 items",
                 len(enc16) == 2, f"got {len(enc16)}")
        for i in range(len(enc16)):
            run_gate(r, f"Two-FPC FPC=16[{i}] no gaps",
                     (enc16[i] | pred16[i]).all().item())
            run_gate(r, f"Two-FPC FPC=16[{i}] no overlaps",
                     not (enc16[i] & pred16[i]).any().item())

    if N8 in groups_by_n:
        _, enc8, _ = groups_by_n[N8]
        run_gate(r, "Two-FPC: FPC=8 group has 2 items",
                 len(enc8) == 2, f"got {len(enc8)}")

    # --- Three-FPC batch: [16, 8, 4, 16, 8, 4] ---
    batch_3fpc = [make_item(16), make_item(8), make_item(4),
                  make_item(16), make_item(8), make_item(4)]
    result_3fpc = collator(batch_3fpc)

    run_gate(r, "Three-FPC batch: 3 groups",
             len(result_3fpc) == 3,
             f"got {len(result_3fpc)}")

    groups3 = {}
    for col, enc, pred in result_3fpc:
        n = enc[0].shape[0]
        groups3[n] = len(enc)

    run_gate(r, "Three-FPC: each group has 2 items",
             all(v == 2 for v in groups3.values()),
             f"group sizes: {groups3}")

    # --- Mask coverage within collated groups ---
    all_cov_3 = True
    no_over_3 = True
    for col, enc, pred in result_3fpc:
        for e, p in zip(enc, pred):
            if not (e | p).all().item():
                all_cov_3 = False
            if (e & p).any().item():
                no_over_3 = False
    run_gate(r, "Three-FPC: all groups no gaps",   all_cov_3)
    run_gate(r, "Three-FPC: all groups no overlaps", no_over_3)

    # --- Counter increments across calls ---
    coll_det = MaskCollator(mg16)
    with coll_det._itr_counter.get_lock():
        coll_det._itr_counter.value = -1

    r1 = coll_det([make_item(16), make_item(16)])
    r2 = coll_det([make_item(16), make_item(16)])

    _, e1, _ = r1[0]
    _, e2, _ = r2[0]
    seeds_differ = any(not torch.equal(a, b) for a, b in zip(e1, e2))
    run_gate(r, "Collator counter increments -> different masks per call",
             seeds_differ)

    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("V-JEPA 2 Video & Masking -- Done-When Gate Validation")
    print("=" * 60)

    gates = [
        validate_tokenization,
        validate_mask_coverage,
        validate_collator_grouping,
    ]

    results: List[GateResult] = []
    for fn in gates:
        try:
            gr = fn()
        except Exception as exc:
            # Create a failed result for unexpected exceptions
            gr = GateResult(fn.__name__)
            gr.add(f"EXCEPTION: {type(exc).__name__}", False, str(exc))
        results.append(gr)

    # Print reports
    for gr in results:
        print(gr.report())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for gr in results:
        status = "PASS" if gr.passed else "FAIL"
        total  = len(gr.checks)
        passed = sum(1 for _, p, _ in gr.checks if p)
        print(f"  [{status}]  {gr.gate_name}  ({passed}/{total} checks)")
        if not gr.passed:
            all_passed = False

    print()
    if all_passed:
        print("All done-when gates PASSED.  V-JEPA 2 masking is ready.")
        return 0
    else:
        print("One or more gates FAILED.  See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
