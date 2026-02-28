#!/usr/bin/env python3
"""
validate_data.py
================
Validates the three Done-When gates for the V-JEPA 2 data pipeline.

Gates
-----
1. Video Loading    -- VideoDataset.__getitem__() returns correctly shaped tensor
                       [C, T, H, W] from synthetic video data; all three clip
                       modes produce valid frame counts.

2. Transform Pipeline -- Train transform produces augmented tensors with correct
                         shape and normalization; eval transform is deterministic.

3. Multi-Source Sampling -- DistributedWeightedSampler respects weights; no
                             duplicate samples across ranks; full coverage per epoch.

Usage
-----
    python scripts/validate_data.py
    python scripts/validate_data.py --verbose
    python scripts/validate_data.py --gate 1      # Run only gate 1
    python scripts/validate_data.py --gate 2,3    # Run gates 2 and 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup: allow running from the skill root directory
# ---------------------------------------------------------------------------
SKILL_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = SKILL_ROOT / "assets"
sys.path.insert(0, str(ASSETS_DIR))

try:
    from video_dataset_template import VideoDataset, compute_frame_indices
    from video_transforms_template import VideoTransformPipeline
    from distributed_sampler_template import DistributedWeightedSampler, ConcatIndices
    from data_config_template import DataConfig, AugConfig
    _IMPORTS_OK = True
except ImportError as e:
    print(f"[ERROR] Failed to import asset modules: {e}")
    print(f"        Ensure assets/ directory is at {ASSETS_DIR}")
    _IMPORTS_OK = False


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------

class GateReport:
    """Accumulates results for a single validation gate."""

    def __init__(self, gate_num: int, gate_name: str):
        self.gate_num  = gate_num
        self.gate_name = gate_name
        self.checks: List[tuple] = []  # (name, passed, detail)
        self.start_time = time.time()

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append((name, passed, detail))
        status = "[PASS]" if passed else "[FAIL]"
        detail_str = f" -- {detail}" if detail else ""
        print(f"  {status} {name}{detail_str}")
        return passed

    def summary(self) -> bool:
        elapsed = time.time() - self.start_time
        passed  = sum(1 for _, p, _ in self.checks if p)
        total   = len(self.checks)
        gate_ok = passed == total
        icon = "OK" if gate_ok else "FAIL"
        print(f"\n  [{icon}] Gate {self.gate_num}: {self.gate_name} "
              f"({passed}/{total} checks passed, {elapsed:.2f}s)")
        return gate_ok


# ---------------------------------------------------------------------------
# Gate 1: Video Loading
# ---------------------------------------------------------------------------

def run_gate1(verbose: bool = False) -> bool:
    """
    Gate 1: Video Loading

    Validates:
    - __getitem__ returns dict with "video" key
    - Output tensor is [C, T, H, W] float32 with values in [0,1]
    - All three clip modes return correct T dimension
    - Circulant padding handles short videos
    - Short video output is non-zero (frames were actually loaded)
    """
    report = GateReport(1, "Video Loading")
    print("\nGate 1: Video Loading")
    print("-" * 50)

    if not _IMPORTS_OK:
        report.check("Import check", False, "asset modules not importable")
        return report.summary()

    IMG_SIZE = 64
    FPC = 8
    SYNTHETIC_FRAMES = 30

    def make_ds(mode: str, n_frames: int = SYNTHETIC_FRAMES) -> VideoDataset:
        return VideoDataset(
            data_paths=[],
            clip_mode=mode,
            frames_per_clip=FPC,
            img_size=IMG_SIZE,
            synthetic_num_frames=n_frames,
        )

    # Check 1.1: Basic shape [C, T, H, W]
    ds = make_ds("fps")
    sample = ds[0]
    expected = (3, FPC, IMG_SIZE, IMG_SIZE)
    report.check(
        "fps mode output shape",
        sample["video"].shape == expected,
        f"got {tuple(sample['video'].shape)}, expected {expected}"
    )

    # Check 1.2: dtype float32
    report.check(
        "Output dtype is float32",
        sample["video"].dtype == torch.float32,
        str(sample["video"].dtype)
    )

    # Check 1.3: Values in [0, 1]
    vmin = sample["video"].min().item()
    vmax = sample["video"].max().item()
    report.check(
        "Values in [0, 1]",
        0.0 <= vmin and vmax <= 1.0,
        f"range=[{vmin:.4f}, {vmax:.4f}]"
    )

    # Check 1.4: Return dict has required keys
    required_keys = {"video", "label", "path"}
    missing = required_keys - set(sample.keys())
    report.check(
        "Return dict has required keys (video, label, path)",
        len(missing) == 0,
        f"missing={missing}" if missing else "all present"
    )

    # Check 1.5-1.7: All three clip modes return FPC frames
    for mode in ["fps", "duration", "frame_step"]:
        ds_m = make_ds(mode)
        s = ds_m[0]
        T_dim = s["video"].shape[1]
        report.check(
            f"clip_mode={mode!r} -> T={FPC}",
            T_dim == FPC,
            f"got T={T_dim}"
        )

    # Check 1.8: Short video circulant padding (3 frames, need 16)
    ds_short = VideoDataset(
        data_paths=[],
        clip_mode="fps",
        frames_per_clip=16,
        img_size=IMG_SIZE,
        synthetic_num_frames=3,
    )
    s_short = ds_short[0]
    expected_short = (3, 16, IMG_SIZE, IMG_SIZE)
    report.check(
        "Short video (3 frames) circulant padding -> shape [3,16,H,W]",
        s_short["video"].shape == expected_short,
        f"got {tuple(s_short['video'].shape)}"
    )

    # Check 1.9: Short video output is non-zero
    report.check(
        "Short video output is non-zero (frames loaded)",
        s_short["video"].abs().sum().item() > 0,
    )

    # Check 1.10: Single-frame video
    ds_1f = VideoDataset(
        data_paths=[],
        clip_mode="fps",
        frames_per_clip=8,
        img_size=IMG_SIZE,
        synthetic_num_frames=1,
    )
    s_1f = ds_1f[0]
    report.check(
        "Single-frame video circulant padding -> shape [3,8,H,W]",
        s_1f["video"].shape == (3, 8, IMG_SIZE, IMG_SIZE),
        f"got {tuple(s_1f['video'].shape)}"
    )

    # Check 1.11: compute_frame_indices fps mode
    indices = compute_frame_indices(
        total_frames=300, frames_per_clip=16,
        clip_mode="fps", native_fps=30.0, target_fps=10.0
    )
    report.check(
        "compute_frame_indices fps -> 16 valid indices in [0,299]",
        len(indices) == 16 and all(0 <= i < 300 for i in indices),
        f"got {len(indices)} indices"
    )

    # Check 1.12: compute_frame_indices duration mode
    indices_dur = compute_frame_indices(
        total_frames=900, frames_per_clip=16,
        clip_mode="duration", native_fps=30.0, clip_duration_sec=3.0
    )
    report.check(
        "compute_frame_indices duration -> 16 valid indices",
        len(indices_dur) == 16 and all(0 <= i < 900 for i in indices_dur),
    )

    # Check 1.13: compute_frame_indices frame_step mode
    indices_fs = compute_frame_indices(
        total_frames=200, frames_per_clip=16,
        clip_mode="frame_step", frame_step=4
    )
    report.check(
        "compute_frame_indices frame_step -> 16 valid indices",
        len(indices_fs) == 16 and all(0 <= i < 200 for i in indices_fs),
    )

    return report.summary()


# ---------------------------------------------------------------------------
# Gate 2: Transform Pipeline
# ---------------------------------------------------------------------------

def run_gate2(verbose: bool = False) -> bool:
    """
    Gate 2: Transform Pipeline

    Validates:
    - Train transform produces [C, T, H, W] with correct size
    - Train transform dtype is float32
    - Eval transform is deterministic (same input -> same output)
    - Normalized values in reasonable range
    - RandomResizedCrop applies same crop to all frames
    """
    report = GateReport(2, "Transform Pipeline")
    print("\nGate 2: Transform Pipeline")
    print("-" * 50)

    if not _IMPORTS_OK:
        report.check("Import check", False, "asset modules not importable")
        return report.summary()

    T, H, W = 8, 256, 256
    IMG_SIZE = 112

    def make_frames(seed: int = 42):
        rng = np.random.RandomState(seed)
        arr = rng.randint(0, 256, (T, H, W, 3), dtype=np.uint8)
        return [arr[t] for t in range(T)]

    cfg = {
        "crop_scale":      (0.3, 1.0),
        "crop_ratio":      (0.75, 1.33),
        "horizontal_flip": True,
        "auto_augment":    False,
        "motion_shift":    True,
        "random_erasing":  0.0,
        "normalize_mean":  (0.485, 0.456, 0.406),
        "normalize_std":   (0.229, 0.224, 0.225),
    }

    pipeline = VideoTransformPipeline(cfg, img_size=IMG_SIZE)
    train_tfm = pipeline.get_train_transform()
    eval_tfm  = pipeline.get_eval_transform()

    frames = make_frames()

    # Check 2.1: Train output shape
    out = train_tfm(frames)
    expected = (3, T, IMG_SIZE, IMG_SIZE)
    report.check(
        f"Train transform output shape {expected}",
        out.shape == expected,
        f"got {tuple(out.shape)}"
    )

    # Check 2.2: Train dtype float32
    report.check(
        "Train transform output dtype float32",
        out.dtype == torch.float32,
        str(out.dtype)
    )

    # Check 2.3: Eval output shape
    eval_out = eval_tfm(frames)
    report.check(
        f"Eval transform output shape {expected}",
        eval_out.shape == expected,
        f"got {tuple(eval_out.shape)}"
    )

    # Check 2.4: Eval is deterministic
    eval_out2 = eval_tfm(frames)
    report.check(
        "Eval transform is deterministic (same input -> same output)",
        torch.allclose(eval_out, eval_out2),
    )

    # Check 2.5: Normalized values in reasonable range
    v_min = out.min().item()
    v_max = out.max().item()
    report.check(
        "Normalized values in [-10, 10]",
        v_min > -10.0 and v_max < 10.0,
        f"range=[{v_min:.3f}, {v_max:.3f}]"
    )

    # Check 2.6: Channel dimension = 3
    report.check(
        "Output has C=3 channels",
        out.shape[0] == 3,
        f"C={out.shape[0]}"
    )

    # Check 2.7: RandomResizedCrop temporal consistency
    from video_transforms_template import VideoRandomResizedCrop, ClipToTensor
    rrc = VideoRandomResizedCrop(IMG_SIZE)
    identical_frames = [frames[0].copy() for _ in range(T)]
    rrc_out = ClipToTensor()(rrc(identical_frames))
    t_ok = all(
        torch.allclose(rrc_out[:, 0], rrc_out[:, t])
        for t in range(1, T)
    )
    report.check(
        "RandomResizedCrop applies same crop to all frames",
        t_ok,
    )

    # Check 2.8: Horizontal flip temporal consistency
    from video_transforms_template import VideoRandomHorizontalFlip
    flip = VideoRandomHorizontalFlip(p=1.0)
    flip_out = ClipToTensor()(flip(identical_frames))
    flip_ok = all(
        torch.allclose(flip_out[:, 0], flip_out[:, t])
        for t in range(1, T)
    )
    report.check(
        "HorizontalFlip applies same decision to all frames",
        flip_ok,
    )

    # Check 2.9: Random erasing cube mode
    from video_transforms_template import VideoRandomErasing
    dummy = torch.ones(3, T, IMG_SIZE, IMG_SIZE)
    eraser = VideoRandomErasing(p=1.0, scale=(0.1, 0.3), cube_mode=True)
    erased = eraser(dummy)
    zero_mask = (erased[:, 0] == 0)
    if zero_mask.any():
        cube_ok = all(
            (erased[:, t][zero_mask] == 0).all().item()
            for t in range(1, T)
        )
    else:
        cube_ok = False
    report.check(
        "RandomErasing cube_mode erases same region in all frames",
        cube_ok,
    )

    # Check 2.10: Robotics pipeline (no flip, fixed scale)
    robot_cfg = dict(cfg, horizontal_flip=False, crop_scale=(0.9, 1.0),
                     motion_shift=False, auto_augment=False)
    rob_pipeline = VideoTransformPipeline(robot_cfg, img_size=IMG_SIZE)
    rob_out = rob_pipeline.get_train_transform()(frames)
    report.check(
        "Robotics pipeline output shape",
        rob_out.shape == expected,
        f"got {tuple(rob_out.shape)}"
    )

    return report.summary()


# ---------------------------------------------------------------------------
# Gate 3: Multi-Source Sampling
# ---------------------------------------------------------------------------

def run_gate3(verbose: bool = False) -> bool:
    """
    Gate 3: Multi-Source Sampling

    Validates:
    - ConcatIndices maps indices correctly
    - DistributedWeightedSampler: no duplicates in single rank
    - No cross-rank duplicates (world_size=2)
    - Full coverage across ranks
    - Weight proportions respected
    - Reproducibility (same epoch -> same order)
    - Different epochs -> different ordering
    """
    report = GateReport(3, "Multi-Source Sampling")
    print("\nGate 3: Multi-Source Sampling")
    print("-" * 50)

    if not _IMPORTS_OK:
        report.check("Import check", False, "asset modules not importable")
        return report.summary()

    # Check 3.1: ConcatIndices basic mapping
    ci = ConcatIndices([1000, 500, 250])
    cases = [
        (0,    (0, 0)),
        (999,  (0, 999)),
        (1000, (1, 0)),
        (1499, (1, 499)),
        (1500, (2, 0)),
        (1749, (2, 249)),
    ]
    all_ok = all(ci[idx] == expected for idx, expected in cases)
    report.check("ConcatIndices basic mapping", all_ok)

    # Check 3.2: ConcatIndices total length
    report.check(
        "ConcatIndices total length = 1750",
        len(ci) == 1750,
        f"got {len(ci)}"
    )

    # Check 3.3: ConcatIndices out-of-range raises IndexError
    oor_ok = False
    try:
        ci[-1]
    except IndexError:
        oor_ok = True
    report.check("ConcatIndices raises IndexError for idx=-1", oor_ok)

    # Check 3.4: No duplicates single rank
    N = 1000
    uniform_w = [1.0] * N
    sampler = DistributedWeightedSampler(uniform_w, N, rank=0, world_size=1, seed=0)
    indices = list(sampler)
    report.check(
        f"Single rank: all {N} unique samples, no duplicates",
        len(set(indices)) == N,
        f"unique={len(set(indices))}"
    )

    # Check 3.5: No cross-rank duplicates
    s0 = DistributedWeightedSampler(uniform_w, N, rank=0, world_size=2, seed=0)
    s1 = DistributedWeightedSampler(uniform_w, N, rank=1, world_size=2, seed=0)
    idx0 = set(list(s0))
    idx1 = set(list(s1))
    overlap = idx0 & idx1
    report.check(
        "No cross-rank duplicates (world_size=2)",
        len(overlap) == 0,
        f"overlap={len(overlap)}"
    )

    # Check 3.6: Full coverage
    covered = idx0 | idx1
    report.check(
        f"Full coverage across ranks (>= {N-2} samples)",
        len(covered) >= N - 2,
        f"covered={len(covered)}/{N}"
    )

    # Check 3.7: Weight distribution (2:1 ratio)
    # Source A: 600 samples weight=2.0, Source B: 400 samples weight=1.0
    w_a = [2.0 / 600] * 600
    w_b = [1.0 / 400] * 400
    ws = DistributedWeightedSampler(w_a + w_b, N, rank=0, world_size=1, seed=42)
    ws_indices = list(ws)
    count_a = sum(1 for i in ws_indices if i < 600)
    prop_a = count_a / len(ws_indices)
    expected_prop = 2.0 / 3.0
    tolerance = 0.10  # 10% tolerance
    report.check(
        f"Weight 2:1 ratio: source_a proportion ~{expected_prop:.0%} (+-{tolerance:.0%})",
        abs(prop_a - expected_prop) < tolerance,
        f"actual={prop_a:.2%}, expected={expected_prop:.2%}"
    )

    # Check 3.8: set_epoch changes ordering
    sampler.set_epoch(0); order0 = list(sampler)
    sampler.set_epoch(1); order1 = list(sampler)
    report.check(
        "set_epoch changes sample ordering",
        order0 != order1,
    )

    # Check 3.9: Same epoch is reproducible
    sampler.set_epoch(5); run1 = list(sampler)
    sampler.set_epoch(5); run2 = list(sampler)
    report.check(
        "Same epoch produces identical ordering",
        run1 == run2,
    )

    # Check 3.10: __len__ matches expected per-rank samples
    import math
    for ws_size in [1, 2, 4]:
        s = DistributedWeightedSampler(uniform_w, N, rank=0, world_size=ws_size, seed=0)
        expected_len = math.ceil(N / ws_size)
        report.check(
            f"__len__ for world_size={ws_size}: expected {expected_len}",
            len(s) == expected_len,
            f"got {len(s)}"
        )

    # Check 3.11: ConcatIndices single source
    ci_s = ConcatIndices([500])
    report.check(
        "ConcatIndices single source: [0]->(0,0), [499]->(0,499)",
        ci_s[0] == (0, 0) and ci_s[499] == (0, 499),
    )

    return report.summary()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate V-JEPA 2 data pipeline done-when gates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gate", type=str, default="1,2,3",
        help="Comma-separated gate numbers to run (default: 1,2,3)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print extra diagnostic information"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    gates_to_run = set()
    for g in args.gate.split(","):
        g = g.strip()
        if g:
            gates_to_run.add(int(g))

    gate_fns = {
        1: run_gate1,
        2: run_gate2,
        3: run_gate3,
    }

    print("=" * 60)
    print("V-JEPA 2 Data Pipeline: Done-When Gate Validation")
    print("=" * 60)
    print(f"Running gates: {sorted(gates_to_run)}")

    results = {}
    for gate_num in sorted(gates_to_run):
        if gate_num not in gate_fns:
            print(f"\n[WARN] Unknown gate number: {gate_num}")
            continue
        results[gate_num] = gate_fns[gate_num](verbose=args.verbose)

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for gate_num, passed in sorted(results.items()):
        status = "PASS" if passed else "FAIL"
        names = {1: "Video Loading", 2: "Transform Pipeline", 3: "Multi-Source Sampling"}
        print(f"  Gate {gate_num} ({names.get(gate_num, '?')}): {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All gates PASSED.")
        return 0
    else:
        print("Some gates FAILED. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
