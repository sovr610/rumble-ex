#!/usr/bin/env python3
"""
masking_benchmark.py
====================
Benchmarks for V-JEPA 2 Video & Masking infrastructure:

    1. Mask generation throughput (MaskGenerator.__call__)
    2. MaskCollator overhead vs raw default_collate
    3. apply_masks speedup (naive vs vectorised)
    4. PatchEmbed3D throughput
    5. End-to-end tokenize + mask pipeline

Usage:
    cd /path/to/vjepa2-video-masking/assets
    python ../scripts/masking_benchmark.py
    python ../scripts/masking_benchmark.py --batch-size 64 --repeats 100
    python ../scripts/masking_benchmark.py --device cuda   # GPU timing
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor

# Allow imports from assets/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(_SCRIPT_DIR, "..", "assets")
sys.path.insert(0, os.path.abspath(_ASSETS_DIR))

from patch_embed_template      import PatchEmbed3D
from mask_generator_template   import MaskGenerator, make_small_mask_generator, make_large_mask_generator
from mask_collator_template    import MaskCollator
from multi_sequence_template   import apply_masks, apply_masks_fast


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def timeit(
    fn:          Callable,
    repeats:     int = 50,
    warmup:      int = 5,
    device:      str = "cpu",
    label:       str = "",
) -> Tuple[float, float, float]:
    """
    Measure wall-clock time of fn().

    Args:
        fn:      Zero-argument callable.
        repeats: Number of timed iterations.
        warmup:  Number of un-timed warm-up iterations.
        device:  If "cuda", synchronise before timing.
        label:   Printed description.

    Returns:
        (mean_ms, min_ms, max_ms) across timed iterations.
    """
    use_cuda = (device == "cuda" and torch.cuda.is_available())

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    # Warm-up
    for _ in range(warmup):
        fn()
    sync()

    times = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)  # ms

    mean_ms = sum(times) / len(times)
    min_ms  = min(times)
    max_ms  = max(times)

    if label:
        print(
            f"  {label:<55}  "
            f"mean={mean_ms:7.2f} ms  "
            f"min={min_ms:7.2f} ms  "
            f"max={max_ms:7.2f} ms"
        )

    return mean_ms, min_ms, max_ms


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f" {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Benchmark 1: MaskGenerator throughput
# ---------------------------------------------------------------------------

def bench_mask_generation(batch_size: int, repeats: int, device: str) -> None:
    section("Benchmark 1: MaskGenerator.__call__ throughput")

    configs = [
        ("small  grid=(8,14,14) npred=8",
         dict(spatial_scale=(0.15, 0.15), temporal_scale=(1.0, 1.0),
              aspect_ratio=(0.75, 1.5), npred=8, grid_size=(8, 14, 14))),
        ("large  grid=(8,14,14) npred=2",
         dict(spatial_scale=(0.7, 0.7), temporal_scale=(1.0, 1.0),
              aspect_ratio=(0.75, 1.5), npred=2, grid_size=(8, 14, 14))),
        ("small  grid=(4,14,14) npred=4",
         dict(spatial_scale=(0.15, 0.15), temporal_scale=(1.0, 1.0),
              aspect_ratio=(0.75, 1.5), npred=4, grid_size=(4, 14, 14))),
        ("large  grid=(8,16,16) npred=2",
         dict(spatial_scale=(0.7, 0.7), temporal_scale=(1.0, 1.0),
              aspect_ratio=(0.75, 1.5), npred=2, grid_size=(8, 16, 16))),
        ("max_keep=500 grid=(8,14,14)",
         dict(spatial_scale=(0.15, 0.15), temporal_scale=(1.0, 1.0),
              aspect_ratio=(0.75, 1.5), npred=8, grid_size=(8, 14, 14),
              max_keep=500)),
    ]

    TARGET_MS = 10.0  # done-when performance target

    for label, kwargs in configs:
        mg = MaskGenerator(**kwargs)
        counter = [0]

        def fn():
            mg(batch_size, seed=counter[0])
            counter[0] += 1

        mean_ms, min_ms, _ = timeit(
            fn, repeats=repeats, warmup=5, device=device,
            label=f"B={batch_size:3d}  {label}",
        )
        tokens_per_sec = (batch_size * mg.N) / (mean_ms / 1e3)
        status = "OK" if mean_ms < TARGET_MS else "SLOW"
        print(f"         tokens/sec: {tokens_per_sec:,.0f}  [{status} target<{TARGET_MS}ms]")


# ---------------------------------------------------------------------------
# Benchmark 2: MaskCollator overhead
# ---------------------------------------------------------------------------

def bench_collator_overhead(batch_size: int, repeats: int, device: str) -> None:
    section("Benchmark 2: MaskCollator overhead vs default_collate")
    from torch.utils.data.dataloader import default_collate

    def make_item(fpc: int = 16) -> dict:
        return {
            "fpc":    fpc,
            "frames": torch.randn(fpc, 3, 14, 14),
            "label":  torch.tensor(0),
        }

    mg16 = MaskGenerator(npred=4, grid_size=(8, 14, 14))
    mg8  = MaskGenerator(npred=4, grid_size=(4, 14, 14))

    # --- Single FPC batch ---
    batch_single = [make_item(16) for _ in range(batch_size)]

    def raw_collate():
        default_collate(batch_single)

    def collator_single():
        col = MaskCollator(mg16)
        col(batch_single)

    mean_raw,  _, _ = timeit(raw_collate,     repeats=repeats, warmup=3,
                             device=device, label=f"B={batch_size:3d}  raw default_collate (baseline)")
    mean_coll, _, _ = timeit(collator_single, repeats=repeats, warmup=3,
                             device=device, label=f"B={batch_size:3d}  MaskCollator single-FPC")

    overhead = mean_coll - mean_raw
    TARGET = 20.0  # done-when target
    status = "OK" if overhead < TARGET else "SLOW"
    print(f"         overhead: {overhead:.2f} ms  [{status} target<{TARGET}ms]")

    # --- Two FPC batch ---
    half = batch_size // 2
    batch_multi = [make_item(16) for _ in range(half)] + \
                  [make_item(8)  for _ in range(batch_size - half)]
    col_multi = MaskCollator({16: mg16, 8: mg8})

    def collator_multi():
        col_multi(batch_multi)

    mean_multi, _, _ = timeit(collator_multi, repeats=repeats, warmup=3,
                              device=device,
                              label=f"B={batch_size:3d}  MaskCollator two-FPC")
    overhead2 = mean_multi - mean_raw
    TARGET2   = 30.0
    status2   = "OK" if overhead2 < TARGET2 else "SLOW"
    print(f"         overhead: {overhead2:.2f} ms  [{status2} target<{TARGET2}ms]")


# ---------------------------------------------------------------------------
# Benchmark 3: apply_masks speedup
# ---------------------------------------------------------------------------

def bench_apply_masks(batch_size: int, repeats: int, device: str) -> None:
    section("Benchmark 3: apply_masks naive vs fast (vectorised)")

    configs = [
        ("N=1568  75% masked", 1568, 0.25),
        ("N=1568  50% masked", 1568, 0.50),
        ("N=2048  75% masked", 2048, 0.25),
        ("N=784   75% masked",  784, 0.25),
    ]

    TARGET_MS = 5.0  # done-when target for apply_masks

    for label, N, vis_frac in configs:
        n_vis = max(1, int(N * vis_frac))
        x     = torch.randn(batch_size, N, 1024).to(device)
        ms    = [torch.zeros(N, dtype=torch.bool) for _ in range(batch_size)]
        for m in ms:
            m[:n_vis] = True

        def fn_naive():
            apply_masks(x, ms)

        def fn_fast():
            apply_masks_fast(x, ms)

        mean_naive, _, _ = timeit(fn_naive, repeats=repeats, warmup=5,
                                  device=device,
                                  label=f"B={batch_size:3d}  naive  {label}")
        mean_fast, _, _  = timeit(fn_fast,  repeats=repeats, warmup=5,
                                  device=device,
                                  label=f"B={batch_size:3d}  fast   {label}")

        speedup = mean_naive / max(mean_fast, 1e-9)
        status  = "OK" if mean_fast < TARGET_MS else "SLOW"
        print(f"         speedup fast/naive: {speedup:.2f}x  "
              f"[fast {status} target<{TARGET_MS}ms]")


# ---------------------------------------------------------------------------
# Benchmark 4: PatchEmbed3D throughput
# ---------------------------------------------------------------------------

def bench_patch_embed(batch_size: int, repeats: int, device: str) -> None:
    section("Benchmark 4: PatchEmbed3D forward pass throughput")

    configs = [
        ("224px 16f  tubelet=2  -> 1568 tokens",
         dict(img_size=224, frames=16, patch_size=16, tubelet_size=2, embed_dim=1024)),
        ("256px 16f  tubelet=2  -> 2048 tokens",
         dict(img_size=256, frames=16, patch_size=16, tubelet_size=2, embed_dim=1024)),
        ("224px  8f  tubelet=2  ->  784 tokens",
         dict(img_size=224, frames=8,  patch_size=16, tubelet_size=2, embed_dim=768)),
    ]

    for label, kwargs in configs:
        pe = PatchEmbed3D(**kwargs).to(device)
        T  = kwargs["frames"]
        H  = kwargs["img_size"]
        x  = torch.randn(batch_size, 3, T, H, H).to(device)

        with torch.no_grad():
            def fn():
                pe(x)

            mean_ms, _, _ = timeit(fn, repeats=repeats, warmup=5,
                                   device=device,
                                   label=f"B={batch_size:3d}  {label}")

        clips_per_sec = (batch_size / (mean_ms / 1e3))
        print(f"         clips/sec: {clips_per_sec:,.1f}")


# ---------------------------------------------------------------------------
# Benchmark 5: End-to-end tokenize + mask
# ---------------------------------------------------------------------------

def bench_end_to_end(batch_size: int, repeats: int, device: str) -> None:
    section("Benchmark 5: End-to-end tokenize + mask + apply_masks")

    pe = PatchEmbed3D(img_size=224, frames=16, patch_size=16,
                      tubelet_size=2, embed_dim=1024).to(device)
    mg = make_small_mask_generator(grid_size=(8, 14, 14), npred=8)

    x_raw = torch.randn(batch_size, 3, 16, 224, 224).to(device)
    N     = pe.num_patches
    D     = 1024

    # Pre-compute a fixed visible ratio mask so all samples have identical N_vis.
    # In training, the MaskCollator ensures this invariant within each FPC group.
    # We use a simple fixed mask here to isolate the tokenize+gather timing.
    N     = pe.num_patches
    n_vis = N // 4   # 75% masked: only 25% visible
    fixed_mask = torch.zeros(N, dtype=torch.bool)
    fixed_mask[:n_vis] = True
    fixed_masks = [fixed_mask for _ in range(batch_size)]

    counter = [0]

    def fn():
        # 1. Tokenise
        with torch.no_grad():
            tokens = pe(x_raw)  # [B, N, D]

        # 2. Apply masks (gather visible tokens)
        # Using apply_masks (safe: tolerates varying N_vis per sample)
        apply_masks(tokens, fixed_masks)

    mean_ms, min_ms, max_ms = timeit(
        fn, repeats=repeats, warmup=5, device=device,
        label=f"B={batch_size:3d}  tokenize+mask+gather  224px 16f",
    )
    clips_per_sec = batch_size / (mean_ms / 1e3)
    print(f"         clips/sec (end-to-end): {clips_per_sec:,.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark V-JEPA 2 masking infrastructure."
    )
    p.add_argument("--batch-size",  type=int, default=32,
                   help="Batch size for benchmarks (default 32).")
    p.add_argument("--repeats",     type=int, default=50,
                   help="Timed iterations per benchmark (default 50).")
    p.add_argument("--device",      type=str, default="cpu",
                   choices=["cpu", "cuda"],
                   help="Device to run benchmarks on (default cpu).")
    p.add_argument("--skip",        nargs="*", default=[],
                   choices=["gen", "collator", "apply", "embed", "e2e"],
                   help="Skip specific benchmarks.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("V-JEPA 2 Video & Masking -- Performance Benchmarks")
    print(f"device={args.device}  batch_size={args.batch_size}  repeats={args.repeats}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU.")
        args.device = "cpu"

    B  = args.batch_size
    R  = args.repeats
    DV = args.device

    if "gen" not in args.skip:
        bench_mask_generation(B, R, DV)

    if "collator" not in args.skip:
        bench_collator_overhead(B, R, DV)

    if "apply" not in args.skip:
        bench_apply_masks(B, R, DV)

    if "embed" not in args.skip:
        bench_patch_embed(B, R, DV)

    if "e2e" not in args.skip:
        bench_end_to_end(B, R, DV)

    print()
    print("=" * 72)
    print("Benchmarks complete.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
