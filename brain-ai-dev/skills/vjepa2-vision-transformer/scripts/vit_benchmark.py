#!/usr/bin/env python3
"""
vit_benchmark.py — Throughput and memory benchmark for V-JEPA 2 ViT variants.

Measures:
  - Throughput: samples/sec (forward pass only)
  - Memory: peak GPU memory in MB (or RSS on CPU)

Output: formatted table to stdout.

Usage:
    python scripts/vit_benchmark.py                         # all variants, CPU
    python scripts/vit_benchmark.py --device cuda           # GPU
    python scripts/vit_benchmark.py --device cuda --variants tiny,base,large
    python scripts/vit_benchmark.py --batch-size 4 --frames 16
    python scripts/vit_benchmark.py --no-fp16               # disable bfloat16 on GPU
    python scripts/vit_benchmark.py --warmup 5 --runs 20    # more iterations

Flags:
    --device    cpu | cuda [default: cpu]
    --variants  comma-separated variant names [default: all]
    --batch-size  batch size [default: 2]
    --frames    number of video frames [default: 8]
    --img-size  spatial resolution [default: 224]
    --patch-size patch size [default: 16]
    --tubelet   tubelet size [default: 2]
    --no-fp16   disable bfloat16 even on GPU
    --warmup    warmup iterations [default: 3]
    --runs      timed iterations [default: 10]
    --forward-only  skip backward pass
    --backward  include backward pass in benchmark
"""

import argparse
import gc
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Add assets to path
_SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_SKILL_DIR, "assets"))

import torch
import torch.nn as nn

from vision_transformer_template import VisionTransformer


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS: Dict[str, dict] = {
    "vit_tiny":     {"embed_dim": 192,  "depth": 12, "num_heads": 3},
    "vit_small":    {"embed_dim": 384,  "depth": 12, "num_heads": 6},
    "vit_base":     {"embed_dim": 768,  "depth": 12, "num_heads": 12},
    "vit_large":    {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "vit_huge":     {"embed_dim": 1280, "depth": 32, "num_heads": 16},
    "vit_giant":    {"embed_dim": 1408, "depth": 40, "num_heads": 16},
    "vit_gigantic": {"embed_dim": 1664, "depth": 48, "num_heads": 16},
}

VARIANT_ACTIVATION_CHECKPOINTING = {"vit_giant", "vit_gigantic"}


# ---------------------------------------------------------------------------
# Memory utilities
# ---------------------------------------------------------------------------

def get_peak_memory_mb(device: str) -> float:
    """Return peak memory usage in MB."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        # CPU: use process RSS
        try:
            import resource
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            return rusage.ru_maxrss / 1024  # Linux: bytes -> KB -> MB
        except Exception:
            return float("nan")


def reset_peak_memory(device: str) -> None:
    """Reset peak memory counters."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def synchronize(device: str) -> None:
    """Synchronize GPU (no-op on CPU)."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Single variant benchmark
# ---------------------------------------------------------------------------

def benchmark_variant(
    name: str,
    variant_cfg: dict,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    frames: int,
    img_size: int,
    patch_size: int,
    tubelet_size: int,
    warmup: int,
    runs: int,
    include_backward: bool,
) -> Optional[Dict]:
    """
    Benchmark a single ViT variant.

    Returns a dict with results, or None if OOM.
    """
    use_act_ckpt = name in VARIANT_ACTIVATION_CHECKPOINTING

    # Build model
    try:
        model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_chans=3,
            embed_dim=variant_cfg["embed_dim"],
            depth=variant_cfg["depth"],
            num_heads=variant_cfg["num_heads"],
            use_activation_checkpointing=use_act_ckpt,
        )
        model = model.to(device=device, dtype=dtype)
        if not include_backward:
            model.training = False
        else:
            model.train()
    except torch.cuda.OutOfMemoryError:
        return {"name": name, "error": "OOM (model init)"}
    except Exception as exc:
        return {"name": name, "error": str(exc)}

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())

    # Prepare input
    if frames > 1:
        x_shape = (batch_size, 3, frames, img_size, img_size)
    else:
        x_shape = (batch_size, 3, img_size, img_size)

    try:
        x = torch.randn(*x_shape, device=device, dtype=dtype)
    except torch.cuda.OutOfMemoryError:
        return {"name": name, "error": "OOM (input allocation)"}

    # Context manager for grad / no-grad
    ctx_fn = torch.enable_grad if include_backward else torch.no_grad

    # Warmup
    for _ in range(warmup):
        try:
            with ctx_fn():
                out = model(x)
                if include_backward:
                    out.sum().backward()
        except torch.cuda.OutOfMemoryError:
            del model, x
            torch.cuda.empty_cache()
            return {"name": name, "error": "OOM (warmup)"}
        except Exception as exc:
            return {"name": name, "error": f"Forward error: {exc}"}
        synchronize(device)

    # Reset peak memory before timed runs
    reset_peak_memory(device)

    # Timed runs
    t_start = time.perf_counter()
    for _ in range(runs):
        try:
            with ctx_fn():
                out = model(x)
                if include_backward:
                    out.sum().backward()
        except torch.cuda.OutOfMemoryError:
            del model, x
            torch.cuda.empty_cache()
            return {"name": name, "error": "OOM (timed run)"}
        synchronize(device)
    t_end = time.perf_counter()

    elapsed_sec = t_end - t_start
    total_samples = batch_size * runs
    throughput = total_samples / elapsed_sec
    peak_mem_mb = get_peak_memory_mb(device)

    # Token count
    if frames > 1:
        n_tokens = (frames // tubelet_size) * (img_size // patch_size) ** 2
    else:
        n_tokens = (img_size // patch_size) ** 2

    del model, x
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "name":         name,
        "n_params_m":   n_params / 1e6,
        "n_tokens":     n_tokens,
        "throughput":   throughput,
        "peak_mem_mb":  peak_mem_mb,
        "ms_per_batch": (elapsed_sec / runs) * 1000,
        "error":        None,
    }


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def print_table(results: List[Dict], include_backward: bool, device: str) -> None:
    """Print a formatted results table."""
    mode = "fwd+bwd" if include_backward else "forward"

    print()
    print(f"{'='*90}")
    print(f"  V-JEPA 2 ViT Benchmark  |  device={device}  |  mode={mode}")
    print(f"{'='*90}")
    print(f"  {'Variant':<14}  {'Params':>8}  {'Tokens':>7}  "
          f"{'Samples/s':>10}  {'ms/batch':>9}  {'Peak Mem(MB)':>13}  {'Status'}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*9}  {'-'*13}  {'-'*10}")

    for r in results:
        if r.get("error"):
            line = (
                f"  {r['name']:<14}  {'N/A':>8}  {'N/A':>7}  "
                f"{'N/A':>10}  {'N/A':>9}  {'N/A':>13}  {r['error']}"
            )
        else:
            line = (
                f"  {r['name']:<14}  "
                f"{r['n_params_m']:>7.1f}M  "
                f"{r['n_tokens']:>7}  "
                f"{r['throughput']:>10.2f}  "
                f"{r['ms_per_batch']:>9.1f}  "
                f"{r['peak_mem_mb']:>13.1f}  "
                f"OK"
            )
        print(line)

    print(f"{'='*90}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark V-JEPA 2 ViT variants: throughput and memory."
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device to benchmark on (default: cpu).",
    )
    parser.add_argument(
        "--variants", default=None,
        help="Comma-separated list of variant names (default: all).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Batch size per iteration (default: 2).",
    )
    parser.add_argument(
        "--frames", type=int, default=8,
        help="Number of video frames (default: 8). Use 1 for image mode.",
    )
    parser.add_argument(
        "--img-size", type=int, default=224,
        help="Spatial resolution (default: 224).",
    )
    parser.add_argument(
        "--patch-size", type=int, default=16,
        help="Patch size (default: 16).",
    )
    parser.add_argument(
        "--tubelet", type=int, default=2,
        help="Tubelet size for 3D conv (default: 2).",
    )
    parser.add_argument(
        "--no-fp16", action="store_true",
        help="Disable bfloat16; use float32 even on GPU.",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup iterations (default: 3).",
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of timed iterations (default: 10).",
    )
    parser.add_argument(
        "--backward", action="store_true",
        help="Include backward pass in benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        device = "cpu"

    # Resolve dtype
    if device == "cuda" and not args.no_fp16:
        dtype = torch.bfloat16
        dtype_name = "bfloat16"
    else:
        dtype = torch.float32
        dtype_name = "float32"

    # Resolve variants
    if args.variants:
        selected = [v.strip() for v in args.variants.split(",")]
        unknown = [v for v in selected if v not in VARIANTS]
        if unknown:
            print(f"ERROR: Unknown variants: {unknown}")
            print(f"Available: {list(VARIANTS.keys())}")
            return 1
        variant_items = [(k, VARIANTS[k]) for k in selected]
    else:
        variant_items = list(VARIANTS.items())

    # Print config
    print()
    print(f"V-JEPA 2 ViT Benchmark")
    print(f"  device:     {device}")
    print(f"  dtype:      {dtype_name}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  frames:     {args.frames}")
    print(f"  img_size:   {args.img_size}x{args.img_size}")
    print(f"  patch_size: {args.patch_size}")
    print(f"  tubelet:    {args.tubelet}")
    print(f"  warmup:     {args.warmup} iterations")
    print(f"  runs:       {args.runs} iterations")
    print(f"  mode:       {'fwd+bwd' if args.backward else 'forward only'}")
    print(f"  variants:   {[k for k, _ in variant_items]}")
    print()

    results = []
    for i, (name, cfg) in enumerate(variant_items):
        print(f"  Benchmarking {name} ({i+1}/{len(variant_items)})...", end="", flush=True)

        result = benchmark_variant(
            name=name,
            variant_cfg=cfg,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            frames=args.frames,
            img_size=args.img_size,
            patch_size=args.patch_size,
            tubelet_size=args.tubelet,
            warmup=args.warmup,
            runs=args.runs,
            include_backward=args.backward,
        )

        if result is None:
            result = {"name": name, "error": "Unknown error"}

        if result.get("error"):
            print(f" {result['error']}")
        else:
            print(
                f" {result['throughput']:.1f} samples/s | "
                f"{result['ms_per_batch']:.1f}ms/batch | "
                f"{result['peak_mem_mb']:.0f}MB"
            )
        results.append(result)

    print_table(results, include_backward=args.backward, device=device)

    # Print notes
    failed = [r for r in results if r.get("error")]
    if failed:
        print("Notes:")
        for r in failed:
            print(f"  {r['name']}: {r['error']}")
        print()
        if any("OOM" in (r.get("error") or "") for r in failed):
            print("  To fix OOM errors for large variants, try:")
            print("    --batch-size 1")
            print("    --frames 4")
            print("    --no-fp16 (or use bfloat16 to reduce memory)")
            print("    Ensure activation_checkpointing is enabled (automatic for Giant/Gigantic)")
            print()

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
