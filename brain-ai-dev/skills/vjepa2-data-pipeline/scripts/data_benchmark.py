#!/usr/bin/env python3
"""
data_benchmark.py
=================
Benchmarks for the V-JEPA 2 data pipeline.

Measures:
1. Video loading throughput (clips per second)
2. Augmentation overhead (transform time vs raw loading)
3. DataLoader throughput scaling with num_workers

Usage
-----
    python scripts/data_benchmark.py
    python scripts/data_benchmark.py --benchmark loading
    python scripts/data_benchmark.py --benchmark augmentation
    python scripts/data_benchmark.py --benchmark workers
    python scripts/data_benchmark.py --all --fpc 16 --img-size 224 --n-clips 200
    python scripts/data_benchmark.py --all --export results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SKILL_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = SKILL_ROOT / "assets"
sys.path.insert(0, str(ASSETS_DIR))

try:
    from video_dataset_template import VideoDataset
    from video_transforms_template import VideoTransformPipeline
    from distributed_sampler_template import DistributedWeightedSampler
    from data_config_template import DataConfig, AugConfig
    _IMPORTS_OK = True
except ImportError as e:
    print(f"[ERROR] Failed to import asset modules: {e}")
    print(f"        Ensure assets/ are at {ASSETS_DIR}")
    _IMPORTS_OK = False


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

class Timer:
    """Context manager for measuring wall-clock time."""

    def __init__(self, name: str = ""):
        self.name    = name
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
        return False

    def __str__(self):
        return f"{self.name}: {self.elapsed*1000:.2f}ms"


def stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, min, max, p50, p95 for a list of floats."""
    arr = np.array(values, dtype=float)
    return {
        "mean":  float(arr.mean()),
        "std":   float(arr.std()),
        "min":   float(arr.min()),
        "max":   float(arr.max()),
        "p50":   float(np.percentile(arr, 50)),
        "p95":   float(np.percentile(arr, 95)),
        "count": len(values),
    }


def print_stats(name: str, values: List[float], unit: str = "clips/s") -> Dict:
    s = stats(values)
    print(f"  {name}:")
    print(f"    mean={s['mean']:.2f} {unit}  std={s['std']:.2f}")
    print(f"    min={s['min']:.2f}  p50={s['p50']:.2f}  p95={s['p95']:.2f}  max={s['max']:.2f}")
    return s


# ---------------------------------------------------------------------------
# Benchmark 1: Video Loading Throughput
# ---------------------------------------------------------------------------

def benchmark_loading(
    n_clips: int = 500,
    frames_per_clip: int = 16,
    img_size: int = 112,
    synthetic_num_frames: int = 300,
) -> Dict[str, Any]:
    """
    Measure raw video loading throughput (no transforms) in clips/second.

    Tests all three clip modes.
    """
    print("\n" + "=" * 60)
    print("Benchmark 1: Video Loading Throughput")
    print("=" * 60)
    print(f"  n_clips={n_clips}, fpc={frames_per_clip}, img_size={img_size}")

    results = {}

    for mode in ["fps", "duration", "frame_step"]:
        ds = VideoDataset(
            data_paths=[],
            clip_mode=mode,
            frames_per_clip=frames_per_clip,
            img_size=img_size,
            synthetic_num_frames=synthetic_num_frames,
        )

        # Warmup
        for i in range(min(5, n_clips)):
            _ = ds[i % len(ds)]

        # Timed run
        times_ms: List[float] = []
        for i in range(n_clips):
            with Timer() as t:
                sample = ds[i % len(ds)]
            times_ms.append(t.elapsed * 1000)

        throughputs = [1000.0 / ms for ms in times_ms]  # clips/second
        s = print_stats(f"clip_mode={mode!r}", throughputs)
        results[mode] = s

        mean_ms = stats(times_ms)["mean"]
        print(f"    avg_load_time={mean_ms:.2f}ms")

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: Augmentation Overhead
# ---------------------------------------------------------------------------

def benchmark_augmentation(
    n_clips: int = 300,
    frames_per_clip: int = 16,
    img_size: int = 112,
) -> Dict[str, Any]:
    """
    Measure augmentation pipeline overhead.

    Compares:
    - Raw loading (no transform)
    - With eval transform (center crop only)
    - With train transform (full augmentation)
    - With train transform + RandAugment
    """
    print("\n" + "=" * 60)
    print("Benchmark 2: Augmentation Overhead")
    print("=" * 60)
    print(f"  n_clips={n_clips}, fpc={frames_per_clip}, img_size={img_size}")

    # Create synthetic frames list to reuse
    rng = np.random.RandomState(0)
    SYNTH_FRAMES = 300
    raw_frames = rng.randint(0, 256, (SYNTH_FRAMES, img_size * 2, img_size * 2, 3), dtype=np.uint8)
    frame_list = [raw_frames[t] for t in range(SYNTH_FRAMES)]
    clip_frames = frame_list[:frames_per_clip]

    results = {}

    configs = {
        "no_augment (eval)": {
            "crop_scale":      (1.0, 1.0),
            "horizontal_flip": False,
            "auto_augment":    False,
            "motion_shift":    False,
            "random_erasing":  0.0,
            "normalize_mean":  (0.485, 0.456, 0.406),
            "normalize_std":   (0.229, 0.224, 0.225),
        },
        "train (no randaug)": {
            "crop_scale":      (0.3, 1.0),
            "crop_ratio":      (0.75, 1.33),
            "horizontal_flip": True,
            "auto_augment":    False,
            "motion_shift":    True,
            "random_erasing":  0.0,
            "normalize_mean":  (0.485, 0.456, 0.406),
            "normalize_std":   (0.229, 0.224, 0.225),
        },
        "train (+ random_erasing)": {
            "crop_scale":      (0.3, 1.0),
            "crop_ratio":      (0.75, 1.33),
            "horizontal_flip": True,
            "auto_augment":    False,
            "motion_shift":    True,
            "random_erasing":  0.25,
            "normalize_mean":  (0.485, 0.456, 0.406),
            "normalize_std":   (0.229, 0.224, 0.225),
        },
    }

    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False

    if has_pil:
        configs["train (+ randaug, n=2 m=9)"] = {
            "crop_scale":      (0.3, 1.0),
            "crop_ratio":      (0.75, 1.33),
            "horizontal_flip": True,
            "auto_augment":    True,
            "rand_augment_n":  2,
            "rand_augment_m":  9,
            "motion_shift":    True,
            "random_erasing":  0.0,
            "normalize_mean":  (0.485, 0.456, 0.406),
            "normalize_std":   (0.229, 0.224, 0.225),
        }

    for cfg_name, cfg_dict in configs.items():
        pipeline = VideoTransformPipeline(cfg_dict, img_size=img_size)

        # Use train transform for all configs except eval
        if "eval" in cfg_name:
            tfm = pipeline.get_eval_transform()
        else:
            tfm = pipeline.get_train_transform()

        # Warmup
        for _ in range(min(5, n_clips)):
            _ = tfm(clip_frames)

        # Timed run
        times_ms: List[float] = []
        for _ in range(n_clips):
            with Timer() as t:
                _ = tfm(clip_frames)
            times_ms.append(t.elapsed * 1000)

        throughputs = [1000.0 / ms for ms in times_ms]
        s = print_stats(cfg_name, throughputs)
        s["mean_ms"] = stats(times_ms)["mean"]
        results[cfg_name] = s
        print(f"    avg_transform_time={s['mean_ms']:.2f}ms/clip")

    # Print relative overhead
    if "no_augment (eval)" in results and "train (no randaug)" in results:
        baseline = results["no_augment (eval)"]["mean_ms"]
        print("\n  Relative overhead vs eval (lower is better):")
        for name, r in results.items():
            overhead = r["mean_ms"] / baseline
            print(f"    {name}: {overhead:.2f}x")

    return results


# ---------------------------------------------------------------------------
# Benchmark 3: DataLoader Scaling with num_workers
# ---------------------------------------------------------------------------

def benchmark_dataloader_workers(
    n_batches: int = 50,
    batch_size: int = 4,
    frames_per_clip: int = 8,
    img_size: int = 64,
    worker_counts: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Measure DataLoader throughput scaling with num_workers.

    Reports clips/second for each worker count.
    """
    from torch.utils.data import DataLoader

    print("\n" + "=" * 60)
    print("Benchmark 3: DataLoader Scaling with num_workers")
    print("=" * 60)
    print(f"  n_batches={n_batches}, batch_size={batch_size}, fpc={frames_per_clip}")

    if worker_counts is None:
        import os
        cpu_count = os.cpu_count() or 4
        worker_counts = [0, 1, 2, 4]
        if cpu_count >= 8:
            worker_counts.append(8)

    aug_cfg = {
        "crop_scale":      (0.5, 1.0),
        "horizontal_flip": False,
        "auto_augment":    False,
        "motion_shift":    False,
        "random_erasing":  0.0,
        "normalize_mean":  (0.485, 0.456, 0.406),
        "normalize_std":   (0.229, 0.224, 0.225),
    }
    pipeline = VideoTransformPipeline(aug_cfg, img_size=img_size)
    train_tfm = pipeline.get_train_transform()

    dataset = VideoDataset(
        data_paths=[],
        clip_mode="fps",
        frames_per_clip=frames_per_clip,
        img_size=img_size,
        transform=train_tfm,
        synthetic_num_frames=300,
    )

    results = {}
    clips_per_batch = batch_size

    for nw in worker_counts:
        # Build DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=nw,
            drop_last=True,
            pin_memory=False,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
        )

        # Warmup: 2 batches
        loader_iter = iter(loader)
        for _ in range(min(2, n_batches)):
            try:
                _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                _ = next(loader_iter)

        # Timed run
        clips_per_second_list = []
        loader_iter = iter(loader)
        completed = 0

        while completed < n_batches:
            try:
                with Timer() as t:
                    batch = next(loader_iter)
                clips_in_batch = batch["video"].shape[0] if isinstance(batch, dict) else batch_size
                cps = clips_in_batch / t.elapsed
                clips_per_second_list.append(cps)
                completed += 1
            except StopIteration:
                loader_iter = iter(loader)

        s = print_stats(f"num_workers={nw}", clips_per_second_list)
        results[nw] = s

        # Cleanup persistent workers
        del loader

    # Speedup vs baseline (num_workers=0)
    if 0 in results:
        baseline = results[0]["mean"]
        print("\n  Speedup vs num_workers=0:")
        for nw, r in sorted(results.items()):
            speedup = r["mean"] / baseline
            print(f"    workers={nw}: {speedup:.2f}x ({r['mean']:.1f} clips/s)")

    return {str(k): v for k, v in results.items()}


# ---------------------------------------------------------------------------
# Benchmark 4: Sampler Throughput
# ---------------------------------------------------------------------------

def benchmark_sampler(
    n_samples: int = 10000,
    world_size: int = 4,
) -> Dict[str, Any]:
    """
    Measure DistributedWeightedSampler iteration throughput.
    """
    print("\n" + "=" * 60)
    print("Benchmark 4: Sampler Throughput")
    print("=" * 60)

    weights = [1.0] * n_samples
    results = {}

    for ws in [1, world_size]:
        s = DistributedWeightedSampler(weights, n_samples, rank=0, world_size=ws, seed=0)
        times_ms = []
        for epoch in range(5):
            s.set_epoch(epoch)
            with Timer() as t:
                _ = list(s)
            times_ms.append(t.elapsed * 1000)

        throughput = [n_samples / (ms / 1000) for ms in times_ms]
        st = print_stats(f"world_size={ws}", throughput, unit="samples/s")
        st["mean_ms"] = stats(times_ms)["mean"]
        results[f"ws_{ws}"] = st
        print(f"    avg_iteration_time={st['mean_ms']:.2f}ms for {n_samples} samples")

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V-JEPA 2 data pipeline benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark", type=str, default="all",
        choices=["all", "loading", "augmentation", "workers", "sampler"],
        help="Which benchmark to run (default: all)"
    )
    parser.add_argument("--fpc",      type=int, default=16,  help="Frames per clip")
    parser.add_argument("--img-size", type=int, default=112, help="Spatial image size")
    parser.add_argument("--n-clips",  type=int, default=200, help="Number of clips to time")
    parser.add_argument("--n-batches",type=int, default=30,  help="Number of batches for loader benchmark")
    parser.add_argument("--batch-size",type=int, default=4,  help="Batch size for loader benchmark")
    parser.add_argument("--n-samples",type=int, default=5000,help="Samples for sampler benchmark")
    parser.add_argument("--export",   type=str, default="",  help="Export results to JSON file")
    parser.add_argument("--verbose",  action="store_true",   help="Extra output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not _IMPORTS_OK:
        print("[ERROR] Cannot run benchmarks: asset imports failed")
        return 1

    print("=" * 60)
    print("V-JEPA 2 Data Pipeline Benchmarks")
    print("=" * 60)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Device:  {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    try:
        import decord
        print(f"  decord:  {decord.__version__} (available)")
    except ImportError:
        print("  decord:  not available (using synthetic mode)")

    all_results: Dict[str, Any] = {
        "config": {
            "fpc":        args.fpc,
            "img_size":   args.img_size,
            "n_clips":    args.n_clips,
            "n_batches":  args.n_batches,
            "batch_size": args.batch_size,
        }
    }

    run_all        = (args.benchmark == "all")
    run_loading    = run_all or args.benchmark == "loading"
    run_aug        = run_all or args.benchmark == "augmentation"
    run_workers    = run_all or args.benchmark == "workers"
    run_sampler    = run_all or args.benchmark == "sampler"

    if run_loading:
        try:
            results = benchmark_loading(
                n_clips=args.n_clips,
                frames_per_clip=args.fpc,
                img_size=args.img_size,
            )
            all_results["loading"] = results
        except Exception as exc:
            print(f"[ERROR] Loading benchmark failed: {exc}")

    if run_aug:
        try:
            results = benchmark_augmentation(
                n_clips=args.n_clips,
                frames_per_clip=args.fpc,
                img_size=args.img_size,
            )
            all_results["augmentation"] = results
        except Exception as exc:
            print(f"[ERROR] Augmentation benchmark failed: {exc}")

    if run_workers:
        try:
            results = benchmark_dataloader_workers(
                n_batches=args.n_batches,
                batch_size=args.batch_size,
                frames_per_clip=min(args.fpc, 8),  # Smaller for speed
                img_size=min(args.img_size, 64),
            )
            all_results["workers"] = results
        except Exception as exc:
            print(f"[ERROR] Workers benchmark failed: {exc}")

    if run_sampler:
        try:
            results = benchmark_sampler(n_samples=args.n_samples)
            all_results["sampler"] = results
        except Exception as exc:
            print(f"[ERROR] Sampler benchmark failed: {exc}")

    # Export results
    if args.export:
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults exported to: {export_path}")

    print("\n" + "=" * 60)
    print("Benchmarks complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
