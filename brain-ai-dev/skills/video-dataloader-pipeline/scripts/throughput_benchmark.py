"""
throughput_benchmark.py -- Benchmark video loading throughput.

Measures frames/sec and batches/sec for both PyTorch (decord-based DataLoader)
and TFRecord (tf.data pipeline) backends.

Usage:
    python throughput_benchmark.py --data_root /path/to/videos --num_workers 4
    python throughput_benchmark.py --backend pytorch --batch_size 8 --num_batches 50
    python throughput_benchmark.py --backend both --json results.json

Output example:
    +----------+------------+------------+--------------+--------------+
    | Backend  | Frames/sec | Batches/sec| Avg batch ms | Peak mem MB  |
    +----------+------------+------------+--------------+--------------+
    | pytorch  |    1234.5  |     19.3   |      52.0    |     1024.0   |
    | tfdata   |    2468.1  |     38.6   |      25.9    |      512.0   |
    +----------+------------+------------+--------------+--------------+

CRITICAL: Never use bare .eval on nn.Module -- use module.train(False) instead.
For inference benchmarks, call model.train(False) before your forward pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
try:
    import decord
    from decord import VideoReader, cpu
    decord.bridge.set_bridge("torch")
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    backend: str
    num_batches: int
    batch_size: int
    num_frames: int
    frames_per_sec: float
    batches_per_sec: float
    avg_batch_ms: float
    total_elapsed_sec: float
    peak_memory_mb: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

def get_peak_memory_mb() -> float:
    """Return peak process memory in MB (RSS)."""
    if _PSUTIL_AVAILABLE:
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    # Fallback: use /proc/self/status on Linux
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Synthetic dataset (used when no real videos are available)
# ---------------------------------------------------------------------------

class _SyntheticVideoDataset(torch.utils.data.Dataset):
    """
    Returns random tensors with the same shape as real video batches.
    Used for benchmarking DataLoader overhead when video files are unavailable.
    """
    def __init__(self, n: int = 1000, num_frames: int = 16, crop_size: int = 224) -> None:
        self.n = n
        self.num_frames = num_frames
        self.crop_size = crop_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        T, C, H, W = self.num_frames, 3, self.crop_size, self.crop_size
        return {
            "video": torch.rand(T, C, H, W, dtype=torch.float32),
            "label": torch.tensor(idx % 10, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# PyTorch benchmark
# ---------------------------------------------------------------------------

def benchmark_pytorch(
    data_root: str,
    num_workers: int,
    batch_size: int,
    num_batches: int,
    num_frames: int,
    crop_size: int,
) -> BenchmarkResult:
    """
    Benchmark PyTorch DataLoader throughput.

    Uses real VideoDataset if decord + data_root are available,
    otherwise falls back to SyntheticVideoDataset.
    """
    print(f"[pytorch] Setting up... (num_workers={num_workers}, batch_size={batch_size})")

    use_real = _DECORD_AVAILABLE and os.path.isdir(data_root)

    if use_real:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "assets"))
        try:
            from video_dataset_template import scan_manifest, VideoDataset, VideoDatasetConfig
            from augmentation_template import (
                build_inference_transform, prepare_video_tensor, _AugmentConfig
            )
            from torchvision import tv_tensors

            cfg = VideoDatasetConfig(num_frames=num_frames, crop_size=crop_size)
            aug_cfg = _AugmentConfig(crop_size=crop_size)
            infer_tfm = build_inference_transform(aug_cfg)

            def _transform(frames_tchw):
                vid = tv_tensors.Video(frames_tchw)
                return infer_tfm(vid)

            manifest = scan_manifest(data_root, split="train")
            if not manifest:
                raise ValueError(f"No videos found in {data_root}/train")
            dataset = VideoDataset(manifest, cfg, transform=_transform)
            source = "real videos"
        except Exception as exc:
            print(f"[pytorch] WARNING: falling back to synthetic dataset ({exc})")
            dataset = _SyntheticVideoDataset(
                n=max(num_batches * batch_size * 2, 500),
                num_frames=num_frames,
                crop_size=crop_size,
            )
            source = "synthetic"
    else:
        dataset = _SyntheticVideoDataset(
            n=max(num_batches * batch_size * 2, 500),
            num_frames=num_frames,
            crop_size=crop_size,
        )
        source = "synthetic (decord not available or data_root not found)"

    print(f"[pytorch] Dataset source: {source} ({len(dataset)} samples)")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )

    # Warm up: 2 batches (first batch is slow due to worker init)
    print("[pytorch] Warming up...")
    warmup_done = 0
    for _ in loader:
        warmup_done += 1
        if warmup_done >= 2:
            break

    batch_times = []
    total_frames = 0
    mem_start = get_peak_memory_mb()

    print(f"[pytorch] Running {num_batches} batches...")
    t_start = time.perf_counter()

    try:
        batch_count = 0
        for batch in loader:
            t_batch_start = time.perf_counter()
            _ = batch["video"]
            t_batch_end = time.perf_counter()
            batch_times.append((t_batch_end - t_batch_start) * 1000)
            total_frames += batch_size * num_frames
            batch_count += 1
            if batch_count >= num_batches:
                break

        t_total = time.perf_counter() - t_start
        peak_mem = get_peak_memory_mb()

        result = BenchmarkResult(
            backend="pytorch",
            num_batches=batch_count,
            batch_size=batch_size,
            num_frames=num_frames,
            frames_per_sec=total_frames / t_total if t_total > 0 else 0.0,
            batches_per_sec=batch_count / t_total if t_total > 0 else 0.0,
            avg_batch_ms=float(np.mean(batch_times)) if batch_times else 0.0,
            total_elapsed_sec=t_total,
            peak_memory_mb=peak_mem - mem_start,
        )

    except Exception as exc:
        result = BenchmarkResult(
            backend="pytorch",
            num_batches=0,
            batch_size=batch_size,
            num_frames=num_frames,
            frames_per_sec=0.0,
            batches_per_sec=0.0,
            avg_batch_ms=0.0,
            total_elapsed_sec=0.0,
            peak_memory_mb=0.0,
            error=str(exc),
        )

    return result


# ---------------------------------------------------------------------------
# tf.data benchmark
# ---------------------------------------------------------------------------

def benchmark_tfdata(
    data_root: str,
    batch_size: int,
    num_batches: int,
    num_frames: int,
    crop_size: int,
) -> BenchmarkResult:
    """
    Benchmark tf.data pipeline throughput from TFRecord files.

    Expects TFRecord files at data_root/tfrecords/train-*.tfrecord.
    Falls back to a synthetic tf.data pipeline if files are not found.
    """
    if not _TF_AVAILABLE:
        return BenchmarkResult(
            backend="tfdata",
            num_batches=0,
            batch_size=batch_size,
            num_frames=num_frames,
            frames_per_sec=0.0,
            batches_per_sec=0.0,
            avg_batch_ms=0.0,
            total_elapsed_sec=0.0,
            peak_memory_mb=0.0,
            error="tensorflow not available",
        )

    print("[tfdata] Setting up...")

    tfrecord_dir = os.path.join(data_root, "tfrecords")
    shard_pattern = os.path.join(tfrecord_dir, "train-*.tfrecord")
    shards_exist = os.path.isdir(tfrecord_dir) and any(
        f.endswith(".tfrecord") for f in os.listdir(tfrecord_dir)
    ) if os.path.isdir(tfrecord_dir) else False

    if shards_exist:
        print(f"[tfdata] Reading TFRecords from {tfrecord_dir}")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "assets"))
        from tf_pipeline_template import TFPipelineConfig, build_tf_video_pipeline

        cfg = TFPipelineConfig(num_frames=num_frames, crop_size=crop_size, augment=False)
        dataset = build_tf_video_pipeline(
            shard_pattern=shard_pattern,
            cfg=cfg,
            batch_size=batch_size,
            is_training=False,
        )
        source = f"TFRecords ({tfrecord_dir})"
    else:
        print("[tfdata] No TFRecords found, using synthetic tf.data.Dataset")
        # Create a synthetic tf.data pipeline of random tensors
        T, C, H, W = num_frames, 3, crop_size, crop_size
        B = batch_size

        def _synthetic_gen():
            while True:
                yield {
                    "video": tf.random.uniform((T, H, W, C), dtype=tf.float32),
                    "label": tf.constant(0, dtype=tf.int32),
                }

        output_sig = {
            "video": tf.TensorSpec(shape=(T, H, W, C), dtype=tf.float32),
            "label": tf.TensorSpec(shape=(), dtype=tf.int32),
        }
        dataset = (
            tf.data.Dataset.from_generator(_synthetic_gen, output_signature=output_sig)
            .batch(B, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        source = "synthetic tf.data"

    print(f"[tfdata] Dataset source: {source}")

    # Warm up
    print("[tfdata] Warming up...")
    for _ in dataset.take(2):
        pass

    batch_times = []
    total_frames = 0
    mem_start = get_peak_memory_mb()
    batch_count = 0
    error_msg = None

    print(f"[tfdata] Running {num_batches} batches...")
    t_start = time.perf_counter()

    try:
        for batch in dataset.take(num_batches):
            t0 = time.perf_counter()
            _ = batch["video"].numpy() if hasattr(batch["video"], "numpy") else batch["video"]
            t1 = time.perf_counter()
            batch_times.append((t1 - t0) * 1000)
            total_frames += batch_size * num_frames
            batch_count += 1
    except Exception as exc:
        error_msg = str(exc)

    t_total = time.perf_counter() - t_start
    peak_mem = get_peak_memory_mb()

    return BenchmarkResult(
        backend="tfdata",
        num_batches=batch_count,
        batch_size=batch_size,
        num_frames=num_frames,
        frames_per_sec=total_frames / t_total if t_total > 0 and not error_msg else 0.0,
        batches_per_sec=batch_count / t_total if t_total > 0 and not error_msg else 0.0,
        avg_batch_ms=float(np.mean(batch_times)) if batch_times else 0.0,
        total_elapsed_sec=t_total,
        peak_memory_mb=peak_mem - mem_start,
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_result(result: BenchmarkResult) -> None:
    """Print a single benchmark result in table-friendly format."""
    if result.error:
        print(f"  Backend:   {result.backend}")
        print(f"  ERROR:     {result.error}")
        return
    print(f"  Backend:        {result.backend}")
    print(f"  Batches run:    {result.num_batches}")
    print(f"  Frames/sec:     {result.frames_per_sec:,.1f}")
    print(f"  Batches/sec:    {result.batches_per_sec:.2f}")
    print(f"  Avg batch ms:   {result.avg_batch_ms:.1f}")
    print(f"  Total time:     {result.total_elapsed_sec:.2f}s")
    print(f"  Peak mem delta: {result.peak_memory_mb:.1f} MB")


def print_comparison_table(results: List[BenchmarkResult]) -> None:
    """Print side-by-side comparison of multiple backends."""
    header = (
        f"{'Backend':<12} {'Frames/sec':>12} {'Batches/sec':>12} "
        f"{'Avg ms':>10} {'Peak MB':>10} {'Status':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        if r.error:
            status = f"ERROR"
        else:
            status = "OK"
        print(
            f"{r.backend:<12} {r.frames_per_sec:>12,.1f} {r.batches_per_sec:>12.2f} "
            f"{r.avg_batch_ms:>10.1f} {r.peak_memory_mb:>10.1f} {status:>8}"
        )
    print(sep)
    if len(results) == 2 and not any(r.error for r in results):
        r0, r1 = results
        ratio = r1.frames_per_sec / r0.frames_per_sec if r0.frames_per_sec > 0 else float("inf")
        faster = r1.backend if ratio > 1 else r0.backend
        print(f"  => {faster} is {max(ratio, 1/ratio if ratio>0 else 0):.2f}x faster in frames/sec")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark video loading throughput for PyTorch and tf.data backends."
    )
    parser.add_argument(
        "--data_root",
        default="",
        help="Root directory of the video dataset (used for both backends)",
    )
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers (PyTorch backend)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for both backends")
    parser.add_argument("--num_batches", type=int, default=50,
                        help="Number of batches to time (after warm-up)")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Frames per clip")
    parser.add_argument("--crop_size", type=int, default=224,
                        help="Spatial resolution of each frame")
    parser.add_argument(
        "--backend",
        choices=["pytorch", "tfdata", "both"],
        default="pytorch",
        help="Which backend(s) to benchmark",
    )
    parser.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="Write results as JSON to this path (for CI integration)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Video DataLoader Pipeline -- Throughput Benchmark")
    print("=" * 70)
    print(f"  batch_size:  {args.batch_size}")
    print(f"  num_frames:  {args.num_frames}")
    print(f"  crop_size:   {args.crop_size}")
    print(f"  num_batches: {args.num_batches} (plus warm-up)")
    print(f"  backends:    {args.backend}")
    print()

    results: List[BenchmarkResult] = []

    if args.backend in ("pytorch", "both"):
        print("-" * 40)
        print("PyTorch Backend")
        print("-" * 40)
        r = benchmark_pytorch(
            data_root=args.data_root,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            num_frames=args.num_frames,
            crop_size=args.crop_size,
        )
        print_result(r)
        results.append(r)

    if args.backend in ("tfdata", "both"):
        print("\n" + "-" * 40)
        print("tf.data Backend")
        print("-" * 40)
        r = benchmark_tfdata(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            num_frames=args.num_frames,
            crop_size=args.crop_size,
        )
        print_result(r)
        results.append(r)

    if len(results) > 1:
        print("\n" + "=" * 70)
        print("Comparison")
        print("=" * 70)
        print_comparison_table(results)

    # JSON output for CI
    if args.json:
        out_data = {
            "config": {
                "batch_size": args.batch_size,
                "num_frames": args.num_frames,
                "crop_size": args.crop_size,
                "num_batches": args.num_batches,
                "num_workers": args.num_workers,
            },
            "results": [r.to_dict() for r in results],
        }
        with open(args.json, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nResults written to {args.json}")

    print("\nDone.")
    # Return non-zero if any backend had an error
    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
