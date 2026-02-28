#!/usr/bin/env python3
"""
scripts/inference_benchmark.py — Latency/Throughput Benchmarks for Inference Optimization

Benchmarks various inference configurations: batch sizes, dtypes, caching,
and sequential vs batched processing. All benchmarks use mock models
(nn.Linear, nn.Sequential) for reproducibility without the full brain_ai system.

Usage:
    python scripts/inference_benchmark.py
    python scripts/inference_benchmark.py --batch-sizes 1 8 32 64
    python scripts/inference_benchmark.py --n-runs 200
    python scripts/inference_benchmark.py --model-sizes small medium large
    python scripts/inference_benchmark.py --output results.json

Output:
    Prints a formatted table of benchmark results and optionally saves
    detailed results to a JSON file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ===========================================================================
# SECTION 1: Mock models
# ===========================================================================

class DictModel(nn.Module):
    """Wrapper that accepts dict inputs."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, inputs):
        if isinstance(inputs, dict):
            x = next(iter(inputs.values()))
        else:
            x = inputs
        return self.inner(x)


def create_model(size: str = "small") -> Tuple[nn.Module, int]:
    """Create a mock model of the specified size.

    Returns:
        (model, input_features) tuple.
    """
    if size == "small":
        inner = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        return DictModel(inner), 64
    elif size == "medium":
        inner = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        return DictModel(inner), 256
    elif size == "large":
        inner = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )
        return DictModel(inner), 1024
    else:
        raise ValueError(f"Unknown model size: {size}")


def model_param_count(model: nn.Module) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Compute model size in MB."""
    total_bytes = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1024 * 1024)


# ===========================================================================
# SECTION 2: Benchmark result dataclass
# ===========================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str = ""
    model_size: str = ""
    model_params: int = 0
    model_mb: float = 0.0
    batch_size: int = 1
    dtype: str = "fp32"
    n_runs: int = 0
    n_warmup: int = 0
    total_samples: int = 0
    total_time_ms: float = 0.0
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_sps: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)

    def summary_line(self) -> str:
        """One-line summary for table output."""
        return (
            f"{self.name:<35} "
            f"bs={self.batch_size:<4} "
            f"{self.dtype:<5} "
            f"lat={self.mean_latency_ms:>8.3f}ms "
            f"p99={self.p99_latency_ms:>8.3f}ms "
            f"tput={self.throughput_sps:>10.1f} sps"
        )


# ===========================================================================
# SECTION 3: Statistical helpers
# ===========================================================================

def compute_percentile(data: List[float], p: float) -> float:
    """Compute the p-th percentile."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (p / 100.0) * (len(s) - 1)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    d = k - f
    return s[f] + d * (s[c] - s[f])


def compute_mean(data: List[float]) -> float:
    if not data:
        return 0.0
    return sum(data) / len(data)


def compute_std(data: List[float]) -> float:
    if len(data) < 2:
        return 0.0
    m = compute_mean(data)
    var = sum((x - m) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(var)


# ===========================================================================
# SECTION 4: Benchmark functions
# ===========================================================================

def benchmark_sequential(
    model: nn.Module,
    in_features: int,
    batch_size: int,
    n_runs: int = 100,
    n_warmup: int = 20,
    dtype: torch.dtype = torch.float32,
) -> BenchmarkResult:
    """Benchmark sequential (one-at-a-time) inference."""
    model.eval()

    sample = torch.randn(1, in_features, dtype=dtype)

    # Warmup
    with torch.inference_mode():
        for _ in range(n_warmup):
            model({"features": sample})

    # Measure
    latencies = []
    with torch.inference_mode():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            for _ in range(batch_size):
                model({"features": sample})
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

    total_samples = n_runs * batch_size
    total_time_ms = sum(latencies)

    return BenchmarkResult(
        name=f"sequential-{batch_size}",
        batch_size=batch_size,
        dtype="fp16" if dtype == torch.float16 else "fp32",
        n_runs=n_runs,
        n_warmup=n_warmup,
        total_samples=total_samples,
        total_time_ms=total_time_ms,
        mean_latency_ms=compute_mean(latencies),
        std_latency_ms=compute_std(latencies),
        p50_latency_ms=compute_percentile(latencies, 50),
        p99_latency_ms=compute_percentile(latencies, 99),
        throughput_sps=(total_samples / (total_time_ms / 1000)) if total_time_ms > 0 else 0,
        latencies_ms=latencies,
    )


def benchmark_batched(
    model: nn.Module,
    in_features: int,
    batch_size: int,
    n_runs: int = 100,
    n_warmup: int = 20,
    dtype: torch.dtype = torch.float32,
) -> BenchmarkResult:
    """Benchmark batched inference."""
    model.eval()

    batch = torch.randn(batch_size, in_features, dtype=dtype)

    # Warmup
    with torch.inference_mode():
        for _ in range(n_warmup):
            model({"features": batch})

    # Measure
    latencies = []
    with torch.inference_mode():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model({"features": batch})
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

    total_samples = n_runs * batch_size
    total_time_ms = sum(latencies)

    return BenchmarkResult(
        name=f"batched-{batch_size}",
        batch_size=batch_size,
        dtype="fp16" if dtype == torch.float16 else "fp32",
        n_runs=n_runs,
        n_warmup=n_warmup,
        total_samples=total_samples,
        total_time_ms=total_time_ms,
        mean_latency_ms=compute_mean(latencies),
        std_latency_ms=compute_std(latencies),
        p50_latency_ms=compute_percentile(latencies, 50),
        p99_latency_ms=compute_percentile(latencies, 99),
        throughput_sps=(total_samples / (total_time_ms / 1000)) if total_time_ms > 0 else 0,
        latencies_ms=latencies,
    )


def benchmark_cached_vs_uncached(
    model: nn.Module,
    in_features: int,
    n_runs: int = 100,
    n_warmup: int = 20,
) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """Benchmark uncached vs cached inference.

    Returns:
        (uncached_result, cached_result) tuple.
    """
    model.eval()
    sample = torch.randn(1, in_features)

    cache: Dict[str, Tensor] = {}

    def make_key(t: Tensor) -> str:
        return hashlib.sha256(t.numpy().tobytes()).hexdigest()

    # Warmup
    with torch.inference_mode():
        for _ in range(n_warmup):
            model({"features": sample})

    # Uncached
    latencies_uncached = []
    with torch.inference_mode():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            output = model({"features": sample})
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_uncached.append(elapsed_ms)

    # Populate cache
    key = make_key(sample)
    with torch.inference_mode():
        cache[key] = model({"features": sample}).detach().clone()

    # Cached
    latencies_cached = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = cache.get(key)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_cached.append(elapsed_ms)

    uncached_total = sum(latencies_uncached)
    cached_total = sum(latencies_cached)

    uncached_result = BenchmarkResult(
        name="uncached",
        n_runs=n_runs,
        total_samples=n_runs,
        total_time_ms=uncached_total,
        mean_latency_ms=compute_mean(latencies_uncached),
        std_latency_ms=compute_std(latencies_uncached),
        p50_latency_ms=compute_percentile(latencies_uncached, 50),
        p99_latency_ms=compute_percentile(latencies_uncached, 99),
        throughput_sps=(n_runs / (uncached_total / 1000)) if uncached_total > 0 else 0,
        latencies_ms=latencies_uncached,
    )

    cached_result = BenchmarkResult(
        name="cached",
        n_runs=n_runs,
        total_samples=n_runs,
        total_time_ms=cached_total,
        mean_latency_ms=compute_mean(latencies_cached),
        std_latency_ms=compute_std(latencies_cached),
        p50_latency_ms=compute_percentile(latencies_cached, 50),
        p99_latency_ms=compute_percentile(latencies_cached, 99),
        throughput_sps=(n_runs / (cached_total / 1000)) if cached_total > 0 else 0,
        latencies_ms=latencies_cached,
    )

    return uncached_result, cached_result


def benchmark_warmup_effect(
    model: nn.Module,
    in_features: int,
    n_cold: int = 10,
    n_warm: int = 10,
    n_warmup: int = 50,
) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """Benchmark cold vs warm inference latency.

    Returns:
        (cold_result, warm_result) tuple.
    """
    model.eval()
    sample = torch.randn(1, in_features)

    # Cold runs (no warmup)
    latencies_cold = []
    with torch.inference_mode():
        for _ in range(n_cold):
            t0 = time.perf_counter()
            model({"features": sample})
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_cold.append(elapsed_ms)

    # Warmup
    with torch.inference_mode():
        for _ in range(n_warmup):
            model({"features": sample})

    # Warm runs
    latencies_warm = []
    with torch.inference_mode():
        for _ in range(n_warm):
            t0 = time.perf_counter()
            model({"features": sample})
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_warm.append(elapsed_ms)

    cold_result = BenchmarkResult(
        name="cold",
        n_runs=n_cold,
        mean_latency_ms=compute_mean(latencies_cold),
        std_latency_ms=compute_std(latencies_cold),
        p50_latency_ms=compute_percentile(latencies_cold, 50),
        p99_latency_ms=compute_percentile(latencies_cold, 99),
        latencies_ms=latencies_cold,
    )

    warm_result = BenchmarkResult(
        name="warm",
        n_runs=n_warm,
        mean_latency_ms=compute_mean(latencies_warm),
        std_latency_ms=compute_std(latencies_warm),
        p50_latency_ms=compute_percentile(latencies_warm, 50),
        p99_latency_ms=compute_percentile(latencies_warm, 99),
        latencies_ms=latencies_warm,
    )

    return cold_result, warm_result


# ===========================================================================
# SECTION 5: Main benchmark runner
# ===========================================================================

def run_benchmarks(
    model_sizes: List[str],
    batch_sizes: List[int],
    n_runs: int = 100,
    n_warmup: int = 20,
    output_path: Optional[str] = None,
) -> List[BenchmarkResult]:
    """Run all benchmarks and return results."""

    all_results: List[BenchmarkResult] = []

    for size in model_sizes:
        model, in_features = create_model(size)
        params = model_param_count(model)
        mb = model_size_mb(model)

        print(f"\n{'='*70}")
        print(f"Model: {size} ({params:,} params, {mb:.2f}MB)")
        print(f"{'='*70}")

        # --- Sequential vs Batched ---
        print(f"\n  Throughput Comparison (sequential vs batched):")
        print(f"  {'Name':<35} {'BS':<5} {'Dtype':<5} "
              f"{'Latency':>10} {'P99':>10} {'Throughput':>12}")
        print(f"  {'-'*80}")

        for bs in batch_sizes:
            # Sequential
            r_seq = benchmark_sequential(
                model, in_features, bs, n_runs=n_runs, n_warmup=n_warmup
            )
            r_seq.model_size = size
            r_seq.model_params = params
            r_seq.model_mb = mb
            all_results.append(r_seq)
            print(f"  {r_seq.summary_line()}")

            # Batched
            r_bat = benchmark_batched(
                model, in_features, bs, n_runs=n_runs, n_warmup=n_warmup
            )
            r_bat.model_size = size
            r_bat.model_params = params
            r_bat.model_mb = mb
            all_results.append(r_bat)
            print(f"  {r_bat.summary_line()}")

            # Speedup
            if r_seq.throughput_sps > 0:
                speedup = r_bat.throughput_sps / r_seq.throughput_sps
                print(f"  {'':>35} --> Batch speedup: {speedup:.2f}x")
            print()

        # --- Cached vs Uncached ---
        print(f"\n  Cache Hit Comparison:")
        r_uncached, r_cached = benchmark_cached_vs_uncached(
            model, in_features, n_runs=n_runs, n_warmup=n_warmup
        )
        r_uncached.model_size = size
        r_cached.model_size = size
        all_results.extend([r_uncached, r_cached])
        print(f"  {r_uncached.summary_line()}")
        print(f"  {r_cached.summary_line()}")
        if r_cached.mean_latency_ms > 0:
            cache_speedup = r_uncached.mean_latency_ms / r_cached.mean_latency_ms
            print(f"  {'':>35} --> Cache speedup: {cache_speedup:.1f}x")

        # --- Warmup Effect ---
        print(f"\n  Warmup Effect:")
        r_cold, r_warm = benchmark_warmup_effect(model, in_features)
        r_cold.model_size = size
        r_warm.model_size = size
        all_results.extend([r_cold, r_warm])
        print(f"  Cold: mean={r_cold.mean_latency_ms:.3f}ms "
              f"std={r_cold.std_latency_ms:.3f}ms")
        print(f"  Warm: mean={r_warm.mean_latency_ms:.3f}ms "
              f"std={r_warm.std_latency_ms:.3f}ms")

    # --- Save results ---
    if output_path:
        serializable = []
        for r in all_results:
            d = asdict(r)
            serializable.append(d)

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return all_results


# ===========================================================================
# SECTION 6: Summary statistics
# ===========================================================================

def print_summary(results: List[BenchmarkResult]) -> None:
    """Print a summary table of all benchmark results."""
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")

    # Group by model size
    sizes = sorted(set(r.model_size for r in results if r.model_size))
    for size in sizes:
        size_results = [r for r in results if r.model_size == size]
        print(f"\n  Model: {size}")
        print(f"  {'Name':<30} {'Mean(ms)':>10} {'P99(ms)':>10} {'Tput(sps)':>12}")
        print(f"  {'-'*65}")
        for r in size_results:
            if r.throughput_sps > 0:
                print(
                    f"  {r.name:<30} {r.mean_latency_ms:>10.3f} "
                    f"{r.p99_latency_ms:>10.3f} {r.throughput_sps:>12.1f}"
                )
            else:
                print(
                    f"  {r.name:<30} {r.mean_latency_ms:>10.3f} "
                    f"{r.p99_latency_ms:>10.3f} {'N/A':>12}"
                )


# ===========================================================================
# SECTION 7: Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inference optimization benchmarks"
    )
    parser.add_argument(
        "--model-sizes",
        nargs="+",
        default=["small", "medium", "large"],
        choices=["small", "medium", "large"],
        help="Model sizes to benchmark",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 16, 32],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=100,
        help="Number of profiling runs per benchmark",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=20,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON",
    )
    args = parser.parse_args()

    print("Inference Optimization Benchmark Suite")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Model sizes: {args.model_sizes}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Runs per benchmark: {args.n_runs}")

    results = run_benchmarks(
        model_sizes=args.model_sizes,
        batch_sizes=args.batch_sizes,
        n_runs=args.n_runs,
        n_warmup=args.n_warmup,
        output_path=args.output,
    )

    print_summary(results)
    print(f"\nTotal benchmarks run: {len(results)}")


if __name__ == "__main__":
    main()
