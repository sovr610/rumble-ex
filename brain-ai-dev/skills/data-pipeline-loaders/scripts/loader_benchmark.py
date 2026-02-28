#!/usr/bin/env python3
"""
scripts/loader_benchmark.py -- Throughput benchmarks for the data loading pipeline.

Measures:
  - Loader instantiation time
  - Batch iteration throughput (samples/sec, batches/sec)
  - Memory usage per phase
  - Augmentation overhead
  - Multi-phase sequential loading

Usage:
    python scripts/loader_benchmark.py
    python scripts/loader_benchmark.py --phases 1 2 3
    python scripts/loader_benchmark.py --batch-sizes 16 32 64 128
    python scripts/loader_benchmark.py --num-batches 50
    python scripts/loader_benchmark.py --report json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

# Add assets to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "assets")
sys.path.insert(0, _ASSETS_DIR)

from phase_loaders_template import (
    DataConfig, BasePhaseLoader,
    SNNPhaseLoader, EncoderPhaseLoader, HTMPhaseLoader,
    WorkspacePhaseLoader, ActiveInfPhaseLoader, ReasoningPhaseLoader,
    MetaPhaseLoader, PHASE_LOADERS, PHASE_NAMES,
)
from augmentation_template import AugmentationPipeline


# ---------------------------------------------------------------------------
# Benchmark result containers
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result for a single benchmark run."""
    phase: int
    phase_name: str
    batch_size: int
    num_batches: int
    num_samples: int
    # Timing
    instantiation_time_s: float = 0.0
    first_batch_time_s: float = 0.0
    total_iteration_time_s: float = 0.0
    avg_batch_time_ms: float = 0.0
    batches_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    # Memory
    peak_memory_mb: float = 0.0
    dataset_memory_mb: float = 0.0
    # Augmentation
    aug_none_batch_ms: float = 0.0
    aug_standard_batch_ms: float = 0.0
    aug_overhead_ratio: float = 0.0


@dataclass
class BenchmarkSuite:
    """Collection of all benchmark results."""
    results: List[BenchmarkResult] = field(default_factory=list)
    total_time_s: float = 0.0
    torch_version: str = ""
    device: str = "cpu"

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_time_s": self.total_time_s,
            "torch_version": self.torch_version,
            "device": self.device,
            "results": [asdict(r) for r in self.results],
        }


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def get_memory_mb() -> float:
    """Get current process memory in MB (approximate)."""
    try:
        import resource
        # maxrss is in KB on Linux, bytes on macOS
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return maxrss / (1024 * 1024)
        return maxrss / 1024
    except ImportError:
        return 0.0


def measure_tensor_memory(batch: Dict[str, Tensor]) -> float:
    """Estimate memory usage of a batch dict in MB."""
    total_bytes = 0
    for key, val in batch.items():
        if isinstance(val, Tensor):
            total_bytes += val.nelement() * val.element_size()
    return total_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def benchmark_loader(
    phase: int,
    batch_size: int = 32,
    num_samples: int = 1000,
    num_batches: int = 20,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run throughput benchmark for a single phase loader.

    Args:
        phase: Training phase (1-7).
        batch_size: Batch size to benchmark.
        num_samples: Number of synthetic samples.
        num_batches: Number of batches to iterate.
        verbose: Print per-batch timing.

    Returns:
        BenchmarkResult with timing and memory measurements.
    """
    cls = PHASE_LOADERS[phase]
    phase_name = PHASE_NAMES[phase]
    config = DataConfig(
        phase=phase,
        batch_size=batch_size,
        dev_num_samples=num_samples,
        mode="dev",
        num_workers=0,
    )

    result = BenchmarkResult(
        phase=phase,
        phase_name=phase_name,
        batch_size=batch_size,
        num_batches=num_batches,
        num_samples=num_samples,
    )

    # Measure instantiation time
    gc.collect()
    mem_before = get_memory_mb()
    t0 = time.perf_counter()
    loader = cls(config, mode="dev")
    result.instantiation_time_s = time.perf_counter() - t0
    result.dataset_memory_mb = get_memory_mb() - mem_before

    # Get train dataloader
    dl = loader.get_train_loader()

    # Measure iteration throughput
    batch_times = []
    total_samples_processed = 0
    first_batch = True

    t_total_start = time.perf_counter()
    for i, batch in enumerate(dl):
        if i >= num_batches:
            break

        t_batch_start = time.perf_counter()

        # Force materialization of all tensors
        for key, val in batch.items():
            if isinstance(val, Tensor):
                _ = val.sum()

        t_batch_end = time.perf_counter()
        batch_time = t_batch_end - t_batch_start

        if first_batch:
            result.first_batch_time_s = batch_time
            first_batch = False

        batch_times.append(batch_time)
        total_samples_processed += batch[list(batch.keys())[0]].shape[0]

        if verbose:
            print(f"    Batch {i + 1}/{num_batches}: {batch_time * 1000:.1f}ms")

    t_total_end = time.perf_counter()

    if batch_times:
        result.total_iteration_time_s = t_total_end - t_total_start
        result.avg_batch_time_ms = sum(batch_times) / len(batch_times) * 1000
        result.batches_per_sec = len(batch_times) / max(sum(batch_times), 1e-9)
        result.samples_per_sec = total_samples_processed / max(sum(batch_times), 1e-9)

    result.peak_memory_mb = get_memory_mb()

    return result


def benchmark_augmentation(
    modality: str,
    shape: Tuple[int, ...],
    num_iterations: int = 100,
    verbose: bool = False,
) -> Tuple[float, float]:
    """Benchmark augmentation overhead for a modality.

    Returns:
        Tuple of (none_ms_per_sample, standard_ms_per_sample).
    """
    if modality == "text":
        x = torch.randint(1, 256, shape, dtype=torch.long)
    else:
        x = torch.randn(*shape)

    # Benchmark "none" strength
    pipeline_none = AugmentationPipeline(modality, strength="none", seed=42)
    tfm_none = pipeline_none.get_train_transforms()
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        _ = tfm_none(x.clone())
    none_time = (time.perf_counter() - t0) / num_iterations * 1000

    # Benchmark "standard" strength
    pipeline_std = AugmentationPipeline(modality, strength="standard", seed=42)
    tfm_std = pipeline_std.get_train_transforms()
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        _ = tfm_std(x.clone())
    std_time = (time.perf_counter() - t0) / num_iterations * 1000

    if verbose:
        ratio = std_time / max(none_time, 0.001)
        print(f"  {modality}: none={none_time:.3f}ms, standard={std_time:.3f}ms, ratio={ratio:.2f}x")

    return none_time, std_time


def benchmark_augmentation_overhead(verbose: bool = False) -> Dict[str, Dict[str, float]]:
    """Benchmark augmentation overhead across all modalities."""
    modalities = {
        "vision": (3, 32, 32),
        "text": (128,),
        "audio": (64, 100),
        "sequence": (50, 16),
    }

    results = {}
    for modality, shape in modalities.items():
        none_ms, std_ms = benchmark_augmentation(modality, shape, verbose=verbose)
        results[modality] = {
            "none_ms": none_ms,
            "standard_ms": std_ms,
            "overhead_ratio": std_ms / max(none_ms, 0.001),
        }

    return results


def benchmark_multi_phase_sequential(
    batch_size: int = 32,
    num_samples: int = 500,
    verbose: bool = False,
) -> float:
    """Measure time to create and iterate one batch from all 7 phases sequentially.

    Returns:
        Total time in seconds.
    """
    t0 = time.perf_counter()
    for phase in range(1, 8):
        config = DataConfig(
            phase=phase, batch_size=batch_size,
            dev_num_samples=num_samples, mode="dev", num_workers=0,
        )
        loader = PHASE_LOADERS[phase](config, mode="dev")
        dl = loader.get_train_loader()
        batch = next(iter(dl))
        # Force materialization
        for key, val in batch.items():
            if isinstance(val, Tensor):
                _ = val.sum()
        if verbose:
            print(f"  Phase {phase} ({PHASE_NAMES[phase]}): OK")
    elapsed = time.perf_counter() - t0
    return elapsed


def benchmark_batch_sizes(
    phase: int = 1,
    batch_sizes: List[int] = None,
    num_samples: int = 2000,
    num_batches: int = 10,
    verbose: bool = False,
) -> List[BenchmarkResult]:
    """Benchmark a phase across multiple batch sizes."""
    if batch_sizes is None:
        batch_sizes = [4, 8, 16, 32, 64, 128]

    results = []
    for bs in batch_sizes:
        if bs > num_samples:
            continue
        result = benchmark_loader(phase, bs, num_samples, num_batches, verbose)
        results.append(result)
        if verbose:
            print(f"  BS={bs}: {result.samples_per_sec:.0f} samples/sec, "
                  f"{result.avg_batch_time_ms:.1f}ms/batch")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_text_report(suite: BenchmarkSuite, aug_results: Dict, seq_time: float):
    """Print a formatted text report."""
    print()
    print("=" * 72)
    print("DATA LOADER BENCHMARK REPORT")
    print("=" * 72)
    print(f"PyTorch version: {suite.torch_version}")
    print(f"Device: {suite.device}")
    print(f"Total benchmark time: {suite.total_time_s:.1f}s")
    print()

    # Per-phase results
    print("-" * 72)
    print(f"{'Phase':<25} {'Init(ms)':<10} {'Batch(ms)':<10} {'Samp/s':<10} "
          f"{'Bat/s':<8} {'Mem(MB)':<10}")
    print("-" * 72)

    for r in suite.results:
        print(f"P{r.phase} {r.phase_name:<20} "
              f"{r.instantiation_time_s * 1000:>8.1f} "
              f"{r.avg_batch_time_ms:>9.2f} "
              f"{r.samples_per_sec:>9.0f} "
              f"{r.batches_per_sec:>7.1f} "
              f"{r.peak_memory_mb:>9.1f}")

    # Augmentation overhead
    print()
    print("-" * 72)
    print("AUGMENTATION OVERHEAD")
    print("-" * 72)
    print(f"{'Modality':<15} {'None(ms)':<12} {'Standard(ms)':<15} {'Overhead':<10}")
    print("-" * 72)
    for mod, vals in aug_results.items():
        print(f"{mod:<15} {vals['none_ms']:>10.3f} {vals['standard_ms']:>13.3f} "
              f"{vals['overhead_ratio']:>8.2f}x")

    # Sequential loading
    print()
    print("-" * 72)
    print(f"Sequential all-7-phases time: {seq_time:.2f}s")
    print("-" * 72)

    # Summary
    print()
    avg_throughput = sum(r.samples_per_sec for r in suite.results) / max(len(suite.results), 1)
    max_init = max(r.instantiation_time_s for r in suite.results)
    max_batch = max(r.avg_batch_time_ms for r in suite.results)

    print("SUMMARY:")
    print(f"  Avg throughput across phases: {avg_throughput:.0f} samples/sec")
    print(f"  Max instantiation time: {max_init * 1000:.1f}ms")
    print(f"  Max avg batch time: {max_batch:.2f}ms")
    print(f"  All phases load in: {seq_time:.2f}s")

    # Pass/fail checks
    all_pass = True
    checks = []

    if seq_time > 30.0:
        checks.append(f"FAIL: Sequential loading took {seq_time:.1f}s (limit: 30s)")
        all_pass = False
    else:
        checks.append(f"PASS: Sequential loading under 30s ({seq_time:.1f}s)")

    if max_init > 5.0:
        checks.append(f"FAIL: Max init time {max_init * 1000:.0f}ms (limit: 5000ms)")
        all_pass = False
    else:
        checks.append(f"PASS: All init times under 5s (max: {max_init * 1000:.0f}ms)")

    # For augmentation overhead, we check absolute time rather than ratio because
    # modalities with identity "none" transform (like text) produce misleading ratios.
    # Threshold: standard augmentation should take < 10ms per sample.
    max_aug_abs = max(v["standard_ms"] for v in aug_results.values()) if aug_results else 0
    max_aug_modality = max(aug_results, key=lambda k: aug_results[k]["standard_ms"]) if aug_results else "N/A"
    if max_aug_abs > 10.0:
        checks.append(f"FAIL: Max aug time {max_aug_abs:.1f}ms ({max_aug_modality}, limit: 10ms/sample)")
        all_pass = False
    else:
        checks.append(f"PASS: Aug time under 10ms/sample (max: {max_aug_abs:.1f}ms, {max_aug_modality})")

    print()
    print("CHECKS:")
    for c in checks:
        print(f"  {c}")

    print()
    if all_pass:
        print("All benchmark checks PASSED.")
    else:
        print("Some benchmark checks FAILED.")

    return all_pass


def print_json_report(suite: BenchmarkSuite, aug_results: Dict, seq_time: float):
    """Print a JSON report."""
    report = suite.to_dict()
    report["augmentation_overhead"] = aug_results
    report["sequential_all_phases_s"] = seq_time
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Data loader throughput benchmarks")
    parser.add_argument("--phases", nargs="+", type=int, default=list(range(1, 8)),
                        help="Phases to benchmark (default: 1-7)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None,
                        help="Multiple batch sizes to compare")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of synthetic samples (default: 1000)")
    parser.add_argument("--num-batches", type=int, default=20,
                        help="Number of batches to iterate (default: 20)")
    parser.add_argument("--report", choices=["text", "json"], default="text",
                        help="Report format (default: text)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose per-batch output")
    parser.add_argument("--skip-aug", action="store_true",
                        help="Skip augmentation benchmark")
    args = parser.parse_args()

    suite = BenchmarkSuite()
    suite.torch_version = torch.__version__
    suite.device = "cuda" if torch.cuda.is_available() else "cpu"

    t_total_start = time.perf_counter()

    # -- Per-phase benchmarks --
    print("Running per-phase benchmarks...", file=sys.stderr)
    for phase in args.phases:
        if args.verbose:
            print(f"\nPhase {phase} ({PHASE_NAMES[phase]}):", file=sys.stderr)

        if args.batch_sizes:
            # Multi-batch-size comparison
            results = benchmark_batch_sizes(
                phase, args.batch_sizes, args.num_samples,
                args.num_batches, args.verbose
            )
            for r in results:
                suite.add(r)
        else:
            result = benchmark_loader(
                phase, args.batch_size, args.num_samples,
                args.num_batches, args.verbose
            )
            suite.add(result)
            if args.verbose:
                print(f"  {result.samples_per_sec:.0f} samples/sec, "
                      f"{result.avg_batch_time_ms:.1f}ms/batch", file=sys.stderr)

    # -- Augmentation overhead --
    aug_results: Dict[str, Dict[str, float]] = {}
    if not args.skip_aug:
        print("\nRunning augmentation benchmarks...", file=sys.stderr)
        aug_results = benchmark_augmentation_overhead(verbose=args.verbose)

    # -- Sequential all-phases --
    print("\nRunning sequential all-phases benchmark...", file=sys.stderr)
    seq_time = benchmark_multi_phase_sequential(
        args.batch_size, args.num_samples, args.verbose
    )

    suite.total_time_s = time.perf_counter() - t_total_start

    # -- Report --
    if args.report == "json":
        print_json_report(suite, aug_results, seq_time)
    else:
        all_pass = print_text_report(suite, aug_results, seq_time)
        sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
