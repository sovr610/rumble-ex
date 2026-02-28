"""
distributed_benchmark.py — Throughput and scaling benchmarks for the
distributed-scaling skill.

Measures:
    1. Forward/backward throughput at various batch sizes
    2. Gradient accumulation overhead vs single-batch
    3. Effective batch size configurations (dev, 1B, 3B, 7B)
    4. Checkpoint save/load latency
    5. Sampler partitioning performance at scale
    6. LR scaling correctness across configurations
    7. Simulated scaling efficiency (1, 2, 4, 8 "ranks")

All benchmarks run on CPU in single-process (no GPU required).
"""

from __future__ import annotations

import copy
import logging
import math
import os
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# Add parent assets directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "assets")
sys.path.insert(0, ASSETS_DIR)

from ddp_wrapper_template import DDPWrapper, ProcessGroupManager
from fsdp_wrapper_template import FSDPWrapper, ShardingPolicy
from gradient_accumulator_template import (
    GradientAccumulator,
    compute_scaled_lr,
    get_cosine_lr,
)
from distributed_launcher_template import LaunchConfig, DistributedLauncher
from rank_aware_loader_template import (
    RankAwareDataLoader,
    SamplerVerifier,
)
from distributed_config_template import DistributedConfig

logger = logging.getLogger(__name__)

try:
    from torch.utils.data.distributed import DistributedSampler
    _DIST_SAMPLER_AVAILABLE = True
except ImportError:
    _DIST_SAMPLER_AVAILABLE = False


# ===========================================================================
# Benchmark infrastructure
# ===========================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    elapsed_seconds: float
    throughput: float  # items/sec or ops/sec
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        tp_str = f"{self.throughput:.1f}"
        t_str = f"{self.elapsed_seconds * 1000:.1f}ms"
        detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
        if detail_str:
            return f"  {self.name}: {tp_str} items/s ({t_str}) [{detail_str}]"
        return f"  {self.name}: {tp_str} items/s ({t_str})"


class BenchmarkSuite:
    """Collects and runs benchmarks."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult) -> None:
        self.results.append(result)
        print(str(result))

    def summary(self) -> Dict[str, Any]:
        return {
            "total_benchmarks": len(self.results),
            "results": [
                {"name": r.name, "elapsed": r.elapsed_seconds,
                 "throughput": r.throughput, **r.details}
                for r in self.results
            ],
        }


def _make_model(in_dim=32, hidden=128, out_dim=10):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def _make_large_model(in_dim=256, hidden=512, out_dim=64, n_layers=4):
    """A larger model for more realistic benchmarking."""
    layers = []
    layers.append(nn.Linear(in_dim, hidden))
    layers.append(nn.ReLU())
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


# ===========================================================================
# Benchmark 1: Forward/Backward throughput
# ===========================================================================

def bench_forward_backward(suite: BenchmarkSuite) -> None:
    """Measure forward+backward throughput at various batch sizes."""
    print("\n--- Benchmark 1: Forward/Backward Throughput ---")

    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for batch_size in [8, 16, 32, 64, 128]:
        n_steps = 50
        data = torch.randn(batch_size, 32)
        target = torch.randn(batch_size, 10)

        # Warmup
        for _ in range(5):
            out = model(data)
            loss = nn.functional.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        start = time.perf_counter()
        for _ in range(n_steps):
            out = model(data)
            loss = nn.functional.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        elapsed = time.perf_counter() - start

        throughput = n_steps * batch_size / elapsed
        suite.add_result(BenchmarkResult(
            name=f"fwd+bwd batch={batch_size}",
            elapsed_seconds=elapsed,
            throughput=throughput,
            details={"batch_size": batch_size, "n_steps": n_steps},
        ))


# ===========================================================================
# Benchmark 2: Gradient accumulation overhead
# ===========================================================================

def bench_gradient_accumulation(suite: BenchmarkSuite) -> None:
    """Compare single-batch vs accumulated gradient training."""
    print("\n--- Benchmark 2: Gradient Accumulation Overhead ---")

    model_ref = _make_model()
    batch_size = 32
    n_optimizer_steps = 20

    for accum_steps in [1, 4, 8, 16]:
        model = copy.deepcopy(model_ref)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        acc = GradientAccumulator(
            accumulation_steps=accum_steps, max_grad_norm=1.0,
        )

        micro_batch = batch_size // accum_steps if accum_steps <= batch_size else 1
        total_micro_steps = n_optimizer_steps * accum_steps

        # Warmup
        for _ in range(accum_steps):
            x = torch.randn(micro_batch, 32)
            out = model(x)
            loss = out.sum()
            acc.step(loss, optimizer, model)
        acc.reset()

        model2 = copy.deepcopy(model_ref)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        acc2 = GradientAccumulator(
            accumulation_steps=accum_steps, max_grad_norm=1.0,
        )

        start = time.perf_counter()
        for _ in range(total_micro_steps):
            x = torch.randn(micro_batch, 32)
            out = model2(x)
            loss = out.sum()
            acc2.step(loss, optimizer2, model2)
        elapsed = time.perf_counter() - start

        effective_samples = n_optimizer_steps * batch_size
        throughput = effective_samples / elapsed

        suite.add_result(BenchmarkResult(
            name=f"accum={accum_steps} micro={micro_batch}",
            elapsed_seconds=elapsed,
            throughput=throughput,
            details={
                "accum_steps": accum_steps,
                "micro_batch": micro_batch,
                "optimizer_steps": acc2.optimizer_steps,
            },
        ))


# ===========================================================================
# Benchmark 3: Effective batch size configurations
# ===========================================================================

def bench_effective_batch_configs(suite: BenchmarkSuite) -> None:
    """Verify effective batch size for all preset configurations."""
    print("\n--- Benchmark 3: Effective Batch Size Configurations ---")

    configs = [
        ("dev", DistributedConfig.for_dev()),
        ("1B (4 GPU)", DistributedConfig.for_1b(num_gpus=4)),
        ("3B (8 GPU)", DistributedConfig.for_3b(num_gpus=8)),
        ("7B (8 GPU)", DistributedConfig.for_7b(num_gpus=8)),
        ("7B constrained", DistributedConfig.for_7b_constrained(num_gpus=8)),
        ("7B multi-node", DistributedConfig.for_multi_node_7b(num_nodes=2)),
    ]

    for name, cfg in configs:
        start = time.perf_counter()
        eff = cfg.effective_batch_size
        scaled_lr = cfg.scaled_learning_rate
        scaled_warmup = cfg.scaled_warmup_steps
        is_valid = cfg.is_valid()
        elapsed = time.perf_counter() - start

        suite.add_result(BenchmarkResult(
            name=f"config: {name}",
            elapsed_seconds=elapsed,
            throughput=0,
            details={
                "effective_batch": eff,
                "scaled_lr": f"{scaled_lr:.6f}",
                "warmup": scaled_warmup,
                "valid": is_valid,
                "strategy": cfg.strategy,
                "world_size": cfg.world_size,
            },
        ))


# ===========================================================================
# Benchmark 4: Checkpoint save/load latency
# ===========================================================================

def bench_checkpoint_latency(suite: BenchmarkSuite) -> None:
    """Measure checkpoint save and load times."""
    print("\n--- Benchmark 4: Checkpoint Save/Load Latency ---")

    for model_name, model_factory in [
        ("small (32-128-10)", lambda: _make_model(32, 128, 10)),
        ("medium (256-512-64)", lambda: _make_large_model(256, 512, 64, 4)),
    ]:
        model = model_factory()
        param_count = sum(p.numel() for p in model.parameters())
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        # Save benchmark
        n_trials = 10
        start = time.perf_counter()
        for _ in range(n_trials):
            wrapper.save_checkpoint(path, epoch=1, step=100)
        save_elapsed = (time.perf_counter() - start) / n_trials

        # Load benchmark
        start = time.perf_counter()
        for _ in range(n_trials):
            wrapper.load_checkpoint(path)
        load_elapsed = (time.perf_counter() - start) / n_trials

        suite.add_result(BenchmarkResult(
            name=f"ckpt save {model_name}",
            elapsed_seconds=save_elapsed,
            throughput=param_count / save_elapsed,
            details={"params": param_count, "file_size_mb": os.path.getsize(path) / 1e6},
        ))

        suite.add_result(BenchmarkResult(
            name=f"ckpt load {model_name}",
            elapsed_seconds=load_elapsed,
            throughput=param_count / load_elapsed,
            details={"params": param_count},
        ))

        os.unlink(path)
        wrapper.cleanup()


# ===========================================================================
# Benchmark 5: Sampler partitioning performance
# ===========================================================================

def bench_sampler_performance(suite: BenchmarkSuite) -> None:
    """Measure sampler creation and verification at various scales."""
    print("\n--- Benchmark 5: Sampler Partitioning Performance ---")

    if not _DIST_SAMPLER_AVAILABLE:
        print("  SKIP: DistributedSampler not available")
        return

    for dataset_size in [1000, 10000, 100000]:
        for world_size in [4, 8]:
            dataset = list(range(dataset_size))

            # Creation + iteration benchmark
            start = time.perf_counter()
            for rank in range(world_size):
                sampler = DistributedSampler(
                    dataset, num_replicas=world_size, rank=rank,
                    shuffle=False, drop_last=True,
                )
                _ = list(sampler)
            elapsed = time.perf_counter() - start

            suite.add_result(BenchmarkResult(
                name=f"sampler size={dataset_size} ws={world_size}",
                elapsed_seconds=elapsed,
                throughput=dataset_size * world_size / elapsed,
                details={"dataset_size": dataset_size, "world_size": world_size},
            ))

    # Verification benchmark
    for dataset_size in [1000, 10000]:
        start = time.perf_counter()
        valid, dups = SamplerVerifier.verify_no_duplicates(
            dataset_size, world_size=8, drop_last=True,
        )
        elapsed = time.perf_counter() - start

        suite.add_result(BenchmarkResult(
            name=f"verify no-dups size={dataset_size}",
            elapsed_seconds=elapsed,
            throughput=dataset_size / elapsed,
            details={"valid": valid, "n_dups": len(dups)},
        ))


# ===========================================================================
# Benchmark 6: LR scaling correctness
# ===========================================================================

def bench_lr_scaling(suite: BenchmarkSuite) -> None:
    """Verify LR scaling math across configurations."""
    print("\n--- Benchmark 6: LR Scaling Correctness ---")

    base_lr = 3e-4
    ref_batch = 256

    test_cases = [
        # (effective_batch, mode, expected_ratio)
        (256, "linear", 1.0),
        (512, "linear", 2.0),
        (1024, "linear", 4.0),
        (256, "sqrt", 1.0),
        (512, "sqrt", math.sqrt(2.0)),
        (1024, "sqrt", 2.0),
    ]

    errors = 0
    start = time.perf_counter()
    for eff_batch, mode, expected_ratio in test_cases:
        lr = compute_scaled_lr(base_lr, eff_batch, ref_batch, mode=mode)
        expected = base_lr * expected_ratio
        if abs(lr - expected) > 1e-10:
            errors += 1
            logger.error(
                f"LR mismatch: eff={eff_batch} mode={mode} "
                f"got {lr} expected {expected}"
            )
    elapsed = time.perf_counter() - start

    suite.add_result(BenchmarkResult(
        name="LR scaling correctness",
        elapsed_seconds=elapsed,
        throughput=len(test_cases) / elapsed,
        details={"test_cases": len(test_cases), "errors": errors},
    ))

    # Cosine schedule sanity
    start = time.perf_counter()
    lrs = []
    for step in range(1001):
        lr = get_cosine_lr(step, warmup_steps=100, total_steps=1000,
                          peak_lr=1.0, min_lr=0.1)
        lrs.append(lr)
    elapsed = time.perf_counter() - start

    # Verify monotonic warmup
    warmup_monotonic = all(lrs[i] <= lrs[i + 1] for i in range(99))
    # Verify peak
    peak_correct = abs(lrs[100] - 1.0) < 1e-6
    # Verify end
    end_correct = abs(lrs[1000] - 0.1) < 1e-6

    suite.add_result(BenchmarkResult(
        name="cosine schedule 1000 steps",
        elapsed_seconds=elapsed,
        throughput=1001 / elapsed,
        details={
            "warmup_monotonic": warmup_monotonic,
            "peak_correct": peak_correct,
            "end_correct": end_correct,
        },
    ))


# ===========================================================================
# Benchmark 7: Simulated scaling efficiency
# ===========================================================================

def bench_scaling_efficiency(suite: BenchmarkSuite) -> None:
    """Simulate scaling efficiency by measuring per-rank processing time.

    Since we cannot do real multi-GPU, we simulate by measuring single-process
    throughput and projecting how it would scale with DDP overhead.
    """
    print("\n--- Benchmark 7: Simulated Scaling Efficiency ---")

    model = _make_model(32, 128, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch_size = 32
    n_steps = 100

    # Baseline: single process
    data = torch.randn(batch_size, 32)
    target = torch.randn(batch_size, 10)

    # Warmup
    for _ in range(10):
        out = model(data)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    start = time.perf_counter()
    for _ in range(n_steps):
        out = model(data)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    baseline_time = time.perf_counter() - start
    baseline_throughput = n_steps * batch_size / baseline_time

    suite.add_result(BenchmarkResult(
        name="baseline (1 GPU simulated)",
        elapsed_seconds=baseline_time,
        throughput=baseline_throughput,
        details={"n_steps": n_steps, "batch_size": batch_size},
    ))

    # Simulated scaling with typical DDP overhead factors
    # These are empirical overhead estimates from the reference docs
    overhead_factors = {
        1: 1.0,
        2: 0.96,
        4: 0.93,
        8: 0.89,
    }

    for world_size, efficiency in overhead_factors.items():
        projected_throughput = baseline_throughput * world_size * efficiency
        projected_time = (n_steps * batch_size * world_size) / projected_throughput

        suite.add_result(BenchmarkResult(
            name=f"projected {world_size} GPU",
            elapsed_seconds=projected_time,
            throughput=projected_throughput,
            details={
                "world_size": world_size,
                "scaling_efficiency": f"{efficiency * 100:.0f}%",
                "speedup": f"{world_size * efficiency:.2f}x",
            },
        ))

    # Gradient accumulation scaling
    for accum_steps in [1, 4, 8, 16]:
        model2 = copy.deepcopy(model)
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        acc = GradientAccumulator(accumulation_steps=accum_steps, max_grad_norm=1.0)

        micro = max(batch_size // accum_steps, 1)
        n_micro = n_steps * accum_steps

        micro_data = torch.randn(micro, 32)
        micro_target = torch.randn(micro, 10)

        start = time.perf_counter()
        for _ in range(n_micro):
            out = model2(micro_data)
            loss = nn.functional.mse_loss(out, micro_target)
            acc.step(loss, opt2, model2)
        elapsed = time.perf_counter() - start

        effective_samples = acc.optimizer_steps * batch_size
        throughput = effective_samples / elapsed

        suite.add_result(BenchmarkResult(
            name=f"accum_scaling accum={accum_steps}",
            elapsed_seconds=elapsed,
            throughput=throughput,
            details={
                "accum_steps": accum_steps,
                "micro_batch": micro,
                "optimizer_steps": acc.optimizer_steps,
                "effective_batch": batch_size,
            },
        ))


# ===========================================================================
# Benchmark 8: Config validation performance
# ===========================================================================

def bench_config_validation(suite: BenchmarkSuite) -> None:
    """Measure config creation and validation performance."""
    print("\n--- Benchmark 8: Config Validation Performance ---")

    presets = [
        ("dev", DistributedConfig.for_dev),
        ("1b", lambda: DistributedConfig.for_1b(4)),
        ("3b", lambda: DistributedConfig.for_3b(8)),
        ("7b", lambda: DistributedConfig.for_7b(8)),
        ("7b_constrained", lambda: DistributedConfig.for_7b_constrained(8)),
        ("multi_node", lambda: DistributedConfig.for_multi_node_7b(2, 8)),
    ]

    n_iterations = 1000

    for name, factory in presets:
        start = time.perf_counter()
        for _ in range(n_iterations):
            cfg = factory()
            _ = cfg.validate()
            _ = cfg.effective_batch_size
            _ = cfg.scaled_learning_rate
        elapsed = time.perf_counter() - start

        suite.add_result(BenchmarkResult(
            name=f"config {name}",
            elapsed_seconds=elapsed,
            throughput=n_iterations / elapsed,
            details={"preset": name, "iterations": n_iterations},
        ))


# ===========================================================================
# Benchmark 9: Memory estimation
# ===========================================================================

def bench_memory_estimation(suite: BenchmarkSuite) -> None:
    """Benchmark memory estimation across model scales."""
    print("\n--- Benchmark 9: Memory Estimation ---")

    scales = [
        ("1M (minimal)", 1_000_000, DistributedConfig.for_dev()),
        ("1B", 1_000_000_000, DistributedConfig.for_1b(4)),
        ("3B", 3_000_000_000, DistributedConfig.for_3b(8)),
        ("7B", 7_000_000_000, DistributedConfig.for_7b(8)),
        ("7B constrained", 7_000_000_000, DistributedConfig.for_7b_constrained(8)),
    ]

    for name, param_count, cfg in scales:
        start = time.perf_counter()
        mem = cfg.estimate_memory_gb(param_count)
        elapsed = time.perf_counter() - start

        suite.add_result(BenchmarkResult(
            name=f"mem_est {name}",
            elapsed_seconds=elapsed,
            throughput=1.0 / max(elapsed, 1e-9),
            details={
                "params": param_count,
                "total_gb": mem["total_gb"],
                "params_gb": mem["params_gb"],
                "strategy": cfg.strategy,
            },
        ))


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("Distributed Scaling — Throughput & Scaling Benchmarks")
    print("=" * 70)
    print(f"Device: CPU (single-process simulation)")
    print(f"PyTorch: {torch.__version__}")
    print()

    suite = BenchmarkSuite()

    bench_forward_backward(suite)
    bench_gradient_accumulation(suite)
    bench_effective_batch_configs(suite)
    bench_checkpoint_latency(suite)
    bench_sampler_performance(suite)
    bench_lr_scaling(suite)
    bench_scaling_efficiency(suite)
    bench_config_validation(suite)
    bench_memory_estimation(suite)

    # Final summary
    summary = suite.summary()
    print()
    print("=" * 70)
    print(f"BENCHMARK SUMMARY: {summary['total_benchmarks']} benchmarks completed")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
