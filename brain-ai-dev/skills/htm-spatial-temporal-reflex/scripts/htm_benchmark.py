#!/usr/bin/env python3
"""
HTM Benchmark -- Performance benchmarking for SP, TM, Reflex, and pipeline.

Usage:
    python htm_benchmark.py [--device cpu|cuda] [--quick] [--json-report PATH]

Benchmarks:
    1. Spatial Pooler throughput (steps/sec)
    2. Temporal Memory throughput (steps/sec)
    3. Reflex Memory lookup latency (us/lookup)
    4. Full pipeline (SP -> TM -> Reflex) throughput
    5. CSR segment store operations
    6. SDR utility function throughput
    7. Memory profiling (segment/synapse storage)
    8. Scaling analysis (varying column_count)
"""

from __future__ import annotations

import os
import sys
import gc
import time
import json
import argparse
import traceback
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup -- 4 levels deep:
#   brain-ai-dev / skills / htm-spatial-temporal-reflex / scripts / htm_benchmark.py
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
)
sys.path.insert(0, _PROJECT_ROOT)

import torch

# ---------------------------------------------------------------------------
# Deterministic seeds
# ---------------------------------------------------------------------------
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Graceful imports -- try upgraded (template) APIs first, fall back to legacy
# ---------------------------------------------------------------------------

# --- SDR utilities ---
_sdr_utils_available = False
try:
    from brain_ai.temporal.sdr_utils import (
        indices_to_dense as sdr_indices_to_dense,
        dense_to_indices as sdr_dense_to_indices,
        sdr_overlap as sdr_overlap_fn,
        sdr_jaccard as sdr_jaccard_fn,
        sdr_hash as sdr_hash_fn,
        random_sdr,
        SDRConfig,
    )
    _sdr_utils_available = True
    _sdr_utils_source = "brain_ai.temporal.sdr_utils"
except ImportError:
    pass

if not _sdr_utils_available:
    # Try the template asset directly
    _asset_dir = os.path.join(os.path.dirname(_SCRIPT_DIR), "assets")
    if _asset_dir not in sys.path:
        sys.path.insert(0, _asset_dir)
    try:
        from sdr_utils_template import (
            indices_to_dense as sdr_indices_to_dense,
            dense_to_indices as sdr_dense_to_indices,
            sdr_overlap as sdr_overlap_fn,
            sdr_jaccard as sdr_jaccard_fn,
            sdr_hash as sdr_hash_fn,
            random_sdr,
            SDRConfig,
        )
        _sdr_utils_available = True
        _sdr_utils_source = "sdr_utils_template (asset)"
    except ImportError:
        _sdr_utils_source = "UNAVAILABLE"

# --- Spatial Pooler (upgraded) ---
_sp_upgraded_available = False
try:
    from brain_ai.temporal.spatial_pooler import SpatialPooler as UpgradedSP, SPConfig
    _sp_upgraded_available = True
    _sp_source = "brain_ai.temporal.spatial_pooler"
except ImportError:
    pass

if not _sp_upgraded_available:
    try:
        from spatial_pooler_template import SpatialPooler as UpgradedSP, SPConfig
        _sp_upgraded_available = True
        _sp_source = "spatial_pooler_template (asset)"
    except ImportError:
        _sp_source = "UNAVAILABLE"

# --- Legacy HTM (always present in brain_ai.temporal.htm) ---
_legacy_available = False
try:
    from brain_ai.temporal.htm import (
        PytorchSpatialPooler as LegacySP,
        PytorchTemporalMemory as LegacyTM,
        ReflexMemory,
        AcceleratedHTM,
        HTMLayer,
        HTMConfig,
        create_accelerated_htm,
    )
    _legacy_available = True
    _legacy_source = "brain_ai.temporal.htm"
except ImportError:
    _legacy_source = "UNAVAILABLE"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result for a single benchmark."""
    name: str
    throughput: float         # ops/sec or steps/sec
    latency_mean_us: float    # mean latency in microseconds
    latency_p99_us: float     # p99 latency in microseconds
    memory_mb: float          # peak memory in MB
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_list: List[float], pct: float) -> float:
    """Compute percentile from a pre-sorted list."""
    if not sorted_list:
        return 0.0
    idx = int(pct * len(sorted_list))
    idx = min(idx, len(sorted_list) - 1)
    return sorted_list[idx]


def _get_process_rss_mb() -> float:
    """Get current process RSS in MB (CPU memory measurement)."""
    try:
        import resource
        # getrusage returns maxrss in KB on Linux
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024.0  # KB -> MB
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Benchmark Suite
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """Runs all HTM benchmarks and collects results."""

    def __init__(self, device: str = "cpu", quick: bool = False):
        self.device = torch.device(device)
        self.quick = quick
        self.results: List[BenchmarkResult] = []

        # Quick mode: fewer iterations for fast feedback
        self.warmup_steps = 10 if quick else 50
        self.bench_steps = 100 if quick else 1000

    def _get_memory_mb(self) -> float:
        """Get current peak memory usage in MB."""
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        else:
            return _get_process_rss_mb()

    def _reset_memory_tracking(self) -> None:
        """Reset peak memory counters."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        gc.collect()

    def _record(
        self,
        name: str,
        latencies: List[float],
        details: Optional[dict] = None,
    ) -> None:
        """Record a benchmark result from a list of latency measurements (in us)."""
        if not latencies:
            self.results.append(BenchmarkResult(
                name=name,
                throughput=0.0,
                latency_mean_us=0.0,
                latency_p99_us=0.0,
                memory_mb=self._get_memory_mb(),
                details=details or {},
            ))
            return

        mean_us = sum(latencies) / len(latencies)
        sorted_lat = sorted(latencies)
        p99_us = _percentile(sorted_lat, 0.99)
        throughput = 1e6 / mean_us if mean_us > 0 else 0.0

        self.results.append(BenchmarkResult(
            name=name,
            throughput=throughput,
            latency_mean_us=mean_us,
            latency_p99_us=p99_us,
            memory_mb=self._get_memory_mb(),
            details=details or {},
        ))

    # -----------------------------------------------------------------------
    # 1. SDR Utility Benchmarks
    # -----------------------------------------------------------------------

    def bench_sdr_utils(self) -> None:
        """Benchmark SDR utility functions: conversion, overlap, jaccard, hash."""
        if not _sdr_utils_available:
            print(f"  [SKIPPED] bench_sdr_utils -- SDR utils not available ({_sdr_utils_source})")
            self.results.append(BenchmarkResult(
                name="sdr_utils",
                throughput=0.0,
                latency_mean_us=0.0,
                latency_p99_us=0.0,
                memory_mb=0.0,
                details={"status": "SKIPPED", "reason": _sdr_utils_source},
            ))
            return

        print("  Running SDR utility benchmarks...")
        self._reset_memory_tracking()

        configs = [
            ("small", 41, 2048),
            ("large", 328, 16384),
        ]

        for tag, K, N in configs:
            B = 16
            a = random_sdr(B, K, N, device=self.device)
            b = random_sdr(B, K, N, device=self.device)

            # --- indices_to_dense ---
            for _ in range(self.warmup_steps):
                sdr_indices_to_dense(a, N)
            latencies = []
            for _ in range(self.bench_steps):
                t0 = time.perf_counter()
                sdr_indices_to_dense(a, N)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                latencies.append((time.perf_counter() - t0) * 1e6)
            self._record(f"sdr_indices_to_dense_{tag}", latencies, {"K": K, "N": N, "B": B})

            # --- dense_to_indices ---
            dense = sdr_indices_to_dense(a, N)
            for _ in range(self.warmup_steps):
                sdr_dense_to_indices(dense, K)
            latencies = []
            for _ in range(self.bench_steps):
                t0 = time.perf_counter()
                sdr_dense_to_indices(dense, K)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                latencies.append((time.perf_counter() - t0) * 1e6)
            self._record(f"sdr_dense_to_indices_{tag}", latencies, {"K": K, "N": N, "B": B})

            # --- sdr_overlap ---
            for _ in range(self.warmup_steps):
                sdr_overlap_fn(a, b)
            latencies = []
            for _ in range(self.bench_steps):
                t0 = time.perf_counter()
                sdr_overlap_fn(a, b)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                latencies.append((time.perf_counter() - t0) * 1e6)
            self._record(f"sdr_overlap_{tag}", latencies, {"K": K, "N": N, "B": B})

            # --- sdr_jaccard ---
            for _ in range(self.warmup_steps):
                sdr_jaccard_fn(a, b)
            latencies = []
            for _ in range(self.bench_steps):
                t0 = time.perf_counter()
                sdr_jaccard_fn(a, b)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                latencies.append((time.perf_counter() - t0) * 1e6)
            self._record(f"sdr_jaccard_{tag}", latencies, {"K": K, "N": N, "B": B})

            # --- sdr_hash ---
            for _ in range(self.warmup_steps):
                sdr_hash_fn(a)
            latencies = []
            for _ in range(self.bench_steps):
                t0 = time.perf_counter()
                sdr_hash_fn(a)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                latencies.append((time.perf_counter() - t0) * 1e6)
            self._record(f"sdr_hash_{tag}", latencies, {"K": K, "N": N, "B": B})

    # -----------------------------------------------------------------------
    # 2. Spatial Pooler Benchmarks
    # -----------------------------------------------------------------------

    def bench_spatial_pooler(self) -> None:
        """Benchmark SP forward pass throughput (upgraded and legacy)."""
        print("  Running Spatial Pooler benchmarks...")
        self._reset_memory_tracking()

        # --- Upgraded SP (template-based) ---
        if _sp_upgraded_available:
            configs = [
                ("small", SPConfig(input_size=512, column_count=256, potential_pool_size=128, binarization_topk=128)),
                ("medium", SPConfig(input_size=4096, column_count=2048, potential_pool_size=256, binarization_topk=200)),
            ]
            if not self.quick:
                configs.append(
                    ("large", SPConfig(input_size=4096, column_count=16384, potential_pool_size=256, binarization_topk=200))
                )

            for label, config in configs:
                try:
                    torch.manual_seed(42)
                    sp = UpgradedSP(config).to(self.device)
                    x = torch.randn(1, config.input_size, device=self.device)

                    # Warmup
                    for _ in range(self.warmup_steps):
                        sp(x, learn=False)

                    # Benchmark (no learning)
                    latencies = []
                    for _ in range(self.bench_steps):
                        t0 = time.perf_counter()
                        sp(x, learn=False)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        latencies.append((time.perf_counter() - t0) * 1e6)
                    self._record(
                        f"sp_upgraded_{label}_nolearn",
                        latencies,
                        {"column_count": config.column_count, "source": _sp_source},
                    )

                    # Benchmark (with learning)
                    latencies = []
                    for _ in range(self.bench_steps):
                        t0 = time.perf_counter()
                        sp(x, learn=True)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        latencies.append((time.perf_counter() - t0) * 1e6)
                    self._record(
                        f"sp_upgraded_{label}_learn",
                        latencies,
                        {"column_count": config.column_count, "source": _sp_source},
                    )

                    del sp
                    gc.collect()
                except Exception as e:
                    print(f"    [ERROR] sp_upgraded_{label}: {e}")
                    traceback.print_exc()
        else:
            print(f"    [SKIPPED] Upgraded SP -- not available ({_sp_source})")

        # --- Legacy SP ---
        if _legacy_available:
            legacy_configs = [
                ("small", 512, 256),
                ("medium", 4096, 2048),
            ]
            for label, input_size, col_count in legacy_configs:
                try:
                    torch.manual_seed(42)
                    sp = LegacySP(
                        input_size=input_size,
                        column_count=col_count,
                        sparsity=0.02,
                    ).to(self.device)
                    x = torch.randn(input_size, device=self.device)

                    for _ in range(self.warmup_steps):
                        sp(x, learn=False)

                    latencies = []
                    for _ in range(self.bench_steps):
                        t0 = time.perf_counter()
                        sp(x, learn=False)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        latencies.append((time.perf_counter() - t0) * 1e6)
                    self._record(
                        f"sp_legacy_{label}_nolearn",
                        latencies,
                        {"column_count": col_count, "source": _legacy_source},
                    )

                    del sp
                    gc.collect()
                except Exception as e:
                    print(f"    [ERROR] sp_legacy_{label}: {e}")
                    traceback.print_exc()
        else:
            print(f"    [SKIPPED] Legacy SP -- not available ({_legacy_source})")

    # -----------------------------------------------------------------------
    # 3. Temporal Memory Benchmarks
    # -----------------------------------------------------------------------

    def bench_temporal_memory(self) -> None:
        """Benchmark TM step with and without learning."""
        if not _legacy_available:
            print(f"  [SKIPPED] bench_temporal_memory -- legacy HTM not available ({_legacy_source})")
            self.results.append(BenchmarkResult(
                name="tm_legacy",
                throughput=0.0, latency_mean_us=0.0, latency_p99_us=0.0, memory_mb=0.0,
                details={"status": "SKIPPED", "reason": _legacy_source},
            ))
            return

        print("  Running Temporal Memory benchmarks...")
        self._reset_memory_tracking()

        configs = [
            ("small", 256, 8),
            ("medium", 1024, 16),
        ]

        for label, col_count, cells in configs:
            try:
                torch.manual_seed(42)
                tm = LegacyTM(
                    column_count=col_count,
                    cells_per_column=cells,
                    activation_threshold=6,
                    min_threshold=4,
                    max_new_synapse_count=10,
                ).to(self.device)

                sparsity = 0.02
                num_active = max(1, int(col_count * sparsity))

                # --- No learning ---
                latencies = []
                for step in range(self.warmup_steps + self.bench_steps):
                    # Generate a sparse active-columns vector
                    active = torch.zeros(col_count, device=self.device)
                    indices = torch.randperm(col_count, device=self.device)[:num_active]
                    active[indices] = 1.0

                    if step >= self.warmup_steps:
                        t0 = time.perf_counter()
                        tm(active, learn=False)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        latencies.append((time.perf_counter() - t0) * 1e6)
                    else:
                        tm(active, learn=False)

                self._record(
                    f"tm_{label}_nolearn",
                    latencies,
                    {"column_count": col_count, "cells_per_column": cells},
                )

                # --- With learning ---
                tm.reset()
                latencies = []
                for step in range(self.warmup_steps + self.bench_steps):
                    active = torch.zeros(col_count, device=self.device)
                    indices = torch.randperm(col_count, device=self.device)[:num_active]
                    active[indices] = 1.0

                    if step >= self.warmup_steps:
                        t0 = time.perf_counter()
                        tm(active, learn=True)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        latencies.append((time.perf_counter() - t0) * 1e6)
                    else:
                        tm(active, learn=True)

                self._record(
                    f"tm_{label}_learn",
                    latencies,
                    {"column_count": col_count, "cells_per_column": cells},
                )

                # Report segment stats
                stats = tm.get_memory_stats()
                self.results[-1].details.update(stats)

                del tm
                gc.collect()
            except Exception as e:
                print(f"    [ERROR] tm_{label}: {e}")
                traceback.print_exc()

    # -----------------------------------------------------------------------
    # 4. Reflex Memory Benchmarks
    # -----------------------------------------------------------------------

    def bench_reflex_memory(self) -> None:
        """Benchmark Reflex Memory lookup (hit vs miss) and store latency."""
        if not _legacy_available:
            print(f"  [SKIPPED] bench_reflex_memory -- legacy HTM not available ({_legacy_source})")
            self.results.append(BenchmarkResult(
                name="reflex",
                throughput=0.0, latency_mean_us=0.0, latency_p99_us=0.0, memory_mb=0.0,
                details={"status": "SKIPPED", "reason": _legacy_source},
            ))
            return

        print("  Running Reflex Memory benchmarks...")
        self._reset_memory_tracking()

        pattern_dim = 512
        max_patterns = 5000
        num_seed_patterns = 500

        try:
            torch.manual_seed(42)
            rm = ReflexMemory(
                pattern_dim=pattern_dim,
                max_patterns=max_patterns,
                promotion_threshold=3,
                similarity_threshold=0.85,
            ).to(self.device)

            # Seed the reflex memory with patterns
            seed_patterns = torch.randn(num_seed_patterns, pattern_dim, device=self.device)
            seed_predictions = torch.randn(num_seed_patterns, pattern_dim, device=self.device)
            for i in range(num_seed_patterns):
                rm.store(seed_patterns[i], seed_predictions[i], force=True)

            # --- Lookup HIT latency ---
            # Use patterns that were stored (should be hits)
            hit_latencies = []
            for _ in range(self.warmup_steps):
                idx = torch.randint(0, num_seed_patterns, (1,)).item()
                rm.lookup(seed_patterns[idx])

            for _ in range(self.bench_steps):
                idx = torch.randint(0, num_seed_patterns, (1,)).item()
                t0 = time.perf_counter()
                result = rm.lookup(seed_patterns[idx])
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                hit_latencies.append((time.perf_counter() - t0) * 1e6)

            self._record(
                "reflex_lookup_hit",
                hit_latencies,
                {"num_stored": rm.num_stored.item(), "pattern_dim": pattern_dim},
            )

            # --- Lookup MISS latency ---
            # Use completely random patterns (should be misses)
            miss_latencies = []
            for _ in range(self.bench_steps):
                novel = torch.randn(pattern_dim, device=self.device)
                t0 = time.perf_counter()
                rm.lookup(novel)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                miss_latencies.append((time.perf_counter() - t0) * 1e6)

            self._record(
                "reflex_lookup_miss",
                miss_latencies,
                {"num_stored": rm.num_stored.item(), "pattern_dim": pattern_dim},
            )

            # --- Store (observe) throughput ---
            store_latencies = []
            for _ in range(self.bench_steps):
                p = torch.randn(pattern_dim, device=self.device)
                pred = torch.randn(pattern_dim, device=self.device)
                t0 = time.perf_counter()
                rm.store(p, pred, force=True)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                store_latencies.append((time.perf_counter() - t0) * 1e6)

            self._record(
                "reflex_store",
                store_latencies,
                {"num_stored": rm.num_stored.item(), "pattern_dim": pattern_dim},
            )

            rm_stats = rm.get_statistics()
            self.results[-1].details.update({
                "hit_rate": rm_stats.get("hit_rate", 0.0),
                "utilization": rm_stats.get("utilization", 0.0),
            })

            del rm
            gc.collect()
        except Exception as e:
            print(f"    [ERROR] bench_reflex_memory: {e}")
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # 5. Full Pipeline Benchmarks
    # -----------------------------------------------------------------------

    def bench_full_pipeline(self) -> None:
        """Benchmark full SP -> TM -> Reflex end-to-end pipeline."""
        if not _legacy_available:
            print(f"  [SKIPPED] bench_full_pipeline -- legacy HTM not available ({_legacy_source})")
            self.results.append(BenchmarkResult(
                name="pipeline",
                throughput=0.0, latency_mean_us=0.0, latency_p99_us=0.0, memory_mb=0.0,
                details={"status": "SKIPPED", "reason": _legacy_source},
            ))
            return

        print("  Running full pipeline benchmarks...")
        self._reset_memory_tracking()

        input_size = 512
        col_count = 1024
        cells = 16
        steps = self.bench_steps

        try:
            # --- Pipeline WITHOUT Reflex (plain HTM) ---
            torch.manual_seed(42)
            htm_layer = HTMLayer(
                HTMConfig(
                    input_size=input_size,
                    column_count=col_count,
                    cells_per_column=cells,
                    sparsity=0.02,
                ),
                use_htm_core=False,
            ).to(self.device)

            x_seq = torch.randn(self.warmup_steps + steps, input_size, device=self.device)

            for i in range(self.warmup_steps):
                htm_layer(x_seq[i], learn=True)

            latencies_htm = []
            for i in range(steps):
                t0 = time.perf_counter()
                htm_layer(x_seq[self.warmup_steps + i], learn=True)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                latencies_htm.append((time.perf_counter() - t0) * 1e6)

            self._record(
                "pipeline_htm_only",
                latencies_htm,
                {"input_size": input_size, "column_count": col_count},
            )

            # --- Pipeline WITH Reflex (AHTM) ---
            torch.manual_seed(42)
            ahtm = create_accelerated_htm(
                input_size=input_size,
                column_count=col_count,
                cells_per_column=cells,
                sparsity=0.02,
                max_reflex_patterns=5000,
                promotion_threshold=3,
            ).to(self.device)

            # Re-generate fresh sequence
            x_seq = torch.randn(self.warmup_steps + steps, input_size, device=self.device)

            for i in range(self.warmup_steps):
                ahtm(x_seq[i], learn=True)

            latencies_ahtm = []
            reflex_hits = 0
            for i in range(steps):
                t0 = time.perf_counter()
                result = ahtm(x_seq[self.warmup_steps + i], learn=True)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                latencies_ahtm.append((time.perf_counter() - t0) * 1e6)
                if result.get("from_reflex") is not None:
                    fr = result["from_reflex"]
                    if isinstance(fr, torch.Tensor):
                        if fr.item():
                            reflex_hits += 1
                    elif fr:
                        reflex_hits += 1

            self._record(
                "pipeline_ahtm",
                latencies_ahtm,
                {
                    "input_size": input_size,
                    "column_count": col_count,
                    "reflex_hits": reflex_hits,
                    "reflex_hit_rate": reflex_hits / max(steps, 1),
                },
            )

            # Compute speedup
            mean_htm = sum(latencies_htm) / len(latencies_htm) if latencies_htm else 1.0
            mean_ahtm = sum(latencies_ahtm) / len(latencies_ahtm) if latencies_ahtm else 1.0
            speedup = mean_htm / mean_ahtm if mean_ahtm > 0 else 0.0
            self.results[-1].details["speedup_vs_htm"] = round(speedup, 3)

            del htm_layer, ahtm
            gc.collect()
        except Exception as e:
            print(f"    [ERROR] bench_full_pipeline: {e}")
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # 6. CSR / Segment Store Operations
    # -----------------------------------------------------------------------

    def bench_csr_operations(self) -> None:
        """Benchmark segment store operations: create, match, add/remove."""
        if not _legacy_available:
            print(f"  [SKIPPED] bench_csr_operations -- legacy HTM not available ({_legacy_source})")
            self.results.append(BenchmarkResult(
                name="csr_ops",
                throughput=0.0, latency_mean_us=0.0, latency_p99_us=0.0, memory_mb=0.0,
                details={"status": "SKIPPED", "reason": _legacy_source},
            ))
            return

        print("  Running CSR / segment store benchmarks...")
        self._reset_memory_tracking()

        col_count = 1024
        cells = 16
        num_cells = col_count * cells

        try:
            torch.manual_seed(42)
            tm = LegacyTM(
                column_count=col_count,
                cells_per_column=cells,
                activation_threshold=6,
                min_threshold=4,
                max_new_synapse_count=10,
            ).to(self.device)

            # Seed TM with some segments by running a sequence
            sparsity = 0.02
            num_active = max(1, int(col_count * sparsity))
            seed_steps = min(200, self.bench_steps)

            for _ in range(seed_steps):
                active = torch.zeros(col_count, device=self.device)
                indices = torch.randperm(col_count, device=self.device)[:num_active]
                active[indices] = 1.0
                tm(active, learn=True)

            stats_before = tm.get_memory_stats()

            # --- Segment matching (get_best_matching_segment) ---
            # Build an active cell set from the last state
            active_cell_set = set(tm._active_indices) if tm._active_indices else set(range(min(50, num_cells)))

            # Pick cells that have segments
            cells_with_segs = list(tm.segments.keys())[:min(100, len(tm.segments))]

            if cells_with_segs:
                # Warmup
                for _ in range(self.warmup_steps):
                    for cid in cells_with_segs[:5]:
                        tm._get_best_matching_segment(cid, active_cell_set)

                latencies = []
                for _ in range(self.bench_steps):
                    cid = cells_with_segs[_ % len(cells_with_segs)]
                    t0 = time.perf_counter()
                    tm._get_best_matching_segment(cid, active_cell_set)
                    latencies.append((time.perf_counter() - t0) * 1e6)

                self._record(
                    "csr_segment_match",
                    latencies,
                    {
                        "cells_with_segments": len(cells_with_segs),
                        "total_segments": stats_before.get("total_segments", 0),
                        "total_synapses": stats_before.get("total_synapses", 0),
                    },
                )
            else:
                self._record("csr_segment_match", [], {"note": "no segments created"})

            # --- Segment creation overhead (new segment on a fresh cell) ---
            import random as _random
            _random.seed(42)

            latencies = []
            for i in range(min(self.bench_steps, 500)):
                cell_id = num_cells + i  # guaranteed fresh cell
                prev_active = list(range(i, i + 10))  # dummy previous active
                t0 = time.perf_counter()
                if cell_id not in tm.segments:
                    tm.segments[cell_id] = []
                new_seg = {pc: tm.initial_permanence for pc in prev_active}
                tm.segments[cell_id].append(new_seg)
                latencies.append((time.perf_counter() - t0) * 1e6)

            self._record(
                "csr_segment_create",
                latencies,
                {"segments_created": len(latencies)},
            )

            stats_after = tm.get_memory_stats()
            self.results[-1].details.update({
                "total_segments_after": stats_after.get("total_segments", 0),
                "total_synapses_after": stats_after.get("total_synapses", 0),
            })

            del tm
            gc.collect()
        except Exception as e:
            print(f"    [ERROR] bench_csr_operations: {e}")
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # 7. Scaling Analysis
    # -----------------------------------------------------------------------

    def bench_scaling(self) -> None:
        """Vary column_count and measure SP and TM throughput at each scale."""
        print("  Running scaling analysis...")

        scales = [256, 1024, 2048, 8192]
        if not self.quick:
            scales.append(16384)

        # --- SP scaling (use upgraded if available, else legacy) ---
        if _sp_upgraded_available:
            for col_count in scales:
                try:
                    self._reset_memory_tracking()
                    torch.manual_seed(42)

                    input_size = min(4096, col_count * 2)
                    pool_size = min(256, input_size)
                    topk = min(200, input_size)

                    config = SPConfig(
                        input_size=input_size,
                        column_count=col_count,
                        potential_pool_size=pool_size,
                        binarization_topk=topk,
                    )
                    sp = UpgradedSP(config).to(self.device)
                    x = torch.randn(1, input_size, device=self.device)

                    for _ in range(self.warmup_steps):
                        sp(x, learn=False)

                    scale_steps = min(self.bench_steps, 200)
                    latencies = []
                    for _ in range(scale_steps):
                        t0 = time.perf_counter()
                        sp(x, learn=False)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        latencies.append((time.perf_counter() - t0) * 1e6)

                    self._record(
                        f"scaling_sp_cols{col_count}",
                        latencies,
                        {"column_count": col_count, "input_size": input_size},
                    )

                    del sp
                    gc.collect()
                except Exception as e:
                    print(f"    [ERROR] scaling_sp_cols{col_count}: {e}")
                    traceback.print_exc()

        elif _legacy_available:
            for col_count in scales:
                try:
                    self._reset_memory_tracking()
                    torch.manual_seed(42)

                    input_size = min(4096, col_count * 2)
                    sp = LegacySP(
                        input_size=input_size,
                        column_count=col_count,
                        sparsity=0.02,
                    ).to(self.device)
                    x = torch.randn(input_size, device=self.device)

                    for _ in range(self.warmup_steps):
                        sp(x, learn=False)

                    scale_steps = min(self.bench_steps, 200)
                    latencies = []
                    for _ in range(scale_steps):
                        t0 = time.perf_counter()
                        sp(x, learn=False)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        latencies.append((time.perf_counter() - t0) * 1e6)

                    self._record(
                        f"scaling_sp_cols{col_count}",
                        latencies,
                        {"column_count": col_count, "input_size": input_size, "api": "legacy"},
                    )

                    del sp
                    gc.collect()
                except Exception as e:
                    print(f"    [ERROR] scaling_sp_cols{col_count}: {e}")
                    traceback.print_exc()
        else:
            print("    [SKIPPED] SP scaling -- no SP available")

        # --- TM scaling ---
        if _legacy_available:
            for col_count in scales:
                if col_count > 8192:
                    # TM at very large scale is extremely slow; skip
                    continue
                try:
                    self._reset_memory_tracking()
                    torch.manual_seed(42)

                    cells = 8
                    tm = LegacyTM(
                        column_count=col_count,
                        cells_per_column=cells,
                        activation_threshold=6,
                        min_threshold=4,
                        max_new_synapse_count=10,
                    ).to(self.device)

                    sparsity = 0.02
                    num_active = max(1, int(col_count * sparsity))
                    scale_steps = min(self.bench_steps, 100)

                    for _ in range(self.warmup_steps):
                        active = torch.zeros(col_count, device=self.device)
                        indices = torch.randperm(col_count, device=self.device)[:num_active]
                        active[indices] = 1.0
                        tm(active, learn=True)

                    latencies = []
                    for _ in range(scale_steps):
                        active = torch.zeros(col_count, device=self.device)
                        indices = torch.randperm(col_count, device=self.device)[:num_active]
                        active[indices] = 1.0
                        t0 = time.perf_counter()
                        tm(active, learn=True)
                        latencies.append((time.perf_counter() - t0) * 1e6)

                    self._record(
                        f"scaling_tm_cols{col_count}",
                        latencies,
                        {"column_count": col_count, "cells_per_column": cells},
                    )

                    del tm
                    gc.collect()
                except Exception as e:
                    print(f"    [ERROR] scaling_tm_cols{col_count}: {e}")
                    traceback.print_exc()
        else:
            print("    [SKIPPED] TM scaling -- not available")

    # -----------------------------------------------------------------------
    # 8. Memory Profiling
    # -----------------------------------------------------------------------

    def bench_memory_profile(self) -> None:
        """Profile TM memory growth over N steps."""
        if not _legacy_available:
            print(f"  [SKIPPED] bench_memory_profile -- legacy HTM not available ({_legacy_source})")
            self.results.append(BenchmarkResult(
                name="memory_profile",
                throughput=0.0, latency_mean_us=0.0, latency_p99_us=0.0, memory_mb=0.0,
                details={"status": "SKIPPED", "reason": _legacy_source},
            ))
            return

        print("  Running memory profiling...")
        self._reset_memory_tracking()

        col_count = 1024
        cells = 16
        profile_steps = 200 if self.quick else 1000

        try:
            torch.manual_seed(42)
            tm = LegacyTM(
                column_count=col_count,
                cells_per_column=cells,
                activation_threshold=6,
                min_threshold=4,
                max_new_synapse_count=10,
                max_segments_per_cell=128,
                max_synapses_per_segment=32,
            ).to(self.device)

            sparsity = 0.02
            num_active = max(1, int(col_count * sparsity))

            memory_snapshots = []
            segment_snapshots = []

            for step in range(profile_steps):
                active = torch.zeros(col_count, device=self.device)
                indices = torch.randperm(col_count, device=self.device)[:num_active]
                active[indices] = 1.0
                tm(active, learn=True)

                # Snapshot every 10% of steps
                if step % max(1, profile_steps // 10) == 0 or step == profile_steps - 1:
                    stats = tm.get_memory_stats()
                    mem_mb = self._get_memory_mb()
                    memory_snapshots.append({
                        "step": step,
                        "memory_mb": round(mem_mb, 2),
                        "total_segments": stats["total_segments"],
                        "total_synapses": stats["total_synapses"],
                        "cells_with_segments": stats["cells_with_segments"],
                    })
                    segment_snapshots.append(stats["total_segments"])

            final_stats = tm.get_memory_stats()

            # Estimate raw memory usage from segment/synapse data
            # Each synapse entry: int key + float value ~ 12 bytes in Python dict
            # Each segment: dict overhead ~ 64 bytes + per-synapse
            est_synapse_bytes = final_stats["total_synapses"] * 12
            est_segment_bytes = final_stats["total_segments"] * 64
            est_total_bytes = est_synapse_bytes + est_segment_bytes

            self.results.append(BenchmarkResult(
                name="memory_profile",
                throughput=profile_steps / (sum(1 for _ in range(1))),  # N/A
                latency_mean_us=0.0,
                latency_p99_us=0.0,
                memory_mb=self._get_memory_mb(),
                details={
                    "profile_steps": profile_steps,
                    "final_segments": final_stats["total_segments"],
                    "final_synapses": final_stats["total_synapses"],
                    "cells_with_segments": final_stats["cells_with_segments"],
                    "avg_synapses_per_segment": round(final_stats["avg_synapses_per_segment"], 2),
                    "est_segment_store_kb": round(est_total_bytes / 1024, 2),
                    "max_segments_per_cell_cap": 128,
                    "max_synapses_per_segment_cap": 32,
                    "snapshots": memory_snapshots,
                    "segment_growth": segment_snapshots,
                    "bounded": (
                        final_stats["total_segments"]
                        <= col_count * cells * 128  # max_segments_per_cell
                    ),
                },
            ))

            del tm
            gc.collect()
        except Exception as e:
            print(f"    [ERROR] bench_memory_profile: {e}")
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def report(self) -> str:
        """Generate formatted benchmark report."""
        lines = [
            "",
            "=" * 72,
            "  HTM Benchmark Report",
            "=" * 72,
            f"  Device: {self.device}",
            f"  Quick mode: {self.quick}",
            f"  Warmup steps: {self.warmup_steps}",
            f"  Bench steps: {self.bench_steps}",
            f"  SDR utils source: {_sdr_utils_source}",
            f"  SP source: {_sp_source}",
            f"  Legacy HTM source: {_legacy_source}",
            "-" * 72,
        ]

        for r in self.results:
            status = r.details.get("status", "")
            if status == "SKIPPED":
                lines.append(f"\n  {r.name}: SKIPPED ({r.details.get('reason', '')})")
                continue

            lines.append(f"\n  {r.name}:")
            if r.throughput > 0:
                lines.append(f"    Throughput:      {r.throughput:>12.1f} steps/sec")
            if r.latency_mean_us > 0:
                lines.append(f"    Latency (mean):  {r.latency_mean_us:>12.1f} us")
            if r.latency_p99_us > 0:
                lines.append(f"    Latency (p99):   {r.latency_p99_us:>12.1f} us")
            if r.memory_mb > 0:
                lines.append(f"    Memory (peak):   {r.memory_mb:>12.1f} MB")

            # Print select details (skip large nested structures)
            for k, v in r.details.items():
                if k in ("status", "reason", "snapshots", "segment_growth"):
                    continue
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.4f}")
                else:
                    lines.append(f"    {k}: {v}")

        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_json(self) -> dict:
        """JSON-serializable report."""
        return {
            "device": str(self.device),
            "quick": self.quick,
            "warmup_steps": self.warmup_steps,
            "bench_steps": self.bench_steps,
            "sources": {
                "sdr_utils": _sdr_utils_source,
                "spatial_pooler": _sp_source,
                "legacy_htm": _legacy_source,
            },
            "benchmarks": [asdict(r) for r in self.results],
        }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTM Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run benchmarks on (default: cpu)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer iterations for fast feedback",
    )
    parser.add_argument(
        "--json-report",
        type=str,
        default=None,
        help="Path to save JSON report",
    )
    args = parser.parse_args()

    # Validate CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    print("=" * 72)
    print("  HTM Performance Benchmark")
    print("=" * 72)
    print(f"  Device:        {args.device}")
    print(f"  Quick mode:    {args.quick}")
    print(f"  PyTorch:       {torch.__version__}")
    print(f"  SDR utils:     {_sdr_utils_source}")
    print(f"  Spatial Pooler:{_sp_source}")
    print(f"  Legacy HTM:    {_legacy_source}")
    print("-" * 72)

    suite = BenchmarkSuite(device=args.device, quick=args.quick)

    # Run all benchmarks, each wrapped in try/except
    benchmarks = [
        ("SDR Utilities", suite.bench_sdr_utils),
        ("Spatial Pooler", suite.bench_spatial_pooler),
        ("Temporal Memory", suite.bench_temporal_memory),
        ("Reflex Memory", suite.bench_reflex_memory),
        ("Full Pipeline", suite.bench_full_pipeline),
        ("CSR Operations", suite.bench_csr_operations),
        ("Memory Profile", suite.bench_memory_profile),
    ]

    if not args.quick:
        # Insert scaling before memory profile
        benchmarks.insert(-1, ("Scaling Analysis", suite.bench_scaling))

    for bench_name, bench_fn in benchmarks:
        print(f"\n[{bench_name}]")
        try:
            bench_fn()
        except Exception as e:
            print(f"  [FATAL] {bench_name} crashed: {e}")
            traceback.print_exc()

    # Print report
    print(suite.report())

    # Save JSON report if requested
    if args.json_report:
        report_data = suite.to_json()
        report_dir = os.path.dirname(os.path.abspath(args.json_report))
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir, exist_ok=True)
        with open(args.json_report, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"\nJSON report saved to {args.json_report}")

    # Summary
    total = len(suite.results)
    skipped = sum(1 for r in suite.results if r.details.get("status") == "SKIPPED")
    ran = total - skipped
    print(f"\nBenchmarks: {ran} ran, {skipped} skipped, {total} total")


if __name__ == "__main__":
    main()
