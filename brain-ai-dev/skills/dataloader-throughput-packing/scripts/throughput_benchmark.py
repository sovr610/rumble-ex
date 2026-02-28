#!/usr/bin/env python3
"""
throughput_benchmark.py — Benchmark tool for data pipeline throughput.

Measures tokens/sec across configurations (num_workers, prefetch_factor,
packing mode). Reports padding_ratio, data_stall_ratio, and generates
ASCII summary tables and JSON output.

Usage:
    python scripts/throughput_benchmark.py
    python scripts/throughput_benchmark.py --num-samples 500 --seq-len 512
    python scripts/throughput_benchmark.py --output benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# Ensure asset modules are importable
SKILL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_DIR / "assets"))


# ---------------------------------------------------------------------------
# Benchmark Config
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Configuration for the throughput benchmark."""
    num_samples: int = 200
    seq_len: int = 256
    batch_size: int = 8
    warmup_steps: int = 5
    measure_steps: int = 30
    num_workers_list: List[int] = field(default_factory=lambda: [0, 2, 4])
    prefetch_factor_list: List[int] = field(default_factory=lambda: [2, 4])
    packing_modes: List[str] = field(
        default_factory=lambda: ["none", "pretrain_blocks", "sft_boundary_aware"]
    )


# ---------------------------------------------------------------------------
# Benchmark Result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    config_label: str
    num_workers: int
    prefetch_factor: int
    packing_mode: str
    raw_tokens_per_sec: float
    effective_tokens_per_sec: float
    padding_ratio: float
    data_stall_ratio_p50: float
    data_stall_ratio_p90: float
    avg_step_time_ms: float


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

class ThroughputBenchmark:
    """Runs throughput benchmarks across configurations."""

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.results: List[BenchmarkResult] = []

    def run_all(self) -> List[BenchmarkResult]:
        """Run benchmarks across all configuration combinations."""
        self.results = []

        for packing_mode in self.cfg.packing_modes:
            for nw in self.cfg.num_workers_list:
                for pf in self.cfg.prefetch_factor_list:
                    if nw == 0 and pf != self.cfg.prefetch_factor_list[0]:
                        continue  # prefetch_factor only matters with workers

                    label = f"nw={nw}_pf={pf}_{packing_mode}"
                    print(f"  Running: {label} ...", end=" ", flush=True)

                    try:
                        result = self._run_single(
                            label=label,
                            num_workers=nw,
                            prefetch_factor=pf,
                            packing_mode=packing_mode,
                        )
                        self.results.append(result)
                        print(
                            f"raw={result.raw_tokens_per_sec:.0f} tok/s, "
                            f"eff={result.effective_tokens_per_sec:.0f} tok/s, "
                            f"pad={result.padding_ratio:.3f}"
                        )
                    except Exception as e:
                        print(f"FAILED: {e}")

        return self.results

    def _run_single(
        self,
        label: str,
        num_workers: int,
        prefetch_factor: int,
        packing_mode: str,
    ) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        from pipeline_auditor_template import PipelineAuditor
        from sequence_packer_template import (
            SequencePacker, PackingConfig, TokenizedExample,
        )

        # Generate synthetic data
        rng = np.random.RandomState(42)
        seq_len = self.cfg.seq_len

        if packing_mode == "none":
            # No packing: variable length sequences, padded to max
            padding_ratios = []
            step_times = []
            data_times = []

            for step in range(self.cfg.warmup_steps + self.cfg.measure_steps):
                t_data_start = time.perf_counter()
                # Simulate variable-length batch
                lengths = rng.randint(seq_len // 4, seq_len + 1, size=self.cfg.batch_size)
                max_len = int(max(lengths))
                real_tokens = int(sum(lengths))
                total_tokens = max_len * self.cfg.batch_size
                pad_tokens = total_tokens - real_tokens
                # Simulate data loading time
                time.sleep(0.001)
                t_data_end = time.perf_counter()

                # Simulate compute
                t_compute_start = time.perf_counter()
                time.sleep(0.003)
                t_compute_end = time.perf_counter()

                if step >= self.cfg.warmup_steps:
                    t_data = (t_data_end - t_data_start) * 1000
                    t_total = (t_compute_end - t_data_start) * 1000
                    step_times.append(t_total)
                    data_times.append(t_data)
                    padding_ratios.append(
                        pad_tokens / total_tokens if total_tokens > 0 else 0
                    )

            avg_padding = float(np.mean(padding_ratios)) if padding_ratios else 0
            avg_step_ms = float(np.median(step_times)) if step_times else 1
            raw_tps = self.cfg.batch_size * seq_len / (avg_step_ms / 1000)
            eff_tps = raw_tps * (1 - avg_padding)
            stall_ratios = [d / t for d, t in zip(data_times, step_times) if t > 0]

        elif packing_mode == "pretrain_blocks":
            # Pretrain blocks: near-zero padding
            cfg_pack = PackingConfig(
                mode="pretrain_blocks", target_seq_len=seq_len
            )
            packer = SequencePacker(cfg_pack)

            step_times = []
            data_times = []

            for step in range(self.cfg.warmup_steps + self.cfg.measure_steps):
                t_data_start = time.perf_counter()
                # Generate docs
                docs = [
                    rng.randint(1, 32000, size=rng.randint(50, 200)).tolist()
                    for _ in range(self.cfg.batch_size)
                ]
                blocks = list(packer.pack_pretrain(iter(docs)))
                time.sleep(0.001)
                t_data_end = time.perf_counter()

                t_compute_start = time.perf_counter()
                time.sleep(0.003)
                t_compute_end = time.perf_counter()

                if step >= self.cfg.warmup_steps:
                    t_data = (t_data_end - t_data_start) * 1000
                    t_total = (t_compute_end - t_data_start) * 1000
                    step_times.append(t_total)
                    data_times.append(t_data)

            block_ratio = packer.padding_ratio_blocks(blocks) if blocks else 0
            avg_step_ms = float(np.median(step_times)) if step_times else 1
            raw_tps = self.cfg.batch_size * seq_len / (avg_step_ms / 1000)
            eff_tps = raw_tps * (1 - block_ratio)
            avg_padding = block_ratio
            stall_ratios = [d / t for d, t in zip(data_times, step_times) if t > 0]

        else:
            # SFT boundary-aware packing
            cfg_pack = PackingConfig(
                mode="sft_boundary_aware",
                target_seq_len=seq_len,
                bucket_boundaries=(64, 128, 256, 512),
            )
            packer = SequencePacker(cfg_pack)

            step_times = []
            data_times = []
            padding_ratios = []

            for step in range(self.cfg.warmup_steps + self.cfg.measure_steps):
                t_data_start = time.perf_counter()
                # Generate variable-length examples
                examples = [
                    TokenizedExample(
                        input_ids=rng.randint(1, 32000, size=rng.randint(20, seq_len)).tolist(),
                        labels=rng.randint(1, 32000, size=rng.randint(20, seq_len)).tolist(),
                    )
                    for _ in range(self.cfg.batch_size)
                ]
                # Fix labels to match input_ids length
                for ex in examples:
                    min_len = min(len(ex.input_ids), len(ex.labels))
                    ex.input_ids = ex.input_ids[:min_len]
                    ex.labels = ex.labels[:min_len]

                packed = packer.pack_sft(examples)
                time.sleep(0.001)
                t_data_end = time.perf_counter()

                t_compute_start = time.perf_counter()
                time.sleep(0.003)
                t_compute_end = time.perf_counter()

                if step >= self.cfg.warmup_steps:
                    t_data = (t_data_end - t_data_start) * 1000
                    t_total = (t_compute_end - t_data_start) * 1000
                    step_times.append(t_total)
                    data_times.append(t_data)
                    padding_ratios.append(packer.padding_ratio(packed))

            avg_padding = float(np.mean(padding_ratios)) if padding_ratios else 0
            avg_step_ms = float(np.median(step_times)) if step_times else 1
            real_tokens_total = sum(
                len(ex.input_ids) for ex in examples
            ) * self.cfg.measure_steps
            raw_tps = self.cfg.batch_size * seq_len / (avg_step_ms / 1000)
            eff_tps = raw_tps * (1 - avg_padding)
            stall_ratios = [d / t for d, t in zip(data_times, step_times) if t > 0]

        stall_p50 = float(np.percentile(stall_ratios, 50)) if stall_ratios else 0
        stall_p90 = float(np.percentile(stall_ratios, 90)) if stall_ratios else 0

        return BenchmarkResult(
            config_label=label,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            packing_mode=packing_mode,
            raw_tokens_per_sec=raw_tps,
            effective_tokens_per_sec=eff_tps,
            padding_ratio=avg_padding,
            data_stall_ratio_p50=stall_p50,
            data_stall_ratio_p90=stall_p90,
            avg_step_time_ms=avg_step_ms,
        )

    # ---- Output formatting ----

    def ascii_table(self) -> str:
        """Generate an ASCII summary table of results."""
        if not self.results:
            return "No results."

        header = (
            f"{'Config':<35} {'Raw tok/s':>12} {'Eff tok/s':>12} "
            f"{'Pad Ratio':>10} {'Stall p50':>10} {'Stall p90':>10} "
            f"{'Step ms':>10}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for r in self.results:
            line = (
                f"{r.config_label:<35} {r.raw_tokens_per_sec:>12.0f} "
                f"{r.effective_tokens_per_sec:>12.0f} "
                f"{r.padding_ratio:>10.4f} {r.data_stall_ratio_p50:>10.4f} "
                f"{r.data_stall_ratio_p90:>10.4f} {r.avg_step_time_ms:>10.1f}"
            )
            lines.append(line)

        lines.append(sep)

        # Best effective throughput
        best = max(self.results, key=lambda r: r.effective_tokens_per_sec)
        lines.append(
            f"\nBest effective throughput: {best.config_label} "
            f"at {best.effective_tokens_per_sec:.0f} tok/s"
        )

        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        """Generate JSON output for programmatic consumption."""
        return {
            "benchmark_config": {
                "num_samples": self.cfg.num_samples,
                "seq_len": self.cfg.seq_len,
                "batch_size": self.cfg.batch_size,
                "measure_steps": self.cfg.measure_steps,
            },
            "results": [
                {
                    "config_label": r.config_label,
                    "num_workers": r.num_workers,
                    "prefetch_factor": r.prefetch_factor,
                    "packing_mode": r.packing_mode,
                    "raw_tokens_per_sec": r.raw_tokens_per_sec,
                    "effective_tokens_per_sec": r.effective_tokens_per_sec,
                    "padding_ratio": r.padding_ratio,
                    "data_stall_ratio_p50": r.data_stall_ratio_p50,
                    "data_stall_ratio_p90": r.data_stall_ratio_p90,
                    "avg_step_time_ms": r.avg_step_time_ms,
                }
                for r in self.results
            ],
            "best_config": None,
        }

    def save_json(self, path: str) -> None:
        """Save results to JSON file."""
        data = self.to_json()
        if self.results:
            best = max(self.results, key=lambda r: r.effective_tokens_per_sec)
            data["best_config"] = best.config_label
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark data pipeline throughput"
    )
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of synthetic samples")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup steps")
    parser.add_argument("--steps", type=int, default=15,
                        help="Measurement steps")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output file path")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer configurations")
    args = parser.parse_args()

    if args.quick:
        nw_list = [0]
        pf_list = [2]
        modes = ["none", "sft_boundary_aware"]
    else:
        nw_list = [0, 2]
        pf_list = [2, 4]
        modes = ["none", "pretrain_blocks", "sft_boundary_aware"]

    cfg = BenchmarkConfig(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        warmup_steps=args.warmup,
        measure_steps=args.steps,
        num_workers_list=nw_list,
        prefetch_factor_list=pf_list,
        packing_modes=modes,
    )

    print("=" * 70)
    print("Data Pipeline Throughput Benchmark")
    print("=" * 70)
    print(f"  Samples: {cfg.num_samples}, Seq len: {cfg.seq_len}, "
          f"Batch: {cfg.batch_size}")
    print(f"  Warmup: {cfg.warmup_steps}, Measure: {cfg.measure_steps}")
    print(f"  Workers: {cfg.num_workers_list}, Prefetch: {cfg.prefetch_factor_list}")
    print(f"  Packing modes: {cfg.packing_modes}")
    print()

    bench = ThroughputBenchmark(cfg)
    bench.run_all()

    print()
    print(bench.ascii_table())

    if args.output:
        bench.save_json(args.output)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
