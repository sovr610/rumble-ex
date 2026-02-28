"""
attention_benchmark.py
=======================
Backend-vs-backend attention throughput and memory comparison.

For each combination of (backend x head_dim x seq_len x dtype):
  - Warmup steps, then timed steps with CUDA synchronization
  - Record: tokens/sec, step_time_ms, peak_memory_mb, kernel_time_ms

Generates a comparison table (text) and optionally JSON output.

Usage:
    python attention_benchmark.py
    python attention_benchmark.py --head_dims 64,128 --seq_lens 512,2048
    python attention_benchmark.py --backends auto,flash,efficient,math
    python attention_benchmark.py --output results.json --table results.txt
    python attention_benchmark.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Ensure assets directory is on path
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
sys.path.insert(0, os.path.abspath(_ASSETS_DIR))

from sdpa_attention_template import BackendConfig, reset_log_state, sdpa_attention

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    """Configuration for a single benchmark run."""
    backend: str
    head_dim: int
    seq_len: int
    dtype: torch.dtype
    batch_size: int = 4
    num_heads: int = 16
    warmup_steps: int = 3
    timed_steps: int = 10


@dataclass
class BenchResult:
    """Result from one benchmark combination."""
    backend: str
    head_dim: int
    seq_len: int
    dtype: str
    batch_size: int
    num_heads: int

    # Timing
    tokens_per_sec: float = 0.0
    step_time_ms: float = 0.0
    kernel_time_ms: float = 0.0

    # Memory
    peak_memory_mb: float = 0.0

    # Status
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "head_dim": self.head_dim,
            "seq_len": self.seq_len,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "num_heads": self.num_heads,
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "step_time_ms": round(self.step_time_ms, 3),
            "kernel_time_ms": round(self.kernel_time_ms, 3),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Backend skip check
# ---------------------------------------------------------------------------


def _should_skip_backend(
    backend: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[bool, str]:
    """
    Return (should_skip, reason) for a given backend/dtype/device combination.
    This avoids crashes by pre-checking known incompatibilities.
    """
    if device.type == "cpu":
        if backend in ("flash", "efficient", "cudnn"):
            return True, f"{backend} requires CUDA device; running on CPU"

    if backend in ("flash", "efficient") and dtype == torch.float32 and device.type == "cuda":
        # Flash/efficient may fall back to Math for float32 -- not an error,
        # but we warn the user. Don't skip; let it run with Math fallback.
        pass

    if backend == "flash":
        if dtype == torch.float32 and device.type == "cuda":
            # Flash cannot run float32; BackendConfig includes Math fallback
            # so it still runs, but report a note
            pass

    return False, ""


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------


def _run_single_bench(cfg: BenchConfig, device: torch.device) -> BenchResult:
    """Execute one benchmark configuration and return the result."""
    result = BenchResult(
        backend=cfg.backend,
        head_dim=cfg.head_dim,
        seq_len=cfg.seq_len,
        dtype=str(cfg.dtype).replace("torch.", ""),
        batch_size=cfg.batch_size,
        num_heads=cfg.num_heads,
    )

    # Pre-check skip conditions
    should_skip, reason = _should_skip_backend(cfg.backend, cfg.dtype, device)
    if should_skip:
        result.skipped = True
        result.skip_reason = reason
        return result

    # Build backend config
    # Use force=False so we always get output (Math fallback if fused not available)
    backend_config = BackendConfig(policy=cfg.backend, force=False, log=False)

    # Build tensors
    shape = (cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim)
    try:
        q = torch.randn(*shape, dtype=cfg.dtype, device=device)
        k = torch.randn(*shape, dtype=cfg.dtype, device=device)
        v = torch.randn(*shape, dtype=cfg.dtype, device=device)
    except RuntimeError as e:
        result.error = f"Tensor allocation failed: {e}"
        return result

    # Warmup
    try:
        for _ in range(cfg.warmup_steps):
            reset_log_state()
            out = sdpa_attention(q, k, v, is_causal=True, training=False,
                                  backend_cfg=backend_config)
            if device.type == "cuda":
                torch.cuda.synchronize()
    except Exception as e:
        result.error = f"Warmup failed: {e}"
        return result

    # Reset peak memory counter before timed runs
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # CUDA event timing setup
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    # Timed steps
    total_kernel_ms = 0.0
    wall_start = time.perf_counter()

    try:
        for _ in range(cfg.timed_steps):
            if device.type == "cuda":
                start_event.record()

            reset_log_state()
            out = sdpa_attention(q, k, v, is_causal=True, training=False,
                                  backend_cfg=backend_config)

            if device.type == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                total_kernel_ms += start_event.elapsed_time(end_event)
    except Exception as e:
        result.error = f"Timed step failed: {e}"
        return result

    wall_end = time.perf_counter()

    # Compute metrics
    total_wall_ms = (wall_end - wall_start) * 1000.0
    step_time_ms = total_wall_ms / cfg.timed_steps
    tokens_per_step = cfg.batch_size * cfg.seq_len
    tokens_per_sec = (tokens_per_step * cfg.timed_steps) / (wall_end - wall_start)

    result.step_time_ms = step_time_ms
    result.tokens_per_sec = tokens_per_sec

    if device.type == "cuda":
        result.kernel_time_ms = total_kernel_ms / cfg.timed_steps
        result.peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        result.kernel_time_ms = step_time_ms  # same as wall on CPU
        result.peak_memory_mb = 0.0  # not tracked on CPU

    return result


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def _format_table(results: List[BenchResult], math_baseline: Dict[str, BenchResult]) -> str:
    """
    Format results as a text comparison table.
    math_baseline maps (head_dim, seq_len, dtype) -> BenchResult for speedup calc.
    """
    lines = []
    lines.append("=" * 90)
    lines.append(
        f"{'Backend':<14} {'dtype':<10} {'seq_len':<8} {'head_dim':<10} "
        f"{'tok/sec':>10} {'step_ms':>9} {'mem_MB':>8} {'speedup':>8}"
    )
    lines.append("-" * 90)

    for r in results:
        if r.skipped:
            lines.append(
                f"{'  ' + r.backend:<14} {'--':<10} {r.seq_len:<8} {r.head_dim:<10} "
                f"{'SKIP':>10}  ({r.skip_reason[:40]})"
            )
            continue
        if r.error:
            lines.append(
                f"{'  ' + r.backend:<14} {r.dtype:<10} {r.seq_len:<8} {r.head_dim:<10} "
                f"{'ERROR':>10}  ({r.error[:40]})"
            )
            continue

        # Compute speedup vs Math baseline
        key = (r.head_dim, r.seq_len, r.dtype)
        speedup_str = "1.00x"
        if key in math_baseline and r.backend != "math":
            baseline_tok = math_baseline[key].tokens_per_sec
            if baseline_tok > 0:
                speedup = r.tokens_per_sec / baseline_tok
                speedup_str = f"{speedup:.2f}x"

        lines.append(
            f"{'  ' + r.backend:<14} {r.dtype:<10} {r.seq_len:<8} {r.head_dim:<10} "
            f"{r.tokens_per_sec:>10,.0f} {r.step_time_ms:>8.2f}  "
            f"{r.peak_memory_mb:>7.1f} {speedup_str:>8}"
        )

    lines.append("=" * 90)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    backends: List[str],
    head_dims: List[int],
    seq_lens: List[int],
    dtypes: List[torch.dtype],
    batch_size: int = 4,
    num_heads: int = 16,
    warmup_steps: int = 3,
    timed_steps: int = 10,
    device: Optional[torch.device] = None,
    output_json: Optional[str] = None,
    output_table: Optional[str] = None,
) -> List[BenchResult]:
    """
    Run all combinations and return results.

    Results are printed to stdout as a table. Optionally saved as JSON.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Attention Benchmark")
    print(f"Device  : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        sm = props.major * 10 + props.minor
        print(f"GPU     : {props.name} (SM{sm})")
    print(f"Backends: {backends}")
    print(f"Dtypes  : {[str(d) for d in dtypes]}")
    print(f"head_dims: {head_dims}")
    print(f"seq_lens: {seq_lens}")
    print(f"Warmup  : {warmup_steps} steps | Timed: {timed_steps} steps")
    print()

    all_results: List[BenchResult] = []
    total_combos = len(backends) * len(head_dims) * len(seq_lens) * len(dtypes)
    done = 0

    for dtype in dtypes:
        for seq_len in seq_lens:
            for head_dim in head_dims:
                for backend in backends:
                    cfg = BenchConfig(
                        backend=backend,
                        head_dim=head_dim,
                        seq_len=seq_len,
                        dtype=dtype,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        warmup_steps=warmup_steps,
                        timed_steps=timed_steps,
                    )
                    done += 1
                    dtype_name = str(dtype).replace("torch.", "")
                    print(
                        f"[{done}/{total_combos}] "
                        f"backend={backend:12s} seq={seq_len:5d} "
                        f"head_dim={head_dim:4d} dtype={dtype_name}",
                        end=" ... ",
                        flush=True,
                    )
                    result = _run_single_bench(cfg, device)
                    all_results.append(result)

                    if result.skipped:
                        print(f"SKIP: {result.skip_reason[:50]}")
                    elif result.error:
                        print(f"ERROR: {result.error[:50]}")
                    else:
                        print(
                            f"{result.tokens_per_sec:,.0f} tok/s  "
                            f"{result.step_time_ms:.2f}ms  "
                            f"{result.peak_memory_mb:.0f}MB"
                        )

    # Build Math baseline for speedup calculation
    math_baseline: Dict[tuple, BenchResult] = {}
    for r in all_results:
        if r.backend == "math" and not r.skipped and not r.error:
            key = (r.head_dim, r.seq_len, r.dtype)
            math_baseline[key] = r

    # Print table
    print()
    table = _format_table(all_results, math_baseline)
    print(table)

    # Save outputs
    if output_json:
        data = {
            "device": str(device),
            "results": [r.to_dict() for r in all_results],
        }
        with open(output_json, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"\nJSON results saved to: {output_json}")

    if output_table:
        with open(output_table, "w") as fh:
            fh.write(table + "\n")
        print(f"Table saved to: {output_table}")

    return all_results


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark attention backends (Flash vs Efficient vs Math etc.)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--backends", default="auto,flash,efficient,math",
        help="Comma-separated backend names to test."
    )
    p.add_argument(
        "--head_dims", default="64,128",
        help="Comma-separated head dimensions to test."
    )
    p.add_argument(
        "--seq_lens", default="256,512",
        help="Comma-separated sequence lengths to test."
    )
    p.add_argument(
        "--dtypes", default="float32",
        help="Comma-separated dtypes: float32, float16, bfloat16."
    )
    p.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for all benchmark runs."
    )
    p.add_argument(
        "--num_heads", type=int, default=16,
        help="Number of attention heads."
    )
    p.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup steps per config."
    )
    p.add_argument(
        "--steps", type=int, default=10,
        help="Number of timed steps per config."
    )
    p.add_argument(
        "--device", default=None,
        help="Device string (e.g. 'cuda:0', 'cpu'). Defaults to cuda:0 if available."
    )
    p.add_argument(
        "--output", default=None, metavar="JSON_FILE",
        help="Save results as JSON to this file."
    )
    p.add_argument(
        "--table", default=None, metavar="TEXT_FILE",
        help="Save comparison table to this file."
    )
    return p.parse_args(argv)


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def main(argv=None) -> None:
    args = _parse_args(argv)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    head_dims = [int(h) for h in args.head_dims.split(",") if h.strip()]
    seq_lens = [int(s) for s in args.seq_lens.split(",") if s.strip()]
    dtype_names = [d.strip() for d in args.dtypes.split(",") if d.strip()]

    dtypes = []
    for name in dtype_names:
        if name not in DTYPE_MAP:
            print(f"ERROR: Unknown dtype {name!r}. Valid: {list(DTYPE_MAP.keys())}")
            sys.exit(1)
        dtypes.append(DTYPE_MAP[name])

    # Deduplicate dtypes (e.g. float32 and fp32 are the same)
    seen_dtypes = []
    for d in dtypes:
        if d not in seen_dtypes:
            seen_dtypes.append(d)
    dtypes = seen_dtypes

    device = torch.device(args.device) if args.device else None

    run_benchmark(
        backends=backends,
        head_dims=head_dims,
        seq_lens=seq_lens,
        dtypes=dtypes,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        warmup_steps=args.warmup,
        timed_steps=args.steps,
        device=device,
        output_json=args.output,
        output_table=args.table,
    )


if __name__ == "__main__":
    main()
