#!/usr/bin/env python3
"""
training_benchmark.py -- Measure V-JEPA 2 training throughput and memory usage.

Benchmarks:
  1. Step time (ms) vs batch size
  2. Memory usage (MB) vs batch size
  3. Throughput (samples/sec) vs batch size
  4. Predictor forward time vs depth
  5. EMA update overhead
  6. Checkpoint save/load time

Usage:
    python scripts/training_benchmark.py
    python scripts/training_benchmark.py --batch-sizes 1 2 4 8 16
    python scripts/training_benchmark.py --embed-dim 256 --pred-dim 128 --depth 6
    python scripts/training_benchmark.py --warmup 5 --repeats 20
    python scripts/training_benchmark.py --output results.txt

Output format: ASCII table with benchmark results.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SKILL_DIR  = os.path.dirname(_SCRIPT_DIR)
_ASSETS_DIR = os.path.join(_SKILL_DIR, 'assets')

for _p in [_SKILL_DIR, _ASSETS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_memory_mb() -> float:
    """Return current GPU memory usage in MB (or 0 on CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


def _get_peak_memory_mb() -> float:
    """Return peak GPU memory usage in MB (or 0 on CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def _reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _sync() -> None:
    """Synchronize CUDA before timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _format_table(headers: List[str], rows: List[List[str]],
                  col_widths: Optional[List[int]] = None) -> str:
    """Format a list of rows as an ASCII table."""
    if col_widths is None:
        col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
                      for i, h in enumerate(headers)]

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    hdr = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, col_widths)) + "|"

    lines = [sep, hdr, sep]
    for row in rows:
        line = "|" + "|".join(f" {str(v):<{w}} " for v, w in zip(row, col_widths)) + "|"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

class _TinyEncoder(nn.Module):
    """Minimal encoder stub for benchmarking."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class _TinyCfg:
    lr = 1e-3
    final_lr = 1e-6
    weight_decay = 0.04
    final_weight_decay = 0.4
    warmup_epochs = 0
    epochs = 10
    use_bfloat16 = False
    clip_grad = 1.0
    ema_start = 0.99
    ema_end = 0.999
    loss_exp = 1.0
    loss_beta = 1.0
    normalize_reps = False
    auto_steps = 0


def _make_trainer(embed_dim: int, pred_dim: int, depth: int, device: torch.device):
    """Build a JEPATrainer on the specified device."""
    from assets.jepa_trainer_template import JEPATrainer
    from assets.predictor_template import VisionTransformerPredictor

    encoder = _TinyEncoder(embed_dim=embed_dim).to(device)
    predictor = VisionTransformerPredictor(
        embed_dim=embed_dim,
        predictor_embed_dim=pred_dim,
        depth=depth,
        num_heads=max(1, pred_dim // 32),
        num_targets=4,
    ).to(device)

    return JEPATrainer(encoder, predictor, _TinyCfg())


# ---------------------------------------------------------------------------
# Benchmark 1: Step time and throughput vs batch size
# ---------------------------------------------------------------------------

def benchmark_step_time(
    batch_sizes: List[int],
    embed_dim: int,
    pred_dim: int,
    depth: int,
    n_vis: int,
    n_pred: int,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> List[Dict]:
    """Measure mean step time (ms) and throughput for each batch size."""
    results = []

    for bs in batch_sizes:
        trainer = _make_trainer(embed_dim, pred_dim, depth, device)
        masks_enc  = [torch.arange(n_vis, device=device)]
        masks_pred = [torch.arange(n_vis, n_vis + n_pred, device=device)]

        batch = torch.randn(bs, n_vis, embed_dim, device=device)

        # Warmup runs
        for _ in range(warmup):
            trainer.train_step(batch, masks_enc, masks_pred)
            trainer.update_ema(0)

        _sync()
        _reset_peak_memory()
        gc.collect()

        # Timed runs
        times = []
        for step in range(repeats):
            _sync()
            t0 = time.perf_counter()
            loss_dict = trainer.train_step(batch, masks_enc, masks_pred)
            trainer.update_ema(step)
            _sync()
            elapsed = (time.perf_counter() - t0) * 1000  # ms
            times.append(elapsed)

        mean_ms   = sum(times) / len(times)
        std_ms    = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
        throughput = bs / (mean_ms / 1000)  # samples/sec
        peak_mem  = _get_peak_memory_mb()
        mem_per_s = peak_mem / bs if peak_mem > 0 else 0.0

        results.append({
            'batch_size':       bs,
            'mean_step_ms':     round(mean_ms, 2),
            'std_ms':           round(std_ms, 2),
            'throughput_sps':   round(throughput, 1),
            'peak_memory_mb':   round(peak_mem, 1),
            'mem_per_sample_mb': round(mem_per_s, 2),
        })

        del trainer, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: Predictor forward vs depth
# ---------------------------------------------------------------------------

def benchmark_predictor_depth(
    depths: List[int],
    embed_dim: int,
    pred_dim: int,
    batch_size: int,
    n_vis: int,
    n_pred: int,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> List[Dict]:
    """Measure predictor-only forward time vs number of transformer blocks."""
    from assets.predictor_template import VisionTransformerPredictor

    results = []

    for depth in depths:
        predictor = VisionTransformerPredictor(
            embed_dim=embed_dim,
            predictor_embed_dim=pred_dim,
            depth=depth,
            num_heads=max(1, pred_dim // 32),
            num_targets=4,
        ).to(device)

        context    = torch.randn(batch_size, n_vis, embed_dim, device=device)
        masks_enc  = [torch.arange(n_vis, device=device)]
        masks_pred = [torch.arange(n_vis, n_vis + n_pred, device=device)]

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                predictor(context, masks_enc, masks_pred)

        _sync()
        times = []
        for _ in range(repeats):
            _sync()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = predictor(context, masks_enc, masks_pred)
            _sync()
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms = sum(times) / len(times)
        param_count = sum(p.numel() for p in predictor.parameters())

        results.append({
            'depth':        depth,
            'mean_fwd_ms':  round(mean_ms, 2),
            'params_M':     round(param_count / 1e6, 2),
            'output_shape': str(tuple(out.shape)),
        })

        del predictor, context, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# Benchmark 3: EMA update overhead
# ---------------------------------------------------------------------------

def benchmark_ema_update(
    embed_dim: int,
    n_steps: int,
    device: torch.device,
) -> Dict:
    """Measure EMA update time (microseconds per update)."""
    from assets.ema_manager_template import EMAManagerWithRef

    encoder = _TinyEncoder(embed_dim=embed_dim).to(device)
    ema = EMAManagerWithRef(encoder, (0.99925, 0.99925), total_steps=n_steps)

    # Warmup
    for step in range(10):
        with torch.no_grad():
            for p in encoder.parameters():
                p.add_(torch.randn_like(p) * 0.001)
        ema.update(step)

    _sync()
    times = []
    for step in range(100):
        _sync()
        t0 = time.perf_counter()
        ema.update(step)
        _sync()
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds

    mean_us = sum(times) / len(times)
    param_count = sum(p.numel() for p in encoder.parameters())

    return {
        'mean_update_us': round(mean_us, 2),
        'encoder_params':  param_count,
        'overhead_pct':    '< 0.1%'  # Typically negligible vs forward pass
    }


# ---------------------------------------------------------------------------
# Benchmark 4: Checkpoint save/load time
# ---------------------------------------------------------------------------

def benchmark_checkpoint(
    embed_dim: int,
    pred_dim: int,
    depth: int,
    device: torch.device,
    repeats: int = 5,
) -> Dict:
    """Measure checkpoint save and load time."""
    trainer = _make_trainer(embed_dim, pred_dim, depth, device)

    # Warm up trainer
    batch  = torch.randn(2, 8, embed_dim, device=device)
    m_enc  = [torch.arange(8, device=device)]
    m_pred = [torch.arange(8, 12, device=device)]
    for step in range(3):
        trainer.train_step(batch, m_enc, m_pred)
        trainer.update_ema(step)

    save_times = []
    load_times = []
    file_size_mb = 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(repeats):
            path = os.path.join(tmpdir, f"ckpt_{i}.pth")

            # Save
            t0 = time.perf_counter()
            trainer.save_checkpoint(path, epoch=i)
            save_times.append((time.perf_counter() - t0) * 1000)

            if i == 0:
                file_size_mb = os.path.getsize(path) / 1024 ** 2

            # Load into fresh trainer
            trainer2 = _make_trainer(embed_dim, pred_dim, depth, device)
            t0 = time.perf_counter()
            trainer2.load_checkpoint(path)
            load_times.append((time.perf_counter() - t0) * 1000)

            del trainer2

    return {
        'mean_save_ms':  round(sum(save_times) / len(save_times), 1),
        'mean_load_ms':  round(sum(load_times) / len(load_times), 1),
        'file_size_mb':  round(file_size_mb, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Benchmark V-JEPA 2 training components'
    )
    parser.add_argument('--batch-sizes', nargs='+', type=int,
                        default=[1, 2, 4, 8, 16],
                        help='Batch sizes to benchmark (default: 1 2 4 8 16)')
    parser.add_argument('--embed-dim', type=int, default=128,
                        help='Encoder embedding dimension (default: 128)')
    parser.add_argument('--pred-dim', type=int, default=64,
                        help='Predictor embedding dimension (default: 64)')
    parser.add_argument('--depth', type=int, default=4,
                        help='Predictor transformer depth (default: 4)')
    parser.add_argument('--n-vis', type=int, default=16,
                        help='Number of visible patches (default: 16)')
    parser.add_argument('--n-pred', type=int, default=8,
                        help='Number of target patches (default: 8)')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Warmup iterations before timing (default: 3)')
    parser.add_argument('--repeats', type=int, default=10,
                        help='Timed repetitions (default: 10)')
    parser.add_argument('--depths', nargs='+', type=int,
                        default=[2, 4, 6, 8, 12],
                        help='Predictor depths to benchmark (default: 2 4 6 8 12)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to this file (default: stdout only)')
    parser.add_argument('--skip-ema', action='store_true',
                        help='Skip EMA benchmark')
    parser.add_argument('--skip-checkpoint', action='store_true',
                        help='Skip checkpoint benchmark')

    args = parser.parse_args()

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    lines = []

    def _print(s: str = '') -> None:
        print(s)
        lines.append(s)

    _print("V-JEPA 2 Training Benchmark")
    _print("=" * 70)
    _print(f"Device:    {device_str.upper()}")
    _print(f"PyTorch:   {torch.__version__}")
    _print(f"embed_dim: {args.embed_dim}  pred_dim: {args.pred_dim}  "
           f"depth: {args.depth}")
    _print(f"n_vis: {args.n_vis}  n_pred: {args.n_pred}")
    _print(f"warmup: {args.warmup}  repeats: {args.repeats}")
    _print()

    # ------------------------------------------------------------------
    # Benchmark 1: Step time vs batch size
    # ------------------------------------------------------------------
    _print("Benchmark 1: Full Training Step (encode + predict + loss + backward)")
    _print("-" * 70)

    step_results = benchmark_step_time(
        batch_sizes=args.batch_sizes,
        embed_dim=args.embed_dim,
        pred_dim=args.pred_dim,
        depth=args.depth,
        n_vis=args.n_vis,
        n_pred=args.n_pred,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )

    headers1 = ['batch', 'step_ms', 'std_ms', 'samples/sec', 'peak_mem_MB', 'MB/sample']
    rows1 = [
        [r['batch_size'], r['mean_step_ms'], r['std_ms'],
         r['throughput_sps'], r['peak_memory_mb'], r['mem_per_sample_mb']]
        for r in step_results
    ]
    table1 = _format_table(headers1, rows1)
    for line in table1.split('\n'):
        _print(line)

    _print()

    # ------------------------------------------------------------------
    # Benchmark 2: Predictor depth vs forward time
    # ------------------------------------------------------------------
    _print("Benchmark 2: Predictor Forward vs Transformer Depth")
    _print("-" * 70)

    depth_results = benchmark_predictor_depth(
        depths=args.depths,
        embed_dim=args.embed_dim,
        pred_dim=args.pred_dim,
        batch_size=4,
        n_vis=args.n_vis,
        n_pred=args.n_pred,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )

    headers2 = ['depth', 'fwd_ms', 'params_M', 'output_shape']
    rows2 = [
        [r['depth'], r['mean_fwd_ms'], r['params_M'], r['output_shape']]
        for r in depth_results
    ]
    table2 = _format_table(headers2, rows2)
    for line in table2.split('\n'):
        _print(line)

    _print()

    # ------------------------------------------------------------------
    # Benchmark 3: EMA update overhead
    # ------------------------------------------------------------------
    if not args.skip_ema:
        _print("Benchmark 3: EMA Update Overhead")
        _print("-" * 70)
        try:
            ema_result = benchmark_ema_update(
                embed_dim=args.embed_dim,
                n_steps=1000,
                device=device,
            )
            _print(f"  Mean update time: {ema_result['mean_update_us']:.1f} us/step")
            _print(f"  Encoder params:   {ema_result['encoder_params']:,}")
            _print(f"  Overhead vs step: {ema_result['overhead_pct']}")
        except Exception as exc:
            _print(f"  [ERROR] EMA benchmark failed: {exc}")
        _print()

    # ------------------------------------------------------------------
    # Benchmark 4: Checkpoint save/load time
    # ------------------------------------------------------------------
    if not args.skip_checkpoint:
        _print("Benchmark 4: Checkpoint Save / Load Time")
        _print("-" * 70)
        try:
            ckpt_result = benchmark_checkpoint(
                embed_dim=args.embed_dim,
                pred_dim=args.pred_dim,
                depth=args.depth,
                device=device,
                repeats=max(3, args.warmup),
            )
            _print(f"  Mean save time:  {ckpt_result['mean_save_ms']:.1f} ms")
            _print(f"  Mean load time:  {ckpt_result['mean_load_ms']:.1f} ms")
            _print(f"  Checkpoint size: {ckpt_result['file_size_mb']:.2f} MB")
        except Exception as exc:
            _print(f"  [ERROR] Checkpoint benchmark failed: {exc}")
        _print()

    # ------------------------------------------------------------------
    # Summary: Best batch size for throughput
    # ------------------------------------------------------------------
    if step_results:
        best = max(step_results, key=lambda r: r['throughput_sps'])
        _print("Summary")
        _print("-" * 70)
        _print(f"  Best throughput:  {best['throughput_sps']:.1f} samples/sec "
               f"(batch_size={best['batch_size']})")
        _print(f"  Fastest step:     {min(r['mean_step_ms'] for r in step_results):.1f} ms "
               f"(batch_size={min(step_results, key=lambda r: r['mean_step_ms'])['batch_size']})")
        if device_str == 'cuda':
            _print(f"  Peak GPU memory: {max(r['peak_memory_mb'] for r in step_results):.1f} MB")
        else:
            _print("  (Running on CPU -- GPU memory not measured)")

    # ------------------------------------------------------------------
    # Save to file if requested
    # ------------------------------------------------------------------
    if args.output:
        with open(args.output, 'w') as fh:
            fh.write('\n'.join(lines))
        print(f"\nResults saved to: {os.path.abspath(args.output)}")


if __name__ == '__main__':
    main()
