#!/usr/bin/env python3
"""
perf_benchmark.py
==================
Memory and throughput benchmark for V-JEPA 2 performance optimizations.

Measures each optimization combination's effect on:
  - Peak GPU memory (MB)
  - Forward pass throughput (samples/sec)
  - Backward pass throughput (samples/sec)

Prints a formatted comparison table.

Usage:
    python scripts/perf_benchmark.py
    python scripts/perf_benchmark.py --batch-size 16 --dim 256 --blocks 6
    python scripts/perf_benchmark.py --cpu-only        # Run on CPU (no CUDA required)
    python scripts/perf_benchmark.py --warmup 3 --iters 20

Output example:
    =====================================================================
    V-JEPA 2 Performance Optimization Benchmark
    =====================================================================
    Device: cuda:0 (NVIDIA A100)  |  Batch: 8  |  Dim: 192  |  Blocks: 6
    =====================================================================
    Configuration                  | Peak Mem (MB) | Fwd (smp/s) | Total (smp/s)
    -----------------------------------------------------------------------
    Baseline (fp32, no opts)       |       1842.3  |      834.2  |       412.1
    + Activation Checkpointing     |       1021.5  |      687.4  |       341.8
    + bfloat16                     |        934.1  |      921.7  |       458.2
    + torch.compile                |        931.2  |     1124.5  |       563.1
    + SDPA                         |        882.4  |     1204.3  |       601.8
    All optimizations              |        870.1  |     1189.2  |       594.1
    =====================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

# Make assets importable
_SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_SKILL_DIR, "assets"))

try:
    from perf_optimizer_template import PerformanceOptimizer, PerfConfig
    _PERF_AVAILABLE = True
except ImportError as e:
    _PERF_AVAILABLE = False
    print(f"Warning: Could not import perf_optimizer_template: {e}")
    print("Continuing with manual optimization wrappers.")


# ---------------------------------------------------------------------------
# Benchmark model
# ---------------------------------------------------------------------------

class SDPAAttention(nn.Module):
    """Multi-head self-attention using SDPA (Flash Attention or math fallback)."""

    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            # Fallback math attention
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block."""

    def __init__(self, dim: int, heads: int = 8, use_sdpa: bool = True) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = SDPAAttention(dim, heads) if use_sdpa else nn.MultiheadAttention(
            dim, heads, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self._use_sdpa = use_sdpa

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_sdpa:
            x = x + self.attn(self.norm1(x))
        else:
            attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
            x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class BenchmarkViT(nn.Module):
    """Minimal ViT encoder for benchmarking."""

    def __init__(
        self,
        seq_len: int = 196,
        dim: int = 192,
        n_blocks: int = 6,
        heads: int = 8,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(3 * 16 * 16, dim)  # Patch embedding simulation
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, use_sdpa=use_sdpa)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(dim)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, patch_dim]
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    name: str
    peak_memory_mb: float
    fwd_throughput: float   # samples/sec
    total_throughput: float  # samples/sec (fwd + bwd)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def measure_throughput(
    model: nn.Module,
    x: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    use_bfloat16: bool,
    n_warmup: int,
    n_iters: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Returns (fwd_samples_per_sec, total_samples_per_sec).
    """
    model.train()
    batch_size = x.shape[0]
    dtype = torch.bfloat16 if (use_bfloat16 and device.type == "cuda") else torch.float32

    def run_fwd():
        if use_bfloat16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                return model(x).sum()
        else:
            return model(x).sum()

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model(x)

    # Sync before timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Forward-only timing
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            run_fwd()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fwd_time = time.perf_counter() - t0
    fwd_sps = (n_iters * batch_size) / fwd_time

    # Full forward+backward timing
    t0 = time.perf_counter()
    for _ in range(n_iters):
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss = model(x).sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = run_fwd()
            loss.backward()
            optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_time = time.perf_counter() - t0
    total_sps = (n_iters * batch_size) / total_time

    return fwd_sps, total_sps


def benchmark_configuration(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    use_bfloat16: bool,
    use_activation_checkpointing: bool,
    use_compile: bool,
    n_warmup: int,
    n_iters: int,
) -> BenchmarkResult:
    """Run a single benchmark configuration and return results."""
    try:
        import copy
        model_copy = copy.deepcopy(model).to(device)
        x_dev = x.to(device)

        optimizer = torch.optim.AdamW(model_copy.parameters(), lr=1e-4)
        scaler: Optional[GradScaler] = None

        # Apply optimizations
        if use_activation_checkpointing and _PERF_AVAILABLE:
            perf = PerformanceOptimizer(
                model_copy,
                PerfConfig(use_activation_checkpointing=True)
            )
            perf.enable_activation_checkpointing()

        if use_bfloat16 and _PERF_AVAILABLE:
            perf_amp = PerformanceOptimizer(model_copy, PerfConfig(use_bfloat16=True))
            scaler = perf_amp.enable_mixed_precision()
        elif use_bfloat16:
            scaler = GradScaler(enabled=False)  # bfloat16 doesn't need scaling

        if use_compile and _PERF_AVAILABLE:
            perf_compile = PerformanceOptimizer(model_copy, PerfConfig(compile_model=True))
            model_copy = perf_compile.compile_model()

        # Reset peak memory stats
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Run benchmark
        fwd_sps, total_sps = measure_throughput(
            model_copy, x_dev, optimizer, scaler, use_bfloat16,
            n_warmup, n_iters, device
        )

        # Measure peak memory
        if device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            peak_mb = 0.0

        # Cleanup
        del model_copy
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return BenchmarkResult(
            name=name,
            peak_memory_mb=peak_mb,
            fwd_throughput=fwd_sps,
            total_throughput=total_sps,
        )

    except Exception as e:
        return BenchmarkResult(
            name=name,
            peak_memory_mb=0.0,
            fwd_throughput=0.0,
            total_throughput=0.0,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_table(results: List[BenchmarkResult], device: torch.device, args: argparse.Namespace) -> None:
    col1_width = 35
    col2_width = 15
    col3_width = 14
    col4_width = 15

    sep = "=" * (col1_width + col2_width + col3_width + col4_width + 6)
    dash = "-" * (col1_width + col2_width + col3_width + col4_width + 6)

    device_name = "CPU"
    if device.type == "cuda":
        try:
            device_name = f"cuda:{device.index} ({torch.cuda.get_device_name(device)})"
        except Exception:
            device_name = f"cuda:{device.index}"

    print(f"\n{sep}")
    print(f"V-JEPA 2 Performance Optimization Benchmark")
    print(f"{sep}")
    print(
        f"Device: {device_name}  |  "
        f"Batch: {args.batch_size}  |  "
        f"Dim: {args.dim}  |  "
        f"Blocks: {args.blocks}  |  "
        f"SeqLen: {args.seq_len}"
    )
    print(f"{sep}")

    header = (
        f"{'Configuration':<{col1_width}} | "
        f"{'Peak Mem (MB)':>{col2_width}} | "
        f"{'Fwd (smp/s)':>{col3_width}} | "
        f"{'Total (smp/s)':>{col4_width}}"
    )
    print(header)
    print(dash)

    baseline_mem = None
    baseline_total = None

    for r in results:
        if r.error:
            row = (
                f"{r.name:<{col1_width}} | "
                f"{'ERROR':>{col2_width}} | "
                f"{'N/A':>{col3_width}} | "
                f"{'N/A':>{col4_width}}"
            )
            print(row)
            print(f"    Error: {r.error[:80]}")
            continue

        if baseline_mem is None:
            baseline_mem = r.peak_memory_mb
            baseline_total = r.total_throughput

        # Memory delta
        if baseline_mem is not None and baseline_mem > 0:
            mem_delta = r.peak_memory_mb - baseline_mem
            mem_pct = (mem_delta / baseline_mem) * 100
            mem_str = f"{r.peak_memory_mb:>8.1f} ({mem_pct:+.0f}%)"
        else:
            mem_str = f"{r.peak_memory_mb:>8.1f}"

        # Throughput delta
        if baseline_total is not None and baseline_total > 0:
            spd_pct = ((r.total_throughput - baseline_total) / baseline_total) * 100
            spd_str = f"{r.total_throughput:>9.0f} ({spd_pct:+.0f}%)"
        else:
            spd_str = f"{r.total_throughput:>9.0f}"

        row = (
            f"{r.name:<{col1_width}} | "
            f"{mem_str:>{col2_width}} | "
            f"{r.fwd_throughput:>{col3_width}.0f} | "
            f"{spd_str:>{col4_width}}"
        )
        print(row)

    print(f"{sep}")

    # Summary
    if len(results) >= 2 and not results[0].error and not results[-1].error:
        mem_saving = results[0].peak_memory_mb - results[-1].peak_memory_mb
        spd_gain = (
            (results[-1].total_throughput - results[0].total_throughput)
            / max(results[0].total_throughput, 1e-6)
        ) * 100
        print(f"\nSummary:")
        print(f"  Memory saved (baseline -> all opts): {mem_saving:.1f} MB")
        print(f"  Throughput gain:                     {spd_gain:+.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark V-JEPA 2 performance optimizations"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length (num patches)")
    parser.add_argument("--dim", type=int, default=128, help="Model embedding dimension")
    parser.add_argument("--blocks", type=int, default=4, help="Number of transformer blocks")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--patch-dim", type=int, default=768, help="Patch input dimension (3*16*16)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    # Device selection
    if args.cpu_only or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Note: Running on CPU. Memory measurements will be zero.")
        if not args.cpu_only:
            print("      Install CUDA-capable PyTorch for GPU benchmarks.")
    else:
        device = torch.device("cuda:0")

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        print(f"GPU Memory: {total_mem:.0f} MB")

    # Build reference model
    patch_dim = 3 * 16 * 16  # Standard patch size
    base_model = BenchmarkViT(
        seq_len=args.seq_len,
        dim=args.dim,
        n_blocks=args.blocks,
        heads=args.heads,
        use_sdpa=False,  # Start without SDPA for baseline
    )

    param_count = sum(p.numel() for p in base_model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.1f}M")

    # Input tensor
    x = torch.randn(args.batch_size, args.seq_len, patch_dim)

    # Define benchmark configurations
    configs = [
        {
            "name": "Baseline (fp32, no opts)",
            "use_bfloat16": False,
            "use_activation_checkpointing": False,
            "use_compile": False,
        },
        {
            "name": "+ Activation Checkpointing",
            "use_bfloat16": False,
            "use_activation_checkpointing": True,
            "use_compile": False,
        },
        {
            "name": "+ bfloat16 only",
            "use_bfloat16": True,
            "use_activation_checkpointing": False,
            "use_compile": False,
        },
        {
            "name": "+ Ckpt + bfloat16",
            "use_bfloat16": True,
            "use_activation_checkpointing": True,
            "use_compile": False,
        },
    ]

    # Add torch.compile configs only if PyTorch >= 2.0
    if hasattr(torch, "compile"):
        configs.append({
            "name": "+ Ckpt + bf16 + compile",
            "use_bfloat16": True,
            "use_activation_checkpointing": True,
            "use_compile": True,
        })

    # Add SDPA config (rebuild model with SDPA enabled)
    configs.append({
        "name": "Ckpt + bf16 + SDPA (full)",
        "use_bfloat16": True,
        "use_activation_checkpointing": True,
        "use_compile": False,
        "_use_sdpa": True,  # Special flag to rebuild model with SDPA
    })

    results: List[BenchmarkResult] = []

    print(f"\nRunning {len(configs)} benchmark configurations...")
    print(f"  Warmup: {args.warmup} iters, Benchmark: {args.iters} iters per config\n")

    for i, cfg in enumerate(configs):
        name = cfg["name"]
        print(f"  [{i+1}/{len(configs)}] {name}...", end=" ", flush=True)

        # Optionally rebuild model with SDPA
        if cfg.pop("_use_sdpa", False):
            model_for_bench = BenchmarkViT(
                seq_len=args.seq_len,
                dim=args.dim,
                n_blocks=args.blocks,
                heads=args.heads,
                use_sdpa=True,  # Use SDPA attention
            )
        else:
            model_for_bench = BenchmarkViT(
                seq_len=args.seq_len,
                dim=args.dim,
                n_blocks=args.blocks,
                heads=args.heads,
                use_sdpa=False,
            )

        result = benchmark_configuration(
            name=name,
            model=model_for_bench,
            x=x,
            device=device,
            use_bfloat16=cfg.get("use_bfloat16", False),
            use_activation_checkpointing=cfg.get("use_activation_checkpointing", False),
            use_compile=cfg.get("use_compile", False),
            n_warmup=args.warmup,
            n_iters=args.iters,
        )
        results.append(result)

        if result.error:
            print(f"ERROR: {result.error[:60]}")
        else:
            print(f"done. Peak={result.peak_memory_mb:.0f}MB, Total={result.total_throughput:.0f} smp/s")

    # Print summary table
    print_table(results, device, args)

    # Optimization impact analysis
    print("\nOptimization Impact Analysis:")
    print("-" * 50)

    if len(results) >= 2 and not results[0].error:
        baseline = results[0]
        for r in results[1:]:
            if r.error:
                continue
            if baseline.peak_memory_mb > 0:
                mem_pct = (
                    (baseline.peak_memory_mb - r.peak_memory_mb) / baseline.peak_memory_mb * 100
                )
                print(f"  {r.name}:")
                print(f"    Memory reduction:    {mem_pct:.1f}%")
            if baseline.total_throughput > 0:
                spd_pct = (
                    (r.total_throughput - baseline.total_throughput)
                    / baseline.total_throughput * 100
                )
                print(f"    Throughput change:   {spd_pct:+.1f}%")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
