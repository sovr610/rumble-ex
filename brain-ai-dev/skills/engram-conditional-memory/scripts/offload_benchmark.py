#!/usr/bin/env python3
"""
Engram Conditional Memory -- Offload Benchmark Suite
=====================================================

Measures embedding retrieval performance under three modes:
  1. On-device (standard F.embedding gather)
  2. CPU offload without prefetch (synchronous H2D copy per batch)
  3. CPU offload with async prefetch (dedicated CUDA stream)

Also benchmarks coalescing efficiency, multi-head hash throughput,
and end-to-end EngramModule forward pass with component breakdown.

Usage:
    python offload_benchmark.py --suite all
    python offload_benchmark.py --suite 1 --device cuda --output results.json
    python offload_benchmark.py --suite 3 --device cuda
    python offload_benchmark.py --suite 4 5 --device cpu

Suites:
    1  Embedding Gather Baseline (on-device)
    2  CPU Offload Without Prefetch
    3  CPU Offload With Async Prefetch
    4  Coalescing Efficiency
    5  Multi-Head Hash Throughput
    6  End-to-End EngramModule
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

WARMUP_RUNS = 5
TIMING_RUNS = 20
SEED = 42

# ---------------------------------------------------------------------------
# Utility: pretty-print helpers
# ---------------------------------------------------------------------------


def _hr(char: str = "-", width: int = 100) -> str:
    return char * width


def _header(title: str, width: int = 100) -> str:
    pad = max(0, width - len(title) - 4)
    left = pad // 2
    right = pad - left
    return f"{'=' * left}  {title}  {'=' * right}"


def _print_table(headers: List[str], rows: List[List[Any]], col_widths: Optional[List[int]] = None) -> None:
    """Print a formatted table to stdout."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_len = len(str(h))
            for row in rows:
                if i < len(row):
                    max_len = max(max_len, len(str(row[i])))
            col_widths.append(max_len + 2)

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*[str(h) for h in headers]))
    print(fmt.format(*["-" * w for w in col_widths]))
    for row in rows:
        padded = [str(row[i]) if i < len(row) else "" for i in range(len(headers))]
        print(fmt.format(*padded))


# ---------------------------------------------------------------------------
# Utility: timing helpers
# ---------------------------------------------------------------------------


class CudaTimer:
    """GPU-precise timer using CUDA events, with wall-clock fallback for CPU."""

    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda = device.type == "cuda"
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        self._wall_start: float = 0.0

    def start(self) -> None:
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
            self.start_event.record(torch.cuda.current_stream(self.device))
        else:
            self._wall_start = time.perf_counter()

    def stop(self) -> float:
        """Return elapsed time in milliseconds."""
        if self.use_cuda:
            self.end_event.record(torch.cuda.current_stream(self.device))
            torch.cuda.synchronize(self.device)
            return self.start_event.elapsed_time(self.end_event)
        else:
            return (time.perf_counter() - self._wall_start) * 1000.0


class MultiTimer:
    """Collect multiple named lap times."""

    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda = device.type == "cuda"
        self._laps: Dict[str, List[float]] = {}

    def _now_ms(self) -> float:
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        return time.perf_counter() * 1000.0

    def lap(self, name: str, elapsed_ms: float) -> None:
        self._laps.setdefault(name, []).append(elapsed_ms)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for name, values in self._laps.items():
            arr = np.array(values)
            out[name] = {
                "mean_ms": float(np.mean(arr)),
                "std_ms": float(np.std(arr)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
            }
        return out


# ---------------------------------------------------------------------------
# Utility: statistics helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkStats:
    """Aggregated statistics for a single configuration."""

    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    throughput_tokens_sec: float = 0.0
    gpu_mem_bytes: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_times(cls, times_ms: List[float], total_tokens: int, gpu_mem_bytes: int = 0,
                   **extra: Any) -> "BenchmarkStats":
        arr = np.array(times_ms)
        mean_ms = float(np.mean(arr))
        throughput = (total_tokens / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0
        return cls(
            mean_ms=mean_ms,
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            throughput_tokens_sec=throughput,
            gpu_mem_bytes=gpu_mem_bytes,
            extra=dict(extra),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "mean_ms": round(self.mean_ms, 4),
            "std_ms": round(self.std_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "throughput_tokens_sec": round(self.throughput_tokens_sec, 2),
            "gpu_mem_bytes": self.gpu_mem_bytes,
        }
        d.update(self.extra)
        return d


def _get_gpu_mem(device: torch.device) -> int:
    """Return current GPU memory allocated in bytes, or 0 for CPU."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    return 0


def _get_peak_gpu_mem(device: torch.device) -> int:
    """Return peak GPU memory allocated in bytes, or 0 for CPU."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    return 0


def _reset_peak_gpu_mem(device: torch.device) -> None:
    """Reset peak memory tracker."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _format_bytes(nbytes: int) -> str:
    """Human-readable byte count."""
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 ** 2:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024 ** 3:
        return f"{nbytes / 1024 ** 2:.1f} MB"
    else:
        return f"{nbytes / 1024 ** 3:.2f} GB"


# ---------------------------------------------------------------------------
# Mock Engram Components (self-contained, no external template dependency)
# ---------------------------------------------------------------------------


class MockMultiHeadHash:
    """
    Deterministic multi-head hash for N-gram -> embedding index mapping.

    Implements multiplicative + XOR + seed hashing as described in the
    multi-head-hashing reference.  All arithmetic is int64 for cross-device
    determinism.
    """

    def __init__(
        self,
        ngram_order: int,
        num_heads: int,
        table_size: int,
        seed: int = 42,
    ):
        self.n = ngram_order
        self.num_heads = num_heads
        self.table_size = table_size

        rng = np.random.RandomState(seed + ngram_order)
        self.coefficients = torch.tensor(
            rng.randint(1, table_size, size=(num_heads, ngram_order)),
            dtype=torch.long,
        )
        self.seeds = torch.tensor(
            rng.randint(0, table_size, size=(num_heads,)),
            dtype=torch.long,
        )

    def hash(self, ngrams: torch.Tensor) -> torch.Tensor:
        """
        Hash N-grams to embedding indices.

        Args:
            ngrams: (batch, seq_len, n) int64 token IDs
        Returns:
            (batch, seq_len, num_heads) int64 embedding indices
        """
        device = ngrams.device
        batch, seq_len, n = ngrams.shape

        coeffs = self.coefficients.to(device)      # (num_heads, n)
        seeds = self.seeds.to(device)               # (num_heads,)

        ngrams_exp = ngrams.unsqueeze(2).long()     # (B, T, 1, n)
        coeffs_exp = coeffs.unsqueeze(0).unsqueeze(0)  # (1, 1, H, n)

        weighted = (ngrams_exp * coeffs_exp).sum(dim=-1)  # (B, T, H)
        seeds_exp = seeds.unsqueeze(0).unsqueeze(0)       # (1, 1, H)
        hashed = (weighted ^ seeds_exp) % self.table_size

        return hashed

    def hash_naive(self, ngrams: torch.Tensor) -> torch.Tensor:
        """
        Naive (loop-based) hash for comparison benchmarking.

        Same output as self.hash() but uses explicit Python loops
        over heads and n-gram positions.
        """
        device = ngrams.device
        batch, seq_len, n = ngrams.shape

        coeffs = self.coefficients.to(device)
        seeds = self.seeds.to(device)

        result = torch.zeros(batch, seq_len, self.num_heads, dtype=torch.long, device=device)

        for h_idx in range(self.num_heads):
            acc = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
            for pos in range(n):
                acc = acc + ngrams[:, :, pos].long() * coeffs[h_idx, pos]
            result[:, :, h_idx] = (acc ^ seeds[h_idx]) % self.table_size

        return result


class MockOffloadableEmbedding(nn.Module):
    """
    Embedding table with optional CPU offload and async prefetch.

    Three modes:
      - on_device: standard nn.Embedding on the compute device
      - offload_sync: table on CPU (pinned), synchronous H2D copy
      - offload_async: table on CPU (pinned), async copy on separate stream
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mode: str = "on_device",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self._device = device or torch.device("cpu")

        if mode == "on_device":
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
            if self._device.type == "cuda":
                self.embedding = self.embedding.to(self._device)
        else:
            # CPU-resident weight with pinned memory
            weight = torch.zeros(num_embeddings, embedding_dim)
            nn.init.normal_(weight, mean=0, std=0.02)
            if torch.cuda.is_available():
                weight = weight.pin_memory()
            self.register_buffer("weight", weight)

            # Prefetch machinery
            self._prefetch_stream: Optional[torch.cuda.Stream] = None
            self._prefetch_buffer: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, ...]]] = None
            if mode == "offload_async" and torch.cuda.is_available():
                self._prefetch_stream = torch.cuda.Stream(device=self._device)

    # ---- async prefetch API ------------------------------------------------

    def prefetch_async(self, indices: torch.Tensor) -> None:
        """Kick off async H2D copy for the unique rows needed by indices."""
        if self.mode != "offload_async" or self._prefetch_stream is None:
            return

        with torch.cuda.stream(self._prefetch_stream):
            flat = indices.flatten()
            unique_ids, inverse = flat.unique(return_inverse=True)
            cpu_ids = unique_ids.cpu()
            rows = self.weight[cpu_ids]
            rows_gpu = rows.to(self._device, non_blocking=True)
            self._prefetch_buffer = (unique_ids, rows_gpu, inverse, indices.shape)

    def _gather_prefetched(self, indices: torch.Tensor) -> torch.Tensor:
        """Wait for the prefetch stream and scatter results."""
        assert self._prefetch_buffer is not None
        torch.cuda.current_stream(self._device).wait_stream(self._prefetch_stream)
        unique_ids, rows_gpu, inverse, orig_shape = self._prefetch_buffer
        self._prefetch_buffer = None

        gathered = rows_gpu[inverse]
        return gathered.view(*orig_shape, self.embedding_dim)

    # ---- synchronous offload -----------------------------------------------

    def _sync_gather(self, indices: torch.Tensor) -> torch.Tensor:
        """Synchronous CPU->GPU transfer with coalescing."""
        flat = indices.flatten()
        unique_ids, inverse = flat.unique(return_inverse=True)
        cpu_ids = unique_ids.cpu()
        rows = self.weight[cpu_ids]
        if indices.is_cuda:
            rows = rows.to(indices.device)
        gathered = rows[inverse]
        return gathered.view(*indices.shape, self.embedding_dim)

    def _sync_gather_no_coalesce(self, indices: torch.Tensor) -> torch.Tensor:
        """Synchronous gather without coalescing (flat index into CPU table)."""
        cpu_indices = indices.cpu()
        rows = self.weight[cpu_indices.flatten()]
        rows = rows.view(*indices.shape, self.embedding_dim)
        if indices.is_cuda:
            rows = rows.to(indices.device)
        return rows

    # ---- forward -----------------------------------------------------------

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if self.mode == "on_device":
            return self.embedding(indices)
        elif self.mode == "offload_async" and self._prefetch_buffer is not None:
            return self._gather_prefetched(indices)
        else:
            return self._sync_gather(indices)


class MockRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class MockContextAwareGating(nn.Module):
    """Gate retrieved memory with hidden state."""

    def __init__(self, hidden_dim: int, memory_dim: int, temperature: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.temperature = temperature

        self.W_K = nn.Linear(memory_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(memory_dim, hidden_dim, bias=False)
        self.query_norm = MockRMSNorm(hidden_dim)
        self.key_norm = MockRMSNorm(hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(
        self, hidden: torch.Tensor, memory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.W_K(memory)
        v = self.W_V(memory)
        q_n = self.query_norm(hidden)
        k_n = self.key_norm(k)
        logits = (q_n * k_n).sum(dim=-1, keepdim=True) * self.scale / self.temperature
        alpha = torch.sigmoid(logits)
        return alpha * v, alpha.squeeze(-1)


class MockEngramModule(nn.Module):
    """
    Minimal end-to-end Engram module for benchmarking.

    Pipeline: hash -> retrieve -> aggregate -> gate -> conv -> project
    """

    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int = 256,
        table_size: int = 131071,
        ngram_orders: Tuple[int, ...] = (2, 3),
        num_heads: int = 2,
        conv_kernel: int = 4,
        conv_dilation: int = 3,
        mode: str = "on_device",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.ngram_orders = ngram_orders
        self.num_heads = num_heads
        self.mode = mode
        self._device = device or torch.device("cpu")

        total_heads = len(ngram_orders) * num_heads
        self.dim_per_head = embedding_dim // total_heads

        # Hash functions (not nn.Module, kept as plain objects)
        self.hashers = {
            n: MockMultiHeadHash(n, num_heads, table_size, seed=SEED + n)
            for n in ngram_orders
        }

        # Embedding tables
        self.embeddings = nn.ModuleDict()
        for n in ngram_orders:
            for k in range(num_heads):
                key = f"ngram{n}_head{k}"
                self.embeddings[key] = MockOffloadableEmbedding(
                    num_embeddings=table_size,
                    embedding_dim=self.dim_per_head,
                    mode=mode,
                    device=self._device,
                )

        # Gating
        self.gating = MockContextAwareGating(hidden_dim, embedding_dim, temperature=1.0)

        # Depthwise causal conv
        padding = (conv_kernel - 1) * conv_dilation
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=conv_kernel,
            padding=padding,
            dilation=conv_dilation,
            groups=hidden_dim,
        )
        self.conv_norm = MockRMSNorm(hidden_dim)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        # Projection from embedding_dim to hidden_dim (if different)
        if hidden_dim != embedding_dim:
            self.project = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.project = nn.Identity()

        self.to(self._device)

    # -- helpers -------------------------------------------------------------

    def _extract_ngrams(self, token_ids: torch.Tensor, n: int) -> torch.Tensor:
        padded = F.pad(token_ids, (n - 1, 0), value=0)
        seq_len = token_ids.shape[1]
        return torch.stack([padded[:, i:i + seq_len] for i in range(n)], dim=-1)

    # -- forward with per-component timing -----------------------------------

    def forward_timed(
        self, token_ids: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass that returns per-component wall-clock times (ms)."""
        device = token_ids.device
        timer = CudaTimer(device)
        times: Dict[str, float] = {}

        # --- hash ---
        timer.start()
        all_indices: Dict[str, torch.Tensor] = {}
        for n in self.ngram_orders:
            ngrams = self._extract_ngrams(token_ids, n)
            indices = self.hashers[n].hash(ngrams)
            for k in range(self.num_heads):
                key = f"ngram{n}_head{k}"
                all_indices[key] = indices[:, :, k]
        times["hash"] = timer.stop()

        # --- retrieve ---
        timer.start()
        emb_parts: List[torch.Tensor] = []
        for n in self.ngram_orders:
            for k in range(self.num_heads):
                key = f"ngram{n}_head{k}"
                emb_parts.append(self.embeddings[key](all_indices[key]))
        memory = torch.cat(emb_parts, dim=-1)
        times["retrieve"] = timer.stop()

        # --- gate ---
        timer.start()
        gated, alpha = self.gating(hidden, memory)
        times["gate"] = timer.stop()

        # --- conv ---
        timer.start()
        seq_len = token_ids.shape[1]
        g_norm = self.conv_norm(gated)
        conv_in = g_norm.transpose(1, 2)
        conv_out = self.conv(conv_in)[:, :, :seq_len].transpose(1, 2)
        out = F.silu(conv_out) + gated
        times["conv"] = timer.stop()

        # --- project ---
        timer.start()
        out = self.project(out)
        times["project"] = timer.stop()

        return out, times

    def forward(self, token_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_timed(token_ids, hidden)
        return out

    def prefetch_all(self, token_ids: torch.Tensor) -> None:
        """Issue async prefetch for all embedding tables."""
        if self.mode != "offload_async":
            return
        for n in self.ngram_orders:
            ngrams = self._extract_ngrams(token_ids, n)
            indices = self.hashers[n].hash(ngrams)
            for k in range(self.num_heads):
                key = f"ngram{n}_head{k}"
                self.embeddings[key].prefetch_async(indices[:, :, k])


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------


def _gen_token_ids(batch: int, seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch, seq_len), dtype=torch.long, device=device)


def _gen_indices(batch: int, seq_len: int, table_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, table_size, (batch, seq_len), dtype=torch.long, device=device)


def _gen_indices_with_unique_ratio(
    batch: int, seq_len: int, table_size: int, unique_ratio: float, device: torch.device,
) -> torch.Tensor:
    """
    Generate index tensor where approximately unique_ratio fraction of
    the flattened IDs are unique.

    When unique_ratio is low, most IDs are repeated (high coalescing benefit).
    """
    total = batch * seq_len
    num_unique = max(1, int(total * unique_ratio))
    num_unique = min(num_unique, table_size)

    pool = torch.randint(0, table_size, (num_unique,), dtype=torch.long)
    # Draw from the pool with replacement to fill the total
    draw_indices = torch.randint(0, num_unique, (total,), dtype=torch.long)
    flat = pool[draw_indices]
    return flat.view(batch, seq_len).to(device)


def _gen_hidden(batch: int, seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch, seq_len, dim, device=device)


# ---------------------------------------------------------------------------
# Suite 1: Embedding Gather Baseline
# ---------------------------------------------------------------------------


def suite1_embedding_gather_baseline(device: torch.device) -> Dict[str, Any]:
    """
    Benchmark standard on-device F.embedding gather.

    Sweeps table_size, embedding_dim, batch_size, seq_len.
    """
    print("\n" + _header("Suite 1: Embedding Gather Baseline (On-Device)"))

    table_sizes = [8191, 65537, 131071, 524287]
    embedding_dims = [128, 256, 512]
    batch_sizes = [1, 4, 16, 64]
    seq_lens = [128, 512, 2048]

    results: Dict[str, Any] = {"suite": 1, "name": "embedding_gather_baseline", "configs": []}

    # ---- Part A: sweep table_size x embedding_dim (fixed batch=16, seq=512) ----
    print("\n  Part A: table_size x embedding_dim  (batch=16, seq_len=512)")
    print(_hr())

    headers_a = ["table_size", "emb_dim", "mean_ms", "std_ms", "min_ms", "max_ms",
                 "tok/sec", "gpu_mem"]
    rows_a: List[List[str]] = []

    fixed_batch = 16
    fixed_seq = 512
    total_tokens = fixed_batch * fixed_seq

    for tsize in table_sizes:
        for edim in embedding_dims:
            torch.manual_seed(SEED)
            emb = nn.Embedding(tsize, edim).to(device)
            indices = _gen_indices(fixed_batch, fixed_seq, tsize, device)

            timer = CudaTimer(device)

            # warmup
            for _ in range(WARMUP_RUNS):
                _ = emb(indices)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            _reset_peak_gpu_mem(device)
            times: List[float] = []
            for _ in range(TIMING_RUNS):
                timer.start()
                _ = emb(indices)
                elapsed = timer.stop()
                times.append(elapsed)

            mem = _get_peak_gpu_mem(device)
            stats = BenchmarkStats.from_times(times, total_tokens, mem)
            results["configs"].append({
                "part": "A",
                "table_size": tsize,
                "embedding_dim": edim,
                "batch_size": fixed_batch,
                "seq_len": fixed_seq,
                **stats.to_dict(),
            })

            rows_a.append([
                str(tsize), str(edim),
                f"{stats.mean_ms:.4f}", f"{stats.std_ms:.4f}",
                f"{stats.min_ms:.4f}", f"{stats.max_ms:.4f}",
                f"{stats.throughput_tokens_sec:.0f}",
                _format_bytes(mem),
            ])

            del emb
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _print_table(headers_a, rows_a)

    # ---- Part B: sweep batch_size x seq_len (fixed table=131071, dim=256) ----
    print(f"\n  Part B: batch_size x seq_len  (table_size=131071, emb_dim=256)")
    print(_hr())

    headers_b = ["batch", "seq_len", "mean_ms", "std_ms", "min_ms", "max_ms",
                 "tok/sec", "gpu_mem"]
    rows_b: List[List[str]] = []

    fixed_tsize = 131071
    fixed_edim = 256
    torch.manual_seed(SEED)
    emb = nn.Embedding(fixed_tsize, fixed_edim).to(device)

    for bs in batch_sizes:
        for sl in seq_lens:
            total_tokens = bs * sl
            indices = _gen_indices(bs, sl, fixed_tsize, device)

            timer = CudaTimer(device)
            for _ in range(WARMUP_RUNS):
                _ = emb(indices)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            _reset_peak_gpu_mem(device)
            times = []
            for _ in range(TIMING_RUNS):
                timer.start()
                _ = emb(indices)
                elapsed = timer.stop()
                times.append(elapsed)

            mem = _get_peak_gpu_mem(device)
            stats = BenchmarkStats.from_times(times, total_tokens, mem)
            results["configs"].append({
                "part": "B",
                "table_size": fixed_tsize,
                "embedding_dim": fixed_edim,
                "batch_size": bs,
                "seq_len": sl,
                **stats.to_dict(),
            })

            rows_b.append([
                str(bs), str(sl),
                f"{stats.mean_ms:.4f}", f"{stats.std_ms:.4f}",
                f"{stats.min_ms:.4f}", f"{stats.max_ms:.4f}",
                f"{stats.throughput_tokens_sec:.0f}",
                _format_bytes(mem),
            ])

    del emb
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _print_table(headers_b, rows_b)

    return results


# ---------------------------------------------------------------------------
# Suite 2: CPU Offload Without Prefetch
# ---------------------------------------------------------------------------


def suite2_cpu_offload_sync(device: torch.device, baseline_results: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Benchmark synchronous CPU->device embedding retrieval.

    Embedding table lives in pinned CPU memory.  For each batch we:
      1. Coalesce (unique + inverse)
      2. Gather unique rows from CPU
      3. Copy to device synchronously
      4. Scatter back via inverse
    """
    if device.type != "cuda":
        print("\n" + _header("Suite 2: CPU Offload Without Prefetch"))
        print("  [SKIPPED] CUDA not available; CPU offload benchmarks require GPU.\n")
        return {"suite": 2, "name": "cpu_offload_sync", "skipped": True, "configs": []}

    print("\n" + _header("Suite 2: CPU Offload Without Prefetch (Sync H2D)"))

    table_sizes = [8191, 65537, 131071, 524287]
    embedding_dims = [128, 256, 512]
    batch_sizes = [1, 4, 16, 64]
    seq_lens = [128, 512, 2048]

    results: Dict[str, Any] = {"suite": 2, "name": "cpu_offload_sync", "configs": []}

    # Build lookup for baseline times
    baseline_lookup: Dict[str, float] = {}
    if baseline_results:
        for cfg in baseline_results.get("configs", []):
            key = f"{cfg.get('table_size')}_{cfg.get('embedding_dim')}_{cfg.get('batch_size')}_{cfg.get('seq_len')}"
            baseline_lookup[key] = cfg.get("mean_ms", 0.0)

    # ---- Part A: table_size x embedding_dim --------------------------------
    print("\n  Part A: table_size x embedding_dim  (batch=16, seq_len=512)")
    print(_hr())

    headers_a = ["table_size", "emb_dim", "mean_ms", "std_ms", "xfer_ms",
                 "coalesce_ratio", "slowdown", "tok/sec"]
    rows_a: List[List[str]] = []

    fixed_batch = 16
    fixed_seq = 512
    total_tokens = fixed_batch * fixed_seq

    for tsize in table_sizes:
        for edim in embedding_dims:
            torch.manual_seed(SEED)
            emb = MockOffloadableEmbedding(tsize, edim, mode="offload_sync", device=device)
            indices = _gen_indices(fixed_batch, fixed_seq, tsize, device)

            # Measure coalescing
            flat = indices.flatten()
            unique_count = flat.unique().numel()
            coalesce_ratio = unique_count / flat.numel()

            timer = CudaTimer(device)
            xfer_timer = CudaTimer(device)

            # warmup
            for _ in range(WARMUP_RUNS):
                _ = emb(indices)

            torch.cuda.synchronize(device)
            times: List[float] = []
            xfer_times: List[float] = []

            for _ in range(TIMING_RUNS):
                # Total latency
                timer.start()
                _ = emb(indices)
                elapsed = timer.stop()
                times.append(elapsed)

                # Isolate transfer time
                xfer_timer.start()
                flat_ids = indices.flatten()
                uniq, inv = flat_ids.unique(return_inverse=True)
                cpu_ids = uniq.cpu()
                rows = emb.weight[cpu_ids]
                _ = rows.to(device)
                xfer_elapsed = xfer_timer.stop()
                xfer_times.append(xfer_elapsed)

            stats = BenchmarkStats.from_times(times, total_tokens)
            mean_xfer = float(np.mean(xfer_times))

            bl_key = f"{tsize}_{edim}_{fixed_batch}_{fixed_seq}"
            bl_ms = baseline_lookup.get(bl_key, 0.0)
            slowdown = stats.mean_ms / bl_ms if bl_ms > 0 else float("nan")

            results["configs"].append({
                "part": "A",
                "table_size": tsize,
                "embedding_dim": edim,
                "batch_size": fixed_batch,
                "seq_len": fixed_seq,
                "coalesce_ratio": round(coalesce_ratio, 4),
                "pcie_xfer_ms": round(mean_xfer, 4),
                "slowdown_vs_baseline": round(slowdown, 2),
                **stats.to_dict(),
            })

            rows_a.append([
                str(tsize), str(edim),
                f"{stats.mean_ms:.4f}", f"{stats.std_ms:.4f}",
                f"{mean_xfer:.4f}",
                f"{coalesce_ratio:.4f}",
                f"{slowdown:.2f}x",
                f"{stats.throughput_tokens_sec:.0f}",
            ])

            del emb
            torch.cuda.empty_cache()

    _print_table(headers_a, rows_a)

    # ---- Part B: batch_size x seq_len --------------------------------------
    print(f"\n  Part B: batch_size x seq_len  (table_size=131071, emb_dim=256)")
    print(_hr())

    headers_b = ["batch", "seq_len", "mean_ms", "std_ms", "xfer_ms",
                 "coalesce_ratio", "slowdown", "tok/sec"]
    rows_b: List[List[str]] = []

    fixed_tsize = 131071
    fixed_edim = 256
    torch.manual_seed(SEED)

    for bs in batch_sizes:
        for sl in seq_lens:
            total_tokens = bs * sl
            emb = MockOffloadableEmbedding(fixed_tsize, fixed_edim, mode="offload_sync", device=device)
            indices = _gen_indices(bs, sl, fixed_tsize, device)

            flat = indices.flatten()
            unique_count = flat.unique().numel()
            coalesce_ratio = unique_count / flat.numel()

            timer = CudaTimer(device)
            xfer_timer = CudaTimer(device)

            for _ in range(WARMUP_RUNS):
                _ = emb(indices)

            torch.cuda.synchronize(device)
            times = []
            xfer_times = []

            for _ in range(TIMING_RUNS):
                timer.start()
                _ = emb(indices)
                elapsed = timer.stop()
                times.append(elapsed)

                xfer_timer.start()
                flat_ids = indices.flatten()
                uniq, inv = flat_ids.unique(return_inverse=True)
                cpu_ids = uniq.cpu()
                rows = emb.weight[cpu_ids]
                _ = rows.to(device)
                xfer_elapsed = xfer_timer.stop()
                xfer_times.append(xfer_elapsed)

            stats = BenchmarkStats.from_times(times, total_tokens)
            mean_xfer = float(np.mean(xfer_times))

            bl_key = f"{fixed_tsize}_{fixed_edim}_{bs}_{sl}"
            bl_ms = baseline_lookup.get(bl_key, 0.0)
            slowdown = stats.mean_ms / bl_ms if bl_ms > 0 else float("nan")

            results["configs"].append({
                "part": "B",
                "table_size": fixed_tsize,
                "embedding_dim": fixed_edim,
                "batch_size": bs,
                "seq_len": sl,
                "coalesce_ratio": round(coalesce_ratio, 4),
                "pcie_xfer_ms": round(mean_xfer, 4),
                "slowdown_vs_baseline": round(slowdown, 2),
                **stats.to_dict(),
            })

            rows_b.append([
                str(bs), str(sl),
                f"{stats.mean_ms:.4f}", f"{stats.std_ms:.4f}",
                f"{mean_xfer:.4f}",
                f"{coalesce_ratio:.4f}",
                f"{slowdown:.2f}x",
                f"{stats.throughput_tokens_sec:.0f}",
            ])

            del emb
            torch.cuda.empty_cache()

    _print_table(headers_b, rows_b)

    return results


# ---------------------------------------------------------------------------
# Suite 3: CPU Offload With Async Prefetch
# ---------------------------------------------------------------------------


def suite3_cpu_offload_async(
    device: torch.device,
    baseline_results: Optional[Dict] = None,
    sync_results: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Benchmark async-prefetched CPU->device embedding retrieval.

    The prefetch is issued on a dedicated CUDA stream *before* the embedding
    is consumed.  We simulate "earlier-layer compute" by performing a matmul
    while the prefetch runs, then measure the effective overlap ratio:

        overlap_ratio = (time hidden by overlap) / (total transfer time)

    An overlap_ratio > 0.5 means prefetch is providing real benefit.
    """
    if device.type != "cuda":
        print("\n" + _header("Suite 3: CPU Offload With Async Prefetch"))
        print("  [SKIPPED] CUDA not available; async prefetch benchmarks require GPU.\n")
        return {"suite": 3, "name": "cpu_offload_async", "skipped": True, "configs": []}

    print("\n" + _header("Suite 3: CPU Offload With Async Prefetch"))

    table_sizes = [8191, 65537, 131071, 524287]
    embedding_dims = [128, 256, 512]
    batch_sizes = [1, 4, 16, 64]
    seq_lens = [128, 512, 2048]

    results: Dict[str, Any] = {"suite": 3, "name": "cpu_offload_async", "configs": []}

    # Build lookups for comparison
    baseline_lookup: Dict[str, float] = {}
    sync_lookup: Dict[str, float] = {}
    if baseline_results:
        for cfg in baseline_results.get("configs", []):
            key = f"{cfg.get('table_size')}_{cfg.get('embedding_dim')}_{cfg.get('batch_size')}_{cfg.get('seq_len')}"
            baseline_lookup[key] = cfg.get("mean_ms", 0.0)
    if sync_results:
        for cfg in sync_results.get("configs", []):
            key = f"{cfg.get('table_size')}_{cfg.get('embedding_dim')}_{cfg.get('batch_size')}_{cfg.get('seq_len')}"
            sync_lookup[key] = cfg.get("mean_ms", 0.0)

    def _simulate_earlier_layer_compute(batch: int, dim: int, dev: torch.device) -> torch.Tensor:
        """Simulate compute that overlaps with prefetch."""
        a = torch.randn(batch, dim, dim, device=dev)
        b = torch.randn(batch, dim, dim, device=dev)
        return torch.bmm(a, b)

    # ---- Part A: table_size x embedding_dim --------------------------------
    print("\n  Part A: table_size x embedding_dim  (batch=16, seq_len=512)")
    print(_hr())

    headers_a = ["table_size", "emb_dim", "mean_ms", "overlap_ratio",
                 "vs_baseline", "vs_sync", "tok/sec"]
    rows_a: List[List[str]] = []

    fixed_batch = 16
    fixed_seq = 512
    total_tokens = fixed_batch * fixed_seq

    for tsize in table_sizes:
        for edim in embedding_dims:
            torch.manual_seed(SEED)
            emb = MockOffloadableEmbedding(tsize, edim, mode="offload_async", device=device)
            indices = _gen_indices(fixed_batch, fixed_seq, tsize, device)

            timer = CudaTimer(device)
            xfer_only_timer = CudaTimer(device)
            overlap_timer = CudaTimer(device)

            # warmup
            for _ in range(WARMUP_RUNS):
                emb.prefetch_async(indices)
                _simulate_earlier_layer_compute(fixed_batch, 64, device)
                _ = emb._gather_prefetched(indices)

            torch.cuda.synchronize(device)
            times: List[float] = []
            overlap_ratios: List[float] = []

            for _ in range(TIMING_RUNS):
                # ---- measure total time (prefetch + overlap compute + gather) ----
                timer.start()
                emb.prefetch_async(indices)
                _simulate_earlier_layer_compute(fixed_batch, 64, device)
                out = emb._gather_prefetched(indices)
                total_ms = timer.stop()
                times.append(total_ms)

                # ---- measure transfer-only time (no overlap) ----
                torch.cuda.synchronize(device)
                xfer_only_timer.start()
                emb2 = MockOffloadableEmbedding(tsize, edim, mode="offload_sync", device=device)
                emb2.weight = emb.weight
                _ = emb2._sync_gather(indices)
                xfer_ms = xfer_only_timer.stop()

                # ---- measure compute-only time ----
                torch.cuda.synchronize(device)
                overlap_timer.start()
                _simulate_earlier_layer_compute(fixed_batch, 64, device)
                compute_ms = overlap_timer.stop()

                # overlap_ratio: how much of the transfer time is hidden by compute
                # If total_time < xfer_time + compute_time, some overlap occurred
                sequential_ms = xfer_ms + compute_ms
                time_saved = max(0.0, sequential_ms - total_ms)
                overlap_ratio = time_saved / max(xfer_ms, 0.001)
                overlap_ratio = min(overlap_ratio, 1.0)
                overlap_ratios.append(overlap_ratio)

            stats = BenchmarkStats.from_times(times, total_tokens)
            mean_overlap = float(np.mean(overlap_ratios))

            bl_key = f"{tsize}_{edim}_{fixed_batch}_{fixed_seq}"
            bl_ms = baseline_lookup.get(bl_key, 0.0)
            sync_ms = sync_lookup.get(bl_key, 0.0)
            vs_bl = stats.mean_ms / bl_ms if bl_ms > 0 else float("nan")
            vs_sync = sync_ms / stats.mean_ms if stats.mean_ms > 0 else float("nan")

            results["configs"].append({
                "part": "A",
                "table_size": tsize,
                "embedding_dim": edim,
                "batch_size": fixed_batch,
                "seq_len": fixed_seq,
                "overlap_ratio": round(mean_overlap, 4),
                "speedup_vs_sync": round(vs_sync, 2),
                "slowdown_vs_baseline": round(vs_bl, 2),
                **stats.to_dict(),
            })

            rows_a.append([
                str(tsize), str(edim),
                f"{stats.mean_ms:.4f}",
                f"{mean_overlap:.4f}",
                f"{vs_bl:.2f}x",
                f"{vs_sync:.2f}x speedup",
                f"{stats.throughput_tokens_sec:.0f}",
            ])

            del emb
            torch.cuda.empty_cache()

    _print_table(headers_a, rows_a)

    # ---- Part B: batch_size x seq_len --------------------------------------
    print(f"\n  Part B: batch_size x seq_len  (table_size=131071, emb_dim=256)")
    print(_hr())

    headers_b = ["batch", "seq_len", "mean_ms", "overlap_ratio",
                 "vs_baseline", "vs_sync", "tok/sec"]
    rows_b: List[List[str]] = []

    fixed_tsize = 131071
    fixed_edim = 256

    for bs in batch_sizes:
        for sl in seq_lens:
            total_tokens = bs * sl
            torch.manual_seed(SEED)
            emb = MockOffloadableEmbedding(fixed_tsize, fixed_edim, mode="offload_async", device=device)
            indices = _gen_indices(bs, sl, fixed_tsize, device)

            timer = CudaTimer(device)
            xfer_only_timer = CudaTimer(device)
            overlap_timer = CudaTimer(device)

            for _ in range(WARMUP_RUNS):
                emb.prefetch_async(indices)
                _simulate_earlier_layer_compute(bs, 64, device)
                _ = emb._gather_prefetched(indices)

            torch.cuda.synchronize(device)
            times = []
            overlap_ratios = []

            for _ in range(TIMING_RUNS):
                timer.start()
                emb.prefetch_async(indices)
                _simulate_earlier_layer_compute(bs, 64, device)
                out = emb._gather_prefetched(indices)
                total_ms = timer.stop()
                times.append(total_ms)

                torch.cuda.synchronize(device)
                xfer_only_timer.start()
                emb_tmp = MockOffloadableEmbedding(fixed_tsize, fixed_edim, mode="offload_sync", device=device)
                emb_tmp.weight = emb.weight
                _ = emb_tmp._sync_gather(indices)
                xfer_ms = xfer_only_timer.stop()

                torch.cuda.synchronize(device)
                overlap_timer.start()
                _simulate_earlier_layer_compute(bs, 64, device)
                compute_ms = overlap_timer.stop()

                sequential_ms = xfer_ms + compute_ms
                time_saved = max(0.0, sequential_ms - total_ms)
                overlap_ratio = time_saved / max(xfer_ms, 0.001)
                overlap_ratio = min(overlap_ratio, 1.0)
                overlap_ratios.append(overlap_ratio)

            stats = BenchmarkStats.from_times(times, total_tokens)
            mean_overlap = float(np.mean(overlap_ratios))

            bl_key = f"{fixed_tsize}_{fixed_edim}_{bs}_{sl}"
            bl_ms = baseline_lookup.get(bl_key, 0.0)
            sync_ms = sync_lookup.get(bl_key, 0.0)
            vs_bl = stats.mean_ms / bl_ms if bl_ms > 0 else float("nan")
            vs_sync = sync_ms / stats.mean_ms if stats.mean_ms > 0 else float("nan")

            results["configs"].append({
                "part": "B",
                "table_size": fixed_tsize,
                "embedding_dim": fixed_edim,
                "batch_size": bs,
                "seq_len": sl,
                "overlap_ratio": round(mean_overlap, 4),
                "speedup_vs_sync": round(vs_sync, 2),
                "slowdown_vs_baseline": round(vs_bl, 2),
                **stats.to_dict(),
            })

            rows_b.append([
                str(bs), str(sl),
                f"{stats.mean_ms:.4f}",
                f"{mean_overlap:.4f}",
                f"{vs_bl:.2f}x",
                f"{vs_sync:.2f}x speedup",
                f"{stats.throughput_tokens_sec:.0f}",
            ])

            del emb
            torch.cuda.empty_cache()

    _print_table(headers_b, rows_b)

    return results


# ---------------------------------------------------------------------------
# Suite 4: Coalescing Efficiency
# ---------------------------------------------------------------------------


def suite4_coalescing_efficiency(device: torch.device) -> Dict[str, Any]:
    """
    Measure the impact of ID coalescing (unique + inverse) on transfer efficiency.

    When the unique_ratio is low (many repeated IDs), coalescing dramatically
    reduces the number of rows that must be transferred over PCIe.
    """
    print("\n" + _header("Suite 4: Coalescing Efficiency"))

    unique_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    table_sizes = [65537, 131071, 524287]
    embedding_dims = [128, 256, 512]
    batch_sizes = [4, 16, 64]
    seq_lens = [128, 512, 2048]

    results: Dict[str, Any] = {"suite": 4, "name": "coalescing_efficiency", "configs": []}

    # ---- Part A: unique_ratio sweep (fixed config) -------------------------
    print("\n  Part A: unique_ratio sweep  (table=131071, dim=256, batch=16, seq=512)")
    print(_hr())

    headers_a = [
        "unique_ratio", "total_ids", "unique_ids", "rows_xferred",
        "bytes_xferred", "coalesce_ms", "no_coalesce_ms", "speedup",
    ]
    rows_a: List[List[str]] = []

    fixed_tsize = 131071
    fixed_edim = 256
    fixed_batch = 16
    fixed_seq = 512
    total_ids = fixed_batch * fixed_seq
    bytes_per_row = fixed_edim * 4  # float32

    use_cuda = device.type == "cuda"

    for ur in unique_ratios:
        torch.manual_seed(SEED)
        indices = _gen_indices_with_unique_ratio(fixed_batch, fixed_seq, fixed_tsize, ur, device)

        # Measure actual uniqueness
        flat = indices.flatten()
        actual_unique = flat.unique().numel()
        rows_transferred = actual_unique
        bytes_transferred = rows_transferred * bytes_per_row

        if use_cuda:
            weight_cpu = torch.randn(fixed_tsize, fixed_edim).pin_memory()
        else:
            weight_cpu = torch.randn(fixed_tsize, fixed_edim)

        timer = CudaTimer(device)

        # -- coalesced gather --
        for _ in range(WARMUP_RUNS):
            flat_ids = indices.flatten()
            uniq, inv = flat_ids.unique(return_inverse=True)
            cpu_ids = uniq.cpu()
            rows = weight_cpu[cpu_ids]
            if use_cuda:
                rows = rows.to(device)
            _ = rows[inv].view(*indices.shape, fixed_edim)

        coalesced_times: List[float] = []
        for _ in range(TIMING_RUNS):
            timer.start()
            flat_ids = indices.flatten()
            uniq, inv = flat_ids.unique(return_inverse=True)
            cpu_ids = uniq.cpu()
            rows = weight_cpu[cpu_ids]
            if use_cuda:
                rows = rows.to(device)
            _ = rows[inv].view(*indices.shape, fixed_edim)
            elapsed = timer.stop()
            coalesced_times.append(elapsed)

        # -- non-coalesced gather --
        for _ in range(WARMUP_RUNS):
            cpu_ids_flat = indices.flatten().cpu()
            rows = weight_cpu[cpu_ids_flat]
            if use_cuda:
                rows = rows.to(device)
            _ = rows.view(*indices.shape, fixed_edim)

        no_coalesce_times: List[float] = []
        for _ in range(TIMING_RUNS):
            timer.start()
            cpu_ids_flat = indices.flatten().cpu()
            rows = weight_cpu[cpu_ids_flat]
            if use_cuda:
                rows = rows.to(device)
            _ = rows.view(*indices.shape, fixed_edim)
            elapsed = timer.stop()
            no_coalesce_times.append(elapsed)

        mean_coalesce = float(np.mean(coalesced_times))
        mean_no_coalesce = float(np.mean(no_coalesce_times))
        speedup = mean_no_coalesce / mean_coalesce if mean_coalesce > 0 else float("nan")

        results["configs"].append({
            "part": "A",
            "unique_ratio_target": ur,
            "unique_ratio_actual": round(actual_unique / total_ids, 4),
            "total_ids": total_ids,
            "unique_ids": actual_unique,
            "rows_transferred": rows_transferred,
            "bytes_transferred": bytes_transferred,
            "coalesced_mean_ms": round(mean_coalesce, 4),
            "no_coalesce_mean_ms": round(mean_no_coalesce, 4),
            "coalescing_speedup": round(speedup, 2),
        })

        rows_a.append([
            f"{ur:.1f}",
            str(total_ids),
            str(actual_unique),
            str(rows_transferred),
            _format_bytes(bytes_transferred),
            f"{mean_coalesce:.4f}",
            f"{mean_no_coalesce:.4f}",
            f"{speedup:.2f}x",
        ])

    _print_table(headers_a, rows_a)

    # ---- Part B: table_size x embedding_dim with fixed unique_ratio --------
    print(f"\n  Part B: table_size x embedding_dim  (unique_ratio=0.3, batch=16, seq=512)")
    print(_hr())

    headers_b = [
        "table_size", "emb_dim", "unique_ids", "bytes_xferred",
        "coalesce_ms", "no_coalesce_ms", "speedup",
    ]
    rows_b: List[List[str]] = []

    fixed_ur = 0.3

    for tsize in table_sizes:
        for edim in embedding_dims:
            torch.manual_seed(SEED)
            indices = _gen_indices_with_unique_ratio(fixed_batch, fixed_seq, tsize, fixed_ur, device)
            flat = indices.flatten()
            actual_unique = flat.unique().numel()
            bytes_per_row_b = edim * 4
            bytes_transferred = actual_unique * bytes_per_row_b

            if use_cuda:
                weight_cpu = torch.randn(tsize, edim).pin_memory()
            else:
                weight_cpu = torch.randn(tsize, edim)

            timer = CudaTimer(device)

            for _ in range(WARMUP_RUNS):
                flat_ids = indices.flatten()
                uniq, inv = flat_ids.unique(return_inverse=True)
                cpu_ids = uniq.cpu()
                rows = weight_cpu[cpu_ids]
                if use_cuda:
                    rows = rows.to(device)
                _ = rows[inv].view(*indices.shape, edim)

            coalesced_times = []
            for _ in range(TIMING_RUNS):
                timer.start()
                flat_ids = indices.flatten()
                uniq, inv = flat_ids.unique(return_inverse=True)
                cpu_ids = uniq.cpu()
                rows = weight_cpu[cpu_ids]
                if use_cuda:
                    rows = rows.to(device)
                _ = rows[inv].view(*indices.shape, edim)
                elapsed = timer.stop()
                coalesced_times.append(elapsed)

            for _ in range(WARMUP_RUNS):
                cpu_flat = indices.flatten().cpu()
                rows = weight_cpu[cpu_flat]
                if use_cuda:
                    rows = rows.to(device)
                _ = rows.view(*indices.shape, edim)

            no_coalesce_times = []
            for _ in range(TIMING_RUNS):
                timer.start()
                cpu_flat = indices.flatten().cpu()
                rows = weight_cpu[cpu_flat]
                if use_cuda:
                    rows = rows.to(device)
                _ = rows.view(*indices.shape, edim)
                elapsed = timer.stop()
                no_coalesce_times.append(elapsed)

            mean_c = float(np.mean(coalesced_times))
            mean_nc = float(np.mean(no_coalesce_times))
            sp = mean_nc / mean_c if mean_c > 0 else float("nan")

            results["configs"].append({
                "part": "B",
                "table_size": tsize,
                "embedding_dim": edim,
                "unique_ratio_target": fixed_ur,
                "unique_ids": actual_unique,
                "bytes_transferred": bytes_transferred,
                "coalesced_mean_ms": round(mean_c, 4),
                "no_coalesce_mean_ms": round(mean_nc, 4),
                "coalescing_speedup": round(sp, 2),
            })

            rows_b.append([
                str(tsize), str(edim),
                str(actual_unique),
                _format_bytes(bytes_transferred),
                f"{mean_c:.4f}",
                f"{mean_nc:.4f}",
                f"{sp:.2f}x",
            ])

    _print_table(headers_b, rows_b)

    # ---- Part C: batch_size x seq_len with different unique_ratios ---------
    print(f"\n  Part C: batch x seq_len  (table=131071, dim=256, unique_ratio=0.3)")
    print(_hr())

    headers_c = [
        "batch", "seq_len", "total_ids", "unique_ids", "bytes_xferred",
        "coalesce_ms", "no_coalesce_ms", "speedup",
    ]
    rows_c: List[List[str]] = []

    for bs in batch_sizes:
        for sl in seq_lens:
            total_ids_c = bs * sl
            torch.manual_seed(SEED)
            indices = _gen_indices_with_unique_ratio(bs, sl, fixed_tsize, fixed_ur, device)
            flat = indices.flatten()
            actual_unique = flat.unique().numel()
            bytes_transferred = actual_unique * bytes_per_row

            if use_cuda:
                w = torch.randn(fixed_tsize, fixed_edim).pin_memory()
            else:
                w = torch.randn(fixed_tsize, fixed_edim)

            timer = CudaTimer(device)

            for _ in range(WARMUP_RUNS):
                flat_ids = indices.flatten()
                uniq, inv = flat_ids.unique(return_inverse=True)
                rows = w[uniq.cpu()]
                if use_cuda:
                    rows = rows.to(device)
                _ = rows[inv].view(*indices.shape, fixed_edim)

            coalesced_times = []
            for _ in range(TIMING_RUNS):
                timer.start()
                flat_ids = indices.flatten()
                uniq, inv = flat_ids.unique(return_inverse=True)
                rows = w[uniq.cpu()]
                if use_cuda:
                    rows = rows.to(device)
                _ = rows[inv].view(*indices.shape, fixed_edim)
                elapsed = timer.stop()
                coalesced_times.append(elapsed)

            for _ in range(WARMUP_RUNS):
                rows = w[indices.flatten().cpu()]
                if use_cuda:
                    rows = rows.to(device)
                _ = rows.view(*indices.shape, fixed_edim)

            no_coalesce_times = []
            for _ in range(TIMING_RUNS):
                timer.start()
                rows = w[indices.flatten().cpu()]
                if use_cuda:
                    rows = rows.to(device)
                _ = rows.view(*indices.shape, fixed_edim)
                elapsed = timer.stop()
                no_coalesce_times.append(elapsed)

            mean_c = float(np.mean(coalesced_times))
            mean_nc = float(np.mean(no_coalesce_times))
            sp = mean_nc / mean_c if mean_c > 0 else float("nan")

            results["configs"].append({
                "part": "C",
                "batch_size": bs,
                "seq_len": sl,
                "total_ids": total_ids_c,
                "unique_ids": actual_unique,
                "bytes_transferred": bytes_transferred,
                "coalesced_mean_ms": round(mean_c, 4),
                "no_coalesce_mean_ms": round(mean_nc, 4),
                "coalescing_speedup": round(sp, 2),
            })

            rows_c.append([
                str(bs), str(sl), str(total_ids_c),
                str(actual_unique),
                _format_bytes(bytes_transferred),
                f"{mean_c:.4f}",
                f"{mean_nc:.4f}",
                f"{sp:.2f}x",
            ])

    _print_table(headers_c, rows_c)

    return results


# ---------------------------------------------------------------------------
# Suite 5: Multi-Head Hash Throughput
# ---------------------------------------------------------------------------


def suite5_hash_throughput(device: torch.device) -> Dict[str, Any]:
    """
    Benchmark hash computation for different configurations.

    Compares vectorized torch implementation against naive Python-loop version.
    """
    print("\n" + _header("Suite 5: Multi-Head Hash Throughput"))

    ngram_orders = [2, 3, 4, 5]
    heads_per_order_list = [1, 2, 4]
    seq_lens = [128, 512, 2048]
    batch_sizes = [1, 4, 16]
    table_size = 131071

    results: Dict[str, Any] = {"suite": 5, "name": "hash_throughput", "configs": []}

    # ---- Part A: ngram_order x num_heads (fixed batch=16, seq=512) ---------
    print("\n  Part A: ngram_order x num_heads  (batch=16, seq=512, table=131071)")
    print(_hr())

    headers_a = [
        "ngram_order", "num_heads", "vec_ms", "naive_ms", "speedup",
        "hashes/sec", "hashes/sec_naive",
    ]
    rows_a: List[List[str]] = []

    fixed_batch = 16
    fixed_seq = 512

    for n_order in ngram_orders:
        for n_heads in heads_per_order_list:
            torch.manual_seed(SEED)
            hasher = MockMultiHeadHash(n_order, n_heads, table_size, seed=SEED)

            token_ids = torch.randint(0, 50000, (fixed_batch, fixed_seq), dtype=torch.long, device=device)
            # Build ngrams
            padded = F.pad(token_ids, (n_order - 1, 0), value=0)
            ngrams = torch.stack([padded[:, i:i + fixed_seq] for i in range(n_order)], dim=-1)

            total_hashes = fixed_batch * fixed_seq * n_heads
            timer = CudaTimer(device)

            # -- vectorized --
            for _ in range(WARMUP_RUNS):
                _ = hasher.hash(ngrams)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            vec_times: List[float] = []
            for _ in range(TIMING_RUNS):
                timer.start()
                _ = hasher.hash(ngrams)
                elapsed = timer.stop()
                vec_times.append(elapsed)

            # -- naive (loop-based) --
            # Only run naive for small configs to avoid extreme slowness
            run_naive = (fixed_batch * fixed_seq <= 8192 and n_heads <= 4 and n_order <= 5)
            naive_times: List[float] = []

            if run_naive:
                for _ in range(min(WARMUP_RUNS, 2)):
                    _ = hasher.hash_naive(ngrams)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                naive_runs = min(TIMING_RUNS, 10)
                for _ in range(naive_runs):
                    timer.start()
                    _ = hasher.hash_naive(ngrams)
                    elapsed = timer.stop()
                    naive_times.append(elapsed)

            mean_vec = float(np.mean(vec_times))
            mean_naive = float(np.mean(naive_times)) if naive_times else float("nan")
            speedup = mean_naive / mean_vec if mean_vec > 0 and not math.isnan(mean_naive) else float("nan")

            hashes_sec_vec = total_hashes / (mean_vec / 1000.0) if mean_vec > 0 else 0.0
            hashes_sec_naive = (
                total_hashes / (mean_naive / 1000.0)
                if not math.isnan(mean_naive) and mean_naive > 0
                else 0.0
            )

            results["configs"].append({
                "part": "A",
                "ngram_order": n_order,
                "num_heads": n_heads,
                "batch_size": fixed_batch,
                "seq_len": fixed_seq,
                "vectorized_mean_ms": round(mean_vec, 4),
                "naive_mean_ms": round(mean_naive, 4) if not math.isnan(mean_naive) else "N/A",
                "speedup": round(speedup, 2) if not math.isnan(speedup) else "N/A",
                "hashes_per_sec_vec": round(hashes_sec_vec, 0),
                "hashes_per_sec_naive": round(hashes_sec_naive, 0) if hashes_sec_naive > 0 else "N/A",
            })

            naive_str = f"{mean_naive:.4f}" if not math.isnan(mean_naive) else "N/A"
            sp_str = f"{speedup:.2f}x" if not math.isnan(speedup) else "N/A"
            hn_str = f"{hashes_sec_naive:.0f}" if hashes_sec_naive > 0 else "N/A"

            rows_a.append([
                str(n_order), str(n_heads),
                f"{mean_vec:.4f}", naive_str, sp_str,
                f"{hashes_sec_vec:.0f}", hn_str,
            ])

    _print_table(headers_a, rows_a)

    # ---- Part B: seq_len x batch_size scaling (fixed order=3, heads=2) -----
    print(f"\n  Part B: seq_len x batch_size scaling  (order=3, heads=2, table=131071)")
    print(_hr())

    headers_b = ["batch", "seq_len", "total_hashes", "vec_ms", "hashes/sec"]
    rows_b: List[List[str]] = []

    fixed_order = 3
    fixed_heads = 2
    hasher_b = MockMultiHeadHash(fixed_order, fixed_heads, table_size, seed=SEED)

    for bs in batch_sizes:
        for sl in seq_lens:
            torch.manual_seed(SEED)
            token_ids = torch.randint(0, 50000, (bs, sl), dtype=torch.long, device=device)
            padded = F.pad(token_ids, (fixed_order - 1, 0), value=0)
            ngrams = torch.stack([padded[:, i:i + sl] for i in range(fixed_order)], dim=-1)

            total_hashes = bs * sl * fixed_heads
            timer = CudaTimer(device)

            for _ in range(WARMUP_RUNS):
                _ = hasher_b.hash(ngrams)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            vec_times = []
            for _ in range(TIMING_RUNS):
                timer.start()
                _ = hasher_b.hash(ngrams)
                elapsed = timer.stop()
                vec_times.append(elapsed)

            mean_vec = float(np.mean(vec_times))
            hps = total_hashes / (mean_vec / 1000.0) if mean_vec > 0 else 0.0

            results["configs"].append({
                "part": "B",
                "ngram_order": fixed_order,
                "num_heads": fixed_heads,
                "batch_size": bs,
                "seq_len": sl,
                "total_hashes": total_hashes,
                "vectorized_mean_ms": round(mean_vec, 4),
                "hashes_per_sec": round(hps, 0),
            })

            rows_b.append([
                str(bs), str(sl), str(total_hashes),
                f"{mean_vec:.4f}",
                f"{hps:.0f}",
            ])

    _print_table(headers_b, rows_b)

    # ---- Part C: vectorization benefit across increasing order -------------
    print(f"\n  Part C: vectorization benefit  (batch=4, seq=512, heads=2)")
    print(_hr())

    headers_c = ["order", "vec_ms", "naive_ms", "speedup", "hashes/sec_vec"]
    rows_c: List[List[str]] = []

    for n_order in [2, 3, 4, 5]:
        torch.manual_seed(SEED)
        hasher_c = MockMultiHeadHash(n_order, 2, table_size, seed=SEED)
        tids = torch.randint(0, 50000, (4, 512), dtype=torch.long, device=device)
        padded = F.pad(tids, (n_order - 1, 0), value=0)
        ngrams = torch.stack([padded[:, i:i + 512] for i in range(n_order)], dim=-1)
        total_h = 4 * 512 * 2

        timer = CudaTimer(device)

        # vec
        for _ in range(WARMUP_RUNS):
            _ = hasher_c.hash(ngrams)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        vec_times = []
        for _ in range(TIMING_RUNS):
            timer.start()
            _ = hasher_c.hash(ngrams)
            elapsed = timer.stop()
            vec_times.append(elapsed)

        # naive
        for _ in range(min(WARMUP_RUNS, 2)):
            _ = hasher_c.hash_naive(ngrams)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        naive_times = []
        for _ in range(min(TIMING_RUNS, 10)):
            timer.start()
            _ = hasher_c.hash_naive(ngrams)
            elapsed = timer.stop()
            naive_times.append(elapsed)

        mv = float(np.mean(vec_times))
        mn = float(np.mean(naive_times))
        sp = mn / mv if mv > 0 else float("nan")
        hps = total_h / (mv / 1000.0) if mv > 0 else 0.0

        results["configs"].append({
            "part": "C",
            "ngram_order": n_order,
            "vectorized_mean_ms": round(mv, 4),
            "naive_mean_ms": round(mn, 4),
            "speedup": round(sp, 2),
            "hashes_per_sec_vec": round(hps, 0),
        })

        rows_c.append([
            str(n_order),
            f"{mv:.4f}", f"{mn:.4f}",
            f"{sp:.2f}x",
            f"{hps:.0f}",
        ])

    _print_table(headers_c, rows_c)

    return results


# ---------------------------------------------------------------------------
# Suite 6: End-to-End EngramModule
# ---------------------------------------------------------------------------


def suite6_end_to_end(device: torch.device) -> Dict[str, Any]:
    """
    Full pipeline benchmark: hash -> retrieve -> aggregate -> gate -> conv -> project.

    Reports per-component time breakdown and peak GPU memory for three modes.
    """
    print("\n" + _header("Suite 6: End-to-End EngramModule"))

    hidden_dims = [256, 512, 1024]
    seq_lens = [128, 512, 2048]
    modes = ["on_device"]
    if device.type == "cuda":
        modes.extend(["offload_sync", "offload_async"])

    # Fixed parameters
    embedding_dim = 256
    table_size = 65537
    ngram_orders = (2, 3)
    num_heads = 2
    fixed_batch = 8

    results: Dict[str, Any] = {"suite": 6, "name": "end_to_end_engram", "configs": []}

    # ---- Part A: hidden_dim x seq_len per mode -----------------------------
    for mode in modes:
        mode_label = {
            "on_device": "On-Device",
            "offload_sync": "Offload (Sync)",
            "offload_async": "Offload (Async Prefetch)",
        }[mode]

        print(f"\n  Mode: {mode_label}  (batch={fixed_batch}, table={table_size})")
        print(_hr())

        headers = [
            "hidden_dim", "seq_len", "total_ms", "hash_ms", "retrieve_ms",
            "gate_ms", "conv_ms", "project_ms", "peak_mem",
        ]
        rows: List[List[str]] = []

        for hdim in hidden_dims:
            for sl in seq_lens:
                # Adjust embedding_dim to match heads
                total_heads = len(ngram_orders) * num_heads
                actual_edim = (total_heads * (embedding_dim // total_heads))

                torch.manual_seed(SEED)

                try:
                    module = MockEngramModule(
                        hidden_dim=hdim,
                        embedding_dim=actual_edim,
                        table_size=table_size,
                        ngram_orders=ngram_orders,
                        num_heads=num_heads,
                        mode=mode,
                        device=device,
                    )
                    module.train(False)
                except Exception as e:
                    print(f"  [ERROR] hidden_dim={hdim}, seq_len={sl}: {e}")
                    continue

                token_ids = _gen_token_ids(fixed_batch, sl, 50000, device)
                hidden = _gen_hidden(fixed_batch, sl, hdim, device)
                total_tokens = fixed_batch * sl

                timer = CudaTimer(device)

                # warmup
                with torch.no_grad():
                    for _ in range(WARMUP_RUNS):
                        if mode == "offload_async":
                            module.prefetch_all(token_ids)
                        _ = module.forward_timed(token_ids, hidden)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                _reset_peak_gpu_mem(device)

                total_times: List[float] = []
                component_times: Dict[str, List[float]] = {
                    "hash": [], "retrieve": [], "gate": [], "conv": [], "project": [],
                }

                with torch.no_grad():
                    for _ in range(TIMING_RUNS):
                        if mode == "offload_async":
                            module.prefetch_all(token_ids)

                        timer.start()
                        out, ct = module.forward_timed(token_ids, hidden)
                        total_ms = timer.stop()
                        total_times.append(total_ms)

                        for comp_name, comp_val in ct.items():
                            component_times.setdefault(comp_name, []).append(comp_val)

                peak_mem = _get_peak_gpu_mem(device)

                mean_total = float(np.mean(total_times))
                comp_means: Dict[str, float] = {}
                for cname, cvals in component_times.items():
                    comp_means[cname] = float(np.mean(cvals))

                throughput = total_tokens / (mean_total / 1000.0) if mean_total > 0 else 0.0

                cfg_result: Dict[str, Any] = {
                    "mode": mode,
                    "hidden_dim": hdim,
                    "seq_len": sl,
                    "batch_size": fixed_batch,
                    "total_mean_ms": round(mean_total, 4),
                    "throughput_tokens_sec": round(throughput, 0),
                    "peak_gpu_mem_bytes": peak_mem,
                }
                for cname, cmean in comp_means.items():
                    cfg_result[f"{cname}_mean_ms"] = round(cmean, 4)
                    if mean_total > 0:
                        cfg_result[f"{cname}_pct"] = round(100.0 * cmean / mean_total, 1)

                results["configs"].append(cfg_result)

                rows.append([
                    str(hdim), str(sl),
                    f"{mean_total:.4f}",
                    f"{comp_means.get('hash', 0):.4f}",
                    f"{comp_means.get('retrieve', 0):.4f}",
                    f"{comp_means.get('gate', 0):.4f}",
                    f"{comp_means.get('conv', 0):.4f}",
                    f"{comp_means.get('project', 0):.4f}",
                    _format_bytes(peak_mem),
                ])

                del module
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        _print_table(headers, rows)

    # ---- Part B: Component percentage breakdown (best config per mode) -----
    print(f"\n  Component Time Breakdown (% of total)")
    print(_hr())

    headers_pct = ["mode", "hidden_dim", "seq_len", "hash%", "retrieve%", "gate%", "conv%", "project%", "bottleneck"]
    rows_pct: List[List[str]] = []

    for cfg in results["configs"]:
        total = cfg.get("total_mean_ms", 0.0)
        if total <= 0:
            continue

        comp_pcts: Dict[str, float] = {}
        for comp in ["hash", "retrieve", "gate", "conv", "project"]:
            comp_pcts[comp] = cfg.get(f"{comp}_pct", 0.0)

        bottleneck = max(comp_pcts, key=lambda k: comp_pcts[k]) if comp_pcts else "N/A"

        rows_pct.append([
            cfg.get("mode", ""),
            str(cfg.get("hidden_dim", "")),
            str(cfg.get("seq_len", "")),
            f"{comp_pcts.get('hash', 0):.1f}%",
            f"{comp_pcts.get('retrieve', 0):.1f}%",
            f"{comp_pcts.get('gate', 0):.1f}%",
            f"{comp_pcts.get('conv', 0):.1f}%",
            f"{comp_pcts.get('project', 0):.1f}%",
            bottleneck,
        ])

    _print_table(headers_pct, rows_pct)

    # ---- Part C: Memory profile per mode (fixed hidden=512, seq=512) -------
    print(f"\n  Memory Profile (hidden=512, seq=512, batch={fixed_batch})")
    print(_hr())

    headers_mem = ["mode", "peak_gpu_mem", "table_params", "table_mem_est"]
    rows_mem: List[List[str]] = []

    total_heads = len(ngram_orders) * num_heads
    dim_per_head = embedding_dim // total_heads
    table_params = total_heads * table_size * dim_per_head
    table_mem_est = table_params * 4  # float32

    for mode in modes:
        for cfg in results["configs"]:
            if cfg.get("mode") == mode and cfg.get("hidden_dim") == 512 and cfg.get("seq_len") == 512:
                rows_mem.append([
                    mode,
                    _format_bytes(cfg.get("peak_gpu_mem_bytes", 0)),
                    f"{table_params:,}",
                    _format_bytes(table_mem_est),
                ])
                break

    _print_table(headers_mem, rows_mem)

    # ---- Part D: Cross-mode comparison -------------------------------------
    if device.type == "cuda" and len(modes) > 1:
        print(f"\n  Cross-Mode Comparison (hidden=512, seq=512)")
        print(_hr())

        headers_cmp = ["mode", "total_ms", "vs_on_device", "throughput_tok/s"]
        rows_cmp: List[List[str]] = []

        on_device_ms = None
        for cfg in results["configs"]:
            if cfg.get("mode") == "on_device" and cfg.get("hidden_dim") == 512 and cfg.get("seq_len") == 512:
                on_device_ms = cfg.get("total_mean_ms", 0.0)
                break

        for mode in modes:
            for cfg in results["configs"]:
                if cfg.get("mode") == mode and cfg.get("hidden_dim") == 512 and cfg.get("seq_len") == 512:
                    t_ms = cfg.get("total_mean_ms", 0.0)
                    ratio = t_ms / on_device_ms if on_device_ms and on_device_ms > 0 else float("nan")
                    tp = cfg.get("throughput_tokens_sec", 0.0)
                    rows_cmp.append([
                        mode,
                        f"{t_ms:.4f}",
                        f"{ratio:.2f}x",
                        f"{tp:.0f}",
                    ])
                    break

        _print_table(headers_cmp, rows_cmp)

    return results


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def _generate_summary(all_results: Dict[str, Any]) -> str:
    """Produce a concise summary of key findings across all suites."""
    lines = [
        "",
        _header("BENCHMARK SUMMARY"),
        "",
    ]

    # Suite 1 key stat: best throughput
    s1 = all_results.get("suite_1", {})
    if s1.get("configs"):
        best = max(s1["configs"], key=lambda c: c.get("throughput_tokens_sec", 0))
        lines.append(
            f"  [Suite 1] On-device baseline peak throughput: "
            f"{best['throughput_tokens_sec']:.0f} tok/sec "
            f"(batch={best.get('batch_size')}, seq={best.get('seq_len')}, "
            f"table={best.get('table_size')}, dim={best.get('embedding_dim')})"
        )

    # Suite 2 key stat: average slowdown
    s2 = all_results.get("suite_2", {})
    if s2.get("configs"):
        slowdowns = [c["slowdown_vs_baseline"] for c in s2["configs"]
                     if isinstance(c.get("slowdown_vs_baseline"), (int, float))
                     and not math.isnan(c.get("slowdown_vs_baseline", float("nan")))]
        if slowdowns:
            avg_slow = float(np.mean(slowdowns))
            lines.append(
                f"  [Suite 2] Sync offload average slowdown vs on-device: {avg_slow:.2f}x"
            )

    # Suite 3 key stats: average overlap ratio and speedup vs sync
    s3 = all_results.get("suite_3", {})
    if s3.get("configs"):
        overlap_vals = [c["overlap_ratio"] for c in s3["configs"]
                        if isinstance(c.get("overlap_ratio"), (int, float))]
        speedup_vals = [c["speedup_vs_sync"] for c in s3["configs"]
                        if isinstance(c.get("speedup_vs_sync"), (int, float))
                        and not math.isnan(c.get("speedup_vs_sync", float("nan")))]
        if overlap_vals:
            avg_overlap = float(np.mean(overlap_vals))
            lines.append(
                f"  [Suite 3] Async prefetch average overlap ratio: {avg_overlap:.3f} "
                f"({'effective' if avg_overlap > 0.5 else 'marginal'} overlap)"
            )
        if speedup_vals:
            avg_sp = float(np.mean(speedup_vals))
            lines.append(
                f"  [Suite 3] Prefetch provides {avg_sp:.2f}x average speedup over sync offload"
            )

    # Suite 4 key stat: max coalescing speedup
    s4 = all_results.get("suite_4", {})
    if s4.get("configs"):
        sp_vals = [c.get("coalescing_speedup", 0.0) for c in s4["configs"]
                   if isinstance(c.get("coalescing_speedup"), (int, float))
                   and not math.isnan(c.get("coalescing_speedup", float("nan")))]
        if sp_vals:
            max_sp = max(sp_vals)
            min_ur_cfg = min(
                [c for c in s4["configs"] if c.get("part") == "A"],
                key=lambda c: c.get("unique_ratio_target", 1.0),
                default=None,
            )
            ur_note = f" (at unique_ratio={min_ur_cfg['unique_ratio_target']})" if min_ur_cfg else ""
            lines.append(
                f"  [Suite 4] Max coalescing speedup: {max_sp:.2f}x{ur_note}"
            )

    # Suite 5 key stat: vectorization benefit
    s5 = all_results.get("suite_5", {})
    if s5.get("configs"):
        sp_vals = [
            c["speedup"]
            for c in s5["configs"]
            if c.get("part") == "C" and isinstance(c.get("speedup"), (int, float))
        ]
        if sp_vals:
            avg_sp = float(np.mean(sp_vals))
            lines.append(
                f"  [Suite 5] Vectorized hash avg speedup over naive loop: {avg_sp:.1f}x"
            )
        best_hps = max(
            (c.get("hashes_per_sec_vec", 0) for c in s5["configs"]
             if isinstance(c.get("hashes_per_sec_vec"), (int, float))),
            default=0,
        )
        if best_hps > 0:
            lines.append(
                f"  [Suite 5] Peak hash throughput: {best_hps:.0f} hashes/sec"
            )

    # Suite 6 key stat: bottleneck identification and mode comparison
    s6 = all_results.get("suite_6", {})
    if s6.get("configs"):
        # Find most common bottleneck
        bottlenecks: Dict[str, int] = {}
        for cfg in s6["configs"]:
            comps = {}
            for comp in ["hash", "retrieve", "gate", "conv", "project"]:
                comps[comp] = cfg.get(f"{comp}_pct", 0.0)
            if comps:
                bn = max(comps, key=lambda k: comps[k])
                bottlenecks[bn] = bottlenecks.get(bn, 0) + 1

        if bottlenecks:
            primary_bn = max(bottlenecks, key=lambda k: bottlenecks[k])
            lines.append(
                f"  [Suite 6] Most common bottleneck component: {primary_bn} "
                f"({bottlenecks[primary_bn]}/{len(s6['configs'])} configs)"
            )

        # Cross-mode speedup
        on_device_cfgs = [c for c in s6["configs"] if c.get("mode") == "on_device"]
        async_cfgs = [c for c in s6["configs"] if c.get("mode") == "offload_async"]
        if on_device_cfgs and async_cfgs:
            on_avg = float(np.mean([c["total_mean_ms"] for c in on_device_cfgs]))
            async_avg = float(np.mean([c["total_mean_ms"] for c in async_cfgs]))
            ratio = async_avg / on_avg if on_avg > 0 else float("nan")
            lines.append(
                f"  [Suite 6] End-to-end async prefetch overhead vs on-device: {ratio:.2f}x"
            )

    lines.append("")
    lines.append(_hr("="))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Engram Conditional Memory -- Offload Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        default=["all"],
        help='Suite(s) to run: 1-6 or "all" (default: all)',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to benchmark on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON (optional)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=WARMUP_RUNS,
        help=f"Number of warmup runs (default: {WARMUP_RUNS})",
    )
    parser.add_argument(
        "--timing-runs",
        type=int,
        default=TIMING_RUNS,
        help=f"Number of timing runs (default: {TIMING_RUNS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility (default: {SEED})",
    )
    return parser.parse_args()


def _resolve_suites(raw: List[str]) -> List[int]:
    """Parse suite arguments into a sorted list of suite numbers."""
    suites = set()
    for s in raw:
        s = s.strip().lower()
        if s == "all":
            return [1, 2, 3, 4, 5, 6]
        try:
            n = int(s)
            if 1 <= n <= 6:
                suites.add(n)
            else:
                print(f"WARNING: ignoring invalid suite number {n} (must be 1-6)")
        except ValueError:
            print(f"WARNING: ignoring invalid suite argument '{s}'")
    return sorted(suites)


def main() -> None:
    args = _parse_args()

    # Override globals
    global WARMUP_RUNS, TIMING_RUNS, SEED
    WARMUP_RUNS = args.warmup_runs
    TIMING_RUNS = args.timing_runs
    SEED = args.seed

    # Seed everything
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(args.device)
    suites = _resolve_suites(args.suite)

    if not suites:
        print("ERROR: no valid suites specified. Use --suite 1-6 or --suite all.")
        sys.exit(1)

    # Print banner
    print(_header("Engram Conditional Memory -- Offload Benchmark", width=100))
    print(f"  Device:       {device}")
    if device.type == "cuda":
        print(f"  GPU:          {torch.cuda.get_device_name(device)}")
        print(f"  CUDA version: {torch.version.cuda}")
    print(f"  PyTorch:      {torch.__version__}")
    print(f"  Suites:       {suites}")
    print(f"  Warmup runs:  {WARMUP_RUNS}")
    print(f"  Timing runs:  {TIMING_RUNS}")
    print(f"  Seed:         {SEED}")
    print(_hr("="))

    all_results: Dict[str, Any] = {
        "device": str(device),
        "pytorch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda) if device.type == "cuda" else None,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "warmup_runs": WARMUP_RUNS,
        "timing_runs": TIMING_RUNS,
        "seed": SEED,
    }

    baseline_results: Optional[Dict] = None
    sync_results: Optional[Dict] = None

    # Run suites
    if 1 in suites:
        baseline_results = suite1_embedding_gather_baseline(device)
        all_results["suite_1"] = baseline_results

    if 2 in suites:
        # Suite 2 needs baseline for slowdown comparison
        if baseline_results is None and 1 not in suites:
            print("\n  [INFO] Running Suite 1 (baseline) first for Suite 2 comparison...")
            baseline_results = suite1_embedding_gather_baseline(device)
            all_results["suite_1"] = baseline_results

        sync_results = suite2_cpu_offload_sync(device, baseline_results)
        all_results["suite_2"] = sync_results

    if 3 in suites:
        # Suite 3 needs both baseline and sync for comparison
        if baseline_results is None and 1 not in suites:
            print("\n  [INFO] Running Suite 1 (baseline) first for Suite 3 comparison...")
            baseline_results = suite1_embedding_gather_baseline(device)
            all_results["suite_1"] = baseline_results
        if sync_results is None and 2 not in suites:
            print("\n  [INFO] Running Suite 2 (sync offload) first for Suite 3 comparison...")
            sync_results = suite2_cpu_offload_sync(device, baseline_results)
            all_results["suite_2"] = sync_results

        s3_results = suite3_cpu_offload_async(device, baseline_results, sync_results)
        all_results["suite_3"] = s3_results

    if 4 in suites:
        s4_results = suite4_coalescing_efficiency(device)
        all_results["suite_4"] = s4_results

    if 5 in suites:
        s5_results = suite5_hash_throughput(device)
        all_results["suite_5"] = s5_results

    if 6 in suites:
        s6_results = suite6_end_to_end(device)
        all_results["suite_6"] = s6_results

    # Summary
    summary = _generate_summary(all_results)
    print(summary)

    # Save to JSON
    if args.output:
        output_path = os.path.abspath(args.output)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Results saved to: {output_path}")
        print()


if __name__ == "__main__":
    main()
