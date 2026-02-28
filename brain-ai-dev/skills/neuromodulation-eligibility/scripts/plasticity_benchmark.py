#!/usr/bin/env python3
"""
plasticity_benchmark.py -- Benchmark for neuromodulation + eligibility traces pipeline.

Self-contained benchmark script that measures throughput and latency of key
operations in the three-factor learning pipeline:

  1. Eligibility trace update throughput (accumulating, replacing, Dutch)
  2. Trace decay overhead (decay-only vs full update, clamp cost)
  3. Neuromodulator computation (DA, ACh, NE, 5-HT, combination functions)
  4. Three-factor weight update (online vs auxiliary loss, clamping overhead)
  5. Full pipeline (trace -> modulator -> update, multi-layer scaling)
  6. Memory profile (peak memory for traces, modulators, diagnostics)

Usage:
    # Run all suites on CPU
    python plasticity_benchmark.py

    # Run specific suite(s)
    python plasticity_benchmark.py --suite 1
    python plasticity_benchmark.py --suite 1 3 5

    # Run on CUDA
    python plasticity_benchmark.py --device cuda

    # Quick mode (fewer iterations, smaller sizes)
    python plasticity_benchmark.py --quick

    # JSON output
    python plasticity_benchmark.py --json

    # Combined
    python plasticity_benchmark.py --suite 1 2 --device cuda --json --quick

Dependencies:
    torch, numpy, time, json, argparse (all stdlib or standard ML stack).
    No brain_ai imports -- all stubs are inline.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUITE_NAMES = {
    1: "Eligibility Trace Update Throughput",
    2: "Trace Decay Overhead",
    3: "Neuromodulator Computation",
    4: "Three-Factor Weight Update",
    5: "Full Pipeline (Trace -> Modulator -> Update)",
    6: "Memory Profile",
}

WARMUP_ITERS = 5
DEFAULT_ITERS = 50
QUICK_ITERS = 10

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


class TimerResult:
    """Holds timing results for a single benchmark."""

    def __init__(self, name: str):
        self.name = name
        self.times_ms: List[float] = []

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.times_ms)) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        return float(np.std(self.times_ms)) if self.times_ms else 0.0

    @property
    def min_ms(self) -> float:
        return float(np.min(self.times_ms)) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return float(np.max(self.times_ms)) if self.times_ms else 0.0

    @property
    def median_ms(self) -> float:
        return float(np.median(self.times_ms)) if self.times_ms else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean_ms": round(self.mean_ms, 4),
            "std_ms": round(self.std_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "median_ms": round(self.median_ms, 4),
            "n_samples": len(self.times_ms),
        }


@contextmanager
def cuda_sync_timer(device: torch.device):
    """Context manager that yields a callable returning elapsed ms.

    Handles CUDA synchronization if on GPU.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        elapsed_holder = [0.0]

        def get_elapsed():
            return elapsed_holder[0]

        try:
            yield get_elapsed
        finally:
            end.record()
            torch.cuda.synchronize(device)
            elapsed_holder[0] = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        elapsed_holder = [0.0]

        def get_elapsed():
            return elapsed_holder[0]

        try:
            yield get_elapsed
        finally:
            elapsed_holder[0] = (time.perf_counter() - t0) * 1000.0


def time_fn(
    fn: Callable[[], Any],
    device: torch.device,
    n_warmup: int = WARMUP_ITERS,
    n_iter: int = DEFAULT_ITERS,
    label: str = "",
) -> TimerResult:
    """Time a callable with warmup, returning a TimerResult."""
    result = TimerResult(label)

    # Warmup
    for _ in range(n_warmup):
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Measured runs
    for _ in range(n_iter):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize(device)
            elapsed = start_event.elapsed_time(end_event)
        else:
            t0 = time.perf_counter()
            fn()
            elapsed = (time.perf_counter() - t0) * 1000.0
        result.times_ms.append(elapsed)

    return result


def tensor_memory_mb(t: torch.Tensor) -> float:
    """Return memory footprint of a tensor in MB."""
    return t.element_size() * t.nelement() / (1024.0 * 1024.0)


def peak_memory_mb(device: torch.device) -> float:
    """Return peak allocated memory in MB (CUDA only). Returns 0 on CPU."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    return 0.0


def reset_peak_memory(device: torch.device):
    """Reset peak memory stats (CUDA only)."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def format_table(
    headers: List[str],
    rows: List[List[str]],
    col_widths: Optional[List[int]] = None,
) -> str:
    """Format a simple ASCII table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(row[i]))
            col_widths.append(max_w + 2)

    def fmt_row(cells: List[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            w = col_widths[i] if i < len(col_widths) else 20
            parts.append(cell.ljust(w))
        return "| " + " | ".join(parts) + " |"

    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    lines = [separator, fmt_row(headers), separator]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(separator)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inline Stubs -- EligibilityTraceModule
# ---------------------------------------------------------------------------


class TraceType(Enum):
    ACCUMULATING = "accumulating"
    REPLACING = "replacing"
    DUTCH = "dutch"


class KernelType(Enum):
    RATE = "rate"
    STDP = "stdp"


class EligibilityTraceModule(nn.Module):
    """Self-contained eligibility trace module for benchmarking.

    Supports accumulating, replacing, and Dutch trace types with
    rate-based (Hebbian outer product) and STDP spike-based kernels.
    All trace computation is fp32.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        tau_e: float = 100.0,
        dt: float = 1.0,
        trace_type: TraceType = TraceType.ACCUMULATING,
        kernel_type: KernelType = KernelType.RATE,
        alpha: float = 0.1,
        clamp_min: float = -5.0,
        clamp_max: float = 5.0,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.tau_e = tau_e
        self.dt = dt
        self.trace_type = trace_type
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Decay factors
        self.decay_e = math.exp(-dt / tau_e)
        self.decay_pre = math.exp(-dt / tau_plus)
        self.decay_post = math.exp(-dt / tau_minus)
        self.A_plus = A_plus
        self.A_minus = A_minus

        # State tensors -- allocated on reset / first use
        self.eligibility: Optional[torch.Tensor] = None
        self.x_pre: Optional[torch.Tensor] = None
        self.x_post: Optional[torch.Tensor] = None

    def reset(self, batch_size: int, device: torch.device):
        """Allocate and zero trace state."""
        self.eligibility = torch.zeros(
            batch_size, self.n_post, self.n_pre, device=device, dtype=torch.float32
        )
        self.x_pre = torch.zeros(
            batch_size, self.n_pre, device=device, dtype=torch.float32
        )
        self.x_post = torch.zeros(
            batch_size, self.n_post, device=device, dtype=torch.float32
        )

    def _ensure_state(self, batch_size: int, device: torch.device):
        if self.eligibility is None or self.eligibility.shape[0] != batch_size:
            self.reset(batch_size, device)

    def _apply_trace_update(
        self, trace: torch.Tensor, correlation: torch.Tensor
    ) -> torch.Tensor:
        """Apply trace type semantics."""
        if self.trace_type == TraceType.ACCUMULATING:
            trace = self.decay_e * trace + correlation
        elif self.trace_type == TraceType.REPLACING:
            decayed = self.decay_e * trace
            use_new = correlation.abs() > decayed.abs()
            trace = torch.where(use_new, correlation, decayed)
        elif self.trace_type == TraceType.DUTCH:
            trace = (1.0 - self.alpha) * self.decay_e * trace + correlation
        return torch.clamp(trace, self.clamp_min, self.clamp_max)

    @torch.no_grad()
    def step_rate(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:
        """Rate-based (Hebbian outer product) eligibility update."""
        pre = pre.float()
        post = post.float()
        B = pre.shape[0]
        self._ensure_state(B, pre.device)

        # Outer product correlation: (B, N_post, N_pre)
        correlation = post.unsqueeze(2) * pre.unsqueeze(1)

        self.eligibility = self._apply_trace_update(self.eligibility, correlation)
        return self.eligibility

    @torch.no_grad()
    def step_stdp(
        self,
        spikes_pre: torch.Tensor,
        spikes_post: torch.Tensor,
    ) -> torch.Tensor:
        """Spike-based STDP eligibility update."""
        spikes_pre = spikes_pre.float()
        spikes_post = spikes_post.float()
        B = spikes_pre.shape[0]
        self._ensure_state(B, spikes_pre.device)

        # Update spike traces
        self.x_pre = self.decay_pre * self.x_pre + spikes_pre
        self.x_post = self.decay_post * self.x_post + spikes_post

        # LTP: post spike -> use pre trace
        ltp = self.A_plus * spikes_post.unsqueeze(2) * self.x_pre.unsqueeze(1)
        # LTD: pre spike -> use post trace
        ltd = self.A_minus * self.x_post.unsqueeze(2) * spikes_pre.unsqueeze(1)

        correlation = ltp - ltd
        self.eligibility = self._apply_trace_update(self.eligibility, correlation)
        return self.eligibility

    @torch.no_grad()
    def step(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch to rate or STDP based on kernel_type."""
        if self.kernel_type == KernelType.RATE:
            return self.step_rate(pre, post)
        else:
            return self.step_stdp(pre, post)

    @torch.no_grad()
    def decay_only(self) -> Optional[torch.Tensor]:
        """Apply decay without new activity (no correlation added)."""
        if self.eligibility is None:
            return None
        if self.trace_type == TraceType.DUTCH:
            self.eligibility = (1.0 - self.alpha) * self.decay_e * self.eligibility
        else:
            self.eligibility = self.decay_e * self.eligibility
        self.eligibility = torch.clamp(
            self.eligibility, self.clamp_min, self.clamp_max
        )
        return self.eligibility

    @torch.no_grad()
    def decay_only_no_clamp(self) -> Optional[torch.Tensor]:
        """Apply decay without clamping (for measuring clamp overhead)."""
        if self.eligibility is None:
            return None
        if self.trace_type == TraceType.DUTCH:
            self.eligibility = (1.0 - self.alpha) * self.decay_e * self.eligibility
        else:
            self.eligibility = self.decay_e * self.eligibility
        return self.eligibility


# ---------------------------------------------------------------------------
# Inline Stubs -- STDP and Rate Kernels (standalone functions)
# ---------------------------------------------------------------------------


@torch.no_grad()
def stdp_kernel(
    spikes_pre: torch.Tensor,
    spikes_post: torch.Tensor,
    x_pre: torch.Tensor,
    x_post: torch.Tensor,
    decay_pre: float,
    decay_post: float,
    A_plus: float = 0.01,
    A_minus: float = 0.012,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standalone STDP kernel. Returns (correlation, updated_x_pre, updated_x_post)."""
    spikes_pre = spikes_pre.float()
    spikes_post = spikes_post.float()
    x_pre = decay_pre * x_pre + spikes_pre
    x_post = decay_post * x_post + spikes_post
    ltp = A_plus * spikes_post.unsqueeze(2) * x_pre.unsqueeze(1)
    ltd = A_minus * x_post.unsqueeze(2) * spikes_pre.unsqueeze(1)
    return ltp - ltd, x_pre, x_post


@torch.no_grad()
def rate_kernel(
    pre: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """Standalone rate-based Hebbian kernel. Returns correlation (B, N_post, N_pre)."""
    return post.unsqueeze(2) * pre.unsqueeze(1)


# ---------------------------------------------------------------------------
# Inline Stubs -- NeuromodulatoryGate
# ---------------------------------------------------------------------------


class DopamineComputer(nn.Module):
    """DA: reward prediction error -> tanh -> [-1, 1]."""

    def __init__(self, alpha_baseline: float = 0.01):
        super().__init__()
        self.w_reward = nn.Parameter(torch.tensor(1.0))
        self.alpha_baseline = alpha_baseline
        self.register_buffer("baseline", torch.tensor(0.0))

    def forward(self, reward: torch.Tensor) -> torch.Tensor:
        reward = reward.float()
        rpe = reward - self.baseline
        return torch.tanh(self.w_reward * rpe)

    @torch.no_grad()
    def update_baseline(self, reward: torch.Tensor):
        mean_r = reward.detach().mean()
        self.baseline.mul_(1 - self.alpha_baseline).add_(self.alpha_baseline * mean_r)


class AcetylcholineComputer(nn.Module):
    """ACh: novelty/uncertainty -> sigmoid -> [0, 1]."""

    def __init__(self):
        super().__init__()
        self.w_novelty = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, novelty: torch.Tensor) -> torch.Tensor:
        raw = self.w_novelty * novelty.float() + self.bias
        return torch.sigmoid(raw)


class NorepinephrineComputer(nn.Module):
    """NE: urgency/surprise -> sigmoid -> [0, 1]."""

    def __init__(self):
        super().__init__()
        self.w_urgency = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, urgency: torch.Tensor) -> torch.Tensor:
        raw = self.w_urgency * urgency.float() + self.bias
        return torch.sigmoid(raw)


class SerotoninComputer(nn.Module):
    """5-HT: patience -> sigmoid -> [0, 1]."""

    def __init__(self):
        super().__init__()
        self.w_patience = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, patience: torch.Tensor) -> torch.Tensor:
        raw = self.w_patience * patience.float() + self.bias
        return torch.sigmoid(raw)


class MLPCombination(nn.Module):
    """Learned MLP combination of 4 modulators -> tanh -> [-1, 1]."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(
        self, da: torch.Tensor, ach: torch.Tensor, ne: torch.Tensor, sht: torch.Tensor
    ) -> torch.Tensor:
        mods = torch.stack([da, ach, ne, sht], dim=-1)
        return self.mlp(mods).squeeze(-1)


class NeuromodulatoryGate(nn.Module):
    """Complete neuromodulatory gating system with DA/ACh/NE/5-HT.

    Supports three combination functions: weighted_sum, gated_product, mlp.
    """

    def __init__(
        self,
        combination_fn: str = "weighted_sum",
        modulator_hidden_dim: int = 64,
    ):
        super().__init__()
        self.combination_fn = combination_fn

        self.da_computer = DopamineComputer()
        self.ach_computer = AcetylcholineComputer()
        self.ne_computer = NorepinephrineComputer()
        self.sht_computer = SerotoninComputer()

        if combination_fn == "weighted_sum":
            self.combination_weights = nn.Parameter(
                torch.tensor([0.4, 0.3, 0.2, 0.1])
            )
        elif combination_fn == "gated_product":
            self.gate_weight = nn.Parameter(torch.tensor(0.7))
        elif combination_fn == "mlp":
            self.combine_mlp = MLPCombination(modulator_hidden_dim)
        else:
            raise ValueError(f"Unknown combination_fn: {combination_fn}")

        # State buffers
        self.register_buffer("reward_baseline", torch.tensor(0.0))

    def forward(
        self,
        reward: torch.Tensor,
        novelty: torch.Tensor,
        urgency: torch.Tensor,
        patience: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Compute all modulators and global_plasticity.

        Returns:
            modulators: dict with DA, ACh, NE, 5HT (each (B,))
            global_plasticity: (B,) combined signal
        """
        da = self.da_computer(reward)
        ach = self.ach_computer(novelty)
        ne = self.ne_computer(urgency)
        sht = self.sht_computer(patience)

        if self.combination_fn == "weighted_sum":
            mods = torch.stack([da, ach, ne, sht], dim=-1)
            gp = torch.tanh((mods * self.combination_weights).sum(dim=-1))
        elif self.combination_fn == "gated_product":
            gate = self.gate_weight * ach + (1 - self.gate_weight) * 0.5
            gp = da * gate
        elif self.combination_fn == "mlp":
            gp = self.combine_mlp(da, ach, ne, sht)
        else:
            gp = da  # fallback

        modulators = {"DA": da, "ACh": ach, "NE": ne, "5HT": sht}
        return modulators, gp

    def forward_single_da(self, reward: torch.Tensor) -> torch.Tensor:
        """Compute only DA (for individual modulator benchmarking)."""
        return self.da_computer(reward)

    def forward_single_ach(self, novelty: torch.Tensor) -> torch.Tensor:
        """Compute only ACh."""
        return self.ach_computer(novelty)

    def forward_single_ne(self, urgency: torch.Tensor) -> torch.Tensor:
        """Compute only NE."""
        return self.ne_computer(urgency)

    def forward_single_sht(self, patience: torch.Tensor) -> torch.Tensor:
        """Compute only 5-HT."""
        return self.sht_computer(patience)


# ---------------------------------------------------------------------------
# Inline Stubs -- ThreeFactorUpdate
# ---------------------------------------------------------------------------


class ThreeFactorUpdate:
    """Three-factor weight update engine: delta_w = lr * M * e.

    Supports online (direct weight mod) and auxiliary_loss modes.
    """

    def __init__(
        self,
        lr: float = 0.001,
        max_delta: float = 0.01,
        w_min: float = -2.0,
        w_max: float = 2.0,
        mode: str = "online",
    ):
        self.lr = lr
        self.max_delta = max_delta
        self.w_min = w_min
        self.w_max = w_max
        self.mode = mode

    @torch.no_grad()
    def apply_update(
        self,
        weights: torch.Tensor,
        eligibility: torch.Tensor,
        mod_signal: torch.Tensor,
        clamp: bool = True,
    ) -> torch.Tensor:
        """Apply three-factor update to weights.

        Args:
            weights: (N_post, N_pre) weight matrix
            eligibility: (B, N_post, N_pre) eligibility trace
            mod_signal: (B,) or scalar modulator
            clamp: whether to apply delta and weight clamping

        Returns:
            updated_weights: (N_post, N_pre)
        """
        # Expand mod_signal for broadcasting
        M = mod_signal.detach()
        while M.dim() < eligibility.dim():
            M = M.unsqueeze(-1)

        # Average over batch
        delta_w = self.lr * (M * eligibility.detach()).mean(dim=0)

        if clamp:
            delta_w = torch.clamp(delta_w, -self.max_delta, self.max_delta)

        weights = weights + delta_w

        if clamp:
            weights = torch.clamp(weights, self.w_min, self.w_max)

        return weights

    @torch.no_grad()
    def apply_update_no_clamp(
        self,
        weights: torch.Tensor,
        eligibility: torch.Tensor,
        mod_signal: torch.Tensor,
    ) -> torch.Tensor:
        """Apply update without any clamping (for overhead measurement)."""
        M = mod_signal.detach()
        while M.dim() < eligibility.dim():
            M = M.unsqueeze(-1)
        delta_w = self.lr * (M * eligibility.detach()).mean(dim=0)
        return weights + delta_w

    def compute_auxiliary_loss(
        self,
        eligibility: torch.Tensor,
        mod_signal: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss between three-factor and backprop directions.

        Args:
            eligibility: (B, N_post, N_pre) trace
            mod_signal: (B,) modulator
            grad: (N_post, N_pre) backprop gradient

        Returns:
            alignment_loss: scalar
        """
        M = mod_signal.detach()
        while M.dim() < eligibility.dim():
            M = M.unsqueeze(-1)
        tf_delta = self.lr * (M * eligibility.detach()).mean(dim=0)
        bp_delta = grad.detach()

        tf_norm = tf_delta / (torch.norm(tf_delta) + 1e-8)
        bp_norm = bp_delta / (torch.norm(bp_delta) + 1e-8)
        return torch.mean((tf_norm - bp_norm) ** 2)


# ---------------------------------------------------------------------------
# Inline Stubs -- FastMemoryAdapter
# ---------------------------------------------------------------------------


class FastMemoryAdapter(nn.Module):
    """Low-rank adapter for fast online weight updates.

    Output: x + x @ down @ up  (residual connection with rank-r adapter).
    """

    def __init__(self, dim: int, rank: int = 32):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.down = nn.Parameter(torch.zeros(dim, rank))
        self.up = nn.Parameter(torch.zeros(rank, dim))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        nn.init.zeros_(self.up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x @ self.down @ self.up


# ---------------------------------------------------------------------------
# Inline Stubs -- PlasticityDiagnostics
# ---------------------------------------------------------------------------


class PlasticityDiagnostics:
    """Lightweight diagnostics logger for trace/modulator stats."""

    def __init__(self, max_steps: int = 1000):
        self.max_steps = max_steps
        self.step_count = 0
        self.trace_stats: List[Dict[str, float]] = []
        self.modulator_stats: List[Dict[str, float]] = []

    def log_trace(self, eligibility: torch.Tensor):
        """Log trace statistics for one step."""
        stats = {
            "mean_abs": eligibility.abs().mean().item(),
            "max_abs": eligibility.abs().max().item(),
            "sparsity": (eligibility == 0).float().mean().item(),
        }
        if len(self.trace_stats) < self.max_steps:
            self.trace_stats.append(stats)
        self.step_count += 1

    def log_modulators(self, modulators: Dict[str, torch.Tensor]):
        """Log modulator statistics for one step."""
        stats = {}
        for name, val in modulators.items():
            stats[f"{name}_mean"] = val.mean().item()
            stats[f"{name}_std"] = val.std().item() if val.numel() > 1 else 0.0
        if len(self.modulator_stats) < self.max_steps:
            self.modulator_stats.append(stats)

    def memory_bytes(self) -> int:
        """Estimate memory consumed by stored diagnostics."""
        # Each dict entry is roughly: key string + float value
        n_trace_entries = len(self.trace_stats) * 3  # 3 keys per trace dict
        n_mod_entries = len(self.modulator_stats) * 8  # 4 modulators * 2 stats
        # Rough estimate: 100 bytes per entry (key + value + dict overhead)
        return (n_trace_entries + n_mod_entries) * 100


# ===========================================================================
# BENCHMARK SUITES
# ===========================================================================


def run_suite_1(
    device: torch.device,
    n_iter: int,
    quick: bool = False,
) -> Dict[str, Any]:
    """Suite 1: Eligibility Trace Update Throughput.

    - Varying sizes: (64x64, 256x256, 1024x1024)
    - Trace types: accumulating, replacing, Dutch
    - Kernels: rate, STDP
    """
    print("\n" + "=" * 72)
    print("Suite 1: Eligibility Trace Update Throughput")
    print("=" * 72)

    sizes = [(64, 64), (256, 256), (1024, 1024)]
    if quick:
        sizes = [(64, 64), (256, 256)]

    trace_types = [TraceType.ACCUMULATING, TraceType.REPLACING, TraceType.DUTCH]
    kernel_types = [KernelType.RATE, KernelType.STDP]
    batch_size = 8

    results = []
    all_rows = []

    for n_pre, n_post in sizes:
        for tt in trace_types:
            for kt in kernel_types:
                label = f"{n_pre}x{n_post}_{tt.value}_{kt.value}"

                module = EligibilityTraceModule(
                    n_pre=n_pre,
                    n_post=n_post,
                    trace_type=tt,
                    kernel_type=kt,
                )
                module = module.to(device)
                module.reset(batch_size, device)

                if kt == KernelType.RATE:
                    pre = torch.randn(batch_size, n_pre, device=device)
                    post = torch.randn(batch_size, n_post, device=device)

                    def fn(m=module, p=pre, q=post):
                        m.step_rate(p, q)

                else:
                    spikes_pre = (torch.rand(batch_size, n_pre, device=device) > 0.8).float()
                    spikes_post = (torch.rand(batch_size, n_post, device=device) > 0.8).float()

                    def fn(m=module, sp=spikes_pre, sq=spikes_post):
                        m.step_stdp(sp, sq)

                tr = time_fn(fn, device, n_iter=n_iter, label=label)

                # Compute derived metrics
                updates_per_sec = 1000.0 / tr.mean_ms if tr.mean_ms > 0 else float("inf")
                trace_mem_mb = tensor_memory_mb(module.eligibility)

                result_entry = {
                    "size": f"{n_pre}x{n_post}",
                    "trace_type": tt.value,
                    "kernel": kt.value,
                    "ms_per_update": round(tr.mean_ms, 4),
                    "std_ms": round(tr.std_ms, 4),
                    "updates_per_sec": round(updates_per_sec, 1),
                    "trace_memory_mb": round(trace_mem_mb, 3),
                }
                results.append(result_entry)

                all_rows.append([
                    f"{n_pre}x{n_post}",
                    tt.value,
                    kt.value,
                    f"{tr.mean_ms:.4f}",
                    f"{tr.std_ms:.4f}",
                    f"{updates_per_sec:.1f}",
                    f"{trace_mem_mb:.3f}",
                ])

                # Cleanup
                del module
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    headers = ["Size", "Trace Type", "Kernel", "ms/update", "std(ms)", "updates/s", "Mem(MB)"]
    print(format_table(headers, all_rows))

    return {"suite": 1, "name": SUITE_NAMES[1], "results": results}


def run_suite_2(
    device: torch.device,
    n_iter: int,
    quick: bool = False,
) -> Dict[str, Any]:
    """Suite 2: Trace Decay Overhead.

    - Decay-only vs full update
    - With/without clamp
    - Varying batch sizes: 1, 8, 32, 128
    """
    print("\n" + "=" * 72)
    print("Suite 2: Trace Decay Overhead")
    print("=" * 72)

    n_pre, n_post = 256, 256
    batch_sizes = [1, 8, 32, 128]
    if quick:
        batch_sizes = [1, 8, 32]

    results = []
    all_rows = []

    for bs in batch_sizes:
        # --- Full update ---
        module_full = EligibilityTraceModule(n_pre=n_pre, n_post=n_post)
        module_full = module_full.to(device)
        module_full.reset(bs, device)
        pre = torch.randn(bs, n_pre, device=device)
        post = torch.randn(bs, n_post, device=device)

        tr_full = time_fn(
            lambda m=module_full, p=pre, q=post: m.step_rate(p, q),
            device,
            n_iter=n_iter,
            label=f"full_update_bs{bs}",
        )

        # --- Decay only (with clamp) ---
        module_decay = EligibilityTraceModule(n_pre=n_pre, n_post=n_post)
        module_decay = module_decay.to(device)
        module_decay.reset(bs, device)
        # Seed with some values
        module_decay.eligibility.uniform_(-2.0, 2.0)

        tr_decay = time_fn(
            lambda m=module_decay: m.decay_only(),
            device,
            n_iter=n_iter,
            label=f"decay_only_bs{bs}",
        )

        # --- Decay only (no clamp) ---
        module_decay_nc = EligibilityTraceModule(n_pre=n_pre, n_post=n_post)
        module_decay_nc = module_decay_nc.to(device)
        module_decay_nc.reset(bs, device)
        module_decay_nc.eligibility.uniform_(-2.0, 2.0)

        tr_decay_nc = time_fn(
            lambda m=module_decay_nc: m.decay_only_no_clamp(),
            device,
            n_iter=n_iter,
            label=f"decay_no_clamp_bs{bs}",
        )

        overhead_ratio = tr_full.mean_ms / tr_decay.mean_ms if tr_decay.mean_ms > 0 else float("inf")
        clamp_overhead = (
            (tr_decay.mean_ms - tr_decay_nc.mean_ms) / tr_decay_nc.mean_ms * 100
            if tr_decay_nc.mean_ms > 0
            else 0.0
        )

        entry = {
            "batch_size": bs,
            "full_update_ms": round(tr_full.mean_ms, 4),
            "decay_only_ms": round(tr_decay.mean_ms, 4),
            "decay_no_clamp_ms": round(tr_decay_nc.mean_ms, 4),
            "overhead_ratio": round(overhead_ratio, 3),
            "clamp_overhead_pct": round(clamp_overhead, 2),
        }
        results.append(entry)

        all_rows.append([
            str(bs),
            f"{tr_full.mean_ms:.4f}",
            f"{tr_decay.mean_ms:.4f}",
            f"{tr_decay_nc.mean_ms:.4f}",
            f"{overhead_ratio:.3f}",
            f"{clamp_overhead:.2f}%",
        ])

        del module_full, module_decay, module_decay_nc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    headers = [
        "Batch",
        "Full(ms)",
        "Decay(ms)",
        "Decay-NoClamp(ms)",
        "Full/Decay",
        "Clamp OH",
    ]
    print(format_table(headers, all_rows))

    return {"suite": 2, "name": SUITE_NAMES[2], "results": results}


def run_suite_3(
    device: torch.device,
    n_iter: int,
    quick: bool = False,
) -> Dict[str, Any]:
    """Suite 3: Neuromodulator Computation.

    - Full gate forward pass
    - Individual modulator timing (DA, ACh, NE, 5-HT)
    - Combination functions (weighted_sum, gated_product, MLP)
    - Varying batch sizes
    """
    print("\n" + "=" * 72)
    print("Suite 3: Neuromodulator Computation")
    print("=" * 72)

    batch_sizes = [1, 8, 32, 128]
    combination_fns = ["weighted_sum", "gated_product", "mlp"]
    if quick:
        batch_sizes = [1, 8, 32]

    results = []
    all_rows_full = []
    all_rows_individual = []
    all_rows_combination = []

    # --- Part A: Full forward pass timing per batch size ---
    print("\n  Part A: Full NeuromodulatoryGate Forward Pass")
    for bs in batch_sizes:
        gate = NeuromodulatoryGate(combination_fn="weighted_sum").to(device)
        gate.train(False)

        reward = torch.randn(bs, device=device)
        novelty = torch.rand(bs, device=device)
        urgency = torch.rand(bs, device=device)
        patience = torch.rand(bs, device=device)

        tr = time_fn(
            lambda g=gate, r=reward, n=novelty, u=urgency, p=patience: g(r, n, u, p),
            device,
            n_iter=n_iter,
            label=f"full_gate_bs{bs}",
        )

        entry = {
            "component": "full_gate",
            "batch_size": bs,
            "ms_per_forward": round(tr.mean_ms, 4),
            "std_ms": round(tr.std_ms, 4),
        }
        results.append(entry)
        all_rows_full.append([str(bs), f"{tr.mean_ms:.4f}", f"{tr.std_ms:.4f}"])

        del gate
        gc.collect()

    headers_full = ["Batch", "ms/forward", "std(ms)"]
    print(format_table(headers_full, all_rows_full))

    # --- Part B: Individual modulator timing ---
    print("\n  Part B: Individual Modulator Timing (batch=32)")
    bs = 32
    gate = NeuromodulatoryGate(combination_fn="weighted_sum").to(device)
    gate.train(False)

    reward = torch.randn(bs, device=device)
    novelty = torch.rand(bs, device=device)
    urgency = torch.rand(bs, device=device)
    patience = torch.rand(bs, device=device)

    for mod_name, fn in [
        ("DA", lambda g=gate, r=reward: g.forward_single_da(r)),
        ("ACh", lambda g=gate, n=novelty: g.forward_single_ach(n)),
        ("NE", lambda g=gate, u=urgency: g.forward_single_ne(u)),
        ("5-HT", lambda g=gate, p=patience: g.forward_single_sht(p)),
    ]:
        tr = time_fn(fn, device, n_iter=n_iter, label=f"modulator_{mod_name}")
        entry = {
            "component": f"modulator_{mod_name}",
            "batch_size": bs,
            "ms_per_forward": round(tr.mean_ms, 4),
            "std_ms": round(tr.std_ms, 4),
        }
        results.append(entry)
        all_rows_individual.append([mod_name, f"{tr.mean_ms:.4f}", f"{tr.std_ms:.4f}"])

    del gate
    gc.collect()

    headers_ind = ["Modulator", "ms/forward", "std(ms)"]
    print(format_table(headers_ind, all_rows_individual))

    # --- Part C: Combination function comparison ---
    print("\n  Part C: Combination Function Comparison (batch=32)")
    bs = 32
    for cfn in combination_fns:
        gate = NeuromodulatoryGate(combination_fn=cfn, modulator_hidden_dim=64).to(device)
        gate.train(False)

        reward = torch.randn(bs, device=device)
        novelty = torch.rand(bs, device=device)
        urgency = torch.rand(bs, device=device)
        patience = torch.rand(bs, device=device)

        tr = time_fn(
            lambda g=gate, r=reward, n=novelty, u=urgency, p=patience: g(r, n, u, p),
            device,
            n_iter=n_iter,
            label=f"combination_{cfn}",
        )

        entry = {
            "component": f"combination_{cfn}",
            "batch_size": bs,
            "ms_per_forward": round(tr.mean_ms, 4),
            "std_ms": round(tr.std_ms, 4),
        }
        results.append(entry)
        all_rows_combination.append([cfn, f"{tr.mean_ms:.4f}", f"{tr.std_ms:.4f}"])

        del gate
        gc.collect()

    headers_comb = ["Combination", "ms/forward", "std(ms)"]
    print(format_table(headers_comb, all_rows_combination))

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"suite": 3, "name": SUITE_NAMES[3], "results": results}


def run_suite_4(
    device: torch.device,
    n_iter: int,
    quick: bool = False,
) -> Dict[str, Any]:
    """Suite 4: Three-Factor Weight Update.

    - Varying weight sizes
    - With/without clamping
    - Online vs auxiliary_loss mode
    """
    print("\n" + "=" * 72)
    print("Suite 4: Three-Factor Weight Update")
    print("=" * 72)

    sizes = [(64, 64), (256, 256), (1024, 1024)]
    if quick:
        sizes = [(64, 64), (256, 256)]

    batch_size = 8
    results = []

    # --- Part A: apply_update for varying weight sizes ---
    print("\n  Part A: apply_update Timing (with clamp)")
    all_rows_a = []

    for n_post, n_pre in sizes:
        updater = ThreeFactorUpdate(lr=0.001, max_delta=0.01, mode="online")
        weights = torch.randn(n_post, n_pre, device=device)
        eligibility = torch.randn(batch_size, n_post, n_pre, device=device)
        mod_signal = torch.randn(batch_size, device=device)

        tr = time_fn(
            lambda u=updater, w=weights.clone(), e=eligibility, m=mod_signal: u.apply_update(
                w, e, m, clamp=True
            ),
            device,
            n_iter=n_iter,
            label=f"apply_update_{n_post}x{n_pre}",
        )

        entry = {
            "component": "apply_update_clamped",
            "size": f"{n_post}x{n_pre}",
            "ms_per_update": round(tr.mean_ms, 4),
            "std_ms": round(tr.std_ms, 4),
        }
        results.append(entry)
        all_rows_a.append([f"{n_post}x{n_pre}", f"{tr.mean_ms:.4f}", f"{tr.std_ms:.4f}"])

    headers_a = ["Size", "ms/update", "std(ms)"]
    print(format_table(headers_a, all_rows_a))

    # --- Part B: With vs without clamping ---
    print("\n  Part B: Clamping Overhead (256x256)")
    all_rows_b = []
    n_post, n_pre = 256, 256
    updater = ThreeFactorUpdate(lr=0.001, max_delta=0.01)
    weights = torch.randn(n_post, n_pre, device=device)
    eligibility = torch.randn(batch_size, n_post, n_pre, device=device)
    mod_signal = torch.randn(batch_size, device=device)

    tr_clamped = time_fn(
        lambda u=updater, w=weights.clone(), e=eligibility, m=mod_signal: u.apply_update(
            w, e, m, clamp=True
        ),
        device,
        n_iter=n_iter,
        label="update_clamped",
    )
    tr_unclamped = time_fn(
        lambda u=updater, w=weights.clone(), e=eligibility, m=mod_signal: u.apply_update_no_clamp(
            w, e, m
        ),
        device,
        n_iter=n_iter,
        label="update_unclamped",
    )

    clamp_overhead = (
        (tr_clamped.mean_ms - tr_unclamped.mean_ms) / tr_unclamped.mean_ms * 100
        if tr_unclamped.mean_ms > 0
        else 0.0
    )

    results.append({
        "component": "clamp_comparison",
        "clamped_ms": round(tr_clamped.mean_ms, 4),
        "unclamped_ms": round(tr_unclamped.mean_ms, 4),
        "clamp_overhead_pct": round(clamp_overhead, 2),
    })
    all_rows_b.append(["clamped", f"{tr_clamped.mean_ms:.4f}", f"{tr_clamped.std_ms:.4f}"])
    all_rows_b.append(["unclamped", f"{tr_unclamped.mean_ms:.4f}", f"{tr_unclamped.std_ms:.4f}"])
    all_rows_b.append(["overhead", f"{clamp_overhead:.2f}%", "---"])

    headers_b = ["Mode", "ms/update", "std(ms)"]
    print(format_table(headers_b, all_rows_b))

    # --- Part C: Online vs auxiliary loss ---
    print("\n  Part C: Online vs Auxiliary Loss (256x256)")
    all_rows_c = []

    # Online mode: apply_update
    tr_online = time_fn(
        lambda u=updater, w=weights.clone(), e=eligibility, m=mod_signal: u.apply_update(
            w, e, m, clamp=True
        ),
        device,
        n_iter=n_iter,
        label="online_mode",
    )

    # Auxiliary loss mode: compute_auxiliary_loss
    grad = torch.randn(n_post, n_pre, device=device)
    tr_aux = time_fn(
        lambda u=updater, e=eligibility, m=mod_signal, g=grad: u.compute_auxiliary_loss(
            e, m, g
        ),
        device,
        n_iter=n_iter,
        label="aux_loss_mode",
    )

    results.append({
        "component": "online_vs_aux",
        "online_ms": round(tr_online.mean_ms, 4),
        "auxiliary_loss_ms": round(tr_aux.mean_ms, 4),
    })
    all_rows_c.append(["online", f"{tr_online.mean_ms:.4f}", f"{tr_online.std_ms:.4f}"])
    all_rows_c.append(["auxiliary_loss", f"{tr_aux.mean_ms:.4f}", f"{tr_aux.std_ms:.4f}"])

    headers_c = ["Mode", "ms/update", "std(ms)"]
    print(format_table(headers_c, all_rows_c))

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"suite": 4, "name": SUITE_NAMES[4], "results": results}


def run_suite_5(
    device: torch.device,
    n_iter: int,
    quick: bool = False,
) -> Dict[str, Any]:
    """Suite 5: Full Pipeline (Trace -> Modulator -> Update).

    - Complete plasticity step
    - Varying n_eligible_layers: 1, 4, 8, 16
    - Varying batch sizes
    """
    print("\n" + "=" * 72)
    print("Suite 5: Full Pipeline (Trace -> Modulator -> Update)")
    print("=" * 72)

    n_pre, n_post = 128, 128
    layer_counts = [1, 4, 8, 16]
    batch_sizes = [1, 8, 32]
    if quick:
        layer_counts = [1, 4, 8]
        batch_sizes = [1, 8]

    results = []
    all_rows = []

    for n_layers in layer_counts:
        for bs in batch_sizes:
            # Build pipeline components
            trace_modules = [
                EligibilityTraceModule(
                    n_pre=n_pre, n_post=n_post, kernel_type=KernelType.RATE
                ).to(device)
                for _ in range(n_layers)
            ]
            for tm in trace_modules:
                tm.reset(bs, device)

            gate = NeuromodulatoryGate(combination_fn="weighted_sum").to(device)
            gate.train(False)
            updater = ThreeFactorUpdate(lr=0.001, max_delta=0.01)

            # Pre-generate inputs
            pre_acts = [torch.randn(bs, n_pre, device=device) for _ in range(n_layers)]
            post_acts = [torch.randn(bs, n_post, device=device) for _ in range(n_layers)]
            weights_list = [torch.randn(n_post, n_pre, device=device) for _ in range(n_layers)]
            reward = torch.randn(bs, device=device)
            novelty = torch.rand(bs, device=device)
            urgency = torch.rand(bs, device=device)
            patience = torch.rand(bs, device=device)

            def full_pipeline_step(
                tms=trace_modules,
                g=gate,
                u=updater,
                pre=pre_acts,
                post=post_acts,
                wts=weights_list,
                r=reward,
                n=novelty,
                ur=urgency,
                p=patience,
            ):
                # Step 1: Update traces for all layers
                traces = []
                for i, tm in enumerate(tms):
                    e = tm.step_rate(pre[i], post[i])
                    traces.append(e)

                # Step 2: Compute neuromodulators
                modulators, gp = g(r, n, ur, p)

                # Step 3: Apply weight updates
                for i in range(len(tms)):
                    wts[i] = u.apply_update(wts[i], traces[i], gp, clamp=True)

            # Time the full pipeline
            tr_total = time_fn(
                full_pipeline_step,
                device,
                n_iter=n_iter,
                label=f"pipeline_{n_layers}layers_bs{bs}",
            )

            # Also time individual components for breakdown
            # -- Trace update only --
            def trace_only(
                tms=trace_modules, pre=pre_acts, post=post_acts,
            ):
                for i, tm in enumerate(tms):
                    tm.step_rate(pre[i], post[i])

            tr_trace = time_fn(
                trace_only, device, n_iter=n_iter, label=f"trace_{n_layers}L_bs{bs}"
            )

            # -- Modulator only --
            def mod_only(g=gate, r=reward, n=novelty, ur=urgency, p=patience):
                g(r, n, ur, p)

            tr_mod = time_fn(
                mod_only, device, n_iter=n_iter, label=f"mod_bs{bs}"
            )

            # -- Update only --
            # Use pre-computed traces
            dummy_traces = [torch.randn(bs, n_post, n_pre, device=device) for _ in range(n_layers)]
            dummy_gp = torch.randn(bs, device=device)

            def update_only(
                u=updater, wts=weights_list, traces=dummy_traces, gp=dummy_gp,
            ):
                for i in range(len(traces)):
                    u.apply_update(wts[i], traces[i], gp, clamp=True)

            tr_update = time_fn(
                update_only, device, n_iter=n_iter, label=f"update_{n_layers}L_bs{bs}"
            )

            entry = {
                "n_layers": n_layers,
                "batch_size": bs,
                "total_ms": round(tr_total.mean_ms, 4),
                "trace_ms": round(tr_trace.mean_ms, 4),
                "modulator_ms": round(tr_mod.mean_ms, 4),
                "update_ms": round(tr_update.mean_ms, 4),
                "trace_pct": round(
                    tr_trace.mean_ms / tr_total.mean_ms * 100 if tr_total.mean_ms > 0 else 0, 1
                ),
                "modulator_pct": round(
                    tr_mod.mean_ms / tr_total.mean_ms * 100 if tr_total.mean_ms > 0 else 0, 1
                ),
                "update_pct": round(
                    tr_update.mean_ms / tr_total.mean_ms * 100 if tr_total.mean_ms > 0 else 0, 1
                ),
            }
            results.append(entry)

            all_rows.append([
                str(n_layers),
                str(bs),
                f"{tr_total.mean_ms:.4f}",
                f"{tr_trace.mean_ms:.4f} ({entry['trace_pct']}%)",
                f"{tr_mod.mean_ms:.4f} ({entry['modulator_pct']}%)",
                f"{tr_update.mean_ms:.4f} ({entry['update_pct']}%)",
            ])

            del trace_modules, gate, updater
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    headers = ["Layers", "Batch", "Total(ms)", "Trace(ms)", "Modulator(ms)", "Update(ms)"]
    print(format_table(headers, all_rows))

    return {"suite": 5, "name": SUITE_NAMES[5], "results": results}


def run_suite_6(
    device: torch.device,
    n_iter: int,
    quick: bool = False,
) -> Dict[str, Any]:
    """Suite 6: Memory Profile.

    - Peak memory for traces at varying sizes
    - Memory for modulator state
    - Memory for diagnostic logging over N steps
    """
    print("\n" + "=" * 72)
    print("Suite 6: Memory Profile")
    print("=" * 72)

    results = []

    # --- Part A: Trace memory at varying sizes ---
    print("\n  Part A: Eligibility Trace Memory")
    sizes = [(64, 64), (256, 256), (512, 512), (1024, 1024)]
    if quick:
        sizes = [(64, 64), (256, 256), (512, 512)]

    batch_sizes_mem = [1, 8, 32]
    all_rows_a = []

    for n_pre, n_post in sizes:
        for bs in batch_sizes_mem:
            module = EligibilityTraceModule(
                n_pre=n_pre, n_post=n_post, kernel_type=KernelType.STDP
            )
            module = module.to(device)

            if device.type == "cuda":
                reset_peak_memory(device)

            module.reset(bs, device)

            # Measure memory of all trace state
            eligibility_mb = tensor_memory_mb(module.eligibility)
            x_pre_mb = tensor_memory_mb(module.x_pre)
            x_post_mb = tensor_memory_mb(module.x_post)
            total_trace_mb = eligibility_mb + x_pre_mb + x_post_mb

            if device.type == "cuda":
                gpu_peak_mb = peak_memory_mb(device)
            else:
                gpu_peak_mb = 0.0

            entry = {
                "component": "trace_memory",
                "size": f"{n_pre}x{n_post}",
                "batch_size": bs,
                "eligibility_mb": round(eligibility_mb, 4),
                "x_pre_mb": round(x_pre_mb, 4),
                "x_post_mb": round(x_post_mb, 4),
                "total_trace_mb": round(total_trace_mb, 4),
                "gpu_peak_mb": round(gpu_peak_mb, 2),
            }
            results.append(entry)

            all_rows_a.append([
                f"{n_pre}x{n_post}",
                str(bs),
                f"{eligibility_mb:.4f}",
                f"{x_pre_mb:.4f}",
                f"{x_post_mb:.4f}",
                f"{total_trace_mb:.4f}",
            ])

            del module
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    headers_a = ["Size", "Batch", "Elig(MB)", "x_pre(MB)", "x_post(MB)", "Total(MB)"]
    print(format_table(headers_a, all_rows_a))

    # --- Part B: Modulator state memory ---
    print("\n  Part B: Neuromodulator Gate Memory")
    all_rows_b = []

    for combo_fn in ["weighted_sum", "gated_product", "mlp"]:
        gate = NeuromodulatoryGate(
            combination_fn=combo_fn, modulator_hidden_dim=64
        ).to(device)

        # Count parameters and buffers
        param_bytes = sum(p.numel() * p.element_size() for p in gate.parameters())
        buffer_bytes = sum(
            b.numel() * b.element_size() for b in gate.buffers()
        )
        param_mb = param_bytes / (1024.0 * 1024.0)
        buffer_mb = buffer_bytes / (1024.0 * 1024.0)
        total_mb = param_mb + buffer_mb
        n_params = sum(p.numel() for p in gate.parameters())

        entry = {
            "component": "modulator_memory",
            "combination_fn": combo_fn,
            "n_params": n_params,
            "param_mb": round(param_mb, 6),
            "buffer_mb": round(buffer_mb, 6),
            "total_mb": round(total_mb, 6),
        }
        results.append(entry)

        all_rows_b.append([
            combo_fn,
            str(n_params),
            f"{param_mb:.6f}",
            f"{buffer_mb:.6f}",
            f"{total_mb:.6f}",
        ])

        del gate
        gc.collect()

    headers_b = ["Combination", "Params", "Param(MB)", "Buffer(MB)", "Total(MB)"]
    print(format_table(headers_b, all_rows_b))

    # --- Part C: Diagnostic logging memory ---
    print("\n  Part C: Diagnostics Logging Memory (over N steps)")
    all_rows_c = []

    step_counts = [100, 500, 1000, 5000]
    if quick:
        step_counts = [100, 500, 1000]

    bs = 8
    n_pre_d, n_post_d = 128, 128

    for n_steps in step_counts:
        diag = PlasticityDiagnostics(max_steps=n_steps)

        # Simulate logging
        for _ in range(n_steps):
            dummy_elig = torch.randn(bs, n_post_d, n_pre_d, device=device)
            dummy_mods = {
                "DA": torch.randn(bs, device=device),
                "ACh": torch.rand(bs, device=device),
                "NE": torch.rand(bs, device=device),
                "5HT": torch.rand(bs, device=device),
            }
            diag.log_trace(dummy_elig)
            diag.log_modulators(dummy_mods)

        diag_bytes = diag.memory_bytes()
        diag_mb = diag_bytes / (1024.0 * 1024.0)

        # Measure actual Python object memory (approximate)
        trace_list_bytes = sys.getsizeof(diag.trace_stats)
        for d in diag.trace_stats[:10]:
            trace_list_bytes += sys.getsizeof(d)
        actual_per_entry = trace_list_bytes / max(1, min(10, len(diag.trace_stats)))
        actual_estimated_mb = (
            actual_per_entry * (len(diag.trace_stats) + len(diag.modulator_stats))
        ) / (1024.0 * 1024.0)

        entry = {
            "component": "diagnostics_memory",
            "n_steps": n_steps,
            "estimated_mb": round(diag_mb, 4),
            "actual_estimated_mb": round(actual_estimated_mb, 4),
            "n_trace_entries": len(diag.trace_stats),
            "n_mod_entries": len(diag.modulator_stats),
        }
        results.append(entry)

        all_rows_c.append([
            str(n_steps),
            str(len(diag.trace_stats)),
            str(len(diag.modulator_stats)),
            f"{diag_mb:.4f}",
        ])

        del diag
        gc.collect()

    headers_c = ["Steps", "Trace Entries", "Mod Entries", "Est. Memory(MB)"]
    print(format_table(headers_c, all_rows_c))

    # --- Part D: Combined memory budget for a full pipeline ---
    print("\n  Part D: Combined Memory Budget (typical production config)")
    all_rows_d = []

    configs = [
        {"name": "minimal", "n_layers": 2, "size": (64, 64), "bs": 1},
        {"name": "dev", "n_layers": 4, "size": (256, 256), "bs": 8},
        {"name": "production", "n_layers": 8, "size": (512, 512), "bs": 32},
    ]
    if quick:
        configs = configs[:2]

    for cfg in configs:
        n_layers = cfg["n_layers"]
        n_pre_c, n_post_c = cfg["size"]
        bs_c = cfg["bs"]

        # Trace memory (per layer)
        single_elig_mb = (bs_c * n_post_c * n_pre_c * 4) / (1024.0 * 1024.0)
        single_xpre_mb = (bs_c * n_pre_c * 4) / (1024.0 * 1024.0)
        single_xpost_mb = (bs_c * n_post_c * 4) / (1024.0 * 1024.0)
        per_layer_mb = single_elig_mb + single_xpre_mb + single_xpost_mb
        total_trace_all_layers = per_layer_mb * n_layers

        # Gate memory (roughly)
        gate_mb = 0.005  # very small for weighted_sum

        # Weights memory
        weights_mb = (n_post_c * n_pre_c * 4) / (1024.0 * 1024.0) * n_layers

        total_pipeline_mb = total_trace_all_layers + gate_mb + weights_mb

        entry = {
            "component": "combined_budget",
            "config": cfg["name"],
            "n_layers": n_layers,
            "size": f"{n_pre_c}x{n_post_c}",
            "batch_size": bs_c,
            "traces_mb": round(total_trace_all_layers, 3),
            "gate_mb": round(gate_mb, 3),
            "weights_mb": round(weights_mb, 3),
            "total_mb": round(total_pipeline_mb, 3),
        }
        results.append(entry)

        all_rows_d.append([
            cfg["name"],
            f"{n_layers}L {n_pre_c}x{n_post_c} bs{bs_c}",
            f"{total_trace_all_layers:.3f}",
            f"{gate_mb:.3f}",
            f"{weights_mb:.3f}",
            f"{total_pipeline_mb:.3f}",
        ])

    headers_d = ["Config", "Shape", "Traces(MB)", "Gate(MB)", "Weights(MB)", "Total(MB)"]
    print(format_table(headers_d, all_rows_d))

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"suite": 6, "name": SUITE_NAMES[6], "results": results}


# ===========================================================================
# Main Entry Point
# ===========================================================================


def print_system_info(device: torch.device):
    """Print system and hardware information."""
    print("=" * 72)
    print("Plasticity Benchmark -- System Information")
    print("=" * 72)
    print(f"  PyTorch version:  {torch.__version__}")
    print(f"  NumPy version:    {np.__version__}")
    print(f"  Python version:   {sys.version.split()[0]}")
    print(f"  Device:           {device}")
    if device.type == "cuda":
        print(f"  CUDA version:     {torch.version.cuda}")
        props = torch.cuda.get_device_properties(device)
        print(f"  GPU:              {props.name}")
        print(f"  GPU memory:       {props.total_mem / 1024**3:.1f} GB")
        print(f"  SM count:         {props.multi_processor_count}")
    print(f"  CPU count:        {os.cpu_count()}")
    print(f"  Torch threads:    {torch.get_num_threads()}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark neuromodulation + eligibility traces pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plasticity_benchmark.py                    # All suites, CPU
  python plasticity_benchmark.py --suite 1 3        # Suites 1 and 3
  python plasticity_benchmark.py --device cuda      # GPU benchmarks
  python plasticity_benchmark.py --quick --json     # Quick run, JSON output
        """,
    )
    parser.add_argument(
        "--suite",
        type=int,
        nargs="*",
        default=None,
        choices=[1, 2, 3, 4, 5, 6],
        help="Suite(s) to run. Default: all (1-6).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to benchmark on. Default: cpu.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (in addition to tables).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer iterations, smaller sizes.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Override number of timed iterations per benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON output file (implies --json).",
    )

    args = parser.parse_args()

    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine iteration count
    if args.iters is not None:
        n_iter = args.iters
    elif args.quick:
        n_iter = QUICK_ITERS
    else:
        n_iter = DEFAULT_ITERS

    # Determine which suites to run
    if args.suite is None or len(args.suite) == 0:
        suites_to_run = [1, 2, 3, 4, 5, 6]
    else:
        suites_to_run = sorted(set(args.suite))

    # Print header
    print_system_info(device)
    print(f"Configuration:")
    print(f"  Suites:     {suites_to_run}")
    print(f"  Iterations: {n_iter}")
    print(f"  Warmup:     {WARMUP_ITERS}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Seed:       {args.seed}")
    print()

    # Run suites
    suite_runners = {
        1: run_suite_1,
        2: run_suite_2,
        3: run_suite_3,
        4: run_suite_4,
        5: run_suite_5,
        6: run_suite_6,
    }

    all_results = []
    total_start = time.perf_counter()

    for suite_id in suites_to_run:
        suite_start = time.perf_counter()
        result = suite_runners[suite_id](device, n_iter=n_iter, quick=args.quick)
        suite_elapsed = time.perf_counter() - suite_start
        result["elapsed_sec"] = round(suite_elapsed, 2)
        all_results.append(result)
        print(f"\n  Suite {suite_id} completed in {suite_elapsed:.2f}s")

    total_elapsed = time.perf_counter() - total_start

    # Summary
    print("\n" + "=" * 72)
    print("BENCHMARK SUMMARY")
    print("=" * 72)
    summary_rows = []
    for r in all_results:
        n_results = len(r.get("results", []))
        summary_rows.append([
            str(r["suite"]),
            r["name"],
            str(n_results),
            f"{r['elapsed_sec']:.2f}s",
        ])
    headers_summary = ["Suite", "Name", "Tests", "Time"]
    print(format_table(headers_summary, summary_rows))
    print(f"\nTotal benchmark time: {total_elapsed:.2f}s")
    print(f"Device: {device}")

    # JSON output
    json_output = {
        "metadata": {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "device": str(device),
            "n_iter": n_iter,
            "quick_mode": args.quick,
            "seed": args.seed,
            "total_elapsed_sec": round(total_elapsed, 2),
        },
        "suites": all_results,
    }

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        json_output["metadata"]["gpu_name"] = props.name
        json_output["metadata"]["gpu_memory_gb"] = round(props.total_mem / 1024**3, 1)

    if args.json or args.output:
        json_str = json.dumps(json_output, indent=2)
        if args.json:
            print("\n--- JSON Output ---")
            print(json_str)

        if args.output:
            output_path = args.output
            with open(output_path, "w") as f:
                f.write(json_str)
            print(f"\nJSON results written to: {output_path}")


if __name__ == "__main__":
    main()
