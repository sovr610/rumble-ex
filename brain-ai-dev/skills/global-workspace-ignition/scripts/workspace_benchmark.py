#!/usr/bin/env python3
"""
Global Workspace Benchmark -- Performance benchmarking for the Global Workspace module.

Usage:
    python workspace_benchmark.py [--device cpu|cuda] [--quick] [--category NAME] [--json-report PATH]

Categories:
    1. competition_throughput      -- Competition scoring tokens/sec across T_total sizes
    2. iterative_round_overhead    -- Wall-clock comparison for max_rounds 1/2/4/6
    3. ignition_computation        -- Ignition score + gate timing, interpretable vs learned
    4. broadcast_adapter           -- Per-adapter latency for BroadcastTo* adapters
    5. working_memory_backend      -- CfC vs LTC vs GRU throughput comparison
    6. full_pipeline               -- End-to-end encoder outputs -> workspace output
    7. scaling_analysis            -- workspace_dim / batch_size / token_count scaling
    8. memory_profiling            -- Parameter counts, peak memory, per-round memory
"""
from __future__ import annotations

import os
import sys
import gc
import math
import time
import json
import argparse
import traceback
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable, Any

# ---------------------------------------------------------------------------
# Path setup -- 4 levels: brain-ai-dev/skills/global-workspace-ignition/scripts
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))
sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------
_workspace_available = False
_workspace_source = "UNAVAILABLE"
try:
    from brain_ai.workspace.global_workspace import (
        GlobalWorkspace, GlobalWorkspaceConfig, AttentionCompetition,
        InformationBroadcast, ModalityProjection,
        SelectionBroadcastConfig, IterativeCompetition, RefinedBroadcast,
        SelectionBroadcastWorkspace, create_global_workspace,
        create_selection_broadcast_workspace,
    )
    _workspace_available = True
    _workspace_source = "brain_ai.workspace.global_workspace"
except ImportError:
    pass

_working_memory_available = False
_working_memory_source = "UNAVAILABLE"
try:
    from brain_ai.workspace.working_memory import (
        WorkingMemory, WorkingMemoryConfig, GRUWorkingMemory,
        create_working_memory, NCPS_AVAILABLE,
    )
    _working_memory_available = True
    _working_memory_source = "brain_ai.workspace.working_memory"
except ImportError:
    NCPS_AVAILABLE = False

ALL_CATEGORIES = [
    "competition_throughput",
    "iterative_round_overhead",
    "ignition_computation",
    "broadcast_adapter",
    "working_memory_backend",
    "full_pipeline",
    "scaling_analysis",
    "memory_profiling",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """
    Result for a single benchmark measurement.

    Attributes:
        name:       Unique identifier for this benchmark point
        mean_time:  Mean latency in milliseconds
        std_time:   Standard deviation of latency in milliseconds
        throughput: Throughput in appropriate units (tokens/sec, steps/sec, etc.)
        memory_mb:  Peak memory usage in megabytes during the benchmark
        details:    Additional metadata (parameters, config, phase info)
    """
    name: str
    mean_time: float = 0.0
    std_time: float = 0.0
    throughput: float = 0.0
    memory_mb: float = 0.0
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _std(vals: List[float]) -> float:
    """Compute sample standard deviation of a list of floats."""
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _percentile(vals: List[float], pct: float) -> float:
    """
    Compute a percentile from a list of values.

    Args:
        vals: Unsorted list of float values
        pct:  Percentile as a fraction (e.g. 0.99 for p99)

    Returns:
        The value at the given percentile
    """
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(int(pct * len(s)), len(s) - 1)
    return s[idx]


def _median(vals: List[float]) -> float:
    """Compute median of a list of floats."""
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


# ---------------------------------------------------------------------------
# System / memory helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Get current process resident set size in MB (Linux)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def _nparams(mod: nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)


def _nparams_all(mod: nn.Module) -> int:
    """Count all parameters (trainable + frozen) in a PyTorch module."""
    return sum(p.numel() for p in mod.parameters())


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_params(n: int) -> str:
    """Format parameter count with human-readable suffix (K/M)."""
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _fmt_ms(ms: float) -> str:
    """Format a time value in milliseconds with appropriate unit."""
    if ms < 0.001:
        return f"{ms * 1000:.2f} us"
    if ms < 1.0:
        return f"{ms:.4f} ms"
    if ms < 1000:
        return f"{ms:.2f} ms"
    return f"{ms / 1000:.2f} s"


def _fmt_tp(tp: float, unit: str = "tok/s") -> str:
    """Format throughput value with human-readable suffix."""
    if tp >= 1e6:
        return f"{tp / 1e6:.2f}M {unit}"
    if tp >= 1e3:
        return f"{tp / 1e3:.1f}K {unit}"
    if tp > 0:
        return f"{tp:.1f} {unit}"
    return "N/A"


def _fmt_mem(mb: float) -> str:
    """Format memory in MB with appropriate precision."""
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    if mb >= 1:
        return f"{mb:.1f} MB"
    return f"{mb * 1024:.1f} KB"


def _set_inference_mode(mod: nn.Module) -> nn.Module:
    """Set module to inference mode (disable dropout, batchnorm tracking)."""
    mod.train(False)
    return mod


# ---------------------------------------------------------------------------
# Benchmark Suite
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """
    Runs all Global Workspace benchmarks and collects results.

    The suite provides 8 benchmark categories that can be run individually
    or together via run_all(). Each category creates multiple BenchmarkResult
    entries stored in self.results.

    Args:
        device: "cpu" or "cuda" -- target device for all benchmarks
        quick:  If True, use fewer warmup/measured iterations and smaller
                parameter sweeps for faster feedback during development
    """

    def __init__(self, device: str = "cpu", quick: bool = False):
        self.device = torch.device(device)
        self.quick = quick
        self.results: List[BenchmarkResult] = []

        # Quick mode reduces iteration counts for fast feedback
        self.warmup_runs = 1 if quick else 3
        self.measured_runs = 3 if quick else 10

    # -------------------------------------------------------------------
    # Timing and memory helpers
    # -------------------------------------------------------------------

    def _sync(self) -> None:
        """Synchronize CUDA stream if running on GPU (no-op on CPU)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _mem_mb(self) -> float:
        """
        Get current peak memory usage in MB.
        Uses CUDA memory stats on GPU, process RSS on CPU.
        """
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        return _rss_mb()

    def _reset_mem(self) -> None:
        """Reset peak memory counters and run garbage collection."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        gc.collect()

    def _timed(
        self,
        fn: Callable,
        warmup: Optional[int] = None,
        measured: Optional[int] = None,
    ) -> List[float]:
        """
        Time a callable with warmup runs followed by measured runs.

        Uses time.perf_counter() for CPU timing and ensures
        torch.cuda.synchronize() is called before and after each
        measured iteration when running on GPU.

        Args:
            fn:       Zero-argument callable to benchmark
            warmup:   Number of warmup runs (default: self.warmup_runs)
            measured: Number of measured runs (default: self.measured_runs)

        Returns:
            List of latencies in milliseconds, one per measured run
        """
        w = warmup if warmup is not None else self.warmup_runs
        m = measured if measured is not None else self.measured_runs

        # Warmup -- bring caches and JIT up to steady state
        for _ in range(w):
            fn()
            self._sync()

        # Measured runs
        latencies_ms: List[float] = []
        for _ in range(m):
            self._sync()
            t0 = time.perf_counter()
            fn()
            self._sync()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        return latencies_ms

    def _rec(
        self,
        name: str,
        lats: List[float],
        throughput: float = 0.0,
        details: Optional[dict] = None,
    ) -> BenchmarkResult:
        """
        Record a benchmark result from a list of latency measurements.

        If throughput is not explicitly provided and lats is non-empty,
        throughput is computed as 1000/mean_ms (ops/sec).

        Args:
            name:       Unique benchmark name
            lats:       List of latencies in milliseconds
            throughput: Explicit throughput value (0 = auto-compute)
            details:    Additional metadata dictionary

        Returns:
            The recorded BenchmarkResult
        """
        d = details or {}
        if not lats:
            r = BenchmarkResult(
                name=name,
                throughput=throughput,
                memory_mb=self._mem_mb(),
                details=d,
            )
        else:
            mean = sum(lats) / len(lats)
            tp = throughput if throughput else (1000.0 / mean if mean > 0 else 0.0)
            r = BenchmarkResult(
                name=name,
                mean_time=mean,
                std_time=_std(lats),
                throughput=tp,
                memory_mb=self._mem_mb(),
                details=d,
            )
        self.results.append(r)
        return r

    def _skip(self, name: str, reason: str) -> None:
        """Record a skipped benchmark with the given reason."""
        self.results.append(
            BenchmarkResult(name=name, details={"status": "SKIPPED", "reason": reason})
        )

    # =======================================================================
    # Category 1: Competition Throughput
    #
    # Measures tokens/sec for the AttentionCompetition module.
    # Varies T_total (number of competing tokens): [128, 256, 512, 1024, 2048]
    # Fixed: K=7 (capacity limit), workspace_dim=512, batch_size=4
    # Separately times: scoring, top-K selection, and slot construction
    # =======================================================================

    def bench_competition_throughput(self) -> None:
        """Benchmark competition scoring throughput across token counts."""
        if not _workspace_available:
            print(f"  [SKIPPED] -- {_workspace_source}"); self._skip("competition_throughput", _workspace_source); return
        print("  Running competition throughput benchmarks...")
        T_totals = [128, 512, 1024] if self.quick else [128, 256, 512, 1024, 2048]
        ws_dim, K, heads, B = 512, 7, 8, 4

        for T in T_totals:
            try:
                self._reset_mem(); torch.manual_seed(42)
                comp = _set_inference_mode(AttentionCompetition(workspace_dim=ws_dim, num_heads=heads,
                             capacity_limit=K, temperature=1.0, dropout=0.0).to(self.device))
                feats = torch.randn(B, T, ws_dim, device=self.device)
                sals  = torch.randn(B, T, 1, device=self.device)
                # Full pass
                with torch.no_grad():
                    lats = self._timed(lambda: comp(feats, sals))
                tps = (B * T) / (sum(lats)/len(lats)) * 1000.0 if lats else 0.0
                self._rec(f"competition_full_T{T}", lats, tps,
                          {"T_total":T,"K":K,"workspace_dim":ws_dim,"batch_size":B})
                # Scoring only
                attn_l, gate_l = comp.attention, comp.gate
                with torch.no_grad():
                    lats_s = self._timed(lambda: (attn_l(feats, feats, feats, need_weights=True), gate_l))
                self._rec(f"competition_scoring_T{T}", lats_s, details={"T_total":T,"phase":"scoring"})
                # TopK only
                with torch.no_grad():
                    att, _ = attn_l(feats, feats, feats, need_weights=True)
                    gs = gate_l(att).squeeze(-1) + sals.squeeze(-1)
                    aw = torch.softmax(gs, dim=-1)
                def _topk():
                    _, ti = torch.topk(aw, K, dim=-1)
                    m = torch.zeros_like(aw); m.scatter_(1, ti, 1.0)
                    return aw * m
                with torch.no_grad():
                    lats_k = self._timed(_topk)
                self._rec(f"competition_topk_T{T}", lats_k, details={"T_total":T,"K":K,"phase":"topk"})
                # Slot construction
                with torch.no_grad():
                    _, ti = torch.topk(aw, K, dim=-1)
                    m = torch.zeros_like(aw); m.scatter_(1, ti, 1.0)
                    mw = aw * m; mw = mw / (mw.sum(-1, keepdim=True)+1e-8)
                norm_l = comp.norm
                with torch.no_grad():
                    lats_sl = self._timed(lambda: norm_l(att * mw.unsqueeze(-1)))
                self._rec(f"competition_slot_T{T}", lats_sl, details={"T_total":T,"phase":"slot"})
                del comp, feats, sals; gc.collect()
            except Exception as e:
                print(f"    [ERROR] T={T}: {e}"); traceback.print_exc()

    # =======================================================================
    # Category 2: Iterative Round Overhead
    #
    # Compares wall-clock time for IterativeCompetition with different
    # max_rounds settings: [1, 2, 4, 6].  Same input each time to
    # isolate the cost of additional selection rounds.
    # Reports: total time, time per round, and convergence round.
    # =======================================================================

    def bench_iterative_round_overhead(self) -> None:
        """Benchmark overhead of additional iterative selection rounds."""
        if not _workspace_available:
            print(f"  [SKIPPED] -- {_workspace_source}"); self._skip("iterative_round_overhead", _workspace_source); return
        print("  Running iterative round overhead benchmarks...")
        rounds_list = [1, 3, 6] if self.quick else [1, 2, 4, 6]
        ws_dim, items, B = 512, 12, 4
        feats = torch.randn(B, items, ws_dim, device=self.device)
        sals  = torch.randn(B, items, 1, device=self.device)

        for mr in rounds_list:
            try:
                self._reset_mem(); torch.manual_seed(42)
                ic = _set_inference_mode(IterativeCompetition(workspace_dim=ws_dim, num_heads=8,
                           selection_rounds=mr, ignition_threshold=0.3,
                           temperature=0.5, dropout=0.0).to(self.device))
                conv = []
                def _run():
                    _, _, info = ic(feats, sals); conv.append(info['selection_rounds'])
                with torch.no_grad():
                    lats = self._timed(_run)
                mean_ms = sum(lats)/len(lats) if lats else 0.0
                avg_conv = sum(conv[-self.measured_runs:])/self.measured_runs if conv else mr
                self._rec(f"iterative_rounds_{mr}", lats, details={
                    "max_rounds":mr, "time_per_round_ms":round(mean_ms/mr,4),
                    "avg_convergence_round":round(avg_conv,2), "num_items":items,
                    "workspace_dim":ws_dim, "batch_size":B})
                del ic; gc.collect()
            except Exception as e:
                print(f"    [ERROR] rounds={mr}: {e}"); traceback.print_exc()

    # =======================================================================
    # Category 3: Ignition Computation
    #
    # Times the ignition score computation (4-component detector),
    # the ignition gate application (threshold + masking), and
    # compares interpretable (hand-crafted 4-component) vs learned
    # (neural network) ignition modes.
    # =======================================================================

    def bench_ignition_computation(self) -> None:
        """Benchmark ignition detection: score, gate, interpretable vs learned."""
        if not _workspace_available:
            print(f"  [SKIPPED] -- {_workspace_source}"); self._skip("ignition_computation", _workspace_source); return
        print("  Running ignition computation benchmarks...")
        ws_dim, items, B = 512, 10, 8
        try:
            self._reset_mem(); torch.manual_seed(42)
            ic = _set_inference_mode(IterativeCompetition(workspace_dim=ws_dim, num_heads=8,
                       selection_rounds=3, ignition_threshold=0.3,
                       temperature=0.5, dropout=0.0).to(self.device))
            feats = torch.randn(B, items, ws_dim, device=self.device)
            mf = feats.mean(dim=1)
            det = ic.ignition_detector
            # Score computation
            with torch.no_grad():
                lats_s = self._timed(lambda: det(mf))
            self._rec("ignition_score_computation", lats_s,
                      details={"workspace_dim":ws_dim,"batch_size":B,"detector_params":_nparams(det)})
            # Gate application
            thr = 0.3
            def _gate():
                s = det(mf); g = (s > thr).float(); return feats * g.unsqueeze(1)
            with torch.no_grad():
                lats_g = self._timed(_gate)
            self._rec("ignition_gate_application", lats_g,
                      details={"threshold":thr,"workspace_dim":ws_dim,"batch_size":B})
            # Interpretable ignition (4 components)
            def _interp():
                am = feats.mean(dim=(1,2)); av = feats.var(dim=(1,2))
                ax = feats.amax(dim=(1,2))
                p = torch.softmax(feats.mean(dim=-1), dim=-1)
                ent = -(p*(p+1e-8).log()).sum(dim=-1)
                return 0.25*(torch.sigmoid(am)+torch.sigmoid(av)+torch.sigmoid(ax)+torch.sigmoid(-ent))
            with torch.no_grad():
                lats_i = self._timed(_interp)
            self._rec("ignition_interpretable", lats_i,
                      details={"mode":"interpretable","components":["mean","var","max","entropy"]})
            # Learned ignition
            with torch.no_grad():
                lats_l = self._timed(lambda: det(mf))
            self._rec("ignition_learned", lats_l,
                      details={"mode":"learned_nn","detector_params":_nparams(det)})
            # Comparison
            mi = sum(lats_i)/len(lats_i) if lats_i else 0.0
            ml = sum(lats_l)/len(lats_l) if lats_l else 0.0
            self._rec("ignition_comparison", [],
                      details={"interpretable_ms":round(mi,4),"learned_ms":round(ml,4),
                               "speedup":round(mi/ml,3) if ml>0 else 0.0})
            del ic; gc.collect()
        except Exception as e:
            print(f"    [ERROR] ignition: {e}"); traceback.print_exc()

    # =======================================================================
    # Category 4: Broadcast Adapter Throughput
    #
    # Times each broadcast adapter type:
    #   - BroadcastToTemporal  (ws_dim -> 512)
    #   - BroadcastToPooled    (ws_dim -> 256)
    #   - BroadcastToSymbolic  (ws_dim -> 128)
    #   - BroadcastToDecision  (ws_dim -> 64)
    # Also benchmarks the full RefinedBroadcast module with feedback.
    # Varies K (capacity): [4, 7, 9, 12]
    # =======================================================================

    def bench_broadcast_adapter(self) -> None:
        """Benchmark per-adapter latency for broadcast projections."""
        if not _workspace_available:
            print(f"  [SKIPPED] -- {_workspace_source}"); self._skip("broadcast_adapter", _workspace_source); return
        print("  Running broadcast adapter benchmarks...")
        ws_dim, B = 512, 4
        K_vals = [4, 7, 12] if self.quick else [4, 7, 9, 12]
        adapters = {"BroadcastToTemporal":512, "BroadcastToPooled":256,
                    "BroadcastToSymbolic":128, "BroadcastToDecision":64}

        for aname, tdim in adapters.items():
            for K in K_vals:
                try:
                    self._reset_mem(); torch.manual_seed(42)
                    ad = _set_inference_mode(nn.Sequential(nn.Linear(ws_dim,ws_dim), nn.ReLU(),
                                             nn.Linear(ws_dim,tdim)).to(self.device))
                    x = torch.randn(B, ws_dim, device=self.device)
                    with torch.no_grad():
                        lats = self._timed(lambda: ad(x))
                    self._rec(f"broadcast_{aname}_K{K}", lats,
                              details={"adapter":aname,"K":K,"target_dim":tdim,
                                       "params":_nparams(ad)})
                    del ad; gc.collect()
                except Exception as e:
                    print(f"    [ERROR] {aname}_K{K}: {e}"); traceback.print_exc()

        # RefinedBroadcast full module
        mod_dims = {"vision":512,"text":512,"audio":256}
        for K in K_vals:
            try:
                self._reset_mem(); torch.manual_seed(42)
                bc = _set_inference_mode(RefinedBroadcast(workspace_dim=ws_dim, modality_dims=mod_dims,
                           broadcast_iterations=2, broadcast_decay=0.9, dropout=0.0).to(self.device))
                wc = torch.randn(B, ws_dim, device=self.device)
                ms = {n:torch.randn(B,d,device=self.device) for n,d in mod_dims.items()}
                with torch.no_grad():
                    lats = self._timed(lambda: bc(wc, ms))
                self._rec(f"broadcast_refined_K{K}", lats,
                          details={"adapter":"RefinedBroadcast","K":K,"params":_nparams(bc)})
                del bc; gc.collect()
            except Exception as e:
                print(f"    [ERROR] refined_K{K}: {e}"); traceback.print_exc()

    # =======================================================================
    # Category 5: Working Memory Backend Comparison
    #
    # Compares CfC (Closed-form Continuous-time), LTC (Liquid Time-Constant),
    # and GRU backends for WorkingMemory.
    # Times four operations: forward pass, state update, buffer write, retrieve.
    # Varies hidden_dim: [256, 512, 1024, 2048, 4096]
    # CfC/LTC only available when ncps package is installed.
    # =======================================================================

    def bench_working_memory_backend(self) -> None:
        """Benchmark working memory backends: forward, state, buffer, retrieve."""
        if not _working_memory_available:
            print(f"  [SKIPPED] -- {_working_memory_source}"); self._skip("working_memory_backend", _working_memory_source); return
        print("  Running working memory backend comparison...")
        hdims = [256, 512, 1024] if self.quick else [256, 512, 1024, 2048, 4096]
        B = 4
        backends = ["cfc","ltc","gru"] if NCPS_AVAILABLE else ["gru"]

        for bk in backends:
            for hd in hdims:
                try:
                    self._reset_mem(); torch.manual_seed(42)
                    wm = _set_inference_mode(create_working_memory(input_dim=hd, hidden_dim=hd,
                               output_dim=hd, mode=bk).to(self.device))
                    x = torch.randn(B, hd, device=self.device)
                    # Forward
                    with torch.no_grad():
                        lf = self._timed(lambda: (wm.reset_state(), wm(x)))
                    self._rec(f"wm_{bk}_forward_h{hd}", lf,
                              details={"backend":bk,"hidden_dim":hd,"batch_size":B,
                                       "phase":"forward","params":_nparams(wm)})
                    # State update (with existing state)
                    wm.reset_state()
                    with torch.no_grad():
                        for _ in range(3): wm(x, update_buffer=False)
                        ls = self._timed(lambda: wm(x, update_buffer=False))
                    self._rec(f"wm_{bk}_state_h{hd}", ls,
                              details={"backend":bk,"hidden_dim":hd,"phase":"state_update"})
                    # Buffer write
                    wm.reset_state()
                    with torch.no_grad():
                        lb = self._timed(lambda: wm.update_buffer(torch.randn(B,hd,device=self.device)))
                    self._rec(f"wm_{bk}_buffer_h{hd}", lb,
                              details={"backend":bk,"hidden_dim":hd,"phase":"buffer_write"})
                    # Retrieve
                    wm.reset_state()
                    for _ in range(5): wm.update_buffer(torch.randn(B,hd,device=self.device))
                    q = torch.randn(B, hd, device=self.device)
                    with torch.no_grad():
                        lr = self._timed(lambda: wm.retrieve(q))
                    buf_n = wm.memory_buffer.shape[1] if wm.memory_buffer is not None else 0
                    self._rec(f"wm_{bk}_retrieve_h{hd}", lr,
                              details={"backend":bk,"hidden_dim":hd,"phase":"retrieve","buffer_items":buf_n})
                    del wm; gc.collect()
                except Exception as e:
                    print(f"    [ERROR] wm_{bk}_h{hd}: {e}"); traceback.print_exc()

    # =======================================================================
    # Category 6: Full Pipeline Throughput
    #
    # End-to-end benchmark: encoder outputs -> workspace output.
    # Uses 3 modalities (vision=512, text=512, audio=256) with batch_size=4.
    # Benchmarks both the base GlobalWorkspace and the improved
    # SelectionBroadcastWorkspace.  For the base workspace, also
    # provides per-stage breakdown (projection, competition, memory, broadcast).
    # Reports steps/sec metric.
    # =======================================================================

    def bench_full_pipeline(self) -> None:
        """Benchmark end-to-end workspace pipeline throughput."""
        if not _workspace_available:
            print(f"  [SKIPPED] -- {_workspace_source}"); self._skip("full_pipeline", _workspace_source); return
        print("  Running full pipeline throughput benchmarks...")
        ws_dim, B = 512, 4
        mod_dims = {"vision":512,"text":512,"audio":256}

        # Base GlobalWorkspace
        try:
            self._reset_mem(); torch.manual_seed(42)
            cfg = GlobalWorkspaceConfig(workspace_dim=ws_dim, num_heads=8, capacity_limit=7, memory_mode="gru")
            gw = _set_inference_mode(GlobalWorkspace(config=cfg, modality_dims=mod_dims).to(self.device))
            inp = {n:torch.randn(B,d,device=self.device) for n,d in mod_dims.items()}

            def _gw(): gw.reset_state(); return gw(inp, return_attention=True)
            with torch.no_grad():
                lats = self._timed(_gw)
            sps = 1000.0/(sum(lats)/len(lats)) if lats else 0.0
            self._rec("pipeline_base_globalworkspace", lats, sps,
                      {"workspace_dim":ws_dim,"batch_size":B,"memory_mode":"gru",
                       "params":_nparams(gw),"steps_per_sec":round(sps,1)})

            # Stage breakdown
            def _proj():
                pj, sl = [], []
                for n,f in inp.items():
                    if n in gw.projections:
                        p, s = gw.projections[n](f); pj.append(p); sl.append(s)
                return torch.stack(pj,1), torch.stack(sl,1)
            with torch.no_grad():
                self._rec("pipeline_base_stage_projection", self._timed(_proj),
                          details={"stage":"projection"})
                ps, ss = _proj()
                self._rec("pipeline_base_stage_competition",
                          self._timed(lambda: gw.competition(ps, ss)), details={"stage":"competition"})
                w, _ = gw.competition(ps, ss); wc = w.sum(dim=1)
                def _mem(): gw.working_memory.reset_state(); return gw.working_memory(wc)
                self._rec("pipeline_base_stage_memory", self._timed(_mem), details={"stage":"memory"})
                gw.working_memory.reset_state(); mo = gw.working_memory(wc)['output']
                self._rec("pipeline_base_stage_broadcast",
                          self._timed(lambda: gw.broadcast(mo)), details={"stage":"broadcast"})
            del gw; gc.collect()
        except Exception as e:
            print(f"    [ERROR] base pipeline: {e}"); traceback.print_exc()

        # SelectionBroadcastWorkspace
        try:
            self._reset_mem(); torch.manual_seed(42)
            sbc = SelectionBroadcastConfig(workspace_dim=ws_dim, num_heads=8, capacity_limit=7,
                                           selection_rounds=3, ignition_threshold=0.3,
                                           broadcast_iterations=2, memory_mode="gru",
                                           use_confidence_gating=True)
            sbw = _set_inference_mode(SelectionBroadcastWorkspace(config=sbc, modality_dims=mod_dims).to(self.device))
            inp = {n:torch.randn(B,d,device=self.device) for n,d in mod_dims.items()}
            mst = {n:torch.randn(B,d,device=self.device) for n,d in mod_dims.items()}
            def _sbw(): sbw.reset_state(); return sbw(inp, modality_states=mst, return_details=True)
            with torch.no_grad():
                lats = self._timed(_sbw)
            sps = 1000.0/(sum(lats)/len(lats)) if lats else 0.0
            self._rec("pipeline_selection_broadcast", lats, sps,
                      {"workspace_dim":ws_dim,"selection_rounds":3,"broadcast_iterations":2,
                       "params":_nparams(sbw),"steps_per_sec":round(sps,1)})
            del sbw; gc.collect()
        except Exception as e:
            print(f"    [ERROR] SB pipeline: {e}"); traceback.print_exc()

    # =======================================================================
    # Category 7: Scaling Analysis
    #
    # Three scaling dimensions:
    #   - workspace_dim: [128, 256, 512, 1024, 2048, 4096]
    #   - batch_size:    [1, 2, 4, 8, 16, 32]
    #   - token_count:   [64, 128, 256, 512, 1024]
    # Reports throughput and memory at each data point.
    # workspace_dim uses GlobalWorkspace; token_count uses AttentionCompetition.
    # =======================================================================

    def bench_scaling_analysis(self) -> None:
        """Benchmark scaling across workspace_dim, batch_size, and token_count."""
        if not _workspace_available:
            print(f"  [SKIPPED] -- {_workspace_source}"); self._skip("scaling_analysis", _workspace_source); return
        print("  Running scaling analysis benchmarks...")
        B_fixed, n_mod = 4, 3

        # workspace_dim scaling
        ws_dims = [128, 512, 2048] if self.quick else [128, 256, 512, 1024, 2048, 4096]
        print("    workspace_dim scaling...")
        for wd in ws_dims:
            try:
                self._reset_mem(); torch.manual_seed(42)
                nh = min(8, wd)
                while wd % nh != 0 and nh > 1: nh -= 1
                md = {f"m{i}":wd for i in range(n_mod)}
                cfg = GlobalWorkspaceConfig(workspace_dim=wd, num_heads=nh, capacity_limit=7,
                                            memory_mode="gru", memory_hidden_dim=wd)
                gw = _set_inference_mode(GlobalWorkspace(config=cfg, modality_dims=md).to(self.device))
                inp = {n:torch.randn(B_fixed,d,device=self.device) for n,d in md.items()}
                def _f(): gw.reset_state(); return gw(inp)
                with torch.no_grad(): lats = self._timed(_f)
                sps = 1000.0/(sum(lats)/len(lats)) if lats else 0.0
                self._rec(f"scaling_wsdim_{wd}", lats, sps,
                          {"axis":"workspace_dim","workspace_dim":wd,"batch_size":B_fixed,
                           "params":_nparams(gw),"steps_per_sec":round(sps,1)})
                del gw; gc.collect()
            except Exception as e:
                print(f"    [ERROR] wsdim={wd}: {e}"); traceback.print_exc()

        # batch_size scaling
        bsizes = [1, 4, 16] if self.quick else [1, 2, 4, 8, 16, 32]
        wd_fix = 512
        print("    batch_size scaling...")
        for bs in bsizes:
            try:
                self._reset_mem(); torch.manual_seed(42)
                md = {f"m{i}":wd_fix for i in range(n_mod)}
                cfg = GlobalWorkspaceConfig(workspace_dim=wd_fix, num_heads=8,
                                            capacity_limit=7, memory_mode="gru")
                gw = _set_inference_mode(GlobalWorkspace(config=cfg, modality_dims=md).to(self.device))
                inp = {n:torch.randn(bs,d,device=self.device) for n,d in md.items()}
                def _f(): gw.reset_state(); return gw(inp)
                with torch.no_grad(): lats = self._timed(_f)
                sps = 1000.0/(sum(lats)/len(lats)) if lats else 0.0
                self._rec(f"scaling_batch_{bs}", lats, bs*sps,
                          {"axis":"batch_size","batch_size":bs,"steps_per_sec":round(sps,1),
                           "samples_per_sec":round(bs*sps,1)})
                del gw; gc.collect()
            except Exception as e:
                print(f"    [ERROR] batch={bs}: {e}"); traceback.print_exc()

        # token_count scaling
        tcounts = [64, 256, 1024] if self.quick else [64, 128, 256, 512, 1024]
        print("    token_count scaling...")
        for T in tcounts:
            try:
                self._reset_mem(); torch.manual_seed(42)
                comp = _set_inference_mode(AttentionCompetition(workspace_dim=wd_fix, num_heads=8,
                             capacity_limit=7, temperature=1.0, dropout=0.0).to(self.device))
                feats = torch.randn(B_fixed, T, wd_fix, device=self.device)
                sals  = torch.randn(B_fixed, T, 1, device=self.device)
                with torch.no_grad(): lats = self._timed(lambda: comp(feats, sals))
                mean_ms = sum(lats)/len(lats) if lats else 1.0
                tps = (B_fixed*T)/mean_ms*1000.0
                self._rec(f"scaling_tokens_{T}", lats, tps,
                          {"axis":"token_count","T_total":T,"tokens_per_sec":round(tps,1)})
                del comp; gc.collect()
            except Exception as e:
                print(f"    [ERROR] tokens={T}: {e}"); traceback.print_exc()

    # =======================================================================
    # Category 8: Memory Profiling
    #
    # Profiles:
    #   - Parameter count for each component of SelectionBroadcastWorkspace
    #   - Peak memory during a forward pass
    #   - Memory usage per iterative round (1/2/3/4/6 rounds)
    #   - Comparison of configurations:
    #       with/without feedback (broadcast_iterations=2 vs 0)
    #       with/without confidence gating
    # =======================================================================

    def bench_memory_profiling(self) -> None:
        """Profile parameter counts, peak memory, per-round cost, and configs."""
        if not _workspace_available:
            print(f"  [SKIPPED] -- {_workspace_source}"); self._skip("memory_profiling", _workspace_source); return
        print("  Running memory profiling...")
        ws_dim, B = 512, 4
        mod_dims = {"vision":512,"text":512,"audio":256}
        sbc = SelectionBroadcastConfig(workspace_dim=ws_dim, num_heads=8, capacity_limit=7,
                                       selection_rounds=3, broadcast_iterations=2,
                                       memory_mode="gru", use_confidence_gating=True)

        # Parameter counts per component
        try:
            torch.manual_seed(42)
            sbw = SelectionBroadcastWorkspace(config=sbc, modality_dims=mod_dims).to(self.device)
            cp = {}
            cp["projections"] = sum(_nparams(p) for p in sbw.projections.values())
            cp["iterative_competition"] = _nparams(sbw.competition)
            cp["working_memory"] = _nparams(sbw.working_memory)
            cp["refined_broadcast"] = _nparams(sbw.broadcast)
            cp["integration"] = _nparams(sbw.integration)
            if hasattr(sbw, 'confidence_estimator'):
                cp["confidence_estimator"] = _nparams(sbw.confidence_estimator)
            cp["total"] = _nparams(sbw)
            self._rec("memory_parameter_counts", [],
                      details={"component_params":cp,
                               "component_params_formatted":{k:_fmt_params(v) for k,v in cp.items()},
                               "total_params_formatted":_fmt_params(cp["total"])})
        except Exception as e:
            print(f"    [ERROR] param counts: {e}"); traceback.print_exc()

        # Peak memory during forward
        try:
            self._reset_mem(); torch.manual_seed(42)
            sbw = _set_inference_mode(SelectionBroadcastWorkspace(config=sbc, modality_dims=mod_dims).to(self.device))
            inp = {n:torch.randn(B,d,device=self.device) for n,d in mod_dims.items()}
            mst = {n:torch.randn(B,d,device=self.device) for n,d in mod_dims.items()}
            self._reset_mem(); mb = self._mem_mb()
            with torch.no_grad():
                sbw.reset_state(); _ = sbw(inp, modality_states=mst, return_details=True)
            ma = self._mem_mb()
            self._rec("memory_peak_forward", [],
                      details={"memory_before_mb":round(mb,2),"memory_after_mb":round(ma,2),
                               "peak_delta_mb":round(ma-mb,2)})
            del sbw; gc.collect()
        except Exception as e:
            print(f"    [ERROR] peak mem: {e}"); traceback.print_exc()

        # Memory per iterative round
        rlist = [1, 3, 6] if self.quick else [1, 2, 3, 4, 6]
        print("    Memory per iterative round...")
        rmem = {}
        for nr in rlist:
            try:
                self._reset_mem(); torch.manual_seed(42)
                rc = SelectionBroadcastConfig(workspace_dim=ws_dim, num_heads=8, capacity_limit=7,
                                              selection_rounds=nr, broadcast_iterations=2,
                                              memory_mode="gru", use_confidence_gating=True)
                sw = _set_inference_mode(SelectionBroadcastWorkspace(config=rc, modality_dims=mod_dims).to(self.device))
                inp = {n:torch.randn(B,d,device=self.device) for n,d in mod_dims.items()}
                self._reset_mem()
                with torch.no_grad(): sw.reset_state(); _ = sw(inp)
                rmem[nr] = self._mem_mb(); del sw; gc.collect()
            except Exception as e:
                print(f"    [ERROR] round={nr}: {e}"); traceback.print_exc(); rmem[nr] = 0.0
        # Per-round delta
        srt = sorted(rmem.keys()); prd = {}
        for i in range(1, len(srt)):
            d = rmem[srt[i]] - rmem[srt[i-1]]; diff = srt[i] - srt[i-1]
            prd[f"{srt[i-1]}_to_{srt[i]}"] = round(d/diff,2) if diff else 0.0
        self._rec("memory_per_round", [],
                  details={"round_memory_mb":{str(k):round(v,2) for k,v in rmem.items()},
                           "per_round_delta_mb":prd})

        # Config comparison: with/without feedback and confidence
        cfgs = {
            "feedback+confidence": {"broadcast_iterations":2,"use_confidence_gating":True},
            "feedback_only":       {"broadcast_iterations":2,"use_confidence_gating":False},
            "confidence_only":     {"broadcast_iterations":0,"use_confidence_gating":True},
            "bare":                {"broadcast_iterations":0,"use_confidence_gating":False},
        }
        print("    Comparing configurations...")
        cmem = {}
        for cn, ov in cfgs.items():
            try:
                self._reset_mem(); torch.manual_seed(42)
                cc = SelectionBroadcastConfig(workspace_dim=ws_dim, num_heads=8, capacity_limit=7,
                                              selection_rounds=3, memory_mode="gru", **ov)
                sw = _set_inference_mode(SelectionBroadcastWorkspace(config=cc, modality_dims=mod_dims).to(self.device))
                inp = {n:torch.randn(B,d,device=self.device) for n,d in mod_dims.items()}
                par = _nparams(sw)
                self._reset_mem()
                with torch.no_grad(): sw.reset_state(); _ = sw(inp)
                mp = self._mem_mb()
                def _cf(): sw.reset_state(); return sw(inp)
                with torch.no_grad(): lats = self._timed(_cf)
                ml = sum(lats)/len(lats) if lats else 0.0
                cmem[cn] = {"memory_mb":round(mp,2),"params":par,"params_fmt":_fmt_params(par),
                            "latency_ms":round(ml,4),**ov}
                del sw; gc.collect()
            except Exception as e:
                print(f"    [ERROR] cfg {cn}: {e}"); traceback.print_exc()
                cmem[cn] = {"error":str(e)}
        self._rec("memory_config_comparison", [], details={"configurations":cmem})

    # =======================================================================
    # Runner -- dispatches to individual benchmark methods
    # =======================================================================

    def run_category(self, cat: str) -> None:
        """Run a single benchmark category by name."""
        dispatch = {
            "competition_throughput":   self.bench_competition_throughput,
            "iterative_round_overhead": self.bench_iterative_round_overhead,
            "ignition_computation":     self.bench_ignition_computation,
            "broadcast_adapter":        self.bench_broadcast_adapter,
            "working_memory_backend":   self.bench_working_memory_backend,
            "full_pipeline":            self.bench_full_pipeline,
            "scaling_analysis":         self.bench_scaling_analysis,
            "memory_profiling":         self.bench_memory_profiling,
        }
        if cat not in dispatch:
            print(f"  Unknown category: {cat}\n  Available: {', '.join(dispatch)}"); return
        print(f"\n[{cat}]")
        try:
            dispatch[cat]()
        except Exception as e:
            print(f"  [FATAL] {cat}: {e}"); traceback.print_exc()

    def run_all(self) -> None:
        """Run all 8 benchmark categories in order."""
        for cat in ALL_CATEGORIES:
            self.run_category(cat)

    # =======================================================================
    # Reporting -- pretty-printed tables and JSON output
    # =======================================================================

    @staticmethod
    def _cat_of(name: str) -> str:
        """Infer benchmark category from result name prefix."""
        for pfx, cat in [("competition_","competition_throughput"),("iterative_","iterative_round_overhead"),
                          ("ignition_","ignition_computation"),("broadcast_","broadcast_adapter"),
                          ("wm_","working_memory_backend"),("pipeline_","full_pipeline"),
                          ("scaling_","scaling_analysis"),("memory_","memory_profiling")]:
            if name.startswith(pfx): return cat
        return "other"

    @staticmethod
    def _tp_unit(name: str) -> str:
        """Infer the appropriate throughput unit from a benchmark name."""
        if "token" in name or "competition" in name:
            return "tok/s"
        if "batch" in name:
            return "samp/s"
        return "steps/s"

    def report(self) -> str:
        """
        Generate a formatted benchmark report with per-result details
        followed by summary tables for each category.

        Returns:
            Multi-line string suitable for printing to console
        """
        L = ["","="*80,"  Global Workspace Benchmark Report","="*80,
             f"  Device: {self.device}  Quick: {self.quick}  Warmup: {self.warmup_runs}  Measured: {self.measured_runs}",
             f"  Workspace: {_workspace_source}  WM: {_working_memory_source}  NCPS: {NCPS_AVAILABLE}",
             "-"*80]
        cur = ""
        for r in self.results:
            c = self._cat_of(r.name)
            if c != cur:
                cur = c; L.append(f"\n  {'='*40}\n  [{c}]\n  {'='*40}")
            if r.details.get("status") == "SKIPPED":
                L.append(f"\n    {r.name}: SKIPPED ({r.details.get('reason','')})"); continue
            L.append(f"\n    {r.name}:")
            if r.mean_time > 0:   L.append(f"      Mean:  {_fmt_ms(r.mean_time):>14s}   Std: {_fmt_ms(r.std_time):>14s}")
            if r.throughput > 0:  L.append(f"      Throughput: {_fmt_tp(r.throughput, self._tp_unit(r.name)):>16s}")
            if r.memory_mb > 0:   L.append(f"      Memory: {r.memory_mb:>10.1f} MB")
            for k,v in r.details.items():
                if k in ("status","reason","component_params","component_params_formatted",
                         "configurations","round_memory_mb","per_round_delta_mb"): continue
                if isinstance(v,float): L.append(f"      {k}: {v:.4f}")
                elif isinstance(v,(dict,list)): pass
                else: L.append(f"      {k}: {v}")

        # Summary tables
        L.extend(["","="*80,"  SUMMARY TABLES","="*80])

        # Competition
        cr = [r for r in self.results if r.name.startswith("competition_full_")]
        if cr:
            L.extend(["","  Competition Throughput:",
                       f"  {'T':>8}  {'Mean(ms)':>10}  {'Std(ms)':>10}  {'Tokens/sec':>14}  {'MB':>8}",
                       f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*14}  {'-'*8}"])
            for r in cr:
                L.append(f"  {r.details.get('T_total','?'):>8}  {r.mean_time:>10.4f}  {r.std_time:>10.4f}  "
                         f"{_fmt_tp(r.throughput):>14s}  {r.memory_mb:>8.1f}")

        # Iterative rounds
        ir = [r for r in self.results if r.name.startswith("iterative_rounds_")]
        if ir:
            L.extend(["","  Iterative Round Overhead:",
                       f"  {'Rnd':>5}  {'Mean(ms)':>10}  {'Std(ms)':>10}  {'Per-Rnd(ms)':>12}  {'Conv':>6}",
                       f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*6}"])
            for r in ir:
                L.append(f"  {r.details.get('max_rounds','?'):>5}  {r.mean_time:>10.4f}  {r.std_time:>10.4f}  "
                         f"{r.details.get('time_per_round_ms',0):>12.4f}  {r.details.get('avg_convergence_round','?'):>6}")

        # Working memory
        wf = [r for r in self.results if "_forward_h" in r.name and r.name.startswith("wm_")]
        if wf:
            L.extend(["","  Working Memory Forward:",
                       f"  {'Back':>6}  {'H':>6}  {'Mean(ms)':>10}  {'Std(ms)':>10}  {'Params':>8}  {'MB':>8}",
                       f"  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}"])
            for r in wf:
                L.append(f"  {r.details.get('backend','?'):>6}  {r.details.get('hidden_dim','?'):>6}  "
                         f"{r.mean_time:>10.4f}  {r.std_time:>10.4f}  "
                         f"{_fmt_params(r.details.get('params',0)):>8}  {r.memory_mb:>8.1f}")

        # WS dim scaling
        ws = [r for r in self.results if r.name.startswith("scaling_wsdim_")]
        if ws:
            L.extend(["","  Workspace Dim Scaling:",
                       f"  {'Dim':>6}  {'Mean(ms)':>10}  {'Steps/s':>10}  {'Params':>8}  {'MB':>8}",
                       f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}"])
            for r in ws:
                L.append(f"  {r.details.get('workspace_dim','?'):>6}  {r.mean_time:>10.4f}  "
                         f"{r.details.get('steps_per_sec',0):>10.1f}  "
                         f"{_fmt_params(r.details.get('params',0)):>8}  {r.memory_mb:>8.1f}")

        # Batch scaling
        bs = [r for r in self.results if r.name.startswith("scaling_batch_")]
        if bs:
            L.extend(["","  Batch Size Scaling:",
                       f"  {'B':>4}  {'Mean(ms)':>10}  {'Samp/s':>12}  {'MB':>8}",
                       f"  {'-'*4}  {'-'*10}  {'-'*12}  {'-'*8}"])
            for r in bs:
                L.append(f"  {r.details.get('batch_size','?'):>4}  {r.mean_time:>10.4f}  "
                         f"{r.details.get('samples_per_sec',0):>12.1f}  {r.memory_mb:>8.1f}")

        # Token scaling
        ts = [r for r in self.results if r.name.startswith("scaling_tokens_")]
        if ts:
            L.extend(["","  Token Count Scaling:",
                       f"  {'T':>6}  {'Mean(ms)':>10}  {'Tok/s':>14}  {'MB':>8}",
                       f"  {'-'*6}  {'-'*10}  {'-'*14}  {'-'*8}"])
            for r in ts:
                L.append(f"  {r.details.get('T_total','?'):>6}  {r.mean_time:>10.4f}  "
                         f"{_fmt_tp(r.details.get('tokens_per_sec',0)):>14s}  {r.memory_mb:>8.1f}")

        # Param breakdown
        pr = next((r for r in self.results if r.name == "memory_parameter_counts"), None)
        if pr:
            cf = pr.details.get("component_params_formatted", {})
            if cf:
                L.extend(["","  Parameter Breakdown:",
                           f"  {'Component':>25}  {'Params':>10}",f"  {'-'*25}  {'-'*10}"])
                for k,v in cf.items():
                    L.append(f"  {k:>25}  {v:>10}")

        # Per-round memory table
        mr = next((r for r in self.results if r.name == "memory_per_round"), None)
        if mr:
            rmem = mr.details.get("round_memory_mb", {})
            prd = mr.details.get("per_round_delta_mb", {})
            if rmem:
                L.extend(["", "  Memory per Iterative Round:",
                           f"  {'Rounds':>8}  {'Peak (MB)':>12}",
                           f"  {'-'*8}  {'-'*12}"])
                for rnd in sorted(rmem.keys(), key=lambda x: int(x)):
                    L.append(f"  {rnd:>8}  {rmem[rnd]:>12.1f}")
            if prd:
                L.append("")
                L.append("  Per-round deltas:")
                for span, delta in prd.items():
                    L.append(f"    {span}: {delta:+.2f} MB/round")

        # Config comparison
        cc = next((r for r in self.results if r.name == "memory_config_comparison"), None)
        if cc:
            cfs = cc.details.get("configurations", {})
            if cfs:
                L.extend(["", "  Configuration Comparison (feedback / confidence gating):",
                           f"  {'Config':>30}  {'Params':>8}  {'Lat(ms)':>10}  {'MB':>8}",
                           f"  {'-'*30}  {'-'*8}  {'-'*10}  {'-'*8}"])
                for n, i in cfs.items():
                    if "error" in i:
                        L.append(f"  {n:>30}  ERROR: {i['error']}")
                    else:
                        L.append(
                            f"  {n:>30}  {i.get('params_fmt','?'):>8}  "
                            f"{i.get('latency_ms',0):>10.4f}  "
                            f"{i.get('memory_mb',0):>8.1f}"
                        )

        # Ignition comparison summary
        ig = next((r for r in self.results if r.name == "ignition_comparison"), None)
        if ig:
            d = ig.details
            L.extend(["", "  Ignition Mode Comparison:",
                       f"    Interpretable: {d.get('interpretable_ms', 0):.4f} ms",
                       f"    Learned (NN):  {d.get('learned_ms', 0):.4f} ms",
                       f"    Ratio (interp/learned): {d.get('speedup', 0):.3f}x"])

        # Pipeline comparison summary
        pb = next((r for r in self.results if r.name == "pipeline_base_globalworkspace"), None)
        ps = next((r for r in self.results if r.name == "pipeline_selection_broadcast"), None)
        if pb and ps and pb.mean_time > 0 and ps.mean_time > 0:
            ratio = ps.mean_time / pb.mean_time
            L.extend(["", "  Pipeline Comparison:",
                       f"    Base GlobalWorkspace:        {pb.mean_time:.2f} ms  "
                       f"({pb.details.get('steps_per_sec', 0):.0f} steps/s, "
                       f"{_fmt_params(pb.details.get('params', 0))} params)",
                       f"    SelectionBroadcast:          {ps.mean_time:.2f} ms  "
                       f"({ps.details.get('steps_per_sec', 0):.0f} steps/s, "
                       f"{_fmt_params(ps.details.get('params', 0))} params)",
                       f"    SB / Base latency ratio:     {ratio:.2f}x"])

        L.extend(["", "=" * 80])
        return "\n".join(L)

    def to_json(self) -> dict:
        """
        Produce a JSON-serializable dictionary of the full benchmark report.

        Structure:
            {
                "device": "cpu",
                "quick": false,
                "warmup_runs": 3,
                "measured_runs": 10,
                "sources": { ... },
                "pytorch_version": "2.x.y",
                "benchmarks": [ { BenchmarkResult fields }, ... ]
            }
        """
        return {
            "device": str(self.device),
            "quick": self.quick,
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "sources": {
                "workspace": _workspace_source,
                "working_memory": _working_memory_source,
                "ncps_available": NCPS_AVAILABLE,
            },
            "pytorch_version": torch.__version__,
            "benchmarks": [asdict(r) for r in self.results],
        }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    CLI entry point for the Global Workspace benchmark suite.

    Parses arguments and runs the requested benchmarks, producing
    a formatted report to stdout and optionally a JSON file.

    Arguments:
        --device cpu|cuda   Target device (default: cpu)
        --quick             Fewer iterations for fast feedback
        --category NAME     Run only one category
        --json-report PATH  Write JSON results to file
    """
    parser = argparse.ArgumentParser(
        description="Global Workspace Performance Benchmark",
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
        help="Quick mode: fewer iterations and smaller configs for fast feedback",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=ALL_CATEGORIES,
        help="Run only a specific benchmark category",
    )
    parser.add_argument(
        "--json-report",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to save JSON report",
    )
    args = parser.parse_args()

    # Validate CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    # Print header
    print("=" * 80)
    print("  Global Workspace Performance Benchmark")
    print("=" * 80)
    print(f"  Device:           {args.device}")
    print(f"  Quick mode:       {args.quick}")
    print(f"  PyTorch:          {torch.__version__}")
    print(f"  Workspace source: {_workspace_source}")
    print(f"  Memory source:    {_working_memory_source}")
    print(f"  NCPS available:   {NCPS_AVAILABLE}")
    if args.category:
        print(f"  Category:         {args.category}")
    print("-" * 80)

    # Run benchmarks
    suite = BenchmarkSuite(device=args.device, quick=args.quick)

    if args.category:
        suite.run_category(args.category)
    else:
        suite.run_all()

    # Print formatted report
    print(suite.report())

    # Save JSON report if requested
    if args.json_report:
        report_dir = os.path.dirname(os.path.abspath(args.json_report))
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir, exist_ok=True)
        with open(args.json_report, "w") as f:
            json.dump(suite.to_json(), f, indent=2, default=str)
        print(f"\nJSON report saved to {args.json_report}")

    # Final summary line
    total = len(suite.results)
    skipped = sum(1 for r in suite.results if r.details.get("status") == "SKIPPED")
    ran = total - skipped
    print(f"\nBenchmarks: {ran} ran, {skipped} skipped, {total} total")


if __name__ == "__main__":
    main()
