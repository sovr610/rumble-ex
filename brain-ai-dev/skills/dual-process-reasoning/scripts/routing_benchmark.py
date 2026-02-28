#!/usr/bin/env python3
"""
Dual-Process Reasoning Benchmark -- S1 latency, S2 convergence speed,
routing overhead, and selective execution efficiency.

Runs on CPU (and optionally CUDA) with synthetic data.  Target: <2 min CPU.

Usage:
    python routing_benchmark.py [--device cpu|cuda] [--config minimal|dev]
    python routing_benchmark.py --seed 42 --warmup 3 --runs 10
    python routing_benchmark.py --output results.json --device cuda
    python routing_benchmark.py --suite 1 3 5

Benchmark Suites:
    1. system1_throughput     -- S1 forward pass latency, batch sizes, pooling
    2. calibration_overhead   -- TemperatureScaler, IsotonicCalibrator cost
    3. system2_convergence    -- S2 convergence speed, eps sweep, selective exec
    4. routing_overhead       -- Routing decision time, novelty scoring, budget
    5. full_pipeline          -- End-to-end DualProcessReasoner forward
    6. selective_execution    -- Scatter/gather speedup vs full-batch S2
    7. scaling_analysis       -- hidden_dim, max_steps, num_prototypes scaling

Exit code 0 when all sanity checks pass; 1 otherwise.

Output format:
    === Dual-Process Reasoning Benchmark ===
    Device: cpu | Config: minimal | Seed: 42

    --- Suite 1: System 1 Throughput ---
      Batch=1:   S1 latency=0.8ms  (1250 items/sec)
      Batch=32:  S1 latency=3.2ms  (10000 items/sec)
      ...

    --- Suite 2: Calibration Overhead ---
      Temperature.calibrate: 0.01ms (negligible)
      ...

    --- Sanity Checks ---
      [PASS] 1. S1 output shape
      [PASS] 2. Calibrated confidence range
      ...
      10/10 sanity checks passed

    === Benchmark Summary (JSON) ===
    {
      "device": "cpu",
      "s1_throughput_items_sec": 10000,
      "s2_avg_convergence_steps": 4.2,
      "routing_overhead_ms": 0.3,
      ...
    }
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SKILL_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SKILL_DIR)))
sys.path.insert(0, _PROJECT_ROOT)
_ASSET_DIR = os.path.join(_SKILL_DIR, "assets")
if os.path.isdir(_ASSET_DIR) and _ASSET_DIR not in sys.path:
    sys.path.insert(0, _ASSET_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Try to import from the project tree; fall back to self-contained stubs.
# ---------------------------------------------------------------------------
_reasoning_available = False
_reasoning_source = "self-contained stubs"
try:
    from brain_ai.reasoning.system2 import (  # type: ignore[import-not-found]
        System1Module,
        System2Module,
        MetacognitionModule,
        DualProcessReasoner,
        System2Config,
    )
    _reasoning_available = True
    _reasoning_source = "brain_ai.reasoning.system2"
except ImportError:
    pass
if not _reasoning_available:
    for _mn in ["dual_process_template", "system1_template"]:
        try:
            __import__(_mn)
            _reasoning_available = True
            _reasoning_source = f"{_mn} (asset)"
            break
        except ImportError:
            pass

# ---------------------------------------------------------------------------
# Architecture notes:
#
# The dual-process reasoning pipeline implements Kahneman's System 1/2 theory:
#   System 1 (fast): Single-pass MLP producing predictions + confidence metrics
#   System 2 (slow): GRU-based iterative refinement with convergence detection
#   Metacognition:   Deterministic routing based on confidence, novelty, anomaly
#
# Benchmark coverage:
#   - Suite 1 measures raw System 1 throughput across batch sizes and pooling
#   - Suite 2 profiles calibration (temperature scaling, isotonic regression)
#   - Suite 3 characterizes System 2 convergence behavior and per-step cost
#   - Suite 4 isolates metacognitive routing overhead and novelty scoring
#   - Suite 5 measures the full integrated pipeline end-to-end
#   - Suite 6 quantifies scatter/gather savings from selective S2 execution
#   - Suite 7 shows how throughput scales with model dimensions
#
# All timings use high-resolution perf_counter with optional CUDA sync.
# Results are deterministic for a given seed + device configuration.
# ---------------------------------------------------------------------------


# ===========================================================================
# Self-contained module stubs
# ===========================================================================
#
# When brain_ai is not installed (or the reasoning templates haven't been
# generated yet), this benchmark uses self-contained stub implementations
# that faithfully match the SKILL.md contracts. This ensures the benchmark
# can run standalone without any external dependencies beyond torch and json.
#
# The stubs implement:
#   - DualProcessFullConfig with 5 scale presets (minimal through 7B)
#   - TemperatureScaler with L-BFGS fitting
#   - IsotonicCalibrator with PAVA algorithm
#   - System1Fast with confidence heads and pooling modes
#   - System2Iterative with GRU refinement and convergence detection
#   - MetacognitiveRouter with 3 novelty scoring methods
#   - DualProcessReasonerBench wiring everything together
#
# ===========================================================================

# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class System1Config:
    """System 1 fast predictor configuration.

    Attributes:
        input_dim:       Workspace representation dimension (matches encoder output)
        hidden_dim:      MLP hidden layer width
        output_dim:      Output logits/embedding dimension
        num_layers:      MLP depth (1-4 layers)
        confidence_head: Enable dedicated learned confidence head
        dropout:         Dropout rate for regularization
        pooling_mode:    How to pool (B,K,D) -> (B,D): mean, attention, or cls
    """
    input_dim: int = 4096
    hidden_dim: int = 512
    output_dim: int = 256
    num_layers: int = 2
    confidence_head: bool = True
    dropout: float = 0.1
    pooling_mode: str = "mean"


@dataclass
class System2Config_Full:
    """System 2 iterative refinement configuration.

    Attributes:
        hidden_dim:           GRU hidden state dimension
        output_dim:           Output logits/embedding dimension
        max_steps:            Maximum refinement iterations
        convergence_eps:      KL divergence stability threshold
        convergence_patience: Consecutive stable steps required to halt
        nan_guard:            Halt on NaN detection (safety mechanism)
    """
    hidden_dim: int = 512
    output_dim: int = 256
    max_steps: int = 10
    convergence_eps: float = 1e-3
    convergence_patience: int = 2
    nan_guard: bool = True


@dataclass
class MetacognitionConfig:
    """Metacognitive routing configuration.

    Controls the routing policy that determines which items go to System 2.
    All weights and thresholds are configurable to tune the S1/S2 tradeoff.
    """
    route_threshold: float = 0.5
    w_conf: float = 1.0
    w_novelty: float = 0.5
    w_anomaly: float = 0.3
    w_budget: float = 0.1
    min_conf_to_skip_s2: float = 0.95
    base_steps: int = 3
    step_scale_alpha: float = 5.0
    always_run_s2: bool = False
    novelty_method: str = "prototype"
    num_prototypes: int = 64


@dataclass
class CalibrationConfig:
    """Calibration configuration for confidence post-processing."""
    method: str = "temperature"
    initial_temperature: float = 1.5
    freeze_after_fit: bool = True


@dataclass
class DualProcessFullConfig:
    """Aggregated configuration for the full dual-process reasoning module.

    Bundles System 1, System 2, metacognition, calibration, and trace
    settings into a single object with factory presets for different
    deployment scales.
    """
    s1: System1Config = field(default_factory=System1Config)
    s2: System2Config_Full = field(default_factory=System2Config_Full)
    meta: MetacognitionConfig = field(default_factory=MetacognitionConfig)
    cal: CalibrationConfig = field(default_factory=CalibrationConfig)
    enable_trace: bool = True

    @classmethod
    def minimal(cls) -> "DualProcessFullConfig":
        """Create minimal configuration preset (~10K params, for tests)."""
        return cls(
            s1=System1Config(128, 64, 32, 1),
            s2=System2Config_Full(64, 32, 5),
            meta=MetacognitionConfig(num_prototypes=16),
            cal=CalibrationConfig(),
        )

    @classmethod
    def dev(cls) -> "DualProcessFullConfig":
        """Create dev configuration preset (~1M params, for iteration)."""
        return cls(
            s1=System1Config(256, 128, 64, 2),
            s2=System2Config_Full(128, 64, 8),
            meta=MetacognitionConfig(num_prototypes=32),
            cal=CalibrationConfig(),
        )

    @classmethod
    def production_1b(cls) -> "DualProcessFullConfig":
        """Create production_1b configuration preset."""
        return cls(
            s1=System1Config(2048, 512, 256, 3),
            s2=System2Config_Full(512, 256, 10),
            meta=MetacognitionConfig(num_prototypes=128),
            cal=CalibrationConfig(),
        )

    @classmethod
    def production_3b(cls) -> "DualProcessFullConfig":
        """Create production_3b configuration preset."""
        return cls(
            s1=System1Config(4096, 1024, 512, 3),
            s2=System2Config_Full(1024, 512, 12),
            meta=MetacognitionConfig(num_prototypes=256),
            cal=CalibrationConfig(),
        )

    @classmethod
    def production_7b(cls) -> "DualProcessFullConfig":
        """Create production_7b configuration preset."""
        return cls(
            s1=System1Config(4096, 2048, 1024, 4),
            s2=System2Config_Full(2048, 1024, 15),
            meta=MetacognitionConfig(num_prototypes=512),
            cal=CalibrationConfig(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire config tree to plain dict."""
        return asdict(self)


# ===========================================================================
# TemperatureScaler
# ===========================================================================


class TemperatureScaler(nn.Module):
    """Platt temperature scaling for confidence calibration.

    Learns a single scalar temperature parameter T such that
    calibrated_probs = softmax(logits / T). The temperature is
    optimized on a held-out validation set using L-BFGS to minimize
    negative log-likelihood.

    After fitting, the temperature should be frozen (freeze_after_fit=True)
    to prevent calibration drift during subsequent operations.

    Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
    """

    def __init__(self, initial_temperature: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(initial_temperature, dtype=torch.float32)
        )
        self._fitted: bool = False

    def calibrate(self, logits: Tensor) -> Tensor:
        """Apply temperature scaling to logits and return softmax probs."""
        return F.softmax(logits / self.temperature.clamp(min=0.01), dim=-1)

    def fit(
        self,
        val_logits: Tensor,
        val_labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """Fit temperature to validation data via L-BFGS. Returns final NLL."""
        opt = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def closure() -> Tensor:
            opt.zero_grad()
            loss = nll(val_logits / self.temperature.clamp(min=0.01), val_labels)
            loss.backward()
            return loss

        opt.step(closure)
        self._fitted = True
        with torch.no_grad():
            return nll(
                val_logits / self.temperature.clamp(min=0.01), val_labels
            ).item()

    def forward(self, logits: Tensor) -> Tensor:
        """Forward pass delegates to calibrate."""
        return self.calibrate(logits)


# ===========================================================================
# IsotonicCalibrator
# ===========================================================================


class IsotonicCalibrator:
    """Isotonic regression calibrator using a pure-torch implementation.

    Bins raw confidence values and computes mean accuracy per bin, then
    enforces monotonicity via the Pool Adjacent Violators Algorithm (PAVA).
    At inference time, maps raw confidence to calibrated probability using
    the fitted piecewise-constant function.

    No external dependencies (no sklearn). Suitable for checkpoint
    serialization alongside the model.
    """

    def __init__(self, num_bins: int = 15) -> None:
        self.num_bins: int = num_bins
        self.bin_edges: Optional[Tensor] = None
        self.bin_values: Optional[Tensor] = None
        self._fitted: bool = False

    def fit(self, raw_confs: Tensor, correct: Tensor) -> None:
        """Fit isotonic calibrator to raw confidences and binary labels."""
        idx = raw_confs.argsort()
        sc = raw_confs[idx]
        sl = correct[idx].float()
        n = sc.shape[0]
        bs = max(n // self.num_bins, 1)
        edges: List[float] = []
        vals: List[float] = []
        for i in range(0, n, bs):
            edges.append(sc[i].item())
            vals.append(sl[i : min(i + bs, n)].mean().item())
        # PAVA monotonicity enforcement
        for i in range(1, len(vals)):
            if vals[i] < vals[i - 1]:
                vals[i] = vals[i - 1]
        self.bin_edges = torch.tensor(edges, dtype=torch.float32)
        self.bin_values = torch.tensor(vals, dtype=torch.float32)
        self._fitted = True

    def calibrate(self, raw_confs: Tensor) -> Tensor:
        """Map raw confidences to calibrated values using fitted function."""
        if not self._fitted:
            return raw_confs
        e = self.bin_edges.to(raw_confs.device)
        v = self.bin_values.to(raw_confs.device)
        indices = torch.searchsorted(e, raw_confs.contiguous())
        return v[indices.clamp(0, len(v) - 1)]


# ===========================================================================
# System1Fast
# ===========================================================================


class System1Fast(nn.Module):
    """System 1: Single-pass fast predictor with confidence heads.

    Accepts (B, D) pooled workspace representations or (B, K, D) slot inputs
    with configurable pooling (mean, attention, cls). Outputs prediction logits
    plus multiple uncertainty metrics for routing decisions.

    Uncertainty metrics produced:
        - conf_raw: max softmax probability (raw confidence)
        - conf_calibrated: dedicated confidence head output (learned calibration)
        - entropy: distribution spread (-sum(p * log(p)))
        - margin: top1_logit - top2_logit (decision boundary distance)

    These metrics feed into the MetacognitiveRouter for S1/S2 routing.
    """

    def __init__(self, cfg: System1Config) -> None:
        super().__init__()
        self.cfg = cfg

        # Pooling components
        if cfg.pooling_mode == "attention":
            self.pool_query = nn.Linear(cfg.input_dim, 1)
        elif cfg.pooling_mode == "cls":
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, cfg.input_dim) * 0.02
            )

        # MLP body
        layers: List[nn.Module] = []
        d = cfg.input_dim
        for _ in range(cfg.num_layers):
            layers.extend(
                [nn.Linear(d, cfg.hidden_dim), nn.GELU(), nn.Dropout(cfg.dropout)]
            )
            d = cfg.hidden_dim
        layers.append(nn.Linear(d, cfg.output_dim))
        self.mlp = nn.Sequential(*layers)

        # Optional learned confidence head
        self.conf_head: Optional[nn.Module] = None
        if cfg.confidence_head:
            self.conf_head = nn.Sequential(
                nn.Linear(cfg.input_dim, cfg.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

    def _pool(self, x: Tensor) -> Tensor:
        """Pool (B,K,D) slot input to (B,D). Pass through (B,D) unchanged."""
        if x.dim() == 2:
            return x
        m = self.cfg.pooling_mode
        if m == "attention":
            w = F.softmax(self.pool_query(x).squeeze(-1), dim=-1).unsqueeze(-1)
            return (x * w).sum(dim=1)
        if m == "cls":
            return x[:, 0]
        # Default: mean pooling
        return x.mean(dim=1)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Run System 1 forward pass. Returns dict with y1 and metrics."""
        p = self._pool(x)
        logits = self.mlp(p)
        probs = F.softmax(logits, dim=-1)
        conf_raw = probs.max(dim=-1).values
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
        if logits.shape[-1] >= 2:
            topk_vals = logits.topk(2, dim=-1).values
            margin = topk_vals[:, 0] - topk_vals[:, 1]
        else:
            margin = torch.zeros_like(conf_raw)
        conf_cal = (
            self.conf_head(p).squeeze(-1) if self.conf_head is not None else conf_raw
        )
        return {
            "y1": logits,
            "conf_raw": conf_raw,
            "conf_calibrated": conf_cal,
            "entropy": entropy,
            "margin": margin,
            "logits": logits,
        }


# ===========================================================================
# System2Iterative
# ===========================================================================


class System2Iterative(nn.Module):
    """System 2: GRU-based iterative refinement loop with convergence detection.

    Starting from System 1's initial prediction, iteratively refines the output
    using a GRU recurrence until convergence is detected or the step budget is
    exhausted.

    Convergence criterion:
        KL(p_k || p_{k-1}) < convergence_eps for ``convergence_patience``
        consecutive steps.

    Halt reasons:
        - "converged": KL stability achieved
        - "max_steps": reached cfg.max_steps without converging
        - "budget_exhausted": per-item budget (from metacognition) hit
        - "nan_guard": NaN detected in output (safety halt)

    Only items routed to S2 enter the loop (scatter/gather pattern for
    efficiency -- see Suite 6 benchmarks for speedup measurements).
    """

    def __init__(
        self, cfg: System2Config_Full, input_dim: int = 128
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim), nn.GELU()
        )
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

    def forward(
        self,
        x: Tensor,
        y1: Tensor,
        steps_budget: Optional[Tensor] = None,
        return_trace: bool = False,
    ) -> Dict[str, Any]:
        """Run iterative refinement loop.

        Args:
            x: Input summary (B, input_dim).
            y1: System 1 initial output (B, output_dim).
            steps_budget: Optional per-item budget tensor (B,).
            return_trace: Whether to collect per-step trace data.

        Returns:
            Dict with y2, steps_used, converged, halt_reason, and
            optionally trace.
        """
        B = x.shape[0]
        dev = x.device
        ms = self.cfg.max_steps
        h = self.state_encoder(x)
        p_prev = F.softmax(y1, dim=-1)
        steps_used = torch.ones(B, dtype=torch.long, device=dev)
        converged = torch.zeros(B, dtype=torch.bool, device=dev)
        halt_reason: List[str] = ["max_steps"] * B
        active = torch.ones(B, dtype=torch.bool, device=dev)
        pat = torch.zeros(B, dtype=torch.long, device=dev)

        if steps_budget is not None:
            pim = steps_budget.long().clamp(1, ms)
        else:
            pim = torch.full((B,), ms, dtype=torch.long, device=dev)

        trace: List[Dict[str, Any]] = []

        for step in range(ms):
            h = self.gru(self.output_proj[0](h), h)
            y_c = self.output_proj(h)
            p_c = F.softmax(y_c, dim=-1)

            # NaN guard
            if self.cfg.nan_guard:
                nm = torch.isnan(y_c).any(dim=-1)
                if nm.any():
                    for i in nm.nonzero(as_tuple=False).squeeze(-1).tolist():
                        if active[i]:
                            active[i] = False
                            halt_reason[i] = "nan_guard"

            # KL convergence check
            kl = F.kl_div(
                (p_c + 1e-8).log(), p_prev, reduction="none"
            ).sum(-1)
            stable = kl < self.cfg.convergence_eps
            pat = torch.where(
                active & stable, pat + 1, torch.zeros_like(pat)
            )
            nc = active & (pat >= self.cfg.convergence_patience)
            be = active & ((step + 1) >= pim)

            for i in nc.nonzero(as_tuple=False).squeeze(-1).tolist():
                if active[i]:
                    active[i] = False
                    converged[i] = True
                    halt_reason[i] = "converged"
                    steps_used[i] = step + 1

            for i in be.nonzero(as_tuple=False).squeeze(-1).tolist():
                if active[i]:
                    active[i] = False
                    halt_reason[i] = "budget_exhausted"
                    steps_used[i] = step + 1

            if return_trace:
                trace.append({
                    "step": step,
                    "kl_mean": kl.mean().item(),
                    "active_frac": active.float().mean().item(),
                })

            p_prev = p_c

            if not active.any():
                break

        # Finalize remaining active items
        for i in active.nonzero(as_tuple=False).squeeze(-1).tolist():
            steps_used[i] = ms
            halt_reason[i] = "max_steps"

        result: Dict[str, Any] = {
            "y2": self.output_proj(h),
            "steps_used": steps_used,
            "converged": converged,
            "halt_reason": halt_reason,
        }
        if return_trace:
            result["trace"] = trace
        return result


# ===========================================================================
# MetacognitiveRouter
# ===========================================================================


class MetacognitiveRouter(nn.Module):
    """Metacognitive Router: deterministic routing policy.

    Computes a route score from multiple signals:
        route_score = w_conf * (1 - calibrated_conf)
                    + w_novelty * novelty
                    + w_anomaly * anomaly

    Routing decision:
        used_system2 = route_score >= route_threshold

    Steps budget allocation:
        steps_budget = clamp(round(base_steps + alpha * route_score), 1, max_steps)

    Properties:
        - Fully deterministic (no sampling, no nondeterministic GPU ops)
        - Per-item routing (no cross-batch dependencies)
        - Hard skip for very confident items (conf >= min_conf_to_skip_s2)

    Novelty scoring methods:
        - prototype: min L2 distance to prototype bank
        - knn: mean distance to k=5 nearest prototypes
        - mahalanobis: Mahalanobis distance to prototype centroid
    """

    def __init__(
        self, cfg: MetacognitionConfig, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "prototypes",
            torch.randn(cfg.num_prototypes, hidden_dim) * 0.1,
        )
        self.novelty_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def compute_novelty_prototype(self, x: Tensor) -> Tensor:
        """Novelty as sigmoid of min L2 distance to prototypes."""
        dists = torch.cdist(
            x.unsqueeze(0), self.prototypes.unsqueeze(0)
        ).squeeze(0)
        return torch.sigmoid(dists.min(-1).values - 1.0)

    def compute_novelty_knn(self, x: Tensor) -> Tensor:
        """Novelty as sigmoid of mean k-nearest-neighbor distance."""
        k = min(5, self.cfg.num_prototypes)
        dists = torch.cdist(
            x.unsqueeze(0), self.prototypes.unsqueeze(0)
        ).squeeze(0)
        topk = dists.topk(k, dim=-1, largest=False).values
        return torch.sigmoid(topk.mean(-1) - 1.0)

    def compute_novelty_mahalanobis(self, x: Tensor) -> Tensor:
        """Novelty as sigmoid of Mahalanobis distance to prototype centroid."""
        c = self.prototypes.mean(0, keepdim=True)
        v = self.prototypes.var(0, keepdim=True).clamp(min=1e-6)
        return torch.sigmoid(((x - c) ** 2 / v).sum(-1).sqrt() - 2.0)

    def compute_novelty(self, x: Tensor) -> Tensor:
        """Dispatch novelty computation to the configured method."""
        m = self.cfg.novelty_method
        if m == "prototype":
            return self.compute_novelty_prototype(x)
        if m == "knn":
            return self.compute_novelty_knn(x)
        if m == "mahalanobis":
            return self.compute_novelty_mahalanobis(x)
        return self.novelty_net(x).squeeze(-1)

    def forward(
        self,
        x: Tensor,
        conf_calibrated: Tensor,
        anomaly: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute routing decision, novelty, and per-item step budget.

        Args:
            x: Hidden features (B, hidden_dim).
            conf_calibrated: Calibrated confidence (B,).
            anomaly: Optional anomaly scores (B,).

        Returns:
            Dict with used_system2, route_score, novelty, steps_budget.
        """
        B = x.shape[0]
        dev = x.device
        c = self.cfg
        nov = self.compute_novelty(x)
        anom = anomaly if anomaly is not None else torch.zeros(B, device=dev)
        rs = (
            c.w_conf * (1.0 - conf_calibrated)
            + c.w_novelty * nov
            + c.w_anomaly * anom
        )
        rs = torch.where(
            conf_calibrated >= c.min_conf_to_skip_s2,
            torch.zeros_like(rs),
            rs,
        )
        if c.always_run_s2:
            us2 = torch.ones(B, dtype=torch.bool, device=dev)
        else:
            us2 = rs >= c.route_threshold
        sb = (c.base_steps + c.step_scale_alpha * rs).round().long().clamp(1, 15)
        return {
            "used_system2": us2,
            "route_score": rs,
            "novelty": nov,
            "steps_budget": sb,
        }


# ===========================================================================
# Trace and Output dataclasses
# ===========================================================================


@dataclass
class ReasoningTrace:
    """JSON-serializable trace of one dual-process forward pass.

    Contains routing decisions, confidence metrics, System 2 step-by-step
    trace (if S2 was invoked), halt reasons, and novelty scores.
    Used for debugging and interpretability when return_details=True.
    """
    batch_size: int
    s2_fraction: float
    route_scores: List[float]
    s1_top_confs: List[float]
    s2_steps: List[Dict[str, Any]]
    halt_reasons: List[str]
    novelty_scores: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ReasoningOutput:
    """Output from the DualProcessReasoner.

    Attributes:
        y:            Final output logits/embedding (B, output_dim)
        used_system2: Per-item boolean mask (B,) indicating S2 invocation
        s1_result:    System 1 outputs (y1, conf_raw, conf_calibrated, etc.)
        s2_result:    System 2 outputs (y2, steps_used, converged, halt_reason)
        trace:        Full reasoning trace (only when return_details=True)
        aux:          Auxiliary metrics (novelty, route_score, steps_budget, s2_fraction)
    """
    y: Tensor
    used_system2: Tensor
    s1_result: Dict[str, Tensor]
    s2_result: Optional[Dict[str, Any]]
    trace: Optional[ReasoningTrace]
    aux: Dict[str, Any]


# ===========================================================================
# DualProcessReasonerBench
# ===========================================================================


class DualProcessReasonerBench(nn.Module):
    """Full dual-process reasoner wiring S1 + S2 + metacognitive routing.

    This is a self-contained benchmarkable implementation that matches the
    SKILL.md public contract:
        forward(x, *, context=None, return_details=False, state=None) -> ReasoningOutput

    Pipeline flow:
        1. System 1 forward pass -> prediction + confidence metrics
        2. Calibrate confidence (temperature scaling or isotonic)
        3. Metacognitive routing -> per-item S1/S2 decision + step budget
        4. System 2 selective execution on routed items (scatter/gather)
        5. Merge S1 and S2 outputs; optionally build reasoning trace

    Hard invariants:
        - Routing is deterministic for a fixed seed
        - return_details=False produces trace=None with near-zero overhead
        - Per-item routing is computed independently (no cross-batch sorting)
    """

    def __init__(self, cfg: DualProcessFullConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.s1 = System1Fast(cfg.s1)
        self.s2 = System2Iterative(cfg.s2, input_dim=cfg.s1.input_dim)
        self.router = MetacognitiveRouter(
            cfg.meta, hidden_dim=cfg.s1.input_dim
        )
        self.calibrator: Optional[TemperatureScaler] = None
        if cfg.cal.method == "temperature":
            self.calibrator = TemperatureScaler(cfg.cal.initial_temperature)
        self.isotonic: Optional[IsotonicCalibrator] = None
        if cfg.cal.method == "isotonic":
            self.isotonic = IsotonicCalibrator()

    @torch.no_grad()
    def forward(
        self,
        x: Tensor,
        *,
        context: Optional[Tensor] = None,
        return_details: bool = False,
        anomaly: Optional[Tensor] = None,
    ) -> ReasoningOutput:
        """Run the full dual-process pipeline.

        Args:
            x: Input tensor (B, D) or (B, K, D).
            context: Unused, reserved for compatibility.
            return_details: Whether to construct a ReasoningTrace.
            anomaly: Optional anomaly scores (B,).

        Returns:
            ReasoningOutput with final predictions and metadata.
        """
        xp = x.mean(dim=1) if x.dim() == 3 else x

        # System 1
        s1 = self.s1(x)
        conf = s1["conf_calibrated"]

        # Calibrate
        if self.calibrator is not None:
            conf = self.calibrator.calibrate(s1["logits"]).max(-1).values
        if self.isotonic is not None and self.isotonic._fitted:
            conf = self.isotonic.calibrate(conf)

        # Route
        ro = self.router(xp, conf, anomaly=anomaly)
        us2 = ro["used_system2"]

        # System 2 selective execution
        s2r: Optional[Dict[str, Any]] = None
        y = s1["y1"].clone()
        if us2.any():
            s2o = self.s2(
                xp[us2],
                s1["y1"][us2],
                steps_budget=ro["steps_budget"][us2],
                return_trace=return_details,
            )
            y[us2] = s2o["y2"]
            s2r = s2o

        # Trace
        trace: Optional[ReasoningTrace] = None
        if return_details:
            trace = ReasoningTrace(
                batch_size=x.shape[0],
                s2_fraction=us2.float().mean().item(),
                route_scores=ro["route_score"].tolist(),
                s1_top_confs=conf.tolist(),
                s2_steps=s2r.get("trace", []) if s2r else [],
                halt_reasons=s2r["halt_reason"] if s2r else [],
                novelty_scores=ro["novelty"].tolist(),
            )

        return ReasoningOutput(
            y=y,
            used_system2=us2,
            s1_result=s1,
            s2_result=s2r,
            trace=trace,
            aux={
                "novelty": ro["novelty"],
                "route_score": ro["route_score"],
                "steps_budget": ro["steps_budget"],
                "s2_fraction": us2.float().mean().item(),
            },
        )


# ===========================================================================
# Benchmark infrastructure
# ===========================================================================


@dataclass
class BenchmarkResult:
    """Result for a single benchmark measurement point.

    Attributes:
        name:       Unique identifier for this benchmark (e.g., 's1_batch_32')
        mean_time:  Mean latency in milliseconds across all runs
        std_time:   Standard deviation of latency in milliseconds
        throughput: Throughput in items/sec (or tokens/sec where applicable)
        memory_mb:  Peak memory usage in megabytes (if measured)
        details:    Additional metadata (batch_size, per_item_ms, etc.)
    """
    name: str
    mean_time: float = 0.0
    std_time: float = 0.0
    throughput: float = 0.0
    memory_mb: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


def _std(v: List[float]) -> float:
    """Compute sample standard deviation of a list of floats."""
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def _nparams(m: nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _nparams_all(m: nn.Module) -> int:
    """Count all parameters (trainable + frozen) in a PyTorch module."""
    return sum(p.numel() for p in m.parameters())


def _rss_mb() -> float:
    """Get current process resident set size in MB (Linux only)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def _percentile(vals: List[float], pct: float) -> float:
    """Compute a percentile from a list of values (pct in [0, 1])."""
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


def _fmt_p(n: int) -> str:
    """Format parameter count with human-readable suffix (K/M)."""
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _fmt_ms(ms: float) -> str:
    """Format a time value in milliseconds with appropriate unit suffix."""
    if ms < 0.001:
        return f"{ms * 1000:.2f}us"
    if ms < 1.0:
        return f"{ms:.4f}ms"
    if ms < 1000:
        return f"{ms:.2f}ms"
    return f"{ms / 1000:.2f}s"


def _fmt_tp(t: float, u: str = "items/sec") -> str:
    """Format throughput value with human-readable suffix (K/M)."""
    if t >= 1e6:
        return f"{t / 1e6:.2f}M {u}"
    if t >= 1e3:
        return f"{t / 1e3:.1f}K {u}"
    return f"{t:.1f} {u}"


class Timer:
    """High-resolution timer context manager with optional CUDA synchronization.

    On CUDA devices, calls torch.cuda.synchronize() before start and after
    stop to ensure accurate wall-clock timing of GPU operations.

    Usage:
        t = Timer(device, sync=True)
        with t:
            model(input)
        print(f"Elapsed: {t.elapsed_ms:.2f}ms")
    """

    def __init__(self, dev: torch.device, sync: bool = False) -> None:
        self.dev = dev
        self.sync = sync and dev.type == "cuda"
        self.elapsed_ms: float = 0.0
        self._s: float = 0.0

    def __enter__(self) -> "Timer":
        if self.sync:
            torch.cuda.synchronize(self.dev)
        self._s = time.perf_counter()
        return self

    def __exit__(self, *a: Any) -> None:
        if self.sync:
            torch.cuda.synchronize(self.dev)
        self.elapsed_ms = (time.perf_counter() - self._s) * 1000.0


def _bench(
    fn: Callable[[], Any],
    warmup: int,
    runs: int,
    dev: torch.device,
) -> Tuple[float, float, List[float]]:
    """Benchmark a callable with warmup and timed runs.

    Args:
        fn:     Callable to benchmark (no arguments).
        warmup: Number of warmup iterations (not timed).
        runs:   Number of timed iterations.
        dev:    torch.device for CUDA sync.

    Returns:
        Tuple of (mean_ms, std_ms, raw_timings_list).
    """
    sync = dev.type == "cuda"
    for _ in range(warmup):
        fn()
        if sync:
            torch.cuda.synchronize(dev)
    ts: List[float] = []
    for _ in range(runs):
        t = Timer(dev, sync=sync)
        with t:
            fn()
        ts.append(t.elapsed_ms)
    mean_val = sum(ts) / len(ts) if ts else 0.0
    return mean_val, _std(ts), ts


def _seed(s: int) -> None:
    """Set deterministic seeds for torch (CPU and CUDA) reproducibility."""
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ===========================================================================
# Benchmark Suite Functions
#
# Each suite returns:
#   results: List[BenchmarkResult] with timing data
#   lines:   List[str] with human-readable output lines
#
# All suites accept (cfg, device, warmup, runs, seed) as arguments.
# ===========================================================================

# ===========================================================================
# Suite 1: System 1 Throughput
# ===========================================================================


def suite_system1_throughput(
    cfg: DualProcessFullConfig,
    device: torch.device,
    warmup: int,
    runs: int,
    seed: int,
) -> Tuple[List[BenchmarkResult], List[str]]:
    """Benchmark System 1 throughput across batch sizes, input shapes, and pooling modes."""
    R: List[BenchmarkResult] = []
    L: List[str] = []
    L.append("--- Suite 1: System 1 Throughput ---")
    _seed(seed)
    s1 = System1Fast(cfg.s1).to(device)
    s1.train(False)
    L.append(f"  S1 params: {_fmt_p(_nparams(s1))}")

    # 1a: Measure S1 latency across batch sizes [1, 8, 32, 128]
    L.append("")
    L.append("  [1a] Batch size sweep (B, D):")
    for bs in [1, 8, 32, 128]:
        x = torch.randn(bs, cfg.s1.input_dim, device=device)
        m, s, _ = _bench(lambda x=x: s1(x), warmup, runs, device)
        tp = bs / m * 1000 if m > 0 else 0
        L.append(
            f"    Batch={bs:>4d}: {_fmt_ms(m)} +/- {_fmt_ms(s)}"
            f"  ({_fmt_tp(tp)})  per_item={_fmt_ms(m / bs)}"
        )
        R.append(BenchmarkResult(
            f"s1_batch_{bs}", m, s, tp,
            details={"batch_size": bs, "per_item_ms": m / bs},
        ))

    # 1b: Compare (B, D) pooled input vs (B, K, D) slot input with K=8
    L.append("")
    L.append("  [1b] Slot input (B,K,D) vs pooled (B,D):")
    x2 = torch.randn(32, cfg.s1.input_dim, device=device)
    x3 = torch.randn(32, 8, cfg.s1.input_dim, device=device)
    m2, _, _ = _bench(lambda: s1(x2), warmup, runs, device)
    m3, _, _ = _bench(lambda: s1(x3), warmup, runs, device)
    L.append(
        f"    (B,D): {_fmt_ms(m2)}  (B,K,D): {_fmt_ms(m3)}"
        f"  overhead: {_fmt_ms(m3 - m2)}"
    )
    R.append(BenchmarkResult(
        "s1_2d_vs_3d", m3,
        details={"2d_ms": m2, "3d_ms": m3, "overhead_ms": m3 - m2},
    ))

    # 1c: Measure overhead of the dedicated confidence head
    L.append("")
    L.append("  [1c] Confidence head overhead:")
    s1nc = System1Fast(System1Config(
        cfg.s1.input_dim, cfg.s1.hidden_dim, cfg.s1.output_dim,
        cfg.s1.num_layers, confidence_head=False,
    )).to(device)
    s1nc.train(False)
    x = torch.randn(32, cfg.s1.input_dim, device=device)
    mc, _, _ = _bench(lambda: s1(x), warmup, runs, device)
    mn, _, _ = _bench(lambda: s1nc(x), warmup, runs, device)
    L.append(
        f"    With conf: {_fmt_ms(mc)}  Without: {_fmt_ms(mn)}"
        f"  Overhead: {_fmt_ms(mc - mn)}"
    )
    R.append(BenchmarkResult(
        "s1_confidence_overhead", mc - mn,
        details={"with_ms": mc, "without_ms": mn},
    ))

    # 1d: Compare pooling strategies: mean, attention-weighted, CLS token
    L.append("")
    L.append("  [1d] Pooling modes (B=32, K=8):")
    for mode in ["mean", "attention", "cls"]:
        sm = System1Fast(System1Config(
            cfg.s1.input_dim, cfg.s1.hidden_dim, cfg.s1.output_dim,
            cfg.s1.num_layers, pooling_mode=mode,
        )).to(device)
        sm.train(False)
        xp = torch.randn(32, 8, cfg.s1.input_dim, device=device)
        mp, sp, _ = _bench(lambda m=sm, x=xp: m(x), warmup, runs, device)
        L.append(f"    {mode:>10s}: {_fmt_ms(mp)} +/- {_fmt_ms(sp)}")
        R.append(BenchmarkResult(
            f"s1_pool_{mode}", mp, sp, details={"mode": mode},
        ))

    return R, L


# ===========================================================================
# Suite 2: Calibration Overhead
# ===========================================================================


def suite_calibration_overhead(
    cfg: DualProcessFullConfig,
    device: torch.device,
    warmup: int,
    runs: int,
    seed: int,
) -> Tuple[List[BenchmarkResult], List[str]]:
    """Benchmark temperature scaling calibrate latency and fit time."""
    R: List[BenchmarkResult] = []
    L: List[str] = []
    L.append("--- Suite 2: Calibration Overhead ---")
    _seed(seed)

    # 2a: Temperature calibrate latency
    L.append("")
    L.append("  [2a] TemperatureScaler.calibrate latency:")
    sc = TemperatureScaler(cfg.cal.initial_temperature).to(device)
    for bs in [32, 128, 512]:
        lo = torch.randn(bs, cfg.s1.output_dim, device=device)
        m, s, _ = _bench(lambda l=lo: sc.calibrate(l), warmup, runs, device)
        L.append(f"    Batch={bs:>4d}: {_fmt_ms(m)} +/- {_fmt_ms(s)}")
        R.append(BenchmarkResult(
            f"temp_calibrate_bs{bs}", m, s, details={"batch_size": bs},
        ))

    # 2b: Temperature fit time
    L.append("")
    L.append("  [2b] TemperatureScaler.fit time:")
    nc = max(cfg.s1.output_dim, 10)
    for vs in [100, 500, 2000, 10000]:
        _seed(seed)
        vl = torch.randn(vs, nc, device=device)
        vla = torch.randint(0, nc, (vs,), device=device)
        fs = TemperatureScaler(cfg.cal.initial_temperature).to(device)
        t = Timer(device, sync=device.type == "cuda")
        with t:
            fs.fit(vl, vla, max_iter=50)
        L.append(
            f"    val_size={vs:>6d}: fit={_fmt_ms(t.elapsed_ms)}"
            f"  T={fs.temperature.item():.3f}"
        )
        R.append(BenchmarkResult(
            f"temp_fit_{vs}", t.elapsed_ms,
            details={"val_size": vs, "T": fs.temperature.item()},
        ))

    # 2c: Isotonic
    L.append("")
    L.append("  [2c] IsotonicCalibrator overhead:")
    for vs in [500, 2000, 10000]:
        _seed(seed)
        rc = torch.rand(vs, device=device)
        co = (torch.rand(vs, device=device) > 0.3).long()
        iso = IsotonicCalibrator(15)
        t = Timer(device, sync=device.type == "cuda")
        with t:
            iso.fit(rc, co)
        tc = torch.rand(128, device=device)
        mi, si, _ = _bench(lambda c=tc: iso.calibrate(c), warmup, runs, device)
        L.append(
            f"    val_size={vs:>6d}: fit={_fmt_ms(t.elapsed_ms)}"
            f"  calibrate(128)={_fmt_ms(mi)}"
        )
        R.append(BenchmarkResult(
            f"iso_fit_{vs}", t.elapsed_ms,
            details={"val_size": vs, "cal_ms": mi},
        ))

    return R, L


# ===========================================================================
# Suite 3: System 2 Convergence
# ===========================================================================


def suite_system2_convergence(
    cfg: DualProcessFullConfig,
    device: torch.device,
    warmup: int,
    runs: int,
    seed: int,
) -> Tuple[List[BenchmarkResult], List[str]]:
    """Benchmark S2 convergence speed, per-step latency, and selective execution."""
    R: List[BenchmarkResult] = []
    L: List[str] = []
    L.append("--- Suite 3: System 2 Convergence ---")
    _seed(seed)
    s2 = System2Iterative(cfg.s2, input_dim=cfg.s1.input_dim).to(device)
    s2.train(False)
    L.append(f"  S2 params: {_fmt_p(_nparams(s2))}")
    bs = 32
    x = torch.randn(bs, cfg.s1.input_dim, device=device)
    y1 = torch.randn(bs, cfg.s2.output_dim, device=device)

    # 3a: Sweep convergence_eps [1e-2, 1e-3, 1e-4, 1e-5]
    L.append("")
    L.append("  [3a] Convergence speed vs eps:")
    for eps in [1e-2, 1e-3, 1e-4, 1e-5]:
        _seed(seed)
        st = System2Iterative(
            System2Config_Full(
                cfg.s2.hidden_dim, cfg.s2.output_dim,
                cfg.s2.max_steps, eps, cfg.s2.convergence_patience,
            ),
            input_dim=cfg.s1.input_dim,
        ).to(device)
        st.train(False)
        m, s, _ = _bench(
            lambda m=st: m(x, y1, return_trace=True),
            warmup, runs, device,
        )
        with torch.no_grad():
            res = st(x, y1)
        avg = res["steps_used"].float().mean().item()
        nconv = res["converged"].sum().item()
        L.append(
            f"    eps={eps:.0e}: {_fmt_ms(m)} +/- {_fmt_ms(s)}"
            f"  avg_steps={avg:.1f}  converged={nconv}/{bs}"
        )
        R.append(BenchmarkResult(
            f"s2_eps_{eps:.0e}", m, s,
            details={"eps": eps, "avg_steps": avg, "n_converged": nconv},
        ))

    # 3b: Measure per-step latency by forcing all steps (eps=1e-10)
    L.append("")
    L.append("  [3b] S2 latency per step:")
    for ms_ in [1, 3, 5, 10]:
        _seed(seed)
        st = System2Iterative(
            System2Config_Full(
                cfg.s2.hidden_dim, cfg.s2.output_dim,
                ms_, 1e-10, ms_ + 1,
            ),
            input_dim=cfg.s1.input_dim,
        ).to(device)
        st.train(False)
        m, s, _ = _bench(lambda m=st: m(x, y1), warmup, runs, device)
        L.append(
            f"    max_steps={ms_:>2d}: total={_fmt_ms(m)}"
            f"  per_step={_fmt_ms(m / ms_)}"
        )
        R.append(BenchmarkResult(
            f"s2_steps_{ms_}", m, s,
            details={"max_steps": ms_, "per_step_ms": m / ms_},
        ))

    # 3c: Compare converging (zero input) vs non-converging (noisy) inputs
    L.append("")
    L.append("  [3c] Converging vs non-converging:")
    _seed(seed)
    sc = System2Iterative(
        cfg.s2, input_dim=cfg.s1.input_dim,
    ).to(device)
    sc.train(False)
    xs = torch.zeros(bs, cfg.s1.input_dim, device=device)
    ys = torch.zeros(bs, cfg.s2.output_dim, device=device)
    xn = torch.randn(bs, cfg.s1.input_dim, device=device) * 10
    yn = torch.randn(bs, cfg.s2.output_dim, device=device) * 10
    ms_stable, _, _ = _bench(lambda: sc(xs, ys), warmup, runs, device)
    ms_noisy, _, _ = _bench(lambda: sc(xn, yn), warmup, runs, device)
    with torch.no_grad():
        rs_stable = sc(xs, ys)
        rs_noisy = sc(xn, yn)
    L.append(
        f"    Stable: {_fmt_ms(ms_stable)}"
        f" steps={rs_stable['steps_used'].float().mean().item():.1f}"
        f" conv={rs_stable['converged'].sum().item()}/{bs}"
    )
    L.append(
        f"    Noisy:  {_fmt_ms(ms_noisy)}"
        f" steps={rs_noisy['steps_used'].float().mean().item():.1f}"
        f" conv={rs_noisy['converged'].sum().item()}/{bs}"
    )
    R.append(BenchmarkResult(
        "s2_stable_vs_noisy", ms_stable,
        details={"stable_ms": ms_stable, "noisy_ms": ms_noisy},
    ))

    # 3d: Measure speedup from processing only a subset through S2
    L.append("")
    L.append("  [3d] Selective S2 execution:")
    tbs = 128
    xf = torch.randn(tbs, cfg.s1.input_dim, device=device)
    yf = torch.randn(tbs, cfg.s2.output_dim, device=device)
    for pct in [0.1, 0.5, 0.9]:
        ss = max(1, int(tbs * pct))
        xsb = xf[:ss]
        ysb = yf[:ss]
        m, s, _ = _bench(
            lambda xs=xsb, ys=ysb: s2(xs, ys),
            warmup, runs, device,
        )
        tp = ss / m * 1000 if m > 0 else 0
        L.append(
            f"    {pct * 100:>3.0f}% ({ss:>3d} items): {_fmt_ms(m)}"
            f"  ({_fmt_tp(tp)})"
        )
        R.append(BenchmarkResult(
            f"s2_selective_{int(pct * 100)}pct", m, s, tp,
            details={"fraction": pct, "subset": ss},
        ))

    return R, L


# ===========================================================================
# Suite 4: Routing Overhead
# ===========================================================================


def suite_routing_overhead(
    cfg: DualProcessFullConfig,
    device: torch.device,
    warmup: int,
    runs: int,
    seed: int,
) -> Tuple[List[BenchmarkResult], List[str]]:
    """Benchmark routing decision time, novelty scoring, and budget allocation."""
    R: List[BenchmarkResult] = []
    L: List[str] = []
    L.append("--- Suite 4: Routing Overhead ---")
    _seed(seed)

    # 4a: Full routing decision latency (should be <1ms for batch 128)
    L.append("")
    L.append("  [4a] Routing decision time:")
    for bs in [1, 32, 128]:
        rt = MetacognitiveRouter(
            cfg.meta, hidden_dim=cfg.s1.input_dim,
        ).to(device)
        rt.train(False)
        x = torch.randn(bs, cfg.s1.input_dim, device=device)
        c = torch.rand(bs, device=device)
        m, s, _ = _bench(
            lambda r=rt, x=x, c=c: r(x, c),
            warmup, runs, device,
        )
        L.append(
            f"    Batch={bs:>4d}: {_fmt_ms(m)} +/- {_fmt_ms(s)}"
            f"  per_item={_fmt_ms(m / bs)}"
        )
        R.append(BenchmarkResult(
            f"route_bs{bs}", m, s,
            details={"batch_size": bs, "per_item_ms": m / bs},
        ))

    # 4b: Compare novelty scoring methods: prototype, knn, mahalanobis
    L.append("")
    L.append("  [4b] Novelty scoring by method:")
    for meth in ["prototype", "knn", "mahalanobis"]:
        _seed(seed)
        rn = MetacognitiveRouter(
            MetacognitionConfig(
                num_prototypes=cfg.meta.num_prototypes,
                novelty_method=meth,
            ),
            hidden_dim=cfg.s1.input_dim,
        ).to(device)
        rn.train(False)
        xn = torch.randn(128, cfg.s1.input_dim, device=device)
        if meth == "prototype":
            fn: Callable[[], Any] = lambda r=rn, x=xn: r.compute_novelty_prototype(x)
        elif meth == "knn":
            fn = lambda r=rn, x=xn: r.compute_novelty_knn(x)
        else:
            fn = lambda r=rn, x=xn: r.compute_novelty_mahalanobis(x)
        m, s, _ = _bench(fn, warmup, runs, device)
        L.append(f"    {meth:>15s}: {_fmt_ms(m)} +/- {_fmt_ms(s)}")
        R.append(BenchmarkResult(
            f"novelty_{meth}", m, s, details={"method": meth},
        ))

    # 4c: Isolate budget allocation overhead (full route minus novelty)
    L.append("")
    L.append("  [4c] Budget allocation overhead:")
    rb = MetacognitiveRouter(
        cfg.meta, hidden_dim=cfg.s1.input_dim,
    ).to(device)
    rb.train(False)
    xb = torch.randn(128, cfg.s1.input_dim, device=device)
    cb = torch.rand(128, device=device)
    mf, _, _ = _bench(lambda: rb(xb, cb), warmup, runs, device)
    mn, _, _ = _bench(lambda: rb.compute_novelty(xb), warmup, runs, device)
    bo = max(0.0, mf - mn)
    L.append(
        f"    Full route: {_fmt_ms(mf)}  Novelty: {_fmt_ms(mn)}"
        f"  Budget+score: {_fmt_ms(bo)}"
    )
    R.append(BenchmarkResult(
        "budget_overhead", bo,
        details={"full_ms": mf, "novelty_ms": mn},
    ))

    return R, L


# ===========================================================================
# Suite 5: Full Pipeline
# ===========================================================================


def suite_full_pipeline(
    cfg: DualProcessFullConfig,
    device: torch.device,
    warmup: int,
    runs: int,
    seed: int,
) -> Tuple[List[BenchmarkResult], List[str]]:
    """Benchmark end-to-end pipeline for various batch sizes and routing mixes."""
    R: List[BenchmarkResult] = []
    L: List[str] = []
    L.append("--- Suite 5: Full Pipeline ---")
    _seed(seed)
    model = DualProcessReasonerBench(cfg).to(device)
    model.train(False)
    L.append(f"  Pipeline params: {_fmt_p(_nparams(model))}")

    # 5a: End-to-end latency across batch sizes [1, 8, 32, 128]
    L.append("")
    L.append("  [5a] End-to-end batch sweep:")
    for bs in [1, 8, 32, 128]:
        x = torch.randn(bs, cfg.s1.input_dim, device=device)
        m, s, _ = _bench(lambda x=x: model(x), warmup, runs, device)
        tp = bs / m * 1000 if m > 0 else 0
        L.append(
            f"    Batch={bs:>4d}: {_fmt_ms(m)} +/- {_fmt_ms(s)}"
            f"  ({_fmt_tp(tp)})"
        )
        R.append(BenchmarkResult(
            f"pipeline_bs{bs}", m, s, tp,
            details={"batch_size": bs},
        ))

    # 5b: Compare routing compositions: all-S1, mixed, all-S2
    L.append("")
    L.append("  [5b] Routing mix (batch=64):")
    bs = 64
    xm = torch.randn(bs, cfg.s1.input_dim, device=device)
    _seed(seed)
    ms1 = DualProcessReasonerBench(DualProcessFullConfig(
        cfg.s1, cfg.s2,
        MetacognitionConfig(
            route_threshold=999.0,
            num_prototypes=cfg.meta.num_prototypes,
        ),
        cfg.cal,
    )).to(device)
    ms1.train(False)
    _seed(seed)
    ms2 = DualProcessReasonerBench(DualProcessFullConfig(
        cfg.s1, cfg.s2,
        MetacognitionConfig(
            always_run_s2=True,
            num_prototypes=cfg.meta.num_prototypes,
        ),
        cfg.cal,
    )).to(device)
    ms2.train(False)
    t1, _, _ = _bench(lambda: ms1(xm), warmup, runs, device)
    tm, _, _ = _bench(lambda: model(xm), warmup, runs, device)
    t2, _, _ = _bench(lambda: ms2(xm), warmup, runs, device)
    sf = model(xm).used_system2.float().mean().item()
    L.append(
        f"    All-S1: {_fmt_ms(t1)}  Mixed: {_fmt_ms(tm)}"
        f" (S2={sf:.0%})  All-S2: {_fmt_ms(t2)}"
    )
    if t1 > 0:
        L.append(f"    S2/S1 ratio: {t2 / t1:.2f}x")
    R.append(BenchmarkResult(
        "pipeline_mix", tm,
        details={
            "s1_ms": t1, "mixed_ms": tm, "s2_ms": t2, "s2_frac": sf,
        },
    ))

    # 5c: Measure overhead of trace construction (return_details=True vs False)
    L.append("")
    L.append("  [5c] return_details overhead:")
    xd = torch.randn(32, cfg.s1.input_dim, device=device)
    md = DualProcessReasonerBench(DualProcessFullConfig(
        cfg.s1, cfg.s2,
        MetacognitionConfig(
            always_run_s2=True,
            num_prototypes=cfg.meta.num_prototypes,
        ),
        cfg.cal, True,
    )).to(device)
    md.train(False)
    mnd, _, _ = _bench(
        lambda: md(xd, return_details=False), warmup, runs, device,
    )
    mwd, _, _ = _bench(
        lambda: md(xd, return_details=True), warmup, runs, device,
    )
    L.append(
        f"    No details: {_fmt_ms(mnd)}  With details: {_fmt_ms(mwd)}"
        f"  Overhead: {_fmt_ms(mwd - mnd)}"
    )
    R.append(BenchmarkResult(
        "pipeline_details_overhead", mwd - mnd,
        details={"no_ms": mnd, "with_ms": mwd},
    ))

    # 5d: Time fraction breakdown: S1 vs routing vs S2 (all-S2 mode)
    L.append("")
    L.append("  [5d] Time breakdown (batch=64, all-S2):")
    xbd = torch.randn(64, cfg.s1.input_dim, device=device)
    cd = torch.rand(64, device=device)
    yd = torch.randn(64, cfg.s2.output_dim, device=device)
    ts1, _, _ = _bench(lambda: md.s1(xbd), warmup, runs, device)
    trt, _, _ = _bench(lambda: md.router(xbd, cd), warmup, runs, device)
    ts2, _, _ = _bench(lambda: md.s2(xbd, yd), warmup, runs, device)
    tot = ts1 + trt + ts2
    if tot > 0:
        L.append(
            f"    S1: {_fmt_ms(ts1)} ({ts1 / tot * 100:.0f}%)"
            f"  Route: {_fmt_ms(trt)} ({trt / tot * 100:.0f}%)"
            f"  S2: {_fmt_ms(ts2)} ({ts2 / tot * 100:.0f}%)"
        )
    else:
        L.append("    S1/Route/S2 breakdown: all zero (too fast to measure)")
    R.append(BenchmarkResult(
        "pipeline_breakdown",
        details={"s1_ms": ts1, "route_ms": trt, "s2_ms": ts2},
    ))

    return R, L


# ===========================================================================
# Suite 6: Selective Execution
# ===========================================================================


def suite_selective_execution(
    cfg: DualProcessFullConfig,
    device: torch.device,
    warmup: int,
    runs: int,
    seed: int,
) -> Tuple[List[BenchmarkResult], List[str]]:
    """Benchmark scatter/gather subset S2 versus full-batch S2 execution."""
    R: List[BenchmarkResult] = []
    L: List[str] = []
    L.append("--- Suite 6: Selective Execution Efficiency ---")
    _seed(seed)
    tbs = 128
    s2 = System2Iterative(
        cfg.s2, input_dim=cfg.s1.input_dim,
    ).to(device)
    s2.train(False)
    xf = torch.randn(tbs, cfg.s1.input_dim, device=device)
    yf = torch.randn(tbs, cfg.s2.output_dim, device=device)
    mf, sf, _ = _bench(lambda: s2(xf, yf), warmup, runs, device)
    L.append(f"  Full batch ({tbs}): {_fmt_ms(mf)} +/- {_fmt_ms(sf)}")
    R.append(BenchmarkResult(
        "selective_full_batch", mf, sf, details={"batch_size": tbs},
    ))

    L.append("  Fraction  Size  Selective   Speedup")
    L.append("  -------  ----  ---------  -------")
    for frac in [0.10, 0.25, 0.50, 0.75, 1.00]:
        ss = max(1, int(tbs * frac))
        mk = torch.zeros(tbs, dtype=torch.bool, device=device)
        mk[:ss] = True
        xs = xf[mk]
        ys = yf[mk]
        _mk = mk

        def _sel(
            xs: Tensor = xs,
            ys: Tensor = ys,
            mk: Tensor = _mk,
        ) -> Tensor:
            o = s2(xs, ys)
            y = torch.zeros(tbs, cfg.s2.output_dim, device=device)
            y[mk] = o["y2"]
            return y

        ms_, ss_, _ = _bench(_sel, warmup, runs, device)
        sp = mf / ms_ if ms_ > 0 else float("inf")
        L.append(
            f"  {frac:>5.0%}  {ss:>5d}  {_fmt_ms(ms_):>9s}  {sp:>6.2f}x"
        )
        R.append(BenchmarkResult(
            f"selective_{int(frac * 100)}pct", ms_, ss_,
            details={
                "fraction": frac, "subset": ss,
                "speedup": sp, "full_ms": mf,
            },
        ))

    return R, L


# ===========================================================================
# Suite 7: Scaling Analysis
# ===========================================================================


def suite_scaling_analysis(
    cfg: DualProcessFullConfig,
    device: torch.device,
    warmup: int,
    runs: int,
    seed: int,
) -> Tuple[List[BenchmarkResult], List[str]]:
    """Benchmark throughput impact of hidden_dim, max_steps, and num_prototypes."""
    R: List[BenchmarkResult] = []
    L: List[str] = []
    L.append("--- Suite 7: Scaling Analysis ---")
    bs = 32

    # 7a: How throughput scales with hidden dimension [128, 256, 512, 1024]
    L.append("")
    L.append("  [7a] hidden_dim scaling:")
    L.append("    hdim    params     S1       S2       Pipeline  throughput")
    for hd in [128, 256, 512, 1024]:
        _seed(seed)
        tc = DualProcessFullConfig(
            System1Config(hd, hd // 2, hd // 4, 2),
            System2Config_Full(hd // 2, hd // 4, 5),
            MetacognitionConfig(num_prototypes=cfg.meta.num_prototypes),
        )
        tm = DualProcessReasonerBench(tc).to(device)
        tm.train(False)
        x = torch.randn(bs, hd, device=device)
        y = torch.randn(bs, hd // 4, device=device)
        t1, _, _ = _bench(lambda m=tm.s1, x=x: m(x), warmup, runs, device)
        t2, _, _ = _bench(
            lambda m=tm.s2, x=x, y=y: m(x, y), warmup, runs, device,
        )
        tp_, _, _ = _bench(lambda m=tm, x=x: m(x), warmup, runs, device)
        th = bs / tp_ * 1000 if tp_ > 0 else 0
        L.append(
            f"    {hd:>5d}  {_fmt_p(_nparams(tm)):>8s}"
            f"  {_fmt_ms(t1):>7s}  {_fmt_ms(t2):>7s}"
            f"  {_fmt_ms(tp_):>8s}  {_fmt_tp(th)}"
        )
        R.append(BenchmarkResult(
            f"scale_hdim_{hd}", tp_, throughput=th,
            details={
                "hdim": hd, "params": _nparams(tm),
                "s1_ms": t1, "s2_ms": t2,
            },
        ))

    # 7b: How S2 latency scales with max_steps [3, 5, 10, 15]
    L.append("")
    L.append("  [7b] max_steps scaling (hidden=128):")
    L.append("    steps  S2_ms    per_step  Pipeline")
    hf = 128
    for ms_ in [3, 5, 10, 15]:
        _seed(seed)
        tc = DualProcessFullConfig(
            System1Config(hf, hf // 2, hf // 4, 1),
            System2Config_Full(hf // 2, hf // 4, ms_, 1e-10, ms_ + 1),
            MetacognitionConfig(
                always_run_s2=True,
                num_prototypes=cfg.meta.num_prototypes,
            ),
        )
        tm = DualProcessReasonerBench(tc).to(device)
        tm.train(False)
        x = torch.randn(bs, hf, device=device)
        y = torch.randn(bs, hf // 4, device=device)
        t2, _, _ = _bench(
            lambda m=tm.s2, x=x, y=y: m(x, y), warmup, runs, device,
        )
        tp_, _, _ = _bench(lambda m=tm, x=x: m(x), warmup, runs, device)
        L.append(
            f"    {ms_:>5d}  {_fmt_ms(t2):>7s}"
            f"  {_fmt_ms(t2 / ms_):>8s}  {_fmt_ms(tp_):>8s}"
        )
        R.append(BenchmarkResult(
            f"scale_steps_{ms_}", tp_,
            details={"steps": ms_, "s2_ms": t2, "per_step": t2 / ms_},
        ))

    # 7c: How novelty/routing scales with prototype bank size [16, 64, 256]
    L.append("")
    L.append("  [7c] num_prototypes scaling:")
    L.append("    protos  novelty   routing")
    for np_ in [16, 64, 256]:
        _seed(seed)
        rt = MetacognitiveRouter(
            MetacognitionConfig(num_prototypes=np_),
            hidden_dim=hf,
        ).to(device)
        rt.train(False)
        xp = torch.randn(128, hf, device=device)
        cp = torch.rand(128, device=device)
        tn, _, _ = _bench(
            lambda r=rt, x=xp: r.compute_novelty(x),
            warmup, runs, device,
        )
        tr, _, _ = _bench(
            lambda r=rt, x=xp, c=cp: r(x, c),
            warmup, runs, device,
        )
        L.append(f"    {np_:>5d}  {_fmt_ms(tn):>8s}  {_fmt_ms(tr):>8s}")
        R.append(BenchmarkResult(
            f"scale_proto_{np_}", tr,
            details={"protos": np_, "novelty_ms": tn},
        ))

    return R, L


# ===========================================================================
# Sanity Checks
# ===========================================================================


def run_sanity_checks(
    cfg: DualProcessFullConfig,
    device: torch.device,
    seed: int,
) -> Tuple[List[Tuple[str, bool, str]], List[str]]:
    """Run 10 sanity checks to validate correctness of all components.

    Checks:
        1. S1 output shape matches config
        2. Calibrated confidence in [0, 1]
        3. S2 convergence produces valid halt_reason
        4. Routing produces boolean mask
        5. Steps budget in [1, max_steps] range
        6. Trace is JSON serializable when enabled
        7. return_details=False is faster than True
        8. Selective execution is faster than full batch (when <50% routed to S2)
        9. All config presets instantiate
        10. Deterministic output for fixed seed

    Returns:
        Tuple of (checks list, formatted lines for display).
    """
    checks: List[Tuple[str, bool, str]] = []
    L: List[str] = []
    L.append("--- Sanity Checks ---")
    _seed(seed)
    model = DualProcessReasonerBench(cfg).to(device)
    model.train(False)

    # Check 1: S1 output shape matches (B, output_dim)
    try:
        x = torch.randn(4, cfg.s1.input_dim, device=device)
        y = model.s1(x)["y1"]
        ok = y.shape == (4, cfg.s1.output_dim)
        checks.append(("1. S1 output shape", ok, f"{tuple(y.shape)}"))
    except Exception as e:
        checks.append(("1. S1 output shape", False, str(e)))

    # Check 2: Calibrated confidence values in valid [0, 1] range
    try:
        c = model.s1(
            torch.randn(16, cfg.s1.input_dim, device=device)
        )["conf_calibrated"]
        ok = bool((c >= 0).all() and (c <= 1).all())
        checks.append((
            "2. Calibrated confidence range", ok,
            f"[{c.min():.4f}, {c.max():.4f}]",
        ))
    except Exception as e:
        checks.append(("2. Calibrated confidence range", False, str(e)))

    # Check 3: S2 halt_reason is one of the valid enum values
    try:
        r = model.s2(
            torch.randn(8, cfg.s1.input_dim, device=device),
            torch.randn(8, cfg.s2.output_dim, device=device),
        )
        valid_reasons = {
            "converged", "max_steps", "budget_exhausted", "nan_guard",
        }
        ok = all(h in valid_reasons for h in r["halt_reason"])
        checks.append((
            "3. S2 halt_reason valid", ok, f"{set(r['halt_reason'])}",
        ))
    except Exception as e:
        checks.append(("3. S2 halt_reason valid", False, str(e)))

    # Check 4: Routing produces a proper boolean mask tensor
    try:
        r = model.router(
            torch.randn(16, cfg.s1.input_dim, device=device),
            torch.rand(16, device=device),
        )
        u = r["used_system2"]
        ok = u.dtype == torch.bool and u.shape == (16,)
        checks.append((
            "4. Routing boolean mask", ok,
            f"dtype={u.dtype} shape={tuple(u.shape)}",
        ))
    except Exception as e:
        checks.append(("4. Routing boolean mask", False, str(e)))

    # Check 5: Steps budget is within [1, max_steps] range
    try:
        r = model.router(
            torch.randn(32, cfg.s1.input_dim, device=device),
            torch.rand(32, device=device),
        )
        b = r["steps_budget"]
        ok = bool((b >= 1).all() and (b <= 15).all())
        checks.append((
            "5. Steps budget range", ok, f"[{b.min()}, {b.max()}]",
        ))
    except Exception as e:
        checks.append(("5. Steps budget range", False, str(e)))

    # Check 6: Reasoning trace is valid JSON when return_details=True
    try:
        _seed(seed)
        mt = DualProcessReasonerBench(DualProcessFullConfig(
            cfg.s1, cfg.s2,
            MetacognitionConfig(
                always_run_s2=True,
                num_prototypes=cfg.meta.num_prototypes,
            ),
            cfg.cal, True,
        )).to(device)
        mt.train(False)
        out = mt(
            torch.randn(4, cfg.s1.input_dim, device=device),
            return_details=True,
        )
        assert out.trace is not None
        j = out.trace.to_json()
        p = json.loads(j)
        ok = isinstance(p, dict) and "batch_size" in p
        checks.append(("6. Trace JSON serializable", ok, f"len={len(j)}"))
    except Exception as e:
        checks.append(("6. Trace JSON serializable", False, str(e)))

    # Check 7: Disabling trace (return_details=False) is not slower
    try:
        _seed(seed)
        md = DualProcessReasonerBench(DualProcessFullConfig(
            cfg.s1, cfg.s2,
            MetacognitionConfig(
                always_run_s2=True,
                num_prototypes=cfg.meta.num_prototypes,
            ),
            cfg.cal, True,
        )).to(device)
        md.train(False)
        x = torch.randn(32, cfg.s1.input_dim, device=device)
        # Warmup both paths
        for _ in range(3):
            md(x, return_details=False)
            md(x, return_details=True)
        tn: List[float] = []
        ty: List[float] = []
        for _ in range(20):
            t = Timer(device, device.type == "cuda")
            with t:
                md(x, return_details=False)
            tn.append(t.elapsed_ms)
        for _ in range(20):
            t = Timer(device, device.type == "cuda")
            with t:
                md(x, return_details=True)
            ty.append(t.elapsed_ms)
        mn_val = sum(tn) / len(tn)
        my_val = sum(ty) / len(ty)
        # Allow 5% tolerance for noise
        ok = mn_val <= my_val * 1.05
        checks.append((
            "7. return_details=False <= True", ok,
            f"no={mn_val:.2f}ms yes={my_val:.2f}ms",
        ))
    except Exception as e:
        checks.append(("7. return_details=False <= True", False, str(e)))

    # Check 8: Processing 25% of batch is faster than full batch
    try:
        _seed(seed)
        sc = System2Iterative(
            cfg.s2, input_dim=cfg.s1.input_dim,
        ).to(device)
        sc.train(False)
        tbs = 128
        ss = tbs // 4
        xf = torch.randn(tbs, cfg.s1.input_dim, device=device)
        yf = torch.randn(tbs, cfg.s2.output_dim, device=device)
        xs = xf[:ss]
        ys = yf[:ss]
        # Warmup
        for _ in range(3):
            sc(xf, yf)
            sc(xs, ys)
        tf: List[float] = []
        ts: List[float] = []
        for _ in range(10):
            t = Timer(device, device.type == "cuda")
            with t:
                sc(xf, yf)
            tf.append(t.elapsed_ms)
        for _ in range(10):
            t = Timer(device, device.type == "cuda")
            with t:
                sc(xs, ys)
            ts.append(t.elapsed_ms)
        mf_val = sum(tf) / len(tf)
        ms_val = sum(ts) / len(ts)
        # Allow 10% tolerance
        ok = ms_val <= mf_val * 1.10
        checks.append((
            "8. Selective < full batch", ok,
            f"full={mf_val:.2f}ms sel(25%)={ms_val:.2f}ms",
        ))
    except Exception as e:
        checks.append(("8. Selective < full batch", False, str(e)))

    # Check 9: All five config presets instantiate and produce output
    try:
        ok = True
        fails: List[str] = []
        for pn in [
            "minimal", "dev", "production_1b",
            "production_3b", "production_7b",
        ]:
            try:
                pc = getattr(DualProcessFullConfig, pn)()
                m_test = DualProcessReasonerBench(pc)
                o = m_test(torch.randn(2, pc.s1.input_dim))
                assert o.y.shape[0] == 2
            except Exception as ex:
                ok = False
                fails.append(f"{pn}: {ex}")
        checks.append((
            "9. All config presets instantiate", ok,
            "OK" if ok else str(fails),
        ))
    except Exception as e:
        checks.append(("9. All config presets instantiate", False, str(e)))

    # Check 10: Deterministic output across 10 runs with same seed
    try:
        outs: List[Tensor] = []
        for _ in range(10):
            _seed(seed)
            dm = DualProcessReasonerBench(cfg).to(device)
            dm.train(False)
            x = torch.randn(8, cfg.s1.input_dim, device=device)
            outs.append(dm(x).y.cpu())
        ok = all(torch.equal(outs[0], outs[i]) for i in range(1, 10))
        checks.append((
            "10. Deterministic output", ok,
            "exact" if ok else "differs",
        ))
    except Exception as e:
        checks.append(("10. Deterministic output", False, str(e)))

    # Format output
    np_ = sum(1 for _, o, _ in checks if o)
    for n, o, d in checks:
        status = "PASS" if o else "FAIL"
        L.append(f"  [{status}] {n}  --  {d}")
    L.append(f"  {np_}/{len(checks)} sanity checks passed")

    return checks, L


# ===========================================================================
# Suite registry and config preset mapping
# ===========================================================================

SUITE_MAP: Dict[int, Tuple[str, Callable[..., Tuple[List[BenchmarkResult], List[str]]]]] = {
    1: ("System 1 Throughput", suite_system1_throughput),
    2: ("Calibration Overhead", suite_calibration_overhead),
    3: ("System 2 Convergence", suite_system2_convergence),
    4: ("Routing Overhead", suite_routing_overhead),
    5: ("Full Pipeline", suite_full_pipeline),
    6: ("Selective Execution", suite_selective_execution),
    7: ("Scaling Analysis", suite_scaling_analysis),
}

CONFIG_MAP: Dict[str, Callable[[], DualProcessFullConfig]] = {
    "minimal": DualProcessFullConfig.minimal,
    "dev": DualProcessFullConfig.dev,
    "production": DualProcessFullConfig.production_1b,
    "production_1b": DualProcessFullConfig.production_1b,
    "production_3b": DualProcessFullConfig.production_3b,
    "production_7b": DualProcessFullConfig.production_7b,
}


# ---------------------------------------------------------------------------
# Summary builder: aggregates results into JSON-serializable report
# ---------------------------------------------------------------------------


def build_summary(
    results: List[BenchmarkResult],
    checks: List[Tuple[str, bool, str]],
    dev_str: str,
    cfg_name: str,
    elapsed: float,
) -> Dict[str, Any]:
    """Build a JSON-serializable summary dictionary from benchmark results.

    Aggregates all BenchmarkResult entries and sanity check outcomes into a
    single dictionary suitable for JSON output. Extracts key convenience
    metrics (s1_throughput, routing_overhead, s2_convergence_steps) at the
    top level for quick reading.

    Args:
        results:    List of BenchmarkResult from all suites.
        checks:     List of (name, passed, detail) from sanity checks.
        dev_str:    Device string (e.g., "cpu" or "cuda:0").
        cfg_name:   Configuration preset name.
        elapsed:    Total wall-clock time in seconds.

    Returns:
        Dict ready for json.dumps().
    """
    s: Dict[str, Any] = {
        "device": dev_str,
        "config": cfg_name,
        "total_time_s": round(elapsed, 2),
        "sanity_checks_passed": sum(1 for _, o, _ in checks if o),
        "sanity_checks_total": len(checks),
    }

    for r in results:
        e: Dict[str, Any] = {"mean_ms": round(r.mean_time, 4)}
        if r.std_time > 0:
            e["std_ms"] = round(r.std_time, 4)
        if r.throughput > 0:
            e["throughput"] = round(r.throughput, 1)
        if r.details:
            e["details"] = r.details
        s[r.name] = e

    # Convenience top-level metrics
    b32 = next((r for r in results if r.name == "s1_batch_32"), None)
    if b32:
        s["s1_throughput_items_sec"] = round(b32.throughput, 1)

    r128 = next((r for r in results if r.name == "route_bs128"), None)
    if r128:
        s["routing_overhead_ms"] = round(r128.mean_time, 4)

    se = next((r for r in results if r.name == "s2_eps_1e-03"), None)
    if se and "avg_steps" in se.details:
        s["s2_avg_convergence_steps"] = se.details["avg_steps"]

    pm = next((r for r in results if r.name == "pipeline_mix"), None)
    if pm:
        s["pipeline_all_s1_ms"] = round(pm.details.get("s1_ms", 0), 2)
        s["pipeline_all_s2_ms"] = round(pm.details.get("s2_ms", 0), 2)

    return s


# ---------------------------------------------------------------------------
# CLI entry point: parses args, runs suites, prints results
# ---------------------------------------------------------------------------


def main() -> int:
    """Main entry point for the benchmark CLI.

    Parses command-line arguments, instantiates the configuration, runs
    selected benchmark suites, executes sanity checks, and outputs results
    in both human-readable and JSON formats.

    Returns:
        0 if all sanity checks pass, 1 if any check fails.
    """
    p = argparse.ArgumentParser(
        description="Dual-Process Reasoning Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python routing_benchmark.py\n"
            "  python routing_benchmark.py --device cuda --config dev\n"
            "  python routing_benchmark.py --suite 1 3 5 --runs 20\n"
            "  python routing_benchmark.py --output results.json\n"
        ),
    )
    p.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device to benchmark on (default: cpu)",
    )
    p.add_argument(
        "--config", default="minimal", choices=list(CONFIG_MAP.keys()),
        help="Configuration preset (default: minimal)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup iterations before timing (default: 3)",
    )
    p.add_argument(
        "--runs", type=int, default=10,
        help="Number of timed runs per benchmark (default: 10)",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Path to write JSON results file (default: stdout only)",
    )
    p.add_argument(
        "--suite", type=int, nargs="*", default=None,
        help="Run specific suites by number (1-7). Default: all suites",
    )
    p.add_argument(
        "--no-sanity", action="store_true",
        help="Skip sanity checks after benchmarks",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output with full tracebacks on errors",
    )
    args = p.parse_args()

    # Resolve device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")

    cfg = CONFIG_MAP[args.config]()
    suites = [
        s for s in (args.suite or list(SUITE_MAP.keys()))
        if s in SUITE_MAP
    ]
    if not suites:
        print("No valid suites")
        return 1

    print(f"\n{'=' * 60}")
    print("=== Dual-Process Reasoning Benchmark ===")
    print(f"Device: {device} | Config: {args.config} | Seed: {args.seed}")
    print(
        f"Warmup: {args.warmup} | Runs: {args.runs}"
        f" | Source: {_reasoning_source}"
    )
    print(f"{'=' * 60}\n")

    _seed(args.seed)
    all_r: List[BenchmarkResult] = []
    t0 = time.perf_counter()

    for sid in suites:
        sn, sf = SUITE_MAP[sid]
        print(f"Running suite {sid}: {sn} ...")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            ts = time.perf_counter()
            r, lines = sf(cfg, device, args.warmup, args.runs, args.seed)
            all_r.extend(r)
            for ln in lines:
                print(ln)
            print(f"  (suite {sid}: {time.perf_counter() - ts:.1f}s)\n")
        except Exception as e:
            print(f"  ERROR: {e}")
            if args.verbose:
                traceback.print_exc()
            print()

    checks: List[Tuple[str, bool, str]] = []
    if not args.no_sanity:
        print("Running sanity checks ...")
        gc.collect()
        try:
            checks, cl = run_sanity_checks(cfg, device, args.seed)
            for ln in cl:
                print(ln)
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            checks = [("error", False, str(e))]
            print()

    elapsed = time.perf_counter() - t0
    summary = build_summary(all_r, checks, str(device), args.config, elapsed)

    print(f"{'=' * 60}")
    print("=== Benchmark Summary (JSON) ===")
    print(json.dumps(summary, indent=2, default=str))
    print(f"{'=' * 60}")
    print(f"Total: {elapsed:.1f}s")

    if args.output:
        op = os.path.abspath(args.output)
        d = os.path.dirname(op)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(op, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Written to: {op}")

    if checks:
        np_ = sum(1 for _, o, _ in checks if o)
        if np_ < len(checks):
            print(f"\nWARNING: {len(checks) - np_} check(s) FAILED.")
            return 1

    return 0


# ---------------------------------------------------------------------------
# Version and metadata
# ---------------------------------------------------------------------------
__version__ = "0.2.0"
__benchmark_suites__ = list(SUITE_MAP.keys())
__config_presets__ = list(CONFIG_MAP.keys())


if __name__ == "__main__":
    sys.exit(main())
