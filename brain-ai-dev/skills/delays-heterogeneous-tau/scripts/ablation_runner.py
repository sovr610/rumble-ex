#!/usr/bin/env python3
"""
ablation_runner.py - 4-config ablation matrix for learnable delays and heterogeneous tau.

Orchestrates training runs across four configurations:
  baseline     : no delays, homogeneous fixed tau
  delays_only  : learnable DCLS delays, homogeneous fixed tau
  hetero_only  : no delays, heterogeneous learnable tau
  both         : learnable DCLS delays, heterogeneous learnable tau

Each configuration is trained with multiple random seeds on a chosen benchmark
dataset. After all runs, a markdown comparison report is generated summarising
accuracy, timing, delay statistics, and tau statistics.

Usage:
    python ablation_runner.py [options]
    python ablation_runner.py --dry-run
    python ablation_runner.py --resume results/ablation_20260218_120000
    python ablation_runner.py --configs baseline,both --seeds 42 --epochs 10
"""

# ---------------------------------------------------------------------------
# SECTION 1: Imports and repo root detection
# ---------------------------------------------------------------------------

import os
import sys
import json
import time
import math
import random
import signal
import logging
import argparse
import warnings
import traceback
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Repo root: scripts/ -> delays-heterogeneous-tau/ -> skills/ -> brain-ai-dev/ -> human-brain/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(_SCRIPT_DIR)
        )
    )
)
if os.path.isdir(os.path.join(_REPO_ROOT, "brain_ai")):
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Optional imports — gracefully handled
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

try:
    import torchvision
    import torchvision.transforms as transforms
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False

try:
    import tonic  # noqa: F401
    _HAS_TONIC = True
except ImportError:
    _HAS_TONIC = False

# brain_ai optional import
try:
    from brain_ai.core.neurons import ATanSurrogate
    _HAS_BRAIN_AI = True
except ImportError:
    _HAS_BRAIN_AI = False


# ---------------------------------------------------------------------------
# SECTION 2: Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DelayConfig:
    """
    Configuration for learnable synaptic delays.

    Attributes:
        enabled   : Whether delays are active.
        mode      : 'off' | 'learnable_dcls'
        max_delay : Maximum discrete delay in timesteps.
        sigma_init: Initial smoothing sigma for DCLS kernel.
        sigma_min : Minimum sigma (annealed toward this during training).
        learn_rate_multiplier: LR scaling factor for delay parameters.
    """
    enabled: bool = False
    mode: str = "off"              # 'off' | 'learnable_dcls'
    max_delay: int = 16
    sigma_init: float = 4.0
    sigma_min: float = 0.5
    learn_rate_multiplier: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TauConfig:
    """
    Configuration for membrane time constants.

    Attributes:
        enabled      : Whether heterogeneous tau is active.
        mode         : 'homogeneous_fixed' | 'heterogeneous_learnable' |
                       'heterogeneous_fixed'
        tau_0        : Target mean tau (ms).
        tau_min      : Hard lower bound.
        tau_max      : Hard upper bound.
        dt           : Simulation timestep.
        init_strategy: 'homogeneous' | 'heterogeneous_gamma' |
                        'heterogeneous_loguniform' | 'preset_bank'
        gamma_k      : Gamma shape parameter.
        gamma_theta  : Gamma scale parameter.
    """
    enabled: bool = False
    mode: str = "homogeneous_fixed"
    tau_0: float = 20.0
    tau_min: float = 1.0
    tau_max: float = 100.0
    dt: float = 1.0
    init_strategy: str = "homogeneous"
    gamma_k: float = 2.0
    gamma_theta: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# SECTION 3: The 4 ablation configurations
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "delay": DelayConfig(enabled=False, mode="off"),
        "tau": TauConfig(
            enabled=False,
            mode="homogeneous_fixed",
            init_strategy="homogeneous",
        ),
        "description": "No delays, homogeneous fixed tau — pure baseline.",
    },
    "delays_only": {
        "delay": DelayConfig(enabled=True, mode="learnable_dcls"),
        "tau": TauConfig(
            enabled=False,
            mode="homogeneous_fixed",
            init_strategy="homogeneous",
        ),
        "description": "Learnable DCLS delays, homogeneous fixed tau.",
    },
    "hetero_only": {
        "delay": DelayConfig(enabled=False, mode="off"),
        "tau": TauConfig(
            enabled=True,
            mode="heterogeneous_learnable",
            init_strategy="heterogeneous_gamma",
        ),
        "description": "No delays, heterogeneous learnable tau (Gamma init).",
    },
    "both": {
        "delay": DelayConfig(enabled=True, mode="learnable_dcls"),
        "tau": TauConfig(
            enabled=True,
            mode="heterogeneous_learnable",
            init_strategy="heterogeneous_gamma",
        ),
        "description": "Learnable DCLS delays + heterogeneous learnable tau.",
    },
}


# ---------------------------------------------------------------------------
# SECTION 4: Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    """All metrics collected for one training epoch."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    # Delay metrics (populated when delay_config.enabled)
    delay_mean: Optional[float] = None
    delay_std: Optional[float] = None
    delay_entropy: Optional[float] = None
    delay_boundary_pct: Optional[float] = None
    sigma: Optional[float] = None
    # Tau metrics (populated when tau_config.enabled)
    tau_mean: Optional[float] = None
    tau_std: Optional[float] = None
    tau_min_val: Optional[float] = None
    tau_max_val: Optional[float] = None
    tau_drift: Optional[float] = None
    # Performance
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    epoch_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}


@dataclass
class RunResult:
    """Final result for one config + seed combination."""
    config_name: str
    seed: int
    best_val_acc: float
    best_epoch: int
    total_epochs: int
    total_time_s: float
    param_count: int
    metrics: List[Dict[str, Any]]
    completed: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# SECTION 5: Heterogeneous tau module (standalone, physics-grounded)
# ---------------------------------------------------------------------------

def _tau_to_tau_raw(tau_0_tensor: torch.Tensor, tau_min: float = 1.0) -> torch.Tensor:
    """
    Invert tau = tau_min + softplus(tau_raw) to get tau_raw.

    tau_0 - tau_min = softplus(tau_raw) = log(1 + exp(tau_raw))
    exp(tau_raw)    = exp(tau_0 - tau_min) - 1
    tau_raw         = log(exp(tau_0 - tau_min) - 1)
    """
    val = tau_0_tensor - tau_min
    return val.expm1().clamp(min=1e-8).log()


def _init_tau_values(n: int, cfg: TauConfig) -> torch.Tensor:
    """Dispatch tau initialization by strategy, returns (n,) tensor in [tau_min, tau_max]."""
    strategy = cfg.init_strategy
    tau_min = cfg.tau_min
    tau_max = cfg.tau_max
    tau_0 = cfg.tau_0

    if strategy == "homogeneous":
        tau = torch.full((n,), float(tau_0)).clamp(tau_min, tau_max)

    elif strategy == "heterogeneous_gamma":
        # Gamma(k, theta) with k=gamma_k, mean ~ tau_0
        k = cfg.gamma_k
        theta = tau_0 / max(k, 1e-6)
        dist = torch.distributions.Gamma(
            concentration=torch.tensor(k, dtype=torch.float32),
            rate=torch.tensor(1.0 / max(theta, 1e-8), dtype=torch.float32),
        )
        tau = dist.sample((n,)).clamp(tau_min, tau_max)

    elif strategy == "heterogeneous_loguniform":
        log_min = math.log(tau_min)
        log_max = math.log(tau_max)
        log_tau = torch.empty(n).uniform_(log_min, log_max)
        tau = log_tau.exp().clamp(tau_min, tau_max)

    elif strategy == "preset_bank":
        presets = [2.0, 5.0, 10.0, 20.0, 50.0]
        preset_tensor = torch.tensor(presets).clamp(tau_min, tau_max)
        indices = torch.arange(n) % len(presets)
        tau = preset_tensor[indices]

    else:
        raise ValueError(f"Unknown tau init strategy: {strategy!r}")

    return tau.float()


class HeterogeneousTau(nn.Module):
    """
    Per-neuron membrane time constants with physics-grounded parametrization.

    Parametrization:
        tau_raw  : unconstrained parameter or buffer
        tau      : tau_min + softplus(tau_raw),  clamped to tau_max
        beta     : exp(-dt / tau),               clamped to [beta_min, beta_max]

    All tau -> beta computation is forced to fp32 per stability spec.
    """

    BETA_MIN = 0.0
    BETA_MAX = 0.999

    def __init__(
        self,
        n: int,
        learnable: bool = True,
        tau_cfg: Optional[TauConfig] = None,
        dt: float = 1.0,
        tau_min: float = 1.0,
        tau_max: float = 100.0,
    ):
        super().__init__()
        cfg = tau_cfg or TauConfig()
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max

        tau_init = _init_tau_values(n, cfg)          # (n,) in [tau_min, tau_max]
        tau_raw_init = _tau_to_tau_raw(tau_init, tau_min)

        if learnable:
            self.tau_raw = nn.Parameter(tau_raw_init)
        else:
            self.register_buffer("tau_raw", tau_raw_init)

        # Store init tau for drift computation
        self.register_buffer("tau_init", tau_init.clone())

    @property
    def tau(self) -> torch.Tensor:
        """Constrained tau in [tau_min, tau_max]."""
        return (self.tau_min + F.softplus(self.tau_raw)).clamp(max=self.tau_max)

    @property
    def beta(self) -> torch.Tensor:
        """Constrained beta in [BETA_MIN, BETA_MAX], computed in fp32."""
        tau_fp32 = self.tau.float()
        beta_fp32 = torch.exp(-self.dt / tau_fp32)
        return beta_fp32.clamp(self.BETA_MIN, self.BETA_MAX)

    def get_stats(self) -> Dict[str, float]:
        """Return diagnostic tau statistics."""
        with torch.no_grad():
            t = self.tau.float()
            tau_drift = ((t - self.tau_init.float()).abs().mean()).item()
            return {
                "tau_mean": t.mean().item(),
                "tau_std": t.std().item() if t.numel() > 1 else 0.0,
                "tau_min_val": t.min().item(),
                "tau_max_val": t.max().item(),
                "tau_drift": tau_drift,
            }

    def check_clamp_pressure(self, layer_name: str = "") -> None:
        """Warn if too many neurons are pressed against the bounds."""
        with torch.no_grad():
            t = self.tau.float()
            frac_min = (t <= self.tau_min * 1.01).float().mean().item()
            frac_max = (t >= self.tau_max * 0.99).float().mean().item()
            if frac_min > 0.05:
                logging.warning(
                    "[%s] %.1f%% neurons at tau_min — risk of dead neurons",
                    layer_name, frac_min * 100,
                )
            if frac_max > 0.05:
                logging.warning(
                    "[%s] %.1f%% neurons at tau_max — consider increasing tau_max",
                    layer_name, frac_max * 100,
                )


# ---------------------------------------------------------------------------
# SECTION 6: DCLS delay module (standalone approximation)
# ---------------------------------------------------------------------------

class DCLSDelay(nn.Module):
    """
    Differentiable Convolutional for Learnable Synaptic delays (DCLS).

    Implements a soft-selection over a ring buffer of past activations using
    a Gaussian-smoothed position kernel. This is a simplified 1-D DCLS
    approximation suitable for dense layers.

    The continuous delay position 'd' is a learnable scalar per output unit.
    The Gaussian kernel of width sigma blurs the discrete position across the
    buffer, making it differentiable w.r.t. d.

    Shape contract:
        input  : (batch, in_features)
        output : (batch, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_delay: int = 16,
        sigma: float = 4.0,
        sigma_min: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_delay = max_delay
        self.sigma = sigma          # mutable — updated by caller via set_sigma()
        self.sigma_min = sigma_min

        # Linear projection weight matrix (no bias)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Learnable delay position per output unit: shape (out_features,)
        # Initialized uniformly in [0, max_delay - 1]
        d_init = torch.rand(out_features) * (max_delay - 1)
        self.delay_pos = nn.Parameter(d_init)

        # Ring buffer for past inputs — initialized lazily
        self._buffer: Optional[torch.Tensor] = None   # (max_delay, batch, in_features)
        self._buf_idx: int = 0

    def set_sigma(self, sigma: float) -> None:
        self.sigma = max(sigma, self.sigma_min)

    def reset_buffer(self) -> None:
        """Clear the delay ring buffer (call between sequences)."""
        self._buffer = None
        self._buf_idx = 0

    def _update_buffer(self, x: torch.Tensor) -> torch.Tensor:
        """Push x into ring buffer and return the weighted-delay output."""
        batch = x.shape[0]
        device = x.device

        if self._buffer is None:
            self._buffer = torch.zeros(
                self.max_delay, batch, self.in_features, device=device
            )
            self._buf_idx = 0

        # Write current input to buffer slot
        self._buffer[self._buf_idx] = x.detach()

        # Build Gaussian kernel over delay positions
        # positions: integer slots [0, max_delay)
        positions = torch.arange(
            self.max_delay, dtype=torch.float32, device=device
        )  # (max_delay,)

        # delay_pos clamped to valid range: (out_features,)
        d = self.delay_pos.clamp(0.0, self.max_delay - 1.0)

        # kernel: (out_features, max_delay)  — un-normalised Gaussian
        diff = positions.unsqueeze(0) - d.unsqueeze(1)   # (out, max_delay)
        kernel = torch.exp(-0.5 * (diff / max(self.sigma, 1e-4)) ** 2)
        kernel = kernel / (kernel.sum(dim=1, keepdim=True) + 1e-8)

        # Stack buffer rotated so current slot is index 0
        indices = [(self._buf_idx - k) % self.max_delay for k in range(self.max_delay)]
        stacked = torch.stack([self._buffer[i] for i in indices], dim=1)
        # stacked: (batch, max_delay, in_features)

        # Project first, then apply kernel weights
        projected = stacked @ self.weight.t()   # (batch, max_delay, out_features)
        kernel_T = kernel.t().unsqueeze(0)       # (1, max_delay, out_features)
        output = (projected * kernel_T).sum(dim=1)   # (batch, out_features)

        self._buf_idx = (self._buf_idx + 1) % self.max_delay
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._update_buffer(x)

    def get_delay_stats(self) -> Dict[str, float]:
        """Diagnostic statistics for the learned delay positions."""
        with torch.no_grad():
            d = self.delay_pos.clamp(0.0, self.max_delay - 1.0)
            positions = torch.arange(
                self.max_delay, dtype=torch.float32, device=d.device
            )
            diff = positions.unsqueeze(0) - d.unsqueeze(1)
            kernel = torch.exp(-0.5 * (diff / max(self.sigma, 1e-4)) ** 2)
            kernel = kernel / (kernel.sum(dim=1, keepdim=True) + 1e-8)
            entropy = -(kernel * (kernel + 1e-12).log()).sum(dim=1).mean()
            boundary_pct = (
                (d <= 1.0) | (d >= self.max_delay - 2.0)
            ).float().mean().item()
            return {
                "delay_mean": d.mean().item(),
                "delay_std": d.std().item() if d.numel() > 1 else 0.0,
                "delay_entropy": entropy.item(),
                "delay_boundary_pct": boundary_pct,
            }


# ---------------------------------------------------------------------------
# SECTION 7: LIF layer with optional delays and/or hetero tau
# ---------------------------------------------------------------------------

class _ATanFallback(torch.autograd.Function):
    """ATan surrogate gradient — fallback when brain_ai is not available."""
    alpha: float = 2.0

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x, = ctx.saved_tensors
        alpha = _ATanFallback.alpha
        grad = alpha / (2.0 * (1.0 + (math.pi * alpha * x) ** 2))
        return grad_out * grad


def _get_spike_fn() -> Any:
    """Return ATan surrogate apply function from brain_ai or local fallback."""
    if _HAS_BRAIN_AI:
        return ATanSurrogate.apply
    return _ATanFallback.apply


class LIFLayerWithFeatures(nn.Module):
    """
    A dense LIF layer that optionally adds:
      - Learnable DCLS delays on the input projection
      - Heterogeneous per-neuron tau (learnable or fixed)

    When delay_config.enabled is False, the layer uses a standard nn.Linear.
    When tau_config.enabled is False, a shared scalar beta is used.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        delay_cfg: DelayConfig,
        tau_cfg: TauConfig,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
    ):
        super().__init__()
        self.out_features = out_features
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism
        self.delay_cfg = delay_cfg
        self.tau_cfg = tau_cfg

        # Input projection (delay or plain linear)
        if delay_cfg.enabled and delay_cfg.mode == "learnable_dcls":
            self.projection: nn.Module = DCLSDelay(
                in_features=in_features,
                out_features=out_features,
                max_delay=delay_cfg.max_delay,
                sigma=delay_cfg.sigma_init,
                sigma_min=delay_cfg.sigma_min,
            )
        else:
            self.projection = nn.Linear(in_features, out_features, bias=True)

        # Tau / beta
        if tau_cfg.enabled:
            learnable = tau_cfg.mode == "heterogeneous_learnable"
            self.tau_module: Optional[HeterogeneousTau] = HeterogeneousTau(
                n=out_features,
                learnable=learnable,
                tau_cfg=tau_cfg,
                dt=tau_cfg.dt,
                tau_min=tau_cfg.tau_min,
                tau_max=tau_cfg.tau_max,
            )
        else:
            self.tau_module = None
            # Scalar fixed beta derived from tau_0 and dt
            beta_val = math.exp(-tau_cfg.dt / max(tau_cfg.tau_0, 1e-6))
            beta_val = max(0.0, min(0.999, beta_val))
            self.register_buffer("fixed_beta", torch.tensor(beta_val))

        self.spike_fn = _get_spike_fn()

        # State
        self.mem: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        self.mem = None
        if self.delay_cfg.enabled and isinstance(self.projection, DCLSDelay):
            self.projection.reset_buffer()

    def set_sigma(self, sigma: float) -> None:
        if isinstance(self.projection, DCLSDelay):
            self.projection.set_sigma(sigma)

    def _get_beta(self) -> torch.Tensor:
        if self.tau_module is not None:
            return self.tau_module.beta  # (out_features,) fp32
        return self.fixed_beta           # scalar

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, in_features)

        Returns:
            spk: (batch, out_features)  binary spikes
            mem: (batch, out_features)  membrane potential
        """
        i = self.projection(x)    # (batch, out_features)

        if self.mem is None:
            self.mem = torch.zeros_like(i)

        beta = self._get_beta()
        if beta.dim() == 1:
            beta = beta.unsqueeze(0)   # (1, out_features)

        # fp32 membrane update (AMP safety)
        mem_fp32 = self.mem.float()
        i_fp32 = i.float()
        beta_fp32 = beta.float()
        new_mem = beta_fp32 * mem_fp32 + i_fp32

        mem_shifted = new_mem - self.threshold
        spk = self.spike_fn(mem_shifted)

        if self.reset_mechanism == "subtract":
            new_mem = new_mem - spk * self.threshold
        else:
            new_mem = new_mem * (1.0 - spk)

        self.mem = new_mem.to(x.dtype)
        return spk.to(x.dtype), self.mem


# ---------------------------------------------------------------------------
# SECTION 8: Ablation model factory
# ---------------------------------------------------------------------------

class AblationSNN(nn.Module):
    """
    Minimal SNN model for ablation experiments.

    Architecture:
      encoder  : Linear(input_dim -> hidden)
      lif_1    : LIFLayerWithFeatures(hidden -> hidden)  [delay + tau]
      lif_2    : LIFLayerWithFeatures(hidden -> hidden)  [delay + tau]
      readout  : Linear(hidden -> num_classes)

    The temporal dimension is handled externally by the forward pass:
    the model accepts a (batch, T, input_dim) sequence and accumulates
    spike counts for classification via temporal mean pooling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        delay_cfg: DelayConfig,
        tau_cfg: TauConfig,
        num_timesteps: int = 98,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.delay_cfg = delay_cfg
        self.tau_cfg = tau_cfg

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        self.lif1 = LIFLayerWithFeatures(
            in_features=hidden_dim,
            out_features=hidden_dim,
            delay_cfg=delay_cfg,
            tau_cfg=tau_cfg,
        )
        self.lif2 = LIFLayerWithFeatures(
            in_features=hidden_dim,
            out_features=hidden_dim,
            delay_cfg=delay_cfg,
            tau_cfg=tau_cfg,
        )
        self.readout = nn.Linear(hidden_dim, num_classes, bias=True)

    def reset_state(self) -> None:
        self.lif1.reset_state()
        self.lif2.reset_state()

    def set_sigma(self, sigma: float) -> None:
        self.lif1.set_sigma(sigma)
        self.lif2.set_sigma(sigma)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (batch, T, input_dim) — sequence of input frames

        Returns:
            logits: (batch, num_classes)
        """
        batch, T, _ = x_seq.shape
        self.reset_state()

        spike_accum = torch.zeros(batch, self.hidden_dim, device=x_seq.device)

        for t in range(T):
            x_t = x_seq[:, t, :]                     # (batch, input_dim)
            h = F.relu(self.encoder(x_t))             # (batch, hidden)
            spk1, _ = self.lif1(h)                    # (batch, hidden)
            spk2, _ = self.lif2(spk1)                 # (batch, hidden)
            spike_accum = spike_accum + spk2

        # Temporal mean of spike count -> logits
        spike_mean = spike_accum / max(T, 1)
        logits = self.readout(spike_mean)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_delay_stats(self) -> Optional[Dict[str, float]]:
        if not self.delay_cfg.enabled:
            return None
        stats: Dict[str, float] = {}
        count = 0
        for _name, m in self.named_modules():
            if isinstance(m, DCLSDelay):
                s = m.get_delay_stats()
                for k, v in s.items():
                    stats[k] = stats.get(k, 0.0) + v
                count += 1
        if count > 0:
            for k in stats:
                stats[k] /= count
        return stats

    def get_tau_stats(self) -> Optional[Dict[str, float]]:
        if not self.tau_cfg.enabled:
            return None
        stats_list = []
        for _name, m in self.named_modules():
            if isinstance(m, HeterogeneousTau):
                stats_list.append(m.get_stats())
        if not stats_list:
            return None
        averaged: Dict[str, float] = {}
        for key in stats_list[0]:
            averaged[key] = sum(s[key] for s in stats_list) / len(stats_list)
        return averaged


def create_ablation_model(
    benchmark: str,
    delay_cfg: DelayConfig,
    tau_cfg: TauConfig,
    device: torch.device,
    num_timesteps: int = 98,
    hidden_dim: int = 256,
) -> AblationSNN:
    """
    Create a small SNN model appropriate for the given benchmark.

    All 4 ablation configs use identical architecture and hidden_dim;
    only the delay and tau sub-modules differ.
    """
    binfo = _get_benchmark_info(benchmark, num_timesteps)
    input_dim = binfo["chunk_size"]
    num_classes = binfo["num_classes"]

    model = AblationSNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        delay_cfg=delay_cfg,
        tau_cfg=tau_cfg,
        num_timesteps=binfo["T"],
    ).to(device)

    return model


# ---------------------------------------------------------------------------
# SECTION 9: Benchmark dataset loaders
# ---------------------------------------------------------------------------

def _get_benchmark_info(benchmark: str, num_timesteps: int = 98) -> Dict[str, Any]:
    """Return metadata dict for a benchmark without loading data."""
    if benchmark in ("smnist", "psmnist"):
        return {
            "T": 98,
            "chunk_size": 8,
            "num_classes": 10,
            "description": "Sequential MNIST (chunked, T=98, chunk=8)",
        }
    elif benchmark == "mnist":
        return {
            "T": 28,
            "chunk_size": 28,
            "num_classes": 10,
            "description": "Standard MNIST (row-by-row, T=28)",
        }
    elif benchmark == "shd":
        return {
            "T": 100,
            "chunk_size": 700,
            "num_classes": 20,
            "description": "Spiking Heidelberg Digits (requires tonic)",
        }
    else:
        raise ValueError(f"Unknown benchmark: {benchmark!r}")


def _smnist_collate(permutation: Optional[torch.Tensor]):
    """Returns a collate_fn for (Sequential|Permuted) MNIST."""
    def _collate(batch: List[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        imgs, labels = zip(*batch)
        imgs_t = torch.stack(imgs)            # (B, 1, 28, 28)
        labels_t = torch.tensor(labels)
        imgs_t = imgs_t.view(imgs_t.shape[0], -1)   # (B, 784)
        if permutation is not None:
            imgs_t = imgs_t[:, permutation]
        imgs_t = imgs_t.view(imgs_t.shape[0], 98, 8)   # (B, 98, 8)
        return imgs_t, labels_t
    return _collate


def _mnist_rowwise_collate(batch: List[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs, labels = zip(*batch)
    imgs_t = torch.stack(imgs).squeeze(1)    # (B, 28, 28)
    labels_t = torch.tensor(labels)
    return imgs_t, labels_t


def build_dataloaders(
    benchmark: str,
    batch_size: int,
    num_workers: int,
    data_dir: str = os.path.join(os.path.expanduser("~"), ".cache", "brain_ai_data"),
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for the given benchmark."""
    os.makedirs(data_dir, exist_ok=True)

    if benchmark in ("smnist", "psmnist"):
        if not _HAS_TORCHVISION:
            raise RuntimeError("torchvision is required for smnist/psmnist benchmark.")
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=tfm
        )
        val_ds = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=tfm
        )
        perm = None
        if benchmark == "psmnist":
            # Fixed permutation for reproducibility (seed independent)
            gen = torch.Generator()
            gen.manual_seed(0)
            perm = torch.randperm(784, generator=gen)
        collate_fn = _smnist_collate(perm)

    elif benchmark == "mnist":
        if not _HAS_TORCHVISION:
            raise RuntimeError("torchvision is required for mnist benchmark.")
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=tfm
        )
        val_ds = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=tfm
        )
        collate_fn = _mnist_rowwise_collate

    elif benchmark == "shd":
        if not _HAS_TONIC:
            raise RuntimeError(
                "benchmark='shd' requires the 'tonic' package. "
                "Install with: pip install tonic\n"
                "Fall back with: --benchmark smnist"
            )
        raise NotImplementedError(
            "SHD loading via tonic is not yet implemented in this script."
        )

    else:
        raise ValueError(f"Unknown benchmark: {benchmark!r}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# SECTION 10: Sigma schedule for DCLS delays
# ---------------------------------------------------------------------------

def compute_sigma(
    epoch: int,
    total_epochs: int,
    sigma_init: float,
    sigma_min: float,
) -> float:
    """
    Cosine-anneal sigma from sigma_init to sigma_min over training.

    High sigma early (soft position preference) anneals to low sigma
    late (sharp position commitment, near-discrete delay).
    """
    frac = epoch / max(total_epochs - 1, 1)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * frac))
    sigma = sigma_min + (sigma_init - sigma_min) * cosine_factor
    return max(sigma, sigma_min)


# ---------------------------------------------------------------------------
# SECTION 11: Training and validation loops
# ---------------------------------------------------------------------------

def _progress_wrap(iterable: Any, desc: str = "", verbose: bool = False) -> Any:
    """Wrap an iterable with tqdm if available and verbose, else plain."""
    if _HAS_TQDM and verbose:
        return tqdm(iterable, desc=desc, leave=False, ncols=80)
    return iterable


def run_train_epoch(
    model: AblationSNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Run one training epoch.

    Returns:
        train_loss     : average cross-entropy loss
        train_acc      : accuracy [0, 1]
        avg_forward_ms : average forward pass time per batch (ms)
        avg_backward_ms: average backward pass time per batch (ms)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_fwd_ms = 0.0
    total_bwd_ms = 0.0
    n_batches = 0

    for x_seq, labels in _progress_wrap(loader, desc="train", verbose=verbose):
        x_seq = x_seq.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        t0 = time.perf_counter()
        logits = model(x_seq)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss = criterion(logits, labels)

        t2 = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = labels.shape[0]
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size
        total_fwd_ms += (t1 - t0) * 1000.0
        total_bwd_ms += (t3 - t2) * 1000.0
        n_batches += 1

    n_batches = max(n_batches, 1)
    return (
        total_loss / max(total_samples, 1),
        total_correct / max(total_samples, 1),
        total_fwd_ms / n_batches,
        total_bwd_ms / n_batches,
    )


@torch.no_grad()
def run_validation_epoch(
    model: AblationSNN,
    loader: DataLoader,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    Validate model on loader.

    Returns:
        val_loss : average cross-entropy loss
        val_acc  : accuracy [0, 1]
    """
    model.training and model.train(False)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_seq, labels in _progress_wrap(loader, desc="val", verbose=verbose):
        x_seq = x_seq.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(x_seq)
        loss = criterion(logits, labels)

        batch_size = labels.shape[0]
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return (
        total_loss / max(total_samples, 1),
        total_correct / max(total_samples, 1),
    )


def _peak_memory_mb(device: torch.device) -> float:
    """Return peak GPU memory allocation in MB, or 0 on CPU."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


# ---------------------------------------------------------------------------
# SECTION 12: Seeding utilities
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    if _HAS_NUMPY:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# SECTION 13: Per-config-per-seed training orchestrator
# ---------------------------------------------------------------------------

# Global interrupt flag — set by SIGINT handler
_INTERRUPTED = False


def _sigint_handler(sig: int, frame: Any) -> None:
    global _INTERRUPTED
    _INTERRUPTED = True
    print(
        "\n[ablation_runner] Interrupt received — saving current state and stopping."
    )


def train_single_config(
    config_name: str,
    delay_cfg: DelayConfig,
    tau_cfg: TauConfig,
    seed: int,
    benchmark: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    output_dir: str,
    num_workers: int,
    verbose: bool,
    resume_epoch: int = 0,
    hidden_dim: int = 256,
    num_timesteps: int = 98,
) -> RunResult:
    """
    Train one ablation configuration with one random seed.

    Saves:
      - {output_dir}/{config_name}/seed_{seed}/metrics.json
      - {output_dir}/{config_name}/seed_{seed}/checkpoint_final.pt
      - {output_dir}/{config_name}/seed_{seed}/training.log

    Returns RunResult with all collected metrics.
    """
    global _INTERRUPTED

    run_dir = os.path.join(output_dir, config_name, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "training.log")
    run_logger = _make_file_logger(f"{config_name}_s{seed}", log_path)

    run_logger.info("=" * 60)
    run_logger.info("Config: %s  |  Seed: %d", config_name, seed)
    run_logger.info("Delay mode  : %s (enabled=%s)", delay_cfg.mode, delay_cfg.enabled)
    run_logger.info("Tau mode    : %s (enabled=%s)", tau_cfg.mode, tau_cfg.enabled)
    run_logger.info("=" * 60)

    set_all_seeds(seed)

    # Build model
    try:
        model = create_ablation_model(
            benchmark=benchmark,
            delay_cfg=delay_cfg,
            tau_cfg=tau_cfg,
            device=device,
            num_timesteps=num_timesteps,
            hidden_dim=hidden_dim,
        )
    except Exception as exc:
        msg = f"Model creation failed: {exc}"
        run_logger.error(msg)
        return RunResult(
            config_name=config_name, seed=seed, best_val_acc=0.0,
            best_epoch=0, total_epochs=0, total_time_s=0.0,
            param_count=0, metrics=[], completed=False, error=msg,
        )

    param_count = model.count_parameters()
    run_logger.info("Parameters: %d", param_count)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # DataLoaders
    try:
        train_loader, val_loader = build_dataloaders(
            benchmark=benchmark,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    except Exception as exc:
        msg = f"Dataset loading failed: {exc}\n{traceback.format_exc()}"
        run_logger.error(msg)
        return RunResult(
            config_name=config_name, seed=seed, best_val_acc=0.0,
            best_epoch=0, total_epochs=0, total_time_s=0.0,
            param_count=param_count, metrics=[], completed=False, error=msg,
        )

    # Load existing metrics if resuming
    all_metrics: List[Dict[str, Any]] = []
    metrics_path = os.path.join(run_dir, "metrics.json")
    checkpoint_path = os.path.join(run_dir, "checkpoint_final.pt")

    if resume_epoch > 0 and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(
                ckpt.get("scheduler_state", scheduler.state_dict())
            )
            run_logger.info("Resumed from epoch %d", resume_epoch)
        except Exception as exc:
            run_logger.warning("Could not load checkpoint: %s", exc)
            resume_epoch = 0

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                existing = json.load(f)
            all_metrics = existing if isinstance(existing, list) else []
        except Exception:
            all_metrics = []

    best_val_acc = max(
        (m.get("val_acc", 0.0) for m in all_metrics), default=0.0
    )
    best_epoch = 0
    run_start = time.time()

    for epoch in range(resume_epoch, epochs):
        if _INTERRUPTED:
            run_logger.info("Interrupted at epoch %d", epoch)
            break

        epoch_start = time.time()

        # Sigma schedule for DCLS delays
        sigma: Optional[float] = None
        if delay_cfg.enabled:
            sigma = compute_sigma(
                epoch=epoch,
                total_epochs=epochs,
                sigma_init=delay_cfg.sigma_init,
                sigma_min=delay_cfg.sigma_min,
            )
            model.set_sigma(sigma)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Train
        try:
            train_loss, train_acc, fwd_ms, bwd_ms = run_train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                verbose=verbose,
            )
        except Exception as exc:
            run_logger.error("Train epoch %d failed: %s", epoch, exc)
            if verbose:
                traceback.print_exc()
            break

        # Validate
        try:
            val_loss, val_acc = run_validation_epoch(
                model=model,
                loader=val_loader,
                device=device,
                verbose=verbose,
            )
        except Exception as exc:
            run_logger.error("Validation epoch %d failed: %s", epoch, exc)
            val_loss, val_acc = 0.0, 0.0

        scheduler.step()

        epoch_time = time.time() - epoch_start
        peak_mem = _peak_memory_mb(device)

        # Gather diagnostics
        delay_stats = model.get_delay_stats() or {}
        tau_stats = model.get_tau_stats() or {}

        # Clamp pressure check for learnable tau
        if tau_cfg.enabled and tau_cfg.mode == "heterogeneous_learnable":
            for layer_name, m in model.named_modules():
                if isinstance(m, HeterogeneousTau):
                    m.check_clamp_pressure(layer_name=layer_name)

        em = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            delay_mean=delay_stats.get("delay_mean"),
            delay_std=delay_stats.get("delay_std"),
            delay_entropy=delay_stats.get("delay_entropy"),
            delay_boundary_pct=delay_stats.get("delay_boundary_pct"),
            sigma=sigma,
            tau_mean=tau_stats.get("tau_mean"),
            tau_std=tau_stats.get("tau_std"),
            tau_min_val=tau_stats.get("tau_min_val"),
            tau_max_val=tau_stats.get("tau_max_val"),
            tau_drift=tau_stats.get("tau_drift"),
            forward_time_ms=fwd_ms,
            backward_time_ms=bwd_ms,
            peak_memory_mb=peak_mem,
            epoch_time_s=epoch_time,
        )
        all_metrics.append(em.to_dict())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        run_logger.info(
            "Epoch %3d/%d | loss=%.4f/%.4f | acc=%.3f/%.3f | t=%.1fs mem=%.1fMB",
            epoch + 1, epochs, train_loss, val_loss, train_acc, val_acc,
            epoch_time, peak_mem,
        )

        if verbose:
            _print_epoch_line(
                config_name=config_name, seed=seed, epoch=epoch, epochs=epochs,
                train_loss=train_loss, train_acc=train_acc,
                val_loss=val_loss, val_acc=val_acc,
                epoch_time=epoch_time,
                delay_stats=delay_stats,
                tau_stats=tau_stats,
            )

        # Incremental save
        _save_json(metrics_path, all_metrics)

        # Checkpoint every 10 epochs and at final epoch
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            _save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_acc=val_acc,
            )

    total_time = time.time() - run_start
    run_logger.info(
        "Done. best_val_acc=%.4f at epoch %d | total=%.1fs",
        best_val_acc, best_epoch, total_time,
    )

    return RunResult(
        config_name=config_name,
        seed=seed,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        total_epochs=len(all_metrics),
        total_time_s=total_time,
        param_count=param_count,
        metrics=all_metrics,
        completed=not _INTERRUPTED,
    )


def _print_epoch_line(
    config_name: str,
    seed: int,
    epoch: int,
    epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    epoch_time: float,
    delay_stats: Dict[str, float],
    tau_stats: Dict[str, float],
) -> None:
    extras = ""
    if delay_stats:
        extras += f"  delay_mean={delay_stats.get('delay_mean', 0.0):.2f}"
    if tau_stats:
        extras += f"  tau_mean={tau_stats.get('tau_mean', 0.0):.1f}ms"
    print(
        f"  [{config_name}|s{seed}] {epoch+1:3d}/{epochs}  "
        f"loss={train_loss:.4f}/{val_loss:.4f}  "
        f"acc={train_acc:.3f}/{val_acc:.3f}  "
        f"t={epoch_time:.1f}s{extras}"
    )


# ---------------------------------------------------------------------------
# SECTION 14: Checkpoint and JSON utilities
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    val_acc: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_acc": val_acc,
        },
        path,
    )


def _save_json(path: str, data: Any) -> None:
    """Atomically write data to path as JSON."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _make_file_logger(name: str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# SECTION 15: Resume helper
# ---------------------------------------------------------------------------

def _detect_completed_seeds(
    output_dir: str,
    config_names: List[str],
    seeds: List[int],
) -> Dict[Tuple[str, int], int]:
    """
    Scan output_dir to find which (config, seed) runs already have metrics.json.

    Returns a mapping (config_name, seed) -> next epoch to run (0 if none found).
    """
    completed: Dict[Tuple[str, int], int] = {}
    for cfg_name in config_names:
        for seed in seeds:
            mpath = os.path.join(
                output_dir, cfg_name, f"seed_{seed}", "metrics.json"
            )
            if not os.path.exists(mpath):
                continue
            try:
                with open(mpath) as f:
                    metrics = json.load(f)
                if isinstance(metrics, list) and metrics:
                    last_epoch = metrics[-1].get("epoch", len(metrics) - 1)
                    completed[(cfg_name, seed)] = last_epoch + 1
                else:
                    completed[(cfg_name, seed)] = 0
            except Exception:
                completed[(cfg_name, seed)] = 0
    return completed


def _load_ablation_config(resume_dir: str) -> Optional[Dict[str, Any]]:
    """Load ablation_config.json from a previous run directory."""
    path = os.path.join(resume_dir, "ablation_config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# SECTION 16: Comparison report generator
# ---------------------------------------------------------------------------

def _collect_all_results(
    output_dir: str,
    config_names: List[str],
    seeds: List[int],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all metrics.json files.

    Returns dict mapping config_name -> list of per-seed summary dicts.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}
    for cfg_name in config_names:
        results[cfg_name] = []
        for seed in seeds:
            mpath = os.path.join(
                output_dir, cfg_name, f"seed_{seed}", "metrics.json"
            )
            if not os.path.exists(mpath):
                continue
            try:
                with open(mpath) as f:
                    metrics = json.load(f)
            except Exception:
                continue
            if not metrics:
                continue

            best_val_acc = max(m.get("val_acc", 0.0) for m in metrics)
            best_ep = max(
                range(len(metrics)),
                key=lambda i: metrics[i].get("val_acc", 0.0),
            )
            last = metrics[-1]

            param_count = None
            ckpt_path = os.path.join(
                output_dir, cfg_name, f"seed_{seed}", "checkpoint_final.pt"
            )
            if os.path.exists(ckpt_path):
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    param_count = sum(
                        p.numel()
                        for p in ckpt["model_state"].values()
                        if isinstance(p, torch.Tensor)
                    )
                except Exception:
                    pass

            n = len(metrics)
            results[cfg_name].append({
                "seed": seed,
                "best_val_acc": best_val_acc,
                "best_epoch": best_ep,
                "total_epochs": n,
                "avg_epoch_time_s": (
                    sum(m.get("epoch_time_s", 0) for m in metrics) / n
                ),
                "avg_peak_memory_mb": (
                    sum(m.get("peak_memory_mb", 0) for m in metrics) / n
                ),
                "final_train_acc": last.get("train_acc", 0.0),
                "final_val_acc": last.get("val_acc", 0.0),
                "delay_mean_final": last.get("delay_mean"),
                "delay_std_final": last.get("delay_std"),
                "delay_entropy_final": last.get("delay_entropy"),
                "tau_mean_final": last.get("tau_mean"),
                "tau_std_final": last.get("tau_std"),
                "tau_drift_final": last.get("tau_drift"),
                "param_count": param_count,
                "metrics": metrics,
            })
    return results


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _ascii_curve(values: List[float], width: int = 50, height: int = 6) -> str:
    """Render a list of floats as a small ASCII bar chart."""
    if not values:
        return "(no data)"
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1e-9

    step = max(1, len(values) // width)
    sampled = values[::step][:width]

    rows = []
    for row in range(height, 0, -1):
        threshold = vmin + (vmax - vmin) * row / height
        line = "".join("#" if v >= threshold else " " for v in sampled)
        rows.append(f"{threshold:6.3f} |{line}|")
    rows.append(" " * 7 + "+" + "-" * len(sampled) + "+")
    rows.append(
        " " * 8 + f"epoch 0{' ':>{max(0, len(sampled) - 10)}}epoch {len(values)}"
    )
    return "\n".join(rows)


def generate_comparison_report(
    output_dir: str,
    config_names: List[str],
    seeds: List[int],
    benchmark: str,
    epochs: int,
) -> str:
    """
    Generate and write comparison_report.md to output_dir.

    Returns the absolute path of the written report.
    """
    results = _collect_all_results(output_dir, config_names, seeds)

    lines: List[str] = []
    md = lines.append

    md("# Ablation Study: Learnable Delays + Heterogeneous Tau")
    md("")
    md(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md(f"**Benchmark**: {benchmark}")
    md(f"**Epochs**: {epochs}")
    md(f"**Seeds**: {', '.join(str(s) for s in seeds)}")
    md(f"**Output directory**: `{output_dir}`")
    md("")

    # ---- Section 1: Accuracy summary table ----
    md("## 1. Accuracy Summary")
    md("")
    md("| Config | Best Val Acc (mean ± std) | Seeds Completed | Params |")
    md("|--------|--------------------------|-----------------|--------|")

    config_summary: Dict[str, Dict[str, Any]] = {}
    for cfg_name in config_names:
        seed_results = results.get(cfg_name, [])
        accs = [r["best_val_acc"] for r in seed_results]
        mean, std = _mean_std(accs)
        params = next(
            (r["param_count"] for r in seed_results if r["param_count"]), None
        )
        params_str = f"{params:,}" if params else "N/A"
        md(
            f"| `{cfg_name}` | {mean:.4f} ± {std:.4f} "
            f"| {len(accs)}/{len(seeds)} | {params_str} |"
        )
        config_summary[cfg_name] = {
            "mean": mean, "std": std, "accs": accs,
            "seed_results": seed_results,
        }
    md("")

    # ---- Section 2: Timing comparison ----
    md("## 2. Timing Comparison")
    md("")
    md("| Config | Avg Epoch Time (s) | Overhead vs Baseline | Peak Memory (MB) |")
    md("|--------|-------------------|----------------------|------------------|")

    baseline_time: Optional[float] = None
    timing_rows: List[Tuple[str, float, float, float]] = []
    for cfg_name in config_names:
        seed_results = results.get(cfg_name, [])
        times = [r["avg_epoch_time_s"] for r in seed_results]
        mems = [r["avg_peak_memory_mb"] for r in seed_results]
        mean_t, _ = _mean_std(times)
        mean_m, _ = _mean_std(mems)
        if cfg_name == "baseline" or baseline_time is None:
            baseline_time = mean_t if mean_t > 0 else None
        timing_rows.append((cfg_name, mean_t, mean_m, 0.0))

    for i, (cfg_name, mean_t, mean_m, _) in enumerate(timing_rows):
        if baseline_time and baseline_time > 0:
            overhead = mean_t / baseline_time
        else:
            overhead = 1.0
        timing_rows[i] = (cfg_name, mean_t, mean_m, overhead)
        md(
            f"| `{cfg_name}` | {mean_t:.2f} | {overhead:.2f}x | {mean_m:.1f} |"
        )
    md("")

    # ---- Section 3: Accuracy curves ----
    md("## 3. Validation Accuracy Curves")
    md("")
    for cfg_name in config_names:
        seed_results = results.get(cfg_name, [])
        if not seed_results:
            md(f"### `{cfg_name}` — no data")
            md("")
            continue
        md(f"### `{cfg_name}`")
        max_ep = max(len(r["metrics"]) for r in seed_results)
        avg_val_accs = []
        for ep in range(max_ep):
            vals = [
                r["metrics"][ep]["val_acc"]
                for r in seed_results
                if ep < len(r["metrics"])
            ]
            avg_val_accs.append(sum(vals) / len(vals) if vals else 0.0)
        md("```")
        md(_ascii_curve(avg_val_accs))
        md("```")
        if avg_val_accs:
            best_ep = avg_val_accs.index(max(avg_val_accs))
            md(f"Best: {max(avg_val_accs):.4f} at epoch {best_ep}")
        md("")

    # ---- Section 4: Delay statistics ----
    has_delay_configs = [
        c for c in config_names
        if ABLATION_CONFIGS.get(c, {}).get("delay", DelayConfig()).enabled
    ]
    if has_delay_configs:
        md("## 4. Delay Statistics (Final Epoch)")
        md("")
        md(
            "| Config | Delay Mean (steps) | Delay Std | "
            "Delay Entropy | Boundary % |"
        )
        md("|--------|--------------------|-----------|---------------|------------|")
        for cfg_name in has_delay_configs:
            sr = results.get(cfg_name, [])
            dm_vals = [r["delay_mean_final"] for r in sr if r["delay_mean_final"] is not None]
            ds_vals = [r["delay_std_final"] for r in sr if r["delay_std_final"] is not None]
            de_vals = [r["delay_entropy_final"] for r in sr if r["delay_entropy_final"] is not None]
            mean_dm, _ = _mean_std(dm_vals)
            mean_ds, _ = _mean_std(ds_vals)
            mean_de, _ = _mean_std(de_vals)
            md(
                f"| `{cfg_name}` | {mean_dm:.2f} | {mean_ds:.2f} "
                f"| {mean_de:.3f} | — |"
            )
        md("")
        md(
            "> Delay positions are in simulation timesteps "
            "(0 = no delay, max_delay-1 = maximum)."
        )
        md(
            "> Entropy measures sharpness of the Gaussian delay kernel "
            "(lower = more committed, sharper position)."
        )
        md("")

    # ---- Section 5: Tau statistics ----
    has_tau_configs = [
        c for c in config_names
        if ABLATION_CONFIGS.get(c, {}).get("tau", TauConfig()).enabled
    ]
    if has_tau_configs:
        md("## 5. Tau Statistics (Final Epoch)")
        md("")
        md("| Config | Tau Mean (ms) | Tau Std | Tau Drift |")
        md("|--------|--------------|---------|-----------|")
        for cfg_name in has_tau_configs:
            sr = results.get(cfg_name, [])
            tm_vals = [r["tau_mean_final"] for r in sr if r["tau_mean_final"] is not None]
            ts_vals = [r["tau_std_final"] for r in sr if r["tau_std_final"] is not None]
            td_vals = [r["tau_drift_final"] for r in sr if r["tau_drift_final"] is not None]
            mean_tm, _ = _mean_std(tm_vals)
            mean_ts, _ = _mean_std(ts_vals)
            mean_td, _ = _mean_std(td_vals)
            md(
                f"| `{cfg_name}` | {mean_tm:.2f} | {mean_ts:.2f} "
                f"| {mean_td:.3f} |"
            )
        md("")
        md(
            "> Tau drift = mean absolute change from initialization (ms). "
            "Near-zero drift in learnable mode may indicate vanishing gradients."
        )
        md("")

    # ---- Section 6: Per-seed detail ----
    md("## 6. Per-Seed Details")
    md("")
    for cfg_name in config_names:
        seed_results = results.get(cfg_name, [])
        if not seed_results:
            continue
        md(f"### `{cfg_name}`")
        md("")
        md(
            "| Seed | Best Val Acc | Best Epoch | "
            "Final Val Acc | Avg Epoch Time (s) |"
        )
        md("|------|-------------|-----------|---------------|--------------------|")
        for r in seed_results:
            md(
                f"| {r['seed']} | {r['best_val_acc']:.4f} | {r['best_epoch']} "
                f"| {r['final_val_acc']:.4f} | {r['avg_epoch_time_s']:.2f} |"
            )
        md("")

    # ---- Section 7: Recommendations ----
    md("## 7. Recommendations")
    md("")
    ranked = sorted(
        [c for c in config_names if config_summary.get(c, {}).get("accs")],
        key=lambda c: config_summary[c]["mean"],
        reverse=True,
    )
    if ranked:
        best_config = ranked[0]
        best_mean = config_summary[best_config]["mean"]
        baseline_mean = config_summary.get("baseline", {}).get("mean", 0.0)
        delta = best_mean - baseline_mean

        md(f"- Best configuration: **`{best_config}`** — mean val accuracy {best_mean:.4f}")
        md(
            f"- Accuracy gain over baseline: **{delta:+.4f}** "
            f"({delta * 100:+.2f} percentage points)"
        )
        if delta > 0.01:
            md(
                "- The gain is substantial (>1 pp). "
                "Recommend adopting this config for production training."
            )
        elif delta > 0.002:
            md(
                "- The gain is moderate (0.2–1 pp). "
                "Evaluate compute overhead before adopting."
            )
        else:
            md(
                "- The gain is marginal (<0.2 pp). "
                "Baseline may be sufficient for this benchmark."
            )
    md("")

    md("### Overhead Assessment")
    md("")
    _baseline_t: Optional[float] = next(
        (t for c, t, _m, _o in timing_rows if c == "baseline"), None
    )
    for cfg_name, mean_t, _mem, overhead in timing_rows:
        if cfg_name == "baseline":
            md(f"- `baseline`: reference ({mean_t:.2f} s/epoch)")
            continue
        if overhead < 1.2:
            verdict = "negligible overhead — use freely"
        elif overhead < 1.5:
            verdict = "moderate overhead — acceptable for most settings"
        elif overhead < 2.0:
            verdict = "significant overhead — justify with accuracy gain"
        else:
            verdict = "high overhead — consider gradient checkpointing or reduced max_delay"
        md(f"- `{cfg_name}`: {overhead:.2f}x baseline — {verdict}")

    md("")
    md("---")
    md("*Generated by `ablation_runner.py`*")
    md("")

    report_text = "\n".join(lines)
    report_path = os.path.join(output_dir, "comparison_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)

    return report_path


# ---------------------------------------------------------------------------
# SECTION 17: Dry run printer
# ---------------------------------------------------------------------------

def run_dry(
    config_names: List[str],
    seeds: List[int],
    benchmark: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    output_dir: str,
    num_workers: int,
    hidden_dim: int = 256,
    num_timesteps: int = 98,
) -> None:
    """Print the full ablation plan without running any training."""
    print("=" * 70)
    print("ABLATION DRY RUN — no training will occur")
    print("=" * 70)
    print(f"  Configs      : {', '.join(config_names)}")
    print(f"  Seeds        : {', '.join(str(s) for s in seeds)}")
    print(f"  Benchmark    : {benchmark}")
    print(f"  Epochs       : {epochs}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Device       : {device}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Num workers  : {num_workers}")
    print(f"  brain_ai     : {'available' if _HAS_BRAIN_AI else 'not found (using fallback)'}")
    print(f"  torchvision  : {'available' if _HAS_TORCHVISION else 'NOT FOUND — required for this benchmark'}")
    print(f"  tonic        : {'available' if _HAS_TONIC else 'not found (required only for shd)'}")
    print(f"  tqdm         : {'available' if _HAS_TQDM else 'not found (fallback to print)'}")
    print()

    binfo = _get_benchmark_info(benchmark, num_timesteps)
    print("Benchmark:")
    for k, v in binfo.items():
        print(f"  {k}: {v}")
    print()

    print("Configurations:")
    print("-" * 70)
    total_runs = 0
    for cfg_name in config_names:
        if cfg_name not in ABLATION_CONFIGS:
            print(f"  [{cfg_name}] UNKNOWN — will be skipped")
            continue
        cfg = ABLATION_CONFIGS[cfg_name]
        delay_cfg: DelayConfig = cfg["delay"]
        tau_cfg: TauConfig = cfg["tau"]
        desc = cfg.get("description", "")

        try:
            model = create_ablation_model(
                benchmark=benchmark,
                delay_cfg=delay_cfg,
                tau_cfg=tau_cfg,
                device=torch.device("cpu"),
                num_timesteps=num_timesteps,
                hidden_dim=hidden_dim,
            )
            params = model.count_parameters()
            params_str = f"{params:,}"
        except Exception as exc:
            params_str = f"ERROR building model: {exc}"

        print(f"  [{cfg_name}]")
        print(f"    Description : {desc}")
        print(f"    Delay       : enabled={delay_cfg.enabled}  mode={delay_cfg.mode}")
        if delay_cfg.enabled:
            print(
                f"                  max_delay={delay_cfg.max_delay}  "
                f"sigma {delay_cfg.sigma_init}->{delay_cfg.sigma_min}"
            )
        print(f"    Tau         : enabled={tau_cfg.enabled}  mode={tau_cfg.mode}")
        if tau_cfg.enabled:
            print(
                f"                  strategy={tau_cfg.init_strategy}  "
                f"tau=[{tau_cfg.tau_min}, {tau_cfg.tau_max}]ms  "
                f"tau_0={tau_cfg.tau_0}ms"
            )
        print(f"    Parameters  : {params_str}")
        total_runs += len(seeds)
    print()

    # Estimated time (rough: 30s/epoch on CPU)
    seconds_per_epoch = 30
    total_seconds = total_runs * epochs * seconds_per_epoch
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    print(f"Total training runs  : {total_runs}  ({len(config_names)} configs x {len(seeds)} seeds)")
    print(f"Estimated time (CPU) : ~{hours}h {minutes}m  (assuming {seconds_per_epoch}s/epoch)")
    print()

    print("Output directory structure:")
    print(f"  {output_dir}/")
    print(f"  |-- ablation_config.json")
    for cfg_name in config_names:
        print(f"  |-- {cfg_name}/")
        for seed in seeds:
            print(f"  |   |-- seed_{seed}/")
            print(f"  |   |   |-- metrics.json")
            print(f"  |   |   |-- checkpoint_final.pt")
            print(f"  |   |   +-- training.log")
    print(f"  +-- comparison_report.md")
    print()
    print("Dry run complete. Remove --dry-run to start training.")


# ---------------------------------------------------------------------------
# SECTION 18: Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    global _INTERRUPTED

    parser = argparse.ArgumentParser(
        description="Ablation runner: learnable delays vs. heterogeneous tau",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="baseline,delays_only,hetero_only,both",
        help="Comma-separated list of configs to run (default: all 4)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456",
        help="Comma-separated list of random seeds (default: 42,123,456)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="smnist",
        choices=["smnist", "psmnist", "mnist", "shd"],
        help="Benchmark dataset (default: smnist)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs per run (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ../results/ablation_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda / cpu / cuda:N (default: auto-detect)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader num_workers (default: 2)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for SNN layers (default: 256)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs, print plan, do not train",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="DIR",
        help="Resume a previous ablation run from this directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-epoch progress to stdout",
    )
    args = parser.parse_args()

    # ---- Parse lists ----
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    unknown = [c for c in config_names if c not in ABLATION_CONFIGS]
    if unknown:
        parser.error(
            f"Unknown config(s): {unknown}. "
            f"Valid configs: {sorted(ABLATION_CONFIGS.keys())}"
        )

    # ---- Device ----
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ---- Output directory ----
    if args.resume:
        output_dir = os.path.abspath(args.resume)
        if not os.path.isdir(output_dir):
            parser.error(f"--resume directory does not exist: {output_dir}")
        prev_cfg = _load_ablation_config(output_dir)
        if prev_cfg:
            print(f"[resume] Loaded previous config from {output_dir}/ablation_config.json")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        skill_root = os.path.dirname(_SCRIPT_DIR)  # delays-heterogeneous-tau/
        default_dir = os.path.join(skill_root, "results", f"ablation_{ts}")
        output_dir = os.path.abspath(args.output_dir or default_dir)

    os.makedirs(output_dir, exist_ok=True)

    binfo = _get_benchmark_info(args.benchmark)
    num_timesteps = binfo["T"]

    # ---- Dry run mode ----
    if args.dry_run:
        run_dry(
            config_names=config_names,
            seeds=seeds,
            benchmark=args.benchmark,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            output_dir=output_dir,
            num_workers=args.num_workers,
            hidden_dim=args.hidden_dim,
            num_timesteps=num_timesteps,
        )
        return

    # ---- Setup global logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(output_dir, "ablation_runner.log")
            ),
        ],
    )
    logger = logging.getLogger("ablation_runner")

    # ---- Save run config ----
    ablation_config_data: Dict[str, Any] = {
        "config_names": config_names,
        "seeds": seeds,
        "benchmark": args.benchmark,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": str(device),
        "num_workers": args.num_workers,
        "hidden_dim": args.hidden_dim,
        "num_timesteps": num_timesteps,
        "output_dir": output_dir,
        "timestamp": datetime.now().isoformat(),
        "configs": {
            name: {
                "delay": ABLATION_CONFIGS[name]["delay"].to_dict(),
                "tau": ABLATION_CONFIGS[name]["tau"].to_dict(),
                "description": ABLATION_CONFIGS[name].get("description", ""),
            }
            for name in config_names
        },
    }
    _save_json(
        os.path.join(output_dir, "ablation_config.json"), ablation_config_data
    )

    # ---- Detect already-completed seeds ----
    resume_map: Dict[Tuple[str, int], int] = {}
    if args.resume:
        resume_map = _detect_completed_seeds(output_dir, config_names, seeds)
        if resume_map:
            logger.info(
                "Resume mode: %d (config, seed) pairs have partial or complete data.",
                len(resume_map),
            )

    # ---- Install interrupt handler ----
    signal.signal(signal.SIGINT, _sigint_handler)

    # ---- Header ----
    print("=" * 70)
    print("Ablation Runner: Learnable Delays + Heterogeneous Tau")
    print("=" * 70)
    print(f"  Configs   : {', '.join(config_names)}")
    print(f"  Seeds     : {', '.join(str(s) for s in seeds)}")
    print(f"  Benchmark : {args.benchmark}  ({binfo['description']})")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Device    : {device}")
    print(f"  Output    : {output_dir}")
    print("=" * 70)
    print()

    # ---- Main training loop ----
    all_results: List[RunResult] = []
    total_runs = len(config_names) * len(seeds)
    run_idx = 0

    for cfg_name in config_names:
        if _INTERRUPTED:
            logger.info("Stopping before config %s due to interrupt.", cfg_name)
            break

        cfg = ABLATION_CONFIGS[cfg_name]
        delay_cfg: DelayConfig = cfg["delay"]
        tau_cfg: TauConfig = cfg["tau"]

        for seed in seeds:
            if _INTERRUPTED:
                logger.info("Stopping before seed %d due to interrupt.", seed)
                break

            run_idx += 1
            key = (cfg_name, seed)
            resume_epoch = resume_map.get(key, 0)

            # Skip if fully completed
            if resume_epoch >= args.epochs:
                logger.info(
                    "[%d/%d] Skipping %s seed=%d (completed %d/%d epochs)",
                    run_idx, total_runs, cfg_name, seed,
                    resume_epoch, args.epochs,
                )
                continue

            if resume_epoch > 0:
                logger.info(
                    "[%d/%d] Resuming %s seed=%d from epoch %d",
                    run_idx, total_runs, cfg_name, seed, resume_epoch,
                )
            else:
                logger.info(
                    "[%d/%d] Starting %s seed=%d",
                    run_idx, total_runs, cfg_name, seed,
                )

            print(
                f"\n[{run_idx}/{total_runs}] "
                f"config={cfg_name!r}  seed={seed}"
                + (f"  (resume from epoch {resume_epoch})" if resume_epoch > 0 else "")
            )

            result = train_single_config(
                config_name=cfg_name,
                delay_cfg=deepcopy(delay_cfg),
                tau_cfg=deepcopy(tau_cfg),
                seed=seed,
                benchmark=args.benchmark,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                output_dir=output_dir,
                num_workers=args.num_workers,
                verbose=args.verbose,
                resume_epoch=resume_epoch,
                hidden_dim=args.hidden_dim,
                num_timesteps=num_timesteps,
            )
            all_results.append(result)

            status_str = "OK" if result.completed else "FAILED"
            print(
                f"  -> {status_str}  "
                f"best_val_acc={result.best_val_acc:.4f} at epoch {result.best_epoch}  "
                f"({result.total_time_s:.1f}s total)"
            )
            if result.error:
                print(f"  -> Error: {result.error}")

    if _INTERRUPTED:
        print(
            "\n[ablation_runner] Training interrupted. "
            "Partial results have been saved."
        )

    # ---- Save summary ----
    summary_path = os.path.join(output_dir, "summary.json")
    summary_data = [
        {
            "config_name": r.config_name,
            "seed": r.seed,
            "best_val_acc": r.best_val_acc,
            "best_epoch": r.best_epoch,
            "total_epochs": r.total_epochs,
            "total_time_s": r.total_time_s,
            "param_count": r.param_count,
            "completed": r.completed,
            "error": r.error,
        }
        for r in all_results
    ]
    _save_json(summary_path, summary_data)
    logger.info("Summary saved: %s", summary_path)

    # ---- Generate comparison report ----
    print("\nGenerating comparison report...")
    try:
        report_path = generate_comparison_report(
            output_dir=output_dir,
            config_names=config_names,
            seeds=seeds,
            benchmark=args.benchmark,
            epochs=args.epochs,
        )
        print(f"  Report written: {report_path}")
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        if args.verbose:
            traceback.print_exc()

    # ---- Final table ----
    print()
    print("=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Config':<16} {'Val Acc (mean+-std)':<24} {'Seeds':<8} {'Params':>10}")
    print("-" * 70)
    for cfg_name in config_names:
        cfg_results = [
            r for r in all_results
            if r.config_name == cfg_name and r.completed
        ]
        accs = [r.best_val_acc for r in cfg_results]
        mean, std = _mean_std(accs)
        params = next((r.param_count for r in cfg_results if r.param_count), 0)
        print(
            f"{cfg_name:<16} {mean:.4f} +- {std:.4f}          "
            f"{len(accs)}/{len(seeds)}    {params:>10,}"
        )
    print("=" * 70)
    print(f"All outputs: {output_dir}")
    print()


# ---------------------------------------------------------------------------
# SECTION 19: Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
