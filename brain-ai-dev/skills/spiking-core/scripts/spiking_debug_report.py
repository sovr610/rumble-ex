#!/usr/bin/env python3
"""
spiking_debug_report.py -- Comprehensive debug report for spiking layers.

Generates a detailed diagnostic report covering:
1. Per-layer firing rates (mean, min, max, std)
2. Membrane potential statistics (mean, var, max, distribution shape)
3. Spike sparsity analysis (dead neurons, saturated neurons)
4. Gradient statistics (norms for W_in, W_rec, threshold, beta)
5. Surrogate gradient analysis (effective width, max gradient)
6. State health check (NaN/Inf detection, numerical range)
7. Temporal dynamics (firing rate over time, burst detection)

Usage:
    python spiking_debug_report.py                    # Full report, all neuron types
    python spiking_debug_report.py --neuron lif       # Single neuron type
    python spiking_debug_report.py --json             # JSON output to stdout
    python spiking_debug_report.py --verbose          # Include per-neuron details
    python spiking_debug_report.py --timesteps 100    # Custom T
    python spiking_debug_report.py --check-gradients  # Include gradient analysis
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup -- locate repo root the same way validate_encoders.py does.
# Layout: brain-ai-dev/skills/spiking-core/scripts/spiking_debug_report.py
#         brain_ai/                                        (repo root, 4 up)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
)
if os.path.isdir(os.path.join(_REPO_ROOT, "brain_ai")):
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Optional brain_ai import -- fall through to inline stubs if unavailable.
# ---------------------------------------------------------------------------
_BRAIN_AI_AVAILABLE = False
_IMPORT_ERROR: Optional[str] = None

try:
    from brain_ai.core.neurons import (
        LIFNeuron,
        AdaptiveLIFNeuron,
        RecurrentLIFNeuron,
        AdvancedLIFNeuron,
        ATanSurrogate,
        FastSigmoidSurrogate,
        StraightThroughSurrogate,
        get_surrogate,
    )
    _BRAIN_AI_AVAILABLE = True
except Exception as _exc:
    _IMPORT_ERROR = "".join(
        traceback.format_exception(type(_exc), _exc, _exc.__traceback__)
    )


# ---------------------------------------------------------------------------
# Inline fallback neuron implementations.
# Used when brain_ai is not importable so the script remains self-contained.
# ---------------------------------------------------------------------------
if not _BRAIN_AI_AVAILABLE:

    class ATanSurrogate(torch.autograd.Function):
        """Arctangent surrogate gradient."""

        alpha = 2.0

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return (x >= 0).float()

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            alpha = ATanSurrogate.alpha
            grad = alpha / (2 * (1 + (torch.pi * alpha * x) ** 2))
            return grad_output * grad

    class FastSigmoidSurrogate(torch.autograd.Function):
        """Fast sigmoid surrogate gradient."""

        slope = 25.0

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return (x >= 0).float()

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            slope = FastSigmoidSurrogate.slope
            grad = slope / (2 * (1 + slope * x.abs()) ** 2)
            return grad_output * grad

    class StraightThroughSurrogate(torch.autograd.Function):
        """Straight-through estimator -- passes gradients unchanged."""

        @staticmethod
        def forward(ctx, x):
            return (x >= 0).float()

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    def get_surrogate(name: str, **kwargs):
        """Return a surrogate gradient apply function by name."""
        if name == "atan":
            if "alpha" in kwargs:
                ATanSurrogate.alpha = kwargs["alpha"]
            return ATanSurrogate.apply
        elif name == "fast_sigmoid":
            if "slope" in kwargs:
                FastSigmoidSurrogate.slope = kwargs["slope"]
            return FastSigmoidSurrogate.apply
        elif name == "straight_through":
            return StraightThroughSurrogate.apply
        else:
            raise ValueError(f"Unknown surrogate: {name}")

    class LIFNeuron(nn.Module):
        """Leaky Integrate-and-Fire neuron (fallback)."""

        def __init__(
            self,
            beta=0.9,
            threshold=1.0,
            reset_mechanism="subtract",
            surrogate="atan",
            surrogate_alpha=2.0,
            learn_beta=False,
            learn_threshold=False,
        ):
            super().__init__()
            self.reset_mechanism = reset_mechanism
            self.spike_fn = get_surrogate(surrogate, alpha=surrogate_alpha)
            if learn_beta:
                self.beta = nn.Parameter(torch.tensor(beta))
            else:
                self.register_buffer("beta", torch.tensor(beta))
            if learn_threshold:
                self.threshold = nn.Parameter(torch.tensor(threshold))
            else:
                self.register_buffer("threshold", torch.tensor(threshold))
            self.mem = None

        def reset_mem(self):
            self.mem = None

        def forward(self, x, mem=None):
            if mem is not None:
                self.mem = mem
            if self.mem is None:
                self.mem = torch.zeros_like(x)
            self.mem = self.beta * self.mem + x
            spk = self.spike_fn(self.mem - self.threshold)
            if self.reset_mechanism == "subtract":
                self.mem = self.mem - spk * self.threshold
            else:
                self.mem = self.mem * (1 - spk)
            return spk, self.mem

    class AdaptiveLIFNeuron(nn.Module):
        """LIF with spike-frequency adaptation (fallback)."""

        def __init__(
            self,
            beta=0.9,
            threshold=1.0,
            adaptation_beta=0.1,
            adaptation_decay=0.95,
            surrogate="atan",
        ):
            super().__init__()
            self.beta = beta
            self.threshold_base = threshold
            self.adaptation_beta = adaptation_beta
            self.adaptation_decay = adaptation_decay
            self.spike_fn = get_surrogate(surrogate)
            self.mem = None
            self.adaptation = None

        def reset_mem(self):
            self.mem = None
            self.adaptation = None

        def forward(self, x):
            if self.mem is None:
                self.mem = torch.zeros_like(x)
                self.adaptation = torch.zeros_like(x)
            threshold = self.threshold_base + self.adaptation_beta * self.adaptation
            self.mem = self.beta * self.mem + x
            spk = self.spike_fn(self.mem - threshold)
            self.mem = self.mem - spk * threshold
            self.adaptation = self.adaptation_decay * self.adaptation + spk
            return spk, self.mem

    class RecurrentLIFNeuron(nn.Module):
        """LIF with lateral recurrent connections (fallback)."""

        def __init__(
            self,
            size,
            beta=0.9,
            threshold=1.0,
            recurrent_weight_scale=0.1,
            surrogate="atan",
        ):
            super().__init__()
            self.beta = beta
            self.threshold = threshold
            self.spike_fn = get_surrogate(surrogate)
            self.recurrent = nn.Linear(size, size, bias=False)
            nn.init.normal_(self.recurrent.weight, std=recurrent_weight_scale)
            self.mem = None
            self.prev_spk = None

        def reset_mem(self):
            self.mem = None
            self.prev_spk = None

        def forward(self, x):
            if self.mem is None:
                self.mem = torch.zeros_like(x)
                self.prev_spk = torch.zeros_like(x)
            recurrent_input = self.recurrent(self.prev_spk)
            self.mem = self.beta * self.mem + x + recurrent_input
            spk = self.spike_fn(self.mem - self.threshold)
            self.mem = self.mem - spk * self.threshold
            self.prev_spk = spk
            return spk, self.mem

    class AdvancedLIFNeuron(nn.Module):
        """LIF with learnable delays and heterogeneous time constants (fallback)."""

        def __init__(
            self,
            size,
            beta_init=0.9,
            threshold=1.0,
            learnable_beta=True,
            max_delay=10,
            use_delays=True,
            use_adaptive_threshold=False,
            surrogate="atan",
        ):
            super().__init__()
            self.size = size
            self.max_delay = max_delay
            self.use_delays = use_delays
            self.use_adaptive_threshold = use_adaptive_threshold
            self.spike_fn = get_surrogate(surrogate)
            logit_beta = math.log(beta_init / (1.0 - beta_init))
            if learnable_beta:
                self.log_beta = nn.Parameter(
                    torch.full((size,), logit_beta) + torch.randn(size) * 0.1
                )
            else:
                self.register_buffer(
                    "log_beta", torch.full((size,), logit_beta)
                )
            self.register_buffer("threshold_base", torch.tensor(threshold))
            if use_delays:
                self.delay_weights = nn.Parameter(torch.zeros(size, max_delay))
                nn.init.normal_(self.delay_weights, std=0.1)
            else:
                self.delay_weights = None
            if use_adaptive_threshold:
                self.adaptation_weight = nn.Parameter(torch.tensor(0.1))
                self.adaptation_decay_param = nn.Parameter(torch.tensor(0.95))
            self.mem = None
            self.spike_history = None
            self.adaptation = None

        @property
        def beta(self):
            return torch.sigmoid(self.log_beta)

        def reset_mem(self):
            self.mem = None
            self.spike_history = None
            self.adaptation = None

        def forward(self, x, mem=None):
            batch_size = x.shape[0]
            device = x.device
            if mem is not None:
                self.mem = mem
            if self.mem is None:
                self.mem = torch.zeros(batch_size, self.size, device=device)
            if self.spike_history is None and self.use_delays:
                self.spike_history = torch.zeros(
                    batch_size, self.max_delay, self.size, device=device
                )
            if self.adaptation is None and self.use_adaptive_threshold:
                self.adaptation = torch.zeros(batch_size, self.size, device=device)

            # Apply delays if history is available
            if self.use_delays and self.spike_history is not None:
                if self.spike_history.shape[1] >= self.max_delay:
                    recent = self.spike_history[:, -self.max_delay:]
                    delay_attn = torch.softmax(self.delay_weights, dim=-1)
                    delayed = torch.einsum("bdn,nd->bn", recent, delay_attn)
                    x = x + 0.1 * delayed

            beta = self.beta
            self.mem = beta.unsqueeze(0) * self.mem + x

            if self.use_adaptive_threshold and self.adaptation is not None:
                threshold = (
                    self.threshold_base
                    + self.adaptation_weight * self.adaptation
                )
            else:
                threshold = self.threshold_base

            spk = self.spike_fn(self.mem - threshold)
            self.mem = self.mem - spk * threshold

            if self.use_adaptive_threshold and self.adaptation is not None:
                self.adaptation = self.adaptation_decay_param * self.adaptation + spk

            if self.use_delays and self.spike_history is not None:
                self.spike_history = torch.cat(
                    [self.spike_history[:, 1:], spk.unsqueeze(1)], dim=1
                )
            return spk, self.mem


# ---------------------------------------------------------------------------
# Terminal colour helpers (same pattern as validate_encoders.py)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m" if _USE_COLOR else text


def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m" if _USE_COLOR else text


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m" if _USE_COLOR else text


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _USE_COLOR else text


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if _USE_COLOR else text


def _cyan(text: str) -> str:
    return f"\033[96m{text}\033[0m" if _USE_COLOR else text


# ===========================================================================
# SECTION 2: Data collection functions
# ===========================================================================


def collect_firing_rates(spikes: torch.Tensor) -> Dict[str, Any]:
    """Compute per-layer firing rate statistics from a spike tensor.

    Args:
        spikes: Tensor of shape ``(T, B, N)`` or ``(B, T, N)``.
                If 2-D ``(T, N)``, a dummy batch dim is added.

    Returns:
        dict with keys ``mean``, ``min``, ``max``, ``std`` (float),
        ``per_neuron_rates`` (List[float]), and ``T``, ``B``, ``N`` (int).
    """
    with torch.no_grad():
        if spikes.ndim == 2:
            spikes = spikes.unsqueeze(1)
        if spikes.ndim != 3:
            raise ValueError(
                f"spikes must be 3-D (T,B,N) or (B,T,N), got {spikes.shape}"
            )

        T, B, N = spikes.shape
        per_neuron = spikes.float().mean(dim=(0, 1))  # (N,)

        mean_rate = per_neuron.mean().item()
        min_rate = per_neuron.min().item()
        max_rate = per_neuron.max().item()
        std_rate = per_neuron.std().item() if N > 1 else 0.0

        return {
            "mean": mean_rate,
            "min": min_rate,
            "max": max_rate,
            "std": std_rate,
            "per_neuron_rates": per_neuron.tolist(),
            "T": T,
            "B": B,
            "N": N,
        }


def collect_membrane_stats(
    membrane_traces: List[torch.Tensor],
) -> Dict[str, float]:
    """Compute membrane potential statistics from a list of per-step tensors.

    Args:
        membrane_traces: List of length ``T``, each tensor ``(B, N)`` or ``(N,)``.

    Returns:
        dict with ``mean``, ``var``, ``max``, ``min``, ``skewness``,
        ``fraction_near_threshold``, ``fraction_near_rest``.
    """
    if not membrane_traces:
        return {
            "mean": float("nan"),
            "var": float("nan"),
            "max": float("nan"),
            "min": float("nan"),
            "skewness": float("nan"),
            "fraction_near_threshold": float("nan"),
            "fraction_near_rest": float("nan"),
        }

    with torch.no_grad():
        stacked = torch.stack([m.float() for m in membrane_traces], dim=0)
        flat = stacked.reshape(-1)

        mean_v = flat.mean().item()
        var_v = flat.var().item()
        max_v = flat.max().item()
        min_v = flat.min().item()

        std_v = math.sqrt(max(var_v, 1e-12))
        skewness = ((flat - mean_v) ** 3).mean().item() / (std_v ** 3 + 1e-12)

        # Threshold assumed to be 1.0 (LIF default)
        threshold_approx = 1.0
        near_thresh = (flat > 0.8 * threshold_approx).float().mean().item()
        near_rest = (flat.abs() < 0.05).float().mean().item()

        return {
            "mean": mean_v,
            "var": var_v,
            "max": max_v,
            "min": min_v,
            "skewness": skewness,
            "fraction_near_threshold": near_thresh,
            "fraction_near_rest": near_rest,
        }


def collect_sparsity_analysis(spikes: torch.Tensor) -> Dict[str, Any]:
    """Analyze spike sparsity across all neurons.

    Args:
        spikes: ``(T, B, N)`` spike tensor (binary).

    Returns:
        dict with ``overall_sparsity``, ``dead_neuron_fraction``,
        ``saturated_neuron_fraction``, ``active_neuron_fraction``,
        ``dead_count``, ``saturated_count``, ``total_neurons`` (ints).
    """
    with torch.no_grad():
        if spikes.ndim == 2:
            spikes = spikes.unsqueeze(1)

        T, B, N = spikes.shape
        spk = spikes.float()

        overall_sparsity = 1.0 - spk.mean().item()

        total_per_neuron = spk.sum(dim=(0, 1))  # (N,)
        max_possible = float(T * B)

        dead_mask = total_per_neuron == 0
        dead_frac = dead_mask.float().mean().item()
        dead_count = int(dead_mask.sum().item())

        sat_mask = total_per_neuron >= 0.9 * max_possible
        sat_frac = sat_mask.float().mean().item()
        sat_count = int(sat_mask.sum().item())

        active_frac = max(0.0, 1.0 - dead_frac - sat_frac)

        return {
            "overall_sparsity": overall_sparsity,
            "dead_neuron_fraction": dead_frac,
            "saturated_neuron_fraction": sat_frac,
            "active_neuron_fraction": active_frac,
            "dead_count": dead_count,
            "saturated_count": sat_count,
            "total_neurons": N,
        }


def collect_gradient_stats(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Collect gradient norms for all parameters after ``loss.backward()``.

    Args:
        model: Network whose ``named_parameters()`` are inspected.

    Returns:
        dict mapping ``param_name`` to
        ``{norm, mean, max, has_nan, has_inf, learnable}``.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for name, param in model.named_parameters():
        entry: Dict[str, Any] = {
            "norm": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
            "has_nan": False,
            "has_inf": False,
            "learnable": param.requires_grad,
        }
        if param.grad is not None:
            g = param.grad.float()
            entry["norm"] = g.norm().item()
            entry["mean"] = g.mean().item()
            entry["max"] = g.abs().max().item()
            entry["has_nan"] = bool(torch.isnan(g).any().item())
            entry["has_inf"] = bool(torch.isinf(g).any().item())
        result[name] = entry
    return result


def collect_temporal_dynamics(spikes: torch.Tensor) -> Dict[str, Any]:
    """Analyze temporal firing patterns in the spike train.

    Args:
        spikes: ``(T, B, N)`` spike tensor.

    Returns:
        dict with ``firing_rate_over_time`` (List[float] of length T),
        ``burst_count``, ``isi_mean``, ``isi_std``, ``isi_cv``,
        ``temporal_correlation``.
    """
    with torch.no_grad():
        if spikes.ndim == 2:
            spikes = spikes.unsqueeze(1)

        T, B, N = spikes.shape
        spk = spikes.float()

        firing_rate_over_time = spk.mean(dim=(1, 2)).tolist()

        overall_mean = spk.mean().item()
        burst_threshold = max(2 * overall_mean, 0.05)
        burst_count = 0
        in_burst = False
        run_len = 0
        for rate in firing_rate_over_time:
            if rate >= burst_threshold:
                run_len += 1
                if run_len >= 3 and not in_burst:
                    burst_count += 1
                    in_burst = True
            else:
                run_len = 0
                in_burst = False

        # Inter-spike intervals (ISI) computed on first batch element
        isi_values: List[float] = []
        for n in range(N):
            spike_times = spk[:, 0, n].nonzero(as_tuple=False).squeeze(-1)
            if spike_times.numel() > 1:
                intervals = (spike_times[1:] - spike_times[:-1]).float()
                isi_values.extend(intervals.tolist())

        if isi_values:
            isi_tensor = torch.tensor(isi_values)
            isi_mean = isi_tensor.mean().item()
            isi_std = isi_tensor.std().item() if len(isi_values) > 1 else 0.0
            isi_cv = isi_std / (isi_mean + 1e-12)
        else:
            isi_mean = float("nan")
            isi_std = float("nan")
            isi_cv = float("nan")

        # Temporal autocorrelation at lag-1
        if T > 1:
            r_t = spk[:-1].mean(dim=(1, 2))
            r_t1 = spk[1:].mean(dim=(1, 2))
            mu_t = r_t.mean()
            mu_t1 = r_t1.mean()
            numerator = ((r_t - mu_t) * (r_t1 - mu_t1)).mean()
            denom = r_t.std() * r_t1.std() + 1e-12
            temporal_correlation = (numerator / denom).item()
        else:
            temporal_correlation = float("nan")

        return {
            "firing_rate_over_time": firing_rate_over_time,
            "burst_count": burst_count,
            "isi_mean": isi_mean,
            "isi_std": isi_std,
            "isi_cv": isi_cv,
            "temporal_correlation": temporal_correlation,
        }


def check_state_health(state: Any) -> Dict[str, Any]:
    """Check state tensors for NaN, Inf, and numerical range issues.

    Args:
        state: A single ``torch.Tensor`` (membrane potential), a ``tuple``
               or ``list`` of tensors, or any other object.

    Returns:
        dict with ``has_nan``, ``has_inf``, ``v_mean``, ``v_max``, ``v_min``,
        ``health_status`` (``'healthy'``, ``'warning'``, or ``'critical'``).
    """
    tensors: List[torch.Tensor] = []
    if isinstance(state, torch.Tensor):
        tensors = [state]
    elif isinstance(state, (list, tuple)):
        tensors = [t for t in state if isinstance(t, torch.Tensor)]
    else:
        return {
            "has_nan": False,
            "has_inf": False,
            "v_mean": float("nan"),
            "v_max": float("nan"),
            "v_min": float("nan"),
            "health_status": "warning",
            "note": f"Unknown state type: {type(state).__name__}",
        }

    if not tensors:
        return {
            "has_nan": False,
            "has_inf": False,
            "v_mean": float("nan"),
            "v_max": float("nan"),
            "v_min": float("nan"),
            "health_status": "warning",
            "note": "No tensors found in state",
        }

    with torch.no_grad():
        all_flat = torch.cat([t.float().reshape(-1) for t in tensors])
        has_nan = bool(torch.isnan(all_flat).any().item())
        has_inf = bool(torch.isinf(all_flat).any().item())
        v_mean = all_flat.mean().item() if not has_nan else float("nan")
        v_max = (
            all_flat.max().item()
            if not (has_nan or has_inf)
            else float("nan")
        )
        v_min = (
            all_flat.min().item()
            if not (has_nan or has_inf)
            else float("nan")
        )

    if has_nan:
        health_status = "critical"
    elif has_inf:
        health_status = "critical"
    elif not math.isnan(v_max) and abs(v_max) > 1e4:
        health_status = "critical"
    elif not math.isnan(v_max) and abs(v_max) > 1e2:
        health_status = "warning"
    else:
        health_status = "healthy"

    return {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "v_mean": v_mean,
        "v_max": v_max,
        "v_min": v_min,
        "health_status": health_status,
    }


def analyze_surrogate_gradient(
    surrogate_name: str,
    neuron_size: int,
    device: str,
) -> Dict[str, float]:
    """Probe the surrogate gradient to measure effective width and peak response.

    Sweeps a range of pre-threshold membrane values through the surrogate
    function and records the backward gradient magnitude.

    Returns:
        dict with ``effective_width`` (half-width at half-max),
        ``peak_gradient``, ``mean_gradient_near_threshold``.
    """
    try:
        x_vals = torch.linspace(
            -3.0, 3.0, 600, device=torch.device(device)
        ).requires_grad_(True)

        surrogate_fn = get_surrogate(surrogate_name)
        output = surrogate_fn(x_vals)
        output.backward(torch.ones_like(output))

        if x_vals.grad is None:
            return {
                "effective_width": float("nan"),
                "peak_gradient": float("nan"),
                "mean_gradient_near_threshold": float("nan"),
            }

        grads = x_vals.grad.detach().abs()
        peak_grad = grads.max().item()

        half_max = 0.5 * peak_grad
        above_half = grads > half_max
        if above_half.any():
            indices = above_half.nonzero(as_tuple=False).squeeze(-1)
            x_det = x_vals.detach()
            width = (x_det[indices[-1]] - x_det[indices[0]]).abs().item()
        else:
            width = float("nan")

        x_det2 = x_vals.detach()
        near_thresh_mask = x_det2.abs() < 0.5
        mean_near = (
            grads[near_thresh_mask].mean().item()
            if near_thresh_mask.any()
            else float("nan")
        )

        return {
            "effective_width": width,
            "peak_gradient": peak_grad,
            "mean_gradient_near_threshold": mean_near,
        }
    except Exception as exc:
        return {
            "effective_width": float("nan"),
            "peak_gradient": float("nan"),
            "mean_gradient_near_threshold": float("nan"),
            "error": str(exc),
        }


# ===========================================================================
# SECTION 3: Tiny test network and report dataclass
# ===========================================================================


class _DebugNetwork(nn.Module):
    """Minimal two-layer network wrapping a configurable spiking neuron.

    Input linear -> neuron_1 -> output linear -> neuron_2.
    Used solely for generating realistic spike traces and gradients.
    """

    def __init__(
        self,
        neuron_name: str,
        neuron_size: int,
        surrogate: str,
        device: str,
    ) -> None:
        super().__init__()
        self.neuron_name = neuron_name
        self.neuron_size = neuron_size

        input_size = max(neuron_size * 2, 32)
        self.input_size = input_size

        self.fc_in = nn.Linear(input_size, neuron_size)
        self.neuron1 = _make_neuron(neuron_name, neuron_size, surrogate)
        self.fc_out = nn.Linear(neuron_size, neuron_size)
        self.neuron2 = _make_neuron(neuron_name, neuron_size, surrogate)

    def reset_mem(self) -> None:
        _reset_neuron(self.neuron1)
        _reset_neuron(self.neuron2)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-timestep forward pass.

        Returns:
            spk1: spikes from layer 1 ``(B, N)``
            mem1: membrane from layer 1 ``(B, N)``
            spk2: spikes from layer 2 ``(B, N)``
            mem2: membrane from layer 2 ``(B, N)``
        """
        cur1 = self.fc_in(x)
        spk1, mem1 = _forward_neuron(self.neuron1, cur1)
        cur2 = self.fc_out(spk1)
        spk2, mem2 = _forward_neuron(self.neuron2, cur2)
        return spk1, mem1, spk2, mem2


def _make_neuron(name: str, size: int, surrogate: str) -> nn.Module:
    """Instantiate the correct neuron class by string name."""
    if name == "lif":
        return LIFNeuron(surrogate=surrogate)
    elif name == "adaptive_lif":
        return AdaptiveLIFNeuron(surrogate=surrogate)
    elif name == "recurrent_lif":
        return RecurrentLIFNeuron(size=size, surrogate=surrogate)
    elif name == "advanced_lif":
        return AdvancedLIFNeuron(size=size, surrogate=surrogate)
    else:
        raise ValueError(f"Unknown neuron type: '{name}'")


def _reset_neuron(neuron: nn.Module) -> None:
    """Call reset_mem() on a neuron if the method exists."""
    if hasattr(neuron, "reset_mem"):
        neuron.reset_mem()


def _forward_neuron(
    neuron: nn.Module, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward through a neuron, handling both legacy and explicit-state APIs.

    Returns:
        ``(spikes, membrane)`` pair.  If the neuron returns only a single
        tensor, membrane is set to zeros of the same shape.
    """
    out = neuron(x)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        return out[0], out[1]
    # Single-return path -- treat as spikes only
    return out, torch.zeros_like(out)


# ---------------------------------------------------------------------------
# DebugReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class DebugReport:
    """Complete diagnostic snapshot for one neuron type / surrogate combination."""

    neuron_type: str
    surrogate: str
    device: str
    timesteps: int
    batch_size: int
    neuron_size: int
    firing_rates: Dict[str, Any]
    membrane_stats: Dict[str, float]
    sparsity: Dict[str, Any]
    surrogate_analysis: Dict[str, float] = field(default_factory=dict)
    gradient_stats: Optional[Dict[str, Dict[str, Any]]] = None
    temporal_dynamics: Optional[Dict[str, Any]] = None
    state_health: Optional[Dict[str, Any]] = None
    brain_ai_available: bool = False
    error: Optional[str] = None

    # ------------------------------------------------------------------
    def _overall_health(self) -> Tuple[str, str]:
        """Return ``(label, color_key)`` for the overall health assessment.

        Color keys: ``'green'``, ``'yellow'``, ``'red'``.
        """
        issues: List[str] = []

        mean_fr = self.firing_rates.get("mean", float("nan"))
        if not math.isnan(mean_fr):
            if mean_fr < 0.001:
                issues.append("firing rate extremely low")
            elif mean_fr > 0.95:
                issues.append("firing rate extremely high")

        dead_frac = self.sparsity.get("dead_neuron_fraction", 0.0)
        sat_frac = self.sparsity.get("saturated_neuron_fraction", 0.0)
        if dead_frac > 0.5:
            issues.append(f"majority dead neurons ({dead_frac:.1%})")
        if sat_frac > 0.5:
            issues.append(f"majority saturated neurons ({sat_frac:.1%})")

        if self.state_health:
            sh = self.state_health.get("health_status", "healthy")
            if sh == "critical":
                issues.append("state health CRITICAL")
            elif sh == "warning":
                issues.append("state health WARNING")

        if self.gradient_stats:
            for pname, ginfo in self.gradient_stats.items():
                if ginfo.get("has_nan"):
                    issues.append(f"NaN gradient in {pname}")
                    break
                if ginfo.get("has_inf"):
                    issues.append(f"Inf gradient in {pname}")
                    break

        mem_max = self.membrane_stats.get("max", float("nan"))
        if not math.isnan(mem_max) and abs(mem_max) > 1e3:
            issues.append(f"membrane exploding (max={mem_max:.1f})")

        if not issues:
            return "HEALTHY", "green"
        elif len(issues) == 1:
            return "WARNING", "yellow"
        else:
            return "CRITICAL", "red"

    # ------------------------------------------------------------------
    def format_text(self, verbose: bool = False) -> str:
        """Return the report as a human-readable text string."""
        lines: List[str] = []
        W = 63

        def div(char: str = "=") -> str:
            return char * W

        surrogate_display = {
            "atan": "ATan",
            "fast_sigmoid": "FastSigmoid",
            "straight_through": "StraightThrough",
        }.get(self.surrogate, self.surrogate)

        neuron_display = {
            "lif": "LIFNeuron",
            "adaptive_lif": "AdaptiveLIFNeuron",
            "recurrent_lif": "RecurrentLIFNeuron",
            "advanced_lif": "AdvancedLIFNeuron",
        }.get(self.neuron_type, self.neuron_type)

        lines.append(div("="))
        lines.append(
            f"Spiking Debug Report -- {neuron_display}"
            f" ({surrogate_display} surrogate)"
        )
        lines.append(
            f"Device: {self.device}  |  T={self.timesteps}  |  "
            f"B={self.batch_size}  |  N={self.neuron_size}"
        )
        if not self.brain_ai_available:
            lines.append(
                "[inline fallback neurons -- brain_ai not on sys.path]"
            )
        lines.append(div("="))

        if self.error:
            lines.append("")
            lines.append(f"ERROR: {self.error}")
            return "\n".join(lines)

        # -- Firing Rates ------------------------------------------------
        lines.append("")
        lines.append(_bold("FIRING RATES") if _USE_COLOR else "FIRING RATES")

        fr = self.firing_rates
        mean_v = fr.get("mean", float("nan"))
        min_v = fr.get("min", float("nan"))
        max_v = fr.get("max", float("nan"))
        std_v = fr.get("std", float("nan"))

        lines.append(f"  Mean:  {mean_v:.4f}  {_ascii_bar(mean_v)}")
        lines.append(f"  Min:   {min_v:.4f}  {_ascii_bar(min_v)}")
        lines.append(f"  Max:   {max_v:.4f}  {_ascii_bar(max_v)}")
        lines.append(f"  Std:   {std_v:.4f}")

        fr_health, fr_color = _rate_health(mean_v)
        lines.append(
            f"  Status: {_colorize(fr_health, fr_color)}"
            f"  (target range: 0.01-0.50)"
        )

        if verbose and "per_neuron_rates" in fr:
            pnr = fr["per_neuron_rates"]
            lines.append(f"  Per-neuron rates (N={len(pnr)}):")
            for i, r in enumerate(pnr):
                lines.append(f"    [{i:>3d}] {r:.4f}  {_ascii_bar(r)}")

        # -- Membrane Potential ------------------------------------------
        lines.append("")
        lines.append(
            _bold("MEMBRANE POTENTIAL")
            if _USE_COLOR
            else "MEMBRANE POTENTIAL"
        )
        ms = self.membrane_stats
        lines.append(
            f"  Mean: {ms.get('mean', float('nan')):.4f}  |  "
            f"Var: {ms.get('var', float('nan')):.4f}  |  "
            f"Max: {ms.get('max', float('nan')):.4f}  |  "
            f"Min: {ms.get('min', float('nan')):.4f}"
        )
        lines.append(
            f"  Skewness: {ms.get('skewness', float('nan')):.4f}  |  "
            f"Near threshold (>0.8*v_th): "
            f"{ms.get('fraction_near_threshold', float('nan')):.1%}  |  "
            f"Near rest (<|0.05|): "
            f"{ms.get('fraction_near_rest', float('nan')):.1%}"
        )
        mem_health, mem_color = _membrane_health(ms)
        lines.append(f"  Status: {_colorize(mem_health, mem_color)}")

        # -- Spike Sparsity ----------------------------------------------
        lines.append("")
        lines.append(
            _bold("SPIKE SPARSITY") if _USE_COLOR else "SPIKE SPARSITY"
        )
        sp = self.sparsity
        overall = sp.get("overall_sparsity", float("nan"))
        dead_frac = sp.get("dead_neuron_fraction", float("nan"))
        sat_frac = sp.get("saturated_neuron_fraction", float("nan"))
        dead_count = sp.get("dead_count", "?")
        sat_count = sp.get("saturated_count", "?")
        total_n = sp.get("total_neurons", "?")

        lines.append(f"  Overall: {overall:.1%} zeros")
        lines.append(
            f"  Dead neurons (0% firing):        "
            f"{dead_count}/{total_n} ({dead_frac:.1%})"
        )
        lines.append(
            f"  Saturated neurons (>90% firing): "
            f"{sat_count}/{total_n} ({sat_frac:.1%})"
        )
        sp_health, sp_color = _sparsity_health(sp)
        lines.append(f"  Status: {_colorize(sp_health, sp_color)}")

        # -- Surrogate Gradient Analysis ---------------------------------
        lines.append("")
        lines.append(
            _bold("SURROGATE GRADIENT ANALYSIS")
            if _USE_COLOR
            else "SURROGATE GRADIENT ANALYSIS"
        )
        sa = self.surrogate_analysis
        lines.append(
            f"  Peak gradient:          "
            f"{sa.get('peak_gradient', float('nan')):.6f}"
        )
        lines.append(
            f"  Effective width (HWHM): "
            f"{sa.get('effective_width', float('nan')):.4f}"
        )
        lines.append(
            f"  Mean near threshold:    "
            f"{sa.get('mean_gradient_near_threshold', float('nan')):.6f}"
        )
        if "error" in sa:
            lines.append(f"  Note: {sa['error']}")

        # -- State Health Check ------------------------------------------
        if self.state_health is not None:
            lines.append("")
            lines.append(
                _bold("STATE HEALTH CHECK")
                if _USE_COLOR
                else "STATE HEALTH CHECK"
            )
            sh = self.state_health
            nan_str = (
                _colorize("YES", "red")
                if sh.get("has_nan")
                else _colorize("no", "green")
            )
            inf_str = (
                _colorize("YES", "red")
                if sh.get("has_inf")
                else _colorize("no", "green")
            )
            lines.append(
                f"  NaN detected: {nan_str}  |  Inf detected: {inf_str}"
            )
            lines.append(
                f"  v_mean: {sh.get('v_mean', float('nan')):.4f}  |  "
                f"v_max: {sh.get('v_max', float('nan')):.4f}  |  "
                f"v_min: {sh.get('v_min', float('nan')):.4f}"
            )
            sh_status = sh.get("health_status", "unknown")
            sh_color = (
                "green"
                if sh_status == "healthy"
                else "yellow"
                if sh_status == "warning"
                else "red"
            )
            lines.append(
                f"  Status: {_colorize(sh_status.upper(), sh_color)}"
            )
            if "note" in sh:
                lines.append(f"  Note: {sh['note']}")

        # -- Gradient Statistics (optional) ------------------------------
        if self.gradient_stats is not None:
            lines.append("")
            lines.append(
                _bold("GRADIENT STATISTICS")
                if _USE_COLOR
                else "GRADIENT STATISTICS"
            )
            all_finite = True
            all_have_grad = True
            for pname, ginfo in self.gradient_stats.items():
                has_nan = ginfo.get("has_nan", False)
                has_inf = ginfo.get("has_inf", False)
                norm_v = ginfo.get("norm", float("nan"))
                mean_g = ginfo.get("mean", float("nan"))
                max_g = ginfo.get("max", float("nan"))
                learnable_mark = (
                    "[learnable]" if ginfo.get("learnable") else "[frozen]"
                )

                if has_nan or has_inf:
                    all_finite = False
                if math.isnan(norm_v):
                    all_have_grad = False

                flag = ""
                if has_nan:
                    flag = _colorize("  [NaN!]", "red")
                elif has_inf:
                    flag = _colorize("  [Inf!]", "red")
                elif math.isnan(norm_v):
                    flag = _colorize("  [no grad]", "yellow")

                lines.append(
                    f"  {pname:<38s}  norm={norm_v:>9.6f}  "
                    f"mean={mean_g:>9.6f}  max={max_g:>9.6f}  "
                    f"{learnable_mark}{flag}"
                )

            if all_finite and all_have_grad:
                grad_health = "HEALTHY (all finite, gradients present)"
                grad_color = "green"
            elif not all_finite:
                grad_health = "WARNING (non-finite gradients detected)"
                grad_color = "red"
            else:
                grad_health = "WARNING (some parameters received no gradient)"
                grad_color = "yellow"
            lines.append(f"  Status: {_colorize(grad_health, grad_color)}")

        # -- Temporal Dynamics -------------------------------------------
        if self.temporal_dynamics is not None:
            lines.append("")
            lines.append(
                _bold("TEMPORAL DYNAMICS")
                if _USE_COLOR
                else "TEMPORAL DYNAMICS"
            )
            td = self.temporal_dynamics
            lines.append(f"  Burst count:               {td.get('burst_count', 0)}")
            isi_m = td.get("isi_mean", float("nan"))
            isi_s = td.get("isi_std", float("nan"))
            isi_cv = td.get("isi_cv", float("nan"))
            lines.append(
                f"  ISI mean (timesteps):      {isi_m:.2f}  |  "
                f"ISI std: {isi_s:.2f}  |  CV: {isi_cv:.3f}"
            )
            lines.append(
                f"  Temporal autocorr (lag-1): "
                f"{td.get('temporal_correlation', float('nan')):.4f}"
            )

            if verbose:
                frot = td.get("firing_rate_over_time", [])
                if frot:
                    lines.append("  Firing rate over time:")
                    for t_idx, rate in enumerate(frot):
                        lines.append(
                            f"    t={t_idx:>3d}  {rate:.4f}  {_ascii_bar(rate)}"
                        )

        # -- Overall Assessment ------------------------------------------
        lines.append("")
        lines.append("-" * W)
        overall_label, overall_color = self._overall_health()
        lines.append(
            f"Overall Assessment:  {_colorize(overall_label, overall_color)}"
        )
        lines.append("=" * W)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "neuron_type": self.neuron_type,
            "surrogate": self.surrogate,
            "device": self.device,
            "timesteps": self.timesteps,
            "batch_size": self.batch_size,
            "neuron_size": self.neuron_size,
            "brain_ai_available": self.brain_ai_available,
            "error": self.error,
            "firing_rates": _sanitize_floats(self.firing_rates),
            "membrane_stats": _sanitize_floats(self.membrane_stats),
            "sparsity": _sanitize_floats(self.sparsity),
            "surrogate_analysis": _sanitize_floats(self.surrogate_analysis),
            "gradient_stats": (
                {k: _sanitize_floats(v) for k, v in self.gradient_stats.items()}
                if self.gradient_stats is not None
                else None
            ),
            "temporal_dynamics": (
                _sanitize_floats(self.temporal_dynamics)
                if self.temporal_dynamics is not None
                else None
            ),
            "state_health": (
                _sanitize_floats(self.state_health)
                if self.state_health is not None
                else None
            ),
        }


# ===========================================================================
# SECTION 4: Report generation
# ===========================================================================


def generate_report(
    neuron_name: str,
    surrogate: str,
    device: str,
    timesteps: int,
    batch_size: int,
    neuron_size: int,
    check_gradients: bool = False,
    verbose: bool = False,
) -> DebugReport:
    """Generate a complete debug report for a neuron configuration.

    Steps:

    1. Create a ``_DebugNetwork`` (two-layer test network).
    2. Forward for ``timesteps`` steps, recording spikes and membrane traces.
    3. If ``check_gradients``: run backward and collect gradient stats.
    4. Collect all metrics and return a ``DebugReport``.

    Args:
        neuron_name: One of ``'lif'``, ``'adaptive_lif'``,
                     ``'recurrent_lif'``, ``'advanced_lif'``.
        surrogate: One of ``'atan'``, ``'fast_sigmoid'``,
                   ``'straight_through'``.
        device: PyTorch device string, e.g. ``'cpu'`` or ``'cuda'``.
        timesteps: Number of simulation timesteps ``T``.
        batch_size: Batch size ``B``.
        neuron_size: Number of neurons per layer ``N``.
        check_gradients: If ``True``, run backward and include gradient stats.
        verbose: Unused in generation; forwarded to format output only.

    Returns:
        ``DebugReport`` with all fields populated.
    """
    try:
        dev = torch.device(device)

        net = _DebugNetwork(
            neuron_name=neuron_name,
            neuron_size=neuron_size,
            surrogate=surrogate,
            device=device,
        ).to(dev)

        input_size = net.input_size

        # Surrogate gradient analysis runs before the main forward loop
        # to avoid any state contamination from the network.
        surrogate_analysis = analyze_surrogate_gradient(
            surrogate, neuron_size, device
        )

        spike_traces_l1: List[torch.Tensor] = []
        spike_traces_l2: List[torch.Tensor] = []
        membrane_traces_l1: List[torch.Tensor] = []
        membrane_traces_l2: List[torch.Tensor] = []
        final_state: Optional[Any] = None
        gradient_stats: Optional[Dict[str, Dict[str, Any]]] = None

        net.reset_mem()

        if check_gradients:
            net.train()
            spike_traces_grad: List[torch.Tensor] = []

            for _t in range(timesteps):
                x = torch.randn(batch_size, input_size, device=dev)
                spk1, mem1, spk2, mem2 = net(x)
                spike_traces_grad.append(spk1)
                spike_traces_l1.append(spk1.detach())
                spike_traces_l2.append(spk2.detach())
                membrane_traces_l1.append(mem1.detach())
                membrane_traces_l2.append(mem2.detach())

            # Differentiable loss via surrogate spike counts
            loss = torch.stack(spike_traces_grad, dim=0).mean()
            loss.backward()
            gradient_stats = collect_gradient_stats(net)
            final_state = mem2.detach()
        else:
            net.train(False)  # inference mode
            with torch.no_grad():
                for _t in range(timesteps):
                    x = torch.randn(batch_size, input_size, device=dev)
                    spk1, mem1, spk2, mem2 = net(x)
                    spike_traces_l1.append(spk1)
                    spike_traces_l2.append(spk2)
                    membrane_traces_l1.append(mem1)
                    membrane_traces_l2.append(mem2)
            final_state = mem2

        # Stack to (T, B, N)
        spikes_l1 = torch.stack(spike_traces_l1, dim=0)

        firing_rates = collect_firing_rates(spikes_l1)
        membrane_stats = collect_membrane_stats(membrane_traces_l1)
        sparsity = collect_sparsity_analysis(spikes_l1)
        temporal_dynamics = collect_temporal_dynamics(spikes_l1)
        state_health = check_state_health(final_state)

        return DebugReport(
            neuron_type=neuron_name,
            surrogate=surrogate,
            device=device,
            timesteps=timesteps,
            batch_size=batch_size,
            neuron_size=neuron_size,
            firing_rates=firing_rates,
            membrane_stats=membrane_stats,
            sparsity=sparsity,
            surrogate_analysis=surrogate_analysis,
            gradient_stats=gradient_stats,
            temporal_dynamics=temporal_dynamics,
            state_health=state_health,
            brain_ai_available=_BRAIN_AI_AVAILABLE,
        )

    except Exception as exc:
        tb = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        return DebugReport(
            neuron_type=neuron_name,
            surrogate=surrogate,
            device=device,
            timesteps=timesteps,
            batch_size=batch_size,
            neuron_size=neuron_size,
            firing_rates={},
            membrane_stats={},
            sparsity={},
            surrogate_analysis={},
            brain_ai_available=_BRAIN_AI_AVAILABLE,
            error=tb,
        )


# ===========================================================================
# SECTION 5: Formatted output helpers
# ===========================================================================

_BAR_WIDTH = 20


def _ascii_bar(
    value: float, max_val: float = 1.0, width: int = _BAR_WIDTH
) -> str:
    """Render a scalar as an ASCII progress bar.

    Example (value=0.35, width=20):  ``'\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2591\u2591\u2591...'``
    """
    if math.isnan(value) or math.isinf(value) or max_val <= 0:
        return "?" * width
    ratio = min(max(value / max_val, 0.0), 1.0)
    filled = round(ratio * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def _colorize(text: str, color: str) -> str:
    """Apply a named terminal color to ``text``."""
    if not _USE_COLOR:
        return text
    mapping: Dict[str, Any] = {
        "green": _green,
        "yellow": _yellow,
        "red": _red,
        "cyan": _cyan,
    }
    fn = mapping.get(color, lambda t: t)
    return fn(text)


def _rate_health(mean_rate: float) -> Tuple[str, str]:
    """Return ``(label, color)`` for a given mean firing rate."""
    if math.isnan(mean_rate):
        return "UNKNOWN", "yellow"
    if mean_rate < 0.001:
        return "CRITICAL (dead network)", "red"
    if mean_rate > 0.95:
        return "CRITICAL (saturated network)", "red"
    if mean_rate < 0.01 or mean_rate > 0.70:
        return "WARNING (outside 0.01-0.70 range)", "yellow"
    return "HEALTHY", "green"


def _membrane_health(ms: Dict[str, float]) -> Tuple[str, str]:
    """Return ``(label, color)`` for membrane potential stats."""
    max_v = ms.get("max", float("nan"))
    has_nan_mem = math.isnan(ms.get("mean", 0.0))
    if has_nan_mem:
        return "CRITICAL (NaN in membrane)", "red"
    if not math.isnan(max_v) and abs(max_v) > 1e4:
        return "CRITICAL (exploding membrane)", "red"
    if not math.isnan(max_v) and abs(max_v) > 1e2:
        return "WARNING (large membrane values)", "yellow"
    return "HEALTHY", "green"


def _sparsity_health(sp: Dict[str, Any]) -> Tuple[str, str]:
    """Return ``(label, color)`` for sparsity analysis."""
    dead = sp.get("dead_neuron_fraction", float("nan"))
    sat = sp.get("saturated_neuron_fraction", float("nan"))
    if math.isnan(dead):
        return "UNKNOWN", "yellow"
    if dead > 0.5:
        return f"CRITICAL (>50% dead neurons: {dead:.1%})", "red"
    if sat > 0.5:
        return f"CRITICAL (>50% saturated neurons: {sat:.1%})", "red"
    if dead > 0.1:
        return f"WARNING ({dead:.1%} dead neurons)", "yellow"
    if sat > 0.1:
        return f"WARNING ({sat:.1%} saturated neurons)", "yellow"
    return "HEALTHY", "green"


def _sanitize_floats(obj: Any) -> Any:
    """Recursively replace non-finite floats with string sentinels for JSON."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


def print_report(report: DebugReport, verbose: bool = False) -> None:
    """Print a formatted debug report to stdout."""
    print(report.format_text(verbose=verbose))


# ===========================================================================
# SECTION 6: CLI entry point
# ===========================================================================

_NEURON_CHOICES = ["lif", "adaptive_lif", "recurrent_lif", "advanced_lif"]
_SURROGATE_CHOICES = ["atan", "fast_sigmoid", "straight_through"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a comprehensive debug report for spiking neural "
            "network layers."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--neuron",
        choices=_NEURON_CHOICES,
        default=None,
        help="Neuron type to report on (default: all).",
    )
    parser.add_argument(
        "--surrogate",
        choices=_SURROGATE_CHOICES,
        default="atan",
        help="Surrogate gradient function (default: atan).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device string (default: cpu).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=25,
        help="Number of simulation timesteps T (default: 25).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size B (default: 4).",
    )
    parser.add_argument(
        "--neuron-size",
        type=int,
        default=64,
        help="Number of neurons per layer N (default: 64).",
    )
    parser.add_argument(
        "--check-gradients",
        action="store_true",
        help="Run backward pass and include gradient statistics.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output to stdout instead of human-readable text.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include per-neuron firing rates and full temporal trace.",
    )
    return parser


def main() -> int:
    """CLI entry point.  Returns exit code (0 = healthy/warning, 1 = critical)."""
    parser = _build_parser()
    args = parser.parse_args()

    neuron_types = [args.neuron] if args.neuron else _NEURON_CHOICES

    # Validate device early
    try:
        dev = torch.device(args.device)
        torch.zeros(1, device=dev)
    except Exception as exc:
        print(
            f"ERROR: Cannot use device '{args.device}': {exc}",
            file=sys.stderr,
        )
        return 1

    if not args.json:
        print()
        print(
            _bold("SNN Spiking Debug Report")
            if _USE_COLOR
            else "SNN Spiking Debug Report"
        )
        print(f"  brain_ai available : {_BRAIN_AI_AVAILABLE}")
        if not _BRAIN_AI_AVAILABLE and _IMPORT_ERROR:
            short_err = _IMPORT_ERROR.strip().split("\n")[-1]
            note_line = f"  import note        : {short_err}"
            print(_dim(note_line) if _USE_COLOR else note_line)
        print(f"  repo root          : {_REPO_ROOT}")
        print(f"  surrogate          : {args.surrogate}")
        print(f"  device             : {args.device}")
        print(
            f"  T / B / N          : "
            f"{args.timesteps} / {args.batch_size} / {args.neuron_size}"
        )
        print(f"  check_gradients    : {args.check_gradients}")
        print(f"  neuron types       : {', '.join(neuron_types)}")
        print()

    reports: List[DebugReport] = []
    for nt in neuron_types:
        report = generate_report(
            neuron_name=nt,
            surrogate=args.surrogate,
            device=args.device,
            timesteps=args.timesteps,
            batch_size=args.batch_size,
            neuron_size=args.neuron_size,
            check_gradients=args.check_gradients,
            verbose=args.verbose,
        )
        reports.append(report)

    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        for report in reports:
            print_report(report, verbose=args.verbose)
            print()

    # Exit code: 1 if any report is critical or errored
    for r in reports:
        if r.error:
            return 1
        label, _ = r._overall_health()
        if label == "CRITICAL":
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
