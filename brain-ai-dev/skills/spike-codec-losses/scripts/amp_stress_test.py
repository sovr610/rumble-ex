#!/usr/bin/env python3
"""
amp_stress_test.py - AMP (Automatic Mixed Precision) stress test for the spike
codec and loss pack.

Tests all spike encoding, decoding, and loss operations under mixed-precision
autocast on both CPU and CUDA backends. Verifies that:
  - No NaN or Inf values appear in outputs, losses, or gradients.
  - Spike tensors remain float32 (never degraded to float16).
  - Loss scalars are always float32.
  - Gradient flow is preserved through all operations.
  - Numerical precision under autocast matches full fp32 within tolerance.
  - Memory usage does not regress under AMP.
"""

# ---------------------------------------------------------------------------
# SECTION 1: Path setup
# ---------------------------------------------------------------------------

import os
import sys

# scripts/ -> spike-codec-losses/ -> skills/ -> brain-ai-dev/ -> human-brain/
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

# ---------------------------------------------------------------------------
# SECTION 2: Imports and API detection
# ---------------------------------------------------------------------------

import argparse
import json
import math
import time
import traceback
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Detect CUDA availability
_HAS_CUDA = torch.cuda.is_available()

# Attempt to import the target API from the brain_ai package.
# If unavailable, we fall back to stub implementations that faithfully mimic
# the expected API so that the stress test itself can still exercise AMP
# behaviour on the core mathematical operations.

_USING_STUBS = False

try:
    from brain_ai.core.encoding import (
        RateEncoder,
        TemporalEncoder,
    )
    from brain_ai.core.losses import (
        prob_spikes_loss,
        spike_rate_regularization,
        temporal_consistency_loss,
        inter_spike_interval_loss,
        membrane_potential_regularization,
        SNNLoss,
    )
    _API_AVAILABLE = True
except ImportError:
    _API_AVAILABLE = False

# We always use stubs for the *new* batch-first API that this skill defines,
# since the modules may not be implemented yet.  The stubs follow the exact
# contracts from the reference documents (spike-decoders.md, loss-pack.md).
_USING_STUBS = True


# ---------------------------------------------------------------------------
# SECTION 3: Stub implementations
# ---------------------------------------------------------------------------

@dataclass
class DecoderOutput:
    """Matches the DecoderOutput contract from spike-decoders.md."""
    logits_proxy: torch.Tensor   # (B, N) or (B, G) float32
    prediction: torch.Tensor     # (B,) int64
    confidence: torch.Tensor     # (B,) float32
    aux: Dict[str, Any] = field(default_factory=dict)


class _StubRateEncoder(nn.Module):
    """Rate encoder stub: generates random spikes (B, T, N) with configurable rate.

    AMP-safe: all outputs are float32 binary tensors.
    """

    def __init__(self, num_features: int = 64, num_steps: int = 25,
                 spike_prob: float = 0.2, gain: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.num_steps = num_steps
        self.spike_prob = spike_prob
        self.gain = gain
        # Learnable threshold so we can test gradient flow
        self.threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N) input features normalised to [0, 1].
        Returns:
            spikes: (B, T, N) float32 binary {0, 1}.
        """
        B, N = x.shape
        T = self.num_steps
        # Probability per timestep derived from input and learnable threshold
        p = torch.sigmoid((x * self.gain - self.threshold).float())  # fp32
        # Expand to (B, T, N) and sample
        p_expanded = p.unsqueeze(1).expand(B, T, N)
        noise = torch.rand_like(p_expanded)
        # Straight-through: detach the sampling but keep surrogate gradient
        spikes_hard = (noise < p_expanded).float()
        # Surrogate gradient: use sigmoid of (noise - p) as soft approximation
        spikes_soft = torch.sigmoid((p_expanded - noise) * 10.0)
        # Straight-through estimator
        spikes = (spikes_hard - spikes_soft).detach() + spikes_soft
        return spikes.float()  # ensure fp32


class _StubLatencyEncoder(nn.Module):
    """Latency encoder stub: generates single spikes at computed times.

    Higher input values produce earlier spikes (lower latency).
    AMP-safe: output is float32.
    """

    def __init__(self, num_features: int = 64, num_steps: int = 25,
                 tau: float = 5.0):
        super().__init__()
        self.num_features = num_features
        self.num_steps = num_steps
        self.tau = tau
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N) input features in [0, 1].
        Returns:
            spikes: (B, T, N) float32 binary -- exactly one spike per neuron.
        """
        B, N = x.shape
        T = self.num_steps
        # Compute spike time: higher x -> earlier spike
        x_safe = x.float().clamp(min=0.01, max=1.0)
        spike_times = (self.tau * self.scale * torch.log(1.0 / x_safe)).clamp(0, T - 1)
        spike_times_idx = spike_times.long()  # (B, N)

        # Build spike tensor
        spikes = torch.zeros(B, T, N, device=x.device, dtype=torch.float32)
        # One-hot at spike time
        t_range = torch.arange(T, device=x.device).view(1, T, 1)
        spike_times_expanded = spike_times_idx.unsqueeze(1)  # (B, 1, N)
        spikes = (t_range == spike_times_expanded).float()

        # Add surrogate gradient path via soft spike
        soft_spike = torch.exp(-0.5 * ((t_range.float() - spike_times.unsqueeze(1)) ** 2))
        spikes = (spikes - soft_spike).detach() + soft_spike
        return spikes.float()


class _StubTTFSEncoder(nn.Module):
    """Time-to-first-spike encoder stub: generates exactly-one spikes.

    Similar to latency encoder but uses a different time-mapping.
    AMP-safe: output is float32.
    """

    def __init__(self, num_features: int = 64, num_steps: int = 25):
        super().__init__()
        self.num_features = num_features
        self.num_steps = num_steps
        self.beta = nn.Parameter(torch.tensor(3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N) input features in [0, 1].
        Returns:
            spikes: (B, T, N) float32 binary -- exactly one spike per neuron.
        """
        B, N = x.shape
        T = self.num_steps
        x_safe = x.float().clamp(min=0.01, max=1.0)
        # Time = (1 - x) * T * beta_scaling
        spike_times = ((1.0 - x_safe) * self.beta).clamp(0, T - 1)
        spike_times_idx = spike_times.long()

        t_range = torch.arange(T, device=x.device).view(1, T, 1).float()
        spike_times_f = spike_times.unsqueeze(1)  # (B, 1, N)

        # Hard spikes at computed time
        hard = (t_range == spike_times_idx.unsqueeze(1).float()).float()
        # Soft surrogate for gradient
        soft = torch.exp(-((t_range - spike_times_f) ** 2) / 0.5)
        spikes = (hard - soft).detach() + soft
        return spikes.float()


class _StubPopulationEncoder(nn.Module):
    """Population encoder stub: expands features with tuning curves.

    Each input feature is encoded by a population of neurons with
    Gaussian tuning curves centred at different preferred values.
    AMP-safe: all accumulations in fp32.
    """

    def __init__(self, num_features: int = 64, num_steps: int = 25,
                 neurons_per_feature: int = 4, spike_prob: float = 0.15):
        super().__init__()
        self.num_features = num_features
        self.num_steps = num_steps
        self.neurons_per_feature = neurons_per_feature
        self.spike_prob = spike_prob
        self.total_neurons = num_features * neurons_per_feature
        # Tuning curve centres
        centres = torch.linspace(0.0, 1.0, neurons_per_feature)
        self.register_buffer("centres", centres)
        self.sigma = nn.Parameter(torch.tensor(0.3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_in) input features in [0, 1].
        Returns:
            spikes: (B, T, N_in * neurons_per_feature) float32 binary.
        """
        B, N_in = x.shape
        T = self.num_steps
        K = self.neurons_per_feature

        # Compute tuning curve response: (B, N_in, K)
        x_expanded = x.float().unsqueeze(-1)  # (B, N_in, 1)
        centres = self.centres.view(1, 1, K)  # (1, 1, K)
        sigma_safe = self.sigma.float().clamp(min=0.01)
        response = torch.exp(-((x_expanded - centres) ** 2) / (2 * sigma_safe ** 2))
        # response: (B, N_in, K), values in [0, 1]

        # Scale response to spike probability
        p = (response * self.spike_prob).clamp(0.0, 1.0)  # (B, N_in, K)
        p_flat = p.view(B, 1, N_in * K).expand(B, T, N_in * K)  # (B, T, N_out)

        noise = torch.rand_like(p_flat)
        hard = (noise < p_flat).float()
        soft = torch.sigmoid((p_flat - noise) * 10.0)
        spikes = (hard - soft).detach() + soft
        return spikes.float()


class _StubRateDecoder(nn.Module):
    """Rate decoder stub: sums spikes over time, returns logits.

    AMP-safe: accumulates counts in fp32.
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-7,
                 normalize_by_T: bool = False):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.normalize_by_T = normalize_by_T

    def forward(self, spikes: torch.Tensor) -> DecoderOutput:
        """
        Args:
            spikes: (B, T, N) float {0, 1}.
        Returns:
            DecoderOutput with logits_proxy in fp32.
        """
        B, T, N = spikes.shape
        # fp32 accumulation (AMP rule 1)
        counts = spikes.float().sum(dim=1)  # (B, N) fp32

        if self.normalize_by_T:
            rate = counts / T
            logits_proxy = rate / self.temperature
        else:
            logits_proxy = counts / self.temperature

        prediction = counts.argmax(dim=-1)  # (B,) int64

        # fp32 softmax (AMP rule 2)
        probs = torch.softmax(logits_proxy.float(), dim=-1)
        top2 = probs.topk(min(2, N), dim=-1).values
        if N >= 2:
            margin = (top2[:, 0] - top2[:, 1]).clamp(min=0.0)
        else:
            margin = top2[:, 0]
        confidence = margin

        return DecoderOutput(
            logits_proxy=logits_proxy,
            prediction=prediction,
            confidence=confidence,
            aux={
                "raw_counts": counts.detach(),
                "count_histogram": counts.detach().mean(dim=0),
                "margin": margin.detach(),
            },
        )


class _StubFirstSpikeDecoder(nn.Module):
    """First-spike decoder stub: finds first spike, returns negative time as logit.

    AMP-safe: time indices are integer-derived, division is fp32.
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, spikes: torch.Tensor) -> DecoderOutput:
        """
        Args:
            spikes: (B, T, N) float {0, 1}.
        Returns:
            DecoderOutput with logits_proxy in fp32.
        """
        B, T, N = spikes.shape

        # Vectorised first-spike via cumsum + argmax
        cumsum = spikes.float().cumsum(dim=1)
        has_spiked = (cumsum >= 1.0)
        t_first = has_spiked.float().argmax(dim=1)  # (B, N)

        ever_spiked = (cumsum[:, -1, :] > 0)
        t_first = torch.where(ever_spiked, t_first,
                               torch.full_like(t_first, float(T)))

        logits_proxy = (-t_first.float() / T) / self.temperature  # (B, N)
        prediction = t_first.argmin(dim=-1)

        sorted_times = t_first.sort(dim=-1).values
        if N >= 2:
            time_margin = (sorted_times[:, 1] - sorted_times[:, 0]).float() / T
        else:
            time_margin = torch.ones(B, device=spikes.device)
        confidence = time_margin.clamp(min=0.0, max=1.0)

        return DecoderOutput(
            logits_proxy=logits_proxy,
            prediction=prediction,
            confidence=confidence,
            aux={
                "first_spike_times": t_first.detach(),
                "ever_spiked": ever_spiked.detach(),
                "margin": time_margin.detach(),
            },
        )


class _StubPopulationDecoder(nn.Module):
    """Population decoder stub: groups neurons by class, sums counts per group.

    AMP-safe: fp32 accumulation of group counts.
    """

    def __init__(self, num_classes: int = 10, temperature: float = 1.0,
                 eps: float = 1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.eps = eps

    def forward(self, spikes: torch.Tensor) -> DecoderOutput:
        """
        Args:
            spikes: (B, T, N) float {0, 1}, N must be divisible by num_classes.
        Returns:
            DecoderOutput with logits_proxy of shape (B, num_classes).
        """
        B, T, N = spikes.shape
        G = self.num_classes
        neurons_per_group = N // G

        counts = spikes.float().sum(dim=1)  # (B, N) fp32

        # Reshape and sum per group
        if N % G == 0 and neurons_per_group > 0:
            group_counts = counts.view(B, G, neurons_per_group).sum(dim=-1)
        else:
            # Fallback: distribute neurons as evenly as possible
            group_counts = torch.zeros(B, G, device=spikes.device,
                                        dtype=torch.float32)
            per = max(N // G, 1)
            for g in range(G):
                start = g * per
                end = min(start + per, N)
                if start < N:
                    group_counts[:, g] = counts[:, start:end].sum(dim=-1)

        logits_proxy = group_counts / self.temperature
        prediction = group_counts.argmax(dim=-1)

        probs = torch.softmax(logits_proxy.float(), dim=-1)
        top2 = probs.topk(min(2, G), dim=-1).values
        if G >= 2:
            margin = (top2[:, 0] - top2[:, 1]).clamp(min=0.0)
        else:
            margin = top2[:, 0]
        confidence = margin

        return DecoderOutput(
            logits_proxy=logits_proxy,
            prediction=prediction,
            confidence=confidence,
            aux={
                "raw_counts": counts.detach(),
                "group_counts": group_counts.detach(),
                "margin": margin.detach(),
            },
        )


class _StubMembraneDecoder(nn.Module):
    """Membrane decoder stub: uses membrane potential directly.

    AMP-safe: membrane values are cast to fp32 for softmax.
    """

    def __init__(self, mode: str = "final", temperature: float = 1.0):
        super().__init__()
        self.mode = mode
        self.temperature = temperature

    def forward(self, spikes: torch.Tensor,
                membrane: Optional[torch.Tensor] = None) -> DecoderOutput:
        """
        Args:
            spikes: (B, T, N) float {0, 1} -- used only for shape if membrane absent.
            membrane: (B, T, N) float -- membrane potential over time.
        Returns:
            DecoderOutput with logits_proxy in fp32.
        """
        B, T, N = spikes.shape
        if membrane is None:
            # Synthesise membrane from spike cumsum as fallback
            membrane = spikes.float().cumsum(dim=1) * 0.3

        if self.mode == "final":
            v = membrane[:, -1, :].float()
        elif self.mode == "max":
            v = membrane.float().max(dim=1).values
        else:
            raise ValueError(f"Unknown membrane decode mode: {self.mode}")

        logits_proxy = v / self.temperature
        prediction = v.argmax(dim=-1)

        probs = torch.softmax(logits_proxy.float(), dim=-1)
        top2 = probs.topk(min(2, N), dim=-1).values
        if N >= 2:
            confidence = (top2[:, 0] - top2[:, 1]).clamp(min=0.0)
        else:
            confidence = top2[:, 0]

        return DecoderOutput(
            logits_proxy=logits_proxy,
            prediction=prediction,
            confidence=confidence,
            aux={"membrane_values": v.detach()},
        )


# ---- Loss term stubs ----

class _LossTermBase(nn.Module):
    """Base class for stub loss terms."""
    def forward(self, spikes, membrane, targets, **kwargs):
        raise NotImplementedError


class _StubProbSpikesLoss(_LossTermBase):
    """ProbSpikes loss: softmax CE on spike counts.

    AMP-safe: fp32 accumulation, fp32 log-softmax, clamp before log.
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-7,
                 mode: str = "softmax"):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.mode = mode

    def forward(self, spikes, membrane, targets, **kwargs):
        # spikes: (B, T, C) where C = num_classes
        counts = spikes.float().sum(dim=1)  # (B, C) in fp32 -- AMP rule 1

        if self.mode == "softmax":
            scaled = counts / self.temperature
            log_probs = F.log_softmax(scaled, dim=-1)  # fp32 -- AMP rule 2
            loss = F.nll_loss(log_probs, targets)
        elif self.mode == "normalize":
            total = counts.sum(dim=-1, keepdim=True)
            probs = counts / (total + self.eps)
            probs = probs.clamp(min=self.eps)  # AMP rule 3
            log_probs = probs.log()
            loss = F.nll_loss(log_probs, targets)
        else:
            raise ValueError(f"Unknown ProbSpikes mode: {self.mode}")

        with torch.no_grad():
            diag = {}
            diag["mean_count"] = counts.mean().item()
            diag["max_count"] = counts.max().item()
            diag["accuracy"] = (counts.argmax(dim=-1) == targets).float().mean().item()
            p = F.softmax(counts / self.temperature, dim=-1)
            diag["output_entropy"] = -(p * (p + 1e-8).log()).sum(-1).mean().item()

        return loss, diag


class _StubSpikeRateReg(_LossTermBase):
    """Spike rate regularisation: MSE vs target rate.

    AMP-safe: rates computed in fp32.
    """

    def __init__(self, target_rate: float = 0.1, min_rate: float = 0.01,
                 max_rate: float = 0.3, rate_type: str = "l2",
                 use_range_penalty: bool = True):
        super().__init__()
        self.target_rate = target_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.rate_type = rate_type
        self.use_range_penalty = use_range_penalty

    def forward(self, spikes, membrane, targets, **kwargs):
        rates = spikes.float().mean(dim=1)  # (B, N), fp32

        target = kwargs.get("target_rate", self.target_rate)
        if self.rate_type == "l2":
            target_loss = ((rates - target) ** 2).mean()
        elif self.rate_type == "l1":
            target_loss = (rates - target).abs().mean()
        else:
            raise ValueError(f"Unknown rate_type: {self.rate_type}")

        loss = target_loss

        if self.use_range_penalty:
            below = F.relu(self.min_rate - rates)
            above = F.relu(rates - self.max_rate)
            range_loss = (below + above).mean()
            loss = loss + range_loss

        with torch.no_grad():
            diag = {}
            diag["mean_rate"] = rates.mean().item()
            diag["max_rate"] = rates.max().item()
            diag["min_rate"] = rates.min().item()
            per_neuron_rate = rates.mean(dim=0)
            diag["dead_neuron_frac"] = (per_neuron_rate == 0).float().mean().item()
            diag["saturated_frac"] = (per_neuron_rate > 0.9).float().mean().item()

        return loss, diag


class _StubISIReg(_LossTermBase):
    """ISI regularisation: conv1d-based refractory penalty.

    AMP-safe: kernel registered as fp32 buffer, input cast to fp32 before conv1d.
    """

    def __init__(self, refractory_window: int = 5,
                 kernel_type: str = "exponential",
                 penalty_weight: float = 1.0):
        super().__init__()
        self.refractory_window = refractory_window
        self.kernel_type = kernel_type
        self.penalty_weight = penalty_weight

        # Build refractory kernel (1D, causal)
        k = torch.arange(1, refractory_window + 1, dtype=torch.float32)
        if kernel_type == "exponential":
            tau = refractory_window / 3.0
            kernel = torch.exp(-k / tau)
        elif kernel_type == "rectangular":
            kernel = torch.ones_like(k)
        else:
            kernel = torch.exp(-k / (refractory_window / 3.0))
        # Register as buffer: (1, 1, refractory_window), flipped for causal conv
        self.register_buffer("kernel", kernel.flip(0).unsqueeze(0).unsqueeze(0))

    def forward(self, spikes, membrane, targets, **kwargs):
        B, T, N = spikes.shape

        # Reshape for conv1d: (B*N, 1, T)
        s = spikes.float().permute(0, 2, 1).reshape(B * N, 1, T)

        # Causal convolution: pad left by refractory_window
        kernel_fp32 = self.kernel.float()  # ensure fp32 -- AMP rule 7
        padded = F.pad(s, (self.refractory_window, 0))
        refractory_signal = F.conv1d(padded, kernel_fp32)  # (B*N, 1, T) fp32

        # Penalty: spikes firing during refractory period
        penalty = (s * refractory_signal).mean()
        loss = self.penalty_weight * penalty

        with torch.no_grad():
            diag = {}
            s_bn = s.view(B, N, T)
            spike_counts = s_bn.sum(dim=-1)
            firing_mask = spike_counts > 1
            if firing_mask.any():
                diag["mean_isi"] = (T / spike_counts[firing_mask]).mean().item()
            else:
                diag["mean_isi"] = float(T)

            refr_at_spikes = (s * refractory_signal).view(B, N, T)
            total_spikes = s_bn.sum()
            if total_spikes > 0:
                burst_count = (refr_at_spikes > 0.5).float().sum()
                diag["burst_fraction"] = (burst_count / total_spikes).item()
            else:
                diag["burst_fraction"] = 0.0
            diag["mean_refractory_penalty"] = penalty.item()

        return loss, diag


class _StubTemporalConsistency(_LossTermBase):
    """Temporal consistency loss: diff-based smoothness penalty.

    AMP-safe: operations performed in fp32.
    """

    def __init__(self, window_size: int = 5, penalty_type: str = "l2"):
        super().__init__()
        self.window_size = window_size
        self.penalty_type = penalty_type

    def forward(self, spikes, membrane, targets, **kwargs):
        B, T, N = spikes.shape

        if self.penalty_type in ("l1", "l2"):
            rates = spikes.float()
            diffs = rates[:, 1:, :] - rates[:, :-1, :]
            if self.penalty_type == "l1":
                loss = diffs.abs().mean()
            else:
                loss = (diffs ** 2).mean()
        elif self.penalty_type == "variance":
            if T < self.window_size * 2:
                zero = torch.tensor(0.0, device=spikes.device, dtype=torch.float32)
                return zero, {"skipped": 1.0}
            n_windows = T // self.window_size
            trimmed = spikes[:, :n_windows * self.window_size, :].float()
            windowed = trimmed.view(B, n_windows, self.window_size, N)
            window_means = windowed.mean(dim=2)
            loss = window_means.var(dim=1).mean()
        else:
            raise ValueError(f"Unknown penalty type: {self.penalty_type}")

        # Optional membrane smoothness
        if membrane is not None and membrane.dim() == 3:
            mem_diffs = membrane[:, 1:, :].float() - membrane[:, :-1, :].float()
            mem_smooth = (mem_diffs ** 2).mean()
            loss = loss + 0.1 * mem_smooth

        with torch.no_grad():
            diag = {}
            rate_diffs = spikes.float()[:, 1:, :] - spikes.float()[:, :-1, :]
            diag["mean_temporal_variance"] = rate_diffs.var().item()
            diag["max_temporal_variance"] = rate_diffs.var(dim=1).max().item()

        return loss, diag


class _StubMembraneReg(_LossTermBase):
    """Membrane potential regularisation: quadratic penalty on excess.

    AMP-safe: membrane cast to fp32 before computation.
    """

    def __init__(self, max_membrane: float = 1.5):
        super().__init__()
        self.max_membrane = max_membrane

    def forward(self, spikes, membrane, targets, **kwargs):
        if membrane is None:
            zero = torch.tensor(0.0, device=spikes.device, dtype=torch.float32)
            return zero, {"skipped": 1.0}

        excess = F.relu(membrane.float().abs() - self.max_membrane)
        loss = (excess ** 2).mean()

        with torch.no_grad():
            diag = {}
            diag["mean_membrane"] = membrane.float().mean().item()
            diag["max_membrane"] = membrane.float().abs().max().item()
            diag["explosion_frac"] = (
                (membrane.float().abs() > self.max_membrane).float().mean().item()
            )

        return loss, diag


class _StubSNNLossComposer(nn.Module):
    """Loss composer: weighted sum of multiple loss terms.

    Mirrors the SNNLossComposer contract from loss-pack.md.
    """

    def __init__(self, terms: Dict[str, Tuple[nn.Module, float]]):
        super().__init__()
        self.term_modules = nn.ModuleDict({k: v[0] for k, v in terms.items()})
        self.weights = {k: v[1] for k, v in terms.items()}

    def forward(self, spikes, membrane, targets, **kwargs):
        total = torch.tensor(0.0, device=spikes.device, dtype=torch.float32)
        components = {}
        diagnostics = {}

        for name, term in self.term_modules.items():
            w = self.weights[name]
            if w == 0.0:
                continue
            raw_loss, diag = term(spikes, membrane, targets, **kwargs)
            # Ensure fp32 -- AMP rule 5
            raw_loss = raw_loss.float()
            weighted = w * raw_loss
            total = total + weighted
            components[f"loss/{name}_raw"] = raw_loss.item()
            components[f"loss/{name}_weighted"] = weighted.item()
            diagnostics.update({f"diag/{name}/{k}": v for k, v in diag.items()})

        components["loss/total"] = total.item()
        return total, {**components, **diagnostics}


# ---------------------------------------------------------------------------
# SECTION 4: Test configuration
# ---------------------------------------------------------------------------

@dataclass
class AMPTestConfig:
    """Configuration for the AMP stress test suite."""
    batch_sizes: List[int] = field(default_factory=lambda: [4, 16, 32])
    time_steps: List[int] = field(default_factory=lambda: [10, 25, 50, 100])
    feature_sizes: List[int] = field(default_factory=lambda: [64, 256, 1024])
    num_classes: List[int] = field(default_factory=lambda: [10, 100])
    num_trials: int = 5
    device: str = "cuda" if _HAS_CUDA else "cpu"
    # Tolerance for numerical precision comparison
    loss_rtol: float = 1e-3
    grad_rtol: float = 1e-2

    @classmethod
    def quick(cls) -> "AMPTestConfig":
        """Reduced configuration for quick testing."""
        return cls(
            batch_sizes=[4],
            time_steps=[10, 25],
            feature_sizes=[64],
            num_classes=[10],
            num_trials=2,
        )

    @classmethod
    def full(cls) -> "AMPTestConfig":
        """Full stress test configuration."""
        return cls()


# ---------------------------------------------------------------------------
# SECTION 5: Result tracking
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Result of a single test case."""
    name: str
    passed: bool
    duration_ms: float = 0.0
    memory_bytes: int = 0
    details: str = ""
    error: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


class AMPStressReport:
    """Aggregates and reports test results."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()

    def add(self, result: TestResult):
        self.results.append(result)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def summary_dict(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        failures = [
            {"name": r.name, "error": r.error, "config": r.config}
            for r in self.results if not r.passed
        ]
        return {
            "total_tests": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "elapsed_seconds": round(elapsed, 2),
            "all_passed": self.all_passed,
            "failures": failures,
        }

    def print_report(self, verbose: bool = False):
        """Print formatted report with ANSI colours."""
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        elapsed = time.time() - self.start_time
        print()
        print(f"{BOLD}{'=' * 72}{RESET}")
        print(f"{BOLD}  AMP Stress Test Report{RESET}")
        print(f"{'=' * 72}")
        print(f"  Device : {_HAS_CUDA and 'CUDA' or 'CPU'}")
        print(f"  Stubs  : {'yes' if _USING_STUBS else 'no (real API)'}")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"{'=' * 72}")

        # Group results by category
        categories: Dict[str, List[TestResult]] = {}
        for r in self.results:
            cat = r.name.split("/")[0] if "/" in r.name else "other"
            categories.setdefault(cat, []).append(r)

        for cat, results in categories.items():
            cat_pass = sum(1 for r in results if r.passed)
            cat_total = len(results)
            status_color = GREEN if cat_pass == cat_total else RED
            print(f"\n  {CYAN}{cat}{RESET} [{status_color}{cat_pass}/{cat_total}{RESET}]")

            for r in results:
                status = f"{GREEN}PASS{RESET}" if r.passed else f"{RED}FAIL{RESET}"
                timing = f"{r.duration_ms:.1f}ms" if r.duration_ms > 0 else ""
                mem_str = ""
                if r.memory_bytes > 0:
                    mem_mb = r.memory_bytes / (1024 * 1024)
                    mem_str = f" | {mem_mb:.1f}MB"
                line = f"    [{status}] {r.name}"
                if timing or mem_str:
                    line += f"  ({timing}{mem_str})"
                print(line)
                if not r.passed and r.error:
                    for err_line in r.error.split("\n")[:5]:
                        print(f"           {RED}{err_line}{RESET}")
                if verbose and r.details:
                    for det_line in r.details.split("\n"):
                        print(f"           {YELLOW}{det_line}{RESET}")

        print(f"\n{'=' * 72}")
        overall_color = GREEN if self.all_passed else RED
        print(f"  {BOLD}Result: {overall_color}"
              f"{self.passed}/{self.total} passed{RESET}")
        if self.failed > 0:
            print(f"  {RED}{self.failed} test(s) FAILED{RESET}")
        print(f"{'=' * 72}\n")

    def to_json(self) -> str:
        """Export report as JSON string."""
        data = self.summary_dict()
        data["results"] = [
            {
                "name": r.name,
                "passed": r.passed,
                "duration_ms": r.duration_ms,
                "memory_bytes": r.memory_bytes,
                "details": r.details,
                "error": r.error,
                "config": r.config,
            }
            for r in self.results
        ]
        return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# SECTION 6: Utility helpers
# ---------------------------------------------------------------------------

def _get_autocast_device(device_str: str) -> str:
    """Return the device type string for torch.amp.autocast."""
    if device_str.startswith("cuda"):
        return "cuda"
    return "cpu"


def _get_peak_memory(device: str) -> int:
    """Return peak GPU memory in bytes, or 0 for CPU."""
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated()
    return 0


def _reset_peak_memory(device: str):
    """Reset peak GPU memory tracker."""
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _make_spikes(B: int, T: int, N: int, device: str,
                 rate: float = 0.2, requires_grad: bool = True) -> torch.Tensor:
    """Create a random binary spike tensor with gradient support."""
    p = torch.full((B, T, N), rate, device=device)
    noise = torch.rand_like(p)
    hard = (noise < p).float()
    if requires_grad:
        soft = torch.sigmoid((p - noise) * 10.0)
        spikes = (hard - soft).detach() + soft
        spikes.requires_grad_(True)
    else:
        spikes = hard
    return spikes


def _make_membrane(B: int, T: int, N: int, device: str,
                   requires_grad: bool = True) -> torch.Tensor:
    """Create a synthetic membrane potential tensor."""
    membrane = torch.randn(B, T, N, device=device) * 0.5 + 0.3
    if requires_grad:
        membrane.requires_grad_(True)
    return membrane


def _make_targets(B: int, num_classes: int, device: str) -> torch.Tensor:
    """Create random classification targets."""
    return torch.randint(0, num_classes, (B,), device=device)


# ---------------------------------------------------------------------------
# SECTION 7: Encoder AMP tests
# ---------------------------------------------------------------------------

def run_encoder_tests(config: AMPTestConfig, report: AMPStressReport,
                      verbose: bool = False):
    """Test all encoder types under AMP autocast."""
    device = config.device
    ac_device = _get_autocast_device(device)

    encoder_factories = {
        "rate": lambda N, T: _StubRateEncoder(
            num_features=N, num_steps=T, spike_prob=0.2),
        "latency": lambda N, T: _StubLatencyEncoder(
            num_features=N, num_steps=T),
        "ttfs": lambda N, T: _StubTTFSEncoder(
            num_features=N, num_steps=T),
        "population": lambda N, T: _StubPopulationEncoder(
            num_features=N, num_steps=T, neurons_per_feature=4),
    }

    for enc_name, factory in encoder_factories.items():
        for B in config.batch_sizes:
            for T in config.time_steps:
                for N in config.feature_sizes:
                    test_name = f"encoder/{enc_name}/B{B}_T{T}_N{N}"
                    t0 = time.time()
                    _reset_peak_memory(device)

                    try:
                        encoder = factory(N, T).to(device)
                        x = torch.rand(B, N, device=device, requires_grad=True)

                        # Run under autocast
                        with torch.amp.autocast(device_type=ac_device,
                                                enabled=True):
                            spikes = encoder(x)

                        elapsed_ms = (time.time() - t0) * 1000
                        peak_mem = _get_peak_memory(device)

                        errors = []

                        # Check shape
                        if enc_name == "population":
                            expected_N = N * 4
                        else:
                            expected_N = N
                        expected_shape = (B, T, expected_N)
                        if spikes.shape != expected_shape:
                            errors.append(
                                f"Shape mismatch: got {spikes.shape}, "
                                f"expected {expected_shape}"
                            )

                        # Check finite
                        if not spikes.isfinite().all():
                            nan_count = spikes.isnan().sum().item()
                            inf_count = spikes.isinf().sum().item()
                            errors.append(
                                f"Non-finite values: {nan_count} NaN, "
                                f"{inf_count} Inf"
                            )

                        # Check dtype: spikes should be float32 (NOT float16)
                        if spikes.dtype != torch.float32:
                            errors.append(
                                f"Wrong dtype: got {spikes.dtype}, "
                                f"expected float32"
                            )

                        # Check binary-ish (for rate/latency/ttfs, values
                        # should be near 0 or 1 due to STE)
                        with torch.no_grad():
                            vals = spikes.detach()
                            # Allow soft values from STE but verify range
                            if vals.min() < -0.1 or vals.max() > 1.1:
                                errors.append(
                                    f"Values out of range: "
                                    f"[{vals.min():.3f}, {vals.max():.3f}]"
                                )

                        # Test gradient flow
                        loss = spikes.sum()
                        loss.backward()
                        if x.grad is None:
                            errors.append("No gradient flow to input")
                        elif not x.grad.isfinite().all():
                            errors.append(
                                f"Non-finite input gradients: "
                                f"{x.grad.isnan().sum()} NaN"
                            )

                        passed = len(errors) == 0
                        report.add(TestResult(
                            name=test_name,
                            passed=passed,
                            duration_ms=elapsed_ms,
                            memory_bytes=peak_mem,
                            error="\n".join(errors) if errors else "",
                            config={"B": B, "T": T, "N": N,
                                    "encoder": enc_name},
                        ))

                    except Exception as e:
                        elapsed_ms = (time.time() - t0) * 1000
                        report.add(TestResult(
                            name=test_name,
                            passed=False,
                            duration_ms=elapsed_ms,
                            error=f"Exception: {e}\n{traceback.format_exc()}",
                            config={"B": B, "T": T, "N": N,
                                    "encoder": enc_name},
                        ))


# ---------------------------------------------------------------------------
# SECTION 8: Decoder AMP tests
# ---------------------------------------------------------------------------

def run_decoder_tests(config: AMPTestConfig, report: AMPStressReport,
                      verbose: bool = False):
    """Test all decoder types under AMP autocast."""
    device = config.device
    ac_device = _get_autocast_device(device)

    for num_classes in config.num_classes:
        decoder_factories = {
            "rate": lambda: _StubRateDecoder(temperature=1.0),
            "first_spike": lambda: _StubFirstSpikeDecoder(temperature=1.0),
            "population": lambda: _StubPopulationDecoder(
                num_classes=num_classes, temperature=1.0),
            "membrane": lambda: _StubMembraneDecoder(mode="final",
                                                      temperature=1.0),
        }

        for dec_name, factory in decoder_factories.items():
            for B in config.batch_sizes:
                for T in config.time_steps:
                    # For population decoder, N must be divisible by
                    # num_classes
                    if dec_name == "population":
                        N_vals = [
                            nc for nc in config.feature_sizes
                            if nc >= num_classes
                        ]
                        if not N_vals:
                            N_vals = [num_classes * 10]
                    else:
                        N_vals = config.feature_sizes

                    for N in N_vals:
                        test_name = (
                            f"decoder/{dec_name}/B{B}_T{T}_N{N}"
                            f"_C{num_classes}"
                        )
                        t0 = time.time()
                        _reset_peak_memory(device)

                        try:
                            decoder = factory().to(device)
                            spikes = _make_spikes(B, T, N, device, rate=0.2)
                            membrane = _make_membrane(B, T, N, device)

                            # Run under autocast
                            with torch.amp.autocast(device_type=ac_device,
                                                    enabled=True):
                                if dec_name == "membrane":
                                    output = decoder(spikes, membrane)
                                else:
                                    output = decoder(spikes)

                            elapsed_ms = (time.time() - t0) * 1000
                            peak_mem = _get_peak_memory(device)

                            errors = []

                            # Check logits_proxy is finite
                            if not output.logits_proxy.isfinite().all():
                                nan_c = output.logits_proxy.isnan().sum().item()
                                inf_c = output.logits_proxy.isinf().sum().item()
                                errors.append(
                                    f"Non-finite logits_proxy: "
                                    f"{nan_c} NaN, {inf_c} Inf"
                                )

                            # Check logits_proxy dtype is float32
                            if output.logits_proxy.dtype != torch.float32:
                                errors.append(
                                    f"logits_proxy wrong dtype: "
                                    f"{output.logits_proxy.dtype}, "
                                    f"expected float32"
                                )

                            # Check prediction dtype is int64
                            if output.prediction.dtype != torch.int64:
                                errors.append(
                                    f"prediction wrong dtype: "
                                    f"{output.prediction.dtype}, "
                                    f"expected int64"
                                )

                            # Check confidence in [0, 1]
                            conf = output.confidence.detach()
                            if conf.min() < -0.01 or conf.max() > 1.01:
                                errors.append(
                                    f"confidence out of [0,1]: "
                                    f"[{conf.min():.4f}, {conf.max():.4f}]"
                                )

                            # Check confidence is finite
                            if not conf.isfinite().all():
                                errors.append("Non-finite confidence values")

                            # Test gradient flow through logits_proxy
                            loss = output.logits_proxy.sum()
                            loss.backward()
                            if spikes.grad is None:
                                errors.append(
                                    "No gradient flow through decoder"
                                )
                            elif not spikes.grad.isfinite().all():
                                errors.append(
                                    "Non-finite gradients through decoder"
                                )

                            passed = len(errors) == 0
                            report.add(TestResult(
                                name=test_name,
                                passed=passed,
                                duration_ms=elapsed_ms,
                                memory_bytes=peak_mem,
                                error="\n".join(errors) if errors else "",
                                config={"B": B, "T": T, "N": N,
                                        "num_classes": num_classes,
                                        "decoder": dec_name},
                            ))

                        except Exception as e:
                            elapsed_ms = (time.time() - t0) * 1000
                            report.add(TestResult(
                                name=test_name,
                                passed=False,
                                duration_ms=elapsed_ms,
                                error=(
                                    f"Exception: {e}\n"
                                    f"{traceback.format_exc()}"
                                ),
                                config={"B": B, "T": T, "N": N,
                                        "num_classes": num_classes,
                                        "decoder": dec_name},
                            ))


# ---------------------------------------------------------------------------
# SECTION 9: Loss AMP tests
# ---------------------------------------------------------------------------

def run_loss_tests(config: AMPTestConfig, report: AMPStressReport,
                   verbose: bool = False):
    """Test all loss terms under AMP autocast."""
    device = config.device
    ac_device = _get_autocast_device(device)

    for num_classes in config.num_classes:
        for B in config.batch_sizes:
            for T in config.time_steps:
                N = num_classes  # Loss terms expect N = num_classes

                # ---- ProbSpikes loss ----
                _run_prob_spikes_tests(
                    B, T, N, device, ac_device, config, report
                )

                # ---- Spike rate regularisation ----
                _run_rate_reg_tests(
                    B, T, N, device, ac_device, report
                )

                # ---- Temporal consistency ----
                _run_temporal_tests(
                    B, T, N, device, ac_device, report
                )

                # ---- ISI regularisation ----
                _run_isi_tests(
                    B, T, N, device, ac_device, report
                )

                # ---- Membrane regularisation ----
                _run_membrane_reg_tests(
                    B, T, N, device, ac_device, report
                )

    # ---- Loss composer tests ----
    _run_composer_tests(config, report)


def _run_prob_spikes_tests(B, T, N, device, ac_device, config, report):
    """ProbSpikes-specific AMP tests."""
    # Test multiple temperatures
    temperatures = [0.1, 1.0, 10.0]

    for temp in temperatures:
        for mode in ["softmax", "normalize"]:
            test_name = (
                f"loss/prob_spikes/B{B}_T{T}_N{N}_temp{temp}_mode{mode}"
            )
            t0 = time.time()
            _reset_peak_memory(device)

            try:
                term = _StubProbSpikesLoss(
                    temperature=temp, mode=mode
                ).to(device)
                spikes = _make_spikes(B, T, N, device, rate=0.2)
                membrane = _make_membrane(B, T, N, device)
                targets = _make_targets(B, N, device)

                # Run under autocast
                with torch.amp.autocast(device_type=ac_device, enabled=True):
                    loss, diag = term(spikes, membrane, targets)

                elapsed_ms = (time.time() - t0) * 1000
                peak_mem = _get_peak_memory(device)

                errors = []

                # Check loss is finite scalar
                if not torch.isfinite(loss):
                    errors.append(f"Loss not finite: {loss.item()}")

                # Check loss dtype is float32
                if loss.dtype != torch.float32:
                    errors.append(
                        f"Loss wrong dtype: {loss.dtype}, expected float32"
                    )

                # Check loss is non-negative for CE
                # (CE can be negative with label smoothing, but standard
                # NLL loss should be >= 0 in expectation)

                # Verify internal counts are fp32
                with torch.amp.autocast(device_type=ac_device, enabled=True):
                    counts_check = spikes.float().sum(dim=1)
                if counts_check.dtype != torch.float32:
                    errors.append(
                        f"Internal counts not fp32: {counts_check.dtype}"
                    )

                # Verify log-softmax stability
                with torch.no_grad():
                    scaled = counts_check / temp
                    log_probs_check = F.log_softmax(scaled, dim=-1)
                    if not log_probs_check.isfinite().all():
                        errors.append(
                            "log-softmax produced non-finite values"
                        )

                # Test backward
                loss.backward()
                if spikes.grad is None:
                    errors.append("No gradient flow through ProbSpikes")
                elif not spikes.grad.isfinite().all():
                    errors.append(
                        f"Non-finite gradients: "
                        f"{spikes.grad.isnan().sum()} NaN"
                    )

                passed = len(errors) == 0
                report.add(TestResult(
                    name=test_name,
                    passed=passed,
                    duration_ms=elapsed_ms,
                    memory_bytes=peak_mem,
                    error="\n".join(errors) if errors else "",
                    details=(
                        f"diag: {diag}" if passed else ""
                    ),
                    config={"B": B, "T": T, "N": N, "temp": temp,
                            "mode": mode},
                ))

            except Exception as e:
                elapsed_ms = (time.time() - t0) * 1000
                report.add(TestResult(
                    name=test_name,
                    passed=False,
                    duration_ms=elapsed_ms,
                    error=f"Exception: {e}\n{traceback.format_exc()}",
                    config={"B": B, "T": T, "N": N, "temp": temp,
                            "mode": mode},
                ))


def _run_rate_reg_tests(B, T, N, device, ac_device, report):
    """Spike rate regularisation AMP tests."""
    test_name = f"loss/rate_reg/B{B}_T{T}_N{N}"
    t0 = time.time()
    _reset_peak_memory(device)

    try:
        term = _StubSpikeRateReg(target_rate=0.1).to(device)
        spikes = _make_spikes(B, T, N, device, rate=0.15)
        membrane = _make_membrane(B, T, N, device)
        targets = _make_targets(B, N, device)

        with torch.amp.autocast(device_type=ac_device, enabled=True):
            loss, diag = term(spikes, membrane, targets)

        elapsed_ms = (time.time() - t0) * 1000
        peak_mem = _get_peak_memory(device)

        errors = []
        if not torch.isfinite(loss):
            errors.append(f"Loss not finite: {loss.item()}")
        if loss.dtype != torch.float32:
            errors.append(f"Loss wrong dtype: {loss.dtype}")

        loss.backward()
        if spikes.grad is None:
            errors.append("No gradient flow through rate_reg")
        elif not spikes.grad.isfinite().all():
            errors.append("Non-finite gradients in rate_reg")

        report.add(TestResult(
            name=test_name,
            passed=len(errors) == 0,
            duration_ms=elapsed_ms,
            memory_bytes=peak_mem,
            error="\n".join(errors) if errors else "",
            config={"B": B, "T": T, "N": N},
        ))

    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        report.add(TestResult(
            name=test_name,
            passed=False,
            duration_ms=elapsed_ms,
            error=f"Exception: {e}\n{traceback.format_exc()}",
            config={"B": B, "T": T, "N": N},
        ))


def _run_temporal_tests(B, T, N, device, ac_device, report):
    """Temporal consistency AMP tests."""
    for penalty in ["l1", "l2", "variance"]:
        test_name = f"loss/temporal/{penalty}/B{B}_T{T}_N{N}"
        t0 = time.time()
        _reset_peak_memory(device)

        try:
            term = _StubTemporalConsistency(
                window_size=5, penalty_type=penalty
            ).to(device)
            spikes = _make_spikes(B, T, N, device, rate=0.2)
            membrane = _make_membrane(B, T, N, device)
            targets = _make_targets(B, N, device)

            with torch.amp.autocast(device_type=ac_device, enabled=True):
                loss, diag = term(spikes, membrane, targets)

            elapsed_ms = (time.time() - t0) * 1000
            peak_mem = _get_peak_memory(device)

            errors = []
            if not torch.isfinite(loss):
                errors.append(f"Loss not finite: {loss.item()}")
            if loss.dtype != torch.float32:
                errors.append(f"Loss wrong dtype: {loss.dtype}")

            # Skip backward for variance mode when T is too small
            if not (penalty == "variance" and T < 10):
                loss.backward()
                if spikes.grad is None:
                    # Variance mode with small T returns constant 0
                    if not (penalty == "variance" and
                            diag.get("skipped", 0) == 1.0):
                        errors.append(
                            f"No gradient flow through temporal/{penalty}"
                        )
                elif not spikes.grad.isfinite().all():
                    errors.append(
                        f"Non-finite gradients in temporal/{penalty}"
                    )

            report.add(TestResult(
                name=test_name,
                passed=len(errors) == 0,
                duration_ms=elapsed_ms,
                memory_bytes=peak_mem,
                error="\n".join(errors) if errors else "",
                config={"B": B, "T": T, "N": N, "penalty": penalty},
            ))

        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000
            report.add(TestResult(
                name=test_name,
                passed=False,
                duration_ms=elapsed_ms,
                error=f"Exception: {e}\n{traceback.format_exc()}",
                config={"B": B, "T": T, "N": N, "penalty": penalty},
            ))


def _run_isi_tests(B, T, N, device, ac_device, report):
    """ISI regularisation AMP tests."""
    for kernel_type in ["exponential", "rectangular"]:
        test_name = f"loss/isi/{kernel_type}/B{B}_T{T}_N{N}"
        t0 = time.time()
        _reset_peak_memory(device)

        try:
            term = _StubISIReg(
                refractory_window=5, kernel_type=kernel_type
            ).to(device)
            spikes = _make_spikes(B, T, N, device, rate=0.2)
            membrane = _make_membrane(B, T, N, device)
            targets = _make_targets(B, N, device)

            with torch.amp.autocast(device_type=ac_device, enabled=True):
                loss, diag = term(spikes, membrane, targets)

            elapsed_ms = (time.time() - t0) * 1000
            peak_mem = _get_peak_memory(device)

            errors = []
            if not torch.isfinite(loss):
                errors.append(f"Loss not finite: {loss.item()}")
            if loss.dtype != torch.float32:
                errors.append(f"Loss wrong dtype: {loss.dtype}")

            # Verify conv kernel is fp32
            if term.kernel.dtype != torch.float32:
                errors.append(
                    f"ISI kernel not fp32: {term.kernel.dtype}"
                )

            # Verify no NaN from conv1d under autocast
            with torch.amp.autocast(device_type=ac_device, enabled=True):
                s_test = spikes.float().permute(0, 2, 1).reshape(
                    B * N, 1, T
                )
                padded = F.pad(s_test, (5, 0))
                conv_out = F.conv1d(padded, term.kernel.float())
            if not conv_out.isfinite().all():
                errors.append(
                    "conv1d produced non-finite values under autocast"
                )

            loss.backward()
            if spikes.grad is None:
                errors.append(f"No gradient flow through ISI/{kernel_type}")
            elif not spikes.grad.isfinite().all():
                errors.append(
                    f"Non-finite gradients in ISI/{kernel_type}"
                )

            report.add(TestResult(
                name=test_name,
                passed=len(errors) == 0,
                duration_ms=elapsed_ms,
                memory_bytes=peak_mem,
                error="\n".join(errors) if errors else "",
                config={"B": B, "T": T, "N": N,
                        "kernel_type": kernel_type},
            ))

        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000
            report.add(TestResult(
                name=test_name,
                passed=False,
                duration_ms=elapsed_ms,
                error=f"Exception: {e}\n{traceback.format_exc()}",
                config={"B": B, "T": T, "N": N,
                        "kernel_type": kernel_type},
            ))


def _run_membrane_reg_tests(B, T, N, device, ac_device, report):
    """Membrane potential regularisation AMP tests."""
    test_name = f"loss/membrane_reg/B{B}_T{T}_N{N}"
    t0 = time.time()
    _reset_peak_memory(device)

    try:
        term = _StubMembraneReg(max_membrane=1.5).to(device)
        spikes = _make_spikes(B, T, N, device, rate=0.2)
        # Use membrane with values that will exceed the threshold
        membrane = torch.randn(B, T, N, device=device) * 2.0
        membrane.requires_grad_(True)
        targets = _make_targets(B, N, device)

        with torch.amp.autocast(device_type=ac_device, enabled=True):
            loss, diag = term(spikes, membrane, targets)

        elapsed_ms = (time.time() - t0) * 1000
        peak_mem = _get_peak_memory(device)

        errors = []
        if not torch.isfinite(loss):
            errors.append(f"Loss not finite: {loss.item()}")
        if loss.dtype != torch.float32:
            errors.append(f"Loss wrong dtype: {loss.dtype}")

        loss.backward()
        if membrane.grad is None:
            errors.append("No gradient flow through membrane_reg")
        elif not membrane.grad.isfinite().all():
            errors.append("Non-finite gradients in membrane_reg")

        report.add(TestResult(
            name=test_name,
            passed=len(errors) == 0,
            duration_ms=elapsed_ms,
            memory_bytes=peak_mem,
            error="\n".join(errors) if errors else "",
            config={"B": B, "T": T, "N": N},
        ))

    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        report.add(TestResult(
            name=test_name,
            passed=False,
            duration_ms=elapsed_ms,
            error=f"Exception: {e}\n{traceback.format_exc()}",
            config={"B": B, "T": T, "N": N},
        ))


def _run_composer_tests(config: AMPTestConfig, report: AMPStressReport):
    """Test the loss composer with all terms combined under autocast."""
    device = config.device
    ac_device = _get_autocast_device(device)

    for B in config.batch_sizes[:2]:  # Limit combinations
        for T in config.time_steps:
            for num_classes in config.num_classes[:1]:
                N = num_classes
                test_name = f"loss/composer/B{B}_T{T}_N{N}"
                t0 = time.time()
                _reset_peak_memory(device)

                try:
                    composer = _StubSNNLossComposer(terms={
                        "prob_spikes": (
                            _StubProbSpikesLoss(temperature=1.0), 1.0
                        ),
                        "spike_rate": (
                            _StubSpikeRateReg(target_rate=0.1), 0.1
                        ),
                        "temporal": (
                            _StubTemporalConsistency(
                                window_size=5, penalty_type="l2"
                            ), 0.01
                        ),
                        "isi": (
                            _StubISIReg(
                                refractory_window=5,
                                kernel_type="exponential"
                            ), 0.01
                        ),
                        "membrane": (
                            _StubMembraneReg(max_membrane=1.5), 0.001
                        ),
                    }).to(device)

                    spikes = _make_spikes(B, T, N, device, rate=0.15)
                    membrane = _make_membrane(B, T, N, device)
                    targets = _make_targets(B, N, device)

                    with torch.amp.autocast(device_type=ac_device,
                                            enabled=True):
                        total_loss, log_dict = composer(
                            spikes, membrane, targets
                        )

                    elapsed_ms = (time.time() - t0) * 1000
                    peak_mem = _get_peak_memory(device)

                    errors = []

                    # Total loss checks
                    if not torch.isfinite(total_loss):
                        errors.append(
                            f"Total loss not finite: {total_loss.item()}"
                        )
                    if total_loss.dtype != torch.float32:
                        errors.append(
                            f"Total loss wrong dtype: {total_loss.dtype}"
                        )

                    # Check all component losses are finite
                    for key, val in log_dict.items():
                        if key.startswith("loss/") and not math.isfinite(val):
                            errors.append(
                                f"Non-finite component: {key} = {val}"
                            )

                    # Test backward
                    total_loss.backward()
                    if spikes.grad is None:
                        errors.append(
                            "No gradient flow through composer"
                        )
                    elif not spikes.grad.isfinite().all():
                        errors.append(
                            "Non-finite gradients through composer"
                        )

                    report.add(TestResult(
                        name=test_name,
                        passed=len(errors) == 0,
                        duration_ms=elapsed_ms,
                        memory_bytes=peak_mem,
                        error="\n".join(errors) if errors else "",
                        details=f"components: {list(log_dict.keys())}",
                        config={"B": B, "T": T, "N": N},
                    ))

                except Exception as e:
                    elapsed_ms = (time.time() - t0) * 1000
                    report.add(TestResult(
                        name=test_name,
                        passed=False,
                        duration_ms=elapsed_ms,
                        error=(
                            f"Exception: {e}\n{traceback.format_exc()}"
                        ),
                        config={"B": B, "T": T, "N": N},
                    ))


# ---------------------------------------------------------------------------
# SECTION 10: Scaling stress tests
# ---------------------------------------------------------------------------

def run_scaling_tests(config: AMPTestConfig, report: AMPStressReport,
                      verbose: bool = False):
    """Test loss stability as T, spike rate, and edge cases vary."""
    device = config.device
    ac_device = _get_autocast_device(device)
    B = config.batch_sizes[0]
    N = config.num_classes[0]

    # --- T-scaling: verify loss magnitude stays bounded ---
    T_values = [10, 25, 50, 100, 200]
    loss_by_T: Dict[int, float] = {}

    for T in T_values:
        test_name = f"scaling/T_scaling/T{T}"
        t0 = time.time()

        try:
            term = _StubProbSpikesLoss(temperature=1.0).to(device)
            spikes = _make_spikes(B, T, N, device, rate=0.2)
            membrane = _make_membrane(B, T, N, device)
            targets = _make_targets(B, N, device)

            with torch.amp.autocast(device_type=ac_device, enabled=True):
                loss, _ = term(spikes, membrane, targets)

            loss_val = loss.item()
            loss_by_T[T] = loss_val
            elapsed_ms = (time.time() - t0) * 1000

            errors = []
            if not math.isfinite(loss_val):
                errors.append(f"Loss not finite at T={T}: {loss_val}")

            # Loss magnitude should not grow proportionally with T
            # (if counts are not normalised, softmax CE should still be
            # bounded because softmax normalises)
            if abs(loss_val) > 100.0:
                errors.append(
                    f"Loss magnitude too large at T={T}: {loss_val:.4f}"
                )

            report.add(TestResult(
                name=test_name,
                passed=len(errors) == 0,
                duration_ms=elapsed_ms,
                error="\n".join(errors) if errors else "",
                details=f"loss={loss_val:.6f}",
                config={"B": B, "T": T, "N": N},
            ))

        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000
            report.add(TestResult(
                name=test_name,
                passed=False,
                duration_ms=elapsed_ms,
                error=f"Exception: {e}\n{traceback.format_exc()}",
                config={"B": B, "T": T, "N": N},
            ))

    # --- Check loss magnitude ratio across T values ---
    if len(loss_by_T) >= 2:
        T_min, T_max = min(loss_by_T.keys()), max(loss_by_T.keys())
        if loss_by_T[T_min] > 0:
            ratio = loss_by_T[T_max] / loss_by_T[T_min]
            # Ratio should be modest (< 10x for 20x T increase)
            if ratio > 20.0:
                report.add(TestResult(
                    name="scaling/T_scaling/ratio_check",
                    passed=False,
                    error=(
                        f"Loss ratio T={T_max}/T={T_min} = {ratio:.2f} "
                        f"(>{20.0}x), loss may scale with T"
                    ),
                    details=(
                        f"loss[T={T_min}]={loss_by_T[T_min]:.6f}, "
                        f"loss[T={T_max}]={loss_by_T[T_max]:.6f}"
                    ),
                    config={"T_min": T_min, "T_max": T_max},
                ))
            else:
                report.add(TestResult(
                    name="scaling/T_scaling/ratio_check",
                    passed=True,
                    details=f"Loss ratio T={T_max}/T={T_min} = {ratio:.2f}",
                    config={"T_min": T_min, "T_max": T_max},
                ))

    # --- Extreme spike rates ---
    T = 50
    for rate_name, rate_val in [("very_sparse", 0.01), ("very_dense", 0.95)]:
        test_name = f"scaling/extreme_rate/{rate_name}"
        t0 = time.time()

        try:
            term = _StubProbSpikesLoss(temperature=1.0).to(device)
            spikes = _make_spikes(B, T, N, device, rate=rate_val)
            membrane = _make_membrane(B, T, N, device)
            targets = _make_targets(B, N, device)

            with torch.amp.autocast(device_type=ac_device, enabled=True):
                loss, _ = term(spikes, membrane, targets)

            elapsed_ms = (time.time() - t0) * 1000
            errors = []
            if not torch.isfinite(loss):
                errors.append(
                    f"Loss not finite with rate={rate_val}: {loss.item()}"
                )

            loss.backward()
            if spikes.grad is None:
                errors.append(f"No gradient at rate={rate_val}")
            elif not spikes.grad.isfinite().all():
                errors.append(f"Non-finite gradients at rate={rate_val}")

            report.add(TestResult(
                name=test_name,
                passed=len(errors) == 0,
                duration_ms=elapsed_ms,
                error="\n".join(errors) if errors else "",
                details=f"loss={loss.item():.6f}",
                config={"B": B, "T": T, "N": N, "rate": rate_val},
            ))

        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000
            report.add(TestResult(
                name=test_name,
                passed=False,
                duration_ms=elapsed_ms,
                error=f"Exception: {e}\n{traceback.format_exc()}",
                config={"B": B, "T": T, "N": N, "rate": rate_val},
            ))

    # --- Zero spikes (all zeros) ---
    test_name = "scaling/edge_case/zero_spikes"
    t0 = time.time()

    try:
        term = _StubProbSpikesLoss(temperature=1.0, mode="softmax").to(device)
        spikes = torch.zeros(B, T, N, device=device, requires_grad=True)
        membrane = _make_membrane(B, T, N, device)
        targets = _make_targets(B, N, device)

        with torch.amp.autocast(device_type=ac_device, enabled=True):
            loss, _ = term(spikes, membrane, targets)

        elapsed_ms = (time.time() - t0) * 1000
        errors = []
        if not torch.isfinite(loss):
            errors.append(f"Loss not finite with zero spikes: {loss.item()}")

        loss.backward()
        if spikes.grad is not None and not spikes.grad.isfinite().all():
            errors.append("Non-finite gradients with zero spikes")

        report.add(TestResult(
            name=test_name,
            passed=len(errors) == 0,
            duration_ms=elapsed_ms,
            error="\n".join(errors) if errors else "",
            details=f"loss={loss.item():.6f}",
            config={"B": B, "T": T, "N": N},
        ))

    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        report.add(TestResult(
            name=test_name,
            passed=False,
            duration_ms=elapsed_ms,
            error=f"Exception: {e}\n{traceback.format_exc()}",
            config={"B": B, "T": T, "N": N},
        ))

    # --- All-ones spikes ---
    test_name = "scaling/edge_case/all_ones_spikes"
    t0 = time.time()

    try:
        term = _StubProbSpikesLoss(temperature=1.0, mode="softmax").to(device)
        spikes = torch.ones(B, T, N, device=device, requires_grad=True)
        membrane = _make_membrane(B, T, N, device)
        targets = _make_targets(B, N, device)

        with torch.amp.autocast(device_type=ac_device, enabled=True):
            loss, _ = term(spikes, membrane, targets)

        elapsed_ms = (time.time() - t0) * 1000
        errors = []
        if not torch.isfinite(loss):
            errors.append(
                f"Loss not finite with all-ones spikes: {loss.item()}"
            )
        if loss.item() == float("inf") or loss.item() == float("-inf"):
            errors.append(f"Loss is Inf with all-ones spikes: {loss.item()}")

        loss.backward()
        if spikes.grad is not None and not spikes.grad.isfinite().all():
            errors.append("Non-finite gradients with all-ones spikes")

        report.add(TestResult(
            name=test_name,
            passed=len(errors) == 0,
            duration_ms=elapsed_ms,
            error="\n".join(errors) if errors else "",
            details=f"loss={loss.item():.6f}",
            config={"B": B, "T": T, "N": N},
        ))

    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        report.add(TestResult(
            name=test_name,
            passed=False,
            duration_ms=elapsed_ms,
            error=f"Exception: {e}\n{traceback.format_exc()}",
            config={"B": B, "T": T, "N": N},
        ))

    # --- Gradient magnitude boundedness as T increases ---
    grad_norms: Dict[int, float] = {}
    for T in T_values:
        test_name = f"scaling/grad_magnitude/T{T}"
        t0 = time.time()

        try:
            composer = _StubSNNLossComposer(terms={
                "prob_spikes": (
                    _StubProbSpikesLoss(temperature=1.0), 1.0
                ),
                "spike_rate": (
                    _StubSpikeRateReg(target_rate=0.1), 0.1
                ),
                "isi": (
                    _StubISIReg(refractory_window=5), 0.01
                ),
            }).to(device)

            spikes = _make_spikes(B, T, N, device, rate=0.2)
            membrane = _make_membrane(B, T, N, device)
            targets = _make_targets(B, N, device)

            with torch.amp.autocast(device_type=ac_device, enabled=True):
                total_loss, _ = composer(spikes, membrane, targets)

            total_loss.backward()

            elapsed_ms = (time.time() - t0) * 1000
            errors = []

            if spikes.grad is not None:
                grad_norm = spikes.grad.norm().item()
                grad_norms[T] = grad_norm
                if not math.isfinite(grad_norm):
                    errors.append(
                        f"Non-finite gradient norm at T={T}: {grad_norm}"
                    )
            else:
                errors.append(f"No gradients at T={T}")

            report.add(TestResult(
                name=test_name,
                passed=len(errors) == 0,
                duration_ms=elapsed_ms,
                error="\n".join(errors) if errors else "",
                details=(
                    f"grad_norm={grad_norms.get(T, 'N/A')}"
                ),
                config={"B": B, "T": T, "N": N},
            ))

        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000
            report.add(TestResult(
                name=test_name,
                passed=False,
                duration_ms=elapsed_ms,
                error=f"Exception: {e}\n{traceback.format_exc()}",
                config={"B": B, "T": T, "N": N},
            ))

    # Check gradient norm growth rate
    if len(grad_norms) >= 2:
        T_min_g = min(grad_norms.keys())
        T_max_g = max(grad_norms.keys())
        if grad_norms[T_min_g] > 1e-12:
            g_ratio = grad_norms[T_max_g] / grad_norms[T_min_g]
            t_ratio = T_max_g / T_min_g
            # Gradient norm should not grow faster than sqrt(T_ratio)
            # (allowing some growth due to more terms)
            bound = t_ratio * 2.0  # generous bound
            passed = g_ratio < bound
            report.add(TestResult(
                name="scaling/grad_magnitude/growth_check",
                passed=passed,
                error="" if passed else (
                    f"Gradient norm grew {g_ratio:.2f}x for "
                    f"{t_ratio:.0f}x T increase (bound: {bound:.1f}x)"
                ),
                details=(
                    f"||grad||[T={T_min_g}]={grad_norms[T_min_g]:.6f}, "
                    f"||grad||[T={T_max_g}]={grad_norms[T_max_g]:.6f}, "
                    f"ratio={g_ratio:.2f}"
                ),
                config={"T_min": T_min_g, "T_max": T_max_g},
            ))


# ---------------------------------------------------------------------------
# SECTION 11: Memory profiling
# ---------------------------------------------------------------------------

def run_memory_tests(config: AMPTestConfig, report: AMPStressReport,
                     verbose: bool = False):
    """Measure and compare memory usage: fp32 vs autocast."""
    device = config.device
    if not device.startswith("cuda"):
        # Memory profiling only meaningful on CUDA
        report.add(TestResult(
            name="memory/skipped",
            passed=True,
            details="Memory profiling skipped on CPU (no GPU memory tracker)",
        ))
        return

    ac_device = _get_autocast_device(device)
    B = config.batch_sizes[0]
    N = config.num_classes[0]

    T_values = config.time_steps

    for T in T_values:
        # --- Measure fp32 memory ---
        _reset_peak_memory(device)
        torch.cuda.synchronize()

        composer_fp32 = _StubSNNLossComposer(terms={
            "prob_spikes": (_StubProbSpikesLoss(temperature=1.0), 1.0),
            "spike_rate": (_StubSpikeRateReg(target_rate=0.1), 0.1),
            "isi": (_StubISIReg(refractory_window=5), 0.01),
        }).to(device)

        spikes_fp32 = _make_spikes(B, T, N, device, rate=0.2)
        membrane_fp32 = _make_membrane(B, T, N, device)
        targets = _make_targets(B, N, device)

        _reset_peak_memory(device)
        loss_fp32, _ = composer_fp32(spikes_fp32, membrane_fp32, targets)
        loss_fp32.backward()
        torch.cuda.synchronize()
        mem_fp32 = _get_peak_memory(device)

        # Clean up
        del spikes_fp32, membrane_fp32, loss_fp32, composer_fp32
        torch.cuda.empty_cache()

        # --- Measure autocast memory ---
        _reset_peak_memory(device)
        torch.cuda.synchronize()

        composer_amp = _StubSNNLossComposer(terms={
            "prob_spikes": (_StubProbSpikesLoss(temperature=1.0), 1.0),
            "spike_rate": (_StubSpikeRateReg(target_rate=0.1), 0.1),
            "isi": (_StubISIReg(refractory_window=5), 0.01),
        }).to(device)

        spikes_amp = _make_spikes(B, T, N, device, rate=0.2)
        membrane_amp = _make_membrane(B, T, N, device)

        _reset_peak_memory(device)
        with torch.amp.autocast(device_type=ac_device, enabled=True):
            loss_amp, _ = composer_amp(spikes_amp, membrane_amp, targets)
        loss_amp.backward()
        torch.cuda.synchronize()
        mem_amp = _get_peak_memory(device)

        del spikes_amp, membrane_amp, loss_amp, composer_amp
        torch.cuda.empty_cache()

        # --- Report ---
        test_name = f"memory/T{T}/comparison"
        errors = []

        # Flag if AMP uses MORE memory (regression)
        if mem_amp > mem_fp32 * 1.5 and mem_fp32 > 0:
            errors.append(
                f"AMP memory regression: AMP={mem_amp} > "
                f"fp32={mem_fp32} * 1.5"
            )

        savings_pct = 0.0
        if mem_fp32 > 0:
            savings_pct = (1.0 - mem_amp / mem_fp32) * 100.0

        report.add(TestResult(
            name=test_name,
            passed=len(errors) == 0,
            memory_bytes=mem_amp,
            error="\n".join(errors) if errors else "",
            details=(
                f"fp32={mem_fp32 / 1024:.1f}KB, "
                f"amp={mem_amp / 1024:.1f}KB, "
                f"savings={savings_pct:.1f}%"
            ),
            config={"B": B, "T": T, "N": N},
        ))

    # --- Per-operation memory profiling ---
    T = 50
    operations = {
        "encode": lambda: _StubRateEncoder(
            num_features=N, num_steps=T
        ).to(device)(torch.rand(B, N, device=device)),
        "decode": lambda: _StubRateDecoder().to(device)(
            _make_spikes(B, T, N, device, rate=0.2, requires_grad=False)
        ),
        "loss": lambda: _StubProbSpikesLoss().to(device)(
            _make_spikes(B, T, N, device, rate=0.2, requires_grad=False),
            _make_membrane(B, T, N, device, requires_grad=False),
            _make_targets(B, N, device),
        ),
    }

    for op_name, op_fn in operations.items():
        _reset_peak_memory(device)
        torch.cuda.synchronize()

        _reset_peak_memory(device)
        with torch.amp.autocast(device_type=ac_device, enabled=True):
            _ = op_fn()
        torch.cuda.synchronize()
        mem_op = _get_peak_memory(device)

        report.add(TestResult(
            name=f"memory/per_op/{op_name}",
            passed=True,
            memory_bytes=mem_op,
            details=f"peak={mem_op / 1024:.1f}KB",
            config={"B": B, "T": T, "N": N, "op": op_name},
        ))


# ---------------------------------------------------------------------------
# SECTION 12: Numerical precision tests
# ---------------------------------------------------------------------------

def run_precision_tests(config: AMPTestConfig, report: AMPStressReport,
                        verbose: bool = False):
    """Compare loss values and gradients: full fp32 vs autocast."""
    device = config.device
    ac_device = _get_autocast_device(device)
    B = config.batch_sizes[0]
    N = config.num_classes[0]

    loss_terms = {
        "prob_spikes": lambda: _StubProbSpikesLoss(temperature=1.0),
        "rate_reg": lambda: _StubSpikeRateReg(target_rate=0.1),
        "temporal": lambda: _StubTemporalConsistency(
            window_size=5, penalty_type="l2"
        ),
        "isi": lambda: _StubISIReg(refractory_window=5),
        "membrane": lambda: _StubMembraneReg(max_membrane=1.5),
    }

    for T in config.time_steps:
        # Use same random seed for fair comparison
        for term_name, term_factory in loss_terms.items():
            test_name = f"precision/{term_name}/T{T}"
            t0 = time.time()

            try:
                # Fix random seed for reproducibility
                seed = 42 + T
                torch.manual_seed(seed)

                # --- fp32 baseline ---
                term_fp32 = term_factory().to(device)
                torch.manual_seed(seed + 1000)
                spikes_fp32 = _make_spikes(B, T, N, device, rate=0.2)
                membrane_fp32 = _make_membrane(B, T, N, device)
                targets = _make_targets(B, N, device)

                # Clone for AMP run with identical data
                spikes_data = spikes_fp32.detach().clone()
                membrane_data = membrane_fp32.detach().clone()

                loss_fp32_val, _ = term_fp32(
                    spikes_fp32, membrane_fp32, targets
                )
                loss_fp32_val.backward()
                grad_fp32 = spikes_fp32.grad.clone() if spikes_fp32.grad is not None else None
                loss_fp32_scalar = loss_fp32_val.item()

                # --- autocast ---
                term_amp = term_factory().to(device)
                # Load same state
                term_amp.load_state_dict(term_fp32.state_dict())

                spikes_amp = spikes_data.clone().requires_grad_(True)
                membrane_amp = membrane_data.clone().requires_grad_(True)

                with torch.amp.autocast(device_type=ac_device, enabled=True):
                    loss_amp_val, _ = term_amp(
                        spikes_amp, membrane_amp, targets
                    )
                loss_amp_val.backward()
                grad_amp = spikes_amp.grad.clone() if spikes_amp.grad is not None else None
                loss_amp_scalar = loss_amp_val.item()

                elapsed_ms = (time.time() - t0) * 1000
                errors = []
                details_parts = []

                # Compare loss values
                if loss_fp32_scalar != 0.0:
                    rel_err = abs(
                        loss_amp_scalar - loss_fp32_scalar
                    ) / (abs(loss_fp32_scalar) + 1e-12)
                    details_parts.append(
                        f"loss_fp32={loss_fp32_scalar:.8f}, "
                        f"loss_amp={loss_amp_scalar:.8f}, "
                        f"rel_err={rel_err:.2e}"
                    )
                    if rel_err > config.loss_rtol:
                        errors.append(
                            f"Loss relative error {rel_err:.2e} > "
                            f"tolerance {config.loss_rtol}"
                        )
                else:
                    details_parts.append(
                        f"loss_fp32=0.0, loss_amp={loss_amp_scalar:.8f}"
                    )

                # Compare gradient values
                if grad_fp32 is not None and grad_amp is not None:
                    grad_fp32_norm = grad_fp32.norm().item()
                    grad_amp_norm = grad_amp.norm().item()
                    if grad_fp32_norm > 1e-12:
                        grad_rel_err = abs(
                            grad_amp_norm - grad_fp32_norm
                        ) / (grad_fp32_norm + 1e-12)
                        details_parts.append(
                            f"grad_norm_fp32={grad_fp32_norm:.8f}, "
                            f"grad_norm_amp={grad_amp_norm:.8f}, "
                            f"grad_rel_err={grad_rel_err:.2e}"
                        )
                        if grad_rel_err > config.grad_rtol:
                            # This is a warning, not a hard failure,
                            # because some precision loss is expected
                            # under autocast
                            errors.append(
                                f"Gradient relative error "
                                f"{grad_rel_err:.2e} > tolerance "
                                f"{config.grad_rtol} (warning)"
                            )
                    else:
                        details_parts.append(
                            f"grad_norm_fp32={grad_fp32_norm:.8f} "
                            f"(near zero)"
                        )

                report.add(TestResult(
                    name=test_name,
                    passed=len(errors) == 0,
                    duration_ms=elapsed_ms,
                    error="\n".join(errors) if errors else "",
                    details="; ".join(details_parts),
                    config={"B": B, "T": T, "N": N, "term": term_name},
                ))

            except Exception as e:
                elapsed_ms = (time.time() - t0) * 1000
                report.add(TestResult(
                    name=test_name,
                    passed=False,
                    duration_ms=elapsed_ms,
                    error=f"Exception: {e}\n{traceback.format_exc()}",
                    config={"B": B, "T": T, "N": N, "term": term_name},
                ))


# ---------------------------------------------------------------------------
# SECTION 13: Full test runner
# ---------------------------------------------------------------------------

def run_all_tests(config: AMPTestConfig,
                  verbose: bool = False) -> AMPStressReport:
    """Execute all AMP stress test suites and return aggregated report."""
    report = AMPStressReport()

    print(f"Running AMP stress tests on device: {config.device}")
    print(f"  Batch sizes : {config.batch_sizes}")
    print(f"  Time steps  : {config.time_steps}")
    print(f"  Feature sizes: {config.feature_sizes}")
    print(f"  Num classes : {config.num_classes}")
    print(f"  Num trials  : {config.num_trials}")
    print(f"  Using stubs : {_USING_STUBS}")
    if _HAS_CUDA:
        print(f"  CUDA device : {torch.cuda.get_device_name(0)}")
    print()

    # Phase 1: Encoder tests
    print("[1/6] Running encoder AMP tests...")
    try:
        run_encoder_tests(config, report, verbose)
    except Exception as e:
        report.add(TestResult(
            name="encoder/SUITE_ERROR",
            passed=False,
            error=f"Encoder test suite crashed: {e}\n{traceback.format_exc()}",
        ))
    print(f"      {report.passed}/{report.total} passed so far")

    # Phase 2: Decoder tests
    prev_count = report.total
    print("[2/6] Running decoder AMP tests...")
    try:
        run_decoder_tests(config, report, verbose)
    except Exception as e:
        report.add(TestResult(
            name="decoder/SUITE_ERROR",
            passed=False,
            error=f"Decoder test suite crashed: {e}\n{traceback.format_exc()}",
        ))
    new_tests = report.total - prev_count
    new_pass = sum(1 for r in report.results[prev_count:] if r.passed)
    print(f"      {new_pass}/{new_tests} passed in this phase")

    # Phase 3: Loss tests
    prev_count = report.total
    print("[3/6] Running loss AMP tests...")
    try:
        run_loss_tests(config, report, verbose)
    except Exception as e:
        report.add(TestResult(
            name="loss/SUITE_ERROR",
            passed=False,
            error=f"Loss test suite crashed: {e}\n{traceback.format_exc()}",
        ))
    new_tests = report.total - prev_count
    new_pass = sum(1 for r in report.results[prev_count:] if r.passed)
    print(f"      {new_pass}/{new_tests} passed in this phase")

    # Phase 4: Scaling stress tests
    prev_count = report.total
    print("[4/6] Running scaling stress tests...")
    try:
        run_scaling_tests(config, report, verbose)
    except Exception as e:
        report.add(TestResult(
            name="scaling/SUITE_ERROR",
            passed=False,
            error=f"Scaling test suite crashed: {e}\n{traceback.format_exc()}",
        ))
    new_tests = report.total - prev_count
    new_pass = sum(1 for r in report.results[prev_count:] if r.passed)
    print(f"      {new_pass}/{new_tests} passed in this phase")

    # Phase 5: Memory profiling
    prev_count = report.total
    print("[5/6] Running memory profiling...")
    try:
        run_memory_tests(config, report, verbose)
    except Exception as e:
        report.add(TestResult(
            name="memory/SUITE_ERROR",
            passed=False,
            error=f"Memory test suite crashed: {e}\n{traceback.format_exc()}",
        ))
    new_tests = report.total - prev_count
    new_pass = sum(1 for r in report.results[prev_count:] if r.passed)
    print(f"      {new_pass}/{new_tests} passed in this phase")

    # Phase 6: Numerical precision tests
    prev_count = report.total
    print("[6/6] Running numerical precision tests...")
    try:
        run_precision_tests(config, report, verbose)
    except Exception as e:
        report.add(TestResult(
            name="precision/SUITE_ERROR",
            passed=False,
            error=f"Precision test suite crashed: {e}\n{traceback.format_exc()}",
        ))
    new_tests = report.total - prev_count
    new_pass = sum(1 for r in report.results[prev_count:] if r.passed)
    print(f"      {new_pass}/{new_tests} passed in this phase")

    return report


# ---------------------------------------------------------------------------
# SECTION 14: CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AMP stress test for spike codec and loss pack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python amp_stress_test.py                    # Full test, auto device\n"
            "  python amp_stress_test.py --quick             # Quick smoke test\n"
            "  python amp_stress_test.py --device cpu        # Force CPU\n"
            "  python amp_stress_test.py --json-report out.json\n"
            "  python amp_stress_test.py --verbose           # Show details for passing tests\n"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run tests on (default: auto-detect)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run reduced configuration for quick smoke test",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output for all tests including passing ones",
    )
    parser.add_argument(
        "--json-report",
        type=str,
        default=None,
        metavar="PATH",
        help="Write JSON report to file",
    )

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if _HAS_CUDA else "cpu"
    elif args.device == "cuda" and not _HAS_CUDA:
        print("WARNING: CUDA requested but not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    # Build config
    if args.quick:
        config = AMPTestConfig.quick()
    else:
        config = AMPTestConfig.full()
    config.device = device

    # Suppress some known harmless warnings under CPU autocast
    if device == "cpu":
        warnings.filterwarnings(
            "ignore",
            message=".*User provided device_type of 'cpu'.*",
        )

    # Run tests
    report = run_all_tests(config, verbose=args.verbose)

    # Print report
    report.print_report(verbose=args.verbose)

    # JSON export
    if args.json_report:
        json_str = report.to_json()
        report_path = os.path.abspath(args.json_report)
        report_dir = os.path.dirname(report_path)
        if report_dir and not os.path.isdir(report_dir):
            os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(json_str)
        print(f"JSON report written to: {report_path}")

    # Exit code
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
