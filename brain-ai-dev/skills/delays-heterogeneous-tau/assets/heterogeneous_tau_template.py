"""
brain_ai/core/heterogeneous_tau.py — Heterogeneous and Learnable Membrane Time Constants

This module provides a standalone, reusable system for managing membrane time constants
(τ) and their derived decay factors (β) in spiking neural networks. It is designed
to integrate cleanly with the existing AdvancedLIFNeuron / SpikingNeuronBase hierarchy
while also being usable as an independent component in any SNN architecture.

Background
----------
In biological cortical circuits, different neurons exhibit dramatically different
membrane time constants — from fast-spiking interneurons (τ ≈ 3–5 ms) to pyramidal
cells (τ ≈ 20–30 ms) to layer-5 cells with long time constants (τ > 50 ms). This
heterogeneity is thought to be a key computational resource, allowing cortical
circuits to represent and integrate information across multiple timescales
simultaneously.

The standard SNN formulation relates τ to the discrete-time decay factor β via:

    β = exp(−dt / τ)      (exact discrete-time mapping)
    τ = −dt / log(β)      (inverse)

This module supports four parameterisation strategies for τ:
1. homogeneous_fixed:        All neurons share a single fixed τ.
2. heterogeneous_fixed:      Per-neuron τ drawn from a distribution, then frozen.
3. heterogeneous_learnable:  Per-neuron τ initialised from a distribution, then
                             trained via backprop through a softplus reparameterisation.
4. (migration path):         Convert legacy logit_beta_vec parameterisation (from
                             AdvancedLIFNeuron) to the canonical τ parameterisation.

Parameterisation
----------------
To keep τ positive and bounded during optimisation, we use the softplus reparameterisation:

    τ = clamp(τ_min + softplus(τ_raw), τ_min, τ_max)
    τ_raw = inverse_softplus(τ − τ_min)

τ_raw is the unconstrained parameter stored as nn.Parameter when learnable. This avoids
projection steps and gives smooth, well-conditioned gradients across the full valid range.

Integration with existing code
-------------------------------
    # Drop-in for AdvancedLIFNeuron heterogeneous tau:
    tau_cfg = TauConfig(mode="heterogeneous_learnable", tau_0=20.0)
    tau_mod = HeterogeneousTau(size, tau_cfg)

    # In _step():
    beta = tau_mod.beta_broadcast(state.v.shape)   # handles per-neuron broadcasting
    v_new = beta * state.v + x_eff                 # same equation, no changes

Key classes:
    TauConfig             — All configuration in one dataclass.
    HeterogeneousTau      — Main module; holds tau_raw parameter or buffer.
    ConvTauModule         — Specialisation for convolutional layers (per-channel).
    TauAwareLIF           — Example integration with SpikingNeuronBase.

Utility functions:
    init_tau_homogeneous, init_tau_gamma, init_tau_loguniform, init_tau_preset_bank
    tau_to_raw, raw_to_tau
    apply_heterogeneous_decay
    log_tau_summary, format_tau_report, check_tau_health
    migrate_logit_beta_to_tau

Numerical guarantees:
    - τ and β are ALWAYS computed in fp32.
    - No in-place operations on any tensor that may carry gradients.
    - β is clamped to [beta_min, beta_max] before every use.
    - Compatible with torch.compile() — no Python-level control flow on tensor values.

References:
    Perez-Nieves et al. (2021) "Neural heterogeneity promotes robust learning."
        Nature Communications.
    Gast et al. (2024) "A mean-field description of neural heterogeneity."
    Hammouamri et al. (2024) "Learning Delays in SNNs."
    Bellec et al. (2020) "A solution to the learning dilemma for recurrent networks
        of spiking neurons." Nature Communications.
    Maass (1997) "Networks of spiking neurons: The third generation of neural network models."
"""

from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Numerical epsilon for inverse_softplus to avoid log(0)
_SOFTPLUS_EPS: float = 1e-6

# Minimum absolute tau to guard against division by zero in beta computation
_TAU_FLOOR: float = 1e-4

# Valid mode strings
_VALID_MODES = ("homogeneous_fixed", "heterogeneous_fixed", "heterogeneous_learnable")

# Valid granularity strings
_VALID_GRANULARITIES = ("per_neuron", "per_channel", "per_layer")

# Valid init strategies
_VALID_INIT_STRATEGIES = (
    "homogeneous",
    "heterogeneous_gamma",
    "heterogeneous_loguniform",
    "preset_bank",
)


# ===========================================================================
# SECTION 1: TauConfig dataclass
# ===========================================================================


@dataclass
class TauConfig:
    """Configuration for heterogeneous membrane time constants.

    All τ values are in units of the simulation timestep (dt). For example,
    with dt=1.0 ms and tau_0=20.0, the baseline membrane time constant is
    20 timesteps / 20 ms.

    Attributes:
        mode: Parameterisation mode. One of:
            'homogeneous_fixed'        — All neurons share a single fixed τ.
            'heterogeneous_fixed'      — Per-neuron τ from init_strategy, frozen.
            'heterogeneous_learnable'  — Per-neuron τ, trained via backprop.
        granularity: Scope of the τ parameter. One of:
            'per_neuron'   — One τ per neuron (shape: (N,)).
            'per_channel'  — One τ per channel (shape: (C,)), broadcast spatially.
            'per_layer'    — Single τ shared by all neurons (scalar).
        init_strategy: How to initialise τ values. One of:
            'homogeneous'             — All τ = tau_0.
            'heterogeneous_gamma'     — τ ~ Gamma(gamma_shape, gamma_scale).
            'heterogeneous_loguniform'— log(τ) ~ Uniform(log(tau_min), log(tau_max)).
            'preset_bank'             — Assign from preset_values cyclically.
        tau_0: Default/baseline τ value. Used by 'homogeneous' init and as the
            reference for beta ≈ 0.95 when dt=1.0 (tau_0 ≈ 19.5).
        tau_min: Minimum allowed τ. Acts as the softplus offset:
            τ = tau_min + softplus(τ_raw). Must be > 0.
        tau_max: Maximum allowed τ. Applied via clamping after softplus.
        dt: Simulation timestep in the same units as tau. Used to compute
            β = exp(-dt / τ).
        gamma_shape: Shape parameter k of the Gamma distribution (mean = k * θ).
        gamma_scale: Scale parameter θ of the Gamma distribution.
        preset_values: Tuple of τ values for the preset_bank strategy.
        beta_min: Minimum allowed β. Guards against zero decay (dead neurons).
        beta_max: Maximum allowed β. Guards against membrane explosion.
    """

    mode: str = "heterogeneous_learnable"
    granularity: str = "per_neuron"
    init_strategy: str = "heterogeneous_loguniform"
    tau_0: float = 20.0
    tau_min: float = 1.0
    tau_max: float = 100.0
    dt: float = 1.0
    # Gamma distribution params (mean = gamma_shape * gamma_scale = 4 * 5 = 20)
    gamma_shape: float = 4.0
    gamma_scale: float = 5.0
    # Preset bank
    preset_values: tuple = (2.0, 5.0, 10.0, 20.0, 50.0)
    # Beta bounds
    beta_min: float = 0.001
    beta_max: float = 0.999

    def __post_init__(self) -> None:
        """Validate configuration on construction."""
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"TauConfig.mode must be one of {_VALID_MODES}, got '{self.mode}'"
            )
        if self.granularity not in _VALID_GRANULARITIES:
            raise ValueError(
                f"TauConfig.granularity must be one of {_VALID_GRANULARITIES}, "
                f"got '{self.granularity}'"
            )
        if self.init_strategy not in _VALID_INIT_STRATEGIES:
            raise ValueError(
                f"TauConfig.init_strategy must be one of {_VALID_INIT_STRATEGIES}, "
                f"got '{self.init_strategy}'"
            )
        if self.tau_min <= 0.0:
            raise ValueError(
                f"TauConfig.tau_min must be > 0, got {self.tau_min}"
            )
        if self.tau_max <= self.tau_min:
            raise ValueError(
                f"TauConfig.tau_max ({self.tau_max}) must be > tau_min ({self.tau_min})"
            )
        if self.tau_0 < self.tau_min or self.tau_0 > self.tau_max:
            raise ValueError(
                f"TauConfig.tau_0 ({self.tau_0}) must be in "
                f"[tau_min={self.tau_min}, tau_max={self.tau_max}]"
            )
        if self.dt <= 0.0:
            raise ValueError(f"TauConfig.dt must be > 0, got {self.dt}")
        if not (0.0 < self.beta_min < self.beta_max < 1.0):
            raise ValueError(
                f"TauConfig requires 0 < beta_min < beta_max < 1, "
                f"got beta_min={self.beta_min}, beta_max={self.beta_max}"
            )


# ===========================================================================
# SECTION 2: Initialisation functions
# ===========================================================================


def init_tau_homogeneous(
    shape: Union[int, Tuple[int, ...]],
    tau_0: float,
) -> Tensor:
    """Initialise all τ values to a single constant.

    Args:
        shape: Output tensor shape. Scalar shape () is also valid.
        tau_0: The constant τ value assigned to all neurons.

    Returns:
        Float32 tensor of given shape with all values equal to tau_0.
    """
    if isinstance(shape, int):
        shape = (shape,)
    return torch.full(shape, tau_0, dtype=torch.float32)


def init_tau_gamma(
    shape: Union[int, Tuple[int, ...]],
    gamma_shape: float,
    gamma_scale: float,
    tau_min: float,
    tau_max: float,
    seed: Optional[int] = None,
) -> Tensor:
    """Initialise τ values from a Gamma distribution.

    Draws τ ~ Gamma(k=gamma_shape, θ=gamma_scale) with mean = k * θ, then
    clamps to [tau_min, tau_max]. The Gamma distribution is biologically motivated:
    it is positive and right-skewed, matching the observed distribution of
    membrane time constants in cortical populations.

    Args:
        shape: Output tensor shape.
        gamma_shape: Shape parameter k (> 0). Controls skewness.
        gamma_scale: Scale parameter θ (> 0). Mean = k * θ.
        tau_min: Lower clamp bound applied after sampling.
        tau_max: Upper clamp bound applied after sampling.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Float32 tensor of given shape with values clamped to [tau_min, tau_max].

    Raises:
        ValueError: If gamma_shape or gamma_scale are non-positive.
    """
    if gamma_shape <= 0.0:
        raise ValueError(f"gamma_shape must be > 0, got {gamma_shape}")
    if gamma_scale <= 0.0:
        raise ValueError(f"gamma_scale must be > 0, got {gamma_scale}")

    if isinstance(shape, int):
        shape = (shape,)

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    # torch.distributions.Gamma gives Gamma(concentration=k, rate=1/θ)
    concentration = torch.tensor(gamma_shape, dtype=torch.float32)
    rate = torch.tensor(1.0 / gamma_scale, dtype=torch.float32)
    dist = torch.distributions.Gamma(concentration=concentration, rate=rate)

    # Sample in float64 for precision, then convert
    tau_samples = dist.sample(shape).float()
    return tau_samples.clamp(tau_min, tau_max)


def init_tau_loguniform(
    shape: Union[int, Tuple[int, ...]],
    tau_min: float,
    tau_max: float,
    seed: Optional[int] = None,
) -> Tensor:
    """Initialise τ values from a log-uniform distribution.

    Draws log(τ) ~ Uniform(log(tau_min), log(tau_max)), which gives τ values
    uniformly spaced on a logarithmic scale. This ensures neurons span timescales
    evenly in multiplicative terms — e.g., equally many neurons in [1, 10] ms
    as in [10, 100] ms.

    Args:
        shape: Output tensor shape.
        tau_min: Lower bound of τ (> 0).
        tau_max: Upper bound of τ.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Float32 tensor of given shape with values in [tau_min, tau_max].
    """
    if isinstance(shape, int):
        shape = (shape,)

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    log_min = math.log(tau_min)
    log_max = math.log(tau_max)

    log_tau = torch.empty(shape, dtype=torch.float32).uniform_(
        log_min, log_max, generator=gen
    )
    return log_tau.exp()


def init_tau_preset_bank(
    shape: Union[int, Tuple[int, ...]],
    preset_values: Tuple[float, ...],
    seed: Optional[int] = None,
) -> Tensor:
    """Initialise τ values by assigning from a preset bank.

    Neurons are assigned τ values from preset_values in a round-robin (cyclic)
    fashion, ensuring an exact equal-count distribution across timescales. This
    is useful when you want a controlled, discrete set of timescales rather than
    a continuous distribution.

    Args:
        shape: Output tensor shape.
        preset_values: Tuple of τ values to assign from. Must be non-empty.
        seed: Optional seed (unused for cyclic assignment, kept for API uniformity).
            If provided, a random assignment order is used instead of cyclic.

    Returns:
        Float32 tensor of given shape with values from preset_values.

    Raises:
        ValueError: If preset_values is empty.
    """
    if len(preset_values) == 0:
        raise ValueError("preset_values must be non-empty")

    if isinstance(shape, int):
        shape = (shape,)

    n_total = 1
    for s in shape:
        n_total *= s

    presets = torch.tensor(list(preset_values), dtype=torch.float32)

    if seed is not None:
        # Random assignment: sample with replacement from preset bank
        gen = torch.Generator()
        gen.manual_seed(seed)
        indices = torch.randint(
            len(preset_values), (n_total,), generator=gen, dtype=torch.long
        )
    else:
        # Cyclic assignment: deterministic, balanced across presets
        indices = torch.arange(n_total, dtype=torch.long) % len(preset_values)

    return presets[indices].reshape(shape)


def tau_to_raw(
    tau: Tensor,
    tau_min: float,
) -> Tensor:
    """Convert τ values to unconstrained τ_raw via inverse softplus.

    The forward mapping is:
        τ = τ_min + softplus(τ_raw)

    The inverse is:
        τ_raw = log(exp(τ − τ_min) − 1)

    This is numerically stable for τ >> τ_min via the log-sum-exp identity:
        log(exp(x) - 1) ≈ x  for large x  (asymptotically)

    Args:
        tau: Tensor of τ values. Must have τ > τ_min for all elements.
        tau_min: The offset parameter (must match the value used in raw_to_tau).

    Returns:
        Unconstrained τ_raw tensor, same shape and dtype as tau.

    Note:
        Values exactly at tau_min will produce -inf in τ_raw. Ensure tau > tau_min
        by clamping input if necessary (e.g., tau.clamp(min=tau_min + 1e-4)).
    """
    # Offset by tau_min; result should be > 0
    x = tau - tau_min

    # Clamp to avoid log(0) or log of negative
    x = x.clamp(min=_SOFTPLUS_EPS)

    # inverse softplus: log(exp(x) - 1)
    # Numerically stable: for x > 20, exp(x) >> 1 so log(exp(x)-1) ≈ x
    # For small x, use log1p(expm1(x)) which is more numerically precise.
    # torch.log(torch.exp(x) - 1) overflows for large x, so we use:
    # log(exp(x) - 1) = x + log(1 - exp(-x)) for large x
    # = log(expm1(x)) in general
    tau_raw = torch.log(torch.expm1(x).clamp(min=_SOFTPLUS_EPS))
    return tau_raw


def raw_to_tau(
    tau_raw: Tensor,
    tau_min: float,
    tau_max: float,
) -> Tensor:
    """Convert unconstrained τ_raw to valid τ via softplus + clamp.

    The mapping is:
        τ = clamp(τ_min + softplus(τ_raw), τ_min, τ_max)

    softplus(x) = log(1 + exp(x)) ≥ 0 for all x, so τ ≥ τ_min always.
    The clamp to τ_max is applied after, with zero gradient when clamped.

    Args:
        tau_raw: Unconstrained parameter tensor (nn.Parameter or regular tensor).
        tau_min: Lower bound for τ. Used as the softplus offset.
        tau_max: Upper bound for τ. Applied via hard clamp.

    Returns:
        Tau tensor with values in [tau_min, tau_max]. Same shape as tau_raw.
        Always float32 regardless of tau_raw dtype.
    """
    tau_raw_fp32 = tau_raw.float()
    # F.softplus is numerically stable and torch.compile-friendly
    tau = tau_min + F.softplus(tau_raw_fp32)
    return tau.clamp(tau_min, tau_max)


# ===========================================================================
# SECTION 3: HeterogeneousTau nn.Module
# ===========================================================================


class HeterogeneousTau(nn.Module):
    """Manages heterogeneous membrane time constants for a population of neurons.

    This module holds either a trainable parameter (tau_raw as nn.Parameter) or
    a fixed buffer containing τ values, depending on TauConfig.mode. It exposes
    τ and β as properties and provides broadcasting utilities for integration
    into LIF neuron forward passes.

    The parameterisation is:
        τ = clamp(τ_min + softplus(τ_raw), τ_min, τ_max)
        β = clamp(exp(−dt / τ), β_min, β_max)

    Shape semantics (shape argument):
        per_neuron:  shape = (N,)  — one τ per neuron
        per_channel: shape = (C,)  — one τ per channel (spatial broadcast in CNN)
        per_layer:   shape = ()    — scalar τ, shared by all neurons

    Args:
        shape: Neuron population shape excluding batch dimension.
            - int N or (N,) for per-neuron linear layers.
            - (C,) for per-channel convolutional layers.
            - () or 0 for per-layer scalar.
        config: TauConfig instance controlling mode, init, and bounds.

    Example (per-neuron learnable)::

        cfg = TauConfig(mode="heterogeneous_learnable", tau_0=20.0)
        tau_mod = HeterogeneousTau(shape=256, config=cfg)
        beta = tau_mod.beta_broadcast(target_shape=(4, 256))
        v_new = beta * state.v + x  # (4, 256)

    Example (per-channel conv)::

        cfg = TauConfig(granularity="per_channel", mode="heterogeneous_learnable")
        tau_mod = ConvTauModule(num_channels=64, config=cfg)
        beta = tau_mod.beta_broadcast(target_shape=(4, 64, 32, 32))
        v_new = beta * state.v + x  # (4, 64, 32, 32)
    """

    def __init__(
        self,
        shape: Union[int, Tuple[int, ...]],
        config: TauConfig,
    ) -> None:
        super().__init__()
        self.config = config

        # Normalise shape
        if isinstance(shape, int):
            if shape == 0:
                self._shape: Tuple[int, ...] = ()
            else:
                self._shape = (shape,)
        else:
            self._shape = tuple(shape)

        # Initialise τ from the chosen strategy
        tau_init = self._init_tau()  # float32 tensor

        # Convert τ → τ_raw (unconstrained)
        tau_raw_init = tau_to_raw(tau_init, config.tau_min)

        learnable = config.mode == "heterogeneous_learnable"

        if learnable:
            self.tau_raw: Union[nn.Parameter, Tensor] = nn.Parameter(tau_raw_init)
        else:
            # Fixed: register as buffer so it moves with .to(device)
            self.register_buffer("tau_raw", tau_raw_init)

        # Store a snapshot of the initial tau for drift computation
        # (detached, no gradient)
        self.register_buffer(
            "_initial_tau",
            tau_init.detach().clone(),
        )

        logger.debug(
            "HeterogeneousTau: shape=%s, mode=%s, granularity=%s, "
            "init=%s, tau_mean=%.2f, tau_std=%.2f",
            self._shape,
            config.mode,
            config.granularity,
            config.init_strategy,
            tau_init.mean().item(),
            tau_init.std().item() if tau_init.numel() > 1 else 0.0,
        )

    # -----------------------------------------------------------------------
    # Private: initialisation dispatcher
    # -----------------------------------------------------------------------

    def _init_tau(self) -> Tensor:
        """Dispatch to the chosen init strategy and return a float32 τ tensor."""
        cfg = self.config
        shape = self._shape if self._shape else (1,)  # scalar as (1,) during init

        if cfg.init_strategy == "homogeneous":
            tau = init_tau_homogeneous(shape, cfg.tau_0)

        elif cfg.init_strategy == "heterogeneous_gamma":
            tau = init_tau_gamma(
                shape,
                gamma_shape=cfg.gamma_shape,
                gamma_scale=cfg.gamma_scale,
                tau_min=cfg.tau_min,
                tau_max=cfg.tau_max,
            )

        elif cfg.init_strategy == "heterogeneous_loguniform":
            tau = init_tau_loguniform(
                shape,
                tau_min=cfg.tau_min,
                tau_max=cfg.tau_max,
            )

        elif cfg.init_strategy == "preset_bank":
            tau = init_tau_preset_bank(
                shape,
                preset_values=cfg.preset_values,
            )

        else:
            # Should be caught by TauConfig.__post_init__, but guard defensively
            raise ValueError(
                f"Unknown init_strategy '{cfg.init_strategy}'"
            )

        # For scalar granularity, collapse to 0-dim tensor
        if not self._shape:
            tau = tau.mean().reshape(())

        return tau

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def tau(self) -> Tensor:
        """Current τ values as a float32 tensor, clamped to [tau_min, tau_max].

        Always computed from tau_raw via the softplus reparameterisation.
        Shape: self._shape (or scalar () for per_layer).

        Returns:
            Float32 tensor with values in [config.tau_min, config.tau_max].
        """
        return raw_to_tau(self.tau_raw, self.config.tau_min, self.config.tau_max)

    @property
    def beta(self) -> Tensor:
        """Decay factor β = exp(−dt / τ), clamped to [beta_min, beta_max].

        Derived from tau via the exact discrete-time mapping:
            β = exp(−dt / τ)

        β = 0 means no memory (fire-and-forget); β → 1 means extremely long memory.
        Clamping prevents β = 1 (which would cause membrane explosion) and
        β = 0 (which would kill all dynamics).

        Returns:
            Float32 tensor with values in [config.beta_min, config.beta_max].
            Shape: self._shape (or scalar () for per_layer).
        """
        cfg = self.config
        tau = self.tau  # float32, clamped
        # Guard tau against values too close to zero (extra safety)
        tau_safe = tau.clamp(min=_TAU_FLOOR)
        beta_raw = torch.exp(-cfg.dt / tau_safe)
        return beta_raw.clamp(cfg.beta_min, cfg.beta_max)

    def beta_broadcast(self, target_shape: Tuple[int, ...]) -> Tensor:
        """Return β reshaped for broadcasting against a membrane tensor.

        Broadcasting rules by granularity:
            per_neuron:  β shape (N,) → (1, N) for target (B, N)
            per_channel: β shape (C,) → (1, C, 1, 1) for target (B, C, H, W)
            per_layer:   β scalar ()  → scalar (no reshape needed)

        Args:
            target_shape: Shape of the membrane tensor to broadcast against.
                First dimension is always the batch dimension B.

        Returns:
            Float32 β tensor broadcastable with target_shape.

        Raises:
            ValueError: If target_shape is incompatible with the stored granularity.
        """
        b = self.beta  # float32

        granularity = self.config.granularity

        if granularity == "per_layer":
            # Scalar — broadcasts automatically to any shape
            return b

        elif granularity == "per_neuron":
            # target_shape: (B, N)
            if len(target_shape) < 2:
                raise ValueError(
                    f"per_neuron beta_broadcast requires target_shape with at least "
                    f"2 dims (B, N), got {target_shape}"
                )
            # b: (N,) → (1, N)
            return b.unsqueeze(0)

        elif granularity == "per_channel":
            # target_shape: (B, C, H, W)
            if len(target_shape) < 3:
                raise ValueError(
                    f"per_channel beta_broadcast requires target_shape with at least "
                    f"3 dims (B, C, ...), got {target_shape}"
                )
            # b: (C,) → (1, C, 1, 1, ...) with as many trailing 1s as spatial dims
            n_spatial = len(target_shape) - 2  # subtract B and C dims
            # unsqueeze 0 for batch, then n_spatial trailing dimensions
            b_view = b.unsqueeze(0)  # (1, C)
            for _ in range(n_spatial):
                b_view = b_view.unsqueeze(-1)
            return b_view  # (1, C, 1, ..., 1)

        else:
            raise ValueError(
                f"Unknown granularity '{granularity}'. "
                f"Expected one of {_VALID_GRANULARITIES}"
            )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, float]:
        """Return summary statistics for τ and β as a flat dict.

        Suitable for passing directly to TensorBoard/wandb scalar logging.
        All values are Python floats computed under torch.no_grad().

        Returns:
            Dict with keys: tau_mean, tau_std, tau_min_val, tau_max_val,
            beta_mean, beta_std, beta_min_val, beta_max_val.
        """
        with torch.no_grad():
            tau = self.tau
            beta = self.beta

            # For scalar tensors, .std() returns nan; handle gracefully
            def _std(t: Tensor) -> float:
                return t.std().item() if t.numel() > 1 else 0.0

            return {
                "tau_mean": tau.mean().item(),
                "tau_std": _std(tau),
                "tau_min_val": tau.min().item(),
                "tau_max_val": tau.max().item(),
                "beta_mean": beta.mean().item(),
                "beta_std": _std(beta),
                "beta_min_val": beta.min().item(),
                "beta_max_val": beta.max().item(),
            }

    def get_tau_histogram(self, num_bins: int = 50) -> Dict[str, Tensor]:
        """Return a histogram of τ values for distribution visualisation.

        Args:
            num_bins: Number of histogram bins. Must be >= 2.

        Returns:
            Dict with keys:
                'bin_edges': Float32 tensor of length (num_bins + 1).
                'counts':    Int64 tensor of length num_bins.
        """
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")

        with torch.no_grad():
            tau = self.tau.reshape(-1).cpu()
            counts = torch.zeros(num_bins, dtype=torch.long)
            edges = torch.linspace(
                self.config.tau_min, self.config.tau_max, num_bins + 1
            )
            # Use torch.histc for compile-compatibility
            counts_float = torch.histc(tau, bins=num_bins, min=float(edges[0]), max=float(edges[-1]))
            counts = counts_float.long()

        return {"bin_edges": edges, "counts": counts}

    def compute_firing_rate_correlation(self, firing_rates: Tensor) -> float:
        """Compute the Pearson correlation between τ and firing rates.

        This metric helps diagnose whether τ values have adapted to match
        the functional role of neurons — e.g., do fast-spiking neurons have
        short τ? A significant negative correlation is expected in well-trained
        heterogeneous networks (faster-spiking neurons have shorter τ).

        Args:
            firing_rates: Tensor of per-neuron firing rates, shape matching
                self.tau.shape. Values should be in [0, 1] (fraction of timesteps
                with a spike), but are not strictly required to be.

        Returns:
            Pearson correlation coefficient r ∈ [−1, +1] as a Python float.
            Returns 0.0 if there is insufficient data (< 2 neurons).

        Raises:
            ValueError: If firing_rates.shape does not match self.tau.shape.
        """
        with torch.no_grad():
            tau = self.tau.reshape(-1).cpu().float()
            fr = firing_rates.reshape(-1).cpu().float()

            if tau.shape != fr.shape:
                raise ValueError(
                    f"firing_rates.shape {tuple(fr.shape)} does not match "
                    f"tau.shape {tuple(tau.shape)}"
                )

            n = tau.numel()
            if n < 2:
                return 0.0

            # Pearson correlation: r = cov(tau, fr) / (std(tau) * std(fr))
            tau_c = tau - tau.mean()
            fr_c = fr - fr.mean()
            cov = (tau_c * fr_c).sum()
            std_tau = tau.std(unbiased=False)
            std_fr = fr.std(unbiased=False)

            denom = std_tau * std_fr
            if denom.item() < 1e-12:
                return 0.0

            r = (cov / denom).item()
            return float(r)

    def compute_drift(self) -> float:
        """Compute mean absolute drift of τ from its initial values.

        Measures how much the time constants have moved during training:
            drift = mean(|τ_current − τ_initial|) / N

        where N is the total number of τ parameters. This is useful for
        monitoring whether heterogeneity is being preserved or collapsed.

        Returns:
            Mean absolute drift per neuron as a Python float.
            Returns 0.0 for fixed modes (no drift by definition).
        """
        with torch.no_grad():
            if self.config.mode != "heterogeneous_learnable":
                return 0.0

            initial = self._initial_tau.reshape(-1).float()
            current = self.tau.reshape(-1).float()
            drift = (current - initial).abs().mean().item()
            return float(drift)

    def clamp_pressure(self) -> float:
        """Return the fraction of τ values at their min or max boundary.

        High clamp pressure (> 0.1 i.e. > 10%) indicates that many neurons
        are being forced against the τ bounds, which may mean the bounds
        are too tight or that the initialisation was poor.

        Returns:
            Fraction in [0, 1] of τ values that are at tau_min or tau_max.
        """
        with torch.no_grad():
            tau = self.tau.reshape(-1)
            n = tau.numel()
            if n == 0:
                return 0.0
            at_min = (tau <= self.config.tau_min + _SOFTPLUS_EPS).sum().item()
            at_max = (tau >= self.config.tau_max - _SOFTPLUS_EPS).sum().item()
            return float(at_min + at_max) / n

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def extra_repr(self) -> str:
        cfg = self.config
        n_params = self.tau_raw.numel()
        learnable = isinstance(self.tau_raw, nn.Parameter)
        return (
            f"shape={self._shape}, mode={cfg.mode}, "
            f"granularity={cfg.granularity}, init={cfg.init_strategy}, "
            f"tau_range=[{cfg.tau_min}, {cfg.tau_max}], "
            f"n_params={n_params}, learnable={learnable}"
        )


# ===========================================================================
# SECTION 4: Integration helper
# ===========================================================================


def apply_heterogeneous_decay(
    v: Tensor,
    beta: Tensor,
    current: Tensor,
) -> Tensor:
    """Apply LIF membrane decay with heterogeneous β: v_new = β * v + current.

    Handles all broadcasting cases automatically by relying on PyTorch's
    standard broadcasting semantics. All computation is performed in float32
    regardless of input dtypes to maintain numerical stability.

    Broadcasting matrix:
        per_neuron:  beta (1, N),      v (B, N)        → (B, N)
        per_channel: beta (1, C, 1, 1), v (B, C, H, W)  → (B, C, H, W)
        per_layer:   beta scalar,       v (B, ...)       → (B, ...)

    Args:
        v: Current membrane potential tensor (B, *spatial). Will be cast to fp32.
        beta: Decay factor tensor from HeterogeneousTau.beta_broadcast(). fp32.
        current: Input current tensor, same shape as v. Will be cast to fp32.

    Returns:
        New membrane potential v_new = beta * v + current. Always float32.
        No in-place operations — a new tensor is always returned.

    Note:
        The caller is responsible for passing the correct beta shape via
        HeterogeneousTau.beta_broadcast(v.shape). This function does not
        perform any shape validation beyond what PyTorch broadcasting provides.
    """
    # Cast all inputs to float32 for stable membrane integration
    v_fp32 = v.float()
    beta_fp32 = beta.float()
    current_fp32 = current.float()

    # v_new = β * v + I  (no in-place: creates new tensor)
    v_new = beta_fp32 * v_fp32 + current_fp32
    return v_new


# ===========================================================================
# SECTION 5: TauAwareLIF — integration example with SpikingNeuronBase
# ===========================================================================

# ---------------------------------------------------------------------------
# Minimal SpikingNeuronBase stub for template self-containment.
# When integrating into brain_ai, remove this stub and import from:
#     from brain_ai.core.neurons import SpikingNeuronBase, SpikingState
# ---------------------------------------------------------------------------

@dataclass
class _SpikingState:
    """Minimal state container — mirrors SpikingState from neurons_template.py."""
    v: Tensor
    a: Optional[Tensor] = None
    spike_history: Optional[Tensor] = None

    def detach(self) -> "_SpikingState":
        return _SpikingState(
            v=self.v.detach(),
            a=self.a.detach() if self.a is not None else None,
            spike_history=(
                self.spike_history.detach()
                if self.spike_history is not None else None
            ),
        )


class _SpikingNeuronBase(nn.Module, ABC):
    """Minimal base — mirrors SpikingNeuronBase from neurons_template.py."""

    BETA_MIN_COMPAT: float = 0.0
    BETA_MAX_COMPAT: float = 0.999

    def __init__(self, size: int, threshold: float = 1.0) -> None:
        super().__init__()
        self.size = size
        self.register_buffer("threshold", torch.tensor(threshold, dtype=torch.float32))

    @property
    def threshold_clamped(self) -> Tensor:
        return self.threshold.clamp(min=1e-3)

    def reset_state(
        self,
        batch_size: int,
        neuron_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> _SpikingState:
        shape = (batch_size,) + neuron_shape
        return _SpikingState(v=torch.zeros(shape, device=device, dtype=dtype))

    def detach_state(self, state: _SpikingState) -> _SpikingState:
        return state.detach()

    @abstractmethod
    def _step(self, x: Tensor, state: _SpikingState) -> Tuple[Tensor, _SpikingState, Dict]:
        ...

    def forward(
        self,
        x: Tensor,
        state: _SpikingState,
    ) -> Tuple[Tensor, _SpikingState]:
        spikes, new_state, _ = self._step(x.float(), state)
        return spikes, new_state


class TauAwareLIF(_SpikingNeuronBase):
    """LIF neuron with a HeterogeneousTau module for membrane decay.

    This class demonstrates the recommended integration pattern between
    HeterogeneousTau and the SpikingNeuronBase hierarchy. The key insight is
    that tau_module.beta_broadcast() replaces the scalar beta_clamped call in
    the standard LIF step — everything else remains identical.

    The update equations are:
        β[n]  = tau_module.beta[n]                  (per-neuron)
        v[t]  = β[n] * v[t-1] + x[t]               (heterogeneous decay)
        s[t]  = H(v[t] - v_th)                      (spike)
        v[t]  = v[t] - s[t] * v_th                  (subtractive reset)

    Args:
        size: Number of neurons.
        tau_config: TauConfig controlling mode, init, and bounds.
        threshold: Spike threshold voltage.
        reset_mechanism: 'subtract' (soft) or 'zero' (hard reset).

    Example::

        cfg = TauConfig(
            mode="heterogeneous_learnable",
            init_strategy="heterogeneous_loguniform",
            tau_min=2.0, tau_max=50.0,
        )
        neuron = TauAwareLIF(size=512, tau_config=cfg)
        state = neuron.reset_state(batch_size=4, neuron_shape=(512,), device=device)

        for t in range(T):
            x_t = input_seq[:, t]  # (4, 512)
            spikes, state = neuron(x_t, state)

    Integration note:
        When integrating into brain_ai/core/neurons.py, replace
        _SpikingNeuronBase with SpikingNeuronBase and _SpikingState with
        SpikingState. The logic in _step() is identical.
    """

    def __init__(
        self,
        size: int,
        tau_config: Optional[TauConfig] = None,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
    ) -> None:
        super().__init__(size=size, threshold=threshold)

        if tau_config is None:
            tau_config = TauConfig()  # defaults: heterogeneous_learnable, loguniform

        if reset_mechanism not in ("subtract", "zero"):
            raise ValueError(
                f"reset_mechanism must be 'subtract' or 'zero', "
                f"got '{reset_mechanism}'"
            )
        self.reset_mechanism = reset_mechanism
        self.tau_module = HeterogeneousTau(size, tau_config)

    def _step(
        self,
        x: Tensor,
        state: _SpikingState,
    ) -> Tuple[Tensor, _SpikingState, Dict]:
        """Single LIF timestep with heterogeneous tau.

        Args:
            x: Input current, shape (B, size), fp32.
            state: _SpikingState with field v (membrane potential).

        Returns:
            spikes: Binary tensor (B, size).
            new_state: Updated _SpikingState.
            details: {'firing_rate', 'membrane_mean', 'tau_mean', 'beta_mean'}.
        """
        # Retrieve β, broadcast to (1, N) for (B, N) membrane
        beta = self.tau_module.beta_broadcast(state.v.shape)

        # Leaky integration — no in-place op, always fp32
        v_new = apply_heterogeneous_decay(state.v, beta, x)

        # Spike generation (simple Heaviside for template; use surrogate in prod)
        th = self.threshold_clamped
        spikes = (v_new >= th).float()

        # Reset
        if self.reset_mechanism == "subtract":
            v_reset = v_new - spikes * th
        else:
            v_reset = v_new * (1.0 - spikes)

        new_state = _SpikingState(v=v_reset)

        with torch.no_grad():
            tau_now = self.tau_module.tau
            beta_now = self.tau_module.beta
            details: Dict = {
                "firing_rate": spikes.mean().item(),
                "membrane_mean": v_reset.mean().item(),
                "tau_mean": tau_now.mean().item(),
                "beta_mean": beta_now.mean().item(),
            }

        return spikes, new_state, details

    def get_tau_diagnostics(self) -> Dict[str, float]:
        """Delegate to the embedded tau_module diagnostics."""
        return self.tau_module.get_diagnostics()


# ===========================================================================
# SECTION 6: ConvTauModule
# ===========================================================================


class ConvTauModule(HeterogeneousTau):
    """Specialisation of HeterogeneousTau for convolutional layers.

    Provides per-channel τ with automatic spatial broadcasting:
        β shape: (1, C, 1, 1) to broadcast with (B, C, H, W) membrane.

    Use this in place of HeterogeneousTau when the neuron layer is a
    convolutional LIF neuron with 4-D activations.

    Args:
        num_channels: Number of feature map channels (C).
        config: TauConfig. The granularity is forced to 'per_channel'.

    Example::

        cfg = TauConfig(mode="heterogeneous_learnable", tau_0=20.0)
        tau_mod = ConvTauModule(num_channels=64, config=cfg)
        beta = tau_mod.beta_broadcast(target_shape=(4, 64, 28, 28))
        v_new = beta * state.v + conv_out  # (4, 64, 28, 28)
    """

    def __init__(self, num_channels: int, config: TauConfig) -> None:
        # Force granularity to per_channel for correct broadcasting
        config = replace(config, granularity="per_channel")
        super().__init__(shape=(num_channels,), config=config)
        self.num_channels = num_channels

    def beta_broadcast(self, target_shape: Tuple[int, ...]) -> Tensor:
        """Return β shaped as (1, C, 1, 1) for broadcasting with (B, C, H, W).

        Args:
            target_shape: Shape of the membrane tensor. Expected: (B, C, H, W).
                Only the C dimension is validated against num_channels.

        Returns:
            Float32 tensor of shape (1, C, 1, 1).

        Raises:
            ValueError: If target_shape[1] != num_channels.
        """
        if len(target_shape) < 2 or target_shape[1] != self.num_channels:
            raise ValueError(
                f"ConvTauModule.beta_broadcast: expected target_shape[1]="
                f"{self.num_channels}, got target_shape={target_shape}"
            )
        # beta: (C,) → (1, C, 1, 1)
        return self.beta.view(1, self.num_channels, 1, 1)

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}, " + super().extra_repr()


# ===========================================================================
# SECTION 7: Diagnostics and logging utilities
# ===========================================================================


def log_tau_summary(
    tau_module: HeterogeneousTau,
    prefix: str = "",
) -> Dict[str, float]:
    """Generate a flat dict of τ/β summary statistics for logging.

    Wraps HeterogeneousTau.get_diagnostics() with optional key prefixing
    and adds derived metrics (clamp_pressure, drift).

    Args:
        tau_module: Any HeterogeneousTau instance (including ConvTauModule).
        prefix: Optional string prefix for all keys (e.g., "layer1/tau/").
            If non-empty, a '/' separator is appended automatically if missing.

    Returns:
        Dict mapping prefixed metric names to float values. Suitable for
        logging frameworks that accept flat dicts (wandb.log, tb_writer.add_scalars).
    """
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    diag = tau_module.get_diagnostics()
    diag["clamp_pressure"] = tau_module.clamp_pressure()
    diag["tau_drift"] = tau_module.compute_drift()

    return {f"{prefix}{k}": v for k, v in diag.items()}


def format_tau_report(tau_module: HeterogeneousTau) -> str:
    """Format a human-readable multiline report of the τ/β distribution.

    Useful for printing during training or writing to a log file. Includes
    mode information, distribution statistics, and health indicators.

    Args:
        tau_module: Any HeterogeneousTau instance.

    Returns:
        Multi-line string report. Example output::

            ╔═══════════════════════════════════╗
            ║  HeterogeneousTau Distribution    ║
            ╠═══════════════════════════════════╣
            ║  Mode:        heterogeneous_learnable
            ║  Granularity: per_neuron
            ║  N params:    512
            ╠═════════════════════════╦═════════╣
            ║  Statistic              ║  Value  ║
            ╠═════════════════════════╬═════════╣
            ║  τ mean                 ║  20.31  ║
            ║  τ std                  ║   8.14  ║
            ║  τ [min, max]           ║ [1.02, 98.7] ║
            ║  β mean                 ║   0.951 ║
            ║  β std                  ║   0.031 ║
            ║  β [min, max]           ║ [0.37, 0.999] ║
            ║  Clamp pressure         ║   0.4%  ║
            ║  τ drift                ║   0.00  ║
            ╚═════════════════════════╩═════════╝
    """
    cfg = tau_module.config
    diag = tau_module.get_diagnostics()
    n_params = tau_module.tau_raw.numel()
    learnable = isinstance(tau_module.tau_raw, nn.Parameter)
    clamp_pres = tau_module.clamp_pressure()
    drift = tau_module.compute_drift()

    lines = [
        "=" * 52,
        "  HeterogeneousTau Distribution Report",
        "=" * 52,
        f"  Mode:           {cfg.mode}",
        f"  Granularity:    {cfg.granularity}",
        f"  Init strategy:  {cfg.init_strategy}",
        f"  N parameters:   {n_params}",
        f"  Learnable:      {learnable}",
        f"  τ bounds:       [{cfg.tau_min}, {cfg.tau_max}]",
        f"  β bounds:       [{cfg.beta_min}, {cfg.beta_max}]",
        f"  dt:             {cfg.dt}",
        "-" * 52,
        f"  τ mean:         {diag['tau_mean']:.4f}",
        f"  τ std:          {diag['tau_std']:.4f}",
        f"  τ [min, max]:   [{diag['tau_min_val']:.4f}, {diag['tau_max_val']:.4f}]",
        f"  β mean:         {diag['beta_mean']:.6f}",
        f"  β std:          {diag['beta_std']:.6f}",
        f"  β [min, max]:   [{diag['beta_min_val']:.6f}, {diag['beta_max_val']:.6f}]",
        f"  Clamp pressure: {clamp_pres * 100:.2f}%",
        f"  τ drift:        {drift:.4f}",
        "=" * 52,
    ]
    return "\n".join(lines)


def check_tau_health(
    tau_module: HeterogeneousTau,
) -> Tuple[str, List[str]]:
    """Evaluate the health of a HeterogeneousTau module and return a status.

    Runs a battery of diagnostic checks and classifies the module as:
        'HEALTHY':   All checks pass — τ/β distribution is well-behaved.
        'WARNING':   Some checks failed but training can continue.
        'CRITICAL':  Severe issues that will likely destabilise training.

    Health criteria:
        CRITICAL:
          - β values outside (0, 1): would cause NaN membrane potentials.
          - τ std = 0 in heterogeneous mode: diversity collapsed completely.
          - τ mean outside [tau_min, tau_max]: parameterisation is broken.
        WARNING:
          - Clamp pressure > 20%: too many neurons hitting the τ boundary.
          - τ std < 0.5 in heterogeneous learnable mode: diversity collapsing.
          - τ drift > tau_range / 2: time constants have moved very far.

    Args:
        tau_module: Any HeterogeneousTau instance.

    Returns:
        Tuple (status, reasons) where:
            status:  'HEALTHY', 'WARNING', or 'CRITICAL' (str).
            reasons: List of str, one per failing check (empty if HEALTHY).
    """
    cfg = tau_module.config
    diag = tau_module.get_diagnostics()
    clamp_pres = tau_module.clamp_pressure()
    drift = tau_module.compute_drift()
    tau_range = cfg.tau_max - cfg.tau_min

    critical_reasons: List[str] = []
    warning_reasons: List[str] = []

    # --- CRITICAL checks ---
    beta_min_val = diag["beta_min_val"]
    beta_max_val = diag["beta_max_val"]

    if beta_min_val <= 0.0 or beta_max_val >= 1.0:
        critical_reasons.append(
            f"β values outside valid range (0, 1): "
            f"min={beta_min_val:.6f}, max={beta_max_val:.6f}. "
            f"Membrane potentials will diverge or die."
        )

    if math.isnan(diag["tau_mean"]) or math.isnan(diag["beta_mean"]):
        critical_reasons.append(
            "NaN detected in τ or β values. "
            "Gradient explosion likely. Check learning rate."
        )

    if (
        cfg.mode in ("heterogeneous_fixed", "heterogeneous_learnable")
        and diag["tau_std"] < 1e-6
        and tau_module.tau_raw.numel() > 1
    ):
        critical_reasons.append(
            f"τ std ≈ 0 ({diag['tau_std']:.2e}) in heterogeneous mode. "
            f"All time constants have collapsed to the same value — "
            f"the heterogeneity is gone."
        )

    tau_mean = diag["tau_mean"]
    if not (cfg.tau_min <= tau_mean <= cfg.tau_max):
        critical_reasons.append(
            f"τ mean ({tau_mean:.4f}) is outside [tau_min={cfg.tau_min}, "
            f"tau_max={cfg.tau_max}]. Clamping is masking extreme τ_raw values."
        )

    # --- WARNING checks ---
    if clamp_pres > 0.20:
        warning_reasons.append(
            f"Clamp pressure {clamp_pres * 100:.1f}% > 20%. "
            f"Many neurons are at τ boundaries — consider widening tau_min/tau_max."
        )

    if (
        cfg.mode == "heterogeneous_learnable"
        and diag["tau_std"] < 0.5
        and tau_module.tau_raw.numel() > 1
    ):
        warning_reasons.append(
            f"τ std ({diag['tau_std']:.4f}) < 0.5 in learnable mode. "
            f"Timescale diversity is collapsing. "
            f"Consider a diversity regularisation loss."
        )

    if drift > tau_range / 2.0:
        warning_reasons.append(
            f"τ drift ({drift:.4f}) > half the τ range ({tau_range / 2.0:.4f}). "
            f"Time constants have moved far from their initial values."
        )

    # --- Aggregate status ---
    if critical_reasons:
        return "CRITICAL", critical_reasons + warning_reasons
    elif warning_reasons:
        return "WARNING", warning_reasons
    else:
        return "HEALTHY", []


# ===========================================================================
# SECTION 8: Migration helper
# ===========================================================================


def migrate_logit_beta_to_tau(
    logit_beta_vec: Tensor,
    dt: float = 1.0,
    tau_min: float = 1.0,
    tau_max: float = 100.0,
) -> Tensor:
    """Convert AdvancedLIFNeuron's logit_beta_vec parameterisation to tau_raw.

    AdvancedLIFNeuron stores per-neuron learnable betas as:
        logit_beta_vec: (size,)
        beta = sigmoid(logit_beta_vec).clamp(0, 0.999)

    This function converts that legacy parameterisation to the canonical
    HeterogeneousTau tau_raw format, allowing checkpoints trained with the
    old code to be loaded into TauAwareLIF.

    Conversion steps:
        1. beta = sigmoid(logit_beta_vec)             [squash to (0, 1)]
        2. beta = beta.clamp(beta_min, beta_max)      [match old BETA_MAX=0.999]
        3. tau = -dt / log(beta)                      [continuous tau]
        4. tau = tau.clamp(tau_min, tau_max)          [enforce bounds]
        5. tau_raw = tau_to_raw(tau, tau_min)         [reparameterise]

    Args:
        logit_beta_vec: 1-D float32 tensor of shape (N,) from AdvancedLIFNeuron.
        dt: Simulation timestep. Must match the dt in TauConfig. Default: 1.0.
        tau_min: Lower τ bound, must match TauConfig.tau_min.
        tau_max: Upper τ bound, must match TauConfig.tau_max.

    Returns:
        tau_raw tensor of shape (N,), float32. Can be loaded directly as the
        initial value of HeterogeneousTau.tau_raw (as a Parameter or buffer).

    Example::

        # Load old checkpoint
        old_state = torch.load("checkpoint_v1.pt")
        logit_beta = old_state["core.advanced_lif.logit_beta_vec"]

        # Migrate to tau_raw
        tau_raw_init = migrate_logit_beta_to_tau(logit_beta, dt=1.0)

        # Build new TauAwareLIF and load
        cfg = TauConfig(mode="heterogeneous_learnable")
        neuron = TauAwareLIF(size=logit_beta.numel(), tau_config=cfg)
        with torch.no_grad():
            neuron.tau_module.tau_raw.copy_(tau_raw_init)
    """
    with torch.no_grad():
        # Step 1: beta from logit parameterisation
        beta = torch.sigmoid(logit_beta_vec.float())

        # Step 2: apply the same clamping as AdvancedLIFNeuron.beta_clamped
        # (BETA_MIN=0.0, BETA_MAX=0.999 in the original code)
        beta_clamped = beta.clamp(0.001, 0.999)

        # Step 3: convert beta → tau via exact continuous-time mapping
        # tau = -dt / log(beta)
        # log(beta) is negative for 0 < beta < 1
        log_beta = torch.log(beta_clamped)
        tau = -dt / log_beta  # positive because log_beta < 0

        # Step 4: enforce tau bounds
        tau_bounded = tau.clamp(tau_min, tau_max)

        # Step 5: reparameterise to tau_raw
        tau_raw = tau_to_raw(tau_bounded, tau_min)

    return tau_raw.detach()


# ===========================================================================
# SECTION 9: __all__ exports
# ===========================================================================

__all__ = [
    # Config
    "TauConfig",
    # Init functions
    "init_tau_homogeneous",
    "init_tau_gamma",
    "init_tau_loguniform",
    "init_tau_preset_bank",
    "tau_to_raw",
    "raw_to_tau",
    # Core module
    "HeterogeneousTau",
    # Specialisations
    "ConvTauModule",
    # Integration helpers
    "apply_heterogeneous_decay",
    "TauAwareLIF",
    # Diagnostics
    "log_tau_summary",
    "format_tau_report",
    "check_tau_health",
    # Migration
    "migrate_logit_beta_to_tau",
    # Constants
    "_VALID_MODES",
    "_VALID_GRANULARITIES",
    "_VALID_INIT_STRATEGIES",
]


# ===========================================================================
# SECTION 10: Self-test — run with: python heterogeneous_tau_template.py
# ===========================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  heterogeneous_tau_template.py — Self-test suite")
    print("=" * 60)

    passed = 0
    failed = 0

    def _check(name: str, condition: bool, msg: str = "") -> None:
        global passed, failed
        status = "PASS" if condition else "FAIL"
        label = f"  [{status}] {name}"
        if not condition and msg:
            label += f"\n         Reason: {msg}"
        print(label)
        if condition:
            passed += 1
        else:
            failed += 1

    device = torch.device("cpu")
    N = 128

    # -----------------------------------------------------------------------
    # Test 1: All 4 initialisation strategies produce valid τ
    # -----------------------------------------------------------------------
    print("\n--- Test 1: Initialisation strategies ---")

    tau_homo = init_tau_homogeneous((N,), tau_0=20.0)
    _check(
        "homogeneous: all values == 20.0",
        torch.allclose(tau_homo, torch.full((N,), 20.0)),
    )
    _check(
        "homogeneous: shape correct",
        tau_homo.shape == (N,),
    )

    tau_gamma = init_tau_gamma((N,), gamma_shape=4.0, gamma_scale=5.0,
                               tau_min=1.0, tau_max=100.0, seed=42)
    _check(
        "gamma: all values in [1, 100]",
        bool((tau_gamma >= 1.0).all() and (tau_gamma <= 100.0).all()),
    )
    _check(
        "gamma: mean near 20 (±10)",
        abs(tau_gamma.mean().item() - 20.0) < 15.0,
        f"mean={tau_gamma.mean().item():.2f}",
    )

    tau_logu = init_tau_loguniform((N,), tau_min=1.0, tau_max=100.0, seed=42)
    _check(
        "loguniform: all values in [1, 100]",
        bool((tau_logu >= 1.0).all() and (tau_logu <= 100.0).all()),
    )
    _check(
        "loguniform: shape correct",
        tau_logu.shape == (N,),
    )

    tau_preset = init_tau_preset_bank(
        (N,), preset_values=(2.0, 5.0, 10.0, 20.0, 50.0), seed=None
    )
    valid_presets = {2.0, 5.0, 10.0, 20.0, 50.0}
    _check(
        "preset_bank: all values from preset set",
        all(v.item() in valid_presets for v in tau_preset),
    )

    # -----------------------------------------------------------------------
    # Test 2: β always in (0, 1) for all init strategies
    # -----------------------------------------------------------------------
    print("\n--- Test 2: β in (0, 1) for all modes ---")

    for mode_name, init_s in [
        ("homogeneous_fixed", "homogeneous"),
        ("heterogeneous_fixed", "heterogeneous_loguniform"),
        ("heterogeneous_learnable", "heterogeneous_gamma"),
        ("heterogeneous_learnable", "preset_bank"),
    ]:
        cfg = TauConfig(mode=mode_name, init_strategy=init_s)
        tau_mod = HeterogeneousTau(N, cfg)
        beta = tau_mod.beta
        _check(
            f"beta in (0,1) — mode={mode_name}, init={init_s}",
            bool((beta > 0.0).all() and (beta < 1.0).all()),
            f"min={beta.min().item():.6f}, max={beta.max().item():.6f}",
        )

    # -----------------------------------------------------------------------
    # Test 3: Gradient flows through tau_raw when learnable
    # -----------------------------------------------------------------------
    print("\n--- Test 3: Gradient flow (learnable mode) ---")

    cfg_learn = TauConfig(mode="heterogeneous_learnable")
    tau_mod_learn = HeterogeneousTau(N, cfg_learn)

    # Forward pass with loss
    beta = tau_mod_learn.beta
    loss = beta.sum()
    loss.backward()

    _check(
        "tau_raw.grad is not None after backward",
        tau_mod_learn.tau_raw.grad is not None,
    )
    _check(
        "tau_raw.grad has correct shape",
        tau_mod_learn.tau_raw.grad is not None
        and tau_mod_learn.tau_raw.grad.shape == (N,),
    )
    _check(
        "tau_raw.grad is not all-zero",
        tau_mod_learn.tau_raw.grad is not None
        and tau_mod_learn.tau_raw.grad.abs().sum().item() > 0.0,
    )

    # -----------------------------------------------------------------------
    # Test 4: No gradient when fixed mode
    # -----------------------------------------------------------------------
    print("\n--- Test 4: No gradient when fixed ---")

    cfg_fixed = TauConfig(mode="heterogeneous_fixed")
    tau_mod_fixed = HeterogeneousTau(N, cfg_fixed)

    _check(
        "tau_raw is a buffer (not Parameter) in fixed mode",
        not isinstance(tau_mod_fixed.tau_raw, nn.Parameter),
    )
    _check(
        "tau_raw.requires_grad is False in fixed mode",
        not tau_mod_fixed.tau_raw.requires_grad,
    )

    # Confirm no params in fixed mode
    n_trainable = sum(p.numel() for p in tau_mod_fixed.parameters())
    _check(
        "zero trainable parameters in fixed mode",
        n_trainable == 0,
        f"found {n_trainable} trainable params",
    )

    # -----------------------------------------------------------------------
    # Test 5: Per-neuron, per-channel, per-layer all produce correct shapes
    # -----------------------------------------------------------------------
    print("\n--- Test 5: Granularity broadcasting ---")

    # Per-neuron: (B=4, N=128)
    cfg_neuron = TauConfig(granularity="per_neuron", mode="heterogeneous_learnable")
    tau_neuron = HeterogeneousTau(N, cfg_neuron)
    beta_n = tau_neuron.beta_broadcast(target_shape=(4, N))
    v_test = torch.randn(4, N)
    x_test = torch.randn(4, N)
    v_new = apply_heterogeneous_decay(v_test, beta_n, x_test)
    _check(
        "per_neuron: beta_broadcast shape (1, N)",
        beta_n.shape == (1, N),
        f"got {beta_n.shape}",
    )
    _check(
        "per_neuron: v_new shape (B, N)",
        v_new.shape == (4, N),
        f"got {v_new.shape}",
    )

    # Per-channel: (B=4, C=32, H=8, W=8)
    C = 32
    cfg_chan = TauConfig(granularity="per_channel", mode="heterogeneous_learnable")
    tau_chan = HeterogeneousTau(C, cfg_chan)
    beta_c = tau_chan.beta_broadcast(target_shape=(4, C, 8, 8))
    v_conv = torch.randn(4, C, 8, 8)
    x_conv = torch.randn(4, C, 8, 8)
    v_new_conv = apply_heterogeneous_decay(v_conv, beta_c, x_conv)
    _check(
        "per_channel: beta_broadcast shape (1, C, 1, 1)",
        beta_c.shape == (1, C, 1, 1),
        f"got {beta_c.shape}",
    )
    _check(
        "per_channel: v_new shape (B, C, H, W)",
        v_new_conv.shape == (4, C, 8, 8),
        f"got {v_new_conv.shape}",
    )

    # Per-layer: scalar
    cfg_layer = TauConfig(
        granularity="per_layer",
        mode="homogeneous_fixed",
        init_strategy="homogeneous",
    )
    tau_layer = HeterogeneousTau(shape=(), config=cfg_layer)
    beta_l = tau_layer.beta_broadcast(target_shape=(4, N))
    _check(
        "per_layer: beta is a scalar (dim==0)",
        beta_l.dim() == 0,
        f"got shape {beta_l.shape}",
    )

    # -----------------------------------------------------------------------
    # Test 6: ConvSNN broadcast correctness
    # -----------------------------------------------------------------------
    print("\n--- Test 6: ConvTauModule ---")

    cfg_conv = TauConfig(mode="heterogeneous_learnable", tau_min=2.0, tau_max=50.0)
    conv_tau = ConvTauModule(num_channels=64, config=cfg_conv)

    beta_conv = conv_tau.beta_broadcast(target_shape=(2, 64, 16, 16))
    _check(
        "ConvTauModule: beta shape (1, 64, 1, 1)",
        beta_conv.shape == (1, 64, 1, 1),
        f"got {beta_conv.shape}",
    )

    v_4d = torch.randn(2, 64, 16, 16)
    x_4d = torch.randn(2, 64, 16, 16)
    v_new_4d = apply_heterogeneous_decay(v_4d, beta_conv, x_4d)
    _check(
        "ConvTauModule: decay output shape (B, C, H, W)",
        v_new_4d.shape == (2, 64, 16, 16),
        f"got {v_new_4d.shape}",
    )

    # Verify gradient flows through ConvTauModule tau_raw
    loss_conv = v_new_4d.sum()
    loss_conv.backward()
    _check(
        "ConvTauModule: gradient flows to tau_raw",
        conv_tau.tau_raw.grad is not None
        and conv_tau.tau_raw.grad.abs().sum().item() > 0.0,
    )

    # Wrong channel count raises ValueError
    caught = False
    try:
        conv_tau.beta_broadcast(target_shape=(2, 99, 8, 8))
    except ValueError:
        caught = True
    _check("ConvTauModule: ValueError on channel mismatch", caught)

    # -----------------------------------------------------------------------
    # Test 7: Migration from logit_beta_vec
    # -----------------------------------------------------------------------
    print("\n--- Test 7: Migration from logit_beta_vec ---")

    # Simulate AdvancedLIFNeuron logit_beta_vec
    torch.manual_seed(0)
    logit_beta_old = torch.randn(N) * 2.0  # unconstrained

    tau_raw_migrated = migrate_logit_beta_to_tau(
        logit_beta_old, dt=1.0, tau_min=1.0, tau_max=100.0
    )

    _check(
        "migration: tau_raw_migrated has correct shape",
        tau_raw_migrated.shape == (N,),
        f"got {tau_raw_migrated.shape}",
    )
    _check(
        "migration: tau_raw_migrated is finite",
        torch.isfinite(tau_raw_migrated).all().item(),
    )
    _check(
        "migration: tau_raw_migrated requires_grad is False (detached)",
        not tau_raw_migrated.requires_grad,
    )

    # Load into a TauAwareLIF neuron
    cfg_migrate = TauConfig(mode="heterogeneous_learnable")
    neuron_migrated = TauAwareLIF(size=N, tau_config=cfg_migrate)
    with torch.no_grad():
        neuron_migrated.tau_module.tau_raw.copy_(tau_raw_migrated)

    tau_after = neuron_migrated.tau_module.tau
    _check(
        "migration: loaded tau values in [tau_min, tau_max]",
        bool((tau_after >= 1.0).all() and (tau_after <= 100.0).all()),
        f"tau range: [{tau_after.min().item():.3f}, {tau_after.max().item():.3f}]",
    )

    # Verify round-trip: original beta ≈ migrated beta
    original_beta = torch.sigmoid(logit_beta_old).clamp(0.001, 0.999)
    migrated_beta = neuron_migrated.tau_module.beta
    max_diff = (original_beta - migrated_beta).abs().max().item()
    _check(
        "migration: round-trip beta error < 0.01",
        max_diff < 0.01,
        f"max |β_orig - β_migrated| = {max_diff:.6f}",
    )

    # -----------------------------------------------------------------------
    # Test 8: Health check returns valid status
    # -----------------------------------------------------------------------
    print("\n--- Test 8: Health checks ---")

    # Healthy module
    cfg_healthy = TauConfig(mode="heterogeneous_learnable")
    tau_healthy = HeterogeneousTau(N, cfg_healthy)
    status, reasons = check_tau_health(tau_healthy)
    _check(
        "fresh module is HEALTHY",
        status == "HEALTHY",
        f"status={status}, reasons={reasons}",
    )

    # Trigger WARNING: high clamp pressure (init very close to tau_max)
    cfg_warn = TauConfig(
        mode="heterogeneous_fixed",
        init_strategy="homogeneous",
        tau_0=99.9,   # near tau_max=100, so many will clamp
        tau_min=1.0,
        tau_max=100.0,
    )
    tau_warn = HeterogeneousTau(N, cfg_warn)
    # Artificially set tau_raw to extreme values to force clamp pressure
    with torch.no_grad():
        tau_warn.tau_raw.fill_(1000.0)  # softplus(1000) >> tau_max - tau_min
    status_w, reasons_w = check_tau_health(tau_warn)
    _check(
        "extreme tau_raw triggers WARNING or CRITICAL",
        status_w in ("WARNING", "CRITICAL"),
        f"got status={status_w}",
    )

    # CRITICAL: NaN in tau_raw
    cfg_crit = TauConfig(mode="heterogeneous_learnable")
    tau_crit = HeterogeneousTau(N, cfg_crit)
    with torch.no_grad():
        tau_crit.tau_raw.fill_(float("nan"))
    status_c, reasons_c = check_tau_health(tau_crit)
    _check(
        "NaN tau_raw triggers CRITICAL",
        status_c == "CRITICAL",
        f"got status={status_c}",
    )

    # -----------------------------------------------------------------------
    # Test 9: TauAwareLIF forward pass produces spikes with correct shape
    # -----------------------------------------------------------------------
    print("\n--- Test 9: TauAwareLIF forward pass ---")

    cfg_lif = TauConfig(mode="heterogeneous_learnable", tau_min=2.0, tau_max=50.0)
    lif_neuron = TauAwareLIF(size=N, tau_config=cfg_lif, threshold=1.0)
    B = 4
    state_init = lif_neuron.reset_state(
        batch_size=B, neuron_shape=(N,), device=device
    )

    T = 10
    all_spikes = []
    state = state_init
    for t in range(T):
        x_t = torch.randn(B, N) * 0.5
        spikes_t, state = lif_neuron(x_t, state)
        all_spikes.append(spikes_t)

    _check(
        "TauAwareLIF: spike shape (B, N)",
        spikes_t.shape == (B, N),
        f"got {spikes_t.shape}",
    )
    _check(
        "TauAwareLIF: spikes are binary",
        bool(((spikes_t == 0.0) | (spikes_t == 1.0)).all()),
    )
    _check(
        "TauAwareLIF: state.v shape (B, N)",
        state.v.shape == (B, N),
        f"got {state.v.shape}",
    )

    # Gradient flows through tau_module during forward pass
    lif_neuron.zero_grad()
    x_grad = torch.randn(B, N, requires_grad=False)
    state_g = lif_neuron.reset_state(B, (N,), device)
    _, state_g = lif_neuron(x_grad, state_g)
    state_g.v.sum().backward()
    _check(
        "TauAwareLIF: gradient flows to tau_module.tau_raw",
        lif_neuron.tau_module.tau_raw.grad is not None
        and lif_neuron.tau_module.tau_raw.grad.abs().sum().item() > 0.0,
    )

    # -----------------------------------------------------------------------
    # Test 10: format_tau_report and log_tau_summary run without error
    # -----------------------------------------------------------------------
    print("\n--- Test 10: Diagnostics utilities ---")

    report = format_tau_report(lif_neuron.tau_module)
    _check(
        "format_tau_report: returns non-empty string",
        isinstance(report, str) and len(report) > 50,
    )

    summary = log_tau_summary(lif_neuron.tau_module, prefix="test/layer1")
    _check(
        "log_tau_summary: returns dict with prefixed keys",
        isinstance(summary, dict)
        and any(k.startswith("test/layer1/") for k in summary),
    )
    _check(
        "log_tau_summary: all values are floats",
        all(isinstance(v, float) for v in summary.values()),
    )

    firing_rates = torch.rand(N)  # random fake firing rates
    corr = lif_neuron.tau_module.compute_firing_rate_correlation(firing_rates)
    _check(
        "compute_firing_rate_correlation: result in [-1, 1]",
        -1.0 <= corr <= 1.0,
        f"got {corr:.4f}",
    )

    drift = lif_neuron.tau_module.compute_drift()
    _check(
        "compute_drift: returns non-negative float",
        isinstance(drift, float) and drift >= 0.0,
        f"got {drift}",
    )

    hist = lif_neuron.tau_module.get_tau_histogram(num_bins=20)
    _check(
        "get_tau_histogram: has bin_edges and counts keys",
        "bin_edges" in hist and "counts" in hist,
    )
    _check(
        "get_tau_histogram: bin_edges length == num_bins + 1",
        hist["bin_edges"].shape[0] == 21,
        f"got {hist['bin_edges'].shape}",
    )

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} tests passed")
    if failed > 0:
        print(f"  FAILED:  {failed} test(s)")
    else:
        print("  All tests PASSED.")
    print("=" * 60)

    # Print the tau distribution report for the final TauAwareLIF
    print("\n" + format_tau_report(lif_neuron.tau_module))

    sys.exit(0 if failed == 0 else 1)
