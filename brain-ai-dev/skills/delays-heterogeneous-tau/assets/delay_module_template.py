"""
brain_ai/core/delays.py — DCLS-Style Learnable Delay Modules

Implements Dilated Convolution with Learnable Spacings (DCLS) adapted for
spiking neural network delay lines. Replaces the soft-attention delay
mechanism in AdvancedLIFNeuron with a differentiable Gaussian interpolation
scheme that enables the network to learn precise integer-like spike delays
through continuous relaxation.

Key design principles:
    1. Gaussian interpolation: continuous delay parameter d is approximated
       by a weighted sum over K integer neighbors, with weights following a
       Gaussian kernel centered at d. As sigma anneals toward 0, the
       effective delay approaches a hard integer assignment.
    2. Granularity: delay parameters may be shared across entire blocks,
       per input-channel, per output-channel, or per-synapse, trading
       expressiveness for memory footprint.
    3. No in-place ops: every tensor operation creates a fresh tensor to
       remain compatible with autograd and torch.compile.
    4. fp32 delay state: delay parameters and sigma buffers are always
       float32 regardless of model AMP dtype to prevent precision loss
       in the sigmoid reparameterisation.
    5. Compatibility with snn_unroll: modules expose a forward(x, history)
       signature compatible with cell_fn wrappers used by snn_unroll.

DCLS reference:
    Hammouamri et al. (2024) "Learning Delays in Spiking Neural Networks using
    Dilated Convolutions with Learnable Spacings"
    https://arxiv.org/abs/2306.17670

Hammouamri et al. adaptation:
    Hammouamri et al. (2024) "Learning Delays in Spiking Neural Networks
    using Dilated Convolutions with Learnable Spacings"
    https://arxiv.org/abs/2306.17670

Canonical integration path:
    Copy to brain_ai/core/delays.py.
    Import DelayLinear, DelayConv1d, DelayConfig, SpikeHistoryBuffer.
    In AdvancedLIFNeuron.__init__, replace delay_weights + delay_coupling
    with a DelayLinear(size, size, delay_config) instance.
    In AdvancedLIFNeuron._step, call delay_linear(x, state.spike_history)
    instead of _apply_delays().

TODO(integration): Wire DelayConfig into SNNConfig so the full pipeline
    can configure delay behaviour declaratively from BrainAIConfig.
"""

from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Numerical floor for sigma to prevent division by zero in Gaussian kernel
SIGMA_MIN: float = 1e-4

# Logit clamp range to prevent d_raw from saturating sigmoid
DRAW_CLAMP: float = 10.0

# Valid granularity modes
GRANULARITY_MODES = ("per_synapse", "per_output", "per_input", "per_block")

# Valid delay modes
DELAY_MODES = ("off", "fixed_random", "learnable_dcls")

# Valid sigma schedule types
SIGMA_SCHEDULES = ("constant", "decreasing")

# Valid delay initialisation strategies
INIT_STRATEGIES = ("uniform", "normal_center", "zeros")


# ===========================================================================
# SECTION 1: DelayConfig dataclass
# ===========================================================================
# NOTE: The canonical DelayConfig definition lives in delay_tau_config_template.py.
# This copy is included here for standalone usage of this template.  When
# integrating into brain_ai, use the single canonical source from config.py
# and import it here instead of defining a duplicate.
# ===========================================================================


@dataclass
class DelayConfig:
    """Configuration for DCLS-style learnable delay modules.

    This dataclass is the single source of truth for all delay-related
    hyperparameters. It may be embedded in SNNConfig to enable declarative
    configuration through BrainAIConfig.

    Attributes:
        mode: Delay mechanism to use.
            'off': No delays; modules act as plain linear/conv layers.
            'fixed_random': Delays sampled at init and frozen (no grad).
            'learnable_dcls': Differentiable Gaussian interpolation (default).
        max_delay: Maximum delay in timesteps (Td). The spike history buffer
            has depth max_delay. Delays d are constrained to [0, max_delay-1].
        num_bins: Number K of Gaussian interpolation bins around each delay.
            Must be an odd integer >= 1. Higher K allows wider sigma while
            keeping approximation error low; K=3 is sufficient for sigma <= 1.5.
        granularity: How delay parameters are shared across synapses.
            'per_synapse': one delay per (out, in) pair — maximum expressiveness,
                maximum memory (out_features * in_features delay params).
            'per_output': one delay per output neuron (broadcast across inputs).
            'per_input': one delay per input (broadcast across outputs).
            'per_block': coarse-grained blocks of size block_size in each dim.
        block_size: Block size for 'per_block' granularity. Each axis is
            divided into ceil(features / block_size) blocks.
        sigma_start: Initial Gaussian sigma. Larger sigma = softer assignment,
            smoother gradients, coarser delay resolution.
        sigma_end: Final sigma after sigma_decay_epochs. Keep >= SIGMA_MIN.
            When discretize_at_inference=True, the final effective sigma is
            irrelevant during inference (delays are rounded to integers).
        sigma_decay_epochs: Number of training epochs over which sigma
            decreases from sigma_start to sigma_end.
        sigma_schedule: Shape of the sigma decay curve.
            'constant': sigma stays at sigma_start forever.
            'decreasing': exponential decay from sigma_start to sigma_end.
        discretize_at_inference: If True, round delays to nearest integer
            during module.training=False forward passes. This removes the
            Gaussian blur entirely and makes inference equivalent to fixed
            integer delays. (Named without 'eval' prefix for hook compat.)
        memory_warn_threshold: If out_features * in_features * max_delay
            exceeds this integer, emit a warning. Default 100M elements.
    """

    mode: str = "learnable_dcls"
    max_delay: int = 16
    num_bins: int = 3
    granularity: str = "per_output"
    block_size: int = 64
    sigma_start: float = 1.0
    sigma_end: float = 0.5
    sigma_decay_epochs: int = 50
    sigma_schedule: str = "decreasing"
    discretize_at_inference: bool = True
    memory_warn_threshold: int = 100_000_000

    def __post_init__(self) -> None:
        """Validate configuration fields after construction."""
        if self.mode not in DELAY_MODES:
            raise ValueError(
                f"DelayConfig.mode must be one of {DELAY_MODES}, got '{self.mode}'"
            )
        if self.max_delay < 1:
            raise ValueError(
                f"DelayConfig.max_delay must be >= 1, got {self.max_delay}"
            )
        if self.num_bins < 1 or self.num_bins % 2 == 0:
            raise ValueError(
                f"DelayConfig.num_bins must be a positive odd integer, "
                f"got {self.num_bins}"
            )
        if self.granularity not in GRANULARITY_MODES:
            raise ValueError(
                f"DelayConfig.granularity must be one of {GRANULARITY_MODES}, "
                f"got '{self.granularity}'"
            )
        if self.block_size < 1:
            raise ValueError(
                f"DelayConfig.block_size must be >= 1, got {self.block_size}"
            )
        if self.sigma_start <= 0.0:
            raise ValueError(
                f"DelayConfig.sigma_start must be positive, got {self.sigma_start}"
            )
        if self.sigma_end <= 0.0:
            raise ValueError(
                f"DelayConfig.sigma_end must be positive, got {self.sigma_end}"
            )
        if self.sigma_decay_epochs < 1:
            raise ValueError(
                f"DelayConfig.sigma_decay_epochs must be >= 1, "
                f"got {self.sigma_decay_epochs}"
            )
        if self.sigma_schedule not in SIGMA_SCHEDULES:
            raise ValueError(
                f"DelayConfig.sigma_schedule must be one of {SIGMA_SCHEDULES}, "
                f"got '{self.sigma_schedule}'"
            )

    @classmethod
    def off(cls) -> "DelayConfig":
        """Convenience constructor: no delays."""
        return cls(mode="off")

    @classmethod
    def fixed(cls, max_delay: int = 16) -> "DelayConfig":
        """Convenience constructor: fixed random delays, no learning."""
        return cls(mode="fixed_random", max_delay=max_delay)

    @classmethod
    def learnable(
        cls,
        max_delay: int = 16,
        granularity: str = "per_output",
        sigma_start: float = 1.0,
        sigma_end: float = 0.5,
        sigma_decay_epochs: int = 50,
    ) -> "DelayConfig":
        """Convenience constructor: learnable DCLS delays with common defaults."""
        return cls(
            mode="learnable_dcls",
            max_delay=max_delay,
            granularity=granularity,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            sigma_decay_epochs=sigma_decay_epochs,
        )


# ===========================================================================
# SECTION 2: Gaussian interpolation helper
# ===========================================================================


def gaussian_kernel(
    positions: Tensor,
    centers: Tensor,
    sigma: float,
) -> Tensor:
    """Compute normalised Gaussian weights for delay positions relative to integer centers.

    For each continuous delay position d, computes:
        g(n; d, sigma) = exp(-(n - d)^2 / (2 * sigma^2))
    for each integer center n in `centers`, then normalises so weights sum
    to 1 along the K-dimension. This implements the differentiable
    relaxation of integer delay assignment used by DCLS.

    As sigma -> 0, the output approaches a one-hot vector at the nearest
    integer center (hard assignment). As sigma grows, it approaches a
    uniform distribution (all delays equally weighted).

    Args:
        positions: Continuous delay values, shape (...,). fp32 required.
            Values should lie in [0, max_delay - 1].
        centers: Integer bin centers, shape (K,). Typically obtained from
            get_bin_centers(); values are in {0, 1, ..., max_delay - 1}.
        sigma: Current Gaussian width. Must be > 0. Values near SIGMA_MIN
            produce near one-hot outputs; values > 2 produce broad weighting.

    Returns:
        weights: Gaussian weights, shape (..., K), dtype matches positions.
            Each slice [..., :] sums to 1 along the K dimension.

    Notes:
        - sigma is clamped to SIGMA_MIN internally to prevent division by zero.
        - centers must be passed as a float tensor for the subtraction; the
          caller is responsible for converting integer centers appropriately.

    Example::

        delays = torch.tensor([2.7, 8.1])          # (2,)
        centers = torch.tensor([2.0, 3.0, 4.0])    # K=3
        w = gaussian_kernel(delays, centers, sigma=0.8)
        # w.shape == (2, 3), w.sum(dim=-1) == [1.0, 1.0]
    """
    # Ensure minimum sigma to avoid numerical issues
    sigma_safe = max(float(sigma), SIGMA_MIN)

    # positions: (...), centers: (K,)
    # Broadcast: positions[..., None] - centers => (..., K)
    diff = positions.unsqueeze(-1) - centers.to(positions.dtype)  # (..., K)

    # Unnormalised Gaussian kernel expressed as log-weights for numerical stability
    log_weights = -(diff ** 2) / (2.0 * sigma_safe ** 2)

    # Normalise via softmax (log-sum-exp under the hood) — equivalent to
    # exp(log_weights) / sum(exp(log_weights)) but avoids overflow
    weights = torch.softmax(log_weights, dim=-1)  # (..., K)

    return weights


def _integer_centers_around(
    delays: Tensor,
    num_bins: int,
    max_delay: int,
) -> Tensor:
    """Compute K integer centers surrounding each continuous delay value.

    For each delay d, returns the K nearest integers centered at round(d),
    clamped to the valid range [0, max_delay - 1].

    Args:
        delays: Continuous delays, shape (...,). Values in [0, max_delay-1].
        num_bins: Number of integer centers K. Must be an odd positive integer.
        max_delay: Upper bound on delay values (exclusive).

    Returns:
        centers: Integer centers, shape (..., K), dtype=torch.int64.
            For K=3 and d=5.7, centers would be [5, 6, 7] (anchor=6, offsets=-1,0,+1).

    Notes:
        Centers are computed relative to the nearest integer to d, not to d
        itself. This keeps the K centers symmetric around the most likely
        integer assignment and avoids boundary issues when d is near 0 or
        max_delay - 1.
    """
    half_k = num_bins // 2
    # Round to nearest integer as the anchor (no grad through this path)
    anchor = delays.detach().round().long()  # (...,)
    # Offsets: [-half_k, ..., 0, ..., +half_k]
    offsets = torch.arange(
        -half_k, -half_k + num_bins,
        device=delays.device,
        dtype=torch.long,
    )  # (K,)
    # Broadcast: (..., 1) + (K,) => (..., K)
    centers = anchor.unsqueeze(-1) + offsets
    # Clamp to valid range [0, max_delay-1]
    centers = centers.clamp(0, max_delay - 1)
    return centers


# ===========================================================================
# SECTION 3: DelayModuleBase — abstract base class
# ===========================================================================


class DelayModuleBase(nn.Module, ABC):
    """Abstract base class for DCLS-style learnable delay modules.

    Manages the d_raw parameter (unconstrained delay in logit space),
    the sigma annealing schedule, and exposes the forward() interface
    that concrete subclasses (DelayLinear, DelayConv1d) must implement.

    Reparameterisation:
        d = (Td - 1) * sigmoid(d_raw)

    This maps d_raw in (-inf, +inf) to delays in [0, Td-1], with the
    sigmoid ensuring differentiability and the (Td-1) factor scaling to
    the full delay range. The inverse is:
        d_raw = logit(d / (Td - 1))

    Attributes registered as buffers (moved with .to(device)):
        sigma (float32 scalar): Current Gaussian width.
        sigma_epoch (int64 scalar): Epoch counter for the sigma schedule.

    Args:
        delay_config: Full DelayConfig instance.
        d_raw_init: Pre-constructed tensor for d_raw. Subclasses supply this
            after determining the correct shape from their granularity mode.
        fixed: If True, d_raw is registered as a buffer, not a parameter.
            Used when mode='fixed_random'.
    """

    def __init__(
        self,
        delay_config: DelayConfig,
        d_raw_init: Tensor,
        fixed: bool = False,
    ) -> None:
        super().__init__()

        self.cfg = delay_config
        self._fixed = fixed

        if fixed:
            self.register_buffer("d_raw", d_raw_init.float())
        else:
            self.d_raw: nn.Parameter = nn.Parameter(d_raw_init.float())

        # Sigma schedule state — always fp32 buffers so they survive .to(device)
        self.register_buffer(
            "sigma",
            torch.tensor(delay_config.sigma_start, dtype=torch.float32),
        )
        self.register_buffer(
            "sigma_epoch",
            torch.tensor(0, dtype=torch.int64),
        )

    # -----------------------------------------------------------------------
    # Delay value properties
    # -----------------------------------------------------------------------

    @property
    def delays(self) -> Tensor:
        """Continuous delay values in [0, max_delay - 1], shape = d_raw.shape.

        Uses the sigmoid reparameterisation:
            d = (Td - 1) * sigmoid(d_raw_clamped)

        d_raw is clamped to [-DRAW_CLAMP, DRAW_CLAMP] before sigmoid to
        prevent saturated gradients. Gradients still flow through d_raw.

        Returns:
            Float tensor of continuous delays. fp32 regardless of model dtype.
        """
        td = float(self.cfg.max_delay - 1)
        if td == 0:
            # Edge case: max_delay == 1, only one valid delay (0)
            return torch.zeros_like(self.d_raw)
        d_clamped = self.d_raw.clamp(-DRAW_CLAMP, DRAW_CLAMP)
        return td * torch.sigmoid(d_clamped)

    @property
    def delays_discrete(self) -> Tensor:
        """Nearest-integer delays clamped to [0, max_delay - 1], shape = d_raw.shape.

        Computed without gradient — this property is used only for
        inference-mode discretised forward passes and diagnostics.

        Returns:
            Long (int64) tensor of integer delay indices.
        """
        with torch.no_grad():
            return self.delays.round().long().clamp(0, self.cfg.max_delay - 1)

    # -----------------------------------------------------------------------
    # Sigma schedule
    # -----------------------------------------------------------------------

    def update_sigma(self, epoch: int) -> float:
        """Update the Gaussian sigma according to the annealing schedule.

        Call this at the beginning of each training epoch, before the forward
        pass. The update is deterministic given epoch, so it is safe to call
        multiple times with the same epoch (idempotent).

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            New sigma value (float) for logging purposes.

        Notes:
            - sigma_epoch buffer is updated via copy_ inside torch.no_grad()
              to avoid adding this to the computation graph.
            - For 'constant' schedule, sigma remains at sigma_start forever.
            - For 'decreasing' schedule, sigma decays exponentially from
              sigma_start to sigma_end over sigma_decay_epochs epochs and
              then stays at sigma_end.
        """
        with torch.no_grad():
            self.sigma_epoch.copy_(torch.tensor(epoch, dtype=torch.int64))

            if self.cfg.sigma_schedule == "constant":
                new_sigma = self.cfg.sigma_start
            elif self.cfg.sigma_schedule == "decreasing":
                new_sigma = (
                    self.cfg.sigma_end
                    + (self.cfg.sigma_start - self.cfg.sigma_end)
                    * math.exp(-epoch / max(1, self.cfg.sigma_decay_epochs))
                )
            else:
                # Unreachable if __post_init__ validates; defensive fallback
                new_sigma = self.cfg.sigma_start

            new_sigma = max(new_sigma, SIGMA_MIN)
            self.sigma.copy_(torch.tensor(new_sigma, dtype=torch.float32))

        return float(self.sigma.item())

    # -----------------------------------------------------------------------
    # Bin center helper
    # -----------------------------------------------------------------------

    def get_bin_centers(self, delays: Tensor) -> Tensor:
        """Compute K integer bin centers surrounding each continuous delay.

        This is a thin wrapper around _integer_centers_around that passes
        the module's K and max_delay values.

        Args:
            delays: Continuous delays from self.delays, shape (...,).

        Returns:
            centers: Long tensor of shape (..., K) with integer delay indices
                in [0, max_delay - 1].
        """
        return _integer_centers_around(
            delays,
            num_bins=self.cfg.num_bins,
            max_delay=self.cfg.max_delay,
        )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def get_delay_diagnostics(self) -> Dict[str, float]:
        """Return a diagnostic dictionary for monitoring delay learning.

        Intended to be called during training and logged to tensorboard/wandb.
        All values are computed under torch.no_grad() to avoid polluting the
        computation graph.

        Returns:
            Dict with the following keys:
                'delay_mean': Mean of all continuous delay values.
                'delay_std': Standard deviation of all continuous delay values.
                'delay_min': Minimum continuous delay.
                'delay_max': Maximum continuous delay.
                'delay_entropy': Normalised entropy of the discrete delay
                    histogram as a measure of diversity (0 = all same delay,
                    1 = uniform spread over [0, Td-1]).
                'boundary_pct': Percentage of delays at the boundary
                    (rounded to 0 or max_delay-1), indicating potential
                    saturation of the sigmoid parameterisation.
                'sigma': Current sigma value.
                'sigma_epoch': Current epoch counter.
        """
        with torch.no_grad():
            d = self.delays.float().flatten()
            d_disc = self.delays_discrete.float().flatten()
            td = float(self.cfg.max_delay)

            delay_mean = d.mean().item()
            delay_std = d.std().item() if d.numel() > 1 else 0.0
            delay_min = d.min().item()
            delay_max = d.max().item()

            # Discrete histogram entropy, normalised to [0, 1]
            if td > 1:
                counts = torch.bincount(
                    d_disc.long().view(-1),
                    minlength=self.cfg.max_delay,
                ).float()
                probs = counts / counts.sum().clamp(min=1.0)
                log_probs = torch.where(
                    probs > 0,
                    torch.log(probs),
                    torch.zeros_like(probs),
                )
                entropy_bits = -(probs * log_probs).sum().item()
                max_entropy = math.log(td)
                entropy_norm = entropy_bits / max_entropy if max_entropy > 0 else 0.0
            else:
                entropy_norm = 0.0

            # Boundary percentage: delays rounded to 0 or max_delay - 1
            boundary = ((d_disc == 0) | (d_disc == self.cfg.max_delay - 1))
            boundary_pct = boundary.float().mean().item() * 100.0

        return {
            "delay_mean": delay_mean,
            "delay_std": delay_std,
            "delay_min": delay_min,
            "delay_max": delay_max,
            "delay_entropy": entropy_norm,
            "boundary_pct": boundary_pct,
            "sigma": float(self.sigma.item()),
            "sigma_epoch": int(self.sigma_epoch.item()),
        }

    # -----------------------------------------------------------------------
    # Memory guard
    # -----------------------------------------------------------------------

    def _check_memory(
        self,
        out_features: int,
        in_features: int,
        batch_size: Optional[int] = None,
    ) -> None:
        """Warn if estimated memory usage of delay computation is excessive.

        The history tensor during forward has shape (B, Td, in_features) and
        the weight tensor has shape (out_features, in_features). The dominant
        cost is iterating over Td delay bins and accumulating out_features outputs.

        Args:
            out_features: Number of output neurons / channels.
            in_features: Number of input neurons / channels.
            batch_size: Batch size for per-batch memory estimate (optional).
        """
        num_elements = out_features * in_features * self.cfg.max_delay
        if num_elements > self.cfg.memory_warn_threshold:
            gb_estimate = (num_elements * 4) / 1e9  # float32, bytes -> GB
            warnings.warn(
                f"DelayModule memory check: "
                f"out={out_features} x in={in_features} x Td={self.cfg.max_delay} "
                f"= {num_elements:,} elements (~{gb_estimate:.2f} GB fp32). "
                f"Consider reducing max_delay or using coarser granularity. "
                f"Threshold: {self.cfg.memory_warn_threshold:,}",
                ResourceWarning,
                stacklevel=3,
            )
        if batch_size is not None:
            hist_elements = batch_size * self.cfg.max_delay * in_features
            hist_gb = (hist_elements * 4) / 1e9
            if hist_elements > self.cfg.memory_warn_threshold // 10:
                warnings.warn(
                    f"DelayModule spike history: "
                    f"B={batch_size} x Td={self.cfg.max_delay} x in={in_features} "
                    f"= {hist_elements:,} fp32 elements (~{hist_gb:.2f} GB). "
                    f"Consider smaller batch size or lower max_delay.",
                    ResourceWarning,
                    stacklevel=3,
                )

    # -----------------------------------------------------------------------
    # Abstract forward — subclasses implement
    # -----------------------------------------------------------------------

    @abstractmethod
    def forward(self, x: Tensor, spike_history: Tensor) -> Tensor:
        """Compute delayed input contribution given the spike history buffer.

        Args:
            x: Direct (un-delayed) input, shape (B, in_features) for
                DelayLinear or (B, C_in, L) for DelayConv1d. fp32.
            spike_history: Circular spike history buffer of shape
                (B, max_delay, in_features) for DelayLinear or
                (B, max_delay, C_in, L) for DelayConv1d. fp32.

        Returns:
            Output tensor of shape (B, out_features) or (B, C_out, L_out)
            depending on subclass. Gaussian-weighted sum over delay taps
            convolved/matmul'd with the weight matrix.
        """

    def extra_repr(self) -> str:
        return (
            f"mode={self.cfg.mode}, "
            f"max_delay={self.cfg.max_delay}, "
            f"num_bins={self.cfg.num_bins}, "
            f"granularity={self.cfg.granularity}, "
            f"sigma={self.sigma.item():.3f}"
        )


# ===========================================================================
# SECTION 4: DelayLinear
# ===========================================================================


class DelayLinear(DelayModuleBase):
    """Fully-connected layer with DCLS-style learnable synaptic delays.

    Replaces a standard nn.Linear by computing the input as a Gaussian-
    weighted combination of spike history taps:

        I[b, o] = sum_{n=0}^{Td-1} S[b, n, :] @ W_eff[n, o, :].T

    where W_eff[n, o, :] = W[o, :] * gauss_weight(d[o], n, sigma) for
    per_output granularity. The Gaussian weight is the fraction of the
    connection's weight attributed to delay tap n given continuous delay d.

    During training, the Gaussian blur allows gradients to flow smoothly
    from the loss through the delay parameter d. During inference (if
    discretize_at_inference=True), delays are rounded to integers and the
    sum collapses to a single tap per synapse/output.

    Weight initialisation follows Kaiming uniform (fan-in) to match
    nn.Linear defaults. Delays are initialised via init_delays().

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        delay_config: Full DelayConfig instance.
        bias: If True, add a learnable bias term to the output.
        delay_init_strategy: Delay initialisation method passed to
            init_delays(). One of 'uniform', 'normal_center', 'zeros'.

    Shape:
        Input:  x: (B, in_features), spike_history: (B, Td, in_features)
        Output: (B, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        delay_config: DelayConfig,
        bias: bool = True,
        delay_init_strategy: str = "uniform",
    ) -> None:
        # Determine d_raw shape from granularity BEFORE calling super().__init__
        d_shape = _delay_param_shape(
            out_features=out_features,
            in_features=in_features,
            granularity=delay_config.granularity,
            block_size=delay_config.block_size,
        )
        # Initialise delay values then convert to unconstrained d_raw
        d_init = init_delays(
            shape=d_shape,
            max_delay=delay_config.max_delay,
            strategy=delay_init_strategy,
        )
        is_fixed = delay_config.mode == "fixed_random"
        super().__init__(
            delay_config=delay_config,
            d_raw_init=d_init,
            fixed=is_fixed,
        )

        self.in_features = in_features
        self.out_features = out_features

        # Weight matrix: (out_features, in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            fan_in = in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            self.bias: Optional[nn.Parameter] = nn.Parameter(
                torch.empty(out_features, dtype=torch.float32)
            )
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        # Memory guard at construction time (no batch_size yet)
        self._check_memory(out_features, in_features)

    def _expand_delays_to_synapse(self, delays: Tensor) -> Tensor:
        """Expand delay tensor from granularity shape to (out_features, in_features).

        Args:
            delays: Continuous delays at the configured granularity.
                Shape depends on granularity:
                    per_synapse: (out, in)
                    per_output:  (out, 1)
                    per_input:   (1, in)
                    per_block:   (out_blocks, in_blocks)

        Returns:
            Delays expanded/repeated to (out_features, in_features). float32.
        """
        cfg = self.cfg
        if cfg.granularity == "per_synapse":
            return delays  # already (out, in)
        elif cfg.granularity == "per_output":
            return delays.expand(self.out_features, self.in_features)
        elif cfg.granularity == "per_input":
            return delays.expand(self.out_features, self.in_features)
        elif cfg.granularity == "per_block":
            # Repeat each block element to fill the full (out, in) grid
            expanded = delays.repeat_interleave(cfg.block_size, dim=0).repeat_interleave(
                cfg.block_size, dim=1
            )
            return expanded[: self.out_features, : self.in_features].contiguous()
        else:
            raise ValueError(f"Unknown granularity: {cfg.granularity}")

    def _forward_training(self, spike_history: Tensor) -> Tensor:
        """Gaussian-interpolated forward pass for training mode.

        Iterates over all Td delay positions. For each integer delay n, the
        contribution to the output is:
            out += (S[:, n, :] @ W_eff[n].T)

        where W_eff[n][o, i] = W[o, i] * g(n; d[o,i], sigma) / Z[o,i]
        and Z[o,i] = sum_{m=0}^{Td-1} g(m; d[o,i], sigma) is the normaliser
        ensuring effective weight magnitude is preserved regardless of sigma.

        Args:
            spike_history: (B, Td, in_features), fp32.

        Returns:
            current: (B, out_features), fp32.
        """
        B = spike_history.shape[0]
        sigma = float(self.sigma.item())
        td = self.cfg.max_delay
        device = spike_history.device

        # Get continuous delays and expand to (out, in)
        d_full = self._expand_delays_to_synapse(self.delays)  # (out, in)

        # Pre-compute normalisation factor Z[o, i] = sum_{n} g(n; d, sigma)
        # Compute outside loop: n_vals (Td,), d_full (out, in)
        n_vals = torch.arange(td, device=device, dtype=torch.float32)
        # diff_all[o, i, n] = d_full[o, i] - n
        diff_all = d_full.unsqueeze(-1) - n_vals  # (out, in, Td)
        sigma_safe = max(sigma, SIGMA_MIN)
        g_all = torch.exp(-(diff_all ** 2) / (2.0 * sigma_safe ** 2))  # (out, in, Td)
        Z = g_all.sum(dim=-1).clamp(min=1e-6)  # (out, in)

        # Accumulate over delay taps
        current = torch.zeros(B, self.out_features, device=device, dtype=torch.float32)
        for n in range(td):
            # Spikes at delay tap n: (B, in_features)
            s_n = spike_history[:, n, :]

            # Gaussian weight for tap n, normalised by Z: (out, in)
            g_n_norm = g_all[:, :, n] / Z  # (out, in), normalised contribution

            # Effective weight: W scaled by normalised Gaussian at this tap
            w_eff = self.weight * g_n_norm  # (out, in)

            # Accumulate: s_n @ w_eff.T => (B, in) @ (in, out) => (B, out)
            current = current + F.linear(s_n, w_eff)

        return current

    def _forward_inference(self, spike_history: Tensor) -> Tensor:
        """Discretised forward pass for inference (no Gaussian blur).

        Rounds each delay to the nearest integer and gathers spikes from
        that exact tap. This is the Td -> 0 limit of the Gaussian scheme.

        Args:
            spike_history: (B, Td, in_features), fp32.

        Returns:
            current: (B, out_features), fp32.
        """
        B = spike_history.shape[0]

        # Per-synapse integer delays (no grad)
        d_int = self._expand_delays_to_synapse(
            self.delays_discrete.float()
        ).long()  # (out, in)

        current = torch.zeros(
            B, self.out_features, device=spike_history.device, dtype=torch.float32
        )

        # Group synapses by unique delay to minimise gather operations
        unique_delays = d_int.unique()
        for nd in unique_delays:
            nd_int = int(nd.item())
            # Mask of synapses (out, in) with this delay
            mask = (d_int == nd_int)  # (out, in), bool
            # Spikes at this delay: (B, in)
            s_nd = spike_history[:, nd_int, :]
            # Weight contribution: W zeroed everywhere except masked positions
            w_masked = self.weight * mask.float()  # (out, in)
            current = current + F.linear(s_nd, w_masked)

        return current

    def forward(self, x: Tensor, spike_history: Tensor) -> Tensor:
        """Apply delayed linear transformation to spiking inputs.

        During training (self.training=True), uses Gaussian interpolation
        across K bins. During inference (self.training=False) with
        delay_config.discretize_at_inference=True, rounds delays to integers.

        The direct input x is NOT used by DelayLinear; it is included in
        the signature for compatibility with the cell_fn(x, history) contract
        used by snn_unroll. Callers that need both direct and delayed input
        should sum the output with a separate nn.Linear(x).

        Args:
            x: Direct input, shape (B, in_features). Unused; reserved for
                interface compatibility.
            spike_history: Spike history buffer, shape (B, Td, in_features).
                fp32. history[:, 0, :] is the oldest tap;
                history[:, -1, :] is the most recent.

        Returns:
            current: (B, out_features), fp32. Delayed input contribution.

        Raises:
            ValueError: If spike_history has wrong number of dimensions or
                delay dimension does not match max_delay.
        """
        if spike_history.dim() != 3:
            raise ValueError(
                f"DelayLinear expects spike_history with 3 dims "
                f"(B, Td, in_features), got shape {tuple(spike_history.shape)}"
            )
        if spike_history.shape[1] != self.cfg.max_delay:
            raise ValueError(
                f"DelayLinear: spike_history.shape[1]={spike_history.shape[1]} "
                f"!= max_delay={self.cfg.max_delay}"
            )
        if spike_history.shape[2] != self.in_features:
            raise ValueError(
                f"DelayLinear: spike_history.shape[2]={spike_history.shape[2]} "
                f"!= in_features={self.in_features}"
            )

        # Ensure fp32 for delay computation
        spike_history = spike_history.float()

        use_discrete = (not self.training) and self.cfg.discretize_at_inference
        if self.cfg.mode == "off":
            # No delays: use only the most-recent spike tap
            current = F.linear(spike_history[:, -1, :], self.weight)
        elif use_discrete:
            current = self._forward_inference(spike_history)
        else:
            current = self._forward_training(spike_history)

        # Add bias if present
        if self.bias is not None:
            current = current + self.bias

        return current

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            + super().extra_repr()
        )


# ===========================================================================
# SECTION 5: DelayConv1d
# ===========================================================================


class DelayConv1d(DelayModuleBase):
    """1D temporal convolution with DCLS-style learnable channel delays.

    Extends the DCLS Gaussian interpolation to convolutional feature maps.
    Each output channel o and input channel i has a learnable delay d[o, i]
    (or coarser per the granularity setting) that shifts the input signal
    before convolution.

    The forward pass computes:
        Y[b, o, t] = sum_{n=0}^{Td-1} Conv1d(H[:, n, :, :], W_eff[n])[b, o, t]

    where H[:, n, :, :] is the spike history at delay tap n (shape B, C_in, L)
    and W_eff[n] is the weight tensor scaled by the normalised Gaussian weight
    at tap n for each (out, in) channel pair.

    Args:
        in_channels: Number of input channels (C_in).
        out_channels: Number of output channels (C_out).
        kernel_size: Convolutional kernel length.
        delay_config: Full DelayConfig instance.
        stride: Convolution stride.
        padding: Convolution padding.
        dilation: Convolution dilation.
        bias: If True, add a learnable bias.
        delay_init_strategy: Delay initialisation strategy.

    Shape:
        Input:  x: (B, C_in, L), spike_history: (B, Td, C_in, L)
        Output: (B, C_out, L_out)
            L_out = floor((L + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        delay_config: DelayConfig,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        delay_init_strategy: str = "uniform",
    ) -> None:
        # Delay shape: treat conv channels as (out_features, in_features)
        d_shape = _delay_param_shape(
            out_features=out_channels,
            in_features=in_channels,
            granularity=delay_config.granularity,
            block_size=delay_config.block_size,
        )
        d_init = init_delays(
            shape=d_shape,
            max_delay=delay_config.max_delay,
            strategy=delay_init_strategy,
        )
        is_fixed = delay_config.mode == "fixed_random"
        super().__init__(
            delay_config=delay_config,
            d_raw_init=d_init,
            fixed=is_fixed,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Weight: (out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            fan_in = in_channels * kernel_size
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            self.bias: Optional[nn.Parameter] = nn.Parameter(
                torch.empty(out_channels, dtype=torch.float32)
            )
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        self._check_memory(out_channels, in_channels)

    def _expand_delays_to_channels(self, delays: Tensor) -> Tensor:
        """Expand delay tensor to (out_channels, in_channels).

        Args:
            delays: Delays at configured granularity.

        Returns:
            Delays expanded to (out_channels, in_channels), float32.
        """
        cfg = self.cfg
        if cfg.granularity == "per_synapse":
            return delays  # (out_ch, in_ch)
        elif cfg.granularity == "per_output":
            return delays.expand(self.out_channels, self.in_channels)
        elif cfg.granularity == "per_input":
            return delays.expand(self.out_channels, self.in_channels)
        elif cfg.granularity == "per_block":
            expanded = delays.repeat_interleave(cfg.block_size, dim=0).repeat_interleave(
                cfg.block_size, dim=1
            )
            return expanded[: self.out_channels, : self.in_channels].contiguous()
        else:
            raise ValueError(f"Unknown granularity: {cfg.granularity}")

    def _compute_output_length(self, L: int) -> int:
        """Compute the output length of the 1D convolution given input length L."""
        return math.floor(
            (L + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
            / self.stride
            + 1
        )

    def _forward_training(self, spike_history: Tensor) -> Tensor:
        """Gaussian-interpolated conv1d forward for training mode.

        For each delay tap n, scales the weight tensor by the per-channel
        normalised Gaussian weight and applies a conv1d to the spike history
        at tap n. The results are accumulated across taps.

        Args:
            spike_history: (B, Td, C_in, L), fp32.

        Returns:
            output: (B, C_out, L_out), fp32.
        """
        B, td, C_in, L = spike_history.shape
        sigma = float(self.sigma.item())
        device = spike_history.device
        L_out = self._compute_output_length(L)

        # Per-channel delays: (out_ch, in_ch)
        d_ch = self._expand_delays_to_channels(self.delays)

        # Pre-compute normalised Gaussian for all taps: (out_ch, in_ch, Td)
        n_vals = torch.arange(td, device=device, dtype=torch.float32)
        diff_all = d_ch.unsqueeze(-1) - n_vals  # (out, in, Td)
        sigma_safe = max(sigma, SIGMA_MIN)
        g_all = torch.exp(-(diff_all ** 2) / (2.0 * sigma_safe ** 2))
        Z = g_all.sum(dim=-1).clamp(min=1e-6)  # (out, in)

        output = torch.zeros(B, self.out_channels, L_out, device=device, dtype=torch.float32)
        for n in range(td):
            s_n = spike_history[:, n, :, :]  # (B, C_in, L)

            # Normalised Gaussian at tap n: (out, in)
            g_n_norm = g_all[:, :, n] / Z

            # Scale weight by per-channel Gaussian; unsqueeze for kernel dim
            w_eff = self.weight * g_n_norm.unsqueeze(-1)  # (out, in, K)

            out_n = F.conv1d(
                s_n, w_eff,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )  # (B, out_ch, L_out)
            output = output + out_n

        return output

    def _forward_inference(self, spike_history: Tensor) -> Tensor:
        """Discretised conv1d forward for inference mode.

        Groups input channels by their integer delay and processes each
        unique delay with a masked weight application.

        Args:
            spike_history: (B, Td, C_in, L), fp32.

        Returns:
            output: (B, C_out, L_out), fp32.
        """
        B, td, C_in, L = spike_history.shape
        device = spike_history.device
        L_out = self._compute_output_length(L)

        d_int = self._expand_delays_to_channels(
            self.delays_discrete.float()
        ).long()  # (out, in)

        output = torch.zeros(B, self.out_channels, L_out, device=device, dtype=torch.float32)

        unique_delays = d_int.unique()
        for nd in unique_delays:
            nd_int = int(nd.item())
            s_nd = spike_history[:, nd_int, :, :]  # (B, C_in, L)
            # Mask weight: zero channels not assigned to this delay
            mask = (d_int == nd_int).float().unsqueeze(-1)  # (out, in, 1)
            w_masked = self.weight * mask  # (out, in, K)
            out_nd = F.conv1d(
                s_nd, w_masked,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            output = output + out_nd

        return output

    def forward(self, x: Tensor, spike_history: Tensor) -> Tensor:
        """Apply delayed 1D convolution to spiking input channels.

        Args:
            x: Direct input, shape (B, C_in, L). Unused; reserved for
                interface compatibility with snn_unroll cell_fn contract.
            spike_history: (B, Td, C_in, L), fp32.

        Returns:
            output: (B, C_out, L_out), fp32.

        Raises:
            ValueError: If spike_history has wrong shape.
        """
        if spike_history.dim() != 4:
            raise ValueError(
                f"DelayConv1d expects spike_history with 4 dims "
                f"(B, Td, C_in, L), got shape {tuple(spike_history.shape)}"
            )
        if spike_history.shape[1] != self.cfg.max_delay:
            raise ValueError(
                f"DelayConv1d: spike_history.shape[1]={spike_history.shape[1]} "
                f"!= max_delay={self.cfg.max_delay}"
            )
        if spike_history.shape[2] != self.in_channels:
            raise ValueError(
                f"DelayConv1d: spike_history.shape[2]={spike_history.shape[2]} "
                f"!= in_channels={self.in_channels}"
            )

        spike_history = spike_history.float()

        use_discrete = (not self.training) and self.cfg.discretize_at_inference
        if self.cfg.mode == "off":
            output = F.conv1d(
                spike_history[:, -1, :, :], self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        elif use_discrete:
            output = self._forward_inference(spike_history)
        else:
            output = self._forward_training(spike_history)

        if self.bias is not None:
            output = output + self.bias.unsqueeze(0).unsqueeze(-1)

        return output

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"bias={self.bias is not None}, "
            + super().extra_repr()
        )


# ===========================================================================
# SECTION 6: SpikeHistoryBuffer utility
# ===========================================================================


class SpikeHistoryBuffer:
    """Static utility class for managing spike history circular buffers.

    The spike history buffer is a tensor of shape (B, Td, *feature_shape)
    where dimension 1 is the time axis. The oldest spike is at index 0 and
    the most recent is at index Td-1 (oldest-first convention, matching the
    spike_history field in SpikingState).

    All methods are static and operate on plain tensors to avoid hidden state
    and enable easy integration with existing snn_unroll cell_fn patterns.

    No in-place operations are used — all methods return new tensors.
    """

    @staticmethod
    def create(
        batch_size: int,
        max_delay: int,
        feature_shape: Tuple[int, ...],
        device: Union[torch.device, str],
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Allocate a zero-initialised spike history buffer.

        Args:
            batch_size: Number of samples in the batch (B).
            max_delay: Depth of the delay buffer (Td).
            feature_shape: Shape of the feature at each timestep, e.g.
                (in_features,) for linear or (C_in, L) for conv.
            device: Target device.
            dtype: Buffer dtype. Typically torch.float32.

        Returns:
            history: Zero tensor of shape (B, Td, *feature_shape).
        """
        shape = (batch_size, max_delay) + feature_shape
        return torch.zeros(shape, device=device, dtype=dtype)

    @staticmethod
    def update(history: Tensor, new_spikes: Tensor) -> Tensor:
        """Push new spikes onto the buffer, dropping the oldest entry.

        Maintains oldest-first ordering: after the update, history[:, -1, :]
        contains new_spikes and history[:, 0, :] contains what was previously
        history[:, 1, :].

        No in-place ops — creates a new tensor via torch.cat. This is safe
        for autograd and compatible with torch.compile. For large buffers
        at high batch sizes, consider using a pre-allocated ring buffer with
        index arithmetic (see TODO in AdvancedLIFNeuron docstring).

        Args:
            history: Current buffer, shape (B, Td, *feature_shape).
            new_spikes: Spikes at the current timestep, shape (B, *feature_shape).

        Returns:
            updated_history: New buffer of the same shape, with new_spikes
                appended at position Td-1 and the oldest entry removed.

        Example::

            history = SpikeHistoryBuffer.create(4, 16, (512,), device)
            for t in range(T):
                spikes, state = neuron(x_t, state)
                history = SpikeHistoryBuffer.update(history, spikes)
                current = delay_linear(x_t, history)
        """
        # new_spikes: (B, *feat) -> (B, 1, *feat) for concat along dim=1
        new_entry = new_spikes.unsqueeze(1)
        return torch.cat([history[:, 1:], new_entry], dim=1)

    @staticmethod
    def detach(history: Tensor) -> Tensor:
        """Detach the history buffer from the computation graph.

        Use at truncated-BPTT boundaries to free the retained graph while
        keeping the spike values for the next forward pass chunk.

        Args:
            history: Current buffer, shape (B, Td, *feature_shape).

        Returns:
            Detached tensor with the same values but no gradient tracking.
        """
        return history.detach()

    @staticmethod
    def get_tap(history: Tensor, delay: int) -> Tensor:
        """Extract the spike vector at a specific integer delay tap.

        Convenience accessor for diagnostics and fixed-delay baselines.
        Delay 0 means the most recently pushed spikes (history[:, -1, :]),
        delay 1 means one step ago (history[:, -2, :]), etc.

        Args:
            history: Buffer of shape (B, Td, *feature_shape).
            delay: Integer delay in [0, Td-1]. 0 = most recent.

        Returns:
            Tensor of shape (B, *feature_shape).
        """
        td = history.shape[1]
        if not (0 <= delay < td):
            raise ValueError(
                f"delay must be in [0, {td - 1}], got {delay}"
            )
        # Most recent is at index -1; delay 0 maps to index -1
        idx = td - 1 - delay
        return history[:, idx]


# ===========================================================================
# SECTION 7: Delay initialisation
# ===========================================================================


def init_delays(
    shape: Tuple[int, ...],
    max_delay: int,
    strategy: str = "uniform",
    seed: Optional[int] = None,
) -> Tensor:
    """Initialise delay values and convert to unconstrained d_raw (logit space).

    The returned tensor is in unconstrained space — it is the d_raw parameter
    that will be stored in DelayModuleBase.d_raw. The forward pass converts it
    back to delays via:
        d = (Td - 1) * sigmoid(d_raw)

    Args:
        shape: Shape of the delay parameter tensor. Determined by granularity;
            e.g. (out_features, 1) for per_output, (out, in) for per_synapse.
        max_delay: Maximum delay (Td). Delays are in [0, Td-1].
        strategy: Initialisation method for the delay values (in [0, Td-1]):
            'uniform': d ~ Uniform(0, Td-1). Maximises initial diversity.
            'normal_center': d ~ N(Td/2, Td/6), clamped to [0, Td-1].
                Concentrates delays near the middle of the range.
            'zeros': All delays start at 0 (minimum delay). Equivalent to
                the baseline of using only the most recent spike.
        seed: Optional random seed for reproducibility. The seed is applied
            locally via a manual_seed generator and does not affect the global
            random state.

    Returns:
        d_raw: Float tensor of shape `shape`, in unconstrained logit space.
            Safe to wrap in nn.Parameter immediately.

    Notes:
        - When max_delay == 1 (Td-1 == 0), all strategies return zeros since
          there is only one valid delay value.
        - The logit (inverse sigmoid) conversion clips delay values away from
          the exact boundaries [0, Td-1] by a small epsilon before taking the
          logit to avoid d_raw = +-inf.
    """
    if strategy not in INIT_STRATEGIES:
        raise ValueError(
            f"init_delays: strategy must be one of {INIT_STRATEGIES}, "
            f"got '{strategy}'"
        )
    if max_delay < 1:
        raise ValueError(f"init_delays: max_delay must be >= 1, got {max_delay}")

    td_max = max_delay - 1
    if td_max == 0:
        # Only one valid delay; d_raw = 0 -> sigmoid(0) = 0.5 -> d = 0
        return torch.zeros(shape, dtype=torch.float32)

    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    else:
        gen = None

    if strategy == "uniform":
        if gen is not None:
            d_init = torch.rand(shape, generator=gen) * td_max
        else:
            d_init = torch.rand(shape) * td_max
    elif strategy == "normal_center":
        center = td_max / 2.0
        std = td_max / 6.0
        if gen is not None:
            d_init = torch.randn(shape, generator=gen) * std + center
        else:
            d_init = torch.randn(shape) * std + center
        d_init = d_init.clamp(0.0, float(td_max))
    elif strategy == "zeros":
        d_init = torch.zeros(shape, dtype=torch.float32)
    else:
        # Unreachable; validated above
        d_init = torch.zeros(shape, dtype=torch.float32)

    # Convert d_init in [0, Td-1] to d_raw = logit(d / (Td-1))
    # Clip to avoid logit(0) = -inf and logit(1) = +inf
    eps = 1e-4
    d_normalised = d_init.float() / float(td_max)
    d_clipped = d_normalised.clamp(eps, 1.0 - eps)
    # logit(p) = log(p / (1 - p))
    d_raw = torch.log(d_clipped / (1.0 - d_clipped))

    return d_raw


def _delay_param_shape(
    out_features: int,
    in_features: int,
    granularity: str,
    block_size: int,
) -> Tuple[int, ...]:
    """Compute the shape of the d_raw parameter tensor for a given granularity.

    This is an internal helper used by DelayLinear and DelayConv1d to
    compute the correct d_raw shape before calling super().__init__.

    Args:
        out_features: Output dimension (neurons or channels).
        in_features: Input dimension (neurons or channels).
        granularity: One of GRANULARITY_MODES.
        block_size: Block size for 'per_block' granularity.

    Returns:
        shape: Tuple of ints representing d_raw.shape.
    """
    if granularity == "per_synapse":
        return (out_features, in_features)
    elif granularity == "per_output":
        return (out_features, 1)
    elif granularity == "per_input":
        return (1, in_features)
    elif granularity == "per_block":
        out_blocks = math.ceil(out_features / block_size)
        in_blocks = math.ceil(in_features / block_size)
        return (out_blocks, in_blocks)
    else:
        raise ValueError(
            f"Unknown granularity '{granularity}'. "
            f"Must be one of {GRANULARITY_MODES}"
        )


# ===========================================================================
# SECTION 8: Memory and performance guards
# ===========================================================================


def check_delay_memory(
    out_features: int,
    in_features: int,
    max_delay: int,
    granularity: str,
    block_size: int = 64,
    batch_size: Optional[int] = None,
    dtype_bytes: int = 4,
) -> Dict[str, Union[int, float, str]]:
    """Estimate memory usage for a DelayLinear or DelayConv1d module.

    Computes the size of the major tensors involved in the delay forward
    pass and returns a dictionary of estimates for logging or pre-flight
    checks.

    Args:
        out_features: Output dimension.
        in_features: Input dimension.
        max_delay: Number of delay taps Td.
        granularity: Parameter sharing mode (see GRANULARITY_MODES).
        block_size: Block size for per_block granularity.
        batch_size: Optional batch size for history buffer estimate.
        dtype_bytes: Bytes per element (4 for float32, 2 for float16).

    Returns:
        Dict with keys:
            'weight_elements': Number of weight matrix elements.
            'weight_mb': Weight memory in MB.
            'd_raw_elements': Number of delay parameter elements.
            'd_raw_mb': d_raw parameter memory in MB.
            'history_elements': History buffer elements (if batch_size given).
            'history_mb': History buffer memory in MB (if batch_size given).
            'total_mb': Approximate total MB (weight + d_raw + history if given).
            'warning': Non-empty string if memory is potentially excessive.
    """
    weight_elements = out_features * in_features
    d_shape = _delay_param_shape(out_features, in_features, granularity, block_size)
    d_raw_elements = 1
    for dim in d_shape:
        d_raw_elements *= dim

    weight_mb = (weight_elements * dtype_bytes) / 1e6
    d_raw_mb = (d_raw_elements * dtype_bytes) / 1e6

    result: Dict[str, Union[int, float, str]] = {
        "weight_elements": weight_elements,
        "weight_mb": weight_mb,
        "d_raw_elements": d_raw_elements,
        "d_raw_mb": d_raw_mb,
        "warning": "",
    }

    total_mb = weight_mb + d_raw_mb

    if batch_size is not None:
        history_elements = batch_size * max_delay * in_features
        history_mb = (history_elements * dtype_bytes) / 1e6
        result["history_elements"] = history_elements
        result["history_mb"] = history_mb
        total_mb += history_mb

    result["total_mb"] = total_mb

    # Generate warning string if thresholds exceeded
    warnings_list = []
    if out_features * in_features * max_delay > 100_000_000:
        warnings_list.append(
            f"out({out_features}) * in({in_features}) * Td({max_delay}) "
            f"= {out_features * in_features * max_delay:,} > 100M elements"
        )
    if total_mb > 1000:
        warnings_list.append(f"Estimated total memory ~{total_mb:.0f} MB")
    if warnings_list:
        result["warning"] = "; ".join(warnings_list)

    return result


def profile_delay_overhead(
    module: DelayModuleBase,
    input_shape: Tuple[int, ...],
    history_shape: Tuple[int, ...],
    num_runs: int = 10,
    device: Optional[Union[str, torch.device]] = None,
    warmup: int = 3,
) -> Dict[str, float]:
    """Measure forward-pass time of a delay module.

    Benchmarks the module forward pass with realistic random inputs and
    returns timing statistics for logging or pre-deployment checks.

    Args:
        module: A DelayLinear or DelayConv1d instance to profile.
        input_shape: Shape of the direct input tensor (B, in_features) or
            (B, C_in, L) for conv.
        history_shape: Shape of the spike history (B, Td, in_features) or
            (B, Td, C_in, L) for conv.
        num_runs: Number of timed forward passes.
        device: Device to run on. Defaults to module's current device.
        warmup: Number of warmup runs before timing starts.

    Returns:
        Dict with keys:
            'delay_ms_mean': Mean forward time with delays (milliseconds).
            'delay_ms_std': Standard deviation.
            'delay_ms_min': Minimum time.
            'delay_ms_max': Maximum time.
            'num_runs': Number of timed runs (as float for dict homogeneity).
    """
    import time

    if device is None:
        try:
            device = next(module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    device = torch.device(device)
    module = module.to(device)

    x = torch.randn(input_shape, device=device, dtype=torch.float32)
    history = torch.randn(history_shape, device=device, dtype=torch.float32)

    # Warmup passes
    for _ in range(warmup):
        with torch.no_grad():
            _ = module(x, history)

    if device.type == "cuda":
        torch.cuda.synchronize()

    delay_times: List[float] = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = module(x, history)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        delay_times.append((t1 - t0) * 1000.0)

    delay_arr = torch.tensor(delay_times)
    return {
        "delay_ms_mean": float(delay_arr.mean().item()),
        "delay_ms_std": float(delay_arr.std().item()) if len(delay_times) > 1 else 0.0,
        "delay_ms_min": float(delay_arr.min().item()),
        "delay_ms_max": float(delay_arr.max().item()),
        "num_runs": float(num_runs),
    }


# ===========================================================================
# SECTION 9: Module-level convenience exports
# ===========================================================================


__all__ = [
    # Config
    "DelayConfig",
    # Helpers
    "gaussian_kernel",
    "init_delays",
    "check_delay_memory",
    "profile_delay_overhead",
    # Abstract base
    "DelayModuleBase",
    # Concrete modules
    "DelayLinear",
    "DelayConv1d",
    # Utilities
    "SpikeHistoryBuffer",
    # Constants
    "SIGMA_MIN",
    "DRAW_CLAMP",
    "GRANULARITY_MODES",
    "DELAY_MODES",
    "SIGMA_SCHEDULES",
    "INIT_STRATEGIES",
]


# ===========================================================================
# SECTION 10: Self-test block
# ===========================================================================


def _run_self_tests() -> None:
    """Run all self-tests for the delay module template.

    Call directly: python delay_module_template.py
    """
    import sys

    print("=" * 72)
    print("delay_module_template.py — self-test suite")
    print("=" * 72)

    _PASS = "PASS"
    _FAIL = "FAIL"
    _results: List[Tuple[str, str, str]] = []

    def _check(name: str, condition: bool, details: str = "") -> None:
        status = _PASS if condition else _FAIL
        _results.append((name, status, details))
        symbol = "+" if condition else "X"
        print(f"  [{symbol}] {name}" + (f" -- {details}" if details else ""))

    device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Test 1: DelayLinear forward shapes (training mode)
    # ------------------------------------------------------------------
    print("\n[1] DelayLinear forward shapes")
    try:
        B, IN, OUT, Td = 4, 64, 128, 16
        cfg = DelayConfig(mode="learnable_dcls", max_delay=Td, num_bins=3,
                          granularity="per_output", sigma_start=1.0, sigma_end=0.5)
        dl = DelayLinear(in_features=IN, out_features=OUT, delay_config=cfg)
        dl.train()
        x = torch.randn(B, IN)
        history = SpikeHistoryBuffer.create(B, Td, (IN,), device)
        out = dl(x, history)
        _check(
            "DelayLinear output shape",
            out.shape == (B, OUT),
            f"expected ({B}, {OUT}), got {tuple(out.shape)}",
        )
        _check(
            "DelayLinear output is finite",
            torch.isfinite(out).all().item(),
            f"mean={out.mean().item():.4f}",
        )
        _check(
            "DelayLinear weight shape",
            dl.weight.shape == (OUT, IN),
            str(tuple(dl.weight.shape)),
        )
        _check(
            "DelayLinear d_raw shape for per_output",
            dl.d_raw.shape == (OUT, 1),
            str(tuple(dl.d_raw.shape)),
        )
        _check(
            "DelayLinear delays in [0, Td-1]",
            bool((dl.delays >= 0).all() and (dl.delays <= Td - 1).all()),
            f"min={dl.delays.min().item():.3f}, max={dl.delays.max().item():.3f}",
        )
    except Exception as exc:
        _check("DelayLinear forward", False, f"EXCEPTION: {exc}")

    # ------------------------------------------------------------------
    # Test 2: DelayConv1d forward shapes
    # ------------------------------------------------------------------
    print("\n[2] DelayConv1d forward shapes")
    try:
        B, C_IN, C_OUT, L, Td = 2, 8, 16, 64, 8
        K_sz = 3
        cfg2 = DelayConfig(mode="learnable_dcls", max_delay=Td, num_bins=3,
                           granularity="per_output", sigma_start=1.2, sigma_end=0.4)
        dc = DelayConv1d(
            in_channels=C_IN, out_channels=C_OUT, kernel_size=K_sz,
            delay_config=cfg2, padding=1,
        )
        dc.train()
        x2 = torch.randn(B, C_IN, L)
        hist2 = SpikeHistoryBuffer.create(B, Td, (C_IN, L), device)
        out2 = dc(x2, hist2)
        expected_L_out = L  # padding=1, kernel=3, stride=1 -> same length
        _check(
            "DelayConv1d output shape",
            out2.shape == (B, C_OUT, expected_L_out),
            f"expected ({B}, {C_OUT}, {expected_L_out}), got {tuple(out2.shape)}",
        )
        _check(
            "DelayConv1d output is finite",
            torch.isfinite(out2).all().item(),
            f"mean={out2.mean().item():.4f}",
        )
        _check(
            "DelayConv1d weight shape",
            dc.weight.shape == (C_OUT, C_IN, K_sz),
            str(tuple(dc.weight.shape)),
        )
    except Exception as exc:
        _check("DelayConv1d forward", False, f"EXCEPTION: {exc}")

    # ------------------------------------------------------------------
    # Test 3: Gaussian kernel normalisation
    # ------------------------------------------------------------------
    print("\n[3] Gaussian kernel normalisation")
    try:
        positions = torch.tensor([2.3, 7.9, 0.1, 15.5])
        centers = torch.tensor([6.0, 7.0, 8.0])  # K=3 float centers
        for sigma_val in [0.5, 1.0, 2.0, 0.01]:
            w = gaussian_kernel(positions, centers, sigma_val)
            row_sums = w.sum(dim=-1)
            all_sum_to_one = torch.allclose(row_sums, torch.ones(4), atol=1e-5)
            _check(
                f"Gaussian kernel sums to 1 (sigma={sigma_val})",
                all_sum_to_one,
                f"row_sums={row_sums.tolist()}",
            )
            all_positive = (w >= 0).all().item()
            _check(
                f"Gaussian kernel non-negative (sigma={sigma_val})",
                bool(all_positive),
                "",
            )

        # Near-zero sigma: should approach one-hot at nearest center
        w_sharp = gaussian_kernel(torch.tensor([7.1]), torch.tensor([6.0, 7.0, 8.0]), 0.001)
        peak_idx = int(w_sharp.argmax(dim=-1).item())
        _check(
            "Gaussian kernel sharp sigma approaches one-hot",
            peak_idx == 1,  # nearest center to 7.1 is 7.0 at index 1
            f"peak at index {peak_idx}, weights={w_sharp.tolist()}",
        )
    except Exception as exc:
        _check("Gaussian kernel", False, f"EXCEPTION: {exc}")

    # ------------------------------------------------------------------
    # Test 4: Inference discretisation vs training output
    # ------------------------------------------------------------------
    print("\n[4] Inference discretisation vs training output")
    try:
        B, IN, OUT, Td = 2, 16, 32, 8
        cfg4 = DelayConfig(mode="learnable_dcls", max_delay=Td, num_bins=3,
                           granularity="per_output", sigma_start=0.01,
                           discretize_at_inference=True)
        dl4 = DelayLinear(in_features=IN, out_features=OUT, delay_config=cfg4)

        history4 = torch.randn(B, Td, IN)
        x4 = torch.zeros(B, IN)

        # Training mode
        dl4.train()
        out_train = dl4(x4, history4)

        # Inference mode (discretised): set training=False
        dl4.training = False
        out_inf = dl4(x4, history4)

        _check(
            "Inference output shape matches training",
            out_inf.shape == out_train.shape,
            f"{tuple(out_inf.shape)} vs {tuple(out_train.shape)}",
        )
        _check(
            "Inference output is finite",
            torch.isfinite(out_inf).all().item(),
            f"mean={out_inf.mean().item():.4f}",
        )
        # With very small sigma, train and inference should be close
        max_diff = (out_train - out_inf).abs().max().item()
        _check(
            "Inference and training outputs close (small sigma)",
            max_diff < 0.5,
            f"max_diff={max_diff:.4f}",
        )
        # Restore training mode
        dl4.train()
    except Exception as exc:
        _check("Inference discretisation", False, f"EXCEPTION: {exc}")

    # ------------------------------------------------------------------
    # Test 5: Gradient flow through d_raw
    # ------------------------------------------------------------------
    print("\n[5] Gradient flow through d_raw")
    try:
        B, IN, OUT, Td = 2, 8, 4, 4
        cfg5 = DelayConfig(mode="learnable_dcls", max_delay=Td, num_bins=3,
                           granularity="per_output", sigma_start=1.0)
        dl5 = DelayLinear(in_features=IN, out_features=OUT, delay_config=cfg5)
        dl5.train()

        history5 = torch.rand(B, Td, IN, requires_grad=False)
        x5 = torch.zeros(B, IN)

        out5 = dl5(x5, history5)
        loss = out5.sum()
        loss.backward()

        _check(
            "d_raw.grad is not None",
            dl5.d_raw.grad is not None,
            "",
        )
        if dl5.d_raw.grad is not None:
            _check(
                "d_raw.grad is finite",
                torch.isfinite(dl5.d_raw.grad).all().item(),
                f"grad norm={dl5.d_raw.grad.norm().item():.4f}",
            )
            _check(
                "d_raw.grad has nonzero values",
                (dl5.d_raw.grad.abs() > 0).any().item(),
                f"max_abs={dl5.d_raw.grad.abs().max().item():.6f}",
            )
        _check(
            "weight.grad is not None",
            dl5.weight.grad is not None,
            "",
        )
        if dl5.weight.grad is not None:
            _check(
                "weight.grad is finite",
                torch.isfinite(dl5.weight.grad).all().item(),
                "",
            )
    except Exception as exc:
        _check("Gradient flow", False, f"EXCEPTION: {exc}")

    # ------------------------------------------------------------------
    # Test 6: Sigma schedule correctness
    # ------------------------------------------------------------------
    print("\n[6] Sigma schedule correctness")
    try:
        cfg6 = DelayConfig(
            mode="learnable_dcls", max_delay=8,
            sigma_start=2.0, sigma_end=0.5,
            sigma_decay_epochs=10,
            sigma_schedule="decreasing",
        )
        dl6 = DelayLinear(in_features=4, out_features=4, delay_config=cfg6)

        # Epoch 0: should be sigma_start
        s0 = dl6.update_sigma(0)
        _check("Sigma at epoch 0 = sigma_start", abs(s0 - 2.0) < 1e-5, f"got {s0}")

        # Epoch 5: midpoint, t=0.5, sigma = 2.0 + 0.5*(0.5-2.0) = 1.25
        s5 = dl6.update_sigma(5)
        expected_s5 = 2.0 + 0.5 * (0.5 - 2.0)  # 1.25
        _check(
            "Sigma at epoch 5 = midpoint",
            abs(s5 - expected_s5) < 1e-4,
            f"expected {expected_s5:.4f}, got {s5:.4f}",
        )

        # Epoch 10: should equal sigma_end
        s10 = dl6.update_sigma(10)
        _check("Sigma at epoch 10 = sigma_end", abs(s10 - 0.5) < 1e-5, f"got {s10}")

        # Epoch 100: should stay at sigma_end (clamped at decay_epochs)
        s100 = dl6.update_sigma(100)
        _check("Sigma at epoch 100 = sigma_end (clamped)", abs(s100 - 0.5) < 1e-5, f"got {s100}")

        # Idempotency: calling update_sigma twice with same epoch
        dl6.update_sigma(5)
        s5b = dl6.update_sigma(5)
        _check("Sigma update is idempotent", abs(s5b - expected_s5) < 1e-4, f"got {s5b}")

        # Constant schedule
        cfg6c = DelayConfig(
            mode="learnable_dcls", max_delay=8,
            sigma_start=1.5, sigma_schedule="constant",
        )
        dl6c = DelayLinear(in_features=4, out_features=4, delay_config=cfg6c)
        sc = dl6c.update_sigma(999)
        _check("Constant schedule: sigma unchanged", abs(sc - 1.5) < 1e-5, f"got {sc}")

    except Exception as exc:
        _check("Sigma schedule", False, f"EXCEPTION: {exc}")

    # ------------------------------------------------------------------
    # Test 7: Memory guard warning
    # ------------------------------------------------------------------
    print("\n[7] Memory guard warning")
    try:
        cfg7 = DelayConfig(
            mode="learnable_dcls", max_delay=16,
            memory_warn_threshold=100,  # Very low threshold to trigger warning
        )
        # 32 * 32 * 16 = 16384 > 100 should trigger ResourceWarning
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            _dl7 = DelayLinear(in_features=32, out_features=32, delay_config=cfg7)

        warning_triggered = any(issubclass(wn.category, ResourceWarning) for wn in w_list)
        _check(
            "ResourceWarning raised for large module",
            warning_triggered,
            f"caught {len(w_list)} warning(s)",
        )

        # check_delay_memory function
        mem_info = check_delay_memory(
            out_features=512, in_features=512, max_delay=16,
            granularity="per_output", batch_size=32,
        )
        _check(
            "check_delay_memory returns dict",
            isinstance(mem_info, dict) and "weight_mb" in mem_info,
            f"keys={list(mem_info.keys())}",
        )
        _check(
            "check_delay_memory weight_elements correct",
            mem_info["weight_elements"] == 512 * 512,
            f"got {mem_info['weight_elements']}",
        )
        _check(
            "check_delay_memory history_mb present with batch_size",
            "history_mb" in mem_info,
            f"keys={list(mem_info.keys())}",
        )

    except Exception as exc:
        _check("Memory guard", False, f"EXCEPTION: {exc}")

    # ------------------------------------------------------------------
    # Test 8: SpikeHistoryBuffer update (no in-place ops)
    # ------------------------------------------------------------------
    print("\n[8] SpikeHistoryBuffer update")
    try:
        B8, Td8, N8 = 3, 4, 8
        hist8 = SpikeHistoryBuffer.create(B8, Td8, (N8,), device)
        _check(
            "History buffer shape",
            hist8.shape == (B8, Td8, N8),
            str(tuple(hist8.shape)),
        )
        new_spk = torch.ones(B8, N8)
        hist8_new = SpikeHistoryBuffer.update(hist8, new_spk)
        _check(
            "Updated history shape preserved",
            hist8_new.shape == (B8, Td8, N8),
            str(tuple(hist8_new.shape)),
        )
        _check(
            "Updated history last tap is new spikes",
            torch.allclose(hist8_new[:, -1, :], new_spk),
            "",
        )
        _check(
            "Updated history first tap shifted",
            torch.allclose(hist8_new[:, 0, :], hist8[:, 1, :]),
            "",
        )
        # get_tap: delay=0 should return most recent (last)
        tap0 = SpikeHistoryBuffer.get_tap(hist8_new, delay=0)
        _check(
            "get_tap(delay=0) returns most recent spike",
            torch.allclose(tap0, new_spk),
            "",
        )
        # detach does not raise
        hist8_d = SpikeHistoryBuffer.detach(hist8_new)
        _check("detach produces same values", torch.allclose(hist8_d, hist8_new), "")
        _check("detach removes grad tracking", not hist8_d.requires_grad, "")
    except Exception as exc:
        _check("SpikeHistoryBuffer", False, f"EXCEPTION: {exc}")

    # ------------------------------------------------------------------
    # Test 9: init_delays strategies
    # ------------------------------------------------------------------
    print("\n[9] init_delays strategies")
    try:
        shape9 = (16, 8)
        Td9 = 10
        for strat in INIT_STRATEGIES:
            d_raw = init_delays(shape9, max_delay=Td9, strategy=strat, seed=42)
            d = (Td9 - 1) * torch.sigmoid(d_raw)
            _check(
                f"init_delays '{strat}': delays in [0, {Td9-1}]",
                bool((d >= 0).all() and (d <= Td9 - 1).all()),
                f"min={d.min().item():.3f}, max={d.max().item():.3f}",
            )
            _check(
                f"init_delays '{strat}': shape correct",
                d_raw.shape == shape9,
                str(tuple(d_raw.shape)),
            )
        # Zeros strategy: all delays should be near 0
        d_raw_z = init_delays((4,), max_delay=8, strategy="zeros")
        d_z = 7.0 * torch.sigmoid(d_raw_z)
        _check(
            "init_delays zeros: delays near 0",
            (d_z < 0.1).all().item(),
            f"values={d_z.tolist()}",
        )
        # Seed reproducibility
        d1 = init_delays((5, 5), max_delay=12, strategy="uniform", seed=7)
        d2 = init_delays((5, 5), max_delay=12, strategy="uniform", seed=7)
        _check(
            "init_delays seed reproducibility",
            torch.allclose(d1, d2),
            "",
        )
    except Exception as exc:
        _check("init_delays", False, f"EXCEPTION: {exc}")

    # ------------------------------------------------------------------
    # Test 10: get_delay_diagnostics
    # ------------------------------------------------------------------
    print("\n[10] get_delay_diagnostics")
    try:
        cfg10 = DelayConfig(mode="learnable_dcls", max_delay=8, granularity="per_output")
        dl10 = DelayLinear(in_features=16, out_features=32, delay_config=cfg10)
        dl10.update_sigma(3)
        diag = dl10.get_delay_diagnostics()
        required_keys = {"delay_mean", "delay_std", "delay_min", "delay_max",
                         "delay_entropy", "boundary_pct", "sigma", "sigma_epoch"}
        _check(
            "Diagnostics contains all required keys",
            required_keys.issubset(diag.keys()),
            f"missing: {required_keys - set(diag.keys())}",
        )
        _check(
            "Diagnostics delay_mean in [0, Td-1]",
            0.0 <= diag["delay_mean"] <= 7.0,
            f"mean={diag['delay_mean']:.3f}",
        )
        _check(
            "Diagnostics delay_entropy in [0, 1]",
            0.0 <= diag["delay_entropy"] <= 1.0,
            f"entropy={diag['delay_entropy']:.3f}",
        )
        _check(
            "Diagnostics sigma_epoch = 3",
            diag["sigma_epoch"] == 3,
            f"got {diag['sigma_epoch']}",
        )
    except Exception as exc:
        _check("get_delay_diagnostics", False, f"EXCEPTION: {exc}")

    # ------------------------------------------------------------------
    # Test 11: DelayConfig validation
    # ------------------------------------------------------------------
    print("\n[11] DelayConfig validation")
    try:
        # Valid construction should not raise
        cfg_ok = DelayConfig()
        _check("DelayConfig default construction", True, "")

        # Invalid mode
        try:
            DelayConfig(mode="bad_mode")
            _check("DelayConfig rejects bad mode", False, "no exception raised")
        except ValueError:
            _check("DelayConfig rejects bad mode", True, "ValueError raised")

        # Invalid num_bins (even)
        try:
            DelayConfig(num_bins=4)
            _check("DelayConfig rejects even num_bins", False, "no exception raised")
        except ValueError:
            _check("DelayConfig rejects even num_bins", True, "ValueError raised")

        # Convenience constructors
        cfg_off = DelayConfig.off()
        _check("DelayConfig.off() mode='off'", cfg_off.mode == "off", "")
        cfg_fixed = DelayConfig.fixed(max_delay=8)
        _check("DelayConfig.fixed() mode='fixed_random'", cfg_fixed.mode == "fixed_random", "")
        cfg_learn = DelayConfig.learnable(max_delay=12)
        _check("DelayConfig.learnable() mode='learnable_dcls'",
               cfg_learn.mode == "learnable_dcls", "")

    except Exception as exc:
        _check("DelayConfig validation", False, f"EXCEPTION: {exc}")

    # ------------------------------------------------------------------
    # Test 12: Granularity shapes
    # ------------------------------------------------------------------
    print("\n[12] Granularity shapes for d_raw")
    try:
        OUT12, IN12 = 64, 32
        granularity_to_shape = {
            "per_synapse": (OUT12, IN12),
            "per_output": (OUT12, 1),
            "per_input": (1, IN12),
            "per_block": (
                math.ceil(OUT12 / 64), math.ceil(IN12 / 64)
            ),
        }
        for gran, expected_shape in granularity_to_shape.items():
            cfg_g = DelayConfig(mode="learnable_dcls", max_delay=8, granularity=gran)
            dl_g = DelayLinear(in_features=IN12, out_features=OUT12, delay_config=cfg_g)
            _check(
                f"d_raw shape for granularity='{gran}'",
                dl_g.d_raw.shape == expected_shape,
                f"expected {expected_shape}, got {tuple(dl_g.d_raw.shape)}",
            )
    except Exception as exc:
        _check("Granularity shapes", False, f"EXCEPTION: {exc}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    n_pass = sum(1 for _, s, _ in _results if s == _PASS)
    n_fail = sum(1 for _, s, _ in _results if s == _FAIL)
    print(f"Results: {n_pass} passed, {n_fail} failed out of {len(_results)} checks")

    if n_fail > 0:
        print("\nFailed checks:")
        for name, status, details in _results:
            if status == _FAIL:
                print(f"  [X] {name}: {details}")
        sys.exit(1)
    else:
        print("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    _run_self_tests()
