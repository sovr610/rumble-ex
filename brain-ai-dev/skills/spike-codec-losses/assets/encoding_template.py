"""
brain_ai/core/encoding.py -- Spike Encoding Suite (Batch-First Convention)

Converts continuous-valued tensors into discrete spike trains and bundles them
in a structured SpikeBatch dataclass.  This module is the **input gateway** to
the spiking neural network core: every modality encoder feeds through one of
these spike encoders before signals reach LIF neurons.

Axis convention (IMPORTANT -- differs from v1):
    All spike tensors are **(B, T, ...)** -- batch dimension first, time second.
    This aligns with standard PyTorch convention (DataLoader yields batch-first)
    and avoids costly permutations in the training loop.  Conversion helpers
    ``time_to_batch_first`` and ``batch_to_time_first`` are provided for
    interop with legacy code that uses (T, B, ...) ordering.

Encoding strategies:
    RateEncoder        -- Bernoulli / Poisson / deterministic rate coding
    LatencyEncoder     -- Multi-spike latency coding (linear / exp / log)
    TTFSEncoder        -- Time-to-first-spike, exactly one spike per neuron
    PopulationEncoder  -- Gaussian tuning curves + rate-coded spike gen
    DeltaEncoder       -- Event-driven ON/OFF change detection

Design principles:
    1. Every encoder returns a SpikeBatch, never a raw tensor
    2. All stochastic sampling happens in fp32 then casts to compact dtype
    3. torch.Generator support for reproducible spike trains
    4. deterministic_eval mode: disable stochasticity at inference time
    5. AMP-safe: explicit float32 upcast around rand/comparison ops

Template for brain_ai/core/encoding.py

References:
    Gerstner & Kistler (2002) "Spiking Neuron Models"
    Auge et al. (2021) "A Survey of Encoding Techniques for SNNs"
    Kim et al. (2022) "Rate Coding or Direct Coding for SNNs?"
"""

from __future__ import annotations

import math
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1: Axis Convention Helpers
# ===========================================================================

#: Dimension index for the batch axis in batch-first layout (B, T, ...).
BATCH_DIM: int = 0

#: Dimension index for the time axis in batch-first layout (B, T, ...).
TIME_DIM: int = 1


def time_to_batch_first(x: Tensor) -> Tensor:
    """Permute a **(T, B, ...)** tensor to **(B, T, ...)** batch-first layout.

    This is the standard conversion when receiving spike trains from legacy
    code that uses time-first ordering (e.g., the original v1 encoders).

    Args:
        x: Tensor of shape ``(T, B, *spatial)`` with at least 2 dimensions.

    Returns:
        Tensor of shape ``(B, T, *spatial)`` sharing the same storage
        (no copy -- this is a view operation).

    Raises:
        ValueError: If ``x`` has fewer than 2 dimensions.
    """
    if x.dim() < 2:
        raise ValueError(
            f"time_to_batch_first requires >= 2 dims, got shape {x.shape}"
        )
    # Build permutation: (1, 0, 2, 3, ..., ndim-1)
    perm = [1, 0] + list(range(2, x.dim()))
    return x.permute(*perm)


def batch_to_time_first(x: Tensor) -> Tensor:
    """Permute a **(B, T, ...)** tensor to **(T, B, ...)** time-first layout.

    Use this when passing batch-first spike trains into modules that expect
    the legacy ``(T, B, ...)`` convention (e.g., older SNN unroll loops).

    Args:
        x: Tensor of shape ``(B, T, *spatial)`` with at least 2 dimensions.

    Returns:
        Tensor of shape ``(T, B, *spatial)`` sharing the same storage.

    Raises:
        ValueError: If ``x`` has fewer than 2 dimensions.
    """
    if x.dim() < 2:
        raise ValueError(
            f"batch_to_time_first requires >= 2 dims, got shape {x.shape}"
        )
    perm = [1, 0] + list(range(2, x.dim()))
    return x.permute(*perm)


def check_batch_first(x: Tensor, name: str = "tensor") -> None:
    """Assert that ``x`` has at least 2 dimensions (batch + time).

    This is a lightweight guard placed at encoder entry points to catch
    misuse early.  It does **not** verify that dim-0 is truly batch --
    that would require metadata we do not carry.

    Args:
        x: Input tensor to validate.
        name: Human-readable name for error messages.

    Raises:
        ValueError: If ``x.dim() < 2``.
    """
    if x.dim() < 2:
        raise ValueError(
            f"{name} must have >= 2 dimensions (B, ...), got shape {x.shape}. "
            f"If you have a 1-D input, unsqueeze a batch dimension first."
        )


# ===========================================================================
# SECTION 2: SpikeBatch Dataclass
# ===========================================================================

@dataclass
class SpikeBatch:
    """Structured container for spike trains produced by encoders.

    Every encoder in this module returns a ``SpikeBatch`` instead of a raw
    tensor.  This ensures downstream code can always access metadata (mask,
    auxiliary information) in a uniform way.

    Layout invariant:
        ``spikes`` is always **(B, T, *feature_shape)** -- batch-first.

    Fields:
        spikes: The spike tensor, dtype is typically ``torch.float32`` or
            ``torch.bool``.  Shape ``(B, T, *feature_shape)``.
        mask: Optional boolean mask of shape ``(B, T)`` where ``True``
            indicates a valid timestep.  Used for variable-length sequences.
            ``None`` means all timesteps are valid.
        aux: Auxiliary metadata dictionary.  Encoders may populate this with
            information such as firing rates, spike times, population maps,
            etc.  Consumers should treat unknown keys as opaque.

    Example::

        sb = SpikeBatch(spikes=torch.zeros(4, 25, 128), mask=None, aux={})
        assert sb.batch_size == 4
        assert sb.time_steps == 25
        assert sb.feature_shape == (128,)
    """

    spikes: Tensor
    mask: Optional[Tensor] = None
    aux: Dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def time_steps(self) -> int:
        """Number of time steps (dimension 1)."""
        return self.spikes.shape[TIME_DIM]

    @property
    def batch_size(self) -> int:
        """Batch size (dimension 0)."""
        return self.spikes.shape[BATCH_DIM]

    @property
    def feature_shape(self) -> Tuple[int, ...]:
        """Spatial / feature dimensions after (B, T).

        For a spike tensor of shape ``(B, T, N)`` this returns ``(N,)``.
        For ``(B, T, C, H, W)`` this returns ``(C, H, W)``.
        """
        return tuple(self.spikes.shape[2:])

    @property
    def device(self) -> torch.device:
        """Device of the spike tensor."""
        return self.spikes.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the spike tensor."""
        return self.spikes.dtype

    def is_binary(self, tolerance: float = 1e-6) -> bool:
        """Check whether all spike values are approximately 0 or 1.

        This is a diagnostic utility -- it scans the entire tensor, so avoid
        calling it in a hot loop.

        Args:
            tolerance: Maximum deviation from 0 or 1 to still count as binary.

        Returns:
            ``True`` if every element is within ``tolerance`` of 0 or 1.
        """
        s = self.spikes
        at_zero = (s.abs() <= tolerance)
        at_one = ((s - 1.0).abs() <= tolerance)
        return bool((at_zero | at_one).all())

    # -------------------------------------------------------------------
    # Movement / casting
    # -------------------------------------------------------------------

    def to(self, device: Union[str, torch.device], **kwargs: Any) -> "SpikeBatch":
        """Move all tensors to ``device``, returning a new SpikeBatch.

        Args:
            device: Target device (e.g., ``"cuda:0"`` or ``torch.device("cpu")``).
            **kwargs: Additional keyword arguments forwarded to ``Tensor.to()``.

        Returns:
            New ``SpikeBatch`` on the target device.
        """
        new_spikes = self.spikes.to(device, **kwargs)
        new_mask = self.mask.to(device, **kwargs) if self.mask is not None else None
        # Move any tensors in aux
        new_aux: Dict[str, Any] = {}
        for k, v in self.aux.items():
            if isinstance(v, Tensor):
                new_aux[k] = v.to(device, **kwargs)
            else:
                new_aux[k] = v
        return SpikeBatch(spikes=new_spikes, mask=new_mask, aux=new_aux)

    def float(self) -> "SpikeBatch":
        """Cast spikes to float32, returning a new SpikeBatch."""
        new_spikes = self.spikes.float()
        new_aux: Dict[str, Any] = {}
        for k, v in self.aux.items():
            if isinstance(v, Tensor) and v.is_floating_point():
                new_aux[k] = v.float()
            else:
                new_aux[k] = v
        return SpikeBatch(spikes=new_spikes, mask=self.mask, aux=new_aux)

    def bool(self) -> "SpikeBatch":
        """Cast spikes to boolean, returning a new SpikeBatch."""
        return SpikeBatch(
            spikes=self.spikes.bool(),
            mask=self.mask,
            aux=dict(self.aux),
        )

    def detach(self) -> "SpikeBatch":
        """Detach all tensors from the computation graph."""
        new_spikes = self.spikes.detach()
        new_mask = self.mask.detach() if self.mask is not None else None
        new_aux: Dict[str, Any] = {}
        for k, v in self.aux.items():
            if isinstance(v, Tensor):
                new_aux[k] = v.detach()
            else:
                new_aux[k] = v
        return SpikeBatch(spikes=new_spikes, mask=new_mask, aux=new_aux)

    # -------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------

    def __getitem__(self, idx: Union[int, slice, Tensor]) -> "SpikeBatch":
        """Index along the batch dimension.

        Args:
            idx: Integer, slice, or boolean/index tensor for batch selection.

        Returns:
            New ``SpikeBatch`` with the selected batch elements.

        Example::

            sb = SpikeBatch(spikes=torch.zeros(4, 25, 128))
            sub = sb[0:2]       # first two samples
            single = sb[0]      # single sample -- adds back batch dim
        """
        new_spikes = self.spikes[idx]
        # If a single int was used, the batch dim is squeezed -- add it back
        if isinstance(idx, int):
            new_spikes = new_spikes.unsqueeze(BATCH_DIM)

        new_mask = None
        if self.mask is not None:
            new_mask = self.mask[idx]
            if isinstance(idx, int):
                new_mask = new_mask.unsqueeze(BATCH_DIM)

        # Subset aux tensors that have a batch dimension
        new_aux: Dict[str, Any] = {}
        for k, v in self.aux.items():
            if isinstance(v, Tensor) and v.shape[0] == self.batch_size:
                selected = v[idx]
                if isinstance(idx, int):
                    selected = selected.unsqueeze(0)
                new_aux[k] = selected
            else:
                new_aux[k] = v

        return SpikeBatch(spikes=new_spikes, mask=new_mask, aux=new_aux)

    # -------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------

    @staticmethod
    def from_dense(
        spikes: Tensor,
        mask: Optional[Tensor] = None,
        **aux_kwargs: Any,
    ) -> "SpikeBatch":
        """Create a SpikeBatch from a dense spike tensor.

        This is the preferred factory when you already have a fully
        materialised spike tensor and want to wrap it with metadata.

        Args:
            spikes: Dense spike tensor of shape ``(B, T, ...)``.
            mask: Optional validity mask ``(B, T)``.
            **aux_kwargs: Additional auxiliary data stored in ``aux``.

        Returns:
            A new ``SpikeBatch`` instance.
        """
        if spikes.dim() < 2:
            raise ValueError(
                f"SpikeBatch.from_dense requires spikes with >= 2 dims "
                f"(B, T, ...), got shape {spikes.shape}"
            )
        if mask is not None:
            if mask.shape[:2] != spikes.shape[:2]:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match spikes "
                    f"batch/time dims {spikes.shape[:2]}"
                )
        return SpikeBatch(spikes=spikes, mask=mask, aux=dict(aux_kwargs))

    def __repr__(self) -> str:
        mask_str = f"mask={self.mask.shape}" if self.mask is not None else "mask=None"
        aux_keys = list(self.aux.keys())
        return (
            f"SpikeBatch(spikes={list(self.spikes.shape)}, "
            f"dtype={self.dtype}, device={self.device}, "
            f"{mask_str}, aux_keys={aux_keys})"
        )


# ===========================================================================
# SECTION 3: EncodingConfig Dataclass
# ===========================================================================

#: Valid encoding type strings.
ENCODING_TYPES: Tuple[str, ...] = (
    "rate_bernoulli",
    "rate_poisson",
    "rate_deterministic",
    "latency",
    "ttfs",
    "population",
    "delta",
)

#: Valid normalization methods.
NORMALIZATION_METHODS: Tuple[str, ...] = ("none", "minmax", "sigmoid", "clamp")

#: Valid latency mapping functions.
LATENCY_MAPPINGS: Tuple[str, ...] = ("linear", "exponential", "log")


@dataclass
class EncodingConfig:
    """Unified configuration for all spike encoders.

    A single config dataclass governs encoder construction.  Fields are
    grouped by relevance -- a ``RateEncoder`` ignores population fields,
    a ``PopulationEncoder`` ignores delta fields, etc.

    Example::

        cfg = EncodingConfig(type="rate_bernoulli", num_steps=30)
        encoder = create_encoder(cfg)
        sb = encoder(x)

    Attributes:
        type: Encoding strategy -- one of ``ENCODING_TYPES``.
        num_steps: Number of time steps in the output spike train.
        dt: Simulation time step (ms).  Affects Poisson rate scaling.
        normalization: How to normalise input before encoding.
        deterministic_eval: If ``True``, stochastic encoders become
            deterministic when ``self.training is False``.
        seed: Optional global seed for the torch.Generator.  ``None``
            means non-deterministic.
        rate_gain: Multiplicative gain on input before rate sampling.
        rate_bias: Additive bias on input before rate sampling.
        population_size: Number of neurons per input dimension in
            PopulationEncoder.
        population_sigma: Width of Gaussian tuning curves.
        latency_mapping: Function mapping input value to spike time.
        t_min: Earliest allowed spike time (clamp floor).
        t_max: Latest allowed spike time.  ``None`` defaults to
            ``num_steps - 1``.
        allow_no_spike: If ``True``, sub-threshold inputs produce no spike
            (TTFSEncoder).
        jitter: Standard deviation of Gaussian jitter added to spike
            times during training (TTFSEncoder).
        delta_threshold: Change threshold for DeltaEncoder ON/OFF spikes.
    """

    type: str = "rate_bernoulli"
    num_steps: int = 25
    dt: float = 1.0
    normalization: str = "minmax"
    deterministic_eval: bool = True
    seed: Optional[int] = None

    # -- Rate encoder params --
    rate_gain: float = 1.0
    rate_bias: float = 0.0

    # -- Population encoder params --
    population_size: int = 8
    population_sigma: float = 0.2

    # -- Latency / TTFS params --
    latency_mapping: str = "exponential"
    t_min: int = 0
    t_max: Optional[int] = None
    allow_no_spike: bool = False
    jitter: float = 0.0

    # -- Delta encoder params --
    delta_threshold: float = 0.1

    def __post_init__(self) -> None:
        """Validate config fields on construction."""
        if self.type not in ENCODING_TYPES:
            raise ValueError(
                f"Unknown encoding type '{self.type}'. "
                f"Valid types: {ENCODING_TYPES}"
            )
        if self.normalization not in NORMALIZATION_METHODS:
            raise ValueError(
                f"Unknown normalization '{self.normalization}'. "
                f"Valid methods: {NORMALIZATION_METHODS}"
            )
        if self.type in ("latency", "ttfs") and self.latency_mapping not in LATENCY_MAPPINGS:
            raise ValueError(
                f"Unknown latency_mapping '{self.latency_mapping}'. "
                f"Valid mappings: {LATENCY_MAPPINGS}"
            )
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        if self.population_size < 1:
            raise ValueError(f"population_size must be >= 1, got {self.population_size}")
        if self.jitter < 0.0:
            raise ValueError(f"jitter must be >= 0.0, got {self.jitter}")
        if self.delta_threshold <= 0.0:
            raise ValueError(f"delta_threshold must be > 0.0, got {self.delta_threshold}")
        if self.t_max is not None and self.t_max < self.t_min:
            raise ValueError(
                f"t_max ({self.t_max}) must be >= t_min ({self.t_min})"
            )


# ===========================================================================
# SECTION 4: Base Encoder
# ===========================================================================

class SpikeEncoder(nn.Module, ABC):
    """Abstract base class for all spike encoders.

    Subclasses must implement ``_encode_impl``.  The public ``encode`` method
    handles normalization, generator management, and wrapping the result
    in a ``SpikeBatch``.

    Attributes:
        config: The ``EncodingConfig`` governing this encoder.
    """

    def __init__(self, config: EncodingConfig) -> None:
        super().__init__()
        self.config = config

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def encode(
        self,
        x: Tensor,
        *,
        num_steps: Optional[int] = None,
        dt: Optional[float] = None,
        mode: Optional[str] = None,
        seed: Optional[int] = None,
        return_aux: bool = False,
    ) -> SpikeBatch:
        """Encode continuous input into a spike train.

        This is the primary entry point.  It normalises the input, resolves
        parameters, and delegates to ``_encode_impl``.

        Args:
            x: Continuous input tensor of shape ``(B, *features)``.
            num_steps: Override ``config.num_steps`` for this call.
            dt: Override ``config.dt`` for this call.
            mode: Override encoding mode (encoder-specific).
            seed: Override ``config.seed`` for this call.
            return_aux: If ``True``, populate ``SpikeBatch.aux`` with
                encoder-specific diagnostic information.

        Returns:
            A ``SpikeBatch`` with spikes in **(B, T, ...)** layout.
        """
        check_batch_first(x, name="encoder input")

        # Resolve overrides
        T = num_steps if num_steps is not None else self.config.num_steps
        dt_val = dt if dt is not None else self.config.dt
        eff_seed = seed if seed is not None else self.config.seed
        gen = self._get_generator(eff_seed)

        # Normalize
        x_norm = self._normalize(x, self.config.normalization)

        # Delegate
        return self._encode_impl(
            x_norm,
            num_steps=T,
            dt=dt_val,
            mode=mode,
            generator=gen,
            return_aux=return_aux,
        )

    def forward(self, x: Tensor, **kwargs: Any) -> SpikeBatch:
        """nn.Module forward -- delegates to ``encode``."""
        return self.encode(x, **kwargs)

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    @abstractmethod
    def _encode_impl(
        self,
        x: Tensor,
        *,
        num_steps: int,
        dt: float,
        mode: Optional[str],
        generator: Optional[torch.Generator],
        return_aux: bool,
    ) -> SpikeBatch:
        """Subclass-specific encoding logic.

        Args:
            x: Normalized input ``(B, *features)``.
            num_steps: Number of time steps.
            dt: Simulation time step.
            mode: Encoder-specific mode override.
            generator: Optional torch.Generator for reproducibility.
            return_aux: Whether to populate aux data.

        Returns:
            A ``SpikeBatch``.
        """
        ...

    def _normalize(self, x: Tensor, method: str) -> Tensor:
        """Normalize input to a standard range before encoding.

        All normalization is done in fp32 to avoid half-precision
        overflow issues with min/max operations.

        Args:
            x: Raw input tensor.
            method: One of ``NORMALIZATION_METHODS``.

        Returns:
            Normalised tensor in the same dtype as input (or fp32 if
            input was lower precision).
        """
        # Perform normalization in fp32 for numerical safety
        x_fp32 = x.float()

        if method == "none":
            return x_fp32

        elif method == "minmax":
            # Per-sample min-max across all non-batch dimensions
            # Flatten features for min/max computation
            flat = x_fp32.flatten(start_dim=1)  # (B, -1)
            x_min = flat.min(dim=1, keepdim=True).values  # (B, 1)
            x_max = flat.max(dim=1, keepdim=True).values  # (B, 1)
            # Reshape min/max for broadcast
            shape_ones = [1] * (x_fp32.dim() - 1)
            x_min = x_min.view(-1, *shape_ones)
            x_max = x_max.view(-1, *shape_ones)
            denom = (x_max - x_min).clamp(min=1e-8)
            return (x_fp32 - x_min) / denom

        elif method == "sigmoid":
            return torch.sigmoid(x_fp32)

        elif method == "clamp":
            return torch.clamp(x_fp32, 0.0, 1.0)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _get_generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        """Create a torch.Generator with the given seed.

        If ``seed`` is ``None``, returns ``None`` (non-deterministic).

        Args:
            seed: Integer seed.

        Returns:
            A ``torch.Generator`` seeded on CPU, or ``None``.
        """
        if seed is None:
            return None
        gen = torch.Generator()
        gen.manual_seed(seed)
        return gen


# ===========================================================================
# SECTION 5: RateEncoder
# ===========================================================================

class RateEncoder(SpikeEncoder):
    """Rate coding: spike probability proportional to input value.

    Three sampling modes:
        **bernoulli** -- Each timestep, draw ``U ~ Uniform(0,1)`` and spike
        if ``U < p`` where ``p = clamp(gain * x + bias, 0, 1)``.

        **poisson** -- Draw spike counts from a Poisson process with rate
        ``p * dt``.  For typical ``dt = 1.0`` this is equivalent to
        Bernoulli (Poisson with rate << 1 is approximately Bernoulli).

        **deterministic** -- Threshold-and-dither: accumulate ``p`` across
        timesteps, emit a spike whenever the accumulator crosses 1.0, then
        subtract 1.0.  Produces perfectly reproducible spike counts
        proportional to rate.

    AMP safety:
        Sampling (``torch.rand``) is always performed in fp32 on CPU
        Generator (if provided) or in the tensor's device fp32.  The
        comparison ``rand < p`` is done in fp32, then the resulting spikes
        are cast to ``torch.float32`` (or ``torch.bool`` if desired).

    deterministic_eval:
        When ``self.training is False`` and ``config.deterministic_eval``
        is ``True``, Bernoulli/Poisson modes fall back to deterministic
        mode automatically.  This removes stochastic variance at inference
        time without changing expected spike counts.

    Args:
        config: An ``EncodingConfig`` with ``type`` starting with ``"rate_"``.

    Example::

        cfg = EncodingConfig(type="rate_bernoulli", num_steps=30, rate_gain=1.2)
        enc = RateEncoder(cfg)
        sb = enc(torch.rand(8, 128))   # (8, 30, 128) SpikeBatch
    """

    def __init__(self, config: EncodingConfig) -> None:
        super().__init__(config)
        self.gain = config.rate_gain
        self.bias = config.rate_bias

    def _encode_impl(
        self,
        x: Tensor,
        *,
        num_steps: int,
        dt: float,
        mode: Optional[str],
        generator: Optional[torch.Generator],
        return_aux: bool,
    ) -> SpikeBatch:
        """Generate rate-coded spike train.

        Args:
            x: Normalized input ``(B, *features)`` in [0, 1] (after normalization).
            num_steps: Number of timesteps T.
            dt: Time step (used for Poisson rate scaling).
            mode: If provided, override the encoding subtype
                  (``"bernoulli"``, ``"poisson"``, ``"deterministic"``).
            generator: Optional torch.Generator.
            return_aux: Whether to include ``rates`` in aux.

        Returns:
            SpikeBatch with spikes ``(B, T, *features)``.
        """
        # Determine effective mode
        subtype = mode
        if subtype is None:
            # Extract from config.type: "rate_bernoulli" -> "bernoulli"
            subtype = self.config.type.replace("rate_", "")
        if subtype not in ("bernoulli", "poisson", "deterministic"):
            subtype = "bernoulli"

        # Apply gain and bias, clamp to valid probability range
        # Force fp32 for all arithmetic
        p = (self.gain * x.float() + self.bias).clamp(0.0, 1.0)  # (B, *features)

        B = p.shape[0]
        feature_shape = p.shape[1:]
        device = p.device

        # deterministic_eval override
        use_deterministic = (subtype == "deterministic")
        if not self.training and self.config.deterministic_eval:
            use_deterministic = True

        if use_deterministic:
            spikes = self._deterministic_rate(p, num_steps)
        elif subtype == "bernoulli":
            spikes = self._bernoulli_rate(p, num_steps, device, generator)
        elif subtype == "poisson":
            spikes = self._poisson_rate(p, num_steps, dt, device, generator)
        else:
            spikes = self._bernoulli_rate(p, num_steps, device, generator)

        # Build aux
        aux: Dict[str, Any] = {}
        if return_aux:
            aux["rates"] = p.detach()  # (B, *features)
            aux["encoding_type"] = subtype

        return SpikeBatch(spikes=spikes, mask=None, aux=aux)

    def _bernoulli_rate(
        self,
        p: Tensor,
        num_steps: int,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        """Bernoulli sampling: spike if U < p.

        All sampling in fp32 on the generator's device (CPU for
        reproducibility), then moved to target device.

        Returns:
            spikes: ``(B, T, *features)`` float32 tensor of 0s and 1s.
        """
        B = p.shape[0]
        feature_shape = p.shape[1:]

        # Expand p to (B, T, *features)
        p_expanded = p.unsqueeze(TIME_DIM).expand(B, num_steps, *feature_shape)

        # Sample uniform random numbers in fp32
        # Generator is always on CPU; we generate on CPU and move
        if generator is not None:
            rand_vals = torch.rand(
                B, num_steps, *feature_shape,
                dtype=torch.float32,
                device="cpu",
                generator=generator,
            ).to(device)
        else:
            rand_vals = torch.rand(
                B, num_steps, *feature_shape,
                dtype=torch.float32,
                device=device,
            )

        # Compare in fp32, result is float32 (0.0 or 1.0)
        spikes = (rand_vals < p_expanded.float()).float()
        return spikes

    def _poisson_rate(
        self,
        p: Tensor,
        num_steps: int,
        dt: float,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        """Poisson process sampling.

        For small rates (p * dt < 0.3), a Poisson process is well
        approximated by Bernoulli with probability ``p * dt``.  For
        higher rates we use the same approximation but clamp the effective
        rate to [0, 1] to keep it a valid probability.

        Returns:
            spikes: ``(B, T, *features)`` float32.
        """
        effective_rate = (p * dt).clamp(0.0, 1.0)
        return self._bernoulli_rate(effective_rate, num_steps, device, generator)

    def _deterministic_rate(
        self,
        p: Tensor,
        num_steps: int,
    ) -> Tensor:
        """Deterministic threshold-and-dither encoding.

        Accumulates the rate ``p`` at each timestep.  When the accumulator
        reaches 1.0, a spike is emitted and the accumulator is decremented
        by 1.0.  This produces exactly ``floor(p * T)`` or ``ceil(p * T)``
        spikes per neuron, evenly distributed across time.

        Returns:
            spikes: ``(B, T, *features)`` float32.
        """
        B = p.shape[0]
        feature_shape = p.shape[1:]
        device = p.device

        spike_list: List[Tensor] = []
        accumulator = torch.zeros_like(p)  # (B, *features)

        for _t in range(num_steps):
            accumulator = accumulator + p
            spike = (accumulator >= 1.0).float()
            accumulator = accumulator - spike  # subtract 1 only where spiked
            spike_list.append(spike)

        # Stack along time dim: list of (B, *features) -> (B, T, *features)
        spikes = torch.stack(spike_list, dim=TIME_DIM)
        return spikes


# ===========================================================================
# SECTION 6: LatencyEncoder
# ===========================================================================

class LatencyEncoder(SpikeEncoder):
    """Latency coding: input value determines *when* spikes occur.

    Higher-valued inputs spike earlier; lower-valued inputs spike later.
    Unlike ``TTFSEncoder``, this encoder can place **multiple** spikes per
    neuron (the primary spike at the computed time, plus optionally a
    burst of ``n_burst`` spikes at consecutive timesteps).

    Three time-mapping functions:
        **linear**: ``t = (1 - x) * (T - 1)``
        **exponential**: ``t = tau * log(1 / clamp(x, eps, 1))``
        **log**: ``t = T * (1 - log(1 + x) / log(2))``

    All time computations are performed in fp32 for AMP safety.

    Args:
        config: An ``EncodingConfig`` with ``type="latency"``.

    Example::

        cfg = EncodingConfig(type="latency", num_steps=25, latency_mapping="exponential")
        enc = LatencyEncoder(cfg)
        sb = enc(torch.rand(8, 128))
        print(sb.spikes.shape)   # (8, 25, 128)
        print(sb.aux["spike_times"].shape)  # (8, 128)
    """

    def __init__(self, config: EncodingConfig) -> None:
        super().__init__(config)
        self.mapping = config.latency_mapping
        self.t_min = config.t_min
        self.t_max = config.t_max

    def _encode_impl(
        self,
        x: Tensor,
        *,
        num_steps: int,
        dt: float,
        mode: Optional[str],
        generator: Optional[torch.Generator],
        return_aux: bool,
    ) -> SpikeBatch:
        """Compute spike times and build spike tensor.

        Args:
            x: Normalized input ``(B, *features)`` in [0, 1].
            num_steps: Number of timesteps T.
            dt: Not used (latency is a mapping, not a rate process).
            mode: If provided, override ``latency_mapping``.
            generator: Not used for latency encoding (deterministic).
            return_aux: Whether to include spike_times in aux.

        Returns:
            SpikeBatch with spikes ``(B, T, *features)``.
        """
        mapping = mode if mode in LATENCY_MAPPINGS else self.mapping
        t_max = self.t_max if self.t_max is not None else (num_steps - 1)

        # Compute spike times in fp32
        spike_times = self._compute_spike_times(
            x.float(), num_steps=num_steps, mapping=mapping
        )

        # Clamp to valid range
        spike_times = spike_times.clamp(float(self.t_min), float(t_max))

        # Round to nearest integer timestep
        spike_times_int = spike_times.round().long()  # (B, *features)

        # Build spike tensor
        spikes = self._times_to_spikes(spike_times_int, num_steps, x.device)

        # Build aux
        aux: Dict[str, Any] = {}
        if return_aux:
            aux["spike_times"] = spike_times.detach()  # (B, *features) float
            aux["spike_times_int"] = spike_times_int.detach()  # (B, *features) long

        return SpikeBatch(spikes=spikes, mask=None, aux=aux)

    def _compute_spike_times(
        self,
        x: Tensor,
        *,
        num_steps: int,
        mapping: str,
    ) -> Tensor:
        """Map input values to continuous spike times.

        Args:
            x: Input in [0, 1], shape ``(B, *features)``, fp32.
            num_steps: Total number of timesteps T.
            mapping: One of ``"linear"``, ``"exponential"``, ``"log"``.

        Returns:
            Continuous spike times ``(B, *features)`` in [0, T-1].
        """
        T = float(num_steps)
        eps = 1e-7

        if mapping == "linear":
            # Higher value -> earlier spike (lower time)
            times = (1.0 - x) * (T - 1.0)

        elif mapping == "exponential":
            # tau * log(1 / x) -- higher x -> lower time -> earlier spike
            tau = T / math.log(1.0 / eps)  # Scale so x=eps -> t=T
            x_safe = x.clamp(min=eps, max=1.0)
            times = tau * torch.log(1.0 / x_safe)

        elif mapping == "log":
            # T * (1 - log(1+x) / log(2)) -- x=1 -> t=0, x=0 -> t=T
            times = T * (1.0 - torch.log1p(x) / math.log(2.0))

        else:
            raise ValueError(f"Unknown latency mapping: {mapping}")

        return times

    def _times_to_spikes(
        self,
        spike_times_int: Tensor,
        num_steps: int,
        device: torch.device,
    ) -> Tensor:
        """Convert integer spike times to a dense spike tensor.

        Uses scatter to avoid Python loops over time steps.

        Args:
            spike_times_int: Integer spike times ``(B, *features)``.
            num_steps: Number of timesteps T.
            device: Target device.

        Returns:
            Spike tensor ``(B, T, *features)`` of float32.
        """
        B = spike_times_int.shape[0]
        feature_shape = spike_times_int.shape[1:]
        N = spike_times_int[0].numel()  # total feature elements

        # Flatten features for scatter: (B, N)
        times_flat = spike_times_int.reshape(B, N).clamp(0, num_steps - 1)

        # Create output: (B, T, N)
        spikes_flat = torch.zeros(B, num_steps, N, dtype=torch.float32, device=device)

        # Scatter: place 1.0 at the spike time for each neuron
        # scatter_ along dim=1 (time), indices (B, 1, N), values = 1
        idx = times_flat.unsqueeze(TIME_DIM)  # (B, 1, N)
        spikes_flat.scatter_(TIME_DIM, idx, 1.0)

        # Reshape back to (B, T, *features)
        spikes = spikes_flat.reshape(B, num_steps, *feature_shape)
        return spikes


# ===========================================================================
# SECTION 7: TTFSEncoder (Time-to-First-Spike)
# ===========================================================================

class TTFSEncoder(SpikeEncoder):
    """Time-to-first-spike encoding: exactly one spike per neuron per window.

    Similar to ``LatencyEncoder`` but enforces the strict one-spike
    constraint and supports two additional features:

    **allow_no_spike**: When ``True``, input values below a threshold
    produce no spike at all (spike_time = -1 in aux).  This is useful
    for sparse inputs where many features are zero.

    **jitter**: During training, Gaussian noise ``N(0, jitter)`` is added
    to continuous spike times before rounding, simulating biological
    timing variability and acting as a regulariser.

    Args:
        config: An ``EncodingConfig`` with ``type="ttfs"``.

    Example::

        cfg = EncodingConfig(type="ttfs", num_steps=25, jitter=0.5, allow_no_spike=True)
        enc = TTFSEncoder(cfg)
        sb = enc(torch.rand(8, 128), return_aux=True)
        print(sb.aux["spike_times"])  # (8, 128) with -1 for no-spike
    """

    def __init__(self, config: EncodingConfig) -> None:
        super().__init__(config)
        self.mapping = config.latency_mapping
        self.t_min = config.t_min
        self.t_max = config.t_max
        self.allow_no_spike = config.allow_no_spike
        self.jitter = config.jitter
        # Threshold below which input is considered "silent"
        self._no_spike_threshold = 0.01

    def _encode_impl(
        self,
        x: Tensor,
        *,
        num_steps: int,
        dt: float,
        mode: Optional[str],
        generator: Optional[torch.Generator],
        return_aux: bool,
    ) -> SpikeBatch:
        """Compute exactly one spike per neuron.

        Args:
            x: Normalized input ``(B, *features)`` in [0, 1].
            num_steps: Number of timesteps T.
            dt: Not used.
            mode: If provided, override ``latency_mapping``.
            generator: Optional generator for jitter reproducibility.
            return_aux: Whether to include spike_times in aux.

        Returns:
            SpikeBatch with spikes ``(B, T, *features)``.
        """
        mapping = mode if mode in LATENCY_MAPPINGS else self.mapping
        t_max = self.t_max if self.t_max is not None else (num_steps - 1)

        # Compute continuous spike times in fp32
        spike_times = self._compute_ttfs_times(
            x.float(), num_steps=num_steps, mapping=mapping
        )

        # Apply jitter during training
        if self.training and self.jitter > 0.0:
            if generator is not None:
                noise = torch.randn(
                    spike_times.shape,
                    dtype=torch.float32,
                    device="cpu",
                    generator=generator,
                ).to(spike_times.device) * self.jitter
            else:
                noise = torch.randn_like(spike_times) * self.jitter
            spike_times = spike_times + noise

        # Clamp to valid range
        spike_times = spike_times.clamp(float(self.t_min), float(t_max))

        # Determine no-spike mask (where input is below threshold)
        no_spike_mask: Optional[Tensor] = None
        if self.allow_no_spike:
            no_spike_mask = (x.float() < self._no_spike_threshold)  # (B, *features)

        # Round to integers
        spike_times_int = spike_times.round().long()

        # Build spike tensor
        spikes = self._ttfs_to_spikes(
            spike_times_int, num_steps, x.device, no_spike_mask
        )

        # Build aux
        aux: Dict[str, Any] = {}
        if return_aux:
            # Store times with -1 for no-spike neurons
            times_out = spike_times.detach().clone()
            if no_spike_mask is not None:
                times_out[no_spike_mask] = -1.0
            aux["spike_times"] = times_out

        return SpikeBatch(spikes=spikes, mask=None, aux=aux)

    def _compute_ttfs_times(
        self,
        x: Tensor,
        *,
        num_steps: int,
        mapping: str,
    ) -> Tensor:
        """Map input values to spike times (same mappings as LatencyEncoder).

        Args:
            x: Input in [0, 1], fp32.
            num_steps: Total timesteps.
            mapping: Mapping function name.

        Returns:
            Continuous spike times ``(B, *features)``.
        """
        T = float(num_steps)
        eps = 1e-7

        if mapping == "linear":
            times = (1.0 - x) * (T - 1.0)
        elif mapping == "exponential":
            tau = T / math.log(1.0 / eps)
            x_safe = x.clamp(min=eps, max=1.0)
            times = tau * torch.log(1.0 / x_safe)
        elif mapping == "log":
            times = T * (1.0 - torch.log1p(x) / math.log(2.0))
        else:
            raise ValueError(f"Unknown mapping: {mapping}")

        return times

    def _ttfs_to_spikes(
        self,
        spike_times_int: Tensor,
        num_steps: int,
        device: torch.device,
        no_spike_mask: Optional[Tensor],
    ) -> Tensor:
        """Build spike tensor with exactly one spike per neuron.

        Args:
            spike_times_int: Integer spike times ``(B, *features)``.
            num_steps: Number of timesteps T.
            device: Target device.
            no_spike_mask: Boolean mask ``(B, *features)`` where ``True``
                means no spike should be placed.

        Returns:
            Spike tensor ``(B, T, *features)`` of float32.
        """
        B = spike_times_int.shape[0]
        feature_shape = spike_times_int.shape[1:]
        N = spike_times_int[0].numel()

        times_flat = spike_times_int.reshape(B, N).clamp(0, num_steps - 1)

        spikes_flat = torch.zeros(B, num_steps, N, dtype=torch.float32, device=device)
        idx = times_flat.unsqueeze(TIME_DIM)  # (B, 1, N)
        spikes_flat.scatter_(TIME_DIM, idx, 1.0)

        # Zero out no-spike neurons
        if no_spike_mask is not None:
            mask_flat = no_spike_mask.reshape(B, N).unsqueeze(TIME_DIM)  # (B, 1, N)
            mask_flat = mask_flat.expand_as(spikes_flat)
            spikes_flat = spikes_flat.masked_fill(mask_flat, 0.0)

        spikes = spikes_flat.reshape(B, num_steps, *feature_shape)
        return spikes


# ===========================================================================
# SECTION 8: PopulationEncoder
# ===========================================================================

class PopulationEncoder(SpikeEncoder):
    """Population coding: each input dimension is represented by a group
    of neurons with overlapping Gaussian receptive fields.

    For an input of shape ``(B, N)``, the output is ``(B, T, N * P)``
    where ``P = population_size``.  Each group of ``P`` neurons covers
    one input dimension with tuning curve centres evenly spaced across
    [0, 1].

    The activation of the ``j``-th neuron for input value ``x_i`` is:

        ``a_{i,j} = exp(-(x_i - c_j)^2 / (2 * sigma^2))``

    These activations are then rate-encoded into spike trains using
    Bernoulli sampling (or deterministic mode at eval time).

    The tuning curve centres can optionally be made learnable
    (``nn.Parameter`` instead of ``register_buffer``).

    Args:
        config: An ``EncodingConfig`` with ``type="population"``.

    Example::

        cfg = EncodingConfig(type="population", num_steps=25, population_size=10)
        enc = PopulationEncoder(cfg)
        sb = enc(torch.rand(8, 64))
        print(sb.spikes.shape)   # (8, 25, 640)
    """

    def __init__(
        self,
        config: EncodingConfig,
        learnable_centers: bool = False,
    ) -> None:
        super().__init__(config)
        self.population_size = config.population_size
        self.sigma = config.population_sigma

        # Tuning curve centres
        centers = torch.linspace(0.0, 1.0, config.population_size)  # (P,)
        if learnable_centers:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)

        # Internal rate encoder for converting activations to spikes
        rate_config = EncodingConfig(
            type="rate_bernoulli",
            num_steps=config.num_steps,
            dt=config.dt,
            normalization="none",  # activations are already in [0, 1]
            deterministic_eval=config.deterministic_eval,
            seed=config.seed,
            rate_gain=1.0,
            rate_bias=0.0,
        )
        self._rate_encoder = RateEncoder(rate_config)

    def _encode_impl(
        self,
        x: Tensor,
        *,
        num_steps: int,
        dt: float,
        mode: Optional[str],
        generator: Optional[torch.Generator],
        return_aux: bool,
    ) -> SpikeBatch:
        """Compute tuning curve activations and generate spikes.

        Args:
            x: Normalized input ``(B, N)`` in [0, 1].
            num_steps: Number of timesteps T.
            dt: Time step for rate sampling.
            mode: Passed to internal RateEncoder.
            generator: Optional generator for reproducibility.
            return_aux: Whether to include population_map and activations.

        Returns:
            SpikeBatch with spikes ``(B, T, N*P)``.
        """
        # x: (B, *features) -- flatten features for population expansion
        B = x.shape[0]
        feature_shape = x.shape[1:]
        x_flat = x.reshape(B, -1).float()  # (B, N)
        N = x_flat.shape[1]
        P = self.population_size

        # Compute Gaussian tuning curve activations
        # x_flat: (B, N, 1), centers: (P,) -> broadcast to (B, N, P)
        x_expanded = x_flat.unsqueeze(-1)           # (B, N, 1)
        centers = self.centers.unsqueeze(0).unsqueeze(0)  # (1, 1, P)
        diff = x_expanded - centers                  # (B, N, P)
        activations = torch.exp(-(diff ** 2) / (2.0 * self.sigma ** 2))  # (B, N, P)

        # Flatten to (B, N*P)
        activations_flat = activations.reshape(B, N * P)

        # Rate-encode the activations into spikes
        # Sync training mode with the internal rate encoder
        self._rate_encoder.train(self.training)

        # Use the internal rate encoder, passing generator if available
        if generator is not None:
            sb = self._rate_encoder._encode_impl(
                activations_flat,
                num_steps=num_steps,
                dt=dt,
                mode=mode,
                generator=generator,
                return_aux=False,
            )
        else:
            sb = self._rate_encoder.encode(
                activations_flat,
                num_steps=num_steps,
                dt=dt,
                mode=mode,
            )

        # Build population map aux
        aux: Dict[str, Any] = {}
        if return_aux:
            # Map: group_idx -> list of neuron indices in the flattened output
            population_map: Dict[int, List[int]] = {}
            for i in range(N):
                population_map[i] = list(range(i * P, (i + 1) * P))
            aux["population_map"] = population_map
            aux["activations"] = activations.detach()  # (B, N, P)
            aux["centers"] = self.centers.detach().clone()

        return SpikeBatch(spikes=sb.spikes, mask=None, aux=aux)


# ===========================================================================
# SECTION 9: DeltaEncoder
# ===========================================================================

class DeltaEncoder(SpikeEncoder):
    """Event-driven change detection: spikes when input changes significantly.

    Mimics dynamic vision sensors (DVS) and biological retinal ganglion cells.
    For each input dimension, two output channels are produced:

    - **ON channel**: fires when input *increases* by more than ``threshold``
    - **OFF channel**: fires when input *decreases* by more than ``threshold``

    The output shape is ``(B, T, N*2)`` where the first ``N`` elements are
    ON channels and the last ``N`` are OFF channels.

    Statefulness:
        The encoder maintains a ``prev_input`` buffer via ``register_buffer``.
        Call ``reset()`` between unrelated sequences to clear state.  If the
        encoder has never seen input (prev_input is all zeros), the first
        forward pass computes deltas against zero.

    For temporal sequences where the input ``x`` already has a time dimension
    ``(B, T_in, N)``, the encoder processes each timestep sequentially and
    concatenates the results.  For static inputs ``(B, N)`` it repeats the
    delta pattern across ``num_steps`` timesteps.

    Args:
        config: An ``EncodingConfig`` with ``type="delta"``.

    Example::

        cfg = EncodingConfig(type="delta", num_steps=10, delta_threshold=0.1)
        enc = DeltaEncoder(cfg)
        # Frame 1
        sb1 = enc(torch.rand(4, 128))   # (4, 10, 256)
        # Frame 2 -- produces spikes where input changed
        sb2 = enc(torch.rand(4, 128))
        enc.reset()  # clear state for next sequence
    """

    def __init__(self, config: EncodingConfig) -> None:
        super().__init__(config)
        self.threshold = config.delta_threshold
        # Off threshold is the same magnitude (symmetric by default)
        self.off_threshold = config.delta_threshold

        # Persistent state buffer -- will be lazily initialised on first forward
        self._prev_input_initialized = False
        self.register_buffer("prev_input", torch.tensor(0.0))

    def reset(self) -> None:
        """Clear the stored previous input.

        Call this between unrelated sequences to prevent spurious
        delta spikes at the boundary.
        """
        self._prev_input_initialized = False
        self.prev_input = torch.tensor(0.0, device=self.prev_input.device)

    def _encode_impl(
        self,
        x: Tensor,
        *,
        num_steps: int,
        dt: float,
        mode: Optional[str],
        generator: Optional[torch.Generator],
        return_aux: bool,
    ) -> SpikeBatch:
        """Compute ON/OFF delta spikes.

        Args:
            x: Normalized input ``(B, *features)`` in [0, 1].
            num_steps: Number of timesteps to repeat the delta pattern.
            dt: Not used.
            mode: Not used.
            generator: Not used (delta encoding is deterministic).
            return_aux: Whether to include delta values in aux.

        Returns:
            SpikeBatch with spikes ``(B, T, N*2)`` (ON then OFF channels).
        """
        B = x.shape[0]
        feature_shape = x.shape[1:]
        x_flat = x.reshape(B, -1).float()  # (B, N)
        N = x_flat.shape[1]
        device = x_flat.device

        # Initialise prev_input if needed
        if not self._prev_input_initialized:
            self.prev_input = torch.zeros(
                B, N, dtype=torch.float32, device=device
            )
            self._prev_input_initialized = True
        else:
            # Handle batch size changes gracefully
            if self.prev_input.shape[0] != B or self.prev_input.shape[-1] != N:
                self.prev_input = torch.zeros(
                    B, N, dtype=torch.float32, device=device
                )

        # Compute delta
        delta = x_flat - self.prev_input.to(device)  # (B, N)

        # ON spikes: positive change exceeds threshold
        on_spikes = (delta > self.threshold).float()   # (B, N)

        # OFF spikes: negative change exceeds threshold
        off_spikes = (-delta > self.off_threshold).float()  # (B, N)

        # Concatenate ON and OFF channels: (B, N*2)
        delta_spikes = torch.cat([on_spikes, off_spikes], dim=-1)  # (B, N*2)

        # Expand across timesteps: (B, T, N*2)
        # The delta event persists for all timesteps in this window
        spikes = delta_spikes.unsqueeze(TIME_DIM).expand(
            B, num_steps, 2 * N
        ).clone()  # clone to own memory (expand returns a view)

        # Update state (detached -- no gradient through state)
        self.prev_input = x_flat.detach().clone()

        # Build aux
        aux: Dict[str, Any] = {}
        if return_aux:
            aux["delta"] = delta.detach()
            aux["on_count"] = on_spikes.sum(dim=-1).detach()   # (B,)
            aux["off_count"] = off_spikes.sum(dim=-1).detach()  # (B,)

        return SpikeBatch(spikes=spikes, mask=None, aux=aux)


# ===========================================================================
# SECTION 10: SpikeDecoder (utility for decoding spike trains)
# ===========================================================================

class SpikeDecoder(nn.Module):
    """Decode spike trains back to continuous values.

    Operates on SpikeBatch inputs (batch-first convention).

    Methods:
        **rate**: Mean firing rate across time: ``sum(spikes, dim=T) / T``.
        **first_spike**: Inverse latency -- earlier first spike = higher value.
        **membrane**: Use externally provided membrane potential.

    Args:
        method: Decoding strategy.

    Example::

        decoder = SpikeDecoder(method="rate")
        continuous = decoder(spike_batch)  # (B, *features)
    """

    def __init__(
        self,
        method: Literal["rate", "first_spike", "membrane"] = "rate",
    ) -> None:
        super().__init__()
        self.method = method

    def forward(
        self,
        spike_input: Union[SpikeBatch, Tensor],
        membrane: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode spike train to continuous output.

        Args:
            spike_input: A SpikeBatch or raw tensor ``(B, T, *features)``.
            membrane: Final membrane potential for ``"membrane"`` method.

        Returns:
            Decoded continuous tensor ``(B, *features)``.
        """
        if isinstance(spike_input, SpikeBatch):
            spikes = spike_input.spikes
        else:
            spikes = spike_input

        if spikes.dim() < 2:
            return spikes

        if self.method == "rate":
            # Mean firing rate across time dimension
            return spikes.mean(dim=TIME_DIM)

        elif self.method == "first_spike":
            # Time of first spike (inverse latency)
            # argmax along time returns the first 1.0 position
            first_spike_time = torch.argmax(spikes, dim=TIME_DIM).float()
            max_time = float(spikes.shape[TIME_DIM])
            # Invert: earlier spike = higher value
            return 1.0 - (first_spike_time / max_time)

        elif self.method == "membrane":
            if membrane is not None:
                return membrane
            # Fallback to rate decoding
            return spikes.mean(dim=TIME_DIM)

        else:
            raise ValueError(f"Unknown decode method: {self.method}")


# ===========================================================================
# SECTION 11: Factory Function
# ===========================================================================

#: Registry mapping encoding type strings to encoder classes.
_ENCODER_REGISTRY: Dict[str, Type[SpikeEncoder]] = {}


def _register_encoder(type_key: str, cls: Type[SpikeEncoder]) -> None:
    """Register an encoder class for a given type key."""
    _ENCODER_REGISTRY[type_key] = cls


# Populate registry
_register_encoder("rate_bernoulli", RateEncoder)
_register_encoder("rate_poisson", RateEncoder)
_register_encoder("rate_deterministic", RateEncoder)
_register_encoder("latency", LatencyEncoder)
_register_encoder("ttfs", TTFSEncoder)
_register_encoder("population", PopulationEncoder)
_register_encoder("delta", DeltaEncoder)


def create_encoder(config: EncodingConfig) -> SpikeEncoder:
    """Factory: create a spike encoder from an EncodingConfig.

    This is the preferred entry point for constructing encoders.  It
    validates the config type and dispatches to the correct class.

    Args:
        config: Encoding configuration.  ``config.type`` must be one
            of ``ENCODING_TYPES``.

    Returns:
        An instantiated ``SpikeEncoder`` subclass.

    Raises:
        ValueError: If ``config.type`` is not recognised.

    Example::

        cfg = EncodingConfig(type="rate_bernoulli", num_steps=30)
        encoder = create_encoder(cfg)
        sb = encoder(torch.rand(8, 128))
    """
    cls = _ENCODER_REGISTRY.get(config.type)
    if cls is None:
        raise ValueError(
            f"Unknown encoding type '{config.type}'. "
            f"Registered types: {list(_ENCODER_REGISTRY.keys())}"
        )
    return cls(config)


# ===========================================================================
# Convenience aliases for backward compatibility
# ===========================================================================

def create_rate_encoder(
    num_steps: int = 25,
    method: str = "bernoulli",
    gain: float = 1.0,
    bias: float = 0.0,
    deterministic_eval: bool = True,
) -> RateEncoder:
    """Convenience: create a RateEncoder with common defaults.

    Args:
        num_steps: Number of time steps.
        method: One of ``"bernoulli"``, ``"poisson"``, ``"deterministic"``.
        gain: Rate gain.
        bias: Rate bias.
        deterministic_eval: Use deterministic mode at eval.

    Returns:
        A configured ``RateEncoder``.
    """
    cfg = EncodingConfig(
        type=f"rate_{method}",
        num_steps=num_steps,
        rate_gain=gain,
        rate_bias=bias,
        deterministic_eval=deterministic_eval,
    )
    return RateEncoder(cfg)


def create_latency_encoder(
    num_steps: int = 25,
    mapping: str = "exponential",
) -> LatencyEncoder:
    """Convenience: create a LatencyEncoder with common defaults."""
    cfg = EncodingConfig(
        type="latency",
        num_steps=num_steps,
        latency_mapping=mapping,
    )
    return LatencyEncoder(cfg)


def create_ttfs_encoder(
    num_steps: int = 25,
    mapping: str = "exponential",
    jitter: float = 0.0,
    allow_no_spike: bool = False,
) -> TTFSEncoder:
    """Convenience: create a TTFSEncoder with common defaults."""
    cfg = EncodingConfig(
        type="ttfs",
        num_steps=num_steps,
        latency_mapping=mapping,
        jitter=jitter,
        allow_no_spike=allow_no_spike,
    )
    return TTFSEncoder(cfg)


def create_population_encoder(
    num_steps: int = 25,
    population_size: int = 8,
    sigma: float = 0.2,
) -> PopulationEncoder:
    """Convenience: create a PopulationEncoder with common defaults."""
    cfg = EncodingConfig(
        type="population",
        num_steps=num_steps,
        population_size=population_size,
        population_sigma=sigma,
    )
    return PopulationEncoder(cfg)


def create_delta_encoder(
    num_steps: int = 10,
    threshold: float = 0.1,
) -> DeltaEncoder:
    """Convenience: create a DeltaEncoder with common defaults."""
    cfg = EncodingConfig(
        type="delta",
        num_steps=num_steps,
        delta_threshold=threshold,
    )
    return DeltaEncoder(cfg)


# ===========================================================================
# SECTION 12: Comprehensive Self-Test
# ===========================================================================

def _run_self_test() -> None:
    """Run all encoding self-tests.

    This function exercises every encoder, the SpikeBatch dataclass,
    axis helpers, AMP compatibility, deterministic_eval, and generator
    reproducibility.  Each test group prints PASS or FAIL.
    """

    # Path setup: find brain_ai root
    # __file__ is .../brain_ai/core/encoding.py
    # 4 levels up: encoding.py -> core -> brain_ai -> <root>
    _this_file = os.path.abspath(__file__)
    _brain_ai_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(_this_file)
            )
        )
    )
    if _brain_ai_root not in sys.path:
        sys.path.insert(0, _brain_ai_root)
        print(f"[self-test] Added to sys.path: {_brain_ai_root}")

    results: Dict[str, str] = {}
    total_pass = 0
    total_fail = 0

    def _record(name: str, passed: bool, detail: str = "") -> None:
        nonlocal total_pass, total_fail
        if passed:
            results[name] = "PASS"
            total_pass += 1
            print(f"  [PASS] {name}")
        else:
            results[name] = f"FAIL: {detail}"
            total_fail += 1
            print(f"  [FAIL] {name}: {detail}")

    device = torch.device("cpu")
    B, N, T = 4, 64, 25

    print("=" * 70)
    print("Spike Encoding Self-Test Suite")
    print("=" * 70)

    # ------------------------------------------------------------------
    # TEST GROUP 1: Axis Helpers
    # ------------------------------------------------------------------
    print("\n--- Test Group 1: Axis Convention Helpers ---")
    try:
        x_tb = torch.randn(T, B, N)
        x_bt = time_to_batch_first(x_tb)
        assert x_bt.shape == (B, T, N), f"Expected (B,T,N), got {x_bt.shape}"
        _record("time_to_batch_first shape", True)
    except Exception as e:
        _record("time_to_batch_first shape", False, str(e))

    try:
        x_bt = torch.randn(B, T, N)
        x_tb = batch_to_time_first(x_bt)
        assert x_tb.shape == (T, B, N), f"Expected (T,B,N), got {x_tb.shape}"
        _record("batch_to_time_first shape", True)
    except Exception as e:
        _record("batch_to_time_first shape", False, str(e))

    try:
        # Roundtrip: (T,B,N) -> (B,T,N) -> (T,B,N)
        x_original = torch.randn(T, B, N)
        x_roundtrip = batch_to_time_first(time_to_batch_first(x_original))
        assert torch.allclose(x_original, x_roundtrip), "Roundtrip failed"
        _record("axis roundtrip", True)
    except Exception as e:
        _record("axis roundtrip", False, str(e))

    try:
        # 4D roundtrip: (T, B, C, H)
        x_4d = torch.randn(T, B, 3, 8)
        x_4d_bt = time_to_batch_first(x_4d)
        assert x_4d_bt.shape == (B, T, 3, 8), f"4D shape wrong: {x_4d_bt.shape}"
        x_4d_back = batch_to_time_first(x_4d_bt)
        assert torch.allclose(x_4d, x_4d_back), "4D roundtrip failed"
        _record("axis 4D roundtrip", True)
    except Exception as e:
        _record("axis 4D roundtrip", False, str(e))

    try:
        check_batch_first(torch.randn(B, N), "test_2d")
        _record("check_batch_first valid 2D", True)
    except Exception as e:
        _record("check_batch_first valid 2D", False, str(e))

    try:
        failed = False
        try:
            check_batch_first(torch.randn(10), "test_1d")
            failed = True  # Should have raised
        except ValueError:
            pass
        if failed:
            _record("check_batch_first rejects 1D", False, "Did not raise ValueError")
        else:
            _record("check_batch_first rejects 1D", True)
    except Exception as e:
        _record("check_batch_first rejects 1D", False, str(e))

    try:
        assert BATCH_DIM == 0, f"BATCH_DIM should be 0, got {BATCH_DIM}"
        assert TIME_DIM == 1, f"TIME_DIM should be 1, got {TIME_DIM}"
        _record("axis constants", True)
    except Exception as e:
        _record("axis constants", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 2: SpikeBatch
    # ------------------------------------------------------------------
    print("\n--- Test Group 2: SpikeBatch Dataclass ---")
    try:
        spk = torch.zeros(B, T, N)
        sb = SpikeBatch(spikes=spk, mask=None, aux={})
        assert sb.batch_size == B, f"batch_size: expected {B}, got {sb.batch_size}"
        assert sb.time_steps == T, f"time_steps: expected {T}, got {sb.time_steps}"
        assert sb.feature_shape == (N,), f"feature_shape: expected ({N},), got {sb.feature_shape}"
        _record("SpikeBatch properties", True)
    except Exception as e:
        _record("SpikeBatch properties", False, str(e))

    try:
        binary_sb = SpikeBatch(spikes=torch.tensor([[[0.0, 1.0, 0.0, 1.0]]]))
        assert binary_sb.is_binary(), "Should be binary"
        non_binary_sb = SpikeBatch(spikes=torch.tensor([[[0.5, 0.3]]]))
        assert not non_binary_sb.is_binary(), "Should not be binary"
        _record("SpikeBatch is_binary", True)
    except Exception as e:
        _record("SpikeBatch is_binary", False, str(e))

    try:
        spk = torch.ones(B, T, N)
        sb = SpikeBatch(spikes=spk)
        assert sb.device == torch.device("cpu")
        assert sb.dtype == torch.float32
        _record("SpikeBatch device/dtype", True)
    except Exception as e:
        _record("SpikeBatch device/dtype", False, str(e))

    try:
        spk = torch.randn(B, T, N)
        sb = SpikeBatch(spikes=spk, aux={"rates": torch.randn(B, N)})
        sb_float = sb.float()
        assert sb_float.spikes.dtype == torch.float32
        _record("SpikeBatch float()", True)
    except Exception as e:
        _record("SpikeBatch float()", False, str(e))

    try:
        spk = torch.ones(B, T, N)
        sb = SpikeBatch(spikes=spk)
        sb_bool = sb.bool()
        assert sb_bool.spikes.dtype == torch.bool
        _record("SpikeBatch bool()", True)
    except Exception as e:
        _record("SpikeBatch bool()", False, str(e))

    try:
        spk = torch.randn(B, T, N, requires_grad=True)
        sb = SpikeBatch(spikes=spk)
        sb_det = sb.detach()
        assert not sb_det.spikes.requires_grad
        _record("SpikeBatch detach()", True)
    except Exception as e:
        _record("SpikeBatch detach()", False, str(e))

    try:
        spk = torch.randn(B, T, N)
        mask = torch.ones(B, T, dtype=torch.bool)
        sb = SpikeBatch(spikes=spk, mask=mask, aux={"val": torch.randn(B, 5)})
        sb_moved = sb.to("cpu")
        assert sb_moved.spikes.device == torch.device("cpu")
        assert sb_moved.mask is not None
        assert sb_moved.mask.device == torch.device("cpu")
        _record("SpikeBatch to(device)", True)
    except Exception as e:
        _record("SpikeBatch to(device)", False, str(e))

    try:
        spk = torch.randn(B, T, N)
        aux_tensor = torch.randn(B, 10)
        sb = SpikeBatch(spikes=spk, aux={"t": aux_tensor, "s": "metadata"})
        # Integer indexing
        sb0 = sb[0]
        assert sb0.batch_size == 1, f"Expected batch_size 1, got {sb0.batch_size}"
        assert sb0.spikes.shape == (1, T, N)
        assert sb0.aux["t"].shape == (1, 10)
        assert sb0.aux["s"] == "metadata"
        # Slice indexing
        sb_slice = sb[0:2]
        assert sb_slice.batch_size == 2
        _record("SpikeBatch __getitem__", True)
    except Exception as e:
        _record("SpikeBatch __getitem__", False, str(e))

    try:
        spk = torch.randn(B, T, N)
        sb = SpikeBatch.from_dense(spk, rates=torch.randn(B, N))
        assert sb.batch_size == B
        assert "rates" in sb.aux
        _record("SpikeBatch from_dense", True)
    except Exception as e:
        _record("SpikeBatch from_dense", False, str(e))

    try:
        sb = SpikeBatch(spikes=torch.zeros(2, 10, 8))
        r = repr(sb)
        assert "SpikeBatch" in r
        _record("SpikeBatch repr", True)
    except Exception as e:
        _record("SpikeBatch repr", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 3: EncodingConfig
    # ------------------------------------------------------------------
    print("\n--- Test Group 3: EncodingConfig ---")
    try:
        cfg = EncodingConfig(type="rate_bernoulli", num_steps=30)
        assert cfg.num_steps == 30
        assert cfg.normalization == "minmax"
        _record("EncodingConfig defaults", True)
    except Exception as e:
        _record("EncodingConfig defaults", False, str(e))

    try:
        raised = False
        try:
            EncodingConfig(type="invalid_type")
        except ValueError:
            raised = True
        assert raised, "Should reject invalid type"
        _record("EncodingConfig rejects invalid type", True)
    except Exception as e:
        _record("EncodingConfig rejects invalid type", False, str(e))

    try:
        raised = False
        try:
            EncodingConfig(type="rate_bernoulli", normalization="invalid")
        except ValueError:
            raised = True
        assert raised, "Should reject invalid normalization"
        _record("EncodingConfig rejects invalid normalization", True)
    except Exception as e:
        _record("EncodingConfig rejects invalid normalization", False, str(e))

    try:
        raised = False
        try:
            EncodingConfig(type="rate_bernoulli", num_steps=0)
        except ValueError:
            raised = True
        assert raised, "Should reject num_steps=0"
        _record("EncodingConfig rejects num_steps=0", True)
    except Exception as e:
        _record("EncodingConfig rejects num_steps=0", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 4: RateEncoder
    # ------------------------------------------------------------------
    print("\n--- Test Group 4: RateEncoder ---")
    x = torch.rand(B, N)

    # Bernoulli mode
    try:
        cfg = EncodingConfig(type="rate_bernoulli", num_steps=T, normalization="clamp")
        enc = RateEncoder(cfg)
        sb = enc(x)
        assert isinstance(sb, SpikeBatch), "Must return SpikeBatch"
        assert sb.spikes.shape == (B, T, N), f"Shape: {sb.spikes.shape}"
        assert sb.is_binary(), "Spikes should be binary"
        _record("RateEncoder bernoulli shape/type", True)
    except Exception as e:
        _record("RateEncoder bernoulli shape/type", False, str(e))

    # Poisson mode
    try:
        cfg = EncodingConfig(type="rate_poisson", num_steps=T, normalization="clamp")
        enc = RateEncoder(cfg)
        sb = enc(x)
        assert sb.spikes.shape == (B, T, N)
        assert sb.is_binary()
        _record("RateEncoder poisson shape/type", True)
    except Exception as e:
        _record("RateEncoder poisson shape/type", False, str(e))

    # Deterministic mode
    try:
        cfg = EncodingConfig(
            type="rate_deterministic", num_steps=T, normalization="clamp"
        )
        enc = RateEncoder(cfg)
        sb = enc(x)
        assert sb.spikes.shape == (B, T, N)
        assert sb.is_binary()
        # Deterministic should produce same output twice
        sb2 = enc(x)
        assert torch.allclose(sb.spikes, sb2.spikes), "Deterministic not reproducible"
        _record("RateEncoder deterministic", True)
    except Exception as e:
        _record("RateEncoder deterministic", False, str(e))

    # return_aux
    try:
        cfg = EncodingConfig(type="rate_bernoulli", num_steps=T, normalization="clamp")
        enc = RateEncoder(cfg)
        sb = enc(x, return_aux=True)
        assert "rates" in sb.aux
        assert sb.aux["rates"].shape == (B, N)
        _record("RateEncoder return_aux", True)
    except Exception as e:
        _record("RateEncoder return_aux", False, str(e))

    # deterministic_eval mode
    try:
        cfg = EncodingConfig(
            type="rate_bernoulli", num_steps=T,
            normalization="clamp", deterministic_eval=True,
        )
        enc = RateEncoder(cfg)
        enc.eval()
        sb1 = enc(x)
        sb2 = enc(x)
        assert torch.allclose(sb1.spikes, sb2.spikes), \
            "deterministic_eval: eval mode should be reproducible"
        _record("RateEncoder deterministic_eval", True)
    except Exception as e:
        _record("RateEncoder deterministic_eval", False, str(e))

    # Gain and bias
    try:
        cfg = EncodingConfig(
            type="rate_bernoulli", num_steps=T,
            normalization="none", rate_gain=0.0, rate_bias=0.0,
        )
        enc = RateEncoder(cfg)
        sb = enc(torch.ones(B, N))
        # gain=0, bias=0 -> p=0 -> no spikes
        assert sb.spikes.sum().item() == 0.0, "gain=0, bias=0 should produce no spikes"
        _record("RateEncoder gain/bias zero", True)
    except Exception as e:
        _record("RateEncoder gain/bias zero", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 5: LatencyEncoder
    # ------------------------------------------------------------------
    print("\n--- Test Group 5: LatencyEncoder ---")
    x = torch.rand(B, N)

    for mapping in ("linear", "exponential", "log"):
        try:
            cfg = EncodingConfig(
                type="latency", num_steps=T,
                latency_mapping=mapping, normalization="clamp",
            )
            enc = LatencyEncoder(cfg)
            sb = enc(x, return_aux=True)
            assert sb.spikes.shape == (B, T, N), f"Shape: {sb.spikes.shape}"
            assert sb.is_binary()
            # Each neuron should have exactly 1 spike
            spike_counts = sb.spikes.sum(dim=TIME_DIM)  # (B, N)
            assert (spike_counts == 1.0).all(), \
                f"Not all neurons have exactly 1 spike (mapping={mapping})"
            assert "spike_times" in sb.aux
            _record(f"LatencyEncoder {mapping}", True)
        except Exception as e:
            _record(f"LatencyEncoder {mapping}", False, str(e))

    # Higher input -> earlier spike
    try:
        cfg = EncodingConfig(
            type="latency", num_steps=50,
            latency_mapping="linear", normalization="none",
        )
        enc = LatencyEncoder(cfg)
        x_high = torch.tensor([[0.9]])
        x_low = torch.tensor([[0.1]])
        sb_high = enc(x_high, return_aux=True)
        sb_low = enc(x_low, return_aux=True)
        time_high = sb_high.aux["spike_times"].item()
        time_low = sb_low.aux["spike_times"].item()
        assert time_high < time_low, \
            f"Higher input should spike earlier: {time_high} vs {time_low}"
        _record("LatencyEncoder ordering", True)
    except Exception as e:
        _record("LatencyEncoder ordering", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 6: TTFSEncoder
    # ------------------------------------------------------------------
    print("\n--- Test Group 6: TTFSEncoder ---")
    x = torch.rand(B, N)

    try:
        cfg = EncodingConfig(
            type="ttfs", num_steps=T,
            latency_mapping="linear", normalization="clamp",
        )
        enc = TTFSEncoder(cfg)
        sb = enc(x)
        assert sb.spikes.shape == (B, T, N)
        assert sb.is_binary()
        # Exactly one spike per neuron
        counts = sb.spikes.sum(dim=TIME_DIM)
        assert (counts == 1.0).all(), "TTFS must have exactly 1 spike per neuron"
        _record("TTFSEncoder basic", True)
    except Exception as e:
        _record("TTFSEncoder basic", False, str(e))

    # allow_no_spike
    try:
        cfg = EncodingConfig(
            type="ttfs", num_steps=T,
            latency_mapping="linear", normalization="none",
            allow_no_spike=True,
        )
        enc = TTFSEncoder(cfg)
        # Input with some near-zero values
        x_sparse = torch.zeros(B, N)
        x_sparse[:, :N // 2] = torch.rand(B, N // 2) * 0.5 + 0.5  # above threshold
        x_sparse[:, N // 2:] = 0.001  # below threshold
        sb = enc(x_sparse, return_aux=True)
        counts = sb.spikes.sum(dim=TIME_DIM)  # (B, N)
        # Neurons with near-zero input should have 0 spikes
        silent_counts = counts[:, N // 2:]
        assert (silent_counts == 0.0).all(), "Silent neurons should have 0 spikes"
        # Check aux spike_times has -1 for silent neurons
        assert "spike_times" in sb.aux
        silent_times = sb.aux["spike_times"][:, N // 2:]
        assert (silent_times == -1.0).all(), "Silent neurons should have time=-1"
        _record("TTFSEncoder allow_no_spike", True)
    except Exception as e:
        _record("TTFSEncoder allow_no_spike", False, str(e))

    # Jitter during training only
    try:
        cfg = EncodingConfig(
            type="ttfs", num_steps=50,
            latency_mapping="linear", normalization="clamp",
            jitter=2.0,
        )
        enc = TTFSEncoder(cfg)
        enc.train()
        x_fixed = torch.full((B, N), 0.5)
        sb1 = enc(x_fixed)
        sb2 = enc(x_fixed)
        # With jitter, two train calls should usually differ
        # (probabilistic -- with jitter=2.0, extremely unlikely to be identical)
        train_differ = not torch.allclose(sb1.spikes, sb2.spikes)

        enc.eval()
        # Without jitter at eval, should be deterministic (no jitter)
        sb3 = enc(x_fixed)
        sb4 = enc(x_fixed)
        eval_same = torch.allclose(sb3.spikes, sb4.spikes)

        assert train_differ, "Jitter should cause training outputs to differ"
        assert eval_same, "Eval mode should be deterministic (no jitter)"
        _record("TTFSEncoder jitter train/eval", True)
    except Exception as e:
        _record("TTFSEncoder jitter train/eval", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 7: PopulationEncoder
    # ------------------------------------------------------------------
    print("\n--- Test Group 7: PopulationEncoder ---")
    x = torch.rand(B, N)

    try:
        P = 10
        cfg = EncodingConfig(
            type="population", num_steps=T,
            population_size=P, population_sigma=0.2,
            normalization="clamp",
        )
        enc = PopulationEncoder(cfg)
        sb = enc(x)
        expected_features = N * P
        assert sb.spikes.shape == (B, T, expected_features), \
            f"Expected (B,T,N*P)=({B},{T},{expected_features}), got {sb.spikes.shape}"
        assert sb.is_binary()
        _record("PopulationEncoder shape/binary", True)
    except Exception as e:
        _record("PopulationEncoder shape/binary", False, str(e))

    try:
        P = 8
        cfg = EncodingConfig(
            type="population", num_steps=T,
            population_size=P, normalization="clamp",
        )
        enc = PopulationEncoder(cfg)
        sb = enc(x, return_aux=True)
        assert "population_map" in sb.aux
        pmap = sb.aux["population_map"]
        assert isinstance(pmap, dict)
        assert len(pmap) == N, f"Expected {N} groups, got {len(pmap)}"
        # Check first group
        assert pmap[0] == list(range(P)), f"Group 0 should be [0..P-1]"
        _record("PopulationEncoder population_map", True)
    except Exception as e:
        _record("PopulationEncoder population_map", False, str(e))

    # Tuning curve: input at center should have highest activation
    try:
        P = 5
        cfg = EncodingConfig(
            type="population", num_steps=100,
            population_size=P, population_sigma=0.1,
            normalization="none", deterministic_eval=True,
        )
        enc = PopulationEncoder(cfg)
        enc.eval()
        # Input exactly at center[2] = 0.5 (for P=5: centers = [0, 0.25, 0.5, 0.75, 1])
        x_center = torch.tensor([[0.5]])
        sb = enc(x_center, return_aux=True)
        # The middle neuron (index 2) should have highest spike count
        spike_counts = sb.spikes.squeeze(0).sum(dim=0)  # (P,)
        max_idx = spike_counts.argmax().item()
        assert max_idx == 2, f"Center neuron should have most spikes, got idx {max_idx}"
        _record("PopulationEncoder tuning curve", True)
    except Exception as e:
        _record("PopulationEncoder tuning curve", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 8: DeltaEncoder
    # ------------------------------------------------------------------
    print("\n--- Test Group 8: DeltaEncoder ---")

    try:
        cfg = EncodingConfig(type="delta", num_steps=10, delta_threshold=0.1)
        enc = DeltaEncoder(cfg)
        x1 = torch.zeros(B, N)
        sb1 = enc(x1)
        assert sb1.spikes.shape == (B, 10, N * 2), \
            f"Expected (B,10,N*2), got {sb1.spikes.shape}"
        _record("DeltaEncoder shape (ON+OFF)", True)
    except Exception as e:
        _record("DeltaEncoder shape (ON+OFF)", False, str(e))

    try:
        cfg = EncodingConfig(type="delta", num_steps=5, delta_threshold=0.1)
        enc = DeltaEncoder(cfg)
        enc.reset()
        # First input: all zeros (delta from 0 -> 0 = no spikes)
        x1 = torch.zeros(B, N)
        sb1 = enc(x1)
        assert sb1.spikes.sum().item() == 0.0, "No change from zero should produce no spikes"
        # Second input: large positive change
        x2 = torch.ones(B, N) * 0.5
        sb2 = enc(x2)
        on_spikes = sb2.spikes[:, :, :N]
        off_spikes = sb2.spikes[:, :, N:]
        assert on_spikes.sum() > 0, "Positive change should produce ON spikes"
        assert off_spikes.sum() == 0, "Positive change should not produce OFF spikes"
        # Third input: large negative change
        x3 = torch.zeros(B, N)
        sb3 = enc(x3)
        on3 = sb3.spikes[:, :, :N]
        off3 = sb3.spikes[:, :, N:]
        assert on3.sum() == 0, "Negative change should not produce ON spikes"
        assert off3.sum() > 0, "Negative change should produce OFF spikes"
        _record("DeltaEncoder ON/OFF logic", True)
    except Exception as e:
        _record("DeltaEncoder ON/OFF logic", False, str(e))

    try:
        cfg = EncodingConfig(type="delta", num_steps=5, delta_threshold=0.1)
        enc = DeltaEncoder(cfg)
        enc.reset()
        x1 = torch.rand(B, N)
        _ = enc(x1)
        enc.reset()
        # After reset, prev_input should be cleared
        assert not enc._prev_input_initialized, "Reset should clear initialization flag"
        _record("DeltaEncoder reset()", True)
    except Exception as e:
        _record("DeltaEncoder reset()", False, str(e))

    try:
        cfg = EncodingConfig(type="delta", num_steps=5, delta_threshold=0.1)
        enc = DeltaEncoder(cfg)
        enc.reset()
        x1 = torch.zeros(B, N)
        _ = enc(x1)
        x2 = torch.ones(B, N) * 0.5
        sb = enc(x2, return_aux=True)
        assert "delta" in sb.aux
        assert "on_count" in sb.aux
        assert "off_count" in sb.aux
        _record("DeltaEncoder return_aux", True)
    except Exception as e:
        _record("DeltaEncoder return_aux", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 9: Factory Function
    # ------------------------------------------------------------------
    print("\n--- Test Group 9: Factory Function ---")

    for enc_type in ENCODING_TYPES:
        try:
            cfg = EncodingConfig(type=enc_type, num_steps=T)
            enc = create_encoder(cfg)
            assert isinstance(enc, SpikeEncoder), f"Expected SpikeEncoder, got {type(enc)}"
            sb = enc(torch.rand(B, N))
            assert isinstance(sb, SpikeBatch), f"Expected SpikeBatch, got {type(sb)}"
            assert sb.batch_size == B
            assert sb.time_steps == T
            _record(f"create_encoder({enc_type})", True)
        except Exception as e:
            _record(f"create_encoder({enc_type})", False, str(e))

    try:
        raised = False
        try:
            bad_cfg = EncodingConfig.__new__(EncodingConfig)
            bad_cfg.type = "nonexistent_encoder_xyz"
            create_encoder(bad_cfg)
        except (ValueError, AttributeError):
            raised = True
        assert raised, "Should reject unknown encoder type"
        _record("create_encoder rejects bad config", True)
    except Exception as e:
        _record("create_encoder rejects bad config", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 10: AMP Compatibility
    # ------------------------------------------------------------------
    print("\n--- Test Group 10: AMP Compatibility ---")

    amp_device = "cpu"
    # torch.amp.autocast works on CPU with dtype=torch.bfloat16
    try:
        cfg = EncodingConfig(type="rate_bernoulli", num_steps=T, normalization="clamp")
        enc = RateEncoder(cfg)
        x_amp = torch.rand(B, N)

        with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16):
            sb = enc(x_amp)
        # Spikes should still be valid binary (sampling was in fp32 internally)
        assert sb.is_binary(), "AMP: spikes should remain binary"
        assert sb.spikes.shape == (B, T, N)
        _record("AMP RateEncoder bernoulli", True)
    except Exception as e:
        _record("AMP RateEncoder bernoulli", False, str(e))

    try:
        cfg = EncodingConfig(type="latency", num_steps=T, normalization="clamp")
        enc = LatencyEncoder(cfg)
        x_amp = torch.rand(B, N)

        with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16):
            sb = enc(x_amp)
        assert sb.is_binary()
        counts = sb.spikes.sum(dim=TIME_DIM)
        assert (counts == 1.0).all(), "AMP: latency should still have exactly 1 spike"
        _record("AMP LatencyEncoder", True)
    except Exception as e:
        _record("AMP LatencyEncoder", False, str(e))

    try:
        cfg = EncodingConfig(type="population", num_steps=T, population_size=4, normalization="clamp")
        enc = PopulationEncoder(cfg)
        x_amp = torch.rand(B, N)

        with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16):
            sb = enc(x_amp)
        assert sb.is_binary()
        assert sb.spikes.shape == (B, T, N * 4)
        _record("AMP PopulationEncoder", True)
    except Exception as e:
        _record("AMP PopulationEncoder", False, str(e))

    try:
        cfg = EncodingConfig(type="delta", num_steps=5, delta_threshold=0.1)
        enc = DeltaEncoder(cfg)
        enc.reset()
        x_amp = torch.rand(B, N)

        with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16):
            sb = enc(x_amp)
        assert sb.is_binary()
        assert sb.spikes.shape == (B, 5, N * 2)
        _record("AMP DeltaEncoder", True)
    except Exception as e:
        _record("AMP DeltaEncoder", False, str(e))

    try:
        cfg = EncodingConfig(type="ttfs", num_steps=T, normalization="clamp")
        enc = TTFSEncoder(cfg)
        x_amp = torch.rand(B, N)

        with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16):
            sb = enc(x_amp)
        assert sb.is_binary()
        counts = sb.spikes.sum(dim=TIME_DIM)
        assert (counts == 1.0).all(), "AMP: TTFS should have exactly 1 spike"
        _record("AMP TTFSEncoder", True)
    except Exception as e:
        _record("AMP TTFSEncoder", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 11: deterministic_eval for all stochastic encoders
    # ------------------------------------------------------------------
    print("\n--- Test Group 11: deterministic_eval ---")

    try:
        cfg = EncodingConfig(
            type="rate_bernoulli", num_steps=T,
            normalization="clamp", deterministic_eval=True,
        )
        enc = RateEncoder(cfg)
        enc.eval()
        x_test = torch.rand(B, N)
        sb1 = enc(x_test)
        sb2 = enc(x_test)
        assert torch.allclose(sb1.spikes, sb2.spikes), \
            "deterministic_eval should make eval-mode output reproducible"
        _record("deterministic_eval RateEncoder", True)
    except Exception as e:
        _record("deterministic_eval RateEncoder", False, str(e))

    try:
        cfg = EncodingConfig(
            type="rate_poisson", num_steps=T,
            normalization="clamp", deterministic_eval=True,
        )
        enc = RateEncoder(cfg)
        enc.eval()
        x_test = torch.rand(B, N)
        sb1 = enc(x_test)
        sb2 = enc(x_test)
        assert torch.allclose(sb1.spikes, sb2.spikes), \
            "deterministic_eval should make Poisson eval-mode reproducible"
        _record("deterministic_eval RateEncoder poisson", True)
    except Exception as e:
        _record("deterministic_eval RateEncoder poisson", False, str(e))

    # PopulationEncoder uses RateEncoder internally
    try:
        cfg = EncodingConfig(
            type="population", num_steps=T,
            population_size=4, normalization="clamp",
            deterministic_eval=True,
        )
        enc = PopulationEncoder(cfg)
        enc.eval()
        x_test = torch.rand(B, N)
        sb1 = enc(x_test)
        sb2 = enc(x_test)
        assert torch.allclose(sb1.spikes, sb2.spikes), \
            "deterministic_eval should make PopulationEncoder eval-mode reproducible"
        _record("deterministic_eval PopulationEncoder", True)
    except Exception as e:
        _record("deterministic_eval PopulationEncoder", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 12: Generator Reproducibility
    # ------------------------------------------------------------------
    print("\n--- Test Group 12: Generator Reproducibility ---")

    try:
        cfg = EncodingConfig(
            type="rate_bernoulli", num_steps=T,
            normalization="clamp", seed=42,
        )
        enc = RateEncoder(cfg)
        enc.train()
        x_test = torch.rand(B, N)
        sb1 = enc(x_test)
        sb2 = enc(x_test)  # same seed in config -> same generator each call
        assert torch.allclose(sb1.spikes, sb2.spikes), \
            "Same seed should produce identical spike trains"
        _record("Generator same seed -> same spikes", True)
    except Exception as e:
        _record("Generator same seed -> same spikes", False, str(e))

    try:
        cfg1 = EncodingConfig(
            type="rate_bernoulli", num_steps=T,
            normalization="clamp", seed=42,
        )
        cfg2 = EncodingConfig(
            type="rate_bernoulli", num_steps=T,
            normalization="clamp", seed=123,
        )
        enc1 = RateEncoder(cfg1)
        enc2 = RateEncoder(cfg2)
        enc1.train()
        enc2.train()
        x_test = torch.rand(B, N)
        sb1 = enc1(x_test)
        sb2 = enc2(x_test)
        assert not torch.allclose(sb1.spikes, sb2.spikes), \
            "Different seeds should (almost certainly) produce different spike trains"
        _record("Generator different seed -> different spikes", True)
    except Exception as e:
        _record("Generator different seed -> different spikes", False, str(e))

    try:
        # Explicit seed override at encode time
        cfg = EncodingConfig(
            type="rate_bernoulli", num_steps=T,
            normalization="clamp",
        )
        enc = RateEncoder(cfg)
        enc.train()
        x_test = torch.rand(B, N)
        sb1 = enc.encode(x_test, seed=99)
        sb2 = enc.encode(x_test, seed=99)
        assert torch.allclose(sb1.spikes, sb2.spikes), \
            "Same explicit seed should produce identical spikes"
        _record("Generator explicit seed override", True)
    except Exception as e:
        _record("Generator explicit seed override", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 13: Normalization Methods
    # ------------------------------------------------------------------
    print("\n--- Test Group 13: Normalization Methods ---")

    try:
        cfg = EncodingConfig(type="rate_bernoulli", num_steps=T, normalization="none")
        enc = RateEncoder(cfg)
        x_test = torch.tensor([[3.0, -1.0, 0.5]])
        normed = enc._normalize(x_test, "none")
        assert torch.allclose(normed, x_test.float()), "none should pass through"
        _record("Normalization: none", True)
    except Exception as e:
        _record("Normalization: none", False, str(e))

    try:
        cfg = EncodingConfig(type="rate_bernoulli", num_steps=T)
        enc = RateEncoder(cfg)
        x_test = torch.tensor([[0.0, 5.0, 10.0]])
        normed = enc._normalize(x_test, "minmax")
        assert normed.min() >= -1e-6, f"minmax min: {normed.min()}"
        assert normed.max() <= 1.0 + 1e-6, f"minmax max: {normed.max()}"
        assert torch.allclose(normed, torch.tensor([[0.0, 0.5, 1.0]])), \
            f"Expected [0, 0.5, 1], got {normed}"
        _record("Normalization: minmax", True)
    except Exception as e:
        _record("Normalization: minmax", False, str(e))

    try:
        cfg = EncodingConfig(type="rate_bernoulli", num_steps=T)
        enc = RateEncoder(cfg)
        x_test = torch.tensor([[0.0, 100.0, -100.0]])
        normed = enc._normalize(x_test, "sigmoid")
        assert (normed >= 0.0).all() and (normed <= 1.0).all(), \
            f"sigmoid should be in [0,1], got [{normed.min()}, {normed.max()}]"
        _record("Normalization: sigmoid", True)
    except Exception as e:
        _record("Normalization: sigmoid", False, str(e))

    try:
        cfg = EncodingConfig(type="rate_bernoulli", num_steps=T)
        enc = RateEncoder(cfg)
        x_test = torch.tensor([[-1.0, 0.5, 2.0]])
        normed = enc._normalize(x_test, "clamp")
        expected = torch.tensor([[0.0, 0.5, 1.0]])
        assert torch.allclose(normed, expected), f"Expected {expected}, got {normed}"
        _record("Normalization: clamp", True)
    except Exception as e:
        _record("Normalization: clamp", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 14: SpikeDecoder
    # ------------------------------------------------------------------
    print("\n--- Test Group 14: SpikeDecoder ---")

    try:
        spk = torch.zeros(B, T, N)
        spk[:, 0, :] = 1.0  # spike at t=0
        spk[:, T // 2, :] = 1.0  # spike at t=T/2
        sb = SpikeBatch(spikes=spk)
        decoder = SpikeDecoder(method="rate")
        out = decoder(sb)
        expected_rate = 2.0 / T
        assert out.shape == (B, N)
        assert torch.allclose(out, torch.full((B, N), expected_rate)), \
            f"Rate decode mismatch: {out.mean()} vs {expected_rate}"
        _record("SpikeDecoder rate", True)
    except Exception as e:
        _record("SpikeDecoder rate", False, str(e))

    try:
        spk = torch.zeros(B, T, N)
        spk[:, 3, :] = 1.0  # first spike at t=3
        spk[:, 10, :] = 1.0  # second spike at t=10
        sb = SpikeBatch(spikes=spk)
        decoder = SpikeDecoder(method="first_spike")
        out = decoder(sb)
        # argmax finds t=3, value = 1 - 3/T
        expected = 1.0 - 3.0 / T
        assert out.shape == (B, N)
        assert torch.allclose(out, torch.full((B, N), expected)), \
            f"First spike decode: {out.mean()} vs {expected}"
        _record("SpikeDecoder first_spike", True)
    except Exception as e:
        _record("SpikeDecoder first_spike", False, str(e))

    try:
        membrane = torch.randn(B, N)
        decoder = SpikeDecoder(method="membrane")
        out = decoder(torch.zeros(B, T, N), membrane=membrane)
        assert torch.allclose(out, membrane)
        _record("SpikeDecoder membrane", True)
    except Exception as e:
        _record("SpikeDecoder membrane", False, str(e))

    try:
        # SpikeBatch input
        sb = SpikeBatch(spikes=torch.ones(B, T, N))
        decoder = SpikeDecoder(method="rate")
        out = decoder(sb)
        assert torch.allclose(out, torch.ones(B, N)), "All-1 spikes -> rate=1.0"
        _record("SpikeDecoder SpikeBatch input", True)
    except Exception as e:
        _record("SpikeDecoder SpikeBatch input", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 15: Convenience Constructors
    # ------------------------------------------------------------------
    print("\n--- Test Group 15: Convenience Constructors ---")

    try:
        enc = create_rate_encoder(num_steps=20, method="poisson", gain=0.8, bias=0.1)
        assert isinstance(enc, RateEncoder)
        assert enc.config.num_steps == 20
        assert enc.config.rate_gain == 0.8
        sb = enc(torch.rand(B, N))
        assert sb.spikes.shape == (B, 20, N)
        _record("create_rate_encoder", True)
    except Exception as e:
        _record("create_rate_encoder", False, str(e))

    try:
        enc = create_latency_encoder(num_steps=30, mapping="log")
        assert isinstance(enc, LatencyEncoder)
        sb = enc(torch.rand(B, N))
        assert sb.spikes.shape == (B, 30, N)
        _record("create_latency_encoder", True)
    except Exception as e:
        _record("create_latency_encoder", False, str(e))

    try:
        enc = create_ttfs_encoder(num_steps=20, jitter=1.0, allow_no_spike=True)
        assert isinstance(enc, TTFSEncoder)
        assert enc.jitter == 1.0
        assert enc.allow_no_spike is True
        sb = enc(torch.rand(B, N))
        assert sb.time_steps == 20
        _record("create_ttfs_encoder", True)
    except Exception as e:
        _record("create_ttfs_encoder", False, str(e))

    try:
        enc = create_population_encoder(num_steps=15, population_size=6, sigma=0.3)
        assert isinstance(enc, PopulationEncoder)
        assert enc.population_size == 6
        sb = enc(torch.rand(B, N))
        assert sb.spikes.shape == (B, 15, N * 6)
        _record("create_population_encoder", True)
    except Exception as e:
        _record("create_population_encoder", False, str(e))

    try:
        enc = create_delta_encoder(num_steps=8, threshold=0.05)
        assert isinstance(enc, DeltaEncoder)
        assert enc.threshold == 0.05
        enc.reset()
        sb = enc(torch.rand(B, N))
        assert sb.spikes.shape == (B, 8, N * 2)
        _record("create_delta_encoder", True)
    except Exception as e:
        _record("create_delta_encoder", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 16: Edge Cases
    # ------------------------------------------------------------------
    print("\n--- Test Group 16: Edge Cases ---")

    # Single timestep
    try:
        cfg = EncodingConfig(type="rate_bernoulli", num_steps=1, normalization="clamp")
        enc = RateEncoder(cfg)
        sb = enc(torch.rand(B, N))
        assert sb.spikes.shape == (B, 1, N)
        _record("Edge: single timestep", True)
    except Exception as e:
        _record("Edge: single timestep", False, str(e))

    # Batch size 1
    try:
        cfg = EncodingConfig(type="rate_bernoulli", num_steps=T, normalization="clamp")
        enc = RateEncoder(cfg)
        sb = enc(torch.rand(1, N))
        assert sb.batch_size == 1
        _record("Edge: batch size 1", True)
    except Exception as e:
        _record("Edge: batch size 1", False, str(e))

    # All-zero input
    try:
        cfg = EncodingConfig(type="rate_bernoulli", num_steps=T, normalization="none")
        enc = RateEncoder(cfg)
        sb = enc(torch.zeros(B, N))
        assert sb.spikes.sum().item() == 0.0, "Zero input -> no spikes"
        _record("Edge: all-zero input", True)
    except Exception as e:
        _record("Edge: all-zero input", False, str(e))

    # All-one input with gain=1
    try:
        cfg = EncodingConfig(
            type="rate_deterministic", num_steps=T,
            normalization="none", rate_gain=1.0,
        )
        enc = RateEncoder(cfg)
        sb = enc(torch.ones(B, N))
        # rate=1.0 deterministic: should spike every timestep
        expected_count = float(T)
        actual_count = sb.spikes.sum(dim=TIME_DIM).mean().item()
        assert abs(actual_count - expected_count) < 1.0, \
            f"Rate 1.0 deterministic: expected ~{expected_count} spikes, got {actual_count}"
        _record("Edge: all-one input deterministic", True)
    except Exception as e:
        _record("Edge: all-one input deterministic", False, str(e))

    # Large feature dimension
    try:
        cfg = EncodingConfig(type="rate_bernoulli", num_steps=5, normalization="clamp")
        enc = RateEncoder(cfg)
        sb = enc(torch.rand(2, 4096))
        assert sb.spikes.shape == (2, 5, 4096)
        _record("Edge: large feature dim (4096)", True)
    except Exception as e:
        _record("Edge: large feature dim (4096)", False, str(e))

    # Multi-dimensional features (e.g., image-like)
    try:
        cfg = EncodingConfig(type="rate_bernoulli", num_steps=T, normalization="clamp")
        enc = RateEncoder(cfg)
        x_img = torch.rand(B, 3, 8, 8)  # (B, C, H, W)
        sb = enc(x_img)
        assert sb.spikes.shape == (B, T, 3, 8, 8), \
            f"Multi-dim features: expected (B,T,3,8,8), got {sb.spikes.shape}"
        _record("Edge: multi-dim features (image-like)", True)
    except Exception as e:
        _record("Edge: multi-dim features (image-like)", False, str(e))

    # LatencyEncoder with t_min / t_max clamping
    try:
        cfg = EncodingConfig(
            type="latency", num_steps=50,
            latency_mapping="linear", normalization="none",
            t_min=5, t_max=20,
        )
        enc = LatencyEncoder(cfg)
        sb = enc(torch.rand(B, N), return_aux=True)
        times = sb.aux["spike_times"]
        assert (times >= 4.5).all(), f"Times should be >= ~5, min={times.min()}"
        assert (times <= 20.5).all(), f"Times should be <= ~20, max={times.max()}"
        _record("Edge: latency t_min/t_max clamping", True)
    except Exception as e:
        _record("Edge: latency t_min/t_max clamping", False, str(e))

    # ------------------------------------------------------------------
    # TEST GROUP 17: Batch-First Convention Consistency
    # ------------------------------------------------------------------
    print("\n--- Test Group 17: Batch-First Convention ---")

    for enc_type in ENCODING_TYPES:
        try:
            cfg = EncodingConfig(type=enc_type, num_steps=T, normalization="clamp")
            enc = create_encoder(cfg)
            if isinstance(enc, DeltaEncoder):
                enc.reset()
            sb = enc(torch.rand(B, N))
            # Verify batch-first: dim 0 = B, dim 1 = T
            assert sb.spikes.shape[BATCH_DIM] == B, \
                f"{enc_type}: dim 0 should be batch={B}, got {sb.spikes.shape[BATCH_DIM]}"
            assert sb.spikes.shape[TIME_DIM] == T, \
                f"{enc_type}: dim 1 should be time={T}, got {sb.spikes.shape[TIME_DIM]}"
            _record(f"batch-first {enc_type}", True)
        except Exception as e:
            _record(f"batch-first {enc_type}", False, str(e))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Self-Test Summary: {total_pass} PASSED, {total_fail} FAILED "
          f"out of {total_pass + total_fail} tests")
    print("=" * 70)

    if total_fail > 0:
        print("\nFailed tests:")
        for name, status in results.items():
            if status.startswith("FAIL"):
                print(f"  - {name}: {status}")
        print()

    if total_fail == 0:
        print("\nAll tests passed successfully.\n")
    else:
        print(f"\n{total_fail} test(s) failed. See details above.\n")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    _run_self_test()
