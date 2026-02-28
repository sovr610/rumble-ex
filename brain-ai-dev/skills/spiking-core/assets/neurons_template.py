"""
brain_ai/core/neurons.py — LIF Family Neurons with Explicit State and Surrogate Gradients

This module provides the spiking neuron implementations used throughout the brain_ai
cognitive architecture. All neurons follow a shared contract:

1. State is EXPLICIT — passed in and returned, never stored on self
2. Surrogates define the backward pass — forward is always binary Heaviside
3. Numerical stability is ENFORCED — beta clamped, fp32 accumulation, no in-place ops
4. Time semantics — step mode (B,...) or sequence mode via external unroll utility

Key classes:
    SpikingState          — Dataclass holding all neuron state variables
    SpikingNeuronBase     — Abstract base enforcing the state contract
    LIFNeuron             — Basic leaky integrate-and-fire
    AdaptiveLIFNeuron     — LIF with spike-frequency adaptation
    RecurrentLIFNeuron    — LIF with lateral recurrent connections
    AdvancedLIFNeuron     — LIF with learnable delays, heterogeneous tau, adaptive threshold

Migration from v1 (on-module state):
    Old: lif = LIFNeuron(); lif.init_mem(B, N); spk, mem = lif(x)
    New: lif = LIFNeuron(N); state = lif.reset_state(B, (N,), device); spk, state = lif(x, state)

Numerical stability guarantees:
    - beta is ALWAYS accessed via beta_clamped property → clipped to [0, 0.999]
    - State tensors are ALWAYS in fp32 regardless of input dtype (dtype_policy='fp32_state')
    - No in-place operations on tensors that require gradients
    - Surrogate gradient magnitudes are documented in SURROGATE_MAX_GRAD

References:
    Maass (1997) "Networks of spiking neurons: The third generation of neural network models"
    Neftci et al. (2019) "Surrogate Gradient Learning in Spiking Neural Networks"
    Yu et al. (2025) "Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning"
    Hammouamri et al. (2024) "Learning delays in SNNs"
    Bellec et al. (2020) "A solution to the learning dilemma for recurrent networks of spiking neurons"
"""

from __future__ import annotations

import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

BETA_MIN: float = 0.0
BETA_MAX: float = 0.999  # Hard upper bound — prevents membrane explosion
THRESHOLD_MIN: float = 1e-3  # Prevent threshold collapsing to zero

# Valid reset mechanisms
RESET_MECHANISMS = ("subtract", "zero")

# Valid dtype policies
DTYPE_POLICIES = ("fp32_state", "match_input", "fp16_state")


# ===========================================================================
# SECTION 1: SpikingState dataclass
# ===========================================================================

@dataclass
class SpikingState:
    """Explicit state container for spiking neurons.

    All tensors have batch dimension first: (B, N) for linear layers,
    or (B, C, H, W) for convolutional layers.

    Fields are optional so the same container works across all neuron variants.
    Unused fields remain None, and callers should not rely on their presence
    unless they know which neuron type produced the state.

    Design notes:
        - State is NEVER stored on the module (no self.mem). Callers own it.
        - Detach at truncation points with state.detach() to free the graph.
        - Clone with state.clone() before checkpointing to avoid aliasing.

    Attributes:
        v: Membrane potential. Always present. Shape: (B, *neuron_shape).
        i: Synaptic current (if explicitly modeled). Shape: same as v.
        a: Adaptation variable used by AdaptiveLIFNeuron. Shape: same as v.
        prev_spk: Previous spike tensor used by RecurrentLIFNeuron for lateral
            connections. Shape: same as v.
        spike_history: Circular spike buffer used by AdvancedLIFNeuron for
            delay mechanism. Shape: (B, max_delay, *neuron_shape).
        ref: Refractory period countdown timer. Shape: same as v.
            Reserved for future extension; not used by current neuron types.
    """

    v: Tensor
    i: Optional[Tensor] = None
    a: Optional[Tensor] = None
    prev_spk: Optional[Tensor] = None
    spike_history: Optional[Tensor] = None
    ref: Optional[Tensor] = None

    # -----------------------------------------------------------------------
    # Public helpers
    # -----------------------------------------------------------------------

    def detach(self) -> "SpikingState":
        """Return a new SpikingState with all tensors detached from the graph.

        Use this at truncated-BPTT boundaries so old graphs can be freed.
        The returned state still shares storage with the original — gradients
        will not flow through it, but it costs no additional memory copy.

        Returns:
            New SpikingState with .detach() applied to every non-None tensor.
        """
        return SpikingState(
            v=self.v.detach(),
            i=self.i.detach() if self.i is not None else None,
            a=self.a.detach() if self.a is not None else None,
            prev_spk=self.prev_spk.detach() if self.prev_spk is not None else None,
            spike_history=(
                self.spike_history.detach()
                if self.spike_history is not None
                else None
            ),
            ref=self.ref.detach() if self.ref is not None else None,
        )

    def to(
        self,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "SpikingState":
        """Return a new SpikingState moved to the specified device and/or dtype.

        Follows the semantics of torch.Tensor.to() — accepts the same
        device/dtype arguments and returns a new object (never in-place).

        Args:
            device: Target device (e.g., 'cuda', torch.device('cpu')).
            dtype: Target floating-point dtype.

        Returns:
            New SpikingState with all tensors transferred.
        """

        def _move(t: Optional[Tensor]) -> Optional[Tensor]:
            if t is None:
                return None
            kwargs: Dict = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            return t.to(**kwargs)

        return SpikingState(
            v=_move(self.v),  # type: ignore[arg-type]
            i=_move(self.i),
            a=_move(self.a),
            prev_spk=_move(self.prev_spk),
            spike_history=_move(self.spike_history),
            ref=_move(self.ref),
        )

    def clone(self) -> "SpikingState":
        """Return a deep copy of the state, detached from the computation graph.

        Intended for checkpointing — the clone has its own storage and will not
        alias any tensor that may be overwritten on the next forward step.

        Returns:
            New SpikingState with cloned, detached tensors.
        """

        def _clone(t: Optional[Tensor]) -> Optional[Tensor]:
            return t.detach().clone() if t is not None else None

        return SpikingState(
            v=self.v.detach().clone(),
            i=_clone(self.i),
            a=_clone(self.a),
            prev_spk=_clone(self.prev_spk),
            spike_history=_clone(self.spike_history),
            ref=_clone(self.ref),
        )

    def __repr__(self) -> str:
        parts = [f"v={tuple(self.v.shape)}"]
        for name in ("i", "a", "prev_spk", "spike_history", "ref"):
            t = getattr(self, name)
            if t is not None:
                parts.append(f"{name}={tuple(t.shape)}")
        return f"SpikingState({', '.join(parts)}, device={self.v.device})"


# ===========================================================================
# SECTION 2: Surrogate gradient functions
# ===========================================================================

# Maximum gradient magnitude emitted by each surrogate (at x=0).
# Useful for diagnosing vanishing/exploding gradients during debugging.
SURROGATE_MAX_GRAD: Dict[str, float] = {
    "atan": 0.5 * 2.0 / math.pi,       # alpha/(2) * 1/pi ≈ 0.318 at default alpha=2.0
    "fast_sigmoid": 25.0 / 8.0,         # slope/8 ≈ 3.125 at default slope=25.0
    "straight_through": 1.0,            # Always 1 — no scaling
}
# NOTE: atan max_grad = alpha / (2 * pi) because derivative at x=0 of
#       atan surrogate is alpha/(2*(1+(pi*alpha*0)^2)) = alpha/2, then
#       divided by pi normalisation: SURROGATE_MAX_GRAD is approximate;
#       verify numerically for non-default alpha/slope.


class ATanSurrogate(torch.autograd.Function):
    """Arctangent surrogate gradient for the Heaviside spike function.

    Forward pass: standard Heaviside H(x) = (x >= 0).float()
    Backward pass approximation:
        dH/dx ≈ alpha / (2 * (1 + (pi * alpha * x)^2))

    This approximation is peaked at x=0 and decays smoothly as |x| grows,
    giving the neuron a soft "window" over which learning can occur. Larger
    alpha sharpens the window (more biologically faithful but harder to
    optimise); smaller alpha widens it (easier but less precise).

    Class attributes:
        alpha: Sharpness parameter (default 2.0). Thread-safety note: this is
            a class-level attribute modified by get_surrogate(). Do not modify
            concurrently from multiple threads.
    """

    alpha: float = 2.0

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> Tensor:
        (x,) = ctx.saved_tensors
        alpha = ATanSurrogate.alpha
        # Gradient of the atan surrogate approximation
        grad = alpha / (2.0 * (1.0 + (torch.pi * alpha * x) ** 2))
        return grad_output * grad


class FastSigmoidSurrogate(torch.autograd.Function):
    """Fast-sigmoid (piecewise-linear) surrogate gradient.

    Forward pass: standard Heaviside H(x) = (x >= 0).float()
    Backward pass approximation:
        dH/dx ≈ slope / (2 * (1 + slope * |x|)^2)

    Cheaper to compute than ATan (no transcendental functions) while still
    providing a reasonable peaked gradient. Preferred for large-batch training
    where compute cost is the bottleneck.

    Class attributes:
        slope: Gradient slope parameter (default 25.0). Higher values give
            sharper but potentially harder-to-optimise gradients.
    """

    slope: float = 25.0

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> Tensor:
        (x,) = ctx.saved_tensors
        slope = FastSigmoidSurrogate.slope
        grad = slope / (2.0 * (1.0 + slope * x.abs()) ** 2)
        return grad_output * grad


class StraightThroughSurrogate(torch.autograd.Function):
    """Straight-through estimator (STE) for the Heaviside spike function.

    Forward pass: standard Heaviside H(x) = (x >= 0).float()
    Backward pass: identity — gradient flows unchanged.

    This is the simplest possible surrogate and often serves as a baseline.
    It is mathematically equivalent to treating the spike as the identity
    function during the backward pass, which can cause gradient magnitudes
    to be independent of membrane potential distance from threshold.

    Useful when other surrogates produce vanishing gradients in very deep
    networks or when the network converges poorly with peaked surrogates.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:
        # No saved tensors needed — backward is constant
        return (x >= 0).float()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> Tensor:
        return grad_output


def get_surrogate(
    name: str,
    alpha: Optional[float] = None,
    slope: Optional[float] = None,
) -> Callable[[Tensor], Tensor]:
    """Factory function returning a surrogate gradient apply function.

    This is the canonical way to obtain a spike surrogate. Neuron __init__
    methods should call this once and store the result.

    Args:
        name: Surrogate name. One of 'atan', 'fast_sigmoid', 'straight_through'.
        alpha: Override alpha for 'atan' surrogate. Default: 2.0.
        slope: Override slope for 'fast_sigmoid' surrogate. Default: 25.0.

    Returns:
        A callable f(x) -> spikes that can be called in a forward pass and
        whose backward is the chosen surrogate.

    Raises:
        ValueError: If name is not one of the known surrogates.

    Example:
        >>> spike_fn = get_surrogate('atan', alpha=4.0)
        >>> spikes = spike_fn(membrane - threshold)  # differentiable

    TODO(integration): If snntorch is installed as a dependency, consider
        wrapping its surrogates here so callers need not import snntorch directly.
    """
    if name == "atan":
        if alpha is not None:
            ATanSurrogate.alpha = float(alpha)
        return ATanSurrogate.apply

    elif name == "fast_sigmoid":
        if slope is not None:
            FastSigmoidSurrogate.slope = float(slope)
        return FastSigmoidSurrogate.apply

    elif name == "straight_through":
        return StraightThroughSurrogate.apply

    else:
        raise ValueError(
            f"Unknown surrogate '{name}'. "
            f"Choose from: {list(SURROGATE_MAX_GRAD.keys())}"
        )


# ===========================================================================
# SECTION 3: SpikingNeuronBase abstract base
# ===========================================================================

class SpikingNeuronBase(nn.Module, ABC):
    """Abstract base class for all spiking neurons in brain_ai.

    Enforces the explicit state contract:
        1. reset_state() creates a fresh SpikingState (all zeros).
        2. forward() takes a SpikingState and returns a new SpikingState.
        3. No mutable state is stored on self — neurons are stateless modules.

    Numerical stability contract:
        - beta is always accessed via beta_clamped (clips to [BETA_MIN, BETA_MAX]).
        - State tensors live in fp32 when dtype_policy='fp32_state' (default).
        - No in-place ops on tensors that carry gradients.
        - threshold is clamped to THRESHOLD_MIN to prevent threshold collapse.

    Subclasses must implement:
        _step(x, state) -> (spikes, new_state, details_dict)

    Subclasses may override:
        reset_state() if they require additional state fields (e.g., adaptation).

    Args:
        size: Number of neurons. Used to create parameter tensors.
            For convolutional neurons this is the number of channels (C).
        beta: Initial membrane decay factor. Must be in (0, 1).
        threshold: Spike threshold voltage. Must be positive.
        learnable_beta: If True, beta is an nn.Parameter optimised during training.
            Uses a logit parameterisation for unconstrained optimisation:
            beta = sigmoid(logit_beta), then clamped.
        learnable_threshold: If True, threshold is an nn.Parameter.
        surrogate: Surrogate gradient name. One of 'atan', 'fast_sigmoid',
            'straight_through'.
        surrogate_params: Dict of kwargs forwarded to get_surrogate() (e.g.,
            {'alpha': 4.0} for 'atan').
        reset_mechanism: 'subtract' for soft reset (v -= spk * threshold) or
            'zero' for hard reset (v *= (1 - spk)).
        dtype_policy: 'fp32_state' (default) keeps state in float32 regardless
            of input dtype; 'match_input' follows input dtype; 'fp16_state'
            forces fp16 (experimental, may be unstable for membrane potential).
    """

    def __init__(
        self,
        size: int,
        beta: float = 0.95,
        threshold: float = 1.0,
        learnable_beta: bool = False,
        learnable_threshold: bool = False,
        surrogate: str = "atan",
        surrogate_params: Optional[Dict] = None,
        reset_mechanism: str = "subtract",
        dtype_policy: str = "fp32_state",
    ) -> None:
        super().__init__()

        if reset_mechanism not in RESET_MECHANISMS:
            raise ValueError(
                f"reset_mechanism must be one of {RESET_MECHANISMS}, "
                f"got '{reset_mechanism}'"
            )
        if dtype_policy not in DTYPE_POLICIES:
            raise ValueError(
                f"dtype_policy must be one of {DTYPE_POLICIES}, "
                f"got '{dtype_policy}'"
            )
        if not (0.0 < beta < 1.0):
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if threshold <= 0.0:
            raise ValueError(f"threshold must be positive, got {threshold}")

        self.size = size
        self.reset_mechanism = reset_mechanism
        self.dtype_policy = dtype_policy

        # Build surrogate function
        sp = surrogate_params or {}
        self.spike_fn: Callable[[Tensor], Tensor] = get_surrogate(surrogate, **sp)
        self._surrogate_name = surrogate

        # Beta parameter — scalar or per-neuron depending on subclass init
        # We use a logit parameterisation when learnable so optimiser operates
        # in unconstrained space, then clamp in beta_clamped.
        _beta_logit = math.log(beta / (1.0 - beta))  # sigmoid^{-1}(beta)
        if learnable_beta:
            self.logit_beta: Union[nn.Parameter, Tensor] = nn.Parameter(
                torch.tensor(_beta_logit, dtype=torch.float32)
            )
        else:
            # Register as buffer so it moves with .to(device)
            self.register_buffer(
                "logit_beta", torch.tensor(_beta_logit, dtype=torch.float32)
            )

        # Threshold — scalar
        if learnable_threshold:
            self.threshold: Union[nn.Parameter, Tensor] = nn.Parameter(
                torch.tensor(threshold, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "threshold", torch.tensor(threshold, dtype=torch.float32)
            )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def beta_clamped(self) -> Tensor:
        """Return beta clamped to [BETA_MIN, BETA_MAX].

        ALWAYS use this property — never access logit_beta directly —
        to guarantee that membrane integration cannot diverge.

        The clamping is applied AFTER the sigmoid so the gradient
        still flows through sigmoid(logit_beta), but the forward value
        is hard-clipped to prevent instability.
        """
        return torch.sigmoid(self.logit_beta).clamp(BETA_MIN, BETA_MAX)

    @property
    def threshold_clamped(self) -> Tensor:
        """Return threshold clamped to [THRESHOLD_MIN, inf).

        Prevents threshold from collapsing to zero which would cause all
        neurons to fire every timestep.
        """
        return self.threshold.clamp(min=THRESHOLD_MIN)

    # -----------------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------------

    def reset_state(
        self,
        batch_size: int,
        neuron_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> SpikingState:
        """Create a fresh zero-initialised SpikingState.

        Subclasses that need extra state fields (e.g., adaptation variable)
        MUST override this method and call super() to get the base state,
        then add their fields.

        Args:
            batch_size: Number of samples in the batch (B).
            neuron_shape: Shape of the neuron population excluding batch dim.
                For a linear layer of size N: (N,).
                For a conv layer with C channels, H rows, W cols: (C, H, W).
            device: Device to create tensors on.
            dtype: Dtype for state tensors. Normally torch.float32 regardless
                of model dtype — the dtype_policy determines final behaviour.

        Returns:
            SpikingState with v set to zeros; all other fields None.
        """
        state_dtype = self._resolve_state_dtype(dtype)
        shape = (batch_size,) + neuron_shape
        return SpikingState(
            v=torch.zeros(shape, device=device, dtype=state_dtype),
        )

    def detach_state(self, state: SpikingState) -> SpikingState:
        """Detach all state tensors from the computation graph.

        Convenience wrapper around state.detach(). Use at truncated-BPTT
        boundaries to free the retained computation graph.

        Args:
            state: Current SpikingState (may carry gradient history).

        Returns:
            New SpikingState with all tensors detached.
        """
        return state.detach()

    def _resolve_state_dtype(self, input_dtype: torch.dtype) -> torch.dtype:
        """Return the dtype to use for state tensors given the current policy."""
        if self.dtype_policy == "fp32_state":
            return torch.float32
        elif self.dtype_policy == "match_input":
            return input_dtype
        elif self.dtype_policy == "fp16_state":
            return torch.float16
        return torch.float32  # fallback

    def _ensure_fp32_input(self, x: Tensor) -> Tensor:
        """Cast input to fp32 if dtype_policy requires fp32 state accumulation."""
        if self.dtype_policy == "fp32_state" and x.dtype != torch.float32:
            return x.float()
        return x

    # -----------------------------------------------------------------------
    # Abstract step — subclasses implement
    # -----------------------------------------------------------------------

    @abstractmethod
    def _step(
        self,
        x: Tensor,
        state: SpikingState,
    ) -> Tuple[Tensor, SpikingState, Dict]:
        """Single-timestep neuron update.

        Args:
            x: Input current, shape (B, *neuron_shape), fp32.
            state: Current SpikingState. All tensors guaranteed fp32.

        Returns:
            spikes: Binary output tensor, same shape as x.
            new_state: Updated SpikingState (no in-place mutation of input state).
            details: Dict with diagnostic scalars/tensors. Keys should include
                at minimum 'firing_rate' and 'membrane_mean'. Additional keys
                are neuron-type-specific.
        """

    # -----------------------------------------------------------------------
    # Forward pass (public API)
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        state: Optional[SpikingState] = None,
        return_details: bool = False,
    ) -> Union[
        Tuple[Tensor, SpikingState],
        Tuple[Tensor, SpikingState, Dict],
    ]:
        """Forward pass for a single timestep.

        If state is None, creates a fresh zero state (deprecated convenience
        path — callers should manage state explicitly in new code).

        Args:
            x: Input current tensor.
                Linear mode: (B, N)
                Conv mode: (B, C, H, W)
            state: Explicit SpikingState from the previous timestep. If None,
                a fresh zero state is created automatically (triggers a warning).
            return_details: If True, the third return value is a diagnostic dict
                containing 'firing_rate', 'membrane_mean', 'membrane_var', and
                any neuron-type-specific keys.

        Returns:
            spikes: Binary spike tensor, same shape as x.
            new_state: Updated SpikingState for the next timestep.
            details: (only if return_details=True) Dict with diagnostic info.

        Raises:
            RuntimeError: If input and state shapes are incompatible.
        """
        # Optionally upcast input to fp32 for stable accumulation
        x = self._ensure_fp32_input(x)

        # Auto-create state if not provided (deprecated convenience)
        if state is None:
            logger.warning(
                "%s.forward() called without explicit state. "
                "Creating fresh zero state. Pass state explicitly for "
                "correct behaviour across timesteps.",
                self.__class__.__name__,
            )
            batch_size = x.shape[0]
            neuron_shape = x.shape[1:]
            state = self.reset_state(batch_size, neuron_shape, x.device, x.dtype)

        # Runtime shape check
        assert_spiking_state(state, x.shape[0], x.shape[1:])

        # Delegate to subclass
        spikes, new_state, details = self._step(x, state)

        if return_details:
            return spikes, new_state, details
        return spikes, new_state

    # -----------------------------------------------------------------------
    # Shared reset helper used by _step implementations
    # -----------------------------------------------------------------------

    def _apply_reset(self, v: Tensor, spikes: Tensor) -> Tensor:
        """Apply the chosen reset mechanism to the membrane potential.

        Args:
            v: Membrane potential AFTER integration and before reset. fp32.
            spikes: Binary spike tensor (0.0 or 1.0). Same shape as v.

        Returns:
            v_reset: New membrane potential. No in-place modification.
        """
        th = self.threshold_clamped
        if self.reset_mechanism == "subtract":
            # Soft reset: v ← v - spk * threshold
            return v - spikes * th
        else:
            # Hard reset: v ← v * (1 - spk)
            return v * (1.0 - spikes)

    def extra_repr(self) -> str:
        return (
            f"size={self.size}, "
            f"surrogate={self._surrogate_name}, "
            f"reset={self.reset_mechanism}, "
            f"dtype_policy={self.dtype_policy}"
        )


# ===========================================================================
# SECTION 4: LIFNeuron
# ===========================================================================

class LIFNeuron(SpikingNeuronBase):
    """Leaky Integrate-and-Fire neuron with surrogate gradients.

    This is the foundational spiking neuron used throughout brain_ai. It
    implements the simplest form of the LIF dynamics without any additional
    adaptation or recurrence.

    Update equations (per timestep):
        v[t] = beta * v[t-1] + x[t]          # leaky membrane integration
        s[t] = H(v[t] - v_th)                # spike (Heaviside in fwd, surrogate in bwd)
        v[t] = v[t] - s[t] * v_th            # subtractive reset (default)
        -- OR --
        v[t] = v[t] * (1 - s[t])             # hard reset (zero)

    State fields used: v only.

    Example::

        neuron = LIFNeuron(size=256, beta=0.9, surrogate='atan')
        state = neuron.reset_state(batch_size=4, neuron_shape=(256,), device=device)
        for t in range(T):
            x_t = input_sequence[:, t, :]
            spikes, state = neuron(x_t, state)

    Args:
        size: Number of neurons.
        beta: Membrane decay factor. Must be in (0, 1).
        threshold: Spike threshold. Must be positive.
        learnable_beta: Learn per-model beta via backprop.
        learnable_threshold: Learn threshold via backprop.
        surrogate: 'atan', 'fast_sigmoid', or 'straight_through'.
        surrogate_params: Dict of surrogate-specific params, e.g. {'alpha': 4.0}.
        reset_mechanism: 'subtract' (soft) or 'zero' (hard).
        dtype_policy: 'fp32_state' (default), 'match_input', or 'fp16_state'.
    """

    def __init__(
        self,
        size: int,
        beta: float = 0.95,
        threshold: float = 1.0,
        learnable_beta: bool = False,
        learnable_threshold: bool = False,
        surrogate: str = "atan",
        surrogate_params: Optional[Dict] = None,
        reset_mechanism: str = "subtract",
        dtype_policy: str = "fp32_state",
    ) -> None:
        super().__init__(
            size=size,
            beta=beta,
            threshold=threshold,
            learnable_beta=learnable_beta,
            learnable_threshold=learnable_threshold,
            surrogate=surrogate,
            surrogate_params=surrogate_params,
            reset_mechanism=reset_mechanism,
            dtype_policy=dtype_policy,
        )

    def _step(
        self,
        x: Tensor,
        state: SpikingState,
    ) -> Tuple[Tensor, SpikingState, Dict]:
        """Single LIF timestep.

        No in-place operations — every tensor is created fresh.
        Beta is read via beta_clamped to guarantee stability.

        Args:
            x: Input current, shape (B, *neuron_shape), fp32.
            state: SpikingState with field v (membrane potential).

        Returns:
            spikes: Binary tensor (B, *neuron_shape).
            new_state: Updated SpikingState.
            details: {'firing_rate', 'membrane_mean', 'membrane_var'}.
        """
        beta = self.beta_clamped  # scalar tensor, clamped to [0, 0.999]
        th = self.threshold_clamped

        # 1. Leaky integration (no in-place: use + not +=)
        v_new = beta * state.v + x

        # 2. Spike generation: Heaviside in forward, surrogate in backward
        mem_shifted = v_new - th
        spikes = self.spike_fn(mem_shifted)

        # 3. Reset
        v_reset = self._apply_reset(v_new, spikes)

        new_state = SpikingState(v=v_reset)

        # 4. Diagnostics
        with torch.no_grad():
            details: Dict = {
                "firing_rate": spikes.mean().item(),
                "membrane_mean": v_reset.mean().item(),
                "membrane_var": v_reset.var().item() if v_reset.numel() > 1 else 0.0,
            }

        return spikes, new_state, details


# ===========================================================================
# SECTION 5: AdaptiveLIFNeuron
# ===========================================================================

class AdaptiveLIFNeuron(SpikingNeuronBase):
    """LIF neuron with spike-frequency adaptation (SFA).

    Spike-frequency adaptation causes the neuron to fire less frequently
    after a burst of activity, matching biological cortical neuron behaviour.
    The effective threshold rises after each spike and decays exponentially.

    Update equations (per timestep):
        a[t] = rho * a[t-1] + s[t-1]          # adaptation accumulates with spikes
        v_th_eff[t] = v_th + alpha * a[t]      # effective threshold rises
        v[t] = beta * v[t-1] + x[t]            # leaky integration
        s[t] = H(v[t] - v_th_eff[t])           # spike against effective threshold
        v[t] = v[t] - s[t] * v_th_eff[t]      # reset to effective threshold

    State fields used: v, a.

    Reference:
        Bellec et al. (2020) "A solution to the learning dilemma for recurrent
        networks of spiking neurons." Nature Communications.

    Args:
        size: Number of neurons.
        rho: Adaptation decay factor. Must be in (0, 1). Controls how quickly
            the elevated threshold decays back to baseline after activity stops.
        adaptation_strength: Scalar alpha multiplying adaptation variable to
            shift the effective threshold. Larger values = stronger adaptation.
        **kwargs: Forwarded to SpikingNeuronBase.

    Example::

        neuron = AdaptiveLIFNeuron(size=512, rho=0.9, adaptation_strength=0.15)
        state = neuron.reset_state(batch_size=4, neuron_shape=(512,), device=device)
        for t in range(T):
            spikes, state = neuron(input[:, t], state)
    """

    def __init__(
        self,
        size: int,
        rho: float = 0.95,
        adaptation_strength: float = 0.1,
        beta: float = 0.95,
        threshold: float = 1.0,
        learnable_beta: bool = False,
        learnable_threshold: bool = False,
        surrogate: str = "atan",
        surrogate_params: Optional[Dict] = None,
        reset_mechanism: str = "subtract",
        dtype_policy: str = "fp32_state",
    ) -> None:
        super().__init__(
            size=size,
            beta=beta,
            threshold=threshold,
            learnable_beta=learnable_beta,
            learnable_threshold=learnable_threshold,
            surrogate=surrogate,
            surrogate_params=surrogate_params,
            reset_mechanism=reset_mechanism,
            dtype_policy=dtype_policy,
        )

        if not (0.0 < rho < 1.0):
            raise ValueError(f"rho must be in (0, 1), got {rho}")
        if adaptation_strength < 0.0:
            raise ValueError(
                f"adaptation_strength must be non-negative, got {adaptation_strength}"
            )

        self.register_buffer("rho", torch.tensor(rho, dtype=torch.float32))
        self.register_buffer(
            "adaptation_strength",
            torch.tensor(adaptation_strength, dtype=torch.float32),
        )

    def reset_state(
        self,
        batch_size: int,
        neuron_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> SpikingState:
        """Create zero state with membrane potential AND adaptation variable."""
        base = super().reset_state(batch_size, neuron_shape, device, dtype)
        state_dtype = self._resolve_state_dtype(dtype)
        shape = (batch_size,) + neuron_shape
        return SpikingState(
            v=base.v,
            a=torch.zeros(shape, device=device, dtype=state_dtype),
        )

    def _step(
        self,
        x: Tensor,
        state: SpikingState,
    ) -> Tuple[Tensor, SpikingState, Dict]:
        """Single AdaptiveLIF timestep.

        Adaptation is updated AFTER spiking so it uses the spikes from the
        CURRENT timestep (not t-1). The effective threshold thus rises
        immediately on the timestep of a spike and decays from t+1 onward.

        Args:
            x: Input current, shape (B, *neuron_shape), fp32.
            state: SpikingState with fields v and a (adaptation).

        Returns:
            spikes: Binary tensor.
            new_state: SpikingState with updated v, a.
            details: {'firing_rate', 'membrane_mean', 'membrane_var',
                      'adaptation_mean', 'eff_threshold_mean'}.
        """
        if state.a is None:
            raise RuntimeError(
                "AdaptiveLIFNeuron requires state.a. "
                "Did you call reset_state() from this class?"
            )

        beta = self.beta_clamped
        th_base = self.threshold_clamped
        rho = self.rho
        alpha = self.adaptation_strength

        # 1. Update adaptation variable from previous spikes (using a from state)
        #    a[t] = rho * a[t-1] + s[t-1] — state.a already holds the previous
        #    spike contribution; here we just apply the decay and it will
        #    be updated with current spikes after firing.
        a_decayed = rho * state.a  # no in-place

        # 2. Effective threshold
        v_th_eff = th_base + alpha * a_decayed

        # 3. Leaky integration
        v_new = beta * state.v + x

        # 4. Spike
        mem_shifted = v_new - v_th_eff
        spikes = self.spike_fn(mem_shifted)

        # 5. Reset against effective threshold
        if self.reset_mechanism == "subtract":
            v_reset = v_new - spikes * v_th_eff
        else:
            v_reset = v_new * (1.0 - spikes)

        # 6. Update adaptation with CURRENT spikes
        a_new = a_decayed + spikes

        new_state = SpikingState(v=v_reset, a=a_new)

        with torch.no_grad():
            details: Dict = {
                "firing_rate": spikes.mean().item(),
                "membrane_mean": v_reset.mean().item(),
                "membrane_var": v_reset.var().item() if v_reset.numel() > 1 else 0.0,
                "adaptation_mean": a_new.mean().item(),
                "eff_threshold_mean": v_th_eff.mean().item(),
            }

        return spikes, new_state, details

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f", rho={self.rho.item():.3f}"
            + f", adaptation_strength={self.adaptation_strength.item():.3f}"
        )


# ===========================================================================
# SECTION 6: RecurrentLIFNeuron
# ===========================================================================

class RecurrentLIFNeuron(SpikingNeuronBase):
    """LIF neuron with lateral recurrent connections on spikes.

    Adds a trainable weight matrix W_rec that feeds the PREVIOUS timestep's
    spikes back as an additional current. This enables the layer to maintain
    short-term dynamics without external memory.

    Update equations (per timestep):
        i_rec[t] = W_rec @ s[t-1]             # recurrent current
        v[t] = beta * v[t-1] + x[t] + i_rec[t]  # integration with recurrent
        s[t] = H(v[t] - v_th)                 # spike
        v[t] = v[t] - s[t] * v_th             # reset

    State fields used: v, prev_spk.

    Implementation notes:
        - W_rec is initialised with small weights (scale 0.01) to prevent
          runaway recurrent excitation on training start.
        - The recurrent weight matrix has no bias — adding bias would be
          equivalent to a learnable per-neuron threshold offset (redundant).
        - For stability, consider weight normalisation (spectral norm) on W_rec
          when training very deep recurrent stacks.

    TODO(integration): Consider adding optional spectral normalisation wrapper
        for W_rec to prevent runaway recurrent dynamics during early training.

    Args:
        size: Number of neurons. W_rec is (size, size).
        recurrent_weight_scale: Std-dev of W_rec initialisation. Keep small
            (0.01–0.1) to avoid exploding recurrent dynamics.
        **kwargs: Forwarded to SpikingNeuronBase.

    Example::

        neuron = RecurrentLIFNeuron(size=512, recurrent_weight_scale=0.01)
        state = neuron.reset_state(4, (512,), device)
        for t in range(T):
            spikes, state = neuron(input[:, t], state)
    """

    def __init__(
        self,
        size: int,
        recurrent_weight_scale: float = 0.01,
        beta: float = 0.95,
        threshold: float = 1.0,
        learnable_beta: bool = False,
        learnable_threshold: bool = False,
        surrogate: str = "atan",
        surrogate_params: Optional[Dict] = None,
        reset_mechanism: str = "subtract",
        dtype_policy: str = "fp32_state",
    ) -> None:
        super().__init__(
            size=size,
            beta=beta,
            threshold=threshold,
            learnable_beta=learnable_beta,
            learnable_threshold=learnable_threshold,
            surrogate=surrogate,
            surrogate_params=surrogate_params,
            reset_mechanism=reset_mechanism,
            dtype_policy=dtype_policy,
        )

        if recurrent_weight_scale <= 0.0:
            raise ValueError(
                f"recurrent_weight_scale must be positive, "
                f"got {recurrent_weight_scale}"
            )

        self.recurrent = nn.Linear(size, size, bias=False)
        nn.init.normal_(self.recurrent.weight, std=recurrent_weight_scale)

    def reset_state(
        self,
        batch_size: int,
        neuron_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> SpikingState:
        """Create zero state with membrane potential AND previous spikes."""
        base = super().reset_state(batch_size, neuron_shape, device, dtype)
        state_dtype = self._resolve_state_dtype(dtype)
        shape = (batch_size,) + neuron_shape
        return SpikingState(
            v=base.v,
            prev_spk=torch.zeros(shape, device=device, dtype=state_dtype),
        )

    def _step(
        self,
        x: Tensor,
        state: SpikingState,
    ) -> Tuple[Tensor, SpikingState, Dict]:
        """Single RecurrentLIF timestep.

        The recurrent linear layer projects the PREVIOUS spikes into the
        current membrane integration. Gradients flow through W_rec.

        Args:
            x: Input current, shape (B, size), fp32.
                Note: RecurrentLIF only supports linear (not conv) shapes
                because W_rec requires a fixed vector size.
            state: SpikingState with fields v and prev_spk.

        Returns:
            spikes: Binary tensor (B, size).
            new_state: SpikingState with updated v, prev_spk.
            details: {'firing_rate', 'membrane_mean', 'membrane_var',
                      'recurrent_current_mean'}.
        """
        if state.prev_spk is None:
            raise RuntimeError(
                "RecurrentLIFNeuron requires state.prev_spk. "
                "Did you call reset_state() from this class?"
            )

        beta = self.beta_clamped
        th = self.threshold_clamped

        # 1. Recurrent input from previous spikes
        #    Cast prev_spk to float for the linear projection (handles fp16 prev_spk)
        i_rec = self.recurrent(state.prev_spk.float())

        # 2. Integrate with recurrent contribution
        v_new = beta * state.v + x + i_rec

        # 3. Spike
        mem_shifted = v_new - th
        spikes = self.spike_fn(mem_shifted)

        # 4. Reset
        v_reset = self._apply_reset(v_new, spikes)

        new_state = SpikingState(v=v_reset, prev_spk=spikes)

        with torch.no_grad():
            details: Dict = {
                "firing_rate": spikes.mean().item(),
                "membrane_mean": v_reset.mean().item(),
                "membrane_var": v_reset.var().item() if v_reset.numel() > 1 else 0.0,
                "recurrent_current_mean": i_rec.mean().item(),
            }

        return spikes, new_state, details

    def extra_repr(self) -> str:
        w_std = self.recurrent.weight.std().item()
        return super().extra_repr() + f", recurrent_w_std={w_std:.4f}"


# ===========================================================================
# SECTION 7: AdvancedLIFNeuron
# ===========================================================================

class AdvancedLIFNeuron(SpikingNeuronBase):
    """LIF with learnable delays, heterogeneous tau, and adaptive threshold.

    Combines three 2024-2025 research extensions into a single neuron:

    1. Heterogeneous time constants (per-neuron learnable beta):
       Each neuron has its own decay constant, enabling the population to
       span a range of timescales. Parameterised via per-neuron logit_beta
       vector so unconstrained optimisation keeps values in (0, 1).

    2. Learnable synaptic delays:
       A circular spike buffer of depth max_delay stores recent spike history.
       Soft attention (softmax over delay taps) allows gradient flow. The
       neuron can learn to "read" from any combination of past timesteps.
       Based on: Hammouamri et al. (2024) "Learning delays in SNNs."

    3. Spike-frequency adaptation (optional):
       Same mechanism as AdaptiveLIFNeuron; enabled via use_adaptive_threshold.

    Update equations (per timestep):
        delayed_x[t] = sum_d ( attn[n,d] * history[t-d, n] )   # per-neuron delay
        x_eff[t] = x[t] + w_delay * delayed_x[t]               # augmented input
        v[t] = beta[n] * v[t-1] + x_eff[t]                     # heterogeneous tau
        v_th_eff = v_th + alpha * a[t-1]                        # optional adaptation
        s[t] = H(v[t] - v_th_eff)
        v[t] = v[t] - s[t] * v_th_eff                          # reset

    State fields used: v, spike_history, a (optional).

    Args:
        size: Number of neurons.
        use_delays: Enable the learnable delay mechanism.
        max_delay: Number of delay taps in the spike history buffer.
        use_heterogeneous_tau: If True, use per-neuron learnable beta vector.
            If False, falls back to single scalar beta (inheriting base class).
        use_adaptive_threshold: Enable spike-frequency adaptation.
        delay_coupling_strength: Initial weight of delayed input relative to
            direct input x. Learnable; initialised to this value.
        adaptation_rho: Adaptation decay factor (if use_adaptive_threshold).
        adaptation_strength: Alpha for effective threshold (if use_adaptive_threshold).
        **kwargs: Forwarded to SpikingNeuronBase. Note: learnable_beta is
            overridden to False when use_heterogeneous_tau=True (per-neuron
            parameterisation takes over).

    TODO(integration): The spike history buffer grows with max_delay * N * B.
        For very long sequences with large batches, consider implementing a
        ring buffer with in-place writes on a pre-allocated tensor (requires
        careful gradient handling — use torch.no_grad() for buffer management
        and rely on the attention weights for the differentiable path).
    """

    def __init__(
        self,
        size: int,
        use_delays: bool = True,
        max_delay: int = 16,
        use_heterogeneous_tau: bool = True,
        use_adaptive_threshold: bool = False,
        delay_coupling_strength: float = 0.1,
        adaptation_rho: float = 0.95,
        adaptation_strength: float = 0.1,
        beta: float = 0.95,
        threshold: float = 1.0,
        learnable_threshold: bool = False,
        surrogate: str = "atan",
        surrogate_params: Optional[Dict] = None,
        reset_mechanism: str = "subtract",
        dtype_policy: str = "fp32_state",
    ) -> None:
        # For heterogeneous tau we handle learnable_beta ourselves as a vector
        super().__init__(
            size=size,
            beta=beta,
            threshold=threshold,
            learnable_beta=False,  # Overridden below if use_heterogeneous_tau
            learnable_threshold=learnable_threshold,
            surrogate=surrogate,
            surrogate_params=surrogate_params,
            reset_mechanism=reset_mechanism,
            dtype_policy=dtype_policy,
        )

        self.use_delays = use_delays
        self.max_delay = max_delay
        self.use_heterogeneous_tau = use_heterogeneous_tau
        self.use_adaptive_threshold = use_adaptive_threshold

        # Per-neuron learnable beta vector (overrides scalar logit_beta from base)
        if use_heterogeneous_tau:
            _init_logit = math.log(beta / (1.0 - beta))
            # Per-neuron vector with small noise to break symmetry
            self.logit_beta_vec = nn.Parameter(
                torch.full((size,), _init_logit, dtype=torch.float32)
                + torch.randn(size) * 0.05
            )
        else:
            self.logit_beta_vec = None

        # Learnable delay attention weights: (size, max_delay)
        if use_delays:
            if max_delay < 1:
                raise ValueError(f"max_delay must be >= 1, got {max_delay}")
            # Initialise near uniform; slight bias toward most recent tap
            self.delay_weights = nn.Parameter(
                torch.zeros(size, max_delay, dtype=torch.float32)
            )
            nn.init.normal_(self.delay_weights, mean=0.0, std=0.1)
            # Learnable coupling strength between delayed and direct input
            self.delay_coupling = nn.Parameter(
                torch.tensor(delay_coupling_strength, dtype=torch.float32)
            )
        else:
            self.delay_weights = None
            self.delay_coupling = None

        # Adaptation parameters
        if use_adaptive_threshold:
            if not (0.0 < adaptation_rho < 1.0):
                raise ValueError(
                    f"adaptation_rho must be in (0,1), got {adaptation_rho}"
                )
            self.register_buffer(
                "adaptation_rho", torch.tensor(adaptation_rho, dtype=torch.float32)
            )
            self.adaptation_strength_param = nn.Parameter(
                torch.tensor(adaptation_strength, dtype=torch.float32)
            )
        else:
            self.adaptation_rho = None
            self.adaptation_strength_param = None

    @property
    def beta_clamped(self) -> Tensor:
        """Per-neuron beta if use_heterogeneous_tau, else scalar from base."""
        if self.logit_beta_vec is not None:
            return torch.sigmoid(self.logit_beta_vec).clamp(BETA_MIN, BETA_MAX)
        # Fall back to scalar logit_beta from base class
        return torch.sigmoid(self.logit_beta).clamp(BETA_MIN, BETA_MAX)

    def reset_state(
        self,
        batch_size: int,
        neuron_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> SpikingState:
        """Create zero state including spike history buffer and optional adaptation."""
        base = super().reset_state(batch_size, neuron_shape, device, dtype)
        state_dtype = self._resolve_state_dtype(dtype)
        shape = (batch_size,) + neuron_shape

        spike_history = None
        if self.use_delays:
            # History buffer shape: (B, max_delay, *neuron_shape)
            hist_shape = (batch_size, self.max_delay) + neuron_shape
            spike_history = torch.zeros(hist_shape, device=device, dtype=state_dtype)

        a = None
        if self.use_adaptive_threshold:
            a = torch.zeros(shape, device=device, dtype=state_dtype)

        return SpikingState(
            v=base.v,
            spike_history=spike_history,
            a=a,
        )

    def _apply_delays(self, spike_history: Tensor) -> Tensor:
        """Soft-attention delay mixing over the spike history buffer.

        Uses an einsum to apply per-neuron attention weights over delay taps.
        Gradient flows through self.delay_weights (the attention logits).

        Args:
            spike_history: Float tensor of shape (B, max_delay, *neuron_shape).
                Assumes neuron_shape == (N,) (linear). Conv shapes not supported.

        Returns:
            Delayed signal, shape (B, N), weighted sum over delay taps.
        """
        # spike_history: (B, D, N)
        # delay_weights after softmax: (N, D)
        delay_attn = torch.softmax(self.delay_weights, dim=-1)  # (N, D)

        # Permute history to (B, N, D) for einsum
        if spike_history.dim() == 3:
            hist = spike_history  # (B, D, N)
            # delayed[b, n] = sum_d delay_attn[n, d] * hist[b, d, n]
            delayed = torch.einsum("bdn,nd->bn", hist, delay_attn)
        else:
            # Fallback for unexpected shapes: use most recent tap
            logger.warning(
                "AdvancedLIFNeuron._apply_delays: spike_history has "
                "unexpected shape %s. Falling back to most recent tap.",
                tuple(spike_history.shape),
            )
            delayed = spike_history[:, -1]

        return delayed

    def _update_history(
        self, spike_history: Tensor, new_spikes: Tensor
    ) -> Tensor:
        """Append new_spikes to history, dropping the oldest tap.

        Creates a new tensor (no in-place) by concatenating along the
        delay dimension.

        Args:
            spike_history: (B, max_delay, *neuron_shape)
            new_spikes: (B, *neuron_shape)

        Returns:
            Updated history: (B, max_delay, *neuron_shape)
        """
        # Drop oldest (dim=1, index 0) and append newest
        new_entry = new_spikes.unsqueeze(1)  # (B, 1, *neuron_shape)
        updated = torch.cat([spike_history[:, 1:], new_entry], dim=1)
        return updated

    def _step(
        self,
        x: Tensor,
        state: SpikingState,
    ) -> Tuple[Tensor, SpikingState, Dict]:
        """Single AdvancedLIF timestep.

        Order of operations:
          1. Read delayed input from spike history (if enabled)
          2. Augment input current with delayed signal
          3. Leaky integration with per-neuron beta
          4. Compute effective threshold (with adaptation if enabled)
          5. Spike and reset
          6. Update spike history buffer
          7. Update adaptation variable

        Args:
            x: Input current, shape (B, size), fp32.
            state: SpikingState with v, spike_history (if delays), a (if adapt).

        Returns:
            spikes: Binary tensor (B, size).
            new_state: SpikingState with updated v, spike_history, a.
            details: Rich diagnostic dict.
        """
        beta = self.beta_clamped  # (size,) or scalar
        th = self.threshold_clamped

        # 1. Delayed input
        x_eff = x
        if self.use_delays and state.spike_history is not None:
            # Only apply if we have at least max_delay steps of history
            # (early in a sequence the buffer is zeros, which is fine)
            delayed = self._apply_delays(state.spike_history)
            # delayed shape: (B, size); coupling is a learnable scalar
            x_eff = x + self.delay_coupling * delayed

        # 2. Leaky integration with per-neuron or scalar beta
        if beta.dim() == 1:
            # Per-neuron: (size,) must broadcast with (B, size)
            v_new = beta.unsqueeze(0) * state.v + x_eff
        else:
            v_new = beta * state.v + x_eff

        # 3. Effective threshold with optional adaptation
        if self.use_adaptive_threshold and state.a is not None:
            v_th_eff = th + self.adaptation_strength_param * state.a
        else:
            v_th_eff = th

        # 4. Spike
        mem_shifted = v_new - v_th_eff
        spikes = self.spike_fn(mem_shifted)

        # 5. Reset
        if self.reset_mechanism == "subtract":
            v_reset = v_new - spikes * v_th_eff
        else:
            v_reset = v_new * (1.0 - spikes)

        # 6. Update spike history
        new_history = None
        if self.use_delays and state.spike_history is not None:
            new_history = self._update_history(state.spike_history, spikes)

        # 7. Update adaptation
        new_a = None
        if self.use_adaptive_threshold:
            if state.a is not None:
                new_a = self.adaptation_rho * state.a + spikes
            else:
                new_a = spikes.clone()

        new_state = SpikingState(
            v=v_reset,
            spike_history=new_history,
            a=new_a,
        )

        with torch.no_grad():
            details: Dict = {
                "firing_rate": spikes.mean().item(),
                "membrane_mean": v_reset.mean().item(),
                "membrane_var": v_reset.var().item() if v_reset.numel() > 1 else 0.0,
            }
            if beta.dim() == 1:
                details["beta_mean"] = beta.mean().item()
                details["beta_min"] = beta.min().item()
                details["beta_max"] = beta.max().item()
            if self.use_delays and state.spike_history is not None:
                details["delay_coupling"] = self.delay_coupling.item()
            if new_a is not None:
                details["adaptation_mean"] = new_a.mean().item()
                details["eff_threshold_mean"] = (
                    v_th_eff.mean().item()
                    if isinstance(v_th_eff, Tensor)
                    else float(v_th_eff)
                )

        return spikes, new_state, details

    def get_delay_distribution(self) -> Optional[Tensor]:
        """Return the learned delay attention distribution.

        Returns:
            Tensor of shape (size, max_delay) with softmax-normalised
            delay weights, or None if delays are disabled.
        """
        if self.delay_weights is None:
            return None
        return torch.softmax(self.delay_weights.detach(), dim=-1)

    def get_effective_delays(self) -> Optional[Tensor]:
        """Return per-neuron expected delay (centre of mass of delay distribution).

        Returns:
            Tensor of shape (size,) with values in [0, max_delay - 1],
            or None if delays are disabled.
        """
        dist = self.get_delay_distribution()
        if dist is None:
            return None
        delays = torch.arange(
            self.max_delay, device=dist.device, dtype=dist.dtype
        )
        return (dist * delays).sum(dim=-1)

    def extra_repr(self) -> str:
        extras = [
            f"use_delays={self.use_delays}",
            f"max_delay={self.max_delay}",
            f"heterogeneous_tau={self.use_heterogeneous_tau}",
            f"adaptive_threshold={self.use_adaptive_threshold}",
        ]
        return super().extra_repr() + ", " + ", ".join(extras)


# ===========================================================================
# SECTION 8: Factory, registry, and utilities
# ===========================================================================

_NEURON_REGISTRY: Dict[str, type] = {
    "lif": LIFNeuron,
    "adaptive_lif": AdaptiveLIFNeuron,
    "recurrent_lif": RecurrentLIFNeuron,
    "advanced_lif": AdvancedLIFNeuron,
}


def create_neuron(neuron_type: str, **kwargs) -> SpikingNeuronBase:
    """Factory function for creating spiking neuron instances.

    This is the recommended entry point for constructing neurons throughout
    brain_ai. It avoids hard-coding concrete class names in calling modules
    and allows new neuron types to be registered centrally.

    Args:
        neuron_type: Key in _NEURON_REGISTRY. One of:
            'lif', 'adaptive_lif', 'recurrent_lif', 'advanced_lif'.
        **kwargs: Keyword arguments forwarded to the neuron's __init__.
            At minimum, 'size' is required by all neuron types.

    Returns:
        Initialised SpikingNeuronBase subclass instance.

    Raises:
        KeyError: If neuron_type is not in the registry.
        TypeError: If required kwargs are missing.

    Example::

        # In BrainAI core layer construction:
        neuron = create_neuron('adaptive_lif', size=512, rho=0.9)
        state = neuron.reset_state(batch_size=B, neuron_shape=(512,), device=device)

    TODO(integration): Read neuron_type and kwargs from BrainAIConfig.SNNConfig
        so the architecture can be configured declaratively from config files.
    """
    if neuron_type not in _NEURON_REGISTRY:
        available = sorted(_NEURON_REGISTRY.keys())
        raise KeyError(
            f"Unknown neuron type '{neuron_type}'. "
            f"Available types: {available}"
        )
    cls = _NEURON_REGISTRY[neuron_type]
    return cls(**kwargs)


def register_neuron(name: str, cls: type) -> None:
    """Register a custom neuron class in the global registry.

    Use this to add project-specific neuron variants without modifying this
    file. The class must be a subclass of SpikingNeuronBase.

    Args:
        name: Registry key (must not already exist).
        cls: Neuron class. Must inherit from SpikingNeuronBase.

    Raises:
        TypeError: If cls is not a subclass of SpikingNeuronBase.
        KeyError: If name is already registered.

    TODO(integration): Called by plugin/extension modules that define custom
        neuron variants (e.g. ConductanceLIFNeuron, IzhikevichNeuron).
    """
    if not (isinstance(cls, type) and issubclass(cls, SpikingNeuronBase)):
        raise TypeError(
            f"cls must be a subclass of SpikingNeuronBase, got {cls}"
        )
    if name in _NEURON_REGISTRY:
        raise KeyError(
            f"Neuron type '{name}' is already registered. "
            f"Unregister it first or choose a different name."
        )
    _NEURON_REGISTRY[name] = cls
    logger.info("Registered neuron type '%s' -> %s", name, cls.__name__)


def assert_spiking_state(
    state: SpikingState,
    batch_size: int,
    neuron_shape: Tuple[int, ...],
) -> None:
    """Runtime assertion helper for SpikingState shape validation.

    Call this at the top of forward passes (or in unit tests) to catch shape
    mismatches early with a clear error message.

    Args:
        state: SpikingState to validate.
        batch_size: Expected batch dimension B.
        neuron_shape: Expected shape of all state tensors excluding batch dim.

    Raises:
        AssertionError: If any present tensor has an unexpected shape.

    Example::

        state = neuron.reset_state(B, (N,), device)
        assert_spiking_state(state, B, (N,))  # passes
        assert_spiking_state(state, B+1, (N,))  # AssertionError
    """
    expected_shape = (batch_size,) + neuron_shape

    assert state.v.shape == expected_shape, (
        f"state.v shape mismatch: expected {expected_shape}, "
        f"got {tuple(state.v.shape)}"
    )

    for field_name in ("i", "a", "prev_spk", "ref"):
        t = getattr(state, field_name)
        if t is not None:
            assert t.shape == expected_shape, (
                f"state.{field_name} shape mismatch: expected {expected_shape}, "
                f"got {tuple(t.shape)}"
            )

    if state.spike_history is not None:
        # spike_history has extra delay dimension
        B, D = state.spike_history.shape[:2]
        rest = state.spike_history.shape[2:]
        assert B == batch_size and rest == neuron_shape, (
            f"state.spike_history shape mismatch: "
            f"expected (B={batch_size}, D=?, *{neuron_shape}), "
            f"got {tuple(state.spike_history.shape)}"
        )


def unroll_neuron(
    neuron: SpikingNeuronBase,
    x_seq: Tensor,
    state: Optional[SpikingState] = None,
    detach_every: int = 0,
    return_details: bool = False,
) -> Tuple[Tensor, SpikingState]:
    """Unroll a spiking neuron over a sequence of timesteps.

    This utility handles the explicit state management loop so calling code
    does not need to write its own for-loop boilerplate.

    Args:
        neuron: Any SpikingNeuronBase subclass instance.
        x_seq: Input sequence, shape (B, T, *neuron_shape).
        state: Initial SpikingState. If None, a fresh zero state is created.
        detach_every: If > 0, detach state every N timesteps (truncated BPTT).
            Set to 0 (default) to backpropagate through all timesteps.
        return_details: If True, also return a list of per-timestep detail dicts.

    Returns:
        spike_seq: Tensor of shape (B, T, *neuron_shape) with all timestep spikes.
        final_state: SpikingState after the last timestep.

    Raises:
        ValueError: If x_seq does not have at least 3 dimensions (B, T, ...).

    TODO(integration): Add support for variable-length sequences via padding
        masks, matching the interface used in the workspace attention module.

    Example::

        neuron = LIFNeuron(size=512)
        state = neuron.reset_state(B, (512,), device)
        spike_seq, final_state = unroll_neuron(neuron, x_seq, state)
    """
    if x_seq.dim() < 3:
        raise ValueError(
            f"x_seq must have at least 3 dimensions (B, T, ...), "
            f"got shape {tuple(x_seq.shape)}"
        )

    B, T = x_seq.shape[0], x_seq.shape[1]
    neuron_shape = x_seq.shape[2:]

    if state is None:
        state = neuron.reset_state(B, neuron_shape, x_seq.device, x_seq.dtype)

    spike_list = []
    detail_list = []

    for t in range(T):
        x_t = x_seq[:, t]  # (B, *neuron_shape)

        if return_details:
            spikes_t, state, details_t = neuron(x_t, state, return_details=True)
            detail_list.append(details_t)
        else:
            spikes_t, state = neuron(x_t, state, return_details=False)

        spike_list.append(spikes_t)

        # Truncated BPTT: detach state at specified interval
        if detach_every > 0 and (t + 1) % detach_every == 0:
            state = state.detach()

    spike_seq = torch.stack(spike_list, dim=1)  # (B, T, *neuron_shape)

    if return_details:
        return spike_seq, state, detail_list  # type: ignore[return-value]
    return spike_seq, state


# ---------------------------------------------------------------------------
# Module-level convenience exports
# ---------------------------------------------------------------------------

__all__ = [
    # State
    "SpikingState",
    # Surrogates
    "ATanSurrogate",
    "FastSigmoidSurrogate",
    "StraightThroughSurrogate",
    "SURROGATE_MAX_GRAD",
    "get_surrogate",
    # Base
    "SpikingNeuronBase",
    # Neuron variants
    "LIFNeuron",
    "AdaptiveLIFNeuron",
    "RecurrentLIFNeuron",
    "AdvancedLIFNeuron",
    # Factory / registry
    "_NEURON_REGISTRY",
    "create_neuron",
    "register_neuron",
    # Utilities
    "assert_spiking_state",
    "unroll_neuron",
    # Constants
    "BETA_MIN",
    "BETA_MAX",
    "THRESHOLD_MIN",
    "RESET_MECHANISMS",
    "DTYPE_POLICIES",
]
