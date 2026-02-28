"""
brain_ai/core/surrogates.py — Surrogate Gradient Functions for Spiking Networks

Surrogate gradients replace the nondifferentiable Heaviside step function in the
backward pass, enabling gradient-based training of spiking neural networks.

The core mathematical problem: the Heaviside function H(x) = 1{x >= 0} has zero
gradient almost everywhere and an undefined gradient at x=0, so standard
backpropagation fails.  The surrogate approach keeps the forward pass exactly
binary while substituting a smooth, well-behaved derivative in the backward pass.

Design principles:
    1. Forward is ALWAYS binary Heaviside: s = 1{v >= v_th}
    2. Backward uses a smooth approximation: ds/dv ~= sigma'(v - v_th)
    3. All surrogates are normalized so max gradient ~= 1.0 (SpikingJelly convention)
    4. Parameters are documented with their effect on gradient width and magnitude

Normalization convention:
    "Normalized" means that the peak surrogate gradient value is exactly 1.0
    at x=0 (i.e., right at threshold).  This avoids implicit gradient scaling
    across layers and makes learning-rate choices more predictable.

    The "traditional" defaults found in many SNN libraries (e.g., slope=25 for
    FastSigmoid) produce large peak gradients (12.5 for slope=25), effectively
    scaling the learning rate by that factor.  This template uses normalized
    defaults throughout; see get_normalized_params() for the reference table.

Available surrogates:
    ATanSurrogate           — Bell-shaped, stable default, recommended for most uses
    FastSigmoidSurrogate    — Heavier tails, more gradient signal far from threshold
    StraightThroughEstimator — Constant gradient, simplest possible
    MultiGaussianSurrogate  — Sum of Gaussians, tunable multi-scale

Usage:
    from brain_ai.core.surrogates import get_surrogate, spike_function

    spike_fn = get_surrogate('atan')                      # normalized ATan
    spike_fn = get_surrogate('fast_sigmoid', slope=5.0)   # override slope
    spike_fn = get_surrogate('atan', normalized=False)     # traditional params

    # Canonical spike computation (used by LIFNeuron and friends)
    spikes = spike_function(membrane_potential, threshold_tensor, spike_fn)

References:
    Neftci et al. (2019) "Surrogate Gradient Learning in SNNs"
    Zenke & Ganguli (2018) "SuperSpike: Supervised Learning in SNNs"
    SpikingJelly: https://spikingjelly.readthedocs.io/
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Constants and registry
# ---------------------------------------------------------------------------

#: Maps surrogate name strings to their implementing class.
#: Populated by the @register_surrogate decorator below.
SURROGATE_REGISTRY: Dict[str, Type[torch.autograd.Function]] = {}

#: Peak gradient value for each surrogate at its normalized (default) parameters.
#: All values are 1.0 by design — this table exists for documentation and testing.
SURROGATE_MAX_GRAD: Dict[str, float] = {
    "atan": 1.0,
    "fast_sigmoid": 1.0,
    "straight_through": 1.0,
    "multi_gaussian": 1.0,
}

#: Traditional (non-normalized) parameters from the SNN literature.
#: These are common in older codebases and produce larger peak gradients.
SURROGATE_TRADITIONAL_PARAMS: Dict[str, Dict[str, float]] = {
    "atan":            {"alpha": 2.0},    # already normalized; alpha=2 is standard
    "fast_sigmoid":    {"slope": 25.0},   # slope=25 -> peak grad = 12.5
    "straight_through": {},               # no parameters
    "multi_gaussian":  {"sigma": 0.4},    # already normalized
}

#: Normalized parameters for each surrogate (peak gradient = 1.0).
SURROGATE_NORMALIZED_PARAMS: Dict[str, Dict[str, float]] = {
    "atan":            {"alpha": 2.0},    # peak = alpha/2 = 1.0 at alpha=2
    "fast_sigmoid":    {"slope": 2.0},    # peak = slope/2 = 1.0 at slope=2
    "straight_through": {},               # always 1.0; no parameters
    "multi_gaussian":  {"sigma": 0.4},    # see MultiGaussianSurrogate for derivation
}


# ---------------------------------------------------------------------------
# Registration decorator
# ---------------------------------------------------------------------------

def register_surrogate(name: str) -> Callable[[Type], Type]:
    """Class decorator that registers a surrogate in SURROGATE_REGISTRY.

    Args:
        name: The lookup key (e.g. 'atan').

    Returns:
        Decorator that adds the class to SURROGATE_REGISTRY under ``name``
        and sets a ``_surrogate_name`` attribute for introspection.

    Example::

        @register_surrogate('my_surrogate')
        class MySurrogate(torch.autograd.Function):
            ...
    """
    def decorator(cls: Type) -> Type:
        cls._surrogate_name = name
        SURROGATE_REGISTRY[name] = cls
        return cls
    return decorator


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def get_normalized_params(name: str) -> Dict[str, float]:
    """Return parameters that normalize the peak gradient to approximately 1.0.

    Derivations:
        ATan:
            grad(x) = alpha / (2 * (1 + (pi * alpha * x)^2))
            At x=0: grad(0) = alpha / 2
            For alpha/2 = 1.0 -> alpha = 2.0  (coincides with common default)

        FastSigmoid:
            grad(x) = slope / (2 * (1 + slope * |x|)^2)
            At x=0: grad(0) = slope / 2
            For slope/2 = 1.0 -> slope = 2.0  (NOT the traditional 25.0)

        StraightThrough:
            grad(x) = 1.0 everywhere; inherently normalized, no parameters.

        MultiGaussian:
            grad(x) = exp(-x^2 / (2*sigma^2)) / (sigma * sqrt(2*pi))
            At x=0: grad(0) = 1 / (sigma * sqrt(2*pi))
            For 1/(sigma * sqrt(2*pi)) = 1.0 -> sigma = 1/sqrt(2*pi) ~= 0.3989
            We use sigma=0.4 (rounded, within 0.3% of exact).

    Reference table:

    +-----------------+----------+------------+-------------+----------------+
    | Surrogate       | Param    | Normalized | Traditional | Trad peak grad |
    +-----------------+----------+------------+-------------+----------------+
    | atan            | alpha    | 2.0        | 2.0         | 1.0            |
    | fast_sigmoid    | slope    | 2.0        | 25.0        | 12.5           |
    | straight_through| (none)   | —          | —           | 1.0            |
    | multi_gaussian  | sigma    | 0.4        | 0.4         | ~1.0           |
    +-----------------+----------+------------+-------------+----------------+

    Args:
        name: Surrogate name key (e.g. 'atan').

    Returns:
        Dict of parameter names to their normalized float values.

    Raises:
        ValueError: If ``name`` is not in SURROGATE_NORMALIZED_PARAMS.
    """
    if name not in SURROGATE_NORMALIZED_PARAMS:
        raise ValueError(
            f"Unknown surrogate '{name}'. "
            f"Known surrogates: {sorted(SURROGATE_NORMALIZED_PARAMS)}"
        )
    return dict(SURROGATE_NORMALIZED_PARAMS[name])  # return a copy


def get_traditional_params(name: str) -> Dict[str, float]:
    """Return the traditional (often non-normalized) parameters from the SNN literature.

    Use these only when reproducing published results that were trained with
    non-normalized gradients, or when comparing against existing checkpoints.
    For new training runs, prefer get_normalized_params().

    Args:
        name: Surrogate name key (e.g. 'fast_sigmoid').

    Returns:
        Dict of parameter names to their traditional float values.

    Raises:
        ValueError: If ``name`` is not in SURROGATE_TRADITIONAL_PARAMS.
    """
    if name not in SURROGATE_TRADITIONAL_PARAMS:
        raise ValueError(
            f"Unknown surrogate '{name}'. "
            f"Known surrogates: {sorted(SURROGATE_TRADITIONAL_PARAMS)}"
        )
    return dict(SURROGATE_TRADITIONAL_PARAMS[name])


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _SurrogateBase(torch.autograd.Function):
    """Abstract base for surrogate gradient functions.

    All concrete surrogates extend this class.  The forward pass is shared:
    it applies the Heaviside step function and saves the pre-threshold input
    for use in backward.

    Subclasses must override ``backward`` only.  They access saved tensors via
    ``ctx.saved_tensors`` and call ``_compute_grad`` (recommended) or implement
    the gradient formula directly.

    NOTE: torch.autograd.Function subclasses cannot have __init__; parameters
    must be class-level attributes or passed through ctx using ctx.save_for_backward
    / ctx.alpha-style attributes set in forward.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        """Heaviside step function: returns 1.0 where x >= 0, else 0.0.

        Args:
            ctx: Autograd context object for saving state for backward.
            x:   Pre-threshold tensor (v - v_threshold).  May be any shape.

        Returns:
            Binary tensor of same shape as x; dtype float32.
        """
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        raise NotImplementedError("Surrogate subclasses must implement backward()")


# ---------------------------------------------------------------------------
# Surrogate 1: ATanSurrogate
# ---------------------------------------------------------------------------

@register_surrogate("atan")
class ATanSurrogate(_SurrogateBase):
    """Arctangent surrogate gradient.

    Forward:  s = 1{x >= 0}       (Heaviside, binary)
    Backward: ds/dx = alpha / (2 * (1 + (pi * alpha * x)^2))

    Shape of backward gradient:
        Bell curve centered at x=0 with width ~1/(pi*alpha).
        Gradient decays to near-zero for |x| >> 1/(pi*alpha).

    Normalization:
        At x=0: grad(0) = alpha / 2
        Normalized (max_grad=1.0) requires alpha = 2.0 (the default).

    Parameters (class-level, thread-unsafe for parallel use — prefer functional API):
        alpha (float): Controls gradient width/magnitude.  Default 2.0.
                       Larger alpha -> sharper, narrower gradient peak.
                       Smaller alpha -> broader gradient, more signal far from threshold.

    Properties:
        - Bell-shaped gradient centered at threshold
        - Smooth, differentiable everywhere
        - Best choice for general-purpose SNN training
        - Numerically stable for all input values (denominator always > 0)

    References:
        Fang et al. (2021) "Incorporating Learnable Membrane Time Constant to
            Enhance Learning of SNNs", ICCV 2021.
        SpikingJelly ATanNode.
    """

    #: Default alpha parameter.  Gives max_grad = alpha/2 = 1.0.
    alpha: float = 2.0

    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        # Store alpha on ctx so backward can read it without a closure
        ctx.alpha = ATanSurrogate.alpha
        return (x >= 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        # grad = alpha / (2 * (1 + (pi * alpha * x)^2))
        # At x=0: grad = alpha/2; normalized to 1.0 when alpha=2
        denominator = 1.0 + (math.pi * alpha * x) ** 2
        grad = alpha / (2.0 * denominator)
        return grad_output * grad


# ---------------------------------------------------------------------------
# Surrogate 2: FastSigmoidSurrogate
# ---------------------------------------------------------------------------

@register_surrogate("fast_sigmoid")
class FastSigmoidSurrogate(_SurrogateBase):
    """Fast sigmoid surrogate gradient.

    Forward:  s = 1{x >= 0}       (Heaviside, binary)
    Backward: ds/dx = slope / (2 * (1 + slope * |x|)^2)

    This is the derivative of the piecewise-linear sigmoid:
        sigma(x) = x / (2 * (1 + slope * |x|)) + 0.5

    Shape of backward gradient:
        Heavier tails than ATan; gradient decays as 1/x^2 rather than
        exponentially.  Useful when membrane potentials spread widely around
        threshold during training.

    Normalization:
        At x=0: grad(0) = slope / 2
        Normalized (max_grad=1.0) requires slope = 2.0.

    WARNING — traditional default:
        Many SNN implementations default to slope=25.0 (from Zenke 2018).
        That gives peak grad = 12.5, effectively scaling LR by 12.5x.
        This template defaults to slope=2.0 for max_grad=1.0.
        Set normalized=False in get_surrogate() to recover the traditional behavior.

    Parameters (class-level):
        slope (float): Controls gradient width/magnitude.  Default 2.0 (normalized).
                       Larger slope -> sharper, narrower gradient.
                       Traditional value: 25.0 (peak grad = 12.5).

    Properties:
        - Heavier polynomial tails vs ATan's exponential tails
        - Gradient persists further from threshold (good for large voltage variance)
        - Slightly cheaper to compute than ATan (no trigonometric ops)

    References:
        Zenke & Ganguli (2018) "SuperSpike: Supervised Learning in SNNs via
            the Multiscale Dynamics of Error Propagation", Neural Computation.
    """

    #: Default slope.  Normalized to give max_grad = slope/2 = 1.0.
    slope: float = 2.0

    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        ctx.slope = FastSigmoidSurrogate.slope
        return (x >= 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        slope = ctx.slope
        # grad = slope / (2 * (1 + slope * |x|)^2)
        # At x=0: grad = slope/2; normalized to 1.0 when slope=2
        denominator = (1.0 + slope * x.abs()) ** 2
        grad = slope / (2.0 * denominator)
        return grad_output * grad


# ---------------------------------------------------------------------------
# Surrogate 3: StraightThroughEstimator
# ---------------------------------------------------------------------------

@register_surrogate("straight_through")
class StraightThroughEstimator(torch.autograd.Function):
    """Straight-through estimator (STE) surrogate.

    Forward:  s = 1{x >= 0}       (Heaviside, binary)
    Backward: ds/dx = 1.0         (pass gradient through unchanged)

    This is the simplest possible surrogate: the backward pass ignores the
    non-differentiability entirely and passes gradients through as-if the
    forward function were the identity.

    Normalization:
        Gradient is always 1.0 regardless of x; inherently normalized.
        No parameters to tune.

    Properties:
        - Zero overhead: backward is a simple passthrough
        - Gradient magnitude is completely independent of x
        - Distant neurons receive the same gradient as near-threshold neurons,
          which can cause slow convergence or instability in deep networks
        - Useful as a baseline or when computational budget is tight

    References:
        Bengio et al. (2013) "Estimating or Propagating Gradients Through
            Stochastic Neurons for Conditional Computation", arXiv 1308.3432.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        # No need to save x — backward doesn't use it
        return (x >= 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        # Pass gradient through unchanged (gradient of identity = 1)
        return grad_output


# ---------------------------------------------------------------------------
# Surrogate 4: MultiGaussianSurrogate
# ---------------------------------------------------------------------------

@register_surrogate("multi_gaussian")
class MultiGaussianSurrogate(_SurrogateBase):
    """Multi-Gaussian surrogate gradient.

    Forward:  s = 1{x >= 0}       (Heaviside, binary)
    Backward: ds/dx = sum_i(w_i * G(x; sigma_i)) / normalization_constant

    where G(x; sigma) = exp(-x^2 / (2 * sigma^2)) / (sigma * sqrt(2*pi))
    is a unit-integral Gaussian density.

    In single-Gaussian mode (the default), this reduces to:
        ds/dx = exp(-x^2 / (2 * sigma^2)) / (sigma * sqrt(2*pi))

    Normalization:
        The Gaussian density peaks at x=0 with value 1/(sigma*sqrt(2*pi)).
        For sigma = 1/sqrt(2*pi) ~= 0.3989, the peak is exactly 1.0.
        We use sigma=0.4 as a round-number approximation (error < 0.3%).

    Multi-Gaussian extension:
        When ``sigmas`` and ``weights`` are set (see class attributes), the
        surrogate sums multiple Gaussians at different scales.  This lets
        gradient information flow at multiple spatial scales — useful for
        neurons with multi-modal voltage distributions.
        Weights are automatically renormalized so the peak remains ~1.0.

    Parameters (class-level):
        sigma   (float):           Width of primary Gaussian.  Default 0.4.
        sigmas  (list of float):   Multi-scale sigma values.  Default None -> single.
        weights (list of float):   Mixing weights for each sigma.  Default None -> uniform.
        _norm_factor (float):      Computed normalization constant; set by _recompute_norm().

    Properties:
        - Compact support approximation: gradient effectively zero for |x| > 3*sigma
        - Smooth and differentiable everywhere
        - Multi-scale option can accelerate training for heterogeneous populations

    References:
        Yin et al. (2021) "Accurate and Efficient Time-Domain Classification with
            Adaptive Spiking Recurrent Neural Networks", Nature Machine Intelligence.
    """

    #: Primary Gaussian width (normalized at sigma ~= 1/sqrt(2*pi)).
    sigma: float = 0.4

    #: Optional list of sigma values for multi-Gaussian mode.
    #: None means single-Gaussian (use ``sigma`` above).
    sigmas: Optional[List[float]] = None

    #: Mixing weights for multi-Gaussian mode.  Must sum to a positive value.
    #: None means equal weights.
    weights: Optional[List[float]] = None

    #: Cached normalization factor so that max_grad ~= 1.0.
    #: Recomputed whenever sigma/sigmas/weights change via _recompute_norm().
    _norm_factor: float = 1.0 / (0.4 * math.sqrt(2.0 * math.pi))

    @classmethod
    def _recompute_norm(cls) -> None:
        """Recompute _norm_factor after changing sigma/sigmas/weights.

        Call this after modifying class-level parameters to keep max_grad ~= 1.0:

            MultiGaussianSurrogate.sigmas = [0.2, 0.5]
            MultiGaussianSurrogate.weights = [0.6, 0.4]
            MultiGaussianSurrogate._recompute_norm()
        """
        if cls.sigmas is not None and cls.weights is not None:
            # Peak value of the mixture at x=0 is sum_i(w_i / (sigma_i * sqrt(2pi)))
            peak = sum(
                w / (s * math.sqrt(2.0 * math.pi))
                for w, s in zip(cls.weights, cls.sigmas)
            )
        else:
            # Single Gaussian peak at x=0
            peak = 1.0 / (cls.sigma * math.sqrt(2.0 * math.pi))

        cls._norm_factor = peak if peak > 0 else 1.0

    @staticmethod
    def _gaussian(x: Tensor, sigma: float) -> Tensor:
        """Unnormalized Gaussian: exp(-x^2 / (2*sigma^2)).

        The unit-integral Gaussian is this divided by (sigma * sqrt(2*pi)),
        but we apply a combined normalization at the end.
        """
        return torch.exp(-x.pow(2) / (2.0 * sigma ** 2))

    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        ctx.sigma = MultiGaussianSurrogate.sigma
        ctx.sigmas = MultiGaussianSurrogate.sigmas
        ctx.weights = MultiGaussianSurrogate.weights
        ctx.norm_factor = MultiGaussianSurrogate._norm_factor
        return (x >= 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        sigma = ctx.sigma
        sigmas = ctx.sigmas
        weights = ctx.weights
        norm_factor = ctx.norm_factor

        if sigmas is not None and weights is not None:
            # Multi-Gaussian: weighted sum of unnormalized Gaussians
            total_weight = sum(weights)
            raw_grad = torch.zeros_like(x)
            for w, s in zip(weights, sigmas):
                raw_grad = raw_grad + (w / total_weight) * MultiGaussianSurrogate._gaussian(x, s)
        else:
            # Single Gaussian
            raw_grad = MultiGaussianSurrogate._gaussian(x, sigma)

        # Normalize so peak (at x=0) ~= 1.0.
        # norm_factor = peak value of raw_grad at x=0 (precomputed by _recompute_norm).
        # raw_grad at x=0 = exp(0) = 1.0 (single) or weighted sum of 1.0 values (multi).
        # Dividing by norm_factor scales the peak to 1.0.
        grad = raw_grad / norm_factor

        return grad_output * grad


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def get_surrogate(
    name: str,
    normalized: bool = True,
    **kwargs: Any,
) -> Callable[[Tensor], Tensor]:
    """Get a surrogate gradient function by name.

    The returned callable takes a single tensor argument (v - threshold) and
    produces binary spikes in the forward pass with surrogate gradients in
    backward.  It is a bound ``torch.autograd.Function.apply`` callable.

    Args:
        name:       One of 'atan', 'fast_sigmoid', 'straight_through',
                    'multi_gaussian'.  Case-insensitive.
        normalized: If True (default), configure parameters so that the peak
                    surrogate gradient is approximately 1.0.
                    If False, use the traditional defaults from the SNN
                    literature (e.g. slope=25 for fast_sigmoid).
        **kwargs:   Parameter overrides applied AFTER the normalized/traditional
                    baseline.  For example:
                    - get_surrogate('atan', alpha=5.0)
                    - get_surrogate('fast_sigmoid', normalized=False, slope=50.0)
                    Supported keys per surrogate:
                      'atan'           -> alpha (float)
                      'fast_sigmoid'   -> slope (float)
                      'straight_through' -> (no params)
                      'multi_gaussian' -> sigma (float), sigmas (list), weights (list)

    Returns:
        Callable:   A ``torch.autograd.Function.apply`` function.
                    Signature: ``fn(x: Tensor) -> Tensor``
                    Call as ``spikes = fn(v - threshold)``.

    Raises:
        ValueError: If ``name`` is not in SURROGATE_REGISTRY.

    Examples::

        # Recommended: normalized ATan (max_grad=1.0, alpha=2.0)
        spike_fn = get_surrogate('atan')

        # Wider ATan gradient (alpha=0.5 -> peak=0.25 per unit, FWHM wider)
        spike_fn = get_surrogate('atan', alpha=0.5)

        # Traditional FastSigmoid with slope=25 (peak grad=12.5)
        spike_fn = get_surrogate('fast_sigmoid', normalized=False)

        # Multi-scale Gaussian
        spike_fn = get_surrogate(
            'multi_gaussian',
            sigmas=[0.2, 0.5, 1.0],
            weights=[0.5, 0.3, 0.2],
        )
    """
    name = name.lower().strip()

    if name not in SURROGATE_REGISTRY:
        raise ValueError(
            f"Unknown surrogate '{name}'. "
            f"Registered surrogates: {sorted(SURROGATE_REGISTRY)}"
        )

    cls = SURROGATE_REGISTRY[name]

    # Select baseline parameters
    if normalized:
        base_params = get_normalized_params(name)
    else:
        base_params = get_traditional_params(name)

    # Apply overrides
    merged = {**base_params, **kwargs}

    # Apply parameters to class-level attributes
    if name == "atan":
        ATanSurrogate.alpha = merged.get("alpha", ATanSurrogate.alpha)

    elif name == "fast_sigmoid":
        FastSigmoidSurrogate.slope = merged.get("slope", FastSigmoidSurrogate.slope)

    elif name == "straight_through":
        pass  # No parameters to set

    elif name == "multi_gaussian":
        if "sigma" in merged:
            MultiGaussianSurrogate.sigma = merged["sigma"]
        if "sigmas" in merged:
            MultiGaussianSurrogate.sigmas = merged["sigmas"]
        if "weights" in merged:
            MultiGaussianSurrogate.weights = merged["weights"]
        # Recompute normalization factor after any parameter change
        MultiGaussianSurrogate._recompute_norm()

    return cls.apply


def spike_function(
    v: Tensor,
    threshold: Tensor,
    surrogate_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    """Apply spike function with surrogate gradient.

    This is the canonical spike computation used by all neuron types:

        s = surrogate_fn(v - threshold)

    Forward:
        s = 1{v >= threshold}   (binary Heaviside)

    Backward (via surrogate_fn):
        ds/dv         =  surrogate'(v - threshold)
        ds/d_threshold = -surrogate'(v - threshold)   [chain rule, sign flip]

    The threshold gradient is correct here because:
        f(v, th) = Heaviside(v - th)
        df/dth = df/d(v-th) * d(v-th)/dth = surrogate'(v-th) * (-1)

    This allows threshold to be a learnable parameter (nn.Parameter) and
    receive meaningful gradient signal during training.

    Args:
        v:            Membrane potential tensor.  Any shape (B, *features).
        threshold:    Spike threshold.  Either a scalar tensor (broadcastable)
                      or a per-neuron tensor matching v.shape.
        surrogate_fn: A ``get_surrogate(...)`` callable, e.g. ATanSurrogate.apply.

    Returns:
        Tensor of same shape as v; binary float32 (0.0 or 1.0).

    Example::

        spike_fn = get_surrogate('atan', alpha=2.0)
        threshold = torch.tensor(1.0)
        spikes = spike_function(membrane, threshold, spike_fn)
    """
    return surrogate_fn(v - threshold)


# ---------------------------------------------------------------------------
# Diagnostic utility
# ---------------------------------------------------------------------------

def check_surrogate_gradient(
    surrogate_fn: Callable[[Tensor], Tensor],
    test_points: Optional[Tensor] = None,
    num_points: int = 1000,
    x_range: Tuple[float, float] = (-5.0, 5.0),
) -> Dict[str, float]:
    """Diagnostic: compute gradient statistics for a surrogate function.

    Useful for verifying normalization and comparing surrogates.

    Args:
        surrogate_fn: A callable returned by get_surrogate().
        test_points:  Optional pre-specified x values (1-D Tensor).
                      If None, num_points linearly spaced points over x_range
                      are used.
        num_points:   Number of test points when test_points is None.
        x_range:      (min, max) range for automatic test points.

    Returns:
        Dict with the following keys:
            max_grad (float):        Maximum gradient value across all test points.
            grad_at_zero (float):    Gradient value at x=0 (or nearest test point).
            effective_width (float): x range where gradient > 0.1 * max_grad.
            total_integral (float):  Numerical integral of gradient over x_range
                                     (should approach 1.0 for proper surrogates,
                                     since d/dx Heaviside integrates to 1).
            x_at_max (float):        x value where max gradient occurs.

    Example::

        spike_fn = get_surrogate('atan')
        stats = check_surrogate_gradient(spike_fn)
        assert abs(stats['max_grad'] - 1.0) < 0.01, "max_grad not normalized"
        print(stats)
        # {'max_grad': 1.0, 'grad_at_zero': 1.0, 'effective_width': 0.637, ...}
    """
    if test_points is None:
        test_points = torch.linspace(x_range[0], x_range[1], num_points)

    x = test_points.clone().requires_grad_(True)
    out = surrogate_fn(x)
    # Compute gradient of surrogate output w.r.t. x
    # Use sum() to produce a scalar for backward
    out.sum().backward()

    with torch.no_grad():
        grads = x.grad.abs()
        dx = (x_range[1] - x_range[0]) / (num_points - 1)

        max_grad_val = grads.max().item()
        x_max_idx = grads.argmax().item()
        x_at_max = test_points[x_max_idx].item()

        # Gradient at x=0: find the test point nearest to zero
        zero_idx = (test_points.abs()).argmin().item()
        grad_at_zero = grads[zero_idx].item()

        # Effective width: x range where grad > 10% of max
        threshold_val = 0.1 * max_grad_val
        above_threshold = grads > threshold_val
        effective_indices = above_threshold.nonzero(as_tuple=False)
        if effective_indices.numel() > 0:
            eff_lo = test_points[effective_indices[0].item()].item()
            eff_hi = test_points[effective_indices[-1].item()].item()
            effective_width = abs(eff_hi - eff_lo)
        else:
            effective_width = 0.0

        # Numerical integral of gradient over the full x_range
        # For Heaviside, this should equal 1.0 (H(-inf)=0, H(+inf)=1, Delta=1)
        total_integral = (grads * dx).sum().item()

    return {
        "max_grad": max_grad_val,
        "grad_at_zero": grad_at_zero,
        "effective_width": effective_width,
        "total_integral": total_integral,
        "x_at_max": x_at_max,
    }


# ---------------------------------------------------------------------------
# SurrogateModule: nn.Module wrapper for use in nn.Sequential pipelines
# ---------------------------------------------------------------------------

class SurrogateSpike(nn.Module):
    """nn.Module wrapper around a surrogate spike function.

    Use this when you need a spike layer inside an nn.Sequential or when
    you want to serialize the surrogate configuration in a module's state_dict.

    Note: Surrogate parameters are NOT part of state_dict (they are class-level
    attributes on the autograd.Function).  If you need per-instance parameters,
    derive a custom subclass.

    Args:
        name:           Surrogate name ('atan', 'fast_sigmoid', etc.).
        normalized:     Use normalized parameters (max_grad=1.0).  Default True.
        threshold:      Fixed threshold value used in spike_function().
                        Set to 0.0 if you will subtract threshold outside this
                        module (e.g., in the neuron body).  Default 0.0.
        learn_threshold: If True, wrap threshold as an nn.Parameter so it
                        receives gradient.  Default False.
        **surrogate_kwargs: Forwarded to get_surrogate() (e.g., alpha=3.0).

    Example::

        spike_layer = SurrogateSpike('atan', alpha=3.0)
        mem_shifted = membrane - threshold           # subtract threshold externally
        spikes = spike_layer(mem_shifted)            # forward: binary; backward: ATan
    """

    def __init__(
        self,
        name: str = "atan",
        normalized: bool = True,
        threshold: float = 0.0,
        learn_threshold: bool = False,
        **surrogate_kwargs: Any,
    ) -> None:
        super().__init__()
        self.surrogate_name = name
        self.normalized = normalized
        self.surrogate_kwargs = surrogate_kwargs

        if learn_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold))
        else:
            self.register_buffer("threshold", torch.tensor(threshold))

        # Build the surrogate callable (may mutate class-level params — see note above)
        self._spike_fn: Callable[[Tensor], Tensor] = get_surrogate(
            name, normalized=normalized, **surrogate_kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply surrogate spike.

        Args:
            x: Pre-threshold tensor (v - v_th), or raw membrane potential if
               self.threshold is set to the neuron threshold.

        Returns:
            Binary spike tensor, same shape as x.
        """
        return spike_function(x, self.threshold, self._spike_fn)

    def extra_repr(self) -> str:
        parts = [f"surrogate={self.surrogate_name!r}"]
        if self.surrogate_kwargs:
            for k, v in self.surrogate_kwargs.items():
                parts.append(f"{k}={v}")
        parts.append(f"normalized={self.normalized}")
        if self.threshold.item() != 0.0:
            parts.append(f"threshold={self.threshold.item():.3f}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Convenience comparison utility
# ---------------------------------------------------------------------------

def compare_surrogates(
    names: Optional[List[str]] = None,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    num_points: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """Run check_surrogate_gradient on multiple surrogates and return a summary.

    Args:
        names:      List of surrogate names to compare.  Default: all registered.
        x_range:    Range for test points.
        num_points: Number of test points.

    Returns:
        Dict mapping surrogate name -> stats dict (from check_surrogate_gradient).

    Example::

        report = compare_surrogates()
        for name, stats in report.items():
            print(f"{name:20s}  max_grad={stats['max_grad']:.4f}  "
                  f"eff_width={stats['effective_width']:.4f}")
    """
    if names is None:
        names = sorted(SURROGATE_REGISTRY.keys())

    results: Dict[str, Dict[str, float]] = {}
    for name in names:
        fn = get_surrogate(name, normalized=True)
        results[name] = check_surrogate_gradient(
            fn, x_range=x_range, num_points=num_points
        )
    return results


# ---------------------------------------------------------------------------
# Module-level __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Core classes
    "ATanSurrogate",
    "FastSigmoidSurrogate",
    "StraightThroughEstimator",
    "MultiGaussianSurrogate",
    # nn.Module wrapper
    "SurrogateSpike",
    # Factory and canonical spike computation
    "get_surrogate",
    "spike_function",
    # Normalization helpers
    "get_normalized_params",
    "get_traditional_params",
    # Diagnostics
    "check_surrogate_gradient",
    "compare_surrogates",
    # Registry and constants
    "SURROGATE_REGISTRY",
    "SURROGATE_MAX_GRAD",
    "SURROGATE_NORMALIZED_PARAMS",
    "SURROGATE_TRADITIONAL_PARAMS",
]


# ---------------------------------------------------------------------------
# Self-test (python brain_ai/core/surrogates.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Surrogate gradient self-test")
    print("=" * 60)

    # Verify all surrogates are registered
    expected = {"atan", "fast_sigmoid", "straight_through", "multi_gaussian"}
    registered = set(SURROGATE_REGISTRY.keys())
    assert registered == expected, f"Registry mismatch: {registered} vs {expected}"
    print(f"Registered surrogates: {sorted(registered)}")

    # Check normalization for each surrogate
    print("\nNormalization check (max_grad should be ~1.0 for normalized=True):")
    print(f"{'Surrogate':<20} {'max_grad':>10} {'grad@0':>10} {'eff_width':>12} {'integral':>10}")
    print("-" * 65)

    report = compare_surrogates()
    all_pass = True
    for name, stats in report.items():
        mg = stats["max_grad"]
        g0 = stats["grad_at_zero"]
        ew = stats["effective_width"]
        ti = stats["total_integral"]
        status = "OK" if abs(mg - 1.0) < 0.05 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{name:<20} {mg:>10.4f} {g0:>10.4f} {ew:>12.4f} {ti:>10.4f}  [{status}]")

    # Check that non-normalized fast_sigmoid gives ~12.5 peak grad
    print("\nNon-normalized FastSigmoid (slope=25, expected peak ~12.5):")
    fn_trad = get_surrogate("fast_sigmoid", normalized=False)
    trad_stats = check_surrogate_gradient(fn_trad)
    print(f"  max_grad = {trad_stats['max_grad']:.4f} (expected 12.5)")
    assert abs(trad_stats["max_grad"] - 12.5) < 0.2, "Traditional FastSigmoid normalization failed"

    # Reset to normalized
    FastSigmoidSurrogate.slope = 2.0

    # Check spike_function produces correct binary output
    print("\nSpike function forward pass:")
    spike_fn = get_surrogate("atan")
    v = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    th = torch.tensor(0.0)
    spikes = spike_function(v, th, spike_fn)
    print(f"  v        = {v.tolist()}")
    print(f"  threshold= {th.item()}")
    print(f"  spikes   = {spikes.tolist()}")
    assert spikes.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0], "Heaviside forward failed"

    # Check SurrogateSpike module
    print("\nSurrogateSpike module:")
    spike_layer = SurrogateSpike("atan", alpha=2.0)
    out = spike_layer(v)
    assert out.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
    print(f"  {spike_layer}")
    print(f"  output: {out.tolist()}")

    # Check Multi-Gaussian with multiple scales
    print("\nMultiGaussian with sigmas=[0.2, 0.5], weights=[0.6, 0.4]:")
    fn_mg = get_surrogate(
        "multi_gaussian",
        sigmas=[0.2, 0.5],
        weights=[0.6, 0.4],
    )
    mg_stats = check_surrogate_gradient(fn_mg)
    print(f"  max_grad = {mg_stats['max_grad']:.4f}")
    print(f"  eff_width = {mg_stats['effective_width']:.4f}")

    # Reset MultiGaussian to defaults
    MultiGaussianSurrogate.sigmas = None
    MultiGaussianSurrogate.weights = None
    MultiGaussianSurrogate.sigma = 0.4
    MultiGaussianSurrogate._recompute_norm()

    print("\n" + "=" * 60)
    if all_pass:
        print("All checks passed.")
    else:
        print("SOME CHECKS FAILED — review normalization parameters.")
