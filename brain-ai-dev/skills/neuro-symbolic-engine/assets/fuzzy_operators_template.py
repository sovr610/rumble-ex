"""
Fuzzy Logic Operators for Neuro-Symbolic Reasoning
===================================================

Implements configurable t-norm / t-conorm families, negation, implication,
quantifier aggregators, and operator bundles aligned with Logic Tensor
Networks (LTN) / Real Logic semantics.

All operator functions are **pure** (no side effects, no state mutation) and
operate on ``torch.Tensor`` values clamped to [0, 1].

Operator families:
    * **Godel**          -- min / max (crisp-ish, sparse gradients)
    * **Product**        -- x*y / x+y-xy  (mathematically clean, vanishing near 0)
    * **Lukasiewicz**    -- max(0,x+y-1) / min(1,x+y)  (piecewise-linear)
    * **Stable Product** -- product with eps-projections (training-safe, default)

Quantifiers use generalized-mean (pMeanError) aggregators with configurable
exponent ``p`` and temperature, supporting variable-length masking.

Usage::

    from brain_ai.reasoning.fuzzy_operators import get_operator_bundle
    bundle = get_operator_bundle(OperatorConfig(bundle="stable_product"))
    result = bundle.and_op(x, y)
    forall_val = bundle.forall(truth_values, mask=mask)

References:
    * Badreddine et al., "Logic Tensor Networks", Artif. Intell. 303, 2022.
    * van Krieken et al., "Analyzing Differentiable Fuzzy Logic Operators",
      Artif. Intell. 302, 2022.
    * Marra et al., "From Statistical Relational to Neuro-Symbolic AI", 2024.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# AMP compatibility helper
# ---------------------------------------------------------------------------

def _amp_fp32(fn: Callable) -> Callable:
    """Decorator: ensure function runs in fp32 even under AMP autocasting.

    Uses ``torch.cuda.amp.custom_fwd`` when available, otherwise falls back
    to a manual ``torch.cuda.amp.autocast(enabled=False)`` wrapper.

    All fuzzy operator functions should be wrapped with this to prevent
    fp16 precision issues that corrupt truth-value semantics.
    """
    try:
        # PyTorch >= 2.0 preferred path
        return torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)(fn)
    except Exception:
        # Fallback for CPU-only or older PyTorch
        from functools import wraps

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Attempt to disable autocast if active
            try:
                with torch.cuda.amp.autocast(enabled=False):
                    # Cast any half-precision tensor args to float32
                    args_fp32 = tuple(
                        a.float() if isinstance(a, Tensor) and a.dtype == torch.float16 else a
                        for a in args
                    )
                    kwargs_fp32 = {
                        k: (v.float() if isinstance(v, Tensor) and v.dtype == torch.float16 else v)
                        for k, v in kwargs.items()
                    }
                    return fn(*args_fp32, **kwargs_fp32)
            except Exception:
                return fn(*args, **kwargs)

        return wrapper


# ============================================================================
# 1. T-Norms  (AND)
# ============================================================================

@_amp_fp32
def godel_and(x: Tensor, y: Tensor) -> Tensor:
    r"""Godel t-norm (minimum).

    .. math:: T_G(x, y) = \min(x, y)

    Identities:
        * :math:`T_G(x, 1) = x`
        * :math:`T_G(x, 0) = 0`

    Gradients are sparse (only one argument receives gradient per element).
    """
    return torch.min(x, y)


@_amp_fp32
def product_and(x: Tensor, y: Tensor) -> Tensor:
    r"""Product t-norm.

    .. math:: T_P(x, y) = x \cdot y

    Identities:
        * :math:`T_P(x, 1) = x`
        * :math:`T_P(x, 0) = 0`

    Warning: gradients vanish when either operand is near 0.
    """
    return x * y


@_amp_fp32
def lukasiewicz_and(x: Tensor, y: Tensor) -> Tensor:
    r"""Lukasiewicz t-norm.

    .. math:: T_L(x, y) = \max(0, x + y - 1)

    Identities:
        * :math:`T_L(x, 1) = x`
        * :math:`T_L(x, 0) = 0`

    Piecewise linear; produces sparse activations.
    """
    return torch.clamp(x + y - 1.0, min=0.0)


# ============================================================================
# 2. T-Conorms  (OR)
# ============================================================================

@_amp_fp32
def godel_or(x: Tensor, y: Tensor) -> Tensor:
    r"""Godel t-conorm (maximum).

    .. math:: S_G(x, y) = \max(x, y)

    Identities:
        * :math:`S_G(x, 0) = x`
        * :math:`S_G(x, 1) = 1`
    """
    return torch.max(x, y)


@_amp_fp32
def product_or(x: Tensor, y: Tensor) -> Tensor:
    r"""Product t-conorm (probabilistic sum).

    .. math:: S_P(x, y) = x + y - x \cdot y

    Identities:
        * :math:`S_P(x, 0) = x`
        * :math:`S_P(x, 1) = 1`
    """
    return x + y - x * y


@_amp_fp32
def lukasiewicz_or(x: Tensor, y: Tensor) -> Tensor:
    r"""Lukasiewicz t-conorm (bounded sum).

    .. math:: S_L(x, y) = \min(1, x + y)

    Identities:
        * :math:`S_L(x, 0) = x`
        * :math:`S_L(x, 1) = 1`
    """
    return torch.clamp(x + y, max=1.0)


# ============================================================================
# 3. Negation
# ============================================================================

@_amp_fp32
def standard_negation(x: Tensor) -> Tensor:
    r"""Standard fuzzy negation (involutive).

    .. math:: N(x) = 1 - x

    Property: :math:`N(N(x)) = x` (involution).
    """
    return 1.0 - x


# ============================================================================
# 4. Implication Operators
# ============================================================================

@_amp_fp32
def reichenbach_implies(x: Tensor, y: Tensor) -> Tensor:
    r"""Reichenbach S-implication (associated with product t-norm).

    .. math:: I_R(x, y) = 1 - x + x \cdot y

    Boundaries:
        * :math:`I_R(1, y) = y`
        * :math:`I_R(0, y) = 1`
        * :math:`I_R(x, 1) = 1`

    This is the S-implication derived from the product t-conorm and
    standard negation: :math:`S(N(x), y)`.
    """
    return 1.0 - x + x * y


@_amp_fp32
def godel_implies(x: Tensor, y: Tensor) -> Tensor:
    r"""Godel residuated implication.

    .. math::
        I_G(x, y) = \begin{cases}
            1   & \text{if } x \leq y \\
            y   & \text{otherwise}
        \end{cases}

    Boundaries:
        * :math:`I_G(1, y) = y`
        * :math:`I_G(0, y) = 1`

    This is the residuum of the Godel t-norm.
    """
    return torch.where(x <= y, torch.ones_like(x), y)


@_amp_fp32
def goguen_implies(x: Tensor, y: Tensor, eps: float = 1e-7) -> Tensor:
    r"""Goguen residuated implication (product residuum).

    .. math::
        I_{Go}(x, y) = \begin{cases}
            1         & \text{if } x \leq y \\
            y / x     & \text{otherwise}
        \end{cases}

    Uses ``eps`` for numerical stability in the division.

    Boundaries:
        * :math:`I_{Go}(1, y) = y`
        * :math:`I_{Go}(0, y) = 1`
    """
    safe_x = x + eps  # avoid division by zero
    ratio = y / safe_x
    return torch.where(x <= y, torch.ones_like(x), torch.clamp(ratio, max=1.0))


@_amp_fp32
def lukasiewicz_implies(x: Tensor, y: Tensor) -> Tensor:
    r"""Lukasiewicz residuated implication.

    .. math:: I_L(x, y) = \min(1, 1 - x + y)

    Boundaries:
        * :math:`I_L(1, y) = y`
        * :math:`I_L(0, y) = 1`

    This is the residuum of the Lukasiewicz t-norm.
    """
    return torch.clamp(1.0 - x + y, max=1.0)


# ============================================================================
# 5. Stable Product Operators  (eps-projected)
# ============================================================================

@_amp_fp32
def pi_0(x: Tensor, eps: float = 1e-4) -> Tensor:
    r"""Project to :math:`[\varepsilon, 1]` -- avoid exact zeros.

    Used to prevent vanishing gradients and zero products in
    the stable product family.

    .. math:: \pi_0(x) = \operatorname{clamp}(x, \min=\varepsilon)
    """
    return torch.clamp(x, min=eps)


@_amp_fp32
def pi_1(x: Tensor, eps: float = 1e-4) -> Tensor:
    r"""Project to :math:`[0, 1 - \varepsilon]` -- avoid exact ones.

    Used to prevent saturation in the stable product t-conorm.

    .. math:: \pi_1(x) = \operatorname{clamp}(x, \max=1 - \varepsilon)
    """
    return torch.clamp(x, max=1.0 - eps)


@_amp_fp32
def stable_product_and(x: Tensor, y: Tensor, eps: float = 1e-4) -> Tensor:
    r"""Stable product t-norm with eps projection.

    .. math:: T_{SP}(x, y) = \pi_0(x) \cdot \pi_0(y)

    **Semantic caveat**: this is NOT a true t-norm because the identity
    element property :math:`T(a, 1) = a` holds only *approximately*
    (within ``eps``).  The projection ensures that gradients remain
    non-zero even when operands are at boundary values, making this
    the recommended operator for gradient-based training.

    Approximate identity:
        * :math:`T_{SP}(a, 1) \approx a` (exact for :math:`a \geq \varepsilon`)
        * :math:`T_{SP}(a, 0) = \varepsilon \cdot \pi_0(a)` (not exactly 0)
    """
    return pi_0(x, eps) * pi_0(y, eps)


@_amp_fp32
def stable_product_or(x: Tensor, y: Tensor, eps: float = 1e-4) -> Tensor:
    r"""Stable product t-conorm with eps projection.

    .. math:: S_{SP}(x, y) = 1 - (1 - \pi_1(x)) \cdot (1 - \pi_1(y))

    **Semantic caveat**: approximate co-identity -- :math:`S_{SP}(a, 0) \approx a`
    within ``eps``.
    """
    return 1.0 - (1.0 - pi_1(x, eps)) * (1.0 - pi_1(y, eps))


@_amp_fp32
def stable_product_not(x: Tensor) -> Tensor:
    r"""Standard negation (same as ``standard_negation``).

    Included for bundle completeness; the negation is the same across
    all families.

    .. math:: N(x) = 1 - x
    """
    return 1.0 - x


@_amp_fp32
def stable_product_implies(x: Tensor, y: Tensor, eps: float = 1e-4) -> Tensor:
    r"""Stable Reichenbach S-implication derived from stable product ops.

    .. math::
        I_{SP}(x, y) = S_{SP}(N(x), T_{SP}(x, y))

    This is the S-implication constructed from the stable product
    t-conorm, standard negation, and stable product t-norm:

    .. math::
        I_{SP}(x, y) = \text{stable\_product\_or}(1 - x,\;
                        \text{stable\_product\_and}(x, y))

    The construction ensures gradient flow at boundaries.

    **Alternative (direct Reichenbach form)**:
        ``1 - pi_0(x, eps) + pi_0(x, eps) * pi_0(y, eps)``
    which is numerically equivalent for most practical cases.
    """
    neg_x = stable_product_not(x)
    xy = stable_product_and(x, y, eps)
    return stable_product_or(neg_x, xy, eps)


# ============================================================================
# 6. Quantifier Aggregators
# ============================================================================

class ForallAggregator(nn.Module):
    r"""Generalized-mean-based universal quantifier (soft min).

    Implements the **pMeanError** formulation from LTN:

    .. math::
        \text{Forall}(v) = 1 - \left(
            \frac{1}{N} \sum_{i=1}^{N} (1 - v_i)^p
        \right)^{1/p}

    As :math:`p \to \infty` this approaches :math:`\min(v)`.
    Larger ``p`` penalizes individual violations more strongly.

    Parameters
    ----------
    p : float
        Generalized mean exponent (default 2.0).
        Higher values make the aggregator stricter (closer to hard min).
    temperature : float
        Scaling factor applied to truth values before aggregation
        (default 1.0).  Values > 1 sharpen, < 1 soften.
    stable : bool
        If True, use log-sum-exp for numerical stability with large ``p``
        (default True).
    """

    def __init__(
        self,
        p: float = 2.0,
        temperature: float = 1.0,
        stable: bool = True,
    ) -> None:
        super().__init__()
        self.p = p
        self.temperature = temperature
        self.stable = stable

    def forward(
        self,
        truth_values: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Aggregate truth values with universal semantics.

        Parameters
        ----------
        truth_values : Tensor
            Shape ``(B, N)`` -- truth values for ``N`` instances per batch item.
        mask : Tensor, optional
            Shape ``(B, N)`` -- binary mask; 1 = valid, 0 = padding.
            If ``None`` all positions are treated as valid.

        Returns
        -------
        Tensor
            Shape ``(B,)`` -- aggregated truth per batch item.
        """
        # Apply temperature scaling
        tv = truth_values * self.temperature

        # Errors: (1 - truth)
        errors = torch.clamp(1.0 - tv, min=0.0, max=1.0)

        p = self.p

        if mask is not None:
            # Zero out padding positions so they do not contribute
            errors = errors * mask.float()
            # Count of valid positions per batch item
            counts = mask.float().sum(dim=-1).clamp(min=1.0)
        else:
            counts = torch.tensor(
                truth_values.shape[-1], dtype=truth_values.dtype, device=truth_values.device
            )

        if self.stable and p > 4.0:
            # Log-sum-exp path for large p (avoids overflow in x^p)
            # (mean(e^p))^(1/p) = exp( (1/p) * log(mean(e^p)) )
            log_errors = torch.log(errors.clamp(min=1e-20))
            scaled = p * log_errors
            if mask is not None:
                scaled = scaled.masked_fill(~mask.bool(), float('-inf'))
            # log-mean-exp
            max_scaled = scaled.max(dim=-1, keepdim=True).values.clamp(min=-40.0)
            exp_shifted = torch.exp(scaled - max_scaled)
            if mask is not None:
                exp_shifted = exp_shifted * mask.float()
            log_mean = max_scaled.squeeze(-1) + torch.log(
                (exp_shifted.sum(dim=-1) / counts).clamp(min=1e-20)
            )
            pmean_error = torch.exp(log_mean / p)
        else:
            # Direct computation path
            powered = errors.pow(p)
            pmean_error = (powered.sum(dim=-1) / counts).pow(1.0 / p)

        result = 1.0 - pmean_error
        return torch.clamp(result, min=0.0, max=1.0)

    def extra_repr(self) -> str:
        return f"p={self.p}, temperature={self.temperature}, stable={self.stable}"


class ExistsAggregator(nn.Module):
    r"""Generalized-mean-based existential quantifier (soft max).

    Implements:

    .. math::
        \text{Exists}(v) = \left(
            \frac{1}{N} \sum_{i=1}^{N} v_i^p
        \right)^{1/p}

    As :math:`p \to \infty` this approaches :math:`\max(v)`.
    Larger ``p`` makes the aggregator more optimistic (closer to hard max).

    Parameters
    ----------
    p : float
        Generalized mean exponent (default 2.0).
    temperature : float
        Scaling factor applied to truth values before aggregation.
    stable : bool
        If True, use log-sum-exp for numerical stability with large ``p``.
    """

    def __init__(
        self,
        p: float = 2.0,
        temperature: float = 1.0,
        stable: bool = True,
    ) -> None:
        super().__init__()
        self.p = p
        self.temperature = temperature
        self.stable = stable

    def forward(
        self,
        truth_values: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Aggregate truth values with existential semantics.

        Parameters
        ----------
        truth_values : Tensor
            Shape ``(B, N)`` -- truth values for ``N`` instances per batch item.
        mask : Tensor, optional
            Shape ``(B, N)`` -- binary mask; 1 = valid, 0 = padding.

        Returns
        -------
        Tensor
            Shape ``(B,)`` -- aggregated truth per batch item.
        """
        tv = truth_values * self.temperature
        tv = torch.clamp(tv, min=0.0, max=1.0)

        p = self.p

        if mask is not None:
            tv_masked = tv * mask.float()
            counts = mask.float().sum(dim=-1).clamp(min=1.0)
        else:
            tv_masked = tv
            counts = torch.tensor(
                truth_values.shape[-1], dtype=truth_values.dtype, device=truth_values.device
            )

        if self.stable and p > 4.0:
            # Log-sum-exp path
            log_tv = torch.log(tv_masked.clamp(min=1e-20))
            scaled = p * log_tv
            if mask is not None:
                scaled = scaled.masked_fill(~mask.bool(), float('-inf'))
            max_scaled = scaled.max(dim=-1, keepdim=True).values.clamp(min=-40.0)
            exp_shifted = torch.exp(scaled - max_scaled)
            if mask is not None:
                exp_shifted = exp_shifted * mask.float()
            log_mean = max_scaled.squeeze(-1) + torch.log(
                (exp_shifted.sum(dim=-1) / counts).clamp(min=1e-20)
            )
            pmean = torch.exp(log_mean / p)
        else:
            powered = tv_masked.pow(p)
            pmean = (powered.sum(dim=-1) / counts).pow(1.0 / p)

        return torch.clamp(pmean, min=0.0, max=1.0)

    def extra_repr(self) -> str:
        return f"p={self.p}, temperature={self.temperature}, stable={self.stable}"


# ============================================================================
# 7. OperatorConfig dataclass
# ============================================================================

@dataclass
class OperatorConfig:
    """Configuration for selecting and parameterising an operator bundle.

    Attributes
    ----------
    bundle : str
        One of ``"godel"``, ``"product"``, ``"lukasiewicz"``,
        ``"stable_product"``.
    eps : float
        Stability epsilon for stable_product projections.
    quantifier_p : float
        Generalized-mean exponent for quantifier aggregators.
    quantifier_temp : float
        Temperature scaling for quantifier aggregators.
    implication_type : str
        Which implication to use within the bundle.
        Options: ``"reichenbach"``, ``"godel"``, ``"goguen"``,
        ``"lukasiewicz"``.  If not explicitly set, each bundle uses
        its natural default.
    """

    bundle: str = "stable_product"
    eps: float = 1e-4
    quantifier_p: float = 2.0
    quantifier_temp: float = 1.0
    implication_type: str = "reichenbach"


# ============================================================================
# 8. OperatorBundle dataclass
# ============================================================================

@dataclass
class OperatorBundle:
    """A complete set of fuzzy connectives for formula evaluation.

    Bundles decouple the *choice* of operator family from the code that
    *evaluates* formulas.  All connectives within a bundle are guaranteed
    to be semantically compatible.

    Attributes
    ----------
    name : str
        Human-readable bundle name.
    and_op : Callable[[Tensor, Tensor], Tensor]
        T-norm (fuzzy AND).
    or_op : Callable[[Tensor, Tensor], Tensor]
        T-conorm (fuzzy OR).
    not_op : Callable[[Tensor], Tensor]
        Fuzzy negation.
    implies_op : Callable[[Tensor, Tensor], Tensor]
        Fuzzy implication.
    forall : ForallAggregator
        Universal quantifier module.
    exists : ExistsAggregator
        Existential quantifier module.
    """

    name: str
    and_op: Callable[[Tensor, Tensor], Tensor]
    or_op: Callable[[Tensor, Tensor], Tensor]
    not_op: Callable[[Tensor], Tensor]
    implies_op: Callable[[Tensor, Tensor], Tensor]
    forall: ForallAggregator
    exists: ExistsAggregator


# ============================================================================
# 9. Bundle Factory Functions
# ============================================================================

def _select_implication(
    family: str,
    override: Optional[str] = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Select implication operator, optionally overriding the natural default.

    Parameters
    ----------
    family : str
        The base operator family name.
    override : str, optional
        Explicit implication type; if ``None``, use the family default.

    Returns
    -------
    Callable
        The selected implication function.
    """
    impl_type = override or {
        "godel": "godel",
        "product": "reichenbach",
        "lukasiewicz": "lukasiewicz",
        "stable_product": "reichenbach",
    }.get(family, "reichenbach")

    impls: Dict[str, Callable[[Tensor, Tensor], Tensor]] = {
        "reichenbach": reichenbach_implies,
        "godel": godel_implies,
        "goguen": goguen_implies,
        "lukasiewicz": lukasiewicz_implies,
    }
    if impl_type not in impls:
        raise ValueError(
            f"Unknown implication type '{impl_type}'. "
            f"Available: {list(impls.keys())}"
        )
    return impls[impl_type]


def _make_godel_bundle(config: OperatorConfig) -> OperatorBundle:
    """Build Godel (min/max) operator bundle."""
    return OperatorBundle(
        name="godel",
        and_op=godel_and,
        or_op=godel_or,
        not_op=standard_negation,
        implies_op=_select_implication("godel", config.implication_type),
        forall=ForallAggregator(p=config.quantifier_p, temperature=config.quantifier_temp),
        exists=ExistsAggregator(p=config.quantifier_p, temperature=config.quantifier_temp),
    )


def _make_product_bundle(config: OperatorConfig) -> OperatorBundle:
    """Build product operator bundle."""
    return OperatorBundle(
        name="product",
        and_op=product_and,
        or_op=product_or,
        not_op=standard_negation,
        implies_op=_select_implication("product", config.implication_type),
        forall=ForallAggregator(p=config.quantifier_p, temperature=config.quantifier_temp),
        exists=ExistsAggregator(p=config.quantifier_p, temperature=config.quantifier_temp),
    )


def _make_lukasiewicz_bundle(config: OperatorConfig) -> OperatorBundle:
    """Build Lukasiewicz operator bundle."""
    return OperatorBundle(
        name="lukasiewicz",
        and_op=lukasiewicz_and,
        or_op=lukasiewicz_or,
        not_op=standard_negation,
        implies_op=_select_implication("lukasiewicz", config.implication_type),
        forall=ForallAggregator(p=config.quantifier_p, temperature=config.quantifier_temp),
        exists=ExistsAggregator(p=config.quantifier_p, temperature=config.quantifier_temp),
    )


def _make_stable_product_bundle(config: OperatorConfig) -> OperatorBundle:
    """Build stable-product operator bundle with eps-projections.

    This is the recommended bundle for gradient-based training.  The eps
    projections prevent zero/one saturation at the cost of approximate
    (not exact) algebraic identities.
    """
    eps = config.eps

    # Capture eps in closures so the bundle's callables carry it
    def _and(x: Tensor, y: Tensor) -> Tensor:
        return stable_product_and(x, y, eps=eps)

    def _or(x: Tensor, y: Tensor) -> Tensor:
        return stable_product_or(x, y, eps=eps)

    def _implies(x: Tensor, y: Tensor) -> Tensor:
        return stable_product_implies(x, y, eps=eps)

    return OperatorBundle(
        name="stable_product",
        and_op=_and,
        or_op=_or,
        not_op=stable_product_not,
        implies_op=_implies,
        forall=ForallAggregator(p=config.quantifier_p, temperature=config.quantifier_temp),
        exists=ExistsAggregator(p=config.quantifier_p, temperature=config.quantifier_temp),
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

_BUNDLE_BUILDERS: Dict[str, Callable[[OperatorConfig], OperatorBundle]] = {
    "godel": _make_godel_bundle,
    "product": _make_product_bundle,
    "lukasiewicz": _make_lukasiewicz_bundle,
    "stable_product": _make_stable_product_bundle,
}


def get_operator_bundle(config: Optional[OperatorConfig] = None) -> OperatorBundle:
    """Factory: create an operator bundle from configuration.

    Parameters
    ----------
    config : OperatorConfig, optional
        If ``None``, uses the default (stable_product) configuration.

    Returns
    -------
    OperatorBundle
        Fully wired bundle with AND, OR, NOT, IMPLIES, FORALL, EXISTS.

    Raises
    ------
    ValueError
        If ``config.bundle`` is not a recognised family name.

    Examples
    --------
    >>> bundle = get_operator_bundle()
    >>> x = torch.tensor([0.8, 0.3])
    >>> y = torch.tensor([0.6, 0.9])
    >>> bundle.and_op(x, y)
    """
    if config is None:
        config = OperatorConfig()

    name = config.bundle.lower()
    if name not in _BUNDLE_BUILDERS:
        raise ValueError(
            f"Unknown operator bundle '{name}'. "
            f"Available: {list(_BUNDLE_BUILDERS.keys())}"
        )
    return _BUNDLE_BUILDERS[name](config)


def list_bundles() -> list[str]:
    """Return sorted list of available bundle names."""
    return sorted(_BUNDLE_BUILDERS.keys())


# ============================================================================
# 10. Formula AST Evaluation Helper
# ============================================================================

@dataclass
class FormulaNode:
    """Minimal AST node for formula representation.

    This is a lightweight structure for use with ``compose_formula``.
    The full rule engine (``rule_engine.py``) uses a richer AST; this
    node type provides the minimal interface needed for recursive
    evaluation.

    Attributes
    ----------
    op : str
        Operator type: ``"AND"``, ``"OR"``, ``"NOT"``, ``"IMPLIES"``,
        ``"FORALL"``, ``"EXISTS"``, ``"ATOM"``.
    children : list
        Child FormulaNode instances (for connectives/quantifiers).
    atom_key : str, optional
        Key into ``truth_cache`` for leaf atoms.
    variable_key : str, optional
        Key identifying the bound variable for quantifiers.
    """

    op: str
    children: list = field(default_factory=list)
    atom_key: Optional[str] = None
    variable_key: Optional[str] = None


def compose_formula(
    bundle: OperatorBundle,
    ast_node: FormulaNode,
    truth_cache: Dict[str, Tensor],
) -> Tensor:
    """Recursively evaluate a formula AST using the given operator bundle.

    This is the core evaluation function that ``rule_engine`` calls to
    compute truth values for complex formulas.

    Parameters
    ----------
    bundle : OperatorBundle
        The set of fuzzy connectives to use.
    ast_node : FormulaNode
        Root of the formula subtree to evaluate.
    truth_cache : dict
        Mapping from atom keys to their grounded truth tensors.
        Updated in-place if intermediate results are cached.

    Returns
    -------
    Tensor
        Truth value(s) for the formula.

    Raises
    ------
    KeyError
        If an ATOM references a key not in ``truth_cache``.
    ValueError
        If the AST node has an unrecognised operator.

    Examples
    --------
    >>> # Evaluate: AND(P(a), IMPLIES(P(a), Q(a)))
    >>> ast = FormulaNode("AND", children=[
    ...     FormulaNode("ATOM", atom_key="P_a"),
    ...     FormulaNode("IMPLIES", children=[
    ...         FormulaNode("ATOM", atom_key="P_a"),
    ...         FormulaNode("ATOM", atom_key="Q_a"),
    ...     ]),
    ... ])
    >>> truth_cache = {"P_a": torch.tensor([0.9]), "Q_a": torch.tensor([0.7])}
    >>> compose_formula(bundle, ast, truth_cache)
    """
    op = ast_node.op.upper()

    if op == "ATOM":
        if ast_node.atom_key is None:
            raise ValueError("ATOM node must have atom_key set")
        if ast_node.atom_key not in truth_cache:
            raise KeyError(
                f"Atom '{ast_node.atom_key}' not found in truth_cache. "
                f"Available keys: {list(truth_cache.keys())}"
            )
        return truth_cache[ast_node.atom_key]

    if op == "NOT":
        if len(ast_node.children) != 1:
            raise ValueError(f"NOT expects 1 child, got {len(ast_node.children)}")
        child_val = compose_formula(bundle, ast_node.children[0], truth_cache)
        return bundle.not_op(child_val)

    if op == "AND":
        if len(ast_node.children) < 2:
            raise ValueError(f"AND expects >= 2 children, got {len(ast_node.children)}")
        result = compose_formula(bundle, ast_node.children[0], truth_cache)
        for child in ast_node.children[1:]:
            child_val = compose_formula(bundle, child, truth_cache)
            result = bundle.and_op(result, child_val)
        return result

    if op == "OR":
        if len(ast_node.children) < 2:
            raise ValueError(f"OR expects >= 2 children, got {len(ast_node.children)}")
        result = compose_formula(bundle, ast_node.children[0], truth_cache)
        for child in ast_node.children[1:]:
            child_val = compose_formula(bundle, child, truth_cache)
            result = bundle.or_op(result, child_val)
        return result

    if op == "IMPLIES":
        if len(ast_node.children) != 2:
            raise ValueError(f"IMPLIES expects 2 children, got {len(ast_node.children)}")
        antecedent = compose_formula(bundle, ast_node.children[0], truth_cache)
        consequent = compose_formula(bundle, ast_node.children[1], truth_cache)
        return bundle.implies_op(antecedent, consequent)

    if op == "FORALL":
        if len(ast_node.children) != 1:
            raise ValueError(f"FORALL expects 1 child (body), got {len(ast_node.children)}")
        body_val = compose_formula(bundle, ast_node.children[0], truth_cache)
        # body_val should be (B, N) where N = number of instances
        if body_val.dim() == 1:
            body_val = body_val.unsqueeze(0)
        return bundle.forall(body_val)

    if op == "EXISTS":
        if len(ast_node.children) != 1:
            raise ValueError(f"EXISTS expects 1 child (body), got {len(ast_node.children)}")
        body_val = compose_formula(bundle, ast_node.children[0], truth_cache)
        if body_val.dim() == 1:
            body_val = body_val.unsqueeze(0)
        return bundle.exists(body_val)

    raise ValueError(
        f"Unknown formula operator '{op}'. "
        f"Supported: ATOM, NOT, AND, OR, IMPLIES, FORALL, EXISTS"
    )


# ============================================================================
# 11. Utility helpers
# ============================================================================

def clamp_truth(x: Tensor) -> Tensor:
    """Clamp tensor to valid truth range [0, 1].

    Useful as a post-processing step after operations that may
    numerically exceed the range.
    """
    return torch.clamp(x, min=0.0, max=1.0)


def truth_to_loss(satisfaction: Tensor) -> Tensor:
    """Convert a satisfiability score to a minimisable loss.

    .. math:: \\mathcal{L} = 1 - \\text{sat}

    Parameters
    ----------
    satisfaction : Tensor
        Scalar or batched satisfaction values in [0, 1].

    Returns
    -------
    Tensor
        Loss values in [0, 1].
    """
    return 1.0 - satisfaction


def batch_and(bundle: OperatorBundle, tensors: list[Tensor]) -> Tensor:
    """Apply AND across a list of tensors (left fold).

    Parameters
    ----------
    bundle : OperatorBundle
        Operator bundle to use.
    tensors : list of Tensor
        List of truth-value tensors (must all be broadcastable).

    Returns
    -------
    Tensor
        Result of applying AND sequentially.

    Raises
    ------
    ValueError
        If ``tensors`` is empty.
    """
    if not tensors:
        raise ValueError("batch_and requires at least one tensor")
    result = tensors[0]
    for t in tensors[1:]:
        result = bundle.and_op(result, t)
    return result


def batch_or(bundle: OperatorBundle, tensors: list[Tensor]) -> Tensor:
    """Apply OR across a list of tensors (left fold).

    Parameters
    ----------
    bundle : OperatorBundle
        Operator bundle to use.
    tensors : list of Tensor
        List of truth-value tensors.

    Returns
    -------
    Tensor
        Result of applying OR sequentially.

    Raises
    ------
    ValueError
        If ``tensors`` is empty.
    """
    if not tensors:
        raise ValueError("batch_or requires at least one tensor")
    result = tensors[0]
    for t in tensors[1:]:
        result = bundle.or_op(result, t)
    return result


# ============================================================================
# 12. Self-Tests
# ============================================================================

def _run_self_tests() -> None:
    """Run comprehensive self-tests for all operator families.

    Tests cover:
        1.  T-norm identity:  AND(x, 1) = x
        2.  T-norm annihilator:  AND(x, 0) = 0
        3.  T-conorm identity:  OR(x, 0) = x
        4.  T-conorm annihilator:  OR(x, 1) = 1
        5.  Negation involution:  NOT(NOT(x)) = x
        6.  Implication boundary:  IMPLIES(1, y) = y
        7.  Implication boundary:  IMPLIES(0, y) = 1
        8.  Stable product approximate identities (with tolerance)
        9.  ForallAggregator: all true -> ~1, one false -> low
        10. ExistsAggregator: one true -> high, all false -> ~0
        11. Quantifier masking correctness
        12. Gradient non-zero for product operators in interior
        13. Gradient non-zero for stable product at boundaries
        14. De Morgan for applicable bundles
        15. Bundle factory produces correct types

    Exits with code 1 on any failure.
    """
    print("=" * 72)
    print("FUZZY OPERATORS -- SELF-TEST SUITE")
    print("=" * 72)

    passed = 0
    failed = 0
    total = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            msg = f"  [FAIL] {name}"
            if detail:
                msg += f"  -- {detail}"
            print(msg)

    torch.manual_seed(42)
    x = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    y = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5])
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)

    atol_exact = 1e-6
    atol_stable = 2e-3  # tolerance for stable_product approximate identities

    # ------------------------------------------------------------------
    # Test 1: T-norm identity AND(x, 1) = x for all exact bundles
    # ------------------------------------------------------------------
    print("\n--- Test 1: T-norm identity AND(x, 1) = x ---")

    for bname in ["godel", "product", "lukasiewicz"]:
        bundle = get_operator_bundle(OperatorConfig(bundle=bname))
        result = bundle.and_op(x, ones)
        close = torch.allclose(result, x, atol=atol_exact)
        check(
            f"{bname}: AND(x, 1) = x",
            close,
            f"max_diff={torch.max(torch.abs(result - x)).item():.2e}",
        )

    # ------------------------------------------------------------------
    # Test 2: T-norm annihilator AND(x, 0) = 0 for all exact bundles
    # ------------------------------------------------------------------
    print("\n--- Test 2: T-norm annihilator AND(x, 0) = 0 ---")

    for bname in ["godel", "product", "lukasiewicz"]:
        bundle = get_operator_bundle(OperatorConfig(bundle=bname))
        result = bundle.and_op(x, zeros)
        close = torch.allclose(result, zeros, atol=atol_exact)
        check(
            f"{bname}: AND(x, 0) = 0",
            close,
            f"max_diff={torch.max(torch.abs(result - zeros)).item():.2e}",
        )

    # ------------------------------------------------------------------
    # Test 3: T-conorm identity OR(x, 0) = x for all exact bundles
    # ------------------------------------------------------------------
    print("\n--- Test 3: T-conorm identity OR(x, 0) = x ---")

    for bname in ["godel", "product", "lukasiewicz"]:
        bundle = get_operator_bundle(OperatorConfig(bundle=bname))
        result = bundle.or_op(x, zeros)
        close = torch.allclose(result, x, atol=atol_exact)
        check(
            f"{bname}: OR(x, 0) = x",
            close,
            f"max_diff={torch.max(torch.abs(result - x)).item():.2e}",
        )

    # ------------------------------------------------------------------
    # Test 4: T-conorm annihilator OR(x, 1) = 1 for all exact bundles
    # ------------------------------------------------------------------
    print("\n--- Test 4: T-conorm annihilator OR(x, 1) = 1 ---")

    for bname in ["godel", "product", "lukasiewicz"]:
        bundle = get_operator_bundle(OperatorConfig(bundle=bname))
        result = bundle.or_op(x, ones)
        close = torch.allclose(result, ones, atol=atol_exact)
        check(
            f"{bname}: OR(x, 1) = 1",
            close,
            f"max_diff={torch.max(torch.abs(result - ones)).item():.2e}",
        )

    # ------------------------------------------------------------------
    # Test 5: Negation involution NOT(NOT(x)) = x
    # ------------------------------------------------------------------
    print("\n--- Test 5: Negation involution NOT(NOT(x)) = x ---")

    for bname in ["godel", "product", "lukasiewicz", "stable_product"]:
        bundle = get_operator_bundle(OperatorConfig(bundle=bname))
        result = bundle.not_op(bundle.not_op(x))
        close = torch.allclose(result, x, atol=atol_exact)
        check(
            f"{bname}: NOT(NOT(x)) = x",
            close,
            f"max_diff={torch.max(torch.abs(result - x)).item():.2e}",
        )

    # ------------------------------------------------------------------
    # Test 6: Implication boundary IMPLIES(1, y) = y
    # ------------------------------------------------------------------
    print("\n--- Test 6: Implication boundary IMPLIES(1, y) = y ---")

    for bname in ["godel", "product", "lukasiewicz"]:
        config = OperatorConfig(bundle=bname)
        # Use each family's natural implication
        natural_impl = {
            "godel": "godel",
            "product": "reichenbach",
            "lukasiewicz": "lukasiewicz",
        }
        config.implication_type = natural_impl[bname]
        bundle = get_operator_bundle(config)
        result = bundle.implies_op(ones, y)
        close = torch.allclose(result, y, atol=atol_exact)
        check(
            f"{bname} ({config.implication_type}): IMPLIES(1, y) = y",
            close,
            f"max_diff={torch.max(torch.abs(result - y)).item():.2e}",
        )

    # ------------------------------------------------------------------
    # Test 7: Implication boundary IMPLIES(0, y) = 1
    # ------------------------------------------------------------------
    print("\n--- Test 7: Implication boundary IMPLIES(0, y) = 1 ---")

    for bname in ["godel", "product", "lukasiewicz"]:
        config = OperatorConfig(bundle=bname)
        natural_impl = {
            "godel": "godel",
            "product": "reichenbach",
            "lukasiewicz": "lukasiewicz",
        }
        config.implication_type = natural_impl[bname]
        bundle = get_operator_bundle(config)
        result = bundle.implies_op(zeros, y)
        close = torch.allclose(result, ones, atol=atol_exact)
        check(
            f"{bname} ({config.implication_type}): IMPLIES(0, y) = 1",
            close,
            f"max_diff={torch.max(torch.abs(result - ones)).item():.2e}",
        )

    # ------------------------------------------------------------------
    # Test 8: Stable product approximate identities with tolerance
    # ------------------------------------------------------------------
    print("\n--- Test 8: Stable product approximate identities ---")

    sp_config = OperatorConfig(bundle="stable_product", eps=1e-4)
    sp = get_operator_bundle(sp_config)

    # AND(x, 1) ~ x  (approximate)
    result_and = sp.and_op(x, ones)
    close_and = torch.allclose(result_and, x, atol=atol_stable)
    check(
        "stable_product: AND(x, 1) ~ x (atol=2e-3)",
        close_and,
        f"max_diff={torch.max(torch.abs(result_and - x)).item():.2e}",
    )

    # OR(x, 0) ~ x  (approximate)
    result_or = sp.or_op(x, zeros)
    close_or = torch.allclose(result_or, x, atol=atol_stable)
    check(
        "stable_product: OR(x, 0) ~ x (atol=2e-3)",
        close_or,
        f"max_diff={torch.max(torch.abs(result_or - x)).item():.2e}",
    )

    # AND(x, 0) ~ 0  (approximate, bounded by eps^2)
    result_and_zero = sp.and_op(x, zeros)
    max_val = torch.max(result_and_zero).item()
    check(
        f"stable_product: AND(x, 0) ~ 0 (max={max_val:.2e})",
        max_val < 0.01,
        f"max_val={max_val:.2e}",
    )

    # OR(x, 1) ~ 1  (approximate)
    result_or_one = sp.or_op(x, ones)
    close_or_one = torch.allclose(result_or_one, ones, atol=atol_stable)
    check(
        "stable_product: OR(x, 1) ~ 1 (atol=2e-3)",
        close_or_one,
        f"max_diff={torch.max(torch.abs(result_or_one - ones)).item():.2e}",
    )

    # ------------------------------------------------------------------
    # Test 9: ForallAggregator -- all true -> ~1, one false -> low
    # ------------------------------------------------------------------
    print("\n--- Test 9: ForallAggregator ---")

    fa = ForallAggregator(p=2.0)

    # All true
    all_true = torch.ones(2, 5)
    fa_all = fa(all_true)
    check(
        "ForallAggregator: all true -> ~1",
        (fa_all > 0.99).all().item(),
        f"values={fa_all.tolist()}",
    )

    # One false (0.0) among trues
    mixed = torch.ones(2, 5)
    mixed[:, 2] = 0.0
    fa_mixed = fa(mixed)
    check(
        "ForallAggregator: one false -> low",
        (fa_mixed < 0.7).all().item(),
        f"values={fa_mixed.tolist()}",
    )

    # Gradient through forall is non-zero
    tv_grad = torch.tensor([[0.8, 0.6, 0.9, 0.5]], requires_grad=True)
    fa_val = fa(tv_grad)
    fa_val.sum().backward()
    check(
        "ForallAggregator: gradient non-zero",
        tv_grad.grad is not None and (tv_grad.grad.abs() > 1e-8).any().item(),
        f"grad={tv_grad.grad}",
    )

    # ------------------------------------------------------------------
    # Test 10: ExistsAggregator -- one true -> high, all false -> ~0
    # ------------------------------------------------------------------
    print("\n--- Test 10: ExistsAggregator ---")

    ea = ExistsAggregator(p=2.0)

    # All false
    all_false = torch.zeros(2, 5)
    ea_false = ea(all_false)
    check(
        "ExistsAggregator: all false -> ~0",
        (ea_false < 0.01).all().item(),
        f"values={ea_false.tolist()}",
    )

    # One true (1.0) among falses
    mixed2 = torch.zeros(2, 5)
    mixed2[:, 3] = 1.0
    ea_mixed = ea(mixed2)
    check(
        "ExistsAggregator: one true -> high",
        (ea_mixed > 0.3).all().item(),
        f"values={ea_mixed.tolist()}",
    )

    # ------------------------------------------------------------------
    # Test 11: Quantifier masking correctness
    # ------------------------------------------------------------------
    print("\n--- Test 11: Quantifier masking ---")

    fa_mask = ForallAggregator(p=2.0)
    ea_mask = ExistsAggregator(p=2.0)

    # truth_values: (1, 4), but only first 2 are valid
    tv_m = torch.tensor([[0.9, 0.8, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    fa_masked = fa_mask(tv_m, mask=mask)
    fa_unmasked_small = fa_mask(torch.tensor([[0.9, 0.8]]))

    # Masked forall should match forall over only the valid elements
    close_fa = torch.allclose(fa_masked, fa_unmasked_small, atol=1e-5)
    check(
        "Forall masking: matches unmasked subset",
        close_fa,
        f"masked={fa_masked.item():.4f}, unmasked={fa_unmasked_small.item():.4f}",
    )

    ea_masked = ea_mask(tv_m, mask=mask)
    ea_unmasked_small = ea_mask(torch.tensor([[0.9, 0.8]]))
    close_ea = torch.allclose(ea_masked, ea_unmasked_small, atol=1e-5)
    check(
        "Exists masking: matches unmasked subset",
        close_ea,
        f"masked={ea_masked.item():.4f}, unmasked={ea_unmasked_small.item():.4f}",
    )

    # ------------------------------------------------------------------
    # Test 12: Gradient non-zero for product operators in interior
    # ------------------------------------------------------------------
    print("\n--- Test 12: Product operator gradients in interior ---")

    prod_bundle = get_operator_bundle(OperatorConfig(bundle="product"))

    x_g = torch.tensor([0.5], requires_grad=True)
    y_g = torch.tensor([0.6], requires_grad=True)
    and_val = prod_bundle.and_op(x_g, y_g)
    and_val.backward()
    check(
        "product AND: grad(x) non-zero at x=0.5, y=0.6",
        x_g.grad is not None and x_g.grad.abs().item() > 1e-8,
        f"grad_x={x_g.grad}",
    )
    check(
        "product AND: grad(y) non-zero at x=0.5, y=0.6",
        y_g.grad is not None and y_g.grad.abs().item() > 1e-8,
        f"grad_y={y_g.grad}",
    )

    x_g2 = torch.tensor([0.5], requires_grad=True)
    y_g2 = torch.tensor([0.6], requires_grad=True)
    or_val = prod_bundle.or_op(x_g2, y_g2)
    or_val.backward()
    check(
        "product OR: grad(x) non-zero at x=0.5, y=0.6",
        x_g2.grad is not None and x_g2.grad.abs().item() > 1e-8,
        f"grad_x={x_g2.grad}",
    )

    # ------------------------------------------------------------------
    # Test 13: Gradient non-zero for stable product at boundaries
    # ------------------------------------------------------------------
    print("\n--- Test 13: Stable product gradients at boundaries ---")

    sp_bundle = get_operator_bundle(OperatorConfig(bundle="stable_product", eps=1e-4))

    # At boundary x=0.0
    x_b = torch.tensor([0.0], requires_grad=True)
    y_b = torch.tensor([0.5], requires_grad=True)
    sp_val = sp_bundle.and_op(x_b, y_b)
    sp_val.backward()
    check(
        "stable_product AND: grad(y) non-zero at x=0.0, y=0.5",
        y_b.grad is not None and y_b.grad.abs().item() > 1e-8,
        f"grad_y={y_b.grad}",
    )

    # At boundary y=1.0
    x_b2 = torch.tensor([0.5], requires_grad=True)
    y_b2 = torch.tensor([1.0], requires_grad=True)
    sp_val2 = sp_bundle.and_op(x_b2, y_b2)
    sp_val2.backward()
    check(
        "stable_product AND: grad(x) non-zero at x=0.5, y=1.0",
        x_b2.grad is not None and x_b2.grad.abs().item() > 1e-8,
        f"grad_x={x_b2.grad}",
    )

    # OR at boundary x=1.0
    x_b3 = torch.tensor([1.0], requires_grad=True)
    y_b3 = torch.tensor([0.5], requires_grad=True)
    sp_or_val = sp_bundle.or_op(x_b3, y_b3)
    sp_or_val.backward()
    check(
        "stable_product OR: grad(y) non-zero at x=1.0, y=0.5",
        y_b3.grad is not None and y_b3.grad.abs().item() > 1e-8,
        f"grad_y={y_b3.grad}",
    )

    # Implication at boundary x=0.0 (should be ~1, with grad for y)
    x_b4 = torch.tensor([0.0], requires_grad=True)
    y_b4 = torch.tensor([0.3], requires_grad=True)
    sp_impl_val = sp_bundle.implies_op(x_b4, y_b4)
    sp_impl_val.backward()
    check(
        "stable_product IMPLIES: grad(y) non-zero at x=0.0, y=0.3",
        y_b4.grad is not None and y_b4.grad.abs().item() > 1e-8,
        f"grad_y={y_b4.grad}",
    )

    # ------------------------------------------------------------------
    # Test 14: De Morgan laws for applicable bundles
    # ------------------------------------------------------------------
    print("\n--- Test 14: De Morgan laws ---")

    # De Morgan: NOT(AND(x,y)) = OR(NOT(x), NOT(y))
    # This holds for all standard t-norm/t-conorm dual pairs
    dm_x = torch.tensor([0.2, 0.5, 0.8, 0.0, 1.0])
    dm_y = torch.tensor([0.3, 0.6, 0.4, 1.0, 0.0])

    for bname in ["godel", "product", "lukasiewicz"]:
        bundle = get_operator_bundle(OperatorConfig(bundle=bname))
        lhs = bundle.not_op(bundle.and_op(dm_x, dm_y))
        rhs = bundle.or_op(bundle.not_op(dm_x), bundle.not_op(dm_y))
        close = torch.allclose(lhs, rhs, atol=1e-5)
        check(
            f"{bname}: NOT(AND(x,y)) = OR(NOT(x), NOT(y))  [De Morgan]",
            close,
            f"max_diff={torch.max(torch.abs(lhs - rhs)).item():.2e}",
        )

    # De Morgan: NOT(OR(x,y)) = AND(NOT(x), NOT(y))
    for bname in ["godel", "product", "lukasiewicz"]:
        bundle = get_operator_bundle(OperatorConfig(bundle=bname))
        lhs = bundle.not_op(bundle.or_op(dm_x, dm_y))
        rhs = bundle.and_op(bundle.not_op(dm_x), bundle.not_op(dm_y))
        close = torch.allclose(lhs, rhs, atol=1e-5)
        check(
            f"{bname}: NOT(OR(x,y)) = AND(NOT(x), NOT(y))  [De Morgan]",
            close,
            f"max_diff={torch.max(torch.abs(lhs - rhs)).item():.2e}",
        )

    # Stable product: approximate De Morgan
    sp_bundle = get_operator_bundle(OperatorConfig(bundle="stable_product", eps=1e-4))
    lhs_sp = sp_bundle.not_op(sp_bundle.and_op(dm_x, dm_y))
    rhs_sp = sp_bundle.or_op(sp_bundle.not_op(dm_x), sp_bundle.not_op(dm_y))
    close_sp = torch.allclose(lhs_sp, rhs_sp, atol=5e-3)
    check(
        "stable_product: NOT(AND(x,y)) ~ OR(NOT(x), NOT(y))  [approx De Morgan]",
        close_sp,
        f"max_diff={torch.max(torch.abs(lhs_sp - rhs_sp)).item():.2e}",
    )

    # ------------------------------------------------------------------
    # Test 15: Bundle factory produces correct types
    # ------------------------------------------------------------------
    print("\n--- Test 15: Bundle factory types ---")

    for bname in ["godel", "product", "lukasiewicz", "stable_product"]:
        bundle = get_operator_bundle(OperatorConfig(bundle=bname))
        check(
            f"get_operator_bundle('{bname}') returns OperatorBundle",
            isinstance(bundle, OperatorBundle),
        )
        check(
            f"  .name == '{bname}'",
            bundle.name == bname,
            f"got '{bundle.name}'",
        )
        check(
            f"  .and_op is callable",
            callable(bundle.and_op),
        )
        check(
            f"  .or_op is callable",
            callable(bundle.or_op),
        )
        check(
            f"  .not_op is callable",
            callable(bundle.not_op),
        )
        check(
            f"  .implies_op is callable",
            callable(bundle.implies_op),
        )
        check(
            f"  .forall is ForallAggregator",
            isinstance(bundle.forall, ForallAggregator),
        )
        check(
            f"  .exists is ExistsAggregator",
            isinstance(bundle.exists, ExistsAggregator),
        )

    # Test error on unknown bundle
    try:
        get_operator_bundle(OperatorConfig(bundle="nonexistent"))
        check("Factory raises on unknown bundle", False, "no exception raised")
    except ValueError:
        check("Factory raises ValueError on unknown bundle", True)

    # ------------------------------------------------------------------
    # Additional test: compose_formula basic evaluation
    # ------------------------------------------------------------------
    print("\n--- Bonus: compose_formula ---")

    bundle = get_operator_bundle(OperatorConfig(bundle="product"))
    # AND(P_a, IMPLIES(P_a, Q_a))
    ast = FormulaNode("AND", children=[
        FormulaNode("ATOM", atom_key="P_a"),
        FormulaNode("IMPLIES", children=[
            FormulaNode("ATOM", atom_key="P_a"),
            FormulaNode("ATOM", atom_key="Q_a"),
        ]),
    ])
    cache = {"P_a": torch.tensor([0.9]), "Q_a": torch.tensor([0.7])}
    result = compose_formula(bundle, ast, cache)
    # AND(0.9, IMPLIES(0.9, 0.7))
    # IMPLIES(0.9, 0.7) = 1 - 0.9 + 0.9*0.7 = 0.1 + 0.63 = 0.73
    # AND(0.9, 0.73) = 0.9 * 0.73 = 0.657
    expected = torch.tensor([0.657])
    close = torch.allclose(result, expected, atol=1e-3)
    check(
        "compose_formula: AND(P, IMPLIES(P, Q)) correct",
        close,
        f"result={result.item():.4f}, expected={expected.item():.4f}",
    )

    # compose_formula with NOT
    ast_not = FormulaNode("NOT", children=[FormulaNode("ATOM", atom_key="P_a")])
    result_not = compose_formula(bundle, ast_not, cache)
    expected_not = torch.tensor([0.1])
    close_not = torch.allclose(result_not, expected_not, atol=1e-5)
    check(
        "compose_formula: NOT(P) correct",
        close_not,
        f"result={result_not.item():.4f}, expected={expected_not.item():.4f}",
    )

    # compose_formula with OR
    ast_or = FormulaNode("OR", children=[
        FormulaNode("ATOM", atom_key="P_a"),
        FormulaNode("ATOM", atom_key="Q_a"),
    ])
    result_or = compose_formula(bundle, ast_or, cache)
    # product OR: 0.9 + 0.7 - 0.9*0.7 = 0.97
    expected_or = torch.tensor([0.97])
    close_or = torch.allclose(result_or, expected_or, atol=1e-3)
    check(
        "compose_formula: OR(P, Q) correct",
        close_or,
        f"result={result_or.item():.4f}, expected={expected_or.item():.4f}",
    )

    # compose_formula error on missing key
    try:
        ast_bad = FormulaNode("ATOM", atom_key="nonexistent")
        compose_formula(bundle, ast_bad, cache)
        check("compose_formula: raises on missing atom", False, "no exception raised")
    except KeyError:
        check("compose_formula: raises KeyError on missing atom", True)

    # ------------------------------------------------------------------
    # Additional test: Commutativity of AND and OR
    # ------------------------------------------------------------------
    print("\n--- Bonus: Commutativity ---")

    for bname in ["godel", "product", "lukasiewicz"]:
        bundle = get_operator_bundle(OperatorConfig(bundle=bname))
        c_x = torch.rand(50)
        c_y = torch.rand(50)
        and_xy = bundle.and_op(c_x, c_y)
        and_yx = bundle.and_op(c_y, c_x)
        check(
            f"{bname}: AND commutativity",
            torch.allclose(and_xy, and_yx, atol=1e-6),
        )
        or_xy = bundle.or_op(c_x, c_y)
        or_yx = bundle.or_op(c_y, c_x)
        check(
            f"{bname}: OR commutativity",
            torch.allclose(or_xy, or_yx, atol=1e-6),
        )

    # ------------------------------------------------------------------
    # Additional test: Output always in [0, 1]
    # ------------------------------------------------------------------
    print("\n--- Bonus: Output range [0, 1] ---")

    rand_x = torch.rand(1000)
    rand_y = torch.rand(1000)

    for bname in ["godel", "product", "lukasiewicz", "stable_product"]:
        bundle = get_operator_bundle(OperatorConfig(bundle=bname))
        for op_name, op_fn in [
            ("AND", lambda: bundle.and_op(rand_x, rand_y)),
            ("OR", lambda: bundle.or_op(rand_x, rand_y)),
            ("NOT", lambda: bundle.not_op(rand_x)),
            ("IMPLIES", lambda: bundle.implies_op(rand_x, rand_y)),
        ]:
            result = op_fn()
            in_range = (result >= -1e-6).all().item() and (result <= 1.0 + 1e-6).all().item()
            check(
                f"{bname} {op_name}: output in [0, 1]",
                in_range,
                f"min={result.min().item():.6f}, max={result.max().item():.6f}",
            )

    # ------------------------------------------------------------------
    # Additional test: Associativity of AND and OR
    # ------------------------------------------------------------------
    print("\n--- Bonus: Associativity ---")

    a_x = torch.rand(100)
    a_y = torch.rand(100)
    a_z = torch.rand(100)

    for bname in ["godel", "product", "lukasiewicz"]:
        bundle = get_operator_bundle(OperatorConfig(bundle=bname))
        # AND(AND(x,y), z) = AND(x, AND(y,z))
        lhs_a = bundle.and_op(bundle.and_op(a_x, a_y), a_z)
        rhs_a = bundle.and_op(a_x, bundle.and_op(a_y, a_z))
        check(
            f"{bname}: AND associativity",
            torch.allclose(lhs_a, rhs_a, atol=1e-5),
            f"max_diff={torch.max(torch.abs(lhs_a - rhs_a)).item():.2e}",
        )

        # OR(OR(x,y), z) = OR(x, OR(y,z))
        lhs_o = bundle.or_op(bundle.or_op(a_x, a_y), a_z)
        rhs_o = bundle.or_op(a_x, bundle.or_op(a_y, a_z))
        check(
            f"{bname}: OR associativity",
            torch.allclose(lhs_o, rhs_o, atol=1e-5),
            f"max_diff={torch.max(torch.abs(lhs_o - rhs_o)).item():.2e}",
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 72)

    if failed > 0:
        print("\nSelf-tests FAILED.")
        sys.exit(1)
    else:
        print("\nAll self-tests PASSED.")


if __name__ == "__main__":
    _run_self_tests()
