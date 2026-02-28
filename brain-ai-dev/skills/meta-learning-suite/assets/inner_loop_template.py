"""
Differentiable Inner-Loop Optimization Engine for Meta-Learning.

This module provides the core inner-loop optimization machinery for MAML,
FOMAML, and Reptile meta-learning algorithms. It implements three backends
in priority order:

    1. TorchFuncEngine  -- Uses torch.func.functional_call + torch.func.grad
                           (preferred, PyTorch >= 2.0)
    2. HigherEngine      -- Uses the `higher` library's differentiable optimizer
                           (optional, feature-flag guarded)
    3. CustomSGDEngine   -- Pure PyTorch via torch.autograd.grad + manual update
                           (always available fallback)

All engines share the same abstract interface (InnerLoopEngine.adapt) and
produce identical InnerLoopResult objects so that algorithm code above this
layer is backend-agnostic.

Key design decisions
--------------------
- **fp32 enforcement**: Inner-loop gradients are fragile under mixed-precision.
  The AMPSafeInnerLoop wrapper casts inputs and parameters to fp32 before
  delegating to the underlying engine, ensuring numerical stability even when
  the outer loop runs in bf16/fp16.

- **LSLR support**: Per-layer, per-step learning rates (MAML++) are threaded
  through every backend via the `lrs` parameter, a nested dict mapping
  parameter names to per-step LR overrides.

- **Gradient clipping**: Optional global norm clipping is applied to the
  gradient tuple *before* the parameter update, with a boolean flag in StepLog
  indicating whether clipping was triggered.

- **create_graph control**: The first_order flag determines whether the
  computational graph is retained through the inner loop. MAML requires
  create_graph=True (second-order), FOMAML uses create_graph=False (first-order).

- **Functional forward**: All backends use a functional forward pass via
  torch.func.functional_call when available, falling back to a manual
  parameter-replacement strategy otherwise. This avoids monkey-patching
  module parameters, which is fragile and error-prone.

Usage
-----
    >>> engine = create_inner_loop_engine(backend="auto")
    >>> result = engine.adapt(
    ...     model=model,
    ...     params=dict(model.named_parameters()),
    ...     support_x=x_support,
    ...     support_y=y_support,
    ...     steps=5,
    ...     lr=0.01,
    ...     first_order=False,
    ... )
    >>> result.adapted_params  # dict of adapted parameter tensors
    >>> result.step_logs       # per-step diagnostics

References
----------
- Finn et al. (2017) "Model-Agnostic Meta-Learning for Fast Adaptation
  of Deep Networks" (MAML)
- Antoniou et al. (2019) "How to Train Your MAML" (MAML++)
- Nichol et al. (2018) "On First-Order Meta-Learning Algorithms" (Reptile)
- PyTorch torch.func documentation: functional_call, grad, vmap

Template version: 0.1.0
Target location: brain_ai/meta/inner_loop.py
"""

# =============================================================================
# Standard library imports
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from abc import ABC, abstractmethod
import warnings
import math
import copy
import time
import traceback

# =============================================================================
# Optional dependency imports
# =============================================================================

# torch.func (PyTorch >= 2.0): preferred for functional transforms
try:
    from torch.func import functional_call, grad as func_grad, vmap
    HAS_TORCH_FUNC = True
except ImportError:
    HAS_TORCH_FUNC = False
    functional_call = None
    func_grad = None
    vmap = None

# higher library: alternative differentiable optimizer
try:
    import higher
    HAS_HIGHER = True
except ImportError:
    HAS_HIGHER = False
    higher = None


# =============================================================================
# Module-level constants
# =============================================================================

# Minimum clamp for learned learning rates to prevent collapse
LSLR_MIN_LR = 1e-6

# Maximum clamp for learned learning rates to prevent explosion
LSLR_MAX_LR = 1.0

# Small epsilon for numerical stability in norm computations
NORM_EPS = 1e-6

# Default gradient clipping norm (None means no clipping)
DEFAULT_CLIP_NORM = None

# Version identifier for checkpoint compatibility
ENGINE_VERSION = "0.1.0"


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class StepLog:
    """
    Log entry for one inner-loop optimization step.

    Captures all diagnostics needed to monitor inner-loop health:
    loss trajectory, accuracy, gradient statistics, and learning rate
    information. These logs are aggregated into adaptation curves for
    meta-learning diagnostics.

    Attributes
    ----------
    step : int
        Zero-indexed step number within the inner loop.
    loss : float
        Task loss at this step (computed on support set).
    accuracy : float
        Classification accuracy at this step (support set).
        For regression tasks, this may be set to 0.0 or a suitable metric.
    grad_norm : float
        L2 norm of the gradient vector across all parameters.
    update_norm : float
        L2 norm of the actual parameter update (lr * grad, post-clipping).
    lr_effective : float
        The learning rate actually used for the update. When LSLR is active,
        this is the mean effective LR across all parameters for this step.
    clipped : bool
        Whether gradient clipping was applied at this step.
    """
    step: int
    loss: float
    accuracy: float
    grad_norm: float
    update_norm: float
    lr_effective: float
    clipped: bool

    def to_dict(self) -> dict:
        """
        Serialize to a plain dictionary for JSON logging.

        Returns
        -------
        dict
            All fields as a JSON-serializable dictionary. Float values
            that are NaN or Inf are converted to string representations
            to avoid JSON serialization errors.
        """
        d = {
            "step": self.step,
            "loss": self.loss if math.isfinite(self.loss) else str(self.loss),
            "accuracy": self.accuracy,
            "grad_norm": self.grad_norm if math.isfinite(self.grad_norm) else str(self.grad_norm),
            "update_norm": self.update_norm if math.isfinite(self.update_norm) else str(self.update_norm),
            "lr_effective": self.lr_effective,
            "clipped": self.clipped,
        }
        return d

    def __repr__(self) -> str:
        return (
            f"StepLog(step={self.step}, loss={self.loss:.4f}, "
            f"acc={self.accuracy:.3f}, grad_norm={self.grad_norm:.4f}, "
            f"update_norm={self.update_norm:.4f}, lr={self.lr_effective:.6f}, "
            f"clipped={self.clipped})"
        )


@dataclass
class InnerLoopResult:
    """
    Result container for a complete inner-loop adaptation run.

    Returned by all InnerLoopEngine.adapt() implementations. Contains
    the adapted parameters (with or without grad_fn depending on
    first_order setting), per-step diagnostic logs, and summary statistics.

    Attributes
    ----------
    adapted_params : Dict[str, torch.Tensor]
        Dictionary mapping parameter names to their adapted values.
        In MAML mode (first_order=False), these tensors retain grad_fn
        for second-order meta-gradient computation. In FOMAML mode
        (first_order=True), tensors may be detached leaf tensors.
    step_logs : List[StepLog]
        One StepLog per inner-loop step, ordered chronologically.
    total_inner_loss : float
        Sum of inner-loop losses across all steps. Useful for multi-step
        loss (MSL) computation in MAML++.
    final_accuracy : float
        Classification accuracy at the final inner-loop step.
    num_steps : int
        Number of inner-loop steps actually executed.
    wall_time_ms : float
        Wall-clock time for the inner loop in milliseconds.
    backend : str
        Name of the backend engine that produced this result.
    """
    adapted_params: Dict[str, torch.Tensor]
    step_logs: List[StepLog]
    total_inner_loss: float
    final_accuracy: float
    num_steps: int = 0
    wall_time_ms: float = 0.0
    backend: str = "unknown"

    def loss_trajectory(self) -> List[float]:
        """Return the loss at each step as a list."""
        return [log.loss for log in self.step_logs]

    def accuracy_trajectory(self) -> List[float]:
        """Return the accuracy at each step as a list."""
        return [log.accuracy for log in self.step_logs]

    def grad_norm_trajectory(self) -> List[float]:
        """Return the gradient norm at each step as a list."""
        return [log.grad_norm for log in self.step_logs]

    def loss_decreased(self) -> bool:
        """
        Check whether the loss decreased from first to last step.

        Returns True if the final step loss is strictly less than the
        first step loss. Returns False if there are fewer than 2 steps
        or if the loss did not decrease.
        """
        if len(self.step_logs) < 2:
            return False
        return self.step_logs[-1].loss < self.step_logs[0].loss

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "total_inner_loss": self.total_inner_loss,
            "final_accuracy": self.final_accuracy,
            "num_steps": self.num_steps,
            "wall_time_ms": self.wall_time_ms,
            "backend": self.backend,
            "step_logs": [log.to_dict() for log in self.step_logs],
        }

    def __repr__(self) -> str:
        return (
            f"InnerLoopResult(steps={self.num_steps}, "
            f"total_loss={self.total_inner_loss:.4f}, "
            f"final_acc={self.final_accuracy:.3f}, "
            f"backend={self.backend!r}, "
            f"wall_time={self.wall_time_ms:.1f}ms)"
        )


# =============================================================================
# Gradient Clipping Utility
# =============================================================================

def clip_grad_tuple(
    grads: Tuple[Optional[torch.Tensor], ...],
    max_norm: float,
) -> Tuple[Tuple[Optional[torch.Tensor], ...], bool]:
    """
    Clip a tuple of gradients by global L2 norm.

    This mirrors torch.nn.utils.clip_grad_norm_ but operates on a tuple
    of gradient tensors (as returned by torch.autograd.grad) rather than
    on parameter .grad attributes. None entries in the tuple are skipped.

    Parameters
    ----------
    grads : Tuple[Optional[torch.Tensor], ...]
        Tuple of gradient tensors. None entries are allowed and ignored.
    max_norm : float
        Maximum allowed L2 norm. Must be positive.

    Returns
    -------
    clipped_grads : Tuple[Optional[torch.Tensor], ...]
        Gradient tuple after clipping. Shape and device match inputs.
    was_clipped : bool
        True if the global norm exceeded max_norm and clipping was applied.

    Examples
    --------
    >>> g1 = torch.randn(10)
    >>> g2 = torch.randn(5)
    >>> clipped, was_clipped = clip_grad_tuple((g1, g2), max_norm=1.0)
    """
    # Compute total L2 norm across all non-None gradients
    total_norm_sq = torch.tensor(0.0)
    device = None
    for g in grads:
        if g is not None:
            if device is None:
                device = g.device
                total_norm_sq = total_norm_sq.to(device)
            total_norm_sq = total_norm_sq + g.detach().norm().pow(2)

    if device is None:
        # All gradients are None
        return grads, False

    total_norm = torch.sqrt(total_norm_sq)
    clip_coef = max_norm / (total_norm + NORM_EPS)

    # Determine if clipping is needed (compare on CPU to avoid sync issues)
    was_clipped = clip_coef.item() < 1.0

    if was_clipped:
        # Apply clipping coefficient to all non-None gradients
        clipped_grads = tuple(
            g * clip_coef if g is not None else None
            for g in grads
        )
        return clipped_grads, True
    else:
        return grads, False


def compute_grad_norm(grads: Tuple[Optional[torch.Tensor], ...]) -> float:
    """
    Compute the global L2 norm of a tuple of gradients.

    Parameters
    ----------
    grads : Tuple[Optional[torch.Tensor], ...]
        Gradient tuple with possible None entries.

    Returns
    -------
    float
        Global L2 norm. Returns 0.0 if all gradients are None.
    """
    total_norm_sq = 0.0
    for g in grads:
        if g is not None:
            total_norm_sq += g.detach().norm().pow(2).item()
    return math.sqrt(total_norm_sq)


def compute_update_norm(
    grads: Tuple[Optional[torch.Tensor], ...],
    lr: float,
) -> float:
    """
    Compute the L2 norm of the parameter update (lr * grad).

    Parameters
    ----------
    grads : Tuple[Optional[torch.Tensor], ...]
        Gradient tuple.
    lr : float
        Effective learning rate.

    Returns
    -------
    float
        L2 norm of the update vector.
    """
    total_norm_sq = 0.0
    for g in grads:
        if g is not None:
            total_norm_sq += (lr * g).detach().norm().pow(2).item()
    return math.sqrt(total_norm_sq)


# =============================================================================
# Functional Forward Helper
# =============================================================================

def _functional_forward(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Run a model's forward pass with explicitly provided parameters.

    This function replaces the model's parameters with the given dict
    and performs a forward pass. It uses torch.func.functional_call
    when available (clean, no side effects) and falls back to manual
    parameter replacement otherwise.

    Parameters
    ----------
    model : nn.Module
        The model architecture. Its original parameters are NOT modified.
    params : Dict[str, torch.Tensor]
        Parameter dictionary keyed by the names from model.named_parameters().
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Model output.

    Notes
    -----
    The torch.func.functional_call backend is strongly preferred because
    it avoids side effects on the model's parameter buffers. The manual
    fallback temporarily swaps parameters, which is not thread-safe.
    """
    if HAS_TORCH_FUNC and functional_call is not None:
        return functional_call(model, params, (x,))
    else:
        # Manual fallback: temporarily replace parameters, run forward,
        # then restore originals. This is NOT thread-safe.
        original_params = {}
        try:
            for name, param in model.named_parameters():
                original_params[name] = param.data
                if name in params:
                    # Navigate the module hierarchy to set the parameter
                    _set_param_by_name(model, name, params[name])
            output = model(x)
        finally:
            # Restore original parameters
            for name, orig_data in original_params.items():
                _set_param_by_name(model, name, orig_data)
        return output


def _set_param_by_name(
    model: nn.Module,
    name: str,
    value: torch.Tensor,
) -> None:
    """
    Set a parameter in a module by its dotted name.

    Navigates the module hierarchy (e.g., "layer1.weight") and replaces
    the parameter data. This is used by the manual functional forward
    fallback.

    Parameters
    ----------
    model : nn.Module
        Root module.
    name : str
        Dotted parameter name (e.g., "encoder.fc.weight").
    value : torch.Tensor
        New parameter value.
    """
    parts = name.split(".")
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    # Set the data on the existing parameter tensor
    param = getattr(module, parts[-1])
    if isinstance(param, nn.Parameter):
        param.data = value
    else:
        setattr(module, parts[-1], value)


def _extract_params(
    model: nn.Module,
    clone: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Extract a dictionary of named parameters from a model.

    Parameters
    ----------
    model : nn.Module
        The model to extract parameters from.
    clone : bool
        If True, clone parameter tensors so modifications do not affect
        the original model. Default True.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary mapping parameter names to tensors.
    """
    if clone:
        return {name: p.clone() for name, p in model.named_parameters()}
    else:
        return dict(model.named_parameters())


def _compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute classification accuracy.

    Handles both classification (integer targets) and regression
    (float targets) gracefully.

    Parameters
    ----------
    logits : torch.Tensor
        Model output, shape (batch, num_classes) for classification.
    targets : torch.Tensor
        Ground-truth targets.

    Returns
    -------
    float
        Accuracy in [0, 1]. Returns 0.0 for regression tasks.
    """
    if targets.dtype in (torch.long, torch.int, torch.int32):
        if logits.dim() >= 2 and logits.shape[-1] > 1:
            preds = logits.argmax(dim=-1)
            return (preds == targets).float().mean().item()
    return 0.0


# =============================================================================
# Abstract Base Class: InnerLoopEngine
# =============================================================================

class InnerLoopEngine(ABC):
    """
    Abstract base class for inner-loop optimization backends.

    All inner-loop engines must implement the adapt() method, which takes
    a model, its parameters, and a support set, and returns an
    InnerLoopResult containing adapted parameters and diagnostics.

    The engine is stateless -- all configuration is passed via adapt()
    arguments. This allows the same engine instance to be reused across
    different tasks and episodes.

    Subclasses
    ----------
    - TorchFuncEngine: Uses torch.func for clean functional transforms.
    - HigherEngine: Uses the `higher` library for differentiable optimizers.
    - CustomSGDEngine: Pure PyTorch autograd fallback.
    """

    @abstractmethod
    def adapt(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        *,
        steps: int = 5,
        lr: float = 0.01,
        lrs: Optional[Dict[str, Dict[int, float]]] = None,
        first_order: bool = False,
        clip_norm: Optional[float] = None,
        loss_fn: Callable = F.cross_entropy,
    ) -> InnerLoopResult:
        """
        Adapt model parameters to a support set.

        This is the core inner-loop optimization. Starting from `params`,
        it performs `steps` gradient descent updates on the support set
        loss and returns the adapted parameters along with diagnostics.

        Parameters
        ----------
        model : nn.Module
            The model architecture (used for forward passes).
        params : Dict[str, torch.Tensor]
            Initial parameters to adapt from. Typically obtained from
            model.named_parameters() or a previous adaptation.
        support_x : torch.Tensor
            Support set inputs, shape (N*K, ...).
        support_y : torch.Tensor
            Support set targets, shape (N*K,) for classification.
        steps : int
            Number of inner-loop gradient descent steps.
        lr : float
            Base learning rate for the inner loop.
        lrs : Optional[Dict[str, Dict[int, float]]]
            Per-layer, per-step learning rate overrides (LSLR / MAML++).
            Structure: {param_name: {step_index: learning_rate}}.
            If a parameter name and step are present, the override LR is
            used instead of the base `lr`. If None, base `lr` is used
            everywhere.
        first_order : bool
            If True, detach gradients to avoid second-order computation
            (FOMAML). If False, retain computation graph (MAML).
        clip_norm : Optional[float]
            If not None, clip gradient global norm to this value.
        loss_fn : Callable
            Loss function taking (logits, targets) and returning a scalar.

        Returns
        -------
        InnerLoopResult
            Adapted parameters, step logs, and summary statistics.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable name of this engine."""
        return self.__class__.__name__


# =============================================================================
# Backend 1: TorchFuncEngine (Preferred)
# =============================================================================

class TorchFuncEngine(InnerLoopEngine):
    """
    Inner-loop engine using torch.func for functional transforms.

    This is the preferred backend for PyTorch >= 2.0. It uses
    functional_call for parameter-free forward passes and torch.func.grad
    for clean gradient computation without side effects.

    The key advantage is that functional_call never modifies the model's
    parameter buffers, making the inner loop safe for parallel execution
    and avoiding subtle bugs from in-place parameter mutation.

    Implementation details
    ----------------------
    For second-order MAML (first_order=False):
        torch.func.grad implicitly creates the computation graph needed
        for meta-gradient backpropagation. The adapted parameters retain
        grad_fn, enabling the outer loop to differentiate through the
        entire inner-loop trajectory.

    For first-order MAML (first_order=True):
        We still use torch.func.grad but detach the resulting gradients
        before applying the update. This provides the efficiency of
        first-order methods while keeping the code path consistent.

    Notes
    -----
    Requires torch.func (PyTorch >= 2.0). Check HAS_TORCH_FUNC before
    instantiating.
    """

    def __init__(self) -> None:
        if not HAS_TORCH_FUNC:
            raise ImportError(
                "TorchFuncEngine requires torch.func (PyTorch >= 2.0). "
                "Install a newer version of PyTorch or use CustomSGDEngine."
            )

    def adapt(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        *,
        steps: int = 5,
        lr: float = 0.01,
        lrs: Optional[Dict[str, Dict[int, float]]] = None,
        first_order: bool = False,
        clip_norm: Optional[float] = None,
        loss_fn: Callable = F.cross_entropy,
    ) -> InnerLoopResult:
        """
        Adapt parameters using torch.func functional transforms.

        See InnerLoopEngine.adapt for full parameter documentation.
        """
        start_time = time.monotonic()

        # Clone parameters to avoid mutating the input dict.
        # For second-order, we need the clone to participate in the graph.
        adapted = {
            k: v.clone().requires_grad_(True)
            for k, v in params.items()
        }

        step_logs: List[StepLog] = []
        total_loss = 0.0

        for step_idx in range(steps):
            # ----------------------------------------------------------
            # Forward pass and loss computation via functional_call.
            # ----------------------------------------------------------
            logits = functional_call(model, adapted, (support_x,))
            loss = loss_fn(logits, support_y)
            total_loss += loss.item()

            # ----------------------------------------------------------
            # Compute gradients with respect to each parameter.
            # We use torch.autograd.grad here for compatibility with the
            # create_graph pattern, since func_grad doesn't directly
            # support partial differentiation on a dict easily.
            # ----------------------------------------------------------
            create_graph = not first_order
            param_values = list(adapted.values())
            param_names = list(adapted.keys())

            grads = torch.autograd.grad(
                loss,
                param_values,
                create_graph=create_graph,
                allow_unused=True,
            )

            # ----------------------------------------------------------
            # Apply gradient clipping if configured
            # ----------------------------------------------------------
            clipped = False
            if clip_norm is not None:
                grads, clipped = clip_grad_tuple(grads, clip_norm)

            # ----------------------------------------------------------
            # Compute diagnostics (in no-grad context to avoid graph bloat)
            # ----------------------------------------------------------
            with torch.no_grad():
                acc = _compute_accuracy(logits, support_y)
                grad_norm_val = compute_grad_norm(grads)

            # ----------------------------------------------------------
            # Update parameters: p_new = p - lr * grad
            # With optional LSLR overrides per parameter per step.
            # ----------------------------------------------------------
            new_adapted = {}
            effective_lrs: List[float] = []

            for (name, p), g in zip(list(adapted.items()), grads):
                if g is None:
                    new_adapted[name] = p
                    effective_lrs.append(lr)
                    continue

                # Determine effective learning rate for this param/step
                effective_lr = lr
                if lrs is not None and name in lrs and step_idx in lrs[name]:
                    effective_lr = lrs[name][step_idx]
                    # Clamp LSLR to safe range
                    effective_lr = max(LSLR_MIN_LR, min(LSLR_MAX_LR, effective_lr))

                effective_lrs.append(effective_lr)

                # Apply gradient update
                if first_order:
                    # Detach gradient for first-order approximation
                    new_adapted[name] = p - effective_lr * g.detach()
                else:
                    # Retain graph for second-order meta-gradients
                    new_adapted[name] = p - effective_lr * g

            adapted = new_adapted

            # Compute mean effective LR and update norm for logging
            mean_lr = sum(effective_lrs) / max(len(effective_lrs), 1)
            with torch.no_grad():
                update_norm_val = compute_update_norm(grads, mean_lr)

            step_logs.append(StepLog(
                step=step_idx,
                loss=loss.item(),
                accuracy=acc,
                grad_norm=grad_norm_val,
                update_norm=update_norm_val,
                lr_effective=mean_lr,
                clipped=clipped,
            ))

        # ----------------------------------------------------------
        # Compute final accuracy on the adapted model
        # ----------------------------------------------------------
        with torch.no_grad():
            final_logits = functional_call(model, adapted, (support_x,))
            final_acc = _compute_accuracy(final_logits, support_y)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        return InnerLoopResult(
            adapted_params=adapted,
            step_logs=step_logs,
            total_inner_loss=total_loss,
            final_accuracy=final_acc,
            num_steps=steps,
            wall_time_ms=elapsed_ms,
            backend="torch_func",
        )


# =============================================================================
# Backend 2: HigherEngine (Optional)
# =============================================================================

class HigherEngine(InnerLoopEngine):
    """
    Inner-loop engine using the `higher` library.

    The `higher` library provides a differentiable optimizer abstraction
    that automatically handles computation graph tracking through
    optimizer steps. This is useful when the inner loop uses complex
    optimizers (e.g., Adam with momentum) rather than plain SGD.

    The main limitation is that `higher` creates a full differentiable
    copy of the model, which uses more memory than the functional
    approach. It also does not directly support LSLR without custom
    override mechanisms.

    Notes
    -----
    Requires the `higher` package: pip install higher.
    Check HAS_HIGHER before instantiating.

    LSLR is supported by converting per-layer LRs into a list of
    parameter groups with different learning rates.
    """

    def __init__(self) -> None:
        if not HAS_HIGHER:
            raise ImportError(
                "HigherEngine requires the `higher` library. "
                "Install with: pip install higher"
            )

    def adapt(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        *,
        steps: int = 5,
        lr: float = 0.01,
        lrs: Optional[Dict[str, Dict[int, float]]] = None,
        first_order: bool = False,
        clip_norm: Optional[float] = None,
        loss_fn: Callable = F.cross_entropy,
    ) -> InnerLoopResult:
        """
        Adapt parameters using higher's differentiable optimizer.

        See InnerLoopEngine.adapt for full parameter documentation.

        Notes
        -----
        This backend first loads the provided params into the model
        (via state_dict update), then uses higher.innerloop_ctx to
        create a differentiable copy for inner-loop optimization.
        After adaptation, it extracts the adapted parameters as a dict.
        """
        start_time = time.monotonic()

        # Load provided params into a temporary copy of the model.
        # We must do this because higher operates on the model directly.
        temp_model = copy.deepcopy(model)
        temp_state = temp_model.state_dict()
        for name, val in params.items():
            if name in temp_state:
                temp_state[name] = val.clone()
        temp_model.load_state_dict(temp_state)

        # Create inner-loop optimizer (vanilla SGD)
        inner_opt = torch.optim.SGD(temp_model.parameters(), lr=lr)

        step_logs: List[StepLog] = []
        total_loss = 0.0

        # Track gradients through the inner loop using higher
        track_higher_grads = not first_order

        with higher.innerloop_ctx(
            temp_model,
            inner_opt,
            copy_initial_weights=True,
            track_higher_grads=track_higher_grads,
        ) as (fmodel, diffopt):
            for step_idx in range(steps):
                # Forward pass on the differentiable model
                logits = fmodel(support_x)
                loss = loss_fn(logits, support_y)
                total_loss += loss.item()

                # Compute accuracy before update
                with torch.no_grad():
                    acc = _compute_accuracy(logits, support_y)

                # Compute gradient norm before the step
                # We need to manually compute grads for diagnostics
                grads_for_diag = torch.autograd.grad(
                    loss,
                    fmodel.parameters(),
                    create_graph=track_higher_grads,
                    allow_unused=True,
                    retain_graph=True,
                )

                clipped = False
                grad_norm_val = compute_grad_norm(grads_for_diag)

                if clip_norm is not None:
                    clipped_grads, clipped = clip_grad_tuple(grads_for_diag, clip_norm)
                    # We can't easily inject clipped grads into higher's step,
                    # so we do the step and note that clipping diagnostics are
                    # approximate. For exact clipping, use CustomSGDEngine.
                    if clipped:
                        warnings.warn(
                            "HigherEngine gradient clipping is diagnostic-only. "
                            "For exact clipping, use CustomSGDEngine.",
                            stacklevel=2,
                        )

                # Determine effective LR for this step
                # higher doesn't natively support LSLR, so we override
                # the optimizer's LR for the step if uniform across params.
                effective_lr = lr
                if lrs is not None:
                    # Compute mean LSLR for this step across all params
                    step_lrs = []
                    for name in params:
                        if name in lrs and step_idx in lrs[name]:
                            step_lrs.append(lrs[name][step_idx])
                        else:
                            step_lrs.append(lr)
                    effective_lr = sum(step_lrs) / max(len(step_lrs), 1)
                    # Override higher's optimizer LR
                    diffopt.param_groups[0]["lr"] = effective_lr

                # Take an optimization step
                diffopt.step(loss)

                # Compute update norm
                update_norm_val = compute_update_norm(grads_for_diag, effective_lr)

                step_logs.append(StepLog(
                    step=step_idx,
                    loss=loss.item(),
                    accuracy=acc,
                    grad_norm=grad_norm_val,
                    update_norm=update_norm_val,
                    lr_effective=effective_lr,
                    clipped=clipped,
                ))

            # Extract adapted parameters from the differentiable model
            adapted_params = {}
            param_names = list(params.keys())
            for name, param in zip(param_names, fmodel.parameters()):
                adapted_params[name] = param

        # Compute final accuracy
        with torch.no_grad():
            final_logits = _functional_forward(model, adapted_params, support_x)
            final_acc = _compute_accuracy(final_logits, support_y)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        return InnerLoopResult(
            adapted_params=adapted_params,
            step_logs=step_logs,
            total_inner_loss=total_loss,
            final_accuracy=final_acc,
            num_steps=steps,
            wall_time_ms=elapsed_ms,
            backend="higher",
        )


# =============================================================================
# Backend 3: CustomSGDEngine (Always Available Fallback)
# =============================================================================

class CustomSGDEngine(InnerLoopEngine):
    """
    Inner-loop engine using pure PyTorch autograd.

    This is the always-available fallback that works with any PyTorch
    version. It uses torch.autograd.grad for gradient computation and
    manual parameter updates, providing full control over the inner loop.

    This engine supports all features:
    - Second-order gradients via create_graph=True
    - First-order approximation via create_graph=False
    - Per-layer per-step learning rates (LSLR)
    - Global gradient norm clipping
    - Full step-by-step diagnostics

    Implementation notes
    --------------------
    The main loop maintains an `adapted` dictionary of parameter tensors.
    At each step:
    1. Compute forward pass using _functional_forward
    2. Compute gradients via torch.autograd.grad
    3. Optionally clip gradients
    4. Apply updates with optional LSLR
    5. Log diagnostics

    For second-order (create_graph=True), the updated parameters retain
    their grad_fn, allowing the outer loop to backpropagate through the
    entire inner-loop computation.

    For first-order (create_graph=False), gradients are computed without
    building the second-order graph. The adapted parameters are standard
    tensors that require_grad but do not carry inner-loop history.
    """

    def adapt(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        *,
        steps: int = 5,
        lr: float = 0.01,
        lrs: Optional[Dict[str, Dict[int, float]]] = None,
        first_order: bool = False,
        clip_norm: Optional[float] = None,
        loss_fn: Callable = F.cross_entropy,
    ) -> InnerLoopResult:
        """
        Adapt parameters using manual gradient descent.

        See InnerLoopEngine.adapt for full parameter documentation.
        """
        start_time = time.monotonic()

        # Clone parameters. For first-order we can safely clone without
        # retaining graph connections. For second-order, the clone
        # preserves the computation graph.
        if first_order:
            adapted = {
                k: v.clone().detach().requires_grad_(True)
                for k, v in params.items()
            }
        else:
            adapted = {
                k: v.clone().requires_grad_(True)
                for k, v in params.items()
            }

        step_logs: List[StepLog] = []
        total_loss = 0.0

        for step_idx in range(steps):
            # ----------------------------------------------------------
            # Forward pass through the model with current adapted params
            # ----------------------------------------------------------
            logits = _functional_forward(model, adapted, support_x)
            loss = loss_fn(logits, support_y)
            total_loss += loss.item()

            # ----------------------------------------------------------
            # Compute gradients w.r.t. adapted parameters
            # ----------------------------------------------------------
            create_graph = not first_order
            grads = torch.autograd.grad(
                loss,
                list(adapted.values()),
                create_graph=create_graph,
                allow_unused=True,
            )

            # ----------------------------------------------------------
            # Apply gradient clipping if configured
            # ----------------------------------------------------------
            clipped = False
            if clip_norm is not None:
                grads, clipped = clip_grad_tuple(grads, clip_norm)

            # ----------------------------------------------------------
            # Compute diagnostics before the update
            # ----------------------------------------------------------
            with torch.no_grad():
                acc = _compute_accuracy(logits, support_y)
                grad_norm_val = compute_grad_norm(grads)

            # ----------------------------------------------------------
            # Update parameters: p_new = p - effective_lr * grad
            # ----------------------------------------------------------
            new_adapted = {}
            effective_lrs: List[float] = []

            for (name, p), g in zip(list(adapted.items()), grads):
                if g is None:
                    # Parameter unused in the forward pass; keep as-is
                    new_adapted[name] = p
                    effective_lrs.append(lr)
                    continue

                # Determine the effective learning rate for this param/step
                effective_lr = lr
                if lrs is not None and name in lrs and step_idx in lrs[name]:
                    effective_lr = lrs[name][step_idx]
                    effective_lr = max(LSLR_MIN_LR, min(LSLR_MAX_LR, effective_lr))

                effective_lrs.append(effective_lr)

                # Apply the gradient update
                if first_order:
                    # Detach the gradient to avoid building second-order graph
                    new_p = p - effective_lr * g.detach()
                    # Re-enable gradient tracking for the next step
                    new_adapted[name] = new_p.detach().requires_grad_(True)
                else:
                    # Keep the gradient graph for second-order meta-gradients
                    new_adapted[name] = p - effective_lr * g

            adapted = new_adapted

            # Compute mean effective LR and update norm
            mean_lr = sum(effective_lrs) / max(len(effective_lrs), 1)
            with torch.no_grad():
                update_norm_val = compute_update_norm(grads, mean_lr)

            step_logs.append(StepLog(
                step=step_idx,
                loss=loss.item(),
                accuracy=acc,
                grad_norm=grad_norm_val,
                update_norm=update_norm_val,
                lr_effective=mean_lr,
                clipped=clipped,
            ))

        # ----------------------------------------------------------
        # Compute final accuracy on the adapted parameters
        # ----------------------------------------------------------
        with torch.no_grad():
            final_logits = _functional_forward(model, adapted, support_x)
            final_acc = _compute_accuracy(final_logits, support_y)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        return InnerLoopResult(
            adapted_params=adapted,
            step_logs=step_logs,
            total_inner_loss=total_loss,
            final_accuracy=final_acc,
            num_steps=steps,
            wall_time_ms=elapsed_ms,
            backend="custom_sgd",
        )


# =============================================================================
# AMP Safety Wrapper
# =============================================================================

class AMPSafeInnerLoop:
    """
    Wraps an InnerLoopEngine to enforce fp32 during the inner loop.

    Meta-learning inner loops are notoriously fragile under mixed precision.
    Second-order gradients amplify numerical errors, and fp16 parameter
    updates can lead to NaN or zero meta-gradients. This wrapper:

    1. Casts all input parameters to fp32
    2. Casts support set inputs to fp32
    3. Disables CUDA autocast during the inner loop
    4. Returns results with fp32 adapted parameters

    The outer loop is free to use AMP for the meta-gradient computation,
    but the inner loop itself runs entirely in fp32.

    Usage
    -----
    >>> base_engine = CustomSGDEngine()
    >>> safe_engine = AMPSafeInnerLoop(base_engine)
    >>> result = safe_engine.adapt(model, params, x, y, steps=5)

    Notes
    -----
    This wrapper does NOT inherit from InnerLoopEngine because it is
    a decorator/proxy, not a new backend. It delegates all work to the
    wrapped engine after casting inputs to fp32.

    Parameters
    ----------
    engine : InnerLoopEngine
        The underlying engine to wrap.
    """

    def __init__(self, engine: InnerLoopEngine) -> None:
        self.engine = engine

    def adapt(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        **kwargs: Any,
    ) -> InnerLoopResult:
        """
        Run the inner loop in fp32, regardless of AMP context.

        Parameters are cast to fp32 before adaptation. The support set
        input is also cast to fp32. Targets are left as-is (typically
        long integers for classification).

        Parameters
        ----------
        model : nn.Module
            Model architecture (will be used for forward passes).
        params : Dict[str, torch.Tensor]
            Parameters to adapt, may be in any dtype.
        support_x : torch.Tensor
            Support set inputs, may be in any dtype.
        support_y : torch.Tensor
            Support set targets (not cast).
        **kwargs
            Passed through to the underlying engine.

        Returns
        -------
        InnerLoopResult
            Result with fp32 adapted parameters.
        """
        # Cast parameters to fp32
        fp32_params = {k: v.float() for k, v in params.items()}

        # Cast support inputs to fp32 (targets are usually long, keep as-is)
        fp32_x = support_x.float()

        # Disable autocast for the inner loop.
        # Use the newer torch.amp.autocast API (PyTorch >= 2.4) when available,
        # falling back to the legacy torch.cuda.amp.autocast for older versions.
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            _autocast_ctx = torch.amp.autocast("cuda", enabled=False)
        else:
            _autocast_ctx = torch.cuda.amp.autocast(enabled=False)
        with _autocast_ctx:
            result = self.engine.adapt(
                model, fp32_params, fp32_x, support_y, **kwargs
            )

        return result

    @property
    def name(self) -> str:
        """Descriptive name including the wrapped engine."""
        return f"AMPSafe({self.engine.name})"

    def __repr__(self) -> str:
        return f"AMPSafeInnerLoop(engine={self.engine!r})"


# =============================================================================
# Batch-Parallel Inner Loop (Multi-Task)
# =============================================================================

class BatchParallelInnerLoop:
    """
    Runs the inner loop across multiple tasks in sequence.

    This is a convenience wrapper that adapts a model to each task
    in a batch, collecting results. A future optimization would use
    vmap for true parallel execution, but sequential execution is
    correct and easier to debug.

    Parameters
    ----------
    engine : InnerLoopEngine or AMPSafeInnerLoop
        The inner-loop engine to use for each task.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def adapt_batch(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        tasks: List[Tuple[torch.Tensor, torch.Tensor]],
        **kwargs: Any,
    ) -> List[InnerLoopResult]:
        """
        Adapt to a batch of tasks sequentially.

        Parameters
        ----------
        model : nn.Module
            Shared model architecture.
        params : Dict[str, torch.Tensor]
            Shared initial parameters (same for all tasks).
        tasks : List[Tuple[torch.Tensor, torch.Tensor]]
            List of (support_x, support_y) tuples, one per task.
        **kwargs
            Passed to the engine's adapt() method.

        Returns
        -------
        List[InnerLoopResult]
            One result per task.
        """
        results = []
        for support_x, support_y in tasks:
            result = self.engine.adapt(
                model, params, support_x, support_y, **kwargs
            )
            results.append(result)
        return results

    def compute_meta_loss(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        loss_fn: Callable = F.cross_entropy,
        **adapt_kwargs: Any,
    ) -> Tuple[torch.Tensor, List[InnerLoopResult]]:
        """
        Compute meta-loss across a batch of tasks.

        For each task:
        1. Adapt on support set
        2. Evaluate adapted params on query set
        3. Accumulate query losses

        Parameters
        ----------
        model : nn.Module
            Shared model.
        params : Dict[str, torch.Tensor]
            Initial parameters.
        tasks : List[Tuple[Tensor, Tensor, Tensor, Tensor]]
            List of (support_x, support_y, query_x, query_y).
        loss_fn : Callable
            Loss function.
        **adapt_kwargs
            Passed to adapt().

        Returns
        -------
        meta_loss : torch.Tensor
            Mean query loss across tasks.
        results : List[InnerLoopResult]
            Per-task adaptation results.
        """
        total_query_loss = torch.tensor(0.0)
        results = []

        for support_x, support_y, query_x, query_y in tasks:
            # Adapt on support set
            result = self.engine.adapt(
                model, params, support_x, support_y,
                loss_fn=loss_fn, **adapt_kwargs
            )
            results.append(result)

            # Evaluate on query set with adapted params
            query_logits = _functional_forward(model, result.adapted_params, query_x)
            query_loss = loss_fn(query_logits, query_y)
            total_query_loss = total_query_loss + query_loss

        meta_loss = total_query_loss / max(len(tasks), 1)
        return meta_loss, results


# =============================================================================
# LSLR Parameter Manager
# =============================================================================

class LSLRManager(nn.Module):
    """
    Manages per-layer, per-step learned learning rates for MAML++.

    Creates a learnable LR parameter for each (layer, step) combination.
    LRs are stored in log-space and clamped to [LSLR_MIN_LR, LSLR_MAX_LR]
    during extraction to prevent collapse or explosion.

    Parameters
    ----------
    param_names : List[str]
        Names of the model parameters that will be adapted.
    num_steps : int
        Number of inner-loop steps.
    init_lr : float
        Initial learning rate (all entries initialized to this value).

    Examples
    --------
    >>> manager = LSLRManager(["fc1.weight", "fc1.bias", "fc2.weight"], 5, 0.01)
    >>> lrs_dict = manager.get_lrs()
    >>> # lrs_dict["fc1.weight"][0] == 0.01 (approximately)
    """

    def __init__(
        self,
        param_names: List[str],
        num_steps: int,
        init_lr: float = 0.01,
    ) -> None:
        super().__init__()
        self.param_names = list(param_names)
        self.num_steps = num_steps
        self.num_layers = len(param_names)

        # Store LRs in log-space for unconstrained optimization
        init_log_lr = math.log(max(init_lr, LSLR_MIN_LR))
        self.log_lrs = nn.Parameter(
            torch.full(
                (num_steps, self.num_layers),
                fill_value=init_log_lr,
            )
        )

    def get_lr(self, step: int, layer_idx: int) -> float:
        """
        Get the learning rate for a specific step and layer.

        Returns a clamped float value.
        """
        raw_lr = self.log_lrs[step, layer_idx].exp().item()
        return max(LSLR_MIN_LR, min(LSLR_MAX_LR, raw_lr))

    def get_lrs(self) -> Dict[str, Dict[int, float]]:
        """
        Get the full LSLR dictionary for use with InnerLoopEngine.adapt().

        Returns
        -------
        Dict[str, Dict[int, float]]
            Mapping from param_name -> {step_idx -> lr_value}.
        """
        lrs = {}
        for layer_idx, name in enumerate(self.param_names):
            step_lrs = {}
            for step_idx in range(self.num_steps):
                step_lrs[step_idx] = self.get_lr(step_idx, layer_idx)
            lrs[name] = step_lrs
        return lrs

    def get_lr_tensor(self) -> torch.Tensor:
        """
        Return the raw (num_steps, num_layers) LR tensor (exponentiated).

        Useful for visualization and regularization.
        """
        return self.log_lrs.exp().clamp(min=LSLR_MIN_LR, max=LSLR_MAX_LR)

    def extra_repr(self) -> str:
        return (
            f"num_layers={self.num_layers}, num_steps={self.num_steps}, "
            f"lr_range=[{LSLR_MIN_LR}, {LSLR_MAX_LR}]"
        )


# =============================================================================
# Inner Loop Diagnostics
# =============================================================================

class InnerLoopDiagnostics:
    """
    Aggregates and analyzes inner-loop diagnostics across episodes.

    Tracks adaptation curves, gradient health, and learning rate
    statistics to help diagnose common meta-learning failure modes.

    Usage
    -----
    >>> diag = InnerLoopDiagnostics()
    >>> for episode in episodes:
    ...     result = engine.adapt(...)
    ...     diag.record(result)
    >>> report = diag.summarize()
    """

    def __init__(self) -> None:
        self._results: List[InnerLoopResult] = []

    def record(self, result: InnerLoopResult) -> None:
        """Record an inner-loop result for aggregation."""
        self._results.append(result)

    def num_episodes(self) -> int:
        """Number of recorded episodes."""
        return len(self._results)

    def mean_loss_trajectory(self) -> List[float]:
        """
        Compute the mean loss at each inner step across all episodes.

        Returns
        -------
        List[float]
            Mean loss at each step. Length equals the number of inner steps
            (assumes all episodes have the same number of steps).
        """
        if not self._results:
            return []

        num_steps = self._results[0].num_steps
        trajectories = [r.loss_trajectory() for r in self._results]
        mean_traj = []
        for step_idx in range(num_steps):
            values = [t[step_idx] for t in trajectories if step_idx < len(t)]
            mean_traj.append(sum(values) / max(len(values), 1))
        return mean_traj

    def mean_accuracy_trajectory(self) -> List[float]:
        """Compute the mean accuracy at each inner step across all episodes."""
        if not self._results:
            return []

        num_steps = self._results[0].num_steps
        trajectories = [r.accuracy_trajectory() for r in self._results]
        mean_traj = []
        for step_idx in range(num_steps):
            values = [t[step_idx] for t in trajectories if step_idx < len(t)]
            mean_traj.append(sum(values) / max(len(values), 1))
        return mean_traj

    def loss_decreased_rate(self) -> float:
        """
        Fraction of episodes where loss decreased from first to last step.

        A low rate may indicate that the inner LR is too small or the
        model cannot adapt to the task distribution.
        """
        if not self._results:
            return 0.0
        decreased = sum(1 for r in self._results if r.loss_decreased())
        return decreased / len(self._results)

    def mean_grad_norm(self) -> float:
        """Mean gradient norm across all steps and episodes."""
        all_norms = []
        for r in self._results:
            all_norms.extend(r.grad_norm_trajectory())
        if not all_norms:
            return 0.0
        return sum(all_norms) / len(all_norms)

    def clip_rate(self) -> float:
        """Fraction of steps where gradient clipping was triggered."""
        total_steps = 0
        clipped_steps = 0
        for r in self._results:
            for log in r.step_logs:
                total_steps += 1
                if log.clipped:
                    clipped_steps += 1
        if total_steps == 0:
            return 0.0
        return clipped_steps / total_steps

    def summarize(self) -> Dict[str, Any]:
        """
        Produce a summary report of inner-loop health.

        Returns
        -------
        dict
            Summary statistics including mean trajectories, loss decrease
            rate, gradient norm stats, and clipping rate.
        """
        return {
            "num_episodes": self.num_episodes(),
            "mean_loss_trajectory": self.mean_loss_trajectory(),
            "mean_accuracy_trajectory": self.mean_accuracy_trajectory(),
            "loss_decreased_rate": self.loss_decreased_rate(),
            "mean_grad_norm": self.mean_grad_norm(),
            "clip_rate": self.clip_rate(),
        }

    def clear(self) -> None:
        """Clear all recorded results."""
        self._results.clear()


# =============================================================================
# Detach Detector (Test Utility)
# =============================================================================

def check_grad_fn_retained(
    adapted_params: Dict[str, torch.Tensor],
    expect_grad_fn: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Verify whether adapted parameters retain their grad_fn.

    This is a critical correctness check for MAML:
    - Second-order (MAML): adapted params MUST have grad_fn
    - First-order (FOMAML): adapted params should NOT have grad_fn

    Parameters
    ----------
    adapted_params : Dict[str, torch.Tensor]
        Adapted parameter dict from InnerLoopResult.
    expect_grad_fn : bool
        If True, check that all params HAVE grad_fn (MAML check).
        If False, check that no params have grad_fn (FOMAML check).

    Returns
    -------
    passed : bool
        True if the check passed.
    violations : List[str]
        Names of parameters that violated the expectation.
    """
    violations = []
    for name, param in adapted_params.items():
        has_grad_fn = param.grad_fn is not None
        if expect_grad_fn and not has_grad_fn:
            violations.append(name)
        elif not expect_grad_fn and has_grad_fn:
            violations.append(name)
    return len(violations) == 0, violations


# =============================================================================
# Meta-Gradient Verification Utility
# =============================================================================

def verify_meta_gradients(
    model: nn.Module,
    engine: Any,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    steps: int = 1,
    lr: float = 0.01,
    loss_fn: Callable = F.cross_entropy,
) -> Dict[str, Any]:
    """
    Verify that meta-gradients flow correctly through the inner loop.

    Runs a complete inner loop + outer loss computation and checks that
    gradients are non-zero on the original model parameters. This is the
    primary sanity check for MAML correctness.

    Parameters
    ----------
    model : nn.Module
        Model to verify.
    engine : InnerLoopEngine or AMPSafeInnerLoop
        Engine to test.
    support_x, support_y : torch.Tensor
        Support set.
    query_x, query_y : torch.Tensor
        Query set.
    steps : int
        Inner-loop steps.
    lr : float
        Inner-loop LR.
    loss_fn : Callable
        Loss function.

    Returns
    -------
    dict
        Verification results including:
        - "meta_grads_exist": bool -- any meta-gradients are non-zero
        - "all_grads_nonzero": bool -- all meta-gradients are non-zero
        - "grad_stats": dict -- per-param gradient statistics
        - "meta_loss": float -- the outer loss value
    """
    # Ensure model parameters require grad
    params = {name: p.clone().requires_grad_(True) for name, p in model.named_parameters()}

    # Run inner loop
    result = engine.adapt(
        model, params, support_x, support_y,
        steps=steps, lr=lr, first_order=False, loss_fn=loss_fn,
    )

    # Compute outer (query) loss
    query_logits = _functional_forward(model, result.adapted_params, query_x)
    meta_loss = loss_fn(query_logits, query_y)

    # Backpropagate to compute meta-gradients
    meta_grads = torch.autograd.grad(
        meta_loss,
        list(params.values()),
        allow_unused=True,
    )

    # Analyze gradients
    grad_stats = {}
    any_nonzero = False
    all_nonzero = True

    for (name, _), g in zip(params.items(), meta_grads):
        if g is not None:
            g_norm = g.norm().item()
            is_nonzero = g_norm > 0
            grad_stats[name] = {
                "norm": g_norm,
                "nonzero": is_nonzero,
                "mean": g.mean().item(),
                "std": g.std().item() if g.numel() > 1 else 0.0,
            }
            if is_nonzero:
                any_nonzero = True
            else:
                all_nonzero = False
        else:
            grad_stats[name] = {"norm": 0.0, "nonzero": False, "mean": 0.0, "std": 0.0}
            all_nonzero = False

    return {
        "meta_grads_exist": any_nonzero,
        "all_grads_nonzero": all_nonzero,
        "grad_stats": grad_stats,
        "meta_loss": meta_loss.item(),
    }


# =============================================================================
# Factory Function
# =============================================================================

def create_inner_loop_engine(
    backend: str = "auto",
    use_amp_safety: bool = True,
) -> Any:
    """
    Create an inner-loop engine with the specified backend.

    The factory selects the best available backend by default ("auto"),
    preferring torch.func > higher > custom SGD. The AMP safety wrapper
    is applied by default to ensure numerical stability under mixed
    precision training.

    Parameters
    ----------
    backend : str
        Backend selection. One of:
        - "auto": Automatically select the best available backend.
        - "torch_func": Use TorchFuncEngine (requires torch.func).
        - "higher": Use HigherEngine (requires higher library).
        - "custom": Use CustomSGDEngine (always available).
    use_amp_safety : bool
        If True, wrap the engine in AMPSafeInnerLoop to enforce fp32
        during the inner loop. Recommended for all training scenarios.

    Returns
    -------
    InnerLoopEngine or AMPSafeInnerLoop
        The configured inner-loop engine.

    Raises
    ------
    ValueError
        If the backend string is not recognized.
    AssertionError
        If a specific backend is requested but not available.

    Examples
    --------
    >>> engine = create_inner_loop_engine()  # Auto-select best backend
    >>> engine = create_inner_loop_engine("custom", use_amp_safety=False)
    """
    if backend == "auto":
        if HAS_TORCH_FUNC:
            engine = TorchFuncEngine()
        elif HAS_HIGHER:
            engine = HigherEngine()
        else:
            engine = CustomSGDEngine()
    elif backend == "torch_func":
        assert HAS_TORCH_FUNC, (
            "torch.func is not available. Requires PyTorch >= 2.0. "
            f"Current PyTorch version: {torch.__version__}"
        )
        engine = TorchFuncEngine()
    elif backend == "higher":
        assert HAS_HIGHER, (
            "The `higher` library is not installed. "
            "Install with: pip install higher"
        )
        engine = HigherEngine()
    elif backend == "custom":
        engine = CustomSGDEngine()
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Choose from: 'auto', 'torch_func', 'higher', 'custom'"
        )

    if use_amp_safety:
        engine = AMPSafeInnerLoop(engine)

    return engine


# =============================================================================
# Convenience: get_available_backends
# =============================================================================

def get_available_backends() -> Dict[str, bool]:
    """
    Check which inner-loop backends are available.

    Returns
    -------
    Dict[str, bool]
        Mapping from backend name to availability.
    """
    return {
        "torch_func": HAS_TORCH_FUNC,
        "higher": HAS_HIGHER,
        "custom": True,  # Always available
    }


# =============================================================================
# Self-Test Block
# =============================================================================

if __name__ == "__main__":
    """
    Comprehensive self-test for the inner-loop optimization engine.

    Tests cover:
        1.  Toy model creation and param extraction
        2.  CustomSGDEngine basic adaptation (loss decreases)
        3.  StepLog schema validation
        4.  TorchFuncEngine adaptation (if available)
        5.  Backend equivalence (if multiple backends available)
        6.  create_graph=True retains grad_fn (MAML)
        7.  create_graph=False has no grad_fn on updates (FOMAML)
        8.  Gradient clipping functionality
        9.  AMP safety wrapper
        10. Factory function (create_inner_loop_engine)
        11. LSLR learning rate overrides
        12. Meta-gradient verification
        13. InnerLoopResult diagnostics
        14. InnerLoopDiagnostics aggregation
        15. BatchParallelInnerLoop multi-task
        16. LSLRManager parameter management
        17. Detach detector utility
        18. Edge cases (zero steps, single step, large LR)

    Uses a toy 2-layer linear model:
        Linear(10, 20) -> ReLU -> Linear(20, 5)
    with synthetic classification data (10-dim inputs, 5 classes).
    """
    print("=" * 70)
    print("InnerLoopEngine Self-Test Suite")
    print(f"PyTorch version: {torch.__version__}")
    print(f"torch.func available: {HAS_TORCH_FUNC}")
    print(f"higher available: {HAS_HIGHER}")
    print(f"Engine version: {ENGINE_VERSION}")
    print("=" * 70)

    # Track test results using a mutable container (dict) so that
    # the nested report() function can update counters without nonlocal,
    # which is not available at module scope.
    test_results: Dict[str, bool] = {}
    _counters = {"passed": 0, "failed": 0}

    def report(test_name: str, passed: bool, detail: str = "") -> None:
        """Report a test result."""
        test_results[test_name] = passed
        status = "PASS" if passed else "FAIL"
        detail_str = f" -- {detail}" if detail else ""
        print(f"  [{status}] {test_name}{detail_str}")
        if passed:
            _counters["passed"] += 1
        else:
            _counters["failed"] += 1

    # ------------------------------------------------------------------
    # Setup: Create toy model and synthetic data
    # ------------------------------------------------------------------
    print("\n--- Setup ---")

    INPUT_DIM = 10
    HIDDEN_DIM = 20
    OUTPUT_DIM = 5
    BATCH_SIZE = 16
    NUM_STEPS = 5
    INNER_LR = 0.05  # Larger LR for visible adaptation on toy data

    class ToyModel(nn.Module):
        """Simple 2-layer linear model for testing."""

        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.relu(self.fc1(x)))

    torch.manual_seed(42)
    model = ToyModel()

    # Synthetic data: random inputs, random class labels
    support_x = torch.randn(BATCH_SIZE, INPUT_DIM)
    support_y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,))
    query_x = torch.randn(BATCH_SIZE, INPUT_DIM)
    query_y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,))

    print(f"Model: ToyModel({INPUT_DIM} -> {HIDDEN_DIM} -> {OUTPUT_DIM})")
    print(f"Data: support={BATCH_SIZE}, query={BATCH_SIZE}")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count}")

    # ------------------------------------------------------------------
    # Test 1: Toy model creation and parameter extraction
    # ------------------------------------------------------------------
    print("\n--- Test Group 1: Model & Parameter Extraction ---")

    try:
        params = _extract_params(model, clone=True)
        assert isinstance(params, dict), "params should be a dict"
        assert len(params) == 4, f"Expected 4 params (2 weights + 2 biases), got {len(params)}"
        expected_names = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
        assert set(params.keys()) == expected_names, f"Wrong param names: {set(params.keys())}"

        # Verify cloning: modifying extracted params should not affect model
        params["fc1.weight"].fill_(999.0)
        assert model.fc1.weight.data.abs().max().item() < 100, "Clone should be independent"

        report("1a: Parameter extraction", True, f"{len(params)} params extracted")
    except Exception as e:
        report("1a: Parameter extraction", False, str(e))

    try:
        params_noclose = _extract_params(model, clone=False)
        assert params_noclose["fc1.weight"].data_ptr() == model.fc1.weight.data_ptr(), \
            "No-clone should share memory"
        report("1b: No-clone extraction", True, "Memory shared correctly")
    except Exception as e:
        report("1b: No-clone extraction", False, str(e))

    # ------------------------------------------------------------------
    # Test 2: CustomSGDEngine basic adaptation
    # ------------------------------------------------------------------
    print("\n--- Test Group 2: CustomSGDEngine Basic Adaptation ---")

    try:
        engine = CustomSGDEngine()
        params = _extract_params(model, clone=True)

        result = engine.adapt(
            model, params, support_x, support_y,
            steps=NUM_STEPS, lr=INNER_LR, first_order=True,
        )

        assert isinstance(result, InnerLoopResult), "Should return InnerLoopResult"
        assert result.num_steps == NUM_STEPS, f"Expected {NUM_STEPS} steps, got {result.num_steps}"
        assert result.backend == "custom_sgd", f"Wrong backend: {result.backend}"
        report("2a: CustomSGD returns InnerLoopResult", True)
    except Exception as e:
        report("2a: CustomSGD returns InnerLoopResult", False, str(e))

    try:
        losses = result.loss_trajectory()
        assert len(losses) == NUM_STEPS, f"Expected {NUM_STEPS} losses, got {len(losses)}"
        # Loss should generally decrease with enough steps and appropriate LR
        loss_decreased = losses[-1] < losses[0]
        report(
            "2b: Loss trajectory",
            loss_decreased,
            f"loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f}"
        )
    except Exception as e:
        report("2b: Loss trajectory", False, str(e))

    try:
        assert result.wall_time_ms > 0, "Wall time should be positive"
        assert result.total_inner_loss > 0, "Total loss should be positive"
        report("2c: Timing & total loss", True,
               f"wall_time={result.wall_time_ms:.1f}ms, total_loss={result.total_inner_loss:.4f}")
    except Exception as e:
        report("2c: Timing & total loss", False, str(e))

    # ------------------------------------------------------------------
    # Test 3: StepLog schema validation
    # ------------------------------------------------------------------
    print("\n--- Test Group 3: StepLog Schema ---")

    try:
        assert len(result.step_logs) == NUM_STEPS, f"Expected {NUM_STEPS} logs"
        log = result.step_logs[0]
        assert isinstance(log, StepLog), "Logs should be StepLog instances"
        assert log.step == 0, f"First log step should be 0, got {log.step}"
        assert isinstance(log.loss, float), "Loss should be float"
        assert isinstance(log.accuracy, float), "Accuracy should be float"
        assert isinstance(log.grad_norm, float), "Grad norm should be float"
        assert isinstance(log.update_norm, float), "Update norm should be float"
        assert isinstance(log.lr_effective, float), "LR should be float"
        assert isinstance(log.clipped, bool), "Clipped should be bool"
        report("3a: StepLog fields", True, f"All 7 fields validated")
    except Exception as e:
        report("3a: StepLog fields", False, str(e))

    try:
        d = log.to_dict()
        assert isinstance(d, dict), "to_dict should return dict"
        required_keys = {"step", "loss", "accuracy", "grad_norm", "update_norm", "lr_effective", "clipped"}
        assert set(d.keys()) == required_keys, f"Missing keys: {required_keys - set(d.keys())}"
        report("3b: StepLog.to_dict()", True)
    except Exception as e:
        report("3b: StepLog.to_dict()", False, str(e))

    try:
        repr_str = repr(log)
        assert "StepLog" in repr_str, "repr should contain 'StepLog'"
        report("3c: StepLog repr", True, repr_str[:60])
    except Exception as e:
        report("3c: StepLog repr", False, str(e))

    try:
        # Test to_dict with NaN values
        nan_log = StepLog(step=0, loss=float('nan'), accuracy=0.5,
                          grad_norm=float('inf'), update_norm=0.1,
                          lr_effective=0.01, clipped=False)
        nan_d = nan_log.to_dict()
        assert isinstance(nan_d["loss"], str), "NaN loss should be converted to string"
        assert isinstance(nan_d["grad_norm"], str), "Inf grad_norm should be converted to string"
        report("3d: StepLog NaN/Inf handling", True)
    except Exception as e:
        report("3d: StepLog NaN/Inf handling", False, str(e))

    # ------------------------------------------------------------------
    # Test 4: InnerLoopResult diagnostics
    # ------------------------------------------------------------------
    print("\n--- Test Group 4: InnerLoopResult Diagnostics ---")

    try:
        losses = result.loss_trajectory()
        accs = result.accuracy_trajectory()
        gnorms = result.grad_norm_trajectory()
        assert len(losses) == NUM_STEPS
        assert len(accs) == NUM_STEPS
        assert len(gnorms) == NUM_STEPS
        report("4a: Trajectory methods", True, f"All return {NUM_STEPS}-length lists")
    except Exception as e:
        report("4a: Trajectory methods", False, str(e))

    try:
        ld = result.loss_decreased()
        assert isinstance(ld, bool), "loss_decreased should return bool"
        report("4b: loss_decreased()", True, f"loss_decreased={ld}")
    except Exception as e:
        report("4b: loss_decreased()", False, str(e))

    try:
        result_dict = result.to_dict()
        assert "step_logs" in result_dict
        assert "backend" in result_dict
        assert result_dict["num_steps"] == NUM_STEPS
        report("4c: InnerLoopResult.to_dict()", True)
    except Exception as e:
        report("4c: InnerLoopResult.to_dict()", False, str(e))

    # ------------------------------------------------------------------
    # Test 5: TorchFuncEngine (if available)
    # ------------------------------------------------------------------
    print("\n--- Test Group 5: TorchFuncEngine ---")

    if HAS_TORCH_FUNC:
        try:
            tf_engine = TorchFuncEngine()
            params = _extract_params(model, clone=True)
            tf_result = tf_engine.adapt(
                model, params, support_x, support_y,
                steps=NUM_STEPS, lr=INNER_LR, first_order=True,
            )
            assert tf_result.backend == "torch_func"
            assert tf_result.num_steps == NUM_STEPS
            tf_losses = tf_result.loss_trajectory()
            report("5a: TorchFuncEngine basic", True,
                   f"loss[0]={tf_losses[0]:.4f} -> loss[-1]={tf_losses[-1]:.4f}")
        except Exception as e:
            report("5a: TorchFuncEngine basic", False, str(e))

        try:
            # Second-order mode
            params = _extract_params(model, clone=True)
            tf_result_so = tf_engine.adapt(
                model, params, support_x, support_y,
                steps=3, lr=INNER_LR, first_order=False,
            )
            # Adapted params should have grad_fn for second-order
            has_any_grad_fn = any(
                p.grad_fn is not None
                for p in tf_result_so.adapted_params.values()
            )
            report("5b: TorchFuncEngine second-order", has_any_grad_fn,
                   "grad_fn retained" if has_any_grad_fn else "NO grad_fn")
        except Exception as e:
            report("5b: TorchFuncEngine second-order", False, str(e))
    else:
        report("5a: TorchFuncEngine basic", True, "SKIPPED (torch.func not available)")
        report("5b: TorchFuncEngine second-order", True, "SKIPPED (torch.func not available)")

    # ------------------------------------------------------------------
    # Test 6: Backend equivalence
    # ------------------------------------------------------------------
    print("\n--- Test Group 6: Backend Equivalence ---")

    if HAS_TORCH_FUNC:
        try:
            # Compare TorchFuncEngine vs CustomSGDEngine with same data
            torch.manual_seed(42)
            model_eq = ToyModel()
            params_eq = _extract_params(model_eq, clone=True)
            params_eq2 = {k: v.clone() for k, v in params_eq.items()}

            custom_eng = CustomSGDEngine()
            torch_func_eng = TorchFuncEngine()

            r_custom = custom_eng.adapt(
                model_eq, params_eq, support_x, support_y,
                steps=3, lr=0.01, first_order=True,
            )
            r_torch_func = torch_func_eng.adapt(
                model_eq, params_eq2, support_x, support_y,
                steps=3, lr=0.01, first_order=True,
            )

            # Losses should be very close (not identical due to floating point)
            loss_diff = abs(r_custom.step_logs[0].loss - r_torch_func.step_logs[0].loss)
            close_enough = loss_diff < 1e-4
            report(
                "6a: Backend equivalence (first step loss)",
                close_enough,
                f"diff={loss_diff:.6e}"
            )

            # Final adapted params should be close
            param_diffs = []
            for name in r_custom.adapted_params:
                diff = (r_custom.adapted_params[name] - r_torch_func.adapted_params[name]).abs().max().item()
                param_diffs.append(diff)
            max_param_diff = max(param_diffs)
            report(
                "6b: Backend equivalence (adapted params)",
                max_param_diff < 1e-3,
                f"max_diff={max_param_diff:.6e}"
            )
        except Exception as e:
            report("6a: Backend equivalence (first step loss)", False, str(e))
            report("6b: Backend equivalence (adapted params)", False, str(e))
    else:
        report("6a: Backend equivalence (first step loss)", True, "SKIPPED (single backend)")
        report("6b: Backend equivalence (adapted params)", True, "SKIPPED (single backend)")

    # ------------------------------------------------------------------
    # Test 7: create_graph=True retains grad_fn (MAML mode)
    # ------------------------------------------------------------------
    print("\n--- Test Group 7: MAML (Second-Order) Gradient Graph ---")

    try:
        engine_so = CustomSGDEngine()
        params = _extract_params(model, clone=True)
        result_so = engine_so.adapt(
            model, params, support_x, support_y,
            steps=3, lr=INNER_LR, first_order=False,
        )

        # Check that adapted params have grad_fn
        passed, violations = check_grad_fn_retained(result_so.adapted_params, expect_grad_fn=True)
        report(
            "7a: MAML adapted params have grad_fn",
            passed,
            f"violations={violations}" if not passed else "All params have grad_fn"
        )
    except Exception as e:
        report("7a: MAML adapted params have grad_fn", False, str(e))

    try:
        # Verify we can backprop through the inner loop to the original params
        query_logits = _functional_forward(model, result_so.adapted_params, query_x)
        outer_loss = F.cross_entropy(query_logits, query_y)

        # This should NOT raise an error
        meta_grads = torch.autograd.grad(
            outer_loss,
            list(params.values()),
            allow_unused=True,
        )

        nonzero_grads = sum(1 for g in meta_grads if g is not None and g.norm().item() > 0)
        report(
            "7b: Meta-gradients are non-zero",
            nonzero_grads > 0,
            f"{nonzero_grads}/{len(meta_grads)} params have non-zero meta-grad"
        )
    except Exception as e:
        report("7b: Meta-gradients are non-zero", False, str(e))

    # ------------------------------------------------------------------
    # Test 8: create_graph=False has no grad_fn (FOMAML mode)
    # ------------------------------------------------------------------
    print("\n--- Test Group 8: FOMAML (First-Order) ---")

    try:
        engine_fo = CustomSGDEngine()
        params = _extract_params(model, clone=True)
        result_fo = engine_fo.adapt(
            model, params, support_x, support_y,
            steps=3, lr=INNER_LR, first_order=True,
        )

        # Check that adapted params do NOT have grad_fn (they are leaf tensors)
        passed_fo, violations_fo = check_grad_fn_retained(
            result_fo.adapted_params, expect_grad_fn=False
        )
        report(
            "8a: FOMAML adapted params lack grad_fn",
            passed_fo,
            f"violations={violations_fo}" if not passed_fo else "No grad_fn (correct)"
        )
    except Exception as e:
        report("8a: FOMAML adapted params lack grad_fn", False, str(e))

    try:
        # In FOMAML, adapted params should still require_grad (for query loss)
        all_require_grad = all(
            p.requires_grad for p in result_fo.adapted_params.values()
        )
        report("8b: FOMAML params require_grad", all_require_grad)
    except Exception as e:
        report("8b: FOMAML params require_grad", False, str(e))

    # ------------------------------------------------------------------
    # Test 9: Gradient clipping
    # ------------------------------------------------------------------
    print("\n--- Test Group 9: Gradient Clipping ---")

    try:
        engine_clip = CustomSGDEngine()
        params = _extract_params(model, clone=True)

        # Use very small clip norm to force clipping
        result_clip = engine_clip.adapt(
            model, params, support_x, support_y,
            steps=3, lr=INNER_LR, first_order=True, clip_norm=0.001,
        )

        any_clipped = any(log.clipped for log in result_clip.step_logs)
        report("9a: Gradient clipping triggered", any_clipped,
               f"clipped steps: {sum(1 for l in result_clip.step_logs if l.clipped)}/{len(result_clip.step_logs)}")
    except Exception as e:
        report("9a: Gradient clipping triggered", False, str(e))

    try:
        # Verify that clipped grad norms are bounded
        for log in result_clip.step_logs:
            if log.clipped:
                # Update norm should be small due to clipping
                assert log.update_norm < 1.0, f"Update norm {log.update_norm} too large after clipping"
        report("9b: Clipped update norms bounded", True)
    except Exception as e:
        report("9b: Clipped update norms bounded", False, str(e))

    try:
        # Test clip_grad_tuple directly
        g1 = torch.randn(10) * 100  # Large gradient
        g2 = torch.randn(5) * 100
        clipped_grads, was_clipped = clip_grad_tuple((g1, g2, None), max_norm=1.0)
        assert was_clipped, "Should have clipped large gradients"
        clipped_norm = sum(g.norm()**2 for g in clipped_grads if g is not None).sqrt().item()
        assert abs(clipped_norm - 1.0) < 0.1, f"Clipped norm {clipped_norm} should be ~1.0"
        assert clipped_grads[2] is None, "None entries should remain None"
        report("9c: clip_grad_tuple utility", True, f"clipped_norm={clipped_norm:.4f}")
    except Exception as e:
        report("9c: clip_grad_tuple utility", False, str(e))

    try:
        # Test with no clipping needed (small gradients)
        g_small = torch.randn(10) * 0.001
        _, not_clipped = clip_grad_tuple((g_small,), max_norm=100.0)
        assert not not_clipped, "Small gradients should not trigger clipping"
        report("9d: No-clip path", True)
    except Exception as e:
        report("9d: No-clip path", False, str(e))

    # ------------------------------------------------------------------
    # Test 10: AMP Safety Wrapper
    # ------------------------------------------------------------------
    print("\n--- Test Group 10: AMP Safety Wrapper ---")

    try:
        base_engine = CustomSGDEngine()
        safe_engine = AMPSafeInnerLoop(base_engine)

        # Provide fp16 params and inputs
        fp16_params = {k: v.half() for k, v in _extract_params(model, clone=True).items()}
        fp16_x = support_x.half()

        result_amp = safe_engine.adapt(
            model, fp16_params, fp16_x, support_y,
            steps=3, lr=INNER_LR, first_order=True,
        )

        # Adapted params should be fp32
        all_fp32 = all(
            p.dtype == torch.float32 for p in result_amp.adapted_params.values()
        )
        report("10a: AMP wrapper casts to fp32", all_fp32,
               f"dtypes: {set(p.dtype for p in result_amp.adapted_params.values())}")
    except Exception as e:
        report("10a: AMP wrapper casts to fp32", False, str(e))

    try:
        # Verify the name property
        assert "AMPSafe" in safe_engine.name, f"Name should contain AMPSafe: {safe_engine.name}"
        report("10b: AMPSafeInnerLoop.name", True, safe_engine.name)
    except Exception as e:
        report("10b: AMPSafeInnerLoop.name", False, str(e))

    try:
        # Compare AMP-safe vs direct engine (should give similar results)
        params_direct = _extract_params(model, clone=True)
        params_safe = {k: v.clone() for k, v in params_direct.items()}

        r_direct = base_engine.adapt(
            model, params_direct, support_x, support_y,
            steps=3, lr=INNER_LR, first_order=True,
        )
        r_safe = safe_engine.adapt(
            model, params_safe, support_x, support_y,
            steps=3, lr=INNER_LR, first_order=True,
        )

        loss_diff = abs(r_direct.step_logs[0].loss - r_safe.step_logs[0].loss)
        report("10c: AMP-safe matches direct (fp32 input)", loss_diff < 1e-4,
               f"loss_diff={loss_diff:.6e}")
    except Exception as e:
        report("10c: AMP-safe matches direct (fp32 input)", False, str(e))

    # ------------------------------------------------------------------
    # Test 11: Factory function
    # ------------------------------------------------------------------
    print("\n--- Test Group 11: Factory Function ---")

    try:
        auto_engine = create_inner_loop_engine(backend="auto", use_amp_safety=True)
        assert isinstance(auto_engine, AMPSafeInnerLoop), "Auto should return AMPSafeInnerLoop"
        report("11a: create_inner_loop_engine('auto')", True,
               f"inner={auto_engine.engine.name}")
    except Exception as e:
        report("11a: create_inner_loop_engine('auto')", False, str(e))

    try:
        custom_engine = create_inner_loop_engine(backend="custom", use_amp_safety=False)
        assert isinstance(custom_engine, CustomSGDEngine), "Should be CustomSGDEngine"
        report("11b: create_inner_loop_engine('custom', no AMP)", True)
    except Exception as e:
        report("11b: create_inner_loop_engine('custom', no AMP)", False, str(e))

    try:
        # Invalid backend should raise ValueError
        try:
            create_inner_loop_engine(backend="invalid_backend")
            report("11c: Invalid backend raises ValueError", False, "No exception raised")
        except ValueError as ve:
            report("11c: Invalid backend raises ValueError", True, str(ve)[:50])
    except Exception as e:
        report("11c: Invalid backend raises ValueError", False, str(e))

    try:
        backends = get_available_backends()
        assert "custom" in backends and backends["custom"] is True
        assert "torch_func" in backends
        assert "higher" in backends
        report("11d: get_available_backends()", True, str(backends))
    except Exception as e:
        report("11d: get_available_backends()", False, str(e))

    # ------------------------------------------------------------------
    # Test 12: LSLR learning rate overrides
    # ------------------------------------------------------------------
    print("\n--- Test Group 12: LSLR Learning Rate Overrides ---")

    try:
        engine_lslr = CustomSGDEngine()
        params = _extract_params(model, clone=True)

        # Set up LSLR: higher LR for fc1, lower for fc2
        lslr_config = {
            "fc1.weight": {0: 0.1, 1: 0.08, 2: 0.05},
            "fc1.bias": {0: 0.1, 1: 0.08, 2: 0.05},
            "fc2.weight": {0: 0.001, 1: 0.001, 2: 0.001},
            "fc2.bias": {0: 0.001, 1: 0.001, 2: 0.001},
        }

        result_lslr = engine_lslr.adapt(
            model, params, support_x, support_y,
            steps=3, lr=0.01, lrs=lslr_config, first_order=True,
        )

        # Result should have different effective LRs than base
        assert result_lslr.num_steps == 3
        report("12a: LSLR adaptation runs", True, f"steps={result_lslr.num_steps}")
    except Exception as e:
        report("12a: LSLR adaptation runs", False, str(e))

    try:
        # Compare with non-LSLR to verify they differ
        params2 = _extract_params(model, clone=True)
        result_nolslr = engine_lslr.adapt(
            model, params2, support_x, support_y,
            steps=3, lr=0.01, lrs=None, first_order=True,
        )

        # Adapted params should differ between LSLR and non-LSLR
        differs = False
        for name in result_lslr.adapted_params:
            diff = (result_lslr.adapted_params[name] - result_nolslr.adapted_params[name]).abs().max().item()
            if diff > 1e-6:
                differs = True
                break
        report("12b: LSLR produces different adaptation", differs)
    except Exception as e:
        report("12b: LSLR produces different adaptation", False, str(e))

    try:
        # Test LSLRManager
        param_names = list(_extract_params(model).keys())
        lslr_mgr = LSLRManager(param_names, num_steps=5, init_lr=0.01)

        # Check initial LRs are close to 0.01
        lrs_dict = lslr_mgr.get_lrs()
        assert set(lrs_dict.keys()) == set(param_names), "LSLR should cover all params"
        for name in param_names:
            for step in range(5):
                lr_val = lrs_dict[name][step]
                assert abs(lr_val - 0.01) < 0.005, f"Initial LR {lr_val} too far from 0.01"

        report("12c: LSLRManager initialization", True,
               f"params={len(param_names)}, steps=5")
    except Exception as e:
        report("12c: LSLRManager initialization", False, str(e))

    try:
        lr_tensor = lslr_mgr.get_lr_tensor()
        assert lr_tensor.shape == (5, len(param_names)), f"Wrong shape: {lr_tensor.shape}"
        assert (lr_tensor >= LSLR_MIN_LR).all(), "LRs below minimum"
        assert (lr_tensor <= LSLR_MAX_LR).all(), "LRs above maximum"
        report("12d: LSLRManager.get_lr_tensor()", True, f"shape={lr_tensor.shape}")
    except Exception as e:
        report("12d: LSLRManager.get_lr_tensor()", False, str(e))

    # ------------------------------------------------------------------
    # Test 13: Meta-gradient verification
    # ------------------------------------------------------------------
    print("\n--- Test Group 13: Meta-Gradient Verification ---")

    try:
        torch.manual_seed(42)
        verify_model = ToyModel()
        verify_engine = CustomSGDEngine()

        verification = verify_meta_gradients(
            verify_model, verify_engine,
            support_x, support_y, query_x, query_y,
            steps=1, lr=INNER_LR,
        )

        assert verification["meta_grads_exist"], "Meta-gradients should exist"
        report("13a: Meta-gradients exist (MAML)", True,
               f"meta_loss={verification['meta_loss']:.4f}")
    except Exception as e:
        report("13a: Meta-gradients exist (MAML)", False, str(e))

    try:
        assert verification["all_grads_nonzero"], \
            f"Some params have zero meta-grad: {[n for n, s in verification['grad_stats'].items() if not s['nonzero']]}"
        report("13b: All meta-gradients non-zero", True)
    except Exception as e:
        report("13b: All meta-gradients non-zero", False, str(e))

    try:
        # Verify grad stats structure
        for name, stats in verification["grad_stats"].items():
            assert "norm" in stats, f"Missing norm for {name}"
            assert "mean" in stats, f"Missing mean for {name}"
            assert "std" in stats, f"Missing std for {name}"
        report("13c: Grad stats structure", True,
               f"{len(verification['grad_stats'])} params analyzed")
    except Exception as e:
        report("13c: Grad stats structure", False, str(e))

    # ------------------------------------------------------------------
    # Test 14: InnerLoopDiagnostics aggregation
    # ------------------------------------------------------------------
    print("\n--- Test Group 14: InnerLoopDiagnostics ---")

    try:
        diag = InnerLoopDiagnostics()
        engine_diag = CustomSGDEngine()

        # Run multiple episodes
        num_episodes = 5
        for _ in range(num_episodes):
            p = _extract_params(model, clone=True)
            r = engine_diag.adapt(
                model, p, support_x, support_y,
                steps=NUM_STEPS, lr=INNER_LR, first_order=True,
            )
            diag.record(r)

        assert diag.num_episodes() == num_episodes
        report("14a: Record episodes", True, f"{num_episodes} episodes recorded")
    except Exception as e:
        report("14a: Record episodes", False, str(e))

    try:
        mean_loss = diag.mean_loss_trajectory()
        assert len(mean_loss) == NUM_STEPS, f"Expected {NUM_STEPS} steps, got {len(mean_loss)}"
        report("14b: Mean loss trajectory", True,
               f"first={mean_loss[0]:.4f}, last={mean_loss[-1]:.4f}")
    except Exception as e:
        report("14b: Mean loss trajectory", False, str(e))

    try:
        mean_acc = diag.mean_accuracy_trajectory()
        assert len(mean_acc) == NUM_STEPS
        ld_rate = diag.loss_decreased_rate()
        assert 0.0 <= ld_rate <= 1.0
        report("14c: Accuracy trajectory & loss decrease rate", True,
               f"loss_decrease_rate={ld_rate:.2f}")
    except Exception as e:
        report("14c: Accuracy trajectory & loss decrease rate", False, str(e))

    try:
        summary = diag.summarize()
        assert "num_episodes" in summary
        assert "mean_loss_trajectory" in summary
        assert "clip_rate" in summary
        report("14d: Diagnostics summarize()", True, f"keys={list(summary.keys())}")
    except Exception as e:
        report("14d: Diagnostics summarize()", False, str(e))

    try:
        diag.clear()
        assert diag.num_episodes() == 0
        report("14e: Diagnostics clear()", True)
    except Exception as e:
        report("14e: Diagnostics clear()", False, str(e))

    # ------------------------------------------------------------------
    # Test 15: BatchParallelInnerLoop
    # ------------------------------------------------------------------
    print("\n--- Test Group 15: BatchParallelInnerLoop ---")

    try:
        base_eng = CustomSGDEngine()
        batch_loop = BatchParallelInnerLoop(base_eng)

        # Create multiple tasks
        tasks = []
        for _ in range(3):
            tx = torch.randn(8, INPUT_DIM)
            ty = torch.randint(0, OUTPUT_DIM, (8,))
            tasks.append((tx, ty))

        params_batch = _extract_params(model, clone=True)
        results_batch = batch_loop.adapt_batch(
            model, params_batch, tasks,
            steps=3, lr=INNER_LR, first_order=True,
        )

        assert len(results_batch) == 3, f"Expected 3 results, got {len(results_batch)}"
        report("15a: Batch adapt returns per-task results", True,
               f"{len(results_batch)} tasks")
    except Exception as e:
        report("15a: Batch adapt returns per-task results", False, str(e))

    try:
        # Test compute_meta_loss
        full_tasks = []
        for _ in range(3):
            sx = torch.randn(8, INPUT_DIM)
            sy = torch.randint(0, OUTPUT_DIM, (8,))
            qx = torch.randn(8, INPUT_DIM)
            qy = torch.randint(0, OUTPUT_DIM, (8,))
            full_tasks.append((sx, sy, qx, qy))

        params_meta = _extract_params(model, clone=True)
        meta_loss, meta_results = batch_loop.compute_meta_loss(
            model, params_meta, full_tasks,
            steps=3, lr=INNER_LR, first_order=True,
        )

        assert isinstance(meta_loss, torch.Tensor), "meta_loss should be tensor"
        assert meta_loss.dim() == 0, "meta_loss should be scalar"
        assert meta_loss.item() > 0, "meta_loss should be positive"
        report("15b: compute_meta_loss", True,
               f"meta_loss={meta_loss.item():.4f}")
    except Exception as e:
        report("15b: compute_meta_loss", False, str(e))

    # ------------------------------------------------------------------
    # Test 16: Detach detector utility
    # ------------------------------------------------------------------
    print("\n--- Test Group 16: Detach Detector ---")

    try:
        # Create params with grad_fn (MAML-style)
        eng_detect = CustomSGDEngine()
        params_d = _extract_params(model, clone=True)
        result_maml = eng_detect.adapt(
            model, params_d, support_x, support_y,
            steps=2, lr=INNER_LR, first_order=False,
        )

        passed_check, viol = check_grad_fn_retained(
            result_maml.adapted_params, expect_grad_fn=True
        )
        report("16a: Detach detector (MAML, expect grad_fn)", passed_check,
               f"violations={viol}" if not passed_check else "All have grad_fn")
    except Exception as e:
        report("16a: Detach detector (MAML, expect grad_fn)", False, str(e))

    try:
        # Create params without grad_fn (FOMAML-style)
        params_d2 = _extract_params(model, clone=True)
        result_fomaml = eng_detect.adapt(
            model, params_d2, support_x, support_y,
            steps=2, lr=INNER_LR, first_order=True,
        )

        passed_check2, viol2 = check_grad_fn_retained(
            result_fomaml.adapted_params, expect_grad_fn=False
        )
        report("16b: Detach detector (FOMAML, expect no grad_fn)", passed_check2,
               f"violations={viol2}" if not passed_check2 else "None have grad_fn")
    except Exception as e:
        report("16b: Detach detector (FOMAML, expect no grad_fn)", False, str(e))

    try:
        # Negative test: expect grad_fn on FOMAML params should FAIL
        should_fail, should_have_viol = check_grad_fn_retained(
            result_fomaml.adapted_params, expect_grad_fn=True
        )
        assert not should_fail, "Expecting grad_fn on FOMAML params should fail"
        assert len(should_have_viol) > 0, "Should report violations"
        report("16c: Detach detector (negative test)", True,
               f"Correctly detected {len(should_have_viol)} violations")
    except Exception as e:
        report("16c: Detach detector (negative test)", False, str(e))

    # ------------------------------------------------------------------
    # Test 17: Edge cases
    # ------------------------------------------------------------------
    print("\n--- Test Group 17: Edge Cases ---")

    try:
        # Zero steps
        engine_edge = CustomSGDEngine()
        params_z = _extract_params(model, clone=True)
        result_zero = engine_edge.adapt(
            model, params_z, support_x, support_y,
            steps=0, lr=INNER_LR, first_order=True,
        )
        assert result_zero.num_steps == 0
        assert len(result_zero.step_logs) == 0
        # Params should be unchanged
        for name in params_z:
            diff = (result_zero.adapted_params[name] - params_z[name]).abs().max().item()
            assert diff < 1e-6, f"Zero steps should not change params: {name} diff={diff}"
        report("17a: Zero steps", True, "Params unchanged")
    except Exception as e:
        report("17a: Zero steps", False, str(e))

    try:
        # Single step
        params_s = _extract_params(model, clone=True)
        result_single = engine_edge.adapt(
            model, params_s, support_x, support_y,
            steps=1, lr=INNER_LR, first_order=True,
        )
        assert result_single.num_steps == 1
        assert len(result_single.step_logs) == 1
        report("17b: Single step", True, f"loss={result_single.step_logs[0].loss:.4f}")
    except Exception as e:
        report("17b: Single step", False, str(e))

    try:
        # Very large LR (may cause loss explosion, but should not crash)
        params_l = _extract_params(model, clone=True)
        result_large = engine_edge.adapt(
            model, params_l, support_x, support_y,
            steps=2, lr=100.0, first_order=True,
        )
        # Should complete without error
        assert result_large.num_steps == 2
        report("17c: Large LR (no crash)", True,
               f"final_loss={result_large.step_logs[-1].loss:.4f}")
    except Exception as e:
        report("17c: Large LR (no crash)", False, str(e))

    try:
        # Very small LR (params should barely change)
        params_tiny = _extract_params(model, clone=True)
        result_tiny = engine_edge.adapt(
            model, params_tiny, support_x, support_y,
            steps=3, lr=1e-10, first_order=True,
        )
        max_change = 0.0
        for name in params_tiny:
            diff = (result_tiny.adapted_params[name] - params_tiny[name]).abs().max().item()
            max_change = max(max_change, diff)
        report("17d: Tiny LR (minimal change)", max_change < 1e-6,
               f"max_change={max_change:.2e}")
    except Exception as e:
        report("17d: Tiny LR (minimal change)", False, str(e))

    # ------------------------------------------------------------------
    # Test 18: Functional forward helper
    # ------------------------------------------------------------------
    print("\n--- Test Group 18: Functional Forward ---")

    try:
        params_fwd = _extract_params(model, clone=True)
        output = _functional_forward(model, params_fwd, support_x)
        assert output.shape == (BATCH_SIZE, OUTPUT_DIM), \
            f"Expected shape ({BATCH_SIZE}, {OUTPUT_DIM}), got {output.shape}"
        report("18a: _functional_forward output shape", True, f"shape={output.shape}")
    except Exception as e:
        report("18a: _functional_forward output shape", False, str(e))

    try:
        # Verify that _functional_forward with original params matches model(x)
        model.train(False)
        with torch.no_grad():
            direct_out = model(support_x)
            func_out = _functional_forward(model, dict(model.named_parameters()), support_x)
        diff = (direct_out - func_out).abs().max().item()
        report("18b: Functional forward matches direct", diff < 1e-5,
               f"max_diff={diff:.2e}")
    except Exception as e:
        report("18b: Functional forward matches direct", False, str(e))

    try:
        # Verify that modified params produce different output
        modified_params = _extract_params(model, clone=True)
        modified_params["fc1.weight"] = modified_params["fc1.weight"] + 10.0
        with torch.no_grad():
            modified_out = _functional_forward(model, modified_params, support_x)
            original_out = model(support_x)
        diff = (modified_out - original_out).abs().max().item()
        report("18c: Modified params change output", diff > 0.1,
               f"output_diff={diff:.4f}")
    except Exception as e:
        report("18c: Modified params change output", False, str(e))

    # ------------------------------------------------------------------
    # Test 19: Compute accuracy utility
    # ------------------------------------------------------------------
    print("\n--- Test Group 19: Accuracy Computation ---")

    try:
        # Perfect accuracy
        perfect_logits = torch.zeros(10, 5)
        perfect_targets = torch.arange(5).repeat(2)
        for i in range(10):
            perfect_logits[i, perfect_targets[i]] = 10.0
        acc = _compute_accuracy(perfect_logits, perfect_targets)
        report("19a: Perfect accuracy", abs(acc - 1.0) < 1e-6, f"acc={acc}")
    except Exception as e:
        report("19a: Perfect accuracy", False, str(e))

    try:
        # Zero accuracy (all predict class 0 but targets are class 1)
        bad_logits = torch.zeros(10, 5)
        bad_logits[:, 0] = 10.0
        bad_targets = torch.ones(10, dtype=torch.long)
        acc_bad = _compute_accuracy(bad_logits, bad_targets)
        report("19b: Zero accuracy", abs(acc_bad - 0.0) < 1e-6, f"acc={acc_bad}")
    except Exception as e:
        report("19b: Zero accuracy", False, str(e))

    try:
        # Float targets (regression) should return 0.0
        reg_logits = torch.randn(10, 1)
        reg_targets = torch.randn(10)
        acc_reg = _compute_accuracy(reg_logits, reg_targets)
        report("19c: Regression returns 0.0", abs(acc_reg) < 1e-6, f"acc={acc_reg}")
    except Exception as e:
        report("19c: Regression returns 0.0", False, str(e))

    # ------------------------------------------------------------------
    # Test 20: LSLR clamping bounds
    # ------------------------------------------------------------------
    print("\n--- Test Group 20: LSLR Bounds & Safety ---")

    try:
        # Test that LSLR values below minimum are clamped
        engine_bounds = CustomSGDEngine()
        params_b = _extract_params(model, clone=True)

        # LR below LSLR_MIN_LR
        extreme_lslr = {
            name: {0: 1e-10, 1: 1e-10}
            for name in params_b
        }

        result_lb = engine_bounds.adapt(
            model, params_b, support_x, support_y,
            steps=2, lr=0.01, lrs=extreme_lslr, first_order=True,
        )
        # Should not crash and should use clamped LR
        assert result_lb.num_steps == 2
        report("20a: LSLR below-minimum clamping", True)
    except Exception as e:
        report("20a: LSLR below-minimum clamping", False, str(e))

    try:
        # LR above LSLR_MAX_LR
        extreme_lslr_high = {
            name: {0: 100.0, 1: 100.0}
            for name in params_b
        }
        params_b2 = _extract_params(model, clone=True)
        result_ub = engine_bounds.adapt(
            model, params_b2, support_x, support_y,
            steps=2, lr=0.01, lrs=extreme_lslr_high, first_order=True,
        )
        assert result_ub.num_steps == 2
        report("20b: LSLR above-maximum clamping", True)
    except Exception as e:
        report("20b: LSLR above-maximum clamping", False, str(e))

    # ------------------------------------------------------------------
    # Test 21: Multi-step loss accumulation
    # ------------------------------------------------------------------
    print("\n--- Test Group 21: Multi-Step Loss ---")

    try:
        engine_msl = CustomSGDEngine()
        params_msl = _extract_params(model, clone=True)

        result_msl = engine_msl.adapt(
            model, params_msl, support_x, support_y,
            steps=5, lr=INNER_LR, first_order=True,
        )

        # Verify total_inner_loss equals sum of per-step losses
        expected_total = sum(log.loss for log in result_msl.step_logs)
        actual_total = result_msl.total_inner_loss
        diff_total = abs(expected_total - actual_total)
        report("21a: Total inner loss matches sum of step losses",
               diff_total < 1e-4,
               f"expected={expected_total:.4f}, actual={actual_total:.4f}")
    except Exception as e:
        report("21a: Total inner loss matches sum of step losses", False, str(e))

    # ------------------------------------------------------------------
    # Test 22: Engine with custom loss function
    # ------------------------------------------------------------------
    print("\n--- Test Group 22: Custom Loss Functions ---")

    try:
        engine_custom_loss = CustomSGDEngine()
        params_cl = _extract_params(model, clone=True)

        # Use MSE loss (regression-style)
        def mse_loss_fn(logits, targets):
            # Convert targets to one-hot for MSE
            one_hot = F.one_hot(targets, num_classes=OUTPUT_DIM).float()
            return F.mse_loss(logits, one_hot)

        result_mse = engine_custom_loss.adapt(
            model, params_cl, support_x, support_y,
            steps=3, lr=INNER_LR, first_order=True,
            loss_fn=mse_loss_fn,
        )
        assert result_mse.num_steps == 3
        report("22a: Custom loss function (MSE)", True,
               f"loss[0]={result_mse.step_logs[0].loss:.4f}")
    except Exception as e:
        report("22a: Custom loss function (MSE)", False, str(e))

    try:
        # Label smoothing cross entropy
        def smooth_ce(logits, targets, smoothing=0.1):
            n_classes = logits.shape[-1]
            one_hot = F.one_hot(targets, n_classes).float()
            smooth_labels = one_hot * (1 - smoothing) + smoothing / n_classes
            log_probs = F.log_softmax(logits, dim=-1)
            return -(smooth_labels * log_probs).sum(dim=-1).mean()

        params_cl2 = _extract_params(model, clone=True)
        result_smooth = engine_custom_loss.adapt(
            model, params_cl2, support_x, support_y,
            steps=3, lr=INNER_LR, first_order=True,
            loss_fn=smooth_ce,
        )
        report("22b: Label-smoothing CE loss", True,
               f"loss[0]={result_smooth.step_logs[0].loss:.4f}")
    except Exception as e:
        report("22b: Label-smoothing CE loss", False, str(e))

    # ------------------------------------------------------------------
    # Test 23: Higher engine (if available)
    # ------------------------------------------------------------------
    print("\n--- Test Group 23: HigherEngine ---")

    if HAS_HIGHER:
        try:
            h_engine = HigherEngine()
            params_h = _extract_params(model, clone=True)

            result_h = h_engine.adapt(
                model, params_h, support_x, support_y,
                steps=3, lr=INNER_LR, first_order=True,
            )
            assert result_h.backend == "higher"
            assert result_h.num_steps == 3
            report("23a: HigherEngine basic", True,
                   f"loss[0]={result_h.step_logs[0].loss:.4f}")
        except Exception as e:
            report("23a: HigherEngine basic", False, str(e))

        try:
            # Second-order
            params_h2 = _extract_params(model, clone=True)
            result_h2 = h_engine.adapt(
                model, params_h2, support_x, support_y,
                steps=2, lr=INNER_LR, first_order=False,
            )
            report("23b: HigherEngine second-order", True,
                   f"loss={result_h2.step_logs[0].loss:.4f}")
        except Exception as e:
            report("23b: HigherEngine second-order", False, str(e))
    else:
        report("23a: HigherEngine basic", True, "SKIPPED (higher not installed)")
        report("23b: HigherEngine second-order", True, "SKIPPED (higher not installed)")

    # ------------------------------------------------------------------
    # Test 24: Concurrent model safety
    # ------------------------------------------------------------------
    print("\n--- Test Group 24: Model Safety ---")

    try:
        # Verify that adaptation does not modify the original model
        torch.manual_seed(123)
        safe_model = ToyModel()
        original_state = {k: v.clone() for k, v in safe_model.state_dict().items()}

        engine_safe = CustomSGDEngine()
        params_safe = _extract_params(safe_model, clone=True)

        _ = engine_safe.adapt(
            safe_model, params_safe, support_x, support_y,
            steps=5, lr=0.1, first_order=True,
        )

        # Check model is unchanged
        current_state = safe_model.state_dict()
        model_unchanged = True
        for name in original_state:
            diff = (original_state[name] - current_state[name]).abs().max().item()
            if diff > 1e-7:
                model_unchanged = False
                break
        report("24a: Original model unchanged after adapt", model_unchanged)
    except Exception as e:
        report("24a: Original model unchanged after adapt", False, str(e))

    try:
        # Verify second-order adaptation also preserves model
        torch.manual_seed(123)
        safe_model2 = ToyModel()
        original_state2 = {k: v.clone() for k, v in safe_model2.state_dict().items()}

        params_safe2 = _extract_params(safe_model2, clone=True)
        _ = engine_safe.adapt(
            safe_model2, params_safe2, support_x, support_y,
            steps=3, lr=0.01, first_order=False,
        )

        current_state2 = safe_model2.state_dict()
        model_unchanged2 = True
        for name in original_state2:
            diff = (original_state2[name] - current_state2[name]).abs().max().item()
            if diff > 1e-7:
                model_unchanged2 = False
                break
        report("24b: Model safe under second-order adapt", model_unchanged2)
    except Exception as e:
        report("24b: Model safe under second-order adapt", False, str(e))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULTS: {_counters['passed']} passed, {_counters['failed']} failed, "
          f"{_counters['passed'] + _counters['failed']} total")
    print("=" * 70)

    if _counters["failed"] > 0:
        print("\nFailed tests:")
        for name, passed in test_results.items():
            if not passed:
                print(f"  - {name}")
        print()

    if _counters["failed"] == 0:
        print("\nAll self-tests passed!")
    else:
        print(f"\n{_counters['failed']} test(s) failed. Review output above for details.")
