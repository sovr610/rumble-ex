# Inner-Loop Engine: Differentiable Optimization for Meta-Learning

## 1. Overview

The inner-loop engine is the computational core of gradient-based meta-learning. Given a model's parameters and a task's support set, perform one or more gradient descent steps to produce *adapted parameters* specialized to that task. The adaptation process must either preserve the computational graph (for second-order MAML) or allow first-order gradient approximation (for FOMAML and Reptile).

Inputs:
- A parameter dictionary `params: Dict[str, Tensor]` extracted from the base model
- A support set `(x_support, y_support)` representing the task's training examples
- A step count, learning rate, and algorithm configuration

Outputs:
- An adapted parameter dictionary `adapted_params: Dict[str, Tensor]`
- A trajectory log `List[StepLog]` recording loss, accuracy, gradient norms, and clipping events at each step

The adapted parameters are used to compute a query-set loss. In MAML, the meta-gradient flows backward through the adapted parameters, through every inner-loop step, and into the original model parameters.

Three non-negotiable properties:

1. **Graph correctness**: In MAML mode, every tensor in `adapted_params` must retain its `grad_fn`, connecting it to the original `params`. Calling `.backward()` on the query loss must produce non-zero gradients on the original parameters.

2. **First-order correctness**: In FOMAML mode, gradients are computed at each inner step but the graph through the update rule is not retained. The meta-gradient treats adapted parameters as produced by a non-differentiable process but still propagates a valid gradient back to the original parameters.

3. **Numerical stability under AMP**: The inner loop must execute in fp32 regardless of any outer autocast context. Meta-gradients through chained gradient operations are extremely sensitive to precision loss; fp16 inner loops produce NaN or zero meta-gradients.

The engine provides three interchangeable backends -- `torch.func` (preferred), `higher` (optional), and custom SGD (required fallback) -- selected automatically based on availability, with manual override via configuration.

---

## 2. torch.func Backend (Preferred)

The preferred implementation uses PyTorch's `torch.func` module (PyTorch 2.0+, stabilized in 2.1+). This module provides functional transformations operating on explicit parameter dictionaries, eliminating module cloning or in-place parameter patching.

### Core API Surface

- **`torch.func.functional_call(module, params_dict, args)`** -- Run a module's forward pass with the provided parameter dictionary instead of its own `.parameters()`. The module structure is used but all weight/bias values come from `params_dict`.

- **`torch.func.grad(fn)`** -- Return a function that computes the gradient of `fn` with respect to its first argument. Compose with `functional_call` to get gradients of a loss with respect to parameters.

- **`torch.vmap(fn)`** -- Vectorize `fn` over a batch dimension. Apply to the entire inner-loop function to process multiple tasks in parallel (2-5x speedup on GPU).

### Implementation Pattern

```python
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad


def torch_func_inner_step(params, model, x, y, lr, create_graph):
    """Single inner-loop step using torch.func.

    Args:
        params: Dict[str, Tensor] -- current parameter values
        model: nn.Module -- architecture (used for structure, not weights)
        x: Tensor -- support set inputs
        y: Tensor -- support set targets
        lr: float or Dict[str, float] -- learning rate(s)
        create_graph: bool -- True for MAML, False for FOMAML

    Returns:
        adapted: Dict[str, Tensor] -- updated parameters
    """
    def loss_fn(p):
        logits = functional_call(model, p, (x,))
        return F.cross_entropy(logits, y)

    grads = grad(loss_fn)(params)

    # Per-parameter update
    if isinstance(lr, dict):
        adapted = {k: p - lr[k] * g for (k, p), g in
                   zip(params.items(), grads.values())}
    else:
        adapted = {k: p - lr * g for (k, p), g in
                   zip(params.items(), grads.values())}

    return adapted


def torch_func_inner_loop(params, model, x, y, steps, lr, create_graph, clip_norm=None):
    """Full inner loop using torch.func.

    Args:
        params: Dict[str, Tensor] -- initial parameters
        model: nn.Module -- architecture
        x: Tensor -- support inputs
        y: Tensor -- support targets
        steps: int -- number of inner-loop steps
        lr: float or Dict[str, float] -- learning rate(s)
        create_graph: bool -- retain graph for second-order gradients
        clip_norm: Optional[float] -- max gradient norm per step

    Returns:
        adapted: Dict[str, Tensor] -- adapted parameters
        logs: List[StepLog] -- per-step diagnostics
    """
    adapted = params  # No clone needed; functional approach never mutates
    logs = []

    for step in range(steps):
        def loss_fn(p):
            logits = functional_call(model, p, (x,))
            return F.cross_entropy(logits, y)

        loss_val = loss_fn(adapted)
        grads = grad(loss_fn)(adapted)

        if clip_norm is not None:
            grads = _clip_grad_dict(grads, clip_norm)

        grad_norm = _compute_grad_norm(grads)
        update_norm = lr * grad_norm if isinstance(lr, (int, float)) else None

        if isinstance(lr, dict):
            adapted = {k: p - lr[k] * grads[k] for k, p in adapted.items()}
        else:
            adapted = {k: p - lr * grads[k] for k, p in adapted.items()}

        logs.append(StepLog(
            step=step,
            loss=loss_val.item(),
            accuracy=_compute_accuracy(model, adapted, x, y),
            grad_norm=grad_norm,
            update_norm=update_norm or 0.0,
            lr_effective=lr if isinstance(lr, (int, float)) else -1.0,
            clipped=clip_norm is not None and grad_norm > clip_norm,
        ))

    return adapted, logs
```

### Advantages

- **No module cloning**: `functional_call` uses the existing module as a template. No `deepcopy`, no memory duplication, no risk of shared state.
- **Composable with vmap**: Vectorize task-parallel inner loops for 2-5x GPU speedup.
- **Maintained by PyTorch core**: Part of the PyTorch distribution; tracks autograd engine changes.
- **Clean graph semantics**: Functional style naturally preserves `grad_fn` on outputs. No ambiguity about in-place operations breaking the graph.

### Pitfalls

- **PyTorch version requirement**: `torch.func` requires PyTorch >= 2.0. The `functional_call` function existed in `torch.nn.utils` in earlier versions with a different signature. Detect the version at import time:

```python
import torch

_HAS_TORCH_FUNC = hasattr(torch, "func") and hasattr(torch.func, "functional_call")

if not _HAS_TORCH_FUNC:
    try:
        from torch.nn.utils._stateless import functional_call as _legacy_functional_call
        _HAS_LEGACY_FUNCTIONAL_CALL = True
    except ImportError:
        _HAS_LEGACY_FUNCTIONAL_CALL = False
```

- **Signature changes**: In PyTorch 2.0, `functional_call` accepted `(module, parameter_and_buffer_dicts, args)` where the second argument could be a tuple of two dicts. In PyTorch 2.1+, it accepts a single merged dict. Always pass a single dict of named parameters and buffers.

- **Buffer handling**: `functional_call` replaces both parameters and buffers if present in the dict. For inner-loop optimization, only pass parameters (not buffers like batch norm running statistics) unless intentionally adapting BN stats. Extract parameters explicitly:

```python
params = {name: p for name, p in model.named_parameters()}
# Do NOT include buffers unless adapting them:
# buffers = {name: b for name, b in model.named_buffers()}
```

- **vmap limitations**: Not all PyTorch operations support vmap. Custom autograd functions, certain in-place operations, and some third-party layer implementations may fail under vmap. Test the full inner loop under vmap before relying on it in production.

---

## 3. higher Library Backend (Optional)

The `facebookresearch/higher` library provides a context manager creating a "patched" module whose parameters are differentiable through optimizer steps. This was the standard approach before `torch.func`.

### Implementation Pattern

```python
import higher


def higher_inner_loop(model, optimizer, x_support, y_support, steps, create_graph):
    """Inner loop using the higher library.

    Args:
        model: nn.Module -- model to adapt
        optimizer: torch.optim.Optimizer -- inner-loop optimizer (e.g., SGD)
        x_support: Tensor -- support inputs
        y_support: Tensor -- support targets
        steps: int -- inner-loop steps
        create_graph: bool -- track_higher_grads flag

    Returns:
        fmodel: higher.patch.MonkeyPatchedModule -- adapted model
        logs: List[StepLog] -- per-step diagnostics
    """
    logs = []

    with higher.innerloop_ctx(
        model,
        optimizer,
        copy_initial_weights=False,
        track_higher_grads=create_graph,
    ) as (fmodel, diffopt):
        for step in range(steps):
            logits = fmodel(x_support)
            loss = F.cross_entropy(logits, y_support)
            diffopt.step(loss)

            logs.append(StepLog(
                step=step,
                loss=loss.item(),
                accuracy=_compute_accuracy_from_logits(logits, y_support),
                grad_norm=_extract_grad_norm(fmodel),
                update_norm=0.0,  # Not easily extractable from higher
                lr_effective=optimizer.defaults['lr'],
                clipped=False,
            ))

    return fmodel, logs
```

### Key Parameters

- **`copy_initial_weights=False`**: Patched model shares tensors with original, so meta-gradients flow back. When True, weights are cloned, breaking the meta-gradient connection. Always use False for MAML/FOMAML.

- **`track_higher_grads`**: Controls whether the optimizer step is differentiable. True for MAML (second-order), False for FOMAML (first-order). Equivalent to `create_graph` in `torch.autograd.grad`.

### Feature Flag

Enable the `higher` backend via configuration:

```python
# In MAMLConfig or InnerLoopConfig:
backend: str = "auto"  # "auto", "torch_func", "higher", "custom"
use_higher: bool = False  # Explicit flag; overrides auto when True
```

When `backend="auto"`, prefer `torch.func` if available. Fall back to `higher` only if `torch.func` is unavailable and `higher` is installed and `use_higher=True`.

### Pitfalls

- **Archived repository**: `higher` is archived since 2023. No new development. Treat as best-effort for legacy codebases or PyTorch < 2.0.
- **Memory overhead**: Module patching plus unrolled graph can consume 3-5x model size per inner step.
- **Incompatible modules**: Custom `__getattr__`, lazy initialization, and non-standard parameter registration may fail. Common failures: custom attention, MoE routing, certain normalization layers.
- **Optimizer compatibility**: Not all optimizer features (momentum, weight decay, LR schedulers) work correctly in the differentiable wrapper.

---

## 4. Custom Differentiable SGD (Required Fallback)

The custom backend is a pure PyTorch implementation using `torch.autograd.grad` for gradient computation and explicit parameter dictionary updates. It requires no external dependencies and works on all PyTorch versions that support `torch.autograd.grad` (effectively all modern versions).

This backend is the required fallback. It must always be available and must produce results identical (within floating-point tolerance) to the other backends.

### Implementation Pattern

```python
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class StepLog:
    step: int
    loss: float
    accuracy: float
    grad_norm: float
    update_norm: float
    lr_effective: float
    clipped: bool


def functional_forward(model, params, x):
    """Forward pass using explicit parameters.

    For models that support functional_call, delegate to it.
    Otherwise, temporarily replace model parameters, run forward,
    and restore originals.
    """
    try:
        from torch.func import functional_call
        return functional_call(model, params, (x,))
    except (ImportError, AttributeError):
        # Manual parameter substitution fallback
        original = {}
        for name, param in model.named_parameters():
            original[name] = param.data
            _set_param(model, name, params[name])
        try:
            output = model(x)
        finally:
            for name, param_data in original.items():
                _set_param(model, name, param_data)
        return output


def custom_inner_loop(
    params: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    steps: int,
    lr: float,
    create_graph: bool,
    clip_norm: Optional[float] = None,
    per_layer_lrs: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, torch.Tensor], List[StepLog]]:
    """Differentiable inner loop using torch.autograd.grad.

    Args:
        params: Named parameter dict from model.named_parameters()
        model: Module providing architecture for forward pass
        support_x: Support set inputs
        support_y: Support set targets
        steps: Number of gradient descent steps
        lr: Base learning rate (overridden per-layer if per_layer_lrs given)
        create_graph: True for MAML (second-order), False for FOMAML
        clip_norm: Maximum gradient norm; None disables clipping
        per_layer_lrs: Optional per-layer learning rate overrides

    Returns:
        adapted: Dict of adapted parameter tensors
        logs: Per-step diagnostic logs
    """
    # Initialize adapted parameters
    # For create_graph=True, keep the reference (graph intact from original params)
    # For create_graph=False, clone to avoid in-place modification of originals
    if create_graph:
        adapted = {k: v for k, v in params.items()}
    else:
        adapted = {k: v.clone() for k, v in params.items()}

    logs = []

    for step in range(steps):
        # Forward pass with current adapted parameters
        logits = functional_forward(model, adapted, support_x)
        loss = F.cross_entropy(logits, support_y)

        # Compute gradients of loss w.r.t. adapted parameters
        grad_tensors = torch.autograd.grad(
            outputs=loss,
            inputs=list(adapted.values()),
            create_graph=create_graph,
            allow_unused=True,
        )

        # Replace None gradients with zeros (for unused parameters)
        grads = []
        for g, (k, p) in zip(grad_tensors, adapted.items()):
            if g is None:
                grads.append(torch.zeros_like(p))
            else:
                grads.append(g)

        # Gradient clipping
        clipped = False
        if clip_norm is not None:
            grads, clipped = clip_grad_tuple(grads, clip_norm)

        # Compute diagnostic norms before update
        grad_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads)).item()

        # Parameter update
        new_adapted = {}
        for (k, p), g in zip(adapted.items(), grads):
            effective_lr = per_layer_lrs[k] if per_layer_lrs and k in per_layer_lrs else lr
            new_adapted[k] = p - effective_lr * g

        update_norm = torch.sqrt(
            sum((new_adapted[k] - adapted[k]).norm() ** 2 for k in adapted)
        ).item()

        adapted = new_adapted

        # Compute accuracy for logging
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == support_y).float().mean().item()

        logs.append(StepLog(
            step=step,
            loss=loss.item(),
            accuracy=accuracy,
            grad_norm=grad_norm,
            update_norm=update_norm,
            lr_effective=lr,
            clipped=clipped,
        ))

    return adapted, logs
```

### The create_graph Parameter

This single boolean is the MAML vs FOMAML distinction at the implementation level:

- **`create_graph=True` (MAML)**: Gradient tensors themselves are differentiable. The meta-gradient flows through the gradient computation of each inner step, producing exact second-order meta-gradients (including Hessian-vector product terms).

- **`create_graph=False` (FOMAML)**: Gradient tensors are leaf tensors with no `grad_fn`. The update `p - lr * g` depends on `p` (connected to graph) but treats `g` as a constant, dropping all second-order terms.

- **Reptile (no graph)**: Standard SGD inner loop. Meta-update is weight-space interpolation: `theta += epsilon * (phi - theta)`. Use `create_graph=False` or `torch.no_grad()`.

### Reptile-Specific Implementation

Reptile does not require any graph construction. The inner loop is standard SGD:

```python
def reptile_inner_loop(
    params: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    steps: int,
    lr: float,
) -> Dict[str, torch.Tensor]:
    """Reptile inner loop: standard SGD, no graph.

    Returns adapted parameters. The Reptile meta-update is:
        theta = theta + epsilon * (adapted - theta)
    computed externally after this function returns.
    """
    adapted = {k: v.clone().detach().requires_grad_(True) for k, v in params.items()}

    for step in range(steps):
        logits = functional_forward(model, adapted, support_x)
        loss = F.cross_entropy(logits, support_y)
        grads = torch.autograd.grad(loss, list(adapted.values()))
        adapted = {
            k: (p - lr * g).detach().requires_grad_(True)
            for (k, p), g in zip(adapted.items(), grads)
        }

    return adapted
```

The outer Reptile update is:

```python
def reptile_outer_update(params, adapted_params_list, epsilon):
    """Apply Reptile meta-update: move params toward average of adapted params."""
    with torch.no_grad():
        for k in params:
            avg_diff = torch.stack([ap[k] - params[k] for ap in adapted_params_list]).mean(dim=0)
            params[k] += epsilon * avg_diff
    return params
```

---

## 5. AMP Safety for Meta-Gradients

Mixed-precision training (AMP) is essential for production-scale models but dangerous for meta-gradient computation. Meta-gradients involve gradients of gradients -- a Hessian-vector product in MAML. Each gradient operation approximately halves effective precision, so fp16 gradients-of-gradients have ~fp8 effective precision, which is insufficient for stable optimization.

### The Problem in Detail

The meta-gradient through a single inner step involves:

```
d(outer_loss) / d(params) = d(outer_loss)/d(adapted) * (I - lr * d²(loss)/d(params)²)
```

The Hessian term `d²(loss)/d(params)²` frequently underflows to zero or overflows to infinity in fp16, producing zero or NaN meta-gradients.

### Required Pattern

Wrap the entire inner loop in an fp32 context, even when the outer training loop uses autocast:

```python
def safe_inner_loop(params, model, support_x, support_y, steps, lr, create_graph, clip_norm):
    """AMP-safe inner loop: always runs in fp32."""

    # Force fp32 for all inner-loop computation
    with torch.amp.autocast('cuda', enabled=False):
        # Cast inputs to fp32
        support_x_fp32 = support_x.float()
        support_y_long = support_y.long()

        # Cast parameters to fp32
        params_fp32 = {k: v.float() for k, v in params.items()}

        # Run inner loop in fp32
        adapted, logs = custom_inner_loop(
            params=params_fp32,
            model=model,
            support_x=support_x_fp32,
            support_y=support_y_long,
            steps=steps,
            lr=lr,
            create_graph=create_graph,
            clip_norm=clip_norm,
        )

    return adapted, logs
```

### GradScaler Interaction

When using `torch.amp.GradScaler` for the outer optimizer:

1. **Do not scale inner-loop losses.** The scale factor becomes part of the computational graph under `create_graph=True`, and its gradient is meaningless.

2. **Maintain fp32 master parameters.** Keep a separate fp32 copy for inner-loop use. Synchronize after each outer step:

```python
with torch.no_grad():
    for name, param in model.named_parameters():
        fp32_params[name].copy_(param.float())
```

3. **Unscale outer gradients normally.** The query-set meta-gradient is a standard first-order gradient and works with GradScaler as usual.

### Verification

Verify AMP wrapping does not change meta-gradient magnitudes: run the same inner loop with and without `autocast`, compare meta-gradients, and assert relative difference < 0.01. See Testing Patterns section F for the full test.

---

## 6. Gradient Clipping in Inner Loop

Inner-loop gradient clipping prevents catastrophic updates during early meta-training. Without clipping, a single large gradient step can move adapted parameters into a region where the query-set loss explodes, destabilizing the outer optimizer.

### Implementation

Clip by global norm across all parameter gradients:

```python
def clip_grad_tuple(grads, max_norm):
    """Clip gradient tuple by global norm.

    Args:
        grads: Sequence of gradient tensors (one per parameter)
        max_norm: Maximum allowed global norm

    Returns:
        clipped_grads: Tuple of (possibly scaled) gradient tensors
        was_clipped: bool indicating whether clipping occurred
    """
    total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads if g is not None))
    clip_coef = max_norm / (total_norm + 1e-6)
    was_clipped = clip_coef.item() < 1.0

    if was_clipped:
        clipped = tuple(g * clip_coef if g is not None else None for g in grads)
    else:
        clipped = tuple(grads)

    return clipped, was_clipped
```

### Application Rules

- Apply per step, before the parameter update. Each inner-loop step gets independently clipped.
- Log clip events in the `StepLog`. A high clip frequency (> 50% of steps) indicates the inner learning rate is too large or the model is in a pathological loss landscape region.
- The default `clip_norm` value is 10.0 (from `MAMLConfig.inner_clip`). For per-layer-per-step learning rates (LSLR), the effective gradient magnitude varies by layer, so a global clip norm may be too aggressive for some layers and too lenient for others. Consider per-layer clipping for LSLR configurations.
- When `create_graph=True`, the clipping operation itself must be differentiable. The implementation above uses multiplication by `clip_coef`, which is differentiable. Do not use `torch.clamp` on individual gradient elements -- element-wise clamping destroys directional information and produces poor meta-gradients.

### Per-Element vs. Global Norm Clipping

The MAML++ paper (Antoniou et al.) uses per-element clamping (`torch.clamp(grad, -c, c)`), which changes gradient direction (e.g., `[100, 1]` becomes `[10, 1]`). Global norm clipping preserves direction by scaling uniformly. Use global norm clipping unless specifically reproducing MAML++ results.

---

## 7. Backend Selection Logic

Select a backend at initialization time based on availability and configuration:

```python
def _select_backend(config):
    """Determine which inner-loop backend to use.

    Priority order:
    1. Explicit override from config.backend (if not "auto")
    2. torch.func (if PyTorch >= 2.0)
    3. higher (if installed and config.use_higher is True)
    4. Custom SGD (always available)

    Returns:
        str: One of "torch_func", "higher", "custom"
    """
    if config.backend != "auto":
        # Validate explicit choice
        if config.backend == "torch_func" and not _HAS_TORCH_FUNC:
            raise RuntimeError(
                "torch.func backend requested but PyTorch < 2.0. "
                "Upgrade PyTorch or use backend='auto'."
            )
        if config.backend == "higher" and not _HAS_HIGHER:
            raise RuntimeError(
                "higher backend requested but 'higher' package not installed. "
                "Install with: pip install higher"
            )
        return config.backend

    # Auto selection
    if _HAS_TORCH_FUNC:
        return "torch_func"
    if _HAS_HIGHER and config.use_higher:
        return "higher"
    return "custom"


# Availability detection (module-level)
_HAS_TORCH_FUNC = False
try:
    from torch.func import functional_call, grad
    _HAS_TORCH_FUNC = True
except ImportError:
    pass

_HAS_HIGHER = False
try:
    import higher
    _HAS_HIGHER = True
except ImportError:
    pass
```

### When to Override Auto Selection

- **Debugging**: Force `backend="custom"` for the most transparent inner loop -- standard `torch.autograd.grad` calls, easy to step through.
- **Reproducing published results**: Force `backend="higher"` when exact reproduction is needed, as backends may differ in floating-point ordering.
- **Benchmarking**: Compare all three backends on the same task. Use `torch_func` for production after confirming correctness against `custom`.

---

## 8. Parameter Dict Management

The inner-loop engine operates on parameter dictionaries rather than module objects.

### Extraction

```python
params = {name: param for name, param in model.named_parameters()}
```

Keys are dot-separated names (`"layer1.weight"`, `"layer2.bias"`). Values share storage with the module and have `requires_grad=True`.

### Buffer Handling

Buffers (batch norm running mean/var) are not in `named_parameters()`. Three modes:

- **Default**: Do not include buffers. Base model buffers are used as-is during `functional_call`.
- **Per-step BN**: Extract buffers separately and pass to `functional_call`. Update running stats each inner step.
- **Frozen BN**: Use base model buffers without modification. Set BN layers to eval mode.

```python
def extract_params_and_buffers(model, include_buffers=False):
    """Extract parameter dict, optionally including buffers."""
    params = {name: param for name, param in model.named_parameters()}
    if include_buffers:
        buffers = {name: buf for name, buf in model.named_buffers()}
        return {**params, **buffers}
    return params
```

### Graph Integrity After Inner Loop

- **MAML (`create_graph=True`)**: Every tensor in `adapted_params` has a `grad_fn` tracing back through all inner steps to the original `params`. Verify with `adapted_params[key].grad_fn is not None`.
- **FOMAML (`create_graph=False`)**: Tensors have `grad_fn` only from the final update (`p - lr * g` where `g` is detached). Meta-gradients flow through `p` but not through the gradient computation.
- **Reptile**: All tensors fully detached. No graph exists.

### Detaching for Evaluation

```python
eval_params = {k: v.detach() for k, v in adapted_params.items()}
```

Detach only for evaluation. During meta-training, detaching breaks meta-gradient flow.

### Parameter Dict Consistency

Maintain key-order consistency between original and adapted dicts. Python 3.7+ guarantees insertion-order iteration. Verify all backends produce dicts with identical keys in the same order.

---

## 9. StepLog Schema

Every inner-loop step produces a diagnostic record:

```python
from dataclasses import dataclass


@dataclass
class StepLog:
    """Diagnostic record for a single inner-loop step.

    Collected into a list of len=inner_steps, one per adaptation step.
    Used for adaptation curve visualization, debugging, and AUAC computation.
    """
    step: int            # Zero-indexed step number within the inner loop
    loss: float          # Cross-entropy loss on support set at this step (before update)
    accuracy: float      # Classification accuracy on support set at this step
    grad_norm: float     # L2 norm of the gradient vector across all parameters
    update_norm: float   # L2 norm of the parameter update (lr * grad, post-clip)
    lr_effective: float  # Actual learning rate used at this step
                         #   - For flat LR: same as config.inner_lr
                         #   - For LSLR: average across layers, or -1 if per-layer
    clipped: bool        # Whether gradient clipping was triggered at this step
```

### Usage in Adaptation Curves

The primary diagnostic is the adaptation curve: accuracy vs inner-loop step. Key metrics:

- **Pre-adaptation accuracy** (`step=0`): initialization quality
- **Adaptation speed**: accuracy gain per step
- **Saturation**: diminishing returns from additional steps
- **AUAC (Area Under the Adaptation Curve)**: trapezoidal integral of accuracy over steps, normalized. Higher = faster, more consistent adaptation.

```python
def compute_auac(logs: List[StepLog]) -> float:
    """Compute Area Under Adaptation Curve from step logs.

    Uses trapezoidal integration of accuracy over steps,
    normalized to [0, 1].
    """
    if len(logs) < 2:
        return logs[0].accuracy if logs else 0.0

    total = 0.0
    for i in range(1, len(logs)):
        total += (logs[i].accuracy + logs[i - 1].accuracy) / 2.0

    return total / (len(logs) - 1)
```

### Serialization

Serialize step logs to JSON for experiment tracking:

```python
import json

def logs_to_json(logs: List[StepLog]) -> str:
    return json.dumps([vars(log) for log in logs], indent=2)

def logs_from_json(json_str: str) -> List[StepLog]:
    return [StepLog(**d) for d in json.loads(json_str)]
```

---

## 10. Testing Patterns

### A) Detach Detector

Verify that MAML inner loop preserves the computational graph:

```python
def test_maml_graph_intact():
    """After inner loop with create_graph=True, adapted params must have grad_fn."""
    model = nn.Linear(10, 5)
    params = {n: p for n, p in model.named_parameters()}
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))

    adapted, _ = custom_inner_loop(
        params, model, x, y,
        steps=3, lr=0.01, create_graph=True,
    )

    for name, tensor in adapted.items():
        assert tensor.grad_fn is not None, (
            f"Adapted param '{name}' has no grad_fn. "
            "The inner loop detached the computational graph. "
            "This breaks second-order MAML."
        )
```

### B) Meta-Gradient Flow Test

Verify that the full MAML pipeline produces non-zero gradients on the original parameters:

```python
def test_meta_gradient_flow():
    """Outer loss from adapted params must produce gradients on original params."""
    model = nn.Linear(10, 5)
    params = {n: p.clone().requires_grad_(True) for n, p in model.named_parameters()}
    x_support = torch.randn(4, 10)
    y_support = torch.randint(0, 5, (4,))
    x_query = torch.randn(4, 10)
    y_query = torch.randint(0, 5, (4,))

    # Inner loop
    adapted, _ = custom_inner_loop(
        params, model, x_support, y_support,
        steps=3, lr=0.01, create_graph=True,
    )

    # Outer loss on query set
    query_logits = functional_forward(model, adapted, x_query)
    outer_loss = F.cross_entropy(query_logits, y_query)

    # Backward
    outer_loss.backward()

    for name, param in params.items():
        assert param.grad is not None, (
            f"Original param '{name}' has no gradient after outer backward. "
            "Meta-gradient flow is broken."
        )
        assert param.grad.abs().sum() > 0, (
            f"Original param '{name}' has zero gradient. "
            "Meta-gradient exists but is numerically zero."
        )
```

### C) FOMAML Does Not Require create_graph

Verify that FOMAML produces gradients without second-order graph:

```python
def test_fomaml_no_second_order():
    """FOMAML should produce gradients without create_graph=True."""
    model = nn.Linear(10, 5)
    params = {n: p.clone().requires_grad_(True) for n, p in model.named_parameters()}
    x_s, y_s = torch.randn(4, 10), torch.randint(0, 5, (4,))
    x_q, y_q = torch.randn(4, 10), torch.randint(0, 5, (4,))

    adapted, _ = custom_inner_loop(
        params, model, x_s, y_s,
        steps=3, lr=0.01, create_graph=False,  # FOMAML
    )

    query_logits = functional_forward(model, adapted, x_q)
    outer_loss = F.cross_entropy(query_logits, y_q)
    outer_loss.backward()

    # Gradients should exist (first-order approximation)
    for name, param in params.items():
        assert param.grad is not None, f"FOMAML: no gradient on '{name}'"
```

### D) Backend Equivalence

Verify that all three backends produce identical adapted parameters for the same input:

```python
def test_backend_equivalence():
    """All backends must produce equivalent adapted params within fp tolerance."""
    model = nn.Linear(10, 5)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))

    # Fix initial params
    params = {n: p.clone() for n, p in model.named_parameters()}

    # Run each backend
    adapted_custom, _ = custom_inner_loop(
        {k: v.clone().requires_grad_(True) for k, v in params.items()},
        model, x, y, steps=3, lr=0.01, create_graph=False,
    )

    adapted_func, _ = torch_func_inner_loop(
        {k: v.clone().requires_grad_(True) for k, v in params.items()},
        model, x, y, steps=3, lr=0.01, create_graph=False,
    )

    # Compare
    for key in adapted_custom:
        diff = (adapted_custom[key] - adapted_func[key]).abs().max()
        assert diff < 1e-5, (
            f"Backend mismatch on '{key}': max diff = {diff:.2e}. "
            "Custom and torch.func backends are not equivalent."
        )
```

### E) Reptile Weight Interpolation

Verify that Reptile's outer update moves parameters toward the adapted parameters:

```python
def test_reptile_direction():
    """Reptile meta-update must move params in the direction of adapted params."""
    model = nn.Linear(10, 5)
    params = {n: p.clone() for n, p in model.named_parameters()}

    adapted = reptile_inner_loop(params, model, x, y, steps=5, lr=0.01)

    epsilon = 0.1
    for k in params:
        old = params[k].clone()
        params[k] = params[k] + epsilon * (adapted[k] - params[k])
        # New params should be closer to adapted than old params were
        old_dist = (old - adapted[k]).norm()
        new_dist = (params[k] - adapted[k]).norm()
        assert new_dist < old_dist, (
            f"Reptile update moved '{k}' away from adapted params"
        )
```

### F) AMP Safety Regression

Verify that inner-loop fp32 enforcement prevents NaN meta-gradients:

```python
def test_amp_no_nan():
    """Inner loop under AMP autocast must not produce NaN meta-gradients."""
    model = nn.Linear(10, 5).cuda()
    params = {n: p.clone().requires_grad_(True) for n, p in model.named_parameters()}
    x = torch.randn(4, 10, device='cuda')
    y = torch.randint(0, 5, (4,), device='cuda')

    with torch.amp.autocast('cuda', enabled=True):
        adapted, _ = safe_inner_loop(
            params, model, x, y,
            steps=5, lr=0.01, create_graph=True, clip_norm=10.0,
        )
        query_logits = functional_forward(model, adapted, x)
        loss = F.cross_entropy(query_logits, y)

    loss.backward()

    for name, param in params.items():
        assert not torch.isnan(param.grad).any(), (
            f"NaN gradient on '{name}' under AMP. "
            "Inner loop fp32 enforcement is broken."
        )
```

---

## Appendix A: InnerLoopEngine Class API

```python
class InnerLoopEngine:
    """Unified inner-loop optimization engine with pluggable backends.

    Provides a single interface for MAML/FOMAML/Reptile inner-loop
    optimization across torch.func, higher, and custom SGD backends.

    Args:
        model: nn.Module -- the model architecture (used for forward pass structure)
        config: InnerLoopConfig -- inner-loop hyperparameters and backend selection
    """

    def __init__(self, model: nn.Module, config: InnerLoopConfig): ...

    @property
    def backend(self) -> str:
        """Active backend name: 'torch_func', 'higher', or 'custom'."""
        ...

    def adapt(
        self,
        params: Dict[str, torch.Tensor],
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        *,
        steps: Optional[int] = None,
        lr: Optional[float] = None,
        per_layer_lrs: Optional[Dict[str, float]] = None,
        create_graph: Optional[bool] = None,
        clip_norm: Optional[float] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[StepLog]]:
        """Perform inner-loop adaptation.

        Args:
            params: Parameter dict from model.named_parameters()
            support_x: Support set inputs, shape (N*K, ...)
            support_y: Support set labels, shape (N*K,)
            steps: Override config.inner_steps
            lr: Override config.inner_lr (base LR)
            per_layer_lrs: Per-parameter LR overrides (for LSLR)
            create_graph: Override config.create_graph
                True = MAML (second-order)
                False = FOMAML (first-order)
                None = use config default
            clip_norm: Override config.inner_clip

        Returns:
            adapted_params: Dict[str, Tensor] with adapted weights
            logs: List[StepLog] with per-step diagnostics
        """
        ...

    def adapt_reptile(
        self,
        params: Dict[str, torch.Tensor],
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        *,
        steps: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Reptile inner loop: standard SGD, no graph.

        Returns adapted params for weight-space interpolation.
        The caller computes theta += epsilon * (adapted - theta).
        """
        ...

    def self_test(self) -> Dict[str, bool]:
        """Run internal diagnostics.

        Tests:
            - graph_intact: create_graph=True preserves grad_fn
            - gradient_flow: outer backward produces non-zero grads
            - fomaml_works: create_graph=False still yields gradients
            - clip_works: gradient clipping reduces large gradients
            - amp_safe: fp32 enforcement under autocast

        Returns:
            Dict mapping test name to pass/fail boolean.
        """
        ...
```

## Appendix B: InnerLoopConfig Dataclass

```python
@dataclass
class InnerLoopConfig:
    """Configuration for the inner-loop engine.

    Aggregated into MAMLConfig; can also be used standalone.
    """

    # Steps and learning rate
    inner_steps: int = 5                    # Number of inner-loop SGD steps
    inner_lr: float = 0.01                  # Base inner-loop learning rate
    inner_clip: float = 10.0                # Gradient clip norm (0 = disabled)

    # Algorithm control
    create_graph: bool = True               # True=MAML, False=FOMAML
    algo: str = "maml"                      # "maml", "fomaml", "reptile"

    # Backend selection
    backend: str = "auto"                   # "auto", "torch_func", "higher", "custom"
    use_higher: bool = False                # Allow higher backend in auto mode

    # AMP safety
    force_fp32: bool = True                 # Force fp32 in inner loop (recommended)

    # LSLR (per-layer per-step LRs) -- managed by MAMLPlusPlusConfig
    # When LSLR is active, inner_lr serves as initialization value
    use_lslr: bool = False

    # Diagnostics
    log_steps: bool = True                  # Collect StepLog at each step
    log_accuracy: bool = True               # Compute accuracy (adds overhead)
```

## Appendix C: Utility Functions

```python
def _compute_grad_norm(grads: Dict[str, torch.Tensor]) -> float:
    """Compute L2 norm across all gradient tensors."""
    return torch.sqrt(
        sum(g.norm() ** 2 for g in grads.values() if g is not None)
    ).item()


def _compute_accuracy(model, params, x, y):
    """Compute classification accuracy using adapted params."""
    with torch.no_grad():
        logits = functional_forward(model, params, x)
        preds = logits.argmax(dim=-1)
        return (preds == y).float().mean().item()


def _clip_grad_dict(grads: Dict[str, torch.Tensor], max_norm: float):
    """Clip gradient dict by global norm. Returns new dict."""
    total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads.values()))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        return {k: g * clip_coef for k, g in grads.items()}
    return grads


def _set_param(model, name, value):
    """Set a parameter on a module by dot-separated name."""
    parts = name.split('.')
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
```

## Appendix D: Common Error Messages and Resolutions

| Error | Cause | Resolution |
|---|---|---|
| `RuntimeError: One of the differentiated Tensors appears to not have been used in the graph` | A parameter in the adapted dict was not used in the forward pass (e.g., unused bias, or a module branch that was not taken) | Pass `allow_unused=True` to `torch.autograd.grad`, or filter `params` to only include parameters that participate in the forward pass for this input |
| `RuntimeError: Trying to backward through the graph a second time` | Called `.backward()` on the outer loss without `retain_graph=True`, after already computing inner-loop gradients through the same graph | Set `retain_graph=True` on the outer `.backward()` call, or restructure to avoid double backward. Usually indicates an architectural issue. |
| `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | Adapted parameters lost their `requires_grad` flag, typically due to `.detach()` or `.data` assignment | Never use `.detach()` on adapted params during MAML. Never assign to `.data` -- create new tensors via arithmetic operations instead. |
| `NaN in meta-gradients` | fp16 inner loop, or gradient explosion from large inner LR | Enable `force_fp32=True` (default). Reduce `inner_lr`. Increase `inner_clip`. |
| `CUDA out of memory` on inner loop | Unrolled computation graph for many inner steps consumes GPU memory linearly with step count | Reduce `inner_steps`. Switch to FOMAML (`create_graph=False`) which uses less memory. Use gradient checkpointing for inner steps. |
| `KeyError` in adapted params | Parameter names from `named_parameters()` do not match keys expected by `functional_call` | Ensure parameter extraction and `functional_call` use the same model instance. Watch for module wrapping (e.g., `DataParallel`) that prepends `"module."` to parameter names. |
