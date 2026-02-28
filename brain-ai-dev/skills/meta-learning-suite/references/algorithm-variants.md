# Algorithm Variants: MAML, FOMAML, Reptile, and MAML++ Enhancements

## Overview

Three core meta-learning algorithms share the same inner-loop adaptation engine but differ in how the outer (meta) update is computed. All three learn an initialization of parameters that can rapidly adapt to new tasks with a few gradient steps. The distinctions lie in what information flows backward from the adapted parameters to the base parameters during the meta-update.

- **MAML (Model-Agnostic Meta-Learning)**: Second-order. Differentiates through the inner-loop optimization itself, retaining the full computation graph of parameter updates. The meta-gradient includes Hessian-vector products that capture how the inner learning rate and loss surface curvature interact.
- **FOMAML (First-Order MAML)**: First-order. Runs the same inner-loop adaptation but discards the computation graph through the update steps. The meta-gradient is computed at the adapted parameter point but does not include second-order terms. Equivalent to treating the inner update as a black box.
- **Reptile**: First-order, no explicit meta-gradient. Moves the base parameters toward the adapted parameters via direct weight-space interpolation. No query set is structurally required (though one is commonly used for evaluation). No `backward()` call in the standard formulation.

MAML++ (Antoniou et al., "How to Train Your MAML") introduces four enhancements that can be layered onto any of the above: per-layer per-step learned learning rates (LSLR), multi-step loss accumulation (MSL), derivative-order annealing, and per-step batch normalization handling. These enhancements address stability and sample-efficiency issues that emerge at scale.

All three algorithms converge to the same underlying goal: find parameters theta such that a few gradient steps on a new task's support set yield good performance on that task's query set. The choice among them is a compute-versus-quality tradeoff.

---

## MAML (Second-Order)

### Mathematical Formulation

Given a distribution over tasks p(T), a model f with parameters theta, and an inner-loop update operator U:

```
For each task T_i in meta-batch {T_1, ..., T_B}:
    phi_i = U(theta, D_support_i)      # s inner gradient steps with create_graph=True
    L_query_i = loss(f_{phi_i}(x_query_i), y_query_i)

Meta-loss = (1/B) * Sum_{i=1..B} L_query_i

theta <- theta - beta * grad_theta(Meta-loss)
```

The inner-loop update U performs s steps of SGD on the support set:

```
phi^(0) = theta
phi^(k+1) = phi^(k) - alpha * grad_{phi^(k)} L_support(phi^(k))    for k = 0, ..., s-1
phi_i = phi^(s)
```

The critical property: `create_graph=True` is passed to `torch.autograd.grad` during each inner step. This means the adapted parameters phi_i retain a full computation graph back through every inner-loop gradient computation to the original theta. When `meta_loss.backward()` executes, the gradients on theta include second-order terms (Hessian-vector products) that capture how theta affects the inner-loop trajectory, not just the final point.

### Implementation

```python
def maml_meta_loss(model, params, tasks, config):
    """
    Compute MAML meta-loss with second-order gradients.

    Args:
        model: nn.Module used for functional_forward
        params: dict of named parameters (the base initialization theta)
        tasks: list of Task namedtuples with .support_x, .support_y, .query_x, .query_y
        config: MAMLConfig with inner_steps, inner_lr, inner_clip

    Returns:
        meta_loss: scalar tensor with grad_fn through inner loop
        metrics: dict with post_adapt_acc, pre_adapt_acc, etc.
    """
    from collections import defaultdict
    meta_loss = 0
    metrics = defaultdict(list)

    for task in tasks:
        # Adapt: s inner steps with create_graph=True (second-order)
        adapted, inner_logs = adapt(
            params, task.support_x, task.support_y,
            steps=config.inner_steps,
            lrs=config.inner_lr,
            first_order=False,       # <-- second-order: retain graph
            clip_norm=config.inner_clip,
        )

        # Evaluate on query set using adapted params
        query_logits = functional_forward(model, adapted, task.query_x)
        query_loss = F.cross_entropy(query_logits, task.query_y)

        meta_loss += query_loss
        metrics['post_acc'].append(accuracy(query_logits, task.query_y))

    return meta_loss / len(tasks), metrics
```

The `adapt` function internally calls `torch.autograd.grad(..., create_graph=True)` at each inner step, which is what makes the outer gradient second-order. After `meta_loss.backward()`, every parameter in `params` receives gradients that account for the full inner-loop trajectory.

### The Inner-Loop Gradient Step (Second-Order)

```python
def inner_step_second_order(params, support_x, support_y, lr, model, clip_norm=None):
    """Single inner-loop gradient step with create_graph=True."""
    logits = functional_forward(model, params, support_x)
    loss = F.cross_entropy(logits, support_y)

    grads = torch.autograd.grad(
        loss,
        params.values(),
        create_graph=True,    # Retain computation graph through this grad call
        allow_unused=True,
    )

    updated = {}
    for (name, param), grad in zip(params.items(), grads):
        if grad is None:
            updated[name] = param
            continue
        if clip_norm is not None:
            grad = torch.clamp(grad, -clip_norm, clip_norm)
        updated[name] = param - lr * grad  # New tensor, connected to theta via grad_fn

    return updated, loss.item()
```

The returned `updated` dictionary contains tensors that depend on the original `params` through the gradient computation. This chain of dependencies is what enables second-order meta-gradients.

### Computational Cost

Second-order MAML requires storing the entire inner-loop computation graph in memory. For s inner steps and a model with P parameters:

- **Memory**: O(s * P) for the computation graph. Each inner step adds a backward graph node for every parameter.
- **Compute**: Roughly 2-3x the cost of a standard forward-backward pass per inner step, because the backward pass itself must be differentiable.
- **Scaling**: For deep models (>10M parameters) with many inner steps (>5), the memory cost can exceed GPU capacity. Gradient checkpointing (`torch.utils.checkpoint`) can trade compute for memory by recomputing intermediate activations during the backward pass.

### When Second-Order Matters

Second-order information is most valuable when:

- The inner learning rate alpha is large relative to the loss surface curvature (the Hessian matters for understanding how alpha interacts with the landscape).
- The number of inner steps is small (1-3), so each step must be maximally informative.
- The task distribution is diverse, requiring the initialization to sit at a point where curvature information differentiates task-specific adaptation directions.

For shallow models or many inner steps (>10), the second-order terms contribute less because the first-order signal already captures most of the useful gradient information.

---

## FOMAML (First-Order MAML)

### Mathematical Formulation

Identical structure to MAML, with one change: the inner-loop gradient computation uses `create_graph=False`.

```
For each task T_i in meta-batch:
    phi_i = U(theta, D_support_i)      # s inner steps with create_graph=False
    L_query_i = loss(f_{phi_i}(x_query_i), y_query_i)

Meta-loss = (1/B) * Sum_{i=1..B} L_query_i

theta <- theta - beta * grad_theta(Meta-loss)
```

The meta-gradient is:

```
grad_theta(Meta-loss) approx (1/B) * Sum_i grad_{phi_i}(L_query_i)
```

This is a first-order approximation: the gradient is computed at the adapted parameters phi_i, but the dependency of phi_i on theta through the inner-loop update chain is ignored. The gradient treats phi_i as if it were a constant with respect to theta, then maps the gradient back to theta's parameter space.

### Implementation

The only difference from MAML is the `first_order=True` flag passed to `adapt`:

```python
def fomaml_meta_loss(model, params, tasks, config):
    """
    Compute FOMAML meta-loss (first-order approximation).

    Identical to maml_meta_loss except first_order=True in adapt().
    """
    from collections import defaultdict
    meta_loss = 0
    metrics = defaultdict(list)

    for task in tasks:
        adapted, inner_logs = adapt(
            params, task.support_x, task.support_y,
            steps=config.inner_steps,
            lrs=config.inner_lr,
            first_order=True,        # <-- first-order: no graph through inner updates
            clip_norm=config.inner_clip,
        )

        query_logits = functional_forward(model, adapted, task.query_x)
        query_loss = F.cross_entropy(query_logits, task.query_y)

        meta_loss += query_loss
        metrics['post_acc'].append(accuracy(query_logits, task.query_y))

    return meta_loss / len(tasks), metrics
```

### The Inner-Loop Gradient Step (First-Order)

```python
def inner_step_first_order(params, support_x, support_y, lr, model, clip_norm=None):
    """Single inner-loop gradient step with create_graph=False."""
    logits = functional_forward(model, params, support_x)
    loss = F.cross_entropy(logits, support_y)

    grads = torch.autograd.grad(
        loss,
        params.values(),
        create_graph=False,   # Do not retain graph through this grad call
        allow_unused=True,
    )

    updated = {}
    for (name, param), grad in zip(params.items(), grads):
        if grad is None:
            updated[name] = param
            continue
        if clip_norm is not None:
            grad = torch.clamp(grad, -clip_norm, clip_norm)
        # Detached grad, but param still has grad_fn if it came from a previous step
        # For true first-order: the graph through the update is not retained
        updated[name] = param - lr * grad

    return updated, loss.item()
```

With `create_graph=False`, the `grads` tensors do not carry backward graph information. The updated parameters are still tensors that can receive gradients (they participate in the query-set forward pass), but the path from theta through the inner-loop updates is severed. The outer gradient only accounts for how the final adapted parameters affect the query loss, not how theta affected the adaptation trajectory.

### Computational Cost

- **Memory**: O(P) per inner step (no computation graph stored through updates). Total memory is dominated by the model size and the single query-set backward pass.
- **Compute**: Roughly 1/2 to 1/3 the compute of full MAML, because the backward pass through inner steps does not need to be differentiable.
- **Scaling**: Scales to large models and many inner steps without the OOM issues that plague second-order MAML.

### Quality vs. Cost

FOMAML is often competitive with full MAML in practice, especially when:

- The model is shallow (Conv4, 4-layer MLP) where second-order effects are modest.
- Many inner steps are used (>5), providing enough first-order signal to compensate for missing curvature information.
- The task distribution is relatively homogeneous, so the initialization does not need fine-grained curvature awareness.

FOMAML tends to underperform MAML when:

- Few inner steps (1-2) are used, making each step's efficiency critical.
- The task distribution is highly diverse, requiring the initialization to differentiate adaptation directions via curvature.
- The model is deep and the inner learning rate is large.

---

## Reptile

### Mathematical Formulation

Reptile takes a fundamentally different approach to the meta-update. Instead of computing gradients of a query-set loss, it directly interpolates between the base parameters and the adapted parameters:

```
For each task T_i in meta-batch:
    phi_i = U(theta, D_support_i)      # s inner steps (standard SGD, no graph needed)

weight_delta = (1/B) * Sum_{i=1..B} (phi_i - theta)

theta <- theta + epsilon * weight_delta
```

Equivalently: `theta <- (1 - epsilon) * theta + epsilon * (1/B) * Sum_i phi_i`

There is no query set required in the core algorithm (though a query set is commonly used for evaluation during training). There is no `backward()` call. The meta-update is a direct manipulation of parameter values.

### Why Reptile Works

Nichol et al. showed that the Reptile gradient (phi - theta) can be decomposed into:

```
phi - theta = alpha * grad_theta(L) + alpha^2 * HVP + O(alpha^3)
```

where HVP represents Hessian-vector products that capture task-specific curvature. The first term is a standard gradient descent step on the support loss. The second term captures curvature information similar to what MAML's second-order terms provide. With multiple inner steps, the Reptile update accumulates both first-order gradient information and implicit second-order curvature, making it a reasonable approximation to MAML despite its simplicity.

### Implementation

```python
def reptile_meta_update(params, tasks, config, epsilon):
    """
    Perform Reptile meta-update via weight interpolation.

    Args:
        params: dict of named parameters (base initialization theta)
        tasks: list of Task namedtuples with .support_x, .support_y
        config: MAMLConfig with inner_steps, inner_lr
        epsilon: interpolation rate (meta-step size), typically decayed over training

    Returns:
        metrics: dict with task losses and adaptation statistics
    """
    from collections import defaultdict
    weight_deltas = {k: torch.zeros_like(v) for k, v in params.items()}
    metrics = defaultdict(list)

    for task in tasks:
        # Standard SGD adaptation (no computation graph needed)
        adapted, inner_logs = adapt(
            params, task.support_x, task.support_y,
            steps=config.inner_steps,
            lrs=config.inner_lr,
            first_order=True,   # No graph needed; Reptile does not differentiate
        )

        # Accumulate weight deltas
        for k in params:
            weight_deltas[k] += (adapted[k].detach() - params[k].detach())

        metrics['inner_loss'].append(inner_logs[-1]['loss'])

    # Average and apply the Reptile update (no optimizer.step, no backward)
    with torch.no_grad():
        for k in params:
            params[k].data += epsilon * weight_deltas[k] / len(tasks)

    return metrics
```

### Epsilon Scheduling

The interpolation rate epsilon is typically decayed over training. A linear decay schedule is standard:

```python
def reptile_epsilon(epoch, total_epochs, epsilon_start=1.0, epsilon_end=0.0):
    """Linear decay of Reptile interpolation rate."""
    return epsilon_start + (epsilon_end - epsilon_start) * (epoch / total_epochs)
```

Starting epsilon too high causes oscillation; the base parameters jump too far toward individual task solutions. Starting epsilon too low causes slow convergence. A common range is `epsilon_start=0.1` decaying to `epsilon_end=0.0` over training.

### Outer Optimizer Wrapping

While Reptile is typically presented without a standard optimizer, it can be wrapped to use Adam or SGD with momentum for the outer update:

```python
def reptile_with_optimizer(params, tasks, config, outer_optimizer):
    """Reptile variant that uses an outer optimizer for the meta-update."""
    weight_deltas = {k: torch.zeros_like(v) for k, v in params.items()}

    for task in tasks:
        adapted, _ = adapt(
            params, task.support_x, task.support_y,
            steps=config.inner_steps,
            lrs=config.inner_lr,
            first_order=True,
        )
        for k in params:
            weight_deltas[k] += (adapted[k].detach() - params[k].detach())

    # Set gradients on params to the negative mean weight delta
    # (negative because optimizers minimize, and the direction toward adapted is desired)
    outer_optimizer.zero_grad()
    for k in params:
        params[k].grad = -weight_deltas[k] / len(tasks)

    outer_optimizer.step()
```

This allows using Adam's adaptive learning rates and momentum for the Reptile outer update, which often improves convergence speed.

### Computational Cost

- **Memory**: O(P) total. No computation graph is stored at any point. The only storage beyond the model itself is the accumulated weight deltas.
- **Compute**: Lowest of the three algorithms. Each inner step is a standard forward-backward-update cycle with no additional overhead.
- **Simplicity**: No query set required in the core loop. No `backward()` for the meta-update. Can be implemented in approximately 20 lines.

---

## Algorithm Comparison

| Property | MAML | FOMAML | Reptile |
|---|---|---|---|
| Gradient order | Second | First | First (implicit) |
| `create_graph` in inner loop | `True` | `False` | `False` |
| Query set required for meta-update | Yes | Yes | No (optional for monitoring) |
| Meta-update mechanism | `optimizer.step()` after `meta_loss.backward()` | `optimizer.step()` after `meta_loss.backward()` | Direct weight interpolation (or wrapped optimizer) |
| Compute cost per meta-step | High (2-3x per inner step) | Medium (1x per inner step) | Low (1x per inner step, no outer backward) |
| Memory cost | O(s * P) for graph | O(P) | O(P) |
| Sample efficiency | Best | Good | Fair |
| Implementation complexity | High | Low (one flag change from MAML) | Low |
| Sensitive hyperparameter | `inner_lr`, `inner_steps` | `inner_lr`, `inner_steps` | `epsilon`, `epsilon_decay`, `inner_lr` |
| Deep model compatibility | Requires gradient checkpointing | Native | Native |
| Supports MAML++ enhancements | Full | LSLR, MSL, BN modes | LSLR only (no query-based MSL in standard form) |

### When to Use Each

- **MAML**: Use when sample efficiency is paramount, the model is small-to-medium (<50M params), inner steps are few (1-5), and GPU memory is sufficient. Best for research settings where maximum few-shot accuracy is the goal.
- **FOMAML**: Use as the default starting point. Provides most of MAML's benefits at a fraction of the cost. Switch to MAML only if FOMAML demonstrably underperforms on the target benchmark.
- **Reptile**: Use when simplicity is valued, memory is constrained, or the task distribution is simple enough that the implicit second-order information in the weight delta is sufficient. Also useful as a pretraining warmup before switching to MAML/FOMAML.

---

## MAML++ Enhancements (Antoniou et al.)

MAML++ introduces four enhancements that address known training instabilities in standard MAML. Each enhancement is independently toggleable via configuration flags. All four can be applied simultaneously.

### Per-Layer Per-Step Learning Rates (LSLR)

#### Problem

A single scalar inner learning rate alpha applied uniformly to all layers and all inner steps is suboptimal. Different layers have different gradient scales (deeper layers often have smaller gradients due to backpropagation attenuation). Different inner steps serve different purposes (early steps make coarse adjustments; later steps refine).

#### Solution

Learn a separate learning rate for each (layer, step) pair:

```
alpha_{l,k} for layer l in {1, ..., L}, step k in {0, ..., s-1}
```

These are `nn.Parameter` tensors updated by the outer optimizer alongside the model parameters theta.

#### Implementation

```python
class LSLR(nn.Module):
    """Per-Layer Per-Step Learned Learning Rates."""

    def __init__(self, layer_names, num_steps, init_lr=0.01):
        super().__init__()
        self.layer_names = layer_names
        self.num_steps = num_steps

        # Parameter layout: one scalar per (step, layer) pair
        self.lrs = nn.ParameterDict({
            f"lr_{layer}_{step}": nn.Parameter(torch.tensor(init_lr))
            for step in range(num_steps)
            for layer in layer_names
        })

    def get_lr(self, layer_name, step):
        """Return the learning rate for a specific layer and step, clamped to safe range."""
        raw = self.lrs[f"lr_{layer_name}_{step}"]
        return torch.clamp(raw, min=1e-6, max=1.0)
```

Usage in the inner loop:

```python
def inner_step_lslr(params, support_x, support_y, step, lslr, model, first_order=False):
    """Inner step with per-layer per-step learning rates."""
    logits = functional_forward(model, params, support_x)
    loss = F.cross_entropy(logits, support_y)

    grads = torch.autograd.grad(
        loss, params.values(),
        create_graph=not first_order,
        allow_unused=True,
    )

    updated = {}
    for (name, param), grad in zip(params.items(), grads):
        if grad is None:
            updated[name] = param
            continue
        lr = lslr.get_lr(name, step)  # Per-layer, per-step
        updated[name] = param - lr * grad

    return updated, loss.item()
```

#### Initialization and Clamping

Initialize all LSLR parameters to the base `inner_lr` value. This ensures the model starts with the same behavior as scalar-LR MAML and can diverge only if the learned rates improve performance.

Clamp to `[1e-6, 1.0]` to prevent:
- Collapse to zero (which halts adaptation for that layer/step).
- Explosion above 1.0 (which causes divergent inner-loop updates).

The LSLR parameters are updated by the same outer optimizer as the model parameters. Their gradients flow through the inner loop (if second-order) or through the query loss (if first-order).

#### Parameter Count Impact

For a model with L named parameter groups and s inner steps, LSLR adds L * s scalar parameters. For a Conv4 backbone (L approximately 8) with s = 5 steps, this is 40 additional parameters -- negligible.

### Multi-Step Loss (MSL)

#### Problem

Standard MAML computes the query loss only after the final inner step. The intermediate inner steps (0 through s-2) receive gradient signal only indirectly, through the chain of computation graph dependencies. For early inner steps, this indirect signal is weak and noisy, leading to "wasted" initial adaptation steps that do not contribute meaningfully to learning.

#### Solution

Compute the query loss at every inner step and combine them with a weighting scheme:

```python
def adapt_with_msl(params, support_data, query_data, config, lslr, model):
    """Inner loop with multi-step loss accumulation."""
    total_query_loss = 0.0
    adapted = dict(params)  # Start from base params
    msl_weights = get_msl_weights(config.msl_weight_scheme, config.inner_steps)

    for step in range(config.inner_steps):
        # Inner update on support set
        adapted, inner_loss = inner_step_lslr(
            adapted, support_data.x, support_data.y,
            step=step, lslr=lslr, model=model,
            first_order=config.first_order,
        )

        # Evaluate query loss at this intermediate point
        query_logits = functional_forward(model, adapted, query_data.x)
        query_loss = F.cross_entropy(query_logits, query_data.y)
        total_query_loss += msl_weights[step] * query_loss

    return total_query_loss, adapted
```

#### Weight Schemes

Three schemes for the MSL weights:

```python
def get_msl_weights(scheme, num_steps):
    """Compute MSL weights for each inner step."""
    if scheme == "uniform":
        # Equal weight to each step
        return [1.0 / num_steps] * num_steps

    elif scheme == "linear_increase":
        # Later steps get more weight: w_k = (k+1) / sum(1..s)
        total = num_steps * (num_steps + 1) / 2
        return [(k + 1) / total for k in range(num_steps)]

    elif scheme == "learned":
        # Return None; caller uses nn.Parameter weights with softmax normalization
        return None

    else:
        raise ValueError(f"Unknown MSL weight scheme: {scheme}")
```

For learned weights:

```python
class LearnedMSLWeights(nn.Module):
    """Learned MSL weights, normalized via softmax."""

    def __init__(self, num_steps):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.ones(num_steps))

    def forward(self):
        return F.softmax(self.raw_weights, dim=0)
```

The softmax normalization prevents the weights from concentrating entirely on the last step (which would recover standard MAML) or collapsing to zero for some steps.

#### Benefits

- **Gradient signal to early steps**: Step 0 receives direct gradient signal from the query loss at step 0, not just indirect signal through steps 1 through s-1.
- **Regularization**: Evaluating at intermediate points encourages each step to be independently useful, preventing the model from relying on a single large final adjustment.
- **Diagnostic value**: The per-step query loss provides an adaptation curve that reveals whether early steps are improving or harming performance.

### Derivative-Order Annealing

#### Problem

Second-order gradients are noisy early in training when the parameters theta are far from a good initialization. The Hessian-vector products amplify noise in the gradient, causing unstable updates that can prevent convergence. First-order gradients are more stable but less informative once the model is in a reasonable region of parameter space.

#### Solution

Start training with first-order gradients (FOMAML) and transition to second-order gradients (MAML) after a specified number of epochs:

```python
def get_first_order_flag(epoch, config):
    """Determine whether to use first-order or second-order gradients."""
    if not config.use_annealing:
        return config.first_order  # Use the static setting

    if epoch < config.annealing_start_epoch:
        return True   # First-order (FOMAML) during warmup
    else:
        return False  # Second-order (MAML) after warmup
```

Usage in the training loop:

```python
for epoch in range(num_epochs):
    first_order = get_first_order_flag(epoch, config)

    for task_batch in episode_sampler:
        adapted, inner_logs = adapt(
            params, task_batch.support,
            steps=config.inner_steps,
            lrs=config.inner_lr,
            first_order=first_order,  # Annealed
        )
        # ... compute meta-loss and update
```

#### Annealing Schedule

A binary switch (first-order before epoch E, second-order after) is the standard approach from the original MAML++ paper. A smoother transition is possible by probabilistically choosing first-order vs. second-order for each task in the meta-batch, with the probability of second-order increasing over training:

```python
def stochastic_annealing(epoch, config, rng):
    """Probabilistic derivative-order annealing."""
    if epoch < config.annealing_start_epoch:
        return True  # Always first-order

    # Linear ramp from 1.0 (always first-order) to 0.0 (always second-order)
    ramp_length = config.annealing_ramp_epochs
    progress = min((epoch - config.annealing_start_epoch) / ramp_length, 1.0)
    p_first_order = 1.0 - progress

    return rng.random() < p_first_order
```

The stochastic variant avoids a discontinuous change in gradient quality at the annealing boundary, which can cause a training loss spike.

#### Typical Settings

For a 200-epoch training run:
- `annealing_start_epoch = 50` (first 25% is FOMAML warmup)
- `annealing_ramp_epochs = 50` (epochs 50-100 transition gradually)
- Epochs 100-200: full second-order MAML

### Per-Step Batch Normalization Handling

#### Problem

Batch normalization maintains running mean and variance statistics that are updated during each forward pass. In meta-learning, the inner loop performs multiple forward passes on the support set, each of which updates the BN statistics. This creates two issues:

1. **Cross-task leakage**: If BN running stats are shared across tasks in the meta-batch, the statistics from one task's support set contaminate another task's adaptation.
2. **Inner-step interference**: BN statistics computed at inner step 0 (before any adaptation) are different from those at inner step s-1 (after substantial adaptation). Using a single set of running stats for all steps mixes pre-adaptation and post-adaptation statistics.

#### Modes

Three BN handling strategies, selectable via `bn_mode` in configuration:

**`"transductive"` (default in many implementations)**

Compute BN statistics from the current batch (support + query) at each forward pass. The running statistics are not used during the inner loop. This is the simplest approach and works well when the support set is large enough for reliable batch statistics.

```python
def set_bn_transductive(model):
    """Set all BN layers to compute stats from current batch."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.track_running_stats = False
```

**`"per_step"` (most correct)**

Maintain separate BN running statistics for each inner step. Before each inner-loop forward pass, swap in the BN state corresponding to the current step index.

```python
class PerStepBN:
    """Manage per-step BN running statistics."""

    def __init__(self, model, num_steps):
        self.num_steps = num_steps
        # Save initial BN state; create copies for each step
        self.bn_states = []
        for step in range(num_steps):
            state = {}
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    state[name] = {
                        'running_mean': module.running_mean.clone(),
                        'running_var': module.running_var.clone(),
                        'num_batches_tracked': module.num_batches_tracked.clone(),
                    }
            self.bn_states.append(state)

    def apply_step_state(self, model, step):
        """Swap in BN statistics for the given inner step."""
        state = self.bn_states[step]
        for name, module in model.named_modules():
            if name in state:
                module.running_mean.copy_(state[name]['running_mean'])
                module.running_var.copy_(state[name]['running_var'])
                module.num_batches_tracked.copy_(state[name]['num_batches_tracked'])

    def save_step_state(self, model, step):
        """Save BN statistics after the given inner step forward pass."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self.bn_states[step][name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'num_batches_tracked': module.num_batches_tracked.clone(),
                }
```

Per-step BN is the most correct approach but requires O(num_steps * num_bn_layers * feature_dim) additional memory for the statistics.

**`"frozen"` (safest)**

Freeze BN running statistics to their pre-inner-loop values. During the inner loop, BN layers use the frozen running statistics rather than computing batch statistics.

```python
def set_bn_frozen(model):
    """Freeze all BN layers to use running stats (no updates during inner loop)."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()  # Uses running_mean/running_var, does not update them
            # Keep affine parameters trainable
            if module.weight is not None:
                module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)
```

Frozen BN is the safest option: no statistics leakage, no per-step storage, deterministic behavior. The cost is a slight accuracy reduction because the BN statistics do not adapt to the task-specific data distribution.

#### Recommendation

Use `"per_step"` for maximum correctness when memory permits. Use `"frozen"` when memory is constrained or when BN-related debugging issues arise. Use `"transductive"` only when the support set size (N * K) is large enough (>= 20 examples) for reliable batch statistics.

---

## Unified API

All three algorithms and their MAML++ variants share a common abstract interface:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class MetaOutput:
    loss: torch.Tensor                    # Scalar meta-loss
    metrics: Dict[str, float]             # pre_adapt_acc, post_adapt_acc, fast_gain, auac
    inner_logs: List[Dict[str, float]]    # Per-step: loss, accuracy, grad_norm, lr_stats
    adapted_params: Dict[str, torch.Tensor]  # Final adapted params (detached for monitoring)

class MetaAlgorithm(ABC):
    @abstractmethod
    def meta_step(self, model, params, task_batch, config) -> MetaOutput:
        """Execute one meta-learning step over a batch of tasks."""
        ...

class MAMLAlgorithm(MetaAlgorithm):
    def meta_step(self, model, params, task_batch, config) -> MetaOutput:
        # Second-order: create_graph=True
        ...

class FOMAMLAlgorithm(MetaAlgorithm):
    def meta_step(self, model, params, task_batch, config) -> MetaOutput:
        # First-order: create_graph=False
        ...

class ReptileAlgorithm(MetaAlgorithm):
    def meta_step(self, model, params, task_batch, config) -> MetaOutput:
        # Weight interpolation, no backward
        ...
```

Factory:

```python
def create_meta_algorithm(algo: str) -> MetaAlgorithm:
    """Create a meta-learning algorithm by name."""
    registry = {
        'maml': MAMLAlgorithm,
        'fomaml': FOMAMLAlgorithm,
        'reptile': ReptileAlgorithm,
    }
    if algo not in registry:
        raise ValueError(
            f"Unknown algorithm: {algo}. Choose from {list(registry.keys())}"
        )
    return registry[algo]()
```

The unified API ensures that all three algorithms can be swapped interchangeably in the training loop. The training script selects the algorithm via `config.algo` and calls `meta_step` without algorithm-specific branches.

---

## Outer Optimizer Integration

### MAML and FOMAML

Both MAML and FOMAML produce a differentiable `meta_loss` tensor. The outer update uses a standard PyTorch optimizer:

```python
outer_optimizer = torch.optim.Adam(model.parameters(), lr=config.outer_lr)

for epoch in range(num_epochs):
    for task_batch in episode_sampler:
        outer_optimizer.zero_grad()

        output = meta_algorithm.meta_step(model, params, task_batch, config)
        output.loss.backward()

        # Optional: clip outer gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        outer_optimizer.step()
```

When LSLR and/or learned MSL weights are used, include their parameters in the outer optimizer's parameter groups:

```python
all_params = list(model.parameters())
if config.use_lslr:
    all_params += list(lslr.parameters())
if config.msl_weights == "learned":
    all_params += list(msl_weight_module.parameters())

outer_optimizer = torch.optim.Adam(all_params, lr=config.outer_lr)
```

### Reptile

Standard Reptile does not use `optimizer.step()`. The meta-update is a direct weight modification:

```python
for epoch in range(num_epochs):
    epsilon = reptile_epsilon(epoch, num_epochs)
    metrics = reptile_meta_update(params, task_batch, config, epsilon)
```

To use an optimizer with Reptile (for momentum, adaptive learning rates), convert the weight delta into a pseudo-gradient as shown in the "Outer Optimizer Wrapping" section above.

### Outer LR Scheduling

Cosine annealing is the standard outer LR schedule for meta-learning:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    outer_optimizer, T_max=num_epochs, eta_min=1e-6
)

for epoch in range(num_epochs):
    # ... meta-training loop ...
    scheduler.step()
```

Step decay (halving the LR every N epochs) is a simpler alternative that works well for shorter training runs.

---

## Common Pitfalls per Algorithm

### MAML

- **Out-of-memory from computation graph**: The inner-loop computation graph grows with each step. For deep models or many inner steps, this can exceed GPU memory. Fix: use gradient checkpointing (`torch.utils.checkpoint.checkpoint`) for the inner-loop forward passes, or reduce `inner_steps` and compensate with LSLR.
- **NaN gradients under AMP**: Second-order gradients through the inner loop are sensitive to float16 precision. Fix: force the entire inner loop to execute in float32, even when the outer training uses AMP. Wrap the inner loop with `torch.cuda.amp.autocast(enabled=False)`.
- **Accidental detach in inner loop**: Any `.detach()`, `.data` assignment, or `torch.no_grad()` context inside the inner loop severs the computation graph and silently converts MAML to FOMAML. Fix: add a detach detector test (see Testing section).
- **Slow convergence with high inner LR**: A large inner LR causes the adapted parameters to overshoot the task-specific optimum, producing noisy meta-gradients. Fix: reduce `inner_lr` or enable per-step gradient clipping.

### FOMAML

- **May underfit complex tasks**: The missing second-order information means FOMAML cannot capture how the initialization's curvature affects adaptation quality. For complex task distributions, this can limit few-shot accuracy. Fix: increase `inner_steps` to provide more first-order signal; if insufficient, switch to full MAML.
- **Same failure mode as MAML if create_graph is accidentally True**: If the first_order flag is not properly propagated, FOMAML silently becomes MAML with higher compute cost. Fix: assert `create_graph=False` in the inner-loop gradient call when first_order mode is selected.

### Reptile

- **Sensitive to epsilon**: Too large an epsilon causes the base parameters to oscillate between task solutions. Too small an epsilon causes slow convergence. Fix: use linear decay from 0.1 to 0.0 over training; tune epsilon_start on a small validation set.
- **No query-set signal**: Standard Reptile updates use only the support set. Without query-set evaluation in the meta-update, there is no explicit few-shot generalization pressure. Fix: monitor query sets during training, and consider the Reptile + query variant that weights the update by query-set performance.
- **Harder to combine with MAML++ MSL**: Multi-step loss requires a query set at each inner step, which conflicts with Reptile's query-free nature. LSLR is compatible with Reptile; MSL and derivative-order annealing are not straightforward to apply.

### All Algorithms

- **Inner LR is the most sensitive hyperparameter**: A factor-of-2 change in inner LR can mean the difference between working and failing. Always grid-search inner_lr in `{0.001, 0.005, 0.01, 0.05, 0.1}` before tuning other hyperparameters.
- **Too many inner steps**: More inner steps is not always better. Beyond 5-10 steps, the adapted parameters may overfit to the support set, reducing query-set generalization. Monitor the adaptation curve (per-step query accuracy) and stop adding steps when query accuracy plateaus or decreases.
- **Mixing up support and query sets**: Computing the meta-loss on the support set instead of the query set defeats the purpose of meta-learning (the model memorizes instead of learning to generalize). Fix: always verify that the query set is class-disjoint or at minimum example-disjoint from the support set.

---

## Testing Each Variant

### Gradient Existence Test

Verify that meta-gradients are non-zero after one meta-step. This catches accidental detach, incorrect create_graph settings, and broken backward paths.

```python
def test_gradient_existence(algorithm, model, task_batch, config):
    """Verify that meta-gradients flow to base parameters."""
    params = dict(model.named_parameters())

    # Zero all gradients
    for p in params.values():
        if p.grad is not None:
            p.grad.zero_()

    output = algorithm.meta_step(model, params, task_batch, config)

    if algorithm != 'reptile':
        output.loss.backward()

    # Check that at least some parameters received non-zero gradients
    has_grad = False
    for name, p in params.items():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, (
        f"{algorithm}: No parameter received non-zero gradient after meta-step"
    )
```

For MAML specifically, add a second-order detector:

```python
def test_second_order_graph(model, params, task_batch, maml_config, fomaml_config):
    """Verify that MAML meta-gradients include second-order terms."""
    # Run with create_graph=True (MAML)
    output_maml = maml_meta_loss(model, params, task_batch, maml_config)
    output_maml[0].backward()
    grad_maml = {
        k: p.grad.clone() for k, p in params.items() if p.grad is not None
    }

    # Zero gradients
    for p in params.values():
        if p.grad is not None:
            p.grad.zero_()

    # Run with create_graph=False (FOMAML)
    output_fomaml = fomaml_meta_loss(model, params, task_batch, fomaml_config)
    output_fomaml[0].backward()
    grad_fomaml = {
        k: p.grad.clone() for k, p in params.items() if p.grad is not None
    }

    # MAML and FOMAML gradients should differ (second-order terms present)
    for k in grad_maml:
        if k in grad_fomaml:
            assert not torch.allclose(grad_maml[k], grad_fomaml[k], atol=1e-6), (
                f"MAML and FOMAML gradients are identical for {k} -- "
                f"second-order terms are missing"
            )
```

### Adaptation Monotonicity Test

For simple tasks (e.g., 2-way 5-shot on linearly separable data), accuracy should increase with each inner step:

```python
def test_adaptation_monotonicity(model, params, simple_task, config):
    """Verify that adaptation improves accuracy on a simple task."""
    accuracies = []
    adapted = dict(params)

    for step in range(config.inner_steps):
        # Evaluate before this step's update
        logits = functional_forward(model, adapted, simple_task.query_x)
        acc = accuracy(logits, simple_task.query_y)
        accuracies.append(acc)

        # Perform inner step
        adapted, _ = inner_step(
            adapted, simple_task.support_x, simple_task.support_y,
            lr=config.inner_lr, model=model, first_order=True,
        )

    # Final accuracy after all steps
    logits = functional_forward(model, adapted, simple_task.query_x)
    accuracies.append(accuracy(logits, simple_task.query_y))

    # At least the last accuracy should exceed the first
    assert accuracies[-1] > accuracies[0], (
        f"Adaptation did not improve accuracy: "
        f"{accuracies[0]:.3f} -> {accuracies[-1]:.3f}"
    )

    # Ideally monotonically increasing (may not hold for complex tasks)
    for i in range(1, len(accuracies)):
        if accuracies[i] < accuracies[i-1] - 0.05:  # Allow small fluctuation
            print(
                f"Warning: accuracy decreased at step {i}: "
                f"{accuracies[i-1]:.3f} -> {accuracies[i]:.3f}"
            )
```

### Algorithm Equivalence Test

MAML and FOMAML should produce the same step-0 accuracy (before any adaptation), since both start from the same initialization theta:

```python
def test_step_zero_equivalence(model, params, task_batch, maml_config, fomaml_config):
    """Verify that MAML and FOMAML produce identical pre-adaptation predictions."""
    output_maml = maml_meta_step(model, params, task_batch, maml_config)
    output_fomaml = fomaml_meta_step(model, params, task_batch, fomaml_config)

    assert abs(
        output_maml.metrics['pre_adapt_acc']
        - output_fomaml.metrics['pre_adapt_acc']
    ) < 1e-6, (
        "MAML and FOMAML disagree on pre-adaptation accuracy"
    )
```

### LSLR Clamping Test

Verify that LSLR parameters remain within the safe range after clamping:

```python
def test_lslr_clamping(lslr):
    """Verify LSLR values are within [1e-6, 1.0] after clamping."""
    for name, param in lslr.lrs.items():
        clamped = torch.clamp(param, min=1e-6, max=1.0)
        assert (clamped >= 1e-6).all(), (
            f"LSLR {name} below minimum: {param.min().item()}"
        )
        assert (clamped <= 1.0).all(), (
            f"LSLR {name} above maximum: {param.max().item()}"
        )
```

### MSL Weight Normalization Test

Verify that learned MSL weights sum to 1.0 after softmax:

```python
def test_msl_weights_normalized(msl_module):
    """Verify MSL weights sum to 1.0."""
    weights = msl_module()  # Returns softmax-normalized weights
    weight_sum = weights.sum().item()
    assert abs(weight_sum - 1.0) < 1e-5, (
        f"MSL weights sum to {weight_sum}, expected 1.0"
    )
```

---

## Appendix A: Full Training Loop Skeleton

```python
def meta_train(model, config, episode_sampler, num_epochs):
    """Complete meta-training loop supporting all three algorithms."""
    params = dict(model.named_parameters())
    algorithm = create_meta_algorithm(config.algo)

    # Setup MAML++ enhancements
    lslr = None
    if config.use_lslr:
        lslr = LSLR(list(params.keys()), config.inner_steps, config.inner_lr)
    msl_weights = None
    if config.msl_weights == "learned":
        msl_weights = LearnedMSLWeights(config.inner_steps)

    # Outer optimizer (include MAML++ parameters)
    opt_params = list(model.parameters())
    if lslr is not None:
        opt_params += list(lslr.parameters())
    if msl_weights is not None:
        opt_params += list(msl_weights.parameters())

    outer_optimizer = torch.optim.Adam(opt_params, lr=config.outer_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        outer_optimizer, T_max=num_epochs, eta_min=1e-6
    )

    for epoch in range(num_epochs):
        # Derivative-order annealing
        first_order = get_first_order_flag(epoch, config)

        for task_batch in episode_sampler.epoch(epoch):
            outer_optimizer.zero_grad()

            if config.algo == 'reptile':
                epsilon = reptile_epsilon(epoch, num_epochs)
                metrics = reptile_meta_update(params, task_batch, config, epsilon)
            else:
                output = algorithm.meta_step(
                    model, params, task_batch, config
                )
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(opt_params, max_norm=10.0)
                outer_optimizer.step()

        scheduler.step()

        # Log adaptation curves, AUAC, per-step accuracy
        log_epoch_metrics(
            epoch,
            output if config.algo != 'reptile' else metrics
        )
```

## Appendix B: Gradient Flow Diagram

```
MAML (second-order):
    theta --> inner_step_0 --> inner_step_1 --> ... --> inner_step_s --> phi
      |           |                |                        |           |
      |     create_graph=True  create_graph=True     create_graph=True |
      |           |                |                        |           |
      |           +--------+-------+------------------------+           |
      |                    |                                            |
      |              grad_fn chain (Hessian-vector products)            |
      |                    |                                            |
      +----<--- meta_loss.backward() flows through entire chain --->----+
                                                                     query_loss


FOMAML (first-order):
    theta --> inner_step_0 --> inner_step_1 --> ... --> inner_step_s --> phi
                                                                        |
                  (graph severed: create_graph=False at each step)      |
                                                                        |
    theta <--- meta_loss.backward() only reaches phi --> query_loss ----+
      |
      +--- gradient at phi mapped back to theta (no Hessian terms)


Reptile (weight interpolation):
    theta --> inner_step_0 --> inner_step_1 --> ... --> inner_step_s --> phi
                                                                        |
                  (no graph at all, standard SGD)                       |
                                                                        |
    theta <--- theta + epsilon * (phi - theta) --- direct update ------+
      |
      +--- no backward() call; weight delta is the update
```

## Appendix C: Hyperparameter Recommendations

| Hyperparameter | MAML | FOMAML | Reptile | Notes |
|---|---|---|---|---|
| `inner_lr` | 0.01 | 0.01 | 0.01 | Most sensitive HP; grid-search first |
| `inner_steps` | 3-5 | 5-10 | 10-50 | Reptile benefits from more steps |
| `outer_lr` | 0.001 | 0.001 | N/A (use epsilon) | Adam with cosine decay |
| `epsilon` (Reptile) | N/A | N/A | 0.1 -> 0.0 | Linear decay over training |
| `inner_clip` | 10.0 | 10.0 | N/A | Per-parameter gradient clipping |
| `meta_batch_size` | 4-8 | 4-16 | 4-16 | Larger batches for FOMAML/Reptile (cheaper) |
| `use_lslr` | Recommended | Recommended | Optional | Always improves or is neutral |
| `use_msl` | Recommended | Recommended | N/A | Not applicable to query-free Reptile |
| `annealing_start_epoch` | 25% of training | N/A | N/A | Only relevant when transitioning FO to SO |
| `bn_mode` | `"per_step"` | `"per_step"` | `"frozen"` | Reptile can use simpler BN handling |

## Appendix D: Complexity Summary

| Operation | MAML | FOMAML | Reptile |
|---|---|---|---|
| Inner-loop forward | O(s * F) | O(s * F) | O(s * F) |
| Inner-loop backward | O(s * B * H) | O(s * B) | O(s * B) |
| Outer backward | O(B + s * H) | O(B) | 0 |
| Peak memory | O(s * P + G) | O(P + G) | O(P) |

Where:
- `s` = inner steps
- `F` = forward pass cost
- `B` = backward pass cost
- `H` = Hessian-vector product cost (roughly equal to B)
- `P` = parameter count
- `G` = computation graph size

For a Conv4 backbone (P approximately 112K) with s = 5:
- MAML peak memory: approximately 10-15x model size (graph storage dominates)
- FOMAML peak memory: approximately 3-4x model size
- Reptile peak memory: approximately 2x model size (base params + weight deltas)

For a ResNet-12 backbone (P approximately 8M) with s = 5:
- MAML peak memory: approximately 80-120x model size (often requires gradient checkpointing)
- FOMAML peak memory: approximately 4-5x model size
- Reptile peak memory: approximately 2x model size
