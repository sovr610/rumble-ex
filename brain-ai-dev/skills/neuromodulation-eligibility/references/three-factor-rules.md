# Three-Factor Weight Update Rules for Neuromodulated Plasticity

## Overview

Three-factor learning rules are the computational bridge between Hebbian synaptic plasticity and reward-driven behavioral learning. In classical Hebbian learning, weight changes depend only on two factors: presynaptic activity and postsynaptic activity. Three-factor rules add a third signal -- a global or semi-global modulator -- that gates whether the correlation between pre and post neurons actually results in a lasting weight change. This document specifies the mathematics, implementation patterns, configuration options, and integration strategies for three-factor plasticity within the brain-ai framework.

---

## 1. Three-Factor Learning Rule Fundamentals

### Canonical Form

The three-factor weight update rule takes the following canonical form:

```
delta_w_ij = lr * M(t) * e_ij(t)
```

where:

- `delta_w_ij` is the change applied to the weight connecting presynaptic neuron j to postsynaptic neuron i.
- `lr` is the plasticity learning rate, a scalar controlling the magnitude of updates.
- `M(t)` is the third-factor modulator signal at time t. This signal encodes reward, novelty, prediction error, or any behaviorally relevant scalar or vector.
- `e_ij(t)` is the eligibility trace for synapse (i, j) at time t. This trace records the recent correlation history between pre and post activity, decaying exponentially over time.

### The Eligibility Trace

The eligibility trace accumulates evidence that a synapse was recently active in a causally relevant pattern. It evolves according to:

```
e_ij(t+1) = decay * e_ij(t) + f(pre_j(t), post_i(t))
```

where `decay` is a factor in (0, 1) controlling how quickly the trace fades, and `f(pre, post)` is a correlation function. Common choices for f include:

- **Hebbian**: `f = pre_j * post_i` -- simple product of activities.
- **Spike-timing dependent**: `f = A_+ * exp(-delta_t / tau_+)` if post fires after pre, `f = -A_- * exp(delta_t / tau_-)` if pre fires after post, where `delta_t = t_post - t_pre`.
- **BCM-style**: `f = pre_j * post_i * (post_i - theta)` where theta is a sliding threshold.
- **Gradient-aligned**: `f = d(loss)/d(w_ij)` computed locally or via surrogate gradients (bridges to backprop).

### The Third Factor (Modulator)

The modulator M(t) is the gating signal. Its key property is:

```
If M(t) = 0, then delta_w_ij = 0 regardless of eligibility.
```

This gating property is what makes three-factor rules fundamentally different from two-factor Hebbian learning. The modulator decouples the timescale of neural activity (milliseconds) from the timescale of behavioral feedback (seconds to minutes). A synapse can accumulate eligibility over fast timescales, but the weight change only materializes when the modulator confirms that the recent activity pattern was behaviorally relevant.

Common modulator sources in the brain-ai framework:

| Modulator | Source | Encodes |
|-----------|--------|---------|
| Dopamine (DA) | Reward prediction error | Surprise in reward outcomes |
| Acetylcholine (ACh) | Novelty / attention signal | Stimulus unexpectedness |
| Norepinephrine (NE) | Arousal / urgency | Global alertness, high-stakes decisions |
| Serotonin (5-HT) | Temporal discounting | Patience, long-horizon credit assignment |
| TD error | Temporal difference computation | Value prediction error |
| Anomaly score | HTM anomaly detection | Sequence novelty |

### Why Three Factors

Two-factor Hebbian learning has no mechanism to assign credit over temporal delays. If a synapse fires and the reward arrives 500ms later, pure Hebbian plasticity has already moved on. Three-factor rules solve this by:

1. Recording recent activity in the eligibility trace (fast, local).
2. Waiting for the modulator to arrive (slow, global or semi-global).
3. Applying the weight change only when both trace and modulator are nonzero.

This is the synaptic mechanism underlying reinforcement learning in biological neural circuits.

---

## 2. Online Plasticity Mode

Online plasticity mode applies three-factor weight updates directly to network parameters during inference or streaming operation, without backpropagation.

### Direct Weight Modification

In online mode, weight updates are applied imperatively:

```python
w_ij += lr * M(t) * e_ij(t)
```

This modification happens in-place on the designated layer parameters. No computational graph is built. No gradients are computed. The update is purely local.

### Target Layer Specification

Specify which layers receive three-factor updates via configuration:

- `"all_eligible"` -- apply to every layer registered in the eligible layer registry.
- Explicit layer names -- a list such as `["snn.recurrent", "workspace.projector", "fast_adapter_0"]`.
- Pattern matching -- glob patterns such as `"*.fast_weight"` to target all parameters matching a naming convention.

### Application Timing

Three-factor updates can be applied at different points:

- **Every forward pass**: immediately after computing outputs, apply the update using the current modulator and eligibility. Suitable for real-time streaming.
- **Every N steps**: accumulate eligibility over N steps, then apply the aggregated update. Reduces computational overhead.
- **On modulator threshold**: apply the update only when `|M(t)| > threshold`. This suppresses updates during behaviorally neutral periods and concentrates plasticity on salient events.

### Fast Memory Adapters

Rather than modifying the full weight matrices of large layers, introduce small "fast memory" adapter matrices specifically for online adaptation:

```python
class FastAdapter(nn.Module):
    def __init__(self, dim, rank=32):
        super().__init__()
        self.down = nn.Parameter(torch.zeros(dim, rank))
        self.up = nn.Parameter(torch.zeros(rank, dim))
        # These are the only parameters modified by three-factor updates

    def forward(self, x):
        return x + x @ self.down @ self.up
```

The adapter has far fewer parameters than the main layer (controlled by `rank`), making online updates cheap. The main layer weights remain frozen during online mode; only the adapter weights are modified by the three-factor rule.

### Properties of Online Mode

- No backpropagation required. All computation is local and forward-only.
- Suitable for continual learning: the model adapts to new data without catastrophic forgetting of the base weights (only adapters change).
- Suitable for streaming inference: each new observation triggers a small adaptation.
- Latency is O(1) per synapse per step -- constant time regardless of network depth.
- The update is not differentiable and does not need to be. Use `.detach()` on eligibility traces and modulators before computing the update.

---

## 3. Hybrid Training Mode

Hybrid mode combines three-factor updates with standard backpropagation. This is the recommended training strategy for the brain-ai framework, as it allows the model to benefit from both gradient-based optimization (high sample efficiency) and bioplausible plasticity (fast adaptation, temporal credit assignment).

### Option A: Auxiliary Loss Term

Define an auxiliary loss that encourages alignment between three-factor updates and backprop gradients:

```
L_three_factor = ||delta_w_three_factor - delta_w_backprop||^2
```

Compute this loss on designated layers only. The total training loss becomes:

```
L_total = L_task + alpha * L_three_factor
```

where `alpha` controls the strength of alignment. This encourages the three-factor update rule to approximate gradient descent on the task loss, effectively training the eligibility trace and modulator to produce useful credit assignment signals.

Implementation:

```python
def auxiliary_alignment_loss(model, eligible_layers, modulator, eligibility_traces):
    loss = 0.0
    for name, param in model.named_parameters():
        if name in eligible_layers and param.grad is not None:
            # Three-factor update direction
            tf_delta = modulator * eligibility_traces[name]
            # Backprop gradient direction
            bp_delta = param.grad.detach()
            # Alignment loss (MSE between directions)
            loss += torch.mean((tf_delta - bp_delta) ** 2)
    return loss
```

### Option B: Regularizer

Add a regularization penalty that penalizes divergence between the eligibility-modulated signal and the true gradient:

```
L_reg = lambda * sum_over_layers ||e_ij * M - grad_w_ij||^2
```

This is mathematically similar to Option A but framed as a regularizer rather than an explicit loss term. The practical difference is in how it interacts with learning rate schedules and gradient clipping.

### Option C: Plasticity Controller

Train a small neural network (the "plasticity controller") that decides, on a per-layer and per-step basis, whether to apply three-factor updates, backprop updates, or a weighted mixture:

```python
class PlasticityController(nn.Module):
    def __init__(self, state_dim, num_layers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_layers)  # one gate per eligible layer
        )

    def forward(self, state):
        # state includes: modulator magnitude, eligibility norm,
        # gradient norm, training step, loss trajectory
        gates = torch.sigmoid(self.net(state))
        return gates  # 0 = pure backprop, 1 = pure three-factor
```

The controller is trained via meta-learning (e.g., MAML outer loop) or reinforcement learning (reward = downstream task improvement). This is the most flexible option but requires the most engineering effort.

### Backprop and Three-Factor Division of Labor

In hybrid mode, assign responsibilities clearly:

- **Backprop** handles the main model parameters (encoders, large weight matrices, output heads). These require high-quality gradients and benefit from Adam/SGD optimization.
- **Three-factor** handles fast adapters, recurrent connection strengths, and any parameters that need to adapt on timescales faster than a training epoch.

This division prevents the two update mechanisms from fighting each other.

---

## 4. Weight Clamping and Stability

Unconstrained three-factor updates can cause weight explosion or collapse. Implement the following stability mechanisms.

### Per-Step Delta Clamping

Clamp the magnitude of each weight change to a maximum per step:

```python
delta_w = lr * modulator * eligibility
delta_w = torch.clamp(delta_w, -max_delta, max_delta)
```

Typical values for `max_delta`: 0.01 for stable layers, 0.001 for sensitive layers (e.g., output heads).

### Absolute Weight Clamping

After applying the update, clamp the resulting weight to an allowable range:

```python
w.data += delta_w
w.data = torch.clamp(w.data, w_min, w_max)
```

Typical ranges: `[-2.0, 2.0]` for general layers, `[-1.0, 1.0]` for adapter weights, `[0.0, 1.0]` for non-negative weights (e.g., attention-like mechanisms).

### Clamp Hit Diagnostics

Track how often clamping activates. A high clamp hit rate indicates instability:

```python
clamp_hits = (delta_w.abs() > max_delta).float().mean().item()
if clamp_hits > 0.1:
    log.warning(f"Layer {name}: {clamp_hits:.1%} of updates clamped. "
                f"Consider reducing lr or increasing max_delta.")
```

Log clamp hit rates per layer per epoch. If clamp hits exceed 10% sustained, reduce the learning rate or increase the decay rate on the eligibility trace.

### Adaptive Learning Rate

Scale the learning rate inversely by the eligibility norm to prevent large updates when eligibility is high:

```python
e_norm = torch.norm(eligibility) + 1e-8
adaptive_lr = lr / max(1.0, e_norm / target_norm)
delta_w = adaptive_lr * modulator * eligibility
```

This ensures that the update magnitude stays roughly constant regardless of the eligibility trace magnitude.

### Detachment

Three-factor updates are not part of the computational graph for backpropagation. Always detach:

```python
with torch.no_grad():
    delta_w = lr * modulator.detach() * eligibility.detach()
    w.data += delta_w
```

This prevents three-factor updates from interfering with gradient computation and avoids memory leaks from retaining the computational graph.

---

## 5. Per-Layer Configuration

Different layers serve different computational roles and require different plasticity configurations.

### Configuration Schema

```python
@dataclass
class LayerPlasticityConfig:
    eligibility_type: str          # "hebbian", "stdp", "bcm", "gradient_aligned"
    modulator: str                 # "DA", "ACh", "NE", "5HT", "td_error", "anomaly"
    learning_rate: float           # per-layer lr
    decay: float                   # eligibility decay rate
    max_delta: float               # per-step clamp
    w_min: float                   # weight floor
    w_max: float                   # weight ceiling
    update_frequency: int          # apply every N steps
    enabled: bool                  # toggle on/off
```

### Example Per-Layer Assignment

| Layer | Eligibility Type | Modulator | Learning Rate | Rationale |
|-------|-----------------|-----------|---------------|-----------|
| `snn.recurrent` | STDP | DA | 0.001 | Spike-timing plasticity gated by reward |
| `workspace.attention` | Hebbian | ACh | 0.0005 | Attention strengthened by novelty |
| `decision.policy` | Gradient-aligned | DA | 0.002 | Policy updates from reward prediction error |
| `fast_adapter_0` | Hebbian | NE | 0.005 | Rapid adaptation under arousal |
| `reasoning.system1` | BCM | 5-HT | 0.0003 | Slow refinement of fast heuristics |
| `temporal.htm_cells` | STDP | Anomaly | 0.001 | Sequence learning gated by novelty |

### Eligible Layer Registry

Maintain a registry of which parameters receive three-factor updates. Parameters not in the registry are never touched by three-factor rules.

```python
class EligibleLayerRegistry:
    def __init__(self):
        self._registry: Dict[str, LayerPlasticityConfig] = {}
        self._eligibility_traces: Dict[str, torch.Tensor] = {}

    def register(self, name: str, param: nn.Parameter, config: LayerPlasticityConfig):
        self._registry[name] = config
        self._eligibility_traces[name] = torch.zeros_like(param.data)

    def is_eligible(self, name: str) -> bool:
        return name in self._registry

    def get_config(self, name: str) -> LayerPlasticityConfig:
        return self._registry[name]

    def get_trace(self, name: str) -> torch.Tensor:
        return self._eligibility_traces[name]

    def set_trace(self, name: str, trace: torch.Tensor):
        self._eligibility_traces[name] = trace

    def eligible_params(self) -> Iterator[Tuple[str, nn.Parameter, LayerPlasticityConfig]]:
        # Yield only registered parameters
        for name, config in self._registry.items():
            yield name, config
```

Register layers during model construction:

```python
def register_eligible_layers(model, registry, config_map):
    for name, param in model.named_parameters():
        if name in config_map:
            registry.register(name, param, config_map[name])
        elif config_map.get("default") and should_be_eligible(name):
            registry.register(name, param, config_map["default"])
```

Non-eligible layers (e.g., batch normalization parameters, embedding tables, output biases) remain untouched by the three-factor rule and are updated only by standard gradient-based optimization.

---

## 6. Convergence Properties

### Reward-Weighted Hebbian Convergence

Under the following conditions, the three-factor rule converges to reward-weighted Hebbian learning:

1. The eligibility trace decay `gamma` satisfies `0 < gamma < 1`.
2. The learning rate satisfies `lr -> 0` over time (standard diminishing step size).
3. The modulator M(t) has bounded variance.

In the limit, the expected weight change becomes:

```
E[delta_w_ij] = lr * E[M(t)] * E[e_ij(t)]
```

When the modulator is mean-zero (as with TD error or centered reward), the weight change is proportional to the covariance between modulator and eligibility: `Cov(M, e)`. Synapses whose activity correlates with positive reward are strengthened; those correlating with negative reward are weakened.

### Policy Gradient Equivalence

When the modulator is the TD error `delta(t) = r(t) + gamma * V(s') - V(s)` and the eligibility trace is the gradient of the log-policy `e_ij = d(log pi(a|s)) / d(w_ij)`, the three-factor rule reduces to the REINFORCE policy gradient:

```
delta_w_ij = lr * delta(t) * d(log pi(a|s)) / d(w_ij)
```

This establishes formal equivalence between three-factor plasticity and actor-critic methods under specific parameterizations.

### Stability Bound

A rough necessary condition for stability:

```
lr * max(|M(t)|) * max(|e_ij(t)|) < 1
```

If this product exceeds 1, weight updates can exceed the current weight magnitude and oscillation or divergence becomes likely. In practice, enforce this bound by choosing the learning rate as:

```
lr < 1.0 / (max_modulator * max_eligibility)
```

where `max_modulator` and `max_eligibility` are estimated from running statistics.

### Monitoring

Track the following diagnostics during training:

- **Running variance of delta_w**: compute `Var(delta_w)` over a sliding window (e.g., 1000 steps). If variance is monotonically increasing, the learning rate is too high or the eligibility decay is too slow.
- **Effective update magnitude**: `mean(|delta_w|) / mean(|w|)` should stay below 0.01 per step. Values above 0.05 indicate instability.
- **Modulator statistics**: track `mean(M)`, `std(M)`, `max(|M|)` to ensure the modulator is well-behaved. A modulator with zero variance provides no learning signal.
- **Eligibility saturation**: track `mean(|e|) / max_possible(|e|)`. If eligibility is always saturated, the decay rate is too slow.
- **Weight drift**: track `||w(t) - w(0)||` over time. Unbounded drift indicates the three-factor updates are not converging.

```python
class PlasticityMonitor:
    def __init__(self, window_size=1000):
        self.delta_w_history = deque(maxlen=window_size)
        self.clamp_hit_rates = {}

    def log_update(self, name, delta_w, weight, modulator, eligibility):
        stats = {
            "delta_w_mean": delta_w.abs().mean().item(),
            "delta_w_max": delta_w.abs().max().item(),
            "relative_update": (delta_w.abs().mean() / (weight.abs().mean() + 1e-8)).item(),
            "modulator_mag": modulator.abs().mean().item(),
            "eligibility_mag": eligibility.abs().mean().item(),
        }
        self.delta_w_history.append(stats)
        return stats

    def check_stability(self):
        if len(self.delta_w_history) < 100:
            return True
        recent = list(self.delta_w_history)[-100:]
        var_trend = [s["delta_w_mean"] for s in recent]
        # Check if variance is increasing
        first_half = sum(var_trend[:50]) / 50
        second_half = sum(var_trend[50:]) / 50
        if second_half > 2 * first_half:
            return False  # Instability detected
        return True
```

---

## 7. Integration with Meta-Learning

### Three-Factor as Inner Loop Alternative

In MAML (Model-Agnostic Meta-Learning), the inner loop performs gradient descent on a support set to adapt the model to a new task. This requires computing gradients through the inner loop, which is expensive (second-order gradients) and memory-intensive.

Three-factor learning offers an alternative inner loop that requires no backward pass:

```
Standard MAML inner loop:
    w' = w - alpha * grad(L_support, w)        # requires backward pass

Three-factor inner loop:
    w' = w + lr * M(support) * e(support)      # forward-only computation
```

The three-factor inner loop runs in O(1) time per parameter per step (no backward pass), compared to O(depth) for gradient-based inner loops. This makes it suitable for meta-learning with many inner loop steps or in real-time settings.

### Complementary Usage with MAML

Use MAML to learn a good initialization of the base weights. Use three-factor learning for online refinement after deployment:

```
Training phase (offline):
    MAML outer loop: optimize initial weights w_0
    MAML inner loop: gradient-based adaptation on tasks

Deployment phase (online):
    Start from w_0
    For each new observation:
        Compute eligibility from forward pass
        Receive modulator (reward, novelty signal)
        Apply three-factor update to fast adapters
```

This decomposition is natural: MAML excels at finding initializations that are easy to adapt from, and three-factor rules excel at performing that adaptation cheaply at runtime.

### Eligibility Traces as Learned Fast Weights

There is a deep connection between eligibility traces and the "fast weights" literature (Ba et al., 2016; Schlag et al., 2021). In fast weight systems, a "slow" weight matrix stores long-term knowledge and a "fast" weight matrix stores recent context. The fast weight matrix is updated by an outer product of recent activations:

```
Fast weight update:   A(t) = decay * A(t-1) + lr * h(t) * h(t)^T
Output:               y = (W_slow + A(t)) * x
```

The eligibility trace is formally analogous to the fast weight matrix. The three-factor rule gates the transfer from eligibility (fast weight candidate) to actual weight change (permanent fast weight) via the modulator. This provides a principled mechanism for deciding which fast weight updates to consolidate.

### Meta-Learned Plasticity Parameters

Use the MAML outer loop to optimize three-factor hyperparameters:

- Learning rates per layer
- Eligibility decay rates per layer
- Modulator routing (which neuromodulator gates which layer)
- Correlation function parameters (STDP time constants, BCM thresholds)

This creates a "learning to learn" system where the outer loop discovers the best plasticity rules and the inner loop applies them.

---

## 8. Implementation Patterns

### ThreeFactorUpdate Class

```python
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from collections import deque


@dataclass
class ThreeFactorConfig:
    default_lr: float = 0.001
    default_decay: float = 0.95
    max_delta: float = 0.01
    w_min: float = -2.0
    w_max: float = 2.0
    update_frequency: int = 1
    target_eligibility_norm: float = 1.0
    adaptive_lr: bool = True
    mode: str = "online"  # "online" or "hybrid"


class ThreeFactorUpdate:
    """Core three-factor weight update engine.

    Manages eligibility traces, applies modulated weight updates,
    and enforces stability constraints across registered layers.
    """

    def __init__(self, config: ThreeFactorConfig):
        self.config = config
        self.registry = EligibleLayerRegistry()
        self.monitor = PlasticityMonitor()
        self.step_count = 0

    def register_layer(
        self,
        name: str,
        param: nn.Parameter,
        layer_config: Optional[LayerPlasticityConfig] = None
    ):
        """Register a parameter for three-factor updates."""
        if layer_config is None:
            layer_config = LayerPlasticityConfig(
                eligibility_type="hebbian",
                modulator="DA",
                learning_rate=self.config.default_lr,
                decay=self.config.default_decay,
                max_delta=self.config.max_delta,
                w_min=self.config.w_min,
                w_max=self.config.w_max,
                update_frequency=self.config.update_frequency,
                enabled=True,
            )
        self.registry.register(name, param, layer_config)

    def update_eligibility(
        self,
        name: str,
        pre: torch.Tensor,
        post: torch.Tensor
    ):
        """Update the eligibility trace for a registered layer."""
        config = self.registry.get_config(name)
        trace = self.registry.get_trace(name)

        if config.eligibility_type == "hebbian":
            correlation = torch.outer(post.detach().flatten(),
                                       pre.detach().flatten())
        elif config.eligibility_type == "stdp":
            # Simplified STDP: positive if post follows pre
            correlation = torch.outer(post.detach().flatten(),
                                       pre.detach().flatten())
            # In full implementation, track spike times for temporal asymmetry
        elif config.eligibility_type == "bcm":
            theta = post.detach().mean()
            correlation = torch.outer(
                (post.detach() * (post.detach() - theta)).flatten(),
                pre.detach().flatten()
            )
        elif config.eligibility_type == "gradient_aligned":
            # Use surrogate gradient if available; otherwise fall back to Hebbian
            correlation = torch.outer(post.detach().flatten(),
                                       pre.detach().flatten())
        else:
            raise ValueError(f"Unknown eligibility type: {config.eligibility_type}")

        # Reshape correlation to match parameter shape if needed
        if correlation.shape != trace.shape:
            correlation = correlation.view(trace.shape)

        # Exponential decay + new correlation
        new_trace = config.decay * trace + (1 - config.decay) * correlation
        self.registry.set_trace(name, new_trace)

    def apply_update(
        self,
        name: str,
        param: nn.Parameter,
        modulator_signal: torch.Tensor
    ) -> Dict[str, float]:
        """Apply the three-factor update to a single parameter.

        Returns diagnostic statistics for monitoring.
        """
        config = self.registry.get_config(name)

        if not config.enabled:
            return {}

        if self.step_count % config.update_frequency != 0:
            return {}

        eligibility = self.registry.get_trace(name)
        M = modulator_signal.detach()

        # Compute raw update
        lr = config.learning_rate

        # Adaptive lr: scale by inverse eligibility norm
        if self.config.adaptive_lr:
            e_norm = torch.norm(eligibility) + 1e-8
            lr = lr / max(1.0, e_norm.item() / self.config.target_eligibility_norm)

        delta_w = lr * M * eligibility.detach()

        # Per-step delta clamping
        clamp_hits_before = (delta_w.abs() > config.max_delta).float().mean().item()
        delta_w = torch.clamp(delta_w, -config.max_delta, config.max_delta)

        # Apply update (no grad)
        with torch.no_grad():
            param.data += delta_w
            # Absolute weight clamping
            param.data = torch.clamp(param.data, config.w_min, config.w_max)

        # Log diagnostics
        stats = self.monitor.log_update(name, delta_w, param.data, M, eligibility)
        stats["clamp_hit_rate"] = clamp_hits_before

        return stats

    def step(self, model: nn.Module, modulator_signals: Dict[str, torch.Tensor]):
        """Apply three-factor updates to all eligible layers.

        Call this after each forward pass (online mode) or at designated
        intervals (hybrid mode).
        """
        self.step_count += 1
        all_stats = {}

        for name, param in model.named_parameters():
            if not self.registry.is_eligible(name):
                continue

            config = self.registry.get_config(name)
            mod_key = config.modulator

            if mod_key not in modulator_signals:
                continue

            stats = self.apply_update(name, param, modulator_signals[mod_key])
            if stats:
                all_stats[name] = stats

        # Stability check
        if not self.monitor.check_stability():
            import logging
            logging.warning(
                "Three-factor plasticity instability detected. "
                "Consider reducing learning rates or increasing eligibility decay."
            )

        return all_stats
```

### Online Mode Integration in Training Loop

```python
def online_training_loop(model, data_stream, three_factor_updater, neuromodulator_system):
    """Online/streaming training loop using three-factor plasticity only."""

    model.eval()  # No backprop; purely forward-only adaptation

    for observation, delayed_reward in data_stream:
        # 1. Forward pass
        with torch.no_grad():
            output, layer_activations = model.forward_with_activations(observation)

        # 2. Update eligibility traces from layer activations
        for name in three_factor_updater.registry._registry:
            if name in layer_activations:
                pre, post = layer_activations[name]
                three_factor_updater.update_eligibility(name, pre, post)

        # 3. Compute modulator signals
        modulator_signals = neuromodulator_system.compute(
            reward=delayed_reward,
            prediction=output,
            observation=observation
        )
        # modulator_signals is a dict: {"DA": tensor, "ACh": tensor, "NE": tensor, "5HT": tensor}

        # 4. Apply three-factor updates
        stats = three_factor_updater.step(model, modulator_signals)

        # 5. Log diagnostics periodically
        if three_factor_updater.step_count % 100 == 0:
            for layer_name, layer_stats in stats.items():
                print(f"  {layer_name}: "
                      f"delta_w={layer_stats['delta_w_mean']:.6f}, "
                      f"clamp_hits={layer_stats['clamp_hit_rate']:.2%}")
```

### Hybrid Mode with Auxiliary Loss

```python
def hybrid_training_step(
    model,
    batch,
    optimizer,
    three_factor_updater,
    neuromodulator_system,
    alpha=0.1
):
    """Single training step in hybrid mode.

    Combines standard backpropagation with three-factor auxiliary loss.
    """
    inputs, targets = batch

    # 1. Forward pass (with grad for backprop)
    output, layer_activations = model.forward_with_activations(inputs)

    # 2. Task loss
    task_loss = nn.functional.cross_entropy(output, targets)

    # 3. Update eligibility traces
    for name in three_factor_updater.registry._registry:
        if name in layer_activations:
            pre, post = layer_activations[name]
            three_factor_updater.update_eligibility(name, pre, post)

    # 4. Compute modulator signals
    modulator_signals = neuromodulator_system.compute(
        reward=compute_reward(output, targets),
        prediction=output,
        observation=inputs
    )

    # 5. Compute auxiliary alignment loss (Option A)
    # First, get backprop gradients for comparison
    task_loss.backward(retain_graph=True)

    alignment_loss = torch.tensor(0.0, device=output.device)
    num_eligible = 0
    for name, param in model.named_parameters():
        if three_factor_updater.registry.is_eligible(name) and param.grad is not None:
            config = three_factor_updater.registry.get_config(name)
            eligibility = three_factor_updater.registry.get_trace(name)
            M = modulator_signals[config.modulator].detach()

            tf_delta = config.learning_rate * M * eligibility.detach()
            bp_delta = param.grad.detach()

            # Normalize both to unit vectors for direction comparison
            tf_norm = tf_delta / (torch.norm(tf_delta) + 1e-8)
            bp_norm = bp_delta / (torch.norm(bp_delta) + 1e-8)

            alignment_loss += torch.mean((tf_norm - bp_norm) ** 2)
            num_eligible += 1

    if num_eligible > 0:
        alignment_loss = alignment_loss / num_eligible

    # 6. Zero gradients, compute total loss, backprop, step
    optimizer.zero_grad()
    total_loss = task_loss + alpha * alignment_loss
    total_loss.backward()
    optimizer.step()

    # 7. Apply three-factor updates to fast adapters (these are not in optimizer)
    three_factor_updater.step(model, modulator_signals)

    return {
        "task_loss": task_loss.item(),
        "alignment_loss": alignment_loss.item(),
        "total_loss": total_loss.item()
    }
```

### Per-Layer Configuration Setup

```python
def setup_three_factor_system(model):
    """Configure and return a ThreeFactorUpdate system for the model."""

    config = ThreeFactorConfig(
        default_lr=0.001,
        default_decay=0.95,
        max_delta=0.01,
        adaptive_lr=True,
        mode="hybrid"
    )

    updater = ThreeFactorUpdate(config)

    # Per-layer configuration map
    layer_configs = {
        "core.snn.recurrent.weight": LayerPlasticityConfig(
            eligibility_type="stdp",
            modulator="DA",
            learning_rate=0.001,
            decay=0.9,
            max_delta=0.01,
            w_min=-2.0, w_max=2.0,
            update_frequency=1,
            enabled=True
        ),
        "workspace.attention.weight": LayerPlasticityConfig(
            eligibility_type="hebbian",
            modulator="ACh",
            learning_rate=0.0005,
            decay=0.95,
            max_delta=0.005,
            w_min=-1.0, w_max=1.0,
            update_frequency=1,
            enabled=True
        ),
        "decision.policy_head.weight": LayerPlasticityConfig(
            eligibility_type="gradient_aligned",
            modulator="DA",
            learning_rate=0.002,
            decay=0.9,
            max_delta=0.02,
            w_min=-2.0, w_max=2.0,
            update_frequency=1,
            enabled=True
        ),
    }

    # Register fast adapters (always eligible)
    for name, param in model.named_parameters():
        if "fast_adapter" in name:
            updater.register_layer(name, param, LayerPlasticityConfig(
                eligibility_type="hebbian",
                modulator="NE",
                learning_rate=0.005,
                decay=0.9,
                max_delta=0.02,
                w_min=-1.0, w_max=1.0,
                update_frequency=1,
                enabled=True
            ))

    # Register configured layers
    for name, param in model.named_parameters():
        if name in layer_configs:
            updater.register_layer(name, param, layer_configs[name])

    return updater
```

### Step-by-Step Example: Delayed Reward Association Task

This example demonstrates three-factor learning on a task where the agent must associate a cue with an action, but the reward arrives after a delay.

**Task**: The agent sees a cue (0 or 1). It must press button A for cue 0, button B for cue 1. The reward arrives 10 steps after the action.

```python
def delayed_reward_example():
    """Demonstrate three-factor learning on delayed reward association."""

    # Simple two-layer network
    model = nn.Sequential(
        nn.Linear(2, 16),   # cue encoding
        nn.ReLU(),
        nn.Linear(16, 2),   # action logits
    )

    # Three-factor system
    config = ThreeFactorConfig(default_lr=0.01, default_decay=0.9, max_delta=0.05)
    updater = ThreeFactorUpdate(config)

    # Register output layer for three-factor updates
    for name, param in model.named_parameters():
        if "2.weight" in name:  # output layer
            updater.register_layer(name, param)

    reward_buffer = deque()  # (step, reward) pairs

    correct_count = 0
    total_count = 0

    for step in range(5000):
        # Generate cue
        cue = torch.randint(0, 2, (1,)).item()
        cue_vec = torch.zeros(2)
        cue_vec[cue] = 1.0

        # Forward pass
        logits = model(cue_vec)
        action = torch.argmax(logits).item()

        # Update eligibility: pre = cue_vec, post = logits
        for name in list(updater.registry._registry.keys()):
            updater.update_eligibility(name, cue_vec, logits)

        # Schedule delayed reward (arrives in 10 steps)
        correct = (action == cue)
        reward = 1.0 if correct else -1.0
        reward_buffer.append((step + 10, reward))

        if correct:
            correct_count += 1
        total_count += 1

        # Check for arrived rewards
        modulator_signal = 0.0
        while reward_buffer and reward_buffer[0][0] <= step:
            _, r = reward_buffer.popleft()
            modulator_signal += r

        # Apply three-factor update if modulator is nonzero
        if modulator_signal != 0.0:
            mod_tensor = torch.tensor(modulator_signal)
            updater.step(model, {"DA": mod_tensor})

        # Log progress
        if (step + 1) % 500 == 0:
            accuracy = correct_count / total_count
            print(f"Step {step+1}: accuracy = {accuracy:.2%}")
            correct_count = 0
            total_count = 0

    # Expected output: accuracy improves from ~50% to ~90%+ despite 10-step delay
    # The eligibility trace bridges the temporal gap between action and reward
```

**What happens in this example step by step**:

1. At step 0, the agent sees cue 0 and takes a random action. The eligibility trace records the correlation between the cue and the output logits. The reward is scheduled for step 10.

2. At steps 1-9, the agent continues acting. Each step updates the eligibility trace with exponential decay (`decay=0.9`), so the trace from step 0 is still partially present: `0.9^9 = 0.387` of its original magnitude.

3. At step 10, the delayed reward from step 0 arrives. The modulator signal becomes nonzero. The three-factor update fires: `delta_w = 0.01 * reward * eligibility`. Because the eligibility trace still retains information about the step-0 correlation (attenuated by `0.9^10 = 0.349`), the weight update correctly associates the cue with the action that produced the reward.

4. Over thousands of steps, the network learns to associate cue 0 with action 0 and cue 1 with action 1, despite the 10-step reward delay. A pure Hebbian rule (without eligibility traces) could not learn this association because the reward is temporally separated from the action.

---

## Summary of Critical Invariants

Maintain these invariants when implementing three-factor plasticity:

1. **Gating**: When `M(t) = 0`, no weight change occurs. This must hold exactly, not approximately.
2. **Detachment**: Three-factor updates must be detached from the autograd graph. Use `torch.no_grad()` and `.detach()` consistently.
3. **Clamping**: Every update must pass through both delta clamping and absolute weight clamping. Never skip clamping.
4. **Registry**: Only registered parameters receive three-factor updates. Unregistered parameters are invisible to the three-factor system.
5. **Monitoring**: Always track update magnitudes and clamp hit rates. Silent instability is the most dangerous failure mode.
6. **Decay**: Eligibility traces must decay. A trace with `decay=1.0` accumulates without bound and violates the stability bound.
7. **Separation**: In hybrid mode, the optimizer and the three-factor updater must operate on disjoint parameter sets, or their interactions must be explicitly managed via the auxiliary loss.
