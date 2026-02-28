# Surrogate Gradients: The Backward Pass Is the Product

## The Problem

Spiking neural networks emit binary spikes using the Heaviside step function:

```
s = H(v - v_th) = 1  if v >= v_th
                  0  otherwise
```

The forward pass is discrete and biologically faithful. The backward pass is broken: dH/dv = 0 almost everywhere, and undefined at v = v_th. Standard backpropagation through the spike function produces zero gradients everywhere, making learning impossible.

The solution is the **surrogate gradient**: during the backward pass only, replace the true derivative of the Heaviside step with a smooth, differentiable approximation. The forward pass remains binary. Only the gradient computation changes.

This is not an approximation of the forward pass. The network still emits real binary spikes during training. The surrogate gradient is a deliberate mathematical fiction introduced solely to make credit assignment tractable.

## Implementation Pattern

Every surrogate gradient in this codebase is implemented as a `torch.autograd.Function` with a split forward/backward contract.

```python
class MySurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_threshold, **params):
        ctx.save_for_backward(v_minus_threshold)
        # Store hyperparameters on ctx for backward use
        ctx.params = params
        return (v_minus_threshold >= 0).float()  # Binary spike: always Heaviside

    @staticmethod
    def backward(ctx, grad_output):
        (v_minus_threshold,) = ctx.saved_tensors
        surrogate_grad = ...  # Smooth function of v_minus_threshold
        return grad_output * surrogate_grad
```

Critical invariants:

- `forward` is ALWAYS the binary Heaviside function. Never use a sigmoid or other soft approximation in forward — that would break the spiking behavior entirely.
- `backward` is ALWAYS the smooth surrogate. This is where the approximation lives.
- The input to both passes is `v - v_th`, the membrane potential minus the threshold. Centering at zero is important: all surrogates are centered at x=0, corresponding to v = v_th.
- Call `ctx.save_for_backward(v_minus_threshold)` before returning from forward. If this is skipped, the backward pass cannot access the pre-activation values and will fail or silently produce wrong gradients.

## A) ATan Surrogate (Default)

**Mathematical form:**

```
σ'(x) = α / (2 · (1 + (π · α · x)²))
```

**Implementation reference** (`brain_ai/core/neurons.py`, lines 13-50):

```python
@staticmethod
def backward(ctx, grad_output):
    (v_minus_threshold,) = ctx.saved_tensors
    alpha = ctx.alpha
    surrogate_grad = alpha / (2.0 * (1.0 + (math.pi * alpha * v_minus_threshold) ** 2))
    return grad_output * surrogate_grad, None
```

**Properties:**

- Bell-shaped curve, symmetric around x=0
- Maximum gradient at x=0: α/2
- Width controlled inversely by α: larger α compresses the bell, concentrating gradient signal near threshold
- Default α=2.0 gives maximum gradient = 1.0, which is the normalization target
- Falls off as 1/x² in the tails, providing finite but nonzero gradient for neurons moderately far from threshold
- Numerically stable across a wide range of membrane potential values
- Most stable for general training with Adam or AdamW optimizers

**When to use:** Start here. ATan is the default because it is well-behaved across learning rates, stable across training phases, and its maximum gradient of 1.0 at default settings requires no additional scaling adjustments.

## B) Fast Sigmoid Surrogate

**Mathematical form:**

```
σ'(x) = slope / (2 · (1 + slope · |x|)²)
```

**Implementation reference** (`brain_ai/core/neurons.py`, lines 51-90):

```python
@staticmethod
def backward(ctx, grad_output):
    (v_minus_threshold,) = ctx.saved_tensors
    slope = ctx.slope
    surrogate_grad = slope / (2.0 * (1.0 + slope * v_minus_threshold.abs()) ** 2)
    return grad_output * surrogate_grad, None
```

**Properties:**

- Heavier tails than ATan due to the absolute value in the denominator — the denominator grows as |x| rather than x², producing slower decay
- Maximum gradient at x=0: slope/2
- Default slope=25.0 gives maximum gradient = 12.5 — this is a significant normalization mismatch relative to ATan's default of 1.0
- For max-gradient-1 normalization: use slope=2.0
- The heavier tails mean neurons whose membrane potential is far from threshold still receive meaningful gradient signal

**When to use:** Use when neurons have wide membrane potential distributions and need gradient signal to propagate even when voltage is far from threshold. This is most relevant during early training when randomly initialized weights produce chaotic membrane potentials. Adjust slope to 2.0 for normalized gradients, or keep at 25.0 only if the learning rate has been tuned specifically for that gradient scale.

## C) Straight-Through Estimator (STE)

**Mathematical form:**

```
σ'(x) = 1.0  (constant everywhere)
```

**Implementation reference** (`brain_ai/core/neurons.py`, lines 91-117 via `get_surrogate`):

```python
@staticmethod
def backward(ctx, grad_output):
    return grad_output, None  # Gradient passes through unchanged
```

**Properties:**

- Simplest possible surrogate gradient — the gradient of the spike function is treated as 1.0 everywhere
- Gradient magnitude is completely independent of how far the membrane potential is from threshold
- A neuron sitting at v = 0 with v_th = 1.0 receives the same gradient as a neuron sitting at v = 0.99
- No hyperparameters to tune
- Maximum gradient is exactly 1.0 by definition

**When to use:** Use for quick experiments, baselines, or when establishing whether surrogate gradient choice matters at all for a given architecture. Also useful when gradient-explosion is not a concern and simplicity is prioritized. Not recommended for production training due to the inability to distinguish near-threshold from far-threshold neurons.

## D) Max-Gradient-1 Normalization

This principle, established in the SpikingJelly framework, states that surrogate gradient parameters should be set so the maximum gradient value is approximately 1.0. The maximum occurs at x=0 (v = v_th) for all smooth surrogates.

**Normalization table:**

| Surrogate | Parameter | For max_grad=1 | Default (existing) | Existing max_grad |
|---|---|---|---|---|
| ATan | alpha | alpha=2.0 | alpha=2.0 | 1.0 (correct) |
| FastSigmoid | slope | slope=2.0 | slope=25.0 | 12.5 (miscalibrated) |
| StraightThrough | none | N/A | N/A | 1.0 (always correct) |

**Why this matters:**

Unnormalized surrogates interact destructively with learning rate schedules and surrogate swapping. Consider what happens when training starts with ATan (max_grad=1.0) and partway through training the surrogate is changed to FastSigmoid at the default slope=25.0 (max_grad=12.5):

- The effective learning rate for the spiking layers increases by a factor of 12.5x instantaneously
- Weights that were trained under stable gradient magnitudes suddenly receive 12.5x larger updates
- This frequently causes loss spikes, divergence, or requires manual learning rate reduction

Normalizing all surrogates to max_grad≈1.0 makes them interchangeable without also changing the learning rate. It also makes multi-surrogate experiments comparable: if ATan at α=2.0 achieves 80% accuracy and FastSigmoid at slope=2.0 achieves 82%, that comparison is meaningful. If FastSigmoid uses slope=25.0, the comparison is confounded by the 12.5x gradient scaling difference.

**Normalization formula:** For any surrogate with maximum gradient g at default parameters, normalize by adjusting the parameter until the peak gradient equals 1.0. For ATan: peak = α/2, so set α=2.0. For FastSigmoid: peak = slope/2, so set slope=2.0.

## E) How Slope Affects Learning

The width of the surrogate gradient window — how far from threshold a neuron must be before it stops receiving gradient signal — fundamentally affects learning dynamics.

**Wide window (low slope or low alpha):**

- Gradient signal reaches neurons whose membrane potential is far from threshold
- Early in training, randomly initialized weights produce membrane potentials distributed broadly across the state space
- A wide window ensures most neurons participate in gradient updates even when few neurons are close to threshold
- Risk: gradient signal is diffuse and does not strongly reinforce threshold-crossing behavior

**Narrow window (high slope or high alpha):**

- Gradient concentrated near threshold; neurons far from threshold receive negligible updates
- Once the network is roughly trained and many neurons are operating near their threshold, this creates sharp selection pressure
- Rewards neurons that are consistently close to firing; penalizes neurons stuck in saturation
- Risk: if few neurons are near threshold (common in early training), gradients vanish and learning stalls

**Curriculum strategy:** Start training with a wide surrogate to ensure gradient signal reaches all neurons regardless of initial membrane potential distribution. As training progresses and the network organizes, anneal toward a narrower surrogate to sharpen threshold-crossing selectivity. This is analogous to temperature annealing in simulated annealing or the gumbel-softmax temperature schedule.

**Practical training phase table:**

| Training Phase | Recommended Width | ATan alpha | FastSigmoid slope (normalized) | Rationale |
|---|---|---|---|---|
| Early (epochs 1-30%) | Wide | alpha=0.5 to 1.0 | slope=0.5 to 1.0 | Random weights need broad gradient coverage |
| Mid (epochs 30-70%) | Medium | alpha=2.0 (default) | slope=2.0 (default) | Network organizing; standard coverage |
| Late (epochs 70-100%) | Narrow | alpha=4.0 to 8.0 | slope=4.0 to 8.0 | Fine-tuning threshold selectivity |

Note: when annealing, keep the max_grad-1 normalization invariant by scaling both ATan alpha and FastSigmoid slope together. Do not mix wide-window ATan with narrow-window FastSigmoid in the same experiment without accounting for gradient magnitude differences.

## F) Gradient Flow Through Spike Chains

In a multi-layer spiking network, gradients flow backward through chains of surrogate functions. The full gradient from loss to first-layer weights involves a product of surrogate gradient terms from every layer:

```
∂L/∂W_1 = ∂L/∂s_L · σ'_L(x_L) · ∂v_L/∂s_{L-1} · σ'_{L-1}(x_{L-1}) · ... · σ'_1(x_1) · ∂i_1/∂W_1
```

where σ'_k(x_k) is the surrogate gradient evaluated at layer k's membrane potential minus threshold.

**The gradient magnitude bound:**

Each surrogate evaluation multiplies the gradient by a value between 0 and its maximum. In the worst case (every neuron at threshold, maximum gradient applied at every layer), the product of K surrogate terms is max_grad^K.

With K=10 layers and max_grad=12.5 (FastSigmoid at slope=25.0): 12.5^10 ≈ 9.3 × 10^10. This is catastrophic gradient explosion.

With K=10 layers and max_grad=1.0 (normalized surrogates): 1.0^10 = 1.0. Gradient magnitude is bounded regardless of network depth.

With K=10 layers and max_grad=0.5: 0.5^10 ≈ 0.001. Gradient vanishing — deep layers learn slowly or not at all.

This is precisely why max-gradient-1 normalization prevents both explosion (max_grad > 1) and vanishing (max_grad < 1). The normalization target of 1.0 is not arbitrary — it is the fixed point that keeps gradient magnitudes stable through arbitrarily deep spike chains under worst-case analysis.

In practice, neurons are not all at threshold simultaneously, so average gradient magnitudes are lower than the worst-case bound. But the worst case determines whether training is numerically stable, and max_grad=1.0 normalization ensures the worst case is harmless.

## G) Surrogate Selection Guide

| Scenario | Recommended | Why |
|---|---|---|
| Default / first attempt | ATan, alpha=2.0 | Stable, normalized, works with standard Adam learning rates |
| Convergence issues, sparse spikes early | FastSigmoid, slope=2.0 | Heavier tails provide gradient to far-threshold neurons |
| Gradient debugging | StraightThrough | Eliminates surrogate as variable; any gradient issues are architectural |
| Hardware/neuromorphic deployment | ATan | Most studied in deployment contexts; predictable behavior |
| Research: surrogate comparison | All three, normalized | Normalize all to max_grad=1.0 before comparing |
| Deep networks (>8 spike layers) | ATan, alpha=2.0 | Max_grad=1.0 critical for deep chains; ATan default already correct |
| Shallow networks (1-3 spike layers) | Any | Gradient chain is short; explosion/vanishing less severe |
| Fine-tuning pretrained spiking network | ATan, alpha=4.0-8.0 | Narrow window for threshold selectivity after coarse training |

## H) Testing Surrogates

**1. Gradient presence test**

Verify that gradients are non-zero and flow back to learnable parameters:

```python
import torch
from brain_ai.core.neurons import get_surrogate

surrogate_fn = get_surrogate('atan', alpha=2.0)
v = torch.randn(4, 256, requires_grad=True)
v_th = torch.zeros(1)
spikes = surrogate_fn(v - v_th)
loss = spikes.sum()
loss.backward()

assert v.grad is not None, "No gradient reaching membrane potential"
assert v.grad.abs().sum() > 0, "Gradient is zero everywhere"
print(f"Gradient present: max={v.grad.abs().max():.4f}, mean={v.grad.abs().mean():.4f}")
```

**2. Surrogate swap test**

Verify that different surrogates produce numerically different gradients, confirming the surrogate is actually being used:

```python
v = torch.randn(4, 256, requires_grad=True)
v_fixed = v.detach().clone().requires_grad_(True)

spikes_atan = get_surrogate('atan', alpha=2.0)(v)
spikes_atan.sum().backward()
grad_atan = v.grad.clone()

spikes_ste = get_surrogate('straight_through')(v_fixed)
spikes_ste.sum().backward()
grad_ste = v_fixed.grad.clone()

assert not torch.allclose(grad_atan, grad_ste), "Surrogates produced identical gradients — swap not working"
print(f"ATan max grad: {grad_atan.abs().max():.4f}")
print(f"STE max grad: {grad_ste.abs().max():.4f}")
```

**3. Finite difference spot-check**

Compare autograd surrogate gradient to a numerical finite difference approximation on a small network. This validates that the backward implementation is correct:

```python
eps = 1e-4
x = torch.tensor([0.0], requires_grad=True)  # At threshold — maximum gradient point
surrogate_fn = get_surrogate('atan', alpha=2.0)

# Autograd gradient
y = surrogate_fn(x)
y.backward()
autograd_grad = x.grad.item()

# Finite difference (note: finite diff of Heaviside is 0, so use surrogate analytically)
# Instead verify the backward formula directly: α / (2*(1+(π*α*x)²)) at x=0
import math
alpha = 2.0
expected_grad = alpha / 2.0  # At x=0: α/(2*(1+0)) = α/2
print(f"Autograd gradient at threshold: {autograd_grad:.6f}")
print(f"Expected (α/2): {expected_grad:.6f}")
assert abs(autograd_grad - expected_grad) < 1e-5, "ATan backward formula incorrect"
```

**4. Gradient magnitude normalization test**

Verify that the maximum gradient of each surrogate matches the expected normalization target:

```python
x = torch.linspace(-2.0, 2.0, 10000, requires_grad=True)
surrogate_fn = get_surrogate('atan', alpha=2.0)
y = surrogate_fn(x)
y.sum().backward()

max_grad = x.grad.abs().max().item()
print(f"ATan alpha=2.0 max gradient: {max_grad:.4f}")
assert abs(max_grad - 1.0) < 0.01, f"ATan alpha=2.0 should have max_grad≈1.0, got {max_grad}"
```

Run equivalent tests for FastSigmoid at slope=2.0 and StraightThrough, verifying all produce max_grad≈1.0 under normalized settings.

## I) Anti-Patterns

**Using soft sigmoid in forward**

```python
# WRONG: This breaks spiking behavior entirely
def forward(ctx, v_minus_threshold):
    soft = torch.sigmoid(v_minus_threshold)
    ctx.save_for_backward(soft)
    return soft  # Not binary! Network no longer spikes.
```

The forward pass must be binary Heaviside. If a soft forward pass is used, the network is no longer a spiking neural network — it becomes a sigmoid network with a misleadingly named surrogate.

**Different surrogates in different layers without normalization**

Using ATan (max_grad=1.0) in early layers and FastSigmoid at slope=25.0 (max_grad=12.5) in later layers creates inconsistent gradient scaling. The later layers effectively have a 12.5x larger learning rate. Apply normalization to all surrogates before mixing them in a single network.

**FastSigmoid slope=25.0 with learning rate tuned for ATan**

If ATan at alpha=2.0 was used to tune the learning rate, switching to FastSigmoid at slope=25.0 (the existing default) multiplies effective gradient magnitude by 12.5. The learning rate is now 12.5x too large for the spiking layers. Either re-tune the learning rate or normalize FastSigmoid to slope=2.0.

**Not saving v_minus_threshold in forward**

```python
# WRONG: ctx.saved_tensors will be empty in backward
def forward(ctx, v_minus_threshold):
    return (v_minus_threshold >= 0).float()

def backward(ctx, grad_output):
    (v_minus_threshold,) = ctx.saved_tensors  # Raises IndexError or returns wrong tensor
```

Always call `ctx.save_for_backward(v_minus_threshold)` before returning from forward. Omitting this will either raise an error or — worse — silently produce incorrect gradients if saved tensors from a previous layer are accidentally reused.

**Detaching spikes before they reach the loss**

```python
# WRONG: Detach breaks the computational graph
spikes = surrogate_fn(v - v_th).detach()  # No gradient flows past this point
output = linear(spikes)
loss = criterion(output, target)
loss.backward()  # Gradients stop at the detach; earlier layers get no gradient
```

Spikes must remain attached to the computational graph from forward through to the loss. The surrogate gradient mechanism depends on PyTorch's autograd graph connecting the loss back through the spike function to the membrane potential and weights. Any `.detach()` call on spikes, or any operation that breaks the graph, kills all gradients in the preceding spike layers.

**Treating surrogate selection as a minor implementation detail**

The choice of surrogate gradient and its normalization is one of the most consequential hyperparameter decisions in a spiking network. It directly controls gradient magnitude through every layer, interacts with learning rate schedules, and determines whether deep spike chains can learn at all. Treat surrogate selection with the same care applied to architecture decisions, and always verify gradient properties empirically using the tests in section H.
