# Adversarial Attacks Reference

## Overview

Adversarial attacks generate minimally-perturbed inputs that cause model misclassification. For the brain_ai system, attacks target encoder inputs (vision, text, audio) rather than internal representations. The multi-layer architecture creates unique attack surfaces: SNN temporal coding provides some natural noise tolerance, but surrogate gradients enable gradient-based attacks through the full pipeline.

This reference covers the primary attack methods used in robustness evaluation, their mathematical foundations, implementation considerations for brain_ai, and modality-specific perturbation budgets.

---

## Threat Model

Before selecting attacks, define the threat model:

- **White-box**: Attacker has full model access including gradients. This is the standard for robustness evaluation. All attacks below operate in white-box mode.
- **Black-box**: Attacker can only query the model. Transfer attacks and query-based attacks apply. Not covered in detail here but relevant for deployment.
- **Perturbation budget**: Maximum allowed modification, measured by a norm (L-inf or L2). The budget must be imperceptible to humans.
- **Targeted vs. untargeted**: Untargeted attacks aim to cause any misclassification. Targeted attacks force a specific wrong class.

For brain_ai evaluation, the standard setup is: white-box, untargeted, L-inf norm with modality-specific epsilon.

---

## FGSM (Fast Gradient Sign Method)

### Mathematical Foundation

FGSM (Goodfellow et al., 2014) is the simplest gradient-based attack. It computes a single gradient step in the direction that maximizes the loss:

```
x_adv = x + epsilon * sign(grad_x(L(f(x), y)))
```

Where:
- `x` is the clean input
- `y` is the true label
- `L` is the loss function (typically cross-entropy)
- `f` is the model
- `epsilon` is the perturbation budget
- `sign()` takes the element-wise sign of the gradient

The perturbation is clipped to ensure the adversarial example remains in the valid input range [0, 1] for images.

### Implementation for brain_ai

```python
def fgsm_attack(model, inputs, targets, epsilon, loss_fn):
    """
    FGSM attack on brain_ai inputs.

    Args:
        model: BrainAI model (or any differentiable classifier)
        inputs: Dict[str, Tensor] mapping modality to input tensor
        targets: Ground truth labels
        epsilon: Perturbation budget (L-inf)
        loss_fn: Loss function (e.g., CrossEntropyLoss)

    Returns:
        Dict[str, Tensor] of adversarial inputs
    """
    # Enable gradients on inputs
    for key in inputs:
        inputs[key].requires_grad_(True)

    # Forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

    # Backward pass
    loss.backward()

    # Generate adversarial examples
    adv_inputs = {}
    for key, x in inputs.items():
        perturbation = epsilon * x.grad.sign()
        adv_inputs[key] = (x + perturbation).clamp(0, 1).detach()

    return adv_inputs
```

### Strengths and Weaknesses

**Strengths:**
- Extremely fast: single forward + backward pass
- Good for quick robustness sanity checks
- Useful as the inner step of adversarial training (Free-AT)

**Weaknesses:**
- Weak attack: often overestimates model robustness
- Single step cannot navigate complex loss landscapes
- Susceptible to gradient masking (model appears robust but is not)
- Should never be used as the sole robustness evaluation

### Brain_ai Considerations

For brain_ai, FGSM through the SNN core requires surrogate gradients. The gradient flows through the surrogate function (ATan, FastSigmoid), which provides a smooth approximation of the step function derivative. This means FGSM gradients are approximate, making FGSM even less reliable as a sole evaluation metric.

---

## PGD (Projected Gradient Descent)

### Mathematical Foundation

PGD (Madry et al., 2018) is the iterative extension of FGSM. It performs multiple small gradient steps, projecting back onto the epsilon-ball after each step:

```
x_0 = x + uniform_noise(-epsilon, epsilon)   # Random initialization
x_{t+1} = Proj_{B(x, epsilon)}(x_t + alpha * sign(grad_{x_t}(L(f(x_t), y))))
```

Where:
- `alpha` is the step size (typically `epsilon * 2 / steps` or `2/255`)
- `Proj_{B(x, epsilon)}` projects onto the L-inf ball of radius epsilon centered at x
- Random initialization helps escape local minima

### Standard Configuration

For L-inf robustness evaluation:
- **epsilon**: 8/255 for images (standard benchmark value)
- **steps**: 20 (standard), 50 or 100 for thorough evaluation
- **step_size**: 2/255 (quarter of epsilon is common)
- **random_start**: True (always use random initialization)
- **restarts**: 1-10 (multiple random starts for reliability)

### Implementation for brain_ai

```python
def pgd_attack(model, inputs, targets, epsilon, steps, step_size, loss_fn):
    """
    PGD attack with projection onto L-inf ball.

    The attack iteratively perturbs inputs in the gradient direction,
    projecting back to the constraint set after each step.
    """
    adv_inputs = {}
    for key, x in inputs.items():
        # Random initialization within epsilon ball
        delta = torch.zeros_like(x).uniform_(-epsilon, epsilon)
        delta = delta.clamp(-epsilon, epsilon)

        for step in range(steps):
            delta.requires_grad_(True)
            x_adv = (x + delta).clamp(0, 1)

            outputs = model({**inputs, key: x_adv})
            loss = loss_fn(outputs, targets)
            loss.backward()

            # Gradient step
            grad = delta.grad.detach()
            delta = delta.detach() + step_size * grad.sign()

            # Project back to epsilon ball
            delta = delta.clamp(-epsilon, epsilon)

            # Ensure valid input range
            delta = (x + delta).clamp(0, 1) - x

        adv_inputs[key] = (x + delta).detach()

    return adv_inputs
```

### L2 Variant

For L2-bounded attacks, replace the sign function with normalized gradients and project onto the L2 ball:

```python
# L2 step
grad_norm = grad.flatten(1).norm(dim=1, keepdim=True)
grad_norm = grad_norm.view(-1, *([1] * (grad.dim() - 1)))
normalized_grad = grad / (grad_norm + 1e-12)
delta = delta + step_size * normalized_grad

# L2 projection
delta_flat = delta.flatten(1)
delta_norm = delta_flat.norm(dim=1, keepdim=True)
factor = torch.min(torch.ones_like(delta_norm), epsilon / (delta_norm + 1e-12))
delta = (delta_flat * factor).view_as(delta)
```

### Brain_ai Considerations

PGD through the brain_ai pipeline must handle:
1. **Surrogate gradients**: The SNN discrete spikes use surrogate functions for backprop. PGD still works but the gradient landscape is noisier.
2. **Multi-modal inputs**: Attack each modality independently or jointly. Joint attacks are stronger but more expensive.
3. **Workspace competition**: The global workspace attention mechanism is differentiable, so gradients flow through it normally.
4. **HTM fallback**: If using LSTM fallback, standard PGD applies. Native HTM is non-differentiable and blocks gradient flow.

---

## AutoAttack

### Overview

AutoAttack (Croce & Hein, 2020) is an ensemble of complementary attacks designed to provide reliable robustness evaluation without hyperparameter tuning:

1. **APGD-CE**: Auto-PGD with cross-entropy loss. Adaptive step size based on loss trajectory.
2. **APGD-T**: Auto-PGD with targeted DLR (Difference of Logits Ratio) loss. Targets the most confusable class.
3. **FAB (Fast Adaptive Boundary)**: Minimizes perturbation size rather than maximizing loss. Finds minimal adversarial examples.
4. **Square Attack**: Black-box, query-based. Does not use gradients. Catches gradient masking.

### APGD (Auto-PGD)

APGD improves PGD with adaptive step sizing:

```
Step size schedule:
  - Start with alpha = 2 * epsilon
  - At checkpoints (fraction of total budget), if attack is not progressing:
    - Halve the step size
    - Restart from best adversarial example found so far
  - Progress is measured by: fraction of samples successfully attacked
```

The DLR loss for targeted APGD:
```
DLR(x, y) = -(z_y - max_{i != y} z_i) / (z_pi1 - z_pi3)
```
Where z are logits and pi is the descending sort of logits. This loss is scale-invariant and avoids the saturation issues of cross-entropy.

### Implementation Strategy

For brain_ai, implement AutoAttack as a sequential pipeline:

```python
class AutoAttack:
    def __init__(self, model, epsilon, norm='Linf'):
        self.attacks = [
            APGD_CE(model, epsilon, norm, steps=100),
            APGD_T(model, epsilon, norm, steps=100),
            FAB(model, epsilon, norm),
            SquareAttack(model, epsilon, norm, queries=5000),
        ]

    def run(self, inputs, targets):
        """Run all attacks, return worst-case results."""
        remaining_correct = targets.clone()  # Track which samples survive

        for attack in self.attacks:
            # Only attack samples not yet fooled
            adv = attack(inputs[remaining_correct], targets[remaining_correct])
            # Update remaining_correct based on new predictions
            ...

        return adv_inputs, robust_accuracy
```

### When to Use

- **Always** for final robustness reporting. AutoAttack is the standard benchmark.
- **Not needed** during adversarial training inner loops (too expensive; use PGD).
- Report AutoAttack robust accuracy alongside clean accuracy.

---

## C&W (Carlini & Wagner) Attack

### Mathematical Foundation

C&W (Carlini & Wagner, 2017) formulates adversarial example generation as an optimization problem:

```
minimize ||delta||_p + c * f(x + delta)
subject to x + delta in [0, 1]
```

Where `f` is an objective function that is negative when the attack succeeds:
```
f(x') = max(Z(x')_y - max_{i != y} Z(x')_i, -kappa)
```

Here Z(x') are logits, y is the true class, and kappa is a confidence margin.

### Change of Variables

To handle the box constraint, C&W uses a change of variables:
```
x + delta = 0.5 * (tanh(w) + 1)
```

This maps the unconstrained variable `w` to the valid range [0, 1] automatically. Optimization is performed over `w` using Adam optimizer.

### Binary Search over c

The constant `c` controls the trade-off between perturbation size and attack success. C&W performs binary search over `c` to find the smallest perturbation that succeeds:

1. Start with a range [c_low, c_high]
2. Try c = (c_low + c_high) / 2
3. If attack succeeds, decrease c (try smaller perturbation)
4. If attack fails, increase c (allow larger perturbation)
5. Repeat for ~10 binary search steps

### Implementation Notes

```python
def cw_attack(model, x, y, c=1.0, kappa=0, steps=1000, lr=0.01):
    """
    C&W L2 attack.

    Uses tanh change of variables and Adam optimizer.
    """
    # Change of variables: w -> x
    w = torch.atanh(2 * x - 1)  # Inverse of 0.5*(tanh(w)+1)
    w.requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)

    best_adv = x.clone()
    best_dist = float('inf') * torch.ones(x.shape[0])

    for step in range(steps):
        x_adv = 0.5 * (torch.tanh(w) + 1)

        logits = model(x_adv)

        # f(x') objective
        real = logits.gather(1, y.unsqueeze(1)).squeeze()
        other = (logits - 1e4 * F.one_hot(y, logits.shape[1])).max(dim=1)[0]
        f_val = torch.clamp(real - other, min=-kappa)

        # L2 distance
        dist = (x_adv - x).flatten(1).norm(dim=1)

        loss = dist.sum() + c * f_val.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track best adversarial examples
        successful = (f_val < 0)
        improved = successful & (dist < best_dist)
        best_adv[improved] = x_adv[improved].detach()
        best_dist[improved] = dist[improved].detach()

    return best_adv
```

### Brain_ai Considerations

C&W is particularly useful for brain_ai because:
- It finds minimal perturbations, revealing the true decision boundary distance
- It works well through surrogate gradients (uses Adam, not sign-based steps)
- It can be adapted for L-inf by replacing L2 distance with L-inf

However, C&W is slow (1000+ optimization steps per batch) and should be used for analysis, not routine evaluation.

---

## Modality-Specific Perturbation Budgets

### Vision (epsilon = 8/255 L-inf)

The standard perturbation budget for image classification robustness:
- **8/255 L-inf**: Each pixel can change by at most 8 intensity levels out of 255. Imperceptible to humans.
- **0.5 L2**: Alternative L2 budget for pixel space.
- **Input range**: [0, 1] after normalization. Clamp after perturbation.

For brain_ai's vision encoder (ConvSNN or ViT-based):
```python
# Vision perturbation
epsilon_vision = 8.0 / 255.0  # ~0.031
step_size_vision = 2.0 / 255.0  # ~0.008
```

### Text Embedding Perturbation

Text is discrete, so perturbation operates in embedding space:
- **epsilon = 0.1 to 1.0** in embedding L2 norm (depends on embedding scale)
- Perturb the continuous embedding vectors, not discrete tokens
- Alternative: character-level or word-level substitution attacks (not gradient-based)

For brain_ai's text encoder:
```python
# Text embedding perturbation
# After embedding lookup, before transformer layers
embeddings = model.encoders['text'].embed(token_ids)
# Perturb embeddings with PGD in L2 norm
epsilon_text = 0.5  # L2 budget in embedding space
```

### Audio Waveform Perturbation

Audio perturbation budgets are measured in signal-to-noise ratio (SNR) or absolute amplitude:
- **epsilon = 0.002 to 0.01** in raw waveform amplitude (for [-1, 1] normalized audio)
- **SNR > 30dB**: Perturbation should be inaudible
- Apply to raw waveform before mel-spectrogram computation

For brain_ai's audio encoder:
```python
# Audio perturbation
epsilon_audio = 0.005  # L-inf on [-1, 1] waveform
# OR in mel-spectrogram space:
epsilon_mel = 0.1  # L-inf on log-mel features
```

### Sensor Data Perturbation

For continuous sensor inputs (robotics, control):
- **epsilon**: 1-5% of the input range, or based on sensor noise floor
- Perturbation should be within the expected measurement noise
- L2 norm is more natural for sensor data

---

## Attack Evaluation Protocol

### Standard Evaluation Pipeline

1. **Clean accuracy**: Baseline on unperturbed test set
2. **FGSM accuracy**: Quick sanity check (do not rely on this alone)
3. **PGD-20 accuracy**: Standard iterative attack (20 steps, epsilon=8/255)
4. **PGD-100 accuracy**: Stronger evaluation (100 steps)
5. **AutoAttack accuracy**: Gold standard (report this number)

### Reporting Conventions

Always report:
- Clean accuracy and robust accuracy as a pair
- Epsilon value and norm type
- Number of attack steps
- Whether random restarts were used and how many

Example reporting format:
```
Model: brain_ai (vision, minimal config)
Dataset: MNIST test set (10,000 samples)
Attack: PGD-20, L-inf, epsilon=8/255, step_size=2/255, 1 restart
Clean accuracy: 98.5%
Robust accuracy: 72.3%
```

### Common Pitfalls

1. **Gradient masking**: If FGSM has higher attack success than PGD, gradients are unreliable. Use AutoAttack (includes Square Attack, which is gradient-free).
2. **Incorrect loss**: Use the actual training loss (usually cross-entropy), not a proxy.
3. **Eval mode issues**: Ensure batch norm is in eval mode during attack. Dropout should be off.
4. **Input preprocessing**: Perturbation budget should be in the same space as the input to the model. If inputs are normalized, adjust epsilon accordingly.
5. **Floating point precision**: Use float32 for attacks, not float16. Gradient precision matters.

---

## Summary Table

| Attack | Type | Steps | Reliable? | Speed | Use Case |
|--------|------|-------|-----------|-------|----------|
| FGSM | Single-step gradient | 1 | No | Very fast | Quick check, Free-AT inner loop |
| PGD | Multi-step gradient | 20-100 | Yes (with restarts) | Moderate | Standard evaluation, adversarial training |
| AutoAttack | Ensemble (4 attacks) | 100+ each | Yes (gold standard) | Slow | Final robustness reporting |
| C&W | Optimization-based | 1000+ | Yes | Very slow | Minimal perturbation analysis |

For brain_ai, the recommended evaluation pipeline is:
1. FGSM for quick iteration during development
2. PGD-20 for routine evaluation during training
3. AutoAttack for final benchmarking and paper-ready numbers
