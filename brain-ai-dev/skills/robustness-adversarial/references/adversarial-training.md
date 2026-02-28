# Adversarial Training Reference

## Overview

Adversarial training improves model robustness by incorporating adversarial examples during the training process. The key insight is that a model trained only on clean data learns decision boundaries that are close to the data manifold, making it vulnerable to small perturbations. Adversarial training pushes decision boundaries away from the data, creating a margin of robustness.

For brain_ai, adversarial training should target the encoder inputs (vision, text, audio) rather than internal representations. The SNN core, HTM, and workspace layers inherit robustness from robust encoders. This reference covers the three main adversarial training methods: PGD-AT, TRADES, and Free-AT, along with curriculum strategies for training stability.

---

## PGD-AT (PGD Adversarial Training)

### Mathematical Foundation

PGD-AT (Madry et al., 2018) solves the min-max optimization problem:

```
min_theta E_{(x,y)~D} [max_{||delta||_inf <= epsilon} L(f_theta(x + delta), y)]
```

The outer minimization trains model parameters theta to minimize loss. The inner maximization finds the worst-case perturbation delta within the epsilon-ball.

In practice, the inner maximization is approximated by running PGD for K steps.

### Training Loop

```python
def pgd_at_train_step(model, x, y, optimizer, epsilon, pgd_steps, step_size):
    """
    Single PGD-AT training step.

    1. Generate adversarial examples via PGD (inner maximization)
    2. Compute loss on adversarial examples
    3. Update model parameters (outer minimization)
    """
    model.train()  # Keep in train mode but handle BN carefully

    # Inner maximization: PGD attack
    delta = torch.zeros_like(x).uniform_(-epsilon, epsilon)
    delta.requires_grad_(True)

    for step in range(pgd_steps):
        x_adv = (x + delta).clamp(0, 1)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()

        # PGD step
        delta_grad = delta.grad.detach()
        delta = delta.detach() + step_size * delta_grad.sign()
        delta = delta.clamp(-epsilon, epsilon)
        delta = (x + delta).clamp(0, 1) - x
        delta.requires_grad_(True)

    # Outer minimization: train on adversarial examples
    x_adv = (x + delta.detach()).clamp(0, 1)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

### Hyperparameters

Standard settings for vision at epsilon=8/255:

| Parameter | Standard Value | Range | Notes |
|-----------|---------------|-------|-------|
| epsilon | 8/255 | 2/255 - 16/255 | L-inf perturbation budget |
| pgd_steps | 7-10 | 3-20 | More steps = stronger attack = slower training |
| step_size | 2/255 | epsilon/4 to epsilon/2 | Step size for PGD inner loop |
| epochs | 100-200 | - | AT needs more epochs than standard training |
| learning_rate | 0.1 | 0.01-0.1 | Standard SGD with cosine decay |
| weight_decay | 5e-4 | 1e-4 to 5e-3 | Regularization |

### Accuracy-Robustness Tradeoff

PGD-AT inherently trades clean accuracy for robust accuracy:
- Clean accuracy typically drops 5-15% compared to standard training
- Robust accuracy at epsilon=8/255 is typically 40-60% on CIFAR-10
- This tradeoff is fundamental and cannot be fully eliminated

To control the tradeoff, mix clean and adversarial examples:
```python
# Mixed training: lambda * adv_loss + (1 - lambda) * clean_loss
loss = lam * F.cross_entropy(model(x_adv), y) + (1 - lam) * F.cross_entropy(model(x), y)
```

### Brain_ai Considerations

For brain_ai, PGD-AT should:
1. **Apply perturbations to encoder inputs only**: Perturb the raw vision/text/audio inputs, not the internal representations after encoding.
2. **Freeze downstream layers during inner loop**: The PGD inner loop only needs encoder + classifier gradients.
3. **Use surrogate gradients**: The SNN core requires surrogate gradients for the inner PGD loop to work.
4. **Phase-specific training**: Adversarial training in phases 1-2 (encoders) provides robustness that propagates through the pipeline.

---

## TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)

### Mathematical Foundation

TRADES (Zhang et al., 2019) separates the training objective into two terms:

```
min_theta E_{(x,y)} [L(f_theta(x), y) + beta * max_{||delta||<=epsilon} KL(f_theta(x) || f_theta(x + delta))]
```

- **First term**: Cross-entropy on clean examples (maintains clean accuracy)
- **Second term**: KL divergence between clean and adversarial predictions (encourages consistent predictions)
- **beta**: Controls the accuracy-robustness tradeoff

The key insight is that clean accuracy and robustness are optimized separately. The beta parameter explicitly controls the tradeoff.

### Implementation

```python
def trades_loss(model, x, y, epsilon, pgd_steps, step_size, beta=6.0):
    """
    TRADES loss: clean CE + beta * KL(clean || adversarial).

    Args:
        model: Neural network
        x: Clean inputs
        y: True labels
        epsilon: Perturbation budget
        pgd_steps: Number of PGD steps for inner maximization
        step_size: PGD step size
        beta: Regularization parameter (higher = more robust, less accurate)

    Returns:
        Total TRADES loss
    """
    # Get clean predictions (detached for inner loop)
    with torch.no_grad():
        clean_logits = model(x)
        clean_probs = F.softmax(clean_logits, dim=1)

    # Inner maximization: find adversarial examples that maximize KL divergence
    delta = torch.zeros_like(x).uniform_(-epsilon, epsilon)
    delta.requires_grad_(True)

    for step in range(pgd_steps):
        x_adv = (x + delta).clamp(0, 1)
        adv_logits = model(x_adv)
        adv_probs = F.log_softmax(adv_logits, dim=1)

        # Maximize KL divergence
        kl_loss = F.kl_div(adv_probs, clean_probs, reduction='batchmean')
        kl_loss.backward()

        delta_grad = delta.grad.detach()
        delta = delta.detach() + step_size * delta_grad.sign()
        delta = delta.clamp(-epsilon, epsilon)
        delta = (x + delta).clamp(0, 1) - x
        delta.requires_grad_(True)

    # Compute final TRADES loss
    x_adv = (x + delta.detach()).clamp(0, 1)

    # Clean cross-entropy
    clean_logits = model(x)
    ce_loss = F.cross_entropy(clean_logits, y)

    # KL regularization
    adv_logits = model(x_adv)
    kl_loss = F.kl_div(
        F.log_softmax(adv_logits, dim=1),
        F.softmax(clean_logits.detach(), dim=1),
        reduction='batchmean'
    )

    return ce_loss + beta * kl_loss
```

### Beta Parameter

The beta parameter controls the tradeoff:

| Beta | Clean Acc | Robust Acc | Behavior |
|------|-----------|------------|----------|
| 0 | Highest | Lowest | Standard training (no robustness) |
| 1 | High | Low | Mild robustness regularization |
| 6 | Medium | Medium | Standard TRADES (recommended) |
| 10 | Low | High | Strong robustness, sacrifices accuracy |

The recommended starting value is beta=6.0. Tune based on the specific accuracy-robustness requirement.

### Advantages Over PGD-AT

1. **Explicit tradeoff control**: Beta directly controls accuracy vs. robustness
2. **Better clean accuracy**: Clean CE term preserves standard accuracy better than PGD-AT
3. **Smoother optimization**: KL divergence is a smoother objective than adversarial CE
4. **Theoretical guarantees**: TRADES has provable connections to adversarial risk bounds

---

## Free-AT (Free Adversarial Training)

### Motivation

PGD-AT is expensive: each training step requires K forward-backward passes for the inner PGD loop plus one for the outer update. Free-AT (Shafahi et al., 2019) eliminates this overhead by reusing the backward pass gradient.

### Key Idea

Instead of running a full PGD attack each step, Free-AT:
1. Maintains a running perturbation delta across minibatch replays
2. Each replay: update both delta (attack) and model parameters (defense) simultaneously
3. The same gradient computation serves both purposes

### Implementation

```python
class FreeATTrainer:
    def __init__(self, model, epsilon, replays=4):
        self.model = model
        self.epsilon = epsilon
        self.replays = replays
        self.delta = None  # Maintained across steps

    def train_step(self, x, y, optimizer):
        """
        Free-AT training step with M replays.

        For each minibatch, replay M times:
          1. Forward pass with x + delta
          2. Backward pass
          3. Update delta (attack) using input gradient
          4. Update model parameters (defense) using parameter gradients
        """
        if self.delta is None or self.delta.shape != x.shape:
            self.delta = torch.zeros_like(x)

        total_loss = 0.0

        for replay in range(self.replays):
            # Forward pass with perturbation
            x_adv = (x + self.delta.detach()).clamp(0, 1)
            x_adv.requires_grad_(True)

            logits = self.model(x_adv)
            loss = F.cross_entropy(logits, y)

            # Backward pass (computes gradients for both model and input)
            optimizer.zero_grad()
            loss.backward()

            # Update perturbation (attack step)
            input_grad = x_adv.grad.detach()
            self.delta = self.delta + self.epsilon * input_grad.sign()
            self.delta = self.delta.clamp(-self.epsilon, self.epsilon)
            self.delta = (x + self.delta).clamp(0, 1) - x

            # Update model parameters (defense step)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / self.replays
```

### Tradeoffs

| Property | PGD-AT | TRADES | Free-AT |
|----------|--------|--------|---------|
| Training cost | K+1 forward-backward | K+2 forward-backward | M forward-backward (reused) |
| Robustness | Strong | Strong (tunable) | Slightly weaker |
| Clean accuracy | Lower | Higher | Moderate |
| Memory | Standard | Standard | Stores delta |
| Simplicity | Simple | Moderate | Simple |

Free-AT with 4-8 replays achieves comparable robustness to PGD-AT at 2-3x lower computational cost.

---

## Curriculum Adversarial Training

### Motivation

Starting adversarial training with full-strength attacks (epsilon=8/255) from epoch 1 can cause training collapse, especially for complex architectures like brain_ai. Curriculum adversarial training gradually increases the attack strength.

### Schedule Types

#### Linear Warmup

```python
def linear_epsilon_schedule(epoch, warmup_epochs, target_epsilon):
    """Linearly increase epsilon during warmup."""
    if epoch < warmup_epochs:
        return target_epsilon * (epoch + 1) / warmup_epochs
    return target_epsilon
```

#### Step Schedule

```python
def step_epsilon_schedule(epoch, milestones, epsilons):
    """Step-wise epsilon increase at milestones.

    Example: milestones=[10, 30, 60], epsilons=[2/255, 4/255, 8/255]
    """
    current_epsilon = epsilons[0]
    for milestone, eps in zip(milestones, epsilons):
        if epoch >= milestone:
            current_epsilon = eps
    return current_epsilon
```

#### Cosine Schedule

```python
import math

def cosine_epsilon_schedule(epoch, total_epochs, target_epsilon):
    """Cosine warmup from 0 to target_epsilon."""
    progress = min(epoch, total_epochs) / total_epochs
    return target_epsilon * 0.5 * (1 - math.cos(math.pi * progress))
```

### Implementation for brain_ai

```python
class CurriculumAdversarialTrainer:
    def __init__(self, model, config):
        self.model = model
        self.target_epsilon = config.epsilon
        self.warmup_epochs = config.warmup_epochs  # e.g., 10
        self.schedule = config.epsilon_schedule     # 'linear', 'step', 'cosine'

    def get_epsilon(self, epoch):
        """Get current epsilon based on schedule."""
        if self.schedule == 'linear':
            return linear_epsilon_schedule(
                epoch, self.warmup_epochs, self.target_epsilon)
        elif self.schedule == 'cosine':
            return cosine_epsilon_schedule(
                epoch, self.warmup_epochs, self.target_epsilon)
        else:
            return self.target_epsilon

    def train_epoch(self, loader, optimizer, epoch):
        epsilon = self.get_epsilon(epoch)
        for x, y in loader:
            # Use current epsilon for this epoch
            loss = pgd_at_train_step(
                self.model, x, y, optimizer, epsilon, ...)
```

### Benefits for brain_ai

Curriculum adversarial training is particularly important for brain_ai because:

1. **SNN stability**: The SNN core's surrogate gradients can become unstable with large perturbations early in training. Gradual epsilon increase allows the SNN to develop stable dynamics first.
2. **Workspace convergence**: The global workspace attention mechanism needs to converge before adversarial perturbations challenge it. Early adversarial training can prevent workspace convergence.
3. **Multi-phase training**: Brain_ai uses a 7-phase training pipeline. Adversarial training should be introduced in phases 1-2 (encoders) with curriculum scheduling.

---

## Practical Considerations

### Avoiding Training Collapse

Adversarial training collapse symptoms:
- Loss diverges to infinity
- Accuracy drops to random chance (10% on 10 classes)
- Gradient norms explode

Prevention strategies:
1. **Reduce epsilon**: Start with epsilon=2/255, increase if stable
2. **Increase warmup**: Use curriculum with 10-20 warmup epochs
3. **Use TRADES**: KL regularization is smoother than CE on adversarial examples
4. **Lower learning rate**: AT needs lower LR than standard training (0.01 vs 0.1)
5. **Gradient clipping**: Clip gradients at norm 1.0

### Monitoring Adversarial Training

During training, track:
- **Clean accuracy** on validation set (should not collapse)
- **PGD-20 robust accuracy** (should improve over epochs)
- **Training loss** on adversarial examples (should decrease)
- **Gradient norm** (should not explode)

After training, assess with:
- **AutoAttack** for reliable robust accuracy
- Compare clean accuracy with standard-trained baseline

### Batch Normalization

Batch normalization interacts poorly with adversarial training because the batch statistics from adversarial examples differ from clean data. Solutions:

1. **Separate batch norm**: Use different BN statistics for clean and adversarial paths
2. **Use model in appropriate mode during attack**: Freeze BN statistics during PGD inner loop
3. **Replace BN with GroupNorm or LayerNorm**: These are independent of batch statistics

For brain_ai, the workspace uses LayerNorm, which avoids this issue. But vision encoders with BN should handle this carefully.

### Memory Considerations

PGD-AT doubles memory usage because both clean and adversarial examples must be in memory. For brain_ai at production scale:
- Use gradient checkpointing to reduce memory
- Use Free-AT with replays to reduce cost
- Consider adversarial training only on encoder layers (freeze downstream)

---

## Brain_ai Adversarial Training Strategy

### Recommended Approach

For brain_ai, the recommended adversarial training strategy:

1. **Phase 1-2 only**: Adversarially train encoders (vision, text, audio). Later phases (HTM, workspace, reasoning) inherit robustness from robust encoders.

2. **TRADES with curriculum**:
   - Epochs 1-10: Clean training only (establish baseline)
   - Epochs 11-20: TRADES with epsilon linearly increasing from 0 to 4/255
   - Epochs 21-50: TRADES with epsilon linearly increasing from 4/255 to 8/255
   - Epochs 51+: Full TRADES at epsilon=8/255, beta=6.0

3. **Modality-specific budgets**:
   - Vision: epsilon=8/255, L-inf
   - Text: epsilon=0.5, L2 in embedding space
   - Audio: epsilon=0.005, L-inf on waveform

4. **Assessment cadence**:
   - Every 5 epochs: PGD-20 robust accuracy
   - End of training: AutoAttack robust accuracy
   - Monitor clean accuracy throughout

### Expected Results

On MNIST with a minimal brain_ai config:
- Standard training: ~98.5% clean, ~0% robust (at 8/255)
- PGD-AT: ~97% clean, ~92% robust
- TRADES (beta=6): ~97.5% clean, ~93% robust

On CIFAR-10 with a larger config:
- Standard training: ~95% clean, ~0% robust
- PGD-AT: ~83% clean, ~50% robust
- TRADES (beta=6): ~85% clean, ~52% robust

---

## Summary

| Method | Key Feature | Best For | Cost |
|--------|------------|----------|------|
| PGD-AT | Min-max training | Maximum robustness | High (K inner steps) |
| TRADES | KL regularization | Accuracy-robustness balance | High (K inner steps) |
| Free-AT | Gradient reuse | Efficiency at scale | Medium (M replays) |
| Curriculum | Gradual epsilon increase | Training stability | Same as base method |

For brain_ai: Use TRADES with curriculum adversarial training on encoder phases. This provides the best balance of robustness, clean accuracy, and training stability for the multi-layer architecture.
