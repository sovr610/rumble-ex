# Elastic Weight Consolidation (EWC): Theory and Implementation

## 1. Overview

Elastic Weight Consolidation (EWC) prevents catastrophic forgetting by adding a quadratic penalty that anchors important parameters near their optimal values for previously learned tasks. The importance of each parameter is estimated by the diagonal of the Fisher information matrix, computed over data from the completed task.

Original paper: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (PNAS, 2017).

---

## 2. Fisher Information Matrix

### Definition

For a neural network with parameters theta and a dataset D = {(x_n, y_n)}, the Fisher information matrix is:

    F = E_{(x,y)~D} [ nabla_theta log p(y|x,theta) * nabla_theta log p(y|x,theta)^T ]

This is the expected outer product of the gradient of the log-likelihood. For classification with softmax output, the log-likelihood is the log of the predicted probability for the true class.

### Diagonal Approximation

The full Fisher matrix is |theta| x |theta|, which is O(N^2) in parameter count -- infeasible for modern networks. EWC uses the diagonal approximation:

    F_ii = E_{(x,y)~D} [ (d/d theta_i log p(y|x,theta))^2 ]

This reduces memory to O(N) and can be computed in a single pass over the dataset.

### Empirical Estimation

In practice, the diagonal Fisher is estimated empirically over a subset of task data:

```
F_diag = zeros_like(theta)
model.train(False)  # CRITICAL: disable dropout and use running BN stats
for (x, y) in fisher_data_loader:
    log_probs = log_softmax(model(x))
    for each sample in batch:
        # Sample from model's own distribution (true Fisher)
        # or use the true label (empirical Fisher -- more common in practice)
        loss = -log_probs[sample_idx, y[sample_idx]]
        loss.backward()
        for name, param in model.named_parameters():
            F_diag[name] += param.grad.data ** 2
        model.zero_grad()
F_diag /= num_samples
```

**Critical**: Fisher MUST be computed with `model.train(False)`. Dropout randomness and batch normalization training statistics introduce noise that corrupts the Fisher estimate. This is the single most common implementation bug in EWC.

### Empirical vs True Fisher

- **True Fisher**: sample y from the model's predictive distribution p(y|x,theta). Mathematically correct but computationally expensive (requires sampling and re-evaluating).
- **Empirical Fisher**: use the ground-truth label y from the dataset. A biased but practical approximation that works well when the model is well-trained on the task. This is what most implementations use.

The empirical Fisher is equivalent to the true Fisher when the model perfectly predicts the data distribution.

---

## 3. EWC Penalty

### Formulation

After training on task A and obtaining optimal parameters theta*_A, the EWC penalty added during training on task B is:

    L_ewc = (lambda / 2) * sum_i F_A,i * (theta_i - theta*_A,i)^2

where:
- lambda is the regularization strength (hyperparameter)
- F_A,i is the diagonal Fisher for parameter i, computed on task A data
- theta*_A,i is the optimal value of parameter i after task A training
- theta_i is the current value of parameter i during task B training

The total loss for task B becomes:

    L_total = L_B(theta) + L_ewc(theta)

### Intuition

Parameters with high Fisher information are "important" for task A -- small changes cause large increases in loss. The quadratic penalty makes these parameters expensive to move, effectively anchoring them near theta*_A. Parameters with low Fisher information are "unimportant" and can be freely modified for task B.

### Lambda Tuning

Lambda controls the trade-off between plasticity (learning new tasks) and stability (retaining old tasks):

- **Lambda too high**: the model cannot learn new tasks (over-regularized)
- **Lambda too low**: the model forgets old tasks (under-regularized)
- **Typical range**: 100 to 10,000 depending on model size and task similarity
- **Heuristic**: start with lambda = 1000 and adjust based on forgetting metrics

Normalization helps: divide F_diag by its maximum value so the penalty scale is independent of Fisher magnitude. Then lambda operates on a normalized scale.

---

## 4. Multi-Task Extension

### Naive Multi-Task EWC

For a sequence of tasks A, B, C, ..., the naive extension stores Fisher diagonals and optimal parameters for every past task:

    L_ewc = (lambda / 2) * sum_t sum_i F_t,i * (theta_i - theta*_t,i)^2

Memory cost: O(T * |theta|) where T is the number of tasks. This grows linearly and becomes impractical for long task sequences.

### Online EWC (Schwarz et al., 2018)

Online EWC addresses the linear memory growth by maintaining a single running Fisher diagonal via exponential moving average:

    F_online = gamma * F_online + F_new

where gamma in (0, 1) is the EMA decay rate (typically 0.9 to 0.99). The optimal parameters are updated to the most recent task's optimal values:

    theta*_online = theta*_new

The penalty becomes:

    L_online_ewc = (lambda / 2) * sum_i F_online,i * (theta_i - theta*_online,i)^2

**Advantages of Online EWC**:
- Memory: O(|theta|) regardless of task count (constant)
- Naturally down-weights very old tasks (exponential decay)
- Simpler implementation (no per-task storage)

**Gamma tuning**:
- gamma close to 1.0: long memory, retains all past tasks strongly
- gamma close to 0.0: short memory, recent tasks dominate
- Typical: gamma = 0.95

---

## 5. Implementation Details

### Fisher Computation Pseudocode

```python
def compute_fisher_diagonal(model, data_loader, num_samples):
    """Compute diagonal Fisher information matrix.

    Args:
        model: Neural network (will be set to train(False) internally)
        data_loader: DataLoader for task data
        num_samples: Number of samples to use for estimation

    Returns:
        Dict[str, Tensor] mapping parameter names to Fisher diagonal values
    """
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    was_training = model.training
    model.train(False)  # CRITICAL: disable dropout, use running BN stats

    count = 0
    for x, y in data_loader:
        if count >= num_samples:
            break

        logits = model(x)
        log_probs = F.log_softmax(logits, dim=-1)

        for i in range(x.size(0)):
            if count >= num_samples:
                break

            model.zero_grad()
            nll = -log_probs[i, y[i]]
            nll.backward(retain_graph=(i < x.size(0) - 1))

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.clone() ** 2

            count += 1

    # Normalize by sample count
    for name in fisher:
        fisher[name] /= count

    model.train(was_training)
    return fisher
```

### EWC Penalty Computation

```python
def ewc_penalty(current_params, star_params, fisher_diag):
    """Compute EWC quadratic penalty.

    Args:
        current_params: Dict of current parameter values
        star_params: Dict of optimal parameter values from previous task
        fisher_diag: Dict of Fisher diagonal values

    Returns:
        Scalar penalty (non-negative by construction)
    """
    penalty = 0.0
    for name in fisher_diag:
        diff = current_params[name] - star_params[name]
        penalty += (fisher_diag[name] * diff ** 2).sum()
    return penalty
```

### Online EWC Update

```python
def update_online_fisher(fisher_online, fisher_new, gamma):
    """Update running Fisher diagonal with EMA.

    Args:
        fisher_online: Current running Fisher (modified in-place)
        fisher_new: Fisher diagonal from the new task
        gamma: EMA decay rate
    """
    for name in fisher_online:
        fisher_online[name] = gamma * fisher_online[name] + fisher_new[name]
```

---

## 6. Relationship to Other Methods

| Method | Importance Source | When Computed | Memory | Online |
|---|---|---|---|---|
| EWC | Fisher diagonal | After task training | O(T*N) or O(N) online | Yes (online variant) |
| SI | Path integral | During task training | O(N) | Yes (by design) |
| MAS | Gradient magnitude | After task training | O(N) | Yes (EMA variant) |

EWC and SI are complementary: EWC estimates importance from the curvature of the loss surface at the optimum, while SI estimates importance from the trajectory taken during optimization. In practice, SI is simpler to implement (no separate Fisher computation pass) but may be less accurate for importance estimation.

---

## 7. Known Limitations

1. **Diagonal approximation**: ignores parameter correlations. Two parameters that are individually unimportant but jointly critical will not be protected.

2. **Point estimate**: theta*_A is a single point. The true posterior over parameters is a distribution, and EWC approximates it with a Gaussian centered at theta*_A with precision F_A.

3. **Task-recency bias**: even with online EWC, recent tasks dominate. Very old tasks may still be forgotten if gamma is not close enough to 1.0.

4. **Shared representations**: if all tasks use the same features (high Fisher everywhere), EWC over-constrains the model. In this regime, PackNet or Progressive Networks may be more appropriate.

5. **Hyperparameter sensitivity**: lambda and gamma both require tuning per domain. There is no universally good default, though lambda=1000 and gamma=0.95 are reasonable starting points.

---

## 8. References

- Kirkpatrick, J., et al. (2017). "Overcoming catastrophic forgetting in neural networks." PNAS, 114(13), 3521-3526.
- Schwarz, J., et al. (2018). "Progress & Compress: A scalable framework for continual learning." ICML 2018.
- Huszar, F. (2018). "Note on the quadratic penalties in elastic weight consolidation." PNAS, 115(11), E2496-E2497.
- Ritter, H., Botev, A., & Barber, D. (2018). "Online Structured Laplace Approximations for Overcoming Catastrophic Forgetting." NeurIPS 2018.
