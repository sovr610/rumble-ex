# Progressive Architectures for Continual Learning

## 1. Overview

Progressive architectures prevent catastrophic forgetting through structural isolation: each task gets its own dedicated parameters, and previously learned parameters are permanently frozen. Forward transfer is enabled through lateral connections or adapters that allow new tasks to leverage representations learned by prior tasks.

Three architectures are covered:
1. **Progressive Neural Networks** (Rusu et al., 2016): add a new column per task with lateral adapters
2. **PackNet** (Mallya & Lazebnik, 2018): iteratively prune and freeze subnetworks within a single model
3. **DEN** (Yoon et al., 2018): dynamically expand the network when existing capacity is insufficient

---

## 2. Progressive Neural Networks

### Architecture

A progressive network is a sequence of "columns" (independent networks), one per task. When task t begins:

1. Instantiate a new column C_t with the same architecture as C_1
2. Freeze all parameters of columns C_1, C_2, ..., C_{t-1}
3. Add lateral connections from each frozen column's hidden layers to the new column
4. Train only C_t and the lateral adapters

```
Task 1:  [C1] -------- output_1

Task 2:  [C1] --lateral--> [C2] -------- output_2
         (frozen)

Task 3:  [C1] --lateral--> [C3] -------- output_3
         (frozen)
         [C2] --lateral--> [C3]
         (frozen)
```

### Lateral Connections

At each hidden layer l, the new column receives lateral input from all prior columns:

    h_t^l = sigma(W_t^l * h_t^{l-1} + sum_{k<t} U_{k->t}^l * h_k^{l-1})

where:
- W_t^l is the standard weight matrix for column t at layer l
- U_{k->t}^l is the lateral adapter from column k to column t at layer l
- h_k^{l-1} is the (frozen) hidden activation of column k at layer l-1
- sigma is the activation function

### Lateral Adapter Design

The simplest adapter is a linear projection:

    U_{k->t}^l : R^{d_k} -> R^{d_t}

For efficiency with many prior columns, use a bottleneck adapter:

    U_{k->t}^l = W_up @ ReLU(W_down @ h_k^{l-1})

where W_down : R^{d_k} -> R^{d_lateral} and W_up : R^{d_lateral} -> R^{d_t}, with d_lateral << d_k.

### Properties

- **Zero forgetting**: frozen columns cannot change, so prior-task performance is preserved exactly
- **Forward transfer**: lateral connections allow new tasks to reuse prior features
- **No backward transfer**: improved understanding from task 3 cannot improve task 1
- **Linear parameter growth**: O(T * |C|) where T is task count and |C| is column size
- **Inference cost**: only the relevant column (plus lateral inputs from all prior columns) is used

### Implementation Skeleton

```python
class ProgressiveColumn(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim,
                 prior_columns, lateral_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.laterals = nn.ModuleList()

        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, h_dim))

            # Lateral connections from all prior columns at this layer
            layer_laterals = nn.ModuleList()
            for col in prior_columns:
                layer_laterals.append(
                    nn.Sequential(
                        nn.Linear(col.hidden_dims[i], lateral_dim),
                        nn.ReLU(),
                        nn.Linear(lateral_dim, h_dim),
                    )
                )
            self.laterals.append(layer_laterals)
            prev_dim = h_dim

        self.head = nn.Linear(prev_dim, output_dim)

    def forward(self, x, prior_hiddens):
        """Forward pass with lateral inputs.

        Args:
            x: Input tensor
            prior_hiddens: List of lists; prior_hiddens[col_idx][layer_idx]
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            # Add lateral contributions
            for col_idx, lateral in enumerate(self.laterals[i]):
                h = h + lateral(prior_hiddens[col_idx][i])
            h = F.relu(h)
        return self.head(h)
```

### Column Freezing

When a task is completed, freeze its column:

```python
def freeze_column(column):
    """Permanently freeze all parameters in a column."""
    for param in column.parameters():
        param.requires_grad = False
    column.train(False)  # Disable dropout, freeze BN
```

**Verification**: after freezing, run a backward pass and verify all frozen parameter gradients are exactly zero (or None).

---

## 3. PackNet (Iterative Pruning + Freezing)

### Concept

PackNet uses a single shared network but allocates separate subnetworks (via binary masks) for each task:

1. Train on task t using all available (non-frozen) weights
2. Prune: identify the most important weights (by magnitude) and create a binary mask
3. Freeze: lock the masked weights for task t; they cannot be modified by future tasks
4. The remaining (pruned) weights are available for task t+1

### Algorithm

```
For task t:
    1. Train model on task t (using only weights not frozen by tasks 1..t-1)
    2. Rank all trainable weights by absolute magnitude
    3. Keep top (1 - prune_ratio) fraction; set mask_t = 1 for these weights
    4. Freeze mask_t weights (set requires_grad = False)
    5. Re-initialize pruned weights to small random values for next task
```

### Mask Management

Each task owns a binary mask over the full parameter space:

```python
masks = {}  # task_id -> {param_name: binary_tensor}

def apply_mask(model, task_id):
    """Zero out parameters not in this task's mask."""
    for name, param in model.named_parameters():
        param.data *= masks[task_id][name]

def get_free_mask(model):
    """Return mask of weights not claimed by any task."""
    free = {n: torch.ones_like(p) for n, p in model.named_parameters()}
    for task_id in masks:
        for name in free:
            free[name] *= (1 - masks[task_id][name])
    return free
```

**Critical invariant**: masks for different tasks must be disjoint:

    mask_t AND mask_s == 0 for all t != s

### Capacity Planning

With prune_ratio = 0.75, each task uses 25% of the available weights:
- Task 1: 25% of N parameters
- Task 2: 25% of remaining 75% = 18.75% of N
- Task 3: 25% of remaining 56.25% = 14.06% of N
- ...

Total capacity for T tasks: N * (1 - (1 - prune_ratio)^T)

After ~10 tasks with prune_ratio=0.75, over 94% of weights are allocated.

### Trade-offs vs Progressive Networks

| Aspect | Progressive Networks | PackNet |
|---|---|---|
| Parameter growth | O(T * |model|) | O(|model|) fixed |
| Forgetting | Zero (by design) | Zero (frozen masks) |
| Forward transfer | Via lateral connections | Via shared early layers |
| Capacity limit | Memory (can always add columns) | Fixed (weights run out) |
| Inference cost | One column + laterals | Full model (one mask) |
| Implementation complexity | Moderate | Lower |

---

## 4. DEN (Dynamically Expandable Networks)

### Concept

DEN (Yoon et al., 2018) combines selective retraining with network expansion:

1. **Selective retraining**: for a new task, identify which neurons are relevant and retrain only those
2. **Network expansion**: if selective retraining is insufficient (loss too high), add new neurons
3. **Network split**: if a neuron is pulled in conflicting directions by old and new tasks, duplicate it

### Algorithm Phases

```
Phase 1: Selective Retraining
    - Use L1 regularization to identify sparse set of important neurons
    - Fine-tune only these neurons on the new task
    - If validation loss < threshold: done

Phase 2: Network Expansion
    - Add new neurons to each layer
    - Train expanded network on new task with group sparsity
    - Remove neurons that were not useful (zero weight)

Phase 3: Network Split
    - For each neuron trained in Phase 1, check if it drifted too far
    - If drift > threshold: duplicate the neuron (old copy frozen, new copy trainable)
```

### When to Use DEN

DEN is more complex than Progressive Networks or PackNet but is appropriate when:
- Task count is very large (> 20)
- Model capacity must be managed dynamically
- Some backward transfer is desired (shared neurons benefit all tasks)

For the brain_ai architecture, Progressive Networks or PackNet are recommended for the initial implementation, with DEN as a future extension.

---

## 5. Lateral Adapter Variants

### Simple Linear

    h_lateral = W @ h_prior

Cheapest option. Works when prior and new feature dimensions are similar.

### Bottleneck

    h_lateral = W_up @ ReLU(W_down @ h_prior)

Reduces parameter count when d_lateral << d_prior. Recommended default.

### Attention-Based

    alpha = softmax(Q @ K^T / sqrt(d))
    h_lateral = alpha @ V

Where Q comes from the new column and K, V come from prior columns. Allows selective attention to relevant prior features. Most expensive but best for heterogeneous tasks.

### Gated

    gate = sigmoid(W_gate @ [h_new; h_prior])
    h_lateral = gate * (W @ h_prior)

Allows the new column to learn which prior features to incorporate. Good balance of expressiveness and cost.

---

## 6. References

- Rusu, A. A., et al. (2016). "Progressive Neural Networks." arXiv:1606.04671.
- Mallya, A. & Lazebnik, S. (2018). "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning." CVPR 2018.
- Yoon, J., et al. (2018). "Lifelong Learning with Dynamically Expandable Networks." ICLR 2018.
- Aljundi, R., et al. (2017). "Expert Gate: Lifelong Learning with a Network of Experts." CVPR 2017.
- Serra, J., et al. (2018). "Overcoming Catastrophic Forgetting with Hard Attention to the Task." ICML 2018.
