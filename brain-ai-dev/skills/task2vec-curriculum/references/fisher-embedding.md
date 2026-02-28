# Fisher-Information Task Embeddings: Probe, Computation, and Normalization

## 1. Overview

A Task2Vec embedding is a fixed-dimensional vector summarizing the computational demands a task places on a reference neural network (the probe). The embedding is derived from the diagonal of the Fisher information matrix computed on the probe's parameters using the task's support set. Two tasks that require similar parameter sensitivities produce similar embeddings; tasks that stress different parts of the probe produce distant embeddings.

The pipeline has three stages:

1. **Probe forward pass** -- run the frozen probe with an ephemeral classification head on the task's support set.
2. **Diagonal Fisher accumulation** -- compute per-sample squared gradients of the log-likelihood and average them.
3. **Projection and normalization** -- select a parameter subset, aggregate per group, apply log1p transform, and L2-normalize.

The output is an E-dimensional vector (E in {256, 512, 1024}) that is L2-normalized, deterministic, and comparable across tasks extracted with the same probe.

Three non-negotiable properties:

1. **Determinism**: Given identical `(seed, dataset split, class_ids, support_indices, transforms)`, the embedding is identical within 1e-7 on the same device and has cosine similarity > 0.9999 cross-device.
2. **Probe isolation**: The probe is never modified during extraction. It is in eval mode with frozen weights, no dropout, and fixed batch normalization statistics.
3. **Numerical stability**: All Fisher computation runs in fp32 regardless of any ambient AMP autocast context.

---

## 2. Probe Network Selection

The probe is a pretrained neural network used as a fixed reference for measuring task similarity. It is NOT the meta-learner and it is NOT trained during embedding extraction. The probe provides a coordinate system in which tasks can be compared: the Fisher information over the probe's parameters captures which weights the task's data would want to change, without actually changing them.

### Supported Probe Backbones

| Backbone | Architecture | Param Count | Input Size | Best For |
|---|---|---|---|---|
| `conv4` | 4-block conv (64-64-64-64), 3x3 kernels, BN, ReLU, 2x2 maxpool | ~112K | 84x84 or 28x28 | Few-shot benchmarks (mini-ImageNet, Omniglot), fast extraction |
| `resnet12` | 4-stage ResNet (64-128-256-512), basic blocks, BN, LeakyReLU | ~8M | 84x84 | Standard few-shot, higher-capacity probes |
| `vit_tiny` | ViT with 12 heads, embed_dim=192, 12 layers, patch_size=16 | ~5.7M | 224x224 | Large-image tasks, attention-based aggregation |

### Probe Requirements

Set the probe to eval mode and freeze all parameters before any extraction:

```python
import torch
import torch.nn as nn


def prepare_probe(probe: nn.Module) -> nn.Module:
    """Prepare probe for deterministic embedding extraction.

    Sets eval mode (disables dropout, fixes BN running stats),
    freezes all parameters (disables gradient accumulation on probe weights),
    and verifies the configuration.
    """
    probe.eval()

    for param in probe.parameters():
        param.requires_grad_(False)

    # Verify: no parameter should require grad
    for name, param in probe.named_parameters():
        assert not param.requires_grad, (
            f"Probe parameter '{name}' still requires grad after freezing"
        )

    # Verify: all BN layers are in eval mode (use running stats, not batch stats)
    for name, module in probe.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            assert not module.training, (
                f"BatchNorm '{name}' is in training mode after probe.eval()"
            )

    return probe
```

The probe must be pretrained on a representative dataset (ImageNet-1k for vision probes). Pretraining quality affects embedding quality -- a randomly initialized probe produces near-uniform Fisher values across all tasks, collapsing the embedding space.

### Ephemeral Classification Head

Each task episode has its own N-way classification. The probe backbone produces a feature vector; an ephemeral linear head maps features to N classes. This head is created fresh for each task and is the only part of the probe-plus-head system that has `requires_grad=True`.

```python
def create_ephemeral_head(
    feature_dim: int,
    n_way: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> nn.Linear:
    """Create a fresh N-way classification head for one episode.

    The head is initialized with Kaiming uniform (default nn.Linear init).
    It has requires_grad=True on both weight and bias.
    It is NOT part of the probe -- it is created and discarded per task.

    Args:
        feature_dim: Output dimension of the probe backbone.
        n_way: Number of classes in this episode.
        device: Target device.
        dtype: Parameter dtype (must be float32 for Fisher computation).

    Returns:
        A fresh nn.Linear(feature_dim, n_way) on the specified device.
    """
    head = nn.Linear(feature_dim, n_way, bias=True)
    head = head.to(device=device, dtype=dtype)
    return head
```

**Critical distinction**: The Fisher information is computed over the probe backbone's parameters, NOT the ephemeral head's parameters. The head exists solely to define a valid loss function (cross-entropy over N classes). The head's gradients are discarded after Fisher computation.

### Probe Signature

Every embedding records which probe produced it. Embeddings from different probes are not comparable.

```python
import hashlib
import json


def compute_probe_signature(
    model_name: str,
    layer_subset: str,
    preprocessing: dict,
) -> str:
    """Compute a deterministic hash identifying the probe configuration.

    Args:
        model_name: Probe backbone name ("conv4", "resnet12", "vit_tiny").
        layer_subset: Parameter subset strategy ("last_block", "per_stage", "all").
        preprocessing: Dict of preprocessing settings (resize, normalize mean/std, etc.).

    Returns:
        Hex digest string uniquely identifying this probe configuration.
    """
    payload = json.dumps({
        "model_name": model_name,
        "layer_subset": layer_subset,
        "preprocessing": preprocessing,
    }, sort_keys=True)

    return hashlib.sha256(payload.encode()).hexdigest()[:16]
```

The signature is stored alongside every embedding in the registry. Before comparing two embeddings, verify their `probe_signature` values match.

---

## 3. Diagonal Fisher Information Computation

The Fisher information matrix F measures the sensitivity of the log-likelihood to changes in model parameters. For a parameter vector theta of dimension D, the full Fisher matrix is DxD -- prohibitively large (tens of millions squared). The diagonal approximation retains only the diagonal entries, reducing storage to D scalars.

### Mathematical Definition

For a single sample (x, y) and model parameters theta:

```
g = d/d(theta) log p(y | x, theta)      # score vector (gradient of log-likelihood)
F_diag = E[(g . g)]                       # diagonal of Fisher = E[element-wise squared gradient]
```

In practice, approximate the expectation with the empirical mean over the support set:

```
F_diag = (1/N) * sum_{i=1}^{N} g_i . g_i
```

where `g_i` is the score vector for sample i and `.` denotes element-wise multiplication (Hadamard product).

### Implementation: Per-Sample Gradient Accumulation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_diagonal_fisher(
    probe: nn.Module,
    head: nn.Linear,
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    param_names: list[str],
) -> dict[str, torch.Tensor]:
    """Compute diagonal Fisher information over probe parameters.

    The probe must be in eval mode with all parameters frozen
    (requires_grad=False). Gradients are computed with respect to
    the probe's parameters by temporarily enabling requires_grad
    on the selected parameter subset, computing per-sample gradients,
    and then re-freezing.

    Args:
        probe: Frozen pretrained probe backbone.
        head: Ephemeral N-way classification head (requires_grad=True).
        x_support: Support set inputs, shape (N_samples, C, H, W).
        y_support: Support set labels, shape (N_samples,).
        param_names: List of probe parameter names to include in Fisher.

    Returns:
        Dict mapping parameter name to diagonal Fisher tensor (same shape as param).
    """
    device = x_support.device
    n_samples = x_support.shape[0]

    # Collect the selected parameters
    selected_params = {}
    for name, param in probe.named_parameters():
        if name in param_names:
            selected_params[name] = param

    # Initialize Fisher accumulators
    fisher_diag = {name: torch.zeros_like(param, dtype=torch.float32, device=device)
                   for name, param in selected_params.items()}

    # Temporarily enable gradients on selected probe parameters
    for param in selected_params.values():
        param.requires_grad_(True)

    try:
        # Per-sample gradient accumulation
        for i in range(n_samples):
            xi = x_support[i:i+1]   # Keep batch dimension: (1, C, H, W)
            yi = y_support[i:i+1]   # (1,)

            # Forward pass through frozen probe + ephemeral head
            features = probe(xi)                        # (1, feature_dim)
            logits = head(features)                     # (1, n_way)
            log_prob = F.log_softmax(logits, dim=-1)    # (1, n_way)
            nll = -log_prob[0, yi[0]]                   # scalar: -log p(y|x,theta)

            # Compute gradient of log-likelihood w.r.t. selected probe params
            grads = torch.autograd.grad(
                outputs=nll,
                inputs=list(selected_params.values()),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            # Accumulate squared gradients (diagonal Fisher)
            for (name, _), g in zip(selected_params.items(), grads):
                if g is not None:
                    fisher_diag[name] += (g.float() ** 2)

        # Normalize by number of samples
        for name in fisher_diag:
            fisher_diag[name] /= n_samples

    finally:
        # Re-freeze probe parameters
        for param in selected_params.values():
            param.requires_grad_(False)

    return fisher_diag
```

### Batched Fisher Computation (Efficient)

Processing one sample at a time is straightforward but slow. For larger support sets, use `torch.func.vmap` and `torch.func.grad` to vectorize the per-sample gradient computation:

```python
from torch.func import functional_call, grad, vmap


def compute_diagonal_fisher_batched(
    probe: nn.Module,
    head: nn.Linear,
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    param_names: list[str],
) -> dict[str, torch.Tensor]:
    """Batched diagonal Fisher using torch.func.vmap.

    Computes per-sample gradients in parallel via vectorization.
    Requires PyTorch >= 2.0 with torch.func support.

    Performance: 3-10x faster than sequential for support sets > 10 samples.
    Memory: O(N_samples * param_count) -- may require chunking for large probes.

    Args:
        probe: Frozen pretrained probe backbone.
        head: Ephemeral N-way classification head.
        x_support: Support set inputs, shape (N_samples, C, H, W).
        y_support: Support set labels, shape (N_samples,).
        param_names: List of probe parameter names to include.

    Returns:
        Dict mapping parameter name to diagonal Fisher tensor.
    """
    device = x_support.device
    n_samples = x_support.shape[0]

    # Build combined parameter dict (probe selected params only)
    selected_params = {}
    for name, param in probe.named_parameters():
        if name in param_names:
            selected_params[name] = param.detach().float().requires_grad_(True)

    # Full parameter dict for functional_call (probe + head)
    all_probe_params = {name: param for name, param in probe.named_parameters()}
    head_params = {f"head.{name}": param for name, param in head.named_parameters()}

    def compute_single_nll(params_to_diff, x_single, y_single):
        """NLL for one sample, differentiable w.r.t. params_to_diff."""
        # Merge selected params back into full probe params
        merged = {**all_probe_params, **params_to_diff}
        features = functional_call(probe, merged, (x_single.unsqueeze(0),))
        logits = functional_call(head, head_params, (features,))
        log_prob = F.log_softmax(logits, dim=-1)
        return -log_prob[0, y_single]

    # Compute per-sample gradients via vmap
    per_sample_grad_fn = vmap(
        grad(compute_single_nll),
        in_dims=(None, 0, 0),  # params shared, data batched
    )

    per_sample_grads = per_sample_grad_fn(selected_params, x_support, y_support)

    # Accumulate diagonal Fisher: mean of squared gradients across samples
    fisher_diag = {}
    for name in selected_params:
        g = per_sample_grads[name].float()    # (N_samples, *param_shape)
        fisher_diag[name] = (g ** 2).mean(dim=0)   # (*param_shape)

    return fisher_diag
```

### fp32 Enforcement

Fisher computation must run in fp32 even when the outer training loop uses AMP. Squared gradients in fp16 suffer from catastrophic overflow (large gradients squared exceed fp16 max ~65504) and underflow (small gradients squared round to zero, destroying information).

```python
def compute_fisher_safe(probe, head, x_support, y_support, param_names):
    """AMP-safe Fisher computation wrapper."""
    with torch.amp.autocast('cuda', enabled=False):
        # Cast inputs to fp32
        x_fp32 = x_support.float()
        y_long = y_support.long()

        # Ensure probe and head are in fp32
        probe_fp32 = probe.float()
        head_fp32 = head.float()

        return compute_diagonal_fisher(
            probe_fp32, head_fp32, x_fp32, y_long, param_names
        )
```

### Gradient Computation: autograd.grad vs torch.func.grad

| Method | API | Use Case | Pros | Cons |
|---|---|---|---|---|
| `torch.autograd.grad` | `grad(loss, params, create_graph=False)` | Per-sample loop | Simple, works on all PyTorch versions | Sequential, slow for large support sets |
| `torch.func.grad` + `vmap` | `vmap(grad(fn))(params, data)` | Batched per-sample | Vectorized, 3-10x faster | Requires PyTorch >= 2.0, higher peak memory |
| `torch.func.grad` only | `grad(fn)(params)` | Single-sample or batched loss | Clean functional style | No per-sample parallelism without vmap |

Prefer `torch.func.grad` + `vmap` when available. Fall back to the `torch.autograd.grad` loop on older PyTorch versions:

```python
_HAS_TORCH_FUNC = False
try:
    from torch.func import functional_call, grad, vmap
    _HAS_TORCH_FUNC = True
except ImportError:
    pass


def compute_diagonal_fisher_auto(probe, head, x_support, y_support, param_names):
    """Select the best available Fisher computation method."""
    if _HAS_TORCH_FUNC:
        return compute_diagonal_fisher_batched(
            probe, head, x_support, y_support, param_names
        )
    return compute_diagonal_fisher(
        probe, head, x_support, y_support, param_names
    )
```

---

## 4. Parameter Subset Selection

The raw diagonal Fisher vector has as many entries as the probe has parameters -- 112K for Conv4, 8M for ResNet-12, 5.7M for ViT-tiny. This is too large for a practical embedding. Parameter subset selection restricts the Fisher computation to a meaningful subset of layers, reducing dimensionality and focusing on the most informative parameters.

### Subset Strategies

| Strategy | Selected Parameters | Typical Parameter Count | Use Case |
|---|---|---|---|
| `"last_block"` | Last conv/transformer block + classifier head | Conv4: ~37K, ResNet-12: ~1.3M, ViT: ~740K | Default; fast, captures task-discriminative features |
| `"per_stage"` | One representative layer per network stage/block | Conv4: ~16K, ResNet-12: ~170K, ViT: ~230K | Balanced coverage across depth |
| `"all"` | All probe parameters | Full model size | Analysis only; too large for practical embeddings |

### Layer Selection Implementation

Layer selection must be deterministic -- the same probe model always selects the same layers in the same order.

```python
def select_parameter_subset(
    probe: nn.Module,
    model_name: str,
    layer_subset: str,
) -> list[str]:
    """Select probe parameter names for Fisher computation.

    Returns a deterministic, sorted list of parameter names.
    The selection is based on model architecture and subset strategy.

    Args:
        probe: The probe model (used for named_parameters enumeration).
        model_name: One of "conv4", "resnet12", "vit_tiny".
        layer_subset: One of "last_block", "per_stage", "all".

    Returns:
        Sorted list of parameter name strings.
    """
    all_names = [name for name, _ in probe.named_parameters()]

    if layer_subset == "all":
        return sorted(all_names)

    if model_name == "conv4":
        return _select_conv4_subset(all_names, layer_subset)
    elif model_name == "resnet12":
        return _select_resnet12_subset(all_names, layer_subset)
    elif model_name == "vit_tiny":
        return _select_vit_tiny_subset(all_names, layer_subset)
    else:
        raise ValueError(f"Unknown probe model: {model_name}")


def _select_conv4_subset(all_names: list[str], strategy: str) -> list[str]:
    """Conv4 has 4 blocks: block0, block1, block2, block3.
    Each block has: conv.weight, bn.weight, bn.bias.
    """
    if strategy == "last_block":
        # Last conv block (block3) parameters
        selected = [n for n in all_names
                    if n.startswith("block3.") or n.startswith("block.3.")]
        if not selected:
            selected = [n for n in all_names if "3" in n.split(".")[0]]
        return sorted(selected)

    elif strategy == "per_stage":
        # One conv layer per block: select conv.weight from each block
        selected = [n for n in all_names if "conv" in n and "weight" in n]
        return sorted(selected)

    raise ValueError(f"Unknown layer_subset: {strategy}")


def _select_resnet12_subset(all_names: list[str], strategy: str) -> list[str]:
    """ResNet-12 has 4 stages: layer1, layer2, layer3, layer4.
    Each stage has basic blocks with conv1, bn1, conv2, bn2, (optional shortcut).
    """
    if strategy == "last_block":
        # Last residual stage (layer4)
        selected = [n for n in all_names if n.startswith("layer4.")]
        return sorted(selected)

    elif strategy == "per_stage":
        # Last conv weight from each stage
        selected = []
        for stage in ["layer1", "layer2", "layer3", "layer4"]:
            stage_convs = [n for n in all_names
                           if n.startswith(f"{stage}.") and "conv" in n
                           and "weight" in n]
            if stage_convs:
                selected.append(sorted(stage_convs)[-1])  # Last conv in stage
        return sorted(selected)

    raise ValueError(f"Unknown layer_subset: {strategy}")


def _select_vit_tiny_subset(all_names: list[str], strategy: str) -> list[str]:
    """ViT-tiny has 12 transformer blocks: blocks.0 through blocks.11.
    Each block has: attn.qkv.weight, attn.proj.weight,
    mlp.fc1.weight, mlp.fc2.weight, etc.
    """
    if strategy == "last_block":
        # Last transformer block (blocks.11)
        selected = [n for n in all_names if n.startswith("blocks.11.")]
        return sorted(selected)

    elif strategy == "per_stage":
        # Divide 12 blocks into 4 stages of 3 blocks each
        # Select attention projection weight from last block in each stage
        stage_indices = [2, 5, 8, 11]
        selected = []
        for idx in stage_indices:
            candidates = [n for n in all_names
                          if n.startswith(f"blocks.{idx}.") and "weight" in n]
            if candidates:
                selected.append(sorted(candidates)[0])  # First weight in block
        return sorted(selected)

    raise ValueError(f"Unknown layer_subset: {strategy}")
```

### Layer Selection Determinism

The returned list must be sorted lexicographically. Any two calls with the same `(model_name, layer_subset)` must return identical lists. The list order determines the order of Fisher values in the final embedding vector. Changing the order changes the embedding.

Record the selected parameter names as part of the probe signature (via the `layer_subset` field in `compute_probe_signature`). If the probe architecture changes (layers renamed, blocks added), the probe signature changes, invalidating all previously computed embeddings.

---

## 5. Aggregation Strategies

After computing the diagonal Fisher over the selected parameter subset, aggregate the per-element Fisher values into groups to produce the target E-dimensional embedding. Raw element-wise Fisher vectors are too large and too sensitive to individual weight positions.

### Aggregation Methods

| Strategy | Grouping | Output per Group | Best For |
|---|---|---|---|
| `"per_channel"` | Conv: group by output channel. Linear: group by output neuron. | Mean Fisher within each output channel/neuron. | Conv backbones (Conv4, ResNet-12). Default. |
| `"per_head"` | Group by attention head (ViT only). Each head's Q, K, V, and projection weights form one group. | Mean Fisher within each head. | ViT probes with multi-head attention. |
| `"per_layer"` | One group per selected layer. All parameters in a layer are summarized by a single scalar. | Mean Fisher across all elements in the layer. | Coarse analysis, very low-dimensional embeddings. |

### Implementation

```python
import torch


def aggregate_fisher(
    fisher_diag: dict[str, torch.Tensor],
    probe: nn.Module,
    model_name: str,
    aggregation: str,
    target_dim: int,
) -> torch.Tensor:
    """Aggregate raw diagonal Fisher into E-dimensional embedding vector.

    Args:
        fisher_diag: Dict mapping param name to Fisher diagonal
                     (same shape as param).
        probe: Probe model (used to inspect layer shapes).
        model_name: Probe backbone name.
        aggregation: One of "per_channel", "per_head", "per_layer".
        target_dim: Target embedding dimension E.

    Returns:
        Tensor of shape (E,) -- unnormalized aggregated Fisher vector.
    """
    groups = []

    if aggregation == "per_channel":
        groups = _aggregate_per_channel(fisher_diag)
    elif aggregation == "per_head":
        groups = _aggregate_per_head(fisher_diag, probe, model_name)
    elif aggregation == "per_layer":
        groups = _aggregate_per_layer(fisher_diag)
    else:
        raise ValueError(f"Unknown aggregation strategy: {aggregation}")

    # Concatenate all group summaries
    raw_embedding = torch.cat(groups, dim=0)  # (D_raw,)

    # Resize to target dimension via adaptive average pooling
    if raw_embedding.shape[0] != target_dim:
        raw_embedding = _resize_embedding(raw_embedding, target_dim)

    return raw_embedding


def _aggregate_per_channel(
    fisher_diag: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    """Group by output channel (dim 0 for conv/linear weights)."""
    groups = []
    for name, fisher in sorted(fisher_diag.items()):
        if fisher.ndim >= 2:
            # Conv weight: (C_out, C_in, H, W) -> mean over (C_in, H, W) -> (C_out,)
            # Linear weight: (out, in) -> mean over (in,) -> (out,)
            channel_mean = fisher.flatten(start_dim=1).mean(dim=1)
            groups.append(channel_mean)
        elif fisher.ndim == 1:
            # Bias: (C,) -> keep as-is
            groups.append(fisher)
        else:
            # Scalar parameter
            groups.append(fisher.unsqueeze(0))
    return groups


def _aggregate_per_head(
    fisher_diag: dict[str, torch.Tensor],
    probe: nn.Module,
    model_name: str,
) -> list[torch.Tensor]:
    """Group by attention head for ViT models.

    Each attention layer's QKV weight has shape (3*embed_dim, embed_dim).
    Split into num_heads groups of (3*head_dim, embed_dim),
    then mean per group.
    """
    num_heads = 12  # ViT-tiny default
    groups = []

    for name, fisher in sorted(fisher_diag.items()):
        if "attn" in name and "qkv" in name and "weight" in name:
            # QKV weight: (3*embed_dim, embed_dim)
            # Reshape to (num_heads, 3*head_dim, embed_dim),
            # mean over last two dims
            three_embed = fisher.shape[0]
            head_dim = three_embed // (3 * num_heads)
            reshaped = fisher.reshape(num_heads, 3 * head_dim, -1)
            per_head = reshaped.flatten(start_dim=1).mean(dim=1)  # (num_heads,)
            groups.append(per_head)
        elif "attn" in name and "proj" in name and "weight" in name:
            # Projection weight: (embed_dim, embed_dim)
            # Reshape to (num_heads, head_dim, embed_dim), mean per head
            embed_dim = fisher.shape[0]
            head_dim = embed_dim // num_heads
            reshaped = fisher.reshape(num_heads, head_dim, -1)
            per_head = reshaped.flatten(start_dim=1).mean(dim=1)  # (num_heads,)
            groups.append(per_head)
        else:
            # Non-attention parameters: use per_channel aggregation
            if fisher.ndim >= 2:
                channel_mean = fisher.flatten(start_dim=1).mean(dim=1)
                groups.append(channel_mean)
            elif fisher.ndim == 1:
                groups.append(fisher)
            else:
                groups.append(fisher.unsqueeze(0))

    return groups


def _aggregate_per_layer(
    fisher_diag: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    """One scalar summary per layer: mean of all Fisher values in that layer."""
    groups = []
    for name, fisher in sorted(fisher_diag.items()):
        layer_mean = fisher.float().mean()
        groups.append(layer_mean.unsqueeze(0))  # (1,)
    return groups


def _resize_embedding(raw: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Resize raw embedding to target dimension via adaptive average pooling.

    Uses 1D adaptive average pooling to downsample or upsample the
    concatenated group summaries to exactly target_dim elements.
    """
    # AdaptiveAvgPool1d expects (batch, channels, length)
    x = raw.unsqueeze(0).unsqueeze(0)  # (1, 1, D_raw)
    x = torch.nn.functional.adaptive_avg_pool1d(x, target_dim)
    return x.squeeze(0).squeeze(0)  # (target_dim,)
```

### Aggregation Dimension Table

Expected raw embedding dimensions before resize, by probe and aggregation:

| Probe | Subset | per_channel | per_head | per_layer |
|---|---|---|---|---|
| Conv4 | last_block | ~67 (64 channels + 3 bias terms) | N/A | ~3 |
| Conv4 | per_stage | ~259 (64x4 + 3) | N/A | ~4 |
| ResNet-12 | last_block | ~1024 | N/A | ~10 |
| ResNet-12 | per_stage | ~960 (64+128+256+512) | N/A | ~4 |
| ViT-tiny | last_block | ~768 | ~48 (12 heads x 4 components) | ~8 |
| ViT-tiny | per_stage | ~768 | ~48 | ~4 |

When the raw dimension is smaller than `target_dim`, adaptive average pooling upsamples (interpolates). When larger, it downsamples. For best results, choose `target_dim` close to the natural raw dimension.

---

## 6. Normalization Pipeline

Raw aggregated Fisher values span many orders of magnitude (1e-8 to 1e+4 or higher). Without normalization, a single large-Fisher layer dominates the entire embedding, drowning out information from lower-Fisher layers. The normalization pipeline transforms raw Fisher values into a scale-invariant, unit-length vector.

### Stage 1: log1p Transform

```
F_agg = log(1 + F_raw)
```

The `log1p` transform compresses the dynamic range. Fisher values of 0.001 and 1000 become 0.001 and 6.9 respectively -- a range of 6900:1 compressed to roughly 6900:1 in log scale. Without log1p, the ratio is preserved linearly and cosine similarity becomes dominated by the largest components.

```python
def apply_log1p(fisher_agg: torch.Tensor) -> torch.Tensor:
    """Apply log(1+x) transform to aggregated Fisher values.

    The +1 ensures log1p(0) = 0 (zero Fisher stays zero).
    Negative values should not occur in Fisher diagonals;
    clamp to zero as safety.
    """
    return torch.log1p(fisher_agg.clamp(min=0.0))
```

### Stage 2: L2 Normalization

```
embed = F_agg / ||F_agg||_2
```

L2 normalization projects the embedding onto the unit hypersphere. Cosine similarity between two embeddings equals their dot product, simplifying distance computation.

```python
def apply_l2_normalize(
    embedding: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L2-normalize embedding to unit length.

    Args:
        embedding: Raw embedding vector, shape (E,).
        eps: Small constant to avoid division by zero.

    Returns:
        Unit-length embedding vector, shape (E,).
    """
    norm = embedding.norm(p=2).clamp(min=eps)
    return embedding / norm
```

### Stage 3: Optional Whitening

Whitening decorrelates embedding dimensions and equalizes their variance. It requires a reference set of task embeddings (computed on the training split).

```
embed_whitened = W @ (embed - mu)
```

where `mu` is the mean embedding across reference tasks and `W` is the inverse square root of the covariance matrix.

```python
import torch


class EmbeddingWhitener:
    """Whitening transform fitted on reference task embeddings.

    Fitted once on the training split's embeddings. Applied identically
    to train, val, and test embeddings. Frozen after fitting.

    The whitening transform is: W @ (x - mu)
    where mu is the reference mean and W = C^{-1/2}
    (inverse sqrt of covariance).
    """

    def __init__(self, embedding_dim: int, regularization: float = 1e-5):
        self.embedding_dim = embedding_dim
        self.regularization = regularization
        self.mu = None           # (E,)
        self.W = None            # (E, E)
        self.fitted = False
        self.n_reference = 0

    def fit(self, reference_embeddings: torch.Tensor) -> None:
        """Fit whitening transform on reference embeddings.

        Args:
            reference_embeddings: Tensor of shape (N_ref, E)
                where N_ref >= E. Must be log1p-transformed
                and L2-normalized embeddings.

        Raises:
            ValueError: If N_ref < E (covariance would be singular).
        """
        n_ref, dim = reference_embeddings.shape
        if n_ref < dim:
            raise ValueError(
                f"Need at least {dim} reference tasks for whitening, "
                f"got {n_ref}. Collect more reference embeddings "
                f"or disable whitening."
            )

        self.mu = reference_embeddings.mean(dim=0)  # (E,)
        centered = reference_embeddings - self.mu    # (N_ref, E)

        # Covariance matrix
        cov = (centered.T @ centered) / (n_ref - 1)  # (E, E)

        # Regularized inverse square root
        reg_eye = self.regularization * torch.eye(
            dim, device=cov.device, dtype=cov.dtype
        )
        cov += reg_eye
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.clamp(min=self.regularization)
        inv_sqrt_eigenvalues = 1.0 / torch.sqrt(eigenvalues)

        # W = V @ diag(1/sqrt(lambda)) @ V^T
        self.W = (
            eigenvectors
            @ torch.diag(inv_sqrt_eigenvalues)
            @ eigenvectors.T
        )

        self.fitted = True
        self.n_reference = n_ref

    def transform(self, embedding: torch.Tensor) -> torch.Tensor:
        """Apply whitening transform to an embedding.

        Args:
            embedding: Shape (E,) or (batch, E).

        Returns:
            Whitened embedding, same shape as input.
        """
        if not self.fitted:
            raise RuntimeError(
                "Whitener has not been fitted. Call .fit() first."
            )

        centered = embedding - self.mu
        if centered.ndim == 1:
            whitened = self.W @ centered
        else:
            whitened = centered @ self.W.T

        # Re-normalize to unit length after whitening
        if whitened.ndim == 1:
            whitened = whitened / whitened.norm(p=2).clamp(min=1e-8)
        else:
            norms = whitened.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            whitened = whitened / norms

        return whitened

    def state_dict(self) -> dict:
        """Serialize whitener state for checkpoint saving."""
        return {
            "mu": self.mu,
            "W": self.W,
            "fitted": self.fitted,
            "n_reference": self.n_reference,
            "embedding_dim": self.embedding_dim,
            "regularization": self.regularization,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore whitener from checkpoint."""
        self.mu = state["mu"]
        self.W = state["W"]
        self.fitted = state["fitted"]
        self.n_reference = state["n_reference"]
```

### Whitening Safety Rules

| Rule | Rationale |
|---|---|
| Fit only on training split embeddings | Prevents information leakage from val/test tasks |
| Require N_ref >= E | Covariance is rank-deficient with fewer samples than dimensions |
| Add regularization to covariance diagonal | Prevents division by near-zero eigenvalues |
| Freeze after fitting | Whitening transform must not change during validation or testing |
| Re-normalize after whitening | Whitening can change vector norms; restore unit-length invariant |
| Store in checkpoint | Whitener state must be saved/loaded alongside embeddings |

### Normalization Pipeline Selection

| Config Value | Pipeline | When |
|---|---|---|
| `"log1p_l2"` | log1p -> L2-norm | Default. Sufficient for most use cases. |
| `"l2"` | L2-norm only (no log1p) | When Fisher values are already on similar scales. Rare. |
| `"whiten"` | log1p -> L2-norm -> whitening -> L2-norm | When reference task set is available and embedding dimensions are correlated. |

```python
def normalize_embedding(
    raw_fisher_agg: torch.Tensor,
    normalize: str,
    whitener: 'EmbeddingWhitener | None' = None,
) -> torch.Tensor:
    """Apply the configured normalization pipeline.

    Args:
        raw_fisher_agg: Aggregated Fisher vector, shape (E,).
        normalize: One of "log1p_l2", "l2", "whiten".
        whitener: Fitted EmbeddingWhitener
                  (required if normalize="whiten").

    Returns:
        Normalized embedding, shape (E,).
    """
    if normalize == "l2":
        return apply_l2_normalize(raw_fisher_agg)

    elif normalize == "log1p_l2":
        transformed = apply_log1p(raw_fisher_agg)
        return apply_l2_normalize(transformed)

    elif normalize == "whiten":
        if whitener is None or not whitener.fitted:
            raise RuntimeError(
                "Whitening requested but no fitted whitener provided. "
                "Fit a whitener on reference task embeddings first."
            )
        transformed = apply_log1p(raw_fisher_agg)
        l2_normed = apply_l2_normalize(transformed)
        return whitener.transform(l2_normed)

    else:
        raise ValueError(f"Unknown normalization: {normalize}")
```

---

## 7. Task ID Canonicalization

Every task embedding must be associated with a deterministic, reproducible identifier. Two extractions of the same task (same dataset, same classes, same support samples, same transforms) must produce the same `task_id`.

### Canonical Task ID Computation

```python
import hashlib
import json


def compute_task_id(
    dataset_name: str,
    split: str,
    class_ids: list[int],
    support_indices: list[int],
    transforms_signature: str,
) -> str:
    """Compute a deterministic task identifier.

    The task ID is a hex digest of the canonicalized task specification.
    Canonicalization rules:
        - class_ids are sorted ascending
        - support_indices are sorted in class-major order:
          first all indices for class_ids[0], then class_ids[1], etc.
          Within each class, indices are sorted ascending.
        - transforms_signature is a hash of the augmentation pipeline

    Args:
        dataset_name: Dataset name string
                      (e.g., "mini-imagenet", "omniglot").
        split: Data split ("train", "val", "test").
        class_ids: List of class indices selected for this episode.
        support_indices: List of per-class sample indices,
                         in class-major order.
        transforms_signature: Hash string of the data
                              augmentation pipeline.

    Returns:
        Hex string task ID (32 characters).
    """
    # Canonicalize: sort class_ids
    sorted_class_ids = sorted(class_ids)

    payload = json.dumps({
        "dataset": dataset_name,
        "split": split,
        "class_ids": sorted_class_ids,
        "support_indices": support_indices,
        "transforms": transforms_signature,
    }, sort_keys=True, separators=(',', ':'))

    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]
```

### Support Sample Ordering

Support samples must be ordered deterministically. The canonical order is:

1. **Class-major**: all samples for the first class (by sorted class ID), then all samples for the second class, and so on.
2. **Within class**: samples ordered by their dataset index, ascending.

```python
def canonicalize_support_order(
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    class_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder support set into canonical class-major order.

    Args:
        x_support: Support inputs, shape (N*K, ...).
        y_support: Support labels, shape (N*K,).
        class_ids: The N class IDs in this episode.

    Returns:
        Reordered (x_support, y_support) in class-major order.
    """
    sorted_classes = sorted(class_ids)
    indices = []
    for cls in sorted_classes:
        cls_mask = (y_support == cls)
        cls_indices = cls_mask.nonzero(as_tuple=True)[0].tolist()
        cls_indices.sort()  # Ascending by position
        indices.extend(cls_indices)

    reorder = torch.tensor(
        indices, dtype=torch.long, device=x_support.device
    )
    return x_support[reorder], y_support[reorder]
```

### Transforms Signature

Hash the augmentation pipeline to distinguish tasks that use different preprocessing:

```python
def compute_transforms_signature(transform_list: list) -> str:
    """Hash the data augmentation pipeline.

    Captures transform class names and their parameters.
    Two identical augmentation pipelines produce the same hash.
    """
    descriptions = []
    for t in transform_list:
        desc = {
            "class": type(t).__name__,
            "params": {k: str(v) for k, v in vars(t).items()
                       if not k.startswith('_')},
        }
        descriptions.append(desc)

    payload = json.dumps(
        descriptions, sort_keys=True, separators=(',', ':')
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
```

---

## 8. Determinism Requirements

Embedding extraction must be fully deterministic. The same inputs, same probe, and same seed must produce bitwise-identical results on the same device and near-identical results across devices.

### Same-Device Determinism (tolerance: 1e-7)

| Requirement | Implementation |
|---|---|
| Set global random seeds | `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)` |
| Enable deterministic algorithms | `torch.use_deterministic_algorithms(True)` |
| Disable CUDNN benchmark | `torch.backends.cudnn.benchmark = False` |
| Enable CUDNN determinism | `torch.backends.cudnn.deterministic = True` |
| Probe in eval mode | `probe.eval()` -- disables dropout, fixes BN |
| No random augmentation during extraction | Augmentations must be deterministic (seeded or removed) |
| Canonical support ordering | Class-major, index-sorted within class |
| fp32 computation | No AMP, no fp16 accumulation |

```python
import torch
import random
import numpy as np


def set_deterministic_extraction(seed: int) -> None:
    """Configure PyTorch for fully deterministic embedding extraction.

    Call this before every extraction call to ensure reproducibility.

    Args:
        seed: Random seed for all random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
```

### Cross-Device Tolerance (cosine similarity > 0.9999)

Floating-point operations produce slightly different results on CPU vs CUDA, and across CUDA architectures (A100 vs V100). Exact bitwise equality is not achievable cross-device. Accept cosine similarity > 0.9999 as the cross-device determinism standard.

```python
def verify_cross_device_consistency(
    embedding_cpu: torch.Tensor,
    embedding_cuda: torch.Tensor,
    threshold: float = 0.9999,
) -> bool:
    """Verify that embeddings from different devices are
    sufficiently similar.
    """
    cos_sim = torch.nn.functional.cosine_similarity(
        embedding_cpu.unsqueeze(0).float(),
        embedding_cuda.cpu().unsqueeze(0).float(),
        dim=1,
    ).item()
    return cos_sim >= threshold
```

### Operations That Break Determinism

| Operation | Problem | Fix |
|---|---|---|
| Dropout in probe | Random mask changes each run | Probe must be in eval mode |
| BN in training mode | Uses batch statistics, varies with batch composition | Probe must be in eval mode |
| Non-deterministic CUDA kernels | `atomicAdd` in backward, `scatter_add` | `torch.use_deterministic_algorithms(True)` |
| Random data augmentation | Different crops/flips each run | Use only deterministic transforms or seed them |
| Non-sorted parameter iteration | Dict ordering may differ across runs if model is constructed differently | Always sort parameter names |

---

## 9. Full Extraction Pipeline

The `extract_task_embedding` function is the top-level entry point. It orchestrates probe preparation, Fisher computation, aggregation, and normalization into a single deterministic call.

### Complete Implementation

```python
import time
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class TaskEmbedding:
    """Container for an extracted task embedding with diagnostics."""
    embedding: torch.Tensor      # (E,) L2-normalized Fisher-derived task vector
    task_id: str                 # Deterministic hash of task specification
    probe_signature: str         # Hash of probe configuration
    diagnostics: dict            # Fisher norm, sparsity, probe loss, extraction time


def extract_task_embedding(
    probe: nn.Module,
    episode: 'TaskEpisode',
    *,
    model_name: str = "conv4",
    layer_subset: str = "last_block",
    aggregation: str = "per_channel",
    embedding_dim: int = 512,
    normalize: str = "log1p_l2",
    whitener: 'EmbeddingWhitener | None' = None,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> TaskEmbedding:
    """Extract a Task2Vec embedding from an episodic task.

    Full pipeline:
        1. Set deterministic mode
        2. Prepare probe (eval, frozen)
        3. Create ephemeral N-way head
        4. Compute diagonal Fisher over selected parameters
        5. Aggregate Fisher per group
        6. Normalize (log1p + L2, optional whitening)
        7. Package with diagnostics

    Args:
        probe: Pretrained probe backbone.
        episode: TaskEpisode with x_support, y_support, task metadata.
        model_name: Probe backbone name
                    ("conv4", "resnet12", "vit_tiny").
        layer_subset: Parameter subset
                      ("last_block", "per_stage", "all").
        aggregation: Aggregation strategy
                     ("per_channel", "per_head", "per_layer").
        embedding_dim: Target embedding dimension E.
        normalize: Normalization pipeline
                   ("log1p_l2", "l2", "whiten").
        whitener: Fitted whitener (required if normalize="whiten").
        seed: Random seed for deterministic extraction.
        device: Computation device.

    Returns:
        TaskEmbedding with embedding vector and diagnostics.
    """
    t_start = time.perf_counter()

    # 1. Deterministic mode
    set_deterministic_extraction(seed)

    # 2. Prepare probe
    probe = prepare_probe(probe)
    probe = probe.to(device=device, dtype=torch.float32)

    # 3. Move data and canonicalize order
    x_support = episode.x_support.to(device=device, dtype=torch.float32)
    y_support = episode.y_support.to(device=device)
    x_support, y_support = canonicalize_support_order(
        x_support, y_support, episode.class_ids
    )

    # 4. Compute probe features dimension
    with torch.no_grad():
        dummy_features = probe(x_support[:1])
        feature_dim = dummy_features.shape[-1]

    # 5. Create ephemeral head
    n_way = len(set(episode.class_ids))
    head = create_ephemeral_head(feature_dim, n_way, device, torch.float32)

    # 6. Select parameter subset
    param_names = select_parameter_subset(probe, model_name, layer_subset)

    # 7. Compute diagonal Fisher
    with torch.amp.autocast('cuda', enabled=False):
        fisher_diag = compute_diagonal_fisher_auto(
            probe, head, x_support, y_support, param_names
        )

    # 8. Aggregate
    fisher_agg = aggregate_fisher(
        fisher_diag, probe, model_name, aggregation, embedding_dim
    )

    # 9. Normalize
    embedding = normalize_embedding(fisher_agg, normalize, whitener)

    # 10. Compute diagnostics
    fisher_norm = sum(f.norm().item() for f in fisher_diag.values())
    total_elements = sum(f.numel() for f in fisher_diag.values())
    fisher_sparsity = (
        sum((f.abs() < 1e-10).sum().item() for f in fisher_diag.values())
        / total_elements
    )

    # Probe loss (for diagnostics only)
    with torch.no_grad():
        features = probe(x_support)
        logits = head(features)
        probe_loss = nn.functional.cross_entropy(logits, y_support).item()

    t_end = time.perf_counter()

    # 11. Compute task ID and probe signature
    preprocessing = {
        "resize": list(x_support.shape[2:]),
        "dtype": "float32",
    }
    probe_sig = compute_probe_signature(
        model_name, layer_subset, preprocessing
    )

    task_id = compute_task_id(
        dataset_name=episode.dataset_name,
        split=episode.split,
        class_ids=episode.class_ids,
        support_indices=episode.support_indices,
        transforms_signature=episode.transforms_signature,
    )

    return TaskEmbedding(
        embedding=embedding.detach().cpu(),
        task_id=task_id,
        probe_signature=probe_sig,
        diagnostics={
            "fisher_norm": fisher_norm,
            "fisher_sparsity": fisher_sparsity,
            "probe_loss": probe_loss,
            "extraction_time_sec": t_end - t_start,
            "embedding_dim": embedding_dim,
            "param_count": total_elements,
            "n_samples": x_support.shape[0],
            "n_way": n_way,
        },
    )
```

### Extraction Performance Budget

| Probe | Subset | Support Size | Device | Expected Time |
|---|---|---|---|---|
| Conv4 | last_block | 25 (5-way 5-shot) | CPU | ~50ms |
| Conv4 | last_block | 25 | CUDA | ~15ms |
| ResNet-12 | last_block | 25 | CUDA | ~80ms |
| ResNet-12 | per_stage | 25 | CUDA | ~40ms |
| ViT-tiny | last_block | 25 | CUDA | ~120ms |
| ViT-tiny | per_stage | 25 | CUDA | ~60ms |

Batched Fisher computation with vmap reduces wall time by 3-10x for support sets larger than 10 samples.

---

## 10. Probe Head Creation for Arbitrary N-way

The ephemeral head must handle any N from 2 to the maximum number of classes in the dataset. The head is a single linear layer with no hidden layers, no activation, and no dropout.

### Head Lifecycle

```
create_ephemeral_head(feature_dim, N)
    |
    v
[Fisher computation over probe params using cross-entropy through this head]
    |
    v
head goes out of scope, garbage collected
```

The head is never saved, never checkpointed, and never reused across tasks. Two extractions of the same task create two separate heads. Because the head uses default `nn.Linear` initialization (Kaiming uniform), and because the probe is frozen, the Fisher over the probe parameters is dominated by the data distribution, not the head initialization. The head initialization affects the absolute scale of Fisher values but not the relative pattern across probe parameters -- and L2 normalization removes scale.

### Verifying Head Independence

To confirm that head initialization does not materially affect the embedding:

```python
def verify_head_independence(
    probe: nn.Module,
    episode: 'TaskEpisode',
    n_trials: int = 5,
    cosine_threshold: float = 0.99,
) -> bool:
    """Verify that different head initializations produce
    similar embeddings.

    Extract embeddings with different random seeds for head
    initialization (but same seed for everything else) and verify
    pairwise cosine > threshold.
    """
    embeddings = []
    for trial in range(n_trials):
        # Use different seed only for head init; same seed for Fisher
        torch.manual_seed(1000 + trial)  # Head init seed
        emb = extract_task_embedding(
            probe, episode, seed=42,  # Fisher computation seed
        )
        embeddings.append(emb.embedding)

    # Check pairwise cosine similarity
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            cos = torch.nn.functional.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0),
                dim=1,
            ).item()
            if cos < cosine_threshold:
                return False
    return True
```

Note: in practice, for full determinism in the pipeline, seed the head initialization as part of the extraction seed so the head is identical across runs of the same task. The independence check above is a diagnostic, not a runtime guarantee.

---

## 11. Testing Patterns

### A) Embedding Determinism Test

```python
def test_embedding_determinism():
    """Two extractions with same seed and data must produce
    identical embeddings."""
    probe = load_pretrained_probe("conv4")
    episode = create_synthetic_episode(n_way=5, k_shot=5, seed=0)

    emb1 = extract_task_embedding(
        probe, episode, seed=42, device=torch.device("cpu")
    )
    emb2 = extract_task_embedding(
        probe, episode, seed=42, device=torch.device("cpu")
    )

    diff = (emb1.embedding - emb2.embedding).abs().max().item()
    assert diff < 1e-7, (
        f"Embeddings differ by {diff:.2e} (tolerance: 1e-7)"
    )
    assert emb1.task_id == emb2.task_id, (
        "Task IDs differ for identical inputs"
    )
    assert emb1.probe_signature == emb2.probe_signature, (
        "Probe signatures differ"
    )
```

### B) Cross-Device Cosine Similarity Test

```python
def test_cross_device_cosine():
    """CPU and CUDA embeddings must have cosine similarity > 0.9999."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    probe = load_pretrained_probe("conv4")
    episode = create_synthetic_episode(n_way=5, k_shot=5, seed=0)

    emb_cpu = extract_task_embedding(
        probe, episode, seed=42, device=torch.device("cpu")
    )
    emb_cuda = extract_task_embedding(
        probe, episode, seed=42, device=torch.device("cuda")
    )

    cos_sim = torch.nn.functional.cosine_similarity(
        emb_cpu.embedding.unsqueeze(0),
        emb_cuda.embedding.unsqueeze(0),
        dim=1,
    ).item()

    assert cos_sim > 0.9999, (
        f"Cross-device cosine: {cos_sim:.6f} (need > 0.9999)"
    )
```

### C) Fisher Non-Degeneracy Test

```python
def test_fisher_non_degenerate():
    """Fisher diagonal should have meaningful variance,
    not be all zeros or all equal."""
    probe = load_pretrained_probe("conv4")
    episode = create_synthetic_episode(n_way=5, k_shot=5, seed=0)
    param_names = select_parameter_subset(probe, "conv4", "last_block")

    fisher = compute_diagonal_fisher(
        prepare_probe(probe),
        create_ephemeral_head(64, 5, torch.device("cpu")),
        episode.x_support, episode.y_support, param_names,
    )

    for name, f in fisher.items():
        assert f.abs().sum() > 0, (
            f"Fisher for '{name}' is all zeros"
        )
        assert f.std() > 1e-10, (
            f"Fisher for '{name}' has zero variance (degenerate)"
        )
```

### D) Different Tasks Produce Different Embeddings

```python
def test_different_tasks_different_embeddings():
    """Tasks with different class compositions must produce
    distinct embeddings."""
    probe = load_pretrained_probe("conv4")

    episode_a = create_synthetic_episode(
        n_way=5, k_shot=5, seed=0, class_offset=0
    )
    episode_b = create_synthetic_episode(
        n_way=5, k_shot=5, seed=0, class_offset=5
    )

    emb_a = extract_task_embedding(probe, episode_a, seed=42)
    emb_b = extract_task_embedding(probe, episode_b, seed=42)

    cos_sim = torch.nn.functional.cosine_similarity(
        emb_a.embedding.unsqueeze(0),
        emb_b.embedding.unsqueeze(0),
        dim=1,
    ).item()

    assert cos_sim < 0.99, (
        f"Different tasks have cosine {cos_sim:.4f} "
        f"-- embeddings are not discriminative"
    )
```

### E) Whitening Requires Minimum Reference Tasks

```python
def test_whitening_minimum_references():
    """Whitener must raise ValueError if reference count
    < embedding dim."""
    whitener = EmbeddingWhitener(embedding_dim=512)

    # Only 100 reference embeddings for dim=512 -- should fail
    too_few = torch.randn(100, 512)
    with pytest.raises(ValueError, match="Need at least 512"):
        whitener.fit(too_few)

    # Enough references -- should succeed
    enough = torch.randn(600, 512)
    whitener.fit(enough)
    assert whitener.fitted
```

---

## 12. Anti-Patterns

**Using the meta-learner as the probe**

The probe must be a fixed, pretrained, external reference. Using the meta-learner (the model being trained) as the probe causes embeddings to change every time the meta-learner updates, destroying the fixed coordinate system needed for task comparison.

**Computing Fisher with the probe in training mode**

Training mode enables dropout and uses batch statistics for BN. Both introduce non-determinism. Always call `probe.eval()` and verify all BN layers are in eval mode.

**Accumulating Fisher in fp16**

Squared gradients in fp16 overflow for values > 256 (256^2 = 65536 > fp16 max 65504). Even moderate gradients produce overflow. Always accumulate in fp32.

**Retaining the computational graph during Fisher computation**

Pass `create_graph=False` to `torch.autograd.grad`. The Fisher computation does not need second-order gradients. Retaining the graph wastes memory and can cause unexpected interactions with the outer training loop.

**Comparing embeddings from different probes**

Embeddings are defined relative to a specific probe. A Conv4 embedding and a ResNet-12 embedding live in different spaces. Always check `probe_signature` equality before computing distances.

**Skipping log1p normalization**

Without log1p, a single layer with Fisher values 1000x larger than others dominates the entire embedding. Cosine similarity becomes a comparison of that one layer, ignoring all others.

**Using random support ordering**

If support samples are fed to the probe in different orders across extractions, per-sample gradient accumulation produces different numerical results (due to floating-point non-commutativity). Always use canonical class-major ordering.

---

## 13. Configuration Reference

### Task2VecConfig Fields

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `probe_model` | `str` | `"conv4"` | `{"conv4", "resnet12", "vit_tiny"}` | Probe backbone architecture |
| `layer_subset` | `str` | `"last_block"` | `{"last_block", "per_stage", "all"}` | Which layers to compute Fisher over |
| `embedding_dim` | `int` | `512` | `{256, 512, 1024}` | Target embedding dimension after aggregation |
| `aggregation` | `str` | `"per_channel"` | `{"per_channel", "per_head", "per_layer"}` | Fisher aggregation grouping |
| `normalize` | `str` | `"log1p_l2"` | `{"log1p_l2", "l2", "whiten"}` | Normalization pipeline |
| `num_fisher_samples` | `int or None` | `None` | `None` or `> 0` | Limit Fisher samples (None = all support) |
| `force_fp32` | `bool` | `True` | -- | Force fp32 for Fisher computation |
| `use_vmap` | `bool` | `True` | -- | Use vmap for batched Fisher (if available) |

### Preset Configurations

```python
@dataclass
class Task2VecConfig:
    probe_model: str = "conv4"
    layer_subset: str = "last_block"
    embedding_dim: int = 512
    aggregation: str = "per_channel"
    normalize: str = "log1p_l2"
    num_fisher_samples: int | None = None
    force_fp32: bool = True
    use_vmap: bool = True

    @classmethod
    def minimal(cls) -> 'Task2VecConfig':
        """Minimal config for unit tests: small probe, low dim."""
        return cls(
            probe_model="conv4",
            layer_subset="per_layer",
            embedding_dim=256,
            aggregation="per_layer",
            normalize="log1p_l2",
            use_vmap=False,
        )

    @classmethod
    def dev(cls) -> 'Task2VecConfig':
        """Dev config: Conv4 probe, 512-dim, fast extraction."""
        return cls(
            probe_model="conv4",
            layer_subset="last_block",
            embedding_dim=512,
            aggregation="per_channel",
            normalize="log1p_l2",
        )

    @classmethod
    def production(cls) -> 'Task2VecConfig':
        """Production config: ResNet-12 probe, 1024-dim, whitening."""
        return cls(
            probe_model="resnet12",
            layer_subset="per_stage",
            embedding_dim=1024,
            aggregation="per_channel",
            normalize="whiten",
        )
```

---

## Appendix A: Mathematical Properties of Diagonal Fisher Embeddings

The diagonal Fisher information has the following properties relevant to task embeddings:

**Positive semi-definiteness**: Every diagonal entry is a squared gradient, hence non-negative. `F_diag[i] >= 0` for all i.

**Interpretation**: `F_diag[i]` measures how much the log-likelihood changes when parameter `theta_i` is perturbed. High Fisher = the task is sensitive to this parameter. Low Fisher = the task is indifferent to this parameter.

**Approximation quality**: The diagonal approximation discards all off-diagonal correlations. For networks with correlated parameters (e.g., adjacent conv filters), this loses information. The aggregation step (per-channel mean) partially recovers correlation structure by averaging over correlated groups.

**Relationship to gradient descent**: The Fisher information is the Hessian of the KL divergence between the model distribution at theta and at theta + delta. Parameters with high Fisher are the ones that gradient descent would modify most aggressively. The Task2Vec embedding captures the "gradient fingerprint" of a task.

**Invariance to label permutation**: Permuting class labels does not change the Fisher over probe parameters (the cross-entropy loss is symmetric in label assignment). Two tasks with the same visual structure but different label indices produce identical embeddings.

## Appendix B: Embedding Dimension Selection Guide

| Embedding Dim | Raw Dimension Coverage | Use Case |
|---|---|---|
| 256 | May require significant downsampling for conv probes | Fast clustering, low storage, coarse task similarity |
| 512 | Good match for Conv4 last_block per_channel (~67 upsampled) or ResNet-12 per_stage (~960 downsampled) | Default; balanced resolution vs. cost |
| 1024 | Preserves more structure from ResNet-12/ViT per_stage | Production; fine-grained task discrimination |

When the raw aggregated dimension is much smaller than `target_dim` (e.g., per_layer aggregation producing 4 values for target 512), the adaptive pooling upsampling introduces redundancy. In such cases, either reduce `target_dim` or use a finer aggregation strategy.

When the raw dimension is much larger than `target_dim` (e.g., `"all"` subset on ResNet-12), significant information loss occurs during downsampling. Use `"per_stage"` or `"last_block"` subset instead of `"all"` for practical embedding dimensions.
