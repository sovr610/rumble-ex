# Contrastive Alignment Theory

> Reference for multi-modal contrastive learning theory, loss functions,
> temperature tuning, CLIP/SigLIP variants, and batch size effects.
> Part of the multi-modal-alignment skill.

---

## 1. Overview

Contrastive alignment learns a shared embedding space where semantically paired
representations from different modalities are close together while unpaired
representations are far apart. The core insight is that by optimizing a
discriminative objective over positive/negative pairs, the model learns
modality-invariant features without requiring explicit feature-level supervision.

---

## 2. InfoNCE Loss

### 2.1 Mathematical Formulation

Given a batch of B aligned pairs `{(a_i, b_i)}_{i=1}^{B}` where `a_i` is from
modality A and `b_i` is from modality B:

```
z_a_i = normalize(proj_a(a_i))    # L2-normalized projection
z_b_i = normalize(proj_b(b_i))    # L2-normalized projection

sim(i, j) = z_a_i . z_b_j / tau   # temperature-scaled cosine similarity

L_a2b = -(1/B) * sum_i log(exp(sim(i,i)) / sum_j exp(sim(i,j)))
L_b2a = -(1/B) * sum_i log(exp(sim(i,i)) / sum_j exp(sim(j,i)))

L_infonce = 0.5 * (L_a2b + L_b2a)
```

### 2.2 Interpretation

InfoNCE is a softmax cross-entropy loss over the similarity matrix with the
identity matrix as target labels. Each row of the similarity matrix defines a
B-way classification problem: "which of the B candidates in modality B is the
correct match for this modality A sample?"

The symmetric formulation (averaging both directions) ensures that the loss
treats both modalities equally, preventing one from dominating the gradient.

### 2.3 Gradient Properties

The gradient of InfoNCE with respect to `z_a_i`:

```
dL/dz_a_i = (1/tau) * sum_j (p(j|i) - 1_{j=i}) * z_b_j
```

Where `p(j|i) = softmax(sim(i,:))_j`. This has an elegant interpretation:
the gradient pushes `z_a_i` toward its true match `z_b_i` and away from
false matches, weighted by the softmax probability of confusion.

### 2.4 Implementation Notes

- **Numerical stability**: Subtract `max(sim(i,:))` before computing softmax
  to prevent overflow. This is equivalent to log-sum-exp trick.
- **Memory**: The similarity matrix is `(B, B)` which is O(B^2). For very
  large batches, consider chunked computation.
- **Precision**: Compute similarity and loss in fp32 even under AMP. The
  temperature division amplifies numerical errors.

---

## 3. Temperature

### 3.1 Role of Temperature

Temperature `tau` controls the sharpness of the contrastive distribution:

| tau | Effect | Regime |
|-----|--------|--------|
| < 0.01 | Very sharp, focuses on hardest negatives | Unstable, risk of gradient explosion |
| 0.01 -- 0.05 | Sharp, good discrimination | CLIP default range |
| 0.05 -- 0.1 | Moderate, balanced learning | Good default |
| 0.1 -- 0.5 | Soft, tolerant of near-misses | Early training, warm-up |
| > 0.5 | Very soft, almost uniform | Not useful, loss uninformative |

### 3.2 Learnable Temperature

CLIP uses a learnable log-temperature `log_tau` initialized to `log(1/0.07)`:

```python
self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

@property
def temperature(self):
    return self.log_temperature.exp().clamp(min=0.001, max=1.0)
```

Benefits of learnable temperature:
- Adapts to the difficulty of the contrastive task
- Adjusts to batch size (larger batches may need lower temperature)
- Compensates for changes in embedding distribution during training

Risks:
- Can diverge to very small values, causing gradient explosion
- Must be clamped to a valid range
- fp32 precision is essential

### 3.3 Temperature Scheduling

Alternative to learnable temperature: schedule temperature during training.

```
tau(t) = tau_init * (tau_final / tau_init) ^ (t / T)
```

This geometric schedule starts warm (high temperature) and progressively
sharpens. Useful when learnable temperature is unstable.

---

## 4. CLIP Variant

### 4.1 Original CLIP Architecture

CLIP (Radford et al., 2021) uses:
- Separate vision and text encoders (ViT + GPT-2 style)
- Linear projection heads (single linear layer + L2 norm)
- Learnable temperature starting at 1/0.07
- Symmetric InfoNCE loss
- Very large batch sizes (32,768)

### 4.2 Key Design Decisions

1. **Linear projector**: CLIP uses a single linear layer, but follow-up work
   (LiT, SLIP) shows MLP projectors improve alignment quality.
2. **Global average pooling**: Vision encoder pools spatial features to a single
   vector before projection. Text encoder uses the [EOS] token embedding.
3. **Large batch**: Contrastive learning quality scales with batch size because
   more negatives improve discrimination. Below B=256, InfoNCE degrades.
4. **No data augmentation in text**: Text augmentation is hard; CLIP relies
   on the natural diversity of image-text pairs.

---

## 5. SigLIP Variant

### 5.1 Motivation

SigLIP (Zhai et al., 2023) replaces the global softmax in InfoNCE with
pairwise sigmoid losses. This avoids:
- The O(B) global normalization that limits distributed training
- The requirement that exactly one positive exists per row

### 5.2 Mathematical Formulation

```
t_ij = z_a_i . z_b_j / tau       # pairwise similarity
y_ij = 1 if i == j else 0         # target label
bias = -log(B - 1)                 # prior correction

L_siglip = -(1/B^2) * sum_{i,j} log(sigma(t_ij * (2*y_ij - 1) + bias*(1 - y_ij)))
```

Where `sigma` is the sigmoid function. The bias term corrects for the
class imbalance (B-1 negatives per positive).

### 5.3 Comparison with InfoNCE

| Property | InfoNCE | SigLIP |
|----------|---------|--------|
| Normalization | Global softmax per row | Pairwise sigmoid |
| Distributed training | Requires all-gather for denominator | Embarrassingly parallel |
| Multiple positives | Not supported (1 positive per row) | Naturally supported |
| Gradient scale | Depends on batch size | Constant per pair |
| Batch size sensitivity | Degrades below B=256 | More robust to small B |
| Implementation complexity | Moderate | Simple |

### 5.4 Implementation

```python
def siglip_loss(sim, temperature, bias):
    """SigLIP pairwise sigmoid contrastive loss."""
    B = sim.shape[0]
    labels = 2.0 * torch.eye(B, device=sim.device) - 1.0  # +1 on diag, -1 off-diag
    logits = sim / temperature
    # Apply bias to negative pairs only
    logits = logits + bias * (1.0 - torch.eye(B, device=sim.device))
    return -F.logsigmoid(labels * logits).mean()
```

---

## 6. Uniformity and Alignment Losses

### 6.1 The Uniformity-Alignment Framework

Wang & Isola (2020) decompose contrastive loss into two properties:

1. **Alignment**: Positive pairs should be close together.
2. **Uniformity**: All representations should be uniformly distributed on
   the unit hypersphere.

```
L_align = E_{(x,y)~pos} [||f(x) - f(y)||^2]

L_uniform = log E_{(x,y)~all} [exp(-2 * ||f(x) - f(y)||^2)]
```

### 6.2 Why Both Matter

- High alignment alone causes **collapse**: all embeddings map to the same point.
- High uniformity alone causes **dispersion**: positive pairs are not distinguished.
- InfoNCE implicitly optimizes both, but adding explicit terms as regularizers
  improves training stability.

### 6.3 Integration with InfoNCE

```
L_total = L_infonce + lambda_uniform * L_uniform + lambda_align * L_align
```

Typical values: `lambda_uniform = 0.1`, `lambda_align = 0.0` (alignment is
already enforced by InfoNCE, so the uniform term is more impactful).

---

## 7. Batch Size Effects

### 7.1 Scaling Laws

Contrastive learning performance scales approximately as:

```
performance ~ alpha * log(B) + beta
```

Where B is the effective batch size. Each doubling of batch size provides
roughly constant improvement.

### 7.2 Effective Strategies for Small Batches

| Technique | Description | Effective B |
|-----------|-------------|-------------|
| Memory bank | Store recent embeddings, use as extra negatives | B + bank_size |
| Momentum encoder | EMA-updated encoder provides stable negatives | 2B to 4B |
| Gradient accumulation | Accumulate similarity matrices across steps | B * accumulation_steps |
| MoCo-style queue | FIFO queue of past embeddings | B + queue_size |

### 7.3 When Batch Size is Not Enough

Even with large batches, contrastive learning can fail if:
- Positives are too easy (trivially different modalities)
- Negatives are too easy (completely unrelated pairs)
- The embedding space is too low-dimensional (information bottleneck)

Hard negative mining (see SKILL.md) addresses the first two issues.

---

## 8. Combined Loss Strategy

For the brain_ai alignment module, the recommended combined loss is:

```
L = L_contrastive + alpha * L_uniformity + beta * L_gap_reg

Where:
    L_contrastive = InfoNCE or SigLIP (configurable)
    L_uniformity  = uniformity loss on the hypersphere
    L_gap_reg     = ||mean(z_a) - mean(z_b)||^2  (modality gap penalty)
```

Default weights: alpha = 0.1, beta = 0.01.

This combination:
1. Learns discriminative cross-modal similarity (contrastive)
2. Prevents representation collapse (uniformity)
3. Reduces modality gap (gap regularization)

---

## 9. Numerical Safety Checklist

| Concern | Mitigation |
|---------|-----------|
| Softmax overflow | Subtract row-max before exp |
| Temperature near zero | Clamp to [0.001, 1.0] |
| fp16 precision loss | Force fp32 for similarity and loss computation |
| Large similarity values | Monitor max(sim); if > 50, temperature too low |
| Gradient explosion | Gradient clipping at 1.0 for temperature parameter |
| NaN in log | Add epsilon (1e-8) inside log for numerical safety |
