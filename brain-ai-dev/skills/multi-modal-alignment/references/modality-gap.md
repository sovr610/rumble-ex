# Modality Gap Phenomenon

> Reference for understanding and mitigating the modality gap in multi-modal
> alignment: causes, measurement, uniformity loss, centering approaches,
> and gap regularization strategies.
> Part of the multi-modal-alignment skill.

---

## 1. Overview

The modality gap is a systematic geometric separation between representations
from different modalities in the shared embedding space, even after contrastive
training achieves high cross-modal retrieval accuracy. First formally
characterized by Liang et al. (2022), the gap manifests as modality-specific
clusters on the unit hypersphere that do not overlap.

This gap is problematic for the brain_ai Global Workspace because:
1. Cross-modal similarity scores are biased by modality identity
2. Workspace competition favors within-modality coherence over true semantic relevance
3. Broadcast packets from different modalities are not directly comparable

---

## 2. Characterization

### 2.1 Measuring the Gap

The modality gap is the distance between modality centroids in the shared space:

```
centroid_a = mean(z_a_i for i in 1..N)
centroid_b = mean(z_b_i for i in 1..N)

gap = ||centroid_a - centroid_b||_2
```

For L2-normalized embeddings, the gap can also be measured via cosine distance:

```
gap_cosine = 1 - cos(centroid_a, centroid_b)
```

### 2.2 Typical Gap Values

| Training Stage | Gap (L2) | Gap (Cosine) | Interpretation |
|---------------|----------|--------------|----------------|
| Random init | 0.5 -- 1.5 | 0.3 -- 0.8 | Large, expected |
| After 1K steps | 0.3 -- 0.8 | 0.1 -- 0.4 | Reducing |
| Converged (no mitigation) | 0.2 -- 0.5 | 0.05 -- 0.2 | Persistent |
| Converged (with mitigation) | 0.05 -- 0.15 | 0.01 -- 0.05 | Acceptable |

### 2.3 Per-Modality Analysis

Track the gap between every pair of modalities:

```
gaps = {}
for (m_a, m_b) in combinations(modalities, 2):
    gaps[(m_a, m_b)] = ||centroid(m_a) - centroid(m_b)||
```

In brain_ai, the most critical gaps are:
- Vision-Text (primary for semantic grounding)
- Audio-Text (for speech understanding)
- Sensors-Vision (for embodied perception)

---

## 3. Causes

### 3.1 Initialization Bias

Random initialization of modality-specific projection heads creates distinct
starting regions in the shared space. Unless training actively pushes modalities
together, these initial clusters persist because contrastive loss only optimizes
relative ordering, not absolute position.

### 3.2 Cone Effect

L2-normalized embeddings lie on the unit hypersphere S^{d-1}. Different
modalities tend to occupy different "cones" (angular regions) on this sphere.
The cone effect arises because:
- Different encoder architectures produce features with different directional
  distributions
- Projection heads map these to different angular regions
- Contrastive loss does not penalize angular separation between modality clusters

### 3.3 Training Dynamics

The softmax in InfoNCE only requires that the correct positive scores higher
than all negatives within the same row. It does not require that positives
from different modalities have similar absolute similarity scores. This means
the loss can reach near-zero even with a large modality gap.

### 3.4 Feature Heterogeneity

Fundamentally, different modalities encode different types of information:
- Vision: spatial layout, texture, color
- Text: symbolic, sequential, abstract
- Audio: spectral, temporal, rhythmic
- Sensors: continuous-valued, time-series

These inherent differences create pressure to maintain distinct regions in
embedding space, even when semantic content overlaps.

---

## 4. Uniformity Loss

### 4.1 Mathematical Definition

The uniformity loss encourages embeddings to be uniformly distributed on
the unit hypersphere:

```
L_uniform = log E_{(z_i, z_j) ~ all pairs} [exp(-t * ||z_i - z_j||^2)]
```

Where `t` is a temperature parameter (default t=2). Lower values of
L_uniform indicate more uniform distributions.

### 4.2 Why It Helps

Uniformity loss directly counteracts the cone effect by penalizing any
clustering of embeddings, whether within-modality or cross-modality.
Combined with the alignment objective of contrastive loss, uniformity
pushes all embeddings to spread out on the sphere while keeping positive
pairs close.

### 4.3 Implementation

```python
def uniformity_loss(z, t=2.0):
    """Log-average-exp of pairwise squared distances."""
    # z: (N, D), L2-normalized
    sq_dist = torch.cdist(z, z, p=2).pow(2)
    # Exclude self-pairs
    mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    return torch.log(torch.exp(-t * sq_dist[mask]).mean() + 1e-8)
```

### 4.4 Cross-Modal Uniformity

Apply uniformity loss separately to each modality AND to the combined pool:

```
L_uniform_total = w_intra * (L_uniform(z_a) + L_uniform(z_b)) / 2
                + w_inter * L_uniform(concat(z_a, z_b))
```

The inter-modality term is crucial for gap reduction because it penalizes
separation between modality clusters.

---

## 5. Centering

### 5.1 Mechanism

Centering subtracts the running mean (centroid) from each modality's
embeddings before computing the contrastive loss:

```
centered_z_a = z_a - EMA(mean(z_a))
centered_z_b = z_b - EMA(mean(z_b))
```

The EMA (exponential moving average) provides a stable estimate of the
centroid without batch-dependent noise.

### 5.2 Implementation

```python
class ModalityCentering(nn.Module):
    def __init__(self, dim, momentum=0.9, num_modalities=5):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('centroids', torch.zeros(num_modalities, dim))
        self.modality_to_idx = {}

    def update_and_center(self, z, modality):
        idx = self.modality_to_idx.setdefault(modality, len(self.modality_to_idx))
        batch_mean = z.mean(dim=0)
        self.centroids[idx] = (
            self.momentum * self.centroids[idx]
            + (1 - self.momentum) * batch_mean.detach()
        )
        return z - self.centroids[idx]
```

### 5.3 Centering + L2 Normalization

After centering, re-normalize to the unit sphere:

```
centered = z - centroid
normalized = F.normalize(centered, dim=-1)
```

This two-step process (center then normalize) effectively removes the
modality-specific directional bias while maintaining the hyperspherical
geometry that cosine similarity requires.

### 5.4 When to Center

| Phase | Center? | Rationale |
|-------|---------|-----------|
| Training (forward) | Yes | Reduces gap during loss computation |
| Training (stored) | No | Store un-centered for EMA update |
| Inference | Optional | Center if gap matters for downstream |
| Workspace input | Yes | Ensures fair cross-modal competition |

---

## 6. Gap Regularization

### 6.1 Direct Gap Penalty

Add a loss term that penalizes the distance between modality centroids:

```
L_gap = sum_{(a,b)} ||mean(z_a) - mean(z_b)||^2
```

This is the most direct approach: explicitly minimize the gap.

### 6.2 Weighted Gap Regularization

Weight gap penalties by modality pair importance:

```
L_gap = sum_{(a,b)} w_{a,b} * ||mean(z_a) - mean(z_b)||^2
```

For brain_ai, vision-text gap is most critical, so it gets the highest weight.

### 6.3 Gradient Analysis

The gradient of L_gap with respect to z_a_i:

```
dL_gap/dz_a_i = (2/B) * (mean(z_a) - mean(z_b))
```

This pushes all embeddings of modality A toward the centroid of modality B.
The 1/B scaling means individual samples receive small gradients, which is
desirable (gap reduction should be a gentle global shift, not a strong
per-sample force).

### 6.4 Recommended Weights

| Combination | Purpose | Default Weight |
|-------------|---------|---------------|
| InfoNCE | Primary alignment | 1.0 |
| Uniformity | Prevent collapse | 0.1 |
| Gap regularization | Close modality gap | 0.01 |

Gap regularization weight should be small (0.001 to 0.05) to avoid
overpowering the contrastive signal.

---

## 7. Advanced Techniques

### 7.1 Prototypical Alignment

Instead of aligning centroids, align prototypes -- cluster centers within
each modality:

```
prototypes_a = kmeans(z_a, K)
prototypes_b = kmeans(z_b, K)
L_proto = optimal_transport_distance(prototypes_a, prototypes_b)
```

More nuanced than centroid alignment but computationally expensive.

### 7.2 Adversarial Modality Confusion

Train a discriminator to predict modality from embeddings; the aligner
optimizes to fool it:

```
L_adv = -H(D(z))  # maximize entropy of modality prediction
```

This forces embeddings to be modality-invariant. However, it can hurt
alignment quality if modality-specific information is useful.

### 7.3 Manifold Mixing

Interpolate between modality embeddings during training:

```
z_mix = alpha * z_a + (1 - alpha) * z_b
z_mix = F.normalize(z_mix, dim=-1)
```

Use mixed embeddings as additional training signal. This creates artificial
points between modality clusters, helping bridge the gap.

---

## 8. Monitoring and Diagnostics

### 8.1 Metrics to Track

| Metric | Formula | Target |
|--------|---------|--------|
| Gap (L2) | `||mean(z_a) - mean(z_b)||_2` | < 0.15 |
| Gap (cosine) | `1 - cos(mean(z_a), mean(z_b))` | < 0.05 |
| Intra-modality spread | `std(z_a)` | > 0.3 |
| Inter-modality overlap | fraction of z_a nearest neighbors that are z_b | > 0.3 |
| Uniformity | `log-avg-exp(-t * sq_dist)` | < -2.0 |

### 8.2 Visualization

1. **t-SNE/UMAP**: Color-code by modality. Well-aligned representations
   show mixed colors; poorly aligned show distinct clusters.
2. **Similarity histogram**: Plot distribution of cos(z_a, z_b) for
   matched vs unmatched pairs. Well-aligned: matched >> unmatched.
3. **Centroid trajectory**: Track centroid positions over training to
   verify convergence.

---

## 9. Integration with brain_ai Workspace

### 9.1 Pre-Workspace Centering

Before encoder outputs enter workspace competition, apply centering:

```
z_vision_centered = z_vision - centroid_vision
z_text_centered = z_text - centroid_text
...
```

This ensures that workspace scoring compares semantics, not modality identity.

### 9.2 Gap-Aware Competition Scoring

Modify the workspace competition scorer to normalize by modality statistics:

```
score_adjusted = score - gap_bias[modality]
```

This compensates for any residual gap that alignment training did not eliminate.

### 9.3 Monitoring in Production

Log gap metrics during inference to detect:
- Distribution shift (gap increasing after deployment)
- New modality integration issues
- Encoder fine-tuning that breaks alignment
