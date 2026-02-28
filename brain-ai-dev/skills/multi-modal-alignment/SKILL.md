---
name: Multi-Modal Alignment for Shared Embedding Space
description: >-
  This skill should be used when the user asks to "align modality representations",
  "multi-modal contrastive learning", "cross-modal alignment", "CLIP-style training",
  "modality gap reduction", "shared embedding space", "cross-attention alignment",
  "vision-language alignment", "audio-text alignment", "representation binding",
  "cross-modal retrieval", "modality projection heads", "alignment loss function",
  "modal invariance", "add contrastive loss", "implement projection heads",
  "fix modality gap", "add uniformity loss", "implement cross-modal retrieval",
  "add hard negative mining", "implement curriculum alignment",
  "add alignment metrics", "fix representation collapse", "add gap regularization",
  "implement SigLIP loss", "add temporal binding", "fix spatial correspondence",
  or mentions InfoNCE, contrastive alignment, projection head architecture,
  modality gap phenomenon, cross-modal similarity, alignment temperature,
  or shared semantic space in the cognitive pipeline.
version: 0.1.0
---

# Multi-Modal Alignment for Shared Embedding Space

## Purpose

This skill ensures that encoder outputs from different modalities (vision, text,
audio, sensors) occupy a shared semantic space where representations can be
meaningfully compared, retrieved, and fused. Without alignment, the Global
Workspace receives representations from different modalities that cluster in
disjoint regions of embedding space, making cross-modal competition, retrieval,
and transfer unreliable. Multi-modal alignment is the critical bridge between
modality-specific encoders and the unified workspace.

## Key Files

| Target Module | Template Asset | Purpose |
|---|---|---|
| `brain_ai/alignment/aligner.py` | `assets/modality_aligner_template.py` | ModalityAligner with projection heads, forward pass |
| `brain_ai/alignment/losses.py` | `assets/contrastive_losses_template.py` | InfoNCE, SigLIP, uniformity + alignment losses |
| `brain_ai/alignment/retrieval.py` | `assets/cross_modal_retriever_template.py` | CrossModalRetriever for retrieval evaluation |
| `brain_ai/alignment/metrics.py` | `assets/alignment_metrics_template.py` | AlignmentMetrics computing gap, recall, uniformity |
| `brain_ai/alignment/config.py` | `assets/alignment_config_template.py` | AlignmentConfig dataclass with validation |

## Public Contract

All alignment interactions use the ModalityAligner interface:

```python
forward(
    modality_a: str,
    feats_a: Tensor,          # (B, T_a, D_enc)
    modality_b: str,
    feats_b: Tensor,          # (B, T_b, D_enc)
    mask_a: Optional[Tensor] = None,  # (B, T_a) bool
    mask_b: Optional[Tensor] = None,  # (B, T_b) bool
) -> AlignmentOutput
```

The `AlignmentOutput` contains projected representations, similarity matrix,
and optional loss terms:

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `proj_a` | `(B, D_align)` | float | Projected + pooled representation for modality A |
| `proj_b` | `(B, D_align)` | float | Projected + pooled representation for modality B |
| `similarity` | `(B, B)` | float | Cross-modal similarity matrix |
| `loss` | `()` | float | Contrastive alignment loss (when training) |
| `metrics` | `Dict` | -- | Alignment diagnostics (gap, uniformity, etc.) |

## Contrastive Alignment (CLIP-Style)

The primary alignment mechanism uses contrastive learning with InfoNCE loss
over paired representations from different modalities.

### InfoNCE Loss

Given a batch of B paired representations `(z_a, z_b)`:

```
sim(i, j) = cos(z_a_i, z_b_j) / temperature
loss = -0.5 * (CE(sim, I) + CE(sim^T, I))
```

Where `CE` is cross-entropy with identity labels (diagonal = positive pairs).
Temperature controls sharpness: lower temperature produces harder contrastive
signals. Typical range: 0.01 to 0.1 (learnable or fixed).

### SigLIP Variant

Replaces softmax cross-entropy with pairwise sigmoid binary cross-entropy:

```
loss = -mean(log_sigmoid(t_ij * (2 * y_ij - 1)))
```

Where `t_ij = sim(i, j) / temperature` and `y_ij = 1` if `i == j` else `0`.
SigLIP scales better with batch size and avoids the global softmax bottleneck.

### Batch Size Effects

Contrastive alignment quality scales with batch size because more in-batch
negatives improve the quality of the learned similarity function. Effective
strategies for small-batch regimes include memory banks, momentum encoders,
and gradient accumulation of the similarity matrix.

## Projection Heads

Learnable projectors map from each encoder's output space to the shared
alignment space. Projection heads are critical for decoupling the encoder
representation (optimized for modality-specific tasks) from the alignment
representation (optimized for cross-modal similarity).

### Architecture Options

| Variant | Architecture | When to Use |
|---|---|---|
| Linear | `Linear(D_enc, D_align)` | Baseline, fast, fewer params |
| MLP-1 | `Linear -> LayerNorm -> GELU -> Linear` | Default, good accuracy/cost tradeoff |
| MLP-2 | `Linear -> LN -> GELU -> Linear -> LN -> GELU -> Linear` | Large-scale, high-capacity |

All projectors output L2-normalized embeddings. The normalization happens after
projection, ensuring that cosine similarity is well-behaved.

### Modality-Specific vs Shared

Each modality gets its own projection head. Sharing projectors across modalities
is an anti-pattern because encoder output distributions differ substantially
between modalities (vision features from patch embeddings vs text features from
token embeddings have very different statistics).

## Modality Gap Problem

Even after contrastive training, representations from different modalities
often cluster in separate regions of the shared space. This "modality gap"
means that a vision embedding and a text embedding describing the same concept
may have high cosine similarity relative to other pairs but still occupy
distinct neighborhoods in absolute terms.

### Causes

1. **Initialization bias** -- random projector init creates modality-specific
   clusters that training may not fully dissolve.
2. **Cone effect** -- L2-normalized embeddings lie on a hypersphere; modalities
   occupy different cones on that sphere.
3. **Training dynamics** -- contrastive loss only requires relative ordering
   (positive > negative), not absolute proximity.

### Solutions

| Technique | Mechanism | Config Field |
|---|---|---|
| Uniformity loss | Penalizes clustering on the hypersphere | `use_uniformity_loss` |
| Centering | Subtract running mean per modality | `use_centering` |
| Gap regularization | Penalize distance between modality centroids | `gap_reg_weight` |
| Shared init | Initialize projectors from common weights | `shared_init` |

## Cross-Attention Alignment

Complementary to contrastive alignment, cross-attention computes soft alignment
scores between token-level representations from two modalities. This is useful
when fine-grained correspondence matters (e.g., which image patch corresponds
to which text token).

```python
# Cross-attention: Q from modality A, K/V from modality B
attn_scores = softmax(Q_a @ K_b^T / sqrt(d_k)) @ V_b
```

Cross-attention alignment produces token-level alignment maps that can be
supervised (when ground-truth correspondences exist) or used unsupervised
as auxiliary alignment signals.

## Hard Negative Mining

Standard contrastive learning uses all in-batch pairs as negatives. Hard
negative mining selects the most informative negatives -- those that are
similar but not matching. Strategies:

1. **In-batch hard negatives** -- for each positive pair, identify the
   highest-scoring negative within the batch.
2. **Semi-hard negatives** -- negatives that are closer than the positive
   but still on the correct side of the margin.
3. **Cross-modal hard negatives** -- negatives from different modalities
   that share partial semantic overlap.

## Curriculum Alignment

Progressive training strategy:

| Phase | Epoch Range | Pairs | Difficulty |
|---|---|---|---|
| Warm-up | 0 -- 5 | Easy, high-similarity pairs | Low temperature |
| Standard | 5 -- 20 | All pairs, uniform sampling | Medium temperature |
| Hard | 20+ | Hard negative emphasis | Higher temperature |

Curriculum alignment prevents early representation collapse by starting with
clearly distinct pairs before introducing ambiguous cases.

## Integration with Global Workspace

Alignment quality directly affects workspace competition. When modality
representations are well-aligned:

1. **Competition scoring** -- cross-modal similarity scores are meaningful,
   enabling fair comparison across modalities.
2. **Workspace slots** -- fused slot representations combine information
   from multiple modalities coherently.
3. **Broadcast quality** -- broadcast packets carry semantically rich content
   that downstream modules can interpret regardless of source modality.

The ModalityAligner runs as a pre-processing step before workspace competition.
Encoder outputs pass through their respective projection heads to produce
aligned representations before entering the workspace.

## Binding Problem

Ensuring correct feature binding across modalities -- the right visual features
associate with the right text tokens -- requires explicit mechanisms:

1. **Temporal synchrony** -- features arriving at the same timestamp from
   different modalities are assumed to correspond.
2. **Spatial correspondence** -- for vision-text pairs, spatial attention maps
   indicate which image regions correspond to which text spans.
3. **Feature binding tags** -- optional learned binding vectors that modulate
   cross-modal attention based on semantic type.

## Configuration Surface

### AlignmentConfig

| Field | Default | Purpose |
|---|---|---|
| `alignment_dim` | 512 | Shared embedding dimension |
| `temperature` | 0.07 | Contrastive loss temperature |
| `learnable_temperature` | True | Make temperature a learnable parameter |
| `loss_type` | `"infonce"` | Loss: infonce, siglip, or combined |
| `projector_type` | `"mlp_1"` | Projector: linear, mlp_1, mlp_2 |
| `projector_dropout` | 0.1 | Dropout in projection heads |
| `use_uniformity_loss` | True | Add uniformity regularization |
| `uniformity_weight` | 0.1 | Weight for uniformity loss |
| `use_centering` | True | Subtract running modality centroids |
| `centering_momentum` | 0.9 | EMA momentum for centroid tracking |
| `gap_reg_weight` | 0.01 | Weight for modality gap regularization |
| `hard_negative_mining` | False | Enable hard negative selection |
| `curriculum_phases` | 3 | Number of curriculum difficulty phases |

Presets: `AlignmentConfig.minimal()`, `AlignmentConfig.dev()`,
`AlignmentConfig.production()`.

## Done-When Gates

| Gate | Criterion |
|---|---|
| Contrastive loss converges | Loss < 2.0 for B >= 64 after 1K steps |
| Modality gap < threshold | Mean centroid distance < 0.3 in shared space |
| Cross-modal retrieval R@1 | > 50% on validation pairs |
| Uniformity score | > -2.0 (log uniformity on hypersphere) |
| No representation collapse | Singular value ratio > 0.01 |
| Gradient flows end-to-end | Non-zero grad on all projection parameters |
| Temperature in valid range | 0.001 <= temperature <= 1.0 after training |
| Alignment metrics stable | Variance of metrics < 5% over 100 steps |

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| All embeddings collapse to a point | Temperature too low or no uniformity loss | Raise temperature, enable uniformity loss |
| Modality gap persists after training | No gap regularization or centering | Enable centering + gap_reg_weight |
| Cross-modal retrieval near random | Projection heads too small or mismatched | Use MLP projector, check D_align |
| Loss oscillates without converging | Learning rate too high for projectors | Reduce projector LR, use warmup |
| Temperature diverges | Unconstrained learnable temperature | Clamp temperature to [0.001, 1.0] |
| Gradient explosion in similarity matrix | Large batch + low temperature | Gradient clipping, increase temperature |

## Anti-Patterns

- **Sharing projectors across modalities** -- each modality needs its own head
- **Skipping L2 normalization** -- cosine similarity requires unit-norm vectors
- **Using MSE instead of contrastive loss** -- MSE collapses representations
- **Ignoring the modality gap** -- leads to unfair workspace competition
- **Training alignment separately from the pipeline** -- must backprop through workspace
- **Using fp16 for temperature** -- temperature needs fp32 precision
- **Batch size < 32 without memory bank** -- too few negatives for contrastive learning

## Additional Resources

### Reference Files

- **`references/contrastive-theory.md`** -- InfoNCE math, temperature tuning, CLIP/SigLIP variants, batch size effects
- **`references/projection-architectures.md`** -- Linear vs MLP projectors, normalization, dimension choices
- **`references/modality-gap.md`** -- Gap phenomenon, uniformity loss, centering, gap regularization
- **`references/binding-mechanisms.md`** -- Temporal binding, spatial correspondence, feature binding
- **`references/testing-matrix.md`** -- All test cases, pytest patterns, done-when checklist

### Asset Templates

- **`assets/modality_aligner_template.py`** -- ModalityAligner, ProjectionHead, AlignmentOutput, self-test
- **`assets/contrastive_losses_template.py`** -- InfoNCE, SigLIP, uniformity + alignment, combined losses, self-test
- **`assets/cross_modal_retriever_template.py`** -- CrossModalRetriever, retrieval evaluation, self-test
- **`assets/alignment_metrics_template.py`** -- AlignmentMetrics, gap measurement, uniformity, SVD health, self-test
- **`assets/alignment_config_template.py`** -- AlignmentConfig dataclass, validation, presets, serialization, self-test

### Scripts

- **`scripts/validate_alignment.py`** -- Runtime contract validation (projection shapes, loss values, metric ranges)
- **`scripts/gen_alignment_tests.py`** -- Generates `tests/test_alignment.py` (~70+ test cases)
- **`scripts/alignment_benchmark.py`** -- Performance benchmarking (throughput, memory, scaling)
