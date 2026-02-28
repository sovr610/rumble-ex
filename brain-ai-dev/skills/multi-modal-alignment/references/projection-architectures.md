# Projection Head Architectures

> Reference for projection head design: linear vs MLP projectors, normalization
> strategies, dimension choices, modality-specific vs shared architectures.
> Part of the multi-modal-alignment skill.

---

## 1. Overview

Projection heads transform encoder-specific representations into a shared
alignment space. The projector is the most critical architectural choice in
multi-modal alignment because it determines how much of the encoder's learned
features are preserved vs discarded in the aligned representation.

Key insight from SimCLR (Chen et al., 2020): projection heads act as
information bottlenecks. The encoder learns rich features; the projector
selects which features are relevant for cross-modal comparison. After training,
the projector can be discarded -- the encoder features (pre-projection) are
often more useful for downstream tasks.

---

## 2. Architecture Variants

### 2.1 Linear Projector

```python
class LinearProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)
```

**Pros**: Fewest parameters, fastest, minimal overhead.
**Cons**: Limited capacity to transform distributions; encoder must learn
alignment-friendly features natively.

**When to use**: Fine-tuning pre-trained encoders where features are already
high-quality. CLIP uses linear projectors.

### 2.2 MLP-1 Projector (Recommended Default)

```python
class MLP1Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.drop(self.act(self.norm(self.fc1(x))))
        return F.normalize(self.fc2(h), dim=-1)
```

**Pros**: Non-linear transformation allows distribution reshaping; LayerNorm
stabilizes training; good accuracy-to-cost ratio.
**Cons**: More parameters than linear; requires tuning hidden_dim.

**When to use**: Default choice. Works well for most encoder-alignment
configurations.

### 2.3 MLP-2 Projector

```python
class MLP2Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.drop1(self.act1(self.norm1(self.fc1(x))))
        h = self.drop2(self.act2(self.norm2(self.fc2(h))))
        return F.normalize(self.fc3(h), dim=-1)
```

**Pros**: Highest capacity; can learn complex distribution transformations.
**Cons**: Most parameters; risk of overfitting on small datasets; slower.

**When to use**: Large-scale models with abundant paired training data.

---

## 3. Normalization

### 3.1 L2 Normalization (Output)

All projectors must L2-normalize their output. This ensures:
- Cosine similarity is equivalent to dot product
- Representations lie on the unit hypersphere
- Temperature scaling behaves predictably
- Gradient magnitudes are bounded

```python
z = F.normalize(proj(x), p=2, dim=-1)
```

### 3.2 LayerNorm (Internal)

LayerNorm after each linear layer (before activation) stabilizes training:
- Prevents activation drift across training
- Makes the projector robust to encoder output scale changes
- Particularly important when the encoder is frozen

### 3.3 BatchNorm vs LayerNorm

| Property | BatchNorm | LayerNorm |
|----------|-----------|-----------|
| Dependency | Across batch | Within sample |
| Small batch | Unstable statistics | Stable |
| Distributed | Requires sync | Independent |
| Variable sequence length | Problematic | Natural |

**Recommendation**: Always use LayerNorm in projection heads. BatchNorm
introduces unwanted batch dependencies that interfere with contrastive
learning (batch statistics leak information about negative pairs).

---

## 4. Dimension Choices

### 4.1 Input Dimension (D_enc)

Determined by the encoder architecture. Common values:

| Encoder | D_enc |
|---------|-------|
| ViT-B | 768 |
| ViT-L | 1024 |
| ViT-H | 1280 |
| GPT-2 small | 768 |
| GPT-2 medium | 1024 |
| Audio (mel + conv) | 256 -- 512 |
| Sensor (CfC) | 128 -- 512 |
| brain_ai workspace | 512 -- 4096 |

### 4.2 Hidden Dimension (D_hidden)

For MLP projectors, the hidden dimension controls capacity:

```
D_hidden = max(D_enc, D_align) * expansion_factor
```

Typical expansion_factor: 2.0 to 4.0. Using a hidden dimension larger than
both input and output creates an information bottleneck in the right direction
(expand then compress).

### 4.3 Output Dimension (D_align)

The shared alignment dimension. Key considerations:

| D_align | Trade-off |
|---------|-----------|
| 128 | Compact, fast retrieval, some information loss |
| 256 | Good default for medium-scale |
| 512 | CLIP default, rich representations |
| 768 | High-capacity, slower retrieval |
| 1024 | Diminishing returns for most tasks |

Rule of thumb: D_align should be at least half of min(D_enc_a, D_enc_b).
Smaller values act as stronger information bottlenecks.

---

## 5. Modality-Specific Design

### 5.1 Why Separate Projectors

Each modality's encoder produces features with different:
- **Mean and variance**: Vision patch features vs text token embeddings
- **Dimensionality**: D_enc may differ across encoders
- **Sparsity patterns**: Audio features are often sparse; text is dense
- **Temporal structure**: Some modalities have strong temporal correlations

A shared projector cannot accommodate these differences. Always use
modality-specific projection heads.

### 5.2 Projector Registry Pattern

```python
class ProjectorRegistry:
    """Maps modality names to their projection heads."""

    def __init__(self):
        self.projectors = nn.ModuleDict()

    def register(self, modality: str, input_dim: int, config):
        self.projectors[modality] = build_projector(
            input_dim=input_dim,
            output_dim=config.alignment_dim,
            projector_type=config.projector_type,
            dropout=config.projector_dropout,
        )

    def project(self, modality: str, features: Tensor) -> Tensor:
        return self.projectors[modality](features)
```

### 5.3 Parameter Counts

| Config | Projector Type | D_enc | D_align | Params/Projector |
|--------|---------------|-------|---------|-----------------|
| Minimal | Linear | 128 | 128 | 16K |
| Dev | MLP-1 | 512 | 256 | 400K |
| Production | MLP-1 | 4096 | 512 | 12M |
| Large | MLP-2 | 4096 | 512 | 50M |

With 5 modalities, multiply by 5 for total projector parameters.

---

## 6. Pooling Before Projection

Encoders produce per-token features `(B, T, D_enc)` but contrastive alignment
operates on single vectors `(B, D_align)`. Pooling strategies:

### 6.1 Mean Pooling (Default)

```python
def mean_pool(feats, mask):
    """Masked mean pooling over the token dimension."""
    feats = feats * mask.unsqueeze(-1).float()
    return feats.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
```

Robust, simple, works well in practice.

### 6.2 CLS/EOS Token

Use a designated token's embedding (CLS for vision, EOS for text). Requires
encoder-specific logic.

### 6.3 Attention Pooling

Learned weighted combination of token embeddings:

```python
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads=1, batch_first=True)

    def forward(self, feats, mask=None):
        query = self.query.expand(feats.shape[0], -1, -1)
        out, _ = self.attn(query, feats, feats, key_padding_mask=~mask)
        return out.squeeze(1)
```

More expressive than mean pooling but adds parameters.

---

## 7. Initialization

### 7.1 Standard Init

Xavier uniform for linear layers, zeros for biases. This is the PyTorch default
and works well for most cases.

### 7.2 Zero-Init Output Layer

Initialize the last linear layer with zeros so the projector starts as
(approximately) zero-output. Combined with a residual connection, this
ensures the projector starts as identity-like:

```python
nn.init.zeros_(self.fc2.weight)
nn.init.zeros_(self.fc2.bias)
```

### 7.3 Shared Initialization (Gap Reduction)

Initialize all modality projectors from the same random weights. This gives
all modalities the same starting point in the shared space, reducing the
initial modality gap:

```python
template_state = build_projector(...).state_dict()
for modality in modalities:
    projectors[modality].load_state_dict(template_state)
```

Note: This only helps at initialization. Training will diverge the projectors
as they adapt to modality-specific input distributions.

---

## 8. Integration with brain_ai

### 8.1 Projector Placement

```
Encoder -> EncoderOutput(feats=(B,T,D_enc)) -> Pool -> ProjectionHead -> z=(B,D_align)
                                                                            |
                                                                    ModalityAligner
                                                                            |
                                                              Workspace Competition
```

### 8.2 Frozen vs Trainable Encoders

| Encoder State | Projector Role | Recommendation |
|--------------|----------------|----------------|
| Frozen | Full alignment burden | MLP-1 or MLP-2 |
| Fine-tuning | Partial alignment | Linear or MLP-1 |
| Training from scratch | Co-learning | MLP-1 |

### 8.3 Gradient Flow

Projector gradients flow back through the pooling layer to the encoder.
When the encoder is frozen, only projector parameters update. When the
encoder is trainable, alignment gradients influence encoder feature learning.

This dual optimization (alignment + encoder task) requires careful learning
rate balancing: projector LR should be 2--10x the encoder LR.
