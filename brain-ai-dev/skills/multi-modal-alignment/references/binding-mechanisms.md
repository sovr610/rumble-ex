# Binding Mechanisms for Cross-Modal Alignment

> Reference for the binding problem in multi-modal alignment: temporal synchrony,
> spatial correspondence, feature binding, and integration with the Global Workspace.
> Part of the multi-modal-alignment skill.

---

## 1. The Binding Problem

The binding problem in multi-modal processing asks: how does the system know
which features from modality A correspond to which features from modality B?
When a person sees a red ball and hears a bouncing sound, the brain must bind
the visual "red" and "round" features with the auditory "bounce" feature into
a unified percept.

In the brain_ai architecture, binding manifests at two levels:
1. **Token-level binding**: Which vision token (patch) corresponds to which
   text token (word)?
2. **Sample-level binding**: Which vision sample corresponds to which audio
   sample in a batch?

Contrastive alignment (CLIP-style) addresses sample-level binding. Token-level
binding requires additional mechanisms described in this document.

---

## 2. Temporal Synchrony

### 2.1 Principle

Features from different modalities that arrive at the same time are assumed to
correspond. This is the simplest binding mechanism and works well when data
streams are naturally synchronized (e.g., video frames with concurrent audio).

### 2.2 Implementation

```python
def temporal_bind(
    feats_a: Tensor,     # (B, T_a, D)
    feats_b: Tensor,     # (B, T_b, D)
    time_a: Tensor,      # (B, T_a) timestamps
    time_b: Tensor,      # (B, T_b) timestamps
    tolerance: float = 0.05,  # seconds
) -> Tensor:
    """Compute temporal binding weights between tokens."""
    # (B, T_a, 1) - (B, 1, T_b) -> (B, T_a, T_b)
    time_diff = (time_a.unsqueeze(-1) - time_b.unsqueeze(-2)).abs()
    binding_weights = torch.exp(-time_diff / tolerance)
    return binding_weights  # soft assignment matrix
```

### 2.3 Temporal Resolution

| Modality Pair | Natural Alignment | Binding Tolerance |
|---------------|-------------------|-------------------|
| Vision-Audio | ~30ms (lip sync) | 50ms |
| Vision-Text | Caption timestamp | 500ms |
| Audio-Text | Transcript word timing | 200ms |
| Vision-Sensors | Hardware sync | 10ms |

### 2.4 When Temporal Binding Fails

- **Asynchronous data**: Caption-image pairs have no temporal correspondence.
- **Variable latency**: Different encoders process at different speeds.
- **Abstract concepts**: "The economy is growing" has no temporal anchor.

For these cases, use semantic binding (Section 4) instead.

---

## 3. Spatial Correspondence

### 3.1 Principle

For modalities with spatial structure (primarily vision), binding can use
spatial location: the image region that a text phrase refers to.

### 3.2 Cross-Attention Spatial Binding

```python
class SpatialBinder(nn.Module):
    """Compute spatial correspondence via cross-attention."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, vision_tokens, text_tokens, vision_mask=None, text_mask=None):
        # Q from text, K/V from vision
        # -> which vision patches does each text token attend to?
        bound, attn_weights = self.cross_attn(
            query=text_tokens,
            key=vision_tokens,
            value=vision_tokens,
            key_padding_mask=~vision_mask if vision_mask is not None else None,
        )
        return bound, attn_weights  # attn_weights: (B, T_text, T_vision)
```

### 3.3 Spatial Binding Maps

The attention weights from cross-attention form a binding map:

```
binding_map[i, j] = attention(text_token_i, vision_patch_j)
```

High values indicate strong binding between text token i and vision patch j.
These maps can be:
- Visualized for interpretability (which word looks at which patch)
- Supervised with ground-truth bounding boxes
- Used as soft alignment targets for the contrastive loss

### 3.4 Grid-Based Binding

For vision encoders that produce grid features (H x W patches):

```python
def grid_bind(attn_weights, grid_h, grid_w):
    """Reshape attention weights to spatial grid."""
    B, T_text, T_vision = attn_weights.shape
    assert T_vision == grid_h * grid_w
    return attn_weights.view(B, T_text, grid_h, grid_w)
```

This enables spatial visualization and region-level alignment.

---

## 4. Feature Binding (Semantic)

### 4.1 Principle

When temporal and spatial signals are absent, binding must rely on semantic
similarity: features from different modalities are bound if they encode
similar semantic content.

### 4.2 Learned Binding Vectors

Add learnable binding tags that modulate cross-modal attention:

```python
class SemanticBinder(nn.Module):
    def __init__(self, dim, num_binding_types=16):
        super().__init__()
        self.binding_queries = nn.Parameter(torch.randn(num_binding_types, dim))
        self.type_proj_a = nn.Linear(dim, num_binding_types)
        self.type_proj_b = nn.Linear(dim, num_binding_types)

    def forward(self, feats_a, feats_b):
        # Soft assignment to binding types
        types_a = F.softmax(self.type_proj_a(feats_a), dim=-1)  # (B, T_a, K)
        types_b = F.softmax(self.type_proj_b(feats_b), dim=-1)  # (B, T_b, K)

        # Binding affinity: tokens bound if they share binding type
        binding = torch.bmm(types_a, types_b.transpose(-1, -2))  # (B, T_a, T_b)
        return binding
```

### 4.3 Binding Type Interpretation

The learned binding types can be interpreted as semantic categories:
- Entity binding (object + name)
- Attribute binding (visual property + descriptor)
- Action binding (motion + verb)
- Relation binding (spatial relation + preposition)

### 4.4 Binding Consistency Loss

Ensure that binding is consistent across modalities:

```
L_binding = ||binding_a2b - binding_b2a.T||_F^2
```

This symmetry constraint prevents the binding from being one-directional.

---

## 5. Binding for the Global Workspace

### 5.1 Pre-Competition Binding

Before tokens enter workspace competition, binding information enhances
the token representations:

```python
def pre_competition_bind(vision_tokens, text_tokens, binder):
    """Enhance tokens with cross-modal binding context."""
    # vision tokens enriched with text context
    vision_bound, v2t_weights = binder(vision_tokens, text_tokens)
    # text tokens enriched with vision context
    text_bound, t2v_weights = binder(text_tokens, vision_tokens)

    # Residual addition
    vision_enhanced = vision_tokens + vision_bound
    text_enhanced = text_tokens + text_bound

    return vision_enhanced, text_enhanced, v2t_weights, t2v_weights
```

### 5.2 Binding-Aware Competition

Workspace competition can use binding information to score tokens:

```
score_token = w_content * content
            + w_salience * salience
            + w_novelty * novelty
            + w_binding * binding_strength
```

Where `binding_strength` measures how well a token binds with tokens from
other modalities. Strongly bound tokens (those with clear cross-modal
correspondences) should be prioritized for workspace access.

### 5.3 Post-Competition Binding Verification

After competition selects workspace slots, verify that binding is preserved:

```python
def verify_binding(slots, binding_maps, slot_sources):
    """Check that selected slots maintain cross-modal binding."""
    for slot_a, slot_b in bound_pairs(slots, binding_maps, slot_sources):
        binding_score = cosine_similarity(slot_a, slot_b)
        if binding_score < threshold:
            warn(f"Binding broken: {slot_a.modality} <-> {slot_b.modality}")
```

---

## 6. Multi-Way Binding

### 6.1 Beyond Pairwise

Real-world scenes involve binding across more than two modalities simultaneously.
A person says "look at the red car" while pointing -- this requires binding:
- Vision (red car patch)
- Text (the phrase "red car")
- Audio (speech prosody indicating emphasis)
- Sensors (pointing gesture direction)

### 6.2 Joint Binding Matrix

Extend pairwise binding to an N-way tensor:

```python
def multiway_binding(features_dict, binders):
    """Compute binding across all modality pairs."""
    modalities = list(features_dict.keys())
    bindings = {}
    for i, m_a in enumerate(modalities):
        for j, m_b in enumerate(modalities):
            if i < j:
                bindings[(m_a, m_b)] = binders[(m_a, m_b)](
                    features_dict[m_a], features_dict[m_b]
                )
    return bindings
```

### 6.3 Binding Consensus

When multiple modality pairs provide binding information, use consensus:

```
consensus_binding(a, b) = mean(
    direct_binding(a, b),
    indirect_binding(a, c) @ indirect_binding(c, b)  for c in other_modalities
)
```

This transitive binding helps when direct cross-modal correspondence is weak
but an intermediate modality provides a bridge.

---

## 7. Binding Failure Modes

| Failure | Cause | Detection | Fix |
|---------|-------|-----------|-----|
| Feature mis-binding | Wrong token pairs linked | Low binding consistency | Binding consistency loss |
| Binding collapse | All tokens bind to same target | Uniform attention weights | Diversity regularization |
| Temporal drift | Timestamp misalignment grows | Increasing binding tolerance needed | Re-synchronize streams |
| Spatial ambiguity | Multiple patches match one word | Diffuse attention map | Hard attention or top-k |
| Missing modality | One modality absent | Binding maps have NaN | Default to self-binding |

---

## 8. Integration Checklist

- [ ] Temporal binding enabled for synchronized modality pairs
- [ ] Spatial binding enabled for vision-text pairs
- [ ] Semantic binding enabled as fallback for all pairs
- [ ] Binding maps logged for interpretability
- [ ] Binding consistency loss added to training objective
- [ ] Pre-competition binding enriches tokens before workspace
- [ ] Binding strength contributes to competition scoring
- [ ] Post-competition verification checks binding preservation
- [ ] Multi-way binding handles 3+ simultaneous modalities
- [ ] Missing modality case handled gracefully
