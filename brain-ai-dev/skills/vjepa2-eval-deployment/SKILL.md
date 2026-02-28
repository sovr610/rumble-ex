---
name: V-JEPA 2 Evaluation & Deployment
description: >
  This skill should be used when the user asks to "run V-JEPA evaluation",
  "frozen backbone probing", "attentive classifier", "video classification",
  "image classification", "action anticipation", "EPIC-Kitchens assessment",
  "multi-head hyperparameter search", "focal loss implementation",
  "class-mean recall metric", "PyTorch Hub integration", "HuggingFace deployment",
  "model factory functions", "preprocessor pipeline",
  "testing V-JEPA", "unit tests for ViT", "codebase conventions",
  or needs guidance on assessment pipelines, frozen probing,
  model deployment, testing strategies, or V-JEPA 2 codebase patterns.
version: 0.1.0
---

# V-JEPA 2 Evaluation & Deployment

## Overview

Guide implementation of assessment pipelines, model deployment, and testing for V-JEPA 2. Cover frozen backbone assessment (video classification, image classification, action anticipation), the attentive pooler/classifier architecture, multi-head hyperparameter search, focal loss, class-mean recall metrics, PyTorch Hub and HuggingFace deployment, model factory functions, testing patterns, and codebase conventions.

## Public Contract

### AttentiveClassifier

Frozen backbone classifier using attentive pooling.

```python
class AttentiveClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, num_queries: int = 1,
                 depth: int = 1, num_heads: int = 1): ...
    def forward(self, encoder_output: Tensor) -> Tensor: ...
```

### FrozenBackboneAssessor

Multi-head frozen backbone assessment pipeline.

```python
class FrozenBackboneAssessor:
    def __init__(self, encoder: nn.Module, config: AssessmentConfig): ...
    def train_probes(self, train_loader: DataLoader, multihead_kwargs: List[Dict]) -> None: ...
    def run_validation(self, val_loader: DataLoader) -> Dict[str, float]: ...
    def best_head(self) -> Tuple[int, Dict[str, float]]: ...
```

### ActionAnticipationClassifier

Multi-output classifier for verb/noun/action prediction.

```python
class ActionAnticipationClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_verbs: int, num_nouns: int,
                 num_actions: int, num_queries: int = 3): ...
    def forward(self, encoder_output: Tensor) -> Tuple[Tensor, Tensor, Tensor]: ...
```

### FocalLoss

Class-imbalance-aware loss for action anticipation.

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0): ...
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor: ...
```

### ClassMeanRecall

Per-class recall metric with distributed support.

```python
class ClassMeanRecall:
    def __init__(self, num_classes: int): ...
    def update(self, predictions: Tensor, targets: Tensor) -> None: ...
    def compute(self) -> float: ...
    def all_reduce(self) -> None: ...  # Sync across ranks
```

### ModelHub

PyTorch Hub and HuggingFace model loading.

```python
def vjepa2_vit_large(pretrained=True) -> Tuple[nn.Module, nn.Module]: ...
def vjepa2_vit_giant(pretrained=True) -> Tuple[nn.Module, nn.Module]: ...
def vjepa2_ac_vit_giant(pretrained=True) -> Tuple[nn.Module, nn.Module]: ...
def vjepa2_preprocessor() -> Callable: ...
```

## Key Concepts

### Frozen Backbone Assessment Pattern

1. Load pretrained encoder (frozen, `requires_grad=False`, inference mode)
2. Create trainable probe/classifier head (AttentiveClassifier)
3. Train only the head while encoder stays frozen
4. Multi-head: train N classifiers simultaneously with different LR/WD combinations
5. Select best head based on validation performance

### Assessment Tasks

| Task | Dataset | Metric | Special Notes |
|------|---------|--------|---------------|
| Video Classification | SSv2, Diving48, K400 | Top-1 Accuracy | Multi-segment x multi-view |
| Image Classification | ImageNet-1K | Top-1 Accuracy | Uses timm transforms |
| Action Anticipation | EPIC-Kitchens 100 | R@5 (ClassMeanRecall) | Focal loss, 3 output heads |

### Multi-Segment Assessment

At test time: `num_segments * num_views_per_segment` clips per video.
Final prediction: average logits across all clips.

### Action Anticipation Details

- 3 query tokens -> verb logits, noun logits, action logits
- Focal loss: `FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)`
- Uses both encoder AND predictor (future-frame prediction)
- `anticipation_time_sec`, `anticipation_point`, `anticipation_duration` parameters

### PyTorch Hub Entry Points

```python
# Hub usage
model = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')
# Dependencies: torch, timm, einops
```

Factory: Creates encoder + predictor, loads pretrained weights from URL.
Default predictor: depth=12, embed_dim=384, 12 heads, 10 mask tokens.
`strict=False` for RoPE compatibility.

### HuggingFace Integration

```python
from transformers import AutoModel, AutoVideoProcessor
model = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
```

### Testing Patterns

- **unittest** framework with GPU-conditional tests (`@skipIf(not torch.cuda.is_available())`)
- Shape verification: `assertEqual(output.shape, expected_shape)`
- Numerical equivalence: `torch.testing.assert_close(a, b)`
- Cross-format consistency: tensor vs numpy paths
- Cosine similarity for non-square inputs

### Codebase Conventions

- **Black** (line-length=119), **isort** (profile=black)
- Config-driven: all hyperparameters from YAML, never hardcoded
- Plugin architecture: `importlib.import_module` for dynamic dispatch
- Wrapper pattern: multi-sequence, monitored dataset, DDP wrappers
- MIT license header on every source file

## Configuration Surface

```python
@dataclass
class AssessmentConfig:
    task: str = "video_classification"
    num_classes: int = 174               # SSv2
    num_queries: int = 1
    probe_depth: int = 1
    multihead_kwargs: List[Dict] = ()    # [{lr, wd}, {lr, wd}, ...]
    num_segments: int = 1
    num_views: int = 3
    val_only: bool = False
    # Action anticipation
    num_verbs: int = 97
    num_nouns: int = 300
    num_actions: int = 3806
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
```

## Done-When Gates

1. **Frozen Probing** — `FrozenBackboneAssessor` trains probe on synthetic data; encoder gradients remain zero; probe loss decreases.
2. **Focal Loss** — `FocalLoss` matches expected values on synthetic logits/targets with known ground truth; handles class imbalance.
3. **Hub Loading** — Model factory function creates valid encoder+predictor; forward pass produces correct output shape.

## Resources

### Reference Files
- **`references/frozen-probing.md`** — Attentive pooler, multi-head search, segment/view assessment
- **`references/action-anticipation.md`** — EPIC-Kitchens, focal loss, class-mean recall, multi-output
- **`references/model-hub.md`** — PyTorch Hub, HuggingFace, factory functions, preprocessor
- **`references/testing-patterns.md`** — unittest conventions, GPU tests, shape/numerical checks
- **`references/testing-matrix.md`** — Test scenarios

### Asset Files
- **`assets/attentive_classifier_template.py`** — AttentiveClassifier, AttentivePooler, self-tests
- **`assets/frozen_assessor_template.py`** — FrozenBackboneAssessor with multi-head search
- **`assets/focal_loss_template.py`** — FocalLoss, ClassMeanRecall, ActionAnticipationClassifier
- **`assets/model_hub_template.py`** — Factory functions, preprocessor, hub entry points
- **`assets/assessment_config_template.py`** — AssessmentConfig with task-specific presets

### Scripts
- **`scripts/validate_assessment.py`** — Validates done-when gates
- **`scripts/gen_assessment_tests.py`** — Generates 100+ pytest test cases
- **`scripts/assessment_benchmark.py`** — Assessment throughput benchmarks
