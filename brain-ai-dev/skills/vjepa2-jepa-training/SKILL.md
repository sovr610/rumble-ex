---
name: V-JEPA 2 Self-Supervised Training
description: >
  This skill should be used when the user asks to "train V-JEPA model",
  "implement JEPA pretext task", "set up EMA target encoder",
  "configure self-supervised training", "implement smooth L1 loss",
  "create training loop for V-JEPA", "optimizer configuration",
  "learning rate schedule", "warmup cosine decay", "EMA momentum schedule",
  "collapse prevention", "predictor architecture", "masked prediction loss",
  "DROID fine-tuning loop", "annealing phase", "cooldown training",
  or needs guidance on V-JEPA 2 self-supervised learning, training loops,
  predictor design, or optimization strategies.
version: 0.1.0
---

# V-JEPA 2 Self-Supervised Training

## Overview

Guide implementation of the V-JEPA 2 self-supervised learning pipeline. The JEPA (Joint Embedding Predictive Architecture) pretext task trains a model to predict masked video representations in latent space. Cover the encoder-predictor-target architecture, EMA target encoder, smooth L1 loss, training loop engineering, optimizer configuration, LR/WD scheduling, and the DROID fine-tuning variant with autoregressive rollout.

## Public Contract

### JEPATrainer

Main training orchestrator for V-JEPA 2 pretraining.

```python
class JEPATrainer:
    def __init__(self, encoder: VisionTransformer, predictor: VisionTransformerPredictor,
                 config: JEPATrainingConfig): ...
    def train_step(self, batch, masks_enc, masks_pred) -> Dict[str, float]: ...
    def update_ema(self, step: int) -> None: ...
    def save_checkpoint(self, path: str, epoch: int) -> None: ...
    def load_checkpoint(self, path: str) -> int: ...  # Returns epoch
```

### VisionTransformerPredictor

Predicts masked representations from context.

```python
class VisionTransformerPredictor(nn.Module):
    def __init__(self, embed_dim, predictor_embed_dim, depth, num_heads,
                 num_targets=10): ...
    def forward(self, context_repr: Tensor, masks_enc: List[Tensor],
                masks_pred: List[Tensor]) -> Tensor: ...
```

### EMAManager

Manages exponential moving average target encoder.

```python
class EMAManager:
    def __init__(self, encoder: nn.Module, ema_schedule: Tuple[float, float],
                 total_steps: int): ...
    def update(self, step: int) -> float: ...  # Returns current momentum
    def get_target_encoder(self) -> nn.Module: ...
```

### LRScheduler

Manual step-based learning rate scheduling.

```python
class LRScheduler:
    def __init__(self, optimizer: Optimizer, config: LRConfig): ...
    def step(self, step: int) -> float: ...  # Returns current LR
```

## Key Concepts

### JEPA Architecture

```
Visible patches -> Context Encoder -> context_repr -+
                                                     +-> Predictor -> pred_repr --+
Mask tokens -----> Mask Token Pool -> mask_tokens ---+                            | Loss
                                                                                  | (Smooth L1)
Full video ------> Target Encoder (EMA, no grad) -> target_repr -----------------+
```

1. Context Encoder processes only visible (unmasked) patches
2. Predictor takes context + mask token placeholders, predicts masked regions
3. Target Encoder (EMA copy) processes full video to produce targets
4. Loss: Smooth L1 between predictor output and EMA target for masked patches only

### Predictor Token Ordering

1. Input projection: `Linear(embed_dim, predictor_embed_dim)`
2. Insert learnable mask tokens at target positions
3. Concatenate context + target tokens
4. Sort by position index, process through transformer
5. Un-sort, extract predictions at target positions
6. Output projection: `Linear(predictor_embed_dim, embed_dim)`

### EMA Target Encoder

- Deep copy of context encoder; updated each step: `theta_target = m * theta_target + (1-m) * theta_encoder`
- Momentum follows cosine schedule: typically `[0.99925, 0.99925]`
- No gradients flow through target encoder
- Prevents collapse without contrastive loss

### Collapse Prevention

JEPA avoids representation collapse through:
- Asymmetric architecture (encoder + separate predictor)
- EMA target (slowly moving targets prevent trivial solutions)
- High masking ratios (forces meaningful prediction)
- No negative samples needed

### Optimizer Configuration

AdamW with 4 parameter groups:
1. Encoder weights (with weight decay)
2. Predictor weights (with weight decay)
3. Encoder biases/1D params (no weight decay)
4. Predictor biases/1D params (no weight decay)

### Learning Rate Schedules

| Schedule | Shape | Use Case |
|----------|-------|----------|
| Warmup + Cosine | Warmup -> cosine decay | Standard pretraining |
| Warmup + Stable + Decay | Warmup -> plateau -> linear decay | Cooldown/annealing |
| Linear Decay | Linear from ref_lr to final_lr | Anneal phase only |

### Progressive Training Strategy

1. **Pretrain**: 256px, 16 frames, full warmup + cosine LR
2. **Cooldown**: 384px, 64 frames, annealing mode, LR decays to near-zero
3. **Robotics post-training**: 256px, 8 frames, frozen encoder, deep frame-causal predictor

### DROID Fine-Tuning Differences

- Loads pretrained checkpoint (encoder + optional predictor)
- Autoregressive multi-step prediction with `auto_steps`
- Frame-causal prediction with block-causal attention mask
- Differential LR: scaled LR for encoder vs predictor
- Optional representation normalization
- No EMA target encoder — direct reconstruction loss

### Loss Function

Smooth L1 with configurable exponent (`loss_exp`, default 1.0):
- Applied only to predicted target token positions
- Optionally normalized representations for DROID training

## Configuration Surface

```python
@dataclass
class JEPATrainingConfig:
    # Architecture
    predictor_embed_dim: int = 384
    predictor_depth: int = 12
    predictor_num_heads: int = 12
    num_mask_tokens: int = 10
    # Optimization
    lr: float = 1e-3
    final_lr: float = 1e-6
    warmup_epochs: int = 40
    epochs: int = 300
    weight_decay: float = 0.04
    final_weight_decay: float = 0.4
    batch_size: int = 64
    use_bfloat16: bool = True
    # EMA
    ema_start: float = 0.99925
    ema_end: float = 0.99925
    # Loss
    loss_exp: float = 1.0
    normalize_reps: bool = False
    # Annealing
    is_anneal: bool = False
    anneal_ckpt: Optional[str] = None
    # Autoregressive (DROID)
    auto_steps: int = 0
    encoder_lr_scale: float = 1.0
```

## Done-When Gates

1. **JEPA Forward** — Full forward (encode -> predict -> target) produces valid loss on synthetic data; loss decreases over 100 steps.
2. **EMA Update** — Target encoder parameters differ from context encoder; momentum follows cosine schedule; `torch.allclose(target, expected)` within tolerance.
3. **Checkpoint Round-Trip** — Save + load checkpoint preserves encoder, predictor, target_encoder, optimizer, and scaler state; training resumes with identical loss.

## Failure Modes

| Mode | Symptom | Fix |
|------|---------|-----|
| Representation collapse | Loss goes to 0, features constant | Verify EMA is updating, masking ratio is high enough |
| NaN loss | Training diverges | Reduce LR, check grad scaler, enable SDPA |
| EMA not updating | Target = encoder always | Verify update() called each step with correct schedule |
| Checkpoint incompatible | Key mismatch on load | Strip `module.`/`backbone.` prefixes, use `strict=False` for RoPE |
| Annealing LR too high | Loss spikes at cooldown start | Verify anneal_ckpt LR matches starting LR |

## Resources

### Reference Files
- **`references/jepa-pretext-task.md`** — JEPA architecture, loss computation, collapse prevention theory
- **`references/predictor-architecture.md`** — Token ordering, mask token insertion, projection layers
- **`references/optimizer-scheduling.md`** — AdamW param groups, LR/WD schedules, warmup strategies
- **`references/progressive-training.md`** — Pretrain -> cooldown -> post-train pipeline, checkpoint handling
- **`references/testing-matrix.md`** — Test scenarios for training infrastructure

### Asset Files
- **`assets/jepa_trainer_template.py`** — JEPATrainer with full training step, self-tests
- **`assets/predictor_template.py`** — VisionTransformerPredictor with token ordering
- **`assets/ema_manager_template.py`** — EMAManager with cosine schedule
- **`assets/lr_scheduler_template.py`** — All LR/WD schedulers (warmup+cosine, linear decay, cosine WD)
- **`assets/training_config_template.py`** — JEPATrainingConfig with validation and presets

### Scripts
- **`scripts/validate_training.py`** — Validates done-when gates
- **`scripts/gen_training_tests.py`** — Generates 100+ pytest test cases
- **`scripts/training_benchmark.py`** — Step throughput and memory benchmarks
