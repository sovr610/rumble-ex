# Action Anticipation — Reference

## Overview

Action anticipation predicts what will happen *before* it occurs. In the V-JEPA 2 context,
this is implemented on EPIC-Kitchens 100 by consuming visible frames and predicting the
future action (verb + noun + combined action) at a specified anticipation time.

## EPIC-Kitchens 100 Task

- **Dataset**: EPIC-Kitchens 100 (egocentric cooking videos)
- **Classes**: 97 verbs, 300 nouns, 3806 action pairs (verb, noun)
- **Metric**: Class-Mean Recall at 5 (R@5, i.e., top-5 recall averaged per class)
- **Model outputs**: 3 logit tensors — verb, noun, action

## Anticipation Parameters

```python
anticipation_time_sec: float = 1.0     # seconds before action starts to make prediction
anticipation_point: float = 0.5        # fraction of observed segment used as context
anticipation_duration: float = 0.5     # seconds of future to anticipate
```

The encoder sees only frames UP TO `anticipation_point`. The predictor then produces
latent representations for the anticipated frames. The classifier heads read from
the predictor output (not the encoder output directly).

## Architecture

```
visible frames
      |
   [Encoder]  <-- frozen or finetuned
      |
encoder tokens [B, N_ctx, D]
      |
   [Predictor]  <-- future-frame prediction network
      |
predicted tokens [B, N_fut, D]
      |
   [ActionAnticipationClassifier]
      |
      +--[AttentivePooler q=0]--> verb logits  [B, num_verbs]
      +--[AttentivePooler q=1]--> noun logits  [B, num_nouns]
      +--[AttentivePooler q=2]--> action logits [B, num_actions]
```

## ActionAnticipationClassifier

```python
class ActionAnticipationClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_verbs: int,
        num_nouns: int,
        num_actions: int,
        num_queries: int = 3,
    ):
        super().__init__()
        # 3 independent poolers sharing the same key/value sequence
        self.verb_pooler   = AttentivePooler(embed_dim, num_queries=1, num_heads=1)
        self.noun_pooler   = AttentivePooler(embed_dim, num_queries=1, num_heads=1)
        self.action_pooler = AttentivePooler(embed_dim, num_queries=1, num_heads=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.verb_head   = nn.Linear(embed_dim, num_verbs)
        self.noun_head   = nn.Linear(embed_dim, num_nouns)
        self.action_head = nn.Linear(embed_dim, num_actions)

    def forward(self, predictor_output: torch.Tensor):
        # predictor_output: [B, N, D]
        x = self.norm(predictor_output)
        verb_feat   = self.verb_pooler(x).squeeze(1)    # [B, D]
        noun_feat   = self.noun_pooler(x).squeeze(1)    # [B, D]
        action_feat = self.action_pooler(x).squeeze(1)  # [B, D]
        return (
            self.verb_head(verb_feat),     # [B, num_verbs]
            self.noun_head(noun_feat),     # [B, num_nouns]
            self.action_head(action_feat), # [B, num_actions]
        )
```

## Focal Loss

Focal Loss addresses the extreme class imbalance in action anticipation
(thousands of classes, very few examples each).

### Formula

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

where:
- `p_t = p` if the sample belongs to the positive class, else `p_t = 1 - p`
- `alpha_t = alpha` for positives, `1 - alpha` for negatives
- `gamma`: focusing parameter (typically 2.0); down-weights easy examples

### Implementation

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: [B, C] unnormalized logits
        # targets: [B] class indices
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")   # [B]
        p_t = torch.exp(-ce_loss)                                       # probability of correct class
        focal_weight = (1.0 - p_t) ** self.gamma
        # alpha_t: alpha for positives, (1-alpha) for negatives
        # For multi-class, use alpha uniformly (common simplification)
        loss = self.alpha * focal_weight * ce_loss
        return loss.mean()
```

### Why Focal Loss for Action Anticipation

- EPIC-Kitchens 100 has 97 verbs and 300 nouns but fewer than 10 examples
  per action pair on average.
- Without focal loss, the model collapses to predicting the most frequent class.
- gamma=2.0 effectively ignores samples classified with p_t > 0.9 (easy examples),
  focusing gradient updates on hard, rare samples.

## ClassMeanRecall Metric

Class-Mean Recall (also called mean per-class recall or balanced accuracy) computes
recall for each class independently, then averages across classes. This prevents
frequent classes from dominating the metric.

```python
class ClassMeanRecall:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.per_class_correct = torch.zeros(self.num_classes)
        self.per_class_total   = torch.zeros(self.num_classes)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        # predictions: [B] predicted class indices (top-k already applied upstream)
        # targets: [B] ground truth class indices
        for c in range(self.num_classes):
            mask = targets == c
            if mask.sum() > 0:
                self.per_class_correct[c] += (predictions[mask] == c).sum().float()
                self.per_class_total[c]   += mask.sum().float()

    def compute(self) -> float:
        # Avoid division by zero for unseen classes
        valid = self.per_class_total > 0
        recall_per_class = self.per_class_correct[valid] / self.per_class_total[valid]
        return recall_per_class.mean().item()

    def all_reduce(self) -> None:
        """Synchronize counters across distributed ranks."""
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.per_class_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.per_class_total,   op=dist.ReduceOp.SUM)
```

### Top-K Recall

For R@5, check whether the ground truth is within the top-5 predictions:

```python
def update_topk(self, logits: torch.Tensor, targets: torch.Tensor, k: int = 5):
    topk_preds = logits.topk(k, dim=-1).indices  # [B, k]
    for c in range(self.num_classes):
        mask = targets == c
        if mask.sum() > 0:
            in_topk = (topk_preds[mask] == c).any(dim=-1)
            self.per_class_correct[c] += in_topk.sum().float()
            self.per_class_total[c]   += mask.sum().float()
```

## Training Recipe

```python
loss_fn_verb   = FocalLoss(alpha=0.25, gamma=2.0)
loss_fn_noun   = FocalLoss(alpha=0.25, gamma=2.0)
loss_fn_action = FocalLoss(alpha=0.25, gamma=2.0)

for frames, verb_labels, noun_labels, action_labels in train_loader:
    with torch.no_grad():
        enc_out  = encoder(frames)
        pred_out = predictor(enc_out, future_mask_tokens)

    verb_logits, noun_logits, action_logits = classifier(pred_out)

    loss = (
        loss_fn_verb(verb_logits, verb_labels)
        + loss_fn_noun(noun_logits, noun_labels)
        + loss_fn_action(action_logits, action_labels)
    ) / 3.0

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Assessment Protocol

```python
metric_verb   = ClassMeanRecall(num_classes=97)
metric_noun   = ClassMeanRecall(num_classes=300)
metric_action = ClassMeanRecall(num_classes=3806)

classifier.train(False)
with torch.no_grad():
    for frames, verb_labels, noun_labels, action_labels in val_loader:
        enc_out  = encoder(frames)
        pred_out = predictor(enc_out, future_mask_tokens)
        verb_logits, noun_logits, action_logits = classifier(pred_out)
        metric_verb.update_topk(verb_logits, verb_labels, k=5)
        metric_noun.update_topk(noun_logits, noun_labels, k=5)
        metric_action.update_topk(action_logits, action_labels, k=5)

# For distributed:
metric_verb.all_reduce()
metric_noun.all_reduce()
metric_action.all_reduce()

print(f"Verb R@5:   {metric_verb.compute():.4f}")
print(f"Noun R@5:   {metric_noun.compute():.4f}")
print(f"Action R@5: {metric_action.compute():.4f}")
```

## Key Differences vs Standard Classification

| Aspect              | Video Classification | Action Anticipation |
|---------------------|---------------------|---------------------|
| Model inputs        | Encoder only         | Encoder + Predictor |
| Loss                | Cross-entropy        | Focal loss          |
| Output heads        | 1                    | 3 (verb/noun/action)|
| Metric              | Top-1 Accuracy       | R@5 ClassMeanRecall |
| Class balance       | Moderate             | Extreme imbalance   |
| Temporal structure  | Full clip            | Observed + future   |
