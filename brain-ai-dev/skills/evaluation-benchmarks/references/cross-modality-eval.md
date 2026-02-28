# Cross-Modality Evaluation Protocols

Multi-modal evaluation strategies for the brain_ai system, covering per-modality accuracy, fusion improvement measurement, ablation protocols, and multi-modal agreement metrics.

---

## 1. Overview

The brain_ai system integrates multiple modalities (vision, text, audio, sensors, engram) through a Global Workspace that competes for attention and produces a unified representation. Cross-modality evaluation measures how well this integration works: whether fusion helps, which modalities contribute most, and whether the system degrades gracefully when modalities are missing.

All encoders output to a unified workspace_dim (default 4096) representation. The Global Workspace uses attention-based competition to select and integrate the most relevant modality signals.

---

## 2. Per-Modality Accuracy

### Protocol

Run the full model with only one modality active at a time. All other modalities receive zero tensors of the correct shape.

```python
def per_modality_accuracy(model, dataset, modalities):
    """Run model with each modality isolated."""
    results = {}
    for target_mod in modalities:
        metrics = MetricsSuite("classify", num_classes)
        for batch in dataset:
            inputs = {}
            for mod in modalities:
                if mod == target_mod:
                    inputs[mod] = batch[mod]
                else:
                    inputs[mod] = torch.zeros_like(batch[mod])
            outputs = model(inputs)
            metrics.update(outputs, batch["targets"])
        results[target_mod] = metrics.compute()
    return results
```

### Expected Behavior

- **Vision**: Should carry most information for image classification tasks.
- **Text**: Should dominate for language understanding tasks.
- **Audio**: Should dominate for speech/sound tasks.
- **Sensors**: Should dominate for control/RL tasks.

### Reporting

```json
{
    "per_modality_accuracy": {
        "vision": 0.82,
        "text": 0.35,
        "audio": 0.28,
        "sensors": null
    },
    "best_single_modality": "vision",
    "best_single_accuracy": 0.82
}
```

---

## 3. Fusion Improvement Ratio (FIR)

### Definition

Measures how much multi-modal fusion improves over the best single modality.

```
FIR = (accuracy_fusion - accuracy_best_single) / accuracy_best_single
```

Or equivalently:

```
FIR = accuracy_fusion / accuracy_best_single - 1.0
```

### Interpretation

| FIR Value | Interpretation |
|-----------|---------------|
| > 0.10 | Strong fusion benefit (>10% improvement) |
| 0.01 - 0.10 | Moderate fusion benefit |
| -0.01 - 0.01 | No significant fusion effect |
| < -0.01 | Fusion hurts performance (integration problem) |

### Extended FIR Metrics

Beyond simple accuracy FIR, compute for all primary metrics:

```python
def compute_fir(fusion_metrics, per_modality_metrics, metric_name="accuracy"):
    best_single = max(
        mod_metrics[metric_name]
        for mod_metrics in per_modality_metrics.values()
        if mod_metrics[metric_name] is not None and not math.isnan(mod_metrics[metric_name])
    )
    if best_single == 0:
        return float("inf") if fusion_metrics[metric_name] > 0 else 0.0
    return fusion_metrics[metric_name] / best_single - 1.0
```

### FIR by Class

Compute FIR per class to identify which classes benefit from fusion:

```python
fir_per_class = {}
for cls_id in range(num_classes):
    fusion_f1 = fusion_per_class[cls_id]["f1"]
    best_single_f1 = max(
        mod_per_class[cls_id]["f1"]
        for mod_per_class in per_modality_per_class.values()
    )
    fir_per_class[cls_id] = fusion_f1 / max(best_single_f1, 1e-10) - 1.0
```

---

## 4. Ablation Protocol

### Single-Modality Ablation

Systematically remove one modality at a time and measure performance drop.

```python
def ablation_study(model, dataset, modalities):
    """Remove one modality at a time."""
    baseline = run_assessment(model, dataset, modalities)

    ablation_results = {}
    for removed_mod in modalities:
        remaining = [m for m in modalities if m != removed_mod]
        result = run_assessment(model, dataset, remaining, zero_removed=True)
        drop = baseline["accuracy"] - result["accuracy"]
        ablation_results[removed_mod] = {
            "removed": removed_mod,
            "remaining_accuracy": result["accuracy"],
            "accuracy_drop": drop,
            "relative_drop": drop / max(baseline["accuracy"], 1e-10),
            "metrics": result,
        }
    return ablation_results
```

### Interpretation of Ablation Results

- **High drop when removing modality X**: X is critical for the task.
- **No drop when removing modality X**: X is redundant or not used.
- **Performance improves when removing X**: X is adding noise, indicates integration problem.

### Progressive Ablation

Remove modalities in order of least to most important:

```python
def progressive_ablation(model, dataset, modalities):
    """Remove modalities one by one, least important first."""
    remaining = list(modalities)
    trajectory = [run_assessment(model, dataset, remaining)]

    while len(remaining) > 1:
        drops = {}
        for mod in remaining:
            subset = [m for m in remaining if m != mod]
            result = run_assessment(model, dataset, subset, zero_removed=True)
            drops[mod] = trajectory[-1]["accuracy"] - result["accuracy"]

        least_important = min(drops, key=drops.get)
        remaining.remove(least_important)
        trajectory.append({
            "removed": least_important,
            "remaining": list(remaining),
            **run_assessment(model, dataset, remaining, zero_removed=True),
        })

    return trajectory
```

### Zero-Input vs Absent-Input Ablation

Two strategies for disabling a modality:

1. **Zero-input** (recommended): Replace modality tensor with zeros. Tests the model's ability to cope with missing information while keeping the architecture intact.

2. **Absent-input**: Remove the modality key from the input dict entirely. Tests the model's handling of missing modality keys but changes the workspace competition dynamics.

```python
# Zero-input ablation
inputs = {mod: (batch[mod] if mod != removed else torch.zeros_like(batch[mod]))
          for mod in modalities}

# Absent-input ablation
inputs = {mod: batch[mod] for mod in modalities if mod != removed}
```

Report both if feasible, as they test different aspects of robustness.

---

## 5. Multi-Modal Agreement

### Definition

Measures how often different modalities agree on the prediction when assessed independently.

```
agreement(m1, m2) = sum(pred_m1[i] == pred_m2[i] for i in 1..N) / N
```

### Pairwise Agreement Matrix

For M modalities, compute an M x M agreement matrix:

```python
def compute_agreement_matrix(per_modality_predictions):
    """Compute pairwise prediction agreement."""
    modalities = list(per_modality_predictions.keys())
    M = len(modalities)
    agreement = torch.zeros(M, M)

    for i, mod_i in enumerate(modalities):
        for j, mod_j in enumerate(modalities):
            preds_i = per_modality_predictions[mod_i]
            preds_j = per_modality_predictions[mod_j]
            agreement[i, j] = (preds_i == preds_j).float().mean().item()

    return agreement, modalities
```

### Interpretation

- **High agreement (>0.8)**: Modalities are redundant or the task is easy.
- **Moderate agreement (0.5-0.8)**: Modalities provide complementary views.
- **Low agreement (<0.5)**: Modalities disagree significantly. Fusion should help if the disagreements are complementary.

### Agreement Conditioned on Correctness

Separate agreement into four categories:

| Both Correct | Both Wrong | m1 Right, m2 Wrong | m1 Wrong, m2 Right |
|---|---|---|---|
| Redundant signal | Shared failure mode | m1 carries unique info | m2 carries unique info |

```python
def conditioned_agreement(preds_m1, preds_m2, targets):
    both_correct = ((preds_m1 == targets) & (preds_m2 == targets)).float().mean()
    both_wrong = ((preds_m1 != targets) & (preds_m2 != targets)).float().mean()
    m1_only = ((preds_m1 == targets) & (preds_m2 != targets)).float().mean()
    m2_only = ((preds_m1 != targets) & (preds_m2 == targets)).float().mean()
    return {
        "both_correct": both_correct.item(),
        "both_wrong": both_wrong.item(),
        "m1_unique": m1_only.item(),
        "m2_unique": m2_only.item(),
    }
```

The "unique" categories are the most valuable for fusion: they represent samples where one modality succeeds and the other fails.

---

## 6. Workspace Attention Analysis

The Global Workspace uses attention to weight modality contributions. Analyzing attention patterns reveals integration dynamics.

### Attention Distribution per Sample

```python
def analyze_workspace_attention(model, dataset):
    """Collect and analyze workspace attention weights."""
    attention_stats = defaultdict(list)

    for batch in dataset:
        outputs = model(batch["inputs"], return_details=True)
        attention = outputs.attention

        for mod, weights in attention.items():
            attention_stats[mod].append(weights.mean().item())

    return {mod: {
        "mean": np.mean(vals),
        "std": np.std(vals),
        "min": np.min(vals),
        "max": np.max(vals),
    } for mod, vals in attention_stats.items()}
```

### Attention Entropy

Low entropy means the workspace is dominated by one modality. High entropy means even competition.

```python
def attention_entropy(attention_weights):
    """Compute entropy of attention distribution."""
    p = F.softmax(attention_weights, dim=-1)
    entropy = -(p * torch.log(p + 1e-10)).sum(dim=-1)
    return entropy.mean().item()
```

### Expected Behavior by Task

- **Vision classification**: Vision should dominate attention (low entropy).
- **VQA**: Vision and text should share attention (moderate entropy).
- **Multimodal sentiment**: All modalities should contribute (high entropy).

---

## 7. Cross-Modal Transfer Assessment

### Zero-Shot Modality Transfer

Check if training on one modality generalizes to another:

```python
def cross_modal_transfer(model, vision_data, text_data, shared_labels):
    """Check if vision knowledge transfers to text tasks and vice versa."""
    vision_trained = train(model, vision_data)
    text_zeroshot = assess(vision_trained, text_data)

    text_trained = train(model, text_data)
    vision_zeroshot = assess(text_trained, vision_data)

    return {
        "vision_to_text": text_zeroshot["accuracy"],
        "text_to_vision": vision_zeroshot["accuracy"],
    }
```

### Embedding Space Alignment

Measure how well modality embeddings align in the shared workspace:

```python
def embedding_alignment(encoded_vision, encoded_text, paired_indices):
    """Compute cosine similarity between paired vision/text embeddings."""
    v = encoded_vision[paired_indices[:, 0]]
    t = encoded_text[paired_indices[:, 1]]
    cos_sim = F.cosine_similarity(v, t, dim=-1)
    return {
        "mean_alignment": cos_sim.mean().item(),
        "std_alignment": cos_sim.std().item(),
    }
```

---

## 8. Complete Cross-Modality Report Schema

```json
{
    "cross_modality_results": {
        "fusion_accuracy": 0.89,
        "per_modality": {
            "vision": {"accuracy": 0.82, "f1_macro": 0.80},
            "text": {"accuracy": 0.35, "f1_macro": 0.33},
            "audio": {"accuracy": 0.28, "f1_macro": 0.25}
        },
        "fusion_improvement_ratio": {
            "accuracy": 0.085,
            "f1_macro": 0.075
        },
        "ablation": {
            "vision": {"accuracy_drop": 0.15, "relative_drop": 0.169},
            "text": {"accuracy_drop": 0.03, "relative_drop": 0.034},
            "audio": {"accuracy_drop": 0.01, "relative_drop": 0.011}
        },
        "agreement_matrix": {
            "modalities": ["vision", "text", "audio"],
            "matrix": [[1.0, 0.55, 0.48], [0.55, 1.0, 0.42], [0.48, 0.42, 1.0]]
        },
        "conditioned_agreement": {
            "vision_text": {
                "both_correct": 0.30,
                "both_wrong": 0.10,
                "vision_unique": 0.52,
                "text_unique": 0.08
            }
        },
        "workspace_attention": {
            "vision": {"mean": 0.65, "std": 0.12},
            "text": {"mean": 0.25, "std": 0.08},
            "audio": {"mean": 0.10, "std": 0.05}
        },
        "attention_entropy": 0.82
    }
}
```

---

## 9. Implementation Checklist

- [ ] Per-modality assessment uses zero-input ablation.
- [ ] FIR computed for all primary metrics, not just accuracy.
- [ ] Agreement matrix is symmetric (verified by assertion).
- [ ] Conditioned agreement fractions sum to 1.0 per pair.
- [ ] Workspace attention analyzed per sample, then aggregated.
- [ ] Cross-modality report includes all sections above.
- [ ] Edge case: single modality system (FIR = 0.0, agreement = 1.0).
- [ ] Edge case: modality with all-zero outputs (flag as "inactive").
- [ ] Deterministic assessment with fixed seeds.
- [ ] All computations in no-grad mode to save memory.

---

## 10. Anti-Patterns

- **Comparing fusion to average single-modality**: Use the best single modality, not the average. Average understates the baseline.
- **Ignoring per-class FIR**: Fusion may help some classes while hurting others. Always report per-class breakdown.
- **Testing fusion with synthetic independent modalities**: Real modalities share information. Synthetic data may overestimate fusion benefit.
- **Not controlling for increased model capacity**: The fusion model has more parameters. Compare against a single-modality model with equivalent capacity.
- **Assessing with mismatched modality pairs**: Ensure vision and text refer to the same samples. Misalignment produces meaningless agreement metrics.
