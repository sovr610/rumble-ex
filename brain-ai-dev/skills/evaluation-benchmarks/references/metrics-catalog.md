# Metrics Catalog

Complete metric definitions, formulas, edge cases, and implementation notes for the brain_ai evaluation infrastructure. All metrics are computed on CPU to avoid GPU memory pressure during evaluation.

---

## 1. Classification Metrics

### 1.1 Accuracy

**Definition**: Fraction of predictions that exactly match the ground truth.

```
accuracy = (number of correct predictions) / (total predictions)
         = sum(pred_i == target_i for i in 1..N) / N
```

**Range**: [0.0, 1.0]

**Edge Cases**:
- Empty batch (N=0): Return 0.0 (not NaN). Log a warning.
- All same predictions: Accuracy is valid but misleading. Always report alongside per-class metrics.
- Multi-label: Use per-label accuracy, then average (see micro/macro below).

**Implementation Notes**:
- Accumulate `correct_count` and `total_count` across batches.
- Use `torch.argmax(predictions, dim=-1)` for logit inputs.
- For probabilities, threshold at 0.5 for binary; argmax for multi-class.

### 1.2 Top-K Accuracy

**Definition**: Fraction of samples where the true label is among the top K predictions.

```
top_k_accuracy = sum(target_i in topk(pred_i, k) for i in 1..N) / N
```

**Typical values**: k=5 for ImageNet-scale (top-5 accuracy).

**Edge Cases**:
- k >= num_classes: Always 1.0 (degenerate). Warn if k >= num_classes.
- k = 1: Equivalent to standard accuracy.
- Tied scores in top-k boundary: Use stable sort (include all ties).

### 1.3 Precision

**Definition**: Of all samples predicted as class c, how many are truly class c.

```
precision_c = TP_c / (TP_c + FP_c)
```

Where:
- TP_c = true positives for class c
- FP_c = false positives for class c

**Edge Cases**:
- TP_c + FP_c = 0 (class never predicted): precision_c = 0.0. Set a `zero_division` flag.
- Binary case: precision for the positive class only (unless macro is requested).

### 1.4 Recall (Sensitivity)

**Definition**: Of all samples truly belonging to class c, how many were predicted as class c.

```
recall_c = TP_c / (TP_c + FN_c)
```

Where:
- FN_c = false negatives for class c

**Edge Cases**:
- TP_c + FN_c = 0 (class absent from targets): recall_c = 0.0. Set a `zero_division` flag.
- Perfect recall (1.0) with low precision indicates over-prediction.

### 1.5 F1 Score

**Definition**: Harmonic mean of precision and recall.

```
F1_c = 2 * (precision_c * recall_c) / (precision_c + recall_c)
```

**Averaging Modes**:

#### Macro F1
Unweighted mean across classes. Treats all classes equally.
```
F1_macro = (1/C) * sum(F1_c for c in 1..C)
```
Best when class balance matters more than sample count.

#### Weighted F1
Weighted by class support (number of true instances per class).
```
F1_weighted = sum(support_c * F1_c for c in 1..C) / sum(support_c)
```
Accounts for class imbalance.

#### Micro F1
Compute precision and recall globally, then F1.
```
precision_micro = sum(TP_c) / sum(TP_c + FP_c)
recall_micro = sum(TP_c) / sum(TP_c + FN_c)
F1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)
```
In multi-class single-label: micro F1 = accuracy.

**Edge Cases**:
- precision + recall = 0: F1 = 0.0 (avoid division by zero).
- Single class with no predictions: F1 for that class is 0.0; macro F1 decreases.
- All classes have zero support: Return 0.0 for all averaging modes.

### 1.6 AUROC (Area Under ROC Curve)

**Definition**: Probability that a randomly chosen positive is ranked higher than a randomly chosen negative. Computed from the Receiver Operating Characteristic curve.

```
AUROC = integral(TPR d(FPR))
```

Where TPR = true positive rate, FPR = false positive rate at varying thresholds.

**Multi-class Extension**: One-vs-rest (OVR) AUROC per class, then average.
```
AUROC_macro = (1/C) * sum(AUROC_c for c in 1..C)
AUROC_weighted = sum(support_c * AUROC_c) / sum(support_c)
```

**Computation** (trapezoidal rule):
1. Sort samples by descending predicted probability for class c.
2. Walk through sorted list, computing cumulative TP and FP rates.
3. Integrate using trapezoidal rule.

**Edge Cases**:
- Only one class present in targets: AUROC is undefined. Return NaN with warning.
- Perfect separation: AUROC = 1.0.
- Random predictions: AUROC approximately 0.5.
- Binary classification: Use probabilities for positive class directly.
- All predictions identical: AUROC = 0.5 (random ranking).

**NaN Handling**: When a class has zero positive or zero negative samples, AUROC for that class is NaN. Exclude NaN classes from macro average and report the count of excluded classes.

### 1.7 Confusion Matrix

**Definition**: C x C matrix where entry (i, j) counts samples with true label i predicted as label j.

```
CM[i][j] = count of samples where target = i and prediction = j
```

**Properties**:
- Row sums = class supports: `sum(CM[i,:]) = support_i`
- Column sums = prediction counts: `sum(CM[:,j]) = predicted_j`
- Diagonal = correct predictions: `CM[i][i] = TP_i`
- Off-diagonal reveals confusion patterns.

**Normalization Options**:
- `none`: Raw counts (default).
- `true`: Normalize by true labels (rows sum to 1). Shows recall per class.
- `pred`: Normalize by predictions (columns sum to 1). Shows precision per class.
- `all`: Normalize by total count. Shows joint probability.

**Edge Cases**:
- Classes with zero samples: Row of zeros. Still include in matrix.
- Predictions for unseen classes: Column may have nonzero values.
- Large num_classes (>1000): Store as sparse matrix or only report top-N confused pairs.

---

## 2. Few-Shot Learning Metrics

### 2.1 N-way K-shot Accuracy

**Definition**: Classification accuracy on query samples after adapting on K support samples per class in N-way episodes.

```
episode_accuracy = correct_queries / total_queries
mean_accuracy = mean(episode_accuracy for e in 1..E)
```

**95% Confidence Interval**:
```
CI_95 = 1.96 * std(episode_accuracies) / sqrt(E)
reported as: mean_accuracy +/- CI_95
```

**Standard Protocols**:
- 5-way 1-shot: N=5, K=1, query=15. Typical for mini-ImageNet.
- 5-way 5-shot: N=5, K=5, query=15.
- 20-way 1-shot: N=20, K=1, query=5. Harder setting (Omniglot).

**Edge Cases**:
- E < 30: CI is unreliable. Recommend E >= 600 for stable estimates.
- K = 0: Zero-shot evaluation (no support). Use a different protocol.
- Adaptation fails (all same prediction): Accuracy = 1/N (random). Flag as "collapsed".

### 2.2 Episode Sampling

Each episode:
1. Sample N classes from the dataset.
2. For each class, sample K support + Q query samples (disjoint).
3. Adapt model on support set.
4. Evaluate on query set.

**Reproducibility**: Use a fixed seed per episode index for deterministic sampling. Store the seed in the report.

---

## 3. Anomaly Detection Metrics

### 3.1 Standard Anomaly Metrics

Computed at a fixed threshold (or best threshold from sweep):

- **Precision**: TP / (TP + FP) -- How many flagged anomalies are real.
- **Recall**: TP / (TP + FN) -- How many real anomalies are caught.
- **F1**: Harmonic mean of precision and recall.

### 3.2 NAB Score (Numenta Anomaly Benchmark)

**Definition**: Reward-based scoring that values early detection and penalizes late/false detections.

```
NAB_score = sum(sigma(y_t) for t in anomaly_windows) / max_possible_score * 100
```

**Scoring Function**:
For each true anomaly window [t_start, t_end]:
- First detection within the window gets a reward based on position:
  ```
  reward(t) = 2 * sigmoid(-5 * (t - t_start) / window_length) - 1
  ```
  Earlier detections get higher reward (up to +1.0 at the start).
- No detection in window: penalty of -1.0.

For false positive detections (outside any anomaly window):
- Each false positive: penalty of -0.11 (standard NAB profile).

**Profiles**:
- Standard: FP_weight = -0.11, FN_weight = -1.0
- Reward low FP: FP_weight = -0.22, FN_weight = -1.0
- Reward low FN: FP_weight = -0.11, FN_weight = -2.0

**Edge Cases**:
- No anomalies in data: NAB score is undefined (only FP penalties). Return NaN.
- Multiple detections in same window: Only the first counts.
- Overlapping windows: Assign detection to the nearest window start.

### 3.3 Threshold Sweep

Evaluate anomaly metrics across a range of thresholds to find the optimal operating point.

```
thresholds = linspace(min_score, max_score, num_steps)
for each threshold:
    compute precision, recall, F1, NAB score
best_threshold = argmax(F1) or argmax(NAB)
```

Report: Best threshold, corresponding metrics, full curve data.

---

## 4. Reasoning Metrics

### 4.1 Exact Match

**Definition**: Binary score -- prediction exactly equals the expected answer.

```
exact_match = 1 if normalize(prediction) == normalize(target) else 0
EM_rate = sum(exact_match_i) / N
```

**Normalization**: Strip whitespace, lowercase, remove articles ("a", "an", "the"), remove punctuation.

**Edge Cases**:
- Empty prediction: exact_match = 0.
- Multiple valid answers: Match against any.
- Numeric answers: Compare as floats with tolerance (1e-6 relative).

### 4.2 Logical Consistency Score

**Definition**: Fraction of derived conclusions that are logically consistent with the premises and rules.

```
consistency = (consistent_derivations) / (total_derivations)
```

**Evaluation**:
1. Present premises and rules.
2. Ask for N conclusions.
3. Check each conclusion against a truth table or symbolic solver.

**Edge Cases**:
- Model produces no conclusions: score = 0.
- Contradictory conclusions (A and not-A): Both are inconsistent, score heavily penalized.

### 4.3 Proof Accuracy

**Definition**: For multi-step reasoning (ProofWriter, FOLIO), score based on correctness of each step.

```
proof_accuracy = (correct_steps) / (total_steps)
```

**Partial credit**: If the final answer is correct but intermediate steps are wrong, report separately.

---

## 5. Continual Learning Metrics

### 5.1 Backward Transfer (BWT)

**Definition**: Average influence that learning task t has on performance of previously learned tasks.

```
BWT = (1 / (T-1)) * sum(R_{T,j} - R_{j,j} for j in 1..T-1)
```

Where R_{i,j} is the accuracy on task j after training on task i.

**Interpretation**:
- BWT < 0: Catastrophic forgetting (learning new tasks hurts old ones).
- BWT = 0: No interference.
- BWT > 0: Positive backward transfer (rare, indicates synergy).

### 5.2 Forward Transfer (FWT)

**Definition**: Average influence that learning task t has on performance of future tasks (zero-shot on unseen tasks).

```
FWT = (1 / (T-1)) * sum(R_{i-1,i} - R_0_i for i in 2..T)
```

Where R_0_i is the performance on task i before any training (random baseline).

### 5.3 Average Accuracy

```
avg_accuracy = (1/T) * sum(R_{T,i} for i in 1..T)
```

Performance on all tasks after all training.

### 5.4 Forgetting Measure

```
forgetting_j = max(R_{i,j} for i in 1..T-1) - R_{T,j}
avg_forgetting = (1/(T-1)) * sum(forgetting_j for j in 1..T-1)
```

Maximum performance drop on any task.

**Edge Cases**:
- T = 1: No backward/forward transfer (undefined). Return NaN.
- Task performance improves over time: Negative forgetting (good).
- All tasks identical: BWT should be 0; non-zero indicates implementation bug.

---

## 6. Cross-Modality Metrics

### 6.1 Per-Modality Accuracy

Evaluate each modality independently by zeroing out all other modality inputs.

```
accuracy_m = evaluate(model, data_m_only)
```

### 6.2 Fusion Improvement Ratio

```
FIR = accuracy_fusion / max(accuracy_m for m in modalities) - 1.0
```

**Interpretation**:
- FIR > 0: Fusion improves over best single modality.
- FIR = 0: No fusion benefit.
- FIR < 0: Fusion hurts -- indicates integration problems.

### 6.3 Multi-Modal Agreement

```
agreement = sum(pred_m1 == pred_m2 for all pairs) / (N * C(M, 2))
```

Where C(M, 2) is the number of modality pairs.

---

## 7. NaN Handling Policy

All metrics follow a consistent NaN handling policy:

1. **Accumulation**: NaN inputs are skipped with a logged warning.
2. **Division by zero**: Return 0.0 (not NaN) for precision/recall/F1 when denominator is zero. Set a `zero_division_flag`.
3. **AUROC undefined**: Return NaN when only one class is present. Exclude from averages.
4. **Empty batches**: Return 0.0 for all metrics. Log warning.
5. **Reporting**: NaN values in reports are serialized as `null` in JSON, empty string in CSV.
6. **Comparison**: When computing deltas between reports, NaN - X = NaN. Flag as "incomparable".

---

## 8. Metric Selection by Task Type

| Task Type | Primary Metrics | Secondary Metrics |
|-----------|----------------|-------------------|
| classify | accuracy, F1_macro, AUROC | top_5_accuracy, F1_weighted, per_class |
| few_shot | mean_accuracy, CI_95 | per_episode_variance |
| anomaly | NAB_score, F1, AUROC | precision, recall, best_threshold |
| reasoning | exact_match, consistency | proof_accuracy, step_accuracy |
| continual | avg_accuracy, BWT, FWT | forgetting, per_task_accuracy |
| cross_modal | per_modality_accuracy, FIR | agreement, ablation_results |

---

## 9. Implementation Checklist

- [ ] All metrics accumulate across batches (no recomputation from scratch).
- [ ] Confusion matrix stored as integer tensor (int64) to avoid float rounding.
- [ ] AUROC computed from raw probabilities (not argmax).
- [ ] F1 edge cases tested with synthetic data matching manual calculation.
- [ ] NAB scoring uses standard profile by default.
- [ ] All metrics support both binary and multi-class.
- [ ] Per-class metrics dictionary keyed by class index (int).
- [ ] Reset method clears all accumulators.
- [ ] Thread-safe accumulation for distributed evaluation.
