# Out-of-Distribution Detection Reference

## Overview

Out-of-distribution (OOD) detection identifies inputs that differ significantly from the training distribution. For brain_ai, this is critical because the multi-layer cognitive pipeline can produce confident but meaningless outputs on OOD data. The architecture provides multiple detection points: encoder features, workspace competition dynamics, SNN firing patterns, and output logits, each offering complementary OOD signals.

This reference covers four OOD detection methods, evaluation metrics, threshold calibration strategies, and the unique workspace entropy approach enabled by brain_ai's global workspace architecture.

---

## Problem Formulation

Given:
- A model `f` trained on in-distribution data `D_in`
- A test input `x` that may come from `D_in` or an unknown distribution `D_out`

Goal: Compute a score `s(x)` such that:
- `s(x)` is low when `x ~ D_in` (in-distribution)
- `s(x)` is high when `x ~ D_out` (out-of-distribution)

Then threshold: `predict_ood(x) = s(x) > tau`, where tau is calibrated on a validation set.

Key challenge: `D_out` is unknown at training time. The detector must generalize to arbitrary OOD distributions.

---

## Energy-Based OOD Scoring

### Mathematical Foundation

Energy-based OOD detection (Liu et al., 2020) uses the Helmholtz free energy of the logit vector as an OOD score:

```
E(x) = -T * log(sum_i exp(f_i(x) / T))
```

Where:
- `f_i(x)` is the i-th logit (pre-softmax output)
- `T` is a temperature parameter (default T=1)
- The energy is the negative log of the partition function

**Intuition**: In-distribution inputs produce logits with one or a few large values (low energy). OOD inputs produce more uniform logits (high energy). The energy score is equivalent to the negative log-sum-exp of logits.

### Relationship to Softmax Confidence

The maximum softmax probability (MSP) baseline (Hendrycks & Gimpel, 2017) uses:
```
s_MSP(x) = -max_i softmax(f(x))_i
```

Energy scoring improves on MSP because:
1. It uses all logits, not just the maximum
2. It is theoretically connected to the likelihood via the Gibbs distribution
3. Temperature scaling provides a tunable knob for separation

### Implementation

```python
def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute energy-based OOD score.

    Args:
        logits: Model output logits, shape (batch, num_classes)
        temperature: Scaling temperature (higher T smooths the score)

    Returns:
        Energy scores, shape (batch,). Higher = more OOD.
    """
    # Energy = -T * logsumexp(logits / T)
    # Negate so higher = more OOD
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return energy
```

### Temperature Tuning

The temperature `T` controls the sensitivity of the energy score:
- **T=1**: Standard energy. Good default.
- **T>1**: Smooths the logit distribution. Better for models with very peaked logits.
- **T<1**: Sharpens the logit distribution. Better for models with flat logits.

Optimal T is selected by maximizing AUROC on a validation set with known OOD data.

### Brain_ai Integration

For brain_ai, energy scoring operates on the classification logits from `decision_heads.classify()`:

```python
# In brain_ai forward pass
system_output = brain(inputs, return_details=True)
logits = system_output.output  # Classification logits
ood_score = energy_score(logits, temperature=1.0)
```

---

## Mahalanobis Distance

### Mathematical Foundation

Mahalanobis distance (Lee et al., 2018) measures how far a feature vector is from the closest class-conditional Gaussian in feature space:

```
M(x) = min_c (f(x) - mu_c)^T Sigma^{-1} (f(x) - mu_c)
```

Where:
- `f(x)` is the feature representation (from an intermediate layer)
- `mu_c` is the mean feature vector for class `c`
- `Sigma` is the shared covariance matrix (tied across classes)
- The minimum is over all classes `c`

**Intuition**: In-distribution features cluster around class means. OOD features are far from all class means in the Mahalanobis metric, which accounts for feature correlations.

### Fitting Phase

Computing Mahalanobis distance requires fitting class statistics on in-distribution data:

```python
class MahalanobisDetector:
    def fit(self, model, train_loader):
        """
        Compute class means and shared covariance from training data.
        """
        features_by_class = defaultdict(list)

        for inputs, targets in train_loader:
            # Extract features from penultimate layer
            feats = model.extract_features(inputs)
            for i, y in enumerate(targets):
                features_by_class[y.item()].append(feats[i])

        # Class means
        self.class_means = {}
        for c, feat_list in features_by_class.items():
            self.class_means[c] = torch.stack(feat_list).mean(dim=0)

        # Shared covariance
        all_centered = []
        for c, feat_list in features_by_class.items():
            feats = torch.stack(feat_list)
            centered = feats - self.class_means[c].unsqueeze(0)
            all_centered.append(centered)

        all_centered = torch.cat(all_centered, dim=0)
        self.precision = torch.inverse(
            torch.mm(all_centered.t(), all_centered) / len(all_centered)
            + 1e-5 * torch.eye(all_centered.shape[1])  # Regularization
        )
```

### Scoring

```python
def mahalanobis_score(self, features):
    """
    Compute Mahalanobis distance to nearest class.

    Args:
        features: Feature vectors, shape (batch, feature_dim)

    Returns:
        Mahalanobis distances, shape (batch,). Higher = more OOD.
    """
    min_distances = torch.full((features.shape[0],), float('inf'))

    for c, mean in self.class_means.items():
        diff = features - mean.unsqueeze(0)
        # M(x) = diff @ precision @ diff^T
        dist = (diff @ self.precision * diff).sum(dim=1)
        min_distances = torch.min(min_distances, dist)

    return -min_distances  # Negate: closer to class = more in-distribution
```

### Multi-Layer Mahalanobis

Using features from multiple layers improves detection:

```python
def multi_layer_mahalanobis(self, inputs):
    """Combine Mahalanobis scores from multiple layers."""
    scores = []
    for layer_name in self.layer_names:
        feats = self.extract_layer(inputs, layer_name)
        score = self.mahalanobis_score(feats)
        scores.append(score)

    # Weighted combination (weights learned on validation set)
    return sum(w * s for w, s in zip(self.layer_weights, scores))
```

### Brain_ai Integration

For brain_ai, extract features at multiple points in the pipeline:

1. **Encoder output**: Features from the modality encoder (e.g., vision encoder output before workspace projection)
2. **Workspace representation**: The integrated workspace vector after global workspace competition
3. **Pre-logit features**: Features from the decision head before the final linear layer

The workspace representation is often the most informative single layer because it integrates all modalities and has been competitively filtered.

---

## Workspace Entropy Method

### Unique to brain_ai

The global workspace architecture provides a unique OOD signal: the entropy of workspace competition. In brain_ai, modality encoders compete for workspace access. The competition produces attention weights across modalities. When the input is OOD, no modality "recognizes" it strongly, leading to high-entropy (uniform) attention.

### Mathematical Formulation

```
H_ws(x) = -sum_m alpha_m(x) * log(alpha_m(x))
```

Where:
- `alpha_m(x)` is the workspace attention weight for modality `m`
- The sum is over all competing modalities
- High entropy indicates uncertainty about which modality should dominate

### Implementation

```python
def workspace_entropy_score(model, inputs):
    """
    Compute workspace competition entropy as OOD score.

    In-distribution inputs produce low-entropy attention (one modality
    dominates). OOD inputs produce high-entropy attention (all modalities
    equally confused).

    Returns:
        Entropy scores, shape (batch,). Higher = more OOD.
    """
    system_output = model(inputs, return_details=True)
    attention = system_output.attention  # Dict of attention weights

    if attention is None:
        raise ValueError("Model must return attention weights for workspace entropy")

    # Collect attention weights across modalities
    # attention shape: (batch, num_modalities) or similar
    attn_weights = torch.stack(list(attention.values()), dim=-1)

    # Normalize to probability distribution
    attn_probs = F.softmax(attn_weights, dim=-1)

    # Compute entropy
    entropy = -(attn_probs * torch.log(attn_probs + 1e-10)).sum(dim=-1)

    return entropy
```

### Advantages Over Standard Methods

1. **Architecture-native**: No additional computation needed beyond the normal forward pass
2. **Modality-aware**: Captures cross-modal inconsistency that single-modality detectors miss
3. **Complementary**: Workspace entropy detects different failure modes than energy or Mahalanobis
4. **Interpretable**: High entropy directly indicates "no modality recognizes this input"

### Limitations

1. Only available for multi-modal inputs (requires workspace competition)
2. Requires the workspace to be enabled (`use_workspace=True`)
3. Single-modality inputs reduce the signal (competition is over a single encoder)
4. The workspace attention mechanism must be differentiable for gradient-based adaptation

### Combining with Other Methods

Best practice is to combine workspace entropy with energy or Mahalanobis:

```python
def combined_ood_score(model, inputs, weights=(0.4, 0.3, 0.3)):
    """
    Combine multiple OOD detection methods.

    Weights: (energy, mahalanobis, workspace_entropy)
    """
    w_e, w_m, w_ws = weights

    logits = model(inputs)
    e_score = energy_score(logits)
    m_score = mahalanobis_score(extract_features(model, inputs))
    ws_score = workspace_entropy_score(model, inputs)

    # Normalize each score to [0, 1] using validation set statistics
    e_norm = (e_score - e_mean) / e_std
    m_norm = (m_score - m_mean) / m_std
    ws_norm = (ws_score - ws_mean) / ws_std

    return w_e * e_norm + w_m * m_norm + w_ws * ws_norm
```

---

## SNN Firing Rate Anomaly

### Concept

SNN neurons have characteristic firing rate patterns for in-distribution inputs. The distribution of firing rates across the SNN layer provides an OOD signal:

- In-distribution: Firing rates match the expected distribution (sparse, with rates near the target rate of ~10%)
- OOD: Firing rates are either too high (hypersynchronous) or too low (quiescent), or have unusual spatial patterns

### Implementation Approach

```python
def snn_firing_rate_score(model, inputs):
    """
    Compute SNN firing rate anomaly score.

    Compares observed firing rates against expected distribution
    from training data.
    """
    # Get spike trains from SNN core
    with torch.no_grad():
        encoded = model.encode(inputs)
        # Access SNN internals (depends on model architecture)
        # spike_rates shape: (batch, num_neurons)
        spike_rates = get_snn_firing_rates(model, encoded)

    # Compare to fitted distribution
    # During fit(), compute mean and std of firing rates per neuron
    rate_zscore = (spike_rates - fitted_mean) / (fitted_std + 1e-8)

    # Aggregate: mean absolute z-score across neurons
    anomaly_score = rate_zscore.abs().mean(dim=1)

    return anomaly_score
```

### Brain_ai Integration

This method requires access to SNN internals, which means the SNN core must expose firing rate information. In brain_ai's SNN core, this can be obtained from the spike counting layer or by monitoring membrane potentials.

---

## Evaluation Metrics

### AUROC (Area Under ROC Curve)

The primary metric for OOD detection. AUROC measures the probability that a randomly chosen OOD sample has a higher score than a randomly chosen in-distribution sample.

```
AUROC = P(s(x_out) > s(x_in))
```

- **AUROC = 1.0**: Perfect separation
- **AUROC = 0.5**: Random guessing
- **Target**: AUROC > 0.9 for reliable detection (done-when gate)

Computed by varying the threshold and plotting True Positive Rate vs False Positive Rate:

```python
def compute_auroc(id_scores, ood_scores):
    """
    Compute AUROC for OOD detection.

    Args:
        id_scores: OOD scores for in-distribution samples
        ood_scores: OOD scores for out-of-distribution samples

    Returns:
        AUROC value in [0, 1]
    """
    labels = torch.cat([
        torch.zeros(len(id_scores)),   # ID = 0
        torch.ones(len(ood_scores)),    # OOD = 1
    ])
    scores = torch.cat([id_scores, ood_scores])

    # Sort by score
    sorted_indices = scores.argsort(descending=True)
    sorted_labels = labels[sorted_indices]

    # Compute TPR and FPR at each threshold
    tpr_list, fpr_list = [], []
    n_pos = sorted_labels.sum().item()
    n_neg = len(sorted_labels) - n_pos

    tp, fp = 0, 0
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Numerical integration (trapezoidal rule)
    auroc = 0.0
    for i in range(1, len(fpr_list)):
        auroc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2

    return auroc
```

### FPR@95TPR (False Positive Rate at 95% True Positive Rate)

FPR@95TPR is the false positive rate when the true positive rate is 95%. It measures how many in-distribution samples are incorrectly flagged as OOD when 95% of actual OOD samples are detected.

```python
def compute_fpr_at_tpr(id_scores, ood_scores, target_tpr=0.95):
    """
    Compute FPR at a given TPR threshold.

    Args:
        id_scores: OOD scores for in-distribution samples
        ood_scores: OOD scores for out-of-distribution samples
        target_tpr: Target true positive rate (default 0.95)

    Returns:
        FPR at the specified TPR
    """
    # Find threshold that achieves target TPR on OOD samples
    sorted_ood = torch.sort(ood_scores, descending=True)[0]
    threshold_idx = int(target_tpr * len(sorted_ood))
    threshold = sorted_ood[min(threshold_idx, len(sorted_ood) - 1)]

    # Compute FPR: fraction of ID samples above threshold
    fpr = (id_scores >= threshold).float().mean().item()

    return fpr
```

- **FPR@95TPR = 0.0**: No false positives when detecting 95% of OOD (ideal)
- **FPR@95TPR = 0.05**: 5% of ID samples are flagged as OOD (acceptable)
- **FPR@95TPR > 0.2**: Detector is unreliable at this operating point

### AUPR (Area Under Precision-Recall Curve)

AUPR is useful when the class balance between ID and OOD is unknown. It is more sensitive to performance on the minority class.

- **AUPR-In**: Treat in-distribution as positive
- **AUPR-Out**: Treat out-of-distribution as positive

Report both for completeness.

---

## Threshold Calibration

### Problem

The OOD score needs a threshold tau to make binary decisions. The threshold must be calibrated on held-out data because:
1. Score distributions vary across models and methods
2. The operating point (desired FPR or TPR) depends on the application
3. A fixed threshold does not generalize across different OOD distributions

### Percentile-Based Calibration

The simplest and most robust approach:

```python
def calibrate_threshold(id_scores, target_fpr=0.05):
    """
    Calibrate OOD threshold based on in-distribution scores.

    Sets threshold at the (1 - target_fpr) percentile of ID scores,
    so that target_fpr fraction of ID samples would be flagged as OOD.

    Args:
        id_scores: OOD scores from in-distribution validation set
        target_fpr: Desired false positive rate

    Returns:
        Calibrated threshold
    """
    sorted_scores = torch.sort(id_scores)[0]
    idx = int((1 - target_fpr) * len(sorted_scores))
    threshold = sorted_scores[min(idx, len(sorted_scores) - 1)]
    return threshold.item()
```

### Validation-Based Calibration

When a small OOD validation set is available, optimize the threshold for maximum F1:

```python
def calibrate_threshold_f1(id_scores, ood_scores, n_thresholds=100):
    """
    Find threshold maximizing F1 score.
    """
    all_scores = torch.cat([id_scores, ood_scores])
    min_s, max_s = all_scores.min(), all_scores.max()
    thresholds = torch.linspace(min_s, max_s, n_thresholds)

    best_f1, best_tau = 0, thresholds[0]
    for tau in thresholds:
        tp = (ood_scores >= tau).sum().float()
        fp = (id_scores >= tau).sum().float()
        fn = (ood_scores < tau).sum().float()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau

    return best_tau.item()
```

---

## Evaluation Protocol for brain_ai

### Standard Benchmark

The standard OOD evaluation protocol for brain_ai:

1. **In-distribution**: MNIST (or the primary training dataset)
2. **OOD datasets**: FashionMNIST, SVHN, Gaussian noise, Uniform noise
3. **Metrics**: AUROC, FPR@95TPR, AUPR for each (ID, OOD) pair
4. **Methods**: Energy, Mahalanobis, workspace entropy (compare all three)

### Done-When Gate

The OOD detection done-when gate requires:
- `OODDetector.evaluate()` achieves AUROC > 0.9 distinguishing MNIST (ID) from FashionMNIST (OOD) with energy-based scoring

This is achievable because MNIST and FashionMNIST have very different feature distributions despite identical image dimensions (28x28 grayscale).

### Reporting Format

```
OOD Detection Results (Energy, T=1.0)
ID: MNIST, OOD: FashionMNIST
  AUROC: 0.952
  FPR@95TPR: 0.032
  AUPR-Out: 0.961

ID: MNIST, OOD: Gaussian Noise
  AUROC: 0.998
  FPR@95TPR: 0.001
  AUPR-Out: 0.999
```

---

## Summary of Detection Methods

| Method | Requires Fitting | Feature Level | Unique to brain_ai | Best Use Case |
|--------|-----------------|---------------|---------------------|---------------|
| Energy | No | Output logits | No | Simple, strong baseline |
| Mahalanobis | Yes (class stats) | Any hidden layer | No | When features are discriminative |
| Workspace entropy | No | Workspace attention | Yes | Multi-modal inputs |
| SNN firing rate | Yes (rate stats) | SNN core | Yes | SNN-specific anomaly |

**Recommended approach**: Start with energy scoring (simplest, no fitting required). Add Mahalanobis if energy alone is insufficient. Use workspace entropy for multi-modal scenarios. Combine methods with learned weights for best overall performance.
