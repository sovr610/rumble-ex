# Testing Matrix Reference

## Overview

This document defines the complete testing matrix for the robustness and adversarial testing infrastructure. It covers five categories: attack generation, OOD detection, corruption benchmarks, adversarial training, and calibration analysis. Each category includes specific test scenarios, expected outcomes, pass/fail criteria, and diagnostic guidance.

The tests are designed to run with simple models (small CNNs or MLPs) on synthetic data, not requiring the full brain_ai system or GPU access. This ensures they can be executed as part of CI/CD pipelines.

---

## Category 1: Attack Generation Tests

### Test Scenario 1.1: FGSM Perturbation Budget

**Objective**: Verify that FGSM perturbations respect the epsilon budget.

| Property | Value |
|----------|-------|
| Input | Random tensor (batch=8, C=1, H=28, W=28) |
| Model | Simple CNN (Conv-ReLU-Conv-ReLU-Linear) |
| Epsilon | 8/255 |
| Metric | L-inf norm of perturbation |
| Pass criterion | `max(abs(x_adv - x)) <= epsilon + 1e-6` for all samples |

### Test Scenario 1.2: FGSM Changes Output

**Objective**: Verify that FGSM perturbation changes model predictions.

| Property | Value |
|----------|-------|
| Input | Correctly classified samples |
| Model | Trained simple CNN (>80% accuracy) |
| Epsilon | 8/255 |
| Metric | Fraction of predictions that change |
| Pass criterion | At least 10% of predictions change |

### Test Scenario 1.3: PGD Stronger Than FGSM

**Objective**: PGD should achieve equal or higher attack success rate than FGSM.

| Property | Value |
|----------|-------|
| Input | Same test batch for both attacks |
| Model | Same trained model |
| Epsilon | 8/255 for both |
| PGD steps | 20 |
| Metric | Attack success rate (fraction fooled) |
| Pass criterion | `pgd_success >= fgsm_success - 0.05` (allowing small tolerance) |

### Test Scenario 1.4: PGD Perturbation Budget

**Objective**: PGD perturbations must stay within the epsilon ball.

| Property | Value |
|----------|-------|
| Steps | 20, 50 |
| Epsilon | 4/255, 8/255, 16/255 |
| Metric | L-inf norm of perturbation |
| Pass criterion | `max(abs(x_adv - x)) <= epsilon + 1e-6` |

### Test Scenario 1.5: PGD Valid Input Range

**Objective**: Adversarial examples must remain in valid input range.

| Property | Value |
|----------|-------|
| Metric | Min and max values of adversarial examples |
| Pass criterion | `x_adv.min() >= 0.0 - 1e-6` and `x_adv.max() <= 1.0 + 1e-6` |

### Test Scenario 1.6: Accuracy Drop at Standard Epsilon

**Objective**: Done-when gate - accuracy drops >20% at epsilon=8/255.

| Property | Value |
|----------|-------|
| Model | Trained model with >80% clean accuracy |
| Attack | PGD-20 at epsilon=8/255 |
| Metric | Clean accuracy - robust accuracy |
| Pass criterion | Accuracy drop > 20 percentage points |

### Test Scenario 1.7: AutoAttack Ensemble

**Objective**: AutoAttack runs all component attacks without error.

| Property | Value |
|----------|-------|
| Components | APGD-CE, APGD-DLR, optional FAB and Square |
| Metric | Completes without exception; returns valid accuracy |
| Pass criterion | Robust accuracy is a valid float in [0, 1] |

### Test Scenario 1.8: Robustness Report Generation

**Objective**: `measure_robustness()` produces a complete report.

| Property | Value |
|----------|-------|
| Epsilons | [0, 2/255, 4/255, 8/255, 16/255] |
| Metric | Report contains accuracy at each epsilon |
| Pass criterion | Report has entries for all epsilons; accuracy at eps=0 equals clean accuracy |

### Test Scenario 1.9: L2 Attack Variant

**Objective**: L2-bounded PGD respects the L2 budget.

| Property | Value |
|----------|-------|
| Epsilon | 0.5 (L2) |
| Metric | L2 norm of perturbation per sample |
| Pass criterion | `perturbation.flatten(1).norm(dim=1) <= epsilon + 1e-4` for all samples |

### Test Scenario 1.10: Targeted Attack

**Objective**: Targeted attack produces predictions matching the target class.

| Property | Value |
|----------|-------|
| Target class | Random class different from true label |
| Epsilon | 16/255 (large for targeted to succeed) |
| Steps | 50 |
| Metric | Fraction of adversarial examples classified as target |
| Pass criterion | At least 30% success rate on simple model |

---

## Category 2: OOD Detection Tests

### Test Scenario 2.1: Energy Score Separation

**Objective**: Energy scores separate in-distribution from OOD data.

| Property | Value |
|----------|-------|
| ID data | Random Gaussian with specific statistics |
| OOD data | Random uniform or Gaussian with different statistics |
| Model | Simple classifier trained on ID |
| Metric | Mean energy score difference |
| Pass criterion | Mean OOD energy > Mean ID energy |

### Test Scenario 2.2: AUROC Computation Correctness

**Objective**: AUROC computation is mathematically correct.

| Property | Value |
|----------|-------|
| Perfect separation | ID scores all < OOD scores |
| Random scores | Both from same distribution |
| Pass criterion | Perfect = 1.0; Random approximately 0.5 |

### Test Scenario 2.3: OOD Detection AUROC > 0.9

**Objective**: Done-when gate - energy-based detection achieves AUROC > 0.9.

| Property | Value |
|----------|-------|
| ID | Synthetic "MNIST-like" data (specific distribution) |
| OOD | Synthetic "FashionMNIST-like" data (different distribution) |
| Model | Trained on ID data |
| Method | Energy-based scoring |
| Pass criterion | AUROC > 0.9 |

### Test Scenario 2.4: FPR at 95% TPR

**Objective**: FPR@95TPR is computed correctly and reasonable.

| Property | Value |
|----------|-------|
| Metric | FPR when 95% of OOD samples are detected |
| Pass criterion | FPR@95TPR < 0.5 (less than half of ID falsely flagged) |

### Test Scenario 2.5: Mahalanobis Fit and Score

**Objective**: Mahalanobis detector fits class statistics and produces valid scores.

| Property | Value |
|----------|-------|
| Fit data | Synthetic multi-class features |
| Test data | In-class and out-of-class features |
| Pass criterion | Scores are finite; in-class scores < out-of-class scores on average |

### Test Scenario 2.6: Threshold Calibration

**Objective**: Calibrated threshold achieves target FPR.

| Property | Value |
|----------|-------|
| Target FPR | 0.05 |
| ID validation data | 1000 samples |
| Pass criterion | Actual FPR within 0.02 of target on held-out ID data |

### Test Scenario 2.7: Workspace Entropy Scoring

**Objective**: Workspace entropy method produces valid scores.

| Property | Value |
|----------|-------|
| Input | Simulated attention weights (multi-modal) |
| ID pattern | One modality dominates (low entropy) |
| OOD pattern | Uniform attention (high entropy) |
| Pass criterion | OOD entropy > ID entropy |

### Test Scenario 2.8: Multiple OOD Distributions

**Objective**: Detector generalizes across different OOD types.

| Property | Value |
|----------|-------|
| OOD types | Gaussian noise, uniform random, shifted distribution |
| Metric | AUROC for each OOD type |
| Pass criterion | AUROC > 0.7 for all OOD types |

---

## Category 3: Corruption Benchmark Tests

### Test Scenario 3.1: Corruption Function Validity

**Objective**: Each corruption function produces valid output.

| Property | Value |
|----------|-------|
| Input | Random tensor in [0, 1] range |
| Corruptions | All 15 types |
| Severities | All 5 levels |
| Pass criteria | Output is finite, shape matches input, values in [0, 1] |

### Test Scenario 3.2: Severity Monotonicity

**Objective**: Done-when gate - errors increase monotonically with severity.

| Property | Value |
|----------|-------|
| Model | Simple CNN |
| Corruptions | All 15 types |
| Metric | Error rate at each severity |
| Pass criterion | error(sev=k) <= error(sev=k+1) + 0.02 for all k (with tolerance) |

### Test Scenario 3.3: MCE Computation

**Objective**: MCE is computed correctly.

| Property | Value |
|----------|-------|
| Model errors | Known synthetic error rates |
| Baseline errors | Known synthetic baseline |
| Pass criterion | MCE matches hand-computed value |

### Test Scenario 3.4: Relative MCE

**Objective**: Relative MCE correctly compares two models.

| Property | Value |
|----------|-------|
| Model A | Known error rates |
| Model B (reference) | Known error rates |
| Pass criterion | Relative MCE = MCE_A / MCE_B (within numerical precision) |

### Test Scenario 3.5: Clean Accuracy Baseline

**Objective**: Corruption at severity 0 (no corruption) matches clean accuracy.

| Property | Value |
|----------|-------|
| Metric | Accuracy on uncorrupted data |
| Pass criterion | Clean accuracy equals model accuracy without corruption |

### Test Scenario 3.6: Corruption Report Completeness

**Objective**: Benchmark report contains all corruption types and severities.

| Property | Value |
|----------|-------|
| Corruptions | 15 types |
| Severities | 5 per type |
| Pass criterion | Report has 15 * 5 = 75 entries |

### Test Scenario 3.7: Tensor-Only Operation

**Objective**: All corruption functions work with PyTorch tensors only (no PIL).

| Property | Value |
|----------|-------|
| Input | torch.Tensor |
| Dependencies | torch only |
| Pass criterion | No ImportError for PIL/numpy; output is torch.Tensor |

---

## Category 4: Adversarial Training Tests

### Test Scenario 4.1: PGD-AT Loss Decrease

**Objective**: Training loss decreases over multiple steps.

| Property | Value |
|----------|-------|
| Model | Simple CNN |
| Steps | 50 training steps |
| Metric | Loss at step 50 vs step 1 |
| Pass criterion | Final loss < initial loss |

### Test Scenario 4.2: PGD-AT Improves Robustness

**Objective**: Adversarial training improves robust accuracy.

| Property | Value |
|----------|-------|
| Model | Simple CNN, before and after AT |
| Assessment | PGD-20 robust accuracy |
| Pass criterion | Robust accuracy after AT > robust accuracy before AT |

### Test Scenario 4.3: TRADES Loss Computation

**Objective**: TRADES loss has correct components.

| Property | Value |
|----------|-------|
| Components | CE loss and KL divergence |
| Pass criteria | Total loss = CE + beta * KL; both components are positive and finite |

### Test Scenario 4.4: TRADES Beta Effect

**Objective**: Higher beta increases robustness at cost of clean accuracy.

| Property | Value |
|----------|-------|
| Beta values | 1.0, 6.0 |
| Metric | Robust accuracy at each beta |
| Pass criterion | Robust accuracy at beta=6 >= robust accuracy at beta=1 |

### Test Scenario 4.5: Free-AT Replay Count

**Objective**: Free-AT with more replays produces better robustness.

| Property | Value |
|----------|-------|
| Replays | 2, 4 |
| Metric | Robust accuracy after training |
| Pass criterion | 4 replays >= 2 replays (with tolerance) |

### Test Scenario 4.6: Curriculum Epsilon Schedule

**Objective**: Epsilon increases correctly over epochs.

| Property | Value |
|----------|-------|
| Schedule | Linear warmup over 10 epochs |
| Target epsilon | 8/255 |
| Pass criterion | epoch 0 epsilon < target; epoch 10 epsilon = target |

### Test Scenario 4.7: Training Does Not Collapse

**Objective**: Adversarial training maintains reasonable accuracy.

| Property | Value |
|----------|-------|
| Metric | Clean accuracy after AT |
| Pass criterion | Clean accuracy > random chance (>15% for 10 classes) |

### Test Scenario 4.8: Gradient Flow

**Objective**: Gradients flow through the model during adversarial training.

| Property | Value |
|----------|-------|
| Metric | Parameter gradient norms |
| Pass criterion | All parameter gradients are non-zero and finite |

---

## Category 5: Calibration Tests

### Test Scenario 5.1: ECE Computation Correctness

**Objective**: Expected Calibration Error is computed correctly.

| Property | Value |
|----------|-------|
| Perfect calibration | Confidence = accuracy in each bin |
| Overconfident | Confidence > accuracy |
| Pass criteria | Perfect calibration ECE approximately 0; Overconfident ECE > 0 |

### Test Scenario 5.2: ECE Range

**Objective**: ECE is in valid range [0, 1].

| Property | Value |
|----------|-------|
| Input | Random logits and targets |
| Pass criterion | 0 <= ECE <= 1 |

### Test Scenario 5.3: Reliability Diagram

**Objective**: Reliability diagram data has correct structure.

| Property | Value |
|----------|-------|
| Output | Bin accuracies, bin confidences, bin counts |
| Pass criteria | Same number of bins; accuracies and confidences in [0, 1] |

### Test Scenario 5.4: Temperature Scaling

**Objective**: Temperature scaling reduces ECE.

| Property | Value |
|----------|-------|
| Model | Overconfident model (logits * 5) |
| Metric | ECE before and after temperature scaling |
| Pass criterion | ECE after temperature scaling <= ECE before |

### Test Scenario 5.5: Temperature Range

**Objective**: Learned temperature is positive and finite.

| Property | Value |
|----------|-------|
| Metric | Optimal temperature value |
| Pass criterion | 0.01 < temperature < 100 |

### Test Scenario 5.6: Calibration Under Shift

**Objective**: ECE increases under distribution shift.

| Property | Value |
|----------|-------|
| Clean data | Original test set |
| Shifted data | Corrupted or perturbed test set |
| Metric | ECE on clean vs shifted |
| Pass criterion | ECE on shifted >= ECE on clean (typically) |

### Test Scenario 5.7: Platt Scaling Alternative

**Objective**: Platt scaling (learned affine transform) reduces ECE.

| Property | Value |
|----------|-------|
| Method | Learn a, b such that calibrated_logits = a * logits + b |
| Pass criterion | ECE after Platt scaling <= ECE before |

---

## Test Infrastructure Requirements

### Simple Test Model

All tests use a simple CNN or MLP, not the full brain_ai system:

```python
class SimpleCNN(nn.Module):
    """Test model: Conv-ReLU-Conv-ReLU-FC for 28x28 images."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(7)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)
```

### Synthetic Data

Tests use synthetic data to avoid dataset dependencies:

```python
def make_synthetic_data(n=200, num_classes=10):
    """Create synthetic classification data."""
    x = torch.randn(n, 1, 28, 28)
    y = torch.randint(0, num_classes, (n,))
    return x, y
```

### Quick Training

Tests that require trained models use rapid training:

```python
def quick_train(model, x, y, epochs=20, lr=0.01):
    """Train model quickly for testing."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
```

---

## Test Summary

| Category | Test Count | Critical Tests (Done-When) |
|----------|-----------|---------------------------|
| Attack Generation | 10 | 1.6 (accuracy drop >20%) |
| OOD Detection | 8 | 2.3 (AUROC > 0.9) |
| Corruption Benchmark | 7 | 3.2 (monotonicity) |
| Adversarial Training | 8 | 4.1, 4.2 (loss decrease, robustness improvement) |
| Calibration | 7 | 5.1, 5.4 (ECE correctness, temperature scaling) |
| **Total** | **40** | **5 critical** |

All tests should complete in under 60 seconds on CPU. No GPU required. No external dataset downloads required.
