---
name: Robustness & Adversarial Testing
description: >
  This skill should be used when the user asks to "test robustness",
  "generate adversarial examples", "detect out-of-distribution inputs",
  "perturbation analysis", "test noise resilience", "FGSM attack",
  "PGD attack", "adversarial training", "OOD detection", "input corruption",
  "stress test the model", "calibration under shift", "robustness benchmark",
  "certified robustness", or needs guidance on adversarial attacks,
  out-of-distribution detection, perturbation testing, or robustness
  evaluation for the brain_ai system.
version: 0.1.0
---

# Robustness & Adversarial Testing

## Overview

Guide implementation of robustness testing and adversarial evaluation infrastructure. The brain_ai system's multi-layer architecture creates unique robustness properties — SNN temporal coding is naturally noise-tolerant, HTM's sparse representations resist corruption, but workspace competition can be disrupted by adversarial perturbations. Cover adversarial attack generation, OOD detection, input corruption benchmarks, adversarial training, certified robustness bounds, and calibration under distribution shift.

## Public Contract

### AdversarialAttacker

Generate adversarial examples using standard attack methods.

```python
class AdversarialAttacker:
    def __init__(self, model: BrainAI, config: AttackConfig): ...
    def fgsm(self, inputs: Dict[str, Tensor], targets: Tensor, epsilon: float) -> Dict[str, Tensor]: ...
    def pgd(self, inputs: Dict[str, Tensor], targets: Tensor, epsilon: float, steps: int = 20) -> Dict[str, Tensor]: ...
    def auto_attack(self, inputs: Dict[str, Tensor], targets: Tensor) -> AttackResult: ...
    def measure_robustness(self, loader: DataLoader, epsilons: List[float]) -> RobustnessReport: ...
```

### OODDetector

Detect out-of-distribution inputs before they reach the model.

```python
class OODDetector:
    def __init__(self, model: BrainAI, config: OODConfig): ...
    def fit(self, in_distribution_loader: DataLoader) -> None: ...
    def detect(self, inputs: Dict[str, Tensor]) -> OODResult: ...
    def get_score(self, inputs: Dict[str, Tensor]) -> Tensor: ...  # Higher = more OOD
    def evaluate(self, id_loader: DataLoader, ood_loader: DataLoader) -> OODMetrics: ...
```

Detection methods:
- **Energy-based**: Energy score from logits (simple, effective)
- **Mahalanobis**: Distance from class-conditional Gaussians in feature space
- **Workspace entropy**: High competition entropy signals unfamiliar inputs
- **SNN firing rate**: OOD inputs produce abnormal firing patterns

### CorruptionBenchmark

Test model under standard input corruptions.

```python
class CorruptionBenchmark:
    def __init__(self, model: BrainAI, config: CorruptionConfig): ...
    def run(self, clean_loader: DataLoader, corruptions: List[str]) -> CorruptionReport: ...
    def get_mce(self) -> float: ...  # Mean Corruption Error
    def get_relative_mce(self, baseline_errors: Dict) -> float: ...
```

Corruption types: gaussian_noise, shot_noise, impulse_noise, defocus_blur, motion_blur, zoom_blur, brightness, contrast, elastic_transform, pixelate, jpeg_compression (severity 1-5).

### AdversarialTrainer

Adversarial training loop with PGD-AT or TRADES.

```python
class AdversarialTrainer:
    def __init__(self, model: BrainAI, config: AdvTrainConfig): ...
    def train_step(self, batch: Dict, targets: Tensor, optimizer: Optimizer) -> Dict[str, float]: ...
    def trades_loss(self, clean: Dict, adv: Dict, targets: Tensor, beta: float) -> Tensor: ...
```

### CalibrationAnalyzer

Measure confidence calibration and reliability under shift.

```python
class CalibrationAnalyzer:
    def __init__(self, model: BrainAI, config: CalibrationConfig): ...
    def compute_ece(self, logits: Tensor, targets: Tensor, n_bins: int = 15) -> float: ...
    def reliability_diagram(self, logits: Tensor, targets: Tensor) -> Figure: ...
    def temperature_scaling(self, val_logits: Tensor, val_targets: Tensor) -> float: ...
    def calibration_under_shift(self, loaders: Dict[str, DataLoader]) -> Dict[str, float]: ...
```

## Key Concepts

### Robustness Properties by Module

| Module | Natural Robustness | Vulnerability |
|--------|-------------------|---------------|
| SNN Core | Temporal coding tolerates noise | Surrogate gradients enable gradient attacks |
| HTM | Sparse representations resist corruption | Column activation patterns can be disrupted |
| Workspace | Competition provides implicit filtering | Adversarial inputs can hijack broadcast |
| Reasoning | System 2 provides verification | System 1 fast path vulnerable |
| Engram | Hash collisions provide fuzzy matching | Targeted hash collisions possible |

### OOD Detection Hierarchy

1. **Input-level**: Statistical tests on raw input features
2. **Encoder-level**: Mahalanobis distance in encoder output space
3. **Workspace-level**: Competition entropy (highest signal-to-noise)
4. **Output-level**: Energy score from classification logits

### Adversarial Training Strategy

For brain_ai, adversarial training should:
- Apply perturbations to encoder inputs (not internal representations)
- Use PGD-AT with ε appropriate per modality (vision: 8/255, text: embedding perturbation)
- Combine with TRADES for accuracy-robustness balance
- Only adversarially train phases 1-2 (encoders); later phases inherit robustness

## Configuration Surface

```python
@dataclass
class AttackConfig:
    method: str = "pgd"                  # fgsm | pgd | auto
    epsilon: float = 8/255               # L-inf budget
    pgd_steps: int = 20
    pgd_step_size: float = 2/255
    norm: str = "linf"                   # linf | l2
    targeted: bool = False

@dataclass
class OODConfig:
    method: str = "energy"               # energy | mahalanobis | workspace_entropy
    temperature: float = 1.0
    threshold: Optional[float] = None    # Auto-calibrated if None

@dataclass
class CorruptionConfig:
    corruptions: List[str] = ("all",)    # "all" or specific names
    severities: List[int] = (1, 2, 3, 4, 5)
    batch_size: int = 64

@dataclass
class AdvTrainConfig:
    method: str = "pgd_at"               # pgd_at | trades | free_at
    epsilon: float = 8/255
    pgd_steps: int = 7
    trades_beta: float = 6.0
    free_at_replays: int = 4
```

## Done-When Gates

1. **Attack Generation** — `AdversarialAttacker.pgd()` produces adversarial examples that fool the model (accuracy drops >20% at ε=8/255 on MNIST); perturbations respect the ε budget.
2. **OOD Detection** — `OODDetector.evaluate()` achieves AUROC >0.9 distinguishing MNIST (ID) from FashionMNIST (OOD) with energy-based scoring.
3. **Corruption Benchmark** — `CorruptionBenchmark.run()` produces valid MCE scores across all 11 corruption types at 5 severities; scores monotonically increase with severity.

## Failure Modes

| Mode | Symptom | Fix |
|------|---------|-----|
| Gradient masking | FGSM works but PGD fails | Use stronger attacks (AutoAttack); check gradient flow |
| OOD threshold too tight | Rejects valid inputs | Calibrate on validation set; use percentile threshold |
| Corruption not applicable | Text/audio corruptions wrong | Use modality-specific corruption functions |
| Adversarial training collapse | Accuracy drops to random | Reduce epsilon; increase warmup; use TRADES |
| Calibration fail | ECE increases after temp scaling | Use Platt scaling; check val set representativeness |

## Anti-Patterns

- Testing with FGSM only — FGSM is a weak attack; always include PGD or AutoAttack
- Reporting clean accuracy only — always report robust accuracy at standard epsilon
- OOD detection without calibration — threshold must be calibrated on held-out data
- Adversarial training all layers — perturb inputs only, not internal representations
- Ignoring modality-specific budgets — ε=8/255 for images ≠ ε for text embeddings

## Resources

### Reference Files
- **`references/adversarial-attacks.md`** — FGSM, PGD, AutoAttack, CW, modality-specific attacks
- **`references/ood-detection.md`** — Energy, Mahalanobis, workspace entropy, evaluation protocols
- **`references/corruption-benchmarks.md`** — ImageNet-C style corruptions, MCE computation
- **`references/adversarial-training.md`** — PGD-AT, TRADES, Free-AT, curriculum adversarial training
- **`references/testing-matrix.md`** — Test scenarios for robustness infrastructure

### Asset Files
- **`assets/attacker_template.py`** — AdversarialAttacker with FGSM, PGD, AutoAttack
- **`assets/ood_detector_template.py`** — OODDetector with energy, Mahalanobis, workspace entropy
- **`assets/corruption_benchmark_template.py`** — CorruptionBenchmark with all corruption types
- **`assets/adversarial_trainer_template.py`** — AdversarialTrainer with PGD-AT, TRADES
- **`assets/calibration_template.py`** — CalibrationAnalyzer with ECE, reliability diagrams
- **`assets/robustness_config_template.py`** — All config dataclasses + validation

### Scripts
- **`scripts/validate_robustness.py`** — Validates robustness infrastructure against done-when gates
- **`scripts/gen_robustness_tests.py`** — Generates 100+ pytest test cases
- **`scripts/robustness_benchmark.py`** — Full robustness evaluation suite with reporting
