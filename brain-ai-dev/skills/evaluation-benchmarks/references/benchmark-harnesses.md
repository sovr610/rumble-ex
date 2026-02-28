# Benchmark Harnesses

Per-phase benchmark setup, data loading, evaluation loops, expected output shapes, and result aggregation for the brain_ai evaluation infrastructure.

---

## 1. Overview

A benchmark harness wraps the evaluation loop for a specific dataset and model configuration. It handles:

1. Dataset loading and preprocessing
2. Batched inference (gradient-free)
3. Metric accumulation
4. Result aggregation and reporting
5. Timing and resource measurement

Each of the 7 cognitive layers has dedicated benchmark configurations for both dev (fast iteration) and production (full evaluation) modes.

---

## 2. BenchmarkHarness Base Class

```python
class BenchmarkHarness:
    def __init__(self, model: BrainAI, config: EvalConfig):
        self.model = model
        self.config = config
        self.metrics = MetricsSuite(config.task_type, config.num_classes, config.device)

    def run(self, dataset: str, split: str = "test") -> BenchmarkResult:
        """Run evaluation on a single dataset split."""
        loader = self._get_dataloader(dataset, split)
        self.metrics.reset()

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                inputs, targets = self._prepare_batch(batch)
                outputs = self.model(inputs)
                predictions = self._extract_predictions(outputs)
                self.metrics.update(predictions, targets)

        return BenchmarkResult(
            dataset=dataset,
            split=split,
            metrics=self.metrics.compute(),
            per_class=self.metrics.per_class_metrics(),
            confusion_matrix=self.metrics.confusion_matrix(),
            config=self.config,
        )

    def run_suite(self, datasets: List[str]) -> List[BenchmarkResult]:
        """Run evaluation on multiple datasets."""
        return [self.run(ds) for ds in datasets]

    def compare(self, results: List[BenchmarkResult]) -> ComparisonReport:
        """Compare results across datasets or model variants."""
        ...
```

### Key Design Decisions

- **Gradient-free**: All evaluation uses `torch.no_grad()` to save memory.
- **Deterministic**: Seeds are set before each dataset evaluation.
- **Batched**: Configurable batch size (default 64, reduce for OOM).
- **Device-aware**: Inputs moved to model device; metrics accumulated on CPU.

---

## 3. Phase-Specific Benchmark Configurations

### Phase 1: SNN Core

**Purpose**: Validate spiking neural network feature extraction quality.

| Setting | Dev | Production |
|---------|-----|------------|
| Dataset | MNIST | CIFAR-10, CIFAR-100 |
| Input shape | (B, 1, 28, 28) | (B, 3, 32, 32) |
| Output shape | (B, 10) | (B, 10) or (B, 100) |
| Primary metric | accuracy | accuracy, F1_macro |
| Batch size | 128 | 64 |
| Expected accuracy | >95% (MNIST) | >85% (CIFAR-10) |
| Time limit | <30s | <10min |

**Data Loading**:
```python
# Dev: MNIST from torchvision
transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
dataset = MNIST(root="data/", train=False, transform=transform)

# Production: CIFAR-10
transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
dataset = CIFAR10(root="data/", train=False, transform=transform)
```

**Input Preparation**:
```python
def _prepare_batch(self, batch):
    images, labels = batch
    inputs = {"vision": images.to(self.device)}
    return inputs, labels.to(self.device)
```

### Phase 2: Modality Encoders

**Purpose**: Validate each encoder produces correct-dimensional output and achieves baseline performance.

| Setting | Dev | Production |
|---------|-----|------------|
| Vision | MNIST | ImageNet-1k (val split) |
| Text | Synthetic sequences | WikiText-103 (perplexity) |
| Audio | Synthetic spectrograms | LibriSpeech (WER) |
| Output dim | encoder.output_dim (4096) | Same |

**Per-Encoder Checks**:
```python
# Vision encoder output
assert encoded["vision"].shape == (B, encoder_config.output_dim)

# Text encoder output
assert encoded["text"].shape == (B, encoder_config.output_dim)

# Audio encoder output
assert encoded["audio"].shape == (B, encoder_config.output_dim)
```

**Cross-Encoder Agreement**: After encoding, check that modality representations are in compatible ranges (mean/std statistics).

### Phase 3: HTM Temporal

**Purpose**: Validate temporal sequence learning and anomaly detection.

| Setting | Dev | Production |
|---------|-----|------------|
| Dataset | Synthetic sequences | NAB, Yahoo S5 |
| Sequence length | 100-500 | 1000-10000 |
| Anomaly fraction | 5% injected | Natural distribution |
| Primary metric | NAB score | NAB score, F1, AUROC |
| Input shape | (B, T, D) | Same |

**Synthetic Data Generation**:
```python
def generate_anomaly_data(n_sequences, seq_len, anomaly_rate=0.05):
    """Generate sine waves with injected point and collective anomalies."""
    data = torch.sin(torch.linspace(0, 20*pi, seq_len)).unsqueeze(0).repeat(n_sequences, 1)
    data += torch.randn_like(data) * 0.1
    labels = torch.zeros(n_sequences, seq_len)
    for i in range(n_sequences):
        n_anomalies = int(seq_len * anomaly_rate)
        positions = torch.randperm(seq_len)[:n_anomalies]
        data[i, positions] += torch.randn(n_anomalies) * 3.0
        labels[i, positions] = 1.0
    return data, labels
```

### Phase 4: Global Workspace

**Purpose**: Validate multi-modal integration and workspace competition.

| Setting | Dev | Production |
|---------|-----|------------|
| Dataset | Simple multimodal (synthetic) | VQA v2, CMU-MultimodalSDK |
| Modalities | vision + text (synthetic) | vision + text + audio |
| Primary metric | accuracy, FIR | accuracy, FIR, agreement |
| Workspace dim | workspace_config.workspace_dim | Same |

**Multi-Modal Evaluation Protocol**:
1. Full fusion evaluation (all modalities).
2. Per-modality ablation (one modality at a time).
3. Compute Fusion Improvement Ratio.
4. Check workspace attention distribution.

### Phase 5: Active Inference

**Purpose**: Validate decision-making under uncertainty.

| Setting | Dev | Production |
|---------|-----|------------|
| Environment | CartPole, MountainCar | D4RL, Minari |
| Metric | cumulative reward | normalized score, EFE accuracy |
| Episodes | 100 | 1000 |
| Time limit | <60s | <30min |

**Evaluation Loop** (different from classification):
```python
def run_control(self, env, n_episodes=100):
    rewards = []
    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            inputs = {"sensors": torch.tensor(obs).unsqueeze(0)}
            action = self.model.act(inputs, deterministic=True)
            obs, reward, done, info = env.step(action.squeeze().numpy())
            total_reward += reward
        rewards.append(total_reward)
    return {"mean_reward": np.mean(rewards), "std_reward": np.std(rewards)}
```

### Phase 6: Reasoning

**Purpose**: Validate logical reasoning capabilities.

| Setting | Dev | Production |
|---------|-----|------------|
| Dataset | Mini-bAbI (task 1-3) | bAbI full, ProofWriter, FOLIO |
| Metric | exact_match, consistency | exact_match, proof_accuracy |
| Samples | 100 | 10000+ |

**bAbI Task Format**:
```
Input: "Mary went to the bathroom. John moved to the hallway. Where is Mary?"
Target: "bathroom"
```

**Evaluation**: Exact match after normalization (lowercase, strip whitespace).

### Phase 7: Meta-Learning

**Purpose**: Validate few-shot adaptation capability.

| Setting | Dev | Production |
|---------|-----|------------|
| Dataset | Omniglot 5-way 1-shot | mini-ImageNet, tiered-ImageNet |
| Episodes | 100 | 600-1000 |
| Protocol | 5-way 1-shot, 5-way 5-shot | 5-way 1-shot, 5-way 5-shot |
| Metric | mean accuracy +/- CI_95 | Same |
| Time limit | <60s | <30min |

---

## 4. Result Aggregation

### BenchmarkResult Structure

```python
@dataclass
class BenchmarkResult:
    dataset: str
    split: str
    metrics: Dict[str, float]
    per_class: Dict[int, Dict[str, float]]
    confusion_matrix: Optional[Tensor]
    config: EvalConfig
    timestamp: str
    duration_seconds: float
    num_samples: int
    model_info: Dict[str, Any]
```

### ComparisonReport Structure

```python
@dataclass
class ComparisonReport:
    results: List[BenchmarkResult]
    deltas: Dict[str, Dict[str, float]]
    baseline: Optional[BenchmarkResult]
    summary: str
```

### Suite Aggregation

When running `run_suite()`, results are aggregated:

```python
def aggregate_suite(results: List[BenchmarkResult]) -> Dict[str, float]:
    all_metrics = defaultdict(list)
    for r in results:
        for k, v in r.metrics.items():
            if not math.isnan(v):
                all_metrics[k].append(v)
    return {k: mean(v) for k, v in all_metrics.items()}
```

---

## 5. Dev vs Production Configuration

### Dev Mode

- Small datasets (MNIST, synthetic data).
- Fewer samples (100-1000).
- Relaxed thresholds.
- Single modality where possible.
- Target: complete in <60 seconds on CPU.

```python
EvalConfig(
    task_type="classify",
    num_classes=10,
    batch_size=128,
    device="cpu",
    n_episodes=100,
    anomaly_window=50,
)
```

### Production Mode

- Full datasets (ImageNet, NAB, bAbI).
- Complete test splits.
- All metrics computed.
- Multi-modality.
- Target: complete in <30 minutes on GPU.

```python
EvalConfig(
    task_type="classify",
    num_classes=1000,
    batch_size=64,
    device="cuda",
    metrics=["accuracy", "top_5_accuracy", "f1_macro", "f1_weighted", "auroc"],
    n_episodes=600,
    anomaly_window=100,
    save_confusion_matrix=True,
    save_per_class=True,
)
```

---

## 6. Data Loading Patterns

### Standard Classification

```python
def _get_dataloader(self, dataset_name, split):
    dataset = self._load_dataset(dataset_name, split)
    return DataLoader(
        dataset,
        batch_size=self.config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
```

### Sequence Data (HTM)

```python
def _get_sequence_loader(self, dataset_name, split):
    dataset = self._load_sequences(dataset_name, split)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=self._pad_sequences,
    )
```

### Few-Shot Episodes

```python
def _sample_episode(self, dataset, n_way, k_shot, n_query):
    classes = random.sample(dataset.classes, n_way)
    support, query = [], []
    for cls in classes:
        indices = dataset.class_indices[cls]
        selected = random.sample(indices, k_shot + n_query)
        support.extend(selected[:k_shot])
        query.extend(selected[k_shot:])
    return support, query
```

---

## 7. Timing and Resource Measurement

Each benchmark records:

```python
timing = {
    "total_seconds": end - start,
    "per_sample_ms": (end - start) / num_samples * 1000,
    "per_batch_ms": (end - start) / num_batches * 1000,
    "peak_memory_mb": torch.cuda.max_memory_allocated() / 1e6 if cuda else None,
    "num_samples": num_samples,
    "num_batches": num_batches,
}
```

---

## 8. Error Handling

### OOM Recovery

```python
try:
    outputs = model(inputs)
except RuntimeError as e:
    if "out of memory" in str(e):
        torch.cuda.empty_cache()
        for sub_batch in split_batch(inputs, factor=2):
            outputs = model(sub_batch)
    else:
        raise
```

### Missing Datasets

```python
def _load_dataset(self, name, split):
    try:
        return load_dataset(name, split)
    except FileNotFoundError:
        logger.warning(f"Dataset {name} not found. Generating synthetic fallback.")
        return self._generate_synthetic(name, split)
```

### Label Mismatch Detection

```python
def _validate_labels(self, predictions, targets):
    pred_classes = predictions.argmax(dim=-1).unique()
    target_classes = targets.unique()
    if not torch.all(torch.isin(target_classes, torch.arange(self.config.num_classes))):
        raise ValueError(f"Target labels outside expected range")
```

---

## 9. Harness Registration

Phase-specific harnesses are registered and can be looked up:

```python
HARNESS_REGISTRY = {
    "phase1_snn": SNNBenchmarkHarness,
    "phase2_encoders": EncoderBenchmarkHarness,
    "phase3_htm": HTMBenchmarkHarness,
    "phase4_workspace": WorkspaceBenchmarkHarness,
    "phase5_inference": ActiveInferenceBenchmarkHarness,
    "phase6_reasoning": ReasoningBenchmarkHarness,
    "phase7_meta": MetaLearningBenchmarkHarness,
}

def get_harness(phase: str, model, config: EvalConfig) -> BenchmarkHarness:
    cls = HARNESS_REGISTRY.get(phase)
    if cls is None:
        raise ValueError(f"Unknown phase: {phase}")
    return cls(model, config)
```

---

## 10. End-to-End Example

```python
from brain_ai import create_brain_ai

model = create_brain_ai(modalities=["vision"], output_type="classify", num_classes=10)

config = EvalConfig(
    task_type="classify",
    num_classes=10,
    batch_size=64,
    metrics=["accuracy", "f1_macro", "auroc"],
    device="cpu",
)

harness = BenchmarkHarness(model, config)
result = harness.run("mnist", split="test")

print(f"Accuracy: {result.metrics['accuracy']:.4f}")
print(f"F1 Macro: {result.metrics['f1_macro']:.4f}")

result.save("runs/run_001/mnist_result.json")
```
