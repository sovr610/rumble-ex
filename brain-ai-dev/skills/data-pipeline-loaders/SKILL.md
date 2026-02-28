---
name: Data Pipeline & Loaders
description: >
  This skill should be used when the user asks to "load a dataset",
  "create data loaders", "add data augmentation", "preprocess input data",
  "set up training data", "create validation splits", "register a dataset",
  "configure data pipeline", "handle multi-modal data loading",
  "load MNIST/CIFAR/ImageNet", "load event camera data", "load sequence data",
  "set up few-shot episodes", "create a dataset registry", or needs guidance on
  data loading, preprocessing, augmentation, or dataset management for the
  brain_ai 7-phase training pipeline.
version: 0.1.0
---

# Data Pipeline & Loaders

## Overview

Guide implementation of complete data loading infrastructure for all 7 training phases. Each phase requires different data modalities, augmentation strategies, and sampling patterns. Cover base loader contracts, per-phase dataset implementations, augmentation pipelines, preprocessing standardization, validation split management, and a unified dataset registry.

## Public Contract

### BasePhaseLoader

Abstract contract all phase-specific loaders must implement.

```python
class BasePhaseLoader(ABC):
    def __init__(self, config: DataConfig, mode: str = "dev"): ...
    @abstractmethod
    def get_train_loader(self) -> DataLoader: ...
    @abstractmethod
    def get_val_loader(self) -> DataLoader: ...
    @abstractmethod
    def get_test_loader(self) -> DataLoader: ...
    @abstractmethod
    def get_dataset_info(self) -> DatasetInfo: ...
    def get_sample_shape(self) -> Dict[str, Tuple[int, ...]]: ...
```

### DatasetRegistry

Central registry for all datasets across phases with auto-download.

```python
class DatasetRegistry:
    def register(self, name: str, loader_cls: Type[BasePhaseLoader], phase: int): ...
    def get_loader(self, name: str, mode: str = "dev") -> BasePhaseLoader: ...
    def list_datasets(self, phase: Optional[int] = None) -> List[DatasetInfo]: ...
    def download(self, name: str, root: str = "data/") -> Path: ...
```

### AugmentationPipeline

Composable augmentation with modality-aware transforms.

```python
class AugmentationPipeline:
    def __init__(self, modality: str, strength: str = "standard"): ...
    def get_train_transforms(self) -> Callable: ...
    def get_eval_transforms(self) -> Callable: ...
    def add_transform(self, transform: Callable, prob: float = 1.0): ...
```

### SplitManager

Deterministic train/val/test split creation and caching.

```python
class SplitManager:
    def __init__(self, seed: int = 42): ...
    def create_splits(self, dataset: Dataset, ratios: Tuple = (0.8, 0.1, 0.1)) -> SplitResult: ...
    def stratified_split(self, dataset: Dataset, labels: Tensor, ratios: Tuple) -> SplitResult: ...
    def save_split_indices(self, path: str) -> None: ...
    def load_split_indices(self, path: str) -> SplitResult: ...
```

## Key Concepts

### Phase-Specific Data Requirements

| Phase | Modality | Dev Dataset | Production Dataset | Loader Class |
|-------|----------|-------------|-------------------|--------------|
| 1 SNN | Vision | MNIST | CIFAR-10/100 | SNNPhaseLoader |
| 2 Encoders | Multi-modal | MNIST+synth | ImageNet + LibriSpeech + WikiText | EncoderPhaseLoader |
| 3 HTM | Sequences | Synthetic seqs | NAB, taxi data | HTMPhaseLoader |
| 4 Workspace | Multi-modal | Simple VQA | VQA v2, CMU-MOSEI | WorkspacePhaseLoader |
| 5 Active Inf. | RL episodes | CartPole | D4RL, Minari | ActiveInfPhaseLoader |
| 6 Reasoning | Logic tasks | Mini-bAbI | bAbI, ProofWriter | ReasoningPhaseLoader |
| 7 Meta-Learn | Episodes | Omniglot | mini-ImageNet | MetaPhaseLoader |

### Augmentation Strategy by Modality

- **Vision**: RandomCrop, HorizontalFlip, ColorJitter, Cutout, RandAugment
- **Text**: Token dropout, synonym replacement, back-translation (optional)
- **Audio**: SpecAugment, time stretching, pitch shifting, noise injection
- **Sequences**: Temporal jittering, subsequence sampling, noise addition
- **RL episodes**: No augmentation (preserve trajectory integrity)

### Dev vs Production Mode

- **Dev**: Small datasets (MNIST-scale), fast iteration, <1min load time
- **Production**: Full-scale datasets, streaming support for large datasets, proper caching

### Preprocessing Contract

All loaders normalize to a common format before returning:
- Vision: `[B, C, H, W]` float32, normalized to [0, 1] or [-1, 1]
- Text: `[B, seq_len]` int64 token IDs
- Audio: `[B, n_mels, T]` float32 log-mel spectrograms
- Sequences: `[B, T, D]` float32 with optional mask `[B, T]`

## Configuration Surface

```python
@dataclass
class DataConfig:
    root_dir: str = "data/"
    phase: int = 1
    mode: str = "dev"                    # dev | production
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    # Splits
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    # Augmentation
    augmentation_strength: str = "standard"  # none | light | standard | heavy
    # Streaming
    use_streaming: bool = False          # For large datasets
    cache_dir: Optional[str] = None
```

## Done-When Gates

1. **Loader Contract** — All 7 phase loaders implement `BasePhaseLoader`; `get_train_loader()` returns a valid DataLoader that yields correctly-shaped tensors matching the preprocessing contract.
2. **Registry Round-Trip** — `DatasetRegistry.get_loader("mnist", "dev")` returns a working loader; `list_datasets(phase=1)` returns all phase-1 datasets.
3. **Augmentation Determinism** — Same seed produces identical augmented batches; eval transforms produce identical output regardless of seed.

## Failure Modes

| Mode | Symptom | Fix |
|------|---------|-----|
| Download failure | Timeout/404 | Implement retry with backoff; cache downloaded data |
| Shape mismatch | RuntimeError in model | Verify preprocessing contract output shapes |
| Worker deadlock | Training hangs | Reduce num_workers, check for pickling issues |
| Memory leak | RAM grows over epochs | Use IterableDataset for large datasets, verify no circular refs |
| Split leakage | Overly optimistic metrics | Use SplitManager with saved indices, verify no overlap |

## Anti-Patterns

- Downloading data inside the training loop — download in setup, not per-epoch
- Hardcoding transforms — use AugmentationPipeline for composability
- Forgetting pin_memory on GPU training — significant throughput impact
- Using random splits without saving indices — breaks reproducibility
- Loading entire production dataset into RAM — use streaming/mmap for large datasets

## Resources

### Reference Files
- **`references/loader-contracts.md`** — BasePhaseLoader interface, DatasetInfo schema, output shapes
- **`references/augmentation-pipeline.md`** — Per-modality transforms, strength presets, composition
- **`references/preprocessing-standards.md`** — Normalization, tokenization, spectrograms, masking
- **`references/dataset-registry.md`** — Registry API, auto-download, caching, versioning
- **`references/testing-matrix.md`** — Test scenarios for data pipeline infrastructure

### Asset Files
- **`assets/base_loader_template.py`** — BasePhaseLoader ABC + DatasetInfo + common utilities
- **`assets/phase_loaders_template.py`** — All 7 phase-specific loader implementations
- **`assets/augmentation_template.py`** — AugmentationPipeline with per-modality transforms
- **`assets/preprocessing_template.py`** — Preprocessing functions for all modalities
- **`assets/split_manager_template.py`** — SplitManager with stratified splitting, index persistence
- **`assets/data_config_template.py`** — DataConfig + DatasetRegistry + validation

### Scripts
- **`scripts/validate_loaders.py`** — Validates all loaders against contracts and done-when gates
- **`scripts/gen_loader_tests.py`** — Generates 100+ pytest test cases for data pipeline
- **`scripts/loader_benchmark.py`** — Throughput benchmarks for data loading pipeline
