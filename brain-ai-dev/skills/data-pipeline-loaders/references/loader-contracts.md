# Loader Contracts — Reference for Data Pipeline & Loaders Skill

This document specifies the abstract base class contract (`BasePhaseLoader`), the `DatasetInfo` schema, output shape specifications per modality, and the `get_sample_shape()` protocol. Use this as the canonical reference when implementing, auditing, or extending phase-specific data loaders for the brain_ai 7-phase training pipeline.

---

## 1. BasePhaseLoader Abstract Base Class

Every phase-specific loader in the brain_ai system must inherit from `BasePhaseLoader` and implement all abstract methods. The ABC enforces a uniform interface so that training scripts, validation tools, and benchmarks can interact with any phase loader through identical method signatures.

### 1.1 Constructor Contract

```python
class BasePhaseLoader(ABC):
    def __init__(self, config: DataConfig, mode: str = "dev"):
        ...
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `DataConfig` | required | Centralized data configuration (batch size, workers, paths, augmentation) |
| `mode` | `str` | `"dev"` | Either `"dev"` (small synthetic/MNIST-scale data, <1 min load) or `"production"` (full-scale datasets, streaming support) |

**Constructor responsibilities:**
1. Store `config` and `mode` as instance attributes.
2. Resolve the dataset root directory from `config.root_dir`.
3. Create the augmentation pipeline by calling `AugmentationPipeline` with the appropriate modality and `config.augmentation_strength`.
4. In `dev` mode, generate or load a small synthetic dataset that matches production shapes.
5. In `production` mode, verify that the dataset exists on disk or trigger download via the `DatasetRegistry`.
6. Create train/val/test splits using `SplitManager` with `config.split_seed`.

**Invariants enforced by the constructor:**
- `mode` must be one of `("dev", "production")`. Raise `ValueError` otherwise.
- `config.batch_size` must be a positive integer.
- `config.num_workers` must be non-negative.
- The dataset root directory must exist or be creatable.

### 1.2 Abstract Methods

#### `get_train_loader() -> DataLoader`

Return a `torch.utils.data.DataLoader` configured for training. This loader:
- Applies training augmentations (random crops, flips, noise, etc.).
- Shuffles data every epoch.
- Uses `pin_memory=config.pin_memory` for GPU transfers.
- Uses `num_workers=config.num_workers` and `prefetch_factor=config.prefetch_factor`.
- Uses `drop_last=True` to prevent partial batches from destabilizing batch normalization.

#### `get_val_loader() -> DataLoader`

Return a DataLoader for validation. This loader:
- Applies evaluation-only transforms (deterministic normalization, no random augmentation).
- Does NOT shuffle.
- Uses `drop_last=False` to measure on every sample.
- Uses the same `num_workers` and `pin_memory` as training.

#### `get_test_loader() -> DataLoader`

Return a DataLoader for final test measurement. Identical contract to `get_val_loader()` but operates on the held-out test split. The test split must never overlap with train or val splits (verified by `SplitManager`).

#### `get_dataset_info() -> DatasetInfo`

Return a `DatasetInfo` dataclass describing the loaded dataset. This is a pure metadata query that does not trigger data loading. It must be callable before any loader is created.

### 1.3 Concrete Methods

#### `get_sample_shape() -> Dict[str, Tuple[int, ...]]`

Returns a dictionary mapping each output key to its expected per-sample tensor shape (without the batch dimension). This method has a default implementation that inspects the first batch from the train loader:

```python
def get_sample_shape(self) -> Dict[str, Tuple[int, ...]]:
    loader = self.get_train_loader()
    batch = next(iter(loader))
    shapes = {}
    if isinstance(batch, dict):
        for key, tensor in batch.items():
            shapes[key] = tuple(tensor.shape[1:])  # Remove batch dim
    elif isinstance(batch, (tuple, list)):
        shapes["input"] = tuple(batch[0].shape[1:])
        if len(batch) > 1:
            shapes["target"] = tuple(batch[1].shape[1:])
    return shapes
```

Phase-specific loaders may override this with hard-coded shapes for efficiency (avoids loading a full batch just to check shapes).

#### `get_collate_fn() -> Optional[Callable]`

Returns a custom collate function if the loader requires non-standard batching (e.g., variable-length sequences needing padding, few-shot episode construction). Returns `None` to use PyTorch's default collate.

#### `get_num_samples() -> Dict[str, int]`

Returns a dictionary with keys `"train"`, `"val"`, `"test"` mapping to the number of samples in each split. Used for logging, progress bars, and learning rate scheduling.

---

## 2. DatasetInfo Schema

The `DatasetInfo` dataclass is the standard metadata container returned by `get_dataset_info()`. It provides everything a training script needs to configure the model and optimizer without loading any data.

```python
@dataclass
class DatasetInfo:
    name: str                              # Human-readable name (e.g., "MNIST", "synthetic_snn")
    phase: int                             # Training phase (1-7)
    modality: str                          # Primary modality: vision, text, audio, sequence, multimodal, rl, episodes
    num_classes: Optional[int]             # Number of classes (None for regression/RL)
    num_train_samples: int                 # Number of training samples
    num_val_samples: int                   # Number of validation samples
    num_test_samples: int                  # Number of test samples
    input_shapes: Dict[str, Tuple[int, ...]]  # Per-key shapes without batch dim
    target_shape: Optional[Tuple[int, ...]]   # Target shape without batch dim (None for unsupervised)
    dtype: torch.dtype                     # Primary tensor dtype (torch.float32 for most)
    target_dtype: torch.dtype              # Target dtype (torch.long for classification, torch.float32 for regression)
    description: str                       # Short description of the dataset
    source: str                            # Origin: "synthetic", "torchvision", "huggingface", "custom"
    version: str                           # Dataset version string (e.g., "1.0.0")
```

### 2.1 Field Semantics

**`name`**: A short, unique identifier. For dev mode synthetic datasets, prefix with `"synthetic_"` (e.g., `"synthetic_snn"`, `"synthetic_htm"`). For production datasets, use the canonical name (e.g., `"mnist"`, `"cifar10"`, `"imagenet21k"`).

**`phase`**: Integer 1-7 corresponding to the brain_ai training phase. Used by the `DatasetRegistry` to filter datasets by phase.

**`modality`**: One of `"vision"`, `"text"`, `"audio"`, `"sequence"`, `"multimodal"`, `"rl"`, `"episodes"`. The workspace integration layer uses this to route data to the correct encoder.

**`input_shapes`**: A dictionary where keys are the names of input tensors and values are shape tuples. For simple datasets this is `{"input": (C, H, W)}`. For multimodal datasets it might be `{"vision": (3, 224, 224), "text": (512,)}`.

**`target_shape`**: The shape of a single target/label tensor. For classification this is `()` (scalar class index). For regression it might be `(D,)`. For unsupervised phases this is `None`.

**`target_dtype`**: Determines the loss function type. `torch.long` signals cross-entropy loss. `torch.float32` signals MSE/L1 loss or policy gradient objectives.

---

## 3. Output Shape Contracts per Modality

Every loader must produce batched tensors that conform exactly to these shape specifications. The brain_ai encoder suite expects these shapes at input and will raise runtime errors on mismatch.

### 3.1 Vision: `[B, C, H, W]`

| Field | Dev Shape | Production Shape | dtype | Value Range |
|---|---|---|---|---|
| Input | `[B, 1, 28, 28]` | `[B, 3, 224, 224]` or `[B, 3, 384, 384]` | `float32` | `[0, 1]` or `[-1, 1]` |
| Target | `[B]` | `[B]` | `int64` | `[0, num_classes)` |

**Normalization convention:** All vision tensors are normalized to `[0, 1]` by default. When using ImageNet-pretrained encoders in production, apply channel-wise normalization to `[-1, 1]` or ImageNet mean/std.

**Spatial dimensions:** Dev mode uses 28x28 (MNIST-scale). Production uses 224x224 (standard) or 384x384 (ViT-Large scale matching `config.encoder.vision_image_size`). The encoder handles resize internally if needed, but the loader should provide the target resolution.

### 3.2 Text: `[B, seq_len]`

| Field | Dev Shape | Production Shape | dtype | Value Range |
|---|---|---|---|---|
| Input (token IDs) | `[B, 128]` | `[B, 512]` or `[B, 8192]` | `int64` | `[0, vocab_size)` |
| Attention mask | `[B, seq_len]` | `[B, seq_len]` | `bool` or `int64` | `{0, 1}` |
| Target | `[B]` or `[B, seq_len]` | `[B]` or `[B, seq_len]` | `int64` | `[0, num_classes)` or `[0, vocab_size)` |

**Padding:** Sequences shorter than `seq_len` are right-padded with the pad token (typically ID 0). The attention mask distinguishes real tokens (1) from padding (0).

**Tokenization:** Dev mode uses a simple character-level or whitespace tokenizer. Production uses a BPE tokenizer with `vocab_size=128000` matching `config.encoder.text_vocab_size`.

### 3.3 Audio: `[B, n_mels, T]`

| Field | Dev Shape | Production Shape | dtype | Value Range |
|---|---|---|---|---|
| Input (mel spectrogram) | `[B, 64, 100]` | `[B, 128, 1000]` | `float32` | log-scale, typically `[-10, 2]` |
| Target | `[B]` | `[B]` or `[B, T_target]` | `int64` | `[0, num_classes)` |

**Spectrogram parameters:** Dev mode uses 64 mel bands, 100 time frames. Production uses 128 mel bands (`config.encoder.audio_n_mels`) and variable-length time frames (padded to maximum in batch). The time dimension `T` varies per utterance; the collate function pads to the maximum length in the batch.

**Normalization:** Log-mel spectrograms are computed as `log(mel + 1e-9)`. Per-channel (per-mel-bin) normalization to zero mean and unit variance is applied after log transform.

### 3.4 Sequences: `[B, T, D]`

| Field | Dev Shape | Production Shape | dtype | Value Range |
|---|---|---|---|---|
| Input | `[B, 50, 16]` | `[B, 200, 64]` or `[B, T, D]` | `float32` | Normalized to `[-1, 1]` or `[0, 1]` |
| Mask | `[B, T]` | `[B, T]` | `bool` | `{True, False}` |
| Target | `[B, T, D]` or `[B]` | `[B, T, D]` or `[B]` | `float32` or `int64` | Depends on task |

**Temporal dimension:** `T` is the sequence length (timesteps). For HTM phase 3, sequences are time series data. For reasoning phase 6, sequences are multi-step logic chains.

**Feature dimension:** `D` is the feature dimension per timestep. Dev mode uses 16. Production uses 64 or matches the workspace dimension.

**Masking:** Boolean mask where `True` indicates valid timesteps. Required for variable-length sequences.

---

## 4. Phase-Specific Contract Extensions

Each of the 7 phases extends the base contract with phase-specific requirements.

### 4.1 Phase 1 -- SNN Core (SNNPhaseLoader)

**Primary modality:** Vision.
**Additional output:** Spike-encoded targets (optional, for spike-based loss functions).
**Dev dataset:** Synthetic MNIST-like grayscale images.
**Shape contract:** `{"input": (1, 28, 28), "target": ()}` in dev; `{"input": (3, 32, 32), "target": ()}` in production (CIFAR-10).

### 4.2 Phase 2 -- Encoders (EncoderPhaseLoader)

**Primary modality:** Multimodal (vision + text + audio).
**Additional output:** Per-modality tensors in a dictionary batch.
**Dev dataset:** Synthetic multimodal samples.
**Shape contract:** `{"vision": (1, 28, 28), "text": (128,), "audio": (64, 100), "target": ()}`.
**Collate:** Custom collate that handles dictionary-of-tensors batching.

### 4.3 Phase 3 -- HTM (HTMPhaseLoader)

**Primary modality:** Sequences.
**Additional output:** Anomaly labels (binary), next-step predictions.
**Dev dataset:** Synthetic periodic sequences with injected anomalies.
**Shape contract:** `{"input": (50, 16), "mask": (50,), "target": (50, 16)}`.

### 4.4 Phase 4 -- Workspace (WorkspacePhaseLoader)

**Primary modality:** Multimodal (all modalities simultaneously).
**Additional output:** Cross-modal alignment labels.
**Dev dataset:** Synthetic paired vision-text-audio samples.
**Shape contract:** `{"vision": (1, 28, 28), "text": (128,), "audio": (64, 100), "target": ()}`.

### 4.5 Phase 5 -- Active Inference (ActiveInfPhaseLoader)

**Primary modality:** RL episodes (state-action-reward trajectories).
**Additional output:** Actions, rewards, done flags, next states.
**Dev dataset:** Synthetic CartPole-like trajectories.
**Shape contract:** `{"state": (4,), "action": (), "reward": (), "next_state": (4,), "done": ()}`.
**Episode structure:** Each sample is a complete trajectory of length `T`.

### 4.6 Phase 6 -- Reasoning (ReasoningPhaseLoader)

**Primary modality:** Text (logic tasks).
**Additional output:** Reasoning chains, answer labels.
**Dev dataset:** Synthetic mini-bAbI tasks.
**Shape contract:** `{"context": (256,), "question": (64,), "target": ()}`.

### 4.7 Phase 7 -- Meta-Learning (MetaPhaseLoader)

**Primary modality:** Episodes (N-way K-shot).
**Additional output:** Support set, query set, episode labels.
**Dev dataset:** Synthetic few-shot episodes from random class prototypes.
**Shape contract:** Support `{"input": (N*K, 1, 28, 28), "target": (N*K,)}`, Query `{"input": (N*Q, 1, 28, 28), "target": (N*Q,)}`.
**Episode parameters:** Default N=5 (ways), K=1 (shot), Q=15 (queries).

---

## 5. Collate Function Specifications

### 5.1 Default Collate

For phases that produce uniform-shape tensors (phases 1, 7), PyTorch's default `collate_fn` is sufficient. Each sample is a tuple `(input_tensor, target_tensor)` and the default collate stacks them into `(B, ...)`.

### 5.2 Dictionary Collate

For multimodal phases (2, 4), a custom collate function stacks each key independently:

```python
def dict_collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([sample[key] for sample in batch]) for key in keys}
```

### 5.3 Padded Sequence Collate

For variable-length phases (3, 6), a padding collate function pads sequences to the maximum length in the batch and produces a mask tensor:

```python
def padded_collate_fn(batch):
    inputs, targets = zip(*batch)
    max_len = max(x.shape[0] for x in inputs)
    padded_inputs = torch.zeros(len(inputs), max_len, inputs[0].shape[-1])
    masks = torch.zeros(len(inputs), max_len, dtype=torch.bool)
    for i, x in enumerate(inputs):
        padded_inputs[i, :x.shape[0]] = x
        masks[i, :x.shape[0]] = True
    targets = torch.stack(targets)
    return {"input": padded_inputs, "mask": masks, "target": targets}
```

### 5.4 Episode Collate

For meta-learning phase 7, the collate function constructs episodes:

```python
def episode_collate_fn(batch):
    support_x = torch.stack([ep["support_x"] for ep in batch])
    support_y = torch.stack([ep["support_y"] for ep in batch])
    query_x = torch.stack([ep["query_x"] for ep in batch])
    query_y = torch.stack([ep["query_y"] for ep in batch])
    return {"support_x": support_x, "support_y": support_y,
            "query_x": query_x, "query_y": query_y}
```

### 5.5 RL Trajectory Collate

For active inference phase 5, the collate function handles trajectory dictionaries:

```python
def trajectory_collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        tensors = [sample[key] for sample in batch]
        collated[key] = torch.stack(tensors)
    return collated
```

---

## 6. Loader Lifecycle

The expected lifecycle of a loader follows this sequence:

```
1. config = DataConfig(phase=N, mode="dev", batch_size=64)
2. loader = PhaseNLoader(config, mode="dev")        # Constructor: resolve data, create splits
3. info = loader.get_dataset_info()                  # Metadata query (no data loaded)
4. shapes = loader.get_sample_shape()                # Shape verification
5. train_dl = loader.get_train_loader()              # Create training DataLoader
6. val_dl = loader.get_val_loader()                  # Create validation DataLoader
7. for epoch in range(num_epochs):
8.     for batch in train_dl:                        # Iterate with augmentation
9.         train_step(batch)
10.    for batch in val_dl:                          # Iterate without augmentation
11.        validation_step(batch)
12. test_dl = loader.get_test_loader()               # Create test DataLoader (end of training)
13. for batch in test_dl:
14.     final_measurement(batch)
```

**Thread safety:** DataLoaders created by the same loader instance share the underlying dataset object. Do not use multiple DataLoaders from the same loader across threads without external synchronization.

**Memory management:** Calling `get_train_loader()` multiple times returns a new DataLoader wrapping the same dataset. The dataset itself is not reloaded. For production streaming datasets, each DataLoader may open independent file handles.

---

## 7. Compliance Checklist

Use this checklist when implementing a new phase loader:

- [ ] Inherits from `BasePhaseLoader`
- [ ] Constructor accepts `DataConfig` and `mode` string
- [ ] `get_train_loader()` returns a `DataLoader` with `shuffle=True` and `drop_last=True`
- [ ] `get_val_loader()` returns a `DataLoader` with `shuffle=False` and `drop_last=False`
- [ ] `get_test_loader()` returns a `DataLoader` with `shuffle=False` and `drop_last=False`
- [ ] `get_dataset_info()` returns a valid `DatasetInfo` with all fields populated
- [ ] `get_sample_shape()` returns shapes matching the modality contract
- [ ] Train/val/test splits have zero index overlap
- [ ] Dev mode loads in under 60 seconds with no network access required
- [ ] Production mode supports lazy/streaming loading for datasets > 10GB
- [ ] Augmentation is applied only to training data, not validation or test
- [ ] All tensors have correct dtypes (`float32` for inputs, `int64` for classification targets)
- [ ] Batch dimension is always first (dim=0)
- [ ] No data leakage between splits
