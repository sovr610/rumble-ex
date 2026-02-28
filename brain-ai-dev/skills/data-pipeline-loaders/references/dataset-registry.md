# Dataset Registry — Reference for Data Pipeline & Loaders Skill

This document specifies the `DatasetRegistry` API, auto-download mechanisms with retry logic, caching strategy, dataset versioning, phase-based listing, and custom dataset registration. Use this as the canonical reference when implementing or extending the dataset management infrastructure for the brain_ai training pipeline.

---

## 1. DatasetRegistry Class Interface

The `DatasetRegistry` is a singleton-pattern class that serves as the central catalog of all datasets available to the brain_ai training pipeline. It maps dataset names to their loader classes, phase associations, and download metadata.

### 1.1 Constructor

```python
class DatasetRegistry:
    def __init__(self, cache_dir: Optional[str] = None):
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cache_dir` | `Optional[str]` | `None` | Root directory for downloaded datasets. Defaults to `"data/"` |

The registry maintains an internal dictionary mapping dataset names to `DatasetEntry` objects:

```python
@dataclass
class DatasetEntry:
    name: str                              # Unique dataset identifier
    loader_cls: Type[BasePhaseLoader]      # Loader class for this dataset
    phase: int                             # Associated training phase (1-7)
    modality: str                          # Primary modality
    description: str                       # Human-readable description
    url: Optional[str]                     # Download URL (None for synthetic)
    version: str                           # Version string
    size_bytes: Optional[int]              # Approximate download size
    checksum: Optional[str]               # SHA-256 checksum of download
    requires_auth: bool                    # Whether download requires authentication
    tags: List[str]                        # Searchable tags
```

### 1.2 Core Methods

#### `register(name, loader_cls, phase, **kwargs) -> None`

Register a new dataset with the registry.

```python
def register(
    self,
    name: str,
    loader_cls: Type[BasePhaseLoader],
    phase: int,
    modality: str = "vision",
    description: str = "",
    url: Optional[str] = None,
    version: str = "1.0.0",
    size_bytes: Optional[int] = None,
    checksum: Optional[str] = None,
    requires_auth: bool = False,
    tags: Optional[List[str]] = None,
) -> None:
```

**Validation rules:**
- `name` must be non-empty and unique (raises `ValueError` on duplicate).
- `phase` must be in range `[1, 7]`.
- `loader_cls` must be a subclass of `BasePhaseLoader`.
- `modality` must be one of the recognized modalities.

**Idempotent re-registration:** If `name` already exists and all fields match, the call is a no-op. If fields differ, raise `ValueError` to prevent silent overwrite. Use `force=True` to override.

#### `get_loader(name, mode="dev", config=None) -> BasePhaseLoader`

Retrieve and instantiate a loader for the named dataset.

```python
def get_loader(
    self,
    name: str,
    mode: str = "dev",
    config: Optional[DataConfig] = None,
) -> BasePhaseLoader:
```

**Behavior:**
1. Look up the `DatasetEntry` by name. Raise `KeyError` if not found.
2. If `config` is None, create a default `DataConfig` with `mode=mode`.
3. Instantiate the loader class: `entry.loader_cls(config, mode)`.
4. Return the instantiated loader.

#### `list_datasets(phase=None) -> List[DatasetInfo]`

List all registered datasets, optionally filtered by phase.

```python
def list_datasets(
    self,
    phase: Optional[int] = None,
    modality: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[DatasetInfo]:
```

**Filtering logic:**
- If `phase` is specified, only return datasets for that phase.
- If `modality` is specified, only return datasets for that modality.
- If `tags` is specified, only return datasets that have ALL specified tags.
- Filters are combined with AND logic.

Returns a list of `DatasetInfo` objects (not `DatasetEntry` -- the public API exposes only the metadata schema, not internal implementation details).

#### `download(name, root=None) -> Path`

Download a dataset to local storage.

```python
def download(
    self,
    name: str,
    root: Optional[str] = None,
) -> Path:
```

**Behavior:**
1. Look up the dataset entry.
2. If `url` is None (synthetic dataset), return the root path immediately.
3. Check if the dataset already exists at `root/name/`. If so, verify checksum and return.
4. Download from `url` with retry logic.
5. Verify checksum.
6. Extract if compressed.
7. Return the path to the downloaded dataset.

---

## 2. Auto-Download with Retry

### 2.1 Retry Strategy

Downloads use exponential backoff with jitter:

```
Attempt 1: Immediate
Attempt 2: Wait 1-2 seconds
Attempt 3: Wait 2-4 seconds
Attempt 4: Wait 4-8 seconds
Attempt 5: Wait 8-16 seconds (final attempt)
```

**Configuration:**

| Parameter | Default | Description |
|---|---|---|
| `max_retries` | 5 | Maximum number of download attempts |
| `base_delay` | 1.0 | Base delay in seconds |
| `max_delay` | 60.0 | Maximum delay between retries |
| `backoff_factor` | 2.0 | Multiplier for exponential backoff |
| `timeout` | 300 | HTTP request timeout in seconds |

### 2.2 Error Handling

| Error Type | Behavior |
|---|---|
| HTTP 404 | Fail immediately (no retry) |
| HTTP 401/403 | Fail with auth error message |
| HTTP 429 | Retry with extended backoff (respect Retry-After header) |
| HTTP 5xx | Retry with standard backoff |
| Timeout | Retry with standard backoff |
| ConnectionError | Retry with standard backoff |
| Checksum mismatch | Delete partial file, retry from scratch |

### 2.3 Progress Reporting

Downloads report progress via a callback or logging:

```python
def download_with_progress(url, dest, callback=None):
    # Report: dataset name, bytes downloaded, total bytes, speed, ETA
    ...
```

For CLI usage, display a progress bar. For programmatic usage, emit log messages at INFO level.

### 2.4 Partial Download Recovery

If a download is interrupted, the partial file is retained with a `.partial` suffix. On the next attempt, the downloader checks for partial files and attempts to resume using HTTP Range headers. If the server does not support range requests, the partial file is deleted and download restarts from the beginning.

---

## 3. Caching Strategy

### 3.1 Directory Structure

```
cache_dir/
  registry.json                    # Cached registry metadata
  datasets/
    mnist/
      v1.0.0/
        train-images.pt
        train-labels.pt
        val-images.pt
        val-labels.pt
        test-images.pt
        test-labels.pt
        metadata.json
    cifar10/
      v1.0.0/
        ...
    synthetic_snn/
      v1.0.0/
        ...
```

### 3.2 Cache Validation

On each access, the registry checks:
1. **Version match:** The cached version matches the requested version.
2. **Integrity check:** A lightweight check (file existence and size) on first access, full checksum verification on `download()`.
3. **Staleness:** The cache metadata includes a timestamp. Datasets older than 30 days trigger a staleness warning (but do not auto-refresh).

### 3.3 Cache Eviction

The registry does not automatically evict cached datasets. Users can manually clear the cache:

```python
registry.clear_cache(name="mnist")        # Clear specific dataset
registry.clear_cache()                     # Clear all cached datasets
registry.get_cache_size() -> int           # Total cache size in bytes
```

### 3.4 In-Memory Caching

For small dev-mode datasets (< 100MB), the registry caches the loaded Dataset objects in memory to avoid repeated disk reads. The memory cache uses a WeakValueDictionary to allow garbage collection when the dataset is no longer referenced.

---

## 4. Dataset Versioning

### 4.1 Version Format

Dataset versions follow semantic versioning: `MAJOR.MINOR.PATCH`.

- **MAJOR:** Incompatible changes (different number of samples, different label schema).
- **MINOR:** Backward-compatible additions (new metadata fields, additional samples).
- **PATCH:** Bug fixes (corrected labels, fixed corrupted samples).

### 4.2 Version Resolution

When requesting a dataset without specifying a version, the registry returns the latest version. When specifying a version, exact match is required.

```python
# Latest version
loader = registry.get_loader("mnist")

# Specific version
loader = registry.get_loader("mnist", version="1.0.0")
```

### 4.3 Version Coexistence

Multiple versions of the same dataset can coexist in the cache directory. Each version has its own subdirectory:

```
datasets/mnist/v1.0.0/
datasets/mnist/v1.1.0/
datasets/mnist/v2.0.0/
```

---

## 5. Phase-Based Dataset Listing

### 5.1 Default Datasets per Phase

The registry comes pre-populated with default datasets for each phase:

| Phase | Dev Datasets | Production Datasets |
|---|---|---|
| 1 SNN | `synthetic_snn` | `mnist`, `cifar10`, `cifar100` |
| 2 Encoders | `synthetic_multimodal` | `imagenet21k`, `librispeech`, `wikitext` |
| 3 HTM | `synthetic_sequences` | `nab`, `taxi`, `ecg` |
| 4 Workspace | `synthetic_workspace` | `vqa_v2`, `cmu_mosei` |
| 5 Active Inf. | `synthetic_cartpole` | `d4rl`, `minari` |
| 6 Reasoning | `synthetic_babi` | `babi`, `proofwriter`, `folio` |
| 7 Meta-Learn | `synthetic_episodes` | `omniglot`, `mini_imagenet` |

### 5.2 Listing API Examples

```python
# All datasets for phase 1
phase1 = registry.list_datasets(phase=1)
# Returns: [DatasetInfo(name="synthetic_snn", ...), DatasetInfo(name="mnist", ...), ...]

# All vision datasets
vision = registry.list_datasets(modality="vision")

# All dev-mode datasets
dev_datasets = [d for d in registry.list_datasets() if d.source == "synthetic"]
```

### 5.3 Phase Compatibility Check

The registry can validate that a dataset is appropriate for a given phase:

```python
def is_compatible(self, name: str, phase: int) -> bool:
    entry = self._entries[name]
    return entry.phase == phase
```

This prevents accidental use of a phase-3 HTM dataset in phase-1 SNN training.

---

## 6. Custom Dataset Registration

### 6.1 Registration Protocol

Users can register custom datasets by providing a loader class and metadata:

```python
from brain_ai.data import DatasetRegistry, BasePhaseLoader, DataConfig

class MyCustomLoader(BasePhaseLoader):
    def __init__(self, config: DataConfig, mode: str = "dev"):
        super().__init__(config, mode)
        # Load custom data...

    def get_train_loader(self):
        ...
    def get_val_loader(self):
        ...
    def get_test_loader(self):
        ...
    def get_dataset_info(self):
        ...

registry = DatasetRegistry()
registry.register(
    name="my_custom_dataset",
    loader_cls=MyCustomLoader,
    phase=1,
    modality="vision",
    description="My custom vision dataset for SNN training",
    version="1.0.0",
    tags=["custom", "vision", "research"],
)
```

### 6.2 Validation on Registration

When a custom dataset is registered, the registry performs these checks:

1. **Class inheritance:** `loader_cls` must be a subclass of `BasePhaseLoader`.
2. **Method implementation:** All abstract methods must be implemented (checked via `inspect`).
3. **Name uniqueness:** The name must not conflict with existing registrations.
4. **Phase range:** Phase must be between 1 and 7.

### 6.3 Decorator-Based Registration

For convenience, datasets can be registered via a decorator:

```python
@registry.register_dataset(name="my_dataset", phase=1, modality="vision")
class MyDatasetLoader(BasePhaseLoader):
    ...
```

This is syntactic sugar for calling `registry.register()` after the class definition.

---

## 7. Registry Persistence

### 7.1 Saving Registry State

The registry state (all entries except loader classes) can be serialized to JSON:

```python
registry.save(path="registry.json")
```

The saved file contains:
```json
{
    "version": "1.0.0",
    "entries": {
        "mnist": {
            "name": "mnist",
            "phase": 1,
            "modality": "vision",
            "description": "...",
            "url": "...",
            "version": "1.0.0",
            "size_bytes": 11490434,
            "checksum": "sha256:...",
            "loader_cls": "brain_ai.data.phase_loaders.SNNPhaseLoader",
            "tags": ["vision", "classification"]
        }
    }
}
```

### 7.2 Loading Registry State

```python
registry = DatasetRegistry.load(path="registry.json")
```

Loader classes are resolved by their fully-qualified module path. If a class cannot be imported, the entry is marked as unavailable but retained in the registry.

---

## 8. Thread Safety

The `DatasetRegistry` uses a threading lock to protect registration and lookup operations. This ensures safe concurrent access from multi-threaded training scripts:

```python
import threading

class DatasetRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._entries = {}

    def register(self, name, ...):
        with self._lock:
            ...

    def get_loader(self, name, ...):
        with self._lock:
            entry = self._entries[name]
        # Instantiation happens outside the lock
        return entry.loader_cls(config, mode)
```

---

## 9. Error Handling

### 9.1 Common Errors

| Error | Cause | Resolution |
|---|---|---|
| `KeyError` | Dataset name not found in registry | Check spelling, list available datasets |
| `ValueError` | Invalid phase, duplicate name, or bad config | Fix the invalid parameter |
| `FileNotFoundError` | Dataset not downloaded | Call `registry.download(name)` first |
| `ChecksumError` | Downloaded file corrupted | Delete and re-download |
| `ImportError` | Loader class not importable | Install required dependencies |
| `TimeoutError` | Download timed out | Retry or check network connection |

### 9.2 Graceful Degradation

When a production dataset is unavailable (network error, missing auth), the registry can fall back to synthetic data:

```python
try:
    loader = registry.get_loader("imagenet21k", mode="production")
except (FileNotFoundError, TimeoutError):
    logger.warning("ImageNet unavailable, falling back to synthetic data")
    loader = registry.get_loader("synthetic_snn", mode="dev")
```

This behavior is opt-in via a `fallback=True` parameter to `get_loader()`.
