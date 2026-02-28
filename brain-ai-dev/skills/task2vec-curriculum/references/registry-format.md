# Task Embedding Registry Format

## Overview

The Task Embedding Registry is a persistent artifact that maps each `task_id` to its Fisher-derived embedding vector and extraction metadata. Store it alongside model checkpoints so that any training run can reproduce its curriculum decisions, replay its task orderings, and visualize its task space without re-extracting embeddings from scratch.

The registry serves four distinct purposes, each placing different demands on the format:

1. **Reproducibility.** Given a checkpoint, reconstruct the exact set of task embeddings that informed curriculum ordering and meta-batch composition during training. This requires that every embedding be paired with the probe signature, extraction seed, and code version that produced it.

2. **Offline analysis.** After training completes, load the registry into a notebook or analysis script to visualize task clusters, compute inter-task distances, and correlate task difficulty with meta-learning performance. This requires efficient batch access to embedding vectors and filterable metadata.

3. **Curriculum replay.** Resume training from a checkpoint and regenerate the same curriculum ordering without re-running Fisher extraction on the full task set. This requires the registry to be a complete, self-contained record of all previously extracted embeddings.

4. **Cross-run comparison.** Compare task spaces across training runs that used different probe networks, datasets, or extraction parameters. This requires explicit versioning and probe signature tracking so that incompatible embeddings are never silently mixed.

Version the registry format explicitly. Every registry file begins with a header line containing the format version. Loaders check this version and refuse to load incompatible formats rather than silently misinterpreting fields.

---

## Storage Format

Use a dual-file format: `task2vec_registry.jsonl` for metadata and `embeddings.npz` for embedding vectors. This separation exists for practical reasons that a single-file format cannot satisfy simultaneously.

### Why Dual Files

**Metadata must be human-readable.** During debugging, an engineer needs to grep for a specific task, inspect its extraction parameters, or count entries by dataset. JSONL (one JSON object per line) supports all of these operations with standard Unix tools and text editors. Searching a binary format requires custom tooling.

**Vectors must be compact.** A registry with 10,000 tasks at 512 dimensions produces 5,120,000 float32 values (approximately 20 MB uncompressed). Storing these as JSON doubles the size and adds parsing overhead. NumPy's compressed archive format (NPZ) stores the same data in roughly 8-12 MB with sub-millisecond load times.

**Independent access patterns.** Metadata queries (filter by dataset, count by split) never touch the embedding matrix. Nearest-neighbor searches over embeddings never parse metadata beyond `task_id`. Separating the files lets each access pattern use the optimal format.

### JSONL File: `task2vec_registry.jsonl`

One JSON object per line. The first line is a header containing registry-level metadata. All subsequent lines are task entries.

```
{"_version": "1.0", "_probe_signature": "conv4_last_block_e512_abc123", "_created": "2026-02-20T14:30:00Z"}
{"task_id": "a1b2c3d4", "dataset": "omniglot", "split": "train", "n_way": 5, ...}
{"task_id": "e5f6g7h8", "dataset": "omniglot", "split": "train", "n_way": 5, ...}
```

The header line is identified by the presence of the `_version` key. Loaders must detect this key and treat the line as metadata rather than a task entry.

### NPZ File: `embeddings.npz`

A NumPy compressed archive containing a single key:

```python
# On save:
np.savez_compressed("embeddings.npz", embeddings=embedding_matrix)

# On load:
data = np.load("embeddings.npz")
embedding_matrix = data["embeddings"]  # shape: (N_tasks, E)
```

The matrix has shape `(N_tasks, E)` where `N_tasks` is the number of task entries in the JSONL file (excluding the header) and `E` is the embedding dimension (typically 512). Row `i` in the matrix corresponds to the task entry at JSONL line `i + 1` (0-indexed after the header). Each task entry includes a `row_index` field that explicitly records this mapping, providing a redundant check against file corruption or out-of-order appends.

### Alternative: Parquet (Optional)

For teams that prefer integrated storage, a single Parquet file can hold both metadata columns and the embedding vector as a fixed-length binary column. This is useful for large-scale analysis with tools like DuckDB or Polars but sacrifices the human-readability advantage of JSONL.

```python
import pyarrow as pa
import pyarrow.parquet as pq

schema = pa.schema([
    ("task_id", pa.string()),
    ("dataset", pa.string()),
    ("split", pa.string()),
    ("n_way", pa.int32()),
    ("k_shot", pa.int32()),
    ("q_query", pa.int32()),
    ("embedding", pa.list_(pa.float32(), E)),
    # ... remaining fields
])
```

The JSONL + NPZ format is the primary format. Parquet export is a convenience utility, not a requirement.

---

## Per-Task Schema

Every task entry in the JSONL file contains the following fields. All fields are required unless marked optional.

### Field Reference

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Deterministic hash of the task identity: `sha256(dataset + split + sorted(class_ids) + sorted(support_indices) + transforms_hash)`, truncated to 16 hex characters. Two tasks with identical constituents must produce identical `task_id` values. |
| `dataset` | `str` | Dataset name, lowercase with underscores. Examples: `"omniglot"`, `"mini_imagenet"`, `"tiered_imagenet"`, `"cifar_fs"`. |
| `split` | `str` | One of `"train"`, `"val"`, `"test"`. Identifies which data partition the task was sampled from. |
| `n_way` | `int` | Number of classes in the episode. |
| `k_shot` | `int` | Number of support samples per class. |
| `q_query` | `int` | Number of query samples per class. |
| `class_ids` | `List[int]` | Sorted list of class indices selected for this episode. Length equals `n_way`. Sorting is mandatory for deterministic hashing. |
| `support_indices` | `List[int]` | Sorted list of sample indices used as the support set. Length equals `n_way * k_shot`. These are global indices into the dataset, not per-class indices. |
| `probe_signature` | `str` | Hash identifying the probe network and extraction configuration: `sha256(model_name + layer_subset + preprocessing_hash)`, truncated to 12 hex characters. Embeddings extracted with different probe signatures are not directly comparable. |
| `extraction_seed` | `int` | Random seed used during Fisher extraction. Combined with `task_id`, this fully determines the embedding value on a given device. |
| `extraction_timestamp` | `str` | ISO 8601 UTC timestamp of when the embedding was extracted. Format: `"2026-02-20T14:30:00Z"`. |
| `code_version` | `str` | Git commit hash (short, 8 characters) or semantic version string of the extraction code. Enables tracing an embedding back to the exact code that produced it. |
| `diagnostics` | `Dict` | Extraction diagnostics (see below). |
| `row_index` | `int` | Zero-based index into the `embeddings.npz` matrix. Row `row_index` of the embedding matrix contains this task's embedding vector. |

### Diagnostics Sub-Object

The `diagnostics` field contains extraction-time measurements that characterize embedding quality without requiring the embedding vector itself.

| Key | Type | Description |
|---|---|---|
| `fisher_norm` | `float` | L2 norm of the raw (pre-normalization) Fisher information vector. Tracks the overall magnitude of parameter sensitivity. Unusually low values indicate a degenerate probe or trivial task. |
| `sparsity` | `float` | Fraction of Fisher diagonal entries below 1e-8. High sparsity (> 0.95) suggests the probe has many dead parameters for this task. |
| `probe_loss` | `float` | Cross-entropy loss of the frozen probe on the support set. Serves as a difficulty proxy: higher loss means the probe found the task harder. |
| `extraction_time_ms` | `float` | Wall-clock time for the full extraction pipeline (forward pass + Fisher accumulation + projection), in milliseconds. Useful for throughput benchmarking. |

### Example Entry

```json
{
  "task_id": "a1b2c3d4e5f6g7h8",
  "dataset": "omniglot",
  "split": "train",
  "n_way": 5,
  "k_shot": 1,
  "q_query": 15,
  "class_ids": [23, 45, 112, 387, 901],
  "support_indices": [1023, 2045, 5601, 17823, 44012],
  "probe_signature": "conv4_lb_e512",
  "extraction_seed": 42,
  "extraction_timestamp": "2026-02-20T14:30:00Z",
  "code_version": "c6db320a",
  "diagnostics": {
    "fisher_norm": 145.32,
    "sparsity": 0.23,
    "probe_loss": 1.61,
    "extraction_time_ms": 47.3
  },
  "row_index": 0
}
```

---

## Registry Operations

The `TaskEmbeddingRegistry` class exposes six core operations. Each operation has a defined contract and failure mode.

### `update_registry(task_id, embedding, meta)`

Insert or replace a task entry. If `task_id` already exists, overwrite both the metadata and the embedding vector.

```python
def update_registry(
    self,
    task_id: str,
    embedding: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    """
    Add or update a task embedding in the registry.

    Args:
        task_id: Deterministic task hash.
        embedding: L2-normalized embedding vector, shape (E,).
        meta: Metadata dict matching the per-task schema.

    Raises:
        ValueError: If embedding.shape != (self.embedding_dim,).
        ValueError: If task_id != meta["task_id"].
    """
```

On insert, append a new row to the embedding matrix and a new line to the in-memory metadata list. On overwrite, replace the existing row in-place without changing `row_index` assignments for other entries.

### `get_embedding(task_id) -> np.ndarray`

Retrieve a single embedding vector by task ID.

```python
def get_embedding(self, task_id: str) -> np.ndarray:
    """
    Return the embedding vector for a single task.

    Args:
        task_id: The task to look up.

    Returns:
        np.ndarray of shape (E,), L2-normalized.

    Raises:
        KeyError: If task_id is not in the registry.
    """
```

### `get_embeddings(task_ids) -> np.ndarray`

Retrieve a batch of embedding vectors. Return order matches the input order.

```python
def get_embeddings(self, task_ids: List[str]) -> np.ndarray:
    """
    Return a batch of embedding vectors.

    Args:
        task_ids: List of task IDs to retrieve.

    Returns:
        np.ndarray of shape (len(task_ids), E).

    Raises:
        KeyError: If any task_id is not in the registry.
    """
```

Implement this as a single NumPy fancy-index operation on the embedding matrix, not a loop over `get_embedding`. For 10,000 tasks, the vectorized path is 100x faster.

### `query(dataset, split, n_way) -> List[str]`

Filter task IDs by metadata fields. All filter arguments are optional; omitted arguments match everything.

```python
def query(
    self,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    n_way: Optional[int] = None,
    k_shot: Optional[int] = None,
) -> List[str]:
    """
    Return task_ids matching all specified filters.

    Args:
        dataset: Filter by dataset name (exact match).
        split: Filter by split (exact match).
        n_way: Filter by number of classes.
        k_shot: Filter by shots per class.

    Returns:
        List of matching task_ids, in insertion order.
    """
```

For registries under 100,000 entries, linear scan over the metadata list is sufficient (sub-millisecond). For larger registries, build secondary index dicts on first query and cache them.

### `save(path)` and `load(path)`

Serialize and deserialize the registry to/from disk.

```python
def save(self, path: str) -> None:
    """
    Save registry to disk as task2vec_registry.jsonl + embeddings.npz.

    Args:
        path: Directory path. Creates the directory if it does not exist.
              Writes two files: {path}/task2vec_registry.jsonl
                                {path}/embeddings.npz
    """

@classmethod
def load(cls, path: str) -> "TaskEmbeddingRegistry":
    """
    Load registry from disk.

    Args:
        path: Directory path containing task2vec_registry.jsonl and embeddings.npz.

    Returns:
        A populated TaskEmbeddingRegistry instance.

    Raises:
        FileNotFoundError: If either file is missing.
        ValueError: If format version is incompatible.
        RuntimeWarning: If probe_signature differs across entries.
    """
```

### `merge(other_registry)`

Combine two registries. Use this when merging results from parallel extraction jobs or combining registries from different dataset splits.

```python
def merge(self, other: "TaskEmbeddingRegistry") -> None:
    """
    Merge another registry into this one.

    For duplicate task_ids:
      - If probe_signature matches, keep the entry with the later timestamp.
      - If probe_signature differs, keep both (multi-probe registry).

    Args:
        other: Registry to merge into self.
    """
```

---

## Checkpoint Integration

### Directory Layout

Save the registry as a subdirectory of the checkpoint directory. This co-locates the registry with the model state it corresponds to.

```
checkpoints/
  meta_maml_omniglot_5w1s_epoch0042.pt
  meta_maml_omniglot_best.pt -> meta_maml_omniglot_5w1s_epoch0042.pt
  task2vec_registry/
    task2vec_registry.jsonl
    embeddings.npz
```

The path convention is `{checkpoint_dir}/task2vec_registry/`. The training script creates this directory on first registry save and updates it at each checkpoint.

### Saving During Training

Save the registry at the same frequency as model checkpoints. The registry is append-only during training (new tasks are extracted as training progresses), so saving is cheap -- only new entries since the last save need to be written.

```python
# In the Phase 7 training loop:
for epoch in range(start_epoch, max_epochs):
    for episode in episode_sampler:
        # Extract embedding if not already in registry
        if episode.task_id not in registry:
            embedding = extractor.extract(episode, seed=extraction_seed)
            registry.update_registry(
                task_id=episode.task_id,
                embedding=embedding.embedding,
                meta=embedding.to_meta_dict(),
            )

        # ... meta-training step ...

    # Save checkpoint + registry together
    save_meta_checkpoint(checkpoint_path, model, optimizer, ...)
    registry.save(os.path.join(checkpoint_dir, "task2vec_registry"))
```

### Loading During Resume

Load the registry alongside the model checkpoint. Validate that the registry's probe signature matches the current extraction configuration.

```python
# Resume from checkpoint:
ckpt = load_meta_checkpoint(args.resume, model, optimizer, ...)
registry = TaskEmbeddingRegistry.load(
    os.path.join(os.path.dirname(args.resume), "task2vec_registry")
)

# Validate probe signature consistency
current_probe_sig = compute_probe_signature(config.task2vec)
registry_probe_sig = registry.header.get("_probe_signature")
if registry_probe_sig and registry_probe_sig != current_probe_sig:
    warnings.warn(
        f"Registry probe_signature '{registry_probe_sig}' differs from "
        f"current config probe_signature '{current_probe_sig}'. "
        f"Existing embeddings may not be comparable to new extractions."
    )
```

### Version Validation on Load

The loader reads the first line of the JSONL file, parses the `_version` field, and validates compatibility before processing any task entries.

```python
def _validate_header(self, header: Dict[str, Any]) -> None:
    version = header.get("_version")
    if version is None:
        raise ValueError(
            "Registry JSONL missing _version header. "
            "This file predates the versioned format."
        )
    major = int(version.split(".")[0])
    if major > SUPPORTED_MAJOR_VERSION:
        raise ValueError(
            f"Registry format version {version} is newer than "
            f"supported version {SUPPORTED_MAJOR_VERSION}.x. "
            f"Upgrade the task2vec module."
        )
```

### Probe Signature Consistency Warning

A single registry may contain entries with different probe signatures if the probe configuration changed mid-training or if two registries were merged. Detect this condition and warn rather than error, since multi-probe registries are valid in some workflows (e.g., comparing embeddings from different probes).

```python
def _check_probe_consistency(self) -> None:
    signatures = set(entry["probe_signature"] for entry in self._entries)
    if len(signatures) > 1:
        warnings.warn(
            f"Registry contains {len(signatures)} distinct probe signatures: "
            f"{signatures}. Embeddings from different probes are not directly "
            f"comparable. Filter by probe_signature before computing distances.",
            RuntimeWarning,
        )
```

---

## Versioning and Compatibility

### Format Version Header

The first line of every JSONL file is a header containing three fields:

```json
{"_version": "1.0", "_probe_signature": "conv4_lb_e512_abc123", "_created": "2026-02-20T14:30:00Z"}
```

| Field | Type | Description |
|---|---|---|
| `_version` | `str` | Semantic version of the registry format. Major version changes indicate breaking schema changes. Minor version changes indicate backward-compatible additions. |
| `_probe_signature` | `str` | Probe signature of the initial extraction run. Serves as a default expectation; individual entries may differ in multi-probe registries. |
| `_created` | `str` | ISO 8601 UTC timestamp of when the registry was first created. |

### Backward Compatibility

When loading a registry with a minor version newer than the loader's version (e.g., loader supports 1.0, file is 1.2), the loader must handle missing fields by substituting defaults. Define a defaults map for each version increment:

```python
FIELD_DEFAULTS = {
    # Fields added in version 1.1:
    "q_query": 15,
    "code_version": "unknown",
    # Fields added in version 1.2:
    "diagnostics": {
        "fisher_norm": 0.0,
        "sparsity": 0.0,
        "probe_loss": 0.0,
        "extraction_time_ms": 0.0,
    },
}

def _apply_defaults(self, entry: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing fields with version-appropriate defaults."""
    for field, default in FIELD_DEFAULTS.items():
        if field not in entry:
            entry[field] = default
    return entry
```

### Forward Compatibility

When loading a registry that contains fields the loader does not recognize, ignore those fields silently. Do not raise an error or warning for unknown keys. This allows newer writers to add fields without breaking older loaders.

```python
# Correct: access only known fields, ignore unknown ones
task_id = entry["task_id"]
dataset = entry["dataset"]
# Unknown fields like entry["new_field_from_v2"] are simply not accessed

# Wrong: validating that no unknown fields exist
# assert set(entry.keys()) <= KNOWN_FIELDS  # DO NOT DO THIS
```

### Probe Signature Mismatch Policy

When entries in a registry have differing `probe_signature` values, issue a warning but do not raise an error. Multi-probe registries arise in two legitimate scenarios:

1. The extraction configuration changed during a long training run (e.g., switching from `conv4` to `resnet12` probe after initial experiments).
2. Two registries from different runs were merged for comparative analysis.

Operations that require comparable embeddings (nearest-neighbor, clustering, curriculum ordering) should filter by `probe_signature` before proceeding:

```python
# Filter to a single probe before computing distances
target_sig = "conv4_lb_e512_abc123"
task_ids = registry.query(dataset="omniglot")
consistent_ids = [
    tid for tid in task_ids
    if registry.get_meta(tid)["probe_signature"] == target_sig
]
embeddings = registry.get_embeddings(consistent_ids)
distances = 1.0 - cosine_similarity(embeddings)
```

---

## Implementation Patterns

### TaskEmbeddingRegistry Class

```python
import json
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


SUPPORTED_MAJOR_VERSION = 1


class TaskEmbeddingRegistry:
    """
    Persistent mapping from task_id to embedding vector + metadata.

    Stores embeddings in a dense NumPy matrix for efficient batch access
    and metadata in an ordered dict for fast lookup and filtering.
    """

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self._entries: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._embeddings: List[np.ndarray] = []
        self._id_to_row: Dict[str, int] = {}
        self._embedding_matrix: Optional[np.ndarray] = None  # lazy cache
        self._dirty = False  # True when _embeddings changed since last matrix build
        self.header: Dict[str, Any] = {
            "_version": "1.0",
            "_probe_signature": None,
            "_created": datetime.now(timezone.utc).isoformat(),
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._entries

    @property
    def task_ids(self) -> List[str]:
        return list(self._entries.keys())

    # ── Core Operations ──────────────────────────────────────────

    def update_registry(
        self,
        task_id: str,
        embedding: np.ndarray,
        meta: Dict[str, Any],
    ) -> None:
        if embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"Expected embedding shape ({self.embedding_dim},), "
                f"got {embedding.shape}"
            )
        if meta.get("task_id") and meta["task_id"] != task_id:
            raise ValueError(
                f"task_id mismatch: argument '{task_id}' vs "
                f"meta '{meta['task_id']}'"
            )

        if task_id in self._id_to_row:
            # Overwrite existing entry
            row_idx = self._id_to_row[task_id]
            self._embeddings[row_idx] = embedding.copy()
            meta["row_index"] = row_idx
            self._entries[task_id] = meta
        else:
            # Append new entry
            row_idx = len(self._embeddings)
            self._embeddings.append(embedding.copy())
            self._id_to_row[task_id] = row_idx
            meta["row_index"] = row_idx
            self._entries[task_id] = meta

        self._dirty = True

    def get_embedding(self, task_id: str) -> np.ndarray:
        if task_id not in self._id_to_row:
            raise KeyError(f"Task '{task_id}' not in registry")
        row_idx = self._id_to_row[task_id]
        return self._embeddings[row_idx].copy()

    def get_embeddings(self, task_ids: List[str]) -> np.ndarray:
        indices = []
        for tid in task_ids:
            if tid not in self._id_to_row:
                raise KeyError(f"Task '{tid}' not in registry")
            indices.append(self._id_to_row[tid])
        matrix = self._get_matrix()
        return matrix[indices]

    def get_meta(self, task_id: str) -> Dict[str, Any]:
        if task_id not in self._entries:
            raise KeyError(f"Task '{task_id}' not in registry")
        return self._entries[task_id]

    def query(
        self,
        dataset: Optional[str] = None,
        split: Optional[str] = None,
        n_way: Optional[int] = None,
        k_shot: Optional[int] = None,
    ) -> List[str]:
        results = []
        for task_id, entry in self._entries.items():
            if dataset is not None and entry.get("dataset") != dataset:
                continue
            if split is not None and entry.get("split") != split:
                continue
            if n_way is not None and entry.get("n_way") != n_way:
                continue
            if k_shot is not None and entry.get("k_shot") != k_shot:
                continue
            results.append(task_id)
        return results

    # ── Persistence ──────────────────────────────────────────────

    def save(self, path: str) -> None:
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Write JSONL
        jsonl_path = dir_path / "task2vec_registry.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(self.header) + "\n")
            for entry in self._entries.values():
                f.write(json.dumps(entry, default=_json_default) + "\n")

        # Write NPZ
        npz_path = dir_path / "embeddings.npz"
        matrix = self._get_matrix()
        np.savez_compressed(str(npz_path), embeddings=matrix)

    @classmethod
    def load(cls, path: str) -> "TaskEmbeddingRegistry":
        dir_path = Path(path)
        jsonl_path = dir_path / "task2vec_registry.jsonl"
        npz_path = dir_path / "embeddings.npz"

        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing {jsonl_path}")
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing {npz_path}")

        # Load embeddings
        data = np.load(str(npz_path))
        embedding_matrix = data["embeddings"]
        embedding_dim = embedding_matrix.shape[1]

        registry = cls(embedding_dim=embedding_dim)

        # Parse JSONL
        with open(jsonl_path, "r") as f:
            lines = f.readlines()

        if not lines:
            raise ValueError("Empty registry JSONL file")

        # First line is the header
        header = json.loads(lines[0])
        if "_version" not in header:
            raise ValueError("Registry JSONL missing _version header")
        registry.header = header
        registry._validate_header(header)

        # Remaining lines are task entries
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entry = registry._apply_defaults(entry)
            task_id = entry["task_id"]
            row_idx = entry["row_index"]

            registry._entries[task_id] = entry
            registry._id_to_row[task_id] = row_idx
            registry._embeddings.append(embedding_matrix[row_idx].copy())

        # Validate row count
        if len(registry._entries) != embedding_matrix.shape[0]:
            warnings.warn(
                f"JSONL has {len(registry._entries)} entries but NPZ has "
                f"{embedding_matrix.shape[0]} rows. Possible corruption.",
                RuntimeWarning,
            )

        registry._check_probe_consistency()
        return registry

    # ── Merge ────────────────────────────────────────────────────

    def merge(self, other: "TaskEmbeddingRegistry") -> None:
        for task_id, entry in other._entries.items():
            other_emb = other.get_embedding(task_id)
            if task_id in self._entries:
                existing = self._entries[task_id]
                same_probe = (
                    existing.get("probe_signature")
                    == entry.get("probe_signature")
                )
                if same_probe:
                    # Keep the entry with the later timestamp
                    existing_ts = existing.get("extraction_timestamp", "")
                    other_ts = entry.get("extraction_timestamp", "")
                    if other_ts > existing_ts:
                        self.update_registry(task_id, other_emb, entry)
                else:
                    # Different probe: suffix the task_id to keep both
                    alt_id = f"{task_id}_{entry.get('probe_signature', 'alt')}"
                    entry_copy = dict(entry)
                    entry_copy["task_id"] = alt_id
                    self.update_registry(alt_id, other_emb, entry_copy)
            else:
                self.update_registry(task_id, other_emb, entry)

    # ── Internal Helpers ─────────────────────────────────────────

    def _get_matrix(self) -> np.ndarray:
        if self._dirty or self._embedding_matrix is None:
            if self._embeddings:
                self._embedding_matrix = np.stack(self._embeddings, axis=0)
            else:
                self._embedding_matrix = np.empty(
                    (0, self.embedding_dim), dtype=np.float32
                )
            self._dirty = False
        return self._embedding_matrix

    def _validate_header(self, header: Dict[str, Any]) -> None:
        version = header.get("_version", "0.0")
        major = int(version.split(".")[0])
        if major > SUPPORTED_MAJOR_VERSION:
            raise ValueError(
                f"Registry format version {version} is newer than "
                f"supported version {SUPPORTED_MAJOR_VERSION}.x"
            )

    def _apply_defaults(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        defaults = {
            "q_query": 15,
            "code_version": "unknown",
            "diagnostics": {
                "fisher_norm": 0.0,
                "sparsity": 0.0,
                "probe_loss": 0.0,
                "extraction_time_ms": 0.0,
            },
        }
        for field, default in defaults.items():
            if field not in entry:
                entry[field] = default
        return entry

    def _check_probe_consistency(self) -> None:
        signatures = set(
            entry.get("probe_signature", "")
            for entry in self._entries.values()
        )
        signatures.discard("")
        if len(signatures) > 1:
            warnings.warn(
                f"Registry contains {len(signatures)} distinct probe "
                f"signatures: {signatures}. Embeddings from different "
                f"probes are not directly comparable.",
                RuntimeWarning,
            )


def _json_default(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
```

### Efficient Querying with Index Structures

For registries exceeding 100,000 entries, build secondary indices on first query to avoid repeated linear scans.

```python
class IndexedRegistry(TaskEmbeddingRegistry):
    """
    Extended registry with secondary indices for fast metadata queries.
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__(embedding_dim)
        self._index_by_dataset: Dict[str, List[str]] = {}
        self._index_by_split: Dict[str, List[str]] = {}
        self._index_by_nway: Dict[int, List[str]] = {}
        self._indices_built = False

    def _build_indices(self) -> None:
        self._index_by_dataset.clear()
        self._index_by_split.clear()
        self._index_by_nway.clear()

        for task_id, entry in self._entries.items():
            ds = entry.get("dataset", "")
            self._index_by_dataset.setdefault(ds, []).append(task_id)

            sp = entry.get("split", "")
            self._index_by_split.setdefault(sp, []).append(task_id)

            nw = entry.get("n_way", 0)
            self._index_by_nway.setdefault(nw, []).append(task_id)

        self._indices_built = True

    def query(
        self,
        dataset: Optional[str] = None,
        split: Optional[str] = None,
        n_way: Optional[int] = None,
        k_shot: Optional[int] = None,
    ) -> List[str]:
        if not self._indices_built:
            self._build_indices()

        # Start with the most selective filter
        candidates = None
        if dataset is not None:
            candidates = set(self._index_by_dataset.get(dataset, []))
        if split is not None:
            split_set = set(self._index_by_split.get(split, []))
            candidates = split_set if candidates is None else candidates & split_set
        if n_way is not None:
            nway_set = set(self._index_by_nway.get(n_way, []))
            candidates = nway_set if candidates is None else candidates & nway_set

        if candidates is None:
            candidates = set(self._entries.keys())

        # Apply remaining filters (k_shot) via linear scan over candidates
        if k_shot is not None:
            candidates = {
                tid for tid in candidates
                if self._entries[tid].get("k_shot") == k_shot
            }

        # Preserve insertion order
        return [tid for tid in self._entries if tid in candidates]
```

### Incremental Updates (Append Without Rewriting)

During training, new tasks are continuously extracted and added to the registry. Avoid rewriting the entire JSONL file on every update. Instead, open the file in append mode and write only the new entries.

```python
class IncrementalWriter:
    """
    Writes new registry entries to JSONL incrementally.

    Opens the file in append mode and writes one line per new entry.
    The NPZ file is rewritten on each flush (unavoidable for compressed
    arrays), but this is infrequent (once per checkpoint).
    """

    def __init__(self, path: str, registry: TaskEmbeddingRegistry):
        self._path = Path(path)
        self._registry = registry
        self._jsonl_path = self._path / "task2vec_registry.jsonl"
        self._npz_path = self._path / "embeddings.npz"
        self._last_saved_count = 0

        # Write header if file does not exist
        if not self._jsonl_path.exists():
            self._path.mkdir(parents=True, exist_ok=True)
            with open(self._jsonl_path, "w") as f:
                f.write(json.dumps(registry.header) + "\n")
            self._last_saved_count = 0

    def append_new_entries(self) -> int:
        """
        Append entries added since the last save.

        Returns the number of new entries written.
        """
        all_entries = list(self._registry._entries.values())
        new_entries = all_entries[self._last_saved_count:]

        if not new_entries:
            return 0

        with open(self._jsonl_path, "a") as f:
            for entry in new_entries:
                f.write(json.dumps(entry, default=_json_default) + "\n")

        self._last_saved_count = len(all_entries)
        return len(new_entries)

    def flush_embeddings(self) -> None:
        """
        Rewrite the NPZ file with the current embedding matrix.

        Call this at checkpoint time, not after every individual update.
        """
        matrix = self._registry._get_matrix()
        np.savez_compressed(str(self._npz_path), embeddings=matrix)
```

### Registry Diff (Compare Two Registries)

Compare two registries to identify tasks that were added, removed, or changed between training runs.

```python
def diff_registries(
    registry_a: TaskEmbeddingRegistry,
    registry_b: TaskEmbeddingRegistry,
    cosine_threshold: float = 0.9999,
) -> Dict[str, Any]:
    """
    Compare two registries and report differences.

    Args:
        registry_a: The reference (older) registry.
        registry_b: The comparison (newer) registry.
        cosine_threshold: Embeddings with cosine similarity above this
            threshold are considered identical.

    Returns:
        Dict with keys:
            "added": List[str] -- task_ids in B but not A
            "removed": List[str] -- task_ids in A but not B
            "changed": List[Dict] -- task_ids in both with different embeddings
            "identical": int -- count of task_ids with matching embeddings
            "metadata_diffs": List[Dict] -- task_ids where metadata changed
    """
    ids_a = set(registry_a.task_ids)
    ids_b = set(registry_b.task_ids)

    added = sorted(ids_b - ids_a)
    removed = sorted(ids_a - ids_b)
    common = sorted(ids_a & ids_b)

    changed = []
    identical_count = 0
    metadata_diffs = []

    for task_id in common:
        emb_a = registry_a.get_embedding(task_id)
        emb_b = registry_b.get_embedding(task_id)

        # Cosine similarity
        cos_sim = float(
            np.dot(emb_a, emb_b)
            / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-12)
        )

        if cos_sim < cosine_threshold:
            changed.append({
                "task_id": task_id,
                "cosine_similarity": cos_sim,
            })
        else:
            identical_count += 1

        # Check metadata differences
        meta_a = registry_a.get_meta(task_id)
        meta_b = registry_b.get_meta(task_id)
        meta_diff_fields = []
        for key in set(meta_a.keys()) | set(meta_b.keys()):
            if key in ("row_index", "extraction_timestamp"):
                continue  # Expected to differ
            if meta_a.get(key) != meta_b.get(key):
                meta_diff_fields.append(key)
        if meta_diff_fields:
            metadata_diffs.append({
                "task_id": task_id,
                "differing_fields": meta_diff_fields,
            })

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "identical": identical_count,
        "metadata_diffs": metadata_diffs,
    }
```

---

## Invariants and Failure Modes

### Critical Invariants

1. **Row index consistency.** For every task entry, `entry["row_index"]` must be a valid index into the embedding matrix, and the vector at that index must be the embedding for that task. Violating this invariant silently returns wrong embeddings. Validate on load by checking `len(entries) == matrix.shape[0]`.

2. **Task ID determinism.** Two tasks with identical `(dataset, split, class_ids, support_indices, transforms)` must produce identical `task_id` values. Violating this creates duplicate entries for the same task. Enforce by computing the hash from sorted, canonical inputs.

3. **Embedding normalization.** All embeddings in the registry must be L2-normalized. Violating this makes cosine distance computations meaningless. Validate on insert: `assert abs(np.linalg.norm(embedding) - 1.0) < 1e-5`.

4. **Header line position.** The header must be the first line of the JSONL file. Appending entries must never insert before the header. The incremental writer opens in append mode, which guarantees this.

### Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError` on `get_embedding` | Task was not extracted before querying | Check `task_id in registry` before access; extract missing tasks on demand |
| Row count mismatch between JSONL and NPZ | Crash during save left files inconsistent | Rewrite both files from in-memory state; use atomic save (write to temp, then rename) |
| Embeddings silently wrong | `row_index` out of sync after entry deletion | Never delete entries; mark as inactive in metadata instead |
| Slow queries on large registries | Linear scan over 100K+ entries | Use `IndexedRegistry` with secondary indices |
| NPZ file grows without bound | Never pruning old/unused embeddings | Implement a compaction pass that removes entries not referenced by any checkpoint |
| Merge produces duplicates | Same task extracted with different probe signatures | Expected behavior; filter by `probe_signature` before distance computation |

---

## Atomic Save Pattern

Protect against partial writes by saving to temporary files first, then atomically renaming.

```python
import tempfile
import shutil


def atomic_save(registry: TaskEmbeddingRegistry, path: str) -> None:
    """
    Save registry with crash safety.

    Write to a temporary directory first, then rename into place.
    On POSIX systems, os.rename is atomic within the same filesystem.
    """
    dir_path = Path(path)
    parent = dir_path.parent

    with tempfile.TemporaryDirectory(dir=str(parent)) as tmp_dir:
        registry.save(tmp_dir)

        # Atomic swap
        backup_path = str(dir_path) + ".backup"
        if dir_path.exists():
            shutil.move(str(dir_path), backup_path)
        shutil.move(tmp_dir, str(dir_path))
        if Path(backup_path).exists():
            shutil.rmtree(backup_path)
```

---

## Size Estimates

| Registry Size | JSONL Size | NPZ Size (E=512) | Total | Load Time |
|---|---|---|---|---|
| 1,000 tasks | ~500 KB | ~2 MB | ~2.5 MB | <100 ms |
| 10,000 tasks | ~5 MB | ~20 MB | ~25 MB | ~200 ms |
| 100,000 tasks | ~50 MB | ~200 MB | ~250 MB | ~2 s |
| 1,000,000 tasks | ~500 MB | ~2 GB | ~2.5 GB | ~15 s |

For registries exceeding 100,000 tasks, consider sharding by dataset or split into separate registry directories. The `merge` operation can recombine shards when needed for cross-dataset analysis.
