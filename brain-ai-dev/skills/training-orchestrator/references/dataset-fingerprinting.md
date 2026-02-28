# Dataset Fingerprinting

## Overview

Dataset identity is a first-class concern in the seven-phase training pipeline. Each phase may consume different datasets -- MNIST for dev-mode smoke tests, ImageNet-21k for production encoder training, Omniglot and mini-ImageNet for Phase 7 meta-learning -- and the same dataset may be subsetted, transformed, or resharded between runs. Without tracked provenance, identical code and seeds still produce different results when data changes silently.

The fingerprinting system assigns a deterministic identity to every dataset consumed during training. It answers three questions: (1) what data was used, (2) can the same data be retrieved, and (3) did the data change between phases or runs. All fingerprints are written to `artifacts/datasets.json` inside the run directory. The `DatasetFingerprinter` in `brain_ai/training/dataset_fingerprint.py` implements all tiers and exposes a single `fingerprint()` entry point with automatic tier selection.

---

## Tiered Fingerprinting Strategy

### Tier 1: HuggingFace Datasets

Use Tier 1 when the dataset is loaded via the `datasets` library. Record:

| Field | Source | Purpose |
|---|---|---|
| `dataset_name` | `dataset.info.dataset_name` or load arg | Canonical identifier |
| `config_name` | `dataset.info.config_name` | Configuration variant |
| `split` | Load argument or `dataset.split` | Data split |
| `version` | `dataset.info.version` | Semantic version |
| `hf_fingerprint` | `dataset._fingerprint` | HF internal fingerprint |
| `num_rows` | `len(dataset)` | Row count sanity check |
| `features_hash` | SHA256 of `str(dataset.info.features)` | Schema identity |

```python
import hashlib

def fingerprint_hf_dataset(dataset, name: str, split: str) -> dict:
    info = dataset.info
    features_str = str(info.features) if info.features else ""
    features_hash = hashlib.sha256(features_str.encode()).hexdigest()
    return {
        "fingerprint_tier": "tier1",
        "dataset_name": name,
        "config_name": getattr(info, "config_name", None),
        "split": split,
        "version": str(info.version) if info.version else None,
        "hf_fingerprint": getattr(dataset, "_fingerprint", None),
        "num_rows": len(dataset),
        "features_hash": f"sha256:{features_hash}",
    }
```

Tier 1 has near-zero overhead -- metadata only, no file I/O.

### Tier 2: Local Files and Shards

Use Tier 2 for local files (image directories, numpy archives, `.pt` files). For each shard, record:

| Field | Source | Purpose |
|---|---|---|
| `relative_path` | Path relative to dataset root | Location identity |
| `byte_size` | `os.path.getsize()` | Size check |
| `mtime` | `os.path.getmtime()`, ISO 8601 | Change detection |
| `fast_hash` | SHA256 of first 4MB + last 4MB | Content identity |

Optionally compute full SHA256 for datasets under 1 GB via `full_hash=True`.

```python
import os
from datetime import datetime, timezone

def fingerprint_local_files(root: str, file_patterns=("*.pt", "*.npy", "*.csv", "*.tar"),
                            full_hash: bool = False) -> dict:
    import glob
    shards = []
    for pattern in file_patterns:
        for filepath in sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True)):
            rel_path = os.path.relpath(filepath, root)
            byte_size = os.path.getsize(filepath)
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath), tz=timezone.utc).isoformat()
            entry = {"relative_path": rel_path, "byte_size": byte_size,
                     "mtime": mtime, "fast_hash": f"sha256:{fast_hash_file(filepath)}"}
            if full_hash and byte_size <= 1_073_741_824:
                entry["full_hash"] = f"sha256:{full_hash_file(filepath)}"
            shards.append(entry)
    return {"fingerprint_tier": "tier2", "root": os.path.abspath(root),
            "num_files": len(shards), "total_bytes": sum(s["byte_size"] for s in shards),
            "shards": shards}
```

### Tier 3: Sampling Identity

Use Tier 3 for subsets of a larger dataset -- dev-mode subsets, few-shot episode definitions, curriculum batches. Record the sampling parameters rather than hashing contents:

| Field | Source | Purpose |
|---|---|---|
| `parent_fingerprint` | Tier 1/2 fingerprint of parent | Link to source |
| `sample_indices` | Sorted list or range expression | Exact subset identity |
| `rng_seed` | Seed used for sampling | Reproduction key |
| `sampling_strategy` | `"random"`, `"first_n"`, `"stratified"`, `"episode"` | Algorithm name |
| `subset_size` | Number of samples | Sanity check |

```python
def fingerprint_subset(parent_fingerprint: dict, indices: list[int],
                       seed: int, strategy: str = "random") -> dict:
    sorted_idx = sorted(indices)
    if sorted_idx == list(range(sorted_idx[0], sorted_idx[-1] + 1)):
        idx_repr = f"range({sorted_idx[0]},{sorted_idx[-1]+1})"
    else:
        idx_repr = sorted_idx
    return {"fingerprint_tier": "tier3", "parent_fingerprint": parent_fingerprint,
            "sample_indices": idx_repr, "rng_seed": seed,
            "sampling_strategy": strategy, "subset_size": len(indices)}
```

### Automatic Tier Selection

The `fingerprint()` entry point inspects the dataset object: use Tier 1 if `datasets.Dataset` is detected, Tier 2 if a directory path is passed, Tier 3 if explicit `indices` and `parent_fingerprint` are provided in kwargs.

---

## Fast Hash Algorithm

Read the first 4 MB and last 4 MB of a file, concatenate, and compute SHA256. For files under 4 MB, hash the entire contents. For files between 4-8 MB, head and tail overlap -- this is harmless and deterministic.

```python
FAST_HASH_CHUNK = 4 * 1024 * 1024  # 4 MB

def fast_hash_file(filepath: str) -> str:
    """SHA256 of first 4MB + last 4MB. < 100ms for any file size."""
    file_size = os.path.getsize(filepath)
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        h.update(f.read(FAST_HASH_CHUNK))
        if file_size > FAST_HASH_CHUNK:
            f.seek(max(0, file_size - FAST_HASH_CHUNK))
            h.update(f.read(FAST_HASH_CHUNK))
    return h.hexdigest()

def full_hash_file(filepath: str) -> str:
    """SHA256 of entire file, read in 8MB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()
```

| File Size | Fast Hash | Full Hash | Notes |
|---|---|---|---|
| < 4 MB | < 5 ms | < 5 ms | Entire file read either way |
| 100 MB | ~10 ms | ~300 ms | Fast hash reads 8 MB total |
| 1 GB | ~10 ms | ~3 s | Fast hash I/O is constant |
| 10 GB | ~15 ms | ~30 s | Seek overhead only |

---

## datasets.json Schema

Every run writes `artifacts/datasets.json`. Three examples follow.

### Tier 1 (HuggingFace)

```json
{
  "schema_version": "1.0",
  "fingerprint_tier": "tier1",
  "datasets": [
    {
      "name": "mnist",
      "role": "primary",
      "config_name": null,
      "split": "train",
      "version": "1.0.0",
      "hf_fingerprint": "abc123def456789...",
      "num_samples": 60000,
      "features_hash": "sha256:e3b0c44298fc1c14...",
      "transforms_signature": "Compose(ToTensor,Normalize(0.1307,0.3081))"
    }
  ],
  "timestamp": "2026-02-20T12:00:00Z"
}
```

### Tier 2 (Local Shards)

```json
{
  "schema_version": "1.0",
  "fingerprint_tier": "tier2",
  "datasets": [
    {
      "name": "imagenet21k",
      "role": "primary",
      "root": "/data/imagenet21k/train",
      "num_files": 128,
      "total_bytes": 137438953472,
      "shards": [
        {"relative_path": "shard_000.tar", "byte_size": 1073741824,
         "mtime": "2026-01-15T08:30:00+00:00", "fast_hash": "sha256:a1b2c3d4e5f6..."},
        {"relative_path": "shard_001.tar", "byte_size": 1073741824,
         "mtime": "2026-01-15T08:31:00+00:00", "fast_hash": "sha256:f6e5d4c3b2a1..."}
      ],
      "transforms_signature": "Compose(RandomResizedCrop(224),RandomHorizontalFlip(),ToTensor,Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]))"
    }
  ],
  "timestamp": "2026-02-20T14:30:00Z"
}
```

### Tier 3 (Dev-Mode Subset)

```json
{
  "schema_version": "1.0",
  "fingerprint_tier": "tier3",
  "datasets": [
    {
      "name": "mnist_dev_subset",
      "role": "primary",
      "parent_fingerprint": {"fingerprint_tier": "tier1", "dataset_name": "mnist",
                             "split": "train", "hf_fingerprint": "abc123def456789..."},
      "sample_indices": "range(0,1000)",
      "rng_seed": 1337,
      "sampling_strategy": "first_n",
      "subset_size": 1000,
      "transforms_signature": "Compose(ToTensor,Normalize(0.1307,0.3081))"
    }
  ],
  "timestamp": "2026-02-20T12:00:00Z"
}
```

### Required Top-Level Fields

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | Always `"1.0"` |
| `fingerprint_tier` | string | `"tier1"`, `"tier2"`, `"tier3"`, or `"mixed"` |
| `datasets` | array | One entry per dataset consumed |
| `datasets[].name` | string | Human-readable identifier |
| `datasets[].role` | string | `"primary"`, `"validation"`, `"auxiliary"` |
| `timestamp` | string (ISO 8601) | Generation time |

---

## Transform Signature

Same data with different transforms produces different behavior. Capture the transform pipeline as a deterministic string.

For `torchvision.transforms.Compose`, use `repr()`. For custom transforms, implement `__repr__` with all constructor args. Hash the string for compact comparison:

```python
def transform_signature(transform_pipeline) -> tuple[str, str]:
    """Return (readable_signature, signature_hash)."""
    if transform_pipeline is None:
        sig = "identity"
    else:
        sig = repr(transform_pipeline)
    sig_hash = hashlib.sha256(sig.encode()).hexdigest()[:16]
    return sig, f"sha256:{sig_hash}"

class CustomNormalize:
    def __init__(self, mean: float, std: float):
        self.mean, self.std = mean, std
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std
    def __repr__(self):
        return f"CustomNormalize(mean={self.mean},std={self.std})"
```

The rule: same constructor args must produce the same `repr()` string; different args must produce different strings.

---

## Cross-Phase Validation

Phase N can declare expected dataset identity from Phase N-1. Phase 4 (Global Workspace) must use the same encoder-processed data as Phase 2. The orchestrator validates this at phase startup.

### Declaring Expectations

```python
PHASE_DATASET_EXPECTATIONS = {
    4: {"primary": {"expected_name": "imagenet21k", "expected_split": "train",
                    "match_fingerprint_from_phase": 2}},
    7: {"primary": {"expected_name": ["omniglot", "mini_imagenet"],
                    "expected_split": "train", "match_fingerprint_from_phase": None}},
}
```

### Validation

```python
def validate_dataset_identity(current: dict, expected: dict,
                              prior_phase_artifacts: str = None,
                              mismatch_mode: str = "error") -> tuple[str, str]:
    if mismatch_mode == "skip":
        return "compatible", "validation skipped"

    expected_names = expected.get("expected_name")
    if isinstance(expected_names, str):
        expected_names = [expected_names]
    current_name = current.get("dataset_name") or current.get("name", "")

    if expected_names and current_name not in expected_names:
        msg = f"Name mismatch: '{current_name}' not in {expected_names}"
        if mismatch_mode == "error":
            raise ValueError(msg)
        return "incompatible", msg

    if expected.get("expected_split") and current.get("split") != expected["expected_split"]:
        msg = f"Split mismatch: '{current.get('split')}' vs '{expected['expected_split']}'"
        if mismatch_mode == "error":
            raise ValueError(msg)
        return "incompatible", msg

    match_phase = expected.get("match_fingerprint_from_phase")
    if match_phase is not None and prior_phase_artifacts:
        import json
        with open(prior_phase_artifacts) as f:
            prior_fp = json.load(f)["datasets"][0]
        return compare_fingerprints(current, prior_fp)

    return "exact", "all checks passed"
```

### Mismatch Modes

| Mode | Behavior | Use Case |
|---|---|---|
| `"error"` | Raise immediately | Production: prevent silent data drift |
| `"warn"` | Log warning, continue | Dev: dataset may change intentionally |
| `"skip"` | No validation | Exploratory runs |

Default: `"error"` in production, `"warn"` in dev (set via `ManifestConfig.dataset_mismatch_mode`).

---

## Dev Mode Dataset Identity

Dev mode uses small subsets (e.g., 1000 MNIST samples). Record the parent dataset fingerprint, subset indices, and sampling seed. Reproduce by loading the parent, sampling with the same seed, and verifying indices match.

```python
def create_dev_subset(dataset, size: int, seed: int) -> tuple:
    import torch
    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=rng)[:size].tolist()
    parent_fp = fingerprint(dataset)
    subset = torch.utils.data.Subset(dataset, indices)
    subset_fp = fingerprint_subset(parent_fp, indices, seed, strategy="random")
    return subset, subset_fp

def reproduce_subset(parent_dataset, fp: dict):
    import torch
    rng = torch.Generator()
    rng.manual_seed(fp["rng_seed"])
    if fp["sampling_strategy"] == "random":
        indices = torch.randperm(len(parent_dataset), generator=rng)[:fp["subset_size"]].tolist()
    elif fp["sampling_strategy"] == "first_n":
        indices = list(range(fp["subset_size"]))
    else:
        raise ValueError(f"Unknown strategy: {fp['sampling_strategy']}")
    return torch.utils.data.Subset(parent_dataset, indices)
```

Index storage: sorted list for subsets under 10,000 samples, `seed + size` for larger random subsets, `range(start, stop)` for contiguous subsets.

---

## Meta-Learning Episode Fingerprinting

Phase 7 treats each episode as a micro-dataset. Record the parent dataset, selected class IDs, per-class sample indices, and the episode seed. The `EpisodeSampler` from the meta-learning-suite skill provides deterministic episodes keyed by `(seed, epoch, episode_idx)`.

```python
def fingerprint_episode(dataset_name: str, class_ids: list[int],
                        support_indices: dict[int, list[int]],
                        query_indices: dict[int, list[int]],
                        episode_seed: int) -> dict:
    import json as _json
    canonical = {"dataset": dataset_name, "classes": sorted(class_ids),
                 "support": {str(k): sorted(v) for k, v in sorted(support_indices.items())},
                 "query": {str(k): sorted(v) for k, v in sorted(query_indices.items())}}
    content_hash = hashlib.sha256(
        _json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"fingerprint_tier": "episode", "dataset_name": dataset_name,
            "class_ids": sorted(class_ids), "episode_seed": episode_seed,
            "n_way": len(class_ids),
            "k_shot": len(next(iter(support_indices.values()))),
            "q_query": len(next(iter(query_indices.values()))),
            "content_hash": f"sha256:{content_hash}"}
```

For batch fingerprinting, record the sampler config plus spot-check hashes of the first N episodes:

```python
def fingerprint_episode_sampler(sampler, dataset_name: str, num_spot_check: int = 10) -> dict:
    hashes = []
    for idx in range(num_spot_check):
        ep = sampler.sample_episode(epoch=0, episode_idx=idx)
        h = hashlib.sha256(ep.support_x.numpy().tobytes() + ep.query_x.numpy().tobytes())
        hashes.append(h.hexdigest()[:16])
    return {"fingerprint_tier": "episode_sampler", "dataset_name": dataset_name,
            "seed": sampler.seed, "n_way": sampler.n_way, "k_shot": sampler.k_shot,
            "q_query": sampler.q_query, "num_available_classes": len(sampler.available_classes),
            "spot_check_hashes": hashes}
```

---

## Fingerprint Comparison

| Level | Meaning | Criteria |
|---|---|---|
| `"exact"` | Byte-identical data | All fields match |
| `"compatible"` | Same logical dataset, possibly reprocessed | Name and split match; fingerprint may differ |
| `"incompatible"` | Different dataset or split | Name or split differs |

```python
def compare_fingerprints(a: dict, b: dict) -> tuple[str, str]:
    name_a = a.get("dataset_name") or a.get("name", "")
    name_b = b.get("dataset_name") or b.get("name", "")
    if name_a != name_b:
        return "incompatible", f"name differs: '{name_a}' vs '{name_b}'"

    split_a, split_b = a.get("split", ""), b.get("split", "")
    if split_a != split_b:
        return "incompatible", f"split differs: '{split_a}' vs '{split_b}'"

    tier_a, tier_b = a.get("fingerprint_tier", ""), b.get("fingerprint_tier", "")

    if tier_a == "tier1" and tier_b == "tier1":
        if a.get("hf_fingerprint") and a["hf_fingerprint"] == b.get("hf_fingerprint"):
            return "exact", "HF fingerprints match"
        if a.get("version") != b.get("version"):
            return "compatible", f"version differs: {a.get('version')} vs {b.get('version')}"
        return "compatible", "same name/split/version, fingerprints differ"

    if tier_a == "tier2" and tier_b == "tier2":
        shards_a = {s["relative_path"]: s["fast_hash"] for s in a.get("shards", [])}
        shards_b = {s["relative_path"]: s["fast_hash"] for s in b.get("shards", [])}
        if shards_a == shards_b:
            return "exact", "all shard hashes match"
        changed = [p for p in shards_a if shards_a.get(p) != shards_b.get(p)]
        return "compatible", f"{len(changed)} shard(s) differ"

    if tier_a == "tier3" and tier_b == "tier3":
        if a.get("sample_indices") == b.get("sample_indices"):
            return "exact", "subset indices match"
        return "compatible", "same parent, different indices"

    return "compatible", f"cross-tier ({tier_a} vs {tier_b})"
```

---

## Performance Considerations

| Tier | Overhead | Scaling | Guidance |
|---|---|---|---|
| Tier 1 | < 1 ms | Constant | Always use for HF datasets |
| Tier 2 fast hash | ~10 ms/file | Linear in file count | Default for local files |
| Tier 2 full hash | ~3 s/GB | Linear in total bytes | Datasets < 1 GB or explicit |
| Tier 3 | < 1 ms | Constant | Always use for subsets |

**Caching.** Store fingerprints in `.dataset_fingerprint_cache.json` next to the data. Invalidate on `mtime` change:

```python
def cached_fast_hash(filepath: str, cache: dict) -> str:
    key = os.path.basename(filepath)
    current_mtime = os.path.getmtime(filepath)
    if key in cache and cache[key].get("mtime") == current_mtime:
        return cache[key]["fast_hash"]
    h = fast_hash_file(filepath)
    cache[key] = {"mtime": current_mtime, "fast_hash": h}
    return h
```

**Distributed training.** Compute fingerprints on rank 0 only, broadcast via `dist.broadcast_object_list()`. All ranks must agree on dataset identity before training begins.

**Training loop integration.** Fingerprint immediately after loading the dataset, before the first training step:

```python
# In scripts/train_phase{N}.py:
dataset = load_dataset(config)
fp = fingerprint(dataset, name=config.dataset_name, split=config.dataset_split)
sig, _ = transform_signature(build_transforms(config))
fp["transforms_signature"] = sig
write_dataset_fingerprints(run_dir, [("primary", fp)])
```

This guarantees `datasets.json` exists even if training crashes on the first step.
