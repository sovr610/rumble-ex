# Manifest Schema Reference

This document is the authoritative reference for `manifest.json` -- the machine-readable provenance record produced by every training run in the brain_ai seven-phase pipeline. The manifest is the single source of truth for reproducing, comparing, and auditing runs.

## Schema Version

Every manifest carries a top-level `schema_version` field using semantic versioning.

```json
{
  "schema_version": "1.0.0"
}
```

**Backward compatibility rules:**

- PATCH increments (1.0.0 -> 1.0.1): additive optional fields only. All existing tooling must continue to work without modification.
- MINOR increments (1.0.0 -> 1.1.0): new required fields with sensible defaults that older manifests can be migrated to automatically. Provide a migration function in `brain_ai/training/manifest.py`.
- MAJOR increments (1.0.0 -> 2.0.0): breaking structural changes. Old manifests require explicit migration scripts. Never perform a major bump without updating all reference documents and validation code simultaneously.

Store the schema version in the manifest module as `MANIFEST_SCHEMA_VERSION = "1.0.0"`. Validate on load: reject manifests with a major version newer than the current code. Accept older major versions only if a migration path exists.

---

## Complete JSON Schema

### Top-Level Structure

```json
{
  "schema_version": "1.0.0",
  "identity": { ... },
  "git": { ... },
  "config": { ... },
  "seeds": { ... },
  "env": { ... },
  "data": { ... },
  "logging": { ... },
  "resume": { ... },
  "checkpoints": { ... },
  "results": { ... }
}
```

All top-level sections are required. Individual fields within sections are marked as required or optional below.

---

### Identity Section

Uniquely identifies this run.

| Field | Type | Required | Description |
|---|---|---|---|
| `run_id` | string | yes | Deterministic identifier (see Run ID Generation below) |
| `ablation_id` | string or null | no | Non-null only for ablation runs; stable hash of overrides |
| `phase` | integer | yes | Training phase, 1 through 7 |
| `mode` | string | yes | `"dev"` or `"production"` |
| `timestamp_start` | string | yes | ISO 8601 UTC timestamp at run start |
| `timestamp_end` | string or null | yes | ISO 8601 UTC timestamp at run end; null while running |
| `hostname` | string | yes | Machine hostname |
| `user` | string | yes | OS username that launched the run |

```json
{
  "identity": {
    "run_id": "2026-02-20_14-30-00_phase4_dev_a1b2c3d",
    "ablation_id": null,
    "phase": 4,
    "mode": "dev",
    "timestamp_start": "2026-02-20T14:30:00Z",
    "timestamp_end": "2026-02-20T15:45:12Z",
    "hostname": "gpu-node-03",
    "user": "researcher"
  }
}
```

**Validation rules:**
- `phase` must be an integer in the range [1, 7].
- `mode` must be exactly `"dev"` or `"production"`.
- `run_id` must match the format `YYYY-MM-DD_HH-MM-SS_phase{N}_{mode}_{git_short}`.
- `timestamp_end` must be null when `results.status` is `"running"`, and non-null otherwise.

---

### Code Provenance Section

Records the exact code state.

| Field | Type | Required | Description |
|---|---|---|---|
| `commit` | string | yes | Full 40-character Git SHA |
| `branch` | string | yes | Branch name at run start |
| `remote_url` | string | yes | Git remote origin URL |
| `dirty` | boolean | yes | True if working tree had uncommitted changes |
| `patch_path` | string or null | conditional | Relative path to saved `git diff` patch file |
| `entrypoint` | string | yes | Script path relative to repo root |
| `cli_args` | list of strings | yes | Complete command-line arguments as passed |

```json
{
  "git": {
    "commit": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
    "branch": "main",
    "remote_url": "https://github.com/org/human-brain.git",
    "dirty": true,
    "patch_path": "artifacts/git_diff.patch",
    "entrypoint": "scripts/train_phase4.py",
    "cli_args": ["--mode", "dev", "--use-amp", "--epochs", "10"]
  }
}
```

**Validation rules:**
- `commit` must be exactly 40 hexadecimal characters.
- If `dirty` is `true`, `patch_path` must be a non-null string pointing to an existing file within the run directory.
- If `dirty` is `false`, `patch_path` must be null.
- `entrypoint` must be a relative path from the repository root.

---

### Config Section

Captures the complete configuration state.

| Field | Type | Required | Description |
|---|---|---|---|
| `brain_ai` | object | yes | Full `BrainAIConfig` serialized as nested dict |
| `overrides` | object | yes | Only the fields explicitly set by the user (CLI args, env vars, or config file overrides); empty dict `{}` if none |
| `resolved` | object | yes | Final config after merging defaults with overrides; identical to `brain_ai` but serves as the canonical "what actually ran" record |
| `feature_flags` | object | yes | Extracted feature flags for quick inspection |

```json
{
  "config": {
    "brain_ai": {
      "snn": {
        "beta": 0.95,
        "num_timesteps": 50,
        "surrogate": "atan",
        "hidden_sizes": [4096, 4096, 2048, 2048],
        "use_learnable_delays": true,
        "use_heterogeneous_tau": true,
        "use_probspikes_loss": true
      },
      "encoder": { "output_dim": 4096, "..." : "..." },
      "workspace": { "workspace_dim": 4096, "..." : "..." },
      "training": { "learning_rate": 0.0003, "batch_size": 32, "..." : "..." },
      "use_snn": true,
      "use_htm": true,
      "use_workspace": true,
      "use_symbolic": true,
      "use_meta": true,
      "use_engram": true
    },
    "overrides": {
      "training.batch_size": 64,
      "snn.num_timesteps": 25
    },
    "resolved": {
      "snn": { "num_timesteps": 25, "..." : "..." },
      "training": { "batch_size": 64, "..." : "..." }
    },
    "feature_flags": {
      "use_snn": true,
      "use_htm": true,
      "use_workspace": true,
      "use_symbolic": true,
      "use_meta": true,
      "use_engram": true
    }
  }
}
```

**Validation rules:**
- `brain_ai` must be a complete serialization of `BrainAIConfig`. No fields may be omitted.
- `feature_flags` must contain exactly the six keys: `use_snn`, `use_htm`, `use_workspace`, `use_symbolic`, `use_meta`, `use_engram`, all boolean.
- Every key in `overrides` must correspond to a valid path in `brain_ai`.
- `resolved` must be the result of applying `overrides` on top of the `brain_ai` defaults. Validate by replaying the merge and checking equality.

---

### Seeds Section

All randomness parameters for deterministic reproduction.

| Field | Type | Required | Description |
|---|---|---|---|
| `base_seed` | integer | yes | Global base seed (default: 1337) |
| `per_phase_offsets` | list of 7 integers | yes | Offset added to base_seed for each phase |
| `torch_deterministic` | boolean | yes | Whether `torch.use_deterministic_algorithms(True)` was called |
| `cudnn_benchmark` | boolean | yes | Value of `torch.backends.cudnn.benchmark` |
| `cudnn_deterministic` | boolean | yes | Value of `torch.backends.cudnn.deterministic` |
| `use_deterministic_algorithms` | boolean | yes | Whether strict determinism was enforced globally |

```json
{
  "seeds": {
    "base_seed": 1337,
    "per_phase_offsets": [0, 100, 200, 300, 400, 500, 600],
    "torch_deterministic": true,
    "cudnn_benchmark": false,
    "cudnn_deterministic": true,
    "use_deterministic_algorithms": true
  }
}
```

**Validation rules:**
- All six fields are required. No field may be null.
- `per_phase_offsets` must be a list of exactly 7 integers.
- `base_seed` must be a non-negative integer.
- If `use_deterministic_algorithms` is `true`, then `cudnn_benchmark` must be `false` and `cudnn_deterministic` must be `true`.

---

### Environment Section

Software and hardware snapshot.

| Field | Type | Required | Description |
|---|---|---|---|
| `python_version` | string | yes | e.g., `"3.11.5"` |
| `os` | string | yes | e.g., `"Linux-6.6.87-x86_64"` |
| `cuda_version` | string or null | yes | e.g., `"12.4"` or null if CPU-only |
| `torch_version` | string | yes | e.g., `"2.5.1+cu124"` |
| `pip_freeze_path` | string | yes | Relative path to `pip_freeze.txt` artifact |
| `hardware` | object | yes | GPU inventory |

The `hardware` object:

| Field | Type | Required | Description |
|---|---|---|---|
| `gpu_names` | list of strings | yes | Name of each GPU (empty list if CPU-only) |
| `gpu_count` | integer | yes | Number of GPUs |
| `gpu_memory_mb` | list of integers | yes | Total memory per GPU in megabytes |

```json
{
  "env": {
    "python_version": "3.11.5",
    "os": "Linux-6.6.87-x86_64",
    "cuda_version": "12.4",
    "torch_version": "2.5.1+cu124",
    "pip_freeze_path": "artifacts/pip_freeze.txt",
    "hardware": {
      "gpu_names": ["NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB"],
      "gpu_count": 2,
      "gpu_memory_mb": [81920, 81920]
    }
  }
}
```

**Validation rules:**
- `gpu_count` must equal the length of both `gpu_names` and `gpu_memory_mb`.
- `pip_freeze_path` must point to an existing file within the run directory.

---

### Data Provenance Section

Records dataset identity for every dataset used in the run.

| Field | Type | Required | Description |
|---|---|---|---|
| `datasets` | list of objects | yes | One entry per dataset |

Each dataset object:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Dataset name (e.g., `"mnist"`, `"imagenet21k"`) |
| `config` | string or null | yes | HuggingFace config name, or null |
| `split` | string | yes | e.g., `"train"`, `"train[:10%]"` |
| `version` | string or null | yes | Dataset version string |
| `fingerprint` | string | yes | Content hash or HF fingerprint |
| `fingerprint_tier` | integer | yes | 1 (HF native), 2 (local hash), or 3 (sample-level) |
| `num_samples` | integer | yes | Number of samples in the split |
| `transforms_signature` | string | yes | Deterministic hash of the transform pipeline |

```json
{
  "data": {
    "datasets": [
      {
        "name": "mnist",
        "config": null,
        "split": "train",
        "version": "1.0.0",
        "fingerprint": "d3b07384d113edec49eaa6238ad5ff00",
        "fingerprint_tier": 1,
        "num_samples": 60000,
        "transforms_signature": "sha256:a9f3e2b1c8d7..."
      },
      {
        "name": "imagenet21k",
        "config": "full",
        "split": "train",
        "version": "2024.1",
        "fingerprint": "sha256:e4c5d6f7a8b9...",
        "fingerprint_tier": 2,
        "num_samples": 14197122,
        "transforms_signature": "sha256:b2c3d4e5f6a7..."
      }
    ]
  }
}
```

**Validation rules:**
- `datasets` must contain at least one entry.
- `fingerprint_tier` must be 1, 2, or 3.
- `num_samples` must be a positive integer.
- `transforms_signature` must be a non-empty string.

---

### Logging Section

Logging backend configuration.

| Field | Type | Required | Description |
|---|---|---|---|
| `tensorboard` | object | yes | TensorBoard settings |
| `wandb` | object | yes | Weights & Biases settings |

```json
{
  "logging": {
    "tensorboard": {
      "enabled": true,
      "log_dir": "logs/tensorboard/"
    },
    "wandb": {
      "enabled": true,
      "project": "brain-ai-phase4",
      "run_id": "abc123xyz"
    }
  }
}
```

**Validation rules:**
- If `wandb.enabled` is `true`, `wandb.project` and `wandb.run_id` must be non-null strings.
- `tensorboard.log_dir` is a relative path within the run directory.

---

### Resume Section

Tracks the complete resume chain so no provenance is lost.

| Field | Type | Required | Description |
|---|---|---|---|
| `enabled` | boolean | yes | Whether this run resumed from a prior run |
| `from_run_id` | string or null | conditional | The run_id being resumed from |
| `from_checkpoint` | string or null | conditional | Path to the checkpoint loaded on resume |
| `phase_boundary_artifacts` | list of strings | no | Paths to phase boundary checkpoint files consumed |
| `resume_events` | list of objects | no | Append-only log of resume events |

Each resume event object:

| Field | Type | Description |
|---|---|---|
| `timestamp` | string | ISO 8601 UTC timestamp of the resume |
| `from_run_id` | string | Which run was resumed from |
| `from_checkpoint` | string | Checkpoint path |
| `step_resumed_at` | integer | Global step at resume point |
| `reason` | string | Why the resume occurred (e.g., `"preemption"`, `"manual"`, `"phase_boundary"`) |

```json
{
  "resume": {
    "enabled": true,
    "from_run_id": "2026-02-19_10-00-00_phase4_dev_b2c3d4e",
    "from_checkpoint": "runs/2026-02-19_10-00-00_phase4_dev_b2c3d4e/checkpoints/phase4/ckpt_step00005000.pt",
    "phase_boundary_artifacts": [
      "runs/2026-02-18_08-00-00_phase3_dev_c3d4e5f/checkpoints/phase3/phase_boundary.pt"
    ],
    "resume_events": [
      {
        "timestamp": "2026-02-20T14:30:00Z",
        "from_run_id": "2026-02-19_10-00-00_phase4_dev_b2c3d4e",
        "from_checkpoint": "runs/.../ckpt_step00005000.pt",
        "step_resumed_at": 5000,
        "reason": "preemption"
      }
    ]
  }
}
```

**Validation rules:**
- If `enabled` is `true`, `from_run_id` must be a non-null string and `from_checkpoint` must be a non-null string.
- If `enabled` is `false`, `from_run_id` and `from_checkpoint` must be null.
- `resume_events` is append-only. Never remove entries.

---

### Checkpoints Section

Checkpoint policy and outputs.

| Field | Type | Required | Description |
|---|---|---|---|
| `save_every_n_steps` | integer | yes | Checkpoint interval in steps |
| `best_metric_key` | string | yes | Metric key used to select the best checkpoint (e.g., `"val/loss"`) |
| `phase_boundary_produced` | string or null | yes | Path to the phase boundary checkpoint if this phase completed successfully; null if not yet produced |

```json
{
  "checkpoints": {
    "save_every_n_steps": 1000,
    "best_metric_key": "val/loss",
    "phase_boundary_produced": "checkpoints/phase4/phase_boundary.pt"
  }
}
```

---

### Results Section

Run outcome -- updated at termination.

| Field | Type | Required | Description |
|---|---|---|---|
| `status` | string | yes | One of: `"running"`, `"completed"`, `"failed"`, `"interrupted"` |
| `best_metrics` | object or null | yes | Best metric values observed during training |
| `final_metrics` | object or null | yes | Metric values at the last step |
| `error` | object or null | yes | Error details if `status` is `"failed"` |

The `error` object (when present):

| Field | Type | Description |
|---|---|---|
| `type` | string | Exception class name |
| `message` | string | Exception message |
| `traceback` | string | Full traceback string |
| `step` | integer or null | Step at which failure occurred |

```json
{
  "results": {
    "status": "completed",
    "best_metrics": {
      "val/loss": 0.0342,
      "val/accuracy": 0.9891,
      "step": 8500
    },
    "final_metrics": {
      "train/loss": 0.0198,
      "val/loss": 0.0356,
      "val/accuracy": 0.9887,
      "step": 10000
    },
    "error": null
  }
}
```

Example of a failed run:

```json
{
  "results": {
    "status": "failed",
    "best_metrics": {
      "val/loss": 0.1523,
      "val/accuracy": 0.9412,
      "step": 3200
    },
    "final_metrics": null,
    "error": {
      "type": "RuntimeError",
      "message": "CUDA out of memory. Tried to allocate 2.00 GiB",
      "traceback": "Traceback (most recent call last):\n  File ...",
      "step": 3245
    }
  }
}
```

**Validation rules:**
- `status` must be one of the four allowed values.
- If `status` is `"running"`, `best_metrics` and `final_metrics` may be null.
- If `status` is `"failed"`, `error` must be non-null.
- If `status` is `"completed"`, `error` must be null and `final_metrics` must be non-null.

---

## Manifest Lifecycle

### Creation

Generate `manifest.json` at the very start of the run, before the first training step. Set `results.status` to `"running"`, `results.best_metrics` to null, `results.final_metrics` to null, `results.error` to null, and `identity.timestamp_end` to null. Write the file to `runs/<run_id>/manifest.json`.

### Update at Completion

When the run ends (success, failure, or interruption):

1. Set `results.status` to the appropriate terminal value.
2. Populate `results.best_metrics` and `results.final_metrics` (or `results.error`).
3. Set `identity.timestamp_end` to the current UTC timestamp.
4. Set `checkpoints.phase_boundary_produced` to the phase boundary artifact path if the phase completed successfully.
5. Write the updated manifest back to the same path. This is the only permitted overwrite of `manifest.json`.

### Resume Behavior

Never overwrite a prior run's manifest when resuming. Instead:

1. Create a new run directory with a new `run_id`.
2. In the new manifest, set `resume.enabled` to `true` and populate `resume.from_run_id` and `resume.from_checkpoint`.
3. Append a new entry to `resume.resume_events`.
4. The original run's manifest remains untouched.

If a run is resumed multiple times within the same run directory (e.g., repeated preemption recovery without new run IDs), append additional entries to `resume.resume_events`. Never delete or modify prior entries.

---

## manifest.lock.json

### Purpose

`manifest.lock.json` is the fully resolved configuration snapshot. It eliminates all `"auto"`, `"default"`, and computed values, replacing them with their concrete resolved values at runtime. This file guarantees that re-running from the lock file produces identical configuration without depending on detection logic.

### When Generated

Generate `manifest.lock.json` at run start, immediately after `manifest.json` is written. It must be written before the first training step.

### Format

The lock file contains exactly the `config.resolved` section from the manifest, but as a standalone top-level object. No other sections are included.

```json
{
  "schema_version": "1.0.0",
  "resolved_config": {
    "snn": {
      "beta": 0.95,
      "num_timesteps": 50,
      "surrogate": "atan",
      "surrogate_alpha": 2.0,
      "dropout": 0.1,
      "hidden_sizes": [4096, 4096, 2048, 2048],
      "use_learnable_delays": true,
      "max_delay": 16,
      "use_heterogeneous_tau": true,
      "use_adaptive_threshold": false,
      "use_probspikes_loss": true,
      "spike_rate_target": 0.1,
      "spike_rate_weight": 0.01,
      "temporal_consistency_weight": 0.001
    },
    "encoder": {
      "output_dim": 4096,
      "vision_channels": [256, 512, 1024, 1024],
      "vision_num_layers": 24
    },
    "workspace": {
      "workspace_dim": 4096,
      "num_heads": 32,
      "capacity_limit": 7
    },
    "training": {
      "learning_rate": 0.0003,
      "batch_size": 64,
      "use_amp": true,
      "amp_dtype": "bfloat16"
    },
    "use_snn": true,
    "use_htm": true,
    "use_workspace": true,
    "use_symbolic": true,
    "use_meta": true,
    "use_engram": true
  }
}
```

Every value in `manifest.lock.json` must be a concrete literal. Forbidden values: `"auto"`, `"default"`, `null` where a concrete value should exist, or any string that implies runtime detection. If a config field uses `"auto"` (e.g., `device: "auto"`), resolve it to the actual detected value (e.g., `"cuda"`) before writing.

---

## Run ID Generation

### Format

```
YYYY-MM-DD_HH-MM-SS_phase{N}_{mode}_{git_short}
```

- `YYYY-MM-DD_HH-MM-SS`: UTC timestamp at run start, zero-padded.
- `phase{N}`: Phase number, e.g., `phase4`.
- `{mode}`: Either `dev` or `production`.
- `{git_short}`: First 7 characters of the current Git commit SHA.

Example: `2026-02-20_14-30-00_phase4_dev_a1b2c3d`

### Uniqueness Guarantees

The combination of second-resolution UTC timestamp and git short SHA provides sufficient uniqueness for all practical purposes. If two runs start within the same second on the same commit, append a 4-character random hex suffix: `2026-02-20_14-30-00_phase4_dev_a1b2c3d_f9e1`.

### Ablation Run ID Derivation

For ablation runs, derive a stable `ablation_id` as follows:

1. Sort the override keys alphabetically.
2. Serialize as a canonical JSON string (sorted keys, no whitespace).
3. Compute SHA-256 of the serialized string.
4. Take the first 8 hex characters.

The ablation run's `run_id` uses the same timestamp format but appends the `ablation_id`:

```
2026-02-20_14-30-00_phase4_dev_a1b2c3d_abl_3f7a2b1c
```

This ensures that re-running the same ablation overrides on the same commit at the same time produces a deterministic, traceable identifier.

```python
import hashlib
import json

def derive_ablation_id(overrides: dict) -> str:
    """Derive a stable ablation ID from override dict."""
    canonical = json.dumps(overrides, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return digest[:8]
```

---

## Cross-Field Validation Rules

Apply these consistency checks when loading or writing a manifest. Reject the manifest if any check fails.

### Identity Checks

1. `identity.phase` must be an integer in [1, 7].
2. `identity.mode` must be `"dev"` or `"production"`.
3. `identity.run_id` must contain `phase{identity.phase}` and `{identity.mode}` as substrings.
4. If `identity.timestamp_end` is non-null, it must be chronologically after `identity.timestamp_start`.

### Code Provenance Checks

5. `git.commit` must be exactly 40 hex characters.
6. If `git.dirty` is `true`, `git.patch_path` must be non-null and the file must exist in the run directory.
7. If `git.dirty` is `false`, `git.patch_path` must be null.

### Config Checks

8. `config.feature_flags` must contain all six boolean keys.
9. Every key path in `config.overrides` must resolve to a valid field in `config.brain_ai`.

### Seeds Checks

10. All six seeds fields must be present and non-null.
11. `seeds.per_phase_offsets` must have exactly 7 elements.
12. If `seeds.use_deterministic_algorithms` is `true`, then `seeds.cudnn_benchmark` must be `false`.

### Resume Checks

13. If `resume.enabled` is `true`, `resume.from_run_id` must be a non-null, non-empty string.
14. If `resume.enabled` is `true`, `resume.from_checkpoint` must be a non-null, non-empty string.
15. If `resume.enabled` is `false`, both `resume.from_run_id` and `resume.from_checkpoint` must be null.

### Results Checks

16. `results.status` must be one of `"running"`, `"completed"`, `"failed"`, `"interrupted"`.
17. If `results.status` is `"failed"`, `results.error` must be non-null with `type`, `message`, and `traceback` fields.
18. If `results.status` is `"completed"`, `results.error` must be null and `results.final_metrics` must be non-null.
19. If `results.status` is `"running"`, `identity.timestamp_end` must be null.

### Cross-Section Checks

20. The phase number embedded in `identity.run_id` must equal `identity.phase`.
21. The mode substring in `identity.run_id` must equal `identity.mode`.
22. If `identity.ablation_id` is non-null, the `run_id` must contain an `_abl_` segment.

---

## Reproduction Protocol

### Step-by-Step Reproduction

To reproduce a run from its manifest:

1. **Checkout code.** Clone the repository and checkout `git.commit`. If `git.dirty` was `true`, apply the patch at `git.patch_path`.

   ```bash
   git clone <git.remote_url> repro && cd repro
   git checkout <git.commit>
   if [ -f <patch_path> ]; then git apply <patch_path>; fi
   ```

2. **Install dependencies.** Install the exact package versions from `pip_freeze.txt`.

   ```bash
   pip install -r <run_dir>/artifacts/pip_freeze.txt
   ```

3. **Apply configuration.** Load `manifest.lock.json` and pass the resolved config to `BrainAIConfig`. Do not rely on defaults -- use only the lock file values.

4. **Set seeds.** Apply all seed values from the `seeds` section before any tensor allocation or data loading.

   ```python
   import torch
   import random
   import numpy as np

   seed = manifest["seeds"]["base_seed"] + manifest["seeds"]["per_phase_offsets"][phase - 1]
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = manifest["seeds"]["cudnn_benchmark"]
   torch.backends.cudnn.deterministic = manifest["seeds"]["cudnn_deterministic"]
   if manifest["seeds"]["use_deterministic_algorithms"]:
       torch.use_deterministic_algorithms(True)
   ```

5. **Run training.** Execute the entrypoint with the original CLI args, or load the config programmatically.

   ```bash
   python <git.entrypoint> <cli_args...>
   ```

### Tolerance Definitions

Exact bit-for-bit reproduction is not guaranteed due to GPU floating-point nondeterminism. Define reproduction success as:

| Metric | Tolerance |
|---|---|
| Training loss | Within 2% relative of original |
| Validation loss | Within 2% relative of original |
| Accuracy (classification) | Within 1% absolute of original |

Calculate relative tolerance as: `abs(original - reproduced) / abs(original) <= 0.02`.

Calculate absolute tolerance as: `abs(original - reproduced) <= 0.01`.

### Strict Determinism Mode

When `seeds.use_deterministic_algorithms` is `true`, PyTorch uses deterministic algorithm implementations for all operations. This incurs a performance penalty (typically 10-20% slower) but narrows the reproduction gap to near-zero on the same hardware. Note that some operations (e.g., `torch.nn.functional.interpolate` with certain modes) raise errors in strict mode. Handle these by providing deterministic alternatives in the codebase.

Cross-hardware reproduction (e.g., A100 vs H100) may still exhibit small numerical differences even in strict mode. Document the hardware in the manifest and treat cross-hardware reproduction as best-effort.

---

## Comparison and Diff

### Comparing Two Manifests

Use manifest comparison to verify whether a reproduction attempt matches the original run's conditions.

### Fields That Must Match Exactly

For two runs to be considered "same experiment," the following fields must be identical:

- `git.commit` (and `git.patch_path` content if dirty)
- `config.resolved` (entire object, deep equality)
- `seeds` (entire object)
- `identity.phase`
- `identity.mode`
- `data.datasets[].fingerprint` (for each dataset)
- `data.datasets[].transforms_signature` (for each dataset)

### Fields to Ignore

These fields vary between runs and are not relevant to reproduction equivalence:

- `identity.run_id`
- `identity.timestamp_start`, `identity.timestamp_end`
- `identity.hostname`, `identity.user`
- `identity.ablation_id`
- `logging.wandb.run_id`
- `results` (entire section -- this is what you are comparing)
- `resume` (entire section -- resume provenance, not experiment identity)

### Fields to Warn On

Differences in these fields do not invalidate reproduction but should be flagged as warnings:

- `env.python_version` -- minor version differences may affect results
- `env.torch_version` -- different torch versions may change numerics
- `env.cuda_version` -- different CUDA versions may affect GPU kernels
- `env.hardware` -- different GPUs may produce different floating-point results

### Comparison Implementation

```python
def compare_manifests(original: dict, reproduced: dict) -> dict:
    """Compare two manifests and return a diff report.

    Returns:
        dict with keys:
            "match": bool -- True if all exact-match fields are identical
            "exact_diffs": list -- fields that must match but differ
            "warnings": list -- fields that differ but are advisory
            "ignored": list -- fields that differ but are irrelevant
    """
    EXACT_FIELDS = [
        "git.commit",
        "config.resolved",
        "seeds",
        "identity.phase",
        "identity.mode",
    ]
    WARN_FIELDS = [
        "env.python_version",
        "env.torch_version",
        "env.cuda_version",
        "env.hardware",
    ]
    IGNORE_FIELDS = [
        "identity.run_id",
        "identity.timestamp_start",
        "identity.timestamp_end",
        "identity.hostname",
        "identity.user",
        "identity.ablation_id",
        "logging.wandb.run_id",
        "results",
        "resume",
    ]
    # ... deep comparison logic
```

### Dataset Comparison

Compare datasets by matching on `name` + `split`, then checking `fingerprint` and `transforms_signature`. If a dataset appears in one manifest but not the other, flag it as a critical mismatch. If fingerprints differ for the same dataset name and split, flag as a critical mismatch (data has changed).

---

## Full Manifest Example

Below is a complete, minimal manifest for a dev-mode Phase 4 run.

```json
{
  "schema_version": "1.0.0",
  "identity": {
    "run_id": "2026-02-20_14-30-00_phase4_dev_a1b2c3d",
    "ablation_id": null,
    "phase": 4,
    "mode": "dev",
    "timestamp_start": "2026-02-20T14:30:00Z",
    "timestamp_end": "2026-02-20T15:45:12Z",
    "hostname": "gpu-node-03",
    "user": "researcher"
  },
  "git": {
    "commit": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
    "branch": "main",
    "remote_url": "https://github.com/org/human-brain.git",
    "dirty": false,
    "patch_path": null,
    "entrypoint": "scripts/train_phase4.py",
    "cli_args": ["--mode", "dev"]
  },
  "config": {
    "brain_ai": {
      "snn": { "beta": 0.95, "num_timesteps": 10, "hidden_sizes": [256, 128] },
      "encoder": { "output_dim": 512 },
      "workspace": { "workspace_dim": 512 },
      "training": { "learning_rate": 0.0003, "batch_size": 8 },
      "use_snn": true,
      "use_htm": false,
      "use_workspace": true,
      "use_symbolic": false,
      "use_meta": false,
      "use_engram": false
    },
    "overrides": {},
    "resolved": {
      "snn": { "beta": 0.95, "num_timesteps": 10, "hidden_sizes": [256, 128] },
      "encoder": { "output_dim": 512 },
      "workspace": { "workspace_dim": 512 },
      "training": { "learning_rate": 0.0003, "batch_size": 8 },
      "use_snn": true,
      "use_htm": false,
      "use_workspace": true,
      "use_symbolic": false,
      "use_meta": false,
      "use_engram": false
    },
    "feature_flags": {
      "use_snn": true,
      "use_htm": false,
      "use_workspace": true,
      "use_symbolic": false,
      "use_meta": false,
      "use_engram": false
    }
  },
  "seeds": {
    "base_seed": 1337,
    "per_phase_offsets": [0, 100, 200, 300, 400, 500, 600],
    "torch_deterministic": true,
    "cudnn_benchmark": false,
    "cudnn_deterministic": true,
    "use_deterministic_algorithms": true
  },
  "env": {
    "python_version": "3.11.5",
    "os": "Linux-6.6.87-x86_64",
    "cuda_version": "12.4",
    "torch_version": "2.5.1+cu124",
    "pip_freeze_path": "artifacts/pip_freeze.txt",
    "hardware": {
      "gpu_names": ["NVIDIA RTX 4090"],
      "gpu_count": 1,
      "gpu_memory_mb": [24576]
    }
  },
  "data": {
    "datasets": [
      {
        "name": "mnist",
        "config": null,
        "split": "train",
        "version": "1.0.0",
        "fingerprint": "d3b07384d113edec49eaa6238ad5ff00",
        "fingerprint_tier": 1,
        "num_samples": 60000,
        "transforms_signature": "sha256:a9f3e2b1c8d7e6f5a4b3c2d1e0f9a8b7"
      }
    ]
  },
  "logging": {
    "tensorboard": {
      "enabled": true,
      "log_dir": "logs/tensorboard/"
    },
    "wandb": {
      "enabled": false,
      "project": null,
      "run_id": null
    }
  },
  "resume": {
    "enabled": false,
    "from_run_id": null,
    "from_checkpoint": null,
    "phase_boundary_artifacts": [],
    "resume_events": []
  },
  "checkpoints": {
    "save_every_n_steps": 1000,
    "best_metric_key": "val/loss",
    "phase_boundary_produced": "checkpoints/phase4/phase_boundary.pt"
  },
  "results": {
    "status": "completed",
    "best_metrics": {
      "val/loss": 0.0342,
      "val/accuracy": 0.9891,
      "step": 8500
    },
    "final_metrics": {
      "train/loss": 0.0198,
      "val/loss": 0.0356,
      "val/accuracy": 0.9887,
      "step": 10000
    },
    "error": null
  }
}
```
