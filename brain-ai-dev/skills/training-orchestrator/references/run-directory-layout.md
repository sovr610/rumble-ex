# Run Directory Layout Reference

This document defines the standardized directory structure produced by every training run in the brain_ai seven-phase pipeline. Every run -- whether a single-phase dev iteration, a full seven-phase production pipeline, or an ablation sweep -- writes its outputs into a self-contained directory under `runs/`. The layout is the canonical contract between the training orchestrator, checkpoint management, logging backends, cleanup utilities, and any downstream tooling that reads run artifacts.

---

## Complete Directory Tree

```
runs/
  <run_id>/
    manifest.json                        # Provenance snapshot (source of truth)
    manifest.lock.json                   # Fully resolved config (no "auto"/"default")
    stdout.log                           # Captured stdout
    stderr.log                           # Captured stderr
    metrics.jsonl                        # Append-only metric log (all phases)
    logs/
      tensorboard/                       # TensorBoard event files
        events.out.tfevents.*
      wandb/                             # W&B local files (if enabled)
        run-<id>/
    checkpoints/
      phase1/
        ckpt_step00001000.pt             # Periodic checkpoint
        ckpt_step00002000.pt
        ckpt_best_val_loss.pt            # Best by metric
        phase_boundary.pt                # Canonical artifact consumed by Phase 2
      phase2/
        ckpt_step00003000.pt
        ckpt_step00004000.pt
        ckpt_best_val_loss.pt
        phase_boundary.pt
      ...
      phase7/
        ckpt_step00020000.pt
        ckpt_final.pt                    # Final model (last step of last phase)
        phase_boundary.pt                # Complete trained model
    artifacts/
      datasets.json                      # Dataset fingerprints
      env.txt                            # Platform info
      pip_freeze.txt                     # Frozen dependencies
      git_diff.patch                     # Working tree diff (only if dirty)
      hardware.json                      # GPU/CPU inventory
    reports/
      summary.json                       # Final metrics, run status, durations
      profiler.json                      # Optional profiling data
      ablation_summary.json              # Present only for ablation runs
```

Every subdirectory is created upfront at run start. No directory is created lazily during training. This prevents race conditions in multi-process setups and ensures that logging backends can open file handles immediately.

---

## Run ID Format

### Standard Pattern

```
YYYY-MM-DD_HH-MM-SS_phase{N}_{mode}_{git_short}
```

| Component | Description | Example |
|---|---|---|
| `YYYY-MM-DD_HH-MM-SS` | UTC timestamp at run start, zero-padded | `2026-02-20_23-15-02` |
| `phase{N}` | Phase number (1-7) or `full` for pipeline runs | `phase4` |
| `{mode}` | Training mode | `dev` or `production` |
| `{git_short}` | First 7 characters of the current Git commit SHA | `a1b2c3d` |

Examples:

- Single phase: `2026-02-20_23-15-02_phase4_dev_a1b2c3d`
- Full pipeline: `2026-02-20_23-15-02_full_dev_a1b2c3d`
- Production phase: `2026-02-20_08-00-00_phase1_production_f9e8d7c`

### Ablation Run ID

Ablation runs use a distinct format that encodes the experiment group, run index, git state, and seed:

```
{experiment_name}_abl_{YYYYMMDD}_run{NNN}_{git_short}_seed{S}
```

Example: `workspace_engram_abl_20260220_run001_f3a2b1c_seed1337`

The `experiment_name` is a human-readable label for the ablation group. The `run{NNN}` counter is zero-padded to three digits within each group.

### Uniqueness Guarantees

The combination of second-resolution UTC timestamp and 7-character git SHA provides practical uniqueness. If a collision is detected (the target directory already exists), append a 4-character random hex suffix:

```
2026-02-20_23-15-02_phase4_dev_a1b2c3d_f9e1
```

### Sorting Property

Run IDs are designed so that lexicographic sort equals chronological sort. The `YYYY-MM-DD_HH-MM-SS` prefix ensures this. Use `sorted(os.listdir("runs/"))` to list runs in chronological order.

---

## File Naming Conventions

### Checkpoints

| File Pattern | Description |
|---|---|
| `ckpt_step{N:08d}.pt` | Periodic checkpoint, step number zero-padded to 8 digits |
| `ckpt_best_{metric_name}.pt` | Best checkpoint by a specific metric |
| `phase_boundary.pt` | Canonical phase-transition artifact |
| `ckpt_final.pt` | Final model at end of training |

Examples:

```
ckpt_step00001000.pt
ckpt_step00012500.pt
ckpt_best_val_loss.pt
ckpt_best_val_acc.pt
phase_boundary.pt
ckpt_final.pt
```

Zero-pad to 8 digits. This supports up to 99,999,999 steps before the width overflows, which exceeds any practical training run. The zero-padding ensures correct lexicographic sorting of checkpoint files.

### Logs and Metrics

| File | Location | Description |
|---|---|---|
| `stdout.log` | Run root | Captured standard output, one per run |
| `stderr.log` | Run root | Captured standard error, one per run |
| `metrics.jsonl` | Run root | Append-only metric log, one line per metric event |

Do not timestamp log file names. Each run has exactly one `stdout.log` and one `stderr.log`. If a run is resumed, new output is appended to the existing log files.

Place `metrics.jsonl` at the run root, not inside `logs/`. This file is the primary structured metric sink and is accessed by cleanup utilities, comparison scripts, and summary generators. Keeping it at the top level makes it easy to find.

---

## Checkpoint Contents

### Periodic Checkpoint (`ckpt_step*.pt`)

Periodic checkpoints contain the complete training state required for exact resumption. Save these using `torch.save` to produce a single `.pt` file.

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "rng_states": {
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": numpy.random.get_state(),
        "python": random.getstate(),
    },
    "global_step": 12000,
    "phase": 4,
    "phase_step": 2000,
    "epoch": 5,
    "best_metrics": {"val_loss": 0.342, "val_acc": 0.891},
    "config_hash": "sha256:a9f3e2b1c8d7e6f5...",
    "schema_version": "1.0",
}
```

| Field | Type | Description |
|---|---|---|
| `model_state_dict` | dict | All model parameters and buffers |
| `optimizer_state_dict` | dict | Optimizer state (momentum, adaptive LR buffers) |
| `scheduler_state_dict` | dict | Learning rate scheduler state |
| `rng_states` | dict | Full RNG state for all four generators |
| `global_step` | int | Total training steps across all phases |
| `phase` | int | Current phase number (1-7) |
| `phase_step` | int | Steps completed within the current phase |
| `epoch` | int | Current epoch within the current phase |
| `best_metrics` | dict | Best metric values observed so far |
| `config_hash` | str | SHA-256 hash of the resolved config for integrity checking |
| `schema_version` | str | Checkpoint schema version for forward compatibility |

### Phase Boundary Checkpoint (`phase_boundary.pt`)

Phase boundary checkpoints are the canonical artifacts consumed by the next phase. They contain model weights and configuration only -- no optimizer state, no scheduler state, no RNG state. This is intentional: the next phase initializes its own optimizer and scheduler.

```python
{
    "model_state_dict": model.state_dict(),
    "phase": 4,
    "config_hash": "sha256:a9f3e2b1c8d7e6f5...",
    "config_snapshot": { ... },  # Serialized BrainAIConfig
    "compatibility": {
        "feature_flags": {"use_snn": True, "use_htm": True, ...},
        "workspace_dim": 4096,
        "encoder_output_dim": 4096,
    },
    "schema_version": "1.0",
}
```

The `compatibility` section records the dimensions and feature flags that the next phase must match. Phase 5 loading a Phase 4 boundary checkpoint validates that workspace dimensions and encoder output dimensions are compatible before proceeding.

### Best Checkpoint (`ckpt_best_*.pt`)

A best checkpoint has the same structure as a periodic checkpoint. When a monitored metric improves, save a new checkpoint to the best path (overwriting the previous best). Alternatively, implement best tracking as a symlink pointing to the periodic checkpoint that achieved the best score. Either approach is acceptable; the symlink approach saves disk space.

### Final Checkpoint (`ckpt_final.pt`)

The final checkpoint has the same structure as a periodic checkpoint. Save it at the very last step of training, regardless of whether the final step improved any metric. This checkpoint represents the state at training termination.

---

## Artifact Files

### datasets.json

Record the identity and fingerprint of every dataset used during the run. Cross-reference with the dataset fingerprinting system defined in the dataset-fingerprinting reference.

```json
{
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
}
```

### env.txt

Capture platform information as a human-readable text file. Generate at run start using Python's `platform` and `sys` modules.

```
Platform: Linux-6.6.87-x86_64-with-glib2.35
Python: 3.11.5 (main, Sep 11 2024, 15:47:00) [GCC 11.4.0]
CUDA: 12.4
cuDNN: 90100
PyTorch: 2.5.1+cu124
```

### pip_freeze.txt

Capture the exact installed package versions. Generate by running `pip freeze` and writing the output to this file. This enables exact dependency reproduction.

### git_diff.patch

Save only when the working tree is dirty at run start (`git diff HEAD` output). When the working tree is clean, do not create this file. The manifest's `git.dirty` field indicates whether this file exists.

### hardware.json

Capture the hardware inventory at run start.

```json
{
  "gpus": [
    {"name": "NVIDIA A100-SXM4-80GB", "memory_gb": 80, "index": 0},
    {"name": "NVIDIA A100-SXM4-80GB", "memory_gb": 80, "index": 1}
  ],
  "cpu": "AMD EPYC 7763 64-Core Processor",
  "ram_gb": 512,
  "hostname": "gpu-node-03"
}
```

Enumerate GPUs using `torch.cuda.get_device_properties(i)` for each device index. Report CPU model from `/proc/cpuinfo` or `platform.processor()`. Report total system RAM from `psutil.virtual_memory().total` if available, or `/proc/meminfo`.

---

## reports/summary.json

Generate `summary.json` at run completion (or failure). This file provides a quick overview of the run without loading full manifests or parsing metric logs.

```json
{
  "run_id": "2026-02-20_23-15-02_phase4_dev_a1b2c3d",
  "status": "completed",
  "phases_completed": [1, 2, 3, 4],
  "total_duration_seconds": 3600,
  "best_metrics": {
    "val_loss": 0.342,
    "val_acc": 0.891
  },
  "final_metrics": {
    "train_loss": 0.298,
    "val_loss": 0.356,
    "val_acc": 0.882
  },
  "error": null,
  "checkpoints_saved": 12,
  "phase_durations": {
    "phase1": 600,
    "phase2": 900,
    "phase3": 750,
    "phase4": 1350
  }
}
```

| Field | Type | Description |
|---|---|---|
| `run_id` | string | The run identifier |
| `status` | string | One of `"completed"`, `"failed"`, `"interrupted"` |
| `phases_completed` | list of int | Phases that finished successfully |
| `total_duration_seconds` | float | Wall-clock time from start to end |
| `best_metrics` | dict | Best metric values observed across all phases |
| `final_metrics` | dict | Metric values at the last training step |
| `error` | object or null | Error details if status is `"failed"` |
| `checkpoints_saved` | int | Total number of checkpoints written |
| `phase_durations` | dict | Wall-clock seconds per phase |

For failed runs, the `error` field contains the exception type, message, and the step at which failure occurred:

```json
{
  "error": {
    "type": "RuntimeError",
    "message": "CUDA out of memory. Tried to allocate 2.00 GiB",
    "step": 3245
  }
}
```

---

## metrics.jsonl Format

`metrics.jsonl` is an append-only, newline-delimited JSON file. Each line is a self-contained JSON object representing one metric event. One writer appends at a time; never seek or overwrite.

```jsonl
{"step":1000,"phase":1,"phase_step":1000,"epoch":2,"train_loss":1.234,"val_loss":1.456,"lr":0.0003,"timestamp":"2026-02-20T23:16:02Z"}
{"step":2000,"phase":1,"phase_step":2000,"epoch":4,"train_loss":0.987,"val_loss":1.123,"lr":0.0003,"timestamp":"2026-02-20T23:17:02Z"}
{"step":3000,"phase":2,"phase_step":1000,"epoch":1,"train_loss":0.876,"val_loss":0.945,"lr":0.0001,"timestamp":"2026-02-20T23:18:02Z"}
```

Required fields per line:

| Field | Type | Description |
|---|---|---|
| `step` | int | Global step across all phases |
| `phase` | int | Current phase number |
| `phase_step` | int | Step within the current phase |
| `epoch` | int | Epoch within the current phase |
| `timestamp` | string | ISO 8601 UTC timestamp |

All other fields are metric-specific and vary by phase. Common fields include `train_loss`, `val_loss`, `val_acc`, `lr`, `grad_norm`, and `spike_rate`. Multi-phase runs produce entries from all phases in the same file, distinguished by the `phase` field.

---

## Multi-Phase Run Handling

### Full Pipeline Runs

When `scripts/train_full_pipeline.py` executes all seven phases, use a single run directory with a single `run_id`. The `phase{N}` component in the run ID is replaced by `full`:

```
2026-02-20_23-15-02_full_dev_a1b2c3d
```

### Checkpoint Organization

Each phase writes its checkpoints into its own subdirectory under `checkpoints/`. Phase N reads the `phase_boundary.pt` from Phase N-1's directory:

```
checkpoints/
  phase1/
    phase_boundary.pt          # Phase 2 loads this
  phase2/
    phase_boundary.pt          # Phase 3 loads this
  ...
  phase7/
    phase_boundary.pt          # Complete trained model
```

### Shared Manifest

The `manifest.json` for a full pipeline run captures all phases in sequence. The `identity.phase` field is set to the current active phase and updated as each phase completes. The `checkpoints.phase_boundary_produced` field is updated after each phase.

### Shared Metric Log

`metrics.jsonl` contains entries from all phases. Filter by the `phase` field to isolate metrics for a specific phase. The `step` field is globally monotonic across phases; the `phase_step` field resets to zero at the start of each phase.

---

## Cleanup and Archival Policy

### Retention Rules

| Artifact | Retention | Rationale |
|---|---|---|
| `manifest.json` | Indefinite | Small file; source of truth for provenance |
| `manifest.lock.json` | Indefinite | Small file; needed for reproduction |
| `reports/summary.json` | Indefinite | Small file; needed for run comparison |
| `metrics.jsonl` | Indefinite | Needed for analysis and visualization |
| `phase_boundary.pt` | Indefinite | Canonical artifact consumed by downstream phases |
| `ckpt_best_*.pt` | Indefinite | Best model weights |
| `ckpt_final.pt` | Keep last only | Final training state |
| `ckpt_step*.pt` | Deletable | Intermediate checkpoints for crash recovery |
| `logs/tensorboard/` | Deletable | Can be regenerated from metrics.jsonl |
| `logs/wandb/` | Deletable | Synced to cloud; local copy is redundant |
| `artifacts/` | Indefinite | Small files; needed for reproduction |

### Cleanup Utility

Provide a `cleanup_run` function that removes intermediate checkpoints while preserving provenance and best artifacts:

```python
def cleanup_run(
    run_id: str,
    base_dir: str = "runs/",
    keep_best: bool = True,
    keep_boundary: bool = True,
    keep_last: bool = True,
) -> dict:
    """Remove intermediate checkpoints from a completed run.

    Never delete manifest.json, manifest.lock.json, reports/summary.json,
    metrics.jsonl, or any file in artifacts/.

    Args:
        run_id: The run directory name.
        base_dir: Parent directory containing run directories.
        keep_best: Retain ckpt_best_*.pt files.
        keep_boundary: Retain phase_boundary.pt files.
        keep_last: Retain the highest-numbered ckpt_step*.pt per phase.

    Returns:
        Dict with keys "deleted" (list of paths) and "retained" (list of paths).
    """
```

### Archival

Archive a completed run by creating a compressed tarball:

```bash
tar -czf runs/archive/2026-02-20_23-15-02_phase4_dev_a1b2c3d.tar.gz \
    -C runs/ 2026-02-20_23-15-02_phase4_dev_a1b2c3d/
```

Run `cleanup_run` before archival to minimize archive size. The archive contains the full directory tree including manifests, summaries, retained checkpoints, artifacts, and metric logs.

### Protected Files

The following files must never be deleted by any cleanup or archival utility:

- `manifest.json`
- `manifest.lock.json`
- `reports/summary.json`
- `metrics.jsonl`
- All files under `artifacts/`

---

## Concurrent Access Safety

### Atomic Writes

Write all files atomically using the temp-file-then-rename pattern. This prevents partial reads by concurrent processes.

```python
import os
import tempfile
import json

def atomic_write_json(path: str, data: dict) -> None:
    """Write JSON atomically via temp file + rename."""
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise
```

### File-Level Access Rules

| File | Access Pattern | Safety Mechanism |
|---|---|---|
| `manifest.json` | Write-once at start, update-once at end | Atomic write via temp + rename |
| `manifest.lock.json` | Write-once at start | Atomic write via temp + rename |
| `metrics.jsonl` | Append-only, one writer | Single-writer guarantee; append is atomic for lines < PIPE_BUF |
| `stdout.log` / `stderr.log` | Append-only, one writer | Redirected from process stdout/stderr |
| `ckpt_*.pt` | Write-once per filename | Save to temp file, then rename to final name |
| `phase_boundary.pt` | Write-once per phase | Same as checkpoints |
| TensorBoard events | Managed by SummaryWriter | Thread-safe by design |

### Checkpoint Atomicity

Save checkpoints to a temporary file in the same directory, then rename to the target filename. This ensures that a reader never sees a partially-written checkpoint file.

```python
def save_checkpoint_atomic(state: dict, path: str) -> None:
    """Save a PyTorch checkpoint atomically."""
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(state, tmp_path)
        os.rename(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
```

---

## Directory Creation

### create_run_directory

Create the entire directory tree at run start. Return a `RunDirectory` object that provides path properties for every subdirectory and file location.

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunDirectory:
    """Typed path accessor for the standardized run directory layout."""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def manifest_lock(self) -> Path:
        return self.root / "manifest.lock.json"

    @property
    def stdout_log(self) -> Path:
        return self.root / "stdout.log"

    @property
    def stderr_log(self) -> Path:
        return self.root / "stderr.log"

    @property
    def metrics_jsonl(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def tensorboard_dir(self) -> Path:
        return self.root / "logs" / "tensorboard"

    @property
    def wandb_dir(self) -> Path:
        return self.root / "logs" / "wandb"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    def phase_checkpoint_dir(self, phase: int) -> Path:
        return self.root / "checkpoints" / f"phase{phase}"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def datasets_json(self) -> Path:
        return self.root / "artifacts" / "datasets.json"

    @property
    def env_txt(self) -> Path:
        return self.root / "artifacts" / "env.txt"

    @property
    def pip_freeze(self) -> Path:
        return self.root / "artifacts" / "pip_freeze.txt"

    @property
    def git_diff_patch(self) -> Path:
        return self.root / "artifacts" / "git_diff.patch"

    @property
    def hardware_json(self) -> Path:
        return self.root / "artifacts" / "hardware.json"

    @property
    def summary_json(self) -> Path:
        return self.root / "reports" / "summary.json"

    @property
    def profiler_json(self) -> Path:
        return self.root / "reports" / "profiler.json"

    @property
    def ablation_summary(self) -> Path:
        return self.root / "reports" / "ablation_summary.json"


def create_run_directory(
    run_id: str,
    base_dir: str = "runs/",
    phases: list[int] | None = None,
) -> RunDirectory:
    """Create the full run directory tree upfront.

    Args:
        run_id: The run identifier (becomes the directory name).
        base_dir: Parent directory for all runs.
        phases: Phase numbers to create checkpoint subdirectories for.
                Defaults to [1, 2, 3, 4, 5, 6, 7] for full pipeline.

    Returns:
        RunDirectory with path properties for each subdirectory.

    Raises:
        FileExistsError: If the run directory already exists.
    """
    if phases is None:
        phases = [1, 2, 3, 4, 5, 6, 7]

    root = Path(base_dir) / run_id
    if root.exists():
        raise FileExistsError(
            f"Run directory already exists: {root}. "
            f"Append a random suffix to the run_id to resolve."
        )

    # Create all subdirectories upfront
    root.mkdir(parents=True)
    (root / "logs" / "tensorboard").mkdir(parents=True)
    (root / "logs" / "wandb").mkdir(parents=True)
    for phase in phases:
        (root / "checkpoints" / f"phase{phase}").mkdir(parents=True)
    (root / "artifacts").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)

    return RunDirectory(root=root)
```

### Key Properties

- **Upfront creation.** All directories are created before training begins. No subdirectory is created lazily during training. This prevents race conditions when multiple threads or processes attempt to write simultaneously.
- **FileExistsError on collision.** If the target directory already exists, raise immediately rather than silently merging. The caller is responsible for generating a unique run ID (see the collision resolution strategy in the Run ID Format section).
- **Phase-specific checkpoint directories.** Only create `checkpoints/phase{N}/` for the phases that will actually run. A single-phase run for Phase 4 creates only `checkpoints/phase4/`. A full pipeline run creates all seven.

---

## Path Resolution Examples

Given a run with ID `2026-02-20_23-15-02_phase4_dev_a1b2c3d` under the default `runs/` base directory, the following absolute paths apply:

| Artifact | Path |
|---|---|
| Manifest | `runs/2026-02-20_23-15-02_phase4_dev_a1b2c3d/manifest.json` |
| Lock file | `runs/2026-02-20_23-15-02_phase4_dev_a1b2c3d/manifest.lock.json` |
| Step 5000 checkpoint | `runs/2026-02-20_23-15-02_phase4_dev_a1b2c3d/checkpoints/phase4/ckpt_step00005000.pt` |
| Best val_loss checkpoint | `runs/2026-02-20_23-15-02_phase4_dev_a1b2c3d/checkpoints/phase4/ckpt_best_val_loss.pt` |
| Phase boundary | `runs/2026-02-20_23-15-02_phase4_dev_a1b2c3d/checkpoints/phase4/phase_boundary.pt` |
| TensorBoard events | `runs/2026-02-20_23-15-02_phase4_dev_a1b2c3d/logs/tensorboard/events.out.tfevents.*` |
| Metric log | `runs/2026-02-20_23-15-02_phase4_dev_a1b2c3d/metrics.jsonl` |
| Hardware info | `runs/2026-02-20_23-15-02_phase4_dev_a1b2c3d/artifacts/hardware.json` |
| Run summary | `runs/2026-02-20_23-15-02_phase4_dev_a1b2c3d/reports/summary.json` |

Within the manifest and summary files, use paths relative to the run root (e.g., `checkpoints/phase4/phase_boundary.pt`). In Python code and CLI output, resolve to absolute paths using the `RunDirectory` object.
