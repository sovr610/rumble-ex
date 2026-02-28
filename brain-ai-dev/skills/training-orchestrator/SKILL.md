---
name: Seven-Phase Training Orchestrator + Reproducible Manifests + Automatic Ablations
description: >-
  This skill should be used when the user asks to "implement training orchestrator",
  "add reproducible manifests", "implement automatic ablations", "add phase boundary validation",
  "implement run manifests", "add experiment tracking", "implement checkpoint naming",
  "add dataset fingerprinting", "implement seeding policy", "add determinism enforcement",
  "implement ablation matrix", "add phase resume", "implement tensorboard logging",
  "add wandb integration", "implement train_full_pipeline", "add run directory layout",
  "implement phase boundary artifacts", "add ablation CSV output", "implement manifest schema",
  "add dataset hashing", "implement worker seeding", "add checkpoint standardization",
  "implement dev/production modes", "add experiment provenance",
  or mentions training orchestration, reproducible experiments, manifest-driven runs,
  automatic ablations, phase boundary validation, dataset fingerprinting, seeding policy,
  checkpoint naming conventions, run directory layout, or seven-phase training pipeline
  in the cognitive pipeline.
version: 0.1.0
---

# Seven-Phase Training Orchestrator + Reproducible Manifests + Automatic Ablations

## Purpose

This skill standardizes the experiment control plane for the entire brain_ai cognitive architecture.
It enforces that every training run — across all seven phases — is reproducible (within defined
tolerance), resumable from any phase boundary, auditable via machine-readable manifests, and
comparable across automatic ablation matrices. The non-negotiable goals are: manifest-driven
provenance, deterministic seeding, standardized run directories, phase boundary validation, and
push-button ablations.

## Key Files

| Target Module | Template Asset | Purpose |
|---|---|---|
| `scripts/train_phase{1..7}.py` | `assets/phase_runner_template.py` | Per-phase training scripts with dev/production modes |
| `scripts/train_full_pipeline.py` | `assets/full_pipeline_template.py` | Multi-phase orchestrator with resume + boundary validation |
| `brain_ai/training/manifest.py` | `assets/manifest_template.py` | RunManifest: provenance capture, serialization, reproduction |
| `brain_ai/training/seeding.py` | `assets/seeding_template.py` | Determinism enforcement, RNG stream isolation, worker seeding |
| `brain_ai/training/checkpointing.py` | `assets/checkpointing_template.py` | Standardized naming, phase boundary artifacts, resume logic |
| `brain_ai/training/ablation.py` | `assets/ablation_template.py` | Ablation matrix generation, execution, CSV summarization |
| `brain_ai/training/logging_utils.py` | `assets/logging_utils_template.py` | TensorBoard + W&B wrappers, metric namespaces, resume semantics |
| `brain_ai/training/dataset_fingerprint.py` | `assets/dataset_fingerprint_template.py` | Tiered dataset hashing, HF fingerprints, shard manifests |
| `brain_ai/training/phase_boundary.py` | `assets/phase_boundary_template.py` | Phase boundary validation, artifact compatibility checks |
| `brain_ai/config.py` (extend) | `assets/training_config_template.py` | TrainingConfig, ManifestConfig, AblationConfig, SeedConfig |

## Public Contract

```python
# RunManifest
capture(config, phase, mode, ...) -> manifest_dict
save(run_dir) / load(run_dir) -> RunManifest
validate_reproduction(original, current, tolerance) -> bool

# PhaseOrchestrator
run_phase(phase, config, *, resume_from=None) -> PhaseResult
run_pipeline(phases, config, *, start_phase=1) -> PipelineResult
validate_boundary(phase, run_dir) -> BoundaryStatus

# AblationRunner
generate_matrix(spec) -> List[AblationRun]
execute(matrix, *, parallel=False) -> AblationReport
summarize(ablation_id) -> DataFrame  # writes ablations.csv
```

## Run Directory Layout

```
runs/<run_id>/
  manifest.json              # Complete provenance snapshot
  manifest.lock.json         # Resolved config + dependency resolution
  stdout.log / stderr.log
  logs/
    tensorboard/             # Event files
    wandb/                   # If enabled
  checkpoints/
    phase{N}/
      ckpt_step00001234.pt
      ckpt_best_val_loss.pt
      phase_boundary.pt      # Canonical artifact for next phase
  artifacts/
    datasets.json            # Dataset fingerprints/hashes
    env.txt / pip_freeze.txt
    git_diff.patch           # If dirty working tree
    hardware.json
  reports/
    summary.json             # Final metrics, best metrics, run status
```

## Manifest Schema

Every run produces `manifest.json` containing:

| Section | Key Fields | Purpose |
|---|---|---|
| Identity | `run_id`, `ablation_id`, `phase`, `mode`, `timestamp` | Run identification |
| Code provenance | `git.commit`, `git.branch`, `git.dirty`, `git.patch` | Code version tracking |
| Config | `config.full`, `config.overrides`, `config.resolved` | Complete configuration |
| Seeds | `base_seed`, `per_phase_offsets`, `deterministic_flags` | Reproduction seeds |
| Environment | `python`, `torch`, `cuda`, `pip_freeze` | Software environment |
| Data | `datasets[].name`, `.fingerprint`, `.split`, `.version` | Data provenance |
| Resume | `resume_from`, `phase_boundary_artifacts` | Resume chain |

See `references/manifest-schema.md` for the complete JSON schema and validation rules.

## Seeding and Determinism Policy

Enforce deterministic training where possible:

- Set Python, NumPy, and Torch seeds from `base_seed + phase_offset`
- Disable cuDNN benchmark; enable deterministic mode
- Worker seeding: `worker_seed = base_seed + worker_id + epoch * 1000`
- Separate RNG streams for augmentation, dropout, and sampling
- Record all flags in manifest; optionally enable `torch.use_deterministic_algorithms(True)`

See `references/seeding-determinism.md` for RNG stream isolation, worker seeding derivation,
and tolerance definitions.

## Dataset Fingerprinting (Tiered)

| Tier | Source | Strategy |
|---|---|---|
| 1 | HuggingFace datasets | Record name/config/split/version/fingerprint |
| 2 | Local files/shards | Path + size + mtime + fast hash (first/last N MB SHA256) |
| 3 | Sampling identity | Record exact sample indices + RNG seeds for subsets |

See `references/dataset-fingerprinting.md` for hashing algorithms, performance, and validation.

## Phase Boundary Validation

Before Phase N starts, validate:
- Required artifacts exist at expected paths
- `workspace_dim`, vocab size, enabled modules match expectations
- Checkpoint schema version matches
- Dataset identity matches cross-phase expectations

Fail fast on mismatch. See `references/phase-boundaries.md` for per-phase artifact contracts
and compatibility matrix.

## Automatic Ablations

Define ablation specs as toggle matrices:

```yaml
toggles:
  use_engram: [false, true]
  use_learnable_delays: [false, true]
  use_ltn: [false, true]
```

Matrix execution generates derived `run_id` per combination, writes `ablations.csv` with
overrides, metrics, status, and duration per run.

See `references/ablation-system.md` for spec DSL, pairwise reduction, execution strategies,
and CSV schema.

## Logging (TensorBoard + W&B)

Stable metric namespace: `train/loss`, `val/loss`, `phase{N}/metric_name`.
Track `global_step` and `phase_step` separately. W&B resume uses stored run ID from manifest.

See `references/logging-integration.md` for namespace conventions, resume semantics, and
histogram/image logging patterns.

## Configuration Surface

### TrainingConfig

| Field | Default | Purpose |
|---|---|---|
| `mode` | `"dev"` | `"dev"`, `"production"` |
| `phases` | `[1,2,3,4,5,6,7]` | Which phases to run |
| `start_phase` | 1 | Resume from this phase |
| `run_dir` | `"runs/"` | Base directory for run outputs |
| `use_amp` | False | Automatic mixed precision |

### ManifestConfig

| Field | Default | Purpose |
|---|---|---|
| `capture_git_diff` | True | Save git diff if dirty |
| `capture_pip_freeze` | True | Save pip freeze output |
| `capture_hardware` | True | Save hardware info |
| `dataset_fingerprint_tier` | `"auto"` | `"auto"`, `"tier1"`, `"tier2"`, `"tier3"` |

### SeedConfig

| Field | Default | Purpose |
|---|---|---|
| `base_seed` | 1337 | Global seed |
| `per_phase_offsets` | `[0,100,200,...]` | Per-phase seed offsets |
| `enforce_deterministic` | True | `torch.use_deterministic_algorithms` in dev |
| `cudnn_benchmark` | False | Disable cuDNN benchmark |

### AblationConfig

| Field | Default | Purpose |
|---|---|---|
| `spec_file` | None | Path to ablation YAML spec |
| `mode` | `"full"` | `"full"` (all combos), `"pairwise"` (reduced) |
| `parallel` | False | Run ablations in parallel |
| `phases` | `[4]` | Which phases to ablate |

Presets: `TrainingFullConfig.minimal()`, `.dev()`, `.production()`.

## Done-When Gates

| Gate | Test | Threshold |
|---|---|---|
| **(a) Manifest reproduction** | Dev-mode run per phase; re-run from manifest; metrics match | Loss within 2% relative, accuracy within 1% absolute |
| **(b) Ablation matrix execution** | 2x2x2 = 8 runs in dev mode; all produce manifests; ablations.csv complete; failures recorded | All 8 runs complete, CSV has 8 rows |
| **(c) Phase resume from boundary** | Run phases 1-3; resume at phase 4; resume at phase 7; boundary checks catch mismatches | No silent misuse; fail-fast on mismatch |

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Metrics differ on re-run | Non-deterministic ops or unseeded RNG | Enable strict determinism; check all RNG streams |
| Phase boundary silently mismatched | No validation before phase start | Enable phase boundary validation; fail fast |
| W&B resume creates duplicate run | Run ID not stored in manifest | Store W&B run ID in manifest; use `resume="must"` |
| Ablation overwrites prior run | Shared run_id across ablation combos | Derive unique run_id per ablation combination |
| Checkpoint naming collision | No phase/step encoding in filename | Use standardized `ckpt_step{N:08d}.pt` naming |
| Dataset changed between phases | No fingerprint validation | Validate dataset identity at phase boundaries |
| Worker seeding non-deterministic | Workers not seeded from base_seed | Use `worker_seed = base_seed + worker_id + epoch*1000` |
| Resume corrupts original manifest | Overwrite instead of append | Never overwrite; append `resume_events` entries |

## Anti-Patterns

- **No manifest** -- every run must produce provenance; no exceptions
- **Hardcoded seeds** -- always derive from SeedConfig, never magic numbers
- **Overwriting runs on resume** -- resume is append-only; new run_id for forks
- **Skipping boundary validation** -- silent mismatches cause weeks of wasted compute
- **Manual ablation edits** -- ablation specs must be machine-generated from YAML
- **Mixing global_step and phase_step** -- track both separately; log both
- **fp16 seed computation** -- all seeding and hashing in int64/int32
- **No dataset fingerprint** -- dataset provenance is mandatory, not optional

## Additional Resources

### Reference Files

- **`references/manifest-schema.md`** -- Complete JSON schema, validation rules, versioning, reproduction protocol
- **`references/seeding-determinism.md`** -- RNG stream isolation, worker seeding, tolerance definitions, PyTorch determinism flags
- **`references/dataset-fingerprinting.md`** -- Tiered hashing, HF fingerprints, shard manifests, validation, performance
- **`references/phase-boundaries.md`** -- Per-phase artifact contracts, compatibility matrix, validation checks, fail-fast policy
- **`references/ablation-system.md`** -- Spec DSL, matrix generation, pairwise reduction, execution, CSV schema
- **`references/logging-integration.md`** -- TensorBoard + W&B namespaces, resume semantics, histogram/image patterns
- **`references/run-directory-layout.md`** -- Directory structure, file naming, cleanup policy, archival

### Asset Templates

- **`assets/manifest_template.py`** -- RunManifest capture, serialization, validation, reproduction check, self-test
- **`assets/seeding_template.py`** -- SeedManager, RNG streams, worker seeding, determinism enforcement, self-test
- **`assets/checkpointing_template.py`** -- CheckpointManager, standardized naming, phase boundary artifacts, resume, self-test
- **`assets/ablation_template.py`** -- AblationSpec, matrix generation, execution, CSV output, self-test
- **`assets/logging_utils_template.py`** -- MetricLogger, TensorBoard + W&B wrappers, namespace, resume, self-test
- **`assets/dataset_fingerprint_template.py`** -- DatasetFingerprinter, tiered hashing, validation, self-test
- **`assets/phase_boundary_template.py`** -- PhaseBoundaryValidator, artifact checks, compatibility matrix, self-test
- **`assets/phase_runner_template.py`** -- PhaseRunner, dev/production modes, per-phase training loop, self-test
- **`assets/full_pipeline_template.py`** -- PipelineOrchestrator, phase chaining, resume, boundary validation, self-test
- **`assets/training_config_template.py`** -- All configs, presets, serialization, self-test

### Scripts

- **`scripts/validate_orchestrator.py`** -- Runtime contract validation (manifest reproduction, ablation execution, phase resume)
- **`scripts/gen_orchestrator_tests.py`** -- Generates `tests/test_orchestrator.py` (~120+ test cases)
- **`scripts/run_ablation.py`** -- CLI for executing ablation specs from YAML files
