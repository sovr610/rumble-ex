# Phase Boundary Validation Reference

This document defines the artifact contracts, compatibility checks, and validation policy
for every phase boundary in the seven-phase training pipeline. Every phase transition
produces a `phase_boundary.pt` file that the next phase consumes. Treat this document as
the authoritative specification for what each boundary artifact must contain, how the
validator inspects it, and what happens when checks fail.

---

## 1. Phase Boundary Artifact Contracts

Each phase produces a boundary artifact consumed by the immediately following phase. The
artifact is a PyTorch checkpoint dictionary saved as `phase_boundary.pt` in the run's
checkpoint directory. Below is the per-transition contract.

### Phase 1 to 2: SNN Core to Modality Encoders

| Field | Contents |
|---|---|
| `model_state_dict` | SNN core weights: LIF neuron parameters, learnable synaptic delays, heterogeneous tau values, surrogate gradient function state |
| `config_snapshot.snn` | Full `SNNConfig` snapshot (beta, num_timesteps, surrogate, hidden_sizes, use_learnable_delays, use_heterogeneous_tau, max_delay) |
| `compatibility.snn_hidden_sizes` | List of hidden layer dimensions, e.g. `[4096, 4096, 2048, 2048]` |
| `compatibility.snn_output_dim` | Output dimension of the SNN core |
| `compatibility.surrogate` | Surrogate gradient function name (`"atan"`, `"fast_sigmoid"`, `"straight_through"`) |

Phase 2 uses these weights to initialize the SNN backbone inside each modality encoder.
The encoder wraps the pretrained SNN core with modality-specific input projections.

### Phase 2 to 3: Modality Encoders to HTM

| Field | Contents |
|---|---|
| `model_state_dict` | All encoder weights (vision, text, audio, sensors, engram if enabled) plus SNN core weights from Phase 1 |
| `config_snapshot.encoder` | Full `EncoderConfig` snapshot |
| `config_snapshot.snn` | Carried forward from Phase 1 boundary |
| `compatibility.workspace_dim` | Must equal `encoder.output_dim` (default 4096). This is the critical integration dimension. |
| `compatibility.enabled_encoders` | List of encoder names present, e.g. `["vision", "text", "audio", "sensors"]` |
| `compatibility.vocab_size` | Text encoder vocabulary size (default 128000) |
| `compatibility.tokenizer_hash` | SHA256 of tokenizer vocabulary file, if applicable |

Phase 3 receives encoder outputs at `workspace_dim` and feeds them into the HTM spatial
pooler. The validator verifies that every enabled encoder is present in the state dict and
that `encoder.output_dim == workspace_dim`.

### Phase 3 to 4: HTM to Global Workspace

| Field | Contents |
|---|---|
| `model_state_dict` | HTM layer state (spatial pooler permanences, temporal memory connections, anomaly thresholds, reflex memory if enabled), plus all prior encoder and SNN weights |
| `config_snapshot.htm` | Full `HTMConfig` snapshot (column_count, cells_per_column, sparsity, activation_threshold, reflex settings) |
| `compatibility.htm_column_count` | Number of HTM columns (default 16384) |
| `compatibility.htm_cells_per_column` | Cells per column (default 64) |
| `compatibility.htm_input_dim` | Must match `workspace_dim` from Phase 2 boundary |

Phase 4 integrates HTM temporal memory into the global workspace competition loop. The
validator checks that the HTM input dimension matches the encoder output dimension and
that the spatial pooler column count matches the current config.

### Phase 4 to 5: Global Workspace to Active Inference

| Field | Contents |
|---|---|
| `model_state_dict` | Global workspace weights (competition attention, broadcast adapters, working memory CfC/LTC parameters), plus all prior weights |
| `config_snapshot.workspace` | Full `WorkspaceConfig` snapshot (workspace_dim, num_heads, capacity_limit, memory_mode, ignition_threshold, selection_rounds) |
| `compatibility.workspace_dim` | Must match across all phases (default 4096) |
| `compatibility.ignition_threshold` | Global ignition threshold (default 0.3) |
| `compatibility.working_memory_capacity` | Miller's Law capacity limit (default 7) |
| `compatibility.num_modalities` | Number of modality projections in the workspace |

Phase 5 uses the workspace output as the observation input to the active inference
generative model. The validator verifies workspace output dimension matches the expected
observation dimension for the state encoder.

### Phase 5 to 6: Active Inference to Reasoning

| Field | Contents |
|---|---|
| `model_state_dict` | Active inference model (generative model encoder/decoder, EFE decomposition networks for pragmatic/epistemic/instrumental value, policy planner weights, empowerment estimator), plus all prior weights |
| `config_snapshot.decision` | Full `DecisionConfig` snapshot (planning_horizon, num_policies, EFE settings, amortized policy flag) |
| `compatibility.generative_model_state_dim` | Internal state dimension of the generative model |
| `compatibility.efe_components` | List of EFE components present: `["pragmatic", "epistemic", "instrumental"]` |
| `compatibility.num_policies` | Number of discrete policies (default 128) |

Phase 6 receives the workspace-integrated representation (after active inference
processing) and routes it through dual-process System 1/System 2 reasoning. The validator
checks that the generative model state dimension aligns with the workspace dimension.

### Phase 6 to 7: Reasoning to Meta-Learning

| Field | Contents |
|---|---|
| `model_state_dict` | Dual-process reasoning weights (System 1 fast network, System 2 iterative GRU, confidence calibration parameters, fuzzy logic operators, symbolic knowledge base embeddings, LTN grounding networks if enabled), plus all prior weights |
| `config_snapshot.reasoning` | Full `ReasoningConfig` snapshot (hidden_dim, num_reasoning_steps, confidence_threshold, System 2 layers/heads, LTN settings) |
| `compatibility.reasoning_hidden_dim` | Must match workspace_dim |
| `compatibility.confidence_calibration_present` | Boolean: True if calibration params are in state dict |
| `compatibility.symbolic_kb_size` | Number of entities/predicates in the knowledge base |
| `compatibility.system2_present` | Boolean: True if System 2 weights are present |

### Phase 7 Output: Final Model Checkpoint

Phase 7 produces two artifacts:

1. **`phase_boundary.pt`** -- The complete model with meta-learned initialization
   weights. This is the canonical "release" checkpoint.
2. **`meta_init.pt`** -- MAML/FOMAML/Reptile initialization point for rapid few-shot
   adaptation at deployment time.

The final boundary includes:

| Field | Contents |
|---|---|
| `model_state_dict` | Complete model: all seven layers trained end-to-end with meta-learning initialization |
| `config_snapshot` | Complete `BrainAIConfig` |
| `meta_init` | Meta-learned parameter initialization (subset used for inner-loop adaptation) |
| `compatibility` | Full compatibility dict from all prior phases |
| `metadata.meta_val_accuracy` | Few-shot validation accuracy (5-way 1-shot) |

---

## 2. `phase_boundary.pt` Format Specification

Every boundary artifact conforms to the following schema:

```python
{
    # -- Identity --
    "schema_version": "1.0",            # Semantic version of this format
    "source_phase": 4,                  # Phase that produced this artifact
    "target_phase": 5,                  # Phase that will consume this artifact
    "source_run_id": "20260220_143022_a1b2c3",  # Unique run identifier

    # -- Model State --
    "model_state_dict": {               # Only weights needed by next phase
        "snn_core.lif0.weight": ...,
        "encoders.vision.conv1.weight": ...,
        "htm.spatial_pooler.permanences": ...,
        "workspace.competition.qkv.weight": ...,
        # ...
    },

    # -- Configuration --
    "config_snapshot": {                # BrainAIConfig serialized as nested dict
        "snn": {"beta": 0.95, "num_timesteps": 50, ...},
        "encoder": {"output_dim": 4096, ...},
        "htm": {"column_count": 16384, ...},
        "workspace": {"workspace_dim": 4096, ...},
        "decision": {"planning_horizon": 8, ...},
        "reasoning": {"hidden_dim": 4096, ...},
        "meta": {"inner_lr": 0.001, ...},
    },

    # -- Feature Flags --
    "feature_flags": {
        "use_snn": True,
        "use_htm": True,
        "use_workspace": True,
        "use_symbolic": True,
        "use_meta": True,
        "use_engram": True,
    },

    # -- Dimensional Compatibility --
    "compatibility": {
        "workspace_dim": 4096,
        "vocab_size": 128000,
        "num_encoders": 4,
        "snn_hidden_sizes": [4096, 4096, 2048, 2048],
        "snn_output_dim": 2048,
        "htm_column_count": 16384,
        "htm_cells_per_column": 64,
        "htm_input_dim": 4096,
        "ignition_threshold": 0.3,
        "working_memory_capacity": 7,
        "num_modalities": 4,
        "generative_model_state_dim": 64,
        "num_policies": 128,
        "reasoning_hidden_dim": 4096,
    },

    # -- Provenance Metadata --
    "metadata": {
        "best_metric": {"val_loss": 0.342, "val_accuracy": 0.891},
        "training_steps": 50000,
        "timestamp": "2026-02-20T14:30:22Z",
        "wall_time_seconds": 86400,
        "dataset_fingerprint": "sha256:abc123...",
    },
}
```

### Schema Version Contract

The `schema_version` field uses semantic versioning:

- **Major bump** (e.g. `"1.0"` to `"2.0"`): Breaking changes to required fields.
  Boundary validator must reject version mismatches on the major component.
- **Minor bump** (e.g. `"1.0"` to `"1.1"`): Additive changes (new optional fields).
  Boundary validator accepts minor version differences with a warning.

---

## 3. Compatibility Matrix

The table below defines what each phase validates before starting. "Source" is the
boundary artifact; "Current" is the live `BrainAIConfig` for the new phase.

| Target Phase | Check | Source Field | Current Config Field | Match Rule |
|---|---|---|---|---|
| 2 | Workspace dim | `compatibility.snn_output_dim` | `encoder.output_dim` | Exact or projection exists |
| 2 | SNN hidden sizes | `compatibility.snn_hidden_sizes` | `snn.hidden_sizes` | Exact match |
| 2 | Surrogate function | `compatibility.surrogate` | `snn.surrogate` | Exact match |
| 3 | Encoder output dim | `compatibility.workspace_dim` | `workspace.workspace_dim` | Exact match |
| 3 | All encoders present | `compatibility.enabled_encoders` | `modalities` | Source is superset of current |
| 3 | Vocab size | `compatibility.vocab_size` | `encoder.text_vocab_size` | Exact match (if text enabled) |
| 4 | HTM input dim | `compatibility.htm_input_dim` | `workspace.workspace_dim` | Exact match |
| 4 | SP column count | `compatibility.htm_column_count` | `htm.column_count` | Exact match |
| 4 | Cells per column | `compatibility.htm_cells_per_column` | `htm.cells_per_column` | Exact match |
| 5 | Workspace output dim | `compatibility.workspace_dim` | `workspace.workspace_dim` | Exact match |
| 5 | Working memory capacity | `compatibility.working_memory_capacity` | `workspace.capacity_limit` | Exact match |
| 5 | Num modalities | `compatibility.num_modalities` | `len(modalities)` | Exact match |
| 6 | Generative model dim | `compatibility.generative_model_state_dim` | `decision.hidden_dim` or inferred | Exact match |
| 6 | EFE components | `compatibility.efe_components` | Inferred from `decision.use_improved_efe` | All required components present |
| 7 | Reasoning weights | `compatibility.system2_present` | `use_symbolic` | True if current config enables symbolic |
| 7 | Calibration params | `compatibility.confidence_calibration_present` | Always required if symbolic=True | Must be True |
| 7 | Reasoning hidden dim | `compatibility.reasoning_hidden_dim` | `reasoning.hidden_dim` | Exact match |

### Critical Invariant: `workspace_dim`

The `workspace_dim` value (default 4096) threads through every phase boundary. All
encoders output this dimension. HTM receives this dimension. The global workspace
operates in this dimension. Active inference observes this dimension. Reasoning processes
this dimension. If `workspace_dim` differs between any two adjacent phases, the boundary
validator must reject the transition.

---

## 4. Validation Checks

Execute the following checks in order before each phase starts. Halt on the first HARD
failure. Accumulate SOFT warnings and print them as a summary before proceeding.

### 4.1 File Existence

```python
def check_file_exists(run_dir: Path, source_phase: int) -> None:
    boundary_path = run_dir / "checkpoints" / f"phase{source_phase}" / "phase_boundary.pt"
    if not boundary_path.exists():
        raise HardFailure(
            f"Missing boundary artifact: {boundary_path}. "
            f"Phase {source_phase} must complete before phase {source_phase + 1} can start."
        )
```

### 4.2 Schema Version

```python
def check_schema_version(artifact: dict, expected_major: int = 1) -> None:
    version = artifact.get("schema_version", "0.0")
    major = int(version.split(".")[0])
    minor = int(version.split(".")[1])
    if major != expected_major:
        raise HardFailure(
            f"Schema version mismatch: artifact has v{version}, "
            f"expected major version {expected_major}"
        )
    if minor > CURRENT_MINOR:
        warn(f"Artifact schema v{version} is newer than validator v{CURRENT_MAJOR}.{CURRENT_MINOR}")
```

### 4.3 State Dict Key Validation

```python
def check_state_dict_keys(artifact: dict, expected_keys: set, phase: int) -> None:
    actual_keys = set(artifact["model_state_dict"].keys())
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys

    if missing:
        raise HardFailure(
            f"Phase {phase} boundary missing required state dict keys: {missing}"
        )
    if extra:
        warn(f"Phase {phase} boundary has {len(extra)} extra state dict keys (ignored): "
             f"{list(extra)[:5]}...")
```

### 4.4 Dimension Checks

```python
def check_dimensions(artifact: dict, current_config: BrainAIConfig, target_phase: int) -> None:
    compat = artifact["compatibility"]

    # workspace_dim is always checked
    if compat["workspace_dim"] != current_config.workspace.workspace_dim:
        raise HardFailure(
            f"workspace_dim mismatch: boundary has {compat['workspace_dim']}, "
            f"current config has {current_config.workspace.workspace_dim}"
        )

    # Phase-specific dimension checks (see compatibility matrix above)
    phase_checks = PHASE_DIMENSION_CHECKS[target_phase]
    for check_name, source_field, config_value in phase_checks:
        source_value = compat.get(source_field)
        if source_value is not None and source_value != config_value:
            raise HardFailure(
                f"{check_name}: boundary has {source_value}, config has {config_value}"
            )
```

### 4.5 Feature Flag Consistency

```python
def check_feature_flags(artifact: dict, current_config: BrainAIConfig) -> None:
    source_flags = artifact["feature_flags"]
    flag_pairs = [
        ("use_snn",       current_config.use_snn),
        ("use_htm",       current_config.use_htm),
        ("use_workspace", current_config.use_workspace),
        ("use_symbolic",  current_config.use_symbolic),
        ("use_meta",      current_config.use_meta),
        ("use_engram",    current_config.use_engram),
    ]
    for flag_name, current_value in flag_pairs:
        source_value = source_flags.get(flag_name, False)
        if source_value and not current_value:
            # Disabling a module that was enabled upstream
            warn(
                f"Feature flag '{flag_name}' was True in source phase "
                f"but is False in current config. Downstream weights for "
                f"this module will be discarded. Confirm this is intentional."
            )
        if not source_value and current_value:
            # Enabling a module that was not trained upstream
            raise HardFailure(
                f"Feature flag '{flag_name}' is True in current config but "
                f"was False in source phase. Cannot enable a module that has "
                f"no pretrained weights. Train it in the source phase first."
            )
```

### 4.6 Config Compatibility

```python
def check_config_compatibility(
    artifact: dict,
    current_config: BrainAIConfig,
    critical_fields: list[str],
) -> None:
    source_config = artifact["config_snapshot"]

    for field_path in critical_fields:
        # field_path is dotted, e.g. "encoder.output_dim"
        source_val = get_nested(source_config, field_path)
        current_val = get_nested_from_config(current_config, field_path)
        if source_val != current_val:
            raise HardFailure(
                f"Config mismatch on critical field '{field_path}': "
                f"boundary has {source_val}, current has {current_val}"
            )
```

Critical fields checked at every boundary:

```python
ALWAYS_CRITICAL = [
    "encoder.output_dim",
    "workspace.workspace_dim",
]

PER_PHASE_CRITICAL = {
    2: ["snn.hidden_sizes", "snn.surrogate", "snn.num_timesteps"],
    3: ["encoder.text_vocab_size"],
    4: ["htm.column_count", "htm.cells_per_column"],
    5: ["workspace.num_heads", "workspace.capacity_limit"],
    6: ["decision.planning_horizon", "decision.num_policies"],
    7: ["reasoning.hidden_dim", "reasoning.confidence_threshold"],
}
```

---

## 5. Fail-Fast Policy

### Severity Levels

| Level | Behavior | Examples |
|---|---|---|
| **HARD** | Abort immediately. Print error. Exit with non-zero code. | Missing boundary file, schema major version mismatch, dimension mismatch, missing required state dict keys, enabling untrained module |
| **SOFT** | Log warning. Continue execution. | Extra state dict keys, non-critical config differences, newer schema minor version, metadata field missing, disabling a previously enabled module |

### Validation Modes

Control severity interpretation with the `boundary_validation_mode` config field:

```python
boundary_validation_mode: str = "normal"  # "strict", "normal", "permissive"
```

| Mode | HARD failures | SOFT failures |
|---|---|---|
| `"strict"` | Abort | Abort (treats all SOFT as HARD) |
| `"normal"` | Abort | Warn and continue |
| `"permissive"` | Warn and continue | Warn and continue |

Use `"strict"` for production training where any deviation is unacceptable. Use
`"permissive"` only for debugging or exploratory ablations where you intentionally
diverge from the validated path.

### Validation Execution Order

Execute checks in this order. Stop at the first HARD failure (in `"normal"` or
`"strict"` mode).

```
1. File existence
2. Schema version
3. Source/target phase numbers match expectation
4. Feature flag consistency
5. Dimension checks (workspace_dim first, then phase-specific)
6. State dict key validation
7. Config compatibility on critical fields
8. Dataset fingerprint validation (if cross-phase dataset match required)
```

---

## 6. Phase Boundary Forking

### Use Case

Ablation experiments commonly train phases 1 through N once, then fork at phase N+1 with
multiple different configurations. For example:

```
Phase 1 --> Phase 2 --> Phase 3 --> Phase 4 (config A)
                                \-> Phase 4 (config B)
                                \-> Phase 4 (config C)
```

### Rules

1. **New run ID per fork.** Each downstream run receives a unique `run_id` derived from
   the ablation spec (e.g. `base_run_id + "_ablation_" + combo_hash`).

2. **Reference, do not copy.** The fork's `phase_boundary.pt` loader reads the upstream
   artifact by path. It does not duplicate the file. Record the source path in the
   downstream manifest:

   ```python
   manifest["resume"]["source_boundary_path"] = "/runs/base_run/checkpoints/phase3/phase_boundary.pt"
   manifest["resume"]["source_run_id"] = "20260220_143022_a1b2c3"
   ```

3. **Source artifact is immutable.** Never modify a `phase_boundary.pt` file after the
   producing phase completes. Downstream phases are consumers, not writers.

4. **Validation still applies.** Even though the boundary was validated for the original
   downstream phase, re-validate it against the forked config. A fork may change
   `workspace_dim` or feature flags, which must be caught.

5. **Multiple consumers.** Any number of downstream runs may share the same upstream
   boundary artifact. The file system path is the canonical reference.

---

## 7. Boundary Artifact Minimality

### Principle

A `phase_boundary.pt` file contains only what the next phase needs to initialize its
model. It does not contain optimizer state, learning rate scheduler state, RNG state, or
gradient buffers. These belong in the full checkpoint (`ckpt_best.pt` or
`ckpt_step{N:08d}.pt`), which is used for resuming within the same phase.

### Rationale

- Boundary artifacts may be shared across ablation forks. Including optimizer state would
  bias the fork toward the source phase's optimization trajectory.
- Boundary artifacts are often smaller than full checkpoints (no Adam moment buffers,
  which double the size).
- Boundary artifacts are the "public API" between phases; full checkpoints are
  "private state" within a phase.

### Size Comparison (7B scale estimate)

| Artifact | Contents | Approximate Size |
|---|---|---|
| `phase_boundary.pt` | Model weights + config + compatibility + metadata | ~14 GB |
| `ckpt_best.pt` | Model weights + optimizer state + scheduler + RNG | ~42 GB |

### What to Include vs. Exclude

| Include in `phase_boundary.pt` | Exclude (full checkpoint only) |
|---|---|
| `model_state_dict` (pruned to next-phase-relevant keys) | `optimizer_state_dict` |
| `config_snapshot` | `scheduler_state_dict` |
| `feature_flags` | `rng_state` (torch, numpy, python) |
| `compatibility` | `scaler_state_dict` (AMP) |
| `metadata` (best metrics, steps, timestamp) | `epoch` / `global_step` for resume |
| `schema_version`, `source_phase`, `target_phase`, `source_run_id` | `gradient_accumulation_count` |

---

## 8. Cross-Phase Dataset Validation

### Dataset Sharing Rules

Not all phases use the same data. The table below defines which phase transitions require
dataset fingerprint consistency and which allow different datasets.

| Transition | Dataset Match Required? | Rationale |
|---|---|---|
| 1 to 2 | Yes (vision) | Phase 2 encoders must process the same visual distribution the SNN core learned |
| 2 to 3 | Yes (all modalities) | HTM learns temporal patterns over the same encoded feature space |
| 3 to 4 | Yes (all modalities) | Global workspace integrates the same modality representations |
| 4 to 5 | Partial | Active inference may introduce RL environments. Vision/text data should match; RL data is new. |
| 5 to 6 | Partial | Reasoning may add reasoning-specific datasets (GSM8K, ARC, bAbI). Base modality data should match. |
| 6 to 7 | No | Meta-learning uses entirely different datasets (Omniglot, mini-ImageNet, meta-dataset). Dataset change is expected. |

### Fingerprint Validation

```python
def check_dataset_fingerprint(
    artifact: dict,
    current_datasets: dict,
    target_phase: int,
) -> None:
    source_fingerprint = artifact["metadata"].get("dataset_fingerprint")
    if source_fingerprint is None:
        warn("Source boundary has no dataset fingerprint. Cannot verify data consistency.")
        return

    transition = (artifact["source_phase"], target_phase)

    if transition in DATASET_MATCH_REQUIRED:
        required_modalities = DATASET_MATCH_REQUIRED[transition]
        for modality in required_modalities:
            source_fp = source_fingerprint.get(modality)
            current_fp = current_datasets.get(modality, {}).get("fingerprint")
            if source_fp and current_fp and source_fp != current_fp:
                raise HardFailure(
                    f"Dataset fingerprint mismatch for '{modality}' at "
                    f"phase {artifact['source_phase']}->{target_phase}: "
                    f"source={source_fp[:16]}..., current={current_fp[:16]}..."
                )

    elif transition in DATASET_PARTIAL_MATCH:
        # Warn on mismatch but do not fail
        for modality in DATASET_PARTIAL_MATCH[transition]:
            source_fp = source_fingerprint.get(modality)
            current_fp = current_datasets.get(modality, {}).get("fingerprint")
            if source_fp and current_fp and source_fp != current_fp:
                warn(
                    f"Dataset fingerprint changed for '{modality}' at "
                    f"phase {artifact['source_phase']}->{target_phase}. "
                    f"This may be intentional (new dataset added)."
                )
    # transition (6, 7): no check needed
```

### Fingerprint Storage

Record dataset fingerprints in the boundary artifact's `metadata` section:

```python
"metadata": {
    "dataset_fingerprint": {
        "vision": "sha256:a1b2c3d4...",
        "text": "sha256:e5f6a7b8...",
        "audio": "sha256:c9d0e1f2...",
        "rl_env": "grid_world_v2:seed=42",  # RL environment identifier
    },
    ...
}
```

---

## 9. Resume vs. Fresh Start

Two distinct operations load weights from disk. Confusing them causes subtle bugs
(wrong learning rate, stale optimizer momentum, broken RNG reproducibility).

### Resume Within Phase

**Purpose:** Continue training the current phase after interruption (crash, preemption,
manual pause).

**What it loads:**
```python
checkpoint = torch.load("ckpt_step00050000.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
torch.set_rng_state(checkpoint["rng_state"]["torch"])
np.random.set_state(checkpoint["rng_state"]["numpy"])
random.setstate(checkpoint["rng_state"]["python"])
if scaler:
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
global_step = checkpoint["global_step"]
epoch = checkpoint["epoch"]
```

**Invariant:** After resume, the training trajectory is identical to what it would have
been without interruption (within floating-point tolerance). Learning rate schedule
position, optimizer momentum, and RNG state are all restored.

### Fresh Start From Boundary

**Purpose:** Begin a new phase using pretrained weights from the previous phase.

**What it loads:**
```python
boundary = torch.load("phase_boundary.pt")
# Validate boundary artifact (Section 4)
validate_boundary(boundary, current_config, target_phase)

# Load only model weights -- strict=False allows new module keys
model.load_state_dict(boundary["model_state_dict"], strict=False)

# Initialize fresh optimizer (no momentum carryover)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

# Initialize fresh scheduler (starts from step 0)
scheduler = CosineAnnealingLR(optimizer, T_max=config.training.total_steps)

# Seed RNG from phase-specific seed
seed = config.seed.base_seed + config.seed.per_phase_offsets[target_phase - 1]
set_all_seeds(seed)
```

**Invariant:** Optimizer state is clean. Learning rate starts from the configured peak
(or warmup start). RNG state is deterministic from the phase seed, not carried over from
the previous phase.

### Decision Logic

```python
def decide_load_strategy(args, run_dir: Path, target_phase: int) -> str:
    phase_ckpt_dir = run_dir / "checkpoints" / f"phase{target_phase}"
    boundary_dir = run_dir / "checkpoints" / f"phase{target_phase - 1}"

    if args.resume and (phase_ckpt_dir / "ckpt_latest.pt").exists():
        # Resume within phase: user explicitly requested resume and checkpoint exists
        return "resume"

    elif (boundary_dir / "phase_boundary.pt").exists():
        # Fresh start from upstream boundary
        return "fresh_from_boundary"

    elif target_phase == 1:
        # Phase 1 has no upstream: train from scratch
        return "from_scratch"

    else:
        raise HardFailure(
            f"Cannot start phase {target_phase}: no resume checkpoint and "
            f"no boundary artifact from phase {target_phase - 1}"
        )
```

### Common Mistake: Accidental Resume as Fresh Start

If you load a `phase_boundary.pt` but also restore the optimizer from a full checkpoint
of the previous phase, you get the wrong learning rate schedule. The optimizer's internal
step counter reflects the previous phase's training, causing the scheduler to compute
incorrect learning rates. Always use a fresh optimizer for cross-phase transitions.

---

## 10. Boundary Validator Implementation Sketch

The following pseudocode ties all checks together into a single entry point.

```python
class PhaseBoundaryValidator:
    """Validates phase boundary artifacts before starting a new training phase."""

    CURRENT_SCHEMA_MAJOR = 1
    CURRENT_SCHEMA_MINOR = 0

    def __init__(self, mode: str = "normal"):
        assert mode in ("strict", "normal", "permissive")
        self.mode = mode
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def validate(
        self,
        run_dir: Path,
        target_phase: int,
        current_config: BrainAIConfig,
        current_datasets: dict | None = None,
    ) -> bool:
        source_phase = target_phase - 1

        # 1. File existence
        boundary_path = run_dir / "checkpoints" / f"phase{source_phase}" / "phase_boundary.pt"
        self._require(boundary_path.exists(),
                      f"Boundary artifact not found: {boundary_path}", hard=True)

        # 2. Load artifact
        artifact = torch.load(boundary_path, map_location="cpu", weights_only=False)

        # 3. Schema version
        self._check_schema_version(artifact)

        # 4. Phase number sanity
        self._require(artifact.get("source_phase") == source_phase,
                      f"source_phase in artifact ({artifact.get('source_phase')}) "
                      f"!= expected ({source_phase})", hard=True)
        self._require(artifact.get("target_phase") == target_phase,
                      f"target_phase in artifact ({artifact.get('target_phase')}) "
                      f"!= expected ({target_phase})", hard=True)

        # 5. Feature flags
        self._check_feature_flags(artifact, current_config)

        # 6. Dimensions
        self._check_dimensions(artifact, current_config, target_phase)

        # 7. State dict keys
        expected_keys = self._expected_keys_for_phase(target_phase, current_config)
        self._check_state_dict_keys(artifact, expected_keys, target_phase)

        # 8. Config compatibility
        self._check_config_compatibility(artifact, current_config, target_phase)

        # 9. Dataset fingerprint
        if current_datasets:
            self._check_dataset_fingerprint(artifact, current_datasets, target_phase)

        # Report
        if self.errors:
            for e in self.errors:
                logger.error(f"[BOUNDARY ERROR] {e}")
            return False

        if self.warnings:
            for w in self.warnings:
                logger.warning(f"[BOUNDARY WARNING] {w}")

        logger.info(f"Phase boundary validation passed for phase {source_phase} -> {target_phase}")
        return True

    def _require(self, condition: bool, message: str, hard: bool = True) -> None:
        if not condition:
            if hard and self.mode != "permissive":
                self.errors.append(message)
            else:
                self.warnings.append(message)
```

---

## 11. Boundary Artifact Creation

Each phase's training script must produce the boundary artifact at the end of successful
training. Use the following helper:

```python
def save_phase_boundary(
    model: nn.Module,
    config: BrainAIConfig,
    run_dir: Path,
    source_phase: int,
    run_id: str,
    best_metric: dict,
    training_steps: int,
    dataset_fingerprint: dict | None = None,
) -> Path:
    """Save a phase boundary artifact for consumption by the next phase."""

    target_phase = source_phase + 1
    boundary_dir = run_dir / "checkpoints" / f"phase{source_phase}"
    boundary_dir.mkdir(parents=True, exist_ok=True)
    boundary_path = boundary_dir / "phase_boundary.pt"

    # Extract only the state dict keys relevant to the next phase
    state_dict = extract_boundary_state_dict(model, source_phase)

    artifact = {
        "schema_version": "1.0",
        "source_phase": source_phase,
        "target_phase": target_phase,
        "source_run_id": run_id,
        "model_state_dict": state_dict,
        "config_snapshot": config_to_dict(config),
        "feature_flags": {
            "use_snn": config.use_snn,
            "use_htm": config.use_htm,
            "use_workspace": config.use_workspace,
            "use_symbolic": config.use_symbolic,
            "use_meta": config.use_meta,
            "use_engram": config.use_engram,
        },
        "compatibility": build_compatibility_dict(config, source_phase),
        "metadata": {
            "best_metric": best_metric,
            "training_steps": training_steps,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dataset_fingerprint": dataset_fingerprint,
        },
    }

    torch.save(artifact, boundary_path)
    logger.info(f"Saved phase boundary artifact: {boundary_path}")
    return boundary_path


def build_compatibility_dict(config: BrainAIConfig, source_phase: int) -> dict:
    """Build the compatibility section based on what has been trained so far."""
    compat = {
        "workspace_dim": config.workspace.workspace_dim,
        "vocab_size": config.encoder.text_vocab_size,
    }

    if source_phase >= 1:
        compat["snn_hidden_sizes"] = config.snn.hidden_sizes
        compat["snn_output_dim"] = config.snn.hidden_sizes[-1]
        compat["surrogate"] = config.snn.surrogate

    if source_phase >= 2:
        compat["num_encoders"] = len(config.modalities) + (1 if config.use_engram else 0)
        compat["enabled_encoders"] = list(config.modalities) + (
            ["engram"] if config.use_engram else []
        )

    if source_phase >= 3:
        compat["htm_column_count"] = config.htm.column_count
        compat["htm_cells_per_column"] = config.htm.cells_per_column
        compat["htm_input_dim"] = config.workspace.workspace_dim

    if source_phase >= 4:
        compat["ignition_threshold"] = config.workspace.ignition_threshold
        compat["working_memory_capacity"] = config.workspace.capacity_limit
        compat["num_modalities"] = len(config.modalities)

    if source_phase >= 5:
        compat["generative_model_state_dim"] = 64  # from ActiveInferenceConfig
        compat["num_policies"] = config.decision.num_policies
        compat["efe_components"] = ["pragmatic", "epistemic"]
        if config.decision.use_improved_efe:
            compat["efe_components"].append("instrumental")

    if source_phase >= 6:
        compat["reasoning_hidden_dim"] = config.reasoning.hidden_dim
        compat["confidence_calibration_present"] = True
        compat["symbolic_kb_size"] = config.reasoning.num_entities
        compat["system2_present"] = config.use_symbolic

    return compat
```

---

## 12. Quick Reference: Validation Checklist Per Phase

Use this table as a go/no-go checklist before launching each phase.

| Target Phase | Prerequisites |
|---|---|
| **Phase 1** | No upstream boundary. Verify dataset is available. Verify `SNNConfig` is consistent with scale preset. |
| **Phase 2** | `phase1/phase_boundary.pt` exists. `snn_hidden_sizes` match. `surrogate` matches. Dataset fingerprint matches (vision). |
| **Phase 3** | `phase2/phase_boundary.pt` exists. `workspace_dim` matches `encoder.output_dim`. All enabled encoders present in state dict. `vocab_size` matches (if text enabled). Dataset fingerprint matches (all modalities). |
| **Phase 4** | `phase3/phase_boundary.pt` exists. `htm_input_dim == workspace_dim`. `column_count` matches. `cells_per_column` matches. Dataset fingerprint matches. |
| **Phase 5** | `phase4/phase_boundary.pt` exists. `workspace_dim` matches. `working_memory_capacity` matches. `num_modalities` matches. Partial dataset match (base modalities). |
| **Phase 6** | `phase5/phase_boundary.pt` exists. `generative_model_state_dim` matches. EFE components present. Partial dataset match. |
| **Phase 7** | `phase6/phase_boundary.pt` exists. Reasoning weights present (`system2_present`). Calibration params present. `reasoning_hidden_dim` matches. No dataset match required (meta-learning uses different data). |
