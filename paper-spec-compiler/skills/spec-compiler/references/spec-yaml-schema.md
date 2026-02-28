# spec.yaml Schema Reference

## Overview

The spec.yaml file is the canonical machine-readable truth extracted from a paper.
Every field either contains an extracted value or is marked `UNRESOLVED`. The schema
maps directly to the IR entities defined in `ir_schema.py`.

## Top-Level Sections

```yaml
meta:           # Paper metadata and provenance
frames:         # Coordinate conventions
spaces:         # State, observation, action, privileged-info spaces
timing:         # Control frequencies, sensor/action delays
reward:         # Reward expression tree with term weights
termination:    # Termination conditions as boolean ASTs
gates_track:    # Gate geometry and pass conditions (domain-specific)
dynamics:       # ODEs, integrators, parameters, domain randomization
perception:     # Input specs, intrinsics, segmentation, augmentations
world_model:    # RSSM/latent model architecture
actor_critic:   # Policy distribution, imagination horizon, discount
training:       # Replay buffer, update ratios, schedules, hyperparams
evaluation:     # Success criteria, metrics, ablations
deployment:     # Hardware targets, inference stack, runtime budgets
imports:        # Baseline spec fragments to inherit from
```

## Section Details

### meta

```yaml
meta:
  paper:
    title: str                    # Full paper title
    arxiv: str                    # arXiv ID (e.g., "2510.14783")
    version: str                  # arXiv version (e.g., "v1", "v2")
    doi: str | null               # DOI if published
    authors: list[str]            # Author list
  sources:
    prefer: list[str]             # Priority: ["arxiv_tex", "pdf", "html"]
    tex_hash: str | null          # SHA256 of LaTeX source tarball
    pdf_hash: str | null          # SHA256 of PDF
  generated_at: str               # ISO 8601 timestamp
  tool_version: str               # Spec compiler version
  unresolved_count: int           # Number of UNRESOLVED fields
```

### frames

```yaml
frames:
  convention: str                 # e.g., "NED", "ENU", "ROS"
  frames: list[str]               # Named frames: ["world", "body", "camera", ...]
  transforms:                     # Optional explicit transforms
    - from: str
      to: str
      type: str                   # "static" | "dynamic"
      description: str
```

### spaces

The informed-POMDP split is first-class. Three space categories:

```yaml
spaces:
  observation_exec:               # What the policy sees at deployment
    - name: str
      dtype: str                  # "float32", "bool", "int64", etc.
      shape: list[int | str]      # Dimensions; str for symbolic (e.g., "N_PARAMS")
      units: str | null           # Physical units if applicable
      source: str                 # Paper reference: "§3.2", "Table 1", "Eq. 4"

  information_train:              # Privileged fields available only during training
    - name: str                   # Convention: prefix with context (e.g., "p_w", "v_w")
      dtype: str
      shape: list[int | str]
      units: str | null
      source: str
      informed_dreamer_key: str | null  # Regex pattern for decoder gating

  action:
    name: str
    dtype: str
    shape: list[int]
    bounds: [float, float]        # [min, max] clamp values
    semantics: str                # e.g., "normalized motor commands"
    source: str

  state:                          # Full state vector (may be superset of obs + info)
    - name: str
      dtype: str
      shape: list[int | str]
      units: str | null
      category: str               # "observable" | "privileged" | "internal"
```

### timing

```yaml
timing:
  control_frequency_hz: float
  sensor_delay_ms: float
  action_delay_ms: float
  timestamping_model: str         # "camera_anchored" | "control_loop_anchored"
  sim_dt: float | null            # Simulation timestep if applicable
  policy_dt: float | null         # Policy decision timestep
  source: str
```

### reward

Reward terms are stored as expression trees, not strings:

```yaml
reward:
  discount: float
  terms:
    - name: str                   # e.g., "progress", "collision_penalty"
      expression_ast:             # Structured expression tree
        op: str                   # "multiply", "add", "clamp", "norm", "min", "max", etc.
        args: list                # Recursive: numbers, field references, or nested ops
      weight: float
      clamp: [float, float] | null
      zeroing_window: str | null  # Conditions when term is zeroed
      source: str
  normalization: str | null       # "symlog", "none", etc.
  source: str
```

**Expression AST node types:**

```yaml
# Literal value
{type: "literal", value: 3.14}

# Field reference
{type: "field", name: "velocity_norm", space: "state"}

# Binary operation
{type: "op", op: "multiply", args: [{...}, {...}]}

# Unary operation
{type: "op", op: "clamp", args: [{...}], params: {min: -1.0, max: 1.0}}

# Function call
{type: "func", name: "norm", args: [{type: "field", name: "v_w"}]}
```

### termination

Termination conditions as boolean ASTs:

```yaml
termination:
  conditions:
    - name: str                   # e.g., "ground_collision", "gate_collision"
      condition_ast:
        op: str                   # "and", "or", "not", "lt", "gt", "le", "ge", "eq"
        args: list                # Boolean sub-expressions or field comparisons
      source: str
  max_episode_steps: int | null
  source: str
```

### gates_track (Domain-Specific)

For drone racing or track-based tasks:

```yaml
gates_track:
  gate_geometry:
    shape: str                    # "square", "circular", etc.
    dimensions: dict              # width, height, thickness
    virtual_thickness: float
    source: str
  pre_post_offsets:
    pre_gate_offset: float
    post_gate_offset: float
    source: str
  pass_condition:
    condition_ast: dict           # Boolean AST for gate pass detection
    source: str
  num_gates: int | str
  source: str
```

### dynamics

```yaml
dynamics:
  model_type: str                 # "ODE", "learned", "hybrid"
  equations:
    - name: str
      latex: str                  # Original LaTeX expression
      expression_ast: dict        # Parsed AST
      source: str
  integrator:
    type: str                     # "RK4", "Euler", "adaptive"
    dt: float
    source: str
  parameters:
    - name: str
      symbol: str                 # LaTeX symbol (e.g., "m", "I_{xx}")
      default_value: float
      units: str
      source: str                 # "Table 3, row 2"
  domain_randomization:
    - parameter: str              # Reference to parameter name
      distribution: str           # "uniform", "normal", "log_uniform"
      range: [float, float]
      resample_frequency: str     # "per_episode", "per_step", "fixed"
      source: str
```

### perception

```yaml
perception:
  input_image:
    resolution: [int, int]
    channels: int
    dtype: str
    source: str
  intrinsics_normalization:
    target_K: list[list[float]] | str  # Camera intrinsic matrix or "UNRESOLVED"
    source: str
  segmentation_model:
    architecture: str
    input_size: [int, int]
    output_classes: int
    source: str
  augmentations:
    - name: str                   # e.g., "GAN_mask_translation", "erosion", "rolling_shutter"
      parameters: dict
      source: str
```

### world_model

```yaml
world_model:
  architecture: str               # "RSSM", "transformer", etc.
  components:
    encoder:
      type: str
      layers: list | str
      source: str
    sequence_model:
      type: str                   # "GRU", "LSTM", "S4"
      hidden_size: int
      source: str
    dynamics_predictor:
      type: str
      source: str
    decoder:
      targets: list[str]          # What it reconstructs
      source: str
    reward_head:
      type: str
      source: str
    continue_head:
      type: str
      source: str
  discrete_latent:
    num_categoricals: int
    num_classes: int
    source: str
  symlog: bool
  normalization: str              # "layer_norm", "none", etc.
  source: str
```

### actor_critic

```yaml
actor_critic:
  policy_distribution: str        # "squashed_normal", "categorical", etc.
  deterministic_eval: bool
  imagination_horizon: int
  discount: float
  lambda_gae: float
  actor:
    hidden_layers: list[int]
    activation: str
    source: str
  critic:
    hidden_layers: list[int]
    activation: str
    source: str
  regularizers:
    - name: str                   # e.g., "action_smoothness"
      coefficient: float
      source: str
  source: str
```

### training

```yaml
training:
  algorithm: str                  # e.g., "DreamerV3 + Informed decoding"
  replay:
    capacity_steps: int
    context_length: int
    sampling: str                 # "uniform", "prioritized"
    source: str
  batch:
    size: int
    length: int
    source: str
  schedule:
    total_env_steps: int
    phases:
      - name: str
        start_step: int
        end_step: int
        learning_rate: float
        entropy_scale: float
        notes: str
        source: str
  train_ratio: float
  optimizer:
    type: str
    lr: float
    eps: float
    clip_grad: float | null
    source: str
  use_amp: bool
  source: str
```

### evaluation

```yaml
evaluation:
  success_criteria:
    - metric: str
      threshold: float
      direction: str              # "higher_is_better" | "lower_is_better"
      source: str
  num_seeds: int
  num_eval_episodes: int
  report_error_bars: bool
  ablations:
    - name: str
      description: str
      source: str
  source: str
```

### deployment

```yaml
deployment:
  target_hardware: str
  inference_stack: list[str]      # e.g., ["JAX", "Torch", "ONNX", "TensorRT"]
  runtime_budgets:
    - module: str
      max_ms: float
      source: str
  safety_constraints:
    - name: str
      description: str
      source: str
  control_frequency_hz: float
  source: str
```

### imports (Baseline Linking)

```yaml
imports:
  - name: str                     # e.g., "dreamerv3"
    version: str                  # Commit hash or version tag
    spec_url: str | null          # URL to baseline spec.yaml
    overrides: list[str]          # Paths in this spec that override the baseline
```

## Validation Rules

1. Every leaf field is either a concrete value or the string `"UNRESOLVED"`
2. Every extracted value has a `source` field tracing to the paper
3. Shape dimensions are either integers or symbolic strings (resolved elsewhere)
4. AST nodes follow the expression grammar defined above
5. `unresolved_count` in meta must match actual UNRESOLVED fields
6. `imports` must specify version/commit for reproducibility
