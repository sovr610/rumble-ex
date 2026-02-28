# Baseline Linking: Composable Spec Fragments

## Concept

Most ML papers build on prior work. SkyDreamer extends DreamerV3 + Informed Dreamer.
Rather than extracting the entire DreamerV3 spec from scratch for every downstream
paper, treat baseline papers as importable spec fragments.

```
SkyDreamer spec = DreamerV3 fragment + Informed Dreamer fragment + SkyDreamer deltas
```

This prevents a critical class of drift: accidentally deviating from DreamerV3 defaults
that the SkyDreamer authors assumed were inherited.

## Import Mechanism

### spec.yaml imports section

```yaml
imports:
  - name: dreamerv3
    version: "a1b2c3d"           # Git commit of reference implementation
    spec_url: "specs/dreamerv3/spec.yaml"
    overrides:                    # Paths this spec explicitly overrides
      - "world_model.discrete_latent"
      - "training.replay.context_length"
      - "actor_critic.imagination_horizon"

  - name: informed_dreamer
    version: "e4f5g6h"
    spec_url: "specs/informed_dreamer/spec.yaml"
    overrides:
      - "spaces.information_train"
      - "world_model.components.decoder.targets"
```

### Resolution Rules

1. **Explicit override wins**: If SkyDreamer spec sets a value, it takes precedence
2. **Baseline fills gaps**: For fields SkyDreamer doesn't mention, use the baseline
3. **Conflict detection**: If both baselines set a value differently and SkyDreamer
   doesn't override, flag as a WARNING
4. **Inheritance chain**: DreamerV3 → Informed Dreamer → SkyDreamer (later overrides earlier)

### Output Files

The resolution process produces three files:

#### spec.resolved.yaml

Fully flattened spec with all imports resolved. Every field has a concrete value
(or UNRESOLVED). Includes provenance annotations showing which spec provided each value:

```yaml
training:
  replay:
    capacity_steps: 10000000     # source: dreamerv3@a1b2c3d
    context_length: 16           # source: skydreamer (override)
    sampling: "uniform"          # source: dreamerv3@a1b2c3d
  train_ratio: 32                # source: dreamerv3@a1b2c3d
```

#### spec.patch.yaml

Only the deltas — what SkyDreamer changes from its baselines:

```yaml
# SkyDreamer changes from DreamerV3 + Informed Dreamer baseline
spaces:
  observation_exec:
    # Added: gate_mask, flight_plan fields specific to drone racing
    - name: "gate_mask"
      dtype: "bool"
      shape: [64, 64]
  information_train:
    # Added: privileged drone state
    - name: "p_w"
      shape: [3]

training:
  replay:
    context_length: 16           # Changed from DreamerV3 default of 64
  schedule:
    total_env_steps: 17000000    # Changed from DreamerV3 default
```

#### spec.conflicts.yaml (if any)

```yaml
conflicts:
  - path: "training.optimizer.lr"
    dreamerv3_value: 1e-4
    informed_dreamer_value: 3e-4
    skydreamer_value: null        # Not specified — REQUIRES override
    resolution: "UNRESOLVED"
```

## Creating Baseline Specs

### DreamerV3 Fragment

Extract from the DreamerV3 paper + reference implementation (danijar/dreamerv3):

Key sections to extract:
- World model RSSM architecture (categorical sizes, hidden dims)
- Actor-critic (imagination horizon, discount, lambda)
- Training (replay, batch, optimizer, train_ratio)
- Normalization (symlog, layer norm)

### Informed Dreamer Fragment

Extract from the Informed Dreamer paper:

Key sections to extract:
- Information state formulation (POMDP → informed-POMDP)
- Decoder gating mechanism (which keys are privileged)
- Training-time decoding protocol
- Any hyperparameter changes from DreamerV3

### Fragment Versioning

Each baseline fragment is versioned by:
1. Paper arXiv version (e.g., "v2")
2. Reference implementation commit hash
3. Spec extraction timestamp

This ensures reproducibility: the exact same extraction from the exact same sources.

## Workflow

### First-Time Setup

```bash
# Extract DreamerV3 baseline spec
python scripts/arxiv_fetch.py --arxiv-id 2301.04104 --output-dir .paper_sources/dreamerv3/
python scripts/tex_parser.py --source-dir .paper_sources/dreamerv3/ --output specs/dreamerv3/ir.json
python scripts/emit_yaml.py --ir specs/dreamerv3/ir.json --output specs/dreamerv3/spec.yaml

# Extract Informed Dreamer baseline spec
python scripts/arxiv_fetch.py --arxiv-id 2306.12815 --output-dir .paper_sources/informed_dreamer/
python scripts/tex_parser.py --source-dir .paper_sources/informed_dreamer/ --output specs/informed_dreamer/ir.json
python scripts/emit_yaml.py --ir specs/informed_dreamer/ir.json --output specs/informed_dreamer/spec.yaml

# Extract SkyDreamer with imports
python scripts/emit_yaml.py --ir ir_output.json --output spec.yaml \
  --import specs/dreamerv3/spec.yaml \
  --import specs/informed_dreamer/spec.yaml
```

### Resolving Imports

```bash
python scripts/compile_configs.py --spec spec.yaml \
  --resolve-imports \
  --output-resolved spec.resolved.yaml \
  --output-patch spec.patch.yaml \
  --output-conflicts spec.conflicts.yaml
```

### Verifying No Unintended Drift

After resolution, the test generator uses `spec.resolved.yaml`:

```bash
python scripts/gen_tests.py --spec spec.resolved.yaml --output-dir tests/spec_compliance/
```

This catches cases where an engineer changes a DreamerV3 default without realizing
the downstream paper depended on it.
