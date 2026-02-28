# Reproducibility Checklist Auto-Generation

## Overview

The spec compiler auto-generates a reproducibility appendix section in spec.md
that mirrors common checklist expectations. This ensures the spec satisfies
reproducibility standards without manual effort.

## Checklist Items

### 1. Full Hyperparameter List

Extract from `training`, `actor_critic`, `world_model`, and `dynamics` sections:

```markdown
## Reproducibility Appendix

### Hyperparameters

| Parameter | Value | Selection Method | Source |
|-----------|-------|-----------------|--------|
| Learning rate | 1e-4 | Grid search | Table A.1 |
| Discount | 0.997 | From DreamerV3 | §3.4 |
| Imagination horizon | 15 | Ablation study | §4.3 |
| Replay capacity | 10M steps | Standard | §3.4 |
| Batch length | 64 | Standard | §3.4 |
| Train ratio | 32 | From DreamerV3 | Table A.1 |
```

Selection method is inferred:
- "From [baseline]" if inherited via imports
- "Grid search" if ablation results are present
- "Standard" if only one value is mentioned
- "UNRESOLVED" if selection method is unclear

### 2. Number of Runs and Seeds

Extract from `evaluation` section:

```markdown
### Experimental Setup

- **Number of seeds**: 3 [Source: §4.1]
- **Evaluation episodes per seed**: 100 [Source: §4.1]
- **Error bars**: Standard deviation across seeds [Source: §4.2]
```

### 3. Compute and Infrastructure

Extract from `deployment` and `training` sections:

```markdown
### Compute Resources

- **Training hardware**: NVIDIA A100 80GB [Source: §4.1]
- **Training time**: ~48 hours [Source: §4.1]
- **Inference hardware**: NVIDIA Jetson Orin [Source: §5.1]
- **Total GPU hours**: ~144 GPU-hours (3 seeds × 48h)
```

### 4. Dataset/Environment Versioning

```markdown
### Environment

- **Simulator**: [Name] version [X.Y.Z]
- **Track configurations**: [N] tracks with [M] gates each
- **Domain randomization**: [K] parameters randomized (see dynamics.domain_randomization)
```

### 5. Evaluation Metrics

Extract from `evaluation.success_criteria`:

```markdown
### Metrics

| Metric | Threshold | Direction | Source |
|--------|-----------|-----------|--------|
| Lap completion rate | > 95% | higher_is_better | §4.2 |
| Average lap time | < 12.0s | lower_is_better | §4.2 |
| Collision rate | < 5% | lower_is_better | §4.2 |
```

## Generation Rules

1. Every row in every table must have a `Source` column tracing to the paper
2. Missing values are marked UNRESOLVED with the expected location
3. Derived values (e.g., total GPU hours) show their calculation
4. Baseline-inherited values are annotated with the baseline name

## Alignment with Standards

### NeurIPS Reproducibility Checklist

The generated appendix maps to NeurIPS checklist items:
- C1: "Full details of all hyperparameters" → Hyperparameter table
- C2: "Number of runs and error bars" → Experimental setup
- C3: "Computing infrastructure" → Compute resources
- C4: "Evaluation metrics" → Metrics table

### RL-Specific Items

Additional items specific to RL papers:
- Environment version and configuration
- Reward function complete specification
- Domain randomization ranges
- Training schedule (staged, curriculum, etc.)
- Evaluation protocol (deterministic vs stochastic policy)
