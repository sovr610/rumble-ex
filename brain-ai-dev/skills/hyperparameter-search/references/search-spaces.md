# Search Spaces for brain_ai Hyperparameter Optimization

## Overview

The brain_ai system has 10 configuration dataclasses aggregated into `BrainAIConfig`, containing hundreds of tunable parameters. Searching all parameters simultaneously is infeasible and wasteful -- most parameters have negligible impact on the final metric. This document defines per-phase search spaces, conditional parameter relationships, constraints, and typical ranges, enabling focused and efficient hyperparameter optimization.

The guiding principle is: **tune 3-7 parameters at a time, phase by phase, starting with the most impactful ones (learning rate, architectural scale, regularization).**

---

## 1. General Design Principles

### Parameter Categories

Every hyperparameter falls into one of four categories:

1. **Critical:** High impact on performance; must be tuned. Examples: learning rate, hidden dimensions.
2. **Important:** Moderate impact; should be tuned after critical params are set. Examples: dropout, warmup steps.
3. **Secondary:** Low impact; use defaults unless fine-tuning. Examples: optimizer betas, gradient clip value.
4. **Fixed:** Should not be tuned; determined by architecture or hardware constraints. Examples: vocabulary size, image size.

### Log-Scale vs Linear-Scale

Parameters that span multiple orders of magnitude should use log-uniform sampling:
- Learning rates: `log_uniform(1e-5, 1e-1)`
- Weight decay: `log_uniform(1e-4, 1e-1)`
- Temperature parameters: `log_uniform(0.01, 10.0)`

Parameters with a narrow, roughly linear range should use uniform sampling:
- Dropout: `uniform(0.0, 0.5)`
- Number of layers: `int_uniform(2, 12)`
- Sparsity: `uniform(0.01, 0.1)`

### Conditional Parameters

brain_ai uses feature flags (`use_snn`, `use_htm`, `use_workspace`, etc.) that gate entire subsystems. Parameters for a subsystem should only be tuned when the corresponding flag is True:

```python
space.add_categorical("use_htm", [True, False])
space.add_conditional("htm.column_count", condition="use_htm == True",
                     sub_space=IntSpace(512, 16384, log=True))
```

---

## 2. Phase 1: SNN Core

### Context

Phase 1 trains the Spiking Neural Network core, the foundation of the brain_ai system. The SNN uses LIF (Leaky Integrate-and-Fire) neurons with surrogate gradients. Key considerations: membrane dynamics (beta, tau), surrogate gradient function, spike regularization, and the core learning rate.

### Search Space Definition

| Parameter | Type | Range | Scale | Priority |
|-----------|------|-------|-------|----------|
| `training.learning_rate` | float | [1e-4, 1e-2] | log | Critical |
| `snn.beta` | float | [0.8, 0.99] | linear | Critical |
| `snn.surrogate` | categorical | [atan, fast_sigmoid, straight_through] | -- | Important |
| `snn.surrogate_alpha` | float | [0.5, 5.0] | linear | Important |
| `snn.dropout` | float | [0.0, 0.3] | linear | Important |
| `snn.num_timesteps` | int | [10, 100] | linear | Secondary |
| `snn.spike_rate_target` | float | [0.05, 0.3] | linear | Secondary |
| `snn.spike_rate_weight` | float | [1e-3, 1e-1] | log | Secondary |
| `training.weight_decay` | float | [1e-4, 1e-1] | log | Important |
| `training.warmup_steps` | int | [500, 5000] | linear | Secondary |

### Constraints

- `snn.beta` must be in `(0, 1)` (membrane decay factor).
- `snn.surrogate_alpha` must be positive (gradient sharpness).
- If `snn.surrogate == "straight_through"`, `surrogate_alpha` is ignored (set to 1.0).
- `snn.spike_rate_weight` should be small relative to the main loss (typically 0.001-0.1).

### Conditional Parameters

```python
if snn.use_learnable_delays:
    tune snn.max_delay in [4, 32] (int, linear)
if snn.use_heterogeneous_tau:
    # tau is learned per-neuron; no explicit param to tune
    pass
if snn.use_adaptive_threshold:
    tune threshold_decay in [0.9, 0.999] (float, linear)
```

### Typical Outcomes

With 100 random+ASHA trials on minimal config:
- Learning rate: typically converges to 1e-3 to 5e-3 range.
- Beta: 0.90-0.95 usually optimal (moderate memory).
- Surrogate: `atan` and `fast_sigmoid` perform similarly; `straight_through` slightly worse for deep networks.
- Dropout: 0.05-0.15 optimal for production-scale SNN.

---

## 3. Phase 2: Modality Encoders

### Context

Phase 2 trains vision, text, and audio encoders while freezing the SNN core. These are transformer-based architectures with standard hyperparameters. The search space depends on which modalities are enabled.

### Search Space Definition

**Common (all modalities):**

| Parameter | Type | Range | Scale | Priority |
|-----------|------|-------|-------|----------|
| `training.learning_rate` | float | [1e-5, 1e-3] | log | Critical |
| `training.weight_decay` | float | [0.01, 0.3] | log | Important |
| `training.warmup_steps` | int | [1000, 10000] | linear | Important |
| `training.grad_clip` | float | [0.5, 2.0] | linear | Secondary |

**Vision encoder (conditional on "vision" in modalities):**

| Parameter | Type | Range | Scale | Priority |
|-----------|------|-------|-------|----------|
| `encoder.vision_num_layers` | int | [12, 32] | linear | Important |
| `encoder.vision_num_heads` | int | [8, 16, 32] | categorical | Secondary |
| `encoder.vision_patch_size` | categorical | [8, 16, 32] | -- | Important |

**Text encoder (conditional on "text" in modalities):**

| Parameter | Type | Range | Scale | Priority |
|-----------|------|-------|-------|----------|
| `encoder.text_num_layers` | int | [12, 48] | linear | Important |
| `encoder.text_num_heads` | int | [8, 16, 32] | categorical | Secondary |
| `encoder.text_max_seq_len` | categorical | [2048, 4096, 8192] | -- | Secondary |

**Audio encoder (conditional on "audio" in modalities):**

| Parameter | Type | Range | Scale | Priority |
|-----------|------|-------|-------|----------|
| `encoder.audio_num_layers` | int | [12, 32] | linear | Important |
| `encoder.audio_n_mels` | categorical | [64, 80, 128] | -- | Secondary |

### Constraints

- `encoder.output_dim` should remain fixed (must match workspace_dim).
- `vision_num_heads` must divide `vision_hidden_dim`.
- `text_num_heads` must divide `text_embed_dim`.
- Reducing `vision_patch_size` quadruples sequence length; may exceed GPU memory.

---

## 4. Phase 3: HTM (Hierarchical Temporal Memory)

### Context

Phase 3 trains the HTM layer for temporal sequence learning. The HTM has biological parameters (column count, cells per column, permanence values) that interact strongly. If `use_htm=False`, this phase is skipped entirely.

### Search Space Definition

| Parameter | Type | Range | Scale | Priority |
|-----------|------|-------|-------|----------|
| `training.learning_rate` | float | [1e-4, 5e-3] | log | Critical |
| `htm.column_count` | int | [512, 16384] | log | Critical |
| `htm.cells_per_column` | int | [8, 64] | linear | Important |
| `htm.sparsity` | float | [0.01, 0.1] | linear | Important |
| `htm.permanence_inc` | float | [0.01, 0.3] | linear | Important |
| `htm.permanence_dec` | float | [0.01, 0.3] | linear | Important |
| `htm.activation_threshold` | int | [5, 25] | linear | Secondary |
| `htm.lstm_num_layers` | int | [2, 8] | linear | Secondary |

### Conditional Parameters

```python
if htm.use_reflex_memory:
    tune htm.reflex_num_tables in [4, 16] (int)
    tune htm.reflex_bits_per_hash in [8, 16] (int)
    tune htm.reflex_promotion_threshold in [3, 10] (int)
    tune htm.reflex_capacity in [10000, 1000000] (int, log)
```

### Constraints

- `permanence_inc` and `permanence_dec` should be roughly balanced; if inc >> dec, all synapses become permanent. If dec >> inc, all synapses are pruned.
- `column_count * cells_per_column` determines memory capacity; larger values need more GPU memory.
- `sparsity * column_count` should give at least 10 active columns for meaningful representations.
- `activation_threshold` must be <= `sparsity * column_count * cells_per_column`.

### Interaction Effects

Strong interactions exist between:
- `column_count` and `sparsity`: more columns allow lower sparsity (denser representations).
- `permanence_inc` and `permanence_dec`: their ratio determines learning speed and stability.
- `cells_per_column` and sequence complexity: more cells enable longer sequence memory.

Consider tuning `column_count` and `sparsity` together in a 2D grid after finding good ranges for other parameters.

---

## 5. Phase 4: Global Workspace

### Context

Phase 4 trains the global workspace theory implementation -- a competition mechanism where multiple cognitive modules (SNN, HTM, encoders) compete for broadcast access. Key parameters control the competition dynamics, broadcast mechanism, and working memory.

### Search Space Definition

| Parameter | Type | Range | Scale | Priority |
|-----------|------|-------|-------|----------|
| `training.learning_rate` | float | [1e-5, 1e-3] | log | Critical |
| `workspace.num_heads` | int | [8, 32] | linear | Important |
| `workspace.capacity_limit` | int | [3, 12] | linear | Secondary |
| `workspace.memory_num_layers` | int | [4, 12] | linear | Important |

### Conditional Parameters

```python
if workspace.use_selection_broadcast:
    tune workspace.selection_rounds in [1, 5] (int)
    tune workspace.ignition_threshold in [0.1, 0.7] (float)
    tune workspace.broadcast_iterations in [1, 4] (int)
    tune workspace.broadcast_decay in [0.7, 0.99] (float)
```

### Constraints

- `workspace.workspace_dim` must match `encoder.output_dim` (4096 by default); do not tune independently.
- `workspace.num_heads` must divide `workspace.workspace_dim`.
- `workspace.ignition_threshold` too low means everything broadcasts (noisy); too high means nothing ignites (silent).
- `workspace.capacity_limit` is inspired by Miller's Law (7 +/- 2); extreme values (1 or 20) are biologically implausible but may work computationally.

---

## 6. Phase 5: Active Inference / Decision

### Context

Phase 5 trains the active inference decision system, which uses Expected Free Energy (EFE) to select actions. The key trade-off is between epistemic (exploration) and pragmatic (exploitation) value.

### Search Space Definition

| Parameter | Type | Range | Scale | Priority |
|-----------|------|-------|-------|----------|
| `training.learning_rate` | float | [1e-5, 1e-3] | log | Critical |
| `decision.planning_horizon` | int | [1, 16] | linear | Critical |
| `decision.epistemic_weight` | float | [0.1, 5.0] | log | Important |
| `decision.num_policies` | int | [16, 256] | log | Important |

### Conditional Parameters

```python
if decision.use_improved_efe:
    tune decision.efe_num_samples in [8, 64] (int)
if decision.use_empowerment:
    tune decision.empowerment_weight in [0.01, 1.0] (float, log)
```

### Constraints

- `planning_horizon` * `num_policies` determines the planning cost; very large values may be too slow.
- `epistemic_weight` controls exploration: too high leads to perpetual exploration; too low leads to greedy exploitation.
- `efe_num_samples` > 32 provides diminishing returns for most tasks.

---

## 7. Phase 6: Reasoning

### Context

Phase 6 trains the dual-process reasoning system (System 1 fast/parallel, System 2 slow/deliberate) and optionally Logic Tensor Networks.

### Search Space Definition

| Parameter | Type | Range | Scale | Priority |
|-----------|------|-------|-------|----------|
| `training.learning_rate` | float | [1e-5, 1e-3] | log | Critical |
| `reasoning.confidence_threshold` | float | [0.5, 0.95] | linear | Critical |
| `reasoning.num_reasoning_steps` | int | [4, 32] | linear | Important |
| `reasoning.system2_layers` | int | [4, 12] | linear | Important |
| `reasoning.logic_type` | categorical | [product, godel, lukasiewicz] | -- | Secondary |

### Conditional Parameters

```python
if use_symbolic and reasoning.use_ltn:
    tune reasoning.ltn_embedding_dim in [64, 256] (int)
    tune reasoning.ltn_num_layers in [2, 6] (int)
    tune reasoning.ltn_p_forall in [1.0, 4.0] (float)
    tune reasoning.ltn_p_exists in [0.1, 1.0] (float)
```

### Constraints

- `confidence_threshold` determines System 1/System 2 routing: too low means System 2 is never used (fast but less accurate); too high means System 2 is always used (accurate but slow).
- `ltn_p_forall` must be >= 1.0 (p-norm for universal quantifier).
- `ltn_p_exists` must be in (0, 1) for existential quantifier semantics.

---

## 8. Phase 7: Meta-Learning

### Context

Phase 7 trains the meta-learning system (MAML/MAML++ and neuromodulation). This phase has the most sensitive hyperparameters because inner-loop learning rates directly affect gradient stability.

### Search Space Definition

| Parameter | Type | Range | Scale | Priority |
|-----------|------|-------|-------|----------|
| `meta.inner_lr` | float | [0.001, 0.5] | log | Critical |
| `meta.outer_lr` | float | [1e-5, 1e-3] | log | Critical |
| `meta.num_inner_steps` | int | [1, 20] | linear | Important |
| `meta.trace_decay` | float | [0.9, 0.999] | linear | Important |
| `meta.ewc_lambda` | float | [10, 10000] | log | Important |
| `meta.gradient_clipping` | float | [0.5, 5.0] | linear | Secondary |

### Conditional Parameters

```python
if meta.use_maml_plus_plus:
    # per-layer per-step LRs are learned; tune initial values
    tune meta_initial_inner_lr in [0.01, 0.1] (float, log)
if meta.use_task2vec:
    tune meta.task_embedding_dim in [64, 512] (int)
```

### Constraints

- `inner_lr` too high causes inner-loop divergence; too low makes adaptation useless.
- `num_inner_steps` * `inner_lr` should be bounded to prevent the inner loop from moving too far from the meta-parameters.
- `ewc_lambda` too high prevents any learning on new tasks; too low allows catastrophic forgetting.
- `first_order=True` is recommended for production (much cheaper); `first_order=False` is more sample-efficient but requires storing the full computation graph.

---

## 9. Cross-Phase Parameters

Some parameters affect multiple phases and should be tuned once for the entire pipeline:

| Parameter | Type | Range | Scale | Affects |
|-----------|------|-------|-------|---------|
| `encoder.output_dim` / `workspace.workspace_dim` | categorical | [512, 1024, 2048, 4096] | -- | All phases (representation size) |
| `training.batch_size` | categorical | [8, 16, 32, 64] | -- | All phases |
| `training.gradient_accumulation_steps` | categorical | [1, 4, 8, 16, 32] | -- | All phases |
| `training.amp_dtype` | categorical | [float16, bfloat16] | -- | All phases |
| Feature flags | boolean | -- | -- | Pipeline structure |

### Feature Flag Search

The six feature flags (`use_snn`, `use_htm`, `use_workspace`, `use_symbolic`, `use_meta`, `use_engram`) create 2^6 = 64 possible pipeline configurations. A practical approach:

1. Start with all flags True (default production config).
2. Ablate each flag individually (6 trials) to measure each module's contribution.
3. If a module's contribution is negative or negligible, disable it.
4. Tune only the remaining enabled modules' parameters.

---

## 10. Search Space Sizing Guide

### Effective Dimensionality

Not all tuned parameters matter equally. The effective dimensionality of the search is determined by parameters with significant impact:

| Phase | Parameters Tuned | Estimated Effective Dim | Recommended Trials |
|-------|:----------------:|:-----------------------:|:------------------:|
| 1 (SNN) | 6-10 | 3-4 | 50-100 |
| 2 (Encoders) | 5-8 | 2-3 | 50-100 |
| 3 (HTM) | 6-10 | 3-5 | 50-100 |
| 4 (Workspace) | 5-8 | 2-3 | 50-100 |
| 5 (Decision) | 4-6 | 2-3 | 50-80 |
| 6 (Reasoning) | 5-8 | 2-3 | 50-100 |
| 7 (Meta) | 5-8 | 3-4 | 50-100 |
| Cross-phase | 5-10 | 3-5 | 100-200 |

### Total Budget Estimation

For a complete HPO campaign across all 7 phases:
- Per phase: ~75 trials average (with ASHA early stopping, effective cost ~25 full-training-equivalent trials).
- Total: ~525 trial evaluations, equivalent to ~175 full training runs.
- At minimal config (1 epoch takes ~1 minute): ~3 hours total.
- At production config (1 epoch takes ~30 minutes): ~90 hours = ~4 days.

### Space Reduction Strategies

1. **LR finder first:** Remove learning rate from the search space by running the LR finder. This eliminates the highest-impact parameter and allows the search to focus on architecture/regularization.

2. **Importance-based pruning:** After 50 random trials, compute parameter importance (fANOVA or correlation). Drop parameters with importance < 5%.

3. **Phase decoupling:** Tune each phase independently. This is slightly suboptimal (phases interact) but reduces the search space from ~50 parameters to ~7 per search.

4. **Transfer across scales:** Tune on minimal config, then verify on production config. The optimal parameter ratios often transfer across scales (e.g., if dropout=0.1 is best at minimal scale, it is likely near-optimal at production scale too).

5. **Fix architectural parameters early:** Tune hidden dimensions, number of layers, etc., early in the HPO campaign with a coarse grid. Then fix them and tune continuous parameters (LR, weight decay, dropout) with Bayesian optimization.

---

## 11. Implementing SearchSpace from Config

The `SearchSpace.from_config_class()` method should use Python introspection to extract tunable parameters from a dataclass:

```python
from dataclasses import fields

def from_config_class(config_cls, overrides=None):
    space = SearchSpace()
    for f in fields(config_cls):
        if f.type == float:
            # Default range: 0.1x to 10x the default value
            default = f.default
            space.add_float(f.name, default * 0.1, default * 10.0)
        elif f.type == int:
            default = f.default
            space.add_int(f.name, max(1, default // 4), default * 4)
        elif f.type == bool:
            space.add_categorical(f.name, [True, False])
        elif f.type == str:
            # Cannot auto-infer choices; must be specified in overrides
            pass
    if overrides:
        for name, spec in overrides.items():
            space.override(name, spec)
    return space
```

This provides a starting point that can be refined with domain knowledge (the per-phase tables above).

---

## Summary

Effective hyperparameter search for brain_ai requires:

1. **Phase-by-phase approach:** Tune each training phase's parameters independently.
2. **Priority-driven selection:** Focus on critical and important parameters first; fix secondary parameters at defaults.
3. **Conditional handling:** Only tune module-specific parameters when the module is enabled.
4. **Constraint awareness:** Respect divisibility constraints, memory limits, and biological plausibility bounds.
5. **Scale-appropriate budgets:** 50-100 trials per phase with ASHA early stopping.
6. **LR finder integration:** Always start with the LR range test to eliminate the most impactful parameter from the search.
