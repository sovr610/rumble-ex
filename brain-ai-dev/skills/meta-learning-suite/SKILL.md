---
name: Meta-Learning Suite (MAML/FOMAML/Reptile + MAML++ Enhancements)
description: >-
  This skill should be used when the user asks to "implement meta-learning",
  "add MAML inner loop", "implement FOMAML", "add Reptile", "implement MAML++",
  "add per-layer per-step learning rates", "implement LSLR", "add multi-step loss",
  "implement episodic sampling", "add few-shot learning", "implement inner-loop optimizer",
  "add second-order meta-gradients", "implement torch.func inner loop",
  "add higher library support", "implement adaptation curves", "add AUAC metrics",
  "implement meta-checkpoint format", "add derivative-order annealing",
  "implement episode sampler", "add Omniglot dataset", "add mini-ImageNet splits",
  "implement Phase 7 runner", "add meta-training loop", "implement fast adaptation",
  "add batch norm handling for meta-learning", "implement differentiable inner loop",
  or mentions MAML, FOMAML, Reptile, meta-gradients, episodic few-shot,
  inner-loop optimization, MAML++ enhancements, or Phase 7 meta-training
  in the cognitive pipeline.
version: 0.1.0
---

# Meta-Learning Suite (MAML/FOMAML/Reptile + MAML++ Enhancements)

## Purpose

This skill standardizes the "learn-to-adapt" stack (Phase 7): a differentiable inner-loop
optimization engine that runs MAML (second-order), FOMAML (first-order), or Reptile
(weight-space interpolation) over episodic few-shot tasks, with optional MAML++ enhancements
(per-layer-per-step LRs, multi-step loss). The non-negotiable goals are correctness of
meta-gradient flow and reproducibility of episodic sampling.

## Key Files

| Target Module | Template Asset | Purpose |
|---|---|---|
| `brain_ai/meta/inner_loop.py` | `assets/inner_loop_template.py` | Differentiable inner-loop engine: torch.func, higher fallback, custom SGD |
| `brain_ai/meta/algorithms.py` | `assets/meta_algorithms_template.py` | MAML, FOMAML, Reptile: adapt(), meta_loss(), outer-loop logic |
| `brain_ai/meta/maml_plus.py` | `assets/maml_plus_template.py` | MAML++ enhancements: LSLR, MSL, derivative-order annealing, BN modes |
| `brain_ai/meta/episode_sampler.py` | `assets/episode_sampler_template.py` | EpisodeSampler, Omniglot, mini-ImageNet, deterministic splits |
| `brain_ai/meta/meta_metrics.py` | `assets/meta_metrics_template.py` | Adaptation curves, AUAC, gradient diagnostics, checkpoint format |
| `brain_ai/config.py` (extend) | `assets/meta_config_template.py` | MAMLConfig, EpisodeConfig, MAMLPlusPlusConfig, CheckpointConfig |

## Public Contract

```python
adapt(params, support_batch, *, steps, lrs, first_order, bn_mode, rng) -> adapted_params, inner_logs
meta_loss(params, task_batch, *, algo, config) -> loss, metrics, logs
```

`params` is a dict of named parameters (from `model.named_parameters()` or `torch.func`
parameter dict). `support_batch` is `(x_support, y_support)` with shapes `(N*K, ...)`.
`task_batch` wraps both support and query sets for multiple tasks.

## MetaOutput Contract

| Field | Shape / Type | Description |
|---|---|---|
| `loss` | scalar | Meta-objective (outer loss summed over tasks) |
| `metrics` | `Dict[str, float]` | `pre_adapt_acc`, `post_adapt_acc`, `fast_gain`, `auac` |
| `inner_logs` | `List[StepLog]` | Per-step: loss, accuracy, grad_norm, update_norm, lr_stats |
| `adapted_params` | `Dict[str, Tensor]` | Final adapted parameters per task (detached for eval) |

**Hard invariants**:
- In MAML mode, `adapted_params` tensors **retain** `grad_fn` (second-order graph intact).
- In FOMAML mode, gradients exist on base `params` but do **not** require `create_graph`.
- Inner-loop runs in **fp32** even under AMP autocast for meta-gradient stability.

## Inner-Loop Engine

Three implementations in priority order:

| Backend | Method | When Used |
|---|---|---|
| `torch.func` | `functional_call` + `torch.func.grad` | Default (PyTorch >=2.0) |
| `higher` | `higher.innerloop_ctx` | Feature flag `use_higher=True` |
| Custom SGD | `torch.autograd.grad` + manual update | Fallback (always available) |

The engine supports: second-order toggle (`create_graph`), per-step gradient clipping,
AMP safety (fp32 master params), and returns the full inner-loop trajectory for diagnostics.

See `references/inner-loop-engine.md` for implementation details, torch.func patterns, and AMP safety.

## Algorithm Variants

| Algorithm | Inner Loop | Outer Update | Order |
|---|---|---|---|
| MAML | Unrolled SGD (create_graph=True) | Gradient through adapted params | Second |
| FOMAML | Unrolled SGD (create_graph=False) | Gradient at adapted params (detached) | First |
| Reptile | Unrolled SGD (no graph) | θ ← θ + ε(φ − θ) weight interpolation | First |

See `references/algorithm-variants.md` for detailed implementations, MAML++ enhancements, and comparison.

## MAML++ Enhancements

| Enhancement | Config Flag | Description |
|---|---|---|
| LSLR | `use_lslr=True` | Per-layer, per-step learned learning rates `α_{layer,step}` |
| MSL | `use_msl=True` | Multi-step loss: weighted query loss at each inner step |
| Annealing | `use_annealing=True` | Start first-order, transition to second-order |
| BN modes | `bn_mode` | `"transductive"`, `"per_step"`, `"frozen"` |

See `references/algorithm-variants.md` for MAML++ math, LSLR parameter layout, and MSL weighting.

## Episodic Task Sampling

EpisodeSampler contract:
- **Deterministic** given `(global_seed, epoch, episode_idx)`
- Produces: N-way class set, K-shot support per class, Q-shot query per class
- Enforces **class-disjoint** train/val/test splits (split on classes, not images)

Datasets:
- **Omniglot**: 1,623 characters from 50 alphabets, 20 examples/character
- **mini-ImageNet**: 100 classes, 64/16/20 train/val/test class split (Ravi & Larochelle)

See `references/episodic-sampling.md` for sampler implementation, reproducibility rules, and augmentation.

## Configuration Surface

### MAMLConfig

| Field | Default | Purpose |
|---|---|---|
| `algo` | `"maml"` | `"maml"`, `"fomaml"`, `"reptile"` |
| `inner_steps` | 5 | Inner-loop adaptation steps |
| `inner_lr` | 0.01 | Base inner-loop learning rate |
| `inner_clip` | 10.0 | Inner-loop gradient clipping norm |
| `second_order` | True | Enable second-order gradients (auto-set by algo) |
| `backend` | `"auto"` | `"auto"`, `"torch_func"`, `"higher"`, `"custom"` |

### MAMLPlusPlusConfig

| Field | Default | Purpose |
|---|---|---|
| `use_lslr` | False | Per-layer, per-step learned LRs |
| `use_msl` | False | Multi-step loss accumulation |
| `msl_weights` | `"uniform"` | `"uniform"`, `"linear_increase"`, `"learned"` |
| `use_annealing` | False | Derivative-order annealing |
| `annealing_start_epoch` | 0 | Epoch to begin second-order |
| `bn_mode` | `"per_step"` | `"transductive"`, `"per_step"`, `"frozen"` |

### EpisodeConfig

| Field | Default | Purpose |
|---|---|---|
| `n_way` | 5 | Number of classes per episode |
| `k_shot` | 1 | Support examples per class |
| `q_query` | 15 | Query examples per class |
| `episodes_per_epoch` | 600 | Episodes per training epoch |
| `dataset` | `"omniglot"` | `"omniglot"`, `"mini_imagenet"`, `"custom"` |

### MetaCheckpointConfig

| Field | Default | Purpose |
|---|---|---|
| `save_inner_lrs` | True | Include LSLR parameters |
| `save_msl_weights` | True | Include MSL weights |
| `save_rng_state` | True | Include RNG state for reproducibility |
| `save_sampler_config` | True | Include episode sampler config |

Presets: `MetaLearningFullConfig.minimal()`, `.dev()`, `.production_1b()`, `.production_3b()`, `.production_7b()`.

## Done-When Gates

| Gate | Test | Threshold |
|---|---|---|
| **(a) Meta-gradient flow** | Toy linear model: 1 inner step MAML vs hand-computed reference; grads non-zero on θ; FOMAML grads exist without create_graph; detach detector on adapted_params | Exact match / grads > 0 |
| **(b) FO variants comparable** | Omniglot 5-way 1-shot (50–200 episodes): MAML, FOMAML, Reptile step-0 and step-1 accuracy and AUAC in same ballpark | Not broken / not flatlining |
| **(c) Phase 7 CPU dev mode** | Conv4 backbone, 50 episodes, 1–3 inner steps: one epoch end-to-end, metrics JSON produced, loss decreases | No crash, loss ↓ |

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Zero meta-gradients | Accidental detach in inner loop | Verify `create_graph=True` for MAML; add detach detector tests |
| NaN in inner loop under AMP | fp16 meta-gradients unstable | Force fp32 for inner loop; use loss scaling + fp32 master |
| Adaptation curve flat | Inner LR too small or too large | Tune `inner_lr`; enable LSLR for per-layer adaptation |
| FOMAML matches random | First-order approx too coarse for deep models | Increase inner steps; try MAML with gradient clipping |
| Reptile diverges | Interpolation rate too high | Reduce epsilon; use linear decay schedule |
| Episode sampling not reproducible | On-the-fly shuffle not seeded to episode | Key RNG to `(global_seed, epoch, episode_idx)` |
| BN statistics leak across tasks | Shared running stats during inner loop | Use `per_step` or `frozen` BN mode |
| LSLR collapses to zero | No lower bound on learned LRs | Clamp LRs to `[1e-6, 1.0]` |
| MSL weights concentrate on last step | Unconstrained optimization | Normalize weights via softmax; try `linear_increase` |
| Checkpoint missing hyperparams | Only saved model weights | Include inner steps, LRs, algo type, sampler config |

## Anti-Patterns

- **Monkey-patching modules** for inner loop — use `functional_call` or explicit param dicts
- **Single scalar LR for all layers** — at minimum support per-layer; LSLR for per-step too
- **Omitting create_graph toggle** — this is the MAML vs FOMAML distinction; must be explicit
- **fp16 inner loop** — meta-gradients are fragile; always fp32 for inner computations
- **Non-deterministic episode sampling** — every episode must be reproducible from seed + index
- **Storing only model weights in checkpoint** — inner-loop hyperparams change the learned initialization
- **Processing all tasks sequentially** — vectorize with vmap or at least batch-parallel where possible
- **Hardcoded 5-way 1-shot** — always parameterize N, K, Q through EpisodeConfig
- **No adaptation curve logging** — step-by-step accuracy is the primary diagnostic

## Additional Resources

### Reference Files

- **`references/inner-loop-engine.md`** — torch.func patterns, higher integration, custom SGD, AMP safety, gradient clipping
- **`references/algorithm-variants.md`** — MAML/FOMAML/Reptile implementations, MAML++ (LSLR, MSL, annealing, BN)
- **`references/episodic-sampling.md`** — EpisodeSampler, Omniglot/mini-ImageNet, deterministic splits, augmentation
- **`references/checkpoint-format.md`** — Checkpoint schema, hyperparameter serialization, RNG state, versioning
- **`references/testing-matrix.md`** — All test cases: gradient flow, adaptation curves, CPU dev mode, checkpoint round-trip

### Asset Templates

- **`assets/inner_loop_template.py`** — InnerLoopEngine: torch.func, higher, custom backends, AMP safety, self-test
- **`assets/meta_algorithms_template.py`** — MAML, FOMAML, Reptile: adapt(), meta_loss(), outer loop, self-test
- **`assets/maml_plus_template.py`** — LSLR, MSL, derivative-order annealing, BN modes, self-test
- **`assets/episode_sampler_template.py`** — EpisodeSampler, OmniglotSampler, MiniImageNetSampler, self-test
- **`assets/meta_metrics_template.py`** — AdaptationCurve, AUAC, gradient diagnostics, checkpoint I/O, self-test
- **`assets/meta_config_template.py`** — All configs, presets, serialization, self-test

### Scripts

- **`scripts/validate_meta_learning.py`** — Runtime contract validation (gradient flow, adaptation curves, CPU dev mode)
- **`scripts/gen_meta_tests.py`** — Generates `tests/test_meta_learning.py` (~80+ test cases)
- **`scripts/meta_benchmark.py`** — Benchmark inner-loop throughput, FO vs SO speed, vectorized episodes
