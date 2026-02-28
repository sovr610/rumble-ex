---
name: Active Inference Agent (Generative Model + EFE + Empowerment)
description: >-
  This skill should be used when the user asks to "implement active inference", "add EFE computation",
  "implement expected free energy", "add empowerment estimation", "implement generative model",
  "add latent state encoder", "implement transition model", "add preference model",
  "implement planning rollouts", "add CEM planner", "implement amortized policy",
  "add pymdp backend", "implement offline RL", "add Minari integration",
  "implement pragmatic value", "add epistemic value", "implement instrumental value",
  "add action selection", "implement belief updating", "add world model training",
  "implement POMDP planning", "add rollout engine", "implement latent imagination",
  "add horizon normalization", "implement cross-entropy method planning",
  "add preference learning", "implement variational empowerment",
  or mentions active inference, expected free energy decomposition, POMDP planning,
  empowerment estimation, latent imagination, or decision-as-inference in the cognitive pipeline.
version: 0.1.0
---

# Active Inference Agent (Generative Model + EFE + Empowerment)

## Purpose

This skill standardizes the "decision as inference" layer (Phase 5): given an observation
(or workspace state), infer a latent state, roll out candidate action sequences through a
learned generative model, score each policy by Expected Free Energy (EFE), and choose (or
amortize) the best action. Optionally swaps in a discrete POMDP backend via `pymdp` for
regression testing.

## Key Files

| Target Module | Template Asset | Purpose |
|---|---|---|
| `brain_ai/decision/active_inference.py` | `assets/active_inference_template.py` | Main agent: reset, infer_state, plan, act |
| `brain_ai/decision/generative_model.py` | `assets/generative_model_template.py` | LatentEncoder q(s|o), Likelihood P(o|s), Transition P(s'|s,a), Preferences |
| `brain_ai/decision/efe.py` | `assets/efe_template.py` | Pure EFE functions, 3-term decomposition, sum invariant |
| `brain_ai/decision/planners.py` | `assets/planners_template.py` | RolloutEngine, RandomShooting, CEM planner |
| `brain_ai/decision/amortized_policy.py` | `assets/amortized_policy_template.py` | Amortized policy distillation from planner |
| `brain_ai/decision/pymdp_backend.py` | `assets/pymdp_backend_template.py` | Optional discrete POMDP via pymdp |
| `brain_ai/config.py` (extend) | `assets/active_inference_config_template.py` | ActiveInferenceConfig, EFEConfig, PlannerConfig, etc. |

## Public Contract

```python
reset(batch_size: int, device: torch.device) -> AgentState
infer_state(o_t, ctx=None, state=None) -> (q_params, s_sample, updated_state)
plan(o_t, ctx=None, state=None, learn=False) -> ActionOutput
act(o_t, ctx=None, state=None, learn=False) -> ActionOutput
```

Input `o_t` is `(B, obs_dim)` from workspace. Optional `ctx` carries workspace slots / WM state.

## ActionOutput Contract

| Field | Shape / Type | Description |
|---|---|---|
| `action` | `(B, action_dim)` or `(B,)` | Continuous action vector or discrete action ids |
| `efe_total` | `(B,)` | Total Expected Free Energy (sum of terms) |
| `efe_terms` | `Dict[str, Tensor]` | `pragmatic`, `epistemic`, `instrumental` (each `(B,)`) |
| `horizon` | `int` | Planning horizon H used |
| `num_rollouts` | `int` | Number of candidate sequences evaluated |
| `planner_type` | `str` | `"random_shooting"`, `"cem"`, or `"amortized"` |
| `seed` | `Optional[int]` | RNG seed for reproducibility |
| `debug` | `Optional[Dict]` | Trajectories summary, uncertainty stats, preference match |

**Hard invariant**: `|sum(efe_terms.values()) - efe_total| < 1e-5` for every batch element.

## Generative Model Stack

Four learned components, each independently testable:

| Component | Notation | Input | Output |
|---|---|---|---|
| Latent encoder | `q(s\|o)` | observation `o_t` (+ optional `ctx`) | posterior params `(mu, log_var)` or categorical logits |
| Likelihood | `P(o\|s)` | latent state `s` | predicted observation distribution params |
| Transition | `P(s'\|s,a)` | latent `s`, action `a` | next-state distribution params (ensemble for uncertainty) |
| Preferences | `C` | — | target observation distribution (fixed, learned, or reward-derived) |

Sampling: reparameterization trick for continuous; Gumbel-softmax for differentiable discrete.
Transition stability: ensemble of models or mean+variance prediction with log-variance clamping.

See `references/generative-model.md` for detailed architecture and implementation.

## EFE Decomposition

Three terms computed by **pure functions** (no side effects, unit-testable in isolation):

```
G(pi) = Sum_{t=1..H} [ pragmatic(t) + epistemic(t) + instrumental(t) ]
```

| Term | Formula | Intuition |
|---|---|---|
| Pragmatic | `E_q(o_t\|pi)[ -log p_pref(o_t) ]` | Penalize outcomes violating preferences |
| Epistemic | `E[ KL( q(s_t\|o_t,pi) \|\| q(s_t\|pi) ) ]` | Reward uncertainty reduction |
| Instrumental | `I(A_{t:t+K}; S_{t+K})` (approx.) | Reward keeping options open (empowerment) |

Instrumental term stored as **negative empowerment** (non-positive) in `efe_terms["instrumental"]`, so all three terms sum directly: `efe_total = w_p*pragmatic + w_e*epistemic + w_i*instrumental`. Telemetry may log the absolute empowerment value separately.

See `references/efe-decomposition.md` for derivations, pure function signatures, and sum invariant testing.

## Planning and Rollouts

| Mode | Description | Use Case |
|---|---|---|
| Random shooting | Sample N action sequences, evaluate EFE, pick best | Baseline, debugging |
| CEM | Iterative: sample -> select elite -> refit distribution -> resample | Better quality, production |
| Amortized | Trained policy pi_theta(a\|o,ctx) approximating planner | Fast inference |

Horizon stability: normalize EFE by horizon (mean per step) or use discount factor gamma.
Candidate pool size and preference/epistemic scale must remain consistent across horizon values.

See `references/planning-rollouts.md` for rollout engine, CEM iterations, and amortized distillation.

## Optional pymdp Backend

When `pymdp` is installed, provide a discrete POMDP backend building A/B/C/D arrays and using
pymdp planning routines. Valuable for:
- Regression tests (neural EFE matches discrete reference on toy problems)
- Debugging decomposition correctness without neural approximation noise

See `references/pymdp-integration.md` for array construction, policy evaluation, and regression tests.

## Offline RL via Minari

Phase 5 offline mode uses Minari for Gymnasium-aligned dataset loading:
1. Load dataset -> extract episodes `(o_t, a_t, o_{t+1}, done, reward)`
2. Train world model components via supervised sequence prediction in latent space
3. Define preferences (fixed from task goal or learned from reward)
4. Evaluate: offline planning on held-out trajectories

See `references/offline-rl-minari.md` for dataset loading, training loops, and evaluation protocols.

## Configuration Surface

### ActiveInferenceConfig

| Field | Default | Purpose |
|---|---|---|
| `obs_dim` | 4096 | Workspace observation dimension |
| `state_dim` | 256 | Latent state dimension |
| `action_dim` | 128 | Action space size |
| `hidden_dim` | 512 | Hidden layer width |
| `planning_horizon` | 8 | Rollout depth H |
| `num_rollouts` | 128 | Candidate action sequences N |

### EFEConfig

| Field | Default | Purpose |
|---|---|---|
| `pragmatic_weight` | 1.0 | Goal-directedness weight |
| `epistemic_weight` | 1.0 | Information gain weight |
| `instrumental_weight` | 0.1 | Empowerment weight |
| `num_samples` | 32 | Monte Carlo samples for EFE |
| `discount_factor` | 0.99 | Temporal discount gamma |
| `normalize_by_horizon` | True | Mean-per-step EFE normalization |

### PlannerConfig

| Field | Default | Purpose |
|---|---|---|
| `planner_type` | `"cem"` | `"random_shooting"`, `"cem"`, `"amortized"`, `"mppi"` |
| `cem_iterations` | 5 | CEM refinement rounds |
| `cem_elite_fraction` | 0.1 | Top fraction for CEM refit |
| `cem_temperature` | 1.0 | Sampling temperature |
| `action_temperature` | 1.0 | Final action selection temperature |

### PreferenceConfig

| Field | Default | Purpose |
|---|---|---|
| `mode` | `"learned"` | `"fixed"`, `"learned"`, `"reward_derived"` |
| `preference_dim` | 256 | Preference embedding dimension |
| `learn_preferences` | True | Allow gradient updates to preferences |
| `prior_strength` | 0.1 | KL regularization toward prior |

Presets: `ActiveInferenceFullConfig.minimal()`, `.dev()`, `.production_1b()`, `.production_3b()`, `.production_7b()`.

## Done-When Gates

| Gate | Test | Threshold |
|---|---|---|
| **(a) EFE sum invariant** | Feed synthetic distributions with closed-form EFE; assert `\|sum(terms) - total\| < 1e-5` | Exact match |
| **(b) Horizon stability** | Same seed, compare actions for H=3,5,7 with normalization; EFE per-step consistent | Action stability |
| **(c) Offline RL end-to-end** | Load Minari dataset, train world model, run planner, produce actions + EFE logs | No crashes, metrics JSON |

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| EFE terms don't sum to total | Side effects in term computation | Use pure functions, test in isolation |
| Actions oscillate with horizon | Unnormalized EFE scaling | Enable `normalize_by_horizon`, fix discount |
| Empowerment collapses to 0 | Source/planning networks collapsed | Add entropy bonus, check gradient flow |
| Transition model predicts mean | Ensemble/variance not used | Enable ensemble or stochastic output |
| Preferences dominate everything | Pragmatic weight too high | Balance weights, log all terms |
| CEM converges to local optima | Too few samples or iterations | Increase `num_rollouts`, `cem_iterations` |
| Amortized policy diverges from planner | Stale training data | Online distillation, periodic refresh |
| pymdp regression fails | Neural EFE scale mismatch | Normalize both before comparison |
| Minari dataset shape mismatch | Wrong environment wrapper | Check obs/action space alignment |

## Anti-Patterns

- **Non-pure EFE functions** — EFE term computation must be side-effect free for testing
- **Summing EFE terms with different scales** — normalize each term before weighting
- **Hardcoded preference distributions** — always use `Preferences` module, even for fixed prefs
- **Skipping ensemble for transition model** — single deterministic model hides uncertainty
- **Training amortized policy on stale data** — distill from current planner, not cached actions
- **Ignoring log-variance clamping** — transition model variance can diverge
- **Using fp16 for EFE computation** — sum invariant needs fp32 precision
- **Branching on planner type** outside planners module — use unified `Planner.plan()` interface
- **No discount or horizon normalization** — EFE magnitude grows with H, causing instability

## Additional Resources

### Reference Files

- **`references/generative-model.md`** — Full spec: latent encoder, likelihood, transition, preferences, sampling, stability
- **`references/efe-decomposition.md`** — EFE derivation, pure functions, sum invariant, term normalization
- **`references/planning-rollouts.md`** — Rollout engine, random shooting, CEM, amortized policy, horizon stability
- **`references/pymdp-integration.md`** — Discrete POMDP backend, A/B/C/D arrays, regression testing
- **`references/offline-rl-minari.md`** — Minari dataset loading, world model training, evaluation protocol
- **`references/testing-matrix.md`** — All test cases: EFE invariant, horizon stability, offline RL, pymdp regression

### Asset Templates

- **`assets/active_inference_template.py`** — ActiveInferenceAgent: reset, infer_state, plan, act, self-test
- **`assets/generative_model_template.py`** — LatentEncoder, LikelihoodDecoder, TransitionModel, Preferences, self-test
- **`assets/efe_template.py`** — Pure EFE functions, pragmatic/epistemic/instrumental, sum invariant, self-test
- **`assets/planners_template.py`** — RolloutEngine, RandomShootingPlanner, CEMPlanner, self-test
- **`assets/amortized_policy_template.py`** — AmortizedPolicy, PlannerDistiller, online/offline training, self-test
- **`assets/pymdp_backend_template.py`** — PyMDPBackend, array construction, EFE regression, self-test
- **`assets/active_inference_config_template.py`** — All configs, presets, serialization, self-test

### Scripts

- **`scripts/validate_active_inference.py`** — Runtime contract validation (EFE sum, planner consistency, state management)
- **`scripts/gen_active_inference_tests.py`** — Generates `tests/test_active_inference.py` (~80+ test cases)
- **`scripts/toy_benchmark.py`** — Deterministic toy benchmark harness (MiniGrid online + Minari offline, <2 min CPU)
