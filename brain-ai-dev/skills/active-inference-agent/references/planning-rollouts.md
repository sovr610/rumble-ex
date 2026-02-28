# Planning and Rollouts

## Overview

Planning in active inference is **latent imagination**: simulate candidate action sequences through
the learned world model entirely in latent space, score each trajectory by Expected Free Energy
(EFE), and select the action sequence that minimizes EFE. No environment interaction is required
during planning -- all evaluation happens inside the agent's generative model.

The planning loop proceeds as follows:

1. Encode the current observation into a latent belief state: `s_t ~ q(s|o_t)`.
2. Generate a pool of candidate action sequences of length H (the planning horizon).
3. For each candidate, unroll the transition model forward H steps in latent space.
4. At each step, decode the predicted observation and compute per-step EFE (pragmatic + epistemic + instrumental).
5. Sum (optionally discount) per-step EFE across the horizon to get a total score per candidate.
6. Select the candidate with the lowest total EFE. Execute only the first action; replan at the next timestep.

Three planner implementations are supported, each sharing a common interface:

| Planner | Strategy | Latency | Quality | Use Case |
|---|---|---|---|---|
| Random shooting | Uniform sampling, single pass | Low | Baseline | Debugging, low-dim actions |
| CEM | Iterative distribution refinement | Medium | High | Production continuous control |
| Amortized policy | Learned neural network | Minimal | Depends on distillation | Fast inference, deployment |

All planners operate through a shared `RolloutEngine` that handles the forward simulation,
observation decoding, and EFE accumulation. The planner's only job is to propose action sequences
and interpret the resulting scores.

---

## Rollout Engine

The rollout engine is the computational core of planning. It takes a batch of initial states and a
tensor of candidate action sequences, unrolls the transition model, decodes observations, and
returns per-step and total EFE scores.

### Core Loop

Given an initial state `s_t ~ q(s|o_t)` and a candidate action sequence `a_{t:t+H}`:

```
for k in range(H):
    s_{t+k+1}_mu, s_{t+k+1}_logvar = transition_model(s_{t+k}, a_{t+k})
    s_{t+k+1} = sample(s_{t+k+1}_mu, s_{t+k+1}_logvar)   # or use mean
    o_{t+k+1} = likelihood_model(s_{t+k+1})
    g_{t+k} = efe_computer(s_{t+k}, s_{t+k+1}, o_{t+k+1}, preferences)
total_efe = sum(gamma^k * g_{t+k} for k in range(H))
```

### Parallelization

Expand the batch dimension to evaluate N rollouts simultaneously. If the original batch size is B,
reshape to `(B * N, state_dim)` before entering the unroll loop:

```python
# action_sequences: (B, N, H, action_dim) for continuous, (B, N, H) for discrete
# Reshape initial state: (B, state_dim) -> (B*N, state_dim)
states = initial_state.unsqueeze(1).expand(B, N, -1).reshape(B * N, state_dim)
actions_flat = action_sequences.reshape(B * N, H, -1)  # (B*N, H, action_dim)
```

After the unroll, reshape EFE back to `(B, N)` and select the best candidate per batch element.

### Memory Management

During inference (not training the planner), detach intermediate states from the computation graph
to prevent memory accumulation across long horizons:

```python
if not self.training:
    state = state.detach()
```

When training the transition model end-to-end through the planner (e.g., Dreamer-style), retain
the graph but use gradient checkpointing for horizons H > 10.

### Deterministic Mode

For reproducible evaluation and debugging, use the mean of the transition model instead of
sampling. Set `deterministic=True` on the rollout engine:

```python
if deterministic:
    next_state = next_state_mu  # skip reparameterization
else:
    next_state = next_state_mu + next_state_std * torch.randn_like(next_state_std)
```

### RolloutEngine Class

```python
@dataclass
class RolloutResult:
    total_efe: torch.Tensor           # (B, N) total EFE per candidate
    per_step_efe: torch.Tensor        # (B, N, H) EFE at each horizon step
    efe_terms: Dict[str, torch.Tensor] # per-term breakdown, each (B, N, H)
    trajectories: torch.Tensor        # (B, N, H, state_dim) latent trajectories
    predicted_obs: torch.Tensor       # (B, N, H, obs_dim) decoded observations


class RolloutEngine(nn.Module):
    def __init__(
        self,
        transition_model: nn.Module,    # P(s'|s,a): returns (mu, logvar)
        likelihood_model: nn.Module,    # P(o|s): returns observation
        efe_computer: nn.Module,        # computes per-step EFE terms
        config: RolloutConfig,
    ):
        super().__init__()
        self.transition_model = transition_model
        self.likelihood_model = likelihood_model
        self.efe_computer = efe_computer
        self.config = config

    def rollout(
        self,
        initial_state: torch.Tensor,       # (B, state_dim)
        action_sequences: torch.Tensor,    # (B, N, H, action_dim) or (B, N, H)
        preferences: torch.Tensor,         # (B, obs_dim) or Preferences module
        deterministic: bool = False,
    ) -> RolloutResult:
        """
        Unroll transition model for all candidate action sequences.

        Returns RolloutResult with total_efe (B, N), per_step_efe (B, N, H),
        efe_terms dict, and latent trajectories (B, N, H, state_dim).
        """
        ...

    def rollout_single(
        self,
        state: torch.Tensor,              # (B*N, state_dim)
        actions: torch.Tensor,             # (B*N, H, action_dim)
        preferences: torch.Tensor,
        deterministic: bool = False,
    ) -> RolloutResult:
        """Inner loop operating on flattened batch. Called by rollout()."""
        ...
```

### RolloutConfig

```python
@dataclass
class RolloutConfig:
    horizon: int = 8                    # planning depth H
    discount_factor: float = 0.99       # gamma for temporal discounting
    normalize_by_horizon: bool = True   # divide total EFE by H
    deterministic: bool = False         # use transition mean (no sampling)
    use_ensemble: bool = True           # ensemble transition for uncertainty
    ensemble_size: int = 5              # number of ensemble members
    gradient_checkpoint: bool = False   # checkpoint for long horizons
```

---

## Random Shooting Planner

The simplest planning strategy. Sample N action sequences uniformly at random, evaluate all of
them through the rollout engine in a single batched forward pass, and pick the one with the lowest
total EFE.

### Algorithm

```
1. For each batch element, sample N action sequences of length H:
   - Discrete: sample each action from Categorical(uniform) independently
   - Continuous: sample from N(0, sigma), clip to [action_low, action_high]
2. Pass all N sequences to rollout engine -> total_efe (B, N)
3. Select best: idx = argmin(total_efe, dim=1)
4. Return action_sequences[b, idx[b], 0] as the first action to execute
```

### Continuous Action Sampling

```python
# Sample from isotropic Gaussian, clip to bounds
actions = torch.randn(B, N, H, action_dim) * sampling_std
actions = actions.clamp(action_low, action_high)
```

### Discrete Action Sampling

```python
# Uniform categorical sampling
actions = torch.randint(0, num_actions, (B, N, H))
```

### Action Selection

Two modes for selecting the final action from the scored candidates:

1. **Argmin** (deterministic): `best_idx = total_efe.argmin(dim=1)`
2. **Softmax sampling** (stochastic): sample from `softmax(-total_efe / temperature)`

Use argmin for evaluation and softmax sampling for training (maintains exploration).

### Configuration

```python
@dataclass
class RandomShootingConfig:
    num_rollouts: int = 128         # N: number of candidate sequences
    action_bounds: Tuple[float, float] = (-1.0, 1.0)  # continuous action clipping
    sampling_std: float = 1.0       # std for Gaussian sampling (continuous)
    selection_mode: str = "argmin"  # "argmin" or "softmax"
    temperature: float = 1.0       # softmax temperature (if selection_mode="softmax")
```

### Advantages

- No iterative refinement: single forward pass through rollout engine.
- Trivially parallelizable: all N rollouts are independent.
- Easy to debug: the entire candidate set is visible for inspection.
- No hyperparameters to tune beyond N and sampling_std.

### Disadvantages

- Sample-inefficient: coverage of action space degrades exponentially with `H * action_dim`.
- Poor for high-dimensional continuous actions (curse of dimensionality).
- No learning across planning steps: each call is independent.

---

## CEM Planner (Cross-Entropy Method)

Iterative refinement of a parametric action distribution. Each iteration samples candidates from
the current distribution, evaluates them, selects the elite subset, and refits the distribution to
the elites. This progressively concentrates probability mass on high-quality action sequences.

### Algorithm

```
1. Initialize distribution: mu = zeros(H, action_dim), sigma = initial_std * ones(H, action_dim)
2. For iter in range(cem_iterations):
   a. Sample N action sequences from N(mu, diag(sigma^2)):
      actions ~ mu + sigma * randn(B, N, H, action_dim)
      Clamp to [action_low, action_high]
   b. Evaluate: total_efe (B, N) = rollout_engine(initial_state, actions, preferences)
   c. Select elites: top ceil(N * elite_fraction) by lowest EFE, per batch element
   d. Refit: mu_new = mean(elites), sigma_new = std(elites)
   e. Apply momentum: mu = alpha * mu_new + (1 - alpha) * mu_old
                       sigma = alpha * sigma_new + (1 - alpha) * sigma_old
3. Final action: mu[0] (first timestep of final mean sequence)
```

### Elite Selection

Select elites independently for each batch element:

```python
# total_efe: (B, N)
num_elites = max(1, int(N * elite_fraction))
_, elite_indices = total_efe.topk(num_elites, dim=1, largest=False)  # lowest EFE
# elite_indices: (B, num_elites)

# Gather elite action sequences
elite_actions = actions.gather(
    dim=1,
    index=elite_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, action_dim)
)  # (B, num_elites, H, action_dim)
```

### Distribution Refitting

```python
mu = elite_actions.mean(dim=1)     # (B, H, action_dim)
sigma = elite_actions.std(dim=1)   # (B, H, action_dim)
sigma = sigma.clamp(min=min_std)   # prevent collapse
```

### Temperature Annealing

Optionally anneal the sampling temperature across CEM iterations to sharpen the distribution:

```python
for i in range(cem_iterations):
    temp = initial_temp * (final_temp / initial_temp) ** (i / max(cem_iterations - 1, 1))
    actions = mu + sigma * temp * torch.randn(B, N, H, action_dim)
```

### Momentum

Blend the new distribution with the previous iteration to stabilize convergence:

```python
mu = momentum * mu_prev + (1 - momentum) * mu_new
sigma = momentum * sigma_prev + (1 - momentum) * sigma_new
```

A momentum of 0.0 means no blending (pure refit). Values of 0.1--0.3 are typical.

### Warm-Starting

Across consecutive planning steps, shift the previous solution forward by one timestep and use it
to initialize the next CEM run:

```python
# Previous solution: mu_prev (H, action_dim)
# Warm-start: shift left, pad last step with zeros
mu_init = torch.cat([mu_prev[1:], torch.zeros(1, action_dim)], dim=0)
```

This significantly reduces the number of CEM iterations needed for convergence in online settings.

### Configuration

```python
@dataclass
class CEMConfig:
    num_rollouts: int = 128             # N: candidates per iteration
    cem_iterations: int = 5             # refinement rounds
    cem_elite_fraction: float = 0.1     # top 10% selected
    cem_temperature: float = 1.0        # sampling temperature
    cem_temperature_final: float = 0.5  # annealed final temperature
    momentum: float = 0.1              # distribution blending factor
    min_std: float = 0.01              # minimum sigma to prevent collapse
    action_bounds: Tuple[float, float] = (-1.0, 1.0)
    warm_start: bool = True            # reuse previous solution
```

---

## Horizon Stability

Horizon stability is a critical requirement: the agent's action selection must not oscillate when
the planning horizon H changes. Without proper normalization, raw EFE grows linearly with H,
causing longer horizons to dominate action ranking and producing unstable behavior.

### The Problem

Raw total EFE is `G = sum_{k=0}^{H-1} g_k`. If per-step EFE `g_k ~ c` (roughly constant), then
`G ~ c * H`. When comparing actions under H=3 versus H=7, the H=7 scores are approximately 2.3x
larger, which can flip action rankings even when per-step preferences are identical.

### Solution 1: Normalize by Horizon

Divide total EFE by H to get mean per-step EFE:

```python
if normalize_by_horizon:
    total_efe = total_efe / H
```

This makes scores directly comparable across horizons. Enable by default.

### Solution 2: Discount Factor

Apply temporal discounting with `gamma < 1`:

```python
# Per-step discount weights
weights = gamma ** torch.arange(H, device=device)  # [1, gamma, gamma^2, ...]
total_efe = (per_step_efe * weights).sum(dim=-1)
```

For `gamma = 0.95` and large H, the effective horizon saturates at `1 / (1 - gamma) = 20` steps,
bounding the total EFE regardless of nominal H.

### Solution 3: Consistent Candidate Pool

Ensure the number of candidate action sequences N does not change with H. If N varies, the
quality of the best candidate changes (more samples = better minimum), confounding horizon
comparisons.

### Weight Consistency

The EFE term weights (`pragmatic_weight`, `epistemic_weight`, `instrumental_weight`) must be
**horizon-independent**. Do not scale weights by H or use horizon-dependent schedules. If weight
tuning is needed, tune on a fixed H and verify stability across H values.

### Verification Protocol

Run the following test to verify horizon stability:

```python
def test_horizon_stability(planner, initial_state, preferences, seed=42):
    """Verify action selection is stable across horizon values."""
    results = {}
    for H in [3, 5, 7]:
        torch.manual_seed(seed)
        planner.config.horizon = H
        result = planner.plan(initial_state, preferences)
        results[H] = {
            'action': result.best_action,
            'per_step_efe': result.per_step_efe.mean().item(),
            'total_efe': result.total_efe.min(dim=1).values.mean().item(),
        }

    # Check 1: actions should be identical (or differ only near decision boundaries)
    assert results[3]['action'] == results[5]['action'] == results[7]['action'], \
        f"Actions changed with horizon: {results}"

    # Check 2: per-step EFE should be consistent (within 20% relative)
    base = results[3]['per_step_efe']
    for H in [5, 7]:
        ratio = abs(results[H]['per_step_efe'] - base) / (abs(base) + 1e-8)
        assert ratio < 0.2, f"Per-step EFE drifted by {ratio:.1%} at H={H}"
```

### Logging Requirements

Always log the following for debugging horizon-related issues:

| Field | Shape | Purpose |
|---|---|---|
| `horizon` | scalar | Current planning horizon H |
| `discount_factor` | scalar | Gamma used for discounting |
| `per_step_efe` | `(B, N, H)` | EFE at each step of each candidate |
| `per_step_pragmatic` | `(B, N, H)` | Pragmatic term per step |
| `per_step_epistemic` | `(B, N, H)` | Epistemic term per step |
| `per_step_instrumental` | `(B, N, H)` | Instrumental term per step |
| `total_efe_raw` | `(B, N)` | Before normalization |
| `total_efe_normalized` | `(B, N)` | After normalization / discounting |

---

## Amortized Policy

Planning is expensive: it requires `N * H` forward passes through the transition model per action
selection (or `N * H * cem_iterations` for CEM). An amortized policy is a neural network trained
to approximate the planner's output in a single forward pass.

### Architecture

```python
class AmortizedPolicy(nn.Module):
    """
    pi_theta(a | o, ctx): fast action selection trained to match planner output.

    Input:
        observation: (B, obs_dim) -- raw observation or workspace state
        context: (B, ctx_dim) -- optional workspace slots, working memory, goal

    Output (discrete):
        logits: (B, num_actions) -- categorical action distribution

    Output (continuous):
        mu: (B, action_dim) -- Gaussian mean
        log_std: (B, action_dim) -- Gaussian log standard deviation
    """

    def __init__(self, config: AmortizedPolicyConfig):
        super().__init__()
        input_dim = config.obs_dim + config.ctx_dim
        layers = []
        dim = input_dim
        for _ in range(config.num_layers):
            layers.extend([nn.Linear(dim, config.hidden_dim), nn.ReLU()])
            dim = config.hidden_dim
        self.trunk = nn.Sequential(*layers)

        if config.discrete:
            self.head = nn.Linear(dim, config.num_actions)
        else:
            self.mu_head = nn.Linear(dim, config.action_dim)
            self.log_std_head = nn.Linear(dim, config.action_dim)

    def forward(
        self,
        observation: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if context is not None:
            x = torch.cat([observation, context], dim=-1)
        else:
            x = observation
        h = self.trunk(x)

        if hasattr(self, 'head'):
            return self.head(h)  # (B, num_actions) logits
        else:
            mu = self.mu_head(h)
            log_std = self.log_std_head(h).clamp(-5.0, 2.0)
            return mu, log_std
```

### Training (Distillation)

Train the amortized policy to replicate the planner's action selection. Three distillation modes:

#### Hard Distillation

Use the planner's selected action as a supervised target:

```python
# Discrete: cross-entropy against planner's argmin action
planner_action = planner.plan(obs, preferences).best_action  # (B,)
logits = policy(obs, ctx)
loss = F.cross_entropy(logits, planner_action)

# Continuous: MSE on the planner's selected action mean
planner_mu = planner.plan(obs, preferences).best_action_sequence[:, 0]  # (B, action_dim)
pred_mu, pred_log_std = policy(obs, ctx)
loss = F.mse_loss(pred_mu, planner_mu)
```

#### Soft Distillation

Match the full distribution over actions, not just the argmax. This preserves multi-modal
structure in the planner's output:

```python
# Discrete: KL divergence between planner's softmax(-EFE) and policy logits
planner_probs = F.softmax(-total_efe / temperature, dim=-1)  # (B, num_actions)
policy_log_probs = F.log_softmax(policy(obs, ctx), dim=-1)
loss = F.kl_div(policy_log_probs, planner_probs, reduction='batchmean')

# Continuous: NLL of planner's action under policy's Gaussian
planner_action = planner.plan(obs, preferences).best_action_sequence[:, 0]
pred_mu, pred_log_std = policy(obs, ctx)
dist = torch.distributions.Normal(pred_mu, pred_log_std.exp())
loss = -dist.log_prob(planner_action).sum(dim=-1).mean()
```

#### Offline Distillation

Train from a dataset of pre-collected (observation, action) pairs, optionally from Minari
episodes. Reweight samples by inverse EFE to prioritize high-quality actions:

```python
# Load Minari dataset episodes
# For each (obs, action) pair, compute EFE retrospectively
weights = F.softmax(-efe_scores / temperature, dim=0)
loss = (weights * per_sample_loss).sum()
```

### Usage Pattern

| Phase | Planner | Amortized Policy | Notes |
|---|---|---|---|
| Training | Active (generates targets) | Distilling in background | Both run every step |
| Inference | Disabled | Active (single forward pass) | Fast action selection |
| Audit | Active (periodic) | Active | Compare outputs, flag drift |

**Training**: Run the full planner on every observation. In a background thread or after each
planning step, compute the distillation loss and update the amortized policy.

**Inference**: Use only the amortized policy. A single forward pass through the network replaces
`N * H * cem_iterations` transition model evaluations.

**Audit**: Every `refresh_interval` steps, run both the planner and the amortized policy. Compare
their selected actions. If the KL divergence between their action distributions exceeds a threshold
(e.g., 0.5 nats), log a warning and increase the distillation learning rate or trigger a
retraining burst.

### EFE Logging

Even when using the amortized policy for action selection, always compute EFE for the selected
action. This provides:

- Monitoring: track whether the amortized policy's actions achieve low EFE.
- Anomaly detection: sudden EFE spikes indicate policy degradation.
- Term breakdown: log pragmatic, epistemic, and instrumental values for interpretability.

### Configuration

```python
@dataclass
class AmortizedPolicyConfig:
    enabled: bool = True
    obs_dim: int = 4096
    ctx_dim: int = 0                    # 0 = no context input
    action_dim: int = 128
    num_actions: int = 10               # for discrete action spaces
    discrete: bool = False              # True for categorical, False for Gaussian
    hidden_dim: int = 512
    num_layers: int = 3
    distill_lr: float = 1e-4
    distill_mode: str = "soft"          # "hard", "soft", "offline"
    distill_batch_size: int = 64
    refresh_interval: int = 100         # steps between planner audits
    drift_threshold: float = 0.5        # KL nats before triggering alert
```

---

## Planner Interface

All planners share a common abstract interface. This allows the agent to swap between planning
strategies without modifying the action selection loop.

```python
@dataclass
class PlanResult:
    best_action: torch.Tensor                   # (B, action_dim) or (B,) for discrete
    best_action_sequence: torch.Tensor          # (B, H, action_dim) full sequence
    efe_total: torch.Tensor                     # (B,) total EFE of best candidate
    efe_terms: Dict[str, torch.Tensor]          # pragmatic, epistemic, instrumental (each (B,))
    per_step_efe: torch.Tensor                  # (B, H) per-step EFE of best candidate
    all_candidate_efe: torch.Tensor             # (B, N) EFE of all candidates
    debug_info: Optional[Dict[str, Any]]        # planner-specific diagnostics


class Planner(ABC):
    @abstractmethod
    def plan(
        self,
        initial_state: torch.Tensor,           # (B, state_dim)
        observation: torch.Tensor,              # (B, obs_dim) for amortized
        preferences: torch.Tensor,              # (B, obs_dim) or Preferences module
        ctx: Optional[torch.Tensor] = None,     # (B, ctx_dim) optional context
    ) -> PlanResult:
        """
        Generate and evaluate candidate action sequences, return best.

        Implementations: RandomShootingPlanner, CEMPlanner, AmortizedPlanner.
        """
        ...

    def reset(self) -> None:
        """
        Reset any internal state between episodes.

        CEM: clear warm-start buffers and momentum.
        Amortized: no-op (stateless).
        Random shooting: no-op (stateless).
        """
        ...
```

### Planner Dispatch

Select the active planner by config string:

```python
def create_planner(config: PlannerConfig, rollout_engine: RolloutEngine) -> Planner:
    if config.planner_type == "random_shooting":
        return RandomShootingPlanner(config, rollout_engine)
    elif config.planner_type == "cem":
        return CEMPlanner(config, rollout_engine)
    elif config.planner_type == "amortized":
        return AmortizedPlanner(config)
    else:
        raise ValueError(f"Unknown planner type: {config.planner_type}")
```

Do not branch on planner type outside the planners module. All downstream code should use the
`Planner.plan()` interface exclusively.

---

## Computational Budget

### Cost Per Action Selection

| Planner | Forward Passes (Transition) | Forward Passes (Likelihood) | Total |
|---|---|---|---|
| Random shooting | `N * H` | `N * H` | `2 * N * H` |
| CEM | `N * H * I` | `N * H * I` | `2 * N * H * I` |
| Amortized | 0 | 0 | 1 (policy network) |

Where N = num_rollouts, H = horizon, I = CEM iterations. Ensemble transition models multiply the
transition cost by `ensemble_size`.

### Latency Estimates

Measured on a single A100 GPU with `state_dim=256`, `action_dim=128`, `obs_dim=4096`:

| N | H | Planner | Ensemble=1 | Ensemble=5 |
|---|---|---|---|---|
| 64 | 4 | Random shooting | ~1.2 ms | ~4.8 ms |
| 128 | 8 | Random shooting | ~5.0 ms | ~21 ms |
| 128 | 8 | CEM (5 iter) | ~25 ms | ~105 ms |
| 256 | 16 | CEM (5 iter) | ~200 ms | ~850 ms |
| 1 | -- | Amortized | ~0.1 ms | -- |

### Memory Footprint

The dominant memory consumer is the trajectory tensor stored during rollouts:

```
memory = B * N * H * state_dim * sizeof(float32)
```

| B | N | H | state_dim | Memory |
|---|---|---|---|---|
| 1 | 128 | 8 | 256 | 1.0 MB |
| 1 | 256 | 16 | 256 | 4.0 MB |
| 32 | 128 | 8 | 256 | 32 MB |
| 32 | 256 | 16 | 256 | 128 MB |

Add equivalent memory for `predicted_obs` tensors (`B * N * H * obs_dim`), which can be 16x larger
if `obs_dim = 4096`.

### Strategies for Reducing Cost

1. **Truncated rollouts**: Use a shorter effective horizon than configured. Start with H=4 and
   increase only when the task requires long-horizon planning.

2. **Action chunking**: Instead of planning one action at a time, commit to executing the first K
   actions of the best sequence before replanning. Reduces planning frequency by K.

3. **Parallel evaluation**: Use the batched rollout engine. All N candidates are evaluated in a
   single batched forward pass, not sequentially.

4. **Amortized warmup**: During early training, use random shooting (cheap). Switch to CEM once
   the transition model is accurate. Distill to amortized policy for deployment.

5. **Adaptive N**: Start with small N for easy decisions (low EFE variance across candidates).
   Increase N when the top candidates have similar scores (ambiguous decisions).

6. **Mixed-precision rollouts**: Use fp16 for transition model forward passes during rollouts.
   Accumulate EFE in fp32 to maintain sum invariant precision.

---

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Actions change when H changes | Raw EFE not normalized by horizon | Enable `normalize_by_horizon` and/or set `gamma < 1` |
| CEM collapses to single action | `min_std` too low or `elite_fraction` too small | Increase `min_std` to 0.05, set `elite_fraction >= 0.1` |
| CEM stuck in local optimum | Too few candidates or iterations | Increase `num_rollouts` to 256+, `cem_iterations` to 8 |
| Random shooting always picks same action | `num_rollouts` too low for action space size | Increase N; for continuous actions use CEM instead |
| Rollout EFE is NaN | Transition model `log_var` diverged | Clamp `log_var` to `[-10, 2]` in transition model |
| Amortized policy diverges from planner | Stale distillation data or low learning rate | Use online distillation, increase `distill_lr`, decrease `refresh_interval` |
| Amortized policy outputs constant action | Distillation mode is "hard" with deterministic planner | Switch to `distill_mode="soft"` to preserve distribution shape |
| Rollout out-of-memory | Large `B * N * H * obs_dim` tensor | Reduce N or H; decode observations lazily (only when needed for EFE) |
| Per-step EFE grows over horizon | Transition model error compounds | Train transition model with multi-step loss; use ensemble |
| Epistemic term dominates early training | Transition model is uncertain everywhere | Anneal `epistemic_weight` from 0 to 1 over first 10K steps |
| Instrumental term is always zero | Empowerment source/planning networks collapsed | Add entropy regularization to source network; check gradients |
| Warm-start CEM causes repetitive actions | Previous solution biases current planning | Reduce `momentum`; add noise to warm-start initialization |

---

## Migration from Existing Code

The existing implementation in `brain_ai/decision/active_inference.py` uses a simple `select_action`
method that loops over discrete actions and evaluates each with `num_samples` Monte Carlo samples.
This section describes the migration path to the full rollout-based planner architecture.

### Current Architecture

```python
# Current: loop over actions, compute EFE per action
class ActiveInferenceAgent(nn.Module):
    def select_action(self, observation):
        state = self.encoder(observation)
        efes = []
        for a in range(self.config.action_dim):
            efe = self.compute_efe(state, a)  # MC samples internally
            efes.append(efe)
        efes = torch.stack(efes, dim=1)
        action_probs = F.softmax(-efes / temperature, dim=-1)
        return torch.multinomial(action_probs, 1)
```

### Migration Steps

**Step 1: Extract generative model components.** Move `GenerativeModel`, `StateEncoder`, and
`Preferences` into `brain_ai/decision/generative_model.py`. These remain unchanged but gain a
clean import path.

**Step 2: Implement pure EFE functions.** Create `brain_ai/decision/efe.py` with stateless
functions for pragmatic, epistemic, and instrumental value. The existing `compute_efe` method
becomes a thin wrapper calling these functions. Verify the sum invariant:
`|pragmatic + epistemic + instrumental - total| < 1e-5`.

**Step 3: Build the RolloutEngine.** Create `brain_ai/decision/planners.py`. The rollout engine
replaces the inner loop of `compute_efe` with a batched, horizon-aware unroll. Initially, wrap the
existing transition model without modification.

**Step 4: Implement RandomShootingPlanner.** This directly replaces the current `for a in range(action_dim)` loop with batched sampling. For discrete actions with small `action_dim`, this is
a drop-in replacement. For continuous actions, it enables proper action-space exploration.

**Step 5: Implement CEMPlanner.** Add iterative refinement on top of the rollout engine. This is
the production planner for continuous action spaces.

**Step 6: Add AmortizedPolicy.** Create `brain_ai/decision/amortized_policy.py`. The existing
`ImprovedActiveInferenceAgent.amortized_policy` (a simple Sequential) becomes the new
`AmortizedPolicy` class with proper distillation training. The existing
`train_amortized_policy` method migrates to a `PlannerDistiller` class.

**Step 7: Wire into BrainAI system.** Update `brain_ai/system.py` to instantiate the planner via
`create_planner()` and call `planner.plan()` instead of `agent.select_action()`. The planner
type is controlled by `PlannerConfig.planner_type` in the configuration.

### Backward Compatibility

Maintain the existing `ActiveInferenceAgent.select_action` method as a convenience wrapper that
internally creates a `RandomShootingPlanner` with `N = action_dim` (one candidate per discrete
action). This ensures that existing code calling `select_action` continues to work without
modification during the migration.

```python
class ActiveInferenceAgent(nn.Module):
    def select_action(self, observation, deterministic=False):
        """Legacy interface. Delegates to planner internally."""
        if not hasattr(self, '_legacy_planner'):
            self._legacy_planner = RandomShootingPlanner(
                config=RandomShootingConfig(num_rollouts=self.config.action_dim),
                rollout_engine=self._build_rollout_engine(),
            )
        state = self.encoder.sample(observation)
        result = self._legacy_planner.plan(state, observation, self.preferences.get_preference())
        if deterministic:
            return result.best_action, {'efe': result.efe_total}
        else:
            probs = F.softmax(-result.all_candidate_efe / self.config.action_temperature, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)
            return action, {'efe': result.efe_total, 'action_probs': probs}
```

### Validation Checklist

After migration, verify:

- [ ] `select_action` produces identical outputs (same seed) before and after migration
- [ ] EFE sum invariant holds: `|sum(terms) - total| < 1e-5` for all batch elements
- [ ] Horizon stability test passes for H=3, 5, 7
- [ ] CEM planner achieves lower EFE than random shooting on the same problem
- [ ] Amortized policy matches planner's action distribution within `drift_threshold`
- [ ] No regression in `examples/inference_demo.py` output quality
- [ ] Memory usage for `B=32, N=128, H=8` stays under 256 MB
