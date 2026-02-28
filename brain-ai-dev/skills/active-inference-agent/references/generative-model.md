# Generative Model Stack

## Overview

The generative model is the core world model for active inference. It provides four independently testable components that together allow the agent to encode observations, predict future states, reconstruct observations from latent states, and evaluate outcomes against preferences. All four components compose into the Expected Free Energy (EFE) computation and the rollout engine for planning.

| Component | Notation | Role in EFE |
|---|---|---|
| Latent encoder | `q(s\|o)` | Approximate posterior; maps high-dim observations to compact latent states |
| Likelihood | `P(o\|s)` | Observation decoder; used for ambiguity (epistemic) and preference matching (pragmatic) |
| Transition | `P(s'\|s,a)` | World dynamics; used by rollout engine for latent imagination |
| Preferences | `C` | Target distribution over observations; defines the pragmatic term in EFE |

Data flow through the stack during a single planning step:

```
Observation o_t (B, 4096)
    |
    v
[Latent Encoder q(s|o)]  -->  s_t ~ q(s|o_t)  (B, state_dim)
    |
    |   for each candidate action a:
    v
[Transition P(s'|s,a)]   -->  s_{t+1} ~ P(s'|s_t, a)  (B, state_dim)
    |
    v
[Likelihood P(o|s)]      -->  o_hat_{t+1} ~ P(o|s_{t+1})  (B, obs_dim)
    |
    v
[Preferences C]          -->  pragmatic_value = -D_KL(o_hat || C)
```

Target file: `brain_ai/decision/generative_model.py`
Existing file to migrate from: `brain_ai/decision/active_inference.py` (classes `StateEncoder`, `GenerativeModel`, `Preferences`)

---

## A) Latent State Encoder q(s|o)

### Purpose

Map high-dimensional workspace observations `o_t` into a compact latent state distribution. The encoder is the approximate posterior in the variational inference sense. It must produce distribution parameters (not point estimates) so that downstream components can reason about uncertainty.

### Input Specification

| Argument | Shape | Required | Description |
|---|---|---|---|
| `obs` | `(B, obs_dim)` | Yes | Observation from workspace, default `obs_dim=4096` |
| `ctx` | `(B, ctx_dim)` or `(B, K, D)` | No | Optional context: workspace slots, working memory state, or neuromodulatory signals |

When `ctx` is `None`, the encoder uses `obs` alone. When `ctx` is provided, the encoder conditions the posterior on both observation and context.

### Output Specification

**Continuous latent (default):**

| Output | Shape | Description |
|---|---|---|
| `mu` | `(B, state_dim)` | Mean of Gaussian posterior |
| `log_sigma` | `(B, state_dim)` | Log standard deviation (not log variance, to avoid the 0.5 factor during sampling) |

Sample via reparameterization: `s = mu + exp(log_sigma) * epsilon`, where `epsilon ~ N(0, I)`.

**Discrete latent (alternative):**

| Output | Shape | Description |
|---|---|---|
| `logits` | `(B, num_discrete_states)` | Categorical logits over discrete states |

Sample via Gumbel-softmax: `s = gumbel_softmax(logits, tau=temperature, hard=False)` during training, `hard=True` during evaluation.

### Architecture

```
obs (B, obs_dim) ----+
                     |---> [concat / cross_attn / film] ---> MLP ---> (mu, log_sigma)
ctx (B, ctx_dim) ----+                                               or (logits)
```

MLP specification (default 3 layers with residual connections):

```python
class LatentEncoder(nn.Module):
    def __init__(self, obs_dim, state_dim, hidden_dim=512, num_layers=3,
                 ctx_dim=0, ctx_mode="concat", latent_type="continuous",
                 num_discrete_states=32):
        super().__init__()
        input_dim = obs_dim + (ctx_dim if ctx_mode == "concat" else 0)

        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
        self.backbone = nn.ModuleList(layers)

        # Residual projections (skip connections when dims match)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        if latent_type == "continuous":
            self.mu_head = nn.Linear(hidden_dim, state_dim)
            self.log_sigma_head = nn.Linear(hidden_dim, state_dim)
        else:
            self.logit_head = nn.Linear(hidden_dim, num_discrete_states)
```

### Context Conditioning Modes

| Mode | Mechanism | When to Use |
|---|---|---|
| `concat` | Concatenate `[obs, ctx]` along feature dim before MLP | Default; simple, low overhead |
| `cross_attention` | Multi-head attention: query=obs embedding, key/value=ctx slots `(B, K, D)` | When ctx has variable-length slot structure |
| `film` | FiLM conditioning: `gamma, beta = film_net(ctx); h = gamma * h + beta` | When ctx is low-dimensional (e.g., neuromodulator levels) |

For `cross_attention` mode, pool the `(B, K, D)` context slots via learned attention before concatenation with the observation embedding:

```python
# Cross-attention context pooling
attn_output = self.ctx_attention(
    query=obs_embed.unsqueeze(1),   # (B, 1, hidden_dim)
    key=ctx_slots,                   # (B, K, D)
    value=ctx_slots                  # (B, K, D)
)  # -> (B, 1, D) -> squeeze -> (B, D)
```

### KL Regularization

Regularize the posterior toward a standard prior to prevent posterior collapse and ensure smooth latent space geometry.

**Continuous:** KL toward `N(0, I)`:

```python
kl_loss = -0.5 * torch.sum(1 + 2*log_sigma - mu.pow(2) - (2*log_sigma).exp(), dim=-1)
# Shape: (B,)
```

**Discrete:** KL toward uniform categorical:

```python
q = F.softmax(logits, dim=-1)
log_q = F.log_softmax(logits, dim=-1)
kl_loss = torch.sum(q * (log_q - math.log(1.0 / num_discrete_states)), dim=-1)
# Shape: (B,)
```

Scale by `kl_weight` (default 1.0). Anneal `kl_weight` from 0 to 1 over initial training steps when using beta-VAE style training (prevents posterior collapse).

### Testing Contract

```python
def test_encoder_shapes():
    enc = LatentEncoder(obs_dim=4096, state_dim=256)
    obs = torch.randn(8, 4096)
    mu, log_sigma = enc(obs)
    assert mu.shape == (8, 256)
    assert log_sigma.shape == (8, 256)

def test_encoder_with_context():
    enc = LatentEncoder(obs_dim=4096, state_dim=256, ctx_dim=512, ctx_mode="concat")
    obs = torch.randn(8, 4096)
    ctx = torch.randn(8, 512)
    mu, log_sigma = enc(obs, ctx=ctx)
    assert mu.shape == (8, 256)

def test_encoder_kl_nonnegative():
    # KL divergence must be >= 0
    mu, log_sigma = enc(torch.randn(32, 4096))
    kl = -0.5 * torch.sum(1 + 2*log_sigma - mu**2 - (2*log_sigma).exp(), dim=-1)
    assert (kl >= -1e-6).all()
```

---

## B) Likelihood Model P(o|s)

### Purpose

Decode latent states back into observation space. The likelihood model serves two roles in the EFE computation: (1) computing ambiguity (epistemic term) by measuring how uncertain predicted observations are, and (2) computing pragmatic value by comparing predicted observations against preferences.

### Input/Output Specification

| Argument | Shape | Description |
|---|---|---|
| `state` | `(B, state_dim)` | Latent state, either sampled or mean from encoder |

| Output | Shape | Description |
|---|---|---|
| `obs_mu` | `(B, obs_dim)` | Predicted observation mean |
| `obs_log_var` | `(B, obs_dim)` | Predicted observation log-variance (diagonal Gaussian) |

For binary/Bernoulli observations (e.g., per-pixel image logits), output raw logits `(B, obs_dim)` and use `F.binary_cross_entropy_with_logits` as the reconstruction loss.

### Architecture

Mirror the encoder architecture. Use symmetric depth and width:

```python
class LikelihoodDecoder(nn.Module):
    def __init__(self, state_dim, obs_dim, hidden_dim=512, num_layers=3,
                 output_type="gaussian"):
        super().__init__()
        self.output_type = output_type

        layers = []
        for i in range(num_layers):
            in_d = state_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
        self.backbone = nn.Sequential(*layers)

        if output_type == "gaussian":
            self.mu_head = nn.Linear(hidden_dim, obs_dim)
            self.log_var_head = nn.Linear(hidden_dim, obs_dim)
        elif output_type == "bernoulli":
            self.logit_head = nn.Linear(hidden_dim, obs_dim)
```

### Training Loss

| Output Type | Loss Function | Formula |
|---|---|---|
| Gaussian | MSE (equivalent to Gaussian NLL with fixed variance) | `0.5 * (obs - obs_mu)^2` summed over `obs_dim` |
| Gaussian (learned var) | Gaussian NLL | `0.5 * (log_var + (obs - obs_mu)^2 / exp(log_var))` |
| Bernoulli | BCE with logits | `F.binary_cross_entropy_with_logits(logits, obs)` |

Prefer learned variance (Gaussian NLL) for production. The decoder variance feeds directly into the epistemic term of EFE, so a fixed-variance assumption masks useful uncertainty information.

### Testing Contract

```python
def test_likelihood_reconstruction():
    dec = LikelihoodDecoder(state_dim=256, obs_dim=4096)
    state = torch.randn(8, 256)
    obs_mu, obs_log_var = dec(state)
    assert obs_mu.shape == (8, 4096)
    assert obs_log_var.shape == (8, 4096)

def test_likelihood_roundtrip():
    # Encode then decode; reconstruction loss should decrease with training
    enc = LatentEncoder(obs_dim=4096, state_dim=256)
    dec = LikelihoodDecoder(state_dim=256, obs_dim=4096)
    obs = torch.randn(8, 4096)
    mu, log_sigma = enc(obs)
    s = mu + torch.exp(log_sigma) * torch.randn_like(log_sigma)
    obs_recon, _ = dec(s)
    loss = F.mse_loss(obs_recon, obs)
    assert loss.requires_grad
```

---

## C) Transition Model P(s'|s,a)

### Purpose

Predict the distribution over next latent states given the current state and an action. This is the core dynamics model used by the rollout engine for latent imagination. Uncertainty estimation from the transition model feeds the epistemic term of EFE.

### Input/Output Specification

| Argument | Shape | Description |
|---|---|---|
| `state` | `(B, state_dim)` | Current latent state |
| `action` | `(B, action_dim)` continuous or `(B,)` discrete | Action taken |

| Output | Shape | Description |
|---|---|---|
| `next_mu` | `(B, state_dim)` | Predicted next-state mean |
| `next_log_var` | `(B, state_dim)` | Predicted next-state log-variance |

For ensemble mode, each member outputs its own `(next_mu, next_log_var)`. Aggregate as described below.

### Action Encoding

| Action Type | Encoding Method |
|---|---|
| Discrete | Learned embedding: `action_embed = self.action_embedding(action_idx)` with `nn.Embedding(num_actions, action_embed_dim)` |
| Continuous | Direct concatenation: `torch.cat([state, action], dim=-1)` |
| One-hot (legacy) | Direct concatenation, same as continuous |

Prefer learned embeddings over one-hot for discrete actions when `action_dim > 32`, as one-hot becomes sparse and wasteful.

### Ensemble Approach (Recommended)

Maintain `K` independent transition models (default `K=5`). Each model has its own parameters and is trained on bootstrapped subsets of the data.

```python
class TransitionEnsemble(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512,
                 ensemble_size=5, action_type="continuous"):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.models = nn.ModuleList([
            TransitionMember(state_dim, action_dim, hidden_dim, action_type)
            for _ in range(ensemble_size)
        ])

    def forward(self, state, action):
        """Returns all ensemble member predictions."""
        predictions = [m(state, action) for m in self.models]
        mus = torch.stack([p[0] for p in predictions])      # (K, B, state_dim)
        log_vars = torch.stack([p[1] for p in predictions]) # (K, B, state_dim)
        return mus, log_vars

    def predict_with_uncertainty(self, state, action):
        """Aggregate ensemble predictions with uncertainty decomposition."""
        mus, log_vars = self.forward(state, action)
        vars_ = torch.exp(log_vars)

        # Ensemble mean prediction
        mean_mu = mus.mean(dim=0)                    # (B, state_dim)

        # Aleatoric uncertainty: mean of individual variances
        aleatoric = vars_.mean(dim=0)                # (B, state_dim)

        # Epistemic uncertainty: variance of means across ensemble
        epistemic = mus.var(dim=0)                   # (B, state_dim)

        # Total predictive variance
        total_var = aleatoric + epistemic            # (B, state_dim)

        return mean_mu, total_var, aleatoric, epistemic
```

**Uncertainty decomposition:**

| Uncertainty Type | Computation | Interpretation |
|---|---|---|
| Aleatoric | Mean of each member's predicted variance | Irreducible noise in dynamics |
| Epistemic | Variance of member means | Model uncertainty (reducible with more data) |
| Total | Aleatoric + Epistemic | Full predictive uncertainty |

The epistemic uncertainty feeds the epistemic value term in EFE. High epistemic uncertainty in a region of state-action space indicates the agent should explore there.

### Single Stochastic Model (Alternative)

When compute budget is constrained (e.g., real-time control), use a single model predicting `(mean, log_var)`:

```python
class TransitionMember(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512,
                 action_type="continuous"):
        super().__init__()
        if action_type == "discrete":
            self.action_embed = nn.Embedding(action_dim, hidden_dim // 4)
            input_dim = state_dim + hidden_dim // 4
        else:
            input_dim = state_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, state_dim)
        self.log_var_head = nn.Linear(hidden_dim, state_dim)

        # Small initialization for variance head to start near deterministic
        nn.init.zeros_(self.log_var_head.weight)
        nn.init.constant_(self.log_var_head.bias, -2.0)

    def forward(self, state, action):
        if hasattr(self, 'action_embed') and action.dim() == 1:
            action = self.action_embed(action.long())
        h = self.net(torch.cat([state, action], dim=-1))
        mu = self.mu_head(h)
        log_var = self.log_var_head(h)
        log_var = torch.clamp(log_var, min=-10.0, max=2.0)
        return mu, log_var
```

### Stability Requirements

| Requirement | Implementation | Reason |
|---|---|---|
| Log-variance clamping | `torch.clamp(log_var, -10.0, 2.0)` | Prevents variance from collapsing to zero (`exp(-10) ~ 5e-5`) or exploding (`exp(2) ~ 7.4`) |
| Variance head init | `nn.init.zeros_(weight)`, `bias=-2.0` | Start near-deterministic, let the model learn to increase variance |
| Gradient clipping | `torch.nn.utils.clip_grad_norm_(transition.parameters(), max_norm=1.0)` | Prevent gradient explosions during early training |
| Residual prediction | Predict `delta_s = s' - s` instead of absolute `s'` | Easier to learn; identity mapping is the zero-init default |

Implement residual prediction by modifying the forward method:

```python
def forward(self, state, action):
    # ... compute mu, log_var as above ...
    mu = state + mu  # Residual: predict delta, add to current state
    return mu, log_var
```

### Testing Contract

```python
def test_transition_shapes():
    trans = TransitionEnsemble(state_dim=256, action_dim=128, ensemble_size=5)
    state = torch.randn(8, 256)
    action = torch.randn(8, 128)
    mean_mu, total_var, aleatoric, epistemic = trans.predict_with_uncertainty(state, action)
    assert mean_mu.shape == (8, 256)
    assert total_var.shape == (8, 256)
    assert (total_var >= 0).all()

def test_transition_ensemble_disagreement():
    # Untrained ensemble should show epistemic uncertainty
    trans = TransitionEnsemble(state_dim=256, action_dim=128, ensemble_size=5)
    state = torch.randn(8, 256)
    action = torch.randn(8, 128)
    _, _, _, epistemic = trans.predict_with_uncertainty(state, action)
    assert epistemic.sum() > 0, "Untrained ensemble must show nonzero epistemic uncertainty"

def test_transition_log_var_clamped():
    member = TransitionMember(state_dim=256, action_dim=128)
    state = torch.randn(8, 256) * 100  # Large input to stress test
    action = torch.randn(8, 128) * 100
    _, log_var = member(state, action)
    assert log_var.min() >= -10.0
    assert log_var.max() <= 2.0
```

---

## D) Preferences (C in Classic Active Inference)

### Purpose

Define the target observation distribution that the agent seeks to realize. Preferences encode "what good outcomes look like" and drive the pragmatic term of EFE. Support three modes to accommodate different training scenarios.

### Preference Modes

#### Mode 1: Fixed Preferences

Handcrafted target distribution over observations. Use for tasks with known goal states (e.g., reaching a target position, maintaining a setpoint).

```python
class FixedPreferences(nn.Module):
    def __init__(self, obs_dim, num_goals=1):
        super().__init__()
        self.register_buffer('pref_mu', torch.zeros(num_goals, obs_dim))
        self.register_buffer('pref_log_var', torch.zeros(num_goals, obs_dim))
        self.register_buffer('goal_weights', torch.ones(num_goals) / num_goals)

    def set_goal(self, goal_idx, mu, log_var=None):
        """Set a fixed goal. No gradient updates."""
        self.pref_mu[goal_idx] = mu
        if log_var is not None:
            self.pref_log_var[goal_idx] = log_var

    def log_prob(self, obs):
        """Log probability of observation under preference distribution."""
        weights = self.goal_weights  # (G,)
        # Gaussian log prob per goal (precision = exp(-log_var))
        diff = obs.unsqueeze(1) - self.pref_mu.unsqueeze(0)  # (B, G, obs_dim)
        pref_precision = torch.exp(-self.pref_log_var).unsqueeze(0)
        log_p = -0.5 * (diff.pow(2) * pref_precision).sum(-1)  # (B, G)
        # Weighted mixture
        log_p = torch.logsumexp(log_p + weights.log().unsqueeze(0), dim=1)  # (B,)
        return log_p
```

No parameters require gradients. Store as buffers. Serialize via `state_dict()`.

#### Mode 2: Learned from Reward

Map external reward signals to a preferred observation distribution. Bridges active inference to standard RL reward signals. Use when the task provides scalar rewards but no explicit goal state.

```python
class RewardDerivedPreferences(nn.Module):
    def __init__(self, obs_dim, hidden_dim=512):
        super().__init__()
        # Maps (obs, reward) -> preference log-probability
        self.reward_encoder = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.pref_mu = nn.Linear(hidden_dim, obs_dim)
        self.pref_log_var = nn.Linear(hidden_dim, obs_dim)

    def update(self, obs, reward):
        """Update preference model from (observation, reward) pairs."""
        x = torch.cat([obs, reward.unsqueeze(-1)], dim=-1)
        h = self.reward_encoder(x)
        mu = self.pref_mu(h)
        log_var = self.pref_log_var(h)
        return mu, log_var

    def log_prob(self, obs, reward_context=None):
        """Evaluate log-probability of obs under current preferences."""
        if reward_context is not None:
            mu, log_var = self.update(obs, reward_context)
        else:
            mu, log_var = self._cached_mu, self._cached_log_var
        diff = obs - mu
        return -0.5 * (log_var + diff.pow(2) / torch.exp(log_var)).sum(-1)
```

Train end-to-end: observations that co-occur with high reward shape the preference distribution to assign high probability to similar observations.

#### Mode 3: Learned from Demonstrations

Train a discriminator to classify "preferred" vs. "non-preferred" observation trajectories. Use when expert demonstrations are available but reward functions are not.

```python
class DemonstrationPreferences(nn.Module):
    def __init__(self, obs_dim, hidden_dim=512):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),  # Binary: preferred / not-preferred
        )

    def log_prob(self, obs):
        """Log probability that obs is preferred (sigmoid output)."""
        logit = self.discriminator(obs).squeeze(-1)
        return F.logsigmoid(logit)  # (B,)

    def train_step(self, preferred_obs, non_preferred_obs):
        """Binary cross-entropy training step."""
        pos_logit = self.discriminator(preferred_obs)
        neg_logit = self.discriminator(non_preferred_obs)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_logit, neg_logit]),
            torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])
        )
        return loss
```

### Unified Preferences Interface

Wrap all three modes behind a common interface:

```python
class Preferences(nn.Module):
    def __init__(self, obs_dim, mode="fixed", hidden_dim=512, num_goals=1):
        super().__init__()
        self.mode = mode
        if mode == "fixed":
            self.backend = FixedPreferences(obs_dim, num_goals)
        elif mode == "reward_derived":
            self.backend = RewardDerivedPreferences(obs_dim, hidden_dim)
        elif mode == "demonstration":
            self.backend = DemonstrationPreferences(obs_dim, hidden_dim)

    def log_prob(self, obs, **kwargs):
        return self.backend.log_prob(obs, **kwargs)

    def save_preferences(self, path):
        torch.save(self.backend.state_dict(), path)

    def load_preferences(self, path):
        self.backend.load_state_dict(torch.load(path, weights_only=True))
```

### Serialization

Preferences must be independently saveable and loadable. This allows:
- Swapping preference modules without retraining the world model
- Sharing preferences across agents
- Versioning preference snapshots during curriculum learning

### Logging

Log per step:
- `preference_log_prob`: mean `log_prob(predicted_obs)` across batch, indicates preference satisfaction
- `preference_mode`: active mode string
- `preference_kl` (if reward-derived): KL between current preference distribution and prior

### Testing Contract

```python
def test_fixed_preferences_log_prob():
    pref = FixedPreferences(obs_dim=4096)
    pref.set_goal(0, mu=torch.zeros(4096), precision=torch.ones(4096))
    obs = torch.randn(8, 4096)
    lp = pref.log_prob(obs)
    assert lp.shape == (8,)
    # Observation at the goal should have highest log prob
    goal_lp = pref.log_prob(torch.zeros(1, 4096))
    assert goal_lp.item() > lp.mean().item()

def test_preferences_serialization():
    pref = Preferences(obs_dim=4096, mode="fixed")
    pref.save_preferences("/tmp/test_pref.pt")
    pref2 = Preferences(obs_dim=4096, mode="fixed")
    pref2.load_preferences("/tmp/test_pref.pt")
    # State dicts must match
    for k in pref.backend.state_dict():
        assert torch.equal(pref.backend.state_dict()[k], pref2.backend.state_dict()[k])
```

---

## Sampling Strategies

### Reparameterization Trick (Continuous)

Used by the latent encoder and transition model. Enables gradient flow through stochastic sampling.

```python
def reparameterize(mu, log_sigma):
    """Sample from N(mu, sigma^2) with gradient flow."""
    std = torch.exp(log_sigma)
    eps = torch.randn_like(std)
    return mu + std * eps
```

Always sample in `fp32`. Convert back to `fp16` only after sampling if using mixed precision for storage.

### Gumbel-Softmax (Discrete)

Used for discrete latent states. Temperature annealing schedule:

```python
def gumbel_softmax_sample(logits, temperature, hard=False):
    """Differentiable discrete sampling."""
    return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)

# Temperature schedule: anneal from 1.0 to 0.1 over training
temperature = max(0.1, 1.0 - step / anneal_steps * 0.9)
```

| Phase | Temperature | Behavior |
|---|---|---|
| Early training | 1.0 | Soft, high entropy, exploration |
| Mid training | 0.5 | Moderate sharpening |
| Late training | 0.1 | Near-discrete, exploitation |

### Straight-Through Estimator (Alternative Discrete)

Use when Gumbel-softmax temperature tuning is problematic:

```python
def straight_through(logits):
    """Hard sample forward, soft gradient backward."""
    hard = F.one_hot(logits.argmax(dim=-1), logits.size(-1)).float()
    soft = F.softmax(logits, dim=-1)
    return hard - soft.detach() + soft  # Gradient flows through soft
```

### Mixed Precision Policy

| Operation | Precision | Reason |
|---|---|---|
| Sampling (reparam, Gumbel) | `fp32` | Numerical stability of log/exp operations |
| Backbone forward pass | `fp16` or `bf16` | Speed and memory; autocast handles this |
| KL divergence computation | `fp32` | Accumulation errors in summed KL terms |
| EFE computation | `fp32` | Sum invariant requires full precision |
| Stored embeddings/buffers | `fp16` | Memory savings for large batch rollouts |

---

## Integration Points

### Workspace to Encoder

The global workspace produces a unified `(B, 4096)` representation regardless of input modality. This is the `obs_dim` input to the latent encoder.

```python
# In the forward pass of BrainAI (system.py)
workspace_output = self.workspace(x)            # (B, 4096)
state_params, state_sample, agent_state = self.active_inference.infer_state(
    o_t=workspace_output,
    ctx=workspace_slots,  # Optional (B, K, D) or None
    state=prev_agent_state,
)
```

### Context Sources

| Context Source | Shape | Content |
|---|---|---|
| Workspace slots | `(B, K, D)` where K=num_slots, D=slot_dim | Competing representations from global workspace |
| Working memory | `(B, wm_dim)` | Maintained state from previous steps |
| Neuromodulatory signals | `(B, 4)` | DA, ACh, NE, 5-HT levels from meta-learning module |

Pool workspace slots to `(B, D)` via mean pooling, max pooling, or learned attention before feeding as context. Select pooling strategy via `ctx_pool_mode` in config.

### Transition Model to Rollout Engine

The rollout engine calls the transition model repeatedly for multi-step latent imagination:

```python
# Rollout engine pseudocode
state = encoder.sample(obs)                     # Initial state from observation
for t in range(horizon):
    action = planner.propose_action(state)      # From CEM or random shooting
    next_mu, total_var, _, epistemic = transition.predict_with_uncertainty(state, action)
    state = reparameterize(next_mu, 0.5 * torch.log(total_var))
    pred_obs_mu, pred_obs_log_var = likelihood(state)
    efe_t = compute_efe_step(pred_obs_mu, pred_obs_log_var, epistemic, preferences)
    total_efe += efe_t
```

### Likelihood Model to EFE

The likelihood model connects to both the pragmatic and epistemic terms:

- **Pragmatic term**: `preferences.log_prob(likelihood.decode(imagined_state))`
- **Epistemic term (ambiguity)**: entropy of `P(o|s)`, computed from predicted `obs_log_var`

---

## Configuration

```python
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class GenerativeModelConfig:
    """Configuration for the generative model stack."""

    # Dimensions
    obs_dim: int = 4096
    state_dim: int = 256
    action_dim: int = 128
    hidden_dim: int = 512
    ctx_dim: int = 0           # 0 means no context input

    # Latent type
    latent_type: str = "continuous"       # "continuous" or "discrete"
    num_discrete_states: int = 32         # Only used if latent_type == "discrete"

    # Architecture depths
    encoder_layers: int = 3
    decoder_layers: int = 3
    transition_layers: int = 2

    # Transition model
    transition_ensemble_size: int = 5
    use_residual_transition: bool = True
    action_type: str = "continuous"       # "continuous" or "discrete"

    # Stability
    log_var_clamp: Tuple[float, float] = (-10.0, 2.0)
    gradient_clip_norm: float = 1.0

    # Training
    kl_weight: float = 1.0
    kl_anneal_steps: int = 1000           # Steps to anneal kl_weight from 0 to target

    # Context conditioning
    ctx_mode: str = "concat"              # "concat", "cross_attention", "film"
    ctx_pool_mode: str = "mean"           # "mean", "max", "attention"

    # Preferences
    preference_mode: str = "fixed"        # "fixed", "reward_derived", "demonstration"
    num_goals: int = 1

    # Discrete sampling
    gumbel_temperature_init: float = 1.0
    gumbel_temperature_min: float = 0.1
    gumbel_anneal_steps: int = 10000

    # Likelihood output
    likelihood_output_type: str = "gaussian"  # "gaussian" or "bernoulli"
```

### Scale Presets

| Preset | state_dim | hidden_dim | ensemble_size | encoder_layers | Approx Params |
|---|---|---|---|---|---|
| `minimal()` | 32 | 64 | 1 | 2 | ~200K |
| `dev()` | 128 | 256 | 3 | 3 | ~5M |
| `production_1b()` | 256 | 512 | 5 | 3 | ~25M |
| `production_3b()` | 512 | 1024 | 5 | 4 | ~80M |
| `production_7b()` | 512 | 1024 | 7 | 4 | ~120M |

---

## Common Failure Modes

| Symptom | Component | Cause | Fix |
|---|---|---|---|
| Latent space collapses to point | Encoder | KL weight too high early in training | Anneal `kl_weight` from 0 to target over `kl_anneal_steps` |
| Posterior ignores observation | Encoder | Decoder too powerful (information bypass) | Reduce decoder capacity or add dropout to decoder |
| Reconstruction loss plateaus high | Likelihood | `state_dim` too small to capture observation structure | Increase `state_dim`; verify encoder gradient flow |
| Predicted variance is always zero | Transition | `log_var_head` stuck at clamp minimum | Check `log_var_clamp` lower bound; increase initialization bias; verify gradient reaches variance head |
| Predicted variance explodes | Transition | No log-variance clamping | Add `torch.clamp(log_var, -10.0, 2.0)` |
| Ensemble members converge to same prediction | Transition | No bootstrap or data shuffling | Train each member on random 80% subset of data; use different random seeds |
| NaN in sampling | All | `log_var` or `log_sigma` unbounded causing `exp()` overflow | Clamp before exponentiation; use `fp32` for sampling |
| Gumbel-softmax produces uniform samples | Encoder | Temperature too high | Verify temperature annealing schedule; check that logits have reasonable magnitude |
| Preferences dominate EFE | Preferences | Preference log-prob scale much larger than epistemic term | Normalize each EFE term independently before weighting; log all term magnitudes |
| Preference model ignores reward signal | Preferences | Reward-derived model underfitting | Increase hidden_dim; check reward normalization; verify reward range |
| State transition predicts input unchanged | Transition | Not using residual prediction; model learns identity | Enable `use_residual_transition`; this makes the zero-output default the identity |
| Context breaks batch dimension | Encoder | Context shape `(B, K, D)` not pooled before concat | Apply `ctx_pool_mode` before feeding to encoder backbone |
| Mixed precision NaN | All | Sampling in `fp16` | Force `fp32` for all sampling operations; use `torch.autocast` context manager boundaries |

---

## Migration from Existing Code

The current implementation in `brain_ai/decision/active_inference.py` has three classes that map to the new architecture as follows:

### Class Mapping

| Existing Class | New Class(es) | Changes Required |
|---|---|---|
| `StateEncoder` | `LatentEncoder` | Add LayerNorm, SiLU activation (replace ReLU), residual connections, optional context input, discrete latent support |
| `GenerativeModel` | `LikelihoodDecoder` + `TransitionEnsemble` | Split into two independent modules; add ensemble support to transition; add learned variance to likelihood; add log-var clamping |
| `Preferences` | `Preferences` (unified interface) | Wrap in mode-based interface; add `log_prob` method; add serialization; add reward-derived and demonstration modes |

### Step-by-Step Migration

**Step 1: Create `brain_ai/decision/generative_model.py`**

Implement `LatentEncoder`, `LikelihoodDecoder`, `TransitionMember`, `TransitionEnsemble`, `FixedPreferences`, `RewardDerivedPreferences`, `DemonstrationPreferences`, and `Preferences` as described in this document.

**Step 2: Update `ActiveInferenceConfig`**

Replace the existing flat config with `GenerativeModelConfig` nested inside `ActiveInferenceConfig`:

```python
# Before (existing)
@dataclass
class ActiveInferenceConfig:
    obs_dim: int = 512
    state_dim: int = 64
    action_dim: int = 10
    hidden_dim: int = 256
    # ...

# After (new)
@dataclass
class ActiveInferenceConfig:
    generative: GenerativeModelConfig = field(default_factory=GenerativeModelConfig)
    planning_horizon: int = 8
    num_rollouts: int = 128
    # EFE, planner, etc. configs...
```

**Step 3: Update `ActiveInferenceAgent.__init__`**

Replace direct class instantiation with new module construction:

```python
# Before
self.encoder = StateEncoder(obs_dim, state_dim, hidden_dim)
self.generative = GenerativeModel(obs_dim, state_dim, action_dim, hidden_dim)
self.preferences = Preferences(obs_dim, learnable=learn_preferences)

# After
from .generative_model import LatentEncoder, LikelihoodDecoder, TransitionEnsemble, Preferences
cfg = self.config.generative
self.encoder = LatentEncoder(cfg.obs_dim, cfg.state_dim, cfg.hidden_dim,
                              cfg.encoder_layers, cfg.ctx_dim, cfg.ctx_mode,
                              cfg.latent_type, cfg.num_discrete_states)
self.likelihood = LikelihoodDecoder(cfg.state_dim, cfg.obs_dim, cfg.hidden_dim,
                                     cfg.decoder_layers, cfg.likelihood_output_type)
self.transition = TransitionEnsemble(cfg.state_dim, cfg.action_dim, cfg.hidden_dim,
                                      cfg.transition_ensemble_size, cfg.action_type)
self.preferences = Preferences(cfg.obs_dim, cfg.preference_mode, cfg.hidden_dim, cfg.num_goals)
```

**Step 4: Update `compute_efe`**

Replace the monolithic EFE computation with calls to the decomposed components:

```python
# Before: single GenerativeModel.predict_obs and predict_next_state
# After: separate likelihood.forward and transition.predict_with_uncertainty

next_mu, total_var, aleatoric, epistemic = self.transition.predict_with_uncertainty(state, action)
next_state = reparameterize(next_mu, 0.5 * torch.log(total_var))
obs_mu, obs_log_var = self.likelihood(next_state)
pragmatic = self.preferences.log_prob(obs_mu)
epistemic_value = epistemic.sum(dim=-1)  # Scalar per batch element
```

**Step 5: Update `system.py` integration**

Change the `obs_dim` default from `512` to `4096` to match workspace output dimension. Update the factory function:

```python
# In system.py _build_decision
self.active_inference = create_active_inference_agent(
    obs_dim=4096,       # was 512
    state_dim=256,      # was 64
    action_dim=self.config.decision.num_classes,
    # ...
)
```

**Step 6: Verify with existing tests**

Run all existing tests to confirm backward compatibility:

```bash
python -m pytest tests/test_active_inference.py -v
```

Add new tests for each component independently (see Testing Contract sections above).

### Backward Compatibility

Maintain the existing `StateEncoder`, `GenerativeModel`, and `Preferences` classes in `active_inference.py` with deprecation warnings. Mark them for removal in version 0.3.0. New code must import from `generative_model.py`.

```python
# In active_inference.py (temporary compatibility shim)
import warnings

class StateEncoder(LatentEncoder):
    def __init__(self, *args, **kwargs):
        warnings.warn("StateEncoder is deprecated; use LatentEncoder from generative_model.py",
                       DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)
```
