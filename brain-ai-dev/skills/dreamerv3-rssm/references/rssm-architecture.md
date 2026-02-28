# RSSM Architecture Reference

## Overview

The Recurrent State Space Model (RSSM) from DreamerV3 (Hafner et al., 2023) is a sequence model that
maintains a latent world state composed of two parts: a deterministic recurrent state and a stochastic
categorical state. The separation enables the model to simultaneously capture precise temporal
sequences (via the GRU) and represent multimodal uncertainty (via the categorical distribution).

This document provides the complete mathematical specification and implementation guidance for all
RSSM components.

---

## State Decomposition

The latent state at timestep `t` consists of:

| Symbol | Name | Shape | Role |
|--------|------|-------|------|
| `h_t` | Deterministic state | `(batch, deter_dim)` | GRU hidden state; carries sequential memory |
| `z_t` | Stochastic state | `(batch, stoch_dim, num_classes)` | Categorical sample; captures multimodal uncertainty |
| `f_t` | Feature vector | `(batch, deter_dim + stoch_dim * num_classes)` | Input to all heads |

The feature vector is formed by flattening and concatenating:

```
f_t = concat(h_t, flatten(z_t))
    = concat(h_t, z_t.reshape(batch, stoch_dim * num_classes))
```

For the default configuration (deter_dim=1024, stoch_dim=32, num_classes=32):

```
f_t shape: (batch, 1024 + 32*32) = (batch, 2048)
```

---

## Block GRU

### Motivation

The standard GRUCell applies the input directly to linear gate layers. The Block GRU replaces this
with a normalized, nonlinear projection step, which stabilizes training for larger hidden dimensions
and improves gradient flow through long sequences.

### Input Projection

Given raw input `x` (the concatenation of the previous stochastic state and the previous action):

```
x_proj = SiLU(LayerNorm(Linear(input_dim -> hidden_dim)(x)))
```

Where:
- `Linear`: maps from `(stoch_dim * num_classes + action_dim)` to `hidden_dim`
- `LayerNorm`: normalizes across the hidden dimension, stabilizing activations
- `SiLU`: smooth gating activation `x * sigmoid(x)`, superior to ReLU for sequence models

### GRU Gate Equations

After the input projection, standard GRU gate mechanics apply, using `x_proj` as the input:

```
r_t = sigmoid(W_r @ concat(x_proj, h_{t-1}) + b_r)    # reset gate
z_t = sigmoid(W_z @ concat(x_proj, h_{t-1}) + b_z)    # update gate
n_t = tanh(W_n @ concat(x_proj, r_t * h_{t-1}) + b_n) # candidate hidden
h_t = (1 - z_t) * h_{t-1} + z_t * n_t                 # new hidden state
```

Where:
- `r_t` (reset gate): controls how much past hidden state contributes to the candidate
- `z_t` (update gate): interpolates between old and new hidden state (avoid confusion with `z_t`
  stochastic state; the gate uses subscript notation only in this section)
- `n_t` (candidate): proposed new hidden state values
- `h_t`: updated deterministic state

### RMSNorm Output

The GRU output is normalized before use:

```
h_t_out = RMSNorm(hidden_dim)(h_t)
```

RMSNorm (root mean square normalization) divides by the RMS of the hidden vector:

```
RMSNorm(h) = h / sqrt(mean(h^2) + epsilon) * gamma
```

Where `gamma` is a learned scale parameter. Unlike LayerNorm, RMSNorm omits the mean subtraction,
making it computationally cheaper while preserving normalization benefits. Use `nn.RMSNorm` from
PyTorch 2.x.

The normalized output `h_t_out` is what gets passed to:
- The prior network
- The posterior network
- All downstream heads (after concatenation with `z_t`)

### PyTorch Implementation Skeleton

```python
class BlockGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1024):
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        # Reset, update, candidate gates
        # Each takes concat(x_proj, h) as input -> hidden_dim output
        self.gate_r = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_z = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_n = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm_out = nn.RMSNorm(hidden_dim)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        x_proj = self.input_proj(x)
        xh = torch.cat([x_proj, h], dim=-1)
        r = torch.sigmoid(self.gate_r(xh))
        z = torch.sigmoid(self.gate_z(xh))
        xrh = torch.cat([x_proj, r * h], dim=-1)
        n = torch.tanh(self.gate_n(xrh))
        h_new = (1 - z) * h + z * n
        return self.norm_out(h_new)
```

---

## Prior Network

### Role

The prior network predicts the distribution over `z_t` given only the deterministic state `h_t`.
It represents what the model expects the world state to be, without observing the actual current
observation. During imagination rollouts, only the prior is used.

### Architecture

A 2-layer MLP:

```
prior_net: Linear(deter_dim, hidden_dim)
        -> LayerNorm(hidden_dim)
        -> SiLU
        -> Linear(hidden_dim, hidden_dim)
        -> LayerNorm(hidden_dim)
        -> SiLU
        -> Linear(hidden_dim, stoch_dim * num_classes)
```

Input dimension: `deter_dim` (e.g., 1024)
Output dimension: `stoch_dim * num_classes` (e.g., 32 * 32 = 1024)

The output is reshaped to `(batch, stoch_dim, num_classes)` to represent `stoch_dim` independent
categorical distributions, each over `num_classes` classes.

### Forward Pass

```python
def prior(self, h: Tensor) -> Tensor:
    # h: (batch, deter_dim)
    logits = self.prior_net(h)                        # (batch, stoch_dim * num_classes)
    return logits.view(-1, self.stoch_dim, self.num_classes)  # (batch, stoch_dim, num_classes)
```

---

## Posterior Network

### Role

The posterior network predicts the distribution over `z_t` given both the deterministic state
`h_t` and the observation embedding `embed_t`. It represents what the model believes the world
state is after seeing the actual observation. This is richer than the prior because it has access
to real sensory data.

The KL divergence between posterior and prior drives learning:
- Prior learns to be more accurate (dynamics loss)
- Posterior learns to be consistent with the prior while encoding observations (representation loss)

### Architecture

Same 2-layer MLP structure as the prior, but with a larger input:

```
posterior_net: Linear(deter_dim + embed_dim, hidden_dim)
            -> LayerNorm(hidden_dim)
            -> SiLU
            -> Linear(hidden_dim, hidden_dim)
            -> LayerNorm(hidden_dim)
            -> SiLU
            -> Linear(hidden_dim, stoch_dim * num_classes)
```

Input dimension: `deter_dim + embed_dim` (e.g., 1024 + 1536 = 2560 for a CNN encoder)

### Forward Pass

```python
def posterior(self, h: Tensor, embed: Tensor) -> Tensor:
    # h: (batch, deter_dim), embed: (batch, embed_dim)
    inp = torch.cat([h, embed], dim=-1)
    logits = self.posterior_net(inp)
    return logits.view(-1, self.stoch_dim, self.num_classes)
```

---

## Categorical State with Unimix

### Unimix Distribution

Raw logits are converted to probabilities using the unimix formula:

```
p(class_k) = (1 - unimix) * softmax(logits)_k + unimix * (1 / num_classes)
```

With default `unimix = 0.01`:
- 99% of the probability mass comes from the learned distribution
- 1% is spread uniformly across all classes

This prevents any class from having zero probability, which would cause:
1. Log-probability of negative infinity when computing KL divergence
2. Gradient death in the categorical sampling path
3. Codebook collapse where some classes are never used

### Straight-Through Estimator

Sampling from a discrete categorical distribution is non-differentiable. The straight-through
estimator enables gradients to flow:

```python
# Sample: non-differentiable
indices = torch.argmax(logits, dim=-1)  # (batch, stoch_dim)
z_hard = F.one_hot(indices, num_classes).float()  # (batch, stoch_dim, num_classes)

# Soft probabilities (differentiable)
probs = unimix_probs(logits)  # (batch, stoch_dim, num_classes)

# Straight-through: gradients flow through probs, not z_hard
z = z_hard - probs.detach() + probs
```

This means:
- Forward pass uses `z_hard` (actual one-hot sample)
- Backward pass uses gradients of `probs` (smooth, differentiable)
- The gradient of the loss w.r.t. `logits` flows through `probs`

---

## RSSM Observe Step

The observe step processes one timestep of an observed sequence:

```
Input:  embed_t  (batch, embed_dim)    — observation embedding at time t
        action_{t-1}  (batch, action_dim) — action taken at t-1
        state_{t-1}   RSSMState         — previous state (h_{t-1}, z_{t-1})

Step 1: Form GRU input
        gru_in = concat(flatten(z_{t-1}), action_{t-1})
        # shape: (batch, stoch_dim * num_classes + action_dim)

Step 2: Update deterministic state
        h_t = BlockGRU(gru_in, h_{t-1})
        # shape: (batch, deter_dim)

Step 3: Compute prior distribution
        prior_logits = prior_net(h_t)
        # shape: (batch, stoch_dim, num_classes)

Step 4: Compute posterior distribution
        posterior_logits = posterior_net(concat(h_t, embed_t))
        # shape: (batch, stoch_dim, num_classes)

Step 5: Sample posterior state
        z_t = straight_through_sample(posterior_logits)
        # shape: (batch, stoch_dim, num_classes)

Output: posterior_state = RSSMState(h_t, z_t, posterior_logits)
        prior_logits    = prior_logits  (for KL computation)
```

---

## RSSM Observe (Sequence)

For a full sequence of length T:

```python
def observe(self, embed_seq, action_seq, state):
    # embed_seq:  (T, batch, embed_dim)
    # action_seq: (T, batch, action_dim)
    posteriors, priors = [], []
    for t in range(T):
        state, prior_logits = self.observe_step(
            embed_seq[t], action_seq[t], state
        )
        posteriors.append(state)
        priors.append(prior_logits)
    return stack(posteriors), stack(priors)
```

The returned posteriors and priors are used to compute the KL balancing loss.

---

## RSSM Imagine Step

The imagine step advances the state using only the prior (no observation):

```
Input:  state_t   RSSMState     — current state
        action_t  Tensor        — action to take

Step 1: Form GRU input
        gru_in = concat(flatten(z_t), action_t)

Step 2: Update deterministic state
        h_{t+1} = BlockGRU(gru_in, h_t)

Step 3: Sample from prior
        prior_logits = prior_net(h_{t+1})
        z_{t+1} = straight_through_sample(prior_logits)

Step 4: Form new state
        state_{t+1} = RSSMState(h_{t+1}, z_{t+1}, prior_logits)
```

---

## RSSM Imagine (Full Rollout)

```python
def imagine(self, policy, state, horizon):
    features, actions, reward_logits, cont_logits = [], [], [], []
    for t in range(horizon):
        feat = get_features(state)           # concat(h, z_flat)
        action = policy(feat)                # callable, returns (batch, action_dim)
        state = self.imagine_step(state, action)
        features.append(feat)
        actions.append(action)
        reward_logits.append(self.reward_head(get_features(state)))
        cont_logits.append(self.cont_head(get_features(state)))
    return ImaginedTrajectory(
        features=stack(features),        # (horizon, batch, feature_dim)
        actions=stack(actions),          # (horizon, batch, action_dim)
        reward_logits=stack(reward_logits),  # (horizon, batch, num_bins)
        continue_logits=stack(cont_logits),  # (horizon, batch, 1)
    )
```

---

## Model Size Table

DreamerV3 provides a set of standard model sizes. All sizes use `stoch_dim=32` (32 categorical
distributions) and `unimix=0.01`. The `num_classes` and `deter_dim` scale together.

| Size | Parameters | `deter_dim` | `hidden_dim` | `num_classes` | Feature Dim |
|------|-----------|-------------|--------------|----------------|-------------|
| XS (12M) | ~12M | 512 | 256 | 16 | 512 + 32*16 = 1024 |
| S (25M) | ~25M | 1024 | 384 | 24 | 1024 + 32*24 = 1792 |
| M (50M) | ~50M | 2048 | 512 | 32 | 2048 + 32*32 = 3072 |
| L (100M) | ~100M | 3072 | 768 | 48 | 3072 + 32*48 = 4608 |
| XL (200M) | ~200M | 4096 | 1024 | 64 | 4096 + 32*64 = 6144 |

Note: The paper's labeling uses 12M/25M/50M/100M/200M. The values in this table follow the
official DreamerV3 configuration files. For this skill's default configuration:
- `deter_dim = 1024`, `hidden_dim = 1024`, `num_classes = 32` (approximately S/M hybrid)

---

## Activations and Normalization

### Activations

SiLU (Swish) is used exclusively in DreamerV3:

```
SiLU(x) = x * sigmoid(x)
```

Properties:
- Smooth everywhere (unlike ReLU)
- Non-monotonic (unlike sigmoid/tanh)
- Self-gated: output naturally bounded near zero for negative inputs
- Better gradient flow than ReLU for deep MLPs

Do NOT use ReLU or GELU in the RSSM components.

### Normalization Strategy

| Location | Normalization | Reason |
|----------|--------------|--------|
| BlockGRU input projection | LayerNorm | Stabilize input before gate computation |
| BlockGRU output | RMSNorm | Normalize hidden state for downstream use |
| Prior/Posterior MLP intermediate | LayerNorm | Standard MLP normalization |
| Prior/Posterior MLP output | None | Raw logits for softmax |
| Feature vector | None | No normalization; used directly as head input |

### LayerNorm vs RMSNorm

- **LayerNorm**: subtracts mean, divides by std, applies scale+bias. Used in MLP intermediate layers.
- **RMSNorm**: only divides by RMS, applies scale (no bias). Used on GRU output. Cheaper and
  sufficient when mean-centering is not needed.

Use `nn.LayerNorm` and `nn.RMSNorm` (available from PyTorch 2.4+). For older PyTorch versions,
implement RMSNorm manually:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale
```

---

## RSSMState Dataclass

```python
@dataclass
class RSSMState:
    deter: Tensor   # (batch, deter_dim)        — deterministic state h_t
    stoch: Tensor   # (batch, stoch_dim, num_classes) — stochastic sample z_t
    logits: Tensor  # (batch, stoch_dim, num_classes) — distribution logits

    @property
    def features(self) -> Tensor:
        return torch.cat([
            self.deter,
            self.stoch.flatten(-2, -1)
        ], dim=-1)

    @property
    def batch_size(self) -> int:
        return self.deter.shape[0]
```

---

## Gradient Flow Summary

Understanding where gradients flow through the RSSM is critical for correct implementation:

1. **Through BlockGRU**: Gradients flow back through time via `h_t`. The LayerNorm and SiLU
   in the input projection stabilize these gradients.

2. **Through straight-through**: Gradients flow through `probs` (the soft categorical
   probabilities), not through the discrete sample. This means the prior/posterior networks
   receive gradients from downstream losses.

3. **KL stop-gradients**: In the dynamics loss, the posterior logits are detached. In the
   representation loss, the prior logits are detached. This implements separate training
   pressures on each network.

4. **Through imagination**: During actor-critic training, gradients flow from the imagined
   returns back through the RSSM parameters via the straight-through estimator. This jointly
   trains the world model and policy.

5. **Reward/continue heads**: These receive gradients from their respective prediction losses
   (cross-entropy for reward via twohot, binary cross-entropy for continue).
