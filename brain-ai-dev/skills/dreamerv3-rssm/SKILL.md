---
name: DreamerV3-Style RSSM World Model
description: >
  This skill should be used when the user asks to "implement DreamerV3 RSSM",
  "build a recurrent state space model", "create Block GRU sequence model",
  "implement unimix categorical", "add symlog twohot prediction heads",
  "implement KL balancing loss", "free nats clipping", "world model loss function",
  "imagination rollout for actor-critic", "straight-through categorical estimator",
  "implement prior and posterior networks", "DreamerV3 world model",
  "symlog transform", "twohot encoding 255 bins", "prevent codebook collapse",
  "DreamerV3 numerical stability", "scale-invariant reward prediction",
  "world model imagination", "RSSM prior posterior KL divergence",
  "Block GRU with RMSNorm", "categorical latent state 32x32",
  or needs guidance on implementing DreamerV3-style world models with
  the full set of numerical stability techniques (symlog, twohot, unimix, KL balancing).
version: 0.1.0
---

# DreamerV3-Style RSSM World Model

## Overview

Implement a complete DreamerV3-style Recurrent State Space Model (RSSM) with all the numerical stability techniques that enable a single set of hyperparameters to work across diverse environments (Atari, DMC, Minecraft) without tuning. The RSSM combines a deterministic recurrence (Block GRU) with a categorical stochastic state, trained via a variational objective with KL balancing and free-nats clipping. All prediction heads use symlog-transformed twohot distributions for scale-invariant learning.

Based on: Hafner et al., "Mastering Diverse Domains through World Models" (2023).

## Public Contract

### RSSM

Core sequence model combining deterministic and stochastic state.

```python
class RSSM(nn.Module):
    def __init__(self, cfg: RSSMConfig): ...
    def initial_state(self, batch_size: int) -> RSSMState: ...
    def observe(self, embed: Tensor, action: Tensor,
                state: RSSMState) -> Tuple[RSSMState, RSSMState]: ...
    def imagine(self, policy: Callable, state: RSSMState,
                horizon: int) -> ImaginedTrajectory: ...
```

### BlockGRU

Modified GRU with Linear -> LayerNorm -> SiLU input gate and RMSNorm on output.

```python
class BlockGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1024): ...
    def forward(self, x: Tensor, h: Tensor) -> Tensor: ...
```

### SymlogTwohot

Scale-invariant distribution for reward/value/observation prediction.

```python
class SymlogTwohot(nn.Module):
    def __init__(self, num_bins: int = 255,
                 low: float = -20.0, high: float = 20.0): ...
    def encode(self, x: Tensor) -> Tensor: ...        # scalar -> twohot
    def decode(self, logits: Tensor) -> Tensor: ...    # logits -> scalar
    def loss(self, logits: Tensor, target: Tensor) -> Tensor: ...
```

### WorldModelLoss

Combined loss with KL balancing, free nats, and prediction losses.

```python
class WorldModelLoss(nn.Module):
    def __init__(self, cfg: LossConfig): ...
    def forward(self, posterior: Distribution, prior: Distribution,
                pred_obs: Tensor, target_obs: Tensor,
                pred_reward: Tensor, target_reward: Tensor,
                pred_cont: Tensor, target_cont: Tensor) -> LossResult: ...
```

## Key Concepts

### RSSM State Decomposition

The latent state has two parts:

| Component | Symbol | Computed From | Purpose |
|-----------|--------|---------------|---------|
| Deterministic | `h_t` | `BlockGRU(h_{t-1}, z_{t-1}, a_{t-1})` | Sequence memory |
| Stochastic | `z_t` | 32 categoricals x 32 classes = 1024-dim | Multimodal uncertainty |

The **feature vector** `f_t = concat(h_t, z_t)` is the input to all prediction heads.

### Block GRU

A standard GRUCell where the input projection is replaced:
- **Standard GRU**: `Linear(input) -> split into gates`
- **Block GRU**: `Linear(input) -> LayerNorm -> SiLU -> GRU gates`
- **Output**: `RMSNorm(h_t)`

Hidden size default: 1024 (configurable per model size).

### Categorical State with Unimix

Stochastic state: 32 independent categorical distributions, each over 32 classes.

**Unimix regularization** prevents codebook collapse:
```
p(class) = 0.99 * softmax(logits) + 0.01 * (1 / num_classes)
```

**Straight-through estimator** for sampling:
```
z_hard = one_hot(argmax(logits))
z = z_hard - logits.detach() + logits    # gradient flows through logits
```

### Prior and Posterior Networks

Both are 2-layer MLPs with SiLU activation and LayerNorm:

| Network | Input | Output |
|---------|-------|--------|
| Prior | `h_t` (deterministic state only) | logits for 32x32 categoricals |
| Posterior | `concat(h_t, embed_t)` | logits for 32x32 categoricals |

### Symlog Twohot Distributions

All prediction heads (reward, continue, observation decoder) use symlog-transformed twohot encoding:

**Symlog/Symexp** (compress large values, preserve small):
```
symlog(x) = sign(x) * ln(|x| + 1)
symexp(x) = sign(x) * (exp(|x|) - 1)
```

**Twohot encoding** of scalar `v`:
1. Apply `symlog(v)` to get compressed value
2. Discretize into K=255 bins uniformly spaced in `[-20, 20]`
3. Find two adjacent bins `b_k, b_{k+1}` bracketing the value
4. Weight: `w_k = |b_{k+1} - symlog(v)| / |b_{k+1} - b_k|`, `w_{k+1} = 1 - w_k`

**Loss**: cross-entropy between predicted logits and twohot target.
**Decode**: `symexp(sum(bin_centers * softmax(logits)))`.

This makes reward prediction scale-invariant across environments spanning rewards from -1 to 1e6.

### KL Balancing with Free Nats

The world model trains via a variational objective with two KL terms:

```
L_dyn = max(free, KL[sg(posterior) || prior])      # trains prior toward posterior
L_rep = max(free, KL[posterior || sg(prior)])       # trains posterior toward prior
L_KL  = 0.5 * L_dyn + 0.1 * L_rep
```

- `sg()` = stop-gradient
- `free = 1.0` nat (free-nats clipping prevents posterior collapse early in training)
- **0.5 / 0.1 weighting**: heavier weight on dynamics loss encourages the prior to be informative; lighter rep loss gives the posterior freedom to encode observations

### Imagination Rollout

Accept a policy callable and horizon, unroll the sequence model for H steps:

```
for t in range(horizon):
    action = policy(features_t)
    h_{t+1} = BlockGRU(h_t, z_t, action)
    z_{t+1} ~ prior(h_{t+1})              # no observations during imagination
    features_{t+1} = concat(h_{t+1}, z_{t+1})
    collect: features, actions, reward_logits, continue_logits
```

Return stacked tensors `(features, actions, reward_logits, continue_logits)` for actor-critic training.

## Configuration Surface

```python
@dataclass
class RSSMConfig:
    deter_dim: int = 1024                  # Block GRU hidden size
    stoch_dim: int = 32                    # Number of categorical distributions
    num_classes: int = 32                  # Classes per distribution
    hidden_dim: int = 1024                 # MLP hidden size
    num_layers: int = 2                    # MLP depth for prior/posterior
    activation: str = "silu"
    norm: str = "layernorm"
    unimix: float = 0.01                   # Uniform mixing ratio

@dataclass
class SymlogTwohotConfig:
    num_bins: int = 255
    low: float = -20.0
    high: float = 20.0

@dataclass
class LossConfig:
    kl_free_nats: float = 1.0
    kl_dyn_scale: float = 0.5             # Weight on dynamics KL
    kl_rep_scale: float = 0.1             # Weight on representation KL
    reward_scale: float = 1.0
    continue_scale: float = 1.0
    obs_scale: float = 1.0
```

## Done-When Gates

1. **RSSM Observe/Imagine Work** — `observe()` produces posterior and prior states with correct shapes (batch, 32, 32). `imagine()` unrolls for H steps and returns stacked tensors. Categorical samples use straight-through + unimix.
2. **Symlog Twohot Round-Trips** — `encode(x)` produces valid twohot vectors summing to 1. `decode(encode(x))` recovers the original value within float tolerance for values in [-1e6, 1e6]. Cross-entropy loss is finite and differentiable.
3. **KL Balancing Correct** — With identical posterior and prior, KL is 0. With `free=1.0`, KL below 1 nat is clamped. Stop-gradient is applied to the correct distribution in each term. Loss weights are 0.5 (dyn) and 0.1 (rep).

## Resources

### Reference Files
- **`references/rssm-architecture.md`** — RSSM equations, Block GRU internals, prior/posterior MLP specs, state shapes, RMSNorm placement, model size table
- **`references/symlog-twohot.md`** — Symlog/symexp formulas, twohot encoding algorithm, bin layout, cross-entropy loss derivation, decode formula, numerical edge cases
- **`references/kl-balancing.md`** — KL decomposition, free-nats rationale, stop-gradient placement, coefficient derivation, posterior collapse prevention, unimix interaction
- **`references/imagination-rollout.md`** — Rollout procedure, policy interface, tensor stacking, actor-critic integration, horizon selection, gradient flow through imagination
- **`references/testing-matrix.md`** — Test scenarios for all components

### Asset Files
- **`assets/block_gru_template.py`** — BlockGRU with modified input projection, RMSNorm output, self-tests
- **`assets/categorical_state_template.py`** — Unimix categorical, straight-through estimator, prior/posterior MLPs, self-tests
- **`assets/symlog_twohot_template.py`** — Symlog/symexp, twohot encode/decode, cross-entropy loss, self-tests
- **`assets/rssm_template.py`** — Full RSSM with observe/imagine, state management, self-tests
- **`assets/world_model_loss_template.py`** — KL balancing, free nats, combined loss, self-tests
- **`assets/rssm_config_template.py`** — All config dataclasses, validation, serialization

### Scripts
- **`scripts/validate_rssm.py`** — Validates done-when gates
- **`scripts/gen_rssm_tests.py`** — Generates 100+ pytest test cases
- **`scripts/rssm_diagnostic.py`** — Diagnostic tool: runs forward pass, prints state shapes, KL values, symlog round-trip errors
