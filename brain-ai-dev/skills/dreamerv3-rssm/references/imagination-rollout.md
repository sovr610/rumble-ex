# Imagination Rollout Reference

## Overview

The imagination rollout is the mechanism by which DreamerV3 trains its actor-critic without
interacting with the real environment during the update step. Starting from states visited
during real experience, the world model extends these states forward in time using only its
prior distribution (no observations), simulating what would happen under different policies.

The actor (policy) and critic (value function) are trained entirely on these imagined
trajectories, making the approach highly sample-efficient.

---

## Rollout Procedure

### Initialization

The rollout begins from a set of real states. In practice, these come from the observe step:
after processing a batch of real experience, the posterior states serve as starting points
for imagination.

```
state_0: RSSMState  — from observe step (posterior)
policy: Callable[[Tensor], Tensor]  — maps features to actions
horizon: int  — number of imagination steps (typically 15)
```

### Per-Step Computation

At each timestep `t` in `[0, horizon - 1]`:

**Step 1**: Extract features from current state.
```
feat_t = concat(state_t.deter, flatten(state_t.stoch))
       = concat(h_t, z_t.reshape(batch, stoch_dim * num_classes))
# shape: (batch, deter_dim + stoch_dim * num_classes)
```

**Step 2**: Query the policy for an action.
```
action_t = policy(feat_t)
# shape: (batch, action_dim)
# For discrete: one-hot or integer index
# For continuous: real-valued vector (e.g., from tanh-squashed Gaussian)
```

**Step 3**: Form GRU input.
```
gru_input = concat(flatten(state_t.stoch), action_t)
# shape: (batch, stoch_dim * num_classes + action_dim)
```

**Step 4**: Advance deterministic state via BlockGRU.
```
h_{t+1} = BlockGRU(gru_input, state_t.deter)
# shape: (batch, deter_dim)
```

**Step 5**: Sample next stochastic state from prior.
```
prior_logits_{t+1} = prior_net(h_{t+1})
# shape: (batch, stoch_dim, num_classes)

z_{t+1} = straight_through_sample(prior_logits_{t+1})
# shape: (batch, stoch_dim, num_classes)
```

**Step 6**: Form new state and extract prediction head outputs.
```
state_{t+1} = RSSMState(h_{t+1}, z_{t+1}, prior_logits_{t+1})
feat_{t+1}  = concat(h_{t+1}, flatten(z_{t+1}))

reward_logits_{t+1} = reward_head(feat_{t+1})
# shape: (batch, num_bins=255)

cont_logits_{t+1} = continue_head(feat_{t+1})
# shape: (batch, 1)
```

**Step 7**: Collect outputs.
```
features.append(feat_t)
actions.append(action_t)
reward_logits.append(reward_logits_{t+1})
continue_logits.append(cont_logits_{t+1})
```

### Output Stacking

After all horizon steps:

```python
ImaginedTrajectory(
    features=torch.stack(features, dim=0),         # (horizon, batch, feature_dim)
    actions=torch.stack(actions, dim=0),            # (horizon, batch, action_dim)
    reward_logits=torch.stack(reward_logits, dim=0), # (horizon, batch, num_bins)
    continue_logits=torch.stack(cont_logits, dim=0), # (horizon, batch, 1)
)
```

The leading dimension is time (horizon), not batch. This convention matches sequence
processing code that iterates over the time dimension.

---

## Policy Interface

### Signature

```python
policy: Callable[[Tensor], Tensor]
# Input:  (batch, feature_dim)  — current feature vector
# Output: (batch, action_dim)   — action to take
```

### Discrete Action Spaces

For discrete actions (e.g., Atari), the policy returns a one-hot vector or a soft sample:

```python
def discrete_policy(features: Tensor) -> Tensor:
    logits = actor_head(features)               # (batch, num_actions)
    dist = Categorical(logits=logits)
    action_idx = dist.sample()                  # (batch,)
    return F.one_hot(action_idx, num_actions).float()  # (batch, num_actions)
```

The one-hot vector is then used as the action input to the GRU.

### Continuous Action Spaces

For continuous actions (e.g., DMControl), the policy returns real-valued actions:

```python
def continuous_policy(features: Tensor) -> Tensor:
    mean = actor_mean_head(features)            # (batch, action_dim)
    log_std = actor_log_std_head(features)      # (batch, action_dim)
    std = torch.exp(log_std.clamp(-10, 2))
    dist = Normal(mean, std)
    action = dist.rsample()                     # reparameterized for gradients
    return torch.tanh(action)                   # squash to (-1, 1)
```

### Policy Requirements During Imagination

The policy must:
1. Accept `(batch, feature_dim)` tensors
2. Return `(batch, action_dim)` tensors
3. Be differentiable (gradients must flow through the policy for actor training)
4. Produce actions compatible with the action space (e.g., bounded continuous)

---

## Gradient Flow Through Imagination

Imagination is not just for computing returns — it is a training objective. Gradients from
the actor-critic losses must flow back through the imagination to update the world model
and policy parameters simultaneously.

### Gradient Path

```
actor_loss (via imagined returns)
    |
    v
reward_decoder(features) ← reward_logits (via symlog twohot)
    |
features = concat(h_t, z_t)
    |
    ├── h_t: BlockGRU(concat(z_{t-1}, action_{t-1}), h_{t-1})
    |   ↑ gradients flow back through GRU time steps
    |
    └── z_t: straight_through_sample(prior_net(h_t))
        ↑ gradients flow through probs, not the discrete sample
```

### Key Implication

The `imagine` method must NOT use `torch.no_grad()`. All operations must track gradients
so that actor and critic training propagates back into:
1. The BlockGRU parameters (through h_t transitions)
2. The prior network parameters (through prior_logits and z_t)
3. The reward and continue head parameters

### Gradient Truncation Option

For very long horizons (H > 20), gradients through the entire horizon can be expensive
and may not improve training. Some implementations truncate gradients at a fixed depth:

```python
# Truncate gradients every K steps
if t % truncate_every == 0:
    state = RSSMState(
        deter=state.deter.detach(),
        stoch=state.stoch.detach(),
        logits=state.logits.detach(),
    )
```

For H=15 (DreamerV3 default), no truncation is needed.

---

## Continue Probability

The continue flag `c_t` indicates whether the episode continues after step `t`. When the
world model predicts `c_t = 0` (episode ends), the imagined trajectory should discount
all future rewards to zero.

### Continue Head

```python
cont_logits = continue_head(features)  # (batch, 1)
cont_prob = torch.sigmoid(cont_logits).squeeze(-1)  # (batch,), probability of continuing
```

### Using Continue in Return Computation

```python
# Imagined trajectory with horizon H
# rewards: (H, batch) — decoded from reward_logits via symlog twohot
# cont_probs: (H, batch) — from sigmoid(continue_logits)

# Lambda-return computation
values = value_head(features)  # (H, batch) — critic estimates
returns = compute_lambda_returns(rewards, cont_probs, values, gamma=0.997, lam=0.95)
```

The `cont_probs` multiplier ensures that once the world model predicts episode termination,
subsequent steps do not contribute to the return:

```
G_t = r_t + gamma * c_t * G_{t+1}
```

---

## Horizon Selection

### Why Horizon = 15

DreamerV3 uses H=15 imagination steps as the default. This balances:

- **Too short (H < 5)**: Actor-critic cannot learn long-horizon credit assignment.
  Policy may become myopic.
- **Too long (H > 30)**: Compounding model errors degrade imagined trajectories.
  The world model's inaccuracies accumulate, leading to hallucinated rewards.
  Gradient computation becomes expensive.
- **H=15**: Sufficient for credit assignment in most environments, while keeping
  compounding error manageable.

### Environment-Specific Considerations

Some environments may benefit from longer horizons:
- Long-horizon tasks (e.g., maze navigation): H=20-25
- Reactive tasks (e.g., Pong): H=10 is sufficient
- Minecraft-style exploration: H=15-20

The DreamerV3 paper uses H=15 across all environments, consistent with its no-tuning philosophy.

---

## Tensor Layout Conventions

### Time-First vs Batch-First

The imagination rollout uses time-first convention: `(horizon, batch, dim)`.

This is consistent with PyTorch's RNN convention and simplifies:
- Iterating over time steps
- Computing returns with `torch.cumprod` or custom lambda-return code
- Masking with continue probabilities

### Indexing Convention

- `trajectory.features[t]` → features at imagination step `t`, shape `(batch, feature_dim)`
- `trajectory.reward_logits[t]` → reward logits at step `t`, shape `(batch, num_bins)`
- `trajectory.continue_logits[t]` → continue logits at step `t`, shape `(batch, 1)`
- `trajectory.actions[t]` → action at step `t`, shape `(batch, action_dim)`

---

## Actor-Critic Integration

### Actor Loss

The actor maximizes imagined returns. Using the straight-through estimator, gradients
flow through the discrete stochastic states:

```python
# Compute lambda-returns from imagined trajectory
rewards = twohot.decode(trajectory.reward_logits)   # (H, batch)
cont_probs = torch.sigmoid(trajectory.continue_logits.squeeze(-1))  # (H, batch)
values = critic(trajectory.features)                 # (H, batch)
returns = lambda_return(rewards, cont_probs, values) # (H, batch)

# Actor loss: maximize returns (minimize negative returns)
actor_loss = -returns.mean()
```

### Critic Loss

The critic minimizes prediction error on imagined returns:

```python
# Critic targets: stop gradient of returns
targets = returns.detach()  # (H, batch)
pred_values = critic(trajectory.features.detach())  # (H, batch)
critic_loss = F.mse_loss(pred_values, targets)
# OR: use symlog twohot for value prediction too
```

### Gradient Blocking

During critic training, block gradients from flowing into the world model:

```python
# Block gradients from critic into world model
features_for_critic = trajectory.features.detach()
```

During actor training, allow gradients to flow:

```python
# Allow gradients from actor into world model
features_for_actor = trajectory.features  # NOT detached
```

---

## Implementation Notes

### Memory Efficiency

For H=15 and batch=16 with feature_dim=2048:

```
features: 15 * 16 * 2048 * 4 bytes = 1.97 MB per batch
actions:  15 * 16 * action_dim * 4 bytes
reward_logits: 15 * 16 * 255 * 4 bytes = 0.24 MB
```

Total per imagination rollout: ~2-3 MB. This is manageable, but for large batches or
very long horizons, consider gradient checkpointing.

### Deterministic vs Stochastic Imagination

Two modes for sampling `z_{t+1}` during imagination:

1. **Stochastic** (default): Sample from categorical with straight-through.
   - Produces diverse trajectories
   - Better for exploration
   - Gradients flow through probs

2. **Deterministic** (argmax): Take argmax of prior logits.
   - Produces single best-guess trajectory
   - Better for evaluation/planning
   - No gradient flow through sampling (argmax is not differentiable)

Training always uses stochastic imagination. Evaluation can use either.
