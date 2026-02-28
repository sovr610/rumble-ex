# CEM Planning: Cross-Entropy Method for Robot Action Optimization

## Overview

The Cross-Entropy Method (CEM) is a derivative-free optimizer used for Model Predictive
Control (MPC) in V-JEPA 2-AC robotics. It treats action planning as a distribution-fitting
problem: iteratively refine a Gaussian distribution over action sequences to concentrate
probability mass on low-cost (goal-reaching) sequences.

---

## Algorithm

### Full CEM Loop

```
Initialize:
  mu    = zeros(horizon, 7)   -- mean action sequence
  sigma = ones(horizon, 7)    -- std dev of action sequence

Repeat for K iterations:
  1. SAMPLE:    sample N sequences from Normal(mu, sigma^2)
                sequences: [N, horizon, 7]

  2. CLIP:      enforce action space constraints
                sequences[:, :, 0:3] = clamp(xyz,         -maxnorm, +maxnorm)
                sequences[:, :, 3:6] = 0                   (orientation zeroed)
                sequences[:, :, 6]   = clamp(gripper, -0.75, +0.75)

  3. ROLLOUT:   for each of N sequences, run world model autoregressively
                final_reprs: [N, repr_dim]

  4. SCORE:     cost_n = L1(final_reprs[n], goal_repr)  -- scalar per sequence

  5. ELITE:     select top-k sequences by lowest cost
                elite_seqs: [k, horizon, 7]

  6. UPDATE:    new_mu    = mean(elite_seqs, dim=0)
                new_sigma = std(elite_seqs, dim=0)
                mu    = momentum * mu    + (1 - momentum) * new_mu
                sigma = momentum * sigma + (1 - momentum) * new_sigma

Return: mu[0]  -- optimal first action to execute
```

### Why L1 Score?

L1 (mean absolute error) between the predicted final representation and the goal
representation is more robust to outliers than L2, and matches the original V-JEPA 2
training loss convention.

```python
cost = torch.mean(torch.abs(predicted_repr - goal_repr), dim=-1)  # [N]
```

---

## Action Space Constraints

### 7-DOF Action Vector

```
action = [dx, dy, dz, droll, dpitch, dyaw, dgripper]
```

| Index | DOF | Constraint | Reason |
|---|---|---|---|
| 0-2 | xyz translation | clamp to ±maxnorm | Prevents large unsafe movements |
| 3-5 | orientation (euler) | set to 0 | Orientation planned separately or fixed |
| 6 | gripper opening | clamp to [-0.75, 0.75] | DROID gripper physical limits |

### Clipping Implementation

```python
def clip_actions(sequences: torch.Tensor, maxnorm: float = 0.02,
                 gripper_range: tuple = (-0.75, 0.75)) -> torch.Tensor:
    """
    Args:
        sequences: [N, horizon, 7] -- action sequences
    Returns:
        clipped: [N, horizon, 7]
    """
    sequences = sequences.clone()
    # XYZ: clamp each component independently
    sequences[..., 0:3] = sequences[..., 0:3].clamp(-maxnorm, maxnorm)
    # Orientation: zero out (CEM doesn't plan orientation)
    sequences[..., 3:6] = 0.0
    # Gripper: clamp to physical range
    sequences[..., 6] = sequences[..., 6].clamp(gripper_range[0], gripper_range[1])
    return sequences
```

---

## Momentum Updates

Momentum stabilizes the CEM distribution across iterations, preventing premature collapse
of sigma to near-zero and avoiding wild oscillations in mu.

```python
# momentum in [0, 1]; higher = more inertia (slower adaptation)
mu    = momentum * mu_prev    + (1 - momentum) * elite_mean
sigma = momentum * sigma_prev + (1 - momentum) * elite_std.clamp(min=1e-6)
```

### Typical Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| horizon | 10 | Steps to plan ahead |
| num_samples (N) | 512 | Action sequences sampled per iteration |
| num_elites (k) | 64 | Top-k sequences retained (k/N = 12.5%) |
| num_iterations (K) | 5 | CEM refinement iterations |
| momentum | 0.1 | Low momentum -> fast adaptation |
| maxnorm | 0.02 m | Maximum xyz step per timestep |
| gripper_range | (-0.75, 0.75) | Physical gripper limits |

### Separate Momentum per DOF Group

```python
@dataclass
class CEMConfig:
    momentum_xyz: float = 0.1      # Low: translation adapts quickly
    momentum_gripper: float = 0.3  # Higher: gripper state changes slowly
```

In the update step:
```python
mu[..., 0:3] = momentum_xyz * mu_prev[..., 0:3] + (1 - momentum_xyz) * new_mu[..., 0:3]
mu[..., 6]   = momentum_g * mu_prev[..., 6]   + (1 - momentum_g) * new_mu[..., 6]
```

---

## Rollout Strategy

### Autoregressive World Model Rollout

```
For each of N sampled sequences:
  repr = current_repr  (from encoder)
  for t in range(horizon):
      repr = world_model.predict_next(repr, sequence[t], current_state)
  final_repr = repr
```

### Batched Rollout for Efficiency

Run all N sequences in parallel by batching:

```python
def batched_rollout(world_model, current_repr, sequences, current_state):
    """
    Args:
        current_repr: [repr_dim] -- single current state
        sequences: [N, horizon, 7] -- N action sequences
        current_state: [7] -- current robot state

    Returns:
        final_reprs: [N, repr_dim]
    """
    N, horizon, _ = sequences.shape

    # Expand current repr to batch dimension
    repr_batch = current_repr.unsqueeze(0).expand(N, -1)  # [N, repr_dim]
    state_batch = current_state.unsqueeze(0).expand(N, -1)  # [N, 7]

    for t in range(horizon):
        actions_t = sequences[:, t, :]  # [N, 7]
        repr_batch = world_model.predict_next(repr_batch, actions_t, state_batch)
        # state_batch remains constant (open-loop planning)

    return repr_batch  # [N, repr_dim]
```

### Open-Loop vs Closed-Loop

- **Open-loop** (default): state_batch stays fixed at current state throughout rollout.
  Faster but less accurate for long horizons.
- **Closed-loop**: update state_batch using forward kinematics after each step.
  More accurate but requires FK model.

---

## Complete CEMPlanner Implementation Sketch

```python
class CEMPlanner:
    def __init__(self, world_model: WorldModel, config: CEMConfig):
        self.world_model = world_model
        self.config = config

    def plan(self, current_repr, goal_repr, current_state) -> torch.Tensor:
        cfg = self.config
        device = current_repr.device

        # Initialize distribution
        mu    = torch.zeros(cfg.horizon, 7, device=device)
        sigma = torch.ones(cfg.horizon, 7, device=device)

        for iteration in range(cfg.num_iterations):
            # 1. Sample
            eps = torch.randn(cfg.num_samples, cfg.horizon, 7, device=device)
            sequences = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

            # 2. Clip
            sequences = clip_actions(sequences, cfg.maxnorm, cfg.gripper_range)

            # 3. Rollout
            final_reprs = batched_rollout(self.world_model, current_repr, sequences, current_state)

            # 4. Score (L1)
            goal_expanded = goal_repr.unsqueeze(0).expand(cfg.num_samples, -1)
            costs = torch.mean(torch.abs(final_reprs - goal_expanded), dim=-1)  # [N]

            # 5. Select elites
            elite_idxs = torch.argsort(costs)[:cfg.num_elites]
            elites = sequences[elite_idxs]  # [k, horizon, 7]

            # 6. Update with momentum
            new_mu    = elites.mean(dim=0)
            new_sigma = elites.std(dim=0).clamp(min=1e-6)

            # Apply per-DOF momentum
            mu_new = mu.clone()
            mu_new[..., 0:3] = cfg.momentum_xyz * mu[..., 0:3] + (1 - cfg.momentum_xyz) * new_mu[..., 0:3]
            mu_new[..., 3:6] = 0.0  # orientation always zeroed
            mu_new[..., 6]   = cfg.momentum_gripper * mu[..., 6] + (1 - cfg.momentum_gripper) * new_mu[..., 6]
            sigma = cfg.momentum_xyz * sigma + (1 - cfg.momentum_xyz) * new_sigma

            mu = mu_new

        # Return optimal first action
        first_action = mu[0]  # [7]
        first_action = clip_actions(first_action.unsqueeze(0).unsqueeze(0), cfg.maxnorm, cfg.gripper_range).squeeze()
        return first_action
```

---

## Convergence Analysis

### Expected Behavior

- Iteration 0: sigma ~ 1.0, random exploration of action space
- Iteration 1-2: sigma begins to collapse around promising regions
- Iteration 3-5: mu converges; sigma << 1 near optimal actions

### Convergence Diagnostics

```python
# Track cost over iterations for debugging
iteration_costs = []
for iteration in range(cfg.num_iterations):
    ...
    best_cost = costs[elite_idxs[0]].item()
    iteration_costs.append(best_cost)

# Should monotonically decrease (or roughly decrease with noise)
assert iteration_costs[-1] < iteration_costs[0], "CEM failed to converge"
```

### Failure Modes

1. **sigma collapse too fast**: reduce momentum (increase inertia)
2. **No improvement after K iterations**: increase num_samples or num_iterations
3. **NaN in sigma**: add `.clamp(min=1e-6)` to sigma updates
4. **Exploding actions**: ensure clip_actions is called before every rollout
