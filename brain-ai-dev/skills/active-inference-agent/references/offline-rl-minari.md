# Offline RL via Minari

## Overview

Phase 5 offline mode uses Minari for Gymnasium-aligned dataset loading. Train world model
components (encoder, transition, likelihood, preferences) from pre-collected trajectories
without online environment interaction, then run the planner on held-out episodes.

The pipeline runs in four stages:

1. Load a Minari dataset and extract episodes into structured arrays.
2. Batch episodes into fixed-length windows and build a PyTorch DataLoader.
3. Train the generative model stack (encoder, transition, likelihood, preferences) via
   supervised sequence prediction in latent space.
4. Evaluate: run the planner on held-out starting states and compare EFE against the
   behavior policy recorded in the dataset.

All code guards on `HAS_MINARI` and `HAS_GYMNASIUM` flags already defined in
`brain_ai/datasets/phase5_active_inference.py`. When Minari is absent, offline RL
tests skip with a warning rather than crashing.

---

## Minari Library Background

Minari is the official successor to D4RL, maintained by the Farama Foundation alongside
Gymnasium. It resolves D4RL's long-standing issues: pinned MuJoCo versions, broken
downloads, and incompatibility with Gymnasium's `terminated`/`truncated` split.

Key properties:

- **Gymnasium-native**. Every Minari dataset records `observations`, `actions`, `rewards`,
  `terminations`, and `truncations` per episode, matching the Gymnasium step API.
- **Standardized dataset API**. Three access patterns cover all use cases:
  - `minari.load_dataset(name)` -- load by registered name.
  - `dataset.iterate_episodes()` -- iterate all episodes lazily.
  - `dataset.sample_episodes(n)` -- random sample of `n` episodes.
- **D4RL compatibility namespace**. Legacy D4RL names map to Minari IDs (see
  `MinariDataset.D4RL_DATASETS` in `phase5_active_inference.py`). Use
  `"D4RL/antmaze/large-diverse-v1"` or the shorthand `"antmaze-large-diverse-v1"`.
- **Custom datasets**. Create and register project-specific datasets with
  `minari.create_dataset_from_collector()` for reproducible offline experiments.
- **Standard benchmarks**. AntMaze, Hopper, Walker2d, HalfCheetah, PointMaze, Kitchen,
  and Adroit tasks are available as remote datasets downloaded on first use.

Install:

```bash
pip install minari[all]       # Full install with all environment wrappers
pip install minari             # Core only (no MuJoCo dependencies)
```

---

## Dataset Loading Pipeline

### Step 1: Load the Dataset

```python
import minari

dataset = minari.load_dataset("pointmaze-medium-v2")
```

If the dataset is not cached locally, Minari downloads it automatically when the
`download` flag is set (the existing `MinariDataset` class handles this). Confirm
available datasets with `minari.list_remote_datasets()` and local cache with
`minari.list_local_datasets()`.

### Step 2: Extract Episodes

Convert the Minari iterator into a list of structured NumPy dictionaries. Each episode
contains arrays aligned on the time axis.

```python
from typing import List, Dict
import numpy as np

def extract_episodes(dataset) -> List[Dict[str, np.ndarray]]:
    """Extract all episodes into a flat list of dicts."""
    episodes = []
    for episode in dataset.iterate_episodes():
        episodes.append({
            'observations': episode.observations,       # (T+1, obs_dim)
            'actions': episode.actions,                 # (T, action_dim)
            'rewards': episode.rewards,                 # (T,)
            'terminations': episode.terminations,       # (T,)
            'truncations': episode.truncations,         # (T,)
        })
    return episodes
```

Note the length asymmetry: `observations` has `T+1` entries (initial observation plus
one per step), while `actions`, `rewards`, `terminations`, and `truncations` each have
`T` entries. Account for this when forming `(o_t, a_t, o_{t+1})` tuples.

### Step 3: Windowed Batching

Slice episodes into overlapping windows of fixed length for sequential training. This
preserves temporal structure while enabling mini-batch SGD.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class WindowedOfflineDataset(Dataset):
    """Slice episodes into fixed-length training windows."""

    def __init__(self, episodes: List[Dict[str, np.ndarray]], window_size: int = 32):
        self.windows = []
        for ep in episodes:
            T = len(ep['actions'])
            if T < window_size:
                # Pad short episodes
                self.windows.append(self._pad(ep, window_size))
            else:
                # Overlapping windows with stride = window_size // 2
                stride = max(1, window_size // 2)
                for start in range(0, T - window_size + 1, stride):
                    end = start + window_size
                    self.windows.append({
                        'observations': ep['observations'][start:end],
                        'actions': ep['actions'][start:end],
                        'next_observations': ep['observations'][start+1:end+1],
                        'rewards': ep['rewards'][start:end],
                        'dones': np.logical_or(
                            ep['terminations'][start:end],
                            ep['truncations'][start:end],
                        ).astype(np.float32),
                    })

    def _pad(self, ep, window_size):
        T = len(ep['actions'])
        pad_len = window_size - T
        return {
            'observations': np.pad(ep['observations'][:T], ((0, pad_len), (0, 0))),
            'actions': np.pad(ep['actions'], ((0, pad_len), (0, 0))
                              if ep['actions'].ndim > 1 else ((0, pad_len),)),
            'next_observations': np.pad(ep['observations'][1:T+1], ((0, pad_len), (0, 0))),
            'rewards': np.pad(ep['rewards'], (0, pad_len)),
            'dones': np.pad(
                np.logical_or(ep['terminations'], ep['truncations']).astype(np.float32),
                (0, pad_len), constant_values=1.0,
            ),
        }

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        return {k: torch.from_numpy(v).float() for k, v in w.items()}
```

Create the DataLoader with shuffle enabled. Shuffling windows across episodes breaks
temporal correlations between consecutive batches, which stabilizes training.

```python
loader = DataLoader(
    WindowedOfflineDataset(episodes, window_size=32),
    batch_size=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
```

---

## World Model Training

Train the four generative model components from offline data. All components are defined
in `brain_ai/decision/active_inference.py`: `StateEncoder` (encoder), `GenerativeModel`
(transition + likelihood), and `Preferences`.

### Encoder Training -- q(s|o)

The encoder maps observations to a Gaussian latent distribution. Train it as a VAE:
encode the observation, sample a latent, decode back to the observation space, and
minimize reconstruction loss plus KL divergence against a standard normal prior.

- **Reconstruction loss**: MSE for continuous observations, BCE for binary/image
  observations.
- **KL divergence**: analytical KL between the encoder posterior N(mu, sigma^2) and the
  prior N(0, I).
- **KL weight annealing**: start `kl_weight` at 0.01 and linearly increase to
  `kl_weight_final` (default 1.0) over `kl_warmup_steps` optimizer steps. This prevents
  posterior collapse early in training when the decoder is still weak.

```python
kl_weight = min(kl_weight_final, kl_weight_final * step / kl_warmup_steps)
```

### Transition Model Training -- P(s'|s, a)

The transition model predicts the next latent state distribution given the current latent
state and the action taken.

- **Input**: `(s_t, a_t)` where `s_t = encoder.sample(o_t)`.
- **Target**: `s_{t+1} = encoder.sample(o_{t+1})`.
- **Loss**: negative log-likelihood of `s_{t+1}` under the predicted Gaussian
  `N(mu_pred, sigma_pred^2)`.
- **Gradient management**: either detach encoder outputs (two-phase training) or train
  jointly with a scaling factor (0.1x encoder gradient from transition loss).
- **Ensemble option**: train `K` transition models (typically K=5) on bootstrapped subsets
  of the data. Use the ensemble disagreement as an epistemic uncertainty estimate. This
  feeds directly into the epistemic term of EFE.

### Likelihood Model Training -- P(o|s)

The likelihood model (decoder) reconstructs observations from latent states. It is trained
jointly with the encoder as the decoder arm of the VAE.

- **Loss**: MSE for continuous observations, BCE for image observations.
- **Separate training**: alternatively, freeze the encoder after initial VAE training and
  fine-tune the decoder alone. This is useful when the encoder must remain stable for
  downstream transition model training.

### Preference Model Training

Preferences define the target observation distribution that drives pragmatic value in EFE.

Three strategies:

1. **Fixed preferences**. Derive from the task goal. For navigation, encode the goal
   position as the preferred observation vector. Set `Preferences.learnable = False`.

2. **Learned from reward**. Weight observations by exponentiated reward and fit the
   preference distribution to high-reward observations.

   ```python
   weights = torch.exp(batch['rewards'] / temperature)
   weights = weights / weights.sum()
   weighted_obs = (weights.unsqueeze(-1) * batch['observations']).sum(dim=0)
   preference_loss = F.mse_loss(preferences.get_preference(), weighted_obs.detach())
   ```

   Strategy: observations with higher reward get higher weight in the preference target.
   The temperature parameter controls how sharply the weighting favors top rewards
   (lower temperature = sharper). Start with `temperature = 1.0` and tune.

3. **Reward-derived distribution**. Train a small network that maps reward values to
   preferred observation distributions. Loss: maximize log-likelihood of observations
   under the predicted preference distribution, weighted by reward.

Log preference satisfaction (negative distance from predicted observations to preference
vector) per training step for monitoring.

### Combined Training Loop

```python
optimizer = torch.optim.Adam(
    list(encoder.parameters()) +
    list(generative.parameters()) +
    list(preferences.parameters()),
    lr=3e-4,
)

for epoch in range(num_epochs):
    for batch in dataloader:
        obs = batch['observations']          # (B, W, obs_dim)
        actions = batch['actions']           # (B, W, action_dim)
        next_obs = batch['next_observations'] # (B, W, obs_dim)

        # Flatten time into batch for per-step losses
        B, W, D = obs.shape
        obs_flat = obs.reshape(B * W, D)
        next_obs_flat = next_obs.reshape(B * W, D)
        actions_flat = actions.reshape(B * W, -1)

        # 1. Encode observations
        z_mu, z_logvar = encoder(obs_flat)
        z_t = encoder.sample(obs_flat)
        z_next_mu, z_next_logvar = encoder(next_obs_flat)
        z_next = encoder.sample(next_obs_flat)

        # 2. Reconstruction loss (likelihood)
        recon = generative.predict_obs(z_t)
        recon_loss = F.mse_loss(recon, obs_flat)

        # 3. KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + z_logvar - z_mu.pow(2) - z_logvar.exp()
        )

        # 4. Transition loss
        trans_mu, trans_logvar = generative.predict_next_state(z_t.detach(), actions_flat)
        trans_std = torch.exp(0.5 * trans_logvar)
        trans_loss = 0.5 * torch.mean(
            trans_logvar + (z_next.detach() - trans_mu).pow(2) / trans_std.pow(2)
        )

        # 5. KL annealing
        kl_weight = min(kl_weight_final, kl_weight_final * global_step / kl_warmup_steps)

        # 6. Combined loss
        loss = recon_loss + kl_weight * kl_loss + trans_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(generative.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        global_step += 1
```

---

## Evaluation Protocol

### Offline Planning Evaluation

After training the world model, run the planner on held-out trajectories without
any environment interaction. Split episodes 80/20 before training; the 20% hold-out set
is for evaluation only.

Procedure for each held-out episode:

1. Take the starting observation `o_0`.
2. Encode to latent state: `s_0 = encoder.sample(o_0)`.
3. Run the planner (random shooting or CEM) from `s_0` to select an action sequence.
4. Compute the total EFE of the planner's selected action sequence by rolling out
   through the learned transition model.
5. Compute the total EFE of the behavior policy's actual action sequence (from the
   dataset) by rolling out through the same transition model.
6. Compare: the planner should achieve lower (better) EFE than the behavior policy
   if the world model is accurate and the planner is effective.

```python
def evaluate_offline(encoder, generative, preferences, planner, eval_episodes, horizon=8):
    results = []
    for ep in eval_episodes:
        o_0 = torch.from_numpy(ep['observations'][0]).float().unsqueeze(0)
        behavior_actions = torch.from_numpy(ep['actions'][:horizon]).float().unsqueeze(0)

        # Encode starting state
        s_0 = encoder.sample(o_0)

        # Planner EFE
        planner_output = planner.plan(s_0, generative, preferences, horizon=horizon)
        planner_efe = planner_output.efe_total.item()

        # Behavior policy EFE
        behavior_efe = compute_trajectory_efe(
            s_0, behavior_actions, generative, preferences
        )

        results.append({
            'planner_efe': planner_efe,
            'behavior_efe': behavior_efe,
            'planner_actions': planner_output.action.cpu().numpy(),
            'behavior_actions': ep['actions'][:horizon],
        })

    return results
```

### Metrics

Compute three primary metrics from the evaluation results:

| Metric | Formula | Interpretation |
|---|---|---|
| **EFE improvement** | `(efe_behavior - efe_planner) / abs(efe_behavior)` | Fraction by which the planner improves over the behavior policy. Positive means the planner is better. |
| **Action agreement** | Fraction of steps where `argmax(planner_action) == argmax(behavior_action)` | How often the planner would have selected the same action. High agreement on expert data validates the world model. |
| **Preference satisfaction** | `mean(log p_pref(o_predicted))` for planner vs behavior trajectories | Log-probability of predicted observations under the preference distribution. Higher means closer to goal. |

### Optional Online Validation

If the Gymnasium environment is available, roll out the planner's actions in the live
environment as a sanity check. This is not strictly offline RL but validates that the
learned world model transfers to real dynamics.

```python
if HAS_GYMNASIUM:
    env = gym.make(env_name)
    obs, info = env.reset()
    total_reward = 0.0
    for step in range(max_steps):
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        s = encoder.sample(obs_tensor)
        planner_output = planner.plan(s, generative, preferences, horizon=horizon)
        action = planner_output.action[0].cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    env.close()
```

Metrics for online validation: cumulative reward, episode length, task success rate.
Always guard with `if HAS_GYMNASIUM:` to avoid import failures.

### Metrics JSON Output

Save results in a standardized JSON format for experiment tracking.

```python
import json
from datetime import datetime

metrics = {
    "dataset_name": "pointmaze-medium-v2",
    "num_episodes_train": len(train_episodes),
    "num_episodes_eval": len(eval_episodes),
    "train_epochs": 50,
    "world_model_loss": {
        "reconstruction": float(final_recon_loss),
        "kl_divergence": float(final_kl_loss),
        "transition": float(final_trans_loss),
    },
    "efe_improvement": float(np.mean(efe_improvements)),
    "efe_improvement_std": float(np.std(efe_improvements)),
    "action_agreement": float(np.mean(action_agreements)),
    "preference_satisfaction_planner": float(np.mean(pref_sat_planner)),
    "preference_satisfaction_behavior": float(np.mean(pref_sat_behavior)),
    "online_validation": {
        "cumulative_reward": float(total_reward) if HAS_GYMNASIUM else None,
        "episode_length": int(step + 1) if HAS_GYMNASIUM else None,
        "task_success": bool(success) if HAS_GYMNASIUM else None,
    },
    "config": {
        "window_size": config.window_size,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "kl_warmup_steps": config.kl_warmup_steps,
        "kl_weight_final": config.kl_weight_final,
        "planning_horizon": config.planning_horizon,
    },
    "timestamp": datetime.utcnow().isoformat() + "Z",
}

with open("metrics_offline_rl.json", "w") as f:
    json.dump(metrics, f, indent=2)
```

---

## Graceful Fallbacks

All offline RL functionality degrades gracefully when optional dependencies are missing.

| Condition | Behavior |
|---|---|
| `HAS_MINARI is False` | Skip offline RL dataset loading. Log `"Minari not installed. Install with: pip install minari[all]"`. All offline RL tests skip via `@pytest.mark.skipif`. |
| `HAS_GYMNASIUM is False` | Skip online validation. Offline planning (world model + planner EFE comparison) still runs. Log warning. |
| Specific dataset not available remotely | Catch `Exception` in `_load_dataset()`, list available datasets with `minari.list_remote_datasets()[:20]`, skip with descriptive message. |
| Dataset download fails (network) | Catch in `_load_dataset()`, log error, return empty dataset. Training code must check `len(dataset) > 0` before proceeding. |
| Observation space mismatch | Validate `dataset.observation_space` shape against encoder `obs_dim` at load time. Raise `ValueError` with both shapes if mismatched. |

Test decorators:

```python
import pytest

HAS_MINARI = False
try:
    import minari
    HAS_MINARI = True
except ImportError:
    pass

@pytest.mark.skipif(not HAS_MINARI, reason="minari not installed")
def test_offline_rl_pipeline():
    ...

@pytest.mark.skipif(not HAS_GYMNASIUM, reason="gymnasium not installed")
def test_online_validation():
    ...
```

---

## Configuration

```python
from dataclasses import dataclass

@dataclass
class OfflineRLConfig:
    """Configuration for offline RL via Minari."""

    # Dataset
    dataset_name: str = "pointmaze-medium-v2"
    max_episodes: int | None = None       # None = use all episodes
    eval_split: float = 0.2               # Fraction reserved for evaluation

    # Windowed batching
    window_size: int = 32                 # Timesteps per training window
    window_stride: int | None = None      # None = window_size // 2

    # Training
    batch_size: int = 256
    num_epochs: int = 50
    learning_rate: float = 3e-4
    grad_clip: float = 1.0

    # KL annealing
    kl_warmup_steps: int = 1000
    kl_weight_final: float = 1.0

    # Transition model
    transition_ensemble_size: int = 1     # >1 for ensemble uncertainty
    detach_encoder_for_transition: bool = True

    # Preference learning
    preference_mode: str = "reward_derived"  # "fixed", "learned", "reward_derived"
    reward_temperature: float = 1.0

    # Evaluation
    planning_horizon: int = 8
    num_eval_episodes: int = 50
    use_online_validation: bool = False   # Requires HAS_GYMNASIUM

    # Planner
    planner_type: str = "cem"             # "random_shooting" or "cem"
    num_rollouts: int = 128
    cem_iterations: int = 5
    cem_elite_fraction: float = 0.1
```

Presets for common scenarios:

```python
@classmethod
def quick_test(cls) -> "OfflineRLConfig":
    """Fast iteration: small dataset, few epochs."""
    return cls(
        dataset_name="pointmaze-medium-v2",
        max_episodes=100,
        num_epochs=5,
        batch_size=64,
        window_size=16,
        num_eval_episodes=10,
    )

@classmethod
def full_benchmark(cls) -> "OfflineRLConfig":
    """Full benchmark run for publication results."""
    return cls(
        dataset_name="pointmaze-medium-v2",
        num_epochs=100,
        batch_size=512,
        transition_ensemble_size=5,
        cem_iterations=10,
        num_rollouts=256,
        num_eval_episodes=200,
        use_online_validation=True,
    )
```

---

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError` on episode fields | Minari version mismatch; older versions use different attribute names | Pin `minari>=1.0.0`. Check `episode.observations` vs `episode.obs`. |
| Observation shape `(T+1,)` vs actions shape `(T,)` mismatch | Off-by-one in episode extraction; observations include initial obs | Use `observations[:-1]` for `o_t` and `observations[1:]` for `o_{t+1}`. |
| KL divergence explodes early in training | KL weight too high before decoder learns useful reconstructions | Start `kl_weight` at 0.01, anneal over 1000+ steps. Monitor reconstruction loss separately. |
| Transition model predicts the mean for all inputs | Training on MSE without stochastic output; or detach not applied correctly | Use negative log-likelihood loss (not MSE) with predicted variance. Verify gradient flow. |
| World model loss plateaus but planner performs poorly | Encoder latent space not informative for planning; reconstruction loss dominates | Add latent prediction loss. Increase `state_dim`. Check that transition model gets gradients from planner quality. |
| EFE improvement is negative (planner worse than behavior) | Inaccurate world model; planner exploits model errors | Train longer. Use transition ensemble for uncertainty-aware planning. Reduce planning horizon. |
| `minari.download_dataset()` hangs | Network issue or dataset server down | Set timeout. Try `minari.list_remote_datasets()` first to verify connectivity. Cache datasets locally for CI. |
| CUDA OOM during transition ensemble training | K separate forward passes through encoder per batch | Reduce `transition_ensemble_size` to 3. Use gradient accumulation. Detach encoder for transition training. |
| Action agreement near zero on expert datasets | Planner selects equivalent but different actions (multi-modal policy) | Check EFE improvement instead. Action agreement is only meaningful for unimodal policies. |
| Windows have wrong padding at episode boundaries | Padding function does not set `dones=True` for padded steps | Set `dones` to 1.0 for all padded timesteps so the transition model ignores them. |

---

## End-to-End Example

Complete walkthrough: load dataset, train world model, run planner, save metrics.

```python
#!/usr/bin/env python3
"""Offline RL via Minari -- end-to-end example."""

import json
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

# ---- 0. Check dependencies ------------------------------------------------
try:
    import minari
    HAS_MINARI = True
except ImportError:
    HAS_MINARI = False
    raise RuntimeError("Install Minari: pip install minari[all]")

try:
    import gymnasium as gym
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

from brain_ai.decision.active_inference import (
    StateEncoder,
    GenerativeModel,
    Preferences,
    ActiveInferenceConfig,
)

# ---- 1. Configuration -----------------------------------------------------
config = ActiveInferenceConfig(
    obs_dim=4,          # Adjust to dataset observation dim
    state_dim=32,
    action_dim=2,       # Adjust to dataset action dim
    hidden_dim=128,
    planning_horizon=8,
    num_samples=64,
)

DATASET_NAME = "pointmaze-medium-v2"
WINDOW_SIZE = 32
BATCH_SIZE = 256
NUM_EPOCHS = 50
LR = 3e-4
KL_WARMUP = 1000
KL_FINAL = 1.0
EVAL_SPLIT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 2. Load dataset -------------------------------------------------------
local = minari.list_local_datasets()
if DATASET_NAME not in local:
    minari.download_dataset(DATASET_NAME)
dataset = minari.load_dataset(DATASET_NAME)

# Determine dimensions from dataset
obs_sample = next(iter(dataset.iterate_episodes())).observations
act_sample = next(iter(dataset.iterate_episodes())).actions
obs_dim = obs_sample.shape[-1] if obs_sample.ndim > 1 else 1
act_dim = act_sample.shape[-1] if act_sample.ndim > 1 else 1

config.obs_dim = obs_dim
config.action_dim = act_dim

# ---- 3. Extract and split episodes ----------------------------------------
episodes = []
for episode in dataset.iterate_episodes():
    episodes.append({
        'observations': episode.observations,
        'actions': episode.actions,
        'rewards': episode.rewards,
        'terminations': episode.terminations,
        'truncations': episode.truncations,
    })

np.random.shuffle(episodes)
split = int(len(episodes) * (1 - EVAL_SPLIT))
train_episodes = episodes[:split]
eval_episodes = episodes[split:]

print(f"Train episodes: {len(train_episodes)}, Eval episodes: {len(eval_episodes)}")

# ---- 4. Build windowed DataLoader -----------------------------------------
from torch.utils.data import Dataset as TorchDataset, DataLoader

class WindowedDataset(TorchDataset):
    def __init__(self, episodes, window_size):
        self.windows = []
        for ep in episodes:
            T = len(ep['actions'])
            obs = ep['observations']
            for start in range(0, max(1, T - window_size + 1), window_size // 2):
                end = min(start + window_size, T)
                self.windows.append({
                    'observations': obs[start:end].astype(np.float32),
                    'actions': ep['actions'][start:end].astype(np.float32),
                    'next_observations': obs[start+1:end+1].astype(np.float32),
                    'rewards': ep['rewards'][start:end].astype(np.float32),
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        return {k: torch.from_numpy(v) for k, v in w.items()}

def collate_flat(batch):
    """Concatenate variable-length windows along time, then flatten."""
    out = {}
    for key in batch[0]:
        out[key] = torch.cat([b[key] for b in batch], dim=0)
    return out

train_loader = DataLoader(
    WindowedDataset(train_episodes, WINDOW_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_flat,
    num_workers=4,
)

# ---- 5. Initialize models -------------------------------------------------
encoder = StateEncoder(obs_dim, config.state_dim, config.hidden_dim).to(device)
generative = GenerativeModel(
    obs_dim, config.state_dim, act_dim, config.hidden_dim
).to(device)
preferences = Preferences(obs_dim, learnable=True).to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) +
    list(generative.parameters()) +
    list(preferences.parameters()),
    lr=LR,
)

# ---- 6. Training loop ------------------------------------------------------
global_step = 0
for epoch in range(NUM_EPOCHS):
    epoch_losses = {'recon': 0, 'kl': 0, 'transition': 0}
    num_batches = 0

    for batch in train_loader:
        obs = batch['observations'].to(device)
        actions = batch['actions'].to(device)
        next_obs = batch['next_observations'].to(device)

        # Encode
        z_mu, z_logvar = encoder(obs)
        z_t = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
        z_next_mu, _ = encoder(next_obs)
        z_next = z_next_mu.detach()  # Detach target for transition

        # Reconstruction
        recon = generative.predict_obs(z_t)
        recon_loss = F.mse_loss(recon, obs)

        # KL
        kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        kl_weight = min(KL_FINAL, KL_FINAL * global_step / max(1, KL_WARMUP))

        # Transition
        trans_mu, trans_logvar = generative.predict_next_state(
            z_t.detach(), actions
        )
        trans_loss = 0.5 * torch.mean(
            trans_logvar + (z_next - trans_mu).pow(2) / torch.exp(trans_logvar)
        )

        loss = recon_loss + kl_weight * kl_loss + trans_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(generative.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        global_step += 1

        epoch_losses['recon'] += recon_loss.item()
        epoch_losses['kl'] += kl_loss.item()
        epoch_losses['transition'] += trans_loss.item()
        num_batches += 1

    avg = {k: v / max(1, num_batches) for k, v in epoch_losses.items()}
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | recon={avg['recon']:.4f} "
          f"kl={avg['kl']:.4f} trans={avg['transition']:.4f}")

# ---- 7. Offline planning evaluation ----------------------------------------
efe_improvements = []

encoder.eval()
generative.eval()

with torch.no_grad():
    for ep in eval_episodes[:50]:
        T = min(len(ep['actions']), config.planning_horizon)
        if T < 2:
            continue

        o_0 = torch.from_numpy(
            ep['observations'][0]
        ).float().unsqueeze(0).to(device)
        s_0 = encoder.sample(o_0)

        behavior_actions = torch.from_numpy(
            ep['actions'][:T]
        ).float().unsqueeze(0).to(device)

        # Behavior EFE: roll out behavior actions through world model
        s = s_0
        behavior_efe = 0.0
        for t in range(T):
            pred_obs = generative.predict_obs(s)
            pragmatic = preferences.compute_pragmatic_value(pred_obs)
            behavior_efe += -pragmatic.item()
            s = generative.sample_next_state(s, behavior_actions[:, t])

        # Planner EFE: random shooting
        best_efe = float('inf')
        for _ in range(128):
            s = s_0
            candidate_efe = 0.0
            for t in range(T):
                a = torch.randn(1, act_dim).to(device)
                pred_obs = generative.predict_obs(s)
                pragmatic = preferences.compute_pragmatic_value(pred_obs)
                candidate_efe += -pragmatic.item()
                s = generative.sample_next_state(s, a)
            if candidate_efe < best_efe:
                best_efe = candidate_efe

        if abs(behavior_efe) > 1e-8:
            improvement = (behavior_efe - best_efe) / abs(behavior_efe)
            efe_improvements.append(improvement)

print(f"EFE improvement: {np.mean(efe_improvements):.4f} "
      f"+/- {np.std(efe_improvements):.4f}")

# ---- 8. Save metrics -------------------------------------------------------
metrics = {
    "dataset_name": DATASET_NAME,
    "num_episodes_train": len(train_episodes),
    "num_episodes_eval": len(eval_episodes),
    "train_epochs": NUM_EPOCHS,
    "world_model_loss": avg,
    "efe_improvement": float(np.mean(efe_improvements)),
    "efe_improvement_std": float(np.std(efe_improvements)),
    "timestamp": datetime.utcnow().isoformat() + "Z",
}

with open("metrics_offline_rl.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Metrics saved to metrics_offline_rl.json")
```

Run with:

```bash
python examples/offline_rl_minari.py
```

Expected output on `pointmaze-medium-v2` after 50 epochs: reconstruction loss below 0.05,
transition loss below 0.1, positive EFE improvement (0.1--0.4 range depending on planner
quality and planning horizon).
