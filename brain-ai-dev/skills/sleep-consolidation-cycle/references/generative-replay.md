# Generative Replay Reference

## Table of Contents

1. [Overview](#1-overview)
2. [Dream Replay from World Model](#2-dream-replay-from-world-model)
3. [Pseudo-Rehearsal Theory](#3-pseudo-rehearsal-theory)
4. [Creative Recombination](#4-creative-recombination)
5. [REM-Like Generative Training](#5-rem-like-generative-training)
6. [Blending Real and Generated Experiences](#6-blending-real-and-generated-experiences)
7. [Quality Control for Generated Experiences](#7-quality-control-for-generated-experiences)
8. [World Model Requirements](#8-world-model-requirements)
9. [Appendix: Troubleshooting](#appendix-a-troubleshooting)

---

## 1. Overview

### Purpose

Generative replay implements the REM-like phase of sleep consolidation. Instead of
replaying stored experiences verbatim (as in NREM), the REM phase uses the world model
to generate novel pseudo-experiences that blend elements of real memories with creative
recombinations. This promotes generalization, prevents overfitting to specific experiences,
and helps the model discover novel state-action combinations.

### Biological Motivation

REM sleep is characterized by vivid, often bizarre dreams that recombine elements of
waking experience in novel ways. Computational theories suggest this serves several
functions:

- **Generalization**: by experiencing novel combinations, the brain extracts general
  principles rather than memorizing specific instances.
- **Catastrophic forgetting prevention**: pseudo-rehearsal of old-ish experiences
  maintains representations that might otherwise be overwritten.
- **Creative problem solving**: novel recombinations can reveal solutions that were not
  apparent during waking experience.
- **Emotional processing**: dream replay may help integrate emotional experiences.

---

## 2. Dream Replay from World Model

### Generation Pipeline

The world model (DreamerV3 RSSM or equivalent) generates pseudo-experiences through
imagination rollouts:

```python
def generate_dream_experiences(
    world_model,
    replay_buffer,
    num_dreams: int,
    horizon: int,
    noise_scale: float = 0.1,
):
    """Generate dream-like pseudo-experiences from the world model.

    Parameters
    ----------
    world_model : nn.Module
        Trained world model with imagine() capability.
    replay_buffer : ReplayBuffer
        Source of initial states for dreaming.
    num_dreams : int
        Number of dream sequences to generate.
    horizon : int
        Length of each dream rollout.
    noise_scale : float
        Scale of noise injected into latent states for creativity.

    Returns
    -------
    list[DreamExperience]
        Generated pseudo-experiences.
    """
    dreams = []
    # Sample starting states from replay buffer
    seed_experiences = replay_buffer.sample(num_dreams)

    with torch.no_grad():
        for seed in seed_experiences:
            # Encode the seed observation to get initial latent state
            initial_state = world_model.encode(seed.observation)

            # Add noise for creative variation
            noisy_state = initial_state + noise_scale * torch.randn_like(initial_state)

            # Roll out the world model
            trajectory = world_model.imagine(
                policy=stochastic_policy,  # random or learned policy
                state=noisy_state,
                horizon=horizon,
            )

            dreams.append(DreamExperience(
                observations=trajectory.observations,
                actions=trajectory.actions,
                rewards=trajectory.rewards,
                states=trajectory.states,
                seed_index=seed.index,
            ))

    return dreams
```

### Stochastic Policy for Dreaming

During dream generation, the action policy should be exploratory (not greedy):

```python
def stochastic_policy(state):
    """Sample actions with entropy for dream diversity."""
    logits = policy_network(state)
    # Add temperature for exploration
    temperature = 1.5  # higher than training temperature
    probs = F.softmax(logits / temperature, dim=-1)
    action = torch.multinomial(probs, 1)
    return action
```

---

## 3. Pseudo-Rehearsal Theory

### Original Concept

Robins (1995) introduced pseudo-rehearsal as a method to prevent catastrophic forgetting
in neural networks. The key idea: instead of storing and replaying actual training
examples, generate pseudo-examples from the network's own learned distribution and
interleave them with new training data.

### Advantages Over Pure Replay

| Property | Real Replay | Pseudo-Rehearsal |
|---|---|---|
| Storage | Requires buffer of real experiences | No storage needed (generated on-the-fly) |
| Privacy | Contains actual data | No actual data stored |
| Diversity | Limited to stored experiences | Can generate novel combinations |
| Coverage | Sparse coverage of state space | Can fill gaps in experience |
| Staleness | Experiences may become irrelevant | Generated from current model (always relevant) |

### Limitations

- **Model quality**: pseudo-rehearsal is only as good as the generative model.
- **Mode collapse**: the generative model may fail to cover all modes of the data distribution.
- **Compounding errors**: in sequential rollouts, small errors compound over time.

### Hybrid Approach (Recommended)

Combine real replay (NREM) with generative replay (REM) for best results:

```python
def hybrid_replay_batch(replay_buffer, world_model, batch_size, blend_ratio):
    """Create a batch mixing real and generated experiences."""
    n_real = int(batch_size * (1 - blend_ratio))
    n_dream = batch_size - n_real

    real_batch = replay_buffer.sample(n_real)
    dream_batch = generate_dream_experiences(world_model, replay_buffer, n_dream, horizon=10)

    return concatenate_batches(real_batch, dream_batch)
```

---

## 4. Creative Recombination

### Mechanism

Creative recombination generates experiences that combine elements from different real
experiences in novel ways:

```python
def creative_recombine(
    experience_a: Experience,
    experience_b: Experience,
    blend_ratio: float = 0.5,
    blend_mode: str = "interpolate",
) -> Experience:
    """Blend two experiences for creative recombination.

    Parameters
    ----------
    experience_a, experience_b : Experience
        Two real experiences to blend.
    blend_ratio : float
        Interpolation weight (0 = pure A, 1 = pure B).
    blend_mode : str
        "interpolate" for latent interpolation,
        "crossover" for temporal crossover.
    """
    if blend_mode == "interpolate":
        # Blend in latent space
        blended_obs = (1 - blend_ratio) * experience_a.obs + blend_ratio * experience_b.obs
        return Experience(obs=blended_obs, action=experience_a.action, ...)

    elif blend_mode == "crossover":
        # Temporal crossover: first half from A, second half from B
        T = experience_a.obs.shape[0]
        crossover_point = int(T * (1 - blend_ratio))
        blended_obs = torch.cat([
            experience_a.obs[:crossover_point],
            experience_b.obs[crossover_point:],
        ], dim=0)
        return Experience(obs=blended_obs, ...)
```

### Types of Creative Recombination

1. **Latent interpolation**: blend observations in latent space, creating intermediate
   states that were never directly experienced.
2. **Temporal crossover**: take the beginning of one experience and the end of another,
   creating novel trajectories.
3. **Context swapping**: keep the actions from one experience but apply them in the
   context (initial state) of another experience.
4. **Noise perturbation**: add structured noise to real experiences, creating slight
   variations that improve robustness.

---

## 5. REM-Like Generative Training

### Training Loop

```python
def rem_phase(model, world_model, replay_buffer, cfg):
    """Execute the REM-like generative replay phase."""
    model.train(False)  # freeze batch norm, disable dropout

    for step in range(cfg.rem_replay_steps):
        # 1. Generate dream experiences
        dreams = generate_dream_experiences(
            world_model, replay_buffer,
            num_dreams=cfg.batch_size,
            horizon=cfg.dream_horizon,
            noise_scale=cfg.dream_noise_scale,
        )

        # 2. Blend with real experiences
        real_batch = replay_buffer.sample(cfg.batch_size)
        blended = blend_batches(real_batch, dreams, cfg.creative_blend_ratio)

        # 3. Train on blended batch
        output = model(blended.observations)
        loss = compute_dream_loss(output, blended)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return PhaseResult(loss=loss.item(), steps=cfg.rem_replay_steps)
```

### Dream Loss

The loss for dream training is typically softer than wake-phase loss, reflecting the
uncertainty in generated experiences:

```python
def compute_dream_loss(output, blended_batch):
    """Compute loss with uncertainty-weighted dream contribution."""
    # Real experiences: standard loss
    real_loss = F.cross_entropy(output[:n_real], blended_batch.targets[:n_real])

    # Dream experiences: softer loss (higher temperature or label smoothing)
    dream_logits = output[n_real:]
    dream_targets = F.softmax(blended_batch.targets[n_real:] / 2.0, dim=-1)
    dream_loss = F.kl_div(
        F.log_softmax(dream_logits, dim=-1),
        dream_targets,
        reduction='batchmean',
    )

    return real_loss + dream_weight * dream_loss
```

---

## 6. Blending Real and Generated Experiences

### Blend Ratios

The creative_blend_ratio controls the proportion of generated content:

| Ratio | Description | When to Use |
|---|---|---|
| 0.0 | Pure real replay (no generation) | World model not trained; early training |
| 0.1 | Mostly real, slight dream augmentation | World model newly trained; conservative |
| 0.3 | Default blend | World model reasonably trained; good balance |
| 0.5 | Equal mix | Highly trained world model; maximum diversity |
| 0.7+ | Mostly generated | Not recommended unless world model is very reliable |

### Blending Strategies

#### Simple Concatenation

Simplest: concatenate real and dream batches:

```python
blended = torch.cat([real_batch, dream_batch], dim=0)
```

#### Interleaved Mixing

Alternate real and dream samples within the batch:

```python
indices = torch.randperm(len(real) + len(dream))
blended = torch.cat([real, dream])[indices]
```

#### Weighted Combination

Weight dream samples by a confidence score from the world model:

```python
dream_weight = world_model.confidence(dream_states)
weighted_dream_loss = dream_weight * per_sample_loss
```

---

## 7. Quality Control for Generated Experiences

### Plausibility Filtering

Not all generated experiences are useful. Filter out implausible ones:

```python
def filter_plausible_dreams(dreams, world_model, threshold=0.5):
    """Remove implausible dream experiences."""
    filtered = []
    for dream in dreams:
        # Compute reconstruction error
        reconstructed = world_model.decode(world_model.encode(dream.observations))
        recon_error = F.mse_loss(reconstructed, dream.observations, reduction='none').mean()

        if recon_error < threshold:
            filtered.append(dream)
    return filtered
```

### Diversity Enforcement

Ensure generated experiences are diverse (not all similar):

```python
def enforce_diversity(dreams, min_distance=0.1):
    """Remove near-duplicate dream experiences."""
    selected = [dreams[0]]
    for dream in dreams[1:]:
        distances = [
            F.cosine_similarity(dream.embedding, s.embedding, dim=-1)
            for s in selected
        ]
        if max(distances) < (1 - min_distance):
            selected.append(dream)
    return selected
```

### Anomaly Detection

Flag dreams that are far out of distribution:

```python
def detect_anomalous_dreams(dreams, replay_buffer):
    """Flag dreams that are anomalously far from real experience distribution."""
    real_stats = compute_statistics(replay_buffer)
    anomalous = []
    for dream in dreams:
        z_score = (dream.mean() - real_stats.mean) / real_stats.std
        if abs(z_score) > 3.0:
            anomalous.append(dream)
    return anomalous
```

---

## 8. World Model Requirements

### Minimum Capabilities

The world model must support:

1. **State encoding**: `encode(observation) -> latent_state`
2. **Imagination rollout**: `imagine(policy, initial_state, horizon) -> trajectory`
3. **Observation decoding**: `decode(latent_state) -> reconstructed_observation`

### Recommended Architecture

The DreamerV3 RSSM (implemented in the `dreamerv3-rssm` skill) provides all required
capabilities:

- **Deterministic path**: Block GRU for temporal coherence in dreams.
- **Stochastic path**: Categorical latent for diversity in generated experiences.
- **Prediction heads**: Reward and continuation predictions for dream quality assessment.

### Training Requirements

The world model must be reasonably well-trained before enabling REM replay:

- Reconstruction loss should be below a threshold (configurable).
- The model should have seen at least `min_buffer_size` experiences.
- Dream rollout quality can be assessed by comparing short rollouts against real continuations.

### Fallback: No World Model

If no world model is available, the REM phase can be:
1. **Skipped entirely** (`enable_rem=False`).
2. **Replaced with noise-augmented real replay**: add structured noise to real experiences
   as a simple form of generative augmentation.

---

## Appendix A: Troubleshooting

| Issue | Cause | Resolution |
|---|---|---|
| Dreams are blurry/incoherent | World model undertrained | Train world model longer; reduce dream_noise_scale |
| All dreams look the same | Low noise; mode collapse in world model | Increase noise_scale; use higher policy temperature |
| Dream training hurts performance | Dream quality too low; blend ratio too high | Reduce creative_blend_ratio; add plausibility filtering |
| World model OOM during dreaming | Rollout horizon too long | Reduce horizon; use gradient checkpointing |
| Dreams diverge from real distribution | Compounding errors in long rollouts | Shorten rollout horizon; re-anchor to real states periodically |
| Creative blending produces NaN | Incompatible state spaces blended | Blend in latent space, not observation space; check dtypes |
