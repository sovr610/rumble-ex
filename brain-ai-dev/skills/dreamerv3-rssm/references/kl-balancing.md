# KL Balancing Reference

## Overview

The RSSM world model is trained using a variational objective (ELBO). The KL divergence term
in this objective encourages the posterior distribution (which sees observations) to remain
close to the prior (which does not). DreamerV3 decomposes this KL into two separate terms
with different coefficients and different stop-gradient placements, a technique called
**KL balancing**.

Additionally, **free nats** clipping prevents the posterior from collapsing to the prior early
in training, when the model has not yet learned to encode useful information.

---

## Variational Objective

### ELBO Decomposition

The standard ELBO (Evidence Lower Bound) for sequential latent variable models:

```
ELBO = E_q[log p(x | z)] - KL[q(z | x) || p(z)]
```

Where:
- `p(x | z)`: observation reconstruction likelihood
- `q(z | x)`: posterior over latent states given observations
- `p(z)`: prior over latent states

In the RSSM context:

```
ELBO = sum_t( E[log p(o_t | h_t, z_t)] + E[log p(r_t | h_t, z_t)] + E[log p(c_t | h_t, z_t)] )
     - sum_t( KL[q(z_t | h_t, o_t) || p(z_t | h_t)] )
```

Where:
- `o_t`: observation at time t
- `r_t`: reward at time t
- `c_t`: continuation flag at time t
- `q(z_t | h_t, o_t)`: posterior (depends on observation)
- `p(z_t | h_t)`: prior (depends only on deterministic state)

### Maximizing ELBO = Minimizing Negative ELBO

Training minimizes the negative ELBO:

```
L = -E[log p(o_t | z_t)] - E[log p(r_t | z_t)] - E[log p(c_t | z_t)]
  + KL[q(z_t | h_t, o_t) || p(z_t | h_t)]
```

The first three terms are prediction losses (reconstruction). The KL term is the focus
of this document.

---

## KL Balancing: Two Terms

DreamerV3 splits the single KL term into two components:

### Dynamics Loss (L_dyn)

```
L_dyn = max(free, KL[sg(q) || p])
       = max(free, KL[stop_gradient(posterior) || prior])
```

- **Stop-gradient on posterior**: gradients do NOT flow into the posterior network
- **Gradients DO flow into the prior network**
- **Effect**: trains the prior to match the posterior's distribution
- **Coefficient**: 0.5

The prior learns to be predictive: given only the deterministic state `h_t`, predict where
the posterior will land when it sees the actual observation.

### Representation Loss (L_rep)

```
L_rep = max(free, KL[q || sg(p)])
       = max(free, KL[posterior || stop_gradient(prior)])
```

- **Stop-gradient on prior**: gradients do NOT flow into the prior network
- **Gradients DO flow into the posterior network**
- **Effect**: trains the posterior to stay close to the prior
- **Coefficient**: 0.1

The posterior learns to encode observations in a way consistent with the prior's structure.

### Combined KL Loss

```
L_KL = 0.5 * L_dyn + 0.1 * L_rep
```

---

## Why Two Terms with Different Weights

### Single KL Alternative

With a single KL and no stop-gradient splitting:
```
L_KL_naive = KL[posterior || prior]
```

Gradient of this w.r.t. posterior parameters: `d/d(phi) KL[q_phi || p]`
Gradient w.r.t. prior parameters: `d/d(theta) KL[q || p_theta]`

Both receive gradients simultaneously with the same magnitude. This works but causes
instability because:
1. The prior trying to match the posterior competes with the posterior trying to match the prior
2. There is no clear separation of objectives

### Asymmetric Stop-Gradient

By stopping gradients selectively:

- `L_dyn = KL[sg(q) || p]`: Only prior parameters receive gradients. The prior learns to
  predict where the posterior will be. This is a supervised-style update for the prior.

- `L_rep = KL[q || sg(p)]`: Only posterior parameters receive gradients. The posterior learns
  to be consistent with the prior. This acts as a regularizer on the posterior.

### Coefficient Rationale (0.5 vs 0.1)

- **0.5 on L_dyn**: Higher weight encourages the prior to become informative quickly. A
  predictive prior is essential for quality imagination rollouts, which are the core of
  actor-critic training.

- **0.1 on L_rep**: Lower weight gives the posterior more freedom to encode observations.
  If the representation loss were too strong, the posterior would collapse toward the prior
  and stop encoding useful information from observations.

The asymmetry (5x more weight on dynamics than representation) reflects the practical
importance of a good prior for downstream planning.

---

## Free Nats

### The Posterior Collapse Problem

Early in training:
1. The model has random parameters
2. The prior and posterior produce similar random outputs
3. The KL is small (near 0)
4. The KL gradient is small
5. The model has no incentive to make the posterior encode observations
6. All training pressure goes to the reconstruction loss
7. The posterior eventually collapses to the prior: `q(z) ≈ p(z)`
8. The KL stays near 0, but the model has lost the ability to encode observations

This is **posterior collapse**, where the stochastic state carries no information and the
model degrades to a deterministic sequence model.

### Free Nats Mechanism

```
L_dyn = max(free_nats, KL[sg(q) || p])
L_rep = max(free_nats, KL[q || sg(p)])
```

With `free_nats = 1.0`:
- When KL < 1.0 nat: loss is clamped to `free_nats`, gradient is 0
- When KL > 1.0 nat: loss is the actual KL, gradient flows normally

Effect:
- Early training: KL is small, so the KL loss contributes no gradient
- The model focuses entirely on reconstruction quality
- As reconstruction improves, the model starts using the stochastic state
- KL rises above 1.0 nat naturally as the posterior learns to encode information
- The free nats threshold is crossed, and KL balancing begins to operate normally

This is analogous to the "warm-up" trick in VAE training, but implemented automatically
via the free nats threshold.

### Implementation

```python
def kl_with_free_nats(kl: Tensor, free_nats: float) -> Tensor:
    # kl: (batch, stoch_dim) — per-distribution KL values
    # Apply free nats clamping per-distribution, then sum over stoch_dim
    kl_clamped = torch.clamp(kl, min=free_nats)  # (batch, stoch_dim)
    return kl_clamped.sum(dim=-1)  # (batch,)
```

Alternative: clamp the total KL rather than per-distribution KL. The per-distribution
approach allows some distributions to be below free nats while others are not, which is
more fine-grained but slightly less clean. Both approaches work in practice.

---

## KL Formula for Categoricals

For two categorical distributions `p` and `q` over `num_classes` classes:

```
KL[q || p] = sum_k(q_k * log(q_k / p_k))
           = sum_k(q_k * (log(q_k) - log(p_k)))
```

For the RSSM, there are `stoch_dim` (32) independent categorical distributions. The total KL is:

```
KL_total = sum_{d=1}^{stoch_dim} KL[q_d || p_d]
```

### PyTorch Implementation: Manual

```python
def kl_categorical(p_logits: Tensor, q_logits: Tensor,
                   unimix: float = 0.01) -> Tensor:
    # p_logits, q_logits: (batch, stoch_dim, num_classes)
    num_classes = p_logits.shape[-1]

    # Apply unimix to both distributions
    p_probs = (1 - unimix) * torch.softmax(p_logits, dim=-1) + unimix / num_classes
    q_probs = (1 - unimix) * torch.softmax(q_logits, dim=-1) + unimix / num_classes

    # KL[q || p] = sum_k q_k * (log q_k - log p_k)
    # Shape: (batch, stoch_dim, num_classes) -> sum over num_classes
    kl = (q_probs * (q_probs.log() - p_probs.log())).sum(dim=-1)
    # kl: (batch, stoch_dim)
    return kl  # per-distribution KL
```

### PyTorch Implementation: Using torch.distributions

```python
from torch.distributions import Categorical, kl_divergence

def kl_categorical_dist(p_logits: Tensor, q_logits: Tensor) -> Tensor:
    # p_logits, q_logits: (batch, stoch_dim, num_classes)
    # NOTE: unimix not applied here — apply before creating distributions
    p_dist = Categorical(logits=p_logits)
    q_dist = Categorical(logits=q_logits)
    kl = kl_divergence(q_dist, p_dist)  # (batch, stoch_dim)
    return kl
```

The manual implementation is preferred because it correctly handles unimix probabilities
(which are not representable as pure logits).

---

## Unimix and KL Divergence

### KL is Undefined Without Unimix

When using pure softmax probabilities, if any class `k` has `p_k = 0` but `q_k > 0`:

```
KL[q || p] contains: q_k * log(q_k / p_k) = q_k * log(q_k / 0) = +infinity
```

This causes:
- NaN in gradients
- Training instability
- Complete breakdown if it occurs frequently

### Unimix Prevents Infinite KL

With unimix = 0.01:
```
p_k >= 0.01 / num_classes = 0.01 / 32 ≈ 0.0003125
```

The minimum probability is always positive, so `log(q_k / p_k)` is always finite.

The maximum KL contribution from a single class:
```
q_k * log(q_k / p_k)  ≤  1.0 * log(1.0 / 0.0003125)  ≈  8.0 nats
```

With 32 classes and 32 distributions, the maximum total KL is bounded by:
```
32 distributions * 32 classes * 8 nats ≈ 8192 nats
```

This large maximum KL is never reached in practice but ensures the loss is always finite.

---

## Complete Forward Pass

```python
class WorldModelLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        self.cfg = cfg

    def forward(
        self,
        posterior_logits: Tensor,   # (batch, stoch_dim, num_classes)
        prior_logits: Tensor,       # (batch, stoch_dim, num_classes)
        # ... other prediction arguments
    ) -> LossResult:
        # Compute KL[sg(posterior) || prior] — trains prior
        kl_dyn_per = kl_categorical(
            p_logits=prior_logits,
            q_logits=posterior_logits.detach(),  # stop-gradient on posterior
            unimix=0.01,
        )  # (batch, stoch_dim)

        # Compute KL[posterior || sg(prior)] — trains posterior
        kl_rep_per = kl_categorical(
            p_logits=prior_logits.detach(),  # stop-gradient on prior
            q_logits=posterior_logits,
            unimix=0.01,
        )  # (batch, stoch_dim)

        # Apply free nats and sum over stoch_dim
        kl_dyn = torch.clamp(kl_dyn_per, min=self.cfg.kl_free_nats).sum(-1).mean()
        kl_rep = torch.clamp(kl_rep_per, min=self.cfg.kl_free_nats).sum(-1).mean()

        # Combine
        total_kl = self.cfg.kl_dyn_scale * kl_dyn + self.cfg.kl_rep_scale * kl_rep
        return total_kl
```

---

## Diagnostics: What to Monitor

During training, log these values to detect problems:

| Metric | Healthy Range | Problem Indicated |
|--------|--------------|-------------------|
| `kl_dyn` (before clamp) | 1-50 nats | < 1: prior not learning; > 500: instability |
| `kl_rep` (before clamp) | 1-20 nats | < 1: posterior collapse; > 200: prior too rigid |
| `free_nats_active_fraction` | 0-0.5 | > 0.8: model not encoding observations |
| `entropy(posterior)` | ln(32) max ≈ 3.47 | < 0.1: mode collapse |
| `entropy(prior)` | ln(32) max ≈ 3.47 | < 0.1: prior became deterministic |

### Detecting Posterior Collapse

```python
# During training, periodically compute:
post_probs = torch.softmax(posterior_logits, dim=-1)  # (batch, stoch_dim, num_classes)
entropy = -(post_probs * post_probs.log()).sum(-1)     # (batch, stoch_dim)
mean_entropy = entropy.mean()  # Should be > 0.5 after early training
```

If `mean_entropy < 0.1`, the posterior has collapsed and the model is no longer encoding
observations.

### Detecting Prior Collapse

```python
prior_probs = torch.softmax(prior_logits, dim=-1)
kl_from_uniform = -(1/num_classes * (prior_probs.log() - math.log(num_classes))).sum(-1)
```

If `kl_from_uniform` is very large, the prior has become nearly deterministic, which limits
its ability to match the posterior.

---

## What Happens Without KL Balancing

### Without Free Nats

With `free_nats = 0.0`:
1. Early training: KL is near 0 (random prior and posterior happen to agree)
2. Both prior and posterior receive KL gradients pushing them toward each other
3. Neither develops useful representations first
4. The KL gradient and reconstruction gradient conflict
5. Training is unstable and often diverges
6. Alternatively: posterior collapses early, producing a degenerate world model

### Without Asymmetric Stop-Gradients

With standard KL (no stop-gradients):
```
L_KL = KL[posterior || prior]
```
1. Both prior and posterior receive equal and opposite gradients
2. The model may oscillate: prior moves toward posterior, posterior moves toward prior
3. No clear separation of dynamics learning vs representation learning
4. In practice, the prior becomes very narrow (overfits to the posterior from recent batches)
5. Imagination quality degrades because the prior learned to track the posterior rather than
   make accurate predictions

### Without Unimix

Without unimix:
1. Some categorical classes may reach probability 0 (especially with one-hot-like distributions)
2. `log(0)` → `-infinity`
3. KL divergence becomes `infinity` or `nan`
4. Training halts due to nan gradients
5. Even if this doesn't occur, some classes may never be used (codebook collapse), reducing
   the effective state space from 32*32=1024 to a much smaller number

---

## Summary of Hyperparameters

| Parameter | Value | Sensitivity |
|-----------|-------|-------------|
| `kl_free_nats` | 1.0 | Moderate: range 0.5-5.0 works; too high prevents learning, too low causes collapse |
| `kl_dyn_scale` | 0.5 | Low: changing to 0.3-0.7 has minor effect |
| `kl_rep_scale` | 0.1 | Low: changing to 0.05-0.2 has minor effect |
| `unimix` | 0.01 | Low: range 0.001-0.05 works |
| `stoch_dim` | 32 | High: determines state capacity |
| `num_classes` | 32 | High: determines per-distribution resolution |

The DreamerV3 paper explicitly states that these hyperparameters are fixed across all
environments and do not require tuning. This is a key advantage over prior world model
approaches that required environment-specific KL weights.
