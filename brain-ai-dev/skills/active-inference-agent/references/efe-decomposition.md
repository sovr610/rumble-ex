# EFE Decomposition

## Overview

Expected Free Energy (EFE) is the objective function for policy evaluation in active
inference. The agent selects actions that minimize EFE, which simultaneously drives
goal achievement, uncertainty reduction, and preservation of future options.

Total EFE for a policy `pi` over planning horizon `H`:

```
G(pi) = Sum_{t=1..H} [ pragmatic(t) + epistemic(t) + instrumental(t) ]
```

Each term is computed by a pure function. The weighted sum of all three terms equals
the total EFE for every batch element, with no separate computation path for the total.
This is the **sum invariant** -- the single non-negotiable correctness property of the
entire EFE module.

Lower EFE indicates a better policy. Action selection converts negative EFE into
action probabilities via softmax: `p(a) = softmax(-G(a) / temperature)`.

---

## Theoretical Foundation

### Connection to Friston's Free Energy Principle

The Free Energy Principle (FEP) posits that biological systems minimize variational
free energy (VFE), an upper bound on surprise: `VFE = E_q[-log p(o,s) + log q(s)]`.
Minimizing VFE with respect to `q` yields perception; minimizing with respect to
action yields behavior (active inference).

### From VFE to EFE

VFE applies to the current timestep. EFE extends this to future timesteps under a
policy, requiring a generative model (transition `P(s'|s,a)` and likelihood `P(o|s)`)
to imagine trajectories:

```
G(pi) = E_q(o_{1:H}, s_{1:H} | pi) [ -log p(o_{1:H}, s_{1:H}) + log q(s_{1:H} | pi) ]
```

### Why Three Terms Instead of Two

The standard Friston decomposition of EFE yields two terms:

```
G_standard(pi, t) = risk(t) + ambiguity(t)
```

Where:
- **Risk**: `KL[ q(o_t|pi) || p_pref(o_t) ]` -- divergence of predicted observations
  from preferences.
- **Ambiguity**: `E_q(s_t|pi)[ H[p(o_t|s_t)] ]` -- expected entropy of the likelihood
  model.

This two-term form covers goal-directedness and epistemic drive but lacks a mechanism
for preserving future controllability. The three-term decomposition adds an
**instrumental** (empowerment) term:

```
G(pi, t) = pragmatic(t) + epistemic(t) + instrumental(t)
```

The pragmatic term absorbs the risk interpretation. The epistemic term captures
information gain (closely related to negative ambiguity but formulated as a KL
between posterior and prior over states). The instrumental term introduces
empowerment -- the mutual information between action sequences and future states --
which penalizes policies that lead to uncontrollable situations.

This three-term form enables the agent to:
1. Pursue goals (pragmatic)
2. Reduce uncertainty about the world (epistemic)
3. Maintain the ability to influence future outcomes (instrumental)

All three behaviors emerge from a single scalar objective, avoiding the need for
separate exploration bonuses or option-preservation heuristics.

---

## Pragmatic Term (Goal-Directed / "Risk")

### Formula

```
pragmatic(t) = E_{q(o_t|pi)}[ -log p_pref(o_t) ]
```

This is the expected negative log-probability of predicted observations under the
preference distribution. It penalizes policies whose predicted outcomes deviate from
desired observations.

### Implementation

Roll out the generative model under policy `pi` to obtain predicted observations,
then evaluate them under the preference distribution. For each timestep `t` in
`1..H`: apply the transition model `q(s_t|s_{t-1}, a_{t-1})`, then the likelihood
model `q(o_t|s_t)`, then score against preferences.

**For Gaussian preferences** (continuous observation spaces):

```
p_pref(o) = N(o; mu_pref, sigma_pref^2)
-log p_pref(o) = 0.5 * [ (o - mu_pref)^2 / sigma_pref^2 + log(2*pi*sigma_pref^2) ]
```

The constant `log(2*pi*sigma_pref^2)` term cancels across policies and can be dropped
for policy comparison, but must be retained if the sum invariant is tested against
analytical values.

**For categorical preferences** (discrete observation spaces):

```
pragmatic(t) = -Sum_i q(o_t = i | pi) * log p_pref(o_t = i)
```

This is the cross-entropy between predicted and preferred observation distributions.

### Pure Function Signature

```python
def compute_pragmatic(
    predicted_obs_mu: torch.Tensor,      # (B, obs_dim)
    predicted_obs_logvar: torch.Tensor,   # (B, obs_dim)
    preference_mu: torch.Tensor,          # (obs_dim,) or (B, obs_dim)
    preference_logvar: torch.Tensor,      # (obs_dim,) or (B, obs_dim)
) -> torch.Tensor:                        # (B,)
    """
    Compute pragmatic EFE term.

    Returns expected negative log-likelihood of predicted observations
    under the preference distribution. Higher values = worse alignment
    with preferences.

    Must NOT modify any state. Must NOT read from self.
    """
```

When Monte Carlo sampling is used instead of closed-form evaluation, pass sampled
observations `(B, num_samples, obs_dim)` and average the negative log-likelihood
across samples.

### Numerical Considerations

- Clamp `preference_logvar` to `[-10, 10]` to prevent division by zero or overflow
  in the `1/sigma^2` term.
- Use `torch.float32` throughout. The sum invariant requires fp32 precision.
- When preferences have very small variance (near-delta), the pragmatic term
  dominates the total EFE. Apply term normalization (see below) to prevent this.

---

## Epistemic Term (Information Gain / "Ambiguity")

### Formula

```
epistemic(t) = E[ KL( q(s_t | o_t, pi) || q(s_t | pi) ) ]
```

This is the expected information gain: the KL divergence between the posterior over
states (after observing `o_t`) and the predictive prior (before observing `o_t`).
Policies that lead to observations carrying more information about the latent state
receive lower EFE (the epistemic term enters with a negative sign in the overall
EFE because higher information gain is desirable).

### Implementation

**For Gaussian distributions** (closed-form KL):

```
KL( N(mu_post, sigma_post^2) || N(mu_prior, sigma_prior^2) )
  = Sum_d [ log(sigma_prior_d / sigma_post_d)
            + (sigma_post_d^2 + (mu_post_d - mu_prior_d)^2) / (2 * sigma_prior_d^2)
            - 0.5 ]
```

Where the sum is over state dimensions `d`. This is exact and differentiable.

In practice:
- `q(s_t | pi)` is the predictive distribution from the transition model (the prior).
- `q(s_t | o_t, pi)` is the posterior after encoding the predicted observation through
  the state encoder.

Compute both distributions from a single forward pass through the generative model.
The prior comes from `P(s_t | s_{t-1}, a_{t-1})`. The posterior comes from encoding
the predicted observation `o_t` through the encoder `q(s|o)`.

**For ensemble-based models** (disagreement proxy):

```
epistemic ≈ Var_k[ mu_k(s_t) ]
```

Where `mu_k` is the mean prediction from ensemble member `k`. Higher disagreement
among ensemble members indicates higher epistemic uncertainty. This is cheaper than
computing KL divergences but is an approximation.

**Monte Carlo approximation for non-Gaussian cases**:

```
epistemic ≈ (1/N) * Sum_{i=1..N} [ log q(s_t^(i) | o_t^(i), pi) - log q(s_t^(i) | pi) ]
```

Where `s_t^(i)` are samples from the posterior. Requires evaluating log-densities
of both distributions at the sample points.

### Pure Function Signature

```python
def compute_epistemic(
    posterior_mu: torch.Tensor,      # (B, state_dim)
    posterior_logvar: torch.Tensor,   # (B, state_dim)
    prior_mu: torch.Tensor,          # (B, state_dim)
    prior_logvar: torch.Tensor,      # (B, state_dim)
) -> torch.Tensor:                   # (B,)
    """
    Compute epistemic EFE term as KL divergence between
    posterior and prior state distributions.

    Returns KL divergence (non-negative). Higher values indicate
    greater expected information gain.

    Must NOT modify any state. Must NOT read from self.
    """
```

### Alternative: Entropy-Based Measure

An equivalent formulation uses the conditional entropy of the likelihood:
`epistemic_alt(t) = H[p(o_t|s_t,a_t)]`. For Gaussian likelihood:
`H = 0.5 * Sum_d [log(2*pi*e*sigma_d^2)]`. This avoids computing the posterior but
requires access to the likelihood model's entropy.

### Numerical Considerations

- Clamp `prior_logvar` from below (e.g., `> -10`) to prevent KL explosion when the
  prior is very confident.
- The KL is always non-negative. If computed values are negative (due to numerical
  error), clamp to zero.
- When `posterior ≈ prior` (near-zero KL), the epistemic term contributes minimally
  to EFE. This is correct behavior -- it means the observation does not carry much
  information.

---

## Instrumental Term (Empowerment)

### Formula

```
empowerment(t..t+K) ≈ I(A_{t:t+K}; S_{t+K})
```

Empowerment is the channel capacity between action sequences and future states. It
measures how much influence the agent's actions have on where it ends up. High
empowerment means the agent is in a controllable region of the state space.

### Tractable Approximations

Exact computation of mutual information is intractable for continuous high-dimensional
spaces. Three practical approximations:

**1. Variational bound (primary method)**

Train two auxiliary networks:
- Source network `q(a|s)`: marginal action distribution given current state.
- Planning network `q(a|s,s')`: infer which action caused the transition from `s` to `s'`.

The variational lower bound on empowerment is:

```
empowerment ≈ E_{a ~ q(a|s), s' ~ P(s'|s,a)} [ log q(a|s,s') - log q(a|s) ]
```

Intuition: if the planning network can identify which action was taken from the state
transition alone, then actions are distinguishable in their effects, and empowerment
is high.

**2. Ensemble disagreement proxy**

When using an ensemble transition model:

```
empowerment_proxy ≈ -Var_k[ f_k(s, a) ]
```

Higher ensemble disagreement for a given `(s, a)` pair means the model is uncertain
about the effect of that action, which implies less control. This is a rough proxy
but avoids training additional networks.

**3. InfoNCE-style bound**

Treat `(a, s')` pairs as positive examples and `(a, s'_neg)` as negatives:

```
empowerment >= E[ log( f(a, s') / (1/K * Sum_j f(a, s'_j)) ) ]
```

Where `f` is a learned critic. This provides a tighter bound than the variational
method when the number of negative samples `K` is large.

### Engineering Details

Empowerment is non-negative (mutual information >= 0). It enters EFE as a
**negative contribution**: `instrumental = -empowerment`, so that all three terms
sum directly to the total. Log empowerment as a positive scalar for debugging;
apply the negation inside the summation only.

### Pure Function Signature

```python
def compute_instrumental(
    state: torch.Tensor,           # (B, state_dim)
    action: torch.Tensor,          # (B, action_dim) or (B,)
    next_state: torch.Tensor,      # (B, state_dim)
    source_logits: torch.Tensor,   # (B, action_dim) -- from source network q(a|s)
    planning_logits: torch.Tensor, # (B, action_dim) -- from planning network q(a|s,s')
) -> torch.Tensor:                 # (B,)
    """
    Compute instrumental EFE term (negative empowerment).

    Returns negative empowerment: lower (more negative) values indicate
    higher empowerment / more control over future states.

    Takes pre-computed logits from source and planning networks rather
    than the networks themselves, to maintain pure function semantics.

    Must NOT modify any state. Must NOT read from self.
    """
```

Note: the pure function takes pre-computed logits, not the networks. The caller is
responsible for running forward passes on the source and planning networks and
passing the resulting logits. This keeps the EFE computation purely functional.

### Training the Empowerment Networks

Train both networks on replay buffer data `(s, a, s')` with cross-entropy losses:
`L_planning = -E[log q_planning(a|s,s')]` and `L_source = -E[log q_source(a|s)]`.
Do not backpropagate EFE gradients through empowerment networks -- use a separate
optimizer.

---

## Sum Invariant

### Hard Requirement

```
|sum(pragmatic, epistemic, instrumental) - efe_total| < 1e-5
```

For every batch element, the sum of the three terms must equal the total EFE within
floating-point tolerance. This is the single most important correctness property of
the EFE module.

### Implementation Strategy

1. Compute each term independently via its pure function.
2. Sum explicitly: `efe_total = pragmatic + epistemic + instrumental`.
3. Never compute the total via a separate code path.
4. Never apply weights inside the individual term functions. Apply weights at the
   summation point:

```python
efe_total = (w_p * pragmatic) + (w_e * epistemic) + (w_i * instrumental)
```

5. Use `torch.float32` for all EFE computation. Cast inputs to fp32 at the entry
   point of each pure function if they arrive in fp16/bf16.
6. Do not use in-place operations (`+=`, `.add_()`) on EFE tensors. Create new
   tensors at each step.

### Assertion Pattern

Insert the following check in the `compute_efe` wrapper (the function that calls all
three term functions and returns the total):

```python
_sum = pragmatic + epistemic + instrumental
assert torch.allclose(efe_total, _sum, atol=1e-5), (
    f"Sum invariant violated: max diff = {(efe_total - _sum).abs().max().item()}"
)
```

In production, gate this behind a debug flag to avoid the overhead. In tests, always
enable it.

### Testing Strategy

1. **Synthetic distributions with closed-form solutions**: construct inputs where each
   term has a known analytical value. Assert the computed terms match within `1e-5`.
2. **Gaussian-Gaussian case**: use Gaussian encoder and Gaussian preferences. Both
   pragmatic and epistemic terms have analytical solutions.
3. **Per-element tolerance**: check `1e-5` tolerance per batch element, not just
   on the mean. Use `torch.allclose` with `atol=1e-5`.
4. **Property-based testing**: generate random inputs, compute all three terms, sum
   them, and verify the sum equals the returned total. Do not check the absolute
   values (they depend on the inputs), only the sum equality.
5. **Gradient flow**: verify that gradients flow through all three terms by checking
   that `efe_total.sum().backward()` produces non-zero gradients on all input tensors.

---

## Pure Function Requirements

All three `compute_*` functions must satisfy:

1. **Side-effect free**: no writes to `self`, no state mutation, no modification of
   input tensors, no global variable access.
2. **Deterministic given same inputs**: when Monte Carlo sampling is used, accept an
   explicit `rng_seed` or `generator` argument. Given the same seed, produce the
   same output.
3. **Independently callable**: no dependencies between terms. `compute_pragmatic`
   must not require output from `compute_epistemic` or vice versa.
4. **Accept only tensor arguments and config scalars**: no `nn.Module` arguments, no
   `self` references. Pre-compute any neural network outputs and pass them as tensors.

### Why These Requirements

Enables unit testing in isolation, concurrent computation on multiple CUDA streams,
gradient isolation (no shared mutable state), and deterministic regression testing.

### Implementation Pattern

Use `@staticmethod` or module-level functions. The caller extracts parameters from
modules and passes them as tensors:

```python
# CORRECT: standalone function
def compute_pragmatic(pred_mu, pred_logvar, pref_mu, pref_logvar):
    ...

# INCORRECT: instance method reading self
class EFE(nn.Module):
    def compute_pragmatic(self, pred_mu):
        pref_mu = self.preference_mu  # Reads self -- violates purity
        ...
```

---

## Term Normalization

### Problem

The three EFE terms operate at fundamentally different scales:
- Pragmatic term scales with observation dimensionality and preference precision.
- Epistemic term scales with state dimensionality.
- Instrumental term (empowerment) is bounded by `log(action_dim)` for discrete actions.

Without normalization, one term can dominate the total EFE, making the weights
ineffective.

### Normalization Strategies

**1. Running mean/std normalization (`"running"`)**

Maintain exponential moving averages of each term's mean and standard deviation:

```python
term_normalized = (term - running_mean) / (running_std + eps)
```

Update statistics during training only. Freeze them during evaluation. Use a decay
factor of `0.99` for the moving average.

Adapts to actual scale; handles non-stationary distributions. Introduces state
(running statistics) that is not comparable across runs.

**2. Fixed range normalization via sigmoid (`"sigmoid"`)**

```python
term_normalized = 2 * sigmoid(term / temperature) - 1
```

Maps each term to `(-1, 1)`. Stateless and bounded, but saturates for extreme
values, losing gradient signal.

**3. No normalization (`"none"`)**

Apply weights directly to raw term values. Appropriate when the scales are known
and stable (e.g., after careful initialization or on well-understood problems).

### Weight Application

Apply weights **after** normalization:

```python
if normalize_terms:
    pragmatic = normalize(pragmatic_raw)
    epistemic = normalize(epistemic_raw)
    instrumental = normalize(instrumental_raw)
else:
    pragmatic = pragmatic_raw
    epistemic = epistemic_raw
    instrumental = instrumental_raw

efe_total = (w_p * pragmatic) + (w_e * epistemic) + (w_i * instrumental)
```

### Logging

Log both raw and normalized values separately:

```python
log = {
    "efe/pragmatic_raw": pragmatic_raw.mean().item(),
    "efe/pragmatic_normalized": pragmatic.mean().item(),
    "efe/epistemic_raw": epistemic_raw.mean().item(),
    "efe/epistemic_normalized": epistemic.mean().item(),
    "efe/instrumental_raw": instrumental_raw.mean().item(),
    "efe/instrumental_normalized": instrumental.mean().item(),
    "efe/total": efe_total.mean().item(),
}
```

This enables diagnosing scale mismatches and verifying that normalization is
functioning correctly.

---

## Horizon Normalization

### Problem

EFE magnitude grows linearly with the planning horizon `H`. Policies evaluated
with `H=3` are not directly comparable to policies evaluated with `H=7` because
the raw EFE totals scale proportionally.

### Mean-Per-Step Normalization

When `normalize_by_horizon = True`:

```python
efe_normalized = efe_total / H
```

This produces a per-step average EFE that is comparable across different horizon
values. Enable this by default.

### Temporal Discounting

Apply a discount factor `gamma` to reduce the influence of distant future timesteps:

```python
G(pi) = Sum_{t=1..H} gamma^(t-1) * [ w_p * pragmatic(t) + w_e * epistemic(t) + w_i * instrumental(t) ]
```

With `gamma < 1`, the effective horizon is approximately `1 / (1 - gamma)`. For
`gamma = 0.99`, this is 100 steps. For `gamma = 0.95`, this is 20 steps.

Discount and mean-per-step normalization can be combined. Apply discounting first,
then divide by the effective number of steps: `Sum_{t=0..H-1} gamma^t`.

### Configuration

Both parameters must be specified in the config and logged at the start of every
run:

```python
@dataclass
class EFEConfig:
    discount_factor: float = 0.99
    normalize_by_horizon: bool = True
```

---

## Closed-Form Test Cases

The following test cases have analytical solutions and serve as ground truth for
unit tests.

### Test Case 1: Gaussian Pragmatic Term

**Setup**: predicted observations and preferences are both Gaussian.

```
predicted:  N(mu_pred, sigma_pred^2)   with  mu_pred = [1.0, 2.0],  sigma_pred = [0.5, 0.5]
preference: N(mu_pref, sigma_pref^2)   with  mu_pref = [0.0, 0.0],  sigma_pref = [1.0, 1.0]
```

**Expected pragmatic value**:

```
pragmatic = E_{o ~ N(mu_pred, sigma_pred^2)}[ -log N(o; mu_pref, sigma_pref^2) ]
          = 0.5 * Sum_d [ (sigma_pred_d^2 + (mu_pred_d - mu_pref_d)^2) / sigma_pref_d^2
                          + log(2*pi*sigma_pref_d^2) ]
          = 0.5 * [ (0.25 + 1.0)/1.0 + log(2*pi) + (0.25 + 4.0)/1.0 + log(2*pi) ]
          = 0.5 * [ 1.25 + 1.8379 + 4.25 + 1.8379 ]
          = 0.5 * 9.1758
          = 4.5879
```

**Test assertion**: `|compute_pragmatic(...) - 4.5879| < 1e-4`.

### Test Case 2: Two-Gaussian KL (Epistemic Term)

**Setup**: posterior and prior are axis-aligned Gaussians.

```
posterior: N(mu_post, sigma_post^2)  with  mu_post = [1.0, -1.0],  sigma_post = [0.5, 0.3]
prior:     N(mu_prior, sigma_prior^2) with mu_prior = [0.0, 0.0],   sigma_prior = [1.0, 1.0]
```

**Expected epistemic value** (KL divergence):

```
KL = Sum_d [ log(sigma_prior_d / sigma_post_d)
             + (sigma_post_d^2 + (mu_post_d - mu_prior_d)^2) / (2 * sigma_prior_d^2)
             - 0.5 ]
   = [ log(1.0/0.5) + (0.25 + 1.0)/2.0 - 0.5 ]
     + [ log(1.0/0.3) + (0.09 + 1.0)/2.0 - 0.5 ]
   = [ 0.6931 + 0.625 - 0.5 ] + [ 1.2040 + 0.545 - 0.5 ]
   = 0.8181 + 1.2490
   = 2.0671
```

**Test assertion**: `|compute_epistemic(...) - 2.0671| < 1e-4`.

### Test Case 3: Known Binary Channel (Empowerment)

**Setup**: binary action space, deterministic transitions.

```
action_dim = 2
P(s' = 0 | s, a = 0) = 1.0,  P(s' = 1 | s, a = 0) = 0.0
P(s' = 0 | s, a = 1) = 0.0,  P(s' = 1 | s, a = 1) = 1.0
```

The channel is noiseless. Each action leads to a distinct state with probability 1.

**Expected empowerment**:

```
I(A; S') = H(S') - H(S'|A) = log(2) - 0 = 0.6931 nats
```

With a uniform source distribution `q(a|s) = [0.5, 0.5]`, the planning network
perfectly identifies the action from the transition, so:

```
empowerment = E[ log q(a|s,s') - log q(a|s) ]
            = E[ log(1.0) - log(0.5) ]
            = 0.6931
```

The instrumental term (negative empowerment) is `-0.6931`.

**Test assertion**: `|compute_instrumental(...) - (-0.6931)| < 1e-3`.

(Looser tolerance because the variational bound may not be perfectly tight.)

### Test Case 4: Sum Invariant Under Known Values

**Setup**: combine test cases 1-3 with unit weights.

```
pragmatic = 4.5879
epistemic = 2.0671
instrumental = -0.6931
efe_total = 4.5879 + 2.0671 + (-0.6931) = 5.9619
```

**Test assertion**: `|efe_total - sum(terms)| < 1e-5`.

This test verifies the sum invariant with known analytical values. If any term
function introduces rounding error or side effects, this test catches it.

---

## Configuration

```python
@dataclass
class EFEConfig:
    # Term weights (applied after normalization)
    pragmatic_weight: float = 1.0
    epistemic_weight: float = 1.0
    instrumental_weight: float = 0.1

    # Monte Carlo sampling
    num_samples: int = 32

    # Temporal discounting
    discount_factor: float = 0.99

    # Horizon normalization
    normalize_by_horizon: bool = True

    # Term normalization
    normalize_terms: bool = False
    term_norm_mode: str = "running"   # "running", "sigmoid", "none"
    term_norm_decay: float = 0.99     # EMA decay for running normalization
    sigmoid_temperature: float = 1.0  # Temperature for sigmoid normalization

    # Precision
    compute_dtype: str = "float32"    # Must be float32 for sum invariant

    # Debug
    assert_sum_invariant: bool = True  # Check sum invariant every forward pass
    sum_invariant_atol: float = 1e-5   # Tolerance for sum invariant check
```

### Weight Guidelines

| Scenario | pragmatic | epistemic | instrumental | Rationale |
|---|---|---|---|---|
| Pure goal pursuit | 1.0 | 0.0 | 0.0 | No exploration, no empowerment |
| Balanced (default) | 1.0 | 1.0 | 0.1 | Moderate exploration, light empowerment |
| Exploration-heavy | 0.5 | 2.0 | 0.1 | Prioritize uncertainty reduction |
| Empowerment-heavy | 0.5 | 0.5 | 1.0 | Prioritize maintaining controllability |
| Early training | 0.3 | 1.5 | 0.5 | Explore before exploiting |
| Late training | 1.5 | 0.3 | 0.1 | Exploit learned model |

Weights can be scheduled during training. A common schedule reduces epistemic weight
and increases pragmatic weight as the model becomes more confident.

---

## Common Failure Modes

| Symptom | Likely Cause | Fix |
|---|---|---|
| `sum invariant violated` assertion fires | EFE total computed via separate path from term sum | Remove separate total computation; define total as `p + e + i` only |
| Sum invariant fails intermittently | fp16/bf16 computation | Cast all EFE tensors to fp32 at function entry |
| Sum invariant fails with large batch | In-place operations causing aliasing | Replace `+=` with `x = x + y` |
| Pragmatic term is NaN | Preference `logvar` too negative (near-zero variance) | Clamp `preference_logvar >= -10` |
| Pragmatic term dominates total | Preference precision too high or obs dim too large | Enable term normalization; reduce preference precision |
| Epistemic term is always near zero | Posterior and prior are nearly identical | Check encoder is receiving gradients; verify obs carry information |
| Epistemic term is negative | Numerical error in KL computation | Clamp KL to `>= 0`; check logvar bounds |
| Instrumental term collapses to zero | Source and planning networks output same distribution | Add entropy regularization to source network; verify training data diversity |
| Instrumental term is positive | Sign convention error | Verify instrumental = negative empowerment (empowerment >= 0, instrumental <= 0) |
| EFE scale changes drastically with horizon | Horizon normalization disabled | Set `normalize_by_horizon = True` |
| Actions identical regardless of weights | Weights applied inside term functions (lost during sum) | Apply weights only at the summation point |
| Gradient is zero for epistemic term | Detached posterior in KL computation | Ensure posterior parameters are on the computation graph |
| Training diverges | Empowerment loss and EFE loss conflicting | Train empowerment networks with separate optimizer; do not backprop EFE through empowerment nets |
| All policies have same EFE | Transition model predicts same next state for all actions | Train transition model with higher capacity; add action embedding |
| Stochastic test failures | MC sampling without fixed seed | Pass explicit `torch.Generator` to all sampling operations in test mode |

---

## Summary of Invariants

1. **Sum invariant**: `|pragmatic + epistemic + instrumental - efe_total| < 1e-5` for every batch element.
2. **Pure function invariant**: all three `compute_*` functions are stateless, deterministic (given seed), and independently callable.
3. **Precision invariant**: all EFE computation uses fp32.
4. **Sign convention**: pragmatic >= 0 (penalty for deviating from preferences), epistemic >= 0 (KL is non-negative), instrumental <= 0 (negative empowerment).
5. **Weight application**: weights multiply normalized (or raw) terms at the summation point, never inside term functions.
6. **Horizon invariant**: when `normalize_by_horizon = True`, per-step EFE is stable across different `H` values for the same policy.
