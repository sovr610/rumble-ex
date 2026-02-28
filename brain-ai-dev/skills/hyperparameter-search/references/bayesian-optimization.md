# Bayesian Optimization for Hyperparameter Search

## Overview

Bayesian optimization is a sequential model-based optimization (SMBO) framework for expensive black-box functions. It builds a probabilistic surrogate model of the objective function from observed evaluations, then uses an acquisition function to decide where to evaluate next. For hyperparameter optimization of the brain_ai system, Bayesian optimization is the primary strategy for focused tuning after initial random exploration.

This document covers Tree-structured Parzen Estimators (TPE), acquisition functions, surrogate model design, Gaussian Processes vs. TPE trade-offs, and CMA-ES for continuous parameter optimization.

---

## 1. Tree-structured Parzen Estimator (TPE)

### Motivation

Standard Bayesian optimization models `p(y | x)` -- the probability of an objective value `y` given parameters `x` -- and uses this to compute an acquisition function. TPE instead models `p(x | y)` -- the probability of parameters given whether the objective is good or bad. This inversion is computationally cheaper and handles categorical/conditional parameters naturally.

### Algorithm

**Setup:**
- Let `D = {(x_1, y_1), ..., (x_n, y_n)}` be the set of observed trials.
- Define a threshold `y*` as the gamma-quantile of observed `y` values (typically gamma=0.25, meaning the top 25%).
- Split observations into two groups:
  - Good: `D_l = {x_i : y_i <= y*}` (for minimization)
  - Bad: `D_g = {x_i : y_i > y*}`
- Model two densities: `l(x) = p(x | D_l)` and `g(x) = p(x | D_g)`.

**Acquisition Function:**
TPE uses the Expected Improvement (EI) criterion, which under the TPE formulation reduces to:

```
EI(x) is proportional to l(x) / g(x)
```

The next evaluation point is:

```
x_next = argmax l(x) / g(x)
```

This is computed by: (1) sampling many candidates from `l(x)`, (2) evaluating `l(x)/g(x)` for each, (3) selecting the one with the highest ratio.

### Density Estimation

For each hyperparameter dimension, TPE fits separate 1D kernel density estimators:

**Continuous parameters:**
- Use a mixture of truncated Gaussian kernels, one centered at each observation in the respective group.
- Bandwidth is determined by a heuristic (e.g., Scott's rule) or adaptive selection.
- Truncation respects the parameter bounds.

```
l(x_j) = (1/|D_l|) * sum_{x_i in D_l} N(x_j; x_i, sigma^2) [truncated to bounds]
g(x_j) = (1/|D_g|) * sum_{x_i in D_g} N(x_j; x_i, sigma^2) [truncated to bounds]
```

**Categorical parameters:**
- Use a smoothed categorical distribution: count the frequency of each category in the group, then smooth toward uniform.

```
p(x_j = c | D_l) = (count(c in D_l) + prior) / (|D_l| + |categories| * prior)
```

**Conditional parameters:**
- When a parameter is conditional (e.g., `htm.column_count` only relevant when `use_htm=True`), TPE only includes observations where the condition is met. This naturally handles the tree structure.

### Gamma Quantile

The threshold `y*` is the gamma-quantile of observed objective values:

```
y* = quantile(y_values, gamma)
```

Where gamma controls the balance between exploration and exploitation:
- **gamma = 0.10:** Only the top 10% are "good." Very selective; exploits aggressively.
- **gamma = 0.25:** Default. Good balance.
- **gamma = 0.50:** Top half are "good." More exploratory.

For the brain_ai system, start with gamma=0.25 and increase to 0.15 as the search matures (shift toward exploitation).

### Number of Candidates

TPE generates `n_candidates` samples from `l(x)` and selects the one with highest `l(x)/g(x)`. More candidates improve the approximation of argmax but cost more compute:

- **n_candidates = 24:** Default in many implementations. Fast but may miss the true argmax.
- **n_candidates = 100:** Better coverage. Recommended for brain_ai.
- **n_candidates = 1000:** Near-optimal but slower. Use for final refinement.

### Warm-Starting

TPE can incorporate results from previous searches (different search spaces, different phases). Simply add the historical `(x, y)` pairs to the observation set `D`. This is particularly useful for:

- Resuming interrupted searches.
- Transferring knowledge between brain_ai training phases (e.g., a good learning rate in Phase 1 suggests a starting point for Phase 2).
- Narrowing the search space based on random search results.

---

## 2. Acquisition Functions

### Expected Improvement (EI)

The most widely used acquisition function. EI measures the expected amount by which a new evaluation will improve over the current best:

```
EI(x) = E[max(y* - f(x), 0)]
```

Where `y*` is the current best observed value (for minimization) and `f(x)` is the (unknown) objective at `x`.

**Under a Gaussian surrogate (GP):**
```
EI(x) = (y* - mu(x)) * Phi(z) + sigma(x) * phi(z)
where z = (y* - mu(x)) / sigma(x)
Phi = standard normal CDF
phi = standard normal PDF
```

**Under TPE:**
```
EI(x) proportional to l(x) / g(x)
```

**Properties:**
- Naturally balances exploration (high sigma regions) and exploitation (low mu regions).
- Zero at observed points.
- Degrades gracefully when the surrogate is inaccurate.

### Upper Confidence Bound (UCB)

```
UCB(x) = mu(x) + beta * sigma(x)
```

Where beta controls the exploration-exploitation trade-off:
- **beta = 1.0:** Moderate exploration.
- **beta = 2.0:** More exploration (recommended for early trials).
- **beta = sqrt(2 * log(t)):** Theoretically optimal (t = trial number), but increases monotonically.

**Properties:**
- Simple to compute.
- Explicit exploration control via beta.
- Works well with GP surrogates; less natural with TPE.

### Probability of Improvement (PI)

```
PI(x) = P(f(x) < y*)  =  Phi((y* - mu(x)) / sigma(x))
```

**Properties:**
- The simplest acquisition function.
- Tends to exploit heavily; may get stuck in local optima.
- Useful when the search is nearly converged and you want to fine-tune.

### Thompson Sampling

Draw a sample function from the posterior and optimize it:

```
f_sample ~ posterior
x_next = argmin f_sample(x)
```

**Properties:**
- Naturally explores because different samples explore different regions.
- Parallelizes well: draw k samples and optimize each one independently.
- More robust to misspecified surrogate than EI.
- Cannot be used directly with TPE (which does not model a function posterior).

### Recommendation for brain_ai

Use **Expected Improvement** via TPE as the default. This is the most well-tested combination for neural network HPO. Switch to Thompson Sampling if running many parallel trials (>4 concurrent).

---

## 3. Surrogate Models

### Gaussian Processes (GP)

A GP places a prior over functions: `f ~ GP(mu, k)`, where `mu` is the mean function and `k` is the kernel (covariance function). After observing data, the posterior is also a GP with updated mean and variance.

**Kernel Selection:**
- **Matern 5/2:** Default for HPO. Assumes the objective is twice-differentiable but not necessarily smooth. Recommended.
- **RBF (Squared Exponential):** Assumes infinite differentiability. Too smooth for most HPO objectives.
- **Matern 3/2:** Assumes only once-differentiable. Good for very noisy objectives.

**Automatic Relevance Determination (ARD):**
Use separate lengthscale parameters per dimension:

```
k(x, x') = sigma^2 * prod_j matern(|x_j - x'_j| / l_j)
```

This allows the GP to learn which parameters are important (short lengthscale = important; long lengthscale = unimportant).

**Scaling:**
- Fitting: O(n^3) per iteration (Cholesky decomposition).
- Prediction: O(n^2) per point.
- Practical limit: ~2000 observations. Beyond this, use sparse GPs or switch to TPE.

### TPE as Surrogate

TPE does not model the objective function directly; it models the parameter distributions conditioned on good/bad outcomes. This is a fundamentally different approach:

| Aspect | GP | TPE |
|--------|-----|-----|
| Models | p(y \| x) | p(x \| y) |
| Parameter types | Continuous (primarily) | Any (continuous, categorical, conditional) |
| Dimensionality | Up to ~20 well | Up to ~100 |
| Observations | Up to ~2000 | Up to ~10000 |
| Interaction modeling | Full (via kernel) | Limited (independent per dimension) |
| Uncertainty | Calibrated | Implicit |
| Parallelism | Needs batch acquisition | Sample from l(x) |

### Random Forest Surrogate (SMAC)

SMAC uses a random forest to model the objective. Predictions come from the forest's mean prediction; uncertainty from the variance across trees.

**Properties:**
- Handles categorical parameters natively.
- Scales to thousands of observations.
- Less sample-efficient than GP for continuous spaces.
- Good for large mixed spaces.

### Recommendation for brain_ai

Use **TPE** as the primary surrogate because:
1. brain_ai's config has many categorical and conditional parameters.
2. The search space can be high-dimensional (20-50 params per phase).
3. TPE scales well to hundreds of trials.
4. TPE handles conditional parameters (e.g., `use_htm` gating HTM params) natively.

Use **GP** only for final fine-tuning of 3-5 continuous parameters (e.g., learning rate, weight decay, temperature) where sample efficiency matters most.

---

## 4. Gaussian Process Details

### Prior Mean Function

Common choices:
- **Constant mean:** `mu(x) = c`. Default; works well when the objective is normalized.
- **Linear mean:** `mu(x) = w^T x + b`. Helpful when there is a known trend.
- **Zero mean with standardized observations:** Standardize y values to zero mean and unit variance. This is the recommended approach.

### Observation Noise

Real HPO objectives are noisy (different random seeds, different data orderings). Model this with additive Gaussian noise:

```
y = f(x) + epsilon, epsilon ~ N(0, sigma_noise^2)
```

`sigma_noise` can be:
- Fixed (e.g., estimated from repeated evaluations of the same configuration).
- Learned as part of the kernel hyperparameters (via marginal likelihood maximization).

### Kernel Hyperparameter Optimization

Fit kernel hyperparameters (lengthscales, signal variance, noise variance) by maximizing the log marginal likelihood:

```
log p(y | X, theta) = -0.5 * (y^T K^{-1} y + log|K| + n*log(2pi))
```

Use L-BFGS or gradient descent. Refit every 5-10 new observations to amortize the cost.

### Handling Categorical Parameters with GP

GPs naturally handle continuous parameters but not categorical ones. Options:
- **One-hot encoding:** Convert categories to binary vectors. Works for small numbers of categories. Increases dimensionality.
- **Hamming kernel:** Use a kernel that measures similarity between categorical values. `k(x, x') = 1 if x == x' else 0`. Combine with continuous kernels via product or sum.
- **Separate GPs per categorical combination:** Fit a separate GP for each unique combination of categorical parameter values. Requires enough data per combination.

### Practical GP Implementation

For brain_ai HPO, a practical GP implementation needs:
1. Matern 5/2 kernel with ARD.
2. Whitened noise.
3. L-BFGS kernel hyperparameter optimization.
4. Expected Improvement acquisition function.
5. Multi-start L-BFGS for acquisition optimization.

This is approximately 200-300 lines of code using only PyTorch (no GPyTorch/BoTorch dependency).

---

## 5. TPE Implementation Details

### Practical TPE Algorithm

```python
def suggest_tpe(observations, space, gamma=0.25, n_candidates=100):
    n_good = max(1, int(gamma * len(observations)))
    sorted_obs = sort_by_objective(observations)
    good = sorted_obs[:n_good]
    bad = sorted_obs[n_good:]

    candidates = []
    for _ in range(n_candidates):
        params = {}
        for dim in space.dimensions:
            if dim.type == 'float':
                params[dim.name] = sample_truncated_gaussian_kde(
                    good_values=[obs[dim.name] for obs in good],
                    low=dim.low, high=dim.high, log=dim.log
                )
            elif dim.type == 'int':
                params[dim.name] = sample_truncated_gaussian_kde_int(
                    good_values=[obs[dim.name] for obs in good],
                    low=dim.low, high=dim.high
                )
            elif dim.type == 'categorical':
                params[dim.name] = sample_smoothed_categorical(
                    good_values=[obs[dim.name] for obs in good],
                    choices=dim.choices
                )
        candidates.append(params)

    # Score each candidate: l(x) / g(x)
    scores = []
    for c in candidates:
        l_score = evaluate_kde(c, good, space)
        g_score = evaluate_kde(c, bad, space)
        scores.append(l_score / max(g_score, 1e-12))

    return candidates[argmax(scores)]
```

### Bandwidth Selection

For the truncated Gaussian KDE, the bandwidth (standard deviation) of each kernel determines the smoothness of the density estimate:

- **Scott's rule:** `sigma = n^{-1/(d+4)} * std(values)`. Good default for low-dimensional data.
- **Silverman's rule:** `sigma = (4/(d+2))^{1/(d+4)} * n^{-1/(d+4)} * std(values)`. Similar to Scott's.
- **Adaptive bandwidth:** Use the distance to the k-th nearest neighbor. More robust to multi-modal distributions.

For HPO, Scott's rule is sufficient because TPE fits 1D KDEs independently (d=1 per parameter).

### Handling Log-Scale Parameters

For parameters sampled on a log scale (e.g., learning rate):
1. Transform observations to log space: `log_values = log(values)`.
2. Fit the KDE in log space.
3. Sample in log space and transform back: `value = exp(log_sample)`.

This ensures the KDE respects the natural scale of the parameter.

### Handling Integer Parameters

For integer parameters (e.g., `num_timesteps`, `column_count`):
1. Treat as continuous during KDE fitting and sampling.
2. Round the sampled value to the nearest integer.
3. Clamp to `[low, high]`.

### Parallel TPE (Constant Liar)

To suggest multiple candidates simultaneously (for parallel evaluation):

1. Suggest the first candidate normally.
2. Add a "lie" observation: (candidate_1, lie_value) where lie_value is the current best (or mean, or worst).
3. Suggest the next candidate using the updated (with lie) observations.
4. Repeat.

This approximation allows suggesting k candidates without waiting for evaluations, at the cost of reduced accuracy. The "lie value" choice controls exploration:
- Best value lie: More exploitation.
- Mean value lie: Balanced.
- Worst value lie: More exploration.

---

## 6. CMA-ES for Continuous Parameters

### Algorithm Overview

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) maintains a multivariate Gaussian search distribution `N(m, sigma^2 * C)` and adapts all three components (mean `m`, step size `sigma`, covariance `C`) based on successful candidates.

### Full Algorithm

```
Input: objective f, initial mean m_0, initial sigma_0, population lambda, parents mu

Initialize:
    m = m_0, sigma = sigma_0, C = I
    p_sigma = 0, p_c = 0  (evolution paths)

For generation g = 0, 1, 2, ...:
    1. SAMPLE: x_i ~ N(m, sigma^2 * C) for i = 1..lambda
    2. EVALUATE: f_i = f(x_i) for i = 1..lambda
    3. SORT: rank x_i by f_i (ascending for minimization)
    4. SELECT: take top mu parents
    5. RECOMBINE: m_new = sum(w_i * x_i for top mu)
    6. UPDATE EVOLUTION PATHS:
       p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mu_eff) * C^{-1/2} * (m_new - m)/sigma
       p_c = (1-c_c)*p_c + h_sigma * sqrt(c_c*(2-c_c)*mu_eff) * (m_new - m)/sigma
    7. UPDATE COVARIANCE:
       C = (1-c_1-c_mu)*C + c_1*(p_c*p_c^T) + c_mu*sum(w_i * z_i*z_i^T)
    8. UPDATE STEP SIZE:
       sigma = sigma * exp((c_sigma/d_sigma) * (||p_sigma||/E[||N(0,I)||] - 1))
    9. m = m_new
```

### Key Properties

**Rotation invariance:** CMA-ES adapts the covariance matrix to match the correlation structure of the objective. A rotated objective function is optimized as efficiently as an axis-aligned one.

**Scale invariance:** The step-size adaptation mechanism ensures CMA-ES can handle objectives with different scales along different dimensions.

**Population size:** The default population size is `lambda = 4 + floor(3 * log(d))` where `d` is the dimensionality. For brain_ai HPO with ~10 continuous parameters, this gives `lambda ~ 11`.

**Number of parents:** `mu = lambda / 2`, with weighted recombination giving more weight to better candidates.

### CMA-ES for brain_ai

CMA-ES is ideal for tuning the continuous parameters within a single training phase:

| Phase | Continuous Parameters | d |
|-------|----------------------|---|
| 1 (SNN) | lr, beta, surrogate_alpha, dropout, spike_rate_target | 5 |
| 2 (Encoders) | lr, dropout, warmup_fraction | 3 |
| 3 (HTM) | permanence_inc, permanence_dec, sparsity | 3 |
| 4 (Workspace) | lr, ignition_threshold, broadcast_decay | 3 |
| 5 (Decision) | lr, epistemic_weight, empowerment_weight | 3 |
| 6 (Reasoning) | lr, confidence_threshold, ltn_p_forall | 3 |
| 7 (Meta) | inner_lr, outer_lr, trace_decay, ewc_lambda | 4 |

With d=3-5, CMA-ES converges in 50-100 evaluations, making it very practical for per-phase tuning.

### CMA-ES Limitations

- **Continuous only:** Does not handle categorical or conditional parameters. Must be combined with an outer search over categorical settings.
- **No early stopping integration:** CMA-ES requires full evaluation of each candidate. Cannot be combined with ASHA/Hyperband directly (though one could evaluate each candidate for fewer epochs).
- **Local search:** CMA-ES can get stuck in local optima. Multiple restarts with different initial means mitigate this (IPOP-CMA-ES: increasing population size on restart).

---

## 7. Trade-offs: GP vs. TPE vs. CMA-ES

### Sample Efficiency

In low dimensions (d <= 10) with continuous parameters, GP-based Bayesian optimization is the most sample-efficient, finding good solutions in 20-50 trials. TPE requires 30-80 trials. CMA-ES requires 50-150 trials.

In higher dimensions (d > 15) or mixed spaces, TPE becomes more sample-efficient because GPs struggle with the curse of dimensionality, while TPE's independent per-dimension modeling sidesteps it (at the cost of ignoring interactions).

### Computational Overhead

| Method | Per-suggestion Cost | Scales With |
|--------|-------------------|-------------|
| GP + EI | O(n^3) fit + O(n^2) predict | Observations^3 |
| TPE | O(n log n) KDE | Observations * dimensions |
| CMA-ES | O(d^2) covariance update | Dimensions^2 |

For brain_ai HPO with hundreds of trials, GP becomes prohibitively expensive unless sparse approximations are used. TPE is practical up to thousands of trials.

### Robustness

- **GP:** Sensitive to kernel choice and hyperparameter optimization. Can produce poor suggestions if the kernel is misspecified or the objective is very noisy.
- **TPE:** Robust to noise and misspecification. The worst case is degradation toward random search. Rarely produces pathologically bad suggestions.
- **CMA-ES:** Robust for smooth continuous objectives. Can fail on highly multi-modal landscapes (use restarts).

### Recommendation Matrix

| Scenario | Recommendation |
|----------|---------------|
| First HPO run, unknown space | Random Search + ASHA |
| Focused tuning, <10 continuous params | CMA-ES |
| Mixed space, >10 params | TPE |
| Very expensive objective (hours/trial) | GP + EI |
| Many parallel workers | Thompson Sampling (GP) or Parallel TPE |
| Phase-specific brain_ai tuning | TPE + ASHA |
| Final fine-tuning, 2-3 params | GP + EI or fine grid search |

---

## 8. Multi-Fidelity Bayesian Optimization

### Combining Bayesian Optimization with ASHA

The most effective HPO strategy combines intelligent sampling (Bayesian optimization) with efficient evaluation (successive halving). The workflow is:

1. TPE suggests a candidate configuration.
2. The candidate is evaluated at the lowest resource level (1 epoch).
3. ASHA decides whether to promote or prune based on the 1-epoch result.
4. Promoted candidates are evaluated at higher resource levels (3, 9, 27, ... epochs).
5. All observations (including pruned, partially-evaluated trials) are fed back to TPE.

### Handling Partially-Evaluated Trials

When a trial is pruned after k epochs, its objective value is the k-epoch metric, which may not be directly comparable to a trial that ran for all epochs. Options:

- **Use the final available metric:** Simple but biased (early-epoch metrics tend to be worse).
- **Predict final performance from partial trajectory:** Fit a learning curve model (e.g., power law) to the k observations and extrapolate. More accurate but adds complexity.
- **Separate by resource level:** Maintain separate TPE models per resource level. Use the most informative level for suggestion.

For brain_ai, use the first option (final available metric) with a small penalty for pruned trials (e.g., add 10% to the loss). This is simple, robust, and avoids over-engineering.

---

## 9. Practical Implementation Checklist

1. **Define the search space** using `SearchSpace` with proper bounds, log-scale flags, and conditionals.
2. **Implement the objective function** that takes a parameter dict, constructs a `BrainAIConfig`, trains for N epochs, and returns a metric.
3. **Initialize TPE** with gamma=0.25, n_candidates=100.
4. **Warm-start** with any prior knowledge (LR finder results, previous searches).
5. **Run n_init=10 random trials** to bootstrap the TPE model.
6. **Run TPE + ASHA** for the remaining budget.
7. **Analyze results:** parameter importance, convergence plot, best configuration.
8. **Validate:** Re-train the best configuration with 3 different seeds to confirm robustness.

---

## Summary

For the brain_ai system:
- **TPE** is the recommended primary Bayesian optimization method due to its handling of categorical and conditional parameters, scalability, and robustness.
- **GP** should be reserved for final fine-tuning of small continuous subsets.
- **CMA-ES** excels at per-phase continuous parameter optimization.
- **Combine with ASHA** for resource-efficient evaluation.
- Always warm-start from LR finder results and prior search data.
