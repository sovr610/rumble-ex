# Search Strategies for Hyperparameter Optimization

## Overview

Hyperparameter optimization (HPO) is the process of selecting the best configuration for a machine learning model from a defined search space. For the brain_ai system with its 10-dataclass configuration (`BrainAIConfig`), the search space can be enormous -- hundreds of parameters across SNN, encoder, HTM, workspace, decision, reasoning, meta-learning, and engram subsystems. Choosing the right search strategy is critical for efficient use of compute.

This document covers four main families of search strategies: grid search, random search, Bayesian optimization, and successive halving methods (Hyperband/ASHA). Each has distinct convergence properties, computational requirements, and applicability profiles.

---

## 1. Grid Search

### Algorithm Description

Grid search exhaustively evaluates every combination of hyperparameter values from a predefined discrete set. Given `k` hyperparameters, each with `n_i` candidate values, grid search evaluates `n_1 * n_2 * ... * n_k` configurations.

**Pseudocode:**
```
for each combination (v_1, ..., v_k) in cartesian_product(values_1, ..., values_k):
    result = evaluate(v_1, ..., v_k)
    record(result)
return best(results)
```

### Convergence Properties

- **Deterministic:** Always evaluates the same set of points, guaranteeing reproducibility.
- **Complete coverage of specified grid:** Every point is evaluated; nothing is missed within the grid.
- **No convergence in the traditional sense:** The search terminates when all points are evaluated, not when a convergence criterion is met.
- **Resolution-dependent:** Quality depends entirely on the grid resolution. If the optimal value falls between grid points, it will be missed.
- **Curse of dimensionality:** The number of evaluations grows exponentially with the number of parameters: O(n^k). With 5 parameters and 10 values each, that is 100,000 evaluations.

### When to Use

- **Few parameters (1-3):** Grid search is practical and thorough when the search space is small.
- **Discrete, bounded choices:** When parameters naturally take on a small set of values (e.g., `surrogate` in `{atan, fast_sigmoid, straight_through}`).
- **Validation/confirmation:** After narrowing down with other methods, grid search on the final 2-3 parameters provides thorough coverage.
- **Interaction effects:** Grid search reveals full interaction patterns between parameters, useful for understanding the loss landscape.

### Limitations for brain_ai

The brain_ai configuration has hundreds of tunable parameters. Even selecting 5 key parameters with 5 values each yields 3,125 trials -- feasible but slow. Grid search should be reserved for phase-specific tuning of 2-4 critical parameters after initial exploration with faster methods.

---

## 2. Random Search

### Algorithm Description

Random search independently samples each hyperparameter from its marginal distribution (uniform, log-uniform, categorical) for a fixed number of trials. This was shown by Bergstra and Bengio (2012) to be surprisingly effective compared to grid search.

**Pseudocode:**
```
for trial in range(n_trials):
    params = {name: sample(distribution) for name, distribution in space.items()}
    result = evaluate(params)
    record(result)
return best(results)
```

### Convergence Properties

- **Probabilistic coverage:** With n trials, the probability of sampling within epsilon of the optimum along any single dimension is approximately `1 - (1 - epsilon)^n`.
- **No curse of dimensionality on effective dimensions:** If only d_eff out of D parameters matter, random search explores the important d_eff dimensions as efficiently as searching a d_eff-dimensional space, unlike grid search which wastes budget on unimportant dimensions.
- **Asymptotic optimality:** As n approaches infinity, random search converges to the global optimum. In practice, 50-200 trials give good results for moderately-sized spaces.
- **Embarrassingly parallel:** Every trial is independent, allowing perfect parallelism across GPUs or machines.
- **Variance decreases as O(1/n):** Expected regret (gap between best found and true optimum) decreases inversely with the number of trials.

### Why Random Beats Grid

Consider a 2D search where only one parameter matters. Grid search with a 10x10 grid explores only 10 unique values of the important parameter. Random search with 100 trials explores 100 unique values. This advantage grows with dimensionality: in 10D with 10 values per parameter, grid needs 10 billion evaluations while random gets 100 unique values per dimension with just 100 trials.

### When to Use

- **Initial exploration:** Random search is the best starting point for any HPO campaign. It quickly identifies promising regions of the space.
- **High-dimensional spaces (>5 parameters):** The key advantage over grid search.
- **Parallel compute available:** Run 50-200 independent trials simultaneously.
- **Unknown parameter importance:** When you don't know which parameters matter most, random search is more efficient than grid search because it does not waste budget on unimportant dimensions.
- **Baseline comparison:** Always run random search as a baseline; if Bayesian optimization can't beat it, the objective is likely noisy or the space is poorly defined.

### Recommended Settings for brain_ai

For phase-specific tuning (3-7 key parameters per phase), 100 random trials provide a strong baseline. For cross-phase optimization, 200+ trials are recommended. Always use log-uniform sampling for learning rates and other scale parameters.

---

## 3. Bayesian Optimization

### Algorithm Description

Bayesian optimization uses a surrogate model to approximate the objective function, then selects the next evaluation point by maximizing an acquisition function that balances exploration (sampling uncertain regions) and exploitation (sampling near known good regions).

**General Pseudocode:**
```
initialize: evaluate n_init random points
for trial in range(n_init, n_trials):
    fit surrogate_model on observed (params, results)
    next_params = argmax acquisition_function(params; surrogate_model)
    result = evaluate(next_params)
    record(result)
return best(results)
```

### Surrogate Models

#### Gaussian Processes (GP)

Gaussian processes model the objective as a multivariate Gaussian distribution. They provide both a mean prediction and uncertainty estimate at any point.

- **Strengths:** Well-calibrated uncertainty; principled; works very well in low dimensions (up to ~15 parameters).
- **Weaknesses:** O(n^3) fitting cost per iteration (cubic in number of observed points); struggles with >20 dimensions; requires careful kernel selection; assumes smooth objective.
- **Best for:** Low-dimensional continuous spaces; very expensive objective functions (hours per evaluation).

#### Tree-structured Parzen Estimator (TPE)

TPE is a non-parametric Bayesian approach that models the density of good configurations (`l(x)`) and bad configurations (`g(x)`) separately, then samples from `l(x)/g(x)`.

- **Strengths:** Handles categorical and conditional parameters naturally; O(n log n) per iteration; scales to high dimensions; robust to noisy objectives.
- **Weaknesses:** Less sample-efficient than GP in low dimensions; does not model parameter interactions as well.
- **Best for:** Mixed spaces (continuous + categorical + conditional); moderate to high dimensions; the brain_ai system's mixed config types.

See `references/bayesian-optimization.md` for detailed algorithm descriptions.

### Acquisition Functions

- **Expected Improvement (EI):** Most common; selects points with highest expected improvement over the current best. Balances exploration/exploitation naturally.
- **Upper Confidence Bound (UCB):** Selects points where `mean + beta * std` is maximized. The beta parameter controls exploration.
- **Probability of Improvement (PI):** Selects the point most likely to improve. Tends to exploit more than EI.
- **Thompson Sampling:** Draws a sample from the posterior and optimizes it. Natural exploration; parallelizes well.

### Convergence Properties

- **Sample efficient:** Typically finds near-optimal solutions in 20-100 trials, compared to hundreds for random search.
- **Superlinear convergence near optimum:** Once the surrogate model is accurate in the promising region, Bayesian optimization converges faster than random search.
- **Depends on surrogate quality:** If the surrogate model is a poor fit (very high noise, many discontinuities, extreme dimensionality), convergence degrades toward random search.
- **Cold start problem:** The first n_init trials (typically 10-20) are essentially random; Bayesian optimization only starts to help after enough data to fit a useful surrogate.

### When to Use

- **Expensive objective functions:** When each trial takes minutes to hours (training a neural network for multiple epochs).
- **Budget-constrained searches:** When you can only afford 20-100 trials total.
- **Focused tuning after random exploration:** Use random search to narrow the space, then Bayesian optimization to find the precise optimum.
- **Per-phase tuning in brain_ai:** Each training phase (SNN, encoders, HTM, etc.) benefits from Bayesian optimization because trial cost is high.

---

## 4. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

### Algorithm Description

CMA-ES is an evolutionary optimization algorithm that maintains a multivariate Gaussian distribution over the search space and adapts both the mean and covariance matrix based on successful candidates.

**Pseudocode:**
```
initialize: mean = center_of_space, C = identity, sigma = initial_step
for generation in range(max_generations):
    samples = [sample_multivariate_normal(mean, sigma^2 * C) for _ in range(population_size)]
    results = [evaluate(s) for s in samples]
    sorted_samples = sort_by_result(samples, results)
    mean = weighted_average(top_mu_samples)
    update C based on evolution paths
    update sigma based on path length
return mean
```

### Convergence Properties

- **Rotation invariant:** Adapts to any correlation structure in the objective.
- **Linear convergence rate on quadratic functions:** Optimal for smooth, continuous objectives.
- **Population-based exploration:** Maintains diversity; less likely to get stuck in local optima than gradient-based methods.
- **Works well for 5-50 continuous dimensions:** Below GP's scaling limit but much more efficient than random search.

### When to Use

- **Continuous parameters only:** CMA-ES does not naturally handle categorical parameters.
- **Moderate dimensionality (5-50):** Excellent for tuning all continuous parameters within a single brain_ai phase.
- **Smooth objective landscape:** Best when the objective is relatively smooth (not dominated by noise).
- **No gradient available:** CMA-ES is derivative-free; useful when the objective is a black box (e.g., downstream task accuracy).

---

## 5. Hyperband and ASHA (Successive Halving)

### Algorithm Description

Successive halving allocates a fixed budget across trials, evaluating all trials at a low resource level (e.g., 1 epoch), then promoting only the top 1/eta fraction to the next resource level (e.g., 3 epochs), and repeating until one trial remains.

**Successive Halving Pseudocode:**
```
trials = sample_n_random_configs(N)
resource = min_resource
while len(trials) > 1:
    evaluate each trial for (resource) units
    keep top 1/eta fraction of trials
    resource *= eta
return best(trials)
```

**Hyperband** runs multiple brackets of successive halving with different trade-offs between number of initial trials and minimum resource per trial. It hedges against the uncertainty of how much resource is needed to distinguish good from bad configurations.

**ASHA (Asynchronous Successive Halving Algorithm)** is the asynchronous version. Instead of waiting for all trials at a rung to complete before promoting, ASHA promotes trials as soon as enough results are available at each rung. This enables much better GPU utilization.

### Convergence Properties

- **Speedup over random search:** Up to `sqrt(n)` speedup by early stopping bad configurations. A trial that would take 100 epochs is terminated after 1 epoch if it is in the bottom 2/3.
- **Assumption-dependent:** Assumes that relative ordering of trials is preserved across resource levels. If a trial that performs poorly at 1 epoch eventually becomes the best at 100 epochs, successive halving will incorrectly prune it.
- **Resource efficient:** Total compute is O(N * max_resource * log_eta(max_resource / min_resource)) rather than O(N * max_resource) for full evaluation of all trials.
- **Combines well with any sampler:** Use Bayesian optimization (TPE) to generate candidate configurations, then use ASHA to efficiently evaluate them.

### Hyperband Brackets

Hyperband runs `s_max + 1` brackets, where `s_max = floor(log_eta(max_resource / min_resource))`. Each bracket `s` starts with `ceil(n * eta^s / (s+1))` trials at resource `max_resource / eta^s`.

Example with eta=3, min_resource=1, max_resource=81:
- Bracket 4: 81 trials, starting at 1 unit
- Bracket 3: 34 trials, starting at 3 units
- Bracket 2: 15 trials, starting at 9 units
- Bracket 1: 8 trials, starting at 27 units
- Bracket 0: 5 trials, starting at 81 units

This ensures robust performance regardless of whether early stopping is helpful for the particular problem.

### When to Use

- **Large search spaces with cheap-to-evaluate proxy:** When you can get a useful signal from partial training (1-5 epochs instead of 50-100).
- **Many candidate configurations:** Hyperband/ASHA shine when you can start 100+ trials and aggressively prune.
- **Mixed with Bayesian sampling:** The combination of TPE sampling + ASHA pruning is extremely effective: TPE selects promising candidates, ASHA filters out the bad ones cheaply.
- **brain_ai phase training:** Each phase has a natural resource axis (epochs). Start many configurations at 1 epoch, keep the top third for 3 epochs, then 9, then 27 epochs.

---

## Strategy Selection Guide

### Decision Flowchart

```
Is the space discrete with <100 total combinations?
  YES -> Grid Search
  NO  ->
    Is this initial exploration (no prior knowledge)?
      YES -> Random Search (100-200 trials)
      NO  ->
        Are parameters mostly continuous?
          YES and <20 params -> CMA-ES or GP-based Bayesian
          YES and >20 params -> TPE with ASHA
          NO (mixed types)  -> TPE with ASHA
```

### Recommended Pipeline for brain_ai

**Step 1: LR Range Test** -- Before any search, run the LR finder on the current phase's model to establish learning rate bounds. This removes one of the most impactful parameters from the search.

**Step 2: Random Exploration** -- Run 50-100 random trials with ASHA early stopping (min_resource=1 epoch, eta=3). This quickly identifies which parameters matter and establishes a baseline.

**Step 3: Importance Analysis** -- Use fANOVA or simple correlation analysis on the random search results to identify the 3-5 most important parameters.

**Step 4: Bayesian Refinement** -- Run 50-100 TPE trials focused on the important parameters, with ASHA pruning. Use the random search results as warm-start data.

**Step 5: Fine Grid Search** -- For the final 2-3 most impactful parameters, run a fine grid search around the best values found by TPE to ensure thorough coverage.

### Cost Comparison

| Strategy | Trials Needed | Parallelism | Wall-clock (relative) | Quality |
|----------|:------------:|:-----------:|:--------------------:|:-------:|
| Grid (5 params, 5 values) | 3,125 | Perfect | 1.0x | Thorough on grid |
| Random | 100-200 | Perfect | 0.03-0.06x | Good |
| TPE (Bayesian) | 50-100 | Limited | 0.02-0.03x | Very good |
| TPE + ASHA | 200 (cheap) | Perfect | 0.01-0.02x | Excellent |
| CMA-ES | 50-200 | Population | 0.02-0.06x | Very good (continuous) |

---

## Implementation Considerations

### Handling Conditional Parameters

The brain_ai config has conditional parameters -- for example, HTM parameters (`column_count`, `cells_per_column`) are only relevant when `use_htm=True`. Search strategies must handle this:

- **Random search:** Sample the boolean flag first; if False, skip HTM parameters.
- **TPE:** Natively handles conditionals by modelling `l(x)` and `g(x)` only over configurations where the condition is met.
- **Grid search:** Enumerate the conditional tree; each branch is a separate sub-grid.
- **CMA-ES:** Does not handle conditionals natively; must be run separately for each boolean combination.

### NaN and Divergence Handling

Training can diverge (NaN loss). Search strategies must handle this:

- Report NaN as the worst possible value (infinity for minimization, negative infinity for maximization).
- Track divergence rate; if >50% of trials diverge, the search space bounds are likely too aggressive.
- In TPE, NaN results contribute to `g(x)` (bad configurations), helping the sampler avoid similar regions.

### Reproducibility

Every trial must be fully reproducible:

- Record the exact parameter configuration.
- Use deterministic seeding (seed = base_seed + trial_id).
- Record the software environment (torch version, CUDA version).
- Store intermediate checkpoints for the best trials.
- Integrate with the training-orchestrator's RunManifest system.

### Parallel Considerations

- **Random search and grid search:** Embarrassingly parallel; distribute trials across GPUs.
- **TPE:** Sequential by nature (each trial depends on previous results), but can suggest multiple candidates at once using constant liar or batch acquisition strategies.
- **ASHA:** Designed for asynchronous parallel execution; trials are promoted independently.
- **CMA-ES:** Population can be evaluated in parallel within each generation; generations are sequential.

---

## Summary

For the brain_ai system with its complex, multi-phase training pipeline and rich configuration surface, the recommended approach is:

1. Use the **LR finder** to establish learning rate bounds for each phase.
2. Use **random search + ASHA** for initial broad exploration (100-200 cheap trials).
3. Use **TPE + ASHA** for focused refinement on important parameters (50-100 trials).
4. Use **grid search** for final fine-tuning of 2-3 critical parameters.
5. Use **CMA-ES** for continuous parameter optimization within individual phases when the space is well-understood.

The combination of these strategies, applied phase-by-phase, provides the best trade-off between compute cost and solution quality.
