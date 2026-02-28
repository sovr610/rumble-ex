# pymdp Integration

## Overview

When `pymdp` (`inferactively-pymdp >= 0.0.7.1`) is installed, the active inference agent gains
an alternate discrete POMDP backend that constructs A/B/C/D arrays from neural model parameters
and delegates planning to pymdp's exact routines. The primary purpose is **regression testing**:
verify that the neural EFE implementation produces results consistent with exact discrete
computations on toy problems where ground truth is known. A secondary purpose is debugging --
discrete arrays are directly inspectable, making it straightforward to isolate whether a
misbehavior originates in the generative model, the EFE decomposition, or the planner.

The backend is off by default. Enable it only for testing and validation. Production inference
always uses the neural continuous pipeline (`ImprovedActiveInferenceAgent` in
`brain_ai/decision/active_inference.py`).

## pymdp Library Background

pymdp implements active inference for discrete state-space POMDPs. Install it with:

```bash
pip install inferactively-pymdp
```

The library centers on four numpy arrays that define a generative model:

| Array | Name | Shape (single factor) | Semantics |
|-------|------|----------------------|-----------|
| **A** | Likelihood | `(num_obs, num_states)` | `A[o, s] = P(o \| s)` -- probability of observing `o` when in state `s` |
| **B** | Transition | `(num_states, num_states, num_actions)` | `B[s', s, a] = P(s' \| s, a)` -- state transition under action `a` |
| **C** | Preferences | `(num_obs,)` | `C[o] = log P_pref(o)` -- log prior preferences over observations |
| **D** | Prior | `(num_states,)` | `D[s] = P(s_0)` -- prior beliefs about the initial state |

For multi-factor or multi-modality problems, each array becomes a **list** of arrays (one per
factor or modality). pymdp normalizes columns of A, normalizes slices of B, and softmaxes C
internally.

Key pymdp classes and functions used by the backend:

- `pymdp.agent.Agent` -- wraps the generative model, performs belief updating via variational
  message passing or marginal message passing (MMP), evaluates policies by expected free energy,
  and selects actions.
- `pymdp.utils.random_A_matrix(num_obs, num_states)` -- generates a random normalized likelihood.
- `pymdp.utils.random_B_matrix(num_states, num_actions)` -- generates a random normalized transition.
- `pymdp.utils.obj_array_uniform(shape_list)` -- creates uniform Dirichlet-like arrays.
- `pymdp.maths.spm_log_single(arr)` -- safe log avoiding -inf.

pymdp computes EFE with an explicit **risk + ambiguity** decomposition:

- **Risk** (pragmatic value): `E_q(o|pi)[ log q(o|pi) - log C(o) ]` -- KL divergence from
  predicted observations to preferred observations.
- **Ambiguity** (epistemic value): `E_q(s|pi)[ H[P(o|s)] ]` -- expected entropy of the
  likelihood under the posterior, measuring how uncertain observations remain even when the
  state is known.

## Array Construction

Build pymdp-compatible arrays from the neural model by discretizing its continuous spaces.
Each subsection below describes one array.

### A Matrix (Likelihood) -- P(o|s)

Construct the A matrix by querying the neural likelihood model `P(o|s)` at each discrete
state center and binning the predicted observations.

1. Obtain state cluster centers `{c_1, ..., c_K}` (see Discretization Strategy below).
2. Obtain observation bin centers `{b_1, ..., b_M}`.
3. For each state center `c_k`, pass it through the neural likelihood model to get predicted
   observation parameters (mean, variance).
4. For each observation bin `b_m`, evaluate the predicted density at `b_m` given state `c_k`.
   Assign `A[m, k] = P(o in bin_m | s = c_k)`.
5. Normalize each column so that `sum_m A[m, k] = 1` for all `k`.

For multi-modal observations (e.g., vision + proprioception), build one A matrix per modality
and store them as a list.

```python
def build_A_matrix(likelihood_model, state_centers, obs_bin_centers):
    num_obs = len(obs_bin_centers)
    num_states = len(state_centers)
    A = np.zeros((num_obs, num_states))

    for k, s_center in enumerate(state_centers):
        s_tensor = torch.tensor(s_center, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_obs = likelihood_model(s_tensor).squeeze(0).numpy()
        # Compute distance from prediction to each bin center
        for m, o_center in enumerate(obs_bin_centers):
            dist_sq = np.sum((pred_obs - o_center) ** 2)
            A[m, k] = np.exp(-0.5 * dist_sq)  # Gaussian kernel
    # Normalize columns
    A = A / (A.sum(axis=0, keepdims=True) + 1e-12)
    return A
```

### B Matrix (Transition) -- P(s'|s,a)

Construct the B matrix by querying the neural transition model `P(s'|s,a)` for each
(state, action) pair.

1. For each state center `c_k` and each action `a`:
   - Pass `(c_k, a)` through the neural transition model to get next-state distribution
     parameters `(mu_{k,a}, sigma_{k,a})`.
   - For each next-state center `c_j`, evaluate the predicted density:
     `B[j, k, a] = P(s' in bin_j | s = c_k, a)`.
2. Normalize each column so that `sum_j B[j, k, a] = 1` for all `(k, a)`.

```python
def build_B_matrix(transition_model, state_centers, num_actions):
    K = len(state_centers)
    B = np.zeros((K, K, num_actions))

    for k, s_center in enumerate(state_centers):
        s_tensor = torch.tensor(s_center, dtype=torch.float32).unsqueeze(0)
        for a in range(num_actions):
            a_onehot = torch.zeros(1, num_actions)
            a_onehot[0, a] = 1.0
            with torch.no_grad():
                mu, log_var = transition_model(s_tensor, a_onehot)
            mu_np = mu.squeeze(0).numpy()
            std_np = np.exp(0.5 * log_var.squeeze(0).numpy())
            for j, s_next in enumerate(state_centers):
                dist_sq = np.sum(((s_next - mu_np) / (std_np + 1e-8)) ** 2)
                B[j, k, a] = np.exp(-0.5 * dist_sq)
    # Normalize
    B = B / (B.sum(axis=0, keepdims=True) + 1e-12)
    return B
```

### C Vector (Preferences)

Construct the C vector by evaluating the neural preference model at each observation bin center.

1. For each observation bin center `b_m`, compute `C[m] = log P_pref(b_m)` using the neural
   preference model.
2. Subtract the maximum for numerical stability: `C = C - max(C)`.

The neural `Preferences` module in `brain_ai/decision/active_inference.py` stores learnable
preference vectors. Evaluate pragmatic value at each bin center and use the resulting scores
as log-preferences.

```python
def build_C_vector(preference_model, obs_bin_centers):
    C = np.zeros(len(obs_bin_centers))
    for m, o_center in enumerate(obs_bin_centers):
        o_tensor = torch.tensor(o_center, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            C[m] = preference_model.compute_pragmatic_value(o_tensor).item()
    C = C - C.max()  # Numerical stability
    return C
```

### D Vector (Prior)

Construct D from the neural encoder's prior over initial states.

1. If the encoder uses a learned prior, evaluate its density at each state center.
2. If the prior is standard normal, compute `D[k] = N(c_k; 0, I)` for each center.
3. Normalize so that `sum_k D[k] = 1`.

For most configurations, a uniform prior is sufficient for regression testing:

```python
def build_D_vector(num_states):
    return np.ones(num_states) / num_states
```

## Discretization Strategy

The neural model operates in continuous spaces. Discretization maps these to finite state/observation/action sets for pymdp.

**State space discretization:**
1. Collect latent state samples by running the encoder on a representative dataset (1000+ samples).
2. Apply K-means clustering with `K = num_discrete_states` (default: 16).
3. Store cluster centers as the discrete state vocabulary.

**Observation space discretization:**
1. Collect observation embeddings from the same dataset.
2. Apply K-means with `M = num_discrete_obs` (default: 16).
3. Store cluster centers as the discrete observation vocabulary.

**Action space:**
- For discrete action agents: use the native action indices directly -- no discretization needed.
- For continuous action agents: discretize into a uniform grid over the action range, or use
  K-means on action samples from the replay buffer. Default: 10 action bins.

**Number of bins:** Choose based on the accuracy/computation tradeoff. More bins improve
approximation fidelity but increase pymdp computation cost (which scales as
`O(K^2 * num_actions * horizon)`). Recommended defaults:

| Space | Default bins | Min (fast) | Max (accurate) |
|-------|-------------|------------|----------------|
| States | 16 | 8 | 64 |
| Observations | 16 | 8 | 64 |
| Actions | 10 | 4 | 32 |

**Cache cluster centers** across evaluations to ensure consistent discretization. Store them
alongside the model checkpoint or in a dedicated `.npz` file. Recompute only when the model
architecture or training data changes substantially.

## PyMDPBackend Class

The backend wraps all pymdp interactions behind a single class.

```python
class PyMDPBackend:
    """Discrete POMDP backend using pymdp for regression testing."""

    def __init__(
        self,
        num_states: int = 16,
        num_obs: int = 16,
        num_actions: int = 10,
        planning_horizon: int = 3,
        inference_algo: str = "MMP",
    ):
        if not PYMDP_AVAILABLE:
            raise ImportError(
                "pymdp is required for PyMDPBackend. "
                "Install with: pip install inferactively-pymdp"
            )
        self.num_states = num_states
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.planning_horizon = planning_horizon
        self.inference_algo = inference_algo
        self.agent = None
        self.state_centers = None
        self.obs_centers = None

    def build_arrays(
        self, neural_model, dataset, num_calibration_samples=1000
    ) -> tuple:
        """
        Estimate discrete A/B/C/D arrays from neural model.

        Args:
            neural_model: The neural ActiveInferenceAgent or ImprovedActiveInferenceAgent.
            dataset: Iterable of observation tensors for calibration.
            num_calibration_samples: Number of samples for K-means fitting.

        Returns:
            Tuple of (A, B, C, D) numpy arrays.
        """
        # 1. Collect latent states and observations for clustering
        # 2. Fit K-means for state and observation spaces
        # 3. Build A, B, C, D using the functions described above
        # 4. Cache cluster centers
        ...
        return A, B, C, D

    def create_agent(self, A, B, C, D) -> "PyMDPAgent":
        """Instantiate a pymdp Agent from arrays."""
        self.agent = PyMDPAgent(
            A=[A], B=[B], C=[C], D=[D],
            policy_len=self.planning_horizon,
            inference_algo=self.inference_algo,
            use_states_info_gain=True,
            action_selection="stochastic",
        )
        return self.agent

    def compute_efe_discrete(self, A, B, C, policies) -> np.ndarray:
        """
        Compute exact EFE for each policy using pymdp internals.

        Args:
            A: Likelihood array.
            B: Transition array.
            C: Preference vector.
            policies: Array of shape (num_policies, horizon) with action indices.

        Returns:
            efe_per_policy: Array of shape (num_policies,) with EFE values.
        """
        # Use pymdp.inference or agent internals to compute G for each policy
        ...
        return efe_per_policy

    def select_action(self, observation_idx: int) -> tuple:
        """
        Run a full pymdp agent step: observe, update beliefs, select action.

        Args:
            observation_idx: Discrete observation index.

        Returns:
            Tuple of (action_idx, info_dict).
        """
        assert self.agent is not None, "Call create_agent() first."
        obs_onehot = pymdp_utils.onehot(observation_idx, self.num_obs)
        qs = self.agent.infer_states([obs_onehot])
        q_pi, efe = self.agent.infer_policies()
        action = self.agent.sample_action()
        return int(action[0]), {
            "beliefs": qs,
            "policy_probs": q_pi,
            "efe": efe,
        }

    def compare_efe(
        self,
        neural_efe: np.ndarray,
        discrete_efe: np.ndarray,
        tolerance: float = 0.1,
    ) -> dict:
        """
        Compare neural EFE against discrete reference.

        Both arrays are min-max normalized to [0, 1] before comparison.

        Args:
            neural_efe: EFE values from neural model, shape (num_policies,).
            discrete_efe: EFE values from pymdp, shape (num_policies,).
            tolerance: Maximum allowed normalized difference.

        Returns:
            Dict with correlation, max_diff, ranking_agreement, and pass/fail.
        """
        # Normalize both to [0, 1]
        def normalize(x):
            r = x.max() - x.min()
            if r < 1e-12:
                return np.zeros_like(x)
            return (x - x.min()) / r

        n_neural = normalize(neural_efe)
        n_discrete = normalize(discrete_efe)

        correlation = float(np.corrcoef(n_neural, n_discrete)[0, 1])
        max_diff = float(np.max(np.abs(n_neural - n_discrete)))

        # Ranking agreement: same top-3 actions
        top3_neural = set(np.argsort(neural_efe)[:3])
        top3_discrete = set(np.argsort(discrete_efe)[:3])
        ranking_agreement = len(top3_neural & top3_discrete) / 3.0

        passed = correlation > 0.9 and max_diff < tolerance
        return {
            "correlation": correlation,
            "max_diff": max_diff,
            "ranking_agreement": ranking_agreement,
            "passed": passed,
        }
```

Place this class in `brain_ai/decision/pymdp_backend.py`. The entire module body must be
guarded: define only a stub if `PYMDP_AVAILABLE` is False.

## Regression Testing Protocol

Follow this six-step protocol to validate that the neural EFE implementation matches
discrete ground truth.

### Step 1: Define a Toy Problem

Use a small, fully specified POMDP where ground-truth arrays are known analytically.
Recommended: a 4-state, 4-observation, 3-action grid world.

```python
# Ground-truth arrays for a 4-state T-maze
num_states, num_obs, num_actions = 4, 4, 3

A_true = np.array([
    [0.9, 0.05, 0.05, 0.0],
    [0.05, 0.9, 0.0, 0.05],
    [0.05, 0.0, 0.9, 0.05],
    [0.0, 0.05, 0.05, 0.9],
])

B_true = np.zeros((4, 4, 3))
B_true[:, :, 0] = np.array([  # Action 0: move left
    [1.0, 0.8, 0.0, 0.0],
    [0.0, 0.2, 0.0, 0.0],
    [0.0, 0.0, 0.2, 0.0],
    [0.0, 0.0, 0.8, 1.0],
])
# ... define actions 1, 2 similarly

C_true = np.array([0.0, 3.0, 0.0, -3.0])  # Prefer observation 1, avoid observation 3
D_true = np.array([0.25, 0.25, 0.25, 0.25])
```

### Step 2: Build the Exact Discrete Reference

Pass ground-truth arrays directly to pymdp.

```python
backend = PyMDPBackend(num_states=4, num_obs=4, num_actions=3, planning_horizon=3)
agent = backend.create_agent(A_true, B_true, C_true, D_true)
```

### Step 3: Train the Neural Model on the Same Problem

Train the neural generative model (encoder, likelihood, transition, preferences) to
approximate the ground-truth arrays. Use supervised learning: sample trajectories from the
discrete model, convert to continuous representations, and train until reconstruction loss
converges.

### Step 4: Extract Discrete Arrays from the Neural Model

Use `backend.build_arrays(neural_model, calibration_dataset)` to discretize the trained
neural model back into A/B/C/D arrays. Use the known state/observation centers from the
toy problem (skip K-means; pass centers directly for maximum fidelity).

### Step 5: Compare EFE Values

Compute EFE under both backends for the same set of policies.

```python
import itertools
policies = np.array(list(itertools.product(range(num_actions), repeat=horizon)))

discrete_efe = backend.compute_efe_discrete(A_true, B_true, C_true, policies)
neural_efe = compute_neural_efe(neural_model, policies, state_centers, obs_centers)

result = backend.compare_efe(neural_efe, discrete_efe, tolerance=0.1)
```

### Step 6: Assert Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Total EFE correlation | > 0.9 | Overall monotonic agreement |
| Pragmatic term correlation | > 0.85 | Preference alignment consistency |
| Epistemic term correlation | > 0.85 | Uncertainty estimation consistency |
| Policy ranking agreement (top-3) | >= 2/3 overlap | Same best actions selected |
| Max normalized difference | < 0.1 | No catastrophic outliers |

If any threshold is violated, the test fails and logs the full comparison dict for debugging.

## EFE Decomposition Mapping

The neural model uses a three-term EFE decomposition (pragmatic + epistemic + instrumental).
pymdp uses a two-term decomposition (risk + ambiguity). Map between them as follows:

| Neural Term | pymdp Term | Relationship |
|-------------|-----------|--------------|
| Pragmatic value | Risk | Both measure how much predicted observations deviate from preferences. The neural pragmatic term uses squared distance in embedding space; pymdp risk uses KL divergence in discrete probability space. After normalization, these should be highly correlated. |
| Epistemic value | Ambiguity | Both measure residual uncertainty about observations. The neural term uses variance of Monte Carlo samples or differential entropy; pymdp ambiguity uses conditional entropy `H[P(o\|s)]` under the posterior. Correlation is typically slightly lower than pragmatic due to approximation differences. |
| Instrumental value (empowerment) | No equivalent | pymdp has no empowerment term. The neural instrumental term captures how many future options an action enables. For regression comparison, either (a) set `instrumental_weight = 0` in the neural model, or (b) compare only the pragmatic + epistemic subtotal. |

When comparing total EFE:
- Neural total = `w_p * pragmatic + w_e * epistemic + w_i * instrumental`
- pymdp total = `risk + ambiguity`
- For valid comparison, set `w_i = 0` or subtract the instrumental term from the neural total.
- Normalize both totals to [0, 1] before computing correlation, because the raw scales differ
  (neural EFE is in embedding-space units; pymdp EFE is in nats).

## Graceful Fallback

Follow these rules to ensure the system works with or without pymdp installed.

1. **Check `PYMDP_AVAILABLE` at import time.** The flag is set in `brain_ai/decision/active_inference.py`:
   ```python
   try:
       import pymdp
       from pymdp import utils as pymdp_utils
       from pymdp.agent import Agent as PyMDPAgent
       PYMDP_AVAILABLE = True
   except ImportError:
       PYMDP_AVAILABLE = False
   ```

2. **Guard all pymdp code.** Every function or method that uses pymdp must check the flag:
   ```python
   def build_arrays(self, ...):
       if not PYMDP_AVAILABLE:
           raise ImportError("pymdp required. Install: pip install inferactively-pymdp")
       ...
   ```

3. **Never import pymdp at module level** outside the try/except block. If `pymdp_backend.py`
   needs pymdp, import it inside `__init__` or individual methods.

4. **Raise informative errors.** When pymdp is missing, the error message must include the
   install command: `pip install inferactively-pymdp`.

5. **Skip tests gracefully.** Decorate pymdp-dependent tests:
   ```python
   import pytest
   from brain_ai.decision.active_inference import PYMDP_AVAILABLE

   @pytest.mark.skipif(not PYMDP_AVAILABLE, reason="pymdp not installed")
   def test_pymdp_efe_regression():
       ...
   ```

6. **No degraded production behavior.** The absence of pymdp must never change the neural
   model's forward pass, training, or inference. It is strictly a testing dependency.

## Configuration

Add the following config dataclass alongside the existing `ActiveInferenceConfig`:

```python
@dataclass
class PyMDPConfig:
    """Configuration for the optional pymdp discrete POMDP backend."""

    enabled: bool = False
    # Off by default. Enable only for regression testing.

    num_discrete_states: int = 16
    # Number of discrete states for K-means discretization.

    num_discrete_obs: int = 16
    # Number of discrete observation bins.

    num_discrete_actions: int = 10
    # Number of discrete actions (for continuous action spaces).
    # Ignored when the action space is already discrete.

    discretization_method: str = "kmeans"
    # Method for mapping continuous to discrete: "kmeans", "uniform", "custom".
    # "kmeans" fits clusters on calibration data.
    # "uniform" creates evenly spaced bins over the value range.
    # "custom" expects externally provided centers.

    num_calibration_samples: int = 1000
    # Number of samples for fitting K-means clusters.

    planning_horizon: int = 3
    # pymdp planning horizon (policy length).

    inference_algo: str = "MMP"
    # pymdp inference algorithm: "MMP" (Marginal Message Passing) or "VANILLA".

    regression_tolerance: float = 0.1
    # Maximum allowed normalized EFE difference for regression tests.

    correlation_threshold: float = 0.9
    # Minimum Pearson correlation between neural and discrete EFE.

    use_for_training: bool = False
    # If True, use pymdp EFE as auxiliary training signal.
    # Default False -- only for testing.

    cache_discretization: bool = True
    # Cache cluster centers across evaluations for consistency.

    cache_path: str = ""
    # Path to store/load cached cluster centers (.npz).
    # Empty string means use a default path relative to the checkpoint directory.
```

## Common Failure Modes

| Symptom | Cause | Diagnosis | Fix |
|---------|-------|-----------|-----|
| A matrix has all-zero columns | State center falls outside the observation model's support | Inspect column sums; plot state centers vs. observation predictions | Increase calibration samples; use tighter K-means initialization |
| B matrix is nearly identity for all actions | Transition model not trained enough; actions have negligible effect | Check B off-diagonal mass; compare against ground truth B | Train transition model longer; verify action encoding |
| C vector is flat (all zeros) | Preference model not learned or not evaluated correctly | Print raw preference scores at bin centers | Verify preference model training; check for gradient flow |
| Correlation is high but ranking disagrees | Nonlinear distortion preserves order for most but not top actions | Scatter-plot neural vs. discrete EFE per policy | Increase discretization bins; train neural model longer |
| pymdp agent raises shape error | Multi-factor arrays wrapped incorrectly | Check that A, B, C, D are lists of arrays even for single factor | Wrap each array in a list: `[A]`, `[B]`, `[C]`, `[D]` |
| Neural EFE is 10x larger than discrete | Scale mismatch between embedding distance and discrete KL | Print raw EFE values before normalization | Always normalize to [0, 1] before comparison |
| Discretization changes between runs | K-means is non-deterministic | Check cluster centers across runs | Set random seed for K-means; enable `cache_discretization` |
| Epistemic term correlation is low despite good pragmatic | Neural variance estimation is noisy for small sample counts | Increase `num_samples` in Monte Carlo EFE | Use 64+ MC samples; consider analytic entropy where possible |
| Test passes on CPU but fails on GPU | Float precision differences between devices | Compare fp32 results across devices | Force fp32 for EFE computation; set `torch.set_float32_matmul_precision('highest')` |

## Tutorial: Running a Regression Test

This walkthrough sets up a complete regression test from scratch.

### Prerequisites

```bash
pip install inferactively-pymdp torch numpy scikit-learn pytest
```

### Step 1: Create the Test File

Create `tests/test_pymdp_regression.py`:

```python
import pytest
import numpy as np
import torch

from brain_ai.decision.active_inference import (
    PYMDP_AVAILABLE,
    ActiveInferenceAgent,
    ActiveInferenceConfig,
)

pytestmark = pytest.mark.skipif(
    not PYMDP_AVAILABLE, reason="pymdp not installed"
)


def make_toy_arrays():
    """Define a 4-state, 4-obs, 3-action toy POMDP with known ground truth."""
    A = np.array([
        [0.9, 0.05, 0.05, 0.0],
        [0.05, 0.9, 0.0, 0.05],
        [0.05, 0.0, 0.9, 0.05],
        [0.0, 0.05, 0.05, 0.9],
    ])

    B = np.zeros((4, 4, 3))
    # Action 0: move left
    B[:, :, 0] = np.array([
        [0.9, 0.8, 0.0, 0.0],
        [0.1, 0.2, 0.1, 0.0],
        [0.0, 0.0, 0.1, 0.1],
        [0.0, 0.0, 0.8, 0.9],
    ])
    # Action 1: stay
    B[:, :, 1] = np.eye(4) * 0.9 + 0.1 / 4
    # Action 2: move right
    B[:, :, 2] = np.array([
        [0.9, 0.0, 0.0, 0.0],
        [0.1, 0.1, 0.0, 0.0],
        [0.0, 0.1, 0.2, 0.1],
        [0.0, 0.8, 0.8, 0.9],
    ])

    C = np.array([0.0, 3.0, 0.0, -3.0])
    D = np.ones(4) / 4.0
    return A, B, C, D


@pytest.fixture
def toy_arrays():
    return make_toy_arrays()


@pytest.fixture
def pymdp_agent(toy_arrays):
    from pymdp.agent import Agent as PyMDPAgent
    A, B, C, D = toy_arrays
    agent = PyMDPAgent(
        A=[A], B=[B], C=[C], D=[D],
        policy_len=3,
        inference_algo="MMP",
        use_states_info_gain=True,
        action_selection="stochastic",
    )
    return agent


def test_pymdp_efe_positive_for_dispreferred(pymdp_agent, toy_arrays):
    """EFE should be higher (worse) for policies leading to dispreferred obs."""
    from pymdp import utils as pymdp_utils
    A, B, C, D = toy_arrays

    # Observe state 0
    obs = pymdp_utils.onehot(0, 4)
    qs = pymdp_agent.infer_states([obs])
    q_pi, efe = pymdp_agent.infer_policies()

    # Policies toward preferred observation (obs 1) should have lower EFE
    assert efe is not None
    assert len(efe) > 0


def test_neural_vs_discrete_correlation(toy_arrays):
    """Neural EFE must correlate > 0.9 with discrete reference."""
    from brain_ai.decision.pymdp_backend import PyMDPBackend

    A, B, C, D = toy_arrays
    backend = PyMDPBackend(
        num_states=4, num_obs=4, num_actions=3, planning_horizon=3
    )

    # Build discrete reference EFE
    agent = backend.create_agent(A, B, C, D)
    # ... (run agent, collect discrete EFE per policy)

    # Build neural model, train on toy data, extract neural EFE
    # ... (neural model setup)

    # Compare
    # result = backend.compare_efe(neural_efe, discrete_efe)
    # assert result["passed"], f"Regression failed: {result}"


def test_ranking_agreement(toy_arrays):
    """Top-3 actions from neural and discrete EFE must overlap by at least 2/3."""
    # Same structure as above: compute both EFEs, compare top-3 rankings.
    pass
```

### Step 2: Run the Test

```bash
# Run only pymdp regression tests
python -m pytest tests/test_pymdp_regression.py -v

# Run with the pymdp marker filter (if using markers)
python -m pytest tests/ -v -m "pymdp"

# Skip if pymdp is absent (tests auto-skip via skipif)
python -m pytest tests/ -v
```

### Step 3: Interpret Results

A passing test confirms that the neural EFE computation agrees with exact discrete EFE within
the specified tolerance. A failing test indicates one of:

- **Low correlation (< 0.9):** The neural generative model is not faithfully approximating the
  discrete arrays. Retrain the neural model with more data or check the likelihood/transition
  architectures.
- **High max difference (> tolerance):** Outlier policies exist where the neural and discrete
  EFE strongly disagree. Inspect those specific policies and check whether the discretization
  is too coarse for the relevant state-action region.
- **Low ranking agreement (< 2/3 top-3 overlap):** The neural model assigns different relative
  ordering to the best policies. This is the most consequential failure -- it means the agent
  would select different actions. Check the preference (C) reconstruction quality first, since
  pragmatic value dominates action ranking in most problems.

### Step 4: Iterate

If the test fails:

1. Increase `num_discrete_states` and `num_discrete_obs` to reduce discretization error.
2. Train the neural model for more epochs on the toy problem.
3. Check individual term correlations (pragmatic vs. epistemic) to localize the mismatch.
4. Set `instrumental_weight = 0` to remove the empowerment term from the comparison.
5. Visualize the reconstructed A and B matrices against ground truth using heatmaps.

Use the diagnostics dict returned by `compare_efe()` to guide debugging. Log it to a JSON
file for CI integration:

```python
import json

result = backend.compare_efe(neural_efe, discrete_efe, tolerance=0.1)
with open("pymdp_regression_result.json", "w") as f:
    json.dump(result, f, indent=2)

assert result["passed"], f"pymdp regression failed: {json.dumps(result, indent=2)}"
```
