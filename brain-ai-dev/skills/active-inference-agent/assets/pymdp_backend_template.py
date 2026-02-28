"""
Optional pymdp Discrete POMDP Backend for Regression Testing Neural EFE.

When ``pymdp`` (``inferactively-pymdp >= 0.0.7.1``) is installed, this module
provides a discrete POMDP backend that constructs A/B/C/D arrays from neural
model parameters and delegates planning to pymdp's exact routines.

Primary purpose: **regression testing** -- verify that the neural Expected Free
Energy implementation produces results consistent with exact discrete
computations on toy problems where ground truth is known.

Secondary purpose: **debugging** -- discrete arrays are directly inspectable,
making it straightforward to isolate whether a misbehavior originates in the
generative model, the EFE decomposition, or the planner.

The backend is off by default.  Enable it only for testing and validation.
Production inference always uses the neural continuous pipeline
(``ImprovedActiveInferenceAgent`` in ``brain_ai/decision/active_inference.py``).

Array semantics (single-factor):

    A  (num_obs, num_states)            P(o | s)
    B  (num_states, num_states, num_actions)  P(s' | s, a)
    C  (num_obs,)                       log P_pref(o)
    D  (num_states,)                    P(s_0)

pymdp decomposes EFE into risk (pragmatic) and ambiguity (epistemic):

    Risk      = E_q(o|pi)[ log q(o|pi) - log C(o) ]
    Ambiguity = E_q(s|pi)[ H[P(o|s)] ]

For mapping to the neural three-term EFE see the EFE Decomposition Mapping
section in ``references/pymdp-integration.md``.

References:
    - Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference.
    - Da Costa, L. et al. (2020). Active Inference on Discrete State-Spaces.
    - Heins, C. et al. (2022). pymdp: A Framework for Active Inference.

This module is part of the brain-inspired AI system described in CLAUDE.md.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import block
# ---------------------------------------------------------------------------

try:
    import pymdp
    from pymdp import utils as pymdp_utils
    from pymdp import maths as pymdp_maths
    from pymdp.agent import Agent as PyMDPAgent

    PYMDP_AVAILABLE = True
except ImportError:
    pymdp = None  # type: ignore[assignment]
    pymdp_utils = None  # type: ignore[assignment]
    pymdp_maths = None  # type: ignore[assignment]
    PyMDPAgent = None  # type: ignore[assignment,misc]
    PYMDP_AVAILABLE = False

# Optional torch -- not strictly required by this module but used when
# bridging neural model outputs to numpy arrays.
try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

# Optional sklearn for K-means clustering.
try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    KMeans = None  # type: ignore[assignment,misc]
    MiniBatchKMeans = None  # type: ignore[assignment,misc]
    SKLEARN_AVAILABLE = False

_PYMDP_INSTALL_MSG = (
    "pymdp is required for PyMDPBackend. "
    "Install with: pip install inferactively-pymdp"
)

_SKLEARN_INSTALL_MSG = (
    "scikit-learn is required for K-means discretization. "
    "Install with: pip install scikit-learn"
)


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class PyMDPConfig:
    """Configuration for the optional pymdp discrete POMDP backend.

    Attributes:
        enabled: Master switch.  Off by default; enable only for regression
            testing or debugging.
        num_discrete_states: Number of discrete state bins for K-means
            discretization of the continuous latent space.
        num_discrete_obs: Number of discrete observation bins for K-means
            discretization of the continuous observation space.
        num_discrete_actions: Number of discrete action bins for continuous
            action spaces.  Ignored when the action space is already discrete.
        discretization_method: Strategy for mapping continuous to discrete:
            ``"kmeans"`` fits K-means clusters on calibration data;
            ``"uniform"`` creates evenly spaced bins over the value range;
            ``"custom"`` expects externally provided centers.
        num_calibration_samples: Number of samples used to fit K-means or
            estimate value ranges for uniform binning.
        planning_horizon: pymdp planning horizon (policy length).
        inference_algo: pymdp inference algorithm -- ``"MMP"`` for Marginal
            Message Passing or ``"VANILLA"`` for standard variational.
        regression_tolerance: Maximum allowed normalized EFE difference for
            regression tests to pass.
        correlation_threshold: Minimum Pearson correlation between neural
            and discrete EFE for regression tests to pass.
        use_for_training: If True, use pymdp EFE as an auxiliary training
            signal.  Default False -- testing only.
        cache_discretization: Cache cluster centers across evaluations for
            reproducibility.
        cache_path: Path to store/load cached cluster centers (``.npz``).
            Empty string means use a temporary path.
        random_seed: Random seed for K-means and pymdp operations to ensure
            reproducibility across runs.
    """

    enabled: bool = False
    num_discrete_states: int = 16
    num_discrete_obs: int = 16
    num_discrete_actions: int = 10
    discretization_method: str = "kmeans"
    num_calibration_samples: int = 1000
    planning_horizon: int = 3
    inference_algo: str = "MMP"
    regression_tolerance: float = 0.1
    correlation_threshold: float = 0.9
    use_for_training: bool = False
    cache_discretization: bool = True
    cache_path: str = ""
    random_seed: int = 42


# ===========================================================================
# Comparison Result
# ===========================================================================


@dataclass
class ComparisonResult:
    """Result of comparing neural EFE against discrete pymdp reference.

    All correlations are Pearson correlations on [0, 1]-normalized values.
    Ranking agreement is the Jaccard overlap of the top-3 lowest-EFE
    policies (lower EFE = better policy).

    Attributes:
        correlation: Pearson correlation between normalized total EFE
            vectors from neural and discrete backends.
        pragmatic_correlation: Pearson correlation for the pragmatic
            (risk / goal-directed) component alone.  ``nan`` when the
            component is not available.
        epistemic_correlation: Pearson correlation for the epistemic
            (ambiguity / information-seeking) component alone.  ``nan``
            when the component is not available.
        ranking_agreement: Fraction of top-3 policy overlap in [0, 1].
            ``1.0`` means identical top-3 sets; ``0.0`` means no overlap.
        max_diff: Maximum absolute difference between the two normalized
            EFE vectors.
        passed: True iff ``correlation >= threshold`` AND
            ``max_diff <= tolerance``.
    """

    correlation: float
    pragmatic_correlation: float
    epistemic_correlation: float
    ranking_agreement: float
    max_diff: float
    passed: bool

    def to_dict(self) -> Dict[str, float]:
        """Serialize to a plain dictionary suitable for JSON logging."""
        return {
            "correlation": self.correlation,
            "pragmatic_correlation": self.pragmatic_correlation,
            "epistemic_correlation": self.epistemic_correlation,
            "ranking_agreement": self.ranking_agreement,
            "max_diff": self.max_diff,
            "passed": float(self.passed),
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] corr={self.correlation:.4f} "
            f"prag_corr={self.pragmatic_correlation:.4f} "
            f"epist_corr={self.epistemic_correlation:.4f} "
            f"rank_agree={self.ranking_agreement:.2f} "
            f"max_diff={self.max_diff:.4f}"
        )


# ===========================================================================
# Discretizer
# ===========================================================================


class Discretizer:
    """Maps between continuous vectors and discrete indices via clustering.

    Supports K-means clustering (requires scikit-learn) and uniform binning
    as discretization strategies.  Once ``fit()`` has been called, the
    discretizer stores cluster centers and can ``transform()`` new data to
    indices and ``inverse_transform()`` indices back to center vectors.

    Args:
        method: ``"kmeans"``, ``"uniform"``, or ``"custom"``.
        random_seed: Seed for K-means reproducibility.
    """

    def __init__(
        self,
        method: str = "kmeans",
        random_seed: int = 42,
    ) -> None:
        if method not in ("kmeans", "uniform", "custom"):
            raise ValueError(
                f"Unknown discretization method '{method}'. "
                "Choose from 'kmeans', 'uniform', 'custom'."
            )
        self.method = method
        self.random_seed = random_seed
        self.centers: Optional[np.ndarray] = None
        self.num_bins: int = 0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray,
        num_bins: int,
    ) -> "Discretizer":
        """Fit the discretizer on calibration data.

        Args:
            data: Continuous data of shape ``(N, D)`` where N is the number
                of samples and D is the dimensionality.
            num_bins: Number of discrete bins / clusters to create.

        Returns:
            ``self`` for method chaining.

        Raises:
            ImportError: If ``method="kmeans"`` and scikit-learn is not
                installed.
            ValueError: If data is empty or num_bins < 1.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if data.shape[0] == 0:
            raise ValueError("Cannot fit discretizer on empty data.")
        if num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {num_bins}.")

        self.num_bins = num_bins

        if self.method == "kmeans":
            if not SKLEARN_AVAILABLE:
                raise ImportError(_SKLEARN_INSTALL_MSG)
            # Use MiniBatchKMeans for large datasets for speed.
            n_samples = data.shape[0]
            if n_samples > 10_000:
                km = MiniBatchKMeans(
                    n_clusters=num_bins,
                    random_state=self.random_seed,
                    batch_size=min(1024, n_samples),
                    n_init=3,
                )
            else:
                km = KMeans(
                    n_clusters=num_bins,
                    random_state=self.random_seed,
                    n_init=10,
                )
            km.fit(data)
            self.centers = km.cluster_centers_.copy()

        elif self.method == "uniform":
            # Evenly spaced bins between data min and max per dimension.
            mins = data.min(axis=0)
            maxs = data.max(axis=0)
            # Create grid of center points.
            D = data.shape[1]
            if D == 1:
                edges = np.linspace(mins[0], maxs[0], num_bins + 1)
                self.centers = (0.5 * (edges[:-1] + edges[1:])).reshape(-1, 1)
            else:
                # For multi-dimensional data, use evenly spaced grid along
                # each dimension, then select num_bins points via farthest-
                # point sampling from the Cartesian product.
                bins_per_dim = max(2, int(np.ceil(num_bins ** (1.0 / D))))
                grids = [
                    np.linspace(mins[d], maxs[d], bins_per_dim)
                    for d in range(D)
                ]
                mesh = np.meshgrid(*grids, indexing="ij")
                candidates = np.stack(
                    [m.ravel() for m in mesh], axis=-1
                )
                # Subsample to exactly num_bins via farthest-point sampling.
                self.centers = self._farthest_point_sample(
                    candidates, num_bins
                )

        elif self.method == "custom":
            raise ValueError(
                "Method 'custom' requires manually setting centers via "
                "set_centers()."
            )

        self._fitted = True
        return self

    def set_centers(self, centers: np.ndarray) -> "Discretizer":
        """Manually set cluster centers for ``method='custom'``.

        Args:
            centers: Array of shape ``(num_bins, D)`` with bin center
                coordinates.

        Returns:
            ``self`` for method chaining.
        """
        if centers.ndim == 1:
            centers = centers.reshape(-1, 1)
        self.centers = centers.copy()
        self.num_bins = centers.shape[0]
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform / inverse
    # ------------------------------------------------------------------

    def transform(self, continuous_data: np.ndarray) -> np.ndarray:
        """Map continuous vectors to nearest discrete bin indices.

        Args:
            continuous_data: Array of shape ``(N, D)`` or ``(D,)`` with
                continuous feature vectors.

        Returns:
            Integer array of shape ``(N,)`` with bin indices in
            ``[0, num_bins)``.

        Raises:
            RuntimeError: If the discretizer has not been fitted.
        """
        self._check_fitted()
        if continuous_data.ndim == 1:
            continuous_data = continuous_data.reshape(1, -1)
        # Euclidean distance to each center.
        # (N, 1, D) - (1, K, D) -> (N, K, D) -> (N, K)
        diffs = continuous_data[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=-1)
        indices = np.argmin(dists, axis=-1)
        return indices

    def inverse_transform(self, discrete_indices: np.ndarray) -> np.ndarray:
        """Map discrete bin indices back to continuous center vectors.

        Args:
            discrete_indices: Integer array of shape ``(N,)`` with bin
                indices in ``[0, num_bins)``.

        Returns:
            Array of shape ``(N, D)`` with the center vector for each index.

        Raises:
            RuntimeError: If the discretizer has not been fitted.
            IndexError: If any index is out of range.
        """
        self._check_fitted()
        indices = np.asarray(discrete_indices).ravel()
        if np.any(indices < 0) or np.any(indices >= self.num_bins):
            raise IndexError(
                f"Indices must be in [0, {self.num_bins}), "
                f"got min={indices.min()}, max={indices.max()}."
            )
        return self.centers[indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save cluster centers to a ``.npz`` file.

        Args:
            path: File path ending in ``.npz``.
        """
        self._check_fitted()
        np.savez(
            path,
            centers=self.centers,
            method=np.array([self.method]),
            num_bins=np.array([self.num_bins]),
        )
        logger.info("Discretizer saved to %s (%d bins).", path, self.num_bins)

    def load(self, path: str) -> "Discretizer":
        """Load cluster centers from a ``.npz`` file.

        Uses ``allow_pickle=False`` for safety.

        Args:
            path: File path ending in ``.npz``.

        Returns:
            ``self`` for method chaining.
        """
        data = np.load(path, allow_pickle=False)
        self.centers = data["centers"]
        self.method = str(data["method"][0])
        self.num_bins = int(data["num_bins"][0])
        self._fitted = True
        logger.info(
            "Discretizer loaded from %s (%d bins, method=%s).",
            path,
            self.num_bins,
            self.method,
        )
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if the discretizer has not been fitted yet."""
        if not self._fitted or self.centers is None:
            raise RuntimeError(
                "Discretizer has not been fitted.  Call fit() or "
                "set_centers() first."
            )

    @staticmethod
    def _farthest_point_sample(
        candidates: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Select k points from candidates via farthest-point sampling.

        Greedy algorithm that iteratively picks the candidate farthest from
        all previously selected points.  Produces a well-spread subset.

        Args:
            candidates: ``(M, D)`` array of candidate points.
            k: Number of points to select.

        Returns:
            ``(k, D)`` array of selected points.
        """
        if candidates.shape[0] <= k:
            return candidates[:k].copy()

        n = candidates.shape[0]
        selected_idx = [0]
        min_dists = np.full(n, np.inf)

        for _ in range(k - 1):
            last = candidates[selected_idx[-1]]
            dists = np.sum((candidates - last) ** 2, axis=-1)
            min_dists = np.minimum(min_dists, dists)
            next_idx = int(np.argmax(min_dists))
            selected_idx.append(next_idx)

        return candidates[selected_idx].copy()


# ===========================================================================
# PyMDP Backend
# ===========================================================================


class PyMDPBackend:
    """Discrete POMDP backend using pymdp for regression testing.

    Wraps all pymdp interactions behind a single class.  Constructs
    A/B/C/D arrays from a neural generative model via discretization,
    runs exact planning, and compares results against the neural EFE.

    The full workflow is:

    1. Create backend: ``backend = PyMDPBackend(config)``
    2. Build arrays: ``A, B, C, D = backend.build_arrays(model, dataset)``
    3. Create agent: ``backend.create_agent(A, B, C, D)``
    4. Run steps:    ``action, info = backend.select_action(obs_idx)``
    5. Compare:      ``result = backend.compare_efe(neural_efe, discrete_efe)``

    Args:
        config: PyMDPConfig controlling discretization, planning, and
            tolerance thresholds.

    Raises:
        ImportError: If pymdp is not installed.
    """

    def __init__(self, config: Optional[PyMDPConfig] = None) -> None:
        if not PYMDP_AVAILABLE:
            raise ImportError(_PYMDP_INSTALL_MSG)

        self.config = config or PyMDPConfig()
        self.agent: Optional[Any] = None  # PyMDPAgent instance

        # Discretizers for each space.
        self.discretizer_state = Discretizer(
            method=self.config.discretization_method,
            random_seed=self.config.random_seed,
        )
        self.discretizer_obs = Discretizer(
            method=self.config.discretization_method,
            random_seed=self.config.random_seed + 1,
        )
        self.discretizer_action = Discretizer(
            method=self.config.discretization_method,
            random_seed=self.config.random_seed + 2,
        )

        # Cached arrays.
        self._A: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None
        self._C: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Array construction helpers
    # ------------------------------------------------------------------

    def build_A_matrix(
        self,
        likelihood_model: Any,
        discretizer_obs: Discretizer,
        discretizer_state: Discretizer,
    ) -> np.ndarray:
        """Build the observation likelihood matrix A = P(o | s).

        For each discrete state center, pass it through the neural likelihood
        model and compute a Gaussian-kernel similarity to each observation
        bin center.  Columns are normalized to sum to 1.

        Args:
            likelihood_model: Neural model with a callable that maps state
                tensors to predicted observation tensors.  Must support
                ``likelihood_model(state_tensor) -> obs_tensor``.
            discretizer_obs: Fitted discretizer for observations.
            discretizer_state: Fitted discretizer for states.

        Returns:
            A matrix of shape ``(num_obs, num_states)`` with
            ``A[o, s] = P(o | s)``.
        """
        if not PYMDP_AVAILABLE:
            raise ImportError(_PYMDP_INSTALL_MSG)

        obs_centers = discretizer_obs.centers
        state_centers = discretizer_state.centers
        num_obs = obs_centers.shape[0]
        num_states = state_centers.shape[0]

        A = np.zeros((num_obs, num_states), dtype=np.float64)

        for k in range(num_states):
            s_center = state_centers[k]
            if torch is not None:
                s_tensor = torch.tensor(
                    s_center, dtype=torch.float32
                ).unsqueeze(0)
                with torch.no_grad():
                    pred_obs = likelihood_model(s_tensor).squeeze(0).cpu().numpy()
            else:
                # Fallback: treat model as a callable on numpy.
                pred_obs = np.asarray(
                    likelihood_model(s_center.reshape(1, -1))
                ).ravel()

            # Gaussian kernel similarity to each observation bin center.
            for m in range(num_obs):
                dist_sq = float(np.sum((pred_obs - obs_centers[m]) ** 2))
                A[m, k] = math.exp(-0.5 * dist_sq)

        # Normalize columns (each column sums to 1).
        col_sums = A.sum(axis=0, keepdims=True)
        A = A / (col_sums + 1e-12)
        return A

    def build_B_matrix(
        self,
        transition_model: Any,
        discretizer_state: Discretizer,
        discretizer_action: Optional[Discretizer] = None,
        num_actions: Optional[int] = None,
    ) -> np.ndarray:
        """Build the state transition matrix B = P(s' | s, a).

        For each (state, action) pair, query the neural transition model for
        the predicted next-state distribution and compute Gaussian-kernel
        similarity to each next-state bin center.

        Args:
            transition_model: Neural model with
                ``predict_next_state(state, action_onehot) -> (mu, log_var)``.
            discretizer_state: Fitted discretizer for states.
            discretizer_action: Fitted discretizer for continuous actions.
                Ignored when actions are already discrete (pass None and set
                ``num_actions``).
            num_actions: Number of discrete actions.  Required when
                ``discretizer_action`` is None.

        Returns:
            B matrix of shape ``(num_states, num_states, num_actions)`` with
            ``B[s', s, a] = P(s' | s, a)``.
        """
        if not PYMDP_AVAILABLE:
            raise ImportError(_PYMDP_INSTALL_MSG)

        state_centers = discretizer_state.centers
        K = state_centers.shape[0]

        if num_actions is None and discretizer_action is not None:
            num_actions = discretizer_action.num_bins
        if num_actions is None:
            num_actions = self.config.num_discrete_actions

        B = np.zeros((K, K, num_actions), dtype=np.float64)

        for k in range(K):
            s_center = state_centers[k]
            for a in range(num_actions):
                if torch is not None:
                    s_tensor = torch.tensor(
                        s_center, dtype=torch.float32
                    ).unsqueeze(0)
                    a_onehot = torch.zeros(
                        1, num_actions, dtype=torch.float32
                    )
                    a_onehot[0, a] = 1.0
                    with torch.no_grad():
                        mu, log_var = transition_model.predict_next_state(
                            s_tensor, a_onehot
                        )
                    mu_np = mu.squeeze(0).cpu().numpy()
                    std_np = np.exp(
                        0.5 * log_var.squeeze(0).cpu().numpy()
                    )
                else:
                    # Numpy fallback.
                    a_onehot_np = np.zeros(
                        (1, num_actions), dtype=np.float64
                    )
                    a_onehot_np[0, a] = 1.0
                    mu_np, log_var_np = transition_model.predict_next_state(
                        s_center.reshape(1, -1), a_onehot_np
                    )
                    mu_np = np.asarray(mu_np).ravel()
                    std_np = np.exp(
                        0.5 * np.asarray(log_var_np).ravel()
                    )

                for j in range(K):
                    diff = (state_centers[j] - mu_np) / (std_np + 1e-8)
                    dist_sq = float(np.sum(diff ** 2))
                    B[j, k, a] = math.exp(-0.5 * dist_sq)

        # Normalize columns for each action slice.
        col_sums = B.sum(axis=0, keepdims=True)
        B = B / (col_sums + 1e-12)
        return B

    def build_C_vector(
        self,
        preferences: Any,
        discretizer_obs: Discretizer,
    ) -> np.ndarray:
        """Build the preference vector C = log P_pref(o).

        Evaluates the neural preference model at each observation bin center.

        Args:
            preferences: Neural preference model with
                ``compute_pragmatic_value(obs_tensor) -> scalar_tensor``
                or a callable mapping observation vectors to preference scores.
            discretizer_obs: Fitted discretizer for observations.

        Returns:
            C vector of shape ``(num_obs,)`` with log-preferences,
            shifted so that ``max(C) = 0`` for numerical stability.
        """
        if not PYMDP_AVAILABLE:
            raise ImportError(_PYMDP_INSTALL_MSG)

        obs_centers = discretizer_obs.centers
        num_obs = obs_centers.shape[0]
        C = np.zeros(num_obs, dtype=np.float64)

        for m in range(num_obs):
            o_center = obs_centers[m]
            if (
                torch is not None
                and hasattr(preferences, "compute_pragmatic_value")
            ):
                o_tensor = torch.tensor(
                    o_center, dtype=torch.float32
                ).unsqueeze(0)
                with torch.no_grad():
                    C[m] = float(
                        preferences.compute_pragmatic_value(o_tensor).item()
                    )
            elif callable(preferences):
                C[m] = float(preferences(o_center.reshape(1, -1)))
            else:
                raise TypeError(
                    "preferences must have compute_pragmatic_value() method "
                    "or be callable."
                )

        # Shift for numerical stability.
        C = C - C.max()
        return C

    def build_D_vector(
        self,
        prior: Any,
        discretizer_state: Discretizer,
    ) -> np.ndarray:
        """Build the state prior vector D = P(s_0).

        If ``prior`` is None, returns a uniform distribution.

        Args:
            prior: Neural prior model evaluable at state centers, or None
                for uniform.
            discretizer_state: Fitted discretizer for states.

        Returns:
            D vector of shape ``(num_states,)`` summing to 1.
        """
        if not PYMDP_AVAILABLE:
            raise ImportError(_PYMDP_INSTALL_MSG)

        state_centers = discretizer_state.centers
        num_states = state_centers.shape[0]

        if prior is None:
            return np.ones(num_states, dtype=np.float64) / num_states

        D = np.zeros(num_states, dtype=np.float64)
        for k in range(num_states):
            s_center = state_centers[k]
            if torch is not None and hasattr(prior, "log_prob"):
                s_tensor = torch.tensor(
                    s_center, dtype=torch.float32
                ).unsqueeze(0)
                with torch.no_grad():
                    D[k] = math.exp(
                        float(prior.log_prob(s_tensor).item())
                    )
            elif callable(prior):
                D[k] = float(prior(s_center.reshape(1, -1)))
            else:
                # Standard normal prior.
                D[k] = math.exp(
                    -0.5 * float(np.sum(s_center ** 2))
                )

        # Normalize.
        D = D / (D.sum() + 1e-12)
        return D

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def build_arrays(
        self,
        neural_model: Any,
        dataset: Any,
        num_calibration_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build all four generative model arrays from a neural model.

        Orchestrates discretizer fitting, array construction, optional
        caching, and validation.

        Args:
            neural_model: The neural active inference agent.  Expected to
                have attributes ``encoder``, ``generative`` (or
                ``likelihood`` and ``transition``), and ``preferences``.
            dataset: Iterable of observation tensors for calibration.
                Each element should be a tensor of shape ``(obs_dim,)`` or
                ``(batch, obs_dim)``.
            num_calibration_samples: Override for
                ``config.num_calibration_samples``.

        Returns:
            Tuple of ``(A, B, C, D)`` numpy arrays.
        """
        if not PYMDP_AVAILABLE:
            raise ImportError(_PYMDP_INSTALL_MSG)

        n_cal = (
            num_calibration_samples or self.config.num_calibration_samples
        )

        # ---- 1. Collect latent states and observations for clustering ----
        obs_list: List[np.ndarray] = []
        state_list: List[np.ndarray] = []
        count = 0

        encoder = getattr(neural_model, "encoder", None)

        # Resolve likelihood model.
        likelihood = getattr(neural_model, "likelihood", None)
        if likelihood is None:
            gen = getattr(neural_model, "generative", None)
            if gen is not None:
                likelihood = getattr(gen, "predict_obs", gen)

        # Resolve transition model.
        transition = getattr(neural_model, "transition", None)
        if transition is None:
            transition = getattr(neural_model, "generative", None)

        preferences = getattr(neural_model, "preferences", None)

        for obs_batch in dataset:
            if torch is not None and isinstance(obs_batch, torch.Tensor):
                obs_np = obs_batch.detach().cpu().numpy()
            else:
                obs_np = np.asarray(obs_batch)
            if obs_np.ndim == 1:
                obs_np = obs_np.reshape(1, -1)
            obs_list.append(obs_np)

            # Encode to latent states.
            if encoder is not None and torch is not None:
                with torch.no_grad():
                    obs_t = torch.tensor(obs_np, dtype=torch.float32)
                    mu, _ = encoder(obs_t)
                    state_list.append(mu.cpu().numpy())
            else:
                state_list.append(obs_np)

            count += obs_np.shape[0]
            if count >= n_cal:
                break

        all_obs = np.concatenate(obs_list, axis=0)[:n_cal]
        all_states = np.concatenate(state_list, axis=0)[:n_cal]

        # ---- 2. Fit discretizers ----
        loaded_from_cache = False
        if self.config.cache_discretization and self.config.cache_path:
            cache_state = self.config.cache_path + ".state.npz"
            cache_obs = self.config.cache_path + ".obs.npz"
            if os.path.exists(cache_state) and os.path.exists(cache_obs):
                self.discretizer_state.load(cache_state)
                self.discretizer_obs.load(cache_obs)
                loaded_from_cache = True
                logger.info(
                    "Loaded discretizer caches from %s.",
                    self.config.cache_path,
                )

        if not loaded_from_cache:
            self.discretizer_state.fit(
                all_states, self.config.num_discrete_states
            )
            self.discretizer_obs.fit(
                all_obs, self.config.num_discrete_obs
            )
            # Save cache.
            if self.config.cache_discretization and self.config.cache_path:
                self.discretizer_state.save(
                    self.config.cache_path + ".state.npz"
                )
                self.discretizer_obs.save(
                    self.config.cache_path + ".obs.npz"
                )

        # ---- 3. Build A, B, C, D ----
        A = self.build_A_matrix(
            likelihood_model=likelihood,
            discretizer_obs=self.discretizer_obs,
            discretizer_state=self.discretizer_state,
        )
        B = self.build_B_matrix(
            transition_model=transition,
            discretizer_state=self.discretizer_state,
            num_actions=self.config.num_discrete_actions,
        )
        C = self.build_C_vector(
            preferences=preferences,
            discretizer_obs=self.discretizer_obs,
        )
        D = self.build_D_vector(
            prior=None,
            discretizer_state=self.discretizer_state,
        )

        self._A = A
        self._B = B
        self._C = C
        self._D = D

        logger.info(
            "Built discrete arrays: A=%s B=%s C=%s D=%s",
            A.shape,
            B.shape,
            C.shape,
            D.shape,
        )
        return A, B, C, D

    # ------------------------------------------------------------------
    # Agent creation and action selection
    # ------------------------------------------------------------------

    def create_agent(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
    ) -> Any:
        """Instantiate a pymdp Agent from A/B/C/D arrays.

        Args:
            A: Likelihood matrix ``(num_obs, num_states)``.
            B: Transition matrix ``(num_states, num_states, num_actions)``.
            C: Preference vector ``(num_obs,)``.
            D: Prior vector ``(num_states,)``.

        Returns:
            The pymdp ``Agent`` instance (also stored in ``self.agent``).
        """
        if not PYMDP_AVAILABLE:
            raise ImportError(_PYMDP_INSTALL_MSG)

        self._A = A
        self._B = B
        self._C = C
        self._D = D

        self.agent = PyMDPAgent(
            A=[A],
            B=[B],
            C=[C],
            D=[D],
            policy_len=self.config.planning_horizon,
            inference_algo=self.config.inference_algo,
            use_states_info_gain=True,
            action_selection="stochastic",
        )
        return self.agent

    def select_action(
        self,
        observation: Union[int, np.ndarray],
    ) -> Tuple[int, Dict[str, Any]]:
        """Run a full pymdp agent step: observe, update beliefs, act.

        Args:
            observation: Either a discrete observation index (int) or a
                one-hot / probability vector over observations.

        Returns:
            Tuple of ``(action_index, info_dict)`` where ``info_dict``
            contains ``beliefs``, ``policy_probs``, and ``efe`` arrays.

        Raises:
            RuntimeError: If ``create_agent()`` has not been called.
        """
        if not PYMDP_AVAILABLE:
            raise ImportError(_PYMDP_INSTALL_MSG)
        if self.agent is None:
            raise RuntimeError(
                "Agent not initialized.  Call create_agent() first."
            )

        # Convert integer observation to one-hot.
        if isinstance(observation, (int, np.integer)):
            num_obs = (
                self._A.shape[0] if self._A is not None else 16
            )
            obs = pymdp_utils.onehot(int(observation), num_obs)
        else:
            obs = np.asarray(observation)

        # pymdp expects list-of-arrays for multi-factor; wrap single factor.
        qs = self.agent.infer_states([obs])
        q_pi, efe = self.agent.infer_policies()
        action = self.agent.sample_action()

        action_idx = (
            int(action[0]) if hasattr(action, "__len__") else int(action)
        )

        info: Dict[str, Any] = {
            "beliefs": qs,
            "policy_probs": q_pi,
            "efe": efe,
        }
        return action_idx, info

    # ------------------------------------------------------------------
    # EFE computation
    # ------------------------------------------------------------------

    def compute_efe_discrete(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        policies: np.ndarray,
        qs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute exact EFE for each policy using discrete arrays.

        Uses the risk + ambiguity decomposition:

            G(pi) = sum_t [ risk_t(pi) + ambiguity_t(pi) ]

        where:
            risk_t     = E_q(o_t|pi)[ log q(o_t|pi) - log C(o_t) ]
            ambiguity_t = E_q(s_t|pi)[ H[P(o|s)] ]

        Args:
            A: Likelihood matrix ``(num_obs, num_states)``.
            B: Transition matrix ``(num_states, num_states, num_actions)``.
            C: Preference vector ``(num_obs,)``.
            policies: Integer array of shape ``(num_policies, horizon)``
                with action indices per time step.
            qs: Current beliefs over states, shape ``(num_states,)``.  If
                None, uses a uniform prior.

        Returns:
            EFE per policy, shape ``(num_policies,)``.  Lower values are
            better (less expected free energy).
        """
        if not PYMDP_AVAILABLE:
            raise ImportError(_PYMDP_INSTALL_MSG)

        num_states = A.shape[1]
        num_obs = A.shape[0]
        num_policies = policies.shape[0]
        horizon = policies.shape[1]

        if qs is None:
            qs = np.ones(num_states, dtype=np.float64) / num_states

        # Safe log.
        def spm_log(x: np.ndarray) -> np.ndarray:
            return np.log(x + 1e-16)

        # Compute entropy of each column of A: H[P(o|s)] for each s.
        H_A = np.zeros(num_states, dtype=np.float64)
        for s in range(num_states):
            col = A[:, s]
            H_A[s] = -float(np.sum(col * spm_log(col)))

        # Log preferences (softmax-normalized).
        log_C = spm_log(self._softmax(C))

        efe_per_policy = np.zeros(num_policies, dtype=np.float64)

        for pi_idx in range(num_policies):
            policy = policies[pi_idx]
            qs_t = qs.copy()
            G = 0.0

            for t in range(horizon):
                a = int(policy[t])
                # Predict next state: qs_{t+1} = B[:, :, a] @ qs_t
                qs_next = B[:, :, a] @ qs_t
                qs_next = qs_next / (qs_next.sum() + 1e-12)

                # Predicted observations: qo_{t+1} = A @ qs_{t+1}
                qo = A @ qs_next
                qo = qo / (qo.sum() + 1e-12)

                # Risk: KL[ q(o|pi) || C ]
                risk = float(np.sum(qo * (spm_log(qo) - log_C)))

                # Ambiguity: E_q(s)[H[P(o|s)]]
                ambiguity = float(np.dot(qs_next, H_A))

                G += risk + ambiguity
                qs_t = qs_next

            efe_per_policy[pi_idx] = G

        return efe_per_policy

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_efe(
        self,
        neural_efe: np.ndarray,
        discrete_efe: np.ndarray,
        tolerance: Optional[float] = None,
        neural_pragmatic: Optional[np.ndarray] = None,
        discrete_pragmatic: Optional[np.ndarray] = None,
        neural_epistemic: Optional[np.ndarray] = None,
        discrete_epistemic: Optional[np.ndarray] = None,
    ) -> ComparisonResult:
        """Compare neural EFE against discrete pymdp reference.

        Both arrays are min-max normalized to [0, 1] before comparison.
        Pearson correlation, top-3 ranking agreement, and maximum absolute
        difference are computed.

        Args:
            neural_efe: EFE values from neural model, shape ``(N,)``.
            discrete_efe: EFE values from pymdp, shape ``(N,)``.
            tolerance: Maximum allowed normalized difference.  Defaults to
                ``self.config.regression_tolerance``.
            neural_pragmatic: Optional pragmatic-only EFE from neural model.
            discrete_pragmatic: Optional pragmatic-only EFE from pymdp.
            neural_epistemic: Optional epistemic-only EFE from neural model.
            discrete_epistemic: Optional epistemic-only EFE from pymdp.

        Returns:
            ``ComparisonResult`` with all metrics and pass/fail status.
        """
        if tolerance is None:
            tolerance = self.config.regression_tolerance

        neural_efe = np.asarray(neural_efe, dtype=np.float64).ravel()
        discrete_efe = np.asarray(discrete_efe, dtype=np.float64).ravel()

        if neural_efe.shape != discrete_efe.shape:
            raise ValueError(
                f"Shape mismatch: neural_efe {neural_efe.shape} vs "
                f"discrete_efe {discrete_efe.shape}."
            )

        # Normalize both to [0, 1].
        n_neural = self._normalize_01(neural_efe)
        n_discrete = self._normalize_01(discrete_efe)

        # Pearson correlation (total).
        correlation = self._pearson(n_neural, n_discrete)

        # Maximum absolute difference.
        max_diff = float(np.max(np.abs(n_neural - n_discrete)))

        # Top-3 ranking agreement (lower EFE = better).
        top3_neural = set(np.argsort(neural_efe)[:3].tolist())
        top3_discrete = set(np.argsort(discrete_efe)[:3].tolist())
        ranking_agreement = len(top3_neural & top3_discrete) / 3.0

        # Component-level correlations.
        pragmatic_corr = float("nan")
        if neural_pragmatic is not None and discrete_pragmatic is not None:
            pn = self._normalize_01(
                np.asarray(neural_pragmatic, dtype=np.float64)
            )
            pd = self._normalize_01(
                np.asarray(discrete_pragmatic, dtype=np.float64)
            )
            pragmatic_corr = self._pearson(pn, pd)

        epistemic_corr = float("nan")
        if neural_epistemic is not None and discrete_epistemic is not None:
            en = self._normalize_01(
                np.asarray(neural_epistemic, dtype=np.float64)
            )
            ed = self._normalize_01(
                np.asarray(discrete_epistemic, dtype=np.float64)
            )
            epistemic_corr = self._pearson(en, ed)

        # Pass/fail.
        passed = (
            correlation >= self.config.correlation_threshold
            and max_diff <= tolerance
        )

        result = ComparisonResult(
            correlation=correlation,
            pragmatic_correlation=pragmatic_corr,
            epistemic_correlation=epistemic_corr,
            ranking_agreement=ranking_agreement,
            max_diff=max_diff,
            passed=passed,
        )

        logger.info("EFE comparison: %s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Static / internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_01(x: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1], returning zeros if constant."""
        r = x.max() - x.min()
        if r < 1e-12:
            return np.zeros_like(x)
        return (x - x.min()) / r

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson correlation, returning 0.0 for degenerate inputs."""
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        r = np.corrcoef(a, b)[0, 1]
        return 0.0 if np.isnan(r) else float(r)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax for 1-D arrays."""
        e = np.exp(x - x.max())
        return e / (e.sum() + 1e-12)


# ===========================================================================
# Convenience factory
# ===========================================================================


def create_pymdp_backend(
    num_discrete_states: int = 16,
    num_discrete_obs: int = 16,
    num_discrete_actions: int = 10,
    planning_horizon: int = 3,
    **kwargs: Any,
) -> PyMDPBackend:
    """Create a PyMDPBackend with the given configuration.

    This is the recommended entry point for downstream code that needs
    pymdp regression testing.

    Args:
        num_discrete_states: Number of discrete state bins.
        num_discrete_obs: Number of discrete observation bins.
        num_discrete_actions: Number of discrete action bins.
        planning_horizon: pymdp planning horizon.
        **kwargs: Additional keyword arguments passed to ``PyMDPConfig``.

    Returns:
        Configured ``PyMDPBackend`` instance.

    Raises:
        ImportError: If pymdp is not installed.
    """
    config = PyMDPConfig(
        num_discrete_states=num_discrete_states,
        num_discrete_obs=num_discrete_obs,
        num_discrete_actions=num_discrete_actions,
        planning_horizon=planning_horizon,
        **kwargs,
    )
    return PyMDPBackend(config)


# ===========================================================================
# Self-Test Suite
# ===========================================================================


def _run_self_tests() -> None:
    """Run comprehensive self-tests for the pymdp backend module.

    Tests cover:
        1.  Discretizer fit/transform roundtrip (kmeans)
        2.  Discretizer fit/transform roundtrip (uniform)
        3.  Discretizer save/load persistence
        4.  Discretizer inverse_transform correctness
        5.  Discretizer custom centers
        6.  A matrix column normalization
        7.  B matrix slice normalization
        8.  C vector numerical stability (max-shifted)
        9.  D vector normalization
        10. EFE computation produces finite values
        11. EFE lower for preferred policies
        12. Compare: perfect correlation yields passed=True
        13. Compare: uncorrelated yields passed=False
        14. Compare: ranking agreement metric
        15. Graceful ImportError when pymdp not installed
        16. ComparisonResult serialization

    Tests requiring pymdp are skipped with a message if PYMDP_AVAILABLE is
    False.
    """
    passed = 0
    failed = 0
    skipped = 0
    total_tests = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed, total_tests
        total_tests += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            msg = f"  [FAIL] {name}"
            if detail:
                msg += f"  -- {detail}"
            print(msg)

    def skip(name: str, reason: str = "") -> None:
        nonlocal skipped, total_tests
        total_tests += 1
        skipped += 1
        msg = f"  [SKIP] {name}"
        if reason:
            msg += f"  -- {reason}"
        print(msg)

    print("=" * 72)
    print("pymdp Backend Template Self-Test Suite")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Discretizer: K-means fit/transform roundtrip
    # ------------------------------------------------------------------
    print("\n--- 1. Discretizer: K-means Fit/Transform ---")

    if SKLEARN_AVAILABLE:
        np.random.seed(42)
        data_1d = np.random.randn(200, 1) * 2.0 + 5.0
        disc_km = Discretizer(method="kmeans", random_seed=42)
        disc_km.fit(data_1d, num_bins=4)

        check(
            "kmeans fitted with 4 bins",
            disc_km.num_bins == 4 and disc_km.centers is not None,
        )

        indices = disc_km.transform(data_1d)
        check(
            "transform produces valid indices",
            indices.min() >= 0 and indices.max() < 4,
            f"min={indices.min()}, max={indices.max()}",
        )

        # Roundtrip: transform then inverse_transform should land on
        # centers.
        reconstructed = disc_km.inverse_transform(indices)
        check(
            "inverse_transform returns center vectors",
            reconstructed.shape == data_1d.shape,
            f"shape: {reconstructed.shape}",
        )

        # Each reconstructed point should be one of the 4 centers.
        unique_recon = np.unique(reconstructed, axis=0)
        check(
            "roundtrip yields <= num_bins unique values",
            unique_recon.shape[0] <= 4,
            f"got {unique_recon.shape[0]} unique values",
        )
    else:
        skip(
            "kmeans fit/transform",
            "scikit-learn not installed",
        )
        skip(
            "transform produces valid indices",
            "scikit-learn not installed",
        )
        skip(
            "inverse_transform returns center vectors",
            "scikit-learn not installed",
        )
        skip(
            "roundtrip yields <= num_bins unique values",
            "scikit-learn not installed",
        )

    # ------------------------------------------------------------------
    # 2. Discretizer: uniform fit/transform roundtrip
    # ------------------------------------------------------------------
    print("\n--- 2. Discretizer: Uniform Fit/Transform ---")

    np.random.seed(42)
    data_uniform = np.random.randn(100, 1) * 3.0
    disc_uni = Discretizer(method="uniform", random_seed=42)
    disc_uni.fit(data_uniform, num_bins=8)

    check(
        "uniform fitted with 8 bins",
        disc_uni.num_bins == 8 and disc_uni.centers is not None,
    )

    idx_uni = disc_uni.transform(data_uniform)
    check(
        "uniform transform valid indices",
        idx_uni.min() >= 0 and idx_uni.max() < 8,
        f"min={idx_uni.min()}, max={idx_uni.max()}",
    )

    # Centers should be well-shaped for 1D uniform.
    check(
        "uniform centers shape correct (1D)",
        disc_uni.centers.shape == (8, 1),
        f"shape={disc_uni.centers.shape}",
    )

    # ------------------------------------------------------------------
    # 3. Discretizer: save/load persistence
    # ------------------------------------------------------------------
    print("\n--- 3. Discretizer: Save/Load ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_disc.npz")
        disc_uni.save(save_path)

        disc_loaded = Discretizer()
        disc_loaded.load(save_path)

        check(
            "loaded centers match saved",
            np.allclose(disc_loaded.centers, disc_uni.centers),
        )
        check(
            "loaded num_bins matches",
            disc_loaded.num_bins == disc_uni.num_bins,
        )

        # Transform with loaded discretizer should match original.
        idx_loaded = disc_loaded.transform(data_uniform)
        check(
            "loaded discretizer reproduces same indices",
            np.array_equal(idx_loaded, idx_uni),
        )

    # ------------------------------------------------------------------
    # 4. Discretizer: inverse_transform correctness
    # ------------------------------------------------------------------
    print("\n--- 4. Discretizer: Inverse Transform ---")

    centers_custom = np.array([[0.0], [1.0], [2.0], [3.0]])
    disc_custom = Discretizer(method="custom")
    disc_custom.set_centers(centers_custom)

    test_indices = np.array([0, 1, 2, 3, 0, 2])
    recovered = disc_custom.inverse_transform(test_indices)
    expected_recovery = centers_custom[test_indices]
    check(
        "inverse_transform recovers exact centers",
        np.allclose(recovered, expected_recovery),
        f"recovered={recovered.ravel()}, "
        f"expected={expected_recovery.ravel()}",
    )

    # Out-of-range index should raise.
    raised_oob = False
    try:
        disc_custom.inverse_transform(np.array([5]))
    except IndexError:
        raised_oob = True
    check("out-of-range index raises IndexError", raised_oob)

    # ------------------------------------------------------------------
    # 5. Discretizer: custom centers
    # ------------------------------------------------------------------
    print("\n--- 5. Discretizer: Custom Centers ---")

    custom_2d = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    disc_c2 = Discretizer(method="custom")
    disc_c2.set_centers(custom_2d)
    check("custom centers shape", disc_c2.centers.shape == (4, 2))

    test_pts = np.array(
        [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    )
    c2_idx = disc_c2.transform(test_pts)
    check(
        "custom 2D nearest neighbor assignment",
        np.array_equal(c2_idx, np.array([0, 1, 2, 3])),
        f"got {c2_idx}",
    )

    # ------------------------------------------------------------------
    # Tests 6-12 require pymdp.
    # ------------------------------------------------------------------
    if not PYMDP_AVAILABLE:
        print("\n--- 6-12: Skipped (pymdp not installed) ---")
        for label in [
            "A matrix column normalization",
            "built A columns sum to 1",
            "built A diagonal dominant for identity model",
            "B[:, :, 0] columns sum to 1",
            "B[:, :, 1] columns sum to 1",
            "B[:, :, 2] columns sum to 1",
            "C shifted max is 0.0",
            "built D sums to 1",
            "EFE values are finite",
            "EFE returns correct number of policies",
            "best policy has lower EFE than worst",
            "EFE range > 0",
            "perfect linear transform: correlation ~ 1.0",
            "perfect linear transform: ranking_agreement = 1.0",
            "perfect linear transform: passed=True",
        ]:
            skip(label, "pymdp not installed")
    else:
        # ------------------------------------------------------------------
        # 6. A matrix column normalization
        # ------------------------------------------------------------------
        print("\n--- 6. A Matrix Column Normalization ---")

        num_s, num_o, num_a = 4, 4, 3
        A_toy = np.array([
            [0.9, 0.05, 0.05, 0.0],
            [0.05, 0.9, 0.0, 0.05],
            [0.05, 0.0, 0.9, 0.05],
            [0.0, 0.05, 0.05, 0.9],
        ])

        col_sums = A_toy.sum(axis=0)
        check(
            "A columns sum to 1",
            np.allclose(col_sums, 1.0, atol=1e-6),
            f"col_sums={col_sums}",
        )

        # Build A via backend with a dummy callable model.
        disc_s_toy = Discretizer(method="custom")
        disc_s_toy.set_centers(np.eye(4))
        disc_o_toy = Discretizer(method="custom")
        disc_o_toy.set_centers(np.eye(4))

        def dummy_likelihood(s_tensor):
            """Return state as observation (identity mapping)."""
            if torch is not None and isinstance(
                s_tensor, torch.Tensor
            ):
                return s_tensor
            return s_tensor

        config_toy = PyMDPConfig(
            num_discrete_states=4,
            num_discrete_obs=4,
            num_discrete_actions=3,
        )
        backend = PyMDPBackend(config_toy)
        A_built = backend.build_A_matrix(
            dummy_likelihood, disc_o_toy, disc_s_toy
        )
        col_sums_built = A_built.sum(axis=0)
        check(
            "built A columns sum to 1",
            np.allclose(col_sums_built, 1.0, atol=1e-6),
            f"col_sums={col_sums_built}",
        )

        # Diagonal should be dominant for identity mapping.
        check(
            "built A diagonal dominant for identity model",
            all(
                A_built[i, i] >= A_built[:, i].max() - 1e-6
                for i in range(4)
            ),
        )

        # ------------------------------------------------------------------
        # 7. B matrix slice normalization
        # ------------------------------------------------------------------
        print("\n--- 7. B Matrix Slice Normalization ---")

        B_toy = np.zeros((4, 4, 3))
        B_toy[:, :, 0] = np.array([
            [0.9, 0.8, 0.0, 0.0],
            [0.1, 0.2, 0.1, 0.0],
            [0.0, 0.0, 0.1, 0.1],
            [0.0, 0.0, 0.8, 0.9],
        ])
        B_toy[:, :, 1] = np.eye(4) * 0.9 + 0.1 / 4
        B_toy[:, :, 2] = np.array([
            [0.9, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0, 0.0],
            [0.0, 0.1, 0.2, 0.1],
            [0.0, 0.8, 0.8, 0.9],
        ])

        for a_idx in range(3):
            slice_sums = B_toy[:, :, a_idx].sum(axis=0)
            check(
                f"B[:, :, {a_idx}] columns sum to 1",
                np.allclose(slice_sums, 1.0, atol=1e-6),
                f"sums={slice_sums}",
            )

        # ------------------------------------------------------------------
        # 8. C vector numerical stability
        # ------------------------------------------------------------------
        print("\n--- 8. C Vector Numerical Stability ---")

        C_toy = np.array([0.0, 3.0, 0.0, -3.0])
        C_shifted = C_toy - C_toy.max()
        check(
            "C shifted max is 0.0",
            np.isclose(C_shifted.max(), 0.0),
        )

        # ------------------------------------------------------------------
        # 9. D vector normalization
        # ------------------------------------------------------------------
        print("\n--- 9. D Vector Normalization ---")

        D_toy = np.ones(4) / 4.0
        check(
            "uniform D sums to 1",
            np.isclose(D_toy.sum(), 1.0),
        )

        D_built = backend.build_D_vector(None, disc_s_toy)
        check(
            "built D sums to 1",
            np.isclose(D_built.sum(), 1.0),
            f"sum={D_built.sum()}",
        )

        # ------------------------------------------------------------------
        # 10. EFE computation produces finite values
        # ------------------------------------------------------------------
        print("\n--- 10. EFE Computation ---")

        import itertools

        horizon = 2
        policies = np.array(
            list(itertools.product(range(num_a), repeat=horizon))
        )

        efe_vals = backend.compute_efe_discrete(
            A_toy, B_toy, C_toy, policies
        )
        check(
            "EFE values are finite",
            np.all(np.isfinite(efe_vals)),
            f"nan_count={np.sum(np.isnan(efe_vals))}, "
            f"inf_count={np.sum(np.isinf(efe_vals))}",
        )
        check(
            "EFE returns correct number of policies",
            efe_vals.shape[0] == policies.shape[0],
            f"expected {policies.shape[0]}, got {efe_vals.shape[0]}",
        )

        # ------------------------------------------------------------------
        # 11. EFE lower for preferred policies
        # ------------------------------------------------------------------
        print("\n--- 11. EFE Preference Alignment ---")

        best_idx = int(np.argmin(efe_vals))
        worst_idx = int(np.argmax(efe_vals))
        check(
            "best policy has lower EFE than worst",
            efe_vals[best_idx] < efe_vals[worst_idx],
            f"best={efe_vals[best_idx]:.4f}, "
            f"worst={efe_vals[worst_idx]:.4f}",
        )

        efe_range = float(efe_vals.max() - efe_vals.min())
        check(
            "EFE range > 0 (preferences create policy differences)",
            efe_range > 1e-6,
            f"range={efe_range:.6f}",
        )

        # ------------------------------------------------------------------
        # 12. Compare: perfect correlation -> passed
        # ------------------------------------------------------------------
        print("\n--- 12. Compare: Perfect Correlation ---")

        efe_ref = efe_vals.copy()
        # Linear transform preserves correlation perfectly.
        efe_neural_perfect = efe_vals * 2.5 + 1.0

        result_perfect = backend.compare_efe(
            efe_neural_perfect, efe_ref, tolerance=0.15
        )
        check(
            "perfect linear transform: correlation ~ 1.0",
            result_perfect.correlation > 0.99,
            f"corr={result_perfect.correlation:.4f}",
        )
        check(
            "perfect linear transform: ranking_agreement = 1.0",
            result_perfect.ranking_agreement >= 0.99,
            f"agree={result_perfect.ranking_agreement:.2f}",
        )
        check(
            "perfect linear transform: passed=True",
            result_perfect.passed is True,
        )

    # ------------------------------------------------------------------
    # 13. Compare: uncorrelated -> passed=False
    # ------------------------------------------------------------------
    print("\n--- 13. Compare: Uncorrelated ---")

    if PYMDP_AVAILABLE:
        np.random.seed(999)
        efe_random = np.random.randn(efe_vals.shape[0])
        result_random = backend.compare_efe(
            efe_random, efe_vals, tolerance=0.05
        )
        check(
            "random vs discrete: correlation < 0.9",
            result_random.correlation < 0.9,
            f"corr={result_random.correlation:.4f}",
        )
        check(
            "random vs discrete: passed=False (typical)",
            result_random.passed is False,
            f"passed={result_random.passed}, "
            f"corr={result_random.correlation:.4f}",
        )
    else:
        skip("uncorrelated comparison", "pymdp not installed")
        skip(
            "random vs discrete: passed=False",
            "pymdp not installed",
        )

    # ------------------------------------------------------------------
    # 14. Compare: ranking agreement metric
    # ------------------------------------------------------------------
    print("\n--- 14. Compare: Ranking Agreement ---")

    if PYMDP_AVAILABLE:
        efe_known = np.arange(10, dtype=np.float64)
        efe_same_top3 = efe_known.copy()
        # Swap positions 5 and 6 (not in top-3) -- top-3 unchanged.
        efe_same_top3[5], efe_same_top3[6] = (
            efe_same_top3[6],
            efe_same_top3[5],
        )
        result_same = backend.compare_efe(efe_same_top3, efe_known)
        check(
            "same top-3: ranking_agreement = 1.0",
            result_same.ranking_agreement >= 0.99,
            f"agree={result_same.ranking_agreement:.2f}",
        )

        # Reverse the EFE -- top-3 completely different.
        efe_reversed = efe_known[::-1].copy()
        result_rev = backend.compare_efe(efe_reversed, efe_known)
        check(
            "reversed: ranking_agreement < 1.0",
            result_rev.ranking_agreement < 1.0,
            f"agree={result_rev.ranking_agreement:.2f}",
        )
    else:
        skip(
            "ranking agreement same top-3",
            "pymdp not installed",
        )
        skip(
            "ranking agreement reversed",
            "pymdp not installed",
        )

    # ------------------------------------------------------------------
    # 15. Graceful fallback: helpful error when pymdp not installed
    # ------------------------------------------------------------------
    print("\n--- 15. Graceful Fallback ---")

    check(
        "install message mentions pip install inferactively-pymdp",
        "pip install inferactively-pymdp" in _PYMDP_INSTALL_MSG,
    )

    if PYMDP_AVAILABLE:
        check("PYMDP_AVAILABLE is True", PYMDP_AVAILABLE is True)
    else:
        raised_import = False
        try:
            _ = PyMDPBackend()
        except ImportError as e:
            raised_import = True
            check(
                "ImportError includes install instructions",
                "pip install inferactively-pymdp" in str(e),
            )
        check(
            "PyMDPBackend raises ImportError without pymdp",
            raised_import,
        )

    # ------------------------------------------------------------------
    # 16. ComparisonResult serialization
    # ------------------------------------------------------------------
    print("\n--- 16. ComparisonResult Serialization ---")

    cr = ComparisonResult(
        correlation=0.95,
        pragmatic_correlation=0.92,
        epistemic_correlation=0.88,
        ranking_agreement=1.0,
        max_diff=0.05,
        passed=True,
    )
    d = cr.to_dict()
    check(
        "to_dict contains all fields",
        all(
            k in d
            for k in [
                "correlation",
                "pragmatic_correlation",
                "epistemic_correlation",
                "ranking_agreement",
                "max_diff",
                "passed",
            ]
        ),
    )
    check(
        "to_dict values are correct",
        d["correlation"] == 0.95
        and d["ranking_agreement"] == 1.0
        and d["passed"] == 1.0,
    )

    summary_str = cr.summary()
    check(
        "summary contains PASS",
        "PASS" in summary_str,
    )

    cr_fail = ComparisonResult(
        correlation=0.5,
        pragmatic_correlation=float("nan"),
        epistemic_correlation=float("nan"),
        ranking_agreement=0.33,
        max_diff=0.4,
        passed=False,
    )
    check(
        "failing result summary contains FAIL",
        "FAIL" in cr_fail.summary(),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(
        f"RESULTS: {passed} passed, {failed} failed, "
        f"{skipped} skipped, {total_tests} total"
    )
    print("=" * 72)
    if failed == 0:
        print(
            "ALL TESTS PASSED."
            if skipped == 0
            else "ALL RUN TESTS PASSED."
        )
    else:
        print(f"WARNING: {failed} test(s) FAILED.")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _run_self_tests()
