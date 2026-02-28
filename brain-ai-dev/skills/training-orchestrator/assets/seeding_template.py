"""
seeding_template.py -- Determinism and Seeding System for Brain-AI Training Pipeline

Provides reproducible training across the seven-phase cognitive pipeline by
managing RNG states, enforcing determinism flags, isolating random streams,
and verifying numerical tolerances for cross-run comparisons.

Classes:
    SeedManager          -- Central seed derivation and RNG state management.
    WorkerSeedFn         -- DataLoader worker_init_fn with epoch-aware seeding.
    DeterminismEnforcer  -- Applies / reports determinism flags (dev vs production).
    RNGStreamIsolator    -- Isolated torch.Generator instances per purpose.
    EpisodeSeedDeriver   -- Episode-level seed derivation for meta-learning (Phase 7).
    ToleranceChecker     -- Numerical comparison utilities for reproducibility checks.

Self-contained: no brain_ai imports required.  CUDA is handled gracefully when
unavailable.
"""

from __future__ import annotations

import hashlib
import os
import random
import struct
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HAS_CUDA = torch.cuda.is_available()

# Default phase offsets -- one per training phase (1-7).  Chosen to be
# well-separated so that per-phase random streams are statistically
# independent.
_DEFAULT_PHASE_OFFSETS: List[int] = [0, 1000, 2000, 3000, 4000, 5000, 6000]

# Component offsets used inside a single phase to separate streams.
_COMPONENT_OFFSETS: Dict[str, int] = {
    "augmentation": 0,
    "dropout": 10,
    "sampling": 20,
    "dataloader": 30,
}

# Known nondeterministic CUDA operations (PyTorch 2.x).
_NONDETERMINISTIC_OPS: List[str] = [
    "torch.Tensor.scatter_add_",
    "torch.Tensor.index_put_ (with accumulate=True)",
    "torch.Tensor.put_ (with accumulate=True)",
    "torch.bincount",
    "torch.nn.functional.interpolate (backward, some modes)",
    "torch.nn.functional.embedding_bag (backward, padding_idx)",
    "torch.nn.CTCLoss (backward)",
    "torch.nn.functional.grid_sample (backward, some modes)",
    "torch.scatter_reduce (some reduce modes)",
    "torch.cumsum (CUDA, float16)",
    "torch.Tensor.index_add_",
    "torch.gather (backward, CUDA)",
]


# =========================================================================
# SeedManager
# =========================================================================

class SeedManager:
    """Central seed derivation and RNG state management.

    Derives per-phase, per-component seeds from a single ``base_seed`` and
    provides utilities to snapshot / restore full RNG state across Python,
    NumPy, and PyTorch (CPU + CUDA).

    Parameters
    ----------
    base_seed : int
        Root seed for the entire training run.
    per_phase_offsets : list[int] | None
        Seven integer offsets, one per training phase.  When *None* the
        default well-separated offsets are used.
    enforce_deterministic : bool
        If *True*, call ``torch.use_deterministic_algorithms(True)`` when
        seeding.
    cudnn_benchmark : bool
        Value assigned to ``torch.backends.cudnn.benchmark``.
    """

    def __init__(
        self,
        base_seed: int = 1337,
        per_phase_offsets: Optional[List[int]] = None,
        enforce_deterministic: bool = True,
        cudnn_benchmark: bool = False,
    ) -> None:
        self.base_seed: int = base_seed
        self.per_phase_offsets: List[int] = (
            list(per_phase_offsets) if per_phase_offsets is not None
            else list(_DEFAULT_PHASE_OFFSETS)
        )
        if len(self.per_phase_offsets) < 7:
            raise ValueError(
                f"per_phase_offsets must have at least 7 entries, "
                f"got {len(self.per_phase_offsets)}"
            )
        self.enforce_deterministic: bool = enforce_deterministic
        self.cudnn_benchmark: bool = cudnn_benchmark

    # ----- seed derivation ------------------------------------------------

    def get_phase_seed(self, phase: int) -> int:
        """Return the seed for a specific training phase (1-indexed).

        Parameters
        ----------
        phase : int
            Training phase number, 1 through 7.

        Returns
        -------
        int
            ``base_seed + per_phase_offsets[phase - 1]``
        """
        if not 1 <= phase <= len(self.per_phase_offsets):
            raise ValueError(
                f"phase must be in [1, {len(self.per_phase_offsets)}], got {phase}"
            )
        return self.base_seed + self.per_phase_offsets[phase - 1]

    def get_component_seed(self, phase: int, component: str) -> int:
        """Return the seed for a specific component within a phase.

        Parameters
        ----------
        phase : int
            Training phase (1-indexed).
        component : str
            One of ``"augmentation"``, ``"dropout"``, ``"sampling"``,
            ``"dataloader"``.

        Returns
        -------
        int
            ``get_phase_seed(phase) + component_offset``
        """
        component = component.lower()
        if component not in _COMPONENT_OFFSETS:
            raise ValueError(
                f"Unknown component '{component}'. "
                f"Valid components: {list(_COMPONENT_OFFSETS.keys())}"
            )
        return self.get_phase_seed(phase) + _COMPONENT_OFFSETS[component]

    # ----- seed_everything ------------------------------------------------

    def seed_everything(self, phase: Optional[int] = None) -> int:
        """Set all RNG seeds and determinism flags.

        If *phase* is ``None`` the raw ``base_seed`` is used; otherwise the
        phase-specific seed is derived first.

        Returns
        -------
        int
            The actual seed that was applied.
        """
        seed = self.get_phase_seed(phase) if phase is not None else self.base_seed

        # Python stdlib
        random.seed(seed)

        # NumPy -- accepts uint32 range
        np.random.seed(seed % (2 ** 32))

        # PyTorch CPU
        torch.manual_seed(seed)

        # PyTorch CUDA (all devices)
        if _HAS_CUDA:
            torch.cuda.manual_seed_all(seed)

        # cuDNN flags
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = self.cudnn_benchmark

        # Full deterministic algorithms (may raise on unsupported ops)
        if self.enforce_deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:  # pragma: no cover -- depends on build
                warnings.warn(
                    "torch.use_deterministic_algorithms(True) failed; "
                    "some operations may be nondeterministic."
                )

        return seed

    # ----- generator factory ----------------------------------------------

    def create_generator(
        self, phase: int, component: str, device: str = "cpu"
    ) -> torch.Generator:
        """Return a ``torch.Generator`` seeded for *phase* + *component*.

        Parameters
        ----------
        phase : int
            Training phase (1-indexed).
        component : str
            Stream name (see ``get_component_seed``).
        device : str
            ``"cpu"`` or ``"cuda"``.

        Returns
        -------
        torch.Generator
        """
        seed = self.get_component_seed(phase, component)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return gen

    # ----- RNG state snapshot / restore -----------------------------------

    def capture_rng_state(self) -> Dict[str, Any]:
        """Capture the current RNG state for all backends.

        Returns
        -------
        dict
            Keys: ``"torch_cpu"``, ``"torch_cuda"`` (list, one per device),
            ``"numpy"``, ``"python"``.
        """
        state: Dict[str, Any] = {
            "torch_cpu": torch.random.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if _HAS_CUDA:
            state["torch_cuda"] = [
                torch.cuda.get_rng_state(d)
                for d in range(torch.cuda.device_count())
            ]
        else:
            state["torch_cuda"] = []
        return state

    def restore_rng_state(self, state_dict: Dict[str, Any]) -> None:
        """Restore RNG state from a dict previously returned by
        ``capture_rng_state``.

        Parameters
        ----------
        state_dict : dict
            Must contain keys ``"torch_cpu"``, ``"numpy"``, ``"python"``
            and optionally ``"torch_cuda"``.
        """
        torch.random.set_rng_state(state_dict["torch_cpu"])
        np.random.set_state(state_dict["numpy"])
        random.setstate(state_dict["python"])

        cuda_states = state_dict.get("torch_cuda", [])
        if _HAS_CUDA and cuda_states:
            for device_idx, s in enumerate(cuda_states):
                if device_idx < torch.cuda.device_count():
                    torch.cuda.set_rng_state(s, device_idx)

    # ----- serialization --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the manager configuration (not current RNG state) to a
        plain dict suitable for JSON / manifest storage."""
        return {
            "base_seed": self.base_seed,
            "per_phase_offsets": list(self.per_phase_offsets),
            "enforce_deterministic": self.enforce_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SeedManager":
        """Reconstruct a ``SeedManager`` from a dict produced by
        ``to_dict``."""
        return cls(
            base_seed=d["base_seed"],
            per_phase_offsets=d.get("per_phase_offsets"),
            enforce_deterministic=d.get("enforce_deterministic", True),
            cudnn_benchmark=d.get("cudnn_benchmark", False),
        )

    def __repr__(self) -> str:
        return (
            f"SeedManager(base_seed={self.base_seed}, "
            f"enforce_deterministic={self.enforce_deterministic}, "
            f"cudnn_benchmark={self.cudnn_benchmark})"
        )


# =========================================================================
# WorkerSeedFn
# =========================================================================

class WorkerSeedFn:
    """Callable ``worker_init_fn`` for ``torch.utils.data.DataLoader``.

    Each DataLoader worker receives a unique but deterministic seed derived
    from the base seed, the worker id, and the current epoch.  Call
    ``set_epoch`` before each epoch to re-seed persistent workers.

    Parameters
    ----------
    base_seed : int
        Root seed (usually from ``SeedManager.get_component_seed``).
    epoch : int
        Starting epoch (default 0).
    """

    def __init__(self, base_seed: int, epoch: int = 0) -> None:
        self.base_seed: int = base_seed
        self.epoch: int = epoch

    def set_epoch(self, epoch: int) -> None:
        """Update epoch counter so persistent workers are re-seeded."""
        self.epoch = epoch

    def __call__(self, worker_id: int) -> None:
        """Called by DataLoader for each worker process at the start of
        every epoch (or when the worker is created for non-persistent
        workers).

        Seed formula: ``base_seed + worker_id + epoch * 1000``
        """
        seed = self.base_seed + worker_id + self.epoch * 1000

        # Python stdlib
        random.seed(seed)

        # NumPy
        np.random.seed(seed % (2 ** 32))

        # PyTorch (worker-local CPU generator)
        torch.manual_seed(seed)

    def __repr__(self) -> str:
        return (
            f"WorkerSeedFn(base_seed={self.base_seed}, epoch={self.epoch})"
        )


# =========================================================================
# DeterminismEnforcer
# =========================================================================

class DeterminismEnforcer:
    """Apply and report determinism-related flags.

    Two modes are supported:

    * **dev** (strict) -- full determinism at the cost of speed.
    * **production** (relaxed) -- allows cuDNN auto-tuner and
      nondeterministic kernels for throughput.

    Parameters
    ----------
    mode : str
        ``"dev"`` or ``"production"``.
    """

    _CUBLAS_VAR = "CUBLAS_WORKSPACE_CONFIG"
    _CUBLAS_VALUE = ":4096:8"

    def __init__(self, mode: str = "dev") -> None:
        mode = mode.lower()
        if mode not in ("dev", "production"):
            raise ValueError(f"mode must be 'dev' or 'production', got '{mode}'")
        self.mode: str = mode

    # ----- flag resolution ------------------------------------------------

    def get_flags(self) -> Dict[str, Any]:
        """Return a dict of all flag settings that ``enforce`` will apply."""
        if self.mode == "dev":
            return {
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "use_deterministic_algorithms": True,
                "CUBLAS_WORKSPACE_CONFIG": self._CUBLAS_VALUE,
            }
        else:
            return {
                "cudnn_deterministic": True,
                "cudnn_benchmark": True,
                "use_deterministic_algorithms": False,
                "CUBLAS_WORKSPACE_CONFIG": None,
            }

    # ----- enforcement ----------------------------------------------------

    def enforce(self) -> Dict[str, Any]:
        """Apply all determinism flags and return the applied settings.

        Returns
        -------
        dict
            The same dict as ``get_flags()``, useful for logging / manifest.
        """
        flags = self.get_flags()

        torch.backends.cudnn.deterministic = flags["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = flags["cudnn_benchmark"]

        if flags["use_deterministic_algorithms"]:
            # Set env var *before* calling the API so that cuBLAS picks it up.
            os.environ[self._CUBLAS_VAR] = self._CUBLAS_VALUE
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                warnings.warn(
                    "torch.use_deterministic_algorithms(True) failed."
                )
        else:
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass  # older PyTorch builds

        return flags

    # ----- diagnostics ----------------------------------------------------

    def check_env_vars(self) -> bool:
        """Return *True* if environment variables are set correctly for the
        current mode.  In production mode this always returns *True*."""
        if self.mode != "dev":
            return True
        val = os.environ.get(self._CUBLAS_VAR, "")
        return val == self._CUBLAS_VALUE

    def warn_nondeterministic_ops(self) -> List[str]:
        """Return a list of known nondeterministic operations.  Useful for
        logging warnings at the start of a training run."""
        return list(_NONDETERMINISTIC_OPS)

    def __repr__(self) -> str:
        return f"DeterminismEnforcer(mode='{self.mode}')"


# =========================================================================
# RNGStreamIsolator
# =========================================================================

class RNGStreamIsolator:
    """Maintain isolated ``torch.Generator`` instances for different random
    number consumers so that changes to one stream (e.g. data augmentation)
    do not affect another (e.g. DataLoader shuffling).

    Each generator is independently seeded via a ``SeedManager``.

    Note on dropout
    ---------------
    Standard ``torch.nn.Dropout`` does **not** accept an external
    ``torch.Generator``.  To use an isolated stream for dropout you must
    either:

    1. Replace ``nn.Dropout`` with a custom module that calls
       ``torch.bernoulli(p, generator=gen)`` manually, **or**
    2. Temporarily swap the global RNG state (see ``SeedManager.
       capture_rng_state`` / ``restore_rng_state``) around the forward pass
       of the dropout layer.

    The generator returned by ``get_dropout_rng`` is provided for approach
    (1); approach (2) is left to the caller.

    Parameters
    ----------
    seed_manager : SeedManager
        Manager used to derive seeds.
    device : str
        Device for generators (``"cpu"`` or ``"cuda"``).
    """

    _STREAM_NAMES: Tuple[str, ...] = (
        "augmentation",
        "dropout",
        "sampling",
        "dataloader",
    )

    def __init__(
        self, seed_manager: SeedManager, device: str = "cpu"
    ) -> None:
        self.seed_manager: SeedManager = seed_manager
        self.device: str = device
        # {phase: {stream_name: Generator}}
        self._generators: Dict[int, Dict[str, torch.Generator]] = {}

    def _ensure_phase(self, phase: int) -> None:
        """Lazily create generators for *phase* if they don't exist yet."""
        if phase not in self._generators:
            self._generators[phase] = {}
            for name in self._STREAM_NAMES:
                gen = self.seed_manager.create_generator(
                    phase, name, device=self.device
                )
                self._generators[phase][name] = gen

    def _get(self, phase: int, name: str) -> torch.Generator:
        self._ensure_phase(phase)
        return self._generators[phase][name]

    # ----- public accessors -----------------------------------------------

    def get_augmentation_rng(self, phase: int) -> torch.Generator:
        """Generator for data augmentation transforms."""
        return self._get(phase, "augmentation")

    def get_dropout_rng(self, phase: int) -> torch.Generator:
        """Generator for dropout (see class docstring for usage notes)."""
        return self._get(phase, "dropout")

    def get_sampling_rng(self, phase: int) -> torch.Generator:
        """Generator for RL rollouts and episode sampling."""
        return self._get(phase, "sampling")

    def get_dataloader_rng(self, phase: int) -> torch.Generator:
        """Generator for DataLoader shuffling."""
        return self._get(phase, "dataloader")

    # ----- reset ----------------------------------------------------------

    def reset_all(self, phase: int) -> None:
        """Re-derive and reset all generators for *phase*.

        Useful at the start of each phase to return to a known state.
        """
        # Discard existing generators so _ensure_phase recreates them.
        self._generators.pop(phase, None)
        self._ensure_phase(phase)

    def __repr__(self) -> str:
        phases = sorted(self._generators.keys()) if self._generators else []
        return (
            f"RNGStreamIsolator(device='{self.device}', "
            f"active_phases={phases})"
        )


# =========================================================================
# EpisodeSeedDeriver
# =========================================================================

class EpisodeSeedDeriver:
    """Derive deterministic seeds for meta-learning episodes (Phase 7).

    Uses SHA-256 hashing so that the derivation is stable across platforms
    and Python versions.
    """

    @staticmethod
    def get_episode_seed(
        global_seed: int, epoch: int, episode_idx: int
    ) -> int:
        """Derive a deterministic seed for a single episode.

        Parameters
        ----------
        global_seed : int
            Top-level seed (e.g. ``SeedManager.base_seed``).
        epoch : int
            Current epoch number.
        episode_idx : int
            Index of the episode within the epoch.

        Returns
        -------
        int
            A 32-bit unsigned integer seed.
        """
        payload = f"{global_seed}:{epoch}:{episode_idx}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        # Unpack first 4 bytes as unsigned 32-bit int (big-endian).
        return struct.unpack(">I", digest[:4])[0]

    @staticmethod
    def get_class_sample_seed(episode_seed: int, class_idx: int) -> int:
        """Further derive a seed for per-class sampling within an episode.

        Parameters
        ----------
        episode_seed : int
            Seed returned by ``get_episode_seed``.
        class_idx : int
            Index of the class being sampled.

        Returns
        -------
        int
            A 32-bit unsigned integer seed.
        """
        payload = f"{episode_seed}:class:{class_idx}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return struct.unpack(">I", digest[:4])[0]


# =========================================================================
# ToleranceChecker
# =========================================================================

class ToleranceChecker:
    """Numerical comparison utilities for verifying reproducibility.

    All methods return *True* when the values are *within* tolerance and
    *False* otherwise.
    """

    _EPSILON: float = 1e-12  # guard against division by zero

    @classmethod
    def check_relative(
        cls,
        a: float,
        b: float,
        tolerance: float = 0.02,
    ) -> bool:
        """Relative tolerance check.

        ``|a - b| / max(|a|, epsilon) < tolerance``
        """
        denom = max(abs(a), cls._EPSILON)
        return abs(a - b) / denom < tolerance

    @staticmethod
    def check_absolute(
        a: float,
        b: float,
        tolerance: float = 0.01,
    ) -> bool:
        """Absolute tolerance check.  ``|a - b| < tolerance``."""
        return abs(a - b) < tolerance

    @staticmethod
    def check_tensor_close(
        a: torch.Tensor,
        b: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """Wraps ``torch.testing.assert_close``; returns bool instead of
        raising."""
        try:
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
            return True
        except AssertionError:
            return False

    @staticmethod
    def check_cosine_similarity(
        a: torch.Tensor,
        b: torch.Tensor,
        threshold: float = 0.99,
    ) -> bool:
        """Cosine similarity check (useful for cross-device comparisons
        where bit-exact equality is not expected).

        Returns *True* when similarity >= *threshold*.
        """
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        cos = torch.nn.functional.cosine_similarity(
            a_flat.unsqueeze(0), b_flat.unsqueeze(0)
        )
        return cos.item() >= threshold

    @classmethod
    def compare_metric_dicts(
        cls,
        original: Dict[str, float],
        reproduced: Dict[str, float],
        tolerances: Optional[Dict[str, float]] = None,
        default_tolerance: float = 0.02,
    ) -> Dict[str, bool]:
        """Compare two metric dicts key-by-key.

        Parameters
        ----------
        original : dict[str, float]
            Baseline metrics.
        reproduced : dict[str, float]
            Metrics from a reproduction run.
        tolerances : dict[str, float] | None
            Per-metric relative tolerances.  Keys not present fall back to
            *default_tolerance*.
        default_tolerance : float
            Fallback relative tolerance.

        Returns
        -------
        dict[str, bool]
            Pass / fail per metric.
        """
        if tolerances is None:
            tolerances = {}
        results: Dict[str, bool] = {}
        all_keys = set(original.keys()) | set(reproduced.keys())
        for key in sorted(all_keys):
            if key not in original or key not in reproduced:
                results[key] = False
                continue
            tol = tolerances.get(key, default_tolerance)
            results[key] = cls.check_relative(original[key], reproduced[key], tol)
        return results


# =========================================================================
# Self-test suite
# =========================================================================

def _banner(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def _pass(name: str) -> None:
    print(f"  [PASS] {name}")


def _fail(name: str, detail: str = "") -> None:
    msg = f"  [FAIL] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    # We do NOT sys.exit here; we collect all failures at the end.


def _run_tests() -> None:
    """Execute the full self-test suite."""

    passed = 0
    failed = 0
    failure_details: List[str] = []

    def check(condition: bool, name: str, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            _pass(name)
            passed += 1
        else:
            _fail(name, detail)
            failed += 1
            failure_details.append(name)

    # Disable deterministic algorithms globally for the test suite since
    # we test DeterminismEnforcer explicitly and don't want side effects.
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

    # ==================================================================
    # SeedManager tests
    # ==================================================================
    _banner("SeedManager")

    sm = SeedManager(base_seed=1337)

    # 1. Phase seed derivation
    check(
        sm.get_phase_seed(1) == 1337 + 0,
        "Phase 1 seed derivation",
    )
    check(
        sm.get_phase_seed(4) == 1337 + 3000,
        "Phase 4 seed derivation",
    )
    check(
        sm.get_phase_seed(7) == 1337 + 6000,
        "Phase 7 seed derivation",
    )

    # 2. Consistency across repeated calls
    check(
        sm.get_phase_seed(3) == sm.get_phase_seed(3),
        "Phase seed consistency (repeated call)",
    )

    # 3. Per-phase seeds differ
    seeds = [sm.get_phase_seed(p) for p in range(1, 8)]
    check(
        len(set(seeds)) == 7,
        "All 7 phase seeds are unique",
    )

    # 4. Component seed derivation
    aug_seed = sm.get_component_seed(2, "augmentation")
    drop_seed = sm.get_component_seed(2, "dropout")
    samp_seed = sm.get_component_seed(2, "sampling")
    dl_seed = sm.get_component_seed(2, "dataloader")
    check(
        len({aug_seed, drop_seed, samp_seed, dl_seed}) == 4,
        "Component seeds within same phase differ",
    )
    check(
        aug_seed == sm.get_phase_seed(2) + 0,
        "Augmentation offset is 0",
    )
    check(
        drop_seed == sm.get_phase_seed(2) + 10,
        "Dropout offset is 10",
    )
    check(
        samp_seed == sm.get_phase_seed(2) + 20,
        "Sampling offset is 20",
    )
    check(
        dl_seed == sm.get_phase_seed(2) + 30,
        "Dataloader offset is 30",
    )

    # 5. seed_everything reproducibility
    sm.seed_everything(phase=1)
    a1 = torch.randn(5)
    sm.seed_everything(phase=1)
    a2 = torch.randn(5)
    check(
        torch.equal(a1, a2),
        "seed_everything produces identical torch.randn (phase 1)",
    )

    # 6. Different phase gives different output
    sm.seed_everything(phase=1)
    b1 = torch.randn(5)
    sm.seed_everything(phase=2)
    b2 = torch.randn(5)
    check(
        not torch.equal(b1, b2),
        "Different phases produce different torch.randn",
    )

    # 7. seed_everything sets Python random
    sm.seed_everything(phase=3)
    py1 = [random.random() for _ in range(5)]
    sm.seed_everything(phase=3)
    py2 = [random.random() for _ in range(5)]
    check(py1 == py2, "seed_everything sets Python random consistently")

    # 8. seed_everything sets numpy
    sm.seed_everything(phase=3)
    np1 = np.random.rand(5).tolist()
    sm.seed_everything(phase=3)
    np2 = np.random.rand(5).tolist()
    check(np1 == np2, "seed_everything sets NumPy random consistently")

    # 9. create_generator returns seeded generator
    g1 = sm.create_generator(1, "augmentation")
    g2 = sm.create_generator(1, "augmentation")
    t1 = torch.randn(10, generator=g1)
    t2 = torch.randn(10, generator=g2)
    check(torch.equal(t1, t2), "create_generator produces reproducible output")

    # 10. RNG state capture/restore round-trip
    sm.seed_everything(phase=5)
    state = sm.capture_rng_state()
    vals_before = (torch.randn(3), np.random.rand(3).copy(), random.random())
    sm.restore_rng_state(state)
    vals_after = (torch.randn(3), np.random.rand(3).copy(), random.random())
    check(
        torch.equal(vals_before[0], vals_after[0]),
        "RNG capture/restore: torch CPU",
    )
    check(
        np.array_equal(vals_before[1], vals_after[1]),
        "RNG capture/restore: NumPy",
    )
    check(
        vals_before[2] == vals_after[2],
        "RNG capture/restore: Python random",
    )

    # 11. 10 consecutive seed_everything calls produce identical torch.randn
    reference = None
    all_match = True
    for i in range(10):
        sm.seed_everything(phase=2)
        sample = torch.randn(20)
        if reference is None:
            reference = sample
        elif not torch.equal(reference, sample):
            all_match = False
            break
    check(all_match, "10 consecutive seed_everything calls give identical randn")

    # 12. Serialization round-trip
    d = sm.to_dict()
    sm2 = SeedManager.from_dict(d)
    check(sm2.base_seed == sm.base_seed, "Serialization: base_seed")
    check(
        sm2.per_phase_offsets == sm.per_phase_offsets,
        "Serialization: per_phase_offsets",
    )
    check(
        sm2.enforce_deterministic == sm.enforce_deterministic,
        "Serialization: enforce_deterministic",
    )
    check(
        sm2.cudnn_benchmark == sm.cudnn_benchmark,
        "Serialization: cudnn_benchmark",
    )

    # 13. from_dict produces same phase seeds
    for p in range(1, 8):
        if sm2.get_phase_seed(p) != sm.get_phase_seed(p):
            check(False, "Serialization: phase seed mismatch after from_dict")
            break
    else:
        check(True, "Serialization: all phase seeds match after from_dict")

    # 14. Invalid phase raises
    raised = False
    try:
        sm.get_phase_seed(0)
    except ValueError:
        raised = True
    check(raised, "get_phase_seed(0) raises ValueError")

    # 15. Invalid component raises
    raised = False
    try:
        sm.get_component_seed(1, "invalid")
    except ValueError:
        raised = True
    check(raised, "get_component_seed with invalid name raises ValueError")

    # ==================================================================
    # WorkerSeedFn tests
    # ==================================================================
    _banner("WorkerSeedFn")

    wf = WorkerSeedFn(base_seed=42, epoch=0)

    # 16. Different workers get different seeds
    wf(0)
    w0_val = random.random()
    wf(1)
    w1_val = random.random()
    check(w0_val != w1_val, "Workers 0 and 1 produce different random values")

    # 17. Same worker, same epoch -> same seed
    wf(3)
    r1 = random.random()
    wf(3)
    r2 = random.random()
    check(r1 == r2, "Same worker+epoch produces same random value")

    # 18. Same worker, same epoch -> same numpy
    wf(3)
    n1 = np.random.rand()
    wf(3)
    n2 = np.random.rand()
    check(n1 == n2, "Same worker+epoch produces same numpy value")

    # 19. Same worker, same epoch -> same torch
    wf(3)
    t1 = torch.randn(1).item()
    wf(3)
    t2 = torch.randn(1).item()
    check(t1 == t2, "Same worker+epoch produces same torch value")

    # 20. Different epoch changes seed
    wf.set_epoch(0)
    wf(2)
    e0 = random.random()
    wf.set_epoch(1)
    wf(2)
    e1 = random.random()
    check(e0 != e1, "Different epochs produce different seeds for same worker")

    # 21. set_epoch updates internal state
    wf.set_epoch(5)
    check(wf.epoch == 5, "set_epoch updates epoch attribute")

    # ==================================================================
    # DeterminismEnforcer tests
    # ==================================================================
    _banner("DeterminismEnforcer")

    de_dev = DeterminismEnforcer(mode="dev")
    de_prod = DeterminismEnforcer(mode="production")

    # 22. Dev flags
    dev_flags = de_dev.get_flags()
    check(dev_flags["cudnn_deterministic"] is True, "Dev: cudnn_deterministic=True")
    check(dev_flags["cudnn_benchmark"] is False, "Dev: cudnn_benchmark=False")
    check(
        dev_flags["use_deterministic_algorithms"] is True,
        "Dev: use_deterministic_algorithms=True",
    )
    check(
        dev_flags["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8",
        "Dev: CUBLAS_WORKSPACE_CONFIG=:4096:8",
    )

    # 23. Production flags
    prod_flags = de_prod.get_flags()
    check(
        prod_flags["cudnn_deterministic"] is True,
        "Prod: cudnn_deterministic=True",
    )
    check(prod_flags["cudnn_benchmark"] is True, "Prod: cudnn_benchmark=True")
    check(
        prod_flags["use_deterministic_algorithms"] is False,
        "Prod: use_deterministic_algorithms=False",
    )
    check(
        prod_flags["CUBLAS_WORKSPACE_CONFIG"] is None,
        "Prod: CUBLAS_WORKSPACE_CONFIG=None",
    )

    # 24. Enforce dev mode applies flags
    de_dev.enforce()
    check(
        torch.backends.cudnn.deterministic is True,
        "Dev enforce: cudnn.deterministic applied",
    )
    check(
        torch.backends.cudnn.benchmark is False,
        "Dev enforce: cudnn.benchmark applied",
    )

    # 25. check_env_vars in dev mode after enforce
    check(
        de_dev.check_env_vars(),
        "Dev: CUBLAS_WORKSPACE_CONFIG set correctly after enforce",
    )

    # 26. Enforce production mode
    de_prod.enforce()
    check(
        torch.backends.cudnn.benchmark is True,
        "Prod enforce: cudnn.benchmark=True applied",
    )

    # 27. Production check_env_vars always True
    check(de_prod.check_env_vars(), "Prod: check_env_vars always True")

    # 28. warn_nondeterministic_ops returns non-empty list
    ops = de_dev.warn_nondeterministic_ops()
    check(len(ops) > 5, "warn_nondeterministic_ops returns known ops list")

    # 29. Invalid mode raises
    raised = False
    try:
        DeterminismEnforcer(mode="invalid")
    except ValueError:
        raised = True
    check(raised, "Invalid DeterminismEnforcer mode raises ValueError")

    # Reset deterministic algorithms to False after dev enforce test
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

    # ==================================================================
    # RNGStreamIsolator tests
    # ==================================================================
    _banner("RNGStreamIsolator")

    sm_iso = SeedManager(base_seed=9999, enforce_deterministic=False)
    iso = RNGStreamIsolator(sm_iso)

    # 30. Generators are created on first access
    gen_aug = iso.get_augmentation_rng(1)
    check(isinstance(gen_aug, torch.Generator), "get_augmentation_rng returns Generator")

    # 31. Same call returns same object
    gen_aug2 = iso.get_augmentation_rng(1)
    check(gen_aug is gen_aug2, "Repeated call returns same Generator object")

    # 32. Different streams return different objects
    gen_dl = iso.get_dataloader_rng(1)
    check(gen_aug is not gen_dl, "augmentation and dataloader are different objects")

    # 33. Independence: consuming one generator doesn't affect another
    iso.reset_all(1)
    g_a = iso.get_augmentation_rng(1)
    g_d = iso.get_dataloader_rng(1)

    # Snapshot dataloader state
    val_d_before = torch.randn(5, generator=g_d).clone()

    # Consume augmentation heavily
    for _ in range(100):
        torch.randn(50, generator=g_a)

    # Reset and re-check dataloader
    iso.reset_all(1)
    g_d_reset = iso.get_dataloader_rng(1)
    val_d_after = torch.randn(5, generator=g_d_reset)
    check(
        torch.equal(val_d_before, val_d_after),
        "Dataloader stream unaffected by augmentation consumption (after reset)",
    )

    # 34. reset_all re-creates generators
    old_gen = iso.get_sampling_rng(2)
    iso.reset_all(2)
    new_gen = iso.get_sampling_rng(2)
    check(old_gen is not new_gen, "reset_all creates new Generator objects")

    # 35. Reset produces same output as initial creation
    sm_iso2 = SeedManager(base_seed=9999, enforce_deterministic=False)
    iso2 = RNGStreamIsolator(sm_iso2)
    g1_fresh = iso2.get_augmentation_rng(3)
    v1 = torch.randn(10, generator=g1_fresh)

    iso2.reset_all(3)
    g1_reset = iso2.get_augmentation_rng(3)
    v2 = torch.randn(10, generator=g1_reset)
    check(torch.equal(v1, v2), "reset_all restores generator to initial state")

    # 36. Different phases have different generators
    gen_p1 = iso.get_sampling_rng(1)
    gen_p5 = iso.get_sampling_rng(5)
    check(gen_p1 is not gen_p5, "Different phases return different generators")

    # 37. Different phases produce different values
    iso.reset_all(1)
    iso.reset_all(5)
    v_p1 = torch.randn(10, generator=iso.get_sampling_rng(1))
    v_p5 = torch.randn(10, generator=iso.get_sampling_rng(5))
    check(
        not torch.equal(v_p1, v_p5),
        "Different phases produce different random values",
    )

    # ==================================================================
    # EpisodeSeedDeriver tests
    # ==================================================================
    _banner("EpisodeSeedDeriver")

    esd = EpisodeSeedDeriver()

    # 38. Determinism
    s1 = esd.get_episode_seed(1337, 0, 0)
    s2 = esd.get_episode_seed(1337, 0, 0)
    check(s1 == s2, "Episode seed is deterministic")

    # 39. Different episodes get different seeds
    s3 = esd.get_episode_seed(1337, 0, 1)
    check(s1 != s3, "Different episode_idx yields different seed")

    # 40. Different epochs get different seeds
    s4 = esd.get_episode_seed(1337, 1, 0)
    check(s1 != s4, "Different epoch yields different seed")

    # 41. Different global seeds
    s5 = esd.get_episode_seed(42, 0, 0)
    check(s1 != s5, "Different global_seed yields different seed")

    # 42. Returns uint32 range
    check(0 <= s1 < 2**32, "Episode seed is in uint32 range")

    # 43. Class sample seed determinism
    cs1 = esd.get_class_sample_seed(s1, 0)
    cs2 = esd.get_class_sample_seed(s1, 0)
    check(cs1 == cs2, "Class sample seed is deterministic")

    # 44. Different classes get different seeds
    cs3 = esd.get_class_sample_seed(s1, 1)
    check(cs1 != cs3, "Different class_idx yields different class sample seed")

    # 45. Class sample seed is in uint32 range
    check(0 <= cs1 < 2**32, "Class sample seed is in uint32 range")

    # 46. Many episode seeds are all unique
    many_seeds = {
        esd.get_episode_seed(1337, e, i)
        for e in range(10)
        for i in range(10)
    }
    check(len(many_seeds) == 100, "100 episode seeds are all unique")

    # ==================================================================
    # ToleranceChecker tests
    # ==================================================================
    _banner("ToleranceChecker")

    tc = ToleranceChecker()

    # 47. Relative tolerance pass
    check(
        tc.check_relative(1.0, 1.01, tolerance=0.02),
        "Relative: 1.0 vs 1.01 within 2%",
    )

    # 48. Relative tolerance fail
    check(
        not tc.check_relative(1.0, 1.05, tolerance=0.02),
        "Relative: 1.0 vs 1.05 outside 2%",
    )

    # 49. Absolute tolerance pass
    check(
        tc.check_absolute(0.5, 0.505, tolerance=0.01),
        "Absolute: 0.5 vs 0.505 within 0.01",
    )

    # 50. Absolute tolerance fail
    check(
        not tc.check_absolute(0.5, 0.52, tolerance=0.01),
        "Absolute: 0.5 vs 0.52 outside 0.01",
    )

    # 51. Tensor close pass
    ta = torch.tensor([1.0, 2.0, 3.0])
    tb = torch.tensor([1.0, 2.0, 3.0])
    check(tc.check_tensor_close(ta, tb), "Tensor close: identical tensors")

    # 52. Tensor close fail
    tc_far = torch.tensor([1.0, 2.0, 4.0])
    check(
        not tc.check_tensor_close(ta, tc_far, rtol=1e-5, atol=1e-8),
        "Tensor close: differing tensors",
    )

    # 53. Cosine similarity pass
    va = torch.tensor([1.0, 0.0, 0.0])
    vb = torch.tensor([0.9999, 0.001, 0.0])
    check(
        tc.check_cosine_similarity(va, vb, threshold=0.99),
        "Cosine similarity: near-parallel vectors pass at 0.99",
    )

    # 54. Cosine similarity fail
    vc = torch.tensor([0.0, 1.0, 0.0])
    check(
        not tc.check_cosine_similarity(va, vc, threshold=0.99),
        "Cosine similarity: orthogonal vectors fail at 0.99",
    )

    # 55. compare_metric_dicts all pass
    orig = {"loss": 0.50, "acc": 0.90, "f1": 0.85}
    repro = {"loss": 0.505, "acc": 0.898, "f1": 0.849}
    results = tc.compare_metric_dicts(orig, repro, default_tolerance=0.02)
    check(
        all(results.values()),
        "compare_metric_dicts: all metrics within tolerance",
    )

    # 56. compare_metric_dicts partial fail
    repro_bad = {"loss": 0.50, "acc": 0.80, "f1": 0.85}
    results_bad = tc.compare_metric_dicts(orig, repro_bad, default_tolerance=0.02)
    check(
        results_bad["loss"] is True and results_bad["acc"] is False,
        "compare_metric_dicts: detects out-of-tolerance metric",
    )

    # 57. compare_metric_dicts with per-metric tolerance
    results_custom = tc.compare_metric_dicts(
        orig, repro_bad,
        tolerances={"acc": 0.15},
        default_tolerance=0.02,
    )
    check(
        results_custom["acc"] is True,
        "compare_metric_dicts: per-metric tolerance override works",
    )

    # 58. compare_metric_dicts missing key
    results_missing = tc.compare_metric_dicts(
        {"a": 1.0}, {"b": 1.0}, default_tolerance=0.02
    )
    check(
        results_missing.get("a") is False and results_missing.get("b") is False,
        "compare_metric_dicts: missing keys report False",
    )

    # 59. Relative tolerance with zero denominator
    # With a=0.0, denom = max(|0.0|, 1e-12) = 1e-12.
    # |0.0 - 1e-14| / 1e-12 = 0.01 < 0.02, so this should pass.
    check(
        tc.check_relative(0.0, 1e-14, tolerance=0.02),
        "Relative: near-zero values handled without division error",
    )

    # ==================================================================
    # Integration tests
    # ==================================================================
    _banner("Integration")

    # 60. Full pipeline: seed, train step, capture, restore, repeat
    sm_int = SeedManager(base_seed=2023, enforce_deterministic=False)
    sm_int.seed_everything(phase=1)
    state_before = sm_int.capture_rng_state()

    # Simulate a training step
    x = torch.randn(4, 16)
    w = torch.randn(16, 8)
    out1 = x @ w
    noise1 = torch.randn(4, 8)
    result1 = out1 + noise1

    # Restore and replay
    sm_int.restore_rng_state(state_before)
    x2 = torch.randn(4, 16)
    w2 = torch.randn(16, 8)
    out2 = x2 @ w2
    noise2 = torch.randn(4, 8)
    result2 = out2 + noise2

    check(torch.equal(result1, result2), "Full pipeline replay is bit-exact")

    # 61. WorkerSeedFn + SeedManager integration
    sm_wk = SeedManager(base_seed=7777, enforce_deterministic=False)
    dl_seed = sm_wk.get_component_seed(3, "dataloader")
    wfn = WorkerSeedFn(base_seed=dl_seed, epoch=0)
    wfn(0)
    val_a = torch.randn(5)
    wfn(0)
    val_b = torch.randn(5)
    check(
        torch.equal(val_a, val_b),
        "WorkerSeedFn + SeedManager: reproducible worker output",
    )

    # 62. EpisodeSeedDeriver + SeedManager integration
    sm_ep = SeedManager(base_seed=555, enforce_deterministic=False)
    phase7_seed = sm_ep.get_phase_seed(7)
    ep_seed = EpisodeSeedDeriver.get_episode_seed(phase7_seed, epoch=0, episode_idx=0)
    random.seed(ep_seed)
    r_val1 = random.random()
    random.seed(ep_seed)
    r_val2 = random.random()
    check(r_val1 == r_val2, "EpisodeSeedDeriver + SeedManager: deterministic episode")

    # 63. RNGStreamIsolator + SeedManager full round-trip
    sm_full = SeedManager(base_seed=3141, enforce_deterministic=False)
    isolator = RNGStreamIsolator(sm_full)
    gen_a = isolator.get_augmentation_rng(2)
    gen_d = isolator.get_dataloader_rng(2)

    va1 = torch.randn(10, generator=gen_a)
    vd1 = torch.randn(10, generator=gen_d)

    isolator.reset_all(2)
    gen_a2 = isolator.get_augmentation_rng(2)
    gen_d2 = isolator.get_dataloader_rng(2)

    va2 = torch.randn(10, generator=gen_a2)
    vd2 = torch.randn(10, generator=gen_d2)

    check(torch.equal(va1, va2), "Isolator reset: augmentation stream reproducible")
    check(torch.equal(vd1, vd2), "Isolator reset: dataloader stream reproducible")

    # 64. DeterminismEnforcer flags round-trip through get_flags
    de = DeterminismEnforcer(mode="dev")
    flags = de.get_flags()
    applied = de.enforce()
    check(flags == applied, "DeterminismEnforcer: get_flags matches enforce return")
    # Reset
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

    # 65. SeedManager with custom offsets
    custom_offsets = [0, 100, 200, 300, 400, 500, 600]
    sm_custom = SeedManager(base_seed=10, per_phase_offsets=custom_offsets)
    check(
        sm_custom.get_phase_seed(3) == 210,
        "Custom offsets: phase 3 seed = 10 + 200",
    )

    # 66. SeedManager too few offsets raises
    raised = False
    try:
        SeedManager(base_seed=0, per_phase_offsets=[1, 2, 3])
    except ValueError:
        raised = True
    check(raised, "SeedManager with < 7 offsets raises ValueError")

    # 67. ToleranceChecker check_tensor_close with tolerances
    t_a = torch.tensor([1.0, 2.0])
    t_b = torch.tensor([1.0 + 1e-6, 2.0 - 1e-6])
    check(
        tc.check_tensor_close(t_a, t_b, rtol=1e-4, atol=1e-4),
        "check_tensor_close: passes with reasonable tolerances",
    )

    # 68. Cosine similarity with identical vectors
    ident = torch.randn(100)
    check(
        tc.check_cosine_similarity(ident, ident, threshold=1.0),
        "Cosine similarity: identical vectors give 1.0",
    )

    # 69. WorkerSeedFn epoch boundary
    wf2 = WorkerSeedFn(base_seed=100)
    wf2.set_epoch(999)
    wf2(0)
    high_epoch_val = random.random()
    wf2.set_epoch(999)
    wf2(0)
    high_epoch_val2 = random.random()
    check(
        high_epoch_val == high_epoch_val2,
        "WorkerSeedFn: high epoch number is still deterministic",
    )

    # 70. SeedManager repr
    check(
        "1337" in repr(sm),
        "SeedManager repr contains base_seed",
    )

    # ==================================================================
    # Summary
    # ==================================================================
    _banner("SUMMARY")

    total = passed + failed
    print(f"\n  Total : {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failure_details:
        print("\n  Failed tests:")
        for name in failure_details:
            print(f"    - {name}")

    if failed > 0:
        print(f"\n  EXIT CODE: 1 ({failed} failure(s))")
        sys.exit(1)
    else:
        print("\n  All tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    _run_tests()
