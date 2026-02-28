"""
base_classes_template.py
========================
Template containing all 8 ABC base classes for the world model scaffold.

This file is consumed by ScaffoldGenerator to write the base.py files for each
subpackage. It can also be executed directly to validate that all ABCs are
correctly defined and that the TypeError enforcement works as expected.

IMPORTANT: Set inference mode via module.train(False), not via .eval().

Usage (self-test):
    python base_classes_template.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 1. BaseEncoder
# ---------------------------------------------------------------------------


class BaseEncoder(ABC):
    """Abstract base class for observation encoders.

    An encoder maps raw observations (images, point clouds, sensor readings)
    to a fixed-dimensional latent vector.  It is stateless: the same input
    always produces the same output given the same parameters.

    Subclasses that inherit from nn.Module should inherit from both::

        class MyEncoder(nn.Module, BaseEncoder): ...

    Abstract methods that must be implemented
    -----------------------------------------
    forward(x)          : (B, *obs_shape) -> (B, embed_dim)
    get_embed_dim()     : returns int embed dimension
    get_output_shape()  : returns tuple, e.g. (embed_dim,) or (H, W, C)
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch of observations into latent embeddings.

        Parameters
        ----------
        x : Tensor
            Shape (B, *obs_shape).  The observation batch.

        Returns
        -------
        Tensor
            Shape (B, embed_dim).  The last dimension equals get_embed_dim().

        Raises
        ------
        ValueError
            If x has the wrong number of dimensions for this encoder.
        """

    @abstractmethod
    def get_embed_dim(self) -> int:
        """Return the integer dimensionality of the output embedding.

        Returns
        -------
        int
            Strictly positive.  Consistent with the last dim of forward() output.
        """

    @abstractmethod
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return the full shape of one output embedding (no batch dimension).

        Returns
        -------
        Tuple[int, ...]
            e.g. (512,) for a flat embedding, (16, 16, 256) for a spatial map.
            For flat encoders: get_output_shape() == (get_embed_dim(),).
        """


# ---------------------------------------------------------------------------
# 2. BaseDynamics
# ---------------------------------------------------------------------------


class BaseDynamics(ABC):
    """Abstract base class for latent-space dynamics models.

    A dynamics model predicts the next latent state given the current state
    and an action.  It can also roll out imagined trajectories using a policy.

    Dimensional contract
    --------------------
    get_state_dim() must equal encoder.get_embed_dim() for any paired encoder.
    BaseWorldModel validates this at construction time.
    """

    @abstractmethod
    def step(self, state: Tensor, action: Tensor) -> Tensor:
        """Predict the next latent state.

        Parameters
        ----------
        state : Tensor
            Shape (B, state_dim).  Current latent state.
        action : Tensor
            Shape (B, action_dim).  Action taken in the current state.

        Returns
        -------
        Tensor
            Shape (B, state_dim).  Predicted next latent state.

        Raises
        ------
        ValueError
            If state.shape[-1] != get_state_dim().
        """

    @abstractmethod
    def imagine(
        self, state: Tensor, policy: Callable[[Tensor], Tensor], horizon: int
    ) -> Tensor:
        """Roll out an imagined trajectory using a policy.

        Parameters
        ----------
        state : Tensor
            Shape (B, state_dim).  Initial latent state.
        policy : Callable[[Tensor], Tensor]
            Maps (B, state_dim) -> (B, action_dim).  Called once per step.
        horizon : int
            Number of steps to imagine.  Must be >= 1.

        Returns
        -------
        Tensor
            Shape (B, horizon, state_dim).  Imagined state sequence,
            not including the initial state.

        Raises
        ------
        ValueError
            If horizon < 1.
        TypeError
            If policy is not callable.
        """

    @abstractmethod
    def get_state_dim(self) -> int:
        """Return the integer dimensionality of the latent state.

        Returns
        -------
        int
            Strictly positive.  Must match encoder.get_embed_dim().
        """


# ---------------------------------------------------------------------------
# 3. BaseMemory
# ---------------------------------------------------------------------------


class BaseMemory(ABC):
    """Abstract base class for episodic / working memory.

    Memory is stateful: it accumulates writes across steps and must be reset
    between episodes.  Concurrent access from multiple threads requires
    external locking.
    """

    @abstractmethod
    def read(self, query: Tensor) -> Tensor:
        """Retrieve memory contents for a query vector.

        Parameters
        ----------
        query : Tensor
            Shape (B, query_dim).

        Returns
        -------
        Tensor
            Shape (B, value_dim).  Returns zeros if memory is empty.
        """

    @abstractmethod
    def write(self, key: Tensor, value: Tensor) -> None:
        """Store a key-value pair in memory.

        Parameters
        ----------
        key : Tensor
            Shape (B, key_dim).
        value : Tensor
            Shape (B, value_dim).

        Raises
        ------
        ValueError
            If key and value have inconsistent batch sizes.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored memory, restoring to the initial empty state.

        Must not reinitialize learnable parameters — only runtime slot tensors
        and addressing state.
        """

    @abstractmethod
    def get_capacity(self) -> int:
        """Return maximum number of key-value pairs this memory can hold.

        Returns
        -------
        int
            Strictly positive, or -1 for unlimited capacity.
        """


# ---------------------------------------------------------------------------
# 4. BasePlanner
# ---------------------------------------------------------------------------


class BasePlanner(ABC):
    """Abstract base class for model-based planners.

    A planner uses a dynamics model to simulate future states in latent space
    and returns an optimized action sequence.  It does not interact with the
    real environment.
    """

    @abstractmethod
    def plan(
        self, state: Tensor, dynamics: BaseDynamics, horizon: int
    ) -> Tensor:
        """Compute an action sequence optimized over the planning horizon.

        Parameters
        ----------
        state : Tensor
            Shape (B, state_dim).  Current latent state.
        dynamics : BaseDynamics
            The dynamics model to simulate with.
        horizon : int
            Number of steps to plan.  Must be >= 1.

        Returns
        -------
        Tensor
            Shape (B, horizon, action_dim).  Planned action sequence.
            Index [:, 0, :] is the action to execute immediately.

        Raises
        ------
        ValueError
            If horizon < 1.
        TypeError
            If dynamics is not a BaseDynamics instance.
        """


# ---------------------------------------------------------------------------
# 5. BaseDecoder
# ---------------------------------------------------------------------------


class BaseDecoder(ABC):
    """Abstract base class for latent-to-observation decoders.

    The decoder is the inverse of the encoder.  It is stateless.

    Dimensional contract
    --------------------
    get_input_dim() must equal encoder.get_embed_dim() for any paired encoder.
    BaseWorldModel validates this at construction time.
    """

    @abstractmethod
    def forward(self, latent: Tensor) -> Tensor:
        """Decode a batch of latent vectors into reconstructed observations.

        Parameters
        ----------
        latent : Tensor
            Shape (B, latent_dim).  latent_dim must equal get_input_dim().

        Returns
        -------
        Tensor
            Shape (B, *get_output_shape()).

        Raises
        ------
        ValueError
            If latent.shape[-1] != get_input_dim().
        """

    @abstractmethod
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return the shape of one decoded observation (no batch dimension).

        Returns
        -------
        Tuple[int, ...]
            e.g. (3, 64, 64) for an RGB image.
        """

    @abstractmethod
    def get_input_dim(self) -> int:
        """Return the expected latent dimensionality.

        Returns
        -------
        int
            Must equal encoder.get_embed_dim() for the paired encoder.
        """


# ---------------------------------------------------------------------------
# 6. BaseTrainer
# ---------------------------------------------------------------------------


class BaseTrainer(ABC):
    """Abstract base class for training orchestrators.

    Encapsulates one gradient update step, a validation loop, and checkpoint
    management.  Concrete trainers implement loss functions, optimizer schedules,
    and mixed-precision policies.

    Note: Use module.train(False) to set inference mode; never call the bare
    module-level inference toggle method directly.
    """

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Run one forward-backward-update cycle.

        Parameters
        ----------
        batch : Any
            A single batch from a DataLoader (dict or tuple of tensors).

        Returns
        -------
        Dict[str, float]
            Scalar metrics.  Must contain key "loss".
        """

    @abstractmethod
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Compute validation metrics without updating parameters.

        Implement this method using torch.no_grad() and module.train(False).

        Parameters
        ----------
        dataloader : DataLoader
            Validation dataloader.

        Returns
        -------
        Dict[str, float]
            Aggregated metrics over the full validation set.
        """

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Persist model and optimizer state to disk.

        Parameters
        ----------
        path : str
            Output file path.  Parent directory must exist.

        Raises
        ------
        OSError
            If the parent directory does not exist.
        """

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state from a checkpoint.

        Parameters
        ----------
        path : str
            Path to checkpoint written by save_checkpoint().

        Raises
        ------
        FileNotFoundError
            If path does not exist.
        """


# ---------------------------------------------------------------------------
# 7. BaseEvaluator
# ---------------------------------------------------------------------------


class BaseEvaluator(ABC):
    """Abstract base class for evaluation harnesses.

    Separates evaluation logic from training.  Given a trained model and a
    dataloader, produces a dict of scalar metrics.

    Note: Set inference mode via module.train(False) inside implementations.
    """

    @abstractmethod
    def evaluate(self, model: Any, dataloader: DataLoader) -> Dict[str, float]:
        """Run full evaluation over dataloader.

        Use torch.no_grad() and module.train(False) inside this method.

        Parameters
        ----------
        model : Any
            Trained model (typically a BaseWorldModel or nn.Module).
        dataloader : DataLoader
            Test or validation dataloader.

        Returns
        -------
        Dict[str, float]
            Evaluation metrics, e.g. {"fvd": 42.3, "psnr": 28.1}.
        """

    @abstractmethod
    def compute_metrics(
        self, predictions: Tensor, targets: Tensor
    ) -> Dict[str, float]:
        """Compute metric values from predictions and ground truth.

        Pure function — no side effects or parameter updates.

        Parameters
        ----------
        predictions : Tensor
            Shape (B, *).
        targets : Tensor
            Shape (B, *), must match predictions.shape.

        Returns
        -------
        Dict[str, float]
            Per-metric scalar values.

        Raises
        ------
        ValueError
            If predictions.shape != targets.shape.
        """


# ---------------------------------------------------------------------------
# 8. BaseExporter
# ---------------------------------------------------------------------------


class BaseExporter(ABC):
    """Abstract base class for model deployment exporters.

    Serializes trained models to deployment formats (TorchScript, ONNX, etc.)
    and validates the exported artifacts.
    """

    @abstractmethod
    def export(self, model: Any, path: str, format: str) -> None:
        """Export a trained model to a deployment artifact.

        Parameters
        ----------
        model : Any
            Trained model to export.
        path : str
            Output file path.  Parent directory must exist.
        format : str
            Export format: "torchscript", "onnx", "tensorrt", etc.

        Raises
        ------
        ValueError
            If format is not supported by this exporter.
        OSError
            If the parent directory of path does not exist.
        """

    @abstractmethod
    def validate_export(self, path: str) -> bool:
        """Verify that the exported artifact is loadable and correct.

        Never raises — validation failures are returned as False.

        Parameters
        ----------
        path : str
            Path to artifact created by export().

        Returns
        -------
        bool
            True if artifact loads and passes a sanity check, False otherwise.
        """


# ---------------------------------------------------------------------------
# Self-test block
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    errors: list[str] = []

    # -----------------------------------------------------------------------
    # Helper: verify that instantiating cls raises TypeError
    # -----------------------------------------------------------------------
    def assert_raises_typeerror(cls, label: str) -> None:
        try:
            cls()  # type: ignore[call-arg]
            errors.append(f"FAIL {label}: expected TypeError, got no error")
        except TypeError:
            print(f"  PASS {label}: raises TypeError as expected")
        except Exception as exc:
            errors.append(
                f"FAIL {label}: expected TypeError, got {type(exc).__name__}: {exc}"
            )

    # -----------------------------------------------------------------------
    # Helper: verify that a complete subclass instantiates
    # -----------------------------------------------------------------------
    def assert_instantiates(cls, label: str) -> None:
        try:
            obj = cls()
            print(
                f"  PASS {label}: instantiates successfully ({type(obj).__name__})"
            )
        except Exception as exc:
            errors.append(f"FAIL {label}: {type(exc).__name__}: {exc}")

    # -----------------------------------------------------------------------
    # BaseEncoder tests
    # -----------------------------------------------------------------------
    print("=== BaseEncoder contract tests ===")

    assert_raises_typeerror(BaseEncoder, "BaseEncoder() raises TypeError")

    class _EncoderMissingEmbed(BaseEncoder):
        def forward(self, x: Tensor) -> Tensor:
            return x

        def get_output_shape(self) -> Tuple[int, ...]:
            return (64,)
        # get_embed_dim deliberately omitted

    assert_raises_typeerror(
        _EncoderMissingEmbed,
        "Encoder missing get_embed_dim raises TypeError",
    )

    class _FullEncoder(BaseEncoder):
        def forward(self, x: Tensor) -> Tensor:
            return torch.zeros(x.shape[0], 64)

        def get_embed_dim(self) -> int:
            return 64

        def get_output_shape(self) -> Tuple[int, ...]:
            return (64,)

    assert_instantiates(_FullEncoder, "Complete BaseEncoder subclass")
    _enc = _FullEncoder()
    assert isinstance(_enc.get_embed_dim(), int), "get_embed_dim() must return int"
    assert isinstance(_enc.get_output_shape(), tuple), "get_output_shape() must return tuple"
    print("  PASS: BaseEncoder return types correct")

    # -----------------------------------------------------------------------
    # BaseDynamics tests
    # -----------------------------------------------------------------------
    print("\n=== BaseDynamics contract tests ===")
    assert_raises_typeerror(BaseDynamics, "BaseDynamics() raises TypeError")

    class _DynMissingImagine(BaseDynamics):
        def step(self, state: Tensor, action: Tensor) -> Tensor:
            return state

        def get_state_dim(self) -> int:
            return 64
        # imagine deliberately omitted

    assert_raises_typeerror(
        _DynMissingImagine, "Dynamics missing imagine raises TypeError"
    )

    class _DynMissingStep(BaseDynamics):
        def imagine(self, state: Tensor, policy: Callable, horizon: int) -> Tensor:
            return torch.zeros(state.shape[0], horizon, 64)

        def get_state_dim(self) -> int:
            return 64
        # step deliberately omitted

    assert_raises_typeerror(
        _DynMissingStep, "Dynamics missing step raises TypeError"
    )

    class _FullDynamics(BaseDynamics):
        def step(self, state: Tensor, action: Tensor) -> Tensor:
            return torch.zeros_like(state)

        def imagine(self, state: Tensor, policy: Callable, horizon: int) -> Tensor:
            B = state.shape[0]
            return torch.zeros(B, horizon, self.get_state_dim())

        def get_state_dim(self) -> int:
            return 64

    assert_instantiates(_FullDynamics, "Complete BaseDynamics subclass")

    # -----------------------------------------------------------------------
    # BaseMemory tests
    # -----------------------------------------------------------------------
    print("\n=== BaseMemory contract tests ===")
    assert_raises_typeerror(BaseMemory, "BaseMemory() raises TypeError")

    class _MemMissingReset(BaseMemory):
        def read(self, query: Tensor) -> Tensor:
            return torch.zeros(query.shape[0], 64)

        def write(self, key: Tensor, value: Tensor) -> None:
            pass

        def get_capacity(self) -> int:
            return 100
        # reset deliberately omitted

    assert_raises_typeerror(
        _MemMissingReset, "Memory missing reset raises TypeError"
    )

    class _FullMemory(BaseMemory):
        def read(self, query: Tensor) -> Tensor:
            return torch.zeros(query.shape[0], 64)

        def write(self, key: Tensor, value: Tensor) -> None:
            pass

        def reset(self) -> None:
            pass

        def get_capacity(self) -> int:
            return 100

    assert_instantiates(_FullMemory, "Complete BaseMemory subclass")
    _mem = _FullMemory()
    assert _mem.get_capacity() == 100, "get_capacity() must return 100"
    print("  PASS: BaseMemory return types correct")

    # -----------------------------------------------------------------------
    # BasePlanner tests
    # -----------------------------------------------------------------------
    print("\n=== BasePlanner contract tests ===")
    assert_raises_typeerror(BasePlanner, "BasePlanner() raises TypeError")

    class _FullPlanner(BasePlanner):
        def plan(
            self, state: Tensor, dynamics: BaseDynamics, horizon: int
        ) -> Tensor:
            return torch.zeros(state.shape[0], horizon, 4)

    assert_instantiates(_FullPlanner, "Complete BasePlanner subclass")

    # -----------------------------------------------------------------------
    # BaseDecoder tests
    # -----------------------------------------------------------------------
    print("\n=== BaseDecoder contract tests ===")
    assert_raises_typeerror(BaseDecoder, "BaseDecoder() raises TypeError")

    class _DecMissingInputDim(BaseDecoder):
        def forward(self, latent: Tensor) -> Tensor:
            return torch.zeros(latent.shape[0], 3, 16, 16)

        def get_output_shape(self) -> Tuple[int, ...]:
            return (3, 16, 16)
        # get_input_dim deliberately omitted

    assert_raises_typeerror(
        _DecMissingInputDim, "Decoder missing get_input_dim raises TypeError"
    )

    class _FullDecoder(BaseDecoder):
        def forward(self, latent: Tensor) -> Tensor:
            return torch.zeros(latent.shape[0], 3, 16, 16)

        def get_output_shape(self) -> Tuple[int, ...]:
            return (3, 16, 16)

        def get_input_dim(self) -> int:
            return 64

    assert_instantiates(_FullDecoder, "Complete BaseDecoder subclass")

    # -----------------------------------------------------------------------
    # BaseTrainer tests
    # -----------------------------------------------------------------------
    print("\n=== BaseTrainer contract tests ===")
    assert_raises_typeerror(BaseTrainer, "BaseTrainer() raises TypeError")

    class _TrainerMissingValidate(BaseTrainer):
        def train_step(self, batch: Any) -> Dict[str, float]:
            return {"loss": 0.0}

        def save_checkpoint(self, path: str) -> None:
            pass

        def load_checkpoint(self, path: str) -> None:
            pass
        # validate deliberately omitted

    assert_raises_typeerror(
        _TrainerMissingValidate, "Trainer missing validate raises TypeError"
    )

    class _FullTrainer(BaseTrainer):
        def train_step(self, batch: Any) -> Dict[str, float]:
            return {"loss": 0.0}

        def validate(self, dataloader: DataLoader) -> Dict[str, float]:
            return {"val_loss": 0.0}

        def save_checkpoint(self, path: str) -> None:
            pass

        def load_checkpoint(self, path: str) -> None:
            pass

    assert_instantiates(_FullTrainer, "Complete BaseTrainer subclass")

    # -----------------------------------------------------------------------
    # BaseEvaluator tests
    # -----------------------------------------------------------------------
    print("\n=== BaseEvaluator contract tests ===")
    assert_raises_typeerror(BaseEvaluator, "BaseEvaluator() raises TypeError")

    class _EvalMissingComputeMetrics(BaseEvaluator):
        def evaluate(
            self, model: Any, dataloader: DataLoader
        ) -> Dict[str, float]:
            return {"psnr": 0.0}
        # compute_metrics deliberately omitted

    assert_raises_typeerror(
        _EvalMissingComputeMetrics,
        "Evaluator missing compute_metrics raises TypeError",
    )

    class _FullEvaluator(BaseEvaluator):
        def evaluate(
            self, model: Any, dataloader: DataLoader
        ) -> Dict[str, float]:
            return {"psnr": 0.0}

        def compute_metrics(
            self, predictions: Tensor, targets: Tensor
        ) -> Dict[str, float]:
            if predictions.shape != targets.shape:
                raise ValueError("Shape mismatch")
            mse = float(((predictions - targets) ** 2).mean())
            return {"mse": mse}

    assert_instantiates(_FullEvaluator, "Complete BaseEvaluator subclass")
    _ev = _FullEvaluator()
    p = torch.ones(4, 8)
    t = torch.zeros(4, 8)
    metrics = _ev.compute_metrics(p, t)
    assert "mse" in metrics, "compute_metrics must return dict with 'mse'"
    print("  PASS: BaseEvaluator compute_metrics works correctly")

    # Test ValueError on shape mismatch
    try:
        _ev.compute_metrics(torch.ones(4, 8), torch.ones(4, 9))
        errors.append("FAIL: compute_metrics shape mismatch did not raise ValueError")
    except ValueError:
        print("  PASS: compute_metrics raises ValueError on shape mismatch")

    # -----------------------------------------------------------------------
    # BaseExporter tests
    # -----------------------------------------------------------------------
    print("\n=== BaseExporter contract tests ===")
    assert_raises_typeerror(BaseExporter, "BaseExporter() raises TypeError")

    class _ExporterMissingValidate(BaseExporter):
        def export(self, model: Any, path: str, format: str) -> None:
            pass
        # validate_export deliberately omitted

    assert_raises_typeerror(
        _ExporterMissingValidate,
        "Exporter missing validate_export raises TypeError",
    )

    class _FullExporter(BaseExporter):
        def export(self, model: Any, path: str, format: str) -> None:
            pass

        def validate_export(self, path: str) -> bool:
            return False

    assert_instantiates(_FullExporter, "Complete BaseExporter subclass")
    _exp = _FullExporter()
    result = _exp.validate_export("/nonexistent/path.pt")
    assert result is False, "validate_export must return bool"
    print("  PASS: BaseExporter validate_export returns bool")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    if errors:
        for e in errors:
            print(f"  {e}")
        print(f"\n{len(errors)} test(s) FAILED.")
        sys.exit(1)
    else:
        print("All base class contract tests PASSED.")
        sys.exit(0)
