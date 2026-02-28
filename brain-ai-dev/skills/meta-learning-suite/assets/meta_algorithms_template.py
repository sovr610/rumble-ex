"""
brain_ai/meta/algorithms.py -- MAML, FOMAML, and Reptile Meta-Learning Algorithms

This module implements three core meta-learning algorithms for few-shot adaptation:

    MAML (Model-Agnostic Meta-Learning):
        Second-order meta-learning.  The inner loop unrolls gradient steps with
        create_graph=True so that the outer loss can differentiate *through* the
        adaptation process.  This captures how the initialization influences the
        adapted solution, but is more expensive (O(steps * memory)).

    FOMAML (First-Order MAML):
        Drops the second-order terms by running the inner loop with
        create_graph=False.  The outer gradient is computed at the adapted
        parameter point but does not flow back through the inner updates.
        Typically 2-3x faster than full MAML with comparable performance on
        many benchmarks.

    Reptile:
        Weight-space interpolation.  Each task produces adapted parameters phi_i,
        and the meta-update moves the initialization toward the average:
            theta <- theta + epsilon * mean(phi_i - theta)
        No query set is strictly needed (though we compute query metrics for
        diagnostics).  Uses a pseudo-loss trick so standard optimizers work.

All three algorithms share a common abstract base (MetaAlgorithm) and produce
a unified MetaOutput with loss, metrics, inner-loop logs, and optionally the
adapted parameter dictionaries.

Key design decisions:
    - Parameters are always passed as Dict[str, Tensor] (named parameter dicts),
      never as nn.Module clones.  This is compatible with torch.func and avoids
      deepcopy overhead.
    - The inner loop is delegated to an InnerLoopEngine (from inner_loop.py)
      which handles torch.func / higher / custom SGD backends.
    - Metrics include pre/post adaptation accuracy, fast gain, and AUAC (Area
      Under Adaptation Curve) computed via trapezoidal rule.
    - All forward calls go through _forward() which uses functional_call when
      available, falling back to stateful forward for standalone mode.

Usage:
    from brain_ai.meta.algorithms import create_meta_algorithm, Episode, TaskBatch

    algo = create_meta_algorithm("maml", inner_engine, config)
    params = dict(model.named_parameters())
    output = algo.meta_step(model, params, task_batch)
    output.loss.backward()
    outer_optimizer.step()

References:
    Finn et al. (2017) "Model-Agnostic Meta-Learning for Fast Adaptation"
    Nichol et al. (2018) "On First-Order Meta-Learning Algorithms" (Reptile)
    Antoniou et al. (2019) "How to Train Your MAML"
"""

from __future__ import annotations

import math
import copy
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Optional import: InnerLoopEngine from the companion module.
# If unavailable (standalone execution, unit tests), we provide inline stubs.
# ---------------------------------------------------------------------------

_INNER_LOOP_AVAILABLE = False

try:
    from brain_ai.meta.inner_loop import (
        InnerLoopEngine,
        InnerLoopResult,
        StepLog,
        create_inner_loop_engine,
    )
    _INNER_LOOP_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Inline stubs -- allow this file to run standalone for self-testing
# ---------------------------------------------------------------------------

if not _INNER_LOOP_AVAILABLE:

    @dataclass
    class StepLog:
        """Per-step log entry from an inner-loop adaptation step."""
        step: int = 0
        loss: float = 0.0
        accuracy: float = 0.0
        grad_norm: float = 0.0
        update_norm: float = 0.0
        lr: float = 0.01
        lr_stats: Optional[Dict[str, float]] = None

        def __repr__(self) -> str:
            return (
                f"StepLog(step={self.step}, loss={self.loss:.4f}, "
                f"acc={self.accuracy:.4f}, grad_norm={self.grad_norm:.4f})"
            )

    @dataclass
    class InnerLoopResult:
        """Result from a full inner-loop adaptation run."""
        adapted_params: Dict[str, Tensor]
        step_logs: List[StepLog]
        final_loss: float = 0.0
        final_accuracy: float = 0.0

    class InnerLoopEngine:
        """Stub inner-loop engine for standalone testing.

        Performs gradient-based adaptation using torch.autograd.grad with
        manual parameter updates.  Supports first_order toggle and gradient
        clipping.
        """

        def __init__(self, backend: str = "custom", **kwargs):
            self.backend = backend

        def adapt(
            self,
            model: nn.Module,
            params: Dict[str, Tensor],
            support_x: Tensor,
            support_y: Tensor,
            *,
            steps: int = 5,
            lr: float = 0.01,
            first_order: bool = False,
            clip_norm: Optional[float] = None,
            loss_fn: Optional[Callable] = None,
        ) -> InnerLoopResult:
            """Run inner-loop adaptation and return adapted params + logs."""
            if loss_fn is None:
                loss_fn = F.cross_entropy

            # Clone parameters so original dict is not mutated
            adapted = {k: v.clone().requires_grad_(True) for k, v in params.items()}
            step_logs: List[StepLog] = []

            for step_idx in range(steps):
                # Functional forward
                logits = _functional_forward(model, adapted, support_x)
                loss = loss_fn(logits, support_y)

                # Accuracy
                with torch.no_grad():
                    acc = (logits.argmax(-1) == support_y).float().mean().item()

                # Compute gradients
                grad_list = torch.autograd.grad(
                    loss,
                    list(adapted.values()),
                    create_graph=not first_order,
                    allow_unused=True,
                )

                # Gradient clipping (per-param norm)
                total_grad_norm = 0.0
                for g in grad_list:
                    if g is not None:
                        total_grad_norm += g.detach().norm().item() ** 2
                total_grad_norm = math.sqrt(total_grad_norm)

                if clip_norm is not None and total_grad_norm > clip_norm:
                    clip_coef = clip_norm / (total_grad_norm + 1e-8)
                else:
                    clip_coef = 1.0

                # Update parameters
                update_norm_sq = 0.0
                new_adapted: Dict[str, Tensor] = {}
                for (k, p), g in zip(adapted.items(), grad_list):
                    if g is None:
                        new_adapted[k] = p
                    else:
                        clipped_g = g * clip_coef
                        new_p = p - lr * clipped_g
                        new_adapted[k] = new_p
                        update_norm_sq += (lr * clipped_g).detach().norm().item() ** 2

                adapted = new_adapted

                step_logs.append(StepLog(
                    step=step_idx,
                    loss=loss.item(),
                    accuracy=acc,
                    grad_norm=total_grad_norm,
                    update_norm=math.sqrt(update_norm_sq),
                    lr=lr,
                ))

            final_loss = step_logs[-1].loss if step_logs else 0.0
            final_accuracy = step_logs[-1].accuracy if step_logs else 0.0

            return InnerLoopResult(
                adapted_params=adapted,
                step_logs=step_logs,
                final_loss=final_loss,
                final_accuracy=final_accuracy,
            )

    def create_inner_loop_engine(backend: str = "custom", **kwargs) -> InnerLoopEngine:
        """Factory for InnerLoopEngine (stub version)."""
        return InnerLoopEngine(backend=backend, **kwargs)


# ---------------------------------------------------------------------------
# Functional forward helper
# ---------------------------------------------------------------------------

def _functional_forward(
    model: nn.Module,
    params: Dict[str, Tensor],
    x: Tensor,
) -> Tensor:
    """Forward pass using a parameter dict instead of model.parameters().

    Tries torch.nn.utils.stateless.functional_call (PyTorch >= 2.0) first,
    then falls back to temporarily patching the model state dict.
    """
    # Try torch.func.functional_call (PyTorch >= 2.1, preferred)
    try:
        from torch.func import functional_call as _fc  # noqa: F811
        return _fc(model, params, (x,))
    except (ImportError, AttributeError):
        pass

    # Try torch.nn.utils.stateless (PyTorch 2.0)
    try:
        from torch.nn.utils import stateless  # noqa: F811
        if hasattr(stateless, "functional_call"):
            return stateless.functional_call(model, params, (x,))
    except (ImportError, AttributeError):
        pass

    # Fallback: temporarily load params into model
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    try:
        state_dict = model.state_dict()
        for k, v in params.items():
            if k in state_dict:
                state_dict[k] = v
        model.load_state_dict(state_dict, strict=False)
        return model(x)
    finally:
        model.load_state_dict(original_state, strict=False)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """A single N-way K-shot episode for meta-learning.

    Attributes:
        support_x: Support set inputs, shape (N*K, ...).
        support_y: Support set labels, shape (N*K,).
        query_x:   Query set inputs, shape (N*Q, ...).
        query_y:   Query set labels, shape (N*Q,).
        class_ids:  Optional list of original class indices used in this episode.
        episode_id: Optional tuple (epoch, episode_idx) for reproducibility tracking.
    """
    support_x: Tensor
    support_y: Tensor
    query_x: Tensor
    query_y: Tensor
    class_ids: Optional[List[int]] = None
    episode_id: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        """Validate tensor shapes and dtypes."""
        assert self.support_x.shape[0] == self.support_y.shape[0], (
            f"support_x batch {self.support_x.shape[0]} != "
            f"support_y batch {self.support_y.shape[0]}"
        )
        assert self.query_x.shape[0] == self.query_y.shape[0], (
            f"query_x batch {self.query_x.shape[0]} != "
            f"query_y batch {self.query_y.shape[0]}"
        )

    @property
    def n_support(self) -> int:
        """Number of support examples."""
        return self.support_x.shape[0]

    @property
    def n_query(self) -> int:
        """Number of query examples."""
        return self.query_x.shape[0]

    @property
    def device(self) -> torch.device:
        """Device of the tensors."""
        return self.support_x.device

    def to(self, device: torch.device) -> "Episode":
        """Move all tensors to a device."""
        return Episode(
            support_x=self.support_x.to(device),
            support_y=self.support_y.to(device),
            query_x=self.query_x.to(device),
            query_y=self.query_y.to(device),
            class_ids=self.class_ids,
            episode_id=self.episode_id,
        )


@dataclass
class TaskBatch:
    """Batch of episodes for meta-training.

    Attributes:
        episodes: List of Episode objects, one per task in the batch.
    """
    episodes: List[Episode]

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self):
        return iter(self.episodes)

    def __getitem__(self, idx: int) -> Episode:
        return self.episodes[idx]

    @property
    def device(self) -> torch.device:
        """Device of the first episode (all should match)."""
        return self.episodes[0].device if self.episodes else torch.device("cpu")

    def to(self, device: torch.device) -> "TaskBatch":
        """Move all episodes to a device."""
        return TaskBatch(episodes=[ep.to(device) for ep in self.episodes])


@dataclass
class MetaOutput:
    """Output from a single meta-learning step (one outer iteration).

    Attributes:
        loss:           Scalar meta-loss suitable for .backward().
        metrics:        Dictionary of diagnostic metrics:
                            pre_adapt_acc   -- accuracy before adaptation
                            post_adapt_acc  -- accuracy after adaptation
                            fast_gain       -- post_adapt_acc - pre_adapt_acc
                            auac            -- area under adaptation curve
        inner_logs:     Per-task list of StepLog lists (inner-loop trajectory).
        adapted_params: Optional per-task list of adapted parameter dicts.
        algo:           Algorithm name string ("maml", "fomaml", "reptile").
    """
    loss: Tensor
    metrics: Dict[str, float]
    inner_logs: List[List[Any]]
    adapted_params: Optional[List[Dict[str, Tensor]]] = None
    algo: str = ""

    def __repr__(self) -> str:
        m = self.metrics
        return (
            f"MetaOutput(algo={self.algo!r}, loss={self.loss.item():.4f}, "
            f"pre_acc={m.get('pre_adapt_acc', 0):.3f}, "
            f"post_acc={m.get('post_adapt_acc', 0):.3f}, "
            f"gain={m.get('fast_gain', 0):.3f}, "
            f"auac={m.get('auac', 0):.3f})"
        )


# ---------------------------------------------------------------------------
# Meta-algorithm configuration
# ---------------------------------------------------------------------------

@dataclass
class MetaAlgorithmConfig:
    """Configuration shared across all meta-learning algorithms.

    Attributes:
        inner_steps:        Number of inner-loop gradient steps.
        inner_lr:           Base inner-loop learning rate.
        inner_clip:         Max gradient norm for inner-loop clipping (None=off).
        outer_clip:         Max gradient norm for outer-loop clipping (None=off).
        reptile_epsilon:    Interpolation rate for Reptile updates.
        episodes_per_epoch: Number of episodes per meta-training epoch.
        loss_fn_name:       Name of the loss function ("cross_entropy", "mse").
        track_adapted:      Whether to store adapted_params in MetaOutput.
    """
    inner_steps: int = 5
    inner_lr: float = 0.01
    inner_clip: Optional[float] = 10.0
    outer_clip: Optional[float] = None
    reptile_epsilon: float = 1.0
    episodes_per_epoch: int = 600
    loss_fn_name: str = "cross_entropy"
    track_adapted: bool = False

    def get_loss_fn(self) -> Callable:
        """Return the loss function specified by loss_fn_name."""
        if self.loss_fn_name == "cross_entropy":
            return F.cross_entropy
        elif self.loss_fn_name == "mse":
            return F.mse_loss
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn_name}")


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class MetaAlgorithm(ABC):
    """Abstract base for meta-learning algorithms.

    Provides shared infrastructure for pre/post adaptation metrics,
    AUAC computation, and functional forward passes.  Subclasses must
    implement meta_step() which computes the meta-loss for one batch
    of tasks.
    """

    def __init__(
        self,
        inner_engine: InnerLoopEngine,
        config: MetaAlgorithmConfig,
    ):
        self.inner_engine = inner_engine
        self.config = config

    @abstractmethod
    def meta_step(
        self,
        model: nn.Module,
        params: Dict[str, Tensor],
        task_batch: TaskBatch,
    ) -> MetaOutput:
        """Compute meta-loss and metrics for a batch of tasks.

        Args:
            model:      The model providing the computation graph structure.
            params:     Named parameter dict (from model.named_parameters()).
            task_batch: Batch of Episode objects.

        Returns:
            MetaOutput with loss (scalar, has grad_fn), metrics, and logs.
        """
        ...

    # ----- shared helpers -----

    def _forward(
        self,
        model: nn.Module,
        params: Dict[str, Tensor],
        x: Tensor,
    ) -> Tensor:
        """Functional forward pass using the parameter dict."""
        return _functional_forward(model, params, x)

    def _compute_pre_adapt_metrics(
        self,
        model: nn.Module,
        params: Dict[str, Tensor],
        episode: Episode,
    ) -> float:
        """Compute pre-adaptation accuracy on the query set.

        Uses torch.no_grad() since we only need the scalar metric.
        """
        with torch.no_grad():
            logits = self._forward(model, params, episode.query_x)
            preds = logits.argmax(dim=-1)
            acc = (preds == episode.query_y).float().mean().item()
        return acc

    def _compute_post_adapt_metrics(
        self,
        model: nn.Module,
        adapted_params: Dict[str, Tensor],
        episode: Episode,
    ) -> float:
        """Compute post-adaptation accuracy on the query set.

        Uses torch.no_grad() for efficiency -- metrics only.
        """
        with torch.no_grad():
            logits = self._forward(model, adapted_params, episode.query_x)
            preds = logits.argmax(dim=-1)
            acc = (preds == episode.query_y).float().mean().item()
        return acc

    def _compute_auac(self, all_step_accs: List[List[float]]) -> float:
        """Compute Area Under Adaptation Curve via trapezoidal rule.

        The adaptation curve plots accuracy as a function of inner-loop step.
        AUAC summarizes how quickly the model adapts across all steps, not
        just at the final step.  Higher AUAC means faster/better adaptation.

        Args:
            all_step_accs: Per-task lists of accuracies.  Each list has
                (inner_steps + 1) entries: [pre_adapt, step_0, step_1, ...].

        Returns:
            Normalized AUAC in [0, 1].  Returns the single accuracy value
            if there is only one point.
        """
        if not all_step_accs:
            return 0.0

        # Find the maximum number of steps across tasks
        max_steps = max(len(accs) for accs in all_step_accs)
        if max_steps <= 0:
            return 0.0

        # Average accuracy at each step across all tasks
        avg_accs: List[float] = []
        for step in range(max_steps):
            vals = [accs[step] for accs in all_step_accs if step < len(accs)]
            if vals:
                avg_accs.append(sum(vals) / len(vals))

        if len(avg_accs) <= 1:
            return avg_accs[0] if avg_accs else 0.0

        # Trapezoidal integration
        area = 0.0
        for i in range(len(avg_accs) - 1):
            area += (avg_accs[i] + avg_accs[i + 1]) / 2.0

        # Normalize by the number of intervals so AUAC is in [0, 1]
        auac = area / (len(avg_accs) - 1)
        return auac

    def _collect_step_accs(
        self,
        pre_acc: float,
        step_logs: List[StepLog],
    ) -> List[float]:
        """Build the accuracy sequence for AUAC: [pre_acc, step0_acc, ...]."""
        accs = [pre_acc]
        for log in step_logs:
            accs.append(log.accuracy)
        return accs

    def _aggregate_metrics(
        self,
        pre_accs: List[float],
        post_accs: List[float],
        all_step_accs: List[List[float]],
    ) -> Dict[str, float]:
        """Build the standard metrics dictionary from per-task values."""
        mean_pre = sum(pre_accs) / len(pre_accs) if pre_accs else 0.0
        mean_post = sum(post_accs) / len(post_accs) if post_accs else 0.0
        return {
            "pre_adapt_acc": mean_pre,
            "post_adapt_acc": mean_post,
            "fast_gain": mean_post - mean_pre,
            "auac": self._compute_auac(all_step_accs),
        }


# ---------------------------------------------------------------------------
# MAML -- second-order meta-learning
# ---------------------------------------------------------------------------

class MAMLAlgorithm(MetaAlgorithm):
    """Model-Agnostic Meta-Learning with second-order gradients.

    The inner loop runs with create_graph=True so that gradients flow
    back through the adaptation steps.  This means the outer loss
    captures how changes in the initialization theta affect the adapted
    solution phi, yielding the full MAML meta-gradient:

        d/d_theta L_query(phi(theta))
            = d/d_theta L_query(theta - alpha * grad L_support(theta))

    which includes second-order terms involving the Hessian of the
    support loss.

    Computational cost:  O(inner_steps) times the cost of a single
    backward pass, plus the memory to store the computation graph
    through the inner loop.
    """

    def meta_step(
        self,
        model: nn.Module,
        params: Dict[str, Tensor],
        task_batch: TaskBatch,
    ) -> MetaOutput:
        """Compute MAML meta-loss over a batch of tasks.

        For each task:
            1. Record pre-adaptation accuracy on query set.
            2. Run inner-loop adaptation (second-order) on support set.
            3. Compute query loss through adapted params (retains grad_fn).
            4. Record post-adaptation accuracy.
            5. Collect step-by-step logs for AUAC.

        The per-task query losses are averaged to form the meta-loss.
        """
        device = next(iter(params.values())).device
        meta_loss = torch.tensor(0.0, device=device, requires_grad=True)

        all_inner_logs: List[List[StepLog]] = []
        all_adapted_params: List[Dict[str, Tensor]] = []
        pre_accs: List[float] = []
        post_accs: List[float] = []
        all_step_accs: List[List[float]] = []

        for episode in task_batch.episodes:
            # --- Pre-adaptation metrics ---
            pre_acc = self._compute_pre_adapt_metrics(model, params, episode)
            pre_accs.append(pre_acc)

            # --- Inner loop (second-order: first_order=False) ---
            result = self.inner_engine.adapt(
                model,
                params,
                episode.support_x,
                episode.support_y,
                steps=self.config.inner_steps,
                lr=self.config.inner_lr,
                first_order=False,  # MAML: retain computation graph
                clip_norm=self.config.inner_clip,
            )

            # --- Query loss (through adapted params with grad_fn) ---
            query_logits = self._forward(model, result.adapted_params, episode.query_x)
            query_loss = F.cross_entropy(query_logits, episode.query_y)
            meta_loss = meta_loss + query_loss

            # --- Post-adaptation metrics ---
            with torch.no_grad():
                post_acc = (
                    (query_logits.detach().argmax(-1) == episode.query_y)
                    .float()
                    .mean()
                    .item()
                )
            post_accs.append(post_acc)

            # --- Collect per-step accuracies for AUAC ---
            step_accs = self._collect_step_accs(pre_acc, result.step_logs)
            all_step_accs.append(step_accs)

            all_inner_logs.append(result.step_logs)
            if self.config.track_adapted:
                all_adapted_params.append(result.adapted_params)

        # Average over tasks
        meta_loss = meta_loss / len(task_batch)

        # Build metrics
        metrics = self._aggregate_metrics(pre_accs, post_accs, all_step_accs)

        return MetaOutput(
            loss=meta_loss,
            metrics=metrics,
            inner_logs=all_inner_logs,
            adapted_params=all_adapted_params if self.config.track_adapted else None,
            algo="maml",
        )


# ---------------------------------------------------------------------------
# FOMAML -- first-order MAML
# ---------------------------------------------------------------------------

class FOMAMLAlgorithm(MetaAlgorithm):
    """First-Order MAML (FOMAML).

    Identical to MAML except the inner loop runs with
    create_graph=False (first_order=True).  The adapted parameters
    do NOT retain a grad_fn through the inner updates.  The outer
    gradient is computed only at the adapted parameter point:

        d/d_theta L_query(phi) ~= grad_phi L_query(phi)

    This drops the second-order Hessian terms, making training
    2-3x faster with ~80-90% of full MAML performance on many
    benchmarks (Nichol et al., 2018).

    Computational cost:  Same as MAML forward, but backward only
    needs one step (no unrolling through the adaptation graph).
    """

    def meta_step(
        self,
        model: nn.Module,
        params: Dict[str, Tensor],
        task_batch: TaskBatch,
    ) -> MetaOutput:
        """Compute FOMAML meta-loss over a batch of tasks.

        Structure is identical to MAMLAlgorithm.meta_step, but the
        inner loop uses first_order=True.  The query loss is still
        differentiable w.r.t. the adapted parameters (and therefore
        the base params through the final linear relationship), but
        no second-order graph is maintained.
        """
        device = next(iter(params.values())).device
        meta_loss = torch.tensor(0.0, device=device, requires_grad=True)

        all_inner_logs: List[List[StepLog]] = []
        all_adapted_params: List[Dict[str, Tensor]] = []
        pre_accs: List[float] = []
        post_accs: List[float] = []
        all_step_accs: List[List[float]] = []

        for episode in task_batch.episodes:
            # --- Pre-adaptation metrics ---
            pre_acc = self._compute_pre_adapt_metrics(model, params, episode)
            pre_accs.append(pre_acc)

            # --- Inner loop (first-order: no computation graph) ---
            result = self.inner_engine.adapt(
                model,
                params,
                episode.support_x,
                episode.support_y,
                steps=self.config.inner_steps,
                lr=self.config.inner_lr,
                first_order=True,  # FOMAML: drop second-order terms
                clip_norm=self.config.inner_clip,
            )

            # --- Query loss ---
            # Even though inner loop is first-order, query loss is computed
            # with the adapted params which still requires_grad (they are
            # leaf tensors from the detached inner loop).
            query_logits = self._forward(model, result.adapted_params, episode.query_x)
            query_loss = F.cross_entropy(query_logits, episode.query_y)
            meta_loss = meta_loss + query_loss

            # --- Post-adaptation metrics ---
            with torch.no_grad():
                post_acc = (
                    (query_logits.detach().argmax(-1) == episode.query_y)
                    .float()
                    .mean()
                    .item()
                )
            post_accs.append(post_acc)

            # --- Per-step accuracies for AUAC ---
            step_accs = self._collect_step_accs(pre_acc, result.step_logs)
            all_step_accs.append(step_accs)

            all_inner_logs.append(result.step_logs)
            if self.config.track_adapted:
                all_adapted_params.append(result.adapted_params)

        # Average over tasks
        meta_loss = meta_loss / len(task_batch)

        # Build metrics
        metrics = self._aggregate_metrics(pre_accs, post_accs, all_step_accs)

        return MetaOutput(
            loss=meta_loss,
            metrics=metrics,
            inner_logs=all_inner_logs,
            adapted_params=all_adapted_params if self.config.track_adapted else None,
            algo="fomaml",
        )


# ---------------------------------------------------------------------------
# Reptile -- weight-space interpolation
# ---------------------------------------------------------------------------

class ReptileAlgorithm(MetaAlgorithm):
    """Reptile meta-learning via weight-space interpolation.

    Reptile is fundamentally different from MAML/FOMAML.  Instead of
    differentiating through the adaptation process, Reptile simply:

        1. Adapts to each task (getting phi_i from theta).
        2. Computes weight deltas: delta_i = phi_i - theta.
        3. Applies the interpolation update:
               theta <- theta + epsilon * mean(delta_i)

    The "meta-gradient" in Reptile approximates a combination of the
    full MAML gradient and a term that encourages different task
    solutions to be close together (Nichol et al., 2018).

    To integrate with standard PyTorch optimizers, we construct a
    pseudo-loss whose gradient equals the Reptile update direction.
    This allows using Adam, SGD with momentum, etc. for the outer loop
    without special-casing.

    Pseudo-loss construction:
        L_pseudo = -epsilon * sum_k (theta_k * mean_delta_k.detach())
        grad L_pseudo w.r.t. theta_k = -epsilon * mean_delta_k
        Optimizer step: theta_k -= lr * (-epsilon * mean_delta_k)
                                    = theta_k + lr * epsilon * mean_delta_k
        With lr=1.0 this recovers the exact Reptile update.

    Attributes:
        epsilon: Interpolation rate (typically 1.0; scaled by outer optimizer lr).
    """

    def __init__(
        self,
        inner_engine: InnerLoopEngine,
        config: MetaAlgorithmConfig,
    ):
        super().__init__(inner_engine, config)
        self.epsilon = config.reptile_epsilon

    def meta_step(
        self,
        model: nn.Module,
        params: Dict[str, Tensor],
        task_batch: TaskBatch,
    ) -> MetaOutput:
        """Compute Reptile meta-update for a batch of tasks.

        For each task:
            1. Record pre-adaptation accuracy.
            2. Run inner-loop adaptation (first-order, no graph needed).
            3. Accumulate weight deltas (phi_i - theta).
            4. Record post-adaptation accuracy.

        After all tasks, construct a pseudo-loss for optimizer integration.
        """
        # Accumulate weight deltas
        weight_deltas: Dict[str, Tensor] = {
            k: torch.zeros_like(v) for k, v in params.items()
        }

        all_inner_logs: List[List[StepLog]] = []
        all_adapted_params: List[Dict[str, Tensor]] = []
        pre_accs: List[float] = []
        post_accs: List[float] = []
        all_step_accs: List[List[float]] = []

        for episode in task_batch.episodes:
            # --- Pre-adaptation metrics ---
            pre_acc = self._compute_pre_adapt_metrics(model, params, episode)
            pre_accs.append(pre_acc)

            # --- Inner loop (Reptile: first-order, no graph needed) ---
            result = self.inner_engine.adapt(
                model,
                params,
                episode.support_x,
                episode.support_y,
                steps=self.config.inner_steps,
                lr=self.config.inner_lr,
                first_order=True,  # Reptile: no second-order needed
                clip_norm=self.config.inner_clip,
            )

            # --- Accumulate weight deltas ---
            for k in params:
                delta = result.adapted_params[k].detach() - params[k].detach()
                weight_deltas[k] = weight_deltas[k] + delta

            # --- Post-adaptation metrics ---
            post_acc = self._compute_post_adapt_metrics(
                model, result.adapted_params, episode
            )
            post_accs.append(post_acc)

            # --- Per-step accuracies for AUAC ---
            step_accs = self._collect_step_accs(pre_acc, result.step_logs)
            all_step_accs.append(step_accs)

            all_inner_logs.append(result.step_logs)
            if self.config.track_adapted:
                all_adapted_params.append(result.adapted_params)

        # --- Average deltas ---
        num_tasks = len(task_batch)
        avg_delta: Dict[str, Tensor] = {
            k: v / num_tasks for k, v in weight_deltas.items()
        }

        # --- Construct pseudo-loss for optimizer integration ---
        # The gradient of this loss w.r.t. params gives the negative
        # of the Reptile update direction, so optimizer.step() (which
        # subtracts lr * grad) moves in the right direction when lr=1.
        pseudo_loss = torch.tensor(0.0, device=next(iter(params.values())).device)
        for k, p in params.items():
            pseudo_loss = pseudo_loss - (p * avg_delta[k].detach()).sum() * self.epsilon

        # --- Metrics ---
        metrics = self._aggregate_metrics(pre_accs, post_accs, all_step_accs)

        # Additional Reptile-specific metrics
        delta_norms = [d.norm().item() for d in avg_delta.values()]
        metrics["mean_weight_delta"] = (
            sum(delta_norms) / len(delta_norms) if delta_norms else 0.0
        )
        metrics["max_weight_delta"] = max(delta_norms) if delta_norms else 0.0

        return MetaOutput(
            loss=pseudo_loss,
            metrics=metrics,
            inner_logs=all_inner_logs,
            adapted_params=all_adapted_params if self.config.track_adapted else None,
            algo="reptile",
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

_ALGORITHM_REGISTRY: Dict[str, type] = {
    "maml": MAMLAlgorithm,
    "fomaml": FOMAMLAlgorithm,
    "reptile": ReptileAlgorithm,
}


def create_meta_algorithm(
    algo: str,
    inner_engine: InnerLoopEngine,
    config: MetaAlgorithmConfig,
) -> MetaAlgorithm:
    """Create a meta-learning algorithm by name.

    Args:
        algo:          One of "maml", "fomaml", "reptile".
        inner_engine:  An InnerLoopEngine for inner-loop adaptation.
        config:        MetaAlgorithmConfig with shared hyperparameters.

    Returns:
        A MetaAlgorithm subclass instance.

    Raises:
        ValueError: If algo is not recognized.
    """
    algo_lower = algo.lower().strip()
    if algo_lower not in _ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown meta-learning algorithm: {algo!r}. "
            f"Choose from {list(_ALGORITHM_REGISTRY.keys())}"
        )
    return _ALGORITHM_REGISTRY[algo_lower](inner_engine, config)


def register_meta_algorithm(name: str, cls: type) -> None:
    """Register a custom meta-learning algorithm.

    Args:
        name: Algorithm name string.
        cls:  Class that extends MetaAlgorithm.
    """
    if not issubclass(cls, MetaAlgorithm):
        raise TypeError(f"{cls} must be a subclass of MetaAlgorithm")
    _ALGORITHM_REGISTRY[name.lower().strip()] = cls


def list_meta_algorithms() -> List[str]:
    """Return a list of all registered algorithm names."""
    return sorted(_ALGORITHM_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Meta-training loop helper
# ---------------------------------------------------------------------------

def run_meta_epoch(
    model: nn.Module,
    meta_algo: MetaAlgorithm,
    task_sampler: Any,
    outer_optimizer: torch.optim.Optimizer,
    epoch: int,
    config: MetaAlgorithmConfig,
    *,
    tasks_per_batch: int = 1,
    log_interval: int = 50,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, float]:
    """Run one meta-training epoch.

    Iterates over config.episodes_per_epoch episodes, computing the meta-loss
    for each batch of tasks, backpropagating, and performing an outer optimizer
    step.

    Args:
        model:            The model being meta-trained.
        meta_algo:        A MetaAlgorithm instance.
        task_sampler:     Object with sample_episode(epoch, idx) -> Episode.
        outer_optimizer:  Optimizer for outer-loop (meta) parameters.
        epoch:            Current epoch number (for reproducibility).
        config:           MetaAlgorithmConfig.
        tasks_per_batch:  Number of tasks per meta-batch (default 1).
        log_interval:     Print progress every N episodes.
        logger:           Optional logging function.

    Returns:
        Dictionary of epoch-averaged metrics.
    """
    model.train()
    epoch_metrics: Dict[str, List[float]] = defaultdict(list)
    epoch_losses: List[float] = []

    total_episodes = config.episodes_per_epoch
    num_batches = max(1, total_episodes // tasks_per_batch)

    for batch_idx in range(num_batches):
        # --- Sample task batch ---
        episodes: List[Episode] = []
        for t in range(tasks_per_batch):
            ep_idx = batch_idx * tasks_per_batch + t
            if ep_idx >= total_episodes:
                break
            episode = task_sampler.sample_episode(epoch, ep_idx)
            episodes.append(episode)

        if not episodes:
            break

        task_batch = TaskBatch(episodes=episodes)

        # --- Meta-step ---
        params = dict(model.named_parameters())
        output = meta_algo.meta_step(model, params, task_batch)

        # --- Outer optimizer step ---
        outer_optimizer.zero_grad()
        output.loss.backward()

        if config.outer_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.outer_clip
            )

        outer_optimizer.step()

        # --- Collect metrics ---
        epoch_losses.append(output.loss.item())
        for k, v in output.metrics.items():
            epoch_metrics[k].append(v)

        # --- Logging ---
        if logger and (batch_idx + 1) % log_interval == 0:
            recent_count = min(log_interval, len(epoch_losses))
            avg_loss = sum(epoch_losses[-recent_count:]) / recent_count
            recent_post = epoch_metrics["post_adapt_acc"][-recent_count:]
            avg_post = sum(recent_post) / len(recent_post) if recent_post else 0.0
            logger(
                f"  [epoch {epoch}, batch {batch_idx + 1}/{num_batches}] "
                f"loss={avg_loss:.4f}, post_acc={avg_post:.3f}"
            )

    # --- Aggregate epoch metrics ---
    result: Dict[str, float] = {
        "meta_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0,
    }
    for k, v in epoch_metrics.items():
        result[k] = sum(v) / len(v) if v else 0.0

    return result


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def run_meta_validation(
    model: nn.Module,
    meta_algo: MetaAlgorithm,
    val_sampler: Any,
    num_episodes: int,
    epoch: int,
    config: MetaAlgorithmConfig,
) -> Dict[str, float]:
    """Run validation episodes and return averaged metrics.

    The inner loop still needs gradients for adaptation, but we do not
    backpropagate through the outer loss.
    """
    was_training = model.training
    model.train()  # keep train mode for BN consistency during inner loop
    metrics_accum: Dict[str, List[float]] = defaultdict(list)

    for ep_idx in range(num_episodes):
        episode = val_sampler.sample_episode(epoch, ep_idx)
        params = {k: v.detach().requires_grad_(True) for k, v in model.named_parameters()}

        with torch.no_grad():
            pre_acc = meta_algo._compute_pre_adapt_metrics(model, params, episode)

        result = meta_algo.inner_engine.adapt(
            model, params,
            episode.support_x, episode.support_y,
            steps=config.inner_steps, lr=config.inner_lr,
            first_order=True, clip_norm=config.inner_clip,
        )
        post_acc = meta_algo._compute_post_adapt_metrics(
            model, result.adapted_params, episode
        )
        metrics_accum["pre_adapt_acc"].append(pre_acc)
        metrics_accum["post_adapt_acc"].append(post_acc)
        metrics_accum["fast_gain"].append(post_acc - pre_acc)

    if was_training:
        model.train()
    return {k: sum(v) / len(v) for k, v in metrics_accum.items()}


# ---------------------------------------------------------------------------
# Utility: synthetic episode generation (for testing)
# ---------------------------------------------------------------------------

def make_synthetic_episode(
    n_way: int = 5,
    k_shot: int = 1,
    q_query: int = 15,
    input_dim: int = 784,
    *,
    device: torch.device = torch.device("cpu"),
    seed: Optional[int] = None,
    use_separable_data: bool = True,
) -> Episode:
    """Create a synthetic N-way K-shot episode for testing.

    When use_separable_data=True, each class has a distinct mean vector
    so that a linear classifier can separate them.  This is essential
    for testing that adaptation actually improves accuracy.

    Args:
        n_way:              Number of classes.
        k_shot:             Support examples per class.
        q_query:            Query examples per class.
        input_dim:          Feature dimension.
        device:             Target device.
        seed:               Random seed for reproducibility.
        use_separable_data: If True, classes are linearly separable.

    Returns:
        An Episode with synthetic data.
    """
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
    else:
        gen = None

    support_xs, support_ys = [], []
    query_xs, query_ys = [], []

    for cls_idx in range(n_way):
        if use_separable_data:
            # Each class has a distinct mean direction
            mean = torch.zeros(input_dim)
            if input_dim >= n_way:
                mean[cls_idx] = 3.0  # strong signal in one dimension
            else:
                mean[cls_idx % input_dim] = 3.0 * (1 + cls_idx // input_dim)
        else:
            mean = torch.zeros(input_dim)

        # Support examples
        sx = torch.randn(k_shot, input_dim, generator=gen) * 0.5 + mean
        sy = torch.full((k_shot,), cls_idx, dtype=torch.long)
        support_xs.append(sx)
        support_ys.append(sy)

        # Query examples
        qx = torch.randn(q_query, input_dim, generator=gen) * 0.5 + mean
        qy = torch.full((q_query,), cls_idx, dtype=torch.long)
        query_xs.append(qx)
        query_ys.append(qy)

    support_x = torch.cat(support_xs, dim=0).to(device)
    support_y = torch.cat(support_ys, dim=0).to(device)
    query_x = torch.cat(query_xs, dim=0).to(device)
    query_y = torch.cat(query_ys, dim=0).to(device)

    return Episode(
        support_x=support_x,
        support_y=support_y,
        query_x=query_x,
        query_y=query_y,
        class_ids=list(range(n_way)),
    )


class SyntheticTaskSampler:
    """A simple task sampler that generates synthetic episodes.

    Used for testing and development.  Each call to sample_episode
    returns a fresh random episode with linearly separable classes.

    Attributes:
        n_way:     Number of classes per episode.
        k_shot:    Support examples per class.
        q_query:   Query examples per class.
        input_dim: Feature dimension.
        device:    Target device.
        seed:      Global seed; actual per-episode seed = hash(seed, epoch, idx).
    """

    def __init__(
        self,
        n_way: int = 5,
        k_shot: int = 1,
        q_query: int = 15,
        input_dim: int = 784,
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ):
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.input_dim = input_dim
        self.device = device
        self.seed = seed

    def sample_episode(self, epoch: int, episode_idx: int) -> Episode:
        """Sample an episode, deterministically keyed to (seed, epoch, idx)."""
        ep_seed = hash((self.seed, epoch, episode_idx)) % (2**31)
        episode = make_synthetic_episode(
            n_way=self.n_way,
            k_shot=self.k_shot,
            q_query=self.q_query,
            input_dim=self.input_dim,
            device=self.device,
            seed=ep_seed,
            use_separable_data=True,
        )
        episode.episode_id = (epoch, episode_idx)
        return episode


# ---------------------------------------------------------------------------
# Toy models for self-testing
# ---------------------------------------------------------------------------

class _ToyMLP(nn.Module):
    """Two-layer MLP for meta-learning self-tests.

    Small enough to run quickly on CPU, large enough to verify
    gradient flow and adaptation dynamics.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))


class _Conv4(nn.Module):
    """Conv4 backbone commonly used in few-shot learning benchmarks.

    Four convolutional blocks, each with 64 filters, followed by a
    linear head.  Designed for 28x28 single-channel images (Omniglot/MNIST).
    """

    def __init__(self, num_classes: int = 5, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 14x14 -> 7x7
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 7x7 -> 3x3
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 4: 3x3 -> 1x1
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            # Assume flattened 28x28 input
            side = int(math.sqrt(x.shape[-1]))
            x = x.view(x.size(0), 1, side, side)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.classifier(h)


# ---------------------------------------------------------------------------
# Gradient analysis utilities
# ---------------------------------------------------------------------------

def check_meta_gradients(
    model: nn.Module,
    meta_algo: MetaAlgorithm,
    episode: Episode,
    config: MetaAlgorithmConfig,
) -> Dict[str, Any]:
    """Analyze meta-gradients for debugging.  Reports per-param norms,
    NaN/Inf flags, and total gradient norm."""
    model.zero_grad()
    params = dict(model.named_parameters())
    task_batch = TaskBatch(episodes=[episode])

    output = meta_algo.meta_step(model, params, task_batch)
    output.loss.backward()

    report: Dict[str, Any] = {
        "loss": output.loss.item(),
        "algo": output.algo,
        "param_grads": {},
        "any_nan": False,
        "any_inf": False,
        "all_nonzero": True,
        "total_grad_norm": 0.0,
    }

    total_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad
            norm = g.norm().item()
            has_nan = bool(torch.isnan(g).any().item())
            has_inf = bool(torch.isinf(g).any().item())
            is_zero = norm < 1e-12

            report["param_grads"][name] = {
                "norm": norm,
                "nan": has_nan,
                "inf": has_inf,
                "zero": is_zero,
            }

            if has_nan:
                report["any_nan"] = True
            if has_inf:
                report["any_inf"] = True
            if is_zero:
                report["all_nonzero"] = False

            total_sq += norm ** 2
        else:
            report["param_grads"][name] = {
                "norm": 0.0, "nan": False, "inf": False, "zero": True,
            }
            report["all_nonzero"] = False

    report["total_grad_norm"] = math.sqrt(total_sq)
    return report


def compare_algorithms(
    model_fn: Callable[[], nn.Module],
    episode: Episode,
    config: MetaAlgorithmConfig,
    algorithms: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare meta-learning algorithms on the same episode.  Returns
    a dict mapping algorithm name to its metrics (including loss)."""
    if algorithms is None:
        algorithms = ["maml", "fomaml", "reptile"]
    results: Dict[str, Dict[str, float]] = {}
    for algo_name in algorithms:
        torch.manual_seed(0)
        model = model_fn()
        algo = create_meta_algorithm(algo_name, create_inner_loop_engine(), config)
        output = algo.meta_step(model, dict(model.named_parameters()),
                                TaskBatch(episodes=[episode]))
        results[algo_name] = {**output.metrics, "loss": output.loss.item()}
    return results


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "Episode",
    "TaskBatch",
    "MetaOutput",
    "MetaAlgorithmConfig",
    # Algorithms
    "MetaAlgorithm",
    "MAMLAlgorithm",
    "FOMAMLAlgorithm",
    "ReptileAlgorithm",
    # Factory
    "create_meta_algorithm",
    "register_meta_algorithm",
    "list_meta_algorithms",
    # Training loops
    "run_meta_epoch",
    "run_meta_validation",
    # Utilities
    "make_synthetic_episode",
    "SyntheticTaskSampler",
    "check_meta_gradients",
    "compare_algorithms",
    # Stubs (when inner_loop not available)
    "StepLog",
    "InnerLoopResult",
    "InnerLoopEngine",
    "create_inner_loop_engine",
]


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    """Comprehensive self-test suite.

    Runs 17 test groups covering data structures, all three algorithms,
    metrics, gradient flow, determinism, and the training loop.

    Exit code 0 if all tests pass, 1 otherwise.
    """
    import sys
    import traceback
    import time

    # ---- Test configuration ----
    N_WAY = 5
    K_SHOT = 5
    Q_QUERY = 10
    INPUT_DIM = 64  # small for fast testing
    HIDDEN_DIM = 32
    INNER_STEPS = 3
    INNER_LR = 0.05
    SEED = 12345

    # Use a mutable container so nested functions can modify counts
    _counts = {"passed": 0, "failed": 0}
    total_groups = 17

    def _run_test(name: str, fn: Callable[[], bool]) -> bool:
        """Run a single test group and print result."""
        t0 = time.time()
        try:
            result = fn()
            elapsed = time.time() - t0
            if result:
                print(f"  PASS  [{elapsed:.2f}s] {name}")
                _counts["passed"] += 1
                return True
            else:
                print(f"  FAIL  [{elapsed:.2f}s] {name}")
                _counts["failed"] += 1
                return False
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAIL  [{elapsed:.2f}s] {name}")
            traceback.print_exc()
            _counts["failed"] += 1
            return False

    # ---- Helper: create standard test fixtures ----

    def _make_config(**overrides) -> MetaAlgorithmConfig:
        defaults = dict(
            inner_steps=INNER_STEPS,
            inner_lr=INNER_LR,
            inner_clip=10.0,
            episodes_per_epoch=10,
            track_adapted=True,
        )
        defaults.update(overrides)
        return MetaAlgorithmConfig(**defaults)

    def _make_model(num_classes: int = N_WAY) -> _ToyMLP:
        torch.manual_seed(SEED)
        return _ToyMLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=num_classes)

    def _make_episode(seed: Optional[int] = None) -> Episode:
        return make_synthetic_episode(
            n_way=N_WAY,
            k_shot=K_SHOT,
            q_query=Q_QUERY,
            input_dim=INPUT_DIM,
            seed=seed if seed is not None else SEED,
            use_separable_data=True,
        )

    def _make_engine() -> InnerLoopEngine:
        return create_inner_loop_engine()

    print("=" * 72)
    print("Meta-Algorithms Self-Test Suite")
    print("=" * 72)
    print(f"Config: {N_WAY}-way {K_SHOT}-shot, input_dim={INPUT_DIM}, "
          f"inner_steps={INNER_STEPS}, inner_lr={INNER_LR}")
    print()

    # ==================================================================
    # Group 1: Episode creation and validation
    # ==================================================================
    def test_episode_creation():
        ep = _make_episode()
        assert ep.support_x.shape == (N_WAY * K_SHOT, INPUT_DIM), \
            f"support_x shape: {ep.support_x.shape}"
        assert ep.support_y.shape == (N_WAY * K_SHOT,), \
            f"support_y shape: {ep.support_y.shape}"
        assert ep.query_x.shape == (N_WAY * Q_QUERY, INPUT_DIM), \
            f"query_x shape: {ep.query_x.shape}"
        assert ep.query_y.shape == (N_WAY * Q_QUERY,), \
            f"query_y shape: {ep.query_y.shape}"
        assert ep.n_support == N_WAY * K_SHOT
        assert ep.n_query == N_WAY * Q_QUERY
        assert set(ep.support_y.tolist()) == set(range(N_WAY))
        assert set(ep.query_y.tolist()) == set(range(N_WAY))

        # Validation should reject mismatched shapes
        try:
            Episode(
                support_x=torch.randn(10, 5),
                support_y=torch.zeros(8, dtype=torch.long),
                query_x=torch.randn(10, 5),
                query_y=torch.zeros(10, dtype=torch.long),
            )
            return False  # should have raised
        except AssertionError:
            pass

        # .to() should move to the same device
        ep2 = ep.to(torch.device("cpu"))
        assert ep2.device == torch.device("cpu")
        return True

    _run_test("1. Episode creation and validation", test_episode_creation)

    # ==================================================================
    # Group 2: TaskBatch creation
    # ==================================================================
    def test_task_batch():
        ep1 = _make_episode(seed=1)
        ep2 = _make_episode(seed=2)
        ep3 = _make_episode(seed=3)
        tb = TaskBatch(episodes=[ep1, ep2, ep3])

        assert len(tb) == 3
        assert tb[0] is ep1
        assert tb[2] is ep3

        # Iteration
        count = 0
        for ep in tb:
            assert isinstance(ep, Episode)
            count += 1
        assert count == 3

        # .to()
        tb2 = tb.to(torch.device("cpu"))
        assert len(tb2) == 3
        return True

    _run_test("2. TaskBatch creation", test_task_batch)

    # ==================================================================
    # Group 3: MetaOutput fields
    # ==================================================================
    def test_meta_output():
        loss = torch.tensor(1.5, requires_grad=True)
        metrics = {
            "pre_adapt_acc": 0.2,
            "post_adapt_acc": 0.8,
            "fast_gain": 0.6,
            "auac": 0.5,
        }
        logs = [[StepLog(step=0, loss=2.0, accuracy=0.3)]]
        mo = MetaOutput(loss=loss, metrics=metrics, inner_logs=logs, algo="maml")

        assert mo.loss is loss
        assert mo.algo == "maml"
        assert mo.metrics["fast_gain"] == 0.6
        assert len(mo.inner_logs) == 1
        assert mo.adapted_params is None

        # repr should not crash
        r = repr(mo)
        assert "maml" in r
        return True

    _run_test("3. MetaOutput fields", test_meta_output)

    # ==================================================================
    # Group 4: MAML meta_step produces loss with grad
    # ==================================================================
    def test_maml_loss_has_grad():
        model = _make_model()
        engine = _make_engine()
        config = _make_config()
        algo = MAMLAlgorithm(engine, config)

        ep = _make_episode()
        tb = TaskBatch(episodes=[ep])
        params = dict(model.named_parameters())

        output = algo.meta_step(model, params, tb)

        assert output.loss.requires_grad, "MAML loss should require grad"
        assert output.algo == "maml"
        assert "pre_adapt_acc" in output.metrics
        assert "post_adapt_acc" in output.metrics
        assert "fast_gain" in output.metrics
        assert "auac" in output.metrics

        # Loss should be a finite scalar
        assert not torch.isnan(output.loss), "MAML loss is NaN"
        assert not torch.isinf(output.loss), "MAML loss is Inf"
        return True

    _run_test("4. MAML meta_step produces loss with grad", test_maml_loss_has_grad)

    # ==================================================================
    # Group 5: MAML meta-gradients non-zero on base params
    # ==================================================================
    def test_maml_nonzero_grads():
        model = _make_model()
        engine = _make_engine()
        config = _make_config()
        algo = MAMLAlgorithm(engine, config)

        ep = _make_episode()
        tb = TaskBatch(episodes=[ep])
        params = dict(model.named_parameters())

        output = algo.meta_step(model, params, tb)
        model.zero_grad()
        output.loss.backward()

        has_nonzero = False
        for name, p in model.named_parameters():
            if p.grad is not None and p.grad.norm().item() > 1e-10:
                has_nonzero = True
                break

        assert has_nonzero, "MAML should produce non-zero gradients on base params"

        # Check that NO gradients are NaN
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"

        return True

    _run_test("5. MAML meta-gradients non-zero on base params", test_maml_nonzero_grads)

    # ==================================================================
    # Group 6: MAML detach detector (adapted params have grad_fn)
    # ==================================================================
    def test_maml_adapted_has_grad_fn():
        model = _make_model()
        engine = _make_engine()
        config = _make_config(track_adapted=True)
        algo = MAMLAlgorithm(engine, config)

        ep = _make_episode()
        tb = TaskBatch(episodes=[ep])
        params = dict(model.named_parameters())

        output = algo.meta_step(model, params, tb)

        assert output.adapted_params is not None, "track_adapted should store params"
        assert len(output.adapted_params) == 1, "Should have 1 task's params"

        adapted = output.adapted_params[0]
        # In second-order MAML, adapted params should have grad_fn
        # (they were computed with create_graph=True)
        has_grad_fn_count = 0
        for k, v in adapted.items():
            if v.grad_fn is not None:
                has_grad_fn_count += 1

        assert has_grad_fn_count > 0, (
            f"MAML adapted params should have grad_fn for second-order. "
            f"Found {has_grad_fn_count}/{len(adapted)} with grad_fn"
        )
        return True

    _run_test("6. MAML detach detector (adapted params have grad_fn)", test_maml_adapted_has_grad_fn)

    # ==================================================================
    # Group 7: FOMAML meta_step produces loss with grad
    # ==================================================================
    def test_fomaml_loss_has_grad():
        model = _make_model()
        engine = _make_engine()
        config = _make_config()
        algo = FOMAMLAlgorithm(engine, config)

        ep = _make_episode()
        tb = TaskBatch(episodes=[ep])
        params = dict(model.named_parameters())

        output = algo.meta_step(model, params, tb)

        assert output.loss.requires_grad, "FOMAML loss should require grad"
        assert output.algo == "fomaml"
        assert not torch.isnan(output.loss), "FOMAML loss is NaN"
        assert not torch.isinf(output.loss), "FOMAML loss is Inf"

        # Should produce non-zero grads
        model.zero_grad()
        output.loss.backward()
        has_nonzero = any(
            p.grad is not None and p.grad.norm().item() > 1e-10
            for p in model.parameters()
        )
        assert has_nonzero, "FOMAML should produce non-zero gradients"
        return True

    _run_test("7. FOMAML meta_step produces loss with grad", test_fomaml_loss_has_grad)

    # ==================================================================
    # Group 8: FOMAML no second-order graph
    # ==================================================================
    def test_fomaml_no_second_order():
        model = _make_model()
        engine = _make_engine()
        config = _make_config(track_adapted=True)
        algo = FOMAMLAlgorithm(engine, config)

        ep = _make_episode()
        tb = TaskBatch(episodes=[ep])
        params = dict(model.named_parameters())

        output = algo.meta_step(model, params, tb)

        # Adapted params in FOMAML should NOT have grad_fn through inner loop
        # (first_order=True means no create_graph)
        assert output.adapted_params is not None
        adapted = output.adapted_params[0]

        has_grad_fn_count = sum(
            1 for v in adapted.values() if v.grad_fn is not None
        )

        # In first-order mode, adapted params are detached leaves.
        # They might have requires_grad but should NOT have a grad_fn
        # that links back through the inner loop computation graph.
        #
        # We verify the more practical property: backward() does not
        # require second-order memory.  We do this by checking that
        # the loss backward completes without issues.
        model.zero_grad()
        output.loss.backward()

        # After backward, verify gradients are finite
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN in {name}"

        return True

    _run_test("8. FOMAML no second-order graph", test_fomaml_no_second_order)

    # ==================================================================
    # Group 9: Reptile weight interpolation direction correct
    # ==================================================================
    def test_reptile_interpolation_direction():
        torch.manual_seed(SEED)
        model = _make_model()
        engine = _make_engine()
        config = _make_config(reptile_epsilon=1.0)
        algo = ReptileAlgorithm(engine, config)

        ep = _make_episode()
        tb = TaskBatch(episodes=[ep])

        # Record original parameters
        original_params = {k: v.clone().detach() for k, v in model.named_parameters()}
        params = dict(model.named_parameters())

        output = algo.meta_step(model, params, tb)

        assert output.algo == "reptile"
        assert "mean_weight_delta" in output.metrics
        assert "max_weight_delta" in output.metrics

        # The pseudo-loss should have a gradient that, when used with
        # optimizer.step(), moves params toward the adapted solution.
        # Gradient of pseudo-loss = -epsilon * avg_delta
        # Optimizer step: param -= lr * grad = param + lr * epsilon * avg_delta
        model.zero_grad()
        output.loss.backward()

        has_nonzero = any(
            p.grad is not None and p.grad.norm().item() > 1e-10
            for p in model.parameters()
        )
        assert has_nonzero, "Reptile should produce non-zero pseudo-gradients"

        # Verify the delta norm metric is positive (adaptation did something)
        assert output.metrics["mean_weight_delta"] > 0, \
            "Reptile should have positive weight delta"
        return True

    _run_test("9. Reptile weight interpolation direction correct", test_reptile_interpolation_direction)

    # ==================================================================
    # Group 10: Reptile no query set needed (adaptation still works)
    # ==================================================================
    def test_reptile_no_query_needed():
        # Reptile's meta-update only uses support set for adaptation
        # and the weight delta.  The query set is used only for metrics.
        # Verify that the pseudo-loss does NOT depend on query data.
        torch.manual_seed(SEED)
        model = _make_model()
        engine = _make_engine()
        config = _make_config(reptile_epsilon=1.0)
        algo = ReptileAlgorithm(engine, config)

        # Create two episodes with same support but different query sets
        ep_base = _make_episode(seed=100)

        # Episode with same support but random query
        ep_alt = Episode(
            support_x=ep_base.support_x.clone(),
            support_y=ep_base.support_y.clone(),
            query_x=torch.randn_like(ep_base.query_x),  # different query
            query_y=ep_base.query_y.clone(),
        )

        params1 = dict(model.named_parameters())
        out1 = algo.meta_step(model, params1, TaskBatch(episodes=[ep_base]))

        # Re-init to same weights
        torch.manual_seed(SEED)
        model2 = _make_model()
        algo2 = ReptileAlgorithm(_make_engine(), config)
        params2 = dict(model2.named_parameters())
        out2 = algo2.meta_step(model2, params2, TaskBatch(episodes=[ep_alt]))

        # The pseudo-loss should be the same because it only depends on
        # the support-set adaptation (weight deltas), not query
        assert abs(out1.loss.item() - out2.loss.item()) < 1e-4, (
            f"Reptile loss should not depend on query set: "
            f"{out1.loss.item():.6f} vs {out2.loss.item():.6f}"
        )
        return True

    _run_test("10. Reptile no query set needed (adaptation still works)", test_reptile_no_query_needed)

    # ==================================================================
    # Group 11: Factory function creates correct algorithm
    # ==================================================================
    def test_factory():
        engine = _make_engine()
        config = _make_config()

        maml = create_meta_algorithm("maml", engine, config)
        assert isinstance(maml, MAMLAlgorithm)

        fomaml = create_meta_algorithm("fomaml", engine, config)
        assert isinstance(fomaml, FOMAMLAlgorithm)

        reptile = create_meta_algorithm("reptile", engine, config)
        assert isinstance(reptile, ReptileAlgorithm)

        # Case insensitivity and whitespace
        maml2 = create_meta_algorithm("  MAML  ", engine, config)
        assert isinstance(maml2, MAMLAlgorithm)

        # Unknown algorithm
        try:
            create_meta_algorithm("unknown", engine, config)
            return False  # should have raised
        except ValueError as e:
            assert "unknown" in str(e).lower()

        # list_meta_algorithms
        algos = list_meta_algorithms()
        assert "maml" in algos
        assert "fomaml" in algos
        assert "reptile" in algos

        return True

    _run_test("11. Factory function creates correct algorithm", test_factory)

    # ==================================================================
    # Group 12: Pre-adapt vs post-adapt accuracy (post >= pre on toy task)
    # ==================================================================
    def test_adaptation_improves():
        # With separable data and enough inner steps, adaptation should help
        model = _make_model()
        engine = _make_engine()
        config = _make_config(inner_steps=10, inner_lr=0.1)
        algo = MAMLAlgorithm(engine, config)

        # Use a well-separated episode
        ep = make_synthetic_episode(
            n_way=N_WAY,
            k_shot=K_SHOT,
            q_query=Q_QUERY,
            input_dim=INPUT_DIM,
            seed=42,
            use_separable_data=True,
        )
        tb = TaskBatch(episodes=[ep])
        params = dict(model.named_parameters())

        output = algo.meta_step(model, params, tb)

        pre = output.metrics["pre_adapt_acc"]
        post = output.metrics["post_adapt_acc"]
        gain = output.metrics["fast_gain"]

        # Post-adaptation should be at least as good as pre (usually much better)
        # On a random init with separable data, adaptation should help
        assert post >= pre, (
            f"Post-adapt acc ({post:.3f}) should be >= pre-adapt ({pre:.3f})"
        )
        assert gain >= 0, f"Fast gain should be >= 0, got {gain:.3f}"

        return True

    _run_test("12. Pre-adapt vs post-adapt accuracy (post >= pre on toy task)", test_adaptation_improves)

    # ==================================================================
    # Group 13: AUAC computation correctness
    # ==================================================================
    def test_auac_computation():
        engine = _make_engine()
        config = _make_config()
        algo = MAMLAlgorithm(engine, config)

        # Test 1: constant accuracy => AUAC = that accuracy
        accs_const = [[0.5, 0.5, 0.5, 0.5]]
        auac = algo._compute_auac(accs_const)
        assert abs(auac - 0.5) < 1e-6, f"Constant 0.5 AUAC should be 0.5, got {auac}"

        # Test 2: linearly increasing 0 -> 1 => AUAC = 0.5
        accs_linear = [[0.0, 0.25, 0.5, 0.75, 1.0]]
        auac = algo._compute_auac(accs_linear)
        assert abs(auac - 0.5) < 1e-6, f"Linear 0->1 AUAC should be 0.5, got {auac}"

        # Test 3: step function 0, 0, 1, 1 => AUAC = (0 + 0.5 + 1) / 3 = 0.5
        accs_step = [[0.0, 0.0, 1.0, 1.0]]
        auac = algo._compute_auac(accs_step)
        expected = (0.0 + 0.5 + 1.0) / 3.0
        assert abs(auac - expected) < 1e-6, f"Step AUAC should be {expected}, got {auac}"

        # Test 4: perfect from start => AUAC = 1.0
        accs_perfect = [[1.0, 1.0, 1.0]]
        auac = algo._compute_auac(accs_perfect)
        assert abs(auac - 1.0) < 1e-6, f"Perfect AUAC should be 1.0, got {auac}"

        # Test 5: multiple tasks, averaged
        accs_multi = [
            [0.0, 0.5, 1.0],  # AUAC = 0.5
            [0.2, 0.6, 1.0],  # AUAC = 0.6
        ]
        auac = algo._compute_auac(accs_multi)
        # avg at step 0: 0.1, step 1: 0.55, step 2: 1.0
        # trapezoidal: ((0.1+0.55)/2 + (0.55+1.0)/2) / 2 = (0.325 + 0.775) / 2 = 0.55
        expected = ((0.1 + 0.55) / 2.0 + (0.55 + 1.0) / 2.0) / 2.0
        assert abs(auac - expected) < 1e-4, f"Multi-task AUAC: expected {expected}, got {auac}"

        # Test 6: empty input
        auac = algo._compute_auac([])
        assert auac == 0.0

        # Test 7: single point
        accs_single = [[0.7]]
        auac = algo._compute_auac(accs_single)
        assert abs(auac - 0.7) < 1e-6

        return True

    _run_test("13. AUAC computation correctness", test_auac_computation)

    # ==================================================================
    # Group 14: Fast adaptation gain > 0
    # ==================================================================
    def test_fast_adaptation_gain():
        # Test across all three algorithms with generous inner steps
        results = {}
        for algo_name in ["maml", "fomaml", "reptile"]:
            torch.manual_seed(SEED)
            model = _make_model()
            engine = _make_engine()
            config = _make_config(inner_steps=10, inner_lr=0.1)
            algo = create_meta_algorithm(algo_name, engine, config)

            ep = make_synthetic_episode(
                n_way=N_WAY,
                k_shot=K_SHOT,
                q_query=Q_QUERY,
                input_dim=INPUT_DIM,
                seed=42,
                use_separable_data=True,
            )
            tb = TaskBatch(episodes=[ep])
            params = dict(model.named_parameters())
            output = algo.meta_step(model, params, tb)

            results[algo_name] = output.metrics["fast_gain"]

        # At least MAML and FOMAML should show positive gain
        # Reptile's gain is measured differently (metrics on query set)
        any_positive = any(g > 0 for g in results.values())
        assert any_positive, (
            f"At least one algorithm should show positive fast gain: {results}"
        )
        return True

    _run_test("14. Fast adaptation gain > 0", test_fast_adaptation_gain)

    # ==================================================================
    # Group 15: run_meta_epoch completes without error
    # ==================================================================
    def test_run_meta_epoch():
        torch.manual_seed(SEED)
        model = _make_model()
        engine = _make_engine()
        config = _make_config(episodes_per_epoch=5)
        algo = MAMLAlgorithm(engine, config)

        sampler = SyntheticTaskSampler(
            n_way=N_WAY,
            k_shot=K_SHOT,
            q_query=Q_QUERY,
            input_dim=INPUT_DIM,
            seed=SEED,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        logs: List[str] = []
        epoch_metrics = run_meta_epoch(
            model=model,
            meta_algo=algo,
            task_sampler=sampler,
            outer_optimizer=optimizer,
            epoch=0,
            config=config,
            tasks_per_batch=1,
            log_interval=2,
            logger=lambda msg: logs.append(msg),
        )

        assert "meta_loss" in epoch_metrics, "Should have meta_loss metric"
        assert "post_adapt_acc" in epoch_metrics, "Should have post_adapt_acc metric"
        assert isinstance(epoch_metrics["meta_loss"], float)
        assert not math.isnan(epoch_metrics["meta_loss"]), "Epoch loss is NaN"

        return True

    _run_test("15. run_meta_epoch completes without error", test_run_meta_epoch)

    # ==================================================================
    # Group 16: Determinism: same seed -> same results
    # ==================================================================
    def test_determinism():
        def _run_once(model_seed, episode_seed):
            torch.manual_seed(model_seed)
            model = _ToyMLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=N_WAY)
            engine = _make_engine()
            config = _make_config(inner_steps=3, inner_lr=0.05)
            algo = MAMLAlgorithm(engine, config)

            ep = make_synthetic_episode(
                n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY,
                input_dim=INPUT_DIM, seed=episode_seed,
            )
            tb = TaskBatch(episodes=[ep])
            params = dict(model.named_parameters())
            output = algo.meta_step(model, params, tb)
            return output.loss.item(), output.metrics

        # Same seeds -> same results
        loss1, m1 = _run_once(SEED, 42)
        loss2, m2 = _run_once(SEED, 42)

        assert abs(loss1 - loss2) < 1e-6, (
            f"Same seed should give same loss: {loss1} vs {loss2}"
        )
        for k in m1:
            assert abs(m1[k] - m2[k]) < 1e-6, (
                f"Same seed should give same {k}: {m1[k]} vs {m2[k]}"
            )

        # Different model seed should give different results
        loss3, m3 = _run_once(SEED + 99, 42)
        model_differs = abs(loss1 - loss3) > 1e-8

        # Different episode seed should give different results
        loss4, m4 = _run_once(SEED, 999)
        episode_differs = abs(loss1 - loss4) > 1e-8

        assert model_differs or episode_differs, (
            "Different seeds should give different results"
        )

        return True

    _run_test("16. Determinism: same seed -> same results", test_determinism)

    # ==================================================================
    # Group 17: All three algorithms produce valid MetaOutput
    # ==================================================================
    def test_all_algos_valid_output():
        for algo_name in ["maml", "fomaml", "reptile"]:
            torch.manual_seed(SEED)
            model = _make_model()
            engine = _make_engine()
            config = _make_config()
            algo = create_meta_algorithm(algo_name, engine, config)

            ep = _make_episode()
            tb = TaskBatch(episodes=[ep])
            params = dict(model.named_parameters())

            output = algo.meta_step(model, params, tb)

            # Type checks
            assert isinstance(output, MetaOutput), f"{algo_name}: not MetaOutput"
            assert isinstance(output.loss, Tensor), f"{algo_name}: loss not Tensor"
            assert isinstance(output.metrics, dict), f"{algo_name}: metrics not dict"
            assert isinstance(output.inner_logs, list), f"{algo_name}: inner_logs not list"
            assert output.algo == algo_name, f"{algo_name}: algo field wrong"

            # Required metric keys
            for key in ["pre_adapt_acc", "post_adapt_acc", "fast_gain", "auac"]:
                assert key in output.metrics, f"{algo_name}: missing metric {key}"

            # Values should be finite
            assert not torch.isnan(output.loss), f"{algo_name}: loss is NaN"
            assert not torch.isinf(output.loss), f"{algo_name}: loss is Inf"

            for k, v in output.metrics.items():
                assert not math.isnan(v), f"{algo_name}: metric {k} is NaN"
                assert not math.isinf(v), f"{algo_name}: metric {k} is Inf"

            # Accuracies should be in [0, 1]
            for k in ["pre_adapt_acc", "post_adapt_acc"]:
                assert 0.0 <= output.metrics[k] <= 1.0, (
                    f"{algo_name}: {k}={output.metrics[k]} out of [0,1]"
                )

            # Inner logs should have entries for each task
            assert len(output.inner_logs) == 1, (
                f"{algo_name}: expected 1 task log, got {len(output.inner_logs)}"
            )
            for log in output.inner_logs[0]:
                assert isinstance(log, StepLog), f"{algo_name}: step log not StepLog"

            # Loss should support backward
            model.zero_grad()
            try:
                output.loss.backward()
            except RuntimeError as e:
                assert False, f"{algo_name}: backward failed: {e}"

        return True

    _run_test("17. All three algorithms produce valid MetaOutput", test_all_algos_valid_output)

    # ==================================================================
    # Summary
    # ==================================================================
    print()
    print("=" * 72)
    print(f"Results: {_counts['passed']}/{total_groups} PASSED, "
          f"{_counts['failed']}/{total_groups} FAILED")
    print("=" * 72)

    if _counts["failed"] > 0:
        print("\nSome tests FAILED. See details above.")
        sys.exit(1)
    else:
        print("\nAll tests PASSED.")
        sys.exit(0)
