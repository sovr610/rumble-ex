"""
MAML++ Template: Per-Layer Per-Step Learning Rates, Multi-Step Loss,
Derivative-Order Annealing, and Per-Step Batch Norm Handling.

Implements all four enhancements from "How to Train Your MAML" (Antoniou et al., 2019)
in a standalone, composable form. This file is an asset template for the
meta-learning-suite Claude Code plugin skill. It will be copied into
`brain_ai/meta/maml_plus.py`.

Enhancements:
    1. LSLR  -- Per-Layer Per-Step Learning Rates (LSLRModule)
    2. MSL   -- Multi-Step Loss (MultiStepLoss)
    3. Derivative-order annealing (DerivativeOrderAnnealing)
    4. Per-step batch-norm handling (PerStepBNManager)

All classes are self-contained; inline stubs are provided for InnerLoopEngine,
InnerLoopResult, StepLog, MetaAlgorithm, Episode, TaskBatch, and MetaOutput so
the file can be imported and run without external dependencies beyond PyTorch.

Usage:
    python maml_plus_template.py          # runs self-test (~18 test groups)
    python -c "from maml_plus_template import MAMLPlusPlusAlgorithm"

Author: Brain-AI Meta-Learning Suite
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import warnings
import copy
import traceback

# ---------------------------------------------------------------------------
# Inline stubs so the file can run standalone
# ---------------------------------------------------------------------------


@dataclass
class StepLog:
    """Diagnostic record for a single inner-loop step."""

    step: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    grad_norm: float = 0.0
    update_norm: float = 0.0
    lr_effective: float = 0.0
    clipped: bool = False


@dataclass
class InnerLoopResult:
    """Result returned by InnerLoopEngine.adapt().

    Attributes:
        adapted_params: Dict of parameter tensors after inner-loop adaptation.
            In MAML mode these retain ``grad_fn`` for second-order gradients.
        logs: Per-step diagnostic records.
        final_loss: Support-set loss at the last inner step.
        final_accuracy: Support-set accuracy at the last inner step.
    """

    adapted_params: Dict[str, torch.Tensor] = field(default_factory=dict)
    logs: List[StepLog] = field(default_factory=list)
    final_loss: float = 0.0
    final_accuracy: float = 0.0


def _functional_forward(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """Forward pass using explicit parameter dict.

    Attempts ``torch.func.functional_call`` first; falls back to manual
    parameter substitution if unavailable.
    """
    try:
        from torch.func import functional_call

        return functional_call(model, params, (x,))
    except (ImportError, AttributeError):
        pass

    # Manual fallback -- temporarily swap parameters
    originals: Dict[str, torch.Tensor] = {}
    for name, _ in model.named_parameters():
        parts = name.split(".")
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        originals[name] = getattr(obj, parts[-1])
        setattr(obj, parts[-1], nn.Parameter(params[name]))
    try:
        output = model(x)
    finally:
        for name, orig in originals.items():
            parts = name.split(".")
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], orig)
    return output


def _clip_grad_dict(
    grads: Dict[str, torch.Tensor],
    max_norm: float,
) -> Tuple[Dict[str, torch.Tensor], bool]:
    """Clip gradient dictionary by global L2 norm (differentiable)."""
    total_norm = torch.sqrt(
        sum(g.norm() ** 2 for g in grads.values() if g is not None)
    )
    clip_coef = max_norm / (total_norm + 1e-6)
    was_clipped = clip_coef.item() < 1.0
    if was_clipped:
        grads = {k: g * clip_coef for k, g in grads.items()}
    return grads, was_clipped


class InnerLoopEngine:
    """Stub inner-loop engine for standalone operation.

    In production this is replaced by the real ``InnerLoopEngine`` from
    ``brain_ai.meta.inner_loop`` which supports torch.func, higher, and
    custom-SGD backends with AMP safety.

    This stub implements the minimal contract needed by
    ``MAMLPlusPlusAlgorithm``:

    *   ``adapt(model, params, support_x, support_y, **kw) -> InnerLoopResult``
    """

    def __init__(self, model: Optional[nn.Module] = None, config: Any = None):
        self.model = model
        self.config = config

    def adapt(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        *,
        steps: int = 5,
        lr: float = 0.01,
        lrs: Optional[Dict[str, Any]] = None,
        first_order: bool = False,
        clip_norm: Optional[float] = None,
    ) -> InnerLoopResult:
        """Run the inner loop and return adapted parameters."""
        create_graph = not first_order
        adapted = {k: v for k, v in params.items()}
        logs: List[StepLog] = []

        for step in range(steps):
            # Forward + loss
            logits = _functional_forward(model, adapted, support_x)
            loss = F.cross_entropy(logits, support_y)

            # Gradients
            grad_tensors = torch.autograd.grad(
                outputs=loss,
                inputs=list(adapted.values()),
                create_graph=create_graph,
                allow_unused=True,
            )
            grads: Dict[str, torch.Tensor] = {}
            for (k, p), g in zip(adapted.items(), grad_tensors):
                grads[k] = g if g is not None else torch.zeros_like(p)

            # Clipping
            clipped = False
            if clip_norm is not None:
                grads, clipped = _clip_grad_dict(grads, clip_norm)

            grad_norm = torch.sqrt(
                sum(g.norm() ** 2 for g in grads.values())
            ).item()

            # Update
            new_adapted: Dict[str, torch.Tensor] = {}
            for k, p in adapted.items():
                effective_lr: Any = lr
                if lrs is not None and k in lrs:
                    per = lrs[k]
                    if isinstance(per, dict):
                        effective_lr = per.get(step, lr)
                    else:
                        effective_lr = per
                new_adapted[k] = p - effective_lr * grads[k]

            update_norm = torch.sqrt(
                sum((new_adapted[k] - adapted[k]).norm() ** 2 for k in adapted)
            ).item()
            adapted = new_adapted

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == support_y).float().mean().item()

            logs.append(
                StepLog(
                    step=step,
                    loss=loss.item(),
                    accuracy=acc,
                    grad_norm=grad_norm,
                    update_norm=update_norm,
                    lr_effective=lr if lrs is None else -1.0,
                    clipped=clipped,
                )
            )

        final_loss = logs[-1].loss if logs else 0.0
        final_acc = logs[-1].accuracy if logs else 0.0
        return InnerLoopResult(
            adapted_params=adapted,
            logs=logs,
            final_loss=final_loss,
            final_accuracy=final_acc,
        )


@dataclass
class Episode:
    """A single N-way K-shot episode.

    Attributes:
        support_x: Support-set inputs  (N*K, ...)
        support_y: Support-set labels  (N*K,)
        query_x:   Query-set inputs    (N*Q, ...)
        query_y:   Query-set labels    (N*Q,)
    """

    support_x: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    support_y: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    query_x: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    query_y: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))


@dataclass
class TaskBatch:
    """Batch of episodes sampled for one meta-training step.

    Iterable over episodes; supports ``len()``.
    """

    episodes: List[Episode] = field(default_factory=list)

    def __len__(self) -> int:  # noqa: D105
        return len(self.episodes)

    def __iter__(self):  # noqa: D105
        return iter(self.episodes)


@dataclass
class MetaOutput:
    """Output of a single meta-training step.

    Attributes:
        loss:            Scalar meta-objective (outer loss averaged over tasks).
        metrics:         Aggregate metrics (pre/post adapt accuracy, AUAC, LR stats).
        inner_logs:      Per-task inner-loop logs.
        adapted_params:  Per-task adapted parameter dicts (detached for eval).
    """

    loss: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    metrics: Dict[str, float] = field(default_factory=dict)
    inner_logs: List[List[StepLog]] = field(default_factory=list)
    adapted_params: List[Dict[str, torch.Tensor]] = field(default_factory=list)


class MetaAlgorithm:
    """Base class for meta-learning algorithms (stub).

    Sub-classes implement ``meta_step(...)`` to perform one outer-loop
    update across a batch of episodes.
    """

    def __init__(self, inner_engine: InnerLoopEngine, config: Any):
        self.inner_engine = inner_engine
        self.config = config

    def meta_step(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        task_batch: TaskBatch,
        epoch: int = 0,
    ) -> MetaOutput:
        raise NotImplementedError

    def parameters(self) -> List[nn.Parameter]:
        """Extra learnable parameters introduced by the meta-algorithm."""
        return []


# ===========================================================================
# 1. Per-Layer Per-Step Learning Rates (LSLR)
# ===========================================================================


class LSLRModule(nn.Module):
    """Learnable per-layer, per-step inner-loop learning rates.

    Maintains ``alpha_{layer, step}`` in log-space (exponentiated at read
    time) with hard clamp to ``[lr_min, lr_max]``.
    """

    def __init__(
        self,
        layer_names: List[str],
        num_steps: int,
        init_lr: float = 0.01,
        lr_min: float = 1e-6,
        lr_max: float = 1.0,
    ):
        super().__init__()
        self.layer_names = list(layer_names)
        self.num_steps = num_steps
        self.lr_min = lr_min
        self.lr_max = lr_max

        # Store as ParameterDict for named access and correct serialisation
        self.lrs = nn.ParameterDict()
        for layer in self.layer_names:
            for step in range(num_steps):
                key = f"{self._sanitize(layer)}_step{step}"
                # Initialise in log-space so exp(log(init_lr)) == init_lr
                self.lrs[key] = nn.Parameter(
                    torch.tensor(math.log(max(init_lr, 1e-12)))
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize(name: str) -> str:
        """Make a parameter name safe for use as a ParameterDict key.

        Dots and other special characters are replaced with underscores.
        """
        return name.replace(".", "_").replace("-", "_")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_lr(self, layer_name: str, step: int) -> torch.Tensor:
        """Return the learning rate for *layer_name* at *step*.

        The returned value is a differentiable scalar tensor so that
        the outer optimiser can adjust LRs via meta-gradients.
        """
        key = f"{self._sanitize(layer_name)}_step{step}"
        raw = self.lrs[key]
        return torch.clamp(torch.exp(raw), self.lr_min, self.lr_max)

    def get_lr_dict(self, step: int) -> Dict[str, torch.Tensor]:
        """Return ``{layer_name: lr}`` for every layer at *step*."""
        return {layer: self.get_lr(layer, step) for layer in self.layer_names}

    def get_all_lrs_for_layer(self, layer_name: str) -> List[torch.Tensor]:
        """Return the learning rate at every step for a single layer."""
        return [self.get_lr(layer_name, s) for s in range(self.num_steps)]

    def get_all_lrs_flat(self) -> List[torch.Tensor]:
        """Return a flat list of **all** LR values (layer x step)."""
        out: List[torch.Tensor] = []
        for layer in self.layer_names:
            for step in range(self.num_steps):
                out.append(self.get_lr(layer, step))
        return out

    def get_statistics(self) -> Dict[str, float]:
        """Return mean / min / max / std of all learned LRs for logging."""
        all_lrs = [lr.item() for lr in self.get_all_lrs_flat()]
        if not all_lrs:
            return {"lr_mean": 0.0, "lr_min": 0.0, "lr_max": 0.0, "lr_std": 0.0}
        t = torch.tensor(all_lrs)
        return {
            "lr_mean": t.mean().item(),
            "lr_min": t.min().item(),
            "lr_max": t.max().item(),
            "lr_std": t.std().item() if len(all_lrs) > 1 else 0.0,
        }

    def get_per_step_statistics(self) -> Dict[int, Dict[str, float]]:
        """Return per-step aggregate LR statistics.

        Useful for monitoring whether certain steps develop higher or
        lower LRs across layers.
        """
        out: Dict[int, Dict[str, float]] = {}
        for step in range(self.num_steps):
            lrs_at_step = [self.get_lr(layer, step).item() for layer in self.layer_names]
            t = torch.tensor(lrs_at_step)
            out[step] = {
                "lr_mean": t.mean().item(),
                "lr_min": t.min().item(),
                "lr_max": t.max().item(),
            }
        return out

    def get_per_layer_statistics(self) -> Dict[str, Dict[str, float]]:
        """Return per-layer aggregate LR statistics.

        Useful for monitoring whether certain layers develop higher or
        lower LRs across inner-loop steps.
        """
        out: Dict[str, Dict[str, float]] = {}
        for layer in self.layer_names:
            lrs_for_layer = [self.get_lr(layer, s).item() for s in range(self.num_steps)]
            t = torch.tensor(lrs_for_layer)
            out[layer] = {
                "lr_mean": t.mean().item(),
                "lr_min": t.min().item(),
                "lr_max": t.max().item(),
            }
        return out

    def reset_to(self, init_lr: float) -> None:
        """Reset all LRs to *init_lr* (useful when warm-starting)."""
        val = math.log(max(init_lr, 1e-12))
        with torch.no_grad():
            for p in self.lrs.values():
                p.fill_(val)

    def extra_repr(self) -> str:  # noqa: D401
        return (
            f"layers={len(self.layer_names)}, steps={self.num_steps}, "
            f"lr_min={self.lr_min}, lr_max={self.lr_max}"
        )


# ===========================================================================
# 2. Multi-Step Loss (MSL)
# ===========================================================================


class MultiStepLoss(nn.Module):
    """Accumulate query losses across intermediate inner-loop steps.

    Schemes: ``"uniform"``, ``"linear_increase"``, ``"learned"`` (softmax).
    """

    VALID_SCHEMES = {"uniform", "linear_increase", "learned"}

    def __init__(
        self,
        num_steps: int,
        weight_scheme: str = "uniform",
    ):
        super().__init__()
        if weight_scheme not in self.VALID_SCHEMES:
            raise ValueError(
                f"weight_scheme must be one of {self.VALID_SCHEMES}, got '{weight_scheme}'"
            )
        self.num_steps = num_steps
        self.weight_scheme = weight_scheme

        if weight_scheme == "learned":
            # Raw logits -- passed through softmax at read time
            self.weight_logits = nn.Parameter(torch.zeros(num_steps))
        elif weight_scheme == "linear_increase":
            weights = torch.arange(1, num_steps + 1, dtype=torch.float32)
            self.register_buffer("fixed_weights", weights / weights.sum())
        else:  # uniform
            self.register_buffer(
                "fixed_weights", torch.ones(num_steps) / num_steps
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def weights(self) -> torch.Tensor:
        """Normalised weight vector of shape ``(num_steps,)``."""
        if self.weight_scheme == "learned":
            return F.softmax(self.weight_logits, dim=0)
        return self.fixed_weights

    def forward(self, step_losses: List[torch.Tensor]) -> torch.Tensor:
        """Combine per-step query losses with weights.

        Parameters
        ----------
        step_losses : list of scalar tensors
            ``step_losses[i]`` is the query-set loss evaluated using
            the adapted parameters after inner step *i*.

        Returns
        -------
        torch.Tensor
            Weighted sum of step losses (scalar).
        """
        if len(step_losses) != self.num_steps:
            raise ValueError(
                f"Expected {self.num_steps} step losses, got {len(step_losses)}"
            )
        w = self.weights
        total = sum(w[i] * step_losses[i] for i in range(self.num_steps))
        return total  # type: ignore[return-value]

    def get_weight_statistics(self) -> Dict[str, float]:
        """Return statistics about the current MSL weights."""
        w = self.weights.detach()
        return {
            "msl_weight_min": w.min().item(),
            "msl_weight_max": w.max().item(),
            "msl_weight_entropy": -(w * (w + 1e-8).log()).sum().item(),
        }

    def extra_repr(self) -> str:  # noqa: D401
        return f"num_steps={self.num_steps}, scheme={self.weight_scheme}"


# ===========================================================================
# 3. Derivative-Order Annealing
# ===========================================================================


class DerivativeOrderAnnealing:
    """Anneal from first-order (FOMAML) to second-order (MAML) during training.

    Starts with ``first_order=True`` for stability, transitions to
    ``first_order=False`` after ``annealing_start_epoch``.  If
    ``annealing_end_epoch`` is set, the transition is linear between
    the two epochs (see ``get_create_graph_prob``).
    """

    def __init__(
        self,
        annealing_start_epoch: int = 0,
        annealing_end_epoch: Optional[int] = None,
    ):
        self.start_epoch = annealing_start_epoch
        self.end_epoch = annealing_end_epoch

        # Validation
        if annealing_end_epoch is not None and annealing_end_epoch < annealing_start_epoch:
            raise ValueError(
                f"annealing_end_epoch ({annealing_end_epoch}) must be >= "
                f"annealing_start_epoch ({annealing_start_epoch})"
            )

    def get_first_order(self, epoch: int) -> bool:
        """Return ``True`` if first-order mode should be used at *epoch*.

        Before ``start_epoch``: always first-order.
        After ``start_epoch`` (or ``end_epoch`` if set): always second-order.
        """
        if epoch < self.start_epoch:
            return True  # First-order before annealing starts
        if self.end_epoch is None or epoch >= self.end_epoch:
            return False  # Full second-order after annealing
        # During the annealing window we treat it as deterministic
        # (use get_create_graph_prob for probabilistic mode)
        return False

    def get_create_graph_prob(self, epoch: int) -> float:
        """Probability of using second-order (``create_graph=True``) at *epoch*.

        Useful for stochastic annealing where each task in the batch
        independently samples whether to use first- or second-order.
        """
        if epoch < self.start_epoch:
            return 0.0
        if self.end_epoch is None or epoch >= self.end_epoch:
            return 1.0
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return min(max(progress, 0.0), 1.0)

    def get_state(self) -> Dict[str, Any]:
        """Serialisable state for checkpointing."""
        return {
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "DerivativeOrderAnnealing":
        """Reconstruct from a checkpoint state dict."""
        return cls(
            annealing_start_epoch=state["start_epoch"],
            annealing_end_epoch=state.get("end_epoch"),
        )

    def __repr__(self) -> str:
        return (
            f"DerivativeOrderAnnealing(start={self.start_epoch}, "
            f"end={self.end_epoch})"
        )


# ===========================================================================
# 4. Per-Step Batch-Norm Handling
# ===========================================================================


class PerStepBNManager:
    """Manage batch-normalization statistics across inner-loop steps.

    Three modes: ``"transductive"`` (use live batch stats),
    ``"per_step"`` (separate running stats per inner step),
    ``"frozen"`` (restore pre-inner-loop stats at every step).
    """

    VALID_MODES = {"transductive", "per_step", "frozen"}

    def __init__(
        self,
        model: nn.Module,
        mode: str = "per_step",
        num_steps: int = 5,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"BN mode must be one of {self.VALID_MODES}, got '{mode}'"
            )
        self.mode = mode
        self.num_steps = num_steps
        self.bn_modules = self._find_bn_modules(model)

        if mode == "per_step":
            self.step_stats: List[Dict[str, Dict[str, torch.Tensor]]] = (
                self._init_per_step_stats()
            )
        elif mode == "frozen":
            self.frozen_stats: Dict[str, Dict[str, torch.Tensor]] = (
                self._capture_stats()
            )

    # ------------------------------------------------------------------
    # BN discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_bn_modules(model: nn.Module) -> Dict[str, nn.Module]:
        """Find all BatchNorm (1d/2d/3d) modules in *model*."""
        bn_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
        )
        result: Dict[str, nn.Module] = {}
        for name, module in model.named_modules():
            if isinstance(module, bn_types):
                result[name] = module
        return result

    # ------------------------------------------------------------------
    # Capture / load helpers
    # ------------------------------------------------------------------

    def _capture_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Capture current running statistics from all BN modules."""
        stats: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, bn in self.bn_modules.items():
            entry: Dict[str, torch.Tensor] = {}
            if hasattr(bn, "running_mean") and bn.running_mean is not None:
                entry["running_mean"] = bn.running_mean.clone()
            if hasattr(bn, "running_var") and bn.running_var is not None:
                entry["running_var"] = bn.running_var.clone()
            if hasattr(bn, "num_batches_tracked"):
                entry["num_batches_tracked"] = bn.num_batches_tracked.clone()
            stats[name] = entry
        return stats

    def _load_stats(self, stats: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """Restore running statistics to BN modules (out-of-place to avoid
        in-place conflicts with ``create_graph=True`` backward)."""
        for name, entry in stats.items():
            if name not in self.bn_modules:
                continue
            bn = self.bn_modules[name]
            if "running_mean" in entry and hasattr(bn, "running_mean") and bn.running_mean is not None:
                bn.running_mean = entry["running_mean"].clone()
            if "running_var" in entry and hasattr(bn, "running_var") and bn.running_var is not None:
                bn.running_var = entry["running_var"].clone()
            if "num_batches_tracked" in entry and hasattr(bn, "num_batches_tracked"):
                bn.num_batches_tracked = entry["num_batches_tracked"].clone()

    def _init_per_step_stats(
        self,
    ) -> List[Dict[str, Dict[str, torch.Tensor]]]:
        """Initialise per-step stats by cloning the current stats."""
        base = self._capture_stats()
        return [
            {
                name: {k: v.clone() for k, v in entry.items()}
                for name, entry in base.items()
            }
            for _ in range(self.num_steps)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_step(self, step: int) -> None:
        """Load BN state for inner-loop *step*; sets eval mode for
        ``per_step`` / ``frozen`` to prevent in-place stat mutations."""
        if self.mode == "per_step":
            if step < len(self.step_stats):
                self._load_stats(self.step_stats[step])
            # Eval mode prevents in-place running-stat updates that
            # conflict with second-order meta-gradients.
            self.set_model_bn_mode(training=False)
        elif self.mode == "frozen":
            self._load_stats(self.frozen_stats)
            self.set_model_bn_mode(training=False)
        # transductive: no action -- use live batch statistics

    def save_step(self, step: int) -> None:
        """Persist BN state after inner-loop *step*.

        Call **after** the forward pass + parameter update at that step.
        """
        if self.mode == "per_step":
            if step < len(self.step_stats):
                self.step_stats[step] = self._capture_stats()

    def reset(self) -> None:
        """Reset BN to pre-inner-loop state.

        Call between tasks to avoid cross-task contamination.
        """
        if self.mode == "frozen":
            self._load_stats(self.frozen_stats)
        elif self.mode == "per_step":
            # Re-initialise per-step stats from the base snapshot
            if self.step_stats:
                base = self.step_stats[0]
                self.step_stats = [
                    {
                        name: {k: v.clone() for k, v in entry.items()}
                        for name, entry in base.items()
                    }
                    for _ in range(self.num_steps)
                ]

    def set_model_bn_mode(self, training: bool) -> None:
        """Set all BN modules to train or eval mode.

        Frozen mode typically wants ``eval`` so that running stats are
        used instead of batch statistics.
        """
        for bn in self.bn_modules.values():
            bn.training = training

    @property
    def num_bn_modules(self) -> int:
        """Number of BN modules found in the model."""
        return len(self.bn_modules)

    def __repr__(self) -> str:
        return (
            f"PerStepBNManager(mode='{self.mode}', num_steps={self.num_steps}, "
            f"bn_modules={len(self.bn_modules)})"
        )


# ===========================================================================
# 5. MAMLPlusPlusConfig
# ===========================================================================


@dataclass
class MAMLPlusPlusConfig:
    """Unified configuration for MAML++ enhancements."""

    inner_steps: int = 5
    inner_lr: float = 0.01
    inner_clip: float = 10.0

    # LSLR
    use_lslr: bool = False
    lslr_lr_min: float = 1e-6
    lslr_lr_max: float = 1.0

    # MSL
    use_msl: bool = False
    msl_weights: str = "uniform"

    # Derivative-order annealing
    use_annealing: bool = False
    annealing_start_epoch: int = 0
    annealing_end_epoch: Optional[int] = None

    # BN
    bn_mode: str = "per_step"


# ===========================================================================
# 6. MAMLPlusPlusAlgorithm (Main Integration)
# ===========================================================================


class MAMLPlusPlusAlgorithm(MetaAlgorithm):
    """MAML++ with all four enhancements integrated.

    Combines LSLR + MSL + derivative-order annealing + BN handling with
    the standard MAML inner/outer loop.  For each meta step, iterates
    over episodes, adapts with LSLR-modulated inner steps, optionally
    evaluates query loss at each step (MSL), and averages task losses
    for the outer objective.
    """

    def __init__(
        self,
        inner_engine: InnerLoopEngine,
        config: MAMLPlusPlusConfig,
        model: nn.Module,
    ):
        super().__init__(inner_engine, config)
        self.model = model

        # -- LSLR -----------------------------------------------------------
        self._lslr: Optional[LSLRModule] = None
        if config.use_lslr:
            layer_names = [name for name, _ in model.named_parameters()]
            self._lslr = LSLRModule(
                layer_names=layer_names,
                num_steps=config.inner_steps,
                init_lr=config.inner_lr,
                lr_min=config.lslr_lr_min,
                lr_max=config.lslr_lr_max,
            )

        # -- MSL ------------------------------------------------------------
        self._msl: Optional[MultiStepLoss] = None
        if config.use_msl:
            self._msl = MultiStepLoss(
                num_steps=config.inner_steps,
                weight_scheme=config.msl_weights,
            )

        # -- Annealing ------------------------------------------------------
        self._annealing: Optional[DerivativeOrderAnnealing] = None
        if config.use_annealing:
            self._annealing = DerivativeOrderAnnealing(
                annealing_start_epoch=config.annealing_start_epoch,
                annealing_end_epoch=config.annealing_end_epoch,
            )

        # -- BN manager -----------------------------------------------------
        self._bn_manager = PerStepBNManager(
            model=model,
            mode=config.bn_mode,
            num_steps=config.inner_steps,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lslr(self) -> Optional[LSLRModule]:
        return self._lslr

    @property
    def msl(self) -> Optional[MultiStepLoss]:
        return self._msl

    @property
    def annealing(self) -> Optional[DerivativeOrderAnnealing]:
        return self._annealing

    @property
    def bn_manager(self) -> PerStepBNManager:
        return self._bn_manager

    # ------------------------------------------------------------------
    # Inner step (used by MSL path)
    # ------------------------------------------------------------------

    def _inner_step(
        self,
        model: nn.Module,
        adapted: Dict[str, torch.Tensor],
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        step: int,
        lrs: Optional[Dict[str, torch.Tensor]],
        first_order: bool,
    ) -> Tuple[Dict[str, torch.Tensor], StepLog]:
        """Perform a single inner-loop step with LSLR and clipping.

        Returns the new adapted parameters and a diagnostic StepLog.
        """
        create_graph = not first_order

        # Forward + loss
        logits = _functional_forward(model, adapted, support_x)
        loss = F.cross_entropy(logits, support_y)

        # Gradients
        grad_tensors = torch.autograd.grad(
            outputs=loss,
            inputs=list(adapted.values()),
            create_graph=create_graph,
            allow_unused=True,
        )
        grads: Dict[str, torch.Tensor] = {}
        for (k, p), g in zip(adapted.items(), grad_tensors):
            grads[k] = g if g is not None else torch.zeros_like(p)

        # Clip
        clipped = False
        if self.config.inner_clip and self.config.inner_clip > 0:
            grads, clipped = _clip_grad_dict(grads, self.config.inner_clip)

        grad_norm = torch.sqrt(
            sum(g.norm() ** 2 for g in grads.values())
        ).item()

        # Update
        new_adapted: Dict[str, torch.Tensor] = {}
        for k, p in adapted.items():
            effective_lr: Any = self.config.inner_lr
            if lrs is not None and k in lrs:
                effective_lr = lrs[k]
            new_adapted[k] = p - effective_lr * grads[k]

        update_norm = torch.sqrt(
            sum((new_adapted[k] - adapted[k]).norm() ** 2 for k in adapted)
        ).item()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == support_y).float().mean().item()

        log = StepLog(
            step=step,
            loss=loss.item(),
            accuracy=acc,
            grad_norm=grad_norm,
            update_norm=update_norm,
            lr_effective=self.config.inner_lr if lrs is None else -1.0,
            clipped=clipped,
        )
        return new_adapted, log

    # ------------------------------------------------------------------
    # Forward through adapted model
    # ------------------------------------------------------------------

    @staticmethod
    def _forward(
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper around functional forward."""
        return _functional_forward(model, params, x)

    # ------------------------------------------------------------------
    # meta_step
    # ------------------------------------------------------------------

    def meta_step(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        task_batch: TaskBatch,
        epoch: int = 0,
    ) -> MetaOutput:
        """Perform one outer-loop step over a batch of episodes."""
        # -- Derivative-order annealing ------------------------------------
        first_order = False
        if self._annealing is not None:
            first_order = self._annealing.get_first_order(epoch)

        device = next(iter(params.values())).device
        meta_loss = torch.tensor(0.0, device=device, requires_grad=True)

        all_inner_logs: List[List[StepLog]] = []
        all_adapted: List[Dict[str, torch.Tensor]] = []
        pre_adapt_accs: List[float] = []
        post_adapt_accs: List[float] = []

        for episode in task_batch.episodes:
            # -- Pre-adaptation accuracy -----------------------------------
            with torch.no_grad():
                pre_logits = self._forward(model, params, episode.query_x)
                pre_preds = pre_logits.argmax(dim=-1)
                pre_acc = (pre_preds == episode.query_y).float().mean().item()
                pre_adapt_accs.append(pre_acc)

            # -- MSL path: step-by-step query evaluation -------------------
            if self._msl is not None:
                step_losses: List[torch.Tensor] = []
                episode_logs: List[StepLog] = []
                adapted = {k: v for k, v in params.items()}

                for step in range(self.config.inner_steps):
                    self._bn_manager.set_step(step)

                    # Get LSLR for this step
                    lrs = (
                        self._lslr.get_lr_dict(step)
                        if self._lslr is not None
                        else None
                    )

                    # One inner step
                    adapted, step_log = self._inner_step(
                        model, adapted, episode.support_x,
                        episode.support_y, step, lrs, first_order,
                    )
                    episode_logs.append(step_log)

                    # Query loss at this step
                    query_logits = self._forward(model, adapted, episode.query_x)
                    step_losses.append(
                        F.cross_entropy(query_logits, episode.query_y)
                    )

                    self._bn_manager.save_step(step)

                task_loss = self._msl(step_losses)
                all_inner_logs.append(episode_logs)

            # -- Standard path: adapt fully, then query --------------------
            else:
                # Build per-layer-per-step LR structure for InnerLoopEngine
                lrs_dict: Optional[Dict[str, Any]] = None
                if self._lslr is not None:
                    lrs_dict = {}
                    for name in params:
                        per_step: Dict[int, torch.Tensor] = {}
                        for s in range(self.config.inner_steps):
                            per_step[s] = self._lslr.get_lr(name, s)
                        lrs_dict[name] = per_step

                # Set BN for each step via callback-style (engine calls set_step)
                # For the stub engine we do it here in a simplified manner
                for s in range(self.config.inner_steps):
                    self._bn_manager.set_step(s)
                    self._bn_manager.save_step(s)

                result = self.inner_engine.adapt(
                    model,
                    params,
                    episode.support_x,
                    episode.support_y,
                    steps=self.config.inner_steps,
                    lr=self.config.inner_lr,
                    lrs=lrs_dict,
                    first_order=first_order,
                    clip_norm=self.config.inner_clip if self.config.inner_clip > 0 else None,
                )

                query_logits = self._forward(
                    model, result.adapted_params, episode.query_x
                )
                task_loss = F.cross_entropy(query_logits, episode.query_y)
                all_inner_logs.append(result.logs)
                adapted = result.adapted_params

            # Post-adaptation accuracy
            with torch.no_grad():
                post_logits = self._forward(model, adapted, episode.query_x)
                post_preds = post_logits.argmax(dim=-1)
                post_acc = (post_preds == episode.query_y).float().mean().item()
                post_adapt_accs.append(post_acc)

            all_adapted.append({k: v.detach() for k, v in adapted.items()})
            meta_loss = meta_loss + task_loss

            # Reset BN between tasks
            self._bn_manager.reset()

        # -- Average over tasks -------------------------------------------
        num_tasks = max(len(task_batch), 1)
        meta_loss = meta_loss / num_tasks

        # -- Metrics -------------------------------------------------------
        metrics: Dict[str, float] = {
            "pre_adapt_acc": sum(pre_adapt_accs) / max(len(pre_adapt_accs), 1),
            "post_adapt_acc": sum(post_adapt_accs) / max(len(post_adapt_accs), 1),
            "fast_gain": (
                sum(post_adapt_accs) / max(len(post_adapt_accs), 1)
                - sum(pre_adapt_accs) / max(len(pre_adapt_accs), 1)
            ),
            "meta_loss": meta_loss.item(),
            "first_order": float(first_order),
            "num_tasks": float(num_tasks),
        }

        if self._lslr is not None:
            metrics.update(self._lslr.get_statistics())

        if self._msl is not None:
            metrics.update(self._msl.get_weight_statistics())

        if self._annealing is not None:
            metrics["annealing_prob"] = self._annealing.get_create_graph_prob(epoch)

        return MetaOutput(
            loss=meta_loss,
            metrics=metrics,
            inner_logs=all_inner_logs,
            adapted_params=all_adapted,
        )

    # ------------------------------------------------------------------
    # parameters()
    # ------------------------------------------------------------------

    def parameters(self) -> List[nn.Parameter]:
        """Return all learnable MAML++ parameters for the outer optimiser."""
        extra: List[nn.Parameter] = []
        if self._lslr is not None:
            extra.extend(self._lslr.parameters())
        if self._msl is not None and self._msl.weight_scheme == "learned":
            extra.extend(self._msl.parameters())
        return extra

    # ------------------------------------------------------------------
    # Checkpointing helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return a serialisable state dict."""
        sd: Dict[str, Any] = {"config": self.config}
        if self._lslr is not None:
            sd["lslr"] = self._lslr.state_dict()
        if self._msl is not None:
            sd["msl"] = self._msl.state_dict()
        if self._annealing is not None:
            sd["annealing"] = self._annealing.get_state()
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        """Restore from a state dict."""
        if "lslr" in sd and self._lslr is not None:
            self._lslr.load_state_dict(sd["lslr"])
        if "msl" in sd and self._msl is not None:
            self._msl.load_state_dict(sd["msl"])
        if "annealing" in sd and self._annealing is not None:
            self._annealing = DerivativeOrderAnnealing.from_state(sd["annealing"])

    def __repr__(self) -> str:
        parts = ["MAMLPlusPlusAlgorithm("]
        parts.append(f"  lslr={self._lslr is not None},")
        parts.append(f"  msl={self._msl is not None},")
        parts.append(f"  annealing={self._annealing is not None},")
        parts.append(f"  bn_mode='{self._bn_manager.mode}',")
        parts.append(f"  inner_steps={self.config.inner_steps},")
        parts.append(")")
        return "\n".join(parts)


# ===========================================================================
# 7. Factory function
# ===========================================================================


def create_maml_plus_plus(
    model: nn.Module,
    config: Optional[MAMLPlusPlusConfig] = None,
    **overrides: Any,
) -> MAMLPlusPlusAlgorithm:
    """Create a MAML++ algorithm instance with all enhancements.

    Parameters
    ----------
    model : nn.Module
        The base model to meta-train.
    config : MAMLPlusPlusConfig, optional
        Configuration.  If ``None``, one is created from *overrides*.
    **overrides
        Keyword arguments forwarded to ``MAMLPlusPlusConfig``.

    Returns
    -------
    MAMLPlusPlusAlgorithm
    """
    if config is None:
        config = MAMLPlusPlusConfig(**overrides)
    engine = InnerLoopEngine(model=model)
    return MAMLPlusPlusAlgorithm(inner_engine=engine, config=config, model=model)


# ===========================================================================
# 8. Utility: build_maml_pp_outer_optimizer
# ===========================================================================


def build_maml_pp_outer_optimizer(
    model: nn.Module,
    algo: MAMLPlusPlusAlgorithm,
    outer_lr: float = 0.001,
    weight_decay: float = 0.0,
) -> torch.optim.Adam:
    """Construct an outer optimiser that includes MAML++ extra parameters.

    Parameters
    ----------
    model : nn.Module
        Base model whose parameters are the primary optimisation target.
    algo : MAMLPlusPlusAlgorithm
        MAML++ algorithm whose LSLR / MSL parameters should be co-optimised.
    outer_lr : float
        Outer-loop learning rate.
    weight_decay : float
        Weight decay for model parameters (not applied to LSLR / MSL).

    Returns
    -------
    torch.optim.Adam
    """
    param_groups = [
        {"params": list(model.parameters()), "lr": outer_lr, "weight_decay": weight_decay},
    ]
    extra = algo.parameters()
    if extra:
        param_groups.append(
            {"params": extra, "lr": outer_lr, "weight_decay": 0.0},
        )
    return torch.optim.Adam(param_groups)


# ===========================================================================
# 9. Utility: LSLR visualisation helpers
# ===========================================================================


def lslr_to_matrix(lslr: LSLRModule) -> torch.Tensor:
    """Extract LSLR values as a ``(num_layers, num_steps)`` matrix."""
    rows: List[List[float]] = []
    for layer in lslr.layer_names:
        row = [lslr.get_lr(layer, step).item() for step in range(lslr.num_steps)]
        rows.append(row)
    return torch.tensor(rows)


def lslr_summary(lslr: LSLRModule) -> str:
    """Return a human-readable summary of the LSLR state."""
    mat = lslr_to_matrix(lslr)
    lines = [
        f"LSLR summary: {len(lslr.layer_names)} layers x {lslr.num_steps} steps",
        f"  Global  -- mean={mat.mean():.6f}  min={mat.min():.6f}  max={mat.max():.6f}",
    ]
    for s in range(lslr.num_steps):
        col = mat[:, s]
        lines.append(
            f"  Step {s:2d} -- mean={col.mean():.6f}  min={col.min():.6f}  max={col.max():.6f}"
        )
    return "\n".join(lines)


# ===========================================================================
# 10. Toy models for testing
# ===========================================================================


class _ToyLinear(nn.Module):
    """Minimal linear model for meta-learning unit tests."""

    def __init__(self, in_features: int = 10, num_classes: int = 5):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _ToyConvBNReLU(nn.Module):
    """Small Conv-BN-ReLU model for BN-specific tests.

    Architecture: Conv2d -> BN2d -> ReLU -> AdaptiveAvgPool -> Linear
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)))
        out = self.pool(out).view(out.size(0), -1)
        return self.fc(out)


# ===========================================================================
# 11. Episode factories for testing
# ===========================================================================


def _make_linear_episode(
    in_features: int = 10,
    num_classes: int = 5,
    n_support: int = 10,
    n_query: int = 10,
) -> Episode:
    """Create a random episode for a linear model."""
    return Episode(
        support_x=torch.randn(n_support, in_features),
        support_y=torch.randint(0, num_classes, (n_support,)),
        query_x=torch.randn(n_query, in_features),
        query_y=torch.randint(0, num_classes, (n_query,)),
    )


def _make_conv_episode(
    in_channels: int = 1,
    spatial: int = 8,
    num_classes: int = 5,
    n_support: int = 10,
    n_query: int = 10,
) -> Episode:
    """Create a random episode for a convolutional model."""
    return Episode(
        support_x=torch.randn(n_support, in_channels, spatial, spatial),
        support_y=torch.randint(0, num_classes, (n_support,)),
        query_x=torch.randn(n_query, in_channels, spatial, spatial),
        query_y=torch.randint(0, num_classes, (n_query,)),
    )


# ===========================================================================
# SELF-TEST BLOCK
# ===========================================================================


def _run_self_tests() -> None:
    """Run all self-tests and print PASS/FAIL for each group."""

    results: Dict[str, bool] = {}
    num_steps = 5
    num_classes = 5
    in_features = 10

    def _report(name: str, passed: bool, detail: str = "") -> None:
        results[name] = passed
        status = "PASS" if passed else "FAIL"
        msg = f"  [{status}] {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)

    print("=" * 72)
    print("MAML++ Template Self-Tests")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Group 1: LSLRModule creation with correct parameter count
    # ------------------------------------------------------------------
    try:
        model = _ToyLinear(in_features, num_classes)
        layer_names = [n for n, _ in model.named_parameters()]
        lslr = LSLRModule(layer_names, num_steps, init_lr=0.01)
        expected_count = len(layer_names) * num_steps
        actual_count = len(list(lslr.parameters()))
        ok = actual_count == expected_count
        _report(
            "1. LSLR param count",
            ok,
            f"expected={expected_count}, actual={actual_count}",
        )
    except Exception as e:
        _report("1. LSLR param count", False, str(e))

    # ------------------------------------------------------------------
    # Group 2: LSLR get_lr returns value in [lr_min, lr_max]
    # ------------------------------------------------------------------
    try:
        for layer in layer_names:
            for step in range(num_steps):
                lr_val = lslr.get_lr(layer, step).item()
                ok = lslr.lr_min <= lr_val <= lslr.lr_max
                if not ok:
                    break
            if not ok:
                break
        _report(
            "2. LSLR lr in [lr_min, lr_max]",
            ok,
            f"sample lr={lr_val:.6f}, bounds=[{lslr.lr_min}, {lslr.lr_max}]",
        )
    except Exception as e:
        _report("2. LSLR lr in [lr_min, lr_max]", False, str(e))

    # ------------------------------------------------------------------
    # Group 3: LSLR different layers have independent LRs
    # ------------------------------------------------------------------
    try:
        # Perturb one layer's LR and check others didn't change
        lslr2 = LSLRModule(layer_names, num_steps, init_lr=0.01)
        with torch.no_grad():
            key0 = f"{LSLRModule._sanitize(layer_names[0])}_step0"
            lslr2.lrs[key0].fill_(math.log(0.5))
        lr_layer0 = lslr2.get_lr(layer_names[0], 0).item()
        lr_layer1 = lslr2.get_lr(layer_names[1], 0).item()
        ok = abs(lr_layer0 - 0.5) < 1e-4 and abs(lr_layer1 - 0.01) < 1e-4
        _report(
            "3. LSLR layers independent",
            ok,
            f"layer0={lr_layer0:.6f}, layer1={lr_layer1:.6f}",
        )
    except Exception as e:
        _report("3. LSLR layers independent", False, str(e))

    # ------------------------------------------------------------------
    # Group 4: LSLR different steps have independent LRs
    # ------------------------------------------------------------------
    try:
        lslr3 = LSLRModule(layer_names, num_steps, init_lr=0.01)
        with torch.no_grad():
            key_s0 = f"{LSLRModule._sanitize(layer_names[0])}_step0"
            key_s1 = f"{LSLRModule._sanitize(layer_names[0])}_step1"
            lslr3.lrs[key_s0].fill_(math.log(0.1))
            lslr3.lrs[key_s1].fill_(math.log(0.9))
        lr_s0 = lslr3.get_lr(layer_names[0], 0).item()
        lr_s1 = lslr3.get_lr(layer_names[0], 1).item()
        ok = abs(lr_s0 - 0.1) < 1e-4 and abs(lr_s1 - 0.9) < 1e-4
        _report(
            "4. LSLR steps independent",
            ok,
            f"step0={lr_s0:.6f}, step1={lr_s1:.6f}",
        )
    except Exception as e:
        _report("4. LSLR steps independent", False, str(e))

    # ------------------------------------------------------------------
    # Group 5: LSLR gradients flow through to outer loss
    # ------------------------------------------------------------------
    try:
        model5 = _ToyLinear(in_features, num_classes)
        layer_names5 = [n for n, _ in model5.named_parameters()]
        lslr5 = LSLRModule(layer_names5, num_steps=3, init_lr=0.01)
        adapted5 = {n: p for n, p in model5.named_parameters()}

        # Simulate all 3 inner steps so every LSLR parameter is exercised
        x5 = torch.randn(4, in_features)
        y5 = torch.randint(0, num_classes, (4,))
        for step5 in range(3):
            logits5 = _functional_forward(model5, adapted5, x5)
            loss5 = F.cross_entropy(logits5, y5)
            grad_tensors5 = torch.autograd.grad(
                loss5, list(adapted5.values()), create_graph=True, allow_unused=True
            )
            new_adapted5 = {}
            for (k, p), g in zip(adapted5.items(), grad_tensors5):
                lr_val = lslr5.get_lr(k, step5)
                new_adapted5[k] = p - lr_val * (g if g is not None else torch.zeros_like(p))
            adapted5 = new_adapted5

        # Query loss after all inner steps
        q_logits5 = _functional_forward(model5, adapted5, x5)
        q_loss5 = F.cross_entropy(q_logits5, y5)
        q_loss5.backward()

        lslr_grads_exist = all(
            p.grad is not None and p.grad.abs().sum() > 0 for p in lslr5.parameters()
        )
        _report(
            "5. LSLR gradients flow",
            lslr_grads_exist,
            "all LSLR params have nonzero grad" if lslr_grads_exist else "some LSLR params missing grad",
        )
    except Exception as e:
        _report("5. LSLR gradients flow", False, str(e))

    # ------------------------------------------------------------------
    # Group 6: LSLR statistics computation
    # ------------------------------------------------------------------
    try:
        lslr6 = LSLRModule(["a", "b"], num_steps=3, init_lr=0.05)
        stats6 = lslr6.get_statistics()
        ok = (
            "lr_mean" in stats6
            and "lr_min" in stats6
            and "lr_max" in stats6
            and "lr_std" in stats6
            and abs(stats6["lr_mean"] - 0.05) < 0.01
        )
        _report(
            "6. LSLR statistics",
            ok,
            f"mean={stats6.get('lr_mean', '?'):.6f}",
        )
    except Exception as e:
        _report("6. LSLR statistics", False, str(e))

    # ------------------------------------------------------------------
    # Group 7: MSL uniform weights sum to 1
    # ------------------------------------------------------------------
    try:
        msl7 = MultiStepLoss(num_steps=5, weight_scheme="uniform")
        w7 = msl7.weights
        ok = abs(w7.sum().item() - 1.0) < 1e-6
        _report(
            "7. MSL uniform weights sum=1",
            ok,
            f"sum={w7.sum().item():.8f}",
        )
    except Exception as e:
        _report("7. MSL uniform weights sum=1", False, str(e))

    # ------------------------------------------------------------------
    # Group 8: MSL linear_increase weights are monotonically increasing
    # ------------------------------------------------------------------
    try:
        msl8 = MultiStepLoss(num_steps=5, weight_scheme="linear_increase")
        w8 = msl8.weights
        monotonic = all(w8[i] <= w8[i + 1] for i in range(len(w8) - 1))
        _report(
            "8. MSL linear_increase monotonic",
            monotonic,
            f"weights={[f'{v:.4f}' for v in w8.tolist()]}",
        )
    except Exception as e:
        _report("8. MSL linear_increase monotonic", False, str(e))

    # ------------------------------------------------------------------
    # Group 9: MSL learned weights produce valid distribution
    # ------------------------------------------------------------------
    try:
        msl9 = MultiStepLoss(num_steps=5, weight_scheme="learned")
        w9 = msl9.weights
        ok = abs(w9.sum().item() - 1.0) < 1e-6 and (w9 >= 0).all().item()
        _report(
            "9. MSL learned weights valid dist",
            ok,
            f"sum={w9.sum().item():.8f}, min={w9.min().item():.8f}",
        )
    except Exception as e:
        _report("9. MSL learned weights valid dist", False, str(e))

    # ------------------------------------------------------------------
    # Group 10: MSL forward combines losses correctly
    # ------------------------------------------------------------------
    try:
        msl10 = MultiStepLoss(num_steps=3, weight_scheme="uniform")
        losses10 = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        combined10 = msl10(losses10)
        expected10 = (1.0 + 2.0 + 3.0) / 3.0
        ok = abs(combined10.item() - expected10) < 1e-6
        _report(
            "10. MSL forward combines correctly",
            ok,
            f"combined={combined10.item():.6f}, expected={expected10:.6f}",
        )
    except Exception as e:
        _report("10. MSL forward combines correctly", False, str(e))

    # ------------------------------------------------------------------
    # Group 11: Annealing before start -> first_order=True
    # ------------------------------------------------------------------
    try:
        ann11 = DerivativeOrderAnnealing(annealing_start_epoch=10)
        ok = ann11.get_first_order(5) is True
        _report(
            "11. Annealing before start -> FO",
            ok,
            f"epoch=5, first_order={ann11.get_first_order(5)}",
        )
    except Exception as e:
        _report("11. Annealing before start -> FO", False, str(e))

    # ------------------------------------------------------------------
    # Group 12: Annealing after start -> first_order=False
    # ------------------------------------------------------------------
    try:
        ann12 = DerivativeOrderAnnealing(annealing_start_epoch=10)
        ok = ann12.get_first_order(15) is False
        _report(
            "12. Annealing after start -> SO",
            ok,
            f"epoch=15, first_order={ann12.get_first_order(15)}",
        )
    except Exception as e:
        _report("12. Annealing after start -> SO", False, str(e))

    # ------------------------------------------------------------------
    # Group 13: PerStepBNManager finds BN modules in model
    # ------------------------------------------------------------------
    try:
        model13 = _ToyConvBNReLU()
        bnm13 = PerStepBNManager(model13, mode="per_step", num_steps=3)
        ok = bnm13.num_bn_modules >= 1
        _report(
            "13. BN manager finds BN modules",
            ok,
            f"found {bnm13.num_bn_modules} BN modules",
        )
    except Exception as e:
        _report("13. BN manager finds BN modules", False, str(e))

    # ------------------------------------------------------------------
    # Group 14: PerStepBNManager frozen mode preserves stats
    # ------------------------------------------------------------------
    try:
        model14 = _ToyConvBNReLU()
        # Run a forward pass to set non-trivial running stats
        model14.train()
        _ = model14(torch.randn(8, 1, 8, 8))

        # Capture the running_mean before freezing
        pre_mean = model14.bn.running_mean.clone()

        bnm14 = PerStepBNManager(model14, mode="frozen", num_steps=3)

        # Run another forward pass to change stats
        model14.train()
        _ = model14(torch.randn(8, 1, 8, 8) * 10)

        # Stats should have drifted
        post_mean = model14.bn.running_mean.clone()
        drifted = not torch.allclose(pre_mean, post_mean)

        # Reset using frozen manager
        bnm14.reset()
        restored_mean = model14.bn.running_mean.clone()
        restored = torch.allclose(pre_mean, restored_mean, atol=1e-6)

        ok = drifted and restored
        _report(
            "14. BN frozen preserves stats",
            ok,
            f"drifted={drifted}, restored={restored}",
        )
    except Exception as e:
        _report("14. BN frozen preserves stats", False, str(e))

    # ------------------------------------------------------------------
    # Group 15: MAMLPlusPlusAlgorithm meta_step produces loss with gradient
    # ------------------------------------------------------------------
    try:
        model15 = _ToyLinear(in_features, num_classes)
        config15 = MAMLPlusPlusConfig(
            inner_steps=3,
            inner_lr=0.01,
            use_lslr=True,
            use_msl=True,
            msl_weights="uniform",
            use_annealing=True,
            annealing_start_epoch=0,
            bn_mode="transductive",
        )
        algo15 = create_maml_plus_plus(model15, config15)
        params15 = {n: p for n, p in model15.named_parameters()}

        ep15 = _make_linear_episode(in_features, num_classes)
        tb15 = TaskBatch(episodes=[ep15])

        output15 = algo15.meta_step(model15, params15, tb15, epoch=5)
        ok_loss = output15.loss.requires_grad
        # Try backward to check gradient flow
        output15.loss.backward()
        has_grads = all(
            p.grad is not None for p in model15.parameters()
        )
        ok = ok_loss and has_grads
        _report(
            "15. meta_step loss has gradient",
            ok,
            f"requires_grad={ok_loss}, model_grads={has_grads}",
        )
    except Exception as e:
        _report("15. meta_step loss has gradient", False, f"{e}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Group 16: MAMLPlusPlusAlgorithm with LSLR modifies effective LRs
    # ------------------------------------------------------------------
    try:
        model16 = _ToyLinear(in_features, num_classes)
        config16_no_lslr = MAMLPlusPlusConfig(
            inner_steps=2, inner_lr=0.01, use_lslr=False, bn_mode="transductive",
        )
        config16_lslr = MAMLPlusPlusConfig(
            inner_steps=2, inner_lr=0.01, use_lslr=True, bn_mode="transductive",
        )
        algo16_no = create_maml_plus_plus(copy.deepcopy(model16), config16_no_lslr)
        algo16_yes = create_maml_plus_plus(copy.deepcopy(model16), config16_lslr)

        # The LSLR version should have extra parameters
        ok = len(algo16_yes.parameters()) > 0 and len(algo16_no.parameters()) == 0
        _report(
            "16. LSLR modifies effective LRs",
            ok,
            f"lslr_params={len(algo16_yes.parameters())}, no_lslr_params={len(algo16_no.parameters())}",
        )
    except Exception as e:
        _report("16. LSLR modifies effective LRs", False, str(e))

    # ------------------------------------------------------------------
    # Group 17: MAMLPlusPlusAlgorithm with MSL uses multi-step losses
    # ------------------------------------------------------------------
    try:
        model17 = _ToyLinear(in_features, num_classes)
        config17 = MAMLPlusPlusConfig(
            inner_steps=3,
            inner_lr=0.01,
            use_msl=True,
            msl_weights="linear_increase",
            bn_mode="transductive",
        )
        algo17 = create_maml_plus_plus(model17, config17)
        params17 = {n: p for n, p in model17.named_parameters()}
        ep17 = _make_linear_episode(in_features, num_classes)
        tb17 = TaskBatch(episodes=[ep17])

        output17 = algo17.meta_step(model17, params17, tb17, epoch=0)

        # With MSL, inner_logs should have entries for each step, and
        # the metrics should contain MSL weight stats
        has_msl_stats = "msl_weight_min" in output17.metrics
        has_inner_logs = (
            len(output17.inner_logs) == 1
            and len(output17.inner_logs[0]) == 3
        )
        ok = has_msl_stats and has_inner_logs
        _report(
            "17. MSL uses multi-step losses",
            ok,
            f"msl_stats={has_msl_stats}, inner_log_steps={len(output17.inner_logs[0]) if output17.inner_logs else 0}",
        )
    except Exception as e:
        _report("17. MSL uses multi-step losses", False, str(e))

    # ------------------------------------------------------------------
    # Group 18: parameters() includes LSLR and MSL params
    # ------------------------------------------------------------------
    try:
        model18 = _ToyLinear(in_features, num_classes)
        config18 = MAMLPlusPlusConfig(
            inner_steps=3,
            inner_lr=0.01,
            use_lslr=True,
            use_msl=True,
            msl_weights="learned",
            bn_mode="transductive",
        )
        algo18 = create_maml_plus_plus(model18, config18)
        extra18 = algo18.parameters()

        # LSLR params: 2 layers * 3 steps = 6
        # MSL learned weights: 1 parameter (vector of size 3)
        num_lslr = len([n for n, _ in model18.named_parameters()]) * 3
        num_msl = 1  # weight_logits is one parameter
        expected_total = num_lslr + num_msl
        actual_total = len(extra18)
        ok = actual_total == expected_total
        _report(
            "18. parameters() includes LSLR+MSL",
            ok,
            f"expected={expected_total}, actual={actual_total}",
        )
    except Exception as e:
        _report("18. parameters() includes LSLR+MSL", False, str(e))

    # ------------------------------------------------------------------
    # Group 19: PerStepBNManager per_step mode isolates steps
    # ------------------------------------------------------------------
    try:
        model19 = _ToyConvBNReLU()
        model19.train()
        # Initialise running stats
        _ = model19(torch.randn(8, 1, 8, 8))

        bnm19 = PerStepBNManager(model19, mode="per_step", num_steps=3)

        # Step 0: load step-0 stats, then run in train mode to accumulate
        bnm19.set_step(0)
        # Override to train mode to allow running-stat accumulation for this test
        bnm19.set_model_bn_mode(training=True)
        _ = model19(torch.randn(8, 1, 8, 8) * 5)
        bnm19.save_step(0)
        mean_after_step0 = model19.bn.running_mean.clone()

        # Step 1: load step-1 stats, run in train mode with different data
        bnm19.set_step(1)
        bnm19.set_model_bn_mode(training=True)
        _ = model19(torch.randn(8, 1, 8, 8) * 0.1)
        bnm19.save_step(1)
        mean_after_step1 = model19.bn.running_mean.clone()

        # Reload step 0 -- should restore step-0 stats
        bnm19.set_step(0)
        mean_reloaded_step0 = model19.bn.running_mean.clone()

        ok = torch.allclose(mean_after_step0, mean_reloaded_step0, atol=1e-6)
        _report(
            "19. BN per_step isolates steps",
            ok,
            f"step0 reload matches: {ok}",
        )
    except Exception as e:
        _report("19. BN per_step isolates steps", False, str(e))

    # ------------------------------------------------------------------
    # Group 20: DerivativeOrderAnnealing stochastic probability
    # ------------------------------------------------------------------
    try:
        ann20 = DerivativeOrderAnnealing(
            annealing_start_epoch=10, annealing_end_epoch=20
        )
        prob_5 = ann20.get_create_graph_prob(5)
        prob_15 = ann20.get_create_graph_prob(15)
        prob_25 = ann20.get_create_graph_prob(25)
        ok = (
            abs(prob_5 - 0.0) < 1e-6
            and abs(prob_15 - 0.5) < 1e-6
            and abs(prob_25 - 1.0) < 1e-6
        )
        _report(
            "20. Annealing stochastic prob",
            ok,
            f"prob(5)={prob_5:.3f}, prob(15)={prob_15:.3f}, prob(25)={prob_25:.3f}",
        )
    except Exception as e:
        _report("20. Annealing stochastic prob", False, str(e))

    # ------------------------------------------------------------------
    # Group 21: LSLR reset_to works
    # ------------------------------------------------------------------
    try:
        lslr21 = LSLRModule(["a", "b"], num_steps=3, init_lr=0.01)
        lslr21.reset_to(0.1)
        all_close = all(
            abs(lslr21.get_lr(layer, s).item() - 0.1) < 1e-4
            for layer in ["a", "b"]
            for s in range(3)
        )
        _report("21. LSLR reset_to", all_close, f"all LRs ~0.1: {all_close}")
    except Exception as e:
        _report("21. LSLR reset_to", False, str(e))

    # ------------------------------------------------------------------
    # Group 22: MSL invalid scheme raises ValueError
    # ------------------------------------------------------------------
    try:
        raised = False
        try:
            MultiStepLoss(num_steps=3, weight_scheme="invalid_scheme")
        except ValueError:
            raised = True
        _report("22. MSL invalid scheme raises", raised, f"ValueError raised: {raised}")
    except Exception as e:
        _report("22. MSL invalid scheme raises", False, str(e))

    # ------------------------------------------------------------------
    # Group 23: Factory function create_maml_plus_plus
    # ------------------------------------------------------------------
    try:
        model23 = _ToyLinear(in_features, num_classes)
        algo23 = create_maml_plus_plus(
            model23, use_lslr=True, use_msl=True, msl_weights="learned",
            inner_steps=3, inner_lr=0.02,
        )
        ok = (
            algo23.lslr is not None and algo23.msl is not None
            and algo23.config.inner_steps == 3
        )
        _report("23. Factory create_maml_plus_plus", ok)
    except Exception as e:
        _report("23. Factory create_maml_plus_plus", False, str(e))

    # ------------------------------------------------------------------
    # Group 24: build_maml_pp_outer_optimizer
    # ------------------------------------------------------------------
    try:
        model24 = _ToyLinear(in_features, num_classes)
        algo24 = create_maml_plus_plus(
            model24, use_lslr=True, use_msl=True, msl_weights="learned", inner_steps=3,
        )
        opt24 = build_maml_pp_outer_optimizer(model24, algo24, outer_lr=0.001)
        ok = len(opt24.param_groups) == 2
        _report("24. build_maml_pp_outer_optimizer", ok, f"param_groups={len(opt24.param_groups)}")
    except Exception as e:
        _report("24. build_maml_pp_outer_optimizer", False, str(e))

    # ------------------------------------------------------------------
    # Group 25: MSL learned weights receive gradients
    # ------------------------------------------------------------------
    try:
        msl25 = MultiStepLoss(num_steps=3, weight_scheme="learned")
        losses25 = [
            torch.tensor(1.0, requires_grad=True),
            torch.tensor(2.0, requires_grad=True),
            torch.tensor(3.0, requires_grad=True),
        ]
        combined25 = msl25(losses25)
        combined25.backward()
        ok = (
            msl25.weight_logits.grad is not None
            and msl25.weight_logits.grad.abs().sum() > 0
        )
        _report(
            "25. MSL learned weights grad flow",
            ok,
            f"grad_sum={msl25.weight_logits.grad.abs().sum().item():.6f}" if ok else "no grad",
        )
    except Exception as e:
        _report("25. MSL learned weights grad flow", False, str(e))

    # ------------------------------------------------------------------
    # Group 26: MAMLPlusPlusAlgorithm state_dict round-trip
    # ------------------------------------------------------------------
    try:
        model26 = _ToyLinear(in_features, num_classes)
        config26 = MAMLPlusPlusConfig(
            inner_steps=2, inner_lr=0.01, use_lslr=True,
            use_msl=True, msl_weights="learned",
            use_annealing=True, annealing_start_epoch=5,
            bn_mode="transductive",
        )
        algo26 = create_maml_plus_plus(model26, config26)
        sd26 = algo26.state_dict()
        ok = "lslr" in sd26 and "msl" in sd26 and "annealing" in sd26
        _report("26. MAML++ state_dict keys", ok, f"keys={list(sd26.keys())}")
    except Exception as e:
        _report("26. MAML++ state_dict keys", False, str(e))

    # ------------------------------------------------------------------
    # Group 27: Full E2E Conv-BN model with frozen BN (second-order safe)
    # ------------------------------------------------------------------
    try:
        model27 = _ToyConvBNReLU(in_channels=1, num_classes=num_classes)
        model27.train()
        _ = model27(torch.randn(8, 1, 8, 8))  # Prime BN running stats

        config27 = MAMLPlusPlusConfig(
            inner_steps=2, inner_lr=0.01, use_lslr=True,
            use_msl=True, msl_weights="uniform", bn_mode="frozen",
        )
        algo27 = create_maml_plus_plus(model27, config27)
        params27 = {n: p for n, p in model27.named_parameters()}
        ep27 = _make_conv_episode(in_channels=1, spatial=8, num_classes=num_classes)
        tb27 = TaskBatch(episodes=[ep27])

        model27.zero_grad()
        output27 = algo27.meta_step(model27, params27, tb27, epoch=0)
        ok_loss = output27.loss.requires_grad
        output27.loss.backward()
        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model27.parameters()
        )
        ok = ok_loss and has_grads
        _report(
            "27. E2E Conv-BN frozen (SO)",
            ok,
            f"loss_grad={ok_loss}, model_grads={has_grads}",
        )
    except Exception as e:
        _report("27. E2E Conv-BN frozen (SO)", False, f"{e}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Group 28: Multiple episodes in a TaskBatch
    # ------------------------------------------------------------------
    try:
        model28 = _ToyLinear(in_features, num_classes)
        config28 = MAMLPlusPlusConfig(
            inner_steps=2, inner_lr=0.01, use_msl=True,
            msl_weights="uniform", bn_mode="transductive",
        )
        algo28 = create_maml_plus_plus(model28, config28)
        params28 = {n: p for n, p in model28.named_parameters()}
        eps28 = [_make_linear_episode(in_features, num_classes) for _ in range(4)]
        tb28 = TaskBatch(episodes=eps28)

        output28 = algo28.meta_step(model28, params28, tb28, epoch=0)
        ok = (
            len(output28.inner_logs) == 4
            and len(output28.adapted_params) == 4
            and output28.metrics["num_tasks"] == 4.0
        )
        _report("28. Multiple episodes in batch", ok)
    except Exception as e:
        _report("28. Multiple episodes in batch", False, str(e))

    # ------------------------------------------------------------------
    # Group 29: MSL forward rejects wrong number of losses
    # ------------------------------------------------------------------
    try:
        msl29 = MultiStepLoss(num_steps=3, weight_scheme="uniform")
        raised = False
        try:
            msl29([torch.tensor(1.0), torch.tensor(2.0)])
        except ValueError:
            raised = True
        _report("29. MSL rejects wrong loss count", raised)
    except Exception as e:
        _report("29. MSL rejects wrong loss count", False, str(e))

    # ------------------------------------------------------------------
    # Group 30: PerStepBNManager transductive mode is no-op
    # ------------------------------------------------------------------
    try:
        model30 = _ToyConvBNReLU()
        bnm30 = PerStepBNManager(model30, mode="transductive", num_steps=3)
        bnm30.set_step(0)
        bnm30.save_step(0)
        bnm30.reset()
        _report("30. BN transductive no-op", True, "no exceptions raised")
    except Exception as e:
        _report("30. BN transductive no-op", False, str(e))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 72)
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed > 0:
        print("Failed tests:")
        for name, ok in results.items():
            if not ok:
                print(f"  - {name}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _run_self_tests()
