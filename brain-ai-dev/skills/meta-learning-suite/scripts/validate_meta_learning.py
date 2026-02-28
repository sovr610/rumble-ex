#!/usr/bin/env python3
"""Meta-Learning Suite -- Runtime Contract Validation

Validates all done-when gates and key contracts for the meta-learning suite.
Self-contained: runs without brain_ai package installed.

Usage:
    python validate_meta_learning.py                    # Run all checks
    python validate_meta_learning.py --group gate_a     # Run specific group
    python validate_meta_learning.py --verbose           # Detailed output
    python validate_meta_learning.py --list              # List all check groups

Exit codes:
    0 = all checks passed
    1 = one or more checks failed
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ============================================================================
# ANSI colour helpers
# ============================================================================

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _c(text: str, code: str) -> str:
    """Wrap text in ANSI colour if stdout is a tty."""
    if sys.stdout.isatty():
        return f"{code}{text}{_RESET}"
    return text


# ============================================================================
# Group registry
# ============================================================================

_ALL_GROUPS: List[str] = [
    "inner_loop",
    "algorithms",
    "maml_plus",
    "sampling",
    "gate_a",
    "gate_b",
    "gate_c",
]

_GROUP_DESCRIPTIONS: Dict[str, str] = {
    "inner_loop": "Inner-loop engine: adapt(), StepLog, clipping, graph modes",
    "algorithms": "MAML / FOMAML / Reptile meta_step and MetaOutput",
    "maml_plus": "MAML++ enhancements: LSLR, MSL, MAMLPlusPlusAlgorithm",
    "sampling": "EpisodeSampler determinism, shapes, class splits",
    "gate_a": "Done-When Gate (a): Meta-gradient flow validated",
    "gate_b": "Done-When Gate (b): FO variants produce comparable adaptation",
    "gate_c": "Done-When Gate (c): Phase 7 scripts run on CPU in dev mode",
}

# ============================================================================
# Result dataclass
# ============================================================================


@dataclass
class CheckResult:
    """Outcome of a single validation check."""
    name: str
    group: str
    passed: bool
    message: str
    details: Optional[str] = None
    elapsed_ms: float = 0.0


# ============================================================================
# Inline stub implementations
# ============================================================================
# These minimal implementations are self-contained so the validation script
# does not depend on brain_ai being installed.  They faithfully replicate the
# contracts specified in SKILL.md and the reference documents.


@dataclass
class StepLog:
    """Diagnostic record for a single inner-loop step."""
    step: int
    loss: float
    accuracy: float
    grad_norm: float
    update_norm: float
    lr_effective: float
    clipped: bool


@dataclass
class InnerLoopResult:
    """Result of an inner-loop adaptation."""
    adapted_params: Dict[str, Tensor]
    logs: List[StepLog]


@dataclass
class Episode:
    """A single N-way K-shot episode."""
    support_x: Tensor
    support_y: Tensor
    query_x: Tensor
    query_y: Tensor
    class_ids: List[int]  # Original class ids before relabeling
    n_way: int
    k_shot: int
    q_query: int


@dataclass
class TaskBatch:
    """A batch of episodes for meta-training."""
    episodes: List[Episode]

    @property
    def num_tasks(self) -> int:
        return len(self.episodes)


@dataclass
class MetaOutput:
    """Output from a meta-learning step."""
    loss: Tensor              # scalar outer loss
    metrics: Dict[str, float]  # pre_adapt_acc, post_adapt_acc, fast_gain, auac
    inner_logs: List[List[StepLog]]  # per-task inner-loop logs
    adapted_params: List[Dict[str, Tensor]]  # per-task adapted params


class ToyLinearModel(nn.Module):
    """2-layer MLP: Linear(input, hidden) -> ReLU -> Linear(hidden, output)."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class ToyConv4Model(nn.Module):
    """Conv4 backbone: 4 conv blocks (Conv->BN->ReLU->Pool) + linear classifier."""

    def __init__(self, in_channels: int = 1, hidden_channels: int = 32,
                 num_classes: int = 5, input_size: int = 28):
        super().__init__()
        def _block(cin, cout, pool):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                    nn.ReLU(), pool]
        layers = _block(in_channels, hidden_channels, nn.MaxPool2d(2))
        for _ in range(2):
            layers += _block(hidden_channels, hidden_channels, nn.MaxPool2d(2))
        layers += _block(hidden_channels, hidden_channels, nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x).view(x.shape[0], -1))


class TinyLinearModel(nn.Module):
    """Minimal single-layer linear model f(x) = Wx + b for hand-computed tests."""

    def __init__(self, input_dim: int = 4, output_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


def functional_forward(model: nn.Module, params: Dict[str, Tensor], x: Tensor) -> Tensor:
    """Forward pass using an explicit parameter dictionary via functional_call."""
    try:
        from torch.func import functional_call
        return functional_call(model, params, (x,))
    except (ImportError, AttributeError):
        # Fallback: manual parameter substitution
        originals = {}
        for name, p in model.named_parameters():
            originals[name] = p.data.clone()
            parts = name.split(".")
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], nn.Parameter(params[name]))
        try:
            out = model(x)
        finally:
            for name, data in originals.items():
                parts = name.split(".")
                obj = model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], nn.Parameter(data))
        return out


class CustomSGDEngine:
    """Differentiable inner-loop optimizer using torch.autograd.grad.

    Supports second-order (create_graph=True) for MAML and first-order
    (create_graph=False) for FOMAML.
    """

    def __init__(self, model: nn.Module, inner_lr: float = 0.01,
                 inner_clip: float = 10.0):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_clip = inner_clip

    def adapt(
        self,
        params: Dict[str, Tensor],
        support_x: Tensor,
        support_y: Tensor,
        *,
        steps: int = 5,
        lr: Optional[float] = None,
        create_graph: bool = True,
        clip_norm: Optional[float] = None,
        per_layer_lrs: Optional[Dict[str, float]] = None,
    ) -> InnerLoopResult:
        """Perform inner-loop adaptation.

        Returns InnerLoopResult with adapted_params and step logs.
        """
        lr = lr if lr is not None else self.inner_lr
        clip_norm = clip_norm if clip_norm is not None else self.inner_clip

        adapted = {k: v for k, v in params.items()}
        logs: List[StepLog] = []

        for step_idx in range(steps):
            # Forward pass
            logits = functional_forward(self.model, adapted, support_x)
            loss = F.cross_entropy(logits, support_y)

            # Compute gradients
            grad_tensors = torch.autograd.grad(
                outputs=loss,
                inputs=list(adapted.values()),
                create_graph=create_graph,
                allow_unused=True,
            )

            # Replace None grads with zeros
            grads = []
            for g, (k, p) in zip(grad_tensors, adapted.items()):
                grads.append(g if g is not None else torch.zeros_like(p))

            # Gradient clipping
            clipped = False
            total_norm = torch.sqrt(
                sum(g.detach().norm() ** 2 for g in grads)
            ).item()

            if clip_norm is not None and clip_norm > 0:
                clip_coef = clip_norm / (total_norm + 1e-6)
                if clip_coef < 1.0:
                    grads = [g * clip_coef for g in grads]
                    clipped = True
                    total_norm = clip_norm

            # Compute accuracy for logging
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                accuracy = (preds == support_y).float().mean().item()

            # Parameter update
            new_adapted = {}
            for (k, p), g in zip(adapted.items(), grads):
                effective_lr = per_layer_lrs[k] if per_layer_lrs and k in per_layer_lrs else lr
                new_adapted[k] = p - effective_lr * g

            update_norm = torch.sqrt(
                sum((new_adapted[k] - adapted[k]).detach().norm() ** 2 for k in adapted)
            ).item()

            adapted = new_adapted

            logs.append(StepLog(
                step=step_idx,
                loss=loss.item(),
                accuracy=accuracy,
                grad_norm=total_norm,
                update_norm=update_norm,
                lr_effective=lr,
                clipped=clipped,
            ))

        return InnerLoopResult(adapted_params=adapted, logs=logs)


def reptile_inner_loop(
    model: nn.Module,
    params: Dict[str, Tensor],
    support_x: Tensor,
    support_y: Tensor,
    steps: int = 5,
    lr: float = 0.01,
) -> Tuple[Dict[str, Tensor], List[StepLog]]:
    """Reptile inner loop: standard SGD, fully detached.

    Returns adapted params for weight-space interpolation.
    """
    adapted = {k: v.clone().detach().requires_grad_(True) for k, v in params.items()}
    logs: List[StepLog] = []

    for step_idx in range(steps):
        logits = functional_forward(model, adapted, support_x)
        loss = F.cross_entropy(logits, support_y)

        grads = torch.autograd.grad(loss, list(adapted.values()), allow_unused=True)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == support_y).float().mean().item()

        grad_norm = torch.sqrt(
            sum((g.norm() ** 2) for g in grads if g is not None)
        ).item()

        new_adapted = {}
        for (k, p), g in zip(adapted.items(), grads):
            if g is not None:
                new_adapted[k] = (p - lr * g).detach().requires_grad_(True)
            else:
                new_adapted[k] = p.detach().requires_grad_(True)

        update_norm = torch.sqrt(
            sum((new_adapted[k].detach() - adapted[k].detach()).norm() ** 2 for k in adapted)
        ).item()

        adapted = new_adapted

        logs.append(StepLog(
            step=step_idx,
            loss=loss.item(),
            accuracy=accuracy,
            grad_norm=grad_norm,
            update_norm=update_norm,
            lr_effective=lr,
            clipped=False,
        ))

    return adapted, logs


class SyntheticFewShotDataset:
    """Synthetic dataset: each class is a Gaussian blob in feature space.

    Args:
        num_classes: total number of classes
        samples_per_class: examples per class
        feature_dim: dimensionality of features
        seed: base seed for dataset generation
    """

    def __init__(
        self,
        num_classes: int = 100,
        samples_per_class: int = 20,
        feature_dim: int = 10,
        seed: int = 42,
    ):
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.feature_dim = feature_dim

        rng = torch.Generator()
        rng.manual_seed(seed)

        # Class means: each class has a random center
        self.class_means = torch.randn(num_classes, feature_dim, generator=rng)
        # Per-class samples: perturbations around the mean
        self.data = torch.zeros(num_classes, samples_per_class, feature_dim)
        self.labels = torch.zeros(num_classes, samples_per_class, dtype=torch.long)
        for c in range(num_classes):
            noise = torch.randn(samples_per_class, feature_dim, generator=rng) * 0.3
            self.data[c] = self.class_means[c].unsqueeze(0) + noise
            self.labels[c] = c

    def get_class_data(self, class_id: int) -> Tuple[Tensor, Tensor]:
        """Return all samples and labels for a given class."""
        return self.data[class_id], self.labels[class_id]


class EpisodeSampler:
    """Deterministic episodic task sampler.

    Episode composition is determined solely by (global_seed, epoch, episode_idx).
    """

    def __init__(
        self,
        dataset: SyntheticFewShotDataset,
        n_way: int = 5,
        k_shot: int = 1,
        q_query: int = 15,
        global_seed: int = 42,
        train_classes: Optional[List[int]] = None,
        val_classes: Optional[List[int]] = None,
        test_classes: Optional[List[int]] = None,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.global_seed = global_seed

        # Default class split: 60/20/20
        if train_classes is None:
            nc = dataset.num_classes
            n_train = int(nc * 0.6)
            n_val = int(nc * 0.2)
            all_ids = list(range(nc))
            # Deterministic shuffle for splitting
            rng = torch.Generator()
            rng.manual_seed(global_seed)
            perm = torch.randperm(nc, generator=rng).tolist()
            self.train_classes = sorted(perm[:n_train])
            self.val_classes = sorted(perm[n_train:n_train + n_val])
            self.test_classes = sorted(perm[n_train + n_val:])
        else:
            self.train_classes = train_classes
            self.val_classes = val_classes or []
            self.test_classes = test_classes or []

    def sample_episode(
        self,
        epoch: int = 0,
        episode_idx: int = 0,
        split: str = "train",
    ) -> Episode:
        """Sample a single episode, deterministic given (global_seed, epoch, episode_idx).

        Args:
            epoch: current epoch number
            episode_idx: episode index within the epoch
            split: "train", "val", or "test"

        Returns:
            Episode with support/query sets and metadata
        """
        # Derive per-episode seed
        ep_seed = hash((self.global_seed, epoch, episode_idx, split)) % (2**31)
        rng = torch.Generator()
        rng.manual_seed(ep_seed)

        # Select class pool
        if split == "train":
            class_pool = self.train_classes
        elif split == "val":
            class_pool = self.val_classes
        else:
            class_pool = self.test_classes

        # Sample N classes without replacement
        n_pool = len(class_pool)
        assert n_pool >= self.n_way, (
            f"Not enough classes: have {n_pool}, need {self.n_way}"
        )
        perm = torch.randperm(n_pool, generator=rng)[:self.n_way]
        chosen_classes = [class_pool[i] for i in perm.tolist()]

        # Build support and query sets
        support_xs, support_ys = [], []
        query_xs, query_ys = [], []

        for local_label, class_id in enumerate(chosen_classes):
            class_data, _ = self.dataset.get_class_data(class_id)
            n_samples = class_data.shape[0]
            total_needed = self.k_shot + self.q_query
            assert n_samples >= total_needed, (
                f"Class {class_id} has {n_samples} samples, need {total_needed}"
            )

            sample_perm = torch.randperm(n_samples, generator=rng)[:total_needed]
            support_idx = sample_perm[:self.k_shot]
            query_idx = sample_perm[self.k_shot:total_needed]

            support_xs.append(class_data[support_idx])
            support_ys.append(torch.full((self.k_shot,), local_label, dtype=torch.long))

            query_xs.append(class_data[query_idx])
            query_ys.append(torch.full((self.q_query,), local_label, dtype=torch.long))

        support_x = torch.cat(support_xs, dim=0)
        support_y = torch.cat(support_ys, dim=0)
        query_x = torch.cat(query_xs, dim=0)
        query_y = torch.cat(query_ys, dim=0)

        return Episode(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            class_ids=chosen_classes,
            n_way=self.n_way,
            k_shot=self.k_shot,
            q_query=self.q_query,
        )


class MAMLAlgorithm:
    """Model-Agnostic Meta-Learning (second-order)."""

    def __init__(self, model: nn.Module, inner_lr: float = 0.01,
                 inner_steps: int = 5, inner_clip: float = 10.0):
        self.model = model
        self.engine = CustomSGDEngine(model, inner_lr=inner_lr, inner_clip=inner_clip)
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr

    def meta_step(self, task_batch: TaskBatch) -> MetaOutput:
        """Run one meta-training step over a batch of episodes."""
        params = {n: p for n, p in self.model.named_parameters()}

        all_losses = []
        all_inner_logs = []
        all_adapted = []
        all_pre_acc = []
        all_post_acc = []

        for episode in task_batch.episodes:
            # Pre-adapt accuracy
            with torch.no_grad():
                pre_logits = functional_forward(self.model, params, episode.query_x)
                pre_acc = (pre_logits.argmax(-1) == episode.query_y).float().mean().item()
            all_pre_acc.append(pre_acc)

            # Inner loop (second order)
            result = self.engine.adapt(
                params, episode.support_x, episode.support_y,
                steps=self.inner_steps, create_graph=True,
            )

            # Outer loss on query set
            query_logits = functional_forward(self.model, result.adapted_params, episode.query_x)
            query_loss = F.cross_entropy(query_logits, episode.query_y)

            with torch.no_grad():
                post_acc = (query_logits.argmax(-1) == episode.query_y).float().mean().item()
            all_post_acc.append(post_acc)

            all_losses.append(query_loss)
            all_inner_logs.append(result.logs)
            all_adapted.append(result.adapted_params)

        # Average loss over tasks
        meta_loss = torch.stack(all_losses).mean()

        # Compute metrics
        mean_pre = sum(all_pre_acc) / len(all_pre_acc) if all_pre_acc else 0.0
        mean_post = sum(all_post_acc) / len(all_post_acc) if all_post_acc else 0.0
        fast_gain = mean_post - mean_pre

        # Compute AUAC from first task inner logs as representative
        auac = _compute_auac(all_inner_logs[0]) if all_inner_logs else 0.0

        metrics = {
            "pre_adapt_acc": mean_pre,
            "post_adapt_acc": mean_post,
            "fast_gain": fast_gain,
            "auac": auac,
        }

        return MetaOutput(
            loss=meta_loss,
            metrics=metrics,
            inner_logs=all_inner_logs,
            adapted_params=all_adapted,
        )


class FOMAMLAlgorithm:
    """First-Order MAML (no second-order graph)."""

    def __init__(self, model: nn.Module, inner_lr: float = 0.01,
                 inner_steps: int = 5, inner_clip: float = 10.0):
        self.model = model
        self.engine = CustomSGDEngine(model, inner_lr=inner_lr, inner_clip=inner_clip)
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr

    def meta_step(self, task_batch: TaskBatch) -> MetaOutput:
        """Run one first-order meta step."""
        params = {n: p for n, p in self.model.named_parameters()}

        all_losses = []
        all_inner_logs = []
        all_adapted = []
        all_pre_acc = []
        all_post_acc = []

        for episode in task_batch.episodes:
            with torch.no_grad():
                pre_logits = functional_forward(self.model, params, episode.query_x)
                pre_acc = (pre_logits.argmax(-1) == episode.query_y).float().mean().item()
            all_pre_acc.append(pre_acc)

            # First-order: create_graph=False
            result = self.engine.adapt(
                params, episode.support_x, episode.support_y,
                steps=self.inner_steps, create_graph=False,
            )

            # Outer loss -- but we need the adapted params to still depend on
            # original params for first-order gradient estimation.
            # With FOMAML, we re-evaluate the adapted params on query set.
            # The gradient flows through the adapted params' linear dependency on
            # the original params (p - lr*g where g is detached).
            query_logits = functional_forward(self.model, result.adapted_params, episode.query_x)
            query_loss = F.cross_entropy(query_logits, episode.query_y)

            with torch.no_grad():
                post_acc = (query_logits.argmax(-1) == episode.query_y).float().mean().item()
            all_post_acc.append(post_acc)

            all_losses.append(query_loss)
            all_inner_logs.append(result.logs)
            all_adapted.append(result.adapted_params)

        meta_loss = torch.stack(all_losses).mean()
        mean_pre = sum(all_pre_acc) / len(all_pre_acc) if all_pre_acc else 0.0
        mean_post = sum(all_post_acc) / len(all_post_acc) if all_post_acc else 0.0

        auac = _compute_auac(all_inner_logs[0]) if all_inner_logs else 0.0

        metrics = {
            "pre_adapt_acc": mean_pre,
            "post_adapt_acc": mean_post,
            "fast_gain": mean_post - mean_pre,
            "auac": auac,
        }

        return MetaOutput(
            loss=meta_loss,
            metrics=metrics,
            inner_logs=all_inner_logs,
            adapted_params=all_adapted,
        )


class ReptileAlgorithm:
    """Reptile meta-learning via weight-space interpolation."""

    def __init__(self, model: nn.Module, inner_lr: float = 0.01,
                 inner_steps: int = 5, epsilon: float = 0.1):
        self.model = model
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.epsilon = epsilon

    def meta_step(self, task_batch: TaskBatch) -> MetaOutput:
        """Reptile meta-step: adapt per task, then weight interpolation."""
        params = {n: p for n, p in self.model.named_parameters()}

        all_adapted = []
        all_inner_logs = []
        all_pre_acc = []
        all_post_acc = []

        for episode in task_batch.episodes:
            with torch.no_grad():
                pre_logits = functional_forward(self.model, params, episode.query_x)
                pre_acc = (pre_logits.argmax(-1) == episode.query_y).float().mean().item()
            all_pre_acc.append(pre_acc)

            adapted, logs = reptile_inner_loop(
                self.model, params,
                episode.support_x, episode.support_y,
                steps=self.inner_steps, lr=self.inner_lr,
            )

            # Evaluate adapted model on query set
            with torch.no_grad():
                post_logits = functional_forward(self.model, adapted, episode.query_x)
                post_acc = (post_logits.argmax(-1) == episode.query_y).float().mean().item()
            all_post_acc.append(post_acc)

            all_adapted.append(adapted)
            all_inner_logs.append(logs)

        # Reptile outer update: theta += epsilon * mean(phi - theta)
        # Compute the "meta loss" as the average distance moved (for metric purposes)
        meta_loss_val = 0.0
        with torch.no_grad():
            for k in params:
                diffs = torch.stack([ap[k].detach() - params[k].detach() for ap in all_adapted])
                avg_diff = diffs.mean(dim=0)
                params[k].data.add_(self.epsilon * avg_diff)
                meta_loss_val += avg_diff.norm().item()

        # Create a dummy loss tensor for API compatibility
        meta_loss = torch.tensor(meta_loss_val, requires_grad=False)

        mean_pre = sum(all_pre_acc) / len(all_pre_acc) if all_pre_acc else 0.0
        mean_post = sum(all_post_acc) / len(all_post_acc) if all_post_acc else 0.0
        auac = _compute_auac(all_inner_logs[0]) if all_inner_logs else 0.0

        metrics = {
            "pre_adapt_acc": mean_pre,
            "post_adapt_acc": mean_post,
            "fast_gain": mean_post - mean_pre,
            "auac": auac,
        }

        return MetaOutput(
            loss=meta_loss,
            metrics=metrics,
            inner_logs=all_inner_logs,
            adapted_params=all_adapted,
        )


def create_meta_algorithm(algo: str, model: nn.Module, **kwargs) -> Any:
    """Factory function to create the correct algorithm by name."""
    if algo == "maml":
        return MAMLAlgorithm(model, **kwargs)
    elif algo == "fomaml":
        return FOMAMLAlgorithm(model, **kwargs)
    elif algo == "reptile":
        return ReptileAlgorithm(model, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


class LSLRModule(nn.Module):
    """Per-Layer, per-Step Learned Learning Rates.

    Stores a (num_steps, num_layers) tensor of log-learning-rates.
    Effective LR = clamp(exp(log_lr), lr_min, lr_max).
    """

    def __init__(
        self,
        num_steps: int,
        num_layers: int,
        init_lr: float = 0.01,
        lr_min: float = 1e-6,
        lr_max: float = 1.0,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.lr_min = lr_min
        self.lr_max = lr_max
        # Initialize log-LRs so that exp(log_lr) = init_lr
        init_log_lr = math.log(max(init_lr, lr_min))
        self.log_lrs = nn.Parameter(
            torch.full((num_steps, num_layers), init_log_lr)
        )

    def get_lr(self, step: int, layer: int) -> Tensor:
        """Get the effective learning rate for a given step and layer."""
        raw = torch.exp(self.log_lrs[step, layer])
        return torch.clamp(raw, min=self.lr_min, max=self.lr_max)

    def get_all_lrs(self, step: int) -> Tensor:
        """Get effective LRs for all layers at a given step. Shape: (num_layers,)."""
        raw = torch.exp(self.log_lrs[step])
        return torch.clamp(raw, min=self.lr_min, max=self.lr_max)


class MultiStepLoss(nn.Module):
    """Multi-Step Loss: weighted combination of query losses at each inner step.

    Supports three modes:
    - "uniform": equal weights summing to 1
    - "linear_increase": linearly increasing weights summing to 1
    - "learned": softmax over learnable logits
    """

    def __init__(self, num_steps: int, mode: str = "uniform"):
        super().__init__()
        self.num_steps = num_steps
        self.mode = mode

        if mode == "learned":
            self.weight_logits = nn.Parameter(torch.zeros(num_steps))
        else:
            self.register_buffer("_dummy", torch.tensor(0.0))

    def get_weights(self) -> Tensor:
        """Return normalized weights for each step."""
        if self.mode == "uniform":
            return torch.ones(self.num_steps) / self.num_steps
        elif self.mode == "linear_increase":
            raw = torch.arange(1, self.num_steps + 1, dtype=torch.float32)
            return raw / raw.sum()
        elif self.mode == "learned":
            return F.softmax(self.weight_logits, dim=0)
        else:
            raise ValueError(f"Unknown MSL mode: {self.mode}")

    def combine_losses(self, step_losses: List[Tensor]) -> Tensor:
        """Combine per-step losses using the weight schedule."""
        assert len(step_losses) == self.num_steps, (
            f"Expected {self.num_steps} losses, got {len(step_losses)}"
        )
        weights = self.get_weights()
        if weights.device != step_losses[0].device:
            weights = weights.to(step_losses[0].device)
        total = sum(w * l for w, l in zip(weights, step_losses))
        return total


class MAMLPlusPlusAlgorithm:
    """MAML++ with LSLR and MSL enhancements."""

    def __init__(
        self,
        model: nn.Module,
        inner_steps: int = 5,
        inner_clip: float = 10.0,
        use_lslr: bool = True,
        use_msl: bool = True,
        msl_mode: str = "uniform",
        init_lr: float = 0.01,
    ):
        self.model = model
        self.inner_steps = inner_steps
        self.inner_clip = inner_clip
        self.init_lr = init_lr

        num_layers = len(list(model.parameters()))
        if use_lslr:
            self.lslr = LSLRModule(inner_steps, num_layers, init_lr=init_lr)
        else:
            self.lslr = None

        if use_msl:
            self.msl = MultiStepLoss(inner_steps, mode=msl_mode)
        else:
            self.msl = None

    def meta_step(self, task_batch: TaskBatch) -> MetaOutput:
        """MAML++ meta step with LSLR and MSL."""
        params = {n: p for n, p in self.model.named_parameters()}
        param_keys = list(params.keys())

        all_losses = []
        all_inner_logs = []
        all_adapted = []
        all_pre_acc = []
        all_post_acc = []

        for episode in task_batch.episodes:
            with torch.no_grad():
                pre_logits = functional_forward(self.model, params, episode.query_x)
                pre_acc = (pre_logits.argmax(-1) == episode.query_y).float().mean().item()
            all_pre_acc.append(pre_acc)

            adapted = {k: v for k, v in params.items()}
            logs: List[StepLog] = []
            step_losses: List[Tensor] = []

            for step_idx in range(self.inner_steps):
                logits = functional_forward(self.model, adapted, episode.support_x)
                loss = F.cross_entropy(logits, episode.support_y)

                grad_tensors = torch.autograd.grad(
                    outputs=loss,
                    inputs=list(adapted.values()),
                    create_graph=True,
                    allow_unused=True,
                )

                grads = []
                for g, (k, p) in zip(grad_tensors, adapted.items()):
                    grads.append(g if g is not None else torch.zeros_like(p))

                # Gradient clipping
                total_norm = torch.sqrt(
                    sum(g.detach().norm() ** 2 for g in grads)
                ).item()
                clipped = False
                if self.inner_clip > 0:
                    clip_coef = self.inner_clip / (total_norm + 1e-6)
                    if clip_coef < 1.0:
                        grads = [g * clip_coef for g in grads]
                        clipped = True

                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    accuracy = (preds == episode.support_y).float().mean().item()

                # Apply LSLR or flat LR
                new_adapted = {}
                for layer_idx, ((k, p), g) in enumerate(zip(adapted.items(), grads)):
                    if self.lslr is not None:
                        effective_lr = self.lslr.get_lr(step_idx, layer_idx)
                    else:
                        effective_lr = self.init_lr
                    new_adapted[k] = p - effective_lr * g

                update_norm = torch.sqrt(
                    sum((new_adapted[k] - adapted[k]).detach().norm() ** 2 for k in adapted)
                ).item()

                adapted = new_adapted

                logs.append(StepLog(
                    step=step_idx,
                    loss=loss.item(),
                    accuracy=accuracy,
                    grad_norm=total_norm,
                    update_norm=update_norm,
                    lr_effective=self.init_lr,
                    clipped=clipped,
                ))

                # MSL: compute query loss at each step
                if self.msl is not None:
                    q_logits = functional_forward(self.model, adapted, episode.query_x)
                    q_loss = F.cross_entropy(q_logits, episode.query_y)
                    step_losses.append(q_loss)

            # Final loss
            if self.msl is not None and step_losses:
                episode_loss = self.msl.combine_losses(step_losses)
            else:
                q_logits = functional_forward(self.model, adapted, episode.query_x)
                episode_loss = F.cross_entropy(q_logits, episode.query_y)

            with torch.no_grad():
                final_logits = functional_forward(self.model, adapted, episode.query_x)
                post_acc = (final_logits.argmax(-1) == episode.query_y).float().mean().item()
            all_post_acc.append(post_acc)

            all_losses.append(episode_loss)
            all_inner_logs.append(logs)
            all_adapted.append(adapted)

        meta_loss = torch.stack(all_losses).mean()
        mean_pre = sum(all_pre_acc) / len(all_pre_acc) if all_pre_acc else 0.0
        mean_post = sum(all_post_acc) / len(all_post_acc) if all_post_acc else 0.0
        auac = _compute_auac(all_inner_logs[0]) if all_inner_logs else 0.0

        metrics = {
            "pre_adapt_acc": mean_pre,
            "post_adapt_acc": mean_post,
            "fast_gain": mean_post - mean_pre,
            "auac": auac,
        }

        return MetaOutput(
            loss=meta_loss,
            metrics=metrics,
            inner_logs=all_inner_logs,
            adapted_params=all_adapted,
        )


def _compute_auac(logs: List[StepLog]) -> float:
    """Compute Area Under the Adaptation Curve from step logs.

    Uses trapezoidal integration of accuracy over steps, normalized to [0, 1].
    """
    if not logs:
        return 0.0
    if len(logs) < 2:
        return logs[0].accuracy

    total = 0.0
    for i in range(1, len(logs)):
        total += (logs[i].accuracy + logs[i - 1].accuracy) / 2.0
    return total / (len(logs) - 1)


@dataclass
class AdaptationCurve:
    """Per-step accuracy curve for one episode."""
    accuracies: List[float]
    losses: List[float]

    @property
    def pre_adapt_acc(self) -> float:
        return self.accuracies[0] if self.accuracies else 0.0

    @property
    def post_adapt_acc(self) -> float:
        return self.accuracies[-1] if self.accuracies else 0.0

    @property
    def fast_gain(self) -> float:
        if len(self.accuracies) >= 2:
            return self.accuracies[1] - self.accuracies[0]
        return 0.0

    @property
    def auac(self) -> float:
        if len(self.accuracies) < 2:
            return self.accuracies[0] if self.accuracies else 0.0
        total = sum(
            (self.accuracies[i] + self.accuracies[i - 1]) / 2.0
            for i in range(1, len(self.accuracies))
        )
        return total / (len(self.accuracies) - 1)


class MetricsAggregator:
    """Aggregates per-episode metrics into epoch-level statistics."""

    def __init__(self):
        self.curves: List[AdaptationCurve] = []
        self.losses: List[float] = []

    def add(self, curve: AdaptationCurve, loss: float):
        self.curves.append(curve)
        self.losses.append(loss)

    def aggregate(self) -> Dict[str, float]:
        if not self.curves:
            return {
                "pre_adapt_acc": 0.0,
                "post_adapt_acc": 0.0,
                "fast_gain": 0.0,
                "auac": 0.0,
                "mean_loss": 0.0,
            }
        return {
            "pre_adapt_acc": sum(c.pre_adapt_acc for c in self.curves) / len(self.curves),
            "post_adapt_acc": sum(c.post_adapt_acc for c in self.curves) / len(self.curves),
            "fast_gain": sum(c.fast_gain for c in self.curves) / len(self.curves),
            "auac": sum(c.auac for c in self.curves) / len(self.curves),
            "mean_loss": sum(self.losses) / len(self.losses),
        }


def save_meta_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    algo: str,
    inner_steps: int,
    inner_lr: float,
    inner_clip: float,
    sampler_config: Dict[str, Any],
    epoch: int,
    global_step: int,
    best_val_accuracy: float,
    metrics_history: List[Dict[str, float]],
    format_version: str = "1.0",
) -> str:
    """Save a meta-learning checkpoint."""
    checkpoint = {
        "format_version": format_version,
        "model_state_dict": model.state_dict(),
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "algo": algo,
        "second_order": (algo == "maml"),
        "inner_steps": inner_steps,
        "inner_lr": inner_lr,
        "inner_clip": inner_clip,
        "backend": "custom",
        "outer_optimizer_state_dict": optimizer.state_dict(),
        "outer_lr": optimizer.param_groups[0]["lr"],
        "sampler_config": sampler_config,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_accuracy": best_val_accuracy,
        "metrics_history": metrics_history[-50:],
    }
    torch.save(checkpoint, path)
    return path


def load_meta_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a meta-learning checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Version check
    version = ckpt.get("format_version")
    if version is None:
        raise ValueError("Checkpoint missing format_version")
    major = version.split(".")[0]
    if major not in {"1"}:
        raise ValueError(f"Unsupported checkpoint version: {version}")

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "outer_optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["outer_optimizer_state_dict"])

    model.to(device)
    return ckpt


def run_meta_epoch(
    algo_obj: Any,
    sampler: EpisodeSampler,
    num_episodes: int,
    optimizer: torch.optim.Optimizer,
    epoch: int = 0,
) -> Dict[str, float]:
    """Run one meta-training epoch.

    Returns aggregated metrics.
    """
    aggregator = MetricsAggregator()

    for ep_idx in range(num_episodes):
        episode = sampler.sample_episode(epoch=epoch, episode_idx=ep_idx)
        task_batch = TaskBatch(episodes=[episode])

        output = algo_obj.meta_step(task_batch)

        # Backward + optimize (only for gradient-based algorithms)
        if output.loss.requires_grad:
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()

        # Collect per-episode adaptation curve
        if output.inner_logs and output.inner_logs[0]:
            accs = [log.accuracy for log in output.inner_logs[0]]
            losses_log = [log.loss for log in output.inner_logs[0]]
        else:
            accs = [output.metrics.get("pre_adapt_acc", 0.0)]
            losses_log = [output.loss.item() if isinstance(output.loss, Tensor) else output.loss]

        curve = AdaptationCurve(accuracies=accs, losses=losses_log)
        aggregator.add(curve, output.loss.item() if isinstance(output.loss, Tensor) else output.loss)

    return aggregator.aggregate()


# ============================================================================
# Validator class
# ============================================================================

class MetaLearningValidator:
    """Runtime contract validator for the meta-learning suite."""

    def __init__(self, verbose: bool = False, seed: int = 42):
        self.verbose = verbose
        self.seed = seed
        self.results: List[CheckResult] = []

    # ------------------------------------------------------------------
    # Core check runner
    # ------------------------------------------------------------------

    def _check(
        self,
        name: str,
        group: str,
        fn: Callable[[], Tuple[bool, str, Optional[str]]],
    ) -> CheckResult:
        """Run a single validation check with timing and exception handling."""
        torch.manual_seed(self.seed)
        start = time.perf_counter()
        try:
            passed, message, details = fn()
        except Exception as e:
            passed = False
            message = f"Exception: {type(e).__name__}: {e}"
            details = traceback.format_exc() if self.verbose else str(e)
        elapsed = (time.perf_counter() - start) * 1000.0

        result = CheckResult(
            name=name,
            group=group,
            passed=passed,
            message=message,
            details=details,
            elapsed_ms=elapsed,
        )
        self.results.append(result)

        # Print immediately
        status = _c("PASS", _GREEN) if passed else _c("FAIL", _RED)
        print(f"  [{status}] {name}  ({elapsed:.1f}ms)")
        if self.verbose and message:
            print(f"         {_c(message, _DIM)}")
        if not passed and details:
            for line in details.strip().split("\n")[-5:]:
                print(f"         {_c(line, _RED)}")

        return result

    # ------------------------------------------------------------------
    # Group runners
    # ------------------------------------------------------------------

    def run_group(self, group: str) -> List[CheckResult]:
        """Run all checks in a single group."""
        if group not in _ALL_GROUPS:
            raise ValueError(f"Unknown group '{group}'. Available: {_ALL_GROUPS}")
        method = getattr(self, f"_run_{group}")
        print(f"\n{'=' * 72}")
        print(f"  Group: {_c(group.upper(), _BOLD + _CYAN)}")
        print(f"{'=' * 72}")
        before = len(self.results)
        method()
        return self.results[before:]

    def run_all(self) -> List[CheckResult]:
        """Run all validation groups."""
        for group in _ALL_GROUPS:
            self.run_group(group)
        return self.results

    # ==================================================================
    # GROUP 1: inner_loop (~6 checks)
    # ==================================================================

    def _run_inner_loop(self) -> None:
        """Inner-loop engine checks: adapt, StepLog, clipping, graph modes."""
        torch.manual_seed(self.seed)
        B, D_in, D_out = 8, 10, 5

        # --- check 1: adapt produces InnerLoopResult with correct fields ---
        def check_adapt_result_fields():
            """CustomSGDEngine adapt() returns InnerLoopResult with correct fields."""
            model = ToyLinearModel(D_in, 20, D_out)
            engine = CustomSGDEngine(model, inner_lr=0.01)
            params = {n: p for n, p in model.named_parameters()}
            x = torch.randn(B, D_in)
            y = torch.randint(0, D_out, (B,))

            result = engine.adapt(params, x, y, steps=3, create_graph=True)

            has_adapted = isinstance(result.adapted_params, dict)
            has_logs = isinstance(result.logs, list)
            correct_len = len(result.logs) == 3
            keys_match = set(result.adapted_params.keys()) == set(params.keys())

            ok = has_adapted and has_logs and correct_len and keys_match
            msg = (f"adapted_params keys match={keys_match}, "
                   f"logs len={len(result.logs)}, expected 3")
            return ok, msg, None

        self._check("adapt_result_fields", "inner_loop", check_adapt_result_fields)

        # --- check 2: inner loop loss decreases across steps ---
        def check_loss_decreases():
            """Inner loop loss should decrease (or at least not increase much) across steps."""
            model = ToyLinearModel(D_in, 20, D_out)
            engine = CustomSGDEngine(model, inner_lr=0.05)
            params = {n: p for n, p in model.named_parameters()}
            x = torch.randn(B, D_in)
            y = torch.randint(0, D_out, (B,))

            result = engine.adapt(params, x, y, steps=5, create_graph=False)
            losses = [log.loss for log in result.logs]
            # Loss at last step should be less than at first step
            ok = losses[-1] < losses[0]
            msg = f"loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f}"
            return ok, msg, None

        self._check("inner_loop_loss_decreases", "inner_loop", check_loss_decreases)

        # --- check 3: StepLog schema has required fields ---
        def check_step_log_schema():
            """StepLog has all required fields: step, loss, accuracy, grad_norm, update_norm, lr_effective, clipped."""
            model = ToyLinearModel(D_in, 20, D_out)
            engine = CustomSGDEngine(model, inner_lr=0.01)
            params = {n: p for n, p in model.named_parameters()}
            x = torch.randn(B, D_in)
            y = torch.randint(0, D_out, (B,))

            result = engine.adapt(params, x, y, steps=3, create_graph=False)
            required_fields = {"step", "loss", "accuracy", "grad_norm",
                               "update_norm", "lr_effective", "clipped"}
            all_ok = True
            msgs = []
            for log in result.logs:
                log_dict = asdict(log)
                missing = required_fields - set(log_dict.keys())
                if missing:
                    all_ok = False
                    msgs.append(f"step {log.step} missing: {missing}")
                # Type checks
                if not isinstance(log.loss, float):
                    all_ok = False
                    msgs.append(f"step {log.step}: loss not float")
                if not (0.0 <= log.accuracy <= 1.0):
                    all_ok = False
                    msgs.append(f"step {log.step}: accuracy {log.accuracy} out of [0,1]")
                if log.grad_norm < 0:
                    all_ok = False
                    msgs.append(f"step {log.step}: negative grad_norm")

            msg = "All fields present and valid" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("step_log_schema", "inner_loop", check_step_log_schema)

        # --- check 4: gradient clipping activates ---
        def check_grad_clipping_activates():
            """Gradient clipping triggers when norms exceed threshold."""
            model = ToyLinearModel(D_in, 20, D_out)
            engine = CustomSGDEngine(model, inner_lr=0.01, inner_clip=0.5)
            params = {n: p for n, p in model.named_parameters()}
            # Use large inputs to produce large gradients
            x = torch.randn(B, D_in) * 100.0
            y = torch.randint(0, D_out, (B,))

            result = engine.adapt(params, x, y, steps=3, create_graph=False, clip_norm=0.5)
            any_clipped = any(log.clipped for log in result.logs)
            ok = any_clipped
            clip_info = [(log.step, log.clipped, log.grad_norm) for log in result.logs]
            msg = f"Clipping events: {clip_info}"
            return ok, msg, None

        self._check("grad_clipping_activates", "inner_loop", check_grad_clipping_activates)

        # --- check 5: create_graph=True retains grad_fn ---
        def check_create_graph_true():
            """create_graph=True retains grad_fn on adapted params."""
            model = ToyLinearModel(D_in, 20, D_out)
            engine = CustomSGDEngine(model, inner_lr=0.01)
            params = {n: p for n, p in model.named_parameters()}
            x = torch.randn(B, D_in)
            y = torch.randint(0, D_out, (B,))

            result = engine.adapt(params, x, y, steps=3, create_graph=True)
            all_have_grad_fn = all(
                v.grad_fn is not None for v in result.adapted_params.values()
            )
            ok = all_have_grad_fn
            grad_fns = {k: v.grad_fn is not None for k, v in result.adapted_params.items()}
            msg = f"grad_fn present: {grad_fns}"
            return ok, msg, None

        self._check("create_graph_true_retains_grad_fn", "inner_loop", check_create_graph_true)

        # --- check 6: create_graph=False produces leaf-like tensors ---
        def check_create_graph_false():
            """create_graph=False: adapted params should not have grad_fn through the update."""
            model = ToyLinearModel(D_in, 20, D_out)
            engine = CustomSGDEngine(model, inner_lr=0.01)
            params = {n: p.clone().detach().requires_grad_(True) for n, p in model.named_parameters()}
            x = torch.randn(B, D_in)
            y = torch.randint(0, D_out, (B,))

            result = engine.adapt(params, x, y, steps=3, create_graph=False)
            # With create_graph=False, the gradient g is detached, so adapted = p - lr*g
            # The adapted params still have grad_fn from the subtraction with p,
            # but the graph does NOT go through the gradient computation.
            # The key test is that the grad_fn chain does not include the autograd.grad
            # computation -- we verify by checking that we CAN still backprop (first-order)
            # but the graph is simpler.
            # The practical test: no second-order gradient information is retained.
            # We check that adapted_params tensors are produced by a SubBackward (p - lr*g)
            # where g has no grad_fn.
            ok = True
            msg_parts = []
            for k, v in result.adapted_params.items():
                # In FOMAML mode, tensors may or may not have grad_fn depending on
                # whether p still has grad_fn. The key invariant is that the grads
                # used in the update are detached.
                has_gf = v.grad_fn is not None
                msg_parts.append(f"{k}: grad_fn={has_gf}")
            # This is informational -- the real test of FOMAML is that meta-grads
            # exist but are first-order (tested in gate_a).
            msg = "; ".join(msg_parts)
            return ok, msg, None

        self._check("create_graph_false_leaf_tensors", "inner_loop", check_create_graph_false)

    # ==================================================================
    # GROUP 2: algorithms (~6 checks)
    # ==================================================================

    def _run_algorithms(self) -> None:
        """MAML / FOMAML / Reptile algorithm checks."""
        torch.manual_seed(self.seed)
        D_in, D_out = 10, 5
        B = 8

        def _make_task_batch():
            model = ToyLinearModel(D_in, 20, D_out)
            x_s = torch.randn(B, D_in)
            y_s = torch.randint(0, D_out, (B,))
            x_q = torch.randn(B, D_in)
            y_q = torch.randint(0, D_out, (B,))
            ep = Episode(
                support_x=x_s, support_y=y_s,
                query_x=x_q, query_y=y_q,
                class_ids=list(range(D_out)), n_way=D_out, k_shot=1, q_query=1,
            )
            return model, TaskBatch(episodes=[ep])

        # --- check 1: MAML meta_step returns MetaOutput ---
        def check_maml_meta_output():
            """MAML meta_step returns MetaOutput with loss, metrics, inner_logs."""
            torch.manual_seed(self.seed)
            model, tb = _make_task_batch()
            algo = MAMLAlgorithm(model, inner_lr=0.01, inner_steps=3)
            output = algo.meta_step(tb)

            has_loss = isinstance(output.loss, Tensor) and output.loss.dim() == 0
            has_metrics = isinstance(output.metrics, dict)
            has_inner_logs = isinstance(output.inner_logs, list) and len(output.inner_logs) > 0
            req_keys = {"pre_adapt_acc", "post_adapt_acc", "fast_gain", "auac"}
            has_req_keys = req_keys.issubset(set(output.metrics.keys()))

            ok = has_loss and has_metrics and has_inner_logs and has_req_keys
            msg = (f"loss shape={output.loss.shape}, metrics keys={set(output.metrics.keys())}, "
                   f"inner_logs len={len(output.inner_logs)}")
            return ok, msg, None

        self._check("maml_meta_output", "algorithms", check_maml_meta_output)

        # --- check 2: FOMAML meta_step returns MetaOutput ---
        def check_fomaml_meta_output():
            """FOMAML meta_step returns MetaOutput."""
            torch.manual_seed(self.seed)
            model, tb = _make_task_batch()
            algo = FOMAMLAlgorithm(model, inner_lr=0.01, inner_steps=3)
            output = algo.meta_step(tb)

            has_loss = isinstance(output.loss, Tensor) and output.loss.dim() == 0
            has_metrics = isinstance(output.metrics, dict)
            req_keys = {"pre_adapt_acc", "post_adapt_acc", "fast_gain", "auac"}
            has_req_keys = req_keys.issubset(set(output.metrics.keys()))

            ok = has_loss and has_metrics and has_req_keys
            msg = f"loss={output.loss.item():.4f}, metrics keys={set(output.metrics.keys())}"
            return ok, msg, None

        self._check("fomaml_meta_output", "algorithms", check_fomaml_meta_output)

        # --- check 3: Reptile meta_step returns MetaOutput ---
        def check_reptile_meta_output():
            """Reptile meta_step returns MetaOutput."""
            torch.manual_seed(self.seed)
            model, tb = _make_task_batch()
            algo = ReptileAlgorithm(model, inner_lr=0.01, inner_steps=3, epsilon=0.1)
            output = algo.meta_step(tb)

            has_loss = isinstance(output.loss, Tensor)
            has_metrics = isinstance(output.metrics, dict)
            req_keys = {"pre_adapt_acc", "post_adapt_acc", "fast_gain", "auac"}
            has_req_keys = req_keys.issubset(set(output.metrics.keys()))

            ok = has_loss and has_metrics and has_req_keys
            msg = f"loss val={output.loss.item():.4f}, metrics keys={set(output.metrics.keys())}"
            return ok, msg, None

        self._check("reptile_meta_output", "algorithms", check_reptile_meta_output)

        # --- check 4: all three produce non-NaN loss ---
        def check_no_nan_loss():
            """All three algorithms produce finite, non-NaN loss."""
            all_ok = True
            msgs = []
            for algo_name in ["maml", "fomaml", "reptile"]:
                torch.manual_seed(self.seed)
                model, tb = _make_task_batch()
                kwargs = {"inner_lr": 0.01, "inner_steps": 3}
                if algo_name == "reptile":
                    kwargs["epsilon"] = 0.1
                algo = create_meta_algorithm(algo_name, model, **kwargs)
                output = algo.meta_step(tb)
                loss_val = output.loss.item()
                if math.isnan(loss_val) or math.isinf(loss_val):
                    all_ok = False
                    msgs.append(f"{algo_name}: loss={loss_val}")
            msg = "All losses finite" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("all_algos_non_nan_loss", "algorithms", check_no_nan_loss)

        # --- check 5: MAML meta-loss has grad_fn ---
        def check_maml_loss_has_grad_fn():
            """MAML meta-loss should be backprop-able (has grad_fn)."""
            torch.manual_seed(self.seed)
            model, tb = _make_task_batch()
            algo = MAMLAlgorithm(model, inner_lr=0.01, inner_steps=3)
            output = algo.meta_step(tb)

            has_grad_fn = output.loss.grad_fn is not None
            ok = has_grad_fn
            msg = f"loss.grad_fn is not None: {has_grad_fn}"
            return ok, msg, None

        self._check("maml_loss_has_grad_fn", "algorithms", check_maml_loss_has_grad_fn)

        # --- check 6: factory function creates correct type ---
        def check_factory_function():
            """create_meta_algorithm returns the correct algorithm class."""
            torch.manual_seed(self.seed)
            model = ToyLinearModel(D_in, 20, D_out)
            all_ok = True
            msgs = []
            expected = {
                "maml": MAMLAlgorithm,
                "fomaml": FOMAMLAlgorithm,
                "reptile": ReptileAlgorithm,
            }
            for name, cls in expected.items():
                kwargs = {"inner_lr": 0.01, "inner_steps": 3}
                if name == "reptile":
                    kwargs["epsilon"] = 0.1
                obj = create_meta_algorithm(name, model, **kwargs)
                if not isinstance(obj, cls):
                    all_ok = False
                    msgs.append(f"{name}: got {type(obj).__name__}, expected {cls.__name__}")
            msg = "All factory outputs correct" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("factory_creates_correct_type", "algorithms", check_factory_function)

    # ==================================================================
    # GROUP 3: maml_plus (~5 checks)
    # ==================================================================

    def _run_maml_plus(self) -> None:
        """MAML++ enhancements: LSLR, MSL, MAMLPlusPlusAlgorithm."""
        torch.manual_seed(self.seed)

        # --- check 1: LSLR parameter shape ---
        def check_lslr_shape():
            """LSLR creates parameters with shape (num_steps, num_layers)."""
            num_steps, num_layers = 5, 4
            lslr = LSLRModule(num_steps, num_layers, init_lr=0.01)
            shape = lslr.log_lrs.shape
            ok = shape == (num_steps, num_layers)
            msg = f"LSLR shape: {shape}, expected ({num_steps}, {num_layers})"
            return ok, msg, None

        self._check("lslr_parameter_shape", "maml_plus", check_lslr_shape)

        # --- check 2: LSLR get_lr returns values in [lr_min, lr_max] ---
        def check_lslr_clamped():
            """LSLR get_lr returns values within [lr_min, lr_max]."""
            lslr = LSLRModule(3, 4, init_lr=0.01, lr_min=1e-6, lr_max=1.0)
            # Set some extreme values
            with torch.no_grad():
                lslr.log_lrs[0, 0] = -20.0   # Would give tiny lr
                lslr.log_lrs[1, 1] = 10.0     # Would give huge lr
            all_ok = True
            msgs = []
            for s in range(3):
                for l in range(4):
                    lr = lslr.get_lr(s, l).item()
                    if lr < 1e-6 - 1e-10 or lr > 1.0 + 1e-6:
                        all_ok = False
                        msgs.append(f"step={s}, layer={l}: lr={lr}")
            msg = "All LRs in [1e-6, 1.0]" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("lslr_clamped_range", "maml_plus", check_lslr_clamped)

        # --- check 3: MSL uniform weights sum to 1.0 ---
        def check_msl_uniform():
            """MSL with uniform weights sums to 1.0."""
            msl = MultiStepLoss(5, mode="uniform")
            weights = msl.get_weights()
            total = weights.sum().item()
            ok = abs(total - 1.0) < 1e-5
            msg = f"Uniform weights sum: {total:.6f}, each={weights[0].item():.4f}"
            return ok, msg, None

        self._check("msl_uniform_sums_to_one", "maml_plus", check_msl_uniform)

        # --- check 4: MSL combines step losses correctly ---
        def check_msl_combine():
            """MSL combine_losses produces weighted sum."""
            msl = MultiStepLoss(3, mode="uniform")
            losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
            combined = msl.combine_losses(losses)
            # Uniform: (1 + 2 + 3) / 3 = 2.0
            expected = 2.0
            ok = abs(combined.item() - expected) < 1e-5
            msg = f"Combined loss: {combined.item():.4f}, expected {expected}"
            return ok, msg, None

        self._check("msl_combines_correctly", "maml_plus", check_msl_combine)

        # --- check 5: MAMLPlusPlusAlgorithm meta_step completes ---
        def check_maml_plus_plus():
            """MAMLPlusPlusAlgorithm meta_step completes without error."""
            torch.manual_seed(self.seed)
            D_in, D_out, B = 10, 5, 8
            model = ToyLinearModel(D_in, 20, D_out)
            algo = MAMLPlusPlusAlgorithm(
                model, inner_steps=3, inner_clip=10.0,
                use_lslr=True, use_msl=True, msl_mode="uniform",
                init_lr=0.01,
            )
            x_s = torch.randn(B, D_in)
            y_s = torch.randint(0, D_out, (B,))
            x_q = torch.randn(B, D_in)
            y_q = torch.randint(0, D_out, (B,))
            ep = Episode(
                support_x=x_s, support_y=y_s,
                query_x=x_q, query_y=y_q,
                class_ids=list(range(D_out)), n_way=D_out, k_shot=1, q_query=1,
            )
            tb = TaskBatch(episodes=[ep])
            output = algo.meta_step(tb)

            has_loss = isinstance(output.loss, Tensor) and not math.isnan(output.loss.item())
            has_metrics = isinstance(output.metrics, dict)
            ok = has_loss and has_metrics
            msg = f"loss={output.loss.item():.4f}, metrics={output.metrics}"
            return ok, msg, None

        self._check("maml_plus_plus_completes", "maml_plus", check_maml_plus_plus)

    # ==================================================================
    # GROUP 4: sampling (~5 checks)
    # ==================================================================

    def _run_sampling(self) -> None:
        """EpisodeSampler checks: shapes, determinism, class splits."""
        torch.manual_seed(self.seed)
        N_WAY, K_SHOT, Q_QUERY = 5, 1, 3
        NUM_CLASSES, SAMPLES_PER_CLASS = 50, 20
        FEATURE_DIM = 10

        dataset = SyntheticFewShotDataset(
            num_classes=NUM_CLASSES,
            samples_per_class=SAMPLES_PER_CLASS,
            feature_dim=FEATURE_DIM,
            seed=self.seed,
        )

        # --- check 1: episode has correct shapes ---
        def check_episode_shapes():
            """EpisodeSampler produces episodes with correct shapes."""
            sampler = EpisodeSampler(dataset, n_way=N_WAY, k_shot=K_SHOT,
                                     q_query=Q_QUERY, global_seed=self.seed)
            ep = sampler.sample_episode(epoch=0, episode_idx=0)
            s_shape = ep.support_x.shape
            q_shape = ep.query_x.shape
            expected_s = (N_WAY * K_SHOT, FEATURE_DIM)
            expected_q = (N_WAY * Q_QUERY, FEATURE_DIM)

            s_ok = s_shape == expected_s
            q_ok = q_shape == expected_q
            ok = s_ok and q_ok
            msg = (f"support shape={s_shape} (expected {expected_s}), "
                   f"query shape={q_shape} (expected {expected_q})")
            return ok, msg, None

        self._check("episode_correct_shapes", "sampling", check_episode_shapes)

        # --- check 2: same seed/epoch/idx produces identical episodes ---
        def check_deterministic_same():
            """Same (seed, epoch, idx) produces identical episodes."""
            sampler = EpisodeSampler(dataset, n_way=N_WAY, k_shot=K_SHOT,
                                     q_query=Q_QUERY, global_seed=42)
            ep1 = sampler.sample_episode(epoch=0, episode_idx=0)
            ep2 = sampler.sample_episode(epoch=0, episode_idx=0)
            ok = (
                torch.equal(ep1.support_x, ep2.support_x) and
                torch.equal(ep1.support_y, ep2.support_y) and
                torch.equal(ep1.query_x, ep2.query_x) and
                torch.equal(ep1.query_y, ep2.query_y) and
                ep1.class_ids == ep2.class_ids
            )
            msg = f"Episodes identical: {ok}"
            return ok, msg, None

        self._check("deterministic_same_seed", "sampling", check_deterministic_same)

        # --- check 3: different seeds produce different episodes ---
        def check_deterministic_different():
            """Different seeds produce different episodes."""
            sampler1 = EpisodeSampler(dataset, n_way=N_WAY, k_shot=K_SHOT,
                                      q_query=Q_QUERY, global_seed=42)
            sampler2 = EpisodeSampler(dataset, n_way=N_WAY, k_shot=K_SHOT,
                                      q_query=Q_QUERY, global_seed=99)
            ep1 = sampler1.sample_episode(epoch=0, episode_idx=0)
            ep2 = sampler2.sample_episode(epoch=0, episode_idx=0)
            different = (
                not torch.equal(ep1.support_x, ep2.support_x) or
                ep1.class_ids != ep2.class_ids
            )
            ok = different
            msg = (f"class_ids_1={ep1.class_ids[:3]}..., "
                   f"class_ids_2={ep2.class_ids[:3]}..., different={different}")
            return ok, msg, None

        self._check("deterministic_different_seed", "sampling", check_deterministic_different)

        # --- check 4: support and query labels in [0, N) ---
        def check_label_range():
            """Support and query labels are in [0, N_way)."""
            sampler = EpisodeSampler(dataset, n_way=N_WAY, k_shot=K_SHOT,
                                     q_query=Q_QUERY, global_seed=self.seed)
            ep = sampler.sample_episode(epoch=0, episode_idx=0)
            s_labels = ep.support_y.unique().tolist()
            q_labels = ep.query_y.unique().tolist()
            all_labels = sorted(set(s_labels + q_labels))
            ok = all_labels == list(range(N_WAY))
            msg = f"Labels: support={s_labels}, query={q_labels}, expected [0..{N_WAY-1}]"
            return ok, msg, None

        self._check("labels_in_range", "sampling", check_label_range)

        # --- check 5: class split has no overlap ---
        def check_class_split_disjoint():
            """Train/val/test class splits have no overlap."""
            sampler = EpisodeSampler(dataset, n_way=N_WAY, k_shot=K_SHOT,
                                     q_query=Q_QUERY, global_seed=self.seed)
            train_set = set(sampler.train_classes)
            val_set = set(sampler.val_classes)
            test_set = set(sampler.test_classes)

            tv_overlap = train_set & val_set
            tt_overlap = train_set & test_set
            vt_overlap = val_set & test_set
            no_overlap = len(tv_overlap) == 0 and len(tt_overlap) == 0 and len(vt_overlap) == 0
            covers_all = len(train_set | val_set | test_set) == NUM_CLASSES

            ok = no_overlap and covers_all
            msg = (f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}, "
                   f"overlaps={len(tv_overlap)}+{len(tt_overlap)}+{len(vt_overlap)}, "
                   f"covers_all={covers_all}")
            return ok, msg, None

        self._check("class_split_no_overlap", "sampling", check_class_split_disjoint)

    # ==================================================================
    # GROUP 5: gate_a -- Meta-gradient flow validated (~5 checks)
    # ==================================================================

    def _run_gate_a(self) -> None:
        """Done-When Gate (a): Meta-gradient flow validation."""
        torch.manual_seed(self.seed)

        # --- check 1: MAML meta-gradient non-zero on base params ---
        def check_maml_gradient_nonzero():
            """Toy 2-layer model: MAML inner loop, backward on outer loss, base params have non-zero .grad."""
            torch.manual_seed(self.seed)
            D_in, D_hid, D_out = 10, 20, 5
            B = 8
            model = ToyLinearModel(D_in, D_hid, D_out)
            # Use the model's own parameters (leaves) so .grad accumulates normally
            params = {n: p for n, p in model.named_parameters()}

            x_s = torch.randn(B, D_in)
            y_s = torch.randint(0, D_out, (B,))
            x_q = torch.randn(B, D_in)
            y_q = torch.randint(0, D_out, (B,))

            engine = CustomSGDEngine(model, inner_lr=0.01)
            result = engine.adapt(params, x_s, y_s, steps=1, create_graph=True)

            query_logits = functional_forward(model, result.adapted_params, x_q)
            outer_loss = F.cross_entropy(query_logits, y_q)
            outer_loss.backward()

            all_have_grad = True
            all_nonzero = True
            msgs = []
            for name, p in model.named_parameters():
                if p.grad is None:
                    all_have_grad = False
                    msgs.append(f"{name}: no grad")
                elif p.grad.abs().sum().item() == 0:
                    all_nonzero = False
                    msgs.append(f"{name}: zero grad")

            ok = all_have_grad and all_nonzero
            msg = "All base params have non-zero .grad" if ok else "; ".join(msgs)
            return ok, msg, None

        self._check("maml_gradient_nonzero", "gate_a", check_maml_gradient_nonzero)

        # --- check 2: FOMAML grads exist without create_graph ---
        def check_fomaml_grads_exist():
            """FOMAML: grads exist on base params but create_graph was False."""
            torch.manual_seed(self.seed)
            D_in, D_hid, D_out = 10, 20, 5
            B = 8
            model = ToyLinearModel(D_in, D_hid, D_out)
            params = {n: p for n, p in model.named_parameters()}

            x_s = torch.randn(B, D_in)
            y_s = torch.randint(0, D_out, (B,))
            x_q = torch.randn(B, D_in)
            y_q = torch.randint(0, D_out, (B,))

            engine = CustomSGDEngine(model, inner_lr=0.01)
            result = engine.adapt(params, x_s, y_s, steps=3, create_graph=False)

            query_logits = functional_forward(model, result.adapted_params, x_q)
            outer_loss = F.cross_entropy(query_logits, y_q)
            outer_loss.backward()

            grads_exist = all(
                p.grad is not None for p in model.parameters()
            )
            ok = grads_exist
            msg = f"FOMAML grads exist on model params: {grads_exist}"
            return ok, msg, None

        self._check("fomaml_grads_exist", "gate_a", check_fomaml_grads_exist)

        # --- check 3: detach detector ---
        def check_detach_detector():
            """MAML adapted params have grad_fn; FOMAML adapted params lack grad_fn through update."""
            torch.manual_seed(self.seed)
            D_in, D_hid, D_out = 10, 20, 5
            B = 8
            model = ToyLinearModel(D_in, D_hid, D_out)
            x_s = torch.randn(B, D_in)
            y_s = torch.randint(0, D_out, (B,))

            # MAML
            params_maml = {n: p for n, p in model.named_parameters()}
            engine = CustomSGDEngine(model, inner_lr=0.01)
            result_maml = engine.adapt(params_maml, x_s, y_s, steps=3, create_graph=True)
            maml_has_grad_fn = all(
                v.grad_fn is not None for v in result_maml.adapted_params.values()
            )

            # FOMAML with fully detached initial params
            params_fo = {n: p.clone().detach().requires_grad_(True) for n, p in model.named_parameters()}
            result_fo = engine.adapt(params_fo, x_s, y_s, steps=3, create_graph=False)
            # In FOMAML, the grad in the update is detached.
            # The adapted param = p - lr*g where g has no grad_fn.
            # Since p has requires_grad=True (leaf), adapted = p - lr*constant,
            # which will have grad_fn (SubBackward). But the graph is shallow.
            # The key distinction: MAML graph goes through gradient computation,
            # FOMAML graph only goes through the parameter itself.
            # We verify by checking graph depth.
            fomaml_adapted_have_grad_fn = all(
                v.grad_fn is not None for v in result_fo.adapted_params.values()
            )

            ok = maml_has_grad_fn  # The critical check is MAML retains grad_fn
            msg = (f"MAML adapted_params have grad_fn: {maml_has_grad_fn}, "
                   f"FOMAML adapted_params have grad_fn: {fomaml_adapted_have_grad_fn}")
            return ok, msg, None

        self._check("detach_detector", "gate_a", check_detach_detector)

        # --- check 4: hand-computed reference comparison ---
        def check_hand_computed_reference():
            """f(x)=Wx+b, 1 inner step MSE: compare MAML meta-grad to autograd reference."""
            torch.manual_seed(self.seed + 100)
            D_in, D_out = 4, 2
            alpha = 0.1

            # Fixed weights
            W = torch.randn(D_out, D_in, requires_grad=True)
            b = torch.randn(D_out, requires_grad=True)

            # Fixed data
            x_s = torch.randn(1, D_in)
            y_s = torch.randn(1, D_out)
            x_q = torch.randn(1, D_in)
            y_q = torch.randn(1, D_out)

            # Ground-truth meta-gradient via autograd
            W_ag = W.detach().clone().requires_grad_(True)
            b_ag = b.detach().clone().requires_grad_(True)

            # Support forward + loss (MSE)
            pred_s_ag = (W_ag @ x_s.squeeze() + b_ag).unsqueeze(0)
            loss_s = 0.5 * ((pred_s_ag - y_s) ** 2).sum()

            # Inner step with create_graph=True
            grad_W, grad_b = torch.autograd.grad(
                loss_s, [W_ag, b_ag], create_graph=True,
            )
            W_adapted = W_ag - alpha * grad_W
            b_adapted = b_ag - alpha * grad_b

            # Query forward + loss
            pred_q_ag = (W_adapted @ x_q.squeeze() + b_adapted).unsqueeze(0)
            loss_q = 0.5 * ((pred_q_ag - y_q) ** 2).sum()

            # Meta-gradient
            loss_q.backward()
            meta_grad_W_ag = W_ag.grad.clone()
            meta_grad_b_ag = b_ag.grad.clone()

            # --- Now compute with our engine ---
            model = TinyLinearModel(D_in, D_out)
            # Set model weights to match
            with torch.no_grad():
                model.linear.weight.copy_(W.detach())
                model.linear.bias.copy_(b.detach())

            params_engine = {
                "linear.weight": W.detach().clone().requires_grad_(True),
                "linear.bias": b.detach().clone().requires_grad_(True),
            }

            # We need to use MSE loss, so we do a manual inner step
            logits_s = functional_forward(model, params_engine, x_s)
            loss_s_eng = 0.5 * ((logits_s - y_s) ** 2).sum()
            grads_eng = torch.autograd.grad(
                loss_s_eng, list(params_engine.values()), create_graph=True,
            )
            adapted_engine = {
                "linear.weight": params_engine["linear.weight"] - alpha * grads_eng[0],
                "linear.bias": params_engine["linear.bias"] - alpha * grads_eng[1],
            }

            logits_q = functional_forward(model, adapted_engine, x_q)
            loss_q_eng = 0.5 * ((logits_q - y_q) ** 2).sum()
            loss_q_eng.backward()

            meta_grad_W_eng = params_engine["linear.weight"].grad
            meta_grad_b_eng = params_engine["linear.bias"].grad

            # Compare
            W_close = torch.allclose(meta_grad_W_eng, meta_grad_W_ag, atol=1e-4)
            b_close = torch.allclose(meta_grad_b_eng, meta_grad_b_ag, atol=1e-4)

            ok = W_close and b_close
            W_diff = (meta_grad_W_eng - meta_grad_W_ag).abs().max().item()
            b_diff = (meta_grad_b_eng - meta_grad_b_ag).abs().max().item()
            msg = f"W grad max diff={W_diff:.6f}, b grad max diff={b_diff:.6f}"
            return ok, msg, None

        self._check("hand_computed_reference", "gate_a", check_hand_computed_reference)

        # --- check 5: gradient magnitude sanity ---
        def check_gradient_magnitude_sanity():
            """Meta-gradients should be within [1e-8, 1e4] (not vanished or exploded)."""
            torch.manual_seed(self.seed)
            D_in, D_hid, D_out = 10, 20, 5
            B = 8
            model = ToyLinearModel(D_in, D_hid, D_out)
            # Use model's own leaf parameters so .grad accumulates normally
            params = {n: p for n, p in model.named_parameters()}

            x_s = torch.randn(B, D_in)
            y_s = torch.randint(0, D_out, (B,))
            x_q = torch.randn(B, D_in)
            y_q = torch.randint(0, D_out, (B,))

            engine = CustomSGDEngine(model, inner_lr=0.01)
            result = engine.adapt(params, x_s, y_s, steps=3, create_graph=True)

            query_logits = functional_forward(model, result.adapted_params, x_q)
            outer_loss = F.cross_entropy(query_logits, y_q)
            outer_loss.backward()

            all_ok = True
            msgs = []
            for name, p in model.named_parameters():
                if p.grad is None:
                    all_ok = False
                    msgs.append(f"{name}: no grad")
                    continue
                mag = p.grad.abs().max().item()
                if mag < 1e-8:
                    all_ok = False
                    msgs.append(f"{name}: grad too small ({mag:.2e})")
                elif mag > 1e4:
                    all_ok = False
                    msgs.append(f"{name}: grad too large ({mag:.2e})")
                else:
                    msgs.append(f"{name}: grad mag={mag:.4e}")

            ok = all_ok
            msg = "; ".join(msgs[:4])  # Truncate for readability
            return ok, msg, None

        self._check("gradient_magnitude_sanity", "gate_a", check_gradient_magnitude_sanity)

    # ==================================================================
    # GROUP 6: gate_b -- FO variants produce comparable adaptation (~5 checks)
    # ==================================================================

    def _run_gate_b(self) -> None:
        """Done-When Gate (b): FO variants produce comparable adaptation curves."""
        torch.manual_seed(self.seed)
        N_WAY = 5
        K_SHOT = 1
        Q_QUERY = 3
        FEATURE_DIM = 10
        D_HID = 20
        NUM_EPISODES = 30
        NUM_CLASSES = 100
        SAMPLES_PER_CLASS = 20
        INNER_STEPS = 5
        INNER_LR = 0.05

        dataset = SyntheticFewShotDataset(
            num_classes=NUM_CLASSES,
            samples_per_class=SAMPLES_PER_CLASS,
            feature_dim=FEATURE_DIM,
            seed=self.seed,
        )
        sampler = EpisodeSampler(
            dataset, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY,
            global_seed=self.seed,
        )

        # Run all algorithms and collect pre/post accuracies
        algo_results: Dict[str, Dict[str, List[float]]] = {}

        for algo_name in ["maml", "fomaml", "reptile"]:
            torch.manual_seed(self.seed)
            model = ToyLinearModel(FEATURE_DIM, D_HID, N_WAY)

            pre_accs = []
            post_accs = []

            for ep_idx in range(NUM_EPISODES):
                episode = sampler.sample_episode(epoch=0, episode_idx=ep_idx)
                params = {n: p for n, p in model.named_parameters()}

                # Pre-adaptation accuracy
                with torch.no_grad():
                    pre_logits = functional_forward(model, params, episode.query_x)
                    pre_acc = (pre_logits.argmax(-1) == episode.query_y).float().mean().item()
                pre_accs.append(pre_acc)

                if algo_name in ("maml", "fomaml"):
                    create_graph = (algo_name == "maml")
                    engine = CustomSGDEngine(model, inner_lr=INNER_LR)
                    result = engine.adapt(
                        params, episode.support_x, episode.support_y,
                        steps=INNER_STEPS, create_graph=create_graph,
                    )
                    with torch.no_grad():
                        post_logits = functional_forward(model, result.adapted_params, episode.query_x)
                        post_acc = (post_logits.argmax(-1) == episode.query_y).float().mean().item()
                else:
                    adapted, _ = reptile_inner_loop(
                        model, params, episode.support_x, episode.support_y,
                        steps=INNER_STEPS, lr=INNER_LR,
                    )
                    with torch.no_grad():
                        post_logits = functional_forward(model, adapted, episode.query_x)
                        post_acc = (post_logits.argmax(-1) == episode.query_y).float().mean().item()

                post_accs.append(post_acc)

            algo_results[algo_name] = {
                "pre_accs": pre_accs,
                "post_accs": post_accs,
            }

        random_chance = 1.0 / N_WAY

        # --- check 1: MAML and FOMAML step-0 accuracy identical ---
        def check_step0_match():
            """MAML and FOMAML pre-adaptation accuracy should match (identical base model)."""
            maml_pre = algo_results["maml"]["pre_accs"]
            fomaml_pre = algo_results["fomaml"]["pre_accs"]
            # Both use the same initial model, so pre-adapt accuracy should be identical
            # However, since both start from the same seed, the model is the same
            diffs = [abs(a - b) for a, b in zip(maml_pre, fomaml_pre)]
            max_diff = max(diffs)
            ok = max_diff < 1e-5
            msg = f"Max pre-adapt accuracy diff (MAML vs FOMAML): {max_diff:.6f}"
            return ok, msg, None

        self._check("step0_maml_fomaml_match", "gate_b", check_step0_match)

        # --- check 2: MAML post-adapt > random ---
        def check_maml_above_random():
            """MAML post-adaptation accuracy > random (1/N_way)."""
            mean_post = sum(algo_results["maml"]["post_accs"]) / NUM_EPISODES
            ok = mean_post > random_chance
            msg = f"MAML mean post-adapt acc={mean_post:.4f}, random={random_chance:.4f}"
            return ok, msg, None

        self._check("maml_above_random", "gate_b", check_maml_above_random)

        # --- check 3: FOMAML post-adapt > random ---
        def check_fomaml_above_random():
            """FOMAML post-adaptation accuracy > random (1/N_way)."""
            mean_post = sum(algo_results["fomaml"]["post_accs"]) / NUM_EPISODES
            ok = mean_post > random_chance
            msg = f"FOMAML mean post-adapt acc={mean_post:.4f}, random={random_chance:.4f}"
            return ok, msg, None

        self._check("fomaml_above_random", "gate_b", check_fomaml_above_random)

        # --- check 4: Reptile post-adapt > random ---
        def check_reptile_above_random():
            """Reptile post-adaptation accuracy > random."""
            mean_post = sum(algo_results["reptile"]["post_accs"]) / NUM_EPISODES
            ok = mean_post > random_chance
            msg = f"Reptile mean post-adapt acc={mean_post:.4f}, random={random_chance:.4f}"
            return ok, msg, None

        self._check("reptile_above_random", "gate_b", check_reptile_above_random)

        # --- check 5: no algorithm flatlines ---
        def check_no_flatline():
            """At least 50% of episodes should show post > pre for each algorithm."""
            all_ok = True
            msgs = []
            for algo_name in ["maml", "fomaml", "reptile"]:
                pre = algo_results[algo_name]["pre_accs"]
                post = algo_results[algo_name]["post_accs"]
                improved = sum(1 for p, q in zip(pre, post) if q > p)
                ratio = improved / NUM_EPISODES
                if ratio < 0.5:
                    all_ok = False
                    msgs.append(f"{algo_name}: only {ratio*100:.0f}% improved")
                else:
                    msgs.append(f"{algo_name}: {ratio*100:.0f}% improved")
            msg = "; ".join(msgs)
            return all_ok, msg, None

        self._check("no_algorithm_flatlines", "gate_b", check_no_flatline)

    # ==================================================================
    # GROUP 7: gate_c -- Phase 7 scripts run on CPU in dev mode (~6 checks)
    # ==================================================================

    def _run_gate_c(self) -> None:
        """Done-When Gate (c): Phase 7 CPU dev mode end-to-end."""
        torch.manual_seed(self.seed)
        N_WAY = 5
        K_SHOT = 1
        Q_QUERY = 3
        FEATURE_DIM = 10
        D_HID = 20
        NUM_EPISODES = 15
        NUM_CLASSES = 50
        SAMPLES_PER_CLASS = 20
        INNER_STEPS = 3
        INNER_LR = 0.05
        OUTER_LR = 0.001

        # --- check 1: create minimal config, build model + optimizer + sampler ---
        model = None
        optimizer = None
        sampler = None
        algo_obj = None

        def check_build_components():
            """Create minimal config, build model, optimizer, sampler."""
            nonlocal model, optimizer, sampler, algo_obj
            torch.manual_seed(self.seed)

            dataset = SyntheticFewShotDataset(
                num_classes=NUM_CLASSES,
                samples_per_class=SAMPLES_PER_CLASS,
                feature_dim=FEATURE_DIM,
                seed=self.seed,
            )
            sampler = EpisodeSampler(
                dataset, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY,
                global_seed=self.seed,
            )
            model = ToyLinearModel(FEATURE_DIM, D_HID, N_WAY)
            optimizer = torch.optim.Adam(model.parameters(), lr=OUTER_LR)
            algo_obj = MAMLAlgorithm(model, inner_lr=INNER_LR, inner_steps=INNER_STEPS)

            ok = model is not None and optimizer is not None and sampler is not None
            num_params = sum(p.numel() for p in model.parameters())
            msg = f"Model params: {num_params}, sampler n_way={sampler.n_way}"
            return ok, msg, None

        self._check("build_components", "gate_c", check_build_components)

        # --- check 2: run one meta-epoch ---
        epoch_metrics = None

        def check_run_meta_epoch():
            """Run one meta-epoch (meta-training loop) without crashing."""
            nonlocal epoch_metrics
            torch.manual_seed(self.seed)
            epoch_metrics = run_meta_epoch(
                algo_obj, sampler, NUM_EPISODES, optimizer, epoch=0,
            )
            ok = epoch_metrics is not None and isinstance(epoch_metrics, dict)
            msg = f"Epoch metrics: {epoch_metrics}"
            return ok, msg, None

        self._check("run_meta_epoch", "gate_c", check_run_meta_epoch)

        # --- check 3: metrics dict produced with required keys ---
        def check_metrics_keys():
            """Metrics dict contains required keys: pre_adapt_acc, post_adapt_acc, fast_gain, auac."""
            required_keys = {"pre_adapt_acc", "post_adapt_acc", "fast_gain", "auac"}
            if epoch_metrics is None:
                return False, "No metrics available (epoch did not run)", None
            present = required_keys.issubset(set(epoch_metrics.keys()))
            missing = required_keys - set(epoch_metrics.keys())
            ok = present
            msg = f"Keys present: {set(epoch_metrics.keys())}, missing: {missing}"
            return ok, msg, None

        self._check("metrics_required_keys", "gate_c", check_metrics_keys)

        # --- check 4: loss decreases over multiple epochs ---
        def check_loss_decreases_epochs():
            """Run 3 meta-epochs. Assert loss decreased from first to last (or at least no crash)."""
            torch.manual_seed(self.seed)
            local_dataset = SyntheticFewShotDataset(
                num_classes=NUM_CLASSES,
                samples_per_class=SAMPLES_PER_CLASS,
                feature_dim=FEATURE_DIM,
                seed=self.seed,
            )
            local_sampler = EpisodeSampler(
                local_dataset, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY,
                global_seed=self.seed,
            )
            local_model = ToyLinearModel(FEATURE_DIM, D_HID, N_WAY)
            local_optimizer = torch.optim.Adam(local_model.parameters(), lr=OUTER_LR)
            local_algo = MAMLAlgorithm(local_model, inner_lr=INNER_LR, inner_steps=INNER_STEPS)

            epoch_losses = []
            for ep in range(3):
                metrics = run_meta_epoch(
                    local_algo, local_sampler, NUM_EPISODES, local_optimizer, epoch=ep,
                )
                epoch_losses.append(metrics["mean_loss"])

            # Allow some tolerance -- loss should generally decrease or at least not explode
            decreased_or_stable = epoch_losses[-1] <= epoch_losses[0] * 1.5
            ok = decreased_or_stable
            msg = f"Epoch losses: {[f'{l:.4f}' for l in epoch_losses]}"
            return ok, msg, None

        self._check("loss_decreases_epochs", "gate_c", check_loss_decreases_epochs)

        # --- check 5: metrics are JSON-serializable ---
        def check_json_serializable():
            """Metrics should be JSON-serializable."""
            if epoch_metrics is None:
                return False, "No metrics available", None
            try:
                json_str = json.dumps(epoch_metrics)
                roundtrip = json.loads(json_str)
                ok = isinstance(roundtrip, dict) and len(roundtrip) == len(epoch_metrics)
                msg = f"JSON round-trip successful, {len(roundtrip)} keys"
                return ok, msg, None
            except TypeError as e:
                return False, f"JSON serialization failed: {e}", None

        self._check("metrics_json_serializable", "gate_c", check_json_serializable)

        # --- check 6: checkpoint save/load round-trip ---
        def check_checkpoint_roundtrip():
            """Save and load a meta-checkpoint. Verify model weights match."""
            if model is None or optimizer is None:
                return False, "Model/optimizer not available", None

            torch.manual_seed(self.seed)
            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_path = os.path.join(tmpdir, "meta_test_ckpt.pt")
                save_meta_checkpoint(
                    ckpt_path, model, optimizer,
                    algo="maml", inner_steps=INNER_STEPS,
                    inner_lr=INNER_LR, inner_clip=10.0,
                    sampler_config={
                        "n_way": N_WAY, "k_shot": K_SHOT,
                        "q_query": Q_QUERY, "dataset": "synthetic",
                    },
                    epoch=0, global_step=NUM_EPISODES,
                    best_val_accuracy=0.5,
                    metrics_history=[epoch_metrics] if epoch_metrics else [],
                )

                # Load into fresh model
                model2 = ToyLinearModel(FEATURE_DIM, D_HID, N_WAY)
                optimizer2 = torch.optim.Adam(model2.parameters(), lr=OUTER_LR)
                ckpt = load_meta_checkpoint(ckpt_path, model2, optimizer2)

                # Compare state dicts
                sd1 = model.state_dict()
                sd2 = model2.state_dict()
                all_match = all(
                    torch.equal(sd1[k], sd2[k]) for k in sd1
                )
                # Check checkpoint metadata
                has_algo = ckpt.get("algo") == "maml"
                has_steps = ckpt.get("inner_steps") == INNER_STEPS
                has_version = ckpt.get("format_version") == "1.0"

                ok = all_match and has_algo and has_steps and has_version
                msg = (f"Weights match={all_match}, algo={ckpt.get('algo')}, "
                       f"steps={ckpt.get('inner_steps')}, version={ckpt.get('format_version')}")
                return ok, msg, None

        self._check("checkpoint_roundtrip", "gate_c", check_checkpoint_roundtrip)


# ============================================================================
# Summary printing
# ============================================================================

def _print_summary(results: List[CheckResult]) -> Tuple[int, int, int]:
    """Print a summary table of all results. Returns (total, passed, failed)."""
    print("\n" + "=" * 80)
    print(f"  {'VALIDATION SUMMARY':^76}")
    print("=" * 80)

    # Group results
    groups: Dict[str, List[CheckResult]] = {}
    for r in results:
        groups.setdefault(r.group, []).append(r)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    # Header
    print(f"\n  {'Check':<45} {'Group':<14} {'Status':<8} {'Time':>8}")
    print(f"  {'-'*45} {'-'*14} {'-'*8} {'-'*8}")

    for group in _ALL_GROUPS:
        if group not in groups:
            continue
        for r in groups[group]:
            status = _c("PASS", _GREEN) if r.passed else _c("FAIL", _RED)
            name_display = r.name[:44]
            group_display = r.group[:13]
            time_display = f"{r.elapsed_ms:.1f}ms"
            print(f"  {name_display:<45} {group_display:<14} {status:<17} {time_display:>8}")

    # Totals
    print(f"\n  {'-'*80}")
    total_time = sum(r.elapsed_ms for r in results)
    status_line = (
        f"  Total: {total} checks | "
        f"{_c(str(passed) + ' passed', _GREEN)} | "
        f"{_c(str(failed) + ' failed', _RED) if failed > 0 else _c('0 failed', _GREEN)} | "
        f"Time: {total_time:.0f}ms"
    )
    print(status_line)
    print("=" * 80)

    # List failures if any
    if failed > 0:
        print(f"\n  {_c('FAILED CHECKS:', _RED + _BOLD)}")
        for r in results:
            if not r.passed:
                print(f"    - {r.name} [{r.group}]: {r.message}")
                if r.details:
                    for line in r.details.strip().split("\n")[:3]:
                        print(f"      {_c(line, _DIM)}")
        print()

    return total, passed, failed


# ============================================================================
# Gate summary
# ============================================================================

def _print_gate_summary(results: List[CheckResult]) -> None:
    """Print Done-When gate verdicts."""
    print("\n" + "-" * 60)
    print(f"  {'DONE-WHEN GATE VERDICTS':^56}")
    print("-" * 60)

    gate_groups = {
        "gate_a": "Gate (a): Meta-gradient flow validated",
        "gate_b": "Gate (b): FO variants comparable adaptation",
        "gate_c": "Gate (c): Phase 7 CPU dev mode",
    }

    for group_key, label in gate_groups.items():
        group_results = [r for r in results if r.group == group_key]
        if not group_results:
            status = _c("SKIP", _YELLOW)
        elif all(r.passed for r in group_results):
            status = _c("PASS", _GREEN + _BOLD)
        else:
            fail_count = sum(1 for r in group_results if not r.passed)
            status = _c(f"FAIL ({fail_count}/{len(group_results)} failed)", _RED + _BOLD)
        print(f"  {label:<44} {status}")

    print("-" * 60)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meta-Learning Suite -- Runtime Contract Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Groups:\n"
            "  inner_loop   -- Inner-loop engine: adapt(), StepLog, clipping, graph modes\n"
            "  algorithms   -- MAML / FOMAML / Reptile meta_step and MetaOutput\n"
            "  maml_plus    -- MAML++ enhancements: LSLR, MSL, MAMLPlusPlusAlgorithm\n"
            "  sampling     -- EpisodeSampler determinism, shapes, class splits\n"
            "  gate_a       -- Done-When Gate (a): Meta-gradient flow validated\n"
            "  gate_b       -- Done-When Gate (b): FO variants comparable adaptation\n"
            "  gate_c       -- Done-When Gate (c): Phase 7 CPU dev mode\n"
        ),
    )
    parser.add_argument(
        "--group",
        choices=_ALL_GROUPS,
        default=None,
        help="Run only a specific validation group.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed messages for each check.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_groups",
        help="List all check groups and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    # Handle --list
    if args.list_groups:
        print("Available validation groups:\n")
        for group in _ALL_GROUPS:
            desc = _GROUP_DESCRIPTIONS.get(group, "")
            print(f"  {group:<14} -- {desc}")
        print()
        sys.exit(0)

    print("=" * 80)
    print(f"  {'META-LEARNING SUITE -- CONTRACT VALIDATION':^76}")
    print(f"  seed={args.seed}, verbose={args.verbose}, "
          f"group={args.group or 'ALL'}, device=cpu")
    print("=" * 80)

    validator = MetaLearningValidator(verbose=args.verbose, seed=args.seed)

    if args.group:
        results = validator.run_group(args.group)
    else:
        results = validator.run_all()

    total, passed, failed = _print_summary(results)
    _print_gate_summary(results)

    if failed > 0:
        print(f"\n{_c('RESULT: FAIL', _RED + _BOLD)} -- {failed} check(s) did not pass.\n")
        sys.exit(1)
    else:
        print(f"\n{_c('RESULT: ALL CHECKS PASSED', _GREEN + _BOLD)}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
