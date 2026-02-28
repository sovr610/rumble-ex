#!/usr/bin/env python3
"""Meta-Learning Suite -- Performance Benchmark

Benchmarks inner-loop throughput, algorithm comparison, and episode sampling.

Usage:
    python meta_benchmark.py                       # Run all benchmarks
    python meta_benchmark.py --suite inner_loop    # Run specific suite
    python meta_benchmark.py --device cuda         # Use GPU
    python meta_benchmark.py --output results.json # Save results to JSON
    python meta_benchmark.py --list                # List all benchmark suites

Benchmark Suites:
    inner_loop       Inner-loop engine throughput (steps/sec)
    fo_vs_so         First-order vs second-order comparison
    algorithms       MAML vs FOMAML vs Reptile speed comparison
    episode_sampling Episode sampling throughput
    lslr_overhead    LSLR parameter overhead measurement
    msl_overhead     Multi-step loss overhead measurement
    memory_profile   Peak memory usage per algorithm
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import struct
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_SUITES = [
    "inner_loop",
    "fo_vs_so",
    "algorithms",
    "episode_sampling",
    "lslr_overhead",
    "msl_overhead",
    "memory_profile",
]

_SEP = "=" * 72


# ============================================================================
# Inline Stubs: Data Containers
# ============================================================================

@dataclass
class Episode:
    """A single few-shot episode with support and query splits.

    Attributes:
        support_x: Support set inputs, shape (N*K, input_dim).
        support_y: Support set labels, shape (N*K,).
        query_x: Query set inputs, shape (N*Q, input_dim).
        query_y: Query set labels, shape (N*Q,).
        n_way: Number of classes in the episode.
        k_shot: Support examples per class.
        q_query: Query examples per class.
    """
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    n_way: int = 5
    k_shot: int = 1
    q_query: int = 15


@dataclass
class TaskBatch:
    """A batch of episodes for a single meta-step.

    Attributes:
        episodes: List of Episode instances comprising the meta-batch.
    """
    episodes: List[Episode]

    @property
    def num_tasks(self) -> int:
        return len(self.episodes)


@dataclass
class StepLog:
    """Per-inner-step diagnostic record.

    Attributes:
        step: Inner-loop step index.
        loss: Support set loss at this step.
        accuracy: Support set accuracy at this step.
        grad_norm: L2 norm of gradients before clipping.
        update_norm: L2 norm of the parameter update.
        lr_used: Learning rate applied at this step.
    """
    step: int
    loss: float
    accuracy: float
    grad_norm: float
    update_norm: float
    lr_used: float


@dataclass
class MetaOutput:
    """Output of a meta-learning step.

    Attributes:
        loss: Meta-objective scalar (outer loss over tasks).
        metrics: Dictionary with pre_adapt_acc, post_adapt_acc, fast_gain.
        inner_logs: Per-task list of per-step StepLog records.
        adapted_params: Per-task adapted parameter dictionaries.
    """
    loss: torch.Tensor
    metrics: Dict[str, float]
    inner_logs: List[List[StepLog]]
    adapted_params: List[Dict[str, torch.Tensor]]


# ============================================================================
# Inline Stubs: Toy Models
# ============================================================================

class ToyLinearModel(nn.Module):
    """Simple 3-layer MLP for benchmarking meta-learning throughput.

    Architecture: input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> output

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Number of output classes.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64,
                 output_dim: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ToyConv4Model(nn.Module):
    """Conv4 backbone commonly used in few-shot learning benchmarks.

    4 convolutional blocks with 3x3 kernels, batch norm (no running stats),
    ReLU, and 2x2 max pooling. Operates on 1-channel 28x28 images by default.
    After 3 pool layers + adaptive avg pool, the feature map is 1x1.

    Args:
        in_channels: Number of input channels (1 for grayscale).
        nf: Number of filters per convolutional block.
        output_dim: Number of output classes.
    """

    def __init__(self, in_channels: int = 1, nf: int = 64,
                 output_dim: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, nf, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(nf, track_running_stats=False)
        self.conv2 = nn.Conv2d(nf, nf, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(nf, track_running_stats=False)
        self.conv3 = nn.Conv2d(nf, nf, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(nf, track_running_stats=False)
        self.conv4 = nn.Conv2d(nf, nf, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(nf, track_running_stats=False)
        self.classifier = nn.Linear(nf, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                         (self.conv3, self.bn3)]:
            x = F.max_pool2d(F.relu(bn(conv(x))), 2)
        x = F.adaptive_avg_pool2d(F.relu(self.bn4(self.conv4(x))), 1)
        return self.classifier(x.view(x.size(0), -1))


# ============================================================================
# Inline Stub: CustomSGDEngine (differentiable inner-loop)
# ============================================================================

class CustomSGDEngine:
    """Simplified differentiable inner-loop engine using torch.autograd.grad.

    Supports both first-order (create_graph=False) and second-order
    (create_graph=True) adaptation. All inner-loop computation runs in fp32
    for meta-gradient stability.

    The engine uses torch.func.functional_call when available (PyTorch 2.0+)
    and falls back to manual F.linear calls for ToyLinearModel.

    Args:
        model: The nn.Module whose structure is used for functional forward.
        create_graph: Whether to retain computation graph (MAML vs FOMAML).
        clip_norm: Optional gradient clipping norm per inner step.
    """

    def __init__(self, model: nn.Module, create_graph: bool = True,
                 clip_norm: Optional[float] = None):
        self.model = model
        self.create_graph = create_graph
        self.clip_norm = clip_norm

    def _functional_forward(self, params: Dict[str, torch.Tensor],
                            x: torch.Tensor) -> torch.Tensor:
        """Run forward pass with the given parameter dictionary.

        Tries torch.func.functional_call first, then falls back to manual
        F.linear calls for ToyLinearModel compatibility.
        """
        try:
            from torch.func import functional_call
            return functional_call(self.model, params, (x,))
        except (ImportError, AttributeError):
            # Manual fallback for ToyLinearModel
            h = F.relu(F.linear(x, params['fc1.weight'], params.get('fc1.bias')))
            h = F.relu(F.linear(h, params['fc2.weight'], params.get('fc2.bias')))
            return F.linear(h, params['fc3.weight'], params.get('fc3.bias'))

    def adapt(self, params: Dict[str, torch.Tensor],
              support_x: torch.Tensor, support_y: torch.Tensor,
              steps: int, lr: float,
              lslr_lrs: Optional[Dict[str, torch.Tensor]] = None
              ) -> Tuple[Dict[str, torch.Tensor], List[StepLog]]:
        """Run the inner loop for the given number of gradient steps.

        Args:
            params: Initial parameter dictionary (will not be mutated).
            support_x: Support set inputs.
            support_y: Support set labels.
            steps: Number of inner-loop gradient steps.
            lr: Base inner-loop learning rate.
            lslr_lrs: Optional per-layer per-step learned LRs from LSLRModule.

        Returns:
            Tuple of (adapted_params, step_logs).
        """
        adapted = {k: (v if self.create_graph else v.clone())
                   for k, v in params.items()}
        logs: List[StepLog] = []

        for si in range(steps):
            # Forward pass and loss
            logits = self._functional_forward(adapted, support_x)
            loss = F.cross_entropy(logits, support_y)

            # Accuracy (detached)
            with torch.no_grad():
                acc = (logits.argmax(-1) == support_y).float().mean().item()

            # Gradients
            grads = torch.autograd.grad(
                loss, list(adapted.values()),
                create_graph=self.create_graph,
                allow_unused=True,
            )

            # Gradient norm and optional clipping
            gn = sum(g.data.norm(2).item() ** 2
                     for g in grads if g is not None) ** 0.5
            if self.clip_norm and gn > self.clip_norm:
                c = self.clip_norm / (gn + 1e-8)
                grads = tuple(g * c if g is not None else None for g in grads)

            # Parameter update with optional per-layer per-step LRs
            un, new = 0.0, {}
            for (name, param), g in zip(adapted.items(), grads):
                if g is None:
                    new[name] = param
                    continue
                if lslr_lrs and name in lslr_lrs:
                    slr = lslr_lrs[name]
                    elr = slr[si] if slr.dim() > 0 and si < slr.shape[0] else slr
                    upd = elr * g
                else:
                    upd = lr * g
                un += upd.data.norm(2).item() ** 2
                new[name] = param - upd

            adapted = new
            logs.append(StepLog(si, loss.item(), acc, gn, un ** 0.5, lr))

        return adapted, logs


# ============================================================================
# Inline Stubs: MAML / FOMAML / Reptile Algorithms
# ============================================================================

def _meta_step_common(model, params, task_batch, inner_steps, inner_lr,
                      clip_norm, create_graph, detach_adapted):
    """Shared implementation for MAML and FOMAML meta-steps.

    Iterates over tasks in the batch, adapts parameters on the support set,
    then evaluates the query loss. The key difference between MAML and FOMAML
    is controlled by create_graph and detach_adapted.

    Args:
        model: Model architecture for functional forward.
        params: Base parameter dictionary (theta).
        task_batch: TaskBatch of episodes.
        inner_steps: Number of inner-loop SGD steps.
        inner_lr: Inner-loop learning rate.
        clip_norm: Optional gradient clipping norm.
        create_graph: True for MAML (second-order), False for FOMAML.
        detach_adapted: True for FOMAML (detach then reattach grad).

    Returns:
        MetaOutput with loss, metrics, logs, and adapted parameters.
    """
    engine = CustomSGDEngine(model, create_graph=create_graph, clip_norm=clip_norm)
    dev = next(iter(params.values())).device
    meta_loss = torch.tensor(0.0, device=dev, requires_grad=True)
    all_logs, all_adapted, pre_accs, post_accs = [], [], [], []

    for ep in task_batch.episodes:
        adapted, logs = engine.adapt(
            params, ep.support_x, ep.support_y,
            steps=inner_steps, lr=inner_lr,
        )
        all_logs.append(logs)

        # Pre-adaptation accuracy on query set
        with torch.no_grad():
            pre_logits = engine._functional_forward(params, ep.query_x)
            pre_accs.append(
                (pre_logits.argmax(-1) == ep.query_y).float().mean().item()
            )

        # Optionally detach adapted params (FOMAML vs MAML)
        if detach_adapted:
            fwd_params = {k: v.detach().requires_grad_(True)
                          for k, v in adapted.items()}
            all_adapted.append(fwd_params)
        else:
            fwd_params = adapted
            all_adapted.append(adapted)

        # Query loss
        q_logits = engine._functional_forward(fwd_params, ep.query_x)
        q_loss = F.cross_entropy(q_logits, ep.query_y)
        meta_loss = meta_loss + q_loss

        # Post-adaptation accuracy
        with torch.no_grad():
            post_accs.append(
                (q_logits.argmax(-1) == ep.query_y).float().mean().item()
            )

    meta_loss = meta_loss / len(task_batch.episodes)
    n = len(pre_accs)
    return MetaOutput(
        loss=meta_loss,
        metrics={
            "pre_adapt_acc": sum(pre_accs) / n,
            "post_adapt_acc": sum(post_accs) / n,
            "fast_gain": (sum(post_accs) - sum(pre_accs)) / n,
        },
        inner_logs=all_logs,
        adapted_params=all_adapted,
    )


def maml_meta_step(model, params, task_batch, inner_steps, inner_lr,
                   clip_norm=None):
    """MAML meta-step: second-order meta-gradients through inner loop.

    Retains the full computation graph (create_graph=True) so that
    meta_loss.backward() produces gradients that include Hessian-vector
    products through the inner-loop trajectory.
    """
    return _meta_step_common(model, params, task_batch, inner_steps, inner_lr,
                             clip_norm, create_graph=True, detach_adapted=False)


def fomaml_meta_step(model, params, task_batch, inner_steps, inner_lr,
                     clip_norm=None):
    """FOMAML meta-step: first-order approximation (no graph through inner loop).

    Same inner-loop adaptation as MAML but with create_graph=False. The adapted
    parameters are detached and re-attached for the query loss computation.
    """
    return _meta_step_common(model, params, task_batch, inner_steps, inner_lr,
                             clip_norm, create_graph=False, detach_adapted=True)


def reptile_meta_step(model, params, task_batch, inner_steps, inner_lr,
                      meta_lr=0.001, clip_norm=None):
    """Reptile meta-step: weight-space interpolation toward adapted params.

    No explicit meta-gradient through the inner loop. The outer update is:
        theta <- theta + meta_lr * (1/B) * sum_i (phi_i - theta)

    Returns a pseudo-loss (sum of absolute deltas) for a consistent API.
    """
    engine = CustomSGDEngine(model, create_graph=False, clip_norm=clip_norm)
    all_logs, all_adapted = [], []
    pre_accs, post_accs = [], []
    delta = {k: torch.zeros_like(v) for k, v in params.items()}

    for ep in task_batch.episodes:
        adapted, logs = engine.adapt(
            {k: v.clone() for k, v in params.items()},
            ep.support_x, ep.support_y,
            steps=inner_steps, lr=inner_lr,
        )
        all_logs.append(logs)
        all_adapted.append(adapted)

        with torch.no_grad():
            pre_accs.append(
                (engine._functional_forward(params, ep.query_x)
                 .argmax(-1) == ep.query_y).float().mean().item()
            )
            post_accs.append(
                (engine._functional_forward(adapted, ep.query_x)
                 .argmax(-1) == ep.query_y).float().mean().item()
            )

        for k in params:
            delta[k] += adapted[k].detach() - params[k].detach()

    for k in delta:
        delta[k] /= len(task_batch.episodes)

    n = max(len(pre_accs), 1)
    return MetaOutput(
        loss=sum(d.abs().sum() for d in delta.values()),
        metrics={
            "pre_adapt_acc": sum(pre_accs) / n,
            "post_adapt_acc": sum(post_accs) / n,
            "fast_gain": (sum(post_accs) - sum(pre_accs)) / n,
            "mean_delta_norm": sum(d.norm().item() for d in delta.values()) / max(len(delta), 1),
        },
        inner_logs=all_logs,
        adapted_params=all_adapted,
    )


# ============================================================================
# Inline Stubs: LSLR & MultiStepLoss (MAML++ Enhancements)
# ============================================================================

class LSLRModule(nn.Module):
    """Per-Layer Per-Step Learned Learning Rates for MAML++.

    Creates a learnable learning rate tensor of shape (num_steps,) for each
    named parameter in the model. Learning rates are clamped to [min_lr, max_lr]
    to prevent collapse to zero or divergence.

    Args:
        named_params: List of (name, param) tuples from model.named_parameters().
        num_steps: Number of inner-loop steps.
        init_lr: Initial value for all learning rates.
        min_lr: Minimum learning rate (lower clamp).
        max_lr: Maximum learning rate (upper clamp).
    """

    def __init__(self, named_params: List[Tuple[str, torch.Tensor]],
                 num_steps: int = 5, init_lr: float = 0.01,
                 min_lr: float = 1e-6, max_lr: float = 1.0):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_params = nn.ParameterDict()
        self._name_map: Dict[str, str] = {}

        for name, _ in named_params:
            safe = name.replace(".", "_")
            self.lr_params[safe] = nn.Parameter(torch.full((num_steps,), init_lr))
            self._name_map[name] = safe

    def get_lrs(self) -> Dict[str, torch.Tensor]:
        """Return clamped per-layer per-step learning rates.

        Returns:
            Dict mapping original param names to (num_steps,) LR tensors.
        """
        return {orig: self.lr_params[safe].clamp(self.min_lr, self.max_lr)
                for orig, safe in self._name_map.items()}


class MultiStepLoss(nn.Module):
    """Multi-Step Loss accumulation for MAML++.

    Instead of computing the query loss only at the final inner step, MSL
    evaluates the query loss at every inner step and combines them with
    learned or fixed weights (uniform, linear_increase, or learned via softmax).

    Args:
        num_steps: Number of inner-loop steps.
        weight_mode: One of 'uniform', 'linear_increase', 'learned'.
    """

    def __init__(self, num_steps: int = 5, weight_mode: str = "uniform"):
        super().__init__()
        self.num_steps = num_steps
        self.weight_mode = weight_mode
        if weight_mode == "learned":
            self.raw_weights = nn.Parameter(torch.zeros(num_steps))
        else:
            self.register_buffer("raw_weights", torch.zeros(num_steps))

    def get_weights(self) -> torch.Tensor:
        """Compute normalized step weights summing to 1.0."""
        if self.weight_mode == "uniform":
            return torch.ones(self.num_steps) / self.num_steps
        elif self.weight_mode == "linear_increase":
            w = torch.arange(1, self.num_steps + 1, dtype=torch.float32)
            return w / w.sum()
        else:  # learned
            return F.softmax(self.raw_weights, dim=0)

    def compute_msl(self, model: nn.Module, engine: CustomSGDEngine,
                    params: Dict[str, torch.Tensor], episode: Episode,
                    inner_steps: int, inner_lr: float,
                    lslr_lrs: Optional[Dict[str, torch.Tensor]] = None,
                    ) -> Tuple[torch.Tensor, List[StepLog]]:
        """Compute weighted query loss at every inner step.

        At each inner step, performs one gradient update on the support set,
        then evaluates the query loss. The final loss is the weighted sum
        of all per-step query losses.

        Args:
            model: Model architecture (used by engine for functional forward).
            engine: CustomSGDEngine configured for the desired order.
            params: Initial parameter dictionary.
            episode: Episode with support and query splits.
            inner_steps: Number of adaptation steps.
            inner_lr: Base learning rate.
            lslr_lrs: Optional LSLR per-layer per-step learning rates.

        Returns:
            Tuple of (weighted_total_loss, step_logs).
        """
        weights = self.get_weights().to(next(iter(params.values())).device)
        adapted = dict(params)
        total_loss = torch.tensor(0.0, device=weights.device)
        logs: List[StepLog] = []

        for si in range(inner_steps):
            # Support loss and gradient
            logits = engine._functional_forward(adapted, episode.support_x)
            sup_loss = F.cross_entropy(logits, episode.support_y)
            grads = torch.autograd.grad(
                sup_loss, list(adapted.values()),
                create_graph=engine.create_graph, allow_unused=True,
            )
            gn = sum(g.data.norm(2).item() ** 2
                     for g in grads if g is not None) ** 0.5

            # Parameter update
            new, un = {}, 0.0
            for (nm, p), g in zip(adapted.items(), grads):
                if g is None:
                    new[nm] = p
                    continue
                if lslr_lrs and nm in lslr_lrs:
                    slr = lslr_lrs[nm]
                    upd = (slr[si] if slr.dim() > 0 and si < slr.shape[0]
                           else slr) * g
                else:
                    upd = inner_lr * g
                un += upd.data.norm(2).item() ** 2
                new[nm] = p - upd
            adapted = new

            # Query loss at this step
            ql = engine._functional_forward(adapted, episode.query_x)
            qloss = F.cross_entropy(ql, episode.query_y)
            with torch.no_grad():
                acc = (ql.argmax(-1) == episode.query_y).float().mean().item()

            w = weights[si] if si < len(weights) else weights[-1]
            total_loss = total_loss + w * qloss
            logs.append(StepLog(si, qloss.item(), acc, gn, un ** 0.5, inner_lr))

        return total_loss, logs


# ============================================================================
# Inline Stubs: Episode Sampler & SyntheticFewShotDataset
# ============================================================================

class SyntheticFewShotDataset:
    """Generates synthetic few-shot classification episodes from Gaussian prototypes.

    Creates a pool of random class prototypes. Episodes sample N classes and
    generate K+Q examples per class as prototype + Gaussian noise.

    Args:
        num_classes: Total number of classes in the pool.
        input_dim: Dimensionality of input features.
        examples_per_class: Not used directly (examples generated on-the-fly).
        noise_std: Standard deviation of Gaussian noise around prototypes.
        seed: Random seed for prototype generation.
        device: Torch device for all tensors.
    """

    def __init__(self, num_classes: int = 100, input_dim: int = 10,
                 examples_per_class: int = 20, noise_std: float = 0.5,
                 seed: int = 42, device: str = "cpu"):
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.noise_std = noise_std
        self.device = torch.device(device)

        gen = torch.Generator()
        gen.manual_seed(seed)
        self.prototypes = torch.randn(
            num_classes, input_dim, generator=gen, device=self.device
        )

    def sample_episode(self, n_way: int = 5, k_shot: int = 1,
                       q_query: int = 15, seed: Optional[int] = None
                       ) -> Episode:
        """Sample a single episode with deterministic seeding.

        Args:
            n_way: Number of classes in the episode.
            k_shot: Support examples per class.
            q_query: Query examples per class.
            seed: Optional seed for deterministic episode generation.

        Returns:
            Episode with support_x, support_y, query_x, query_y.
        """
        gen = torch.Generator()
        gen.manual_seed(
            seed if seed is not None
            else int(time.perf_counter() * 1e9) % (2 ** 31)
        )

        # Select N classes uniformly without replacement
        perm = torch.randperm(self.num_classes, generator=gen)[:n_way]

        sx, sy, qx, qy = [], [], [], []
        for lbl, ci in enumerate(perm):
            n = k_shot + q_query
            examples = (
                self.prototypes[ci].unsqueeze(0).expand(n, -1)
                + self.noise_std * torch.randn(
                    n, self.input_dim, generator=gen, device=self.device
                )
            )
            sx.append(examples[:k_shot])
            qx.append(examples[k_shot:])
            sy.extend([lbl] * k_shot)
            qy.extend([lbl] * q_query)

        return Episode(
            support_x=torch.cat(sx, dim=0),
            support_y=torch.tensor(sy, dtype=torch.long, device=self.device),
            query_x=torch.cat(qx, dim=0),
            query_y=torch.tensor(qy, dtype=torch.long, device=self.device),
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
        )


class EpisodeSampler:
    """Deterministic episode sampler keyed to (global_seed, epoch, episode_idx).

    Ensures every episode is exactly reproducible from its index alone.
    Uses SHA-256 hashing of (global_seed, epoch, episode_idx) for the
    per-episode seed computation.

    Args:
        dataset: SyntheticFewShotDataset to sample from.
        global_seed: Base seed for the experiment.
    """

    def __init__(self, dataset: SyntheticFewShotDataset,
                 global_seed: int = 42):
        self.dataset = dataset
        self.global_seed = global_seed

    def _episode_seed(self, epoch: int, episode_idx: int) -> int:
        """Compute deterministic seed from (global_seed, epoch, episode_idx)."""
        raw = struct.pack(">III", self.global_seed, epoch, episode_idx)
        digest = hashlib.sha256(raw).digest()
        return int.from_bytes(digest[:4], "big")

    def sample(self, epoch: int, episode_idx: int,
               n_way: int = 5, k_shot: int = 1, q_query: int = 15
               ) -> Episode:
        """Sample a deterministic episode."""
        seed = self._episode_seed(epoch, episode_idx)
        return self.dataset.sample_episode(
            n_way=n_way, k_shot=k_shot, q_query=q_query, seed=seed
        )


# ============================================================================
# Benchmark Infrastructure
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result for a single benchmark measurement.

    Attributes:
        name: Human-readable benchmark name.
        metric: Unit string ("ms", "ms/step", "ms/meta-step", "MB", "x", "%").
        value: The measured value.
        details: Extra metadata (std, min, max, repeats, etc.).
    """
    name: str
    metric: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)


class BenchmarkSuite:
    """Core benchmarking harness with warmup/repeat timing and memory tracking.

    Args:
        name: Name of this benchmark suite.
        device: Torch device string ("cpu" or "cuda").
    """

    def __init__(self, name: str, device: str = "cpu"):
        self.name = name
        self.device = torch.device(device)
        self.results: List[BenchmarkResult] = []

    def bench(self, fn, name: str, metric: str = "ms",
              warmup: int = 3, repeats: int = 10) -> float:
        """Time a callable with warmup and averaging.

        Args:
            fn: Zero-argument callable to benchmark.
            name: Human-readable name for this measurement.
            metric: Unit string for reporting.
            warmup: Number of warmup calls (results discarded).
            repeats: Number of timed calls.

        Returns:
            Mean time in seconds.
        """
        for _ in range(warmup):
            fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        times: List[float] = []
        for _ in range(repeats):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        mean_t = sum(times) / len(times)
        std_t = (sum((t - mean_t) ** 2 for t in times) / len(times)) ** 0.5
        self.results.append(BenchmarkResult(
            name=name, metric=metric, value=mean_t * 1000,
            details={"std_ms": std_t * 1000, "min_ms": min(times) * 1000,
                     "max_ms": max(times) * 1000, "repeats": repeats},
        ))
        return mean_t

    def record(self, name: str, metric: str, value: float,
               details: Optional[Dict[str, Any]] = None) -> None:
        """Record a pre-computed benchmark result."""
        self.results.append(BenchmarkResult(
            name=name, metric=metric, value=value, details=details or {},
        ))

    def peak_mem_mb(self) -> float:
        """Get peak memory in MB (GPU max_memory_allocated or CPU RSS)."""
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return 0.0

    def reset_mem(self) -> None:
        """Reset GPU peak memory stats and run garbage collection."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()


# ============================================================================
# Helpers
# ============================================================================

def _make_tb(ds: SyntheticFewShotDataset, num_tasks: int = 4,
             n_way: int = 5, k_shot: int = 1, q_query: int = 15,
             seed_base: int = 0) -> TaskBatch:
    """Create a TaskBatch from the synthetic dataset."""
    return TaskBatch([
        ds.sample_episode(n_way, k_shot, q_query, seed=seed_base + i)
        for i in range(num_tasks)
    ])


def _make_mp(hidden_dim: int = 64, input_dim: int = 10,
             output_dim: int = 5, device: str = "cpu"
             ) -> Tuple[ToyLinearModel, Dict[str, torch.Tensor]]:
    """Create a ToyLinearModel and extract its parameter dict."""
    model = ToyLinearModel(input_dim, hidden_dim, output_dim).to(device)
    params = {n: p.clone().detach().requires_grad_(True)
              for n, p in model.named_parameters()}
    return model, params


def _fresh_p(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Clone and detach a parameter dict, re-enabling gradients."""
    return {k: v.clone().detach().requires_grad_(True) for k, v in params.items()}


# ============================================================================
# Suite 1: Inner-Loop Throughput
# ============================================================================

def bench_inner_loop(device: str = "cpu") -> BenchmarkSuite:
    """Benchmark inner-loop engine throughput across configurations.

    Measures:
        - Per-step time for 1, 3, 5, 10 inner steps
        - Throughput with vs without gradient clipping
        - AMP safety overhead (CUDA) or CPU baseline
        - Scaling with model hidden dim (10, 100, 1000)
        - Scaling with support set size (k_shot = 1, 5, 10, 20)
    """
    suite = BenchmarkSuite("inner_loop", device)
    idim, odim = 10, 5
    ds = SyntheticFewShotDataset(num_classes=50, input_dim=idim, seed=42, device=device)
    model, params = _make_mp(64, idim, odim, device)
    engine = CustomSGDEngine(model, create_graph=True)
    print("  Running inner-loop throughput benchmarks...")

    # Varying inner steps
    for ns in [1, 3, 5, 10]:
        ep = ds.sample_episode(n_way=odim, k_shot=1, seed=100)
        def _r(ns=ns, e=ep):
            engine.adapt(_fresh_p(params), e.support_x, e.support_y, steps=ns, lr=0.01)
        t = suite.bench(_r, f"inner_loop_{ns}step", "ms/step", warmup=3, repeats=10)
        suite.results[-1].value = (t * 1000) / ns
        suite.results[-1].details.update(total_ms=t * 1000, inner_steps=ns)

    # With vs without gradient clipping
    ep = ds.sample_episode(n_way=odim, k_shot=1, seed=200)
    ec = CustomSGDEngine(model, True, clip_norm=10.0)
    en = CustomSGDEngine(model, True, clip_norm=None)

    def _clip():
        ec.adapt(_fresh_p(params), ep.support_x, ep.support_y, 5, 0.01)
    def _noclip():
        en.adapt(_fresh_p(params), ep.support_x, ep.support_y, 5, 0.01)

    suite.bench(_clip, "inner_loop_with_clip", "ms")
    suite.bench(_noclip, "inner_loop_no_clip", "ms")

    # AMP safety (CUDA) or CPU baseline
    if device == "cuda" and torch.cuda.is_available():
        def _amp():
            with torch.amp.autocast("cuda", enabled=True):
                with torch.amp.autocast("cuda", enabled=False):
                    p = {k: v.clone().detach().float().requires_grad_(True)
                         for k, v in params.items()}
                    engine.adapt(p, ep.support_x.float(), ep.support_y, 5, 0.01)
        def _noamp():
            engine.adapt(_fresh_p(params), ep.support_x, ep.support_y, 5, 0.01)
        suite.bench(_amp, "inner_loop_amp_safety", "ms")
        suite.bench(_noamp, "inner_loop_no_amp", "ms")
    else:
        suite.bench(_noclip, "inner_loop_cpu_baseline", "ms")

    # Scaling with hidden dim
    for hdim in [10, 100, 1000]:
        m, p = _make_mp(hdim, idim, odim, device)
        eng = CustomSGDEngine(m, True)
        eh = ds.sample_episode(n_way=odim, k_shot=1, seed=300 + hdim)
        def _rh(eng=eng, p=p, eh=eh):
            eng.adapt(_fresh_p(p), eh.support_x, eh.support_y, 5, 0.01)
        suite.bench(_rh, f"inner_loop_hdim{hdim}", "ms")
        del m, p; gc.collect()

    # Scaling with support set size
    for ks in [1, 5, 10, 20]:
        mb, pb = _make_mp(64, idim, odim, device)
        eb = CustomSGDEngine(mb, True)
        epb = ds.sample_episode(n_way=odim, k_shot=ks, q_query=1, seed=400 + ks)
        def _rb(eb=eb, pb=pb, epb=epb):
            eb.adapt(_fresh_p(pb), epb.support_x, epb.support_y, 5, 0.01)
        suite.bench(_rb, f"inner_loop_kshot{ks}_ss{ks * odim}", "ms")
        del mb, pb; gc.collect()

    return suite


# ============================================================================
# Suite 2: First-Order vs Second-Order Comparison
# ============================================================================

def bench_fo_vs_so(device: str = "cpu") -> BenchmarkSuite:
    """Compare first-order (FOMAML) vs second-order (MAML) performance.

    Measures:
        - Time per meta-step for each mode (including backward)
        - Peak memory for each mode
        - Speedup factor: SO_time / FO_time
    """
    suite = BenchmarkSuite("fo_vs_so", device)
    idim, odim = 10, 5
    ds = SyntheticFewShotDataset(50, idim, seed=42, device=device)
    tb = _make_tb(ds, 4, odim, 1)
    model, params = _make_mp(64, idim, odim, device)
    print("  Running FO vs SO comparison benchmarks...")

    def _so():
        o = maml_meta_step(model, _fresh_p(params), tb, 5, 0.01)
        o.loss.backward()
    def _fo():
        o = fomaml_meta_step(model, _fresh_p(params), tb, 5, 0.01)
        o.loss.backward()

    suite.reset_mem()
    so_t = suite.bench(_so, "fo_vs_so_second_order", "ms/meta-step", 2, 8)
    so_m = suite.peak_mem_mb()

    suite.reset_mem()
    fo_t = suite.bench(_fo, "fo_vs_so_first_order", "ms/meta-step", 2, 8)
    fo_m = suite.peak_mem_mb()

    suite.record("fo_vs_so_so_memory", "MB", so_m, {"mode": "second_order"})
    suite.record("fo_vs_so_fo_memory", "MB", fo_m, {"mode": "first_order"})
    speedup = (so_t / fo_t) if fo_t > 0 else 0.0
    suite.record("fo_vs_so_speedup", "x", speedup,
                 {"so_ms": so_t * 1000, "fo_ms": fo_t * 1000})
    return suite


# ============================================================================
# Suite 3: Algorithm Comparison (MAML vs FOMAML vs Reptile)
# ============================================================================

def bench_algorithms(device: str = "cpu") -> BenchmarkSuite:
    """Compare MAML, FOMAML, and Reptile algorithm performance.

    Measures:
        - Time per meta-step for 5-way 1-shot and 5-way 5-shot
        - Peak memory per algorithm
        - Scaling with tasks per batch (1, 4, 8, 16)
    """
    suite = BenchmarkSuite("algorithms", device)
    idim, odim, ns, lr = 10, 5, 5, 0.01
    ds = SyntheticFewShotDataset(50, idim, seed=42, device=device)
    model, params = _make_mp(64, idim, odim, device)
    algos = {
        "maml": maml_meta_step,
        "fomaml": fomaml_meta_step,
        "reptile": reptile_meta_step,
    }
    print("  Running algorithm comparison benchmarks...")

    # 5-way 1-shot and 5-way 5-shot
    for ks, tag, sb in [(1, "5w1s", 1000), (5, "5w5s", 2000)]:
        tb = _make_tb(ds, 4, odim, ks, seed_base=sb)
        for aname, afn in algos.items():
            def _r(fn=afn, tb=tb, an=aname):
                o = fn(model, _fresh_p(params), tb, ns, lr)
                if an != "reptile":
                    o.loss.backward()
            suite.bench(_r, f"{aname}_{tag}_5step", "ms/meta-step", 2, 8)

    # Peak memory per algorithm (5w1s config)
    tb1 = _make_tb(ds, 4, odim, 1, seed_base=1000)
    for aname, afn in algos.items():
        suite.reset_mem()
        o = afn(model, _fresh_p(params), tb1, ns, lr)
        if aname != "reptile":
            o.loss.backward()
        suite.record(f"{aname}_memory", "MB", suite.peak_mem_mb(),
                     {"algo": aname, "config": "5w1s_5step"})
        del o; gc.collect()

    # Scaling with task batch size
    for nt in [1, 4, 8, 16]:
        tb = _make_tb(ds, nt, odim, 1, seed_base=3000 + nt)
        for aname, afn in algos.items():
            def _r(fn=afn, tb=tb, an=aname):
                o = fn(model, _fresh_p(params), tb, ns, lr)
                if an != "reptile":
                    o.loss.backward()
            suite.bench(_r, f"{aname}_tasks{nt}", "ms/meta-step", 2, 6)

    return suite


# ============================================================================
# Suite 4: Episode Sampling Throughput
# ============================================================================

def bench_episode_sampling(device: str = "cpu") -> BenchmarkSuite:
    """Benchmark episode sampling throughput.

    Measures:
        - Raw episode sampling speed (episodes/sec) for 5-way 1-shot
        - Episode seed computation throughput (SHA-256 hash speed)
        - Scaling with N-way (5, 10, 20) and K-shot (1, 5, 10)
    """
    suite = BenchmarkSuite("episode_sampling", device)
    ds = SyntheticFewShotDataset(100, 10, seed=42, device=device)
    sampler = EpisodeSampler(ds, 42)
    print("  Running episode sampling throughput benchmarks...")

    # Raw sampling throughput
    nep = 200
    def _sb():
        for i in range(nep):
            ds.sample_episode(5, 1, 15, seed=i)
    t = suite.bench(_sb, "episode_sampling_5w1s", "ms", 1, 5)
    suite.results[-1].details["episodes_per_sec"] = nep / t if t > 0 else 0

    # Seed computation throughput
    nsd = 10000
    def _sc():
        for i in range(nsd):
            sampler._episode_seed(0, i)
    ts = suite.bench(_sc, "seed_computation", "ms", 2, 10)
    suite.results[-1].details["seeds_per_sec"] = nsd / ts if ts > 0 else 0

    # Scaling with N-way and K-shot
    configs = [
        (5, 1, 15), (5, 5, 15), (5, 10, 15),
        (10, 1, 10), (10, 5, 10),
        (20, 1, 5), (20, 5, 5),
    ]
    for nw, ks, qq in configs:
        ne = 100
        def _s(nw=nw, ks=ks, qq=qq):
            for i in range(ne):
                ds.sample_episode(nw, ks, qq, seed=i)
        tc = suite.bench(_s, f"episode_{nw}w{ks}s_{qq}q", "ms", 1, 5)
        suite.results[-1].details.update(
            episodes_per_sec=ne / tc if tc > 0 else 0,
            n_way=nw, k_shot=ks, q_query=qq,
            support_size=nw * ks, query_size=nw * qq,
        )

    return suite


# ============================================================================
# Suite 5: LSLR Overhead
# ============================================================================

def bench_lslr_overhead(device: str = "cpu") -> BenchmarkSuite:
    """Benchmark LSLR (per-layer per-step learning rate) overhead.

    Measures:
        - Time with LSLR vs without LSLR
        - Memory with LSLR vs without LSLR
        - Overhead as function of num_steps (1, 3, 5, 10)
    """
    suite = BenchmarkSuite("lslr_overhead", device)
    idim, odim, ns, lr = 10, 5, 5, 0.01
    ds = SyntheticFewShotDataset(50, idim, seed=42, device=device)
    ep = ds.sample_episode(odim, 1, seed=500)
    model, params = _make_mp(64, idim, odim, device)
    lslr = LSLRModule(list(model.named_parameters()), ns, lr).to(device)
    engine = CustomSGDEngine(model, True)
    lrs = lslr.get_lrs()
    print("  Running LSLR overhead benchmarks...")

    # Time with vs without
    def _wl():
        engine.adapt(_fresh_p(params), ep.support_x, ep.support_y, ns, lr, lrs)
    def _nl():
        engine.adapt(_fresh_p(params), ep.support_x, ep.support_y, ns, lr, None)

    tw = suite.bench(_wl, "lslr_with_lslr", "ms", 3, 10)
    tn = suite.bench(_nl, "lslr_without_lslr", "ms", 3, 10)
    oh = ((tw - tn) / tn * 100) if tn > 0 else 0.0
    suite.record("lslr_overhead_pct", "%", oh,
                 {"with_ms": tw * 1000, "without_ms": tn * 1000})

    # Memory delta
    suite.reset_mem(); _wl(); mw = suite.peak_mem_mb()
    suite.reset_mem(); _nl(); mn = suite.peak_mem_mb()
    suite.record("lslr_memory_with", "MB", mw)
    suite.record("lslr_memory_without", "MB", mn)
    suite.record("lslr_memory_delta", "MB", mw - mn,
                 {"num_lslr_params": sum(p.numel() for p in lslr.parameters())})

    # Scaling with num_steps
    nlayers = len(list(model.named_parameters()))
    for s in [1, 3, 5, 10]:
        ls = LSLRModule(list(model.named_parameters()), s, lr).to(device)
        lrs_s = ls.get_lrs()
        def _rs(lrs_s=lrs_s, s=s):
            engine.adapt(_fresh_p(params), ep.support_x, ep.support_y, s, lr, lrs_s)
        suite.bench(_rs, f"lslr_steps{s}", "ms", 2, 8)
        suite.results[-1].details.update(
            num_steps=s, num_layers=nlayers,
            total_lr_params=sum(p.numel() for p in ls.parameters()),
            layers_x_steps=nlayers * s,
        )
        del ls; gc.collect()

    return suite


# ============================================================================
# Suite 6: Multi-Step Loss Overhead
# ============================================================================

def bench_msl_overhead(device: str = "cpu") -> BenchmarkSuite:
    """Benchmark Multi-Step Loss (MSL) overhead.

    Measures:
        - Time with MSL vs final-step-only (including backward)
        - Peak memory with MSL vs final-step-only
        - Overhead scaling with inner_steps (1, 3, 5, 10)
    """
    suite = BenchmarkSuite("msl_overhead", device)
    idim, odim, ns, lr = 10, 5, 5, 0.01
    ds = SyntheticFewShotDataset(50, idim, seed=42, device=device)
    ep = ds.sample_episode(odim, 1, 15, seed=600)
    model, params = _make_mp(64, idim, odim, device)
    engine = CustomSGDEngine(model, True)
    msl = MultiStepLoss(ns, "uniform")
    print("  Running MSL overhead benchmarks...")

    # MSL vs final-step-only
    def _wm():
        l, _ = msl.compute_msl(model, engine, _fresh_p(params), ep, ns, lr)
        l.backward()
    def _fo():
        a, _ = engine.adapt(_fresh_p(params), ep.support_x, ep.support_y, ns, lr)
        q_logits = engine._functional_forward(a, ep.query_x)
        F.cross_entropy(q_logits, ep.query_y).backward()

    tm = suite.bench(_wm, "msl_with_msl", "ms", 2, 8)
    tf = suite.bench(_fo, "msl_final_only", "ms", 2, 8)
    oh = ((tm - tf) / tf * 100) if tf > 0 else 0.0
    suite.record("msl_overhead_pct", "%", oh,
                 {"msl_ms": tm * 1000, "final_ms": tf * 1000})

    # Memory comparison
    suite.reset_mem(); _wm(); mm = suite.peak_mem_mb()
    suite.reset_mem(); _fo(); mf = suite.peak_mem_mb()
    suite.record("msl_memory_with", "MB", mm)
    suite.record("msl_memory_without", "MB", mf)
    suite.record("msl_memory_delta", "MB", mm - mf)

    # Scaling with inner_steps
    for s in [1, 3, 5, 10]:
        ms = MultiStepLoss(s, "uniform")
        def _rm(ms=ms, s=s):
            l, _ = ms.compute_msl(model, engine, _fresh_p(params), ep, s, lr)
            l.backward()
        def _rf(s=s):
            a, _ = engine.adapt(_fresh_p(params), ep.support_x, ep.support_y, s, lr)
            F.cross_entropy(engine._functional_forward(a, ep.query_x), ep.query_y).backward()
        tm2 = suite.bench(_rm, f"msl_steps{s}_with", "ms", 2, 8)
        tf2 = suite.bench(_rf, f"msl_steps{s}_without", "ms", 2, 8)
        oh2 = ((tm2 - tf2) / tf2 * 100) if tf2 > 0 else 0.0
        suite.record(f"msl_steps{s}_overhead", "%", oh2,
                     {"inner_steps": s, "msl_ms": tm2 * 1000, "final_ms": tf2 * 1000})

    return suite


# ============================================================================
# Suite 7: Memory Profile
# ============================================================================

def bench_memory_profile(device: str = "cpu") -> BenchmarkSuite:
    """Profile peak memory usage for each algorithm across model sizes.

    Measures:
        - MAML/FOMAML/Reptile peak memory by hidden dim (32, 64, 128, 256, 512)
        - MAML memory per inner step (incremental) at hidden_dim=128
    """
    suite = BenchmarkSuite("memory_profile", device)
    idim, odim, ns, lr = 10, 5, 5, 0.01
    ds = SyntheticFewShotDataset(50, idim, seed=42, device=device)
    tb = _make_tb(ds, 4, odim, 1, seed_base=7000)
    algos = {
        "maml": maml_meta_step,
        "fomaml": fomaml_meta_step,
        "reptile": reptile_meta_step,
    }
    print("  Running memory profiling benchmarks...")

    # Peak memory by model size for each algorithm
    for hdim in [32, 64, 128, 256, 512]:
        model, params = _make_mp(hdim, idim, odim, device)
        pc = sum(p.numel() for p in model.parameters())
        for aname, afn in algos.items():
            suite.reset_mem()
            o = afn(model, _fresh_p(params), tb, ns, lr)
            if aname != "reptile":
                o.loss.backward()
            suite.record(
                f"mem_{aname}_hdim{hdim}", "MB", suite.peak_mem_mb(),
                {"algo": aname, "hidden_dim": hdim, "param_count": pc},
            )
            del o; gc.collect()
        del model, params; gc.collect()

    # Memory per inner step (MAML, hidden_dim=128)
    model, params = _make_mp(128, idim, odim, device)
    mems: List[float] = []
    for s in [1, 2, 3, 5, 7, 10]:
        suite.reset_mem()
        o = maml_meta_step(model, _fresh_p(params), tb, s, lr)
        o.loss.backward()
        m = suite.peak_mem_mb()
        mems.append(m)
        suite.record(
            f"mem_maml_steps{s}", "MB", m,
            {"inner_steps": s, "algo": "maml", "hidden_dim": 128},
        )
        del o; gc.collect()

    if len(mems) >= 2:
        incr = (mems[-1] - mems[0]) / 9.0
        suite.record(
            "mem_maml_per_step_incr", "MB/step", incr,
            {"base_1step": mems[0], "peak_10step": mems[-1]},
        )

    del model, params; gc.collect()
    return suite


# ============================================================================
# Runner: Execute Suites
# ============================================================================

SUITE_REG: Dict[str, Any] = {
    "inner_loop": bench_inner_loop,
    "fo_vs_so": bench_fo_vs_so,
    "algorithms": bench_algorithms,
    "episode_sampling": bench_episode_sampling,
    "lslr_overhead": bench_lslr_overhead,
    "msl_overhead": bench_msl_overhead,
    "memory_profile": bench_memory_profile,
}


def run_suites(names: List[str], device: str = "cpu"
               ) -> Dict[str, BenchmarkSuite]:
    """Run the specified benchmark suites, catching crashes gracefully.

    Args:
        names: List of suite names to execute.
        device: Torch device string.

    Returns:
        Dictionary mapping suite name to its BenchmarkSuite.
    """
    results: Dict[str, BenchmarkSuite] = {}
    for name in names:
        if name not in SUITE_REG:
            print(f"  [WARNING] Unknown suite: {name}, skipping.")
            continue
        print(f"\n--- Suite: {name} ---")
        try:
            results[name] = SUITE_REG[name](device)
        except Exception as e:
            print(f"  [FATAL] Suite '{name}' crashed: {e}")
            traceback.print_exc()
            s = BenchmarkSuite(name, device)
            s.record(f"{name}_error", "error", 0.0, {"error": str(e)})
            results[name] = s
    return results


# ============================================================================
# Output Formatting
# ============================================================================

def format_report(suites: Dict[str, BenchmarkSuite], device: str) -> str:
    """Format all benchmark results into a human-readable report.

    Args:
        suites: Dictionary of suite_name -> BenchmarkSuite.
        device: Device string for the report header.

    Returns:
        Multi-line formatted report string.
    """
    lines = [
        "", _SEP,
        "  Meta-Learning Benchmark",
        _SEP,
        f"  Device: {device}",
        f"  PyTorch: {torch.__version__}",
        "",
    ]

    for sn, suite in suites.items():
        lines.append(f"--- Suite: {sn} ---")
        for r in suite.results:
            std = r.details.get("std_ms", 0.0)
            u = r.metric
            if u in ("ms", "ms/step", "ms/meta-step"):
                tail = f"  (+/-{std:.2f})" if std > 0 else ""
                lines.append(f"  {r.name:40s}: {r.value:10.2f} {u:15s}{tail}")
            elif u == "MB":
                lines.append(f"  {r.name:40s}: {r.value:10.2f} {u}")
            elif u == "MB/step":
                lines.append(f"  {r.name:40s}: {r.value:10.4f} {u}")
            elif u == "x":
                lines.append(f"  {r.name:40s}: {r.value:10.2f}x")
            elif u == "%":
                lines.append(f"  {r.name:40s}: {r.value:10.2f}%")
            elif u == "error":
                err = r.details.get("error", "unknown")
                lines.append(f"  {r.name:40s}: ERROR - {err}")
            else:
                lines.append(f"  {r.name:40s}: {r.value:10.2f} {u}")
        lines.append("")

    total = sum(len(s.results) for s in suites.values())
    lines += [_SEP, f"  Total measurements: {total}", _SEP, ""]
    return "\n".join(lines)


def suites_to_json(suites: Dict[str, BenchmarkSuite], device: str) -> dict:
    """Convert all suite results to a JSON-serializable dictionary.

    Args:
        suites: Dictionary of suite_name -> BenchmarkSuite.
        device: Device string.

    Returns:
        JSON-serializable dictionary with device, timestamp, suites.
    """
    return {
        "device": device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pytorch_version": torch.__version__,
        "suites": {
            n: [asdict(r) for r in s.results]
            for n, s in suites.items()
        },
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Meta-Learning Suite -- Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--suite", type=str, default=None, choices=ALL_SUITES,
        help="Run a specific benchmark suite (default: all).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device to run benchmarks on (default: cpu).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results.",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_suites",
        help="List all available benchmark suites and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point: parse args, run suites, print report, save JSON."""
    args = parse_args()

    # List mode
    if args.list_suites:
        descs = {
            "inner_loop": "Inner-loop engine throughput (steps/sec)",
            "fo_vs_so": "First-order vs second-order comparison",
            "algorithms": "MAML vs FOMAML vs Reptile speed comparison",
            "episode_sampling": "Episode sampling throughput",
            "lslr_overhead": "LSLR parameter overhead measurement",
            "msl_overhead": "Multi-step loss overhead measurement",
            "memory_profile": "Peak memory usage per algorithm",
        }
        print("Available benchmark suites:\n")
        for n in ALL_SUITES:
            print(f"  {n:20s}  {descs.get(n, '')}")
        print()
        return

    # Validate CUDA
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # Determine suites
    names = [args.suite] if args.suite else list(ALL_SUITES)

    # Header
    print(_SEP)
    print("  Meta-Learning Benchmark")
    print(_SEP)
    print(f"  Device:   {device}")
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  Suites:   {', '.join(names)}")
    print(_SEP)

    # Run benchmarks
    suites = run_suites(names, device)

    # Print report
    report = format_report(suites, device)
    print(report)

    # Save JSON results
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(suites_to_json(suites, device), f, indent=2, default=str)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
