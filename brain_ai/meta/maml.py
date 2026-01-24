"""
Model-Agnostic Meta-Learning (MAML) Module

Implements MAML and variants for few-shot learning:
- Standard MAML: Second-order gradients through inner loop
- First-Order MAML (FOMAML): Faster, ignores second-order terms
- Reptile: Even simpler, just move toward task solutions

Meta-learning enables rapid adaptation to new tasks with minimal data,
similar to how humans quickly learn new concepts from few examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass
from copy import deepcopy
import higher  # For differentiable optimization (optional)


@dataclass
class MAMLConfig:
    """Configuration for MAML."""
    inner_lr: float = 0.01  # Learning rate for inner loop
    outer_lr: float = 0.001  # Learning rate for meta-update
    num_inner_steps: int = 5  # Number of gradient steps per task
    first_order: bool = False  # Use first-order approximation
    num_tasks_per_batch: int = 4  # Tasks per meta-batch


class InnerLoopOptimizer:
    """
    Performs inner loop optimization for a single task.

    Can operate in:
    - Differentiable mode: Tracks gradients through inner loop
    - Non-differentiable mode: Faster but no meta-gradients
    """

    def __init__(
        self,
        lr: float = 0.01,
        num_steps: int = 5,
        first_order: bool = False,
    ):
        self.lr = lr
        self.num_steps = num_steps
        self.first_order = first_order

    def step(
        self,
        model: nn.Module,
        loss_fn: Callable,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        """
        Adapt model to task using support set.

        Args:
            model: Model to adapt
            loss_fn: Loss function
            support_x: Support set inputs
            support_y: Support set targets

        Returns:
            Adapted model (or original model with updated parameters)
        """
        # Clone model for inner loop
        adapted_model = self._clone_model(model)

        for _ in range(self.num_steps):
            # Forward pass
            outputs = adapted_model(support_x)
            loss = loss_fn(outputs, support_y)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_model.parameters(),
                create_graph=not self.first_order,
            )

            # Update parameters
            with torch.no_grad() if self.first_order else torch.enable_grad():
                for param, grad in zip(adapted_model.parameters(), grads):
                    param.data = param.data - self.lr * grad

        return adapted_model

    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Clone model with gradient tracking."""
        cloned = deepcopy(model)

        # Ensure parameters track gradients
        for param in cloned.parameters():
            param.requires_grad = True

        return cloned


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning.

    Learns initialization parameters that can quickly adapt
    to new tasks with few gradient steps.

    Args:
        model: The model to meta-train
        config: MAMLConfig with hyperparameters
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[MAMLConfig] = None,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.config = config or MAMLConfig(**kwargs)

        # Inner loop optimizer
        self.inner_optimizer = InnerLoopOptimizer(
            lr=self.config.inner_lr,
            num_steps=self.config.num_inner_steps,
            first_order=self.config.first_order,
        )

        # Track meta-parameters separately
        self.meta_parameters = list(self.model.parameters())

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: Callable,
    ) -> nn.Module:
        """
        Adapt model to task using support set.

        This is the inner loop of MAML.

        Args:
            support_x: Support set inputs (few examples)
            support_y: Support set targets
            loss_fn: Task-specific loss function

        Returns:
            Task-adapted model
        """
        return self.inner_optimizer.step(
            self.model,
            loss_fn,
            support_x,
            support_y,
        )

    def meta_learn(
        self,
        tasks: List[Dict[str, torch.Tensor]],
        loss_fn: Callable,
        outer_optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Perform one meta-learning step over multiple tasks.

        Args:
            tasks: List of task dicts with support/query splits
                Each task has: support_x, support_y, query_x, query_y
            loss_fn: Loss function
            outer_optimizer: Optimizer for meta-parameters

        Returns:
            Dict with meta-loss and task losses
        """
        outer_optimizer.zero_grad()

        total_meta_loss = 0.0
        task_losses = []

        for task in tasks:
            # Inner loop: adapt to task
            adapted_model = self.adapt(
                task['support_x'],
                task['support_y'],
                loss_fn,
            )

            # Evaluate on query set
            query_outputs = adapted_model(task['query_x'])
            query_loss = loss_fn(query_outputs, task['query_y'])

            total_meta_loss += query_loss
            task_losses.append(query_loss.item())

        # Average meta-loss
        meta_loss = total_meta_loss / len(tasks)

        # Outer loop: update meta-parameters
        meta_loss.backward()
        outer_optimizer.step()

        return {
            'meta_loss': meta_loss.item(),
            'task_losses': task_losses,
        }

    def forward(
        self,
        x: torch.Tensor,
        support_x: Optional[torch.Tensor] = None,
        support_y: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional adaptation.

        If support set provided, adapts before inference.
        """
        if support_x is not None and support_y is not None and loss_fn is not None:
            adapted = self.adapt(support_x, support_y, loss_fn)
            return adapted(x)
        else:
            return self.model(x)


class FOMAML(MAML):
    """
    First-Order MAML.

    Ignores second-order gradient terms for efficiency.
    Often works nearly as well as full MAML.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[MAMLConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = MAMLConfig(first_order=True, **kwargs)
        else:
            config.first_order = True

        super().__init__(model, config)


class Reptile(nn.Module):
    """
    Reptile meta-learning algorithm.

    Even simpler than MAML - just moves toward task solutions:
    θ = θ + ε(θ' - θ)

    where θ' is the task-adapted parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
    ):
        super().__init__()

        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: Callable,
    ) -> Dict[str, torch.Tensor]:
        """
        Adapt to task, returning parameter differences.

        Returns dict of (param_name, param_diff) for Reptile update.
        """
        # Clone and adapt
        adapted_model = deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        for _ in range(self.num_inner_steps):
            optimizer.zero_grad()
            outputs = adapted_model(support_x)
            loss = loss_fn(outputs, support_y)
            loss.backward()
            optimizer.step()

        # Compute parameter differences
        diffs = {}
        for (name, param), (_, adapted_param) in zip(
            self.model.named_parameters(),
            adapted_model.named_parameters()
        ):
            diffs[name] = adapted_param.data - param.data

        return diffs

    def meta_learn(
        self,
        tasks: List[Dict[str, torch.Tensor]],
        loss_fn: Callable,
    ) -> Dict[str, float]:
        """
        Reptile meta-update: average task directions.

        Args:
            tasks: List of task dicts
            loss_fn: Loss function

        Returns:
            Dict with losses
        """
        # Accumulate parameter differences
        accumulated_diffs = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
        }

        task_losses = []

        for task in tasks:
            diffs = self.adapt(
                task['support_x'],
                task['support_y'],
                loss_fn,
            )

            for name in accumulated_diffs:
                accumulated_diffs[name] += diffs[name]

            # Evaluate
            with torch.no_grad():
                adapted_model = deepcopy(self.model)
                for name, param in adapted_model.named_parameters():
                    param.data += diffs[name]
                outputs = adapted_model(task['query_x'])
                loss = loss_fn(outputs, task['query_y'])
                task_losses.append(loss.item())

        # Average and apply Reptile update
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data += self.outer_lr * accumulated_diffs[name] / len(tasks)

        return {
            'task_losses': task_losses,
            'mean_loss': sum(task_losses) / len(task_losses),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MetaLearner(nn.Module):
    """
    Unified meta-learning interface.

    Supports multiple meta-learning algorithms with consistent API.
    """

    def __init__(
        self,
        model: nn.Module,
        algorithm: str = "maml",
        **kwargs,
    ):
        super().__init__()

        self.algorithm = algorithm

        if algorithm == "maml":
            self.meta_learner = MAML(model, **kwargs)
        elif algorithm == "fomaml":
            self.meta_learner = FOMAML(model, **kwargs)
        elif algorithm == "reptile":
            self.meta_learner = Reptile(model, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: Callable,
    ):
        """Adapt to task."""
        return self.meta_learner.adapt(support_x, support_y, loss_fn)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.meta_learner(x, **kwargs)


# Factory function
def create_meta_learner(
    model: nn.Module,
    algorithm: str = "fomaml",
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    num_inner_steps: int = 5,
    **kwargs,
) -> MetaLearner:
    """
    Create meta-learner with specified algorithm.

    Args:
        model: Model to meta-train
        algorithm: 'maml', 'fomaml', or 'reptile'
        inner_lr: Inner loop learning rate
        outer_lr: Outer loop learning rate
        num_inner_steps: Steps per task adaptation
    """
    if algorithm in ("maml", "fomaml"):
        config = MAMLConfig(
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            num_inner_steps=num_inner_steps,
            first_order=(algorithm == "fomaml"),
            **kwargs,
        )
        return MetaLearner(model, algorithm, config=config)
    else:
        return MetaLearner(
            model,
            algorithm,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            num_inner_steps=num_inner_steps,
        )
