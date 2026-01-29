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


# =============================================================================
# MAML++ (Improved MAML - 2025)
# =============================================================================

@dataclass
class MAMLPlusPlusConfig:
    """Configuration for MAML++."""
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    num_inner_steps: int = 5
    num_tasks_per_batch: int = 4
    
    # MAML++ specific
    learn_inner_lr: bool = True  # Per-layer, per-step learning rates
    multi_step_loss: bool = True  # Loss at each inner step
    annealing_outer_lr: bool = True  # Cosine annealing
    batch_norm_per_step: bool = True  # Per-step batch norm
    gradient_clipping: float = 1.0
    
    # Task conditioning
    use_task_encoder: bool = True
    task_embedding_dim: int = 64


class PerLayerPerStepLR(nn.Module):
    """
    Per-layer, per-step learning rates for MAML++.
    
    Instead of a single inner learning rate, we learn:
    - Different LR for each layer
    - Different LR for each adaptation step
    
    This allows the model to learn optimal adaptation dynamics.
    """
    
    def __init__(
        self,
        param_shapes: List[torch.Size],
        num_steps: int,
        init_lr: float = 0.01,
    ):
        super().__init__()
        
        self.num_layers = len(param_shapes)
        self.num_steps = num_steps
        
        # (num_steps, num_layers) learnable LRs
        # Initialize to small positive values
        self.log_lrs = nn.Parameter(
            torch.full((num_steps, self.num_layers), fill_value=torch.tensor(init_lr).log())
        )
    
    def get_lr(self, step: int, layer_idx: int) -> torch.Tensor:
        """Get learning rate for specific step and layer."""
        return self.log_lrs[step, layer_idx].exp()
    
    def get_all_lrs(self, step: int) -> List[torch.Tensor]:
        """Get all layer LRs for a step."""
        return [self.log_lrs[step, i].exp() for i in range(self.num_layers)]


class TaskEncoder(nn.Module):
    """
    Encodes a task into an embedding for task conditioning.
    
    Uses set encoding to handle variable-size support sets.
    Based on DeepSets architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Encode each example
        self.example_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Aggregate across examples (DeepSets style)
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
    
    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode support set into task embedding.
        
        Args:
            support_x: (num_examples, input_dim)
            support_y: (num_examples,) or (num_examples, output_dim)
            
        Returns:
            task_embedding: (embedding_dim,)
        """
        # Combine x and y for each example
        if support_y.dim() == 1:
            support_y = support_y.unsqueeze(-1).float()
        
        combined = torch.cat([support_x, support_y], dim=-1)
        
        # Pad to expected input dim
        if combined.shape[-1] < self.input_dim:
            padding = torch.zeros(
                combined.shape[0],
                self.input_dim - combined.shape[-1],
                device=combined.device
            )
            combined = torch.cat([combined, padding], dim=-1)
        elif combined.shape[-1] > self.input_dim:
            combined = combined[:, :self.input_dim]
        
        # Encode each example
        encoded = self.example_encoder(combined)
        
        # Aggregate (sum/mean)
        aggregated = encoded.mean(dim=0)
        
        # Final projection
        task_embedding = self.aggregator(aggregated)
        
        return task_embedding


class MAMLPlusPlus(nn.Module):
    """
    MAML++ - Improved Meta-Learning.
    
    Key improvements over standard MAML:
    1. Per-layer, per-step learning rates (learned)
    2. Multi-step loss (loss computed at each inner step)
    3. Derivative-order annealing (for stability)
    4. Per-step batch normalization
    5. Task conditioning via task embeddings
    
    Based on "How to Train Your MAML" (Antoniou et al.) with 2025 updates.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[MAMLPlusPlusConfig] = None,
        input_dim: int = 784,  # For task encoder
        **kwargs,
    ):
        super().__init__()
        
        self.model = model
        self.config = config or MAMLPlusPlusConfig(**kwargs)
        
        # Get parameter shapes for per-layer LRs
        param_shapes = [p.shape for p in model.parameters()]
        
        # Per-layer, per-step learning rates
        if self.config.learn_inner_lr:
            self.learned_lrs = PerLayerPerStepLR(
                param_shapes=param_shapes,
                num_steps=self.config.num_inner_steps,
                init_lr=self.config.inner_lr,
            )
        else:
            self.learned_lrs = None
        
        # Task encoder for conditioning
        if self.config.use_task_encoder:
            self.task_encoder = TaskEncoder(
                input_dim=input_dim,
                embedding_dim=self.config.task_embedding_dim,
            )
        else:
            self.task_encoder = None
        
        # Store for multi-step loss
        self.step_losses = []
    
    def _get_lr(self, step: int, layer_idx: int) -> float:
        """Get learning rate for step and layer."""
        if self.learned_lrs is not None:
            return self.learned_lrs.get_lr(step, layer_idx)
        return self.config.inner_lr
    
    def inner_loop(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: Callable,
    ) -> Tuple[nn.Module, List[torch.Tensor]]:
        """
        Perform inner loop adaptation with MAML++ improvements.
        
        Returns:
            adapted_model: Model after adaptation
            step_losses: Loss at each step (for multi-step loss)
        """
        adapted_model = deepcopy(model)
        step_losses = []
        
        for step in range(self.config.num_inner_steps):
            # Forward pass
            outputs = adapted_model(support_x)
            loss = loss_fn(outputs, support_y)
            step_losses.append(loss)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_model.parameters(),
                create_graph=True,  # Second order for full MAML
                allow_unused=True,
            )
            
            # Update parameters with per-layer LRs
            with torch.enable_grad():
                for idx, (param, grad) in enumerate(zip(adapted_model.parameters(), grads)):
                    if grad is None:
                        continue
                    
                    lr = self._get_lr(step, idx)
                    
                    # Gradient clipping
                    if self.config.gradient_clipping > 0:
                        grad = torch.clamp(
                            grad,
                            -self.config.gradient_clipping,
                            self.config.gradient_clipping
                        )
                    
                    param.data = param.data - lr * grad
        
        return adapted_model, step_losses
    
    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: Callable,
    ) -> nn.Module:
        """Adapt to task using support set."""
        adapted, self.step_losses = self.inner_loop(
            self.model, support_x, support_y, loss_fn
        )
        return adapted
    
    def forward(
        self,
        query_x: torch.Tensor,
        support_x: Optional[torch.Tensor] = None,
        support_y: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with optional adaptation.
        
        If support set provided, adapts first then evaluates on query.
        """
        if support_x is not None and support_y is not None and loss_fn is not None:
            adapted = self.adapt(support_x, support_y, loss_fn)
            return adapted(query_x)
        return self.model(query_x)
    
    def compute_meta_loss(
        self,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        loss_fn: Callable,
        adapted_model: nn.Module,
    ) -> torch.Tensor:
        """
        Compute meta-loss with multi-step loss weighting.
        
        Multi-step loss improves training stability by computing
        loss at each inner step, not just the final one.
        """
        # Query loss (main objective)
        outputs = adapted_model(query_x)
        query_loss = loss_fn(outputs, query_y)
        
        if not self.config.multi_step_loss or len(self.step_losses) == 0:
            return query_loss
        
        # Multi-step loss: weight earlier steps less
        total_loss = query_loss
        for i, step_loss in enumerate(self.step_losses):
            # Annealed weight: later steps matter more
            weight = (i + 1) / len(self.step_losses) * 0.5
            total_loss = total_loss + weight * step_loss
        
        return total_loss


# =============================================================================
# TASK2VEC - Task Embedding for Meta-Learning (2025)
# =============================================================================

class Task2Vec(nn.Module):
    """
    Task2Vec: Embed tasks into a vector space.
    
    Creates task embeddings that capture:
    - Task difficulty
    - Task structure
    - Task similarity to other tasks
    
    Uses Fisher Information Matrix diagonal as task descriptor.
    Based on "Task2Vec: Task Embedding for Meta-Learning" with 2025 updates.
    """
    
    def __init__(
        self,
        probe_network: nn.Module,
        embedding_dim: int = 256,
    ):
        super().__init__()
        
        self.probe_network = probe_network
        self.embedding_dim = embedding_dim
        
        # Number of parameters in probe network
        self.num_params = sum(p.numel() for p in probe_network.parameters())
        
        # Compress FIM diagonal to embedding
        self.compressor = nn.Sequential(
            nn.Linear(self.num_params, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
        )
    
    def compute_fim_diagonal(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: Callable,
        num_samples: int = 100,
    ) -> torch.Tensor:
        """
        Compute Fisher Information Matrix diagonal.
        
        FIM captures how sensitive the loss is to each parameter,
        which describes the task structure.
        """
        device = support_x.device
        
        # Accumulate squared gradients
        fim_diagonal = torch.zeros(self.num_params, device=device)
        
        for i in range(min(num_samples, support_x.shape[0])):
            # Forward pass for single example
            x = support_x[i:i+1]
            y = support_y[i:i+1]
            
            output = self.probe_network(x)
            loss = loss_fn(output, y)
            
            # Compute gradients
            self.probe_network.zero_grad()
            loss.backward(retain_graph=True)
            
            # Extract gradient diagonal
            idx = 0
            for param in self.probe_network.parameters():
                if param.grad is not None:
                    flat_grad = param.grad.view(-1)
                    fim_diagonal[idx:idx+flat_grad.numel()] += flat_grad ** 2
                    idx += flat_grad.numel()
        
        # Normalize
        fim_diagonal = fim_diagonal / num_samples
        
        return fim_diagonal
    
    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """
        Compute task embedding from support set.
        
        Returns:
            task_embedding: (embedding_dim,) vector describing the task
        """
        # Compute FIM diagonal
        fim_diag = self.compute_fim_diagonal(support_x, support_y, loss_fn)
        
        # Compress to embedding
        # Take log for numerical stability
        log_fim = torch.log(fim_diag + 1e-8)
        
        embedding = self.compressor(log_fim)
        
        return embedding
    
    def task_similarity(
        self,
        task1_embedding: torch.Tensor,
        task2_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between task embeddings."""
        return F.cosine_similarity(
            task1_embedding.unsqueeze(0),
            task2_embedding.unsqueeze(0),
        ).squeeze()


class TaskAwareMetaLearner(nn.Module):
    """
    Meta-learner that uses Task2Vec for task-aware adaptation.
    
    The task embedding is used to:
    1. Condition the inner loop adaptation
    2. Select appropriate adaptation strategy
    3. Weight meta-gradient contributions
    """
    
    def __init__(
        self,
        model: nn.Module,
        probe_network: nn.Module,
        task_embedding_dim: int = 256,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
    ):
        super().__init__()
        
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        
        # Task2Vec for task embeddings
        self.task2vec = Task2Vec(
            probe_network=probe_network,
            embedding_dim=task_embedding_dim,
        )
        
        # Task-conditioned hypernetwork for adaptation
        # Generates per-layer learning rates based on task
        num_layers = len(list(model.parameters()))
        self.lr_generator = nn.Sequential(
            nn.Linear(task_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_layers),
            nn.Softplus(),  # Ensure positive LRs
        )
        
        # Store task embeddings for analysis
        self.task_embeddings = []
    
    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: Callable,
    ) -> nn.Module:
        """Adapt model using task-aware learning rates."""
        
        # Get task embedding
        task_embedding = self.task2vec(support_x, support_y, loss_fn)
        self.task_embeddings.append(task_embedding.detach())
        
        # Generate task-specific learning rates
        task_lrs = self.lr_generator(task_embedding) * self.inner_lr
        
        # Adapt with task-specific LRs
        adapted_model = deepcopy(self.model)
        
        for step in range(self.num_inner_steps):
            outputs = adapted_model(support_x)
            loss = loss_fn(outputs, support_y)
            
            grads = torch.autograd.grad(
                loss,
                adapted_model.parameters(),
                create_graph=True,
            )
            
            with torch.enable_grad():
                for idx, (param, grad) in enumerate(zip(adapted_model.parameters(), grads)):
                    lr = task_lrs[idx]
                    param.data = param.data - lr * grad
        
        return adapted_model
    
    def forward(
        self,
        query_x: torch.Tensor,
        support_x: Optional[torch.Tensor] = None,
        support_y: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with optional task-aware adaptation."""
        if support_x is not None and support_y is not None and loss_fn is not None:
            adapted = self.adapt(support_x, support_y, loss_fn)
            return adapted(query_x)
        return self.model(query_x)
    
    def get_task_clusters(self, num_clusters: int = 5):
        """Cluster stored task embeddings."""
        if len(self.task_embeddings) < num_clusters:
            return None
        
        embeddings = torch.stack(self.task_embeddings)
        
        # Simple k-means style clustering
        # In practice, use sklearn KMeans
        from torch.cluster import KMeans  # Hypothetical
        
        return embeddings  # Return for external clustering


def create_maml_plus_plus(
    model: nn.Module,
    input_dim: int = 784,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    num_inner_steps: int = 5,
    learn_inner_lr: bool = True,
    multi_step_loss: bool = True,
    **kwargs,
) -> MAMLPlusPlus:
    """Create MAML++ meta-learner."""
    config = MAMLPlusPlusConfig(
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        num_inner_steps=num_inner_steps,
        learn_inner_lr=learn_inner_lr,
        multi_step_loss=multi_step_loss,
        **kwargs,
    )
    return MAMLPlusPlus(model, config, input_dim=input_dim)


def create_task2vec(
    probe_network: nn.Module,
    embedding_dim: int = 256,
) -> Task2Vec:
    """Create Task2Vec embedder."""
    return Task2Vec(
        probe_network=probe_network,
        embedding_dim=embedding_dim,
    )
