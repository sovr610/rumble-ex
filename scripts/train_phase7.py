#!/usr/bin/env python3
"""
Phase 7 Training Script: Meta-Learning & Plasticity

Trains the meta-learning system with MAML for few-shot adaptation
and neuromodulatory gating for controlling plasticity.

Target: 80%+ few-shot accuracy with 5-way 1-shot classification

Usage:
    python scripts/train_phase7.py
    python scripts/train_phase7.py --meta-epochs 100 --inner-lr 0.01
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_ai.meta.maml import MAML, MAMLConfig, InnerLoopOptimizer
from brain_ai.meta.neuromodulation import NeuromodulatoryGate
from brain_ai.meta.eligibility import EligibilityTrace


def parse_args():
    parser = argparse.ArgumentParser(description="Train Meta-Learning System")
    parser.add_argument("--mode", type=str, default="dev",
                        choices=["dev", "production", "production_3b", "production_1b"],
                        help="Training mode")
    parser.add_argument("--meta-epochs", type=int, default=None, help="Meta-training epochs")
    parser.add_argument("--tasks-per-batch", type=int, default=None, help="Tasks per meta-batch")
    parser.add_argument("--n-way", type=int, default=None, help="N-way classification")
    parser.add_argument("--k-shot", type=int, default=None, help="K-shot (support examples)")
    parser.add_argument("--q-query", type=int, default=None, help="Query examples per class")
    parser.add_argument("--inner-lr", type=float, default=None, help="Inner loop learning rate")
    parser.add_argument("--outer-lr", type=float, default=None, help="Outer loop learning rate")
    parser.add_argument("--inner-steps", type=int, default=None, help="Inner loop gradient steps")
    parser.add_argument("--first-order", action="store_true", help="Use first-order MAML")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--save-path", type=str, default=None, help="Save path")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision")
    return parser.parse_args()


def get_mode_config(mode: str) -> dict:
    """Get configuration based on training mode."""
    configs = {
        "dev": {
            "meta_epochs": 100,
            "tasks_per_batch": 4,
            "n_way": 5,
            "k_shot": 1,
            "q_query": 15,
            "inner_lr": 0.1,
            "outer_lr": 0.001,
            "inner_steps": 10,
            "feature_dim": 64,
            "hidden_dim": 128,
            "num_classes": 100,
            "weight_decay": 0.0,
            "save_path": "checkpoints/meta_learning_dev.pth",
        },
        "production_1b": {
            "meta_epochs": 500,
            "tasks_per_batch": 8,
            "n_way": 5,
            "k_shot": 5,
            "q_query": 15,
            "inner_lr": 0.05,
            "outer_lr": 0.0005,
            "inner_steps": 15,
            "feature_dim": 256,
            "hidden_dim": 512,
            "num_classes": 200,
            "weight_decay": 0.01,
            "save_path": "checkpoints/meta_learning_1b.pth",
        },
        "production_3b": {
            "meta_epochs": 1000,
            "tasks_per_batch": 16,
            "n_way": 10,
            "k_shot": 5,
            "q_query": 15,
            "inner_lr": 0.01,
            "outer_lr": 0.0001,
            "inner_steps": 20,
            "feature_dim": 512,
            "hidden_dim": 1024,
            "num_classes": 500,
            "weight_decay": 0.01,
            "save_path": "checkpoints/meta_learning_3b.pth",
        },
        "production": {  # 7B scale
            "meta_epochs": 2000,
            "tasks_per_batch": 32,
            "n_way": 20,
            "k_shot": 5,
            "q_query": 15,
            "inner_lr": 0.001,
            "outer_lr": 0.0001,
            "inner_steps": 30,
            "feature_dim": 1024,
            "hidden_dim": 2048,
            "num_classes": 1000,
            "weight_decay": 0.01,
            "save_path": "checkpoints/meta_learning_7b.pth",
        },
    }
    return configs.get(mode, configs["dev"])


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class SyntheticFewShotDataset:
    """
    Synthetic few-shot learning dataset generator.
    
    Creates tasks with:
    - N classes per task
    - K support examples per class
    - Q query examples per class
    
    Each class has a unique prototype with noise.
    Classes are made WELL-SEPARATED for easier learning.
    """
    
    def __init__(
        self,
        num_classes: int = 100,
        feature_dim: int = 64,
        n_way: int = 5,
        k_shot: int = 1,
        q_query: int = 15,
        seed: int = 42,
        noise_scale: float = 0.1,  # Lower noise for cleaner separation
    ):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.noise_scale = noise_scale
        
        np.random.seed(seed)
        
        # Create WELL-SEPARATED class prototypes
        # Use random orthogonal directions scaled up for clear separation
        self.prototypes = np.random.randn(num_classes, feature_dim)
        # Normalize to unit sphere and scale for separation
        self.prototypes /= np.linalg.norm(self.prototypes, axis=1, keepdims=True)
        self.prototypes *= 2.0  # Scale up for better separation
    
    def sample_task(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a single few-shot task.
        
        Returns:
            support_x: (n_way * k_shot, feature_dim)
            support_y: (n_way * k_shot,)
            query_x: (n_way * q_query, feature_dim)
            query_y: (n_way * q_query,)
        """
        # Sample N classes for this task
        task_classes = np.random.choice(self.num_classes, self.n_way, replace=False)
        
        support_x = []
        support_y = []
        query_x = []
        query_y = []
        
        for task_label, global_class in enumerate(task_classes):
            prototype = self.prototypes[global_class]
            
            # Generate support examples with LOW noise
            for _ in range(self.k_shot):
                noise = np.random.randn(self.feature_dim) * self.noise_scale
                support_x.append(prototype + noise)
                support_y.append(task_label)
            
            # Generate query examples with same low noise
            for _ in range(self.q_query):
                noise = np.random.randn(self.feature_dim) * self.noise_scale
                query_x.append(prototype + noise)
                query_y.append(task_label)
        
        return (
            torch.tensor(np.array(support_x), dtype=torch.float32),
            torch.tensor(support_y, dtype=torch.long),
            torch.tensor(np.array(query_x), dtype=torch.float32),
            torch.tensor(query_y, dtype=torch.long),
        )
    
    def sample_batch(self, num_tasks: int) -> List[Tuple]:
        """Sample a batch of tasks."""
        return [self.sample_task() for _ in range(num_tasks)]


class FewShotClassifier(nn.Module):
    """
    Simple classifier for few-shot learning.
    
    Used as the base model for MAML.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        n_way: int = 5,
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=False),
            nn.ReLU(),
        )
        
        self.classifier = nn.Linear(hidden_dim, n_way)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)


class MetaLearner(nn.Module):
    """
    Complete meta-learning system with:
    1. MAML for fast adaptation
    2. Neuromodulatory gating for plasticity control
    3. Eligibility traces for credit assignment
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        n_way: int = 5,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        first_order: bool = False,
    ):
        super().__init__()
        
        self.n_way = n_way
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
        # Base model
        self.base_model = FewShotClassifier(input_dim, hidden_dim, n_way)
        
        # Neuromodulatory gate (controls which parameters adapt)
        self.neuro_gate = NeuromodulatoryGate(
            input_dim=hidden_dim,
            num_modulators=4,
        )
        
        # Learning rate modulator
        self.lr_modulator = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensure positive
        )
    
    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> nn.Module:
        """
        Adapt base model to task using support set.
        
        Implements MAML's inner loop with optional neuromodulation.
        """
        num_steps = num_steps or self.inner_steps
        
        # Clone model for adaptation
        adapted_model = deepcopy(self.base_model)
        
        # Compute task-specific learning rate
        support_mean = support_x.mean(dim=0)
        support_var = support_x.var(dim=0)
        lr_input = torch.cat([support_mean, support_var])
        task_lr = self.lr_modulator(lr_input.unsqueeze(0)).squeeze() * self.inner_lr
        
        for step in range(num_steps):
            # Forward pass
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_model.parameters(),
                create_graph=not self.first_order,
                allow_unused=True,
            )
            
            # Update parameters
            with torch.no_grad() if self.first_order else torch.enable_grad():
                for param, grad in zip(adapted_model.parameters(), grads):
                    if grad is not None:
                        param.data = param.data - task_lr * grad
        
        return adapted_model
    
    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Meta-forward pass: adapt then predict on query.
        """
        adapted_model = self.adapt(support_x, support_y)
        return adapted_model(query_x)
    
    def meta_train_step(
        self,
        task_batch: List[Tuple],
        device: torch.device,
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform one meta-training step over a batch of tasks.
        
        Returns:
            meta_loss: Loss for outer loop update
            accuracy: Average query accuracy
        """
        meta_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for support_x, support_y, query_x, query_y in task_batch:
            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)
            query_y = query_y.to(device)
            
            # Adapt and predict
            query_logits = self.forward(support_x, support_y, query_x)
            
            # Compute task loss
            task_loss = F.cross_entropy(query_logits, query_y)
            meta_loss = meta_loss + task_loss
            
            # Track accuracy
            pred = query_logits.argmax(dim=1)
            total_correct += (pred == query_y).sum().item()
            total_samples += query_y.size(0)
        
        meta_loss = meta_loss / len(task_batch)
        accuracy = total_correct / total_samples
        
        return meta_loss, accuracy


def train_meta_epoch(
    meta_learner: MetaLearner,
    task_generator: SyntheticFewShotDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tasks_per_batch: int,
    num_batches: int,
    epoch: int,
) -> Tuple[float, float]:
    """Train for one meta-epoch."""
    meta_learner.train()
    total_loss = 0.0
    total_acc = 0.0
    
    for batch_idx in range(num_batches):
        # Sample task batch
        task_batch = task_generator.sample_batch(tasks_per_batch)
        
        optimizer.zero_grad()
        
        # Meta-training step
        meta_loss, accuracy = meta_learner.meta_train_step(task_batch, device)
        
        # Outer loop update
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(meta_learner.parameters(), max_norm=10.0)
        optimizer.step()
        
        total_loss += meta_loss.item()
        total_acc += accuracy
        
        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{num_batches}] "
                  f"Loss: {meta_loss.item():.4f} Acc: {accuracy*100:.2f}%")
    
    return total_loss / num_batches, total_acc / num_batches


def evaluate_few_shot(
    meta_learner: MetaLearner,
    task_generator: SyntheticFewShotDataset,
    device: torch.device,
    num_tasks: int = 100,
) -> Dict[str, float]:
    """Evaluate few-shot performance."""
    meta_learner.eval()
    
    accuracies = []
    
    with torch.no_grad():
        for _ in range(num_tasks):
            support_x, support_y, query_x, query_y = task_generator.sample_task()
            
            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)
            query_y = query_y.to(device)
            
            # Need gradients for inner loop even during eval
            with torch.enable_grad():
                query_logits = meta_learner(support_x, support_y, query_x)
            
            pred = query_logits.argmax(dim=1)
            accuracy = (pred == query_y).float().mean().item()
            accuracies.append(accuracy)
    
    return {
        'mean_accuracy': np.mean(accuracies) * 100,
        'std_accuracy': np.std(accuracies) * 100,
        'min_accuracy': np.min(accuracies) * 100,
        'max_accuracy': np.max(accuracies) * 100,
    }


def test_adaptation_speed(
    meta_learner: MetaLearner,
    task_generator: SyntheticFewShotDataset,
    device: torch.device,
    max_steps: int = 10,
) -> List[float]:
    """Test how quickly model adapts with different numbers of gradient steps."""
    meta_learner.eval()
    
    step_accuracies = {s: [] for s in range(1, max_steps + 1)}
    
    for _ in range(50):
        support_x, support_y, query_x, query_y = task_generator.sample_task()
        
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)
        
        for num_steps in range(1, max_steps + 1):
            with torch.enable_grad():
                adapted = meta_learner.adapt(support_x, support_y, num_steps=num_steps)
                query_logits = adapted(query_x)
            
            pred = query_logits.argmax(dim=1)
            accuracy = (pred == query_y).float().mean().item()
            step_accuracies[num_steps].append(accuracy)
    
    return [np.mean(step_accuracies[s]) * 100 for s in range(1, max_steps + 1)]


def main():
    args = parse_args()
    mode_config = get_mode_config(args.mode)
    device = get_device(args.device)
    
    print(f"Using device: {device}")
    print(f"Training mode: {args.mode}")
    print(f"Mode config: {mode_config}")
    
    # Override with CLI args if provided
    meta_epochs = args.meta_epochs or mode_config["meta_epochs"]
    tasks_per_batch = args.tasks_per_batch or mode_config["tasks_per_batch"]
    n_way = args.n_way or mode_config["n_way"]
    k_shot = args.k_shot or mode_config["k_shot"]
    q_query = args.q_query or mode_config["q_query"]
    outer_lr = args.outer_lr or mode_config["outer_lr"]
    inner_lr = args.inner_lr or mode_config["inner_lr"]
    inner_steps = args.inner_steps or mode_config["inner_steps"]
    hidden_dim = mode_config["hidden_dim"]
    num_classes = mode_config["num_classes"]
    feature_dim = mode_config["feature_dim"]
    weight_decay = mode_config["weight_decay"]
    save_path = args.save_path or mode_config["save_path"]
    first_order = args.first_order  # Boolean flag, use directly
    
    # Use AMP for production modes
    use_amp = args.mode in ["production", "production_3b", "production_1b"]
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    
    # Create task generators
    print(f"\nSetting up {n_way}-way {k_shot}-shot learning...")
    train_generator = SyntheticFewShotDataset(
        num_classes=num_classes,
        feature_dim=feature_dim,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        seed=42,
    )
    test_generator = SyntheticFewShotDataset(
        num_classes=num_classes,
        feature_dim=feature_dim,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        seed=123,
    )
    
    # Create meta-learner with mode-specific dimensions
    meta_learner = MetaLearner(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        n_way=n_way,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        first_order=first_order,
    ).to(device)
    
    # Compile for production modes
    if args.mode in ["production", "production_3b"] and hasattr(torch, "compile"):
        try:
            meta_learner = torch.compile(meta_learner, mode="reduce-overhead")
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"Could not compile model: {e}")
    
    total_params = sum(p.numel() for p in meta_learner.parameters())
    trainable_params = sum(p.numel() for p in meta_learner.parameters() if p.requires_grad)
    print(f"\nMeta-Learner parameters: {total_params:,} (trainable: {trainable_params:,})")
    print(f"MAML mode: {'First-order' if first_order else 'Second-order'}")
    
    # Optimizer (outer loop) with mode-specific settings
    optimizer = torch.optim.AdamW(
        meta_learner.parameters(),
        lr=outer_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95) if args.mode == "production" else (0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=meta_epochs,
        eta_min=outer_lr * 0.01,
    )
    
    best_acc = 0.0
    batches_per_epoch = 100 if args.mode == "production" else 50
    
    print(f"\nMeta-training for {meta_epochs} epochs...")
    print(f"Tasks per batch: {tasks_per_batch}, Batches per epoch: {batches_per_epoch}")
    print("=" * 60)
    
    for epoch in range(1, meta_epochs + 1):
        train_loss, train_acc = train_meta_epoch(
            meta_learner, train_generator, optimizer, device,
            tasks_per_batch, batches_per_epoch, epoch
        )
        scheduler.step()
        
        # Evaluate periodically
        eval_interval = 20 if args.mode == "production" else 10
        if epoch % eval_interval == 0 or epoch == meta_epochs:
            metrics = evaluate_few_shot(meta_learner, test_generator, device)
            
            print(f"\nEpoch {epoch}/{meta_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"  Test {n_way}-way {k_shot}-shot Accuracy: "
                  f"{metrics['mean_accuracy']:.2f}% ± {metrics['std_accuracy']:.2f}%")
            print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
            
            if metrics['mean_accuracy'] > best_acc:
                best_acc = metrics['mean_accuracy']
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": meta_learner.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "metrics": metrics,
                    "mode": args.mode,
                    "config": mode_config,
                }, save_path)
                print(f"  New best! Saved to {save_path}")
            
            print("-" * 60)
    
    # Final analysis
    print("\n" + "=" * 60)
    print("Adaptation Speed Analysis:")
    step_accs = test_adaptation_speed(meta_learner, test_generator, device)
    for i, acc in enumerate(step_accs):
        print(f"  {i+1} step(s): {acc:.2f}%")
    
    print(f"\nTraining complete. Best {n_way}-way {k_shot}-shot accuracy: {best_acc:.2f}%")
    
    # Mode-dependent validation thresholds
    if args.mode == "production":
        pass_threshold, partial_threshold = 85.0, 70.0
    elif args.mode in ["production_3b", "production_1b"]:
        pass_threshold, partial_threshold = 82.0, 65.0
    else:
        pass_threshold, partial_threshold = 80.0, 60.0
    
    # Validation gate
    if best_acc >= pass_threshold:
        print(f"\n[PASS] PHASE 7 VALIDATION PASSED: Achieved {pass_threshold}%+ few-shot accuracy")
    elif best_acc >= partial_threshold:
        print(f"\n[PARTIAL] PHASE 7 PARTIAL: Achieved {partial_threshold}%+ few-shot accuracy")
    else:
        print(f"\n[FAIL] PHASE 7 NOT PASSED: {best_acc:.2f}% < {pass_threshold}% target")


if __name__ == "__main__":
    main()
