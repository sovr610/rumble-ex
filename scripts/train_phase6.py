#!/usr/bin/env python3
"""
Phase 6 Training Script: Neuro-Symbolic Reasoning

Trains the dual-process System 1/System 2 reasoning module with
fuzzy logic operations for verified multi-step reasoning.

Target: 85%+ accuracy on symbolic reasoning tasks with interpretable traces

Usage:
    python scripts/train_phase6.py
    python scripts/train_phase6.py --epochs 30 --reasoning-steps 5
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_ai.reasoning.symbolic import SymbolicReasoner, SymbolicConfig, FuzzyLogic
from brain_ai.reasoning.system2 import DualProcessReasoner, System2Config


def parse_args():
    parser = argparse.ArgumentParser(description="Train Neuro-Symbolic Reasoner")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--reasoning-steps", type=int, default=5, help="Max reasoning steps")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--save-path", type=str, default="checkpoints/neuro_symbolic.pth", help="Save path")
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class LogicalReasoningDataset(Dataset):
    """
    Dataset for testing neuro-symbolic reasoning.
    
    Tasks include:
    1. Propositional logic inference (A AND B -> C)
    2. Transitive reasoning (A > B, B > C -> A > C)
    3. Multi-step deduction chains
    4. Fuzzy logic operations
    
    Each sample has:
    - Encoded premises
    - Reasoning query
    - Ground truth answer
    - Difficulty level (for System 1/2 routing)
    """
    
    def __init__(
        self,
        num_samples: int = 5000,
        input_dim: int = 128,
        max_steps: int = 5,
        train: bool = True,
    ):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.max_steps = max_steps
        
        np.random.seed(42 if train else 123)
        
        self.samples = []
        self.labels = []
        self.difficulties = []
        self.rule_types = []
        
        for i in range(num_samples):
            sample, label, difficulty, rule_type = self._generate_sample()
            self.samples.append(sample)
            self.labels.append(label)
            self.difficulties.append(difficulty)
            self.rule_types.append(rule_type)
    
    def _generate_sample(self) -> Tuple[torch.Tensor, int, float, str]:
        """Generate a single reasoning sample."""
        
        # Choose task type
        task_type = np.random.choice([
            'conjunction', 'disjunction', 'implication',
            'transitive', 'chain', 'fuzzy_threshold'
        ])
        
        # Generate based on task
        if task_type == 'conjunction':
            # A AND B = C (easy, System 1)
            a = np.random.rand() > 0.5
            b = np.random.rand() > 0.5
            c = a and b
            
            features = self._encode_logic(a, b, 'and')
            label = 1 if c else 0
            difficulty = 0.2  # Easy
            
        elif task_type == 'disjunction':
            # A OR B = C
            a = np.random.rand() > 0.5
            b = np.random.rand() > 0.5
            c = a or b
            
            features = self._encode_logic(a, b, 'or')
            label = 1 if c else 0
            difficulty = 0.3
            
        elif task_type == 'implication':
            # A -> B (A implies B)
            a = np.random.rand() > 0.5
            b = np.random.rand() > 0.5
            c = (not a) or b  # Implication: ¬A ∨ B
            
            features = self._encode_logic(a, b, 'implies')
            label = 1 if c else 0
            difficulty = 0.5
            
        elif task_type == 'transitive':
            # A > B, B > C -> A > C (requires System 2)
            vals = np.random.rand(3)
            a_gt_b = vals[0] > vals[1]
            b_gt_c = vals[1] > vals[2]
            a_gt_c = vals[0] > vals[2]
            
            features = self._encode_transitive(vals, a_gt_b, b_gt_c)
            label = 1 if a_gt_c else 0
            difficulty = 0.7
            
        elif task_type == 'chain':
            # Multi-step chain: if A then B, if B then C, A -> C?
            chain_length = np.random.randint(2, 4)
            start_val = np.random.rand() > 0.5
            
            # Propagate through chain
            current = start_val
            for _ in range(chain_length):
                current = current  # Each step preserves value (for simplicity)
            
            features = self._encode_chain(start_val, chain_length)
            label = 1 if current else 0
            difficulty = 0.3 + chain_length * 0.2
            
        else:  # fuzzy_threshold
            # Fuzzy: if value > 0.5, then True (with noise)
            val = np.random.rand()
            noise = np.random.randn() * 0.1
            result = (val + noise) > 0.5
            
            features = self._encode_fuzzy(val, 0.5)
            label = 1 if result else 0
            difficulty = 0.4
        
        return (
            torch.tensor(features, dtype=torch.float32),
            label,
            min(1.0, difficulty),
            task_type,
        )
    
    def _encode_logic(self, a: bool, b: bool, op: str) -> np.ndarray:
        """Encode a logical operation."""
        features = np.zeros(self.input_dim)
        
        # Encode operands
        features[0] = float(a)
        features[1] = float(b)
        
        # Encode operation
        op_idx = {'and': 0, 'or': 1, 'implies': 2, 'not': 3}
        features[10 + op_idx.get(op, 0)] = 1.0
        
        # Add random features for context
        features[20:50] = np.random.randn(30) * 0.5
        
        return features
    
    def _encode_transitive(self, vals: np.ndarray, a_gt_b: bool, b_gt_c: bool) -> np.ndarray:
        """Encode transitive relation."""
        features = np.zeros(self.input_dim)
        
        # Encode values
        features[0:3] = vals
        
        # Encode relations
        features[10] = float(a_gt_b)
        features[11] = float(b_gt_c)
        
        # Transitive marker
        features[20] = 1.0
        
        features[30:60] = np.random.randn(30) * 0.3
        
        return features
    
    def _encode_chain(self, start: bool, length: int) -> np.ndarray:
        """Encode reasoning chain."""
        features = np.zeros(self.input_dim)
        
        features[0] = float(start)
        features[5] = length / 5.0  # Normalized length
        
        # Chain marker
        features[25] = 1.0
        
        features[30:60] = np.random.randn(30) * 0.3
        
        return features
    
    def _encode_fuzzy(self, val: float, threshold: float) -> np.ndarray:
        """Encode fuzzy threshold task."""
        features = np.zeros(self.input_dim)
        
        features[0] = val
        features[1] = threshold
        features[2] = val - threshold  # Margin
        
        # Fuzzy marker
        features[30] = 1.0
        
        features[40:70] = np.random.randn(30) * 0.3
        
        return features
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return (
            self.samples[idx],
            self.labels[idx],
            self.difficulties[idx],
        )


class NeuroSymbolicClassifier(nn.Module):
    """
    Combined System 1/System 2 classifier with symbolic reasoning.
    
    - System 1: Fast, direct classification for easy problems
    - System 2: Deliberate multi-step reasoning for hard problems
    - Routing based on confidence
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_reasoning_steps: int = 5,
        confidence_threshold: float = 0.7,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.confidence_threshold = confidence_threshold
        
        # System 1: Fast path
        self.system1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        # System 1 confidence
        self.system1_confidence = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # System 2: Deliberate reasoning
        self.system2_config = System2Config(
            hidden_dim=hidden_dim,
            num_reasoning_steps=num_reasoning_steps,
            confidence_threshold=confidence_threshold,
        )
        
        # System 2 encoder
        self.system2_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Reasoning GRU
        self.reasoning_gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Fuzzy logic module
        self.fuzzy_logic = FuzzyLogic(logic_type="product")
        
        # Predicate networks (for logical operations)
        self.predicate_and = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        self.predicate_or = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # System 2 output
        self.system2_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        # System 2 confidence
        self.system2_confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_trace: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with System 1/2 routing.
        """
        batch_size = x.shape[0]
        
        # System 1 (always run for comparison)
        s1_logits = self.system1(x)
        s1_confidence = self.system1_confidence(x).squeeze(-1)
        
        # System 2 (deliberate reasoning)
        s2_state = self.system2_encoder(x)
        reasoning_trace = [s2_state.clone()]
        
        for step in range(self.system2_config.num_reasoning_steps):
            # GRU reasoning step
            s2_state = self.reasoning_gru(s2_state, s2_state)
            
            # Apply fuzzy logic operations
            and_result = self.predicate_and(s2_state)
            or_result = self.predicate_or(s2_state)
            
            # Combine logical results back into state
            logic_features = torch.cat([and_result, or_result], dim=-1)
            s2_state = s2_state + F.pad(logic_features, (0, self.hidden_dim - 2))
            
            reasoning_trace.append(s2_state.clone())
        
        s2_logits = self.system2_output(s2_state)
        s2_confidence = self.system2_confidence(s2_state).squeeze(-1)
        
        # Route based on System 1 confidence
        use_system2 = (s1_confidence < self.confidence_threshold).float()
        
        # Blend outputs
        final_logits = (
            (1 - use_system2.unsqueeze(-1)) * s1_logits +
            use_system2.unsqueeze(-1) * s2_logits
        )
        final_confidence = (
            (1 - use_system2) * s1_confidence +
            use_system2 * s2_confidence
        )
        
        result = {
            'logits': final_logits,
            'confidence': final_confidence,
            's1_logits': s1_logits,
            's1_confidence': s1_confidence,
            's2_logits': s2_logits,
            's2_confidence': s2_confidence,
            'use_system2': use_system2,
        }
        
        if return_trace:
            result['trace'] = torch.stack(reasoning_trace, dim=1)
        
        return result


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    s2_usage = 0
    
    for batch_idx, (data, target, difficulty) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        difficulty = difficulty.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Classification loss
        loss = criterion(output['logits'], target)
        
        # Confidence regularization (encourage calibrated confidence)
        predicted = output['logits'].argmax(dim=1)
        correct_mask = (predicted == target).float()
        confidence_target = correct_mask * 0.9 + (1 - correct_mask) * 0.1
        confidence_loss = F.mse_loss(output['confidence'], confidence_target)
        
        # System 2 routing loss (use S2 for hard problems)
        routing_target = (difficulty > 0.5).float()
        routing_loss = F.binary_cross_entropy(
            output['use_system2'],
            routing_target,
        )
        
        # Total loss
        total_loss_batch = loss + 0.1 * confidence_loss + 0.1 * routing_loss
        total_loss_batch.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pred = output['logits'].argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
        s2_usage += output['use_system2'].sum().item()
        
        if batch_idx % 30 == 0:
            print(f"  Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.4f}")
    
    return (
        total_loss / len(train_loader),
        100.0 * correct / total,
        100.0 * s2_usage / total,
    )


def evaluate(model, test_loader, device) -> Dict[str, float]:
    """Evaluate model with detailed metrics."""
    model.eval()
    correct = 0
    total = 0
    s1_correct = 0
    s2_correct = 0
    s1_total = 0
    s2_total = 0
    
    difficulties = {'easy': [], 'medium': [], 'hard': []}
    
    with torch.no_grad():
        for data, target, difficulty in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data, return_trace=True)
            
            pred = output['logits'].argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            # Analyze per-system performance
            use_s2 = output['use_system2'] > 0.5
            
            s1_mask = ~use_s2
            s2_mask = use_s2
            
            if s1_mask.any():
                s1_pred = output['s1_logits'][s1_mask].argmax(dim=1)
                s1_correct += (s1_pred == target[s1_mask]).sum().item()
                s1_total += s1_mask.sum().item()
            
            if s2_mask.any():
                s2_pred = output['s2_logits'][s2_mask].argmax(dim=1)
                s2_correct += (s2_pred == target[s2_mask]).sum().item()
                s2_total += s2_mask.sum().item()
            
            # Track by difficulty
            for i, d in enumerate(difficulty):
                is_correct = pred[i] == target[i]
                if d < 0.4:
                    difficulties['easy'].append(is_correct.item())
                elif d < 0.7:
                    difficulties['medium'].append(is_correct.item())
                else:
                    difficulties['hard'].append(is_correct.item())
    
    return {
        'accuracy': 100.0 * correct / total,
        's1_accuracy': 100.0 * s1_correct / max(s1_total, 1),
        's2_accuracy': 100.0 * s2_correct / max(s2_total, 1),
        's1_usage': 100.0 * s1_total / total,
        's2_usage': 100.0 * s2_total / total,
        'easy_accuracy': 100.0 * np.mean(difficulties['easy']) if difficulties['easy'] else 0,
        'medium_accuracy': 100.0 * np.mean(difficulties['medium']) if difficulties['medium'] else 0,
        'hard_accuracy': 100.0 * np.mean(difficulties['hard']) if difficulties['hard'] else 0,
    }


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating logical reasoning datasets...")
    train_dataset = LogicalReasoningDataset(
        num_samples=10000,
        input_dim=128,
        max_steps=args.reasoning_steps,
        train=True,
    )
    test_dataset = LogicalReasoningDataset(
        num_samples=2000,
        input_dim=128,
        max_steps=args.reasoning_steps,
        train=False,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    # Create model
    model = NeuroSymbolicClassifier(
        input_dim=128,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        num_reasoning_steps=args.reasoning_steps,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Neuro-Symbolic Classifier parameters: {total_params:,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, s2_usage = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        metrics = evaluate(model, test_loader, device)
        scheduler.step()
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  System 1 Acc: {metrics['s1_accuracy']:.2f}% (usage: {metrics['s1_usage']:.1f}%)")
        print(f"  System 2 Acc: {metrics['s2_accuracy']:.2f}% (usage: {metrics['s2_usage']:.1f}%)")
        print(f"  By Difficulty - Easy: {metrics['easy_accuracy']:.1f}% | "
              f"Medium: {metrics['medium_accuracy']:.1f}% | Hard: {metrics['hard_accuracy']:.1f}%")
        
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            save_dir = Path(args.save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            }, args.save_path)
            print(f"  New best! Saved to {args.save_path}")
        
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print(f"Training complete. Best accuracy: {best_acc:.2f}%")
    
    # Validation gate
    if best_acc >= 85.0:
        print("\n[PASS] PHASE 6 VALIDATION PASSED: Achieved 85%+ accuracy")
    elif best_acc >= 70.0:
        print("\n[PARTIAL] PHASE 6 PARTIAL: Achieved 70%+ accuracy")
    else:
        print(f"\n[FAIL] PHASE 6 NOT PASSED: {best_acc:.2f}% < 85% target")


if __name__ == "__main__":
    main()
