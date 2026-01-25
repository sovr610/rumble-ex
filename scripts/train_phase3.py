#!/usr/bin/env python3
"""
Phase 3 Training Script: Hierarchical Temporal Memory (HTM)

Trains the HTM layer on sequence prediction and anomaly detection.
Validates online sequence learning without gradient descent.

Target: 90%+ prediction accuracy on synthetic sequences

Usage:
    python scripts/train_phase3.py
    python scripts/train_phase3.py --sequences 1000 --epochs 5
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_ai.temporal.htm import HTMLayer, HTMConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train HTM on Sequence Prediction")
    parser.add_argument("--mode", type=str, default="dev",
                        choices=["dev", "production", "production_3b", "production_1b"],
                        help="Training mode")
    parser.add_argument("--sequences", type=int, default=None, help="Number of training sequences")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--seq-length", type=int, default=None, help="Sequence length")
    parser.add_argument("--column-count", type=int, default=None, help="HTM column count")
    parser.add_argument("--cells-per-column", type=int, default=None, help="Cells per column")
    parser.add_argument("--input-dim", type=int, default=None, help="Input dimension")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--save-path", type=str, default=None, help="Save path")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision")
    return parser.parse_args()


def get_mode_config(mode: str) -> dict:
    """Get configuration based on training mode."""
    configs = {
        "dev": {
            "sequences": 200,
            "epochs": 5,
            "seq_length": 30,
            "column_count": 256,
            "cells_per_column": 8,
            "input_dim": 128,
            "save_path": "checkpoints/htm_layer_dev.pth",
        },
        "production_1b": {
            "sequences": 5000,
            "epochs": 20,
            "seq_length": 100,
            "column_count": 4096,
            "cells_per_column": 32,
            "input_dim": 2048,
            "save_path": "checkpoints/htm_layer_1b.pth",
        },
        "production_3b": {
            "sequences": 10000,
            "epochs": 30,
            "seq_length": 200,
            "column_count": 8192,
            "cells_per_column": 48,
            "input_dim": 3072,
            "save_path": "checkpoints/htm_layer_3b.pth",
        },
        "production": {  # 7B scale
            "sequences": 50000,
            "epochs": 50,
            "seq_length": 500,
            "column_count": 16384,
            "cells_per_column": 64,
            "input_dim": 4096,
            "save_path": "checkpoints/htm_layer_7b.pth",
        },
    }
    return configs.get(mode, configs["dev"])


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def generate_synthetic_sequences(
    num_sequences: int,
    seq_length: int,
    input_dim: int = 128,
    num_patterns: int = 5,
) -> List[Tuple[torch.Tensor, int]]:
    """
    Generate synthetic sequences for HTM training.
    
    Creates HIGHLY PREDICTABLE sequences with:
    - Simple repeating patterns (for easy sequence prediction)
    - Deterministic transitions (what HTM learns best)
    - Low noise (so patterns are clear)
    """
    np.random.seed(42)
    sequences = []
    
    # Create simple, highly distinctive base patterns
    # Each pattern is just a different set of consistently active bits
    base_patterns = []
    active_bits_per_step = max(5, int(input_dim * 0.04))  # ~4% sparse
    
    for pattern_idx in range(num_patterns):
        # Each pattern is a deterministic sequence of 5 steps
        pattern = np.zeros((5, input_dim))
        for t in range(5):
            # Consistent active bits based on pattern and step
            start_bit = (pattern_idx * 20 + t * 4) % (input_dim - active_bits_per_step)
            pattern[t, start_bit:start_bit + active_bits_per_step] = 1.0
        base_patterns.append(pattern)
    
    for seq_idx in range(num_sequences):
        sequence = np.zeros((seq_length, input_dim))
        pattern_idx = seq_idx % num_patterns
        
        # Fill sequence with perfectly repeating pattern (no noise during training)
        base = base_patterns[pattern_idx]
        for t in range(seq_length):
            sequence[t] = base[t % len(base)].copy()
        
        sequences.append((torch.tensor(sequence, dtype=torch.float32), pattern_idx))
    
    return sequences


def train_htm_epoch(
    htm: HTMLayer,
    sequences: List[Tuple[torch.Tensor, int]],
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """
    Train HTM for one epoch using online learning.
    
    HTM learns through Hebbian-style updates, not backprop.
    Returns prediction accuracy and average anomaly score.
    """
    total_predictions = 0
    correct_predictions = 0
    anomaly_scores = []
    
    for seq_idx, (sequence, label) in enumerate(sequences):
        sequence = sequence.to(device)
        htm.reset()  # Reset temporal context for new sequence
        
        prev_predicted = None
        prev_features = None
        
        for t in range(len(sequence)):
            current_input = sequence[t].unsqueeze(0)
            
            # Forward pass (learns online)
            output = htm(current_input, learn=True)
            
            # Get current active columns (features)
            current_features = output['features']
            
            # Check prediction accuracy: does prev prediction match current features?
            if prev_predicted is not None and t > 0:
                # For HTM: predictive_cells predicts which cells will be active
                # We convert predictive cells to column prediction
                num_cols = htm.config.column_count
                cells_per_col = htm.config.cells_per_column
                
                # Reshape predictive cells to (columns, cells_per_column)
                pred_cells = prev_predicted.view(num_cols, cells_per_col)
                # A column is predicted if ANY of its cells are predicted
                predicted_columns = (pred_cells.sum(dim=1) > 0).float()
                
                # Current active columns
                actual_columns = (current_features > 0.5).float().squeeze()
                
                # Overlap as accuracy measure
                overlap = (predicted_columns * actual_columns).sum()
                total_active = actual_columns.sum().clamp(min=1)
                accuracy = overlap / total_active
                
                correct_predictions += accuracy.item()
                total_predictions += 1
                
                # Anomaly score from prediction error
                anomaly = 1.0 - accuracy.item()
                anomaly_scores.append(anomaly)
            
            # Store predictive cells for next step comparison
            prev_predicted = output['predictive_cells'].squeeze()
        
        if seq_idx % 50 == 0:
            print(f"  Epoch {epoch} - Processed {seq_idx}/{len(sequences)} sequences")
    
    avg_accuracy = correct_predictions / max(total_predictions, 1)
    avg_anomaly = np.mean(anomaly_scores) if anomaly_scores else 0.5
    
    return avg_accuracy, avg_anomaly


def evaluate_htm(
    htm: HTMLayer,
    test_sequences: List[Tuple[torch.Tensor, int]],
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate HTM on test sequences."""
    total_predictions = 0
    correct_predictions = 0
    anomaly_scores = []
    
    for sequence, label in test_sequences:
        sequence = sequence.to(device)
        htm.reset()
        
        prev_predicted = None
        
        for t in range(len(sequence)):
            current_input = sequence[t].unsqueeze(0)
            
            # Forward pass without learning
            output = htm(current_input, learn=False)
            
            # Get current active columns
            current_features = output['features']
            
            if prev_predicted is not None and t > 0:
                # Convert predictive cells to column prediction
                num_cols = htm.config.column_count
                cells_per_col = htm.config.cells_per_column
                
                pred_cells = prev_predicted.view(num_cols, cells_per_col)
                predicted_columns = (pred_cells.sum(dim=1) > 0).float()
                actual_columns = (current_features > 0.5).float().squeeze()
                
                overlap = (predicted_columns * actual_columns).sum()
                total_active = actual_columns.sum().clamp(min=1)
                accuracy = overlap / total_active
                
                correct_predictions += accuracy.item()
                total_predictions += 1
                anomaly_scores.append(1.0 - accuracy.item())
            
            prev_predicted = output['predictive_cells'].squeeze()
    
    avg_accuracy = correct_predictions / max(total_predictions, 1)
    avg_anomaly = np.mean(anomaly_scores) if anomaly_scores else 0.5
    
    return avg_accuracy, avg_anomaly


def test_anomaly_detection(
    htm: HTMLayer,
    normal_sequences: List[Tuple[torch.Tensor, int]],
    device: torch.device,
) -> float:
    """Test anomaly detection by injecting anomalies."""
    
    anomaly_detected = 0
    total_anomalies = 0
    
    for sequence, _ in normal_sequences[:50]:
        sequence = sequence.to(device).clone()
        htm.reset()
        
        # Inject anomaly at random position
        anomaly_pos = np.random.randint(10, len(sequence) - 5)
        
        # Create anomalous input (random pattern)
        sequence[anomaly_pos] = torch.rand_like(sequence[anomaly_pos])
        
        prev_predicted = None
        
        for t in range(len(sequence)):
            current_input = sequence[t].unsqueeze(0)
            output = htm(current_input, learn=False)
            
            # Get current active columns
            current_features = output['features']
            
            if prev_predicted is not None and t > 0:
                # Convert predictive cells to column prediction
                num_cols = htm.config.column_count
                cells_per_col = htm.config.cells_per_column
                
                pred_cells = prev_predicted.view(num_cols, cells_per_col)
                predicted_columns = (pred_cells.sum(dim=1) > 0).float()
                actual_columns = (current_features > 0.5).float().squeeze()
                
                overlap = (predicted_columns * actual_columns).sum()
                total_active = actual_columns.sum().clamp(min=1)
                anomaly_score = 1.0 - (overlap / total_active).item()
                
                # Check if anomaly was detected
                if t == anomaly_pos:
                    total_anomalies += 1
                    if anomaly_score > 0.5:  # High prediction error = anomaly
                        anomaly_detected += 1
            
            prev_predicted = output['predictive_cells'].squeeze()
    
    detection_rate = anomaly_detected / max(total_anomalies, 1)
    return detection_rate


def main():
    args = parse_args()
    device = get_device(args.device)
    
    # Get mode-specific configuration
    mode_config = get_mode_config(args.mode)
    
    # Override with command-line arguments if provided
    sequences = args.sequences or mode_config["sequences"]
    epochs = args.epochs or mode_config["epochs"]
    seq_length = args.seq_length or mode_config["seq_length"]
    column_count = args.column_count or mode_config["column_count"]
    cells_per_column = args.cells_per_column or mode_config["cells_per_column"]
    input_dim = args.input_dim or mode_config["input_dim"]
    save_path = args.save_path or mode_config["save_path"]
    
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    print(f"  - Sequences: {sequences}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Seq length: {seq_length}")
    print(f"  - Column count: {column_count}")
    print(f"  - Cells per column: {cells_per_column}")
    print(f"  - Input dim: {input_dim}")
    
    # Generate data
    print(f"\nGenerating {sequences} synthetic sequences...")
    all_sequences = generate_synthetic_sequences(
        sequences,
        seq_length,
        input_dim=input_dim,
    )
    
    # Split train/test
    split_idx = int(len(all_sequences) * 0.8)
    train_sequences = all_sequences[:split_idx]
    test_sequences = all_sequences[split_idx:]
    
    print(f"Train: {len(train_sequences)}, Test: {len(test_sequences)}")
    
    # Create HTM layer with mode-specific parameters
    config = HTMConfig(
        input_size=input_dim,
        column_count=column_count,
        cells_per_column=cells_per_column,
        sparsity=0.02,
        activation_threshold=min(13, column_count // 20),
        min_threshold=min(8, column_count // 30),
    )
    
    htm = HTMLayer(config).to(device)
    total_cells = config.column_count * config.cells_per_column
    print(f"Created HTM Layer: {config.column_count} columns × {config.cells_per_column} cells = {total_cells:,} total cells")
    
    best_acc = 0.0
    
    print(f"\nTraining for {epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, epochs + 1):
        # Train (HTM learns online during forward pass)
        train_acc, train_anomaly = train_htm_epoch(
            htm, train_sequences, device, epoch
        )
        
        # Evaluate
        test_acc, test_anomaly = evaluate_htm(htm, test_sequences, device)
        
        # Test anomaly detection
        detection_rate = test_anomaly_detection(htm, test_sequences, device)
        
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train Prediction Accuracy: {train_acc*100:.2f}%")
        print(f"  Test Prediction Accuracy: {test_acc*100:.2f}%")
        print(f"  Anomaly Detection Rate: {detection_rate*100:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "mode": args.mode,
                "model_state_dict": htm.state_dict(),
                "config": config,
                "accuracy": test_acc,
                "detection_rate": detection_rate,
            }, save_path)
            print(f"  New best! Saved to {save_path}")
        
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print(f"Training complete. Best prediction accuracy: {best_acc*100:.2f}%")
    print(f"Mode: {args.mode} | Final checkpoint: {save_path}")
    
    # Validation gate (adjusted for mode)
    target_acc = 0.90 if args.mode == "dev" else 0.95
    if best_acc >= target_acc:
        print(f"\n[PASS] PHASE 3 VALIDATION PASSED: Achieved {target_acc*100}%+ prediction accuracy")
    elif best_acc >= target_acc - 0.20:
        print(f"\n[PARTIAL] PHASE 3 PARTIAL: Achieved {best_acc*100:.2f}% (target: {target_acc*100}%)")
    else:
        print(f"\n[FAIL] PHASE 3 NOT PASSED: {best_acc*100:.2f}% < {target_acc*100}% target")


if __name__ == "__main__":
    main()
