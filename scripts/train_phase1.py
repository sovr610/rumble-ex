#!/usr/bin/env python3
"""
Phase 1 Training Script: SNN Core on MNIST

Validates the SNN implementation by training on MNIST.
Target: 98% accuracy.

Usage:
    python scripts/train_phase1.py
    python scripts/train_phase1.py --epochs 20 --batch-size 128
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_ai.core import SNNCore, ConvSNN
from brain_ai.config import BrainAIConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train SNN on MNIST")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.9, help="Membrane decay")
    parser.add_argument("--num-steps", type=int, default=25, help="Simulation timesteps")
    parser.add_argument("--hidden", type=int, nargs="+", default=[800, 400], help="Hidden layer sizes")
    parser.add_argument("--model", choices=["ff", "conv"], default="ff", help="Model type")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--save-path", type=str, default="checkpoints/snn_mnist.pth", help="Save path")
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_mnist(data_dir: str, batch_size: int):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Flatten for feedforward SNN
        if hasattr(model, 'conv_layers'):
            pass  # ConvSNN handles 4D input
        else:
            data = data.view(data.size(0), -1)

        optimizer.zero_grad()

        # Forward pass
        spike_record, mem = model(data)

        # Rate-coded output
        output = spike_record.sum(dim=0)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def run_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Run model accuracy check."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            if hasattr(model, 'conv_layers'):
                pass
            else:
                data = data.view(data.size(0), -1)

            spike_record, _ = model(data)
            output = spike_record.sum(dim=0)
            pred = output.argmax(dim=1)

            correct += (pred == target).sum().item()
            total += target.size(0)

    return 100.0 * correct / total


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load data
    print("Loading MNIST...")
    train_loader, test_loader = load_mnist(args.data_dir, args.batch_size)

    # Create model
    if args.model == "ff":
        model = SNNCore(
            input_size=784,
            hidden_sizes=args.hidden,
            output_size=10,
            beta=args.beta,
            num_steps=args.num_steps,
            surrogate="atan",
            dropout=0.2,
        )
        print(f"Created feedforward SNN: 784 -> {args.hidden} -> 10")
    else:
        model = ConvSNN(
            input_channels=1,
            channels=[32, 64, 128],
            fc_sizes=[256],
            num_classes=10,
            beta=args.beta,
            num_steps=args.num_steps,
        )
        print("Created ConvSNN: 1 -> [32, 64, 128] -> 256 -> 10")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0.0

    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        test_acc = run_evaluation(model, test_loader, device)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            # Save checkpoint
            save_dir = Path(args.save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": test_acc,
                "config": {
                    "hidden": args.hidden,
                    "beta": args.beta,
                    "num_steps": args.num_steps,
                }
            }, args.save_path)
            print(f"  New best! Saved to {args.save_path}")

        print("-" * 60)

    print("\n" + "=" * 60)
    print(f"Training complete. Best accuracy: {best_acc:.2f}%")

    # Validation gate check
    if best_acc >= 98.0:
        print("\n[PASS] PHASE 1 VALIDATION PASSED: Achieved 98%+ accuracy")
    elif best_acc >= 95.0:
        print("\n[PARTIAL] PHASE 1 PARTIAL: Achieved 95%+ accuracy (close to target)")
    else:
        print(f"\n[FAIL] PHASE 1 NOT PASSED: {best_acc:.2f}% < 98% target")
        print("  Try: more epochs, different hyperparameters, or conv model")


if __name__ == "__main__":
    main()
