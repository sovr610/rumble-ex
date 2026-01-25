#!/usr/bin/env python3
"""
Phase 1 Training Script: SNN Core on Vision Datasets

Supports both development mode (MNIST) and production mode (ImageNet-21K).

Development mode: Validates SNN implementation on MNIST (target: 98%)
Production mode: Full-scale training on ImageNet-21K for 7B model

Usage:
    # Development (quick validation)
    python scripts/train_phase1.py --mode dev
    
    # Production (7B scale)
    python scripts/train_phase1.py --mode production --dataset imagenet21k
    
    # Multi-GPU production
    torchrun --nproc_per_node=8 scripts/train_phase1.py \
        --mode production --dataset imagenet21k
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
    parser = argparse.ArgumentParser(description="Train SNN Vision Encoder")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="dev",
                        choices=["dev", "production", "production_3b", "production_1b"],
                        help="Training mode (dev uses MNIST, production uses ImageNet)")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "cifar10", "imagenet21k", "laion"],
                        help="Dataset to use")
    
    # Training hyperparameters (overrides config defaults)
    parser.add_argument("--epochs", type=int, default=None, 
                        help="Number of epochs (default: 10 for dev, 90 for production)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size per GPU (default: 128 dev, 64 production)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (uses config default if not specified)")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                        help="Gradient accumulation steps")
    
    # SNN-specific (for dev mode override)
    parser.add_argument("--beta", type=float, default=None, help="Membrane decay")
    parser.add_argument("--num-steps", type=int, default=None, help="Simulation timesteps")
    parser.add_argument("--hidden", type=int, nargs="+", default=None, help="Hidden layer sizes")
    
    # Model architecture
    parser.add_argument("--model", choices=["ff", "conv"], default="conv", help="Model type")
    
    # Hardware
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (PyTorch 2+)")
    
    # I/O
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--save-path", type=str, default=None, help="Save path (auto-generated if not set)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def get_config(mode: str) -> BrainAIConfig:
    """Get configuration based on training mode."""
    if mode == "production":
        return BrainAIConfig.production_7b()
    elif mode == "production_3b":
        return BrainAIConfig.production_3b()
    elif mode == "production_1b":
        return BrainAIConfig.production_1b()
    else:
        return BrainAIConfig.minimal()


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

    return train_loader, test_loader, 10  # 10 classes


def load_cifar10(data_dir: str, batch_size: int):
    """Load CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader, 10  # 10 classes


def load_imagenet21k(data_dir: str, batch_size: int, image_size: int = 224):
    """Load ImageNet-21K dataset (requires download)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required for ImageNet-21K. Install: pip install datasets")
    
    transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Check for local ImageNet first (faster)
    imagenet_path = Path(data_dir) / "imagenet"
    if imagenet_path.exists():
        print(f"Found local ImageNet at {imagenet_path}")
        train_dataset = datasets.ImageFolder(
            imagenet_path / "train", transform=transform
        )
        val_dataset = datasets.ImageFolder(
            imagenet_path / "val", transform=transform
        )
        num_classes = len(train_dataset.classes)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=8, pin_memory=True
        )
        
        return train_loader, val_loader, num_classes
    else:
        # Fall back to HuggingFace streaming
        print("Loading ImageNet-21K from HuggingFace (streaming)...")
        print("This may take a while for first download (~1.2TB)")
        print("For faster loading, download to ./data/imagenet/")
        
        # Use streaming for large dataset
        hf_dataset = load_dataset(
            "imagenet-21k", 
            split="train",
            streaming=True
        )
        
        # Wrap streaming dataset for DataLoader
        # Note: Streaming doesn't support random access easily
        raise NotImplementedError(
            "Streaming ImageNet-21K not yet fully supported. "
            "Please download ImageNet locally to ./data/imagenet/ or "
            "use --dataset mnist/cifar10 for testing."
        )


def load_data(dataset: str, data_dir: str, batch_size: int, config: BrainAIConfig):
    """Load dataset based on name."""
    if dataset == "mnist":
        return load_mnist(data_dir, batch_size)
    elif dataset == "cifar10":
        return load_cifar10(data_dir, batch_size)
    elif dataset == "imagenet21k":
        return load_imagenet21k(data_dir, batch_size, config.encoder.vision_image_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    scaler=None,
    accum_steps: int = 1,
) -> tuple:
    """Train for one epoch with optional AMP and gradient accumulation."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Flatten for feedforward SNN
        if hasattr(model, 'conv_layers'):
            pass  # ConvSNN handles 4D input
        else:
            data = data.view(data.size(0), -1)

        # Forward pass with optional AMP
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                spike_record, mem = model(data)
                output = spike_record.sum(dim=0)
                loss = criterion(output, target) / accum_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            spike_record, mem = model(data)
            output = spike_record.sum(dim=0)
            loss = criterion(output, target) / accum_steps
            loss.backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item() * accum_steps:.4f}")

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
    
    # Get configuration based on mode
    config = get_config(args.mode)
    print(f"\nTraining mode: {args.mode}")
    print(f"Configuration: {type(config).__name__}")
    
    # Set defaults based on mode
    if args.mode == "dev":
        epochs = args.epochs or 10
        batch_size = args.batch_size or 128
        lr = args.lr or 1e-3
        hidden = args.hidden or [800, 400]
        beta = args.beta or 0.9
        num_steps = args.num_steps or 25
        dataset = args.dataset if args.dataset != "imagenet21k" else "mnist"
        save_path = args.save_path or "checkpoints/snn_mnist.pth"
    else:
        # Production mode - use config defaults
        epochs = args.epochs or 90
        batch_size = args.batch_size or config.training.batch_size
        lr = args.lr or config.training.learning_rate
        hidden = args.hidden or config.snn.hidden_sizes
        beta = args.beta or config.snn.beta
        num_steps = args.num_steps or config.snn.num_timesteps
        dataset = args.dataset
        save_path = args.save_path or f"checkpoints/snn_{args.mode}_{dataset}.pth"
    
    # Load data
    print(f"\nLoading {dataset} dataset...")
    train_loader, test_loader, num_classes = load_data(
        dataset, args.data_dir, batch_size, config
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Classes: {num_classes}")

    # Create model based on dataset
    input_size = 784 if dataset == "mnist" else None  # Conv for other datasets
    input_channels = 1 if dataset == "mnist" else 3
    
    if args.model == "ff" and dataset == "mnist":
        model = SNNCore(
            input_size=input_size,
            hidden_sizes=hidden,
            output_size=num_classes,
            beta=beta,
            num_steps=num_steps,
            surrogate=config.snn.surrogate,
            dropout=config.snn.dropout,
        )
        print(f"\nCreated feedforward SNN: {input_size} -> {hidden} -> {num_classes}")
    else:
        # Use ConvSNN for vision datasets
        if args.mode == "dev":
            channels = [32, 64, 128]
            fc_sizes = [256]
        else:
            # Production scale channels
            channels = config.encoder.vision_channels
            fc_sizes = [config.encoder.output_dim // 2]
        
        model = ConvSNN(
            input_channels=input_channels,
            channels=channels,
            fc_sizes=fc_sizes,
            num_classes=num_classes,
            beta=beta,
            num_steps=num_steps,
        )
        print(f"\nCreated ConvSNN: {input_channels} -> {channels} -> {fc_sizes} -> {num_classes}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Compile model if requested (PyTorch 2+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Mixed precision scaler
    scaler = None
    if args.use_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler('cuda')
        print("Using automatic mixed precision (AMP)")

    # Optimizer and loss
    if args.mode == "dev":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        # AdamW with weight decay for production
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr,
            betas=(config.training.beta1, config.training.beta2),
            weight_decay=config.training.weight_decay
        )
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler for production
    scheduler = None
    if args.mode != "dev":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=config.training.min_learning_rate
        )
        print(f"Using cosine LR schedule: {lr} -> {config.training.min_learning_rate}")
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('accuracy', 0.0)
        print(f"  Resumed from epoch {start_epoch-1}, best acc: {best_acc:.2f}%")

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print("=" * 60)

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            scaler=scaler, accum_steps=args.gradient_accumulation
        )
        test_acc = run_evaluation(model, test_loader, device)
        
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = lr

        print(f"\nEpoch {epoch}/{epochs} [lr={current_lr:.2e}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            # Save checkpoint
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": test_acc,
                "config": {
                    "mode": args.mode,
                    "dataset": dataset,
                    "hidden": hidden,
                    "beta": beta,
                    "num_steps": num_steps,
                }
            }, save_path)
            print(f"  ★ New best! Saved to {save_path}")

        print("-" * 60)

    print("\n" + "=" * 60)
    print(f"Training complete. Best accuracy: {best_acc:.2f}%")

    # Validation gate check
    target_acc = 98.0 if dataset == "mnist" else 80.0  # Different targets
    if best_acc >= target_acc:
        print(f"\n[PASS] PHASE 1 VALIDATION PASSED: Achieved {target_acc}%+ accuracy")
    elif best_acc >= target_acc - 3.0:
        print(f"\n[PARTIAL] PHASE 1 PARTIAL: Close to {target_acc}% target")
    else:
        print(f"\n[FAIL] PHASE 1 NOT PASSED: {best_acc:.2f}% < {target_acc}% target")
        print("  Try: more epochs, different hyperparameters, or conv model")


if __name__ == "__main__":
    main()
