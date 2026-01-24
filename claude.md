# Building a Brain-Inspired AI System: Complete Implementation Guide

> **Purpose**: This document provides step-by-step instructions for an LLM or developer to build an advanced brain-inspired AI architecture. It synthesizes 2024-2025 research into actionable code patterns, framework selection, and integration strategies.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Environment Setup](#2-environment-setup)
3. [Phase 1: Spiking Neural Network Core](#3-phase-1-spiking-neural-network-core)
4. [Phase 2: Event-Driven Sensory Processing](#4-phase-2-event-driven-sensory-processing)
5. [Phase 3: Hierarchical Temporal Memory](#5-phase-3-hierarchical-temporal-memory)
6. [Phase 4: Global Workspace & Working Memory](#6-phase-4-global-workspace--working-memory)
7. [Phase 5: Decision & Action System](#7-phase-5-decision--action-system)
8. [Phase 6: Neuro-Symbolic Reasoning](#8-phase-6-neuro-symbolic-reasoning)
9. [Phase 7: Meta-Learning & Plasticity](#9-phase-7-meta-learning--plasticity)
10. [System Integration](#10-system-integration)
11. [Deployment Options](#11-deployment-options)
12. [Key Resources & Papers](#12-key-resources--papers)

---

## 1. Architecture Overview

The system consists of seven integrated layers, each serving a distinct computational purpose that biological brains implement but conventional deep learning struggles to replicate:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BRAIN-INSPIRED AI ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ SENSORY  │───▶│   SNN    │───▶│   HTM    │───▶│ GLOBAL WORKSPACE │  │
│  │ (Events) │    │  (Core)  │    │(Temporal)│    │    (Attention)   │  │
│  └──────────┘    └──────────┘    └──────────┘    └────────┬─────────┘  │
│       │               │               │                   │            │
│       │               │               │          ┌────────┴────────┐   │
│       │               │               │          ▼                 ▼   │
│       │               │               │    ┌──────────┐    ┌──────────┐│
│       │               │               │    │ DECISION │◀───│ SYMBOLIC ││
│       │               │               │    │ (Action) │    │(Reasoning││
│       │               │               │    └────┬─────┘    └──────────┘│
│       │               │               │         │                      │
│       │               │               │         ▼                      │
│       │               │               │    ┌──────────┐                │
│       │               │               │    │   META   │                │
│       │               │               │    │(Learning)│                │
│       │               │               │    └────┬─────┘                │
│       │               │               │         │                      │
│       └───────────────┴───────────────┴─────────┴──────────────────────│
│                         ▲ Plasticity Feedback Loop                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Purpose | Key Technology | Brain Analog |
|-------|---------|----------------|--------------|
| **Sensory** | Event-driven input processing | DVS cameras, predictive coding | Retina, V1 |
| **SNN Core** | Spike-based feature extraction | snnTorch/SpikingJelly | Cortical columns |
| **HTM** | Online sequence learning | htm.core | Neocortical sequence memory |
| **Global Workspace** | Information integration | Liquid NNs (ncps) | Prefrontal cortex |
| **Decision** | Evidence accumulation | pymdp (active inference) | Basal ganglia |
| **Symbolic** | Verified reasoning | Logic Tensor Networks | Language areas |
| **Meta-Learning** | Plasticity modulation | MAML, eligibility traces | Neuromodulatory systems |

---

## 2. Environment Setup

### 2.1 Create Virtual Environment

```bash
# Create and activate environment
python -m venv brain_ai_env
source brain_ai_env/bin/activate  # Linux/Mac
# or: brain_ai_env\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip
```

### 2.2 Install Core Dependencies

```bash
# PyTorch (required foundation)
pip install torch torchvision torchaudio

# Spiking Neural Networks
pip install snntorch                    # Primary SNN framework
pip install spikingjelly               # Alternative with neuromorphic deployment

# Liquid Neural Networks
pip install ncps                        # Neural Circuit Policies

# Active Inference
pip install inferactively-pymdp        # Active inference for POMDPs

# Utility packages
pip install numpy scipy matplotlib
pip install tensorboard wandb          # Experiment tracking
pip install einops                     # Tensor operations
```

### 2.3 Install HTM (requires build from source)

```bash
# Option 1: Try pip (may not have latest)
pip install htm.core

# Option 2: Build from source (recommended)
git clone https://github.com/htm-community/htm.core.git
cd htm.core
pip install -r requirements.txt
python setup.py install
```

### 2.4 Verify Installation

```python
# test_installation.py
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import snntorch as snn
print(f"snnTorch: imported successfully")

from ncps.torch import CfC, LTC
print(f"ncps: imported successfully")

import pymdp
print(f"pymdp: imported successfully")

try:
    import htm
    print(f"htm.core: imported successfully")
except ImportError:
    print("htm.core: not installed (build from source if needed)")

print("\n✓ All core dependencies installed!")
```

---

## 3. Phase 1: Spiking Neural Network Core

The SNN core is the computational workhorse. It processes information using discrete spikes rather than continuous activations, enabling deployment on neuromorphic hardware at <1W power consumption.

### 3.1 Understanding the LIF Neuron

The Leaky Integrate-and-Fire (LIF) neuron is the fundamental building block. It integrates input current over time, firing a spike when membrane potential exceeds a threshold:

```
Membrane dynamics: U[t+1] = β·U[t] + W·X[t+1] - S[t]·V_thresh
Spike generation:  S[t] = 1 if U[t] > V_thresh else 0
```

Where:
- `β` (beta) = membrane potential decay rate (typically 0.8-0.95)
- `U[t]` = membrane potential at time t
- `W·X[t]` = weighted input current
- `S[t]` = output spike (binary)

### 3.2 Basic SNN with snnTorch

```python
# snn_core.py
"""
Spiking Neural Network Core Implementation
Uses surrogate gradient descent for training.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SNNCore(nn.Module):
    """
    A basic feedforward SNN with configurable layers.
    
    The network processes input over T timesteps, accumulating
    spikes to form a rate-coded output. This can be deployed
    on neuromorphic hardware (Loihi, Akida) for energy efficiency.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        beta: float = 0.9,
        num_steps: int = 25,
        spike_grad: str = "atan"
    ):
        super().__init__()
        
        self.num_steps = num_steps
        
        # Select surrogate gradient function
        # This approximates the non-differentiable spike during backprop
        if spike_grad == "atan":
            spike_fn = surrogate.atan(alpha=2.0)
        elif spike_grad == "fast_sigmoid":
            spike_fn = surrogate.fast_sigmoid(slope=25)
        elif spike_grad == "straight_through":
            spike_fn = surrogate.straight_through_estimator()
        else:
            spike_fn = surrogate.atan()
        
        # Build layers
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(sizes) - 1):
            # Linear layer (synaptic connections)
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            
            # LIF neuron layer
            # init_hidden=True allows use with nn.Sequential
            layers.append(snn.Leaky(
                beta=beta,
                spike_grad=spike_fn,
                init_hidden=True,
                output=True  # Return both spike and membrane potential
            ))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SNN.
        
        Args:
            x: Input tensor of shape (batch, features) or (batch, time, features)
               If 2D, the same input is presented for num_steps timesteps
               If 3D, each timestep uses its own input
        
        Returns:
            spike_record: Spikes over time, shape (time, batch, output)
            mem_record: Final membrane potentials, shape (batch, output)
        """
        # Handle both static and temporal inputs
        if x.dim() == 2:
            # Static input: repeat for each timestep
            x = x.unsqueeze(0).repeat(self.num_steps, 1, 1)
        else:
            # Temporal input: transpose to (time, batch, features)
            x = x.transpose(0, 1)
        
        spike_record = []
        mem_record = []
        
        # Reset hidden states
        for layer in self.network:
            if hasattr(layer, 'reset_hidden'):
                layer.reset_hidden()
        
        # Process each timestep
        for t in range(x.shape[0]):
            spk, mem = self.network(x[t])
            spike_record.append(spk)
            mem_record.append(mem)
        
        # Stack outputs
        spike_record = torch.stack(spike_record)  # (time, batch, output)
        
        return spike_record, mem_record[-1]
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions using spike rate coding.
        
        The predicted class is the neuron that fired most frequently.
        """
        spike_record, _ = self.forward(x)
        # Sum spikes over time, argmax over classes
        spike_counts = spike_record.sum(dim=0)  # (batch, output)
        return spike_counts.argmax(dim=1)


def train_snn_epoch(
    model: SNNCore,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: str = "rate"
) -> float:
    """
    Train the SNN for one epoch.
    
    Args:
        model: The SNN model
        dataloader: Training data loader
        optimizer: Optimizer (Adam recommended)
        device: CPU or CUDA device
        loss_fn: "rate" (cross-entropy on spike counts) or 
                 "membrane" (cross-entropy on final membrane potential)
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    # Use cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        targets = targets.to(device)
        
        # Flatten images if needed (for MNIST/CIFAR)
        if data.dim() > 2:
            data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        
        spike_record, mem = model(data)
        
        if loss_fn == "rate":
            # Loss on spike rate (sum over time)
            output = spike_record.sum(dim=0)
        else:
            # Loss on final membrane potential
            output = mem
        
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


# Example usage
if __name__ == "__main__":
    # Create a simple SNN for MNIST
    model = SNNCore(
        input_size=784,      # 28x28 images
        hidden_sizes=[256, 128],
        output_size=10,      # 10 digits
        beta=0.9,
        num_steps=25
    )
    
    # Example forward pass
    batch = torch.randn(32, 784)  # 32 samples
    spikes, mem = model(batch)
    
    print(f"Spike record shape: {spikes.shape}")  # (25, 32, 10)
    print(f"Final membrane shape: {mem.shape}")   # (32, 10)
    print(f"Predictions: {model.predict(batch)}")
```

### 3.3 Convolutional SNN for Vision

```python
# snn_conv.py
"""
Convolutional SNN for image classification.
Achieves ~94% on CIFAR-10 with proper training.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class ConvSNN(nn.Module):
    """
    Spiking Convolutional Neural Network.
    
    Architecture follows the pattern:
    Conv -> LIF -> Pool -> Conv -> LIF -> Pool -> FC -> LIF -> FC -> LIF
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        beta: float = 0.9,
        num_steps: int = 25
    ):
        super().__init__()
        
        self.num_steps = num_steps
        spike_grad = surrogate.atan(alpha=2.0)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        
        self.fc2 = nn.Linear(256, num_classes)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input images, shape (batch, channels, height, width)
        
        Returns:
            spike_record: (time, batch, num_classes)
            mem: Final membrane potential (batch, num_classes)
        """
        spike_record = []
        
        # Reset all LIF neurons
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()
        self.lif4.reset_mem()
        self.lif5.reset_mem()
        
        for _ in range(self.num_steps):
            # Conv block 1
            cur = self.pool1(self.conv1(x))
            spk, _ = self.lif1(cur)
            
            # Conv block 2
            cur = self.pool2(self.conv2(spk))
            spk, _ = self.lif2(cur)
            
            # Conv block 3
            cur = self.pool3(self.conv3(spk))
            spk, _ = self.lif3(cur)
            
            # Flatten
            spk = spk.view(spk.size(0), -1)
            
            # FC block 1
            cur = self.fc1(spk)
            spk, _ = self.lif4(cur)
            
            # Output layer
            cur = self.fc2(spk)
            spk, mem = self.lif5(cur)
            
            spike_record.append(spk)
        
        return torch.stack(spike_record), mem


class SpikingResidualBlock(nn.Module):
    """
    Residual block with spiking neurons.
    Implements SEW (Spike-Element-Wise) ResNet pattern.
    """
    
    def __init__(
        self,
        channels: int,
        beta: float = 0.9,
        connect_fn: str = "ADD"  # ADD or AND or IAND
    ):
        super().__init__()
        
        spike_grad = surrogate.atan()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        
        self.connect_fn = connect_fn
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out, _ = self.lif1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Spike-Element-Wise connection
        if self.connect_fn == "ADD":
            out = out + identity
        elif self.connect_fn == "AND":
            out = out * identity
        elif self.connect_fn == "IAND":
            out = out * (1 - identity)
        
        out, _ = self.lif2(out)
        return out
```

### 3.4 Training Script for MNIST

```python
# train_snn_mnist.py
"""
Complete training script for SNN on MNIST.
Expected accuracy: ~98% with proper hyperparameters.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import functional as SF
from snntorch import utils

# Import our model
from snn_core import SNNCore

def main():
    # Configuration
    config = {
        "batch_size": 128,
        "epochs": 10,
        "lr": 1e-3,
        "beta": 0.9,
        "num_steps": 25,
        "hidden_sizes": [800, 400],
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    device = torch.device(config["device"])
    print(f"Using device: {device}")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )
    
    # Model
    model = SNNCore(
        input_size=784,
        hidden_sizes=config["hidden_sizes"],
        output_size=10,
        beta=config["beta"],
        num_steps=config["num_steps"]
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        train_correct = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            spike_record, mem = model(data)
            
            # Rate-coded loss: sum spikes over time
            output = spike_record.sum(dim=0)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (output.argmax(1) == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} [{batch_idx*len(data)}/{len(train_loader.dataset)}] "
                      f"Loss: {loss.item():.4f}")
        
        train_acc = 100 * train_correct / len(train_loader.dataset)
        
        # Evaluation
        model.eval()
        test_correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(-1, 784).to(device)
                target = target.to(device)
                
                spike_record, _ = model(data)
                output = spike_record.sum(dim=0)
                test_correct += (output.argmax(1) == target).sum().item()
        
        test_acc = 100 * test_correct / len(test_loader.dataset)
        
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"Train Accuracy: {train_acc:.2f}%")
        print(f"Test Accuracy: {test_acc:.2f}%\n")
    
    # Save model
    torch.save(model.state_dict(), "snn_mnist.pth")
    print("Model saved to snn_mnist.pth")

if __name__ == "__main__":
    main()
```

---

## 4. Phase 2: Event-Driven Sensory Processing

Event-driven processing mirrors how biological sensors work: instead of sampling at fixed frame rates, they respond only to changes. This reduces data by 100-1000x while preserving microsecond temporal resolution.

### 4.1 Working with Neuromorphic Datasets

```python
# event_processing.py
"""
Event-driven sensory processing for neuromorphic data.
Supports DVS cameras and event-based datasets.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# SpikingJelly provides excellent neuromorphic dataset support
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets import pad_sequence_collate

class EventProcessor:
    """
    Processes raw events from neuromorphic sensors.
    
    Events are tuples of (timestamp, x, y, polarity) representing
    brightness changes at each pixel location.
    """
    
    def __init__(
        self,
        height: int = 128,
        width: int = 128,
        num_bins: int = 16,
        representation: str = "frame"  # "frame", "voxel", or "raw"
    ):
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.representation = representation
    
    def events_to_frames(
        self,
        events: dict,
        num_frames: int = None
    ) -> torch.Tensor:
        """
        Convert events to frame representation.
        
        Events are binned into temporal windows and accumulated
        into frames, creating a (T, 2, H, W) tensor where
        channel 0 = ON events, channel 1 = OFF events.
        
        Args:
            events: Dict with keys 't', 'x', 'y', 'p' (polarity)
            num_frames: Number of output frames (default: self.num_bins)
        
        Returns:
            frames: (T, 2, H, W) tensor
        """
        if num_frames is None:
            num_frames = self.num_bins
        
        t = events['t']
        x = events['x']
        y = events['y']
        p = events['p']
        
        # Normalize timestamps to [0, num_frames-1]
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_normalized = (t - t_min) / (t_max - t_min) * (num_frames - 1)
        else:
            t_normalized = np.zeros_like(t)
        
        bin_indices = t_normalized.astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, num_frames - 1)
        
        # Accumulate events into frames
        frames = np.zeros((num_frames, 2, self.height, self.width), dtype=np.float32)
        
        for i in range(len(t)):
            bin_idx = bin_indices[i]
            pol = int(p[i])  # 0 or 1
            frames[bin_idx, pol, y[i], x[i]] += 1
        
        return torch.from_numpy(frames)
    
    def events_to_voxel_grid(
        self,
        events: dict,
        num_bins: int = None
    ) -> torch.Tensor:
        """
        Convert events to voxel grid representation.
        
        Similar to frames but with temporal interpolation
        for smoother representation.
        """
        if num_bins is None:
            num_bins = self.num_bins
        
        t = events['t'].astype(np.float32)
        x = events['x']
        y = events['y']
        p = events['p'].astype(np.float32) * 2 - 1  # Convert to -1, +1
        
        # Normalize timestamps
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_normalized = (t - t_min) / (t_max - t_min) * (num_bins - 1)
        else:
            t_normalized = np.zeros_like(t)
        
        voxel = np.zeros((num_bins, self.height, self.width), dtype=np.float32)
        
        # Bilinear interpolation in time
        for i in range(len(t)):
            t_idx = t_normalized[i]
            t_floor = int(np.floor(t_idx))
            t_ceil = min(t_floor + 1, num_bins - 1)
            
            dt = t_idx - t_floor
            
            voxel[t_floor, y[i], x[i]] += p[i] * (1 - dt)
            voxel[t_ceil, y[i], x[i]] += p[i] * dt
        
        return torch.from_numpy(voxel)


def load_dvs_gesture_dataset(
    root_dir: str,
    num_frames: int = 16,
    train: bool = True
) -> Dataset:
    """
    Load DVS128 Gesture dataset with frame representation.
    
    Args:
        root_dir: Path to dataset
        num_frames: Number of temporal bins
        train: Training or test split
    
    Returns:
        Dataset yielding (frames, label) pairs
    """
    dataset = DVS128Gesture(
        root=root_dir,
        train=train,
        data_type='frame',
        frames_number=num_frames,
        split_by='number'
    )
    return dataset


# Example: Custom dataset for raw events
class EventDataset(Dataset):
    """
    Generic dataset for event data stored as numpy files.
    """
    
    def __init__(
        self,
        event_files: list[str],
        labels: list[int],
        processor: EventProcessor
    ):
        self.event_files = event_files
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.event_files)
    
    def __getitem__(self, idx):
        # Load events from file
        events = np.load(self.event_files[idx], allow_pickle=True).item()
        
        # Convert to frames
        frames = self.processor.events_to_frames(events)
        label = self.labels[idx]
        
        return frames, label


# Predictive coding layer
class PredictiveCodingLayer(torch.nn.Module):
    """
    Implements predictive coding: only transmit prediction errors.
    
    This layer maintains a prediction of the next input and
    outputs only the difference (prediction error). This mirrors
    how biological sensory systems process information.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        precision_weighted: bool = True
    ):
        super().__init__()
        
        # Prediction network
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )
        
        # Input encoder
        self.encoder = torch.nn.Linear(input_dim, hidden_dim)
        
        # Precision (inverse variance) for weighting errors
        if precision_weighted:
            self.log_precision = torch.nn.Parameter(torch.zeros(input_dim))
        else:
            self.register_buffer('log_precision', torch.zeros(input_dim))
        
        self.hidden_state = None
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, input_dim)
        
        Returns:
            prediction_error: Precision-weighted error
            hidden: Updated hidden state
        """
        batch_size = x.shape[0]
        
        # Initialize hidden state if needed
        if self.hidden_state is None or self.hidden_state.shape[0] != batch_size:
            self.hidden_state = torch.zeros(
                batch_size, self.encoder.out_features, device=x.device
            )
        
        # Generate prediction from previous hidden state
        prediction = self.predictor(self.hidden_state)
        
        # Compute prediction error
        error = x - prediction
        
        # Weight by precision (attention mechanism)
        precision = torch.exp(self.log_precision)
        weighted_error = error * precision
        
        # Update hidden state
        self.hidden_state = self.encoder(x)
        
        return weighted_error, self.hidden_state
    
    def reset(self):
        """Reset hidden state for new sequence."""
        self.hidden_state = None
```

---

## 5. Phase 3: Hierarchical Temporal Memory

HTM learns sequences online without supervision and naturally detects anomalies when predictions fail. It uses Sparse Distributed Representations (SDRs) that provide massive capacity and noise tolerance.

### 5.1 HTM Implementation with htm.core

```python
# htm_layer.py
"""
Hierarchical Temporal Memory for sequence learning.
Uses htm.core library (community fork of Numenta's NuPIC).
"""

try:
    from htm.bindings.sdr import SDR
    from htm.bindings.algorithms import SpatialPooler, TemporalMemory
    from htm.bindings.encoders import ScalarEncoder
    HTM_AVAILABLE = True
except ImportError:
    HTM_AVAILABLE = False
    print("Warning: htm.core not installed. HTM features will be unavailable.")

import numpy as np

class HTMLayer:
    """
    Hierarchical Temporal Memory layer for sequence learning.
    
    Key properties:
    - Online unsupervised learning (no batches needed)
    - No catastrophic forgetting
    - Anomaly detection via prediction failure
    - High-order sequence prediction
    """
    
    def __init__(
        self,
        input_size: int,
        column_count: int = 2048,
        cells_per_column: int = 32,
        sparsity: float = 0.02,
        permanence_inc: float = 0.1,
        permanence_dec: float = 0.1
    ):
        if not HTM_AVAILABLE:
            raise ImportError("htm.core is required for HTMLayer")
        
        self.input_size = input_size
        self.column_count = column_count
        self.cells_per_column = cells_per_column
        
        # Calculate active columns based on sparsity
        self.num_active_columns = int(column_count * sparsity)
        
        # Input SDR
        self.input_sdr = SDR(input_size)
        
        # Spatial Pooler: converts input to sparse representation
        self.sp = SpatialPooler(
            inputDimensions=[input_size],
            columnDimensions=[column_count],
            potentialRadius=input_size,
            potentialPct=0.85,
            globalInhibition=True,
            localAreaDensity=sparsity,
            synPermInactiveDec=permanence_dec,
            synPermActiveInc=permanence_inc,
            synPermConnected=0.1,
            boostStrength=3.0,
            seed=42
        )
        
        # Active columns SDR
        self.active_columns = SDR(column_count)
        
        # Temporal Memory: learns sequences
        self.tm = TemporalMemory(
            columnDimensions=[column_count],
            cellsPerColumn=cells_per_column,
            activationThreshold=13,
            initialPermanence=0.21,
            connectedPermanence=0.5,
            minThreshold=10,
            maxNewSynapseCount=20,
            permanenceIncrement=permanence_inc,
            permanenceDecrement=permanence_dec,
            predictedSegmentDecrement=0.0,
            maxSegmentsPerCell=255,
            maxSynapsesPerSegment=255,
            seed=42
        )
        
        # Track anomaly history for smoothing
        self.anomaly_history = []
        self.anomaly_window = 1000
    
    def encode_input(self, values: np.ndarray) -> SDR:
        """
        Encode numeric input into SDR.
        
        For production use, consider using htm.encoders for
        proper semantic encoding of different data types.
        """
        # Simple binary encoding
        # More sophisticated encoders available in htm.encoders
        active_bits = np.where(values > 0.5)[0]
        self.input_sdr.sparse = active_bits.tolist()
        return self.input_sdr
    
    def process(
        self,
        input_sdr: SDR,
        learn: bool = True
    ) -> dict:
        """
        Process one timestep through the HTM.
        
        Args:
            input_sdr: Sparse distributed representation of input
            learn: Whether to learn from this input
        
        Returns:
            dict with keys:
                - 'active_cells': Currently active cells
                - 'predicted_cells': Cells predicted for next step
                - 'anomaly': Anomaly score (0-1)
                - 'anomaly_likelihood': Smoothed anomaly probability
        """
        # Spatial Pooler: input -> active columns
        self.sp.compute(input_sdr, learn, self.active_columns)
        
        # Temporal Memory: active columns -> active/predicted cells
        self.tm.compute(self.active_columns, learn)
        
        # Get anomaly score
        # High anomaly = current input was not predicted
        anomaly = self.tm.anomaly
        
        # Compute anomaly likelihood (smoothed probability)
        self.anomaly_history.append(anomaly)
        if len(self.anomaly_history) > self.anomaly_window:
            self.anomaly_history.pop(0)
        
        mean_anomaly = np.mean(self.anomaly_history)
        std_anomaly = np.std(self.anomaly_history) + 1e-6
        anomaly_likelihood = 1 - np.exp(-(anomaly - mean_anomaly) / std_anomaly)
        anomaly_likelihood = max(0, min(1, anomaly_likelihood))
        
        return {
            'active_cells': self.tm.getActiveCells().sparse.copy(),
            'predicted_cells': self.tm.getPredictiveCells().sparse.copy(),
            'anomaly': anomaly,
            'anomaly_likelihood': anomaly_likelihood
        }
    
    def predict_next(self, steps: int = 1) -> np.ndarray:
        """
        Get predictions for next N steps.
        
        Returns predicted column indices.
        """
        predicted = self.tm.getPredictiveCells()
        # Convert cell indices to column indices
        predicted_columns = np.unique(predicted.sparse // self.cells_per_column)
        return predicted_columns
    
    def reset(self):
        """Reset temporal memory state for new sequence."""
        self.tm.reset()
        self.anomaly_history = []


class HTMPredictor:
    """
    Complete HTM system with encoder and predictor.
    """
    
    def __init__(
        self,
        input_dim: int,
        min_val: float = 0.0,
        max_val: float = 1.0,
        resolution: float = 0.01
    ):
        if not HTM_AVAILABLE:
            raise ImportError("htm.core is required")
        
        # Calculate SDR dimensions
        n_buckets = int((max_val - min_val) / resolution)
        n = n_buckets * 21  # Each bucket activates ~21 bits
        w = 21  # Active bits per encoding
        
        self.encoder = ScalarEncoder(
            n=n,
            w=w,
            minVal=min_val,
            maxVal=max_val,
            clipInput=True
        )
        
        self.htm = HTMLayer(
            input_size=n,
            column_count=2048,
            cells_per_column=32
        )
        
        # For prediction decoding
        self.predictor_resolution = resolution
        self.prediction_buffer = {}
    
    def learn(self, value: float) -> float:
        """
        Learn from a single value and return anomaly score.
        """
        # Encode
        encoding = SDR(self.encoder.n)
        self.encoder.encode(value, encoding)
        
        # Process
        result = self.htm.process(encoding, learn=True)
        
        return result['anomaly']
    
    def predict(self) -> float:
        """
        Get prediction for next value.
        """
        predicted_columns = self.htm.predict_next()
        
        if len(predicted_columns) == 0:
            return None
        
        # Simple decoding: average of predicted column indices
        # (More sophisticated decoding would use learned classifier)
        avg_column = np.mean(predicted_columns)
        predicted_value = avg_column / self.htm.column_count
        
        return predicted_value


# Example usage
def demo_htm_anomaly_detection():
    """
    Demo: HTM for anomaly detection in time series.
    """
    if not HTM_AVAILABLE:
        print("htm.core not available, skipping demo")
        return
    
    # Create predictor
    predictor = HTMPredictor(
        input_dim=100,
        min_val=0.0,
        max_val=100.0,
        resolution=0.5
    )
    
    # Generate synthetic data with anomaly
    np.random.seed(42)
    normal_data = 50 + 10 * np.sin(np.linspace(0, 4*np.pi, 200))
    normal_data += np.random.randn(200) * 2
    
    # Insert anomaly
    anomaly_idx = 150
    normal_data[anomaly_idx:anomaly_idx+5] = 90  # Sudden spike
    
    # Process through HTM
    anomaly_scores = []
    for value in normal_data:
        score = predictor.learn(value)
        anomaly_scores.append(score)
    
    # Find detected anomalies
    threshold = np.percentile(anomaly_scores, 95)
    detected = np.where(np.array(anomaly_scores) > threshold)[0]
    
    print(f"Anomaly inserted at index: {anomaly_idx}")
    print(f"Anomalies detected at indices: {detected}")
    print(f"Detection threshold: {threshold:.3f}")


if __name__ == "__main__":
    demo_htm_anomaly_detection()
```

---

## 6. Phase 4: Global Workspace & Working Memory

The Global Workspace implements "ignition" from Global Workspace Theory: information from specialized modules competes for access to a capacity-limited workspace, then broadcasts globally. Liquid Neural Networks provide the temporal dynamics.

### 6.1 Liquid Neural Networks with ncps

```python
# global_workspace.py
"""
Global Workspace and Working Memory Implementation.

Uses Liquid Time-Constant Networks (LTCs) and Neural Circuit Policies (NCPs)
for biologically-inspired temporal processing with compact, interpretable models.
"""

import torch
import torch.nn as nn
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP

class LiquidWorkingMemory(nn.Module):
    """
    Working memory using Liquid Neural Networks.
    
    LTCs use input-dependent time constants, meaning the network's
    temporal dynamics adapt to the input. This enables:
    - Compact models (19 neurons matched human driving performance)
    - Interpretable dynamics
    - Robust temporal processing
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        output_size: int = None,
        mode: str = "cfc"  # "cfc" (faster) or "ltc" (more expressive)
    ):
        super().__init__()
        
        if output_size is None:
            output_size = hidden_size
        
        # Select model type
        if mode == "cfc":
            # Closed-form Continuous-time (faster, no ODE solver)
            self.rnn = CfC(input_size, hidden_size, batch_first=True)
        elif mode == "ltc":
            # Liquid Time-Constant (original, uses ODE solver)
            self.rnn = LTC(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.output_proj = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
    
    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence (batch, time, features)
            h0: Initial hidden state (batch, hidden_size)
        
        Returns:
            output: (batch, time, output_size)
            hn: Final hidden state (batch, hidden_size)
        """
        if h0 is None:
            h0 = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        
        # Process through liquid network
        output, hn = self.rnn(x, h0)
        
        # Project to output dimension
        output = self.output_proj(output)
        
        return output, hn


class NeuralCircuitPolicy(nn.Module):
    """
    Neural Circuit Policy for hierarchical processing.
    
    NCPs are inspired by C. elegans nervous system structure:
    - Sensory neurons receive input
    - Inter neurons process internally
    - Command neurons integrate information
    - Motor neurons produce output
    
    The sparse, structured connectivity enables interpretable computation.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        inter_neurons: int = 12,
        command_neurons: int = 8,
        motor_neurons: int = None,
        sensory_fanout: int = 4,
        inter_fanout: int = 4,
        recurrent_command: int = 4,
        motor_fanin: int = 4
    ):
        super().__init__()
        
        if motor_neurons is None:
            motor_neurons = output_size
        
        # Create NCP wiring
        wiring = NCP(
            inter_neurons=inter_neurons,
            command_neurons=command_neurons,
            motor_neurons=motor_neurons,
            sensory_fanout=sensory_fanout,
            inter_fanout=inter_fanout,
            recurrent_command_synapses=recurrent_command,
            motor_fanin=motor_fanin
        )
        
        # Create LTC with NCP wiring
        self.rnn = LTC(input_size, wiring, batch_first=True)
        
        # Output projection if needed
        if motor_neurons != output_size:
            self.output_proj = nn.Linear(motor_neurons, output_size)
        else:
            self.output_proj = nn.Identity()
        
        self.hidden_size = wiring.units
    
    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence (batch, time, features)
            h0: Initial hidden state
        
        Returns:
            output: (batch, time, output_size)
            hn: Final hidden state
        """
        if h0 is None:
            h0 = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        
        output, hn = self.rnn(x, h0)
        output = self.output_proj(output)
        
        return output, hn


class GlobalWorkspace(nn.Module):
    """
    Global Workspace for information integration.
    
    Implements the "ignition" pattern from Global Workspace Theory:
    1. Specialist modules process different aspects of input
    2. Outputs compete for access to workspace (via attention)
    3. Winner broadcasts globally to all modules
    
    This enables coordinated action across specialized systems.
    """
    
    def __init__(
        self,
        input_dims: dict[str, int],  # Named inputs from different modules
        workspace_size: int = 256,
        num_heads: int = 8,
        capacity_limit: int = 7  # Miller's Law: 7±2 items
    ):
        super().__init__()
        
        self.input_dims = input_dims
        self.workspace_size = workspace_size
        self.capacity_limit = capacity_limit
        
        # Project each input to workspace dimension
        self.input_projections = nn.ModuleDict({
            name: nn.Linear(dim, workspace_size)
            for name, dim in input_dims.items()
        })
        
        # Competition mechanism (cross-attention)
        self.competition = nn.MultiheadAttention(
            embed_dim=workspace_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Workspace state (using Liquid NN for temporal dynamics)
        self.workspace_memory = LiquidWorkingMemory(
            input_size=workspace_size,
            hidden_size=workspace_size,
            output_size=workspace_size,
            mode="cfc"
        )
        
        # Broadcast projections (back to each module)
        self.broadcast_projections = nn.ModuleDict({
            name: nn.Linear(workspace_size, dim)
            for name, dim in input_dims.items()
        })
        
        # Capacity gate
        self.capacity_gate = nn.Linear(workspace_size, 1)
    
    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        workspace_state: torch.Tensor = None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: Dict of named inputs from different modules
                   Each tensor: (batch, features) or (batch, time, features)
            workspace_state: Previous workspace hidden state
        
        Returns:
            broadcasts: Dict of signals broadcast back to each module
            attention_weights: Which inputs "won" the competition
            new_workspace_state: Updated workspace state
        """
        batch_size = list(inputs.values())[0].size(0)
        
        # Project all inputs to workspace dimension
        projected = []
        for name, x in inputs.items():
            # Handle both 2D and 3D inputs
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add time dimension
            proj = self.input_projections[name](x)  # (batch, time, workspace)
            projected.append(proj)
        
        # Stack for attention: (batch, num_inputs * time, workspace)
        stacked = torch.cat(projected, dim=1)
        
        # Competition via self-attention
        # Query = current workspace state, Keys/Values = all inputs
        if workspace_state is None:
            query = stacked.mean(dim=1, keepdim=True)  # (batch, 1, workspace)
        else:
            query = workspace_state.unsqueeze(1)
        
        # Apply attention with capacity constraint
        attended, attention_weights = self.competition(
            query, stacked, stacked
        )  # attended: (batch, 1, workspace)
        
        # Apply capacity gate (soft top-k selection)
        gate_scores = self.capacity_gate(stacked).squeeze(-1)  # (batch, num_inputs*time)
        gate_probs = torch.softmax(gate_scores, dim=-1)
        
        # Select top-k based on capacity
        topk_probs, topk_idx = torch.topk(gate_probs, self.capacity_limit, dim=-1)
        
        # Masked attention
        mask = torch.zeros_like(gate_probs)
        mask.scatter_(1, topk_idx, 1.0)
        
        # Update workspace state through liquid memory
        workspace_input = attended.squeeze(1).unsqueeze(1)  # (batch, 1, workspace)
        _, new_workspace_state = self.workspace_memory(
            workspace_input, workspace_state
        )
        
        # Broadcast to all modules
        broadcasts = {}
        for name in inputs.keys():
            broadcasts[name] = self.broadcast_projections[name](new_workspace_state)
        
        return broadcasts, attention_weights.squeeze(1), new_workspace_state


# Example usage
def demo_global_workspace():
    """
    Demo: Global Workspace integrating multiple modules.
    """
    # Create workspace accepting inputs from 3 modules
    workspace = GlobalWorkspace(
        input_dims={
            "visual": 512,
            "auditory": 256,
            "proprioceptive": 128
        },
        workspace_size=256,
        num_heads=8
    )
    
    # Simulated inputs from different modules
    batch_size = 4
    inputs = {
        "visual": torch.randn(batch_size, 512),
        "auditory": torch.randn(batch_size, 256),
        "proprioceptive": torch.randn(batch_size, 128)
    }
    
    # Process through workspace
    broadcasts, attention, state = workspace(inputs)
    
    print("Global Workspace Demo:")
    print(f"  Input modules: {list(inputs.keys())}")
    print(f"  Attention shape: {attention.shape}")
    print(f"  Workspace state shape: {state.shape}")
    for name, broadcast in broadcasts.items():
        print(f"  Broadcast to {name}: {broadcast.shape}")


if __name__ == "__main__":
    demo_global_workspace()
```

---

## 7. Phase 5: Decision & Action System

Decisions emerge from evidence accumulation until a threshold is reached. Active Inference provides a principled framework that naturally balances exploitation (do what works) and exploration (reduce uncertainty).

### 7.1 Active Inference with pymdp

```python
# decision_system.py
"""
Decision and Action System using Active Inference.

Active Inference casts perception, learning, and action as Bayesian inference
under a single objective: minimize expected free energy. This naturally
balances exploitation (maximize expected reward) and exploration (reduce uncertainty).
"""

import numpy as np
from typing import Optional
import pymdp
from pymdp.agent import Agent
from pymdp import utils

class ActiveInferenceAgent:
    """
    Active Inference agent for decision-making under uncertainty.
    
    The agent maintains beliefs about hidden states and selects actions
    to minimize expected free energy, which combines:
    - Pragmatic value (achieving goals/preferences)
    - Epistemic value (reducing uncertainty)
    
    This provides a principled solution to the exploration-exploitation tradeoff.
    """
    
    def __init__(
        self,
        num_states: list[int],      # Number of states per factor
        num_observations: list[int], # Number of observations per modality
        num_actions: list[int],      # Number of actions per control factor
        A: Optional[list] = None,    # Observation model P(o|s)
        B: Optional[list] = None,    # Transition model P(s'|s,a)
        C: Optional[list] = None,    # Preference model (log preferences over obs)
        D: Optional[list] = None,    # Prior over initial states
        planning_horizon: int = 3,
        inference_algo: str = "MMP"  # Marginal Message Passing
    ):
        """
        Initialize the Active Inference agent.
        
        Args:
            num_states: List of state dimensionalities for each hidden state factor
            num_observations: List of observation dimensionalities for each modality
            num_actions: List of action counts for each control factor
            A: Observation likelihood matrices (one per modality)
            B: Transition matrices (one per controllable state factor)
            C: Log preferences over observations (agent's "goals")
            D: Prior beliefs about initial states
            planning_horizon: How many steps ahead to plan
            inference_algo: Inference algorithm ("MMP" or "VANILLA")
        """
        self.num_states = num_states
        self.num_observations = num_observations
        self.num_actions = num_actions
        
        # Initialize generative model components
        if A is None:
            # Default: uniform observation model
            A = self._initialize_A(num_observations, num_states)
        
        if B is None:
            # Default: identity transitions
            B = self._initialize_B(num_states, num_actions)
        
        if C is None:
            # Default: uniform preferences (no goals)
            C = self._initialize_C(num_observations)
        
        if D is None:
            # Default: uniform prior over initial states
            D = self._initialize_D(num_states)
        
        # Create pymdp agent
        self.agent = Agent(
            A=A, B=B, C=C, D=D,
            policy_len=planning_horizon,
            inference_algo=inference_algo,
            use_states_info_gain=True,  # Enable epistemic value
            action_selection="stochastic"
        )
        
        # Track history
        self.observation_history = []
        self.action_history = []
        self.belief_history = []
    
    def _initialize_A(self, num_obs, num_states):
        """Initialize observation model as random but normalized."""
        A = []
        for no in num_obs:
            # Create observation matrix for this modality
            # Shape: (num_obs, *num_states)
            a_shape = [no] + num_states
            a = np.random.rand(*a_shape) + 0.1
            # Normalize so columns sum to 1
            a = a / a.sum(axis=0, keepdims=True)
            A.append(a)
        return A
    
    def _initialize_B(self, num_states, num_actions):
        """Initialize transition model as identity with noise."""
        B = []
        for i, ns in enumerate(num_states):
            na = num_actions[i] if i < len(num_actions) else 1
            # Shape: (ns, ns, na) - next_state x current_state x action
            b = np.zeros((ns, ns, na))
            for a in range(na):
                # Mostly identity with some transition probability
                b[:, :, a] = np.eye(ns) * 0.9 + 0.1 / ns
            B.append(b)
        return B
    
    def _initialize_C(self, num_obs):
        """Initialize preferences as uniform (no goals)."""
        C = []
        for no in num_obs:
            c = np.zeros(no)
            C.append(c)
        return C
    
    def _initialize_D(self, num_states):
        """Initialize prior as uniform over states."""
        D = []
        for ns in num_states:
            d = np.ones(ns) / ns
            D.append(d)
        return D
    
    def set_preferences(
        self,
        modality_idx: int,
        preferred_obs: int,
        preference_strength: float = 2.0
    ):
        """
        Set preferences for a specific observation.
        
        This defines the agent's "goals" - what observations it prefers to receive.
        The agent will take actions that it believes will lead to preferred observations.
        
        Args:
            modality_idx: Which observation modality
            preferred_obs: Which observation is preferred
            preference_strength: How strongly to prefer (log scale)
        """
        C = self.agent.C.copy()
        C[modality_idx] = np.zeros(self.num_observations[modality_idx])
        C[modality_idx][preferred_obs] = preference_strength
        self.agent.C = C
    
    def observe(self, observation: list[int]) -> np.ndarray:
        """
        Process an observation and update beliefs.
        
        Args:
            observation: List of observation indices (one per modality)
        
        Returns:
            Updated beliefs about hidden states
        """
        self.observation_history.append(observation)
        
        # Convert to one-hot
        obs_vec = [utils.onehot(o, self.num_observations[i]) 
                   for i, o in enumerate(observation)]
        
        # Infer hidden states
        qs = self.agent.infer_states(obs_vec)
        
        self.belief_history.append(qs)
        return qs
    
    def decide(self) -> list[int]:
        """
        Select an action using active inference.
        
        The agent evaluates policies (action sequences) based on
        expected free energy, which balances:
        - Pragmatic value: Will this achieve my goals?
        - Epistemic value: Will this reduce my uncertainty?
        
        Returns:
            List of action indices (one per control factor)
        """
        # Infer policies and their expected free energies
        q_pi, efe = self.agent.infer_policies()
        
        # Sample action from policy posterior
        action = self.agent.sample_action()
        
        self.action_history.append(action)
        return list(action)
    
    def get_expected_free_energy(self) -> np.ndarray:
        """
        Get expected free energy for each policy.
        
        Lower EFE = better policy (more likely to achieve goals + reduce uncertainty)
        """
        _, efe = self.agent.infer_policies()
        return efe
    
    def get_epistemic_value(self) -> float:
        """
        Get the epistemic (information-seeking) component of the last decision.
        """
        return self.agent.states_info_gain.sum() if hasattr(self.agent, 'states_info_gain') else 0.0


class DriftDiffusionModel:
    """
    Drift-Diffusion Model for evidence accumulation decisions.
    
    Decisions emerge from noisy evidence accumulation:
    dx = v*dt + σ*dW
    
    Where:
    - v = drift rate (quality of evidence)
    - σ = diffusion coefficient (noise)
    - Decision when x crosses upper (+a) or lower (-a) boundary
    
    This matches human decision-making behavior remarkably well.
    """
    
    def __init__(
        self,
        drift_rate: float = 0.3,
        threshold: float = 1.0,
        noise: float = 0.5,
        dt: float = 0.01,
        bias: float = 0.0
    ):
        """
        Args:
            drift_rate: Rate of evidence accumulation (positive = toward upper boundary)
            threshold: Decision boundary (symmetric: ±threshold)
            noise: Diffusion coefficient (decision noise)
            dt: Time step for simulation
            bias: Starting point bias (-threshold to +threshold)
        """
        self.drift_rate = drift_rate
        self.threshold = threshold
        self.noise = noise
        self.dt = dt
        self.bias = bias
    
    def simulate_trial(
        self,
        max_time: float = 10.0,
        return_trajectory: bool = False
    ) -> dict:
        """
        Simulate a single decision trial.
        
        Returns:
            dict with:
                - 'choice': 1 (upper) or 0 (lower)
                - 'rt': Response time
                - 'trajectory': Evidence trajectory (if requested)
        """
        evidence = self.bias
        time = 0.0
        trajectory = [evidence] if return_trajectory else None
        
        while time < max_time:
            # Accumulate evidence
            evidence += self.drift_rate * self.dt
            evidence += self.noise * np.sqrt(self.dt) * np.random.randn()
            
            time += self.dt
            
            if return_trajectory:
                trajectory.append(evidence)
            
            # Check boundaries
            if evidence >= self.threshold:
                return {
                    'choice': 1,
                    'rt': time,
                    'trajectory': trajectory
                }
            elif evidence <= -self.threshold:
                return {
                    'choice': 0,
                    'rt': time,
                    'trajectory': trajectory
                }
        
        # Timeout: choose based on current evidence
        return {
            'choice': 1 if evidence > 0 else 0,
            'rt': max_time,
            'trajectory': trajectory
        }
    
    def set_threshold(self, threshold: float):
        """
        Adjust decision threshold.
        
        Higher threshold = slower, more accurate decisions
        Lower threshold = faster, less accurate decisions
        
        This implements the speed-accuracy tradeoff.
        """
        self.threshold = threshold


# Example usage
def demo_active_inference():
    """
    Demo: Active Inference agent navigating a simple environment.
    """
    # Simple grid world: 3x3 grid with goal in corner
    # States: 9 positions
    # Observations: Position (9 possibilities)
    # Actions: Stay, Up, Down, Left, Right (5 actions)
    
    agent = ActiveInferenceAgent(
        num_states=[9],      # 9 grid positions
        num_observations=[9], # Observe current position
        num_actions=[5],      # 5 movement actions
        planning_horizon=3
    )
    
    # Set goal: prefer position 8 (bottom-right corner)
    agent.set_preferences(
        modality_idx=0,
        preferred_obs=8,
        preference_strength=3.0
    )
    
    # Simulate agent navigating
    current_pos = 0  # Start at top-left
    
    print("Active Inference Navigation Demo:")
    print(f"  Start position: {current_pos}")
    print(f"  Goal position: 8")
    
    for step in range(10):
        # Observe current position
        beliefs = agent.observe([current_pos])
        
        # Decide next action
        action = agent.decide()
        
        # Apply action (simplified: just move toward goal)
        if action[0] == 1:  # Up
            current_pos = max(0, current_pos - 3)
        elif action[0] == 2:  # Down
            current_pos = min(8, current_pos + 3)
        elif action[0] == 3:  # Left
            current_pos = max(0, current_pos - 1)
        elif action[0] == 4:  # Right
            current_pos = min(8, current_pos + 1)
        
        print(f"  Step {step+1}: Action={action[0]}, Position={current_pos}")
        
        if current_pos == 8:
            print("  Goal reached!")
            break


if __name__ == "__main__":
    demo_active_inference()
```

---

## 8. Phase 6: Neuro-Symbolic Reasoning

Pure neural networks struggle with multi-step logical reasoning. This layer adds explicit symbolic structure for verified reasoning without hallucinations.

### 8.1 Logic Tensor Networks Concept

```python
# neuro_symbolic.py
"""
Neuro-Symbolic Reasoning Layer.

Combines neural pattern recognition with symbolic logical reasoning.
Key approaches:
- Logic Tensor Networks: Differentiable first-order logic
- Graph Neural Networks: Relational reasoning
- Neural-guided symbolic search
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

class SymbolicReasoner(nn.Module):
    """
    Simple neuro-symbolic reasoning module.
    
    Implements differentiable logic operations:
    - Fuzzy AND: min(a, b) or a * b
    - Fuzzy OR: max(a, b) or a + b - a*b
    - Fuzzy NOT: 1 - a
    - Fuzzy IMPLIES: min(1, 1 - a + b)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_predicates: int,
        num_constants: int,
        logic_type: str = "product"  # "product" or "godel"
    ):
        """
        Args:
            embedding_dim: Dimension of entity/predicate embeddings
            num_predicates: Number of learnable predicates
            num_constants: Number of entities/constants
            logic_type: Type of fuzzy logic ("product" = probabilistic, "godel" = min/max)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.logic_type = logic_type
        
        # Learnable embeddings
        self.constant_embeddings = nn.Embedding(num_constants, embedding_dim)
        self.predicate_embeddings = nn.Embedding(num_predicates, embedding_dim)
        
        # Grounding functions: map embeddings to truth values
        self.unary_grounding = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        self.binary_grounding = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def ground_predicate(
        self,
        predicate_idx: int,
        constant_indices: list[int]
    ) -> torch.Tensor:
        """
        Compute truth value of predicate(constants).
        
        Args:
            predicate_idx: Index of predicate
            constant_indices: Indices of constant arguments
        
        Returns:
            Truth value in [0, 1]
        """
        pred_emb = self.predicate_embeddings(
            torch.tensor([predicate_idx])
        )
        
        const_embs = self.constant_embeddings(
            torch.tensor(constant_indices)
        )
        
        if len(constant_indices) == 1:
            # Unary predicate
            combined = torch.cat([pred_emb, const_embs], dim=-1)
            truth = self.unary_grounding(combined)
        else:
            # Binary predicate
            combined = torch.cat([pred_emb, const_embs.flatten().unsqueeze(0)], dim=-1)
            truth = self.binary_grounding(combined)
        
        return truth.squeeze()
    
    def fuzzy_and(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuzzy conjunction."""
        if self.logic_type == "product":
            return a * b
        else:  # godel
            return torch.min(a, b)
    
    def fuzzy_or(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuzzy disjunction."""
        if self.logic_type == "product":
            return a + b - a * b
        else:  # godel
            return torch.max(a, b)
    
    def fuzzy_not(self, a: torch.Tensor) -> torch.Tensor:
        """Fuzzy negation."""
        return 1 - a
    
    def fuzzy_implies(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuzzy implication: a → b."""
        if self.logic_type == "product":
            # Reichenbach implication
            return 1 - a + a * b
        else:  # godel
            # Gödel implication
            return torch.where(a <= b, torch.ones_like(a), b)
    
    def forall(self, values: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Universal quantifier (∀): conjunction over domain."""
        if self.logic_type == "product":
            return values.prod(dim=dim)
        else:
            return values.min(dim=dim).values
    
    def exists(self, values: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Existential quantifier (∃): disjunction over domain."""
        if self.logic_type == "product":
            # Using log-sum-exp for numerical stability
            return 1 - (1 - values).prod(dim=dim)
        else:
            return values.max(dim=dim).values


class RuleInductionModule(nn.Module):
    """
    Learn logical rules from data using neural networks.
    
    Instead of hand-coding rules, this module learns rule weights
    that can be interpreted symbolically.
    """
    
    def __init__(
        self,
        num_predicates: int,
        max_rule_length: int = 3,
        num_rules: int = 100
    ):
        super().__init__()
        
        self.num_predicates = num_predicates
        self.max_rule_length = max_rule_length
        self.num_rules = num_rules
        
        # Rule templates: which predicates appear in rule body
        # Shape: (num_rules, max_rule_length, num_predicates)
        self.rule_body_weights = nn.Parameter(
            torch.randn(num_rules, max_rule_length, num_predicates)
        )
        
        # Rule head: which predicate is the conclusion
        # Shape: (num_rules, num_predicates)
        self.rule_head_weights = nn.Parameter(
            torch.randn(num_rules, num_predicates)
        )
        
        # Rule confidence/strength
        self.rule_confidence = nn.Parameter(torch.zeros(num_rules))
    
    def get_rules(self, threshold: float = 0.5) -> list[dict]:
        """
        Extract interpretable rules above confidence threshold.
        
        Returns list of dicts with:
            - 'head': Index of conclusion predicate
            - 'body': List of premise predicate indices
            - 'confidence': Rule strength
        """
        rules = []
        
        confidences = torch.sigmoid(self.rule_confidence)
        head_probs = F.softmax(self.rule_head_weights, dim=-1)
        body_probs = F.softmax(self.rule_body_weights, dim=-1)
        
        for r in range(self.num_rules):
            if confidences[r] > threshold:
                head = head_probs[r].argmax().item()
                body = []
                for pos in range(self.max_rule_length):
                    if body_probs[r, pos].max() > 0.5:
                        body.append(body_probs[r, pos].argmax().item())
                
                rules.append({
                    'head': head,
                    'body': body,
                    'confidence': confidences[r].item()
                })
        
        return rules


class System2Reasoner(nn.Module):
    """
    System 2 (slow, deliberate) reasoning module.
    
    Implements multi-step reasoning through iterative refinement:
    1. Initial fast inference (System 1)
    2. Check for inconsistencies or low confidence
    3. If needed, engage deliberate reasoning (System 2)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_reasoning_steps: int = 5,
        deliberation_threshold: float = 0.7
    ):
        super().__init__()
        
        self.num_steps = num_reasoning_steps
        self.threshold = deliberation_threshold
        
        # System 1: Fast pattern matching
        self.system1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Confidence estimator
        self.confidence = nn.Linear(hidden_dim, 1)
        
        # System 2: Iterative refinement
        self.reasoning_step = nn.GRUCell(hidden_dim, hidden_dim)
        self.step_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        
        # Output projection
        self.output = nn.Linear(hidden_dim, input_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        force_deliberate: bool = False
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: Input tensor (batch, input_dim)
            force_deliberate: Force System 2 reasoning regardless of confidence
        
        Returns:
            output: Reasoned output
            info: Dict with confidence, steps_used, etc.
        """
        batch_size = x.shape[0]
        
        # System 1: Fast inference
        h = self.system1(x)
        confidence = torch.sigmoid(self.confidence(h))
        
        steps_used = torch.ones(batch_size, device=x.device)
        
        # Check if deliberation needed
        needs_deliberation = (confidence.squeeze() < self.threshold) | force_deliberate
        
        if needs_deliberation.any():
            # System 2: Deliberate reasoning for uncertain cases
            deliberate_mask = needs_deliberation.unsqueeze(-1)
            
            # Store reasoning trajectory
            trajectory = [h]
            
            for step in range(self.num_steps):
                # Self-attention over reasoning trajectory
                trajectory_tensor = torch.stack(trajectory, dim=1)
                attended, _ = self.step_attention(
                    h.unsqueeze(1), trajectory_tensor, trajectory_tensor
                )
                attended = attended.squeeze(1)
                
                # GRU update
                h_new = self.reasoning_step(attended, h)
                
                # Only update uncertain cases
                h = torch.where(deliberate_mask, h_new, h)
                trajectory.append(h)
                
                # Update steps counter
                steps_used = torch.where(
                    needs_deliberation, 
                    steps_used + 1, 
                    steps_used
                )
                
                # Early stopping if confidence increases
                new_confidence = torch.sigmoid(self.confidence(h))
                still_uncertain = new_confidence.squeeze() < self.threshold
                needs_deliberation = needs_deliberation & still_uncertain
                
                if not needs_deliberation.any():
                    break
        
        output = self.output(h)
        
        info = {
            'confidence': confidence,
            'steps_used': steps_used,
            'deliberation_used': (steps_used > 1).float().mean()
        }
        
        return output, info


# Example usage
def demo_neuro_symbolic():
    """
    Demo: Neuro-symbolic reasoning.
    """
    # Create reasoner with 5 predicates and 10 constants
    reasoner = SymbolicReasoner(
        embedding_dim=32,
        num_predicates=5,
        num_constants=10,
        logic_type="product"
    )
    
    # Example: Compute "Parent(x, y) AND Parent(y, z) → Grandparent(x, z)"
    # Predicate indices: Parent=0, Grandparent=1
    # Constants: Alice=0, Bob=1, Charlie=2
    
    # Ground predicates
    parent_ab = reasoner.ground_predicate(0, [0, 1])  # Parent(Alice, Bob)
    parent_bc = reasoner.ground_predicate(0, [1, 2])  # Parent(Bob, Charlie)
    grandparent_ac = reasoner.ground_predicate(1, [0, 2])  # Grandparent(Alice, Charlie)
    
    # Compute rule satisfaction
    premise = reasoner.fuzzy_and(parent_ab, parent_bc)
    rule_satisfaction = reasoner.fuzzy_implies(premise, grandparent_ac)
    
    print("Neuro-Symbolic Reasoning Demo:")
    print(f"  Parent(Alice, Bob) = {parent_ab.item():.3f}")
    print(f"  Parent(Bob, Charlie) = {parent_bc.item():.3f}")
    print(f"  Grandparent(Alice, Charlie) = {grandparent_ac.item():.3f}")
    print(f"  Rule satisfaction = {rule_satisfaction.item():.3f}")


if __name__ == "__main__":
    demo_neuro_symbolic()
```

---

## 9. Phase 7: Meta-Learning & Plasticity

The brain doesn't just learn—it learns how to learn. This layer enables rapid adaptation to new tasks and gates when learning should occur.

### 9.1 Meta-Learning Implementation

```python
# meta_learning.py
"""
Meta-Learning and Plasticity Modulation.

Implements "learning to learn" through:
- MAML (Model-Agnostic Meta-Learning): Learn initializations for fast adaptation
- Neuromodulation: Gate when and what to learn
- Eligibility traces: Enable learning on behavioral timescales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Callable

class MAML:
    """
    Model-Agnostic Meta-Learning.
    
    Learns model initializations that enable rapid few-shot adaptation.
    The key insight: some initializations are better for learning than others.
    
    Algorithm:
    1. Sample batch of tasks
    2. For each task:
       a. Adapt model with few gradient steps on task
       b. Evaluate adapted model on held-out data
    3. Update initial parameters to improve post-adaptation performance
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        first_order: bool = False  # Use first-order approximation for efficiency
    ):
        """
        Args:
            model: The model to meta-learn
            inner_lr: Learning rate for task adaptation
            outer_lr: Learning rate for meta-update
            num_inner_steps: Number of gradient steps for adaptation
            first_order: If True, ignore second-order gradients (faster but less accurate)
        """
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order
        
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    
    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: Callable
    ) -> nn.Module:
        """
        Adapt model to a specific task using support set.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            loss_fn: Loss function for the task
        
        Returns:
            Adapted model (does not modify original)
        """
        # Clone model for adaptation
        adapted_model = deepcopy(self.model)
        
        for _ in range(self.num_inner_steps):
            # Forward pass
            outputs = adapted_model(support_x)
            loss = loss_fn(outputs, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_model.parameters(),
                create_graph=not self.first_order
            )
            
            # Update parameters
            with torch.no_grad() if self.first_order else torch.enable_grad():
                for param, grad in zip(adapted_model.parameters(), grads):
                    param.data = param.data - self.inner_lr * grad
        
        return adapted_model
    
    def meta_update(
        self,
        tasks: list[tuple],
        loss_fn: Callable
    ) -> float:
        """
        Perform meta-update across batch of tasks.
        
        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples
            loss_fn: Loss function
        
        Returns:
            Average meta-loss
        """
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Adapt to task
            adapted_model = self.adapt(support_x, support_y, loss_fn)
            
            # Evaluate on query set
            query_outputs = adapted_model(query_x)
            task_loss = loss_fn(query_outputs, query_y)
            
            meta_loss += task_loss
        
        # Average and backprop
        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()


class NeuromodulatoryGate(nn.Module):
    """
    Neuromodulatory gating for plasticity control.
    
    Biological brains use neuromodulators (dopamine, acetylcholine, etc.)
    to control when and what to learn. This module learns a gating signal
    that modulates learning rate based on context.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_modulators: int = 4  # Mimicking DA, ACh, NE, 5-HT
    ):
        super().__init__()
        
        self.num_modulators = num_modulators
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Modulator networks
        self.modulators = nn.ModuleList([
            nn.Linear(32, 1) for _ in range(num_modulators)
        ])
        
        # Combine modulators into learning rate multiplier
        self.combiner = nn.Linear(num_modulators, 1)
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Compute learning rate multiplier from context.
        
        Args:
            context: Contextual information (batch, input_dim)
        
        Returns:
            Learning rate multiplier in [0, 2]
        """
        encoded = self.context_encoder(context)
        
        modulator_values = []
        for mod in self.modulators:
            val = torch.sigmoid(mod(encoded))
            modulator_values.append(val)
        
        combined = torch.cat(modulator_values, dim=-1)
        multiplier = 2 * torch.sigmoid(self.combiner(combined))
        
        return multiplier.squeeze(-1)


class EligibilityTrace(nn.Module):
    """
    Eligibility traces for temporal credit assignment.
    
    In biological learning, synapses become "eligible" for modification
    when pre-synaptic and post-synaptic activity coincide. The actual
    modification occurs when a neuromodulatory signal arrives.
    
    This enables learning from delayed rewards (solving the temporal
    credit assignment problem).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        trace_decay: float = 0.95
    ):
        super().__init__()
        
        self.trace_decay = trace_decay
        
        # Synaptic weights
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Eligibility trace (not a parameter, but persistent state)
        self.register_buffer(
            'trace',
            torch.zeros(output_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with eligibility trace update.
        
        Args:
            x: Input tensor (batch, input_dim)
        
        Returns:
            Output tensor (batch, output_dim)
        """
        # Standard linear transformation
        output = F.linear(x, self.weight, self.bias)
        
        # Update eligibility trace (Hebbian-like: pre * post correlation)
        # Averaged over batch
        pre = x.mean(dim=0)  # (input_dim,)
        post = torch.sigmoid(output).mean(dim=0)  # (output_dim,)
        
        # Outer product gives correlation matrix
        correlation = post.unsqueeze(-1) * pre.unsqueeze(0)
        
        # Decay and accumulate trace
        self.trace = self.trace_decay * self.trace + correlation
        
        return output
    
    def modulated_update(
        self,
        modulation_signal: float,
        learning_rate: float = 0.01
    ):
        """
        Apply modulated weight update using eligibility trace.
        
        Args:
            modulation_signal: Reward/error signal (e.g., TD error)
            learning_rate: Base learning rate
        """
        with torch.no_grad():
            # Weight change proportional to trace * modulation
            delta_w = learning_rate * modulation_signal * self.trace
            self.weight.data += delta_w
            
            # Optionally decay trace after update
            self.trace *= 0.5  # Partial reset
    
    def reset_trace(self):
        """Reset eligibility trace to zero."""
        self.trace.zero_()


class PlasticNetwork(nn.Module):
    """
    Network with learnable plasticity rules.
    
    Instead of fixed learning rules (like SGD), this network
    learns how to modify its own weights based on experience.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ):
        super().__init__()
        
        # Main network weights
        self.w1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.w2 = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)
        
        # Hebbian plasticity coefficients (learnable)
        self.alpha1 = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        self.alpha2 = nn.Parameter(torch.zeros(output_dim, hidden_dim))
        
        # Hebbian traces (not parameters)
        self.register_buffer('hebb1', torch.zeros(hidden_dim, input_dim))
        self.register_buffer('hebb2', torch.zeros(output_dim, hidden_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with plastic weights.
        
        The effective weight is: w + alpha * hebb
        where alpha controls how much plasticity affects the weight.
        """
        # Effective weights = fixed + plastic
        eff_w1 = self.w1 + self.alpha1 * self.hebb1
        eff_w2 = self.w2 + self.alpha2 * self.hebb2
        
        # Forward pass
        h = torch.relu(F.linear(x, eff_w1))
        out = F.linear(h, eff_w2)
        
        # Update Hebbian traces (outer product of activations)
        with torch.no_grad():
            self.hebb1 = 0.95 * self.hebb1 + 0.05 * (h.T @ x) / x.shape[0]
            self.hebb2 = 0.95 * self.hebb2 + 0.05 * (out.T @ h) / h.shape[0]
        
        return out
    
    def reset_plasticity(self):
        """Reset Hebbian traces for new episode."""
        self.hebb1.zero_()
        self.hebb2.zero_()


# Example usage
def demo_meta_learning():
    """
    Demo: MAML for few-shot learning.
    """
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    # Create MAML wrapper
    maml = MAML(
        model=model,
        inner_lr=0.01,
        outer_lr=0.001,
        num_inner_steps=5
    )
    
    # Simulate meta-training
    print("Meta-Learning Demo:")
    
    for epoch in range(3):
        # Sample batch of tasks
        tasks = []
        for _ in range(4):  # 4 tasks per batch
            # Each task has different linear relationship
            w = torch.randn(10)
            support_x = torch.randn(5, 10)  # 5-shot
            support_y = (support_x @ w).unsqueeze(-1)
            query_x = torch.randn(10, 10)
            query_y = (query_x @ w).unsqueeze(-1)
            tasks.append((support_x, support_y, query_x, query_y))
        
        # Meta-update
        loss = maml.meta_update(tasks, nn.MSELoss())
        print(f"  Epoch {epoch+1}: Meta-loss = {loss:.4f}")


if __name__ == "__main__":
    demo_meta_learning()
```

---

## 10. System Integration

Now we integrate all components into a unified architecture.

### 10.1 Complete Brain-Inspired System

```python
# brain_ai_system.py
"""
Complete Brain-Inspired AI System.

Integrates all seven layers into a unified architecture:
1. Event-driven sensory processing
2. Spiking neural network core
3. Hierarchical temporal memory
4. Global workspace / working memory
5. Decision and action system
6. Neuro-symbolic reasoning
7. Meta-learning and plasticity
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

# Import components (assuming they're in the same directory)
# from snn_core import SNNCore, ConvSNN
# from event_processing import EventProcessor, PredictiveCodingLayer
# from htm_layer import HTMLayer
# from global_workspace import GlobalWorkspace, LiquidWorkingMemory
# from decision_system import ActiveInferenceAgent, DriftDiffusionModel
# from neuro_symbolic import System2Reasoner
# from meta_learning import NeuromodulatoryGate, EligibilityTrace

class BrainInspiredAI(nn.Module):
    """
    Complete brain-inspired AI architecture.
    
    This system processes information through multiple specialized
    layers that mirror biological brain organization:
    
    - Sparse, event-driven processing (like biological neurons)
    - Predictive coding (only transmit prediction errors)
    - Online sequence learning (without catastrophic forgetting)
    - Global workspace for information integration
    - Principled decision-making under uncertainty
    - Verified symbolic reasoning when needed
    - Adaptive learning rate based on context
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 10,
        num_timesteps: int = 25,
        use_snn: bool = True,
        use_htm: bool = False,  # Requires htm.core
        use_workspace: bool = True,
        use_symbolic: bool = True,
        use_meta: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_timesteps = num_timesteps
        
        # Feature flags
        self.use_snn = use_snn
        self.use_htm = use_htm
        self.use_workspace = use_workspace
        self.use_symbolic = use_symbolic
        self.use_meta = use_meta
        
        # Layer 1: Predictive Coding (sensory preprocessing)
        self.predictive_coding = PredictiveCodingLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        
        # Layer 2: SNN Core
        if use_snn:
            self.snn = SNNCore(
                input_size=input_dim,
                hidden_sizes=[hidden_dim, hidden_dim // 2],
                output_size=hidden_dim // 2,
                num_steps=num_timesteps
            )
        else:
            self.snn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2)
            )
        
        # Layer 3: HTM (if available)
        if use_htm:
            try:
                self.htm = HTMLayer(
                    input_size=hidden_dim // 2,
                    column_count=1024,
                    cells_per_column=16
                )
            except ImportError:
                print("HTM not available, using placeholder")
                self.htm = None
                use_htm = False
        
        # Layer 4: Global Workspace
        if use_workspace:
            workspace_inputs = {
                "snn_features": hidden_dim // 2,
                "context": hidden_dim // 4
            }
            self.workspace = GlobalWorkspace(
                input_dims=workspace_inputs,
                workspace_size=hidden_dim
            )
            self.workspace_memory = LiquidWorkingMemory(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                mode="cfc"
            )
        
        # Layer 5: Decision System (simplified)
        self.decision_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Layer 6: System 2 Reasoner
        if use_symbolic:
            self.reasoner = System2Reasoner(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_reasoning_steps=3
            )
        
        # Layer 7: Neuromodulatory Gate
        if use_meta:
            self.neuromodulator = NeuromodulatoryGate(
                input_dim=hidden_dim,
                num_modulators=4
            )
        
        # Output layers
        self.output_layer = nn.Linear(output_dim, output_dim)
        
        # State tracking
        self.workspace_state = None
        self.memory_state = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass through the brain-inspired architecture.
        
        Args:
            x: Input tensor (batch, features) or (batch, time, features)
            return_intermediates: If True, return intermediate representations
        
        Returns:
            Dict containing:
                - 'output': Final output (batch, output_dim)
                - 'confidence': Decision confidence
                - 'learning_rate_mult': Neuromodulatory learning rate multiplier
                - 'intermediates': (optional) Dict of intermediate representations
        """
        batch_size = x.shape[0]
        intermediates = {}
        
        # Ensure 2D input for non-temporal processing
        if x.dim() == 3:
            # Average over time for now
            x = x.mean(dim=1)
        
        # Layer 1: Predictive Coding
        prediction_error, sensory_hidden = self.predictive_coding(x)
        intermediates['prediction_error'] = prediction_error
        
        # Layer 2: SNN Processing
        if self.use_snn:
            spikes, snn_mem = self.snn(prediction_error)
            snn_features = spikes.sum(dim=0)  # Rate coding
        else:
            snn_features = self.snn(prediction_error)
        intermediates['snn_features'] = snn_features
        
        # Layer 3: HTM (if available)
        if self.use_htm and self.htm is not None:
            # Would process through HTM here
            # For now, pass through
            htm_output = snn_features
        else:
            htm_output = snn_features
        
        # Layer 4: Global Workspace
        if self.use_workspace:
            # Create context from sensory hidden state
            context = sensory_hidden[:, :self.hidden_dim // 4]
            
            workspace_inputs = {
                "snn_features": htm_output,
                "context": context
            }
            
            broadcasts, attention, self.workspace_state = self.workspace(
                workspace_inputs,
                self.workspace_state
            )
            
            # Process through working memory
            workspace_output = broadcasts["snn_features"]
            workspace_output = workspace_output.unsqueeze(1)  # Add time dim
            memory_output, self.memory_state = self.workspace_memory(
                workspace_output, self.memory_state
            )
            memory_output = memory_output.squeeze(1)
            
            intermediates['attention'] = attention
            intermediates['workspace'] = memory_output
            
            features = memory_output
        else:
            features = htm_output
            # Pad to correct dimension
            if features.shape[-1] != self.hidden_dim:
                features = F.pad(features, (0, self.hidden_dim - features.shape[-1]))
        
        # Layer 5: Decision
        decision_output = self.decision_layer(features)
        
        # Layer 6: Symbolic Reasoning (if low confidence)
        if self.use_symbolic:
            reasoned, reason_info = self.reasoner(features)
            confidence = reason_info['confidence']
            
            # Blend based on confidence
            alpha = 1 - confidence  # Use reasoning more when confidence is low
            final_features = alpha * reasoned + (1 - alpha) * features
            decision_output = self.decision_layer(final_features)
            
            intermediates['reasoning_steps'] = reason_info['steps_used']
        else:
            confidence = torch.ones(batch_size, 1, device=x.device)
        
        # Layer 7: Neuromodulation
        if self.use_meta:
            learning_rate_mult = self.neuromodulator(features)
        else:
            learning_rate_mult = torch.ones(batch_size, device=x.device)
        
        # Final output
        output = self.output_layer(decision_output)
        
        result = {
            'output': output,
            'confidence': confidence,
            'learning_rate_mult': learning_rate_mult
        }
        
        if return_intermediates:
            result['intermediates'] = intermediates
        
        return result
    
    def reset_state(self):
        """Reset all stateful components for new episode."""
        self.workspace_state = None
        self.memory_state = None
        self.predictive_coding.reset()


# Simplified placeholder implementations for demo
class PredictiveCodingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, input_dim)
        self.hidden = None
    
    def forward(self, x):
        if self.hidden is None:
            self.hidden = torch.zeros(x.shape[0], self.encoder.out_features, device=x.device)
        prediction = self.predictor(self.hidden)
        error = x - prediction
        self.hidden = self.encoder(x)
        return error, self.hidden
    
    def reset(self):
        self.hidden = None


class SNNCore(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i+1])
            for i in range(len(sizes)-1)
        ])
    
    def forward(self, x):
        spikes = []
        mem = torch.zeros(x.shape[0], self.layers[-1].out_features, device=x.device)
        for t in range(self.num_steps):
            out = x
            for layer in self.layers:
                out = torch.relu(layer(out))
            spike = (out > 0.5).float()
            spikes.append(spike)
            mem = 0.9 * mem + out
        return torch.stack(spikes), mem


class GlobalWorkspace(nn.Module):
    def __init__(self, input_dims, workspace_size):
        super().__init__()
        self.projs = nn.ModuleDict({
            k: nn.Linear(v, workspace_size) for k, v in input_dims.items()
        })
        self.attention = nn.MultiheadAttention(workspace_size, 4, batch_first=True)
    
    def forward(self, inputs, state=None):
        projected = [self.projs[k](v.unsqueeze(1)) for k, v in inputs.items()]
        stacked = torch.cat(projected, dim=1)
        if state is None:
            query = stacked.mean(dim=1, keepdim=True)
        else:
            query = state.unsqueeze(1)
        attended, attn = self.attention(query, stacked, stacked)
        new_state = attended.squeeze(1)
        broadcasts = {k: new_state for k in inputs.keys()}
        return broadcasts, attn.squeeze(1), new_state


class LiquidWorkingMemory(nn.Module):
    def __init__(self, input_size, hidden_size, mode="cfc"):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
    
    def forward(self, x, h0=None):
        if h0 is not None:
            h0 = h0.unsqueeze(0)
        output, hn = self.rnn(x, h0)
        return output, hn.squeeze(0)


class System2Reasoner(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_reasoning_steps=3):
        super().__init__()
        self.steps = num_reasoning_steps
        self.reason = nn.GRUCell(hidden_dim, hidden_dim)
        self.conf = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, force=False):
        h = x
        for _ in range(self.steps):
            h = self.reason(x, h)
        conf = torch.sigmoid(self.conf(h))
        return h, {'confidence': conf, 'steps_used': torch.tensor(self.steps)}


class NeuromodulatoryGate(nn.Module):
    def __init__(self, input_dim, num_modulators=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return 2 * self.net(x).squeeze(-1)


# Demo
def demo_full_system():
    """Demo the complete brain-inspired AI system."""
    
    system = BrainInspiredAI(
        input_dim=784,
        hidden_dim=256,
        output_dim=10,
        use_snn=True,
        use_htm=False,
        use_workspace=True,
        use_symbolic=True,
        use_meta=True
    )
    
    # Test forward pass
    batch = torch.randn(4, 784)
    result = system(batch, return_intermediates=True)
    
    print("Brain-Inspired AI System Demo:")
    print(f"  Output shape: {result['output'].shape}")
    print(f"  Confidence: {result['confidence'].mean():.3f}")
    print(f"  Learning rate multiplier: {result['learning_rate_mult'].mean():.3f}")
    print(f"  Intermediate keys: {list(result['intermediates'].keys())}")


if __name__ == "__main__":
    demo_full_system()
```

---

## 11. Deployment Options

### 11.1 Neuromorphic Hardware Deployment

```python
# deployment.py
"""
Deployment options for brain-inspired AI systems.

Neuromorphic hardware provides massive energy efficiency gains
by natively supporting spike-based computation.
"""

# Intel Loihi 2 deployment via Lava
def deploy_to_loihi():
    """
    Deploy SNN to Intel Loihi 2.
    
    Requires:
    - Intel Neuromorphic Research Community membership
    - Lava framework: pip install lava-nc
    """
    try:
        from lava.lib.dl import slayer
        from lava.proc.lif import LIF
        from lava.proc.dense import Dense
        
        print("Lava framework available for Loihi deployment")
        # See Intel Lava documentation for full examples
    except ImportError:
        print("Lava not installed. Install via: pip install lava-nc")


# BrainChip Akida deployment
def deploy_to_akida():
    """
    Deploy to BrainChip Akida.
    
    Akida supports CNN, RNN, and Vision Transformers with on-chip learning.
    Commercial hardware available ($799 dev kit).
    
    Requires:
    - pip install akida
    - pip install cnn2snn  # For conversion
    """
    try:
        import akida
        from cnn2snn import convert
        
        print(f"Akida SDK version: {akida.__version__}")
        # See BrainChip documentation for full examples
    except ImportError:
        print("Akida SDK not installed. Install via: pip install akida cnn2snn")


# SpikingJelly provides neuromorphic export
def export_for_neuromorphic(model, example_input, output_path="model_neuromorphic"):
    """
    Export SpikingJelly model for neuromorphic deployment.
    
    SpikingJelly supports export to:
    - CuPy backend for GPU acceleration
    - ONNX for hardware portability
    - Darwin chip format
    """
    try:
        from spikingjelly.activation_based import functional
        
        # Reset model state
        functional.reset_net(model)
        
        # Export to ONNX (if supported)
        import torch.onnx
        torch.onnx.export(
            model,
            example_input,
            f"{output_path}.onnx",
            opset_version=11
        )
        print(f"Model exported to {output_path}.onnx")
    except Exception as e:
        print(f"Export failed: {e}")
```

---

## 12. Key Resources & Papers

### Frameworks

| Framework | Purpose | Repository |
|-----------|---------|------------|
| **snnTorch** | SNN training | github.com/jeshraghian/snntorch |
| **SpikingJelly** | Full-stack SNN | github.com/fangwei123456/spikingjelly |
| **ncps** | Liquid NNs | github.com/mlech26l/ncps |
| **pymdp** | Active Inference | github.com/infer-actively/pymdp |
| **htm.core** | HTM | github.com/htm-community/htm.core |
| **Lava** | Loihi deployment | github.com/lava-nc/lava |

### Key Papers

1. **Spiking Neural Networks**: "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence" (Science Advances, 2023)

2. **Liquid Neural Networks**: "Liquid Time-constant Networks" (AAAI 2021) - Hasani et al.

3. **Active Inference**: "pymdp: A Python library for active inference in discrete state spaces" (JOSS, 2022)

4. **Predictive Coding**: "Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?" (arXiv 2022)

5. **Neuromorphic Computing**: "Intel's Hala Point system" (Intel, 2024) - 1.15B neurons

6. **HTM**: "Biological and Machine Intelligence" (BAMI) - Numenta

### Tutorials

- snnTorch tutorials: snntorch.readthedocs.io/en/latest/tutorials
- SpikingJelly docs: spikingjelly.readthedocs.io
- pymdp notebooks: github.com/infer-actively/pymdp/tree/master/examples
- ncps Google Colab: github.com/mlech26l/ncps#tutorials

---

## Summary

This guide provides a complete roadmap for building a brain-inspired AI system:

1. **Start with Phase 1**: Get snnTorch working with MNIST (~1 week)
2. **Add Event Processing**: Integrate neuromorphic datasets (weeks 2-3)
3. **Build HTM Layer**: For sequence learning without forgetting (weeks 4-5)
4. **Integrate Global Workspace**: Using Liquid NNs for working memory (weeks 6-8)
5. **Add Decision System**: Implement active inference (weeks 9-12)
6. **Neuro-Symbolic Reasoning**: For verified logical inference (weeks 13-16)
7. **Meta-Learning**: Adaptive plasticity modulation (weeks 17-20)
8. **System Integration**: Connect all components (weeks 21-26)

The key insight: biological brains solve intelligence problems with ~20 watts. By mimicking their computational principles—sparse event-driven processing, predictive coding, sequence learning, global workspace integration, and principled decision-making—we can build more efficient and robust AI systems.

---

*Document generated for LLM implementation guidance. Last updated: January 2025*