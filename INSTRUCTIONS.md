# Instructions: Run, Train, and Inference

This guide covers how to set up, run, train, and perform inference with the Brain-Inspired AI System.

## Table of Contents

1. [Setup](#setup)
2. [Running the System](#running-the-system)
3. [Training](#training)
4. [Inference](#inference)
5. [Configuration Options](#configuration-options)
6. [Troubleshooting](#troubleshooting)

---

## Setup

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM (16GB+ recommended for full system)

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd human-brain

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "from brain_ai import create_brain_ai; print('Installation successful!')"
```

### Optional: Install HTM Core

For native Hierarchical Temporal Memory (instead of LSTM fallback):

```bash
# Build from source (requires CMake)
git clone https://github.com/htm-community/htm.core
cd htm.core
python setup.py install
```

---

## Running the System

### Basic Usage

```python
from brain_ai import create_brain_ai
import torch

# Create a brain with default settings
brain = create_brain_ai(
    modalities=['vision'],
    output_type='classify',
    num_classes=10,
    device='auto'  # auto-detect GPU/CPU
)

# Prepare input (example: random image batch)
images = torch.randn(4, 1, 28, 28)  # batch of 4, 1 channel, 28x28

# Run forward pass
output = brain({'vision': images})
print(f"Output shape: {output.shape}")  # [4, 10]
print(f"Predictions: {output.argmax(dim=1)}")
```

### Multi-Modal Usage

```python
from brain_ai import create_brain_ai
import torch

# Create multi-modal brain
brain = create_brain_ai(
    modalities=['vision', 'text'],
    output_type='classify',
    num_classes=100,
    use_htm=True,
    use_symbolic=True,
    device='cuda'
)

# Prepare inputs
images = torch.randn(4, 3, 224, 224)  # RGB images
text_ids = torch.randint(0, 50000, (4, 128))  # tokenized text

# Forward pass
output = brain({
    'vision': images,
    'text': text_ids
})
```

### Getting Detailed Output

```python
# Enable detailed output mode
result = brain(inputs, return_details=True)

# Access components
print(f"Predictions: {result.output}")
print(f"Workspace representation: {result.workspace.shape}")
print(f"Confidence: {result.confidence}")
print(f"Attention weights: {result.attention}")
print(f"Reasoning trace: {result.reasoning_trace}")
print(f"Modulators: {result.modulators}")
```

---

## Training

### Phase 1: SNN Core Training (MNIST)

The first training phase validates the SNN core on MNIST classification.

```bash
# Basic training
python scripts/train_phase1.py

# With custom parameters
python scripts/train_phase1.py \
    --epochs 20 \
    --batch-size 128 \
    --model ff \
    --hidden 800 400 \
    --num-steps 25 \
    --lr 1e-3 \
    --device cuda

# Train convolutional SNN
python scripts/train_phase1.py \
    --model conv \
    --epochs 20 \
    --batch-size 64
```

#### Training Script Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 128 | Batch size for training |
| `--model` | ff | Model type: `ff` (feedforward) or `conv` (convolutional) |
| `--hidden` | 800 400 | Hidden layer sizes (for ff model) |
| `--num-steps` | 25 | SNN timesteps |
| `--lr` | 1e-3 | Learning rate |
| `--device` | auto | Device: `auto`, `cuda`, or `cpu` |
| `--seed` | 42 | Random seed |

#### Expected Results

- **Feedforward SNN**: 97-98% accuracy on MNIST
- **Convolutional SNN**: 98-99% accuracy on MNIST
- Checkpoints saved to `checkpoints/snn_mnist.pth`

### Custom Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from brain_ai import create_brain_ai

# Create model
brain = create_brain_ai(
    modalities=['vision'],
    output_type='classify',
    num_classes=10,
    device='cuda'
)

# Setup training
optimizer = torch.optim.Adam(brain.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    brain.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        output = brain({'vision': images})
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    # Validation - set model to inference mode
    brain.train(False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            output = brain({'vision': images})
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Epoch {epoch}, Accuracy: {100.*correct/total:.2f}%")

# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': brain.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'accuracy': 100.*correct/total,
}, 'checkpoints/brain_ai_checkpoint.pth')
```

### Training with Engram Memory

```python
from brain_ai import create_brain_ai

# Enable Engram for text tasks
brain = create_brain_ai(
    modalities=['text'],
    output_type='classify',
    num_classes=5,  # e.g., sentiment classes
    use_engram=True,
    device='cuda'
)

# Forward pass includes N-gram memory lookup
output = brain({
    'text': text_ids,
    'token_ids': text_ids,  # Required for Engram lookup
})
```

### Training with TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/brain_ai_experiment')

for epoch in range(num_epochs):
    # ... training code ...

    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)

    # Log confidence histogram
    if result.confidence is not None:
        writer.add_histogram('Confidence', result.confidence, epoch)

writer.close()
```

Launch TensorBoard:
```bash
tensorboard --logdir=runs
```

---

## Inference

### Basic Inference

```python
import torch
from brain_ai import create_brain_ai

# Load trained model
brain = create_brain_ai(
    modalities=['vision'],
    output_type='classify',
    num_classes=10,
    device='cuda'
)

# Load checkpoint
checkpoint = torch.load('checkpoints/brain_ai_checkpoint.pth')
brain.load_state_dict(checkpoint['model_state_dict'])

# Set to inference mode (disables dropout, etc.)
brain.train(False)

# Run inference
with torch.no_grad():
    output = brain({'vision': image})
    prediction = output.argmax(dim=1)
    confidence = torch.softmax(output, dim=1).max(dim=1).values

print(f"Prediction: {prediction.item()}")
print(f"Confidence: {confidence.item():.2%}")
```

### Batch Inference

```python
brain.train(False)  # Set to inference mode
all_predictions = []
all_confidences = []

with torch.no_grad():
    for images in test_loader:
        images = images.cuda()
        output = brain({'vision': images})

        probs = torch.softmax(output, dim=1)
        preds = probs.argmax(dim=1)
        confs = probs.max(dim=1).values

        all_predictions.extend(preds.cpu().tolist())
        all_confidences.extend(confs.cpu().tolist())
```

### Inference with Detailed Analysis

```python
brain.train(False)  # Set to inference mode
with torch.no_grad():
    result = brain(inputs, return_details=True)

    # Get interpretable outputs
    prediction = result.output.argmax(dim=1)

    # Analyze attention weights (which modality contributed most)
    if result.attention is not None:
        print("Modality attention weights:")
        for modality, weight in result.attention.items():
            print(f"  {modality}: {weight.mean().item():.3f}")

    # Check reasoning trace (if symbolic reasoning enabled)
    if result.reasoning_trace is not None:
        print("\nReasoning trace:")
        for step in result.reasoning_trace:
            print(f"  {step}")

    # System confidence
    print(f"\nSystem confidence: {result.confidence.item():.2%}")
```

### Text Generation Inference

```python
from brain_ai import create_brain_ai

# Create model for text generation
brain = create_brain_ai(
    modalities=['text'],
    output_type='generate',
    vocab_size=50000,
    device='cuda'
)

# Load weights
brain.load_state_dict(torch.load('checkpoints/text_gen.pth')['model_state_dict'])
brain.train(False)  # Set to inference mode

# Generate text (autoregressive)
def generate(prompt_ids, max_length=100):
    generated = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(max_length):
            output = brain({'text': generated})
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

    return generated

result = generate(prompt_ids)
```

### Control/Robotics Inference

```python
from brain_ai import create_control_agent

# Create control agent
brain = create_control_agent(control_dim=6)  # 6-DOF control
brain.load_state_dict(torch.load('checkpoints/control.pth')['model_state_dict'])
brain.train(False)  # Set to inference mode

# Real-time control loop
while running:
    # Get sensor readings
    sensor_data = get_sensor_readings()

    with torch.no_grad():
        action = brain({'sensors': sensor_data})

    # Execute action
    execute_action(action.cpu().numpy())
```

---

## Configuration Options

### Full Configuration Reference

```python
from brain_ai.config import BrainAIConfig, SNNConfig, HTMConfig

config = BrainAIConfig(
    # Feature flags
    use_snn=True,
    use_htm=True,
    use_workspace=True,
    use_symbolic=True,
    use_meta=True,
    use_engram=False,

    # SNN settings
    snn=SNNConfig(
        num_timesteps=25,      # Integration timesteps
        beta=0.9,              # Membrane decay (0-1)
        threshold=1.0,         # Spike threshold
        surrogate='atan',      # Gradient: 'atan', 'fast_sigmoid', 'straight_through'
        dropout=0.2,
    ),

    # HTM settings
    htm=HTMConfig(
        column_count=2048,
        cells_per_column=32,
        sparsity=0.02,
    ),

    # Workspace settings
    workspace=WorkspaceConfig(
        workspace_dim=512,
        num_heads=8,
        capacity_limit=7,      # Miller's law
    ),

    # Decision settings
    decision=DecisionConfig(
        planning_horizon=3,
        epistemic_weight=1.0,  # Exploration vs exploitation
    ),

    # Reasoning settings
    reasoning=ReasoningConfig(
        num_reasoning_steps=5,
        use_system2=True,
    ),

    # Meta-learning settings
    meta=MetaConfig(
        inner_lr=0.01,
        outer_lr=0.001,
        num_inner_steps=5,
    ),

    # Engram settings
    engram=EngramConfig(
        vocab_size=50000,
        table_size=10_000_003,  # Prime number
        embedding_dim=256,
        ngram_orders=[2, 3],
        num_heads=4,
        offload_to_cpu=False,
    ),
)
```

### Quick Presets

```python
# Minimal configuration for testing
config = BrainAIConfig.minimal()

# Vision-only experiments
config = BrainAIConfig.for_vision_only()

# Custom from config
brain = create_brain_ai(config=config)
```

---

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**
```python
# Reduce batch size
train_loader = DataLoader(dataset, batch_size=32)  # was 128

# Enable gradient checkpointing
brain.enable_gradient_checkpointing()

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    output = brain(inputs)
```

**HTM Not Available**
```
Warning: htm.core not available, using LSTM fallback
```
This is normal if `htm.core` isn't installed. The system uses LSTM as a fallback.

**CUDA Not Found**
```python
# Force CPU
brain = create_brain_ai(..., device='cpu')
```

**Import Errors**
```bash
# Ensure you're in the right directory
cd /path/to/human-brain

# Ensure package is importable
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Performance Tips

1. **Use GPU**: 10-100x faster than CPU
2. **Batch processing**: Process multiple samples together
3. **Reduce timesteps**: Lower `num_timesteps` for faster but less accurate SNN
4. **Disable unused modules**: Set `use_htm=False`, etc. if not needed
5. **Use Engram for text**: O(1) lookup is much faster than deep attention

### Debug Mode

```python
# Enable verbose output
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model structure
print(brain)

# Check parameter count
total_params = sum(p.numel() for p in brain.parameters())
trainable_params = sum(p.numel() for p in brain.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

---

## Testing

Run the test suite to verify installation:

```bash
# All tests
python -m pytest tests/ -v

# Specific tests
python -m pytest tests/test_snn.py -v
python -m pytest tests/test_engram.py -v

# With coverage report
python -m pytest tests/ --cov=brain_ai --cov-report=html
```

---

## Next Steps

1. Start with Phase 1 training on MNIST to validate SNN core
2. Experiment with multi-modal inputs
3. Enable additional modules (HTM, symbolic reasoning)
4. Fine-tune configuration for your specific task
5. Explore the detailed architecture in `claude.md`
