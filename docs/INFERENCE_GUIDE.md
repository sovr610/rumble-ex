# 🧠 Brain-Inspired AI - Inference Guide

This guide shows how to interact with the Brain-Inspired AI model and perform inference.

## Quick Start

### 1. Loading a Trained Model

```python
from brain_ai import BrainInference

# Load from checkpoint
brain = BrainInference.load('checkpoints/model.pth')

# Or create a new model (for testing)
from brain_ai import create_brain_ai

model = create_brain_ai(
    modalities=['vision', 'text'],
    output_type='classify',
    num_classes=10,
)
brain = BrainInference(model=model)
```

### 2. Single Image Classification

```python
# From file path
result = brain.classify_image('path/to/image.jpg')

# From tensor
import torch
image = torch.randn(1, 3, 224, 224)  # [B, C, H, W]
result = brain.infer({'vision': image})

print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Top-5: {result.top_k_classes}")
```

### 3. Text Classification

```python
result = brain.classify_text("This is an amazing product!")

print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.2%}")
```

### 4. Multi-Modal Inference

```python
# Combine vision + text
result = brain.infer({
    'vision': image_tensor,
    'text': "What is in this image?",
})

print(f"Modalities used: {result.modalities_used}")
print(f"Prediction: {result.prediction}")
```

### 5. Text Generation

```python
# Create generator model
model = create_brain_ai(
    modalities=['text'],
    output_type='generate',
    vocab_size=10000,
)
brain = BrainInference(model=model)

# Generate text
output = brain.generate(
    prompt="Once upon a time",
    max_length=100,
    temperature=0.8,
)
```

### 6. Control Agent (RL)

```python
from brain_ai import create_control_agent

# Create agent
agent = create_control_agent(
    state_dim=32,
    action_dim=4,
)
brain = BrainInference(model=agent)

# Get action from observation
observation = torch.randn(32)
action = brain.get_action(observation, deterministic=True)

# Reset for new episode
brain.reset()
```

## InferenceResult Structure

Each inference call returns an `InferenceResult` with:

| Field | Type | Description |
|-------|------|-------------|
| `output` | Tensor | Raw model output |
| `prediction` | int/str | Predicted class index or name |
| `confidence` | float | Confidence score (0-1) |
| `probabilities` | Tensor | Class probabilities |
| `top_k_classes` | list | Top-k predictions with scores |
| `workspace_state` | Tensor | Global workspace representation |
| `attention_weights` | Tensor | Attention from workspace |
| `reasoning_used` | bool | Whether reasoning was triggered |
| `anomaly_score` | float | HTM anomaly detection score |
| `modalities_used` | list | Input modalities processed |
| `inference_time_ms` | float | Processing time in ms |

## Command Line Interface

```bash
# Interactive mode
python -m brain_ai.cli interactive --checkpoint model.pth

# Single classification
python -m brain_ai.cli classify --image photo.jpg
python -m brain_ai.cli classify --text "Sample text"

# Text generation
python -m brain_ai.cli generate --prompt "Once upon a time"

# Batch processing
python -m brain_ai.cli batch --input "images/*.jpg" --output results.json

# Start API server
python -m brain_ai.cli serve --port 8000

# Run demo
python -m brain_ai.cli demo
```

## API Server

Start a REST API server:

```bash
python -m brain_ai.cli serve --port 8000
```

Endpoints:
- `GET /health` - Health check
- `POST /classify/image` - Upload image for classification
- `POST /classify/text` - Text classification
- `POST /generate` - Text generation

Example request:
```bash
curl -X POST "http://localhost:8000/classify/text" \
     -F "text=This is a test"
```

## Batch Processing

For processing many images efficiently:

```python
# List of image paths
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']

# Batch classify
results = brain.batch_classify(image_paths, batch_size=32)

for path, result in zip(image_paths, results):
    print(f"{path}: {result.prediction} ({result.confidence:.2%})")
```

## Accessing Internal States

The brain exposes its internal representations:

```python
result = brain.infer({'vision': image})

# Global workspace state (consciousness)
workspace = result.workspace_state  # [batch, hidden_dim]

# Attention weights (what the model focused on)
attention = result.attention_weights  # [batch, num_experts, num_inputs]

# HTM anomaly score (novelty detection)
anomaly = result.anomaly_score  # 0.0 (familiar) to 1.0 (novel)

# Was symbolic reasoning used?
if result.reasoning_used:
    print("System-2 reasoning was activated")
```

## Using Class Names

For human-readable predictions:

```python
class_names = ['cat', 'dog', 'bird', 'fish', 'horse']

brain = BrainInference(
    model=model,
    class_names=class_names,
)

result = brain.classify_image('photo.jpg')
print(result.prediction)  # "cat" instead of 0
print(result.top_k_classes)  # [('cat', 0.85), ('dog', 0.10), ...]
```

## Saving and Loading Checkpoints

```python
import torch

# Save after training
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': {
        'modalities': ['vision', 'text'],
        'output_type': 'classify',
        'num_classes': 10,
    },
    'epoch': 100,
    'accuracy': 0.95,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load for inference
brain = BrainInference.load('checkpoint.pth')
```

## Complete Example

```python
import torch
from brain_ai import BrainInference, create_brain_ai

# 1. Create or load model
model = create_brain_ai(
    modalities=['vision', 'text', 'audio'],
    output_type='classify',
    num_classes=100,
)

# 2. Wrap for inference
brain = BrainInference(
    model=model,
    device='cuda',  # or 'cpu', 'mps'
    class_names=['class_0', 'class_1', ...],  # optional
)

# 3. Prepare inputs
image = torch.randn(1, 3, 224, 224)
text = "A description of the scene"
audio = torch.randn(1, 1, 16000)

# 4. Run multi-modal inference
result = brain.infer({
    'vision': image,
    'text': text,
    'audio': audio,
})

# 5. Use results
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Processing time: {result.inference_time_ms:.1f}ms")
print(f"Anomaly score: {result.anomaly_score:.4f}")

# 6. Interactive testing
brain.interactive()
```

## Model Presets

Quick-start configurations:

```python
from brain_ai import (
    create_vision_classifier,
    create_multimodal_system,
    create_control_agent,
)

# Vision-only classifier
vision_model = create_vision_classifier(num_classes=10)

# Multi-modal system
multimodal = create_multimodal_system(
    num_classes=100,
    modalities=['vision', 'text'],
)

# RL control agent
agent = create_control_agent(
    state_dim=32,
    action_dim=4,
)
```

## Next Steps

- Run the demo: `python examples/inference_demo.py`
- Train a model: `python scripts/train_phase1.py`
- See dataset loaders: [DATASET_LOADERS_GUIDE.md](DATASET_LOADERS_GUIDE.md)
