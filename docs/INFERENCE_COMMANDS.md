# Inference Commands Guide

> Quick reference for running inference with the brain-inspired AI model using the CLI.

---

## Prerequisites

```bash
# Activate virtual environment
source human/bin/activate

# Ensure model checkpoints exist
ls checkpoints/

# Optional: Install server dependencies
pip install fastapi uvicorn python-multipart
```

---

## Quick Start

```bash
# Run demo with all modalities
python -m brain_ai.cli demo

# Start interactive session
python -m brain_ai.cli interactive

# Get model info
python -m brain_ai.cli info --checkpoint checkpoints/snn_mnist.pth
```

---

## Classification

### Image Classification

```bash
# Classify single image
python -m brain_ai.cli classify \
    --checkpoint checkpoints/vision_encoder.pth \
    --image path/to/image.jpg

# With top-k predictions
python -m brain_ai.cli classify \
    --checkpoint checkpoints/vision_encoder.pth \
    --image path/to/image.jpg \
    --top-k 10

# Using GPU
python -m brain_ai.cli classify \
    --checkpoint checkpoints/vision_encoder.pth \
    --image path/to/image.jpg \
    --device cuda
```

### Text Classification

```bash
# Classify text
python -m brain_ai.cli classify \
    --checkpoint checkpoints/neuro_symbolic.pth \
    --text "This is a sample text to classify"

# With verbose output
python -m brain_ai.cli classify \
    --checkpoint checkpoints/neuro_symbolic.pth \
    --text "Analyze this sentence for sentiment" \
    --top-k 5 \
    --verbose
```

### Audio Classification

```bash
# Classify audio file
python -m brain_ai.cli classify \
    --checkpoint checkpoints/snn_mnist.pth \
    --audio path/to/audio.wav

# Spoken digit recognition (SHD trained)
python -m brain_ai.cli classify \
    --checkpoint checkpoints/snn_shd.pth \
    --audio recordings/spoken_digit.wav \
    --top-k 3
```

---

## Text Generation

```bash
# Basic generation
python -m brain_ai.cli generate \
    --checkpoint checkpoints/global_workspace.pth \
    --prompt "Once upon a time"

# With parameters
python -m brain_ai.cli generate \
    --checkpoint checkpoints/global_workspace.pth \
    --prompt "The brain-inspired AI system" \
    --max-length 200 \
    --temperature 0.8

# Creative generation (higher temperature)
python -m brain_ai.cli generate \
    --checkpoint checkpoints/global_workspace.pth \
    --prompt "In a world where AI" \
    --max-length 500 \
    --temperature 1.2

# Focused generation (lower temperature)
python -m brain_ai.cli generate \
    --checkpoint checkpoints/global_workspace.pth \
    --prompt "The definition of neural networks is" \
    --max-length 100 \
    --temperature 0.3
```

---

## Batch Processing

### Process Multiple Images

```bash
# Process all images in directory
python -m brain_ai.cli batch \
    --checkpoint checkpoints/vision_encoder.pth \
    --input "images/*.jpg" \
    --output results.json

# With custom batch size
python -m brain_ai.cli batch \
    --checkpoint checkpoints/vision_encoder.pth \
    --input "data/test_images/*.png" \
    --output batch_results.json \
    --batch-size 64

# Process specific file patterns
python -m brain_ai.cli batch \
    --checkpoint checkpoints/vision_encoder.pth \
    --input "dataset/**/*.jpg" \
    --output nested_results.json \
    --batch-size 32
```

### Output Format

Batch results are saved as JSON:

```json
{
  "results": [
    {
      "file": "images/cat.jpg",
      "prediction": 3,
      "confidence": 0.95,
      "top_k_classes": [[3, 0.95], [5, 0.03], [1, 0.01]],
      "inference_time_ms": 12.5
    }
  ],
  "total": 100
}
```

---

## Interactive Mode

Start an interactive session for exploratory inference:

```bash
# Basic interactive mode
python -m brain_ai.cli interactive

# With specific checkpoint
python -m brain_ai.cli interactive \
    --checkpoint checkpoints/global_workspace.pth

# With GPU
python -m brain_ai.cli interactive \
    --checkpoint checkpoints/global_workspace.pth \
    --device cuda
```

### Interactive Commands

Once in interactive mode:

```
> image path/to/image.jpg     # Classify image
> text "Some text here"       # Classify text
> audio path/to/audio.wav     # Classify audio
> multimodal                  # Multi-modal input mode
> state                       # Show workspace state
> help                        # Show available commands
> quit                        # Exit session
```

---

## Demo Mode

Run demonstrations with sample inputs:

```bash
# Run all demos
python -m brain_ai.cli demo

# Vision only
python -m brain_ai.cli demo --modality vision

# Text only
python -m brain_ai.cli demo --modality text

# Audio only
python -m brain_ai.cli demo --modality audio

# With specific checkpoint
python -m brain_ai.cli demo \
    --checkpoint checkpoints/global_workspace.pth \
    --modality all
```

---

## Inference Server

Start a REST API server for remote inference:

### Start Server

```bash
# Default settings (port 8000)
python -m brain_ai.cli serve \
    --checkpoint checkpoints/global_workspace.pth

# Custom host and port
python -m brain_ai.cli serve \
    --checkpoint checkpoints/global_workspace.pth \
    --host 0.0.0.0 \
    --port 8080

# With GPU
python -m brain_ai.cli serve \
    --checkpoint checkpoints/global_workspace.pth \
    --device cuda \
    --port 8000
```

### API Endpoints

Once server is running:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/classify/image` | POST | Classify uploaded image |
| `/classify/text` | POST | Classify text |
| `/generate` | POST | Generate text |
| `/docs` | GET | OpenAPI documentation |

### API Examples

```bash
# Health check
curl http://localhost:8000/health

# Classify image
curl -X POST http://localhost:8000/classify/image \
    -F "file=@image.jpg"

# Classify text
curl -X POST http://localhost:8000/classify/text \
    -F "text=This is a test sentence"

# Generate text
curl -X POST http://localhost:8000/generate \
    -F "prompt=Once upon a time" \
    -F "max_length=100" \
    -F "temperature=0.8"
```

### Python Client Example

```python
import requests

# Classify image
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/classify/image',
        files={'file': f}
    )
    print(response.json())

# Classify text
response = requests.post(
    'http://localhost:8000/classify/text',
    data={'text': 'Sample text to classify'}
)
print(response.json())

# Generate text
response = requests.post(
    'http://localhost:8000/generate',
    data={
        'prompt': 'The future of AI is',
        'max_length': 150,
        'temperature': 0.7
    }
)
print(response.json())
```

---

## Python API Usage

### Direct Inference

```python
from brain_ai.inference import BrainInference

# Load from checkpoint
brain = BrainInference.load('checkpoints/global_workspace.pth')

# Classify image
result = brain.classify_image('path/to/image.jpg', top_k=5)
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.2%}")

# Classify text
result = brain.classify_text("Sample text to analyze")
print(result.to_dict())

# Multi-modal inference
result = brain.infer({
    'vision': image_tensor,
    'text': "What is in this image?",
})
```

### Load Pretrained Models

```python
from brain_ai.inference import BrainInference

# Load by preset name
brain = BrainInference.from_pretrained('vision-classifier')
brain = BrainInference.from_pretrained('multimodal-reasoning')
brain = BrainInference.from_pretrained('text-classifier')
brain = BrainInference.from_pretrained('control-agent')
```

### Batch Inference

```python
from brain_ai.inference import BrainInference

brain = BrainInference.load('checkpoints/vision_encoder.pth')

# Process multiple images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = brain.batch_classify(image_paths, batch_size=32)

for path, result in zip(image_paths, results):
    print(f"{path}: {result.prediction} ({result.confidence:.2%})")
```

---

## Model-Specific Inference

### SNN Core (Phase 1)

```bash
# MNIST digit classification
python -m brain_ai.cli classify \
    --checkpoint checkpoints/snn_mnist.pth \
    --image digit.png \
    --top-k 10
```

### Vision Encoder (Phase 2)

```bash
# Event-based vision classification
python -m brain_ai.cli classify \
    --checkpoint checkpoints/vision_encoder.pth \
    --image event_frame.png
```

### HTM Layer (Phase 3)

```python
# HTM is typically used for sequence prediction
from brain_ai.temporal.htm import HTMLayer

htm = HTMLayer.load('checkpoints/htm_layer.pth')
prediction, anomaly = htm.predict_next(current_input)
print(f"Anomaly score: {anomaly:.3f}")
```

### Global Workspace (Phase 4)

```bash
# Multi-modal fusion
python -m brain_ai.cli demo \
    --checkpoint checkpoints/global_workspace.pth \
    --modality all
```

### Active Inference (Phase 5)

```python
# Control and decision making
from brain_ai.decision.active_inference import ActiveInferenceAgent

agent = ActiveInferenceAgent.load('checkpoints/active_inference.pth')
action = agent.select_action(observation)
print(f"Selected action: {action}")
```

### Neuro-Symbolic (Phase 6)

```bash
# Reasoning with interpretable traces
python -m brain_ai.cli classify \
    --checkpoint checkpoints/neuro_symbolic.pth \
    --text "If A implies B and B implies C, does A imply C?" \
    --verbose
```

### Meta-Learning (Phase 7)

```python
# Few-shot adaptation
from brain_ai.meta.maml import MAML

maml = MAML.load('checkpoints/meta_learning.pth')

# Adapt to new task
adapted_model = maml.adapt(support_x, support_y, steps=5)
predictions = adapted_model(query_x)
```

---

## Global CLI Arguments

These arguments work with all commands:

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--checkpoint` | `-c` | None | Path to model checkpoint |
| `--device` | | auto | Device (auto/cuda/cpu/mps) |
| `--verbose` | `-v` | false | Verbose output |

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Use CPU instead
python -m brain_ai.cli classify --device cpu --image image.jpg

# Reduce batch size for batch processing
python -m brain_ai.cli batch --batch-size 8 --input "*.jpg" --output results.json
```

### Missing Dependencies

```bash
# Vision processing
pip install torchvision pillow

# Audio processing
pip install torchaudio scipy

# Text processing
pip install transformers

# Server
pip install fastapi uvicorn python-multipart
```

### No Checkpoint Found

```bash
# Check available checkpoints
ls -la checkpoints/

# Train a model first
python scripts/train_phase1.py

# Or use demo mode (creates default model)
python -m brain_ai.cli demo
```
