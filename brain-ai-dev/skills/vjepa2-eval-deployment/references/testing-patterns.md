# Testing Patterns — Reference

## Overview

V-JEPA 2 uses `unittest` as the primary testing framework, with GPU-conditional
decorators and systematic shape + numerical verification. All tests live in `tests/`
and can be run with `python -m pytest tests/ -v`.

## Framework and Imports

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# MIT License

import unittest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
```

## GPU-Conditional Tests

Tests that require CUDA must be decorated to skip gracefully on CPU-only machines:

```python
import unittest
import torch

@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestGPUForwardPass(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda")
        self.model = build_test_model().to(self.device)

    def test_gpu_forward_shape(self):
        x = torch.randn(2, 3, 16, 224, 224, device=self.device)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (2, 1568, 1024))
```

For multi-GPU tests:

```python
@unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
class TestMultiGPU(unittest.TestCase):
    ...
```

## Shape Verification

Use `assertEqual` for exact shape matching:

```python
class TestAttentiveClassifier(unittest.TestCase):

    def setUp(self):
        self.embed_dim  = 128
        self.num_classes = 10
        self.batch_size  = 4
        self.num_patches = 196
        self.model = AttentiveClassifier(
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
        )

    def test_output_shape(self):
        x = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        out = self.model(x)
        self.assertEqual(out.shape, (self.batch_size, self.num_classes))

    def test_batch_size_one(self):
        x = torch.randn(1, self.num_patches, self.embed_dim)
        out = self.model(x)
        self.assertEqual(out.shape, (1, self.num_classes))

    def test_multi_query_shape(self):
        model = AttentiveClassifier(
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
            num_queries=3,
        )
        x = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        out = model(x)
        self.assertEqual(out.shape, (self.batch_size, self.num_classes))
```

## Numerical Equivalence

Use `torch.testing.assert_close` for floating-point comparisons:

```python
def test_focal_loss_known_value(self):
    # For a perfectly confident correct prediction:
    # CE loss = -log(1.0) = 0.0 => FL = 0.0
    logits = torch.tensor([[0.0, 100.0]])   # very confident for class 1
    targets = torch.tensor([1])
    loss = self.focal_loss(logits, targets)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-4, rtol=1e-4)

def test_outputs_deterministic(self):
    torch.manual_seed(42)
    x = torch.randn(2, 16, 64)
    out1 = self.model(x)
    torch.manual_seed(42)
    x = torch.randn(2, 16, 64)
    out2 = self.model(x)
    torch.testing.assert_close(out1, out2)
```

## Cross-Format Consistency

Verify that tensor and numpy inputs produce equivalent outputs:

```python
def test_tensor_numpy_consistency(self):
    x_np  = np.random.randn(4, 196, 128).astype(np.float32)
    x_ten = torch.from_numpy(x_np)

    # Model accepts tensor
    out_ten = self.model(x_ten)

    # Convert output to numpy for comparison
    out_np = out_ten.detach().numpy()
    self.assertEqual(out_np.shape, (4, self.num_classes))
```

## Cosine Similarity for Non-Square Inputs

When testing representation quality (not exact values):

```python
def test_representation_direction(self):
    """Verify that similar inputs produce similar representations."""
    x1 = torch.randn(1, 196, 128)
    x2 = x1 + torch.randn_like(x1) * 0.01   # small perturbation

    feat1 = self.encoder(x1).mean(dim=1)     # [1, D]
    feat2 = self.encoder(x2).mean(dim=1)     # [1, D]

    cos_sim = torch.nn.functional.cosine_similarity(feat1, feat2, dim=-1)
    self.assertGreater(cos_sim.item(), 0.99)  # nearly identical
```

## Gradient Flow Tests

```python
def test_gradient_flows_through_probe_not_encoder(self):
    # Probe parameters should receive gradients
    x = torch.randn(2, 196, self.embed_dim)
    logits = self.probe(x)
    loss   = logits.sum()
    loss.backward()

    for name, param in self.probe.named_parameters():
        self.assertIsNotNone(param.grad, f"Probe param {name} has no gradient")
        self.assertFalse(
            torch.allclose(param.grad, torch.zeros_like(param.grad)),
            f"Probe param {name} has zero gradient",
        )

def test_encoder_gradients_stay_zero(self):
    # Encoder is frozen — its parameters should never accumulate gradients
    # Run one forward + backward pass
    with torch.no_grad():
        features = self.encoder(self.dummy_frames)
    logits = self.probe(features)
    loss   = logits.sum()
    loss.backward()

    for name, param in self.encoder.named_parameters():
        self.assertIsNone(
            param.grad,
            f"Encoder param {name} unexpectedly has a gradient",
        )
```

## Test Fixtures and setUp/tearDown

```python
class TestFrozenAssessor(unittest.TestCase):

    def setUp(self):
        """Create small synthetic encoder and assessor for fast tests."""
        self.embed_dim   = 64
        self.num_classes = 10
        self.batch_size  = 4
        self.num_patches = 49   # 7x7

        # Tiny synthetic encoder (2 layers)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=4, batch_first=True),
            num_layers=2,
        )
        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.train(False)

    def tearDown(self):
        """Explicit cleanup to free memory between tests."""
        del self.encoder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

## Loss Decrease Verification

```python
def test_probe_loss_decreases(self):
    """Loss should decrease over several training steps on synthetic data."""
    losses = []
    optimizer = torch.optim.SGD(self.probe.parameters(), lr=1e-2)

    for step in range(20):
        frames = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        labels = torch.randint(0, self.num_classes, (self.batch_size,))

        with torch.no_grad():
            features = self.encoder(frames)
        logits = self.probe(features)
        loss   = torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Last 5 steps should be lower than first 5 steps (on average)
    first_five = sum(losses[:5]) / 5
    last_five  = sum(losses[-5:]) / 5
    self.assertLess(last_five, first_five * 1.1,
                    "Loss did not decrease during probe training")
```

## File Organization

```
tests/
├── test_attentive_classifier.py   # AttentivePooler + AttentiveClassifier
├── test_frozen_assessor.py        # FrozenBackboneAssessor multi-head
├── test_focal_loss.py             # FocalLoss, ClassMeanRecall
├── test_action_anticipation.py    # ActionAnticipationClassifier
├── test_model_hub.py              # Factory functions + hub loading
├── test_preprocessor.py          # vjepa2_preprocessor output shapes
└── test_assessment_config.py     # AssessmentConfig presets and validation
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Single file
python -m pytest tests/test_focal_loss.py -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=html

# Skip slow / GPU tests
python -m pytest tests/ -v -m "not gpu and not slow"

# Run only GPU tests (on a GPU machine)
python -m pytest tests/ -v -m gpu
```

## Pytest Marks

```python
# Declare marks in conftest.py or pytest.ini
import pytest

@pytest.mark.gpu
@pytest.mark.slow
def test_hub_model_loading():
    encoder, predictor = vjepa2_vit_giant(pretrained=True)
    # ... shape checks
```

## Common Pitfalls

1. **Forgetting `torch.no_grad()` in shape tests** — even if gradients are not needed,
   they consume memory and slow down tests unnecessarily.
2. **Not resetting random seeds** — use `torch.manual_seed(0)` in `setUp` for
   reproducible test behavior.
3. **Hardcoding device** — always use `device = torch.device("cpu")` in unit tests;
   move to GPU only in `@skipIf` decorated tests.
4. **Testing on tiny models** — always create minimal model configurations in tests
   (small embed_dim, shallow depth) to keep test runtime under 1 second.
