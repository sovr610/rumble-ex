# Model Hub — Reference

## Overview

V-JEPA 2 models are distributed through two channels:
1. **PyTorch Hub** — `torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')`
2. **HuggingFace Hub** — `AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256")`

Both channels provide the same pretrained weights but with different APIs.

## PyTorch Hub

### Entry Points

```python
import torch

# ViT-Large encoder + predictor
encoder, predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')

# ViT-Giant encoder + predictor
encoder, predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')

# ViT-Giant + Action Classifier (anticipation checkpoint)
encoder, predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')
```

### hubconf.py Pattern

The `hubconf.py` in the repo root defines the entry points:

```python
dependencies = ["torch", "timm", "einops"]

def vjepa2_vit_giant(pretrained: bool = True):
    """
    V-JEPA 2 ViT-Giant encoder and predictor.
    Returns (encoder, predictor) tuple.
    """
    encoder  = _build_vit_giant_encoder()
    predictor = _build_default_predictor(embed_dim=1408, depth=12)

    if pretrained:
        url = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_vitg.pt"
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        # strict=False because RoPE positional embeddings may vary in shape
        encoder.load_state_dict(state["encoder"], strict=False)
        predictor.load_state_dict(state["predictor"], strict=False)

    return encoder, predictor
```

### Why strict=False

RoPE (Rotary Position Embedding) buffers are computed lazily based on sequence length
at runtime. The checkpoint may store them for a specific resolution, but the model
should adapt to arbitrary resolutions. Using `strict=False` skips these mismatched keys.

### Model Specifications

| Model         | embed_dim | depth | num_heads | Params |
|---------------|-----------|-------|-----------|--------|
| ViT-Small     | 384       | 12    | 6         | ~22M   |
| ViT-Base      | 768       | 12    | 12        | ~86M   |
| ViT-Large     | 1024      | 24    | 16        | ~307M  |
| ViT-Giant     | 1408      | 40    | 16        | ~1.1B  |

### Default Predictor Architecture

```python
# Predictor is a narrow transformer
predictor_depth    = 12
predictor_embed_dim = 384
predictor_num_heads = 12
num_mask_tokens    = 10   # learnable mask token for future patches
```

## Factory Functions

These factory functions are the canonical pattern for creating models programmatically
(without Hub's network dependency):

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# MIT License

import torch
import torch.nn as nn
from typing import Tuple


def _build_vit_encoder(
    img_size: int = 224,
    patch_size: int = 16,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    tubelet_size: int = 2,
    use_rope: bool = True,
) -> nn.Module:
    """Build a ViT encoder for video understanding."""
    from timm.models.vision_transformer import VisionTransformer
    # Video ViT extends timm's image ViT with temporal dimension
    encoder = VideoVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        tubelet_size=tubelet_size,
    )
    return encoder


def _build_predictor(
    embed_dim: int = 384,
    depth: int = 12,
    num_heads: int = 12,
    context_embed_dim: int = 768,
) -> nn.Module:
    """Build a narrow predictor transformer."""
    return VJEPAPredictor(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        context_embed_dim=context_embed_dim,
    )


def vjepa2_vit_large(pretrained: bool = True) -> Tuple[nn.Module, nn.Module]:
    encoder   = _build_vit_encoder(embed_dim=1024, depth=24, num_heads=16)
    predictor = _build_predictor(embed_dim=384, context_embed_dim=1024)
    if pretrained:
        _load_weights(encoder, predictor, "vjepa2_vitl.pt")
    return encoder, predictor


def vjepa2_vit_giant(pretrained: bool = True) -> Tuple[nn.Module, nn.Module]:
    encoder   = _build_vit_encoder(embed_dim=1408, depth=40, num_heads=16)
    predictor = _build_predictor(embed_dim=384, context_embed_dim=1408)
    if pretrained:
        _load_weights(encoder, predictor, "vjepa2_vitg.pt")
    return encoder, predictor


def vjepa2_ac_vit_giant(pretrained: bool = True) -> Tuple[nn.Module, nn.Module]:
    """Action-classifier checkpoint: predictor used for anticipation."""
    encoder   = _build_vit_encoder(embed_dim=1408, depth=40, num_heads=16)
    predictor = _build_predictor(embed_dim=384, context_embed_dim=1408)
    if pretrained:
        _load_weights(encoder, predictor, "vjepa2_ac_vitg.pt")
    return encoder, predictor


def _load_weights(
    encoder: nn.Module,
    predictor: nn.Module,
    filename: str,
    base_url: str = "https://dl.fbaipublicfiles.com/vjepa2/",
) -> None:
    import torch
    state = torch.hub.load_state_dict_from_url(
        base_url + filename,
        map_location="cpu",
        check_hash=False,
    )
    encoder.load_state_dict(state["encoder"], strict=False)
    predictor.load_state_dict(state["predictor"], strict=False)
```

## HuggingFace Integration

### Loading

```python
from transformers import AutoModel, AutoVideoProcessor

# Model names follow the pattern:
#   facebook/vjepa2-{arch}-fpc{frames_per_clip}-{resolution}
model     = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
```

### Preprocessing Pipeline

```python
import torch
from torchvision import transforms

def vjepa2_preprocessor(
    crop_size: int = 224,
    short_side_size: int = 256,
    frames_per_clip: int = 16,
    mean: tuple = (0.485, 0.456, 0.406),
    std:  tuple = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Standard V-JEPA 2 video preprocessing pipeline.

    Returns a transform that accepts video frames as a tensor [T, C, H, W]
    or a list of PIL images and outputs a normalized tensor [C, T, H, W].
    """
    return transforms.Compose([
        # Temporal: sample `frames_per_clip` frames uniformly
        TemporalSubsample(frames_per_clip),
        # Spatial: resize short side, then center crop
        transforms.Resize(short_side_size, antialias=True),
        transforms.CenterCrop(crop_size),
        # Normalize
        transforms.Normalize(mean=mean, std=std),
    ])
```

### HuggingFace Usage Example

```python
import torch
from transformers import AutoModel, AutoVideoProcessor
from PIL import Image
import numpy as np

processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
model     = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256")

# Load video frames (list of PIL Images or numpy array [T, H, W, C])
frames = [Image.open(f"frame_{i:04d}.jpg") for i in range(16)]

inputs = processor(videos=frames, return_tensors="pt")
# inputs["pixel_values"]: [1, C, T, H, W]

with torch.no_grad():
    outputs = model(**inputs)

# last_hidden_state: [1, N, D] where N = num_patches
features = outputs.last_hidden_state
```

## Dependency Management

```python
# hubconf.py dependencies declaration
dependencies = ["torch", "timm", "einops"]

# Minimum versions
# torch  >= 2.0
# timm   >= 0.9
# einops >= 0.6
```

## Offline / Air-Gapped Deployment

For environments without internet access:

```python
import torch

# Pre-download the checkpoint
# wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_vitg.pt -O /mnt/models/vjepa2_vitg.pt

# Load from local path
state = torch.load("/mnt/models/vjepa2_vitg.pt", map_location="cpu")
encoder.load_state_dict(state["encoder"], strict=False)
predictor.load_state_dict(state["predictor"], strict=False)
```

## Output Shape Reference

Given input `[B, C, T, H, W]`:

| Model     | Patch Size | T frames | H=W | Num patches N | Output shape          |
|-----------|-----------|----------|-----|---------------|-----------------------|
| ViT-L     | 16, 2     | 16       | 224 | 8*14*14=1568  | [B, 1568, 1024]       |
| ViT-G     | 16, 2     | 16       | 224 | 8*14*14=1568  | [B, 1568, 1408]       |
| ViT-G     | 16, 2     | 64       | 256 | 32*16*16=8192 | [B, 8192, 1408]       |

Predictor output shape equals `[B, N_future, predictor_embed_dim]` where
`N_future` = number of masked/future patch positions.
