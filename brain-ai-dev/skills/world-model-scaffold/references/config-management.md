# Config Management — World Model Scaffold

This document describes the YAML configuration structure, Hydra integration strategy,
config merging, environment variable substitution, and validation rules for the world model
scaffold.

---

## Overview

Configuration is split into four files, each addressing a distinct concern. This separation
keeps files small, makes overrides surgical, and allows CI jobs to swap only the hardware
config when moving between local and cluster environments.

| File | Concern |
|------|---------|
| `configs/model.yaml` | Architecture choices and component hyperparameters |
| `configs/training.yaml` | Optimizer, scheduler, loss weights, gradient clipping |
| `configs/dataset.yaml` | Data sources, preprocessing, sequence format |
| `configs/hardware.yaml` | Device, precision, distributed strategy, compilation |

---

## File: model.yaml

The model config specifies which component implementations to use and their hyperparameters.
The `type` field maps to a registry key; `params` are passed as kwargs to the constructor.

```yaml
encoder:
  type: vit
  params:
    embed_dim: 512
    image_size: 64
    patch_size: 8
    num_heads: 8
    depth: 6
    dropout: 0.1

dynamics:
  type: rssm
  params:
    state_dim: 512
    action_dim: 6
    hidden_dim: 256
    num_layers: 2
    cell_type: gru

decoder:
  type: conv
  params:
    input_dim: 512
    output_shape: [3, 64, 64]
    hidden_channels: [256, 128, 64]

memory:
  enabled: false
  type: episodic
  params:
    capacity: 1000
    key_dim: 512
    value_dim: 512

planner:
  enabled: false
  type: cem
  params:
    num_samples: 1000
    num_elites: 100
    num_iterations: 5
    action_dim: 6
    horizon: 12
```

**Required fields:** `encoder.type`, `encoder.params`, `dynamics.type`, `dynamics.params`,
`decoder.type`, `decoder.params`.

**Optional fields:** `memory` (default disabled), `planner` (default disabled).

---

## File: training.yaml

```yaml
optimizer:
  type: adamw
  lr: 3.0e-4
  weight_decay: 1.0e-5
  betas: [0.9, 0.999]
  eps: 1.0e-8

scheduler:
  type: cosine
  warmup_steps: 1000
  total_steps: 100000
  min_lr: 1.0e-6

loss:
  recon_weight: 1.0
  kl_weight: 0.1
  reward_weight: 1.0
  cont_weight: 0.1

training:
  epochs: 100
  batch_size: 32
  grad_clip: 100.0
  log_every: 100
  validate_every: 1000
  save_every: 5000
  seed: 42

checkpoint:
  save_dir: checkpoints/
  keep_last: 5
  resume_from: null
```

**Required fields:** `optimizer.type`, `optimizer.lr`, `training.epochs`, `training.batch_size`.

**Allowed optimizer types:** `adam`, `adamw`, `sgd`, `rmsprop`.

**Allowed scheduler types:** `cosine`, `linear`, `constant`, `one_cycle`.

---

## File: dataset.yaml

```yaml
data:
  root: /data/world_model_dataset
  format: hdf5

splits:
  train: train/
  val: val/
  test: test/

sequence:
  length: 16
  frame_skip: 1
  image_size: [64, 64]
  channels: 3

preprocessing:
  normalize: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  random_crop: false
  horizontal_flip: false
  color_jitter:
    enabled: false
    brightness: 0.4
    contrast: 0.4
    saturation: 0.4

loader:
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
  persistent_workers: true
```

**Required fields:** `data.root`, `splits.train`, `splits.val`, `sequence.length`.

**Environment variable substitution:** `data.root` supports `${DATA_ROOT}` syntax (see below).

---

## File: hardware.yaml

```yaml
device: cuda

precision:
  dtype: float32
  amp: false
  amp_dtype: float16

distributed:
  strategy: none
  num_gpus: 1
  backend: nccl
  find_unused_parameters: false

compile:
  enabled: false
  mode: default
  fullgraph: false
  dynamic: false

memory:
  gradient_checkpointing: false
  pin_memory: true
  empty_cache_every: 0

performance:
  cudnn_benchmark: true
  tf32: true
  deterministic: false
```

**Allowed device values:** `cpu`, `cuda`, `cuda:0`, `cuda:1`, etc.

**Allowed precision.dtype values:** `float32`, `float16`, `bfloat16`.

**Allowed distributed.strategy values:** `none`, `ddp`, `fsdp`, `deepspeed`.

**Allowed compile.mode values:** `default`, `reduce-overhead`, `max-autotune`.

---

## Hydra Integration

The scaffold generates flat YAML files that are Hydra-compatible but do not require Hydra.
To integrate with Hydra:

### Directory Structure for Hydra

```
configs/
├── config.yaml          # Root config listing defaults
├── model/
│   ├── vit_rssm.yaml    # ViT + RSSM configuration
│   └── mamba_tf.yaml    # Mamba + Transformer configuration
├── training/
│   ├── default.yaml
│   └── fast.yaml        # Reduced epochs for debugging
├── dataset/
│   ├── atari.yaml
│   └── dmcontrol.yaml
└── hardware/
    ├── local.yaml
    └── cluster.yaml
```

### Root Config

```yaml
# configs/config.yaml
defaults:
  - model: vit_rssm
  - training: default
  - dataset: atari
  - hardware: local
  - _self_
```

### Hydra Override Examples

```bash
# Change encoder type via CLI
python train.py model.encoder.type=mamba

# Run with cluster hardware config
python train.py hardware=cluster

# Override learning rate
python train.py training.optimizer.lr=1e-3

# Multi-run sweep
python train.py -m training.optimizer.lr=1e-3,3e-4,1e-4
```

### Structured Config (Optional)

For strong typing, define dataclasses that Hydra can validate:

```python
from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import MISSING

@dataclass
class EncoderConfig:
    type: str = MISSING
    embed_dim: int = 512

@dataclass
class ModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
```

---

## Config Merging

To merge a base config with experiment-specific overrides without Hydra:

```python
import yaml
from typing import Any, Dict

def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base. override wins on conflicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(base_path: str, override_path: Optional[str] = None) -> Dict[str, Any]:
    with open(base_path) as f:
        config = yaml.safe_load(f)
    if override_path is not None:
        with open(override_path) as f:
            override = yaml.safe_load(f)
        config = deep_merge(config, override)
    return config
```

Usage:

```python
config = load_config("configs/model.yaml", "experiments/mamba_exp/model.yaml")
```

---

## Environment Variable Substitution

The scaffold loader supports `${VAR_NAME}` syntax in YAML string values:

```yaml
data:
  root: ${DATA_ROOT}/atari
```

Loading with substitution:

```python
import os
import re
import yaml

def substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Replace ${VAR_NAME} with os.environ[VAR_NAME] in all string values."""
    def _sub(value):
        if isinstance(value, str):
            def replace(match):
                var = match.group(1)
                if var not in os.environ:
                    raise KeyError(f"Environment variable '{var}' not set")
                return os.environ[var]
            return re.sub(r'\$\{(\w+)\}', replace, value)
        if isinstance(value, dict):
            return {k: _sub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sub(v) for v in value]
        return value
    return _sub(config)
```

---

## Config Validation

Validate configs immediately after loading to catch errors before any model construction:

```python
REQUIRED_MODEL_KEYS = {
    "encoder": ["type"],
    "dynamics": ["type"],
    "decoder": ["type"],
}

REQUIRED_TRAINING_KEYS = {
    "optimizer": ["type", "lr"],
    "training": ["epochs", "batch_size"],
}

def validate_model_config(config: Dict[str, Any]) -> None:
    for section, keys in REQUIRED_MODEL_KEYS.items():
        if section not in config:
            raise KeyError(f"model.yaml missing required section: '{section}'")
        for key in keys:
            if key not in config[section]:
                raise KeyError(f"model.yaml[{section}] missing required key: '{key}'")

def validate_cross_config(model_cfg: Dict, dynamics_cfg: Dict) -> None:
    """Check that declared dimensions are internally consistent."""
    enc_dim = model_cfg["encoder"]["params"].get("embed_dim")
    dyn_dim = model_cfg["dynamics"]["params"].get("state_dim")
    if enc_dim is not None and dyn_dim is not None and enc_dim != dyn_dim:
        raise ValueError(
            f"Config conflict: encoder.embed_dim={enc_dim} != dynamics.state_dim={dyn_dim}"
        )
```

---

## Loading All Configs

```python
def load_all_configs(config_dir: str) -> Dict[str, Any]:
    configs = {}
    for name in ["model", "training", "dataset", "hardware"]:
        path = os.path.join(config_dir, f"{name}.yaml")
        with open(path) as f:
            configs[name] = yaml.safe_load(f)
    configs = substitute_env_vars(configs)
    validate_model_config(configs["model"])
    return configs
```
