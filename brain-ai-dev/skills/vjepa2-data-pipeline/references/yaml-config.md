# YAML Configuration Reference

## Standard YAML Sections

V-JEPA 2 uses a flat YAML file with top-level sections. Each section is loaded
as a plain Python dict; all access uses `dict.get("key", default)` to ensure
safe defaults when keys are absent.

```yaml
# ============================================================
# V-JEPA 2 Training Configuration
# ============================================================

app:
  name: vjepa2-pretrain
  log_dir: ./logs
  checkpoint_dir: ./checkpoints
  seed: 42
  debug: false

meta:
  read_checkpoint: null       # Resume path, or null
  load_checkpoint: null
  log_freq: 100
  checkpoint_freq: 1000
  num_epochs: 100
  use_amp: true               # Automatic Mixed Precision

mask:
  patch_size: 16
  pred_mask_scale: [0.15, 0.4]
  enc_mask_scale:  [0.85, 1.0]
  aspect_ratio: [0.75, 1.5]
  num_enc_masks: 1
  num_pred_masks: 4
  allow_overlap: false

model:
  model_name: vit_large_patch16_224
  pred_depth: 12
  pred_embed_dim: 384
  uniform_power: true
  use_mask_tokens: false
  zero_init_mask_tokens: true

data:
  data_paths:
    - /data/kinetics700/train_index.txt
    - /data/howto100m/train_index.txt
  data_weights:
    - 1.0
    - 2.0
  clip_mode: fps              # fps | duration | frame_step
  frames_per_clip: 16
  target_fps: 10
  clip_duration_sec: 3.2      # used when clip_mode=duration
  frame_step: 4               # used when clip_mode=frame_step
  img_size: 224
  num_workers: 8
  batch_size: 64
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  use_gpu_decode: false       # decord GPU decoding

data_aug:
  crop_scale: [0.3, 1.0]
  crop_ratio: [0.75, 1.33]
  horizontal_flip: true
  auto_augment: false
  rand_augment_n: 2
  rand_augment_m: 9
  motion_shift: true
  random_erasing: 0.0
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std:  [0.229, 0.224, 0.225]

loss:
  loss_exp: 1.0               # Exponent for VICReg/JEPA loss
  reg_coeff: 0.0              # Regularization coefficient

optimization:
  wd: 0.04
  final_wd: 0.4
  num_epochs: 100
  warmup: 40
  start_lr: 0.0002
  ref_lr: 0.001
  final_lr: 1.0e-06
  ema: [0.998, 1.0]          # Exponential moving average [start, end]
```

---

## dict.get Pattern

All configuration loading MUST use `dict.get("key", default)` to allow
partial configuration overrides:

```python
# CORRECT: safe access with fallback
clip_mode  = data_cfg.get("clip_mode",  "fps")
batch_size = data_cfg.get("batch_size", 64)
use_amp    = meta_cfg.get("use_amp",    True)

# WRONG: direct access raises KeyError on missing keys
clip_mode  = data_cfg["clip_mode"]   # Do NOT do this
```

### Loading Utility

```python
import yaml
from typing import Any, Dict, Optional

def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config file and return as nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def get_section(config: Dict, section: str) -> Dict:
    """Safely extract a top-level section, returning empty dict if missing."""
    return config.get(section, {}) or {}
```

---

## Progressive Training Configurations

V-JEPA 2 uses three training phases, each with its own YAML config file.
Later phases override specific settings while inheriting defaults.

### Phase 1: Pretraining

```yaml
# configs/pretrain.yaml
app:
  name: vjepa2-pretrain

meta:
  num_epochs: 100
  use_amp: true

data:
  clip_mode: fps
  frames_per_clip: 16
  target_fps: 10
  img_size: 224
  batch_size: 64

data_aug:
  crop_scale: [0.3, 1.0]
  auto_augment: false
  motion_shift: true

optimization:
  ref_lr: 0.001
  warmup: 40
  wd: 0.04
```

### Phase 2: Cooldown

```yaml
# configs/cooldown.yaml
app:
  name: vjepa2-cooldown

meta:
  num_epochs: 10
  read_checkpoint: ./checkpoints/pretrain_epoch100.pth

data:
  clip_mode: fps
  frames_per_clip: 16
  target_fps: 10
  img_size: 224
  batch_size: 32        # Smaller batch during cooldown

data_aug:
  crop_scale: [0.5, 1.0]   # Less aggressive crops
  auto_augment: false
  motion_shift: false       # Disable motion shift in cooldown

optimization:
  ref_lr: 0.0001        # Lower LR
  warmup: 0             # No warmup during cooldown
  final_lr: 1.0e-07
  wd: 0.04
```

### Phase 3: Post-Training (Fine-tuning)

```yaml
# configs/post_train.yaml
app:
  name: vjepa2-finetune

meta:
  num_epochs: 20
  read_checkpoint: ./checkpoints/cooldown_final.pth

data:
  data_paths:
    - /data/downstream_task/train_index.txt
  clip_mode: duration
  clip_duration_sec: 3.2
  frames_per_clip: 32   # More frames for fine-tuning
  img_size: 224
  batch_size: 16

data_aug:
  crop_scale: [0.5, 1.0]
  horizontal_flip: true
  auto_augment: true    # Enable RandAugment for fine-tuning
  rand_augment_n: 2
  rand_augment_m: 9
  random_erasing: 0.25  # Enable random erasing

optimization:
  ref_lr: 0.0001
  warmup: 5
  final_lr: 0.0
  wd: 0.01
```

---

## Config Merging

Override base config with task-specific overrides:

```python
def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Deep merge: override values take precedence over base values.
    Handles nested dicts recursively.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (key in result and
            isinstance(result[key], dict) and
            isinstance(value, dict)):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result

# Usage
base   = load_yaml("configs/pretrain.yaml")
overrd = load_yaml("configs/my_experiment.yaml")
config = merge_configs(base, overrd)
```

---

## Dataclass Construction from YAML

```python
@dataclass
class DataConfig:
    data_paths: List[str] = field(default_factory=list)
    data_weights: List[float] = field(default_factory=list)
    clip_mode: str = "fps"
    frames_per_clip: int = 16
    target_fps: int = 10
    clip_duration_sec: float = 3.2
    frame_step: int = 4
    img_size: int = 224
    num_workers: int = 8
    batch_size: int = 64
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    use_gpu_decode: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataConfig":
        """Construct DataConfig from a YAML section dict."""
        return cls(
            data_paths        = d.get("data_paths",        []),
            data_weights      = d.get("data_weights",      []),
            clip_mode         = d.get("clip_mode",         "fps"),
            frames_per_clip   = d.get("frames_per_clip",   16),
            target_fps        = d.get("target_fps",        10),
            clip_duration_sec = d.get("clip_duration_sec", 3.2),
            frame_step        = d.get("frame_step",        4),
            img_size          = d.get("img_size",          224),
            num_workers       = d.get("num_workers",       8),
            batch_size        = d.get("batch_size",        64),
            pin_memory        = d.get("pin_memory",        True),
            persistent_workers= d.get("persistent_workers",True),
            prefetch_factor   = d.get("prefetch_factor",   2),
            use_gpu_decode    = d.get("use_gpu_decode",    False),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DataConfig":
        config = load_yaml(yaml_path)
        return cls.from_dict(get_section(config, "data"))
```

---

## Validation Rules

Before training starts, validate config consistency:

```python
def validate_config(cfg: Dict) -> List[str]:
    """
    Return list of validation errors (empty if valid).
    """
    errors = []
    data = cfg.get("data", {})

    # Check clip_mode
    valid_modes = {"fps", "duration", "frame_step"}
    mode = data.get("clip_mode", "fps")
    if mode not in valid_modes:
        errors.append(f"clip_mode must be one of {valid_modes}, got {mode!r}")

    # Check weights match paths
    paths   = data.get("data_paths",   [])
    weights = data.get("data_weights", [])
    if weights and len(weights) != len(paths):
        errors.append(
            f"data_weights length ({len(weights)}) must match "
            f"data_paths length ({len(paths)})"
        )

    # Check positive batch size
    bs = data.get("batch_size", 64)
    if bs <= 0:
        errors.append(f"batch_size must be positive, got {bs}")

    # Check FPC
    fpc = data.get("frames_per_clip", 16)
    if fpc <= 0:
        errors.append(f"frames_per_clip must be positive, got {fpc}")

    return errors
```
