"""
data_config_template.py
========================
DataConfig and AugConfig dataclasses with YAML parsing utilities.

Public API
----------
DataConfig         -- data loading configuration
AugConfig          -- augmentation configuration
load_yaml(path)    -- load YAML file to dict
get_section(cfg, section) -- safely extract a top-level section
merge_configs(base, override) -- deep merge two config dicts
validate_config(cfg) -- validate config dict, returns list of error strings

Usage
-----
# From YAML file
cfg = load_yaml("configs/pretrain.yaml")
data_cfg = DataConfig.from_dict(get_section(cfg, "data"))
aug_cfg  = AugConfig.from_dict(get_section(cfg, "data_aug"))

# From scratch
data_cfg = DataConfig(batch_size=64, frames_per_clip=16)
aug_cfg  = AugConfig(auto_augment=True, random_erasing=0.25)

# YAML round-trip
d = data_cfg.to_dict()
data_cfg2 = DataConfig.from_dict(d)
assert data_cfg == data_cfg2
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# YAML support (optional: gracefully handle missing pyyaml)
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    yaml = None


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """
    Configuration for video data loading and dataset construction.

    All fields have safe defaults so partial YAML configs are valid.

    Attributes
    ----------
    data_paths : List[str]
        Paths to video files, index .txt files, or directories.
        Multiple paths enable multi-source mixing.
    data_weights : List[float]
        Per-source mixing weights. Length must match data_paths if non-empty.
        Defaults to uniform weights when empty.
    clip_mode : str
        Frame sampling mode: "fps" | "duration" | "frame_step".
    frames_per_clip : int
        Number of frames T in each clip.
    target_fps : int
        Target frames per second (clip_mode="fps" only).
    clip_duration_sec : float
        Clip duration in seconds (clip_mode="duration" only).
    frame_step : int
        Frame stride (clip_mode="frame_step" only).
    img_size : int
        Target spatial size (H=W=img_size).
    num_workers : int
        DataLoader worker processes. 0 = main process only.
    batch_size : int
        Training batch size per GPU.
    pin_memory : bool
        Pin DataLoader output to CUDA pinned memory.
    persistent_workers : bool
        Keep worker processes alive between epochs.
    prefetch_factor : int
        Number of batches prefetched per worker.
    use_gpu_decode : bool
        Use decord GPU decoding (requires CUDA decord build).
    """

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

    VALID_CLIP_MODES = frozenset({"fps", "duration", "frame_step"})

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataConfig":
        """
        Construct DataConfig from a dict using safe .get() access.

        All keys are optional; missing keys receive class defaults.

        Parameters
        ----------
        d : Dict[str, Any]
            Dictionary (typically from a YAML section).
        """
        return cls(
            data_paths        = list(d.get("data_paths",         [])),
            data_weights      = list(d.get("data_weights",       [])),
            clip_mode         = str(d.get("clip_mode",           "fps")),
            frames_per_clip   = int(d.get("frames_per_clip",     16)),
            target_fps        = int(d.get("target_fps",          10)),
            clip_duration_sec = float(d.get("clip_duration_sec", 3.2)),
            frame_step        = int(d.get("frame_step",          4)),
            img_size          = int(d.get("img_size",            224)),
            num_workers       = int(d.get("num_workers",         8)),
            batch_size        = int(d.get("batch_size",          64)),
            pin_memory        = bool(d.get("pin_memory",         True)),
            persistent_workers= bool(d.get("persistent_workers", True)),
            prefetch_factor   = int(d.get("prefetch_factor",     2)),
            use_gpu_decode    = bool(d.get("use_gpu_decode",     False)),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str, section: str = "data") -> "DataConfig":
        """Load DataConfig from a YAML file's ``data`` section."""
        raw = load_yaml(yaml_path)
        return cls.from_dict(get_section(raw, section))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict suitable for YAML serialization."""
        return {
            "data_paths":         list(self.data_paths),
            "data_weights":       list(self.data_weights),
            "clip_mode":          self.clip_mode,
            "frames_per_clip":    self.frames_per_clip,
            "target_fps":         self.target_fps,
            "clip_duration_sec":  self.clip_duration_sec,
            "frame_step":         self.frame_step,
            "img_size":           self.img_size,
            "num_workers":        self.num_workers,
            "batch_size":         self.batch_size,
            "pin_memory":         self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor":    self.prefetch_factor,
            "use_gpu_decode":     self.use_gpu_decode,
        }

    def to_yaml_str(self) -> str:
        """Serialize to YAML string."""
        if not _YAML_AVAILABLE:
            raise ImportError("pyyaml is required for YAML serialization")
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Return list of validation error strings (empty if valid)."""
        errors = []
        if self.clip_mode not in self.VALID_CLIP_MODES:
            errors.append(
                f"clip_mode={self.clip_mode!r} is invalid. "
                f"Must be one of {sorted(self.VALID_CLIP_MODES)}."
            )
        if self.data_weights and len(self.data_weights) != len(self.data_paths):
            errors.append(
                f"data_weights length ({len(self.data_weights)}) must match "
                f"data_paths length ({len(self.data_paths)})."
            )
        if self.batch_size <= 0:
            errors.append(f"batch_size must be > 0, got {self.batch_size}.")
        if self.frames_per_clip <= 0:
            errors.append(f"frames_per_clip must be > 0, got {self.frames_per_clip}.")
        if self.img_size <= 0:
            errors.append(f"img_size must be > 0, got {self.img_size}.")
        if self.num_workers < 0:
            errors.append(f"num_workers must be >= 0, got {self.num_workers}.")
        if self.target_fps <= 0:
            errors.append(f"target_fps must be > 0, got {self.target_fps}.")
        if self.clip_duration_sec <= 0:
            errors.append(f"clip_duration_sec must be > 0, got {self.clip_duration_sec}.")
        if self.frame_step <= 0:
            errors.append(f"frame_step must be > 0, got {self.frame_step}.")
        return errors

    def assert_valid(self) -> None:
        """Raise ValueError if configuration is invalid."""
        validation_errors = self.validate()
        if validation_errors:
            raise ValueError(
                "DataConfig validation errors:\n" +
                "\n".join(f"  - {e}" for e in validation_errors)
            )

    def __post_init__(self):
        # Convert tuples to lists for consistency
        if isinstance(self.data_paths, tuple):
            self.data_paths = list(self.data_paths)
        if isinstance(self.data_weights, tuple):
            self.data_weights = list(self.data_weights)


# ---------------------------------------------------------------------------
# AugConfig
# ---------------------------------------------------------------------------

@dataclass
class AugConfig:
    """
    Configuration for video augmentation pipeline.

    All fields have safe defaults matching V-JEPA 2 pretraining settings.

    Attributes
    ----------
    crop_scale : Tuple[float, float]
        (min, max) fraction of image area for RandomResizedCrop.
    crop_ratio : Tuple[float, float]
        (min, max) aspect ratio for RandomResizedCrop.
    horizontal_flip : bool
        Enable random horizontal flip (disable for robotics).
    auto_augment : bool
        Enable RandAugment per-frame augmentations.
    rand_augment_n : int
        Number of RandAugment operations per frame.
    rand_augment_m : int
        Magnitude of RandAugment operations (0-30).
    motion_shift : bool
        Enable temporal motion shift across frames.
    random_erasing : float
        Probability of cube-mode random erasing (0.0 = disabled).
    normalize_mean : Tuple[float, float, float]
        Per-channel normalization mean (ImageNet defaults).
    normalize_std : Tuple[float, float, float]
        Per-channel normalization std (ImageNet defaults).
    """

    crop_scale: Tuple[float, float] = (0.3, 1.0)
    crop_ratio: Tuple[float, float] = (0.75, 1.33)
    horizontal_flip: bool = True
    auto_augment: bool = False
    rand_augment_n: int = 2
    rand_augment_m: int = 9
    motion_shift: bool = True
    random_erasing: float = 0.0
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float]  = (0.229, 0.224, 0.225)

    # ------------------------------------------------------------------
    # Preset configurations
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "AugConfig":
        """Standard V-JEPA 2 pretraining augmentation."""
        return cls()

    @classmethod
    def robotics(cls) -> "AugConfig":
        """
        Robotics-specific augmentation.
        - No horizontal flip (direction-sensitive)
        - Near-fixed scale (minimal spatial jitter)
        - No RandAugment or motion shift
        """
        return cls(
            crop_scale=(0.9, 1.0),
            crop_ratio=(1.0, 1.0),
            horizontal_flip=False,
            auto_augment=False,
            motion_shift=False,
            random_erasing=0.0,
        )

    @classmethod
    def strong(cls) -> "AugConfig":
        """Strong augmentation for downstream fine-tuning."""
        return cls(
            crop_scale=(0.2, 1.0),
            horizontal_flip=True,
            auto_augment=True,
            rand_augment_n=2,
            rand_augment_m=12,
            motion_shift=True,
            random_erasing=0.25,
        )

    @classmethod
    def minimal(cls) -> "AugConfig":
        """Minimal augmentation for deterministic evaluation."""
        return cls(
            crop_scale=(1.0, 1.0),
            horizontal_flip=False,
            auto_augment=False,
            motion_shift=False,
            random_erasing=0.0,
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AugConfig":
        """
        Construct AugConfig from a dict using safe .get() access.

        Handles both list and tuple input for crop_scale, crop_ratio, etc.
        """
        def _to_tuple2(key, default):
            val = d.get(key, default)
            return tuple(val) if not isinstance(val, tuple) else val

        def _to_tuple3(key, default):
            val = d.get(key, default)
            return tuple(val) if not isinstance(val, tuple) else val

        return cls(
            crop_scale      = _to_tuple2("crop_scale",      (0.3, 1.0)),
            crop_ratio      = _to_tuple2("crop_ratio",      (0.75, 1.33)),
            horizontal_flip = bool(d.get("horizontal_flip", True)),
            auto_augment    = bool(d.get("auto_augment",    False)),
            rand_augment_n  = int(d.get("rand_augment_n",  2)),
            rand_augment_m  = int(d.get("rand_augment_m",  9)),
            motion_shift    = bool(d.get("motion_shift",    True)),
            random_erasing  = float(d.get("random_erasing", 0.0)),
            normalize_mean  = _to_tuple3("normalize_mean", (0.485, 0.456, 0.406)),
            normalize_std   = _to_tuple3("normalize_std",  (0.229, 0.224, 0.225)),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str, section: str = "data_aug") -> "AugConfig":
        """Load AugConfig from a YAML file's ``data_aug`` section."""
        raw = load_yaml(yaml_path)
        return cls.from_dict(get_section(raw, section))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict (tuples -> lists for YAML)."""
        return {
            "crop_scale":       list(self.crop_scale),
            "crop_ratio":       list(self.crop_ratio),
            "horizontal_flip":  self.horizontal_flip,
            "auto_augment":     self.auto_augment,
            "rand_augment_n":   self.rand_augment_n,
            "rand_augment_m":   self.rand_augment_m,
            "motion_shift":     self.motion_shift,
            "random_erasing":   self.random_erasing,
            "normalize_mean":   list(self.normalize_mean),
            "normalize_std":    list(self.normalize_std),
        }

    def to_yaml_str(self) -> str:
        if not _YAML_AVAILABLE:
            raise ImportError("pyyaml is required for YAML serialization")
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        aug_errors = []
        if not (0 < self.crop_scale[0] <= self.crop_scale[1] <= 1.0):
            aug_errors.append(
                f"crop_scale must satisfy 0 < min <= max <= 1.0, got {self.crop_scale}"
            )
        if not (0 < self.crop_ratio[0] <= self.crop_ratio[1]):
            aug_errors.append(
                f"crop_ratio must satisfy 0 < min <= max, got {self.crop_ratio}"
            )
        if not (0.0 <= self.random_erasing <= 1.0):
            aug_errors.append(
                f"random_erasing must be in [0,1], got {self.random_erasing}"
            )
        if self.rand_augment_n < 0:
            aug_errors.append(f"rand_augment_n must be >= 0, got {self.rand_augment_n}")
        if not (0 <= self.rand_augment_m <= 30):
            aug_errors.append(f"rand_augment_m must be in [0,30], got {self.rand_augment_m}")
        return aug_errors


# ---------------------------------------------------------------------------
# YAML utilities
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Absolute or relative path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dict. Returns empty dict for empty files.

    Raises
    ------
    ImportError
        If pyyaml is not installed.
    FileNotFoundError
        If the path does not exist.
    """
    if not _YAML_AVAILABLE:
        raise ImportError(
            "pyyaml is required for YAML loading. "
            "Install with: pip install pyyaml"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        result = yaml.safe_load(f)
    return result or {}


def load_yaml_string(yaml_str: str) -> Dict[str, Any]:
    """
    Load YAML from a string (useful in tests and CI).

    Parameters
    ----------
    yaml_str : str
        Raw YAML content.
    """
    if not _YAML_AVAILABLE:
        raise ImportError("pyyaml is required")
    result = yaml.safe_load(yaml_str)
    return result or {}


def get_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Safely extract a top-level section from a config dict.

    Returns an empty dict if the section is missing or null,
    instead of raising KeyError.

    Parameters
    ----------
    config : Dict
        Top-level configuration dict.
    section : str
        Section name (e.g., "data", "data_aug", "meta").

    Returns
    -------
    Dict[str, Any]
        Section dict, or empty dict if not present.
    """
    value = config.get(section, {})
    return value if isinstance(value, dict) else {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge two configuration dicts. Values in ``override`` take precedence.

    Nested dicts are merged recursively. Non-dict values are replaced.

    Parameters
    ----------
    base : Dict
        Base configuration (lower priority).
    override : Dict
        Override configuration (higher priority).

    Returns
    -------
    Dict[str, Any]
        Merged configuration dict.

    Examples
    --------
    >>> base = {"data": {"batch_size": 64, "img_size": 224}}
    >>> override = {"data": {"batch_size": 32}}
    >>> merge_configs(base, override)
    {'data': {'batch_size': 32, 'img_size': 224}}
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_config(cfg: Dict[str, Any]) -> List[str]:
    """
    Validate a raw config dict (from YAML load) and return a list of errors.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Top-level config dict with sections "data", "data_aug", etc.

    Returns
    -------
    List[str]
        Error messages. Empty list means the config is valid.
    """
    found_errors: List[str] = []

    data_sect = get_section(cfg, "data")
    aug_sect  = get_section(cfg, "data_aug")

    # Validate DataConfig
    try:
        data_cfg = DataConfig.from_dict(data_sect)
        found_errors.extend(data_cfg.validate())
    except (TypeError, ValueError) as exc:
        found_errors.append(f"DataConfig construction error: {exc}")

    # Validate AugConfig
    try:
        aug_cfg = AugConfig.from_dict(aug_sect)
        found_errors.extend(aug_cfg.validate())
    except (TypeError, ValueError) as exc:
        found_errors.append(f"AugConfig construction error: {exc}")

    # Cross-section validation
    meta_sect = get_section(cfg, "meta")
    opt_sect  = get_section(cfg, "optimization")

    num_epochs_meta = meta_sect.get("num_epochs", None)
    num_epochs_opt  = opt_sect.get("num_epochs", None)
    if (num_epochs_meta is not None and num_epochs_opt is not None and
            num_epochs_meta != num_epochs_opt):
        found_errors.append(
            f"num_epochs mismatch: meta.num_epochs={num_epochs_meta} vs "
            f"optimization.num_epochs={num_epochs_opt}. Use one or the other."
        )

    return found_errors


# ---------------------------------------------------------------------------
# Convenience: standard YAML config template as a string
# ---------------------------------------------------------------------------

PRETRAIN_CONFIG_TEMPLATE = """\
app:
  name: vjepa2-pretrain
  log_dir: ./logs
  checkpoint_dir: ./checkpoints
  seed: 42
  debug: false

meta:
  read_checkpoint: null
  log_freq: 100
  checkpoint_freq: 1000
  num_epochs: 100
  use_amp: true

mask:
  patch_size: 16
  pred_mask_scale: [0.15, 0.4]
  enc_mask_scale: [0.85, 1.0]
  aspect_ratio: [0.75, 1.5]
  num_enc_masks: 1
  num_pred_masks: 4
  allow_overlap: false

model:
  model_name: vit_large_patch16_224
  pred_depth: 12
  pred_embed_dim: 384

data:
  data_paths: []
  data_weights: []
  clip_mode: fps
  frames_per_clip: 16
  target_fps: 10
  clip_duration_sec: 3.2
  frame_step: 4
  img_size: 224
  num_workers: 8
  batch_size: 64
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  use_gpu_decode: false

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
  normalize_std: [0.229, 0.224, 0.225]

loss:
  loss_exp: 1.0
  reg_coeff: 0.0

optimization:
  wd: 0.04
  final_wd: 0.4
  num_epochs: 100
  warmup: 40
  start_lr: 0.0002
  ref_lr: 0.001
  final_lr: 1.0e-06
  ema: [0.998, 1.0]
"""


def write_template_config(output_path: str) -> None:
    """
    Write the standard pretrain config template to a file.

    Parameters
    ----------
    output_path : str
        Destination path for the YAML file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(PRETRAIN_CONFIG_TEMPLATE)
    print(f"Wrote template config to {output_path}")


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("DataConfig & AugConfig Self-Tests")
    print("=" * 60)

    PASS = "[PASS]"
    FAIL = "[FAIL]"
    test_errors = []

    # -------------------------------------------------------------------
    # Test 1: DataConfig defaults
    # -------------------------------------------------------------------
    dc = DataConfig()
    if dc.clip_mode == "fps" and dc.frames_per_clip == 16 and dc.img_size == 224:
        print(f"{PASS} Test 1: DataConfig defaults correct")
    else:
        msg = f"Test 1 FAILED: {dc}"
        print(f"{FAIL} {msg}")
        test_errors.append(msg)

    # -------------------------------------------------------------------
    # Test 2: AugConfig defaults
    # -------------------------------------------------------------------
    ac = AugConfig()
    if (ac.crop_scale == (0.3, 1.0) and
            ac.horizontal_flip is True and
            ac.auto_augment is False and
            ac.random_erasing == 0.0 and
            ac.normalize_mean == (0.485, 0.456, 0.406)):
        print(f"{PASS} Test 2: AugConfig defaults correct")
    else:
        msg = f"Test 2 FAILED: {ac}"
        print(f"{FAIL} {msg}")
        test_errors.append(msg)

    # -------------------------------------------------------------------
    # Test 3: DataConfig.from_dict with partial dict
    # -------------------------------------------------------------------
    d = {"batch_size": 32, "frames_per_clip": 8, "clip_mode": "duration"}
    dc2 = DataConfig.from_dict(d)
    if dc2.batch_size == 32 and dc2.frames_per_clip == 8 and dc2.clip_mode == "duration":
        print(f"{PASS} Test 3: DataConfig.from_dict with partial dict")
    else:
        msg = f"Test 3 FAILED: {dc2}"
        print(f"{FAIL} {msg}")
        test_errors.append(msg)

    # -------------------------------------------------------------------
    # Test 4: DataConfig round-trip (to_dict -> from_dict)
    # -------------------------------------------------------------------
    dc3 = DataConfig(
        batch_size=32, frames_per_clip=8, clip_mode="frame_step",
        data_paths=["/a", "/b"], data_weights=[1.0, 2.0]
    )
    d3 = dc3.to_dict()
    dc3_rt = DataConfig.from_dict(d3)
    if dc3 == dc3_rt:
        print(f"{PASS} Test 4: DataConfig YAML round-trip")
    else:
        msg = f"Test 4 FAILED: original={dc3}, recovered={dc3_rt}"
        print(f"{FAIL} {msg}")
        test_errors.append(msg)

    # -------------------------------------------------------------------
    # Test 5: AugConfig round-trip
    # -------------------------------------------------------------------
    ac3 = AugConfig(crop_scale=(0.5, 0.9), auto_augment=True, random_erasing=0.25)
    ac3_rt = AugConfig.from_dict(ac3.to_dict())
    if ac3 == ac3_rt:
        print(f"{PASS} Test 5: AugConfig round-trip")
    else:
        msg = f"Test 5 FAILED: original={ac3}, recovered={ac3_rt}"
        print(f"{FAIL} {msg}")
        test_errors.append(msg)

    # -------------------------------------------------------------------
    # Test 6: Config validation -- invalid clip_mode
    # -------------------------------------------------------------------
    dc_bad = DataConfig(clip_mode="bogus")
    errs = dc_bad.validate()
    if errs and any("clip_mode" in e for e in errs):
        print(f"{PASS} Test 6: Invalid clip_mode detected: {errs[0]}")
    else:
        msg = f"Test 6 FAILED: no error for invalid clip_mode, got {errs}"
        print(f"{FAIL} {msg}")
        test_errors.append(msg)

    # -------------------------------------------------------------------
    # Test 7: Config validation -- weight/path mismatch
    # -------------------------------------------------------------------
    dc_wm = DataConfig(data_paths=["/a", "/b"], data_weights=[1.0, 2.0, 3.0])
    errs_wm = dc_wm.validate()
    if errs_wm and any("weight" in e.lower() for e in errs_wm):
        print(f"{PASS} Test 7: Weight/path mismatch detected")
    else:
        msg = f"Test 7 FAILED: {errs_wm}"
        print(f"{FAIL} {msg}")
        test_errors.append(msg)

    # -------------------------------------------------------------------
    # Test 8: Valid config passes validation
    # -------------------------------------------------------------------
    dc_valid = DataConfig(data_paths=["/a"], data_weights=[1.0], clip_mode="fps")
    errs_v = dc_valid.validate()
    if not errs_v:
        print(f"{PASS} Test 8: Valid DataConfig passes validation")
    else:
        msg = f"Test 8 FAILED: {errs_v}"
        print(f"{FAIL} {msg}")
        test_errors.append(msg)

    # -------------------------------------------------------------------
    # Test 9: merge_configs deep merge
    # -------------------------------------------------------------------
    base = {"data": {"batch_size": 64, "img_size": 224}, "meta": {"num_epochs": 100}}
    over = {"data": {"batch_size": 32}}
    merged = merge_configs(base, over)
    if merged["data"]["batch_size"] == 32 and merged["data"]["img_size"] == 224:
        print(f"{PASS} Test 9: merge_configs deep merge correct")
    else:
        msg = f"Test 9 FAILED: {merged}"
        print(f"{FAIL} {msg}")
        test_errors.append(msg)

    # -------------------------------------------------------------------
    # Test 10: validate_config with YAML string
    # -------------------------------------------------------------------
    if _YAML_AVAILABLE:
        raw = load_yaml_string(PRETRAIN_CONFIG_TEMPLATE)
        v_errors = validate_config(raw)
        if not v_errors:
            print(f"{PASS} Test 10: Template config passes validate_config()")
        else:
            msg = f"Test 10 FAILED: {v_errors}"
            print(f"{FAIL} {msg}")
            test_errors.append(msg)
    else:
        print("[SKIP] Test 10: pyyaml not available")

    # -------------------------------------------------------------------
    # Test 11: AugConfig presets
    # -------------------------------------------------------------------
    rob = AugConfig.robotics()
    if not rob.horizontal_flip and rob.crop_scale == (0.9, 1.0):
        print(f"{PASS} Test 11: AugConfig.robotics() preset correct")
    else:
        msg = f"Test 11 FAILED: {rob}"
        print(f"{FAIL} {msg}")
        test_errors.append(msg)

    # -------------------------------------------------------------------
    # Test 12: DataConfig assert_valid raises on invalid
    # -------------------------------------------------------------------
    try:
        DataConfig(batch_size=-1).assert_valid()
        msg = "Test 12 FAILED: should have raised ValueError"
        print(f"{FAIL} {msg}")
        test_errors.append(msg)
    except ValueError:
        print(f"{PASS} Test 12: assert_valid raises ValueError on invalid config")

    # -------------------------------------------------------------------
    # Test 13: get_section returns empty dict for missing section
    # -------------------------------------------------------------------
    empty = get_section({"data": {"a": 1}}, "missing_section")
    if empty == {}:
        print(f"{PASS} Test 13: get_section returns {{}} for missing section")
    else:
        msg = f"Test 13 FAILED: got {empty}"
        print(f"{FAIL} {msg}")
        test_errors.append(msg)

    # -------------------------------------------------------------------
    # Test 14: AugConfig from YAML string
    # -------------------------------------------------------------------
    if _YAML_AVAILABLE:
        yaml_str = (
            "data_aug:\n"
            "  crop_scale: [0.5, 1.0]\n"
            "  horizontal_flip: false\n"
            "  auto_augment: true\n"
            "  random_erasing: 0.25\n"
        )
        raw2 = load_yaml_string(yaml_str)
        ac_loaded = AugConfig.from_dict(get_section(raw2, "data_aug"))
        if (ac_loaded.crop_scale == (0.5, 1.0) and
                not ac_loaded.horizontal_flip and
                ac_loaded.auto_augment and
                ac_loaded.random_erasing == 0.25):
            print(f"{PASS} Test 14: AugConfig from YAML string")
        else:
            msg = f"Test 14 FAILED: {ac_loaded}"
            print(f"{FAIL} {msg}")
            test_errors.append(msg)
    else:
        print("[SKIP] Test 14: pyyaml not available")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print()
    if test_errors:
        print(f"FAILED: {len(test_errors)} test(s) failed:")
        for e in test_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All tests passed.")
