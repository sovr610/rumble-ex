"""
All configuration dataclasses for the video dataloader pipeline.

Provides validated dataclasses with to_dict() / from_dict() serialization
for VideoDatasetConfig, DataModuleConfig, TFRecordConfig, and TFPipelineConfig.

Design principles:
- Validation in __post_init__ -- fail fast on bad config
- to_dict() / from_dict() for JSON round-trip serialization
- Cross-config compatibility check via check_compatibility()

CRITICAL: Never use bare .eval on nn.Module -- use module.train(False) instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# VideoDatasetConfig
# ---------------------------------------------------------------------------

@dataclass
class VideoDatasetConfig:
    """
    Configuration for VideoDataset frame sampling, augmentation, and normalization.
    """

    num_frames: int = 16
    stride: int = 4
    crop_size: int = 224
    crop_scale: Tuple[float, float] = (0.5, 1.0)
    crop_ratio: Tuple[float, float] = (0.75, 1.333)
    hflip_prob: float = 0.5
    color_jitter: Tuple[float, ...] = (0.4, 0.4, 0.2, 0.1)
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {self.num_frames}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.crop_size < 1:
            raise ValueError(f"crop_size must be >= 1, got {self.crop_size}")
        if self.crop_scale[0] >= self.crop_scale[1]:
            raise ValueError(f"crop_scale must be ascending, got {self.crop_scale}")
        if self.crop_ratio[0] >= self.crop_ratio[1]:
            raise ValueError(f"crop_ratio must be ascending, got {self.crop_ratio}")
        if not (0.0 <= self.hflip_prob <= 1.0):
            raise ValueError(f"hflip_prob must be in [0,1], got {self.hflip_prob}")
        if len(self.color_jitter) != 4:
            raise ValueError(f"color_jitter must have 4 values, got {len(self.color_jitter)}")
        if len(self.normalize_mean) != 3:
            raise ValueError(f"normalize_mean must have 3 values, got {len(self.normalize_mean)}")
        if len(self.normalize_std) != 3:
            raise ValueError(f"normalize_std must have 3 values, got {len(self.normalize_std)}")
        if any(s <= 0.0 for s in self.normalize_std):
            raise ValueError(f"normalize_std values must be > 0, got {self.normalize_std}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert tuples to lists for JSON compatibility
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoDatasetConfig":
        fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in d.items() if k in fields}
        # Convert lists back to tuples where needed
        tuple_fields = {
            "crop_scale", "crop_ratio", "color_jitter", "normalize_mean", "normalize_std"
        }
        for k in tuple_fields:
            if k in filtered and isinstance(filtered[k], list):
                filtered[k] = tuple(filtered[k])
        return cls(**filtered)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "VideoDatasetConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# DataModuleConfig
# ---------------------------------------------------------------------------

@dataclass
class DataModuleConfig:
    """
    Configuration for VideoDataModule (Lightning DataModule wrapper).
    """

    data_root: str = ""
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    batch_size: int = 8
    num_workers_per_gpu: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_workers_per_gpu < 0:
            raise ValueError(
                f"num_workers_per_gpu must be >= 0, got {self.num_workers_per_gpu}"
            )
        if self.prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {self.prefetch_factor}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataModuleConfig":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "DataModuleConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# TFRecordConfig
# ---------------------------------------------------------------------------

@dataclass
class TFRecordConfig:
    """
    Configuration for TFRecordConverter (video -> sharded TFRecord files).
    """

    num_shards: int = 256
    num_frames: int = 16
    stride: int = 4
    crop_size: int = 224
    compression: str = "none"    # "none" | "jpeg"
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        if self.num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {self.num_shards}")
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {self.num_frames}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.crop_size < 1:
            raise ValueError(f"crop_size must be >= 1, got {self.crop_size}")
        if self.compression not in ("none", "jpeg"):
            raise ValueError(
                f"compression must be 'none' or 'jpeg', got '{self.compression}'"
            )
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError(f"jpeg_quality must be 1-100, got {self.jpeg_quality}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TFRecordConfig":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TFRecordConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# TFPipelineConfig
# ---------------------------------------------------------------------------

@dataclass
class TFPipelineConfig:
    """
    Configuration for the tf.data video pipeline (TFRecord reader for TPU training).
    """

    shuffle_buffer: int = 10_000
    num_frames: int = 16
    crop_size: int = 224
    augment: bool = True
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if self.shuffle_buffer < 1:
            raise ValueError(f"shuffle_buffer must be >= 1, got {self.shuffle_buffer}")
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {self.num_frames}")
        if self.crop_size < 1:
            raise ValueError(f"crop_size must be >= 1, got {self.crop_size}")
        if len(self.normalize_mean) != 3:
            raise ValueError(f"normalize_mean must have 3 values, got {len(self.normalize_mean)}")
        if len(self.normalize_std) != 3:
            raise ValueError(f"normalize_std must have 3 values, got {len(self.normalize_std)}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TFPipelineConfig":
        fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in d.items() if k in fields}
        tuple_fields = {"normalize_mean", "normalize_std"}
        for k in tuple_fields:
            if k in filtered and isinstance(filtered[k], list):
                filtered[k] = tuple(filtered[k])
        return cls(**filtered)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TFPipelineConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Cross-config compatibility check
# ---------------------------------------------------------------------------

def check_compatibility(
    dataset_cfg: VideoDatasetConfig,
    tf_cfg: TFPipelineConfig,
) -> None:
    """
    Verify that VideoDatasetConfig and TFPipelineConfig agree on key dimensions.

    Raises ValueError if num_frames or crop_size mismatch.
    This guards against accidentally training PyTorch and TF pipelines with
    different temporal or spatial resolutions.
    """
    errors = []
    if dataset_cfg.num_frames != tf_cfg.num_frames:
        errors.append(
            f"num_frames mismatch: VideoDatasetConfig={dataset_cfg.num_frames} "
            f"vs TFPipelineConfig={tf_cfg.num_frames}"
        )
    if dataset_cfg.crop_size != tf_cfg.crop_size:
        errors.append(
            f"crop_size mismatch: VideoDatasetConfig={dataset_cfg.crop_size} "
            f"vs TFPipelineConfig={tf_cfg.crop_size}"
        )
    if errors:
        raise ValueError("Config compatibility check failed:\n" + "\n".join(errors))


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    passed = 0
    failed = 0

    def check(condition: bool, name: str, detail: str = "") -> None:
        global passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            msg = f"  FAIL: {name}"
            if detail:
                msg += f" -- {detail}"
            print(msg)
            failed += 1

    print("=" * 60)
    print("video_config_template Self-Tests")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Test 1: All config defaults
    # -----------------------------------------------------------------------
    print("\n[Test 1] All config defaults")
    ds_cfg = VideoDatasetConfig()
    dm_cfg = DataModuleConfig()
    tf_cfg = TFRecordConfig()
    tp_cfg = TFPipelineConfig()

    check(ds_cfg.num_frames == 16, f"VideoDatasetConfig.num_frames=16")
    check(ds_cfg.crop_size == 224, f"VideoDatasetConfig.crop_size=224")
    check(dm_cfg.batch_size == 8, f"DataModuleConfig.batch_size=8")
    check(dm_cfg.num_workers_per_gpu == 4, f"DataModuleConfig.num_workers_per_gpu=4")
    check(tf_cfg.num_shards == 256, f"TFRecordConfig.num_shards=256")
    check(tf_cfg.compression == "none", f"TFRecordConfig.compression='none'")
    check(tp_cfg.shuffle_buffer == 10_000, f"TFPipelineConfig.shuffle_buffer=10000")
    check(tp_cfg.augment is True, f"TFPipelineConfig.augment=True")

    # -----------------------------------------------------------------------
    # Test 2: Validation catches invalid values for all configs
    # -----------------------------------------------------------------------
    print("\n[Test 2] Validation across all config classes")

    # VideoDatasetConfig
    for kwargs, desc in [
        ({"num_frames": 0}, "VideoDatasetConfig rejects num_frames=0"),
        ({"stride": 0}, "VideoDatasetConfig rejects stride=0"),
        ({"crop_size": 0}, "VideoDatasetConfig rejects crop_size=0"),
        ({"crop_scale": (0.8, 0.5)}, "VideoDatasetConfig rejects inverted crop_scale"),
        ({"crop_ratio": (2.0, 1.0)}, "VideoDatasetConfig rejects inverted crop_ratio"),
        ({"hflip_prob": -0.1}, "VideoDatasetConfig rejects negative hflip_prob"),
        ({"hflip_prob": 1.1}, "VideoDatasetConfig rejects hflip_prob > 1"),
    ]:
        try:
            VideoDatasetConfig(**kwargs)
            check(False, desc, "no exception raised")
        except ValueError:
            check(True, desc)

    # DataModuleConfig
    for kwargs, desc in [
        ({"batch_size": 0}, "DataModuleConfig rejects batch_size=0"),
        ({"num_workers_per_gpu": -1}, "DataModuleConfig rejects negative workers"),
        ({"prefetch_factor": 0}, "DataModuleConfig rejects prefetch_factor=0"),
    ]:
        try:
            DataModuleConfig(**kwargs)
            check(False, desc, "no exception raised")
        except ValueError:
            check(True, desc)

    # TFRecordConfig
    for kwargs, desc in [
        ({"num_shards": 0}, "TFRecordConfig rejects num_shards=0"),
        ({"compression": "zstd"}, "TFRecordConfig rejects unknown compression"),
        ({"jpeg_quality": 0}, "TFRecordConfig rejects quality=0"),
        ({"jpeg_quality": 101}, "TFRecordConfig rejects quality=101"),
    ]:
        try:
            TFRecordConfig(**kwargs)
            check(False, desc, "no exception raised")
        except ValueError:
            check(True, desc)

    # TFPipelineConfig
    for kwargs, desc in [
        ({"shuffle_buffer": 0}, "TFPipelineConfig rejects shuffle_buffer=0"),
        ({"num_frames": 0}, "TFPipelineConfig rejects num_frames=0"),
        ({"crop_size": 0}, "TFPipelineConfig rejects crop_size=0"),
    ]:
        try:
            TFPipelineConfig(**kwargs)
            check(False, desc, "no exception raised")
        except ValueError:
            check(True, desc)

    # -----------------------------------------------------------------------
    # Test 3: Round-trip serialization for all configs
    # -----------------------------------------------------------------------
    print("\n[Test 3] Round-trip serialization (to_dict / from_dict)")

    # VideoDatasetConfig
    orig_ds = VideoDatasetConfig(num_frames=32, crop_size=112, stride=2)
    restored_ds = VideoDatasetConfig.from_dict(orig_ds.to_dict())
    check(restored_ds.num_frames == 32, f"VideoDatasetConfig: num_frames=32")
    check(restored_ds.crop_size == 112, f"VideoDatasetConfig: crop_size=112")
    check(restored_ds.crop_scale == orig_ds.crop_scale, "VideoDatasetConfig: crop_scale preserved")
    check(isinstance(restored_ds.normalize_mean, tuple), "VideoDatasetConfig: normalize_mean is tuple")

    # DataModuleConfig
    orig_dm = DataModuleConfig(data_root="/data/video", batch_size=32, num_workers_per_gpu=8)
    restored_dm = DataModuleConfig.from_dict(orig_dm.to_dict())
    check(restored_dm.batch_size == 32, "DataModuleConfig: batch_size=32")
    check(restored_dm.data_root == "/data/video", "DataModuleConfig: data_root preserved")

    # TFRecordConfig
    orig_tf = TFRecordConfig(num_shards=128, compression="jpeg", jpeg_quality=85)
    restored_tf = TFRecordConfig.from_dict(orig_tf.to_dict())
    check(restored_tf.num_shards == 128, "TFRecordConfig: num_shards=128")
    check(restored_tf.compression == "jpeg", "TFRecordConfig: compression='jpeg'")
    check(restored_tf.jpeg_quality == 85, "TFRecordConfig: jpeg_quality=85")

    # TFPipelineConfig
    orig_tp = TFPipelineConfig(shuffle_buffer=5000, num_frames=8, crop_size=112, augment=False)
    restored_tp = TFPipelineConfig.from_dict(orig_tp.to_dict())
    check(restored_tp.shuffle_buffer == 5000, "TFPipelineConfig: shuffle_buffer=5000")
    check(restored_tp.augment is False, "TFPipelineConfig: augment=False")
    check(isinstance(restored_tp.normalize_mean, tuple), "TFPipelineConfig: normalize_mean is tuple")

    # -----------------------------------------------------------------------
    # Test 4: JSON file round-trip
    # -----------------------------------------------------------------------
    print("\n[Test 4] JSON file round-trip (to_json / from_json)")
    with tempfile.TemporaryDirectory() as tmp:
        # VideoDatasetConfig
        ds_json_path = os.path.join(tmp, "ds_config.json")
        orig_ds.to_json(ds_json_path)
        loaded_ds = VideoDatasetConfig.from_json(ds_json_path)
        check(loaded_ds.num_frames == 32, "VideoDatasetConfig JSON: num_frames=32")
        check(loaded_ds.crop_size == 112, "VideoDatasetConfig JSON: crop_size=112")

        # DataModuleConfig
        dm_json_path = os.path.join(tmp, "dm_config.json")
        orig_dm.to_json(dm_json_path)
        loaded_dm = DataModuleConfig.from_json(dm_json_path)
        check(loaded_dm.batch_size == 32, "DataModuleConfig JSON: batch_size=32")

        # TFRecordConfig
        tf_json_path = os.path.join(tmp, "tf_config.json")
        orig_tf.to_json(tf_json_path)
        loaded_tf = TFRecordConfig.from_json(tf_json_path)
        check(loaded_tf.compression == "jpeg", "TFRecordConfig JSON: compression='jpeg'")

        # TFPipelineConfig
        tp_json_path = os.path.join(tmp, "tp_config.json")
        orig_tp.to_json(tp_json_path)
        loaded_tp = TFPipelineConfig.from_json(tp_json_path)
        check(loaded_tp.shuffle_buffer == 5000, "TFPipelineConfig JSON: shuffle_buffer=5000")

    # -----------------------------------------------------------------------
    # Test 5: Cross-config compatibility check
    # -----------------------------------------------------------------------
    print("\n[Test 5] Cross-config compatibility check")

    # Compatible pair
    ds_compat = VideoDatasetConfig(num_frames=16, crop_size=224)
    tp_compat = TFPipelineConfig(num_frames=16, crop_size=224)
    try:
        check_compatibility(ds_compat, tp_compat)
        check(True, "compatible configs pass check")
    except ValueError:
        check(False, "compatible configs pass check", "ValueError raised unexpectedly")

    # Incompatible: num_frames mismatch
    tp_mismatch = TFPipelineConfig(num_frames=8, crop_size=224)
    try:
        check_compatibility(ds_compat, tp_mismatch)
        check(False, "incompatible num_frames raises ValueError", "no exception raised")
    except ValueError:
        check(True, "incompatible num_frames raises ValueError")

    # Incompatible: crop_size mismatch
    tp_mismatch2 = TFPipelineConfig(num_frames=16, crop_size=112)
    try:
        check_compatibility(ds_compat, tp_mismatch2)
        check(False, "incompatible crop_size raises ValueError", "no exception raised")
    except ValueError:
        check(True, "incompatible crop_size raises ValueError")

    # -----------------------------------------------------------------------
    # Test 6: to_dict returns JSON-serializable types
    # -----------------------------------------------------------------------
    print("\n[Test 6] to_dict returns JSON-serializable types")
    for cfg_inst, name in [
        (VideoDatasetConfig(), "VideoDatasetConfig"),
        (DataModuleConfig(), "DataModuleConfig"),
        (TFRecordConfig(), "TFRecordConfig"),
        (TFPipelineConfig(), "TFPipelineConfig"),
    ]:
        d = cfg_inst.to_dict()
        try:
            json.dumps(d)
            check(True, f"{name}.to_dict() is JSON-serializable")
        except (TypeError, ValueError) as exc:
            check(False, f"{name}.to_dict() is JSON-serializable", str(exc))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Results: {passed} PASSED, {failed} FAILED out of {passed + failed} total")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
