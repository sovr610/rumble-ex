"""
brain_ai/data/base_loader.py -- BasePhaseLoader ABC, DatasetInfo dataclass, common utilities.

This module defines the abstract base class that all phase-specific data loaders must
implement, the DatasetInfo metadata schema, and shared utility functions for collation,
padding, and batching.

Key classes:
    DatasetInfo          -- Metadata dataclass for dataset introspection
    BasePhaseLoader      -- ABC enforcing the loader contract
    SyntheticDataset     -- Utility for generating synthetic datasets in dev mode

Key functions:
    dict_collate_fn      -- Collate for dictionary-of-tensors batches
    padded_collate_fn    -- Collate with sequence padding and masking
    episode_collate_fn   -- Collate for meta-learning episodes
    trajectory_collate_fn -- Collate for RL trajectories

Usage:
    class MyPhaseLoader(BasePhaseLoader):
        def get_train_loader(self) -> DataLoader: ...
        def get_val_loader(self) -> DataLoader: ...
        def get_test_loader(self) -> DataLoader: ...
        def get_dataset_info(self) -> DatasetInfo: ...
"""

from __future__ import annotations

import math
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SECTION 1: DataConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Centralized configuration for data loading across all phases."""
    root_dir: str = "data/"
    phase: int = 1
    mode: str = "dev"                        # dev | production
    batch_size: int = 64
    num_workers: int = 0                     # 0 for in-process loading (safe default)
    pin_memory: bool = False
    prefetch_factor: int = 2
    # Splits
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    # Augmentation
    augmentation_strength: str = "standard"  # none | light | standard | heavy
    # Streaming
    use_streaming: bool = False
    cache_dir: Optional[str] = None
    # Dev-mode dataset sizes
    dev_num_samples: int = 1000
    # Vision
    dev_image_size: int = 28
    dev_image_channels: int = 1
    prod_image_size: int = 224
    prod_image_channels: int = 3
    # Text
    dev_seq_len: int = 128
    prod_seq_len: int = 512
    dev_vocab_size: int = 256
    prod_vocab_size: int = 128000
    # Audio
    dev_n_mels: int = 64
    dev_audio_T: int = 100
    prod_n_mels: int = 128
    prod_audio_T: int = 1000
    # Sequences
    dev_seq_T: int = 50
    dev_seq_D: int = 16
    prod_seq_T: int = 200
    prod_seq_D: int = 64
    # RL
    dev_state_dim: int = 4
    dev_action_dim: int = 2
    # Meta-learning
    dev_n_way: int = 5
    dev_k_shot: int = 1
    dev_q_queries: int = 15
    # Classification
    dev_num_classes: int = 10

    def validate(self) -> List[str]:
        """Validate configuration, return list of error messages."""
        errors = []
        if self.mode not in ("dev", "production"):
            errors.append(f"mode must be 'dev' or 'production', got '{self.mode}'")
        if self.batch_size < 1:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            errors.append(f"num_workers must be non-negative, got {self.num_workers}")
        if not (1 <= self.phase <= 7):
            errors.append(f"phase must be 1-7, got {self.phase}")
        if self.val_ratio < 0 or self.val_ratio > 1:
            errors.append(f"val_ratio must be in [0,1], got {self.val_ratio}")
        if self.test_ratio < 0 or self.test_ratio > 1:
            errors.append(f"test_ratio must be in [0,1], got {self.test_ratio}")
        if self.val_ratio + self.test_ratio >= 1.0:
            errors.append("val_ratio + test_ratio must be < 1.0")
        if self.augmentation_strength not in ("none", "light", "standard", "heavy"):
            errors.append(f"augmentation_strength must be none/light/standard/heavy, got '{self.augmentation_strength}'")
        return errors


# ---------------------------------------------------------------------------
# SECTION 2: DatasetInfo dataclass
# ---------------------------------------------------------------------------

@dataclass
class DatasetInfo:
    """Metadata container for a dataset, returned by get_dataset_info()."""
    name: str
    phase: int
    modality: str
    num_classes: Optional[int] = None
    num_train_samples: int = 0
    num_val_samples: int = 0
    num_test_samples: int = 0
    input_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    target_shape: Optional[Tuple[int, ...]] = None
    dtype: torch.dtype = torch.float32
    target_dtype: torch.dtype = torch.long
    description: str = ""
    source: str = "synthetic"
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# SECTION 3: SyntheticDataset utility
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """A synthetic dataset that generates random data matching given shapes.

    Used for dev-mode testing and self-tests. All data is generated once in
    the constructor and stored in memory.
    """

    def __init__(
        self,
        num_samples: int,
        input_shapes: Dict[str, Tuple[int, ...]],
        input_dtypes: Optional[Dict[str, torch.dtype]] = None,
        target_shape: Optional[Tuple[int, ...]] = None,
        target_dtype: torch.dtype = torch.long,
        num_classes: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.input_shapes = input_shapes
        self.target_shape = target_shape
        self.target_dtype = target_dtype
        self.num_classes = num_classes

        gen = torch.Generator()
        gen.manual_seed(seed)

        self.data: Dict[str, Tensor] = {}
        dtypes = input_dtypes or {}

        for key, shape in input_shapes.items():
            dtype = dtypes.get(key, torch.float32)
            full_shape = (num_samples,) + shape
            if dtype in (torch.long, torch.int64, torch.int32):
                max_val = num_classes if num_classes and "target" not in key else 256
                self.data[key] = torch.randint(0, max_val, full_shape, generator=gen, dtype=dtype)
            elif dtype == torch.bool:
                self.data[key] = torch.rand(full_shape, generator=gen) > 0.3
            else:
                self.data[key] = torch.randn(full_shape, generator=gen, dtype=dtype)

        # Generate targets
        if target_shape is not None:
            full_target_shape = (num_samples,) + target_shape
            if target_dtype in (torch.long, torch.int64):
                nc = num_classes if num_classes else 10
                self.data["target"] = torch.randint(0, nc, full_target_shape, generator=gen, dtype=target_dtype)
            else:
                self.data["target"] = torch.randn(full_target_shape, generator=gen).to(target_dtype)
        elif num_classes is not None:
            self.data["target"] = torch.randint(0, num_classes, (num_samples,), generator=gen, dtype=target_dtype)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {key: val[idx] for key, val in self.data.items()}


# ---------------------------------------------------------------------------
# SECTION 4: Collate functions
# ---------------------------------------------------------------------------

def dict_collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Collate for dictionary-of-tensors batches (multimodal phases)."""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        tensors = [sample[key] for sample in batch]
        collated[key] = torch.stack(tensors, dim=0)
    return collated


def padded_collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Collate with padding for variable-length sequences.

    Expects each sample to have 'input' of shape [T_i, D] and optionally
    'target' of shape [T_i, D] or scalar. Produces padded tensors and masks.
    """
    inputs = [sample["input"] for sample in batch]
    max_len = max(x.shape[0] for x in inputs)
    feat_dim = inputs[0].shape[-1] if inputs[0].ndim > 1 else 1
    B = len(inputs)

    padded = torch.zeros(B, max_len, feat_dim)
    masks = torch.zeros(B, max_len, dtype=torch.bool)

    for i, x in enumerate(inputs):
        length = x.shape[0]
        if x.ndim == 1:
            padded[i, :length, 0] = x
        else:
            padded[i, :length] = x
        masks[i, :length] = True

    result = {"input": padded, "mask": masks}

    # Handle targets
    if "target" in batch[0]:
        targets = [sample["target"] for sample in batch]
        if targets[0].ndim >= 1 and targets[0].shape[0] > 1:
            # Sequence targets
            target_dim = targets[0].shape[-1] if targets[0].ndim > 1 else 1
            padded_targets = torch.zeros(B, max_len, target_dim)
            for i, t in enumerate(targets):
                length = t.shape[0]
                if t.ndim == 1:
                    padded_targets[i, :length, 0] = t
                else:
                    padded_targets[i, :length] = t
            result["target"] = padded_targets
        else:
            result["target"] = torch.stack(targets)

    return result


def episode_collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Collate for meta-learning episodes (support/query sets)."""
    return {
        "support_x": torch.stack([ep["support_x"] for ep in batch]),
        "support_y": torch.stack([ep["support_y"] for ep in batch]),
        "query_x": torch.stack([ep["query_x"] for ep in batch]),
        "query_y": torch.stack([ep["query_y"] for ep in batch]),
    }


def trajectory_collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Collate for RL trajectory dictionaries."""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        tensors = [sample[key] for sample in batch]
        collated[key] = torch.stack(tensors, dim=0)
    return collated


# ---------------------------------------------------------------------------
# SECTION 5: BasePhaseLoader ABC
# ---------------------------------------------------------------------------

class BasePhaseLoader(ABC):
    """Abstract base class that all phase-specific loaders must implement.

    Enforces a uniform interface for data loading across all 7 training phases
    of the brain_ai system.
    """

    def __init__(self, config: DataConfig, mode: str = "dev"):
        if mode not in ("dev", "production"):
            raise ValueError(f"mode must be 'dev' or 'production', got '{mode}'")
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid DataConfig: {'; '.join(errors)}")

        self.config = config
        self.mode = mode
        self.root_dir = Path(config.root_dir)

    @abstractmethod
    def get_train_loader(self) -> DataLoader:
        """Return a DataLoader for training data with augmentation and shuffling."""
        ...

    @abstractmethod
    def get_val_loader(self) -> DataLoader:
        """Return a DataLoader for validation data without augmentation."""
        ...

    @abstractmethod
    def get_test_loader(self) -> DataLoader:
        """Return a DataLoader for test data without augmentation."""
        ...

    @abstractmethod
    def get_dataset_info(self) -> DatasetInfo:
        """Return metadata about the loaded dataset."""
        ...

    def get_sample_shape(self) -> Dict[str, Tuple[int, ...]]:
        """Return per-key sample shapes (without batch dim) by inspecting a batch."""
        loader = self.get_train_loader()
        batch = next(iter(loader))
        shapes = {}
        if isinstance(batch, dict):
            for key, tensor in batch.items():
                if isinstance(tensor, Tensor):
                    shapes[key] = tuple(tensor.shape[1:])
        elif isinstance(batch, (tuple, list)):
            shapes["input"] = tuple(batch[0].shape[1:])
            if len(batch) > 1:
                shapes["target"] = tuple(batch[1].shape[1:])
        return shapes

    def get_collate_fn(self) -> Optional[Callable]:
        """Return a custom collate function, or None for default."""
        return None

    def get_num_samples(self) -> Dict[str, int]:
        """Return number of samples per split."""
        info = self.get_dataset_info()
        return {
            "train": info.num_train_samples,
            "val": info.num_val_samples,
            "test": info.num_test_samples,
        }

    def _make_loader(
        self,
        dataset: Dataset,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
    ) -> DataLoader:
        """Utility to create a DataLoader with standard settings."""
        kwargs = {
            "batch_size": self.config.batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "pin_memory": self.config.pin_memory,
            "num_workers": self.config.num_workers,
        }
        if self.config.num_workers > 0:
            kwargs["prefetch_factor"] = self.config.prefetch_factor
        if collate_fn is not None:
            kwargs["collate_fn"] = collate_fn
        return DataLoader(dataset, **kwargs)


# ---------------------------------------------------------------------------
# SECTION 6: Utility functions
# ---------------------------------------------------------------------------

def create_split_indices(
    total: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Create deterministic train/val/test split indices.

    Returns three non-overlapping lists of indices.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(total, generator=gen).tolist()

    n_test = int(total * test_ratio)
    n_val = int(total * val_ratio)
    n_train = total - n_val - n_test

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    return train_idx, val_idx, test_idx


def verify_no_overlap(
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
) -> bool:
    """Verify that three index lists have no overlap."""
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    return (
        len(train_set & val_set) == 0
        and len(train_set & test_set) == 0
        and len(val_set & test_set) == 0
    )


def compute_class_weights(labels: Tensor, num_classes: int) -> Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()
    counts = counts.clamp(min=1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights


# ============================================================================
# SELF-TESTS
# ============================================================================

if __name__ == "__main__":
    import sys
    import traceback

    passed = 0
    failed = 0
    test_results = []

    def run_test(name, fn):
        global passed, failed
        try:
            fn()
            passed += 1
            test_results.append(("PASS", name))
        except Exception as e:
            failed += 1
            test_results.append(("FAIL", name, str(e)))
            traceback.print_exc()

    # ---- DataConfig tests ----

    def test_dataconfig_defaults():
        cfg = DataConfig()
        assert cfg.mode == "dev"
        assert cfg.batch_size == 64
        assert cfg.phase == 1
    run_test("DataConfig defaults", test_dataconfig_defaults)

    def test_dataconfig_validate_ok():
        cfg = DataConfig()
        errors = cfg.validate()
        assert len(errors) == 0, f"Unexpected errors: {errors}"
    run_test("DataConfig validate OK", test_dataconfig_validate_ok)

    def test_dataconfig_validate_bad_mode():
        cfg = DataConfig(mode="invalid")
        errors = cfg.validate()
        assert any("mode" in e for e in errors)
    run_test("DataConfig validate bad mode", test_dataconfig_validate_bad_mode)

    def test_dataconfig_validate_bad_batch():
        cfg = DataConfig(batch_size=0)
        errors = cfg.validate()
        assert any("batch_size" in e for e in errors)
    run_test("DataConfig validate bad batch", test_dataconfig_validate_bad_batch)

    def test_dataconfig_validate_bad_workers():
        cfg = DataConfig(num_workers=-1)
        errors = cfg.validate()
        assert any("num_workers" in e for e in errors)
    run_test("DataConfig validate bad workers", test_dataconfig_validate_bad_workers)

    def test_dataconfig_validate_bad_phase():
        cfg = DataConfig(phase=0)
        errors = cfg.validate()
        assert any("phase" in e for e in errors)
    run_test("DataConfig validate bad phase", test_dataconfig_validate_bad_phase)

    def test_dataconfig_validate_bad_ratios():
        cfg = DataConfig(val_ratio=0.6, test_ratio=0.5)
        errors = cfg.validate()
        assert any("ratio" in e.lower() for e in errors)
    run_test("DataConfig validate bad ratios", test_dataconfig_validate_bad_ratios)

    def test_dataconfig_validate_bad_aug():
        cfg = DataConfig(augmentation_strength="extreme")
        errors = cfg.validate()
        assert any("augmentation" in e for e in errors)
    run_test("DataConfig validate bad augmentation", test_dataconfig_validate_bad_aug)

    # ---- DatasetInfo tests ----

    def test_datasetinfo_creation():
        info = DatasetInfo(
            name="test",
            phase=1,
            modality="vision",
            num_classes=10,
            num_train_samples=800,
            num_val_samples=100,
            num_test_samples=100,
            input_shapes={"input": (1, 28, 28)},
            target_shape=(),
        )
        assert info.name == "test"
        assert info.phase == 1
        assert info.dtype == torch.float32
        assert info.target_dtype == torch.long
    run_test("DatasetInfo creation", test_datasetinfo_creation)

    def test_datasetinfo_defaults():
        info = DatasetInfo(name="x", phase=1, modality="text")
        assert info.source == "synthetic"
        assert info.version == "1.0.0"
        assert info.num_classes is None
    run_test("DatasetInfo defaults", test_datasetinfo_defaults)

    # ---- SyntheticDataset tests ----

    def test_synthetic_dataset_len():
        ds = SyntheticDataset(
            num_samples=100,
            input_shapes={"input": (3, 32, 32)},
            num_classes=10,
        )
        assert len(ds) == 100
    run_test("SyntheticDataset length", test_synthetic_dataset_len)

    def test_synthetic_dataset_shapes():
        ds = SyntheticDataset(
            num_samples=50,
            input_shapes={"input": (1, 28, 28)},
            target_shape=(),
            num_classes=10,
        )
        sample = ds[0]
        assert sample["input"].shape == (1, 28, 28)
        assert sample["target"].shape == ()
    run_test("SyntheticDataset shapes", test_synthetic_dataset_shapes)

    def test_synthetic_dataset_dtypes():
        ds = SyntheticDataset(
            num_samples=50,
            input_shapes={"tokens": (128,)},
            input_dtypes={"tokens": torch.long},
            num_classes=10,
        )
        sample = ds[0]
        assert sample["tokens"].dtype == torch.long
    run_test("SyntheticDataset dtypes", test_synthetic_dataset_dtypes)

    def test_synthetic_dataset_bool():
        ds = SyntheticDataset(
            num_samples=50,
            input_shapes={"mask": (128,)},
            input_dtypes={"mask": torch.bool},
        )
        sample = ds[0]
        assert sample["mask"].dtype == torch.bool
    run_test("SyntheticDataset bool dtype", test_synthetic_dataset_bool)

    def test_synthetic_dataset_reproducibility():
        ds1 = SyntheticDataset(num_samples=10, input_shapes={"x": (4,)}, seed=123)
        ds2 = SyntheticDataset(num_samples=10, input_shapes={"x": (4,)}, seed=123)
        assert torch.equal(ds1[0]["x"], ds2[0]["x"])
    run_test("SyntheticDataset reproducibility", test_synthetic_dataset_reproducibility)

    def test_synthetic_dataset_different_seeds():
        ds1 = SyntheticDataset(num_samples=10, input_shapes={"x": (4,)}, seed=1)
        ds2 = SyntheticDataset(num_samples=10, input_shapes={"x": (4,)}, seed=2)
        assert not torch.equal(ds1[0]["x"], ds2[0]["x"])
    run_test("SyntheticDataset different seeds", test_synthetic_dataset_different_seeds)

    def test_synthetic_dataset_multimodal():
        ds = SyntheticDataset(
            num_samples=20,
            input_shapes={"vision": (1, 28, 28), "text": (128,), "audio": (64, 100)},
            input_dtypes={"text": torch.long},
            num_classes=5,
        )
        sample = ds[0]
        assert sample["vision"].shape == (1, 28, 28)
        assert sample["text"].shape == (128,)
        assert sample["text"].dtype == torch.long
        assert sample["audio"].shape == (64, 100)
        assert "target" in sample
    run_test("SyntheticDataset multimodal", test_synthetic_dataset_multimodal)

    def test_synthetic_target_range():
        ds = SyntheticDataset(
            num_samples=200,
            input_shapes={"x": (4,)},
            target_shape=(),
            num_classes=5,
        )
        targets = torch.stack([ds[i]["target"] for i in range(200)])
        assert targets.min() >= 0
        assert targets.max() < 5
    run_test("SyntheticDataset target range", test_synthetic_target_range)

    # ---- Collate function tests ----

    def test_dict_collate():
        batch = [
            {"x": torch.randn(3, 32, 32), "y": torch.tensor(1)},
            {"x": torch.randn(3, 32, 32), "y": torch.tensor(2)},
        ]
        result = dict_collate_fn(batch)
        assert result["x"].shape == (2, 3, 32, 32)
        assert result["y"].shape == (2,)
    run_test("dict_collate_fn", test_dict_collate)

    def test_padded_collate():
        batch = [
            {"input": torch.randn(10, 4), "target": torch.tensor(0)},
            {"input": torch.randn(15, 4), "target": torch.tensor(1)},
        ]
        result = padded_collate_fn(batch)
        assert result["input"].shape == (2, 15, 4)
        assert result["mask"].shape == (2, 15)
        assert result["mask"][0, 9].item() is True
        assert result["mask"][0, 14].item() is False
        assert result["mask"][1, 14].item() is True
    run_test("padded_collate_fn", test_padded_collate)

    def test_padded_collate_seq_target():
        batch = [
            {"input": torch.randn(10, 4), "target": torch.randn(10, 4)},
            {"input": torch.randn(15, 4), "target": torch.randn(15, 4)},
        ]
        result = padded_collate_fn(batch)
        assert result["target"].shape == (2, 15, 4)
    run_test("padded_collate_fn sequence targets", test_padded_collate_seq_target)

    def test_episode_collate():
        batch = [
            {"support_x": torch.randn(5, 1, 28, 28), "support_y": torch.randint(0, 5, (5,)),
             "query_x": torch.randn(15, 1, 28, 28), "query_y": torch.randint(0, 5, (15,))},
            {"support_x": torch.randn(5, 1, 28, 28), "support_y": torch.randint(0, 5, (5,)),
             "query_x": torch.randn(15, 1, 28, 28), "query_y": torch.randint(0, 5, (15,))},
        ]
        result = episode_collate_fn(batch)
        assert result["support_x"].shape == (2, 5, 1, 28, 28)
        assert result["query_y"].shape == (2, 15)
    run_test("episode_collate_fn", test_episode_collate)

    def test_trajectory_collate():
        batch = [
            {"state": torch.randn(4), "action": torch.tensor(1), "reward": torch.tensor(1.0)},
            {"state": torch.randn(4), "action": torch.tensor(0), "reward": torch.tensor(0.5)},
        ]
        result = trajectory_collate_fn(batch)
        assert result["state"].shape == (2, 4)
        assert result["action"].shape == (2,)
    run_test("trajectory_collate_fn", test_trajectory_collate)

    # ---- BasePhaseLoader tests (via concrete subclass) ----

    class _MockLoader(BasePhaseLoader):
        def __init__(self, config, mode="dev"):
            super().__init__(config, mode)
            n = config.dev_num_samples
            self.train_ds = SyntheticDataset(int(n * 0.8), {"input": (1, 28, 28)}, num_classes=10)
            self.val_ds = SyntheticDataset(int(n * 0.1), {"input": (1, 28, 28)}, num_classes=10, seed=99)
            self.test_ds = SyntheticDataset(int(n * 0.1), {"input": (1, 28, 28)}, num_classes=10, seed=199)

        def get_train_loader(self):
            return self._make_loader(self.train_ds, shuffle=True, drop_last=True, collate_fn=dict_collate_fn)

        def get_val_loader(self):
            return self._make_loader(self.val_ds, shuffle=False, drop_last=False, collate_fn=dict_collate_fn)

        def get_test_loader(self):
            return self._make_loader(self.test_ds, shuffle=False, drop_last=False, collate_fn=dict_collate_fn)

        def get_dataset_info(self):
            return DatasetInfo(name="mock", phase=1, modality="vision", num_classes=10,
                              num_train_samples=len(self.train_ds),
                              num_val_samples=len(self.val_ds),
                              num_test_samples=len(self.test_ds),
                              input_shapes={"input": (1, 28, 28)}, target_shape=())

    def test_base_loader_mode_validation():
        cfg = DataConfig()
        try:
            _MockLoader(cfg, mode="bad")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
    run_test("BasePhaseLoader mode validation", test_base_loader_mode_validation)

    def test_base_loader_dev_mode():
        cfg = DataConfig(batch_size=8, dev_num_samples=100)
        loader = _MockLoader(cfg, mode="dev")
        assert loader.mode == "dev"
    run_test("BasePhaseLoader dev mode", test_base_loader_dev_mode)

    def test_base_loader_train_returns_dataloader():
        cfg = DataConfig(batch_size=8, dev_num_samples=100)
        loader = _MockLoader(cfg, mode="dev")
        dl = loader.get_train_loader()
        assert isinstance(dl, DataLoader)
    run_test("BasePhaseLoader train returns DataLoader", test_base_loader_train_returns_dataloader)

    def test_base_loader_train_batch_shape():
        cfg = DataConfig(batch_size=8, dev_num_samples=100)
        loader = _MockLoader(cfg, mode="dev")
        dl = loader.get_train_loader()
        batch = next(iter(dl))
        assert batch["input"].shape == (8, 1, 28, 28)
    run_test("BasePhaseLoader train batch shape", test_base_loader_train_batch_shape)

    def test_base_loader_get_dataset_info():
        cfg = DataConfig(batch_size=8, dev_num_samples=100)
        loader = _MockLoader(cfg, mode="dev")
        info = loader.get_dataset_info()
        assert isinstance(info, DatasetInfo)
        assert info.name == "mock"
    run_test("BasePhaseLoader get_dataset_info", test_base_loader_get_dataset_info)

    def test_base_loader_get_sample_shape():
        cfg = DataConfig(batch_size=8, dev_num_samples=100)
        loader = _MockLoader(cfg, mode="dev")
        shapes = loader.get_sample_shape()
        assert "input" in shapes
        assert shapes["input"] == (1, 28, 28)
    run_test("BasePhaseLoader get_sample_shape", test_base_loader_get_sample_shape)

    def test_base_loader_get_num_samples():
        cfg = DataConfig(batch_size=8, dev_num_samples=100)
        loader = _MockLoader(cfg, mode="dev")
        counts = loader.get_num_samples()
        assert "train" in counts
        assert "val" in counts
        assert "test" in counts
        assert counts["train"] == 80
    run_test("BasePhaseLoader get_num_samples", test_base_loader_get_num_samples)

    # ---- Split utility tests ----

    def test_create_split_indices_no_overlap():
        train, val, test = create_split_indices(1000, 0.1, 0.1, seed=42)
        assert verify_no_overlap(train, val, test)
    run_test("create_split_indices no overlap", test_create_split_indices_no_overlap)

    def test_create_split_indices_covers_all():
        train, val, test = create_split_indices(1000, 0.1, 0.1, seed=42)
        assert len(train) + len(val) + len(test) == 1000
    run_test("create_split_indices covers all", test_create_split_indices_covers_all)

    def test_create_split_indices_ratios():
        train, val, test = create_split_indices(1000, 0.1, 0.1, seed=42)
        assert len(val) == 100
        assert len(test) == 100
        assert len(train) == 800
    run_test("create_split_indices ratios", test_create_split_indices_ratios)

    def test_create_split_indices_reproducible():
        t1, v1, s1 = create_split_indices(500, seed=99)
        t2, v2, s2 = create_split_indices(500, seed=99)
        assert t1 == t2 and v1 == v2 and s1 == s2
    run_test("create_split_indices reproducible", test_create_split_indices_reproducible)

    def test_verify_no_overlap_true():
        assert verify_no_overlap([0, 1, 2], [3, 4], [5, 6, 7])
    run_test("verify_no_overlap true", test_verify_no_overlap_true)

    def test_verify_no_overlap_false():
        assert not verify_no_overlap([0, 1, 2], [2, 3], [5, 6])
    run_test("verify_no_overlap false", test_verify_no_overlap_false)

    # ---- Class weights tests ----

    def test_compute_class_weights_balanced():
        labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        weights = compute_class_weights(labels, 5)
        assert weights.shape == (5,)
        assert torch.allclose(weights, torch.ones(5), atol=1e-5)
    run_test("compute_class_weights balanced", test_compute_class_weights_balanced)

    def test_compute_class_weights_imbalanced():
        labels = torch.tensor([0, 0, 0, 0, 1])
        weights = compute_class_weights(labels, 2)
        assert weights[1] > weights[0]  # Minority class gets higher weight
    run_test("compute_class_weights imbalanced", test_compute_class_weights_imbalanced)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"BASE LOADER TEMPLATE SELF-TESTS: {passed} passed, {failed} failed")
    print("=" * 60)
    for result in test_results:
        status = result[0]
        name = result[1]
        extra = f" -- {result[2]}" if len(result) > 2 else ""
        print(f"  [{status}] {name}{extra}")

    sys.exit(0 if failed == 0 else 1)
