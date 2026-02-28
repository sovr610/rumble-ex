"""
brain_ai/data/phase_loaders.py -- All 7 phase-specific data loaders.

Each loader inherits from BasePhaseLoader and provides synthetic datasets
in dev mode. Production mode stubs are included for future dataset integration.

Phase loaders:
    SNNPhaseLoader          -- Phase 1: SNN core, vision data
    EncoderPhaseLoader      -- Phase 2: Multi-modal encoders
    HTMPhaseLoader          -- Phase 3: HTM sequence learning
    WorkspacePhaseLoader    -- Phase 4: Global workspace, multimodal
    ActiveInfPhaseLoader    -- Phase 5: Active inference, RL episodes
    ReasoningPhaseLoader    -- Phase 6: Dual-process reasoning, logic tasks
    MetaPhaseLoader         -- Phase 7: Meta-learning, few-shot episodes

Usage:
    from brain_ai.data.base_loader import DataConfig
    from brain_ai.data.phase_loaders import SNNPhaseLoader

    config = DataConfig(phase=1, mode="dev", batch_size=32)
    loader = SNNPhaseLoader(config, mode="dev")
    train_dl = loader.get_train_loader()
    batch = next(iter(train_dl))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset

# ---------------------------------------------------------------------------
# Inline dependencies (in production these are imported from base_loader)
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Centralized configuration for data loading across all phases."""
    root_dir: str = "data/"
    phase: int = 1
    mode: str = "dev"
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    augmentation_strength: str = "standard"
    use_streaming: bool = False
    cache_dir: Optional[str] = None
    dev_num_samples: int = 1000
    dev_image_size: int = 28
    dev_image_channels: int = 1
    prod_image_size: int = 224
    prod_image_channels: int = 3
    dev_seq_len: int = 128
    prod_seq_len: int = 512
    dev_vocab_size: int = 256
    prod_vocab_size: int = 128000
    dev_n_mels: int = 64
    dev_audio_T: int = 100
    prod_n_mels: int = 128
    prod_audio_T: int = 1000
    dev_seq_T: int = 50
    dev_seq_D: int = 16
    prod_seq_T: int = 200
    prod_seq_D: int = 64
    dev_state_dim: int = 4
    dev_action_dim: int = 2
    dev_n_way: int = 5
    dev_k_shot: int = 1
    dev_q_queries: int = 15
    dev_num_classes: int = 10

    def validate(self) -> List[str]:
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
            errors.append(f"augmentation_strength invalid: '{self.augmentation_strength}'")
        return errors


@dataclass
class DatasetInfo:
    """Metadata container for a dataset."""
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


class SyntheticDataset(Dataset):
    """Generates random data matching given shapes. Used for dev mode."""

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
# Collate functions
# ---------------------------------------------------------------------------

def dict_collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([sample[key] for sample in batch]) for key in keys}


def padded_collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
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
    if "target" in batch[0]:
        targets = [sample["target"] for sample in batch]
        if targets[0].ndim >= 1 and targets[0].shape[0] > 1:
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
    return {
        "support_x": torch.stack([ep["support_x"] for ep in batch]),
        "support_y": torch.stack([ep["support_y"] for ep in batch]),
        "query_x": torch.stack([ep["query_x"] for ep in batch]),
        "query_y": torch.stack([ep["query_y"] for ep in batch]),
    }


def trajectory_collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([sample[key] for sample in batch]) for key in keys}


# ---------------------------------------------------------------------------
# BasePhaseLoader ABC
# ---------------------------------------------------------------------------

class BasePhaseLoader(ABC):
    """Abstract base class for all phase-specific data loaders."""

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
    def get_train_loader(self) -> DataLoader: ...

    @abstractmethod
    def get_val_loader(self) -> DataLoader: ...

    @abstractmethod
    def get_test_loader(self) -> DataLoader: ...

    @abstractmethod
    def get_dataset_info(self) -> DatasetInfo: ...

    def get_sample_shape(self) -> Dict[str, Tuple[int, ...]]:
        loader = self.get_train_loader()
        batch = next(iter(loader))
        shapes = {}
        if isinstance(batch, dict):
            for key, tensor in batch.items():
                if isinstance(tensor, Tensor):
                    shapes[key] = tuple(tensor.shape[1:])
        return shapes

    def get_collate_fn(self) -> Optional[Callable]:
        return None

    def get_num_samples(self) -> Dict[str, int]:
        info = self.get_dataset_info()
        return {"train": info.num_train_samples, "val": info.num_val_samples, "test": info.num_test_samples}

    def _make_loader(self, dataset: Dataset, shuffle: bool = False,
                     drop_last: bool = False, collate_fn: Optional[Callable] = None) -> DataLoader:
        kwargs: Dict[str, Any] = {
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


def _create_split_indices(total: int, val_ratio: float = 0.1,
                          test_ratio: float = 0.1, seed: int = 42
                          ) -> Tuple[List[int], List[int], List[int]]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(total, generator=gen).tolist()
    n_test = int(total * test_ratio)
    n_val = int(total * val_ratio)
    n_train = total - n_val - n_test
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


# ============================================================================
# PHASE 1: SNN Core -- SNNPhaseLoader
# ============================================================================

class SNNPhaseLoader(BasePhaseLoader):
    """Phase 1 data loader: SNN core training with vision data.

    Dev: Synthetic MNIST-like grayscale images [1, 28, 28].
    Production: CIFAR-10/100 style images [3, 32, 32].
    """

    def __init__(self, config: DataConfig, mode: str = "dev"):
        super().__init__(config, mode)
        n = config.dev_num_samples
        if mode == "dev":
            C, H, W = config.dev_image_channels, config.dev_image_size, config.dev_image_size
            nc = config.dev_num_classes
            full_ds = SyntheticDataset(
                num_samples=n,
                input_shapes={"input": (C, H, W)},
                num_classes=nc,
                seed=config.split_seed,
            )
        else:
            C, H, W = config.prod_image_channels, config.prod_image_size, config.prod_image_size
            nc = 100
            full_ds = SyntheticDataset(
                num_samples=n,
                input_shapes={"input": (C, H, W)},
                num_classes=nc,
                seed=config.split_seed,
            )

        train_idx, val_idx, test_idx = _create_split_indices(
            n, config.val_ratio, config.test_ratio, config.split_seed
        )
        self.train_ds = Subset(full_ds, train_idx)
        self.val_ds = Subset(full_ds, val_idx)
        self.test_ds = Subset(full_ds, test_idx)
        self._nc = nc
        self._input_shape = (C, H, W)
        self._n_train = len(train_idx)
        self._n_val = len(val_idx)
        self._n_test = len(test_idx)

    def get_train_loader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True, drop_last=True, collate_fn=dict_collate_fn)

    def get_val_loader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False, drop_last=False, collate_fn=dict_collate_fn)

    def get_test_loader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False, drop_last=False, collate_fn=dict_collate_fn)

    def get_dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="synthetic_snn" if self.mode == "dev" else "cifar",
            phase=1, modality="vision", num_classes=self._nc,
            num_train_samples=self._n_train, num_val_samples=self._n_val,
            num_test_samples=self._n_test,
            input_shapes={"input": self._input_shape}, target_shape=(),
            description="SNN core vision dataset",
        )


# ============================================================================
# PHASE 2: Encoders -- EncoderPhaseLoader
# ============================================================================

class EncoderPhaseLoader(BasePhaseLoader):
    """Phase 2 data loader: Multi-modal encoder training.

    Dev: Synthetic multimodal samples (vision + text + audio).
    Production: ImageNet + LibriSpeech + WikiText style data.
    """

    def __init__(self, config: DataConfig, mode: str = "dev"):
        super().__init__(config, mode)
        n = config.dev_num_samples
        if mode == "dev":
            shapes = {
                "vision": (config.dev_image_channels, config.dev_image_size, config.dev_image_size),
                "text": (config.dev_seq_len,),
                "audio": (config.dev_n_mels, config.dev_audio_T),
            }
            dtypes = {"text": torch.long}
            nc = config.dev_num_classes
        else:
            shapes = {
                "vision": (config.prod_image_channels, config.prod_image_size, config.prod_image_size),
                "text": (config.prod_seq_len,),
                "audio": (config.prod_n_mels, config.prod_audio_T),
            }
            dtypes = {"text": torch.long}
            nc = 1000

        full_ds = SyntheticDataset(
            num_samples=n,
            input_shapes=shapes,
            input_dtypes=dtypes,
            num_classes=nc,
            seed=config.split_seed,
        )

        train_idx, val_idx, test_idx = _create_split_indices(
            n, config.val_ratio, config.test_ratio, config.split_seed
        )
        self.train_ds = Subset(full_ds, train_idx)
        self.val_ds = Subset(full_ds, val_idx)
        self.test_ds = Subset(full_ds, test_idx)
        self._nc = nc
        self._shapes = shapes
        self._n_train = len(train_idx)
        self._n_val = len(val_idx)
        self._n_test = len(test_idx)

    def get_train_loader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True, drop_last=True, collate_fn=dict_collate_fn)

    def get_val_loader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False, drop_last=False, collate_fn=dict_collate_fn)

    def get_test_loader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False, drop_last=False, collate_fn=dict_collate_fn)

    def get_dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="synthetic_multimodal" if self.mode == "dev" else "multimodal_prod",
            phase=2, modality="multimodal", num_classes=self._nc,
            num_train_samples=self._n_train, num_val_samples=self._n_val,
            num_test_samples=self._n_test,
            input_shapes=self._shapes, target_shape=(),
            description="Multi-modal encoder dataset (vision + text + audio)",
        )

    def get_collate_fn(self) -> Optional[Callable]:
        return dict_collate_fn


# ============================================================================
# PHASE 3: HTM -- HTMPhaseLoader
# ============================================================================

class _HTMSyntheticDataset(Dataset):
    """Synthetic temporal sequences with anomalies for HTM training.

    Generates periodic sinusoidal sequences with injected anomalies.
    Targets are next-step predictions of shape [T, D].
    """

    def __init__(self, num_samples: int, seq_T: int, seq_D: int,
                 anomaly_ratio: float = 0.1, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        gen = torch.Generator()
        gen.manual_seed(seed)

        # Generate periodic base patterns
        t = torch.linspace(0, 4 * math.pi, seq_T).unsqueeze(0).unsqueeze(-1)
        freqs = torch.rand(num_samples, 1, seq_D, generator=gen) * 2.0 + 0.5
        phases = torch.rand(num_samples, 1, seq_D, generator=gen) * 2.0 * math.pi
        self.sequences = torch.sin(t * freqs + phases)  # [N, T, D]

        # Add noise
        noise = torch.randn(num_samples, seq_T, seq_D, generator=gen) * 0.05
        self.sequences = self.sequences + noise

        # Create next-step targets (shifted by 1)
        self.targets = torch.zeros_like(self.sequences)
        self.targets[:, :-1, :] = self.sequences[:, 1:, :]
        self.targets[:, -1, :] = self.sequences[:, -1, :]

        # Create masks (all valid)
        self.masks = torch.ones(num_samples, seq_T, dtype=torch.bool)

        # Inject anomalies
        n_anomalies = max(1, int(num_samples * anomaly_ratio))
        anom_indices = torch.randperm(num_samples, generator=gen)[:n_anomalies]
        for idx in anom_indices:
            start = torch.randint(0, seq_T // 2, (1,), generator=gen).item()
            length = torch.randint(3, min(10, seq_T // 4), (1,), generator=gen).item()
            self.sequences[idx, start:start + length] += torch.randn(length, seq_D) * 2.0

        self.anomaly_labels = torch.zeros(num_samples, dtype=torch.long)
        self.anomaly_labels[anom_indices] = 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "input": self.sequences[idx],       # [T, D]
            "mask": self.masks[idx],             # [T]
            "target": self.targets[idx],         # [T, D]
            "anomaly": self.anomaly_labels[idx], # scalar
        }


class HTMPhaseLoader(BasePhaseLoader):
    """Phase 3 data loader: HTM sequence learning.

    Dev: Synthetic periodic sequences with anomalies.
    Production: NAB / taxi / ECG time series data.
    """

    def __init__(self, config: DataConfig, mode: str = "dev"):
        super().__init__(config, mode)
        n = config.dev_num_samples
        if mode == "dev":
            seq_T, seq_D = config.dev_seq_T, config.dev_seq_D
        else:
            seq_T, seq_D = config.prod_seq_T, config.prod_seq_D

        full_ds = _HTMSyntheticDataset(n, seq_T, seq_D, seed=config.split_seed)
        train_idx, val_idx, test_idx = _create_split_indices(
            n, config.val_ratio, config.test_ratio, config.split_seed
        )
        self.train_ds = Subset(full_ds, train_idx)
        self.val_ds = Subset(full_ds, val_idx)
        self.test_ds = Subset(full_ds, test_idx)
        self._seq_T = seq_T
        self._seq_D = seq_D
        self._n_train = len(train_idx)
        self._n_val = len(val_idx)
        self._n_test = len(test_idx)

    def get_train_loader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True, drop_last=True, collate_fn=dict_collate_fn)

    def get_val_loader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False, drop_last=False, collate_fn=dict_collate_fn)

    def get_test_loader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False, drop_last=False, collate_fn=dict_collate_fn)

    def get_dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="synthetic_sequences" if self.mode == "dev" else "nab_sequences",
            phase=3, modality="sequence",
            num_train_samples=self._n_train, num_val_samples=self._n_val,
            num_test_samples=self._n_test,
            input_shapes={"input": (self._seq_T, self._seq_D), "mask": (self._seq_T,)},
            target_shape=(self._seq_T, self._seq_D),
            target_dtype=torch.float32,
            description="HTM temporal sequence dataset with anomaly labels",
        )

    def get_collate_fn(self) -> Optional[Callable]:
        return dict_collate_fn


# ============================================================================
# PHASE 4: Workspace -- WorkspacePhaseLoader
# ============================================================================

class WorkspacePhaseLoader(BasePhaseLoader):
    """Phase 4 data loader: Global workspace integration.

    Dev: Synthetic paired multimodal samples (vision + text + audio).
    Production: VQA v2 / CMU-MOSEI style data.
    """

    def __init__(self, config: DataConfig, mode: str = "dev"):
        super().__init__(config, mode)
        n = config.dev_num_samples
        if mode == "dev":
            shapes = {
                "vision": (config.dev_image_channels, config.dev_image_size, config.dev_image_size),
                "text": (config.dev_seq_len,),
                "audio": (config.dev_n_mels, config.dev_audio_T),
            }
            dtypes = {"text": torch.long}
            nc = config.dev_num_classes
        else:
            shapes = {
                "vision": (config.prod_image_channels, config.prod_image_size, config.prod_image_size),
                "text": (config.prod_seq_len,),
                "audio": (config.prod_n_mels, config.prod_audio_T),
            }
            dtypes = {"text": torch.long}
            nc = 100

        full_ds = SyntheticDataset(
            num_samples=n,
            input_shapes=shapes,
            input_dtypes=dtypes,
            num_classes=nc,
            seed=config.split_seed + 4,  # different seed from phase 2
        )

        train_idx, val_idx, test_idx = _create_split_indices(
            n, config.val_ratio, config.test_ratio, config.split_seed
        )
        self.train_ds = Subset(full_ds, train_idx)
        self.val_ds = Subset(full_ds, val_idx)
        self.test_ds = Subset(full_ds, test_idx)
        self._nc = nc
        self._shapes = shapes
        self._n_train = len(train_idx)
        self._n_val = len(val_idx)
        self._n_test = len(test_idx)

    def get_train_loader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True, drop_last=True, collate_fn=dict_collate_fn)

    def get_val_loader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False, drop_last=False, collate_fn=dict_collate_fn)

    def get_test_loader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False, drop_last=False, collate_fn=dict_collate_fn)

    def get_dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="synthetic_workspace" if self.mode == "dev" else "vqa_workspace",
            phase=4, modality="multimodal", num_classes=self._nc,
            num_train_samples=self._n_train, num_val_samples=self._n_val,
            num_test_samples=self._n_test,
            input_shapes=self._shapes, target_shape=(),
            description="Workspace multimodal integration dataset",
        )

    def get_collate_fn(self) -> Optional[Callable]:
        return dict_collate_fn


# ============================================================================
# PHASE 5: Active Inference -- ActiveInfPhaseLoader
# ============================================================================

class _RLTrajectoryDataset(Dataset):
    """Synthetic RL trajectories for active inference training.

    Simulates CartPole-like episodes with state, action, reward, next_state, done.
    """

    def __init__(self, num_samples: int, state_dim: int, action_dim: int, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        gen = torch.Generator()
        gen.manual_seed(seed)

        # States: random positions + velocities
        self.states = torch.randn(num_samples, state_dim, generator=gen)

        # Actions: discrete (0 or 1 for CartPole-like)
        self.actions = torch.randint(0, action_dim, (num_samples,), generator=gen, dtype=torch.long)

        # Rewards: mostly 1.0 with some terminal 0.0
        self.rewards = torch.ones(num_samples)
        terminal_mask = torch.rand(num_samples, generator=gen) < 0.1
        self.rewards[terminal_mask] = 0.0

        # Next states: state + small delta
        delta = torch.randn(num_samples, state_dim, generator=gen) * 0.1
        self.next_states = self.states + delta

        # Done flags: 1.0 where reward is 0.0 (terminal)
        self.dones = torch.zeros(num_samples)
        self.dones[terminal_mask] = 1.0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "state": self.states[idx],           # [state_dim]
            "action": self.actions[idx],         # scalar
            "reward": self.rewards[idx],         # scalar
            "next_state": self.next_states[idx], # [state_dim]
            "done": self.dones[idx],             # scalar
        }


class ActiveInfPhaseLoader(BasePhaseLoader):
    """Phase 5 data loader: Active inference with RL episodes.

    Dev: Synthetic CartPole-like trajectories.
    Production: D4RL / Minari offline RL datasets.
    """

    def __init__(self, config: DataConfig, mode: str = "dev"):
        super().__init__(config, mode)
        n = config.dev_num_samples
        state_dim = config.dev_state_dim
        action_dim = config.dev_action_dim

        full_ds = _RLTrajectoryDataset(n, state_dim, action_dim, seed=config.split_seed)
        train_idx, val_idx, test_idx = _create_split_indices(
            n, config.val_ratio, config.test_ratio, config.split_seed
        )
        self.train_ds = Subset(full_ds, train_idx)
        self.val_ds = Subset(full_ds, val_idx)
        self.test_ds = Subset(full_ds, test_idx)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._n_train = len(train_idx)
        self._n_val = len(val_idx)
        self._n_test = len(test_idx)

    def get_train_loader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True, drop_last=True, collate_fn=trajectory_collate_fn)

    def get_val_loader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False, drop_last=False, collate_fn=trajectory_collate_fn)

    def get_test_loader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False, drop_last=False, collate_fn=trajectory_collate_fn)

    def get_dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="synthetic_cartpole" if self.mode == "dev" else "d4rl",
            phase=5, modality="rl",
            num_train_samples=self._n_train, num_val_samples=self._n_val,
            num_test_samples=self._n_test,
            input_shapes={"state": (self._state_dim,), "action": (), "next_state": (self._state_dim,)},
            target_shape=(),
            target_dtype=torch.float32,
            description="Active inference RL trajectory dataset",
        )

    def get_collate_fn(self) -> Optional[Callable]:
        return trajectory_collate_fn


# ============================================================================
# PHASE 6: Reasoning -- ReasoningPhaseLoader
# ============================================================================

class _ReasoningSyntheticDataset(Dataset):
    """Synthetic logic reasoning tasks inspired by bAbI.

    Each sample has a context (token IDs), a question (token IDs), and a target class.
    """

    def __init__(self, num_samples: int, context_len: int = 256,
                 question_len: int = 64, vocab_size: int = 256,
                 num_classes: int = 10, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        gen = torch.Generator()
        gen.manual_seed(seed)

        # Context tokens
        self.contexts = torch.randint(1, vocab_size, (num_samples, context_len),
                                      generator=gen, dtype=torch.long)
        # Pad last 20% of context to simulate variable length
        pad_starts = torch.randint(int(context_len * 0.6), context_len,
                                   (num_samples,), generator=gen)
        for i in range(num_samples):
            self.contexts[i, pad_starts[i]:] = 0

        # Question tokens
        self.questions = torch.randint(1, vocab_size, (num_samples, question_len),
                                       generator=gen, dtype=torch.long)
        q_pad_starts = torch.randint(int(question_len * 0.5), question_len,
                                     (num_samples,), generator=gen)
        for i in range(num_samples):
            self.questions[i, q_pad_starts[i]:] = 0

        # Answer targets
        self.targets = torch.randint(0, num_classes, (num_samples,),
                                     generator=gen, dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "context": self.contexts[idx],    # [context_len] int64
            "question": self.questions[idx],  # [question_len] int64
            "target": self.targets[idx],      # scalar int64
        }


class ReasoningPhaseLoader(BasePhaseLoader):
    """Phase 6 data loader: Dual-process reasoning with logic tasks.

    Dev: Synthetic mini-bAbI tasks.
    Production: bAbI / ProofWriter datasets.
    """

    def __init__(self, config: DataConfig, mode: str = "dev"):
        super().__init__(config, mode)
        n = config.dev_num_samples
        if mode == "dev":
            context_len = 256
            question_len = 64
            vocab_size = config.dev_vocab_size
            nc = config.dev_num_classes
        else:
            context_len = config.prod_seq_len
            question_len = 128
            vocab_size = config.prod_vocab_size
            nc = 20

        full_ds = _ReasoningSyntheticDataset(
            n, context_len, question_len, vocab_size, nc, seed=config.split_seed
        )
        train_idx, val_idx, test_idx = _create_split_indices(
            n, config.val_ratio, config.test_ratio, config.split_seed
        )
        self.train_ds = Subset(full_ds, train_idx)
        self.val_ds = Subset(full_ds, val_idx)
        self.test_ds = Subset(full_ds, test_idx)
        self._nc = nc
        self._context_len = context_len
        self._question_len = question_len
        self._n_train = len(train_idx)
        self._n_val = len(val_idx)
        self._n_test = len(test_idx)

    def get_train_loader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True, drop_last=True, collate_fn=dict_collate_fn)

    def get_val_loader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False, drop_last=False, collate_fn=dict_collate_fn)

    def get_test_loader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False, drop_last=False, collate_fn=dict_collate_fn)

    def get_dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="synthetic_babi" if self.mode == "dev" else "babi",
            phase=6, modality="text", num_classes=self._nc,
            num_train_samples=self._n_train, num_val_samples=self._n_val,
            num_test_samples=self._n_test,
            input_shapes={"context": (self._context_len,), "question": (self._question_len,)},
            target_shape=(),
            description="Reasoning logic task dataset",
        )


# ============================================================================
# PHASE 7: Meta-Learning -- MetaPhaseLoader
# ============================================================================

class _FewShotEpisodeDataset(Dataset):
    """Synthetic few-shot episodes for meta-learning.

    Each episode contains N*K support samples and N*Q query samples
    drawn from N random class prototypes.
    """

    def __init__(self, num_episodes: int, n_way: int, k_shot: int,
                 q_queries: int, image_channels: int, image_size: int,
                 seed: int = 42):
        super().__init__()
        self.num_episodes = num_episodes
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.C = image_channels
        self.H = image_size
        self.W = image_size

        gen = torch.Generator()
        gen.manual_seed(seed)

        nk = n_way * k_shot
        nq = n_way * q_queries

        # Generate class prototypes for each episode
        # Each class has a different random prototype image
        self.support_x = torch.zeros(num_episodes, nk, image_channels, image_size, image_size)
        self.support_y = torch.zeros(num_episodes, nk, dtype=torch.long)
        self.query_x = torch.zeros(num_episodes, nq, image_channels, image_size, image_size)
        self.query_y = torch.zeros(num_episodes, nq, dtype=torch.long)

        for ep in range(num_episodes):
            prototypes = torch.randn(n_way, image_channels, image_size, image_size, generator=gen)
            # Support set
            for c in range(n_way):
                for k in range(k_shot):
                    idx = c * k_shot + k
                    noise = torch.randn(image_channels, image_size, image_size, generator=gen) * 0.3
                    self.support_x[ep, idx] = prototypes[c] + noise
                    self.support_y[ep, idx] = c
            # Query set
            for c in range(n_way):
                for q in range(q_queries):
                    idx = c * q_queries + q
                    noise = torch.randn(image_channels, image_size, image_size, generator=gen) * 0.3
                    self.query_x[ep, idx] = prototypes[c] + noise
                    self.query_y[ep, idx] = c

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "support_x": self.support_x[idx],  # [N*K, C, H, W]
            "support_y": self.support_y[idx],   # [N*K]
            "query_x": self.query_x[idx],       # [N*Q, C, H, W]
            "query_y": self.query_y[idx],        # [N*Q]
        }


class MetaPhaseLoader(BasePhaseLoader):
    """Phase 7 data loader: Meta-learning with few-shot episodes.

    Dev: Synthetic few-shot episodes from random class prototypes.
    Production: Omniglot / mini-ImageNet style episodes.
    """

    def __init__(self, config: DataConfig, mode: str = "dev"):
        super().__init__(config, mode)
        n = config.dev_num_samples
        n_way = config.dev_n_way
        k_shot = config.dev_k_shot
        q_queries = config.dev_q_queries

        if mode == "dev":
            C = config.dev_image_channels
            H = config.dev_image_size
        else:
            C = config.prod_image_channels
            H = config.prod_image_size

        full_ds = _FewShotEpisodeDataset(
            n, n_way, k_shot, q_queries, C, H, seed=config.split_seed
        )
        train_idx, val_idx, test_idx = _create_split_indices(
            n, config.val_ratio, config.test_ratio, config.split_seed
        )
        self.train_ds = Subset(full_ds, train_idx)
        self.val_ds = Subset(full_ds, val_idx)
        self.test_ds = Subset(full_ds, test_idx)
        self._n_way = n_way
        self._k_shot = k_shot
        self._q_queries = q_queries
        self._C = C
        self._H = H
        self._n_train = len(train_idx)
        self._n_val = len(val_idx)
        self._n_test = len(test_idx)

    def get_train_loader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True, drop_last=True, collate_fn=episode_collate_fn)

    def get_val_loader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False, drop_last=False, collate_fn=episode_collate_fn)

    def get_test_loader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False, drop_last=False, collate_fn=episode_collate_fn)

    def get_dataset_info(self) -> DatasetInfo:
        nk = self._n_way * self._k_shot
        nq = self._n_way * self._q_queries
        return DatasetInfo(
            name="synthetic_episodes" if self.mode == "dev" else "mini_imagenet",
            phase=7, modality="episodes", num_classes=self._n_way,
            num_train_samples=self._n_train, num_val_samples=self._n_val,
            num_test_samples=self._n_test,
            input_shapes={
                "support_x": (nk, self._C, self._H, self._H),
                "support_y": (nk,),
                "query_x": (nq, self._C, self._H, self._H),
                "query_y": (nq,),
            },
            target_shape=None,
            description="Meta-learning few-shot episode dataset",
        )

    def get_collate_fn(self) -> Optional[Callable]:
        return episode_collate_fn


# ============================================================================
# PHASE LOADER REGISTRY (convenience mapping)
# ============================================================================

PHASE_LOADERS: Dict[int, Type[BasePhaseLoader]] = {
    1: SNNPhaseLoader,
    2: EncoderPhaseLoader,
    3: HTMPhaseLoader,
    4: WorkspacePhaseLoader,
    5: ActiveInfPhaseLoader,
    6: ReasoningPhaseLoader,
    7: MetaPhaseLoader,
}

PHASE_NAMES: Dict[int, str] = {
    1: "SNN Core",
    2: "Encoders",
    3: "HTM",
    4: "Workspace",
    5: "Active Inference",
    6: "Reasoning",
    7: "Meta-Learning",
}


def get_phase_loader(phase: int, config: Optional[DataConfig] = None,
                     mode: str = "dev") -> BasePhaseLoader:
    """Convenience function to get a phase loader by phase number."""
    if phase not in PHASE_LOADERS:
        raise ValueError(f"Unknown phase {phase}. Valid phases: 1-7")
    if config is None:
        config = DataConfig(phase=phase, mode=mode, batch_size=32, dev_num_samples=500)
    return PHASE_LOADERS[phase](config, mode)


# ============================================================================
# SELF-TESTS
# ============================================================================

if __name__ == "__main__":
    import sys
    import traceback
    import time

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

    # Helper config
    cfg = DataConfig(batch_size=8, dev_num_samples=200)

    # ========== Phase 1: SNN ==========

    def test_snn_inherits_base():
        assert issubclass(SNNPhaseLoader, BasePhaseLoader)
    run_test("P1 SNNPhaseLoader inherits BasePhaseLoader", test_snn_inherits_base)

    def test_snn_constructor():
        loader = SNNPhaseLoader(cfg, mode="dev")
        assert loader.mode == "dev"
    run_test("P1 SNNPhaseLoader constructor", test_snn_constructor)

    def test_snn_train_loader():
        loader = SNNPhaseLoader(cfg, mode="dev")
        dl = loader.get_train_loader()
        assert isinstance(dl, DataLoader)
        batch = next(iter(dl))
        assert "input" in batch
        assert batch["input"].shape[0] == 8
        assert batch["input"].shape[1:] == (1, 28, 28)
    run_test("P1 SNNPhaseLoader train loader shape", test_snn_train_loader)

    def test_snn_val_loader():
        loader = SNNPhaseLoader(cfg, mode="dev")
        dl = loader.get_val_loader()
        batch = next(iter(dl))
        assert batch["input"].dtype == torch.float32
        assert batch["target"].dtype == torch.long
    run_test("P1 SNNPhaseLoader val loader dtypes", test_snn_val_loader)

    def test_snn_test_loader():
        loader = SNNPhaseLoader(cfg, mode="dev")
        dl = loader.get_test_loader()
        assert isinstance(dl, DataLoader)
    run_test("P1 SNNPhaseLoader test loader", test_snn_test_loader)

    def test_snn_dataset_info():
        loader = SNNPhaseLoader(cfg, mode="dev")
        info = loader.get_dataset_info()
        assert info.phase == 1
        assert info.modality == "vision"
        assert info.num_classes == 10
    run_test("P1 SNNPhaseLoader dataset info", test_snn_dataset_info)

    def test_snn_no_nan():
        loader = SNNPhaseLoader(cfg, mode="dev")
        batch = next(iter(loader.get_train_loader()))
        assert not torch.isnan(batch["input"]).any()
    run_test("P1 SNNPhaseLoader no NaN", test_snn_no_nan)

    # ========== Phase 2: Encoders ==========

    def test_encoder_inherits_base():
        assert issubclass(EncoderPhaseLoader, BasePhaseLoader)
    run_test("P2 EncoderPhaseLoader inherits BasePhaseLoader", test_encoder_inherits_base)

    def test_encoder_constructor():
        loader = EncoderPhaseLoader(cfg, mode="dev")
        assert loader.mode == "dev"
    run_test("P2 EncoderPhaseLoader constructor", test_encoder_constructor)

    def test_encoder_multimodal_batch():
        loader = EncoderPhaseLoader(cfg, mode="dev")
        batch = next(iter(loader.get_train_loader()))
        assert "vision" in batch
        assert "text" in batch
        assert "audio" in batch
        assert "target" in batch
        assert batch["vision"].shape == (8, 1, 28, 28)
        assert batch["text"].shape == (8, 128)
        assert batch["audio"].shape == (8, 64, 100)
        assert batch["text"].dtype == torch.long
    run_test("P2 EncoderPhaseLoader multimodal batch shapes", test_encoder_multimodal_batch)

    def test_encoder_info():
        loader = EncoderPhaseLoader(cfg, mode="dev")
        info = loader.get_dataset_info()
        assert info.phase == 2
        assert info.modality == "multimodal"
    run_test("P2 EncoderPhaseLoader dataset info", test_encoder_info)

    def test_encoder_collate():
        loader = EncoderPhaseLoader(cfg, mode="dev")
        assert loader.get_collate_fn() is not None
    run_test("P2 EncoderPhaseLoader has collate fn", test_encoder_collate)

    # ========== Phase 3: HTM ==========

    def test_htm_inherits_base():
        assert issubclass(HTMPhaseLoader, BasePhaseLoader)
    run_test("P3 HTMPhaseLoader inherits BasePhaseLoader", test_htm_inherits_base)

    def test_htm_constructor():
        loader = HTMPhaseLoader(cfg, mode="dev")
        assert loader.mode == "dev"
    run_test("P3 HTMPhaseLoader constructor", test_htm_constructor)

    def test_htm_sequence_batch():
        loader = HTMPhaseLoader(cfg, mode="dev")
        batch = next(iter(loader.get_train_loader()))
        assert "input" in batch
        assert "mask" in batch
        assert "target" in batch
        assert batch["input"].shape == (8, 50, 16)
        assert batch["mask"].shape == (8, 50)
        assert batch["mask"].dtype == torch.bool
        assert batch["target"].shape == (8, 50, 16)
    run_test("P3 HTMPhaseLoader sequence batch shapes", test_htm_sequence_batch)

    def test_htm_info():
        loader = HTMPhaseLoader(cfg, mode="dev")
        info = loader.get_dataset_info()
        assert info.phase == 3
        assert info.modality == "sequence"
    run_test("P3 HTMPhaseLoader dataset info", test_htm_info)

    def test_htm_has_anomaly_labels():
        loader = HTMPhaseLoader(cfg, mode="dev")
        batch = next(iter(loader.get_train_loader()))
        assert "anomaly" in batch
        assert batch["anomaly"].dtype == torch.long
    run_test("P3 HTMPhaseLoader anomaly labels", test_htm_has_anomaly_labels)

    def test_htm_mask_all_true():
        loader = HTMPhaseLoader(cfg, mode="dev")
        batch = next(iter(loader.get_train_loader()))
        assert batch["mask"].all()
    run_test("P3 HTMPhaseLoader mask all valid", test_htm_mask_all_true)

    # ========== Phase 4: Workspace ==========

    def test_workspace_inherits_base():
        assert issubclass(WorkspacePhaseLoader, BasePhaseLoader)
    run_test("P4 WorkspacePhaseLoader inherits BasePhaseLoader", test_workspace_inherits_base)

    def test_workspace_multimodal():
        loader = WorkspacePhaseLoader(cfg, mode="dev")
        batch = next(iter(loader.get_train_loader()))
        assert "vision" in batch
        assert "text" in batch
        assert "audio" in batch
        assert batch["vision"].shape == (8, 1, 28, 28)
        assert batch["text"].dtype == torch.long
    run_test("P4 WorkspacePhaseLoader multimodal batch", test_workspace_multimodal)

    def test_workspace_info():
        loader = WorkspacePhaseLoader(cfg, mode="dev")
        info = loader.get_dataset_info()
        assert info.phase == 4
        assert info.modality == "multimodal"
    run_test("P4 WorkspacePhaseLoader dataset info", test_workspace_info)

    # ========== Phase 5: Active Inference ==========

    def test_activeinf_inherits_base():
        assert issubclass(ActiveInfPhaseLoader, BasePhaseLoader)
    run_test("P5 ActiveInfPhaseLoader inherits BasePhaseLoader", test_activeinf_inherits_base)

    def test_activeinf_trajectory_batch():
        loader = ActiveInfPhaseLoader(cfg, mode="dev")
        batch = next(iter(loader.get_train_loader()))
        assert "state" in batch
        assert "action" in batch
        assert "reward" in batch
        assert "next_state" in batch
        assert "done" in batch
        assert batch["state"].shape == (8, 4)
        assert batch["action"].dtype == torch.long
        assert batch["reward"].dtype == torch.float32
        assert batch["done"].dtype == torch.float32
    run_test("P5 ActiveInfPhaseLoader trajectory batch", test_activeinf_trajectory_batch)

    def test_activeinf_done_values():
        loader = ActiveInfPhaseLoader(cfg, mode="dev")
        batch = next(iter(loader.get_train_loader()))
        done_vals = batch["done"].unique()
        for v in done_vals:
            assert v.item() in (0.0, 1.0)
    run_test("P5 ActiveInfPhaseLoader done values in {0, 1}", test_activeinf_done_values)

    def test_activeinf_info():
        loader = ActiveInfPhaseLoader(cfg, mode="dev")
        info = loader.get_dataset_info()
        assert info.phase == 5
        assert info.modality == "rl"
    run_test("P5 ActiveInfPhaseLoader dataset info", test_activeinf_info)

    # ========== Phase 6: Reasoning ==========

    def test_reasoning_inherits_base():
        assert issubclass(ReasoningPhaseLoader, BasePhaseLoader)
    run_test("P6 ReasoningPhaseLoader inherits BasePhaseLoader", test_reasoning_inherits_base)

    def test_reasoning_batch():
        loader = ReasoningPhaseLoader(cfg, mode="dev")
        batch = next(iter(loader.get_train_loader()))
        assert "context" in batch
        assert "question" in batch
        assert "target" in batch
        assert batch["context"].shape == (8, 256)
        assert batch["question"].shape == (8, 64)
        assert batch["context"].dtype == torch.long
        assert batch["question"].dtype == torch.long
        assert batch["target"].dtype == torch.long
    run_test("P6 ReasoningPhaseLoader batch shapes", test_reasoning_batch)

    def test_reasoning_info():
        loader = ReasoningPhaseLoader(cfg, mode="dev")
        info = loader.get_dataset_info()
        assert info.phase == 6
        assert info.modality == "text"
    run_test("P6 ReasoningPhaseLoader dataset info", test_reasoning_info)

    def test_reasoning_context_has_padding():
        loader = ReasoningPhaseLoader(cfg, mode="dev")
        batch = next(iter(loader.get_train_loader()))
        # At least some tokens should be padding (0)
        assert (batch["context"] == 0).any()
    run_test("P6 ReasoningPhaseLoader context has padding", test_reasoning_context_has_padding)

    # ========== Phase 7: Meta-Learning ==========

    def test_meta_inherits_base():
        assert issubclass(MetaPhaseLoader, BasePhaseLoader)
    run_test("P7 MetaPhaseLoader inherits BasePhaseLoader", test_meta_inherits_base)

    def test_meta_episode_batch():
        loader = MetaPhaseLoader(cfg, mode="dev")
        batch = next(iter(loader.get_train_loader()))
        assert "support_x" in batch
        assert "support_y" in batch
        assert "query_x" in batch
        assert "query_y" in batch
        # N=5, K=1 => support: 5 samples; Q=15 => query: 75 samples
        assert batch["support_x"].shape == (8, 5, 1, 28, 28)
        assert batch["support_y"].shape == (8, 5)
        assert batch["query_x"].shape == (8, 75, 1, 28, 28)
        assert batch["query_y"].shape == (8, 75)
    run_test("P7 MetaPhaseLoader episode batch shapes", test_meta_episode_batch)

    def test_meta_label_range():
        loader = MetaPhaseLoader(cfg, mode="dev")
        batch = next(iter(loader.get_train_loader()))
        assert batch["support_y"].min() >= 0
        assert batch["support_y"].max() < 5
        assert batch["query_y"].min() >= 0
        assert batch["query_y"].max() < 5
    run_test("P7 MetaPhaseLoader label range [0, N)", test_meta_label_range)

    def test_meta_info():
        loader = MetaPhaseLoader(cfg, mode="dev")
        info = loader.get_dataset_info()
        assert info.phase == 7
        assert info.modality == "episodes"
        assert info.num_classes == 5
    run_test("P7 MetaPhaseLoader dataset info", test_meta_info)

    # ========== Cross-Phase Tests ==========

    def test_all_phases_instantiate():
        for phase in range(1, 8):
            c = DataConfig(phase=phase, batch_size=4, dev_num_samples=100)
            loader = PHASE_LOADERS[phase](c, mode="dev")
            assert loader is not None
    run_test("All 7 phases instantiate", test_all_phases_instantiate)

    def test_all_phases_produce_batches():
        for phase in range(1, 8):
            c = DataConfig(phase=phase, batch_size=4, dev_num_samples=100)
            loader = PHASE_LOADERS[phase](c, mode="dev")
            batch = next(iter(loader.get_train_loader()))
            assert isinstance(batch, dict)
            assert len(batch) > 0
    run_test("All 7 phases produce batches", test_all_phases_produce_batches)

    def test_all_phases_have_dataset_info():
        for phase in range(1, 8):
            c = DataConfig(phase=phase, batch_size=4, dev_num_samples=100)
            loader = PHASE_LOADERS[phase](c, mode="dev")
            info = loader.get_dataset_info()
            assert isinstance(info, DatasetInfo)
            assert info.phase == phase
    run_test("All 7 phases have valid dataset info", test_all_phases_have_dataset_info)

    def test_all_phases_no_nan():
        for phase in range(1, 8):
            c = DataConfig(phase=phase, batch_size=4, dev_num_samples=100)
            loader = PHASE_LOADERS[phase](c, mode="dev")
            batch = next(iter(loader.get_train_loader()))
            for key, val in batch.items():
                if val.is_floating_point():
                    assert not torch.isnan(val).any(), f"Phase {phase} key {key} has NaN"
    run_test("All 7 phases no NaN in batches", test_all_phases_no_nan)

    def test_all_phases_no_inf():
        for phase in range(1, 8):
            c = DataConfig(phase=phase, batch_size=4, dev_num_samples=100)
            loader = PHASE_LOADERS[phase](c, mode="dev")
            batch = next(iter(loader.get_train_loader()))
            for key, val in batch.items():
                if val.is_floating_point():
                    assert not torch.isinf(val).any(), f"Phase {phase} key {key} has Inf"
    run_test("All 7 phases no Inf in batches", test_all_phases_no_inf)

    def test_get_phase_loader_convenience():
        loader = get_phase_loader(1)
        assert isinstance(loader, SNNPhaseLoader)
    run_test("get_phase_loader convenience function", test_get_phase_loader_convenience)

    def test_get_phase_loader_invalid():
        try:
            get_phase_loader(0)
            assert False, "Should raise ValueError"
        except ValueError:
            pass
    run_test("get_phase_loader invalid phase", test_get_phase_loader_invalid)

    def test_phase_loader_bad_mode():
        try:
            SNNPhaseLoader(cfg, mode="bad")
            assert False, "Should raise ValueError"
        except ValueError:
            pass
    run_test("Phase loader rejects bad mode", test_phase_loader_bad_mode)

    def test_dev_mode_fast():
        start = time.time()
        for phase in range(1, 8):
            c = DataConfig(phase=phase, batch_size=4, dev_num_samples=100)
            loader = PHASE_LOADERS[phase](c, mode="dev")
            _ = next(iter(loader.get_train_loader()))
        elapsed = time.time() - start
        assert elapsed < 30.0, f"All dev loaders took {elapsed:.1f}s (limit: 30s)"
    run_test("All dev loaders within time limit", test_dev_mode_fast)

    def test_sample_shape_consistency():
        for phase in range(1, 8):
            c = DataConfig(phase=phase, batch_size=4, dev_num_samples=100)
            loader = PHASE_LOADERS[phase](c, mode="dev")
            shapes = loader.get_sample_shape()
            assert isinstance(shapes, dict)
            assert len(shapes) > 0
    run_test("get_sample_shape consistent for all phases", test_sample_shape_consistency)

    def test_num_samples_consistency():
        c = DataConfig(batch_size=4, dev_num_samples=100)
        loader = SNNPhaseLoader(c, mode="dev")
        counts = loader.get_num_samples()
        assert counts["train"] + counts["val"] + counts["test"] == 100
    run_test("get_num_samples adds up to total", test_num_samples_consistency)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"PHASE LOADERS SELF-TESTS: {passed} passed, {failed} failed")
    print("=" * 60)
    for result in test_results:
        status = result[0]
        name = result[1]
        extra = f" -- {result[2]}" if len(result) > 2 else ""
        print(f"  [{status}] {name}{extra}")

    sys.exit(0 if failed == 0 else 1)
