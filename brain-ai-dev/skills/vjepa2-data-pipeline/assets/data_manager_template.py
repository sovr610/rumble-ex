"""
data_manager_template.py
=========================
DataManager: unified factory for building video DataLoaders.

Public API
----------
DataManager(config: DataConfig)
    .build_train_loader(mask_collator=None) -> DataLoader
    .build_eval_loader()                    -> DataLoader

The DataManager:
  - Constructs VideoDataset(s) from config.data_paths
  - Builds DistributedWeightedSampler for training
  - Attaches mask_collator as collate_fn when provided
  - Seeds DataLoader workers deterministically via LCG algorithm
  - Supports multi-source mixing via per-source weights
"""

from __future__ import annotations

import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Internal imports (from same package directory in production)
try:
    from video_dataset_template import VideoDataset
    from video_transforms_template import VideoTransformPipeline
    from distributed_sampler_template import DistributedWeightedSampler, ConcatIndices
    from data_config_template import DataConfig, AugConfig
except ImportError:
    # Allow standalone use: stubs will be replaced in production
    VideoDataset = None
    VideoTransformPipeline = None
    DistributedWeightedSampler = None
    ConcatIndices = None
    DataConfig = None
    AugConfig = None


# ---------------------------------------------------------------------------
# LCG Worker Seeding
# ---------------------------------------------------------------------------

# POSIX LCG parameters (Numerical Recipes)
_LCG_A = 1664525
_LCG_C = 1013904223
_LCG_M = 2 ** 32


def worker_init_fn(worker_id: int) -> None:
    """
    Deterministic LCG-based seeding for DataLoader worker processes.

    Each worker receives a unique, reproducible seed derived from:
      - PyTorch's worker seed (set by DataLoader from generator)
      - The worker's ID

    Formula: seed = (A * (base + worker_id) + C) % M

    Sets seeds for: random, numpy, torch.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        # Single-process mode — use default seeding
        return

    base_seed = int(worker_info.seed) % (2 ** 31)
    seed = (_LCG_A * (base_seed + worker_id) + _LCG_C) % _LCG_M

    random.seed(seed)
    np.random.seed(seed % (2 ** 31))
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Default collate with optional mask collator
# ---------------------------------------------------------------------------

def _default_collate_with_mask(
    batch: List[Dict[str, Any]],
    mask_collator: Optional[Callable] = None,
) -> Any:
    """
    Collate a batch of samples from VideoDataset.

    When mask_collator is provided, it receives the collated batch dict
    and returns (batch, masks_enc, masks_pred).

    Otherwise, returns the standard collated dict with batched tensors.
    """
    # Collate video tensors: List[[C,T,H,W]] -> [B,C,T,H,W]
    videos = torch.stack([s["video"] for s in batch], dim=0)
    labels = torch.tensor([s.get("label", -1) for s in batch], dtype=torch.long)
    paths  = [s.get("path", "") for s in batch]

    collated = {"video": videos, "label": labels, "path": paths}

    if mask_collator is not None:
        return mask_collator(collated)
    return collated


def make_collate_fn(
    mask_collator: Optional[Callable] = None,
) -> Callable:
    """Returns a collate function closure capturing the mask_collator."""
    def collate(batch):
        return _default_collate_with_mask(batch, mask_collator)
    return collate


# ---------------------------------------------------------------------------
# Multi-source dataset builder
# ---------------------------------------------------------------------------

def build_multi_source_dataset(
    config: "DataConfig",
    transform: Optional[Callable] = None,
) -> Tuple[Dataset, List[float]]:
    """
    Build a combined dataset from multiple data_paths with per-source weights.

    Returns
    -------
    dataset : Dataset
        A single Dataset whose indices span all sources sequentially.
    sample_weights : List[float]
        Per-sample weights for DistributedWeightedSampler.
    """
    data_paths = config.data_paths if config.data_paths else []
    weights    = list(config.data_weights) if config.data_weights else []

    # Default: uniform weights if not specified
    if not weights:
        weights = [1.0] * max(1, len(data_paths))

    # Pad or trim weights to match data_paths
    if len(weights) < len(data_paths):
        weights.extend([1.0] * (len(data_paths) - len(weights)))
    weights = weights[:len(data_paths)]

    if not data_paths:
        # Synthetic mode: create a single placeholder dataset
        dataset = VideoDataset(
            data_paths=[],
            clip_mode=config.clip_mode,
            frames_per_clip=config.frames_per_clip,
            target_fps=config.target_fps,
            clip_duration_sec=getattr(config, "clip_duration_sec", 3.2),
            frame_step=getattr(config, "frame_step", 4),
            transform=transform,
            img_size=config.img_size,
            use_gpu_decode=getattr(config, "use_gpu_decode", False),
        )
        sample_weights = [1.0] * len(dataset)
        return dataset, sample_weights

    if len(data_paths) == 1:
        # Single source: no multi-source overhead
        dataset = VideoDataset(
            data_paths=data_paths,
            clip_mode=config.clip_mode,
            frames_per_clip=config.frames_per_clip,
            target_fps=config.target_fps,
            clip_duration_sec=getattr(config, "clip_duration_sec", 3.2),
            frame_step=getattr(config, "frame_step", 4),
            transform=transform,
            img_size=config.img_size,
            use_gpu_decode=getattr(config, "use_gpu_decode", False),
        )
        n = len(dataset)
        w = weights[0]
        sample_weights = [w / n if n > 0 else 0.0] * n
        return dataset, sample_weights

    # Multiple sources: build individual datasets then combine
    sub_datasets = []
    for path in data_paths:
        ds = VideoDataset(
            data_paths=[path],
            clip_mode=config.clip_mode,
            frames_per_clip=config.frames_per_clip,
            target_fps=config.target_fps,
            clip_duration_sec=getattr(config, "clip_duration_sec", 3.2),
            frame_step=getattr(config, "frame_step", 4),
            transform=transform,
            img_size=config.img_size,
            use_gpu_decode=getattr(config, "use_gpu_decode", False),
        )
        sub_datasets.append(ds)

    # Build per-sample weights by expanding per-source weights
    sample_weights: List[float] = []
    for ds, w in zip(sub_datasets, weights):
        n = len(ds)
        per_sample = w / n if n > 0 else 0.0
        sample_weights.extend([per_sample] * n)

    # Concatenate datasets using ConcatDataset-style wrapper
    combined = MultiSourceConcatDataset(sub_datasets)
    return combined, sample_weights


class MultiSourceConcatDataset(Dataset):
    """
    Concatenates multiple VideoDataset instances behind a unified index.
    Uses ConcatIndices for O(log n) index routing.
    """

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        sizes = [len(d) for d in datasets]
        self._index_map = ConcatIndices(sizes)

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, global_idx: int) -> Dict[str, Any]:
        dataset_idx, sample_idx = self._index_map[global_idx]
        return self.datasets[dataset_idx][sample_idx]


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------

class DataManager:
    """
    Unified factory for constructing train and eval DataLoaders.

    Handles:
    - Multi-source dataset construction with per-source weights
    - Transform pipeline composition from AugConfig
    - DistributedWeightedSampler for distributed training
    - Worker seeding via LCG algorithm
    - MaskCollator integration as collate_fn

    Parameters
    ----------
    config : DataConfig
        Data loading configuration.
    aug_config : AugConfig, optional
        Augmentation configuration. If None, uses AugConfig defaults.
    rank : int
        Rank of this process in distributed training (default 0).
    world_size : int
        Total number of processes (default 1).
    seed : int
        Base random seed for reproducibility.
    """

    def __init__(
        self,
        config: "DataConfig",
        aug_config: Optional["AugConfig"] = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        self.config = config
        self.aug_config = aug_config or (AugConfig() if AugConfig else {})
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

    def build_train_loader(
        self,
        mask_collator: Optional[Callable] = None,
        epoch: int = 0,
    ) -> DataLoader:
        """
        Build the training DataLoader with weighted distributed sampling.

        Parameters
        ----------
        mask_collator : Optional[Callable]
            If provided, used as collate_fn. Receives the batched dict and
            returns (batch, masks_enc, masks_pred).
        epoch : int
            Current epoch for sampler shuffle reproducibility.

        Returns
        -------
        DataLoader
            Configured DataLoader ready for training.
        """
        # Build transform pipeline
        pipeline = VideoTransformPipeline(self.aug_config, img_size=self.config.img_size)
        train_transform = pipeline.get_train_transform()

        # Build dataset
        dataset, sample_weights = build_multi_source_dataset(
            self.config, transform=train_transform
        )

        # Build distributed weighted sampler
        sampler = DistributedWeightedSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            rank=self.rank,
            world_size=self.world_size,
            seed=self.seed,
            replacement=False,
        )
        sampler.set_epoch(epoch)

        # Determine DataLoader kwargs
        num_workers = self.config.num_workers
        persistent = (
            getattr(self.config, "persistent_workers", True) and num_workers > 0
        )
        prefetch = getattr(self.config, "prefetch_factor", 2) if num_workers > 0 else None

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=make_collate_fn(mask_collator),
            pin_memory=getattr(self.config, "pin_memory", True),
            drop_last=True,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            worker_init_fn=worker_init_fn,
        )

        return loader

    def build_eval_loader(self, shuffle: bool = False) -> DataLoader:
        """
        Build the evaluation DataLoader (deterministic, no distributed sampling).

        Parameters
        ----------
        shuffle : bool
            Whether to shuffle the eval set (default False).

        Returns
        -------
        DataLoader
            Configured DataLoader for evaluation.
        """
        # Eval uses deterministic center-crop transform
        pipeline = VideoTransformPipeline(self.aug_config, img_size=self.config.img_size)
        eval_transform = pipeline.get_eval_transform()

        dataset, _ = build_multi_source_dataset(self.config, transform=eval_transform)

        num_workers = self.config.num_workers
        persistent = (
            getattr(self.config, "persistent_workers", True) and num_workers > 0
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=make_collate_fn(None),
            pin_memory=getattr(self.config, "pin_memory", False),
            drop_last=False,
            persistent_workers=persistent,
            worker_init_fn=worker_init_fn,
        )

        return loader

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Return metadata about the configured dataset(s).
        """
        dataset, sample_weights = build_multi_source_dataset(self.config)
        return {
            "num_samples": len(dataset),
            "num_sources": max(1, len(self.config.data_paths)),
            "clip_mode": self.config.clip_mode,
            "frames_per_clip": self.config.frames_per_clip,
            "img_size": self.config.img_size,
            "batch_size": self.config.batch_size,
            "batches_per_epoch": len(dataset) // (self.config.batch_size * self.world_size),
        }

    def __repr__(self) -> str:
        info = self.get_dataset_info()
        return (
            f"DataManager("
            f"samples={info['num_samples']}, "
            f"sources={info['num_sources']}, "
            f"rank={self.rank}/{self.world_size}, "
            f"batch={self.config.batch_size})"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("DataManager Self-Tests")
    print("=" * 60)

    PASS = "[PASS]"
    FAIL = "[FAIL]"
    errors = []

    # ------------------------------------------------------------------
    # Check all required imports are available
    # ------------------------------------------------------------------
    if any(x is None for x in [VideoDataset, VideoTransformPipeline,
                                DistributedWeightedSampler, ConcatIndices,
                                DataConfig, AugConfig]):
        print("[WARN] Some imports failed; running with reduced test coverage")
        print("       Ensure all template files are in the same directory")

    # ------------------------------------------------------------------
    # Build minimal configs for testing.
    # Use batch_size=1 to avoid drop_last discarding the single synthetic
    # sample when testing with the train loader.
    # ------------------------------------------------------------------
    if DataConfig is not None:
        config = DataConfig(
            data_paths=[],       # Synthetic mode (1 record)
            clip_mode="fps",
            frames_per_clip=4,
            target_fps=10,
            img_size=64,
            num_workers=0,       # Main process for test speed
            batch_size=1,        # 1 so drop_last keeps the single synthetic sample
            pin_memory=False,
            persistent_workers=False,
        )
        aug = AugConfig(
            crop_scale=(0.5, 1.0),
            horizontal_flip=True,
            auto_augment=False,
            motion_shift=False,
            random_erasing=0.0,
        )
    else:
        # Fallback: build config-like objects manually
        class _Cfg:
            data_paths = []
            data_weights = []
            clip_mode = "fps"
            frames_per_clip = 4
            target_fps = 10
            clip_duration_sec = 3.2
            frame_step = 4
            img_size = 64
            num_workers = 0
            batch_size = 1
            pin_memory = False
            persistent_workers = False
            prefetch_factor = None
            use_gpu_decode = False

        class _Aug:
            crop_scale = (0.5, 1.0)
            crop_ratio = (0.75, 1.33)
            horizontal_flip = True
            auto_augment = False
            rand_augment_n = 2
            rand_augment_m = 9
            motion_shift = False
            random_erasing = 0.0
            normalize_mean = (0.485, 0.456, 0.406)
            normalize_std  = (0.229, 0.224, 0.225)

        config = _Cfg()
        aug    = _Aug()

    manager = DataManager(config, aug_config=aug, rank=0, world_size=1, seed=42)

    # -------------------------------------------------------------------
    # Test 1: Build train loader without errors
    # -------------------------------------------------------------------
    try:
        train_loader = manager.build_train_loader()
        print(f"{PASS} Test 1: build_train_loader() succeeded")
    except Exception as e:
        msg = f"Test 1 FAILED: {e}"
        print(f"{FAIL} {msg}")
        errors.append(msg)
        train_loader = None

    # -------------------------------------------------------------------
    # Test 2: Train loader yields correct batch structure
    # Note: with drop_last=True and batch_size=1, there is exactly 1 batch.
    # -------------------------------------------------------------------
    batch = None
    if train_loader is not None:
        try:
            batch_iter = iter(train_loader)
            batch = next(batch_iter)
            # batch is a dict (no mask_collator used here)
            if isinstance(batch, dict) and "video" in batch:
                print(f"{PASS} Test 2: Batch has 'video' key")
            elif isinstance(batch, tuple):
                # mask_collator returned tuple -- unlikely without one, but handle
                batch = batch[0] if isinstance(batch[0], dict) else None
                if batch is not None and "video" in batch:
                    print(f"{PASS} Test 2: Batch (from tuple) has 'video' key")
                else:
                    msg = "Test 2 FAILED: no 'video' key in batch"
                    print(f"{FAIL} {msg}")
                    errors.append(msg)
                    batch = None
            else:
                msg = f"Test 2 FAILED: unexpected batch type={type(batch)}"
                print(f"{FAIL} {msg}")
                errors.append(msg)
                batch = None
        except StopIteration:
            msg = "Test 2 FAILED: train_loader is empty (drop_last dropped all batches)"
            print(f"{FAIL} {msg}")
            errors.append(msg)
        except Exception as e:
            msg = f"Test 2 FAILED: {e}"
            print(f"{FAIL} {msg}")
            errors.append(msg)
            batch = None

    # -------------------------------------------------------------------
    # Test 3: Batch video shape is [B, C, T, H, W]
    # -------------------------------------------------------------------
    if batch is not None and isinstance(batch, dict) and "video" in batch:
        v = batch["video"]
        B, C, T, H, W = v.shape
        expected = (config.batch_size, 3, config.frames_per_clip,
                    config.img_size, config.img_size)
        actual = (B, C, T, H, W)
        if actual == expected:
            print(f"{PASS} Test 3: Video batch shape {actual}")
        else:
            msg = f"Test 3 FAILED: expected {expected}, got {actual}"
            print(f"{FAIL} {msg}")
            errors.append(msg)

    # -------------------------------------------------------------------
    # Test 4: Build eval loader
    # -------------------------------------------------------------------
    try:
        eval_loader = manager.build_eval_loader()
        print(f"{PASS} Test 4: build_eval_loader() succeeded")
    except Exception as e:
        msg = f"Test 4 FAILED: {e}"
        print(f"{FAIL} {msg}")
        errors.append(msg)
        eval_loader = None

    # -------------------------------------------------------------------
    # Test 5: Eval loader yields batch with correct shape
    # Eval loader has drop_last=False, so the single synthetic sample is returned.
    # -------------------------------------------------------------------
    if eval_loader is not None:
        try:
            eval_batch = next(iter(eval_loader))
            if isinstance(eval_batch, tuple):
                eval_batch = eval_batch[0]
            v = eval_batch["video"]
            # B may be 1 (single sample) for eval loader with drop_last=False
            B_actual = v.shape[0]
            C, T, H, W = v.shape[1], v.shape[2], v.shape[3], v.shape[4]
            shape_ok = (C == 3 and T == config.frames_per_clip and
                        H == config.img_size and W == config.img_size and
                        B_actual >= 1)
            if shape_ok:
                print(f"{PASS} Test 5: Eval batch shape {tuple(v.shape)}")
            else:
                msg = f"Test 5 FAILED: shape {tuple(v.shape)} has wrong C/T/H/W"
                print(f"{FAIL} {msg}")
                errors.append(msg)
        except StopIteration:
            msg = "Test 5 FAILED: eval_loader is empty"
            print(f"{FAIL} {msg}")
            errors.append(msg)
        except Exception as e:
            msg = f"Test 5 FAILED: {e}"
            print(f"{FAIL} {msg}")
            errors.append(msg)

    # -------------------------------------------------------------------
    # Test 6: Worker seeding LCG produces unique seeds
    # -------------------------------------------------------------------
    base = 12345
    seeds = set()
    for wid in range(8):
        s = (_LCG_A * (base + wid) + _LCG_C) % _LCG_M
        seeds.add(s)
    if len(seeds) == 8:
        print(f"{PASS} Test 6: LCG produces 8 unique worker seeds")
    else:
        msg = f"Test 6 FAILED: only {len(seeds)} unique seeds from 8 workers"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 7: get_dataset_info returns expected keys
    # -------------------------------------------------------------------
    try:
        info = manager.get_dataset_info()
        required_keys = {"num_samples", "clip_mode", "frames_per_clip", "batch_size"}
        missing = required_keys - set(info.keys())
        if not missing:
            print(f"{PASS} Test 7: get_dataset_info has all required keys")
        else:
            msg = f"Test 7 FAILED: missing info keys {missing}"
            print(f"{FAIL} {msg}")
            errors.append(msg)
    except Exception as e:
        msg = f"Test 7 FAILED: {e}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 8: mask_collator integration
    # -------------------------------------------------------------------
    def dummy_mask_collator(batch_dict):
        """Minimal mask collator: returns (batch, [enc_mask], [pred_mask])."""
        B = batch_dict["video"].shape[0]
        enc_mask  = [torch.ones(B, 10)]   # dummy mask tokens
        pred_mask = [torch.ones(B, 20)]
        return batch_dict, enc_mask, pred_mask

    # Test mask_collator using eval_loader (drop_last=False guarantees a batch)
    try:
        masked_eval_loader = manager.build_eval_loader()
        # Patch the collate_fn by rebuilding the loader with mask_collator
        from torch.utils.data import DataLoader
        pipeline_t = VideoTransformPipeline(aug, img_size=config.img_size)
        test_ds, test_w = build_multi_source_dataset(config, transform=pipeline_t.get_eval_transform())
        masked_test_loader = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=make_collate_fn(dummy_mask_collator),
            drop_last=False,
        )
        result = next(iter(masked_test_loader))
        # Should be a tuple (batch, enc_masks, pred_masks)
        if isinstance(result, tuple) and len(result) == 3:
            print(f"{PASS} Test 8: mask_collator returns (batch, enc, pred) tuple")
        else:
            msg = f"Test 8 FAILED: result type={type(result)}, len={len(result) if hasattr(result,'__len__') else 'N/A'}"
            print(f"{FAIL} {msg}")
            errors.append(msg)
    except StopIteration:
        msg = "Test 8 FAILED: masked loader is empty"
        print(f"{FAIL} {msg}")
        errors.append(msg)
    except Exception as e:
        msg = f"Test 8 FAILED: {e}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print()
    if errors:
        print(f"FAILED: {len(errors)} test(s) failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All tests passed.")
