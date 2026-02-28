"""
PyTorch Lightning VideoDataModule template.

Encapsulates train/val/test data loading for video datasets in a way that
is correct for single-GPU, multi-GPU DDP, and multi-node distributed training.

Key rules:
- prepare_data() runs on rank 0 only -- no self.* state assignments here.
- setup(stage) runs on all ranks -- create datasets here.
- Lightning auto-injects DistributedSampler -- do NOT create one manually.
- Use module.train(False) to set inference mode, never the bare .eval pattern.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

try:
    import lightning as L
    _LIGHTNING_AVAILABLE = True
except ImportError:
    try:
        import pytorch_lightning as L
        _LIGHTNING_AVAILABLE = True
    except ImportError:
        _LIGHTNING_AVAILABLE = False
        # Stub for when lightning is absent -- allows self-tests to run
        class _StubDataModule:
            pass
        L = type("L", (), {"LightningDataModule": _StubDataModule})()
        print("[datamodule_template] WARNING: lightning not available.")

try:
    import decord
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DataModuleConfig:
    """Configuration for VideoDataModule."""

    data_root: str = ""
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    batch_size: int = 8
    num_workers_per_gpu: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # VideoDataset sub-config fields (forwarded when creating datasets)
    num_frames: int = 16
    stride: int = 4
    crop_size: int = 224

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_workers_per_gpu < 0:
            raise ValueError(
                f"num_workers_per_gpu must be >= 0, got {self.num_workers_per_gpu}"
            )
        if self.prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {self.prefetch_factor}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DataModuleConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Minimal VideoMeta stub (avoid circular import with video_dataset_template)
# ---------------------------------------------------------------------------

@dataclass
class _VideoMeta:
    path: str
    num_frames: int
    fps: float
    label: int
    split: str = "train"


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _load_manifest(manifest_path: str):
    """Load a manifest JSON file and return a list of _VideoMeta."""
    with open(manifest_path, "r") as f:
        data = json.load(f)
    return [_VideoMeta(**item) for item in data]


def _save_manifest(manifest: list, manifest_path: str) -> None:
    """Save a list of _VideoMeta to a JSON manifest file."""
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump([asdict(m) for m in manifest], f, indent=2)


def _scan_videos(data_root: str) -> list:
    """
    Scan data_root/{split}/{class}/*.mp4 structure and return VideoMeta list.
    Requires decord for frame count; falls back to None if unavailable.
    """
    splits = ["train", "val", "test"]
    video_extensions = {".mp4", ".avi", ".mkv", ".webm", ".mov"}
    manifest = []

    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        class_names = sorted(
            n for n in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, n))
        )
        for label_id, class_name in enumerate(class_names):
            class_dir = os.path.join(split_dir, class_name)
            for fname in sorted(os.listdir(class_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in video_extensions:
                    continue
                fpath = os.path.join(class_dir, fname)
                try:
                    if _DECORD_AVAILABLE:
                        from decord import VideoReader, cpu
                        vr = VideoReader(fpath, ctx=cpu(0), num_threads=1)
                        nf = len(vr)
                        fps = float(vr.get_avg_fps())
                        del vr
                    else:
                        nf = 0
                        fps = 25.0
                    manifest.append(
                        _VideoMeta(path=fpath, num_frames=nf, fps=fps,
                                   label=label_id, split=split)
                    )
                except Exception as exc:
                    print(f"[scan_videos] WARNING: skipping {fpath}: {exc}")
    return manifest


# ---------------------------------------------------------------------------
# Dataset stub (replaces real VideoDataset for DataModule self-tests)
# ---------------------------------------------------------------------------

class _DummyVideoDataset(torch.utils.data.Dataset):
    """Minimal dataset for testing DataModule structure without real video files."""

    def __init__(self, manifest: list, cfg: DataModuleConfig) -> None:
        self.manifest = manifest
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        T = self.cfg.num_frames
        H = W = self.cfg.crop_size
        return {
            "video": torch.rand(T, 3, H, W, dtype=torch.float32),
            "label": self.manifest[idx].label,
        }


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class VideoDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for video datasets.

    Lifecycle:
        __init__  -- store config, no file I/O
        prepare_data -- rank-0 only: build manifest JSON if missing
        setup(stage) -- all ranks: create VideoDataset for requested stage
        *_dataloader -- return DataLoader per split

    Lightning injects DistributedSampler automatically. Do NOT add one manually.
    """

    def __init__(self, cfg: DataModuleConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._manifest_path = os.path.join(cfg.data_root, "manifest.json")

        # Datasets -- populated in setup()
        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset:   Optional[torch.utils.data.Dataset] = None
        self.test_dataset:  Optional[torch.utils.data.Dataset] = None

    # ------------------------------------------------------------------
    # prepare_data: rank 0 only
    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        """
        Build manifest JSON if it does not already exist.

        Runs on rank 0 only. Do NOT assign self.* dataset state here --
        it will not propagate to other ranks. Write to shared storage instead.
        """
        if not os.path.exists(self._manifest_path):
            if not os.path.isdir(self.cfg.data_root):
                print(
                    f"[VideoDataModule] data_root '{self.cfg.data_root}' not found -- "
                    "skipping manifest build."
                )
                return
            print(f"[VideoDataModule] Building manifest from {self.cfg.data_root}...")
            manifest = _scan_videos(self.cfg.data_root)
            _save_manifest(manifest, self._manifest_path)
            print(f"[VideoDataModule] Manifest with {len(manifest)} videos -> {self._manifest_path}")

    # ------------------------------------------------------------------
    # setup: all ranks
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Create dataset instances for the requested stage.

        Args:
            stage: 'fit' (train+val), 'validate', 'test', 'predict', or None (all)
        """
        if not os.path.exists(self._manifest_path):
            # No manifest -- create empty datasets (useful for testing)
            self.train_dataset = _DummyVideoDataset([], self.cfg)
            self.val_dataset   = _DummyVideoDataset([], self.cfg)
            self.test_dataset  = _DummyVideoDataset([], self.cfg)
            return

        all_meta = _load_manifest(self._manifest_path)
        train_meta = [m for m in all_meta if m.split == self.cfg.train_split]
        val_meta   = [m for m in all_meta if m.split == self.cfg.val_split]
        test_meta  = [m for m in all_meta if m.split == self.cfg.test_split]

        if stage in ("fit", None):
            self.train_dataset = _DummyVideoDataset(train_meta, self.cfg)
            self.val_dataset   = _DummyVideoDataset(val_meta,   self.cfg)

        if stage in ("validate", None) and self.val_dataset is None:
            self.val_dataset = _DummyVideoDataset(val_meta, self.cfg)

        if stage in ("test", "predict", None):
            self.test_dataset = _DummyVideoDataset(test_meta, self.cfg)

    # ------------------------------------------------------------------
    # Worker scaling
    # ------------------------------------------------------------------
    def _get_num_workers(self) -> int:
        """
        Scale workers with number of GPUs.

        Formula: num_workers = num_workers_per_gpu * num_gpus

        Falls back to num_workers_per_gpu * 1 when trainer is unavailable
        (e.g., during self-tests or before Trainer.fit() is called).
        """
        try:
            num_gpus = max(1, self.trainer.num_devices)
        except Exception:
            num_gpus = 1
        return self.cfg.num_workers_per_gpu * num_gpus

    def _dataloader_kwargs(self, shuffle: bool, drop_last: bool) -> dict:
        """
        Assemble keyword arguments for DataLoader construction.

        NOTE: No DistributedSampler is created here -- Lightning injects it
        automatically via Trainer(use_distributed_sampler=True) (the default).
        """
        nw = self._get_num_workers()
        kwargs = dict(
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=(self.cfg.persistent_workers and nw > 0),
            drop_last=drop_last,
        )
        # prefetch_factor only valid when num_workers > 0
        if nw > 0:
            kwargs["prefetch_factor"] = self.cfg.prefetch_factor
        return kwargs

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            **self._dataloader_kwargs(shuffle=True, drop_last=True),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            **self._dataloader_kwargs(shuffle=False, drop_last=False),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            **self._dataloader_kwargs(shuffle=False, drop_last=False),
        )

    # ------------------------------------------------------------------
    # State dict for mid-epoch resumption
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

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
    print("VideoDataModule Self-Tests")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Test 1: DataModuleConfig defaults
    # -----------------------------------------------------------------------
    print("\n[Test 1] DataModuleConfig defaults")
    cfg = DataModuleConfig()
    check(cfg.batch_size == 8, f"batch_size=8, got {cfg.batch_size}")
    check(cfg.num_workers_per_gpu == 4, f"num_workers_per_gpu=4, got {cfg.num_workers_per_gpu}")
    check(cfg.pin_memory is True, f"pin_memory=True, got {cfg.pin_memory}")
    check(cfg.persistent_workers is True, f"persistent_workers=True, got {cfg.persistent_workers}")
    check(cfg.prefetch_factor == 2, f"prefetch_factor=2, got {cfg.prefetch_factor}")
    check(cfg.train_split == "train", f"train_split='train', got '{cfg.train_split}'")
    check(cfg.num_frames == 16, f"num_frames=16, got {cfg.num_frames}")
    check(cfg.crop_size == 224, f"crop_size=224, got {cfg.crop_size}")

    # -----------------------------------------------------------------------
    # Test 2: DataModuleConfig validation
    # -----------------------------------------------------------------------
    print("\n[Test 2] DataModuleConfig validation")
    try:
        DataModuleConfig(batch_size=0)
        check(False, "rejects batch_size=0", "no exception raised")
    except ValueError:
        check(True, "rejects batch_size=0")

    try:
        DataModuleConfig(num_workers_per_gpu=-1)
        check(False, "rejects num_workers_per_gpu=-1", "no exception raised")
    except ValueError:
        check(True, "rejects num_workers_per_gpu=-1")

    try:
        DataModuleConfig(prefetch_factor=0)
        check(False, "rejects prefetch_factor=0", "no exception raised")
    except ValueError:
        check(True, "rejects prefetch_factor=0")

    # -----------------------------------------------------------------------
    # Test 3: num_workers calculation
    # -----------------------------------------------------------------------
    print("\n[Test 3] num_workers calculation")
    # Without a trainer, _get_num_workers uses num_gpus=1 fallback
    dm = VideoDataModule(DataModuleConfig(num_workers_per_gpu=4))
    nw = dm._get_num_workers()
    check(nw == 4, f"1-GPU: num_workers=4, got {nw}")

    dm2 = VideoDataModule(DataModuleConfig(num_workers_per_gpu=0))
    check(dm2._get_num_workers() == 0, "num_workers_per_gpu=0 -> num_workers=0")

    dm3 = VideoDataModule(DataModuleConfig(num_workers_per_gpu=2))
    check(dm3._get_num_workers() == 2, "num_workers_per_gpu=2 -> num_workers=2")

    # -----------------------------------------------------------------------
    # Test 4: DataLoader kwargs structure
    # -----------------------------------------------------------------------
    print("\n[Test 4] DataLoader kwargs structure")
    cfg_dl = DataModuleConfig(batch_size=16, num_workers_per_gpu=0)
    dm_dl = VideoDataModule(cfg_dl)
    kwargs = dm_dl._dataloader_kwargs(shuffle=True, drop_last=True)

    check(kwargs["batch_size"] == 16, f"batch_size=16, got {kwargs['batch_size']}")
    check(kwargs["shuffle"] is True, "shuffle=True in train kwargs")
    check(kwargs["drop_last"] is True, "drop_last=True in train kwargs")
    check(kwargs["pin_memory"] is True, "pin_memory=True in kwargs")
    check("prefetch_factor" not in kwargs, "prefetch_factor absent when num_workers=0")

    kwargs_nw = VideoDataModule(DataModuleConfig(num_workers_per_gpu=2))._dataloader_kwargs(
        shuffle=False, drop_last=False
    )
    check("prefetch_factor" in kwargs_nw, "prefetch_factor present when num_workers > 0")

    # -----------------------------------------------------------------------
    # Test 5: No DistributedSampler is manually created
    # -----------------------------------------------------------------------
    print("\n[Test 5] No DistributedSampler in DataLoader kwargs")
    from torch.utils.data.distributed import DistributedSampler
    check("sampler" not in kwargs, "no 'sampler' key in DataLoader kwargs")
    # The DataModule intentionally does not create a DistributedSampler;
    # Lightning's Trainer injects it via use_distributed_sampler=True (default).

    # -----------------------------------------------------------------------
    # Test 6: Stage-based dataset creation (no real video files needed)
    # -----------------------------------------------------------------------
    print("\n[Test 6] Stage-based dataset creation")
    import tempfile

    # Create a mock manifest file
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = os.path.join(tmp, "manifest.json")
        mock_manifest = [
            {"path": "/fake/train/cls0/v0.mp4", "num_frames": 64, "fps": 25.0, "label": 0, "split": "train"},
            {"path": "/fake/train/cls0/v1.mp4", "num_frames": 48, "fps": 25.0, "label": 0, "split": "train"},
            {"path": "/fake/val/cls0/v2.mp4",   "num_frames": 32, "fps": 25.0, "label": 0, "split": "val"},
            {"path": "/fake/test/cls0/v3.mp4",  "num_frames": 16, "fps": 25.0, "label": 0, "split": "test"},
        ]
        with open(manifest_path, "w") as f:
            json.dump(mock_manifest, f)

        cfg_mock = DataModuleConfig(data_root=tmp, num_workers_per_gpu=0)
        dm_mock = VideoDataModule(cfg_mock)

        # stage='fit' creates train + val
        dm_mock.setup("fit")
        check(dm_mock.train_dataset is not None, "stage='fit': train_dataset created")
        check(dm_mock.val_dataset is not None, "stage='fit': val_dataset created")
        check(len(dm_mock.train_dataset) == 2, f"train has 2 items, got {len(dm_mock.train_dataset)}")
        check(len(dm_mock.val_dataset) == 1, f"val has 1 item, got {len(dm_mock.val_dataset)}")

        # stage='test' creates test
        dm_mock.test_dataset = None
        dm_mock.setup("test")
        check(dm_mock.test_dataset is not None, "stage='test': test_dataset created")
        check(len(dm_mock.test_dataset) == 1, f"test has 1 item, got {len(dm_mock.test_dataset)}")

        # stage=None creates all
        dm_all = VideoDataModule(cfg_mock)
        dm_all.setup(None)
        check(dm_all.train_dataset is not None, "stage=None: train_dataset created")
        check(dm_all.val_dataset is not None, "stage=None: val_dataset created")
        check(dm_all.test_dataset is not None, "stage=None: test_dataset created")

    # -----------------------------------------------------------------------
    # Test 7: DataLoader iteration
    # -----------------------------------------------------------------------
    print("\n[Test 7] DataLoader iteration")
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = os.path.join(tmp, "manifest.json")
        mock_manifest = [
            {"path": f"/fake/train/c/{i}.mp4", "num_frames": 64, "fps": 25.0, "label": i % 3, "split": "train"}
            for i in range(20)
        ] + [
            {"path": f"/fake/val/c/{i}.mp4", "num_frames": 32, "fps": 25.0, "label": 0, "split": "val"}
            for i in range(5)
        ]
        with open(manifest_path, "w") as f:
            json.dump(mock_manifest, f)

        cfg_iter = DataModuleConfig(
            data_root=tmp,
            batch_size=4,
            num_workers_per_gpu=0,
            num_frames=8,
            crop_size=64,
        )
        dm_iter = VideoDataModule(cfg_iter)
        dm_iter.setup("fit")

        train_loader = dm_iter.train_dataloader()
        val_loader   = dm_iter.val_dataloader()

        # Iterate a few batches from train
        batch_count = 0
        for batch in train_loader:
            check(batch["video"].shape[0] <= 4, f"batch size <= 4, got {batch['video'].shape[0]}")
            check(batch["video"].ndim == 5, f"video has 5 dims (B,T,C,H,W)")
            batch_count += 1
            if batch_count >= 3:
                break
        check(batch_count >= 1, f"train loader yields batches, got {batch_count}")

        # Val loader
        val_batches = list(val_loader)
        check(len(val_batches) >= 1, f"val loader yields batches, got {len(val_batches)}")

    # -----------------------------------------------------------------------
    # Test 8: Config serialization round-trip
    # -----------------------------------------------------------------------
    print("\n[Test 8] Config serialization round-trip")
    original_cfg = DataModuleConfig(
        data_root="/data/video",
        batch_size=16,
        num_workers_per_gpu=8,
        num_frames=32,
        crop_size=112,
    )
    d = original_cfg.to_dict()
    restored_cfg = DataModuleConfig.from_dict(d)

    check(restored_cfg.batch_size == 16, f"batch_size round-trips: {restored_cfg.batch_size}")
    check(restored_cfg.num_frames == 32, f"num_frames round-trips: {restored_cfg.num_frames}")
    check(restored_cfg.data_root == "/data/video", f"data_root round-trips")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Results: {passed} PASSED, {failed} FAILED out of {passed + failed} total")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
