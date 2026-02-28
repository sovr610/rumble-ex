"""
Deterministic Episodic Task Sampler for Few-Shot Meta-Learning

Provides a reproducible, platform-independent episodic sampling framework for
few-shot meta-learning (Phase 7 of the cognitive pipeline). Given a dataset
with class labels, samples N-way K-shot episodes deterministically based on
(global_seed, epoch, episode_idx).

Key invariant: same (seed, epoch, episode_idx) ALWAYS produces the identical
episode regardless of platform, worker, or execution order.

Supported datasets:
  - Omniglot (1,623 chars x 50 alphabets, 20 examples each, optional 4x rotations)
  - mini-ImageNet (100 classes, 600 images each, 64/16/20 Ravi & Larochelle split)
  - Synthetic (Gaussian clusters, no external data dependency)

This template is an asset for the meta-learning-suite Claude Code skill.
It is intended to be copied into brain_ai/meta/episode_sampler.py.

Usage::

    from brain_ai.meta.episode_sampler import (
        EpisodeSampler, Episode, ClassSplit,
        OmniglotFewShotDataset, MiniImageNetFewShotDataset,
        SyntheticFewShotDataset, EpisodeDataLoader,
        TaskBatch, create_episode_sampler,
    )

    # Quick start with synthetic data (no downloads required)
    sampler = create_episode_sampler("synthetic", "train", n_way=5, k_shot=1)
    episode = sampler.sample_episode(epoch=0, episode_idx=0)
    print(episode.support_x.shape)  # (5, 64) for 5-way 1-shot

Architecture notes:
  - EpisodeSampler is the core engine; it indexes into pre-loaded tensors.
  - Dataset classes (Omniglot, MiniImageNet, Synthetic) handle data loading
    and produce class splits, then delegate to EpisodeSampler.
  - EpisodeDataLoader wraps EpisodeSampler with an iterator interface.
  - TaskBatch groups multiple episodes for vectorized meta-training.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import hashlib
import struct
import warnings
import math
import json

try:
    from torchvision import transforms, datasets as tv_datasets
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass
class Episode:
    """A single N-way K-shot episode for few-shot meta-learning.

    An episode consists of a *support set* (the few labelled examples the
    model adapts on) and a *query set* (held-out examples used to evaluate
    the adapted model). Labels are always remapped to [0, N) where N is the
    number of ways so that the inner-loop classifier head is compact.

    Attributes:
        support_x: Input features for the support set.
            Shape (N*K, C, H, W) for images or (N*K, D) for vectors.
        support_y: Integer class labels for the support set in [0, N).
            Shape (N*K,).
        query_x: Input features for the query set.
            Shape (N*Q, C, H, W) for images or (N*Q, D) for vectors.
        query_y: Integer class labels for the query set in [0, N).
            Shape (N*Q,).
        class_ids: The original dataset class IDs selected for this episode
            (useful for logging, debugging, and dataset analysis).
        episode_id: A tuple (epoch, episode_idx) uniquely identifying this
            episode within a training run.
    """
    support_x: torch.Tensor    # (N*K, C, H, W) or (N*K, D)
    support_y: torch.Tensor    # (N*K,) labels in [0, N)
    query_x: torch.Tensor      # (N*Q, C, H, W) or (N*Q, D)
    query_y: torch.Tensor      # (N*Q,) labels in [0, N)
    class_ids: List[int]       # original dataset class IDs
    episode_id: Tuple[int, int]  # (epoch, episode_idx)

    @property
    def n_way(self) -> int:
        return len(self.class_ids)

    @property
    def k_shot(self) -> int:
        return self.support_x.size(0) // max(self.n_way, 1)

    @property
    def q_query(self) -> int:
        return self.query_x.size(0) // max(self.n_way, 1)

    @property
    def device(self) -> torch.device:
        return self.support_x.device

    def to(self, device: torch.device) -> "Episode":
        return Episode(
            support_x=self.support_x.to(device),
            support_y=self.support_y.to(device),
            query_x=self.query_x.to(device),
            query_y=self.query_y.to(device),
            class_ids=self.class_ids,
            episode_id=self.episode_id,
        )

    def pin_memory(self) -> "Episode":
        return Episode(
            support_x=self.support_x.pin_memory(),
            support_y=self.support_y.pin_memory(),
            query_x=self.query_x.pin_memory(),
            query_y=self.query_y.pin_memory(),
            class_ids=self.class_ids,
            episode_id=self.episode_id,
        )

    def summary(self) -> str:
        sx = tuple(self.support_x.shape)
        qx = tuple(self.query_x.shape)
        return (
            f"Episode(epoch={self.episode_id[0]}, idx={self.episode_id[1]}, "
            f"{self.n_way}-way {self.k_shot}-shot {self.q_query}-query, "
            f"support={sx}, query={qx}, classes={self.class_ids})"
        )


@dataclass
class ClassSplit:
    """Train/val/test split on *classes* (not images).

    This is the standard approach for few-shot learning evaluation: all images
    within a class belong to the same split, and the splits are disjoint so
    the model never sees val/test classes during meta-training.

    Attributes:
        train: Class IDs allocated to meta-training.
        val:   Class IDs allocated to meta-validation.
        test:  Class IDs allocated to meta-testing.
    """
    train: List[int]
    val: List[int]
    test: List[int]

    def validate(self) -> None:
        """Assert no overlap between splits."""
        train_set, val_set, test_set = set(self.train), set(self.val), set(self.test)
        assert train_set.isdisjoint(val_set), f"Train/val overlap: {train_set & val_set}"
        assert train_set.isdisjoint(test_set), f"Train/test overlap: {train_set & test_set}"
        assert val_set.isdisjoint(test_set), f"Val/test overlap: {val_set & test_set}"

    @property
    def total_classes(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def summary(self) -> str:
        return f"ClassSplit(train={len(self.train)}, val={len(self.val)}, test={len(self.test)})"

    def to_dict(self) -> Dict[str, List[int]]:
        return {"train": list(self.train), "val": list(self.val), "test": list(self.test)}

    @classmethod
    def from_dict(cls, d: Dict[str, List[int]]) -> "ClassSplit":
        return cls(train=d["train"], val=d["val"], test=d["test"])


# ============================================================================
# EpisodeSampler
# ============================================================================

class EpisodeSampler:
    """Deterministic episodic task sampler for few-shot learning.

    Given a dataset with class labels, samples N-way K-shot episodes
    deterministically based on (global_seed, epoch, episode_idx).

    Key invariant: same (seed, epoch, episode_idx) always produces
    the identical episode regardless of platform or worker.
    """

    def __init__(
        self,
        data: torch.Tensor,          # (total_samples, ...)
        labels: torch.Tensor,         # (total_samples,)
        available_classes: List[int],
        n_way: int = 5,
        k_shot: int = 1,
        q_query: int = 15,
        seed: int = 42,
        transform: Optional[Any] = None,
    ):
        self.data = data
        self.labels = labels
        self.available_classes = sorted(available_classes)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.seed = seed
        self.transform = transform

        if self.n_way <= 0:
            raise ValueError(f"n_way must be positive, got {self.n_way}")
        if self.k_shot <= 0:
            raise ValueError(f"k_shot must be positive, got {self.k_shot}")
        if self.q_query <= 0:
            raise ValueError(f"q_query must be positive, got {self.q_query}")

        # Pre-compute class-to-indices mapping
        self.class_to_indices: Dict[int, List[int]] = {}
        for cls in self.available_classes:
            self.class_to_indices[cls] = (labels == cls).nonzero(as_tuple=True)[0].tolist()

        # Filter out under-populated classes
        min_samples = self.k_shot + self.q_query
        self._underpopulated: List[int] = []
        for cls, indices in self.class_to_indices.items():
            if len(indices) < min_samples:
                self._underpopulated.append(cls)
                warnings.warn(
                    f"Class {cls} has {len(indices)} samples, need {min_samples}. Skipping."
                )

        self._viable_classes = [
            c for c in self.available_classes
            if len(self.class_to_indices[c]) >= min_samples
        ]
        if len(self._viable_classes) < self.n_way:
            raise ValueError(
                f"Only {len(self._viable_classes)} viable classes but n_way={self.n_way}"
            )

    def _episode_seed(self, epoch: int, episode_idx: int) -> int:
        """Compute a deterministic seed for a specific episode.

        Uses SHA-256 to hash (global_seed, epoch, episode_idx) into a
        well-distributed 63-bit integer. This avoids the correlation
        artefacts that would arise from simple arithmetic combinations
        like seed + epoch * 10000 + episode_idx.

        Args:
            epoch:       Current training epoch.
            episode_idx: Index of the episode within the epoch.

        Returns:
            A non-negative integer suitable for torch.Generator.manual_seed.
        """
        raw = struct.pack(">qqq", self.seed, epoch, episode_idx)
        h = hashlib.sha256(raw).digest()
        return struct.unpack(">q", h[:8])[0] % (2**63)

    def sample_episode(self, epoch: int, episode_idx: int) -> Episode:
        """Sample a single episode deterministically.

        Given the same (self.seed, epoch, episode_idx), this method will
        always return the *identical* episode (same classes, same support
        indices, same query indices) across platforms.

        Args:
            epoch:       Current training epoch (used for seed derivation).
            episode_idx: Episode index within the epoch.

        Returns:
            An Episode with support and query sets.
        """
        rng = torch.Generator()
        rng.manual_seed(self._episode_seed(epoch, episode_idx))

        # Sample N classes
        class_perm = torch.randperm(len(self._viable_classes), generator=rng)
        selected_classes = [self._viable_classes[class_perm[i].item()] for i in range(self.n_way)]

        support_x_list, support_y_list = [], []
        query_x_list, query_y_list = [], []
        needed = self.k_shot + self.q_query

        for new_label, class_id in enumerate(selected_classes):
            indices = self.class_to_indices[class_id]
            perm = torch.randperm(len(indices), generator=rng)
            selected = [indices[perm[i].item()] for i in range(needed)]

            for idx in selected[:self.k_shot]:
                x = self.data[idx]
                if self.transform is not None:
                    x = self.transform(x)
                support_x_list.append(x)
                support_y_list.append(new_label)

            for idx in selected[self.k_shot:needed]:
                x = self.data[idx]
                if self.transform is not None:
                    x = self.transform(x)
                query_x_list.append(x)
                query_y_list.append(new_label)

        return Episode(
            support_x=torch.stack(support_x_list),
            support_y=torch.tensor(support_y_list, dtype=torch.long),
            query_x=torch.stack(query_x_list),
            query_y=torch.tensor(query_y_list, dtype=torch.long),
            class_ids=selected_classes,
            episode_id=(epoch, episode_idx),
        )

    def sample_batch(self, epoch: int, start_idx: int, batch_size: int) -> List[Episode]:
        """Sample a contiguous batch of episodes."""
        return [self.sample_episode(epoch, start_idx + i) for i in range(batch_size)]

    def class_coverage(self, epoch: int, num_episodes: int) -> Dict[int, int]:
        """Count per-class sampling frequency over num_episodes."""
        counts: Dict[int, int] = {c: 0 for c in self._viable_classes}
        for idx in range(num_episodes):
            for c in self.sample_episode(epoch, idx).class_ids:
                counts[c] += 1
        return counts

    def config_dict(self) -> Dict[str, Any]:
        return {
            "n_way": self.n_way, "k_shot": self.k_shot, "q_query": self.q_query,
            "seed": self.seed, "num_available_classes": len(self.available_classes),
            "num_viable_classes": len(self._viable_classes),
            "num_underpopulated": len(self._underpopulated),
        }

    def __repr__(self) -> str:
        return (
            f"EpisodeSampler(n_way={self.n_way}, k_shot={self.k_shot}, "
            f"q_query={self.q_query}, seed={self.seed}, "
            f"classes={len(self._viable_classes)}/{len(self.available_classes)})"
        )


# ============================================================================
# Omniglot Few-Shot Dataset
# ============================================================================
# 1,623 characters from 50 alphabets, 20 examples/char
# Background: 30 alphabets (964 chars), Evaluation: 20 alphabets (659 chars)
# With rotations (0/90/180/270): 6,492 effective classes
# Image size: 105x105 grayscale -> commonly resized to 28x28

class OmniglotFewShotDataset:
    """Omniglot dataset prepared for few-shot episodic sampling."""

    NUM_BACKGROUND_ALPHABETS = 30
    NUM_EVALUATION_ALPHABETS = 20
    NUM_BACKGROUND_CHARS = 964
    NUM_EVALUATION_CHARS = 659
    EXAMPLES_PER_CHAR = 20

    def __init__(self, root: str, split: str = "train", resize: int = 28,
                 use_rotations: bool = True, download: bool = False):
        self.root = Path(root)
        self.split = split
        self.resize = resize
        self.use_rotations = use_rotations
        self.download = download
        self.data: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.num_classes: int = 0
        self._alphabet_to_chars: Dict[str, List[int]] = {}
        self._background_chars: List[int] = []
        self._evaluation_chars: List[int] = []
        self._load()

    def _load(self) -> None:
        if HAS_TORCHVISION and self.download:
            self._load_torchvision()
        elif self.root.exists():
            self._load_from_directory()
        else:
            warnings.warn(f"Omniglot not found at {self.root}. Using synthetic stand-in.")
            self._create_synthetic_standin()

    def _load_torchvision(self) -> None:
        transform = transforms.Compose([transforms.Resize(self.resize), transforms.ToTensor()])
        bg = tv_datasets.Omniglot(root=str(self.root), background=True, download=True, transform=transform)
        ev = tv_datasets.Omniglot(root=str(self.root), background=False, download=True, transform=transform)

        data_list, label_list = [], []
        bg_ids = set()
        for img, idx in bg:
            data_list.append(img); label_list.append(idx); bg_ids.add(idx)
        self._background_chars = sorted(bg_ids)
        offset = len(bg_ids)
        ev_ids = set()
        for img, idx in ev:
            new_id = idx + offset
            data_list.append(img); label_list.append(new_id); ev_ids.add(new_id)
        self._evaluation_chars = sorted(ev_ids)

        base_data = torch.stack(data_list)
        base_labels = torch.tensor(label_list, dtype=torch.long)
        base_n = offset + len(ev_ids)
        if self.use_rotations:
            self._apply_rotations(base_data, base_labels, base_n)
        else:
            self.data, self.labels, self.num_classes = base_data, base_labels, base_n

    def _load_from_directory(self) -> None:
        bg_dir = self.root / "images_background"
        ev_dir = self.root / "images_evaluation"
        if not bg_dir.exists() and not ev_dir.exists():
            bg_dir = self.root / "omniglot-py" / "images_background"
            ev_dir = self.root / "omniglot-py" / "images_evaluation"
        if not bg_dir.exists():
            warnings.warn(f"No Omniglot dirs at {self.root}. Using synthetic.")
            self._create_synthetic_standin()
            return

        data_list, label_list = [], []
        class_id = 0

        def load_split(split_dir):
            nonlocal class_id
            char_ids = []
            if not split_dir.exists():
                return char_ids
            for alpha_dir in sorted(split_dir.iterdir()):
                if not alpha_dir.is_dir(): continue
                for char_dir in sorted(alpha_dir.iterdir()):
                    if not char_dir.is_dir(): continue
                    char_ids.append(class_id)
                    for img_path in sorted(char_dir.glob("*.png")):
                        if HAS_PIL:
                            img = Image.open(img_path).convert("L")
                            img = img.resize((self.resize, self.resize), Image.BILINEAR)
                            import numpy as np
                            t = torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0
                        else:
                            t = torch.rand(1, self.resize, self.resize)
                        data_list.append(t); label_list.append(class_id)
                    class_id += 1
            return char_ids

        self._background_chars = load_split(bg_dir)
        self._evaluation_chars = load_split(ev_dir)
        if not data_list:
            self._create_synthetic_standin(); return

        base_data = torch.stack(data_list)
        base_labels = torch.tensor(label_list, dtype=torch.long)
        if self.use_rotations:
            self._apply_rotations(base_data, base_labels, class_id)
        else:
            self.data, self.labels, self.num_classes = base_data, base_labels, class_id

    def _apply_rotations(self, base_data, base_labels, base_n):
        """4x rotation augmentation (0/90/180/270). Each rotation = distinct class."""
        all_data, all_labels = [base_data], [base_labels]
        for rot in range(1, 4):
            all_data.append(torch.rot90(base_data, k=rot, dims=[-2, -1]))
            all_labels.append(base_labels + rot * base_n)
        self.data = torch.cat(all_data, dim=0)
        self.labels = torch.cat(all_labels, dim=0)
        self.num_classes = base_n * 4
        bg_rot, ev_rot = [], []
        for rot in range(4):
            off = rot * base_n
            bg_rot.extend([c + off for c in self._background_chars])
            ev_rot.extend([c + off for c in self._evaluation_chars])
        self._background_chars, self._evaluation_chars = bg_rot, ev_rot

    def _create_synthetic_standin(self) -> None:
        base_n = self.NUM_BACKGROUND_CHARS + self.NUM_EVALUATION_CHARS
        rng = torch.Generator().manual_seed(12345)
        data_list, label_list = [], []
        for cls in range(base_n):
            proto = torch.randn(1, 1, self.resize, self.resize, generator=rng)
            for _ in range(self.EXAMPLES_PER_CHAR):
                noise = torch.randn(1, 1, self.resize, self.resize, generator=rng) * 0.3
                data_list.append(torch.sigmoid(proto + noise).squeeze(0))
                label_list.append(cls)
        self._background_chars = list(range(self.NUM_BACKGROUND_CHARS))
        self._evaluation_chars = list(range(self.NUM_BACKGROUND_CHARS, base_n))
        base_data = torch.stack(data_list)
        base_labels = torch.tensor(label_list, dtype=torch.long)
        if self.use_rotations:
            self._apply_rotations(base_data, base_labels, base_n)
        else:
            self.data, self.labels, self.num_classes = base_data, base_labels, base_n

    def get_class_split(self) -> ClassSplit:
        """Background alphabets -> train (80%) + val (20%); evaluation -> test."""
        bg, ev = sorted(self._background_chars), sorted(self._evaluation_chars)
        if not self.use_rotations:
            n_val = max(1, int(len(bg) * 0.2))
            return ClassSplit(train=bg[n_val:], val=bg[:n_val], test=ev)
        base_total = self.num_classes // 4
        base_bg = [c for c in bg if c < base_total]
        base_ev = [c for c in ev if c < base_total]
        n_val = max(1, int(len(base_bg) * 0.2))
        base_val, base_train, base_test = sorted(base_bg)[:n_val], sorted(base_bg)[n_val:], sorted(base_ev)
        train_cls, val_cls, test_cls = [], [], []
        for rot in range(4):
            off = rot * base_total
            train_cls.extend([c + off for c in base_train])
            val_cls.extend([c + off for c in base_val])
            test_cls.extend([c + off for c in base_test])
        split = ClassSplit(train=sorted(train_cls), val=sorted(val_cls), test=sorted(test_cls))
        split.validate()
        return split

    def get_sampler(self, split: str, n_way: int = 5, k_shot: int = 1,
                    q_query: int = 15, seed: int = 42, transform: Optional[Any] = None) -> EpisodeSampler:
        assert self.data is not None, "Dataset not loaded"
        cs = self.get_class_split()
        classes = {"train": cs.train, "val": cs.val, "test": cs.test}.get(split)
        if classes is None:
            raise ValueError(f"Unknown split: {split!r}")
        return EpisodeSampler(self.data, self.labels, classes, n_way, k_shot, q_query, seed, transform)

    def __repr__(self) -> str:
        n = self.data.shape[0] if self.data is not None else 0
        return f"OmniglotFewShotDataset(classes={self.num_classes}, samples={n}, rotations={self.use_rotations})"


# ============================================================================
# mini-ImageNet Few-Shot Dataset
# ============================================================================
# 100 classes from ImageNet, 600 images/class, 84x84 RGB
# Ravi & Larochelle split: 64 train / 16 val / 20 test

class MiniImageNetFewShotDataset:
    """mini-ImageNet dataset for few-shot episodic sampling."""

    TRAIN_CLASSES = 64
    VAL_CLASSES = 16
    TEST_CLASSES = 20
    TOTAL_CLASSES = 100
    IMAGES_PER_CLASS = 600

    def __init__(self, root: str, resize: int = 84, download: bool = False):
        self.root = Path(root)
        self.resize = resize
        self.data: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.num_classes: int = 0
        self._class_names: List[str] = []
        self._load()

    def _load(self) -> None:
        # Try .pt files (safe torch serialization)
        for fname in ["mini-imagenet.pt", "mini_imagenet.pt"]:
            fpath = self.root / fname
            if fpath.exists():
                ckpt = torch.load(fpath, map_location="cpu", weights_only=True)
                if isinstance(ckpt, dict):
                    self.data = ckpt.get("data", ckpt.get("images"))
                    self.labels = ckpt.get("labels", ckpt.get("targets"))
                    if self.data is not None and self.labels is not None:
                        self.num_classes = len(self.labels.unique()); return
                warnings.warn(f"Unexpected .pt format at {fpath}.")
                break

        # Try numpy
        for fname in ["mini_imagenet_data.npy", "data.npy"]:
            dp, lp = self.root / fname, self.root / fname.replace("data", "labels")
            if dp.exists() and lp.exists():
                import numpy as np
                self.data = torch.from_numpy(np.load(str(dp))).float()
                if self.data.dim() == 4 and self.data.shape[-1] == 3:
                    self.data = self.data.permute(0, 3, 1, 2)
                if self.data.max() > 1.0: self.data = self.data / 255.0
                self.labels = torch.from_numpy(np.load(str(lp))).long()
                self.num_classes = len(self.labels.unique()); return

        # Try class subdirectories
        if self.root.exists():
            subdirs = [d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if len(subdirs) >= 50 and HAS_PIL:
                self._load_from_class_dirs(); return

        warnings.warn(f"mini-ImageNet not found at {self.root}. Using synthetic stand-in.")
        self._create_synthetic_standin()

    def _load_from_class_dirs(self) -> None:
        import numpy as np
        all_data, all_labels, cid = [], [], 0
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith("."): continue
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}: continue
                img = Image.open(img_path).convert("RGB").resize((self.resize, self.resize), Image.BILINEAR)
                all_data.append(torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0)
                all_labels.append(cid)
            cid += 1
        if not all_data: self._create_synthetic_standin(); return
        self.data = torch.stack(all_data)
        self.labels = torch.tensor(all_labels, dtype=torch.long)
        self.num_classes = cid

    def _create_synthetic_standin(self) -> None:
        spc = min(self.IMAGES_PER_CLASS, 30)
        rng = torch.Generator().manual_seed(54321)
        data_list, label_list = [], []
        for cls in range(self.TOTAL_CLASSES):
            proto = torch.randn(1, 3, self.resize, self.resize, generator=rng) * 0.5
            for _ in range(spc):
                noise = torch.randn(1, 3, self.resize, self.resize, generator=rng) * 0.2
                data_list.append(torch.sigmoid(proto + noise).squeeze(0))
                label_list.append(cls)
        self.data = torch.stack(data_list)
        self.labels = torch.tensor(label_list, dtype=torch.long)
        self.num_classes = self.TOTAL_CLASSES

    def get_class_split(self) -> ClassSplit:
        """Standard 64/16/20 split (Ravi & Larochelle)."""
        # Try JSON split file
        for name in ["split.json", "splits.json", "class_splits.json"]:
            p = self.root / name
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                split = ClassSplit(train=d.get("train", []), val=d.get("val", []), test=d.get("test", []))
                split.validate(); return split
        all_cls = sorted(set(self.labels.tolist()))
        nt = min(self.TRAIN_CLASSES, len(all_cls))
        nv = min(self.VAL_CLASSES, len(all_cls) - nt)
        split = ClassSplit(train=all_cls[:nt], val=all_cls[nt:nt+nv], test=all_cls[nt+nv:])
        split.validate(); return split

    def get_sampler(self, split: str, n_way: int = 5, k_shot: int = 1,
                    q_query: int = 15, seed: int = 42, transform: Optional[Any] = None) -> EpisodeSampler:
        assert self.data is not None
        cs = self.get_class_split()
        classes = {"train": cs.train, "val": cs.val, "test": cs.test}.get(split)
        if classes is None: raise ValueError(f"Unknown split: {split!r}")
        return EpisodeSampler(self.data, self.labels, classes, n_way, k_shot, q_query, seed, transform)

    def __repr__(self) -> str:
        n = self.data.shape[0] if self.data is not None else 0
        return f"MiniImageNetFewShotDataset(classes={self.num_classes}, samples={n}, resize={self.resize})"


# ============================================================================
# Synthetic Few-Shot Dataset
# ============================================================================

class SyntheticFewShotDataset:
    """Synthetic Gaussian-cluster dataset for testing without real data."""

    def __init__(self, num_classes: int = 100, samples_per_class: int = 20,
                 feature_dim: int = 64, seed: int = 42, image_like: bool = False,
                 cluster_spread: float = 0.5, center_scale: float = 3.0):
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.feature_dim = feature_dim
        self.seed = seed
        self.image_like = image_like
        self.cluster_spread = cluster_spread
        self.center_scale = center_scale

        rng = torch.Generator().manual_seed(seed)
        eff_dim = max(feature_dim, 784) if image_like else feature_dim
        centers = torch.randn(num_classes, eff_dim, generator=rng) * center_scale

        data_list, label_list = [], []
        for cls in range(num_classes):
            samples = centers[cls].unsqueeze(0) + torch.randn(samples_per_class, eff_dim, generator=rng) * cluster_spread
            if image_like:
                img = torch.zeros(samples_per_class, 1, 28, 28)
                img[:, 0] = samples[:, :784].view(samples_per_class, 28, 28)
                data_list.append(img)
            else:
                data_list.append(samples)
            label_list.append(torch.full((samples_per_class,), cls, dtype=torch.long))

        self.data = torch.cat(data_list, dim=0)
        self.labels = torch.cat(label_list, dim=0)

    def get_class_split(self, train_frac=0.64, val_frac=0.16) -> ClassSplit:
        all_cls = list(range(self.num_classes))
        nt, nv = int(self.num_classes * train_frac), int(self.num_classes * val_frac)
        split = ClassSplit(train=all_cls[:nt], val=all_cls[nt:nt+nv], test=all_cls[nt+nv:])
        split.validate(); return split

    def get_sampler(self, split: str, n_way: int = 5, k_shot: int = 1,
                    q_query: int = 15, seed: int = 42, transform: Optional[Any] = None) -> EpisodeSampler:
        cs = self.get_class_split()
        classes = {"train": cs.train, "val": cs.val, "test": cs.test}.get(split)
        if classes is None: raise ValueError(f"Unknown split: {split!r}")
        return EpisodeSampler(self.data, self.labels, classes, n_way, k_shot, q_query, seed, transform)

    def __repr__(self) -> str:
        return f"SyntheticFewShotDataset(classes={self.num_classes}, data={tuple(self.data.shape)})"


# ============================================================================
# EpisodeDataLoader
# ============================================================================

class EpisodeDataLoader:
    """DataLoader-style iterator that yields episodes."""

    def __init__(self, sampler: EpisodeSampler, episodes_per_epoch: int = 600,
                 num_workers: int = 0, collate_to_task_batch: bool = False,
                 task_batch_size: int = 4, pin_memory: bool = False):
        self.sampler = sampler
        self.episodes_per_epoch = episodes_per_epoch
        self.num_workers = num_workers
        self.collate_to_task_batch = collate_to_task_batch
        self.task_batch_size = task_batch_size
        self.pin_memory = pin_memory
        self.epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        if self.collate_to_task_batch:
            return math.ceil(self.episodes_per_epoch / self.task_batch_size)
        return self.episodes_per_epoch

    def _sample_episode(self, idx: int) -> Episode:
        ep = self.sampler.sample_episode(self.epoch, idx)
        if self.pin_memory:
            ep = ep.pin_memory()
        return ep

    def __iter__(self):
        if self.num_workers > 0:
            yield from self._iter_threaded()
        else:
            yield from self._iter_sequential()

    def _iter_sequential(self):
        if not self.collate_to_task_batch:
            for idx in range(self.episodes_per_epoch):
                yield self._sample_episode(idx)
        else:
            batch: List[Episode] = []
            for idx in range(self.episodes_per_epoch):
                batch.append(self._sample_episode(idx))
                if len(batch) == self.task_batch_size:
                    yield TaskBatch.collate(batch); batch = []
            if batch:
                yield TaskBatch.collate(batch)

    def _iter_threaded(self):
        from concurrent.futures import ThreadPoolExecutor
        from collections import OrderedDict
        prefetch = min(self.num_workers * 2, self.episodes_per_epoch)
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = OrderedDict()
            for i in range(min(prefetch, self.episodes_per_epoch)):
                futures[i] = pool.submit(self._sample_episode, i)
            nxt = prefetch
            if not self.collate_to_task_batch:
                for idx in range(self.episodes_per_epoch):
                    yield futures.pop(idx).result()
                    if nxt < self.episodes_per_epoch:
                        futures[nxt] = pool.submit(self._sample_episode, nxt); nxt += 1
            else:
                batch: List[Episode] = []
                for idx in range(self.episodes_per_epoch):
                    batch.append(futures.pop(idx).result())
                    if len(batch) == self.task_batch_size:
                        yield TaskBatch.collate(batch); batch = []
                    if nxt < self.episodes_per_epoch:
                        futures[nxt] = pool.submit(self._sample_episode, nxt); nxt += 1
                if batch:
                    yield TaskBatch.collate(batch)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "episodes_per_epoch": self.episodes_per_epoch, "num_workers": self.num_workers,
            "collate_to_task_batch": self.collate_to_task_batch,
            "task_batch_size": self.task_batch_size, "pin_memory": self.pin_memory,
            "epoch": self.epoch, "sampler": self.sampler.config_dict(),
        }

    def __repr__(self) -> str:
        return (f"EpisodeDataLoader(episodes={self.episodes_per_epoch}, epoch={self.epoch}, "
                f"batch={self.collate_to_task_batch}, workers={self.num_workers})")


# ============================================================================
# TaskBatch
# ============================================================================

@dataclass
class TaskBatch:
    """Batch of episodes for vectorized meta-training."""
    episodes: List[Episode]

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Episode:
        return self.episodes[idx]

    def __iter__(self):
        return iter(self.episodes)

    @staticmethod
    def collate(episodes: List[Episode]) -> "TaskBatch":
        return TaskBatch(episodes=list(episodes))

    def to(self, device: torch.device) -> "TaskBatch":
        return TaskBatch(episodes=[ep.to(device) for ep in self.episodes])

    def pin_memory(self) -> "TaskBatch":
        return TaskBatch(episodes=[ep.pin_memory() for ep in self.episodes])

    @property
    def n_way(self) -> int:
        return self.episodes[0].n_way if self.episodes else 0

    @property
    def k_shot(self) -> int:
        return self.episodes[0].k_shot if self.episodes else 0

    def stacked_support(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (B, N*K, ...) and (B, N*K) stacked support tensors."""
        return torch.stack([e.support_x for e in self.episodes]), torch.stack([e.support_y for e in self.episodes])

    def stacked_query(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (B, N*Q, ...) and (B, N*Q) stacked query tensors."""
        return torch.stack([e.query_x for e in self.episodes]), torch.stack([e.query_y for e in self.episodes])

    def summary(self) -> str:
        if not self.episodes: return "TaskBatch(empty)"
        e = self.episodes[0]
        return f"TaskBatch(tasks={len(self)}, {e.n_way}-way {e.k_shot}-shot {e.q_query}-query)"


# ============================================================================
# Factory Function
# ============================================================================

def create_episode_sampler(
    dataset: str, split: str, n_way: int = 5, k_shot: int = 1, q_query: int = 15,
    seed: int = 42, root: Optional[str] = None, transform: Optional[Any] = None, **kwargs,
) -> EpisodeSampler:
    """Create an EpisodeSampler for the specified dataset and split."""
    dataset = dataset.lower().strip()
    if dataset == "omniglot":
        ds = OmniglotFewShotDataset(root=root or "data/omniglot", split=split, **kwargs)
        return ds.get_sampler(split, n_way, k_shot, q_query, seed, transform)
    elif dataset in ("mini_imagenet", "mini-imagenet", "miniimagenet"):
        ds = MiniImageNetFewShotDataset(root=root or "data/mini_imagenet", **kwargs)
        return ds.get_sampler(split, n_way, k_shot, q_query, seed, transform)
    elif dataset == "synthetic":
        ds = SyntheticFewShotDataset(**kwargs)
        return ds.get_sampler(split, n_way, k_shot, q_query, seed, transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Supported: omniglot, mini_imagenet, synthetic")


# ============================================================================
# Utilities
# ============================================================================

def episode_to_dict(episode: Episode) -> Dict[str, Any]:
    """Convert an Episode to a JSON-serializable dictionary."""
    return {
        "episode_id": list(episode.episode_id), "class_ids": episode.class_ids,
        "n_way": episode.n_way, "k_shot": episode.k_shot, "q_query": episode.q_query,
        "support_x_shape": list(episode.support_x.shape), "support_y": episode.support_y.tolist(),
        "query_x_shape": list(episode.query_x.shape), "query_y": episode.query_y.tolist(),
    }


def verify_episode_determinism(sampler: EpisodeSampler, epoch: int = 0,
                                episode_idx: int = 0, num_trials: int = 5) -> bool:
    """Verify same (epoch, idx) produces identical episodes across trials."""
    ref = sampler.sample_episode(epoch, episode_idx)
    for _ in range(num_trials):
        ep = sampler.sample_episode(epoch, episode_idx)
        if not torch.equal(ep.support_x, ref.support_x): return False
        if not torch.equal(ep.support_y, ref.support_y): return False
        if not torch.equal(ep.query_x, ref.query_x): return False
        if not torch.equal(ep.query_y, ref.query_y): return False
        if ep.class_ids != ref.class_ids: return False
    return True


def compute_class_balance(sampler: EpisodeSampler, epoch: int, num_episodes: int) -> Dict[str, float]:
    """Compute min/max/mean/std of per-class sampling frequency."""
    counts = sampler.class_coverage(epoch, num_episodes)
    vals = list(counts.values())
    if not vals: return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    t = torch.tensor(vals, dtype=torch.float)
    return {"min": t.min().item(), "max": t.max().item(), "mean": t.mean().item(),
            "std": t.std().item(), "total_classes": len(vals),
            "zero_count": sum(1 for v in vals if v == 0)}


def split_summary(class_split: ClassSplit) -> str:
    """Format a ClassSplit as a human-readable one-line string."""
    return (f"Train: {len(class_split.train)} | Val: {len(class_split.val)} | "
            f"Test: {len(class_split.test)} | Total: {class_split.total_classes}")


def validate_episode(episode: Episode, n_way: int, k_shot: int, q_query: int) -> List[str]:
    """Run a suite of structural checks on an episode.

    Returns a list of error messages (empty if the episode is valid).
    Useful for debugging custom samplers or dataset integrations.

    Args:
        episode:  The episode to validate.
        n_way:    Expected number of classes.
        k_shot:   Expected support examples per class.
        q_query:  Expected query examples per class.

    Returns:
        List of error strings. Empty list means the episode is valid.
    """
    errors: List[str] = []

    # Shape checks
    exp_s = n_way * k_shot
    exp_q = n_way * q_query
    if episode.support_x.shape[0] != exp_s:
        errors.append(f"support_x has {episode.support_x.shape[0]} samples, expected {exp_s}")
    if episode.support_y.shape[0] != exp_s:
        errors.append(f"support_y has {episode.support_y.shape[0]} labels, expected {exp_s}")
    if episode.query_x.shape[0] != exp_q:
        errors.append(f"query_x has {episode.query_x.shape[0]} samples, expected {exp_q}")
    if episode.query_y.shape[0] != exp_q:
        errors.append(f"query_y has {episode.query_y.shape[0]} labels, expected {exp_q}")

    # Label range checks
    if episode.support_y.numel() > 0:
        if episode.support_y.min().item() < 0:
            errors.append(f"support_y has negative label: {episode.support_y.min().item()}")
        if episode.support_y.max().item() >= n_way:
            errors.append(f"support_y has label >= n_way: {episode.support_y.max().item()}")
    if episode.query_y.numel() > 0:
        if episode.query_y.min().item() < 0:
            errors.append(f"query_y has negative label: {episode.query_y.min().item()}")
        if episode.query_y.max().item() >= n_way:
            errors.append(f"query_y has label >= n_way: {episode.query_y.max().item()}")

    # Class coverage: all N classes should appear in support and query
    if len(episode.support_y.unique()) != n_way:
        errors.append(f"support has {len(episode.support_y.unique())} unique classes, expected {n_way}")
    if len(episode.query_y.unique()) != n_way:
        errors.append(f"query has {len(episode.query_y.unique())} unique classes, expected {n_way}")

    # Class IDs count
    if len(episode.class_ids) != n_way:
        errors.append(f"class_ids has {len(episode.class_ids)} entries, expected {n_way}")
    if len(set(episode.class_ids)) != len(episode.class_ids):
        errors.append(f"class_ids has duplicates: {episode.class_ids}")

    return errors


def create_dataloader(
    dataset: str,
    split: str,
    n_way: int = 5,
    k_shot: int = 1,
    q_query: int = 15,
    episodes_per_epoch: int = 600,
    seed: int = 42,
    root: Optional[str] = None,
    num_workers: int = 0,
    collate_to_task_batch: bool = False,
    task_batch_size: int = 4,
    pin_memory: bool = False,
    **kwargs,
) -> EpisodeDataLoader:
    """Convenience function to create an EpisodeDataLoader in one call.

    Combines create_episode_sampler() and EpisodeDataLoader construction.

    Args:
        dataset:            Dataset name (omniglot, mini_imagenet, synthetic).
        split:              Split name (train, val, test).
        n_way:              Classes per episode.
        k_shot:             Support examples per class.
        q_query:            Query examples per class.
        episodes_per_epoch: Number of episodes per training epoch.
        seed:               Global seed for reproducibility.
        root:               Data directory path.
        num_workers:        Background worker threads.
        collate_to_task_batch: If True, yield TaskBatch instead of Episode.
        task_batch_size:    Episodes per TaskBatch.
        pin_memory:         Pin tensors for faster GPU transfer.
        **kwargs:           Additional args for the dataset constructor.

    Returns:
        A configured EpisodeDataLoader.
    """
    sampler = create_episode_sampler(
        dataset=dataset, split=split, n_way=n_way, k_shot=k_shot,
        q_query=q_query, seed=seed, root=root, **kwargs,
    )
    return EpisodeDataLoader(
        sampler=sampler,
        episodes_per_epoch=episodes_per_epoch,
        num_workers=num_workers,
        collate_to_task_batch=collate_to_task_batch,
        task_batch_size=task_batch_size,
        pin_memory=pin_memory,
    )


def episode_accuracy(episode: Episode, logits_fn) -> Dict[str, float]:
    """Compute support and query accuracy given a logits function.

    Useful for quick evaluation of a classifier on an episode.

    Args:
        episode:   The episode to evaluate.
        logits_fn: Callable that takes (support_x, support_y, query_x)
                   and returns query logits of shape (N*Q, N).

    Returns:
        Dict with 'support_acc' and 'query_acc' as floats in [0, 1].
    """
    with torch.no_grad():
        query_logits = logits_fn(episode.support_x, episode.support_y, episode.query_x)
        query_preds = query_logits.argmax(dim=-1)
        query_acc = (query_preds == episode.query_y).float().mean().item()

        # Also compute support accuracy (training set fit)
        support_logits = logits_fn(episode.support_x, episode.support_y, episode.support_x)
        support_preds = support_logits.argmax(dim=-1)
        support_acc = (support_preds == episode.support_y).float().mean().item()

    return {"support_acc": support_acc, "query_acc": query_acc}


# ============================================================================
# Self-Test Block
# ============================================================================

def _run_self_tests() -> None:
    """Run comprehensive self-tests. All use SyntheticFewShotDataset."""
    import traceback, sys

    results: List[Tuple[str, bool, str]] = []
    total_tests = passed_tests = 0

    def run_test(name, fn):
        nonlocal total_tests, passed_tests
        total_tests += 1
        try:
            fn(); results.append((name, True, "")); passed_tests += 1
            print(f"  PASS: {name}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  FAIL: {name}\n        {e}")
            if "--verbose" in sys.argv: traceback.print_exc()

    print("=" * 72)
    print("Episode Sampler Self-Tests")
    print("=" * 72 + "\n")

    # 1. SyntheticFewShotDataset creates correct data shapes
    def test_synthetic_shapes():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20, feature_dim=64)
        assert ds.data.shape == (1000, 64), f"Got {ds.data.shape}"
        assert ds.labels.shape == (1000,)
        assert ds.num_classes == 50
        ds2 = SyntheticFewShotDataset(num_classes=30, samples_per_class=10, feature_dim=64, image_like=True)
        assert ds2.data.shape == (300, 1, 28, 28), f"Got {ds2.data.shape}"
    run_test("1. SyntheticFewShotDataset correct data shapes", test_synthetic_shapes)

    # 2. ClassSplit validation passes for disjoint splits
    def test_split_valid():
        s = ClassSplit(train=[0,1,2,3], val=[4,5], test=[6,7,8,9])
        s.validate()
        assert s.total_classes == 10
    run_test("2. ClassSplit validation passes for disjoint splits", test_split_valid)

    # 3. ClassSplit validation fails for overlapping splits
    def test_split_overlap():
        for t, v, te in [([0,1,2],[2,3],[4,5]), ([0,1,2],[3,4],[1,5]), ([0,1],[2,3],[3,4])]:
            try:
                ClassSplit(train=t, val=v, test=te).validate()
                raise RuntimeError("Should have raised")
            except AssertionError:
                pass
    run_test("3. ClassSplit validation fails for overlapping splits", test_split_overlap)

    # 4. EpisodeSampler creates valid episodes
    def test_sampler_creates():
        ds = SyntheticFewShotDataset(num_classes=100, samples_per_class=20, feature_dim=64)
        sampler = ds.get_sampler("train", n_way=5, k_shot=1, q_query=15, seed=42)
        ep = sampler.sample_episode(0, 0)
        assert isinstance(ep, Episode) and ep.episode_id == (0, 0) and len(ep.class_ids) == 5
    run_test("4. EpisodeSampler creates valid episodes", test_sampler_creates)

    # 5. Episode has correct N*K support, N*Q query shapes
    def test_episode_shapes():
        ds = SyntheticFewShotDataset(num_classes=100, samples_per_class=30, feature_dim=64)
        ep = ds.get_sampler("train", 5, 3, 10, 42).sample_episode(0, 0)
        assert ep.support_x.shape == (15, 64) and ep.query_x.shape == (50, 64)
        assert ep.support_y.shape == (15,) and ep.query_y.shape == (50,)
        assert ep.n_way == 5 and ep.k_shot == 3 and ep.q_query == 10
        ds2 = SyntheticFewShotDataset(num_classes=100, samples_per_class=30, feature_dim=64, image_like=True)
        ep2 = ds2.get_sampler("train", 5, 3, 10, 42).sample_episode(0, 0)
        assert ep2.support_x.shape == (15, 1, 28, 28) and ep2.query_x.shape == (50, 1, 28, 28)
    run_test("5. Episode has correct N*K support, N*Q query shapes", test_episode_shapes)

    # 6. Labels are in [0, N) range
    def test_labels_range():
        ds = SyntheticFewShotDataset(num_classes=100, samples_per_class=25, feature_dim=32)
        sampler = ds.get_sampler("train", 7, 2, 5, 99)
        for i in range(20):
            ep = sampler.sample_episode(0, i)
            assert ep.support_y.min() >= 0 and ep.support_y.max() < 7
            assert ep.query_y.min() >= 0 and ep.query_y.max() < 7
            assert len(ep.support_y.unique()) == 7 and len(ep.query_y.unique()) == 7
    run_test("6. Labels are in [0, N) range", test_labels_range)

    # 7. Support and query sets are disjoint
    def test_disjoint():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20, feature_dim=64, seed=42)
        sampler = ds.get_sampler("train", 5, 3, 5, 42)
        for i in range(30):
            ep = sampler.sample_episode(0, i)
            for lbl in range(ep.n_way):
                sup = ep.support_x[ep.support_y == lbl]
                qry = ep.query_x[ep.query_y == lbl]
                for s in range(sup.shape[0]):
                    for q in range(qry.shape[0]):
                        assert not torch.equal(sup[s], qry[q]), f"Ep {i} class {lbl}: overlap"
    run_test("7. Support and query sets are disjoint", test_disjoint)

    # 8. Determinism: same (seed, epoch, idx) -> identical episode
    def test_determinism():
        ds = SyntheticFewShotDataset(num_classes=80, samples_per_class=25, feature_dim=48, seed=42)
        sampler = ds.get_sampler("train", 5, 2, 10, 12345)
        for epoch, idx in [(0,0),(0,1),(0,99),(1,0),(5,42),(100,500)]:
            e1, e2 = sampler.sample_episode(epoch, idx), sampler.sample_episode(epoch, idx)
            assert torch.equal(e1.support_x, e2.support_x) and torch.equal(e1.query_x, e2.query_x)
            assert e1.class_ids == e2.class_ids
        assert verify_episode_determinism(sampler, 0, 0) and verify_episode_determinism(sampler, 3, 42)
    run_test("8. Determinism: same (seed, epoch, idx) -> identical episode", test_determinism)

    # 9. Different seeds -> different episodes
    def test_diff_seeds():
        ds = SyntheticFewShotDataset(num_classes=100, samples_per_class=20, feature_dim=64)
        sa = ds.get_sampler("train", 5, 1, 15, 42)
        sb = ds.get_sampler("train", 5, 1, 15, 999)
        a, b = sa.sample_episode(0, 0), sb.sample_episode(0, 0)
        assert a.class_ids != b.class_ids or not torch.equal(a.support_x, b.support_x)
        c, d = sa.sample_episode(0, 0), sa.sample_episode(0, 1)
        assert c.class_ids != d.class_ids or not torch.equal(c.support_x, d.support_x)
        e, f = sa.sample_episode(0, 0), sa.sample_episode(1, 0)
        assert e.class_ids != f.class_ids or not torch.equal(e.support_x, f.support_x)
    run_test("9. Different seeds -> different episodes", test_diff_seeds)

    # 10. Class IDs logged correctly
    def test_class_ids():
        ds = SyntheticFewShotDataset(num_classes=100, samples_per_class=20, feature_dim=32)
        train_cls = set(ds.get_class_split().train)
        sampler = ds.get_sampler("train", 5, 1, 15, 42)
        for i in range(50):
            ep = sampler.sample_episode(0, i)
            assert all(c in train_cls for c in ep.class_ids)
            assert len(ep.class_ids) == 5 and len(set(ep.class_ids)) == 5
    run_test("10. Class IDs logged correctly", test_class_ids)

    # 11. Coverage: all classes sampled after many episodes
    def test_coverage():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20, feature_dim=32)
        sampler = ds.get_sampler("train", 5, 1, 5, 42)
        counts = sampler.class_coverage(0, 200)
        assert all(v > 0 for v in counts.values()), "Some classes never sampled"
        stats = compute_class_balance(sampler, 0, 200)
        assert stats["zero_count"] == 0 and stats["mean"] > 0
    run_test("11. Coverage: all classes sampled after many episodes", test_coverage)

    # 12. EpisodeDataLoader yields correct number of episodes
    def test_loader_count():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20, feature_dim=32)
        sampler = ds.get_sampler("train", 5, 1, 5, 42)
        loader = EpisodeDataLoader(sampler, episodes_per_epoch=25)
        loader.set_epoch(0)
        eps = list(loader)
        assert len(eps) == 25 and all(isinstance(e, Episode) for e in eps)
        # Batched mode
        lb = EpisodeDataLoader(sampler, episodes_per_epoch=25, collate_to_task_batch=True, task_batch_size=4)
        lb.set_epoch(0)
        batches = list(lb)
        assert len(batches) == math.ceil(25 / 4)
        assert sum(len(b) for b in batches) == 25
    run_test("12. EpisodeDataLoader yields correct number of episodes", test_loader_count)

    # 13. EpisodeDataLoader respects epoch setting
    def test_loader_epoch():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20, feature_dim=32)
        sampler = ds.get_sampler("train", 5, 1, 5, 42)
        loader = EpisodeDataLoader(sampler, episodes_per_epoch=10)
        loader.set_epoch(0); e0 = list(loader)
        loader.set_epoch(1); e1 = list(loader)
        loader.set_epoch(0); e0b = list(loader)
        # Same epoch replays identically
        for a, b in zip(e0, e0b):
            assert torch.equal(a.support_x, b.support_x) and a.class_ids == b.class_ids
        # Different epochs differ
        assert any(not torch.equal(a.support_x, b.support_x) or a.class_ids != b.class_ids
                    for a, b in zip(e0, e1))
    run_test("13. EpisodeDataLoader respects epoch setting", test_loader_epoch)

    # 14. TaskBatch collation works
    def test_task_batch():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20, feature_dim=32)
        sampler = ds.get_sampler("train", 5, 2, 5, 42)
        eps = sampler.sample_batch(0, 0, 4)
        batch = TaskBatch.collate(eps)
        assert len(batch) == 4 and batch.n_way == 5 and batch.k_shot == 2
        assert isinstance(batch[0], Episode)
        sx, sy = batch.stacked_support()
        assert sx.shape == (4, 10, 32) and sy.shape == (4, 10)
        qx, qy = batch.stacked_query()
        assert qx.shape == (4, 25, 32) and qy.shape == (4, 25)
        assert "TaskBatch" in batch.summary()
        assert TaskBatch(episodes=[]).summary() == "TaskBatch(empty)"
    run_test("14. TaskBatch collation works", test_task_batch)

    # 15. Omniglot rotations multiply classes by 4
    def test_omniglot_rotations():
        d0 = OmniglotFewShotDataset(root="/tmp/nonexistent_omniglot", use_rotations=False)
        d4 = OmniglotFewShotDataset(root="/tmp/nonexistent_omniglot", use_rotations=True)
        base = d0.NUM_BACKGROUND_CHARS + d0.NUM_EVALUATION_CHARS
        assert d0.num_classes == base and d4.num_classes == base * 4
        assert d4.data.shape[0] == d0.data.shape[0] * 4
        s0, s4 = d0.get_class_split(), d4.get_class_split()
        s0.validate(); s4.validate()
        assert len(s4.train) == len(s0.train) * 4
        assert len(s4.val) == len(s0.val) * 4
        assert len(s4.test) == len(s0.test) * 4
    run_test("15. Omniglot rotations multiply classes by 4 (synthetic stand-in)", test_omniglot_rotations)

    # 16. Factory function creates samplers
    def test_factory():
        s = create_episode_sampler("synthetic", "train", 5, 1, 10, 42,
                                   num_classes=60, samples_per_class=25, feature_dim=48)
        assert isinstance(s, EpisodeSampler)
        ep = s.sample_episode(0, 0)
        assert ep.n_way == 5 and ep.k_shot == 1 and ep.q_query == 10
        sv = create_episode_sampler("synthetic", "val", 3, 2, 5, 42,
                                    num_classes=60, samples_per_class=25, feature_dim=48)
        assert sv.sample_episode(0, 0).n_way == 3
        # Check class split integrity
        ds = SyntheticFewShotDataset(num_classes=60, samples_per_class=25, feature_dim=48)
        sp = ds.get_class_split()
        assert all(c in sp.train for c in ep.class_ids)
        assert all(c in sp.val for c in sv.sample_episode(0, 0).class_ids)
        try:
            create_episode_sampler("unknown", "train")
            assert False, "Should raise"
        except ValueError:
            pass
        assert isinstance(create_episode_sampler("mini-imagenet", "train", 5, 1, 5, 42), EpisodeSampler)
    run_test("16. Factory function creates samplers", test_factory)

    # 17. _episode_seed produces well-distributed seeds
    def test_seed_distribution():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20)
        sampler = ds.get_sampler("train", 5, 1, 5, 42)
        seeds = {sampler._episode_seed(i // 100, i % 100) for i in range(10000)}
        assert len(seeds) == 10000, f"Collisions: {10000 - len(seeds)}"
        assert all(0 <= sampler._episode_seed(i, 0) < 2**63 for i in range(100))
        sa = EpisodeSampler(ds.data, ds.labels, list(range(32)), 5, 1, 5, seed=1)
        sb = EpisodeSampler(ds.data, ds.labels, list(range(32)), 5, 1, 5, seed=2)
        assert sa._episode_seed(0, 0) != sb._episode_seed(0, 0)
        assert len({sampler._episode_seed(e, i) for e, i in [(0,0),(0,1),(1,0),(1,1)]}) == 4
    run_test("17. _episode_seed produces well-distributed seeds", test_seed_distribution)

    # 18. Episode.to() and summary()
    def test_episode_ops():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20)
        ep = ds.get_sampler("train", 5, 1, 5, 42).sample_episode(0, 0)
        ep2 = ep.to(torch.device("cpu"))
        assert torch.equal(ep2.support_x, ep.support_x) and ep2.class_ids == ep.class_ids
        assert "5-way" in ep.summary()
    run_test("18. Episode.to() and summary()", test_episode_ops)

    # 19. EpisodeSampler with transform
    def test_transform():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20, feature_dim=64)
        sp = ds.get_sampler("train", 5, 1, 5, 42)
        sx = ds.get_sampler("train", 5, 1, 5, 42, transform=lambda x: x * 2.0)
        ep, epx = sp.sample_episode(0, 0), sx.sample_episode(0, 0)
        assert ep.class_ids == epx.class_ids
        assert torch.allclose(epx.support_x, ep.support_x * 2.0, atol=1e-6)
        assert torch.allclose(epx.query_x, ep.query_x * 2.0, atol=1e-6)
    run_test("19. EpisodeSampler with transform", test_transform)

    # 20. ClassSplit serialization round-trip
    def test_split_serial():
        orig = ClassSplit(train=[0,1,2,3,4], val=[5,6,7], test=[8,9])
        d = orig.to_dict()
        r = ClassSplit.from_dict(d)
        assert r.train == orig.train and r.val == orig.val and r.test == orig.test
        r.validate()
        r2 = ClassSplit.from_dict(json.loads(json.dumps(d)))
        assert r2.train == orig.train
    run_test("20. ClassSplit serialization round-trip", test_split_serial)

    # 21. EpisodeSampler validation errors
    def test_validation():
        ds = SyntheticFewShotDataset(num_classes=10, samples_per_class=5, feature_dim=32)
        for kwargs in [
            dict(available_classes=list(range(3)), n_way=5, k_shot=1, q_query=1),
            dict(available_classes=list(range(10)), n_way=2, k_shot=3, q_query=3),
            dict(available_classes=list(range(10)), n_way=0, k_shot=1, q_query=1),
        ]:
            try:
                EpisodeSampler(ds.data, ds.labels, seed=42, **kwargs)
                assert False, f"Should raise for {kwargs}"
            except ValueError:
                pass
    run_test("21. EpisodeSampler validation errors", test_validation)

    # 22. config_dict and repr
    def test_config_repr():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20, feature_dim=32)
        sampler = ds.get_sampler("train", 5, 1, 5, 42)
        cfg = sampler.config_dict()
        assert cfg["n_way"] == 5 and cfg["seed"] == 42
        json.dumps(cfg)  # must be serializable
        assert "EpisodeSampler" in repr(sampler)
        loader = EpisodeDataLoader(sampler, episodes_per_epoch=100)
        assert loader.config_dict()["episodes_per_epoch"] == 100
        assert "EpisodeDataLoader" in repr(loader)
    run_test("22. config_dict and repr", test_config_repr)

    # 23. MiniImageNet synthetic stand-in
    def test_mini_imagenet():
        ds = MiniImageNetFewShotDataset(root="/tmp/nonexistent_mini_imagenet")
        assert ds.num_classes == 100 and ds.data is not None
        sp = ds.get_class_split()
        sp.validate()
        assert len(sp.train) == 64 and len(sp.val) == 16 and len(sp.test) == 20
        ep = ds.get_sampler("train", 5, 1, 5, 42).sample_episode(0, 0)
        assert all(c in sp.train for c in ep.class_ids)
        assert "MiniImageNet" in repr(ds)
    run_test("23. MiniImageNet synthetic stand-in", test_mini_imagenet)

    # 24. episode_to_dict utility
    def test_ep_dict():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20, feature_dim=32)
        ep = ds.get_sampler("train", 5, 2, 3, 42).sample_episode(0, 0)
        d = episode_to_dict(ep)
        assert d["n_way"] == 5 and d["k_shot"] == 2 and d["q_query"] == 3
        assert d["support_x_shape"][0] == 10 and d["query_x_shape"][0] == 15
        json.dumps(d)
    run_test("24. episode_to_dict utility", test_ep_dict)

    # 25. Threaded EpisodeDataLoader determinism
    def test_threaded():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20, feature_dim=32)
        sampler = ds.get_sampler("train", 5, 1, 5, 42)
        l0 = EpisodeDataLoader(sampler, episodes_per_epoch=15, num_workers=0)
        l0.set_epoch(0); seq = list(l0)
        lt = EpisodeDataLoader(sampler, episodes_per_epoch=15, num_workers=2)
        lt.set_epoch(0); thr = list(lt)
        assert len(thr) == len(seq)
        for a, b in zip(seq, thr):
            assert torch.equal(a.support_x, b.support_x) and a.class_ids == b.class_ids
    run_test("25. Threaded EpisodeDataLoader determinism", test_threaded)

    # 26. Cluster separability
    def test_separability():
        ds = SyntheticFewShotDataset(num_classes=10, samples_per_class=50,
                                     feature_dim=64, seed=42, cluster_spread=0.1, center_scale=5.0)
        centroids = torch.stack([ds.data[ds.labels == c].mean(0) for c in range(10)])
        min_inter = min((centroids[i] - centroids[j]).norm().item()
                        for i in range(10) for j in range(i+1, 10))
        max_intra = max((ds.data[ds.labels == c] - ds.data[ds.labels == c].mean(0)).norm(dim=1).max().item()
                        for c in range(10))
        assert min_inter > max_intra, f"inter={min_inter:.3f} <= intra={max_intra:.3f}"
    run_test("26. SyntheticFewShotDataset cluster separability", test_separability)

    # 27. TaskBatch advanced operations
    def test_batch_adv():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20, feature_dim=32)
        batch = TaskBatch.collate(ds.get_sampler("train", 5, 2, 3, 42).sample_batch(0, 0, 3))
        bc = batch.to(torch.device("cpu"))
        assert len(bc) == 3 and torch.equal(bc[0].support_x, batch[0].support_x)
        sx, sy = batch.stacked_support()
        assert sx.shape == (3, 10, 32)
        qx, qy = batch.stacked_query()
        assert qx.shape == (3, 15, 32)
        for i, ep in enumerate(batch):
            assert torch.equal(sy[i], ep.support_y) and torch.equal(qy[i], ep.query_y)
    run_test("27. TaskBatch advanced operations", test_batch_adv)

    # 28. validate_episode utility catches structural errors
    def test_validate_episode():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20, feature_dim=32)
        sampler = ds.get_sampler("train", 5, 2, 3, 42)
        ep = sampler.sample_episode(0, 0)
        # Valid episode should produce no errors
        errs = validate_episode(ep, 5, 2, 3)
        assert len(errs) == 0, f"Unexpected errors: {errs}"
        # Wrong n_way should produce errors
        errs2 = validate_episode(ep, 3, 2, 3)
        assert len(errs2) > 0, "Expected errors for wrong n_way"
        # Wrong k_shot should produce errors
        errs3 = validate_episode(ep, 5, 1, 3)
        assert len(errs3) > 0, "Expected errors for wrong k_shot"
    run_test("28. validate_episode utility catches structural errors", test_validate_episode)

    # 29. create_dataloader convenience function
    def test_create_dataloader():
        loader = create_dataloader(
            "synthetic", "train", n_way=5, k_shot=1, q_query=5,
            episodes_per_epoch=10, seed=42,
            num_classes=50, samples_per_class=20, feature_dim=32,
        )
        assert isinstance(loader, EpisodeDataLoader)
        loader.set_epoch(0)
        eps = list(loader)
        assert len(eps) == 10
        assert all(isinstance(e, Episode) for e in eps)
        # With task batching
        loader_b = create_dataloader(
            "synthetic", "train", n_way=5, k_shot=1, q_query=5,
            episodes_per_epoch=10, seed=42, collate_to_task_batch=True,
            task_batch_size=3, num_classes=50, samples_per_class=20, feature_dim=32,
        )
        loader_b.set_epoch(0)
        batches = list(loader_b)
        assert all(isinstance(b, TaskBatch) for b in batches)
        assert sum(len(b) for b in batches) == 10
    run_test("29. create_dataloader convenience function", test_create_dataloader)

    # 30. Sampler sample_batch produces correct count
    def test_sample_batch():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20, feature_dim=32)
        sampler = ds.get_sampler("train", 5, 1, 5, 42)
        batch = sampler.sample_batch(epoch=0, start_idx=10, batch_size=8)
        assert len(batch) == 8
        # Each episode should have correct idx
        for i, ep in enumerate(batch):
            assert ep.episode_id == (0, 10 + i)
        # Episodes should be individually deterministic
        for i, ep in enumerate(batch):
            ref = sampler.sample_episode(0, 10 + i)
            assert torch.equal(ep.support_x, ref.support_x)
            assert ep.class_ids == ref.class_ids
    run_test("30. Sampler sample_batch produces correct count and IDs", test_sample_batch)

    # 31. SyntheticFewShotDataset repr and get_class_split fractions
    def test_synthetic_repr_split():
        ds = SyntheticFewShotDataset(num_classes=100, samples_per_class=10, feature_dim=16)
        r = repr(ds)
        assert "SyntheticFewShotDataset" in r and "100" in r
        # Default split: 64/16/20
        sp = ds.get_class_split()
        assert len(sp.train) == 64 and len(sp.val) == 16 and len(sp.test) == 20
        sp.validate()
        # Custom split fractions
        sp2 = ds.get_class_split(train_frac=0.5, val_frac=0.2)
        assert len(sp2.train) == 50 and len(sp2.val) == 20 and len(sp2.test) == 30
        sp2.validate()
    run_test("31. SyntheticFewShotDataset repr and custom split fractions", test_synthetic_repr_split)

    # 32. EpisodeDataLoader with batching and partial final batch
    def test_loader_partial_batch():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20, feature_dim=32)
        sampler = ds.get_sampler("train", 5, 1, 5, 42)
        # 7 episodes, batch size 3 -> 3 batches (3, 3, 1)
        loader = EpisodeDataLoader(
            sampler, episodes_per_epoch=7,
            collate_to_task_batch=True, task_batch_size=3,
        )
        loader.set_epoch(0)
        batches = list(loader)
        sizes = [len(b) for b in batches]
        assert sizes == [3, 3, 1], f"Expected [3, 3, 1], got {sizes}"
        assert sum(sizes) == 7
    run_test("32. EpisodeDataLoader partial final batch", test_loader_partial_batch)

    # 33. Multiple sampler instances with same config are independent
    def test_independent_samplers():
        ds = SyntheticFewShotDataset(num_classes=50, samples_per_class=20, feature_dim=32)
        s1 = ds.get_sampler("train", 5, 1, 5, 42)
        s2 = ds.get_sampler("train", 5, 1, 5, 42)
        # Both should produce identical episodes
        for i in range(20):
            e1, e2 = s1.sample_episode(0, i), s2.sample_episode(0, i)
            assert torch.equal(e1.support_x, e2.support_x)
            assert e1.class_ids == e2.class_ids
        # Sampling from s1 should not affect s2
        _ = s1.sample_episode(99, 999)
        e2_check = s2.sample_episode(0, 0)
        e1_check = s1.sample_episode(0, 0)
        assert torch.equal(e1_check.support_x, e2_check.support_x)
    run_test("33. Multiple sampler instances are independent", test_independent_samplers)

    # Summary
    print(f"\n{'=' * 72}")
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print("=" * 72)
    if passed_tests < total_tests:
        print("\nFailed:")
        for n, ok, m in results:
            if not ok: print(f"  - {n}: {m}")
    else:
        print("\nAll tests passed.")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    _run_self_tests()
