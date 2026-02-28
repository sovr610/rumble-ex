# Copyright (c) Meta Platforms, Inc. and affiliates.
# MIT License
#
# assessment_config_template.py
#
# AssessmentConfig: central configuration dataclass for all V-JEPA 2
# assessment tasks. Includes factory presets for the standard benchmark tasks.

from __future__ import annotations

import copy
import unittest
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# AssessmentConfig
# ---------------------------------------------------------------------------

@dataclass
class AssessmentConfig:
    """
    Configuration for V-JEPA 2 frozen backbone assessment pipelines.

    All hyperparameters are defined here; nothing should be hardcoded in
    training or assessment scripts.

    Attributes:
        task:              Assessment task name. One of:
                           "video_classification", "image_classification",
                           "action_anticipation".
        num_classes:       Number of output classes. Overridden by task presets.
        embed_dim:         Encoder output dimensionality (set from model spec).
        num_queries:       Number of query tokens in AttentivePooler.
        probe_depth:       Depth of the AttentivePooler (number of cross-attn blocks).
        num_heads:         Number of attention heads in the AttentivePooler.
        multihead_kwargs:  List of dicts, each specifying {"lr", "wd"} for one probe head.
                           Empty list means single head with default lr/wd.
        num_segments:      Number of temporal segments per video (at test time).
        num_views:         Number of spatial crops per segment (at test time).
        val_only:          If True, skip training and only run validation.
        # Optimizer defaults
        lr:                Default learning rate for single-head probing.
        weight_decay:      Default weight decay.
        momentum:          SGD momentum.
        num_epochs:        Number of training epochs.
        batch_size:        Training batch size.
        # Action anticipation
        num_verbs:         Number of verb classes (EPIC-Kitchens: 97).
        num_nouns:         Number of noun classes (EPIC-Kitchens: 300).
        num_actions:       Number of action pair classes (EPIC-Kitchens: 3806).
        focal_alpha:       Focal loss alpha parameter.
        focal_gamma:       Focal loss gamma parameter.
        anticipation_time_sec:    Seconds before action onset to make prediction.
        anticipation_point:       Fraction of clip used as context.
        anticipation_duration:    Duration of future to anticipate (seconds).
        # Checkpoint / output
        output_dir:        Directory for checkpoints and logs.
        checkpoint_freq:   Save a checkpoint every N epochs (0 = only at end).
        log_freq:          Log metrics every N steps.
        # Model
        model_name:        V-JEPA 2 model variant ("vit_large", "vit_giant", "ac_vit_giant").
        pretrained:        Load pretrained weights.
    """

    # ---- Task ----
    task: str                        = "video_classification"
    num_classes: int                 = 174   # SSv2 default

    # ---- Model ----
    embed_dim: int                   = 1408  # ViT-Giant default
    model_name: str                  = "vit_giant"
    pretrained: bool                 = True

    # ---- Probe architecture ----
    num_queries: int                 = 1
    probe_depth: int                 = 1
    num_heads: int                   = 1
    dropout: float                   = 0.0

    # ---- Multi-head search ----
    multihead_kwargs: List[Dict]     = field(default_factory=list)

    # ---- Assessment protocol ----
    num_segments: int                = 1
    num_views: int                   = 3
    val_only: bool                   = False

    # ---- Optimizer ----
    lr: float                        = 1e-3
    weight_decay: float              = 1e-4
    momentum: float                  = 0.9
    num_epochs: int                  = 10
    batch_size: int                  = 64

    # ---- Action anticipation ----
    num_verbs: int                   = 97
    num_nouns: int                   = 300
    num_actions: int                 = 3806
    focal_alpha: float               = 0.25
    focal_gamma: float               = 2.0
    anticipation_time_sec: float     = 1.0
    anticipation_point: float        = 0.5
    anticipation_duration: float     = 0.5

    # ---- Checkpoint / output ----
    output_dir: str                  = "outputs"
    checkpoint_freq: int             = 1
    log_freq: int                    = 50

    # ---- Misc ----
    seed: int                        = 0
    num_workers: int                 = 4
    pin_memory: bool                 = True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        valid_tasks = {"video_classification", "image_classification", "action_anticipation"}
        if self.task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}, got {self.task!r}")

        valid_models = {"vit_small", "vit_base", "vit_large", "vit_giant", "ac_vit_giant"}
        if self.model_name not in valid_models:
            raise ValueError(f"model_name must be one of {valid_models}, got {self.model_name!r}")

        if self.num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {self.num_classes}")
        if self.embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {self.embed_dim}")
        if self.num_queries < 1:
            raise ValueError(f"num_queries must be >= 1, got {self.num_queries}")
        if self.probe_depth < 1:
            raise ValueError(f"probe_depth must be >= 1, got {self.probe_depth}")
        if self.num_segments < 1:
            raise ValueError(f"num_segments must be >= 1, got {self.num_segments}")
        if self.num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {self.num_views}")
        if not (0.0 < self.focal_alpha <= 1.0):
            raise ValueError(f"focal_alpha must be in (0, 1], got {self.focal_alpha}")
        if self.focal_gamma < 0.0:
            raise ValueError(f"focal_gamma must be >= 0, got {self.focal_gamma}")
        if not (0.0 < self.anticipation_point < 1.0):
            raise ValueError(
                f"anticipation_point must be in (0, 1), got {self.anticipation_point}"
            )
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a plain dictionary for YAML/JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AssessmentConfig":
        """Create AssessmentConfig from a plain dictionary."""
        return cls(**d)

    def clone(self, **overrides) -> "AssessmentConfig":
        """Return a copy with specific fields overridden."""
        d = self.to_dict()
        d.update(overrides)
        return AssessmentConfig.from_dict(d)

    # ------------------------------------------------------------------
    # Task presets
    # ------------------------------------------------------------------

    @classmethod
    def video_classification_ssv2(cls) -> "AssessmentConfig":
        """
        Something-Something-v2 (SSv2) video classification.

        174 classes. Standard protocol: 1 segment, 3 views at test time.
        Multi-head search over 8 (lr, wd) combinations.
        """
        return cls(
            task="video_classification",
            num_classes=174,
            embed_dim=1408,
            model_name="vit_giant",
            num_queries=1,
            probe_depth=1,
            num_heads=1,
            num_segments=1,
            num_views=3,
            num_epochs=15,
            batch_size=64,
            multihead_kwargs=[
                {"lr": 1e-3, "wd": 1e-4},
                {"lr": 2e-3, "wd": 1e-4},
                {"lr": 5e-4, "wd": 1e-4},
                {"lr": 1e-3, "wd": 1e-3},
                {"lr": 2e-3, "wd": 1e-3},
                {"lr": 5e-4, "wd": 1e-3},
                {"lr": 1e-2, "wd": 0.0},
                {"lr": 5e-3, "wd": 0.0},
            ],
        )

    @classmethod
    def video_classification_kinetics(cls) -> "AssessmentConfig":
        """
        Kinetics-400 video classification.

        400 classes. Multi-segment assessment: 5 segments x 3 views = 15 clips.
        """
        return cls(
            task="video_classification",
            num_classes=400,
            embed_dim=1408,
            model_name="vit_giant",
            num_queries=1,
            probe_depth=1,
            num_heads=1,
            num_segments=5,
            num_views=3,
            num_epochs=15,
            batch_size=64,
            multihead_kwargs=[
                {"lr": 1e-3, "wd": 1e-4},
                {"lr": 2e-3, "wd": 1e-4},
                {"lr": 5e-4, "wd": 1e-3},
                {"lr": 1e-2, "wd": 0.0},
            ],
        )

    @classmethod
    def video_classification_diving48(cls) -> "AssessmentConfig":
        """
        Diving-48 fine-grained video classification.

        48 classes. Requires temporal reasoning — standard 5x3 protocol.
        """
        return cls(
            task="video_classification",
            num_classes=48,
            embed_dim=1408,
            model_name="vit_giant",
            num_queries=1,
            probe_depth=1,
            num_segments=5,
            num_views=3,
            num_epochs=30,
            batch_size=32,
            multihead_kwargs=[
                {"lr": 1e-3, "wd": 1e-4},
                {"lr": 5e-3, "wd": 1e-4},
                {"lr": 1e-2, "wd": 0.0},
            ],
        )

    @classmethod
    def image_classification_imagenet(cls) -> "AssessmentConfig":
        """
        ImageNet-1K image classification.

        1000 classes. Each image treated as a single-frame "video".
        Uses timm preprocessing transforms.
        """
        return cls(
            task="image_classification",
            num_classes=1000,
            embed_dim=1408,
            model_name="vit_giant",
            num_queries=1,
            probe_depth=1,
            num_heads=1,
            num_segments=1,
            num_views=1,
            num_epochs=20,
            batch_size=256,
            multihead_kwargs=[
                {"lr": 1e-3, "wd": 1e-4},
                {"lr": 2e-3, "wd": 1e-4},
                {"lr": 5e-4, "wd": 1e-3},
                {"lr": 1e-2, "wd": 0.0},
            ],
        )

    @classmethod
    def action_anticipation_epic(cls) -> "AssessmentConfig":
        """
        EPIC-Kitchens 100 action anticipation.

        97 verbs, 300 nouns, 3806 action pairs.
        Metric: Class-Mean Recall @ 5.
        Uses focal loss (alpha=0.25, gamma=2.0).
        Anticipation: 1 second before action onset.
        """
        return cls(
            task="action_anticipation",
            num_classes=3806,    # action pairs (used as primary num_classes)
            embed_dim=1408,
            model_name="ac_vit_giant",
            num_queries=3,
            probe_depth=1,
            num_heads=1,
            num_segments=1,
            num_views=1,
            num_epochs=20,
            batch_size=32,
            multihead_kwargs=[
                {"lr": 1e-3, "wd": 1e-4},
                {"lr": 2e-3, "wd": 1e-4},
            ],
            # Action-specific fields
            num_verbs=97,
            num_nouns=300,
            num_actions=3806,
            focal_alpha=0.25,
            focal_gamma=2.0,
            anticipation_time_sec=1.0,
            anticipation_point=0.5,
            anticipation_duration=0.5,
        )


# ---------------------------------------------------------------------------
# YAML I/O helpers (optional dependency)
# ---------------------------------------------------------------------------

def save_config(config: AssessmentConfig, path: str) -> None:
    """Save AssessmentConfig to a YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required for save_config(). pip install pyyaml")
    with open(path, "w") as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False)


def load_config(path: str) -> AssessmentConfig:
    """Load AssessmentConfig from a YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required for load_config(). pip install pyyaml")
    with open(path) as f:
        d = yaml.safe_load(f)
    return AssessmentConfig.from_dict(d)


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------

class _TestAssessmentConfig(unittest.TestCase):

    def test_default_task(self):
        cfg = AssessmentConfig()
        self.assertEqual(cfg.task, "video_classification")

    def test_default_num_classes(self):
        cfg = AssessmentConfig()
        self.assertEqual(cfg.num_classes, 174)

    def test_invalid_task_raises(self):
        with self.assertRaises(ValueError):
            AssessmentConfig(task="unsupported_task")

    def test_invalid_num_classes_raises(self):
        with self.assertRaises(ValueError):
            AssessmentConfig(num_classes=0)

    def test_invalid_focal_alpha_raises(self):
        with self.assertRaises(ValueError):
            AssessmentConfig(focal_alpha=0.0)

    def test_invalid_focal_gamma_raises(self):
        with self.assertRaises(ValueError):
            AssessmentConfig(focal_gamma=-1.0)

    def test_invalid_anticipation_point_raises(self):
        with self.assertRaises(ValueError):
            AssessmentConfig(anticipation_point=0.0)
        with self.assertRaises(ValueError):
            AssessmentConfig(anticipation_point=1.0)

    def test_invalid_lr_raises(self):
        with self.assertRaises(ValueError):
            AssessmentConfig(lr=0.0)

    def test_multihead_kwargs_default_empty(self):
        cfg = AssessmentConfig()
        self.assertEqual(len(cfg.multihead_kwargs), 0)


class _TestPresets(unittest.TestCase):

    def test_ssv2_num_classes(self):
        cfg = AssessmentConfig.video_classification_ssv2()
        self.assertEqual(cfg.num_classes, 174)

    def test_ssv2_task(self):
        cfg = AssessmentConfig.video_classification_ssv2()
        self.assertEqual(cfg.task, "video_classification")

    def test_ssv2_has_multihead_kwargs(self):
        cfg = AssessmentConfig.video_classification_ssv2()
        self.assertGreater(len(cfg.multihead_kwargs), 0)

    def test_imagenet_num_classes(self):
        cfg = AssessmentConfig.image_classification_imagenet()
        self.assertEqual(cfg.num_classes, 1000)

    def test_imagenet_task(self):
        cfg = AssessmentConfig.image_classification_imagenet()
        self.assertEqual(cfg.task, "image_classification")

    def test_epic_num_verbs(self):
        cfg = AssessmentConfig.action_anticipation_epic()
        self.assertEqual(cfg.num_verbs, 97)

    def test_epic_num_nouns(self):
        cfg = AssessmentConfig.action_anticipation_epic()
        self.assertEqual(cfg.num_nouns, 300)

    def test_epic_num_actions(self):
        cfg = AssessmentConfig.action_anticipation_epic()
        self.assertEqual(cfg.num_actions, 3806)

    def test_epic_task(self):
        cfg = AssessmentConfig.action_anticipation_epic()
        self.assertEqual(cfg.task, "action_anticipation")

    def test_epic_focal_alpha_in_range(self):
        cfg = AssessmentConfig.action_anticipation_epic()
        self.assertGreater(cfg.focal_alpha, 0.0)
        self.assertLessEqual(cfg.focal_alpha, 1.0)

    def test_epic_focal_gamma_nonnegative(self):
        cfg = AssessmentConfig.action_anticipation_epic()
        self.assertGreaterEqual(cfg.focal_gamma, 0.0)

    def test_diving48_num_classes(self):
        cfg = AssessmentConfig.video_classification_diving48()
        self.assertEqual(cfg.num_classes, 48)

    def test_kinetics_num_classes(self):
        cfg = AssessmentConfig.video_classification_kinetics()
        self.assertEqual(cfg.num_classes, 400)

    def test_kinetics_multi_segment(self):
        cfg = AssessmentConfig.video_classification_kinetics()
        self.assertGreater(cfg.num_segments, 1)

    def test_all_presets_are_valid(self):
        """All presets should pass validation without raising."""
        for preset_fn in [
            AssessmentConfig.video_classification_ssv2,
            AssessmentConfig.video_classification_kinetics,
            AssessmentConfig.video_classification_diving48,
            AssessmentConfig.image_classification_imagenet,
            AssessmentConfig.action_anticipation_epic,
        ]:
            cfg = preset_fn()
            self.assertIsInstance(cfg, AssessmentConfig)


class _TestConfigRoundTrip(unittest.TestCase):

    def test_to_dict_and_back(self):
        cfg1 = AssessmentConfig.video_classification_ssv2()
        d    = cfg1.to_dict()
        cfg2 = AssessmentConfig.from_dict(d)
        self.assertEqual(cfg1.task,        cfg2.task)
        self.assertEqual(cfg1.num_classes, cfg2.num_classes)
        self.assertEqual(cfg1.embed_dim,   cfg2.embed_dim)

    def test_clone_with_override(self):
        cfg1 = AssessmentConfig()
        cfg2 = cfg1.clone(num_classes=1000, task="image_classification")
        self.assertEqual(cfg2.num_classes, 1000)
        self.assertEqual(cfg2.task,        "image_classification")
        self.assertEqual(cfg1.num_classes, 174)   # original unchanged

    def test_clone_preserves_multihead_kwargs(self):
        cfg1 = AssessmentConfig.video_classification_ssv2()
        cfg2 = cfg1.clone(num_epochs=30)
        self.assertEqual(cfg2.num_epochs,        30)
        self.assertEqual(len(cfg2.multihead_kwargs), len(cfg1.multihead_kwargs))


if __name__ == "__main__":
    print("Running AssessmentConfig self-tests...")
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(_TestAssessmentConfig)
    suite.addTests(loader.loadTestsFromTestCase(_TestPresets))
    suite.addTests(loader.loadTestsFromTestCase(_TestConfigRoundTrip))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nAll self-tests passed.")
    else:
        raise SystemExit(1)
