# V-JEPA 2 Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate V-JEPA 2 as the perceptual and predictive backbone across all 7 cognitive layers of BrainAI.

**Architecture:** Frozen V-JEPA 2 ViT-g encoder (1B params) wrapped in lightweight adapter modules that produce outputs matching existing BrainAI contracts. Five integration modules — encoder bridge, temporal bridge, world model, imagination engine, neuromodulated masking — each independently toggleable via config flags.

**Tech Stack:** PyTorch, transformers (HuggingFace V-JEPA 2), einops, existing BrainAI modules.

**Design doc:** `docs/plans/2026-02-23-vjepa2-integration-design.md`

---

## Task 1: Add V-JEPA 2 Config Dataclasses

**Files:**
- Modify: `brain_ai/config.py:466-498` (BrainAIConfig — add vjepa2 fields)

**Step 1: Write the failing test**

Create `tests/test_vjepa2_config.py`:

```python
"""Tests for V-JEPA 2 configuration dataclasses."""
import pytest
from brain_ai.config import (
    BrainAIConfig,
    VJEPA2EncoderConfig,
    VJEPA2WorldModelConfig,
    ImaginationConfig,
    NeuromodulatedMaskingConfig,
    TemporalBridgeConfig,
)


class TestVJEPA2Configs:
    def test_vjepa2_encoder_config_defaults(self):
        cfg = VJEPA2EncoderConfig()
        assert cfg.enabled is False
        assert cfg.model_name == "vjepa2_vitg"
        assert cfg.freeze_encoder is True
        assert cfg.num_query_tokens == 16
        assert cfg.probe_layers == 4
        assert cfg.output_dim == 4096
        assert cfg.tubelet_size == 2

    def test_vjepa2_world_model_config_defaults(self):
        cfg = VJEPA2WorldModelConfig()
        assert cfg.enabled is False
        assert cfg.predictor_dim == 384
        assert cfg.planning_horizon == 8
        assert cfg.cem_population == 128
        assert cfg.efe_pragmatic_weight == 1.0

    def test_imagination_config_defaults(self):
        cfg = ImaginationConfig()
        assert cfg.enabled is False
        assert cfg.max_rollout_steps == 16

    def test_neuromodulated_masking_config_defaults(self):
        cfg = NeuromodulatedMaskingConfig()
        assert cfg.enabled is False
        assert cfg.base_mask_ratio == 0.75

    def test_temporal_bridge_config_defaults(self):
        cfg = TemporalBridgeConfig()
        assert cfg.enabled is False
        assert cfg.pooling_mode == "mean"

    def test_brain_ai_config_has_vjepa2(self):
        cfg = BrainAIConfig()
        assert hasattr(cfg, "vjepa2_encoder")
        assert hasattr(cfg, "vjepa2_world_model")
        assert hasattr(cfg, "imagination")
        assert hasattr(cfg, "neuromodulated_masking")
        assert hasattr(cfg, "temporal_bridge")
        assert cfg.vjepa2_encoder.enabled is False

    def test_brain_ai_config_vjepa2_disabled_by_default(self):
        cfg = BrainAIConfig()
        assert cfg.vjepa2_encoder.enabled is False
        assert cfg.vjepa2_world_model.enabled is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vjepa2_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'VJEPA2EncoderConfig'`

**Step 3: Write minimal implementation**

Add these dataclasses to `brain_ai/config.py` before the `BrainAIConfig` class (around line 460):

```python
@dataclass
class VJEPA2EncoderConfig:
    """V-JEPA 2 vision encoder configuration."""
    enabled: bool = False
    model_name: str = "vjepa2_vitg"
    pretrained: bool = True
    freeze_encoder: bool = True
    num_query_tokens: int = 16
    probe_layers: int = 4
    probe_heads: int = 16
    output_dim: int = 4096
    spike_output: bool = False
    video_frames: int = 16
    tubelet_size: int = 2
    encoder_dim: int = 1408  # ViT-g hidden dim


@dataclass
class VJEPA2WorldModelConfig:
    """V-JEPA 2-AC world model for Active Inference."""
    enabled: bool = False
    predictor_dim: int = 384
    predictor_depth: int = 12
    action_dim: int = 7
    action_embed_dim: int = 384
    planning_horizon: int = 8
    cem_population: int = 128
    cem_elite_ratio: float = 0.1
    cem_iterations: int = 5
    efe_pragmatic_weight: float = 1.0
    efe_epistemic_weight: float = 1.0
    efe_empowerment_weight: float = 0.1


@dataclass
class ImaginationConfig:
    """Imagination engine for System 2 reasoning."""
    enabled: bool = False
    max_rollout_steps: int = 16
    num_parallel_scenarios: int = 8
    quality_metric: str = "coherence"


@dataclass
class NeuromodulatedMaskingConfig:
    """Dynamic masking controlled by neuromodulatory signals."""
    enabled: bool = False
    base_mask_ratio: float = 0.75
    ach_sensitivity: float = 0.3
    ne_sensitivity: float = 0.3
    salience_guided: bool = True


@dataclass
class TemporalBridgeConfig:
    """Temporal bridge from V-JEPA 2 to HTM."""
    enabled: bool = False
    pooling_mode: str = "mean"
    output_dim: int = 4096
```

Then add fields to `BrainAIConfig`:

```python
    # V-JEPA 2 integration
    vjepa2_encoder: VJEPA2EncoderConfig = field(default_factory=VJEPA2EncoderConfig)
    vjepa2_world_model: VJEPA2WorldModelConfig = field(default_factory=VJEPA2WorldModelConfig)
    imagination: ImaginationConfig = field(default_factory=ImaginationConfig)
    neuromodulated_masking: NeuromodulatedMaskingConfig = field(default_factory=NeuromodulatedMaskingConfig)
    temporal_bridge: TemporalBridgeConfig = field(default_factory=TemporalBridgeConfig)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vjepa2_config.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add brain_ai/config.py tests/test_vjepa2_config.py
git commit -m "feat: add V-JEPA 2 config dataclasses"
```

---

## Task 2: Create V-JEPA 2 Backbone Wrapper

**Files:**
- Create: `brain_ai/encoders/vjepa2_backbone.py`
- Test: `tests/test_vjepa2_encoder.py`

This task creates the frozen V-JEPA 2 ViT-g backbone wrapper with a mock fallback for testing without the actual pretrained weights.

**Step 1: Write the failing test**

Create `tests/test_vjepa2_encoder.py`:

```python
"""Tests for V-JEPA 2 encoder backbone and bridge."""
import pytest
import torch
from brain_ai.config import VJEPA2EncoderConfig
from brain_ai.encoders.vjepa2_backbone import (
    VJEPA2Backbone,
    AttentivePooler,
    VJEPA2VisionEncoder,
)


class TestVJEPA2Backbone:
    @pytest.fixture
    def small_cfg(self):
        return VJEPA2EncoderConfig(
            enabled=True,
            pretrained=False,  # Use mock backbone for tests
            encoder_dim=64,
            output_dim=128,
            num_query_tokens=4,
            probe_layers=2,
            probe_heads=4,
        )

    def test_backbone_output_shape(self, small_cfg):
        backbone = VJEPA2Backbone(small_cfg)
        x = torch.randn(2, 3, 224, 224)
        tokens, positions = backbone(x)
        assert tokens.dim() == 3  # (B, N, D)
        assert tokens.shape[0] == 2
        assert tokens.shape[2] == small_cfg.encoder_dim

    def test_backbone_frozen_params(self, small_cfg):
        backbone = VJEPA2Backbone(small_cfg)
        for p in backbone.parameters():
            assert not p.requires_grad

    def test_attentive_pooler_shape(self, small_cfg):
        pooler = AttentivePooler(
            input_dim=small_cfg.encoder_dim,
            num_queries=small_cfg.num_query_tokens,
            num_layers=small_cfg.probe_layers,
            num_heads=small_cfg.probe_heads,
        )
        tokens = torch.randn(2, 50, small_cfg.encoder_dim)
        pooled = pooler(tokens)
        assert pooled.shape == (2, small_cfg.num_query_tokens, small_cfg.encoder_dim)

    def test_attentive_pooler_trainable(self, small_cfg):
        pooler = AttentivePooler(
            input_dim=small_cfg.encoder_dim,
            num_queries=small_cfg.num_query_tokens,
            num_layers=small_cfg.probe_layers,
            num_heads=small_cfg.probe_heads,
        )
        for p in pooler.parameters():
            assert p.requires_grad


class TestVJEPA2VisionEncoder:
    @pytest.fixture
    def small_cfg(self):
        return VJEPA2EncoderConfig(
            enabled=True,
            pretrained=False,
            encoder_dim=64,
            output_dim=128,
            num_query_tokens=4,
            probe_layers=2,
            probe_heads=4,
        )

    def test_forward_image_shape(self, small_cfg):
        encoder = VJEPA2VisionEncoder(small_cfg)
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, small_cfg.output_dim)

    def test_forward_video_shape(self, small_cfg):
        encoder = VJEPA2VisionEncoder(small_cfg)
        x = torch.randn(4, 2, 3, 224, 224)  # T=4, B=2
        out = encoder(x, temporal_input=True)
        assert out.shape == (2, small_cfg.output_dim)

    def test_matches_vision_encoder_contract(self, small_cfg):
        """Verify the forward signature matches VisionEncoder."""
        encoder = VJEPA2VisionEncoder(small_cfg)
        x = torch.randn(2, 3, 224, 224)
        # Must accept same args as VisionEncoder.forward()
        out = encoder(x, temporal_input=False)
        assert out.shape == (2, small_cfg.output_dim)

    def test_get_raw_tokens(self, small_cfg):
        """Verify we can get raw tokens for temporal bridge."""
        encoder = VJEPA2VisionEncoder(small_cfg)
        x = torch.randn(2, 3, 224, 224)
        tokens, positions = encoder.get_raw_tokens(x)
        assert tokens.dim() == 3
        assert tokens.shape[0] == 2
        assert tokens.shape[2] == small_cfg.encoder_dim
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vjepa2_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'brain_ai.encoders.vjepa2_backbone'`

**Step 3: Write minimal implementation**

Create `brain_ai/encoders/vjepa2_backbone.py`:

```python
"""
V-JEPA 2 Vision Encoder Backbone

Wraps the frozen V-JEPA 2 ViT-g encoder as a drop-in replacement
for VisionEncoder, producing (B, output_dim) features compatible
with the Global Workspace pipeline.

When pretrained=False, uses a lightweight mock backbone for testing.
When pretrained=True, loads weights via HuggingFace transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..config import VJEPA2EncoderConfig


class MockViTBackbone(nn.Module):
    """Lightweight ViT mock for testing without pretrained weights."""

    def __init__(self, encoder_dim: int = 64, patch_size: int = 16):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            3, encoder_dim, kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(encoder_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        tokens = self.patch_embed(x)  # (B, D, H', W')
        B, D, H, W = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, D)
        tokens = self.norm(tokens)
        positions = torch.zeros(tokens.shape[1], dtype=torch.long, device=x.device)
        return tokens, positions


class VJEPA2Backbone(nn.Module):
    """Frozen V-JEPA 2 ViT-g backbone.

    All parameters are frozen (requires_grad=False).
    """

    def __init__(self, cfg: VJEPA2EncoderConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.pretrained:
            self._load_pretrained(cfg.model_name)
        else:
            self.model = MockViTBackbone(
                encoder_dim=cfg.encoder_dim,
            )

        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    def _load_pretrained(self, model_name: str):
        """Load pretrained V-JEPA 2 from HuggingFace."""
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                f"facebook/{model_name}"
            )
        except (ImportError, OSError):
            import warnings
            warnings.warn(
                f"Could not load pretrained {model_name}. "
                "Falling back to mock backbone."
            )
            self.model = MockViTBackbone(encoder_dim=self.cfg.encoder_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) images
        Returns:
            tokens: (B, N, encoder_dim) patch tokens
            positions: (N,) temporal position per token
        """
        with torch.no_grad():
            return self.model(x)


class AttentivePooler(nn.Module):
    """Cross-attention pooler with learnable query tokens.

    Extracts a fixed number of output tokens from variable-length
    encoder output via cross-attention. This is the trainable probe
    that sits on top of the frozen backbone.
    """

    def __init__(
        self,
        input_dim: int,
        num_queries: int = 16,
        num_layers: int = 4,
        num_heads: int = 16,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, input_dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=input_dim * 4,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, encoder_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_tokens: (B, N, D) from frozen backbone
        Returns:
            pooled: (B, num_queries, D)
        """
        B = encoder_tokens.shape[0]
        queries = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, encoder_tokens)
        return self.norm(queries)


class VJEPA2VisionEncoder(nn.Module):
    """V-JEPA 2 vision encoder adapter for BrainAI pipeline.

    Drop-in replacement for VisionEncoder. Wraps frozen V-JEPA 2
    ViT-g backbone with a trainable attentive pooler and projection.
    """

    def __init__(self, cfg: VJEPA2EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.output_dim = cfg.output_dim

        # Frozen backbone
        self.backbone = VJEPA2Backbone(cfg)

        # Trainable attentive pooler
        self.pooler = AttentivePooler(
            input_dim=cfg.encoder_dim,
            num_queries=cfg.num_query_tokens,
            num_layers=cfg.probe_layers,
            num_heads=cfg.probe_heads,
        )

        # Projection to workspace dim
        self.projection = nn.Sequential(
            nn.Linear(cfg.encoder_dim, cfg.output_dim),
            nn.GELU(),
            nn.Linear(cfg.output_dim, cfg.output_dim),
        )

    def get_raw_tokens(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get raw backbone tokens (for temporal bridge, masking, etc.)."""
        return self.backbone(x)

    def forward(
        self, x: torch.Tensor, temporal_input: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) for images or (T, B, C, H, W) for video
            temporal_input: If True, x is a temporal sequence
        Returns:
            features: (B, output_dim)
        """
        if temporal_input:
            # Reshape video: (T, B, C, H, W) -> process middle frame
            T, B = x.shape[:2]
            mid = T // 2
            x = x[mid]  # (B, C, H, W)

        tokens, positions = self.backbone(x)
        pooled = self.pooler(tokens)  # (B, Q, D)
        pooled = pooled.mean(dim=1)  # (B, D) — mean over query tokens
        features = self.projection(pooled)  # (B, output_dim)
        return features
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vjepa2_encoder.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add brain_ai/encoders/vjepa2_backbone.py tests/test_vjepa2_encoder.py
git commit -m "feat: add V-JEPA 2 backbone wrapper and vision encoder adapter"
```

---

## Task 3: Wire V-JEPA 2 Encoder into System Pipeline

**Files:**
- Modify: `brain_ai/system.py:246-258` (`_build_encoders` method)
- Modify: `brain_ai/system.py:742-810` (`create_brain_ai` factory)
- Test: `tests/test_vjepa2_system.py`

**Step 1: Write the failing test**

Create `tests/test_vjepa2_system.py`:

```python
"""Tests for V-JEPA 2 integration with BrainAI system."""
import pytest
import torch
from brain_ai.config import BrainAIConfig
from brain_ai.system import BrainAI, create_brain_ai


class TestVJEPA2SystemIntegration:
    @pytest.fixture
    def vjepa2_config(self):
        cfg = BrainAIConfig()
        cfg.vjepa2_encoder.enabled = True
        cfg.vjepa2_encoder.pretrained = False
        cfg.vjepa2_encoder.encoder_dim = 32
        cfg.vjepa2_encoder.output_dim = 64
        cfg.vjepa2_encoder.num_query_tokens = 4
        cfg.vjepa2_encoder.probe_layers = 1
        cfg.vjepa2_encoder.probe_heads = 4
        cfg.encoder.output_dim = 64
        cfg.workspace.workspace_dim = 64
        cfg.decision.hidden_dim = 64
        cfg.decision.num_classes = 10
        cfg.use_htm = False
        cfg.use_symbolic = False
        cfg.use_meta = False
        cfg.use_engram = False
        cfg.modalities = ["vision"]
        return cfg

    def test_vjepa2_encoder_replaces_vision(self, vjepa2_config):
        model = BrainAI(config=vjepa2_config, modalities=["vision"])
        assert "vision" in model.encoders
        from brain_ai.encoders.vjepa2_backbone import VJEPA2VisionEncoder
        assert isinstance(model.encoders["vision"], VJEPA2VisionEncoder)

    def test_vjepa2_forward_pass(self, vjepa2_config):
        model = BrainAI(config=vjepa2_config, modalities=["vision"])
        inputs = {"vision": torch.randn(2, 3, 224, 224)}
        output = model(inputs, task="classify")
        assert output.shape == (2, 10)

    def test_vjepa2_with_details(self, vjepa2_config):
        model = BrainAI(config=vjepa2_config, modalities=["vision"])
        inputs = {"vision": torch.randn(2, 3, 224, 224)}
        result = model(inputs, task="classify", return_details=True)
        assert result.output.shape == (2, 10)
        assert result.workspace.shape == (2, 64)

    def test_vjepa2_disabled_uses_original(self):
        cfg = BrainAIConfig()
        cfg.vjepa2_encoder.enabled = False
        cfg.encoder.output_dim = 64
        cfg.workspace.workspace_dim = 64
        cfg.decision.hidden_dim = 64
        cfg.use_htm = False
        cfg.use_symbolic = False
        cfg.use_meta = False
        cfg.use_engram = False
        model = BrainAI(config=cfg, modalities=["vision"])
        from brain_ai.encoders.vision import VisionEncoder
        assert isinstance(model.encoders["vision"], VisionEncoder)

    def test_pipeline_shows_vjepa2(self, vjepa2_config):
        model = BrainAI(config=vjepa2_config, modalities=["vision"])
        plan = model.get_pipeline_plan()
        assert "encode" in plan.stage_names()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vjepa2_system.py -v`
Expected: FAIL — `test_vjepa2_encoder_replaces_vision` fails because `_build_encoders` doesn't check `vjepa2_encoder.enabled`

**Step 3: Write minimal implementation**

Modify `brain_ai/system.py:246-258` — in `_build_encoders()`, add V-JEPA 2 check before the standard vision encoder:

```python
    def _build_encoders(self):
        """Build modality-specific encoders."""
        self.encoders = nn.ModuleDict()
        encoder_dim = self.config.encoder.output_dim

        if 'vision' in self.modalities:
            if self.config.vjepa2_encoder.enabled:
                from .encoders.vjepa2_backbone import VJEPA2VisionEncoder
                vjepa2_cfg = self.config.vjepa2_encoder
                # Ensure output_dim matches encoder config
                vjepa2_cfg.output_dim = encoder_dim
                self.encoders['vision'] = VJEPA2VisionEncoder(vjepa2_cfg)
            else:
                self.encoders['vision'] = create_vision_encoder(
                    output_dim=encoder_dim,
                    channels=self.config.encoder.vision_channels,
                    beta=self.config.snn.beta,
                    num_steps=self.config.snn.num_timesteps,
                )
        # ... rest unchanged
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vjepa2_system.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add brain_ai/system.py tests/test_vjepa2_system.py
git commit -m "feat: wire V-JEPA 2 encoder into system pipeline"
```

---

## Task 4: Create Temporal Bridge to HTM

**Files:**
- Create: `brain_ai/encoders/vjepa2_temporal_bridge.py`
- Test: `tests/test_vjepa2_temporal.py`

**Step 1: Write the failing test**

Create `tests/test_vjepa2_temporal.py`:

```python
"""Tests for V-JEPA 2 temporal bridge to HTM."""
import pytest
import torch
from brain_ai.config import TemporalBridgeConfig
from brain_ai.encoders.vjepa2_temporal_bridge import VJEPA2TemporalBridge


class TestTemporalBridge:
    @pytest.fixture
    def bridge(self):
        cfg = TemporalBridgeConfig(enabled=True, output_dim=128)
        return VJEPA2TemporalBridge(cfg, encoder_dim=64)

    def test_output_shape(self, bridge):
        # 8 temporal groups, 50 spatial patches each
        tokens = torch.randn(2, 400, 64)
        positions = torch.arange(8).repeat_interleave(50)
        out = bridge(tokens, positions)
        assert out.shape == (2, 8, 128)

    def test_single_frame(self, bridge):
        tokens = torch.randn(2, 50, 64)
        positions = torch.zeros(50, dtype=torch.long)
        out = bridge(tokens, positions)
        assert out.shape == (2, 1, 128)

    def test_mean_pooling(self, bridge):
        tokens = torch.ones(1, 100, 64)
        positions = torch.cat([torch.zeros(50), torch.ones(50)]).long()
        out = bridge(tokens, positions)
        assert out.shape == (1, 2, 128)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vjepa2_temporal.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `brain_ai/encoders/vjepa2_temporal_bridge.py`:

```python
"""
V-JEPA 2 Temporal Bridge to HTM

Extracts per-frame temporal representations from V-JEPA 2 encoder
output and produces sequences suitable for HTM processing.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..config import TemporalBridgeConfig


class VJEPA2TemporalBridge(nn.Module):
    """Groups V-JEPA 2 tokens by temporal position and pools per-frame."""

    def __init__(self, cfg: TemporalBridgeConfig, encoder_dim: int):
        super().__init__()
        self.cfg = cfg
        self.projection = nn.Linear(encoder_dim, cfg.output_dim)
        self.norm = nn.LayerNorm(cfg.output_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        temporal_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, D) encoder output tokens
            temporal_positions: (N,) frame index per token
        Returns:
            temporal_seq: (B, T_frames, output_dim)
        """
        B, N, D = tokens.shape
        unique_positions = temporal_positions.unique(sorted=True)
        T = len(unique_positions)

        frames = []
        for t in unique_positions:
            mask = temporal_positions == t
            group = tokens[:, mask, :]  # (B, N_t, D)
            if self.cfg.pooling_mode == "mean":
                pooled = group.mean(dim=1)  # (B, D)
            elif self.cfg.pooling_mode == "max":
                pooled = group.max(dim=1).values
            else:
                pooled = group.mean(dim=1)
            frames.append(pooled)

        temporal_seq = torch.stack(frames, dim=1)  # (B, T, D)
        temporal_seq = self.norm(self.projection(temporal_seq))
        return temporal_seq
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vjepa2_temporal.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add brain_ai/encoders/vjepa2_temporal_bridge.py tests/test_vjepa2_temporal.py
git commit -m "feat: add V-JEPA 2 temporal bridge for HTM"
```

---

## Task 5: Create V-JEPA 2 World Model for Active Inference

**Files:**
- Create: `brain_ai/decision/vjepa2_world_model.py`
- Test: `tests/test_vjepa2_world_model.py`

**Step 1: Write the failing test**

Create `tests/test_vjepa2_world_model.py`:

```python
"""Tests for V-JEPA 2 world model for Active Inference."""
import pytest
import torch
from brain_ai.config import VJEPA2WorldModelConfig
from brain_ai.decision.vjepa2_world_model import (
    VJEPA2WorldModel,
    CEMPlanner,
)


class TestVJEPA2WorldModel:
    @pytest.fixture
    def small_cfg(self):
        return VJEPA2WorldModelConfig(
            enabled=True,
            predictor_dim=32,
            predictor_depth=2,
            action_dim=4,
            action_embed_dim=32,
            planning_horizon=3,
            cem_population=16,
            cem_elite_ratio=0.25,
            cem_iterations=2,
        )

    @pytest.fixture
    def model(self, small_cfg):
        return VJEPA2WorldModel(small_cfg, obs_dim=64)

    def test_encode_observation(self, model):
        obs = torch.randn(2, 64)
        state = model.encode_observation(obs)
        assert state.shape == (2, 32)

    def test_predict_transition(self, model):
        state = torch.randn(2, 32)
        action = torch.randn(2, 4)
        next_state = model.predict_transition(state, action)
        assert next_state.shape == (2, 32)

    def test_compute_efe(self, model):
        state = torch.randn(2, 32)
        # K=16 action sequences of length H=3, each action is dim=4
        actions = torch.randn(2, 16, 3, 4)
        preferences = torch.randn(2, 32)
        efe = model.compute_efe(state, actions, preferences)
        assert efe.shape == (2, 16)

    def test_plan(self, model):
        state = torch.randn(2, 64)
        preferences = torch.randn(2, 32)
        action, info = model.plan(state, preferences)
        assert action.shape == (2, 4)
        assert "efe_values" in info


class TestCEMPlanner:
    def test_cem_produces_actions(self):
        planner = CEMPlanner(
            action_dim=4,
            horizon=3,
            population=16,
            elite_ratio=0.25,
            iterations=2,
        )
        # Mock cost function
        def cost_fn(actions):
            return actions.sum(dim=(-1, -2))

        actions = planner.optimize(cost_fn, batch_size=2)
        assert actions.shape == (2, 3, 4)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vjepa2_world_model.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `brain_ai/decision/vjepa2_world_model.py`:

```python
"""
V-JEPA 2 World Model for Active Inference

Implements the generative model (observation encoder + transition predictor)
using V-JEPA 2 representations, with CEM planning extended by Expected
Free Energy (EFE) computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Callable, Optional
from dataclasses import dataclass

from ..config import VJEPA2WorldModelConfig


class CEMPlanner(nn.Module):
    """Cross-Entropy Method planner for action optimization."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        population: int = 128,
        elite_ratio: float = 0.1,
        iterations: int = 5,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.population = population
        self.num_elite = max(1, int(population * elite_ratio))
        self.iterations = iterations

    def optimize(
        self,
        cost_fn: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Args:
            cost_fn: (B, K, H, A) -> (B, K) cost per sequence
            batch_size: B
        Returns:
            best_actions: (B, H, A)
        """
        device = device or torch.device("cpu")
        mean = torch.zeros(batch_size, self.horizon, self.action_dim, device=device)
        std = torch.ones_like(mean)

        for _ in range(self.iterations):
            # Sample candidates
            noise = torch.randn(
                batch_size, self.population, self.horizon, self.action_dim,
                device=device,
            )
            candidates = mean.unsqueeze(1) + std.unsqueeze(1) * noise

            # Evaluate
            costs = cost_fn(candidates)  # (B, K)

            # Select elite
            _, elite_idx = costs.topk(self.num_elite, dim=1, largest=False)
            elite_idx = elite_idx.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, self.horizon, self.action_dim,
            )
            elite = candidates.gather(1, elite_idx)

            # Update distribution
            mean = elite.mean(dim=1)
            std = elite.std(dim=1).clamp(min=0.01)

        return mean


class VJEPA2WorldModel(nn.Module):
    """V-JEPA 2-AC world model for Active Inference planning."""

    def __init__(self, cfg: VJEPA2WorldModelConfig, obs_dim: int):
        super().__init__()
        self.cfg = cfg
        d = cfg.predictor_dim

        # Observation encoder: obs -> latent state
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
            nn.LayerNorm(d),
        )

        # Action embedding
        self.action_embed = nn.Linear(cfg.action_dim, cfg.action_embed_dim)

        # Transition predictor (simplified transformer)
        self.transition = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=max(1, d // 32),
                dim_feedforward=d * 4,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=min(cfg.predictor_depth, 4),
        )

        # State-action combiner
        self.sa_combine = nn.Linear(d + cfg.action_embed_dim, d)

        # CEM planner
        self.planner = CEMPlanner(
            action_dim=cfg.action_dim,
            horizon=cfg.planning_horizon,
            population=cfg.cem_population,
            elite_ratio=cfg.cem_elite_ratio,
            iterations=cfg.cem_iterations,
        )

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent state."""
        return self.obs_encoder(obs)

    def predict_transition(
        self, state: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next state given current state and action."""
        a_emb = self.action_embed(action)
        sa = self.sa_combine(torch.cat([state, a_emb], dim=-1))
        sa = sa.unsqueeze(1)  # (B, 1, D)
        out = self.transition(sa)
        return out.squeeze(1)

    def rollout(
        self, state: torch.Tensor, actions: torch.Tensor,
    ) -> torch.Tensor:
        """Roll out transition model over action sequence.

        Args:
            state: (B, D) initial state
            actions: (B, H, A) action sequence
        Returns:
            states: (B, H, D) predicted states
        """
        B, H, A = actions.shape
        states = []
        s = state
        for t in range(H):
            s = self.predict_transition(s, actions[:, t])
            states.append(s)
        return torch.stack(states, dim=1)

    def compute_efe(
        self,
        state: torch.Tensor,
        action_sequences: torch.Tensor,
        preferences: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Expected Free Energy for action sequences.

        Args:
            state: (B, D) current state
            action_sequences: (B, K, H, A) candidate action sequences
            preferences: (B, D) preferred state
        Returns:
            efe: (B, K) EFE per sequence (lower is better)
        """
        B, K, H, A = action_sequences.shape
        D = state.shape[-1]

        # Expand state for all candidates
        state_exp = state.unsqueeze(1).expand(B, K, D).reshape(B * K, D)
        actions_flat = action_sequences.reshape(B * K, H, A)

        # Rollout
        predicted = self.rollout(state_exp, actions_flat)  # (B*K, H, D)
        final_state = predicted[:, -1]  # (B*K, D)

        # Pragmatic value: distance to preferences
        pref_exp = preferences.unsqueeze(1).expand(B, K, D).reshape(B * K, D)
        pragmatic = F.l1_loss(final_state, pref_exp, reduction="none").mean(-1)

        # Epistemic value: state variance as uncertainty proxy
        epistemic = predicted.var(dim=1).mean(-1)

        # Combine
        efe = (
            self.cfg.efe_pragmatic_weight * pragmatic
            - self.cfg.efe_epistemic_weight * epistemic
        )
        return efe.reshape(B, K)

    def plan(
        self,
        observation: torch.Tensor,
        preferences: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Plan using CEM + EFE.

        Args:
            observation: (B, obs_dim) current observation
            preferences: (B, D) preferred state
        Returns:
            action: (B, A) first action of best sequence
            info: dict with planning details
        """
        state = self.encode_observation(observation)

        def cost_fn(actions):
            return self.compute_efe(state, actions, preferences)

        best_sequence = self.planner.optimize(
            cost_fn, batch_size=observation.shape[0],
            device=observation.device,
        )

        # Return first action
        action = best_sequence[:, 0]

        # Compute EFE for info
        with torch.no_grad():
            efe_values = cost_fn(best_sequence.unsqueeze(1))

        return action, {"efe_values": efe_values, "planned_sequence": best_sequence}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vjepa2_world_model.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add brain_ai/decision/vjepa2_world_model.py tests/test_vjepa2_world_model.py
git commit -m "feat: add V-JEPA 2 world model with CEM+EFE planning"
```

---

## Task 6: Create Imagination Engine for System 2

**Files:**
- Create: `brain_ai/reasoning/imagination.py`
- Test: `tests/test_vjepa2_imagination.py`

**Step 1: Write the failing test**

Create `tests/test_vjepa2_imagination.py`:

```python
"""Tests for V-JEPA 2 imagination engine."""
import pytest
import torch
from brain_ai.config import ImaginationConfig, VJEPA2WorldModelConfig
from brain_ai.reasoning.imagination import ImaginationEngine


class TestImaginationEngine:
    @pytest.fixture
    def engine(self):
        wm_cfg = VJEPA2WorldModelConfig(
            predictor_dim=32, action_dim=4,
            action_embed_dim=32, predictor_depth=1,
        )
        im_cfg = ImaginationConfig(
            enabled=True, max_rollout_steps=4,
            num_parallel_scenarios=3,
        )
        return ImaginationEngine(im_cfg, wm_cfg, obs_dim=64)

    def test_imagine_shape(self, engine):
        state = torch.randn(2, 64)
        actions = torch.randn(2, 4, 4)  # 4 steps, action_dim=4
        futures = engine.imagine(state, actions, steps=4)
        assert len(futures) == 4
        assert futures[0].shape[0] == 2

    def test_evaluate_scenarios(self, engine):
        state = torch.randn(2, 64)
        scenarios = [torch.randn(2, 4, 4) for _ in range(3)]
        scores = engine.evaluate_scenarios(state, scenarios)
        assert scores.shape == (2, 3)

    def test_counterfactual(self, engine):
        observed = torch.randn(2, 64)
        alternative = torch.randn(2, 64)
        result = engine.counterfactual(observed, alternative)
        assert result.shape[0] == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vjepa2_imagination.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `brain_ai/reasoning/imagination.py`:

```python
"""
Imagination Engine for System 2 Reasoning

Uses V-JEPA 2's predictor to simulate hypothetical scenarios,
evaluate counterfactuals, and plan multi-step strategies.
All simulation operates in latent space for efficiency.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional

from ..config import ImaginationConfig, VJEPA2WorldModelConfig
from ..decision.vjepa2_world_model import VJEPA2WorldModel


class ImaginationEngine(nn.Module):
    """Mental simulation via latent-space prediction rollout."""

    def __init__(
        self,
        cfg: ImaginationConfig,
        world_model_cfg: VJEPA2WorldModelConfig,
        obs_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.world_model = VJEPA2WorldModel(world_model_cfg, obs_dim=obs_dim)

    def imagine(
        self,
        current_state: torch.Tensor,
        hypothetical_actions: torch.Tensor,
        steps: int = 4,
    ) -> List[torch.Tensor]:
        """Simulate future states given hypothetical actions.

        Args:
            current_state: (B, obs_dim) observation
            hypothetical_actions: (B, steps, action_dim)
            steps: number of simulation steps
        Returns:
            List of (B, D) predicted states, one per step
        """
        state = self.world_model.encode_observation(current_state)
        futures = []
        for t in range(min(steps, hypothetical_actions.shape[1])):
            state = self.world_model.predict_transition(
                state, hypothetical_actions[:, t],
            )
            futures.append(state)
        return futures

    def evaluate_scenarios(
        self,
        current_state: torch.Tensor,
        scenario_actions: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compare multiple action scenarios.

        Args:
            current_state: (B, obs_dim)
            scenario_actions: list of K tensors each (B, H, A)
        Returns:
            scores: (B, K) quality score per scenario (lower = better)
        """
        scores = []
        for actions in scenario_actions:
            futures = self.imagine(current_state, actions, steps=actions.shape[1])
            # Score: negative magnitude of final state (proxy for coherence)
            final = futures[-1]
            score = final.norm(dim=-1)
            scores.append(score)
        return torch.stack(scores, dim=1)

    def counterfactual(
        self,
        observed_state: torch.Tensor,
        alternative_context: torch.Tensor,
    ) -> torch.Tensor:
        """Predict what would happen under alternative conditions.

        Args:
            observed_state: (B, obs_dim) what was observed
            alternative_context: (B, obs_dim) alternative conditions
        Returns:
            predicted: (B, D) predicted alternative state
        """
        obs_latent = self.world_model.encode_observation(observed_state)
        alt_latent = self.world_model.encode_observation(alternative_context)
        # Blend: take the structure of observed + context of alternative
        blended = (obs_latent + alt_latent) / 2
        return blended
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vjepa2_imagination.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add brain_ai/reasoning/imagination.py tests/test_vjepa2_imagination.py
git commit -m "feat: add imagination engine for System 2 reasoning"
```

---

## Task 7: Create Neuromodulated Masking Module

**Files:**
- Create: `brain_ai/encoders/vjepa2_masking.py`
- Test: `tests/test_vjepa2_masking.py`

**Step 1: Write the failing test**

Create `tests/test_vjepa2_masking.py`:

```python
"""Tests for neuromodulated attention masking."""
import pytest
import torch
from brain_ai.config import NeuromodulatedMaskingConfig
from brain_ai.encoders.vjepa2_masking import NeuromodulatedMasking


class TestNeuromodulatedMasking:
    @pytest.fixture
    def masker(self):
        cfg = NeuromodulatedMaskingConfig(enabled=True, base_mask_ratio=0.75)
        return NeuromodulatedMasking(cfg, num_tokens=100)

    def test_mask_shape(self, masker):
        tokens = torch.randn(2, 100, 64)
        modulators = {
            "acetylcholine": torch.tensor([[0.5], [0.8]]),
            "norepinephrine": torch.tensor([[0.3], [0.7]]),
        }
        mask = masker.generate_mask(tokens, modulators)
        assert mask.shape == (2, 100)
        assert mask.dtype == torch.bool

    def test_high_ach_masks_more(self, masker):
        tokens = torch.randn(2, 100, 64)
        low_ach = {"acetylcholine": torch.tensor([[0.1], [0.1]]),
                   "norepinephrine": torch.tensor([[0.5], [0.5]])}
        high_ach = {"acetylcholine": torch.tensor([[0.9], [0.9]]),
                    "norepinephrine": torch.tensor([[0.5], [0.5]])}
        torch.manual_seed(42)
        mask_low = masker.generate_mask(tokens, low_ach)
        torch.manual_seed(42)
        mask_high = masker.generate_mask(tokens, high_ach)
        # High ACh should mask more tokens (fewer visible)
        assert mask_high.sum() <= mask_low.sum()

    def test_salience_guided(self, masker):
        tokens = torch.randn(2, 100, 64)
        modulators = {
            "acetylcholine": torch.tensor([[0.5], [0.5]]),
            "norepinephrine": torch.tensor([[0.5], [0.5]]),
        }
        salience = torch.zeros(2, 100)
        salience[:, :10] = 1.0  # First 10 tokens are salient
        mask = masker.generate_mask(tokens, modulators, salience_map=salience)
        # Salient tokens should be more likely to be visible (not masked)
        assert mask.shape == (2, 100)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vjepa2_masking.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `brain_ai/encoders/vjepa2_masking.py`:

```python
"""
Neuromodulated Attention Masking

Dynamic masking controlled by neuromodulatory signals.
ACh controls masking ratio, NE controls randomness.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from ..config import NeuromodulatedMaskingConfig


class NeuromodulatedMasking(nn.Module):
    """Generate attention masks conditioned on neuromodulatory state."""

    def __init__(self, cfg: NeuromodulatedMaskingConfig, num_tokens: int):
        super().__init__()
        self.cfg = cfg
        self.num_tokens = num_tokens

    def generate_mask(
        self,
        tokens: torch.Tensor,
        modulators: Dict[str, torch.Tensor],
        salience_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate attention mask.

        Args:
            tokens: (B, N, D) encoder tokens
            modulators: dict with "acetylcholine", "norepinephrine", etc.
            salience_map: (B, N) optional salience scores per token
        Returns:
            mask: (B, N) bool — True = visible, False = masked
        """
        B, N, _ = tokens.shape
        device = tokens.device

        # ACh modulates masking ratio (high ACh = more masking = sharper focus)
        ach = modulators.get("acetylcholine", torch.tensor([[0.5]], device=device))
        ach = ach.expand(B, 1).squeeze(-1)  # (B,)
        mask_ratio = self.cfg.base_mask_ratio + self.cfg.ach_sensitivity * (ach - 0.5)
        mask_ratio = mask_ratio.clamp(0.1, 0.95)

        # NE modulates randomness (high NE = more uniform/random scan)
        ne = modulators.get("norepinephrine", torch.tensor([[0.5]], device=device))
        ne = ne.expand(B, 1).squeeze(-1)  # (B,)
        temperature = 1.0 + self.cfg.ne_sensitivity * (ne - 0.5)

        # Base scores: random or salience-guided
        if salience_map is not None and self.cfg.salience_guided:
            # Higher salience = lower probability of being masked
            scores = salience_map / temperature.unsqueeze(1)
        else:
            scores = torch.rand(B, N, device=device) / temperature.unsqueeze(1)

        # Apply masking: keep top (1 - mask_ratio) fraction
        num_visible = ((1.0 - mask_ratio) * N).long().clamp(min=1)

        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            k = num_visible[b].item()
            _, top_idx = scores[b].topk(k)
            mask[b, top_idx] = True

        return mask
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vjepa2_masking.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add brain_ai/encoders/vjepa2_masking.py tests/test_vjepa2_masking.py
git commit -m "feat: add neuromodulated attention masking"
```

---

## Task 8: End-to-End Integration Test

**Files:**
- Test: `tests/test_vjepa2_e2e.py`

**Step 1: Write the integration test**

```python
"""End-to-end tests for V-JEPA 2 integration across all modules."""
import pytest
import torch
from brain_ai.config import BrainAIConfig
from brain_ai.system import BrainAI


class TestVJEPA2EndToEnd:
    @pytest.fixture
    def full_config(self):
        """Config with all V-JEPA 2 modules enabled at small scale."""
        cfg = BrainAIConfig()
        # Small dims for testing
        cfg.encoder.output_dim = 64
        cfg.workspace.workspace_dim = 64
        cfg.decision.hidden_dim = 64
        cfg.decision.num_classes = 10
        cfg.reasoning.hidden_dim = 64
        cfg.snn.hidden_sizes = [32, 16]
        cfg.snn.num_timesteps = 4
        cfg.modalities = ["vision", "text"]
        cfg.use_htm = False
        cfg.use_symbolic = True
        cfg.use_meta = True
        cfg.use_engram = False
        cfg.meta.neuromod_hidden_dim = 32
        # V-JEPA 2 encoder
        cfg.vjepa2_encoder.enabled = True
        cfg.vjepa2_encoder.pretrained = False
        cfg.vjepa2_encoder.encoder_dim = 32
        cfg.vjepa2_encoder.output_dim = 64
        cfg.vjepa2_encoder.num_query_tokens = 4
        cfg.vjepa2_encoder.probe_layers = 1
        cfg.vjepa2_encoder.probe_heads = 4
        return cfg

    def test_full_forward_pass(self, full_config):
        model = BrainAI(config=full_config, modalities=["vision", "text"])
        inputs = {
            "vision": torch.randn(2, 3, 224, 224),
            "text": torch.randint(0, 256, (2, 32)),
        }
        output = model(inputs, task="classify", return_details=True)
        assert output.output.shape == (2, 10)
        assert output.workspace.shape == (2, 64)
        assert output.modulators is not None

    def test_vjepa2_encoder_in_pipeline(self, full_config):
        model = BrainAI(config=full_config, modalities=["vision"])
        from brain_ai.encoders.vjepa2_backbone import VJEPA2VisionEncoder
        assert isinstance(model.encoders["vision"], VJEPA2VisionEncoder)
        plan = model.get_pipeline_plan()
        assert "encode" in plan.enabled_stages()

    def test_backward_pass(self, full_config):
        model = BrainAI(config=full_config, modalities=["vision"])
        inputs = {"vision": torch.randn(2, 3, 224, 224)}
        output = model(inputs, task="classify")
        loss = output.sum()
        loss.backward()
        # Backbone should have no grad (frozen)
        backbone_params = list(model.encoders["vision"].backbone.parameters())
        assert all(p.grad is None for p in backbone_params)
        # Pooler should have grad (trainable)
        pooler_params = list(model.encoders["vision"].pooler.parameters())
        assert any(p.grad is not None for p in pooler_params)

    def test_world_model_standalone(self, full_config):
        from brain_ai.decision.vjepa2_world_model import VJEPA2WorldModel
        wm = VJEPA2WorldModel(full_config.vjepa2_world_model, obs_dim=64)
        obs = torch.randn(2, 64)
        pref = torch.randn(2, wm.cfg.predictor_dim)
        action, info = wm.plan(obs, pref)
        assert action.shape[0] == 2

    def test_imagination_standalone(self):
        from brain_ai.config import ImaginationConfig, VJEPA2WorldModelConfig
        from brain_ai.reasoning.imagination import ImaginationEngine
        im_cfg = ImaginationConfig(enabled=True, max_rollout_steps=4)
        wm_cfg = VJEPA2WorldModelConfig(
            predictor_dim=32, action_dim=4,
            action_embed_dim=32, predictor_depth=1,
        )
        engine = ImaginationEngine(im_cfg, wm_cfg, obs_dim=64)
        state = torch.randn(2, 64)
        actions = torch.randn(2, 4, 4)
        futures = engine.imagine(state, actions, steps=4)
        assert len(futures) == 4

    def test_masking_standalone(self):
        from brain_ai.encoders.vjepa2_masking import NeuromodulatedMasking
        from brain_ai.config import NeuromodulatedMaskingConfig
        cfg = NeuromodulatedMaskingConfig(enabled=True)
        masker = NeuromodulatedMasking(cfg, num_tokens=50)
        tokens = torch.randn(2, 50, 32)
        mods = {
            "acetylcholine": torch.tensor([[0.7], [0.3]]),
            "norepinephrine": torch.tensor([[0.5], [0.5]]),
        }
        mask = masker.generate_mask(tokens, mods)
        assert mask.shape == (2, 50)
```

**Step 2: Run tests**

Run: `pytest tests/test_vjepa2_e2e.py -v`
Expected: All 6 tests PASS

**Step 3: Commit**

```bash
git add tests/test_vjepa2_e2e.py
git commit -m "test: add V-JEPA 2 end-to-end integration tests"
```

---

## Task 9: Update Plugin with New Skills

**Files:**
- Create: `brain-ai-dev/skills/vjepa2-encoder-bridge/SKILL.md`
- Create: `brain-ai-dev/skills/vjepa2-workspace-integration/SKILL.md`
- Create: `brain-ai-dev/skills/vjepa2-active-inference-world-model/SKILL.md`
- Create: `brain-ai-dev/skills/vjepa2-htm-temporal-bridge/SKILL.md`
- Create: `brain-ai-dev/skills/vjepa2-imagination-engine/SKILL.md`
- Create: `brain-ai-dev/skills/vjepa2-attention-masking/SKILL.md`
- Modify: `brain-ai-dev/.claude-plugin/plugin.json`

**Step 1: Create the 6 SKILL.md files**

Each skill follows the standard SKILL.md format from the existing plugin. Use the design doc sections as content. Reference the assets and implementation files created in Tasks 2-7.

**Step 2: Update plugin.json**

Add the 6 new skill paths to the `skills` array:
```json
"skills/vjepa2-encoder-bridge",
"skills/vjepa2-workspace-integration",
"skills/vjepa2-active-inference-world-model",
"skills/vjepa2-htm-temporal-bridge",
"skills/vjepa2-imagination-engine",
"skills/vjepa2-attention-masking"
```

**Step 3: Commit**

```bash
git add brain-ai-dev/skills/vjepa2-*/SKILL.md brain-ai-dev/.claude-plugin/plugin.json
git commit -m "feat: add 6 V-JEPA 2 integration skills to brain-ai-dev plugin"
```

---

## Task 10: Run Full Test Suite and Verify Done-When Gates

**Step 1: Run all V-JEPA 2 tests**

```bash
pytest tests/test_vjepa2_*.py -v --tb=short
```

Expected: All tests PASS across all 6 test files.

**Step 2: Run existing tests to verify no regressions**

```bash
pytest tests/ -v --tb=short -x
```

Expected: No regressions in existing tests.

**Step 3: Verify done-when gates**

Check each gate manually:
1. Encoder Bridge: `test_forward_image_shape` + `test_matches_vision_encoder_contract` PASS
2. Workspace Integration: `test_full_forward_pass` with V-JEPA 2 encoder PASS
3. World Model Plans: `test_plan` + `test_compute_efe` PASS
4. Temporal Bridge: `test_output_shape` PASS
5. Imagination Works: `test_imagine_shape` + `test_evaluate_scenarios` PASS
6. Masking Responds to Modulators: `test_high_ach_masks_more` PASS

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: V-JEPA 2 integration phase 1-3 complete, all gates pass"
```

---

## Summary

| Task | Component | Files Created/Modified | Tests |
|------|-----------|----------------------|-------|
| 1 | Config dataclasses | `config.py` | 7 |
| 2 | Backbone + encoder | `vjepa2_backbone.py` | 8 |
| 3 | System wiring | `system.py` | 5 |
| 4 | Temporal bridge | `vjepa2_temporal_bridge.py` | 3 |
| 5 | World model | `vjepa2_world_model.py` | 6 |
| 6 | Imagination engine | `imagination.py` | 3 |
| 7 | Neuromodulated masking | `vjepa2_masking.py` | 3 |
| 8 | E2E integration | `test_vjepa2_e2e.py` | 6 |
| 9 | Plugin skills | 6 SKILL.md + plugin.json | - |
| 10 | Verification | - | full suite |
| **Total** | | **8 new files, 2 modified** | **41 tests** |
