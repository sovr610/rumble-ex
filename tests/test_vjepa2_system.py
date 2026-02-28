"""Tests for V-JEPA 2 integration with BrainAI system."""
import pytest
import torch
from brain_ai.config import BrainAIConfig
from brain_ai.system import BrainAI


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

    def test_pipeline_shows_encode(self, vjepa2_config):
        model = BrainAI(config=vjepa2_config, modalities=["vision"])
        plan = model.get_pipeline_plan()
        assert "encode" in plan.stage_names()
