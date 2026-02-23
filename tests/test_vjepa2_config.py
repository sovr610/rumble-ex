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
