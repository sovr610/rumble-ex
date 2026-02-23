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
            pretrained=False,
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
        assert tokens.dim() == 3
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
        encoder = VJEPA2VisionEncoder(small_cfg)
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x, temporal_input=False)
        assert out.shape == (2, small_cfg.output_dim)

    def test_get_raw_tokens(self, small_cfg):
        encoder = VJEPA2VisionEncoder(small_cfg)
        x = torch.randn(2, 3, 224, 224)
        tokens, positions = encoder.get_raw_tokens(x)
        assert tokens.dim() == 3
        assert tokens.shape[0] == 2
        assert tokens.shape[2] == small_cfg.encoder_dim
