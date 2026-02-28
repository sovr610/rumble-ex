"""
System Integration Tests for BrainAI

End-to-end tests verifying the full BrainAI system pipeline:
  - Factory functions (create_brain_ai, create_vision_classifier, etc.)
  - Forward pass through all subsystems
  - Convenience methods (classify, generate, act)
  - Feature flags and config presets
  - Error handling
"""

import pytest
import torch
import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from brain_ai.system import (
    BrainAI,
    create_brain_ai,
    create_vision_classifier,
    create_multimodal_system,
    create_control_agent,
    SystemOutput,
)
from brain_ai.config import BrainAIConfig


# ---------------------------------------------------------------------------
# Shared small-dimension defaults for fast tests
# ---------------------------------------------------------------------------
WORKSPACE_DIM = 64
ENCODER_OUTPUT_DIM = 64
SNN_TIMESTEPS = 4
TEXT_VOCAB = 256
TEXT_EMBED = 64
TEXT_LAYERS = 1
TEXT_HEADS = 2
HTM_COLUMNS = 64
HTM_CELLS = 4
BATCH = 2

# Symbolic reasoning requires reasoning_hidden == workspace_dim internally.
# The System2Config.reasoning_hidden defaults to 512 and is not settable
# through create_brain_ai, so tests that enable the symbolic reasoner use
# workspace_dim = 512 to avoid a dimension mismatch in the GRU cell.
SYMBOLIC_WORKSPACE_DIM = 512


def _small_config(**overrides) -> BrainAIConfig:
    """Return a BrainAIConfig with tiny dimensions for fast CPU tests."""
    cfg = BrainAIConfig()
    cfg.encoder.output_dim = ENCODER_OUTPUT_DIM
    cfg.encoder.vision_channels = [8, 16]
    cfg.encoder.text_vocab_size = TEXT_VOCAB
    cfg.encoder.text_embed_dim = TEXT_EMBED
    cfg.encoder.text_num_layers = TEXT_LAYERS
    cfg.encoder.text_num_heads = TEXT_HEADS
    cfg.encoder.audio_n_mels = 16
    cfg.encoder.sensor_input_dim = 16
    cfg.encoder.sensor_hidden_dim = 32
    cfg.snn.num_timesteps = SNN_TIMESTEPS
    cfg.snn.hidden_sizes = [32, 16]
    cfg.snn.beta = 0.9
    cfg.htm.column_count = HTM_COLUMNS
    cfg.htm.cells_per_column = HTM_CELLS
    cfg.workspace.workspace_dim = WORKSPACE_DIM
    cfg.workspace.num_heads = 2
    cfg.workspace.memory_hidden_dim = WORKSPACE_DIM
    cfg.decision.num_classes = 10
    cfg.decision.control_dim = 4
    cfg.decision.state_dim = 16
    cfg.decision.planning_horizon = 2
    cfg.reasoning.hidden_dim = WORKSPACE_DIM
    cfg.reasoning.num_reasoning_steps = 2
    cfg.meta.neuromod_hidden_dim = 32
    cfg.engram.vocab_size = TEXT_VOCAB
    cfg.engram.embedding_dim = ENCODER_OUTPUT_DIM
    cfg.engram.ngram_orders = (2, 3)
    cfg.engram.num_heads = 2
    cfg.engram.table_size = 1009  # small prime
    for key, val in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg


def _small_brain(
    modalities=None,
    output_type="classify",
    use_htm=False,
    use_symbolic=False,
    use_meta=False,
    use_engram=False,
):
    """Helper: build a small BrainAI via the factory, choosing the right
    workspace dimension depending on whether symbolic reasoning is enabled."""
    ws = SYMBOLIC_WORKSPACE_DIM if use_symbolic else WORKSPACE_DIM
    brain = create_brain_ai(
        modalities=modalities or ["vision"],
        output_type=output_type,
        workspace_dim=ws,
        use_htm=use_htm,
        use_symbolic=use_symbolic,
        use_meta=use_meta,
        use_engram=use_engram,
        device="cpu",
    )
    brain.train(False)
    return brain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_vision_brain():
    """Vision-only classifier with all subsystems, small dims."""
    return _small_brain(
        modalities=["vision"],
        use_htm=True,
        use_symbolic=True,
        use_meta=True,
    )


@pytest.fixture
def vision_images():
    """Random batch of tiny images (B=2, C=3, H=32, W=32)."""
    return torch.randn(BATCH, 3, 32, 32)


@pytest.fixture
def small_multimodal_brain():
    """Vision + text brain, small dims."""
    return _small_brain(
        modalities=["vision", "text"],
        use_htm=True,
        use_symbolic=True,
        use_meta=True,
    )


@pytest.fixture
def small_control_brain():
    """Sensor-based control brain, small dims."""
    return _small_brain(
        modalities=["sensors"],
        output_type="control",
        use_htm=True,
        use_symbolic=True,
        use_meta=True,
    )


# ===================================================================
# 1. test_create_brain_ai_default
# ===================================================================
class TestCreateBrainAIDefault:
    def test_create_brain_ai_default(self):
        """create_brain_ai with minimal overrides yields a BrainAI instance."""
        brain = create_brain_ai(
            modalities=["vision"],
            workspace_dim=WORKSPACE_DIM,
            use_htm=False,
            use_symbolic=False,
            use_meta=False,
            device="cpu",
        )
        assert isinstance(brain, BrainAI)


# ===================================================================
# 2. test_vision_classifier_forward
# ===================================================================
class TestVisionClassifierForward:
    def test_output_shape(self, small_vision_brain, vision_images):
        """Vision classifier produces [B, num_classes] logits."""
        out = small_vision_brain({"vision": vision_images})
        assert out.shape == (BATCH, 10)

    def test_via_factory(self, vision_images):
        """create_vision_classifier convenience factory."""
        brain = create_vision_classifier(
            num_classes=10,
            workspace_dim=WORKSPACE_DIM,
            use_htm=False,
            use_symbolic=False,
            use_meta=False,
            device="cpu",
        )
        brain.train(False)
        out = brain({"vision": vision_images})
        assert out.shape == (BATCH, 10)


# ===================================================================
# 3. test_multimodal_vision_text
# ===================================================================
class TestMultimodalVisionText:
    def test_multimodal_forward(self, small_multimodal_brain, vision_images):
        """Vision + text produces valid output."""
        text_ids = torch.randint(0, TEXT_VOCAB, (BATCH, 8))
        inputs = {"vision": vision_images, "text": text_ids}
        out = small_multimodal_brain(inputs)
        assert out.shape[0] == BATCH
        assert out.dim() == 2

    def test_via_factory(self, vision_images):
        """create_multimodal_system factory with small dims."""
        brain = create_multimodal_system(
            modalities=["vision", "text"],
            workspace_dim=WORKSPACE_DIM,
            use_htm=False,
            use_symbolic=False,
            use_meta=False,
            device="cpu",
        )
        brain.train(False)
        text_ids = torch.randint(0, TEXT_VOCAB, (BATCH, 8))
        out = brain({"vision": vision_images, "text": text_ids})
        assert out.shape[0] == BATCH


# ===================================================================
# 4. test_control_agent
# ===================================================================
class TestControlAgent:
    def test_control_forward(self, small_control_brain):
        """Control brain produces action tensor of correct shape."""
        sensor_input = torch.randn(BATCH, 16)
        out = small_control_brain({"sensors": sensor_input})
        assert out.shape == (BATCH, 4)

    def test_via_factory(self):
        """create_control_agent factory."""
        brain = create_control_agent(
            state_dim=16,
            action_dim=4,
            workspace_dim=WORKSPACE_DIM,
            use_htm=False,
            use_symbolic=False,
            use_meta=False,
            device="cpu",
        )
        brain.train(False)
        out = brain({"sensors": torch.randn(BATCH, 16)})
        assert out.shape == (BATCH, 4)


# ===================================================================
# 5. test_active_inference_task
# ===================================================================
class TestActiveInferenceTask:
    def test_active_inference_runs(self, small_vision_brain, vision_images):
        """task='active_inference' completes without error."""
        out = small_vision_brain(
            {"vision": vision_images}, task="active_inference"
        )
        assert out.shape[0] == BATCH
        assert out.dim() >= 1


# ===================================================================
# 6. test_return_details
# ===================================================================
class TestReturnDetails:
    def test_system_output_fields(self, small_vision_brain, vision_images):
        """return_details=True yields a SystemOutput with expected fields."""
        result = small_vision_brain(
            {"vision": vision_images}, return_details=True
        )
        assert isinstance(result, SystemOutput)
        assert result.output is not None
        assert result.workspace is not None
        assert result.confidence is not None
        # attention may or may not be None depending on workspace implementation
        # but output, workspace, confidence must exist

    def test_output_tensor_shapes(self, small_vision_brain, vision_images):
        """SystemOutput tensors have batch dimension."""
        result = small_vision_brain(
            {"vision": vision_images}, return_details=True
        )
        assert result.output.shape[0] == BATCH
        assert result.workspace.shape[0] == BATCH
        assert result.confidence.shape[0] == BATCH


# ===================================================================
# 7. test_reset_state
# ===================================================================
class TestResetState:
    def test_reset_no_error(self, small_vision_brain, vision_images):
        """reset_state() after a forward pass raises no errors."""
        small_vision_brain({"vision": vision_images})
        small_vision_brain.reset_state()  # should not raise

    def test_reset_before_forward(self, small_vision_brain):
        """reset_state() before any forward pass is safe."""
        small_vision_brain.reset_state()

    def test_double_reset(self, small_vision_brain, vision_images):
        """Two consecutive resets are fine."""
        small_vision_brain({"vision": vision_images})
        small_vision_brain.reset_state()
        small_vision_brain.reset_state()


# ===================================================================
# 8. test_classify_convenience
# ===================================================================
class TestClassifyConvenience:
    def test_classify_returns_tensor(self, small_vision_brain, vision_images):
        """brain.classify(inputs) returns class logits."""
        logits = small_vision_brain.classify({"vision": vision_images})
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (BATCH, 10)


# ===================================================================
# 9. test_generate_convenience
# ===================================================================
class TestGenerateConvenience:
    def test_generate_returns_tensor(self, small_multimodal_brain, vision_images):
        """brain.generate(inputs) returns a tensor."""
        text_ids = torch.randint(0, TEXT_VOCAB, (BATCH, 8))
        inputs = {"vision": vision_images, "text": text_ids}
        result = small_multimodal_brain.generate(inputs, max_length=5)
        assert isinstance(result, torch.Tensor)


# ===================================================================
# 10. test_act_convenience
# ===================================================================
class TestActConvenience:
    def test_act_returns_action(self, small_control_brain):
        """brain.act(inputs) returns an action tensor."""
        sensor_input = torch.randn(BATCH, 16)
        action = small_control_brain.act({"sensors": sensor_input})
        assert isinstance(action, torch.Tensor)
        assert action.shape == (BATCH, 4)

    def test_act_deterministic(self, small_control_brain):
        """Deterministic actions have same shape."""
        sensor_input = torch.randn(BATCH, 16)
        action = small_control_brain.act(
            {"sensors": sensor_input}, deterministic=True
        )
        assert action.shape == (BATCH, 4)


# ===================================================================
# 11. test_config_presets
# ===================================================================
class TestConfigPresets:
    def test_minimal_preset(self):
        """BrainAIConfig.minimal() can construct a system without error."""
        cfg = BrainAIConfig.minimal()
        # Just verify fields are set to small values
        assert cfg.use_htm is False
        assert cfg.use_symbolic is False
        assert cfg.use_meta is False
        assert cfg.use_engram is False

    def test_minimal_preset_builds_model(self):
        """A BrainAI built from the minimal preset runs a forward pass."""
        cfg = BrainAIConfig.minimal()
        cfg.modalities = ["vision"]
        # Override large defaults to keep test fast
        cfg.encoder.vision_channels = [8, 16]
        cfg.snn.num_timesteps = 2
        model = BrainAI(config=cfg, modalities=["vision"], output_type="classify")
        model.train(False)
        out = model({"vision": torch.randn(1, 3, 32, 32)})
        assert out.shape[0] == 1

    @pytest.mark.parametrize(
        "preset_name",
        ["minimal", "for_vision_only"],
    )
    def test_preset_no_error(self, preset_name):
        """Config class-method presets don't raise on construction."""
        cfg = getattr(BrainAIConfig, preset_name)()
        assert isinstance(cfg, BrainAIConfig)


# ===================================================================
# 12. test_feature_flags
# ===================================================================
class TestFeatureFlags:
    @pytest.mark.parametrize(
        "flags",
        [
            dict(use_htm=False, use_symbolic=False, use_meta=False),
            dict(use_htm=True, use_symbolic=False, use_meta=False),
            dict(use_htm=False, use_symbolic=True, use_meta=False),
            dict(use_htm=False, use_symbolic=False, use_meta=True),
            dict(use_htm=True, use_symbolic=True, use_meta=True),
        ],
        ids=[
            "none",
            "htm_only",
            "symbolic_only",
            "meta_only",
            "all",
        ],
    )
    def test_feature_flag_combinations(self, flags, vision_images):
        """System runs with various feature-flag combinations."""
        # Symbolic reasoning internally uses reasoning_hidden=512 (System2Config
        # default), so the workspace dimension must be >= 512 to match.
        ws = SYMBOLIC_WORKSPACE_DIM if flags.get("use_symbolic") else WORKSPACE_DIM
        brain = create_brain_ai(
            modalities=["vision"],
            workspace_dim=ws,
            device="cpu",
            **flags,
        )
        brain.train(False)
        out = brain({"vision": vision_images})
        assert out.shape[0] == BATCH

    def test_disabled_components_are_none(self):
        """When flags are False, the corresponding modules are None."""
        brain = create_brain_ai(
            modalities=["vision"],
            workspace_dim=WORKSPACE_DIM,
            use_htm=False,
            use_symbolic=False,
            use_meta=False,
            device="cpu",
        )
        assert brain.htm is None
        assert brain.reasoner is None
        assert brain.neuromodulation is None


# ===================================================================
# 13. test_engram_encoder
# ===================================================================
class TestEngramEncoder:
    def test_engram_forward(self):
        """System with use_engram=True processes token_ids input."""
        cfg = _small_config(
            use_htm=False,
            use_symbolic=False,
            use_meta=False,
            use_engram=True,
            use_workspace=True,
        )
        cfg.modalities = ["vision"]
        model = BrainAI(config=cfg, modalities=["vision"], output_type="classify")
        model.train(False)

        images = torch.randn(BATCH, 3, 32, 32)
        token_ids = torch.randint(0, TEXT_VOCAB, (BATCH, 8))
        out = model({"vision": images, "token_ids": token_ids})
        assert out.shape[0] == BATCH

    def test_engram_encoder_present(self):
        """When use_engram=True, 'engram' key is in encoders dict."""
        cfg = _small_config(use_engram=True)
        cfg.modalities = ["vision"]
        model = BrainAI(config=cfg, modalities=["vision"])
        assert "engram" in model.encoders


# ===================================================================
# 14. test_device_placement
# ===================================================================
class TestDevicePlacement:
    def test_cpu_placement(self):
        """Model parameters are on CPU when device='cpu'."""
        brain = create_brain_ai(
            modalities=["vision"],
            workspace_dim=WORKSPACE_DIM,
            use_htm=False,
            use_symbolic=False,
            use_meta=False,
            device="cpu",
        )
        for param in brain.parameters():
            assert param.device.type == "cpu"
            break  # checking first param is sufficient

    def test_factory_auto_device(self):
        """device='auto' resolves to cpu on machines without CUDA."""
        brain = create_brain_ai(
            modalities=["vision"],
            workspace_dim=WORKSPACE_DIM,
            use_htm=False,
            use_symbolic=False,
            use_meta=False,
            device="auto",
        )
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        first_param = next(brain.parameters())
        assert first_param.device.type == expected


# ===================================================================
# 15. test_unknown_task_raises
# ===================================================================
class TestUnknownTaskRaises:
    def test_unknown_task_valueerror(self, small_vision_brain, vision_images):
        """Passing an unknown task string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task"):
            small_vision_brain({"vision": vision_images}, task="bogus_task")

    @pytest.mark.parametrize(
        "bad_task",
        ["predict", "infer", "foo_bar"],
    )
    def test_various_bad_tasks(self, small_vision_brain, vision_images, bad_task):
        """Several invalid task strings all raise ValueError."""
        with pytest.raises(ValueError):
            small_vision_brain({"vision": vision_images}, task=bad_task)


# ===================================================================
# Extra: gradient flow sanity check
# ===================================================================
class TestGradientFlow:
    def test_classify_gradient(self):
        """Gradients flow from classification loss back to encoder."""
        brain = create_brain_ai(
            modalities=["vision"],
            workspace_dim=WORKSPACE_DIM,
            use_htm=False,
            use_symbolic=False,
            use_meta=False,
            device="cpu",
        )
        brain.train(True)
        images = torch.randn(BATCH, 3, 32, 32)
        logits = brain({"vision": images})
        loss = logits.sum()
        loss.backward()
        # At least one encoder param should have a gradient
        has_grad = False
        for p in brain.encoders["vision"].parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradient reached the vision encoder"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
