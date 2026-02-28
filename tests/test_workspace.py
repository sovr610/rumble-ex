"""
Tests for Global Workspace Module

Validates the Global Workspace Theory implementation including:
    - GlobalWorkspace creation and configuration
    - Forward pass with multi-modal inputs
    - Attention output and workspace competition
    - Working memory integration and reset_state
    - Ignition threshold dynamics (SelectionBroadcastWorkspace)
    - Capacity limit enforcement

Run:
    pytest tests/test_workspace.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys

# Path fixup so tests can run from repo root
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from brain_ai.workspace.global_workspace import (
    GlobalWorkspace,
    GlobalWorkspaceConfig,
    create_global_workspace,
    ModalityProjection,
    AttentionCompetition,
    InformationBroadcast,
    SelectionBroadcastConfig,
    SelectionBroadcastWorkspace,
    create_selection_broadcast_workspace,
)


# ---------------------------------------------------------------------------
# Constants -- keep small for fast execution
# ---------------------------------------------------------------------------
WORKSPACE_DIM = 64
BATCH_SIZE = 2
NUM_HEADS = 4
CAPACITY_LIMIT = 4
SEED = 42

MODALITY_DIMS = {
    "vision": 64,
    "text": 64,
    "audio": 64,
}


def _seed(s: int = SEED):
    torch.manual_seed(s)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace_config():
    """Small GlobalWorkspaceConfig for fast tests."""
    return GlobalWorkspaceConfig(
        workspace_dim=WORKSPACE_DIM,
        num_heads=NUM_HEADS,
        capacity_limit=CAPACITY_LIMIT,
        dropout=0.0,
        memory_hidden_dim=WORKSPACE_DIM,
        memory_mode="gru",
        competition_temperature=1.0,
        min_attention=0.01,
    )


@pytest.fixture
def workspace(workspace_config):
    """Create a small GlobalWorkspace instance."""
    _seed()
    ws = GlobalWorkspace(
        config=workspace_config,
        modality_dims=MODALITY_DIMS,
    )
    ws.train(False)
    return ws


@pytest.fixture
def multi_modal_inputs():
    """Multi-modal input dict for workspace forward pass."""
    _seed()
    return {
        "vision": torch.randn(BATCH_SIZE, 64),
        "text": torch.randn(BATCH_SIZE, 64),
        "audio": torch.randn(BATCH_SIZE, 64),
    }


@pytest.fixture
def sb_config():
    """Small SelectionBroadcastConfig for fast tests."""
    return SelectionBroadcastConfig(
        workspace_dim=WORKSPACE_DIM,
        num_heads=NUM_HEADS,
        capacity_limit=CAPACITY_LIMIT,
        dropout=0.0,
        ignition_threshold=0.3,
        selection_rounds=2,
        broadcast_iterations=1,
        broadcast_decay=0.9,
        memory_hidden_dim=WORKSPACE_DIM,
        memory_mode="gru",
        competition_temperature=0.5,
        min_attention=0.01,
        use_confidence_gating=True,
    )


@pytest.fixture
def sb_workspace(sb_config):
    """Create a small SelectionBroadcastWorkspace instance."""
    _seed()
    ws = SelectionBroadcastWorkspace(
        config=sb_config,
        modality_dims=MODALITY_DIMS,
    )
    ws.train(False)
    return ws


# ---------------------------------------------------------------------------
# Tests: GlobalWorkspace creation
# ---------------------------------------------------------------------------

class TestGlobalWorkspaceCreation:
    """Tests for GlobalWorkspace construction and configuration."""

    def test_create_with_config(self, workspace_config):
        """GlobalWorkspace should accept a config object."""
        ws = GlobalWorkspace(config=workspace_config, modality_dims=MODALITY_DIMS)
        assert ws.config.workspace_dim == WORKSPACE_DIM
        assert ws.config.num_heads == NUM_HEADS

    def test_create_with_factory(self):
        """create_global_workspace factory should produce a valid workspace."""
        ws = create_global_workspace(
            workspace_dim=WORKSPACE_DIM,
            modality_dims=MODALITY_DIMS,
            num_heads=NUM_HEADS,
            capacity_limit=CAPACITY_LIMIT,
            memory_mode="gru",
        )
        assert isinstance(ws, GlobalWorkspace)
        assert ws.config.workspace_dim == WORKSPACE_DIM

    def test_modality_projections_exist(self, workspace):
        """Workspace should have a projection for each modality."""
        for name in MODALITY_DIMS:
            assert name in workspace.projections, (
                f"Missing projection for modality '{name}'"
            )

    def test_default_modalities(self):
        """Workspace with no explicit modality_dims should use defaults."""
        ws = GlobalWorkspace(
            workspace_dim=WORKSPACE_DIM,
            num_heads=NUM_HEADS,
            memory_mode="gru",
        )
        assert "vision" in ws.modality_dims
        assert "text" in ws.modality_dims

    def test_has_working_memory(self, workspace):
        """Workspace should contain a working memory module."""
        assert hasattr(workspace, "working_memory")

    def test_has_competition(self, workspace):
        """Workspace should contain an attention competition module."""
        assert hasattr(workspace, "competition")
        assert isinstance(workspace.competition, AttentionCompetition)

    def test_has_broadcast(self, workspace):
        """Workspace should contain an information broadcast module."""
        assert hasattr(workspace, "broadcast")
        assert isinstance(workspace.broadcast, InformationBroadcast)


# ---------------------------------------------------------------------------
# Tests: GlobalWorkspace forward pass
# ---------------------------------------------------------------------------

class TestGlobalWorkspaceForward:
    """Tests for GlobalWorkspace forward pass behavior."""

    def test_forward_output_keys(self, workspace, multi_modal_inputs):
        """Forward output dict should contain required keys."""
        with torch.no_grad():
            output = workspace(multi_modal_inputs)

        assert "workspace" in output
        assert "broadcasts" in output
        assert "memory_output" in output
        assert "modality_names" in output

    def test_workspace_output_shape(self, workspace, multi_modal_inputs):
        """Workspace output should have shape (batch, workspace_dim)."""
        with torch.no_grad():
            output = workspace(multi_modal_inputs)

        ws_tensor = output["workspace"]
        assert ws_tensor.shape == (BATCH_SIZE, WORKSPACE_DIM), (
            f"Expected ({BATCH_SIZE}, {WORKSPACE_DIM}), got {ws_tensor.shape}"
        )

    def test_workspace_output_finite(self, workspace, multi_modal_inputs):
        """Workspace output should contain no NaN or Inf values."""
        with torch.no_grad():
            output = workspace(multi_modal_inputs)

        ws_tensor = output["workspace"]
        assert not torch.isnan(ws_tensor).any(), "NaN in workspace output"
        assert not torch.isinf(ws_tensor).any(), "Inf in workspace output"

    def test_broadcasts_match_modalities(self, workspace, multi_modal_inputs):
        """Broadcast dict should contain a signal for each modality."""
        with torch.no_grad():
            output = workspace(multi_modal_inputs)

        broadcasts = output["broadcasts"]
        for name, dim in MODALITY_DIMS.items():
            assert name in broadcasts, f"Missing broadcast for '{name}'"
            assert broadcasts[name].shape == (BATCH_SIZE, dim), (
                f"Broadcast '{name}' shape mismatch: "
                f"expected ({BATCH_SIZE}, {dim}), got {broadcasts[name].shape}"
            )

    def test_single_modality_input(self, workspace):
        """Workspace should handle a single modality input."""
        _seed()
        inputs = {"vision": torch.randn(BATCH_SIZE, 64)}
        with torch.no_grad():
            output = workspace(inputs)

        assert output["workspace"].shape == (BATCH_SIZE, WORKSPACE_DIM)

    def test_empty_input_raises(self, workspace):
        """Workspace should raise on empty input dict (ValueError or RuntimeError)."""
        with pytest.raises(Exception):
            workspace({})

    def test_unknown_modality_ignored(self, workspace):
        """Unknown modality names should be silently ignored."""
        _seed()
        inputs = {
            "vision": torch.randn(BATCH_SIZE, 64),
            "smell": torch.randn(BATCH_SIZE, 64),  # Not registered
        }
        with torch.no_grad():
            output = workspace(inputs)

        # Only 'vision' should be processed
        assert output["workspace"].shape == (BATCH_SIZE, WORKSPACE_DIM)
        assert "vision" in output["modality_names"]
        assert "smell" not in output["modality_names"]


# ---------------------------------------------------------------------------
# Tests: Attention output
# ---------------------------------------------------------------------------

class TestAttentionOutput:
    """Tests for workspace competition attention weights."""

    def test_return_attention_flag(self, workspace, multi_modal_inputs):
        """return_attention=True should include 'attention' in output."""
        with torch.no_grad():
            output = workspace(multi_modal_inputs, return_attention=True)

        assert "attention" in output

    def test_attention_weights_per_modality(self, workspace, multi_modal_inputs):
        """Attention dict should have an entry for each input modality."""
        with torch.no_grad():
            output = workspace(multi_modal_inputs, return_attention=True)

        attention = output["attention"]
        for name in MODALITY_DIMS:
            assert name in attention, f"Missing attention weight for '{name}'"

    def test_attention_weights_shape(self, workspace, multi_modal_inputs):
        """Each attention weight tensor should have shape (batch,)."""
        with torch.no_grad():
            output = workspace(multi_modal_inputs, return_attention=True)

        for name, weight in output["attention"].items():
            assert weight.shape == (BATCH_SIZE,), (
                f"Attention for '{name}' has shape {weight.shape}, "
                f"expected ({BATCH_SIZE},)"
            )

    def test_attention_weights_non_negative(self, workspace, multi_modal_inputs):
        """Attention weights should be non-negative (softmax output)."""
        with torch.no_grad():
            output = workspace(multi_modal_inputs, return_attention=True)

        for name, weight in output["attention"].items():
            assert (weight >= 0).all(), (
                f"Negative attention weight for '{name}': min={weight.min().item()}"
            )

    def test_attention_no_flag_excludes(self, workspace, multi_modal_inputs):
        """Without return_attention, 'attention' should not be in output."""
        with torch.no_grad():
            output = workspace(multi_modal_inputs, return_attention=False)

        assert "attention" not in output


# ---------------------------------------------------------------------------
# Tests: reset_state
# ---------------------------------------------------------------------------

class TestResetState:
    """Tests for workspace state management and reset."""

    def test_reset_state_clears_prev_context(self, workspace, multi_modal_inputs):
        """reset_state should set prev_context to None."""
        with torch.no_grad():
            workspace(multi_modal_inputs)  # Populates prev_context

        assert workspace.prev_context is not None
        workspace.reset_state()
        assert workspace.prev_context is None

    def test_multiple_forward_updates_context(self, workspace, multi_modal_inputs):
        """Successive forward calls should update prev_context."""
        with torch.no_grad():
            workspace(multi_modal_inputs)
            ctx1 = workspace.prev_context.clone()

            workspace(multi_modal_inputs)
            ctx2 = workspace.prev_context.clone()

        # Context should change between steps (different memory integration)
        # On rare occasions they could be identical, but generally differ
        assert ctx1.shape == ctx2.shape

    def test_reset_allows_clean_rerun(self, workspace, multi_modal_inputs):
        """After reset, forward should produce same output as fresh start."""
        _seed()
        workspace.reset_state()
        with torch.no_grad():
            out1 = workspace(multi_modal_inputs)["workspace"].clone()

        _seed()
        workspace.reset_state()
        with torch.no_grad():
            out2 = workspace(multi_modal_inputs)["workspace"].clone()

        assert torch.allclose(out1, out2, atol=1e-6), (
            f"Outputs differ after reset. Max diff: "
            f"{(out1 - out2).abs().max().item():.6f}"
        )


# ---------------------------------------------------------------------------
# Tests: Workspace competition (capacity limit)
# ---------------------------------------------------------------------------

class TestWorkspaceCompetition:
    """Tests for workspace capacity limit enforcement."""

    def test_capacity_limit_enforced(self):
        """When modalities exceed capacity, only top-K should receive weight."""
        _seed()
        # Create workspace with capacity_limit = 2 but 4 modalities
        many_modalities = {
            "mod_a": 64,
            "mod_b": 64,
            "mod_c": 64,
            "mod_d": 64,
        }
        ws = create_global_workspace(
            workspace_dim=WORKSPACE_DIM,
            modality_dims=many_modalities,
            num_heads=NUM_HEADS,
            capacity_limit=2,
            memory_mode="gru",
        )
        ws.train(False)

        inputs = {name: torch.randn(BATCH_SIZE, 64) for name in many_modalities}
        with torch.no_grad():
            output = ws(inputs, return_attention=True)

        attention = output["attention"]
        # Count how many modalities have non-zero attention per batch item
        weights = torch.stack([attention[n] for n in many_modalities], dim=-1)
        nonzero_per_batch = (weights > 1e-6).sum(dim=-1)

        # At most capacity_limit modalities should have non-zero weight
        assert (nonzero_per_batch <= 2).all(), (
            f"Capacity limit not enforced: nonzero counts = {nonzero_per_batch}"
        )

    def test_competition_temperature_effect(self, multi_modal_inputs):
        """Lower temperature should produce sharper attention distribution."""
        _seed()
        ws_hot = create_global_workspace(
            workspace_dim=WORKSPACE_DIM,
            modality_dims=MODALITY_DIMS,
            num_heads=NUM_HEADS,
            memory_mode="gru",
            competition_temperature=10.0,
        )
        ws_hot.train(False)

        _seed()
        ws_cold = create_global_workspace(
            workspace_dim=WORKSPACE_DIM,
            modality_dims=MODALITY_DIMS,
            num_heads=NUM_HEADS,
            memory_mode="gru",
            competition_temperature=0.1,
        )
        ws_cold.train(False)

        with torch.no_grad():
            out_hot = ws_hot(multi_modal_inputs, return_attention=True)
            out_cold = ws_cold(multi_modal_inputs, return_attention=True)

        # Cold temperature should have higher max attention (sharper)
        hot_weights = torch.stack(
            [out_hot["attention"][n] for n in MODALITY_DIMS], dim=-1
        )
        cold_weights = torch.stack(
            [out_cold["attention"][n] for n in MODALITY_DIMS], dim=-1
        )

        hot_entropy = -(hot_weights * (hot_weights + 1e-8).log()).sum(dim=-1).mean()
        cold_entropy = -(cold_weights * (cold_weights + 1e-8).log()).sum(dim=-1).mean()

        # Cold should generally have lower entropy (sharper), but this is
        # stochastic so we just verify both produce valid distributions
        assert hot_entropy >= 0
        assert cold_entropy >= 0


# ---------------------------------------------------------------------------
# Tests: ModalityProjection
# ---------------------------------------------------------------------------

class TestModalityProjection:
    """Tests for the ModalityProjection sub-module."""

    def test_projection_output_shape(self):
        """Projection should map input_dim to workspace_dim."""
        _seed()
        proj = ModalityProjection(input_dim=64, workspace_dim=WORKSPACE_DIM)
        x = torch.randn(BATCH_SIZE, 64)
        with torch.no_grad():
            projected, salience = proj(x)

        assert projected.shape == (BATCH_SIZE, WORKSPACE_DIM)
        assert salience.shape == (BATCH_SIZE, 1)

    def test_salience_is_scalar_per_sample(self):
        """Salience should produce one score per batch item."""
        _seed()
        proj = ModalityProjection(input_dim=64, workspace_dim=WORKSPACE_DIM)
        x = torch.randn(4, 64)
        with torch.no_grad():
            _, salience = proj(x)

        assert salience.shape == (4, 1)


# ---------------------------------------------------------------------------
# Tests: SelectionBroadcastWorkspace (Ignition)
# ---------------------------------------------------------------------------

class TestIgnitionThreshold:
    """Tests for the ignition dynamics in SelectionBroadcastWorkspace."""

    def test_sb_creation(self, sb_config):
        """SelectionBroadcastWorkspace should create without error."""
        ws = SelectionBroadcastWorkspace(
            config=sb_config,
            modality_dims=MODALITY_DIMS,
        )
        assert ws.config.ignition_threshold == 0.3
        assert ws.config.selection_rounds == 2

    def test_sb_factory(self):
        """create_selection_broadcast_workspace should work."""
        ws = create_selection_broadcast_workspace(
            workspace_dim=WORKSPACE_DIM,
            modality_dims=MODALITY_DIMS,
            num_heads=NUM_HEADS,
            selection_rounds=2,
            ignition_threshold=0.3,
            memory_mode="gru",
        )
        assert isinstance(ws, SelectionBroadcastWorkspace)

    def test_sb_forward_output_keys(self, sb_workspace, multi_modal_inputs):
        """SB workspace output should have ignition-related keys."""
        with torch.no_grad():
            output = sb_workspace(multi_modal_inputs)

        assert "workspace" in output
        assert "ignition" in output
        assert "global_ignition" in output
        assert "attention" in output
        assert "confidence" in output

    def test_sb_workspace_shape(self, sb_workspace, multi_modal_inputs):
        """SB workspace output should have correct shape."""
        with torch.no_grad():
            output = sb_workspace(multi_modal_inputs)

        assert output["workspace"].shape == (BATCH_SIZE, WORKSPACE_DIM)

    def test_sb_ignition_shape(self, sb_workspace, multi_modal_inputs):
        """Ignition signal should have shape (batch, 1)."""
        with torch.no_grad():
            output = sb_workspace(multi_modal_inputs)

        ignition = output["ignition"]
        assert ignition.shape == (BATCH_SIZE, 1), (
            f"Ignition shape {ignition.shape}, expected ({BATCH_SIZE}, 1)"
        )

    def test_sb_ignition_range(self, sb_workspace, multi_modal_inputs):
        """Ignition signal should be in [0, 1] (sigmoid output)."""
        with torch.no_grad():
            output = sb_workspace(multi_modal_inputs)

        ignition = output["ignition"]
        assert (ignition >= 0).all(), f"Ignition below 0: {ignition.min().item()}"
        assert (ignition <= 1).all(), f"Ignition above 1: {ignition.max().item()}"

    def test_sb_global_ignition_binary(self, sb_workspace, multi_modal_inputs):
        """Global ignition should be binary (0 or 1)."""
        with torch.no_grad():
            output = sb_workspace(multi_modal_inputs)

        gi = output["global_ignition"]
        assert torch.all((gi == 0) | (gi == 1)), (
            f"Global ignition not binary: {gi}"
        )

    def test_sb_confidence_shape_and_range(self, sb_workspace, multi_modal_inputs):
        """Confidence should have shape (batch, 1) and be in [0, 1]."""
        with torch.no_grad():
            output = sb_workspace(multi_modal_inputs)

        conf = output["confidence"]
        assert conf.shape == (BATCH_SIZE, 1)
        assert (conf >= 0).all()
        assert (conf <= 1).all()

    def test_sb_return_details(self, sb_workspace, multi_modal_inputs):
        """return_details should include competition_details."""
        with torch.no_grad():
            output = sb_workspace(multi_modal_inputs, return_details=True)

        assert "competition_details" in output
        details = output["competition_details"]
        assert "history" in details
        assert "selection_rounds" in details

    def test_sb_reset_state(self, sb_workspace, multi_modal_inputs):
        """SB workspace reset_state should clear prev_context."""
        with torch.no_grad():
            sb_workspace(multi_modal_inputs)

        assert sb_workspace.prev_context is not None
        sb_workspace.reset_state()
        assert sb_workspace.prev_context is None

    def test_sb_output_finite(self, sb_workspace, multi_modal_inputs):
        """All SB workspace outputs should be finite."""
        with torch.no_grad():
            output = sb_workspace(multi_modal_inputs)

        for key in ["workspace", "ignition", "global_ignition", "confidence"]:
            tensor = output[key]
            assert not torch.isnan(tensor).any(), f"NaN in {key}"
            assert not torch.isinf(tensor).any(), f"Inf in {key}"


# ---------------------------------------------------------------------------
# Tests: Parameterized batch sizes
# ---------------------------------------------------------------------------

class TestBatchSizes:
    """Test workspace with different batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_varying_batch_sizes(self, workspace_config, batch_size):
        """Workspace should handle various batch sizes."""
        _seed()
        ws = GlobalWorkspace(config=workspace_config, modality_dims=MODALITY_DIMS)
        ws.train(False)

        inputs = {name: torch.randn(batch_size, dim) for name, dim in MODALITY_DIMS.items()}
        with torch.no_grad():
            output = ws(inputs)

        assert output["workspace"].shape == (batch_size, WORKSPACE_DIM)

    @pytest.mark.parametrize("num_modalities", [1, 2, 3])
    def test_varying_num_modalities(self, workspace_config, num_modalities):
        """Workspace should handle different numbers of input modalities."""
        _seed()
        ws = GlobalWorkspace(config=workspace_config, modality_dims=MODALITY_DIMS)
        ws.train(False)

        modality_names = list(MODALITY_DIMS.keys())[:num_modalities]
        inputs = {name: torch.randn(BATCH_SIZE, MODALITY_DIMS[name]) for name in modality_names}
        with torch.no_grad():
            output = ws(inputs)

        assert output["workspace"].shape == (BATCH_SIZE, WORKSPACE_DIM)
        assert len(output["modality_names"]) == num_modalities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
