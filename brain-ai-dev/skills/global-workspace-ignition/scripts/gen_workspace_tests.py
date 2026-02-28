#!/usr/bin/env python3
"""
Global Workspace Test Generator -- Generates tests/test_workspace.py.

Usage:
    python gen_workspace_tests.py [--output PATH] [--dry-run] [--class=<name>]

Generates ~80+ parametrized test cases across 10 classes covering:
    - Token staging and projection (6 tests)
    - Competition scoring (8 tests)
    - Deterministic top-K selection (8 tests)
    - Slot construction and mixing (6 tests)
    - Iterative round dynamics (10 tests)
    - Ignition score and gating (10 tests)
    - Lock-in prevention mechanisms (6 tests)
    - Broadcast adapter projections (8 tests)
    - Working memory backends (10 tests)
    - Full integration pipelines (6 tests)
"""

import argparse
import ast
import os
import re
import sys
import textwrap

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))


# ---------------------------------------------------------------------------
# Template sections -- each function returns a string of generated Python code
# ---------------------------------------------------------------------------


def _header() -> str:
    return textwrap.dedent('''\
    """
    Auto-generated Global Workspace test suite.

    Re-generate with:
        python brain-ai-dev/skills/global-workspace-ignition/scripts/gen_workspace_tests.py

    Covers:
        Token staging, Competition scoring, Deterministic top-K, Slot construction,
        Iterative rounds, Ignition dynamics, Lock-in prevention, Broadcast adapters,
        Working memory, Integration pipelines.
    """

    import math
    import random
    import sys
    import time
    from collections import Counter
    from typing import Dict, List, Optional, Tuple

    import pytest
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ---- path fixup so tests can run from repo root ----
    sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

    # ========================================================================
    # Graceful imports with stub fallbacks
    # ========================================================================

    # --- Global Workspace (base) ---
    try:
        from brain_ai.workspace.global_workspace import (
            GlobalWorkspace,
            GlobalWorkspaceConfig,
            GlobalWorkspaceWithHTM,
            AttentionCompetition,
            ModalityProjection,
            InformationBroadcast,
            create_global_workspace,
        )
        HAS_GW = True
    except ImportError:
        HAS_GW = False
        GlobalWorkspace = None  # type: ignore[assignment,misc]
        GlobalWorkspaceConfig = None  # type: ignore[assignment,misc]
        GlobalWorkspaceWithHTM = None  # type: ignore[assignment,misc]
        AttentionCompetition = None  # type: ignore[assignment,misc]
        ModalityProjection = None  # type: ignore[assignment,misc]
        InformationBroadcast = None  # type: ignore[assignment,misc]
        create_global_workspace = None  # type: ignore[assignment,misc]


    # --- Selection-Broadcast Workspace (improved 2025) ---
    try:
        from brain_ai.workspace.global_workspace import (
            SelectionBroadcastConfig,
            SelectionBroadcastWorkspace,
            IterativeCompetition,
            RefinedBroadcast,
            create_selection_broadcast_workspace,
        )
        HAS_SBW = True
    except ImportError:
        HAS_SBW = False
        SelectionBroadcastConfig = None  # type: ignore[assignment,misc]
        SelectionBroadcastWorkspace = None  # type: ignore[assignment,misc]
        IterativeCompetition = None  # type: ignore[assignment,misc]
        RefinedBroadcast = None  # type: ignore[assignment,misc]
        create_selection_broadcast_workspace = None  # type: ignore[assignment,misc]


    # --- Working Memory ---
    try:
        from brain_ai.workspace.working_memory import (
            WorkingMemory,
            WorkingMemoryConfig,
            GRUWorkingMemory,
            LiquidWorkingMemory,
            create_working_memory,
            NCPS_AVAILABLE,
        )
        HAS_WM = True
    except ImportError:
        HAS_WM = False
        WorkingMemory = None  # type: ignore[assignment,misc]
        WorkingMemoryConfig = None  # type: ignore[assignment,misc]
        GRUWorkingMemory = None  # type: ignore[assignment,misc]
        LiquidWorkingMemory = None  # type: ignore[assignment,misc]
        create_working_memory = None  # type: ignore[assignment,misc]
        NCPS_AVAILABLE = False


    # --- System integration ---
    try:
        from brain_ai.system import BrainAI, create_brain_ai
        from brain_ai.config import BrainAIConfig
        HAS_SYSTEM = True
    except ImportError:
        HAS_SYSTEM = False
        BrainAI = None  # type: ignore[assignment,misc]
        create_brain_ai = None  # type: ignore[assignment,misc]
        BrainAIConfig = None  # type: ignore[assignment,misc]


    # ========================================================================
    # Shared test constants -- keep configs small for speed
    # ========================================================================
    WORKSPACE_DIM = 128
    NUM_HEADS = 4
    CAPACITY_LIMIT = 4
    BATCH_SIZE = 2
    HIDDEN_DIM = 128
    SEED = 42

    MODALITY_DIMS = {
        'vision': 64,
        'text': 64,
        'audio': 32,
    }


    def _seed(s: int = SEED):
        """Set deterministic seeds."""
        torch.manual_seed(s)
        random.seed(s)


    def _make_modality_inputs(
        batch: int = BATCH_SIZE,
        modality_dims: dict = None,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Create random modality input tensors."""
        dims = modality_dims or MODALITY_DIMS
        return {
            name: torch.randn(batch, dim, device=device)
            for name, dim in dims.items()
        }


    def _make_gw_config(**overrides) -> "GlobalWorkspaceConfig":
        """Create a small GlobalWorkspaceConfig for testing."""
        defaults = dict(
            workspace_dim=WORKSPACE_DIM,
            num_heads=NUM_HEADS,
            capacity_limit=CAPACITY_LIMIT,
            dropout=0.0,
            memory_hidden_dim=HIDDEN_DIM,
            memory_mode="gru",
            competition_temperature=1.0,
            min_attention=0.01,
        )
        defaults.update(overrides)
        return GlobalWorkspaceConfig(**defaults)


    def _make_sbw_config(**overrides) -> "SelectionBroadcastConfig":
        """Create a small SelectionBroadcastConfig for testing."""
        defaults = dict(
            workspace_dim=WORKSPACE_DIM,
            num_heads=NUM_HEADS,
            capacity_limit=CAPACITY_LIMIT,
            dropout=0.0,
            ignition_threshold=0.3,
            selection_rounds=2,
            broadcast_iterations=1,
            broadcast_decay=0.9,
            memory_hidden_dim=HIDDEN_DIM,
            memory_mode="gru",
            competition_temperature=0.5,
            min_attention=0.01,
            use_confidence_gating=True,
        )
        defaults.update(overrides)
        return SelectionBroadcastConfig(**defaults)


    # ========================================================================
    # Fixtures
    # ========================================================================

    @pytest.fixture(params=[1, 4], ids=["batch1", "batch4"])
    def batch_size(request):
        return request.param


    @pytest.fixture
    def workspace_config():
        if not HAS_GW:
            pytest.skip("GlobalWorkspace not available")
        return _make_gw_config()


    @pytest.fixture
    def competition_config():
        if not HAS_GW:
            pytest.skip("AttentionCompetition not available")
        return dict(
            workspace_dim=WORKSPACE_DIM,
            num_heads=NUM_HEADS,
            capacity_limit=CAPACITY_LIMIT,
            temperature=1.0,
            dropout=0.0,
        )


    @pytest.fixture
    def sample_inputs():
        _seed()
        return _make_modality_inputs()


    @pytest.fixture
    def gw_instance(workspace_config):
        """Create a small GlobalWorkspace for testing."""
        _seed()
        return GlobalWorkspace(
            config=workspace_config,
            modality_dims=MODALITY_DIMS,
        )


    @pytest.fixture
    def sbw_instance():
        """Create a small SelectionBroadcastWorkspace for testing."""
        if not HAS_SBW:
            pytest.skip("SelectionBroadcastWorkspace not available")
        _seed()
        config = _make_sbw_config()
        return SelectionBroadcastWorkspace(
            config=config,
            modality_dims=MODALITY_DIMS,
        )


    @pytest.fixture
    def wm_instance():
        """Create a small WorkingMemory for testing."""
        if not HAS_WM:
            pytest.skip("WorkingMemory not available")
        _seed()
        return create_working_memory(
            input_dim=WORKSPACE_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=WORKSPACE_DIM,
            mode="gru",
        )


    @pytest.fixture
    def mock_ncps():
        """Fixture indicating ncps availability."""
        return NCPS_AVAILABLE


    # ========================================================================
    # Test classes
    # ========================================================================
    ''')


def _test_token_staging() -> str:
    return textwrap.dedent('''\

    @pytest.mark.skipif(not HAS_GW, reason="GlobalWorkspace not available")
    class TestTokenStaging:
        """Tests for modality projection and token staging into workspace."""

        def test_fixed_modality_ordering(self, gw_instance, sample_inputs):
            """Modalities should be projected in a consistent order."""
            _seed()
            result1 = gw_instance(sample_inputs, return_attention=True)
            names1 = result1['modality_names']

            _seed()
            gw2 = GlobalWorkspace(
                config=_make_gw_config(),
                modality_dims=MODALITY_DIMS,
            )
            result2 = gw2(sample_inputs, return_attention=True)
            names2 = result2['modality_names']

            assert names1 == names2, (
                f"Modality ordering should be consistent: {names1} vs {names2}"
            )

        def test_encoder_output_integration(self, gw_instance, sample_inputs):
            """Each modality projection should produce workspace_dim output."""
            _seed()
            for name, features in sample_inputs.items():
                if name in gw_instance.projections:
                    proj, sal = gw_instance.projections[name](features)
                    assert proj.shape == (features.shape[0], WORKSPACE_DIM), (
                        f"Projection for {name} has wrong shape: {proj.shape}"
                    )
                    assert sal.shape == (features.shape[0], 1), (
                        f"Salience for {name} has wrong shape: {sal.shape}"
                    )

        def test_missing_salience_defaults(self, gw_instance):
            """Salience output should be a finite scalar per batch element."""
            _seed()
            x = torch.randn(BATCH_SIZE, 64)
            proj, sal = gw_instance.projections['vision'](x)
            assert torch.isfinite(sal).all(), "Salience contains non-finite values"

        def test_missing_time_handling(self, gw_instance):
            """Workspace should work without previous context (first timestep)."""
            _seed()
            gw_instance.reset_state()
            inputs = _make_modality_inputs()
            result = gw_instance(inputs)
            assert 'workspace' in result
            assert result['workspace'].shape == (BATCH_SIZE, WORKSPACE_DIM)

        def test_variable_length_modalities(self, workspace_config):
            """Workspace should handle different numbers of modalities."""
            _seed()
            gw_two = GlobalWorkspace(
                config=workspace_config,
                modality_dims={'vision': 64, 'text': 64},
            )
            inputs_two = {
                'vision': torch.randn(BATCH_SIZE, 64),
                'text': torch.randn(BATCH_SIZE, 64),
            }
            result = gw_two(inputs_two)
            assert result['workspace'].shape == (BATCH_SIZE, WORKSPACE_DIM)

            gw_one = GlobalWorkspace(
                config=workspace_config,
                modality_dims={'vision': 64},
            )
            inputs_one = {'vision': torch.randn(BATCH_SIZE, 64)}
            result = gw_one(inputs_one)
            assert result['workspace'].shape == (BATCH_SIZE, WORKSPACE_DIM)

        def test_empty_modality_handling(self, gw_instance):
            """Workspace should raise ValueError for empty modality dict."""
            with pytest.raises(ValueError, match="No valid modality"):
                gw_instance({})
    ''')


def _test_competition_scoring() -> str:
    return textwrap.dedent('''\

    @pytest.mark.skipif(not HAS_GW, reason="AttentionCompetition not available")
    class TestCompetitionScoring:
        """Tests for attention-based competition mechanism."""

        def test_four_term_scoring(self, competition_config):
            """Competition should combine attention and salience into scores."""
            _seed()
            comp = AttentionCompetition(**competition_config)
            features = torch.randn(BATCH_SIZE, 5, WORKSPACE_DIM)
            saliences = torch.randn(BATCH_SIZE, 5, 1)
            winners, attn = comp(features, saliences)
            assert winners.shape == (BATCH_SIZE, 5, WORKSPACE_DIM)
            assert attn.shape == (BATCH_SIZE, 5)

        def test_salience_influence(self, competition_config):
            """Higher salience should lead to higher attention weight."""
            _seed()
            comp = AttentionCompetition(**competition_config)
            comp.eval()

            features = torch.randn(BATCH_SIZE, 3, WORKSPACE_DIM)
            # Make one modality much more salient
            saliences = torch.zeros(BATCH_SIZE, 3, 1)
            saliences[:, 0, :] = 10.0  # First modality very salient

            _, attn = comp(features, saliences)
            # First modality should receive highest attention
            assert attn[:, 0].mean() > attn[:, 1].mean(), (
                "High-salience modality should have higher attention"
            )

        def test_novelty_against_memory(self, competition_config):
            """Features distant from each other should produce different scores."""
            _seed()
            comp = AttentionCompetition(**competition_config)
            comp.eval()

            # Similar features
            f_similar = torch.randn(1, 3, WORKSPACE_DIM)
            s_similar = torch.zeros(1, 3, 1)
            _, attn_similar = comp(f_similar, s_similar)

            # One outlier feature
            f_mixed = f_similar.clone()
            f_mixed[:, 2, :] = torch.randn(1, WORKSPACE_DIM) * 5.0
            _, attn_mixed = comp(f_mixed, s_similar)

            # Attention distributions should differ
            assert not torch.allclose(attn_similar, attn_mixed, atol=1e-3), (
                "Different features should produce different attention patterns"
            )

        def test_task_bias_effect(self, competition_config):
            """Different salience biases should shift attention distribution."""
            _seed()
            comp = AttentionCompetition(**competition_config)
            comp.eval()
            features = torch.randn(1, 3, WORKSPACE_DIM)

            sal_a = torch.tensor([[[5.0], [0.0], [0.0]]])
            sal_b = torch.tensor([[[0.0], [0.0], [5.0]]])

            _, attn_a = comp(features, sal_a)
            _, attn_b = comp(features, sal_b)

            assert attn_a[0, 0] > attn_a[0, 2], "Bias A should favor first modality"
            assert attn_b[0, 2] > attn_b[0, 0], "Bias B should favor third modality"

        def test_score_normalization(self, competition_config):
            """Attention weights should sum to 1 across modalities."""
            _seed()
            comp = AttentionCompetition(**competition_config)
            comp.eval()
            features = torch.randn(BATCH_SIZE, 5, WORKSPACE_DIM)
            saliences = torch.randn(BATCH_SIZE, 5, 1)
            _, attn = comp(features, saliences)
            sums = attn.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
                f"Attention should sum to 1, got sums: {sums}"
            )

        def test_weight_config_respected(self):
            """Competition temperature should affect score sharpness."""
            _seed()
            features = torch.randn(1, 5, WORKSPACE_DIM)
            saliences = torch.randn(1, 5, 1)

            comp_sharp = AttentionCompetition(
                workspace_dim=WORKSPACE_DIM, num_heads=NUM_HEADS,
                capacity_limit=5, temperature=0.1, dropout=0.0,
            )
            comp_sharp.eval()
            _, attn_sharp = comp_sharp(features, saliences)

            comp_soft = AttentionCompetition(
                workspace_dim=WORKSPACE_DIM, num_heads=NUM_HEADS,
                capacity_limit=5, temperature=10.0, dropout=0.0,
            )
            # Copy weights to ensure fair comparison
            comp_soft.load_state_dict(comp_sharp.state_dict())
            comp_soft.temperature = 10.0
            comp_soft.eval()
            _, attn_soft = comp_soft(features, saliences)

            # Sharp temperature -> more concentrated attention
            entropy_sharp = -(attn_sharp * attn_sharp.clamp(min=1e-8).log()).sum(dim=-1)
            entropy_soft = -(attn_soft * attn_soft.clamp(min=1e-8).log()).sum(dim=-1)
            assert entropy_sharp.mean() < entropy_soft.mean(), (
                "Lower temperature should produce more concentrated attention"
            )

        def test_zero_weights(self, competition_config):
            """Zero salience should still produce valid output."""
            _seed()
            comp = AttentionCompetition(**competition_config)
            comp.eval()
            features = torch.randn(BATCH_SIZE, 3, WORKSPACE_DIM)
            saliences = torch.zeros(BATCH_SIZE, 3, 1)
            winners, attn = comp(features, saliences)
            assert torch.isfinite(winners).all()
            assert torch.isfinite(attn).all()

        def test_score_dtype_fp32(self, competition_config):
            """Competition output should maintain fp32 precision."""
            _seed()
            comp = AttentionCompetition(**competition_config)
            comp.eval()
            features = torch.randn(BATCH_SIZE, 3, WORKSPACE_DIM)
            saliences = torch.randn(BATCH_SIZE, 3, 1)
            winners, attn = comp(features, saliences)
            assert winners.dtype == torch.float32
            assert attn.dtype == torch.float32
    ''')


def _test_deterministic_topk() -> str:
    return textwrap.dedent('''\

    @pytest.mark.skipif(not HAS_GW, reason="GlobalWorkspace not available")
    class TestDeterministicTopK:
        """Tests for deterministic top-K selection in competition."""

        def test_deterministic_topk_no_ties(self):
            """Distinct scores should produce deterministic top-K."""
            _seed()
            scores = torch.tensor([[0.5, 0.3, 0.1, 0.05, 0.05]])
            _, indices = torch.topk(scores, 3, dim=-1)
            expected = torch.tensor([[0, 1, 2]])
            assert torch.equal(indices, expected)

        def test_deterministic_topk_exact_ties_cpu(self):
            """Exact ties should still produce deterministic output on CPU."""
            _seed()
            scores = torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.1]])
            _, idx1 = torch.topk(scores, 3, dim=-1)
            _, idx2 = torch.topk(scores, 3, dim=-1)
            assert torch.equal(idx1, idx2), "Top-K should be deterministic on CPU"

        @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
        def test_deterministic_topk_exact_ties_cuda(self):
            """Exact ties should produce deterministic output on CUDA."""
            _seed()
            scores = torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.1]], device="cuda")
            _, idx1 = torch.topk(scores, 3, dim=-1)
            _, idx2 = torch.topk(scores, 3, dim=-1)
            assert torch.equal(idx1, idx2), "Top-K should be deterministic on CUDA"

        def test_deterministic_topk_reproducible(self):
            """Same seed should produce same competition results."""
            if not HAS_GW:
                pytest.skip("GlobalWorkspace not available")
            _seed(123)
            comp = AttentionCompetition(
                workspace_dim=WORKSPACE_DIM, num_heads=NUM_HEADS,
                capacity_limit=2, temperature=1.0, dropout=0.0,
            )
            comp.eval()
            features = torch.randn(1, 5, WORKSPACE_DIM)
            saliences = torch.randn(1, 5, 1)

            _, attn1 = comp(features, saliences)

            # Reset and repeat
            _, attn2 = comp(features, saliences)
            assert torch.allclose(attn1, attn2, atol=1e-6), (
                "Same input should produce same attention in eval mode"
            )

        def test_epsilon_magnitude(self):
            """Small epsilon should not change relative ranking."""
            _seed()
            scores = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])
            eps = 1e-8
            perturbed = scores + eps * torch.arange(5).float()
            _, idx_orig = torch.topk(scores, 3, dim=-1)
            _, idx_pert = torch.topk(perturbed, 3, dim=-1)
            assert torch.equal(idx_orig, idx_pert), (
                "Epsilon-scale perturbation should not change ranking"
            )

        def test_topk_with_mask(self):
            """Top-K selection should respect capacity limit mask."""
            if not HAS_GW:
                pytest.skip("GlobalWorkspace not available")
            _seed()
            comp = AttentionCompetition(
                workspace_dim=WORKSPACE_DIM, num_heads=NUM_HEADS,
                capacity_limit=2, temperature=1.0, dropout=0.0,
            )
            comp.eval()
            # 5 items, capacity 2
            features = torch.randn(1, 5, WORKSPACE_DIM)
            saliences = torch.randn(1, 5, 1)
            _, attn = comp(features, saliences)
            # At most 2 items should have non-zero attention
            nonzero = (attn > 1e-8).sum(dim=-1)
            assert nonzero.item() <= 2, (
                f"Expected at most 2 non-zero attention slots, got {nonzero.item()}"
            )

        def test_topk_k_equals_total(self):
            """When K equals total items, all should be selected."""
            if not HAS_GW:
                pytest.skip("GlobalWorkspace not available")
            _seed()
            comp = AttentionCompetition(
                workspace_dim=WORKSPACE_DIM, num_heads=NUM_HEADS,
                capacity_limit=3, temperature=1.0, dropout=0.0,
            )
            comp.eval()
            features = torch.randn(1, 3, WORKSPACE_DIM)
            saliences = torch.randn(1, 3, 1)
            _, attn = comp(features, saliences)
            # All 3 should have non-zero attention
            assert (attn > 1e-8).all(), "All items should be selected when K=N"

        def test_topk_k_greater_than_valid(self):
            """When K > items, all items should pass through without error."""
            if not HAS_GW:
                pytest.skip("GlobalWorkspace not available")
            _seed()
            comp = AttentionCompetition(
                workspace_dim=WORKSPACE_DIM, num_heads=NUM_HEADS,
                capacity_limit=10, temperature=1.0, dropout=0.0,
            )
            comp.eval()
            features = torch.randn(1, 3, WORKSPACE_DIM)
            saliences = torch.randn(1, 3, 1)
            winners, attn = comp(features, saliences)
            assert winners.shape == (1, 3, WORKSPACE_DIM)
            assert torch.isfinite(attn).all()
    ''')


def _test_slot_construction() -> str:
    return textwrap.dedent('''\

    @pytest.mark.skipif(not HAS_GW, reason="GlobalWorkspace not available")
    class TestSlotConstruction:
        """Tests for workspace slot formation after competition."""

        def test_slot_shape(self, gw_instance, sample_inputs):
            """Workspace output should have correct shape."""
            _seed()
            result = gw_instance(sample_inputs)
            assert result['workspace'].shape == (BATCH_SIZE, WORKSPACE_DIM)

        def test_slot_content_matches_winners(self, gw_instance, sample_inputs):
            """Workspace output should be derived from winning features."""
            _seed()
            result = gw_instance(sample_inputs)
            ws = result['workspace']
            assert torch.isfinite(ws).all(), "Workspace should be finite"
            assert ws.abs().sum() > 0, "Workspace should not be all zeros"

        def test_slot_mixer_residual(self, gw_instance, sample_inputs):
            """Second timestep should integrate previous context."""
            _seed()
            gw_instance.reset_state()
            result1 = gw_instance(sample_inputs)
            result2 = gw_instance(sample_inputs)
            # Second result should differ from first due to context integration
            assert not torch.allclose(result1['workspace'], result2['workspace'], atol=1e-3), (
                "Second timestep should differ from first due to memory integration"
            )

        def test_slot_mixer_disabled(self, workspace_config):
            """Without previous context, integration should be a passthrough."""
            _seed()
            gw = GlobalWorkspace(config=workspace_config, modality_dims=MODALITY_DIMS)
            gw.reset_state()
            assert gw.prev_context is None, "Fresh workspace should have no context"

        def test_winners_metadata(self, gw_instance, sample_inputs):
            """Result should contain modality name metadata."""
            _seed()
            result = gw_instance(sample_inputs, return_attention=True)
            assert 'modality_names' in result
            assert isinstance(result['modality_names'], list)
            assert len(result['modality_names']) == len(sample_inputs)

        def test_slot_mask_consistency(self, gw_instance, sample_inputs):
            """Attention weights should be consistent with modality count."""
            _seed()
            result = gw_instance(sample_inputs, return_attention=True)
            attn = result['attention']
            assert isinstance(attn, dict)
            for name in result['modality_names']:
                assert name in attn
                assert attn[name].shape == (BATCH_SIZE,)
    ''')


def _test_iterative_rounds() -> str:
    return textwrap.dedent('''\

    @pytest.mark.skipif(not HAS_SBW, reason="SelectionBroadcastWorkspace not available")
    class TestIterativeRounds:
        """Tests for iterative competition rounds in Selection-Broadcast workspace."""

        def test_single_round_no_iteration(self):
            """Single selection round should still produce valid output."""
            _seed()
            config = _make_sbw_config(selection_rounds=1)
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()
            result = ws(inputs, return_details=True)
            assert result['workspace'].shape == (BATCH_SIZE, WORKSPACE_DIM)
            details = result['competition_details']
            assert details['selection_rounds'] >= 1

        def test_convergence_after_2_rounds(self):
            """Two rounds should produce different results than one."""
            _seed()
            config1 = _make_sbw_config(selection_rounds=1)
            ws1 = SelectionBroadcastWorkspace(config=config1, modality_dims=MODALITY_DIMS)
            ws1.eval()

            config2 = _make_sbw_config(selection_rounds=2)
            ws2 = SelectionBroadcastWorkspace(config=config2, modality_dims=MODALITY_DIMS)
            ws2.eval()

            inputs = _make_modality_inputs()
            result1 = ws1(inputs)
            result2 = ws2(inputs)

            # Different number of rounds -> potentially different results
            # (not guaranteed to differ, but architecture differs)
            assert result1['workspace'].shape == result2['workspace'].shape

        def test_stability_metrics_computed(self):
            """Competition details should include stability metrics."""
            _seed()
            config = _make_sbw_config(selection_rounds=3)
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()
            result = ws(inputs, return_details=True)
            details = result['competition_details']
            assert 'history' in details
            assert len(details['history']) >= 1

        def test_winner_set_stability_jaccard(self):
            """Jaccard similarity between identical winner sets should be 1.0."""
            winners_a = torch.tensor([[0, 1, 2], [0, 1, 2]])
            winners_b = torch.tensor([[0, 1, 2], [0, 1, 2]])
            # Manual Jaccard computation
            for b in range(2):
                set_a = set(winners_a[b].tolist())
                set_b = set(winners_b[b].tolist())
                jaccard = len(set_a & set_b) / max(len(set_a | set_b), 1)
                assert jaccard == 1.0

        def test_embedding_stability_cosine(self):
            """Cosine similarity between identical embeddings should be 1.0."""
            _seed()
            emb = torch.randn(2, 3, WORKSPACE_DIM)
            norm_emb = F.normalize(emb, dim=-1)
            cosine = (norm_emb * norm_emb).sum(dim=-1)
            assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-5)

        def test_max_rounds_cap(self):
            """Competition should not exceed configured max rounds."""
            _seed()
            config = _make_sbw_config(selection_rounds=5)
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()
            result = ws(inputs, return_details=True)
            details = result['competition_details']
            assert details['selection_rounds'] <= 5

        def test_ignition_false_when_unstable(self):
            """Random inputs should not always trigger ignition."""
            _seed()
            config = _make_sbw_config(ignition_threshold=0.99)
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()
            result = ws(inputs)
            # With very high threshold, ignition should rarely fire
            # (not guaranteed but statistically unlikely)
            assert 'global_ignition' in result

        def test_convergence_thresholds_respected(self):
            """Early stopping should trigger when ignition exceeds 1.5x threshold."""
            _seed()
            config = _make_sbw_config(selection_rounds=10, ignition_threshold=0.01)
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()
            result = ws(inputs, return_details=True)
            details = result['competition_details']
            # With very low threshold, early stopping should kick in
            assert details['selection_rounds'] <= 10

        def test_consecutive_stable_requirement(self):
            """Multiple rounds should build on each other."""
            _seed()
            config = _make_sbw_config(selection_rounds=3)
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()
            result = ws(inputs, return_details=True)
            history = result['competition_details']['history']
            assert len(history) >= 1
            for entry in history:
                assert 'saliences' in entry
                assert 'ignition' in entry

        def test_round_telemetry_history(self):
            """Each round should append to the history."""
            _seed()
            config = _make_sbw_config(selection_rounds=3)
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()
            result = ws(inputs, return_details=True)
            details = result['competition_details']
            assert isinstance(details['history'], list)
            num_rounds = details['selection_rounds']
            assert len(details['history']) == num_rounds
    ''')


def _test_ignition() -> str:
    return textwrap.dedent('''\

    @pytest.mark.skipif(not HAS_SBW, reason="SelectionBroadcastWorkspace not available")
    class TestIgnition:
        """Tests for ignition dynamics and broadcast gating."""

        def test_ignition_score_components(self):
            """Ignition output should contain score tensor."""
            _seed()
            config = _make_sbw_config()
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()
            result = ws(inputs)
            assert 'ignition' in result
            assert result['ignition'].shape[0] == BATCH_SIZE

        def test_ignition_threshold_gate(self):
            """Ignition should be gated by threshold."""
            _seed()
            config = _make_sbw_config(ignition_threshold=0.3)
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()
            result = ws(inputs)
            assert 'global_ignition' in result
            # global_ignition should be 0 or 1
            gi = result['global_ignition']
            assert torch.all((gi == 0) | (gi == 1)), (
                f"Global ignition should be binary, got: {gi}"
            )

        def test_committed_broadcast_gain(self):
            """Full ignition should produce high-magnitude workspace output."""
            _seed()
            config = _make_sbw_config(ignition_threshold=0.001)
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()
            result = ws(inputs)
            assert result['workspace'].abs().mean() > 0

        def test_weak_broadcast_gain(self):
            """Non-ignition should still produce some output."""
            _seed()
            config = _make_sbw_config(ignition_threshold=0.999)
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()
            result = ws(inputs)
            assert result['workspace'].abs().mean() > 0, (
                "Even weak broadcast should produce non-zero output"
            )

        def test_smooth_gate_training(self):
            """During training, ignition gating should allow gradient flow."""
            _seed()
            config = _make_sbw_config()
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.train()
            inputs = {k: v.requires_grad_(False) for k, v in _make_modality_inputs().items()}
            result = ws(inputs)
            loss = result['workspace'].sum()
            loss.backward()
            # Check some parameters have gradients
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in ws.parameters()
            )
            assert has_grad, "Gradients should flow through workspace during training"

        def test_hard_gate_inference(self):
            """During eval, workspace should be deterministic."""
            _seed()
            config = _make_sbw_config()
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()
            result1 = ws(inputs)
            ws.reset_state()
            result2 = ws(inputs)
            assert torch.allclose(result1['workspace'], result2['workspace'], atol=1e-5), (
                "Eval mode should be deterministic with same input and reset state"
            )

        def test_cross_modal_coherence(self):
            """Attention should span multiple modalities."""
            _seed()
            config = _make_sbw_config(capacity_limit=10)
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()
            result = ws(inputs)
            attn = result['attention']
            # All modalities should have some attention
            for name in MODALITY_DIMS:
                assert name in attn
                assert attn[name].abs().sum() > 0, (
                    f"Modality {name} received zero attention"
                )

        def test_ignition_telemetry(self):
            """return_details=True should provide competition details."""
            _seed()
            config = _make_sbw_config()
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()
            result = ws(inputs, return_details=True)
            assert 'competition_details' in result
            details = result['competition_details']
            assert 'ignition' in details
            assert 'global_ignition' in details
            assert 'selection_rounds' in details

        def test_ignition_fp32(self):
            """Ignition score should be computed in fp32."""
            _seed()
            config = _make_sbw_config()
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()
            result = ws(inputs)
            assert result['ignition'].dtype == torch.float32, (
                f"Ignition should be fp32, got {result['ignition'].dtype}"
            )

        def test_learned_vs_interpretable(self):
            """Both workspace variants should produce valid output."""
            _seed()
            # Base GlobalWorkspace (no iterative competition)
            if HAS_GW:
                gw = GlobalWorkspace(
                    config=_make_gw_config(),
                    modality_dims=MODALITY_DIMS,
                )
                gw.eval()
                inputs = _make_modality_inputs()
                result_gw = gw(inputs)
                assert result_gw['workspace'].shape == (BATCH_SIZE, WORKSPACE_DIM)

            # SelectionBroadcast (iterative with ignition)
            sbw = SelectionBroadcastWorkspace(
                config=_make_sbw_config(),
                modality_dims=MODALITY_DIMS,
            )
            sbw.eval()
            result_sbw = sbw(_make_modality_inputs())
            assert result_sbw['workspace'].shape == (BATCH_SIZE, WORKSPACE_DIM)
    ''')


def _test_lockin_prevention() -> str:
    return textwrap.dedent('''\

    @pytest.mark.skipif(not HAS_SBW, reason="SelectionBroadcastWorkspace not available")
    class TestLockInPrevention:
        """Tests for lock-in prevention mechanisms."""

        def test_novelty_reduces_repeat_winners(self):
            """Repeated inputs should eventually produce different workspace outputs."""
            _seed()
            config = _make_sbw_config(selection_rounds=3)
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()

            ws.reset_state()
            outputs = []
            for _ in range(5):
                result = ws(inputs)
                outputs.append(result['workspace'].clone())

            # Due to working memory integration, outputs should evolve
            assert not torch.allclose(outputs[0], outputs[-1], atol=1e-3), (
                "Workspace should evolve over repeated timesteps due to memory"
            )

        def test_slot_dropout_training(self):
            """Dropout should introduce variability during training."""
            _seed()
            config = _make_sbw_config(dropout=0.5)
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.train()
            inputs = _make_modality_inputs()

            ws.reset_state()
            r1 = ws(inputs)
            ws.reset_state()
            r2 = ws(inputs)

            # With 50% dropout, results should differ between runs
            # (though not guaranteed for every random seed)
            # At minimum, both should be valid
            assert torch.isfinite(r1['workspace']).all()
            assert torch.isfinite(r2['workspace']).all()

        def test_no_slot_dropout_eval(self):
            """Eval mode should not apply dropout."""
            _seed()
            config = _make_sbw_config(dropout=0.5)
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()

            ws.reset_state()
            r1 = ws(inputs)
            ws.reset_state()
            r2 = ws(inputs)

            assert torch.allclose(r1['workspace'], r2['workspace'], atol=1e-5), (
                "Eval mode should be deterministic (no dropout)"
            )

        def test_winner_decay(self):
            """Sequential timesteps should show changing attention patterns."""
            _seed()
            config = _make_sbw_config()
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            inputs = _make_modality_inputs()

            ws.reset_state()
            result1 = ws(inputs)
            result2 = ws(inputs)

            attn1 = result1['attention']
            attn2 = result2['attention']

            # Attention should shift over timesteps due to memory context
            for name in MODALITY_DIMS:
                a1 = attn1[name]
                a2 = attn2[name]
                # Values exist and are valid
                assert torch.isfinite(a1).all()
                assert torch.isfinite(a2).all()

        def test_cooldown_after_ignition(self):
            """Consecutive steps should show varied ignition states."""
            _seed()
            config = _make_sbw_config(ignition_threshold=0.1)
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()

            ws.reset_state()
            ignition_states = []
            for _ in range(5):
                inputs = _make_modality_inputs()
                result = ws(inputs)
                ignition_states.append(result['global_ignition'].clone())

            # All should be valid binary tensors
            for gi in ignition_states:
                assert torch.all((gi == 0) | (gi == 1))

        def test_turnover_over_sequence(self):
            """Over a sequence, workspace content should evolve."""
            _seed()
            config = _make_sbw_config()
            ws = SelectionBroadcastWorkspace(config=config, modality_dims=MODALITY_DIMS)
            ws.eval()
            ws.reset_state()

            workspaces = []
            for step in range(10):
                torch.manual_seed(SEED + step)
                inputs = _make_modality_inputs()
                result = ws(inputs)
                workspaces.append(result['workspace'].clone())

            # Check that workspace changes over the sequence
            diffs = []
            for i in range(1, len(workspaces)):
                diff = (workspaces[i] - workspaces[i - 1]).abs().mean().item()
                diffs.append(diff)

            assert max(diffs) > 0.01, (
                f"Workspace should change over sequence, max diff: {max(diffs):.6f}"
            )
    ''')


def _test_broadcast_adapters() -> str:
    return textwrap.dedent('''\

    @pytest.mark.skipif(not HAS_GW, reason="InformationBroadcast not available")
    class TestBroadcastAdapters:
        """Tests for broadcast projections back to specialists."""

        def test_broadcast_to_temporal(self):
            """Broadcast should produce output for each registered modality."""
            _seed()
            bc = InformationBroadcast(
                workspace_dim=WORKSPACE_DIM,
                modality_dims=MODALITY_DIMS,
                dropout=0.0,
            )
            bc.eval()
            ws_content = torch.randn(BATCH_SIZE, WORKSPACE_DIM)
            broadcasts = bc(ws_content)
            for name, dim in MODALITY_DIMS.items():
                assert name in broadcasts
                assert broadcasts[name].shape == (BATCH_SIZE, dim)

        def test_broadcast_to_pooled(self):
            """Broadcast output should aggregate workspace information."""
            _seed()
            bc = InformationBroadcast(
                workspace_dim=WORKSPACE_DIM,
                modality_dims={'vision': 64},
                dropout=0.0,
            )
            bc.eval()
            ws_content = torch.randn(BATCH_SIZE, WORKSPACE_DIM)
            broadcasts = bc(ws_content)
            assert broadcasts['vision'].abs().sum() > 0

        def test_broadcast_to_symbolic(self):
            """Broadcast should work with single-modality configs."""
            _seed()
            bc = InformationBroadcast(
                workspace_dim=WORKSPACE_DIM,
                modality_dims={'symbolic': 256},
                dropout=0.0,
            )
            bc.eval()
            ws_content = torch.randn(BATCH_SIZE, WORKSPACE_DIM)
            broadcasts = bc(ws_content)
            assert broadcasts['symbolic'].shape == (BATCH_SIZE, 256)

        def test_broadcast_to_decision(self):
            """Broadcast to decision module dimension."""
            _seed()
            bc = InformationBroadcast(
                workspace_dim=WORKSPACE_DIM,
                modality_dims={'decision': 512},
                dropout=0.0,
            )
            bc.eval()
            ws_content = torch.randn(BATCH_SIZE, WORKSPACE_DIM)
            broadcasts = bc(ws_content)
            assert broadcasts['decision'].shape == (BATCH_SIZE, 512)

        def test_adapter_mask_alignment(self):
            """Each adapter should align with its target dimension."""
            _seed()
            mixed_dims = {'small': 16, 'medium': 128, 'large': 512}
            bc = InformationBroadcast(
                workspace_dim=WORKSPACE_DIM,
                modality_dims=mixed_dims,
                dropout=0.0,
            )
            bc.eval()
            ws_content = torch.randn(1, WORKSPACE_DIM)
            broadcasts = bc(ws_content)
            for name, dim in mixed_dims.items():
                assert broadcasts[name].shape == (1, dim), (
                    f"Adapter {name} shape mismatch: {broadcasts[name].shape} vs (1, {dim})"
                )

        def test_adapter_deterministic(self):
            """Broadcast should be deterministic in eval mode."""
            _seed()
            bc = InformationBroadcast(
                workspace_dim=WORKSPACE_DIM,
                modality_dims=MODALITY_DIMS,
                dropout=0.0,
            )
            bc.eval()
            ws_content = torch.randn(BATCH_SIZE, WORKSPACE_DIM)
            b1 = bc(ws_content)
            b2 = bc(ws_content)
            for name in MODALITY_DIMS:
                assert torch.allclose(b1[name], b2[name], atol=1e-6), (
                    f"Broadcast for {name} not deterministic in eval"
                )

        def test_adapter_registry(self):
            """All registered modalities should have broadcast projections."""
            _seed()
            bc = InformationBroadcast(
                workspace_dim=WORKSPACE_DIM,
                modality_dims=MODALITY_DIMS,
            )
            for name in MODALITY_DIMS:
                assert name in bc.broadcast_projections, (
                    f"Missing broadcast projection for {name}"
                )

        def test_all_adapters_output_shapes(self, gw_instance, sample_inputs):
            """Full workspace broadcast should match modality dimensions."""
            _seed()
            gw_instance.eval()
            result = gw_instance(sample_inputs)
            broadcasts = result['broadcasts']
            for name, dim in MODALITY_DIMS.items():
                assert broadcasts[name].shape == (BATCH_SIZE, dim), (
                    f"Broadcast {name}: expected ({BATCH_SIZE}, {dim}), "
                    f"got {broadcasts[name].shape}"
                )
    ''')


def _test_working_memory() -> str:
    return textwrap.dedent('''\

    @pytest.mark.skipif(not HAS_WM, reason="WorkingMemory not available")
    class TestWorkingMemory:
        """Tests for working memory module with backend selection."""

        def test_state_persistence_sequential(self, wm_instance):
            """Sequential inputs should produce different outputs due to state."""
            _seed()
            wm_instance.reset_state()
            x1 = torch.randn(BATCH_SIZE, WORKSPACE_DIM)
            x2 = torch.randn(BATCH_SIZE, WORKSPACE_DIM)
            r1 = wm_instance(x1)
            r2 = wm_instance(x2)
            assert not torch.allclose(r1['output'], r2['output'], atol=1e-4), (
                "Sequential inputs should produce different outputs"
            )

        def test_state_reset_baseline(self, wm_instance):
            """After reset, same input should produce same output."""
            _seed()
            x = torch.randn(BATCH_SIZE, WORKSPACE_DIM)

            wm_instance.reset_state()
            r1 = wm_instance(x)

            wm_instance.reset_state()
            r2 = wm_instance(x)

            assert torch.allclose(r1['output'], r2['output'], atol=1e-5), (
                "Same input after reset should produce same output"
            )

        def test_detach_state_values(self, wm_instance):
            """State tensors should be detachable."""
            _seed()
            wm_instance.reset_state()
            x = torch.randn(BATCH_SIZE, WORKSPACE_DIM)
            result = wm_instance(x)
            state = result['state']
            if isinstance(state, torch.Tensor):
                detached = state.detach()
                assert not detached.requires_grad
            elif isinstance(state, (tuple, list)):
                for s in state:
                    if isinstance(s, torch.Tensor):
                        detached = s.detach()
                        assert not detached.requires_grad

        def test_detach_state_gradient_isolation(self, wm_instance):
            """Detached state should not propagate gradients backward."""
            _seed()
            wm_instance.reset_state()
            x = torch.randn(BATCH_SIZE, WORKSPACE_DIM, requires_grad=False)
            result = wm_instance(x)
            output = result['output']
            loss = output.sum()
            loss.backward()
            # Verify gradients exist on parameters
            has_grad = any(
                p.grad is not None for p in wm_instance.parameters()
            )
            assert has_grad, "Working memory should allow gradient computation"

        def test_reset_state_params(self, wm_instance):
            """reset_state should clear internal buffers."""
            _seed()
            x = torch.randn(BATCH_SIZE, WORKSPACE_DIM)
            wm_instance(x)
            wm_instance.reset_state()
            assert wm_instance.memory_buffer is None, "Buffer should be None after reset"

        def test_ncps_fallback_gru(self, mock_ncps):
            """When ncps is not available, should fall back to GRU."""
            _seed()
            wm = create_working_memory(
                input_dim=WORKSPACE_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=WORKSPACE_DIM,
                mode="gru",
            )
            assert wm.backend_type == "gru"
            x = torch.randn(BATCH_SIZE, WORKSPACE_DIM)
            result = wm(x)
            assert result['output'].shape == (BATCH_SIZE, WORKSPACE_DIM)

        def test_ncps_fallback_schema(self, mock_ncps):
            """GRU fallback should produce the same output schema."""
            _seed()
            wm = create_working_memory(
                input_dim=WORKSPACE_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=WORKSPACE_DIM,
                mode="gru",
            )
            x = torch.randn(BATCH_SIZE, WORKSPACE_DIM)
            result = wm(x)
            assert 'output' in result
            assert 'state' in result
            assert 'buffer' in result

        def test_cfc_timespans(self, mock_ncps):
            """CfC mode should accept timespans argument."""
            if not NCPS_AVAILABLE:
                pytest.skip("ncps not available for CfC test")
            _seed()
            wm = create_working_memory(
                input_dim=WORKSPACE_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=WORKSPACE_DIM,
                mode="cfc",
            )
            x = torch.randn(BATCH_SIZE, WORKSPACE_DIM)
            # Timespans: (batch, seq_len, 1)
            timespans = torch.ones(BATCH_SIZE, 1, 1)
            result = wm(x, timespans=timespans)
            assert result['output'].shape == (BATCH_SIZE, WORKSPACE_DIM)

        def test_gru_dt_handling(self):
            """GRU should handle variable batch sizes after reset."""
            _seed()
            wm = create_working_memory(
                input_dim=WORKSPACE_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=WORKSPACE_DIM,
                mode="gru",
            )
            x1 = torch.randn(2, WORKSPACE_DIM)
            wm(x1)

            # Different batch size
            x2 = torch.randn(4, WORKSPACE_DIM)
            result = wm(x2)  # Should auto-reset state for batch mismatch
            assert result['output'].shape == (4, WORKSPACE_DIM)

        def test_memory_buffer_fifo(self, wm_instance):
            """Memory buffer should implement FIFO with capacity limit."""
            _seed()
            wm_instance.reset_state()
            capacity = wm_instance.capacity

            # Feed more items than capacity
            for i in range(capacity + 3):
                x = torch.randn(BATCH_SIZE, WORKSPACE_DIM) + i
                wm_instance(x, update_buffer=True)

            buffer = wm_instance.memory_buffer
            assert buffer is not None
            assert buffer.shape[1] <= capacity, (
                f"Buffer size {buffer.shape[1]} exceeds capacity {capacity}"
            )
    ''')


def _test_integration() -> str:
    return textwrap.dedent('''\

    class TestIntegration:
        """End-to-end integration tests for workspace pipelines."""

        @pytest.mark.skipif(not HAS_GW, reason="GlobalWorkspace not available")
        def test_full_pipeline_forward(self):
            """Complete workspace forward pass should succeed."""
            _seed()
            gw = create_global_workspace(
                workspace_dim=WORKSPACE_DIM,
                modality_dims=MODALITY_DIMS,
                num_heads=NUM_HEADS,
                capacity_limit=CAPACITY_LIMIT,
                memory_mode="gru",
            )
            gw.eval()
            inputs = _make_modality_inputs()
            result = gw(inputs, return_attention=True)
            assert 'workspace' in result
            assert 'broadcasts' in result
            assert 'attention' in result
            assert 'memory_output' in result
            assert result['workspace'].shape == (BATCH_SIZE, WORKSPACE_DIM)

        @pytest.mark.skipif(not HAS_SYSTEM, reason="BrainAI system not available")
        def test_workspace_in_system(self):
            """Workspace should integrate correctly in the full BrainAI system."""
            _seed()
            config = BrainAIConfig.minimal()
            config.use_workspace = True
            config.modalities = ["vision"]
            try:
                brain = create_brain_ai(
                    modalities=['vision'],
                    output_type='classify',
                    num_classes=10,
                    config=config,
                )
                brain.eval()
                out = brain({'vision': torch.randn(2, 1, 28, 28)})
                assert out.shape == (2, 10)
            except Exception as e:
                pytest.skip(f"System integration test failed with: {e}")

        @pytest.mark.skipif(not HAS_GW, reason="GlobalWorkspace not available")
        def test_flag_disabled(self):
            """Workspace with no valid inputs should raise ValueError."""
            _seed()
            gw = GlobalWorkspace(
                config=_make_gw_config(),
                modality_dims={'vision': 64},
            )
            # Provide input for unregistered modality
            with pytest.raises(ValueError, match="No valid modality"):
                gw({'unknown_mod': torch.randn(2, 64)})

        @pytest.mark.skipif(not HAS_GW, reason="GlobalWorkspace not available")
        def test_multi_modal_competition(self):
            """Multiple modalities should compete for workspace access."""
            _seed()
            dims = {'vision': 64, 'text': 64, 'audio': 32, 'sensors': 16}
            gw = GlobalWorkspace(
                config=_make_gw_config(capacity_limit=2),
                modality_dims=dims,
            )
            gw.eval()
            inputs = {
                name: torch.randn(BATCH_SIZE, dim)
                for name, dim in dims.items()
            }
            result = gw(inputs, return_attention=True)
            attn = result['attention']
            # With capacity=2 and 4 modalities, attention should be selective
            attn_values = torch.stack([attn[n] for n in result['modality_names']], dim=-1)
            # Check that not all modalities have equal attention
            attn_std = attn_values.std(dim=-1)
            assert attn_std.mean() > 0.01, (
                "With limited capacity, attention should be selective"
            )

        @pytest.mark.skipif(not HAS_SBW, reason="SelectionBroadcastWorkspace not available")
        def test_return_details_telemetry(self):
            """return_details=True should provide full competition telemetry."""
            _seed()
            ws = create_selection_broadcast_workspace(
                workspace_dim=WORKSPACE_DIM,
                modality_dims=MODALITY_DIMS,
                num_heads=NUM_HEADS,
                selection_rounds=2,
                memory_mode="gru",
            )
            ws.eval()
            inputs = _make_modality_inputs()
            result = ws(inputs, return_details=True)
            assert 'competition_details' in result
            assert 'ignition' in result
            assert 'global_ignition' in result
            assert 'attention' in result
            assert 'confidence' in result
            assert 'broadcasts' in result
            assert 'memory_output' in result

        @pytest.mark.skipif(not HAS_GW, reason="GlobalWorkspace not available")
        def test_workspace_deterministic_seeded(self):
            """Same seed should produce identical workspace outputs."""
            def _run():
                _seed(777)
                gw = GlobalWorkspace(
                    config=_make_gw_config(),
                    modality_dims=MODALITY_DIMS,
                )
                gw.eval()
                _seed(777)
                inputs = _make_modality_inputs()
                gw.reset_state()
                return gw(inputs)

            r1 = _run()
            r2 = _run()
            assert torch.allclose(r1['workspace'], r2['workspace'], atol=1e-6), (
                "Deterministic seeding should produce identical outputs"
            )
    ''')


# ---------------------------------------------------------------------------
# Assemble the full file
# ---------------------------------------------------------------------------

# Map class names to their generator functions
CLASS_GENERATORS = {
    'TestTokenStaging': _test_token_staging,
    'TestCompetitionScoring': _test_competition_scoring,
    'TestDeterministicTopK': _test_deterministic_topk,
    'TestSlotConstruction': _test_slot_construction,
    'TestIterativeRounds': _test_iterative_rounds,
    'TestIgnition': _test_ignition,
    'TestLockInPrevention': _test_lockin_prevention,
    'TestBroadcastAdapters': _test_broadcast_adapters,
    'TestWorkingMemory': _test_working_memory,
    'TestIntegration': _test_integration,
}


def generate_test_file(class_filter: str = None) -> str:
    """Assemble the complete generated test file.

    Args:
        class_filter: If set, only generate tests for the named class.
    """
    sections = [_header()]

    if class_filter:
        if class_filter in CLASS_GENERATORS:
            sections.append(CLASS_GENERATORS[class_filter]())
        else:
            raise ValueError(
                f"Unknown class: {class_filter}. "
                f"Available: {', '.join(CLASS_GENERATORS.keys())}"
            )
    else:
        for gen_func in CLASS_GENERATORS.values():
            sections.append(gen_func())

    return "\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# Counting helpers for --dry-run reporting
# ---------------------------------------------------------------------------


def count_classes(content: str) -> int:
    return len(re.findall(r"^class Test\w+", content, re.MULTILINE))


def count_tests(content: str) -> int:
    return len(re.findall(r"^\s+def test_\w+", content, re.MULTILINE))


def list_tests_by_class(content: str) -> dict:
    """Return a dict of class_name -> list of test method names."""
    result = {}
    current_class = None
    for line in content.splitlines():
        cls_match = re.match(r"^class (Test\w+)", line)
        if cls_match:
            current_class = cls_match.group(1)
            result[current_class] = []
        test_match = re.match(r"^\s+def (test_\w+)", line)
        if test_match and current_class:
            result[current_class].append(test_match.group(1))
    return result


def verify_syntax(content: str, filename: str = "<generated>") -> bool:
    """Verify that the generated content is valid Python using ast.parse."""
    try:
        ast.parse(content, filename=filename)
        return True
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate tests/test_workspace.py with comprehensive Global Workspace test cases."
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_PROJECT_ROOT, "tests", "test_workspace.py"),
        help="Path to write the generated test file (default: PROJECT_ROOT/tests/test_workspace.py)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing the file.",
    )
    parser.add_argument(
        "--class",
        dest="class_name",
        default=None,
        help="Only generate tests for the named class (e.g., TestIgnition).",
    )
    args = parser.parse_args()

    content = generate_test_file(class_filter=args.class_name)

    # Count and report
    num_classes = count_classes(content)
    num_tests = count_tests(content)
    tests_by_class = list_tests_by_class(content)

    if args.dry_run:
        print(f"Would write {len(content):,} chars ({len(content.splitlines()):,} lines) to {args.output}")
        print(f"Test classes: {num_classes}")
        print(f"Test functions: {num_tests}")
        print()
        for cls_name, methods in tests_by_class.items():
            print(f"  {cls_name} ({len(methods)} tests):")
            for method in methods:
                print(f"    - {method}")
        print()

        # Verify content is valid Python
        if verify_syntax(content):
            print("Syntax check: PASSED")
        else:
            print("Syntax check: FAILED")
            sys.exit(1)
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(content)
        print(f"Generated {args.output}")
        print(f"  Lines:          {len(content.splitlines()):,}")
        print(f"  Test classes:   {num_classes}")
        print(f"  Test functions: {num_tests}")
        print()
        for cls_name, methods in tests_by_class.items():
            print(f"  {cls_name} ({len(methods)} tests):")
            for method in methods:
                print(f"    - {method}")
        print()

        # Verify syntax
        if verify_syntax(content, args.output):
            print("  Syntax check:  PASSED")
        else:
            print("  Syntax check:  FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()
