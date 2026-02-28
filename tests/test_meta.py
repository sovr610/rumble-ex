"""
Tests for Neuromodulatory Gate Module

Validates the neuromodulation system including:
    - NeuromodulatoryGate creation and configuration
    - Forward pass with workspace tensor
    - Individual modulator outputs (DA, 5-HT, NE, ACh)
    - Anomaly score modulation of ACh
    - Confidence input modulation of NE
    - Learning rate multiplier and exploration bonus
    - Global gain computation

Run:
    pytest tests/test_meta.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys

# Path fixup so tests can run from repo root
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from brain_ai.meta.neuromodulation import (
    NeuromodulatoryGate,
    NeuromodulationConfig,
    create_neuromodulatory_gate,
    DopamineSystem,
    AcetylcholineSystem,
    NorepinephrineSystem,
    SerotoninSystem,
    ModulatorNetwork,
    PlasticityController,
)


# ---------------------------------------------------------------------------
# Constants -- keep small for fast execution
# ---------------------------------------------------------------------------
INPUT_DIM = 64
HIDDEN_DIM = 32
BATCH_SIZE = 4
SEED = 42


def _seed(s: int = SEED):
    torch.manual_seed(s)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Small NeuromodulationConfig for fast tests."""
    return NeuromodulationConfig(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_modulators=4,
        min_lr_multiplier=0.0,
        max_lr_multiplier=2.0,
        baseline_activity=0.5,
        adaptation_rate=0.1,
    )


@pytest.fixture
def gate(config):
    """Create a NeuromodulatoryGate with small dims."""
    _seed()
    g = NeuromodulatoryGate(config=config)
    g.train(False)
    return g


@pytest.fixture
def workspace_tensor():
    """Standard workspace-like input tensor."""
    _seed()
    return torch.randn(BATCH_SIZE, INPUT_DIM)


@pytest.fixture
def anomaly_score():
    """Anomaly score tensor in [0, 1]."""
    _seed()
    return torch.rand(BATCH_SIZE, 1)


@pytest.fixture
def confidence():
    """Confidence tensor in [0, 1]."""
    _seed()
    return torch.rand(BATCH_SIZE, 1)


@pytest.fixture
def prediction_error():
    """Prediction error tensor."""
    _seed()
    return torch.randn(BATCH_SIZE, 1)


# ---------------------------------------------------------------------------
# Tests: Creation
# ---------------------------------------------------------------------------

class TestNeuromodulatoryGateCreation:
    """Tests for NeuromodulatoryGate construction."""

    def test_create_with_config(self, config):
        """NeuromodulatoryGate should accept a config object."""
        g = NeuromodulatoryGate(config=config)
        assert g.config.input_dim == INPUT_DIM
        assert g.config.hidden_dim == HIDDEN_DIM

    def test_create_with_factory(self):
        """create_neuromodulatory_gate factory should work."""
        g = create_neuromodulatory_gate(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
        )
        assert isinstance(g, NeuromodulatoryGate)
        assert g.config.input_dim == INPUT_DIM

    def test_create_with_kwargs(self):
        """NeuromodulatoryGate should accept keyword arguments."""
        g = NeuromodulatoryGate(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
        )
        assert g.config.input_dim == INPUT_DIM

    def test_has_dopamine(self, gate):
        """Gate should have a dopamine system."""
        assert hasattr(gate, "dopamine")
        assert isinstance(gate.dopamine, DopamineSystem)

    def test_has_acetylcholine(self, gate):
        """Gate should have an acetylcholine system."""
        assert hasattr(gate, "acetylcholine")
        assert isinstance(gate.acetylcholine, AcetylcholineSystem)

    def test_has_norepinephrine(self, gate):
        """Gate should have a norepinephrine system."""
        assert hasattr(gate, "norepinephrine")
        assert isinstance(gate.norepinephrine, NorepinephrineSystem)

    def test_has_serotonin(self, gate):
        """Gate should have a serotonin system."""
        assert hasattr(gate, "serotonin")
        assert isinstance(gate.serotonin, SerotoninSystem)

    def test_has_integration(self, gate):
        """Gate should have an integration network."""
        assert hasattr(gate, "integration")

    def test_has_exploration_bonus(self, gate):
        """Gate should have an exploration bonus network."""
        assert hasattr(gate, "exploration_bonus")


# ---------------------------------------------------------------------------
# Tests: Forward pass
# ---------------------------------------------------------------------------

class TestForwardPass:
    """Tests for NeuromodulatoryGate forward pass."""

    def test_forward_output_keys(self, gate, workspace_tensor):
        """Forward output should contain required keys."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        assert "lr_multiplier" in output
        assert "modulators" in output
        assert "exploration_bonus" in output
        assert "global_gain" in output

    def test_forward_with_no_optional_inputs(self, gate, workspace_tensor):
        """Forward should work with only the required input tensor."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        assert output["lr_multiplier"].shape == (BATCH_SIZE,)

    def test_forward_with_all_optional_inputs(
        self, gate, workspace_tensor, anomaly_score, confidence, prediction_error
    ):
        """Forward should work with all optional inputs provided."""
        _seed()
        reward = torch.randn(BATCH_SIZE)
        with torch.no_grad():
            output = gate(
                workspace_tensor,
                anomaly_score=anomaly_score,
                confidence=confidence,
                prediction_error=prediction_error,
                reward=reward,
            )

        assert output["lr_multiplier"].shape == (BATCH_SIZE,)

    def test_output_finite(self, gate, workspace_tensor):
        """All outputs should be finite."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        assert not torch.isnan(output["lr_multiplier"]).any(), "NaN in lr_multiplier"
        assert not torch.isinf(output["lr_multiplier"]).any(), "Inf in lr_multiplier"
        assert not torch.isnan(output["exploration_bonus"]).any(), "NaN in exploration_bonus"
        assert not torch.isnan(output["global_gain"]).any(), "NaN in global_gain"


# ---------------------------------------------------------------------------
# Tests: Individual modulator outputs (DA, 5-HT, NE, ACh)
# ---------------------------------------------------------------------------

class TestModulatorOutputs:
    """Tests for individual modulator activities."""

    def test_modulators_dict_keys(self, gate, workspace_tensor):
        """Modulators dict should contain all four neurotransmitters."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        mods = output["modulators"]
        assert "dopamine" in mods
        assert "acetylcholine" in mods
        assert "norepinephrine" in mods
        assert "serotonin" in mods

    def test_dopamine_shape(self, gate, workspace_tensor):
        """Dopamine activity should have shape (batch,)."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        da = output["modulators"]["dopamine"]
        assert da.shape == (BATCH_SIZE,), f"DA shape {da.shape}"

    def test_acetylcholine_shape(self, gate, workspace_tensor):
        """Acetylcholine activity should have shape (batch,)."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        ach = output["modulators"]["acetylcholine"]
        assert ach.shape == (BATCH_SIZE,), f"ACh shape {ach.shape}"

    def test_norepinephrine_shape(self, gate, workspace_tensor):
        """Norepinephrine activity should have shape (batch,)."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        ne = output["modulators"]["norepinephrine"]
        assert ne.shape == (BATCH_SIZE,), f"NE shape {ne.shape}"

    def test_serotonin_shape(self, gate, workspace_tensor):
        """Serotonin activity should have shape (batch,)."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        sht = output["modulators"]["serotonin"]
        assert sht.shape == (BATCH_SIZE,), f"5-HT shape {sht.shape}"

    def test_modulator_activities_finite(self, gate, workspace_tensor):
        """All modulator activities should be finite."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        for name, activity in output["modulators"].items():
            assert not torch.isnan(activity).any(), f"NaN in {name}"
            assert not torch.isinf(activity).any(), f"Inf in {name}"


# ---------------------------------------------------------------------------
# Tests: Individual modulator sub-modules
# ---------------------------------------------------------------------------

class TestDopamineSystem:
    """Tests for the DopamineSystem."""

    def test_dopamine_forward(self, workspace_tensor):
        """DopamineSystem should produce activity output."""
        _seed()
        da = DopamineSystem(INPUT_DIM, HIDDEN_DIM)
        da.train(False)
        with torch.no_grad():
            activity = da(workspace_tensor)
        assert activity.shape == (BATCH_SIZE,)

    def test_dopamine_rpe_computation(self, workspace_tensor):
        """DopamineSystem should compute reward prediction error."""
        _seed()
        da = DopamineSystem(INPUT_DIM, HIDDEN_DIM)
        da.train(False)
        reward = torch.randn(BATCH_SIZE)
        with torch.no_grad():
            rpe, predicted_value = da.compute_rpe(workspace_tensor, reward=reward)

        assert rpe.shape == (BATCH_SIZE,)
        assert predicted_value.shape == (BATCH_SIZE,)

    def test_dopamine_rpe_without_reward(self, workspace_tensor):
        """RPE without explicit reward should return zeros."""
        _seed()
        da = DopamineSystem(INPUT_DIM, HIDDEN_DIM)
        da.train(False)
        with torch.no_grad():
            rpe, _ = da.compute_rpe(workspace_tensor, reward=None)

        assert torch.all(rpe == 0), "RPE without reward should be zero"


class TestAcetylcholineSystem:
    """Tests for the AcetylcholineSystem."""

    def test_acetylcholine_forward(self, workspace_tensor):
        """AcetylcholineSystem should produce activity output."""
        _seed()
        ach = AcetylcholineSystem(INPUT_DIM, HIDDEN_DIM)
        ach.train(False)
        with torch.no_grad():
            activity = ach(workspace_tensor)
        assert activity.shape == (BATCH_SIZE,)

    def test_acetylcholine_novelty(self, workspace_tensor):
        """AcetylcholineSystem should compute novelty signal."""
        _seed()
        ach = AcetylcholineSystem(INPUT_DIM, HIDDEN_DIM)
        ach.train(False)
        with torch.no_grad():
            novelty = ach.compute_novelty(workspace_tensor)

        assert novelty.shape == (BATCH_SIZE,)
        assert (novelty >= 0).all() and (novelty <= 1).all()


class TestNorepinephrineSystem:
    """Tests for the NorepinephrineSystem."""

    def test_norepinephrine_forward(self, workspace_tensor):
        """NorepinephrineSystem should produce activity output."""
        _seed()
        ne = NorepinephrineSystem(INPUT_DIM, HIDDEN_DIM)
        ne.train(False)
        with torch.no_grad():
            activity = ne(workspace_tensor)
        assert activity.shape == (BATCH_SIZE,)


class TestSerotoninSystem:
    """Tests for the SerotoninSystem."""

    def test_serotonin_forward(self, workspace_tensor):
        """SerotoninSystem should produce activity output."""
        _seed()
        sht = SerotoninSystem(INPUT_DIM, HIDDEN_DIM)
        sht.train(False)
        with torch.no_grad():
            activity = sht(workspace_tensor)
        assert activity.shape == (BATCH_SIZE,)


# ---------------------------------------------------------------------------
# Tests: Anomaly score input
# ---------------------------------------------------------------------------

class TestAnomalyScoreInput:
    """Tests for anomaly score modulation of ACh."""

    def test_anomaly_increases_ach(self, gate, workspace_tensor):
        """High anomaly score should increase acetylcholine activity."""
        with torch.no_grad():
            # No anomaly
            out_no_anomaly = gate(workspace_tensor)
            ach_baseline = out_no_anomaly["modulators"]["acetylcholine"].clone()

            # High anomaly
            high_anomaly = torch.ones(BATCH_SIZE, 1)
            out_high_anomaly = gate(workspace_tensor, anomaly_score=high_anomaly)
            ach_high = out_high_anomaly["modulators"]["acetylcholine"]

        # ACh with high anomaly should generally be >= baseline
        # (exact depends on the 0.3 * anomaly_score addition)
        assert (ach_high >= ach_baseline - 0.01).all(), (
            f"ACh did not increase with high anomaly. "
            f"Baseline mean: {ach_baseline.mean():.4f}, "
            f"High anomaly mean: {ach_high.mean():.4f}"
        )

    def test_anomaly_clamped_to_01(self, gate, workspace_tensor):
        """ACh should be clamped to [0, 1] even with extreme anomaly."""
        high_anomaly = torch.ones(BATCH_SIZE, 1) * 5.0  # Extreme value
        with torch.no_grad():
            output = gate(workspace_tensor, anomaly_score=high_anomaly)

        ach = output["modulators"]["acetylcholine"]
        assert (ach >= 0).all(), f"ACh below 0: {ach.min().item()}"
        assert (ach <= 1).all(), f"ACh above 1: {ach.max().item()}"

    def test_zero_anomaly_score(self, gate, workspace_tensor):
        """Zero anomaly should not modify ACh from baseline."""
        zero_anomaly = torch.zeros(BATCH_SIZE, 1)
        with torch.no_grad():
            out_zero = gate(workspace_tensor, anomaly_score=zero_anomaly)
            out_none = gate(workspace_tensor, anomaly_score=None)

        ach_zero = out_zero["modulators"]["acetylcholine"]
        ach_none = out_none["modulators"]["acetylcholine"]

        # With zero anomaly, the 0.3 * 0 = 0 addition should yield same result
        assert torch.allclose(ach_zero, ach_none, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Confidence input
# ---------------------------------------------------------------------------

class TestConfidenceInput:
    """Tests for confidence modulation of NE."""

    def test_low_confidence_increases_ne(self, gate, workspace_tensor):
        """Low confidence should increase norepinephrine (arousal)."""
        with torch.no_grad():
            # No confidence input
            out_none = gate(workspace_tensor)
            ne_baseline = out_none["modulators"]["norepinephrine"].clone()

            # Zero confidence (maximum uncertainty)
            zero_conf = torch.zeros(BATCH_SIZE, 1)
            out_low = gate(workspace_tensor, confidence=zero_conf)
            ne_low_conf = out_low["modulators"]["norepinephrine"]

        # Low confidence should increase NE
        assert (ne_low_conf >= ne_baseline - 0.01).all(), (
            f"NE did not increase with low confidence. "
            f"Baseline: {ne_baseline.mean():.4f}, Low conf: {ne_low_conf.mean():.4f}"
        )

    def test_high_confidence_no_ne_increase(self, gate, workspace_tensor):
        """High confidence should not increase NE."""
        with torch.no_grad():
            out_none = gate(workspace_tensor)
            ne_baseline = out_none["modulators"]["norepinephrine"].clone()

            high_conf = torch.ones(BATCH_SIZE, 1)
            out_high = gate(workspace_tensor, confidence=high_conf)
            ne_high_conf = out_high["modulators"]["norepinephrine"]

        # With confidence=1.0, the 0.2 * (1 - 1.0) = 0 term adds nothing
        assert torch.allclose(ne_high_conf, ne_baseline, atol=1e-5)

    def test_ne_clamped_to_01(self, gate, workspace_tensor):
        """NE should be clamped to [0, 1] even with extreme inputs."""
        extreme_conf = torch.zeros(BATCH_SIZE, 1) - 10.0  # Very negative
        with torch.no_grad():
            output = gate(workspace_tensor, confidence=extreme_conf)

        ne = output["modulators"]["norepinephrine"]
        assert (ne >= 0).all(), f"NE below 0: {ne.min().item()}"
        assert (ne <= 1).all(), f"NE above 1: {ne.max().item()}"


# ---------------------------------------------------------------------------
# Tests: Learning rate multiplier
# ---------------------------------------------------------------------------

class TestLearningRateMultiplier:
    """Tests for learning rate multiplier computation."""

    def test_lr_multiplier_shape(self, gate, workspace_tensor):
        """LR multiplier should have shape (batch,)."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        lr = output["lr_multiplier"]
        assert lr.shape == (BATCH_SIZE,)

    def test_lr_multiplier_range(self, gate, workspace_tensor):
        """LR multiplier should be in [min_lr, max_lr] range."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        lr = output["lr_multiplier"]
        assert (lr >= gate.config.min_lr_multiplier - 1e-5).all(), (
            f"LR below min: {lr.min().item()}"
        )
        assert (lr <= gate.config.max_lr_multiplier + 1e-5).all(), (
            f"LR above max: {lr.max().item()}"
        )

    def test_exploration_bonus_shape(self, gate, workspace_tensor):
        """Exploration bonus should have shape (batch,)."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        eb = output["exploration_bonus"]
        assert eb.shape == (BATCH_SIZE,)

    def test_exploration_bonus_non_negative(self, gate, workspace_tensor):
        """Exploration bonus should be non-negative (Softplus output)."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        eb = output["exploration_bonus"]
        assert (eb >= 0).all(), f"Negative exploration bonus: {eb.min().item()}"

    def test_global_gain_shape(self, gate, workspace_tensor):
        """Global gain should have shape (batch,)."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        gg = output["global_gain"]
        assert gg.shape == (BATCH_SIZE,)

    def test_global_gain_positive(self, gate, workspace_tensor):
        """Global gain should be positive (0.5 + NE)."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        gg = output["global_gain"]
        assert (gg > 0).all(), f"Non-positive global gain: {gg.min().item()}"


# ---------------------------------------------------------------------------
# Tests: Gradient modulation
# ---------------------------------------------------------------------------

class TestGradientModulation:
    """Tests for gradient modulation functionality."""

    def test_modulate_gradients(self, gate, workspace_tensor):
        """modulate_gradients should scale gradients by lr_multiplier."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        lr_mult = output["lr_multiplier"]

        # Create mock gradients
        gradients = {
            "weight1": torch.randn(64, 64),
            "weight2": torch.randn(32, 64),
            "none_grad": None,
        }

        modulated = gate.modulate_gradients(gradients, lr_mult)

        assert "weight1" in modulated
        assert "weight2" in modulated
        assert "none_grad" not in modulated  # None grads are skipped

    def test_modulated_gradient_shape(self, gate, workspace_tensor):
        """Modulated gradients should preserve original shape."""
        with torch.no_grad():
            output = gate(workspace_tensor)

        lr_mult = output["lr_multiplier"]
        grad = torch.randn(64, 64)
        modulated = gate.modulate_gradients({"w": grad}, lr_mult)

        assert modulated["w"].shape == grad.shape


# ---------------------------------------------------------------------------
# Tests: PlasticityController
# ---------------------------------------------------------------------------

class TestPlasticityController:
    """Tests for the PlasticityController."""

    def test_plasticity_controller_creation(self, config):
        """PlasticityController should create without error."""
        pc = PlasticityController(config=config)
        assert hasattr(pc, "gate")
        assert hasattr(pc, "mode_predictor")

    def test_plasticity_controller_forward(self, config, workspace_tensor):
        """PlasticityController should produce mode and lr outputs."""
        _seed()
        pc = PlasticityController(config=config)
        pc.train(False)

        with torch.no_grad():
            output = pc(workspace_tensor)

        assert "mode" in output
        assert "mode_probs" in output
        assert "lr_multiplier" in output
        assert "modulators" in output

    def test_mode_probs_sum_to_one(self, config, workspace_tensor):
        """Mode probabilities should sum to 1 (softmax)."""
        _seed()
        pc = PlasticityController(config=config)
        pc.train(False)

        with torch.no_grad():
            output = pc(workspace_tensor)

        mp = output["mode_probs"]
        sums = mp.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_mode_probs_shape(self, config, workspace_tensor):
        """Mode probs should have shape (batch, 3) for 3 modes."""
        _seed()
        pc = PlasticityController(config=config)
        pc.train(False)

        with torch.no_grad():
            output = pc(workspace_tensor)

        assert output["mode_probs"].shape == (BATCH_SIZE, 3)

    def test_mode_values(self, config, workspace_tensor):
        """Mode should be in {0, 1, 2} (static, online, few-shot)."""
        _seed()
        pc = PlasticityController(config=config)
        pc.train(False)

        with torch.no_grad():
            output = pc(workspace_tensor)

        mode = output["mode"]
        assert torch.all((mode >= 0) & (mode <= 2)), f"Invalid modes: {mode}"


# ---------------------------------------------------------------------------
# Tests: ModulatorNetwork base class
# ---------------------------------------------------------------------------

class TestModulatorNetwork:
    """Tests for the ModulatorNetwork base class."""

    def test_activity_history_initialized(self, workspace_tensor):
        """ModulatorNetwork should have a zero-initialized history buffer."""
        _seed()
        mod = ModulatorNetwork(INPUT_DIM, HIDDEN_DIM, baseline=0.5)
        assert mod.activity_history.shape == (100,)
        assert (mod.activity_history == 0).all()

    def test_update_history(self, workspace_tensor):
        """update_history should record activity mean."""
        _seed()
        mod = ModulatorNetwork(INPUT_DIM, HIDDEN_DIM, baseline=0.5)
        mod.train(False)
        with torch.no_grad():
            activity = mod(workspace_tensor)

        mod.update_history(activity)
        assert mod.history_idx.item() == 1
        assert mod.activity_history[0].item() != 0.0


# ---------------------------------------------------------------------------
# Tests: Parameterized
# ---------------------------------------------------------------------------

class TestParameterized:
    """Parameterized tests for various configurations."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_varying_batch_sizes(self, config, batch_size):
        """Gate should handle various batch sizes."""
        _seed()
        g = NeuromodulatoryGate(config=config)
        g.train(False)
        x = torch.randn(batch_size, INPUT_DIM)

        with torch.no_grad():
            output = g(x)

        assert output["lr_multiplier"].shape == (batch_size,)
        for name, activity in output["modulators"].items():
            assert activity.shape == (batch_size,), (
                f"{name} shape mismatch for batch_size={batch_size}"
            )

    @pytest.mark.parametrize("input_dim", [32, 64, 128])
    def test_varying_input_dims(self, input_dim):
        """Gate should handle various input dimensions."""
        _seed()
        g = create_neuromodulatory_gate(input_dim=input_dim, hidden_dim=HIDDEN_DIM)
        g.train(False)
        x = torch.randn(BATCH_SIZE, input_dim)

        with torch.no_grad():
            output = g(x)

        assert output["lr_multiplier"].shape == (BATCH_SIZE,)

    @pytest.mark.parametrize("anomaly_val", [0.0, 0.5, 1.0])
    def test_anomaly_score_values(self, gate, workspace_tensor, anomaly_val):
        """Gate should handle different anomaly score values."""
        anomaly = torch.full((BATCH_SIZE, 1), anomaly_val)
        with torch.no_grad():
            output = gate(workspace_tensor, anomaly_score=anomaly)

        assert output["lr_multiplier"].shape == (BATCH_SIZE,)
        assert not torch.isnan(output["lr_multiplier"]).any()


# ---------------------------------------------------------------------------
# Tests: Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Tests for gradient flow through the neuromodulatory gate."""

    def test_gradient_flows_through_gate(self, config):
        """Gradients should flow through the gate to modulator parameters."""
        _seed()
        g = NeuromodulatoryGate(config=config)
        g.train()

        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        output = g(x)
        loss = output["lr_multiplier"].sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in g.parameters()
        )
        assert has_grad, "No gradients flowed through neuromodulatory gate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
