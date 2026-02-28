"""
Tests for Dual Process Reasoning Module

Validates the dual-process reasoning system including:
    - DualProcessReasoner creation and configuration
    - Forward pass with input tensor
    - Confidence output range and shape
    - Reasoning trace from System 2
    - System 1 fast path
    - System 2 deliberation trigger
    - Metacognition module integration

Run:
    pytest tests/test_reasoning.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys

# Path fixup so tests can run from repo root
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from brain_ai.reasoning.system2 import (
    DualProcessReasoner,
    System2Config,
    create_dual_process_reasoner,
    System1Module,
    System2Module,
    MetacognitionModule,
)


# ---------------------------------------------------------------------------
# Constants -- keep small for fast execution
# ---------------------------------------------------------------------------
HIDDEN_DIM = 64
BATCH_SIZE = 4
SEED = 42


def _seed(s: int = SEED):
    torch.manual_seed(s)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Small System2Config for fast tests."""
    return System2Config(
        hidden_dim=HIDDEN_DIM,
        confidence_threshold=0.7,
        max_iterations=3,
        min_iterations=1,
        system1_hidden=32,
        system1_layers=1,
        num_reasoning_steps=2,
        reasoning_hidden=64,
        use_metacognition=True,
        meta_hidden=32,
    )


@pytest.fixture
def reasoner(config):
    """Create a DualProcessReasoner with small dims."""
    _seed()
    r = DualProcessReasoner(config=config)
    r.train(False)
    return r


@pytest.fixture
def input_tensor():
    """Standard input tensor for tests."""
    _seed()
    return torch.randn(BATCH_SIZE, HIDDEN_DIM)


@pytest.fixture
def system1():
    """Standalone System1Module."""
    _seed()
    return System1Module(
        input_dim=HIDDEN_DIM,
        output_dim=HIDDEN_DIM,
        hidden_dim=32,
        num_layers=1,
    )


@pytest.fixture
def system2():
    """Standalone System2Module."""
    _seed()
    return System2Module(
        input_dim=HIDDEN_DIM,
        output_dim=HIDDEN_DIM,
        hidden_dim=64,
        num_reasoning_steps=2,
        use_symbolic=False,  # Avoid symbolic dependency
    )


@pytest.fixture
def metacognition():
    """Standalone MetacognitionModule."""
    _seed()
    return MetacognitionModule(
        hidden_dim=HIDDEN_DIM,
        meta_hidden=32,
    )


# ---------------------------------------------------------------------------
# Tests: Creation
# ---------------------------------------------------------------------------

class TestDualProcessCreation:
    """Tests for DualProcessReasoner construction."""

    def test_create_with_config(self, config):
        """DualProcessReasoner should accept a config object."""
        r = DualProcessReasoner(config=config)
        assert r.config.hidden_dim == HIDDEN_DIM
        assert r.config.confidence_threshold == 0.7

    def test_create_with_factory(self):
        """create_dual_process_reasoner factory should work."""
        r = create_dual_process_reasoner(
            hidden_dim=HIDDEN_DIM,
            confidence_threshold=0.7,
            use_metacognition=True,
        )
        assert isinstance(r, DualProcessReasoner)
        assert r.config.hidden_dim == HIDDEN_DIM

    def test_create_without_metacognition(self):
        """Reasoner should work without metacognition module."""
        r = create_dual_process_reasoner(
            hidden_dim=HIDDEN_DIM,
            use_metacognition=False,
        )
        assert r.metacognition is None

    def test_create_with_metacognition(self, reasoner):
        """Reasoner should have metacognition when configured."""
        assert reasoner.metacognition is not None
        assert isinstance(reasoner.metacognition, MetacognitionModule)

    def test_has_system1(self, reasoner):
        """Reasoner should have a System1Module."""
        assert hasattr(reasoner, "system1")
        assert isinstance(reasoner.system1, System1Module)

    def test_has_system2(self, reasoner):
        """Reasoner should have a System2Module."""
        assert hasattr(reasoner, "system2")
        assert isinstance(reasoner.system2, System2Module)

    def test_has_integration(self, reasoner):
        """Reasoner should have an integration layer."""
        assert hasattr(reasoner, "integration")


# ---------------------------------------------------------------------------
# Tests: Forward pass
# ---------------------------------------------------------------------------

class TestForwardPass:
    """Tests for DualProcessReasoner forward pass."""

    def test_forward_output_keys(self, reasoner, input_tensor):
        """Forward output should contain required keys."""
        with torch.no_grad():
            output = reasoner(input_tensor)

        required_keys = ["output", "confidence", "system_used", "sys1_output", "sys2_output"]
        for key in required_keys:
            assert key in output, f"Missing key '{key}' in forward output"

    def test_output_shape(self, reasoner, input_tensor):
        """Output tensor should have shape (batch, hidden_dim)."""
        with torch.no_grad():
            output = reasoner(input_tensor)

        assert output["output"].shape == (BATCH_SIZE, HIDDEN_DIM), (
            f"Expected ({BATCH_SIZE}, {HIDDEN_DIM}), "
            f"got {output['output'].shape}"
        )

    def test_output_finite(self, reasoner, input_tensor):
        """Output should contain no NaN or Inf values."""
        with torch.no_grad():
            output = reasoner(input_tensor)

        assert not torch.isnan(output["output"]).any(), "NaN in output"
        assert not torch.isinf(output["output"]).any(), "Inf in output"

    def test_sys1_output_shape(self, reasoner, input_tensor):
        """System 1 output should match hidden_dim."""
        with torch.no_grad():
            output = reasoner(input_tensor)

        assert output["sys1_output"].shape == (BATCH_SIZE, HIDDEN_DIM)

    def test_sys2_output_shape(self, reasoner, input_tensor):
        """System 2 output should match hidden_dim (may be zeros if not used)."""
        with torch.no_grad():
            output = reasoner(input_tensor)

        assert output["sys2_output"].shape == (BATCH_SIZE, HIDDEN_DIM)

    def test_system_used_shape(self, reasoner, input_tensor):
        """system_used should have shape (batch,)."""
        with torch.no_grad():
            output = reasoner(input_tensor)

        assert output["system_used"].shape == (BATCH_SIZE,)

    def test_system_used_binary(self, reasoner, input_tensor):
        """system_used should be binary (0 for sys1, 1 for sys2)."""
        with torch.no_grad():
            output = reasoner(input_tensor)

        su = output["system_used"]
        assert torch.all((su == 0) | (su == 1)), (
            f"system_used not binary: {su}"
        )


# ---------------------------------------------------------------------------
# Tests: Confidence output
# ---------------------------------------------------------------------------

class TestConfidence:
    """Tests for confidence estimation."""

    def test_confidence_shape(self, reasoner, input_tensor):
        """Confidence should have shape (batch, 1)."""
        with torch.no_grad():
            output = reasoner(input_tensor)

        assert output["confidence"].shape == (BATCH_SIZE, 1), (
            f"Confidence shape {output['confidence'].shape}, "
            f"expected ({BATCH_SIZE}, 1)"
        )

    def test_confidence_range(self, reasoner, input_tensor):
        """Confidence should be in [0, 1] (sigmoid output)."""
        with torch.no_grad():
            output = reasoner(input_tensor)

        conf = output["confidence"]
        assert (conf >= 0).all(), f"Confidence below 0: {conf.min().item()}"
        assert (conf <= 1).all(), f"Confidence above 1: {conf.max().item()}"

    def test_confidence_finite(self, reasoner, input_tensor):
        """Confidence should be finite."""
        with torch.no_grad():
            output = reasoner(input_tensor)

        assert not torch.isnan(output["confidence"]).any()

    def test_return_details_includes_confidence_breakdown(self, reasoner, input_tensor):
        """return_details should include per-system confidences."""
        with torch.no_grad():
            output = reasoner(input_tensor, return_details=True)

        assert "sys1_confidence" in output
        assert "sys2_confidence" in output
        assert "system_probs" in output

    def test_sys1_confidence_range(self, reasoner, input_tensor):
        """System 1 confidence should be in [0, 1]."""
        with torch.no_grad():
            output = reasoner(input_tensor, return_details=True)

        c = output["sys1_confidence"]
        assert (c >= 0).all() and (c <= 1).all()


# ---------------------------------------------------------------------------
# Tests: Reasoning trace
# ---------------------------------------------------------------------------

class TestReasoningTrace:
    """Tests for System 2 reasoning trace."""

    def test_system2_returns_trace(self, system2, input_tensor):
        """System 2 should return a trace when requested."""
        with torch.no_grad():
            result = system2(input_tensor, return_trace=True)

        assert "trace" in result
        assert "confidence_trace" in result

    def test_trace_shape(self, system2, input_tensor):
        """Trace should have shape (batch, num_steps+1, hidden_dim)."""
        with torch.no_grad():
            result = system2(input_tensor, return_trace=True)

        trace = result["trace"]
        # Initial state + reasoning steps
        assert trace.dim() == 3
        assert trace.shape[0] == BATCH_SIZE
        assert trace.shape[2] == 64  # reasoning hidden dim

    def test_confidence_trace_shape(self, system2, input_tensor):
        """Confidence trace should have shape (batch, num_steps, 1)."""
        with torch.no_grad():
            result = system2(input_tensor, return_trace=True)

        ct = result["confidence_trace"]
        assert ct.dim() == 3
        assert ct.shape[0] == BATCH_SIZE
        assert ct.shape[2] == 1

    def test_num_iterations_reported(self, system2, input_tensor):
        """System 2 should report how many iterations were used."""
        with torch.no_grad():
            result = system2(input_tensor)

        assert "num_iterations" in result
        n_iter = result["num_iterations"].item()
        assert 1 <= n_iter <= 2  # max_iterations=2 in fixture

    def test_trace_not_returned_by_default(self, system2, input_tensor):
        """Without return_trace, trace should not be in output."""
        with torch.no_grad():
            result = system2(input_tensor, return_trace=False)

        assert "trace" not in result


# ---------------------------------------------------------------------------
# Tests: System 1 fast path
# ---------------------------------------------------------------------------

class TestSystem1FastPath:
    """Tests for System 1 fast processing."""

    def test_system1_output_shape(self, system1, input_tensor):
        """System 1 should produce (output, confidence) tuple."""
        with torch.no_grad():
            output, confidence = system1(input_tensor)

        assert output.shape == (BATCH_SIZE, HIDDEN_DIM)
        assert confidence.shape == (BATCH_SIZE, 1)

    def test_system1_confidence_sigmoid(self, system1, input_tensor):
        """System 1 confidence should be in [0, 1]."""
        with torch.no_grad():
            _, confidence = system1(input_tensor)

        assert (confidence >= 0).all()
        assert (confidence <= 1).all()

    def test_force_system1(self, reasoner, input_tensor):
        """force_system=1 should use only System 1."""
        with torch.no_grad():
            output = reasoner(input_tensor, force_system=1)

        # system_used should all be 0 (System 1)
        assert (output["system_used"] == 0).all(), (
            "force_system=1 should result in system_used=0 for all samples"
        )

    def test_system1_fast(self, system1, input_tensor):
        """System 1 should produce finite output."""
        with torch.no_grad():
            output, confidence = system1(input_tensor)

        assert not torch.isnan(output).any()
        assert not torch.isnan(confidence).any()


# ---------------------------------------------------------------------------
# Tests: System 2 deliberation
# ---------------------------------------------------------------------------

class TestSystem2Deliberation:
    """Tests for System 2 deliberate reasoning."""

    def test_system2_output_keys(self, system2, input_tensor):
        """System 2 output should contain required keys."""
        with torch.no_grad():
            result = system2(input_tensor)

        assert "output" in result
        assert "confidence" in result
        assert "num_iterations" in result
        assert "state" in result

    def test_system2_output_shape(self, system2, input_tensor):
        """System 2 output should match hidden_dim."""
        with torch.no_grad():
            result = system2(input_tensor)

        assert result["output"].shape == (BATCH_SIZE, HIDDEN_DIM)

    def test_system2_state_shape(self, system2, input_tensor):
        """System 2 internal state should match reasoning_hidden."""
        with torch.no_grad():
            result = system2(input_tensor)

        assert result["state"].shape == (BATCH_SIZE, 64)  # reasoning_hidden

    def test_force_system2(self, reasoner, input_tensor):
        """force_system=2 should use System 2 for all samples."""
        with torch.no_grad():
            output = reasoner(input_tensor, force_system=2)

        # system_used should all be 1 (System 2)
        assert (output["system_used"] == 1).all(), (
            "force_system=2 should result in system_used=1 for all samples"
        )

    def test_system2_max_iterations_override(self, system2, input_tensor):
        """max_iterations override should be respected."""
        with torch.no_grad():
            result = system2(input_tensor, max_iterations=1)

        assert result["num_iterations"].item() <= 1

    def test_system2_finite_output(self, system2, input_tensor):
        """System 2 should produce finite outputs."""
        with torch.no_grad():
            result = system2(input_tensor)

        assert not torch.isnan(result["output"]).any()
        assert not torch.isnan(result["confidence"]).any()


# ---------------------------------------------------------------------------
# Tests: Metacognition
# ---------------------------------------------------------------------------

class TestMetacognition:
    """Tests for MetacognitionModule."""

    def test_metacognition_output_keys(self, metacognition, input_tensor):
        """Metacognition should output uncertainty, novelty, effort, system_probs."""
        with torch.no_grad():
            result = metacognition(input_tensor)

        assert "uncertainty" in result
        assert "novelty" in result
        assert "effort" in result
        assert "system_probs" in result

    def test_uncertainty_range(self, metacognition, input_tensor):
        """Uncertainty should be in [0, 1]."""
        with torch.no_grad():
            result = metacognition(input_tensor)

        u = result["uncertainty"]
        assert (u >= 0).all() and (u <= 1).all()

    def test_novelty_range(self, metacognition, input_tensor):
        """Novelty should be in [0, 1]."""
        with torch.no_grad():
            result = metacognition(input_tensor)

        n = result["novelty"]
        assert (n >= 0).all() and (n <= 1).all()

    def test_effort_range(self, metacognition, input_tensor):
        """Effort should be in [0, 1]."""
        with torch.no_grad():
            result = metacognition(input_tensor)

        e = result["effort"]
        assert (e >= 0).all() and (e <= 1).all()

    def test_system_probs_sum_to_one(self, metacognition, input_tensor):
        """System probabilities should sum to 1 (softmax)."""
        with torch.no_grad():
            result = metacognition(input_tensor)

        sp = result["system_probs"]
        sums = sp.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            f"System probs do not sum to 1: {sums}"
        )

    def test_system_probs_shape(self, metacognition, input_tensor):
        """System probs should have shape (batch, 2)."""
        with torch.no_grad():
            result = metacognition(input_tensor)

        assert result["system_probs"].shape == (BATCH_SIZE, 2)

    def test_metacognition_with_sys1_confidence(self, metacognition, input_tensor):
        """Metacognition should accept optional system1_confidence."""
        _seed()
        confidence = torch.rand(BATCH_SIZE, 1)
        with torch.no_grad():
            result = metacognition(input_tensor, system1_confidence=confidence)

        assert "system_probs" in result


# ---------------------------------------------------------------------------
# Tests: Parameterized
# ---------------------------------------------------------------------------

class TestParameterized:
    """Parameterized tests for edge cases."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_varying_batch_sizes(self, config, batch_size):
        """Reasoner should handle various batch sizes."""
        _seed()
        r = DualProcessReasoner(config=config)
        r.train(False)
        x = torch.randn(batch_size, HIDDEN_DIM)

        with torch.no_grad():
            output = r(x)

        assert output["output"].shape == (batch_size, HIDDEN_DIM)
        assert output["confidence"].shape == (batch_size, 1)

    @pytest.mark.parametrize("force_system", [1, 2])
    def test_force_system_produces_output(self, reasoner, input_tensor, force_system):
        """Both forced system modes should produce valid output."""
        with torch.no_grad():
            output = reasoner(input_tensor, force_system=force_system)

        assert output["output"].shape == (BATCH_SIZE, HIDDEN_DIM)
        assert not torch.isnan(output["output"]).any()

    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.9])
    def test_confidence_threshold_affects_routing(self, threshold):
        """Different confidence thresholds should affect system routing."""
        _seed()
        r = create_dual_process_reasoner(
            hidden_dim=HIDDEN_DIM,
            confidence_threshold=threshold,
            use_metacognition=False,
            reasoning_hidden=HIDDEN_DIM,
            system1_hidden=32,
            system1_layers=1,
            num_reasoning_steps=2,
        )
        r.train(False)
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM)

        with torch.no_grad():
            output = r(x)

        # Just verify it runs without error with different thresholds
        assert output["output"].shape == (BATCH_SIZE, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# Tests: Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Tests for gradient flow through the dual-process reasoner."""

    def test_gradient_flows_through_system1(self, config):
        """Gradients should flow through System 1."""
        _seed()
        r = DualProcessReasoner(config=config)
        r.train()

        x = torch.randn(BATCH_SIZE, HIDDEN_DIM)
        output = r(x, force_system=1)
        loss = output["output"].sum()
        loss.backward()

        # System 1 parameters should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in r.system1.parameters()
        )
        assert has_grad, "No gradients flowed through System 1"

    def test_gradient_flows_through_system2(self, config):
        """Gradients should flow through System 2."""
        _seed()
        r = DualProcessReasoner(config=config)
        r.train()

        x = torch.randn(BATCH_SIZE, HIDDEN_DIM)
        output = r(x, force_system=2)
        loss = output["output"].sum()
        loss.backward()

        # System 2 parameters should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in r.system2.parameters()
        )
        assert has_grad, "No gradients flowed through System 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
