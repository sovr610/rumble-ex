"""
Appraisal Module Template -- Scherer-Inspired Appraisal Computation.

Implements four appraisal checks (relevance, congruence, coping potential,
norm compatibility) that map internal cognitive signals to appraisal outputs.
These outputs feed into the ValenceArousalSpace to produce affective state.

Target file: ``brain_ai/affect/appraisal.py``

Key classes:
    - ``AppraisalModule``: Main appraisal computation (analytic or learned).
    - ``AppraisalResult``: Dataclass holding appraisal check outputs.

References:
    - Scherer, K. R. (2009). The dynamic architecture of emotion.
    - Scherer, K. R. (2001). Appraisal considered as a process of multilevel
      sequential checking.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class InternalSignals:
    """Internal cognitive signals that drive appraisal.

    All fields are (B,) tensors. These signals are produced by other modules
    in the cognitive pipeline and consumed by the AppraisalModule.

    Attributes:
        reward_prediction_error: RPE from DA system / active inference, (-inf, inf).
        epistemic_uncertainty: Transition model ensemble disagreement, [0, inf).
        prediction_error_magnitude: Sensory prediction error norm, [0, inf).
        novelty_score: HTM anomaly or embedding distance, [0, 1].
        homeostatic_deviation: Distance from homeostatic setpoints, [0, inf).
    """
    reward_prediction_error: Tensor
    epistemic_uncertainty: Tensor
    prediction_error_magnitude: Tensor
    novelty_score: Tensor
    homeostatic_deviation: Tensor

    def to(self, device: torch.device) -> InternalSignals:
        """Move all tensors to the specified device."""
        return InternalSignals(
            reward_prediction_error=self.reward_prediction_error.to(device),
            epistemic_uncertainty=self.epistemic_uncertainty.to(device),
            prediction_error_magnitude=self.prediction_error_magnitude.to(device),
            novelty_score=self.novelty_score.to(device),
            homeostatic_deviation=self.homeostatic_deviation.to(device),
        )

    def float(self) -> InternalSignals:
        """Cast all tensors to float32."""
        return InternalSignals(
            reward_prediction_error=self.reward_prediction_error.float(),
            epistemic_uncertainty=self.epistemic_uncertainty.float(),
            prediction_error_magnitude=self.prediction_error_magnitude.float(),
            novelty_score=self.novelty_score.float(),
            homeostatic_deviation=self.homeostatic_deviation.float(),
        )

    @classmethod
    def zeros(cls, batch_size: int, device: torch.device) -> InternalSignals:
        """Create zero signals (neutral input)."""
        z = torch.zeros(batch_size, device=device)
        return cls(
            reward_prediction_error=z.clone(),
            epistemic_uncertainty=z.clone(),
            prediction_error_magnitude=z.clone(),
            novelty_score=z.clone(),
            homeostatic_deviation=z.clone(),
        )


@dataclass
class AppraisalResult:
    """Output of the appraisal module.

    All fields are (B,) tensors with specified ranges.

    Attributes:
        relevance: How significant is the current situation? [0, 1].
        congruence: Does this match goals? [-1, 1]. Positive = congruent.
        coping_potential: Can the system handle this? [0, 1].
        norm_compatibility: Is this within expected bounds? [0, 1].
    """
    relevance: Tensor
    congruence: Tensor
    coping_potential: Tensor
    norm_compatibility: Tensor


# ============================================================================
# AppraisalModule
# ============================================================================

class AppraisalModule(nn.Module):
    """Computes appraisal checks from internal signals.

    Implements Scherer's component process model with four sequential checks:
    relevance, congruence, coping potential, and norm compatibility.

    Supports two modes:
    - Analytic (default): fixed sigmoid/tanh mappings
    - Learned: MLP replaces analytic mappings (when ``use_learned=True``)

    Args:
        relevance_scale: Sensitivity of relevance check.
        congruence_scale: Sensitivity of congruence check.
        coping_scale: Sensitivity of coping potential check.
        norm_scale: Sensitivity of norm compatibility check.
        relevance_threshold: Minimum relevance for full arousal response.
        use_learned: Use MLP for appraisal instead of analytic.
        hidden_dim: Hidden dimension for learned appraisal MLP.
        mood_congruence_bias: Strength of mood-congruent appraisal bias.
    """

    def __init__(
        self,
        relevance_scale: float = 1.0,
        congruence_scale: float = 1.0,
        coping_scale: float = 1.0,
        norm_scale: float = 0.5,
        relevance_threshold: float = 0.1,
        use_learned: bool = False,
        hidden_dim: int = 64,
        mood_congruence_bias: float = 0.1,
    ):
        super().__init__()
        self.relevance_scale = relevance_scale
        self.congruence_scale = congruence_scale
        self.coping_scale = coping_scale
        self.norm_scale = norm_scale
        self.relevance_threshold = relevance_threshold
        self.mood_congruence_bias = mood_congruence_bias

        if use_learned:
            self.learned_net = nn.Sequential(
                nn.Linear(5, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 4),
            )
        else:
            self.learned_net = None

    def forward(
        self,
        signals: InternalSignals,
        mood_valence: Optional[Tensor] = None,
    ) -> AppraisalResult:
        """Run all appraisal checks on the given signals.

        Args:
            signals: Internal cognitive signals.
            mood_valence: Current mood valence (B,) for mood-congruent bias.
                If None, no mood bias is applied.

        Returns:
            AppraisalResult with all four check outputs.
        """
        s = signals.float()

        if self.learned_net is not None:
            return self._learned_appraisal(s, mood_valence)

        relevance = self._compute_relevance(
            s.prediction_error_magnitude, s.novelty_score
        )
        congruence = self._compute_congruence(
            s.reward_prediction_error, mood_valence
        )
        coping_potential = self._compute_coping_potential(
            s.epistemic_uncertainty
        )
        norm_compatibility = self._compute_norm_compatibility(
            s.homeostatic_deviation
        )

        return AppraisalResult(
            relevance=relevance,
            congruence=congruence,
            coping_potential=coping_potential,
            norm_compatibility=norm_compatibility,
        )

    def _compute_relevance(
        self, pe_magnitude: Tensor, novelty: Tensor
    ) -> Tensor:
        """Relevance check: is this significant?

        Returns (B,) in [0, 1]. Zero inputs -> 0.5 from sigmoid, but we
        shift so zero inputs -> 0.0 for neutral behavior.
        """
        raw = pe_magnitude + novelty
        # 2*sigmoid(x) - 1 maps 0 -> 0, but range is [-1, 1]
        # We want [0, 1], so use sigmoid with offset
        # sigmoid(0) = 0.5, so subtract 0.5 and multiply by 2 for [0,1]
        # Alternative: use tanh and clamp to [0, 1]
        relevance = torch.tanh(self.relevance_scale * 0.5 * raw)
        return torch.clamp(relevance, 0.0, 1.0)

    def _compute_congruence(
        self, rpe: Tensor, mood_valence: Optional[Tensor] = None
    ) -> Tensor:
        """Congruence check: does this match goals?

        Returns (B,) in [-1, 1]. Positive RPE -> positive congruence.
        """
        biased_rpe = rpe
        if mood_valence is not None:
            biased_rpe = rpe + self.mood_congruence_bias * mood_valence
        return torch.tanh(self.congruence_scale * biased_rpe)

    def _compute_coping_potential(self, uncertainty: Tensor) -> Tensor:
        """Coping potential: can the system handle this?

        Returns (B,) in [0, 1]. Low uncertainty -> high coping.
        Zero uncertainty -> 0.5 (neutral coping).
        """
        # We want: zero uncertainty -> 0.5, high uncertainty -> 0.0
        # sigmoid(scale * (1 - uncertainty)) with appropriate centering
        # At uncertainty=0: sigmoid(scale * 1) ~ 0.73 for scale=1
        # Use a shifted version: sigmoid(scale * -uncertainty) for cleaner 0-mapping
        # sigmoid(0) = 0.5 when uncertainty=0
        return torch.sigmoid(-self.coping_scale * uncertainty)

    def _compute_norm_compatibility(self, homeostatic_deviation: Tensor) -> Tensor:
        """Norm compatibility: is this within expected bounds?

        Returns (B,) in [0, 1]. Zero deviation -> 0.5.
        Large deviation -> 0.
        """
        return torch.sigmoid(-self.norm_scale * homeostatic_deviation)

    def _learned_appraisal(
        self, signals: InternalSignals, mood_valence: Optional[Tensor]
    ) -> AppraisalResult:
        """Learned appraisal via MLP."""
        x = torch.stack([
            signals.reward_prediction_error,
            signals.epistemic_uncertainty,
            signals.prediction_error_magnitude,
            signals.novelty_score,
            signals.homeostatic_deviation,
        ], dim=-1)  # (B, 5)

        if mood_valence is not None:
            # Add mood bias to RPE channel
            x[:, 0] = x[:, 0] + self.mood_congruence_bias * mood_valence

        out = self.learned_net(x)  # (B, 4)

        return AppraisalResult(
            relevance=torch.sigmoid(out[:, 0]),
            congruence=torch.tanh(out[:, 1]),
            coping_potential=torch.sigmoid(out[:, 2]),
            norm_compatibility=torch.sigmoid(out[:, 3]),
        )


# ============================================================================
# Self-test
# ============================================================================

def _run_self_tests() -> None:
    """Self-tests for AppraisalModule."""
    passed = 0
    failed = 0
    errors_log: List[str] = []

    def _test(name: str, fn) -> None:
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            print(f"  PASS: {name}")
        except Exception as exc:
            failed += 1
            errors_log.append(f"  FAIL: {name} -- {exc}")
            print(f"  FAIL: {name} -- {exc}")

    print("=" * 72)
    print("AppraisalModule -- Self-Tests")
    print("=" * 72)

    device = torch.device("cpu")
    B = 4
    module = AppraisalModule()
    module.train(False)

    # Test 1: Zero signals -> neutral appraisal
    def test_neutral():
        signals = InternalSignals.zeros(B, device)
        result = module(signals)
        # Zero PE and novelty -> zero relevance
        assert (result.relevance >= 0).all()
        assert (result.relevance <= 1).all()
        # Zero RPE -> zero congruence
        assert torch.allclose(result.congruence, torch.zeros(B), atol=1e-6)
        # Zero uncertainty -> coping = sigmoid(0) = 0.5
        assert torch.allclose(
            result.coping_potential, torch.full((B,), 0.5), atol=1e-6
        )
    _test("Zero signals -> neutral appraisal", test_neutral)

    # Test 2: Relevance increases with PE magnitude
    def test_relevance_monotonic():
        s_low = InternalSignals.zeros(B, device)
        s_low.prediction_error_magnitude = torch.full((B,), 0.5)
        s_high = InternalSignals.zeros(B, device)
        s_high.prediction_error_magnitude = torch.full((B,), 5.0)
        r_low = module(s_low).relevance
        r_high = module(s_high).relevance
        assert (r_high > r_low).all(), f"low={r_low}, high={r_high}"
    _test("Relevance increases with PE magnitude", test_relevance_monotonic)

    # Test 3: Positive RPE -> positive congruence
    def test_congruence_positive():
        s = InternalSignals.zeros(B, device)
        s.reward_prediction_error = torch.full((B,), 2.0)
        result = module(s)
        assert (result.congruence > 0).all()
    _test("Positive RPE -> positive congruence", test_congruence_positive)

    # Test 4: Negative RPE -> negative congruence
    def test_congruence_negative():
        s = InternalSignals.zeros(B, device)
        s.reward_prediction_error = torch.full((B,), -2.0)
        result = module(s)
        assert (result.congruence < 0).all()
    _test("Negative RPE -> negative congruence", test_congruence_negative)

    # Test 5: High uncertainty -> low coping
    def test_coping_decreases():
        s_low = InternalSignals.zeros(B, device)
        s_low.epistemic_uncertainty = torch.full((B,), 0.1)
        s_high = InternalSignals.zeros(B, device)
        s_high.epistemic_uncertainty = torch.full((B,), 5.0)
        c_low = module(s_low).coping_potential
        c_high = module(s_high).coping_potential
        assert (c_low > c_high).all(), f"low_unc={c_low}, high_unc={c_high}"
    _test("High uncertainty -> lower coping", test_coping_decreases)

    # Test 6: High homeostatic deviation -> low norm compatibility
    def test_norm_decreases():
        s_low = InternalSignals.zeros(B, device)
        s_low.homeostatic_deviation = torch.full((B,), 0.1)
        s_high = InternalSignals.zeros(B, device)
        s_high.homeostatic_deviation = torch.full((B,), 5.0)
        n_low = module(s_low).norm_compatibility
        n_high = module(s_high).norm_compatibility
        assert (n_low > n_high).all(), f"low_dev={n_low}, high_dev={n_high}"
    _test("High deviation -> lower norm compatibility", test_norm_decreases)

    # Test 7: All outputs bounded
    def test_bounded():
        s = InternalSignals(
            reward_prediction_error=torch.full((B,), 1e6),
            epistemic_uncertainty=torch.full((B,), 1e6),
            prediction_error_magnitude=torch.full((B,), 1e6),
            novelty_score=torch.ones(B),
            homeostatic_deviation=torch.full((B,), 1e6),
        )
        result = module(s)
        assert (result.relevance >= 0).all() and (result.relevance <= 1).all()
        assert (result.congruence >= -1).all() and (result.congruence <= 1).all()
        assert (result.coping_potential >= 0).all() and (result.coping_potential <= 1).all()
        assert (result.norm_compatibility >= 0).all() and (result.norm_compatibility <= 1).all()
    _test("All outputs bounded with extreme inputs", test_bounded)

    # Test 8: Deterministic
    def test_deterministic():
        s = InternalSignals(
            reward_prediction_error=torch.randn(B),
            epistemic_uncertainty=torch.rand(B),
            prediction_error_magnitude=torch.rand(B),
            novelty_score=torch.rand(B),
            homeostatic_deviation=torch.rand(B),
        )
        r1 = module(s)
        r2 = module(s)
        assert torch.allclose(r1.relevance, r2.relevance, atol=1e-7)
        assert torch.allclose(r1.congruence, r2.congruence, atol=1e-7)
    _test("Deterministic appraisal", test_deterministic)

    # Test 9: Mood congruence bias
    def test_mood_bias():
        s = InternalSignals.zeros(B, device)
        s.reward_prediction_error = torch.full((B,), 0.01)  # ambiguous
        mood_neg = torch.full((B,), -1.0)
        mood_pos = torch.full((B,), 1.0)
        c_neg = module(s, mood_valence=mood_neg).congruence
        c_pos = module(s, mood_valence=mood_pos).congruence
        assert (c_pos > c_neg).all(), f"neg_mood={c_neg}, pos_mood={c_pos}"
    _test("Mood congruence bias", test_mood_bias)

    # Test 10: Learned appraisal mode
    def test_learned():
        learned_mod = AppraisalModule(use_learned=True, hidden_dim=16)
        learned_mod.train(False)
        s = InternalSignals.zeros(B, device)
        s.reward_prediction_error = torch.randn(B)
        result = learned_mod(s)
        assert result.relevance.shape == (B,)
        assert (result.relevance >= 0).all() and (result.relevance <= 1).all()
        assert (result.congruence >= -1).all() and (result.congruence <= 1).all()
    _test("Learned appraisal mode", test_learned)

    # Report
    print()
    print("-" * 72)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors_log:
        for e in errors_log:
            print(e)
    print("-" * 72)
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    _run_self_tests()
