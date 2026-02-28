"""
Valence-Arousal-Dominance (PAD) Space -- Dimensional Model Template.

Provides the mathematical foundation for mapping internal signals to a
three-dimensional affective space based on Russell's circumplex model
extended with the dominance dimension (Mehrabian & Russell PAD model).

Target file: ``brain_ai/affect/space.py``

Key classes:
    - ``ValenceArousalSpace``: Main class for PAD space operations.
    - ``CircumplexProjection``: 2D projection utilities (V-A plane).

The PAD space is a unit cube [-1, 1]^3 where:
    - Valence (V): pleasure-displeasure axis
    - Arousal (A): activation-deactivation axis
    - Dominance (D): control-submission axis

References:
    - Russell, J. A. (1980). A circumplex model of affect.
    - Mehrabian, A. & Russell, J. A. (1974). An Approach to Environmental Psychology.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Named emotion regions in PAD space
# ============================================================================

EMOTION_REGIONS: Dict[str, Tuple[float, float, float]] = {
    "happy":       ( 0.8,  0.3,  0.5),
    "excited":     ( 0.7,  0.8,  0.6),
    "content":     ( 0.6, -0.3,  0.5),
    "calm":        ( 0.3, -0.7,  0.4),
    "bored":       (-0.3, -0.6, -0.2),
    "sad":         (-0.7, -0.3, -0.5),
    "angry":       (-0.6,  0.7,  0.6),
    "afraid":      (-0.6,  0.8, -0.6),
    "surprised":   ( 0.1,  0.9, -0.1),
    "frustrated":  (-0.5,  0.6,  0.5),
    "curious":     ( 0.2,  0.6,  0.5),
    "anxious":     (-0.4,  0.7, -0.5),
    "neutral":     ( 0.0,  0.0,  0.0),
}


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class PADPoint:
    """A point in the PAD (Pleasure-Arousal-Dominance) space.

    Attributes:
        valence: Pleasure-displeasure axis, [-1, 1].
        arousal: Activation-deactivation axis, [-1, 1].
        dominance: Control-submission axis, [-1, 1].
    """
    valence: Tensor
    arousal: Tensor
    dominance: Tensor

    @property
    def intensity(self) -> Tensor:
        """Euclidean distance from neutral (0, 0, 0)."""
        return torch.sqrt(
            self.valence ** 2 + self.arousal ** 2 + self.dominance ** 2
        )

    @property
    def circumplex_angle(self) -> Tensor:
        """Angle in the V-A circumplex plane (radians)."""
        return torch.atan2(self.arousal, self.valence)

    @property
    def circumplex_radius(self) -> Tensor:
        """Radius in the V-A circumplex plane."""
        return torch.sqrt(self.valence ** 2 + self.arousal ** 2)

    def as_tensor(self) -> Tensor:
        """Stack to (B, 3) tensor [V, A, D]."""
        return torch.stack([self.valence, self.arousal, self.dominance], dim=-1)

    @classmethod
    def from_tensor(cls, t: Tensor) -> PADPoint:
        """Create from (B, 3) tensor."""
        return cls(valence=t[..., 0], arousal=t[..., 1], dominance=t[..., 2])

    def nearest_emotion(self) -> List[str]:
        """Find the nearest named emotion for each batch element."""
        pad_tensor = self.as_tensor().float()  # (B, 3)
        regions = torch.tensor(
            [EMOTION_REGIONS[k] for k in EMOTION_REGIONS],
            dtype=torch.float32,
            device=pad_tensor.device,
        )  # (N_emotions, 3)
        names = list(EMOTION_REGIONS.keys())

        # (B, N_emotions) distances
        dists = torch.cdist(pad_tensor.unsqueeze(0), regions.unsqueeze(0)).squeeze(0)
        indices = dists.argmin(dim=-1)  # (B,)
        return [names[i.item()] for i in indices]


# ============================================================================
# ValenceArousalSpace
# ============================================================================

class ValenceArousalSpace(nn.Module):
    """PAD space operations: signal-to-affect mapping, geometry, and distance.

    This module provides both analytic (tanh/sigmoid) and optional learned
    mappings from internal signals to the PAD space.

    Args:
        valence_scale: Scaling for valence computation.
        arousal_scale: Scaling for arousal computation.
        dominance_scale: Scaling for dominance computation.
        output_clamp: Hard clamp magnitude for all dimensions.
        hidden_dim: Hidden dimension for optional learned mapping.
        use_learned_mapping: If True, use an MLP instead of analytic mapping.
    """

    def __init__(
        self,
        valence_scale: float = 1.0,
        arousal_scale: float = 1.0,
        dominance_scale: float = 1.0,
        output_clamp: float = 1.0,
        hidden_dim: int = 64,
        use_learned_mapping: bool = False,
    ):
        super().__init__()
        self.valence_scale = valence_scale
        self.arousal_scale = arousal_scale
        self.dominance_scale = dominance_scale
        self.output_clamp = output_clamp
        self.use_learned_mapping = use_learned_mapping

        if use_learned_mapping:
            self.mapping_net = nn.Sequential(
                nn.Linear(5, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3),
            )
        else:
            self.mapping_net = None

    def from_signals(
        self,
        reward_prediction_error: Tensor,
        epistemic_uncertainty: Tensor,
        prediction_error_magnitude: Tensor,
        novelty_score: Tensor,
        homeostatic_deviation: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Map internal signals to PAD coordinates.

        All inputs are (B,) tensors. Outputs are (B,) tensors in [-1, 1].

        Args:
            reward_prediction_error: RPE from DA system, (-inf, inf).
            epistemic_uncertainty: From transition model ensemble, [0, inf).
            prediction_error_magnitude: Sensory PE norm, [0, inf).
            novelty_score: HTM anomaly or embedding distance, [0, 1].
            homeostatic_deviation: Distance from setpoints, [0, inf).

        Returns:
            Tuple of (valence, arousal, dominance), each (B,) in [-1, 1].
        """
        # Ensure fp32
        rpe = reward_prediction_error.float()
        unc = epistemic_uncertainty.float()
        pe_mag = prediction_error_magnitude.float()
        nov = novelty_score.float()
        hd = homeostatic_deviation.float()

        if self.use_learned_mapping and self.mapping_net is not None:
            signals = torch.stack([rpe, unc, pe_mag, nov, hd], dim=-1)  # (B, 5)
            pad_raw = self.mapping_net(signals)  # (B, 3)
            pad = torch.tanh(pad_raw)
            valence = pad[:, 0]
            arousal = pad[:, 1]
            dominance = pad[:, 2]
        else:
            valence = self._compute_valence(rpe, hd)
            arousal = self._compute_arousal(pe_mag, unc, nov)
            dominance = self._compute_dominance(unc)

        valence = torch.clamp(valence, -self.output_clamp, self.output_clamp)
        arousal = torch.clamp(arousal, -self.output_clamp, self.output_clamp)
        dominance = torch.clamp(dominance, -self.output_clamp, self.output_clamp)

        return valence, arousal, dominance

    def _compute_valence(
        self, rpe: Tensor, homeostatic_deviation: Tensor
    ) -> Tensor:
        """Analytic valence computation.

        Valence is driven by RPE (congruence) and homeostatic norm deviation.
        Zero RPE and zero deviation -> zero valence (neutral).
        """
        congruence = torch.tanh(self.valence_scale * rpe)
        norm_compat = torch.tanh(-0.5 * homeostatic_deviation)
        # norm_compat is 0 when deviation is 0, negative when deviation is large
        valence = 0.7 * congruence + 0.3 * norm_compat
        return valence

    def _compute_arousal(
        self, pe_magnitude: Tensor, uncertainty: Tensor, novelty: Tensor
    ) -> Tensor:
        """Analytic arousal computation.

        Arousal is driven by PE magnitude, uncertainty, and novelty.
        All zero -> arousal = 0 (neutral). All signals increase arousal.
        """
        # Combined signal, all non-negative
        raw = pe_magnitude + uncertainty + novelty
        # sigmoid(0) = 0.5, so shift: 2*sigmoid(x) - 1 maps 0 -> 0
        # We use tanh instead for cleaner zero -> zero mapping
        arousal = torch.tanh(self.arousal_scale * 0.5 * raw)
        return arousal

    def _compute_dominance(self, uncertainty: Tensor) -> Tensor:
        """Analytic dominance computation.

        Dominance is inversely related to uncertainty.
        Zero uncertainty -> dominance ~= 0 (neutral).
        High uncertainty -> negative dominance (low control).
        """
        # tanh(-x) maps 0 -> 0, high uncertainty -> negative
        dominance = torch.tanh(-self.dominance_scale * 0.5 * uncertainty)
        # Negate so low uncertainty = positive dominance
        dominance = -dominance
        return dominance

    def distance(self, p1: PADPoint, p2: PADPoint) -> Tensor:
        """Euclidean distance between two PAD points.

        Args:
            p1: First PAD point.
            p2: Second PAD point.

        Returns:
            (B,) tensor of distances.
        """
        d = (p1.valence - p2.valence) ** 2
        d = d + (p1.arousal - p2.arousal) ** 2
        d = d + (p1.dominance - p2.dominance) ** 2
        return torch.sqrt(d)

    def cosine_similarity(self, p1: PADPoint, p2: PADPoint) -> Tensor:
        """Cosine similarity between two PAD vectors.

        Returns:
            (B,) tensor in [-1, 1].
        """
        t1 = p1.as_tensor()  # (B, 3)
        t2 = p2.as_tensor()  # (B, 3)
        return F.cosine_similarity(t1, t2, dim=-1)

    def interpolate(
        self, p1: PADPoint, p2: PADPoint, alpha: float
    ) -> PADPoint:
        """Linear interpolation between two PAD points.

        Args:
            p1: Start point.
            p2: End point.
            alpha: Interpolation factor, 0 = p1, 1 = p2.

        Returns:
            Interpolated PAD point.
        """
        v = (1 - alpha) * p1.valence + alpha * p2.valence
        a = (1 - alpha) * p1.arousal + alpha * p2.arousal
        d = (1 - alpha) * p1.dominance + alpha * p2.dominance
        return PADPoint(valence=v, arousal=a, dominance=d)


# ============================================================================
# Self-test
# ============================================================================

def _run_self_tests() -> None:
    """Self-tests for ValenceArousalSpace."""
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
    print("ValenceArousalSpace -- Self-Tests")
    print("=" * 72)

    device = torch.device("cpu")
    B = 4

    space = ValenceArousalSpace(
        valence_scale=1.0,
        arousal_scale=1.0,
        dominance_scale=1.0,
    )
    space.train(False)

    # Test 1: Neutral on zero inputs
    def test_neutral():
        v, a, d = space.from_signals(
            torch.zeros(B, device=device),
            torch.zeros(B, device=device),
            torch.zeros(B, device=device),
            torch.zeros(B, device=device),
            torch.zeros(B, device=device),
        )
        assert torch.allclose(v, torch.zeros(B), atol=1e-6), f"valence={v}"
        assert torch.allclose(a, torch.zeros(B), atol=1e-6), f"arousal={a}"
        assert torch.allclose(d, torch.zeros(B), atol=1e-6), f"dominance={d}"
    _test("Neutral on zero inputs", test_neutral)

    # Test 2: Bounded outputs with extreme inputs
    def test_bounded():
        v, a, d = space.from_signals(
            torch.full((B,), 1e6),
            torch.full((B,), 1e6),
            torch.full((B,), 1e6),
            torch.ones(B),
            torch.full((B,), 1e6),
        )
        assert v.abs().max() <= 1.0 + 1e-7, f"valence max={v.abs().max()}"
        assert a.abs().max() <= 1.0 + 1e-7, f"arousal max={a.abs().max()}"
        assert d.abs().max() <= 1.0 + 1e-7, f"dominance max={d.abs().max()}"
    _test("Bounded outputs (extreme positive)", test_bounded)

    # Test 3: Bounded outputs with extreme negative inputs
    def test_bounded_neg():
        v, a, d = space.from_signals(
            torch.full((B,), -1e6),
            torch.zeros(B),
            torch.zeros(B),
            torch.zeros(B),
            torch.zeros(B),
        )
        assert v.abs().max() <= 1.0 + 1e-7
    _test("Bounded outputs (extreme negative)", test_bounded_neg)

    # Test 4: Positive RPE -> positive valence
    def test_positive_rpe_positive_valence():
        v, _, _ = space.from_signals(
            torch.full((B,), 2.0),
            torch.zeros(B),
            torch.zeros(B),
            torch.zeros(B),
            torch.zeros(B),
        )
        assert (v > 0).all(), f"Expected positive valence, got {v}"
    _test("Positive RPE -> positive valence", test_positive_rpe_positive_valence)

    # Test 5: Negative RPE -> negative valence
    def test_negative_rpe_negative_valence():
        v, _, _ = space.from_signals(
            torch.full((B,), -2.0),
            torch.zeros(B),
            torch.zeros(B),
            torch.zeros(B),
            torch.zeros(B),
        )
        assert (v < 0).all(), f"Expected negative valence, got {v}"
    _test("Negative RPE -> negative valence", test_negative_rpe_negative_valence)

    # Test 6: High PE magnitude -> positive arousal
    def test_pe_arousal():
        _, a, _ = space.from_signals(
            torch.zeros(B),
            torch.zeros(B),
            torch.full((B,), 5.0),
            torch.zeros(B),
            torch.zeros(B),
        )
        assert (a > 0).all(), f"Expected positive arousal, got {a}"
    _test("High PE magnitude -> positive arousal", test_pe_arousal)

    # Test 7: High uncertainty -> negative dominance (low coping)
    def test_uncertainty_dominance():
        _, _, d = space.from_signals(
            torch.zeros(B),
            torch.full((B,), 5.0),
            torch.zeros(B),
            torch.zeros(B),
            torch.zeros(B),
        )
        # High uncertainty should push dominance negative
        # (low coping potential)
        assert (d < 0).all(), f"Expected negative dominance, got {d}"
    _test("High uncertainty -> negative dominance", test_uncertainty_dominance)

    # Test 8: Determinism
    def test_determinism():
        rpe = torch.randn(B)
        unc = torch.rand(B)
        pe = torch.rand(B)
        nov = torch.rand(B)
        hd = torch.rand(B)
        v1, a1, d1 = space.from_signals(rpe, unc, pe, nov, hd)
        v2, a2, d2 = space.from_signals(rpe, unc, pe, nov, hd)
        assert torch.allclose(v1, v2, atol=1e-7)
        assert torch.allclose(a1, a2, atol=1e-7)
        assert torch.allclose(d1, d2, atol=1e-7)
    _test("Deterministic mapping", test_determinism)

    # Test 9: PADPoint operations
    def test_pad_point():
        p = PADPoint(
            valence=torch.tensor([0.5, -0.3]),
            arousal=torch.tensor([0.8, 0.1]),
            dominance=torch.tensor([0.2, -0.7]),
        )
        assert p.intensity.shape == (2,)
        assert p.circumplex_angle.shape == (2,)
        t = p.as_tensor()
        assert t.shape == (2, 3)
        p2 = PADPoint.from_tensor(t)
        assert torch.allclose(p.valence, p2.valence)
    _test("PADPoint operations", test_pad_point)

    # Test 10: Nearest emotion
    def test_nearest():
        p = PADPoint(
            valence=torch.tensor([0.8, -0.7]),
            arousal=torch.tensor([0.3, 0.7]),
            dominance=torch.tensor([0.5, 0.6]),
        )
        names = p.nearest_emotion()
        assert len(names) == 2
        assert names[0] == "happy"
        assert names[1] == "angry"
    _test("Nearest emotion lookup", test_nearest)

    # Test 11: Distance and similarity
    def test_distance():
        p1 = PADPoint(
            valence=torch.tensor([1.0]),
            arousal=torch.tensor([0.0]),
            dominance=torch.tensor([0.0]),
        )
        p2 = PADPoint(
            valence=torch.tensor([-1.0]),
            arousal=torch.tensor([0.0]),
            dominance=torch.tensor([0.0]),
        )
        d = space.distance(p1, p2)
        assert torch.allclose(d, torch.tensor([2.0]), atol=1e-5)
        cos = space.cosine_similarity(p1, p2)
        assert cos.item() < 0, f"Expected negative cosine, got {cos}"
    _test("Distance and cosine similarity", test_distance)

    # Test 12: Interpolation
    def test_interpolation():
        p1 = PADPoint(
            valence=torch.tensor([0.0]),
            arousal=torch.tensor([0.0]),
            dominance=torch.tensor([0.0]),
        )
        p2 = PADPoint(
            valence=torch.tensor([1.0]),
            arousal=torch.tensor([1.0]),
            dominance=torch.tensor([1.0]),
        )
        mid = space.interpolate(p1, p2, 0.5)
        assert torch.allclose(mid.valence, torch.tensor([0.5]), atol=1e-5)
        assert torch.allclose(mid.arousal, torch.tensor([0.5]), atol=1e-5)
    _test("Interpolation", test_interpolation)

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
