"""
Precision + Numerics Stabilizer - LossSpikeDetector Template
=============================================================
Rolling-median-based loss spike detector with configurable window and
spike percentage threshold.

CRITICAL: Never call the inference-mode shorthand on PyTorch modules.
Use module.train(False) instead.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LossSpikeDetector:
    """Detects loss spikes using a rolling median baseline.

    Design
    ------
    Maintains a circular buffer of the last `window` loss values.
    The baseline is the **median** of this buffer (immune to single outliers).
    A spike is detected when the current loss exceeds:
        median * (1 + spike_pct / 100)

    The current loss is NOT included in the median computation — it is
    compared against the historical median.

    Parameters
    ----------
    window : int
        Rolling window size (number of recent steps to track). Default: 100.
    spike_pct : float
        Percentage increase above rolling median to classify as a spike.
        Example: 200.0 means current > 3x median triggers a spike.
    min_samples : int
        Minimum number of samples in buffer before spike detection begins.
        Default: 10. Prevents false alarms during early training warm-up.

    Examples
    --------
    ::

        detector = LossSpikeDetector(window=100, spike_pct=200.0)

        for step, batch in enumerate(dataloader):
            loss = train_step(batch)
            if detector.is_spike(loss):
                # trigger immediate sentinel check
                run_all_sentinels()
    """

    def __init__(
        self,
        window: int = 100,
        spike_pct: float = 200.0,
        min_samples: int = 10,
    ):
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if spike_pct <= 0:
            raise ValueError(f"spike_pct must be > 0, got {spike_pct}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")

        self.window = window
        self.spike_pct = spike_pct
        self.min_samples = min_samples

        self._buffer: deque = deque(maxlen=window)
        self._spike_count: int = 0
        self._last_spike_magnitude: float = 0.0
        self._last_median: float = float("nan")

    # ------------------------------------------------------------------
    # Core spike detection
    # ------------------------------------------------------------------

    def is_spike(self, current_loss: float) -> bool:
        """Check if current_loss is a spike relative to the rolling median.

        The buffer is updated AFTER the spike check — the current loss is
        compared against historical median, then added to history.

        Parameters
        ----------
        current_loss : float
            The scalar loss value for the current step.

        Returns
        -------
        bool
            True if current_loss exceeds median * (1 + spike_pct / 100).
            False if buffer has fewer than min_samples entries.
            False if current_loss is NaN/Inf (non-finite losses are not
            classified as spikes — use the NaN streak system for those).
        """
        # Non-finite losses are handled separately by the NaN streak system
        if not math.isfinite(current_loss):
            self._buffer.append(current_loss)
            return False

        # Not enough history yet
        if len(self._buffer) < self.min_samples:
            self._buffer.append(current_loss)
            return False

        # Compute median of history (excluding current)
        median = self._compute_median()
        self._last_median = median

        # Compute threshold
        threshold = median * (1.0 + self.spike_pct / 100.0)

        # Check spike (strictly greater than threshold)
        is_spike = current_loss > threshold

        if is_spike:
            magnitude = current_loss / median if median != 0 else float("inf")
            self._spike_count += 1
            self._last_spike_magnitude = magnitude
            logger.warning(
                "Loss spike detected: current=%.4f, median=%.4f, "
                "magnitude=%.2fx (%.1f%% above median, threshold=%.1f%%).",
                current_loss,
                median,
                magnitude,
                self.spike_pct,
                self.spike_pct,
            )

        # Add current to buffer after comparison
        self._buffer.append(current_loss)

        return is_spike

    def _compute_median(self) -> float:
        """Compute median of the current buffer contents.

        Uses sorted list for simplicity. For large windows, consider
        a heap-based O(log n) approach.

        Returns
        -------
        float
            Median value.
        """
        # Filter out non-finite values for robust median
        finite_vals = [v for v in self._buffer if math.isfinite(v)]
        if not finite_vals:
            return float("nan")

        sorted_vals = sorted(finite_vals)
        n = len(sorted_vals)
        mid = n // 2

        if n % 2 == 1:
            return sorted_vals[mid]
        else:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0

    # ------------------------------------------------------------------
    # Stats and state
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return a summary statistics dict.

        Returns
        -------
        dict
            Keys: median, current (last seen), spike_count, spike_magnitude,
                  buffer_size, spike_detected (bool).
        """
        current = self._buffer[-1] if self._buffer else float("nan")
        median = self._compute_median()

        if math.isfinite(median) and math.isfinite(current):
            threshold = median * (1.0 + self.spike_pct / 100.0)
            spike_detected = current > threshold
        else:
            spike_detected = False

        return {
            "median": median,
            "current": current,
            "mean": self._compute_mean(),
            "spike_count_total": self._spike_count,
            "last_spike_magnitude": self._last_spike_magnitude,
            "buffer_size": len(self._buffer),
            "spike_detected": spike_detected,
            "spike_threshold": median * (1.0 + self.spike_pct / 100.0) if math.isfinite(median) else float("nan"),
        }

    def _compute_mean(self) -> float:
        """Compute mean of finite buffer values."""
        finite_vals = [v for v in self._buffer if math.isfinite(v)]
        if not finite_vals:
            return float("nan")
        return sum(finite_vals) / len(finite_vals)

    def reset(self):
        """Clear the loss history buffer and reset counters."""
        self._buffer.clear()
        self._spike_count = 0
        self._last_spike_magnitude = 0.0
        self._last_median = float("nan")

    @property
    def buffer_size(self) -> int:
        """Current number of entries in the rolling buffer."""
        return len(self._buffer)

    @property
    def spike_count(self) -> int:
        """Total number of spikes detected since initialization or last reset."""
        return self._spike_count


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Running loss_spike_detector_template.py self-tests...")
    failures = []

    # --- T1: No spike for consistent losses ---
    try:
        detector = LossSpikeDetector(window=20, spike_pct=200.0, min_samples=5)
        # Fill buffer
        for _ in range(15):
            result = detector.is_spike(1.0)
        # Another 1.0 should not spike
        assert not detector.is_spike(1.0), "Consistent 1.0 should not be a spike"
        print("  [PASS] T1: No spike for consistent losses")
    except Exception as e:
        failures.append(f"T1 no spike: {e}")
        print(f"  [FAIL] T1: {e}")

    # --- T2: Spike detected for 3x jump (spike_pct=100 -> threshold=2x median) ---
    try:
        detector = LossSpikeDetector(window=50, spike_pct=100.0, min_samples=5)
        # Fill with 1.0
        for _ in range(20):
            detector.is_spike(1.0)
        # 3x jump exceeds threshold (2x median)
        result = detector.is_spike(3.0)
        assert result is True, f"3.0 should spike vs median=1.0 at 100% threshold (threshold=2.0)"
        print("  [PASS] T2: Spike detected for 3x jump")
    except Exception as e:
        failures.append(f"T2 spike detected: {e}")
        print(f"  [FAIL] T2: {e}")

    # --- T3: Rolling median stable with one outlier ---
    try:
        detector = LossSpikeDetector(window=20, spike_pct=200.0, min_samples=5)
        # Fill with 1.0
        for _ in range(15):
            detector.is_spike(1.0)
        # One large spike (returns True for spike, adds 100.0 to buffer)
        detector.is_spike(100.0)
        # Next normal value should NOT spike - median should still be ~1.0
        # because 100.0 is just 1 outlier in a buffer of 16 values
        result = detector.is_spike(1.1)
        assert result is False, (
            f"1.1 should not spike after one outlier, median should still be ~1.0"
        )
        stats = detector.get_stats()
        assert stats["median"] < 2.0, f"Median should be ~1.0, got {stats['median']}"
        print(f"  [PASS] T3: Median stable with outlier (median={stats['median']:.2f})")
    except Exception as e:
        failures.append(f"T3 median stability: {e}")
        print(f"  [FAIL] T3: {e}")

    # --- T4: Boundary — exactly at threshold does NOT trigger spike ---
    try:
        detector = LossSpikeDetector(window=50, spike_pct=200.0, min_samples=5)
        # Fill with 1.0 -> median=1.0, threshold=3.0
        for _ in range(20):
            detector.is_spike(1.0)
        # Exactly at threshold: 3.0 -> NOT strictly greater
        result = detector.is_spike(3.0)
        assert result is False, (
            "Exactly at threshold (3.0 == 3.0) should NOT be a spike (strictly greater)"
        )
        print("  [PASS] T4: Boundary - exactly at threshold is not a spike")
    except Exception as e:
        failures.append(f"T4 boundary: {e}")
        print(f"  [FAIL] T4: {e}")

    # --- T5: Empty buffer returns False ---
    try:
        detector = LossSpikeDetector(window=10, spike_pct=200.0, min_samples=5)
        result = detector.is_spike(999.0)
        assert result is False, "Empty buffer (< min_samples) should not spike"
        print("  [PASS] T5: Empty buffer returns False (insufficient history)")
    except Exception as e:
        failures.append(f"T5 empty buffer: {e}")
        print(f"  [FAIL] T5: {e}")

    # --- T6: Window fills correctly (maxlen behavior) ---
    try:
        detector = LossSpikeDetector(window=10, spike_pct=200.0, min_samples=5)
        for i in range(20):
            detector.is_spike(float(i + 1))  # 1.0, 2.0, ..., 20.0
        # Buffer maxlen=10, so last 10 values are 11..20, median ~ 15.5
        assert detector.buffer_size == 10, f"Buffer size should be 10, got {detector.buffer_size}"
        stats = detector.get_stats()
        assert 14.0 <= stats["median"] <= 16.0, (
            f"Median of 11..20 should be ~15.5, got {stats['median']}"
        )
        print(f"  [PASS] T6: Window fills correctly (size={detector.buffer_size}, median={stats['median']:.1f})")
    except Exception as e:
        failures.append(f"T6 window fill: {e}")
        print(f"  [FAIL] T6: {e}")

    # --- T7: reset clears buffer and counters ---
    try:
        detector = LossSpikeDetector(window=20, spike_pct=100.0, min_samples=5)
        for _ in range(15):
            detector.is_spike(1.0)
        detector.is_spike(5.0)  # spike
        assert detector.spike_count >= 1

        detector.reset()
        assert detector.buffer_size == 0, f"After reset, buffer should be empty"
        assert detector.spike_count == 0, "After reset, spike_count should be 0"
        print("  [PASS] T7: reset clears buffer and counters")
    except Exception as e:
        failures.append(f"T7 reset: {e}")
        print(f"  [FAIL] T7: {e}")

    # --- T8: get_stats returns expected keys ---
    try:
        detector = LossSpikeDetector(window=20, spike_pct=200.0, min_samples=5)
        for v in [1.0, 1.1, 0.9, 1.0, 1.2, 0.8, 1.0]:
            detector.is_spike(v)
        stats = detector.get_stats()
        for key in ["median", "current", "mean", "spike_count_total",
                    "last_spike_magnitude", "buffer_size", "spike_detected", "spike_threshold"]:
            assert key in stats, f"Missing key: {key}"
        print(f"  [PASS] T8: get_stats returns all expected keys")
    except Exception as e:
        failures.append(f"T8 get_stats: {e}")
        print(f"  [FAIL] T8: {e}")

    # --- T9: NaN losses do not spike but are added to buffer ---
    try:
        detector = LossSpikeDetector(window=20, spike_pct=200.0, min_samples=5)
        for _ in range(10):
            detector.is_spike(1.0)
        result = detector.is_spike(float("nan"))
        assert result is False, "NaN loss should not be classified as a spike"
        # Buffer should contain the NaN
        assert detector.buffer_size == 11, f"Buffer size should be 11, got {detector.buffer_size}"
        print("  [PASS] T9: NaN losses not classified as spikes")
    except Exception as e:
        failures.append(f"T9 nan loss: {e}")
        print(f"  [FAIL] T9: {e}")

    # --- T10: spike_pct=200 means threshold = 3x median ---
    try:
        detector = LossSpikeDetector(window=50, spike_pct=200.0, min_samples=5)
        for _ in range(20):
            detector.is_spike(1.0)

        # 2.99 should NOT spike (below 3.0)
        r_below = detector.is_spike(2.99)
        assert r_below is False, f"2.99 should not spike (threshold=3.0)"

        # Rebuild buffer after 2.99 was added
        detector.reset()
        for _ in range(20):
            detector.is_spike(1.0)

        # 3.01 SHOULD spike (strictly above 3.0)
        r_above = detector.is_spike(3.01)
        assert r_above is True, f"3.01 should spike (threshold=3.0)"
        print("  [PASS] T10: spike_pct=200 -> threshold=3x median, boundary correct")
    except Exception as e:
        failures.append(f"T10 spike_pct math: {e}")
        print(f"  [FAIL] T10: {e}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} test(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All 10 self-tests passed.")
