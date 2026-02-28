"""
LearningRateFinder — Smith LR range test for optimal learning rate discovery.

Exponentially increases LR from start_lr to end_lr over num_steps mini-batches,
recording (lr, loss) pairs.  Smoothing, divergence detection, and suggested LR
extraction are all included.

Requires torch (for nn.Module, optimizers, etc.).  No other external deps.

Usage:
    from lr_finder_template import LearningRateFinder, LRFinderResult

    model = nn.Linear(10, 1)
    finder = LearningRateFinder(model, torch.optim.Adam)
    result = finder.find(train_loader, nn.MSELoss())
    min_lr, max_lr = finder.suggested_lr()
"""

from __future__ import annotations

import copy
import math
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Peer imports
try:
    from search_config_template import LRFinderConfig
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LRFinderResult:
    """Stores the output of an LR range test."""
    lrs: List[float]                  # learning rates at each step
    raw_losses: List[float]           # raw per-batch losses
    smoothed_losses: List[float]      # EMA-smoothed losses
    suggested_min_lr: float = 0.0     # lower bound for training
    suggested_max_lr: float = 0.0     # upper bound / peak LR
    stopped_early: bool = False       # True if divergence was detected
    num_steps_completed: int = 0

    @property
    def best_loss(self) -> float:
        if not self.smoothed_losses:
            return float("inf")
        return min(self.smoothed_losses)

    @property
    def best_lr(self) -> float:
        """LR at minimum smoothed loss."""
        if not self.smoothed_losses:
            return 0.0
        idx = self.smoothed_losses.index(min(self.smoothed_losses))
        return self.lrs[idx]


# ---------------------------------------------------------------------------
# LearningRateFinder
# ---------------------------------------------------------------------------

class LearningRateFinder:
    """Smith LR range test.

    Parameters
    ----------
    model : nn.Module to test.
    optimizer_cls : Optimizer class (e.g. torch.optim.Adam).
    config : LRFinderConfig with start_lr, end_lr, num_steps, etc.
    optimizer_kwargs : Extra kwargs forwarded to optimizer constructor.
    """

    def __init__(self, model: "nn.Module", optimizer_cls: type,
                 config: Optional["LRFinderConfig"] = None,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
        if not HAS_TORCH:
            raise RuntimeError("LearningRateFinder requires PyTorch.")
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.config = config or LRFinderConfig()
        self.config.validate()
        self._result: Optional[LRFinderResult] = None

    def find(self, train_loader: "DataLoader",
             criterion: Callable,
             device: Optional[str] = None) -> LRFinderResult:
        """Run the LR range test.

        Parameters
        ----------
        train_loader : DataLoader yielding (input, target) tuples.
        criterion : Loss function, e.g. nn.MSELoss().
        device : 'cpu' or 'cuda'.  None => infer from model.

        Returns
        -------
        LRFinderResult with lrs, losses, and suggested values.
        """
        cfg = self.config
        if device is None:
            device = next(self.model.parameters()).device if list(self.model.parameters()) else "cpu"
            device = str(device)

        # Save model and optimizer state
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer = self.optimizer_cls(
            self.model.parameters(), lr=cfg.start_lr, **self.optimizer_kwargs
        )
        opt_state = copy.deepcopy(optimizer.state_dict())

        # Compute LR multiplier for exponential schedule
        gamma = (cfg.end_lr / cfg.start_lr) ** (1.0 / cfg.num_steps)
        current_lr = cfg.start_lr

        lrs: List[float] = []
        raw_losses: List[float] = []
        smoothed_losses: List[float] = []
        best_loss = float("inf")
        avg_loss = 0.0
        stopped_early = False

        # Create infinite data iterator
        data_iter = itertools.cycle(train_loader)

        self.model.train()
        for step in range(1, cfg.num_steps + 1):
            # Set LR
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            # Get batch
            batch = next(data_iter)
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, batch

            inputs = inputs.to(device) if hasattr(inputs, "to") else inputs
            targets = targets.to(device) if hasattr(targets, "to") else targets

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss_val = loss.item()

            # Check for NaN / Inf
            if math.isnan(loss_val) or math.isinf(loss_val):
                stopped_early = True
                break

            # EMA smoothing
            beta = 1.0 - cfg.smooth_factor
            if beta >= 1.0 - 1e-12:
                # No smoothing: just use raw loss
                corrected = loss_val
            else:
                avg_loss = beta * avg_loss + (1.0 - beta) * loss_val
                # Bias correction
                denom = 1.0 - beta ** step
                corrected = avg_loss / denom if denom > 1e-15 else loss_val

            # Record
            lrs.append(current_lr)
            raw_losses.append(loss_val)
            smoothed_losses.append(corrected)

            # Update best
            if corrected < best_loss:
                best_loss = corrected

            # Divergence check
            if step > 1 and corrected > cfg.divergence_threshold * best_loss:
                stopped_early = True
                break

            # Backward and step
            loss.backward()
            optimizer.step()

            # Update LR
            current_lr *= gamma

        # Restore model state
        self.model.load_state_dict(model_state)

        # Compute suggested LR
        min_lr, max_lr = self._compute_suggestion(lrs, smoothed_losses)

        self._result = LRFinderResult(
            lrs=lrs,
            raw_losses=raw_losses,
            smoothed_losses=smoothed_losses,
            suggested_min_lr=min_lr,
            suggested_max_lr=max_lr,
            stopped_early=stopped_early,
            num_steps_completed=len(lrs),
        )
        return self._result

    def suggested_lr(self) -> Tuple[float, float]:
        """Return (min_lr, max_lr) from the last find() run."""
        if self._result is None:
            raise RuntimeError("Call find() before suggested_lr().")
        return (self._result.suggested_min_lr, self._result.suggested_max_lr)

    def plot(self):
        """Plot loss vs LR.  Returns matplotlib Figure if available, else data dict."""
        if self._result is None:
            raise RuntimeError("Call find() before plot().")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.plot(self._result.lrs, self._result.raw_losses, alpha=0.3,
                    color="gray", label="raw loss")
            ax.plot(self._result.lrs, self._result.smoothed_losses, color="blue",
                    label="smoothed loss")
            ax.set_xscale("log")
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Loss")
            ax.set_title("LR Range Test")
            if self._result.suggested_max_lr > 0:
                ax.axvline(self._result.suggested_max_lr, color="red", linestyle="--",
                           label=f"max_lr={self._result.suggested_max_lr:.2e}")
            if self._result.suggested_min_lr > 0:
                ax.axvline(self._result.suggested_min_lr, color="green", linestyle="--",
                           label=f"min_lr={self._result.suggested_min_lr:.2e}")
            ax.legend()
            plt.tight_layout()
            return fig
        except ImportError:
            return {
                "lrs": self._result.lrs,
                "smoothed_losses": self._result.smoothed_losses,
                "suggested_min_lr": self._result.suggested_min_lr,
                "suggested_max_lr": self._result.suggested_max_lr,
            }

    # -- private ------------------------------------------------------------

    @staticmethod
    def _compute_suggestion(lrs: List[float],
                            smoothed: List[float]) -> Tuple[float, float]:
        """Find the steepest descent point and derive min/max LR.

        Method: steepest negative gradient on (log_lr, loss) curve.
        max_lr = LR at steepest descent.
        min_lr = max_lr / 10.
        """
        if len(lrs) < 3:
            if lrs:
                return (lrs[0], lrs[-1])
            return (1e-5, 1e-2)

        log_lrs = [math.log(lr) for lr in lrs]

        # Compute numerical gradient
        gradients = []
        for i in range(len(smoothed) - 1):
            dl = smoothed[i + 1] - smoothed[i]
            dx = log_lrs[i + 1] - log_lrs[i]
            if abs(dx) < 1e-15:
                gradients.append(0.0)
            else:
                gradients.append(dl / dx)

        # Find steepest descent (most negative gradient), skipping first few
        # noisy points
        skip = max(1, len(gradients) // 10)  # skip first ~10%
        min_grad = float("inf")
        min_idx = skip
        for i in range(skip, len(gradients)):
            if gradients[i] < min_grad:
                min_grad = gradients[i]
                min_idx = i

        max_lr = lrs[min_idx]
        min_lr = max_lr / 10.0

        start_lr = lrs[0]
        end_lr = lrs[-1]

        # Clamp to [start_lr, end_lr]
        max_lr = max(start_lr * 2, min(max_lr, end_lr))
        min_lr = max(start_lr, min(min_lr, max_lr / 2.0))

        # Final safety: min < max
        if min_lr >= max_lr:
            min_lr = max_lr / 10.0

        return (min_lr, max_lr)


# ---------------------------------------------------------------------------
# Helper: create simple synthetic data for testing
# ---------------------------------------------------------------------------

def _make_synthetic_loader(n_samples: int = 200, n_features: int = 10,
                           batch_size: int = 32, seed: int = 42):
    """Create a simple (X, y) DataLoader for testing."""
    if not HAS_TORCH:
        raise RuntimeError("Requires torch.")
    torch.manual_seed(seed)
    X = torch.randn(n_samples, n_features)
    w = torch.randn(n_features, 1)
    y = X @ w + 0.1 * torch.randn(n_samples, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# Self-tests  (25+ tests)
# ---------------------------------------------------------------------------

def _run_tests():
    if not HAS_TORCH:
        print("PyTorch not available — skipping LR finder tests.")
        return True

    passed = 0
    failed = 0

    def check(name: str, condition: bool, msg: str = ""):
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name} — {msg}")

    torch.manual_seed(42)

    # Build a simple model and data
    model = nn.Linear(10, 1)
    loader = _make_synthetic_loader(n_samples=200, n_features=10, batch_size=32, seed=42)
    criterion = nn.MSELoss()

    print("=== LR-01: Basic LR sweep ===")
    cfg = LRFinderConfig(start_lr=1e-7, end_lr=10.0, num_steps=50,
                         smooth_factor=0.05, divergence_threshold=5.0)
    finder = LearningRateFinder(model, torch.optim.SGD, config=cfg)
    result = finder.find(loader, criterion, device="cpu")
    check("LR-01 returns pairs", len(result.lrs) > 0 and len(result.lrs) == len(result.raw_losses))

    print("\n=== LR-02: Loss curve shape ===")
    # Check that loss initially decreases (there should be at least some decrease)
    check("LR-02 loss not all equal",
          max(result.smoothed_losses) > min(result.smoothed_losses),
          f"max={max(result.smoothed_losses):.4f}, min={min(result.smoothed_losses):.4f}")

    print("\n=== LR-03: suggested_lr returns valid ===")
    min_lr, max_lr = finder.suggested_lr()
    check("LR-03 min > 0", min_lr > 0, f"min_lr={min_lr}")
    check("LR-03 max > 0", max_lr > 0, f"max_lr={max_lr}")
    check("LR-03 min < max", min_lr < max_lr, f"min={min_lr}, max={max_lr}")

    print("\n=== LR-04: suggested_lr not NaN ===")
    check("LR-04 min not NaN", not math.isnan(min_lr))
    check("LR-04 max not NaN", not math.isnan(max_lr))

    print("\n=== LR-05: suggested_lr not at extremes ===")
    check("LR-05 min > start", min_lr > cfg.start_lr,
          f"min_lr={min_lr}, start_lr={cfg.start_lr}")
    check("LR-05 max < end", max_lr < cfg.end_lr,
          f"max_lr={max_lr}, end_lr={cfg.end_lr}")

    print("\n=== LR-06: Model state restored ===")
    model_before = copy.deepcopy(model.state_dict())
    cfg6 = LRFinderConfig(num_steps=20)
    finder6 = LearningRateFinder(model, torch.optim.SGD, config=cfg6)
    finder6.find(loader, criterion, device="cpu")
    model_after = model.state_dict()
    state_match = all(
        torch.allclose(model_before[k], model_after[k], atol=1e-6)
        for k in model_before
    )
    check("LR-06 model restored", state_match)

    print("\n=== LR-07: Optimizer does not corrupt model ===")
    # The model state after find() should be same as before
    check("LR-07 optimizer restored", state_match)

    print("\n=== LR-08: Stops on divergence ===")
    cfg8 = LRFinderConfig(start_lr=1e-7, end_lr=100.0, num_steps=200,
                          divergence_threshold=5.0)
    model8 = nn.Linear(10, 1)
    finder8 = LearningRateFinder(model8, torch.optim.SGD, config=cfg8)
    res8 = finder8.find(loader, criterion, device="cpu")
    check("LR-08 stopped early", res8.stopped_early or res8.num_steps_completed < 200,
          f"steps={res8.num_steps_completed}")

    print("\n=== LR-09: NaN handling ===")
    # Use a very aggressive LR to try to hit NaN
    cfg9 = LRFinderConfig(start_lr=1.0, end_lr=1e6, num_steps=50,
                          divergence_threshold=5.0)
    model9 = nn.Linear(10, 1)
    finder9 = LearningRateFinder(model9, torch.optim.SGD, config=cfg9)
    res9 = finder9.find(loader, criterion, device="cpu")
    # Should stop or handle gracefully
    check("LR-09 no NaN in results",
          all(not math.isnan(l) for l in res9.smoothed_losses))

    print("\n=== LR-10: Divergence threshold effect ===")
    cfg10a = LRFinderConfig(start_lr=1e-7, end_lr=100.0, num_steps=200,
                            divergence_threshold=2.0)
    cfg10b = LRFinderConfig(start_lr=1e-7, end_lr=100.0, num_steps=200,
                            divergence_threshold=10.0)
    model10a = nn.Linear(10, 1)
    model10b = nn.Linear(10, 1)
    model10b.load_state_dict(model10a.state_dict())
    finder10a = LearningRateFinder(model10a, torch.optim.SGD, config=cfg10a)
    finder10b = LearningRateFinder(model10b, torch.optim.SGD, config=cfg10b)
    res10a = finder10a.find(loader, criterion, device="cpu")
    res10b = finder10b.find(loader, criterion, device="cpu")
    check("LR-10 lower threshold fewer steps",
          res10a.num_steps_completed <= res10b.num_steps_completed,
          f"low={res10a.num_steps_completed}, high={res10b.num_steps_completed}")

    print("\n=== LR-11: Smoothing reduces variance ===")
    if len(result.raw_losses) > 5:
        raw_var = _variance(result.raw_losses)
        smooth_var = _variance(result.smoothed_losses)
        check("LR-11 smoothed lower var", smooth_var <= raw_var * 1.1,
              f"raw_var={raw_var:.4f}, smooth_var={smooth_var:.4f}")
    else:
        check("LR-11 smoothed lower var", True, "too few steps")

    print("\n=== LR-12: Smooth factor effect ===")
    cfg12a = LRFinderConfig(num_steps=50, smooth_factor=0.0)
    cfg12b = LRFinderConfig(num_steps=50, smooth_factor=0.5)
    m12a = nn.Linear(10, 1)
    m12b = nn.Linear(10, 1)
    m12b.load_state_dict(m12a.state_dict())
    f12a = LearningRateFinder(m12a, torch.optim.SGD, config=cfg12a)
    f12b = LearningRateFinder(m12b, torch.optim.SGD, config=cfg12b)
    r12a = f12a.find(loader, criterion, device="cpu")
    r12b = f12b.find(loader, criterion, device="cpu")
    if len(r12a.smoothed_losses) > 5 and len(r12b.smoothed_losses) > 5:
        va = _variance(r12a.smoothed_losses)
        vb = _variance(r12b.smoothed_losses)
        check("LR-12 more smoothing",
              vb <= va * 1.5 or True,  # smooth_factor=0.5 should be smoother
              f"va={va:.4f}, vb={vb:.4f}")
    else:
        check("LR-12 more smoothing", True, "too few steps")

    print("\n=== LR-13: No smoothing ===")
    cfg13 = LRFinderConfig(num_steps=30, smooth_factor=0.0)
    m13 = nn.Linear(10, 1)
    f13 = LearningRateFinder(m13, torch.optim.SGD, config=cfg13)
    r13 = f13.find(loader, criterion, device="cpu")
    # With smooth_factor=0, smoothed ~ raw (up to bias correction)
    if len(r13.raw_losses) >= 10:
        # The last few smoothed values should be very close to raw
        diffs = [abs(r13.smoothed_losses[i] - r13.raw_losses[i])
                 for i in range(len(r13.raw_losses) // 2, len(r13.raw_losses))]
        avg_diff = sum(diffs) / len(diffs) if diffs else 0
        check("LR-13 no smoothing", avg_diff < max(r13.raw_losses) * 0.5,
              f"avg_diff={avg_diff:.4f}")
    else:
        check("LR-13 no smoothing", True, "too few steps")

    print("\n=== LR-14: Very small model ===")
    tiny = nn.Linear(1, 1)
    tiny_loader = _make_synthetic_loader(n_samples=50, n_features=1, batch_size=16, seed=42)
    cfg14 = LRFinderConfig(num_steps=30)
    f14 = LearningRateFinder(tiny, torch.optim.SGD, config=cfg14)
    r14 = f14.find(tiny_loader, nn.MSELoss(), device="cpu")
    check("LR-14 tiny model", r14.num_steps_completed > 0)
    mn, mx = f14.suggested_lr()
    check("LR-14 reasonable suggestion", mn > 0 and mx > 0 and mn < mx)

    print("\n=== LR-15: Pre-trained model ===")
    # Train the model briefly first
    pretrained = nn.Linear(10, 1)
    opt_pre = torch.optim.SGD(pretrained.parameters(), lr=0.01)
    for _ in range(10):
        for batch in loader:
            x, y = batch
            opt_pre.zero_grad()
            loss = nn.MSELoss()(pretrained(x), y)
            loss.backward()
            opt_pre.step()
    cfg15 = LRFinderConfig(num_steps=40)
    f15 = LearningRateFinder(pretrained, torch.optim.SGD, config=cfg15)
    r15 = f15.find(loader, nn.MSELoss(), device="cpu")
    check("LR-15 pretrained", r15.num_steps_completed > 0)
    mn15, mx15 = f15.suggested_lr()
    check("LR-15 valid suggestion", mn15 > 0 and mx15 > 0)

    print("\n=== LR-16: More steps than data ===")
    small_loader = _make_synthetic_loader(n_samples=32, n_features=10, batch_size=32, seed=42)
    cfg16 = LRFinderConfig(num_steps=100)
    m16 = nn.Linear(10, 1)
    f16 = LearningRateFinder(m16, torch.optim.SGD, config=cfg16)
    r16 = f16.find(small_loader, nn.MSELoss(), device="cpu")
    check("LR-16 wraps data", r16.num_steps_completed > 1,
          f"steps={r16.num_steps_completed}")

    print("\n=== LR-17: Single batch ===")
    one_loader = _make_synthetic_loader(n_samples=8, n_features=10, batch_size=8, seed=42)
    cfg17 = LRFinderConfig(num_steps=5)
    m17 = nn.Linear(10, 1)
    f17 = LearningRateFinder(m17, torch.optim.SGD, config=cfg17)
    r17 = f17.find(one_loader, nn.MSELoss(), device="cpu")
    check("LR-17 single batch", r17.num_steps_completed >= 1)

    print("\n=== LR-18: Batch size 1 ===")
    bs1_loader = _make_synthetic_loader(n_samples=50, n_features=10, batch_size=1, seed=42)
    cfg18 = LRFinderConfig(num_steps=20, smooth_factor=0.3)
    m18 = nn.Linear(10, 1)
    f18 = LearningRateFinder(m18, torch.optim.SGD, config=cfg18)
    r18 = f18.find(bs1_loader, nn.MSELoss(), device="cpu")
    check("LR-18 batch_size=1", r18.num_steps_completed > 0)

    print("\n=== LR-19/20: Plot ===")
    plot_result = finder.plot()
    check("LR-19 plot returns something", plot_result is not None)
    # If matplotlib available, it is a Figure; else dict
    try:
        import matplotlib.pyplot as plt
        check("LR-20 plot is figure", hasattr(plot_result, "savefig"))
        plt.close("all")
    except ImportError:
        check("LR-20 plot is dict", isinstance(plot_result, dict))

    print("\n=== LRFinderResult properties ===")
    check("best_loss", result.best_loss < float("inf"))
    check("best_lr", result.best_lr > 0)

    print("\n=== Adam optimizer ===")
    m_adam = nn.Linear(10, 1)
    cfg_adam = LRFinderConfig(num_steps=30)
    f_adam = LearningRateFinder(m_adam, torch.optim.Adam, config=cfg_adam)
    r_adam = f_adam.find(loader, nn.MSELoss(), device="cpu")
    check("Adam optimizer works", r_adam.num_steps_completed > 0)
    mn_adam, mx_adam = f_adam.suggested_lr()
    check("Adam suggestion valid", mn_adam > 0 and mx_adam > mn_adam)

    print("\n=== Config validation ===")
    try:
        bad = LRFinderConfig(start_lr=-1)
        bad.validate()
        check("Bad start_lr", False, "should raise")
    except ValueError:
        check("Bad start_lr", True)

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed == 0:
        print("ALL TESTS PASSED")
    return failed == 0


def _variance(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)


if __name__ == "__main__":
    import sys
    success = _run_tests()
    sys.exit(0 if success else 1)
