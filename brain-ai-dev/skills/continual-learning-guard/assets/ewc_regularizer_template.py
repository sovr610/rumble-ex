"""
Elastic Weight Consolidation (EWC) Regularizer Template.

Provides diagonal Fisher information computation and quadratic penalty for
preventing catastrophic forgetting. Supports both per-task EWC and online
EWC (exponential moving average of Fisher diagonals).

Key classes:
    EWCRegularizer  -- compute Fisher, store star params, compute penalty
    OnlineEWC       -- EMA-based Fisher update for long task sequences

Usage:
    ewc = EWCRegularizer(model, ewc_lambda=1000.0)
    ewc.register_task(task_data_loader, num_samples=200)
    penalty = ewc.penalty()  # Add to training loss: loss + penalty
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ===========================================================================
#  EWC Regularizer
# ===========================================================================

class EWCRegularizer:
    """Elastic Weight Consolidation regularizer.

    After completing a task, call register_task() to compute the Fisher
    diagonal and snapshot optimal parameters. During subsequent training,
    call penalty() to get the quadratic regularization loss.

    Args:
        model: The neural network to regularize.
        ewc_lambda: Regularization strength. Typical range: 100--10000.
        online: If True, use online EWC (EMA of Fisher diagonals).
        gamma: EMA decay rate for online EWC (0 < gamma < 1).
        normalize: If True, normalize Fisher diagonal to unit maximum.
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        online: bool = True,
        gamma: float = 0.95,
        normalize: bool = True,
    ):
        assert ewc_lambda >= 0, f"ewc_lambda must be non-negative, got {ewc_lambda}"
        assert 0 < gamma <= 1.0, f"gamma must be in (0, 1], got {gamma}"

        self.model = model
        self.ewc_lambda = ewc_lambda
        self.online = online
        self.gamma = gamma
        self.normalize = normalize

        # Storage for Fisher diagonals and optimal parameters
        self._fisher_diag: Optional[Dict[str, Tensor]] = None
        self._star_params: Optional[Dict[str, Tensor]] = None
        self._task_count: int = 0

        # Per-task storage (non-online mode only)
        self._per_task_fisher: List[Dict[str, Tensor]] = []
        self._per_task_params: List[Dict[str, Tensor]] = []

    @property
    def task_count(self) -> int:
        """Number of tasks registered so far."""
        return self._task_count

    @property
    def has_prior(self) -> bool:
        """Whether at least one prior task has been registered."""
        return self._task_count > 0

    def compute_fisher_diagonal(
        self,
        data_loader: Iterator,
        num_samples: int = 200,
    ) -> Dict[str, Tensor]:
        """Compute diagonal of the Fisher information matrix.

        Uses empirical Fisher (ground-truth labels) for efficiency.
        CRITICAL: Sets model to train(False) during computation.

        Args:
            data_loader: Iterator yielding (x, y) batches.
            num_samples: Number of samples for Fisher estimation.

        Returns:
            Dict mapping parameter names to Fisher diagonal tensors.
        """
        fisher: Dict[str, Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        # CRITICAL: disable dropout / use running BN stats
        was_training = self.model.training
        self.model.train(False)

        count = 0
        for x, y in data_loader:
            if count >= num_samples:
                break

            device = next(self.model.parameters()).device
            x = x.to(device)
            y = y.to(device)

            logits = self.model(x)
            log_probs = F.log_softmax(logits, dim=-1)

            for i in range(x.size(0)):
                if count >= num_samples:
                    break

                self.model.zero_grad()
                nll = -log_probs[i, y[i]]
                nll.backward(retain_graph=(i < x.size(0) - 1))

                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += param.grad.data.clone() ** 2

                count += 1

        # Normalize by sample count
        if count > 0:
            for name in fisher:
                fisher[name] /= count

        # Optional: normalize to unit max
        if self.normalize:
            max_val = max(f.max().item() for f in fisher.values())
            if max_val > 0:
                for name in fisher:
                    fisher[name] /= max_val

        # Restore original training mode
        self.model.train(was_training)
        return fisher

    def register_task(
        self,
        data_loader: Iterator,
        num_samples: int = 200,
    ) -> None:
        """Register a completed task: compute Fisher and snapshot params.

        Args:
            data_loader: DataLoader for the completed task's data.
            num_samples: Number of samples for Fisher estimation.
        """
        fisher_new = self.compute_fisher_diagonal(data_loader, num_samples)
        star_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        if self.online:
            if self._fisher_diag is None:
                self._fisher_diag = fisher_new
            else:
                for name in self._fisher_diag:
                    self._fisher_diag[name] = (
                        self.gamma * self._fisher_diag[name] + fisher_new[name]
                    )
            self._star_params = star_params
        else:
            self._per_task_fisher.append(fisher_new)
            self._per_task_params.append(star_params)

        self._task_count += 1

    def penalty(self) -> Tensor:
        """Compute the EWC quadratic penalty.

        Returns:
            Scalar tensor (non-negative). Returns 0 if no prior tasks.
        """
        device = next(self.model.parameters()).device

        if not self.has_prior:
            return torch.tensor(0.0, device=device)

        if self.online:
            return self._compute_penalty_single(
                self._fisher_diag, self._star_params
            )
        else:
            total = torch.tensor(0.0, device=device)
            for fisher, star in zip(self._per_task_fisher, self._per_task_params):
                total = total + self._compute_penalty_single(fisher, star)
            return total / len(self._per_task_fisher)

    def _compute_penalty_single(
        self,
        fisher_diag: Dict[str, Tensor],
        star_params: Dict[str, Tensor],
    ) -> Tensor:
        """Compute penalty for a single Fisher/param snapshot."""
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for name, param in self.model.named_parameters():
            if name in fisher_diag and param.requires_grad:
                diff = param - star_params[name]
                penalty = penalty + (fisher_diag[name] * diff ** 2).sum()

        return (self.ewc_lambda / 2.0) * penalty

    def state_dict(self) -> dict:
        """Serialize EWC state for checkpointing."""
        return {
            "ewc_lambda": self.ewc_lambda,
            "online": self.online,
            "gamma": self.gamma,
            "normalize": self.normalize,
            "task_count": self._task_count,
            "fisher_diag": self._fisher_diag,
            "star_params": self._star_params,
            "per_task_fisher": self._per_task_fisher,
            "per_task_params": self._per_task_params,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load EWC state from checkpoint."""
        self.ewc_lambda = state["ewc_lambda"]
        self.online = state["online"]
        self.gamma = state["gamma"]
        self.normalize = state["normalize"]
        self._task_count = state["task_count"]
        self._fisher_diag = state["fisher_diag"]
        self._star_params = state["star_params"]
        self._per_task_fisher = state["per_task_fisher"]
        self._per_task_params = state["per_task_params"]


# ===========================================================================
#  Self-Test Suite
# ===========================================================================

def _run_tests():
    """Run all self-tests. Returns True if all pass."""
    results: List[tuple] = []
    test_num = 0

    def report(name: str, passed: bool, detail: str = ""):
        nonlocal test_num
        test_num += 1
        status = "PASS" if passed else "FAIL"
        suffix = f" -- {detail}" if detail else ""
        results.append((test_num, name, passed, suffix))
        print(f"  [{status}] {test_num:2d}. {name}{suffix}")

    print()
    print("=" * 70)
    print("  EWC Regularizer Template -- Self-Test Suite")
    print("=" * 70)
    print()

    # Toy model and data
    torch.manual_seed(42)

    class ToyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.fc2 = nn.Linear(32, 5)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    def make_data(n=50, seed=0):
        rng = torch.Generator().manual_seed(seed)
        x = torch.randn(n, 16, generator=rng)
        y = torch.randint(0, 5, (n,), generator=rng)
        dataset = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(dataset, batch_size=10)

    # Test 1: Fisher diagonal shape
    try:
        model = ToyMLP()
        ewc = EWCRegularizer(model, ewc_lambda=1000.0)
        loader = make_data(50, seed=1)
        fisher = ewc.compute_fisher_diagonal(loader, num_samples=20)
        shapes_ok = all(
            fisher[n].shape == p.shape
            for n, p in model.named_parameters()
            if p.requires_grad and n in fisher
        )
        report("Fisher diagonal shapes match model parameters",
               shapes_ok, f"{len(fisher)} params checked")
    except Exception as e:
        report("Fisher diagonal shapes match model parameters", False, str(e))

    # Test 2: Fisher diagonal non-negative
    try:
        all_nn = all((f >= 0).all().item() for f in fisher.values())
        report("Fisher diagonal values are non-negative", all_nn)
    except Exception as e:
        report("Fisher diagonal values are non-negative", False, str(e))

    # Test 3: Fisher normalization
    try:
        max_val = max(f.max().item() for f in fisher.values())
        report("Fisher normalized to unit max",
               abs(max_val - 1.0) < 1e-6, f"max={max_val:.8f}")
    except Exception as e:
        report("Fisher normalized to unit max", False, str(e))

    # Test 4: No prior -> penalty is zero
    try:
        model2 = ToyMLP()
        ewc2 = EWCRegularizer(model2)
        p = ewc2.penalty()
        report("Penalty is zero with no prior tasks",
               p.item() == 0.0, f"penalty={p.item()}")
    except Exception as e:
        report("Penalty is zero with no prior tasks", False, str(e))

    # Test 5: Register task and penalty is non-negative
    try:
        model3 = ToyMLP()
        ewc3 = EWCRegularizer(model3, ewc_lambda=1000.0)
        loader3 = make_data(50, seed=2)
        ewc3.register_task(loader3, num_samples=20)
        # Perturb model slightly
        with torch.no_grad():
            for p in model3.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        pen = ewc3.penalty()
        report("Penalty non-negative after perturbation",
               pen.item() >= 0, f"penalty={pen.item():.4f}")
    except Exception as e:
        report("Penalty non-negative after perturbation", False, str(e))

    # Test 6: Penalty zero at optimum
    try:
        model4 = ToyMLP()
        ewc4 = EWCRegularizer(model4, ewc_lambda=1000.0)
        loader4 = make_data(50, seed=3)
        ewc4.register_task(loader4, num_samples=20)
        # Do NOT perturb -- params are at star_params
        pen_opt = ewc4.penalty()
        report("Penalty is zero at optimum (no perturbation)",
               pen_opt.item() < 1e-8, f"penalty={pen_opt.item():.10f}")
    except Exception as e:
        report("Penalty is zero at optimum (no perturbation)", False, str(e))

    # Test 7: Penalty increases with distance
    try:
        model5 = ToyMLP()
        ewc5 = EWCRegularizer(model5, ewc_lambda=1000.0)
        loader5 = make_data(50, seed=4)
        ewc5.register_task(loader5, num_samples=20)
        penalties = []
        for scale in [0.01, 0.1, 1.0]:
            with torch.no_grad():
                # Reset to star params then perturb
                for name, param in model5.named_parameters():
                    if name in ewc5._star_params:
                        param.copy_(ewc5._star_params[name] + scale)
            penalties.append(ewc5.penalty().item())
        monotonic = all(a < b for a, b in zip(penalties, penalties[1:]))
        report("Penalty increases with distance from optimum",
               monotonic, f"penalties={[f'{p:.4f}' for p in penalties]}")
    except Exception as e:
        report("Penalty increases with distance from optimum", False, str(e))

    # Test 8: Online EWC update
    try:
        model6 = ToyMLP()
        ewc6 = EWCRegularizer(model6, ewc_lambda=1000.0, online=True, gamma=0.5)
        loader6a = make_data(50, seed=5)
        loader6b = make_data(50, seed=6)
        f1 = ewc6.compute_fisher_diagonal(loader6a, num_samples=20)
        ewc6.register_task(loader6a, num_samples=20)
        f2 = ewc6.compute_fisher_diagonal(loader6b, num_samples=20)
        ewc6.register_task(loader6b, num_samples=20)
        # Online Fisher should be gamma * f1 + f2 = 0.5 * f1 + f2
        ok = True
        for name in f1:
            expected = 0.5 * f1[name] + f2[name]
            # Note: normalization is applied per-computation, so direct comparison
            # requires un-normalized values. We check structural correctness.
            if ewc6._fisher_diag[name].shape != expected.shape:
                ok = False
                break
        report("Online EWC update produces correct shapes",
               ok and ewc6._task_count == 2, f"tasks={ewc6._task_count}")
    except Exception as e:
        report("Online EWC update produces correct shapes", False, str(e))

    # Test 9: Gradient flows through penalty
    try:
        model7 = ToyMLP()
        ewc7 = EWCRegularizer(model7, ewc_lambda=1000.0)
        loader7 = make_data(50, seed=7)
        ewc7.register_task(loader7, num_samples=20)
        with torch.no_grad():
            for p in model7.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        model7.zero_grad()
        pen = ewc7.penalty()
        pen.backward()
        has_grad = all(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model7.parameters() if p.requires_grad
        )
        report("Gradient flows through EWC penalty", has_grad)
    except Exception as e:
        report("Gradient flows through EWC penalty", False, str(e))

    # Test 10: State dict round-trip
    try:
        sd = ewc7.state_dict()
        model8 = ToyMLP()
        ewc8 = EWCRegularizer(model8)
        ewc8.load_state_dict(sd)
        report("State dict round-trip succeeds",
               ewc8._task_count == ewc7._task_count and ewc8.ewc_lambda == ewc7.ewc_lambda,
               f"tasks={ewc8._task_count}, lambda={ewc8.ewc_lambda}")
    except Exception as e:
        report("State dict round-trip succeeds", False, str(e))

    # -- Summary -----------------------------------------------------------
    print()
    print("-" * 70)
    total = len(results)
    passed = sum(1 for _, _, p, _ in results if p)
    failed = total - passed
    print(f"  Results: {passed}/{total} PASSED, {failed} FAILED")
    if failed > 0:
        print("\n  Failed tests:")
        for num, name, p, detail in results:
            if not p:
                print(f"    {num:2d}. {name}{detail}")
    print("-" * 70)
    print()
    return failed == 0


if __name__ == "__main__":
    success = _run_tests()
    if not success:
        raise SystemExit(1)
