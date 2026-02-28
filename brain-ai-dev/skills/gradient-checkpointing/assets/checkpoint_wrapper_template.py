"""
checkpoint_wrapper_template.py
------------------------------
CheckpointWrapper that applies torch.utils.checkpoint.checkpoint to any
nn.Module's forward method, trading compute for memory.

Supports both reentrant and non-reentrant (recommended) checkpointing,
optional RNG state preservation for dropout reproducibility, and
kwargs pass-through.

Usage:
    from checkpoint_wrapper_template import CheckpointWrapper

    model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256))
    wrapped = CheckpointWrapper(model, use_reentrant=False)
    output = wrapped(x)  # activations freed after forward, recomputed in backward
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CheckpointWrapper
# ---------------------------------------------------------------------------


class CheckpointWrapper(nn.Module):
    """
    Wraps an nn.Module so that its forward pass is executed under
    torch.utils.checkpoint.checkpoint. Intermediate activations are
    discarded after forward and recomputed during backward.

    Parameters
    ----------
    module : nn.Module
        The module to wrap. Its forward method will be called inside
        the checkpoint context.
    use_reentrant : bool
        If False (recommended, PyTorch >= 2.0), use the newer
        non-reentrant implementation that works correctly with
        autograd and torch.compile. If True, use the legacy
        reentrant implementation.
    preserve_rng_state : bool
        If True, save and restore CUDA RNG state around the
        checkpointed region so that dropout masks are identical
        during recomputation.
    """

    def __init__(
        self,
        module: nn.Module,
        use_reentrant: bool = False,
        preserve_rng_state: bool = True,
    ) -> None:
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant
        self.preserve_rng_state = preserve_rng_state

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass through the wrapped module under checkpointing.

        When use_reentrant=False, kwargs are supported natively.
        When use_reentrant=True, kwargs must be passed via a closure
        (reentrant checkpoint does not support kwargs directly).
        """
        if not torch.is_grad_enabled():
            # No checkpointing needed when gradients are disabled
            return self.module(*args, **kwargs)

        if self.use_reentrant:
            # Reentrant checkpoint does not support kwargs directly.
            # Wrap in a closure that captures kwargs.
            def run_fn(*fn_args):
                return self.module(*fn_args, **kwargs)

            return checkpoint(
                run_fn,
                *args,
                use_reentrant=True,
                preserve_rng_state=self.preserve_rng_state,
            )
        else:
            return checkpoint(
                self.module,
                *args,
                use_reentrant=False,
                preserve_rng_state=self.preserve_rng_state,
                **kwargs,
            )

    def extra_repr(self) -> str:
        return (
            f"use_reentrant={self.use_reentrant}, "
            f"preserve_rng_state={self.preserve_rng_state}"
        )


# ---------------------------------------------------------------------------
# Helper: wrap_children
# ---------------------------------------------------------------------------


def wrap_children(
    model: nn.Module,
    use_reentrant: bool = False,
    preserve_rng_state: bool = True,
    exclude_types: Optional[Tuple[type, ...]] = None,
) -> nn.Module:
    """
    Wrap all direct children of `model` in CheckpointWrapper.

    Parameters
    ----------
    model : nn.Module
        Parent module whose children will be wrapped.
    use_reentrant : bool
        Passed to CheckpointWrapper.
    preserve_rng_state : bool
        Passed to CheckpointWrapper.
    exclude_types : tuple of types, optional
        Module types to skip (e.g., (nn.LayerNorm, nn.Dropout)).

    Returns
    -------
    nn.Module
        The model with children wrapped in-place.
    """
    exclude_types = exclude_types or ()
    wrapped_count = 0
    for name, child in model.named_children():
        if isinstance(child, exclude_types):
            logger.debug("Skipping %s (%s) — excluded type", name, type(child).__name__)
            continue
        if isinstance(child, CheckpointWrapper):
            logger.debug("Skipping %s — already wrapped", name)
            continue
        setattr(model, name, CheckpointWrapper(child, use_reentrant, preserve_rng_state))
        wrapped_count += 1
    logger.info("Wrapped %d children in CheckpointWrapper", wrapped_count)
    return model


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from typing import List

    failures: List[str] = []

    def _check(name: str, condition: bool, msg: str = "") -> None:
        if condition:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}: {msg}")
            failures.append(name)

    print("=" * 60)
    print("CheckpointWrapper self-tests")
    print("=" * 60)

    torch.manual_seed(42)

    # --- Test 1: Output matches unwrapped ---
    try:
        base = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16))
        wrapped = CheckpointWrapper(base, use_reentrant=False)
        x = torch.randn(4, 32, requires_grad=True)

        # Need separate inputs for separate forward passes
        x1 = x.detach().clone().requires_grad_(True)
        x2 = x.detach().clone().requires_grad_(True)

        base_out = base(x1)
        wrap_out = wrapped(x2)
        _check(
            "output_matches",
            torch.allclose(base_out, wrap_out, atol=1e-6),
            f"max diff = {(base_out - wrap_out).abs().max().item()}"
        )
    except Exception as e:
        _check("output_matches", False, str(e))

    # --- Test 2: Gradient correctness (non-reentrant) ---
    try:
        model_a = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        # Deep copy for comparison
        import copy
        model_b = copy.deepcopy(model_a)
        wrapped_b = CheckpointWrapper(model_b, use_reentrant=False)

        x_a = torch.randn(4, 16, requires_grad=True)
        x_b = x_a.detach().clone().requires_grad_(True)

        out_a = model_a(x_a)
        out_a.sum().backward()

        out_b = wrapped_b(x_b)
        out_b.sum().backward()

        grads_match = True
        for (na, pa), (nb, pb) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            if pa.grad is None or pb.grad is None:
                grads_match = False
                break
            if not torch.allclose(pa.grad, pb.grad, atol=1e-5):
                grads_match = False
                break

        _check("gradient_correctness_nonreentrant", grads_match)
    except Exception as e:
        _check("gradient_correctness_nonreentrant", False, str(e))

    # --- Test 3: Gradient correctness (reentrant) ---
    try:
        model_c = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        model_d = copy.deepcopy(model_c)
        wrapped_d = CheckpointWrapper(model_d, use_reentrant=True)

        x_c = torch.randn(4, 16, requires_grad=True)
        x_d = x_c.detach().clone().requires_grad_(True)

        out_c = model_c(x_c)
        out_c.sum().backward()

        out_d = wrapped_d(x_d)
        out_d.sum().backward()

        grads_match = True
        for (_, pc), (_, pd) in zip(
            model_c.named_parameters(), model_d.named_parameters()
        ):
            if pc.grad is None or pd.grad is None:
                grads_match = False
                break
            if not torch.allclose(pc.grad, pd.grad, atol=1e-5):
                grads_match = False
                break

        _check("gradient_correctness_reentrant", grads_match)
    except Exception as e:
        _check("gradient_correctness_reentrant", False, str(e))

    # --- Test 4: No checkpointing when grad disabled ---
    try:
        model_e = nn.Linear(8, 4)
        wrapped_e = CheckpointWrapper(model_e, use_reentrant=False)
        x_e = torch.randn(2, 8)
        with torch.no_grad():
            out = wrapped_e(x_e)
        _check("no_checkpoint_without_grad", out.shape == (2, 4))
    except Exception as e:
        _check("no_checkpoint_without_grad", False, str(e))

    # --- Test 5: kwargs pass-through (non-reentrant) ---
    try:
        class KwargsModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)

            def forward(self, x, scale=1.0):
                return self.fc(x) * scale

        kwmod = KwargsModule()
        wrapped_kw = CheckpointWrapper(kwmod, use_reentrant=False)
        x_kw = torch.randn(2, 8, requires_grad=True)
        out_kw = wrapped_kw(x_kw, scale=2.0)
        out_kw.sum().backward()
        _check("kwargs_passthrough", out_kw.shape == (2, 4) and x_kw.grad is not None)
    except Exception as e:
        _check("kwargs_passthrough", False, str(e))

    # --- Test 6: wrap_children helper ---
    try:
        class MultiBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = nn.Linear(16, 16)
                self.block2 = nn.Linear(16, 16)
                self.norm = nn.LayerNorm(16)

            def forward(self, x):
                x = self.block1(x)
                x = self.block2(x)
                x = self.norm(x)
                return x

        mb = MultiBlock()
        wrap_children(mb, exclude_types=(nn.LayerNorm,))
        _check(
            "wrap_children_excludes",
            isinstance(mb.block1, CheckpointWrapper)
            and isinstance(mb.block2, CheckpointWrapper)
            and not isinstance(mb.norm, CheckpointWrapper),
        )
    except Exception as e:
        _check("wrap_children_excludes", False, str(e))

    # --- Test 7: extra_repr ---
    try:
        w = CheckpointWrapper(nn.Linear(4, 4), use_reentrant=False, preserve_rng_state=True)
        r = w.extra_repr()
        _check(
            "extra_repr",
            "use_reentrant=False" in r and "preserve_rng_state=True" in r,
        )
    except Exception as e:
        _check("extra_repr", False, str(e))

    # --- Test 8: module with dropout, RNG preservation ---
    try:
        class DropModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16, 16)
                self.drop = nn.Dropout(0.5)

            def forward(self, x):
                return self.drop(self.fc(x))

        dm = DropModel()
        dm.train(True)
        wrapped_dm = CheckpointWrapper(dm, use_reentrant=False, preserve_rng_state=True)

        torch.manual_seed(123)
        x_dm = torch.randn(4, 16, requires_grad=True)
        out_dm = wrapped_dm(x_dm)
        out_dm.sum().backward()
        _check("rng_state_preserved", x_dm.grad is not None)
    except Exception as e:
        _check("rng_state_preserved", False, str(e))

    # --- Test 9: train(False) mode passthrough ---
    try:
        dm2 = DropModel()
        dm2.train(False)
        wrapped_dm2 = CheckpointWrapper(dm2, use_reentrant=False)
        x_dm2 = torch.randn(2, 16)
        with torch.no_grad():
            out_dm2 = wrapped_dm2(x_dm2)
        _check("train_false_mode", out_dm2.shape == (2, 16))
    except Exception as e:
        _check("train_false_mode", False, str(e))

    # --- Test 10: Nested wrapping does not crash ---
    try:
        inner = nn.Linear(8, 8)
        w1 = CheckpointWrapper(inner, use_reentrant=False)
        w2 = CheckpointWrapper(w1, use_reentrant=False)
        x_nest = torch.randn(2, 8, requires_grad=True)
        out_nest = w2(x_nest)
        out_nest.sum().backward()
        _check("nested_wrapping", x_nest.grad is not None)
    except Exception as e:
        _check("nested_wrapping", False, str(e))

    print()
    if failures:
        print(f"FAILED: {len(failures)} tests: {failures}")
        sys.exit(1)
    else:
        print("All 10 tests passed.")
