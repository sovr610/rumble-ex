"""
custom_dist_ops_template.py
============================
Gradient-propagating distributed communication ops for V-JEPA 2.

Standard torch.distributed.all_gather does not propagate gradients.
These custom autograd.Function subclasses implement:

  - AllGather:    gather tensors from all ranks (backward: all_reduce + slice)
  - AllReduceSum: sum tensors across ranks     (backward: identity)
  - AllReduce:    average tensors across ranks (backward: identity)

All ops safely no-op in single-process mode (world_size == 1), making them
usable in unit tests and single-GPU training without any distributed setup.

Usage:
    global_z = AllGather.apply(local_z)          # [B, D] -> [world_size*B, D]
    global_sum = AllReduceSum.apply(local_val)    # sum across ranks
    global_avg = AllReduce.apply(local_metric)    # mean across ranks
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_world_size() -> int:
    """Return world_size, or 1 if distributed is not initialized."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def _get_rank() -> int:
    """Return rank, or 0 if distributed is not initialized."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


# ---------------------------------------------------------------------------
# AllGather
# ---------------------------------------------------------------------------

class AllGather(torch.autograd.Function):
    """
    Gathers tensors from all ranks and concatenates along dim 0.

    Forward:
        Input:  x of shape [N, ...]  (on each rank)
        Output: concatenated tensor of shape [world_size * N, ...]

        When world_size == 1, returns a clone of x (no communication).

    Backward:
        grad_output shape: [world_size * N, ...]
        1. All-reduce grad_output (sum) across ranks so every rank accumulates
           all incoming gradient contributions.
        2. Slice out this rank's local portion: [rank*N : (rank+1)*N]
        Output: grad shape [N, ...] matching the forward input.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:
        world_size = _get_world_size()
        rank = _get_rank()

        ctx.rank = rank
        ctx.world_size = world_size
        ctx.input_size = x.shape[0]
        ctx.save_for_backward(x)

        if world_size == 1:
            return x.clone()

        # Require contiguous tensor for NCCL
        x_contig = x.contiguous()

        # Allocate buffers for all ranks
        tensor_list = [torch.zeros_like(x_contig) for _ in range(world_size)]
        dist.all_gather(tensor_list, x_contig)

        return torch.cat(tensor_list, dim=0)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> Tensor:
        rank = ctx.rank
        world_size = ctx.world_size
        n = ctx.input_size

        if world_size == 1:
            return grad_output

        # All-reduce: every rank needs the full gradient so it can compute the
        # correct gradient w.r.t. its local input after slicing.
        grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)

        # Return only this rank's slice
        return grad_output[rank * n : (rank + 1) * n]


# ---------------------------------------------------------------------------
# AllReduceSum
# ---------------------------------------------------------------------------

class AllReduceSum(torch.autograd.Function):
    """
    All-reduces (sums) a tensor across all ranks.

    Every rank receives the same result: sum of all ranks' inputs.

    Forward:
        out = sum_over_all_ranks(x)
        (Identity when world_size == 1)

    Backward:
        grad_input = grad_output  (identity: d/dx of sum w.r.t. each input is 1)
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:
        if _get_world_size() == 1:
            return x.clone()

        out = x.clone().contiguous()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> Tensor:
        # Identity: gradient of a sum w.r.t. each addend is 1
        return grad_output


# ---------------------------------------------------------------------------
# AllReduce
# ---------------------------------------------------------------------------

class AllReduce(torch.autograd.Function):
    """
    All-reduces (averages) a tensor across all ranks.

    Every rank receives: mean of all ranks' inputs = sum / world_size.

    Forward:
        out = sum_over_all_ranks(x) / world_size
        (Identity when world_size == 1)

    Backward:
        grad_input = grad_output  (identity)

    Note:
        The division by world_size is absorbed into the forward pass.
        The backward is identity because dividing in forward already
        accounts for the scaling — autograd tracks the full op.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:
        world_size = _get_world_size()
        if world_size == 1:
            return x.clone()

        out = x.clone().contiguous()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        out = out / world_size
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> Tensor:
        return grad_output


# ---------------------------------------------------------------------------
# Convenience functional wrappers
# ---------------------------------------------------------------------------

def all_gather(x: Tensor) -> Tensor:
    """Functional wrapper for AllGather.apply(x)."""
    return AllGather.apply(x)


def all_reduce_sum(x: Tensor) -> Tensor:
    """Functional wrapper for AllReduceSum.apply(x)."""
    return AllReduceSum.apply(x)


def all_reduce(x: Tensor) -> Tensor:
    """Functional wrapper for AllReduce.apply(x)."""
    return AllReduce.apply(x)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Custom distributed ops self-tests (single-process mode)")
    print("=" * 60)

    # In single-process mode (world_size == 1), all ops are identity functions.
    # This validates forward shapes and gradient flow without requiring NCCL.

    # ------------------------------------------------------------------
    # Test 1: AllGather forward is identity for world_size=1
    # ------------------------------------------------------------------
    print("\n[Test 1] AllGather forward (world_size=1 identity)")

    shapes = [(4, 128), (1, 512), (8, 32, 64), (2,)]
    for shape in shapes:
        x = torch.randn(*shape)
        y = AllGather.apply(x)
        assert y.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {y.shape}"
        assert torch.allclose(y, x), "AllGather should be identity for world_size=1"
    print(f"  Tested shapes: {shapes}  PASS")

    # ------------------------------------------------------------------
    # Test 2: AllGather backward shape is correct
    # ------------------------------------------------------------------
    print("\n[Test 2] AllGather backward shape")

    for shape in shapes:
        x = torch.randn(*shape, requires_grad=True)
        y = AllGather.apply(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should not be None"
        assert x.grad.shape == x.shape, (
            f"Grad shape mismatch: expected {x.shape}, got {x.grad.shape}"
        )
    print(f"  Backward shapes correct for {shapes}  PASS")

    # ------------------------------------------------------------------
    # Test 3: AllGather backward values are correct (world_size=1)
    # ------------------------------------------------------------------
    print("\n[Test 3] AllGather backward values (world_size=1)")

    x = torch.randn(4, 8, requires_grad=True)
    y = AllGather.apply(x)
    # Upstream grad of all ones
    upstream = torch.ones_like(y)
    y.backward(upstream)
    assert torch.allclose(x.grad, upstream), "Grad should equal upstream for world_size=1"
    print("  Grad values correct  PASS")

    # ------------------------------------------------------------------
    # Test 4: AllGather gradient does not block (non-zero grad)
    # ------------------------------------------------------------------
    print("\n[Test 4] AllGather gradient does not block")

    x = torch.randn(3, 5, requires_grad=True)
    y = AllGather.apply(x)
    (y * 2.0).sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0, "Grad should be non-zero"
    print("  Non-zero gradient confirmed  PASS")

    # ------------------------------------------------------------------
    # Test 5: AllReduceSum forward is identity for world_size=1
    # ------------------------------------------------------------------
    print("\n[Test 5] AllReduceSum forward (world_size=1 identity)")

    for shape in shapes:
        x = torch.randn(*shape)
        y = AllReduceSum.apply(x)
        assert y.shape == x.shape
        assert torch.allclose(y, x)
    print(f"  Identity confirmed for {shapes}  PASS")

    # ------------------------------------------------------------------
    # Test 6: AllReduceSum backward is identity
    # ------------------------------------------------------------------
    print("\n[Test 6] AllReduceSum backward (identity)")

    x = torch.randn(4, 4, requires_grad=True)
    y = AllReduceSum.apply(x)
    upstream = torch.randn_like(y)
    y.backward(upstream)
    assert x.grad is not None
    assert torch.allclose(x.grad, upstream), "AllReduceSum backward should be identity"
    print("  Identity backward confirmed  PASS")

    # ------------------------------------------------------------------
    # Test 7: AllReduce forward is identity for world_size=1
    # ------------------------------------------------------------------
    print("\n[Test 7] AllReduce forward (world_size=1 identity)")

    for shape in shapes:
        x = torch.randn(*shape)
        y = AllReduce.apply(x)
        assert y.shape == x.shape
        assert torch.allclose(y, x), f"AllReduce should be identity for world_size=1, shape={shape}"
    print(f"  Identity confirmed for {shapes}  PASS")

    # ------------------------------------------------------------------
    # Test 8: AllReduce backward is identity
    # ------------------------------------------------------------------
    print("\n[Test 8] AllReduce backward (identity)")

    x = torch.randn(3, 7, requires_grad=True)
    y = AllReduce.apply(x)
    upstream = torch.randn_like(y)
    y.backward(upstream)
    assert x.grad is not None
    assert torch.allclose(x.grad, upstream), "AllReduce backward should be identity"
    print("  Identity backward confirmed  PASS")

    # ------------------------------------------------------------------
    # Test 9: grad_fn is not None (ops are differentiable)
    # ------------------------------------------------------------------
    print("\n[Test 9] grad_fn present (ops are differentiable)")

    x = torch.randn(2, 4, requires_grad=True)
    assert AllGather.apply(x).grad_fn is not None, "AllGather should have grad_fn"
    assert AllReduceSum.apply(x).grad_fn is not None, "AllReduceSum should have grad_fn"
    assert AllReduce.apply(x).grad_fn is not None, "AllReduce should have grad_fn"
    print("  All grad_fns present  PASS")

    # ------------------------------------------------------------------
    # Test 10: Functional wrappers work
    # ------------------------------------------------------------------
    print("\n[Test 10] Functional wrappers")

    x = torch.randn(4, 16, requires_grad=True)
    y1 = all_gather(x)
    y2 = all_reduce_sum(x)
    y3 = all_reduce(x)
    for y, name in [(y1, "all_gather"), (y2, "all_reduce_sum"), (y3, "all_reduce")]:
        assert y.shape == x.shape, f"{name} wrapper shape mismatch"
        assert y.grad_fn is not None, f"{name} wrapper should be differentiable"
    print("  Functional wrappers correct  PASS")

    print("\n" + "=" * 60)
    print("All custom distributed ops self-tests PASSED")
    print("=" * 60)
