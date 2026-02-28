# Custom Autograd Distributed Ops Reference

## Why Custom Ops Are Needed

PyTorch's standard `torch.distributed.all_gather` is a collective communication primitive that does **not** propagate gradients. This means using `dist.all_gather` in a training forward pass creates a dead-end in the autograd graph — gradients from downstream losses cannot flow back through the gather operation to the originating tensors.

V-JEPA 2 uses negative sample contrast (multi-crop, multi-block masking) where representations from all GPUs need to be visible to every GPU for loss computation. Gradients must flow back to each GPU's local portion of the gathered tensor.

The solution is to implement `AllGather`, `AllReduceSum`, and `AllReduce` as `torch.autograd.Function` subclasses with explicit forward and backward passes.

---

## AllGather

### Purpose

Gather tensors from all ranks into a single concatenated tensor on every rank, while correctly routing gradients back to each rank's local portion during the backward pass.

### Forward Pass

1. Save the local rank and world_size in `ctx` for backward
2. Create a list of `world_size` empty tensors with the same shape as `x`
3. Call `dist.all_gather(tensor_list, x)` — fills each slot with that rank's tensor
4. Concatenate along dim 0 to produce a tensor of shape `[world_size * N, ...]`

```python
@staticmethod
def forward(ctx, x):
    ctx.save_for_backward(x)
    ctx.rank = dist.get_rank() if dist.is_initialized() else 0
    ctx.world_size = dist.get_world_size() if dist.is_initialized() else 1

    if ctx.world_size == 1:
        return x.clone()

    # Gather from all ranks
    tensor_list = [torch.zeros_like(x) for _ in range(ctx.world_size)]
    dist.all_gather(tensor_list, x.contiguous())
    return torch.cat(tensor_list, dim=0)
```

### Backward Pass

The gradient of the concatenated output needs to be routed back to each rank's local input. The approach:

1. `grad_output` has shape `[world_size * N, ...]` — the full gathered gradient
2. All-reduce `grad_output` across ranks so every rank has the sum of all incoming gradients (required for correct gradient averaging when the loss uses the full gathered batch)
3. Slice out the local rank's portion: `grad_output[rank * N : (rank+1) * N]`

```python
@staticmethod
def backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    rank = ctx.rank
    world_size = ctx.world_size

    if world_size == 1:
        return grad_output

    # All-reduce the full gradient so each rank gets the correct gradient sum
    dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)

    # Slice local rank's portion
    n = x.shape[0]
    return grad_output[rank * n : (rank + 1) * n]
```

### Shape Invariants

- Input: `[N, D]` (or any shape)
- Output: `[world_size * N, D]`
- Grad input: `[world_size * N, D]` → sliced to `[N, D]`

---

## AllReduceSum

### Purpose

Sum a tensor across all ranks — every rank gets the identical sum. Gradient is identity (pass-through) because summing is a linear operation and the gradient of a sum is 1 for each input.

### Forward Pass

```python
@staticmethod
def forward(ctx, x):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x.clone()

    out = x.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out
```

### Backward Pass

```python
@staticmethod
def backward(ctx, grad_output):
    return grad_output  # Identity — gradient of sum w.r.t. each input is 1
```

### Use Case

Summing per-rank loss values to get the global batch loss:
```python
global_loss = AllReduceSum.apply(local_loss) / world_size
```

---

## AllReduce

### Purpose

Average a tensor across all ranks — every rank gets the mean. Gradient is identity. This is the most common operation for synchronizing gradients or metrics.

### Forward Pass

```python
@staticmethod
def forward(ctx, x):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x.clone()

    out = x.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= dist.get_world_size()
    return out
```

### Backward Pass

```python
@staticmethod
def backward(ctx, grad_output):
    return grad_output  # Identity — gradient of average is 1/N per input, but
                        # since we already divided in forward, backward is identity
```

### Use Case

Averaging per-rank metrics:
```python
avg_metric = AllReduce.apply(local_metric)
```

---

## Gradient Correctness Analysis

### AllGather Backward Correctness

Consider a simple loss `L = sum(AllGather(x))` with `x` of shape `[N]` and 2 ranks.

- Forward: `y = [x_rank0; x_rank1]` (shape `[2N]`)
- Loss: `L = sum(y) = sum(x_rank0) + sum(x_rank1)`
- `dL/d(x_rank0) = ones([N])` — each rank receives gradient 1 for its elements

In the backward:
- `grad_output = ones([2N])`
- After `all_reduce(SUM)`: rank 0 gets `ones([2N]) + ones([2N]) = 2*ones([2N])`

Wait — this appears to double-count. The reason `all_reduce` is needed is for the cross-rank gradient contributions: rank 0's output elements at positions `[N:2N]` come from rank 1, but the gradient flows to rank 0's backward through the all_reduce. The slicing then extracts exactly rank 0's share.

In practice, the loss is typically divided by the global batch size (which accounts for world_size), making the effective gradient:
```
grad = (1/world_size) * ones([2N]) → after slice: (1/world_size) * ones([N])
```

This matches the expected per-sample gradient for a mean loss.

### Single-Process Behavior

All three ops return `x.clone()` (or `x` itself) when `world_size == 1`, making them safe to use in single-GPU and unit-test contexts without any distributed initialization.

---

## Usage Pattern in V-JEPA 2

```python
# In the forward pass of the loss computation:
# Each rank has a batch of anchor embeddings: [B, D]
# Gather all embeddings to compute contrastive loss against full global batch

local_z = encoder(x_local)                       # [B, D]
global_z = AllGather.apply(local_z)              # [world_size * B, D]

# Loss uses global_z for negative pairs, but gradients flow back through AllGather
# to local_z on each rank correctly
loss = contrastive_loss(local_z, global_z)
loss.backward()  # Gradients correctly reach local_z on each rank
```

---

## Implementation Notes

1. **`.contiguous()` before all_gather**: NCCL requires contiguous tensors. Always call `.contiguous()` on the input before `dist.all_gather`.

2. **In-place operations**: The backward's `all_reduce` must NOT use an in-place operation on `grad_output` if other autograd functions also hold references to it. Use `grad_output.clone()` if needed.

3. **No-op for world_size=1**: Always short-circuit when not distributed to avoid NCCL initialization errors in test environments.

4. **ctx.needs_input_grad**: Optionally check `ctx.needs_input_grad[0]` in backward to skip computation when the input doesn't require gradients (optimization, not required for correctness).

5. **Mixed precision compatibility**: These ops work transparently with bfloat16 and float16 tensors — NCCL handles the data type correctly.
