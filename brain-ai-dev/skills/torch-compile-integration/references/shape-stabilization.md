# Shape Stabilization Reference

## The Recompile Problem

`torch.compile` traces Python bytecode and captures a computation graph. The first time compiled code runs, it records the shapes (and sometimes values) of all tensors as **guards**. On subsequent calls, it checks these guards. If any guard fails, it triggers a **recompilation**.

Recompilation is expensive (seconds to minutes). For training workloads with variable sequence lengths (e.g., NLP data of varying lengths), naive usage leads to a recompile on nearly every batch.

---

## How Guards Work

When Dynamo traces a function with input `x` of shape `[2, 512]`, it installs guards like:

```
x.shape[0] == 2     # batch size
x.shape[1] == 512   # sequence length
x.dtype == torch.long
x.device.type == 'cuda'
```

On the next call with `x` of shape `[2, 347]`, the guard `x.shape[1] == 512` fails. Dynamo recompiles for the new shape `[2, 347]`.

**Recompile budget**: `torch._dynamo.config.cache_size_limit` (default: 8). After this many recompiles for a given frame (function call site), Dynamo gives up and runs that frame in eager mode permanently. You'll see a warning in the logs.

---

## Dynamic Shape Modes

### `dynamic=None` (default) — Recommended

```python
model = torch.compile(model)  # dynamic=None is the default
```

Starts static. On the first shape mismatch, Dynamo auto-generalizes the dimension to a symbolic size (installs a range guard instead of an equality guard). Subsequent calls within that range don't recompile.

**Behavior:**
1. First call with shape `[2, 512]` — compiles static.
2. Second call with shape `[2, 347]` — guard fails, Dynamo generalizes dim 1 to symbolic.
3. Third call with shape `[2, 891]` — symbolic guard passes, no recompile.

Best default behavior — gets static efficiency on the first call, adapts automatically.

### `dynamic=False` — Fully Static

```python
model = torch.compile(model, dynamic=False)
```

Installs strict equality guards on all dimensions. Any shape change forces recompile. Use only when:
- Shapes are truly fixed (fixed batch size AND fixed sequence length).
- You want maximum static specialization for peak performance.

Risk: With any shape variation (e.g., last batch in epoch is smaller), causes continuous recompiles.

### `dynamic=True` — Force Dynamic

```python
model = torch.compile(model, dynamic=True)
```

Forces all dimensions to be symbolic. The PyTorch documentation explicitly labels this as "not recommended" and "testing-oriented." Generates slower kernels because the compiler cannot use shape-specific optimizations (e.g., tile sizes, loop unrolling). Do not use in production.

---

## Strategy A: Bucketing (Preferred for LLM Training)

Bucketing groups variable-length sequences into a small set of fixed shapes. Instead of `N` unique shapes (one per unique length), you have only `K` shapes (one per bucket). For `K=4` buckets, you get at most 4 compiled specializations.

### Algorithm

```
Define bucket boundaries: [256, 512, 1024, 2048]

For each sequence of length L:
    If L <= 256: pad to 256
    Elif L <= 512: pad to 512
    Elif L <= 1024: pad to 1024
    Elif L <= 2048: pad to 2048
    Else: error (sequence too long) or extend buckets

Within each bucket: pad all sequences to the bucket boundary
```

### Implementation in DataLoader

```python
def bucket_collate_fn(batch, buckets=(256, 512, 1024, 2048), pad_token_id=0):
    """DataLoader collate_fn that pads to bucket boundaries."""
    input_ids = [item["input_ids"] for item in batch]
    max_len = max(len(ids) for ids in input_ids)

    # Find ceiling bucket
    bucket_size = next((b for b in sorted(buckets) if b >= max_len), max(buckets))

    # Pad all sequences to bucket_size
    padded = torch.full((len(batch), bucket_size), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), bucket_size, dtype=torch.long)
    for i, ids in enumerate(input_ids):
        padded[i, :len(ids)] = torch.tensor(ids)
        attention_mask[i, :len(ids)] = 1

    return {"input_ids": padded, "attention_mask": attention_mask}
```

### Properties

- Compiled model sees only `len(buckets)` unique shapes → `len(buckets)` compiled specializations.
- Padding with `pad_token_id` is transparent to the model (attention mask handles it).
- Sort sequences by length before batching to maximize bucket utilization.
- Sort + bucket in the DataLoader sampler for maximum efficiency:

```python
class BucketedSampler(torch.utils.data.Sampler):
    """Sort by length, then group into batches of same bucket."""
    def __init__(self, lengths, batch_size, buckets=(256, 512, 1024, 2048)):
        self.indices_by_bucket = {}
        for idx, length in enumerate(lengths):
            bucket = next((b for b in sorted(buckets) if b >= length), max(buckets))
            self.indices_by_bucket.setdefault(bucket, []).append(idx)

    def __iter__(self):
        for bucket_indices in self.indices_by_bucket.values():
            random.shuffle(bucket_indices)
            yield from bucket_indices
```

---

## Strategy B: `torch._dynamo.mark_dynamic`

Pre-annotate specific tensor dimensions as dynamic before the first compiled call. Dynamo then installs range guards (not equality guards) for those dimensions from the start.

```python
import torch._dynamo

# Must be called BEFORE invoking compiled code
# NOT inside forward() — call on the input tensors each step
torch._dynamo.mark_dynamic(input_ids, dim=1, min=1, max=max_seq_len)
torch._dynamo.mark_dynamic(attention_mask, dim=1, min=1, max=max_seq_len)

# Now call compiled model — dim 1 is treated as symbolic
output = compiled_model(input_ids=input_ids, attention_mask=attention_mask)
```

**Important constraints:**
- Call `mark_dynamic` on the input tensors each forward step, before calling the compiled model.
- Do NOT call inside the forward method — the compiled graph doesn't execute Python that way.
- `min` and `max` constrain the range guard. The compiler can use these bounds for optimization.
- Only mark dimensions that actually vary. Batch dimension rarely varies in practice (use DataLoader with `drop_last=True`).

**When to use over bucketing:**
- When your framework doesn't easily support custom collate functions.
- When you have irregular batches that don't fit neatly into buckets.
- When you want to experiment with dynamic shapes without padding overhead.

---

## Recompile Budget and Monitoring

### Cache Size Limit

```python
import torch._dynamo
torch._dynamo.config.cache_size_limit = 16  # default is 8
```

Increase if you have legitimately many unique shapes (e.g., multi-modal with many resolution variants). Don't increase indiscriminately — each cached compilation uses memory.

### Logging Recompiles

```bash
# Show when and why recompiles happen
TORCH_LOGS=recompiles python train.py

# Show guard installation and shape analysis
TORCH_LOGS=dynamic python train.py

# Both together
TORCH_LOGS=recompiles,dynamic python train.py
```

Example output:
```
[recompiles] Recompiling function forward in model.py:45 due to:
    - tensor 'input_ids' size mismatch at index 1: expected 512, got 347
```

### Programmatic Recompile Counting

```python
import torch._dynamo

# Reset compilation state (useful in tests)
torch._dynamo.reset()

# Get compilation statistics (PyTorch 2.2+)
stats = torch._dynamo.utils.counters
print(f"Graph compilations: {stats['stats']['unique_graphs']}")
```

---

## Shape Padding Option

TorchInductor can automatically pad tensor shapes to GPU-friendly sizes:

```python
model = torch.compile(
    model,
    options={"shape_padding": True}
)
```

Pads dimensions to multiples of 8 (or other hardware-optimal alignment). Improves memory bandwidth efficiency and tensor core utilization. Small memory overhead. Recommended for transformer workloads.

This is **not** the same as bucketing — it's a low-level memory alignment optimization within a single shape, not a strategy for handling shape variation.

---

## Interaction: Bucketing + mark_dynamic Together

You can combine both strategies:
1. Use bucketing to reduce unique shapes from `N` to `K`.
2. Use `mark_dynamic` to further tell the compiler that within a bucket, the sequence dimension is bounded.

```python
# In collate_fn: pad to bucket boundary (e.g., 512)
batch = bucket_collate_fn(raw_batch)

# Before each compiled call: mark dim 1 as dynamic within bucket bounds
torch._dynamo.mark_dynamic(batch["input_ids"], dim=1, min=1, max=512)
torch._dynamo.mark_dynamic(batch["attention_mask"], dim=1, min=1, max=512)

output = compiled_model(**batch)
```

This gives maximum flexibility within each bucket while keeping the guard checks efficient.

---

## Summary: Strategy Selection

| Situation | Strategy | Notes |
|-----------|----------|-------|
| Fixed shapes (inference serving) | `dynamic=False` | Maximum static optimization |
| Variable seq len, LLM training | Bucketing (4-8 buckets) | Best performance/overhead balance |
| Variable shapes, few unique shapes | `dynamic=None` (auto) | Let Dynamo generalize automatically |
| Variable shapes, known range | `mark_dynamic` with min/max | Good control, avoids padding |
| Highly irregular shapes, debugging | `dynamic=True` | Do not use in production |

**Default recommendation for training**: Bucketing with 4 buckets `[256, 512, 1024, 2048]` + `dynamic=None`.
