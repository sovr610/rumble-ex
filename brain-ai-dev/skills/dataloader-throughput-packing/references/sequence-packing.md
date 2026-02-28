# Sequence Packing

## Overview

Sequence packing eliminates wasted compute on padding tokens. Two distinct modes serve different training scenarios: pretraining block building and SFT boundary-aware packing.

## Pretraining Block Builder

### Concept

Concatenate tokenized documents into fixed-length blocks of `target_seq_len` tokens. A rolling buffer accumulates tokens from multiple documents and emits full blocks.

### Algorithm

```python
def build_blocks(token_streams, target_seq_len):
    buffer = []
    doc_boundaries = []
    current_offset = 0

    for doc_tokens in token_streams:
        buffer.extend(doc_tokens)
        current_offset += len(doc_tokens)
        doc_boundaries.append(current_offset)

        while len(buffer) >= target_seq_len:
            block = buffer[:target_seq_len]
            buffer = buffer[target_seq_len:]
            # Adjust boundaries for the emitted block
            block_bounds = [b for b in doc_boundaries if b <= target_seq_len]
            doc_boundaries = [b - target_seq_len for b in doc_boundaries if b > target_seq_len]
            current_offset -= target_seq_len
            yield PackedBlock(
                input_ids=block,
                doc_boundaries=block_bounds,
                padding_count=0
            )

    # Handle remainder: pad to target_seq_len
    if buffer:
        padding_count = target_seq_len - len(buffer)
        block = buffer + [pad_token_id] * padding_count
        yield PackedBlock(
            input_ids=block,
            doc_boundaries=doc_boundaries,
            padding_count=padding_count
        )
```

### Properties

- Near-zero padding: only the final block may have padding.
- Document boundaries tracked for optional loss masking at cross-document positions.
- Order-preserving: documents appear in stream order within blocks.
- Memory-efficient: only one block's worth of buffer at a time.

## SFT Boundary-Aware Packing

### Concept

Pack variable-length fine-tuning examples into a single flattened sequence, while preserving attention isolation between examples using `cu_seqlens`.

### cu_seqlens Construction

`cu_seqlens` (cumulative sequence lengths) is an int32 tensor that marks where each example starts and ends in the flattened input:

```python
# Given examples of lengths [128, 256, 100]
# cu_seqlens = [0, 128, 384, 484]
# The i-th example spans input_ids[cu_seqlens[i] : cu_seqlens[i+1]]

lengths = [len(ex) for ex in examples]
cu_seqlens = [0]
for l in lengths:
    cu_seqlens.append(cu_seqlens[-1] + l)
cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
```

### position_ids Reset at Boundaries

Each packed example gets its own position IDs starting from 0:

```python
position_ids = []
for length in lengths:
    position_ids.extend(range(length))
position_ids = torch.tensor(position_ids, dtype=torch.long)

# Result for lengths [128, 256, 100]:
# [0, 1, 2, ..., 127, 0, 1, 2, ..., 255, 0, 1, 2, ..., 99]
```

This is critical for rotary position embeddings (RoPE) to work correctly with packed sequences.

### Label Masking Per Example

Each example's labels are preserved independently. System/prompt tokens can be masked with -100:

```python
all_labels = []
for ex in examples:
    # ex.labels has -100 for prompt tokens, real token IDs for completion
    all_labels.extend(ex.labels)
labels = torch.tensor(all_labels, dtype=torch.long)
```

### PackedBatch Output

```python
@dataclass
class PackedBatch:
    input_ids: torch.Tensor      # [total_tokens]
    cu_seqlens: torch.Tensor     # [num_examples + 1], int32
    position_ids: torch.Tensor   # [total_tokens]
    labels: torch.Tensor         # [total_tokens]
    max_seqlen: int              # max individual example length (for kernel dispatch)
```

## Attention Kernels for Packed Sequences

### PyTorch varlen_attn (torch >= 2.5)

```python
from torch.nn.attention.varlen import varlen_attn

# Inside the attention layer:
attn_output = varlen_attn(
    query,          # [total_tokens, num_heads, head_dim]
    key,            # [total_tokens, num_heads, head_dim]
    value,          # [total_tokens, num_heads, head_dim]
    cu_seq_q,       # [batch_size + 1], int32 - cumulative query lengths
    cu_seq_k,       # [batch_size + 1], int32 - cumulative key lengths
    max_q,          # int - max query sequence length
    max_k,          # int - max key sequence length
    is_causal=True
)
```

**Properties:**
- Compilable with `torch.compile` (registered as a custom op).
- Handles variable-length sequences without padding.
- Each subsequence defined by `cu_seqlens` attends only to itself.

### FlashAttention varlen_func

```python
from flash_attn import flash_attn_varlen_func

attn_output = flash_attn_varlen_func(
    q,                  # [total_tokens, num_heads, head_dim]
    k,                  # [total_tokens, num_heads, head_dim]
    v,                  # [total_tokens, num_heads, head_dim]
    cu_seqlens_q,       # [batch_size + 1], int32
    cu_seqlens_k,       # [batch_size + 1], int32
    max_seqlen_q,       # int
    max_seqlen_k,       # int
    causal=True
)
```

**Properties:**
- Fused CUDA kernel, no materialized attention matrix.
- O(N) memory instead of O(N^2).
- Requires `cu_seqlens` in int32 on the same device as q/k/v.

## HF DataCollatorWithFlattening

### Overview

Hugging Face's `DataCollatorWithFlattening` implements packing at the collator level. It concatenates examples and produces `position_ids` that reset at boundaries.

### Usage

```python
from transformers import DataCollatorWithFlattening

collator = DataCollatorWithFlattening()
# or with SFTTrainer:
trainer = SFTTrainer(
    model=model,
    data_collator=collator,
    # padding_free=True  # alternative flag in newer versions
)
```

### Performance Impact

| Dataset | Throughput Gain | Memory Reduction |
|---------|----------------|-----------------|
| FLAN (high variance) | 2x | ~20% |
| OrcaMath (low variance) | 1.4x | ~10% |

High-variance datasets (wide length distribution) benefit more because they waste more tokens on padding without packing.

## Bucketing

### Concept

Group samples by approximate length before packing. This reduces:
1. **Fragmentation**: Samples of similar length pack more efficiently.
2. **Shape churn**: `torch.compile` recompiles on new shapes; bucketing reduces unique shapes.

### Bucket Boundaries

```python
bucket_boundaries = [256, 512, 1024, 2048]

def assign_bucket(length: int) -> int:
    for boundary in bucket_boundaries:
        if length <= boundary:
            return boundary
    return bucket_boundaries[-1]  # cap at max
```

### Pack Within Bucket

```python
bucketed = defaultdict(list)
for example in examples:
    bucket = assign_bucket(len(example.input_ids))
    bucketed[bucket].append(example)

packed_batches = []
for bucket_len, bucket_examples in bucketed.items():
    # Pack examples within this bucket
    packed = packer.pack_sft(bucket_examples)
    packed_batches.append(packed)
```

### Benefits

- Within-bucket examples have similar lengths, so packing wastes less space.
- Each bucket produces sequences of a predictable total length, reducing `torch.compile` graph breaks.
- Typical buckets: `[256, 512, 1024, 2048]` cover most LLM fine-tuning scenarios.

## Padding Ratio Computation

```python
padding_ratio = padded_tokens / (padded_tokens + real_tokens)
```

Where:
- `padded_tokens` = number of padding tokens in the batch
- `real_tokens` = number of non-padding tokens in the batch

### Interpretation

| padding_ratio | Interpretation |
|---------------|---------------|
| 0.0 | Perfect packing (no waste) |
| < 0.05 | Excellent (pretraining blocks, good bucketing) |
| 0.05 - 0.20 | Good (well-bucketed SFT) |
| 0.20 - 0.50 | Poor (naive padding without packing) |
| > 0.50 | Severe waste (very heterogeneous lengths, no packing) |

## Effective vs Raw Tokens/sec

```python
raw_tokens_per_sec = global_batch_size * seq_len / step_time_sec
effective_tokens_per_sec = real_nonpad_tokens / step_time_sec
```

- **Raw**: Total tokens processed (including padding). This is what hardware counters see.
- **Effective**: Only real (non-padding) tokens. This is what the model actually learns from.

The ratio `effective / raw` equals `1 - padding_ratio`. Gate performance decisions on **effective** tokens/sec, not raw.

### Example

```
global_batch_size = 8, seq_len = 2048, step_time = 0.5s
raw = 8 * 2048 / 0.5 = 32,768 tokens/sec

If 25% of tokens are padding:
effective = 32,768 * 0.75 = 24,576 tokens/sec
```

Enabling packing might slow raw tokens/sec slightly (packing overhead) but dramatically increase effective tokens/sec by eliminating padding.
