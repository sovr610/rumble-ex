# CPU Offload and Async Prefetch — Reference for Engram Conditional Memory Skill

This document specifies the CPU offload mechanism, async prefetch pipeline, CUDA stream/event synchronization, coalescing strategy, benchmarking methodology, and distributed sharding compatibility for the Engram memory subsystem. Use this as the canonical reference when implementing, auditing, or debugging `OffloadableEmbedding`, `PrefetchPlan`, and the retrieval scheduler in `brain_ai/memory/hash_embedding.py`.

---

## 1. Overview

Engram's hash-based addressing creates a unique system-level opportunity: because embedding indices are computed deterministically from input tokens alone (not from hidden states), the index computation for layer L can be performed before layer L's forward pass begins. This means embedding rows can be prefetched from host (CPU) memory to device (GPU) memory and overlapped with the compute of earlier layers.

This is the key architectural insight that distinguishes Engram from standard embedding lookups:

```
Standard Embedding:    hidden_state -> compute indices -> gather (serial, on-device)
Engram (on-device):    input_ids -> hash indices -> gather (serial, but index is cheap)
Engram (offload):      input_ids -> hash indices -> prefetch(async) -> ... compute ... -> consume
```

The hash function is pure integer arithmetic on `input_ids`:

```python
# Hash computation depends ONLY on input tokens and frozen coefficients
hash_id = (sum(c_i * x_i for i in range(n)) ^ seed) % table_size
```

No gradients, no hidden states, no layer outputs are needed. This means:

1. All hash IDs for all layers can be computed upfront in a single pass.
2. The resulting IDs can be used to schedule asynchronous PCIe transfers.
3. The transfers overlap with transformer/SNN compute on the default CUDA stream.

### Why This Matters at Scale

At production scale (7B parameter model), Engram tables can reach tens of millions of rows. A single table with 10M rows at 256 dimensions in fp16 consumes ~5.1 GB. With multiple N-gram orders and heads, the total can exceed GPU memory. CPU offload with async prefetch lets the system access these massive tables without reserving GPU memory for the full table.

| Scale | Table Rows | Embedding Dim | dtype | Memory per Table | Total (4 orders x 2 heads) |
|---|---|---|---|---|---|
| Minimal (tests) | 131,071 | 256 | fp32 | ~134 MB | ~1.1 GB |
| Dev | 1,048,573 | 256 | fp16 | ~537 MB | ~4.3 GB |
| Production | 10,000,003 | 256 | fp16 | ~5.1 GB | ~40.8 GB |

At production scale, on-device mode is infeasible on a single GPU. Offload + prefetch is mandatory.

---

## 2. Two Runtime Modes

The `OffloadableEmbedding` class supports two distinct runtime modes, selected at construction time via the `weights_on_cpu` flag. Both modes produce identical outputs (within floating-point tolerance) but differ in memory placement and transfer strategy.

### 2.1 Mode 1: On-Device (Baseline)

```
weights_on_cpu = False
```

Embedding tables live entirely in GPU global memory. Retrieval is a standard `F.embedding()` gather operation. This is the simplest and fastest mode for small-to-medium tables.

**Data flow:**

```
input_ids  (B, T)          [GPU]
    |
    v
hash_ids   (B, T, H)       [GPU]     <- integer arithmetic, microseconds
    |
    v
F.embedding(table, ids)     [GPU]     <- coalesced global memory read
    |
    v
embeddings (B, T, H, D_e)  [GPU]
```

**Characteristics:**

| Property | Value |
|---|---|
| Latency | ~10-50 us per gather (depends on batch size) |
| Memory | Full table resident on GPU |
| Bandwidth | GPU HBM bandwidth (~2 TB/s on A100) |
| Use case | Training; inference with small tables |
| Gradient flow | Standard autograd through `nn.Embedding` |

**Implementation:**

```python
class OffloadableEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, weights_on_cpu=False, ...):
        if not weights_on_cpu:
            # Standard on-device embedding
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        if not self.weights_on_cpu:
            return self.embedding(ids)
```

### 2.2 Mode 2: CPU Offload + Async Prefetch

```
weights_on_cpu = True
```

Master weights reside in host (CPU) memory, optionally pinned for faster DMA transfers. Only the active rows needed for the current batch are transferred to GPU. The transfer can happen asynchronously on a dedicated CUDA stream, overlapping with other compute.

**Data flow:**

```
input_ids  (B, T)            [CPU or GPU]
    |
    v
hash_ids   (B, T, H)         [CPU or GPU]     <- computed upfront
    |
    v
unique_ids, inverse           [CPU]            <- coalescing
    |
    v
host_table[unique_ids]        [CPU, pinned]    <- gather from host memory
    |
    v  (async H2D on prefetch stream)
device_buffer                 [GPU]            <- only active rows
    |
    v
device_buffer[inverse]        [GPU]            <- reconstruct full tensor
    |
    v
embeddings (B, T, H, D_e)    [GPU]
```

**Characteristics:**

| Property | Value |
|---|---|
| Latency | 50-500 us per transfer (depends on row count, PCIe gen) |
| Memory (GPU) | Only active rows per batch (typically 0.1-5% of table) |
| Memory (CPU) | Full table in pinned host memory |
| Bandwidth | PCIe 4.0 x16: ~25 GB/s; PCIe 5.0 x16: ~50 GB/s |
| Use case | Inference with large tables; memory-constrained GPUs |
| Gradient flow | Supported but slower (H2D + D2H for grad) |

**When offload is beneficial:**

- Table does not fit in GPU memory (production scale)
- Multiple tables compete for GPU memory with model weights
- Inference-only workloads where training gradient overhead is not a concern
- Batch sizes are small enough that active rows are a small fraction of the table

**When offload is NOT beneficial:**

- Table fits comfortably in GPU memory
- Training with large batches (active rows approach full table)
- PCIe bandwidth is the bottleneck (many concurrent transfers)

---

## 3. OffloadableEmbedding API

The `OffloadableEmbedding` class is the unified interface for both runtime modes. It is a `torch.nn.Module` and participates in standard PyTorch model management (state_dict, to(), etc.).

### 3.1 Constructor

```python
class OffloadableEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weights_on_cpu: bool = False,
        storage_dtype: torch.dtype = torch.float16,
        pin_memory: bool = True,
    ):
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_embeddings` | `int` | required | Number of rows in the embedding table (should be prime for hash tables) |
| `embedding_dim` | `int` | required | Dimension of each embedding vector |
| `weights_on_cpu` | `bool` | `False` | If `True`, store master weights in host memory |
| `storage_dtype` | `torch.dtype` | `torch.float16` | Precision for host storage (fp16 or bf16 to save memory) |
| `pin_memory` | `bool` | `True` | If `True`, use pinned (page-locked) host memory for faster DMA |

**Construction behavior by mode:**

```python
def __init__(self, ...):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.weights_on_cpu = weights_on_cpu
    self.storage_dtype = storage_dtype
    self.compute_dtype = torch.float32  # always compute in fp32

    if weights_on_cpu:
        # Allocate on CPU in storage dtype
        weight_data = torch.zeros(
            num_embeddings, embedding_dim, dtype=storage_dtype
        )
        nn.init.normal_(weight_data, mean=0, std=0.02)

        if pin_memory and torch.cuda.is_available():
            weight_data = weight_data.pin_memory()

        # Register as buffer (not parameter) to avoid optimizer tracking
        # For training, wrap separately or use a custom optimizer hook
        self.register_buffer('weight_cpu', weight_data, persistent=True)

        # Prefetch state
        self._prefetch_event: Optional[torch.cuda.Event] = None
        self._prefetch_result: Optional[torch.Tensor] = None
        self._prefetch_inverse: Optional[torch.Tensor] = None
        self._prefetch_stream: Optional[torch.cuda.Stream] = None

        if torch.cuda.is_available():
            self._prefetch_stream = torch.cuda.Stream()
    else:
        # Standard on-device embedding
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
```

**Pinned memory details:**

Pinned (page-locked) memory bypasses the OS paging mechanism, allowing the CUDA runtime to DMA directly between host and device memory without an intermediate copy through a staging buffer. This roughly doubles H2D transfer throughput for large transfers.

```
Non-pinned:  CPU pageable -> CPU pinned staging -> GPU  (two copies)
Pinned:      CPU pinned -> GPU                          (one copy, DMA)
```

The cost of pinning is that the memory is non-swappable and reduces available host memory for other processes. For Engram tables this is acceptable because the tables are long-lived and accessed frequently.

### 3.2 lookup(ids) -> embeddings

Standard synchronous retrieval. In on-device mode, this is a direct `F.embedding()` call. In offload mode, this performs a synchronous H2D copy (no overlap).

```python
def lookup(self, ids: torch.Tensor) -> torch.Tensor:
    """
    Synchronous embedding lookup.

    Args:
        ids: (*, ) integer tensor of embedding indices.
             Any shape is supported; output will be (*, embedding_dim).

    Returns:
        embeddings: (*, embedding_dim) in compute_dtype on the same
                    device as ids (or CUDA if offloading).
    """
    if not self.weights_on_cpu:
        return self.embedding(ids)

    # Offload path: coalesce, transfer, reconstruct
    original_shape = ids.shape
    flat_ids = ids.reshape(-1)

    unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)

    # Gather from CPU table
    cpu_ids = unique_ids.cpu()
    host_rows = self.weight_cpu[cpu_ids]  # (num_unique, D) in storage_dtype

    # Transfer to device, cast to compute dtype
    device = ids.device if ids.is_cuda else torch.device('cuda')
    device_rows = host_rows.to(device=device, dtype=self.compute_dtype)

    # Reconstruct via inverse mapping
    result = device_rows[inverse]
    return result.reshape(*original_shape, self.embedding_dim)
```

**Shape contract:**

| Input | Shape | dtype | Device |
|---|---|---|---|
| `ids` | `(B, T, H)` or any `(*)` | `torch.long` | GPU or CPU |
| **Output** | `(B, T, H, D)` or `(*, D)` | `self.compute_dtype` | same as `ids` or GPU |

### 3.3 prefetch(ids, stream) -> None

Initiates an asynchronous H2D transfer on a dedicated CUDA stream. The transfer runs concurrently with compute on the default stream. Does nothing in on-device mode.

```python
def prefetch(
    self,
    ids: torch.Tensor,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """
    Asynchronously prefetch embeddings for given IDs.

    This method returns immediately. The actual transfer happens on
    the prefetch stream. Call consume_prefetched() later to get results.

    Args:
        ids: (*, ) integer tensor of embedding indices.
        stream: Optional CUDA stream to use. If None, uses internal stream.

    Raises:
        RuntimeError: If a previous prefetch has not been consumed.
    """
    if not self.weights_on_cpu:
        return  # No-op for on-device mode

    if self._prefetch_result is not None:
        raise RuntimeError(
            "Previous prefetch not consumed. Call consume_prefetched() first."
        )

    prefetch_stream = stream or self._prefetch_stream
    if prefetch_stream is None:
        raise RuntimeError("No CUDA stream available for prefetch")

    flat_ids = ids.reshape(-1)
    unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)
    cpu_ids = unique_ids.cpu()

    # Gather rows from host table (CPU operation)
    host_rows = self.weight_cpu[cpu_ids]  # (num_unique, D) in storage_dtype

    # Async transfer on prefetch stream
    with torch.cuda.stream(prefetch_stream):
        device_rows = host_rows.to(
            device=torch.device('cuda'),
            dtype=self.compute_dtype,
            non_blocking=True,
        )

        # Record event for synchronization
        event = torch.cuda.Event()
        event.record(prefetch_stream)

    # Store state for consumption
    self._prefetch_event = event
    self._prefetch_result = device_rows
    self._prefetch_inverse = inverse
    self._original_shape = ids.shape
```

**Critical invariant:** `prefetch()` must not block the calling (default) stream. All GPU operations happen on `prefetch_stream`. The only CPU-side work is the host table gather and the `torch.unique()` call, both of which are fast for typical batch sizes.

### 3.4 consume_prefetched() -> embeddings

Waits for the prefetch transfer to complete and returns the result. This is the synchronization point.

```python
def consume_prefetched(self) -> torch.Tensor:
    """
    Wait for prefetch to complete and return embeddings.

    Returns:
        embeddings: (*, embedding_dim) in compute_dtype on GPU.

    Raises:
        RuntimeError: If no prefetch is pending.
    """
    if self._prefetch_result is None:
        raise RuntimeError("No pending prefetch. Call prefetch() first.")

    # Wait for the async transfer to complete on the current stream
    if self._prefetch_event is not None:
        torch.cuda.current_stream().wait_event(self._prefetch_event)

    # Reconstruct full tensor via inverse mapping
    result = self._prefetch_result[self._prefetch_inverse]
    result = result.reshape(*self._original_shape, self.embedding_dim)

    # Clear state
    self._prefetch_event = None
    self._prefetch_result = None
    self._prefetch_inverse = None
    self._original_shape = None

    return result
```

**Timing discipline:**

```
Time ---->

Default stream:  [... layer L-2 compute ...][wait_event][consume + layer L compute ...]
Prefetch stream:            [H2D transfer][record event]

The overlap window is the time between prefetch() and consume_prefetched().
Wider window = more compute overlapped with transfer = better performance.
```

### 3.5 Storage dtype and Casting

Host tables are stored in `storage_dtype` (typically fp16 or bf16) to reduce host memory by 2x compared to fp32. On transfer to GPU, rows are cast to `compute_dtype` (fp32 by default) for numerical stability.

| Location | dtype | Purpose |
|---|---|---|
| Host (CPU) | `storage_dtype` (fp16/bf16) | Memory savings: 2 bytes/element vs 4 |
| Device (GPU) | `compute_dtype` (fp32) | Numerical stability for downstream ops |
| Gradient (if training) | fp32 | Standard autograd precision |

**Casting correctness:**

The cast from fp16 to fp32 is lossless (fp16 is a subset of fp32). The cast from bf16 to fp32 is also lossless. Therefore, no information is lost during the H2D transfer. The only precision difference versus on-device mode is if the on-device table is stored in fp32 and the host table is in fp16 -- the fp16 quantization at construction time introduces a one-time rounding error.

```python
# Correctness check: host fp16 -> device fp32 matches direct fp32 within tolerance
weight_fp32 = torch.randn(1000, 256)
weight_fp16 = weight_fp32.half()
restored = weight_fp16.float()

# This is the quantization error from the one-time conversion
max_error = (weight_fp32 - restored).abs().max()
# Typical: max_error ~ 0.001 for normally distributed weights
# Cosine similarity: > 0.9999
```

---

## 4. Coalescing Strategy

Before transferring embedding rows from host to device, duplicate IDs must be deduplicated to avoid wasting PCIe bandwidth. This process is called coalescing.

### 4.1 The Problem

In a typical batch with sequence length T and H hash heads across multiple N-gram orders, the total number of embedding lookups is `B * T * H_total`. Many of these IDs will collide (same N-gram appearing at different positions, or hash collisions mapping different N-grams to the same row). Transferring each ID individually would waste bandwidth on redundant rows.

**Example:**

```
Batch size B=4, sequence length T=512, total heads H=8
Total lookups: 4 * 512 * 8 = 16,384 IDs
Unique IDs (typical): ~6,000-10,000 (depending on vocabulary distribution)
Coalescing ratio: 6000/16384 = 0.37 (63% bandwidth savings)
```

### 4.2 Algorithm

```python
def _coalesce_ids(ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deduplicate IDs and compute inverse mapping.

    Args:
        ids: (N,) flat tensor of embedding indices (int64)

    Returns:
        unique_ids: (U,) tensor of unique indices, U <= N
        inverse: (N,) tensor such that unique_ids[inverse] == ids
    """
    unique_ids, inverse = torch.unique(ids, return_inverse=True)
    return unique_ids, inverse
```

**Reconstruction:**

```python
# After transferring unique rows to GPU:
fetched = host_table[unique_ids].to(device)  # (U, D)

# Reconstruct the full tensor using inverse mapping:
result = fetched[inverse]  # (N, D) -- each row is fetched[inverse[i]]
# Guarantee: result[i] == host_table[ids[i]] for all i
```

### 4.3 Coalescing Ratio

The coalescing ratio measures the effectiveness of deduplication:

```
coalescing_ratio = num_unique / num_total
```

| Ratio | Interpretation | Typical Scenario |
|---|---|---|
| 1.0 | No duplicates; no bandwidth savings | All IDs are distinct (unlikely) |
| 0.5 | 2x average reuse; 50% bandwidth savings | Moderate repetition |
| 0.1 | 10x average reuse; 90% bandwidth savings | Highly repetitive input |
| 0.01 | 100x average reuse | Degenerate case (e.g., padding-heavy batch) |

Lower is better. In practice, coalescing ratios of 0.3-0.6 are typical for natural language inputs with Engram's default table sizes.

### 4.4 Factors Affecting Coalescing

| Factor | Effect on Ratio | Why |
|---|---|---|
| Larger batch size | Lower ratio (better) | More opportunities for ID collision |
| Longer sequences | Lower ratio (better) | More N-gram repetition within sequences |
| Smaller table size | Lower ratio (better) | Hash collisions increase overlap |
| Higher N-gram order | Higher ratio (worse) | Longer N-grams are more unique |
| More hash heads | Higher ratio (worse) | Different heads map to different rows |
| Repetitive input text | Lower ratio (better) | Same N-grams produce same IDs |

### 4.5 Telemetry

Coalescing statistics should be reported per batch for monitoring:

```python
telemetry = {
    'coalesce_ratio': num_unique / num_total,       # float in [0, 1]
    'num_unique_ids': num_unique,                     # int
    'num_total_ids': num_total,                       # int
    'bandwidth_savings_pct': (1 - num_unique / num_total) * 100,  # percent
    'transfer_bytes': num_unique * embedding_dim * dtype_size,    # bytes
}
```

---

## 5. CUDA Stream/Event Synchronization

Async prefetch requires careful coordination between two CUDA streams: the default compute stream (where transformer/SNN layers execute) and a dedicated prefetch stream (where H2D transfers happen). Incorrect synchronization leads to either data races (consuming data before transfer completes) or unnecessary serialization (blocking the compute stream while waiting for transfer).

### 5.1 Stream and Event Primitives

```python
# Create a dedicated prefetch stream (do this once, at module construction)
prefetch_stream = torch.cuda.Stream()

# Create events for synchronization (lightweight, reusable)
transfer_done = torch.cuda.Event(enable_timing=False)
```

**Stream semantics:** Operations enqueued on different streams can execute concurrently on the GPU. Operations on the same stream execute in order. An event recorded on one stream can be waited on by another stream, creating a dependency edge.

**Event semantics:** `event.record(stream)` marks a point in `stream`'s execution. `other_stream.wait_event(event)` makes `other_stream` wait until `event`'s recorded point is reached.

### 5.2 Synchronization Workflow

The complete prefetch/consume cycle has four phases:

```
Phase 1: Compute hash IDs (default stream or CPU)
Phase 2: Launch async copy (prefetch stream)
Phase 3: Compute other layers (default stream, concurrent with Phase 2)
Phase 4: Wait and consume (default stream waits for prefetch stream)
```

Detailed implementation:

```python
# === Phase 1: Compute hash IDs ===
# This happens on the default stream or CPU.
# Hash IDs depend only on input_ids, so this can be done upfront.
hash_ids = multi_head_hash(input_ids)  # (B, T, H_total), int64

# === Phase 2: Launch async H2D transfer ===
flat_ids = hash_ids.reshape(-1)
unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)
cpu_ids = unique_ids.cpu()

# Gather rows on CPU (this is a CPU operation, not on any CUDA stream)
host_rows = weight_cpu[cpu_ids]  # (U, D) in storage_dtype, pinned memory

with torch.cuda.stream(prefetch_stream):
    # Async copy: pinned CPU -> GPU
    # non_blocking=True means this returns immediately on the CPU side
    device_rows = host_rows.to(
        device='cuda',
        dtype=torch.float32,
        non_blocking=True,
    )
    # Record an event marking "transfer complete"
    transfer_done = torch.cuda.Event()
    transfer_done.record(prefetch_stream)

# === Phase 3: Meanwhile, compute other layers on default stream ===
# The default stream does NOT wait for the prefetch stream.
# This is where the overlap happens.
hidden = transformer_layer_L_minus_2(hidden)
hidden = transformer_layer_L_minus_1(hidden)

# === Phase 4: Synchronize and consume ===
# Make the default stream wait for the transfer to finish
torch.cuda.current_stream().wait_event(transfer_done)

# Now device_rows is guaranteed to be valid on the default stream
embeddings = device_rows[inverse]  # Reconstruct full tensor
embeddings = embeddings.reshape(B, T, H_total, D)
```

### 5.3 Stream Diagram

```
Time ------->

Default stream:   [hash_ids][            layer L-2            ][     layer L-1     ][wait][consume + layer L]
                         \                                                             ^
                          \                                                           /
                           prefetch()                                    wait_event()
                                \                                       /
Prefetch stream:                 [  H2D transfer (async)  ][record event]

                  |<---------- overlap window ----------->|
```

The overlap window is the time between `prefetch()` returning and `wait_event()` being reached on the default stream. The wider this window (more compute between prefetch and consume), the more transfer latency is hidden.

### 5.4 Critical Rules

**Rule 1: Never synchronize too early.**

```python
# BAD: This destroys overlap
prefetch_stream.synchronize()  # Blocks CPU until transfer done
hidden = transformer_layer(hidden)  # Now this runs AFTER transfer

# GOOD: Use event-based waiting
torch.cuda.current_stream().wait_event(transfer_done)
# Only the GPU default stream waits; CPU continues immediately
```

**Rule 2: Never consume before synchronization.**

```python
# BAD: Data race -- device_rows may not be ready
device_rows = host_rows.to('cuda', non_blocking=True)
embeddings = device_rows[inverse]  # UNDEFINED BEHAVIOR

# GOOD: Wait for the event first
torch.cuda.current_stream().wait_event(transfer_done)
embeddings = device_rows[inverse]  # Safe
```

**Rule 3: Never reuse the prefetch buffer before consumption.**

```python
# BAD: Overwrites in-flight data
embedding.prefetch(ids_layer_5)
embedding.prefetch(ids_layer_6)  # Overwrites layer 5's buffer!

# GOOD: Consume before next prefetch
embedding.prefetch(ids_layer_5)
# ... compute ...
result_5 = embedding.consume_prefetched()
embedding.prefetch(ids_layer_6)
```

**Rule 4: Use `non_blocking=True` for all H2D transfers on the prefetch stream.**

```python
# BAD: Synchronous transfer (blocks prefetch stream AND prevents overlap)
device_rows = host_rows.to('cuda')

# GOOD: Non-blocking transfer (returns immediately on CPU)
device_rows = host_rows.to('cuda', non_blocking=True)
```

### 5.5 Error Handling and Deadlock Prevention

If the prefetch stream stalls (driver bug, memory error, etc.), the default stream will wait indefinitely at `wait_event()`. To prevent deadlocks in production:

```python
PREFETCH_TIMEOUT_MS = 5000  # 5 seconds -- generous for PCIe transfer

def consume_prefetched_safe(self, timeout_ms: int = PREFETCH_TIMEOUT_MS) -> torch.Tensor:
    """Consume with deadlock detection."""
    if self._prefetch_event is None:
        raise RuntimeError("No pending prefetch")

    # Query whether the event has completed
    # Note: event.query() returns True if all work before the event is done
    if not self._prefetch_event.query():
        # Not done yet -- wait with timeout
        start = torch.cuda.Event(enable_timing=True)
        start.record()

        torch.cuda.current_stream().wait_event(self._prefetch_event)
        torch.cuda.synchronize()  # Force completion for timeout check

        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        if elapsed_ms > timeout_ms:
            self._clear_prefetch_state()
            raise TimeoutError(
                f"Prefetch took {elapsed_ms:.1f}ms, exceeding "
                f"timeout of {timeout_ms}ms. Possible deadlock."
            )
    else:
        # Already done -- just synchronize the stream
        torch.cuda.current_stream().wait_event(self._prefetch_event)

    result = self._prefetch_result[self._prefetch_inverse]
    result = result.reshape(*self._original_shape, self.embedding_dim)
    self._clear_prefetch_state()
    return result

def _clear_prefetch_state(self):
    """Reset all prefetch state."""
    self._prefetch_event = None
    self._prefetch_result = None
    self._prefetch_inverse = None
    self._original_shape = None
```

---

## 6. Prefetch Scheduler / Retrieval Plan

The prefetch scheduler coordinates multiple `OffloadableEmbedding` tables across multiple layers. It precomputes all hash IDs upfront and schedules prefetch operations to maximize overlap with layer compute.

### 6.1 PrefetchPlan Data Structure

```python
@dataclass
class PrefetchPlan:
    """
    Precomputed retrieval plan for all Engram-augmented layers.

    Created once per forward pass from input_ids. Contains all hash IDs
    and a schedule for when to trigger prefetches.

    Attributes:
        layer_hash_ids: Dict mapping layer_id -> hash_ids tensor (B, T, H_total)
        layer_unique_ids: Dict mapping layer_id -> (unique_ids, inverse) tuple
        schedule: List of (trigger_at_layer, prefetch_for_layer) tuples
        prefetch_ahead: Number of layers ahead to start prefetch
        telemetry: Per-layer coalescing stats
    """
    layer_hash_ids: Dict[int, torch.Tensor]
    layer_unique_ids: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    schedule: List[Tuple[int, int]]
    prefetch_ahead: int
    telemetry: Dict[int, Dict[str, float]]
```

### 6.2 Plan Construction

```python
def precompute_retrieval_plan(
    input_ids: torch.Tensor,
    layer_ids: List[int],
    multi_head_hashers: Dict[int, MultiHeadHash],
    offload_config: 'OffloadConfig',
) -> PrefetchPlan:
    """
    Precompute all hash IDs and build a prefetch schedule.

    This function runs ONCE at the start of the forward pass.
    All computation here is integer arithmetic on input_ids -- no
    gradients, no hidden states, no GPU compute needed.

    Args:
        input_ids: (B, T) input token IDs (after tokenizer compression)
        layer_ids: sorted list of layer indices that have Engram augmentation
        multi_head_hashers: per-layer MultiHeadHash instances
        offload_config: OffloadConfig with prefetch_ahead_layers

    Returns:
        PrefetchPlan ready for use during the forward pass
    """
    layer_hash_ids = {}
    layer_unique_ids = {}
    telemetry = {}

    for layer_id in layer_ids:
        hasher = multi_head_hashers[layer_id]

        # Compute hash IDs -- pure integer arithmetic
        # This works identically on CPU or GPU
        hash_ids = hasher.hash_all_orders(input_ids)  # (B, T, H_total)
        layer_hash_ids[layer_id] = hash_ids

        # Precompute coalescing
        flat_ids = hash_ids.reshape(-1)
        unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)
        layer_unique_ids[layer_id] = (unique_ids, inverse)

        # Record telemetry
        num_total = flat_ids.numel()
        num_unique = unique_ids.numel()
        telemetry[layer_id] = {
            'coalesce_ratio': num_unique / max(num_total, 1),
            'num_unique': num_unique,
            'num_total': num_total,
        }

    # Build schedule: prefetch for layer L is triggered at layer L - prefetch_ahead
    prefetch_ahead = offload_config.prefetch_ahead_layers
    schedule = []
    for layer_id in layer_ids:
        trigger_at = layer_id - prefetch_ahead
        schedule.append((trigger_at, layer_id))

    # Sort by trigger point
    schedule.sort(key=lambda x: x[0])

    return PrefetchPlan(
        layer_hash_ids=layer_hash_ids,
        layer_unique_ids=layer_unique_ids,
        schedule=schedule,
        prefetch_ahead=prefetch_ahead,
        telemetry=telemetry,
    )
```

### 6.3 Schedule Semantics

The schedule is a list of `(trigger_at_layer, prefetch_for_layer)` tuples. When the orchestrator enters layer `trigger_at_layer`, it should call `prefetch_for_layer(prefetch_for_layer)`.

**Example with `prefetch_ahead_layers = 2`:**

```
Engram-augmented layers: [4, 8, 12, 16, 20]
Schedule:
  (2, 4)   -- at layer 2, start prefetch for layer 4
  (6, 8)   -- at layer 6, start prefetch for layer 8
  (10, 12) -- at layer 10, start prefetch for layer 12
  (14, 16) -- at layer 14, start prefetch for layer 16
  (18, 20) -- at layer 18, start prefetch for layer 20
```

If `trigger_at_layer < 0`, the prefetch should be triggered before the first layer (at the start of the forward pass):

```
Engram-augmented layers: [1, 5]
prefetch_ahead_layers = 2
Schedule:
  (-1, 1)  -- trigger before layer 0 (i.e., during embedding/input processing)
  (3, 5)   -- at layer 3, start prefetch for layer 5
```

### 6.4 Orchestrator Integration

The orchestrator (`BrainAI.forward()` or the layer loop) uses the plan as follows:

```python
def forward_with_prefetch(self, input_ids, hidden_states):
    # Step 1: Build the plan (cheap, runs once)
    plan = precompute_retrieval_plan(
        input_ids, self.engram_layer_ids,
        self.multi_head_hashers, self.offload_config,
    )

    # Step 2: Handle early prefetches (trigger_at < 0)
    for trigger_at, target_layer in plan.schedule:
        if trigger_at < 0:
            self._trigger_prefetch(plan, target_layer)

    # Step 3: Run layer loop
    for layer_id in range(self.num_layers):
        # Check if any prefetch should be triggered at this layer
        for trigger_at, target_layer in plan.schedule:
            if trigger_at == layer_id:
                self._trigger_prefetch(plan, target_layer)

        # Run the actual layer
        if layer_id in self.engram_layer_ids:
            # Consume the prefetched data
            engram_emb = self.engram_tables[layer_id].consume_prefetched()
            hidden_states = self.layers[layer_id](
                hidden_states, engram_embeddings=engram_emb
            )
        else:
            hidden_states = self.layers[layer_id](hidden_states)

    return hidden_states

def _trigger_prefetch(self, plan, target_layer):
    """Start async prefetch for a target layer."""
    unique_ids, inverse = plan.layer_unique_ids[target_layer]
    self.engram_tables[target_layer].prefetch(
        plan.layer_hash_ids[target_layer],
        stream=self._prefetch_stream,
    )
```

### 6.5 Choosing `prefetch_ahead_layers`

The optimal value depends on per-layer compute time and PCIe transfer time:

```
optimal_prefetch_ahead = ceil(transfer_time / per_layer_compute_time)
```

| GPU | Per-layer time (ms) | Transfer time (ms, typical) | Optimal ahead |
|---|---|---|---|
| A100 (PCIe 4.0) | 2-5 | 1-3 | 1 |
| A100 (PCIe 4.0) | 2-5 | 5-10 (large batch) | 2-3 |
| RTX 3090 (PCIe 4.0) | 3-8 | 2-5 | 1-2 |
| RTX 4090 (PCIe 4.0) | 2-4 | 1-3 | 1 |
| T4 (PCIe 3.0) | 5-10 | 3-8 | 1-2 |

Default: `prefetch_ahead_layers = 2` provides a good balance. Setting it too high wastes GPU memory (multiple batches of prefetched data in flight). Setting it too low risks the transfer not completing in time.

---

## 7. Pinned Buffer Pool

Repeated allocation and deallocation of pinned memory is expensive (requires OS kernel calls to page-lock memory). A buffer pool pre-allocates a fixed set of pinned buffers and recycles them across forward passes.

### 7.1 Design

```python
class PinnedBufferPool:
    """
    Pool of pre-allocated pinned host buffers for H2D transfers.

    Uses double-buffering: while one buffer is being transferred to GPU,
    the other can be filled with the next batch of data.

    Args:
        num_buffers: Number of buffers in the pool (2 for double-buffering)
        max_rows: Maximum number of unique rows per transfer
        embedding_dim: Dimension of each embedding row
        dtype: Data type for buffers (should match compute_dtype)
    """
    def __init__(
        self,
        num_buffers: int = 2,
        max_rows: int = 65536,
        embedding_dim: int = 256,
        dtype: torch.dtype = torch.float32,
    ):
        self.buffers = []
        self.in_use = []

        for _ in range(num_buffers):
            buf = torch.empty(
                max_rows, embedding_dim,
                dtype=dtype,
                pin_memory=True,
            )
            self.buffers.append(buf)
            self.in_use.append(False)

        self.max_rows = max_rows
        self.embedding_dim = embedding_dim
        self.dtype = dtype

    def acquire(self) -> Tuple[int, torch.Tensor]:
        """
        Acquire a free buffer from the pool.

        Returns:
            (buffer_id, buffer_tensor) -- the buffer is marked as in-use.

        Raises:
            RuntimeError: If no buffers are available.
        """
        for i, used in enumerate(self.in_use):
            if not used:
                self.in_use[i] = True
                return i, self.buffers[i]
        raise RuntimeError(
            f"All {len(self.buffers)} pinned buffers are in use. "
            "Increase pool size or ensure consume() is called promptly."
        )

    def release(self, buffer_id: int) -> None:
        """Release a buffer back to the pool."""
        self.in_use[buffer_id] = False

    def memory_bytes(self) -> int:
        """Total memory used by the pool."""
        per_buffer = self.max_rows * self.embedding_dim * self.buffers[0].element_size()
        return len(self.buffers) * per_buffer
```

### 7.2 Buffer Sizing

The buffer must be large enough to hold the maximum number of unique rows that could be transferred in a single prefetch operation:

```
max_rows = max(num_unique_ids) across all layers and batches
```

Conservative estimate:

```python
# Upper bound: all IDs in a batch are unique
max_possible_unique = batch_size * seq_len * total_heads

# Practical estimate: use coalescing ratio from profiling
estimated_unique = int(max_possible_unique * expected_coalesce_ratio)

# Add 20% headroom
buffer_rows = int(estimated_unique * 1.2)
```

**Memory per buffer:**

```
buffer_bytes = max_rows * embedding_dim * dtype_bytes
```

| max_rows | embedding_dim | dtype | Buffer Size |
|---|---|---|---|
| 16,384 | 256 | fp32 | 16 MB |
| 32,768 | 256 | fp32 | 32 MB |
| 65,536 | 256 | fp32 | 64 MB |
| 65,536 | 256 | fp16 | 32 MB |

With double-buffering (2 buffers), the total pinned memory is 2x the above.

### 7.3 Double-Buffering Pattern

```
Time ------->

Buffer A:  [fill with layer 4 data][--- H2D transfer --->][release]  [fill with layer 12 ...]
Buffer B:            [fill with layer 8 data][--- H2D transfer --->][release]

Default:   [layer 2 compute][layer 3][layer 4: consume A][layer 5][layer 6][layer 7][layer 8: consume B]
```

While buffer A is being transferred for layer 4's embeddings, buffer B can be filled with layer 8's data. By the time layer 4 consumes buffer A and releases it, buffer B's transfer may already be in progress.

### 7.4 Integration with OffloadableEmbedding

```python
class OffloadableEmbedding(nn.Module):
    def __init__(self, ..., buffer_pool: Optional[PinnedBufferPool] = None):
        # ...
        self._buffer_pool = buffer_pool
        self._active_buffer_id: Optional[int] = None

    def prefetch(self, ids, stream=None):
        if not self.weights_on_cpu:
            return

        prefetch_stream = stream or self._prefetch_stream
        flat_ids = ids.reshape(-1)
        unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)
        num_unique = unique_ids.numel()

        if self._buffer_pool is not None:
            # Use pooled buffer
            buf_id, buf = self._buffer_pool.acquire()
            self._active_buffer_id = buf_id

            # Fill buffer with rows from host table
            cpu_ids = unique_ids.cpu()
            buf[:num_unique].copy_(self.weight_cpu[cpu_ids])

            with torch.cuda.stream(prefetch_stream):
                device_rows = buf[:num_unique].to(
                    device='cuda',
                    dtype=self.compute_dtype,
                    non_blocking=True,
                )
                event = torch.cuda.Event()
                event.record(prefetch_stream)
        else:
            # Fallback: allocate on the fly
            cpu_ids = unique_ids.cpu()
            host_rows = self.weight_cpu[cpu_ids]
            with torch.cuda.stream(prefetch_stream):
                device_rows = host_rows.to(
                    device='cuda', dtype=self.compute_dtype, non_blocking=True,
                )
                event = torch.cuda.Event()
                event.record(prefetch_stream)

        self._prefetch_event = event
        self._prefetch_result = device_rows
        self._prefetch_inverse = inverse
        self._original_shape = ids.shape

    def consume_prefetched(self):
        # ... standard consume logic as in Section 3.4 ...
        result = self._consume_inner()
        # Release buffer back to pool
        if self._buffer_pool is not None and self._active_buffer_id is not None:
            self._buffer_pool.release(self._active_buffer_id)
            self._active_buffer_id = None
        return result
```

---

## 8. Benchmarking Methodology

Benchmarking the offload + prefetch system requires comparing three configurations to isolate the contribution of each optimization.

### 8.1 Three Configurations

| Config | Description | What It Measures |
|---|---|---|
| **(a) On-device baseline** | Full table on GPU, standard `F.embedding()` gather | Best-case latency (no PCIe involved) |
| **(b) CPU offload, sync** | Table on CPU, synchronous `H2D` copy per batch | Offload overhead without prefetch |
| **(c) CPU offload, async prefetch** | Table on CPU, async prefetch overlapped with compute | Offload with latency hiding |

### 8.2 Metrics

**Primary metrics:**

| Metric | Unit | How to Measure |
|---|---|---|
| Throughput | tokens/sec | `(batch_size * seq_len) / wall_time` per forward pass |
| Latency | ms/batch | Wall-clock time for one forward pass |
| Overlap ratio | dimensionless [0, 1] | `time_hidden / total_transfer_time` |

**Overlap ratio** is the key metric for prefetch effectiveness:

```
overlap_ratio = (transfer_time - visible_wait_time) / transfer_time
```

Where:
- `transfer_time` = total time spent on H2D transfers (measured on prefetch stream)
- `visible_wait_time` = time the default stream is blocked waiting for transfers

```
overlap_ratio = 0.0  -> No overlap (transfer is entirely on critical path)
overlap_ratio = 0.5  -> Half of transfer is hidden behind compute
overlap_ratio = 1.0  -> Transfer is completely hidden (ideal)
```

An overlap ratio above 0.5 means the prefetch is providing meaningful benefit. Above 0.8 is excellent.

**Secondary metrics:**

| Metric | Unit | Purpose |
|---|---|---|
| PCIe utilization | GB/s | Actual vs theoretical bandwidth |
| Coalescing ratio | dimensionless | Effectiveness of ID deduplication |
| GPU memory delta | MB | Memory savings vs on-device baseline |
| Buffer pool hit rate | % | Pool utilization efficiency |

### 8.3 Measurement Protocol

```python
def benchmark_engram_retrieval(
    config: str,  # 'on_device', 'offload_sync', 'offload_async'
    table_size: int,
    embedding_dim: int,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> Dict[str, float]:
    """
    Benchmark a single configuration.

    Returns dict with throughput, latency, overlap_ratio, memory_mb.
    """
    # Setup
    model = build_engram_model(config, table_size, embedding_dim, num_heads)
    input_ids = torch.randint(0, 50000, (batch_size, seq_len), device='cuda')

    # Warmup (critical for CUDA: JIT compilation, memory allocation)
    for _ in range(num_warmup):
        _ = model(input_ids)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iterations):
        _ = model(input_ids)
    end_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(end_event)
    per_batch_ms = total_ms / num_iterations
    tokens_per_sec = (batch_size * seq_len) / (per_batch_ms / 1000)

    return {
        'latency_ms': per_batch_ms,
        'throughput_tokens_sec': tokens_per_sec,
        'gpu_memory_mb': torch.cuda.max_memory_allocated() / 1e6,
    }
```

### 8.4 Measuring Overlap Ratio

Overlap ratio requires instrumented prefetch/consume calls:

```python
class InstrumentedOffloadableEmbedding(OffloadableEmbedding):
    """Adds timing instrumentation for benchmarking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transfer_start = torch.cuda.Event(enable_timing=True)
        self._transfer_end = torch.cuda.Event(enable_timing=True)
        self._wait_start = torch.cuda.Event(enable_timing=True)
        self._wait_end = torch.cuda.Event(enable_timing=True)
        self.total_transfer_ms = 0.0
        self.total_wait_ms = 0.0

    def prefetch(self, ids, stream=None):
        prefetch_stream = stream or self._prefetch_stream
        with torch.cuda.stream(prefetch_stream):
            self._transfer_start.record(prefetch_stream)

        super().prefetch(ids, stream)

        with torch.cuda.stream(prefetch_stream):
            self._transfer_end.record(prefetch_stream)

    def consume_prefetched(self):
        self._wait_start.record()  # on default stream
        result = super().consume_prefetched()
        self._wait_end.record()

        torch.cuda.synchronize()
        self.total_transfer_ms += self._transfer_start.elapsed_time(self._transfer_end)
        self.total_wait_ms += self._wait_start.elapsed_time(self._wait_end)
        return result

    @property
    def overlap_ratio(self) -> float:
        if self.total_transfer_ms == 0:
            return 0.0
        visible_wait = self.total_wait_ms
        return max(0.0, 1.0 - visible_wait / self.total_transfer_ms)
```

### 8.5 Sweep Parameters

The benchmark should sweep over these axes to characterize performance across operating points:

| Parameter | Values to Sweep | Why |
|---|---|---|
| `table_size` | 131k, 1M, 10M | Transfer size scales with unique rows |
| `batch_size` | 1, 4, 16, 64 | Affects coalescing ratio and transfer size |
| `seq_len` | 128, 512, 2048 | Affects total IDs and coalescing |
| `embedding_dim` | 64, 128, 256, 512 | Affects bytes per row |
| `num_heads` | 2, 4, 8 | Affects total IDs per position |
| `prefetch_ahead` | 0, 1, 2, 3 | Affects overlap window |

### 8.6 Expected Results Pattern

```
Configuration          | Latency (ms) | Throughput (tok/s) | Memory (MB)
-----------------------|--------------|--------------------|-----------
(a) On-device          | 1.2          | 870,000            | 2,048
(b) Offload sync       | 4.8          | 213,000            | 128
(c) Offload async (1L) | 2.1          | 488,000            | 128
(c) Offload async (2L) | 1.5          | 683,000            | 128
(c) Offload async (3L) | 1.3          | 790,000            | 128
```

Key observations:
- On-device is fastest but uses the most GPU memory.
- Synchronous offload is 3-4x slower (PCIe latency on critical path).
- Async prefetch with 2 layers ahead recovers most of the on-device performance.
- Async prefetch with 3 layers ahead approaches on-device speed.
- GPU memory in offload mode is 10-20x lower than on-device.

---

## 9. Distributed Sharding Compatibility

For tables that exceed a single machine's host memory (e.g., 10B+ rows across multiple N-gram orders), the tables can be sharded across multiple GPUs or hosts. The `OffloadableEmbedding` API is designed to be forward-compatible with sharding without requiring API changes.

### 9.1 Range-Based Partitioning

Each rank owns a contiguous range of embedding rows:

```
Total table size: N rows
Number of shards: S
Shard i owns rows: [i * (N // S), (i+1) * (N // S))
```

```python
def compute_shard_id(hash_id: int, table_size: int, num_shards: int) -> int:
    """Determine which shard owns a given hash ID."""
    shard_size = table_size // num_shards
    return hash_id // shard_size

def compute_local_id(hash_id: int, table_size: int, num_shards: int) -> int:
    """Compute the local row index within the owning shard."""
    shard_size = table_size // num_shards
    return hash_id % shard_size
```

### 9.2 Sharded Lookup Flow

```
1. Compute hash_ids on each rank (deterministic, all ranks get same IDs)
2. For each ID, compute shard_id = hash_id // shard_size
3. Partition IDs by destination shard
4. All-to-All: send ID lists to owning ranks, receive ID lists from requesters
5. Each rank gathers rows from its local shard
6. All-to-All: send gathered rows back to requesting ranks
7. Each rank assembles the full embedding tensor
```

### 9.3 API Compatibility

The current `OffloadableEmbedding` API does not need to change for sharding:

| Method | Single-device Behavior | Future Sharded Behavior |
|---|---|---|
| `lookup(ids)` | Gather from local CPU/GPU table | Route to shards, All-to-All, assemble |
| `prefetch(ids)` | Async H2D on prefetch stream | Async All-to-All on prefetch stream |
| `consume_prefetched()` | Wait for H2D, return | Wait for All-to-All, return |

The key design decision that enables this is that `lookup()` and `prefetch()` accept abstract integer IDs and return embeddings. The caller does not need to know whether the IDs are served from a local table, a remote shard, or a cache.

### 9.4 Shard-Aware Coalescing

With sharding, coalescing happens in two stages:

```
Stage 1 (local): Deduplicate IDs per rank
Stage 2 (cross-rank): Deduplicate IDs per destination shard
```

```python
def shard_aware_coalesce(
    flat_ids: torch.Tensor,
    table_size: int,
    num_shards: int,
    local_rank: int,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Coalesce IDs grouped by destination shard.

    Returns:
        Dict mapping shard_id -> (unique_ids_for_shard, inverse_for_shard)
    """
    shard_size = table_size // num_shards
    shard_ids = flat_ids // shard_size  # Which shard each ID belongs to

    result = {}
    for shard in range(num_shards):
        mask = shard_ids == shard
        shard_flat_ids = flat_ids[mask]
        if shard_flat_ids.numel() > 0:
            unique, inverse = torch.unique(shard_flat_ids, return_inverse=True)
            result[shard] = (unique, inverse, mask)

    return result
```

### 9.5 Communication Primitives

| Primitive | PyTorch API | Purpose |
|---|---|---|
| All-to-All | `torch.distributed.all_to_all()` | Exchange ID lists and embeddings between ranks |
| All-Gather | `torch.distributed.all_gather()` | Broadcast small metadata (counts, sizes) |
| Barrier | `torch.distributed.barrier()` | Synchronize before/after sharded operations |

### 9.6 Current Scope

The current implementation supports single-device offload only. Sharding is documented here for forward compatibility. The API contract is designed so that a `ShardedOffloadableEmbedding` subclass can replace `OffloadableEmbedding` without changing the caller.

```python
# Current: single-device
embedding = OffloadableEmbedding(
    num_embeddings=10_000_003,
    embedding_dim=256,
    weights_on_cpu=True,
)

# Future: sharded (same API)
embedding = ShardedOffloadableEmbedding(
    num_embeddings=10_000_003,
    embedding_dim=256,
    num_shards=8,
    local_rank=rank,
)

# Caller code is identical in both cases:
embedding.prefetch(hash_ids)
# ... compute ...
result = embedding.consume_prefetched()
```

---

## 10. Correctness Invariants

The offload and prefetch system must satisfy the following invariants for every forward pass. These are testable properties that the validation script (`scripts/validate_engram.py`) checks.

### 10.1 Numerical Equivalence

**Invariant 1: CPU offload output matches on-device output.**

```python
def test_offload_numerical_equivalence():
    """Offload mode must produce the same embeddings as on-device mode."""
    num_embeddings = 131_071
    embedding_dim = 256
    ids = torch.randint(0, num_embeddings, (4, 512, 8), device='cuda')

    # On-device reference
    emb_device = OffloadableEmbedding(
        num_embeddings, embedding_dim, weights_on_cpu=False
    ).cuda()

    # CPU offload
    emb_offload = OffloadableEmbedding(
        num_embeddings, embedding_dim,
        weights_on_cpu=True,
        storage_dtype=torch.float32,  # fp32 for exact match
    )

    # Copy weights to ensure same initialization
    emb_offload.weight_cpu.copy_(emb_device.embedding.weight.data.cpu())

    out_device = emb_device.lookup(ids)
    out_offload = emb_offload.lookup(ids)

    # Exact match when both are fp32
    assert torch.allclose(out_device, out_offload, atol=0, rtol=0)
```

**Invariant 2: fp16 storage matches fp32 within tolerance.**

```python
def test_fp16_storage_tolerance():
    """fp16 host storage must match fp32 on-device within cosine > 0.9999."""
    emb_fp32 = OffloadableEmbedding(
        131_071, 256, weights_on_cpu=True, storage_dtype=torch.float32
    )
    emb_fp16 = OffloadableEmbedding(
        131_071, 256, weights_on_cpu=True, storage_dtype=torch.float16
    )

    # Initialize fp16 from fp32 (simulates one-time quantization)
    emb_fp16.weight_cpu.copy_(emb_fp32.weight_cpu.half())

    ids = torch.randint(0, 131_071, (4, 512), device='cuda')

    out_fp32 = emb_fp32.lookup(ids)
    out_fp16 = emb_fp16.lookup(ids)

    # Cosine similarity check
    cos_sim = F.cosine_similarity(
        out_fp32.reshape(-1, 256),
        out_fp16.reshape(-1, 256),
        dim=-1,
    ).mean()
    assert cos_sim > 0.9999, f"Cosine similarity {cos_sim} below threshold"
```

### 10.2 Prefetch/Consume Cycle Correctness

**Invariant 3: Prefetch output matches synchronous lookup.**

```python
def test_prefetch_matches_sync():
    """Async prefetch must produce identical results to synchronous lookup."""
    emb = OffloadableEmbedding(131_071, 256, weights_on_cpu=True)
    ids = torch.randint(0, 131_071, (4, 512, 8), device='cuda')

    # Synchronous path
    sync_result = emb.lookup(ids)

    # Async path
    emb.prefetch(ids)
    async_result = emb.consume_prefetched()

    assert torch.allclose(sync_result, async_result, atol=1e-6, rtol=1e-5)
```

**Invariant 4: No deadlocks under repeated cycles.**

```python
def test_no_deadlocks(num_cycles=50, timeout_sec=30):
    """Repeated prefetch/consume cycles must complete within timeout."""
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Deadlock detected after {timeout_sec}s")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)

    try:
        emb = OffloadableEmbedding(131_071, 256, weights_on_cpu=True)
        for i in range(num_cycles):
            ids = torch.randint(0, 131_071, (4, 512), device='cuda')
            emb.prefetch(ids)
            result = emb.consume_prefetched()
            assert result.shape == (4, 512, 256)
    finally:
        signal.alarm(0)
```

### 10.3 Coalescing Correctness

**Invariant 5: Repeated IDs produce identical embeddings.**

```python
def test_coalescing_correctness():
    """IDs that appear multiple times must map to the same embedding."""
    emb = OffloadableEmbedding(131_071, 256, weights_on_cpu=True)

    # Create IDs with known repetitions
    base_ids = torch.tensor([100, 200, 100, 300, 200, 100], device='cuda')
    result = emb.lookup(base_ids)

    # Positions 0, 2, 5 all have ID 100 -- must be identical
    assert torch.equal(result[0], result[2])
    assert torch.equal(result[0], result[5])

    # Positions 1, 4 both have ID 200 -- must be identical
    assert torch.equal(result[1], result[4])
```

**Invariant 6: Coalescing preserves inverse mapping.**

```python
def test_inverse_mapping_correctness():
    """unique_ids[inverse] must exactly reconstruct the original IDs."""
    ids = torch.randint(0, 1000, (4096,))
    unique_ids, inverse = torch.unique(ids, return_inverse=True)

    reconstructed = unique_ids[inverse]
    assert torch.equal(ids, reconstructed)
```

### 10.4 dtype Casting Correctness

**Invariant 7: Host fp16/bf16 to device fp32 casting is lossless for the stored precision.**

```python
def test_dtype_casting_roundtrip():
    """Cast to storage_dtype and back must not introduce additional error."""
    weight_fp32 = torch.randn(1000, 256)

    for storage_dtype in [torch.float16, torch.bfloat16]:
        weight_stored = weight_fp32.to(storage_dtype)
        weight_restored = weight_stored.to(torch.float32)

        # The cast fp32 -> storage -> fp32 introduces quantization error.
        # But cast storage -> fp32 is LOSSLESS (no additional error).
        weight_stored_again = weight_restored.to(storage_dtype)
        assert torch.equal(weight_stored, weight_stored_again), (
            f"Roundtrip through {storage_dtype} is not stable"
        )
```

### 10.5 Invariant Summary Table

| ID | Invariant | Threshold | Test Method |
|---|---|---|---|
| 1 | Offload matches on-device (fp32 storage) | Exact match | `torch.allclose(atol=0)` |
| 2 | fp16 storage matches fp32 | Cosine > 0.9999 | `F.cosine_similarity` |
| 3 | Prefetch matches sync lookup | `atol=1e-6` | `torch.allclose` |
| 4 | No deadlocks over 50 cycles | Complete within 30s | `signal.alarm` timeout |
| 5 | Repeated IDs produce identical embeddings | Exact match | `torch.equal` |
| 6 | Coalescing inverse mapping | Exact reconstruction | `torch.equal` |
| 7 | dtype roundtrip stability | Exact match in storage dtype | `torch.equal` |

---

## 11. Memory Accounting

Accurate memory accounting is essential for capacity planning and for reporting memory savings in telemetry.

### 11.1 Host Memory

**Embedding table (master weights):**

```
host_table_bytes = num_embeddings * embedding_dim * storage_dtype_bytes
```

| num_embeddings | embedding_dim | storage_dtype | Host Memory |
|---|---|---|---|
| 131,071 | 256 | fp16 (2B) | 67.1 MB |
| 131,071 | 256 | fp32 (4B) | 134.2 MB |
| 1,048,573 | 256 | fp16 (2B) | 537.1 MB |
| 10,000,003 | 256 | fp16 (2B) | 5,120.0 MB |
| 10,000,003 | 256 | bf16 (2B) | 5,120.0 MB |

**With multiple tables (N orders x H heads per order):**

```
total_host_bytes = num_tables * host_table_bytes
num_tables = len(ngram_orders) * num_heads_per_order
```

| Config | Orders | Heads/Order | Tables | Per-Table (fp16, 131k, D=256) | Total Host |
|---|---|---|---|---|---|
| Minimal | (2, 3) | 2 | 4 | 67 MB | 268 MB |
| Dev | (2, 3, 4) | 2 | 6 | 67 MB | 402 MB |
| Production | (2, 3, 4, 5) | 2 | 8 | 5,120 MB | 40,960 MB |

### 11.2 Pinned Buffer Memory

```
pinned_buffer_bytes = num_buffers * max_rows * embedding_dim * compute_dtype_bytes
```

With double-buffering (`num_buffers = 2`):

| max_rows | embedding_dim | compute_dtype | Pinned Memory |
|---|---|---|---|
| 16,384 | 256 | fp32 (4B) | 32 MB |
| 32,768 | 256 | fp32 (4B) | 64 MB |
| 65,536 | 256 | fp32 (4B) | 128 MB |

### 11.3 Device (GPU) Memory

In offload mode, GPU memory holds only:

1. **Active rows per batch:** The unique embedding rows needed for the current batch.
2. **Inverse mapping tensor:** Integer tensor for reconstruction.
3. **Prefetch buffer (in-flight):** One set of active rows being transferred.

```
device_active_bytes = num_unique * embedding_dim * compute_dtype_bytes
device_inverse_bytes = num_total * 8  # int64
device_prefetch_bytes = device_active_bytes  # double-buffered in-flight
```

**Typical device memory per batch:**

| Scenario | num_unique | embedding_dim | Active Rows | Inverse | Total GPU |
|---|---|---|---|---|---|
| Small batch (B=1, T=128) | ~800 | 256 | 0.8 MB | 0.008 MB | ~1.6 MB |
| Medium batch (B=8, T=512) | ~12,000 | 256 | 12.3 MB | 0.25 MB | ~25 MB |
| Large batch (B=64, T=2048) | ~180,000 | 256 | 184 MB | 8 MB | ~376 MB |

### 11.4 Memory Savings vs On-Device

```python
def compute_memory_savings(
    num_embeddings: int,
    embedding_dim: int,
    num_tables: int,
    num_unique_per_batch: int,
    storage_dtype_bytes: int = 2,
    compute_dtype_bytes: int = 4,
) -> Dict[str, float]:
    """Compute memory savings from offload mode."""

    # On-device: full tables in compute_dtype
    on_device_bytes = (
        num_tables * num_embeddings * embedding_dim * compute_dtype_bytes
    )

    # Offload: only active rows on GPU, tables on CPU
    gpu_active_bytes = (
        num_tables * num_unique_per_batch * embedding_dim * compute_dtype_bytes
    )
    # Add prefetch in-flight buffer
    gpu_prefetch_bytes = gpu_active_bytes  # worst case: separate in-flight set
    gpu_total_bytes = gpu_active_bytes + gpu_prefetch_bytes

    # Host memory
    host_bytes = (
        num_tables * num_embeddings * embedding_dim * storage_dtype_bytes
    )

    return {
        'on_device_gpu_mb': on_device_bytes / 1e6,
        'offload_gpu_mb': gpu_total_bytes / 1e6,
        'offload_host_mb': host_bytes / 1e6,
        'gpu_savings_mb': (on_device_bytes - gpu_total_bytes) / 1e6,
        'gpu_savings_pct': (1 - gpu_total_bytes / on_device_bytes) * 100,
    }
```

**Example savings (production config):**

```
num_embeddings = 10,000,003
embedding_dim = 256
num_tables = 8 (4 orders x 2 heads)
num_unique_per_batch = 50,000 (typical for B=16, T=2048)

On-device GPU:   8 * 10M * 256 * 4B = 81,920 MB = ~80 GB
Offload GPU:     8 * 50k * 256 * 4B * 2 = 819 MB  = ~0.8 GB
Host (CPU):      8 * 10M * 256 * 2B = 40,960 MB = ~40 GB

GPU savings: 81,120 MB (99.0%)
```

### 11.5 Memory Reporting

The `OffloadableEmbedding` should expose a method for memory reporting:

```python
def memory_report(self) -> Dict[str, float]:
    """Return memory usage breakdown in megabytes."""
    dtype_bytes = {
        torch.float16: 2, torch.bfloat16: 2,
        torch.float32: 4, torch.float64: 8,
    }

    if self.weights_on_cpu:
        host_mb = (
            self.num_embeddings
            * self.embedding_dim
            * dtype_bytes[self.storage_dtype]
        ) / 1e6

        device_mb = 0.0
        if self._prefetch_result is not None:
            device_mb = (
                self._prefetch_result.numel()
                * dtype_bytes[self.compute_dtype]
            ) / 1e6

        return {
            'mode': 'offload',
            'host_memory_mb': host_mb,
            'device_memory_mb': device_mb,
            'pinned': self.weight_cpu.is_pinned() if hasattr(self, 'weight_cpu') else False,
            'storage_dtype': str(self.storage_dtype),
        }
    else:
        device_mb = (
            self.num_embeddings
            * self.embedding_dim
            * dtype_bytes.get(self.embedding.weight.dtype, 4)
        ) / 1e6

        return {
            'mode': 'on_device',
            'host_memory_mb': 0.0,
            'device_memory_mb': device_mb,
            'storage_dtype': str(self.embedding.weight.dtype),
        }
```

---

## 12. End-to-End Integration Example

This section ties together all components into a complete forward pass with prefetch.

### 12.1 Setup

```python
import torch
from brain_ai.memory.hash_embedding import (
    MultiHeadHash, OffloadableEmbedding, PrefetchPlan,
)
from brain_ai.memory.tokenizer_compression import TokenizerCompression
from brain_ai.memory.engram import EngramConfig, EngramModule

# Configuration
config = EngramConfig(
    vocab_size=128_000,
    embedding_dim=256,
    ngram_orders=(2, 3, 4),
    num_heads=2,
    table_size=131_071,  # prime
    offload_to_cpu=True,
    prefetch=True,
)

offload_config = OffloadConfig(
    weights_on_cpu=True,
    use_async_prefetch=True,
    prefetch_ahead_layers=2,
    pin_memory=True,
    storage_dtype='float16',
)
```

### 12.2 Forward Pass with Prefetch

```python
def forward_pass_with_prefetch(
    model,
    input_ids: torch.Tensor,   # (B, T)
    hidden_states: torch.Tensor,  # (B, T, D)
):
    """
    Complete forward pass demonstrating the prefetch pipeline.

    Timeline:
    1. Precompute all hash IDs (cheap integer math)
    2. Build retrieval plan
    3. Layer loop with prefetch scheduling
    """
    B, T = input_ids.shape

    # === Step 1: Tokenizer compression ===
    canonical_ids = model.tokenizer_compression.compress_ids(input_ids)

    # === Step 2: Precompute retrieval plan ===
    plan = precompute_retrieval_plan(
        canonical_ids,
        layer_ids=model.engram_layer_ids,
        multi_head_hashers=model.hashers,
        offload_config=offload_config,
    )

    # === Step 3: Trigger early prefetches ===
    for trigger_at, target_layer in plan.schedule:
        if trigger_at < 0:
            model.engram_tables[target_layer].prefetch(
                plan.layer_hash_ids[target_layer]
            )

    # === Step 4: Layer loop ===
    for layer_id in range(model.num_layers):
        # Check prefetch schedule
        for trigger_at, target_layer in plan.schedule:
            if trigger_at == layer_id:
                model.engram_tables[target_layer].prefetch(
                    plan.layer_hash_ids[target_layer]
                )

        # Execute layer
        if layer_id in model.engram_layer_ids:
            # Consume prefetched embeddings
            engram_embeddings = model.engram_tables[layer_id].consume_prefetched()

            # Run engram-augmented layer
            hidden_states = model.layers[layer_id](
                hidden_states,
                engram_embeddings=engram_embeddings,
                attention_mask=model.attention_mask,
            )
        else:
            # Standard transformer layer
            hidden_states = model.layers[layer_id](hidden_states)

    # === Step 5: Collect telemetry ===
    telemetry = {
        'plan': plan.telemetry,
        'memory': {
            lid: model.engram_tables[lid].memory_report()
            for lid in model.engram_layer_ids
        },
    }

    return hidden_states, telemetry
```

### 12.3 Telemetry Output Example

```python
{
    'plan': {
        4: {'coalesce_ratio': 0.38, 'num_unique': 6234, 'num_total': 16384},
        8: {'coalesce_ratio': 0.41, 'num_unique': 6726, 'num_total': 16384},
        12: {'coalesce_ratio': 0.39, 'num_unique': 6398, 'num_total': 16384},
    },
    'memory': {
        4: {'mode': 'offload', 'host_memory_mb': 67.1, 'device_memory_mb': 6.4, 'pinned': True},
        8: {'mode': 'offload', 'host_memory_mb': 67.1, 'device_memory_mb': 6.9, 'pinned': True},
        12: {'mode': 'offload', 'host_memory_mb': 67.1, 'device_memory_mb': 6.5, 'pinned': True},
    },
}
```

---

## 13. Common Failure Modes and Debugging

### 13.1 Failure Mode Table

| Symptom | Root Cause | Diagnostic | Fix |
|---|---|---|---|
| Deadlock on `consume_prefetched()` | Missing `event.record()` after transfer | Check if `_prefetch_event` is None | Ensure event is recorded on prefetch stream after H2D copy |
| Data corruption (wrong embeddings) | Consuming before event is signaled | Add assertions comparing sync vs async results | Ensure `wait_event()` before accessing `device_rows` |
| No overlap (async same speed as sync) | Prefetch triggered too late | Measure time between `prefetch()` and `consume()` | Increase `prefetch_ahead_layers` |
| OOM on CPU (host memory) | Too many pinned tables | Check host memory usage with `psutil` | Reduce table size or use fp16 storage |
| OOM on GPU | Too many active rows or leaking prefetch buffers | Check `torch.cuda.memory_allocated()` | Ensure `consume_prefetched()` is always called; check buffer pool release |
| Slow H2D transfer | Non-pinned memory | Check `weight_cpu.is_pinned()` | Ensure `pin_memory=True` in constructor |
| `RuntimeError: previous prefetch not consumed` | Double prefetch without consume | Review prefetch/consume call order | Always consume before next prefetch for same table |
| Different results across runs | Non-deterministic `torch.unique` ordering | Check that inverse mapping is used, not raw unique order | `torch.unique` ordering is deterministic for sorted output (default) |
| Gradient NaN in training with offload | Mixed precision casting issues | Check dtype of returned embeddings | Ensure compute_dtype is fp32; use `torch.autocast` correctly |

### 13.2 Debugging Checklist

1. **Verify determinism:** Run the same input twice and compare hash IDs. They must be identical.
2. **Verify coalescing:** Check that `unique_ids[inverse]` exactly reconstructs the original `flat_ids`.
3. **Verify transfer:** After `consume_prefetched()`, compare result against `lookup()` on the same IDs.
4. **Verify no leaks:** After 100 forward passes, GPU memory should not grow.
5. **Verify stream isolation:** Prefetch operations should not appear on the default stream timeline in `torch.profiler`.
6. **Verify pinned memory:** `weight_cpu.is_pinned()` must return `True` when `pin_memory=True`.

### 13.3 Profiling with torch.profiler

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=5),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step in range(9):
        hidden, telemetry = forward_pass_with_prefetch(
            model, input_ids, hidden_states
        )
        prof.step()
```

In the TensorBoard trace, look for:
- **Prefetch stream activity:** H2D memcpy operations should overlap with compute kernels on the default stream.
- **Gap between prefetch and consume:** The wider this gap, the more overlap is achieved.
- **No spurious synchronizations:** `cudaStreamSynchronize` or `cudaDeviceSynchronize` calls between prefetch and consume indicate lost overlap.

---

## 14. Configuration Reference

### 14.1 OffloadConfig Fields

```python
@dataclass
class OffloadConfig:
    """Configuration for CPU offload and async prefetch."""

    weights_on_cpu: bool = False
    """If True, store embedding tables in host (CPU) memory."""

    use_async_prefetch: bool = False
    """If True, enable async prefetch on a dedicated CUDA stream.
    Requires weights_on_cpu=True and CUDA availability."""

    prefetch_ahead_layers: int = 2
    """Number of layers ahead to trigger prefetch.
    Higher values give more overlap but use more GPU memory for in-flight data."""

    pin_memory: bool = True
    """If True, use pinned (page-locked) host memory for embedding tables.
    Roughly doubles H2D transfer throughput. Requires CUDA."""

    storage_dtype: str = "float16"
    """Precision for host storage. Options: 'float16', 'bfloat16', 'float32'.
    fp16/bf16 halve host memory vs fp32. Cast to compute_dtype on transfer."""

    buffer_pool_size: int = 2
    """Number of pinned buffers in the pool (2 = double-buffering)."""

    max_unique_rows: int = 65536
    """Maximum unique rows per transfer. Determines buffer size.
    Set based on expected batch_size * seq_len * coalesce_ratio."""

    prefetch_timeout_ms: int = 5000
    """Timeout for prefetch completion in milliseconds.
    Raises TimeoutError if exceeded (deadlock detection)."""
```

### 14.2 Preset Configurations

```python
@classmethod
def minimal(cls) -> 'OffloadConfig':
    """For unit tests. No offload, no prefetch."""
    return cls(
        weights_on_cpu=False,
        use_async_prefetch=False,
    )

@classmethod
def dev(cls) -> 'OffloadConfig':
    """For development. Offload enabled, modest table sizes."""
    return cls(
        weights_on_cpu=True,
        use_async_prefetch=True,
        prefetch_ahead_layers=1,
        pin_memory=True,
        storage_dtype='float16',
        max_unique_rows=16384,
    )

@classmethod
def production(cls) -> 'OffloadConfig':
    """For production inference. Full offload, aggressive prefetch."""
    return cls(
        weights_on_cpu=True,
        use_async_prefetch=True,
        prefetch_ahead_layers=2,
        pin_memory=True,
        storage_dtype='float16',
        buffer_pool_size=2,
        max_unique_rows=65536,
        prefetch_timeout_ms=10000,
    )
```

### 14.3 Interaction with Other Configs

| Config | Interaction with OffloadConfig |
|---|---|
| `HashConfig.table_size` | Determines `num_embeddings` for `OffloadableEmbedding`. Larger tables need more host memory. |
| `HashConfig.per_layer_salt` | Each layer has different hash IDs, which affects coalescing per layer. |
| `EngramConfig.ngram_orders` | More orders = more tables = more host memory. |
| `EngramConfig.num_heads_per_order` | More heads = more tables = more host memory. |
| `EngramConfig.embedding_dim` | Larger D = larger rows = more bytes per transfer. |
| `GatingConfig.*` | No direct interaction; gating happens after embeddings are on device. |

---

## 15. Glossary

| Term | Definition |
|---|---|
| **H2D transfer** | Host-to-Device memory copy (CPU to GPU over PCIe) |
| **D2H transfer** | Device-to-Host memory copy (GPU to CPU over PCIe) |
| **PCIe** | Peripheral Component Interconnect Express; the bus connecting CPU and GPU |
| **Pinned memory** | Page-locked host memory that cannot be swapped to disk; enables direct DMA |
| **DMA** | Direct Memory Access; hardware-level memory transfer without CPU involvement |
| **Coalescing** | Deduplicating IDs before transfer to reduce bandwidth usage |
| **Coalescing ratio** | `num_unique / num_total`; lower is better (more deduplication) |
| **Prefetch stream** | Dedicated CUDA stream for async H2D transfers |
| **Default stream** | The primary CUDA stream where compute kernels execute |
| **CUDA Event** | Lightweight synchronization primitive; marks a point in a stream's execution |
| **Overlap ratio** | Fraction of transfer time hidden behind compute; higher is better |
| **Buffer pool** | Pre-allocated set of pinned buffers recycled across forward passes |
| **Double-buffering** | Using two buffers alternately: fill one while transferring the other |
| **Shard** | A partition of the embedding table assigned to one rank in distributed mode |
| **Range-based partitioning** | Sharding strategy where each rank owns a contiguous row range |
| **All-to-All** | Distributed primitive: each rank sends/receives data to/from all other ranks |
| **Storage dtype** | Precision used for host-side weight storage (fp16/bf16 to save memory) |
| **Compute dtype** | Precision used for on-device computation (fp32 for stability) |
| **Retrieval plan** | Precomputed schedule of hash IDs and prefetch timing for all layers |
| **Inverse mapping** | Tensor such that `unique_ids[inverse] == original_ids` |
