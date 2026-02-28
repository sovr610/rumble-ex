# Multi-Level Caching Architecture for Brain-Inspired AI Inference

This document describes the multi-level caching system for brain_ai inference. The cognitive pipeline produces intermediate representations at each stage (encoder outputs, SNN spike patterns, HTM column states, workspace representations) that are expensive to recompute. Caching these intermediates provides dramatic speedups for repeated or similar inputs. The implementation is in `assets/cache_manager_template.py`.

---

## 1. Cache Level Design

### 1.1 Three-Level Hierarchy

The caching system mirrors CPU cache hierarchies, adapted for deep learning inference:

```
        +------------------+
        |   L1: GPU SRAM   |  Fastest, smallest
        |  (tensor cache)  |  ~512MB default
        +------------------+
                |
        +------------------+
        |  L2: CPU Pinned  |  Medium speed, larger
        |    (RAM cache)   |  ~2GB default
        +------------------+
                |
        +------------------+
        |  L3: Disk mmap   |  Slowest, unlimited
        |  (persistent)    |  configurable path
        +------------------+
```

### 1.2 L1: GPU Tensor Cache

**Purpose:** Cache hot encoder outputs and workspace representations directly in GPU memory. These are the most frequently reused intermediates.

**Implementation:** An ordered dictionary mapping cache keys to GPU tensors. LRU eviction when the memory budget is exceeded.

**What to cache at L1:**
- Encoder outputs for recently seen inputs (vision features, text embeddings)
- Workspace competition results (the integrated multimodal representation)
- System 1 outputs for confident predictions

**Memory budget:** Default 512MB. For a 4096-dim workspace representation in fp16, each cached entry is ~8KB. A 512MB L1 cache holds approximately 65,000 entries.

**Access latency:** Sub-microsecond (GPU memory access, no PCIe transfer).

```python
class L1Cache:
    def __init__(self, max_memory_mb: int = 512, device: str = "cuda"):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.device = device
        self.cache = OrderedDict()
        self.current_bytes = 0

    def get(self, key: str) -> Optional[Tensor]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Tensor) -> None:
        entry_bytes = value.nelement() * value.element_size()
        while self.current_bytes + entry_bytes > self.max_memory_bytes and self.cache:
            _, evicted = self.cache.popitem(last=False)
            self.current_bytes -= evicted.nelement() * evicted.element_size()
        self.cache[key] = value.to(self.device)
        self.current_bytes += entry_bytes
```

### 1.3 L2: CPU Pinned Memory Cache

**Purpose:** Cache warm intermediates that do not fit in GPU memory but are accessed frequently enough to justify keeping in RAM. Pinned memory enables faster GPU transfers than pageable memory.

**What to cache at L2:**
- Engram hash embeddings (large but reusable across similar text inputs)
- HTM column state snapshots (for sequence continuations)
- Overflow from L1 (evicted entries that may be reused)

**Memory budget:** Default 2GB. Pinned memory is a limited system resource; do not over-allocate.

**Access latency:** ~10 microseconds (PCIe transfer from pinned CPU memory to GPU).

```python
class L2Cache:
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.current_bytes = 0

    def get(self, key: str) -> Optional[Tensor]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Tensor) -> None:
        cpu_value = value.detach().cpu().pin_memory()
        entry_bytes = cpu_value.nelement() * cpu_value.element_size()
        while self.current_bytes + entry_bytes > self.max_memory_bytes and self.cache:
            _, evicted = self.cache.popitem(last=False)
            self.current_bytes -= evicted.nelement() * evicted.element_size()
        self.cache[key] = cpu_value
        self.current_bytes += entry_bytes
```

### 1.4 L3: Disk-Backed mmap Cache

**Purpose:** Persistent cache for infrequently accessed but expensive-to-compute results. Survives process restarts. Useful for engram memory lookups and precomputed features.

**What to cache at L3:**
- Precomputed encoder features for static datasets
- HTM learned column patterns
- Engram hash table segments

**Storage:** Memory-mapped files using `numpy.memmap` or `torch.UntypedStorage.from_file`. Each cache entry is a separate file named by its cache key hash.

**Access latency:** ~1ms (disk read, OS page cache may improve this).

```python
class L3Cache:
    def __init__(self, cache_dir: str = "/tmp/brain_ai_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.index = {}  # key -> (filename, shape, dtype)

    def get(self, key: str) -> Optional[Tensor]:
        if key not in self.index:
            return None
        filename, shape, dtype = self.index[key]
        path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(path):
            del self.index[key]
            return None
        data = torch.load(path, weights_only=True)
        return data

    def put(self, key: str, value: Tensor) -> None:
        filename = hashlib.sha256(key.encode()).hexdigest()[:16] + ".pt"
        path = os.path.join(self.cache_dir, filename)
        torch.save(value.detach().cpu(), path)
        self.index[key] = (filename, tuple(value.shape), str(value.dtype))
```

---

## 2. Cache Key Generation

### 2.1 The Key Generation Problem

Cache keys must uniquely identify the computation that produced a cached value. For brain_ai, the key must capture:
1. The input data (tensor content)
2. The model version (weights may have changed)
3. The module that produced the output (encoder name, layer index)
4. Any configuration that affects the output (dtype, feature flags)

### 2.2 Input Hashing

Hashing raw tensor data is the most reliable approach but can be slow for large tensors. Strategies:

**Full tensor hash (accurate, slow):**
```python
def tensor_hash(t: Tensor) -> str:
    """SHA-256 of tensor bytes. Accurate but O(n) in tensor size."""
    data = t.detach().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()
```

**Sampled hash (fast, approximate):**
```python
def tensor_hash_sampled(t: Tensor, n_samples: int = 1024) -> str:
    """Hash a random subset of tensor elements. Fast but has collision risk."""
    flat = t.detach().flatten()
    if flat.numel() <= n_samples:
        indices = torch.arange(flat.numel())
    else:
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(flat.numel(), generator=generator)[:n_samples]
    sampled = flat[indices].cpu().numpy().tobytes()
    return hashlib.sha256(sampled).hexdigest()
```

**Shape+stats hash (fastest, least accurate):**
```python
def tensor_hash_stats(t: Tensor) -> str:
    """Hash based on shape, mean, std, min, max. Very fast, collision-prone."""
    stats = f"{tuple(t.shape)}_{t.mean().item():.6f}_{t.std().item():.6f}"
    stats += f"_{t.min().item():.6f}_{t.max().item():.6f}"
    return hashlib.sha256(stats.encode()).hexdigest()
```

**Recommendation:** Use sampled hashing for L1 (fast, called frequently) and full hashing for L3 (slower but persistent, correctness matters more).

### 2.3 Composite Key Structure

The full cache key combines input hash with context:

```python
def make_cache_key(input_hash: str, module_name: str, model_version: str,
                   config_hash: str) -> str:
    """Construct a composite cache key."""
    components = f"{module_name}:{model_version}:{config_hash}:{input_hash}"
    return hashlib.sha256(components.encode()).hexdigest()
```

The `model_version` is a hash of the model's state_dict checksums, updated whenever weights change. This ensures cache invalidation on model updates (Section 4).

---

## 3. Eviction Policies

### 3.1 LRU (Least Recently Used)

The default eviction policy. When the cache is full and a new entry must be added, the entry that was accessed least recently is evicted. Python's `OrderedDict` provides O(1) LRU operations.

**Pros:** Simple, effective for workloads with temporal locality.
**Cons:** Does not account for entry size or computation cost.

### 3.2 TTL (Time-to-Live)

Each cache entry has an expiration time. Entries are evicted when their TTL expires, regardless of access patterns. Useful for:
- SNN state caches (state becomes stale after a few time steps)
- Workspace competition results (may become irrelevant as context changes)

```python
@dataclass
class CacheEntry:
    value: Tensor
    created_at: float
    ttl_seconds: float
    access_count: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds
```

### 3.3 Size-Aware LRU

Evicts entries by LRU order but considers entry size. Prefers evicting large entries when memory pressure is high, even if they were recently accessed. This prevents a single large entry from evicting many small, frequently-used entries.

```python
def evict_size_aware(self, needed_bytes: int):
    """Evict entries preferring larger, older entries."""
    candidates = sorted(
        self.cache.items(),
        key=lambda kv: (kv[1].access_count, -kv[1].size_bytes)
    )
    freed = 0
    for key, entry in candidates:
        if freed >= needed_bytes:
            break
        self._remove(key)
        freed += entry.size_bytes
```

### 3.4 Brain_AI-Specific Eviction Priorities

Not all cached values are equally expensive to recompute:

| Cached Value | Recompute Cost | Eviction Priority |
|---|---|---|
| Encoder outputs (vision) | High (ViT forward pass) | Keep (low priority eviction) |
| Encoder outputs (text) | Medium (transformer pass) | Keep |
| SNN spike patterns | Low (fast forward) | Evict first |
| HTM column states | Medium (sparse ops) | Medium |
| Workspace representations | High (cross-modal attention) | Keep |
| System 1 outputs | Low (single linear) | Evict first |
| Engram embeddings | High (hash lookups) | Keep |

The cache manager can use a cost-aware eviction policy that considers recompute cost when choosing which entries to evict.

---

## 4. Cache Invalidation

### 4.1 Model Update Invalidation

When model weights change (fine-tuning, meta-learning adaptation, continual learning), all cached outputs are stale. The cache must be invalidated.

**Full invalidation:** Clear all cache levels. Simple but wasteful if only a few modules changed.

**Module-level invalidation:** Track which modules were updated and only invalidate cache entries produced by those modules. Requires the composite key structure from Section 2.3.

```python
class CacheManager:
    def __init__(self):
        self.model_version = None

    def on_model_update(self, model: nn.Module, updated_modules: List[str] = None):
        """Called after model weights change."""
        new_version = self._compute_model_version(model)
        if new_version != self.model_version:
            if updated_modules is None:
                self.invalidate()  # Full invalidation
            else:
                for module_name in updated_modules:
                    self.invalidate_module(module_name)
            self.model_version = new_version
```

### 4.2 Input-Dependent Invalidation

Some cache entries depend on mutable external state:
- Engram hash tables may be updated during training.
- HTM learns new temporal patterns.
- Meta-learning adjusts neuromodulatory gains.

For these, use TTL-based expiration rather than explicit invalidation.

### 4.3 Invalidation Strategies by Module

| Module | Invalidation Trigger | Strategy |
|---|---|---|
| Encoders | Weight update | Module-level invalidation |
| SNN | Weight update, state drift | TTL (short, ~10s) |
| HTM | Pattern learning | TTL (medium, ~60s) |
| Workspace | Any module update | Full invalidation |
| Reasoning | Threshold change | Config-aware invalidation |
| Engram | Hash table update | Module-level invalidation |
| Meta | Modulator change | TTL (short, ~10s) |

---

## 5. Memory Budgeting

### 5.1 Budget Allocation

Total inference memory = model weights + activations + cache. The cache budget should be allocated to avoid OOM:

```python
def compute_cache_budget(model_size_mb: float, gpu_memory_mb: float,
                         activation_headroom_factor: float = 2.0) -> dict:
    """Compute cache memory budget given GPU constraints."""
    activation_mb = model_size_mb * activation_headroom_factor
    available_mb = gpu_memory_mb - model_size_mb - activation_mb

    l1_budget = max(0, available_mb * 0.25)
    l2_budget = 2048  # Default 2GB CPU
    return {"l1_mb": l1_budget, "l2_mb": l2_budget}
```

### 5.2 Budget by Configuration

| Config | GPU Memory | Model | Activations | L1 Cache | Available |
|--------|-----------|-------|------------|----------|----------|
| minimal (A100 80GB) | 80GB | 4MB | 8MB | 20GB | 59GB |
| 1B (A100 80GB) | 80GB | 4GB | 8GB | 17GB | 51GB |
| 3B (A100 80GB) | 80GB | 12GB | 24GB | 11GB | 33GB |
| 7B (A100 80GB) | 80GB | 28GB | 56GB | 0GB* | -4GB |

*The 7B model exceeds single-GPU capacity with activations. Use CPU offloading, gradient checkpointing, or multi-GPU inference. L1 cache is not viable; rely on L2 (CPU) and L3 (disk).

### 5.3 Dynamic Budget Adjustment

Monitor GPU memory usage at runtime and adjust cache budgets:

```python
def adjust_cache_budget(cache_manager, target_free_mb: float = 1024):
    """Shrink L1 cache if GPU memory is running low."""
    if torch.cuda.is_available():
        free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
        if free_mb < target_free_mb:
            while free_mb < target_free_mb and cache_manager.l1.cache:
                cache_manager.l1.evict_one()
                free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
```

---

## 6. Cache Warming Strategies

### 6.1 Cold Start Problem

After a model restart or cache invalidation, all caches are empty. The first batch of requests will be slow. Cache warming precomputes and caches results for expected inputs.

### 6.2 Warming Approaches

**Static warming:** Precompute cache entries for a fixed set of "common" inputs. Works well when the input distribution is known and stable.

```python
def warm_cache(model, cache_manager, common_inputs: List[Dict[str, Tensor]]):
    """Pre-populate cache with common inputs."""
    model.eval()
    with torch.inference_mode():
        for inp in common_inputs:
            output = model(inp)
            key = cache_manager.compute_key(inp)
            cache_manager.put(key, output)
```

**Replay warming:** After a restart, replay the most recent N requests from a log. This approximates the pre-restart cache state.

**Incremental warming:** During warmup, cache the warmup outputs. Since warmup runs at multiple batch sizes, this populates the cache with representative entries.

### 6.3 Warming for Brain_AI Modules

Different modules benefit from different warming strategies:

- **Encoders:** Warm with representative images/text from each class. Encoder outputs are the most reused cache entries.
- **Engram:** Preload the hash table into L2 cache. The engram embedding table is static between updates.
- **HTM:** Cannot meaningfully warm (stateful, sequence-dependent). Skip.
- **Workspace:** Warm after encoder caches are populated, since workspace results depend on encoder outputs.

---

## 7. Cache Coherence in Multi-GPU Setups

### 7.1 The Coherence Problem

When running inference on multiple GPUs (data parallelism or model parallelism), each GPU may have its own L1 cache. These caches can diverge:
- The same input processed on different GPUs produces the same output, but both GPUs cache it independently (wasted memory).
- After a model update on one GPU, other GPUs' caches are stale.

### 7.2 Shared L2/L3

L2 and L3 caches are naturally shared (CPU memory and disk are accessible from all GPUs). Only L1 (per-GPU) needs coherence management.

### 7.3 Broadcast Invalidation

When any GPU triggers a cache invalidation, broadcast the invalidation signal to all GPUs:

```python
def invalidate_distributed(cache_manager, process_group=None):
    """Broadcast cache invalidation across all ranks."""
    if torch.distributed.is_initialized():
        signal = torch.tensor([1], device="cuda")
        torch.distributed.broadcast(signal, src=0, group=process_group)
    cache_manager.invalidate()
```

---

## 8. Cache Statistics and Monitoring

### 8.1 CacheStats Dataclass

```python
@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    l1_size_mb: float = 0.0
    l2_size_mb: float = 0.0
    l3_size_mb: float = 0.0
    l1_entries: int = 0
    l2_entries: int = 0
    l3_entries: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

### 8.2 Metrics to Track

- **hit_rate:** Fraction of lookups that found a cached value. Target >80% for steady-state workloads.
- **eviction_rate:** Evictions per second. High values indicate cache is too small.
- **l1_utilization:** L1 memory used / L1 budget. Should be near 100% in steady state.
- **avg_lookup_time_us:** Average time for a cache lookup. Should be less than 1us for L1, less than 10us for L2.
- **memory_savings_mb:** Estimated GPU memory saved by caching (vs recomputing).

### 8.3 Cache Effectiveness Analysis

To determine if caching is helping:

1. Run inference with cache enabled and disabled.
2. Compare throughput and latency.
3. If cache hit rate is below 10%, the workload has low temporal locality and caching adds overhead without benefit. Consider disabling.
4. If cache hit rate exceeds 50% and speedup exceeds 2x, caching is highly effective.

---

## 9. Security and Correctness Considerations

### 9.1 Cache Poisoning

If an attacker can inject entries into the cache, they can cause the model to return incorrect results. Mitigations:
- Use cryptographic hashes (SHA-256) for cache keys.
- Validate cache entries against model outputs periodically.
- Do not expose the cache API to untrusted clients.

### 9.2 Determinism

Caching can mask non-determinism in the model. If the model produces different outputs for the same input (e.g., due to dropout, which should be disabled at inference), the cached value is always the first computation's result. Ensure `model.eval()` is called before caching.

### 9.3 Memory Leaks

Long-running inference servers can accumulate cache entries without bound if eviction is not working correctly. Monitor cache size over time and alert if it exceeds the budget by more than 10%.
