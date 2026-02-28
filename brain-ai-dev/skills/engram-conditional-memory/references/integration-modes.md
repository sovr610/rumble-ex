# Integration Modes Reference

Detailed implementation reference for the two Engram integration modes within the
brain_ai architecture. Load this when Claude needs deep implementation details for
the encoder-competition pipeline (Phase 1), the layer-augmentation pipeline
(Phase 2), prefetch hook scheduling, telemetry reporting, checkpoint format,
feature flag configuration, ablation support, or cross-module interactions.

---

## 1. Overview

Engram conditional memory participates in the brain_ai cognitive pipeline through
two complementary integration modes. Each mode serves a distinct role and can be
enabled independently via feature flags in `BrainAIConfig`.

| Mode | Class | File | Role |
|---|---|---|---|
| Phase 1: Encoder-Competition | `EngramTextEncoder` | `brain_ai/encoders/engram_encoder.py` | Produces workspace-aligned representations that compete with other modality encoders in the Global Workspace |
| Phase 2: Layer-Augmentation | `EngramAugmentedLayer` | `brain_ai/layers/engram_layer.py` | Injected at selected backbone layers; adds retrieved memory residually before attention/FFN |

The two modes share the same underlying primitives -- `TokenizerCompression`,
`MultiHeadHash`, `OffloadableEmbedding`, `ContextAwareGating`, and depthwise
causal convolution -- but wire them differently depending on the integration
point in the forward pass.

### When to Use Each Mode

**Phase 1 (Encoder-Competition)** is the lightweight path. The Engram encoder
stands alongside `VisionEncoder`, `TextEncoder`, `AudioEncoder`, and
`SensorEncoder` as a peer modality source. The Global Workspace attention
mechanism learns when to prefer the Engram representation (fast O(1) pattern
retrieval) versus the transformer-based text representation (full compositional
processing). Use when:

- The workload is dominated by formulaic, idiomatic, or named-entity-heavy text
- Low latency is critical and the cost of full transformer encoding is excessive
- Ablation studies compare retrieval-based vs. attention-based text understanding

**Phase 2 (Layer-Augmentation)** is the deep integration path. Engram deltas
are injected directly into the hidden states of a transformer (or SNN) backbone
at configurable layer indices. This is the approach described in the DeepSeek
Engram paper and provides the strongest performance gains because the backbone
can adapt its attention and FFN processing to the presence of retrieved memory.
Use when:

- Production accuracy matters most
- The model has a deep backbone (12+ layers) with clear layer boundaries
- Prefetch scheduling can overlap retrieval with backbone compute

**Both modes can be active simultaneously.** Phase 1 provides a fast text signal
for workspace competition while Phase 2 enriches internal representations at
depth. The feature flags `use_engram_encoder` and `use_engram_layers` control
each independently.

### Relationship to Existing Code

| File | Key Classes | Lines (approx.) |
|---|---|---|
| `brain_ai/encoders/engram_encoder.py` | `EngramTextEncoder`, `create_engram_encoder` | ~118 |
| `brain_ai/layers/engram_layer.py` | `EngramAugmentedLayer`, `create_engram_layer` | ~165 |
| `brain_ai/memory/engram.py` | `EngramConfig`, `EngramEmbedding`, `ContextAwareGating`, `EngramModule`, `RMSNorm` | ~355 |
| `brain_ai/memory/tokenizer_compression.py` | `TokenizerCompression` | ~116 |
| `brain_ai/memory/hash_embedding.py` | `MultiHeadHash`, `OffloadableEmbedding` | ~187 |
| `brain_ai/config.py` | `EngramConfig` (system-level), `BrainAIConfig` | ~559 |
| `brain_ai/system.py` | `BrainAI` (orchestrator, wires both modes) | ~544 |

---

## 2. Phase 1: Encoder-Competition Mode

### 2.1 Module Signature

`EngramTextEncoder` lives in `brain_ai/encoders/engram_encoder.py`.

```python
class EngramTextEncoder(nn.Module):
    def __init__(
        self,
        config: EngramConfig,        # from brain_ai.memory.engram
        output_dim: int = 512,       # must match workspace_dim for competition
        use_positional: bool = True,  # add learned positional encoding
        max_seq_len: int = 2048,
    ): ...

    def forward(
        self,
        token_ids: torch.Tensor,          # (B, T) int64, raw token IDs
        attention_mask: Optional[Tensor],  # (B, T) float/bool, 1=valid 0=pad
    ) -> torch.Tensor:                    # (B, D_workspace)
        ...
```

### 2.2 Internal Pipeline

```
token_ids (B, T)
    |
    v
[TokenizerCompression]  -- surjective map to canonical IDs
    v
canonical_ids (B, T)
    v
[N-gram Extraction]     -- extract_ngrams for each order n in ngram_orders
    v
ngrams_n (B, T, n) for each n
    v
[MultiHeadHash]         -- K hash functions per order, multiplicative-XOR
    v
hash_ids (B, T, K) per order
    v
[OffloadableEmbedding]  -- gather from K tables per order
    v
per_head_embs [(B, T, dim_per_head)] x (num_orders * K)
    v
[Concatenation]         -- cat along embedding dim
    v
raw_memory (B, T, embedding_dim)
    v
[Positional Encoding]   -- add learned pos_encoding[:, :T, :]
    v
positioned_memory (B, T, embedding_dim)
    v
[Masked Mean Pooling]   -- pool over T using attention_mask
    v
pooled (B, embedding_dim)
    v
[Output Projection]     -- Linear -> GELU -> LayerNorm
    v
output (B, D_workspace)
```

#### Shape Tracking

| Stage | Shape | Notes |
|---|---|---|
| Input `token_ids` | `(B, T)` | int64, range `[0, vocab_size)` |
| Compressed IDs | `(B, T)` | int64, range `[0, compressed_vocab_size)` |
| N-grams (order n) | `(B, T, n)` | zero-padded at start for positions < n |
| Hash IDs per head | `(B, T, K)` | K=`num_heads`, int64, range `[0, table_size)` |
| Per-head embedding | `(B, T, dim_per_head)` | `dim_per_head = embedding_dim // (num_orders * K)` |
| Concatenated memory | `(B, T, embedding_dim)` | e.g., 4096 for production |
| Masked mean pool | `(B, embedding_dim)` | T dimension collapsed |
| Output projection | `(B, D_workspace)` | D_workspace = 4096 (unified workspace dim) |

#### Dimension Calculation

```
total_retrievals = len(ngram_orders) * num_heads
dim_per_embedding = embedding_dim // total_retrievals
```

Production (`ngram_orders=(2,3,4)`, `num_heads=32`, `embedding_dim=4096`):
`3 * 32 = 96` retrievals, `4096 // 96 = 42` dim per head.

Dev (`ngram_orders=(2,3)`, `num_heads=4`, `embedding_dim=256`):
`2 * 4 = 8` retrievals, `256 // 8 = 32` dim per head.

**Invariant**: `dim_per_embedding * total_retrievals == embedding_dim` must hold.
Choose compatible configs to avoid truncation.

### 2.3 Gating in Phase 1 (No Backbone Hidden States)

In Phase 2, `ContextAwareGating` uses backbone hidden states as the query signal.
In Phase 1 there is no backbone -- `EngramTextEncoder` IS the encoder. The current
implementation omits explicit gating:

1. Raw retrieved embeddings are used directly after concatenation
2. Positional encoding provides sequence-position awareness
3. The output projection (`Linear -> GELU -> LayerNorm`) acts as a learned
   transformation that implicitly gates relevant information

If explicit gating is desired (for ablation or future development), a lightweight
learned projection of the raw embeddings can serve as the "hidden states":

```python
# Optional Phase 1 gating (not in current implementation)
class Phase1Gating(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_proj = nn.Linear(embedding_dim, hidden_dim)
        self.gate = ContextAwareGating(hidden_dim, embedding_dim)

    def forward(self, memory):
        pseudo_hidden = self.hidden_proj(memory)  # (B, T, hidden_dim)
        gated, alpha = self.gate(pseudo_hidden, memory)
        return gated, alpha
```

The hidden projection starts random, so the gate initializes near 0.5. If
`use_context_gate` is True in Phase 1 config, this path activates.

### 2.4 Mask Semantics

Phase 1 follows the encoder suite contract:

- `attention_mask` shape: `(B, T)`, values `{0, 1}`, where `1` = valid, `0` = pad
- Masked mean pooling: `sum(embeddings * mask) / sum(mask).clamp(min=1)`
- If `attention_mask is None`, uniform mean pooling across all T positions

N-gram extraction zero-pads the left boundary (`F.pad(token_ids, (n-1, 0), value=0)`).
The first `n-1` positions have partial N-grams with zero-filled prefixes. The
attention_mask does NOT zero them at this stage -- it only affects final pooling.
For stricter masking, apply `raw_memory = raw_memory * mask.unsqueeze(-1)` before
concatenation.

### 2.5 Registration in the Orchestrator

`BrainAI._build_encoders()` in `brain_ai/system.py` registers the Engram
encoder when `config.use_engram` is True:

```python
if self.config.use_engram:
    self.encoders['engram'] = create_engram_encoder(
        output_dim=encoder_dim,
        vocab_size=self.config.engram.vocab_size,
        embedding_dim=self.config.engram.embedding_dim,
        ngram_orders=self.config.engram.ngram_orders,
        num_heads=self.config.engram.num_heads,
        table_size=self.config.engram.table_size,
    )
```

The Engram encoder is included in the workspace `modality_dims` map and invoked
separately in the forward pass because it requires `token_ids`:

```python
if 'engram' in self.encoders and 'token_ids' in inputs:
    encoded['engram'] = self.encoders['engram'](inputs['token_ids'])
```

**Input requirement**: The caller must include `'token_ids'` in the `inputs` dict.
If only `'text'` is provided (pre-embedded), the Engram encoder is silently skipped.

### 2.6 Competition in the Global Workspace

The Engram `(B, D_workspace)` output competes alongside other encoder outputs:

```
                   +-----------+
 vision (B, D) --> |           |
   text (B, D) --> | Workspace | --> workspace (B, D)
  audio (B, D) --> | Attention |     + attention weights per modality
 engram (B, D) --> |           |
                   +-----------+
```

Empirical attention weight patterns:
- Formulaic/idiomatic text: `{"text": 0.25, "engram": 0.70, ...}`
- Novel compositional text: `{"text": 0.80, "engram": 0.15, ...}`
- Named entities: mixed (Engram provides entity embedding; TextEncoder
  provides compositional context)

---

## 3. Phase 2: Layer-Augmentation Mode

### 3.1 Module Signature

`EngramAugmentedLayer` lives in `brain_ai/layers/engram_layer.py`.

```python
class EngramAugmentedLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        engram_config: EngramConfig,
        snn_config: Optional[SNNConfig] = None,
        use_engram: bool = True,
        use_snn: bool = False,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ): ...

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, D) hidden states
        token_ids: Optional[torch.Tensor] = None,  # (B, T) int64
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                             # (B, T, D)
        ...
```

### 3.2 Internal Architecture

Pre-norm residual pipeline:

```
x (B, T, D)
    +--- [RMSNorm] --> [EngramModule] --> engram_delta --> x = x + engram_delta
    +--- [RMSNorm] --> [SNN] -----------> snn_delta ----> x = x + snn_delta [optional]
    +--- [RMSNorm] --> [MultiheadAttn] -> attn_out -----> x = x + attn_out
    +--- [RMSNorm] --> [FFN] -----------> ffn_out ------> x = x + ffn_out
    v
output (B, T, D)
```

**Integration point**: Engram delta is added BEFORE attention and FFN, matching
the DeepSeek paper. Attention and FFN adapt their processing based on retrieved
memory patterns.

### 3.3 EngramModule Pipeline (Shared Core)

Both modes use `EngramModule` (`brain_ai/memory/engram.py`) as the retrieval engine:

```
token_ids (B, T)  -->  [EngramEmbedding]  -->  memory (B, T, E)
                           |-- TokenizerCompression
                           |-- N-gram extraction per order
                           |-- MultiHeadHash per order
                           |-- OffloadableEmbedding per head
                           |-- Concatenate all heads

hidden_states (B, T, D) + memory (B, T, E)  -->  [ContextAwareGating]
    |-- W_K(memory) -> keys, W_V(memory) -> values
    |-- alpha = sigmoid(dot(query_norm(h), key_norm(k)) / sqrt(D) / temp)
    |-- gated = alpha * values

gated (B, T, D)  -->  [RMSNorm] --> [DepthwiseCausalConv1d] --> conv_out (B, T, D)

output = silu(conv_out) + gated  -->  delta for residual addition
```

Returns `(output, info)` where `info = {'gate_values': (B, T), 'memory_norm': scalar}`.

### 3.4 Insertion Layer Configuration

Not all backbone layers need Engram augmentation. Recommended insertion strategy:

| Backbone Depth | Recommended Insertion Layers | Rationale |
|---|---|---|
| 12 layers | `[3, 6, 9]` | Every 3rd layer |
| 24 layers | `[4, 8, 12, 16]` | Every 4th, skip bottom and top |
| 32 layers | `[4, 8, 12, 16, 20, 24]` | Every 4th layer |
| 48 layers | `[6, 12, 18, 24, 30, 36]` | Every 6th layer |

The system-level `engram_layer_idx: int = 2` (legacy single layer) is extended to
`engram_insertion_layers: list[int]` for multi-layer insertion.

### 3.5 Per-Layer Salt for Hash Decorrelation

Without per-layer salts, identical N-grams produce identical hash IDs at every
insertion layer -- every layer retrieves the same rows. Per-layer salts decorrelate
by XOR-ing a layer-specific value into the hash seeds:

```python
class LayerAwareMultiHeadHash(MultiHeadHash):
    def __init__(self, ngram_order, num_heads, table_size, seed, layer_salt):
        super().__init__(ngram_order, num_heads, table_size, seed)
        self.seeds = self.seeds ^ layer_salt

def generate_layer_salts(num_layers, base_seed=42):
    rng = np.random.RandomState(base_seed)
    return [rng.randint(0, 2**31) for _ in range(num_layers)]
```

**Invariant**: Salts are generated once at initialization and frozen into the
checkpoint. They must NOT change between training and inference.

### 3.6 Compute-Now vs. Consume-Prefetched Paths

**Path A: Compute-Now (Default)** -- all hash/retrieve/gate operations inline:

```python
x_norm = self.engram_norm(x)
engram_out, info = self.engram(token_ids, x_norm)
x = x + engram_out
```

**Path B: Consume-Prefetched (Offload Mode)** -- hash IDs precomputed for ALL
insertion layers at forward-pass start; embeddings prefetched asynchronously:

```python
def forward(self, x, token_ids, attention_mask, prefetched_embeddings=None):
    x_norm = self.engram_norm(x)
    if prefetched_embeddings is not None:
        memory = prefetched_embeddings  # (B, T, embedding_dim)
        gated, alpha = self.gating(x_norm, memory)
        # Continue with conv + residual...
    else:
        engram_out, info = self.engram(token_ids, x_norm)  # fallback
        x = x + engram_out
```

Key insight: hash IDs depend only on `token_ids` (not `hidden_states`), so they
can be computed once and reused. Only gating requires per-layer `hidden_states`.

---

## 4. Prefetch Hook Scheduling

### 4.1 Prefetch Plan

At forward-pass start, the orchestrator computes hash IDs for all insertion layers:

```python
class PrefetchPlan:
    def __init__(self, token_ids, insertion_layers, engram_modules):
        self.hash_ids = {}  # layer_idx -> (B, T, total_heads) int64
        for layer_idx in insertion_layers:
            self.hash_ids[layer_idx] = engram_modules[layer_idx].compute_hash_ids(token_ids)
```

This is lightweight (integer hash only, no embedding gather) and takes < 1ms.

### 4.2 Hook API

The backbone calls `engram_prefetch_hook(layer_idx, input_ids, stream)` at each
layer boundary. The hook triggers async prefetch N layers ahead:

```python
def engram_prefetch_hook(layer_idx, prefetch_plan, engram_modules, prefetch_stream):
    target_layer = layer_idx + prefetch_ahead_layers
    if target_layer not in prefetch_plan.hash_ids:
        return
    hash_ids = prefetch_plan.get_prefetch_ids(target_layer)
    module = engram_modules[target_layer]
    with torch.cuda.stream(prefetch_stream):
        for order_idx, n in enumerate(module.ngram_orders):
            for head_idx in range(module.num_heads):
                key = f"ngram{n}_head{head_idx}"
                head_ids = hash_ids[:, :, order_idx * module.num_heads + head_idx]
                module.embeddings[key].prefetch_async(head_ids)
```

### 4.3 Scheduling Timeline

For 24-layer backbone, Engram at `[4, 8, 12, 16]`, `prefetch_ahead_layers=2`:

```
Layer 2:  PREFETCH for layer 4 starts (async on prefetch stream)
Layer 3:  prefetch continues (overlaps with layer 3 compute)
Layer 4:  sync stream; CONSUME prefetched embeddings
Layer 6:  PREFETCH for layer 8 starts
Layer 7:  prefetch continues
Layer 8:  sync stream; CONSUME prefetched embeddings
Layer 10: PREFETCH for layer 12 starts
...
```

### 4.4 Configurable Prefetch Depth

| `prefetch_ahead_layers` | Behavior | Tradeoff |
|---|---|---|
| 0 | Synchronous gather | Maximum latency, zero overlap |
| 1 | One layer early | Minimal overlap |
| 2 (recommended) | Two layers early | Good overlap for typical layer compute |
| 3+ | Three+ layers early | Better overlap, more pinned memory |

### 4.5 CUDA Stream Synchronization

```python
# Prefetch (on dedicated stream)
with torch.cuda.stream(self._prefetch_stream):
    unique_ids = flat_ids.unique()
    embeddings = self.weight[unique_ids.cpu()].cuda(non_blocking=True)
    self._prefetch_buffer = (unique_ids, embeddings)

# Consumption (on compute stream) -- MUST sync first
torch.cuda.current_stream().wait_stream(self._prefetch_stream)
cached_indices, cached_embeddings = self._prefetch_buffer
```

**Critical**: `wait_stream()` must be called BEFORE accessing the buffer.
Missing this causes race conditions.

### 4.6 Coalescing for Bandwidth Efficiency

The same N-gram at multiple positions produces duplicate IDs. Prefetch coalesces
to unique IDs, then reconstructs via inverse mapping at consumption:

```python
def _gather_from_cache(self, indices, cached_indices, cached_embeddings):
    index_to_pos = torch.zeros(self.num_embeddings, dtype=torch.long, device=device)
    index_to_pos[cached_indices] = torch.arange(len(cached_indices), device=device)
    cache_positions = index_to_pos[indices.flatten()]
    return cached_embeddings[cache_positions].view(*indices.shape, self.embedding_dim)
```

Coalescing ratio (`unique / total`): ~0.7-0.9 for short sequences, ~0.4-0.6
for long sequences, ~0.3-0.5 for entity-heavy text. Lower = more savings.

---

## 5. Telemetry Reporting

### 5.1 Activation

When `return_details=True`, `EngramModule.forward()` returns an `info` dict.
At the system level, telemetry is assembled and placed into
`SystemOutput.attention["engram_telemetry"]`.

### 5.2 Gate Statistics

```python
gate_telemetry = {
    "gate_mean": float,          # Mean alpha across (B, T)
    "gate_std": float,
    "gate_min": float,
    "gate_max": float,
    "gate_sparsity": float,      # Fraction where alpha < 1e-6
    "gate_saturation": float,    # Fraction where alpha > 1 - 1e-6
    "per_layer": {               # Phase 2 only
        4:  {"mean": float, "std": float, "sparsity": float},
        8:  {"mean": float, "std": float, "sparsity": float},
        ...
    },
    "aggregate_mean": float,
    "aggregate_std": float,
}
```

| Pattern | Gate Mean | Gate Sparsity | Interpretation |
|---|---|---|---|
| Healthy | 0.2 - 0.6 | 0.1 - 0.4 | Selective memory usage |
| Collapsed (off) | < 0.05 | > 0.9 | Not contributing; check gradient flow |
| Saturated (on) | > 0.95 | < 0.01 | Dominating; lower gate_temperature |
| Bimodal | ~0.5 mean | < 0.1 both | Good specialization |

### 5.3 Collision Proxies

```python
collision_telemetry = {
    "unique_id_ratio": {   # per head/order/layer: unique_ids / total_positions
        "ngram2_head0_layer4": float, ...
    },
    "mean_unique_ratio": float,
    "min_unique_ratio": float,
    "collision_hotspot": str,
}
```

Target unique ratios: >0.9 for 10M tables, >0.95 for 100M tables.

### 5.4 Prefetch Statistics (Offload Mode Only)

```python
prefetch_telemetry = {
    "bytes_transferred": {4: int, 8: int, ...},
    "unique_rows_fetched": {4: int, 8: int, ...},
    "coalescing_ratio": {4: float, 8: float, ...},
    "overlap_timing_ms": {4: float, 8: float, ...},
    "total_bytes_transferred": int,
    "mean_coalescing_ratio": float,
    "prefetch_hit_rate": float,
}
```

### 5.5 Compression Statistics

```python
compression_telemetry = {
    "compression_ratio": float,            # ~0.77 for 128k tokenizer
    "num_equivalence_classes": int,
    "largest_equivalence_class": int,
    "singleton_fraction": float,
    "table_hash": str,                     # SHA-256 for version tracking
}
```

### 5.6 Assembled Output

```python
engram_telemetry = {
    "gate": gate_telemetry,
    "collision": collision_telemetry,
    "prefetch": prefetch_telemetry,        # Only when offload enabled
    "compression": compression_telemetry,
    "per_layer": {4: {...}, 8: {...}, ...}, # Phase 2 only
    "aggregate": {
        "gate_mean": float,
        "gate_sparsity": float,
        "mean_unique_ratio": float,
        "total_bytes_transferred": int,
    },
}
```

---

## 6. Checkpoint Format

### 6.1 Components

| Component | Format | Required | Description |
|---|---|---|---|
| Compression table | `.engram_compression.pt` | Yes | Surjective mapping `(vocab_size,)` int64 + metadata hash |
| Hash config | `.engram_hash_config.json` | Yes | Head sizes, salts, multipliers, prime table sizes, per-layer seeds |
| Embedding weights | in `state_dict` | Yes | All `ngram{n}_head{k}` embedding tables |
| Gating weights | in `state_dict` | Yes | W_K, W_V, query_norm, key_norm, RMSNorm gammas |
| Conv weights | in `state_dict` | Yes | Depthwise causal conv weight + bias |
| Optimizer state | separate `.opt` file | Training only | Adam/AdamW state for Engram parameters |
| Module config | `.engram_config.json` | Yes | Full `EngramConfig` serialization + schema version |

### 6.2 State Dict Key Prefixing

```python
# Phase 1 encoder keys
encoders.engram.engram_embedding.embeddings.ngram2_head0.embedding.weight
encoders.engram.pos_encoding
encoders.engram.output_proj.0.weight   # Linear
encoders.engram.output_proj.2.weight   # LayerNorm gamma

# Phase 2 layer keys (per insertion layer)
layer_4.engram.ngram_embedding.embeddings.ngram2_head0.embedding.weight
layer_4.engram.gating.W_K.weight
layer_4.engram.gating.W_V.weight
layer_4.engram.gating.query_norm.weight
layer_4.engram.gating.key_norm.weight
layer_4.engram.conv.weight
layer_4.engram.conv.bias
layer_4.engram.conv_norm.weight
layer_4.engram_norm.weight
# Repeat for layers 8, 12, 16...
```

### 6.3 Compression Artifact

Saved separately because it is derived from the tokenizer (not from gradient
descent), potentially large, and shared between Phase 1 and Phase 2:

```python
def save_compression(compressor, path):
    torch.save({
        "projection": compressor.projection,
        "vocab_size": compressor.vocab_size,
        "compressed_size": compressor.compressed_size,
        "mode": compressor.mode,
        "metadata_hash": compute_table_hash(compressor.projection),
        "schema_version": 1,
    }, path)

def load_compression(path, expected_hash=None):
    data = torch.load(path)
    if expected_hash and data["metadata_hash"] != expected_hash:
        raise ValueError(f"Compression table hash mismatch")
    compressor = TokenizerCompression(
        vocab_size=data["vocab_size"],
        compressed_size=data["compressed_size"],
        mode=data["mode"],
    )
    compressor.projection = data["projection"]
    return compressor
```

### 6.4 Hash Config Artifact

Saved as JSON for human readability:

```json
{
    "schema_version": 1,
    "ngram_orders": [2, 3, 4],
    "num_heads": 32,
    "table_size": 100000007,
    "hash_fn": "mult_xor",
    "base_seed": 42,
    "per_layer_salts": {"4": 1847293654, "8": 982374561, ...},
    "coefficients": {"order_2": [[...]], "order_3": [[...]], ...},
    "seeds_per_head": {"order_2": [...], "order_3": [...], ...}
}
```

**Invariant**: If hash config loaded from checkpoint does not match the current
module's config, loading must fail with a clear error.

### 6.5 Version Compatibility

| Schema Version | Changes | Migration |
|---|---|---|
| 1 | Initial format | -- |
| 2 (planned) | Per-layer gate bias info | Fill default (-2.0) |
| 3 (planned) | Streaming hash support | Set `streaming=False` |

```python
def migrate_checkpoint(data, target_version):
    current = data.get("schema_version", 1)
    while current < target_version:
        if current == 1:
            data["gate_init_bias"] = -2.0; current = 2
        elif current == 2:
            data["streaming"] = False; current = 3
    data["schema_version"] = current
    return data
```

---

## 7. Feature Flag Configuration

### 7.1 Flag Hierarchy

```
use_engram (master)
    +-- use_engram_encoder     (Phase 1: encoder-competition)
    +-- use_engram_layers      (Phase 2: layer-augmentation)
    +-- use_tokenizer_compression  (compression step)
    +-- use_context_gate           (gating mechanism)
    +-- use_depthwise_conv         (causal convolution)
    +-- use_cpu_offload            (CPU offload mode)
    +-- use_async_prefetch         (async prefetch, requires offload)
```

### 7.2 Flag Definitions

| Flag | Default | Requires | Description |
|---|---|---|---|
| `use_engram` | `True` (prod) / `False` (minimal) | -- | Master toggle |
| `use_engram_encoder` | `True` | `use_engram` | Phase 1 mode |
| `use_engram_layers` | `True` | `use_engram` | Phase 2 mode |
| `use_tokenizer_compression` | `True` | `use_engram` | Compression; disable for ablation |
| `use_context_gate` | `True` | `use_engram` | Gating; disable for ablation |
| `use_depthwise_conv` | `True` | `use_engram` | Causal conv; disable for ablation |
| `use_cpu_offload` | `False` | `use_engram` | CPU offload for large tables |
| `use_async_prefetch` | `False` | `use_cpu_offload` | Async prefetch with CUDA stream |

### 7.3 Validation

```python
def validate_engram_flags(config):
    errors = []
    if not config.use_engram:
        return errors  # Master off: sub-flags ignored
    if config.use_async_prefetch and not config.use_cpu_offload:
        errors.append("use_async_prefetch requires use_cpu_offload")
    if config.use_engram_layers and not config.engram_insertion_layers:
        errors.append("use_engram_layers requires non-empty engram_insertion_layers")
    if config.use_cpu_offload and not torch.cuda.is_available():
        errors.append("use_cpu_offload requires CUDA")
    return errors
```

### 7.4 Presets

| Preset | Params | Offload | Orders | Heads | Table Size |
|---|---|---|---|---|---|
| `minimal()` | ~1M | No | (2,) | 2 | 1,009 |
| `dev()` | ~10M | No | (2, 3) | 4 | 10,000,003 |
| `production()` | ~2.5B | Yes | (2, 3, 4) | 32 | 100,000,007 |

Flags are read at `__init__` time and determine which subcomponents are created.
They CANNOT be toggled after construction without rebuilding the module.

---

## 8. Ablation Support

### 8.1 Principle

Every ablation is a configuration change, not a code change.

### 8.2 Key Ablations

#### (a) No Compression

`config.engram.use_compression = False` -- raw token IDs go directly to hashing.
Increases collision rate, decreases table utilization.

#### (b) No Gating

`config.use_context_gate = False` -- retrieved embeddings added directly via
`W_V(memory)` without the sigmoid gate. Worse training stability (2-5% accuracy
drop in reported experiments).

#### (c) No Conv

`config.use_depthwise_conv = False` -- gated output added directly. Reduces
receptive field beyond the N-gram window. Slight speed gain.

#### (d) No Per-Layer Salt

`config.engram.per_layer_salt = False` -- same hash mapping at all layers.
Reduces layer diversity and effective capacity.

#### (e) Single Head Per Order

`config.engram.num_heads = 1` -- higher collision rate, no cross-head averaging.

### 8.3 Configuration Matrix

| Ablation | Flag(s) | Phase 1 Effect | Phase 2 Effect |
|---|---|---|---|
| no_compression | `use_tokenizer_compression=False` | Raw IDs to hash | Raw IDs to hash |
| no_gating | `use_context_gate=False` | Direct add | Direct add before attn |
| no_conv | `use_depthwise_conv=False` | No effect (no conv) | Skip conv |
| no_salt | `per_layer_salt=False` | No effect (single layer) | Same hash all layers |
| single_head | `num_heads=1` | One hash per order | One hash per order |
| no_phase1 | `use_engram_encoder=False` | Disabled | Phase 2 only |
| no_phase2 | `use_engram_layers=False` | Phase 1 only | Disabled |
| engram_off | `use_engram=False` | Both disabled | Both disabled |

### 8.4 Running Ablations

```python
ablation_configs = {
    "baseline": {},
    "no_compression": {"use_tokenizer_compression": False},
    "no_gating": {"use_context_gate": False},
    "no_conv": {"use_depthwise_conv": False},
    "no_salt": {"per_layer_salt": False},
    "single_head": {"num_heads": 1},
    "no_phase1": {"use_engram_encoder": False},
    "no_phase2": {"use_engram_layers": False},
    "no_engram": {"use_engram": False},
}

for name, overrides in ablation_configs.items():
    config = BrainAIConfig()
    for key, val in overrides.items():
        if hasattr(config.engram, key):
            setattr(config.engram, key, val)
        elif hasattr(config, key):
            setattr(config, key, val)
    model = BrainAI(config)
    results[name] = train_and_evaluate(model, dataset)
```

---

## 9. Interaction with Other Modules

### 9.1 Global Workspace

**Phase 1**: Engram encoder output enters workspace as a peer modality, competing
via attention weights. Workspace learns to attend to Engram for formulaic text and
to TextEncoder for compositional text.

**Phase 2**: No direct workspace interaction. Engram deltas modify hidden states
within backbone layers; modified states reach workspace through normal pipeline.

### 9.2 SNN Core

`EngramAugmentedLayer` optionally includes SNN processing after Engram delta:
`x -> [Engram] -> [SNN] -> [Attention] -> [FFN]`. The SNN processes Engram-enriched
hidden states -- semantic memory (Engram) feeds cortical column processing (SNN).

### 9.3 HTM

Complementary memory types:

| Aspect | Engram | HTM |
|---|---|---|
| Memory type | Static N-gram patterns (semantic) | Dynamic temporal sequences (episodic) |
| Lookup | O(1) hash-based, deterministic | Sequence matching, prediction-based |
| Adaptation | Trained embeddings (slow) | Online permanence updates (fast) |

No direct connection; they interact through the shared workspace representation.

### 9.4 Reasoning System

Engram gate telemetry can inform System 1/System 2 routing:

| Gate Sparsity | Interpretation | Routing |
|---|---|---|
| < 0.2 | Most positions use memory | Prefer System 1 (fast) |
| 0.2 - 0.6 | Mixed known/novel | Balanced |
| > 0.6 | Most positions suppress memory | Prefer System 2 (deliberate) |

This connection is optional and requires explicit wiring in the reasoning config.

### 9.5 Meta-Learning

During MAML inner loop:

- **Embedding tables**: FREEZE. Hash-addressed rows are input-dependent; sparse
  updates from few examples are noisy and affect all inputs hashing to those rows.
- **Gating parameters**: ADAPT. Gate learns task-specific trust/suppress decisions.
- **Conv parameters**: FREEZE. Small, well-trained, task-agnostic.

```python
for name, param in model.named_parameters():
    if "engram_embedding" in name or "embeddings" in name:
        param.requires_grad_(False)
    elif "gating" in name:
        param.requires_grad_(True)
    elif "conv" in name:
        param.requires_grad_(False)
```

Neuromodulatory system can modulate gate temperature dynamically:
`gate_temperature = base_temperature * (1.0 + ach_signal)` -- higher ACh
produces sharper gates and more selective memory retrieval.

---

## 10. Initialization and Startup

### 10.1 Sequence

```
Step 1: Build tokenizer compression table (or load from cache)
    v
Step 2: Initialize hash multipliers and prime table sizes
    v
Step 3: Allocate embedding tables (on device or host per offload config)
    v
Step 4: Initialize gating projections (Xavier) and conv weights (ZERO init)
    v
Step 5: If offload: allocate pinned buffers, create prefetch CUDA stream
    v
Step 6: If Phase 2 + offload: register prefetch hooks in backbone
    v
Step 7: Validate all components with a dummy forward pass
```

### 10.2 Compression Table

```python
def build_compression_table(config, tokenizer=None):
    cache_path = config.compression_cache_path
    if cache_path and os.path.exists(cache_path):
        compressor = load_compression(cache_path)
        assert compressor.vocab_size == config.vocab_size
        return compressor
    compressor = TokenizerCompression(
        vocab_size=config.vocab_size,
        compressed_size=config.compressed_vocab_size,
        mode=config.tokenizer_mode,
        tokenizer=tokenizer,
    )
    if cache_path:
        save_compression(compressor, cache_path)
    return compressor
```

Building from a real tokenizer can take seconds for 128K vocab; the cache avoids
repeat cost.

### 10.3 Hash Initialization

Hash function coefficients and seeds are generated from a fixed seed and stored
as buffers. They are NOT learnable parameters.

### 10.4 Embedding Allocation

Memory budget:

| Config | Tables | Rows | Dim | Bytes (fp32) | Bytes (fp16) |
|---|---|---|---|---|---|
| Dev | 8 | 10M each | 32 | ~2.4 GB | ~1.2 GB |
| Prod | 96 | 100M each | 42 | ~1.5 TB | ~750 GB |

Production tables REQUIRE CPU offload. Per-forward-pass GPU memory is much smaller
because only coalesced unique rows are transferred.

### 10.5 Weight Initialization

- **Gating projections**: Xavier/Glorot uniform
- **RMSNorm**: ones init (standard)
- **Conv**: ZERO init -- Engram module produces zero output at training start,
  ensuring the model starts from pre-trained baseline and gradually learns to
  use memory

### 10.6 Prefetch Infrastructure

If offload enabled: move tables to CPU, pin memory, create prefetch stream,
allocate staging buffers. If Phase 2 + offload: register forward hooks on
backbone layers at `layer_idx = insertion_idx - prefetch_ahead_layers`.

### 10.7 Validation

Run a dummy forward pass (`B=2, T=16`) through both Phase 1 and Phase 2
components. Assert output shapes match expectations and no NaN values appear.

---

## 11. Invariants and Contracts Summary

### Shape Contracts

| Interface | Input | Output | Invariant |
|---|---|---|---|
| `EngramTextEncoder.forward` | `(B, T)` int, `(B, T)` mask | `(B, D_ws)` | D_ws matches other encoders |
| `EngramAugmentedLayer.forward` | `(B, T, D)`, `(B, T)` int, `(B, T)` mask | `(B, T, D)` | Same shape as input |
| `EngramModule.forward` | `(B, T)` int, `(B, T, D)` float | `(B, T, D)` + dict | Delta shape = hidden shape |
| `TokenizerCompression.compress` | `(B, T)` int | `(B, T)` int | Range `[0, compressed_size)` |
| `MultiHeadHash.hash` | `(B, T, n)` int | `(B, T, K)` int | Range `[0, table_size)` |
| `ContextAwareGating.forward` | `(B, T, D)`, `(B, T, E)` | `(B, T, D)` + `(B, T)` | Gate in [0, 1] |

### Determinism

| Operation | Guarantee |
|---|---|
| Compression | Same inputs + same table -> identical outputs |
| Hashing | Same N-grams + same seed -> identical IDs (CPU and CUDA) |
| Embedding lookup | Same IDs + same weights -> identical embeddings |
| Gating | NOT deterministic across devices (float reduction order) |

### AMP Safety

| Component | Precision | Rationale |
|---|---|---|
| Hash computation | int64 always | Integer ops must be exact |
| RMSNorm in gating | fp32 forced | Prevents NaN under fp16/bf16 |
| Sigmoid gate | fp16/bf16 safe | Output bounded [0, 1] |
| Embedding lookup | fp32 or bf16 | fp16 can overflow for large embeddings |
| Conv | fp16/bf16 safe | Standard conv ops |
