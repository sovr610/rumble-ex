# DeepSeek Engram Integration Design

**Date:** 2026-01-18
**Status:** Approved for implementation

## Overview

Integrate DeepSeek's Engram memory system into the brain-inspired AI architecture. Engram provides O(1) static pattern retrieval via n-gram hashing, complementing the existing memory systems (HTM, Working Memory, Global Workspace).

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Integration approach | Both encoder-style and layer-style (phased) | Quick wins first, then deeper integration |
| Tokenizer handling | Configurable (shared/dedicated) | Flexibility for experimentation |
| N-gram orders | Standard [2, 3] | Proven in paper, reasonable memory |
| Hash table size | Production scale (~10M entries) | Lower collision rate, serious deployment |

## Architecture

### Cognitive Mapping

```
BIOLOGICAL                    AI IMPLEMENTATION
──────────────                ─────────────────
Semantic Memory    ────────►  Engram (N-gram lookup) - O(1) retrieval
Episodic Memory    ────────►  HTM / Sequence Memory - Online learning
Working Memory     ────────►  Global Workspace (Liquid NNs) - Capacity-limited
Procedural         ────────►  SNN / Transformer weights - Learned computation
```

### File Structure

```
brain_ai/
├── memory/                          # New directory
│   ├── __init__.py
│   ├── engram.py                    # Core Engram module (~400 lines)
│   ├── tokenizer_compression.py     # Tokenizer normalization (~100 lines)
│   └── hash_embedding.py            # Multi-head hash + embedding (~150 lines)
├── encoders/
│   └── engram_encoder.py            # Phase 1: Encoder-style (~100 lines)
└── layers/
    └── engram_layer.py              # Phase 2: Layer-style (~150 lines)
```

## Phase 1: Core Module

### TokenizerCompression

Maps semantically equivalent tokens to canonical IDs:
- "Apple" → "apple"
- " the" → "the"
- "\n" → " "

Achieves ~23% vocabulary reduction while preserving semantics.

```python
class TokenizerCompression:
    def __init__(self, tokenizer, mode: str = "shared"):
        self.projection = self._build_compression_map(tokenizer)

    def compress(self, token_ids: Tensor) -> Tensor:
        return self.projection[token_ids]
```

### MultiHeadHash

K independent hash functions per n-gram order to reduce collisions:

```python
# φ_{n,k}(g) = (Σ_i c_{i,k} * x_i) XOR seed_k mod M

class MultiHeadHash(nn.Module):
    def __init__(self, table_size: int, num_heads: int, ngram_orders: tuple):
        self.coefficients = nn.ParameterDict()  # Per order, per head
        self.seeds = {}  # Fixed random seeds per head

    def forward(self, ngrams: Tensor) -> Tensor:
        # Returns: (batch, seq, num_orders, num_heads)
```

### EngramEmbedding

O(1) lookup via hash indices:

```python
class EngramEmbedding(nn.Module):
    def __init__(self, config: EngramConfig):
        self.tables = nn.ModuleDict()
        for order in config.ngram_orders:
            for head in range(config.num_heads):
                dim = config.embedding_dim // (len(config.ngram_orders) * config.num_heads)
                self.tables[f"{order}_{head}"] = nn.Embedding(config.table_size, dim)

    def forward(self, hash_indices: Tensor) -> Tensor:
        # Lookup and concatenate across orders and heads
```

### ContextAwareGating

Gates retrieved memory based on transformer hidden state:

```python
# α = σ(RMSNorm(h)ᵀ · RMSNorm(Ke) / √d)
# Output = α · Ve

class ContextAwareGating(nn.Module):
    def __init__(self, hidden_dim: int, memory_dim: int):
        self.query_norm = RMSNorm(hidden_dim)
        self.key_proj = nn.Linear(memory_dim, hidden_dim)
        self.value_proj = nn.Linear(memory_dim, hidden_dim)

    def forward(self, hidden: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        # Returns (gated_output, gate_values)
```

### Data Flow

```
Token IDs → Compress → Extract N-grams → Multi-Head Hash → Embedding Lookup
                                                                ↓
Hidden State ────────────────────────────► Context-Aware Gating
                                                                ↓
                                                    Gated Memory Output
```

## Phase 2: Encoder-Style Integration

Engram as a fast text encoder competing in Global Workspace:

```python
class EngramTextEncoder(nn.Module):
    def __init__(self, config: EngramConfig, output_dim: int):
        self.compression = TokenizerCompression(config)
        self.ngram_extractor = NGramExtractor(config.ngram_orders)
        self.hash = MultiHeadHash(config)
        self.embedding = EngramEmbedding(config)
        self.output_proj = nn.Linear(config.embedding_dim, output_dim)

    def forward(self, token_ids: Tensor) -> Tensor:
        compressed = self.compression(token_ids)
        ngrams = self.ngram_extractor(compressed)
        hashes = self.hash(ngrams)
        memory = self.embedding(hashes)
        pooled = memory.mean(dim=1)
        return self.output_proj(pooled)
```

Integration in `BrainAI.forward()`:
- Engram encoder registered alongside other encoders
- Competes in workspace attention with vision, text, audio
- Attention weights reveal when Engram wins vs transformer

## Phase 3: Layer-Style Integration

Engram embedded within transformer layers at layer 1:

```python
class EngramAugmentedLayer(nn.Module):
    def forward(self, x: Tensor, token_ids: Tensor = None) -> Tensor:
        # 1. Engram (semantic memory - O(1) static patterns)
        if self.use_engram and token_ids is not None:
            engram_out, gate = self.engram(token_ids, x)
            x = x + engram_out

        # 2. SNN (temporal dynamics)
        if hasattr(self, 'snn'):
            snn_out, spikes = self.snn(x)
            x = x + snn_out

        # 3. Attention (global context)
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out

        # 4. FFN (nonlinear transform)
        x = x + self.ffn(self.norm2(x))

        return self.norm3(x)
```

**Why layer 1?** (from paper ablation):
- Too early (layer 0): Hidden states lack context for gating
- Too late (layer 10+): Model already wasted depth on pattern reconstruction
- Optimal (layer 1-2): Balances context availability with early intervention

## Phase 4: CPU Offloading & Prefetching

For production scale (10M entries, ~5GB):

```python
class OffloadableEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 offload: bool = False, prefetch: bool = True):
        if offload:
            self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))
            self.weight.data = self.weight.data.pin_memory()
            self._prefetch_stream = torch.cuda.Stream()

    def prefetch_async(self, indices: Tensor):
        """Call during previous layer to hide transfer latency."""
        with torch.cuda.stream(self._prefetch_stream):
            unique_indices = indices.unique()
            self._prefetch_buffer = (
                unique_indices,
                self.weight[unique_indices.cpu()].cuda(non_blocking=True)
            )

    def forward(self, indices: Tensor) -> Tensor:
        if self._prefetch_buffer is not None:
            torch.cuda.current_stream().wait_stream(self._prefetch_stream)
            return self._gather_from_cache(indices, *self._prefetch_buffer)
        return self.weight[indices.cpu()].cuda()
```

**Expected overhead:** 1.9% for 100B parameters (from paper measurements)

## Configuration

```python
@dataclass
class EngramConfig:
    # Hash table
    table_size: int = 10_000_003      # Prime number, production scale
    num_heads: int = 4                 # Hash heads per n-gram order
    ngram_orders: tuple = (2, 3)       # Bigrams and trigrams

    # Embeddings
    embedding_dim: int = 256           # Per-order dimension

    # Tokenizer
    tokenizer_mode: str = "shared"     # "shared" or "dedicated"
    vocab_size: int = 50_000           # For compression mapping

    # Offloading
    offload_to_cpu: bool = False       # For large tables
    prefetch: bool = True              # Enable prefetching

    # Gating
    gate_temperature: float = 1.0      # Softmax temperature

# Feature flag in BrainAIConfig
use_engram: bool = False
engram_layer_idx: int = 1
```

## Testing Strategy

### Unit Tests (`tests/test_engram.py`)

1. **TokenizerCompression**: Verify ~23% reduction, semantic equivalence
2. **MultiHeadHash**: Uniform distribution, low collision rate
3. **EngramEmbedding**: O(1) lookup time independent of table size
4. **ContextAwareGating**: Suppresses irrelevant memory (α → 0)

### Integration Tests

1. **EngramEncoder**: Output shape matches other encoders
2. **Workspace competition**: Engram participates in attention
3. **EngramLayer**: Full pipeline works (engram → snn → attn → ffn)
4. **Offload correctness**: CPU offload produces same results as GPU

### Validation

1. **Throughput**: Engram adds <5% overhead
2. **Memory**: Offload reduces GPU usage
3. **Quality**: Formulaic phrase completion improves

## Expected Benefits

From Engram paper results:

| Benchmark | Without Engram | With Engram | Gain |
|-----------|----------------|-------------|------|
| MMLU | 57.4 | 60.4 | +3.0 |
| BBH | 50.9 | 55.9 | +5.0 |
| HumanEval | 37.8 | 40.8 | +3.0 |

Key insight: Gains are NOT limited to knowledge tasks. Larger gains in reasoning (BBH +5.0) because Engram frees early layers from pattern reconstruction, effectively deepening the network for complex reasoning.

## Implementation Order

1. Create `brain_ai/memory/` directory with core modules
2. Add `EngramConfig` to `config.py`
3. Implement `EngramTextEncoder` (Phase 1)
4. Update `BrainAI` to use engram encoder
5. Add tests for Phase 1
6. Implement `EngramAugmentedLayer` (Phase 2)
7. Add CPU offloading
8. Full integration tests
