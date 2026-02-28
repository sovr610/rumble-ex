# Multi-Head Deterministic Hashing for Engram Conditional Memory

## 1. Overview

The Engram conditional memory subsystem retrieves static embedding vectors from
hash tables indexed by deterministic functions of N-gram token sequences. The
core idea, described in the Engram paper as a "lightweight multiplicative-XOR
hash," is to map compressed N-gram tuples to rows of learned embedding tables
using only integer arithmetic. No neural network forward pass is required to
compute the address -- the index depends exclusively on the input token IDs and
fixed multiplier constants.

Because the hash is deterministic and depends only on token IDs (not on hidden
states or any floating-point intermediate), the resulting indices can be computed
well before the embeddings are needed. This property is what makes CPU offload
and asynchronous prefetch possible: the system knows which rows to fetch before
the layer that consumes them has begun executing.

**Multi-head design.** A single hash function per N-gram order would concentrate
all collisions into one table. Multiple independent hash heads per N-gram order
spread collisions across tables with different multiplier families, so two
N-grams that collide under head 0 are unlikely to also collide under head 1.
Retrieved embeddings from all heads are aggregated (summed, averaged, or
concatenated then projected) to form the final memory vector for each position.

**Scope of this document.** This reference covers the multiplicative-XOR hash
function family, multi-head layout across N-gram orders, prime table sizing
policy, per-layer salt/seed for collision decorrelation, N-gram ID computation
from canonical token sequences, streaming hash stability for autoregressive
generation, collision analysis and diagnostics, determinism guarantees, and
distributed sharding compatibility.

Related references: tokenizer compression (`tokenizer-compression.md`),
CPU offload and prefetch (`offload-prefetch.md`), context-aware gating
(`gating-fusion.md`), and integration modes (`integration-modes.md`).


---


## 2. Hash Function Family

### 2.1 Multiplicative-XOR Hash Definition

For an N-gram of order `n` consisting of token IDs `(t_1, t_2, ..., t_n)`, a
single hash head computes:

```
h = ((t_1 * p_1) ^ (t_2 * p_2) ^ ... ^ (t_n * p_n)) % table_size
```

where:

- `t_i` is the i-th canonical token ID in the N-gram (int64)
- `p_i` is the i-th prime multiplier for this head (int64, fixed at init)
- `^` is bitwise XOR
- `%` is integer modulus
- `table_size` is the number of rows in the embedding table (prime)

The combination of multiplication (which spreads bits upward) and XOR (which
mixes bits without carry propagation) produces good hash distribution with
minimal computation.

### 2.2 Why Multiplicative-XOR

| Property | Requirement | How Mult-XOR Satisfies It |
|---|---|---|
| Determinism | Same inputs always produce same output | Pure integer arithmetic, no randomness |
| Speed | Must not bottleneck forward pass | O(n) multiplies + XOR per N-gram |
| Distribution | Uniform spread across table rows | Large prime multipliers + XOR folding |
| No float dependency | Identical on CPU and CUDA | All operations are int64 |
| Differentiability | Not required | Hash indices feed `nn.Embedding`, which handles gradients |

Alternatives not used: cryptographic hashes (too slow), polynomial rolling
hashes (sequential, hard to vectorize), random projections (float-dependent),
learned hashes (require a forward pass), tabulation hashing (memory-heavy).

### 2.3 Algorithmic Implementation

```python
def multiplicative_xor_hash(
    ngram_ids: torch.Tensor,     # (B, T, n) int64 canonical token IDs
    multipliers: torch.Tensor,   # (n,) int64 prime multipliers for this head
    table_size: int,             # prime modulus
) -> torch.Tensor:               # (B, T) int64 hash indices
    """
    Compute multiplicative-XOR hash for a batch of N-grams.
    All arithmetic is int64 to guarantee cross-device determinism.
    """
    assert ngram_ids.dtype == torch.int64
    assert multipliers.dtype == torch.int64

    weighted = ngram_ids * multipliers  # (B, T, n) broadcast

    # XOR-fold across the N-gram dimension
    h = weighted[..., 0]
    for i in range(1, weighted.shape[-1]):
        h = h ^ weighted[..., i]

    h = h % table_size
    return h  # (B, T) int64
```

For N-gram orders 2-4, the XOR loop has 1-3 iterations and is negligible
compared to the embedding gather that follows.

### 2.4 Integer Overflow Considerations

The product `t_i * p_i` must not overflow signed int64 (`max = 2^63 - 1`).
With token IDs up to 200,000 and multipliers up to 10^12, the product is at
most 2 * 10^17, well within range. PyTorch int64 tensors follow two's-complement
semantics on overflow.

**Policy:** Multipliers are drawn from `[2^20, 2^50]` to keep products safely
below 2^63 while providing good bit dispersion.

```python
MIN_MULTIPLIER = 1 << 20    # 1,048,576
MAX_MULTIPLIER = 1 << 50    # 1,125,899,906,842,624
```

### 2.5 XOR vs. Addition Variants

The existing implementation in `brain_ai/memory/hash_embedding.py` uses additive
combination followed by XOR with a per-head seed:

```python
weighted = (ngrams_expanded * coeffs_expanded).sum(dim=-1)  # additive
hashed = (weighted ^ seeds_expanded) % self.table_size       # XOR with seed
```

| Variant | Formula | Trade-off |
|---|---|---|
| Pure XOR-fold | `(t1*p1) ^ (t2*p2) ^ ... % M` | Best bit mixing; order-independent |
| Add-then-XOR-seed | `(sum(ti*pi) ^ seed) % M` | Seed decorrelates heads |
| Pure additive | `sum(ti*pi) % M` | Simplest; weaker high-bit mixing |

All three are valid and configurable via `HashConfig.hash_fn`.


---


## 3. Multi-Head Design

### 3.1 Motivation

For V = 40,000 compressed vocab and M = 131,071 table size, there are ~1.6B
possible bigrams mapping to ~131K buckets -- about 12,200 collisions per bucket.
Multiple independent hash heads ensure that two N-grams colliding under one head
are unlikely to collide under all heads, so the aggregated embedding retains
correct information from non-colliding heads.

### 3.2 Head Layout

```
H_total = sum(num_heads_per_order[n] for n in ngram_orders)
```

**Default configuration:**

| N-gram Order | Heads | Table Size | Dim per Head |
|---|---|---|---|
| 2 (bigram) | 2 | 131,071 | 64 |
| 3 (trigram) | 2 | 131,071 | 64 |
| **Total** | **4** | -- | **256** (concat) |

### 3.3 Head Metadata

```python
@dataclass
class HeadSpec:
    """Specification for a single hash head."""
    order: int                    # N-gram order (e.g., 2 for bigram)
    head_index: int               # Index within this order (0, 1, ...)
    global_index: int             # Index across all heads (0 .. H_total-1)
    table_size: int               # Number of embedding rows (prime)
    multipliers: List[int]        # Per-position multipliers, length = order
    seed: int                     # XOR seed for this head
    embedding_dim: int            # Output dimension for this head
    layer_salt: Optional[int]     # Per-layer decorrelation salt
```

### 3.4 Multiplier Generation

Multipliers are generated deterministically from a seed using `numpy.random.RandomState`.
Per-order seeds are derived as `base_seed + order` to ensure independence:

```python
def generate_multipliers(order, num_heads, table_size, seed):
    rng = np.random.RandomState(seed)
    coefficients = torch.tensor(
        rng.randint(1, table_size, size=(num_heads, order)), dtype=torch.int64,
    )
    seeds = torch.tensor(
        rng.randint(0, table_size, size=(num_heads,)), dtype=torch.int64,
    )
    return coefficients, seeds
```

### 3.5 Aggregation Strategies

| Strategy | Output Dim | Description |
|---|---|---|
| **Concatenation** (default) | `H_total * dim_per_head` | Each head contributes a disjoint slice |
| **Sum** | `embedding_dim` | All heads share the same dim; summed |
| **Mean** | `embedding_dim` | Sum normalized by H_total |
| **Concat-then-project** | `target_dim` | Concat followed by learned linear projection |

### 3.6 Per-Head Embedding Tables

Each head has an independent embedding table keyed as `"ngram{n}_head{k}"`:

```
Shape: (table_size, dim_per_head)
Dtype: float32 (or float16 for CPU offload)
Init:  N(0, 0.02)
```

Total embedding parameters: `H_total * table_size * dim_per_head`. At production
scale (4 heads, 10M table, dim=64): ~2.56B parameters -- the dominant cost of
the Engram subsystem, making CPU offload essential.


---


## 4. Prime Sizing Policy

### 4.1 Why Prime Table Sizes

With prime modulus M, `(t * p) % M` distributes uniformly regardless of p's
factorization. Non-prime sizes (especially powers of 2) cause systematic bias:
`(t * p) % 2^17` depends only on low 17 bits, and even multipliers waste half
the table.

### 4.2 Two Modes

**Mode A: Fixed primes.** User provides a prime in `HashConfig.table_size`.
Primality is verified at init; non-prime raises `ValueError`.

**Mode B: Auto-generated.** With `auto_prime=True`, the system finds the
smallest prime >= `table_size` and freezes it into the checkpoint.

### 4.3 Prime Search

```python
def is_prime_trial_division(n: int) -> bool:
    """O(sqrt(n)) primality test. Suitable for n < 10^7."""
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

def is_prime_miller_rabin(n: int, k: int = 20) -> bool:
    """Probabilistic test for large n. False positive < 4^{-20}."""
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0: return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1; d //= 2
    import random
    rng = random.Random(42)  # deterministic witnesses
    for _ in range(k):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else: return False
    return True

def next_prime(n: int) -> int:
    test = is_prime_trial_division if n < 10_000_000 else is_prime_miller_rabin
    if n <= 2: return 2
    candidate = n if n % 2 != 0 else n + 1
    while not test(candidate):
        candidate += 2
    return candidate
```

### 4.4 Common Prime Table Sizes

| Scale | Target | Prime | Memory/Head (f32, dim=64) |
|---|---|---|---|
| Unit test | 1,000 | 1,009 | 251 KB |
| Dev | 10,000 | 10,007 | 2.4 MB |
| Default | 131,000 | 131,071 | 32 MB |
| Medium | 1,000,000 | 1,000,003 | 244 MB |
| Production | 10,000,000 | 10,000,019 | 2.4 GB |
| Large prod | 50,000,000 | 50,000,017 | 12.2 GB |

### 4.5 Per-Order Variable Sizes

Higher-order N-grams have exponentially more combinations. An optional
`per_order_table_sizes: Dict[int, int]` overrides the base `table_size`
per order (e.g., `{2: 131071, 3: 262139, 4: 524287}`).


---


## 5. Per-Layer Salt/Seed

### 5.1 Motivation

When Engram layers are inserted at multiple backbone positions (e.g., layers
4, 8, 12, 16), identical N-grams produce identical hash IDs at every layer
without differentiation. Per-layer salting:

1. Allows each layer to retrieve different embeddings for the same N-gram.
2. Decorrelates collisions: N-grams that collide at layer 4 are unlikely to
   also collide at layer 12.

### 5.2 Salt Derivation

```python
def compute_layer_salt(base_seed: int, layer_index: int) -> int:
    LAYER_PRIME = 6_364_136_223_846_793_005  # Knuth's LCG multiplier
    salt = (base_seed * LAYER_PRIME + layer_index * 2_654_435_761) & 0x7FFFFFFFFFFFFFFF
    return salt
```

### 5.3 Applying the Salt

The salt XOR-s with the intermediate hash before the final modulus:

```python
def hash_with_layer_salt(ngram_ids, multipliers, table_size, layer_salt):
    weighted = ngram_ids * multipliers
    h = weighted[..., 0]
    for i in range(1, weighted.shape[-1]):
        h = h ^ weighted[..., i]
    h = h ^ layer_salt   # decorrelation
    h = h % table_size
    return h
```

### 5.4 Shared Tables vs. Separate Tables

| Strategy | Memory | Decorrelation |
|---|---|---|
| **Shared tables, different salt** (default) | 1x tables | Different rows retrieved per layer |
| **Separate tables per layer** | Lx tables | Fully independent |

Shared tables with per-layer salt is the default because it decorrelates
collisions without multiplying the embedding parameter count.

### 5.5 Salt Registry

Salts are stored in a `SaltRegistry` that serializes into checkpoints. Salts
are saved explicitly (not recomputed on load) to guard against changes to the
derivation function between code versions.


---


## 6. N-gram ID Computation

### 6.1 Input and Output Shapes

| Tensor | Shape | Dtype | Description |
|---|---|---|---|
| `input_ids` | `(B, T)` | int64 | Raw tokenizer output |
| `attention_mask` | `(B, T)` | bool/int | 1 = real token, 0 = padding |
| `canonical_ids` | `(B, T)` | int64 | After tokenizer compression |
| `hash_ids` | `(B, T, H_total)` | int64 | Multi-head hash indices |

### 6.2 Suffix N-gram Extraction

For position `t` and order `n`: `ngram(t, n) = (ids[t-n+1], ..., ids[t])`.
Left-padding with `pad_id=0` handles positions where `t < n-1`.

```python
def extract_suffix_ngrams(canonical_ids, order, pad_id=0):
    B, T = canonical_ids.shape
    padded = F.pad(canonical_ids, (order - 1, 0), value=pad_id)  # (B, T+n-1)
    ngrams = torch.stack(
        [padded[:, i : i + T] for i in range(order)], dim=-1,
    )  # (B, T, n)
    return ngrams
```

### 6.3 Full Pipeline

```python
def compute_hash_ids(canonical_ids, attention_mask, head_specs, pad_id=0):
    B, T = canonical_ids.shape
    H_total = len(head_specs)
    hash_ids = torch.zeros(B, T, H_total, dtype=torch.int64, device=canonical_ids.device)
    masked_ids = canonical_ids * attention_mask.long()

    for head in head_specs:
        ngrams = extract_suffix_ngrams(masked_ids, head.order, pad_id)
        multipliers = torch.tensor(head.multipliers, dtype=torch.int64, device=masked_ids.device)

        if head.layer_salt is not None:
            h = hash_with_layer_salt(ngrams, multipliers, head.table_size, head.layer_salt)
        else:
            h = multiplicative_xor_hash(ngrams, multipliers, head.table_size)

        h = h * attention_mask.long()  # pad -> empty_hash_id (0)
        hash_ids[:, :, head.global_index] = h

    return hash_ids
```

### 6.4 Padding Treatment

1. Multiply `canonical_ids` by `attention_mask` before extraction (pad -> 0).
2. Multiply hash output by `attention_mask` after hashing (pad -> 0).
3. Embedding table row 0 is zero-initialized so padding positions contribute
   a zero vector.

### 6.5 Shape Walkthrough

For `B=2, T=6, ngram_orders=(2,3), num_heads_per_order=2`:

```
canonical_ids:   (2, 6)     int64
bigram ngrams:   (2, 6, 2)  int64 -> hash heads 0,1: (2, 6) each
trigram ngrams:  (2, 6, 3)  int64 -> hash heads 2,3: (2, 6) each
hash_ids:        (2, 6, 4)  int64
```


---


## 7. Streaming Hash Stability

### 7.1 Invariant

During autoregressive generation, incremental hash computation for position `t`
must produce bit-identical hash IDs to a full batch recompute over `[0..t]`.

### 7.2 Rolling Buffer

```python
@dataclass
class StreamingHashState:
    buffer: torch.Tensor    # (B, K-1) int64 -- last K-1 canonical IDs
    position: int           # current sequence position
    max_order: int          # K = max N-gram order

    @classmethod
    def create(cls, batch_size, max_order, device):
        return cls(
            buffer=torch.zeros(batch_size, max_order - 1, dtype=torch.int64, device=device),
            position=0, max_order=max_order,
        )

    def update(self, new_canonical_id):
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[:, -1] = new_canonical_id
        self.position += 1

    def get_all_ngrams(self, new_canonical_id):
        ngrams = {}
        for order in range(2, self.max_order + 1):
            prefix_start = self.max_order - 1 - (order - 1)
            prefix = self.buffer[:, prefix_start:]
            full = torch.cat([prefix, new_canonical_id.unsqueeze(1)], dim=1)
            ngrams[order] = full
        return ngrams
```

### 7.3 Incremental Computation

At each step: call `get_all_ngrams(new_id)` to build N-grams, hash through each
head, then call `update(new_id)` to shift the buffer.

### 7.4 Consistency Verification

```python
def verify_streaming_consistency(canonical_ids, head_specs, max_order):
    B, T = canonical_ids.shape
    mask = torch.ones(B, T, dtype=torch.bool, device=canonical_ids.device)
    batch_ids = compute_hash_ids(canonical_ids, mask, head_specs)

    state = StreamingHashState.create(B, max_order, canonical_ids.device)
    stream_ids = []
    for t in range(T):
        ngrams = state.get_all_ngrams(canonical_ids[:, t])
        h = torch.zeros(B, len(head_specs), dtype=torch.int64, device=canonical_ids.device)
        for head in head_specs:
            ng = ngrams[head.order].unsqueeze(1)
            mults = torch.tensor(head.multipliers, dtype=torch.int64, device=canonical_ids.device)
            if head.layer_salt is not None:
                val = hash_with_layer_salt(ng, mults, head.table_size, head.layer_salt)
            else:
                val = multiplicative_xor_hash(ng, mults, head.table_size)
            h[:, head.global_index] = val.squeeze(1)
        stream_ids.append(h)
        state.update(canonical_ids[:, t])

    stream_ids = torch.stack(stream_ids, dim=1)
    return torch.all(batch_ids == stream_ids).item()
```

### 7.5 State Serialization

The streaming state (`buffer`, `position`, `max_order`) is serializable for
saving/restoring generation sessions.


---


## 8. Collision Analysis

### 8.1 Unique ID Ratio

The primary diagnostic metric per head:

```
unique_ratio = count_unique(hash_ids[:, :, head]) / count_nonpad(hash_ids[:, :, head])
```

```python
def compute_unique_ratios(hash_ids, attention_mask):
    B, T, H = hash_ids.shape
    ratios = {}
    mask = attention_mask.flatten().bool()
    for h in range(H):
        valid = hash_ids[:, :, h].flatten()[mask]
        total = valid.numel()
        ratios[h] = valid.unique().numel() / total if total > 0 else 1.0
    return ratios
```

### 8.2 Expected Unique Ratio

For uniform hash to M buckets with N items:

```
E[unique] = M * (1 - (1 - 1/M)^N)  ~  N * (1 - N/(2M))  for N << M
```

| Table Size (M) | Items (N = B*T) | Expected Unique Ratio |
|---|---|---|
| 131,071 | 1,024 | 0.9961 |
| 131,071 | 8,192 | 0.9688 |
| 131,071 | 65,536 | 0.7788 |
| 10,000,019 | 1,024 | 0.99995 |
| 10,000,019 | 8,192 | 0.99959 |
| 10,000,019 | 65,536 | 0.99672 |

At production table sizes (10M+), collisions are rare even for long sequences.

### 8.3 Sanity Check

Minimum acceptable unique ratio: **0.90**. Diagnostic logging warns when any
head falls below this threshold and recommends increasing table size.

### 8.4 Per-Head Telemetry

Collision statistics are reported in the `telemetry` dict:

```python
telemetry['hash_collision'] = {
    'unique_ratios': {h: float for h in range(H_total)},
    'mean_unique_ratio': float,
    'min_unique_ratio': float,
    'worst_head': int,
}
```

### 8.5 Cross-Head Independence

The core assumption of multi-head hashing: collisions are independent across
heads. Verified by checking that positions colliding on ALL heads simultaneously
occurs at rate approximately `(1/M)^H_total`, which is negligible for
reasonable table sizes.

### 8.6 Mitigation Strategies

| Strategy | Trade-off |
|---|---|
| Increase table size | Linear memory increase |
| Add more heads | Linear param + compute increase |
| Per-layer salt | No extra memory |
| Larger prime multipliers | Negligible cost |
| Collision-aware training loss | Requires collision tracking |


---


## 9. Determinism Guarantees

### 9.1 Core Invariant

> Given identical `(canonical_ids, seed, config)`, the output `hash_ids` is
> bit-for-bit identical regardless of device (CPU/CUDA), CUDA device index,
> distributed rank, PyTorch random state, input dtype (int32 cast to int64),
> or AMP autocast context.

### 9.2 How Determinism Is Achieved

**No floating-point intermediates.** All hash arithmetic is int64.

```python
# CORRECT
weighted = ngram_ids.long() * multipliers.long()  # int64
h = weighted[..., 0] ^ weighted[..., 1]           # int64

# WRONG
weighted = ngram_ids.float() * multipliers.float()  # float32 -- not exact
```

**No torch random state dependency.** Multipliers are generated from
`numpy.random.RandomState(seed)` at init and stored as fixed buffers.

**Cross-dtype safety.** Input is cast to int64 as the first pipeline operation.

**AMP safety.** Int64 operations are not affected by `autocast`. The embedding
lookup that follows IS affected, but that is downstream of the hash.

**Cross-PyTorch-version stability.** Only basic tensor ops (`*`, `^`, `%`) are
used. NumPy `RandomState` (Mersenne Twister) is stable across versions.

### 9.3 Verification Tests

**Cross-device:** Compute hash_ids on CPU and CUDA; assert bit-for-bit match.

**Cross-rank:** In distributed training, all ranks compute hash_ids for the same
input and verify via `all_gather` that all outputs are identical.

**Cross-dtype:** Hash int32 input and int64 input; assert identical output.


---


## 10. Distributed Sharding Compatibility

### 10.1 Motivation

At production scale, 4 heads with 50M-row tables at dim=64 float32 require
51.2 GB -- exceeding single-GPU memory. Sharding across devices is necessary.

### 10.2 Range-Based Sharding

```python
def compute_shard_id(hash_id, table_size, num_shards):
    shard_size = (table_size + num_shards - 1) // num_shards
    return hash_id // shard_size

def global_to_local_index(hash_id, shard_id, table_size, num_shards):
    shard_size = (table_size + num_shards - 1) // num_shards
    return hash_id - shard_id * shard_size
```

Each shard holds `nn.Embedding(shard_size, dim)` instead of the full table.

### 10.3 All-to-All Communication Pattern

```
1. All ranks compute hash_ids locally (deterministic, no communication)
2. Each rank routes hash_ids to owning shards
3. All-to-all: send non-local hash_ids to owning ranks
4. Each rank gathers embeddings from its local table
5. All-to-all: return embeddings to requesting ranks
6. Each rank reassembles the full embedding tensor
```

### 10.4 API Abstraction

The `EmbeddingRetriever` interface abstracts over local, offloaded, and sharded
retrieval. `EngramModule` calls `retriever.retrieve(hash_ids, head_index)`
without knowing the backend.

### 10.5 Hash ID Properties for Sharding

| Property | Benefit |
|---|---|
| Int64 type | Direct arithmetic for shard routing |
| Uniform distribution | Even load balance |
| Deterministic | All ranks agree on routing without communication |
| No gradient through hash | Shard routing is non-differentiable |

### 10.6 Shard Balance

With uniform hash distribution over prime M and S shards, expected lookups per
shard is `N/S` with variance `N*(S-1)/S^2`. Balance analysis:

```python
def analyze_shard_balance(hash_ids, table_size, num_shards):
    flat = hash_ids.flatten()
    shard_size = (table_size + num_shards - 1) // num_shards
    shard_ids = flat // shard_size
    counts = torch.bincount(shard_ids.long(), minlength=num_shards)
    return {
        'imbalance_ratio': counts.max().item() / max(counts.float().mean().item(), 1),
    }
```


---


## 11. Configuration Reference

### 11.1 HashConfig Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `hash_fn` | `str` | `"mult_xor"` | Hash family: `"mult_xor"`, `"add_xor_seed"`, `"additive"` |
| `table_size` | `int` | `131071` | Base table size (prime recommended) |
| `use_prime_sizes` | `bool` | `True` | Verify/auto-generate prime sizes |
| `auto_prime` | `bool` | `False` | Auto-find next prime >= table_size |
| `per_layer_salt` | `bool` | `True` | Per-layer hash decorrelation |
| `seed` | `int` | `42` | Base seed for multiplier generation |
| `per_order_table_sizes` | `Optional[Dict]` | `None` | Override per N-gram order |

### 11.2 EngramConfig Fields (Hash-Related)

| Field | Type | Default | Description |
|---|---|---|---|
| `ngram_orders` | `Tuple[int, ...]` | `(2, 3)` | N-gram orders |
| `num_heads_per_order` | `int` | `2` | Heads per N-gram order |
| `embedding_dim` | `int` | `256` | Total embedding dim |
| `max_ngram_order` | `int` | `4` | Max K for streaming buffer |

### 11.3 Scale Presets

```python
HashConfig(table_size=1009, seed=42)       # Minimal: ~258K params
HashConfig(table_size=10007, seed=42)      # Dev: ~2.6M params
HashConfig(table_size=10000019, seed=42)   # Production: ~2.56B params
```


---


## 12. End-to-End Algorithm

```
ALGORITHM: Engram Multi-Head Hash Pipeline

INPUT:  input_ids (B,T), attention_mask (B,T), config, layer_index
OUTPUT: hash_ids (B, T, H_total) int64

1. COMPRESS:     canonical_ids = compress(input_ids)          # (B, T) int64
2. MASK:         canonical_ids *= attention_mask.long()       # pad -> 0
3. CAST:         canonical_ids = canonical_ids.to(int64)

4. FOR EACH order n:
   4a. EXTRACT:  ngrams = extract_suffix_ngrams(canonical_ids, n)  # (B, T, n)
   4b. FOR EACH head k:
       i.   multipliers = head_specs[(n,k)].multipliers
       ii.  salt = compute_layer_salt(seed, layer_index) if per_layer_salt
       iii. weighted = ngrams * multipliers; h = xor_fold(weighted)
            if salt: h ^= salt; h %= table_size
       iv.  h *= attention_mask.long()
       v.   hash_ids[:, :, global_idx] = h

5. RETURN hash_ids
```


---


## 13. Testing Checklist

### Unit Tests

| Test | Pass Condition |
|---|---|
| `test_hash_determinism` (10 runs) | All hash_ids identical |
| `test_cross_device` (CPU vs CUDA) | Bit-for-bit match |
| `test_dtype_cast` (int32 vs int64 input) | Identical hash_ids |
| `test_padding_zeros` | Pad positions = empty_hash_id (0) |
| `test_prime_verification` (non-prime) | Raises ValueError |
| `test_auto_prime` | Finds correct next prime |
| `test_streaming_consistency` | Batch == streaming at all positions |
| `test_per_layer_salt` (diff layers) | Different hash_ids |
| `test_same_layer_same_salt` | Identical hash_ids |
| `test_unique_ratio` | > 0.90 for default config |
| `test_head_independence` | Different hash_ids per head |
| `test_output_shape` | Correct (B, T, H_total) |
| `test_amp_invariance` | Identical with/without autocast |

### Integration Tests

| Test | Pass Condition |
|---|---|
| `test_hash_to_embedding` | Output shape (B, T, embedding_dim) |
| `test_checkpoint_roundtrip` | Hash_ids identical after load |
| `test_gradient_flow` | Embedding gradients non-zero |

### Stress Tests

| Test | Pass Condition |
|---|---|
| `test_long_sequence` (T=16384) | No OOM, correct shapes |
| `test_large_vocab` (IDs to 200K) | No int64 overflow |
| `test_many_heads` (32 heads) | Correct H_total=32 |


---


## 14. Anti-Patterns

**Float intermediate in hash:**
```python
# WRONG: float32 loses precision for int64 values above 2^23
h = (ngram_ids.float() * multipliers.float()).long()
```

**torch.rand for multipliers:**
```python
# WRONG: depends on torch random state, not reproducible across ranks
multipliers = torch.randint(1, table_size, (num_heads, order))
```
Use `numpy.random.RandomState(seed)`.

**Shared seed across orders:**
```python
# WRONG: same PRNG stream for bigrams and trigrams
for order in [2, 3, 4]:
    mults = generate_multipliers(order, num_heads, table_size, seed=42)
```
Derive per-order seeds: `seed + order`.

**Non-prime table size:**
```python
# WRONG: 131072 = 2^17, systematic hash bias
config = HashConfig(table_size=131072, use_prime_sizes=False)
```

**Ignoring padding:** Apply `attention_mask` before N-gram extraction.

**Mutable multipliers:** Never modify after init; invalidates all learned
embeddings.

**Forgetting salt in checkpoint:** Save computed salt values explicitly; do not
rely on recomputation.

**XOR-reduce API:** `torch.bitwise_xor` has no `.reduce()`. Use a loop or
`functools.reduce`.


---


## 15. Glossary

| Term | Definition |
|---|---|
| **Canonical ID** | Token ID after compression; surjective mapping from raw vocab |
| **Collision** | Two distinct N-grams mapping to the same table row |
| **Empty hash ID** | Hash ID 0 for padding; row 0 is zero-initialized |
| **Global index** | Head index across all orders (0 to H_total-1) |
| **H_total** | Total hash heads across all N-gram orders |
| **Head** | A (multiplier set, table) pair mapping N-grams to embeddings |
| **Layer salt** | Per-layer XOR value for collision decorrelation |
| **Multiplier** | Large integer multiplied with each token ID in the N-gram |
| **N-gram order** | Count of consecutive tokens in a suffix N-gram |
| **Prime sizing** | Using primes for table sizes to improve distribution |
| **Rolling buffer** | Fixed-size buffer of recent IDs for streaming hash |
| **Suffix N-gram** | Tuple of last n canonical IDs ending at position t |
| **Table size** | Number of rows in an embedding table (should be prime) |
| **Unique ID ratio** | Fraction of distinct hash IDs among non-pad positions |
| **XOR-fold** | Reducing int64 values to one value via bitwise XOR |


---


## 16. References

- **DeepSeek Engram paper** -- multiplicative-XOR hash, multi-head design, and
  context-aware gating.
- **Knuth, "The Art of Computer Programming", Vol. 3** -- multiplicative hashing
  and prime modulus analysis.
- **Carter and Wegman (1979), "Universal Classes of Hash Functions"** --
  theoretical foundation for independent hash families.
- **`brain_ai/memory/hash_embedding.py`** -- `MultiHeadHash` and
  `OffloadableEmbedding` implementation.
- **`brain_ai/memory/engram.py`** -- `EngramEmbedding`, `ContextAwareGating`,
  and `EngramModule` implementation.
- **`brain_ai/config.py`** -- `EngramConfig` and related dataclasses.
