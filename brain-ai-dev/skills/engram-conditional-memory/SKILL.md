---
name: Engram Conditional Memory (N-gram Hash Lookup + Offload/Prefetch)
description: >-
  This skill should be used when the user asks to "implement engram memory",
  "add N-gram hash lookup", "implement tokenizer compression", "add engram layer",
  "implement CPU offload embeddings", "add async prefetch", "implement multi-head hashing",
  "add context-aware gating", "implement depthwise causal conv", "add engram encoder",
  "implement hash embedding retrieval", "add collision mitigation", "implement offloadable embedding",
  "add prefetch scheduler", "implement engram augmented layer", "add residual fusion",
  "implement RMSNorm gating", "add engram telemetry", "implement streaming N-gram cache",
  "add prime-sized hash tables", "implement tokenizer equivalence merging",
  or mentions engram memory, N-gram hashing, deterministic addressing, CPU offload + prefetch,
  tokenizer compression, context-aware gating, depthwise causal convolution, hash embedding,
  encoder-competition mode, layer-augmentation mode, or DeepSeek-style conditional memory
  in the cognitive pipeline.
version: 0.1.0
---

# Engram Conditional Memory (N-gram Hash Lookup + Offload/Prefetch)

## Purpose

This skill standardizes the Engram conditional memory subsystem: tokenizer compression to
canonical IDs, suffix N-gram extraction, deterministic multi-head hashing, embedding retrieval
with collision mitigation, context-aware gating (RMSNorm on q/k), optional depthwise causal
convolution, and residual fusion into the host network. Two integration modes are supported:
Phase 1 encoder-competition (EngramTextEncoder competes in the Global Workspace) and Phase 2
layer-augmentation (EngramAugmentedLayer injected at selected backbone layers). The non-negotiable
goals are deterministic addressing (enabling CPU offload + async prefetch) and AMP-safe gating.

## Key Files

| Target Module | Template Asset | Purpose |
|---|---|---|
| `brain_ai/memory/tokenizer_compression.py` | `assets/tokenizer_compression_template.py` | TokenizerCompression: canonical ID mapping, special token policy, save/load |
| `brain_ai/memory/hash_embedding.py` | `assets/hash_embedding_template.py` | MultiHeadHash, OffloadableEmbedding, PrefetchPlan, prime sizing |
| `brain_ai/memory/engram.py` | `assets/engram_template.py` | N-gram extraction, ContextAwareGating, DepthwiseCausalConv, EngramModule |
| `brain_ai/layers/engram_layer.py` | `assets/engram_layer_template.py` | EngramAugmentedLayer, EngramTextEncoder, integration modes |
| `brain_ai/config.py` (extend) | `assets/engram_config_template.py` | EngramConfig, HashConfig, OffloadConfig, GatingConfig |

## Public Contract

```python
# TokenizerCompression
build_from_tokenizer(tokenizer, *, normalization, special_policy) -> None
compress_ids(input_ids) -> canonical_ids  # same shape, table lookup
serialize(path) / load(path)

# EngramModule
forward(hidden_states, input_ids, attention_mask, *, cache_state=None) -> delta, telemetry

# EngramAugmentedLayer (Phase 2)
forward(hidden_states, input_ids, attention_mask, **kwargs) -> augmented_hidden
```

`input_ids` is `(B, T)` integer token IDs. `hidden_states` is `(B, T, D)` from the host
backbone. `canonical_ids` has the same shape as `input_ids` with compressed vocabulary.

## Core Output Contract

| Field | Shape / Type | Description |
|---|---|---|
| `canonical_ids` | `(B, T)` int | Compressed token IDs via surjective mapping |
| `hash_ids` | `(B, T, H_total)` int | Multi-head hash indices across all N-gram orders |
| `retrieved` | `(B, T, D_emb)` | Aggregated embeddings from hash tables |
| `gate` | `(B, T, 1)` or `(B, T, H)` | Context-aware gating scalars in [0, 1] |
| `delta` | `(B, T, D)` | Fused output for residual addition |
| `telemetry` | `Optional[Dict]` | Gate stats, collision proxies, prefetch stats |

**Hard invariants**:
- Hash IDs are deterministic across CPU/CUDA/distributed ranks given same seed/config.
- Gating outputs are bounded [0, 1] and AMP-safe (no NaN under mixed precision).
- CPU offload mode runs without deadlocks; prefetch overlaps non-trivially with compute.
- Tokenizer compression is deterministic and versioned: same inputs produce same lookup table.

## Tokenizer Compression

Surjective mapping collapsing textually equivalent tokens into canonical IDs. Normalization
recipe (NFKC + lowercasing) reduces effective vocabulary (~23% in 128k tokenizer case study).

Key constraints: special tokens (pad/bos/eos/unk) are invariant; fast path is int table lookup;
artifact is persistable with metadata hash for version tracking.

See `references/tokenizer-compression.md` for normalization recipes, equivalence class
construction, special token policies, and serialization format.

## Multi-Head Hashing

Deterministic multiplicative-XOR hash maps compressed N-grams to embedding rows. Multiple
heads per N-gram order reduce collisions. Prime-sized tables improve distribution.

Per-layer salt/seed enables identical N-grams to map differently at different insertion layers,
decorrelating collisions. Head sizes are either fixed primes from config or generated once
and frozen into the checkpoint.

See `references/multi-head-hashing.md` for hash functions, prime sizing policy, per-layer
multipliers, collision analysis, and streaming hash stability.

## Offload and Prefetch

Because indices depend only on input tokens (not hidden states), embeddings can be prefetched
from host memory over PCIe and overlapped with earlier layer compute.

Two runtime modes: (1) on-device baseline (standard embedding gather), (2) CPU offload with
async prefetch (coalesced unique IDs, pinned buffers, dedicated CUDA stream, event sync).

See `references/offload-prefetch.md` for the retrieval plan API, coalescing strategy, CUDA
stream/event synchronization, benchmarking methodology, and distributed sharding compatibility.

## Context-Aware Gating and Fusion

Retrieved embeddings are static priors; gating uses hidden state as query and memory as
key/value with RMSNorm on q/k for stability. Scalar gate suppresses contradictory retrievals.

Depthwise causal convolution (kernel 4, dilation tied to max N-gram order) expands receptive
field before residual fusion.

See `references/gating-fusion.md` for projection design, RMSNorm stability, gate statistics,
depthwise conv specification, and AMP safety patterns.

## Integration Modes

| Mode | Module | Mechanism |
|---|---|---|
| Phase 1: Encoder-competition | `EngramTextEncoder` | Produces workspace-aligned `(B, T, D_ws)` representation; competes in Global Workspace |
| Phase 2: Layer-augmentation | `EngramAugmentedLayer` | Inserted at configurable backbone layers; computes delta residually before attention/FFN |

Phase 2 supports prefetch hooks: "prefetch for layer L" triggered N layers earlier.

See `references/integration-modes.md` for encoder pipeline, layer insertion patterns, prefetch
hook scheduling, telemetry reporting, and checkpoint format.

## Configuration Surface

### EngramConfig

| Field | Default | Purpose |
|---|---|---|
| `max_ngram_order` | 4 | Maximum N-gram suffix length K |
| `embedding_dim` | 256 | Per-head embedding dimension |
| `num_heads_per_order` | 2 | Hash heads per N-gram order |
| `use_tokenizer_compression` | True | Enable/disable compression |
| `use_context_gate` | True | Enable/disable gating |
| `use_depthwise_conv` | True | Enable/disable causal conv |
| `conv_kernel_size` | 4 | Depthwise conv kernel |

### HashConfig

| Field | Default | Purpose |
|---|---|---|
| `hash_fn` | `"mult_xor"` | Hash function family |
| `use_prime_sizes` | True | Prime-modulus table sizes |
| `table_size` | 131071 | Base table size (prime) |
| `per_layer_salt` | True | Different salts per insertion layer |
| `seed` | 42 | Deterministic hash seed |

### OffloadConfig

| Field | Default | Purpose |
|---|---|---|
| `weights_on_cpu` | False | CPU offload mode |
| `use_async_prefetch` | False | Enable async prefetch |
| `prefetch_ahead_layers` | 2 | Layers ahead to start prefetch |
| `pin_memory` | True | Use pinned host buffers |
| `storage_dtype` | `"float16"` | Host storage precision |

### GatingConfig

| Field | Default | Purpose |
|---|---|---|
| `gate_type` | `"scalar"` | `"scalar"`, `"per_head"` |
| `use_rmsnorm` | True | RMSNorm on q/k projections |
| `gate_init_bias` | -2.0 | Initial gate bias (conservative) |
| `conv_dilation` | 4 | Causal conv dilation |

Presets: `EngramFullConfig.minimal()`, `.dev()`, `.production()`.

## Done-When Gates

| Gate | Test | Threshold |
|---|---|---|
| **(a) Determinism** | Fixed input_ids + seed; hash IDs identical across 10 runs on CPU and CUDA; tokenizer compression produces same table across runs | Exact match |
| **(b) Offload correctness** | CPU offload output matches on-device output within tolerance; no deadlocks across 50 repeated forward passes with prefetch | Cosine > 0.9999 / no hangs |
| **(c) Integration modes** | Phase 1 encoder produces workspace-aligned shapes with correct masking; Phase 2 layer insertion matches baseline numerically when offload disabled | Shape correct / tolerance match |

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Hash IDs differ across devices | Non-deterministic integer ops | Force int64 hashing, avoid float intermediates |
| NaN in gating under AMP | RMSNorm precision loss | Force fp32 for norm computation |
| Deadlock in prefetch | Missing CUDA event sync | Ensure event.wait() before consuming prefetched data |
| Tokenizer compression non-deterministic | Dict iteration order or locale | Sort equivalence classes, fix locale in normalization |
| Gate always saturated at 1 | Bias too high | Use negative init bias (-2.0), check gradient flow |
| Embedding table OOM on GPU | Table too large for device | Enable CPU offload mode |
| N-grams leak padding | Mask not applied before extraction | Apply attention_mask to canonical_ids before hashing |
| Prefetch no overlap | Prefetch triggered too late | Increase prefetch_ahead_layers |

## Anti-Patterns

- **Hashing in float** -- hash computation must use integer types for determinism
- **Shared salt across layers** -- per-layer salts decorrelate collisions; use HashConfig
- **fp16 for RMSNorm** -- gating normalization needs fp32 under AMP
- **Blocking prefetch** -- the whole point is async overlap; never synchronize too early
- **Mutable compression table** -- freeze after build; never modify at runtime
- **Skipping coalescing** -- repeated IDs waste PCIe bandwidth; always unique + inverse
- **Non-causal conv** -- depthwise conv must have causal padding (no future leakage)
- **Hardcoded table sizes** -- use HashConfig with prime sizing, not magic numbers

## Additional Resources

### Reference Files

- **`references/tokenizer-compression.md`** -- Normalization recipes, equivalence classes, special token policy, serialization, version hashing
- **`references/multi-head-hashing.md`** -- Hash functions, prime sizing, per-layer multipliers, collision analysis, streaming stability
- **`references/offload-prefetch.md`** -- Retrieval plan API, coalescing, CUDA stream/event sync, benchmarking, distributed sharding
- **`references/gating-fusion.md`** -- Projection design, RMSNorm stability, gate statistics, depthwise conv, AMP safety
- **`references/integration-modes.md`** -- Encoder-competition, layer-augmentation, prefetch hooks, telemetry, checkpoint format

### Asset Templates

- **`assets/tokenizer_compression_template.py`** -- TokenizerCompression, normalization, equivalence classes, save/load, self-test
- **`assets/hash_embedding_template.py`** -- MultiHeadHash, OffloadableEmbedding, PrefetchPlan, prime sizing, self-test
- **`assets/engram_template.py`** -- N-gram extraction, ContextAwareGating, DepthwiseCausalConv, EngramModule, self-test
- **`assets/engram_layer_template.py`** -- EngramAugmentedLayer, EngramTextEncoder, integration modes, self-test
- **`assets/engram_config_template.py`** -- All configs, presets, serialization, self-test

### Scripts

- **`scripts/validate_engram.py`** -- Runtime contract validation (determinism, offload, integration modes)
- **`scripts/gen_engram_tests.py`** -- Generates `tests/test_engram.py` (~90+ test cases)
- **`scripts/offload_benchmark.py`** -- Benchmark gather baseline, offload, prefetch overlap, throughput
