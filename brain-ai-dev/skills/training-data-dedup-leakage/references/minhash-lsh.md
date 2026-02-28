# MinHash + LSH Reference

## Purpose

MinHash with Locality-Sensitive Hashing (LSH) enables efficient detection of
near-duplicate documents by approximating Jaccard similarity without computing
all-pairs comparisons. This is the workhorse of Stage 3 (fuzzy dedup).

---

## MinHash Theory

### Jaccard Similarity

For two sets A and B, the Jaccard similarity is:

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

For documents, we represent each document as a **set of shingles** (word or character
n-grams), then compute Jaccard over these shingle sets.

### MinHash Signature

MinHash approximates Jaccard similarity using **k independent hash functions**.

**Algorithm**:
1. Given a document, compute its shingle set S.
2. For each of k hash functions h_1, ..., h_k, compute:
   `sig_i = min({h_i(s) : s ∈ S})`
3. The MinHash signature is the vector `[sig_1, ..., sig_k]`.

**Key property**: For two sets A and B:
```
P(min_hash(A) == min_hash(B)) = J(A, B)
```

Therefore, the fraction of signature positions where two documents agree is an
unbiased estimator of their Jaccard similarity.

### Number of Permutations (num_perm)

The standard error of the Jaccard estimate with k permutations is:

```
SE = sqrt(J(1-J) / k)
```

For J=0.7 and k=256: SE = sqrt(0.7 * 0.3 / 256) = 0.0286 (about 2.9%)

| num_perm | SE at J=0.5 | SE at J=0.7 | Memory per doc |
|----------|-------------|-------------|----------------|
| 64       | 6.25%       | 5.73%       | 256 bytes      |
| 128      | 4.42%       | 4.05%       | 512 bytes      |
| 256      | 3.12%       | 2.86%       | 1 KB           |
| 512      | 2.21%       | 2.03%       | 2 KB           |

**Recommendation**: 256 permutations balances accuracy and memory for production use.

### Shingle Generation

**Word shingles** (recommended for natural language):
```python
def word_shingles(text: str, n: int = 5) -> set:
    words = text.split()
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}
```

**Character shingles** (better for short texts or code):
```python
def char_shingles(text: str, n: int = 13) -> set:
    return {text[i:i+n] for i in range(len(text) - n + 1)}
```

**BigCode production setting**: 5-gram word shingles (5 consecutive words).

---

## LSH Banding

### Concept

Instead of comparing all O(n^2) signature pairs, LSH uses **banding** to hash
sub-vectors of the signature. Only pairs that collide in at least one band are
considered candidates.

### Parameters

- **b** = number of bands
- **r** = number of rows per band
- **k** = b * r = total number of permutations (num_perm)

### Candidate Probability Formula

The probability that two documents with true Jaccard similarity s are identified
as candidates:

```
P(candidate) = 1 - (1 - s^r)^b
```

where:
- `s` = true Jaccard similarity
- `r` = rows per band
- `b` = number of bands

### S-Curve Analysis

This formula produces an S-shaped curve with a **threshold** at approximately:

```
t ≈ (1/b)^(1/r)
```

| b (bands) | r (rows) | k=b*r | Threshold | P(0.5) | P(0.7) | P(0.9) |
|-----------|----------|-------|-----------|--------|--------|--------|
| 5         | 25       | 125   | ~0.55     | 0.187  | 0.976  | 1.000  |
| 10        | 25       | 250   | ~0.63     | 0.009  | 0.823  | 1.000  |
| 5         | 51       | 255   | ~0.72     | 0.000  | 0.381  | 1.000  |
| 20        | 13       | 260   | ~0.58     | 0.060  | 0.988  | 1.000  |
| 32        | 8        | 256   | ~0.53     | 0.413  | 0.999  | 1.000  |

### Choosing b and r

Given a desired threshold t and num_perm k:

```python
import math

def optimal_bands(threshold: float, num_perm: int) -> tuple:
    """Find (b, r) that minimizes threshold error for given num_perm."""
    best = None
    for b in range(1, num_perm + 1):
        if num_perm % b != 0:
            continue
        r = num_perm // b
        t = (1.0 / b) ** (1.0 / r)
        error = abs(t - threshold)
        if best is None or error < best[0]:
            best = (error, b, r)
    return best[1], best[2]  # b, r
```

### BigCode Production Parameters

The BigCode project (StarCoder, etc.) uses these battle-tested settings:

```
num_perm = 256
threshold = 0.7
ngram_size = 5        # word 5-grams
num_bands = 5         # 5 bands of 51 rows (256 / 5 ≈ 51, last band padded)
```

**Note**: When num_perm is not perfectly divisible by num_bands, the last band
has fewer rows. Most libraries handle this automatically.

---

## Library Comparison

### rensa (Recommended)

**Rust-backed MinHash library, ~50x faster than datasketch.**

- Install: `pip install rensa`
- API:

```python
from rensa import RMinHash, RMinHashLSH

# Create signature
mh = RMinHash(num_perm=128, seed=42)
for shingle in shingles:
    mh.update(shingle.encode("utf-8"))
signature = mh.digest()

# LSH index
lsh = RMinHashLSH(threshold=0.7, num_perm=128, num_bands=5)
lsh.insert(doc_id, signature)
candidates = lsh.query(query_signature)
```

**Advantages**:
- 50x faster hash computation than datasketch
- Low memory overhead from Rust implementation
- Thread-safe, suitable for multiprocessing
- Seed-based reproducibility

**Limitations**:
- Newer library, smaller community
- Fewer features (no weighted MinHash, no HyperLogLog)

### datasketch (Fallback)

**Pure Python, widely used, feature-rich.**

- Install: `pip install datasketch`
- API:

```python
from datasketch import MinHash, MinHashLSH

# Create signature
mh = MinHash(num_perm=256)
for shingle in shingles:
    mh.update(shingle.encode("utf-8"))

# LSH index
lsh = MinHashLSH(threshold=0.7, num_perm=256)
lsh.insert("doc_id", mh)
result = lsh.query(query_mh)
```

**Advantages**:
- Mature, well-documented, widely cited
- Supports weighted MinHash, b-bit MinHash, HyperLogLog++
- Redis-backed LSH for distributed operation
- Large community, many tutorials

**Limitations**:
- Pure Python: ~50x slower than rensa for hash computation
- Higher memory footprint per MinHash object

### text-dedup (CLI Pipeline)

**End-to-end dedup toolkit with CLI interface.**

- Install: `pip install text-dedup`
- CLI:

```bash
python -m text_dedup.minhash \
    --path /data/corpus \
    --output /data/deduped \
    --threshold 0.7 \
    --num-perm 256 \
    --ngram 5 \
    --num-bands 5 \
    --column text
```

- TOML configuration:

```toml
[minhash]
threshold = 0.7
num_perm = 256
ngram = 5
num_bands = 5
column = "text"
```

**Advantages**:
- Batteries-included: MinHash, SimHash, SuffixArray, BloomFilter, ExactDedup
- HuggingFace datasets integration
- TOML config for reproducibility
- Spark backend for large-scale processing

**Limitations**:
- Less flexible for custom pipeline integration
- Opinionated about data formats (expects HF datasets or Arrow)

### NeMo Curator (GPU-Accelerated)

**NVIDIA's GPU-accelerated data curation toolkit.**

- Install: `pip install nemo-curator[cuda12x]`
- API:

```python
from nemo_curator import MinHashDedup
from nemo_curator.utils.distributed import get_client

client = get_client("gpu")
deduper = MinHashDedup(
    num_perm=256,
    threshold=0.7,
    ngram=5,
    num_bands=5,
)
result = deduper(dataset)
```

**Advantages**:
- GPU-accelerated MinHash computation (cuML)
- Handles 100B+ token corpora
- Integrated with NeMo training pipeline
- Dask distributed backend

**Limitations**:
- Requires NVIDIA GPU with CUDA
- Heavy dependency chain (RAPIDS, Dask, cuML)
- Less suitable for small-scale or CPU-only environments

---

## Comparison Matrix

| Feature | rensa | datasketch | text-dedup | NeMo Curator |
|---------|-------|------------|------------|--------------|
| Speed | 50x baseline | 1x baseline | 1x (wraps datasketch) | GPU-accelerated |
| Memory | Low (Rust) | Medium (Python) | Medium | High (GPU VRAM) |
| API Style | Library | Library | CLI + Library | Library |
| Reproducibility | Seed-based | Seed-based | TOML config | Config dict |
| Scale | 10B tokens | 1B tokens | 10B tokens | 100B+ tokens |
| GPU Required | No | No | No | Yes |
| Install Complexity | Low | Low | Medium | High |
| Community Size | Small | Large | Medium | Medium |
| Weighted MinHash | No | Yes | No | No |

---

## Parameter Tuning Guidance

### Threshold Selection

| Use Case | Recommended Threshold | Rationale |
|----------|----------------------|-----------|
| Aggressive dedup (web scrape) | 0.5 - 0.6 | Remove all near-duplicates |
| Standard dedup (training data) | 0.7 | BigCode default, good balance |
| Conservative dedup (curated data) | 0.8 - 0.9 | Only very similar documents |
| Plagiarism detection | 0.3 - 0.5 | Catch loosely similar content |

### num_perm Selection

- **64**: Fast, rough dedup. Good for prototyping.
- **128**: Reasonable accuracy for most use cases.
- **256**: Production standard (BigCode). Recommended.
- **512+**: Diminishing returns. Only for high-precision research.

### ngram_size Selection

- **3-gram words**: Catches rephrased content but high false positive rate.
- **5-gram words**: Standard. Good for paragraph-level similarity.
- **7-gram words**: More conservative. Better for long-document dedup.
- **13-gram characters**: Common for short-text or code dedup.

### num_bands vs Accuracy Tradeoffs

Increasing bands (b) with constant num_perm:
- **More bands**: Lower threshold, more candidates, more false positives, fewer false negatives
- **Fewer bands**: Higher threshold, fewer candidates, fewer false positives, more false negatives

### Memory Budget Estimation

```
Per-document memory = num_perm * 4 bytes (uint32 signatures)
LSH index memory ≈ num_docs * num_perm * 4 bytes * 1.5 (overhead)

Example: 100M documents, 256 perms
  = 100M * 256 * 4 * 1.5 = 153.6 GB
```

For corpora exceeding available RAM, consider:
1. Sharded LSH (partition by document hash prefix)
2. Disk-backed LSH (datasketch with Redis or LevelDB backend)
3. Two-pass: first pass computes signatures to disk, second pass builds LSH

---

## Implementation Checklist

- [ ] Choose shingle type (word vs char) and size based on corpus
- [ ] Select num_perm based on accuracy requirements
- [ ] Compute optimal (b, r) from threshold and num_perm
- [ ] Implement with rensa (preferred) with datasketch fallback
- [ ] Verify with known near-duplicate injection test
- [ ] Measure false positive rate on random non-duplicate pairs
- [ ] Profile memory and throughput on representative subset
- [ ] Document parameters in pipeline config for reproducibility
