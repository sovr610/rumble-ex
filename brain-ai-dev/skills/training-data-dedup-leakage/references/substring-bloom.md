# Substring Dedup + Bloom Filter Reference

## Purpose

This reference covers two complementary deduplication techniques:

1. **Suffix array substring dedup** -- detects verbatim shared substrings across
   documents (Stage 4)
2. **Bloom filter fast-path** -- streaming probabilistic exact-match dedup with
   bounded memory (Stage 5)

---

## Suffix Array Substring Dedup

### Overview

Suffix arrays enable efficient detection of all shared substrings of length >= L
across a corpus. This catches repeated boilerplate passages, copied paragraphs,
and shared headers/footers that document-level hashing misses.

### Theory

A **suffix array** SA for a string S of length n is a sorted array of all suffixes
of S represented by their starting positions. Combined with the **LCP (Longest
Common Prefix) array**, it enables O(n) detection of all repeated substrings.

```
String: "banana$"
Suffixes:         Sorted:        SA:  LCP:
0: banana$       5: a$           5    0
1: anana$        3: ana$         3    1
2: nana$         1: anana$       1    3
3: ana$          0: banana$      0    0
4: na$           4: na$          4    0
5: a$            2: nana$        2    2
6: $             6: $            6    0
```

### google-research/deduplicate-text-datasets

The primary implementation for large-scale substring dedup is the
[deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets)
project from Google Research (Lee et al., 2022).

#### Architecture

```
Corpus (text files)
  → Concatenate with sentinel separators
  → Build suffix array (Rust, O(n) algorithm)
  → Compute LCP array
  → Find duplicate substrings (length >= threshold)
  → Output byte ranges to remove
  → Apply removals to produce deduplicated corpus
```

#### Installation and Usage

```bash
# Clone the repository
git clone https://github.com/google-research/deduplicate-text-datasets
cd deduplicate-text-datasets

# Build the Rust components
cargo build --release

# Step 1: Build suffix array
python scripts/make_suffix_array.py \
    --input corpus.txt \
    --output corpus.sa

# Step 2: Find duplicate substrings
python scripts/find_duplicates.py \
    --input corpus.txt \
    --suffix-array corpus.sa \
    --length-threshold 200 \
    --output duplicates.json

# Step 3: Remove duplicates
python scripts/apply_removals.py \
    --input corpus.txt \
    --duplicates duplicates.json \
    --output corpus_deduped.txt
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `length-threshold` | 200 | Minimum substring length to consider as duplicate (characters) |
| `k` | 100 | Merge duplicate regions within k characters of each other |

#### Memory Requirements

The suffix array requires approximately **5-8x the corpus size** in memory:
- Suffix array: 4 bytes per character (int32 positions)
- LCP array: 4 bytes per character
- Original text: 1 byte per character

| Corpus Size | RAM Required |
|-------------|-------------|
| 1 GB        | 5-8 GB      |
| 10 GB       | 50-80 GB    |
| 100 GB      | 500-800 GB  |

For corpora exceeding available RAM, use the **sharded suffix array** approach:
partition the corpus into chunks, build per-chunk suffix arrays, and merge results.

#### When to Use

Enable substring dedup when:
- Corpus contains web scrapes with shared navigation/footer elements
- Documents share long code snippets or license texts
- Repeated boilerplate passages inflate corpus size by > 5%
- You need to remove verbatim copied paragraphs while keeping the
  surrounding unique content

Skip when:
- Corpus is already curated (books, papers with minimal overlap)
- Memory budget is tight (suffix arrays are memory-hungry)
- Document-level dedup (exact + MinHash) already achieves target dedup ratio

---

## Bloom Filter Theory

### Overview

A Bloom filter is a space-efficient probabilistic data structure that tests
whether an element is a member of a set. It may return **false positives** but
**never false negatives**.

### Structure

- **Bit array** of m bits, initially all 0
- **k independent hash functions**, each mapping elements to positions in [0, m)

**Insert(x)**: Set bits at positions h_1(x), h_2(x), ..., h_k(x) to 1.

**Query(x)**: Check if ALL bits at h_1(x), h_2(x), ..., h_k(x) are 1.
If any bit is 0, x is definitely not in the set. If all are 1, x is
*probably* in the set (may be a false positive).

### Optimal Parameters

Given:
- **n** = expected number of elements
- **p** = desired false positive rate

Optimal number of hash functions:
```
k = (m / n) * ln(2)
```

Optimal bit array size:
```
m = -(n * ln(p)) / (ln(2))^2
```

False positive rate:
```
FP = (1 - e^(-kn/m))^k
```

### Sizing Tables

#### Given n (items) and p (FP rate), compute m (bits) and k (hash functions)

| n (items) | p (FP rate) | m (bits) | m (bytes) | k (hashes) |
|-----------|-------------|----------|-----------|------------|
| 100K      | 0.01        | 958,506  | 117 KB    | 7          |
| 100K      | 0.001       | 1,437,759 | 175 KB   | 10         |
| 1M        | 0.01        | 9,585,059 | 1.1 MB   | 7          |
| 1M        | 0.001       | 14,377,588 | 1.7 MB  | 10         |
| 10M       | 0.01        | 95,850,584 | 11.4 MB | 7          |
| 10M       | 0.001       | 143,775,875 | 17.1 MB | 10        |
| 100M      | 0.01        | 958,505,839 | 114 MB  | 7          |
| 100M      | 0.001       | 1,437,758,757 | 171 MB | 10       |
| 1B        | 0.01        | 9.58 Gbits | 1.1 GB   | 7          |
| 1B        | 0.001       | 14.38 Gbits | 1.7 GB  | 10         |

#### Memory comparison: Hash Set vs Bloom Filter

| n (items) | Hash Set (SHA-256, 32B each) | Bloom Filter (p=0.001) | Savings |
|-----------|------------------------------|------------------------|---------|
| 1M        | 32 MB                        | 1.7 MB                 | 18.8x   |
| 10M       | 320 MB                       | 17.1 MB                | 18.7x   |
| 100M      | 3.2 GB                       | 171 MB                 | 18.7x   |
| 1B        | 32 GB                        | 1.7 GB                 | 18.8x   |

### Python Implementation

```python
import math
import mmh3  # MurmurHash3
from bitarray import bitarray

class BloomFilter:
    """Simple Bloom filter implementation."""

    def __init__(self, expected_items: int, fp_rate: float = 0.001):
        self.n = expected_items
        self.p = fp_rate
        self.m = self._optimal_m(expected_items, fp_rate)
        self.k = self._optimal_k(self.m, expected_items)
        self.bits = bitarray(self.m)
        self.bits.setall(0)
        self.count = 0

    @staticmethod
    def _optimal_m(n: int, p: float) -> int:
        """Compute optimal bit array size."""
        return int(-n * math.log(p) / (math.log(2) ** 2))

    @staticmethod
    def _optimal_k(m: int, n: int) -> int:
        """Compute optimal number of hash functions."""
        return max(1, int((m / n) * math.log(2)))

    def _hashes(self, item: bytes) -> list:
        """Compute k hash positions using double hashing."""
        h1 = mmh3.hash(item, seed=0) % self.m
        h2 = mmh3.hash(item, seed=42) % self.m
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, item: bytes) -> None:
        """Add an item to the filter."""
        for pos in self._hashes(item):
            self.bits[pos] = 1
        self.count += 1

    def __contains__(self, item: bytes) -> bool:
        """Check if item might be in the filter (may return false positive)."""
        return all(self.bits[pos] for pos in self._hashes(item))

    @property
    def estimated_fp_rate(self) -> float:
        """Estimate current false positive rate based on fill ratio."""
        fill = self.bits.count(1) / self.m
        return fill ** self.k
```

---

## Streaming Dedup with Bloom Filters

### Use Case

For streaming pipelines where documents arrive one at a time and memory is
constrained, a Bloom filter provides probabilistic exact-match dedup without
storing all hashes in memory.

### Architecture

```
Stream of documents
  → Canonicalize
  → Hash (SHA-256)
  → Check Bloom filter:
      If present → probable duplicate, discard (or verify with exact store)
      If absent  → definitely new, add to Bloom filter, pass through
  → Clean stream
```

### False Positive Handling

Since Bloom filters have false positives, a small fraction of unique documents
will be incorrectly flagged as duplicates. Strategies:

1. **Accept the loss**: If FP rate is 0.001 (0.1%), losing 0.1% of unique
   documents is acceptable for most training corpora.
2. **Two-stage verification**: Use Bloom filter as first pass, then verify
   positives against an exact store (disk-backed hash set or database).
3. **Counting Bloom filter**: Track insert counts to allow deletion, but
   at higher memory cost (4 bits per slot instead of 1).

### Streaming Pipeline Pattern

```python
class StreamingDedup:
    """Streaming exact dedup using Bloom filter."""

    def __init__(self, expected_items: int, fp_rate: float = 0.001):
        self.bloom = BloomFilter(expected_items, fp_rate)
        self.stats = {"seen": 0, "duplicates": 0, "passed": 0}

    def process(self, documents):
        """Process a stream of documents, yielding unique ones."""
        import hashlib
        for doc in documents:
            self.stats["seen"] += 1
            doc_hash = hashlib.sha256(doc.encode("utf-8")).digest()
            if doc_hash in self.bloom:
                self.stats["duplicates"] += 1
                continue
            self.bloom.add(doc_hash)
            self.stats["passed"] += 1
            yield doc
```

---

## Integration with the Pipeline

### Stage 4: Substring Dedup (Optional)

Position in pipeline: after MinHash fuzzy dedup, before Bloom filter.

```python
class SubstringDedup:
    """Wrapper for suffix-array substring dedup."""

    def __init__(self, min_length: int = 200):
        self.min_length = min_length

    def fit(self, corpus: list) -> dict:
        """Find duplicate substrings across corpus.

        Returns dict mapping doc_idx -> list of (start, end) byte ranges
        to remove.
        """
        # In production, this calls the Rust suffix array implementation
        # from google-research/deduplicate-text-datasets
        pass

    def transform(self, corpus: list, removals: dict) -> list:
        """Apply substring removals to produce deduplicated corpus."""
        result = []
        for idx, doc in enumerate(corpus):
            if idx in removals:
                # Remove identified duplicate substrings
                chars = list(doc)
                for start, end in sorted(removals[idx], reverse=True):
                    chars[start:end] = []
                result.append("".join(chars))
            else:
                result.append(doc)
        return result
```

### Stage 5: Bloom Filter Fast-Path

Position in pipeline: after substring dedup, before leakage guard.

The Bloom filter serves two purposes:
1. **Streaming exact dedup** for new documents being added to an existing corpus
2. **Memory-efficient duplicate tracking** when the hash set exceeds RAM

```python
class BloomFastPath:
    """Bloom filter stage for streaming/memory-constrained exact dedup."""

    def __init__(self, expected_items: int, fp_rate: float = 0.001):
        self.bloom = BloomFilter(expected_items, fp_rate)

    def fit(self, corpus: list) -> list:
        """Process corpus through Bloom filter, returning unique documents."""
        import hashlib
        unique = []
        for doc in corpus:
            h = hashlib.sha256(doc.encode("utf-8")).digest()
            if h not in self.bloom:
                self.bloom.add(h)
                unique.append(doc)
        return unique
```

### Combined Pipeline Flow

```
Raw Corpus
  → Stage 1: Canonicalization
  → Stage 2: Exact Dedup (hash set, catches 100% of exact dups)
  → Stage 3: MinHash + LSH (catches near-duplicates above threshold)
  → Stage 4: Substring Dedup [optional] (catches shared passages)
  → Stage 5: Bloom Filter [optional] (streaming exact check for new data)
  → Stage 6: Leakage Guard (checks against holdout sets)
  → Stage 7: Reporting
  → Clean Corpus
```

**Note**: Stages 2 and 5 both do exact dedup but serve different roles:
- Stage 2 uses a full hash set for the batch corpus (no false positives)
- Stage 5 uses a Bloom filter for streaming/incremental additions (space-efficient
  but with configurable false positive rate)

---

## Performance Comparison

| Method | Time Complexity | Space Complexity | False Positives | False Negatives |
|--------|----------------|------------------|-----------------|-----------------|
| Hash Set (Stage 2) | O(n) | O(n * 32B) | 0% | 0% |
| Bloom Filter (Stage 5) | O(n * k) | O(m bits) | configurable | 0% |
| Suffix Array (Stage 4) | O(n log n) | O(5-8n) | 0% | 0% |
| MinHash + LSH (Stage 3) | O(n * k) | O(n * k * 4B) | configurable | configurable |

Where n = corpus size in documents (or characters for suffix array), k = number of
hash functions/permutations, m = Bloom filter bit array size.
