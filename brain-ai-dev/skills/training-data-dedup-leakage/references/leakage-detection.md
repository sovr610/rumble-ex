# Leakage Detection Reference

## Purpose

Leakage detection ensures that no holdout/test/validation data contaminates the
training corpus. Even partial contamination can inflate evaluation metrics and mask
real model weaknesses. This reference covers three complementary detection methods
and the audit trail infrastructure.

Design principle: **assume contamination exists; prove it does not.**

---

## N-Gram Overlap Detection

### Method

Compute the set of n-grams in a training document and measure the intersection
ratio with each holdout document's n-gram set.

### Algorithm

```
For a training document T and holdout document H:

1. Compute n-gram sets:
   ngrams_T = {T[i:i+n] for i in range(len(T) - n + 1)}
   ngrams_H = {H[i:i+n] for i in range(len(H) - n + 1)}

2. Compute overlap ratio:
   overlap = |ngrams_T ∩ ngrams_H| / |ngrams_H|

3. Flag if overlap > threshold (default: 0.8)
```

### Why 13-grams?

The standard choice for contamination detection is **13-gram word overlap**:

- **13 consecutive words** are extremely unlikely to co-occur by chance in
  independently written text
- Shorter n-grams (5-8) produce many false positives from common phrases
- Longer n-grams (20+) miss paraphrased contamination
- The GPT-3 paper (Brown et al., 2020) used 13-gram overlap for contamination analysis
- OpenAI continues to use 13-gram overlap in subsequent model evaluations

### Overlap Ratio Direction

The ratio is computed relative to the **holdout document**, not the training document:

```
overlap = |ngrams_T ∩ ngrams_H| / |ngrams_H|
```

This catches both:
- **Full contamination**: Training doc contains the entire holdout doc (ratio ~1.0)
- **Partial contamination**: Training doc contains significant portions of holdout (ratio > 0.8)

### Implementation

```python
from typing import Set, List, Tuple

def compute_word_ngrams(text: str, n: int = 13) -> Set[Tuple[str, ...]]:
    """Compute set of word n-grams from text."""
    words = text.lower().split()
    if len(words) < n:
        return set()
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

def ngram_overlap_ratio(
    train_text: str,
    holdout_text: str,
    n: int = 13,
) -> float:
    """Compute n-gram overlap ratio (intersection / holdout n-grams)."""
    train_ngrams = compute_word_ngrams(train_text, n)
    holdout_ngrams = compute_word_ngrams(holdout_text, n)
    if not holdout_ngrams:
        return 0.0
    intersection = train_ngrams & holdout_ngrams
    return len(intersection) / len(holdout_ngrams)
```

### Performance Optimization

For large holdout sets, pre-compute n-gram sets and store them in memory:

```python
# Pre-compute holdout n-gram index
holdout_ngram_index = {}
for doc_id, text in holdout_docs:
    ngrams = compute_word_ngrams(text, n=13)
    holdout_ngram_index[doc_id] = ngrams

# For each training doc, check against all holdout docs
for train_text in training_corpus:
    train_ngrams = compute_word_ngrams(train_text, n=13)
    for holdout_id, holdout_ngrams in holdout_ngram_index.items():
        overlap = len(train_ngrams & holdout_ngrams) / len(holdout_ngrams)
        if overlap > 0.8:
            flag_contamination(train_text, holdout_id, overlap)
```

For very large holdout sets (>100K documents), consider:
1. **Bloom filter pre-filter**: Check if any training n-gram exists in a Bloom filter
   of all holdout n-grams before doing exact set intersection.
2. **Inverted index**: Map each n-gram to the set of holdout documents containing it.
   For a training doc, look up each of its n-grams and accumulate counts per holdout doc.

---

## Holdout Fingerprinting

### Method

Pre-compute MinHash signatures for all holdout documents. For each training candidate,
compute its MinHash and query the holdout LSH index. This catches fuzzy contamination
(paraphrased or lightly edited versions of holdout data).

### Implementation

```python
class HoldoutFingerprinter:
    """Pre-compute and store MinHash fingerprints for holdout sets."""

    def __init__(self, num_perm: int = 256, threshold: float = 0.7):
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = None  # Initialized on register
        self.fingerprints = {}  # holdout_name -> {doc_id -> MinHash}

    def register_holdout(self, name: str, documents: dict):
        """Register a holdout set by computing MinHash for each document."""
        from datasketch import MinHash, MinHashLSH

        if self.lsh is None:
            self.lsh = MinHashLSH(
                threshold=self.threshold,
                num_perm=self.num_perm,
            )

        self.fingerprints[name] = {}
        for doc_id, text in documents.items():
            mh = self._compute_minhash(text)
            self.fingerprints[name][doc_id] = mh
            key = f"{name}::{doc_id}"
            self.lsh.insert(key, mh)

    def check(self, text: str) -> list:
        """Check a training text against all holdout fingerprints."""
        if self.lsh is None:
            return []
        mh = self._compute_minhash(text)
        candidates = self.lsh.query(mh)
        results = []
        for key in candidates:
            holdout_name, doc_id = key.split("::", 1)
            holdout_mh = self.fingerprints[holdout_name][doc_id]
            similarity = mh.jaccard(holdout_mh)
            results.append({
                "holdout_set": holdout_name,
                "holdout_doc_id": doc_id,
                "jaccard_similarity": similarity,
            })
        return results

    def _compute_minhash(self, text: str):
        from datasketch import MinHash
        mh = MinHash(num_perm=self.num_perm)
        words = text.lower().split()
        for i in range(len(words) - 4):
            shingle = " ".join(words[i:i+5])
            mh.update(shingle.encode("utf-8"))
        return mh
```

### Fingerprint Storage

For persistent holdout fingerprints across pipeline runs:

```python
import json
import numpy as np

def save_fingerprints(fingerprints: dict, path: str):
    """Save MinHash fingerprints to JSON-compatible format."""
    data = {}
    for name, docs in fingerprints.items():
        data[name] = {
            doc_id: mh.hashvalues.tolist()
            for doc_id, mh in docs.items()
        }
    with open(path, "w") as f:
        json.dump(data, f)

def load_fingerprints(path: str, num_perm: int = 256) -> dict:
    """Load MinHash fingerprints from JSON."""
    from datasketch import MinHash
    with open(path) as f:
        data = json.load(f)
    fingerprints = {}
    for name, docs in data.items():
        fingerprints[name] = {}
        for doc_id, hashvals in docs.items():
            mh = MinHash(num_perm=num_perm)
            mh.hashvalues = np.array(hashvals, dtype=np.uint64)
            fingerprints[name][doc_id] = mh
    return fingerprints
```

---

## Fuzzy Contamination Checking

### Multi-Level Detection Strategy

Combine all three methods in a cascade for comprehensive leakage detection:

```
For each training document T:

Level 1 - Exact Match (cheapest):
  hash(T) in holdout_hash_set? → EXACT_CONTAMINATION

Level 2 - N-gram Overlap (moderate cost):
  For each holdout doc H:
    overlap_ratio(T, H, n=13) > 0.8? → NGRAM_CONTAMINATION

Level 3 - Fuzzy Match (most expensive):
  query holdout LSH index for T:
    any result with Jaccard > 0.7? → FUZZY_CONTAMINATION
```

### Severity Classification

| Match Type | Severity | Confidence | Action |
|------------|----------|------------|--------|
| Exact match (hash) | Critical | 100% | Remove |
| N-gram overlap > 0.9 | Critical | ~95% | Remove |
| N-gram overlap 0.8-0.9 | High | ~85% | Remove + review |
| Fuzzy match > 0.8 | High | ~80% | Remove + review |
| Fuzzy match 0.7-0.8 | Medium | ~60% | Flag for manual review |
| Fuzzy match 0.5-0.7 | Low | ~30% | Log, do not remove |

### Paraphrase Detection Challenges

Fuzzy matching with MinHash may miss sophisticated paraphrases that change many
words while preserving meaning. Additional strategies:

1. **Sentence embedding similarity**: Compute sentence embeddings (e.g., with
   `sentence-transformers`) and flag cosine similarity > 0.95 against holdout.
2. **ROUGE-L**: Compute longest common subsequence ratio. Catches reordered content.
3. **Back-translation artifacts**: If holdout was machine-translated and
   back-translated, word-level similarity drops but n-gram patterns persist.

---

## Audit Trail JSON Schema

### Schema Definition

Every pipeline run produces a JSON audit trail for reproducibility and compliance.

```json
{
  "$schema": "dedup_audit_v1",
  "pipeline_run_id": "uuid-v4",
  "timestamp": "2026-02-22T12:00:00Z",
  "config": {
    "canonicalization": {
      "lowercase": true,
      "unicode_normalize": "NFKC",
      "strip_urls": false,
      "strip_emails": false
    },
    "exact_dedup": {
      "hash_algo": "sha256",
      "bloom_prefilter": false
    },
    "minhash": {
      "num_perm": 256,
      "threshold": 0.7,
      "ngram_size": 5,
      "num_bands": 5,
      "backend": "rensa"
    },
    "leakage": {
      "ngram_size": 13,
      "overlap_threshold": 0.8,
      "fuzzy_threshold": 0.7
    }
  },
  "summary": {
    "total_input": 1000000,
    "exact_duplicates_removed": 45000,
    "fuzzy_duplicates_removed": 12000,
    "substring_duplicates_removed": 0,
    "leakage_flagged": 23,
    "total_output": 942977,
    "dedup_ratio": 0.057,
    "processing_time_seconds": 3600.5
  },
  "per_stage_timing": {
    "canonicalization": 120.3,
    "exact_dedup": 450.1,
    "minhash_dedup": 2800.0,
    "substring_dedup": 0.0,
    "bloom_filter": 15.2,
    "leakage_guard": 200.5,
    "reporting": 14.4
  },
  "per_stage_memory_mb": {
    "canonicalization": 512,
    "exact_dedup": 2048,
    "minhash_dedup": 8192,
    "bloom_filter": 128,
    "leakage_guard": 1024
  },
  "leakage_details": [
    {
      "train_doc_id": "doc_123456",
      "holdout_set": "test_v1",
      "holdout_doc_id": "test_42",
      "match_type": "ngram_overlap",
      "similarity_score": 0.92,
      "flagged_at": "2026-02-22T13:15:30Z"
    }
  ],
  "jaccard_distribution": {
    "histogram_bins": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "counts": [0, 0, 0, 500, 800, 1200, 3000, 8000, 2000, 500, 0]
  },
  "removed_ids": {
    "exact_dedup": ["doc_001", "doc_002"],
    "fuzzy_dedup": ["doc_100", "doc_101"],
    "leakage": ["doc_500"]
  },
  "holdout_contamination_summary": {
    "test_v1": {"exact": 5, "ngram": 12, "fuzzy": 6},
    "validation_v1": {"exact": 0, "ngram": 0, "fuzzy": 0}
  }
}
```

### Audit Trail Requirements

1. **Completeness**: Every removed or flagged document has a recorded reason
2. **Reproducibility**: Configuration is fully specified; re-running with same
   config on same data produces the same audit trail
3. **Traceability**: Each flagged item links back to the specific holdout document
   and match type that triggered the flag
4. **Immutability**: Audit trails are append-only; do not modify previous runs

---

## Benchmark Contamination Datasets

### HuggingFace BenchmarkContamination

The `benchmark_contamination` dataset on HuggingFace provides known contaminated
and clean pairs for testing leakage detection systems.

```python
from datasets import load_dataset

ds = load_dataset("swj0419/benchmark_contamination")
# Contains columns: text, benchmark, is_contaminated, contamination_type
```

### GPT-3 Contamination Analysis

Brown et al. (2020) published contamination analysis for GPT-3 across multiple
benchmarks. Key findings:

- **LAMBADA**: 1.6% 13-gram overlap (considered clean)
- **HellaSwag**: 5.6% overlap (moderate contamination)
- **WebText test set**: 11.2% overlap with Common Crawl training data

### Creating Synthetic Test Data

For unit testing leakage detection, create controlled contamination:

```python
def create_contamination_test_set():
    """Create synthetic holdout and contaminated training data."""
    holdout = [
        "The quick brown fox jumps over the lazy dog near the river bank on a sunny afternoon",
        "Machine learning models require large datasets for training and evaluation",
    ]

    training = [
        # Exact contamination
        holdout[0],
        # Partial contamination (first half)
        " ".join(holdout[1].split()[:8]) + " and something else entirely new",
        # Paraphrase contamination
        "A fast brown fox leaps over a sleepy dog by the stream on a bright day",
        # Clean (no contamination)
        "Completely unrelated text about quantum physics and black holes",
    ]

    expected_flags = [
        {"doc_idx": 0, "type": "exact", "holdout_idx": 0},
        {"doc_idx": 1, "type": "ngram", "holdout_idx": 1},
        {"doc_idx": 2, "type": "fuzzy", "holdout_idx": 0},
    ]

    return holdout, training, expected_flags
```

---

## Production Deployment Checklist

- [ ] Pre-compute holdout fingerprints (MinHash + n-gram sets) and persist to disk
- [ ] Load holdout fingerprints at pipeline startup (not per-document)
- [ ] Run leakage check as the *last* dedup stage (after exact and fuzzy dedup
      have already removed most noise)
- [ ] Set n-gram overlap threshold conservatively (0.8) to avoid false positives
- [ ] Log all flagged samples with full context (match type, score, holdout doc ID)
- [ ] Generate audit trail JSON for every pipeline run
- [ ] Periodically re-check after adding new holdout sets (e.g., new benchmarks)
- [ ] Store audit trails alongside the clean corpus for provenance
