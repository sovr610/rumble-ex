---
name: Training Data Dedup + Leakage Guard
description: >
  This skill should be used when the user asks to "deduplicate training data",
  "remove duplicate samples", "implement MinHash LSH deduplication",
  "detect train-test leakage", "compute n-gram overlap between splits",
  "add data fingerprinting", "canonicalize text for dedup",
  "run exact-match deduplication", "substring deduplication with suffix arrays",
  "configure dedup thresholds", "audit data contamination",
  "compute Jaccard similarity between documents",
  "rensa MinHash", "text-dedup pipeline",
  "content hash dedup", "bloom filter dedup",
  "leakage guard holdout set", "data quality pipeline",
  "NeMo Curator dedup", "BigCode dedup parameters",
  "fuzzy dedup near-duplicate detection",
  or needs guidance on building a multi-stage deduplication and leakage
  detection pipeline for training corpora with exact, fuzzy, and substring
  methods plus holdout contamination checking.
version: 0.1.0
---

# Training Data Dedup + Leakage Guard

## Overview

Build a multi-stage pipeline that removes exact duplicates, near-duplicates (fuzzy), and contaminated samples from training corpora before they reach the model. Ensure no holdout/test data leaks into the training split.

Design principle: **canonicalize first, hash cheaply, dedup progressively, guard holdout always.**

## Pipeline Stages

Seven sequential stages, each with a clear input/output contract:

```
Raw Corpus
  → Stage 1: Canonicalization (normalize whitespace, case, unicode)
  → Stage 2: Content-Hash Exact Dedup (SHA-256 / MD5)
  → Stage 3: MinHash + LSH Fuzzy Dedup (Jaccard ≥ threshold)
  → Stage 4: Substring Dedup (suffix array, optional)
  → Stage 5: Bloom Filter Fast-Path (streaming exact check)
  → Stage 6: Leakage Guard (n-gram overlap with holdout sets)
  → Stage 7: Reporting + Audit Trail
  → Clean Corpus
```

## Public Contract

### Canonicalizer

Normalize text to a stable form before hashing.

```python
class Canonicalizer:
    def __init__(self, cfg: CanonConfig): ...
    def __call__(self, text: str) -> str: ...
```

Operations: lowercase, unicode NFKC, collapse whitespace, strip URLs/emails (optional), remove boilerplate headers.

### ExactDedup

Content-hash exact deduplication using SHA-256.

```python
class ExactDedup:
    def __init__(self, hash_algo: str = "sha256"): ...
    def fit(self, corpus: Iterable[str]) -> DedupReport: ...
    def is_duplicate(self, text: str) -> bool: ...
```

### MinHashDedup

Fuzzy near-duplicate detection using MinHash + LSH.

```python
class MinHashDedup:
    def __init__(self, cfg: MinHashConfig): ...
    def fit(self, corpus: Iterable[str]) -> DedupReport: ...
    def query(self, text: str) -> List[Tuple[int, float]]: ...
```

Key parameters (BigCode production defaults):
- `num_perm=256` — number of hash permutations
- `threshold=0.7` — Jaccard similarity threshold
- `ngram_size=5` — word n-gram shingle size
- `num_bands=5` — LSH bands (auto-computed from threshold if omitted)

### LeakageGuard

Check training samples against holdout/test sets for contamination.

```python
class LeakageGuard:
    def __init__(self, cfg: LeakageConfig): ...
    def register_holdout(self, name: str, texts: Iterable[str]) -> None: ...
    def check(self, text: str) -> LeakageResult: ...
    def audit(self, corpus: Iterable[str]) -> LeakageReport: ...
```

Detection methods: exact match, n-gram overlap ratio, MinHash similarity against holdout fingerprints.

### DedupPipeline

Orchestrate all stages into a single configurable pipeline.

```python
class DedupPipeline:
    def __init__(self, cfg: DedupPipelineConfig): ...
    def run(self, corpus: Iterable[str], holdouts: Dict[str, Iterable[str]]) -> PipelineResult: ...
    def report(self) -> Dict[str, Any]: ...
```

## Key Concepts

### Stage 1 — Canonicalization

Normalize before any comparison. Without canonicalization, trivial formatting differences (extra spaces, different Unicode representations) cause false negatives.

Standard pipeline: `NFKC normalize → lowercase → collapse whitespace → strip control chars`. Optional: remove URLs, emails, code comments, boilerplate headers.

### Stage 2 — Exact Dedup

Hash each canonicalized document with SHA-256. Store hashes in a set. O(n) time, O(n) memory for hash set. For corpora exceeding memory, use sorted hash files with external merge or a Bloom filter pre-filter.

### Stage 3 — MinHash + LSH Fuzzy Dedup

For near-duplicate detection (paraphrases, minor edits). Compute MinHash signatures from word n-gram shingles, then use LSH banding to find candidate pairs above a Jaccard threshold.

**Library options:**
- **rensa** — Rust-backed, 50x faster than datasketch. API: `RMinHash(num_perm, seed)`, `RMinHashLSH(threshold, num_perm, num_bands)`.
- **datasketch** — Pure Python, widely used. `MinHash`, `MinHashLSH`.
- **text-dedup** — CLI pipeline: `python -m text_dedup.minhash`. TOML config. Supports MinHash, SimHash, SuffixArray, BloomFilter.
- **NeMo Curator** — GPU-accelerated MinHash+LSH for massive corpora (100B+ tokens).

### Stage 4 — Substring Dedup (Optional)

Detect verbatim substring overlaps using suffix arrays. Based on google-research/deduplicate-text-datasets (Rust). Useful for removing boilerplate, license headers, repeated passages.

Enable when the corpus contains many documents sharing long common substrings (e.g., web scrapes with navigation chrome).

### Stage 5 — Bloom Filter Fast-Path

For streaming dedup where memory is constrained. Bloom filter gives probabilistic exact-match checking with configurable false-positive rate. Use as a first-pass filter before full MinHash.

### Stage 6 — Leakage Guard

Register all holdout/test/validation sets. For each training candidate, check:
1. **Exact match** — content hash exists in holdout hash set
2. **N-gram overlap** — compute 13-gram overlap ratio; flag if > threshold (default 0.8)
3. **Fuzzy match** — MinHash Jaccard against holdout fingerprints; flag if > threshold

Log every flagged sample with the holdout set name, match type, and similarity score.

### Stage 7 — Reporting

Emit a JSON audit report with:
- Total samples processed, exact dups removed, fuzzy dups removed, leakage flagged
- Per-stage timing and memory usage
- Distribution of Jaccard similarities for fuzzy matches
- Per-holdout contamination counts
- Sample IDs of all removed/flagged items (for reproducibility)

## Configuration Surface

```python
@dataclass
class DedupPipelineConfig:
    # Canonicalization
    lowercase: bool = True
    unicode_normalize: str = "NFKC"
    strip_urls: bool = False
    strip_emails: bool = False
    # Exact dedup
    hash_algo: str = "sha256"
    use_bloom_prefilter: bool = False
    bloom_expected_items: int = 10_000_000
    bloom_fp_rate: float = 0.001
    # MinHash LSH
    minhash_enabled: bool = True
    num_perm: int = 256
    jaccard_threshold: float = 0.7
    ngram_size: int = 5
    num_bands: Optional[int] = None       # auto from threshold
    minhash_backend: str = "rensa"         # rensa | datasketch | text_dedup
    # Substring dedup
    substring_enabled: bool = False
    min_substring_len: int = 200
    # Leakage guard
    leakage_enabled: bool = True
    leakage_ngram: int = 13
    leakage_overlap_threshold: float = 0.8
    leakage_fuzzy_threshold: float = 0.7
    # Reporting
    report_path: str = "dedup_report.json"
    save_removed_ids: bool = True
```

## Done-When Gates

1. **Canonicalization Works** — Identical documents with different whitespace/casing/unicode produce the same canonical form. Round-trip test: `canon(canon(text)) == canon(text)`.
2. **Exact Dedup Correct** — All bit-identical documents (post-canonicalization) are detected. Zero false negatives. Known-duplicate pairs produce matching hashes.
3. **Fuzzy Dedup Calibrated** — MinHash+LSH detects pairs above the Jaccard threshold. Injection test: insert a known near-duplicate pair (edit distance < 30%) and verify detection. False positive rate is below 5%.
4. **Leakage Guard Catches Contamination** — Insert a verbatim holdout sample into training; guard flags it. Insert a paraphrased version; guard flags it (fuzzy match). N-gram overlap correctly identifies partial contamination.
5. **Pipeline End-to-End** — Full pipeline runs on a test corpus, removes known duplicates, flags known leakage, and produces a valid JSON report with correct counts.

## Resources

### Reference Files
- **`references/canonicalization.md`** — Unicode normalization, whitespace collapsing, URL/email stripping, boilerplate removal, idempotency testing
- **`references/minhash-lsh.md`** — MinHash theory, LSH banding math, rensa vs datasketch vs text-dedup comparison, BigCode production parameters, parameter tuning guide
- **`references/leakage-detection.md`** — N-gram overlap detection, holdout fingerprinting, fuzzy contamination checking, audit trail format, benchmark contamination datasets
- **`references/substring-bloom.md`** — Suffix array substring dedup, Bloom filter theory and sizing, streaming dedup patterns, google-research/deduplicate-text-datasets usage
- **`references/testing-matrix.md`** — Test scenarios for all 7 pipeline stages

### Asset Files
- **`assets/canonicalizer_template.py`** — Canonicalizer with normalization pipeline, idempotency self-tests
- **`assets/exact_dedup_template.py`** — ExactDedup with SHA-256 hashing, Bloom filter option, self-tests
- **`assets/minhash_dedup_template.py`** — MinHashDedup with rensa/datasketch backends, LSH banding, self-tests
- **`assets/leakage_guard_template.py`** — LeakageGuard with exact/ngram/fuzzy checking, holdout registration, self-tests
- **`assets/dedup_pipeline_template.py`** — DedupPipeline orchestrating all stages, reporting, self-tests
- **`assets/dedup_config_template.py`** — DedupPipelineConfig with validation, serialization, self-tests

### Scripts
- **`scripts/validate_dedup.py`** — Validates done-when gates (canonicalization, exact, fuzzy, leakage, pipeline)
- **`scripts/gen_dedup_tests.py`** — Generates pytest test cases covering all 7 stages
- **`scripts/dedup_benchmark.py`** — Benchmarks dedup throughput across backends and corpus sizes
