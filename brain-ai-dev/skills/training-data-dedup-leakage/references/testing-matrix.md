# Testing Matrix Reference

## Purpose

This document defines test scenarios for all seven stages of the dedup + leakage
pipeline. Each test has a clear input, expected output, and pass criteria.

---

## Stage 1: Canonicalization

### 1.1 Unicode Edge Cases

| Test ID | Input | Expected Output | Pass Criteria |
|---------|-------|-----------------|---------------|
| C-U01 | `"\ufb01nance"` (fi ligature + nance) | `"finance"` | NFKC decomposes ligature |
| C-U02 | `"\uff21\uff22\uff23"` (fullwidth ABC) | `"abc"` | NFKC + lowercase |
| C-U03 | `"e\u0301"` (e + combining acute) | `"\u00e9"` | NFKC recomposition |
| C-U04 | `"\u2160\u2161\u2162"` (Roman I II III) | `"iii"` | NFKC + lowercase |
| C-U05 | `"\u00B2\u00B3"` (superscript 2,3) | `"23"` | NFKC compatibility |
| C-U06 | `"Caf\u00e9"` | `"caf\u00e9"` | Already composed, just lowercase |
| C-U07 | `"\u3000text\u3000"` (ideographic space) | `"text"` | Strip + collapse |
| C-U08 | `""` (empty string) | `""` | Empty input handled |
| C-U09 | `"\u200b\u200c\u200d"` (zero-width chars) | `""` | Zero-width stripped |
| C-U10 | `"\U0001f600 hello"` (emoji) | `"\U0001f600 hello"` | Emoji preserved |

### 1.2 Whitespace Collapsing

| Test ID | Input | Expected Output | Pass Criteria |
|---------|-------|-----------------|---------------|
| C-W01 | `"hello   world"` | `"hello world"` | Multiple spaces collapsed |
| C-W02 | `"hello\t\tworld"` | `"hello world"` | Tabs collapsed |
| C-W03 | `"hello\n\nworld"` | `"hello world"` | Newlines collapsed |
| C-W04 | `"  hello  "` | `"hello"` | Leading/trailing stripped |
| C-W05 | `"hello\r\nworld"` | `"hello world"` | CRLF handled |
| C-W06 | `"hello\u00a0world"` | `"hello world"` | NBSP collapsed |
| C-W07 | `"   "` | `""` | Whitespace-only -> empty |
| C-W08 | `"a"` | `"a"` | Single char unchanged |

### 1.3 URL/Email Stripping

| Test ID | Input | Expected Output | Pass Criteria |
|---------|-------|-----------------|---------------|
| C-S01 | `"visit https://example.com today"` | `"visit today"` | URL removed |
| C-S02 | `"email user@example.com please"` | `"email please"` | Email removed |
| C-S03 | `"http://a.com and http://b.com"` | `"and"` | Multiple URLs removed |
| C-S04 | `"no urls here"` | `"no urls here"` | No change when no URLs |
| C-S05 | `"ftp://files.example.com/data"` | `""` | FTP URL removed |

### 1.4 Idempotency

| Test ID | Input | Pass Criteria |
|---------|-------|---------------|
| C-I01 | Already-canonical text | `canon(text) == text` |
| C-I02 | Text with mixed issues | `canon(canon(text)) == canon(text)` |
| C-I03 | Unicode combining chars | `canon(canon(text)) == canon(text)` |
| C-I04 | Whitespace-heavy text | `canon(canon(text)) == canon(text)` |
| C-I05 | 10,000 random strings (hypothesis) | All satisfy idempotency |

---

## Stage 2: Exact Dedup

### 2.1 Collision Resistance

| Test ID | Input | Expected Output | Pass Criteria |
|---------|-------|-----------------|---------------|
| E-C01 | Two identical strings | Same hash | True positive: detected |
| E-C02 | Two strings differing by 1 char | Different hashes | No false positive |
| E-C03 | Empty string vs empty string | Same hash | Edge case: empty dedup |
| E-C04 | Unicode equivalent forms (post-canon) | Same hash | Canon + hash works |
| E-C05 | 1000 unique documents | 0 duplicates found | No false positives |

### 2.2 Known Duplicates

| Test ID | Input | Expected Output | Pass Criteria |
|---------|-------|-----------------|---------------|
| E-D01 | Corpus with 10 exact duplicates | Exactly 10 removed | Correct count |
| E-D02 | All docs identical (N copies) | N-1 removed, 1 kept | Keeps first occurrence |
| E-D03 | No duplicates in corpus | 0 removed | No false removals |
| E-D04 | Interleaved dups and unique | Correct partition | Order preserved |
| E-D05 | Large corpus (100K) with 5% exact dups | ~5K removed (+-0.5K) | Scalability |

### 2.3 Bloom Filter Pre-Filter

| Test ID | Input | Expected Output | Pass Criteria |
|---------|-------|-----------------|---------------|
| E-B01 | Known dup checked against Bloom | `True` (in filter) | True positive |
| E-B02 | Known unique checked against Bloom | `False` (probably) | Low FP rate |
| E-B03 | FP rate measurement (10K unique) | Measured FP < 2 * configured FP | Within tolerance |
| E-B04 | Bloom filter at capacity | FP rate increases gracefully | No crash |

---

## Stage 3: MinHash + LSH Fuzzy Dedup

### 3.1 Threshold Calibration

| Test ID | Pair Jaccard | Threshold | Expected | Pass Criteria |
|---------|-------------|-----------|----------|---------------|
| M-T01 | 0.95 | 0.7 | Detected | High similarity caught |
| M-T02 | 0.75 | 0.7 | Detected | Just above threshold |
| M-T03 | 0.65 | 0.7 | Not detected | Just below threshold |
| M-T04 | 0.30 | 0.7 | Not detected | Low similarity passes |
| M-T05 | 1.00 | 0.7 | Detected | Exact duplicate caught |

### 3.2 Injection Tests

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| M-I01 | Insert doc with 10% words changed | Detected (J ~ 0.8) |
| M-I02 | Insert doc with 20% words changed | Detected (J ~ 0.7) |
| M-I03 | Insert doc with 40% words changed | Not detected (J ~ 0.5) |
| M-I04 | Insert doc with sentences reordered | Detected (same shingles) |
| M-I05 | Insert doc with added paragraph | Detected if most content shared |
| M-I06 | Two completely unrelated docs | Not detected |

### 3.3 Backend Consistency

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| M-B01 | rensa and datasketch same corpus | Same near-dup pairs (within 5%) |
| M-B02 | rensa deterministic with same seed | Identical results across runs |
| M-B03 | datasketch deterministic with same seed | Identical results across runs |
| M-B04 | Empty document handling | No crash, no false matches |
| M-B05 | Very short document (< ngram_size words) | Graceful handling (skip or warn) |

### 3.4 False Positive Rate

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| M-F01 | 1000 random unique docs, threshold 0.7 | FP rate < 5% |
| M-F02 | 10000 random unique docs, threshold 0.7 | FP rate < 5% |
| M-F03 | Adversarial near-threshold pairs | Correct classification > 90% |

---

## Stage 4: Substring Dedup

### 4.1 Overlapping Passages

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| S-O01 | Two docs sharing 500-char passage | Passage identified as duplicate |
| S-O02 | Shared passage < min_length (200) | Not flagged (below threshold) |
| S-O03 | Shared passage = exactly min_length | Flagged (boundary case) |
| S-O04 | Three docs sharing same passage | All instances detected |
| S-O05 | Nested shared substrings | Correctly handled (no double-removal) |

### 4.2 Edge Cases

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| S-E01 | Single document (no pair) | No duplicates found |
| S-E02 | All documents identical | All content flagged as duplicate |
| S-E03 | Documents with no overlap | No duplicates found |
| S-E04 | Very short documents (< min_length) | Skipped gracefully |
| S-E05 | Binary/special characters | No crash, correct byte handling |

---

## Stage 5: Bloom Filter

### 5.1 False Positive Rate Verification

| Test ID | Config (n, p) | Test Procedure | Pass Criteria |
|---------|---------------|----------------|---------------|
| B-F01 | 10K, 0.01 | Insert 10K items, query 10K non-members | Measured FP < 0.02 |
| B-F02 | 100K, 0.001 | Insert 100K items, query 100K non-members | Measured FP < 0.002 |
| B-F03 | 1M, 0.01 | Insert 1M items, query 100K non-members | Measured FP < 0.02 |
| B-F04 | 10K, 0.01 | Insert 20K items (2x capacity) | FP rate increases but < 0.1 |

### 5.2 Correctness

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| B-C01 | Insert item, query same item | Always returns True (no false negatives) |
| B-C02 | Query before any inserts | Always returns False |
| B-C03 | Insert same item twice | Still returns True, no crash |
| B-C04 | Insert N items, query all N | All return True |
| B-C05 | Hash function distribution | Bits uniformly distributed |

---

## Stage 6: Leakage Guard

### 6.1 Exact Holdout Detection

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| L-E01 | Verbatim holdout sample in training | Detected, match_type="exact" |
| L-E02 | Holdout sample with extra whitespace (pre-canon) | Detected after canonicalization |
| L-E03 | Holdout sample with different casing (pre-canon) | Detected after canonicalization |
| L-E04 | Non-holdout sample | Not flagged |
| L-E05 | Empty holdout set | No errors, nothing flagged |

### 6.2 Fuzzy Holdout Detection

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| L-F01 | Holdout with 10% words replaced | Detected, match_type="fuzzy" |
| L-F02 | Holdout with 30% words replaced | Detected if J > threshold |
| L-F03 | Holdout with 50% words replaced | Not detected (below threshold) |
| L-F04 | Holdout with sentences reordered | Detected (shingle overlap) |
| L-F05 | Completely unrelated text | Not flagged |

### 6.3 N-gram Holdout Detection

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| L-N01 | Training doc contains 100% of holdout 13-grams | Detected, overlap=1.0 |
| L-N02 | Training doc contains 90% of holdout 13-grams | Detected, overlap=0.9 |
| L-N03 | Training doc contains 50% of holdout 13-grams | Not flagged (below 0.8) |
| L-N04 | Training doc much longer than holdout | Detected if holdout embedded |
| L-N05 | Very short holdout (< 13 words) | Graceful handling (skip n-gram) |

### 6.4 Multi-Holdout

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| L-M01 | Sample matches holdout A but not B | Reports only match with A |
| L-M02 | Sample matches both holdout A and B | Reports matches with both |
| L-M03 | Three holdout sets, contamination in one | Correct per-holdout counts |

---

## Stage 7: Pipeline End-to-End

### 7.1 Full Pipeline

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| P-E01 | Synthetic corpus with known dups + leakage | All known dups removed, all leakage flagged |
| P-E02 | Clean corpus (no dups, no leakage) | No documents removed or flagged |
| P-E03 | All documents identical | N-1 removed, 1 kept |
| P-E04 | Pipeline with all stages disabled | Passthrough, no modifications |
| P-E05 | Pipeline produces valid JSON report | Report passes schema validation |

### 7.2 Report Validation

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| P-R01 | Report total_input count | Matches actual input count |
| P-R02 | Report exact_duplicates count | Matches known duplicate count |
| P-R03 | Report fuzzy_duplicates count | Within 10% of expected |
| P-R04 | Report leakage_flagged count | Matches known contamination count |
| P-R05 | Report timing fields | All > 0, sum ≈ total time |
| P-R06 | Report removed_ids | All IDs present, no extras |

### 7.3 Ordering and Determinism

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| P-O01 | Run pipeline twice, same config + data | Identical output |
| P-O02 | Shuffle input order | Same documents removed (different IDs OK) |
| P-O03 | Stages execute in correct order | Canon before hash, hash before MinHash, etc. |

---

## Test Data Generation

### Synthetic Corpus Generator

```python
def generate_test_corpus(
    n_unique: int = 100,
    n_exact_dups: int = 10,
    n_near_dups: int = 10,
    n_leakage: int = 5,
    doc_length: int = 200,
    seed: int = 42,
) -> dict:
    """Generate a synthetic test corpus with known duplicates and leakage.

    Returns:
        {
            "corpus": [...],
            "holdout": [...],
            "expected_exact_dups": [...],
            "expected_near_dups": [...],
            "expected_leakage": [...],
        }
    """
    ...
```

### Test Fixture Files

For integration tests, maintain static fixture files:
- `fixtures/clean_corpus.jsonl` -- 100 unique documents
- `fixtures/dup_corpus.jsonl` -- 100 documents with 10 exact, 10 near-duplicate
- `fixtures/holdout_set.jsonl` -- 20 holdout documents
- `fixtures/contaminated_corpus.jsonl` -- training corpus with 5 holdout leaks
- `fixtures/expected_report.json` -- expected audit trail for dup_corpus

---

## Coverage Matrix Summary

| Stage | Test Count | Categories |
|-------|-----------|------------|
| 1. Canonicalization | 23 | Unicode, whitespace, URL/email, idempotency |
| 2. Exact Dedup | 13 | Collision, known dups, Bloom pre-filter |
| 3. MinHash | 15 | Threshold, injection, backend, FP rate |
| 4. Substring | 10 | Overlapping passages, edge cases |
| 5. Bloom Filter | 9 | FP rate verification, correctness |
| 6. Leakage Guard | 17 | Exact, fuzzy, n-gram, multi-holdout |
| 7. Pipeline E2E | 11 | Full pipeline, report, determinism |
| **Total** | **98** | |
