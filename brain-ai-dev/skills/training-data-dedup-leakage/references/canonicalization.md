# Canonicalization Reference

## Purpose

Canonicalization transforms text into a stable, normalized form so that semantically
identical documents with superficial formatting differences produce the same representation.
This is the critical first stage of any dedup pipeline -- without it, trivial differences
(extra spaces, Unicode variants, casing) cause false negatives in downstream dedup stages.

Design principle: **every downstream comparison operates on canonical form only.**

---

## Unicode NFKC Normalization

### Why NFKC

Unicode defines four normalization forms: NFC, NFD, NFKC, NFKD. We use **NFKC**
(Compatibility Composition) because it:

1. Decomposes compatibility characters (e.g., ligatures, width variants) then recomposes
2. Maps visually similar characters to the same codepoint
3. Is the standard recommended by WHATWG for text comparison use cases

### Examples

| Raw Input | NFKC Output | Explanation |
|-----------|-------------|-------------|
| `\ufb01` (fi ligature) | `fi` | Compatibility decomposition |
| `\u2126` (Ohm sign) | `\u03A9` (Greek Omega) | Canonical equivalence |
| `\uff21` (fullwidth A) | `A` | Width normalization |
| `\u00e9` (e-acute precomposed) | `\u00e9` | Already composed, no change |
| `e\u0301` (e + combining acute) | `\u00e9` | Recomposition |
| `\u2160` (Roman numeral I) | `I` | Compatibility mapping |
| `\u00B2` (superscript 2) | `2` | Compatibility decomposition |

### Python Implementation

```python
import unicodedata

def normalize_unicode(text: str) -> str:
    """Apply NFKC normalization to text.

    NFKC first applies compatibility decomposition (NFKD) then canonical
    composition (NFC). This maps compatibility characters to their canonical
    equivalents and recomposes combining sequences.
    """
    return unicodedata.normalize("NFKC", text)
```

### Edge Cases

- **Emoji**: NFKC generally leaves modern emoji untouched, but some older emoji
  sequences may change. If preserving emoji is important, consider flagging emoji
  spans and skipping normalization on them.
- **CJK Compatibility Ideographs**: NFKC maps CJK compatibility ideographs to
  their unified forms. This is usually desirable for dedup.
- **Mathematical Symbols**: `\u2102` (double-struck C) maps to `C`, which may
  lose domain-specific meaning. Accept this for dedup purposes.

---

## Whitespace Collapsing

### Standard Algorithm

1. Replace all Unicode whitespace characters (not just ASCII space) with a single space
2. Strip leading and trailing whitespace
3. Collapse runs of multiple spaces into one

### Unicode Whitespace Characters

The following characters are treated as whitespace and collapsed:

| Codepoint | Name |
|-----------|------|
| `U+0009` | Tab |
| `U+000A` | Line Feed |
| `U+000B` | Vertical Tab |
| `U+000C` | Form Feed |
| `U+000D` | Carriage Return |
| `U+0020` | Space |
| `U+0085` | Next Line |
| `U+00A0` | No-Break Space |
| `U+1680` | Ogham Space Mark |
| `U+2000`-`U+200A` | Various typographic spaces |
| `U+2028` | Line Separator |
| `U+2029` | Paragraph Separator |
| `U+202F` | Narrow No-Break Space |
| `U+205F` | Medium Mathematical Space |
| `U+3000` | Ideographic Space |

### Implementation

```python
import re

# Match any Unicode whitespace character(s)
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)

def collapse_whitespace(text: str) -> str:
    """Replace all whitespace sequences (including Unicode) with a single space."""
    return _WHITESPACE_RE.sub(" ", text).strip()
```

---

## URL and Email Stripping

### URL Regex Pattern

```python
import re

# Matches http(s) and ftp URLs, including those with paths, query strings, fragments
_URL_RE = re.compile(
    r"https?://[^\s<>\"']+|ftp://[^\s<>\"']+|www\.[^\s<>\"']+",
    re.IGNORECASE,
)

def strip_urls(text: str) -> str:
    """Remove URLs from text, replacing with a single space."""
    return _URL_RE.sub(" ", text)
```

### Email Regex Pattern

```python
# RFC 5322 simplified -- covers the vast majority of real-world email addresses
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
)

def strip_emails(text: str) -> str:
    """Remove email addresses from text, replacing with a single space."""
    return _EMAIL_RE.sub(" ", text)
```

### Design Notes

- **Replace with space, not empty string**: Prevents accidental word concatenation.
  `"visit https://example.com today"` becomes `"visit today"`, not `"visit today"`.
- **Order of operations**: Strip URLs and emails *before* whitespace collapsing so
  the replacement spaces get collapsed in the next step.
- **False positives**: The URL regex may match things like version strings
  (`http://1.2.3`). This is acceptable for dedup purposes -- we want stability,
  not perfect URL detection.

---

## Boilerplate Removal Heuristics

### Common Boilerplate Patterns

Web scrapes and document corpora often contain repeated structural elements that
inflate false-positive duplicate counts or mask true content similarity.

#### Navigation / Chrome

```python
_BOILERPLATE_PATTERNS = [
    # Cookie consent banners
    re.compile(r"(?i)we use cookies.*?(?:accept|dismiss|learn more)"),
    # Navigation breadcrumbs
    re.compile(r"(?i)^(home\s*[>/|].*){3,}$", re.MULTILINE),
    # Share buttons
    re.compile(r"(?i)share\s+(on\s+)?(twitter|facebook|linkedin|reddit)"),
    # Common footers
    re.compile(r"(?i)all rights reserved\.?\s*$", re.MULTILINE),
    re.compile(r"(?i)copyright\s+(\(c\)|©)\s*\d{4}", re.MULTILINE),
]
```

#### License Headers

For code corpora, strip standard license headers (Apache 2.0, MIT, GPL preamble):

```python
_LICENSE_HEADER_RE = re.compile(
    r"(?i)^(#|//|/\*|\*)\s*(licensed under|copyright|permission is hereby granted).*?$",
    re.MULTILINE,
)
```

#### Strategy

1. **Frequency-based**: Hash each line across the corpus. Lines appearing in > N%
   of documents are likely boilerplate. Remove them. Typical threshold: 1-5%.
2. **Pattern-based**: Apply regex patterns for known boilerplate structures.
3. **Positional**: The first and last K lines of web-scraped documents are often
   navigation/footer. Consider trimming or weighting them lower.

### Implementation Notes

- Boilerplate removal is **optional** and should be configurable. Some use cases
  (legal document analysis) need to preserve copyright notices.
- Always apply boilerplate removal *before* content hashing to avoid hash
  instability from boilerplate variations.

---

## Idempotency Testing

### Definition

Canonicalization must be **idempotent**: applying it twice produces the same result
as applying it once. Formally: `canon(canon(text)) == canon(text)` for all inputs.

This is a critical invariant. Non-idempotent canonicalization creates subtle bugs
where the same document produces different hashes depending on how many times it
passed through the pipeline.

### Test Approach

```python
def test_canonicalization_idempotent(canonicalizer, texts):
    """Verify that canonicalization is idempotent on a diverse test set."""
    for text in texts:
        once = canonicalizer(text)
        twice = canonicalizer(once)
        assert once == twice, (
            f"Canonicalization is not idempotent.\n"
            f"  Input:  {text!r}\n"
            f"  Once:   {once!r}\n"
            f"  Twice:  {twice!r}"
        )
```

### Test Corpus Design

The test set should cover:

1. **Unicode edge cases**: Combining characters, ligatures, fullwidth forms,
   mathematical symbols, CJK compatibility ideographs, RTL markers
2. **Whitespace variants**: Tabs, no-break spaces, zero-width spaces, mixed newlines
   (CRLF, LF, CR), ideographic spaces
3. **Empty/minimal inputs**: Empty string, single character, whitespace-only
4. **URLs and emails**: Interleaved with normal text, at boundaries
5. **Already-canonical text**: Should pass through unchanged
6. **Multi-language**: Latin, Cyrillic, Arabic, CJK, Devanagari, Thai
7. **Control characters**: Null bytes, BEL, ESC, DELETE
8. **Very long strings**: 10MB+ to catch performance regressions

### Property-Based Testing

Use `hypothesis` for property-based idempotency verification:

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=0, max_size=10000))
def test_canon_idempotent_property(text):
    canon = Canonicalizer(CanonConfig())
    result = canon(text)
    assert canon(result) == result
```

### Stability Across Versions

Pin the Python version and `unicodedata` database version in your environment.
Unicode normalization results can change between Unicode standard updates (rare
but possible). Document the Unicode version used:

```python
import unicodedata
print(f"Unicode version: {unicodedata.unidata_version}")
# e.g., "15.1.0"
```

---

## Full Canonicalization Pipeline

The recommended order of operations:

```
text
  -> strip control characters (except whitespace)
  -> NFKC normalize
  -> lowercase
  -> strip URLs (optional)
  -> strip emails (optional)
  -> remove boilerplate (optional)
  -> collapse whitespace
  -> strip leading/trailing whitespace
  -> canonical text
```

**Order matters**:
- NFKC before lowercase (NFKC may introduce case-sensitive characters)
- URL/email stripping before whitespace collapse (replacements create spaces)
- Whitespace collapse last (all previous steps may introduce whitespace artifacts)

---

## Performance Considerations

| Corpus Size | Strategy |
|-------------|----------|
| < 1M docs | Single-threaded Python, `unicodedata.normalize` |
| 1M - 100M | `multiprocessing.Pool` with chunked batches |
| 100M+ | Distribute with Ray/Dask, or use Rust `unicode-normalization` crate via PyO3 |

Canonicalization is typically I/O-bound when reading from disk and CPU-bound on
normalization. Profile before optimizing.
