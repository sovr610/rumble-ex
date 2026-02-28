# Tokenizer Compression Reference

## Table of Contents

1. [Overview](#1-overview)
2. [Normalization Recipes](#2-normalization-recipes)
3. [Equivalence Class Construction](#3-equivalence-class-construction)
4. [Special Token Policy](#4-special-token-policy)
5. [Fast Path Runtime](#5-fast-path-runtime)
6. [Serialization Format](#6-serialization-format)
7. [Determinism Guarantees](#7-determinism-guarantees)
8. [Compression Statistics](#8-compression-statistics)
9. [Integration with N-gram Pipeline](#9-integration-with-n-gram-pipeline)
10. [Appendix: Troubleshooting](#appendix-a-troubleshooting)

---

## 1. Overview

### Purpose

Tokenizer compression is a surjective mapping that collapses textually equivalent
tokens into canonical IDs. Its role in the Engram subsystem is to increase N-gram
space density: when multiple token IDs that decode to the same (or equivalent) text
are unified under a single canonical ID, the downstream N-gram hash tables see fewer
distinct tokens. This means more N-gram collisions are "true" collisions (same semantic
content) rather than surface-variant noise, and the hash table space is utilized more
efficiently.

### Core Idea

Given a tokenizer vocabulary `V` of size `|V|`, build a mapping
`P : V -> V'` where `|V'| < |V|`. Tokens `a` and `b` map to the same canonical ID
if and only if `normalize(decode(a)) == normalize(decode(b))`.

```
                    decode          normalize         re-encode
   token_id  --->  "Hello"   --->  "hello"    --->  canonical_id
   token_id  --->  " Hello"  --->  "hello"    --->  same canonical_id
   token_id  --->  "HELLO"   --->  "hello"    --->  same canonical_id
```

The mapping `P` is stored as a flat `int32` lookup table of size `|V|`. At runtime,
compression is a single gather operation with no string processing.

### Empirical Compression

The Engram paper reports approximately 23% effective vocabulary reduction on a
128k-token BPE tokenizer. This means:

| Metric | Value |
|---|---|
| Original vocabulary size | 128,000 |
| Compressed vocabulary size | ~98,560 |
| Equivalence classes created | ~29,440 classes merge 2+ tokens |
| Singleton classes | ~69,120 (tokens that only map to themselves) |
| Compression ratio | ~0.77 |

These numbers vary by tokenizer. BPE tokenizers (GPT-style) and SentencePiece
tokenizers (LLaMA-style) both exhibit similar compression ratios because both
produce surface variants for case, whitespace, and Unicode normalization forms.

### Properties

| Property | Guarantee |
|---|---|
| Surjective | Every canonical ID has at least one original token mapping to it |
| Deterministic | Same (tokenizer, recipe, seed, policy) always produces the same table |
| Idempotent | Applying compression twice gives the same result as once: `P[P[x]] == P[x]` for canonical IDs |
| Special-token invariant | Special tokens always map to themselves |
| Platform-independent | Pure Python + numpy build step; no system locale dependency |

### Data Flow Position

```
raw_input_ids (B, T)
       |
       v
 [TokenizerCompression]   <-- table lookup, O(1) per token
       |
       v
canonical_ids (B, T)       <-- same shape, reduced cardinality
       |
       v
 [N-gram Extraction]       <-- suffix windows of canonical_ids
       |
       v
 [Multi-Head Hashing]      <-- deterministic hash to embedding rows
       |
       v
 [Embedding Retrieval]     <-- gather from hash tables
```

Compression is always the first transformation applied to input token IDs. It
occurs before any N-gram extraction, hashing, or embedding operations.

---

## 2. Normalization Recipes

### Overview

A normalization recipe is an ordered list of text transformations applied to the
decoded text of each token. The recipe determines which tokens are considered
equivalent. The order of operations matters because some transformations produce
different results depending on their input.

### Available Normalization Steps

#### 2.1 NFKC Unicode Normalization

NFKC (Normalization Form Compatibility Composition) decomposes characters by
compatibility and then recomposes by canonical equivalence. This collapses many
Unicode variants into a single form.

```python
import unicodedata

def nfkc_normalize(text: str) -> str:
    """Apply NFKC Unicode normalization."""
    return unicodedata.normalize('NFKC', text)
```

Examples of NFKC normalization:

| Input | Output | Reason |
|---|---|---|
| `"\uff21"` (fullwidth A) | `"A"` | Compatibility decomposition |
| `"\u2126"` (ohm sign) | `"\u03a9"` (omega) | Canonical equivalence |
| `"\ufb01"` (fi ligature) | `"fi"` | Compatibility decomposition |
| `"\u00e9"` (e-acute) | `"\u00e9"` | Already in NFC; unchanged |
| `"\u0065\u0301"` (e + combining acute) | `"\u00e9"` | Canonical composition |
| `"\u2160"` (Roman numeral I) | `"I"` | Compatibility decomposition |
| `"\u00bd"` (vulgar fraction 1/2) | `"1\u20442"` | Compatibility decomposition |

NFKC is the most aggressive standard Unicode normalization form. It must be
applied first because subsequent steps (like case folding) may behave differently
depending on the Unicode normalization form of the input.

#### 2.2 Case Folding (Lowercasing)

Case folding converts all cased characters to their lowercase equivalents. This
uses Python's `str.lower()` which is Unicode-aware and handles locale-independent
case folding.

```python
def case_fold(text: str) -> str:
    """Convert to lowercase (Unicode-aware case folding)."""
    return text.lower()
```

Examples:

| Input | Output |
|---|---|
| `"Hello"` | `"hello"` |
| `"WORLD"` | `"world"` |
| `"\u00c9"` (E-acute) | `"\u00e9"` (e-acute) |
| `"\u0130"` (I with dot above) | `"i\u0307"` (i + combining dot) |
| `"123"` | `"123"` (unchanged) |

Important: Case folding must come after NFKC normalization. Applying NFKC after
case folding can produce different results for certain characters (e.g., characters
that decompose differently depending on case).

#### 2.3 Whitespace Normalization

Whitespace normalization strips leading and trailing whitespace and collapses
internal runs of whitespace to a single space character.

```python
def normalize_whitespace(text: str) -> str:
    """Strip and collapse whitespace."""
    return ' '.join(text.split())
```

The `str.split()` method without arguments splits on any Unicode whitespace character
(spaces, tabs, newlines, non-breaking spaces, etc.) and removes empty strings from
the result. Joining with a single space produces normalized output.

Examples:

| Input | Output |
|---|---|
| `"  hello  "` | `"hello"` |
| `"hello   world"` | `"hello world"` |
| `"\thello\nworld"` | `"hello world"` |
| `"\u00a0hello"` (nbsp) | `"hello"` |
| `""` | `""` |
| `"   "` | `""` |

BPE tokenizers frequently produce tokens with leading whitespace (e.g., ` hello`
vs `hello`). Whitespace normalization is the primary driver of equivalence classes
for these tokenizers.

#### 2.4 Accent/Diacritic Stripping (Optional)

Accent stripping decomposes characters to NFD form and removes combining marks
(Unicode category `Mn` - Mark, Nonspacing). This is an aggressive normalization
that loses information and should only be enabled when the downstream task does
not need accent distinction.

```python
def strip_accents(text: str) -> str:
    """Remove diacritical marks (combining characters)."""
    # Decompose to NFD so accents become separate combining characters
    nfd = unicodedata.normalize('NFD', text)
    # Remove combining marks
    stripped = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    # Re-compose to NFC for consistency
    return unicodedata.normalize('NFC', stripped)
```

Examples:

| Input | Output |
|---|---|
| `"\u00e9"` (e-acute) | `"e"` |
| `"\u00f1"` (n-tilde) | `"n"` |
| `"\u00fc"` (u-umlaut) | `"u"` |
| `"na\u00efve"` (naive with diaeresis) | `"naive"` |
| `"hello"` | `"hello"` (unchanged) |

Warning: Accent stripping can collapse semantically distinct words in some languages.
For example, in French, "ou" (or) and "o\u00f9" (where) become identical. Enable
only when the application tolerates this loss.

If accent stripping is enabled, it must come after NFKC normalization (which may
compose characters) and before or after case folding (order between these two does
not matter since accents and case are orthogonal).

### Recipe Configuration

A recipe is specified as an ordered list of step names:

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class NormalizationRecipe:
    """Ordered list of normalization steps for tokenizer compression."""
    steps: List[str] = field(default_factory=lambda: [
        "nfkc",
        "case_fold",
        "whitespace",
    ])
```

Valid step names and their corresponding functions:

| Step Name | Function | Lossy? | Default? |
|---|---|---|---|
| `"nfkc"` | NFKC Unicode normalization | No (for most text) | Yes |
| `"case_fold"` | `str.lower()` | Yes (loses case) | Yes |
| `"whitespace"` | Strip + collapse whitespace | Yes (loses whitespace patterns) | Yes |
| `"strip_accents"` | Remove combining marks | Yes (loses diacritics) | No |

### Order of Operations

The canonical order is:

```
1. nfkc           -- normalize Unicode forms first
2. case_fold      -- lowercase after Unicode normalization
3. whitespace     -- collapse whitespace after casing
4. strip_accents  -- (optional) remove diacritics last
```

This order matters for correctness:

- NFKC before case fold: NFKC may decompose characters that case fold differently
  in composed vs decomposed form.
- NFKC before whitespace: NFKC normalizes certain whitespace-like characters
  (e.g., non-breaking spaces) that `str.split()` should then handle.
- Case fold before strip accents: Order between these is less critical but we
  standardize on case-first for reproducibility.

### Applying the Recipe

```python
NORMALIZERS = {
    "nfkc": lambda text: unicodedata.normalize('NFKC', text),
    "case_fold": lambda text: text.lower(),
    "whitespace": lambda text: ' '.join(text.split()),
    "strip_accents": strip_accents,  # function defined above
}

def apply_recipe(text: str, recipe: NormalizationRecipe) -> str:
    """Apply normalization recipe to text, step by step."""
    for step_name in recipe.steps:
        if step_name not in NORMALIZERS:
            raise ValueError(
                f"Unknown normalization step: {step_name!r}. "
                f"Valid steps: {sorted(NORMALIZERS.keys())}"
            )
        text = NORMALIZERS[step_name](text)
    return text
```

### Recipe Validation

At build time, validate: all step names are in the valid set, no duplicates,
at least one step present, and NFKC precedes case_fold when both are present.
Raise `ValueError` on any violation.

---

## 3. Equivalence Class Construction

### Overview

Equivalence class construction is the build-time process that creates the surjective
mapping. For each token in the vocabulary, we decode it to text, apply the
normalization recipe, and group tokens whose normalized text is identical. Each
group is an equivalence class, and one member is selected as the canonical ID.

### Algorithm

```
Input:
  - tokenizer: object with get_vocab() -> Dict[str, int] and decode(id) -> str
  - recipe: NormalizationRecipe
  - special_token_ids: Set[int]
  - canonical_policy: "lowest_id" | "most_frequent"

Output:
  - lookup_table: int32 array of shape (vocab_size,)
  - equivalence_classes: Dict[int, List[int]]  (canonical_id -> [member_ids])

Algorithm:
  1. For each (token_text, token_id) in sorted(tokenizer.get_vocab().items()):
       a. If token_id in special_token_ids: skip (handled separately)
       b. decoded = tokenizer.decode([token_id])
       c. normalized = apply_recipe(decoded, recipe)
       d. Append (normalized, token_id) to groups[normalized]

  2. For each normalized_text in sorted(groups.keys()):
       a. member_ids = groups[normalized_text]
       b. canonical = select_canonical(member_ids, canonical_policy)
       c. For each id in member_ids:
            lookup_table[id] = canonical
       d. equivalence_classes[canonical] = member_ids

  3. For each special_id in special_token_ids:
       lookup_table[special_id] = special_id

  4. Return lookup_table, equivalence_classes
```

### Detailed Implementation

```python
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

def build_equivalence_classes(
    tokenizer,
    recipe: NormalizationRecipe,
    special_token_ids: Set[int],
    canonical_policy: str = "lowest_id",
    frequency_counts: Optional[Dict[int, int]] = None,
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Build the surjective mapping from original token IDs to canonical IDs.

    Args:
        tokenizer: Tokenizer with get_vocab() and decode() methods.
        recipe: Normalization recipe to apply.
        special_token_ids: Token IDs that must map to themselves.
        canonical_policy: "lowest_id" or "most_frequent".
        frequency_counts: Token frequency counts (required if policy is "most_frequent").

    Returns:
        lookup_table: int32 numpy array of shape (vocab_size,).
        equivalence_classes: Dict mapping canonical_id -> list of member IDs.
    """
    vocab = tokenizer.get_vocab()  # Dict[str, int]
    vocab_size = max(vocab.values()) + 1

    # Initialize lookup table as identity mapping
    lookup_table = np.arange(vocab_size, dtype=np.int32)

    # Group tokens by normalized text
    # Key: normalized text, Value: list of token IDs
    groups: Dict[str, List[int]] = defaultdict(list)

    for token_text, token_id in sorted(vocab.items(), key=lambda x: x[1]):
        # Skip special tokens -- they are handled separately
        if token_id in special_token_ids:
            continue

        # Decode the token to its text representation
        # Some tokenizers require special handling for decode
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            # If decode fails, treat as singleton (maps to self)
            decoded = token_text

        # Apply normalization recipe
        normalized = apply_recipe(decoded, recipe)

        # Group by normalized text
        groups[normalized].append(token_id)

    # Build equivalence classes
    equivalence_classes: Dict[int, List[int]] = {}

    # Sort groups by their normalized text for deterministic iteration
    for normalized_text in sorted(groups.keys()):
        member_ids = sorted(groups[normalized_text])  # sort for determinism

        if len(member_ids) == 0:
            continue

        # Select canonical ID
        canonical = _select_canonical(
            member_ids, canonical_policy, frequency_counts
        )

        # Map all members to canonical
        for mid in member_ids:
            lookup_table[mid] = canonical

        equivalence_classes[canonical] = member_ids

    # Enforce special token invariance
    for special_id in special_token_ids:
        if special_id < vocab_size:
            lookup_table[special_id] = special_id

    return lookup_table, equivalence_classes


def _select_canonical(
    member_ids: List[int],
    policy: str,
    frequency_counts: Optional[Dict[int, int]],
) -> int:
    """
    Select the canonical ID from an equivalence class.

    Args:
        member_ids: Sorted list of token IDs in the class.
        policy: "lowest_id" or "most_frequent".
        frequency_counts: Token frequency counts.

    Returns:
        The canonical token ID.
    """
    if policy == "lowest_id":
        return member_ids[0]  # Already sorted, first is lowest

    elif policy == "most_frequent":
        if frequency_counts is None:
            raise ValueError(
                "frequency_counts required for 'most_frequent' policy"
            )
        # Select the member with highest frequency; break ties by lowest ID
        return max(
            member_ids,
            key=lambda mid: (frequency_counts.get(mid, 0), -mid),
        )

    else:
        raise ValueError(f"Unknown canonical policy: {policy!r}")
```

### Canonical ID Selection Policies

| Policy | Description | When to Use |
|---|---|---|
| `"lowest_id"` | Select the member with the smallest token ID | Default; deterministic, simple |
| `"most_frequent"` | Select the member that appears most often in a reference corpus | When downstream models have token-frequency-dependent biases |

The `"lowest_id"` policy is recommended as the default because:

1. It is deterministic without requiring external data (no corpus needed).
2. It tends to select the "base" form of a token (lower IDs often correspond to
   more common token forms in BPE vocabularies).
3. It is simpler to reproduce across environments.

### Handling Multi-Token Normalizations

Some tokens do not round-trip cleanly through decode-normalize. Problematic cases:

| Case | Description | Handling |
|---|---|---|
| Decode produces multiple tokens | `decode([id])` may produce text that encodes to multiple tokens | Group by normalized text, not by re-encoding |
| Empty decode | Some token IDs decode to empty string | Treat as singleton (map to self) |
| Control characters | Tokens representing control characters | Treat as singleton unless they normalize to identical text |
| Byte-level tokens | `<0x41>` style byte tokens in LLaMA tokenizers | Decode to actual byte, normalize, group if equivalent |

For safe decoding, wrap `tokenizer.decode([token_id])` in a try/except with
the raw `token_text` from `get_vocab()` as fallback. If decode produces an
empty or whitespace-only string, also fall back to `token_text`.

### Idempotency Invariant

After building the lookup table, the following must hold for all canonical IDs:
for any canonical ID `c` in `range(lookup_table)`, `lookup_table[c] == c`.
This ensures `P[P[x]] == P[x]` -- applying compression twice gives the same
result as once, which is important for cache reconstruction.

The invariant holds naturally because the canonical ID of each class is a member
of the class: `lookup_table[canonical] = canonical` is set during construction.

### Equivalence Class Size Distribution

In a typical 128k BPE tokenizer with the default recipe (NFKC + case fold + whitespace):

| Class Size | Approximate Count | Description |
|---|---|---|
| 1 (singleton) | ~70% of classes | Token has no equivalent variants |
| 2 | ~20% of classes | One variant (usually case: `"Hello"` / `"hello"`) |
| 3 | ~7% of classes | Two variants (case + whitespace: `" Hello"` / `"hello"` / `"Hello"`) |
| 4-5 | ~2.5% of classes | Multiple variants (case + whitespace + Unicode) |
| 6+ | ~0.5% of classes | Rare; usually Unicode normalization edge cases |

The distribution is heavily skewed toward singletons. Most of the vocabulary
reduction comes from the relatively small number of classes with size >= 2, but
these classes cover high-frequency tokens (words that appear with and without
leading whitespace, capitalized and lowercase, etc.).

---

## 4. Special Token Policy

### Overview

Special tokens (padding, beginning-of-sequence, end-of-sequence, unknown, etc.)
must never be merged with other tokens. They serve structural roles in the input
that must be preserved exactly. The special token policy explicitly defines which
tokens are protected from equivalence merging.

### Default Special Tokens

| Token Name | Typical String | Role |
|---|---|---|
| `pad` | `<pad>`, `[PAD]` | Padding for batch alignment |
| `bos` | `<s>`, `[CLS]` | Beginning of sequence |
| `eos` | `</s>`, `[SEP]` | End of sequence |
| `unk` | `<unk>`, `[UNK]` | Unknown / out-of-vocabulary |

### Configurable Additional Special Tokens

| Token Name | Typical String | When to Include |
|---|---|---|
| `sep` | `[SEP]`, `</s>` | Sentence pair tasks |
| `cls` | `[CLS]`, `<s>` | Classification tasks |
| `mask` | `[MASK]`, `<mask>` | Masked language modeling |
| custom | any | User-defined special tokens |

### Policy Data Structure

```python
@dataclass
class SpecialTokenPolicy:
    """Special tokens are invariant under compression (map to themselves)."""
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    unk_token_id: Optional[int] = None
    additional_special_ids: List[int] = field(default_factory=list)

    def all_special_ids(self) -> Set[int]:
        ids = set()
        for attr in ['pad_token_id', 'bos_token_id',
                      'eos_token_id', 'unk_token_id']:
            val = getattr(self, attr)
            if val is not None:
                ids.add(val)
        ids.update(self.additional_special_ids)
        return ids
```

Validation at build time must check: all IDs in `[0, vocab_size)`, no duplicates
among named slots, and after table construction: `lookup_table[sid] == sid` for
every `sid` in `all_special_ids()`.

### Auto-Detection from Tokenizer

When a HuggingFace tokenizer object is available, special token IDs can be
auto-detected by reading `pad_token_id`, `bos_token_id`, `eos_token_id`,
`unk_token_id`, and `additional_special_tokens_ids` attributes. For tokenizers
using `all_special_ids`, subtract the named IDs and use the remainder as
additional special tokens.

### Interaction with Equivalence Classes

During equivalence class construction, special token IDs are excluded from
grouping entirely. This prevents the following failure mode:

```
# BAD: Without special token protection
token 0 ("<pad>") normalizes to ""
token 5000 ("") normalizes to ""
-> Both in same equivalence class, canonical = 0
-> token 5000 now maps to 0 (pad), breaking its semantics

# GOOD: With special token protection
token 0 ("<pad>") is protected, skipped during grouping
token 5000 ("") normalizes to "", forms singleton class
-> token 0 maps to 0 (invariant)
-> token 5000 maps to 5000 (singleton)
```

---

## 5. Fast Path Runtime

### Overview

At inference and training time, tokenizer compression is a single table lookup
on an integer tensor. There are no string operations at runtime. The lookup table
is a pre-built `int32` array stored as a tensor attribute.

### Core Operation

```python
canonical_ids = lookup_table[input_ids]
```

This is an integer gather operation. Both PyTorch and NumPy support this as
advanced indexing.

### PyTorch Implementation

```python
import torch
from typing import Union

class TokenizerCompressor:
    """
    Runtime tokenizer compression via table lookup.

    This class holds the pre-built lookup table and provides the
    compress_ids() method for fast runtime compression.
    """

    def __init__(self, lookup_table: Union[np.ndarray, torch.Tensor]):
        """
        Args:
            lookup_table: int32 array of shape (vocab_size,).
                          lookup_table[i] = canonical ID for token i.
        """
        if isinstance(lookup_table, np.ndarray):
            self._table = torch.from_numpy(lookup_table).to(torch.int32)
        elif isinstance(lookup_table, torch.Tensor):
            self._table = lookup_table.to(torch.int32)
        else:
            raise TypeError(
                f"Expected np.ndarray or torch.Tensor, got {type(lookup_table)}"
            )

        self._vocab_size = self._table.shape[0]

    def compress_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compress token IDs to canonical IDs via table lookup.

        Args:
            input_ids: Integer tensor of any shape. Values must be in
                       [0, vocab_size).

        Returns:
            Tensor of same shape and device as input_ids, with canonical IDs.
            dtype: torch.int32 (or torch.long depending on platform).
        """
        # Move table to same device as input (cached after first call)
        table = self._table.to(input_ids.device)

        # Clamp to valid range (safety against out-of-range IDs)
        clamped = input_ids.clamp(0, self._vocab_size - 1)

        # Table lookup: advanced indexing
        canonical = table[clamped.long()]

        return canonical
```

### NumPy Implementation

```python
class TokenizerCompressorNumpy:
    """NumPy variant for CPU-only pipelines (e.g., data preprocessing)."""

    def __init__(self, lookup_table: np.ndarray):
        self._table = lookup_table.astype(np.int32)
        self._vocab_size = self._table.shape[0]

    def compress_ids(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Compress token IDs to canonical IDs.

        Args:
            input_ids: Integer array of any shape.

        Returns:
            Array of same shape with canonical IDs (int32).
        """
        clamped = np.clip(input_ids, 0, self._vocab_size - 1)
        return self._table[clamped]
```

### Batch Compatibility

The table lookup operation works on tensors of any shape:

| Input Shape | Output Shape | Use Case |
|---|---|---|
| `(B, T)` | `(B, T)` | Standard batched sequences |
| `(T,)` | `(T,)` | Single sequence |
| `(B, T, K)` | `(B, T, K)` | Pre-extracted N-gram windows |
| `(N,)` | `(N,)` | Flat token list |

No reshape or special handling is needed. Advanced indexing broadcasts naturally.

### Device Handling

The lookup table must reside on the same device as the input tensor. Two strategies:

**Strategy 1: Lazy transfer (recommended for single-device)**

```python
# Table stays on CPU until first use, then cached on device
if self._table.device != input_ids.device:
    self._table = self._table.to(input_ids.device)
```

**Strategy 2: Registered buffer (recommended for nn.Module integration)**

Use `self.register_buffer('lookup_table', table, persistent=True)`. The buffer
moves with `model.to(device)`, is saved in `state_dict()`, and is always on the
correct device without manual transfer.

### Performance Characteristics

| Operation | Time Complexity | Memory |
|---|---|---|
| Build lookup table | O(V) where V = vocab size | O(V) int32 = 4V bytes |
| Runtime compression | O(B * T) | O(B * T) int32 for output |
| Device transfer | O(V) one-time | Cached after first transfer |

For a 128k vocabulary:
- Lookup table size: 128,000 * 4 bytes = 512 KB
- Runtime for batch (32, 2048): 65,536 lookups, sub-millisecond on GPU

The runtime cost is negligible compared to any neural network operation. It is
never a bottleneck.

### AMP Compatibility

Table lookup operates on integer tensors only. It is inherently AMP-safe because:

1. No floating-point operations are involved.
2. The output is `int32` / `int64`, not a floating-point type.
3. No gradients flow through the compression step (it is not differentiable).

The compression step should be placed outside any `torch.autocast` context, though
it will work correctly inside one (autocast does not affect integer operations).

---

## 6. Serialization Format

### Overview

The tokenizer compression artifact must be serializable to disk and loadable
without access to the original tokenizer. This enables:

1. Building compression once and reusing across training runs.
2. Shipping compression tables with model checkpoints.
3. Verifying that a loaded table matches the expected configuration.

### File Structure

The serialized artifact consists of two files:

```
compression_artifact/
  lookup_table.npy       # int32 numpy array, shape (vocab_size,)
  metadata.json          # JSON with build parameters and statistics
```

Or as a single file:

```
compression_artifact.npz   # numpy compressed archive with 'lookup_table' and 'metadata'
```

### Lookup Table Format

```python
# Save
np.save(path / "lookup_table.npy", lookup_table)

# Load
lookup_table = np.load(path / "lookup_table.npy")
assert lookup_table.dtype == np.int32
assert lookup_table.ndim == 1
```

| Property | Value |
|---|---|
| dtype | `int32` |
| shape | `(vocab_size,)` |
| size on disk | ~500 KB for 128k vocab (uncompressed) |
| size on disk (compressed) | ~200-300 KB for 128k vocab (npz) |

### Metadata Schema

```json
{
    "version": "1.0",
    "version_hash": "sha256:abcdef1234567890...",
    "tokenizer_name": "meta-llama/Llama-2-7b-hf",
    "vocab_size": 128000,
    "compressed_vocab_size": 98560,
    "num_equivalence_classes": 98560,
    "compression_ratio": 0.77,
    "normalization_recipe": ["nfkc", "case_fold", "whitespace"],
    "canonical_policy": "lowest_id",
    "special_token_policy": {
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "unk_token_id": 0,
        "additional_special_ids": [32000, 32001]
    },
    "seed": 42,
    "build_timestamp": "2026-01-15T10:30:00Z",
    "build_platform": "CPython 3.11.5 / Linux",
    "class_size_distribution": {
        "1": 69120,
        "2": 19712,
        "3": 6886,
        "4": 2458,
        "5": 384
    }
}
```

### Metadata Field Descriptions

| Field | Type | Description |
|---|---|---|
| `version` | string | Serialization format version (currently `"1.0"`) |
| `version_hash` | string | SHA-256 hash for verifying build identity |
| `tokenizer_name` | string | Name or path of the source tokenizer |
| `vocab_size` | int | Original vocabulary size |
| `compressed_vocab_size` | int | Number of unique canonical IDs |
| `num_equivalence_classes` | int | Same as `compressed_vocab_size` |
| `compression_ratio` | float | `compressed_vocab_size / vocab_size` |
| `normalization_recipe` | list[str] | Ordered list of normalization steps |
| `canonical_policy` | string | Policy used to select canonical IDs |
| `special_token_policy` | object | Special token configuration |
| `seed` | int | Determinism seed (used in tie-breaking if needed) |
| `build_timestamp` | string | ISO 8601 timestamp of build time |
| `build_platform` | string | Python version and OS for debugging |
| `class_size_distribution` | object | Map of class_size -> count |

### Version Hash

The version hash uniquely identifies a compression table based on its inputs.
Two tables with the same version hash are guaranteed to be identical.

```python
import hashlib
import json

def compute_version_hash(
    tokenizer_name: str,
    normalization_recipe: List[str],
    special_token_ids: Set[int],
    seed: int,
) -> str:
    """
    Compute SHA-256 version hash for a compression table.

    The hash depends only on the inputs that determine the table contents.
    Build timestamp, platform, and statistics are NOT included.

    Args:
        tokenizer_name: Name or path of the tokenizer.
        normalization_recipe: Ordered list of normalization step names.
        special_token_ids: Set of special token IDs.
        seed: Determinism seed.

    Returns:
        Hex-encoded SHA-256 hash string prefixed with "sha256:".
    """
    # Build a deterministic canonical representation
    hash_input = json.dumps({
        "tokenizer_name": tokenizer_name,
        "normalization_recipe": normalization_recipe,
        "special_token_ids": sorted(special_token_ids),
        "seed": seed,
    }, sort_keys=True, separators=(',', ':'))

    h = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    return f"sha256:{h}"
```

### Save / Load Implementation

**Save** (compressed `.npz` format):

```python
def save_compression(lookup_table, metadata, path):
    metadata_bytes = json.dumps(metadata, indent=2).encode('utf-8')
    np.savez_compressed(
        Path(path).with_suffix('.npz'),
        lookup_table=lookup_table,
        metadata=np.frombuffer(metadata_bytes, dtype=np.uint8),
    )
```

**Load** with optional hash verification:

```python
def load_compression(path, expected_hash=None):
    data = np.load(Path(path).with_suffix('.npz'), allow_pickle=False)
    lookup_table = data['lookup_table']
    metadata = json.loads(data['metadata'].tobytes().decode('utf-8'))

    assert lookup_table.dtype == np.int32
    assert lookup_table.ndim == 1

    if expected_hash is not None:
        actual = metadata.get('version_hash', '')
        if actual != expected_hash:
            raise ValueError(f"Hash mismatch: {expected_hash} vs {actual}")

    return lookup_table, metadata
```

Alternatively, save as separate files: `lookup_table.npy` + `metadata.json`
in a directory. The `.npz` format is preferred for portability.

### Checkpoint Integration

When using `register_buffer` in an `nn.Module`, the lookup table is automatically
included in `state_dict()` and restored by `load_state_dict()`. For standalone
artifact management, use the `save_compression` / `load_compression` functions.

---

## 7. Determinism Guarantees

### Overview

The tokenizer compression build process must be fully deterministic: given the
same inputs, it must always produce the exact same lookup table, byte-for-byte.
This is critical because:

1. Hash IDs computed from canonical IDs must be reproducible across runs.
2. Distributed training requires all ranks to have identical compression tables.
3. Version hashes must match between build time and load time.

### Determinism Contract

```
Given:
  - tokenizer T (identified by name/path)
  - normalization_recipe R
  - special_token_policy S
  - seed s
  - canonical_policy P

Then:
  build(T, R, S, s, P) == build(T, R, S, s, P)

for any:
  - Python version (3.9+)
  - Operating system (Linux, macOS, Windows)
  - Hardware (x86, ARM)
  - Time of execution
  - Locale settings
```

### Sources of Non-Determinism and Mitigations

#### 7.1 Dict Iteration Order

Python dicts are insertion-ordered since 3.7, but the order depends on the
tokenizer's `get_vocab()` implementation. Mitigation: sort all intermediate
data structures.

```python
# WRONG: depends on dict iteration order
for token_text, token_id in tokenizer.get_vocab().items():
    ...

# CORRECT: sort by token_id for deterministic iteration
for token_text, token_id in sorted(
    tokenizer.get_vocab().items(), key=lambda x: x[1]
):
    ...
```

Similarly, equivalence classes must be iterated in sorted order:

```python
# WRONG: dict order may vary
for normalized_text, members in groups.items():
    ...

# CORRECT: sort by normalized text
for normalized_text in sorted(groups.keys()):
    members = sorted(groups[normalized_text])
    ...
```

#### 7.2 Locale-Dependent Unicode Operations

Python's `str.lower()` can produce different results depending on the system locale
for certain characters (e.g., Turkish dotless i). Mitigation: use ASCII-safe
lowering for the ASCII range and standard Unicode case folding for the rest.

In practice, `str.lower()` in Python is locale-independent for all Unicode
characters except in very rare edge cases. The main risk is from C library
locale settings affecting underlying operations.

Mitigation: set `LC_ALL=C.UTF-8` and `LANG=C.UTF-8` in `os.environ` before
building. Call `locale.setlocale(locale.LC_ALL, 'C.UTF-8')` with a fallback
to `'C'` if the UTF-8 locale is unavailable.

#### 7.3 Floating-Point Operations

The build process must not use any floating-point operations that could introduce
platform-dependent rounding. Since tokenizer compression operates entirely on
strings and integers, this is naturally satisfied. The only floating-point value
in the output is the compression ratio, which is computed for reporting only and
is not part of the version hash.

#### 7.4 Hash Function Determinism

The SHA-256 version hash uses `hashlib.sha256`, which is deterministic across
all platforms. The input to the hash function is a canonical JSON string with
sorted keys and no extra whitespace.

#### 7.5 Tokenizer Decode Determinism

The `tokenizer.decode()` function must produce the same output for the same input
across platforms. This is generally true for HuggingFace tokenizers and SentencePiece.
However, some tokenizers have version-dependent behavior. Mitigation: pin the
tokenizer version in the metadata and verify at load time.

### Verification

Build the table N times and compare. For cross-platform verification, compare
version hashes (SHA-256 of build inputs):

```python
# Same-platform determinism check
tables = [build_equivalence_classes(tokenizer, recipe, special_ids)[0]
          for _ in range(10)]
assert all(np.array_equal(tables[0], t) for t in tables[1:])

# Cross-platform: compare version hashes from metadata
assert meta_machine_a['version_hash'] == meta_machine_b['version_hash']
```

### Pure Python/NumPy Build Constraint

The build step must use only pure Python and NumPy operations. No system calls,
no C extensions beyond NumPy, no random number generators. This ensures the
build is reproducible on any platform with a standard Python installation.

Banned operations in the build step:

| Operation | Reason |
|---|---|
| `random.random()` | Non-deterministic without seed |
| `os.urandom()` | Non-deterministic by design |
| `time.time()` | Varies by execution time (used in metadata only, not in table) |
| `multiprocessing.Pool` | Non-deterministic ordering |
| Platform-specific C calls | May behave differently across OS/arch |
| `torch` operations | CUDA non-determinism possible; use NumPy for build |

---

## 8. Compression Statistics

### Overview

After building the compression table, compute and report statistics that
characterize the quality and extent of the compression. These statistics serve
as validation (the compression should be meaningful) and as diagnostics
(unusual statistics may indicate misconfiguration).

### Statistics Data Structure

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class CompressionStatistics:
    """Statistics about the compression table."""
    original_vocab_size: int
    compressed_vocab_size: int
    compression_ratio: float
    num_equivalence_classes: int
    num_singleton_classes: int
    num_merged_classes: int
    max_class_size: int
    mean_class_size: float
    median_class_size: int
    class_size_distribution: Dict[int, int]  # size -> count
    num_special_tokens: int
    tokens_merged: int  # original_vocab_size - compressed_vocab_size
```

### Computing Statistics

```python
def compute_statistics(
    lookup_table: np.ndarray,
    equivalence_classes: Dict[int, List[int]],
    special_token_ids: Set[int],
) -> CompressionStatistics:
    """
    Compute compression statistics from the built table.

    Args:
        lookup_table: int32 array of shape (vocab_size,).
        equivalence_classes: canonical_id -> [member_ids].
        special_token_ids: Set of special token IDs.

    Returns:
        CompressionStatistics with all computed fields.
    """
    original_vocab = len(lookup_table)
    unique_canonical = len(np.unique(lookup_table))

    # Class sizes
    class_sizes = [len(members) for members in equivalence_classes.values()]

    # Size distribution
    size_dist: Dict[int, int] = {}
    for size in class_sizes:
        size_dist[size] = size_dist.get(size, 0) + 1

    singletons = sum(1 for s in class_sizes if s == 1)
    merged = sum(1 for s in class_sizes if s > 1)

    return CompressionStatistics(
        original_vocab_size=original_vocab,
        compressed_vocab_size=unique_canonical,
        compression_ratio=unique_canonical / original_vocab,
        num_equivalence_classes=len(equivalence_classes),
        num_singleton_classes=singletons,
        num_merged_classes=merged,
        max_class_size=max(class_sizes) if class_sizes else 0,
        mean_class_size=(
            sum(class_sizes) / len(class_sizes) if class_sizes else 0
        ),
        median_class_size=(
            sorted(class_sizes)[len(class_sizes) // 2] if class_sizes else 0
        ),
        class_size_distribution=dict(sorted(size_dist.items())),
        num_special_tokens=len(special_token_ids),
        tokens_merged=original_vocab - unique_canonical,
    )
```

### Expected Ranges

These ranges apply to typical BPE and SentencePiece tokenizers with the default
recipe (NFKC + case fold + whitespace):

| Metric | Expected Range | Flag if Outside |
|---|---|---|
| Compression ratio | 0.70 - 0.85 | < 0.50 (too aggressive) or > 0.95 (too weak) |
| Tokens merged | 15% - 30% of vocab | < 5% or > 50% |
| Max class size | 2 - 10 | > 20 (suspicious tokenizer) |
| Singleton fraction | 60% - 80% | < 40% or > 95% |
| Mean class size | 1.1 - 1.5 | > 2.0 |

### Validation Assertions

After computing statistics, assert:

1. `compressed_vocab_size < original_vocab_size` -- compression occurred
2. `compression_ratio > 0.50` -- not too aggressive
3. `num_merged_classes > 0` -- at least some merges
4. `max_class_size < 100` -- no suspiciously large classes
5. `compressed_vocab_size > 0` -- not degenerate

### Example Report

```
=== Tokenizer Compression Report ===

  Original vocabulary size:      128,000
  Compressed vocabulary size:     98,560
  Tokens merged:                  29,440
  Compression ratio:              0.7700
  Effective reduction:             23.0%

  Equivalence classes:             98,560
    Singleton (size=1):            69,120
    Merged (size>1):               29,440

  Max class size:                       5
  Mean class size:                   1.30
  Median class size:                    1

  Special tokens (invariant):           5

  Class size distribution:
    size   1:   69,120 ( 70.1%) ###################################
    size   2:   19,712 ( 20.0%) ##########
    size   3:    6,886 (  7.0%) ###
    size   4:    2,458 (  2.5%) #
    size   5:      384 (  0.4%)

========================================
```

### Per-Recipe Comparison

When evaluating recipe options, compare statistics across different recipes:

| Recipe | Compression Ratio | Tokens Merged | Max Class Size |
|---|---|---|---|
| `["nfkc"]` only | 0.97 | ~3% | 3 |
| `["nfkc", "case_fold"]` | 0.85 | ~15% | 4 |
| `["nfkc", "case_fold", "whitespace"]` | 0.77 | ~23% | 5 |
| `["nfkc", "case_fold", "whitespace", "strip_accents"]` | 0.73 | ~27% | 8 |

The default recipe (`nfkc + case_fold + whitespace`) provides the best
balance between compression and information preservation.

---

## 9. Integration with N-gram Pipeline

### Overview

Tokenizer compression is the first stage of the Engram N-gram pipeline. Its
output -- canonical IDs -- feeds directly into N-gram extraction, which then
feeds into multi-head hashing and embedding retrieval. The compression step
fundamentally improves the quality of the entire downstream pipeline.

### Data Flow

```
+----------------------------------------------------------+
|  Input: raw_input_ids    shape: (B, T)    dtype: int64   |
+----------------------------+-----------------------------+
                             |
                             v
+----------------------------------------------------------+
|  TokenizerCompression      table lookup on int tensor    |
|  canonical_ids = lookup_table[input_ids]                 |
|  shape: (B, T)             dtype: int32                  |
|  Effective vocab: ~77% of original                       |
+----------------------------+-----------------------------+
                             |
                             v
+----------------------------------------------------------+
|  N-gram Suffix Extraction                                |
|  For each position t, extract suffixes:                  |
|    order 1: [canonical_ids[t]]                           |
|    order 2: [canonical_ids[t-1], canonical_ids[t]]       |
|    order 3: [canonical_ids[t-2], canonical_ids[t-1],     |
|              canonical_ids[t]]                           |
|    ...up to order K                                      |
|  shape: (B, T, K, max_order) with padding                |
+----------------------------+-----------------------------+
                             |
                             v
+----------------------------------------------------------+
|  Multi-Head Hashing                                      |
|  hash(n-gram) -> row index in embedding table            |
|  Multiple heads per order for collision reduction         |
|  shape: (B, T, H_total) where H_total = K * num_heads   |
+----------------------------+-----------------------------+
                             |
                             v
+----------------------------------------------------------+
|  Embedding Retrieval                                     |
|  Gather rows from hash embedding tables                  |
|  Aggregate across heads (mean or learned combination)    |
|  shape: (B, T, D_emb)                                   |
+----------------------------------------------------------+
```

### Why Compression Improves N-gram Quality

#### 9.1 Denser N-gram Space

Without compression, the N-gram space is sparsely populated. Many N-grams differ
only in surface form:

```
Without compression:
  N-gram [" The", " cat"]  -> hash A
  N-gram ["The", " cat"]   -> hash B   (different hash, same meaning)
  N-gram [" the", " cat"]  -> hash C   (different hash, same meaning)
  N-gram ["the", " cat"]   -> hash D   (different hash, same meaning)

With compression:
  N-gram ["the", "cat"]    -> hash X   (single hash for all variants)
```

Compression collapses these surface variants, so the same semantic N-gram always
produces the same hash. This means:

1. The embedding for "the cat" is trained with 4x more examples (all variants
   contribute to the same embedding row).
2. The hash table has fewer "wasted" rows storing duplicate semantics.
3. Collision rate for semantically distinct N-grams decreases because the effective
   N-gram vocabulary is smaller.

#### 9.2 Better Hash Table Utilization

Consider a hash table with `M` rows and `N` distinct N-grams:

| Metric | Without Compression | With Compression |
|---|---|---|
| Distinct unigrams | V = 128,000 | V' = 98,560 |
| Distinct bigrams (theoretical max) | V^2 = 16.4B | V'^2 = 9.7B |
| Distinct bigrams (observed, typical corpus) | ~50M | ~30M |
| Load factor (observed N-grams / table size) | higher | lower |
| Semantic collision rate | higher | lower |

The reduced vocabulary means fewer distinct N-grams, which means lower load factor
in the hash tables, which means fewer collisions. The collisions that do occur are
more likely to be between genuinely different semantic content rather than surface
variants.

#### 9.3 Compression is Applied Once

A critical design property: compression is applied once at input time, before any
N-gram operations. It is not applied per-layer or per-head. This means:

1. The cost is amortized across all downstream operations.
2. All N-gram orders benefit from the same compression.
3. The canonical IDs are stable throughout the forward pass.

```python
class EngramModule(nn.Module):
    def forward(self, hidden_states, input_ids, attention_mask):
        # Step 1: Compress (once, at the start)
        canonical_ids = self.compressor.compress_ids(input_ids)  # (B, T)

        # Step 2: Extract N-grams from canonical IDs
        ngrams = self.extract_ngrams(canonical_ids, attention_mask)

        # Step 3: Hash N-grams to embedding indices
        hash_ids = self.hasher(ngrams)

        # Step 4: Retrieve and aggregate embeddings
        retrieved = self.embedding.gather(hash_ids)

        # Step 5: Gate and fuse with hidden states
        delta = self.gate_and_fuse(hidden_states, retrieved)

        return delta
```

### Mask Interaction

When computing N-grams, positions masked by `attention_mask` must not contribute
to N-gram suffixes. Compression does not affect masking -- the mask is applied
to the canonical IDs, not the raw IDs:

```python
# Apply mask after compression
canonical_ids = compressor.compress_ids(input_ids)  # (B, T)

# Mask padding positions to a designated padding canonical ID
# (which is the same as the original pad token ID, since pad is invariant)
if attention_mask is not None:
    pad_canonical = compressor.compress_ids(
        torch.tensor([pad_token_id])
    ).item()
    canonical_ids = canonical_ids.masked_fill(
        attention_mask == 0,
        pad_canonical,
    )
```

### Deterministic Hash Chain

The full chain from raw input to hash index must be deterministic:

```
raw_input_ids  --(compression)-->  canonical_ids  --(ngram extraction)-->  ngrams
  --(hashing)-->  hash_ids  --(gather)-->  embeddings
```

Because compression is deterministic (same input always produces same output),
and hashing is deterministic (integer operations with fixed seed), the entire
chain is deterministic. This enables:

1. **CPU offload**: Hash IDs can be computed on CPU and used to prefetch
   embeddings before they are needed on GPU.
2. **Caching**: N-gram hash IDs for a given input can be cached and reused.
3. **Distributed consistency**: All ranks compute identical hash IDs for the
   same input, enabling sharded embedding tables.

### Integration Checklist

When integrating tokenizer compression with the N-gram pipeline, verify:

| Check | How to Verify |
|---|---|
| Compression is applied before N-gram extraction | Trace forward pass; `compress_ids` call precedes `extract_ngrams` |
| Canonical IDs have correct shape | `assert canonical_ids.shape == input_ids.shape` |
| Canonical IDs are in valid range | `assert canonical_ids.max() < compressed_vocab_size` |
| Special tokens are preserved | `assert canonical_ids[mask_positions] == pad_canonical_id` |
| Compression is idempotent | `assert (compress(compress(x)) == compress(x)).all()` |
| Determinism holds | Run twice, compare: `assert (ids_run1 == ids_run2).all()` |
| Mask is applied after compression | Inspect code order |
| Lookup table is on correct device | `assert lookup_table.device == input_ids.device` |

---

## Appendix A: Troubleshooting

### Common Issues

| Issue | Symptom | Diagnosis | Fix |
|---|---|---|---|
| No compression | `compressed_vocab == original_vocab` | Recipe is empty or all tokens are singletons | Verify recipe has at least `["nfkc", "case_fold", "whitespace"]` |
| Too much compression | `compression_ratio < 0.50` | Overly aggressive normalization | Remove `"strip_accents"` or other aggressive steps |
| Non-deterministic tables | Tables differ across runs | Dict iteration order or locale | Sort all intermediate structures; set `LC_ALL=C.UTF-8` |
| Special tokens merged | Special tokens map to non-special IDs | Policy not correctly configured | Verify `SpecialTokenPolicy` includes all special IDs |
| Decode errors | Some tokens fail to decode | Tokenizer bug or byte-level tokens | Use fallback to raw token text |
| Version hash mismatch | Load fails with hash error | Different build parameters | Check tokenizer version, recipe, and special policy match |
| Idempotency failure | `lookup_table[canonical] != canonical` | Bug in canonical selection | Canonical must be a member of its own class |
| Memory error during build | OOM on large vocabulary | Storing all decoded strings | Process in chunks or use generator pattern |

### Diagnostic Snippets

```python
# Inspect merged equivalence classes
for canonical, members in eq_classes.items():
    if len(members) > 1:
        texts = [tokenizer.decode([m]) for m in members]
        print(f"  Canonical {canonical}: {members} -> {texts}")

# Verify invariants
for sid in special_policy.all_special_ids():
    assert lookup_table[sid] == sid, f"Special token {sid} not invariant"
for c in np.unique(lookup_table):
    assert lookup_table[c] == c, f"Idempotency violation at {c}"
```

### Edge Cases by Tokenizer Family

| Tokenizer | Edge Case | Handling |
|---|---|---|
| GPT-2 BPE | Tokens with leading `\u0120` (space marker) | `\u0120` decodes to space; whitespace normalization handles it |
| LLaMA SentencePiece | Byte-level tokens `<0x41>` | Decode to actual byte; normalize normally |
| BERT WordPiece | `##` prefix for subwords | `##` decodes as-is; no special handling needed |
| T5 SentencePiece | `\u2581` (lower one eighth block) as space | NFKC does not normalize this; whitespace norm handles if decoded as space |
| Tiktoken (GPT-4) | Bytes-to-unicode mapping | Decode handles mapping; normalize the decoded text |

### Performance Benchmarks (Reference)

Build time for common tokenizers on a single CPU core:

| Tokenizer | Vocab Size | Build Time | Table Size (disk) |
|---|---|---|---|
| GPT-2 | 50,257 | ~0.3s | 196 KB |
| LLaMA-2 | 32,000 | ~0.2s | 125 KB |
| LLaMA-3 | 128,256 | ~0.8s | 500 KB |
| GPT-4 (cl100k) | 100,256 | ~0.6s | 391 KB |
| BERT | 30,522 | ~0.2s | 119 KB |

Runtime compression (table lookup) on GPU:

| Batch Size | Seq Length | Time (A100) | Time (RTX 3090) |
|---|---|---|---|
| 1 | 2,048 | <0.01 ms | <0.01 ms |
| 32 | 2,048 | <0.01 ms | <0.01 ms |
| 128 | 4,096 | <0.02 ms | <0.03 ms |
| 256 | 8,192 | <0.03 ms | <0.05 ms |

Runtime cost is negligible in all practical scenarios. The table lookup is
memory-bandwidth bound, and the table fits easily in L2 cache on modern GPUs.
