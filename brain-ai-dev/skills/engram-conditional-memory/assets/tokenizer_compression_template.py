#!/usr/bin/env python3
"""
Tokenizer Compression Template for Engram Conditional Memory.

Surjective mapping that collapses textually equivalent tokens into canonical IDs.
Normalization recipe (NFKC + lowercasing + whitespace normalization) reduces effective
vocabulary (~23% in 128k tokenizer case study). Special tokens (pad/bos/eos/unk) are
invariant. Fast path at inference is pure int table lookup -- no string operations.

This template is the reference implementation for:
    brain_ai/memory/tokenizer_compression.py

Usage:
    from tokenizer_compression_template import (
        TokenizerCompression,
        TokenizerCompressionConfig,
        NormalizationPipeline,
        SpecialTokenPolicy,
        CompressionStats,
        MockTokenizer,
    )

    config = TokenizerCompressionConfig()
    compressor = TokenizerCompression(config)
    compressor.build_from_tokenizer(tokenizer)
    canonical_ids = compressor.compress_ids(input_ids)

Run self-tests:
    python tokenizer_compression_template.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

__version__ = "0.1.0"
_SERIALIZATION_FORMAT_VERSION = 1
_MAGIC_BYTES = b"TKCM"  # Tokenizer Compression Magic


# ---------------------------------------------------------------------------
# Normalization steps -- pure functions str -> str
# ---------------------------------------------------------------------------

def _nfkc(text: str) -> str:
    """Apply Unicode NFKC normalization.

    NFKC decomposes compatibility characters and recomposes canonical
    equivalences.  E.g. ligature 'fi' -> 'fi', fullwidth 'A' -> 'A'.
    """
    return unicodedata.normalize("NFKC", text)


def _nfc(text: str) -> str:
    """Apply Unicode NFC normalization (canonical decomposition + composition)."""
    return unicodedata.normalize("NFC", text)


def _nfkd(text: str) -> str:
    """Apply Unicode NFKD normalization (compatibility decomposition)."""
    return unicodedata.normalize("NFKD", text)


def _nfd(text: str) -> str:
    """Apply Unicode NFD normalization (canonical decomposition)."""
    return unicodedata.normalize("NFD", text)


def _lowercase(text: str) -> str:
    """Case-fold to lowercase.

    Uses Python ``str.lower()`` which is locale-independent for Unicode
    (Python normalizes via Unicode case folding rules).
    """
    return text.lower()


def _casefold(text: str) -> str:
    """Aggressive Unicode case folding (e.g. German sharp-s -> ss)."""
    return text.casefold()


def _strip_whitespace(text: str) -> str:
    """Strip leading and trailing whitespace."""
    return text.strip()


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace into a single space.

    Handles all Unicode whitespace categories, not only ASCII space.
    """
    return re.sub(r"\s+", " ", text)


def _strip_accents(text: str) -> str:
    """Remove combining diacritical marks (accents).

    Decomposes via NFD then filters out category Mn (Mark, Nonspacing).
    Re-composes via NFC afterwards for consistency.
    """
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def _strip_control_chars(text: str) -> str:
    """Remove Unicode control characters (category Cc/Cf) except common whitespace."""
    keep = {"\n", "\r", "\t", " "}
    return "".join(ch for ch in text if ch in keep or unicodedata.category(ch) not in ("Cc", "Cf"))


def _normalize_quotes(text: str) -> str:
    """Normalize fancy/smart quotes to ASCII equivalents."""
    replacements = {
        "\u2018": "'", "\u2019": "'",  # left/right single
        "\u201A": "'",                 # single low-9
        "\u201C": '"', "\u201D": '"',  # left/right double
        "\u201E": '"',                 # double low-9
        "\u2039": "'", "\u203A": "'",  # single guillemets
        "\u00AB": '"', "\u00BB": '"',  # double guillemets
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _normalize_dashes(text: str) -> str:
    """Normalize em-dashes, en-dashes, and other dash variants to ASCII hyphen."""
    dash_chars = "\u2010\u2011\u2012\u2013\u2014\u2015\uFE58\uFE63\uFF0D"
    for ch in dash_chars:
        text = text.replace(ch, "-")
    return text


def _identity(text: str) -> str:
    """Identity function -- no-op normalization step."""
    return text


# Registry of named normalization steps
_STEP_REGISTRY: Dict[str, Callable[[str], str]] = {
    "nfkc": _nfkc,
    "nfc": _nfc,
    "nfkd": _nfkd,
    "nfd": _nfd,
    "lowercase": _lowercase,
    "casefold": _casefold,
    "strip_whitespace": _strip_whitespace,
    "collapse_whitespace": _collapse_whitespace,
    "strip_accents": _strip_accents,
    "strip_control_chars": _strip_control_chars,
    "normalize_quotes": _normalize_quotes,
    "normalize_dashes": _normalize_dashes,
    "identity": _identity,
}


# ---------------------------------------------------------------------------
# NormalizationPipeline
# ---------------------------------------------------------------------------

class NormalizationPipeline:
    """Configurable sequence of text normalization steps.

    Each step is a pure function ``str -> str``.  The pipeline applies steps
    in order.  The result is deterministic and locale-independent (all
    operations use Python's Unicode-aware string methods).

    Parameters
    ----------
    steps : sequence of str
        Ordered list of step names.  Valid names are:
        ``nfkc``, ``nfc``, ``nfkd``, ``nfd``, ``lowercase``, ``casefold``,
        ``strip_whitespace``, ``collapse_whitespace``, ``strip_accents``,
        ``strip_control_chars``, ``normalize_quotes``, ``normalize_dashes``,
        ``identity``.

    Raises
    ------
    ValueError
        If an unknown step name is provided.

    Examples
    --------
    >>> pipe = NormalizationPipeline(["nfkc", "lowercase", "strip_whitespace"])
    >>> pipe("  Hello\u3000World  ")
    'hello world'
    """

    # Default recipe matching the SKILL.md specification
    DEFAULT_STEPS: ClassVar[Tuple[str, ...]] = (
        "nfkc",
        "lowercase",
        "collapse_whitespace",
        "strip_whitespace",
    )

    def __init__(self, steps: Optional[Sequence[str]] = None) -> None:
        if steps is None:
            steps = list(self.DEFAULT_STEPS)
        self._step_names: Tuple[str, ...] = tuple(steps)
        self._functions: Tuple[Callable[[str], str], ...] = tuple(
            self._resolve(name) for name in self._step_names
        )

    # -- public API ----------------------------------------------------------

    def __call__(self, text: str) -> str:
        """Apply the full normalization pipeline to *text*."""
        for fn in self._functions:
            text = fn(text)
        return text

    def normalize(self, text: str) -> str:
        """Alias for ``__call__`` -- apply the pipeline."""
        return self(text)

    @property
    def step_names(self) -> Tuple[str, ...]:
        """Return the ordered tuple of step names."""
        return self._step_names

    def to_recipe_string(self) -> str:
        """Serialize the pipeline as a comma-separated recipe string.

        The recipe string is used as part of the version hash for
        reproducibility tracking.
        """
        return ",".join(self._step_names)

    @classmethod
    def from_recipe_string(cls, recipe: str) -> "NormalizationPipeline":
        """Reconstruct a pipeline from a recipe string."""
        if not recipe or recipe.strip() == "":
            return cls(steps=[])
        steps = [s.strip() for s in recipe.split(",") if s.strip()]
        return cls(steps=steps)

    def __repr__(self) -> str:
        return f"NormalizationPipeline(steps={list(self._step_names)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NormalizationPipeline):
            return NotImplemented
        return self._step_names == other._step_names

    def __hash__(self) -> int:
        return hash(self._step_names)

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _resolve(name: str) -> Callable[[str], str]:
        """Resolve a step name to its function."""
        fn = _STEP_REGISTRY.get(name)
        if fn is None:
            valid = ", ".join(sorted(_STEP_REGISTRY.keys()))
            raise ValueError(
                f"Unknown normalization step '{name}'. Valid steps: {valid}"
            )
        return fn

    @classmethod
    def available_steps(cls) -> List[str]:
        """Return a sorted list of all registered step names."""
        return sorted(_STEP_REGISTRY.keys())


# ---------------------------------------------------------------------------
# SpecialTokenPolicy
# ---------------------------------------------------------------------------

class SpecialTokenPolicy:
    """Defines which token IDs are 'special' and must map to themselves.

    Special tokens (pad, bos, eos, unk, and any user-specified extras) are
    excluded from equivalence-class merging so their canonical ID is always
    their original ID.

    Parameters
    ----------
    pad_id : int or None
        Padding token ID.
    bos_id : int or None
        Beginning-of-sequence token ID.
    eos_id : int or None
        End-of-sequence token ID.
    unk_id : int or None
        Unknown token ID.
    additional_special_ids : sequence of int
        Extra IDs that must also be treated as special (e.g. ``<mask>``).
    """

    def __init__(
        self,
        pad_id: Optional[int] = None,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        unk_id: Optional[int] = None,
        additional_special_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.additional_special_ids: Tuple[int, ...] = tuple(
            additional_special_ids or []
        )

    # -- public API ----------------------------------------------------------

    @property
    def all_special_ids(self) -> FrozenSet[int]:
        """Return the frozen set of all special token IDs (excluding None)."""
        ids: Set[int] = set()
        for sid in (self.pad_id, self.bos_id, self.eos_id, self.unk_id):
            if sid is not None:
                ids.add(sid)
        ids.update(self.additional_special_ids)
        return frozenset(ids)

    def is_special(self, token_id: int) -> bool:
        """Return True if *token_id* is a special token."""
        return token_id in self.all_special_ids

    def validate(self, vocab_size: int) -> None:
        """Validate that all special IDs are within ``[0, vocab_size)``.

        Raises
        ------
        ValueError
            If any special ID is out of range.
        """
        for sid in self.all_special_ids:
            if sid < 0 or sid >= vocab_size:
                raise ValueError(
                    f"Special token ID {sid} is out of range [0, {vocab_size})"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "additional_special_ids": list(self.additional_special_ids),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpecialTokenPolicy":
        """Reconstruct from a dictionary."""
        return cls(
            pad_id=d.get("pad_id"),
            bos_id=d.get("bos_id"),
            eos_id=d.get("eos_id"),
            unk_id=d.get("unk_id"),
            additional_special_ids=d.get("additional_special_ids", []),
        )

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "SpecialTokenPolicy":
        """Attempt to auto-detect special tokens from a tokenizer object.

        Probes for common attribute names used by HuggingFace tokenizers.
        Returns a policy with whatever IDs were found.
        """
        pad_id = getattr(tokenizer, "pad_token_id", None)
        bos_id = getattr(tokenizer, "bos_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        unk_id = getattr(tokenizer, "unk_token_id", None)

        additional: List[int] = []
        # Some tokenizers have all_special_ids
        if hasattr(tokenizer, "all_special_ids"):
            known = {pad_id, bos_id, eos_id, unk_id}
            for sid in tokenizer.all_special_ids:
                if sid not in known and sid is not None:
                    additional.append(sid)

        return cls(
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            unk_id=unk_id,
            additional_special_ids=sorted(set(additional)),
        )

    def sorted_ids(self) -> List[int]:
        """Return special IDs as a sorted list (for deterministic hashing)."""
        return sorted(self.all_special_ids)

    def __repr__(self) -> str:
        return (
            f"SpecialTokenPolicy(pad={self.pad_id}, bos={self.bos_id}, "
            f"eos={self.eos_id}, unk={self.unk_id}, "
            f"extra={list(self.additional_special_ids)})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpecialTokenPolicy):
            return NotImplemented
        return self.to_dict() == other.to_dict()


# ---------------------------------------------------------------------------
# CompressionStats
# ---------------------------------------------------------------------------

@dataclass
class CompressionStats:
    """Statistics about a completed tokenizer compression build.

    Attributes
    ----------
    original_vocab_size : int
        Size of the original tokenizer vocabulary.
    compressed_vocab_size : int
        Number of unique canonical IDs after compression.
    compression_ratio : float
        ``compressed_vocab_size / original_vocab_size``.  Values < 1 indicate
        vocabulary reduction.
    num_equivalence_classes : int
        Number of distinct equivalence classes (normalized text groups).
    max_class_size : int
        Size of the largest equivalence class.
    mean_class_size : float
        Average equivalence class size.
    median_class_size : float
        Median equivalence class size.
    num_singleton_classes : int
        Number of classes with exactly one member (no merging).
    num_special_tokens_preserved : int
        Number of special tokens that map to themselves.
    build_time_seconds : float
        Wall-clock time to build the compression table.
    normalization_recipe : str
        The normalization recipe string used.
    version_hash : str
        SHA-256 version hash of the build configuration.
    """

    original_vocab_size: int = 0
    compressed_vocab_size: int = 0
    compression_ratio: float = 1.0
    num_equivalence_classes: int = 0
    max_class_size: int = 0
    mean_class_size: float = 0.0
    median_class_size: float = 0.0
    num_singleton_classes: int = 0
    num_special_tokens_preserved: int = 0
    build_time_seconds: float = 0.0
    normalization_recipe: str = ""
    version_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CompressionStats":
        """Reconstruct from a dictionary."""
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=== Tokenizer Compression Stats ===",
            f"  Original vocab size:      {self.original_vocab_size:,}",
            f"  Compressed vocab size:    {self.compressed_vocab_size:,}",
            f"  Compression ratio:        {self.compression_ratio:.4f}",
            f"  Vocabulary reduction:     {(1 - self.compression_ratio) * 100:.1f}%",
            f"  Equivalence classes:      {self.num_equivalence_classes:,}",
            f"  Max class size:           {self.max_class_size:,}",
            f"  Mean class size:          {self.mean_class_size:.2f}",
            f"  Median class size:        {self.median_class_size:.1f}",
            f"  Singleton classes:        {self.num_singleton_classes:,}",
            f"  Special tokens preserved: {self.num_special_tokens_preserved:,}",
            f"  Build time:               {self.build_time_seconds:.3f}s",
            f"  Version hash:             {self.version_hash[:16]}...",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# EquivalenceClass -- internal data structure
# ---------------------------------------------------------------------------

@dataclass
class EquivalenceClass:
    """A group of token IDs that map to the same normalized text.

    Attributes
    ----------
    normalized_text : str
        The normalized form shared by all members.
    member_ids : list of int
        Original token IDs in this class, sorted ascending.
    canonical_id : int
        The chosen representative ID for the class.
    """

    normalized_text: str
    member_ids: List[int] = field(default_factory=list)
    canonical_id: int = -1

    @property
    def size(self) -> int:
        """Number of tokens in this equivalence class."""
        return len(self.member_ids)

    def select_canonical(
        self,
        policy: Literal["lowest_id", "most_frequent"] = "lowest_id",
        frequency_map: Optional[Dict[int, int]] = None,
    ) -> int:
        """Select the canonical ID from the members.

        Parameters
        ----------
        policy : str
            ``"lowest_id"`` picks the smallest ID (deterministic, simple).
            ``"most_frequent"`` picks the ID with highest frequency in
            *frequency_map* (ties broken by lowest ID).
        frequency_map : dict, optional
            Mapping from token ID to usage frequency.  Required if
            *policy* is ``"most_frequent"``.

        Returns
        -------
        int
            The selected canonical ID.
        """
        if not self.member_ids:
            raise ValueError("Cannot select canonical from empty class")

        sorted_members = sorted(self.member_ids)

        if policy == "lowest_id":
            self.canonical_id = sorted_members[0]
        elif policy == "most_frequent":
            if frequency_map is None:
                # Fall back to lowest_id
                self.canonical_id = sorted_members[0]
            else:
                # Sort by (-frequency, id) for deterministic tie-breaking
                best = min(
                    sorted_members,
                    key=lambda tid: (-frequency_map.get(tid, 0), tid),
                )
                self.canonical_id = best
        else:
            raise ValueError(f"Unknown canonical selection policy: {policy!r}")

        return self.canonical_id


# ---------------------------------------------------------------------------
# TokenizerCompressionConfig
# ---------------------------------------------------------------------------

@dataclass
class TokenizerCompressionConfig:
    """Configuration for tokenizer compression.

    Attributes
    ----------
    normalization_recipe : list of str
        Ordered normalization steps.  Default: ``["nfkc", "lowercase",
        "collapse_whitespace", "strip_whitespace"]``.
    canonical_selection_policy : str
        How to pick the canonical ID per equivalence class.
        ``"lowest_id"`` or ``"most_frequent"``.
    special_token_policy : SpecialTokenPolicy or None
        Explicit special token policy.  If None, will be auto-detected
        from the tokenizer during ``build_from_tokenizer``.
    seed : int
        Seed for deterministic operations (used in version hashing).
    strip_accents : bool
        If True, add ``"strip_accents"`` to the normalization recipe.
    use_casefold : bool
        If True, use ``"casefold"`` instead of ``"lowercase"`` for more
        aggressive Unicode case folding.
    normalize_quotes : bool
        If True, add ``"normalize_quotes"`` step.
    normalize_dashes : bool
        If True, add ``"normalize_dashes"`` step.
    strip_control_chars : bool
        If True, add ``"strip_control_chars"`` step.
    tokenizer_name : str
        Name/identifier of the tokenizer, used in version hashing.
    frequency_map : dict or None
        Token ID -> usage count mapping for ``"most_frequent"`` policy.
    """

    normalization_recipe: List[str] = field(
        default_factory=lambda: ["nfkc", "lowercase", "collapse_whitespace", "strip_whitespace"]
    )
    canonical_selection_policy: str = "lowest_id"
    special_token_policy: Optional[SpecialTokenPolicy] = None
    seed: int = 42
    strip_accents: bool = False
    use_casefold: bool = False
    normalize_quotes: bool = False
    normalize_dashes: bool = False
    strip_control_chars: bool = False
    tokenizer_name: str = "unknown"
    frequency_map: Optional[Dict[int, int]] = None

    def build_normalization_pipeline(self) -> NormalizationPipeline:
        """Construct the full normalization pipeline from config flags.

        Starts with ``normalization_recipe`` then appends optional steps
        based on boolean flags (in deterministic order).
        """
        steps = list(self.normalization_recipe)

        if self.use_casefold and "casefold" not in steps:
            # Replace lowercase with casefold if present
            if "lowercase" in steps:
                idx = steps.index("lowercase")
                steps[idx] = "casefold"
            else:
                steps.append("casefold")

        if self.strip_accents and "strip_accents" not in steps:
            # Insert after nfkc/nfc if present, else append
            insert_after = None
            for norm in ("nfkc", "nfc", "nfkd", "nfd"):
                if norm in steps:
                    insert_after = steps.index(norm) + 1
                    break
            if insert_after is not None:
                steps.insert(insert_after, "strip_accents")
            else:
                steps.append("strip_accents")

        if self.strip_control_chars and "strip_control_chars" not in steps:
            steps.append("strip_control_chars")

        if self.normalize_quotes and "normalize_quotes" not in steps:
            steps.append("normalize_quotes")

        if self.normalize_dashes and "normalize_dashes" not in steps:
            steps.append("normalize_dashes")

        return NormalizationPipeline(steps=steps)


# ---------------------------------------------------------------------------
# TokenizerCompression -- main class
# ---------------------------------------------------------------------------

class TokenizerCompression:
    """Surjective vocabulary mapping that collapses textually equivalent tokens.

    The core idea: many tokenizers assign distinct IDs to tokens that are
    textually equivalent after normalization (e.g. ``"Hello"`` vs ``"hello"``,
    or ``"\\u00e9"`` vs ``"e\\u0301"``).  This class groups such tokens into
    equivalence classes and maps each original ID to a single canonical ID.

    At runtime, compression is a pure integer table lookup -- no string
    operations.  This makes it suitable for the hot path in training and
    inference.

    Parameters
    ----------
    config : TokenizerCompressionConfig
        Full configuration for normalization, special tokens, and policies.

    Attributes
    ----------
    _lookup : numpy.ndarray or torch.Tensor
        Int32 lookup table of shape ``(original_vocab_size,)`` mapping
        ``original_id -> canonical_id``.
    _equivalence_classes : list of EquivalenceClass
        The computed equivalence classes (populated after build).
    _stats : CompressionStats
        Build statistics (populated after build).
    _version_hash : str
        SHA-256 hash of the build configuration for reproducibility tracking.
    _is_built : bool
        Whether ``build_from_tokenizer`` has been called.
    """

    def __init__(self, config: Optional[TokenizerCompressionConfig] = None) -> None:
        if config is None:
            config = TokenizerCompressionConfig()
        self._config = config
        self._pipeline: Optional[NormalizationPipeline] = None
        self._special_policy: Optional[SpecialTokenPolicy] = None

        # Populated by build_from_tokenizer
        self._lookup: Optional[np.ndarray] = None
        self._lookup_torch: Optional[Any] = None  # torch.Tensor cache
        self._equivalence_classes: List[EquivalenceClass] = []
        self._stats: Optional[CompressionStats] = None
        self._version_hash: str = ""
        self._is_built: bool = False
        self._original_vocab_size: int = 0
        self._compressed_vocab_size: int = 0
        self._tokenizer_name: str = config.tokenizer_name

    # -----------------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------------

    def build_from_tokenizer(
        self,
        tokenizer: Any,
        *,
        normalization_recipe: Optional[Sequence[str]] = None,
        special_token_policy: Optional[SpecialTokenPolicy] = None,
        frequency_map: Optional[Dict[int, int]] = None,
    ) -> "TokenizerCompression":
        """Build the compression table from a tokenizer's vocabulary.

        For each token ID in the vocabulary:
        1. Decode the ID to its text string.
        2. Apply the normalization pipeline (NFKC, case fold, whitespace, etc.).
        3. Group tokens by normalized text into equivalence classes.
        4. Select a canonical ID per class (policy: "lowest_id" or "most_frequent").
        5. Build int32 lookup table: ``self._lookup[original_id] = canonical_id``.
        6. Special tokens always map to themselves.

        Parameters
        ----------
        tokenizer : object
            Any tokenizer with a ``vocab_size`` property (or ``__len__``)
            and a ``decode(id) -> str`` method.
        normalization_recipe : sequence of str, optional
            Override the config's normalization recipe for this build.
        special_token_policy : SpecialTokenPolicy, optional
            Override the config's special token policy for this build.
        frequency_map : dict, optional
            Override the config's frequency map for "most_frequent" policy.

        Returns
        -------
        TokenizerCompression
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If the tokenizer lacks required methods/attributes.
        """
        start_time = time.monotonic()

        # -- resolve configuration overrides ----------------------------------
        if normalization_recipe is not None:
            pipeline = NormalizationPipeline(steps=list(normalization_recipe))
        else:
            pipeline = self._config.build_normalization_pipeline()
        self._pipeline = pipeline

        if special_token_policy is not None:
            self._special_policy = special_token_policy
        elif self._config.special_token_policy is not None:
            self._special_policy = self._config.special_token_policy
        else:
            self._special_policy = SpecialTokenPolicy.from_tokenizer(tokenizer)

        freq_map = frequency_map or self._config.frequency_map
        policy = self._config.canonical_selection_policy

        # -- resolve vocab size -----------------------------------------------
        vocab_size = self._get_vocab_size(tokenizer)
        self._original_vocab_size = vocab_size

        # Validate special token IDs
        self._special_policy.validate(vocab_size)

        special_ids = self._special_policy.all_special_ids

        # -- resolve tokenizer name -------------------------------------------
        if hasattr(tokenizer, "name_or_path"):
            self._tokenizer_name = tokenizer.name_or_path
        elif hasattr(tokenizer, "name"):
            self._tokenizer_name = tokenizer.name
        else:
            self._tokenizer_name = self._config.tokenizer_name

        # -- step 1-2: decode + normalize each token --------------------------
        # Mapping: normalized_text -> list of original token IDs
        norm_groups: Dict[str, List[int]] = {}

        for token_id in range(vocab_size):
            if token_id in special_ids:
                # Special tokens form their own singleton class
                # We handle them separately below
                continue

            text = self._decode_token(tokenizer, token_id)
            normalized = pipeline(text)

            if normalized not in norm_groups:
                norm_groups[normalized] = []
            norm_groups[normalized].append(token_id)

        # -- step 3: build equivalence classes --------------------------------
        equivalence_classes: List[EquivalenceClass] = []

        # Sort by normalized text for determinism
        for norm_text in sorted(norm_groups.keys()):
            members = sorted(norm_groups[norm_text])
            ec = EquivalenceClass(
                normalized_text=norm_text,
                member_ids=members,
            )
            ec.select_canonical(policy=policy, frequency_map=freq_map)
            equivalence_classes.append(ec)

        # Add special tokens as singleton classes
        for sid in sorted(special_ids):
            text = self._decode_token(tokenizer, sid)
            ec = EquivalenceClass(
                normalized_text=f"__special_{sid}__",
                member_ids=[sid],
                canonical_id=sid,
            )
            equivalence_classes.append(ec)

        self._equivalence_classes = equivalence_classes

        # -- step 4-5: build lookup table -------------------------------------
        lookup = np.arange(vocab_size, dtype=np.int32)  # identity default

        # Remap canonical IDs to a contiguous range for efficiency
        # First, collect all unique canonical IDs
        canonical_id_set: Dict[int, int] = {}  # old_canonical -> new_canonical
        new_id_counter = 0

        # Special tokens keep their original IDs as canonical
        for sid in sorted(special_ids):
            canonical_id_set[sid] = sid
            # Ensure new_id_counter doesn't collide
            if sid >= new_id_counter:
                new_id_counter = sid + 1

        # Assign contiguous IDs to non-special equivalence classes
        # We do NOT remap to contiguous here -- we keep original canonical IDs
        # because the lookup table maps original_id -> canonical_original_id.
        # The "canonical_id" in each class is already a valid original token ID.

        for ec in equivalence_classes:
            if ec.canonical_id in special_ids:
                # Special token class -- already handled
                continue
            for member_id in ec.member_ids:
                lookup[member_id] = ec.canonical_id

        # Special tokens map to themselves (already identity from init, but
        # explicit for clarity)
        for sid in special_ids:
            lookup[sid] = sid

        self._lookup = lookup
        self._lookup_torch = None  # invalidate cache

        # -- compute stats ----------------------------------------------------
        unique_canonical = len(set(lookup.tolist()))
        class_sizes = [ec.size for ec in equivalence_classes if ec.canonical_id not in special_ids]
        if not class_sizes:
            class_sizes = [1]

        build_time = time.monotonic() - start_time
        self._compressed_vocab_size = unique_canonical

        self._version_hash = self._compute_version_hash()

        self._stats = CompressionStats(
            original_vocab_size=vocab_size,
            compressed_vocab_size=unique_canonical,
            compression_ratio=unique_canonical / vocab_size if vocab_size > 0 else 1.0,
            num_equivalence_classes=len(equivalence_classes),
            max_class_size=max(class_sizes) if class_sizes else 0,
            mean_class_size=float(np.mean(class_sizes)) if class_sizes else 0.0,
            median_class_size=float(np.median(class_sizes)) if class_sizes else 0.0,
            num_singleton_classes=sum(1 for s in class_sizes if s == 1),
            num_special_tokens_preserved=len(special_ids),
            build_time_seconds=build_time,
            normalization_recipe=pipeline.to_recipe_string(),
            version_hash=self._version_hash,
        )

        self._is_built = True
        return self

    # -----------------------------------------------------------------------
    # Compress IDs (fast path)
    # -----------------------------------------------------------------------

    def compress_ids(
        self,
        input_ids: Union[np.ndarray, "torch.Tensor"],
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Map original token IDs to canonical IDs via table lookup.

        This is the fast path: pure integer indexing, no string operations.
        Works with both ``torch.Tensor`` and ``numpy.ndarray``.  Preserves
        input shape and device.

        Parameters
        ----------
        input_ids : numpy.ndarray or torch.Tensor
            Integer tensor of original token IDs, any shape.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Canonical IDs with the same shape, dtype, and device as input.

        Raises
        ------
        RuntimeError
            If ``build_from_tokenizer`` has not been called.
        """
        if not self._is_built:
            raise RuntimeError(
                "Compression table not built. Call build_from_tokenizer() first."
            )

        if HAS_TORCH and isinstance(input_ids, torch.Tensor):
            return self._compress_torch(input_ids)
        elif isinstance(input_ids, np.ndarray):
            return self._compress_numpy(input_ids)
        else:
            raise TypeError(
                f"Unsupported input type: {type(input_ids)}. "
                f"Expected torch.Tensor or numpy.ndarray."
            )

    def _compress_torch(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        """Torch fast path for compress_ids."""
        device = input_ids.device
        original_dtype = input_ids.dtype

        # Build or retrieve cached torch lookup tensor
        if self._lookup_torch is None or self._lookup_torch.device != device:
            self._lookup_torch = torch.from_numpy(self._lookup).long().to(device)

        # Clamp to valid range
        clamped = input_ids.long().clamp(0, self._original_vocab_size - 1)

        # Table lookup
        result = self._lookup_torch[clamped]

        # Preserve original integer dtype if possible
        if original_dtype in (torch.int32, torch.int16, torch.int8):
            result = result.to(original_dtype)
        elif original_dtype == torch.long:
            pass  # already long
        else:
            result = result.long()

        return result

    def _compress_numpy(self, input_ids: np.ndarray) -> np.ndarray:
        """Numpy fast path for compress_ids."""
        clamped = np.clip(input_ids, 0, self._original_vocab_size - 1).astype(np.intp)
        result = self._lookup[clamped]
        return result.astype(input_ids.dtype) if input_ids.dtype != np.int32 else result

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def serialize(self, path: Union[str, Path]) -> None:
        """Save the compression table and metadata to disk.

        File format:
        1. 4-byte magic (``TKCM``).
        2. 4-byte format version (little-endian uint32).
        3. 4-byte metadata JSON length (little-endian uint32).
        4. Metadata JSON (UTF-8 encoded).
        5. Lookup table as raw int32 bytes (little-endian).

        Parameters
        ----------
        path : str or Path
            Output file path.

        Raises
        ------
        RuntimeError
            If not built yet.
        """
        if not self._is_built:
            raise RuntimeError("Cannot serialize before building.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "format_version": _SERIALIZATION_FORMAT_VERSION,
            "version_hash": self._version_hash,
            "tokenizer_name": self._tokenizer_name,
            "original_vocab_size": self._original_vocab_size,
            "compressed_vocab_size": self._compressed_vocab_size,
            "normalization_recipe": self._pipeline.to_recipe_string() if self._pipeline else "",
            "special_token_policy": self._special_policy.to_dict() if self._special_policy else {},
            "canonical_selection_policy": self._config.canonical_selection_policy,
            "seed": self._config.seed,
            "stats": self._stats.to_dict() if self._stats else {},
            "module_version": __version__,
        }

        meta_bytes = json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8")
        lookup_bytes = self._lookup.tobytes()

        with open(path, "wb") as f:
            f.write(_MAGIC_BYTES)
            f.write(struct.pack("<I", _SERIALIZATION_FORMAT_VERSION))
            f.write(struct.pack("<I", len(meta_bytes)))
            f.write(meta_bytes)
            f.write(lookup_bytes)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        verify_hash: bool = True,
        expected_hash: Optional[str] = None,
    ) -> "TokenizerCompression":
        """Load a compression table from disk.

        Parameters
        ----------
        path : str or Path
            Input file path.
        verify_hash : bool
            If True and *expected_hash* is provided, verify the version hash.
        expected_hash : str, optional
            Expected version hash for verification.

        Returns
        -------
        TokenizerCompression
            A fully reconstructed instance ready for ``compress_ids``.

        Raises
        ------
        ValueError
            If magic bytes or format version are wrong, or hash mismatch.
        FileNotFoundError
            If file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")

        with open(path, "rb") as f:
            # Read magic
            magic = f.read(4)
            if magic != _MAGIC_BYTES:
                raise ValueError(
                    f"Invalid magic bytes: expected {_MAGIC_BYTES!r}, got {magic!r}"
                )

            # Read format version
            fmt_version = struct.unpack("<I", f.read(4))[0]
            if fmt_version != _SERIALIZATION_FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported format version: {fmt_version} "
                    f"(expected {_SERIALIZATION_FORMAT_VERSION})"
                )

            # Read metadata
            meta_len = struct.unpack("<I", f.read(4))[0]
            meta_bytes = f.read(meta_len)
            metadata = json.loads(meta_bytes.decode("utf-8"))

            # Read lookup table
            lookup_bytes = f.read()

        # Verify hash if requested
        stored_hash = metadata.get("version_hash", "")
        if verify_hash and expected_hash is not None:
            if stored_hash != expected_hash:
                raise ValueError(
                    f"Version hash mismatch: stored={stored_hash[:16]}... "
                    f"expected={expected_hash[:16]}..."
                )

        # Reconstruct
        vocab_size = metadata["original_vocab_size"]
        lookup = np.frombuffer(lookup_bytes, dtype=np.int32).copy()
        if len(lookup) != vocab_size:
            raise ValueError(
                f"Lookup table size mismatch: {len(lookup)} != {vocab_size}"
            )

        # Build config from metadata
        recipe_str = metadata.get("normalization_recipe", "")
        special_dict = metadata.get("special_token_policy", {})

        config = TokenizerCompressionConfig(
            normalization_recipe=recipe_str.split(",") if recipe_str else [],
            canonical_selection_policy=metadata.get("canonical_selection_policy", "lowest_id"),
            special_token_policy=SpecialTokenPolicy.from_dict(special_dict) if special_dict else None,
            seed=metadata.get("seed", 42),
            tokenizer_name=metadata.get("tokenizer_name", "unknown"),
        )

        instance = cls(config=config)
        instance._lookup = lookup
        instance._lookup_torch = None
        instance._original_vocab_size = vocab_size
        instance._compressed_vocab_size = metadata.get("compressed_vocab_size", len(set(lookup.tolist())))
        instance._version_hash = stored_hash
        instance._tokenizer_name = metadata.get("tokenizer_name", "unknown")
        instance._pipeline = NormalizationPipeline.from_recipe_string(recipe_str)
        instance._special_policy = SpecialTokenPolicy.from_dict(special_dict) if special_dict else SpecialTokenPolicy()
        instance._is_built = True

        # Reconstruct stats
        stats_dict = metadata.get("stats", {})
        if stats_dict:
            instance._stats = CompressionStats.from_dict(stats_dict)
        else:
            instance._stats = CompressionStats(
                original_vocab_size=vocab_size,
                compressed_vocab_size=instance._compressed_vocab_size,
                compression_ratio=instance._compressed_vocab_size / vocab_size if vocab_size > 0 else 1.0,
            )

        return instance

    # -----------------------------------------------------------------------
    # Version hashing
    # -----------------------------------------------------------------------

    def _compute_version_hash(self) -> str:
        """Compute SHA-256 version hash for reproducibility tracking.

        Hash inputs:
        - tokenizer_name
        - normalization_recipe_str
        - special_token_ids_sorted
        - seed
        """
        hasher = hashlib.sha256()

        hasher.update(self._tokenizer_name.encode("utf-8"))

        if self._pipeline is not None:
            hasher.update(self._pipeline.to_recipe_string().encode("utf-8"))

        if self._special_policy is not None:
            ids_str = ",".join(str(i) for i in self._special_policy.sorted_ids())
            hasher.update(ids_str.encode("utf-8"))

        hasher.update(str(self._config.seed).encode("utf-8"))

        return hasher.hexdigest()

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def compression_ratio(self) -> float:
        """Ratio of compressed to original vocab size. Values < 1 mean reduction."""
        if not self._is_built:
            return 1.0
        return self._compressed_vocab_size / self._original_vocab_size if self._original_vocab_size > 0 else 1.0

    @property
    def num_equivalence_classes(self) -> int:
        """Number of distinct equivalence classes."""
        return len(self._equivalence_classes)

    @property
    def original_vocab_size(self) -> int:
        """Size of the original tokenizer vocabulary."""
        return self._original_vocab_size

    @property
    def compressed_vocab_size(self) -> int:
        """Number of unique canonical IDs after compression."""
        return self._compressed_vocab_size

    @property
    def stats(self) -> Optional[CompressionStats]:
        """Build statistics, or None if not yet built."""
        return self._stats

    @property
    def is_built(self) -> bool:
        """Whether the compression table has been built."""
        return self._is_built

    @property
    def version_hash(self) -> str:
        """SHA-256 version hash of the build configuration."""
        return self._version_hash

    @property
    def lookup_table(self) -> Optional[np.ndarray]:
        """The raw int32 lookup table (read-only view)."""
        if self._lookup is None:
            return None
        view = self._lookup.view()
        view.flags.writeable = False
        return view

    @property
    def pipeline(self) -> Optional[NormalizationPipeline]:
        """The normalization pipeline used for building."""
        return self._pipeline

    @property
    def special_token_policy(self) -> Optional[SpecialTokenPolicy]:
        """The special token policy used for building."""
        return self._special_policy

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_vocab_size(tokenizer: Any) -> int:
        """Extract vocabulary size from a tokenizer object."""
        if hasattr(tokenizer, "vocab_size"):
            vs = tokenizer.vocab_size
            if callable(vs):
                return vs()
            return vs
        if hasattr(tokenizer, "get_vocab"):
            return len(tokenizer.get_vocab())
        if hasattr(tokenizer, "__len__"):
            return len(tokenizer)
        raise ValueError(
            "Cannot determine vocab size from tokenizer. "
            "Expected 'vocab_size' property, 'get_vocab()' method, or '__len__'."
        )

    @staticmethod
    def _decode_token(tokenizer: Any, token_id: int) -> str:
        """Decode a single token ID to its text string.

        Tries multiple methods in order of preference:
        1. ``tokenizer.decode([token_id])``
        2. ``tokenizer.id_to_token(token_id)``
        3. ``tokenizer.convert_ids_to_tokens([token_id])[0]``
        4. Reverse vocab lookup
        """
        # Method 1: decode (most common for HF tokenizers)
        if hasattr(tokenizer, "decode"):
            try:
                text = tokenizer.decode([token_id])
                if isinstance(text, str):
                    return text
            except (TypeError, IndexError, KeyError):
                pass
            # Some tokenizers accept a single int
            try:
                text = tokenizer.decode(token_id)
                if isinstance(text, str):
                    return text
            except (TypeError, IndexError, KeyError):
                pass

        # Method 2: id_to_token
        if hasattr(tokenizer, "id_to_token"):
            try:
                text = tokenizer.id_to_token(token_id)
                if text is not None:
                    return text
            except (TypeError, IndexError, KeyError):
                pass

        # Method 3: convert_ids_to_tokens
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            try:
                tokens = tokenizer.convert_ids_to_tokens([token_id])
                if tokens and tokens[0] is not None:
                    return tokens[0]
            except (TypeError, IndexError, KeyError):
                pass

        # Method 4: reverse vocab lookup
        if hasattr(tokenizer, "get_vocab"):
            try:
                vocab = tokenizer.get_vocab()
                inv = {v: k for k, v in vocab.items()}
                if token_id in inv:
                    return inv[token_id]
            except Exception:
                pass

        # Fallback: return the string representation of the ID
        return str(token_id)

    def get_equivalence_class(self, token_id: int) -> Optional[EquivalenceClass]:
        """Find the equivalence class containing *token_id*."""
        for ec in self._equivalence_classes:
            if token_id in ec.member_ids:
                return ec
        return None

    def get_canonical_id(self, token_id: int) -> int:
        """Get the canonical ID for a single token ID.

        Parameters
        ----------
        token_id : int
            Original token ID.

        Returns
        -------
        int
            The canonical ID.
        """
        if not self._is_built:
            raise RuntimeError("Not built yet.")
        if 0 <= token_id < self._original_vocab_size:
            return int(self._lookup[token_id])
        raise ValueError(f"Token ID {token_id} out of range [0, {self._original_vocab_size})")

    def __repr__(self) -> str:
        if self._is_built:
            return (
                f"TokenizerCompression(vocab={self._original_vocab_size}->"
                f"{self._compressed_vocab_size}, "
                f"ratio={self.compression_ratio:.3f}, "
                f"classes={self.num_equivalence_classes})"
            )
        return "TokenizerCompression(not built)"


# ---------------------------------------------------------------------------
# LookupTableOps -- low-level operations on lookup tables
# ---------------------------------------------------------------------------

class LookupTableOps:
    """Low-level operations on int32 lookup tables.

    These are static methods used internally by TokenizerCompression and
    available for external inspection/debugging.
    """

    @staticmethod
    def identity_table(size: int) -> np.ndarray:
        """Create an identity lookup table (each ID maps to itself)."""
        return np.arange(size, dtype=np.int32)

    @staticmethod
    def compose(table_a: np.ndarray, table_b: np.ndarray) -> np.ndarray:
        """Compose two lookup tables: result[i] = table_b[table_a[i]].

        Parameters
        ----------
        table_a : ndarray
            First lookup table (applied first).
        table_b : ndarray
            Second lookup table (applied to the result of table_a).

        Returns
        -------
        ndarray
            Composed lookup table.
        """
        indices = np.clip(table_a, 0, len(table_b) - 1)
        return table_b[indices].astype(np.int32)

    @staticmethod
    def invert(table: np.ndarray) -> Dict[int, List[int]]:
        """Invert a lookup table: canonical_id -> list of original IDs.

        Returns
        -------
        dict
            Mapping from canonical ID to sorted list of original IDs.
        """
        inv: Dict[int, List[int]] = {}
        for orig_id, canon_id in enumerate(table):
            canon_id = int(canon_id)
            if canon_id not in inv:
                inv[canon_id] = []
            inv[canon_id].append(orig_id)
        for k in inv:
            inv[k].sort()
        return inv

    @staticmethod
    def is_identity(table: np.ndarray) -> bool:
        """Check if a lookup table is the identity mapping."""
        expected = np.arange(len(table), dtype=np.int32)
        return np.array_equal(table, expected)

    @staticmethod
    def fixed_points(table: np.ndarray) -> np.ndarray:
        """Return indices where table[i] == i (fixed points)."""
        indices = np.arange(len(table), dtype=np.int32)
        return np.where(table == indices)[0]

    @staticmethod
    def compression_ratio(table: np.ndarray) -> float:
        """Compute the compression ratio of a lookup table."""
        unique = len(np.unique(table))
        return unique / len(table) if len(table) > 0 else 1.0

    @staticmethod
    def validate_table(table: np.ndarray, vocab_size: int) -> List[str]:
        """Validate a lookup table for consistency.

        Returns a list of error messages (empty if valid).
        """
        errors = []
        if table.dtype != np.int32:
            errors.append(f"Expected int32 dtype, got {table.dtype}")
        if len(table) != vocab_size:
            errors.append(f"Table size {len(table)} != vocab_size {vocab_size}")
        if np.any(table < 0):
            errors.append(f"Negative values found in table")
        max_val = int(np.max(table)) if len(table) > 0 else -1
        if max_val >= vocab_size:
            errors.append(f"Max value {max_val} >= vocab_size {vocab_size}")
        return errors


# ---------------------------------------------------------------------------
# MockTokenizer -- for testing without transformers dependency
# ---------------------------------------------------------------------------

class MockTokenizer:
    """A simple tokenizer for testing tokenizer compression.

    Generates a vocabulary of ~1000 tokens including case variants, unicode
    variants, whitespace variants, and accented characters.  Provides
    ``decode(id) -> str``, ``vocab_size``, and special token attributes.

    Parameters
    ----------
    size : int
        Approximate vocabulary size.  Actual size may differ slightly due
        to how variants are generated.
    seed : int
        Random seed for reproducible vocabulary generation.
    """

    # Special token IDs
    PAD_ID: ClassVar[int] = 0
    BOS_ID: ClassVar[int] = 1
    EOS_ID: ClassVar[int] = 2
    UNK_ID: ClassVar[int] = 3

    def __init__(self, size: int = 1000, seed: int = 42) -> None:
        self._seed = seed
        self._vocab: Dict[int, str] = {}
        self._text_to_id: Dict[str, int] = {}
        self._size = size
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build a synthetic vocabulary with interesting compression properties."""
        rng = np.random.RandomState(self._seed)

        # Special tokens
        self._vocab[0] = "<pad>"
        self._vocab[1] = "<bos>"
        self._vocab[2] = "<eos>"
        self._vocab[3] = "<unk>"

        next_id = 4

        # Base words (lowercase)
        base_words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "need", "dare", "ought", "used", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "up", "about", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "because", "as", "until",
            "while", "although", "though", "if", "or", "and", "but",
            "nor", "yet", "hello", "world", "cat", "dog", "house",
            "tree", "water", "fire", "earth", "air", "sun", "moon",
            "star", "river", "mountain", "ocean", "forest", "city",
            "country", "people", "person", "child", "man", "woman",
            "time", "year", "day", "night", "morning", "evening",
            "food", "drink", "sleep", "walk", "run", "jump", "sit",
            "stand", "think", "know", "feel", "see", "hear", "speak",
            "write", "read", "learn", "teach", "work", "play", "live",
            "love", "hate", "want", "need", "give", "take", "make",
            "come", "go", "say", "tell", "ask", "try", "use", "find",
            "get", "put", "keep", "let", "begin", "seem", "help",
            "show", "turn", "move", "grow", "open", "close", "stop",
            "start", "end", "begin", "change", "follow", "lead",
        ]

        # Add lowercase versions
        for word in base_words:
            if next_id >= self._size:
                break
            self._vocab[next_id] = word
            next_id += 1

        # Add uppercase variants (should merge with lowercase)
        for word in base_words[:60]:
            if next_id >= self._size:
                break
            self._vocab[next_id] = word.upper()
            next_id += 1

        # Add title-case variants
        for word in base_words[:40]:
            if next_id >= self._size:
                break
            self._vocab[next_id] = word.capitalize()
            next_id += 1

        # Add leading-space variants (common in BPE tokenizers like GPT-2)
        for word in base_words[:50]:
            if next_id >= self._size:
                break
            self._vocab[next_id] = " " + word
            next_id += 1

        # Add trailing-space variants
        for word in base_words[:30]:
            if next_id >= self._size:
                break
            self._vocab[next_id] = word + " "
            next_id += 1

        # Add multi-space variants
        for word in base_words[:20]:
            if next_id >= self._size:
                break
            self._vocab[next_id] = "  " + word + "  "
            next_id += 1

        # Add unicode/accent variants
        accented_pairs = [
            ("cafe", "caf\u00e9"),          # e-acute
            ("naive", "na\u00efve"),         # i-diaeresis
            ("resume", "r\u00e9sum\u00e9"), # e-acute
            ("facade", "fa\u00e7ade"),       # c-cedilla
            ("cliche", "clich\u00e9"),       # e-acute
            ("role", "r\u00f4le"),           # o-circumflex
            ("fiance", "fianc\u00e9"),       # e-acute
            ("souffle", "souffl\u00e9"),     # e-acute
            ("entree", "entr\u00e9e"),       # e-acute
            ("puree", "pur\u00e9e"),         # e-acute
        ]
        for base, accented in accented_pairs:
            if next_id + 1 >= self._size:
                break
            self._vocab[next_id] = base
            next_id += 1
            self._vocab[next_id] = accented
            next_id += 1

        # Add fullwidth variants (NFKC should normalize these)
        fullwidth_map = str.maketrans(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            "\uff21\uff22\uff23\uff24\uff25\uff26\uff27\uff28\uff29\uff2a"
            "\uff2b\uff2c\uff2d\uff2e\uff2f\uff30\uff31\uff32\uff33\uff34"
            "\uff35\uff36\uff37\uff38\uff39\uff3a\uff41\uff42\uff43\uff44"
            "\uff45\uff46\uff47\uff48\uff49\uff4a\uff4b\uff4c\uff4d\uff4e"
            "\uff4f\uff50\uff51\uff52\uff53\uff54\uff55\uff56\uff57\uff58"
            "\uff59\uff5a\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17"
            "\uff18\uff19"
        )
        for word in base_words[:15]:
            if next_id >= self._size:
                break
            self._vocab[next_id] = word.translate(fullwidth_map)
            next_id += 1

        # Add ligature variants
        ligature_pairs = [
            ("fi", "\ufb01"),   # fi ligature
            ("fl", "\ufb02"),   # fl ligature
            ("ff", "\ufb00"),   # ff ligature
            ("ffi", "\ufb03"),  # ffi ligature
            ("ffl", "\ufb04"),  # ffl ligature
        ]
        for base, lig in ligature_pairs:
            if next_id + 1 >= self._size:
                break
            self._vocab[next_id] = base
            next_id += 1
            self._vocab[next_id] = lig
            next_id += 1

        # Add composed vs decomposed unicode
        composed_decomposed = [
            ("\u00e9", "e\u0301"),   # e-acute: composed vs decomposed
            ("\u00f1", "n\u0303"),   # n-tilde
            ("\u00fc", "u\u0308"),   # u-diaeresis
            ("\u00e7", "c\u0327"),   # c-cedilla
        ]
        for comp, decomp in composed_decomposed:
            if next_id + 1 >= self._size:
                break
            self._vocab[next_id] = comp
            next_id += 1
            self._vocab[next_id] = decomp
            next_id += 1

        # Fill remaining slots with generated tokens
        syllables = ["ba", "ka", "lo", "mi", "no", "pa", "ra", "si", "to", "zu",
                      "de", "fi", "go", "he", "ji", "ku", "le", "mo", "ne", "po"]
        while next_id < self._size:
            idx_a = rng.randint(0, len(syllables))
            idx_b = rng.randint(0, len(syllables))
            word = syllables[idx_a % len(syllables)] + syllables[idx_b % len(syllables)]
            # Add variation
            variant_type = rng.randint(0, 4)
            if variant_type == 0:
                self._vocab[next_id] = word
            elif variant_type == 1:
                self._vocab[next_id] = word.upper()
            elif variant_type == 2:
                self._vocab[next_id] = word.capitalize()
            else:
                self._vocab[next_id] = " " + word
            next_id += 1

        # Build reverse mapping
        self._text_to_id = {v: k for k, v in self._vocab.items()}
        self._actual_size = len(self._vocab)

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self._actual_size

    @property
    def pad_token_id(self) -> int:
        return self.PAD_ID

    @property
    def bos_token_id(self) -> int:
        return self.BOS_ID

    @property
    def eos_token_id(self) -> int:
        return self.EOS_ID

    @property
    def unk_token_id(self) -> int:
        return self.UNK_ID

    @property
    def all_special_ids(self) -> List[int]:
        return [self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID]

    @property
    def name_or_path(self) -> str:
        return f"mock-tokenizer-{self._actual_size}"

    def decode(self, ids: Union[int, List[int]]) -> str:
        """Decode token ID(s) to text."""
        if isinstance(ids, int):
            return self._vocab.get(ids, f"<id_{ids}>")
        elif isinstance(ids, list):
            if len(ids) == 1:
                return self._vocab.get(ids[0], f"<id_{ids[0]}>")
            return " ".join(self._vocab.get(i, f"<id_{i}>") for i in ids)
        raise TypeError(f"Expected int or list, got {type(ids)}")

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Return the token string for a given ID."""
        return self._vocab.get(token_id)

    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a {text: id} dictionary."""
        return dict(self._text_to_id)

    def __len__(self) -> int:
        return self._actual_size

    def __repr__(self) -> str:
        return f"MockTokenizer(size={self._actual_size})"


# ---------------------------------------------------------------------------
# Helper: make_default_compressor
# ---------------------------------------------------------------------------

def make_default_compressor(
    tokenizer: Any,
    *,
    strip_accents: bool = False,
    seed: int = 42,
) -> TokenizerCompression:
    """Convenience function to create and build a compressor with defaults.

    Parameters
    ----------
    tokenizer : object
        The tokenizer to compress.
    strip_accents : bool
        Whether to strip accents in normalization.
    seed : int
        Deterministic seed.

    Returns
    -------
    TokenizerCompression
        A fully built compressor ready for ``compress_ids``.
    """
    config = TokenizerCompressionConfig(
        strip_accents=strip_accents,
        seed=seed,
    )
    compressor = TokenizerCompression(config)
    compressor.build_from_tokenizer(tokenizer)
    return compressor


# ===========================================================================
# Self-tests
# ===========================================================================

def _run_tests() -> None:
    """Run comprehensive self-tests for tokenizer compression.

    Runs 35+ tests covering determinism, correctness, serialization,
    edge cases, and performance characteristics.
    """
    passed = 0
    failed = 0
    total = 0

    def _test(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  PASS  [{total:02d}] {name}")
        else:
            failed += 1
            msg = f"  FAIL  [{total:02d}] {name}"
            if detail:
                msg += f" -- {detail}"
            print(msg)

    print("=" * 72)
    print("Tokenizer Compression -- Self-Tests")
    print("=" * 72)

    # -- Setup ----------------------------------------------------------------
    tokenizer = MockTokenizer(size=1000, seed=42)
    config = TokenizerCompressionConfig(seed=42)

    # =========================================================================
    # 1. NormalizationPipeline tests
    # =========================================================================
    print("\n--- NormalizationPipeline ---")

    # Test 1: NFKC normalizes compatibility characters
    pipe = NormalizationPipeline(["nfkc"])
    _test(
        "NFKC normalizes fullwidth chars",
        pipe("\uff28\uff45\uff4c\uff4c\uff4f") == "Hello",
        f"got {pipe(chr(0xff28) + chr(0xff45) + chr(0xff4c) + chr(0xff4c) + chr(0xff4f))!r}",
    )

    # Test 2: NFKC normalizes ligatures
    _test(
        "NFKC normalizes fi ligature",
        pipe("\ufb01") == "fi",
        f"got {pipe(chr(0xfb01))!r}",
    )

    # Test 3: Lowercase works
    pipe_lower = NormalizationPipeline(["lowercase"])
    _test(
        "Lowercase converts uppercase",
        pipe_lower("HELLO World") == "hello world",
    )

    # Test 4: Casefold is more aggressive than lowercase
    pipe_fold = NormalizationPipeline(["casefold"])
    _test(
        "Casefold handles sharp-s",
        pipe_fold("\u00df") == "ss",
        f"got {pipe_fold(chr(0x00df))!r}",
    )

    # Test 5: Whitespace collapse
    pipe_ws = NormalizationPipeline(["collapse_whitespace", "strip_whitespace"])
    _test(
        "Whitespace collapse and strip",
        pipe_ws("  hello   world  ") == "hello world",
    )

    # Test 6: Strip accents
    pipe_acc = NormalizationPipeline(["strip_accents"])
    _test(
        "Strip accents removes diacritics",
        pipe_acc("caf\u00e9") == "cafe",
        f"got {pipe_acc('caf\u00e9')!r}",
    )

    # Test 7: Full default pipeline
    default_pipe = NormalizationPipeline()
    _test(
        "Default pipeline normalizes case+whitespace",
        default_pipe("  HELLO   World  ") == "hello world",
    )

    # Test 8: Pipeline determinism
    _test(
        "Pipeline is deterministic",
        default_pipe("Test\u00e9") == default_pipe("Test\u00e9"),
    )

    # Test 9: Recipe string round-trip
    recipe_str = default_pipe.to_recipe_string()
    reconstructed = NormalizationPipeline.from_recipe_string(recipe_str)
    _test(
        "Recipe string round-trip preserves pipeline",
        reconstructed == default_pipe,
        f"original={default_pipe.step_names}, reconstructed={reconstructed.step_names}",
    )

    # Test 10: Unknown step raises ValueError
    try:
        NormalizationPipeline(["nonexistent_step"])
        _test("Unknown step raises ValueError", False, "No exception raised")
    except ValueError:
        _test("Unknown step raises ValueError", True)

    # =========================================================================
    # 2. SpecialTokenPolicy tests
    # =========================================================================
    print("\n--- SpecialTokenPolicy ---")

    # Test 11: Policy from mock tokenizer
    policy = SpecialTokenPolicy.from_tokenizer(tokenizer)
    _test(
        "Auto-detect special tokens from mock tokenizer",
        policy.pad_id == 0 and policy.bos_id == 1 and policy.eos_id == 2 and policy.unk_id == 3,
        f"got pad={policy.pad_id}, bos={policy.bos_id}, eos={policy.eos_id}, unk={policy.unk_id}",
    )

    # Test 12: all_special_ids
    _test(
        "all_special_ids returns correct set",
        policy.all_special_ids == frozenset({0, 1, 2, 3}),
    )

    # Test 13: is_special
    _test(
        "is_special identifies special tokens",
        all(policy.is_special(i) for i in [0, 1, 2, 3]) and not policy.is_special(4),
    )

    # Test 14: Validate rejects out-of-range IDs
    bad_policy = SpecialTokenPolicy(pad_id=9999)
    try:
        bad_policy.validate(100)
        _test("Validate rejects out-of-range IDs", False)
    except ValueError:
        _test("Validate rejects out-of-range IDs", True)

    # Test 15: Policy serialization round-trip
    d = policy.to_dict()
    policy2 = SpecialTokenPolicy.from_dict(d)
    _test(
        "Policy serialization round-trip",
        policy == policy2,
    )

    # =========================================================================
    # 3. TokenizerCompression build tests
    # =========================================================================
    print("\n--- TokenizerCompression Build ---")

    # Test 16: Basic build succeeds
    compressor = TokenizerCompression(config)
    compressor.build_from_tokenizer(tokenizer)
    _test(
        "Basic build succeeds",
        compressor.is_built,
    )

    # Test 17: Compression happens (vocab reduction)
    _test(
        "Compression reduces vocabulary",
        compressor.compressed_vocab_size < compressor.original_vocab_size,
        f"original={compressor.original_vocab_size}, compressed={compressor.compressed_vocab_size}",
    )

    # Test 18: Compression ratio is between 0 and 1
    _test(
        "Compression ratio in (0, 1)",
        0 < compressor.compression_ratio < 1,
        f"ratio={compressor.compression_ratio:.4f}",
    )

    # Test 19: Stats are populated
    stats = compressor.stats
    _test(
        "Stats are populated",
        stats is not None and stats.original_vocab_size > 0,
    )

    # Test 20: Class sizes are reasonable
    _test(
        "Max class size >= 1",
        stats is not None and stats.max_class_size >= 1,
    )

    _test(
        "Mean class size >= 1",
        stats is not None and stats.mean_class_size >= 1.0,
    )

    # =========================================================================
    # 4. Determinism tests
    # =========================================================================
    print("\n--- Determinism ---")

    # Test 22: Build twice with same config produces identical tables
    config_a = TokenizerCompressionConfig(seed=42)
    comp_a = TokenizerCompression(config_a)
    comp_a.build_from_tokenizer(tokenizer)

    config_b = TokenizerCompressionConfig(seed=42)
    comp_b = TokenizerCompression(config_b)
    comp_b.build_from_tokenizer(tokenizer)

    _test(
        "Determinism: identical lookup tables from same config",
        np.array_equal(comp_a._lookup, comp_b._lookup),
    )

    # Test 23: Version hash determinism
    _test(
        "Version hash determinism",
        comp_a.version_hash == comp_b.version_hash,
        f"hash_a={comp_a.version_hash[:16]}, hash_b={comp_b.version_hash[:16]}",
    )

    # Test 24: Version hash sensitivity to recipe change
    config_diff = TokenizerCompressionConfig(
        normalization_recipe=["nfkc", "casefold", "strip_accents", "strip_whitespace"],
        seed=42,
    )
    comp_diff = TokenizerCompression(config_diff)
    comp_diff.build_from_tokenizer(tokenizer)
    _test(
        "Version hash sensitivity: different recipe -> different hash",
        comp_a.version_hash != comp_diff.version_hash,
    )

    # =========================================================================
    # 5. Special token invariance
    # =========================================================================
    print("\n--- Special Token Invariance ---")

    # Test 25: pad/bos/eos/unk map to themselves
    for name, sid in [("pad", 0), ("bos", 1), ("eos", 2), ("unk", 3)]:
        canonical = compressor.get_canonical_id(sid)
        _test(
            f"Special token {name} (id={sid}) maps to itself",
            canonical == sid,
            f"got canonical={canonical}",
        )

    # =========================================================================
    # 6. compress_ids tests
    # =========================================================================
    print("\n--- compress_ids ---")

    # Test 29: Shape preservation (1D)
    if HAS_TORCH:
        ids_1d = torch.tensor([4, 5, 6, 7, 8], dtype=torch.long)
        result_1d = compressor.compress_ids(ids_1d)
        _test(
            "Shape preservation: 1D tensor",
            result_1d.shape == ids_1d.shape,
            f"input={ids_1d.shape}, output={result_1d.shape}",
        )

        # Test 30: Shape preservation (2D batch)
        ids_2d = torch.tensor([[4, 5, 6], [7, 8, 9]], dtype=torch.long)
        result_2d = compressor.compress_ids(ids_2d)
        _test(
            "Shape preservation: 2D (B, T) tensor",
            result_2d.shape == ids_2d.shape,
            f"input={ids_2d.shape}, output={result_2d.shape}",
        )

        # Test 31: Shape preservation (3D)
        ids_3d = torch.tensor([[[4, 5], [6, 7]], [[8, 9], [10, 11]]], dtype=torch.long)
        result_3d = compressor.compress_ids(ids_3d)
        _test(
            "Shape preservation: 3D tensor",
            result_3d.shape == ids_3d.shape,
        )

        # Test 32: Device preservation (CPU)
        _test(
            "Device preservation: CPU tensor stays on CPU",
            result_1d.device == ids_1d.device,
        )

        # Test 33: Device preservation (CUDA if available)
        if torch.cuda.is_available():
            ids_cuda = ids_1d.cuda()
            result_cuda = compressor.compress_ids(ids_cuda)
            _test(
                "Device preservation: CUDA tensor stays on CUDA",
                result_cuda.is_cuda,
            )
        else:
            _test(
                "Device preservation: CUDA tensor stays on CUDA (SKIPPED - no CUDA)",
                True,
            )

        # Test 34: Empty tensor
        ids_empty = torch.tensor([], dtype=torch.long)
        result_empty = compressor.compress_ids(ids_empty)
        _test(
            "Empty tensor handled gracefully",
            result_empty.shape == ids_empty.shape and len(result_empty) == 0,
        )

        # Test 35: Single-token input
        ids_single = torch.tensor([42], dtype=torch.long)
        result_single = compressor.compress_ids(ids_single)
        _test(
            "Single-token input",
            result_single.shape == (1,),
        )
    else:
        # Torch not available -- test with numpy only
        for i in range(29, 36):
            _test(f"Torch test (SKIPPED - torch not available)", True)

    # Test 36: Numpy compatibility
    ids_np = np.array([4, 5, 6, 7, 8], dtype=np.int32)
    result_np = compressor.compress_ids(ids_np)
    _test(
        "Numpy compatibility: ndarray input",
        isinstance(result_np, np.ndarray) and result_np.shape == ids_np.shape,
    )

    # Test 37: Numpy batch
    ids_np_2d = np.array([[4, 5, 6], [7, 8, 9]], dtype=np.int32)
    result_np_2d = compressor.compress_ids(ids_np_2d)
    _test(
        "Numpy batch: (B, T) ndarray",
        isinstance(result_np_2d, np.ndarray) and result_np_2d.shape == ids_np_2d.shape,
    )

    # =========================================================================
    # 7. Serialization tests
    # =========================================================================
    print("\n--- Serialization ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "compression_table.tkcm")

        # Test 38: Serialize succeeds
        compressor.serialize(save_path)
        _test(
            "Serialize writes file",
            os.path.exists(save_path),
        )

        # Test 39: File has correct magic bytes
        with open(save_path, "rb") as f:
            magic = f.read(4)
        _test(
            "Serialized file has correct magic bytes",
            magic == _MAGIC_BYTES,
        )

        # Test 40: Load succeeds
        loaded = TokenizerCompression.load(save_path)
        _test(
            "Load succeeds and is_built",
            loaded.is_built,
        )

        # Test 41: Round-trip: identical lookup table
        _test(
            "Round-trip: identical lookup table",
            np.array_equal(compressor._lookup, loaded._lookup),
        )

        # Test 42: Round-trip: identical version hash
        _test(
            "Round-trip: identical version hash",
            compressor.version_hash == loaded.version_hash,
        )

        # Test 43: Round-trip: identical vocab sizes
        _test(
            "Round-trip: identical vocab sizes",
            compressor.original_vocab_size == loaded.original_vocab_size
            and compressor.compressed_vocab_size == loaded.compressed_vocab_size,
        )

        # Test 44: Round-trip: compress_ids gives same results
        if HAS_TORCH:
            test_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)
            r_orig = compressor.compress_ids(test_ids)
            r_loaded = loaded.compress_ids(test_ids)
            _test(
                "Round-trip: compress_ids identical results",
                torch.equal(r_orig, r_loaded),
            )
        else:
            test_ids_np = np.array([10, 20, 30, 40, 50], dtype=np.int32)
            r_orig = compressor.compress_ids(test_ids_np)
            r_loaded = loaded.compress_ids(test_ids_np)
            _test(
                "Round-trip: compress_ids identical results (numpy)",
                np.array_equal(r_orig, r_loaded),
            )

        # Test 45: Hash verification on load
        try:
            TokenizerCompression.load(save_path, verify_hash=True, expected_hash="wrong_hash")
            _test("Hash verification rejects wrong hash", False)
        except ValueError:
            _test("Hash verification rejects wrong hash", True)

        # Test 46: Hash verification passes with correct hash
        loaded_ok = TokenizerCompression.load(
            save_path,
            verify_hash=True,
            expected_hash=compressor.version_hash,
        )
        _test(
            "Hash verification passes with correct hash",
            loaded_ok.is_built,
        )

    # =========================================================================
    # 8. Fast path verification
    # =========================================================================
    print("\n--- Fast Path ---")

    # Test 47: compress_ids timing -- should be fast (no string ops)
    if HAS_TORCH:
        large_ids = torch.randint(0, tokenizer.vocab_size, (64, 512), dtype=torch.long)

        # Warm up
        _ = compressor.compress_ids(large_ids)

        start = time.monotonic()
        num_iters = 100
        for _ in range(num_iters):
            _ = compressor.compress_ids(large_ids)
        elapsed = time.monotonic() - start

        per_call_ms = (elapsed / num_iters) * 1000
        _test(
            f"Fast path: compress_ids < 10ms per (64x512) batch (actual: {per_call_ms:.2f}ms)",
            per_call_ms < 10.0,
            f"elapsed={per_call_ms:.2f}ms",
        )
    else:
        large_ids_np = np.random.randint(0, tokenizer.vocab_size, (64, 512), dtype=np.int32)
        _ = compressor.compress_ids(large_ids_np)
        start = time.monotonic()
        num_iters = 100
        for _ in range(num_iters):
            _ = compressor.compress_ids(large_ids_np)
        elapsed = time.monotonic() - start
        per_call_ms = (elapsed / num_iters) * 1000
        _test(
            f"Fast path (numpy): compress_ids < 10ms per (64x512) batch ({per_call_ms:.2f}ms)",
            per_call_ms < 10.0,
        )

    # =========================================================================
    # 9. Edge cases and advanced tests
    # =========================================================================
    print("\n--- Edge Cases ---")

    # Test 48: build_from_tokenizer not called -> compress_ids raises
    fresh = TokenizerCompression()
    try:
        if HAS_TORCH:
            fresh.compress_ids(torch.tensor([1, 2, 3]))
        else:
            fresh.compress_ids(np.array([1, 2, 3], dtype=np.int32))
        _test("compress_ids before build raises RuntimeError", False)
    except RuntimeError:
        _test("compress_ids before build raises RuntimeError", True)

    # Test 49: serialize before build raises
    try:
        fresh.serialize("/tmp/should_not_exist.tkcm")
        _test("serialize before build raises RuntimeError", False)
    except RuntimeError:
        _test("serialize before build raises RuntimeError", True)

    # Test 50: Equivalence class merging is correct
    # "the" (lowercase) and "THE" (uppercase) should map to the same canonical
    idx_lower = None
    idx_upper = None
    for tid, text in tokenizer._vocab.items():
        if text == "the":
            idx_lower = tid
        elif text == "THE":
            idx_upper = tid
    if idx_lower is not None and idx_upper is not None:
        canon_lower = compressor.get_canonical_id(idx_lower)
        canon_upper = compressor.get_canonical_id(idx_upper)
        _test(
            "Case variants merge: 'the' and 'THE' have same canonical",
            canon_lower == canon_upper,
            f"lower_id={idx_lower}->canon={canon_lower}, upper_id={idx_upper}->canon={canon_upper}",
        )
    else:
        _test(
            "Case variants merge (SKIPPED - variants not in vocab)",
            True,
        )

    # Test 51: Whitespace variants merge
    # Use "the" which is in base_words[:50] so " the" variant exists
    idx_plain_ws = None
    idx_spaced_ws = None
    for tid, text in tokenizer._vocab.items():
        if text == "the" and idx_plain_ws is None:
            idx_plain_ws = tid
        elif text == " the" and idx_spaced_ws is None:
            idx_spaced_ws = tid
    if idx_plain_ws is not None and idx_spaced_ws is not None:
        # With collapse_whitespace + strip_whitespace, " the" -> "the"
        canon_plain_ws = compressor.get_canonical_id(idx_plain_ws)
        canon_spaced_ws = compressor.get_canonical_id(idx_spaced_ws)
        _test(
            "Whitespace variants merge: 'the' and ' the'",
            canon_plain_ws == canon_spaced_ws,
            f"plain_id={idx_plain_ws}->canon={canon_plain_ws}, "
            f"spaced_id={idx_spaced_ws}->canon={canon_spaced_ws}",
        )
    else:
        _test(
            "Whitespace variants merge (SKIPPED - variants not in vocab)",
            True,
        )

    # Test 52: Out-of-range IDs are clamped
    if HAS_TORCH:
        oob_ids = torch.tensor([999999], dtype=torch.long)
        result_oob = compressor.compress_ids(oob_ids)
        _test(
            "Out-of-range IDs are clamped (no crash)",
            result_oob.shape == (1,),
        )
    else:
        oob_ids_np = np.array([999999], dtype=np.int32)
        result_oob = compressor.compress_ids(oob_ids_np)
        _test(
            "Out-of-range IDs are clamped (no crash, numpy)",
            result_oob.shape == (1,),
        )

    # Test 53: LookupTableOps.is_identity on identity table
    ident = LookupTableOps.identity_table(100)
    _test(
        "LookupTableOps.is_identity detects identity",
        LookupTableOps.is_identity(ident),
    )

    # Test 54: Compressor table is NOT identity (compression happened)
    _test(
        "Built table is NOT identity",
        not LookupTableOps.is_identity(compressor._lookup),
    )

    # Test 55: LookupTableOps.compression_ratio matches
    table_ratio = LookupTableOps.compression_ratio(compressor._lookup)
    _test(
        "LookupTableOps.compression_ratio consistent",
        abs(table_ratio - compressor.compression_ratio) < 0.01,
        f"ops={table_ratio:.4f}, prop={compressor.compression_ratio:.4f}",
    )

    # Test 56: LookupTableOps.validate_table passes for built table
    errors = LookupTableOps.validate_table(compressor._lookup, compressor.original_vocab_size)
    _test(
        "LookupTableOps.validate_table passes for built table",
        len(errors) == 0,
        f"errors: {errors}" if errors else "",
    )

    # Test 57: LookupTableOps.fixed_points includes special tokens
    fps = LookupTableOps.fixed_points(compressor._lookup)
    _test(
        "Fixed points include special tokens",
        all(sid in fps for sid in [0, 1, 2, 3]),
    )

    # =========================================================================
    # 10. Equivalence class lookup tests
    # =========================================================================
    print("\n--- Equivalence Class Lookup ---")

    # Test: get_equivalence_class works for a known token
    ec_found = compressor.get_equivalence_class(4)
    _test(
        "get_equivalence_class returns a class for token 4",
        ec_found is not None and 4 in ec_found.member_ids,
    )

    # Test: get_equivalence_class returns None for out-of-range
    ec_missing = compressor.get_equivalence_class(999999)
    _test(
        "get_equivalence_class returns None for unknown token",
        ec_missing is None,
    )

    # =========================================================================
    # 11. LookupTableOps.compose test
    # =========================================================================
    print("\n--- LookupTableOps.compose ---")

    # Test: composing identity with any table yields the same table
    ident_small = LookupTableOps.identity_table(compressor.original_vocab_size)
    composed = LookupTableOps.compose(ident_small, compressor._lookup)
    _test(
        "Compose identity with table yields same table",
        np.array_equal(composed, compressor._lookup),
    )

    # =========================================================================
    # 12. LookupTableOps.invert test
    # =========================================================================
    print("\n--- LookupTableOps.invert ---")

    inverted = LookupTableOps.invert(compressor._lookup)
    # Every canonical ID should map back to at least one original ID
    _test(
        "Inverted table has entries for all canonical IDs",
        len(inverted) == compressor.compressed_vocab_size,
        f"inverted keys={len(inverted)}, compressed={compressor.compressed_vocab_size}",
    )

    # =========================================================================
    # 13. Recipe comparison via multiple builds
    # =========================================================================
    print("\n--- Recipe Comparison ---")

    # Build with minimal recipe
    config_minimal = TokenizerCompressionConfig(
        normalization_recipe=["nfkc"],
        seed=42,
    )
    comp_minimal = TokenizerCompression(config_minimal)
    comp_minimal.build_from_tokenizer(tokenizer)

    # Build with aggressive recipe
    config_aggressive = TokenizerCompressionConfig(
        normalization_recipe=["nfkc", "casefold", "strip_accents", "collapse_whitespace", "strip_whitespace"],
        seed=42,
    )
    comp_aggressive = TokenizerCompression(config_aggressive)
    comp_aggressive.build_from_tokenizer(tokenizer)

    _test(
        "More aggressive recipe yields more compression",
        comp_aggressive.compression_ratio <= comp_minimal.compression_ratio,
        f"aggressive={comp_aggressive.compression_ratio:.4f}, "
        f"minimal={comp_minimal.compression_ratio:.4f}",
    )

    # =========================================================================
    # 16. Config flag tests
    # =========================================================================
    print("\n--- Config Flags ---")

    # Test 70: strip_accents flag
    acc_config = TokenizerCompressionConfig(strip_accents=True, seed=42)
    acc_pipe = acc_config.build_normalization_pipeline()
    _test(
        "strip_accents flag adds strip_accents step",
        "strip_accents" in acc_pipe.step_names,
    )

    # Test 71: use_casefold flag replaces lowercase
    cf_config = TokenizerCompressionConfig(use_casefold=True, seed=42)
    cf_pipe = cf_config.build_normalization_pipeline()
    _test(
        "use_casefold replaces lowercase with casefold",
        "casefold" in cf_pipe.step_names and "lowercase" not in cf_pipe.step_names,
    )

    # Test 72: CompressionStats summary is non-empty
    _test(
        "CompressionStats.summary() produces output",
        len(compressor.stats.summary()) > 100,
    )

    # =========================================================================
    # 17. Torch int dtype tests
    # =========================================================================
    if HAS_TORCH:
        print("\n--- Torch dtype handling ---")

        # Test 73: int32 input
        ids_i32 = torch.tensor([4, 5, 6], dtype=torch.int32)
        result_i32 = compressor.compress_ids(ids_i32)
        _test(
            "int32 input preserves dtype",
            result_i32.dtype == torch.int32,
            f"got dtype={result_i32.dtype}",
        )

        # Test 74: int64 input
        ids_i64 = torch.tensor([4, 5, 6], dtype=torch.int64)
        result_i64 = compressor.compress_ids(ids_i64)
        _test(
            "int64 input preserves dtype",
            result_i64.dtype == torch.int64,
            f"got dtype={result_i64.dtype}",
        )

    # =========================================================================
    # 18. MockTokenizer sanity
    # =========================================================================
    print("\n--- MockTokenizer ---")

    _test(
        "MockTokenizer vocab_size > 0",
        tokenizer.vocab_size > 0,
        f"size={tokenizer.vocab_size}",
    )

    _test(
        "MockTokenizer decode round-trips for special tokens",
        tokenizer.decode(0) == "<pad>" and tokenizer.decode(1) == "<bos>",
    )

    _test(
        "MockTokenizer has case variants",
        any(tokenizer.decode(i).isupper() for i in range(4, min(200, tokenizer.vocab_size))),
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 72)
    print(f"Results: {passed} passed, {failed} failed, {total} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"WARNING: {failed} test(s) FAILED")
    print("=" * 72)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    _run_tests()
