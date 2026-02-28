"""
Engram Conditional Memory -- Configuration Template.

Comprehensive configuration dataclasses for the Engram subsystem, covering:
  - EngramConfig:               Core N-gram hash embedding parameters
  - HashConfig:                 Deterministic hash function configuration
  - OffloadConfig:              CPU offload and async prefetch controls
  - GatingConfig:               Context-aware gating (RMSNorm q/k, conv)
  - TokenizerCompressionConfig: Normalization recipe and equivalence merging
  - EngramEncoderConfig:        Phase 1 -- Engram as workspace-competing encoder
  - EngramLayerConfig:          Phase 2 -- Engram injected at backbone layers
  - EngramFullConfig:           Aggregated config with presets and serialization

Design invariants:
  1. All hash indices are deterministic given the same seed, config, and input.
  2. Gating outputs are bounded [0, 1] and AMP-safe (no NaN under mixed precision).
  3. CPU offload mode operates without deadlocks; prefetch overlaps compute.
  4. Tokenizer compression is deterministic and versioned.
  5. Serialization is JSON-compatible (no torch types in serialized form).

Usage:
    from engram_config_template import EngramFullConfig

    # Quick preset for unit tests
    cfg = EngramFullConfig.minimal()
    cfg.validate()

    # Production inference with CPU offload
    cfg = EngramFullConfig.production()
    cfg.validate()

    # Round-trip serialization
    d = cfg.to_dict()
    cfg2 = EngramFullConfig.from_dict(d)
    assert cfg == cfg2

    # JSON persistence
    cfg.save_json("engram_config.json")
    cfg3 = EngramFullConfig.load_json("engram_config.json")
    assert cfg == cfg3
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_VERSION: str = "1.0.0"
"""Serialization format version for migration compatibility."""

_ALLOWED_HASH_FNS: frozenset = frozenset({"mult_xor"})
"""Hash function families implemented.  Extend as new families land."""

_ALLOWED_GATE_TYPES: frozenset = frozenset({"scalar", "per_head"})
"""Gate output shapes.  scalar -> (B,T,1);  per_head -> (B,T,H)."""

_ALLOWED_ACTIVATIONS: frozenset = frozenset({"silu", "gelu"})
"""Post-convolution activation functions."""

_ALLOWED_STORAGE_DTYPES: frozenset = frozenset({"float16", "bfloat16", "float32"})
"""Host-side storage precisions for offloaded embeddings."""

_ALLOWED_COMPUTE_DTYPES: frozenset = frozenset({"float16", "bfloat16", "float32"})
"""Device-side compute precisions."""

_ALLOWED_NORM_STEPS: frozenset = frozenset({
    "nfkc", "nfkd", "nfc", "nfd",
    "lowercase", "strip_whitespace",
    "strip_accents", "collapse_unicode",
})
"""Normalization recipe steps for tokenizer compression."""

_ALLOWED_SPECIAL_TOKEN_POLICIES: frozenset = frozenset({
    "preserve_all", "preserve_standard",
})
"""How to handle special tokens during compression."""

_ALLOWED_CANONICAL_SELECTIONS: frozenset = frozenset({
    "lowest_id", "most_frequent",
})
"""Strategy for choosing canonical ID from an equivalence class."""

# Primes useful for table sizing -- small list for preset reference.
_PRIME_TABLE_SIZES: Dict[str, int] = {
    "tiny":       521,
    "minimal":    1021,
    "small":      8191,
    "medium":     65537,
    "large":      131071,
    "xlarge":     524287,
    "production": 10_000_019,
}

T = TypeVar("T")

# Registry mapping class names to their types, populated at module load.
# Used by from_dict() to resolve nested config types without eval().
_CONFIG_CLASS_REGISTRY: Dict[str, type] = {}


def _register_config_class(cls: type) -> type:
    """Register a config class in the global registry for from_dict resolution."""
    _CONFIG_CLASS_REGISTRY[cls.__name__] = cls
    return cls


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _is_prime(n: int) -> bool:
    """Deterministic primality check for moderate-sized integers."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _next_prime(n: int) -> int:
    """Return *n* if prime, else the next prime above *n*."""
    if n <= 2:
        return 2
    candidate = n if n % 2 != 0 else n + 1
    while not _is_prime(candidate):
        candidate += 2
    return candidate


def _dtype_name_to_torch(name: str) -> Any:
    """
    Map a string dtype name to a ``torch.dtype``.

    Returns the torch dtype object, or *None* if torch is not available.
    This avoids a hard dependency on torch at import time.
    """
    try:
        import torch
    except ImportError:
        return None
    _map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return _map.get(name)


def _resolve_field_type(ftype: Any) -> Any:
    """
    Resolve a dataclass field type annotation to an actual type.

    Handles both direct types and string forward-references by looking them
    up in the config class registry.  This avoids any use of eval().
    """
    if isinstance(ftype, type):
        return ftype
    if isinstance(ftype, str):
        # Strip Optional[], etc. -- we only care about config classes
        stripped = ftype.strip()
        # Try direct lookup in our registry
        if stripped in _CONFIG_CLASS_REGISTRY:
            return _CONFIG_CLASS_REGISTRY[stripped]
        # Handle quoted names like 'EngramConfig'
        stripped_unquoted = stripped.strip("'\"")
        if stripped_unquoted in _CONFIG_CLASS_REGISTRY:
            return _CONFIG_CLASS_REGISTRY[stripped_unquoted]
    return ftype


# ---------------------------------------------------------------------------
# Mixin: validation + serialization
# ---------------------------------------------------------------------------

class _ConfigMixin:
    """
    Shared behaviour for all Engram config dataclasses.

    Subclasses must be ``@dataclass`` instances.  This mixin adds:
      - ``validate()`` -- raises ``ValueError`` on constraint violations.
      - ``to_dict()`` / ``from_dict()`` -- JSON-safe serialization.
      - ``save_json()`` / ``load_json()`` -- file-level convenience.
    """

    # -- Validation --------------------------------------------------------

    def validate(self) -> None:
        """
        Validate all fields.

        Subclasses override ``_validate_fields()`` to add domain checks.
        This method always calls ``_validate_fields()`` and is the public API.
        """
        self._validate_fields()

    def _validate_fields(self) -> None:
        """Override in each config to check field constraints."""
        pass  # base: nothing to validate

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this config to a JSON-compatible dictionary.

        Nested config objects are recursively serialized.  Tuples become lists
        in JSON, but ``from_dict()`` reconstructs the correct types.
        """
        result: Dict[str, Any] = {"__version__": _CONFIG_VERSION}
        for f in fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            if isinstance(val, _ConfigMixin):
                result[f.name] = val.to_dict()
            elif isinstance(val, tuple):
                result[f.name] = list(val)
            else:
                result[f.name] = copy.deepcopy(val)
        return result

    @classmethod
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        """
        Reconstruct a config from a dictionary produced by ``to_dict()``.

        Unknown keys (including ``__version__``) are silently dropped so that
        forward-compatible loading works when new fields are added.
        """
        # Strip meta keys
        d = {k: v for k, v in d.items() if not k.startswith("__")}

        # Identify nested config fields and their types
        init_kwargs: Dict[str, Any] = {}
        for f in fields(cls):  # type: ignore[arg-type]
            if f.name not in d:
                continue
            val = d[f.name]

            # Resolve the field type safely via registry lookup
            ftype = _resolve_field_type(f.type)

            if isinstance(ftype, type) and issubclass(ftype, _ConfigMixin):
                init_kwargs[f.name] = ftype.from_dict(val)
            elif isinstance(val, list) and f.name in _TUPLE_FIELDS:
                init_kwargs[f.name] = tuple(val)
            else:
                init_kwargs[f.name] = val
        return cls(**init_kwargs)

    # -- File I/O ----------------------------------------------------------

    def save_json(self, path: Union[str, Path]) -> None:
        """Write this config to a JSON file at *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls: Type[T], path: Union[str, Path]) -> T:
        """Read a config from a JSON file at *path*."""
        with open(path, "r", encoding="utf-8") as fp:
            return cls.from_dict(json.load(fp))

    # -- Equality (fallback when __eq__ is not auto-generated) -------------

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        for f in fields(self):  # type: ignore[arg-type]
            if getattr(self, f.name) != getattr(other, f.name):
                return False
        return True

    def __repr__(self) -> str:
        parts = []
        for f in fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            parts.append(f"{f.name}={val!r}")
        cls_name = type(self).__name__
        return f"{cls_name}({', '.join(parts)})"


# Set of field names that are tuples in the Python dataclass but become
# lists in JSON.  ``from_dict`` converts them back.
_TUPLE_FIELDS: frozenset = frozenset({
    "ngram_orders",
})


# =========================================================================
# 1. EngramConfig
# =========================================================================

@_register_config_class
@dataclass
class EngramConfig(_ConfigMixin):
    """
    Core configuration for the Engram N-gram hash embedding module.

    Controls the N-gram suffix orders, per-head embedding dimensionality,
    backbone hidden dimension for gating projections, and miscellaneous
    switches for convolution, gating, and tokenizer compression.

    Attributes:
        max_ngram_order:       Maximum N-gram suffix length *K*.  Orders
                               ``2 .. K`` are used, so ``K >= 2`` is required.
        embedding_dim:         Per-head embedding dimension.  Each hash head
                               retrieves a vector of this size.
        num_heads_per_order:   Number of hash heads per N-gram order.
                               More heads reduce collision impact.
        hidden_dim:            Backbone hidden dimension used for gating q/k
                               projections.  Must match the host transformer's
                               hidden dim when used in layer-augmentation mode.
        gate_dim:              Projection dimension for gating query/key.
        pad_id:                Padding token ID (used to mask N-gram extraction).
        use_tokenizer_compression:  Enable surjective vocab compression.
        use_context_gate:      Enable context-aware gating (vs. always-on).
        use_depthwise_conv:    Apply depthwise causal conv before fusion.
        conv_kernel_size:      Kernel size for the causal depthwise conv.
    """

    max_ngram_order: int = 4
    embedding_dim: int = 256
    num_heads_per_order: int = 2
    hidden_dim: int = 512
    gate_dim: int = 128
    pad_id: int = 0
    use_tokenizer_compression: bool = True
    use_context_gate: bool = True
    use_depthwise_conv: bool = True
    conv_kernel_size: int = 4

    # -- Computed properties -----------------------------------------------

    @property
    def total_heads(self) -> int:
        """Total number of hash heads across all N-gram orders.

        With ``max_ngram_order = K`` and ``num_heads_per_order = H``,
        the orders used are ``2, 3, ..., K``  (i.e. ``K - 1`` orders),
        giving ``H * (K - 1)`` total heads.
        """
        return self.num_heads_per_order * (self.max_ngram_order - 1)

    @property
    def ngram_orders(self) -> Tuple[int, ...]:
        """Tuple of N-gram orders derived from *max_ngram_order*.

        Example: ``max_ngram_order=4`` -> ``(2, 3, 4)``.
        """
        return tuple(range(2, self.max_ngram_order + 1))

    @property
    def total_embedding_dim(self) -> int:
        """Total retrieved embedding width (all heads concatenated)."""
        return self.total_heads * self.embedding_dim

    @property
    def dim_per_head(self) -> int:
        """Embedding dimension per individual head (same as embedding_dim)."""
        return self.embedding_dim

    # -- Validation --------------------------------------------------------

    def _validate_fields(self) -> None:
        if self.max_ngram_order < 2:
            raise ValueError(
                f"max_ngram_order must be >= 2, got {self.max_ngram_order}"
            )
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be > 0, got {self.embedding_dim}"
            )
        if self.num_heads_per_order < 1:
            raise ValueError(
                f"num_heads_per_order must be >= 1, got {self.num_heads_per_order}"
            )
        if self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {self.hidden_dim}"
            )
        if self.gate_dim <= 0:
            raise ValueError(
                f"gate_dim must be > 0, got {self.gate_dim}"
            )
        if self.pad_id < 0:
            raise ValueError(
                f"pad_id must be >= 0, got {self.pad_id}"
            )
        if self.conv_kernel_size < 1:
            raise ValueError(
                f"conv_kernel_size must be >= 1, got {self.conv_kernel_size}"
            )


# =========================================================================
# 2. HashConfig
# =========================================================================

@_register_config_class
@dataclass
class HashConfig(_ConfigMixin):
    """
    Deterministic hash function configuration.

    Controls the hash function family, table sizing, per-layer salting, and
    the deterministic seed.  Only ``mult_xor`` is supported initially;
    extending to other families requires adding to ``_ALLOWED_HASH_FNS``.

    The ``mult_xor`` family computes::

        h_k(x_1, ..., x_n) = ((a_{k,1} * x_1) ^ (a_{k,2} * x_2) ^ ...) % M

    where ``a_{k,i}`` are random primes derived from the seed, and ``M`` is
    the table size.

    Attributes:
        hash_fn:           Hash function family name.
        use_prime_sizes:   Whether to enforce prime table sizes (improves
                           distribution uniformity).
        table_size:        Base hash table size (per head).  When
                           ``use_prime_sizes`` is True, this is snapped to
                           the next prime >= the given value.
        per_layer_salt:    Use different hash salts at each insertion layer
                           to decorrelate collisions across layers.
        seed:              Deterministic seed for coefficient generation.
    """

    hash_fn: str = "mult_xor"
    use_prime_sizes: bool = True
    table_size: int = 131071
    per_layer_salt: bool = True
    seed: int = 42

    # -- Computed properties -----------------------------------------------

    @property
    def effective_table_size(self) -> int:
        """Actual table size after optional prime rounding."""
        if self.use_prime_sizes:
            return _next_prime(self.table_size)
        return self.table_size

    # -- Validation --------------------------------------------------------

    def _validate_fields(self) -> None:
        if self.table_size <= 0:
            raise ValueError(
                f"table_size must be > 0, got {self.table_size}"
            )
        if self.seed < 0:
            raise ValueError(
                f"seed must be >= 0, got {self.seed}"
            )
        if self.hash_fn not in _ALLOWED_HASH_FNS:
            raise ValueError(
                f"hash_fn must be one of {sorted(_ALLOWED_HASH_FNS)}, "
                f"got {self.hash_fn!r}"
            )


# =========================================================================
# 3. OffloadConfig
# =========================================================================

@_register_config_class
@dataclass
class OffloadConfig(_ConfigMixin):
    """
    CPU offload and async prefetch configuration.

    When ``weights_on_cpu`` is True, embedding tables are stored in pinned
    host memory and transferred to the compute device on demand.  Enabling
    ``use_async_prefetch`` schedules transfers ``prefetch_ahead_layers``
    layers in advance using CUDA streams, overlapping data movement with
    computation.

    Attributes:
        weights_on_cpu:         Store embedding weights in CPU pinned memory.
        use_async_prefetch:     Enable async D2H / H2D prefetch (requires
                                ``weights_on_cpu = True``).
        prefetch_ahead_layers:  How many layers ahead to initiate prefetch.
        pin_memory:             Use ``torch.cuda.pin_memory()`` for host
                                buffers (ignored when offload is disabled).
        storage_dtype:          Host-side storage precision.
        compute_dtype:          Device-side compute precision.
        prefetch_timeout_ms:    Timeout in milliseconds for waiting on a
                                prefetch CUDA event before falling back to
                                synchronous copy.
    """

    weights_on_cpu: bool = False
    use_async_prefetch: bool = False
    prefetch_ahead_layers: int = 2
    pin_memory: bool = True
    storage_dtype: str = "float16"
    compute_dtype: str = "float32"
    prefetch_timeout_ms: int = 5000

    # -- Computed properties -----------------------------------------------

    @property
    def storage_torch_dtype(self) -> Any:
        """``torch.dtype`` for host storage, or None if torch unavailable."""
        return _dtype_name_to_torch(self.storage_dtype)

    @property
    def compute_torch_dtype(self) -> Any:
        """``torch.dtype`` for device compute, or None if torch unavailable."""
        return _dtype_name_to_torch(self.compute_dtype)

    # -- Validation --------------------------------------------------------

    def _validate_fields(self) -> None:
        if self.use_async_prefetch and not self.weights_on_cpu:
            raise ValueError(
                "use_async_prefetch=True requires weights_on_cpu=True.  "
                "Async prefetch only applies to CPU-offloaded embeddings."
            )
        if self.prefetch_ahead_layers < 1:
            raise ValueError(
                f"prefetch_ahead_layers must be >= 1, got "
                f"{self.prefetch_ahead_layers}"
            )
        if self.storage_dtype not in _ALLOWED_STORAGE_DTYPES:
            raise ValueError(
                f"storage_dtype must be one of {sorted(_ALLOWED_STORAGE_DTYPES)}, "
                f"got {self.storage_dtype!r}"
            )
        if self.compute_dtype not in _ALLOWED_COMPUTE_DTYPES:
            raise ValueError(
                f"compute_dtype must be one of {sorted(_ALLOWED_COMPUTE_DTYPES)}, "
                f"got {self.compute_dtype!r}"
            )
        if self.prefetch_timeout_ms <= 0:
            raise ValueError(
                f"prefetch_timeout_ms must be > 0, got {self.prefetch_timeout_ms}"
            )
        if self.pin_memory is not True and self.pin_memory is not False:
            raise ValueError(
                f"pin_memory must be bool, got {type(self.pin_memory).__name__}"
            )


# =========================================================================
# 4. GatingConfig
# =========================================================================

@_register_config_class
@dataclass
class GatingConfig(_ConfigMixin):
    """
    Context-aware gating configuration.

    Controls how the backbone hidden state modulates Engram output.  The
    gating mechanism projects the hidden state through RMSNorm'd query/key
    heads, computes a scalar (or per-head) gate in [0, 1], and scales the
    retrieved embedding before residual fusion.

    An optional depthwise causal convolution provides local context mixing
    before the gating decision.

    Attributes:
        gate_type:        ``"scalar"`` produces a single gate per position;
                          ``"per_head"`` produces one gate per hash head.
        use_rmsnorm:      Apply RMSNorm to q/k projections for stability.
        rmsnorm_eps:      Epsilon for RMSNorm denominator.
        gate_init_bias:   Initial bias for the gate logit.  Negative values
                          (e.g. -2.0) make the gate conservative at init,
                          letting the backbone dominate early in training.
        activation:       Post-convolution activation (``"silu"`` or ``"gelu"``).
        conv_dilation:    Dilation factor for causal depthwise conv.  Typically
                          tied to ``max_ngram_order`` for matched receptive field.
        residual_scale:   Multiplicative scale applied to the gated output
                          before residual addition.
    """

    gate_type: str = "scalar"
    use_rmsnorm: bool = True
    rmsnorm_eps: float = 1e-6
    gate_init_bias: float = -2.0
    activation: str = "silu"
    conv_dilation: int = 4
    residual_scale: float = 1.0

    # -- Validation --------------------------------------------------------

    def _validate_fields(self) -> None:
        if self.gate_type not in _ALLOWED_GATE_TYPES:
            raise ValueError(
                f"gate_type must be one of {sorted(_ALLOWED_GATE_TYPES)}, "
                f"got {self.gate_type!r}"
            )
        if not math.isfinite(self.gate_init_bias):
            raise ValueError(
                f"gate_init_bias must be finite, got {self.gate_init_bias}"
            )
        if self.activation not in _ALLOWED_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ALLOWED_ACTIVATIONS)}, "
                f"got {self.activation!r}"
            )
        if self.rmsnorm_eps <= 0:
            raise ValueError(
                f"rmsnorm_eps must be > 0, got {self.rmsnorm_eps}"
            )
        if self.conv_dilation < 1:
            raise ValueError(
                f"conv_dilation must be >= 1, got {self.conv_dilation}"
            )
        if self.residual_scale <= 0:
            raise ValueError(
                f"residual_scale must be > 0, got {self.residual_scale}"
            )


# =========================================================================
# 5. TokenizerCompressionConfig
# =========================================================================

@_register_config_class
@dataclass
class TokenizerCompressionConfig(_ConfigMixin):
    """
    Tokenizer compression (equivalence merging) configuration.

    Surjective mapping that collapses textually equivalent tokens into
    canonical IDs.  A normalization recipe (e.g. NFKC + lowercasing)
    reduces effective vocabulary by ~23% for 128k tokenizers.

    Special tokens (pad, bos, eos, unk) are invariant under all policies.

    Attributes:
        normalization_recipe:   Ordered list of normalization steps to apply
                                to each token's surface text.  Valid steps:
                                ``nfkc``, ``nfkd``, ``nfc``, ``nfd``,
                                ``lowercase``, ``strip_whitespace``,
                                ``strip_accents``, ``collapse_unicode``.
        special_token_policy:   ``"preserve_all"`` keeps every token marked
                                as special by the tokenizer.
                                ``"preserve_standard"`` keeps only pad, bos,
                                eos, unk and compresses the rest.
        additional_special_ids: Extra token IDs to treat as special (never
                                compressed), beyond those detected by the
                                tokenizer.
        canonical_selection:    Strategy for picking the canonical ID from an
                                equivalence class.  ``"lowest_id"`` picks the
                                numerically smallest; ``"most_frequent"``
                                picks the most frequent in a reference corpus.
    """

    normalization_recipe: List[str] = field(
        default_factory=lambda: ["nfkc", "lowercase", "strip_whitespace"]
    )
    special_token_policy: str = "preserve_all"
    additional_special_ids: List[int] = field(default_factory=list)
    canonical_selection: str = "lowest_id"

    # -- Validation --------------------------------------------------------

    def _validate_fields(self) -> None:
        if not isinstance(self.normalization_recipe, list):
            raise ValueError(
                f"normalization_recipe must be a list, got "
                f"{type(self.normalization_recipe).__name__}"
            )
        for step in self.normalization_recipe:
            if not isinstance(step, str):
                raise ValueError(
                    f"Each normalization_recipe entry must be a str, "
                    f"got {type(step).__name__}: {step!r}"
                )
            if step not in _ALLOWED_NORM_STEPS:
                raise ValueError(
                    f"Unknown normalization step {step!r}.  "
                    f"Allowed: {sorted(_ALLOWED_NORM_STEPS)}"
                )
        if self.special_token_policy not in _ALLOWED_SPECIAL_TOKEN_POLICIES:
            raise ValueError(
                f"special_token_policy must be one of "
                f"{sorted(_ALLOWED_SPECIAL_TOKEN_POLICIES)}, "
                f"got {self.special_token_policy!r}"
            )
        if self.canonical_selection not in _ALLOWED_CANONICAL_SELECTIONS:
            raise ValueError(
                f"canonical_selection must be one of "
                f"{sorted(_ALLOWED_CANONICAL_SELECTIONS)}, "
                f"got {self.canonical_selection!r}"
            )
        if not isinstance(self.additional_special_ids, list):
            raise ValueError(
                f"additional_special_ids must be a list, got "
                f"{type(self.additional_special_ids).__name__}"
            )
        for sid in self.additional_special_ids:
            if not isinstance(sid, int):
                raise ValueError(
                    f"Each additional_special_id must be an int, "
                    f"got {type(sid).__name__}: {sid!r}"
                )
            if sid < 0:
                raise ValueError(
                    f"additional_special_ids entries must be >= 0, got {sid}"
                )


# =========================================================================
# 6. EngramEncoderConfig  (Phase 1)
# =========================================================================

@_register_config_class
@dataclass
class EngramEncoderConfig(_ConfigMixin):
    """
    Phase 1 configuration: Engram as a workspace-competing text encoder.

    In Phase 1, an ``EngramTextEncoder`` competes with the standard
    transformer-based text encoder for workspace attention.  The encoder
    maps token IDs through N-gram hash lookup, adds positional encoding,
    pools, and projects to ``workspace_dim``.

    Attributes:
        vocab_size:      Full vocabulary size (before compression).
        embed_dim:       Per-head embedding dimension in the Engram module.
        workspace_dim:   Output dimension matching other workspace encoders.
        engram:          Core Engram N-gram parameters.
        hash:            Hash function parameters.
        compression:     Tokenizer compression parameters.
    """

    vocab_size: int = 32000
    embed_dim: int = 256
    workspace_dim: int = 4096
    engram: EngramConfig = field(default_factory=EngramConfig)
    hash: HashConfig = field(default_factory=HashConfig)
    compression: TokenizerCompressionConfig = field(
        default_factory=TokenizerCompressionConfig
    )

    # -- Validation --------------------------------------------------------

    def _validate_fields(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be > 0, got {self.vocab_size}"
            )
        if self.embed_dim <= 0:
            raise ValueError(
                f"embed_dim must be > 0, got {self.embed_dim}"
            )
        if self.workspace_dim <= 0:
            raise ValueError(
                f"workspace_dim must be > 0, got {self.workspace_dim}"
            )
        # Validate nested configs
        self.engram.validate()
        self.hash.validate()
        self.compression.validate()


# =========================================================================
# 7. EngramLayerConfig  (Phase 2)
# =========================================================================

@_register_config_class
@dataclass
class EngramLayerConfig(_ConfigMixin):
    """
    Phase 2 configuration: Engram injected at selected backbone layers.

    In Phase 2, ``EngramAugmentedLayer`` modules are inserted at specific
    backbone layers (identified by ``insertion_layers``).  Each augmented
    layer adds an Engram retrieval + gating step before the standard
    attention + FFN blocks.

    Attributes:
        insertion_layers:  Zero-indexed backbone layer indices where Engram
                           augmentation is applied.
        engram:            Core Engram N-gram parameters.
        hash:              Hash function parameters.
        offload:           CPU offload and prefetch parameters.
        gating:            Context-aware gating parameters.
        compression:       Tokenizer compression parameters.
    """

    insertion_layers: List[int] = field(
        default_factory=lambda: [4, 8, 12, 16]
    )
    engram: EngramConfig = field(default_factory=EngramConfig)
    hash: HashConfig = field(default_factory=HashConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    compression: TokenizerCompressionConfig = field(
        default_factory=TokenizerCompressionConfig
    )

    # -- Computed properties -----------------------------------------------

    @property
    def num_insertion_layers(self) -> int:
        """Number of backbone layers augmented with Engram."""
        return len(self.insertion_layers)

    @property
    def total_tables(self) -> int:
        """Total embedding tables across all heads and insertion layers.

        Each insertion layer has ``engram.total_heads`` tables, one per hash
        head.  This is relevant for memory budgeting.
        """
        return self.num_insertion_layers * self.engram.total_heads

    @property
    def estimated_table_memory_bytes(self) -> int:
        """
        Rough memory estimate for all embedding tables (bytes).

        Assumes each entry is ``embedding_dim`` floats stored at
        ``offload.storage_dtype`` precision.
        """
        dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}
        bpe = dtype_bytes.get(self.offload.storage_dtype, 4)
        rows = self.hash.effective_table_size
        cols = self.engram.embedding_dim
        num_tables = self.total_tables
        return num_tables * rows * cols * bpe

    # -- Validation --------------------------------------------------------

    def _validate_fields(self) -> None:
        if not isinstance(self.insertion_layers, list):
            raise ValueError(
                f"insertion_layers must be a list, got "
                f"{type(self.insertion_layers).__name__}"
            )
        if len(self.insertion_layers) == 0:
            raise ValueError("insertion_layers must not be empty")
        for idx in self.insertion_layers:
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(
                    f"Each insertion_layer must be a non-negative int, "
                    f"got {idx!r}"
                )
        if len(self.insertion_layers) != len(set(self.insertion_layers)):
            raise ValueError(
                f"insertion_layers contains duplicates: {self.insertion_layers}"
            )
        # Validate nested configs
        self.engram.validate()
        self.hash.validate()
        self.offload.validate()
        self.gating.validate()
        self.compression.validate()


# =========================================================================
# 8. EngramFullConfig  (aggregated)
# =========================================================================

@_register_config_class
@dataclass
class EngramFullConfig(_ConfigMixin):
    """
    Aggregated Engram configuration combining encoder and layer configs.

    This is the top-level config that the ``BrainAI`` system holds.  It
    provides convenience flags (``use_engram``, ``use_engram_encoder``,
    ``use_engram_layers``, ``use_cpu_offload``, ``use_async_prefetch``)
    and preset class methods for common configurations.

    Attributes:
        encoder:              Phase 1 encoder-competition config.
        layer:                Phase 2 layer-augmentation config.
        use_engram:           Master toggle for the entire Engram subsystem.
        use_engram_encoder:   Enable Phase 1 (encoder-competition mode).
        use_engram_layers:    Enable Phase 2 (layer-augmentation mode).
        use_cpu_offload:      Convenience alias -- propagated to
                              ``layer.offload.weights_on_cpu``.
        use_async_prefetch:   Convenience alias -- propagated to
                              ``layer.offload.use_async_prefetch``.
    """

    encoder: EngramEncoderConfig = field(default_factory=EngramEncoderConfig)
    layer: EngramLayerConfig = field(default_factory=EngramLayerConfig)
    use_engram: bool = True
    use_engram_encoder: bool = False
    use_engram_layers: bool = True
    use_cpu_offload: bool = False
    use_async_prefetch: bool = False

    # -- Post-init: propagate convenience flags ----------------------------

    def __post_init__(self) -> None:
        """Propagate convenience aliases into nested configs."""
        self._propagate_convenience_flags()

    def _propagate_convenience_flags(self) -> None:
        """Sync top-level convenience bools into the nested offload config."""
        if self.use_cpu_offload:
            self.layer.offload.weights_on_cpu = True
        if self.use_async_prefetch:
            self.layer.offload.use_async_prefetch = True
            # async prefetch implies CPU offload
            self.layer.offload.weights_on_cpu = True

    # -- Validation --------------------------------------------------------

    def _validate_fields(self) -> None:
        # Propagate first so cross-field checks see the right state
        self._propagate_convenience_flags()

        if not isinstance(self.use_engram, bool):
            raise ValueError(
                f"use_engram must be bool, got {type(self.use_engram).__name__}"
            )
        if not isinstance(self.use_engram_encoder, bool):
            raise ValueError(
                f"use_engram_encoder must be bool, got "
                f"{type(self.use_engram_encoder).__name__}"
            )
        if not isinstance(self.use_engram_layers, bool):
            raise ValueError(
                f"use_engram_layers must be bool, got "
                f"{type(self.use_engram_layers).__name__}"
            )

        # Cross-field: async prefetch requires cpu offload
        if self.use_async_prefetch and not self.use_cpu_offload:
            # We auto-fix via propagation, but also validate the underlying
            if not self.layer.offload.weights_on_cpu:
                raise ValueError(
                    "use_async_prefetch=True requires use_cpu_offload=True "
                    "or layer.offload.weights_on_cpu=True"
                )

        # Validate nested configs
        self.encoder.validate()
        self.layer.validate()

    # -- Presets -----------------------------------------------------------

    @classmethod
    def _build_preset(
        cls,
        *,
        # EngramConfig params
        max_ngram_order: int = 4,
        embedding_dim: int = 256,
        num_heads_per_order: int = 2,
        hidden_dim: int = 512,
        gate_dim: int = 128,
        use_tokenizer_compression: bool = True,
        use_depthwise_conv: bool = True,
        conv_kernel_size: int = 4,
        # HashConfig params
        table_size: int = 131071,
        per_layer_salt: bool = True,
        # OffloadConfig params
        weights_on_cpu: bool = False,
        use_async_prefetch: bool = False,
        prefetch_ahead_layers: int = 2,
        pin_memory: bool = True,
        storage_dtype: str = "float16",
        compute_dtype: str = "float32",
        # GatingConfig params
        use_rmsnorm: bool = True,
        conv_dilation: int = 4,
        # CompressionConfig params
        normalization_recipe: Optional[List[str]] = None,
        # EncoderConfig params
        vocab_size: int = 128000,
        embed_dim: int = 256,
        workspace_dim: int = 4096,
        # LayerConfig params
        insertion_layers: Optional[List[int]] = None,
        # Top-level flags
        use_cpu_offload: bool = False,
        top_use_async_prefetch: bool = False,
    ) -> "EngramFullConfig":
        """Internal helper to build preset configs with shared structure."""
        if normalization_recipe is None:
            normalization_recipe = ["nfkc", "lowercase", "strip_whitespace"]
        if insertion_layers is None:
            insertion_layers = [4, 8, 12, 16]

        engram = EngramConfig(
            max_ngram_order=max_ngram_order,
            embedding_dim=embedding_dim,
            num_heads_per_order=num_heads_per_order,
            hidden_dim=hidden_dim,
            gate_dim=gate_dim,
            use_tokenizer_compression=use_tokenizer_compression,
            use_context_gate=True,
            use_depthwise_conv=use_depthwise_conv,
            conv_kernel_size=conv_kernel_size,
        )
        hash_cfg = HashConfig(
            table_size=table_size,
            per_layer_salt=per_layer_salt,
            seed=42,
        )
        offload = OffloadConfig(
            weights_on_cpu=weights_on_cpu,
            use_async_prefetch=use_async_prefetch,
            prefetch_ahead_layers=prefetch_ahead_layers,
            pin_memory=pin_memory,
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
        )
        gating = GatingConfig(
            gate_type="scalar",
            use_rmsnorm=use_rmsnorm,
            gate_init_bias=-2.0,
            activation="silu",
            conv_dilation=conv_dilation,
            residual_scale=1.0,
        )
        compression = TokenizerCompressionConfig(
            normalization_recipe=list(normalization_recipe),
            special_token_policy="preserve_all",
            canonical_selection="lowest_id",
        )

        encoder_cfg = EngramEncoderConfig(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            workspace_dim=workspace_dim,
            engram=copy.deepcopy(engram),
            hash=copy.deepcopy(hash_cfg),
            compression=copy.deepcopy(compression),
        )
        layer_cfg = EngramLayerConfig(
            insertion_layers=list(insertion_layers),
            engram=engram,
            hash=hash_cfg,
            offload=offload,
            gating=gating,
            compression=compression,
        )

        return cls(
            encoder=encoder_cfg,
            layer=layer_cfg,
            use_engram=True,
            use_engram_encoder=False,
            use_engram_layers=True,
            use_cpu_offload=use_cpu_offload,
            use_async_prefetch=top_use_async_prefetch,
        )

    @classmethod
    def minimal(cls) -> "EngramFullConfig":
        """
        Minimal config for unit tests.

        Small tables (1021), 2 orders, 1 head/order, no offload,
        no tokenizer compression.  Fast to construct and run.
        """
        return cls._build_preset(
            max_ngram_order=2, embedding_dim=32, num_heads_per_order=1,
            hidden_dim=64, gate_dim=16, use_tokenizer_compression=False,
            use_depthwise_conv=False, conv_kernel_size=2,
            table_size=1021, per_layer_salt=False,
            weights_on_cpu=False, use_async_prefetch=False,
            prefetch_ahead_layers=1, pin_memory=False,
            storage_dtype="float32", compute_dtype="float32",
            use_rmsnorm=False, conv_dilation=1,
            normalization_recipe=["nfkc", "lowercase"],
            vocab_size=1000, embed_dim=32, workspace_dim=64,
            insertion_layers=[1],
            use_cpu_offload=False, top_use_async_prefetch=False,
        )

    @classmethod
    def dev(cls) -> "EngramFullConfig":
        """
        Development config.

        Medium tables (8191), 4 orders, 2 heads/order, no offload.
        Suitable for local development and smoke tests on GPU.
        """
        return cls._build_preset(
            max_ngram_order=4, embedding_dim=128, num_heads_per_order=2,
            hidden_dim=256, gate_dim=64,
            table_size=8191,
            vocab_size=32000, embed_dim=128, workspace_dim=512,
            use_cpu_offload=False, top_use_async_prefetch=False,
        )

    @classmethod
    def production(cls) -> "EngramFullConfig":
        """
        Production inference config.

        Large tables (131071), 4 orders, 2 heads/order, CPU offload
        with async prefetch enabled.  Designed for inference serving
        where GPU memory is constrained.
        """
        return cls._build_preset(
            table_size=131071,
            weights_on_cpu=True, use_async_prefetch=True,
            use_cpu_offload=True, top_use_async_prefetch=True,
        )

    @classmethod
    def production_training(cls) -> "EngramFullConfig":
        """
        Production training config.

        Large tables (131071) kept on GPU (no offload), 4 orders,
        2 heads/order.  Designed for training when sufficient GPU
        memory is available.
        """
        return cls._build_preset(
            table_size=131071,
            storage_dtype="float32", compute_dtype="float32",
            use_cpu_offload=False, top_use_async_prefetch=False,
        )

    # -- Utility methods ---------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        lines = [
            "EngramFullConfig Summary",
            "=" * 50,
            f"  Master toggle (use_engram):     {self.use_engram}",
            f"  Phase 1 encoder:               {self.use_engram_encoder}",
            f"  Phase 2 layers:                {self.use_engram_layers}",
            f"  CPU offload:                   {self.use_cpu_offload}",
            f"  Async prefetch:                {self.use_async_prefetch}",
            "",
            "  Encoder (Phase 1):",
            f"    vocab_size:                  {self.encoder.vocab_size}",
            f"    embed_dim:                   {self.encoder.embed_dim}",
            f"    workspace_dim:               {self.encoder.workspace_dim}",
            "",
            "  Layer (Phase 2):",
            f"    insertion_layers:            {self.layer.insertion_layers}",
            f"    max_ngram_order:             {self.layer.engram.max_ngram_order}",
            f"    ngram_orders:                {self.layer.engram.ngram_orders}",
            f"    num_heads_per_order:         {self.layer.engram.num_heads_per_order}",
            f"    total_heads:                 {self.layer.engram.total_heads}",
            f"    embedding_dim:               {self.layer.engram.embedding_dim}",
            f"    total_embedding_dim:         {self.layer.engram.total_embedding_dim}",
            "",
            "  Hash:",
            f"    hash_fn:                     {self.layer.hash.hash_fn}",
            f"    table_size (requested):      {self.layer.hash.table_size}",
            f"    table_size (effective):       {self.layer.hash.effective_table_size}",
            f"    per_layer_salt:              {self.layer.hash.per_layer_salt}",
            f"    seed:                        {self.layer.hash.seed}",
            "",
            "  Offload:",
            f"    weights_on_cpu:              {self.layer.offload.weights_on_cpu}",
            f"    use_async_prefetch:          {self.layer.offload.use_async_prefetch}",
            f"    storage_dtype:               {self.layer.offload.storage_dtype}",
            f"    compute_dtype:               {self.layer.offload.compute_dtype}",
            "",
            "  Gating:",
            f"    gate_type:                   {self.layer.gating.gate_type}",
            f"    use_rmsnorm:                 {self.layer.gating.use_rmsnorm}",
            f"    gate_init_bias:              {self.layer.gating.gate_init_bias}",
            f"    activation:                  {self.layer.gating.activation}",
            "",
            "  Compression:",
            f"    recipe:                      {self.layer.compression.normalization_recipe}",
            f"    special_token_policy:         {self.layer.compression.special_token_policy}",
            f"    canonical_selection:          {self.layer.compression.canonical_selection}",
            "",
            "  Memory Estimate:",
            f"    total tables:                {self.layer.total_tables}",
            f"    est. table memory:           "
            f"{self.layer.estimated_table_memory_bytes / (1024**2):.1f} MB",
        ]
        return "\n".join(lines)

    def with_offload(self, enable: bool = True) -> "EngramFullConfig":
        """Return a copy with CPU offload toggled."""
        cfg = copy.deepcopy(self)
        cfg.use_cpu_offload = enable
        cfg.layer.offload.weights_on_cpu = enable
        if not enable:
            cfg.use_async_prefetch = False
            cfg.layer.offload.use_async_prefetch = False
        return cfg

    def with_prefetch(self, enable: bool = True) -> "EngramFullConfig":
        """Return a copy with async prefetch toggled (implies offload)."""
        cfg = copy.deepcopy(self)
        cfg.use_async_prefetch = enable
        cfg.layer.offload.use_async_prefetch = enable
        if enable:
            cfg.use_cpu_offload = True
            cfg.layer.offload.weights_on_cpu = True
        return cfg

    def with_insertion_layers(self, layers: List[int]) -> "EngramFullConfig":
        """Return a copy with different insertion layer indices."""
        cfg = copy.deepcopy(self)
        cfg.layer.insertion_layers = list(layers)
        return cfg

    def with_table_size(self, size: int) -> "EngramFullConfig":
        """Return a copy with a different hash table size."""
        cfg = copy.deepcopy(self)
        cfg.layer.hash.table_size = size
        cfg.encoder.hash.table_size = size
        return cfg


# =========================================================================
# Module-level factory functions
# =========================================================================

def create_engram_config(
    preset: str = "dev",
    **overrides: Any,
) -> EngramFullConfig:
    """
    Factory function to create an EngramFullConfig from a named preset.

    Args:
        preset:    One of ``"minimal"``, ``"dev"``, ``"production"``,
                   ``"production_training"``.
        overrides: Keyword arguments applied to the top-level config after
                   preset construction (e.g. ``use_engram_encoder=True``).

    Returns:
        Configured and validated ``EngramFullConfig``.

    Raises:
        ValueError: If *preset* is unknown or validation fails.
    """
    factories = {
        "minimal": EngramFullConfig.minimal,
        "dev": EngramFullConfig.dev,
        "production": EngramFullConfig.production,
        "production_training": EngramFullConfig.production_training,
    }
    if preset not in factories:
        raise ValueError(
            f"Unknown preset {preset!r}.  Choose from {sorted(factories.keys())}"
        )
    cfg = factories[preset]()
    for key, val in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(
                f"EngramFullConfig has no attribute {key!r}"
            )
        setattr(cfg, key, val)
    cfg.validate()
    return cfg


def merge_into_brain_config(
    brain_config: Any,
    engram_full: EngramFullConfig,
) -> None:
    """
    Merge an ``EngramFullConfig`` into a ``BrainAIConfig``-like object.

    Sets ``use_engram`` and populates the engram-related fields on the
    brain config.  This is a convenience for bridging the detailed Engram
    config system with the existing top-level config in ``brain_ai/config.py``.

    Args:
        brain_config:  A ``BrainAIConfig`` (or duck-typed equivalent).
        engram_full:   The detailed Engram config to merge.
    """
    brain_config.use_engram = engram_full.use_engram
    # If the brain config has an engram field, populate relevant attrs
    if hasattr(brain_config, "engram"):
        bc_engram = brain_config.engram
        layer_engram = engram_full.layer.engram
        layer_hash = engram_full.layer.hash
        if hasattr(bc_engram, "embedding_dim"):
            bc_engram.embedding_dim = layer_engram.embedding_dim
        if hasattr(bc_engram, "ngram_orders"):
            bc_engram.ngram_orders = layer_engram.ngram_orders
        if hasattr(bc_engram, "num_heads"):
            bc_engram.num_heads = layer_engram.total_heads
        if hasattr(bc_engram, "table_size"):
            bc_engram.table_size = layer_hash.effective_table_size
        if hasattr(bc_engram, "offload_to_cpu"):
            bc_engram.offload_to_cpu = engram_full.use_cpu_offload
        if hasattr(bc_engram, "prefetch"):
            bc_engram.prefetch = engram_full.use_async_prefetch
        if hasattr(bc_engram, "use_compression"):
            bc_engram.use_compression = layer_engram.use_tokenizer_compression
        if hasattr(bc_engram, "conv_kernel_size"):
            bc_engram.conv_kernel_size = layer_engram.conv_kernel_size
        if hasattr(bc_engram, "conv_dilation"):
            bc_engram.conv_dilation = engram_full.layer.gating.conv_dilation


# =========================================================================
# Utility: config diff
# =========================================================================

def config_diff(
    a: _ConfigMixin,
    b: _ConfigMixin,
    prefix: str = "",
) -> List[str]:
    """
    Return a list of human-readable strings describing field differences.

    Recursively compares nested configs.

    Args:
        a:       First config.
        b:       Second config (must be same type as *a*).
        prefix:  Dot-prefix for nested field names.

    Returns:
        List of difference strings, empty if configs are equal.
    """
    if type(a) is not type(b):
        return [f"{prefix}: type mismatch {type(a).__name__} vs {type(b).__name__}"]
    diffs: List[str] = []
    for f in fields(a):  # type: ignore[arg-type]
        fname = f"{prefix}.{f.name}" if prefix else f.name
        va = getattr(a, f.name)
        vb = getattr(b, f.name)
        if isinstance(va, _ConfigMixin) and isinstance(vb, _ConfigMixin):
            diffs.extend(config_diff(va, vb, prefix=fname))
        elif va != vb:
            diffs.append(f"{fname}: {va!r} -> {vb!r}")
    return diffs


# =========================================================================
# Utility: memory budget estimation
# =========================================================================

def estimate_memory_budget(
    config: EngramFullConfig,
    include_encoder: bool = True,
) -> Dict[str, float]:
    """
    Estimate memory usage in megabytes for each component.

    This is a rough estimate based on table sizes and embedding dimensions.
    Actual usage depends on padding, alignment, and framework overhead.

    Args:
        config:           The full Engram config.
        include_encoder:  Whether to include Phase 1 encoder tables.

    Returns:
        Dictionary mapping component names to estimated MB.
    """
    budget: Dict[str, float] = {}
    dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}

    # Phase 2 layer tables
    layer = config.layer
    bpe = dtype_bytes.get(layer.offload.storage_dtype, 4)
    layer_tables = layer.total_tables
    rows = layer.hash.effective_table_size
    cols = layer.engram.embedding_dim
    layer_mb = (layer_tables * rows * cols * bpe) / (1024 ** 2)
    budget["layer_tables_mb"] = round(layer_mb, 2)

    # Phase 2 gating parameters (small)
    gating_params = (
        layer.engram.hidden_dim * layer.engram.gate_dim * 2  # q, k projections
        + layer.engram.gate_dim  # bias
    )
    gating_params *= layer.num_insertion_layers
    budget["layer_gating_mb"] = round(
        (gating_params * 4) / (1024 ** 2), 4
    )  # always float32 for gating

    # Phase 1 encoder tables (if requested)
    if include_encoder and config.use_engram_encoder:
        enc = config.encoder
        enc_engram = enc.engram
        enc_hash = enc.hash
        enc_tables = enc_engram.total_heads
        enc_rows = enc_hash.effective_table_size
        enc_cols = enc_engram.embedding_dim
        enc_mb = (enc_tables * enc_rows * enc_cols * 4) / (1024 ** 2)
        budget["encoder_tables_mb"] = round(enc_mb, 2)
    else:
        budget["encoder_tables_mb"] = 0.0

    budget["total_mb"] = round(sum(budget.values()), 2)
    return budget


# =========================================================================
# Utility: prime table size recommendation
# =========================================================================

def recommend_table_size(
    vocab_size: int,
    max_ngram_order: int,
    collision_budget: float = 0.01,
) -> int:
    """
    Recommend a prime table size based on the expected key space.

    The key space for an N-gram of order *n* over a vocabulary of size *V*
    is *V^n*.  To keep the expected collision rate under *collision_budget*,
    we size the table to::

        M >= min(V^n, V^n / collision_budget)

    capped at a practical limit.

    Args:
        vocab_size:        Vocabulary size.
        max_ngram_order:   Maximum N-gram order.
        collision_budget:  Target max collision probability per bucket.

    Returns:
        Recommended prime table size.
    """
    max_key_space = vocab_size ** max_ngram_order
    ideal_size = int(max_key_space * (1 - collision_budget))
    # Cap at a practical maximum
    practical_cap = 10_000_019
    recommended = min(ideal_size, practical_cap)
    recommended = max(recommended, 1021)  # minimum floor
    return _next_prime(recommended)


# =========================================================================
# Self-tests
# =========================================================================

def _run_self_tests() -> None:
    """
    Run a comprehensive suite of self-tests.

    Each test prints PASS or FAIL with a descriptive label.  The process
    exits with code 1 if any test fails.
    """
    passed = 0
    failed = 0
    total = 0

    def _test(label: str, fn):
        nonlocal passed, failed, total
        total += 1
        try:
            fn()
            print(f"  PASS  [{total:02d}] {label}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  [{total:02d}] {label}")
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print("Engram Config Template -- Self-Tests")
    print("=" * 60)

    # ---- 1. Default construction -----------------------------------------

    def test_engram_config_default():
        cfg = EngramConfig()
        assert cfg.max_ngram_order == 4
        assert cfg.embedding_dim == 256

    _test("EngramConfig default construction", test_engram_config_default)

    def test_hash_config_default():
        cfg = HashConfig()
        assert cfg.hash_fn == "mult_xor"
        assert cfg.table_size == 131071

    _test("HashConfig default construction", test_hash_config_default)

    def test_offload_config_default():
        cfg = OffloadConfig()
        assert cfg.weights_on_cpu is False
        assert cfg.storage_dtype == "float16"

    _test("OffloadConfig default construction", test_offload_config_default)

    def test_gating_config_default():
        cfg = GatingConfig()
        assert cfg.gate_type == "scalar"
        assert cfg.gate_init_bias == -2.0

    _test("GatingConfig default construction", test_gating_config_default)

    def test_compression_config_default():
        cfg = TokenizerCompressionConfig()
        assert "nfkc" in cfg.normalization_recipe
        assert cfg.canonical_selection == "lowest_id"

    _test("TokenizerCompressionConfig default construction",
          test_compression_config_default)

    def test_encoder_config_default():
        cfg = EngramEncoderConfig()
        assert cfg.vocab_size == 32000
        assert cfg.workspace_dim == 4096

    _test("EngramEncoderConfig default construction",
          test_encoder_config_default)

    def test_layer_config_default():
        cfg = EngramLayerConfig()
        assert cfg.insertion_layers == [4, 8, 12, 16]

    _test("EngramLayerConfig default construction",
          test_layer_config_default)

    def test_full_config_default():
        cfg = EngramFullConfig()
        assert cfg.use_engram is True
        assert cfg.use_engram_layers is True
        assert cfg.use_engram_encoder is False

    _test("EngramFullConfig default construction", test_full_config_default)

    # ---- 2. Validation passes --------------------------------------------

    def test_engram_config_valid():
        EngramConfig().validate()

    _test("EngramConfig validates with defaults", test_engram_config_valid)

    def test_hash_config_valid():
        HashConfig().validate()

    _test("HashConfig validates with defaults", test_hash_config_valid)

    def test_offload_config_valid():
        OffloadConfig().validate()

    _test("OffloadConfig validates with defaults", test_offload_config_valid)

    def test_gating_config_valid():
        GatingConfig().validate()

    _test("GatingConfig validates with defaults", test_gating_config_valid)

    def test_compression_config_valid():
        TokenizerCompressionConfig().validate()

    _test("TokenizerCompressionConfig validates with defaults",
          test_compression_config_valid)

    def test_full_config_valid():
        EngramFullConfig().validate()

    _test("EngramFullConfig validates with defaults", test_full_config_valid)

    # ---- 3. Validation fails on bad values -------------------------------

    def test_negative_table_size():
        cfg = HashConfig(table_size=-1)
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "table_size" in str(e)

    _test("HashConfig rejects negative table_size",
          test_negative_table_size)

    def test_bad_hash_fn():
        cfg = HashConfig(hash_fn="unknown_hash")
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "hash_fn" in str(e)

    _test("HashConfig rejects unknown hash_fn", test_bad_hash_fn)

    def test_ngram_order_too_low():
        cfg = EngramConfig(max_ngram_order=1)
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "max_ngram_order" in str(e)

    _test("EngramConfig rejects max_ngram_order < 2",
          test_ngram_order_too_low)

    def test_negative_embedding_dim():
        cfg = EngramConfig(embedding_dim=-10)
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "embedding_dim" in str(e)

    _test("EngramConfig rejects negative embedding_dim",
          test_negative_embedding_dim)

    def test_zero_heads():
        cfg = EngramConfig(num_heads_per_order=0)
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "num_heads_per_order" in str(e)

    _test("EngramConfig rejects zero num_heads_per_order",
          test_zero_heads)

    def test_bad_gate_type():
        cfg = GatingConfig(gate_type="vector")
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "gate_type" in str(e)

    _test("GatingConfig rejects invalid gate_type", test_bad_gate_type)

    def test_inf_gate_bias():
        cfg = GatingConfig(gate_init_bias=float("inf"))
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "gate_init_bias" in str(e)

    _test("GatingConfig rejects infinite gate_init_bias",
          test_inf_gate_bias)

    def test_bad_activation():
        cfg = GatingConfig(activation="tanh")
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "activation" in str(e)

    _test("GatingConfig rejects invalid activation", test_bad_activation)

    def test_bad_storage_dtype():
        cfg = OffloadConfig(storage_dtype="int8")
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "storage_dtype" in str(e)

    _test("OffloadConfig rejects invalid storage_dtype",
          test_bad_storage_dtype)

    def test_bad_norm_step():
        cfg = TokenizerCompressionConfig(
            normalization_recipe=["nfkc", "rot13"]
        )
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "rot13" in str(e)

    _test("TokenizerCompressionConfig rejects unknown norm step",
          test_bad_norm_step)

    def test_bad_canonical_selection():
        cfg = TokenizerCompressionConfig(canonical_selection="random")
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "canonical_selection" in str(e)

    _test("TokenizerCompressionConfig rejects invalid canonical_selection",
          test_bad_canonical_selection)

    def test_negative_seed():
        cfg = HashConfig(seed=-1)
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "seed" in str(e)

    _test("HashConfig rejects negative seed", test_negative_seed)

    def test_negative_vocab_size():
        cfg = EngramEncoderConfig(vocab_size=0)
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "vocab_size" in str(e)

    _test("EngramEncoderConfig rejects zero vocab_size",
          test_negative_vocab_size)

    # ---- 4. Cross-field validation ---------------------------------------

    def test_async_prefetch_without_cpu():
        cfg = OffloadConfig(
            use_async_prefetch=True,
            weights_on_cpu=False,
        )
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "weights_on_cpu" in str(e)

    _test("OffloadConfig: async_prefetch requires weights_on_cpu",
          test_async_prefetch_without_cpu)

    def test_duplicate_insertion_layers():
        cfg = EngramLayerConfig(insertion_layers=[4, 4, 8])
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "duplicates" in str(e).lower()

    _test("EngramLayerConfig rejects duplicate insertion_layers",
          test_duplicate_insertion_layers)

    def test_empty_insertion_layers():
        cfg = EngramLayerConfig(insertion_layers=[])
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "empty" in str(e).lower()

    _test("EngramLayerConfig rejects empty insertion_layers",
          test_empty_insertion_layers)

    # ---- 5. Presets construct and validate --------------------------------

    def test_preset_minimal():
        cfg = EngramFullConfig.minimal()
        cfg.validate()
        assert cfg.layer.hash.table_size == 1021

    _test("Preset: minimal() constructs and validates",
          test_preset_minimal)

    def test_preset_dev():
        cfg = EngramFullConfig.dev()
        cfg.validate()
        assert cfg.layer.hash.table_size == 8191

    _test("Preset: dev() constructs and validates", test_preset_dev)

    def test_preset_production():
        cfg = EngramFullConfig.production()
        cfg.validate()
        assert cfg.layer.offload.weights_on_cpu is True
        assert cfg.layer.offload.use_async_prefetch is True

    _test("Preset: production() constructs and validates",
          test_preset_production)

    def test_preset_production_training():
        cfg = EngramFullConfig.production_training()
        cfg.validate()
        assert cfg.layer.offload.weights_on_cpu is False
        assert cfg.layer.hash.table_size == 131071

    _test("Preset: production_training() constructs and validates",
          test_preset_production_training)

    # ---- 6. Serialization round-trip -------------------------------------

    def test_engram_config_roundtrip():
        original = EngramConfig(
            max_ngram_order=3,
            embedding_dim=128,
            num_heads_per_order=4,
        )
        d = original.to_dict()
        restored = EngramConfig.from_dict(d)
        assert original == restored

    _test("EngramConfig to_dict/from_dict round-trip",
          test_engram_config_roundtrip)

    def test_hash_config_roundtrip():
        original = HashConfig(table_size=8191, seed=99)
        d = original.to_dict()
        restored = HashConfig.from_dict(d)
        assert original == restored

    _test("HashConfig to_dict/from_dict round-trip",
          test_hash_config_roundtrip)

    def test_full_config_roundtrip():
        original = EngramFullConfig.dev()
        d = original.to_dict()
        restored = EngramFullConfig.from_dict(d)
        assert original == restored

    _test("EngramFullConfig to_dict/from_dict round-trip",
          test_full_config_roundtrip)

    def test_production_roundtrip():
        original = EngramFullConfig.production()
        d = original.to_dict()
        restored = EngramFullConfig.from_dict(d)
        assert original == restored

    _test("EngramFullConfig production() round-trip",
          test_production_roundtrip)

    # ---- 7. JSON round-trip ----------------------------------------------

    def test_json_roundtrip():
        original = EngramFullConfig.dev()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_config.json")
            original.save_json(path)
            restored = EngramFullConfig.load_json(path)
        assert original == restored

    _test("EngramFullConfig JSON save/load round-trip",
          test_json_roundtrip)

    def test_json_roundtrip_production():
        original = EngramFullConfig.production()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "prod_config.json")
            original.save_json(path)
            restored = EngramFullConfig.load_json(path)
        assert original == restored

    _test("EngramFullConfig production JSON round-trip",
          test_json_roundtrip_production)

    # ---- 8. Computed properties ------------------------------------------

    def test_total_heads_computation():
        cfg = EngramConfig(max_ngram_order=4, num_heads_per_order=2)
        # orders = (2, 3, 4) -> 3 orders, 2 heads each = 6
        assert cfg.total_heads == 6

    _test("total_heads = num_heads_per_order * (max_ngram_order - 1)",
          test_total_heads_computation)

    def test_total_heads_order2():
        cfg = EngramConfig(max_ngram_order=2, num_heads_per_order=3)
        # orders = (2,) -> 1 order, 3 heads each = 3
        assert cfg.total_heads == 3

    _test("total_heads with max_ngram_order=2", test_total_heads_order2)

    def test_ngram_orders_property():
        cfg = EngramConfig(max_ngram_order=5)
        assert cfg.ngram_orders == (2, 3, 4, 5)

    _test("ngram_orders derived from max_ngram_order",
          test_ngram_orders_property)

    def test_total_embedding_dim():
        cfg = EngramConfig(
            max_ngram_order=3, num_heads_per_order=2, embedding_dim=64
        )
        # 2 orders * 2 heads * 64 = 256
        assert cfg.total_embedding_dim == 256

    _test("total_embedding_dim = total_heads * embedding_dim",
          test_total_embedding_dim)

    def test_effective_table_size_prime():
        cfg = HashConfig(table_size=100, use_prime_sizes=True)
        assert cfg.effective_table_size == 101  # next prime >= 100

    _test("effective_table_size rounds up to prime",
          test_effective_table_size_prime)

    def test_effective_table_size_no_prime():
        cfg = HashConfig(table_size=100, use_prime_sizes=False)
        assert cfg.effective_table_size == 100

    _test("effective_table_size passes through without prime rounding",
          test_effective_table_size_no_prime)

    def test_layer_num_insertion_layers():
        cfg = EngramLayerConfig(insertion_layers=[1, 5, 9])
        assert cfg.num_insertion_layers == 3

    _test("num_insertion_layers matches list length",
          test_layer_num_insertion_layers)

    def test_estimated_table_memory():
        cfg = EngramLayerConfig(
            insertion_layers=[0],
            engram=EngramConfig(
                max_ngram_order=2,
                num_heads_per_order=1,
                embedding_dim=4,
            ),
            hash=HashConfig(table_size=100, use_prime_sizes=False),
            offload=OffloadConfig(storage_dtype="float32"),
        )
        # 1 layer * 1 head * 100 rows * 4 cols * 4 bytes = 1600 bytes
        assert cfg.estimated_table_memory_bytes == 1600

    _test("estimated_table_memory_bytes calculation",
          test_estimated_table_memory)

    # ---- 9. Preset differences -------------------------------------------

    def test_minimal_smaller_than_production():
        mini = EngramFullConfig.minimal()
        prod = EngramFullConfig.production()
        assert mini.layer.hash.table_size < prod.layer.hash.table_size
        assert mini.layer.engram.embedding_dim < prod.layer.engram.embedding_dim

    _test("Minimal preset has smaller tables than production",
          test_minimal_smaller_than_production)

    def test_production_has_offload():
        prod = EngramFullConfig.production()
        train = EngramFullConfig.production_training()
        assert prod.use_cpu_offload is True
        assert train.use_cpu_offload is False

    _test("Production has offload, training does not",
          test_production_has_offload)

    def test_dev_between_minimal_and_prod():
        mini = EngramFullConfig.minimal()
        dev = EngramFullConfig.dev()
        prod = EngramFullConfig.production()
        assert mini.layer.hash.table_size < dev.layer.hash.table_size
        assert dev.layer.hash.table_size < prod.layer.hash.table_size

    _test("Dev preset table size between minimal and production",
          test_dev_between_minimal_and_prod)

    # ---- 10. Type safety -------------------------------------------------

    def test_string_field_type():
        """Ensure string fields are actually strings after construction."""
        cfg = HashConfig()
        assert isinstance(cfg.hash_fn, str)
        cfg2 = GatingConfig()
        assert isinstance(cfg2.gate_type, str)
        assert isinstance(cfg2.activation, str)

    _test("String fields have str type", test_string_field_type)

    def test_list_field_type():
        """Ensure list fields are actually lists after construction."""
        cfg = TokenizerCompressionConfig()
        assert isinstance(cfg.normalization_recipe, list)
        assert isinstance(cfg.additional_special_ids, list)

    _test("List fields have list type", test_list_field_type)

    def test_bool_field_type():
        """Ensure bool fields are actually bools after construction."""
        cfg = EngramFullConfig()
        assert isinstance(cfg.use_engram, bool)
        assert isinstance(cfg.use_engram_encoder, bool)
        assert isinstance(cfg.use_engram_layers, bool)
        assert isinstance(cfg.use_cpu_offload, bool)

    _test("Bool fields have bool type", test_bool_field_type)

    # ---- 11. Default values are sensible ---------------------------------

    def test_defaults_not_none():
        """All defaults should be concrete values, not None."""
        for cls in [
            EngramConfig, HashConfig, OffloadConfig, GatingConfig,
            TokenizerCompressionConfig, EngramEncoderConfig,
            EngramLayerConfig, EngramFullConfig,
        ]:
            cfg = cls()
            for f in fields(cfg):
                val = getattr(cfg, f.name)
                assert val is not None, (
                    f"{cls.__name__}.{f.name} has None default"
                )

    _test("No field has None as default value", test_defaults_not_none)

    def test_pad_id_default():
        cfg = EngramConfig()
        assert cfg.pad_id == 0

    _test("pad_id defaults to 0", test_pad_id_default)

    # ---- 12. Version field -----------------------------------------------

    def test_version_in_serialized():
        cfg = EngramConfig()
        d = cfg.to_dict()
        assert "__version__" in d
        assert d["__version__"] == _CONFIG_VERSION

    _test("__version__ present in serialized dict", test_version_in_serialized)

    def test_version_in_full_config():
        cfg = EngramFullConfig.dev()
        d = cfg.to_dict()
        assert "__version__" in d
        # Nested configs also have version
        assert "__version__" in d["encoder"]
        assert "__version__" in d["layer"]

    _test("__version__ in full config and nested configs",
          test_version_in_full_config)

    def test_version_survives_roundtrip():
        cfg = EngramFullConfig.dev()
        d = cfg.to_dict()
        version = d["__version__"]
        restored = EngramFullConfig.from_dict(d)
        d2 = restored.to_dict()
        assert d2["__version__"] == version

    _test("Version survives dict round-trip", test_version_survives_roundtrip)

    # ---- 13. Factory function --------------------------------------------

    def test_factory_dev():
        cfg = create_engram_config("dev")
        assert cfg.layer.hash.table_size == 8191

    _test("create_engram_config('dev') works", test_factory_dev)

    def test_factory_unknown():
        try:
            create_engram_config("unknown_preset")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "unknown_preset" in str(e)

    _test("create_engram_config rejects unknown preset",
          test_factory_unknown)

    def test_factory_with_overrides():
        cfg = create_engram_config("minimal", use_engram_encoder=True)
        assert cfg.use_engram_encoder is True

    _test("create_engram_config with overrides", test_factory_with_overrides)

    # ---- 14. Config diff -------------------------------------------------

    def test_config_diff_equal():
        a = EngramConfig()
        b = EngramConfig()
        assert config_diff(a, b) == []

    _test("config_diff returns empty for equal configs",
          test_config_diff_equal)

    def test_config_diff_different():
        a = EngramConfig(max_ngram_order=3)
        b = EngramConfig(max_ngram_order=5)
        diffs = config_diff(a, b)
        assert len(diffs) == 1
        assert "max_ngram_order" in diffs[0]

    _test("config_diff detects field differences",
          test_config_diff_different)

    # ---- 15. Builder methods ---------------------------------------------

    def test_with_offload():
        cfg = EngramFullConfig.dev()
        cfg2 = cfg.with_offload(True)
        assert cfg2.use_cpu_offload is True
        assert cfg2.layer.offload.weights_on_cpu is True
        # Original unchanged
        assert cfg.use_cpu_offload is False

    _test("with_offload returns modified copy", test_with_offload)

    def test_with_prefetch():
        cfg = EngramFullConfig.dev()
        cfg2 = cfg.with_prefetch(True)
        assert cfg2.use_async_prefetch is True
        assert cfg2.use_cpu_offload is True  # implied

    _test("with_prefetch enables offload implicitly",
          test_with_prefetch)

    def test_with_table_size():
        cfg = EngramFullConfig.dev()
        cfg2 = cfg.with_table_size(65537)
        assert cfg2.layer.hash.table_size == 65537
        assert cfg2.encoder.hash.table_size == 65537

    _test("with_table_size updates both layer and encoder",
          test_with_table_size)

    def test_with_insertion_layers():
        cfg = EngramFullConfig.dev()
        cfg2 = cfg.with_insertion_layers([0, 3, 7])
        assert cfg2.layer.insertion_layers == [0, 3, 7]
        # Original unchanged
        assert cfg.layer.insertion_layers == [4, 8, 12, 16]

    _test("with_insertion_layers returns modified copy",
          test_with_insertion_layers)

    # ---- 16. Memory estimation -------------------------------------------

    def test_memory_budget():
        cfg = EngramFullConfig.minimal()
        budget = estimate_memory_budget(cfg)
        assert "total_mb" in budget
        assert budget["total_mb"] >= 0

    _test("estimate_memory_budget returns valid budget",
          test_memory_budget)

    # ---- 17. Summary -----------------------------------------------------

    def test_summary():
        cfg = EngramFullConfig.production()
        s = cfg.summary()
        assert "EngramFullConfig Summary" in s
        assert "total tables" in s.lower() or "total_tables" in s.lower()

    _test("summary() produces readable output", test_summary)

    # ---- 18. Prime utility -----------------------------------------------

    def test_is_prime():
        assert _is_prime(2)
        assert _is_prime(131071)
        assert not _is_prime(4)
        assert not _is_prime(1)

    _test("_is_prime utility works correctly", test_is_prime)

    def test_next_prime():
        assert _next_prime(100) == 101
        assert _next_prime(131071) == 131071  # already prime
        assert _next_prime(1) == 2

    _test("_next_prime utility works correctly", test_next_prime)

    # ---- 19. Recommend table size ----------------------------------------

    def test_recommend_table_size():
        size = recommend_table_size(1000, 2)
        assert _is_prime(size)
        assert size >= 1021

    _test("recommend_table_size returns a prime", test_recommend_table_size)

    # ---- 20. Convenience propagation -------------------------------------

    def test_convenience_propagation():
        cfg = EngramFullConfig(use_async_prefetch=True)
        assert cfg.layer.offload.use_async_prefetch is True
        assert cfg.layer.offload.weights_on_cpu is True

    _test("Convenience flags propagate to nested offload config",
          test_convenience_propagation)

    # ---- 21. Nested validation cascade -----------------------------------

    def test_nested_validation_cascade():
        """EngramFullConfig.validate() catches nested config errors."""
        cfg = EngramFullConfig()
        cfg.layer.hash.hash_fn = "bad_fn"
        try:
            cfg.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "hash_fn" in str(e)

    _test("EngramFullConfig.validate() cascades to nested configs",
          test_nested_validation_cascade)

    # ---- Results ---------------------------------------------------------

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {total} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed.")


# =========================================================================
# Entry point
# =========================================================================

if __name__ == "__main__":
    _run_self_tests()
