"""
Task Embedding Registry — Persistent storage for Task2Vec embeddings.

This module implements TaskEmbeddingRegistry, a complete system for storing,
querying, serializing, and analyzing Task2Vec embeddings alongside their
metadata. It supports two persistence formats:

  1. JSONL + NPZ: metadata lines in task2vec_registry.jsonl, embedding
     vectors in embeddings.npz (default, most efficient).
  2. Parquet: single-file format with embeddings packed as byte columns
     (optional, requires pandas + pyarrow).

The registry is the canonical artifact saved alongside training checkpoints
so that curriculum ordering, clustering, and meta-batch composition can be
reproduced deterministically.

Format version: 1.0

Usage::

    from embedding_registry_template import TaskEmbeddingRegistry, RegistryEntry
    import numpy as np

    reg = TaskEmbeddingRegistry(embedding_dim=512)
    reg.update("task_001", np.random.randn(512), dataset="omniglot", split="train")
    reg.save(Path("./registry_dir"))
    loaded = TaskEmbeddingRegistry.load(Path("./registry_dir"))
    emb = loaded.get_embedding("task_001")  # shape (512,)
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports — guarded so the module works with numpy alone.
# ---------------------------------------------------------------------------
try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    _HAS_PANDAS = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FORMAT_VERSION = "1.0"
_JSONL_FILENAME = "task2vec_registry.jsonl"
_NPZ_FILENAME = "embeddings.npz"
_PARQUET_FILENAME = "task2vec_registry.parquet"
_EMBEDDING_NPZ_KEY = "embeddings"


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class RegistryHeader:
    """Header metadata for a serialised registry file.

    The header is stored as the *first* line of the JSONL file and always
    carries field names prefixed with ``_`` so that it can be distinguished
    from regular entry lines.

    Attributes:
        version: Format version string (currently ``"1.0"``).
        probe_signature: Hash identifying the probe model and layer subset
            used to extract embeddings.  An empty string means unknown.
        created: Unix timestamp of when the registry was first created.
        n_entries: Number of embedding entries in the file (not counting
            the header line itself).
        embedding_dim: Dimensionality of every embedding vector.
    """

    version: str = _FORMAT_VERSION
    probe_signature: str = ""
    created: float = 0.0
    n_entries: int = 0
    embedding_dim: int = 512

    # ----- serialisation helpers -----

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict with ``_``-prefixed keys for JSONL storage."""
        return {
            "_version": self.version,
            "_probe_signature": self.probe_signature,
            "_created": self.created,
            "_n_entries": self.n_entries,
            "_embedding_dim": self.embedding_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegistryHeader":
        """Construct from a dict with ``_``-prefixed keys.

        Unknown fields are silently ignored for forward compatibility.
        Missing fields fall back to defaults.
        """
        return cls(
            version=d.get("_version", _FORMAT_VERSION),
            probe_signature=d.get("_probe_signature", ""),
            created=float(d.get("_created", 0.0)),
            n_entries=int(d.get("_n_entries", 0)),
            embedding_dim=int(d.get("_embedding_dim", 512)),
        )


@dataclass
class RegistryEntry:
    """A single task embedding together with its extraction metadata.

    Attributes:
        task_id: Deterministic hash uniquely identifying the task episode.
        embedding: The L2-normalised Fisher-derived embedding vector of
            shape ``(E,)``.
        dataset: Name of the source dataset (e.g. ``"omniglot"``).
        split: Dataset split — ``"train"``, ``"val"``, or ``"test"``.
        n_way: Number of classes in the episode.
        k_shot: Number of support samples per class.
        q_query: Number of query samples per class.
        class_ids: Sorted list of class indices used in the episode.
        support_indices: Indices that selected support samples.
        probe_signature: Hash of the probe model + layer subset used for
            extraction, linking this embedding to a specific probe.
        extraction_seed: RNG seed used during Fisher computation.
        extraction_timestamp: Unix timestamp when extraction happened.
        code_version: Version string of the extraction code.
        diagnostics: Dictionary of diagnostic scalars (Fisher norm,
            sparsity, probe loss, extraction wall-time, etc.).
        row_index: Position of this entry's embedding vector inside the
            NPZ array.  Set automatically during serialisation.
    """

    task_id: str = ""
    embedding: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    dataset: str = ""
    split: str = ""
    n_way: int = 0
    k_shot: int = 0
    q_query: int = 0
    class_ids: List[int] = field(default_factory=list)
    support_indices: List[int] = field(default_factory=list)
    probe_signature: str = ""
    extraction_seed: int = 0
    extraction_timestamp: float = 0.0
    code_version: str = _FORMAT_VERSION
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    row_index: int = -1

    # ----- serialisation helpers (embedding excluded) -----

    def meta_dict(self) -> Dict[str, Any]:
        """Return all fields *except* the embedding array as a JSON-safe dict.

        This dict is written to the JSONL file.  The embedding itself is
        stored separately in the NPZ.
        """
        return {
            "task_id": self.task_id,
            "dataset": self.dataset,
            "split": self.split,
            "n_way": self.n_way,
            "k_shot": self.k_shot,
            "q_query": self.q_query,
            "class_ids": list(self.class_ids),
            "support_indices": list(self.support_indices),
            "probe_signature": self.probe_signature,
            "extraction_seed": self.extraction_seed,
            "extraction_timestamp": self.extraction_timestamp,
            "code_version": self.code_version,
            "diagnostics": dict(self.diagnostics),
            "row_index": self.row_index,
        }

    @classmethod
    def from_meta_dict(cls, d: Dict[str, Any]) -> "RegistryEntry":
        """Construct an entry from its meta dict.

        The ``embedding`` field is left empty — callers must populate it
        from the NPZ separately.  Unknown keys are silently ignored for
        forward compatibility.
        """
        return cls(
            task_id=d.get("task_id", ""),
            dataset=d.get("dataset", ""),
            split=d.get("split", ""),
            n_way=int(d.get("n_way", 0)),
            k_shot=int(d.get("k_shot", 0)),
            q_query=int(d.get("q_query", 0)),
            class_ids=list(d.get("class_ids", [])),
            support_indices=list(d.get("support_indices", [])),
            probe_signature=d.get("probe_signature", ""),
            extraction_seed=int(d.get("extraction_seed", 0)),
            extraction_timestamp=float(d.get("extraction_timestamp", 0.0)),
            code_version=d.get("code_version", _FORMAT_VERSION),
            diagnostics=dict(d.get("diagnostics", {})),
            row_index=int(d.get("row_index", -1)),
        )

    # ----- copy -----

    def clone(self) -> "RegistryEntry":
        """Return a deep copy of this entry."""
        new = RegistryEntry(
            task_id=self.task_id,
            embedding=self.embedding.copy(),
            dataset=self.dataset,
            split=self.split,
            n_way=self.n_way,
            k_shot=self.k_shot,
            q_query=self.q_query,
            class_ids=list(self.class_ids),
            support_indices=list(self.support_indices),
            probe_signature=self.probe_signature,
            extraction_seed=self.extraction_seed,
            extraction_timestamp=self.extraction_timestamp,
            code_version=self.code_version,
            diagnostics=dict(self.diagnostics),
            row_index=self.row_index,
        )
        return new


# ============================================================================
# Helper utilities
# ============================================================================


def _validate_embedding(embedding: np.ndarray, expected_dim: int) -> np.ndarray:
    """Coerce *embedding* to a 1-D float32 numpy array of length *expected_dim*.

    Accepts numpy arrays and (if torch is available) torch Tensors.

    Raises:
        ValueError: If the embedding has the wrong number of elements.
        TypeError: If *embedding* is not a supported type.
    """
    if _HAS_TORCH and isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()

    if not isinstance(embedding, np.ndarray):
        try:
            embedding = np.asarray(embedding, dtype=np.float32)
        except Exception as exc:
            raise TypeError(
                f"Cannot convert embedding of type {type(embedding)} to ndarray"
            ) from exc

    embedding = embedding.astype(np.float32, copy=False).ravel()

    if embedding.shape[0] != expected_dim:
        raise ValueError(
            f"Embedding dim mismatch: got {embedding.shape[0]}, "
            f"expected {expected_dim}"
        )
    return embedding


def _compute_embedding_hash(embedding: np.ndarray) -> str:
    """Return a short hex digest summarising the embedding vector.

    Used for change detection in :py:meth:`TaskEmbeddingRegistry.diff`.
    """
    raw = embedding.tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


# ============================================================================
# TaskEmbeddingRegistry
# ============================================================================


class TaskEmbeddingRegistry:
    """Persistent, queryable store for Task2Vec embeddings and metadata.

    The registry maintains an ordered list of :class:`RegistryEntry` objects
    together with a contiguous ``(N, E)`` float32 numpy matrix.  Entries are
    keyed by ``task_id``; duplicate inserts silently overwrite previous
    entries.

    Parameters:
        embedding_dim: Dimensionality of each embedding vector.  Every
            vector added via :meth:`update` must match this value.

    Example::

        reg = TaskEmbeddingRegistry(embedding_dim=256)
        reg.update("t1", my_embedding, dataset="mini-imagenet", split="train")
        reg.save(Path("/tmp/registry"))
        loaded = TaskEmbeddingRegistry.load(Path("/tmp/registry"))
    """

    # ----- class-level constants -----
    FORMAT_VERSION: ClassVar[str] = _FORMAT_VERSION

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, embedding_dim: int = 512) -> None:
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        self._embedding_dim: int = embedding_dim
        self._entries: Dict[str, RegistryEntry] = {}  # task_id -> entry
        self._insertion_order: List[str] = []  # ordered task_ids
        self._created: float = time.time()
        self._probe_signatures: Set[str] = set()
        logger.debug(
            "TaskEmbeddingRegistry created (embedding_dim=%d)", embedding_dim
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of stored embeddings."""
        return self._embedding_dim

    @property
    def created(self) -> float:
        """Unix timestamp of registry creation."""
        return self._created

    @property
    def probe_signatures(self) -> Set[str]:
        """Set of all probe signatures present in the registry."""
        return set(self._probe_signatures)

    # ------------------------------------------------------------------
    # Update operations
    # ------------------------------------------------------------------

    def update(
        self,
        task_id: str,
        embedding: np.ndarray,
        *,
        dataset: str = "",
        split: str = "",
        n_way: int = 0,
        k_shot: int = 0,
        q_query: int = 0,
        class_ids: Optional[List[int]] = None,
        support_indices: Optional[List[int]] = None,
        probe_signature: str = "",
        extraction_seed: int = 0,
        extraction_timestamp: float = 0.0,
        code_version: str = _FORMAT_VERSION,
        diagnostics: Optional[Dict[str, Any]] = None,
        **extra_meta: Any,
    ) -> None:
        """Add or overwrite an embedding entry.

        Parameters:
            task_id: Unique identifier for the task episode.
            embedding: Embedding vector of shape ``(E,)``.  Torch tensors
                are automatically converted.
            dataset: Source dataset name.
            split: Dataset split.
            n_way: Number of classes.
            k_shot: Support shots per class.
            q_query: Query samples per class.
            class_ids: Class indices used.
            support_indices: Sample indices for support set.
            probe_signature: Hash of probe model configuration.
            extraction_seed: RNG seed used during extraction.
            extraction_timestamp: When extraction occurred (Unix).
            code_version: Version of extraction code.
            diagnostics: Scalar diagnostics dict.
            **extra_meta: Silently ignored (forward compat).

        Raises:
            ValueError: If the embedding dimension does not match.
        """
        emb = _validate_embedding(embedding, self._embedding_dim)

        if class_ids is None:
            class_ids = []
        if support_indices is None:
            support_indices = []
        if diagnostics is None:
            diagnostics = {}
        if extraction_timestamp == 0.0:
            extraction_timestamp = time.time()

        entry = RegistryEntry(
            task_id=task_id,
            embedding=emb,
            dataset=dataset,
            split=split,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
            class_ids=list(class_ids),
            support_indices=list(support_indices),
            probe_signature=probe_signature,
            extraction_seed=extraction_seed,
            extraction_timestamp=extraction_timestamp,
            code_version=code_version,
            diagnostics=dict(diagnostics),
            row_index=-1,  # assigned during save
        )

        is_overwrite = task_id in self._entries
        self._entries[task_id] = entry
        if not is_overwrite:
            self._insertion_order.append(task_id)
        if probe_signature:
            self._probe_signatures.add(probe_signature)

        if is_overwrite:
            logger.debug("Overwrote entry for task_id=%s", task_id)
        else:
            logger.debug(
                "Added entry for task_id=%s (total=%d)", task_id, len(self._entries)
            )

    def update_batch(self, entries: List[RegistryEntry]) -> None:
        """Add or overwrite multiple entries at once.

        Each entry's ``embedding`` must already have the correct
        dimensionality.

        Parameters:
            entries: Sequence of :class:`RegistryEntry` objects.

        Raises:
            ValueError: If any embedding dimension does not match.
        """
        for entry in entries:
            emb = _validate_embedding(entry.embedding, self._embedding_dim)
            entry.embedding = emb

            is_overwrite = entry.task_id in self._entries
            self._entries[entry.task_id] = entry
            if not is_overwrite:
                self._insertion_order.append(entry.task_id)
            if entry.probe_signature:
                self._probe_signatures.add(entry.probe_signature)

        logger.debug("Batch-updated %d entries (total=%d)", len(entries), len(self._entries))

    # ------------------------------------------------------------------
    # Remove operations
    # ------------------------------------------------------------------

    def remove(self, task_id: str) -> bool:
        """Remove an entry by task_id.

        Parameters:
            task_id: The entry to remove.

        Returns:
            ``True`` if the entry existed and was removed, ``False`` otherwise.
        """
        if task_id not in self._entries:
            return False
        del self._entries[task_id]
        self._insertion_order = [t for t in self._insertion_order if t != task_id]
        logger.debug("Removed entry task_id=%s (total=%d)", task_id, len(self._entries))
        return True

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def get_embedding(self, task_id: str) -> np.ndarray:
        """Return the embedding vector for *task_id*.

        Parameters:
            task_id: The task to look up.

        Returns:
            A *copy* of the embedding as a float32 ndarray of shape ``(E,)``.

        Raises:
            KeyError: If *task_id* is not in the registry.
        """
        if task_id not in self._entries:
            raise KeyError(f"task_id '{task_id}' not found in registry")
        return self._entries[task_id].embedding.copy()

    def get_embeddings(self, task_ids: List[str]) -> np.ndarray:
        """Return a batch of embeddings as a ``(N, E)`` matrix.

        Parameters:
            task_ids: List of N task identifiers to look up.

        Returns:
            Float32 ndarray of shape ``(N, E)``.

        Raises:
            KeyError: If any task_id is missing.
        """
        embeddings = []
        for tid in task_ids:
            embeddings.append(self.get_embedding(tid))
        return np.stack(embeddings, axis=0)

    def get_entry(self, task_id: str) -> RegistryEntry:
        """Return the full :class:`RegistryEntry` for *task_id*.

        The returned entry is a deep copy — mutating it will not affect
        the registry.

        Raises:
            KeyError: If *task_id* is not in the registry.
        """
        if task_id not in self._entries:
            raise KeyError(f"task_id '{task_id}' not found in registry")
        return self._entries[task_id].clone()

    def query(
        self,
        *,
        dataset: Optional[str] = None,
        split: Optional[str] = None,
        n_way: Optional[int] = None,
        k_shot: Optional[int] = None,
        q_query: Optional[int] = None,
        probe_signature: Optional[str] = None,
    ) -> List[str]:
        """Return task_ids matching all supplied filter criteria.

        Parameters are combined with AND semantics: only entries matching
        *every* non-None argument are returned.  When called with no
        arguments, all task_ids are returned in insertion order.

        Parameters:
            dataset: Filter by dataset name.
            split: Filter by split.
            n_way: Filter by number of classes.
            k_shot: Filter by shots per class.
            q_query: Filter by query samples.
            probe_signature: Filter by probe signature.

        Returns:
            List of matching task_ids in insertion order.
        """
        result: List[str] = []
        for tid in self._insertion_order:
            entry = self._entries[tid]
            if dataset is not None and entry.dataset != dataset:
                continue
            if split is not None and entry.split != split:
                continue
            if n_way is not None and entry.n_way != n_way:
                continue
            if k_shot is not None and entry.k_shot != k_shot:
                continue
            if q_query is not None and entry.q_query != q_query:
                continue
            if probe_signature is not None and entry.probe_signature != probe_signature:
                continue
            result.append(tid)
        return result

    def all_task_ids(self) -> List[str]:
        """Return all registered task_ids in insertion order."""
        return list(self._insertion_order)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of entries in the registry."""
        return len(self._entries)

    def __contains__(self, task_id: object) -> bool:
        """Check whether *task_id* is present."""
        return task_id in self._entries

    def __iter__(self) -> Iterator[str]:
        """Iterate over task_ids in insertion order."""
        return iter(self._insertion_order)

    def __repr__(self) -> str:
        return (
            f"TaskEmbeddingRegistry(n_entries={len(self)}, "
            f"embedding_dim={self._embedding_dim})"
        )

    # ------------------------------------------------------------------
    # I/O — JSONL + NPZ (default format)
    # ------------------------------------------------------------------

    def _build_header(self) -> RegistryHeader:
        """Construct a :class:`RegistryHeader` reflecting current state."""
        sig = ""
        if len(self._probe_signatures) == 1:
            sig = next(iter(self._probe_signatures))
        elif len(self._probe_signatures) > 1:
            logger.warning(
                "Registry contains %d distinct probe signatures: %s. "
                "Header will record an empty probe_signature.",
                len(self._probe_signatures),
                self._probe_signatures,
            )
        return RegistryHeader(
            version=_FORMAT_VERSION,
            probe_signature=sig,
            created=self._created,
            n_entries=len(self._entries),
            embedding_dim=self._embedding_dim,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Write the registry to *path* as JSONL + NPZ.

        Two files are created inside *path* (which must be a directory,
        created if necessary):

        - ``task2vec_registry.jsonl`` — one header line followed by one
          JSON line per entry (all metadata, no embedding vectors).
        - ``embeddings.npz`` — a single ``"embeddings"`` key mapping to
          a ``(N, E)`` float32 array.

        The row order in the NPZ matches the line order in the JSONL (after
        the header line).  Each entry's ``row_index`` field is set accordingly.

        Parameters:
            path: Directory in which to save the files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        jsonl_path = path / _JSONL_FILENAME
        npz_path = path / _NPZ_FILENAME

        # Assign row indices
        ordered_entries: List[RegistryEntry] = []
        for idx, tid in enumerate(self._insertion_order):
            entry = self._entries[tid]
            entry.row_index = idx
            ordered_entries.append(entry)

        # Build embedding matrix
        n = len(ordered_entries)
        if n > 0:
            emb_matrix = np.stack(
                [e.embedding for e in ordered_entries], axis=0
            )  # (N, E)
        else:
            emb_matrix = np.empty((0, self._embedding_dim), dtype=np.float32)

        # Write NPZ
        np.savez_compressed(str(npz_path), **{_EMBEDDING_NPZ_KEY: emb_matrix})

        # Write JSONL (header + entries)
        header = self._build_header()
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(header.to_dict(), separators=(",", ":")) + "\n")
            for entry in ordered_entries:
                f.write(json.dumps(entry.meta_dict(), separators=(",", ":")) + "\n")

        logger.info(
            "Saved registry to %s (%d entries, embedding_dim=%d)",
            path,
            n,
            self._embedding_dim,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TaskEmbeddingRegistry":
        """Load a registry from a JSONL + NPZ directory.

        Parameters:
            path: Directory containing ``task2vec_registry.jsonl`` and
                ``embeddings.npz``.

        Returns:
            A fully populated :class:`TaskEmbeddingRegistry`.

        Raises:
            FileNotFoundError: If required files are missing.
            ValueError: If the format version is incompatible or data is
                inconsistent.
        """
        path = Path(path)
        jsonl_path = path / _JSONL_FILENAME
        npz_path = path / _NPZ_FILENAME

        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        # Load NPZ
        with np.load(str(npz_path)) as npz:
            if _EMBEDDING_NPZ_KEY not in npz:
                raise ValueError(
                    f"NPZ file missing '{_EMBEDDING_NPZ_KEY}' key"
                )
            emb_matrix = npz[_EMBEDDING_NPZ_KEY].astype(np.float32)

        # Parse JSONL
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        if len(lines) == 0:
            raise ValueError("JSONL file is empty")

        # First line is the header
        header_dict = json.loads(lines[0])
        header = RegistryHeader.from_dict(header_dict)

        # Version check
        major_version = header.version.split(".")[0]
        if major_version not in ("1",):
            raise ValueError(
                f"Unsupported format version: {header.version} "
                f"(this code supports 1.x)"
            )

        embedding_dim = header.embedding_dim

        # Validate embedding matrix shape
        if emb_matrix.ndim == 1 and emb_matrix.shape[0] == 0:
            # Empty registry saved as (0,) — reshape
            emb_matrix = emb_matrix.reshape(0, embedding_dim)
        if emb_matrix.ndim != 2:
            raise ValueError(
                f"Expected 2-D embedding matrix, got shape {emb_matrix.shape}"
            )
        if emb_matrix.shape[0] > 0 and emb_matrix.shape[1] != embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: header says {embedding_dim}, "
                f"NPZ has {emb_matrix.shape[1]}"
            )

        # Build registry
        registry = cls(embedding_dim=embedding_dim)
        registry._created = header.created

        entry_lines = lines[1:]
        if len(entry_lines) != emb_matrix.shape[0]:
            raise ValueError(
                f"Entry count mismatch: {len(entry_lines)} JSONL entries, "
                f"{emb_matrix.shape[0]} embedding rows"
            )

        for line in entry_lines:
            d = json.loads(line)
            entry = RegistryEntry.from_meta_dict(d)
            row_idx = entry.row_index
            if row_idx < 0 or row_idx >= emb_matrix.shape[0]:
                raise ValueError(
                    f"Invalid row_index {row_idx} for task_id='{entry.task_id}'"
                )
            entry.embedding = emb_matrix[row_idx].copy()
            registry._entries[entry.task_id] = entry
            registry._insertion_order.append(entry.task_id)
            if entry.probe_signature:
                registry._probe_signatures.add(entry.probe_signature)

        logger.info(
            "Loaded registry from %s (%d entries, embedding_dim=%d, version=%s)",
            path,
            len(registry),
            embedding_dim,
            header.version,
        )
        return registry

    # ------------------------------------------------------------------
    # I/O — Parquet (optional single-file format)
    # ------------------------------------------------------------------

    def save_parquet(self, path: Union[str, Path]) -> None:
        """Write the registry to a single Parquet file.

        The Parquet file stores all metadata columns plus the embedding
        vector packed as raw bytes in a ``embedding_bytes`` column.

        Requires ``pandas`` and ``pyarrow`` (or ``fastparquet``).

        Parameters:
            path: File path for the output ``.parquet`` file.

        Raises:
            ImportError: If pandas is not available.
        """
        if not _HAS_PANDAS:
            raise ImportError(
                "pandas is required for Parquet support. "
                "Install with: pip install pandas pyarrow"
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        records: List[Dict[str, Any]] = []
        for idx, tid in enumerate(self._insertion_order):
            entry = self._entries[tid]
            rec = entry.meta_dict()
            rec["row_index"] = idx
            # Pack embedding as bytes for compact storage
            rec["embedding_bytes"] = entry.embedding.tobytes()
            # Store diagnostics as JSON string
            rec["diagnostics"] = json.dumps(rec.get("diagnostics", {}))
            # Store class_ids and support_indices as JSON strings
            rec["class_ids"] = json.dumps(rec.get("class_ids", []))
            rec["support_indices"] = json.dumps(rec.get("support_indices", []))
            records.append(rec)

        df = pd.DataFrame(records)

        # Add header metadata as Parquet file metadata
        header = self._build_header()
        metadata = {
            "_registry_header": json.dumps(header.to_dict()),
        }

        # Write parquet with metadata
        table = None
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(df)
            existing_meta = table.schema.metadata or {}
            existing_meta.update(
                {k.encode(): v.encode() for k, v in metadata.items()}
            )
            table = table.replace_schema_metadata(existing_meta)
            pq.write_table(table, str(path))
        except ImportError:
            # Fallback: plain pandas to_parquet (no custom metadata)
            df.to_parquet(str(path), index=False)
            logger.warning(
                "pyarrow not available — wrote Parquet without header metadata"
            )

        logger.info(
            "Saved registry to Parquet %s (%d entries)", path, len(self)
        )

    @classmethod
    def load_parquet(cls, path: Union[str, Path]) -> "TaskEmbeddingRegistry":
        """Load a registry from a Parquet file.

        Parameters:
            path: Path to the ``.parquet`` file.

        Returns:
            A fully populated :class:`TaskEmbeddingRegistry`.

        Raises:
            ImportError: If pandas is not available.
            FileNotFoundError: If the file does not exist.
        """
        if not _HAS_PANDAS:
            raise ImportError(
                "pandas is required for Parquet support. "
                "Install with: pip install pandas pyarrow"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        # Try to read header metadata via pyarrow
        header_dict: Dict[str, Any] = {}
        try:
            import pyarrow.parquet as pq

            pf = pq.read_schema(str(path))
            meta = pf.metadata or {}
            raw = meta.get(b"_registry_header", None)
            if raw is not None:
                header_dict = json.loads(raw.decode())
        except (ImportError, Exception):
            pass

        df = pd.read_parquet(str(path))

        if len(df) == 0 and not header_dict:
            # Empty parquet with no header — create minimal registry
            embedding_dim = 512
            if header_dict:
                embedding_dim = int(header_dict.get("_embedding_dim", 512))
            return cls(embedding_dim=embedding_dim)

        # Determine embedding dim
        if header_dict:
            embedding_dim = int(header_dict.get("_embedding_dim", 512))
        elif "embedding_bytes" in df.columns and len(df) > 0:
            first_bytes = df["embedding_bytes"].iloc[0]
            if isinstance(first_bytes, bytes):
                embedding_dim = len(first_bytes) // 4  # float32
            else:
                embedding_dim = 512
        else:
            embedding_dim = 512

        registry = cls(embedding_dim=embedding_dim)
        if header_dict:
            header = RegistryHeader.from_dict(header_dict)
            registry._created = header.created

        for _, row in df.iterrows():
            rec = dict(row)

            # Decode JSON-encoded fields
            if isinstance(rec.get("diagnostics"), str):
                rec["diagnostics"] = json.loads(rec["diagnostics"])
            if isinstance(rec.get("class_ids"), str):
                rec["class_ids"] = json.loads(rec["class_ids"])
            if isinstance(rec.get("support_indices"), str):
                rec["support_indices"] = json.loads(rec["support_indices"])

            # Reconstruct embedding
            emb_bytes = rec.pop("embedding_bytes", None)
            if emb_bytes is not None and isinstance(emb_bytes, bytes):
                emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
            else:
                emb = np.zeros(embedding_dim, dtype=np.float32)

            entry = RegistryEntry.from_meta_dict(rec)
            entry.embedding = emb
            registry._entries[entry.task_id] = entry
            registry._insertion_order.append(entry.task_id)
            if entry.probe_signature:
                registry._probe_signatures.add(entry.probe_signature)

        logger.info(
            "Loaded registry from Parquet %s (%d entries, embedding_dim=%d)",
            path,
            len(registry),
            embedding_dim,
        )
        return registry

    # ------------------------------------------------------------------
    # Analysis operations
    # ------------------------------------------------------------------

    def embedding_matrix(self) -> np.ndarray:
        """Return all embeddings as a contiguous ``(N, E)`` float32 matrix.

        Rows are ordered by insertion order.

        Returns:
            ndarray of shape ``(len(self), embedding_dim)``.
        """
        if len(self) == 0:
            return np.empty((0, self._embedding_dim), dtype=np.float32)
        return np.stack(
            [self._entries[tid].embedding for tid in self._insertion_order],
            axis=0,
        )

    def merge(self, other: "TaskEmbeddingRegistry") -> "TaskEmbeddingRegistry":
        """Combine this registry with *other*, returning a new registry.

        When both registries contain the same ``task_id``, the entry from
        *other* wins (later data is fresher).  The embedding dimension must
        match.

        Parameters:
            other: Another registry to merge in.

        Returns:
            A new :class:`TaskEmbeddingRegistry` with entries from both.

        Raises:
            ValueError: If embedding dimensions differ.
        """
        if self._embedding_dim != other._embedding_dim:
            raise ValueError(
                f"Cannot merge registries with different embedding dims: "
                f"{self._embedding_dim} vs {other._embedding_dim}"
            )

        merged = TaskEmbeddingRegistry(embedding_dim=self._embedding_dim)
        merged._created = min(self._created, other._created)

        # Add self entries first
        for tid in self._insertion_order:
            entry = self._entries[tid].clone()
            merged._entries[tid] = entry
            merged._insertion_order.append(tid)
            if entry.probe_signature:
                merged._probe_signatures.add(entry.probe_signature)

        # Add/overwrite with other entries
        for tid in other._insertion_order:
            entry = other._entries[tid].clone()
            if tid in merged._entries:
                # Overwrite — remove from insertion order then re-add at end
                merged._entries[tid] = entry
                merged._insertion_order = [
                    t for t in merged._insertion_order if t != tid
                ]
                merged._insertion_order.append(tid)
            else:
                merged._entries[tid] = entry
                merged._insertion_order.append(tid)
            if entry.probe_signature:
                merged._probe_signatures.add(entry.probe_signature)

        logger.info(
            "Merged registries: %d + %d -> %d entries",
            len(self),
            len(other),
            len(merged),
        )
        return merged

    def diff(self, other: "TaskEmbeddingRegistry") -> Dict[str, Any]:
        """Compare this registry with *other* and describe the differences.

        Parameters:
            other: The registry to compare against.

        Returns:
            A dict with keys:

            - ``"added"``: task_ids in *other* but not in *self*.
            - ``"removed"``: task_ids in *self* but not in *other*.
            - ``"changed"``: task_ids in both whose embeddings differ.
            - ``"unchanged"``: task_ids in both with identical embeddings.
            - ``"self_count"``: number of entries in *self*.
            - ``"other_count"``: number of entries in *other*.
        """
        self_ids = set(self._entries.keys())
        other_ids = set(other._entries.keys())

        added = sorted(other_ids - self_ids)
        removed = sorted(self_ids - other_ids)

        changed: List[str] = []
        unchanged: List[str] = []
        for tid in sorted(self_ids & other_ids):
            h_self = _compute_embedding_hash(self._entries[tid].embedding)
            h_other = _compute_embedding_hash(other._entries[tid].embedding)
            if h_self == h_other:
                unchanged.append(tid)
            else:
                changed.append(tid)

        return {
            "added": added,
            "removed": removed,
            "changed": changed,
            "unchanged": unchanged,
            "self_count": len(self),
            "other_count": len(other),
        }

    def summary(self) -> Dict[str, Any]:
        """Return aggregate statistics about the registry.

        Returns:
            A dict with keys:

            - ``"n_entries"``: total number of entries.
            - ``"embedding_dim"``: embedding dimensionality.
            - ``"datasets"``: sorted list of unique dataset names.
            - ``"splits"``: sorted list of unique splits.
            - ``"n_way_values"``: sorted list of unique n_way values.
            - ``"k_shot_values"``: sorted list of unique k_shot values.
            - ``"probe_signatures"``: sorted list of unique probe sigs.
            - ``"created"``: registry creation timestamp.
            - ``"format_version"``: format version string.
            - ``"embedding_norm_mean"``: mean L2 norm of embeddings.
            - ``"embedding_norm_std"``: std of L2 norms.
        """
        datasets: Set[str] = set()
        splits: Set[str] = set()
        n_ways: Set[int] = set()
        k_shots: Set[int] = set()
        norms: List[float] = []

        for entry in self._entries.values():
            if entry.dataset:
                datasets.add(entry.dataset)
            if entry.split:
                splits.add(entry.split)
            if entry.n_way > 0:
                n_ways.add(entry.n_way)
            if entry.k_shot > 0:
                k_shots.add(entry.k_shot)
            norms.append(float(np.linalg.norm(entry.embedding)))

        norm_arr = np.array(norms) if norms else np.array([0.0])

        return {
            "n_entries": len(self),
            "embedding_dim": self._embedding_dim,
            "datasets": sorted(datasets),
            "splits": sorted(splits),
            "n_way_values": sorted(n_ways),
            "k_shot_values": sorted(k_shots),
            "probe_signatures": sorted(self._probe_signatures),
            "created": self._created,
            "format_version": _FORMAT_VERSION,
            "embedding_norm_mean": float(norm_arr.mean()),
            "embedding_norm_std": float(norm_arr.std()),
        }

    # ------------------------------------------------------------------
    # Torch interop
    # ------------------------------------------------------------------

    def to_tensor(self, task_ids: Optional[List[str]] = None) -> Any:
        """Return embeddings as a torch.Tensor (if torch is available).

        Parameters:
            task_ids: Optional subset of task_ids.  If ``None``, returns
                all embeddings in insertion order.

        Returns:
            A ``torch.Tensor`` of shape ``(N, E)`` on CPU.

        Raises:
            ImportError: If torch is not installed.
        """
        if not _HAS_TORCH:
            raise ImportError("torch is required for to_tensor()")
        if task_ids is None:
            mat = self.embedding_matrix()
        else:
            mat = self.get_embeddings(task_ids)
        return torch.from_numpy(mat)

    # ------------------------------------------------------------------
    # Cosine distance utilities
    # ------------------------------------------------------------------

    def pairwise_cosine_distance(
        self, task_ids: Optional[List[str]] = None
    ) -> np.ndarray:
        """Compute the pairwise cosine distance matrix.

        Cosine distance is defined as ``1 - cosine_similarity``.

        Parameters:
            task_ids: Subset of tasks to compute distances for.
                If ``None``, uses all tasks.

        Returns:
            Symmetric ``(N, N)`` float32 array of cosine distances.
        """
        if task_ids is None:
            mat = self.embedding_matrix()
        else:
            mat = self.get_embeddings(task_ids)

        if mat.shape[0] == 0:
            return np.empty((0, 0), dtype=np.float32)

        # Normalise rows
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normed = mat / norms

        # Cosine similarity
        cos_sim = normed @ normed.T
        # Clip for numerical safety
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        return (1.0 - cos_sim).astype(np.float32)

    def nearest_tasks(
        self, task_id: str, k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find the *k* nearest tasks by cosine distance.

        Parameters:
            task_id: The reference task.
            k: Number of neighbours to return.

        Returns:
            List of ``(task_id, cosine_distance)`` tuples, sorted by
            ascending distance.  Does not include *task_id* itself.

        Raises:
            KeyError: If *task_id* is not in the registry.
        """
        if task_id not in self._entries:
            raise KeyError(f"task_id '{task_id}' not found")

        ref_emb = self._entries[task_id].embedding
        ref_norm = np.linalg.norm(ref_emb)
        if ref_norm < 1e-12:
            ref_norm = 1e-12

        results: List[Tuple[str, float]] = []
        for tid in self._insertion_order:
            if tid == task_id:
                continue
            other_emb = self._entries[tid].embedding
            other_norm = np.linalg.norm(other_emb)
            if other_norm < 1e-12:
                other_norm = 1e-12
            cos_sim = float(np.dot(ref_emb, other_emb) / (ref_norm * other_norm))
            cos_dist = 1.0 - cos_sim
            results.append((tid, cos_dist))

        results.sort(key=lambda x: x[1])
        return results[:k]

    # ------------------------------------------------------------------
    # Iteration and slicing utilities
    # ------------------------------------------------------------------

    def entries(self) -> List[RegistryEntry]:
        """Return all entries as a list of deep copies in insertion order."""
        return [self._entries[tid].clone() for tid in self._insertion_order]

    def subset(self, task_ids: List[str]) -> "TaskEmbeddingRegistry":
        """Create a new registry containing only the specified task_ids.

        Parameters:
            task_ids: The subset of tasks to keep.

        Returns:
            A new :class:`TaskEmbeddingRegistry` with only the requested
            entries.

        Raises:
            KeyError: If any task_id is not found.
        """
        sub = TaskEmbeddingRegistry(embedding_dim=self._embedding_dim)
        sub._created = self._created
        for tid in task_ids:
            if tid not in self._entries:
                raise KeyError(f"task_id '{tid}' not found in registry")
            entry = self._entries[tid].clone()
            sub._entries[tid] = entry
            sub._insertion_order.append(tid)
            if entry.probe_signature:
                sub._probe_signatures.add(entry.probe_signature)
        return sub

    def filter(
        self,
        predicate: Any,  # Callable[[RegistryEntry], bool]
    ) -> "TaskEmbeddingRegistry":
        """Create a new registry containing entries that satisfy *predicate*.

        Parameters:
            predicate: A callable ``(RegistryEntry) -> bool``.

        Returns:
            A new :class:`TaskEmbeddingRegistry`.
        """
        sub = TaskEmbeddingRegistry(embedding_dim=self._embedding_dim)
        sub._created = self._created
        for tid in self._insertion_order:
            entry = self._entries[tid]
            if predicate(entry):
                cloned = entry.clone()
                sub._entries[tid] = cloned
                sub._insertion_order.append(tid)
                if cloned.probe_signature:
                    sub._probe_signatures.add(cloned.probe_signature)
        return sub

    # ------------------------------------------------------------------
    # Integrity checking
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Run internal consistency checks.

        Returns:
            A list of warning/error messages.  An empty list means the
            registry is consistent.
        """
        issues: List[str] = []

        # Check insertion order vs entries dict
        if set(self._insertion_order) != set(self._entries.keys()):
            issues.append(
                "Insertion order keys do not match entry dict keys"
            )

        if len(self._insertion_order) != len(set(self._insertion_order)):
            issues.append("Duplicate task_ids in insertion order")

        # Check embedding dims
        for tid, entry in self._entries.items():
            if entry.embedding.shape != (self._embedding_dim,):
                issues.append(
                    f"Entry '{tid}' has embedding shape {entry.embedding.shape}, "
                    f"expected ({self._embedding_dim},)"
                )

        # Check probe signature consistency
        sigs = {e.probe_signature for e in self._entries.values() if e.probe_signature}
        if len(sigs) > 1:
            issues.append(
                f"Mixed probe signatures detected: {sigs}"
            )

        return issues


# ============================================================================
# Self-test block
# ============================================================================

if __name__ == "__main__":
    import sys
    import traceback

    logging.basicConfig(level=logging.WARNING)

    _pass_count = 0
    _fail_count = 0
    _test_number = 0

    def _report(test_name: str, passed: bool, detail: str = "") -> None:
        global _pass_count, _fail_count, _test_number
        _test_number += 1
        status = "PASS" if passed else "FAIL"
        if passed:
            _pass_count += 1
        else:
            _fail_count += 1
        suffix = f" -- {detail}" if detail else ""
        print(f"  [{status}] Test {_test_number:02d}: {test_name}{suffix}")

    def _run_test(test_name: str, fn: Any) -> None:
        try:
            fn()
        except Exception as exc:
            global _fail_count, _test_number
            _test_number += 1
            _fail_count += 1
            print(f"  [FAIL] Test {_test_number:02d}: {test_name} -- EXCEPTION: {exc}")
            traceback.print_exc()

    print("=" * 72)
    print("TaskEmbeddingRegistry Self-Tests")
    print("=" * 72)

    DIM = 64  # small dim for fast tests
    rng = np.random.RandomState(42)

    # Helper to create random embedding
    def _rand_emb(dim: int = DIM) -> np.ndarray:
        v = rng.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        return v

    # ---- Test 01: Empty registry ----
    def test_01_empty_registry() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        _report("Empty registry has len == 0", len(reg) == 0, f"len={len(reg)}")

    _run_test("Empty registry has len == 0", test_01_empty_registry)

    # ---- Test 02: Add single entry and retrieve ----
    def test_02_single_entry() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        emb = _rand_emb()
        reg.update("task_001", emb, dataset="omniglot", split="train", n_way=5, k_shot=1)
        ok = len(reg) == 1 and "task_001" in reg
        _report("Add single entry, verify retrieval", ok, f"len={len(reg)}")

    _run_test("Add single entry", test_02_single_entry)

    # ---- Test 03: Batch add entries ----
    def test_03_batch_add() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        entries = []
        for i in range(10):
            e = RegistryEntry(
                task_id=f"batch_{i:03d}",
                embedding=_rand_emb(),
                dataset="mini-imagenet",
                split="train",
                n_way=5,
                k_shot=5,
            )
            entries.append(e)
        reg.update_batch(entries)
        all_found = all(f"batch_{i:03d}" in reg for i in range(10))
        _report(
            "Batch add 10 entries, all retrievable",
            len(reg) == 10 and all_found,
            f"len={len(reg)}",
        )

    _run_test("Batch add entries", test_03_batch_add)

    # ---- Test 04: get_embedding returns correct vector ----
    def test_04_get_embedding() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        emb = _rand_emb()
        reg.update("vec_test", emb, dataset="d")
        retrieved = reg.get_embedding("vec_test")
        match = np.allclose(emb, retrieved, atol=1e-7)
        _report("get_embedding returns correct vector", match)

    _run_test("get_embedding correctness", test_04_get_embedding)

    # ---- Test 05: get_embeddings batch returns correct shape ----
    def test_05_get_embeddings_batch() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        for i in range(5):
            reg.update(f"b_{i}", _rand_emb(), dataset="d")
        ids = [f"b_{i}" for i in range(5)]
        mat = reg.get_embeddings(ids)
        ok = mat.shape == (5, DIM) and mat.dtype == np.float32
        _report(
            "get_embeddings batch shape (N, E)",
            ok,
            f"shape={mat.shape}, dtype={mat.dtype}",
        )

    _run_test("get_embeddings batch shape", test_05_get_embeddings_batch)

    # ---- Test 06: query by dataset ----
    def test_06_query_dataset() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg.update("d1_t1", _rand_emb(), dataset="omniglot")
        reg.update("d1_t2", _rand_emb(), dataset="omniglot")
        reg.update("d2_t1", _rand_emb(), dataset="cifar")
        result = reg.query(dataset="omniglot")
        ok = set(result) == {"d1_t1", "d1_t2"}
        _report("Query by dataset filters correctly", ok, f"result={result}")

    _run_test("Query by dataset", test_06_query_dataset)

    # ---- Test 07: query by split ----
    def test_07_query_split() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg.update("s1", _rand_emb(), split="train")
        reg.update("s2", _rand_emb(), split="val")
        reg.update("s3", _rand_emb(), split="train")
        result = reg.query(split="train")
        ok = set(result) == {"s1", "s3"}
        _report("Query by split filters correctly", ok, f"result={result}")

    _run_test("Query by split", test_07_query_split)

    # ---- Test 08: query by n_way ----
    def test_08_query_nway() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg.update("nw5_1", _rand_emb(), n_way=5)
        reg.update("nw5_2", _rand_emb(), n_way=5)
        reg.update("nw20", _rand_emb(), n_way=20)
        result = reg.query(n_way=5)
        ok = set(result) == {"nw5_1", "nw5_2"}
        _report("Query by n_way filters correctly", ok, f"result={result}")

    _run_test("Query by n_way", test_08_query_nway)

    # ---- Test 09: combined query (dataset + split) ----
    def test_09_combined_query() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg.update("c1", _rand_emb(), dataset="omni", split="train")
        reg.update("c2", _rand_emb(), dataset="omni", split="val")
        reg.update("c3", _rand_emb(), dataset="cifar", split="train")
        reg.update("c4", _rand_emb(), dataset="omni", split="train")
        result = reg.query(dataset="omni", split="train")
        ok = set(result) == {"c1", "c4"}
        _report("Combined query (dataset + split)", ok, f"result={result}")

    _run_test("Combined query", test_09_combined_query)

    # ---- Test 10: Save/load JSONL+NPZ round-trip ----
    def test_10_save_load_roundtrip() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        embs = {}
        for i in range(8):
            e = _rand_emb()
            tid = f"rt_{i:03d}"
            reg.update(
                tid, e,
                dataset="ds" if i < 4 else "other",
                split="train",
                n_way=5,
                k_shot=1,
                probe_signature="probe_abc",
                extraction_seed=42,
            )
            embs[tid] = e

        with tempfile.TemporaryDirectory() as tmpdir:
            reg.save(Path(tmpdir))
            loaded = TaskEmbeddingRegistry.load(Path(tmpdir))

        ok_len = len(loaded) == 8
        ok_ids = set(loaded.all_task_ids()) == set(embs.keys())
        ok_embs = all(
            np.allclose(loaded.get_embedding(tid), embs[tid], atol=1e-7)
            for tid in embs
        )
        all_ok = ok_len and ok_ids and ok_embs
        _report(
            "Save/load JSONL+NPZ round-trip preserves all entries",
            all_ok,
            f"len_ok={ok_len}, ids_ok={ok_ids}, embs_ok={ok_embs}",
        )

    _run_test("Save/load round-trip", test_10_save_load_roundtrip)

    # ---- Test 11: Loaded embeddings match originals ----
    def test_11_loaded_embeddings_match() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        expected = _rand_emb()
        reg.update("match_test", expected, dataset="d")
        with tempfile.TemporaryDirectory() as tmpdir:
            reg.save(Path(tmpdir))
            loaded = TaskEmbeddingRegistry.load(Path(tmpdir))
        actual = loaded.get_embedding("match_test")
        match = np.allclose(expected, actual, atol=1e-7)
        max_diff = float(np.max(np.abs(expected - actual)))
        _report(
            "Loaded embedding matches original within tolerance",
            match,
            f"max_diff={max_diff:.2e}",
        )

    _run_test("Loaded embeddings match", test_11_loaded_embeddings_match)

    # ---- Test 12: JSONL first line is valid header ----
    def test_12_jsonl_header() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg.update("hdr_test", _rand_emb())
        with tempfile.TemporaryDirectory() as tmpdir:
            reg.save(Path(tmpdir))
            jsonl_path = Path(tmpdir) / _JSONL_FILENAME
            with open(jsonl_path, "r") as f:
                first_line = f.readline().strip()
            header = json.loads(first_line)
        has_version = "_version" in header
        has_dim = "_embedding_dim" in header
        correct_dim = header.get("_embedding_dim") == DIM
        ok = has_version and has_dim and correct_dim
        _report(
            "JSONL first line is valid header",
            ok,
            f"version={header.get('_version')}, dim={header.get('_embedding_dim')}",
        )

    _run_test("JSONL header validation", test_12_jsonl_header)

    # ---- Test 13: NPZ contains 'embeddings' key ----
    def test_13_npz_key() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg.update("npz_test", _rand_emb())
        with tempfile.TemporaryDirectory() as tmpdir:
            reg.save(Path(tmpdir))
            npz_path = Path(tmpdir) / _NPZ_FILENAME
            with np.load(str(npz_path)) as npz:
                has_key = _EMBEDDING_NPZ_KEY in npz
                shape = npz[_EMBEDDING_NPZ_KEY].shape if has_key else None
        ok = has_key and shape == (1, DIM)
        _report(
            "NPZ contains 'embeddings' key with correct shape",
            ok,
            f"has_key={has_key}, shape={shape}",
        )

    _run_test("NPZ key check", test_13_npz_key)

    # ---- Test 14: Duplicate task_id overwrites ----
    def test_14_duplicate_overwrite() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        emb1 = _rand_emb()
        emb2 = _rand_emb()
        reg.update("dup", emb1, dataset="old")
        reg.update("dup", emb2, dataset="new")
        ok_len = len(reg) == 1
        ok_emb = np.allclose(reg.get_embedding("dup"), emb2, atol=1e-7)
        ok_ds = reg.get_entry("dup").dataset == "new"
        ok = ok_len and ok_emb and ok_ds
        _report(
            "Duplicate task_id overwrites correctly",
            ok,
            f"len={len(reg)}, emb_match={ok_emb}, ds={reg.get_entry('dup').dataset}",
        )

    _run_test("Duplicate overwrite", test_14_duplicate_overwrite)

    # ---- Test 15: __contains__ ----
    def test_15_contains() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg.update("exists", _rand_emb())
        ok_pos = "exists" in reg
        ok_neg = "missing" not in reg
        ok = ok_pos and ok_neg
        _report(
            "__contains__ returns True/False correctly",
            ok,
            f"exists={ok_pos}, missing_absent={ok_neg}",
        )

    _run_test("__contains__", test_15_contains)

    # ---- Test 16: merge combines registries ----
    def test_16_merge() -> None:
        reg_a = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg_b = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg_a.update("a1", _rand_emb(), dataset="da")
        reg_a.update("shared", _rand_emb(), dataset="da")
        reg_b.update("b1", _rand_emb(), dataset="db")
        reg_b.update("shared", _rand_emb(), dataset="db")  # overwrites a's shared
        merged = reg_a.merge(reg_b)
        ok_len = len(merged) == 3  # a1, shared (from b), b1
        ok_ids = set(merged.all_task_ids()) == {"a1", "shared", "b1"}
        # 'shared' should come from reg_b
        ok_ds = merged.get_entry("shared").dataset == "db"
        ok = ok_len and ok_ids and ok_ds
        _report(
            "Merge combines registries, other wins on conflict",
            ok,
            f"len={len(merged)}, ids={merged.all_task_ids()}, shared_ds={merged.get_entry('shared').dataset}",
        )

    _run_test("Merge registries", test_16_merge)

    # ---- Test 17: diff detects added/removed ----
    def test_17_diff() -> None:
        reg_a = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg_b = TaskEmbeddingRegistry(embedding_dim=DIM)
        shared_emb = _rand_emb()
        reg_a.update("only_a", _rand_emb())
        reg_a.update("shared", shared_emb.copy())
        reg_b.update("shared", shared_emb.copy())
        reg_b.update("only_b", _rand_emb())
        d = reg_a.diff(reg_b)
        ok_added = d["added"] == ["only_b"]
        ok_removed = d["removed"] == ["only_a"]
        ok_unchanged = d["unchanged"] == ["shared"]
        ok_changed = d["changed"] == []
        ok = ok_added and ok_removed and ok_unchanged and ok_changed
        _report(
            "Diff detects added/removed/unchanged entries",
            ok,
            f"added={d['added']}, removed={d['removed']}, unchanged={d['unchanged']}, changed={d['changed']}",
        )

    _run_test("Diff detection", test_17_diff)

    # ---- Test 18: embedding_matrix shape ----
    def test_18_embedding_matrix() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        for i in range(7):
            reg.update(f"em_{i}", _rand_emb())
        mat = reg.embedding_matrix()
        ok = mat.shape == (7, DIM) and mat.dtype == np.float32
        _report(
            "embedding_matrix returns correct shape (N, E)",
            ok,
            f"shape={mat.shape}, dtype={mat.dtype}",
        )

    _run_test("embedding_matrix shape", test_18_embedding_matrix)

    # ---- Test 19: summary contains expected keys ----
    def test_19_summary() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg.update("sum1", _rand_emb(), dataset="omni", split="train", n_way=5, k_shot=1)
        reg.update("sum2", _rand_emb(), dataset="cifar", split="val", n_way=10, k_shot=5)
        s = reg.summary()
        expected_keys = {
            "n_entries", "embedding_dim", "datasets", "splits",
            "n_way_values", "k_shot_values", "probe_signatures",
            "created", "format_version", "embedding_norm_mean",
            "embedding_norm_std",
        }
        ok_keys = expected_keys.issubset(set(s.keys()))
        ok_n = s["n_entries"] == 2
        ok_ds = set(s["datasets"]) == {"cifar", "omni"}
        ok = ok_keys and ok_n and ok_ds
        _report(
            "Summary contains expected keys and correct values",
            ok,
            f"keys_ok={ok_keys}, n={s['n_entries']}, ds={s['datasets']}",
        )

    _run_test("Summary keys", test_19_summary)

    # ---- Test 20: Empty save/load ----
    def test_20_empty_save_load() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        with tempfile.TemporaryDirectory() as tmpdir:
            reg.save(Path(tmpdir))
            loaded = TaskEmbeddingRegistry.load(Path(tmpdir))
        ok = len(loaded) == 0 and loaded.embedding_dim == DIM
        _report(
            "Empty registry save/load works correctly",
            ok,
            f"len={len(loaded)}, dim={loaded.embedding_dim}",
        )

    _run_test("Empty save/load", test_20_empty_save_load)

    # ---- Test 21: Version header present after load ----
    def test_21_version_header() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg.update("vh1", _rand_emb())
        with tempfile.TemporaryDirectory() as tmpdir:
            reg.save(Path(tmpdir))
            jsonl_path = Path(tmpdir) / _JSONL_FILENAME
            with open(jsonl_path, "r") as f:
                first = json.loads(f.readline())
            loaded = TaskEmbeddingRegistry.load(Path(tmpdir))
        ok_header = first.get("_version") == _FORMAT_VERSION
        ok_loaded = len(loaded) == 1
        _report(
            "Version header present and correct after save/load",
            ok_header and ok_loaded,
            f"version={first.get('_version')}",
        )

    _run_test("Version header after load", test_21_version_header)

    # ---- Test 22: Row index consistency ----
    def test_22_row_index_consistency() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        embs = {}
        for i in range(5):
            tid = f"ri_{i}"
            e = _rand_emb()
            reg.update(tid, e)
            embs[tid] = e

        with tempfile.TemporaryDirectory() as tmpdir:
            reg.save(Path(tmpdir))
            jsonl_path = Path(tmpdir) / _JSONL_FILENAME
            npz_path = Path(tmpdir) / _NPZ_FILENAME

            with open(jsonl_path, "r") as f:
                lines = f.read().strip().split("\n")
            entry_lines = lines[1:]

            with np.load(str(npz_path)) as npz:
                mat = npz[_EMBEDDING_NPZ_KEY]

            all_ok = True
            for line in entry_lines:
                d = json.loads(line)
                tid = d["task_id"]
                row_idx = d["row_index"]
                stored_emb = mat[row_idx]
                if not np.allclose(stored_emb, embs[tid], atol=1e-7):
                    all_ok = False
                    break

        _report(
            "Row index links JSONL entry to correct NPZ row",
            all_ok,
        )

    _run_test("Row index consistency", test_22_row_index_consistency)

    # ---- Test 23: all_task_ids preserves insertion order ----
    def test_23_insertion_order() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        order = ["zz", "aa", "mm", "bb", "xx"]
        for tid in order:
            reg.update(tid, _rand_emb())
        retrieved = reg.all_task_ids()
        ok = retrieved == order
        _report(
            "all_task_ids preserves insertion order",
            ok,
            f"expected={order}, got={retrieved}",
        )

    _run_test("Insertion order", test_23_insertion_order)

    # ---- Test 24: Embedding dim mismatch raises ValueError ----
    def test_24_dim_mismatch() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        wrong_dim = np.zeros(DIM + 10, dtype=np.float32)
        try:
            reg.update("bad_dim", wrong_dim)
            _report("Embedding dim mismatch raises ValueError", False, "No exception raised")
        except ValueError:
            _report("Embedding dim mismatch raises ValueError", True)

    _run_test("Dim mismatch error", test_24_dim_mismatch)

    # ---- Test 25: get_entry returns deep copy ----
    def test_25_deep_copy() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        emb = _rand_emb()
        reg.update("copy_test", emb, dataset="original")
        entry = reg.get_entry("copy_test")
        entry.dataset = "mutated"
        entry.embedding[0] = 999.0
        # Original should be unaffected
        original = reg.get_entry("copy_test")
        ok_ds = original.dataset == "original"
        ok_emb = original.embedding[0] != 999.0
        _report(
            "get_entry returns deep copy (mutations don't affect registry)",
            ok_ds and ok_emb,
        )

    _run_test("Deep copy isolation", test_25_deep_copy)

    # ---- Test 26: validate on healthy registry returns empty list ----
    def test_26_validate_healthy() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        for i in range(3):
            reg.update(f"v_{i}", _rand_emb(), probe_signature="same_probe")
        issues = reg.validate()
        ok = len(issues) == 0
        _report("Validate on healthy registry returns no issues", ok, f"issues={issues}")

    _run_test("Validate healthy", test_26_validate_healthy)

    # ---- Test 27: pairwise_cosine_distance shape ----
    def test_27_cosine_distance() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        for i in range(4):
            reg.update(f"cd_{i}", _rand_emb())
        dist = reg.pairwise_cosine_distance()
        ok_shape = dist.shape == (4, 4)
        # Diagonal should be ~0
        ok_diag = np.allclose(np.diag(dist), 0.0, atol=1e-5)
        # Symmetric
        ok_sym = np.allclose(dist, dist.T, atol=1e-6)
        ok = ok_shape and ok_diag and ok_sym
        _report(
            "Pairwise cosine distance: shape, diagonal, symmetry",
            ok,
            f"shape={dist.shape}, diag_max={np.max(np.diag(dist)):.2e}",
        )

    _run_test("Cosine distance matrix", test_27_cosine_distance)

    # ---- Test 28: nearest_tasks returns sorted results ----
    def test_28_nearest_tasks() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        # Create a reference and some neighbours
        ref = np.ones(DIM, dtype=np.float32)
        ref /= np.linalg.norm(ref)
        reg.update("ref", ref)
        # Close neighbour (small perturbation)
        close = ref.copy()
        close[0] += 0.01
        close /= np.linalg.norm(close)
        reg.update("close", close)
        # Far neighbour (orthogonal-ish)
        far = _rand_emb()
        reg.update("far", far)

        neighbours = reg.nearest_tasks("ref", k=2)
        ok_len = len(neighbours) == 2
        ok_order = neighbours[0][0] == "close"  # close should be first
        ok = ok_len and ok_order
        _report(
            "nearest_tasks returns sorted by distance",
            ok,
            f"neighbours={[(n, f'{d:.4f}') for n, d in neighbours]}",
        )

    _run_test("Nearest tasks", test_28_nearest_tasks)

    # ---- Test 29: subset creates correct sub-registry ----
    def test_29_subset() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        for i in range(6):
            reg.update(f"sub_{i}", _rand_emb())
        sub = reg.subset(["sub_1", "sub_3", "sub_5"])
        ok_len = len(sub) == 3
        ok_ids = set(sub.all_task_ids()) == {"sub_1", "sub_3", "sub_5"}
        ok = ok_len and ok_ids
        _report("Subset creates correct sub-registry", ok, f"len={len(sub)}")

    _run_test("Subset", test_29_subset)

    # ---- Test 30: diff detects changed embeddings ----
    def test_30_diff_changed() -> None:
        reg_a = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg_b = TaskEmbeddingRegistry(embedding_dim=DIM)
        emb_orig = _rand_emb()
        emb_diff = _rand_emb()
        reg_a.update("same_id", emb_orig)
        reg_b.update("same_id", emb_diff)
        d = reg_a.diff(reg_b)
        ok = d["changed"] == ["same_id"] and d["added"] == [] and d["removed"] == []
        _report(
            "Diff detects changed embedding for same task_id",
            ok,
            f"changed={d['changed']}",
        )

    _run_test("Diff changed embeddings", test_30_diff_changed)

    # ---- Test 31: remove entry ----
    def test_31_remove() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg.update("rm_1", _rand_emb())
        reg.update("rm_2", _rand_emb())
        removed = reg.remove("rm_1")
        ok = removed and len(reg) == 1 and "rm_1" not in reg and "rm_2" in reg
        not_found = not reg.remove("nonexistent")
        _report(
            "Remove entry works correctly",
            ok and not_found,
            f"len={len(reg)}, rm_1_gone={'rm_1' not in reg}",
        )

    _run_test("Remove entry", test_31_remove)

    # ---- Test 32: filter creates filtered registry ----
    def test_32_filter() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg.update("f1", _rand_emb(), n_way=5)
        reg.update("f2", _rand_emb(), n_way=10)
        reg.update("f3", _rand_emb(), n_way=5)
        filtered = reg.filter(lambda e: e.n_way == 5)
        ok = len(filtered) == 2 and set(filtered.all_task_ids()) == {"f1", "f3"}
        _report("Filter creates correct filtered registry", ok, f"len={len(filtered)}")

    _run_test("Filter registry", test_32_filter)

    # ---- Test 33: torch tensor conversion ----
    def test_33_torch_tensor() -> None:
        if not _HAS_TORCH:
            _report("Torch tensor conversion (skipped — torch not installed)", True, "SKIP")
            return
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        for i in range(3):
            reg.update(f"torch_{i}", _rand_emb())
        t = reg.to_tensor()
        ok_type = isinstance(t, torch.Tensor)
        ok_shape = t.shape == (3, DIM)
        ok = ok_type and ok_shape
        _report("to_tensor returns correct torch.Tensor", ok, f"shape={t.shape}")

    _run_test("Torch tensor conversion", test_33_torch_tensor)

    # ---- Test 34: torch tensor as input ----
    def test_34_torch_input() -> None:
        if not _HAS_TORCH:
            _report("Torch tensor as input (skipped — torch not installed)", True, "SKIP")
            return
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        t = torch.randn(DIM)
        reg.update("from_torch", t)
        retrieved = reg.get_embedding("from_torch")
        expected = t.numpy()
        ok = np.allclose(retrieved, expected, atol=1e-6)
        _report("Torch tensor accepted as embedding input", ok)

    _run_test("Torch tensor input", test_34_torch_input)

    # ---- Test 35: multiple probe signatures warning ----
    def test_35_multi_probe() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg.update("p1", _rand_emb(), probe_signature="probe_a")
        reg.update("p2", _rand_emb(), probe_signature="probe_b")
        issues = reg.validate()
        has_warning = any("probe" in i.lower() for i in issues)
        ok = has_warning and len(reg.probe_signatures) == 2
        _report(
            "Mixed probe signatures detected by validate",
            ok,
            f"signatures={reg.probe_signatures}, issues={issues}",
        )

    _run_test("Multi probe signatures", test_35_multi_probe)

    # ---- Test 36: save/load round-trip preserves metadata ----
    def test_36_metadata_roundtrip() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        reg.update(
            "meta_rt",
            _rand_emb(),
            dataset="mini-imagenet",
            split="val",
            n_way=5,
            k_shot=5,
            q_query=15,
            class_ids=[2, 7, 13, 19, 25],
            support_indices=[0, 1, 2, 3, 4],
            probe_signature="probe_xyz_123",
            extraction_seed=1337,
            code_version="1.0",
            diagnostics={"fisher_norm": 42.5, "sparsity": 0.7},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            reg.save(Path(tmpdir))
            loaded = TaskEmbeddingRegistry.load(Path(tmpdir))
        e = loaded.get_entry("meta_rt")
        ok = (
            e.dataset == "mini-imagenet"
            and e.split == "val"
            and e.n_way == 5
            and e.k_shot == 5
            and e.q_query == 15
            and e.class_ids == [2, 7, 13, 19, 25]
            and e.support_indices == [0, 1, 2, 3, 4]
            and e.probe_signature == "probe_xyz_123"
            and e.extraction_seed == 1337
            and e.diagnostics.get("fisher_norm") == 42.5
            and e.diagnostics.get("sparsity") == 0.7
        )
        _report("Save/load round-trip preserves all metadata fields", ok)

    _run_test("Metadata round-trip", test_36_metadata_roundtrip)

    # ---- Test 37: KeyError on missing task_id ----
    def test_37_key_error() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        try:
            reg.get_embedding("nonexistent")
            _report("KeyError on missing task_id", False, "No exception")
        except KeyError:
            _report("KeyError on missing task_id", True)

    _run_test("KeyError missing task_id", test_37_key_error)

    # ---- Test 38: embedding_matrix on empty registry ----
    def test_38_empty_matrix() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        mat = reg.embedding_matrix()
        ok = mat.shape == (0, DIM)
        _report("embedding_matrix on empty registry returns (0, E)", ok, f"shape={mat.shape}")

    _run_test("Empty embedding_matrix", test_38_empty_matrix)

    # ---- Test 39: large batch stress test ----
    def test_39_large_batch() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        n = 500
        entries = []
        for i in range(n):
            entries.append(
                RegistryEntry(
                    task_id=f"stress_{i:05d}",
                    embedding=_rand_emb(),
                    dataset=f"ds_{i % 5}",
                    split="train" if i % 3 == 0 else "val",
                    n_way=5 if i % 2 == 0 else 10,
                    k_shot=1,
                )
            )
        reg.update_batch(entries)
        ok_len = len(reg) == n
        mat = reg.embedding_matrix()
        ok_shape = mat.shape == (n, DIM)
        q = reg.query(dataset="ds_0", split="train")
        # ds_0 when i%5==0, train when i%3==0 -> i%15==0
        expected_count = len([i for i in range(n) if i % 5 == 0 and i % 3 == 0])
        ok_query = len(q) == expected_count
        ok = ok_len and ok_shape and ok_query
        _report(
            f"Large batch stress test ({n} entries)",
            ok,
            f"len={len(reg)}, mat={mat.shape}, query_match={len(q)}/{expected_count}",
        )

    _run_test("Large batch stress test", test_39_large_batch)

    # ---- Test 40: save/load large registry round-trip ----
    def test_40_large_roundtrip() -> None:
        reg = TaskEmbeddingRegistry(embedding_dim=DIM)
        n = 200
        for i in range(n):
            reg.update(f"lr_{i:04d}", _rand_emb(), dataset="big", split="train")
        with tempfile.TemporaryDirectory() as tmpdir:
            reg.save(Path(tmpdir))
            loaded = TaskEmbeddingRegistry.load(Path(tmpdir))
        ok_len = len(loaded) == n
        # Spot-check a few embeddings
        spot_ok = all(
            np.allclose(
                reg.get_embedding(f"lr_{i:04d}"),
                loaded.get_embedding(f"lr_{i:04d}"),
                atol=1e-7,
            )
            for i in range(0, n, 20)
        )
        ok = ok_len and spot_ok
        _report(
            f"Large registry ({n} entries) save/load round-trip",
            ok,
            f"len={len(loaded)}, spot_checks_pass={spot_ok}",
        )

    _run_test("Large round-trip", test_40_large_roundtrip)

    # ---- Summary ----
    print("=" * 72)
    total = _pass_count + _fail_count
    print(f"Results: {_pass_count}/{total} passed, {_fail_count}/{total} failed")
    if _fail_count == 0:
        print("All tests passed.")
    else:
        print(f"WARNING: {_fail_count} test(s) failed!")
    print("=" * 72)
    sys.exit(0 if _fail_count == 0 else 1)
