#!/usr/bin/env python3
"""
diff_spec.py — Compare spec.yaml against existing repo code/config files.

For each field in spec.yaml, scans the repository for config files (YAML, JSON,
Python) and reports matches, mismatches, and fields not found in any repo file.
Produces a Markdown diff report with per-field tables and actionable recommendations.

Heuristic matching strategy:
  1. Key normalisation: compare keys after stripping prefixes, converting
     camelCase/PascalCase to snake_case, and removing separators (``_``, ``-``).
  2. Dotted-path search: look for keys at any nesting depth that contain the
     spec field name as a suffix (e.g. spec path ``training.optimizer.lr`` matches
     a config key ``optimizer_lr`` or ``lr`` in an ``optimizer`` block).
  3. Value matching: numeric spec values are matched against numeric constants in
     Python files (assignments, dict literals, keyword arguments).
  4. Multi-format scan: YAML/JSON files are parsed and walked as dicts; Python
     files are scanned with regex for assignments and common config patterns.

Usage
-----
  python diff_spec.py --spec spec.yaml --repo-root /path/to/repo

  python diff_spec.py --spec spec.yaml --repo-root /path/to/repo \\
      --output reports/spec_diff.md \\
      --config-patterns "**/*.yaml,**/*.json,**/*.py"

Dependencies
------------
  stdlib only + PyYAML (pip install pyyaml)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required.  Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOL_VERSION = "0.1.0"
UNRESOLVED = "UNRESOLVED"

# Directories to skip unconditionally during repo scan.
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    "node_modules", ".venv", "venv", "env", ".tox", "dist", "build",
    ".paper_sources", "site-packages",
})

# Spec sections excluded from diffing (not framework config).
_SKIP_SECTIONS: frozenset[str] = frozenset({
    "meta", "frames", "imports",
})

# Keys within spec sections that carry provenance / bookkeeping, not config values.
_SKIP_KEYS: frozenset[str] = frozenset({
    "source", "sources", "name", "symbol", "latex", "expression_ast",
    "condition_ast", "description", "semantics", "notes",
    "informed_dreamer_key", "generated_at", "tool_version", "unresolved_count",
})

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FieldResult:
    """Comparison result for a single spec field."""

    spec_path: str          # Dotted path in spec (e.g. "training.optimizer.lr")
    spec_value: Any         # Value from spec.yaml
    status: str             # "match" | "mismatch" | "not_found"
    code_value: Any = None  # Value found in the repo (None if not_found)
    file_ref: str = ""      # "file:line" reference in the repo
    recommendation: str = ""


@dataclass
class ScanResult:
    """Aggregated scan results for one repo file."""

    path: Path
    kv_pairs: dict[str, Any] = field(default_factory=dict)  # normalised_key → value
    raw_pairs: dict[str, Any] = field(default_factory=dict)  # original_key → value
    line_map: dict[str, int] = field(default_factory=dict)   # original_key → line number


# ---------------------------------------------------------------------------
# Key normalisation
# ---------------------------------------------------------------------------

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _to_snake(name: str) -> str:
    """Convert camelCase or PascalCase to snake_case."""
    return _CAMEL_RE.sub("_", name).lower()


def _normalise_key(key: str) -> str:
    """Normalise a config key for fuzzy matching.

    Steps:
      1. Strip common prefixes (``env_``, ``model_``, ``train_``, ``cfg_``).
      2. Convert camelCase/PascalCase to snake_case.
      3. Replace hyphens and dots with underscores.
      4. Collapse repeated underscores.
      5. Strip leading/trailing underscores.
      6. Lower-case.
    """
    s = _to_snake(key)
    s = re.sub(r"[-\.]", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_").lower()
    for prefix in ("env_", "model_", "train_", "cfg_", "config_", "opt_", "hp_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def _path_leaf(dotted_path: str) -> str:
    """Return the last component of a dotted path."""
    return dotted_path.rsplit(".", 1)[-1]


def _path_tail_n(dotted_path: str, n: int = 2) -> str:
    """Return the last *n* components of a dotted path joined by underscore."""
    parts = dotted_path.split(".")
    return "_".join(parts[-n:]) if len(parts) >= n else dotted_path


# ---------------------------------------------------------------------------
# Spec flattening
# ---------------------------------------------------------------------------


def _flatten_spec(spec: dict) -> dict[str, Any]:
    """Flatten spec.yaml to ``{dotted_path: value}`` for diffable fields.

    Excludes bookkeeping sections, UNRESOLVED values, and source/meta keys.
    Lists that consist entirely of dicts (e.g. reward.terms) are descende into
    using numeric indices.
    """
    result: dict[str, Any] = {}

    def _walk(d: Any, path: str) -> None:
        if isinstance(d, dict):
            for k, v in d.items():
                if k in _SKIP_KEYS:
                    continue
                child_path = f"{path}.{k}" if path else k
                if isinstance(v, (dict, list)):
                    _walk(v, child_path)
                elif v != UNRESOLVED and v is not None:
                    result[child_path] = v
        elif isinstance(d, list):
            for i, item in enumerate(d):
                _walk(item, f"{path}[{i}]")

    for section, content in spec.items():
        if section in _SKIP_SECTIONS:
            continue
        _walk(content, section)

    return result


# ---------------------------------------------------------------------------
# Repo file scanning
# ---------------------------------------------------------------------------


def _glob_files(repo_root: Path, patterns: list[str]) -> list[Path]:
    """Collect files matching any of the glob patterns under *repo_root*.

    Skips files inside directories listed in ``_SKIP_DIRS``.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in repo_root.glob(pattern):
            if not path.is_file():
                continue
            if path.resolve() in seen:
                continue
            # Skip if any parent directory is in _SKIP_DIRS.
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            seen.add(path.resolve())
            found.append(path)
    return sorted(found)


def _parse_yaml_json(path: Path) -> ScanResult:
    """Parse a YAML or JSON config file into a ScanResult."""
    result = ScanResult(path=path)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            raw_text = fh.read()
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(raw_text)
        else:
            data = json.loads(raw_text)
    except Exception:
        return result

    if not isinstance(data, dict):
        return result

    def _walk(d: Any, prefix: str) -> None:
        if isinstance(d, dict):
            for k, v in d.items():
                full = f"{prefix}.{k}" if prefix else str(k)
                norm = _normalise_key(str(k))
                if not isinstance(v, (dict, list)):
                    result.raw_pairs[full] = v
                    result.kv_pairs[norm] = v
                    # Also index by just the leaf key.
                    result.kv_pairs[_normalise_key(str(k))] = v
                else:
                    _walk(v, full)
        elif isinstance(d, list):
            for i, item in enumerate(d):
                _walk(item, f"{prefix}[{i}]")

    _walk(data, "")
    return result


def _parse_python(path: Path) -> ScanResult:
    """Parse a Python file for config assignments and dict literals.

    Strategies:
      1. Simple assignments: ``NAME = VALUE`` at module level or inside functions.
      2. Dict literals: ``{ "key": value }`` patterns (module-level assignments).
      3. Function keyword arguments: ``func(key=value)`` calls.
      4. Dataclass/NamedTuple field defaults.

    Uses ``ast`` for accuracy; falls back to regex for files that fail to parse.
    """
    result = ScanResult(path=path)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            raw_text = fh.read()
    except OSError:
        return result

    # --- AST-based extraction ---
    try:
        tree = ast.parse(raw_text, filename=str(path))
        lines = raw_text.splitlines()
        _extract_from_ast(tree, result, lines)
    except SyntaxError:
        # Fall back to regex extraction for non-parsable files.
        _extract_from_regex(raw_text, result)

    return result


def _ast_literal(node: ast.expr) -> Any:
    """Safely evaluate a constant AST node to a Python value."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _ast_literal(node.operand)
        if isinstance(inner, (int, float)):
            return -inner
    if isinstance(node, ast.List):
        items = [_ast_literal(elt) for elt in node.elts]
        if all(v is not None for v in items):
            return items
    if isinstance(node, ast.Tuple):
        items = [_ast_literal(elt) for elt in node.elts]
        if all(v is not None for v in items):
            return tuple(items)
    return None


def _extract_from_ast(tree: ast.Module, result: ScanResult, lines: list[str]) -> None:
    """Walk the AST and extract config-like assignments."""

    def _process_assignment(key: str, value_node: ast.expr, lineno: int) -> None:
        val = _ast_literal(value_node)
        if val is None:
            return
        norm = _normalise_key(key)
        result.raw_pairs[key] = val
        result.kv_pairs[norm] = val
        result.line_map[key] = lineno

    def _process_dict(d: ast.Dict, prefix: str, lineno: int) -> None:
        for key_node, val_node in zip(d.keys, d.values):
            if key_node is None:
                continue
            key_lit = _ast_literal(key_node)
            if not isinstance(key_lit, str):
                continue
            full_key = f"{prefix}.{key_lit}" if prefix else key_lit
            if isinstance(val_node, ast.Dict):
                _process_dict(val_node, full_key, lineno)
            else:
                _process_assignment(full_key, val_node, lineno)

    for node in ast.walk(tree):
        # Module-level and function-level simple assignments: X = value
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    _process_assignment(target.id, node.value, node.lineno)
                elif isinstance(target, ast.Attribute):
                    _process_assignment(target.attr, node.value, node.lineno)
            # Also walk dict RHS directly assigned to a name.
            if isinstance(node.value, ast.Dict) and node.targets:
                if isinstance(node.targets[0], ast.Name):
                    _process_dict(node.value, "", node.lineno)

        # Annotated assignments: x: int = 5
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name):
                _process_assignment(node.target.id, node.value, node.lineno)

        # Keyword arguments in function calls: Config(lr=1e-4)
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg is not None:
                    _process_assignment(kw.arg, kw.value, node.lineno)


_ASSIGN_RE = re.compile(
    r"""(?:^|\b)                           # Start or word boundary
        (?:["\']?)                         # Optional quote (dict key)
        ([\w_]+)                           # Key name
        (?:["\']?)\s*                      # Optional closing quote + whitespace
        (?:=|:)\s*                         # Assignment or colon
        ([\-+]?\d+(?:\.\d+)?(?:[eE][\-+]?\d+)?)  # Numeric value
    """,
    re.VERBOSE | re.MULTILINE,
)


def _extract_from_regex(raw_text: str, result: ScanResult) -> None:
    """Fallback: extract key=numeric_value pairs via regex."""
    for lineno, line in enumerate(raw_text.splitlines(), start=1):
        for m in _ASSIGN_RE.finditer(line):
            key = m.group(1)
            raw_val = m.group(2)
            try:
                val: Any = int(raw_val) if "." not in raw_val and "e" not in raw_val.lower() else float(raw_val)
            except ValueError:
                continue
            norm = _normalise_key(key)
            result.raw_pairs[key] = val
            result.kv_pairs[norm] = val
            result.line_map[key] = lineno


# ---------------------------------------------------------------------------
# Value comparison
# ---------------------------------------------------------------------------


def _values_match(spec_val: Any, code_val: Any) -> bool:
    """Return True if spec_val and code_val are semantically equal.

    Handles:
      - Exact equality.
      - Numeric closeness (1% relative tolerance for floats).
      - String case-insensitive equality.
      - List/tuple element-wise comparison.
    """
    if spec_val == code_val:
        return True
    if isinstance(spec_val, (int, float)) and isinstance(code_val, (int, float)):
        try:
            spec_f, code_f = float(spec_val), float(code_val)
            if spec_f == 0.0 and code_f == 0.0:
                return True
            if spec_f == 0.0:
                return abs(code_f) < 1e-12
            return abs(spec_f - code_f) / abs(spec_f) < 0.01  # 1% tolerance
        except (ValueError, ZeroDivisionError):
            pass
    if isinstance(spec_val, str) and isinstance(code_val, str):
        return spec_val.strip().lower() == code_val.strip().lower()
    if isinstance(spec_val, (list, tuple)) and isinstance(code_val, (list, tuple)):
        if len(spec_val) != len(code_val):
            return False
        return all(_values_match(a, b) for a, b in zip(spec_val, code_val))
    return False


# ---------------------------------------------------------------------------
# Field matching engine
# ---------------------------------------------------------------------------


def _candidate_keys(spec_path: str) -> list[str]:
    """Generate a list of normalised candidate key strings to search for.

    Given ``training.optimizer.lr``, generates:
      - ``training_optimizer_lr``
      - ``optimizer_lr``
      - ``lr``
      - ``learning_rate``  (common expansions)
      - ``train_ratio`` etc.
    """
    parts = re.sub(r"\[\d+\]", "", spec_path).split(".")
    candidates: list[str] = []

    # All suffixes of the path joined by underscore.
    for i in range(len(parts)):
        suffix = "_".join(parts[i:])
        candidates.append(_normalise_key(suffix))

    # Leaf-only.
    leaf = _normalise_key(parts[-1])
    candidates.append(leaf)

    # Common expansions for terse keys.
    _EXPANSIONS: dict[str, list[str]] = {
        "lr": ["learning_rate", "lr"],
        "eps": ["epsilon", "eps", "adam_eps"],
        "clip_grad": ["clip_grad_norm", "grad_clip", "max_grad_norm"],
        "capacity_steps": ["buffer_size", "replay_size", "capacity"],
        "context_length": ["sequence_length", "context_len", "minlen", "chunk_len"],
        "train_ratio": ["update_to_data", "grad_steps_per_env_step", "train_ratio"],
        "imag_horizon": ["imagination_horizon", "horizon"],
        "discount": ["gamma", "discount", "reward_discount"],
        "lambda_gae": ["lmbda", "gae_lambda", "return_lambda", "td_lambda"],
        "batch_size": ["batch_size", "bs"],
        "batch_length": ["seq_len", "batch_length", "chunk_length"],
        "total_env_steps": ["steps", "total_steps", "max_steps", "num_steps"],
        "num_categoricals": ["stoch_size", "rssm_stoch"],
        "num_classes": ["category_size", "rssm_classes"],
        "imagination_horizon": ["imag_horizon", "horizon"],
    }
    if leaf in _EXPANSIONS:
        candidates.extend(_EXPANSIONS[leaf])

    # Remove duplicates while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _search_field(
    spec_path: str,
    spec_val: Any,
    scan_results: list[ScanResult],
) -> FieldResult:
    """Search all scan results for a spec field and return a FieldResult."""
    candidates = _candidate_keys(spec_path)

    # Prefer longer (more specific) candidates first.
    candidates_sorted = sorted(candidates, key=len, reverse=True)

    best_match: FieldResult | None = None
    best_mismatch: FieldResult | None = None

    for scan in scan_results:
        for cand in candidates_sorted:
            if cand not in scan.kv_pairs:
                continue
            code_val = scan.kv_pairs[cand]

            # Find a line reference if possible.
            line_ref = ""
            for raw_key, raw_val in scan.raw_pairs.items():
                raw_norm = _normalise_key(_path_leaf(raw_key))
                if raw_norm == cand and raw_val == code_val:
                    lineno = scan.line_map.get(raw_key, 0)
                    line_ref = f"{scan.path}:{lineno}" if lineno else str(scan.path)
                    break
            if not line_ref:
                line_ref = str(scan.path)

            if _values_match(spec_val, code_val):
                fr = FieldResult(
                    spec_path=spec_path,
                    spec_value=spec_val,
                    status="match",
                    code_value=code_val,
                    file_ref=line_ref,
                    recommendation="",
                )
                # Exact match on longer key → strong result; return immediately.
                return fr
            else:
                # Record the first mismatch we find.
                if best_mismatch is None:
                    rec = _recommendation(spec_path, spec_val, code_val, line_ref)
                    best_mismatch = FieldResult(
                        spec_path=spec_path,
                        spec_value=spec_val,
                        status="mismatch",
                        code_value=code_val,
                        file_ref=line_ref,
                        recommendation=rec,
                    )

    if best_match:
        return best_match
    if best_mismatch:
        return best_mismatch

    # Not found anywhere.
    return FieldResult(
        spec_path=spec_path,
        spec_value=spec_val,
        status="not_found",
        code_value=None,
        file_ref="",
        recommendation=_recommendation_not_found(spec_path, spec_val),
    )


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------

def _recommendation(spec_path: str, spec_val: Any, code_val: Any, file_ref: str) -> str:
    """Generate a human-readable recommendation for a mismatch."""
    leaf = _path_leaf(spec_path)
    lines: list[str] = [
        f"Update `{leaf}` in `{Path(file_ref).name}` from `{code_val}` to `{spec_val}` "
        f"to match the paper spec."
    ]
    # Numeric-specific advice.
    if isinstance(spec_val, (int, float)) and isinstance(code_val, (int, float)):
        ratio = spec_val / code_val if code_val != 0 else float("inf")
        if abs(ratio - 1.0) < 0.5:
            lines.append("Values are within 50% of each other — verify units are consistent.")
        elif ratio > 10 or ratio < 0.1:
            lines.append("Values differ by >10x — check for unit conversion (e.g. ms vs s, steps vs epochs).")
    return " ".join(lines)


def _recommendation_not_found(spec_path: str, spec_val: Any) -> str:
    """Generate a recommendation for a field not found in the repo."""
    leaf = _path_leaf(spec_path)
    return (
        f"No config key matching `{leaf}` found in scanned files. "
        f"Add `{leaf} = {spec_val!r}` to the appropriate config file, "
        f"or verify the key uses a different naming convention."
    )


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

_STATUS_ICON: dict[str, str] = {
    "match": "MATCH",
    "mismatch": "MISMATCH",
    "not_found": "NOT FOUND",
}

_STATUS_BADGE: dict[str, str] = {
    "match": ":white_check_mark:",
    "mismatch": ":x:",
    "not_found": ":grey_question:",
}


def _format_value(val: Any) -> str:
    """Format a value for Markdown table display."""
    if val is None:
        return "—"
    if isinstance(val, float):
        # Use scientific notation for very small/large values.
        if val != 0 and (abs(val) < 1e-3 or abs(val) > 1e6):
            return f"`{val:.3e}`"
        return f"`{val}`"
    if isinstance(val, (list, tuple)):
        return f"`{list(val)}`"
    return f"`{val!r}`"


def _md_table_row(fr: FieldResult) -> str:
    spec_v = _format_value(fr.spec_value)
    code_v = _format_value(fr.code_value)
    status = _STATUS_ICON[fr.status]
    file_ref = f"`{Path(fr.file_ref).name}`" if fr.file_ref else "—"
    # Truncate long paths.
    path_display = fr.spec_path
    if len(path_display) > 50:
        path_display = "…" + path_display[-47:]
    return f"| `{path_display}` | {spec_v} | {code_v} | {file_ref} | {status} |"


def _generate_report(
    spec: dict,
    results: list[FieldResult],
    repo_root: Path,
    spec_path: Path,
    scanned_files: list[Path],
) -> str:
    """Generate the full Markdown diff report."""
    n_match = sum(1 for r in results if r.status == "match")
    n_mismatch = sum(1 for r in results if r.status == "mismatch")
    n_not_found = sum(1 for r in results if r.status == "not_found")
    n_total = len(results)

    meta = spec.get("meta", {})
    paper = meta.get("paper", {})
    paper_title = paper.get("title", "Unknown paper")
    arxiv_id = paper.get("arxiv", "")
    arxiv_link = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""

    lines: list[str] = []

    # --- Header ---
    lines.append("# Spec vs. Repo Diff Report")
    lines.append("")
    lines.append(f"**Generated by:** `diff_spec.py` v{TOOL_VERSION}")
    lines.append(f"**Spec file:** `{spec_path}`")
    lines.append(f"**Repo root:** `{repo_root}`")
    if paper_title and paper_title != UNRESOLVED:
        paper_ref = f"[{paper_title}]({arxiv_link})" if arxiv_link else paper_title
        lines.append(f"**Paper:** {paper_ref}")
    lines.append(f"**Scanned files:** {len(scanned_files)}")
    lines.append("")

    # --- Summary ---
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Status | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Match | {n_match} |")
    lines.append(f"| Mismatch | {n_mismatch} |")
    lines.append(f"| Not Found | {n_not_found} |")
    lines.append(f"| **Total** | **{n_total}** |")
    lines.append("")

    if n_match == n_total:
        lines.append("> All spec fields match the repository. No action required.")
        lines.append("")
    elif n_mismatch == 0 and n_not_found > 0:
        lines.append(
            f"> No mismatches found, but {n_not_found} spec field(s) are not "
            "present in scanned config files."
        )
        lines.append("")

    # --- Per-section tables ---
    lines.append("## Per-Field Comparison")
    lines.append("")

    # Group by top-level section.
    from collections import defaultdict
    by_section: dict[str, list[FieldResult]] = defaultdict(list)
    for fr in results:
        section = fr.spec_path.split(".")[0]
        by_section[section].append(fr)

    for section in sorted(by_section.keys()):
        section_results = by_section[section]
        sec_match = sum(1 for r in section_results if r.status == "match")
        sec_total = len(section_results)

        lines.append(f"### `{section}` ({sec_match}/{sec_total} matched)")
        lines.append("")
        lines.append("| Spec Path | Spec Value | Code Value | File | Status |")
        lines.append("|-----------|-----------|------------|------|--------|")
        for fr in sorted(section_results, key=lambda r: (r.status, r.spec_path)):
            lines.append(_md_table_row(fr))
        lines.append("")

    # --- Mismatch details ---
    mismatches = [r for r in results if r.status == "mismatch"]
    if mismatches:
        lines.append("## Mismatch Details")
        lines.append("")
        lines.append(
            "The following fields were found in the repository but have different values "
            "than the paper spec."
        )
        lines.append("")
        for fr in sorted(mismatches, key=lambda r: r.spec_path):
            lines.append(f"### `{fr.spec_path}`")
            lines.append("")
            lines.append(f"- **Spec value:** `{fr.spec_value}`")
            lines.append(f"- **Code value:** `{fr.code_value}`")
            lines.append(f"- **Location:** `{fr.file_ref}`")
            lines.append(f"- **Recommendation:** {fr.recommendation}")
            lines.append("")

    # --- Not-found details ---
    not_found = [r for r in results if r.status == "not_found"]
    if not_found:
        lines.append("## Fields Not Found in Repository")
        lines.append("")
        lines.append(
            "These spec fields could not be matched to any key in the scanned config files. "
            "They may use a different naming convention, be hard-coded in non-config code, "
            "or be missing entirely."
        )
        lines.append("")
        lines.append("| Spec Path | Spec Value | Recommendation |")
        lines.append("|-----------|-----------|----------------|")
        for fr in sorted(not_found, key=lambda r: r.spec_path):
            spec_v = _format_value(fr.spec_value)
            rec = fr.recommendation.replace("|", r"\|")
            path_display = fr.spec_path
            if len(path_display) > 50:
                path_display = "…" + path_display[-47:]
            lines.append(f"| `{path_display}` | {spec_v} | {rec} |")
        lines.append("")

    # --- Scanned files ---
    lines.append("## Scanned Files")
    lines.append("")
    lines.append(f"The following {len(scanned_files)} file(s) were scanned:")
    lines.append("")
    for f in sorted(scanned_files):
        rel = _try_relative(f, repo_root)
        lines.append(f"- `{rel}`")
    lines.append("")

    # --- Footer ---
    lines.append("---")
    lines.append(f"*Generated by `diff_spec.py` v{TOOL_VERSION}*")
    lines.append("")

    return "\n".join(lines)


def _try_relative(path: Path, base: Path) -> str:
    """Return path relative to base, or absolute string if not possible."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compare spec.yaml against existing repo code/config to find mismatches.\n"
            "Produces a Markdown report with per-field match status and recommendations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
              # Scan a repo with default settings:
              python diff_spec.py --spec spec.yaml --repo-root /path/to/repo

              # Custom output path:
              python diff_spec.py --spec spec.yaml --repo-root /path/to/repo \\
                  --output reports/spec_diff.md

              # Restrict to specific file types:
              python diff_spec.py --spec spec.yaml --repo-root /path/to/repo \\
                  --config-patterns "**/*.yaml,configs/**/*.json"
        """),
    )
    p.add_argument(
        "--spec",
        required=True,
        metavar="FILE",
        help="Path to spec.yaml (output of emit_yaml.py).",
    )
    p.add_argument(
        "--repo-root",
        required=True,
        metavar="DIR",
        help="Path to the repository root to scan.",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Path for the output Markdown report (default: reports/spec_diff.md).",
    )
    p.add_argument(
        "--config-patterns",
        default="**/*.yaml,**/*.yml,**/*.json,**/*.py",
        metavar="PATTERNS",
        help=(
            "Comma-separated glob patterns to find config files. "
            'Default: "**/*.yaml,**/*.yml,**/*.json,**/*.py"'
        ),
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=2000,
        metavar="N",
        help="Maximum number of files to scan (default: 2000, safety limit).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Resolve paths ---
    spec_path = Path(args.spec).resolve()
    repo_root = Path(args.repo_root).resolve()

    if not spec_path.exists():
        print(f"ERROR: spec file not found: {spec_path}", file=sys.stderr)
        return 1
    if not repo_root.is_dir():
        print(f"ERROR: repo-root is not a directory: {repo_root}", file=sys.stderr)
        return 1

    # --- Output path ---
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = repo_root / "reports" / "spec_diff.md"

    # --- Load spec ---
    print(f"Loading spec: {spec_path}", file=sys.stderr)
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh)
    except Exception as exc:
        print(f"ERROR: failed to load spec: {exc}", file=sys.stderr)
        return 1

    if not isinstance(spec, dict):
        print("ERROR: spec is not a YAML mapping.", file=sys.stderr)
        return 1

    # --- Flatten spec to diffable fields ---
    flat_spec = _flatten_spec(spec)
    print(f"Spec fields to diff: {len(flat_spec)}", file=sys.stderr)

    # --- Scan repo files ---
    patterns = [p.strip() for p in args.config_patterns.split(",") if p.strip()]
    print(f"Scanning repo: {repo_root}", file=sys.stderr)
    print(f"Patterns: {patterns}", file=sys.stderr)

    all_files = _glob_files(repo_root, patterns)
    if len(all_files) > args.max_files:
        print(
            f"WARNING: found {len(all_files)} files; limiting to {args.max_files} "
            "(increase --max-files to scan more).",
            file=sys.stderr,
        )
        all_files = all_files[: args.max_files]

    print(f"Files to scan: {len(all_files)}", file=sys.stderr)

    scan_results: list[ScanResult] = []
    yaml_json_exts = {".yaml", ".yml", ".json"}
    py_exts = {".py"}

    for i, file_path in enumerate(all_files, start=1):
        if i % 200 == 0:
            print(f"  Scanning file {i}/{len(all_files)}...", file=sys.stderr)
        ext = file_path.suffix.lower()
        if ext in yaml_json_exts:
            scan_results.append(_parse_yaml_json(file_path))
        elif ext in py_exts:
            scan_results.append(_parse_python(file_path))

    total_kv = sum(len(s.kv_pairs) for s in scan_results)
    print(f"Scan complete. Total key-value pairs extracted: {total_kv}", file=sys.stderr)

    # --- Match spec fields ---
    print("Matching spec fields against repo...", file=sys.stderr)
    results: list[FieldResult] = []
    for spec_path_str, spec_val in flat_spec.items():
        fr = _search_field(spec_path_str, spec_val, scan_results)
        results.append(fr)

    n_match = sum(1 for r in results if r.status == "match")
    n_mismatch = sum(1 for r in results if r.status == "mismatch")
    n_not_found = sum(1 for r in results if r.status == "not_found")
    print(
        f"Results: {n_match} match, {n_mismatch} mismatch, {n_not_found} not found",
        file=sys.stderr,
    )

    # --- Generate report ---
    report = _generate_report(spec, results, repo_root, spec_path, all_files)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"Report written to: {output_path}", file=sys.stderr)

    # Return non-zero if there are mismatches or unresolved fields.
    if n_mismatch > 0 or n_not_found > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
