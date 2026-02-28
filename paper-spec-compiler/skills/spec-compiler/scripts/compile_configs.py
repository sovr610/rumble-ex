#!/usr/bin/env python3
"""
compile_configs.py — Map spec.yaml to framework config overrides.

Converts a compiled spec.yaml (produced by emit_yaml.py) into:
  1. A framework config override file (DreamerV3-style or generic flat key=value).
  2. Optionally, a fully-resolved spec with all baseline imports merged in
     (spec.resolved.yaml), showing per-field provenance.
  3. Optionally, a patch file showing only deltas from the baselines
     (spec.patch.yaml).
  4. Optionally, a conflicts file listing fields where baselines disagree and the
     top-level spec does not resolve the conflict (spec.conflicts.yaml).

Import resolution rules (baseline-linking.md):
  1. Explicit value in the top-level spec wins.
  2. Baseline fills gaps for fields not mentioned in the top-level spec.
  3. If two baselines disagree and the top-level spec does not override, flag a
     conflict.
  4. Inheritance order: first import listed in `imports` has lowest priority;
     last import listed has next-lowest; top-level spec has highest priority.
     (Later entries in the imports list override earlier ones for gap-filling.)

DreamerV3 key mapping:
  spec path                              → dreamerv3 key
  ------------------------------------------------------------------
  training.replay.capacity_steps        → replay.capacity
  training.replay.context_length        → replay.minlen  (context)
  training.replay.sampling              → replay.prioritize_ends
  training.batch.size                   → batch_size
  training.batch.length                 → batch_length
  training.train_ratio                  → train_ratio
  training.optimizer.lr                 → model_opt.lr
  training.optimizer.eps                → model_opt.eps
  training.optimizer.clip_grad          → model_opt.clip
  training.optimizer.type               → model_opt.opt
  training.schedule.total_env_steps     → steps
  actor_critic.imagination_horizon      → imag_horizon
  actor_critic.discount                 → discount
  actor_critic.lambda_gae               → return_lambda
  actor_critic.policy_distribution      → actor_dist
  actor_critic.deterministic_eval       → eval_noise
  world_model.discrete_latent.num_categoricals  → rssm_stoch
  world_model.discrete_latent.num_classes       → rssm_classes
  world_model.components.sequence_model.hidden_size → rssm_deter
  world_model.symlog                    → encoder.symlog_inputs
  world_model.normalization             → norm
  reward.discount                       → discount
  timing.control_frequency_hz           → env_hz

Usage
-----
  # Generic flat config (default):
  python compile_configs.py --spec spec.yaml --output config_overrides.yaml

  # DreamerV3 config:
  python compile_configs.py --spec spec.yaml --format dreamerv3 \\
      --output dreamer_overrides.yaml

  # With full import resolution:
  python compile_configs.py --spec spec.yaml --resolve-imports \\
      --output-resolved spec.resolved.yaml \\
      --output-patch spec.patch.yaml \\
      --output-conflicts spec.conflicts.yaml

Dependencies
------------
  stdlib only + PyYAML (pip install pyyaml)
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from copy import deepcopy
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


# ---------------------------------------------------------------------------
# Custom YAML dumper (deterministic, no aliases)
# ---------------------------------------------------------------------------


class _NoAliasDumper(yaml.Dumper):
    """YAML dumper that never emits anchors/aliases and sorts keys."""

    def ignore_aliases(self, data: Any) -> bool:  # noqa: ARG002
        return True


_NoAliasDumper.add_representer(
    type(None),
    lambda d, _: d.represent_scalar("tag:yaml.org,2002:null", "null"),
)


def _dump_yaml(data: Any, *, sort_keys: bool = False) -> str:
    return yaml.dump(
        data,
        Dumper=_NoAliasDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=sort_keys,
        width=120,
    )


# ---------------------------------------------------------------------------
# Deep path utilities
# ---------------------------------------------------------------------------


def _deep_get(d: Any, path: str, default: Any = None) -> Any:
    """Retrieve a value from a nested dict/list by a dotted path string.

    Supports list index notation: ``training.schedule.phases[0].learning_rate``.
    Returns *default* if any step along the path is missing or the object is not
    a dict/list at that level.
    """
    parts = _split_path(path)
    node = d
    for part in parts:
        if isinstance(part, int):
            if not isinstance(node, list) or part >= len(node):
                return default
            node = node[part]
        elif isinstance(node, dict):
            if part not in node:
                return default
            node = node[part]
        else:
            return default
    return node


def _split_path(path: str) -> list[str | int]:
    """Split a dotted path string into a list of str/int keys.

    e.g. ``"training.schedule.phases[0].lr"`` → ``["training","schedule","phases",0,"lr"]``
    """
    import re
    parts: list[str | int] = []
    for token in re.split(r"\.", path):
        m = re.match(r"^(\w+)\[(\d+)\]$", token)
        if m:
            parts.append(m.group(1))
            parts.append(int(m.group(2)))
        elif re.match(r"^\d+$", token):
            parts.append(int(token))
        else:
            parts.append(token)
    return parts


def _deep_set(d: dict, path: str, value: Any) -> None:
    """Set a value in a nested dict by a dotted path, creating intermediate dicts."""
    parts = _split_path(path)
    node = d
    for part in parts[:-1]:
        if isinstance(part, int):
            # Can't auto-create list entries; skip
            return
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    last = parts[-1]
    if not isinstance(last, int):
        node[last] = value


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict into dotted-path → value pairs.

    Lists are not descended (they are treated as atomic values).
    """
    result: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(_flatten(v, full_key))
            else:
                result[full_key] = v
    else:
        result[prefix] = d
    return result


# ---------------------------------------------------------------------------
# DreamerV3 key mapping
# ---------------------------------------------------------------------------

# Maps spec dotted paths → DreamerV3 nested config paths.
# The value is a tuple (dreamer_path, transform_fn | None).
# transform_fn, if given, converts the spec value to the dreamer value.

def _bool_invert(v: Any) -> bool:
    """Invert a boolean (used for deterministic_eval → eval_noise)."""
    if isinstance(v, bool):
        return not v
    return v


def _sampling_to_prioritize(v: Any) -> bool:
    """Convert sampling string to dreamer's boolean prioritize_ends."""
    if isinstance(v, str):
        return v.lower() == "prioritized"
    return False


_DREAMERV3_MAP: list[tuple[str, str, Any]] = [
    # (spec_path, dreamer_path, transform_fn_or_None)
    # --- replay ---
    ("training.replay.capacity_steps",          "replay.capacity",           None),
    ("training.replay.context_length",          "replay.minlen",             None),
    ("training.replay.sampling",                "replay.prioritize_ends",    _sampling_to_prioritize),
    # --- batch ---
    ("training.batch.size",                     "batch_size",                None),
    ("training.batch.length",                   "batch_length",              None),
    # --- training schedule ---
    ("training.train_ratio",                    "train_ratio",               None),
    ("training.schedule.total_env_steps",       "steps",                     None),
    # --- optimizer ---
    ("training.optimizer.type",                 "model_opt.opt",             None),
    ("training.optimizer.lr",                   "model_opt.lr",              None),
    ("training.optimizer.eps",                  "model_opt.eps",             None),
    ("training.optimizer.clip_grad",            "model_opt.clip",            None),
    # --- actor-critic ---
    ("actor_critic.imagination_horizon",        "imag_horizon",              None),
    ("actor_critic.discount",                   "discount",                  None),
    ("actor_critic.lambda_gae",                 "return_lambda",             None),
    ("actor_critic.policy_distribution",        "actor_dist",                None),
    ("actor_critic.deterministic_eval",         "eval_noise",                _bool_invert),
    ("actor_critic.actor.hidden_layers",        "actor.layers",              None),
    ("actor_critic.actor.activation",           "actor.act",                 None),
    ("actor_critic.critic.hidden_layers",       "critic.layers",             None),
    ("actor_critic.critic.activation",          "critic.act",                None),
    # --- world model ---
    ("world_model.discrete_latent.num_categoricals", "rssm_stoch",           None),
    ("world_model.discrete_latent.num_classes",      "rssm_classes",         None),
    ("world_model.components.sequence_model.hidden_size", "rssm_deter",      None),
    ("world_model.symlog",                      "encoder.symlog_inputs",     None),
    ("world_model.normalization",               "norm",                      None),
    ("world_model.architecture",                "rssm_type",                 None),
    # --- reward ---
    ("reward.discount",                         "discount",                  None),  # same as ac discount; last write wins
    ("reward.normalization",                    "reward_norm",               None),
    # --- timing ---
    ("timing.control_frequency_hz",             "env_hz",                    None),
    ("timing.sim_dt",                           "sim_dt",                    None),
    # --- training misc ---
    ("training.use_amp",                        "jit",                       None),
]


def _spec_to_dreamerv3(spec: dict) -> dict:
    """Convert a spec dict to a nested DreamerV3 config override dict."""
    dreamer: dict = {}
    for spec_path, dreamer_path, transform in _DREAMERV3_MAP:
        val = _deep_get(spec, spec_path)
        if val is None or val == UNRESOLVED:
            continue
        if transform is not None:
            val = transform(val)
        _deep_set(dreamer, dreamer_path, val)
    return dreamer


# ---------------------------------------------------------------------------
# Generic flat config
# ---------------------------------------------------------------------------


def _spec_to_generic(spec: dict) -> dict[str, Any]:
    """Convert a spec dict to a flat key=value config dict.

    Skips UNRESOLVED values, meta section, and source/imports bookkeeping.
    """
    skip_top = {"meta", "imports"}
    skip_keys = {"source", "sources"}

    def _walk(d: Any, prefix: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if k in skip_keys:
                    continue
                full = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    result.update(_walk(v, full))
                elif isinstance(v, list):
                    # Emit lists as-is (skip if contains dicts — too verbose)
                    if all(not isinstance(item, dict) for item in v):
                        if v != [UNRESOLVED] and v != UNRESOLVED:
                            result[full] = v
                else:
                    if v != UNRESOLVED and v is not None:
                        result[full] = v
        return result

    flat: dict[str, Any] = {}
    for section, content in spec.items():
        if section in skip_top:
            continue
        flat.update(_walk(content, section))
    return flat


# ---------------------------------------------------------------------------
# Baseline import resolution
# ---------------------------------------------------------------------------


def _load_spec(path: Path) -> dict | None:
    """Load a YAML spec file and return its content as a dict, or None on error."""
    if not path.exists():
        print(f"WARNING: spec file not found: {path}", file=sys.stderr)
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception as exc:
        print(f"WARNING: failed to load spec {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"WARNING: spec at {path} is not a YAML mapping", file=sys.stderr)
        return None
    return data


def _find_baseline_spec(spec_url: str | None, spec_path: Path) -> Path | None:
    """Resolve a baseline spec_url to an absolute path.

    Tries:
      1. Absolute path.
      2. Relative to the directory that contains the top-level spec.yaml.
      3. Relative to CWD.
    """
    if not spec_url:
        return None
    candidate = Path(spec_url)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    # Relative to the top-level spec's directory.
    sibling = spec_path.parent / spec_url
    if sibling.exists():
        return sibling
    # Relative to CWD.
    cwd_rel = Path.cwd() / spec_url
    if cwd_rel.exists():
        return cwd_rel
    return None


def _annotate_provenance(
    value: Any,
    source_label: str,
) -> Any:
    """Wrap a scalar value in a provenance-annotated dict.

    For scalars: ``{"_value": v, "_source": label}``.
    For dicts: add ``"__source__": label`` key (non-destructive if already there).
    For lists: return as-is (provenance tracked at parent level).
    """
    if isinstance(value, dict):
        annotated = dict(value)
        annotated.setdefault("__source__", source_label)
        return annotated
    # Scalars and lists: wrap in a sentinel dict only if not already wrapped.
    return {"_value": value, "_source": source_label}


def _merge_specs(
    baseline: dict,
    override: dict,
    override_label: str,
    baseline_label: str,
    conflicts: list[dict],
    existing_provenance: dict[str, str],
) -> tuple[dict, dict[str, str]]:
    """Merge *override* on top of *baseline*.

    Returns:
      - merged dict (values from override take precedence; baseline fills gaps)
      - updated provenance dict mapping dotted-path → source label

    Conflict detection: If both baseline and a prior value from a *different*
    baseline exist and the override does not set this field, flag a conflict.
    """
    merged = deepcopy(baseline)
    provenance = dict(existing_provenance)

    def _merge_recursive(base: dict, over: dict, path: str) -> dict:
        result = dict(base)
        for k, v in over.items():
            full_path = f"{path}.{k}" if path else k
            if k not in result or result[k] is None:
                # Gap-fill from override.
                result[k] = deepcopy(v)
                provenance[full_path] = override_label
            elif v == UNRESOLVED or v is None:
                # Override has UNRESOLVED — keep baseline.
                pass
            elif isinstance(v, dict) and isinstance(result[k], dict):
                result[k] = _merge_recursive(result[k], v, full_path)
            elif result[k] != v:
                # Potential conflict: baseline has a value, override changes it.
                existing_source = provenance.get(full_path, baseline_label)
                if existing_source != override_label:
                    # The override is explicitly changing a value set by a different source.
                    # In normal operation (top-level spec calling this) this is fine —
                    # it means the top-level spec intentionally overrides the baseline.
                    # Record as conflict only when BOTH are baselines (not top-level).
                    conflicts.append({
                        "path": full_path,
                        f"{existing_source}_value": result[k],
                        f"{override_label}_value": v,
                        "resolution": "baseline_conflict",
                    })
                result[k] = deepcopy(v)
                provenance[full_path] = override_label
            else:
                # Same value — no-op, keep existing provenance.
                pass
        return result

    merged = _merge_recursive(merged, override, "")
    return merged, provenance


def _resolve_imports(
    spec: dict,
    spec_path: Path,
) -> tuple[dict, dict[str, str], list[dict], dict]:
    """Resolve all imports listed in spec.imports.

    Returns:
      - resolved: fully merged spec dict
      - provenance: mapping dotted-path → source label ("spec_name@version" or "top_level")
      - conflicts: list of conflict dicts
      - patch: dict of fields where the top-level spec differs from the resolved baseline
    """
    imports_list: list[dict] = spec.get("imports", [])
    conflicts: list[dict] = []
    provenance: dict[str, str] = {}

    # Start with an empty merged baseline; accumulate baselines in import order
    # (first import = lowest priority → last import = higher priority → top-level = highest).
    merged_baseline: dict = {}

    for imp in imports_list:
        name: str = imp.get("name", "unknown")
        version: str = imp.get("version", "UNRESOLVED")
        spec_url: str | None = imp.get("spec_url")
        label = f"{name}@{version}"

        baseline_path = _find_baseline_spec(spec_url, spec_path)
        if baseline_path is None:
            print(
                f"  WARNING: cannot locate baseline spec '{name}' (spec_url={spec_url!r}). "
                "Skipping.",
                file=sys.stderr,
            )
            continue

        baseline_spec = _load_spec(baseline_path)
        if baseline_spec is None:
            continue

        print(f"  Loaded baseline: {label} from {baseline_path}", file=sys.stderr)

        merged_baseline, provenance = _merge_specs(
            merged_baseline,
            baseline_spec,
            override_label=label,
            baseline_label="(empty)",
            conflicts=conflicts,
            existing_provenance=provenance,
        )

    # Clear conflicts accumulated between baselines — those are baseline-vs-baseline
    # conflicts that the top-level spec will (or should) resolve.
    baseline_conflicts = list(conflicts)
    conflicts.clear()

    # Now merge the top-level spec on top of the combined baselines.
    top_label = "top_level"
    resolved, provenance = _merge_specs(
        merged_baseline,
        spec,
        override_label=top_label,
        baseline_label="baseline",
        conflicts=conflicts,
        existing_provenance=provenance,
    )

    # Carry forward any unresolved baseline conflicts where the top-level spec
    # did NOT provide a resolution.
    resolved_paths = {c["path"] for c in conflicts}
    for bc in baseline_conflicts:
        path = bc["path"]
        if path not in resolved_paths:
            top_val = _deep_get(spec, path)
            if top_val is None or top_val == UNRESOLVED:
                bc["top_level_value"] = UNRESOLVED
                bc["resolution"] = "UNRESOLVED"
                conflicts.append(bc)
            # else: top-level resolved it — no conflict to report.

    # Compute the patch: fields where top-level spec differs from merged baseline.
    patch: dict = {}
    flat_spec = _flatten({k: v for k, v in spec.items() if k not in {"meta", "imports", "source"}})
    flat_baseline = _flatten({k: v for k, v in merged_baseline.items() if k not in {"meta", "imports", "source"}})

    for path, spec_val in flat_spec.items():
        if spec_val == UNRESOLVED or spec_val is None:
            continue
        baseline_val = flat_baseline.get(path)
        if baseline_val != spec_val:
            _deep_set(patch, path, {"spec_value": spec_val, "baseline_value": baseline_val})

    return resolved, provenance, conflicts, patch


def _annotate_resolved_with_provenance(
    resolved: dict,
    provenance: dict[str, str],
) -> dict:
    """Insert inline ``# source:`` comments as a special ``__provenance__`` key.

    YAML doesn't support inline comments programmatically, so we instead inject a
    sibling ``__provenance__`` key alongside each value that has a known source.
    Downstream tools can strip these when loading.

    Rather than modifying the dict in place we build a parallel structure where
    every scalar value is replaced by ``{_value: v, _source: label}``.
    """
    flat_prov = provenance  # dotted-path → label

    def _annotate_recursive(d: Any, path: str) -> Any:
        if isinstance(d, dict):
            result: dict = {}
            for k, v in d.items():
                if k.startswith("__"):
                    result[k] = v
                    continue
                full = f"{path}.{k}" if path else k
                annotated_v = _annotate_recursive(v, full)
                result[k] = annotated_v
                if full in flat_prov and not isinstance(v, dict):
                    result[f"__{k}_source__"] = flat_prov[full]
            return result
        elif isinstance(d, list):
            return [_annotate_recursive(item, f"{path}[]") for item in d]
        else:
            return d

    return _annotate_recursive(resolved, "")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_dreamerv3_config(spec: dict, output_path: Path) -> None:
    """Write DreamerV3-style nested config override YAML."""
    dreamer = _spec_to_dreamerv3(spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = textwrap.dedent("""\
        # DreamerV3 config overrides — generated by compile_configs.py
        # DO NOT EDIT manually; re-run compile_configs.py to regenerate.
        # Apply with: dreamerv3 --configs defaults --config this_file.yaml
        #
    """)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(_dump_yaml(dreamer, sort_keys=False))
    print(f"DreamerV3 config overrides written to: {output_path}", file=sys.stderr)


def _write_generic_config(spec: dict, output_path: Path) -> None:
    """Write a flat key=value YAML config.

    Each entry is emitted as ``dotted.key: value`` on its own line.  Values are
    serialised using PyYAML's inline (flow) style to keep the file compact while
    remaining valid YAML.  Strings that need quoting are quoted automatically.
    """
    flat = _spec_to_generic(spec)
    # Sort for reproducibility.
    flat_sorted = dict(sorted(flat.items()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = textwrap.dedent("""\
        # Generic flat config overrides — generated by compile_configs.py
        # DO NOT EDIT manually; re-run compile_configs.py to regenerate.
        #
    """)

    def _inline(v: Any) -> str:
        """Serialise a scalar/list/bool to a compact YAML value string."""
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            # Quote strings that would be misinterpreted by YAML parsers.
            needs_quote = any(c in v for c in ":#{}[]|>&*!,?") or v in (
                "true", "false", "null", "yes", "no", "on", "off",
            ) or v == ""
            if needs_quote:
                return yaml.dump(v, default_flow_style=True, default_style='"').strip()
            return v
        if isinstance(v, float):
            # Represent floats without trailing document markers.
            r = repr(v)
            return r
        if isinstance(v, list):
            # Compact flow-style list.
            items = ", ".join(_inline(item) for item in v)
            return f"[{items}]"
        return str(v)

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        for k, v in flat_sorted.items():
            fh.write(f"{k}: {_inline(v)}\n")
    print(f"Generic flat config written to: {output_path}", file=sys.stderr)


def _write_resolved_spec(
    resolved: dict,
    provenance: dict[str, str],
    output_path: Path,
) -> None:
    """Write spec.resolved.yaml with provenance annotations."""
    annotated = _annotate_resolved_with_provenance(resolved, provenance)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = textwrap.dedent("""\
        # spec.resolved.yaml — fully merged spec with baseline imports resolved.
        # Generated by compile_configs.py --resolve-imports
        #
        # Fields annotated with __<key>_source__ show which spec provided that value.
        # Source labels:
        #   "top_level"         — set by the main spec.yaml
        #   "<name>@<version>"  — inherited from a baseline import
        #
    """)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(_dump_yaml(annotated, sort_keys=False))
    print(f"spec.resolved.yaml written to: {output_path}", file=sys.stderr)


def _write_patch(patch: dict, output_path: Path) -> None:
    """Write spec.patch.yaml — only fields that differ from baselines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = textwrap.dedent("""\
        # spec.patch.yaml — only the deltas from baseline imports.
        # Generated by compile_configs.py --resolve-imports --output-patch
        #
        # Each entry shows: spec_value (what this spec sets) vs baseline_value (what the baseline had).
        # Fields absent from baselines are shown with baseline_value: null.
        #
    """)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(_dump_yaml(patch, sort_keys=True))
    print(f"spec.patch.yaml written to: {output_path}", file=sys.stderr)


def _write_conflicts(conflicts: list[dict], output_path: Path) -> None:
    """Write spec.conflicts.yaml."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = textwrap.dedent("""\
        # spec.conflicts.yaml — fields where baseline imports disagree.
        # Generated by compile_configs.py --resolve-imports --output-conflicts
        #
        # resolution: "UNRESOLVED" means the top-level spec does not override this field —
        # a human must decide which baseline value to use.
        #
    """)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        if conflicts:
            fh.write(_dump_yaml({"conflicts": conflicts}, sort_keys=False))
        else:
            fh.write("conflicts: []\n")
    count = len(conflicts)
    print(
        f"spec.conflicts.yaml written to: {output_path} ({count} conflict(s))",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Map spec.yaml to framework config overrides (DreamerV3-style or generic).\n"
            "Optionally resolve baseline imports to produce resolved/patch/conflict files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
              # Generic flat config (default format):
              python compile_configs.py --spec spec.yaml --output config_overrides.yaml

              # DreamerV3 config:
              python compile_configs.py --spec spec.yaml --format dreamerv3 \\
                  --output dreamer_overrides.yaml

              # Full import resolution:
              python compile_configs.py --spec spec.yaml --resolve-imports \\
                  --output-resolved spec.resolved.yaml \\
                  --output-patch spec.patch.yaml \\
                  --output-conflicts spec.conflicts.yaml

              # All outputs at once:
              python compile_configs.py --spec spec.yaml --format dreamerv3 \\
                  --output dreamer_overrides.yaml --resolve-imports \\
                  --output-resolved spec.resolved.yaml \\
                  --output-patch spec.patch.yaml \\
                  --output-conflicts spec.conflicts.yaml
        """),
    )
    p.add_argument(
        "--spec",
        required=True,
        metavar="FILE",
        help="Path to spec.yaml (output of emit_yaml.py).",
    )
    p.add_argument(
        "--format",
        choices=["dreamerv3", "generic"],
        default="generic",
        metavar="FORMAT",
        help='Target config format: "dreamerv3" or "generic" (default: generic).',
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        help=(
            "Path for the output config override YAML. "
            "Defaults to dreamer_overrides.yaml or config_overrides.yaml depending on --format."
        ),
    )
    p.add_argument(
        "--resolve-imports",
        action="store_true",
        default=False,
        help=(
            "Resolve baseline imports listed in spec.imports. "
            "Produces resolved/patch/conflict outputs."
        ),
    )
    p.add_argument(
        "--output-resolved",
        metavar="FILE",
        default=None,
        help="Path for spec.resolved.yaml (requires --resolve-imports).",
    )
    p.add_argument(
        "--output-patch",
        metavar="FILE",
        default=None,
        help="Path for spec.patch.yaml — only deltas from baselines (requires --resolve-imports).",
    )
    p.add_argument(
        "--output-conflicts",
        metavar="FILE",
        default=None,
        help="Path for spec.conflicts.yaml (requires --resolve-imports).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Validate arg combinations ---
    if args.resolve_imports is False and any(
        [args.output_resolved, args.output_patch, args.output_conflicts]
    ):
        print(
            "WARNING: --output-resolved / --output-patch / --output-conflicts "
            "have no effect without --resolve-imports.",
            file=sys.stderr,
        )

    # --- Load spec ---
    spec_path = Path(args.spec).resolve()
    if not spec_path.exists():
        print(f"ERROR: spec file not found: {spec_path}", file=sys.stderr)
        return 1

    print(f"Loading spec: {spec_path}", file=sys.stderr)
    spec = _load_spec(spec_path)
    if spec is None:
        return 1

    # --- Determine output path ---
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        default_name = "dreamer_overrides.yaml" if args.format == "dreamerv3" else "config_overrides.yaml"
        output_path = spec_path.parent / default_name

    # --- Write main config override ---
    print(f"Format: {args.format}", file=sys.stderr)
    if args.format == "dreamerv3":
        _write_dreamerv3_config(spec, output_path)
    else:
        _write_generic_config(spec, output_path)

    # --- Optionally resolve imports ---
    if args.resolve_imports:
        imports_list: list[dict] = spec.get("imports", [])
        if not imports_list:
            print(
                "WARNING: --resolve-imports specified but spec has no imports section "
                "or imports list is empty.",
                file=sys.stderr,
            )

        print("Resolving baseline imports...", file=sys.stderr)
        resolved, provenance, conflicts, patch = _resolve_imports(spec, spec_path)

        # Write resolved spec.
        if args.output_resolved:
            resolved_path = Path(args.output_resolved).resolve()
        else:
            resolved_path = spec_path.parent / "spec.resolved.yaml"
        _write_resolved_spec(resolved, provenance, resolved_path)

        # Write patch.
        if args.output_patch:
            patch_path = Path(args.output_patch).resolve()
        else:
            patch_path = spec_path.parent / "spec.patch.yaml"
        _write_patch(patch, patch_path)

        # Write conflicts.
        if args.output_conflicts:
            conflicts_path = Path(args.output_conflicts).resolve()
        else:
            conflicts_path = spec_path.parent / "spec.conflicts.yaml"
        _write_conflicts(conflicts, conflicts_path)

        # Summary.
        print(
            f"Import resolution complete. "
            f"Fields resolved: {len(provenance)}. "
            f"Conflicts: {len(conflicts)}. "
            f"Patch entries: {len(_flatten(patch))}.",
            file=sys.stderr,
        )
        if conflicts:
            unresolved_conflicts = [c for c in conflicts if c.get("resolution") == "UNRESOLVED"]
            if unresolved_conflicts:
                print(
                    f"WARNING: {len(unresolved_conflicts)} conflict(s) require manual resolution. "
                    f"See {conflicts_path}.",
                    file=sys.stderr,
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
