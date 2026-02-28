#!/usr/bin/env python3
"""Validate spec.yaml against the IR schema and consistency rules.

Checks:
  1. Schema compliance: all required sections and fields present
  2. Source tracing: every extracted value has a paper reference
  3. Informed-POMDP invariant: no privileged fields in execution obs
  4. Shape consistency: symbolic dimensions reference known parameters
  5. UNRESOLVED audit: count and report all unresolved fields
  6. Lock file consistency: hashes match, unresolved counts agree
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")


# ---------------------------------------------------------------------------
# Schema: required top-level sections and their required sub-fields
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = {
    "meta": ["paper"],
    "spaces": [],  # at least one sub-key expected
}

OPTIONAL_SECTIONS = [
    "frames", "timing", "reward", "termination", "gates_track",
    "dynamics", "perception", "world_model", "actor_critic",
    "training", "evaluation", "deployment", "imports",
]

# Fields where "UNRESOLVED" is a valid sentinel
RESOLVABLE_MARKER = "UNRESOLVED"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ValidationReport:
    """Collects errors, warnings, and info messages."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []
        self.unresolved: list[str] = []  # paths to UNRESOLVED fields

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def error(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    def add_info(self, msg: str):
        self.info.append(msg)

    def add_unresolved(self, path: str):
        self.unresolved.append(path)

    def summary(self) -> str:
        lines = []
        lines.append(f"Validation {'PASSED' if self.ok else 'FAILED'}")
        lines.append(f"  Errors:     {len(self.errors)}")
        lines.append(f"  Warnings:   {len(self.warnings)}")
        lines.append(f"  UNRESOLVED: {len(self.unresolved)}")
        lines.append("")
        if self.errors:
            lines.append("ERRORS:")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
            lines.append("")
        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
            lines.append("")
        if self.unresolved:
            lines.append("UNRESOLVED FIELDS:")
            for u in self.unresolved:
                lines.append(f"  ? {u}")
            lines.append("")
        if self.info:
            lines.append("INFO:")
            for i in self.info:
                lines.append(f"  · {i}")
        return "\n".join(lines)


def _walk_leaves(obj, path=""):
    """Yield (dotted_path, value) for every leaf in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_leaves(v, f"{path}[{i}]")
    else:
        yield path, obj


def _has_source(obj) -> bool:
    """Check if a dict-like object has a source trace field."""
    if isinstance(obj, dict):
        return "source" in obj
    return False


def _walk_dicts_with_source(obj, path=""):
    """Yield (path, dict) for dicts that should have source traces."""
    if isinstance(obj, dict):
        if "source" in obj:
            yield path, obj
        for k, v in obj.items():
            yield from _walk_dicts_with_source(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_dicts_with_source(v, f"{path}[{i}]")


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def check_schema(spec: dict, report: ValidationReport):
    """Check required sections and fields are present."""
    for section, required_fields in REQUIRED_SECTIONS.items():
        if section not in spec:
            report.error(f"Missing required section: {section}")
            continue
        for field in required_fields:
            if field not in spec[section]:
                report.error(f"Missing required field: {section}.{field}")

    present_sections = [s for s in OPTIONAL_SECTIONS if s in spec]
    missing_sections = [s for s in OPTIONAL_SECTIONS if s not in spec]
    report.add_info(f"Present sections: {', '.join(present_sections)}")
    if missing_sections:
        report.warn(f"Missing optional sections: {', '.join(missing_sections)}")

    # Spaces must have at least one sub-section
    if "spaces" in spec:
        space_keys = [k for k in spec["spaces"] if isinstance(spec["spaces"].get(k), (list, dict))]
        if not space_keys:
            report.error("spaces section has no sub-sections (observation_exec, information_train, action, state)")


def check_unresolved(spec: dict, report: ValidationReport):
    """Find and count all UNRESOLVED fields."""
    for path, value in _walk_leaves(spec):
        if value == RESOLVABLE_MARKER:
            report.add_unresolved(path)

    # Check meta.unresolved_count consistency
    meta_count = spec.get("meta", {}).get("unresolved_count")
    actual_count = len(report.unresolved)
    if meta_count is not None and meta_count != actual_count:
        report.error(
            f"meta.unresolved_count ({meta_count}) != actual UNRESOLVED fields ({actual_count})"
        )
    elif meta_count is None and actual_count > 0:
        report.warn(
            f"meta.unresolved_count not set, but {actual_count} UNRESOLVED fields found"
        )


def check_source_traces(spec: dict, report: ValidationReport):
    """Verify source traces have at least one location field."""
    location_fields = ["section", "table", "equation", "figure", "page"]
    missing_count = 0

    for path, obj in _walk_dicts_with_source(spec):
        if path.startswith("meta"):
            continue  # meta doesn't need source traces
        source = obj.get("source")
        if source is None:
            continue
        if isinstance(source, str):
            # String source is acceptable (e.g., "§3.2")
            continue
        if isinstance(source, dict):
            has_location = any(source.get(f) is not None for f in location_fields)
            if not has_location:
                missing_count += 1
                if missing_count <= 10:  # cap verbose output
                    report.warn(f"Source trace at {path}.source has no location fields")

    if missing_count > 10:
        report.warn(f"... and {missing_count - 10} more source traces without locations")
    if missing_count == 0:
        report.add_info("All source traces have location fields")


def check_informed_split(spec: dict, report: ValidationReport):
    """Verify informed-POMDP invariant: privileged fields not in execution obs."""
    spaces = spec.get("spaces", {})
    obs_exec = spaces.get("observation_exec", [])
    info_train = spaces.get("information_train", [])

    if not obs_exec or not info_train:
        report.add_info("Informed-POMDP check skipped (missing observation or information fields)")
        return

    exec_names = set()
    for field in obs_exec:
        if isinstance(field, dict) and "name" in field:
            exec_names.add(field["name"])

    train_names = set()
    for field in info_train:
        if isinstance(field, dict) and "name" in field:
            train_names.add(field["name"])

    overlap = exec_names & train_names
    if overlap:
        report.error(
            f"Informed-POMDP violation: fields in both observation_exec and "
            f"information_train: {overlap}"
        )
    else:
        report.add_info("Informed-POMDP split: OK (no privileged fields in execution obs)")


def check_shapes(spec: dict, report: ValidationReport):
    """Check that symbolic shape dimensions reference known parameters."""
    # Collect known symbolic names from dynamics parameters
    known_symbols = set()
    dynamics = spec.get("dynamics", {})
    for param in dynamics.get("parameters", []):
        if isinstance(param, dict) and "name" in param:
            known_symbols.add(param["name"])

    # Check all shape fields across spaces
    spaces = spec.get("spaces", {})
    for space_key in ["observation_exec", "information_train", "state"]:
        fields = spaces.get(space_key, [])
        for field in fields:
            if not isinstance(field, dict):
                continue
            shape = field.get("shape", [])
            for dim in shape:
                if isinstance(dim, str) and dim != RESOLVABLE_MARKER:
                    if dim not in known_symbols:
                        report.warn(
                            f"Symbolic dimension '{dim}' in "
                            f"spaces.{space_key}.{field.get('name', '?')}.shape "
                            f"not found in dynamics.parameters"
                        )


def check_lock_file(spec_path: Path, spec_text: str, report: ValidationReport):
    """Validate spec.lock.json consistency if it exists."""
    lock_path = spec_path.parent / "spec.lock.json"
    if not lock_path.exists():
        report.add_info("No spec.lock.json found (skipping lock validation)")
        return

    try:
        lock = json.loads(lock_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        report.error(f"Failed to read spec.lock.json: {e}")
        return

    # Check hash
    expected_hash = hashlib.sha256(spec_text.encode()).hexdigest()
    lock_hash = lock.get("spec_yaml_sha256")
    if lock_hash and lock_hash != expected_hash:
        report.error(
            f"spec.lock.json hash mismatch: lock={lock_hash[:16]}... "
            f"actual={expected_hash[:16]}..."
        )
    elif lock_hash:
        report.add_info("spec.lock.json hash: OK")

    # Check unresolved count
    lock_unresolved = lock.get("unresolved_count")
    if lock_unresolved is not None:
        actual = len(report.unresolved)
        if lock_unresolved != actual:
            report.error(
                f"spec.lock.json unresolved_count ({lock_unresolved}) "
                f"!= actual ({actual})"
            )


def check_reward_ast(spec: dict, report: ValidationReport):
    """Check reward expression ASTs are well-formed (not just strings)."""
    reward = spec.get("reward", {})
    terms = reward.get("terms", [])
    for i, term in enumerate(terms):
        if not isinstance(term, dict):
            continue
        ast = term.get("expression_ast")
        if ast == RESOLVABLE_MARKER:
            continue  # Already counted as UNRESOLVED
        if isinstance(ast, str):
            report.warn(
                f"reward.terms[{i}].expression_ast is a string, not a structured AST. "
                f"Consider parsing into ExprNode format."
            )
        elif isinstance(ast, dict):
            if "type" not in ast and "op" not in ast:
                report.warn(
                    f"reward.terms[{i}].expression_ast missing 'type' or 'op' key"
                )


def check_termination_ast(spec: dict, report: ValidationReport):
    """Check termination condition ASTs are well-formed."""
    termination = spec.get("termination", {})
    conditions = termination.get("conditions", [])
    for i, cond in enumerate(conditions):
        if not isinstance(cond, dict):
            continue
        ast = cond.get("condition_ast")
        if ast == RESOLVABLE_MARKER:
            continue
        if isinstance(ast, str):
            report.warn(
                f"termination.conditions[{i}].condition_ast is a string, not a structured AST. "
                f"Consider parsing into BoolNode format."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate(spec_path: Path) -> ValidationReport:
    """Run all validation checks on a spec.yaml file."""
    report = ValidationReport()

    spec_text = spec_path.read_text()
    try:
        spec = yaml.safe_load(spec_text)
    except yaml.YAMLError as e:
        report.error(f"Invalid YAML: {e}")
        return report

    if not isinstance(spec, dict):
        report.error("spec.yaml root must be a YAML mapping")
        return report

    check_schema(spec, report)
    check_unresolved(spec, report)
    check_source_traces(spec, report)
    check_informed_split(spec, report)
    check_shapes(spec, report)
    check_reward_ast(spec, report)
    check_termination_ast(spec, report)
    check_lock_file(spec_path, spec_text, report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate spec.yaml against schema and consistency rules"
    )
    parser.add_argument(
        "--spec", required=True, type=Path,
        help="Path to spec.yaml"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output report as JSON instead of text"
    )
    args = parser.parse_args()

    if not args.spec.exists():
        print(f"Error: {args.spec} not found", file=sys.stderr)
        sys.exit(2)

    report = validate(args.spec)

    if args.json:
        output = {
            "valid": report.ok,
            "errors": report.errors,
            "warnings": report.warnings,
            "unresolved": report.unresolved,
            "unresolved_count": len(report.unresolved),
            "info": report.info,
        }
        print(json.dumps(output, indent=2))
    else:
        print(report.summary())

    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
