#!/usr/bin/env python3
"""
gen_tests.py — Paper-drift detector: generate pytest compliance tests from spec.yaml.

Given a compiled spec.yaml produced by emit_yaml.py, this script generates a
suite of pytest test files that verify your implementation matches the paper's
stated specification.  Tests cover:

  * Space shapes, dtypes, and action bounds
  * Reward discount and expression ASTs
  * Termination condition boolean logic
  * Domain-randomization ranges and resample frequencies
  * Control-loop timing and pipeline delays
  * Informed-POMDP privileged-info isolation
  * Stub tests for every UNRESOLVED field (skipped, but visible in CI)

Generated tests are ~80% complete templates.  The remaining 20% is project-specific
wiring (env construction, policy loading, model access) marked with TODO comments.

Usage
-----
    python gen_tests.py --spec spec.yaml --output-dir tests/spec_compliance/
    python gen_tests.py --spec spec.yaml  # default output-dir: tests/spec_compliance/

    # Via environment variable:
    SPEC_YAML_PATH=/path/to/spec.yaml python gen_tests.py

Dependencies
------------
    stdlib only + PyYAML (pip install pyyaml)
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML is required.  Install with: pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNRESOLVED = "UNRESOLVED"
TOOL_VERSION = "0.1.0"

# Dtype → numpy/torch dtype string mapping used in assertions.
_DTYPE_MAP: dict[str, str] = {
    "float32": "float32",
    "float64": "float64",
    "bool": "bool",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "int8": "int8",
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _is_unresolved(value: Any) -> bool:
    """Return True if *value* is the sentinel string 'UNRESOLVED' (recursive)."""
    if isinstance(value, str):
        return value == UNRESOLVED
    return False


def _safe_name(raw: str) -> str:
    """Convert an arbitrary string into a valid Python identifier fragment."""
    import re
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", str(raw))
    # Collapse runs of underscores.
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "field"


def _collect_unresolved_paths(value: Any, path: str) -> list[tuple[str, Any]]:
    """
    Recursively walk *value* and collect (json_path, context_hint) tuples
    for every leaf that equals the string 'UNRESOLVED'.
    """
    results: list[tuple[str, Any]] = []
    if isinstance(value, str) and value == UNRESOLVED:
        results.append((path, None))
    elif isinstance(value, dict):
        for k, v in value.items():
            child_path = f"{path}.{k}" if path else k
            results.extend(_collect_unresolved_paths(v, child_path))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            child_path = f"{path}[{i}]"
            results.extend(_collect_unresolved_paths(item, child_path))
    return results


def _shape_to_python(shape: list[Any]) -> tuple[str, str]:
    """
    Render a shape list to a Python tuple literal, treating symbolic dims as 0.

    Returns (tuple_literal, comment_line) where comment_line is a companion
    comment string (may be empty) describing any symbolic dimensions.
    Keeping comments separate prevents inline-comment-breaks-syntax issues.
    """
    parts = []
    symbol_hints: list[str] = []
    for i, dim in enumerate(shape):
        if isinstance(dim, int):
            parts.append(str(dim))
        elif isinstance(dim, str) and dim != UNRESOLVED:
            # Symbolic dimension: use 0 as placeholder, record the symbol name.
            parts.append("0")
            symbol_hints.append(f"dim[{i}] is symbolic: '{dim}'")
        else:
            parts.append("0")
            symbol_hints.append(f"dim[{i}] is UNRESOLVED")
    tuple_lit = "(" + ", ".join(parts) + ("," if len(parts) == 1 else "") + ")"
    comment = ("  # " + "; ".join(symbol_hints)) if symbol_hints else ""
    return tuple_lit, comment


def _source_ref(field: dict | None) -> str:
    """Extract the source string from a field dict, falling back gracefully."""
    if not field:
        return "unknown source"
    src = field.get("source", "unknown source")
    if _is_unresolved(src) or not src:
        return "unknown source"
    return str(src)


def _indent(text: str, spaces: int = 4) -> str:
    """Indent every line of *text* by *spaces* spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


def _wrap_docstring(text: str) -> str:
    """Format *text* as a properly indented triple-quoted docstring body."""
    lines = textwrap.wrap(text, width=88)
    return '"""' + lines[0] + ('"""' if len(lines) == 1 else "\n    " + "\n    ".join(lines[1:]) + '\n    """')


def _render_ast_comment(ast: Any, indent: int = 0) -> str:
    """
    Render an expression/boolean AST dict as a human-readable comment block.
    Only used for generating TODO hints in test bodies — not for evaluation.
    """
    if ast is None or _is_unresolved(ast):
        return "# AST: UNRESOLVED"
    if isinstance(ast, dict):
        node_type = ast.get("type", ast.get("op", "?"))
        return f"# AST node type: {node_type}"
    return f"# AST: {ast!r}"


# ---------------------------------------------------------------------------
# AST evaluator code-generator (produces Python source as a string)
# ---------------------------------------------------------------------------


def _gen_eval_expr_ast(ast: Any, state_var: str = "state") -> str:
    """
    Generate a Python expression string that evaluates an expression AST dict
    against a state dict.  Used to produce in-test reference computations.

    The generated expression uses the synthetic state dict directly; it does
    NOT import from the project under test.

    Returns a Python expression string suitable for embedding in generated test source.
    """
    if ast is None or _is_unresolved(ast):
        return "None  # UNRESOLVED AST"

    if not isinstance(ast, dict):
        # Scalar literal passed directly (legacy format).
        return repr(ast)

    node_type = ast.get("type")

    if node_type == "literal":
        val = ast.get("value", 0.0)
        return repr(float(val)) if isinstance(val, (int, float)) else repr(val)

    if node_type == "field":
        name = ast.get("name", "unknown_field")
        return f'{state_var}.get("{name}", 0.0)'

    if node_type == "op":
        op = ast.get("op", "")
        args = ast.get("args", [])
        params = ast.get("params", {})
        evaluated_args = [_gen_eval_expr_ast(a, state_var) for a in args]

        if op == "add":
            return "(" + " + ".join(evaluated_args) + ")"
        if op == "subtract":
            return "(" + " - ".join(evaluated_args) + ")"
        if op == "multiply":
            return "(" + " * ".join(evaluated_args) + ")"
        if op == "divide":
            return "(" + " / ".join(evaluated_args) + ")"
        if op == "clamp":
            lo = params.get("min", -1.0)
            hi = params.get("max", 1.0)
            inner = evaluated_args[0] if evaluated_args else "0.0"
            return f"max({lo!r}, min({hi!r}, {inner}))"
        if op in ("min", "max"):
            return f"{op}({', '.join(evaluated_args)})"
        if op == "norm":
            inner = evaluated_args[0] if evaluated_args else "0.0"
            return f"({inner} ** 2) ** 0.5  # norm"
        if op == "pow":
            base = evaluated_args[0] if len(evaluated_args) > 0 else "0.0"
            exp_ = evaluated_args[1] if len(evaluated_args) > 1 else "1.0"
            return f"({base} ** {exp_})"
        # Unknown op — emit as comment with zero fallback.
        return f"0.0  # TODO: unsupported op '{op}'"

    if node_type == "func":
        func_name = ast.get("func_name", "unknown")
        args = ast.get("args", [])
        evaluated_args = [_gen_eval_expr_ast(a, state_var) for a in args]
        if func_name == "norm":
            inner = evaluated_args[0] if evaluated_args else "0.0"
            return f"({inner} ** 2) ** 0.5"
        return f"0.0  # TODO: unsupported func '{func_name}'"

    # Fallback for legacy dict format with 'op' at top level (non-typed nodes).
    op = ast.get("op", "")
    args = ast.get("args", [])
    if op and args:
        evaluated_args = [_gen_eval_expr_ast(a, state_var) for a in args]
        if op == "multiply":
            return "(" + " * ".join(evaluated_args) + ")"
        if op == "add":
            return "(" + " + ".join(evaluated_args) + ")"
    return "0.0  # TODO: interpret AST manually"


def _gen_eval_bool_ast(ast: Any, state_var: str = "state") -> str:
    """
    Generate a Python expression string that evaluates a boolean AST dict.
    Used to produce reference truth values in termination condition tests.
    """
    if ast is None or _is_unresolved(ast):
        return "None  # UNRESOLVED boolean AST"

    if not isinstance(ast, dict):
        return repr(ast)

    node_type = ast.get("type")

    if node_type == "compare":
        op = ast.get("op", "eq")
        left = _gen_eval_expr_ast(ast.get("left", {}), state_var)
        right = _gen_eval_expr_ast(ast.get("right", {}), state_var)
        op_map = {"lt": "<", "gt": ">", "le": "<=", "ge": ">=", "eq": "==", "ne": "!="}
        py_op = op_map.get(op, "==")
        return f"({left} {py_op} {right})"

    if node_type == "logic":
        logic_op = ast.get("logic_op", "and")
        children = ast.get("children", [])
        evaluated = [_gen_eval_bool_ast(c, state_var) for c in children]
        if logic_op == "not" and evaluated:
            return f"(not {evaluated[0]})"
        if logic_op == "and":
            return "(" + " and ".join(evaluated) + ")"
        if logic_op == "or":
            return "(" + " or ".join(evaluated) + ")"
        return "(False)"

    if node_type == "field_check":
        left = ast.get("left")
        if left and isinstance(left, dict):
            name = left.get("name", "unknown")
            return f"bool({state_var}.get(\"{name}\", False))"
        return "False  # TODO: field_check"

    # Legacy format without 'type' key — op-based.
    op = ast.get("op", "")
    args = ast.get("args", [])
    if op in ("lt", "gt", "le", "ge", "eq", "ne") and len(args) >= 2:
        left = _gen_eval_expr_ast(args[0], state_var)
        right = _gen_eval_expr_ast(args[1], state_var)
        op_map = {"lt": "<", "gt": ">", "le": "<=", "ge": ">=", "eq": "==", "ne": "!="}
        return f"({left} {op_map[op]} {right})"
    if op == "and" and args:
        evaluated = [_gen_eval_bool_ast(a, state_var) for a in args]
        return "(" + " and ".join(evaluated) + ")"
    if op == "or" and args:
        evaluated = [_gen_eval_bool_ast(a, state_var) for a in args]
        return "(" + " or ".join(evaluated) + ")"
    if op == "not" and args:
        return f"(not {_gen_eval_bool_ast(args[0], state_var)})"

    return "False  # TODO: interpret boolean AST manually"


def _extract_bool_ast_fields(ast: Any) -> list[str]:
    """Walk a boolean AST and collect field names referenced in comparisons."""
    if not isinstance(ast, dict):
        return []
    results: list[str] = []
    node_type = ast.get("type")
    if node_type == "compare":
        for side in ("left", "right"):
            node = ast.get(side)
            if isinstance(node, dict) and node.get("type") == "field":
                name = node.get("name")
                if name:
                    results.append(name)
            elif isinstance(node, dict):
                results.extend(_extract_bool_ast_fields(node))
    elif node_type == "logic":
        for child in ast.get("children", []):
            results.extend(_extract_bool_ast_fields(child))
    elif node_type == "field_check":
        left = ast.get("left")
        if isinstance(left, dict):
            name = left.get("name")
            if name:
                results.append(name)
    # Legacy format.
    for arg in ast.get("args", []):
        results.extend(_extract_bool_ast_fields(arg))
    return results


def _extract_expr_ast_fields(ast: Any) -> list[str]:
    """Walk an expression AST and collect field names referenced."""
    if not isinstance(ast, dict):
        return []
    results: list[str] = []
    if ast.get("type") == "field":
        name = ast.get("name")
        if name:
            results.append(name)
    for arg in ast.get("args", []):
        results.extend(_extract_expr_ast_fields(arg))
    return results


# ---------------------------------------------------------------------------
# Synthetic state builder helper
# ---------------------------------------------------------------------------


def _build_synthetic_state_entries(spec: dict) -> list[tuple[str, str]]:
    """
    Return a list of (field_name, default_expr) pairs covering all fields in
    spaces.state + spaces.observation_exec + spaces.information_train.
    Used by make_synthetic_state in conftest.py.
    """
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()

    spaces = spec.get("spaces") or {}

    def _add_fields(fields: list[dict]) -> None:
        for f in fields:
            name = f.get("name", "")
            if not name or _is_unresolved(name) or name in seen:
                continue
            seen.add(name)
            shape = f.get("shape", [])
            dtype = f.get("dtype", "float32")
            if _is_unresolved(dtype):
                dtype = "float32"
            # Default to scalar 0.0 so that AST-based comparisons work correctly
            # without requiring numpy/torch.  Users replace these with real
            # env observations when wiring up tests.
            if dtype == "bool":
                default_expr = "False"
            else:
                default_expr = "0.0"

            entries.append((name, default_expr))

    _add_fields(spaces.get("observation_exec") or [])
    _add_fields(spaces.get("information_train") or [])
    _add_fields(spaces.get("state") or [])
    return entries


# ---------------------------------------------------------------------------
# File generators
# ---------------------------------------------------------------------------


def _gen_conftest(spec: dict, spec_path: str) -> str:
    """Generate conftest.py content."""
    state_entries = _build_synthetic_state_entries(spec)
    entries_code_lines = []
    for name, expr in state_entries:
        entries_code_lines.append(f'        "{name}": {expr},')
    entries_code = "\n".join(entries_code_lines) if entries_code_lines else "        # No state fields found in spec"

    # Check if information_train has fields for the decoder targets fixture.
    info_train = (spec.get("spaces") or {}).get("information_train") or []
    info_names = [f.get("name", "") for f in info_train if not _is_unresolved(f.get("name", ""))]
    info_names_repr = repr(info_names)

    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        # ---------------------------------------------------------------------------
        # conftest.py — Shared fixtures for spec compliance tests.
        # Generated by gen_tests.py {TOOL_VERSION}
        # Source spec: {spec_path}
        #
        # SPEC PATH is configurable via the SPEC_YAML_PATH environment variable:
        #   SPEC_YAML_PATH=/path/to/spec.yaml pytest tests/spec_compliance/
        # ---------------------------------------------------------------------------
        \"\"\"
        Shared fixtures for spec.yaml compliance tests.

        All fixtures in this conftest.py are available to every test in the
        tests/spec_compliance/ directory without explicit import.

        spec.yaml path resolution order:
          1. SPEC_YAML_PATH environment variable
          2. --spec-path pytest CLI option (registered below)
          3. Default: {spec_path!r}
        \"\"\"

        from __future__ import annotations

        import os
        import pathlib
        from typing import Any

        import pytest
        import yaml

        # ---------------------------------------------------------------------------
        # Spec path resolution
        # ---------------------------------------------------------------------------

        _DEFAULT_SPEC_PATH = {spec_path!r}


        def _resolve_spec_path() -> pathlib.Path:
            \"\"\"Resolve spec.yaml path from environment or default.\"\"\"
            env_path = os.environ.get("SPEC_YAML_PATH")
            if env_path:
                return pathlib.Path(env_path)
            return pathlib.Path(_DEFAULT_SPEC_PATH)


        # ---------------------------------------------------------------------------
        # pytest CLI option for spec path
        # ---------------------------------------------------------------------------


        def pytest_addoption(parser: pytest.Parser) -> None:
            parser.addoption(
                "--spec-path",
                default=None,
                metavar="PATH",
                help="Path to spec.yaml (overrides SPEC_YAML_PATH env var and default).",
            )


        # ---------------------------------------------------------------------------
        # Core fixtures
        # ---------------------------------------------------------------------------


        @pytest.fixture(scope="session")
        def spec(request: pytest.FixtureRequest) -> dict[str, Any]:
            \"\"\"
            Load and return the compiled spec.yaml as a nested dict.

            This fixture is session-scoped: the YAML file is read exactly once
            per pytest run.  Tests should treat this dict as read-only.

            spec.yaml: (root)
            \"\"\"
            # CLI option takes highest priority.
            cli_path = request.config.getoption("--spec-path", default=None)
            if cli_path:
                spec_file = pathlib.Path(cli_path)
            else:
                spec_file = _resolve_spec_path()

            if not spec_file.exists():
                pytest.fail(
                    f"spec.yaml not found at {{spec_file}}. "
                    "Set SPEC_YAML_PATH or use --spec-path to override."
                )

            with spec_file.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)

            assert isinstance(data, dict), (
                f"spec.yaml at {{spec_file}} did not parse as a YAML mapping."
            )
            return data


        @pytest.fixture(scope="session")
        def spec_spaces(spec: dict[str, Any]) -> dict[str, Any]:
            \"\"\"Return spec['spaces'] or empty dict if absent.  spec.yaml: spaces\"\"\"
            return spec.get("spaces") or {{}}


        @pytest.fixture(scope="session")
        def spec_timing(spec: dict[str, Any]) -> dict[str, Any]:
            \"\"\"Return spec['timing'] or empty dict if absent.  spec.yaml: timing\"\"\"
            return spec.get("timing") or {{}}


        @pytest.fixture(scope="session")
        def spec_reward(spec: dict[str, Any]) -> dict[str, Any]:
            \"\"\"Return spec['reward'] or empty dict if absent.  spec.yaml: reward\"\"\"
            return spec.get("reward") or {{}}


        @pytest.fixture(scope="session")
        def spec_termination(spec: dict[str, Any]) -> dict[str, Any]:
            \"\"\"Return spec['termination'] or empty dict if absent.  spec.yaml: termination\"\"\"
            return spec.get("termination") or {{}}


        @pytest.fixture(scope="session")
        def spec_dynamics(spec: dict[str, Any]) -> dict[str, Any]:
            \"\"\"Return spec['dynamics'] or empty dict if absent.  spec.yaml: dynamics\"\"\"
            return spec.get("dynamics") or {{}}


        @pytest.fixture(scope="session")
        def information_train_names(spec_spaces: dict[str, Any]) -> list[str]:
            \"\"\"
            Return the list of field names declared in spaces.information_train.

            These are the privileged training-only fields that must NOT appear
            in execution observations.

            spec.yaml: spaces.information_train[*].name
            \"\"\"
            fields = spec_spaces.get("information_train") or []
            return [f.get("name", "") for f in fields if f.get("name") and f.get("name") != "UNRESOLVED"]


        @pytest.fixture
        def make_synthetic_state(spec: dict[str, Any]):
            \"\"\"
            Factory fixture: returns a callable that produces synthetic state dicts.

            The returned function accepts keyword overrides for individual fields.
            All fields present in spaces.state, spaces.observation_exec, and
            spaces.information_train are initialised to zero (or False for booleans).

            Usage:
                def test_something(make_synthetic_state):
                    state = make_synthetic_state(velocity=1.5)
                    assert state["velocity"] == 1.5

            spec.yaml: spaces (all sub-sections)
            \"\"\"
            def _factory(**overrides: Any) -> dict[str, Any]:
                state: dict[str, Any] = {{
        {entries_code}
                }}
                state.update(overrides)
                return state

            return _factory


        @pytest.fixture(scope="session")
        def expected_information_train_names() -> list[str]:
            \"\"\"
            Hardcoded list of information_train field names from spec.

            Used in informed-POMDP tests to verify isolation without loading
            the live spec again.  Extracted at code-generation time.

            spec.yaml: spaces.information_train[*].name
            \"\"\"
            return {info_names_repr}
        """)


# ---------------------------------------------------------------------------


def _gen_test_spaces(spec: dict, spec_path: str) -> str:
    """Generate test_spaces.py content."""
    spaces = spec.get("spaces") or {}
    obs_exec = spaces.get("observation_exec") or []
    info_train = spaces.get("information_train") or []
    action = spaces.get("action") or {}

    lines: list[str] = []
    lines.append(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        # ---------------------------------------------------------------------------
        # test_spaces.py — Space shape / dtype / bounds compliance tests.
        # Generated by gen_tests.py {TOOL_VERSION}
        # Source spec: {spec_path}
        # ---------------------------------------------------------------------------
        \"\"\"
        Verify that observation, information, and action spaces match spec.yaml.

        Each test checks one structural property (shape, dtype, or bounds) of one
        named field.  Tests are independent so failures are precisely localised.

        HOW TO WIRE UP:
        ---------------
        1. Replace the TODO env/policy creation stubs with your actual env/policy.
        2. Replace obs["field_name"] with however your env returns observations.
        3. For action bounds, replace policy.sample_action(obs) with real sampling.
        \"\"\"

        from __future__ import annotations

        # TODO: import your environment and policy constructors here.
        # Example:
        #   from myproject.envs import DroneRacingEnv
        #   from myproject.policies import load_policy

        import pytest

        # ---------------------------------------------------------------------------
        # Fixtures (provided by conftest.py — no import needed)
        # ---------------------------------------------------------------------------
        # spec, spec_spaces, make_synthetic_state


        # ===========================================================================
        # Observation exec fields
        # ===========================================================================
        """))

    if not obs_exec:
        lines.append(textwrap.dedent("""\
            @pytest.mark.skip(reason="No observation_exec fields found in spec.yaml")
            def test_observation_exec_empty():
                \"\"\"spec.yaml: spaces.observation_exec — no fields declared\"\"\"
                raise NotImplementedError("No fields to test")
        """))
    else:
        for idx, field in enumerate(obs_exec):
            name = field.get("name", f"field_{idx}")
            dtype = field.get("dtype", UNRESOLVED)
            shape = field.get("shape", [])
            source = _source_ref(field)
            safe = _safe_name(name)

            if _is_unresolved(name):
                lines.append(textwrap.dedent(f"""\
                    @pytest.mark.skip(reason="UNRESOLVED: spaces.observation_exec[{idx}].name — {source}")
                    def test_observation_exec_{idx}_UNRESOLVED():
                        \"\"\"spec.yaml: spaces.observation_exec[{idx}] — name is UNRESOLVED\"\"\"
                        raise NotImplementedError("UNRESOLVED field")

                """))
                continue

            shape_unresolved = (
                _is_unresolved(shape)
                if not isinstance(shape, list)
                else (
                    any(_is_unresolved(d) for d in shape)
                    or any(isinstance(d, str) and not _is_unresolved(d) for d in shape)
                )
            )  # Also treat symbolic (non-int, non-UNRESOLVED) dims as unresolved for shape tests
            dtype_unresolved = _is_unresolved(dtype)

            # Shape test
            if shape_unresolved:
                lines.append(textwrap.dedent(f"""\
                    @pytest.mark.skip(reason="UNRESOLVED: spaces.observation_exec[{idx}].{name}.shape — {source}")
                    def test_obs_exec_{safe}_shape(spec_spaces):
                        \"\"\"spec.yaml: spaces.observation_exec[{idx}] {name} shape — UNRESOLVED\"\"\"
                        raise NotImplementedError("UNRESOLVED shape")

                """))
            else:
                shape_tuple, shape_comment = _shape_to_python(shape if isinstance(shape, list) else [shape])
                lines.append(textwrap.dedent(f"""\
                    def test_obs_exec_{safe}_shape(spec_spaces):
                        \"\"\"spec.yaml: spaces.observation_exec[{idx}] {name} {dtype} {shape} — {source}\"\"\"
                        # TODO: replace with actual env reset and obs extraction.
                        # Example:
                        #   env = DroneRacingEnv()
                        #   obs, _ = env.reset()
                        #   field = obs["{name}"]
                        #   assert field.shape == {shape_tuple}{shape_comment}
                        expected_shape = {shape_tuple}{shape_comment}
                        spec_field = next(
                            (f for f in spec_spaces.get("observation_exec", [])
                             if f.get("name") == "{name}"),
                            None,
                        )
                        assert spec_field is not None, "Field '{name}' missing from spec"
                        spec_shape = tuple(
                            d for d in spec_field.get("shape", [])
                            if isinstance(d, int)
                        )
                        # NOTE: symbolic dimensions are skipped in spec_shape comparison.
                        concrete_expected = tuple(d for d in expected_shape if isinstance(d, int))
                        assert concrete_expected == spec_shape, (
                            f"Shape mismatch for '{name}': code={{concrete_expected}} spec={{spec_shape}}"
                        )

                """))

            # Dtype test
            if dtype_unresolved:
                lines.append(textwrap.dedent(f"""\
                    @pytest.mark.skip(reason="UNRESOLVED: spaces.observation_exec[{idx}].{name}.dtype — {source}")
                    def test_obs_exec_{safe}_dtype(spec_spaces):
                        \"\"\"spec.yaml: spaces.observation_exec[{idx}] {name} dtype — UNRESOLVED\"\"\"
                        raise NotImplementedError("UNRESOLVED dtype")

                """))
            else:
                lines.append(textwrap.dedent(f"""\
                    def test_obs_exec_{safe}_dtype(spec_spaces):
                        \"\"\"spec.yaml: spaces.observation_exec[{idx}] {name} dtype={dtype} — {source}\"\"\"
                        # TODO: replace with actual observation extraction.
                        # Example:
                        #   obs, _ = env.reset()
                        #   field = obs["{name}"]
                        #   assert str(field.dtype) == "{dtype}"
                        spec_field = next(
                            (f for f in spec_spaces.get("observation_exec", [])
                             if f.get("name") == "{name}"),
                            None,
                        )
                        assert spec_field is not None, "Field '{name}' missing from spec"
                        assert spec_field.get("dtype") == "{dtype}", (
                            f"Dtype mismatch for '{name}': spec says '{dtype}'"
                        )

                """))

    lines.append(textwrap.dedent("""\

        # ===========================================================================
        # Information train (privileged) fields
        # ===========================================================================
    """))

    if not info_train:
        lines.append(textwrap.dedent("""\
            @pytest.mark.skip(reason="No information_train fields found in spec.yaml")
            def test_information_train_empty():
                \"\"\"spec.yaml: spaces.information_train — no fields declared\"\"\"
                raise NotImplementedError("No fields to test")
        """))
    else:
        for idx, field in enumerate(info_train):
            name = field.get("name", f"info_field_{idx}")
            dtype = field.get("dtype", UNRESOLVED)
            shape = field.get("shape", [])
            source = _source_ref(field)
            dreamer_key = field.get("informed_dreamer_key")
            safe = _safe_name(name)

            if _is_unresolved(name):
                lines.append(textwrap.dedent(f"""\
                    @pytest.mark.skip(reason="UNRESOLVED: spaces.information_train[{idx}].name — {source}")
                    def test_information_train_{idx}_UNRESOLVED():
                        \"\"\"spec.yaml: spaces.information_train[{idx}] — name is UNRESOLVED\"\"\"
                        raise NotImplementedError("UNRESOLVED field")

                """))
                continue

            shape_unresolved = (
                _is_unresolved(shape)
                if not isinstance(shape, list)
                else (
                    any(_is_unresolved(d) for d in shape)
                    or any(isinstance(d, str) and not _is_unresolved(d) for d in shape)
                )
            )  # Also treat symbolic (non-int, non-UNRESOLVED) dims as unresolved for shape tests

            if shape_unresolved:
                lines.append(textwrap.dedent(f"""\
                    @pytest.mark.skip(reason="UNRESOLVED: spaces.information_train[{idx}].{name}.shape — {source}")
                    def test_info_train_{safe}_shape(spec_spaces):
                        \"\"\"spec.yaml: spaces.information_train[{idx}] {name} shape — UNRESOLVED\"\"\"
                        raise NotImplementedError("UNRESOLVED shape")

                """))
            else:
                shape_tuple, shape_comment = _shape_to_python(shape if isinstance(shape, list) else [shape])
                lines.append(textwrap.dedent(f"""\
                    def test_info_train_{safe}_shape(spec_spaces):
                        \"\"\"spec.yaml: spaces.information_train[{idx}] {name} {dtype} {shape} — {source}\"\"\"
                        # TODO: replace with actual decoder/world-model output extraction.
                        # Example:
                        #   decoder_output = world_model.decode(latent)
                        #   field = decoder_output["{name}"]
                        #   assert field.shape == {shape_tuple}{shape_comment}
                        expected_shape = {shape_tuple}{shape_comment}
                        spec_field = next(
                            (f for f in spec_spaces.get("information_train", [])
                             if f.get("name") == "{name}"),
                            None,
                        )
                        assert spec_field is not None, "Field '{name}' missing from spec"
                        spec_shape = tuple(
                            d for d in spec_field.get("shape", [])
                            if isinstance(d, int)
                        )
                        concrete_expected = tuple(d for d in expected_shape if isinstance(d, int))
                        assert concrete_expected == spec_shape, (
                            f"Shape mismatch for '{name}': code={{concrete_expected}} spec={{spec_shape}}"
                        )

                """))

            if dreamer_key and not _is_unresolved(dreamer_key):
                lines.append(textwrap.dedent(f"""\
                    def test_info_train_{safe}_dreamer_key_pattern(spec_spaces):
                        \"\"\"
                        spec.yaml: spaces.information_train[{idx}] {name}
                        informed_dreamer_key regex must match the field name — {source}
                        \"\"\"
                        import re
                        pattern = {dreamer_key!r}
                        assert re.search(pattern, "{name}"), (
                            f"Field '{name}' does not match its own dreamer_key pattern {{pattern!r}}"
                        )
                        # TODO: also verify your world-model decoder uses this exact regex
                        # to gate which fields it decodes.  Example:
                        #   decoder_keys = world_model.get_decoder_gating_patterns()
                        #   assert any(re.fullmatch(p, "{name}") for p in decoder_keys)

                """))

    lines.append(textwrap.dedent("""\

        # ===========================================================================
        # Action space
        # ===========================================================================
    """))

    if not action or _is_unresolved(action):
        lines.append(textwrap.dedent("""\
            @pytest.mark.skip(reason="UNRESOLVED: spaces.action — not declared in spec")
            def test_action_UNRESOLVED():
                \"\"\"spec.yaml: spaces.action — UNRESOLVED\"\"\"
                raise NotImplementedError("UNRESOLVED action space")
        """))
    else:
        act_name = action.get("name", UNRESOLVED)
        act_dtype = action.get("dtype", UNRESOLVED)
        act_shape = action.get("shape", [])
        act_bounds = action.get("bounds", [UNRESOLVED, UNRESOLVED])
        act_source = _source_ref(action)

        if not _is_unresolved(act_shape):
            shape_tuple, shape_comment = _shape_to_python(act_shape if isinstance(act_shape, list) else [act_shape])
            lines.append(textwrap.dedent(f"""\
                def test_action_shape(spec_spaces):
                    \"\"\"spec.yaml: spaces.action shape={act_shape} — {act_source}\"\"\"
                    # TODO: sample an action from your policy and verify its shape.
                    # Example:
                    #   obs, _ = env.reset()
                    #   action = policy.sample_action(obs)
                    #   assert action.shape == {shape_tuple}{shape_comment}
                    action_spec = spec_spaces.get("action", {{}})
                    if action_spec == "UNRESOLVED":
                        pytest.skip("action spec is UNRESOLVED")
                    expected_shape = {shape_tuple}{shape_comment}
                    spec_shape = tuple(
                        d for d in action_spec.get("shape", [])
                        if isinstance(d, int)
                    )
                    concrete_expected = tuple(d for d in expected_shape if isinstance(d, int))
                    assert concrete_expected == spec_shape, (
                        f"Action shape mismatch: code={{concrete_expected}} spec={{spec_shape}}"
                    )

            """))
        else:
            lines.append(textwrap.dedent(f"""\
                @pytest.mark.skip(reason="UNRESOLVED: spaces.action.shape — {act_source}")
                def test_action_shape():
                    \"\"\"spec.yaml: spaces.action shape — UNRESOLVED\"\"\"
                    raise NotImplementedError("UNRESOLVED action shape")

            """))

        if not _is_unresolved(act_dtype):
            lines.append(textwrap.dedent(f"""\
                def test_action_dtype(spec_spaces):
                    \"\"\"spec.yaml: spaces.action dtype={act_dtype} — {act_source}\"\"\"
                    # TODO: replace with real action extraction.
                    action_spec = spec_spaces.get("action", {{}})
                    if action_spec == "UNRESOLVED":
                        pytest.skip("action spec is UNRESOLVED")
                    assert action_spec.get("dtype") == "{act_dtype}", (
                        "Action dtype does not match spec: expected {act_dtype}"
                    )

            """))

        bounds_unresolved = (
            isinstance(act_bounds, list)
            and len(act_bounds) == 2
            and not (_is_unresolved(act_bounds[0]) or _is_unresolved(act_bounds[1]))
        )
        if isinstance(act_bounds, list) and len(act_bounds) == 2 and not _is_unresolved(act_bounds[0]) and not _is_unresolved(act_bounds[1]):
            lo, hi = act_bounds[0], act_bounds[1]
            lines.append(textwrap.dedent(f"""\
                def test_action_bounds_min(spec_spaces):
                    \"\"\"spec.yaml: spaces.action bounds[0]={lo} (lower bound) — {act_source}\"\"\"
                    # TODO: sample 1000 actions and verify none violates the lower bound.
                    # Example:
                    #   for _ in range(1000):
                    #       obs, _ = env.reset()
                    #       action = policy.sample_action(obs)
                    #       assert float(action.min()) >= {lo!r}, f"Action below lower bound"
                    action_spec = spec_spaces.get("action", {{}})
                    if action_spec == "UNRESOLVED":
                        pytest.skip("action spec is UNRESOLVED")
                    spec_bounds = action_spec.get("bounds", [None, None])
                    if isinstance(spec_bounds, list) and len(spec_bounds) >= 1:
                        assert spec_bounds[0] == pytest.approx({lo!r}), (
                            f"Action lower bound mismatch: spec says {lo!r}"
                        )

                def test_action_bounds_max(spec_spaces):
                    \"\"\"spec.yaml: spaces.action bounds[1]={hi} (upper bound) — {act_source}\"\"\"
                    # TODO: sample 1000 actions and verify none violates the upper bound.
                    # Example:
                    #   for _ in range(1000):
                    #       obs, _ = env.reset()
                    #       action = policy.sample_action(obs)
                    #       assert float(action.max()) <= {hi!r}, f"Action above upper bound"
                    action_spec = spec_spaces.get("action", {{}})
                    if action_spec == "UNRESOLVED":
                        pytest.skip("action spec is UNRESOLVED")
                    spec_bounds = action_spec.get("bounds", [None, None])
                    if isinstance(spec_bounds, list) and len(spec_bounds) >= 2:
                        assert spec_bounds[1] == pytest.approx({hi!r}), (
                            f"Action upper bound mismatch: spec says {hi!r}"
                        )

            """))
        else:
            lines.append(textwrap.dedent(f"""\
                @pytest.mark.skip(reason="UNRESOLVED: spaces.action.bounds — {act_source}")
                def test_action_bounds():
                    \"\"\"spec.yaml: spaces.action bounds — UNRESOLVED\"\"\"
                    raise NotImplementedError("UNRESOLVED action bounds")

            """))

    return "\n".join(lines)


# ---------------------------------------------------------------------------


def _gen_test_reward(spec: dict, spec_path: str) -> str:
    """Generate test_reward.py content."""
    reward = spec.get("reward") or {}
    discount = reward.get("discount", UNRESOLVED)
    terms = reward.get("terms") or []

    lines: list[str] = []
    lines.append(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        # ---------------------------------------------------------------------------
        # test_reward.py — Reward expression compliance tests.
        # Generated by gen_tests.py {TOOL_VERSION}
        # Source spec: {spec_path}
        # ---------------------------------------------------------------------------
        \"\"\"
        Verify that reward computation matches spec.yaml expression trees.

        HOW TO WIRE UP:
        ---------------
        1. Replace the TODO reward function stubs with calls to your actual reward
           implementation (e.g., reward_fn.compute(state, next_state)).
        2. The spec_ast_result variables compute a reference value from the spec AST.
           Compare these against your implementation's output.
        3. Use pytest.approx() for float comparisons (already in place).
        \"\"\"

        from __future__ import annotations

        # TODO: import your reward function here.
        # Example:
        #   from myproject.rewards import compute_reward

        import pytest


        # ===========================================================================
        # Reward discount
        # ===========================================================================

    """))

    if _is_unresolved(discount):
        lines.append(textwrap.dedent(f"""\
            @pytest.mark.skip(reason="UNRESOLVED: reward.discount")
            def test_discount_UNRESOLVED():
                \"\"\"spec.yaml: reward.discount — UNRESOLVED\"\"\"
                raise NotImplementedError("UNRESOLVED discount")
        """))
    else:
        lines.append(textwrap.dedent(f"""\
            def test_discount_value(spec_reward):
                \"\"\"
                spec.yaml: reward.discount = {discount!r}

                Verify the discount (gamma) used by the actor-critic matches the spec.
                \"\"\"
                spec_discount = spec_reward.get("discount")
                if spec_discount == "UNRESOLVED" or spec_discount is None:
                    pytest.skip("discount is UNRESOLVED in spec")
                # TODO: replace config_discount with your actual config value.
                # Example:
                #   from myproject.config import TrainConfig
                #   config_discount = TrainConfig.load().discount
                config_discount = {discount!r}  # TODO: wire up actual config
                assert config_discount == pytest.approx({discount!r}, rel=1e-6), (
                    f"Discount {{config_discount}} != spec {discount!r}"
                )
        """))

    lines.append(textwrap.dedent("""\

        # ===========================================================================
        # Reward terms
        # ===========================================================================
    """))

    if not terms:
        lines.append(textwrap.dedent("""\
            @pytest.mark.skip(reason="No reward terms found in spec.yaml")
            def test_reward_terms_empty():
                \"\"\"spec.yaml: reward.terms — no terms declared\"\"\"
                raise NotImplementedError("No reward terms")
        """))
    else:
        for idx, term in enumerate(terms):
            name = term.get("name", f"term_{idx}")
            weight = term.get("weight", UNRESOLVED)
            ast = term.get("expression_ast")
            clamp = term.get("clamp")
            source = _source_ref(term)
            safe = _safe_name(name)

            if _is_unresolved(ast) or ast is None:
                lines.append(textwrap.dedent(f"""\
                    @pytest.mark.skip(
                        reason=(
                            "UNRESOLVED: reward.terms[{idx}] '{name}' expression_ast — "
                            "{source} — manual extraction required"
                        )
                    )
                    def test_reward_term_{safe}_expression():
                        \"\"\"
                        spec.yaml: reward.terms[{idx}] '{name}' expression_ast — UNRESOLVED

                        Manual extraction from paper required.  Source: {source}
                        \"\"\"
                        raise NotImplementedError("UNRESOLVED expression_ast")

                """))
            else:
                # Collect fields referenced in AST for synthetic state hints.
                referenced_fields = _extract_expr_ast_fields(ast)
                field_hints = "\n".join(
                    f"        # state['{f}'] — set to a meaningful test value"
                    for f in referenced_fields
                ) or "        # (no named fields detected in AST)"
                ast_expr = _gen_eval_expr_ast(ast, "state")
                ast_comment = _render_ast_comment(ast)

                lines.append(textwrap.dedent(f"""\
                    def test_reward_term_{safe}_expression(spec_reward, make_synthetic_state):
                        \"\"\"
                        spec.yaml: reward.terms[{idx}] '{name}' weight={weight!r}
                        Source: {source}

                        Evaluates the spec AST against a synthetic transition and
                        compares to your reward function's output.
                        \"\"\"
                        # Construct a synthetic state with relevant fields set.
                        # Adjust these values to produce a non-trivial reward signal.
                        state = make_synthetic_state(
                {field_hints}
                        )
                        next_state = make_synthetic_state(
                {field_hints}
                        )
                        {ast_comment}
                        spec_ast_result = {ast_expr}

                        # TODO: call your actual reward function and compare.
                        # Example:
                        #   code_reward = reward_fn.compute_{safe}(state, next_state)
                        #   assert code_reward == pytest.approx(spec_ast_result, rel=1e-4), (
                        #       f"'{name}' reward {{code_reward}} != spec AST result {{spec_ast_result}}"
                        #   )
                        # For now: verify the spec AST evaluates without error and returns a number.
                        assert isinstance(spec_ast_result, (int, float)), (
                            f"Spec AST for '{name}' did not evaluate to a number: {{spec_ast_result!r}}"
                        )

                """))

            if not _is_unresolved(weight):
                lines.append(textwrap.dedent(f"""\
                    def test_reward_term_{safe}_weight(spec_reward):
                        \"\"\"spec.yaml: reward.terms[{idx}] '{name}' weight={weight!r} — {source}\"\"\"
                        terms = spec_reward.get("terms", [])
                        spec_term = next((t for t in terms if t.get("name") == "{name}"), None)
                        assert spec_term is not None, "Term '{name}' not found in spec"
                        spec_weight = spec_term.get("weight")
                        if spec_weight == "UNRESOLVED":
                            pytest.skip("weight is UNRESOLVED for term '{name}'")
                        # TODO: compare against your config.
                        # Example:
                        #   assert reward_cfg.weights["{name}"] == pytest.approx({weight!r})
                        assert spec_weight == pytest.approx({weight!r}, rel=1e-6), (
                            f"Weight for '{name}': spec={{spec_weight!r}}, expected {weight!r}"
                        )

                """))

            if clamp and isinstance(clamp, (list, tuple)) and len(clamp) == 2:
                lo, hi = clamp[0], clamp[1]
                if not _is_unresolved(lo) and not _is_unresolved(hi):
                    lines.append(textwrap.dedent(f"""\
                        def test_reward_term_{safe}_clamp(spec_reward):
                            \"\"\"spec.yaml: reward.terms[{idx}] '{name}' clamp=[{lo}, {hi}] — {source}\"\"\"
                            # TODO: verify your reward function clamps this term.
                            # Example:
                            #   raw_value = 1e9  # extreme value
                            #   clamped = reward_fn.compute_{safe}_clamped(state, next_state)
                            #   assert {lo!r} <= clamped <= {hi!r}
                            spec_term = next(
                                (t for t in spec_reward.get("terms", []) if t.get("name") == "{name}"),
                                None,
                            )
                            assert spec_term is not None
                            spec_clamp = spec_term.get("clamp", [None, None])
                            assert spec_clamp[0] == pytest.approx({lo!r}), f"Clamp min mismatch for '{name}'"
                            assert spec_clamp[1] == pytest.approx({hi!r}), f"Clamp max mismatch for '{name}'"

                    """))

    return "\n".join(lines)


# ---------------------------------------------------------------------------


def _gen_test_termination(spec: dict, spec_path: str) -> str:
    """Generate test_termination.py content."""
    termination = spec.get("termination") or {}
    conditions = termination.get("conditions") or []
    max_steps = termination.get("max_episode_steps", UNRESOLVED)

    lines: list[str] = []
    lines.append(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        # ---------------------------------------------------------------------------
        # test_termination.py — Termination condition compliance tests.
        # Generated by gen_tests.py {TOOL_VERSION}
        # Source spec: {spec_path}
        # ---------------------------------------------------------------------------
        \"\"\"
        Verify termination conditions match spec.yaml boolean ASTs.

        Each condition generates three test functions:
          - Positive case: condition SHOULD trigger (→ episode ends)
          - Negative case: condition should NOT trigger
          - Boundary case: test at the exact threshold value

        HOW TO WIRE UP:
        ---------------
        1. Replace TODO env construction with your actual environment.
        2. Replace TODO termination function calls with your implementation.
        3. Adjust synthetic state values to valid ranges for positive/negative cases.
        \"\"\"

        from __future__ import annotations

        # TODO: import your termination function here.
        # Example:
        #   from myproject.termination import check_termination

        import pytest

    """))

    if not conditions:
        lines.append(textwrap.dedent("""\
            @pytest.mark.skip(reason="No termination conditions found in spec.yaml")
            def test_termination_empty():
                \"\"\"spec.yaml: termination.conditions — no conditions declared\"\"\"
                raise NotImplementedError("No conditions")
        """))
    else:
        for idx, cond in enumerate(conditions):
            name = cond.get("name", f"condition_{idx}")
            ast = cond.get("condition_ast")
            source = _source_ref(cond)
            safe = _safe_name(name)

            if _is_unresolved(ast) or ast is None:
                lines.append(textwrap.dedent(f"""\
                    @pytest.mark.skip(
                        reason=(
                            "UNRESOLVED: termination.conditions[{idx}] '{name}' condition_ast — "
                            "{source} — manual extraction required"
                        )
                    )
                    def test_termination_{safe}_positive():
                        \"\"\"spec.yaml: termination.conditions[{idx}] '{name}' — UNRESOLVED\"\"\"
                        raise NotImplementedError("UNRESOLVED condition_ast")

                    @pytest.mark.skip(reason="UNRESOLVED: termination.conditions[{idx}] '{name}'")
                    def test_termination_{safe}_negative():
                        \"\"\"spec.yaml: termination.conditions[{idx}] '{name}' — UNRESOLVED\"\"\"
                        raise NotImplementedError("UNRESOLVED condition_ast")

                    @pytest.mark.skip(reason="UNRESOLVED: termination.conditions[{idx}] '{name}'")
                    def test_termination_{safe}_boundary():
                        \"\"\"spec.yaml: termination.conditions[{idx}] '{name}' — UNRESOLVED\"\"\"
                        raise NotImplementedError("UNRESOLVED condition_ast")

                """))
            else:
                referenced_fields = _extract_bool_ast_fields(ast)
                ast_expr = _gen_eval_bool_ast(ast, "state")

                # Build field override hints.
                def _field_hint_block(suffix: str) -> str:
                    if not referenced_fields:
                        return "        # (no fields detected in AST)"
                    hints = []
                    for f in referenced_fields:
                        hints.append(f"        # TODO: set {f} to a value where condition {suffix}")
                    return "\n".join(hints)

                positive_hints = _field_hint_block("IS triggered")
                negative_hints = _field_hint_block("is NOT triggered")
                boundary_hints = _field_hint_block("is at the exact threshold")

                lines.append(textwrap.dedent(f"""\
                    def test_termination_{safe}_positive(make_synthetic_state):
                        \"\"\"
                        spec.yaml: termination.conditions[{idx}] '{name}' — positive case.

                        The condition SHOULD evaluate to True (episode terminates).
                        Source: {source}
                        \"\"\"
                        # Construct a state where this termination condition fires.
                        state = make_synthetic_state(
                {positive_hints}
                        )
                        # Spec AST reference evaluation:
                        spec_result = {ast_expr}

                        # TODO: call your actual termination checker and compare.
                        # Example:
                        #   code_result = check_termination_{safe}(state)
                        #   assert code_result == spec_result, (
                        #       f"Termination '{name}' mismatch: code={{code_result}} spec={{spec_result}}"
                        #   )
                        # For now: verify the spec AST evaluates without error.
                        assert isinstance(spec_result, bool) or spec_result is None, (
                            f"Spec AST for '{name}' did not evaluate to bool: {{spec_result!r}}"
                        )

                    def test_termination_{safe}_negative(make_synthetic_state):
                        \"\"\"
                        spec.yaml: termination.conditions[{idx}] '{name}' — negative case.

                        The condition should NOT trigger (episode continues).
                        Source: {source}
                        \"\"\"
                        state = make_synthetic_state(
                {negative_hints}
                        )
                        spec_result = {ast_expr}
                        # TODO: assert NOT terminated.
                        # Example:
                        #   assert not check_termination_{safe}(state), (
                        #       f"Termination '{name}' should not fire: state={{state}}"
                        #   )
                        _ = spec_result  # Suppress unused variable warning.

                    def test_termination_{safe}_boundary(make_synthetic_state):
                        \"\"\"
                        spec.yaml: termination.conditions[{idx}] '{name}' — boundary case.

                        Tests at the exact threshold.  Strict vs. non-strict inequalities
                        (< vs. <=) must match the spec AST exactly.
                        Source: {source}
                        \"\"\"
                        state = make_synthetic_state(
                {boundary_hints}
                        )
                        spec_result = {ast_expr}
                        # TODO: assert boundary behavior matches spec.
                        # Example:
                        #   code_result = check_termination_{safe}(state)
                        #   assert code_result == spec_result, (
                        #       f"Boundary mismatch for '{name}': code={{code_result}} spec={{spec_result}}"
                        #   )
                        _ = spec_result

                """))

    if not _is_unresolved(max_steps) and max_steps is not None:
        lines.append(textwrap.dedent(f"""\

            def test_max_episode_steps(spec_termination):
                \"\"\"
                spec.yaml: termination.max_episode_steps = {max_steps}

                Verify the environment's episode length limit matches the spec.
                \"\"\"
                spec_max = spec_termination.get("max_episode_steps")
                if spec_max == "UNRESOLVED" or spec_max is None:
                    pytest.skip("max_episode_steps is UNRESOLVED")
                assert spec_max == {max_steps!r}, (
                    f"max_episode_steps {{spec_max}} != spec {max_steps!r}"
                )
                # TODO: also verify your env respects this limit.
                # Example:
                #   env = MyEnv(max_steps={max_steps})
                #   for _ in range({max_steps} + 5):
                #       _, _, terminated, truncated, _ = env.step(zero_action)
                #       if truncated:
                #           break
                #   assert truncated, "Episode did not truncate at max_episode_steps"
        """))

    return "\n".join(lines)


# ---------------------------------------------------------------------------


def _gen_test_domain_rand(spec: dict, spec_path: str) -> str:
    """Generate test_domain_rand.py content."""
    dynamics = spec.get("dynamics") or {}
    domain_rand = dynamics.get("domain_randomization") or []

    lines: list[str] = []
    lines.append(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        # ---------------------------------------------------------------------------
        # test_domain_rand.py — Domain randomization compliance tests.
        # Generated by gen_tests.py {TOOL_VERSION}
        # Source spec: {spec_path}
        # ---------------------------------------------------------------------------
        \"\"\"
        Verify domain randomization ranges and resample frequencies match spec.yaml.

        Each randomized parameter gets:
          - A range test: sample 1000 times, verify all samples within [lo, hi]
          - A resample frequency test: verify per_episode / per_step semantics

        HOW TO WIRE UP:
        ---------------
        1. Replace env.sample_param(name) with your actual parameter sampling call.
        2. Replace env.reset() / env.step() with your environment's API.
        \"\"\"

        from __future__ import annotations

        # TODO: import your environment here.
        # Example:
        #   from myproject.envs import DroneRacingEnv

        import pytest

    """))

    if not domain_rand:
        lines.append(textwrap.dedent("""\
            @pytest.mark.skip(reason="No domain_randomization entries in spec.yaml")
            def test_domain_rand_empty():
                \"\"\"spec.yaml: dynamics.domain_randomization — no entries\"\"\"
                raise NotImplementedError("No domain randomization")
        """))
    else:
        for idx, entry in enumerate(domain_rand):
            param = entry.get("parameter", f"param_{idx}")
            distribution = entry.get("distribution", UNRESOLVED)
            range_ = entry.get("range", [UNRESOLVED, UNRESOLVED])
            freq = entry.get("resample_frequency", UNRESOLVED)
            source = _source_ref(entry)
            safe = _safe_name(param)

            range_unresolved = (
                not isinstance(range_, (list, tuple))
                or len(range_) != 2
                or _is_unresolved(range_[0])
                or _is_unresolved(range_[1])
            )

            if range_unresolved:
                lines.append(textwrap.dedent(f"""\
                    @pytest.mark.skip(
                        reason="UNRESOLVED: dynamics.domain_randomization[{idx}] '{param}' range — {source}"
                    )
                    def test_domain_rand_{safe}_range():
                        \"\"\"spec.yaml: dynamics.domain_randomization[{idx}] '{param}' range — UNRESOLVED\"\"\"
                        raise NotImplementedError("UNRESOLVED range")

                """))
            else:
                lo, hi = range_[0], range_[1]
                lines.append(textwrap.dedent(f"""\
                    def test_domain_rand_{safe}_range(spec_dynamics):
                        \"\"\"
                        spec.yaml: dynamics.domain_randomization[{idx}] '{param}'
                        distribution={distribution!r} range=[{lo}, {hi}]
                        Source: {source}

                        Sample the parameter 1000 times and verify all samples stay
                        within the spec-declared range.
                        \"\"\"
                        lo, hi = {lo!r}, {hi!r}

                        # TODO: replace with actual parameter sampling from your env.
                        # Example:
                        #   env = DroneRacingEnv()
                        #   samples = []
                        #   for _ in range(1000):
                        #       env.reset()
                        #       samples.append(env.get_param("{param}"))
                        #
                        # For now: verify the spec declares the right bounds.
                        dr_entries = spec_dynamics.get("domain_randomization", [])
                        spec_entry = next(
                            (e for e in dr_entries if e.get("parameter") == "{param}"),
                            None,
                        )
                        assert spec_entry is not None, (
                            "Domain rand entry '{param}' not found in spec"
                        )
                        spec_range = spec_entry.get("range", [None, None])
                        assert spec_range[0] == pytest.approx({lo!r}), (
                            f"Range lo mismatch for '{param}': spec={{spec_range[0]!r}} expected {lo!r}"
                        )
                        assert spec_range[1] == pytest.approx({hi!r}), (
                            f"Range hi mismatch for '{param}': spec={{spec_range[1]!r}} expected {hi!r}"
                        )
                        # TODO: uncomment after wiring up env:
                        # outliers = [s for s in samples if not (lo <= s <= hi)]
                        # assert not outliers, (
                        #     f"{{len(outliers)}} samples outside [{lo!r}, {hi!r}] for '{param}': {{outliers[:5]}}"
                        # )

                """))

            if _is_unresolved(freq):
                lines.append(textwrap.dedent(f"""\
                    @pytest.mark.skip(
                        reason="UNRESOLVED: dynamics.domain_randomization[{idx}] '{param}' resample_frequency — {source}"
                    )
                    def test_domain_rand_{safe}_resample_frequency():
                        \"\"\"spec.yaml: dynamics.domain_randomization[{idx}] '{param}' resample_frequency — UNRESOLVED\"\"\"
                        raise NotImplementedError("UNRESOLVED resample_frequency")

                """))
            elif freq == "per_episode":
                lines.append(textwrap.dedent(f"""\
                    def test_domain_rand_{safe}_resample_per_episode(spec_dynamics):
                        \"\"\"
                        spec.yaml: dynamics.domain_randomization[{idx}] '{param}'
                        resample_frequency=per_episode — {source}

                        Verify the parameter stays constant within one episode but
                        MAY change across episode resets.
                        \"\"\"
                        # TODO: replace with actual env API.
                        # Example:
                        #   env = DroneRacingEnv()
                        #   env.reset()
                        #   v0 = env.get_param("{param}")
                        #   zero_action = [0.0] * env.action_space.shape[0]
                        #   for _ in range(50):
                        #       env.step(zero_action)
                        #   v1 = env.get_param("{param}")
                        #   assert v0 == v1, f"'{param}' changed mid-episode (spec: per_episode)"
                        dr_entries = spec_dynamics.get("domain_randomization", [])
                        spec_entry = next(
                            (e for e in dr_entries if e.get("parameter") == "{param}"),
                            None,
                        )
                        assert spec_entry is not None
                        assert spec_entry.get("resample_frequency") == "per_episode", (
                            "resample_frequency should be 'per_episode' per spec"
                        )

                """))
            elif freq == "per_step":
                lines.append(textwrap.dedent(f"""\
                    def test_domain_rand_{safe}_resample_per_step(spec_dynamics):
                        \"\"\"
                        spec.yaml: dynamics.domain_randomization[{idx}] '{param}'
                        resample_frequency=per_step — {source}

                        Verify the parameter is resampled at every environment step.
                        \"\"\"
                        # TODO: replace with actual env API.
                        # Example:
                        #   env = DroneRacingEnv()
                        #   env.reset()
                        #   zero_action = [0.0] * env.action_space.shape[0]
                        #   values = [env.get_param("{param}")]
                        #   for _ in range(10):
                        #       env.step(zero_action)
                        #       values.append(env.get_param("{param}"))
                        #   assert len(set(values)) > 1, "'{param}' never changed (spec: per_step)"
                        dr_entries = spec_dynamics.get("domain_randomization", [])
                        spec_entry = next(
                            (e for e in dr_entries if e.get("parameter") == "{param}"),
                            None,
                        )
                        assert spec_entry is not None
                        assert spec_entry.get("resample_frequency") == "per_step"

                """))
            else:
                lines.append(textwrap.dedent(f"""\
                    def test_domain_rand_{safe}_resample_frequency_{_safe_name(freq)}(spec_dynamics):
                        \"\"\"
                        spec.yaml: dynamics.domain_randomization[{idx}] '{param}'
                        resample_frequency={freq!r} — {source}
                        \"\"\"
                        dr_entries = spec_dynamics.get("domain_randomization", [])
                        spec_entry = next(
                            (e for e in dr_entries if e.get("parameter") == "{param}"),
                            None,
                        )
                        assert spec_entry is not None
                        assert spec_entry.get("resample_frequency") == {freq!r}, (
                            f"resample_frequency mismatch for '{param}'"
                        )

                """))

    return "\n".join(lines)


# ---------------------------------------------------------------------------


def _gen_test_timing(spec: dict, spec_path: str) -> str:
    """Generate test_timing.py content."""
    timing = spec.get("timing") or {}
    control_hz = timing.get("control_frequency_hz", UNRESOLVED)
    sensor_ms = timing.get("sensor_delay_ms", UNRESOLVED)
    action_ms = timing.get("action_delay_ms", UNRESOLVED)
    timestamping_model = timing.get("timestamping_model", UNRESOLVED)
    sim_dt = timing.get("sim_dt")
    source = _source_ref(timing)

    lines: list[str] = []
    lines.append(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        # ---------------------------------------------------------------------------
        # test_timing.py — Control-loop timing compliance tests.
        # Generated by gen_tests.py {TOOL_VERSION}
        # Source spec: {spec_path}
        # ---------------------------------------------------------------------------
        \"\"\"
        Verify timing parameters match spec.yaml.

        Tests cover:
          - control_frequency_hz (policy step rate)
          - sensor_delay_ms (observation pipeline latency)
          - action_delay_ms (actuator command latency)
          - timestamping_model (camera-anchored vs. control-loop-anchored)

        HOW TO WIRE UP:
        ---------------
        1. Replace env.get_control_dt() with your env's actual control period accessor.
        2. Replace env.get_observation_delay_steps() with your delay buffer size.
        3. Replace env.get_action_delay_steps() with your action delay buffer size.
        \"\"\"

        from __future__ import annotations

        # TODO: import your environment here.
        # Example:
        #   from myproject.envs import DroneRacingEnv

        import pytest

    """))

    # control_frequency_hz
    if _is_unresolved(control_hz):
        lines.append(textwrap.dedent(f"""\
            @pytest.mark.skip(reason="UNRESOLVED: timing.control_frequency_hz — {source}")
            def test_control_frequency_hz():
                \"\"\"spec.yaml: timing.control_frequency_hz — UNRESOLVED\"\"\"
                raise NotImplementedError("UNRESOLVED control_frequency_hz")

        """))
    else:
        expected_dt = 1.0 / float(control_hz)
        lines.append(textwrap.dedent(f"""\
            def test_control_frequency_hz(spec_timing):
                \"\"\"
                spec.yaml: timing.control_frequency_hz = {control_hz}
                (dt = {expected_dt:.6f} s)
                Source: {source}

                Verify the environment's control loop runs at the spec rate.
                \"\"\"
                spec_hz = spec_timing.get("control_frequency_hz")
                if spec_hz == "UNRESOLVED" or spec_hz is None:
                    pytest.skip("control_frequency_hz is UNRESOLVED")
                assert float(spec_hz) == pytest.approx({control_hz!r}, rel=1e-4), (
                    f"control_frequency_hz {{spec_hz}} != spec {control_hz!r}"
                )
                # TODO: verify your env control dt:
                # Example:
                #   env = DroneRacingEnv()
                #   dt = env.get_control_dt()
                #   assert dt == pytest.approx({expected_dt!r}, rel=1e-4), (
                #       f"Control dt {{dt}} != 1 / {control_hz!r} = {expected_dt!r}"
                #   )

        """))

    # sensor_delay_ms
    if _is_unresolved(sensor_ms):
        lines.append(textwrap.dedent(f"""\
            @pytest.mark.skip(reason="UNRESOLVED: timing.sensor_delay_ms — {source}")
            def test_sensor_delay_ms():
                \"\"\"spec.yaml: timing.sensor_delay_ms — UNRESOLVED\"\"\"
                raise NotImplementedError("UNRESOLVED sensor_delay_ms")

        """))
    else:
        delay_steps_expr = (
            f"int({sensor_ms!r} / (1000.0 / {control_hz!r}))"
            if not _is_unresolved(control_hz)
            else "None  # cannot compute without control_frequency_hz"
        )
        lines.append(textwrap.dedent(f"""\
            def test_sensor_delay_ms(spec_timing):
                \"\"\"
                spec.yaml: timing.sensor_delay_ms = {sensor_ms} ms
                Source: {source}

                Verify the observation pipeline introduces the spec-declared delay.
                \"\"\"
                spec_delay = spec_timing.get("sensor_delay_ms")
                if spec_delay == "UNRESOLVED" or spec_delay is None:
                    pytest.skip("sensor_delay_ms is UNRESOLVED")
                assert float(spec_delay) == pytest.approx({sensor_ms!r}, rel=1e-4), (
                    f"sensor_delay_ms {{spec_delay}} != spec {sensor_ms!r}"
                )
                # Derive expected delay steps (depends on control frequency).
                ctrl_hz = spec_timing.get("control_frequency_hz")
                if ctrl_hz and ctrl_hz != "UNRESOLVED":
                    expected_delay_steps = int({sensor_ms!r} / (1000.0 / float(ctrl_hz)))
                    _ = expected_delay_steps  # TODO: compare against env's delay buffer size.
                    # Example:
                    #   env = DroneRacingEnv()
                    #   assert env.get_observation_delay_steps() == expected_delay_steps

        """))

    # action_delay_ms
    if _is_unresolved(action_ms):
        lines.append(textwrap.dedent(f"""\
            @pytest.mark.skip(reason="UNRESOLVED: timing.action_delay_ms — {source}")
            def test_action_delay_ms():
                \"\"\"spec.yaml: timing.action_delay_ms — UNRESOLVED\"\"\"
                raise NotImplementedError("UNRESOLVED action_delay_ms")

        """))
    else:
        lines.append(textwrap.dedent(f"""\
            def test_action_delay_ms(spec_timing):
                \"\"\"
                spec.yaml: timing.action_delay_ms = {action_ms} ms
                Source: {source}

                Verify the actuator command pipeline introduces the spec-declared delay.
                \"\"\"
                spec_delay = spec_timing.get("action_delay_ms")
                if spec_delay == "UNRESOLVED" or spec_delay is None:
                    pytest.skip("action_delay_ms is UNRESOLVED")
                assert float(spec_delay) == pytest.approx({action_ms!r}, rel=1e-4), (
                    f"action_delay_ms {{spec_delay}} != spec {action_ms!r}"
                )
                ctrl_hz = spec_timing.get("control_frequency_hz")
                if ctrl_hz and ctrl_hz != "UNRESOLVED":
                    expected_delay_steps = int({action_ms!r} / (1000.0 / float(ctrl_hz)))
                    _ = expected_delay_steps
                    # TODO: compare against env's action delay buffer size.
                    # Example:
                    #   env = DroneRacingEnv()
                    #   assert env.get_action_delay_steps() == expected_delay_steps

        """))

    # timestamping_model
    if not _is_unresolved(timestamping_model) and timestamping_model:
        lines.append(textwrap.dedent(f"""\
            def test_timestamping_model(spec_timing):
                \"\"\"
                spec.yaml: timing.timestamping_model = {timestamping_model!r}
                Source: {source}

                Verify the timestamping convention used in the data pipeline.
                \"\"\"
                spec_model = spec_timing.get("timestamping_model")
                if spec_model == "UNRESOLVED" or spec_model is None:
                    pytest.skip("timestamping_model is UNRESOLVED")
                assert spec_model == {timestamping_model!r}, (
                    f"timestamping_model {{spec_model!r}} != spec {timestamping_model!r}"
                )
                # TODO: verify your env uses this timestamping convention.

        """))

    # sim_dt (optional field)
    if sim_dt is not None and not _is_unresolved(sim_dt):
        lines.append(textwrap.dedent(f"""\
            def test_sim_dt(spec_timing):
                \"\"\"
                spec.yaml: timing.sim_dt = {sim_dt}
                Source: {source}

                Verify the simulator timestep matches the spec (if sim_dt is specified).
                \"\"\"
                spec_sim_dt = spec_timing.get("sim_dt")
                if spec_sim_dt is None or spec_sim_dt == "UNRESOLVED":
                    pytest.skip("sim_dt is not declared in spec")
                assert float(spec_sim_dt) == pytest.approx({sim_dt!r}, rel=1e-6), (
                    f"sim_dt {{spec_sim_dt}} != spec {sim_dt!r}"
                )
                # TODO: verify your simulator dt:
                # Example:
                #   from myproject.simulation import SimConfig
                #   assert SimConfig.dt == pytest.approx({sim_dt!r})

        """))

    return "\n".join(lines)


# ---------------------------------------------------------------------------


def _gen_test_informed_pomdp(spec: dict, spec_path: str) -> str:
    """Generate test_informed_pomdp.py content."""
    spaces = spec.get("spaces") or {}
    obs_exec = spaces.get("observation_exec") or []
    info_train = spaces.get("information_train") or []

    obs_names = [f.get("name", "") for f in obs_exec if not _is_unresolved(f.get("name", ""))]
    info_names = [f.get("name", "") for f in info_train if not _is_unresolved(f.get("name", ""))]

    # Gather dreamer key patterns.
    dreamer_patterns = [
        (f.get("name", ""), f.get("informed_dreamer_key", ""))
        for f in info_train
        if f.get("informed_dreamer_key") and not _is_unresolved(f.get("informed_dreamer_key", ""))
    ]

    lines: list[str] = []
    lines.append(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        # ---------------------------------------------------------------------------
        # test_informed_pomdp.py — Informed-POMDP privileged info isolation tests.
        # Generated by gen_tests.py {TOOL_VERSION}
        # Source spec: {spec_path}
        # ---------------------------------------------------------------------------
        \"\"\"
        Verify the informed-POMDP split is correctly implemented.

        The core invariant:
          - information_train fields MUST NOT appear in execution observations.
          - information_train fields MUST be available as decoder targets during training.
          - decoder gating regex patterns must match their declared field names.

        These are the most critical structural tests for Informed Dreamer-style papers.

        HOW TO WIRE UP:
        ---------------
        1. Replace policy.get_observation_keys(mode="execution") with your policy API.
        2. Replace world_model.get_decoder_target_keys() with your world model API.
        3. The regex gating tests work against spec data alone (no extra wiring needed).
        \"\"\"

        from __future__ import annotations

        import re

        # TODO: import your policy and world model here.
        # Example:
        #   from myproject.policies import load_policy
        #   from myproject.world_model import WorldModel

        import pytest


        # ---------------------------------------------------------------------------
        # Data extracted at code-gen time (from spec.yaml)
        # ---------------------------------------------------------------------------

        _EXEC_FIELD_NAMES: list[str] = {obs_names!r}
        _TRAIN_FIELD_NAMES: list[str] = {info_names!r}

    """))

    if not info_names:
        lines.append(textwrap.dedent("""\
            @pytest.mark.skip(
                reason="No information_train fields in spec.yaml — informed-POMDP not applicable"
            )
            def test_informed_pomdp_not_applicable():
                \"\"\"spec.yaml: spaces.information_train — empty\"\"\"
                raise NotImplementedError("No privileged fields")
        """))
    else:
        lines.append(textwrap.dedent(f"""\
            def test_privileged_fields_excluded_from_exec_obs(
                information_train_names: list[str],
                spec_spaces: dict,
            ):
                \"\"\"
                spec.yaml: spaces.information_train fields must NOT appear in
                spaces.observation_exec.

                POMDP split invariant: privileged training info must be invisible
                to the deployed policy.
                \"\"\"
                exec_names = set(
                    f.get("name", "")
                    for f in spec_spaces.get("observation_exec", [])
                    if f.get("name") and f.get("name") != "UNRESOLVED"
                )
                train_names = set(information_train_names)
                overlap = exec_names & train_names
                assert len(overlap) == 0, (
                    f"Privileged fields appear in both observation_exec and "
                    f"information_train (POMDP split violation): {{sorted(overlap)}}"
                )
                # TODO: also check at runtime that the deployed policy never receives
                # privileged keys.  Example:
                #   policy = load_policy()
                #   exec_keys = set(policy.get_observation_keys(mode="execution"))
                #   assert not (exec_keys & train_names), (
                #       f"Privileged keys leaked to execution: {{exec_keys & train_names}}"
                #   )

            def test_privileged_fields_available_in_training_decoder(
                information_train_names: list[str],
                spec_spaces: dict,
            ):
                \"\"\"
                spec.yaml: spaces.information_train fields MUST be available as
                decoder targets during world-model training.

                Informed Dreamer uses these fields as auxiliary prediction targets
                to provide learning signal about privileged state.
                \"\"\"
                train_names = set(information_train_names)
                # TODO: verify against your world model's decoder target keys.
                # Example:
                #   world_model = WorldModel.load(checkpoint_path)
                #   decoder_targets = set(world_model.get_decoder_target_keys())
                #   missing = train_names - decoder_targets
                #   assert len(missing) == 0, (
                #       f"Privileged fields missing from decoder targets: {{missing}}"
                #   )
                #
                # For now: verify spec declares these fields as training-only.
                info_train = spec_spaces.get("information_train") or []
                spec_train_names = {{
                    f.get("name", "")
                    for f in info_train
                    if f.get("name") and f.get("name") != "UNRESOLVED"
                       and f.get("training_only", True)
                }}
                assert spec_train_names == train_names, (
                    f"Mismatch in training-only fields: spec={{spec_train_names}} expected={{train_names}}"
                )

        """))

        if dreamer_patterns:
            lines.append(textwrap.dedent("""\

            """))
            for field_name, pattern in dreamer_patterns:
                safe = _safe_name(field_name)
                lines.append(textwrap.dedent(f"""\
                    def test_dreamer_key_pattern_{safe}(spec_spaces):
                        \"\"\"
                        spec.yaml: spaces.information_train field '{field_name}'
                        informed_dreamer_key = {pattern!r}

                        The decoder gating regex must match the field name it gates.
                        If it doesn't match its own name, the decoder will silently
                        fail to produce outputs for that field.
                        \"\"\"
                        pattern = {pattern!r}
                        field_name = "{field_name}"
                        assert re.search(pattern, field_name), (
                            f"Field '{{field_name}}' does not match its own "
                            f"dreamer_key pattern {{pattern!r}}.  "
                            "Check the regex in spec.yaml."
                        )
                        # TODO: verify your world-model decoder uses this regex.
                        # Example:
                        #   world_model = WorldModel.load(checkpoint_path)
                        #   gating_patterns = world_model.get_decoder_gating_patterns()
                        #   assert any(
                        #       re.fullmatch(p, "{field_name}") for p in gating_patterns
                        #   ), f"'{field_name}' not gated in decoder"

                """))

    return "\n".join(lines)


# ---------------------------------------------------------------------------


def _gen_test_unresolved(spec: dict, spec_path: str) -> str:
    """Generate test_unresolved.py content."""
    all_unresolved = _collect_unresolved_paths(spec, "")

    lines: list[str] = []
    lines.append(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        # ---------------------------------------------------------------------------
        # test_unresolved.py — Stub tests for all UNRESOLVED spec fields.
        # Generated by gen_tests.py {TOOL_VERSION}
        # Source spec: {spec_path}
        # ---------------------------------------------------------------------------
        \"\"\"
        Every field marked UNRESOLVED in spec.yaml generates one skipped stub test.

        Purpose
        -------
        - These tests are VISIBLE in CI output (skipped, not hidden).
        - They serve as a TODO list for manual paper extraction.
        - Configure --strict-markers in pytest.ini to fail CI until all are resolved.

        Workflow
        --------
        1. Resolve each UNRESOLVED field in spec.yaml.
        2. Re-run gen_tests.py to regenerate this file.
        3. The corresponding stub will disappear (replaced by a real test elsewhere).

        Total UNRESOLVED fields detected: {len(all_unresolved)}
        \"\"\"

        from __future__ import annotations

        import pytest

    """))

    if not all_unresolved:
        lines.append(textwrap.dedent("""\
            def test_no_unresolved_fields_remain():
                \"\"\"
                spec.yaml: no UNRESOLVED fields detected.

                All fields have been extracted from the paper.
                This test passes to confirm a clean spec.
                \"\"\"
                assert True, "All spec fields are resolved — great work!"
        """))
    else:
        for path, _ in all_unresolved:
            safe = _safe_name(path.replace("[", "_").replace("]", ""))
            # Derive section from path prefix.
            section = path.split(".")[0] if "." in path else path.split("[")[0]

            # Try to find a source reference for this path by navigating the spec.
            source_ref = "see spec.yaml for paper reference"
            # Walk the spec to find any nearby source field.
            parts = path.replace("[", ".[").split(".")
            node = spec
            for part in parts[:-1]:
                if not isinstance(node, (dict, list)):
                    break
                if part.startswith("[") and isinstance(node, list):
                    try:
                        idx = int(part[1:-1])
                        node = node[idx]
                    except (ValueError, IndexError):
                        break
                elif isinstance(node, dict):
                    node = node.get(part, {})
                else:
                    break
            if isinstance(node, dict) and node.get("source") and not _is_unresolved(node.get("source")):
                source_ref = node["source"]

            lines.append(textwrap.dedent(f"""\
                @pytest.mark.skip(
                    reason=(
                        "UNRESOLVED: {path} — "
                        "manual paper extraction required.  "
                        "Source hint: {source_ref}"
                    )
                )
                def test_UNRESOLVED_{safe}():
                    \"\"\"
                    spec.yaml: {path}

                    This field was not extracted from the paper.
                    Resolve it by updating spec.yaml and re-running gen_tests.py.

                    Section: {section}
                    Source hint: {source_ref}
                    \"\"\"
                    raise NotImplementedError(
                        "UNRESOLVED field: {path}\\n"
                        "Update spec.yaml and re-run gen_tests.py."
                    )

            """))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def generate_all(spec_path: Path, output_dir: Path) -> None:
    """Load spec.yaml and write all test files to output_dir."""
    if not spec_path.exists():
        print(f"ERROR: spec.yaml not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    with spec_path.open("r", encoding="utf-8") as fh:
        spec: dict = yaml.safe_load(fh)

    if not isinstance(spec, dict):
        print("ERROR: spec.yaml must be a YAML mapping at the top level.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure there's an __init__.py so pytest collects properly.
    init_py = output_dir / "__init__.py"
    if not init_py.exists():
        init_py.write_text(
            "# Auto-generated by gen_tests.py — do not edit manually.\n",
            encoding="utf-8",
        )

    spec_path_str = str(spec_path.resolve())

    files: dict[str, str] = {
        "conftest.py":          _gen_conftest(spec, spec_path_str),
        "test_spaces.py":       _gen_test_spaces(spec, spec_path_str),
        "test_reward.py":       _gen_test_reward(spec, spec_path_str),
        "test_termination.py":  _gen_test_termination(spec, spec_path_str),
        "test_domain_rand.py":  _gen_test_domain_rand(spec, spec_path_str),
        "test_timing.py":       _gen_test_timing(spec, spec_path_str),
        "test_informed_pomdp.py": _gen_test_informed_pomdp(spec, spec_path_str),
        "test_unresolved.py":   _gen_test_unresolved(spec, spec_path_str),
    }

    counts = {
        "resolved": 0,
        "unresolved_stubs": 0,
        "skipped_stubs": 0,
    }

    for filename, content in files.items():
        out_path = output_dir / filename
        out_path.write_text(content, encoding="utf-8")
        size = out_path.stat().st_size
        n_tests = content.count("\n    def test_")
        n_skips = content.count("@pytest.mark.skip")
        counts["resolved"] += n_tests - n_skips
        counts["skipped_stubs"] += n_skips
        print(
            f"  wrote {filename:<28} "
            f"({size:>6} bytes, {n_tests:>3} tests, {n_skips:>3} skipped)",
            file=sys.stderr,
        )

    total_unresolved = len(_collect_unresolved_paths(spec, ""))
    print("", file=sys.stderr)
    print(f"Output directory:     {output_dir.resolve()}", file=sys.stderr)
    print(f"Files written:        {len(files)}", file=sys.stderr)
    print(f"UNRESOLVED fields:    {total_unresolved}", file=sys.stderr)
    print(f"Skipped stubs:        {counts['skipped_stubs']}", file=sys.stderr)
    print("", file=sys.stderr)
    if total_unresolved > 0:
        print(
            f"NOTE: {total_unresolved} fields are UNRESOLVED.  "
            "Resolve them in spec.yaml and re-run gen_tests.py.",
            file=sys.stderr,
        )
    print(
        "TODO: Wire up project-specific env/policy/model imports in the generated files.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gen_tests.py",
        description=(
            "Paper drift detector: generate pytest compliance tests from spec.yaml.\n\n"
            "Reads a compiled spec.yaml and emits a suite of test files covering\n"
            "spaces, reward, termination, domain randomization, timing, informed-POMDP,\n"
            "and UNRESOLVED field stubs.  Generated tests are ~80%% complete templates;\n"
            "the remaining 20%% is project-specific wiring marked with TODO comments."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
              # Basic usage:
              python gen_tests.py --spec spec.yaml

              # Custom output directory:
              python gen_tests.py --spec spec.yaml --output-dir src/tests/compliance/

              # Via environment variable (useful in CI):
              SPEC_YAML_PATH=artifacts/spec.yaml python gen_tests.py

              # Run generated tests:
              pytest tests/spec_compliance/ -v

              # Run with strict skips (fail CI on any UNRESOLVED stub):
              pytest tests/spec_compliance/ -v --strict-markers -m "not skip"

            Environment Variables
            ---------------------
              SPEC_YAML_PATH   Path to spec.yaml (used by generated conftest.py
                               at runtime, not by gen_tests.py itself).
        """),
    )
    p.add_argument(
        "--spec",
        metavar="PATH",
        default=os.environ.get("SPEC_YAML_PATH", "spec.yaml"),
        help=(
            "Path to spec.yaml (default: $SPEC_YAML_PATH or 'spec.yaml')."
        ),
    )
    p.add_argument(
        "--output-dir",
        metavar="DIR",
        default="tests/spec_compliance/",
        help=(
            "Directory to write generated test files into "
            "(default: 'tests/spec_compliance/').  Created if it does not exist."
        ),
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"gen_tests.py {TOOL_VERSION}",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    spec_path = Path(args.spec).resolve()
    output_dir = Path(args.output_dir).resolve()

    print(f"gen_tests.py {TOOL_VERSION}", file=sys.stderr)
    print(f"  spec:       {spec_path}", file=sys.stderr)
    print(f"  output-dir: {output_dir}", file=sys.stderr)
    print("", file=sys.stderr)

    generate_all(spec_path, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
