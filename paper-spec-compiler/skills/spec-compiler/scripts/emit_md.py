#!/usr/bin/env python3
"""
emit_md.py — Markdown spec emitter with full traceability links.

Converts an IR JSON file or a spec.yaml file into a human-readable Markdown
document.  Every extracted value is annotated with its paper source location
(e.g. "[§3.2]", "[Table 1]", "[Eq. 4]").  UNRESOLVED fields are highlighted
with warning markers and TODO blocks.  The document closes with a
Reproducibility Appendix and an UNRESOLVED Items summary.

Design goals
------------
* Traceability: every value cites the paper location it came from.
* UNRESOLVED visibility: no silent gaps — every missing value is flagged.
* Reproducibility appendix: auto-generated from training/evaluation/deployment
  sections to satisfy NeurIPS / RL-paper standards.
* Single-source: accepts either --ir (raw IR JSON) or --spec (spec.yaml);
  internally normalises both to the same dict structure.

Usage
-----
    # From IR JSON (produced by tex_parser.py or ir_schema serialisation):
    python emit_md.py --ir ir_output.json --output spec.md

    # From already-emitted spec.yaml:
    python emit_md.py --spec spec.yaml --output spec.md
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# UNRESOLVED sentinel and priority classification
# ---------------------------------------------------------------------------

UNRESOLVED = "UNRESOLVED"

_HIGH_PRIORITY_SECTIONS = {"reward", "termination", "dynamics", "spaces"}
_MEDIUM_PRIORITY_SECTIONS = {"training", "actor_critic", "world_model", "timing"}


def _unresolved_priority(section: str) -> str:
    if section in _HIGH_PRIORITY_SECTIONS:
        return "HIGH"
    if section in _MEDIUM_PRIORITY_SECTIONS:
        return "MEDIUM"
    return "LOW"


def _is_unresolved(value: Any) -> bool:
    return isinstance(value, str) and value == UNRESOLVED


# ---------------------------------------------------------------------------
# Source citation formatter
# ---------------------------------------------------------------------------


def _cite(source: Any) -> str:
    """
    Convert a source value to a bracketed citation string.

    Accepts:
    - A dict (SourceTrace from ir_schema) with fields: section, table,
      equation, figure, page, tex_file, line_range.
    - A string (already-formatted citation from spec.yaml, e.g. "§3.2, Table 1").
    - None / missing → returns empty string.
    """
    if not source:
        return ""

    if isinstance(source, str):
        if source == UNRESOLVED or not source.strip():
            return ""
        return f" [{source}]"

    if isinstance(source, dict):
        parts: list[str] = []
        if source.get("section"):
            parts.append(source["section"])
        if source.get("table"):
            parts.append(source["table"])
        if source.get("equation"):
            parts.append(source["equation"])
        if source.get("figure"):
            parts.append(source["figure"])
        if source.get("page"):
            parts.append(f"p.{source['page']}")
        if not parts and source.get("tex_file"):
            lr = source.get("line_range")
            if lr:
                parts.append(f"{source['tex_file']}:{lr[0]}-{lr[1]}")
            else:
                parts.append(source["tex_file"])
        return f" [{', '.join(parts)}]" if parts else ""

    return ""


# ---------------------------------------------------------------------------
# Value formatter — resolves or shows UNRESOLVED
# ---------------------------------------------------------------------------


def _fmt(value: Any, source: Any = None) -> str:
    """
    Format a leaf value for Markdown inline display.

    If the value is UNRESOLVED, emits a highlighted warning span.
    Otherwise, formats the value and appends the citation.
    """
    if _is_unresolved(value):
        return "**`UNRESOLVED`** ⚠️"
    if value is None:
        return "_null_"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`{_cite(source)}"
    if isinstance(value, float):
        # Show floats cleanly: avoid scientific notation for small epsilons.
        s = repr(value)
        return f"`{s}`{_cite(source)}"
    if isinstance(value, (int, str)):
        return f"`{value}`{_cite(source)}"
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_fmt(v) for v in value)
        return f"[{inner}]{_cite(source)}"
    return f"`{value!r}`{_cite(source)}"


def _fmt_inline(value: Any) -> str:
    """Format a value without appended citation (for use in table cells)."""
    if _is_unresolved(value):
        return "⚠️ `UNRESOLVED`"
    if value is None:
        return "_null_"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, float):
        return f"`{repr(value)}`"
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_fmt_inline(v) for v in value)
        return f"[{inner}]"
    return f"`{value}`"


# ---------------------------------------------------------------------------
# UNRESOLVED collector
# ---------------------------------------------------------------------------


def _collect_unresolved(
    value: Any, path: str, results: list[dict]
) -> None:
    """
    Recursively collect UNRESOLVED leaf values.

    Appends dicts with keys: path, section, priority.
    """
    if isinstance(value, str) and value == UNRESOLVED:
        section = path.split(".")[0] if path else "unknown"
        results.append(
            {
                "path": path,
                "section": section,
                "priority": _unresolved_priority(section),
            }
        )
    elif isinstance(value, dict):
        for k, v in value.items():
            child = f"{path}.{k}" if path else k
            _collect_unresolved(v, child, results)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            child = f"{path}[{i}]"
            _collect_unresolved(item, child, results)


# ---------------------------------------------------------------------------
# Markdown table builder
# ---------------------------------------------------------------------------


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown pipe-table from header strings and row strings."""
    if not rows:
        return ""
    # Compute column widths.
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))

    def _pad(text: str, width: int) -> str:
        return text + " " * (width - len(text))

    header_line = "| " + " | ".join(_pad(h, col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = []
    for row in rows:
        padded = []
        for i, cell in enumerate(row):
            w = col_widths[i] if i < len(col_widths) else len(cell)
            padded.append(_pad(cell, w))
        data_lines.append("| " + " | ".join(padded) + " |")

    return "\n".join([header_line, sep_line] + data_lines)


# ---------------------------------------------------------------------------
# TODO block generator
# ---------------------------------------------------------------------------


def _todo_block(field_path: str, description: str, source_hint: str = "") -> str:
    """Generate a fenced TODO block for an UNRESOLVED field."""
    hint = f" [{source_hint}]" if source_hint else ""
    return (
        f"> **TODO (spec):** `{field_path}` is UNRESOLVED{hint}.\n"
        f"> {description}\n"
        f"> Manual extraction required before this spec can be finalised."
    )


# ---------------------------------------------------------------------------
# AST pretty-printer
# ---------------------------------------------------------------------------


def _pp_ast(node: Any, indent: int = 0) -> str:
    """Pretty-print an expression or boolean AST node as indented text."""
    pad = "  " * indent
    if _is_unresolved(node):
        return f"{pad}UNRESOLVED ⚠️"
    if node is None:
        return f"{pad}null"
    if not isinstance(node, dict):
        return f"{pad}{node!r}"

    ntype = node.get("type", "?")

    if ntype == "literal":
        return f"{pad}literal({node.get('value')})"

    if ntype == "field":
        space = node.get("space", "")
        name = node.get("name", "?")
        return f"{pad}field({name}" + (f", space={space}" if space else "") + ")"

    if ntype == "op":
        op = node.get("op", "?")
        args = node.get("args", [])
        params = node.get("params", {})
        lines = [f"{pad}op({op})"]
        for arg in args:
            lines.append(_pp_ast(arg, indent + 1))
        if params:
            lines.append(f"{pad}  params={params}")
        return "\n".join(lines)

    if ntype == "func":
        func_name = node.get("func_name", node.get("name", "?"))
        args = node.get("args", [])
        lines = [f"{pad}func({func_name})"]
        for arg in args:
            lines.append(_pp_ast(arg, indent + 1))
        return "\n".join(lines)

    if ntype == "compare":
        op = node.get("op", "?")
        left = node.get("left")
        right = node.get("right")
        lines = [f"{pad}compare({op})"]
        if left:
            lines.append(_pp_ast(left, indent + 1))
        if right:
            lines.append(_pp_ast(right, indent + 1))
        return "\n".join(lines)

    if ntype == "logic":
        logic_op = node.get("logic_op", "?")
        children = node.get("children", [])
        lines = [f"{pad}logic({logic_op})"]
        for child in children:
            lines.append(_pp_ast(child, indent + 1))
        return "\n".join(lines)

    # Generic fallback: dump as code.
    return f"{pad}{json.dumps(node, indent=2)}"


# ---------------------------------------------------------------------------
# Section renderers (one per spec.yaml section)
# ---------------------------------------------------------------------------


class _DocBuilder:
    """Accumulates markdown sections and renders them at the end."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._unresolved_items: list[dict] = []
        self._toc_entries: list[tuple[str, str]] = []  # (anchor, title)

    def _anchor(self, title: str) -> str:
        """GitHub-compatible anchor from a section title."""
        return title.lower().replace(" ", "-").replace("/", "").replace("(", "").replace(")", "")

    def section(self, title: str, level: int = 2) -> None:
        """Emit a Markdown heading and register it in the TOC."""
        prefix = "#" * level
        anchor = self._anchor(title)
        if level <= 3:
            self._toc_entries.append((anchor, title, level))
        self._parts.append(f"\n{prefix} {title}\n")

    def p(self, text: str) -> None:
        """Emit a paragraph."""
        self._parts.append(f"\n{text}\n")

    def kv(self, key: str, value: Any, source: Any = None, indent: int = 0) -> None:
        """Emit a key-value row as a bullet."""
        pad = "  " * indent
        self._parts.append(f"{pad}- **{key}**: {_fmt(value, source)}")

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        """Emit a Markdown table."""
        t = _md_table(headers, rows)
        if t:
            self._parts.append(f"\n{t}\n")

    def code(self, text: str, lang: str = "") -> None:
        """Emit a fenced code block."""
        self._parts.append(f"\n```{lang}\n{text}\n```\n")

    def todo(self, field_path: str, description: str, source_hint: str = "") -> None:
        """Emit a TODO block for an UNRESOLVED field."""
        self._parts.append(f"\n{_todo_block(field_path, description, source_hint)}\n")

    def hr(self) -> None:
        self._parts.append("\n---\n")

    def register_unresolved(self, items: list[dict]) -> None:
        self._unresolved_items.extend(items)

    def render_toc(self) -> str:
        """Render the table of contents."""
        lines = ["## Table of Contents\n"]
        for entry in self._toc_entries:
            anchor, title, level = entry
            indent = "  " * (level - 2)
            lines.append(f"{indent}- [{title}](#{anchor})")
        return "\n".join(lines)

    def render(self) -> str:
        """Combine TOC placeholder and all parts."""
        body = "\n".join(self._parts)
        toc = self.render_toc()
        return toc + "\n" + body


# ---------------------------------------------------------------------------
# Individual section renderers
# ---------------------------------------------------------------------------


def _render_meta(doc: _DocBuilder, spec: dict) -> None:
    """Render the meta / paper info section."""
    meta = spec.get("meta", {})
    paper = meta.get("paper", {})
    sources = meta.get("sources", {})

    doc.section("Paper Metadata", level=2)

    title = paper.get("title", UNRESOLVED)
    if not _is_unresolved(title):
        doc.p(f"**{title}**")
    else:
        doc.p("**Title:** ⚠️ `UNRESOLVED`")

    authors = paper.get("authors", [])
    if authors:
        doc.p("**Authors:** " + ", ".join(str(a) for a in authors))

    doc.section("Bibliographic Details", level=3)
    doc.kv("arXiv ID", paper.get("arxiv", UNRESOLVED))
    doc.kv("Version", paper.get("version", UNRESOLVED))
    doc.kv("DOI", paper.get("doi") or "_not specified_")
    doc.section("Source Provenance", level=3)
    doc.kv("Source priority", sources.get("prefer", ["arxiv_tex", "pdf", "html"]))
    doc.kv("LaTeX tarball SHA-256", sources.get("tex_hash") or "_not computed_")
    doc.kv("PDF SHA-256", sources.get("pdf_hash") or "_not computed_")
    doc.kv("Generated at", meta.get("generated_at", UNRESOLVED))
    doc.kv("Tool version", meta.get("tool_version", UNRESOLVED))
    unresolved_count = meta.get("unresolved_count", 0)
    doc.p(
        f"> **Completeness:** {unresolved_count} field(s) are `UNRESOLVED`. "
        "See the [UNRESOLVED Items](#unresolved-items) section for details."
    )


def _render_frames(doc: _DocBuilder, spec: dict) -> None:
    """Render the coordinate frames section."""
    frames = spec.get("frames")
    if not frames:
        return

    doc.section("Coordinate Frames", level=2)
    src = frames.get("source", "")
    doc.kv("Convention", frames.get("convention", UNRESOLVED), src)
    named_frames = frames.get("frames", [])
    if named_frames:
        doc.p("**Named frames:** " + ", ".join(f"`{f}`" for f in named_frames))

    transforms = frames.get("transforms", [])
    if transforms:
        doc.section("Frame Transforms", level=3)
        rows = []
        for t in transforms:
            rows.append(
                [
                    f"`{t.get('from', UNRESOLVED)}`",
                    f"`{t.get('to', UNRESOLVED)}`",
                    _fmt_inline(t.get("type", UNRESOLVED)),
                    t.get("description", UNRESOLVED),
                ]
            )
        doc.table(["From", "To", "Type", "Description"], rows)


def _render_spaces(doc: _DocBuilder, spec: dict) -> None:
    """Render the state/observation/action spaces section."""
    spaces = spec.get("spaces")
    if not spaces:
        return

    doc.section("Spaces", level=2)
    doc.p(
        "The informed-POMDP split separates fields available at deployment "
        "(`observation_exec`) from privileged training-time fields (`information_train`)."
    )

    def _space_table(fields: list[dict], section_path: str) -> None:
        if not fields:
            doc.p("_None defined._")
            return
        rows = []
        for f in fields:
            name = f.get("name", UNRESOLVED)
            dtype = f.get("dtype", UNRESOLVED)
            shape = f.get("shape", [UNRESOLVED])
            units = f.get("units") or "—"
            source = f.get("source", "")
            informed_key = f.get("informed_dreamer_key", "")

            name_cell = _fmt_inline(name)
            dtype_cell = _fmt_inline(dtype)
            shape_str = "[" + ", ".join(str(d) for d in shape) + "]"
            units_cell = _fmt_inline(units)
            src_cell = f"[{source}]" if source and source != UNRESOLVED else "—"
            if informed_key:
                name_cell += f" (key: `{informed_key}`)"

            rows.append([name_cell, dtype_cell, shape_str, units_cell, src_cell])
        doc.table(["Name", "dtype", "Shape", "Units", "Source"], rows)

    doc.section("Observation Space (exec)", level=3)
    _space_table(spaces.get("observation_exec", []), "spaces.observation_exec")

    doc.section("Privileged Information (train only)", level=3)
    _space_table(spaces.get("information_train", []), "spaces.information_train")

    doc.section("Action Space", level=3)
    action = spaces.get("action")
    if _is_unresolved(action) or not action:
        doc.p("⚠️ **Action space is `UNRESOLVED`.**")
        doc.todo("spaces.action", "Identify the action space from the paper's problem formulation.")
    else:
        src = action.get("source", "")
        doc.kv("Name", action.get("name", UNRESOLVED), src)
        doc.kv("dtype", action.get("dtype", UNRESOLVED))
        doc.kv("Shape", action.get("shape", UNRESOLVED))
        doc.kv("Bounds", action.get("bounds", UNRESOLVED))
        doc.kv("Semantics", action.get("semantics", UNRESOLVED))

    state = spaces.get("state", [])
    if state:
        doc.section("Full State Space", level=3)
        _space_table(state, "spaces.state")

    # Collect UNRESOLVED items.
    ur: list[dict] = []
    _collect_unresolved(spaces, "spaces", ur)
    doc.register_unresolved(ur)


def _render_timing(doc: _DocBuilder, spec: dict) -> None:
    """Render the timing / control loop section."""
    timing = spec.get("timing")
    if not timing:
        return

    doc.section("Timing", level=2)
    src = timing.get("source", "")
    doc.kv("Control frequency", timing.get("control_frequency_hz", UNRESOLVED), src, 0)
    doc.kv("Sensor delay", timing.get("sensor_delay_ms", UNRESOLVED), src)
    doc.kv("Action delay", timing.get("action_delay_ms", UNRESOLVED), src)
    if timing.get("timestamping_model"):
        doc.kv("Timestamping model", timing["timestamping_model"], src)
    if timing.get("sim_dt") is not None:
        doc.kv("Simulation dt", timing["sim_dt"], src)
    if timing.get("policy_dt") is not None:
        doc.kv("Policy dt", timing["policy_dt"], src)

    ur: list[dict] = []
    _collect_unresolved(timing, "timing", ur)
    doc.register_unresolved(ur)


def _render_reward(doc: _DocBuilder, spec: dict) -> None:
    """Render the reward function section."""
    reward = spec.get("reward")
    if not reward:
        return

    doc.section("Reward Function", level=2)
    src = reward.get("source", "")
    doc.kv("Discount factor (γ)", reward.get("discount", UNRESOLVED), src)
    doc.kv("Normalization", reward.get("normalization", UNRESOLVED))

    terms = reward.get("terms", [])
    if terms:
        doc.section("Reward Terms", level=3)
        for term in terms:
            tname = term.get("name", UNRESOLVED)
            tsrc = term.get("source", "")
            doc.p(f"#### `{tname}`{_cite(tsrc)}")
            doc.kv("Weight", term.get("weight", UNRESOLVED))
            if term.get("clamp") is not None:
                doc.kv("Clamp", term["clamp"])
            if term.get("zeroing_window"):
                doc.kv("Zeroing window", term["zeroing_window"])
            expr = term.get("expression_ast")
            if _is_unresolved(expr):
                doc.p("**Expression:** ⚠️ `UNRESOLVED`")
                doc.todo(
                    f"reward.terms[{tname}].expression_ast",
                    "Locate and parse the reward expression for this term.",
                    str(tsrc) if tsrc else "",
                )
            else:
                doc.p("**Expression AST:**")
                doc.code(_pp_ast(expr), lang="")

    ur: list[dict] = []
    _collect_unresolved(reward, "reward", ur)
    doc.register_unresolved(ur)


def _render_termination(doc: _DocBuilder, spec: dict) -> None:
    """Render the episode termination section."""
    term = spec.get("termination")
    if not term:
        return

    doc.section("Termination Conditions", level=2)
    src = term.get("source", "")
    max_steps = term.get("max_episode_steps", UNRESOLVED)
    doc.kv("Max episode steps", max_steps, src)

    conditions = term.get("conditions", [])
    if conditions:
        doc.section("Termination Conditions", level=3)
        for cond in conditions:
            cname = cond.get("name", UNRESOLVED)
            csrc = cond.get("source", "")
            doc.p(f"#### `{cname}`{_cite(csrc)}")
            ast = cond.get("condition_ast")
            if _is_unresolved(ast):
                doc.p("**Condition:** ⚠️ `UNRESOLVED`")
                doc.todo(
                    f"termination.conditions[{cname}].condition_ast",
                    "Parse the boolean termination condition.",
                    str(csrc) if csrc else "",
                )
            else:
                doc.p("**Condition AST:**")
                doc.code(_pp_ast(ast), lang="")

    ur: list[dict] = []
    _collect_unresolved(term, "termination", ur)
    doc.register_unresolved(ur)


def _render_gates_track(doc: _DocBuilder, spec: dict) -> None:
    """Render the domain-specific gates/track section."""
    gt = spec.get("gates_track")
    if not gt:
        return

    doc.section("Gates / Track (Domain-Specific)", level=2)
    doc.kv("Number of gates", gt.get("num_gates", UNRESOLVED))

    gg = gt.get("gate_geometry")
    if not _is_unresolved(gg) and isinstance(gg, dict):
        doc.section("Gate Geometry", level=3)
        src = gg.get("source", "")
        doc.kv("Shape", gg.get("shape", UNRESOLVED), src)
        doc.kv("Virtual thickness", gg.get("virtual_thickness", UNRESOLVED), src)
        dims = gg.get("dimensions", {})
        if dims:
            for k, v in dims.items():
                doc.kv(k, v)

    ppo = gt.get("pre_post_offsets")
    if not _is_unresolved(ppo) and isinstance(ppo, dict):
        doc.section("Waypoint Offsets", level=3)
        src = ppo.get("source", "")
        doc.kv("Pre-gate offset", ppo.get("pre_gate_offset", UNRESOLVED), src)
        doc.kv("Post-gate offset", ppo.get("post_gate_offset", UNRESOLVED), src)

    pc = gt.get("pass_condition")
    if not _is_unresolved(pc) and isinstance(pc, dict):
        doc.section("Pass Condition", level=3)
        ast = pc.get("condition_ast")
        if _is_unresolved(ast):
            doc.p("⚠️ `UNRESOLVED`")
        else:
            doc.code(_pp_ast(ast), lang="")

    ur: list[dict] = []
    _collect_unresolved(gt, "gates_track", ur)
    doc.register_unresolved(ur)


def _render_dynamics(doc: _DocBuilder, spec: dict) -> None:
    """Render the dynamics model section."""
    dyn = spec.get("dynamics")
    if not dyn:
        return

    doc.section("Dynamics", level=2)
    src = dyn.get("source", "")
    doc.kv("Model type", dyn.get("model_type", UNRESOLVED), src)

    integrator = dyn.get("integrator")
    if not _is_unresolved(integrator) and isinstance(integrator, dict):
        doc.section("Integrator", level=3)
        isrc = integrator.get("source", "")
        doc.kv("Type", integrator.get("type", UNRESOLVED), isrc)
        doc.kv("Timestep (dt)", integrator.get("dt", UNRESOLVED), isrc)

    equations = dyn.get("equations", [])
    if equations:
        doc.section("Equations", level=3)
        for eq in equations:
            ename = eq.get("name", UNRESOLVED)
            esrc = eq.get("source", "")
            doc.p(f"#### `{ename}`{_cite(esrc)}")
            latex = eq.get("latex", UNRESOLVED)
            if not _is_unresolved(latex):
                doc.code(latex, lang="latex")
            ast = eq.get("expression_ast")
            if not _is_unresolved(ast) and ast is not None:
                doc.p("**Parsed AST:**")
                doc.code(_pp_ast(ast), lang="")
            elif _is_unresolved(ast):
                doc.p("**AST:** ⚠️ `UNRESOLVED` — expression too complex for automatic parsing.")

    params = dyn.get("parameters", [])
    if params:
        doc.section("Physical Parameters", level=3)
        rows = []
        for p in params:
            name = _fmt_inline(p.get("name", UNRESOLVED))
            sym = _fmt_inline(p.get("symbol", UNRESOLVED))
            val = _fmt_inline(p.get("default_value", UNRESOLVED))
            units = _fmt_inline(p.get("units", UNRESOLVED))
            psrc = p.get("source", "")
            src_cell = f"[{psrc}]" if psrc and psrc != UNRESOLVED else "—"
            rows.append([name, sym, val, units, src_cell])
        doc.table(["Parameter", "Symbol", "Default", "Units", "Source"], rows)

    dr = dyn.get("domain_randomization", [])
    if dr:
        doc.section("Domain Randomization", level=3)
        rows = []
        for entry in dr:
            param = _fmt_inline(entry.get("parameter", UNRESOLVED))
            dist = _fmt_inline(entry.get("distribution", UNRESOLVED))
            rng = entry.get("range", [UNRESOLVED, UNRESOLVED])
            rng_str = (
                f"[{_fmt_inline(rng[0])}, {_fmt_inline(rng[1])}]"
                if isinstance(rng, (list, tuple))
                else _fmt_inline(rng)
            )
            freq = _fmt_inline(entry.get("resample_frequency", UNRESOLVED))
            esrc = entry.get("source", "")
            src_cell = f"[{esrc}]" if esrc and esrc != UNRESOLVED else "—"
            rows.append([param, dist, rng_str, freq, src_cell])
        doc.table(["Parameter", "Distribution", "Range", "Resample Freq.", "Source"], rows)

    ur: list[dict] = []
    _collect_unresolved(dyn, "dynamics", ur)
    doc.register_unresolved(ur)


def _render_perception(doc: _DocBuilder, spec: dict) -> None:
    """Render the perception pipeline section."""
    perc = spec.get("perception")
    if not perc:
        return

    doc.section("Perception", level=2)
    src = perc.get("source", "")

    img = perc.get("input_image")
    if _is_unresolved(img):
        doc.p("**Input image:** ⚠️ `UNRESOLVED`")
        doc.todo("perception.input_image", "Identify camera/sensor resolution and channel count.", str(src))
    elif isinstance(img, dict):
        doc.section("Input Image", level=3)
        isrc = img.get("source", "")
        doc.kv("Resolution", img.get("resolution", UNRESOLVED), isrc)
        doc.kv("Channels", img.get("channels", UNRESOLVED), isrc)
        doc.kv("dtype", img.get("dtype", UNRESOLVED))

    inorm = perc.get("intrinsics_normalization")
    if isinstance(inorm, dict):
        doc.section("Intrinsics Normalization", level=3)
        doc.kv("Target K matrix", inorm.get("target_K", UNRESOLVED), inorm.get("source", ""))

    seg = perc.get("segmentation_model")
    if isinstance(seg, dict):
        doc.section("Segmentation Model", level=3)
        ssrc = seg.get("source", "")
        doc.kv("Architecture", seg.get("architecture", UNRESOLVED), ssrc)
        doc.kv("Input size", seg.get("input_size", UNRESOLVED))
        doc.kv("Output classes", seg.get("output_classes", UNRESOLVED))

    augs = perc.get("augmentations", [])
    if augs:
        doc.section("Augmentations", level=3)
        for aug in augs:
            aname = aug.get("name", UNRESOLVED)
            asrc = aug.get("source", "")
            params = aug.get("parameters", {})
            doc.p(f"- **`{aname}`**{_cite(asrc)}" + (f": {params}" if params else ""))

    ur: list[dict] = []
    _collect_unresolved(perc, "perception", ur)
    doc.register_unresolved(ur)


def _render_world_model(doc: _DocBuilder, spec: dict) -> None:
    """Render the world model (RSSM / transformer) section."""
    wm = spec.get("world_model")
    if not wm:
        return

    doc.section("World Model", level=2)
    src = wm.get("source", "")
    doc.kv("Architecture", wm.get("architecture", UNRESOLVED), src)
    doc.kv("symlog normalization", wm.get("symlog", False))
    doc.kv("Normalization", wm.get("normalization", UNRESOLVED))

    dl = wm.get("discrete_latent")
    if not _is_unresolved(dl) and isinstance(dl, dict):
        doc.section("Discrete Latent", level=3)
        dlsrc = dl.get("source", "")
        doc.kv("Number of categoricals", dl.get("num_categoricals", UNRESOLVED), dlsrc)
        doc.kv("Number of classes per categorical", dl.get("num_classes", UNRESOLVED), dlsrc)

    components = wm.get("components", {})
    if components:
        doc.section("Components", level=3)
        rows = []
        for cname, comp in sorted(components.items()):
            if not isinstance(comp, dict):
                continue
            ctype = _fmt_inline(comp.get("type", UNRESOLVED))
            layers = comp.get("layers")
            layers_str = _fmt_inline(layers) if layers is not None else "—"
            hidden = comp.get("hidden_size")
            hidden_str = _fmt_inline(hidden) if hidden is not None else "—"
            csrc = comp.get("source", "")
            src_cell = f"[{csrc}]" if csrc and csrc != UNRESOLVED else "—"
            rows.append([f"`{cname}`", ctype, layers_str, hidden_str, src_cell])
        doc.table(["Component", "Type", "Layers", "Hidden Size", "Source"], rows)

    ur: list[dict] = []
    _collect_unresolved(wm, "world_model", ur)
    doc.register_unresolved(ur)


def _render_actor_critic(doc: _DocBuilder, spec: dict) -> None:
    """Render the actor-critic section."""
    ac = spec.get("actor_critic")
    if not ac:
        return

    doc.section("Actor-Critic", level=2)
    src = ac.get("source", "")
    doc.kv("Policy distribution", ac.get("policy_distribution", UNRESOLVED), src)
    doc.kv("Deterministic eval", ac.get("deterministic_eval", True))
    doc.kv("Imagination horizon", ac.get("imagination_horizon", UNRESOLVED), src)
    doc.kv("Discount (γ)", ac.get("discount", UNRESOLVED), src)
    doc.kv("GAE lambda (λ)", ac.get("lambda_gae", UNRESOLVED), src)

    def _net(key: str, net: dict | str) -> None:
        if _is_unresolved(net) or not isinstance(net, dict):
            doc.p(f"**{key}:** ⚠️ `UNRESOLVED`")
            return
        nsrc = net.get("source", "")
        doc.kv("Hidden layers", net.get("hidden_layers", UNRESOLVED), nsrc)
        doc.kv("Activation", net.get("activation", UNRESOLVED))

    doc.section("Actor Network", level=3)
    _net("Actor", ac.get("actor", UNRESOLVED))
    doc.section("Critic Network", level=3)
    _net("Critic", ac.get("critic", UNRESOLVED))

    regularizers = ac.get("regularizers", [])
    if regularizers:
        doc.section("Regularizers", level=3)
        rows = []
        for r in regularizers:
            rows.append(
                [
                    _fmt_inline(r.get("name", UNRESOLVED)),
                    _fmt_inline(r.get("coefficient", UNRESOLVED)),
                    f"[{r.get('source', '')}]" if r.get("source") else "—",
                ]
            )
        doc.table(["Name", "Coefficient", "Source"], rows)

    ur: list[dict] = []
    _collect_unresolved(ac, "actor_critic", ur)
    doc.register_unresolved(ur)


def _render_training(doc: _DocBuilder, spec: dict) -> None:
    """Render the training configuration section."""
    train = spec.get("training")
    if not train:
        return

    doc.section("Training", level=2)
    src = train.get("source", "")
    doc.kv("Algorithm", train.get("algorithm", UNRESOLVED), src)
    doc.kv("Train ratio", train.get("train_ratio", UNRESOLVED), src)
    doc.kv("Use AMP", train.get("use_amp", False))

    replay = train.get("replay")
    if not _is_unresolved(replay) and isinstance(replay, dict):
        doc.section("Replay Buffer", level=3)
        rsrc = replay.get("source", "")
        doc.kv("Capacity (steps)", replay.get("capacity_steps", UNRESOLVED), rsrc)
        doc.kv("Context length", replay.get("context_length", UNRESOLVED), rsrc)
        doc.kv("Sampling", replay.get("sampling", UNRESOLVED))

    batch = train.get("batch")
    if not _is_unresolved(batch) and isinstance(batch, dict):
        doc.section("Batch Configuration", level=3)
        bsrc = batch.get("source", "")
        doc.kv("Batch size", batch.get("size", UNRESOLVED), bsrc)
        doc.kv("Batch length", batch.get("length", UNRESOLVED), bsrc)

    opt = train.get("optimizer")
    if not _is_unresolved(opt) and isinstance(opt, dict):
        doc.section("Optimizer", level=3)
        osrc = opt.get("source", "")
        doc.kv("Type", opt.get("type", UNRESOLVED), osrc)
        doc.kv("Learning rate (lr)", opt.get("lr", UNRESOLVED), osrc)
        doc.kv("Epsilon (eps)", opt.get("eps", UNRESOLVED))
        if opt.get("clip_grad") is not None:
            doc.kv("Gradient clip norm", opt["clip_grad"])

    schedule = train.get("schedule")
    if not _is_unresolved(schedule) and isinstance(schedule, dict):
        doc.section("Training Schedule", level=3)
        ssrc = schedule.get("source", "")
        doc.kv("Total environment steps", schedule.get("total_env_steps", UNRESOLVED), ssrc)
        phases = schedule.get("phases", [])
        if phases:
            rows = []
            for ph in phases:
                psrc = ph.get("source", "")
                rows.append(
                    [
                        _fmt_inline(ph.get("name", UNRESOLVED)),
                        _fmt_inline(ph.get("start_step", UNRESOLVED)),
                        _fmt_inline(ph.get("end_step", UNRESOLVED)),
                        _fmt_inline(ph.get("learning_rate", UNRESOLVED)),
                        _fmt_inline(ph.get("entropy_scale", UNRESOLVED)),
                        f"[{psrc}]" if psrc and psrc != UNRESOLVED else "—",
                    ]
                )
            doc.table(
                ["Phase", "Start Step", "End Step", "LR", "Entropy Scale", "Source"],
                rows,
            )

    ur: list[dict] = []
    _collect_unresolved(train, "training", ur)
    doc.register_unresolved(ur)


def _render_evaluation(doc: _DocBuilder, spec: dict) -> None:
    """Render the evaluation protocol section."""
    evl = spec.get("evaluation")
    if not evl:
        return

    doc.section("Evaluation", level=2)
    src = evl.get("source", "")
    doc.kv("Number of seeds", evl.get("num_seeds", UNRESOLVED), src)
    doc.kv("Eval episodes per seed", evl.get("num_eval_episodes", UNRESOLVED), src)
    doc.kv("Report error bars", evl.get("report_error_bars", False))

    criteria = evl.get("success_criteria", [])
    if criteria:
        doc.section("Success Criteria", level=3)
        rows = []
        for c in criteria:
            rows.append(
                [
                    _fmt_inline(c.get("metric", UNRESOLVED)),
                    _fmt_inline(c.get("threshold", UNRESOLVED)),
                    _fmt_inline(c.get("direction", UNRESOLVED)),
                    f"[{c.get('source', '')}]" if c.get("source") else "—",
                ]
            )
        doc.table(["Metric", "Threshold", "Direction", "Source"], rows)

    ablations = evl.get("ablations", [])
    if ablations:
        doc.section("Ablation Studies", level=3)
        for ab in ablations:
            abname = ab.get("name", UNRESOLVED)
            abdesc = ab.get("description", UNRESOLVED)
            absrc = ab.get("source", "")
            doc.p(f"- **`{abname}`**{_cite(absrc)}: {abdesc}")

    ur: list[dict] = []
    _collect_unresolved(evl, "evaluation", ur)
    doc.register_unresolved(ur)


def _render_deployment(doc: _DocBuilder, spec: dict) -> None:
    """Render the deployment / inference section."""
    dep = spec.get("deployment")
    if not dep:
        return

    doc.section("Deployment", level=2)
    src = dep.get("source", "")
    doc.kv("Target hardware", dep.get("target_hardware", UNRESOLVED), src)
    stack = dep.get("inference_stack", [])
    if stack:
        doc.kv("Inference stack", ", ".join(f"`{s}`" for s in stack))
    doc.kv("Control frequency", dep.get("control_frequency_hz", UNRESOLVED))

    budgets = dep.get("runtime_budgets", [])
    if budgets:
        doc.section("Runtime Budgets", level=3)
        rows = []
        for b in budgets:
            rows.append(
                [
                    _fmt_inline(b.get("module", UNRESOLVED)),
                    _fmt_inline(b.get("max_ms", UNRESOLVED)),
                    f"[{b.get('source', '')}]" if b.get("source") else "—",
                ]
            )
        doc.table(["Module", "Max latency (ms)", "Source"], rows)

    safety = dep.get("safety_constraints", [])
    if safety:
        doc.section("Safety Constraints", level=3)
        for s in safety:
            sname = s.get("name", UNRESOLVED)
            sdesc = s.get("description", UNRESOLVED)
            ssrc = s.get("source", "")
            doc.p(f"- **`{sname}`**{_cite(ssrc)}: {sdesc}")

    ur: list[dict] = []
    _collect_unresolved(dep, "deployment", ur)
    doc.register_unresolved(ur)


def _render_imports(doc: _DocBuilder, spec: dict) -> None:
    """Render the baseline imports section."""
    imports = spec.get("imports", [])
    if not imports:
        return

    doc.section("Baseline Imports", level=2)
    doc.p(
        "This spec inherits from the following baseline specs. "
        "Fields not explicitly overridden here use the baseline values."
    )
    for imp in imports:
        name = imp.get("name", UNRESOLVED)
        version = imp.get("version", UNRESOLVED)
        url = imp.get("spec_url")
        overrides = imp.get("overrides", [])

        header = f"- **{name}** @ `{version}`"
        if url:
            header += f" ([spec]({url}))"
        doc.p(header)
        if overrides:
            doc.p("  Overrides in this spec:")
            for ov in overrides:
                doc.p(f"  - `{ov}`")


# ---------------------------------------------------------------------------
# Reproducibility appendix
# ---------------------------------------------------------------------------


def _render_reproducibility_appendix(doc: _DocBuilder, spec: dict) -> None:
    """
    Auto-generate a Reproducibility Appendix section.

    Covers: hyperparameters, experimental setup, compute, metrics.
    Mirrors the NeurIPS reproducibility checklist.
    """
    doc.hr()
    doc.section("Reproducibility Appendix", level=2)
    doc.p(
        "This appendix is auto-generated from the spec. "
        "It maps to NeurIPS reproducibility checklist items C1–C4 "
        "and RL-specific items."
    )

    # --- Hyperparameters table ---
    doc.section("Hyperparameters (C1: Full list)", level=3)

    hp_rows: list[list[str]] = []

    def _src_cell(source: Any) -> str:
        """Format a source value as a plain table cell string (no brackets added)."""
        if not source:
            return "—"
        if isinstance(source, str):
            return source if source and source != UNRESOLVED else "—"
        if isinstance(source, dict):
            parts: list[str] = []
            if source.get("section"):
                parts.append(source["section"])
            if source.get("table"):
                parts.append(source["table"])
            if source.get("equation"):
                parts.append(source["equation"])
            if source.get("figure"):
                parts.append(source["figure"])
            if source.get("page"):
                parts.append(f"p.{source['page']}")
            return ", ".join(parts) if parts else "—"
        return str(source) if source else "—"

    def _add_hp(param: str, value: Any, source: Any, selection: str = "Standard") -> None:
        hp_rows.append(
            [param, _fmt_inline(value), selection, _src_cell(source)]
        )

    train = spec.get("training", {}) or {}
    ac = spec.get("actor_critic", {}) or {}
    wm = spec.get("world_model", {}) or {}

    # Training hyperparameters
    src_t = train.get("source", "")
    _add_hp("Algorithm", train.get("algorithm", UNRESOLVED), src_t)
    _add_hp("Train ratio", train.get("train_ratio", UNRESOLVED), src_t)
    _add_hp("Use AMP", train.get("use_amp", UNRESOLVED), src_t)

    replay = train.get("replay") or {}
    if isinstance(replay, dict):
        src_r = replay.get("source", "")
        _add_hp("Replay capacity (steps)", replay.get("capacity_steps", UNRESOLVED), src_r)
        _add_hp("Replay context length", replay.get("context_length", UNRESOLVED), src_r)

    batch = train.get("batch") or {}
    if isinstance(batch, dict):
        src_b = batch.get("source", "")
        _add_hp("Batch size", batch.get("size", UNRESOLVED), src_b)
        _add_hp("Batch length", batch.get("length", UNRESOLVED), src_b)

    opt = train.get("optimizer") or {}
    if isinstance(opt, dict):
        src_o = opt.get("source", "")
        _add_hp("Optimizer type", opt.get("type", UNRESOLVED), src_o)
        _add_hp("Learning rate", opt.get("lr", UNRESOLVED), src_o, "Tuned")
        _add_hp("Optimizer epsilon", opt.get("eps", UNRESOLVED), src_o)
        if opt.get("clip_grad") is not None:
            _add_hp("Gradient clip norm", opt["clip_grad"], src_o)

    schedule = train.get("schedule") or {}
    if isinstance(schedule, dict):
        src_s = schedule.get("source", "")
        _add_hp("Total env steps", schedule.get("total_env_steps", UNRESOLVED), src_s)

    # Actor-critic hyperparameters
    src_ac = ac.get("source", "")
    _add_hp("Discount (γ)", ac.get("discount", UNRESOLVED), src_ac)
    _add_hp("GAE lambda (λ)", ac.get("lambda_gae", UNRESOLVED), src_ac)
    _add_hp("Imagination horizon", ac.get("imagination_horizon", UNRESOLVED), src_ac)
    _add_hp("Policy distribution", ac.get("policy_distribution", UNRESOLVED), src_ac)

    actor_net = ac.get("actor") or {}
    if isinstance(actor_net, dict):
        _add_hp("Actor hidden layers", actor_net.get("hidden_layers", UNRESOLVED), actor_net.get("source", ""))
        _add_hp("Actor activation", actor_net.get("activation", UNRESOLVED), actor_net.get("source", ""))

    critic_net = ac.get("critic") or {}
    if isinstance(critic_net, dict):
        _add_hp("Critic hidden layers", critic_net.get("hidden_layers", UNRESOLVED), critic_net.get("source", ""))

    # World model hyperparameters
    src_wm = wm.get("source", "")
    _add_hp("World model architecture", wm.get("architecture", UNRESOLVED), src_wm)
    dl = wm.get("discrete_latent") or {}
    if isinstance(dl, dict):
        src_dl = dl.get("source", "")
        _add_hp("Discrete latent categoricals", dl.get("num_categoricals", UNRESOLVED), src_dl)
        _add_hp("Discrete latent classes", dl.get("num_classes", UNRESOLVED), src_dl)
    _add_hp("symlog normalization", wm.get("symlog", False), src_wm)
    _add_hp("Layer normalization", wm.get("normalization", UNRESOLVED), src_wm)

    # Reward
    reward = spec.get("reward") or {}
    src_r2 = reward.get("source", "")
    _add_hp("Reward discount", reward.get("discount", UNRESOLVED), src_r2)
    _add_hp("Reward normalization", reward.get("normalization", UNRESOLVED), src_r2)

    # Timing
    timing = spec.get("timing") or {}
    src_tim = timing.get("source", "")
    _add_hp("Control frequency (Hz)", timing.get("control_frequency_hz", UNRESOLVED), src_tim)
    _add_hp("Sensor delay (ms)", timing.get("sensor_delay_ms", UNRESOLVED), src_tim)
    _add_hp("Action delay (ms)", timing.get("action_delay_ms", UNRESOLVED), src_tim)

    if hp_rows:
        doc.table(["Parameter", "Value", "Selection Method", "Source"], hp_rows)

    # --- Experimental setup (C2) ---
    doc.section("Experimental Setup (C2: Seeds and error bars)", level=3)
    evl = spec.get("evaluation") or {}
    src_ev = evl.get("source", "")
    doc.kv("Number of seeds", evl.get("num_seeds", UNRESOLVED), src_ev)
    doc.kv("Eval episodes per seed", evl.get("num_eval_episodes", UNRESOLVED), src_ev)
    doc.kv("Error bars reported", evl.get("report_error_bars", UNRESOLVED))

    # --- Compute resources (C3) ---
    doc.section("Compute Resources (C3: Infrastructure)", level=3)
    dep = spec.get("deployment") or {}
    src_dep = dep.get("source", "")
    doc.kv("Training / inference hardware", dep.get("target_hardware", UNRESOLVED), src_dep)
    stack = dep.get("inference_stack", [])
    if stack:
        doc.kv("Software stack", ", ".join(f"`{s}`" for s in stack))

    # Derived: total GPU hours estimate (if seeds and total_env_steps are known).
    num_seeds = evl.get("num_seeds")
    total_steps = schedule.get("total_env_steps") if isinstance(schedule, dict) else None
    if (
        isinstance(num_seeds, int)
        and isinstance(total_steps, int)
        and not _is_unresolved(num_seeds)
        and not _is_unresolved(total_steps)
    ):
        doc.p(
            f"> **Derived:** {num_seeds} seed(s) × "
            f"{total_steps:,} env steps = "
            f"{num_seeds * total_steps:,} total env steps."
        )
    else:
        doc.p("> **Total GPU hours:** ⚠️ `UNRESOLVED` — requires training time per seed.")

    # --- Evaluation metrics (C4) ---
    doc.section("Evaluation Metrics (C4: Success criteria)", level=3)
    criteria = evl.get("success_criteria", [])
    if criteria:
        rows2 = []
        for c in criteria:
            rows2.append(
                [
                    _fmt_inline(c.get("metric", UNRESOLVED)),
                    _fmt_inline(c.get("threshold", UNRESOLVED)),
                    _fmt_inline(c.get("direction", UNRESOLVED)),
                    f"[{c.get('source', '')}]" if c.get("source") else "—",
                ]
            )
        doc.table(["Metric", "Threshold", "Direction", "Source"], rows2)
    else:
        doc.p("⚠️ No success criteria defined — `UNRESOLVED`.")
        doc.todo("evaluation.success_criteria", "Define quantitative success criteria.", "")

    # --- RL-specific: domain randomization ---
    dyn = spec.get("dynamics") or {}
    dr = dyn.get("domain_randomization", [])
    if dr:
        doc.section("Domain Randomization (RL-specific)", level=3)
        rows3 = []
        for entry in dr:
            rng = entry.get("range", [UNRESOLVED, UNRESOLVED])
            rng_str = (
                f"[{_fmt_inline(rng[0])}, {_fmt_inline(rng[1])}]"
                if isinstance(rng, (list, tuple))
                else _fmt_inline(rng)
            )
            rows3.append(
                [
                    _fmt_inline(entry.get("parameter", UNRESOLVED)),
                    _fmt_inline(entry.get("distribution", UNRESOLVED)),
                    rng_str,
                    _fmt_inline(entry.get("resample_frequency", UNRESOLVED)),
                ]
            )
        doc.table(["Parameter", "Distribution", "Range", "Resample Freq."], rows3)


# ---------------------------------------------------------------------------
# UNRESOLVED summary section
# ---------------------------------------------------------------------------


def _render_unresolved_summary(doc: _DocBuilder) -> None:
    """Render the UNRESOLVED Items summary section."""
    items = doc._unresolved_items
    if not items:
        doc.section("UNRESOLVED Items", level=2)
        doc.p("All fields are resolved. The spec is complete.")
        return

    # Sort: HIGH first, then MEDIUM, then LOW; within priority alphabetically.
    priority_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    items_sorted = sorted(
        items, key=lambda x: (priority_rank.get(x["priority"], 99), x["path"])
    )

    # De-duplicate by path.
    seen: set[str] = set()
    unique: list[dict] = []
    for item in items_sorted:
        if item["path"] not in seen:
            seen.add(item["path"])
            unique.append(item)

    high = [i for i in unique if i["priority"] == "HIGH"]
    medium = [i for i in unique if i["priority"] == "MEDIUM"]
    low = [i for i in unique if i["priority"] == "LOW"]

    doc.section("UNRESOLVED Items", level=2)
    doc.p(
        f"**{len(unique)} field(s) require manual extraction.** "
        f"High: {len(high)}, Medium: {len(medium)}, Low: {len(low)}."
    )
    doc.p(
        "> This spec cannot be finalised until all HIGH-priority items are resolved. "
        "See `spec.lock.json` for the current hash and unresolved count."
    )

    def _priority_badge(p: str) -> str:
        badges = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        return badges.get(p, "⚪")

    if high:
        doc.section("High Priority", level=3)
        rows = []
        for item in high:
            rows.append(
                [
                    f"{_priority_badge('HIGH')} HIGH",
                    f"`{item['path']}`",
                    item["section"],
                ]
            )
        doc.table(["Priority", "Field Path", "Section"], rows)

    if medium:
        doc.section("Medium Priority", level=3)
        rows = []
        for item in medium:
            rows.append(
                [
                    f"{_priority_badge('MEDIUM')} MEDIUM",
                    f"`{item['path']}`",
                    item["section"],
                ]
            )
        doc.table(["Priority", "Field Path", "Section"], rows)

    if low:
        doc.section("Low Priority", level=3)
        rows = []
        for item in low:
            rows.append(
                [
                    f"{_priority_badge('LOW')} LOW",
                    f"`{item['path']}`",
                    item["section"],
                ]
            )
        doc.table(["Priority", "Field Path", "Section"], rows)


# ---------------------------------------------------------------------------
# Load spec from IR or spec.yaml
# ---------------------------------------------------------------------------


def _load_spec_from_ir(ir_path: Path) -> dict:
    """
    Load an IR JSON file and convert it to spec dict format.

    Rather than duplicating the full emit_yaml.py logic, we import it if
    available; otherwise fall back to a direct JSON → YAML passthrough.
    """
    with ir_path.open("r", encoding="utf-8") as fh:
        ir: dict = json.load(fh)

    # Try to use emit_yaml.build_spec_dict if it is co-located.
    emit_yaml_path = ir_path.parent / "emit_yaml.py"
    if emit_yaml_path.exists():
        try:
            import importlib.util

            spec_module = importlib.util.spec_from_file_location(
                "emit_yaml", str(emit_yaml_path)
            )
            mod = importlib.util.module_from_spec(spec_module)
            spec_module.loader.exec_module(mod)
            return mod.build_spec_dict(ir, [])
        except Exception as exc:
            print(
                f"WARNING: Could not import emit_yaml.py ({exc}); "
                "falling back to raw IR passthrough.",
                file=sys.stderr,
            )

    return ir


def _load_spec_from_yaml(spec_path: Path) -> dict:
    """Load a spec.yaml file."""
    with spec_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"spec.yaml is not a YAML mapping: {spec_path}")
    return data


# ---------------------------------------------------------------------------
# Top-level document renderer
# ---------------------------------------------------------------------------

SECTION_RENDERERS = [
    ("meta", _render_meta),
    ("frames", _render_frames),
    ("spaces", _render_spaces),
    ("timing", _render_timing),
    ("reward", _render_reward),
    ("termination", _render_termination),
    ("gates_track", _render_gates_track),
    ("dynamics", _render_dynamics),
    ("perception", _render_perception),
    ("world_model", _render_world_model),
    ("actor_critic", _render_actor_critic),
    ("training", _render_training),
    ("evaluation", _render_evaluation),
    ("deployment", _render_deployment),
    ("imports", _render_imports),
]


def render_spec_md(spec: dict, source_path: str) -> str:
    """
    Render a spec dict into a Markdown document.

    Parameters
    ----------
    spec:
        The spec dict (either from ir_schema or spec.yaml).
    source_path:
        Human-readable description of the input file (for the document header).

    Returns
    -------
    str
        The full Markdown document.
    """
    doc = _DocBuilder()

    # --- Document header ---
    meta = spec.get("meta", {})
    paper = meta.get("paper", {})
    title = paper.get("title", "Untitled Paper")

    header_lines = [
        f"# Spec: {title}",
        "",
        f"> Generated from: `{source_path}`  ",
        f"> Tool version: `{meta.get('tool_version', 'unknown')}`  ",
        f"> Generated at: `{meta.get('generated_at', 'unknown')}`  ",
        f"> Unresolved fields: `{meta.get('unresolved_count', '?')}`  ",
        "",
    ]

    arxiv = paper.get("arxiv")
    version = paper.get("version")
    if arxiv:
        arxiv_url = f"https://arxiv.org/abs/{arxiv}"
        if version:
            arxiv_url += f"{version}"
        header_lines.append(f"> arXiv: [{arxiv}{version or ''}]({arxiv_url})")

    doc._parts.insert(0, "\n".join(header_lines))

    # --- Render each section ---
    for section_key, renderer in SECTION_RENDERERS:
        if section_key in spec or section_key == "meta":
            renderer(doc, spec)

    # --- Reproducibility appendix ---
    _render_reproducibility_appendix(doc, spec)

    # --- UNRESOLVED summary ---
    _render_unresolved_summary(doc)

    return doc.render()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Markdown spec emitter: converts IR JSON or spec.yaml into a "
            "human-readable spec.md with traceability links and reproducibility appendix."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # From IR JSON:
  python emit_md.py --ir ir_output.json --output spec.md

  # From spec.yaml:
  python emit_md.py --spec spec.yaml --output spec.md

  # Custom output location:
  python emit_md.py --ir ir_output.json --output /tmp/my_paper/spec.md
""",
    )

    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--ir",
        metavar="FILE",
        help="Path to IR JSON file (output of tex_parser.py or ir_schema serialisation).",
    )
    input_group.add_argument(
        "--spec",
        metavar="FILE",
        help="Path to spec.yaml (output of emit_yaml.py).",
    )
    p.add_argument(
        "--output",
        default="spec.md",
        metavar="FILE",
        help="Path for the output spec.md (default: spec.md).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_path = Path(args.output).resolve()

    # --- Load input ---
    if args.ir:
        ir_path = Path(args.ir).resolve()
        if not ir_path.exists():
            print(f"ERROR: IR file does not exist: {ir_path}", file=sys.stderr)
            return 1
        print(f"Loading IR from: {ir_path}", file=sys.stderr)
        try:
            spec = _load_spec_from_ir(ir_path)
        except json.JSONDecodeError as exc:
            print(f"ERROR: Failed to parse IR JSON: {exc}", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"ERROR: Cannot read IR file: {exc}", file=sys.stderr)
            return 1
        source_desc = str(ir_path)
    else:
        spec_path = Path(args.spec).resolve()
        if not spec_path.exists():
            print(f"ERROR: spec.yaml does not exist: {spec_path}", file=sys.stderr)
            return 1
        print(f"Loading spec from: {spec_path}", file=sys.stderr)
        try:
            spec = _load_spec_from_yaml(spec_path)
        except (yaml.YAMLError, ValueError) as exc:
            print(f"ERROR: Failed to parse spec.yaml: {exc}", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"ERROR: Cannot read spec.yaml: {exc}", file=sys.stderr)
            return 1
        source_desc = str(spec_path)

    # --- Render ---
    print("Rendering Markdown...", file=sys.stderr)
    md_text = render_spec_md(spec, source_desc)

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(md_text)

    print(f"spec.md written to: {output_path}", file=sys.stderr)

    # Count UNRESOLVED in the rendered content for summary.
    ur_count = md_text.count("`UNRESOLVED`")
    print(
        f"Done. Approximate UNRESOLVED markers in output: {ur_count}.",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
