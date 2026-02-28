#!/usr/bin/env python3
r"""
tex_parser.py -- LaTeX source parser for the spec-compiler IR.

Parses arXiv LaTeX tarballs and extracts four structured categories:
  1. Symbol definitions  (\newcommand, \def, \DeclareMathOperator)
  2. Equations           (equation, align, gather, multline, eqnarray)
  3. Tables              (tabular, table, longtable)
  4. Document structure  (\section, \subsection, \subsubsection, \paragraph)

Output is a single JSON file suitable for downstream IR population.

Usage
-----
    python tex_parser.py --source-dir /path/to/tex/ --output output.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SourceTrace:
    file: str
    line_start: int
    line_end: int


@dataclass
class SymbolDef:
    name: str
    latex_definition: str
    expansion: str
    source_trace: dict


@dataclass
class Equation:
    label: Optional[str]
    raw_latex: str
    ast: Any  # dict or "UNRESOLVED"
    source_trace: dict


@dataclass
class Table:
    caption: Optional[str]
    headers: list[str]
    rows: list[list[str]]
    source_trace: dict


@dataclass
class Section:
    level: str          # "section" | "subsection" | "subsubsection" | "paragraph"
    title: str
    file: str
    line_start: int
    line_end: int       # line where next same-or-higher level begins; -1 if unknown


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# LaTeX comment: everything from an unescaped % to end of line
_COMMENT_RE = re.compile(r'(?<!\\)%.*$', re.MULTILINE)


def strip_comments(text: str) -> str:
    """Remove LaTeX line comments (unescaped % to end of line)."""
    return _COMMENT_RE.sub('', text)


def read_file_lines(path: Path) -> list[str]:
    """Read a file, returning lines. Falls back to latin-1 if UTF-8 fails."""
    try:
        return path.read_text(encoding='utf-8').splitlines()
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding='latin-1').splitlines()
        except Exception:
            return []


def extract_balanced_braces(text: str, start: int) -> tuple[str, int]:
    """
    Given `text` and an index `start` pointing at '{', extract the contents
    of the matching closing brace, handling nesting.

    Returns (contents, end_index) where end_index is the index after '}'.
    Returns ("", start) on failure.
    """
    if start >= len(text) or text[start] != '{':
        return "", start

    depth = 0
    i = start
    buf: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == '\\':
            # Escaped character — include both chars literally
            if i + 1 < len(text):
                buf.append(text[i:i + 2])
                i += 2
            else:
                buf.append(ch)
                i += 1
            continue
        if ch == '{':
            depth += 1
            if depth > 1:
                buf.append(ch)
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return ''.join(buf), i + 1
            else:
                buf.append(ch)
        else:
            buf.append(ch)
        i += 1
    # Unbalanced — return what we have
    return ''.join(buf), i


def strip_formatting(text: str) -> str:
    r"""
    Strip common LaTeX formatting commands, leaving only their argument text.
    Handles: \textbf, \textit, \emph, \text, \textrm, \texttt, \textsf,
             \mathbf, \mathrm, \mathit, \mathsf, \mathbb, \mathcal, \boldsymbol,
             \underline, \overline, \widehat, \widetilde, \hat, \tilde,
             \bar, \vec, \dot, \ddot.
    Also strips dollar signs for inline math.
    """
    formatting_cmds = (
        r'\\(?:textbf|textit|textsl|textrm|texttt|textsf|textsc|emph|text'
        r'|mathbf|mathrm|mathit|mathsf|mathbb|mathcal|boldsymbol'
        r'|underline|overline|widehat|widetilde|hat|tilde|bar|vec|dot|ddot'
        r'|mbox|hbox|vbox|fbox|phantom|hphantom|vphantom)'
    )
    pattern = re.compile(formatting_cmds + r'\s*\{')

    while True:
        m = pattern.search(text)
        if not m:
            break
        before = text[:m.start()]
        rest = text[m.end() - 1:]   # starts at '{'
        inner, end = extract_balanced_braces(rest, 0)
        text = before + inner + rest[end:]

    # Strip inline math delimiters
    text = re.sub(r'\$\$?', '', text)
    return text.strip()


def normalize_cell(text: str) -> str:
    """Normalize a table cell value: strip formatting, collapse whitespace."""
    text = strip_formatting(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# File resolution
# ---------------------------------------------------------------------------


def resolve_input_files(source_dir: Path, warnings: list[str]) -> list[Path]:
    r"""
    Find the root .tex file(s) and resolve all \input{} / \include{} chains.

    Strategy:
      1. Look for a file that contains \documentclass (the root).
      2. If none found, use all .tex files in the directory.
      3. BFS-expand \input / \include references.
    """
    all_tex = sorted(source_dir.rglob('*.tex'))
    if not all_tex:
        warnings.append(f"No .tex files found in {source_dir}")
        return []

    # Find root file(s)
    root_files: list[Path] = []
    for p in all_tex:
        lines = read_file_lines(p)
        combined = '\n'.join(lines)
        if r'\documentclass' in combined:
            root_files.append(p)

    if not root_files:
        warnings.append(
            "No \\documentclass found; treating all .tex files as roots."
        )
        return all_tex

    # BFS expansion
    ordered: list[Path] = []
    seen: set[Path] = set()
    queue = list(root_files)

    input_re = re.compile(r'\\(?:input|include)\s*\{([^}]+)\}')

    while queue:
        current = queue.pop(0)
        current_resolved = current.resolve()
        if current_resolved in seen:
            continue
        seen.add(current_resolved)
        if not current.exists():
            warnings.append(f"Referenced file not found: {current}")
            continue
        ordered.append(current)

        lines = read_file_lines(current)
        text = strip_comments('\n'.join(lines))
        for m in input_re.finditer(text):
            ref = m.group(1).strip()
            candidate = (current.parent / ref)
            if not candidate.suffix:
                candidate = candidate.with_suffix('.tex')
            candidate = candidate.resolve()
            if candidate.exists() and candidate not in seen:
                queue.append(candidate)
            elif not candidate.exists():
                # Try without adding .tex (already has extension)
                alt = (current.parent / ref).resolve()
                if alt.exists() and alt not in seen:
                    queue.append(alt)
                else:
                    warnings.append(f"Cannot resolve \\input{{{ref}}} from {current}")

    return ordered


# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------

# Matches \newcommand{\name}[nargs][default]{definition}
# and     \newcommand*{\name}...
_NEWCOMMAND_RE = re.compile(
    r'\\(?:newcommand|renewcommand|providecommand)\*?\s*'
    r'\{?(\\[A-Za-z@]+)\}?'          # command name
    r'(?:\s*\[(\d+)\])?'              # optional arg count
    r'(?:\s*\[[^\]]*\])?'             # optional default
    r'\s*(\{)',                        # opening brace of definition
    re.DOTALL,
)

# Matches \def\name{definition}
_DEF_RE = re.compile(
    r'\\def\s*(\\[A-Za-z@]+)\s*'
    r'(?:#\d\s*)*'                    # optional parameter tokens #1 #2 ...
    r'(\{)',
    re.DOTALL,
)

# Matches \DeclareMathOperator{*}{\name}{expansion}
_DECLAREMATH_RE = re.compile(
    r'\\DeclareMathOperator\*?\s*'
    r'\{?(\\[A-Za-z@]+)\}?'
    r'\s*(\{)',
    re.DOTALL,
)


def _extract_symbols(text: str, filename: str, lines: list[str]) -> list[dict]:
    """Extract symbol definitions from a stripped (comment-free) text."""

    def line_of(pos: int) -> int:
        return text[:pos].count('\n') + 1

    def make_trace(pos_start: int, pos_end: int) -> dict:
        ls = line_of(pos_start)
        le = line_of(pos_end)
        return {"file": filename, "line_start": ls, "line_end": le}

    results: list[dict] = []
    seen_names: set[str] = set()

    def add_result(name: str, raw_def: str, expansion: str, trace: dict):
        if name in seen_names:
            return
        seen_names.add(name)
        results.append({
            "name": name,
            "latex_definition": raw_def,
            "expansion": expansion,
            "source_trace": trace,
        })

    # \newcommand / \renewcommand / \providecommand
    for m in _NEWCOMMAND_RE.finditer(text):
        name = m.group(1)
        brace_start = m.start(3)
        expansion, end = extract_balanced_braces(text, brace_start)
        raw = text[m.start():end]
        trace = make_trace(m.start(), end)
        add_result(name, raw.strip(), expansion.strip(), trace)

    # \def
    for m in _DEF_RE.finditer(text):
        name = m.group(1)
        brace_start = m.start(2)
        expansion, end = extract_balanced_braces(text, brace_start)
        raw = text[m.start():end]
        trace = make_trace(m.start(), end)
        add_result(name, raw.strip(), expansion.strip(), trace)

    # \DeclareMathOperator
    for m in _DECLAREMATH_RE.finditer(text):
        name = m.group(1)
        brace_start = m.start(2)
        expansion, end = extract_balanced_braces(text, brace_start)
        raw = text[m.start():end]
        trace = make_trace(m.start(), end)
        add_result(name, raw.strip(), expansion.strip(), trace)

    return results


# ---------------------------------------------------------------------------
# Equation extraction
# ---------------------------------------------------------------------------

_EQ_ENVS = (
    'equation', 'equation*',
    'align', 'align*',
    'gather', 'gather*',
    'multline', 'multline*',
    'eqnarray', 'eqnarray*',
    'flalign', 'flalign*',
    'alignat', 'alignat*',
)

_LABEL_RE = re.compile(r'\\label\s*\{([^}]+)\}')


def _find_env_spans(text: str, env: str) -> list[tuple[int, int]]:
    """
    Return list of (start, end) character indices for each occurrence of
    \\begin{env}...\\end{env}, handling nested environments naively.
    """
    begin_pat = re.compile(r'\\begin\s*\{' + re.escape(env) + r'\}')
    end_pat = re.compile(r'\\end\s*\{' + re.escape(env) + r'\}')

    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        bm = begin_pat.search(text, pos)
        if not bm:
            break
        depth = 1
        search_pos = bm.end()
        start = bm.start()
        while depth > 0:
            next_begin = begin_pat.search(text, search_pos)
            next_end = end_pat.search(text, search_pos)
            if not next_end:
                # Unbalanced
                search_pos = len(text)
                break
            if next_begin and next_begin.start() < next_end.start():
                depth += 1
                search_pos = next_begin.end()
            else:
                depth -= 1
                if depth == 0:
                    spans.append((start, next_end.end()))
                search_pos = next_end.end()
        pos = search_pos
    return spans


# ---------------------------------------------------------------------------
# Simple expression AST builder
# ---------------------------------------------------------------------------


def _try_build_ast(latex: str) -> Any:
    r"""
    Attempt to parse a simple LaTeX math expression into an AST.

    Handles:
      - Numeric literals (integers, decimals)
      - Named identifiers (single letters or \commands)
      - Binary +, -, *, \cdot, \times
      - Fractions: \frac{num}{den}
      - Superscripts (^{}) / subscripts (_{}) on atoms
      - Parenthesised sub-expressions

    Returns a dict AST on success, or "UNRESOLVED" for anything too complex.

    The grammar handled (roughly):
      expr   := term (('+' | '-') term)*
      term   := factor (('*' | '\cdot' | '\times') factor)*
      factor := atom ('^' '{' expr '}')? ('_' '{' expr '}')?
      atom   := number | identifier | '\frac' '{' expr '}' '{' expr '}'
               | '(' expr ')' | '{' expr '}'
    """
    latex = latex.strip()
    # Remove \label{}, \tag{}, alignment markers, newlines
    latex = re.sub(r'\\label\s*\{[^}]*\}', '', latex)
    latex = re.sub(r'\\tag\s*\{[^}]*\}', '', latex)
    latex = re.sub(r'\\nonumber', '', latex)
    latex = re.sub(r'\\\\', ' ', latex)
    latex = re.sub(r'&', ' ', latex)
    latex = latex.strip()

    # If it's a multi-line / compound equation (contains = with complex sides),
    # delegate to UNRESOLVED quickly to avoid false parses
    eq_count = latex.count('=')
    if eq_count > 1:
        return "UNRESOLVED"

    # Tokenise
    tokens = _tokenise(latex)
    if tokens is None:
        return "UNRESOLVED"

    parser = _ExprParser(tokens)
    try:
        ast = parser.parse_expr()
    except _ParseError:
        return "UNRESOLVED"

    if not parser.at_end():
        return "UNRESOLVED"

    return ast


# Token types
_TK_NUM = 'NUM'
_TK_IDENT = 'IDENT'
_TK_PLUS = '+'
_TK_MINUS = '-'
_TK_STAR = '*'
_TK_LPAREN = '('
_TK_RPAREN = ')'
_TK_LBRACE = '{'
_TK_RBRACE = '}'
_TK_CARET = '^'
_TK_UNDERSCORE = '_'
_TK_FRAC = 'FRAC'
_TK_SQRT = 'SQRT'
_TK_CDOT = 'CDOT'
_TK_TIMES = 'TIMES'
_TK_EQUALS = '='
_TK_EOF = 'EOF'

_TOKEN_RE = re.compile(
    r'\s*(?:'
    r'(\\frac)'
    r'|(\\sqrt)'
    r'|(\\cdot|\\times)'
    r'|(\\[A-Za-z]+)'     # LaTeX command / identifier
    r'|([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)'   # number
    r'|([A-Za-z])'        # single letter identifier
    r'|(\+)'
    r'|(-)'
    r'|(\*)'
    r'|(\()'
    r'|(\))'
    r'|(\{)'
    r'|(\})'
    r'|(\^)'
    r'|(\_)'
    r'|(=)'
    r'|(.)'               # fallthrough — unknown char
    r')\s*',
    re.DOTALL,
)


def _tokenise(text: str) -> Optional[list[tuple[str, str]]]:
    """Tokenise a simple LaTeX math expression. Returns None if unknown tokens found."""
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            break
        if m.start() == m.end():
            break

        g = m.lastindex
        val = m.group(g)
        pos = m.end()

        if g == 1:
            tokens.append((_TK_FRAC, val))
        elif g == 2:
            tokens.append((_TK_SQRT, val))
        elif g == 3:
            tokens.append((_TK_CDOT, val))
        elif g == 4:
            tokens.append((_TK_IDENT, val))
        elif g == 5:
            tokens.append((_TK_NUM, val))
        elif g == 6:
            tokens.append((_TK_IDENT, val))
        elif g == 7:
            tokens.append((_TK_PLUS, val))
        elif g == 8:
            tokens.append((_TK_MINUS, val))
        elif g == 9:
            tokens.append((_TK_STAR, val))
        elif g == 10:
            tokens.append((_TK_LPAREN, val))
        elif g == 11:
            tokens.append((_TK_RPAREN, val))
        elif g == 12:
            tokens.append((_TK_LBRACE, val))
        elif g == 13:
            tokens.append((_TK_RBRACE, val))
        elif g == 14:
            tokens.append((_TK_CARET, val))
        elif g == 15:
            tokens.append((_TK_UNDERSCORE, val))
        elif g == 16:
            tokens.append((_TK_EQUALS, val))
        elif g == 17:
            # Unknown character — bail out
            return None

    tokens.append((_TK_EOF, ''))
    return tokens


class _ParseError(Exception):
    pass


class _ExprParser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self._tokens = tokens
        self._pos = 0

    def peek(self) -> tuple[str, str]:
        return self._tokens[self._pos]

    def consume(self) -> tuple[str, str]:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def expect(self, kind: str) -> tuple[str, str]:
        tok = self.peek()
        if tok[0] != kind:
            raise _ParseError(f"Expected {kind}, got {tok}")
        return self.consume()

    def at_end(self) -> bool:
        return self._tokens[self._pos][0] == _TK_EOF

    def parse_expr(self) -> dict:
        """Parse: additive (('=') additive)*  — '=' has lowest precedence."""
        left = self.parse_additive()
        while self.peek()[0] == _TK_EQUALS:
            self.consume()
            right = self.parse_additive()
            left = {"type": "op", "op": "eq", "args": [left, right]}
        return left

    def parse_additive(self) -> dict:
        """Parse: term (('+' | '-') term)*"""
        left = self.parse_term()
        while self.peek()[0] in (_TK_PLUS, _TK_MINUS):
            op_tok = self.consume()
            right = self.parse_term()
            op = 'add' if op_tok[0] == _TK_PLUS else 'subtract'
            left = {"type": "op", "op": op, "args": [left, right]}
        return left

    def parse_term(self) -> dict:
        """Parse: factor (('*' | cdot | times) factor)*"""
        left = self.parse_factor()
        while self.peek()[0] in (_TK_STAR, _TK_CDOT):
            self.consume()
            right = self.parse_factor()
            left = {"type": "op", "op": "multiply", "args": [left, right]}
        return left

    def parse_factor(self) -> dict:
        """Parse: atom ('^' brace_expr)? ('_' brace_expr)?"""
        atom = self.parse_atom()
        # Handle superscript
        if self.peek()[0] == _TK_CARET:
            self.consume()
            exp = self.parse_brace_or_atom()
            atom = {"type": "op", "op": "pow", "args": [atom, exp]}
        # Handle subscript (attach as metadata, not semantic op)
        if self.peek()[0] == _TK_UNDERSCORE:
            self.consume()
            sub = self.parse_brace_or_atom()
            atom = {"type": "subscript", "base": atom, "sub": sub}
        return atom

    def parse_brace_or_atom(self) -> dict:
        if self.peek()[0] == _TK_LBRACE:
            self.consume()  # '{'
            inner = self.parse_expr()
            if self.peek()[0] == _TK_RBRACE:
                self.consume()
            return inner
        return self.parse_atom()

    def parse_atom(self) -> dict:
        tok_type, tok_val = self.peek()

        if tok_type == _TK_NUM:
            self.consume()
            try:
                return {"type": "literal", "value": float(tok_val)}
            except ValueError:
                raise _ParseError(f"Bad number: {tok_val}")

        if tok_type == _TK_IDENT:
            self.consume()
            return {"type": "field", "name": tok_val}

        if tok_type == _TK_FRAC:
            self.consume()
            num = self.parse_brace_or_atom()
            den = self.parse_brace_or_atom()
            return {"type": "op", "op": "divide", "args": [num, den]}

        if tok_type == _TK_SQRT:
            self.consume()
            arg = self.parse_brace_or_atom()
            return {"type": "func", "func_name": "sqrt", "args": [arg]}

        if tok_type == _TK_LPAREN:
            self.consume()
            inner = self.parse_expr()
            if self.peek()[0] == _TK_RPAREN:
                self.consume()
            return inner

        if tok_type == _TK_LBRACE:
            self.consume()
            inner = self.parse_expr()
            if self.peek()[0] == _TK_RBRACE:
                self.consume()
            return inner

        if tok_type == _TK_MINUS:
            # Unary minus
            self.consume()
            operand = self.parse_factor()
            return {"type": "op", "op": "negate", "args": [operand]}

        raise _ParseError(f"Unexpected token: {tok_type!r} {tok_val!r}")


def _extract_equations(
    text: str, filename: str, warnings: list[str]
) -> list[dict]:
    """Extract all equation environments from stripped text."""

    def line_of(pos: int) -> int:
        return text[:pos].count('\n') + 1

    results: list[dict] = []

    for env in _EQ_ENVS:
        for start, end in _find_env_spans(text, env):
            body = text[start:end]
            ls = line_of(start)
            le = line_of(end)
            trace = {"file": filename, "line_start": ls, "line_end": le}

            # Extract label
            label_m = _LABEL_RE.search(body)
            label = label_m.group(1).strip() if label_m else None

            # Raw LaTeX: strip begin/end tags
            raw = body
            raw = re.sub(r'^\\begin\s*\{[^}]+\}', '', raw).strip()
            raw = re.sub(r'\\end\s*\{[^}]+\}$', '', raw).strip()

            # Try to build AST
            # For align/gather environments, process each sub-equation
            if env.startswith(('align', 'gather', 'eqnarray', 'flalign', 'alignat')):
                # Each row separated by \\
                rows = re.split(r'\\\\', raw)
                if len(rows) == 1:
                    ast = _try_build_ast(raw)
                else:
                    ast = "UNRESOLVED"
            else:
                ast = _try_build_ast(raw)

            if ast == "UNRESOLVED":
                warnings.append(
                    f"{filename}:{ls}: equation env '{env}'"
                    + (f" label={label!r}" if label else "")
                    + " — AST marked UNRESOLVED (complex expression)"
                )

            results.append({
                "label": label,
                "raw_latex": raw,
                "ast": ast,
                "source_trace": trace,
            })

    return results


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

# Row separator rules
_ROW_SEP_RE = re.compile(
    r'\\(?:hline|toprule|midrule|bottomrule|cline\{[^}]*\})'
)

_CAPTION_RE = re.compile(r'\\caption\s*(?:\[[^\]]*\])?\s*\{')

_MULTICOLUMN_RE = re.compile(
    r'\\multicolumn\s*\{(\d+)\}\s*\{[^}]*\}\s*\{'
)


def _extract_caption(text: str) -> Optional[str]:
    """Extract the first \\caption{...} from text."""
    m = _CAPTION_RE.search(text)
    if not m:
        return None
    content, _ = extract_balanced_braces(text, m.end() - 1)
    return normalize_cell(content)


def _split_row(row_text: str) -> list[str]:
    """
    Split a table row on & separators, respecting brace nesting.
    Handles \\multicolumn by expanding into repeated cells.
    """
    cells: list[str] = []
    depth = 0
    current: list[str] = []
    i = 0
    while i < len(row_text):
        ch = row_text[i]
        if ch == '\\' and i + 1 < len(row_text):
            # Check for \\ (row end) or multicolumn
            if row_text[i + 1] == '\\':
                break  # Row terminator
            current.append(ch)
            i += 1
            current.append(row_text[i])
            i += 1
            continue
        if ch == '{':
            depth += 1
            current.append(ch)
        elif ch == '}':
            depth -= 1
            current.append(ch)
        elif ch == '&' and depth == 0:
            cells.append(''.join(current))
            current = []
        else:
            current.append(ch)
        i += 1

    cells.append(''.join(current))

    # Expand \multicolumn{n}{...}{content} into n cells
    expanded: list[str] = []
    for cell in cells:
        mm = _MULTICOLUMN_RE.match(cell.strip())
        if mm:
            span = int(mm.group(1))
            content_start = cell.index('{', cell.index(mm.group(0)) + len(mm.group(0)) - 1)
            content, _ = extract_balanced_braces(cell, content_start)
            val = normalize_cell(content)
            expanded.extend([val] * span)
        else:
            expanded.append(normalize_cell(cell))

    return expanded


def _parse_tabular_body(body: str) -> tuple[list[str], list[list[str]]]:
    """
    Parse the body of a tabular/longtable environment into headers + rows.

    Strategy:
      - Split on \\hline / \\toprule / \\midrule / \\bottomrule
      - First non-empty block of rows (before first separator after content
        starts) = header
      - Remaining blocks = data rows
    """
    # Remove the column-spec argument: first {...} after \begin{tabular}
    # (already stripped by caller)

    # Remove separator commands, splitting into segments
    segments = _ROW_SEP_RE.split(body)
    # Each segment contains zero or more rows separated by \\

    all_row_groups: list[list[list[str]]] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # Split on row terminator \\
        raw_rows = re.split(r'(?<!\\)\\\\', seg)
        group: list[list[str]] = []
        for raw_row in raw_rows:
            raw_row = raw_row.strip()
            if not raw_row:
                continue
            # Skip rows that are entirely formatting
            if re.match(r'^\\[a-zA-Z]+\s*$', raw_row):
                continue
            cells = _split_row(raw_row)
            # Filter out completely empty rows
            if all(c == '' for c in cells):
                continue
            group.append(cells)
        if group:
            all_row_groups.append(group)

    if not all_row_groups:
        return [], []

    # Heuristic: first group = headers if table has multiple groups
    if len(all_row_groups) >= 2:
        header_group = all_row_groups[0]
        # Flatten header group into a single header row (usually 1 row)
        if len(header_group) == 1:
            headers = header_group[0]
        else:
            # Multi-row header: join with '/'
            max_cols = max(len(r) for r in header_group)
            headers = []
            for col in range(max_cols):
                parts = [r[col] for r in header_group if col < len(r)]
                headers.append(' / '.join(p for p in parts if p))
        data_rows = [row for grp in all_row_groups[1:] for row in grp]
    else:
        # Only one group — treat first row as header
        rows = all_row_groups[0]
        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []

    return headers, data_rows


def _extract_tables(
    text: str, filename: str, warnings: list[str]
) -> list[dict]:
    """Extract all table-like environments."""

    def line_of(pos: int) -> int:
        return text[:pos].count('\n') + 1

    results: list[dict] = []

    # Outer containers: table, table*, figure (for captions)
    outer_envs = ('table', 'table*', 'longtable', 'longtable*')
    inner_envs = ('tabular', 'tabular*', 'tabularx', 'tabulary',
                  'longtable', 'longtable*')

    processed_spans: set[tuple[int, int]] = set()

    for outer_env in outer_envs:
        for o_start, o_end in _find_env_spans(text, outer_env):
            outer_body = text[o_start:o_end]
            caption = _extract_caption(outer_body)
            ls = line_of(o_start)
            le = line_of(o_end)
            trace = {"file": filename, "line_start": ls, "line_end": le}

            # Find tabular inside
            found_inner = False
            for inner_env in inner_envs:
                for i_start, i_end in _find_env_spans(outer_body, inner_env):
                    abs_start = o_start + i_start
                    abs_end = o_start + i_end
                    if (abs_start, abs_end) in processed_spans:
                        continue
                    processed_spans.add((abs_start, abs_end))

                    inner_body = outer_body[i_start:i_end]
                    # Strip \begin{env}{col_spec}
                    inner_body = re.sub(
                        r'^\\begin\s*\{[^}]+\}\s*(?:\[[^\]]*\])?\s*\{[^}]*\}', '',
                        inner_body
                    ).strip()
                    # Strip \end{env}
                    inner_body = re.sub(r'\\end\s*\{[^}]+\}$', '', inner_body).strip()

                    try:
                        headers, rows = _parse_tabular_body(inner_body)
                    except Exception as exc:
                        warnings.append(
                            f"{filename}:{ls}: failed to parse tabular body: {exc}"
                        )
                        headers, rows = [], []

                    results.append({
                        "caption": caption,
                        "headers": headers,
                        "rows": rows,
                        "source_trace": trace,
                    })
                    found_inner = True
                    break
                if found_inner:
                    break

            if not found_inner:
                # Outer env with no inner tabular (longtable is itself tabular)
                pass

    # Standalone tabular / longtable not inside table environment
    for inner_env in inner_envs:
        for i_start, i_end in _find_env_spans(text, inner_env):
            if (i_start, i_end) in processed_spans:
                continue
            processed_spans.add((i_start, i_end))

            ls = line_of(i_start)
            le = line_of(i_end)
            trace = {"file": filename, "line_start": ls, "line_end": le}

            body = text[i_start:i_end]
            body = re.sub(
                r'^\\begin\s*\{[^}]+\}\s*(?:\[[^\]]*\])?\s*\{[^}]*\}', '',
                body
            ).strip()
            body = re.sub(r'\\end\s*\{[^}]+\}$', '', body).strip()

            try:
                headers, rows = _parse_tabular_body(body)
            except Exception as exc:
                warnings.append(
                    f"{filename}:{ls}: failed to parse standalone tabular: {exc}"
                )
                headers, rows = [], []

            results.append({
                "caption": None,
                "headers": headers,
                "rows": rows,
                "source_trace": trace,
            })

    return results


# ---------------------------------------------------------------------------
# Section hierarchy extraction
# ---------------------------------------------------------------------------

_SECTION_LEVELS = ['section', 'subsection', 'subsubsection', 'paragraph']
_LEVEL_RANK = {lvl: i for i, lvl in enumerate(_SECTION_LEVELS)}

# Matches \section[short]{title} or \section{title} or \section*{title}
_SECTION_RE = re.compile(
    r'\\(section|subsection|subsubsection|paragraph)\*?\s*'
    r'(?:\[[^\]]*\])?\s*'  # optional short title
    r'\{',
)


def _extract_sections(
    text: str, filename: str, raw_lines: list[str]
) -> list[dict]:
    """Extract document structure sections from text."""
    results: list[dict] = []

    def line_of(pos: int) -> int:
        return text[:pos].count('\n') + 1

    for m in _SECTION_RE.finditer(text):
        level = m.group(1)
        brace_start = m.end() - 1  # points to '{'
        title_raw, end = extract_balanced_braces(text, brace_start)
        title = normalize_cell(title_raw)
        ls = line_of(m.start())

        results.append({
            "level": level,
            "title": title,
            "file": filename,
            "line_start": ls,
            "line_end": -1,   # resolved below
        })

    # Resolve line_end: each section ends where the next same-or-higher-level starts
    for i, sec in enumerate(results):
        my_rank = _LEVEL_RANK.get(sec["level"], 99)
        for j in range(i + 1, len(results)):
            next_rank = _LEVEL_RANK.get(results[j]["level"], 99)
            if next_rank <= my_rank:
                sec["line_end"] = results[j]["line_start"] - 1
                break

    return results


# ---------------------------------------------------------------------------
# Main parser orchestration
# ---------------------------------------------------------------------------


def parse_source_dir(
    source_dir: Path, warnings: list[str]
) -> tuple[list[Path], list[dict], list[dict], list[dict], list[dict]]:
    """
    Orchestrate parsing of all .tex files.

    Returns (files, symbols, equations, tables, sections).
    """
    files = resolve_input_files(source_dir, warnings)
    if not files:
        # Fallback: all .tex files
        files = sorted(source_dir.rglob('*.tex'))

    all_symbols: list[dict] = []
    all_equations: list[dict] = []
    all_tables: list[dict] = []
    all_sections: list[dict] = []

    for tex_path in files:
        rel = str(tex_path.relative_to(source_dir))
        raw_lines = read_file_lines(tex_path)
        if not raw_lines:
            warnings.append(f"Empty or unreadable file: {tex_path}")
            continue

        raw_text = '\n'.join(raw_lines)
        stripped = strip_comments(raw_text)

        try:
            syms = _extract_symbols(stripped, rel, raw_lines)
            all_symbols.extend(syms)
        except Exception as exc:
            warnings.append(f"{rel}: symbol extraction error: {exc}")

        try:
            eqs = _extract_equations(stripped, rel, warnings)
            all_equations.extend(eqs)
        except Exception as exc:
            warnings.append(f"{rel}: equation extraction error: {exc}")

        try:
            tbls = _extract_tables(stripped, rel, warnings)
            all_tables.extend(tbls)
        except Exception as exc:
            warnings.append(f"{rel}: table extraction error: {exc}")

        try:
            secs = _extract_sections(stripped, rel, raw_lines)
            all_sections.extend(secs)
        except Exception as exc:
            warnings.append(f"{rel}: section extraction error: {exc}")

    return files, all_symbols, all_equations, all_tables, all_sections


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Parse LaTeX source files from an arXiv tarball and extract "
            "symbol definitions, equations, tables, and document structure "
            "into a structured JSON file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Parse a single paper tarball (already extracted):
  python tex_parser.py --source-dir ./arxiv_src/ --output extracted.json

  # Verbose output to see warnings inline:
  python tex_parser.py --source-dir ./arxiv_src/ --output extracted.json --verbose
""",
    )
    p.add_argument(
        '--source-dir',
        required=True,
        metavar='DIR',
        help='Directory containing .tex files (from arXiv tarball).',
    )
    p.add_argument(
        '--output',
        required=True,
        metavar='FILE',
        help='Path for output JSON file.',
    )
    p.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print extraction warnings to stderr during processing.',
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    source_dir = Path(args.source_dir).resolve()
    output_path = Path(args.output).resolve()

    if not source_dir.exists():
        print(f"ERROR: source-dir does not exist: {source_dir}", file=sys.stderr)
        return 1

    if not source_dir.is_dir():
        print(f"ERROR: source-dir is not a directory: {source_dir}", file=sys.stderr)
        return 1

    warnings: list[str] = []

    print(f"Parsing LaTeX sources in: {source_dir}", file=sys.stderr)

    files, symbols, equations, tables, sections = parse_source_dir(
        source_dir, warnings
    )

    rel_files = []
    for f in files:
        try:
            rel_files.append(str(f.relative_to(source_dir)))
        except ValueError:
            rel_files.append(str(f))

    output = {
        "source_dir": str(source_dir),
        "files_processed": rel_files,
        "symbols": symbols,
        "equations": equations,
        "tables": tables,
        "sections": sections,
        "extraction_warnings": warnings,
    }

    if args.verbose:
        for w in warnings:
            print(f"WARNING: {w}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(
        f"Done. Processed {len(rel_files)} file(s). "
        f"Extracted: {len(symbols)} symbols, {len(equations)} equations, "
        f"{len(tables)} tables, {len(sections)} sections. "
        f"{len(warnings)} warning(s).",
        file=sys.stderr,
    )
    print(f"Output written to: {output_path}", file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
