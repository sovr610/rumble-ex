---
name: Paper-to-Spec Compiler
description: >
  This skill should be used when the user asks to "extract a spec from a paper",
  "compile a paper into spec.yaml", "generate compliance tests from a paper",
  "create an executable spec", "parse arXiv paper into config", "detect paper drift",
  "diff code against paper", "generate spec from LaTeX", "validate a spec",
  "check if code matches the paper", or mentions converting academic ML/RL papers
  into machine-readable specifications. Treats papers as typed intermediate
  representations and emits spec.yaml, spec.md, compliance tests, and drift reports.
version: 0.1.0
---

# Paper→Repo Executable Spec Compiler

## Purpose

Convert academic ML/RL papers into machine-readable executable specifications. Treat
each paper as a typed intermediate representation (IR), extract every symbol, constant,
and structural choice that matters, and emit:

- **spec.yaml** — canonical machine-readable truth
- **spec.md** — human-readable spec with traceability links to paper locations
- **spec.lock.json** — frozen hashes + provenance (paper version, commit, extraction timestamp)
- **tests/spec_*** — generated compliance tests ("paper drift detector")
- **reports/spec_diff.md** — mismatch report between spec and existing code/config

## Core Workflow

### Phase 1: Fetch Paper Sources

Run `scripts/arxiv_fetch.py` with the arXiv ID or URL. This downloads:
- PDF (always available)
- LaTeX source tarball (when available — strongly preferred)
- HTML fallback (last resort)

```bash
python scripts/arxiv_fetch.py --arxiv-id 2510.14783 --output-dir .paper_sources/
```

### Phase 2: Extract Into IR

Priority order of truth sources (non-negotiable):

1. **LaTeX sources** (best) — parse TeX AST for symbol definitions, equations, tables,
   figure captions. Run `scripts/tex_parser.py`.
2. **PDF fallback** — layout-aware extraction for tables + math blocks. If math is not
   reliably parseable, mark as UNRESOLVED.
3. **HTML** (last resort) — often drops math fidelity.

**Critical rule**: If extraction cannot determine a numeric constant or inequality
threshold, mark it `UNRESOLVED` and generate a blocking TODO in spec.md plus a failing
test stub. Never silently substitute a guess.

```bash
python scripts/tex_parser.py --source-dir .paper_sources/ --output ir_output.json
```

The IR is defined by Pydantic models in `scripts/ir_schema.py`. Consult
**`references/ir-entities.md`** for the full type system.

### Phase 3: Validate and Normalize IR

Load the extracted IR into Pydantic models for validation:
- Check all required fields are present or marked UNRESOLVED
- Normalize units, coordinate frame conventions, naming
- Flag informed-POMDP split (training-only vs execution fields)

### Phase 4: Emit Spec Artifacts

Generate outputs from the validated IR:

```bash
python scripts/emit_yaml.py --ir ir_output.json --output spec.yaml
python scripts/emit_md.py --ir ir_output.json --output spec.md
```

The spec.yaml schema is documented in **`references/spec-yaml-schema.md`**.

### Phase 5: Generate Compliance Tests

```bash
python scripts/gen_tests.py --spec spec.yaml --output-dir tests/spec_compliance/
```

Test categories (see **`references/test-generation.md`** for patterns):
- Space shape/dtype checks
- Reward expression evaluation against synthetic transitions
- Termination condition boolean logic
- Domain randomization range enforcement
- Timing/delay correctness
- Informed-POMDP key gating

### Phase 6: Diff Against Existing Code (Optional)

If a repo already has an implementation:

```bash
python scripts/diff_spec.py --spec spec.yaml --repo-root . --output reports/spec_diff.md
python scripts/compile_configs.py --spec spec.yaml --format dreamerv3 --output config_overrides.yaml
```

## UNRESOLVED Marking Protocol

This is the primary anti-drift mechanism. When extraction fails:

1. Set the field value to `"UNRESOLVED"` in spec.yaml
2. Add a `# TODO(spec): <description> [paper §X.Y / Table Z]` in spec.md
3. Generate a failing test stub: `test_UNRESOLVED_<field_name>` that raises
   `pytest.skip("UNRESOLVED: <field> — manual extraction required")`
4. Block spec.lock.json finalization until all UNRESOLVEDs are resolved

## Baseline Linking

For papers that extend prior work (e.g., SkyDreamer extends DreamerV3 + Informed Dreamer):

- spec.yaml supports `imports: [dreamerv3@<commit>, informed_dreamer@<commit>]`
- Emit `spec.resolved.yaml` (fully flattened) + `spec.patch.yaml` (only deltas)
- This prevents drift from upstream defaults the paper authors relied on

See **`references/baseline-linking.md`** for the import mechanism.

## Reproducibility Payload

The compiler auto-generates a reproducibility appendix in spec.md covering:
- Full hyperparameter list with selection method
- Compute infrastructure and runtime
- Dataset/environment versioning
- Evaluation metrics, error bars, number of seeds

See **`references/reproducibility-checklist.md`** for checklist items.

## Additional Resources

### Reference Files

- **`references/spec-yaml-schema.md`** — Full spec.yaml schema with all sections and field types
- **`references/ir-entities.md`** — IR type system (Pydantic models, entity definitions)
- **`references/extraction-strategy.md`** — Detailed extraction patterns for LaTeX/PDF/HTML
- **`references/test-generation.md`** — Compliance test patterns and examples
- **`references/baseline-linking.md`** — Import mechanism for composing specs from prior work
- **`references/reproducibility-checklist.md`** — Auto-generated reproducibility appendix items

### Scripts

- **`scripts/arxiv_fetch.py`** — Download paper PDF + LaTeX sources from arXiv
- **`scripts/tex_parser.py`** — Parse LaTeX AST for tables, equations, symbols
- **`scripts/ir_schema.py`** — Pydantic IR models with validation
- **`scripts/emit_yaml.py`** — Deterministic spec.yaml emitter
- **`scripts/emit_md.py`** — spec.md emitter with traceability links
- **`scripts/gen_tests.py`** — Generate pytest compliance tests from spec
- **`scripts/compile_configs.py`** — Map spec → framework config overrides
- **`scripts/diff_spec.py`** — Diff repo code/config against spec
- **`scripts/validate_spec.py`** — Validate spec.yaml against schema

### Assets

- **`assets/spec-yaml-template.yaml`** — Skeleton spec.yaml with all sections
- **`assets/spec-lock-template.json`** — Lock file template with provenance fields
