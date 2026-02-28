# Paper→Repo Executable Spec Compiler — Design Document

**Date**: 2026-02-18
**Status**: Implemented (v0.1.0)
**Location**: `paper-spec-compiler/` (Claude Code plugin in repo root)

## Problem

When implementing ML/RL papers, engineers silently diverge from the paper's actual
specifications. Constants get "simplified," reward terms get approximated, privileged
information leaks into execution, and domain randomization ranges drift. There is no
automated mechanism to detect these divergences.

## Solution

Treat academic papers as typed intermediate representations (IR) and compile them into
machine-readable specifications with compliance tests. The "paper drift detector" breaks
CI when code diverges from spec.

## Architecture

```
Paper (LaTeX/PDF) → [extract/] → IR (Pydantic) → [emit/] → spec.yaml + spec.md
                                      ↓
                                 [compile/] → config overrides + diff reports
                                      ↓
                                 [tests_gen/] → pytest compliance tests
```

### Plugin Structure

```
paper-spec-compiler/
├── .claude-plugin/plugin.json
├── .gitignore
└── skills/spec-compiler/
    ├── SKILL.md              (~1,050 words, core workflow)
    ├── references/           (6 files, ~3,600 words)
    │   ├── spec-yaml-schema.md
    │   ├── ir-entities.md
    │   ├── extraction-strategy.md
    │   ├── test-generation.md
    │   ├── baseline-linking.md
    │   └── reproducibility-checklist.md
    ├── scripts/              (9 Python scripts)
    │   ├── arxiv_fetch.py
    │   ├── tex_parser.py
    │   ├── ir_schema.py
    │   ├── emit_yaml.py
    │   ├── emit_md.py
    │   ├── gen_tests.py
    │   ├── compile_configs.py
    │   ├── diff_spec.py
    │   └── validate_spec.py
    └── assets/               (2 templates)
        ├── spec-yaml-template.yaml
        └── spec-lock-template.json
```

### Output Artifacts

| Artifact | Purpose |
|----------|---------|
| `spec.yaml` | Canonical machine-readable truth |
| `spec.md` | Human-readable spec with paper traceability links |
| `spec.lock.json` | Frozen hashes + provenance |
| `tests/spec_compliance/` | Generated pytest compliance tests |
| `reports/spec_diff.md` | Mismatch report (spec vs code) |
| `spec.resolved.yaml` | Flattened spec with baseline imports resolved |
| `spec.patch.yaml` | Only deltas from baselines |

### Key Design Decisions

1. **UNRESOLVED sentinel**: Fields that cannot be extracted are marked "UNRESOLVED"
   and generate failing test stubs. This is the anti-drift mechanism — never guess.

2. **Informed-POMDP as first-class**: Training-only privileged information is
   structurally distinct from execution observations, matching Informed Dreamer
   conventions.

3. **Baseline linking**: Papers inherit from prior work via `imports:` in spec.yaml.
   SkyDreamer = DreamerV3 + Informed Dreamer + deltas. Prevents drift from
   inherited defaults.

4. **Expression ASTs**: Reward terms and termination conditions are stored as
   structured expression trees, not strings. Enables automated evaluation in tests.

5. **Progressive disclosure**: SKILL.md stays lean (~1,050 words). Detailed schema,
   extraction patterns, and test templates live in `references/`.

## Extraction Priority

1. LaTeX sources (best) — parse TeX AST for symbols, equations, tables
2. PDF fallback — layout-aware extraction, UNRESOLVED for unreliable math
3. HTML (last resort) — ar5iv, lower fidelity

## Spec YAML Schema Sections

meta, frames, spaces, timing, reward, termination, gates_track, dynamics,
perception, world_model, actor_critic, training, evaluation, deployment, imports

Full schema documented in `references/spec-yaml-schema.md`.

## Compliance Test Categories

1. Space shape/dtype checks
2. Reward expression evaluation
3. Termination condition logic
4. Domain randomization ranges
5. Timing/delay correctness
6. Informed-POMDP key gating
7. UNRESOLVED stubs (failing, blocking)

## Dependencies

- Python 3.10+
- pydantic >= 2.0
- PyYAML
- requests (for arxiv_fetch.py only)
