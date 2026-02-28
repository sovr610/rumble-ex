# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

brain-ai-dev is a Claude Code plugin containing 43 skills that guide development of the `brain_ai` cognitive architecture. Each skill encodes domain-specific knowledge (spiking neural networks, world models, encoder design, training orchestration, etc.) and activates based on trigger phrases in user prompts. The plugin does not contain the architecture itself -- it provides structured guidance, templates, and validation tooling for building it.

The parent project lives at `/home/sovr610/human-brain/` with `brain_ai/` as the main Python package. This plugin is at `/home/sovr610/human-brain/brain-ai-dev/`.

## Plugin Structure

```
brain-ai-dev/
  .claude-plugin/plugin.json   # Manifest listing all 43 skills
  skills/<name>/
    SKILL.md                   # Skill definition with YAML frontmatter
    references/                # Detailed reference docs (theory, implementation, testing)
    assets/                    # Python template files with self-test blocks
    scripts/                   # Validation, test generation, and benchmark scripts
```

- **plugin.json** -- declares plugin name, version, description, and the ordered list of skill paths.
- **SKILL.md** -- the primary file Claude Code reads when a skill activates. Contains YAML frontmatter (name, description with trigger phrases, version) and a structured body.
- **references/** -- deep-dive documents split out from SKILL.md to keep it concise. Typical files: theory docs, implementation specs, testing matrices, state management guides.
- **assets/** -- Python template files intended to be adapted into `brain_ai/` modules. Each must include an `if __name__ == "__main__"` self-test block.
- **scripts/** -- runnable Python scripts. Standard set: `validate_*.py` (runtime contract validation), `gen_*_tests.py` (pytest test generation), `*_benchmark.py` or `*_diagnostic.py` (profiling/debug tools).

## SKILL.md Conventions

### Frontmatter

```yaml
---
name: BrainAI <Module Name>
description: >
  This skill should be used when the user asks to "trigger phrase 1",
  "trigger phrase 2", ... or mentions ClassName, file_name,
  or concept_name in the BrainAI cognitive architecture.
version: 0.1.0
---
```

- `name` and `description` must be in third person ("This skill should be used when...").
- `description` must list specific trigger phrases in quotes so the plugin router can match them.

### Body

- Use imperative/infinitive form ("Enforce correctness", "Guide the design"), not second person ("You should...").
- Target 1500-2000 words. Move detailed content (equations, full schemas, test matrices) into `references/`.
- Standard sections: Purpose, Key Files, Core Architecture/Contract, Anti-Patterns, Additional Resources (Reference Files, Scripts, Assets).

## Architecture Mapping

Skills map to `brain_ai` package modules:

| Skill | Target Module |
|-------|--------------|
| system-orchestrator | `brain_ai/system.py`, `brain_ai/pipeline.py`, `brain_ai/types.py` |
| spiking-core | `brain_ai/core/snn.py`, `brain_ai/core/neurons.py`, `brain_ai/core/surrogates.py` |
| encoder-suite | `brain_ai/encoders/` |
| global-workspace-ignition | `brain_ai/workspace/` |
| dual-process-reasoning | `brain_ai/reasoning/` |
| active-inference-agent | `brain_ai/decision/` |
| engram-conditional-memory | `brain_ai/memory/` |
| meta-learning-suite | `brain_ai/meta/` |
| htm-spatial-temporal-reflex | `brain_ai/temporal/` |
| training-orchestrator | `brain_ai/training/` |
| dreamerv3-rssm, world-model-scaffold | World model subsystem |
| vjepa2-* | V-JEPA2 vision transformer subsystem |

## Important Rules

### Use `module.train(False)` instead of the `.e` + `val()` method

A security hook blocks the standard PyTorch evaluation-mode method call. All asset templates and generated code must use `module.train(False)` to switch a module to evaluation mode. This applies everywhere: self-tests, validation scripts, and any code generated for the `brain_ai` package.

### Asset self-tests are mandatory

Every Python file in `assets/` must end with an `if __name__ == "__main__"` block that runs self-contained validation tests. These tests should print pass/fail results and exit with code 1 on failure.

### Reference file coverage

Reference documents should cover three areas: theory/background, implementation specification, and testing criteria (typically in a `testing-matrix.md`).

### Script naming conventions

Each skill's `scripts/` directory follows a standard pattern:
- `validate_<domain>.py` -- runtime contract checks
- `gen_<domain>_tests.py` -- generates parameterized pytest suites
- `<domain>_benchmark.py` or `<domain>_diagnostic.py` -- profiling and debug reporting

## Testing

Scripts in each skill's `scripts/` directory generate pytest tests targeting the `brain_ai` package. Run from the parent project root:

```bash
pytest tests/ -v
```

Individual skill validation scripts can be run directly:

```bash
python skills/<name>/scripts/validate_<domain>.py
python skills/<name>/scripts/validate_<domain>.py --json    # machine-readable output
python skills/<name>/scripts/validate_<domain>.py --verbose # detailed output
```

## Key Type Contracts

The architecture enforces typed contracts at every module boundary. The central ones:

- `EncoderOutput(feats=(B,T,D), mask=(B,T), salience, time, spike, aux)` -- all encoders produce this
- `SystemOutput(output, confidence, details)` -- final pipeline output
- `BrainAIState(wm_state, htm_state, snn_state, belief_state, eligibility_state, rng_state)` -- consolidated stateful module state
- `PipelineContext` / `PipelinePlan` with `Stage` objects -- orchestrator execution plan

All tensors at module boundaries must match `workspace_dim` for dimension D, share the same device, and avoid silent dtype drift.
