# BrainAI System Orchestrator + Feature-Flag Wiring — Design Document

**Date**: 2026-02-18
**Status**: Skill implemented (v0.1.0), orchestrator refactor pending
**Location**: `brain-ai-dev/` (Claude Code plugin in repo root)

## Problem

The current `brain_ai/system.py` uses scattered `if/else` conditionals for feature
flags, a minimal `SystemOutput` dataclass, no typed pipeline contracts, no state
container, and no telemetry. This makes the system fragile: disabling a module can
break downstream stages, state lives hidden inside modules, and debugging requires
ad-hoc print statements.

## Solution

Refactor the orchestrator into a typed, deterministic pipeline with:
- `PipelinePlan` built once at init (no scattered conditionals)
- Typed contracts at every module boundary (`EncoderOutput`, `WorkspaceOutput`, etc.)
- Bypass adapters for every feature flag (disabled modules produce valid neutral output)
- Consolidated `BrainAIState` for all stateful modules
- Structured `TelemetrySink` for debugging and monitoring
- Deterministic execution (fixed modality order, stable top-k, per-module RNG)

## Plugin Structure

```
brain-ai-dev/
├── .claude-plugin/plugin.json
├── .gitignore
└── skills/system-orchestrator/
    ├── SKILL.md              (~183 lines, core architecture guide)
    ├── references/           (7 files, ~1,570 lines)
    │   ├── type-contracts.md
    │   ├── pipeline-plan.md
    │   ├── feature-flags.md
    │   ├── state-management.md
    │   ├── telemetry.md
    │   ├── reproducibility.md
    │   └── testing-matrix.md
    ├── scripts/              (3 Python scripts, ~1,371 lines)
    │   ├── validate_contracts.py
    │   ├── gen_flag_tests.py
    │   └── deps_report.py
    └── assets/
        └── types_template.py (starter template for brain_ai/types.py)
```

## Concrete Deliverables (Code Changes)

When the skill is used to guide the refactor:

| File | Role |
|------|------|
| `brain_ai/types.py` | All typed contracts (from `assets/types_template.py`) |
| `brain_ai/pipeline.py` | `PipelinePlan`, `Stage`, `PipelineContext` |
| `brain_ai/system.py` | Refactored `BrainAI` using PipelinePlan |
| `brain_ai/utils/repro.py` | `set_global_seed`, `stable_topk`, manifest helpers |
| `tests/test_system_orchestrator.py` | Generated via `gen_flag_tests.py` |

## Key Design Decisions

1. **PipelinePlan over conditionals**: Build stage list once at init, execute in forward
2. **Bypass adapters**: Every flag has a bypass producing schema-identical neutral output
3. **BrainAIState**: Single serializable container for all module state
4. **Two-tier details**: Cheap metrics always collected; heavy traces only on request
5. **TelemetrySink**: Structured events, not print statements
6. **Per-module RNG**: Keyed by (global_seed, module_name) for reproducibility
7. **Contract assertions**: Cheap runtime checks, disableable in production

## Dependencies

- Python 3.10+
- torch >= 2.0
- brain_ai package (the code being orchestrated)
