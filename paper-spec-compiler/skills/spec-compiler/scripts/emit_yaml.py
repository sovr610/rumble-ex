#!/usr/bin/env python3
"""
emit_yaml.py — Deterministic YAML emitter for the paper-spec compiler.

Converts an IR JSON file (produced by tex_parser.py and populated into the
ir_schema.py SpecIR model) into a canonical spec.yaml file.  Also writes
a spec.lock.json sidecar with provenance and integrity information.

Design goals
------------
* Deterministic: identical input always produces byte-identical output.
  Keys are sorted within each section; section order is fixed by SECTION_ORDER.
* UNRESOLVED is explicit: any field the extractor could not determine is
  emitted as the string "UNRESOLVED", never omitted.
* Baseline linking: --import flags attach previously-compiled spec.yaml files
  to the imports section so downstream tools can resolve inherited values.
* Lock file: spec.lock.json records the paper arXiv ID, extraction timestamp,
  SHA-256 of the emitted YAML, and the count of UNRESOLVED fields.

Usage
-----
    python emit_yaml.py --ir ir_output.json --output spec.yaml
    python emit_yaml.py --ir ir_output.json --output spec.yaml \\
        --import baseline/dreamerv3/spec.yaml \\
        --import baseline/informed_dreamer/spec.yaml

The --import flag may be repeated for multiple baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------

TOOL_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Section ordering — meta first, imports last.
# ---------------------------------------------------------------------------

SECTION_ORDER = [
    "meta",
    "frames",
    "spaces",
    "timing",
    "reward",
    "termination",
    "gates_track",
    "dynamics",
    "perception",
    "world_model",
    "actor_critic",
    "training",
    "evaluation",
    "deployment",
    "imports",
]

# Keywords in table captions that hint at which IR section the table belongs to.
# Maps lowercase keywords → IR section path fragment.
TABLE_CAPTION_HINTS: list[tuple[str, str]] = [
    ("hyperparameter", "training"),
    ("hyper-parameter", "training"),
    ("training", "training"),
    ("optimizer", "training"),
    ("network architecture", "world_model"),
    ("architecture", "world_model"),
    ("simulation parameter", "dynamics"),
    ("physical parameter", "dynamics"),
    ("dynamics parameter", "dynamics"),
    ("parameter", "dynamics"),
    ("domain randomization", "dynamics"),
    ("randomization", "dynamics"),
    ("reward", "reward"),
    ("observation", "spaces"),
    ("state space", "spaces"),
    ("action space", "spaces"),
    ("timing", "timing"),
    ("delay", "timing"),
    ("evaluation", "evaluation"),
    ("metric", "evaluation"),
    ("ablation", "evaluation"),
    ("deployment", "deployment"),
    ("hardware", "deployment"),
]

# Keywords in equation labels / section titles that hint at IR section.
EQUATION_LABEL_HINTS: list[tuple[str, str]] = [
    ("reward", "reward"),
    ("rew", "reward"),
    ("termination", "termination"),
    ("term", "termination"),
    ("dynamics", "dynamics"),
    ("dyn", "dynamics"),
    ("motion", "dynamics"),
    ("kinematics", "dynamics"),
    ("loss", "training"),
    ("objective", "training"),
    ("policy", "actor_critic"),
    ("actor", "actor_critic"),
    ("critic", "actor_critic"),
    ("value", "actor_critic"),
    ("world_model", "world_model"),
    ("rssm", "world_model"),
    ("observation", "spaces"),
]


# ---------------------------------------------------------------------------
# Custom YAML representer for deterministic, readable output
# ---------------------------------------------------------------------------


def _represent_str(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    """Emit strings; use block style for multi-line, literal for UNRESOLVED."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def _represent_float(dumper: yaml.Dumper, data: float) -> yaml.ScalarNode:
    """Emit floats without scientific notation where possible."""
    if data != data:  # NaN
        return dumper.represent_scalar("tag:yaml.org,2002:float", ".nan")
    if data == float("inf"):
        return dumper.represent_scalar("tag:yaml.org,2002:float", ".inf")
    if data == float("-inf"):
        return dumper.represent_scalar("tag:yaml.org,2002:float", "-.inf")
    # Use repr to avoid floating-point drift; strip trailing zeros for readability.
    s = repr(data)
    return dumper.represent_scalar("tag:yaml.org,2002:float", s)


def _represent_ordered_dict(
    dumper: yaml.Dumper, data: dict
) -> yaml.MappingNode:
    """Emit dicts in insertion order (already sorted by our code)."""
    return dumper.represent_mapping(
        "tag:yaml.org,2002:map",
        list(data.items()),
        flow_style=False,
    )


class _SpecDumper(yaml.Dumper):
    """Custom YAML dumper for spec files."""
    pass


_SpecDumper.add_representer(str, _represent_str)
_SpecDumper.add_representer(float, _represent_float)
_SpecDumper.add_representer(dict, _represent_ordered_dict)
_SpecDumper.add_representer(
    type(None),
    lambda d, _: d.represent_scalar("tag:yaml.org,2002:null", "null"),
)


def _dump_yaml(data: Any) -> str:
    """Serialise *data* to a YAML string using the spec dumper."""
    return yaml.dump(
        data,
        Dumper=_SpecDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,  # We control ordering explicitly.
        width=120,
    )


# ---------------------------------------------------------------------------
# Source trace → human-readable citation string
# ---------------------------------------------------------------------------


def _format_source(source: dict | None) -> str:
    """Convert a SourceTrace dict to a compact citation string."""
    if not source:
        return "UNRESOLVED"
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
    if source.get("tex_file") and not parts:
        parts.append(source["tex_file"])
    if source.get("line_range") and not parts:
        lr = source["line_range"]
        parts.append(f"lines {lr[0]}-{lr[1]}")
    return ", ".join(parts) if parts else "UNRESOLVED"


# ---------------------------------------------------------------------------
# UNRESOLVED counting helpers
# ---------------------------------------------------------------------------


def _count_unresolved(value: Any) -> int:
    """Recursively count leaf values equal to the string 'UNRESOLVED'."""
    if isinstance(value, str):
        return 1 if value == "UNRESOLVED" else 0
    if isinstance(value, dict):
        return sum(_count_unresolved(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_unresolved(item) for item in value)
    return 0


def _collect_unresolved_paths(
    value: Any, path: str, results: list[tuple[str, str]]
) -> None:
    """Recursively collect (json_path, section) pairs for UNRESOLVED fields."""
    if isinstance(value, str) and value == "UNRESOLVED":
        # Determine section from path prefix.
        section = path.split(".")[0] if path else "unknown"
        results.append((path, section))
    elif isinstance(value, dict):
        for k, v in value.items():
            child_path = f"{path}.{k}" if path else k
            _collect_unresolved_paths(v, child_path, results)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            child_path = f"{path}[{i}]"
            _collect_unresolved_paths(item, child_path, results)


# ---------------------------------------------------------------------------
# Priority helpers for UNRESOLVED fields
# ---------------------------------------------------------------------------

_HIGH_PRIORITY_SECTIONS = {"reward", "termination", "dynamics", "spaces"}
_MEDIUM_PRIORITY_SECTIONS = {"training", "actor_critic", "world_model", "timing"}


def _unresolved_priority(section: str) -> str:
    if section in _HIGH_PRIORITY_SECTIONS:
        return "HIGH"
    if section in _MEDIUM_PRIORITY_SECTIONS:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Table → IR section heuristic
# ---------------------------------------------------------------------------


def _classify_table_caption(caption: str | None) -> str:
    """Return the best-guess IR section name for a table given its caption."""
    if not caption:
        return "dynamics"  # Safe default; parameters tables most common.
    lower = caption.lower()
    for keyword, section in TABLE_CAPTION_HINTS:
        if keyword in lower:
            return section
    return "dynamics"


def _classify_equation_label(label: str | None) -> str:
    """Return the best-guess IR section name for an equation given its label."""
    if not label:
        return "dynamics"
    lower = label.lower()
    for keyword, section in EQUATION_LABEL_HINTS:
        if keyword in lower:
            return section
    return "dynamics"


# ---------------------------------------------------------------------------
# IR → spec dict mappers (one per top-level section)
# ---------------------------------------------------------------------------


def _map_meta(ir: dict) -> dict:
    """Map IR meta section → spec.yaml meta section."""
    meta_ir = ir.get("meta", {})
    paper_ir = meta_ir.get("paper", {})
    sources_ir = meta_ir.get("sources", {})

    paper: dict = {
        "title": paper_ir.get("title", "UNRESOLVED"),
        "arxiv": paper_ir.get("arxiv", "UNRESOLVED"),
        "version": paper_ir.get("version", "UNRESOLVED"),
        "doi": paper_ir.get("doi", None),
        "authors": paper_ir.get("authors", []),
    }

    sources: dict = {
        "prefer": sources_ir.get("prefer", ["arxiv_tex", "pdf", "html"]),
        "tex_hash": sources_ir.get("tex_hash", None),
        "pdf_hash": sources_ir.get("pdf_hash", None),
    }

    return {
        "paper": paper,
        "sources": sources,
        "generated_at": meta_ir.get(
            "generated_at",
            datetime.now(timezone.utc).isoformat(),
        ),
        "tool_version": meta_ir.get("tool_version", TOOL_VERSION),
        "unresolved_count": meta_ir.get("unresolved_count", 0),
    }


def _map_frames(ir: dict) -> dict | None:
    """Map IR frames section → spec.yaml frames section."""
    frames_ir = ir.get("frames")
    if not frames_ir:
        return None

    transforms = []
    for t in frames_ir.get("transforms", []):
        transforms.append(
            {
                "from": t.get("from") or t.get("from_frame", "UNRESOLVED"),
                "to": t.get("to") or t.get("to_frame", "UNRESOLVED"),
                "type": t.get("type", "UNRESOLVED"),
                "description": t.get("description", "UNRESOLVED"),
            }
        )

    result: dict = {
        "convention": frames_ir.get("convention", "UNRESOLVED"),
        "frames": frames_ir.get("frames", []),
    }
    if transforms:
        result["transforms"] = transforms
    if frames_ir.get("source"):
        result["source"] = _format_source(frames_ir["source"])
    return result


def _map_space_field(field: dict) -> dict:
    """Map a SpaceField dict to spec format."""
    return {
        "name": field.get("name", "UNRESOLVED"),
        "dtype": field.get("dtype", "UNRESOLVED"),
        "shape": field.get("shape", ["UNRESOLVED"]),
        "units": field.get("units"),
        "source": _format_source(field.get("source")),
    }


def _map_informed_field(field: dict) -> dict:
    """Map an InformedField dict to spec format."""
    result = _map_space_field(field)
    if field.get("informed_dreamer_key"):
        result["informed_dreamer_key"] = field["informed_dreamer_key"]
    result["training_only"] = field.get("training_only", True)
    return result


def _map_spaces(ir: dict) -> dict | None:
    """Map IR spaces section → spec.yaml spaces section."""
    spaces_ir = ir.get("spaces")
    if not spaces_ir:
        return None

    result: dict = {}

    obs_exec = spaces_ir.get("observation_exec", [])
    if obs_exec:
        result["observation_exec"] = [_map_space_field(f) for f in obs_exec]
    else:
        result["observation_exec"] = []

    info_train = spaces_ir.get("information_train", [])
    if info_train:
        result["information_train"] = [_map_informed_field(f) for f in info_train]
    else:
        result["information_train"] = []

    action = spaces_ir.get("action")
    if action:
        result["action"] = {
            "name": action.get("name", "UNRESOLVED"),
            "dtype": action.get("dtype", "UNRESOLVED"),
            "shape": action.get("shape", ["UNRESOLVED"]),
            "bounds": action.get("bounds", ["UNRESOLVED", "UNRESOLVED"]),
            "semantics": action.get("semantics", "UNRESOLVED"),
            "source": _format_source(action.get("source")),
        }
    else:
        result["action"] = "UNRESOLVED"

    state = spaces_ir.get("state", [])
    if state:
        result["state"] = [_map_space_field(f) for f in state]

    return result


def _map_timing(ir: dict) -> dict | None:
    """Map IR timing section → spec.yaml timing section."""
    timing_ir = ir.get("timing")
    if not timing_ir:
        return None

    result: dict = {
        "control_frequency_hz": timing_ir.get("control_frequency_hz", "UNRESOLVED"),
        "sensor_delay_ms": timing_ir.get("sensor_delay_ms", "UNRESOLVED"),
        "action_delay_ms": timing_ir.get("action_delay_ms", "UNRESOLVED"),
        "timestamping_model": timing_ir.get("timestamping_model", "UNRESOLVED"),
    }
    if timing_ir.get("sim_dt") is not None:
        result["sim_dt"] = timing_ir["sim_dt"]
    if timing_ir.get("policy_dt") is not None:
        result["policy_dt"] = timing_ir["policy_dt"]
    if timing_ir.get("source"):
        result["source"] = _format_source(timing_ir["source"])
    return result


def _map_expr_ast(ast: Any) -> Any:
    """Pass-through the expression AST (already structured), or 'UNRESOLVED'."""
    if ast == "UNRESOLVED" or ast is None:
        return "UNRESOLVED"
    return ast


def _map_reward_term(term: dict) -> dict:
    """Map a RewardTerm dict to spec format."""
    result: dict = {
        "name": term.get("name", "UNRESOLVED"),
        "expression_ast": _map_expr_ast(term.get("expression_ast")),
        "weight": term.get("weight", "UNRESOLVED"),
    }
    if term.get("clamp") is not None:
        result["clamp"] = term["clamp"]
    if term.get("zeroing_window"):
        result["zeroing_window"] = term["zeroing_window"]
    result["source"] = _format_source(term.get("source"))
    return result


def _map_reward(ir: dict) -> dict | None:
    """Map IR reward section → spec.yaml reward section."""
    reward_ir = ir.get("reward")
    if not reward_ir:
        return None

    result: dict = {
        "discount": reward_ir.get("discount", "UNRESOLVED"),
        "terms": [_map_reward_term(t) for t in reward_ir.get("terms", [])],
        "normalization": reward_ir.get("normalization", "UNRESOLVED"),
    }
    if reward_ir.get("source"):
        result["source"] = _format_source(reward_ir["source"])
    return result


def _map_bool_ast(ast: Any) -> Any:
    """Pass-through the boolean condition AST, or 'UNRESOLVED'."""
    if ast == "UNRESOLVED" or ast is None:
        return "UNRESOLVED"
    return ast


def _map_termination_condition(cond: dict) -> dict:
    """Map a TerminationCondition dict to spec format."""
    return {
        "name": cond.get("name", "UNRESOLVED"),
        "condition_ast": _map_bool_ast(cond.get("condition_ast")),
        "source": _format_source(cond.get("source")),
    }


def _map_termination(ir: dict) -> dict | None:
    """Map IR termination section → spec.yaml termination section."""
    term_ir = ir.get("termination")
    if not term_ir:
        return None

    result: dict = {
        "conditions": [
            _map_termination_condition(c) for c in term_ir.get("conditions", [])
        ],
        "max_episode_steps": term_ir.get("max_episode_steps", "UNRESOLVED"),
    }
    if term_ir.get("source"):
        result["source"] = _format_source(term_ir["source"])
    return result


def _map_gates_track(ir: dict) -> dict | None:
    """Map IR gates_track section → spec.yaml gates_track section."""
    gt_ir = ir.get("gates_track")
    if not gt_ir:
        return None

    result: dict = {}

    gg = gt_ir.get("gate_geometry")
    if gg:
        result["gate_geometry"] = {
            "shape": gg.get("shape", "UNRESOLVED"),
            "dimensions": gg.get("dimensions", {}),
            "virtual_thickness": gg.get("virtual_thickness", "UNRESOLVED"),
            "source": _format_source(gg.get("source")),
        }
    else:
        result["gate_geometry"] = "UNRESOLVED"

    ppo = gt_ir.get("pre_post_offsets")
    if ppo:
        result["pre_post_offsets"] = {
            "pre_gate_offset": ppo.get("pre_gate_offset", "UNRESOLVED"),
            "post_gate_offset": ppo.get("post_gate_offset", "UNRESOLVED"),
            "source": _format_source(ppo.get("source")),
        }
    else:
        result["pre_post_offsets"] = "UNRESOLVED"

    pc = gt_ir.get("pass_condition")
    if pc:
        result["pass_condition"] = {
            "condition_ast": _map_bool_ast(pc.get("condition_ast")),
            "source": _format_source(pc.get("source")),
        }
    else:
        result["pass_condition"] = "UNRESOLVED"

    result["num_gates"] = gt_ir.get("num_gates", "UNRESOLVED")

    if gt_ir.get("source"):
        result["source"] = _format_source(gt_ir["source"])
    return result


def _map_equation(eq: dict) -> dict:
    """Map an Equation dict to spec format."""
    return {
        "name": eq.get("name", "UNRESOLVED"),
        "latex": eq.get("latex", "UNRESOLVED"),
        "expression_ast": _map_expr_ast(eq.get("expression_ast")),
        "source": _format_source(eq.get("source")),
    }


def _map_dynamics_param(param: dict) -> dict:
    """Map a DynamicsParam dict to spec format."""
    return {
        "name": param.get("name", "UNRESOLVED"),
        "symbol": param.get("symbol", "UNRESOLVED"),
        "default_value": param.get("default_value", "UNRESOLVED"),
        "units": param.get("units", "UNRESOLVED"),
        "source": _format_source(param.get("source")),
    }


def _map_domain_rand(entry: dict) -> dict:
    """Map a DomainRandEntry dict to spec format."""
    return {
        "parameter": entry.get("parameter", "UNRESOLVED"),
        "distribution": entry.get("distribution", "UNRESOLVED"),
        "range": entry.get("range", ["UNRESOLVED", "UNRESOLVED"]),
        "resample_frequency": entry.get("resample_frequency", "UNRESOLVED"),
        "source": _format_source(entry.get("source")),
    }


def _map_dynamics(ir: dict) -> dict | None:
    """Map IR dynamics section → spec.yaml dynamics section."""
    dyn_ir = ir.get("dynamics")
    if not dyn_ir:
        return None

    result: dict = {
        "model_type": dyn_ir.get("model_type", "UNRESOLVED"),
        "equations": [_map_equation(e) for e in dyn_ir.get("equations", [])],
    }

    integrator = dyn_ir.get("integrator")
    if integrator:
        integ: dict = {
            "type": integrator.get("type", "UNRESOLVED"),
            "dt": integrator.get("dt", "UNRESOLVED"),
        }
        if integrator.get("source"):
            integ["source"] = _format_source(integrator["source"])
        result["integrator"] = integ
    else:
        result["integrator"] = "UNRESOLVED"

    result["parameters"] = [
        _map_dynamics_param(p) for p in dyn_ir.get("parameters", [])
    ]
    result["domain_randomization"] = [
        _map_domain_rand(d) for d in dyn_ir.get("domain_randomization", [])
    ]

    if dyn_ir.get("source"):
        result["source"] = _format_source(dyn_ir["source"])
    return result


def _map_perception(ir: dict) -> dict | None:
    """Map IR perception section → spec.yaml perception section."""
    perc_ir = ir.get("perception")
    if not perc_ir:
        return None

    result: dict = {}

    img = perc_ir.get("input_image")
    if img:
        result["input_image"] = {
            "resolution": img.get("resolution", ["UNRESOLVED", "UNRESOLVED"]),
            "channels": img.get("channels", "UNRESOLVED"),
            "dtype": img.get("dtype", "UNRESOLVED"),
            "source": _format_source(img.get("source")),
        }
    else:
        result["input_image"] = "UNRESOLVED"

    inorm = perc_ir.get("intrinsics_normalization")
    if inorm:
        result["intrinsics_normalization"] = {
            "target_K": inorm.get("target_K", "UNRESOLVED"),
            "source": _format_source(inorm.get("source")),
        }

    seg = perc_ir.get("segmentation_model")
    if seg:
        result["segmentation_model"] = {
            "architecture": seg.get("architecture", "UNRESOLVED"),
            "input_size": seg.get("input_size", ["UNRESOLVED", "UNRESOLVED"]),
            "output_classes": seg.get("output_classes", "UNRESOLVED"),
            "source": _format_source(seg.get("source")),
        }

    augs = perc_ir.get("augmentations", [])
    result["augmentations"] = [
        {
            "name": a.get("name", "UNRESOLVED"),
            "parameters": a.get("parameters", {}),
            "source": _format_source(a.get("source")),
        }
        for a in augs
    ]

    if perc_ir.get("source"):
        result["source"] = _format_source(perc_ir["source"])
    return result


def _map_model_component(comp: dict) -> dict:
    """Map a ModelComponent dict to spec format."""
    result: dict = {
        "type": comp.get("type", "UNRESOLVED"),
    }
    if comp.get("layers") is not None:
        result["layers"] = comp["layers"]
    if comp.get("hidden_size") is not None:
        result["hidden_size"] = comp["hidden_size"]
    result["source"] = _format_source(comp.get("source"))
    return result


def _map_world_model(ir: dict) -> dict | None:
    """Map IR world_model section → spec.yaml world_model section."""
    wm_ir = ir.get("world_model")
    if not wm_ir:
        return None

    result: dict = {
        "architecture": wm_ir.get("architecture", "UNRESOLVED"),
        "components": {
            k: _map_model_component(v)
            for k, v in sorted(wm_ir.get("components", {}).items())
        },
    }

    dl = wm_ir.get("discrete_latent")
    if dl:
        result["discrete_latent"] = {
            "num_categoricals": dl.get("num_categoricals", "UNRESOLVED"),
            "num_classes": dl.get("num_classes", "UNRESOLVED"),
            "source": _format_source(dl.get("source")),
        }
    else:
        result["discrete_latent"] = "UNRESOLVED"

    result["symlog"] = wm_ir.get("symlog", False)
    result["normalization"] = wm_ir.get("normalization", "UNRESOLVED")

    if wm_ir.get("source"):
        result["source"] = _format_source(wm_ir["source"])
    return result


def _map_network_spec(net: dict | None) -> dict:
    """Map a NetworkSpec dict to spec format."""
    if not net:
        return {"hidden_layers": "UNRESOLVED", "activation": "UNRESOLVED", "source": "UNRESOLVED"}
    return {
        "hidden_layers": net.get("hidden_layers", "UNRESOLVED"),
        "activation": net.get("activation", "UNRESOLVED"),
        "source": _format_source(net.get("source")),
    }


def _map_actor_critic(ir: dict) -> dict | None:
    """Map IR actor_critic section → spec.yaml actor_critic section."""
    ac_ir = ir.get("actor_critic")
    if not ac_ir:
        return None

    result: dict = {
        "policy_distribution": ac_ir.get("policy_distribution", "UNRESOLVED"),
        "deterministic_eval": ac_ir.get("deterministic_eval", True),
        "imagination_horizon": ac_ir.get("imagination_horizon", "UNRESOLVED"),
        "discount": ac_ir.get("discount", "UNRESOLVED"),
        "lambda_gae": ac_ir.get("lambda_gae", "UNRESOLVED"),
        "actor": _map_network_spec(ac_ir.get("actor")),
        "critic": _map_network_spec(ac_ir.get("critic")),
        "regularizers": [
            {
                "name": r.get("name", "UNRESOLVED"),
                "coefficient": r.get("coefficient", "UNRESOLVED"),
                "source": _format_source(r.get("source")),
            }
            for r in ac_ir.get("regularizers", [])
        ],
    }
    if ac_ir.get("source"):
        result["source"] = _format_source(ac_ir["source"])
    return result


def _map_training(ir: dict) -> dict | None:
    """Map IR training section → spec.yaml training section."""
    train_ir = ir.get("training")
    if not train_ir:
        return None

    result: dict = {
        "algorithm": train_ir.get("algorithm", "UNRESOLVED"),
    }

    replay = train_ir.get("replay")
    if replay:
        rep: dict = {
            "capacity_steps": replay.get("capacity_steps", "UNRESOLVED"),
            "context_length": replay.get("context_length", "UNRESOLVED"),
            "sampling": replay.get("sampling", "UNRESOLVED"),
        }
        if replay.get("source"):
            rep["source"] = _format_source(replay["source"])
        result["replay"] = rep
    else:
        result["replay"] = "UNRESOLVED"

    batch = train_ir.get("batch")
    if batch:
        b: dict = {
            "size": batch.get("size", "UNRESOLVED"),
            "length": batch.get("length", "UNRESOLVED"),
        }
        if batch.get("source"):
            b["source"] = _format_source(batch["source"])
        result["batch"] = b
    else:
        result["batch"] = "UNRESOLVED"

    schedule = train_ir.get("schedule")
    if schedule:
        phases = []
        for ph in schedule.get("phases", []):
            p: dict = {
                "name": ph.get("name", "UNRESOLVED"),
                "start_step": ph.get("start_step", "UNRESOLVED"),
                "end_step": ph.get("end_step", "UNRESOLVED"),
                "learning_rate": ph.get("learning_rate", "UNRESOLVED"),
                "entropy_scale": ph.get("entropy_scale", "UNRESOLVED"),
            }
            if ph.get("notes"):
                p["notes"] = ph["notes"]
            if ph.get("source"):
                p["source"] = _format_source(ph["source"])
            phases.append(p)
        sched: dict = {
            "total_env_steps": schedule.get("total_env_steps", "UNRESOLVED"),
            "phases": phases,
        }
        if schedule.get("source"):
            sched["source"] = _format_source(schedule["source"])
        result["schedule"] = sched
    else:
        result["schedule"] = "UNRESOLVED"

    result["train_ratio"] = train_ir.get("train_ratio", "UNRESOLVED")

    optimizer = train_ir.get("optimizer")
    if optimizer:
        opt: dict = {
            "type": optimizer.get("type", "UNRESOLVED"),
            "lr": optimizer.get("lr", "UNRESOLVED"),
            "eps": optimizer.get("eps", "UNRESOLVED"),
            "clip_grad": optimizer.get("clip_grad", "UNRESOLVED"),
        }
        if optimizer.get("source"):
            opt["source"] = _format_source(optimizer["source"])
        result["optimizer"] = opt
    else:
        result["optimizer"] = "UNRESOLVED"

    result["use_amp"] = train_ir.get("use_amp", False)

    if train_ir.get("source"):
        result["source"] = _format_source(train_ir["source"])
    return result


def _map_evaluation(ir: dict) -> dict | None:
    """Map IR evaluation section → spec.yaml evaluation section."""
    eval_ir = ir.get("evaluation")
    if not eval_ir:
        return None

    criteria = []
    for c in eval_ir.get("success_criteria", []):
        criteria.append(
            {
                "metric": c.get("metric", "UNRESOLVED"),
                "threshold": c.get("threshold", "UNRESOLVED"),
                "direction": c.get("direction", "UNRESOLVED"),
                "source": _format_source(c.get("source")),
            }
        )

    ablations = []
    for a in eval_ir.get("ablations", []):
        ablations.append(
            {
                "name": a.get("name", "UNRESOLVED"),
                "description": a.get("description", "UNRESOLVED"),
                "source": _format_source(a.get("source")),
            }
        )

    result: dict = {
        "success_criteria": criteria,
        "num_seeds": eval_ir.get("num_seeds", "UNRESOLVED"),
        "num_eval_episodes": eval_ir.get("num_eval_episodes", "UNRESOLVED"),
        "report_error_bars": eval_ir.get("report_error_bars", False),
        "ablations": ablations,
    }
    if eval_ir.get("source"):
        result["source"] = _format_source(eval_ir["source"])
    return result


def _map_deployment(ir: dict) -> dict | None:
    """Map IR deployment section → spec.yaml deployment section."""
    dep_ir = ir.get("deployment")
    if not dep_ir:
        return None

    budgets = []
    for b in dep_ir.get("runtime_budgets", []):
        budgets.append(
            {
                "module": b.get("module", "UNRESOLVED"),
                "max_ms": b.get("max_ms", "UNRESOLVED"),
                "source": _format_source(b.get("source")),
            }
        )

    safety = []
    for s in dep_ir.get("safety_constraints", []):
        safety.append(
            {
                "name": s.get("name", "UNRESOLVED"),
                "description": s.get("description", "UNRESOLVED"),
                "source": _format_source(s.get("source")),
            }
        )

    result: dict = {
        "target_hardware": dep_ir.get("target_hardware", "UNRESOLVED"),
        "inference_stack": dep_ir.get("inference_stack", []),
        "runtime_budgets": budgets,
        "safety_constraints": safety,
        "control_frequency_hz": dep_ir.get("control_frequency_hz", "UNRESOLVED"),
    }
    if dep_ir.get("source"):
        result["source"] = _format_source(dep_ir["source"])
    return result


# ---------------------------------------------------------------------------
# Baseline import handling
# ---------------------------------------------------------------------------


def _load_baseline_imports(import_paths: list[Path]) -> list[dict]:
    """
    Load baseline spec.yaml files and convert them to import entries.

    For each baseline, reads its meta section to extract name/version,
    then records the overrides list (initially empty — the user or a
    downstream tool will populate it).
    """
    entries: list[dict] = []
    for path in import_paths:
        if not path.exists():
            print(
                f"WARNING: --import path does not exist: {path}",
                file=sys.stderr,
            )
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                baseline = yaml.safe_load(fh)
        except Exception as exc:
            print(
                f"WARNING: failed to parse baseline spec {path}: {exc}",
                file=sys.stderr,
            )
            continue

        if not isinstance(baseline, dict):
            print(
                f"WARNING: baseline spec {path} is not a YAML mapping",
                file=sys.stderr,
            )
            continue

        meta = baseline.get("meta", {})
        paper = meta.get("paper", {})
        # Derive a name from the paper title (lowercase, first word)
        title = paper.get("title", "")
        name = title.split()[0].lower() if title else path.stem
        arxiv = paper.get("arxiv", "")
        version = paper.get("version", "")
        spec_version = f"{arxiv}@{version}" if arxiv else meta.get("tool_version", "UNRESOLVED")

        entries.append(
            {
                "name": name,
                "version": spec_version,
                "spec_url": str(path.resolve()),
                "overrides": [],
            }
        )
    return entries


def _map_imports(ir: dict, extra_imports: list[dict]) -> list[dict]:
    """Merge IR imports with imports derived from --import flags."""
    ir_imports = ir.get("imports", [])
    result: list[dict] = []

    for imp in ir_imports:
        result.append(
            {
                "name": imp.get("name", "UNRESOLVED"),
                "version": imp.get("version", "UNRESOLVED"),
                "spec_url": imp.get("spec_url"),
                "overrides": imp.get("overrides", []),
            }
        )

    for imp in extra_imports:
        # Avoid duplicates by name.
        existing_names = {e["name"] for e in result}
        if imp["name"] not in existing_names:
            result.append(imp)

    return result


# ---------------------------------------------------------------------------
# tex_parser IR integration: promote extracted tables/equations into sections
# ---------------------------------------------------------------------------


def _integrate_tex_parser_data(ir: dict) -> dict:
    """
    If the IR contains raw tex_parser output (tables, equations, sections),
    heuristically promote extracted data into appropriate IR sections.

    tex_parser.py produces a flat structure:
      {
        "tables": [...],
        "equations": [...],
        "sections": [...],
        "symbols": [...],
      }

    This function converts those flat lists into structured IR sections
    where the structured sections are missing or empty.
    """
    # Do not overwrite explicitly populated sections.
    if "tables" not in ir and "equations" not in ir:
        return ir

    ir = dict(ir)  # Shallow copy so we don't mutate the caller's dict.

    tables: list[dict] = ir.pop("tables", [])
    equations: list[dict] = ir.pop("equations", [])
    sections: list[dict] = ir.pop("sections", [])
    ir.pop("symbols", None)
    ir.pop("files_processed", None)
    ir.pop("extraction_warnings", None)
    ir.pop("source_dir", None)

    # Build a section-title → section-label map for context lookup.
    section_label_map: dict[str, str] = {}
    for sec in sections:
        title_lower = sec.get("title", "").lower()
        for keyword, section_name in TABLE_CAPTION_HINTS:
            if keyword in title_lower:
                section_label_map[title_lower] = section_name
                break

    # --- Map tables into dynamics.parameters and training sections ---
    dynamics_params: list[dict] = []
    training_params: dict[str, Any] = {}

    for i, tbl in enumerate(tables):
        caption = tbl.get("caption", "")
        section_hint = _classify_table_caption(caption)
        headers = tbl.get("headers", [])
        rows = tbl.get("rows", [])
        src_trace = tbl.get("source_trace", {})
        table_id = f"Table {i + 1}"

        source: dict = {}
        if caption:
            source["table"] = table_id
        if src_trace.get("file"):
            source["tex_file"] = src_trace["file"]
        if src_trace.get("line_start"):
            ls = src_trace["line_start"]
            le = src_trace.get("line_end", ls)
            source["line_range"] = [ls, le]
        if not source:
            source["table"] = table_id

        if section_hint == "dynamics" and not ir.get("dynamics", {}).get("parameters"):
            # Heuristic: columns 0=name, 1=symbol, 2=value/default
            name_col = 0
            sym_col = 1 if len(headers) > 1 else None
            val_col = 2 if len(headers) > 2 else (1 if len(headers) > 1 else None)
            unit_col = None
            for ci, h in enumerate(headers):
                hl = h.lower()
                if any(k in hl for k in ("unit", "dim")):
                    unit_col = ci

            for row in rows:
                if not row:
                    continue
                name = row[name_col] if len(row) > name_col else "UNRESOLVED"
                symbol = row[sym_col] if (sym_col is not None and len(row) > sym_col) else "UNRESOLVED"
                val_raw = row[val_col] if (val_col is not None and len(row) > val_col) else "UNRESOLVED"
                units = row[unit_col] if (unit_col is not None and len(row) > unit_col) else "UNRESOLVED"

                # Attempt numeric parse of default_value.
                try:
                    default_value: Any = float(val_raw.replace(",", ""))
                except (ValueError, AttributeError):
                    default_value = "UNRESOLVED"

                dynamics_params.append(
                    {
                        "name": name,
                        "symbol": symbol,
                        "default_value": default_value,
                        "units": units,
                        "source": source,
                    }
                )

        elif section_hint == "training" and rows and not ir.get("training"):
            # Build a flat key-value dict of hyperparameters.
            if len(headers) >= 2:
                key_col, val_col2 = 0, 1
                for row in rows:
                    if len(row) >= 2:
                        k = row[key_col].strip()
                        v_raw = row[val_col2].strip()
                        try:
                            v: Any = float(v_raw.replace(",", ""))
                            if v == int(v):
                                v = int(v)
                        except (ValueError, AttributeError):
                            v = v_raw if v_raw else "UNRESOLVED"
                        training_params[k] = v

    # Only inject dynamics.parameters if none already exist.
    if dynamics_params and not ir.get("dynamics", {}).get("parameters"):
        if "dynamics" not in ir:
            ir["dynamics"] = {}
        if not isinstance(ir["dynamics"], dict):
            ir["dynamics"] = {}
        if not ir["dynamics"].get("parameters"):
            ir["dynamics"]["parameters"] = dynamics_params

    # --- Map equations into dynamics.equations ---
    dyn_equations: list[dict] = []
    for i, eq in enumerate(equations):
        label = eq.get("label")
        section_hint_eq = _classify_equation_label(label)
        src_trace_eq = eq.get("source_trace", {})

        source_eq: dict = {}
        if label:
            source_eq["equation"] = f"Eq. {label}"
        if src_trace_eq.get("file"):
            source_eq["tex_file"] = src_trace_eq["file"]
        if src_trace_eq.get("line_start"):
            ls = src_trace_eq["line_start"]
            le = src_trace_eq.get("line_end", ls)
            source_eq["line_range"] = [ls, le]
        if not source_eq:
            source_eq["equation"] = f"Eq. {i + 1}"

        eq_dict: dict = {
            "name": label if label else f"equation_{i + 1}",
            "latex": eq.get("raw_latex", "UNRESOLVED"),
            "expression_ast": _map_expr_ast(eq.get("ast")),
            "source": source_eq,
        }

        if section_hint_eq == "dynamics":
            dyn_equations.append(eq_dict)

    if dyn_equations and not ir.get("dynamics", {}).get("equations"):
        if "dynamics" not in ir:
            ir["dynamics"] = {}
        if not isinstance(ir["dynamics"], dict):
            ir["dynamics"] = {}
        if not ir["dynamics"].get("equations"):
            ir["dynamics"]["equations"] = dyn_equations

    return ir


# ---------------------------------------------------------------------------
# Top-level spec dict construction
# ---------------------------------------------------------------------------


def build_spec_dict(ir: dict, extra_imports: list[dict]) -> dict:
    """
    Convert a raw IR dict (from ir_schema.SpecIR.to_dict() or tex_parser output)
    into the ordered spec.yaml dict.

    Parameters
    ----------
    ir:
        Raw IR dict.
    extra_imports:
        Import entries derived from --import baseline spec files.

    Returns
    -------
    dict
        Ordered spec dict ready for YAML serialisation.
    """
    # If this looks like raw tex_parser output, promote it first.
    if "tables" in ir or "equations" in ir:
        ir = _integrate_tex_parser_data(ir)

    # Build an unordered map of section → value.
    section_map: dict[str, Any] = {}

    meta_val = _map_meta(ir)
    section_map["meta"] = meta_val

    frames_val = _map_frames(ir)
    if frames_val is not None:
        section_map["frames"] = frames_val

    spaces_val = _map_spaces(ir)
    if spaces_val is not None:
        section_map["spaces"] = spaces_val

    timing_val = _map_timing(ir)
    if timing_val is not None:
        section_map["timing"] = timing_val

    reward_val = _map_reward(ir)
    if reward_val is not None:
        section_map["reward"] = reward_val

    term_val = _map_termination(ir)
    if term_val is not None:
        section_map["termination"] = term_val

    gt_val = _map_gates_track(ir)
    if gt_val is not None:
        section_map["gates_track"] = gt_val

    dyn_val = _map_dynamics(ir)
    if dyn_val is not None:
        section_map["dynamics"] = dyn_val

    perc_val = _map_perception(ir)
    if perc_val is not None:
        section_map["perception"] = perc_val

    wm_val = _map_world_model(ir)
    if wm_val is not None:
        section_map["world_model"] = wm_val

    ac_val = _map_actor_critic(ir)
    if ac_val is not None:
        section_map["actor_critic"] = ac_val

    train_val = _map_training(ir)
    if train_val is not None:
        section_map["training"] = train_val

    eval_val = _map_evaluation(ir)
    if eval_val is not None:
        section_map["evaluation"] = eval_val

    dep_val = _map_deployment(ir)
    if dep_val is not None:
        section_map["deployment"] = dep_val

    imports_val = _map_imports(ir, extra_imports)
    section_map["imports"] = imports_val

    # Assemble in canonical order.
    ordered: dict = {}
    for section in SECTION_ORDER:
        if section in section_map:
            ordered[section] = section_map[section]

    # Update unresolved_count in meta to reflect reality.
    total_unresolved = _count_unresolved(
        {k: v for k, v in ordered.items() if k != "meta"}
    )
    ordered["meta"]["unresolved_count"] = total_unresolved

    return ordered


# ---------------------------------------------------------------------------
# Lock file generation
# ---------------------------------------------------------------------------


def build_lock_dict(
    spec_yaml_text: str,
    ir: dict,
    output_path: Path,
) -> dict:
    """
    Build the spec.lock.json content.

    Parameters
    ----------
    spec_yaml_text:
        The already-serialised spec.yaml text (used for SHA-256).
    ir:
        Original IR dict for extracting metadata.
    output_path:
        Path to the output spec.yaml (used for context only).
    """
    sha256 = hashlib.sha256(spec_yaml_text.encode("utf-8")).hexdigest()

    meta_ir = ir.get("meta", {})
    paper_ir = meta_ir.get("paper", {})

    unresolved_count = _count_unresolved(
        {k: v for k, v in yaml.safe_load(spec_yaml_text).items() if k != "meta"}
    )

    return {
        "paper_arxiv_id": paper_ir.get("arxiv", "UNRESOLVED"),
        "paper_version": paper_ir.get("version", "UNRESOLVED"),
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_version": TOOL_VERSION,
        "spec_yaml_path": str(output_path.resolve()),
        "spec_yaml_sha256": sha256,
        "unresolved_count": unresolved_count,
        "lock_complete": unresolved_count == 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Deterministic YAML emitter: converts IR JSON to spec.yaml and "
            "writes spec.lock.json alongside it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Basic emission from ir_schema.SpecIR JSON:
  python emit_yaml.py --ir ir_output.json --output spec.yaml

  # With baseline linking:
  python emit_yaml.py --ir ir_output.json --output spec.yaml \\
      --import specs/dreamerv3/spec.yaml \\
      --import specs/informed_dreamer/spec.yaml

  # Custom output directory:
  python emit_yaml.py --ir ir_output.json --output /tmp/my_spec/spec.yaml
""",
    )
    p.add_argument(
        "--ir",
        required=True,
        metavar="FILE",
        help="Path to the IR JSON file (output of tex_parser.py or ir_schema serialisation).",
    )
    p.add_argument(
        "--output",
        default="spec.yaml",
        metavar="FILE",
        help="Path for the output spec.yaml (default: spec.yaml).",
    )
    p.add_argument(
        "--import",
        dest="imports",
        action="append",
        default=[],
        metavar="SPEC_YAML",
        help=(
            "Path to a baseline spec.yaml to link as an import. "
            "May be repeated for multiple baselines."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    ir_path = Path(args.ir).resolve()
    output_path = Path(args.output).resolve()
    import_paths = [Path(p).resolve() for p in args.imports]

    # --- Load IR JSON ---
    if not ir_path.exists():
        print(f"ERROR: IR file does not exist: {ir_path}", file=sys.stderr)
        return 1

    print(f"Loading IR from: {ir_path}", file=sys.stderr)
    try:
        with ir_path.open("r", encoding="utf-8") as fh:
            ir: dict = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Failed to parse IR JSON: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: Cannot read IR file: {exc}", file=sys.stderr)
        return 1

    if not isinstance(ir, dict):
        print("ERROR: IR JSON must be a top-level mapping.", file=sys.stderr)
        return 1

    # --- Load baseline imports ---
    extra_imports: list[dict] = []
    if import_paths:
        print(
            f"Loading {len(import_paths)} baseline spec(s) for import linking...",
            file=sys.stderr,
        )
        extra_imports = _load_baseline_imports(import_paths)

    # --- Build spec dict ---
    print("Building spec dict...", file=sys.stderr)
    spec_dict = build_spec_dict(ir, extra_imports)

    # --- Emit YAML ---
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec_yaml_text = _dump_yaml(spec_dict)

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("# spec.yaml — generated by emit_yaml.py\n")
        fh.write("# DO NOT EDIT manually; re-run emit_yaml.py to regenerate.\n")
        fh.write("#\n")
        unresolved_n = spec_dict["meta"]["unresolved_count"]
        fh.write(f"# UNRESOLVED fields: {unresolved_n}\n")
        fh.write("#\n")
        fh.write(spec_yaml_text)

    print(f"spec.yaml written to: {output_path}", file=sys.stderr)

    # --- Write lock file ---
    lock_path = output_path.with_name("spec.lock.json")
    lock_dict = build_lock_dict(spec_yaml_text, ir, output_path)

    with lock_path.open("w", encoding="utf-8") as fh:
        json.dump(lock_dict, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"spec.lock.json written to: {lock_path}", file=sys.stderr)

    # --- Summary ---
    unresolved_count = spec_dict["meta"]["unresolved_count"]
    sections_present = [
        s for s in SECTION_ORDER
        if s in spec_dict and s not in ("meta", "imports")
    ]
    print(
        f"Done. Sections: {len(sections_present)}/{len(SECTION_ORDER) - 2}. "
        f"UNRESOLVED fields: {unresolved_count}.",
        file=sys.stderr,
    )

    if unresolved_count > 0:
        print(
            f"WARNING: {unresolved_count} field(s) are UNRESOLVED. "
            "spec.lock.json is not finalisable until all are resolved.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
