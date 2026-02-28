#!/usr/bin/env python3
"""
ir_schema.py — Pydantic v2 Intermediate Representation (IR) models.

The IR is the internal typed representation that sits between raw paper
extraction and spec emission.  All extractors (LaTeX, PDF, HTML) produce IR
entities.  All emitters (YAML, Markdown) consume them.

Design principles
-----------------
* Everything is typed — Pydantic v2 enforces field types at construction time.
* UNRESOLVED is explicit — fields use the ``Resolvable[T]`` union alias rather
  than silent defaults or None.
* Provenance is mandatory — every extracted value carries a ``SourceTrace``.
* Informed-POMDP is first-class — training-only vs execution fields are
  structurally distinct via ``InformedField``.

Usage
-----
    from ir_schema import SpecIR

    ir = SpecIR.from_dict(raw_dict)
    print(ir.count_unresolved())
    ir.validate_sources()
    ir.validate_informed_split()
    yaml_dict = ir.to_dict()
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Generic resolvable alias
# ---------------------------------------------------------------------------

T = TypeVar("T")

# A field whose value the extractor may not have been able to determine.
# Assign the literal string "UNRESOLVED" when extraction fails; never use None
# as a sentinel for extraction failure — None means "not applicable".
Resolvable = Union[T, Literal["UNRESOLVED"]]


# ---------------------------------------------------------------------------
# SourceTrace — provenance back to the paper
# ---------------------------------------------------------------------------


class SourceTrace(BaseModel):
    """Traces an extracted value back to its location in the paper.

    At least one location field (section, table, equation, figure, or page)
    must be non-None.  The validator below enforces this invariant so that
    extractors cannot accidentally produce untraceable values.
    """

    section: Optional[str] = None   # e.g. "§3.2", "§4.1"
    table: Optional[str] = None     # e.g. "Table 1", "Table 3"
    equation: Optional[str] = None  # e.g. "Eq. 4", "Eq. 12"
    figure: Optional[str] = None    # e.g. "Fig. 2 caption"
    page: Optional[int] = None      # PDF page number (1-indexed)
    tex_file: Optional[str] = None  # Relative path within the LaTeX tarball
    line_range: Optional[tuple[int, int]] = None  # Inclusive line range in TeX
    raw_text: Optional[str] = None  # Verbatim extracted text snippet

    @model_validator(mode="after")
    def at_least_one_location(self) -> "SourceTrace":
        location_fields = (
            self.section,
            self.table,
            self.equation,
            self.figure,
            self.page,
        )
        if all(f is None for f in location_fields):
            raise ValueError(
                "SourceTrace must have at least one non-None location field "
                "(section, table, equation, figure, or page)."
            )
        return self


# ---------------------------------------------------------------------------
# Expression AST
# ---------------------------------------------------------------------------


class ExprNode(BaseModel):
    """AST node for mathematical expressions.

    Four node types are recognised:

    ``literal``
        A concrete numeric value.  ``value`` must be set.

    ``field``
        A reference to a named field in one of the spaces.  ``name`` must be
        set; ``space`` indicates which space (``"state"``, ``"observation"``,
        ``"action"``).

    ``op``
        An n-ary mathematical operation.  ``op`` names the operation
        (``"add"``, ``"subtract"``, ``"multiply"``, ``"divide"``,
        ``"clamp"``, ``"min"``, ``"max"``, ``"norm"``, ``"pow"``, …).
        ``args`` contains the operands.  ``params`` carries keyword arguments
        (e.g. ``{"min": -1.0, "max": 1.0}`` for a clamp).

    ``func``
        A named function call.  ``func_name`` must be set; ``args`` contains
        positional arguments.
    """

    model_config = {"arbitrary_types_allowed": True}

    type: Literal["literal", "field", "op", "func"]

    # --- literal ---
    value: Optional[float] = None

    # --- field reference ---
    name: Optional[str] = None
    space: Optional[str] = None  # "state" | "observation" | "action"

    # --- operation ---
    op: Optional[str] = None
    args: list["ExprNode"] = Field(default_factory=list)
    params: dict[str, float] = Field(default_factory=dict)

    # --- function call ---
    func_name: Optional[str] = None

    @model_validator(mode="after")
    def _check_type_constraints(self) -> "ExprNode":
        if self.type == "literal" and self.value is None:
            raise ValueError("ExprNode type='literal' requires 'value' to be set.")
        if self.type == "field" and self.name is None:
            raise ValueError("ExprNode type='field' requires 'name' to be set.")
        if self.type == "op" and self.op is None:
            raise ValueError("ExprNode type='op' requires 'op' to be set.")
        if self.type == "func" and self.func_name is None:
            raise ValueError("ExprNode type='func' requires 'func_name' to be set.")
        return self


# Allow forward-referenced recursive definition.
ExprNode.model_rebuild()


# ---------------------------------------------------------------------------
# Boolean AST
# ---------------------------------------------------------------------------


class BoolNode(BaseModel):
    """AST node for boolean conditions (termination, gate-pass, …).

    Three node types:

    ``compare``
        A comparison between two expressions or values.  ``op`` is one of
        ``"lt"``, ``"gt"``, ``"le"``, ``"ge"``, ``"eq"``, ``"ne"``.
        ``left`` and ``right`` can be ``BoolNode`` or ``ExprNode``.

    ``logic``
        A logical connective.  ``logic_op`` is ``"and"``, ``"or"``, or
        ``"not"``.  ``children`` contains the operands.

    ``field_check``
        A simple boolean field (e.g. "collision_detected").  ``name`` must be
        set on the embedded ``ExprNode`` passed as ``left``.
    """

    model_config = {"arbitrary_types_allowed": True}

    type: Literal["compare", "logic", "field_check"]

    # --- compare ---
    op: Optional[str] = None  # "lt" | "gt" | "le" | "ge" | "eq" | "ne"
    left: Optional[Union["BoolNode", ExprNode]] = None
    right: Optional[Union["BoolNode", ExprNode]] = None

    # --- logic ---
    logic_op: Optional[str] = None  # "and" | "or" | "not"
    children: list["BoolNode"] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_type_constraints(self) -> "BoolNode":
        if self.type == "compare" and self.op is None:
            raise ValueError("BoolNode type='compare' requires 'op' to be set.")
        if self.type == "logic" and self.logic_op is None:
            raise ValueError("BoolNode type='logic' requires 'logic_op' to be set.")
        return self


BoolNode.model_rebuild()


# ---------------------------------------------------------------------------
# Space field models
# ---------------------------------------------------------------------------


class SpaceField(BaseModel):
    """A single named field in an observation/state/action space."""

    name: str
    dtype: str                         # "float32", "bool", "int64", …
    shape: list[Union[int, str]]        # int for concrete, str for symbolic
    units: Optional[str] = None
    source: SourceTrace


class InformedField(SpaceField):
    """Privileged information available only during training (informed POMDP).

    These fields must NOT appear in ``observation_exec`` — the
    ``validate_informed_split`` check on ``SpecIR`` enforces this.
    """

    informed_dreamer_key: Optional[str] = None  # Regex for decoder gating
    training_only: bool = True


class ActionSpace(BaseModel):
    """The continuous or discrete action space."""

    name: str
    dtype: str
    shape: list[Union[int, str]]
    bounds: tuple[float, float]         # (min, max) clamping values
    semantics: str                      # Human description of what actions mean
    source: SourceTrace


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


class RewardTerm(BaseModel):
    """One additive term in the total reward signal."""

    name: str
    expression_ast: Union[ExprNode, Literal["UNRESOLVED"]]
    weight: Union[float, Literal["UNRESOLVED"]]
    clamp: Optional[tuple[float, float]] = None
    zeroing_window: Optional[str] = None  # Condition string (e.g. "within_3s_of_gate")
    source: SourceTrace


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


class TerminationCondition(BaseModel):
    """A single named condition that ends an episode."""

    name: str
    condition_ast: Union[BoolNode, Literal["UNRESOLVED"]]
    source: SourceTrace


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


class DynamicsParam(BaseModel):
    """A physical parameter of the dynamics model (mass, inertia, drag, …)."""

    name: str
    symbol: str                              # LaTeX symbol (e.g. "m", r"I_{xx}")
    default_value: Union[float, Literal["UNRESOLVED"]]
    units: str
    source: SourceTrace


class DomainRandEntry(BaseModel):
    """Domain-randomisation specification for one parameter."""

    parameter: str                           # Reference to DynamicsParam.name
    distribution: str                        # "uniform" | "normal" | "log_uniform"
    range: tuple[float, float]
    resample_frequency: str                  # "per_episode" | "per_step" | "fixed"
    source: SourceTrace


class Equation(BaseModel):
    """A named mathematical equation from the paper."""

    name: str
    latex: str                               # Original LaTeX string
    expression_ast: Union[ExprNode, Literal["UNRESOLVED"]]
    source: SourceTrace


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------


class Augmentation(BaseModel):
    """A data-augmentation step applied to observations."""

    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    source: SourceTrace


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------


class ModelComponent(BaseModel):
    """One architectural component of the world model or actor-critic."""

    type: str                                # "MLP", "GRU", "CNN", "transformer", …
    layers: Optional[Union[list[Any], str]] = None
    hidden_size: Optional[Union[int, str]] = None
    source: SourceTrace


# ---------------------------------------------------------------------------
# Actor-critic
# ---------------------------------------------------------------------------


class Regularizer(BaseModel):
    """A regularisation term applied during actor/critic training."""

    name: str
    coefficient: Union[float, Literal["UNRESOLVED"]]
    source: SourceTrace


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


class RuntimeBudget(BaseModel):
    """Latency budget for one inference module."""

    module: str
    max_ms: Union[float, Literal["UNRESOLVED"]]
    source: SourceTrace


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class Criterion(BaseModel):
    """A single success / performance criterion."""

    metric: str
    threshold: Union[float, Literal["UNRESOLVED"]]
    direction: Literal["higher_is_better", "lower_is_better"]
    source: SourceTrace


# ---------------------------------------------------------------------------
# Baseline import
# ---------------------------------------------------------------------------


class BaselineImport(BaseModel):
    """Reference to a prior-work spec that this spec extends."""

    name: str                                # e.g. "dreamerv3"
    version: str                             # Commit hash or version tag
    spec_url: Optional[str] = None           # URL to the upstream spec.yaml
    overrides: list[str] = Field(default_factory=list)  # JSONPaths overridden here


# ---------------------------------------------------------------------------
# Top-level section models
# ---------------------------------------------------------------------------


class PaperInfo(BaseModel):
    """Bibliographic metadata for the paper being compiled."""

    title: str
    arxiv: Optional[str] = None
    version: Optional[str] = None           # arXiv version string, e.g. "v2"
    doi: Optional[str] = None
    authors: list[str] = Field(default_factory=list)


class SourcesInfo(BaseModel):
    """Provenance of the raw paper sources used during extraction."""

    prefer: list[str] = Field(
        default_factory=lambda: ["arxiv_tex", "pdf", "html"],
        description="Priority order of source types actually used.",
    )
    tex_hash: Optional[str] = None          # SHA-256 of the LaTeX tarball
    pdf_hash: Optional[str] = None          # SHA-256 of the PDF


class Meta(BaseModel):
    """Paper metadata, extraction provenance, and quality metrics."""

    paper: PaperInfo
    sources: SourcesInfo = Field(default_factory=SourcesInfo)
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tool_version: str = "0.1.0"
    unresolved_count: int = 0


# --- Coordinate frames ---


class FrameTransform(BaseModel):
    """An explicit static or dynamic transform between two named frames."""

    from_frame: str = Field(alias="from")
    to_frame: str = Field(alias="to")
    type: Literal["static", "dynamic"]
    description: str

    model_config = {"populate_by_name": True}


class FrameConventions(BaseModel):
    """Coordinate frame conventions used in the paper."""

    convention: str                         # e.g. "NED", "ENU", "ROS"
    frames: list[str] = Field(default_factory=list)
    transforms: list[FrameTransform] = Field(default_factory=list)
    source: Optional[SourceTrace] = None


# --- Spaces ---


class Spaces(BaseModel):
    """All space definitions for the task."""

    observation_exec: list[SpaceField] = Field(
        default_factory=list,
        description="Fields available to the policy at deployment.",
    )
    information_train: list[InformedField] = Field(
        default_factory=list,
        description="Privileged fields available only during training.",
    )
    action: Optional[ActionSpace] = None
    state: list[SpaceField] = Field(
        default_factory=list,
        description="Full latent state (superset of obs + info).",
    )


# --- Timing ---


class Timing(BaseModel):
    """Control-loop timing and delay specifications."""

    control_frequency_hz: Union[float, Literal["UNRESOLVED"]] = "UNRESOLVED"
    sensor_delay_ms: Union[float, Literal["UNRESOLVED"]] = "UNRESOLVED"
    action_delay_ms: Union[float, Literal["UNRESOLVED"]] = "UNRESOLVED"
    timestamping_model: Optional[str] = None  # "camera_anchored" | "control_loop_anchored"
    sim_dt: Optional[Union[float, Literal["UNRESOLVED"]]] = None
    policy_dt: Optional[Union[float, Literal["UNRESOLVED"]]] = None
    source: Optional[SourceTrace] = None


# --- Reward ---


class Reward(BaseModel):
    """Reward signal specification."""

    discount: Union[float, Literal["UNRESOLVED"]] = "UNRESOLVED"
    terms: list[RewardTerm] = Field(default_factory=list)
    normalization: Optional[str] = None     # "symlog" | "none" | …
    source: Optional[SourceTrace] = None


# --- Termination ---


class Termination(BaseModel):
    """Episode termination specification."""

    conditions: list[TerminationCondition] = Field(default_factory=list)
    max_episode_steps: Optional[Union[int, Literal["UNRESOLVED"]]] = None
    source: Optional[SourceTrace] = None


# --- Gates / Track (domain-specific) ---


class GateGeometry(BaseModel):
    """Physical geometry of one gate type."""

    shape: str                              # "square" | "circular" | …
    dimensions: dict[str, Any] = Field(default_factory=dict)
    virtual_thickness: Union[float, Literal["UNRESOLVED"]] = "UNRESOLVED"
    source: SourceTrace


class PrePostOffsets(BaseModel):
    """Waypoint offsets before and after each gate."""

    pre_gate_offset: Union[float, Literal["UNRESOLVED"]] = "UNRESOLVED"
    post_gate_offset: Union[float, Literal["UNRESOLVED"]] = "UNRESOLVED"
    source: SourceTrace


class PassCondition(BaseModel):
    """Boolean condition determining a successful gate pass."""

    condition_ast: Union[BoolNode, Literal["UNRESOLVED"]]
    source: SourceTrace


class GatesTrack(BaseModel):
    """Domain-specific gate/track specification (e.g. drone racing)."""

    gate_geometry: Optional[GateGeometry] = None
    pre_post_offsets: Optional[PrePostOffsets] = None
    pass_condition: Optional[PassCondition] = None
    num_gates: Optional[Union[int, str]] = None
    source: Optional[SourceTrace] = None


# --- Dynamics ---


class IntegratorSpec(BaseModel):
    """Numerical integrator configuration."""

    type: str                               # "RK4" | "Euler" | "adaptive"
    dt: Union[float, Literal["UNRESOLVED"]] = "UNRESOLVED"
    source: Optional[SourceTrace] = None


class Dynamics(BaseModel):
    """Dynamics model specification."""

    model_type: str = "ODE"                 # "ODE" | "learned" | "hybrid"
    equations: list[Equation] = Field(default_factory=list)
    integrator: Optional[IntegratorSpec] = None
    parameters: list[DynamicsParam] = Field(default_factory=list)
    domain_randomization: list[DomainRandEntry] = Field(default_factory=list)
    source: Optional[SourceTrace] = None


# --- Perception ---


class ImageSpec(BaseModel):
    """Specification for one image input stream."""

    resolution: tuple[int, int]
    channels: int
    dtype: str
    source: SourceTrace


class IntrinsicsNormalization(BaseModel):
    """Camera intrinsics normalisation applied before the encoder."""

    target_K: Union[list[list[float]], Literal["UNRESOLVED"]] = "UNRESOLVED"
    source: SourceTrace


class SegmentationModel(BaseModel):
    """Perception segmentation model used for masking."""

    architecture: str
    input_size: tuple[int, int]
    output_classes: Union[int, Literal["UNRESOLVED"]] = "UNRESOLVED"
    source: SourceTrace


class Perception(BaseModel):
    """Perception pipeline specification."""

    input_image: Optional[ImageSpec] = None
    intrinsics_normalization: Optional[IntrinsicsNormalization] = None
    segmentation_model: Optional[SegmentationModel] = None
    augmentations: list[Augmentation] = Field(default_factory=list)
    source: Optional[SourceTrace] = None


# --- World model ---


class DiscreteLatent(BaseModel):
    """Categorical latent variable configuration."""

    num_categoricals: Union[int, Literal["UNRESOLVED"]] = "UNRESOLVED"
    num_classes: Union[int, Literal["UNRESOLVED"]] = "UNRESOLVED"
    source: SourceTrace


class WorldModel(BaseModel):
    """World-model (RSSM/transformer) architecture."""

    architecture: str                       # "RSSM" | "transformer" | …
    components: dict[str, ModelComponent] = Field(default_factory=dict)
    discrete_latent: Optional[DiscreteLatent] = None
    symlog: bool = False
    normalization: Optional[str] = None     # "layer_norm" | "none" | …
    source: Optional[SourceTrace] = None


# --- Actor-critic ---


class NetworkSpec(BaseModel):
    """Specification for an MLP network (actor or critic)."""

    hidden_layers: Union[list[int], Literal["UNRESOLVED"]] = "UNRESOLVED"
    activation: Optional[str] = None
    source: SourceTrace


class ActorCritic(BaseModel):
    """Policy and value network specification."""

    policy_distribution: str               # "squashed_normal" | "categorical" | …
    deterministic_eval: bool = True
    imagination_horizon: Union[int, Literal["UNRESOLVED"]] = "UNRESOLVED"
    discount: Union[float, Literal["UNRESOLVED"]] = "UNRESOLVED"
    lambda_gae: Union[float, Literal["UNRESOLVED"]] = "UNRESOLVED"
    actor: Optional[NetworkSpec] = None
    critic: Optional[NetworkSpec] = None
    regularizers: list[Regularizer] = Field(default_factory=list)
    source: Optional[SourceTrace] = None


# --- Training ---


class ReplayConfig(BaseModel):
    """Experience replay buffer configuration."""

    capacity_steps: Union[int, Literal["UNRESOLVED"]] = "UNRESOLVED"
    context_length: Union[int, Literal["UNRESOLVED"]] = "UNRESOLVED"
    sampling: str = "uniform"               # "uniform" | "prioritized"
    source: Optional[SourceTrace] = None


class BatchConfig(BaseModel):
    """Batch shape for world-model training."""

    size: Union[int, Literal["UNRESOLVED"]] = "UNRESOLVED"
    length: Union[int, Literal["UNRESOLVED"]] = "UNRESOLVED"
    source: Optional[SourceTrace] = None


class TrainingPhase(BaseModel):
    """A distinct phase of the training schedule."""

    name: str
    start_step: Union[int, Literal["UNRESOLVED"]] = "UNRESOLVED"
    end_step: Union[int, Literal["UNRESOLVED"]] = "UNRESOLVED"
    learning_rate: Union[float, Literal["UNRESOLVED"]] = "UNRESOLVED"
    entropy_scale: Union[float, Literal["UNRESOLVED"]] = "UNRESOLVED"
    notes: Optional[str] = None
    source: Optional[SourceTrace] = None


class TrainingSchedule(BaseModel):
    """Full training schedule across all phases."""

    total_env_steps: Union[int, Literal["UNRESOLVED"]] = "UNRESOLVED"
    phases: list[TrainingPhase] = Field(default_factory=list)
    source: Optional[SourceTrace] = None


class OptimizerConfig(BaseModel):
    """Optimiser hyper-parameters."""

    type: str = "Adam"
    lr: Union[float, Literal["UNRESOLVED"]] = "UNRESOLVED"
    eps: Union[float, Literal["UNRESOLVED"]] = "UNRESOLVED"
    clip_grad: Optional[Union[float, Literal["UNRESOLVED"]]] = None
    source: Optional[SourceTrace] = None


class Training(BaseModel):
    """Full training configuration."""

    algorithm: str
    replay: Optional[ReplayConfig] = None
    batch: Optional[BatchConfig] = None
    schedule: Optional[TrainingSchedule] = None
    train_ratio: Union[float, Literal["UNRESOLVED"]] = "UNRESOLVED"
    optimizer: Optional[OptimizerConfig] = None
    use_amp: bool = False
    source: Optional[SourceTrace] = None


# --- Evaluation ---


class AblationEntry(BaseModel):
    """One ablation study documented in the paper."""

    name: str
    description: str
    source: SourceTrace


class Evaluation(BaseModel):
    """Evaluation protocol and success criteria."""

    success_criteria: list[Criterion] = Field(default_factory=list)
    num_seeds: Union[int, Literal["UNRESOLVED"]] = "UNRESOLVED"
    num_eval_episodes: Union[int, Literal["UNRESOLVED"]] = "UNRESOLVED"
    report_error_bars: bool = False
    ablations: list[AblationEntry] = Field(default_factory=list)
    source: Optional[SourceTrace] = None


# --- Deployment ---


class SafetyConstraint(BaseModel):
    """A hard safety constraint for real-world deployment."""

    name: str
    description: str
    source: SourceTrace


class Deployment(BaseModel):
    """Deployment target and inference constraints."""

    target_hardware: str
    inference_stack: list[str] = Field(default_factory=list)
    runtime_budgets: list[RuntimeBudget] = Field(default_factory=list)
    safety_constraints: list[SafetyConstraint] = Field(default_factory=list)
    control_frequency_hz: Optional[Union[float, Literal["UNRESOLVED"]]] = None
    source: Optional[SourceTrace] = None


# ---------------------------------------------------------------------------
# Root IR model
# ---------------------------------------------------------------------------


def _walk_value(value: Any) -> int:
    """Recursively count how many leaf values equal the string "UNRESOLVED"."""
    if isinstance(value, str):
        return 1 if value == "UNRESOLVED" else 0
    if isinstance(value, BaseModel):
        return sum(_walk_value(v) for v in value.__dict__.values())
    if isinstance(value, dict):
        return sum(_walk_value(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_walk_value(item) for item in value)
    return 0


def _collect_sources(value: Any, traces: list[SourceTrace]) -> None:
    """Recursively gather all SourceTrace instances from a model tree."""
    if isinstance(value, SourceTrace):
        traces.append(value)
    elif isinstance(value, BaseModel):
        for v in value.__dict__.values():
            _collect_sources(v, traces)
    elif isinstance(value, dict):
        for v in value.values():
            _collect_sources(v, traces)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_sources(item, traces)


class SpecIR(BaseModel):
    """Root model aggregating all IR sections extracted from a single paper.

    All section fields are optional so that partially-extracted specs can be
    created, validated, and incrementally completed.
    """

    meta: Meta
    frames: Optional[FrameConventions] = None
    spaces: Optional[Spaces] = None
    timing: Optional[Timing] = None
    reward: Optional[Reward] = None
    termination: Optional[Termination] = None
    gates_track: Optional[GatesTrack] = None
    dynamics: Optional[Dynamics] = None
    perception: Optional[Perception] = None
    world_model: Optional[WorldModel] = None
    actor_critic: Optional[ActorCritic] = None
    training: Optional[Training] = None
    evaluation: Optional[Evaluation] = None
    deployment: Optional[Deployment] = None
    imports: list[BaselineImport] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def count_unresolved(self) -> int:
        """Walk the entire model tree and count ``"UNRESOLVED"`` leaf values.

        The returned count should equal ``meta.unresolved_count`` after the
        final extraction pass.  Discrepancies indicate that meta was not
        updated after a field was resolved or newly marked UNRESOLVED.

        Returns
        -------
        int
            Number of leaf values equal to the string ``"UNRESOLVED"``.
        """
        # Exclude meta itself from the walk to avoid double-counting the
        # unresolved_count field (it's an int, not a string, so it wouldn't
        # match anyway, but being explicit is clearer).
        total = 0
        for field_name, value in self.__dict__.items():
            if field_name == "meta":
                continue
            total += _walk_value(value)
        return total

    def validate_sources(self) -> list[str]:
        """Verify every SourceTrace has at least one non-None location field.

        Returns
        -------
        list[str]
            List of violation descriptions.  Empty list means all sources are
            valid.  Note: the SourceTrace validator already enforces this at
            construction time; this method is a secondary audit pass useful
            when loading raw dicts that bypassed Pydantic construction.
        """
        traces: list[SourceTrace] = []
        _collect_sources(self, traces)

        violations: list[str] = []
        for trace in traces:
            location_fields = (
                trace.section,
                trace.table,
                trace.equation,
                trace.figure,
                trace.page,
            )
            if all(f is None for f in location_fields):
                violations.append(
                    f"SourceTrace has no location: raw_text={trace.raw_text!r} "
                    f"tex_file={trace.tex_file!r}"
                )
        return violations

    def validate_informed_split(self) -> list[str]:
        """Enforce the informed-POMDP invariant.

        No name that appears in ``spaces.information_train`` may also appear
        in ``spaces.observation_exec``.  Violations mean the extractor
        incorrectly labelled a field or duplicated it across spaces.

        Returns
        -------
        list[str]
            List of field names that violate the invariant.  Empty means clean.
        """
        if self.spaces is None:
            return []

        exec_names = {f.name for f in self.spaces.observation_exec}
        train_names = {f.name for f in self.spaces.information_train}
        overlap = exec_names & train_names

        if overlap:
            return [
                f"Field '{n}' appears in both observation_exec and information_train."
                for n in sorted(overlap)
            ]
        return []

    def validate_shapes(self) -> list[str]:
        """Check that symbolic shape dimensions are defined elsewhere in the spec.

        A symbolic dimension (a string such as ``"N_PARAMS"``) is considered
        resolved if it appears as a ``DynamicsParam.name`` or as a string that
        looks like a concrete constant (pure digits).

        Returns
        -------
        list[str]
            Descriptions of unresolved symbolic dimensions.
        """
        defined_symbols: set[str] = set()
        if self.dynamics:
            for p in self.dynamics.parameters:
                defined_symbols.add(p.name)

        violations: list[str] = []

        def _check_field(sf: SpaceField) -> None:
            for dim in sf.shape:
                if isinstance(dim, str) and not dim.isdigit():
                    if dim not in defined_symbols:
                        violations.append(
                            f"Space field '{sf.name}' has symbolic dimension "
                            f"'{dim}' not defined in dynamics.parameters."
                        )

        if self.spaces:
            for f in self.spaces.observation_exec:
                _check_field(f)
            for f in self.spaces.information_train:
                _check_field(f)
            for f in self.spaces.state:
                _check_field(f)

        return violations

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the IR to a plain Python dict suitable for YAML emission.

        Uses Pydantic's ``model_dump`` with ``mode="json"`` so that all values
        are JSON-compatible primitives.  ``None`` fields are excluded to keep
        emitted YAML compact.

        Returns
        -------
        dict
            Recursively serialised representation.
        """
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpecIR":
        """Deserialise from a raw dict (e.g. loaded from a JSON/YAML file).

        Parameters
        ----------
        data:
            Raw dict, e.g. the result of ``json.load(...)`` or
            ``yaml.safe_load(...)``.

        Returns
        -------
        SpecIR
            Validated IR instance.

        Raises
        ------
        pydantic.ValidationError
            If the data does not conform to the schema.
        """
        return cls.model_validate(data)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def sync_unresolved_count(self) -> None:
        """Update ``meta.unresolved_count`` to match ``count_unresolved()``."""
        self.meta.unresolved_count = self.count_unresolved()

    def summary(self) -> str:
        """Return a one-line human-readable summary of the IR state."""
        n_unresolved = self.count_unresolved()
        sections_present = [
            name
            for name in (
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
            )
            if getattr(self, name) is not None
        ]
        return (
            f"SpecIR[paper={self.meta.paper.title!r}] "
            f"sections={len(sections_present)}/{13} "
            f"unresolved={n_unresolved} "
            f"imports={len(self.imports)}"
        )


# ---------------------------------------------------------------------------
# __main__ — minimal smoke-test / demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # ------------------------------------------------------------------
    # Build a minimal but structurally complete spec
    # ------------------------------------------------------------------

    paper_source = SourceTrace(section="§1", page=1, raw_text="Title page")

    minimal_spec = SpecIR(
        meta=Meta(
            paper=PaperInfo(
                title="SkyDreamer: Informed World Models for Agile Drone Racing",
                arxiv="2510.14783",
                version="v1",
                authors=["Alice Researcher", "Bob Engineer"],
            ),
            sources=SourcesInfo(
                prefer=["arxiv_tex", "pdf"],
                tex_hash="abc123",
                pdf_hash="def456",
            ),
            tool_version="0.1.0",
            unresolved_count=0,  # Will be updated below
        ),
        frames=FrameConventions(
            convention="NED",
            frames=["world", "body", "camera"],
            source=SourceTrace(section="§2.1"),
        ),
        spaces=Spaces(
            observation_exec=[
                SpaceField(
                    name="p_w",
                    dtype="float32",
                    shape=[3],
                    units="m",
                    source=SourceTrace(section="§3.1", table="Table 1"),
                ),
                SpaceField(
                    name="v_w",
                    dtype="float32",
                    shape=[3],
                    units="m/s",
                    source=SourceTrace(section="§3.1", table="Table 1"),
                ),
                SpaceField(
                    name="q_wb",
                    dtype="float32",
                    shape=[4],
                    units=None,
                    source=SourceTrace(section="§3.1", table="Table 1"),
                ),
            ],
            information_train=[
                InformedField(
                    name="gate_positions_w",
                    dtype="float32",
                    shape=[10, 3],
                    units="m",
                    informed_dreamer_key=r"^gate_.*",
                    training_only=True,
                    source=SourceTrace(section="§3.2"),
                ),
            ],
            action=ActionSpace(
                name="motor_thrusts",
                dtype="float32",
                shape=[4],
                bounds=(-1.0, 1.0),
                semantics="Normalised collective motor thrust commands.",
                source=SourceTrace(section="§3.3", table="Table 2"),
            ),
        ),
        timing=Timing(
            control_frequency_hz=50.0,
            sensor_delay_ms=10.0,
            action_delay_ms=5.0,
            timestamping_model="camera_anchored",
            sim_dt=0.001,
            source=SourceTrace(section="§4.1"),
        ),
        reward=Reward(
            discount=0.997,
            terms=[
                RewardTerm(
                    name="progress",
                    expression_ast=ExprNode(
                        type="op",
                        op="multiply",
                        args=[
                            ExprNode(type="literal", value=1.0),
                            ExprNode(
                                type="field",
                                name="gate_progress",
                                space="state",
                            ),
                        ],
                    ),
                    weight=1.0,
                    clamp=(-5.0, 5.0),
                    source=SourceTrace(equation="Eq. 3", section="§4.2"),
                ),
                RewardTerm(
                    name="collision_penalty",
                    expression_ast="UNRESOLVED",
                    weight="UNRESOLVED",
                    source=SourceTrace(section="§4.2"),
                ),
            ],
            normalization="symlog",
            source=SourceTrace(section="§4.2"),
        ),
        termination=Termination(
            conditions=[
                TerminationCondition(
                    name="ground_collision",
                    condition_ast=BoolNode(
                        type="compare",
                        op="lt",
                        left=ExprNode(
                            type="field",
                            name="height_above_ground",
                            space="state",
                        ),
                        right=ExprNode(type="literal", value=0.0),
                    ),
                    source=SourceTrace(section="§4.3"),
                ),
            ],
            max_episode_steps=1000,
            source=SourceTrace(section="§4.3"),
        ),
        dynamics=Dynamics(
            model_type="ODE",
            equations=[
                Equation(
                    name="translational_dynamics",
                    latex=r"\dot{v}_w = \frac{1}{m}(F_w - m g e_3)",
                    expression_ast="UNRESOLVED",
                    source=SourceTrace(equation="Eq. 1", section="§2.2"),
                ),
            ],
            integrator=IntegratorSpec(
                type="RK4",
                dt=0.001,
                source=SourceTrace(section="§2.2"),
            ),
            parameters=[
                DynamicsParam(
                    name="mass",
                    symbol="m",
                    default_value=0.752,
                    units="kg",
                    source=SourceTrace(table="Table 3", section="§A.1"),
                ),
                DynamicsParam(
                    name="arm_length",
                    symbol="l",
                    default_value="UNRESOLVED",
                    units="m",
                    source=SourceTrace(table="Table 3"),
                ),
            ],
            domain_randomization=[
                DomainRandEntry(
                    parameter="mass",
                    distribution="uniform",
                    range=(0.67, 0.84),
                    resample_frequency="per_episode",
                    source=SourceTrace(table="Table 4", section="§A.2"),
                ),
            ],
        ),
        world_model=WorldModel(
            architecture="RSSM",
            components={
                "encoder": ModelComponent(
                    type="CNN",
                    layers=["Conv2d(3,32,4,2)", "Conv2d(32,64,4,2)"],
                    source=SourceTrace(section="§5.1"),
                ),
                "sequence_model": ModelComponent(
                    type="GRU",
                    hidden_size=512,
                    source=SourceTrace(section="§5.1"),
                ),
            },
            discrete_latent=DiscreteLatent(
                num_categoricals=32,
                num_classes=32,
                source=SourceTrace(section="§5.1"),
            ),
            symlog=True,
            normalization="layer_norm",
            source=SourceTrace(section="§5.1"),
        ),
        actor_critic=ActorCritic(
            policy_distribution="squashed_normal",
            deterministic_eval=True,
            imagination_horizon=15,
            discount=0.997,
            lambda_gae=0.95,
            actor=NetworkSpec(
                hidden_layers=[512, 512, 512],
                activation="SiLU",
                source=SourceTrace(section="§5.2"),
            ),
            critic=NetworkSpec(
                hidden_layers=[512, 512, 512],
                activation="SiLU",
                source=SourceTrace(section="§5.2"),
            ),
            regularizers=[
                Regularizer(
                    name="action_smoothness",
                    coefficient=1e-4,
                    source=SourceTrace(section="§5.2"),
                ),
            ],
            source=SourceTrace(section="§5.2"),
        ),
        training=Training(
            algorithm="DreamerV3 + Informed decoding",
            replay=ReplayConfig(
                capacity_steps=2_000_000,
                context_length=64,
                sampling="uniform",
                source=SourceTrace(section="§5.3"),
            ),
            batch=BatchConfig(
                size=16,
                length=64,
                source=SourceTrace(section="§5.3"),
            ),
            schedule=TrainingSchedule(
                total_env_steps=5_000_000,
                phases=[
                    TrainingPhase(
                        name="warmup",
                        start_step=0,
                        end_step=100_000,
                        learning_rate=1e-4,
                        entropy_scale=3e-4,
                        source=SourceTrace(section="§5.3"),
                    ),
                ],
            ),
            train_ratio=64.0,
            optimizer=OptimizerConfig(
                type="Adam",
                lr=1e-4,
                eps=1e-8,
                clip_grad=100.0,
                source=SourceTrace(section="§5.3"),
            ),
            use_amp=True,
            source=SourceTrace(section="§5.3"),
        ),
        evaluation=Evaluation(
            success_criteria=[
                Criterion(
                    metric="gate_pass_rate",
                    threshold=0.9,
                    direction="higher_is_better",
                    source=SourceTrace(section="§6.1"),
                ),
            ],
            num_seeds=5,
            num_eval_episodes=100,
            report_error_bars=True,
            source=SourceTrace(section="§6.1"),
        ),
        deployment=Deployment(
            target_hardware="NVIDIA Jetson AGX Orin",
            inference_stack=["PyTorch", "CUDA"],
            runtime_budgets=[
                RuntimeBudget(
                    module="world_model_step",
                    max_ms=5.0,
                    source=SourceTrace(section="§7"),
                ),
            ],
            safety_constraints=[
                SafetyConstraint(
                    name="max_velocity",
                    description="Velocity must not exceed 20 m/s.",
                    source=SourceTrace(section="§7"),
                ),
            ],
            control_frequency_hz=50.0,
            source=SourceTrace(section="§7"),
        ),
        imports=[
            BaselineImport(
                name="dreamerv3",
                version="a1b2c3d4",
                spec_url="https://example.com/dreamerv3_spec.yaml",
                overrides=["world_model.discrete_latent", "training.optimizer"],
            ),
        ],
    )

    # Synchronise meta.unresolved_count with actual count
    minimal_spec.sync_unresolved_count()

    # ------------------------------------------------------------------
    # Run all validations
    # ------------------------------------------------------------------

    print("=" * 70)
    print("SpecIR Validation Demo")
    print("=" * 70)
    print()
    print(minimal_spec.summary())
    print()

    n_unresolved = minimal_spec.count_unresolved()
    print(f"count_unresolved() = {n_unresolved}")
    assert minimal_spec.meta.unresolved_count == n_unresolved, (
        f"meta.unresolved_count ({minimal_spec.meta.unresolved_count}) "
        f"!= count_unresolved() ({n_unresolved})"
    )
    print("  meta.unresolved_count matches count_unresolved()  [OK]")
    print()

    source_violations = minimal_spec.validate_sources()
    if source_violations:
        print("validate_sources() violations:")
        for v in source_violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("validate_sources() — no violations  [OK]")

    split_violations = minimal_spec.validate_informed_split()
    if split_violations:
        print("validate_informed_split() violations:")
        for v in split_violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("validate_informed_split() — no violations  [OK]")

    shape_violations = minimal_spec.validate_shapes()
    if shape_violations:
        # Shape violations are warnings only in this demo (symbolic dims may be
        # legitimately unresolved at extraction time).
        print("validate_shapes() warnings:")
        for v in shape_violations:
            print(f"  [WARN] {v}")
    else:
        print("validate_shapes() — no violations  [OK]")

    print()

    # ------------------------------------------------------------------
    # Round-trip serialisation
    # ------------------------------------------------------------------

    serialised = minimal_spec.to_dict()
    round_tripped = SpecIR.from_dict(serialised)

    assert round_tripped.meta.paper.title == minimal_spec.meta.paper.title
    assert round_tripped.count_unresolved() == n_unresolved
    print("to_dict() / from_dict() round-trip  [OK]")
    print()

    # Pretty-print a subset of the serialised dict
    print("Serialised reward section:")
    print(json.dumps(serialised.get("reward", {}), indent=2))
    print()
    print("Serialised meta section:")
    print(json.dumps(serialised.get("meta", {}), indent=2))
    print()
    print("All assertions passed.")
