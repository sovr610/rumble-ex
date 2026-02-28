"""
symbolic_config_template.py
===========================

Configuration dataclasses for the neuro-symbolic engine in a brain-inspired AI
system.  Every subsystem (fuzzy-logic operators, entity grounding, rule
networks, dataset adapters) has its own typed config, and ``SymbolicFullConfig``
aggregates them all into a single validated object that can be serialised to /
deserialised from JSON.

Scale presets mirror the broader BrainAI configuration hierarchy:

* ``SymbolicFullConfig.minimal()``       -- ~100K params, unit tests
* ``SymbolicFullConfig.dev()``           -- ~1M params, development
* ``SymbolicFullConfig.production_1b()`` -- sized for ~1B-param BrainAI
* ``SymbolicFullConfig.production_3b()`` -- sized for ~3B-param BrainAI
* ``SymbolicFullConfig.production_7b()`` -- sized for ~7B-param BrainAI

Run this file directly (``python symbolic_config_template.py``) to execute the
built-in self-test suite.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# 1. OperatorConfig
# ---------------------------------------------------------------------------

@dataclass
class OperatorConfig:
    """Configuration for fuzzy logic operator bundle.

    Bundles define how conjunction (AND), disjunction (OR), and negation (NOT)
    are realised in the differentiable logic layer:

    * **godel**          -- min / max / 1-x  (hard, non-differentiable at ties)
    * **product**        -- a*b / a+b-a*b / 1-x  (smooth but can saturate)
    * **lukasiewicz**    -- max(a+b-1,0) / min(a+b,1) / 1-x
    * **stable_product** -- product with clamped inputs in [eps, 1-eps]

    Parameters
    ----------
    bundle : str
        Operator bundle name.
    eps : float
        Stability epsilon used by ``stable_product`` to clamp truth values
        away from exact 0 / 1 before computing products.
    quantifier_p : float
        Exponent *p* for the generalised-mean quantifier
        (forall ~ p -> -inf, exists ~ p -> +inf).
    quantifier_temp : float
        Temperature scaling applied after the generalised-mean aggregation.
    implication_type : str
        Material implication variant used by rule evaluation.
    """

    bundle: str = "stable_product"
    eps: float = 1e-4
    quantifier_p: float = 2.0
    quantifier_temp: float = 1.0
    implication_type: str = "reichenbach"

    # valid option sets -------------------------------------------------------
    _VALID_BUNDLES = ("godel", "product", "lukasiewicz", "stable_product")
    _VALID_IMPLICATIONS = ("reichenbach", "godel", "goguen", "lukasiewicz")

    def validate(self) -> None:
        """Raise ``AssertionError`` if any field is out of range."""
        assert self.bundle in self._VALID_BUNDLES, (
            f"bundle must be one of {self._VALID_BUNDLES}, got '{self.bundle}'"
        )
        assert self.eps > 0, f"eps must be > 0, got {self.eps}"
        assert self.quantifier_p >= 1.0, (
            f"quantifier_p must be >= 1.0, got {self.quantifier_p}"
        )
        assert self.quantifier_temp > 0, (
            f"quantifier_temp must be > 0, got {self.quantifier_temp}"
        )
        assert self.implication_type in self._VALID_IMPLICATIONS, (
            f"implication_type must be one of {self._VALID_IMPLICATIONS}, "
            f"got '{self.implication_type}'"
        )


# ---------------------------------------------------------------------------
# 2. GroundingConfig
# ---------------------------------------------------------------------------

@dataclass
class GroundingConfig:
    """Configuration for entity extraction and predicate grounding.

    The grounding module takes a workspace-level representation (typically
    4096-dim in production) and produces a set of entity embeddings plus
    unary predicate and binary relation truth-value tensors.

    Parameters
    ----------
    entity_dim : int
        Dimensionality of each entity embedding vector.
    workspace_dim : int
        Dimensionality of the incoming workspace representation.
    hidden_dim : int
        Hidden layer size inside the grounding MLPs.
    extractor_type : str
        Entity extraction strategy.  ``"slot_identity"`` treats each slot as
        an entity; ``"proposal_head"`` uses a learned cross-attention head to
        propose entities from a set of queries.
    max_entities : int
        Maximum number of entities the system can track simultaneously.
    predicate_type : str
        Scoring function for unary predicates.
    relation_type : str
        Scoring function for binary relations.
    num_predicates : int
        Number of learnable unary predicates.
    num_relations : int
        Number of learnable binary relations.
    use_ltn : bool
        If True, use Logic Tensor Network grounding (requires ``ltn`` package).
    ntn_slices : int
        Number of tensor slices when ``predicate_type`` or ``relation_type``
        is ``"ntn"`` (Neural Tensor Network).
    dropout : float
        Dropout rate inside grounding networks.
    proposal_num_queries : int
        Number of learned query vectors for the proposal head extractor.
    proposal_num_heads : int
        Number of cross-attention heads in the proposal head extractor.
    dedup_threshold : float
        Cosine similarity threshold above which two entity proposals are
        considered duplicates and merged.
    """

    entity_dim: int = 256
    workspace_dim: int = 4096
    hidden_dim: int = 512
    extractor_type: str = "slot_identity"
    max_entities: int = 32
    predicate_type: str = "mlp"
    relation_type: str = "bilinear"
    num_predicates: int = 32
    num_relations: int = 16
    use_ltn: bool = False
    ntn_slices: int = 4
    dropout: float = 0.1
    proposal_num_queries: int = 16
    proposal_num_heads: int = 4
    dedup_threshold: float = 0.95

    # valid option sets -------------------------------------------------------
    _VALID_EXTRACTOR_TYPES = ("slot_identity", "proposal_head")
    _VALID_SCORING_TYPES = ("mlp", "bilinear", "ntn")

    def validate(self) -> None:
        """Raise ``AssertionError`` if any field is out of range."""
        assert self.entity_dim > 0, (
            f"entity_dim must be > 0, got {self.entity_dim}"
        )
        assert self.workspace_dim > 0, (
            f"workspace_dim must be > 0, got {self.workspace_dim}"
        )
        assert self.hidden_dim > 0, (
            f"hidden_dim must be > 0, got {self.hidden_dim}"
        )
        assert self.extractor_type in self._VALID_EXTRACTOR_TYPES, (
            f"extractor_type must be one of {self._VALID_EXTRACTOR_TYPES}, "
            f"got '{self.extractor_type}'"
        )
        assert self.max_entities > 0, (
            f"max_entities must be > 0, got {self.max_entities}"
        )
        assert self.predicate_type in self._VALID_SCORING_TYPES, (
            f"predicate_type must be one of {self._VALID_SCORING_TYPES}, "
            f"got '{self.predicate_type}'"
        )
        assert self.relation_type in self._VALID_SCORING_TYPES, (
            f"relation_type must be one of {self._VALID_SCORING_TYPES}, "
            f"got '{self.relation_type}'"
        )
        assert self.num_predicates > 0, (
            f"num_predicates must be > 0, got {self.num_predicates}"
        )
        assert self.num_relations > 0, (
            f"num_relations must be > 0, got {self.num_relations}"
        )
        assert self.ntn_slices > 0, (
            f"ntn_slices must be > 0, got {self.ntn_slices}"
        )
        assert 0.0 <= self.dropout < 1.0, (
            f"dropout must be in [0.0, 1.0), got {self.dropout}"
        )
        assert self.proposal_num_queries > 0, (
            f"proposal_num_queries must be > 0, got {self.proposal_num_queries}"
        )
        assert self.proposal_num_heads > 0, (
            f"proposal_num_heads must be > 0, got {self.proposal_num_heads}"
        )
        assert 0.0 < self.dedup_threshold <= 1.0, (
            f"dedup_threshold must be in (0.0, 1.0], got {self.dedup_threshold}"
        )


# ---------------------------------------------------------------------------
# 3. RuleConfig
# ---------------------------------------------------------------------------

@dataclass
class RuleConfig:
    """Configuration for rule network and constraint loss.

    The rule network maintains a bank of differentiable first-order rules.
    Each rule produces a truth value that is aggregated into a symbolic
    constraint loss:

        L_total = L_task + constraint_weight * L_symbolic

    Parameters
    ----------
    max_rules : int
        Maximum number of rules in the rule bank.
    use_attention : bool
        Whether to use attention-based rule composition.
    attention_dim : int
        Dimensionality of the attention key/query space.
    rule_embed_dim : int
        Dimensionality of each rule's learned embedding.
    violation_agg : str
        How individual rule violations are aggregated into a scalar loss.
        ``"mean"`` is a simple average; ``"soft_min"`` uses a temperature-
        scaled softmin; ``"p_mean"`` uses the generalised p-mean.
    p_mean_p : float
        Exponent for the generalised p-mean aggregation.
    constraint_weight : float
        Lambda multiplier for the symbolic constraint loss term.
    warmup_steps : int
        Number of training steps over which ``constraint_weight`` is linearly
        ramped from 0 to its configured value.
    top_k_logs : int
        Number of most-violated rules to log during training.
    full_tensor_logs : bool
        If True, log the full rule-satisfaction tensor (expensive).
    soft_min_temp : float
        Temperature parameter for the ``"soft_min"`` aggregation mode.
    """

    max_rules: int = 64
    use_attention: bool = True
    attention_dim: int = 256
    rule_embed_dim: int = 128
    violation_agg: str = "mean"
    p_mean_p: float = 2.0
    constraint_weight: float = 0.1
    warmup_steps: int = 1000
    top_k_logs: int = 5
    full_tensor_logs: bool = False
    soft_min_temp: float = 0.1

    # valid option sets -------------------------------------------------------
    _VALID_VIOLATION_AGGS = ("mean", "soft_min", "p_mean")

    def validate(self) -> None:
        """Raise ``AssertionError`` if any field is out of range."""
        assert self.max_rules > 0, (
            f"max_rules must be > 0, got {self.max_rules}"
        )
        assert self.attention_dim > 0, (
            f"attention_dim must be > 0, got {self.attention_dim}"
        )
        assert self.rule_embed_dim > 0, (
            f"rule_embed_dim must be > 0, got {self.rule_embed_dim}"
        )
        assert self.violation_agg in self._VALID_VIOLATION_AGGS, (
            f"violation_agg must be one of {self._VALID_VIOLATION_AGGS}, "
            f"got '{self.violation_agg}'"
        )
        assert self.p_mean_p >= 1.0, (
            f"p_mean_p must be >= 1.0, got {self.p_mean_p}"
        )
        assert self.constraint_weight >= 0.0, (
            f"constraint_weight must be >= 0.0, got {self.constraint_weight}"
        )
        assert self.warmup_steps >= 0, (
            f"warmup_steps must be >= 0, got {self.warmup_steps}"
        )
        assert self.top_k_logs >= 0, (
            f"top_k_logs must be >= 0, got {self.top_k_logs}"
        )
        assert self.soft_min_temp > 0, (
            f"soft_min_temp must be > 0, got {self.soft_min_temp}"
        )


# ---------------------------------------------------------------------------
# 4. SymbolicConfig
# ---------------------------------------------------------------------------

@dataclass
class SymbolicConfig:
    """Top-level symbolic reasoning configuration.

    This controls the high-level shape of the neuro-symbolic reasoning module
    that sits inside the BrainAI cognitive pipeline.

    Parameters
    ----------
    entity_dim : int
        Dimensionality of entity embeddings throughout the symbolic stack.
    num_predicates : int
        Number of learnable unary predicates.
    num_relations : int
        Number of learnable binary relations.
    use_ltn : bool
        If True, prefer Logic Tensor Network primitives when available.
    hidden_dim : int
        Hidden layer size for internal MLPs.
    output_dim : int
        Dimensionality of the symbolic reasoning output that is fed back
        into the global workspace.
    """

    entity_dim: int = 256
    num_predicates: int = 32
    num_relations: int = 16
    use_ltn: bool = False
    hidden_dim: int = 512
    output_dim: int = 256

    def validate(self) -> None:
        """Raise ``AssertionError`` if any field is out of range."""
        assert self.entity_dim > 0, (
            f"entity_dim must be > 0, got {self.entity_dim}"
        )
        assert self.num_predicates > 0, (
            f"num_predicates must be > 0, got {self.num_predicates}"
        )
        assert self.num_relations > 0, (
            f"num_relations must be > 0, got {self.num_relations}"
        )
        assert self.hidden_dim > 0, (
            f"hidden_dim must be > 0, got {self.hidden_dim}"
        )
        assert self.output_dim > 0, (
            f"output_dim must be > 0, got {self.output_dim}"
        )


# ---------------------------------------------------------------------------
# 5. DatasetConfig
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Configuration for dataset adapters.

    The neuro-symbolic engine supports three dataset backends:

    * **proofwriter** -- Microsoft ProofWriter benchmark (rule-based QA).
    * **folio**       -- FOLIO natural-language FOL benchmark.
    * **synthetic**   -- Procedurally generated entities, predicates, and
      rules for rapid prototyping and unit testing.

    Parameters
    ----------
    adapter_type : str
        Which dataset adapter to instantiate.
    data_dir : str
        Root directory for the dataset files.  Ignored by ``"synthetic"``.
    use_structured : bool
        ProofWriter-specific: prefer the structured (machine-readable)
        variant of the dataset over the natural-language variant.
    use_fol : bool
        FOLIO-specific: include first-order logic annotations alongside
        the natural-language premises and conclusions.
    max_depth : int or None
        ProofWriter-specific: maximum proof depth to include.  ``None``
        includes all depths.
    cache_dir : str
        Directory for caching preprocessed data.
    max_examples : int or None
        Cap the number of examples loaded (useful for debugging).
    synthetic_num_entities : int
        Number of entities in the synthetic world.
    synthetic_num_predicates : int
        Number of unary predicates in the synthetic world.
    synthetic_num_relations : int
        Number of binary relations in the synthetic world.
    synthetic_num_rules : int
        Number of rules in the synthetic world.
    synthetic_seed : int
        Random seed for reproducible synthetic data generation.
    """

    adapter_type: str = "synthetic"
    data_dir: str = ""
    use_structured: bool = True
    use_fol: bool = True
    max_depth: Optional[int] = None
    cache_dir: str = ".cache/symbolic"
    max_examples: Optional[int] = None
    synthetic_num_entities: int = 10
    synthetic_num_predicates: int = 5
    synthetic_num_relations: int = 3
    synthetic_num_rules: int = 5
    synthetic_seed: int = 42

    # valid option sets -------------------------------------------------------
    _VALID_ADAPTER_TYPES = ("proofwriter", "folio", "synthetic")

    def validate(self) -> None:
        """Raise ``AssertionError`` if any field is out of range."""
        assert self.adapter_type in self._VALID_ADAPTER_TYPES, (
            f"adapter_type must be one of {self._VALID_ADAPTER_TYPES}, "
            f"got '{self.adapter_type}'"
        )
        if self.max_depth is not None:
            assert self.max_depth > 0, (
                f"max_depth must be > 0 or None, got {self.max_depth}"
            )
        if self.max_examples is not None:
            assert self.max_examples > 0, (
                f"max_examples must be > 0 or None, got {self.max_examples}"
            )
        assert self.synthetic_num_entities > 0, (
            f"synthetic_num_entities must be > 0, got {self.synthetic_num_entities}"
        )
        assert self.synthetic_num_predicates > 0, (
            f"synthetic_num_predicates must be > 0, got {self.synthetic_num_predicates}"
        )
        assert self.synthetic_num_relations > 0, (
            f"synthetic_num_relations must be > 0, got {self.synthetic_num_relations}"
        )
        assert self.synthetic_num_rules > 0, (
            f"synthetic_num_rules must be > 0, got {self.synthetic_num_rules}"
        )


# ---------------------------------------------------------------------------
# 6. SymbolicFullConfig (aggregation)
# ---------------------------------------------------------------------------

@dataclass
class SymbolicFullConfig:
    """Aggregated configuration for the entire neuro-symbolic engine.

    Composes all sub-configs and provides factory class methods for standard
    scale presets that mirror the broader ``BrainAIConfig`` hierarchy.

    Attributes
    ----------
    symbolic : SymbolicConfig
        Top-level symbolic reasoning dimensions.
    operator : OperatorConfig
        Fuzzy logic operator bundle selection and tuning.
    grounding : GroundingConfig
        Entity extraction and predicate grounding parameters.
    rule : RuleConfig
        Rule bank and constraint loss parameters.
    dataset : DatasetConfig
        Dataset adapter selection and parameters.
    """

    symbolic: SymbolicConfig = field(default_factory=SymbolicConfig)
    operator: OperatorConfig = field(default_factory=OperatorConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    rule: RuleConfig = field(default_factory=RuleConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # -- validation -----------------------------------------------------------

    def validate(self) -> None:
        """Validate every sub-config and cross-config consistency."""
        self.symbolic.validate()
        self.operator.validate()
        self.grounding.validate()
        self.rule.validate()
        self.dataset.validate()
        # Cross-config dimensional consistency
        assert self.symbolic.entity_dim == self.grounding.entity_dim, (
            f"symbolic.entity_dim ({self.symbolic.entity_dim}) must match "
            f"grounding.entity_dim ({self.grounding.entity_dim})"
        )
        assert self.symbolic.num_predicates == self.grounding.num_predicates, (
            f"symbolic.num_predicates ({self.symbolic.num_predicates}) must match "
            f"grounding.num_predicates ({self.grounding.num_predicates})"
        )
        assert self.symbolic.num_relations == self.grounding.num_relations, (
            f"symbolic.num_relations ({self.symbolic.num_relations}) must match "
            f"grounding.num_relations ({self.grounding.num_relations})"
        )
        assert self.symbolic.use_ltn == self.grounding.use_ltn, (
            f"symbolic.use_ltn ({self.symbolic.use_ltn}) must match "
            f"grounding.use_ltn ({self.grounding.use_ltn})"
        )

    # -- scale presets --------------------------------------------------------

    @classmethod
    def minimal(cls) -> SymbolicFullConfig:
        """~100K params, for unit tests."""
        return cls(
            symbolic=SymbolicConfig(
                entity_dim=64,
                num_predicates=4,
                num_relations=2,
                hidden_dim=128,
                output_dim=64,
            ),
            operator=OperatorConfig(bundle="stable_product"),
            grounding=GroundingConfig(
                entity_dim=64,
                workspace_dim=256,
                hidden_dim=128,
                max_entities=8,
                num_predicates=4,
                num_relations=2,
            ),
            rule=RuleConfig(
                max_rules=8,
                attention_dim=64,
                rule_embed_dim=32,
                warmup_steps=10,
            ),
            dataset=DatasetConfig(
                adapter_type="synthetic",
                synthetic_num_entities=5,
                synthetic_num_predicates=3,
                synthetic_num_relations=2,
                synthetic_num_rules=3,
            ),
        )

    @classmethod
    def dev(cls) -> SymbolicFullConfig:
        """~1M params, for development."""
        return cls(
            symbolic=SymbolicConfig(
                entity_dim=128,
                num_predicates=16,
                num_relations=8,
                hidden_dim=256,
                output_dim=128,
            ),
            operator=OperatorConfig(bundle="stable_product"),
            grounding=GroundingConfig(
                entity_dim=128,
                workspace_dim=512,
                hidden_dim=256,
                max_entities=16,
                num_predicates=16,
                num_relations=8,
            ),
            rule=RuleConfig(
                max_rules=32,
                attention_dim=128,
                rule_embed_dim=64,
            ),
            dataset=DatasetConfig(adapter_type="synthetic"),
        )

    @classmethod
    def production_1b(cls) -> SymbolicFullConfig:
        """Scaled for ~1B param BrainAI."""
        return cls(
            symbolic=SymbolicConfig(
                entity_dim=256,
                num_predicates=32,
                num_relations=16,
                hidden_dim=512,
                output_dim=256,
            ),
            operator=OperatorConfig(bundle="stable_product"),
            grounding=GroundingConfig(
                entity_dim=256,
                workspace_dim=2048,
                hidden_dim=512,
                max_entities=32,
                num_predicates=32,
                num_relations=16,
            ),
            rule=RuleConfig(
                max_rules=64,
                attention_dim=256,
                rule_embed_dim=128,
            ),
            dataset=DatasetConfig(adapter_type="synthetic"),
        )

    @classmethod
    def production_3b(cls) -> SymbolicFullConfig:
        """Scaled for ~3B param BrainAI."""
        return cls(
            symbolic=SymbolicConfig(
                entity_dim=384,
                num_predicates=48,
                num_relations=24,
                hidden_dim=768,
                output_dim=384,
            ),
            operator=OperatorConfig(bundle="stable_product"),
            grounding=GroundingConfig(
                entity_dim=384,
                workspace_dim=3072,
                hidden_dim=768,
                max_entities=48,
                num_predicates=48,
                num_relations=24,
            ),
            rule=RuleConfig(
                max_rules=96,
                attention_dim=384,
                rule_embed_dim=192,
            ),
            dataset=DatasetConfig(adapter_type="synthetic"),
        )

    @classmethod
    def production_7b(cls) -> SymbolicFullConfig:
        """Scaled for ~7B param BrainAI."""
        return cls(
            symbolic=SymbolicConfig(
                entity_dim=512,
                num_predicates=64,
                num_relations=32,
                hidden_dim=1024,
                output_dim=512,
            ),
            operator=OperatorConfig(bundle="stable_product"),
            grounding=GroundingConfig(
                entity_dim=512,
                workspace_dim=4096,
                hidden_dim=1024,
                max_entities=64,
                num_predicates=64,
                num_relations=32,
            ),
            rule=RuleConfig(
                max_rules=128,
                attention_dim=512,
                rule_embed_dim=256,
            ),
            dataset=DatasetConfig(adapter_type="synthetic"),
        )


# ---------------------------------------------------------------------------
# 7. Serialization helpers
# ---------------------------------------------------------------------------

def config_to_dict(config: SymbolicFullConfig) -> Dict[str, Any]:
    """Convert a ``SymbolicFullConfig`` to a JSON-serializable dict.

    Private fields (prefixed with ``_``) are excluded from the output.

    Parameters
    ----------
    config : SymbolicFullConfig
        The configuration to serialise.

    Returns
    -------
    Dict[str, Any]
        Nested dictionary representation.
    """
    raw: Dict[str, Any] = asdict(config)
    # Strip private/internal keys that dataclasses.asdict may include
    # (class-level constants defined as ClassVar are not captured by asdict,
    #  but we guard against it defensively).
    def _strip_private(d: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: _strip_private(v) if isinstance(v, dict) else v
            for k, v in d.items()
            if not k.startswith("_")
        }
    return _strip_private(raw)


def config_from_dict(d: Dict[str, Any]) -> SymbolicFullConfig:
    """Reconstruct a ``SymbolicFullConfig`` from a dict (e.g. loaded JSON).

    Parameters
    ----------
    d : Dict[str, Any]
        Nested dictionary previously produced by ``config_to_dict``.

    Returns
    -------
    SymbolicFullConfig
        Reconstructed configuration object.
    """
    symbolic_d = d.get("symbolic", {})
    operator_d = d.get("operator", {})
    grounding_d = d.get("grounding", {})
    rule_d = d.get("rule", {})
    dataset_d = d.get("dataset", {})

    return SymbolicFullConfig(
        symbolic=SymbolicConfig(**symbolic_d),
        operator=OperatorConfig(**operator_d),
        grounding=GroundingConfig(**grounding_d),
        rule=RuleConfig(**rule_d),
        dataset=DatasetConfig(**dataset_d),
    )


def save_config(config: SymbolicFullConfig, path: str) -> None:
    """Save a ``SymbolicFullConfig`` to a JSON file.

    Parameters
    ----------
    config : SymbolicFullConfig
        The configuration to persist.
    path : str
        Filesystem path for the output JSON file.  Parent directories are
        created automatically if they do not exist.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config_to_dict(config), fh, indent=2, sort_keys=False)


def load_config(path: str) -> SymbolicFullConfig:
    """Load a ``SymbolicFullConfig`` from a JSON file.

    Parameters
    ----------
    path : str
        Filesystem path to a JSON file previously written by ``save_config``.

    Returns
    -------
    SymbolicFullConfig
        The deserialized configuration.
    """
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return config_from_dict(d)


# ---------------------------------------------------------------------------
# 8. Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    """Execute the built-in self-test suite.

    Each test prints PASS or FAIL.  The process exits with code 1 on the
    first failure.
    """

    passed = 0
    failed = 0

    def _check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            msg = f"  FAIL  {name}"
            if detail:
                msg += f"  -- {detail}"
            print(msg)
            failed += 1

    print("=" * 72)
    print("Neuro-Symbolic Config Template -- Self-Tests")
    print("=" * 72)

    # 1. Each config dataclass instantiates with defaults --------------------
    print("\n--- Test 1: default instantiation ---")
    try:
        oc = OperatorConfig()
        gc = GroundingConfig()
        rc = RuleConfig()
        sc = SymbolicConfig()
        dc = DatasetConfig()
        fc = SymbolicFullConfig()
        _check("default_instantiation", True)
    except Exception as exc:
        _check("default_instantiation", False, str(exc))

    # 2. Each config validates successfully with defaults --------------------
    print("\n--- Test 2: default validation ---")
    try:
        OperatorConfig().validate()
        GroundingConfig().validate()
        RuleConfig().validate()
        SymbolicConfig().validate()
        DatasetConfig().validate()
        SymbolicFullConfig().validate()
        _check("default_validation", True)
    except AssertionError as exc:
        _check("default_validation", False, str(exc))

    # 3. Invalid bundle name raises assertion --------------------------------
    print("\n--- Test 3: invalid operator bundle ---")
    try:
        OperatorConfig(bundle="invalid_bundle").validate()
        _check("invalid_bundle_raises", False, "No assertion raised")
    except AssertionError:
        _check("invalid_bundle_raises", True)

    # 4. Invalid eps raises assertion ----------------------------------------
    print("\n--- Test 4: invalid eps ---")
    try:
        OperatorConfig(eps=-1.0).validate()
        _check("invalid_eps_raises", False, "No assertion raised")
    except AssertionError:
        _check("invalid_eps_raises", True)

    # 5. SymbolicFullConfig.minimal() creates valid config -------------------
    print("\n--- Test 5: minimal preset ---")
    try:
        cfg = SymbolicFullConfig.minimal()
        cfg.validate()
        _check("minimal_preset_valid", True)
    except AssertionError as exc:
        _check("minimal_preset_valid", False, str(exc))

    # 6. SymbolicFullConfig.dev() creates valid config -----------------------
    print("\n--- Test 6: dev preset ---")
    try:
        cfg = SymbolicFullConfig.dev()
        cfg.validate()
        _check("dev_preset_valid", True)
    except AssertionError as exc:
        _check("dev_preset_valid", False, str(exc))

    # 7. SymbolicFullConfig.production_1b() creates valid config -------------
    print("\n--- Test 7: production_1b preset ---")
    try:
        cfg = SymbolicFullConfig.production_1b()
        cfg.validate()
        _check("production_1b_preset_valid", True)
    except AssertionError as exc:
        _check("production_1b_preset_valid", False, str(exc))

    # 8. SymbolicFullConfig.production_3b() creates valid config -------------
    print("\n--- Test 8: production_3b preset ---")
    try:
        cfg = SymbolicFullConfig.production_3b()
        cfg.validate()
        _check("production_3b_preset_valid", True)
    except AssertionError as exc:
        _check("production_3b_preset_valid", False, str(exc))

    # 9. SymbolicFullConfig.production_7b() creates valid config -------------
    print("\n--- Test 9: production_7b preset ---")
    try:
        cfg = SymbolicFullConfig.production_7b()
        cfg.validate()
        _check("production_7b_preset_valid", True)
    except AssertionError as exc:
        _check("production_7b_preset_valid", False, str(exc))

    # 10. Config round-trip: to_dict -> from_dict matches original -----------
    print("\n--- Test 10: dict round-trip ---")
    try:
        for preset_name, preset_fn in [
            ("minimal", SymbolicFullConfig.minimal),
            ("dev", SymbolicFullConfig.dev),
            ("production_1b", SymbolicFullConfig.production_1b),
            ("production_3b", SymbolicFullConfig.production_3b),
            ("production_7b", SymbolicFullConfig.production_7b),
        ]:
            original = preset_fn()
            d = config_to_dict(original)
            restored = config_from_dict(d)
            d2 = config_to_dict(restored)
            assert d == d2, (
                f"Round-trip mismatch for preset '{preset_name}'"
            )
        _check("dict_round_trip", True)
    except AssertionError as exc:
        _check("dict_round_trip", False, str(exc))

    # 11. Config save/load round-trip (using tempfile) -----------------------
    print("\n--- Test 11: file save/load round-trip ---")
    try:
        original = SymbolicFullConfig.production_1b()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as tmp:
            tmp_path = tmp.name
        try:
            save_config(original, tmp_path)
            loaded = load_config(tmp_path)
            assert config_to_dict(original) == config_to_dict(loaded), (
                "File round-trip produced different config"
            )
            _check("file_round_trip", True)
        finally:
            os.unlink(tmp_path)
    except (AssertionError, OSError) as exc:
        _check("file_round_trip", False, str(exc))

    # 12. All presets have consistent dimensions -----------------------------
    print("\n--- Test 12: cross-config dimensional consistency ---")
    try:
        for preset_name, preset_fn in [
            ("minimal", SymbolicFullConfig.minimal),
            ("dev", SymbolicFullConfig.dev),
            ("production_1b", SymbolicFullConfig.production_1b),
            ("production_3b", SymbolicFullConfig.production_3b),
            ("production_7b", SymbolicFullConfig.production_7b),
        ]:
            cfg = preset_fn()
            # entity_dim must match between symbolic and grounding
            assert cfg.symbolic.entity_dim == cfg.grounding.entity_dim, (
                f"{preset_name}: entity_dim mismatch "
                f"({cfg.symbolic.entity_dim} vs {cfg.grounding.entity_dim})"
            )
            # num_predicates must match
            assert cfg.symbolic.num_predicates == cfg.grounding.num_predicates, (
                f"{preset_name}: num_predicates mismatch "
                f"({cfg.symbolic.num_predicates} vs {cfg.grounding.num_predicates})"
            )
            # num_relations must match
            assert cfg.symbolic.num_relations == cfg.grounding.num_relations, (
                f"{preset_name}: num_relations mismatch "
                f"({cfg.symbolic.num_relations} vs {cfg.grounding.num_relations})"
            )
            # use_ltn must match
            assert cfg.symbolic.use_ltn == cfg.grounding.use_ltn, (
                f"{preset_name}: use_ltn mismatch"
            )
        _check("dimensional_consistency", True)
    except AssertionError as exc:
        _check("dimensional_consistency", False, str(exc))

    # 13 (bonus). Invalid violation_agg raises assertion ---------------------
    print("\n--- Test 13: invalid violation_agg ---")
    try:
        RuleConfig(violation_agg="invalid_agg").validate()
        _check("invalid_violation_agg_raises", False, "No assertion raised")
    except AssertionError:
        _check("invalid_violation_agg_raises", True)

    # 14 (bonus). Invalid extractor_type raises assertion --------------------
    print("\n--- Test 14: invalid extractor_type ---")
    try:
        GroundingConfig(extractor_type="bad_extractor").validate()
        _check("invalid_extractor_type_raises", False, "No assertion raised")
    except AssertionError:
        _check("invalid_extractor_type_raises", True)

    # 15 (bonus). Invalid adapter_type raises assertion ----------------------
    print("\n--- Test 15: invalid adapter_type ---")
    try:
        DatasetConfig(adapter_type="nonexistent").validate()
        _check("invalid_adapter_type_raises", False, "No assertion raised")
    except AssertionError:
        _check("invalid_adapter_type_raises", True)

    # 16 (bonus). Cross-config mismatch detected by SymbolicFullConfig ------
    print("\n--- Test 16: cross-config mismatch detection ---")
    try:
        bad_cfg = SymbolicFullConfig(
            symbolic=SymbolicConfig(entity_dim=64),
            grounding=GroundingConfig(entity_dim=128),  # mismatch!
        )
        bad_cfg.validate()
        _check("cross_config_mismatch_raises", False, "No assertion raised")
    except AssertionError:
        _check("cross_config_mismatch_raises", True)

    # -- summary -------------------------------------------------------------
    print("\n" + "=" * 72)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 72)

    if failed > 0:
        raise SystemExit(1)
    print("\nAll tests passed.")


if __name__ == "__main__":
    _run_self_tests()
