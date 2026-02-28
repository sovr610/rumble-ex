# Dataset Adapters Reference

## Table of Contents

1. [Overview](#1-overview)
2. [Common Adapter Interface](#2-common-adapter-interface)
3. [ProofWriter Adapter](#3-proofwriter-adapter)
4. [FOLIO Adapter](#4-folio-adapter)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Entity Embedding Strategies](#6-entity-embedding-strategies)
7. [Integration with Training Loop](#7-integration-with-training-loop)
8. [Code Patterns](#8-code-patterns)
9. [Appendix A: ProofWriter Example Walkthrough](#appendix-a-proofwriter-example-walkthrough)
10. [Appendix B: FOLIO Example Walkthrough](#appendix-b-folio-example-walkthrough)
11. [Appendix C: Supported FOL Operators and AST Mappings](#appendix-c-supported-fol-operators-and-ast-mappings)

---

## 1. Overview

The dataset adapter layer bridges external symbolic reasoning benchmarks and the
Neuro-Symbolic Engine's internal representation. Two adapters are provided:
**ProofWriter** (multi-hop entailment over English rulebases) and **FOLIO**
(first-order logic inference with human-annotated FOL formulas). Both adapters
parse raw dataset examples into a unified `SymbolicExample` schema, which feeds
directly into `SymbolicReasoner.forward()`.

### Design Goals

- **Uniform output**: Both adapters produce the same `SymbolicExample` and
  `SymbolicBatch` types, so the training loop and reasoner are dataset-agnostic.
- **Lossless logic preservation**: Parsing must preserve the logical structure of
  the source data. Fallback heuristics are permitted only when the original
  sentence or formula is unparseable, and failures must be logged.
- **Preprocessing once, train many**: Parsed ASTs and entity vocabularies are
  cached to disk so that repeated training runs skip the parsing phase entirely.
- **Graceful degradation**: When an example cannot be fully parsed, the adapter
  emits a valid `SymbolicExample` with as many rules as could be extracted, plus
  a `metadata["parse_warnings"]` list for debugging.

### Target Module

The adapters live in `brain_ai/reasoning/dataset_adapters.py` and are
instantiated by name through the `ADAPTER_REGISTRY`:

```python
ADAPTER_REGISTRY: Dict[str, Type[SymbolicDatasetAdapter]] = {
    "proofwriter": ProofWriterAdapter,
    "folio": FOLIOAdapter,
}
```

---

## 2. Common Adapter Interface

Both adapters implement `SymbolicDatasetAdapter`, ensuring identical call
signatures and output types across datasets.

### SymbolicDatasetAdapter Base Class

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path


class SymbolicDatasetAdapter(ABC):
    """Base class for all symbolic dataset adapters."""

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        max_entities: int = 32,
        max_rules: int = 64,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / ".cache"
        self.max_entities = max_entities
        self.max_rules = max_rules

    @abstractmethod
    def load(self, split: str = "train") -> List[SymbolicExample]:
        """Load and parse a dataset split into SymbolicExample objects.

        Args:
            split: One of "train", "val", "test".

        Returns:
            List of parsed examples. Each example is self-contained.
        """
        ...

    @abstractmethod
    def collate(self, examples: List[SymbolicExample]) -> SymbolicBatch:
        """Collate a list of examples into a batched tensor structure.

        Handles padding, masking, and variable-length rule lists.

        Args:
            examples: List of SymbolicExample objects (one mini-batch).

        Returns:
            SymbolicBatch with padded tensors and masks.
        """
        ...
```

### SymbolicExample Dataclass

Every adapter produces `SymbolicExample` instances. One instance corresponds to
one dataset row (one theory + one query).

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class SymbolicExample:
    """A single symbolic reasoning example."""

    entities: List[str]
    """Entity names appearing in this example.
    Example: ["cat", "dog", "mouse"]."""

    predicates: Dict[str, int]
    """Mapping from predicate name to arity.
    Example: {"is_blue": 1, "chases": 2, "is_cold": 1}."""

    rules: List[RuleAST]
    """Compiled rule ASTs forming the theory/knowledge base.
    Includes both facts (zero-premise rules) and conditional rules."""

    query: QueryAST
    """The compiled query AST representing the question to evaluate."""

    label: int
    """Ground-truth label.
    0 = contradicted/false, 1 = entailed/true, 2 = unknown."""

    label_float: float
    """Soft label for BCE training.
    0.0 = contradicted, 1.0 = entailed, 0.5 = unknown."""

    proof_chain: Optional[List[int]]
    """Ordered list of rule indices forming the proof. None when proof
    supervision is unavailable or the label is unknown."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Auxiliary metadata: source dataset, example ID, proof depth,
    parse warnings, original text, etc."""
```

### SymbolicExample Field Contracts

| Field | Type | Constraints |
|---|---|---|
| `entities` | `List[str]` | Length in `[1, max_entities]`. No duplicates. |
| `predicates` | `Dict[str, int]` | Keys are normalized predicate names. Values are arity (1 for unary, 2 for binary). |
| `rules` | `List[RuleAST]` | Length in `[0, max_rules]`. Empty only if the theory has no parseable rules. |
| `query` | `QueryAST` | Always present. Must reference only entities and predicates from this example. |
| `label` | `int` | One of `{0, 1, 2}`. |
| `label_float` | `float` | One of `{0.0, 0.5, 1.0}`. |
| `proof_chain` | `Optional[List[int]]` | Indices into the `rules` list. None when unavailable. |
| `metadata` | `Dict[str, Any]` | Must contain `"source"` (dataset name) and `"example_id"` (unique identifier). |

### SymbolicBatch Dataclass

The collate function produces `SymbolicBatch`, which holds padded tensors
suitable for batched forward passes through the reasoner.

```python
import torch
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SymbolicBatch:
    """Batched symbolic reasoning data, padded and masked."""

    entity_ids: torch.LongTensor
    """(B, max_N) padded entity ID tensor for embedding lookup.
    Padding positions use ID 0 (reserved for <PAD>)."""

    entity_mask: torch.BoolTensor
    """(B, max_N) boolean mask. True for valid entities, False for padding."""

    rules: List[List[RuleAST]]
    """List of B rule lists. Each inner list has variable length.
    Not tensorized because rule ASTs have heterogeneous structure."""

    queries: List[QueryAST]
    """List of B query ASTs, one per batch item."""

    labels: torch.FloatTensor
    """(B,) soft truth labels for the queries."""

    label_ids: torch.LongTensor
    """(B,) categorical label IDs (0=contradicted, 1=entailed, 2=unknown)."""

    proof_chains: Optional[List[Optional[List[int]]]]
    """List of B proof chains. None entries for examples without proof supervision."""

    metadata: List[Dict[str, Any]]
    """List of B metadata dicts, one per example."""
```

### SymbolicBatch Shape Summary

| Field | Shape / Type | Padding Strategy |
|---|---|---|
| `entity_ids` | `(B, max_N)` | Pad with 0 (`<PAD>` token) |
| `entity_mask` | `(B, max_N)` | `False` at padding positions |
| `rules` | `List[List[RuleAST]]` | Variable-length lists (no padding) |
| `queries` | `List[QueryAST]` | One per batch item (no padding) |
| `labels` | `(B,)` | No padding needed |
| `label_ids` | `(B,)` | No padding needed |
| `proof_chains` | `List[Optional[List[int]]]` | None entries preserved |

---

## 3. ProofWriter Adapter

### Dataset Background

ProofWriter is a synthetic dataset for multi-hop logical reasoning. Each example
contains a small English rulebase (facts and rules), a natural language question,
and an answer (True, False, or Unknown). The dataset is stratified by proof depth
(0, 1, 2, 3, 5 hops), controlling reasoning complexity. Structured variants
provide logic forms directly, reducing parsing ambiguity.

### Data Format

Each raw example contains three fields:

| Field | Type | Description |
|---|---|---|
| `context` | `str` | English paragraph listing facts and rules. Facts are simple declarative sentences. Rules use "if...then..." patterns. |
| `question` | `str` | English question, typically "The cat is blue." with a True/False/Unknown answer expected. |
| `answer` | `str` | One of `"True"`, `"False"`, `"Unknown"`. |

Structured variants add a fourth field:

| Field | Type | Description |
|---|---|---|
| `triples` | `Dict[str, Dict]` | Pre-extracted entity-predicate triples for facts. |
| `rules_structured` | `List[Dict]` | Pre-extracted rule structures with head/body. |
| `proof` | `List[str]` | Ordered list of rule/fact identifiers forming the proof. |

### Proof Depth Splits

| Depth | Reasoning Steps | Typical Theory Size | Use Case |
|---|---|---|---|
| 0 | Direct lookup | 5-10 facts, 2-5 rules | Baseline, sanity check |
| 1 | Single rule application | 5-15 facts, 3-8 rules | Basic chaining |
| 2 | Two-step chain | 8-20 facts, 5-12 rules | Standard multi-hop |
| 3 | Three-step chain | 10-25 facts, 8-15 rules | Complex reasoning |
| 5 | Five-step chain | 15-30 facts, 10-20 rules | Stress test |

### Adapter Responsibilities

The `ProofWriterAdapter` performs four tasks:

1. **Parse facts and rules** from English (or structured variants) into
   constrained logic form.
2. **Build a per-example theory**: a list of `RuleAST` objects representing
   premises and implications.
3. **Generate the query**: compile the question into a `QueryAST`.
4. **Extract proof supervision** (when available): map proof step identifiers to
   rule indices in the compiled theory.

### Parsing Pipeline

#### Step 1: Fact Extraction

Convert declarative English sentences into atomic predicates.

| English Pattern | Parsed Form | AST Node |
|---|---|---|
| `"The cat is blue."` | `is_blue(cat)` | `Atom("is_blue", ["cat"])` |
| `"The cat chases the dog."` | `chases(cat, dog)` | `Atom("chases", ["cat", "dog"])` |
| `"The cat is not blue."` | `NOT is_blue(cat)` | `Not(Atom("is_blue", ["cat"]))` |
| `"The cat is big and blue."` | `is_big(cat) AND is_blue(cat)` | `And(Atom("is_big", ["cat"]), Atom("is_blue", ["cat"]))` |

Fact extraction uses a deterministic pattern matcher that identifies the subject
noun phrase, the copula or verb, and the object or adjective complement.

```python
FACT_PATTERNS = [
    # "The X is Y." -> is_Y(X)
    (r"^The (\w+) is (\w+)\.$",
     lambda m: Atom(f"is_{m.group(2)}", [m.group(1)])),
    # "The X is not Y." -> NOT is_Y(X)
    (r"^The (\w+) is not (\w+)\.$",
     lambda m: Not(Atom(f"is_{m.group(2)}", [m.group(1)]))),
    # "The X VERBs the Y." -> VERB(X, Y)
    (r"^The (\w+) (\w+)s the (\w+)\.$",
     lambda m: Atom(m.group(2), [m.group(1), m.group(3)])),
]
```

#### Step 2: Rule Extraction

Convert conditional English sentences into universally quantified implications.

| English Pattern | Parsed Form |
|---|---|
| `"If something is blue then it is cold."` | `FORALL x: is_blue(x) IMPLIES is_cold(x)` |
| `"If something chases the dog then it is big."` | `FORALL x: chases(x, dog) IMPLIES is_big(x)` |
| `"If something is blue and big then it is cold."` | `FORALL x: (is_blue(x) AND is_big(x)) IMPLIES is_cold(x)` |
| `"If something is cold then it chases the dog."` | `FORALL x: is_cold(x) IMPLIES chases(x, dog)` |

Rule extraction parses the antecedent ("if" clause) and consequent ("then"
clause) separately, then wraps the result in a universal quantifier over the
implicit variable.

```python
RULE_PATTERNS = [
    # "If something is X then it is Y."
    (r"^If something is (\w+) then it is (\w+)\.$",
     lambda m: ForAll("x", Implies(
         Atom(f"is_{m.group(1)}", ["x"]),
         Atom(f"is_{m.group(2)}", ["x"]),
     ))),
    # "If something is X and Y then it is Z."
    (r"^If something is (\w+) and (\w+) then it is (\w+)\.$",
     lambda m: ForAll("x", Implies(
         And(Atom(f"is_{m.group(1)}", ["x"]),
             Atom(f"is_{m.group(2)}", ["x"])),
         Atom(f"is_{m.group(3)}", ["x"]),
     ))),
]
```

#### Step 3: Entity Resolution

Map noun phrases to canonical entity IDs. The resolver maintains a per-example
entity table and handles:

- **Definite references**: "The cat" maps to entity `"cat"`.
- **Pronoun binding**: "it" in rules binds to the quantified variable.
- **Implicit variables**: "something" introduces a universally quantified variable.

| Noun Phrase | Resolution |
|---|---|
| `"The cat"` | Entity ID `"cat"` |
| `"something"` | Variable `"x"` (quantifier-bound) |
| `"it"` | Variable `"x"` (back-reference to "something") |
| `"the dog"` | Entity ID `"dog"` |

#### Step 4: Predicate Normalization

Standardize predicate names across examples to build a shared vocabulary.

- Convert to snake_case: `"is blue"` becomes `"is_blue"`.
- Strip articles: `"chases the"` becomes `"chases"`.
- Merge synonyms (configurable): `"is big"` and `"is large"` map to `"is_big"`.
- Record arity: unary predicates take one argument, binary predicates take two.

#### Step 5: Error Handling

When a sentence cannot be matched by any pattern:

1. Log a warning with the unparseable sentence text and example ID.
2. Skip the sentence (do not include it in the theory).
3. Add `"parse_warnings"` to `metadata` with the list of failed sentences.
4. If a structured variant is available, fall back to parsing the structured form.
5. If neither English nor structured parsing succeeds for the query, mark the
   example with `metadata["skip"] = True` and exclude it from training.

### Output Schema

Each parsed ProofWriter example produces:

| Field | Source | Notes |
|---|---|---|
| `entities` | Extracted from facts and rules | Deduplicated, sorted alphabetically |
| `predicates` | Normalized from verbs/adjectives | Arity auto-detected |
| `rules` | Facts (as zero-premise rules) + conditional rules | Facts become `Atom(...)`, rules become `ForAll(Implies(...))` |
| `query` | Parsed question sentence | Single `Atom` or `Not(Atom)` |
| `label` | Mapped from answer string | `"True"` -> 1, `"False"` -> 0, `"Unknown"` -> 2 |
| `proof_chain` | Mapped from proof identifiers (if available) | Indices into `rules` list |
| `metadata` | Source info | `{"source": "proofwriter", "example_id": ..., "depth": ..., "parse_warnings": [...]}` |

---

## 4. FOLIO Adapter

### Dataset Background

FOLIO (First-Order Logic Inference) is a human-annotated dataset pairing natural
language premises and conclusions with first-order logic (FOL) annotations. Each
example contains a set of NL premises, a NL conclusion, FOL translations for
each sentence, and a label indicating whether the conclusion is entailed,
contradicted, or neutral given the premises. The FOL annotations are verified
against an automated theorem prover, making them the authoritative source of
logical structure.

### Data Format

Each raw example contains these fields:

| Field | Type | Description |
|---|---|---|
| `premises` | `List[str]` | Natural language premise sentences. |
| `premises_FOL` | `List[str]` | FOL translation of each premise, in standard notation. |
| `conclusion` | `str` | Natural language conclusion sentence. |
| `conclusion_FOL` | `str` | FOL translation of the conclusion. |
| `label` | `str` | One of `"True"`, `"False"`, `"Unknown"`. |

### FOL Notation Conventions

FOLIO uses standard first-order logic notation:

| Symbol | Meaning | Example |
|---|---|---|
| `forall` / `∀` | Universal quantifier | `forall x. (P(x) -> Q(x))` |
| `exists` / `∃` | Existential quantifier | `exists x. (P(x) & Q(x))` |
| `&` / `∧` | Conjunction (AND) | `P(x) & Q(x)` |
| `\|` / `∨` | Disjunction (OR) | `P(x) \| Q(x)` |
| `->` / `→` | Implication | `P(x) -> Q(x)` |
| `<->` / `↔` | Biconditional | `P(x) <-> Q(x)` |
| `~` / `¬` | Negation | `~P(x)` |
| `(`, `)` | Grouping | `(P(x) & Q(x)) -> R(x)` |

Constants begin with a lowercase letter or are capitalized proper nouns.
Variables are single lowercase letters typically introduced by quantifiers.
Predicates are capitalized or CamelCase identifiers.

### Adapter Responsibilities

The `FOLIOAdapter` performs four tasks:

1. **Load FOL annotations** as the primary source of logical structure (not NL).
2. **Translate FOL strings into internal AST**: parse quantifiers, predicates,
   connectives, and build the typed AST.
3. **Map constants and variables** into the grounding domain: constants become
   entity IDs, variables are tracked for quantifier scope.
4. **Generate the query** from the conclusion FOL and assign the label.

### FOL Parsing Pipeline

#### Step 1: Tokenize the FOL String

Split the FOL string into tokens: quantifiers, identifiers, connectives,
parentheses.

```python
FOL_TOKEN_PATTERN = re.compile(
    r"(forall|exists|->|<->|[&|~()]|\w+)"
)

def tokenize_fol(fol_string: str) -> List[str]:
    """Split a FOL string into tokens."""
    tokens = FOL_TOKEN_PATTERN.findall(fol_string.strip())
    return tokens
```

#### Step 2: Parse into AST Nodes

Recursive descent parser converts the token stream into a tree of AST nodes.
Operator precedence (lowest to highest):

1. `<->` (biconditional)
2. `->` (implication)
3. `|` (disjunction)
4. `&` (conjunction)
5. `~` (negation)
6. Atomic predicates and quantifiers

```python
def parse_fol(tokens: List[str]) -> ASTNode:
    """Parse a FOL token list into an AST.

    Uses recursive descent with standard precedence.
    """
    pos = [0]  # mutable position tracker

    def parse_biconditional():
        left = parse_implication()
        while pos[0] < len(tokens) and tokens[pos[0]] == "<->":
            pos[0] += 1
            right = parse_implication()
            left = Iff(left, right)
        return left

    def parse_implication():
        left = parse_disjunction()
        while pos[0] < len(tokens) and tokens[pos[0]] == "->":
            pos[0] += 1
            right = parse_disjunction()
            left = Implies(left, right)
        return left

    def parse_disjunction():
        left = parse_conjunction()
        while pos[0] < len(tokens) and tokens[pos[0]] == "|":
            pos[0] += 1
            right = parse_conjunction()
            left = Or(left, right)
        return left

    def parse_conjunction():
        left = parse_negation()
        while pos[0] < len(tokens) and tokens[pos[0]] == "&":
            pos[0] += 1
            right = parse_negation()
            left = And(left, right)
        return left

    def parse_negation():
        if pos[0] < len(tokens) and tokens[pos[0]] == "~":
            pos[0] += 1
            operand = parse_negation()
            return Not(operand)
        return parse_atom()

    def parse_atom():
        token = tokens[pos[0]]
        if token in ("forall", "exists"):
            return parse_quantifier(token)
        if token == "(":
            pos[0] += 1
            node = parse_biconditional()
            assert tokens[pos[0]] == ")", f"Expected ')' at position {pos[0]}"
            pos[0] += 1
            return node
        return parse_predicate()

    # ... (parse_quantifier and parse_predicate implementations)

    result = parse_biconditional()
    return result
```

#### Step 3: Predicate Extraction with Arity Detection

Walk the AST to collect all predicate symbols and their arities.

| FOL Fragment | Predicate | Arity |
|---|---|---|
| `Student(x)` | `Student` | 1 (unary) |
| `TakesClass(x, y)` | `TakesClass` | 2 (binary) |
| `Between(x, y, z)` | `Between` | 3 (ternary) |

Arity conflicts (same predicate used with different argument counts) raise a
`PredicateArityError` with the example ID and conflicting occurrences.

#### Step 4: Constant/Variable Discrimination

Constants and variables are distinguished by context:

| Criterion | Classification | Example |
|---|---|---|
| Introduced by `forall` or `exists` | Variable | `x` in `forall x. P(x)` |
| Not bound by any quantifier | Constant | `john` in `Likes(john, x)` |
| Capitalized proper noun | Constant | `Mary` in `Student(Mary)` |

The parser maintains a scope stack to track which variables are bound by
enclosing quantifiers. Any identifier appearing as a predicate argument that is
not in the current scope is classified as a constant and added to the entity
list.

#### Step 5: Scope Resolution for Quantified Variables

Nested quantifiers create nested scopes. The scope resolution pass ensures that
each variable reference binds to the innermost enclosing quantifier of the same
name.

```
forall x. (P(x) -> exists x. Q(x))
              |                |
        outer x           inner x (shadows outer)
```

Shadowing is permitted but logged as a warning in `metadata["scope_warnings"]`.

### Output Schema

Each parsed FOLIO example produces:

| Field | Source | Notes |
|---|---|---|
| `entities` | Constants extracted from FOL annotations | Deduplicated, sorted |
| `predicates` | Predicate symbols with arities from FOL | Auto-detected |
| `rules` | Compiled ASTs from premise FOL strings | One `RuleAST` per premise |
| `query` | Compiled AST from conclusion FOL | Single `QueryAST` |
| `label` | Mapped from label string | `"True"` -> 1, `"False"` -> 0, `"Unknown"` -> 2 |
| `proof_chain` | None (FOLIO does not provide proofs) | Always None |
| `metadata` | Source info + NL text | `{"source": "folio", "example_id": ..., "premises_nl": [...], "conclusion_nl": ..., "scope_warnings": [...]}` |

### Optional: Theory Satisfiability Objective

Beyond the standard query truth prediction task, the adapter can produce an
auxiliary satisfiability objective: treat the premises as constraints that must
be simultaneously satisfiable, and query whether the conclusion is consistent
with them. Enable this mode by setting `satisfiability_mode=True` in the adapter
constructor. When enabled, the adapter wraps all premise rules into a single
conjunction and provides it as an additional constraint for the
`constraint_loss` computation.

---

## 5. Preprocessing Pipeline

### Tokenization of Entity Names

Entity names are tokenized for embedding lookup. Two strategies are available:

| Strategy | Tokenization | Use Case |
|---|---|---|
| **Word-level** | Split on whitespace and underscores, lowercase | Fixed entity vocabularies (ProofWriter) |
| **Subword** | Use the system text encoder's tokenizer | Open-vocabulary entities (FOLIO) |

For word-level tokenization, build a vocabulary from the training split and
assign integer IDs:

```python
class EntityVocab:
    """Manages entity name -> integer ID mapping."""

    def __init__(self):
        self.name_to_id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_name: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self._next_id = 2

    def add(self, name: str) -> int:
        if name not in self.name_to_id:
            self.name_to_id[name] = self._next_id
            self.id_to_name[self._next_id] = name
            self._next_id += 1
        return self.name_to_id[name]

    def encode(self, name: str) -> int:
        return self.name_to_id.get(name, self.name_to_id["<UNK>"])

    def __len__(self) -> int:
        return self._next_id
```

### Predicate Vocabulary Management

Maintain a shared predicate vocabulary across the entire dataset. The vocabulary
is built during the first `load()` call on the training split, then frozen for
validation and test splits.

```python
class PredicateVocab:
    """Manages predicate name -> (ID, arity) mapping."""

    def __init__(self):
        self.name_to_entry: Dict[str, Tuple[int, int]] = {}
        self._next_id = 0
        self._frozen = False

    def register(self, name: str, arity: int) -> int:
        if name in self.name_to_entry:
            existing_id, existing_arity = self.name_to_entry[name]
            if existing_arity != arity:
                raise PredicateArityError(
                    f"Predicate '{name}' registered with arity "
                    f"{existing_arity}, now seen with arity {arity}"
                )
            return existing_id
        if self._frozen:
            return -1  # Unknown predicate at eval time
        pred_id = self._next_id
        self.name_to_entry[name] = (pred_id, arity)
        self._next_id += 1
        return pred_id

    def freeze(self) -> None:
        self._frozen = True
```

### Rule Deduplication

Across examples in the same dataset, identical rules appear frequently
(ProofWriter reuses rule templates). Deduplicate by hashing the rule AST
structure:

1. Serialize the rule AST to a canonical string form (sorted argument order for
   commutative operators).
2. Hash the canonical string.
3. Store a mapping from hash to `RuleAST` object.
4. In each `SymbolicExample`, store rule references (hashes or indices into a
   shared rule table).

This reduces memory for large datasets and enables shared rule embeddings in the
`RuleNetwork`.

### Data Augmentation

Apply augmentations at load time (configurable, disabled by default):

| Augmentation | Description | Effect |
|---|---|---|
| **Negated queries** | Negate the query and flip the label (`True` <-> `False`, `Unknown` unchanged). | Doubles the effective query count. |
| **Rule permutation** | Randomly shuffle the order of rules in the theory. | Prevents the model from relying on rule ordering. |
| **Entity renaming** | Replace entity names with random tokens. | Forces reliance on structural reasoning, not name memorization. |
| **Subset theories** | Drop a random subset of non-essential rules. | Trains robustness to incomplete theories. |

### Train/Val/Test Split Handling

- Build vocabularies (entity, predicate) from the **training split only**.
- Freeze vocabularies before processing validation and test splits.
- Unknown entities at eval time receive the `<UNK>` ID.
- Unknown predicates at eval time receive ID `-1` and are flagged in metadata.

### Caching

Preprocessed data is cached to disk after the first successful `load()` call:

```
{cache_dir}/
  {dataset_name}/
    {split}/
      examples.json       # List[SymbolicExample] serialized as JSON
      entity_vocab.json    # EntityVocab state
      predicate_vocab.json # PredicateVocab state
      metadata.json        # Parse statistics, version, timestamp
```

On subsequent `load()` calls, check the cache:

1. Compare the dataset file modification time against `metadata.json["timestamp"]`.
2. If the cache is fresh, load from `examples.json` directly.
3. If the cache is stale or missing, re-parse and overwrite.

The cache is invalidated automatically when the adapter code version changes
(tracked in `metadata.json["adapter_version"]`).

---

## 6. Entity Embedding Strategies

The adapter produces entity IDs; the embedding strategy determines how those IDs
become dense vectors for the reasoner.

### Lookup Embedding

Assign each entity a learned embedding vector from a fixed table.

| Property | Value |
|---|---|
| **Input** | Entity ID (integer) |
| **Output** | `(D_ent,)` embedding vector |
| **Trainable parameters** | `num_entities * D_ent` |
| **Suitable for** | Closed-vocabulary datasets (ProofWriter) |

```python
class EntityLookupEmbedding(nn.Module):
    def __init__(self, vocab_size: int, entity_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, entity_dim, padding_idx=0)

    def forward(self, entity_ids: torch.LongTensor) -> torch.Tensor:
        return self.embedding(entity_ids)  # (B, N, D_ent)
```

### Encoder Embedding

Pass entity name tokens through a text encoder to produce embeddings. Suitable
for open-vocabulary datasets where entities are arbitrary noun phrases.

| Property | Value |
|---|---|
| **Input** | Entity name string (tokenized) |
| **Output** | `(D_ent,)` embedding vector |
| **Trainable parameters** | Text encoder parameters + projection layer |
| **Suitable for** | Open-vocabulary datasets (FOLIO) |

```python
class EntityEncoderEmbedding(nn.Module):
    def __init__(self, text_encoder: nn.Module, encoder_dim: int, entity_dim: int):
        super().__init__()
        self.text_encoder = text_encoder
        self.projection = nn.Linear(encoder_dim, entity_dim)

    def forward(self, entity_token_ids: torch.LongTensor,
                entity_mask: torch.BoolTensor) -> torch.Tensor:
        # entity_token_ids: (B, N, L) tokenized entity names
        # Reshape, encode, pool, project
        B, N, L = entity_token_ids.shape
        flat_ids = entity_token_ids.view(B * N, L)
        flat_mask = entity_mask.view(B * N, L)
        encoded = self.text_encoder(flat_ids, attention_mask=flat_mask)
        pooled = encoded[:, 0, :]  # CLS token
        projected = self.projection(pooled)  # (B*N, D_ent)
        return projected.view(B, N, -1)  # (B, N, D_ent)
```

### Workspace Embedding

At inference time (runtime mode), entities come from the upstream workspace
module's `EntityExtractor` rather than from the dataset adapter. The adapter
provides entity IDs for training-time lookup; the workspace provides entity
embeddings directly at inference time.

| Mode | Entity Source | Embedding Source |
|---|---|---|
| **Training** | Adapter `entity_ids` | `EntityLookupEmbedding` or `EntityEncoderEmbedding` |
| **Inference** | Workspace `EntityExtractor` | Workspace slot embeddings (already `(B, N, D_ent)`) |

The reasoner's `forward()` method accepts either `entity_ids` (training) or
`entity_embeddings` (inference), dispatching to the appropriate path:

```python
def forward(self, entities, ...):
    if isinstance(entities, torch.LongTensor):
        entity_emb = self.entity_embedding(entities)  # Lookup
    else:
        entity_emb = entities  # Already embedded (workspace mode)
```

---

## 7. Integration with Training Loop

### Forward Pass

Adapter outputs feed directly into `SymbolicReasoner.forward()`:

```python
adapter = ADAPTER_REGISTRY["proofwriter"](data_dir="data/proofwriter")
examples = adapter.load(split="train")
batch = adapter.collate(examples[:batch_size])

output = reasoner(
    entities=batch.entity_ids,       # (B, max_N)
    entity_mask=batch.entity_mask,   # (B, max_N)
    rules=batch.rules,               # List[List[RuleAST]]
    queries=batch.queries,           # List[QueryAST]
)
```

### Loss Computation

Combine query truth prediction loss with constraint satisfaction loss:

```python
# Query truth prediction: BCE against ground-truth labels
query_loss = F.binary_cross_entropy(
    output.truth.squeeze(-1),   # (B,) predicted truth in [0,1]
    batch.labels,               # (B,) ground-truth soft labels
)

# Constraint loss: penalize rule violations
constraint_loss = output.constraint_loss  # scalar

# Combined loss
total_loss = query_loss + config.constraint_weight * constraint_loss
total_loss.backward()
```

### Constraint Weight Warmup

Avoid dominating the task loss with constraint loss early in training:

```python
def get_constraint_weight(step: int, config: RuleConfig) -> float:
    """Linearly warm up the constraint weight over warmup_steps."""
    if step >= config.warmup_steps:
        return config.constraint_weight
    return config.constraint_weight * (step / config.warmup_steps)
```

### Metrics

Track the following metrics during training and evaluation:

| Metric | Computation | Meaning |
|---|---|---|
| **Query accuracy** | `(predicted_label == true_label).mean()` | Percentage of correctly classified queries |
| **Mean violation rate** | `output.violation_stats["mean_violation"]` | Average rule violation across the batch |
| **Per-rule violation** | `output.violation_stats["per_rule_violation"]` | Violation rate for each rule (identifies problematic rules) |
| **Proof chain accuracy** | Compare predicted proof with `proof_chain` | Percentage of correctly identified proof steps (when supervision available) |
| **Parse coverage** | `examples_without_warnings / total_examples` | Fraction of examples fully parsed without fallback |

### DataLoader Integration

Wrap the adapter with a standard PyTorch `Dataset` and `DataLoader`:

```python
class SymbolicDataset(torch.utils.data.Dataset):
    def __init__(self, adapter: SymbolicDatasetAdapter, split: str = "train"):
        self.examples = adapter.load(split)
        self.adapter = adapter

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SymbolicExample:
        return self.examples[idx]


def symbolic_collate_fn(examples: List[SymbolicExample]) -> SymbolicBatch:
    """Collate function for DataLoader. Delegates to the adapter."""
    return adapter.collate(examples)


dataloader = torch.utils.data.DataLoader(
    SymbolicDataset(adapter, split="train"),
    batch_size=32,
    shuffle=True,
    collate_fn=symbolic_collate_fn,
    num_workers=4,
    pin_memory=True,
)
```

---

## 8. Code Patterns

### ProofWriterAdapter Class

```python
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple


class ProofWriterAdapter(SymbolicDatasetAdapter):
    """Adapter for the ProofWriter multi-hop reasoning dataset."""

    LABEL_MAP = {"True": 1, "False": 0, "Unknown": 2}
    LABEL_FLOAT_MAP = {"True": 1.0, "False": 0.0, "Unknown": 0.5}

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        max_entities: int = 32,
        max_rules: int = 64,
        use_structured: bool = True,
        augment: bool = False,
    ):
        super().__init__(data_dir, cache_dir, max_entities, max_rules)
        self.use_structured = use_structured
        self.augment = augment
        self.entity_vocab = EntityVocab()
        self.predicate_vocab = PredicateVocab()

    def load(self, split: str = "train") -> List[SymbolicExample]:
        cached = self._try_load_cache(split)
        if cached is not None:
            return cached

        raw_data = self._load_raw(split)
        examples = []
        parse_failures = 0

        for raw in raw_data:
            try:
                example = self._parse_example(raw)
                if example.metadata.get("skip"):
                    parse_failures += 1
                    continue
                examples.append(example)
            except Exception as e:
                parse_failures += 1
                # Log and continue; do not abort the entire split
                _log_parse_error(raw, e)

        if split == "train":
            self.predicate_vocab.freeze()

        self._save_cache(split, examples)
        return examples

    def _parse_example(self, raw: Dict[str, Any]) -> SymbolicExample:
        context = raw["context"]
        question = raw["question"]
        answer = raw["answer"]

        # Split context into sentences
        sentences = [s.strip() for s in context.split(".") if s.strip()]

        # Parse facts and rules
        rules: List[RuleAST] = []
        entities_set: set = set()
        predicates_found: Dict[str, int] = {}
        warnings: List[str] = []

        for sentence in sentences:
            sentence = sentence.strip() + "."
            ast = self._parse_sentence(sentence)
            if ast is None:
                warnings.append(f"Unparseable: {sentence}")
                continue
            rules.append(ast)
            self._collect_symbols(ast, entities_set, predicates_found)

        # Parse query
        query = self._parse_query(question)
        if query is None:
            return SymbolicExample(
                entities=[], predicates={}, rules=[], query=Atom("UNPARSED", []),
                label=2, label_float=0.5, proof_chain=None,
                metadata={"skip": True, "source": "proofwriter",
                          "example_id": raw.get("id", "unknown")},
            )
        self._collect_symbols(query, entities_set, predicates_found)

        # Register symbols in vocabularies
        entities = sorted(entities_set)[:self.max_entities]
        for ent in entities:
            self.entity_vocab.add(ent)
        for pred_name, arity in predicates_found.items():
            self.predicate_vocab.register(pred_name, arity)

        # Extract proof chain if available
        proof_chain = self._extract_proof_chain(raw, rules)

        return SymbolicExample(
            entities=entities,
            predicates=predicates_found,
            rules=rules[:self.max_rules],
            query=query,
            label=self.LABEL_MAP[answer],
            label_float=self.LABEL_FLOAT_MAP[answer],
            proof_chain=proof_chain,
            metadata={
                "source": "proofwriter",
                "example_id": raw.get("id", "unknown"),
                "depth": raw.get("depth", -1),
                "parse_warnings": warnings,
            },
        )

    def _parse_sentence(self, sentence: str) -> Optional[RuleAST]:
        """Try each pattern; return first match or None."""
        for pattern, builder in FACT_PATTERNS + RULE_PATTERNS:
            match = re.match(pattern, sentence)
            if match:
                return builder(match)
        return None

    def _parse_query(self, question: str) -> Optional[QueryAST]:
        """Parse the question string into a query AST."""
        for pattern, builder in FACT_PATTERNS:
            match = re.match(pattern, question.strip())
            if match:
                return builder(match)
        return None

    def _collect_symbols(
        self, ast: ASTNode, entities: set, predicates: Dict[str, int]
    ) -> None:
        """Walk the AST and collect entity names and predicate symbols."""
        if isinstance(ast, Atom):
            predicates[ast.name] = len(ast.args)
            for arg in ast.args:
                if not _is_variable(arg):
                    entities.add(arg)
        elif isinstance(ast, (And, Or, Implies, Iff)):
            self._collect_symbols(ast.left, entities, predicates)
            self._collect_symbols(ast.right, entities, predicates)
        elif isinstance(ast, Not):
            self._collect_symbols(ast.operand, entities, predicates)
        elif isinstance(ast, (ForAll, Exists)):
            self._collect_symbols(ast.body, entities, predicates)

    def _extract_proof_chain(
        self, raw: Dict, rules: List[RuleAST]
    ) -> Optional[List[int]]:
        """Map proof identifiers to rule indices."""
        proof = raw.get("proof")
        if proof is None:
            return None
        chain = []
        for step_id in proof:
            for idx, rule in enumerate(rules):
                if rule.metadata.get("source_id") == step_id:
                    chain.append(idx)
                    break
        return chain if chain else None

    def collate(self, examples: List[SymbolicExample]) -> SymbolicBatch:
        return _collate_examples(examples, self.entity_vocab)
```

### FOLIOAdapter Class

```python
class FOLIOAdapter(SymbolicDatasetAdapter):
    """Adapter for the FOLIO first-order logic dataset."""

    LABEL_MAP = {"True": 1, "False": 0, "Unknown": 2}
    LABEL_FLOAT_MAP = {"True": 1.0, "False": 0.0, "Unknown": 0.5}

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        max_entities: int = 32,
        max_rules: int = 64,
        satisfiability_mode: bool = False,
    ):
        super().__init__(data_dir, cache_dir, max_entities, max_rules)
        self.satisfiability_mode = satisfiability_mode
        self.entity_vocab = EntityVocab()
        self.predicate_vocab = PredicateVocab()

    def load(self, split: str = "train") -> List[SymbolicExample]:
        cached = self._try_load_cache(split)
        if cached is not None:
            return cached

        raw_data = self._load_raw(split)
        examples = []

        for raw in raw_data:
            try:
                example = self._parse_example(raw)
                examples.append(example)
            except FOLParseError as e:
                _log_parse_error(raw, e)

        if split == "train":
            self.predicate_vocab.freeze()

        self._save_cache(split, examples)
        return examples

    def _parse_example(self, raw: Dict[str, Any]) -> SymbolicExample:
        premises_fol = raw["premises_FOL"]
        conclusion_fol = raw["conclusion_FOL"]
        label = raw["label"]

        entities_set: set = set()
        predicates_found: Dict[str, int] = {}
        scope_warnings: List[str] = []

        # Parse premise FOL strings into rule ASTs
        rules: List[RuleAST] = []
        for i, fol_str in enumerate(premises_fol):
            tokens = tokenize_fol(fol_str)
            ast = parse_fol(tokens)
            warnings = _check_variable_scoping(ast)
            scope_warnings.extend(warnings)
            rules.append(ast)
            self._collect_symbols_fol(ast, entities_set, predicates_found)

        # Parse conclusion FOL into query AST
        query_tokens = tokenize_fol(conclusion_fol)
        query = parse_fol(query_tokens)
        self._collect_symbols_fol(query, entities_set, predicates_found)

        # Register symbols
        entities = sorted(entities_set)[:self.max_entities]
        for ent in entities:
            self.entity_vocab.add(ent)
        for pred_name, arity in predicates_found.items():
            self.predicate_vocab.register(pred_name, arity)

        return SymbolicExample(
            entities=entities,
            predicates=predicates_found,
            rules=rules[:self.max_rules],
            query=query,
            label=self.LABEL_MAP[label],
            label_float=self.LABEL_FLOAT_MAP[label],
            proof_chain=None,  # FOLIO does not provide proof chains
            metadata={
                "source": "folio",
                "example_id": raw.get("id", "unknown"),
                "premises_nl": raw.get("premises", []),
                "conclusion_nl": raw.get("conclusion", ""),
                "scope_warnings": scope_warnings,
            },
        )

    def _collect_symbols_fol(
        self, ast: ASTNode, entities: set, predicates: Dict[str, int]
    ) -> None:
        """Walk a FOL AST and collect constants and predicate symbols."""
        if isinstance(ast, Atom):
            predicates[ast.name] = len(ast.args)
            for arg in ast.args:
                if not _is_bound_variable(arg, ast):
                    entities.add(arg)
        elif isinstance(ast, (And, Or, Implies, Iff)):
            self._collect_symbols_fol(ast.left, entities, predicates)
            self._collect_symbols_fol(ast.right, entities, predicates)
        elif isinstance(ast, Not):
            self._collect_symbols_fol(ast.operand, entities, predicates)
        elif isinstance(ast, (ForAll, Exists)):
            self._collect_symbols_fol(ast.body, entities, predicates)

    def collate(self, examples: List[SymbolicExample]) -> SymbolicBatch:
        return _collate_examples(examples, self.entity_vocab)
```

### Collate Function

```python
def _collate_examples(
    examples: List[SymbolicExample],
    entity_vocab: EntityVocab,
) -> SymbolicBatch:
    """Collate variable-length SymbolicExamples into a padded batch."""
    B = len(examples)

    # Determine max entity count in this batch
    max_N = max(len(ex.entities) for ex in examples)

    # Build padded entity ID tensor
    entity_ids = torch.zeros(B, max_N, dtype=torch.long)
    entity_mask = torch.zeros(B, max_N, dtype=torch.bool)

    for i, ex in enumerate(examples):
        n = len(ex.entities)
        for j, name in enumerate(ex.entities):
            entity_ids[i, j] = entity_vocab.encode(name)
        entity_mask[i, :n] = True

    # Collect rules (variable-length, not tensorized)
    rules = [ex.rules for ex in examples]
    queries = [ex.query for ex in examples]

    # Labels
    labels = torch.tensor([ex.label_float for ex in examples], dtype=torch.float32)
    label_ids = torch.tensor([ex.label for ex in examples], dtype=torch.long)

    # Proof chains (None entries preserved)
    proof_chains = [ex.proof_chain for ex in examples]

    # Metadata
    metadata = [ex.metadata for ex in examples]

    return SymbolicBatch(
        entity_ids=entity_ids,
        entity_mask=entity_mask,
        rules=rules,
        queries=queries,
        labels=labels,
        label_ids=label_ids,
        proof_chains=proof_chains,
        metadata=metadata,
    )
```

---

## Appendix A: ProofWriter Example Walkthrough

### Raw Input

```json
{
  "id": "PW-D2-0042",
  "context": "The cat is blue. The cat is big. The dog is red. If something is blue then it is cold. If something is cold and big then it chases the dog.",
  "question": "The cat chases the dog.",
  "answer": "True",
  "depth": 2
}
```

### Step 1: Sentence Splitting

```
["The cat is blue", "The cat is big", "The dog is red",
 "If something is blue then it is cold",
 "If something is cold and big then it chases the dog"]
```

### Step 2: Fact Parsing

| Sentence | Parsed AST |
|---|---|
| `"The cat is blue."` | `Atom("is_blue", ["cat"])` |
| `"The cat is big."` | `Atom("is_big", ["cat"])` |
| `"The dog is red."` | `Atom("is_red", ["dog"])` |

### Step 3: Rule Parsing

| Sentence | Parsed AST |
|---|---|
| `"If something is blue then it is cold."` | `ForAll("x", Implies(Atom("is_blue", ["x"]), Atom("is_cold", ["x"])))` |
| `"If something is cold and big then it chases the dog."` | `ForAll("x", Implies(And(Atom("is_cold", ["x"]), Atom("is_big", ["x"])), Atom("chases", ["x", "dog"])))` |

### Step 4: Query Parsing

| Question | Parsed AST |
|---|---|
| `"The cat chases the dog."` | `Atom("chases", ["cat", "dog"])` |

### Step 5: Symbol Collection

**Entities**: `["cat", "dog"]`

**Predicates**: `{"is_blue": 1, "is_big": 1, "is_red": 1, "is_cold": 1, "chases": 2}`

### Step 6: SymbolicExample

```python
SymbolicExample(
    entities=["cat", "dog"],
    predicates={"is_blue": 1, "is_big": 1, "is_red": 1, "is_cold": 1, "chases": 2},
    rules=[
        Atom("is_blue", ["cat"]),
        Atom("is_big", ["cat"]),
        Atom("is_red", ["dog"]),
        ForAll("x", Implies(Atom("is_blue", ["x"]), Atom("is_cold", ["x"]))),
        ForAll("x", Implies(
            And(Atom("is_cold", ["x"]), Atom("is_big", ["x"])),
            Atom("chases", ["x", "dog"]),
        )),
    ],
    query=Atom("chases", ["cat", "dog"]),
    label=1,
    label_float=1.0,
    proof_chain=[0, 3, 4],  # fact0 -> rule3 -> rule4
    metadata={
        "source": "proofwriter",
        "example_id": "PW-D2-0042",
        "depth": 2,
        "parse_warnings": [],
    },
)
```

### Step 7: Batched Tensors (Batch of 1)

```
entity_ids:   [[2, 3]]          # cat=2, dog=3 (0=PAD, 1=UNK)
entity_mask:  [[True, True]]
rules:        [[Atom(...), Atom(...), Atom(...), ForAll(...), ForAll(...)]]
queries:      [Atom("chases", ["cat", "dog"])]
labels:       [1.0]
label_ids:    [1]
```

---

## Appendix B: FOLIO Example Walkthrough

### Raw Input

```json
{
  "id": "FOLIO-0117",
  "premises": [
    "All students who study hard pass the exam.",
    "John is a student.",
    "John studies hard."
  ],
  "premises_FOL": [
    "forall x. ((Student(x) & StudiesHard(x)) -> PassExam(x))",
    "Student(john)",
    "StudiesHard(john)"
  ],
  "conclusion": "John passes the exam.",
  "conclusion_FOL": "PassExam(john)",
  "label": "True"
}
```

### Step 1: Tokenize Premise FOL Strings

Premise 1: `"forall x. ((Student(x) & StudiesHard(x)) -> PassExam(x))"`

```
Tokens: ["forall", "x", "(", "(", "Student", "(", "x", ")", "&",
         "StudiesHard", "(", "x", ")", ")", "->", "PassExam", "(", "x", ")", ")"]
```

Premise 2: `"Student(john)"`

```
Tokens: ["Student", "(", "john", ")"]
```

Premise 3: `"StudiesHard(john)"`

```
Tokens: ["StudiesHard", "(", "john", ")"]
```

### Step 2: Parse into AST

| FOL String | Parsed AST |
|---|---|
| `forall x. ((Student(x) & StudiesHard(x)) -> PassExam(x))` | `ForAll("x", Implies(And(Atom("Student", ["x"]), Atom("StudiesHard", ["x"])), Atom("PassExam", ["x"])))` |
| `Student(john)` | `Atom("Student", ["john"])` |
| `StudiesHard(john)` | `Atom("StudiesHard", ["john"])` |

Conclusion: `"PassExam(john)"`

```
Atom("PassExam", ["john"])
```

### Step 3: Symbol Extraction

**Constants (entities)**: `["john"]` -- `x` is a bound variable, not a constant.

**Predicates**:

| Name | Arity |
|---|---|
| `Student` | 1 |
| `StudiesHard` | 1 |
| `PassExam` | 1 |

### Step 4: SymbolicExample

```python
SymbolicExample(
    entities=["john"],
    predicates={"Student": 1, "StudiesHard": 1, "PassExam": 1},
    rules=[
        ForAll("x", Implies(
            And(Atom("Student", ["x"]), Atom("StudiesHard", ["x"])),
            Atom("PassExam", ["x"]),
        )),
        Atom("Student", ["john"]),
        Atom("StudiesHard", ["john"]),
    ],
    query=Atom("PassExam", ["john"]),
    label=1,
    label_float=1.0,
    proof_chain=None,  # FOLIO does not provide proofs
    metadata={
        "source": "folio",
        "example_id": "FOLIO-0117",
        "premises_nl": [
            "All students who study hard pass the exam.",
            "John is a student.",
            "John studies hard.",
        ],
        "conclusion_nl": "John passes the exam.",
        "scope_warnings": [],
    },
)
```

### Step 5: Batched Tensors (Batch of 1)

```
entity_ids:   [[2]]             # john=2 (0=PAD, 1=UNK)
entity_mask:  [[True]]
rules:        [[ForAll(...), Atom(...), Atom(...)]]
queries:      [Atom("PassExam", ["john"])]
labels:       [1.0]
label_ids:    [1]
```

---

## Appendix C: Supported FOL Operators and AST Mappings

### AST Node Types

| AST Node | Constructor | FOL Syntax | Arity | Description |
|---|---|---|---|---|
| `Atom` | `Atom(name, args)` | `P(x)`, `R(x, y)` | n-ary | Atomic predicate application |
| `Not` | `Not(operand)` | `~P(x)`, `¬P(x)` | Unary | Logical negation |
| `And` | `And(left, right)` | `P & Q`, `P ∧ Q` | Binary | Conjunction |
| `Or` | `Or(left, right)` | `P \| Q`, `P ∨ Q` | Binary | Disjunction |
| `Implies` | `Implies(left, right)` | `P -> Q`, `P → Q` | Binary | Material implication |
| `Iff` | `Iff(left, right)` | `P <-> Q`, `P ↔ Q` | Binary | Biconditional |
| `ForAll` | `ForAll(var, body)` | `forall x. P(x)`, `∀x. P(x)` | Quantifier | Universal quantification |
| `Exists` | `Exists(var, body)` | `exists x. P(x)`, `∃x. P(x)` | Quantifier | Existential quantification |

### AST Node Base Class

```python
from dataclasses import dataclass
from typing import List


@dataclass
class ASTNode:
    """Base class for all AST nodes."""

    def children(self) -> List["ASTNode"]:
        """Return child nodes for tree traversal."""
        raise NotImplementedError

    def to_str(self) -> str:
        """Canonical string representation for hashing/dedup."""
        raise NotImplementedError


@dataclass
class Atom(ASTNode):
    name: str         # Predicate name
    args: List[str]   # Arguments (entity names or variable names)

    def children(self) -> List[ASTNode]:
        return []

    def to_str(self) -> str:
        return f"{self.name}({', '.join(self.args)})"


@dataclass
class Not(ASTNode):
    operand: ASTNode

    def children(self) -> List[ASTNode]:
        return [self.operand]

    def to_str(self) -> str:
        return f"~({self.operand.to_str()})"


@dataclass
class And(ASTNode):
    left: ASTNode
    right: ASTNode

    def children(self) -> List[ASTNode]:
        return [self.left, self.right]

    def to_str(self) -> str:
        args = sorted([self.left.to_str(), self.right.to_str()])
        return f"({args[0]} & {args[1]})"


@dataclass
class Or(ASTNode):
    left: ASTNode
    right: ASTNode

    def children(self) -> List[ASTNode]:
        return [self.left, self.right]

    def to_str(self) -> str:
        args = sorted([self.left.to_str(), self.right.to_str()])
        return f"({args[0]} | {args[1]})"


@dataclass
class Implies(ASTNode):
    left: ASTNode   # antecedent
    right: ASTNode  # consequent

    def children(self) -> List[ASTNode]:
        return [self.left, self.right]

    def to_str(self) -> str:
        return f"({self.left.to_str()} -> {self.right.to_str()})"


@dataclass
class Iff(ASTNode):
    left: ASTNode
    right: ASTNode

    def children(self) -> List[ASTNode]:
        return [self.left, self.right]

    def to_str(self) -> str:
        args = sorted([self.left.to_str(), self.right.to_str()])
        return f"({args[0]} <-> {args[1]})"


@dataclass
class ForAll(ASTNode):
    var: str
    body: ASTNode

    def children(self) -> List[ASTNode]:
        return [self.body]

    def to_str(self) -> str:
        return f"forall {self.var}. ({self.body.to_str()})"


@dataclass
class Exists(ASTNode):
    var: str
    body: ASTNode

    def children(self) -> List[ASTNode]:
        return [self.body]

    def to_str(self) -> str:
        return f"exists {self.var}. ({self.body.to_str()})"
```

### Operator-to-Fuzzy Mapping

When the AST is evaluated by the `SymbolicReasoner`, each logical connective
maps to a fuzzy operator from the configured bundle:

| AST Node | Fuzzy Operator | Bundle Method |
|---|---|---|
| `And` | T-norm | `bundle.and_(x, y)` |
| `Or` | T-conorm | `bundle.or_(x, y)` |
| `Not` | Standard negation | `bundle.not_(x)` |
| `Implies` | S-implication or residuated | `bundle.implies(x, y)` |
| `Iff` | Biconditional | `bundle.iff(x, y)` |
| `ForAll` | Generalized mean (low p) | `bundle.forall(values, p)` |
| `Exists` | Generalized mean (high p) | `bundle.exists(values, p)` |
| `Atom` | Predicate grounding | `predicate_module(entity_embeddings)` |

The AST evaluator walks the tree bottom-up, evaluating `Atom` nodes via
predicate/relation modules, then combining truth values with the appropriate
fuzzy operator at each internal node. Quantifiers aggregate truth values across
all entity groundings for the bound variable.

### Type Aliases

For clarity in the adapter codebase, the following type aliases are used:

```python
# Rule ASTs are any AST node representing a theory premise
RuleAST = ASTNode

# Query ASTs are any AST node representing the formula to evaluate
QueryAST = ASTNode
```

Both `RuleAST` and `QueryAST` are structurally identical AST nodes. The type
aliases exist purely for documentation and readability in function signatures.
