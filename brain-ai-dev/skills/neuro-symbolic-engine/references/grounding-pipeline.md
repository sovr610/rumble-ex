# Grounding Pipeline Reference

This document specifies the grounding pipeline that maps symbolic entities, predicates, and relations to neural modules. The pipeline follows the Logic Tensor Networks (LTN) / Real Logic framework where every symbol is grounded as a neural computational graph operating on real-valued tensors. All truth values lie in [0,1]. All modules must be differentiable, checkpointable, and auditable.

The grounding pipeline sits between the Global Workspace module (which provides slot representations of multi-modal content) and the Rule Engine (which evaluates symbolic formulas over grounded entities). Its responsibility is threefold: extract entities from workspace representations, instantiate typed predicate and relation modules, and expose a unified grounding object that the rule engine can query by symbol name.

---

## 1. Entity Extraction

The EntityExtractor converts workspace slot representations into a set of entity embeddings suitable for symbolic evaluation. Two extraction modes are supported, selected via `GroundingConfig.extractor_type`.

### 1.1 slot_identity Mode

Treat each workspace slot directly as an entity. The number of entities equals the number of workspace slots (N = K).

**Input**: Workspace slots tensor of shape `(B, K, D_ws)` where B is batch size, K is the number of workspace slots (typically bounded by `WorkspaceConfig.capacity_limit`), and D_ws is the workspace dimension.

**Processing steps**:

1. Apply a learned linear projection from D_ws to D_ent: `proj = nn.Linear(D_ws, D_ent)`.
2. Apply LayerNorm over the entity dimension: `norm = nn.LayerNorm(D_ent)`.
3. Apply the entity mask to zero out padding slots.

**Output**: Entity tensor `(B, K, D_ent)` and boolean mask `(B, K)`.

```python
class SlotIdentityExtractor(nn.Module):
    def __init__(self, workspace_dim: int, entity_dim: int):
        super().__init__()
        self.proj = nn.Linear(workspace_dim, entity_dim)
        self.norm = nn.LayerNorm(entity_dim)

    def forward(
        self,
        slots: torch.Tensor,           # (B, K, D_ws)
        slot_mask: torch.BoolTensor,    # (B, K) — True where valid
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        entities = self.norm(self.proj(slots))       # (B, K, D_ent)
        entities = entities * slot_mask.unsqueeze(-1) # zero out padding
        return entities, slot_mask
```

This mode is the default (`GroundingConfig.extractor_type = "slot_identity"`). It is deterministic, has minimal overhead, and preserves a one-to-one mapping between workspace slots and entities, which simplifies debugging.

### 1.2 proposal_head Mode

An entity proposal network that clusters workspace slots into a fixed number of N entity proposals via cross-attention. Use this mode when the number of logical entities should differ from the number of workspace slots, or when entity deduplication is required.

**Input**: Workspace slots `(B, K, D_ws)` and slot mask `(B, K)`.

**Processing steps**:

1. Maintain N learnable query vectors `(N, D_ent)` as `nn.Parameter`.
2. Expand queries to batch dimension: `(B, N, D_ent)`.
3. Apply multi-head cross-attention where queries attend to workspace slots as keys and values. Use the slot mask as the key padding mask to prevent attending to padding positions.
4. Project attention output through a feedforward block: `Linear(D_ent, D_ent) -> GELU -> Linear(D_ent, D_ent)`.
5. Apply LayerNorm.
6. Compute per-entity confidence scores: `Linear(D_ent, 1) -> Sigmoid` producing `(B, N)`.
7. Build entity mask from confidence scores by thresholding at 0.5 (or use a configurable threshold).
8. Optionally apply entity deduplication (see Section 7).

**Output**: Entity tensor `(B, N, D_ent)`, boolean mask `(B, N)`, confidence scores `(B, N)`.

```python
class ProposalHeadExtractor(nn.Module):
    def __init__(
        self,
        workspace_dim: int,
        entity_dim: int,
        num_proposals: int,
        num_heads: int = 8,
        dedup_threshold: float = 0.9,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_proposals, entity_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=entity_dim,
            num_heads=num_heads,
            kdim=workspace_dim,
            vdim=workspace_dim,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(entity_dim, entity_dim),
            nn.GELU(),
            nn.Linear(entity_dim, entity_dim),
        )
        self.norm = nn.LayerNorm(entity_dim)
        self.confidence_head = nn.Sequential(
            nn.Linear(entity_dim, 1),
            nn.Sigmoid(),
        )
        self.dedup_threshold = dedup_threshold

    def forward(
        self,
        slots: torch.Tensor,           # (B, K, D_ws)
        slot_mask: torch.BoolTensor,    # (B, K)
    ) -> Tuple[torch.Tensor, torch.BoolTensor, torch.Tensor]:
        B = slots.shape[0]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, N, D_ent)

        # Cross-attention: queries attend to workspace slots
        # Invert mask for nn.MultiheadAttention key_padding_mask (True = ignore)
        key_padding_mask = ~slot_mask
        attended, _ = self.cross_attn(
            queries, slots, slots,
            key_padding_mask=key_padding_mask,
        )

        entities = self.norm(self.ffn(attended) + attended)    # (B, N, D_ent)
        confidence = self.confidence_head(entities).squeeze(-1) # (B, N)
        entity_mask = confidence > 0.5

        return entities, entity_mask, confidence
```

### 1.3 Entity Mask Handling

All downstream modules (predicates, relations, rule engine) must respect the entity mask `(B, N)`. Conventions:

- `True` means the entity at that position is valid.
- `False` means padding. The corresponding entity embedding must be zeroed and excluded from quantifier aggregation.
- When computing pairwise relation matrices `(B, N, N)`, construct a pair mask as `mask_i.unsqueeze(-1) & mask_j.unsqueeze(-2)` to exclude pairs involving padding entities.
- Predicate outputs for masked entities must be clamped to 0.0 so they do not influence FORALL/EXISTS aggregators.

### 1.4 Projection: D_ws to D_ent

Both extractor modes share the same projection semantics. The projection chain is always:

```
Linear(D_ws, D_ent) -> LayerNorm(D_ent)
```

The workspace dimension D_ws is defined by `WorkspaceConfig.workspace_dim` (default 4096 at production scale). The entity dimension D_ent is defined by `SymbolicConfig.entity_dim` (default 256). This dimension reduction is intentional: entity embeddings should be compact representations focused on symbolic identity, not full multi-modal representations.

### 1.5 EntityExtractor Factory

```python
def create_entity_extractor(
    config: GroundingConfig,
    workspace_dim: int,
    entity_dim: int,
) -> nn.Module:
    if config.extractor_type == "slot_identity":
        return SlotIdentityExtractor(workspace_dim, entity_dim)
    elif config.extractor_type == "proposal_head":
        return ProposalHeadExtractor(
            workspace_dim=workspace_dim,
            entity_dim=entity_dim,
            num_proposals=config.max_entities,
        )
    else:
        raise ValueError(f"Unknown extractor_type: {config.extractor_type}")
```

---

## 2. Predicate Modules

Predicates are unary functions P(x) mapping a single entity embedding to a truth value in [0,1]. Each predicate is a named neural module registered in a `PredicateRegistry`.

### 2.1 MLP Predicate

The default predicate architecture. Used when `GroundingConfig.predicate_type = "mlp"`.

```python
class MLPPredicate(nn.Module):
    def __init__(self, entity_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, entities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entities: (B, N, D_ent)
        Returns:
            truth_values: (B, N) in [0, 1]
        """
        return self.net(entities).squeeze(-1)
```

**Batch evaluation**: Given entities `(B, N, D_ent)`, calling `P(entities)` produces truth values `(B, N)` -- one truth value per entity per batch element. Apply the entity mask after evaluation to zero out padding positions.

### 2.2 PredicateRegistry

A `PredicateRegistry` manages named predicate modules. It is implemented as an `nn.ModuleDict` so all parameters participate in `state_dict` serialization and optimizer parameter groups.

```python
class PredicateRegistry(nn.ModuleDict):
    def __init__(self, entity_dim: int, hidden_dim: int):
        super().__init__()
        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim

    def register_predicate(self, name: str, module: Optional[nn.Module] = None):
        """Register a named predicate. Creates MLP predicate if module is None."""
        if module is None:
            module = MLPPredicate(self.entity_dim, self.hidden_dim)
        self[name] = module

    def evaluate(
        self,
        name: str,
        entities: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Evaluate named predicate, masking invalid entities."""
        truth = self[name](entities)              # (B, N)
        truth = truth * mask.float()               # zero out padding
        return truth

    def evaluate_all(
        self,
        entities: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate all registered predicates. Returns dict of name -> (B, N)."""
        return {name: self.evaluate(name, entities, mask) for name in self}
```

Access pattern: `predicates["is_mammal"]` returns the `nn.Module`; `predicates.evaluate("is_mammal", entities, mask)` returns masked truth values.

### 2.3 Typed Predicates (Optional)

When entities have heterogeneous types (e.g., objects vs. events vs. agents), maintain separate predicate families per type. Implement this by adding a `type_id` tensor `(B, N)` and routing each entity to the predicate family matching its type:

```python
# In forward pass:
for type_idx, pred_family in enumerate(self.typed_predicates):
    type_mask = (type_ids == type_idx) & entity_mask
    truth[type_mask] = pred_family(entities[type_mask])
```

Typed predicates are optional and disabled by default. Enable them when the entity space is semantically partitioned and different predicate architectures are needed per type.

---

## 3. Relation Modules

Relations are binary functions R(x, y) mapping a pair of entity embeddings to a truth value in [0,1]. Three architectures are supported, selected via `GroundingConfig.relation_type`.

### 3.1 MLP Relation

Concatenate entity pair embeddings and pass through an MLP.

```python
class MLPRelation(nn.Module):
    def __init__(self, entity_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(entity_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, e_x: torch.Tensor, e_y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e_x: (B, N, D_ent) or (B, D_ent)
            e_y: (B, N, D_ent) or (B, D_ent)
        Returns:
            truth: matching shape, values in [0, 1]
        """
        combined = torch.cat([e_x, e_y], dim=-1)
        return self.net(combined).squeeze(-1)
```

### 3.2 Bilinear Relation

A bilinear scoring function. More parameter-efficient than MLP for moderate D_ent and more interpretable -- the learned weight matrix W encodes compatibility between entity feature dimensions.

```python
class BilinearRelation(nn.Module):
    def __init__(self, entity_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(entity_dim, entity_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, e_x: torch.Tensor, e_y: torch.Tensor) -> torch.Tensor:
        """e_x^T W e_y + bias -> sigmoid."""
        # e_x: (B, ..., D), e_y: (B, ..., D)
        score = torch.einsum("...i,ij,...j->...", e_x, self.W, e_y) + self.bias
        return torch.sigmoid(score)
```

The bilinear relation is the default (`GroundingConfig.relation_type = "bilinear"`). It captures directional relationships: R(x,y) is not necessarily equal to R(y,x) because W is not constrained to be symmetric.

### 3.3 Neural Tensor Network (NTN) Relation

The NTN combines bilinear interaction with an MLP, providing the highest expressivity at the cost of more parameters.

```python
class NTNRelation(nn.Module):
    def __init__(self, entity_dim: int, num_slices: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.num_slices = num_slices
        # Bilinear tensor: k slices of (D, D) matrices
        self.W = nn.Parameter(torch.randn(num_slices, entity_dim, entity_dim) * 0.02)
        # Linear layer on concatenation
        self.V = nn.Linear(entity_dim * 2, num_slices)
        self.b = nn.Parameter(torch.zeros(num_slices))
        # Output projection
        self.out = nn.Sequential(
            nn.Linear(num_slices, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, e_x: torch.Tensor, e_y: torch.Tensor) -> torch.Tensor:
        """
        e_x^T W[1..k] e_y + V[e_x; e_y] + b -> tanh -> linear -> sigmoid.
        """
        # Bilinear: (B, ..., k) via einsum over k slices
        bilinear = torch.einsum("...i,kij,...j->...k", e_x, self.W, e_y)
        # Linear on concatenation
        linear = self.V(torch.cat([e_x, e_y], dim=-1))
        # Combine
        h = torch.tanh(bilinear + linear + self.b)
        return self.out(h).squeeze(-1)
```

### 3.4 Batch Evaluation Over All Entity Pairs

To evaluate a relation over all entity pairs, expand entities into pairwise form and produce a truth matrix `(B, N, N)`:

```python
def evaluate_pairwise(
    relation: nn.Module,
    entities: torch.Tensor,   # (B, N, D_ent)
    mask: torch.BoolTensor,   # (B, N)
) -> torch.Tensor:
    B, N, D = entities.shape
    e_x = entities.unsqueeze(2).expand(B, N, N, D)  # (B, N, N, D)
    e_y = entities.unsqueeze(1).expand(B, N, N, D)  # (B, N, N, D)
    truth_matrix = relation(e_x, e_y)                # (B, N, N)

    # Apply pair mask
    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)  # (B, N, N)
    truth_matrix = truth_matrix * pair_mask.float()
    return truth_matrix
```

### 3.5 Symmetric vs. Asymmetric Relations

- **Asymmetric** (default): R(x,y) != R(y,x) in general. Use for relations like "parent_of", "causes", "larger_than".
- **Symmetric**: Enforce R(x,y) = R(y,x) by averaging: `truth = 0.5 * (R(e_x, e_y) + R(e_y, e_x))`. Alternatively, sort entity pair by some canonical order before passing to the relation module.

Declare symmetry in the relation metadata so the rule engine can exploit it for optimization (evaluate only upper triangle of the truth matrix and mirror).

### 3.6 RelationRegistry

```python
class RelationRegistry(nn.ModuleDict):
    def __init__(self, entity_dim: int, hidden_dim: int, default_type: str = "bilinear"):
        super().__init__()
        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim
        self.default_type = default_type
        self._metadata: Dict[str, Dict] = {}

    def register_relation(
        self,
        name: str,
        module: Optional[nn.Module] = None,
        symmetric: bool = False,
        relation_type: Optional[str] = None,
    ):
        rtype = relation_type or self.default_type
        if module is None:
            if rtype == "mlp":
                module = MLPRelation(self.entity_dim, self.hidden_dim)
            elif rtype == "bilinear":
                module = BilinearRelation(self.entity_dim)
            elif rtype == "ntn":
                module = NTNRelation(self.entity_dim)
            else:
                raise ValueError(f"Unknown relation_type: {rtype}")
        self[name] = module
        self._metadata[name] = {"symmetric": symmetric, "type": rtype}

    def evaluate(
        self,
        name: str,
        e_x: torch.Tensor,
        e_y: torch.Tensor,
    ) -> torch.Tensor:
        return self[name](e_x, e_y)

    def evaluate_pairwise(
        self,
        name: str,
        entities: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        return evaluate_pairwise(self[name], entities, mask)
```

---

## 4. Real Logic Grounding (use_ltn=True Mode)

When `GroundingConfig.use_ltn` (mapped from `ReasoningConfig.use_ltn`) is `True`, the grounding pipeline operates in full LTN / Real Logic mode. In this mode, every symbol in the logical language -- constants, variables, function symbols, predicates, and relations -- is grounded onto real-valued tensors through neural computational graphs. The grounding object G is the central registry that maps symbol names to their neural implementations.

### 4.1 Constants and Terms

A **constant** is a fixed point in the grounding space represented as a learnable `nn.Parameter` of shape `(D_ent,)`. Constants correspond to known entities (e.g., specific objects, categories).

```python
class GroundedConstant(nn.Module):
    def __init__(self, entity_dim: int, init_value: Optional[torch.Tensor] = None):
        super().__init__()
        if init_value is not None:
            self.value = nn.Parameter(init_value.clone())
        else:
            self.value = nn.Parameter(torch.randn(entity_dim) * 0.1)

    def forward(self) -> torch.Tensor:
        return self.value
```

A **variable** ranges over a domain of entities. It is not a learnable parameter but rather a reference to a batch of entity embeddings `(B, N, D_ent)` provided at runtime. The rule engine binds variables to entity sets during formula evaluation.

A **term** is either a constant, a variable, or the result of applying a function symbol to other terms. Terms are always tensors in the entity embedding space.

### 4.2 Function Symbols

A **function symbol** f of arity k maps k entity embeddings to a new entity embedding. It is grounded as a neural network: `f: R^(k * D_ent) -> R^D_ent`.

```python
class GroundedFunction(nn.Module):
    def __init__(self, arity: int, entity_dim: int, hidden_dim: int):
        super().__init__()
        self.arity = arity
        self.net = nn.Sequential(
            nn.Linear(entity_dim * arity, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, entity_dim),
        )

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        assert len(args) == self.arity
        combined = torch.cat(args, dim=-1)
        return self.net(combined)
```

Use ELU activations (following LTN convention) rather than ReLU for smoother gradients in the grounding space.

### 4.3 Predicates in Real Logic Mode

In Real Logic mode, predicates are identical to the predicate modules described in Section 2, but they are registered in the grounding object rather than in a standalone registry. The grounding object tracks their arity and symbol type metadata.

### 4.4 Grounding Object Structure

The central grounding object maps every symbol name to a triple of `(module, symbol_type, arity)`:

| Symbol Type | Module Class | Arity | Output |
|---|---|---|---|
| `"constant"` | `GroundedConstant` | 0 | `(D_ent,)` |
| `"function"` | `GroundedFunction` | k >= 1 | `(*, D_ent)` |
| `"predicate"` | `MLPPredicate` or similar | 1 | `(*, )` in [0,1] |
| `"relation"` | `BilinearRelation` / `NTNRelation` / `MLPRelation` | 2 | `(*, )` in [0,1] |

```python
class Grounding(nn.Module):
    def __init__(self, entity_dim: int, hidden_dim: int):
        super().__init__()
        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim
        self.symbols = nn.ModuleDict()
        self._symbol_meta: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        module: nn.Module,
        symbol_type: str,
        arity: int,
    ):
        self.symbols[name] = module
        self._symbol_meta[name] = {"type": symbol_type, "arity": arity}

    def ground(self, name: str, *args: torch.Tensor) -> torch.Tensor:
        """Evaluate a grounded symbol with the given arguments."""
        meta = self._symbol_meta[name]
        module = self.symbols[name]
        if meta["type"] == "constant":
            return module()
        else:
            return module(*args)

    def get_meta(self, name: str) -> Dict[str, Any]:
        return self._symbol_meta[name]

    @property
    def symbol_names(self) -> List[str]:
        return list(self.symbols.keys())
```

### 4.5 Relationship Between Standard and LTN Modes

When `use_ltn=False` (default), the pipeline uses separate `PredicateRegistry` and `RelationRegistry` objects. Entity extraction still applies, but constants and function symbols are not available.

When `use_ltn=True`, the pipeline uses the unified `Grounding` object. The `PredicateRegistry` and `RelationRegistry` are replaced by symbol registrations in the grounding. The EntityExtractor is unchanged.

---

## 5. Checkpointing and Auditing

### 5.1 State Dict Compatibility

All grounding components are `nn.Module` subclasses, so `torch.save(grounding.state_dict(), path)` and `grounding.load_state_dict(torch.load(path))` work natively. The symbol metadata (`_symbol_meta`) is not part of the state dict and must be serialized separately or reconstructed from the configuration.

Save the full grounding snapshot as:

```python
def save_grounding(grounding: Grounding, path: str):
    snapshot = {
        "state_dict": grounding.state_dict(),
        "symbol_meta": grounding._symbol_meta,
        "entity_dim": grounding.entity_dim,
        "hidden_dim": grounding.hidden_dim,
    }
    torch.save(snapshot, path)

def load_grounding(path: str) -> Grounding:
    snapshot = torch.load(path, weights_only=False)
    grounding = Grounding(snapshot["entity_dim"], snapshot["hidden_dim"])
    # Reconstruct modules from metadata before loading state dict
    for name, meta in snapshot["symbol_meta"].items():
        module = _create_module_from_meta(meta, snapshot["entity_dim"], snapshot["hidden_dim"])
        grounding.register(name, module, meta["type"], meta["arity"])
    grounding.load_state_dict(snapshot["state_dict"])
    return grounding
```

### 5.2 Symbol-to-Parameter Mapping

Expose a mapping from symbol names to their parameter sets. This is critical for:

- Applying different learning rates per symbol (e.g., freeze predicate parameters while training relation parameters).
- Debugging which parameters are receiving gradients.
- Computing per-symbol gradient norms for training diagnostics.

```python
def symbol_parameters(self) -> Dict[str, List[nn.Parameter]]:
    """Return {symbol_name: [param1, param2, ...]} for all symbols."""
    return {
        name: list(self.symbols[name].parameters())
        for name in self.symbols
    }
```

### 5.3 Auditing API

The `audit()` method returns a structured dictionary describing every symbol's parameter shapes. Use this to verify that a checkpoint was loaded correctly, to check that newly registered symbols have the expected dimensions, and to diagnose why a particular rule is violated (by inspecting the parameter statistics of involved symbols).

```python
def audit(self) -> Dict[str, Dict[str, Any]]:
    """
    Return audit info for all grounded symbols.

    Returns:
        {
            "is_mammal": {
                "type": "predicate",
                "arity": 1,
                "parameters": {
                    "net.0.weight": (hidden_dim, entity_dim),
                    "net.0.bias": (hidden_dim,),
                    "net.2.weight": (1, hidden_dim),
                    "net.2.bias": (1,),
                },
                "total_params": <int>,
                "requires_grad": True,
            },
            ...
        }
    """
    result = {}
    for name in self.symbols:
        module = self.symbols[name]
        meta = self._symbol_meta[name]
        params = {}
        total = 0
        grad_status = True
        for pname, p in module.named_parameters():
            params[pname] = tuple(p.shape)
            total += p.numel()
            if not p.requires_grad:
                grad_status = False
        result[name] = {
            "type": meta["type"],
            "arity": meta["arity"],
            "parameters": params,
            "total_params": total,
            "requires_grad": grad_status,
        }
    return result
```

### 5.4 Serialization Format

Grounding snapshots use the following structure on disk:

```
grounding_snapshot.pt
  ├── state_dict          # OrderedDict of parameter tensors
  ├── symbol_meta         # Dict[str, Dict] — type, arity per symbol
  ├── entity_dim          # int
  └── hidden_dim          # int
```

For human-readable inspection, export the audit dict as JSON alongside the checkpoint:

```python
import json
audit_info = grounding.audit()
with open("grounding_audit.json", "w") as f:
    json.dump(audit_info, f, indent=2, default=str)
```

---

## 6. Integration with Workspace

### 6.1 Data Flow

The EntityExtractor receives workspace slot representations from the Global Workspace module. The data flow is:

```
Multi-Modal Inputs
    |
    v
Global Workspace (competition + broadcast)
    |
    v
workspace slots: (B, K, D_ws)  +  slot attention weights: (B, K)
    |
    v
EntityExtractor (slot_identity or proposal_head)
    |
    v
entities: (B, N, D_ent)  +  entity mask: (B, N)
    |
    v
Predicate/Relation Modules  -->  Rule Engine
```

The workspace output dictionary (from `GlobalWorkspace.forward()` or `SelectionBroadcastWorkspace.forward()`) provides:

- `workspace` key: the aggregated workspace representation `(B, D_ws)`. This is the summary vector, not the individual slots.
- The per-slot features are the `winners` tensor `(B, K, D_ws)` available from the competition step. Access this by returning attention-weighted features before the summation step, or by storing them in the output dictionary.

When the workspace uses `capacity_limit=7` (Miller's Law), the maximum K is 7 slots. The slot identity extractor then produces at most 7 entities. For richer entity sets, use the proposal head extractor with `max_entities > K`.

### 6.2 Context Injection

Optionally concatenate the workspace summary vector to each entity embedding to provide global context. This allows predicates and relations to condition on the overall scene representation, not just individual entity features.

```python
class ContextInjector(nn.Module):
    def __init__(self, workspace_dim: int, entity_dim: int):
        super().__init__()
        self.context_proj = nn.Linear(workspace_dim, entity_dim)

    def forward(
        self,
        entities: torch.Tensor,          # (B, N, D_ent)
        workspace_summary: torch.Tensor,  # (B, D_ws)
    ) -> torch.Tensor:
        ctx = self.context_proj(workspace_summary)  # (B, D_ent)
        ctx = ctx.unsqueeze(1).expand_as(entities)  # (B, N, D_ent)
        return entities + ctx                        # residual addition
```

Enable context injection when predicates need scene-level information (e.g., "is_occluded" depends on the spatial layout of other objects). Disable it when predicates should be purely entity-local.

### 6.3 Workspace Slot Attention Weights

The Global Workspace competition produces attention weights `(B, K)` indicating which slots "won" access to consciousness. Pass these weights to the entity extractor as salience priors:

- In `slot_identity` mode, multiply entity embeddings by attention weights to bias the grounding toward salient entities.
- In `proposal_head` mode, use attention weights as an additional input feature to the cross-attention mechanism (concatenate to slot features before serving as keys/values).

This ensures that entities corresponding to attended workspace slots have stronger grounding, which aligns the symbolic layer with the workspace's attentional state.

---

## 7. Entity Proposal Head (Detailed)

This section expands on the proposal_head extractor type introduced in Section 1.2.

### 7.1 Cross-Attention Mechanism

The proposal head uses N learnable query vectors that attend to K workspace slots. The queries are initialized from a zero-mean normal distribution with small standard deviation (0.02) to break symmetry while keeping initial attention uniform.

Architecture:

```
query:  (B, N, D_ent)   -- learnable, expanded per batch
key:    (B, K, D_ws)     -- workspace slots (projected if D_ws != D_ent)
value:  (B, K, D_ws)     -- workspace slots
                         -- key_padding_mask from slot_mask

Output: (B, N, D_ent) attended entity proposals
```

Use 8 attention heads (configurable) with `D_ent / num_heads` per head. When `D_ws != D_ent`, the cross-attention module uses `kdim=D_ws` and `vdim=D_ws` to handle the dimension mismatch, with internal projection to `D_ent`.

### 7.2 Confidence Scores

Each entity proposal has an associated confidence score in [0,1]. This score indicates how well-supported the entity is by the workspace evidence. Confidence is computed by a single-layer projection from the entity embedding followed by sigmoid.

Training signal for confidence: when supervised entity labels are available, train confidence with binary cross-entropy. When unsupervised, confidence is trained end-to-end through the symbolic constraint loss -- entities that participate in satisfied rules will have their confidence reinforced.

### 7.3 Entity Deduplication

When multiple query vectors converge on the same workspace slot, duplicated entity proposals may arise. Apply cosine similarity deduplication:

```python
def deduplicate_entities(
    entities: torch.Tensor,          # (B, N, D_ent)
    confidence: torch.Tensor,        # (B, N)
    threshold: float = 0.9,
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """Merge entity proposals that are too similar."""
    B, N, D = entities.shape
    # Normalize for cosine similarity
    normed = F.normalize(entities, dim=-1)               # (B, N, D)
    sim = torch.bmm(normed, normed.transpose(1, 2))      # (B, N, N)

    # For each pair above threshold, keep the one with higher confidence
    mask = torch.ones(B, N, dtype=torch.bool, device=entities.device)
    for i in range(N):
        for j in range(i + 1, N):
            duplicate = sim[:, i, j] > threshold          # (B,)
            lower_conf = confidence[:, i] < confidence[:, j]
            # Mask out the lower-confidence duplicate
            mask[:, i] = mask[:, i] & ~(duplicate & lower_conf)
            mask[:, j] = mask[:, j] & ~(duplicate & ~lower_conf)

    return entities, mask
```

For large N, replace the nested loop with a vectorized approach: compute the full similarity matrix, apply threshold, and use a greedy non-maximum suppression pass sorted by confidence.

### 7.4 End-to-End Training

The entity proposal head is trained jointly with the rest of the symbolic pipeline. Gradients flow from:

1. **Symbolic constraint loss** -> predicate/relation outputs -> entity embeddings -> cross-attention weights -> workspace slot values. This trains the proposal head to extract entities that satisfy symbolic rules.
2. **Task loss** (e.g., classification, generation) -> output heads -> workspace -> slots -> entity proposals. This trains the proposal head to extract task-relevant entities.
3. **Confidence loss** (optional) -> confidence head -> entity embeddings. This trains the confidence scores when supervised labels are available.

No separate entity proposal pretraining phase is required, though pretraining on an object detection objective can accelerate convergence.

---

## 8. Code Patterns

### 8.1 EntityExtractor Factory (Complete)

```python
def create_entity_extractor(
    config: GroundingConfig,
    workspace_dim: int,
    entity_dim: int,
) -> nn.Module:
    """Factory for entity extractors.

    Args:
        config: GroundingConfig with extractor_type and max_entities.
        workspace_dim: D_ws from WorkspaceConfig.
        entity_dim: D_ent from SymbolicConfig.

    Returns:
        nn.Module implementing (slots, mask) -> (entities, mask[, confidence])
    """
    if config.extractor_type == "slot_identity":
        return SlotIdentityExtractor(workspace_dim, entity_dim)
    elif config.extractor_type == "proposal_head":
        return ProposalHeadExtractor(
            workspace_dim=workspace_dim,
            entity_dim=entity_dim,
            num_proposals=config.max_entities,
            num_heads=8,
            dedup_threshold=0.9,
        )
    else:
        raise ValueError(f"Unknown extractor_type: {config.extractor_type}")
```

### 8.2 PredicateRegistry Class

See Section 2.2 for the full implementation. Key integration pattern:

```python
# Initialization
predicates = PredicateRegistry(entity_dim=256, hidden_dim=512)
predicates.register_predicate("is_mammal")
predicates.register_predicate("is_large")
predicates.register_predicate("is_dangerous")

# Forward pass
entities, mask = entity_extractor(workspace_slots, slot_mask)
truth_is_mammal = predicates.evaluate("is_mammal", entities, mask)  # (B, N)
all_truths = predicates.evaluate_all(entities, mask)                 # dict
```

### 8.3 RelationRegistry Class

See Section 3.6 for the full implementation. Key integration pattern:

```python
# Initialization
relations = RelationRegistry(entity_dim=256, hidden_dim=512, default_type="bilinear")
relations.register_relation("parent_of", symmetric=False)
relations.register_relation("similar_to", symmetric=True, relation_type="bilinear")
relations.register_relation("causes", symmetric=False, relation_type="ntn")

# Forward pass — pairwise truth matrix
truth_matrix = relations.evaluate_pairwise("parent_of", entities, mask)  # (B, N, N)
```

### 8.4 Grounding Checkpoint Save/Load

```python
# Save
save_grounding(grounding, "checkpoints/grounding_epoch_10.pt")

# Load
grounding = load_grounding("checkpoints/grounding_epoch_10.pt")

# Verify
audit = grounding.audit()
for name, info in audit.items():
    print(f"{name}: {info['type']}, arity={info['arity']}, "
          f"params={info['total_params']}, grad={info['requires_grad']}")
```

### 8.5 Auditing Symbol-Parameter Mapping

```python
# Get per-symbol parameters for optimizer groups
sym_params = grounding.symbol_parameters()

# Create optimizer with per-symbol learning rates
param_groups = []
for name, params in sym_params.items():
    meta = grounding.get_meta(name)
    lr = 1e-3 if meta["type"] == "predicate" else 1e-4
    param_groups.append({"params": params, "lr": lr, "name": name})

optimizer = torch.optim.AdamW(param_groups)

# Diagnostic: per-symbol gradient norms
for name, params in sym_params.items():
    grad_norm = sum(p.grad.norm().item() for p in params if p.grad is not None)
    print(f"{name}: grad_norm = {grad_norm:.6f}")
```

### 8.6 Full Pipeline Assembly

```python
def build_grounding_pipeline(
    grounding_config: GroundingConfig,
    symbolic_config: SymbolicConfig,
    workspace_config: WorkspaceConfig,
) -> Dict[str, nn.Module]:
    """Assemble the complete grounding pipeline."""
    entity_dim = symbolic_config.entity_dim
    hidden_dim = symbolic_config.hidden_dim
    workspace_dim = workspace_config.workspace_dim

    extractor = create_entity_extractor(grounding_config, workspace_dim, entity_dim)

    if symbolic_config.use_ltn:
        grounding = Grounding(entity_dim, hidden_dim)
        # Predicates and relations are registered in the Grounding object
        # by the caller based on the theory being modeled.
        return {
            "extractor": extractor,
            "grounding": grounding,
        }
    else:
        predicates = PredicateRegistry(entity_dim, hidden_dim)
        relations = RelationRegistry(
            entity_dim, hidden_dim, default_type=grounding_config.relation_type
        )
        return {
            "extractor": extractor,
            "predicates": predicates,
            "relations": relations,
        }
```

---

## Appendix A: Entity / Predicate / Relation Shape Reference

| Component | Input Shape(s) | Output Shape | Value Range | Notes |
|---|---|---|---|---|
| **SlotIdentityExtractor** | slots `(B, K, D_ws)`, mask `(B, K)` | entities `(B, K, D_ent)`, mask `(B, K)` | real | N = K |
| **ProposalHeadExtractor** | slots `(B, K, D_ws)`, mask `(B, K)` | entities `(B, N, D_ent)`, mask `(B, N)`, conf `(B, N)` | real / [0,1] | N from config |
| **MLPPredicate** | entities `(B, N, D_ent)` | truth `(B, N)` | [0, 1] | Apply mask after |
| **BilinearRelation** | e_x `(B, ..., D_ent)`, e_y `(B, ..., D_ent)` | truth `(B, ...)` | [0, 1] | Asymmetric by default |
| **MLPRelation** | e_x `(B, ..., D_ent)`, e_y `(B, ..., D_ent)` | truth `(B, ...)` | [0, 1] | Concat + MLP |
| **NTNRelation** | e_x `(B, ..., D_ent)`, e_y `(B, ..., D_ent)` | truth `(B, ...)` | [0, 1] | k bilinear slices |
| **Pairwise evaluation** | entities `(B, N, D_ent)`, mask `(B, N)` | truth_matrix `(B, N, N)` | [0, 1] | pair_mask applied |
| **GroundedConstant** | (none) | value `(D_ent,)` | real | Learnable parameter |
| **GroundedFunction** | k args of `(*, D_ent)` | result `(*, D_ent)` | real | ELU activations |
| **ContextInjector** | entities `(B, N, D_ent)`, summary `(B, D_ws)` | entities `(B, N, D_ent)` | real | Residual addition |

**Dimension defaults** (from configs):

| Dimension | Config Source | Default (minimal) | Default (production 7B) |
|---|---|---|---|
| D_ws | `WorkspaceConfig.workspace_dim` | 512 | 4096 |
| D_ent | `SymbolicConfig.entity_dim` | 256 | 256 |
| hidden_dim | `SymbolicConfig.hidden_dim` | 512 | 512 |
| K (slots) | `WorkspaceConfig.capacity_limit` | 7 | 7 |
| N (entities) | `GroundingConfig.max_entities` | 32 | 32 |

---

## Appendix B: Integration Points with Other BrainAI Modules

| Upstream Module | Interface | Data Provided to Grounding |
|---|---|---|
| **Global Workspace** (`brain_ai/workspace/global_workspace.py`) | `workspace` output dict | Slot features `(B, K, D_ws)`, attention weights `(B, K)`, summary `(B, D_ws)` |
| **SNN Core** (`brain_ai/core/`) | Via workspace integration | Spike-coded features projected to workspace dim; indirect input to entities |
| **HTM** (`brain_ai/temporal/`) | Via workspace integration | Temporal predictions and anomaly scores modulate workspace attention |
| **Encoders** (`brain_ai/encoders/`) | Via workspace competition | Multi-modal features compete for workspace slots; winners become entity candidates |

| Downstream Module | Interface | Data Consumed from Grounding |
|---|---|---|
| **Rule Engine** (`brain_ai/reasoning/rule_engine.py`) | Entity embeddings + registries/grounding | Entities `(B, N, D_ent)`, mask `(B, N)`, predicate/relation evaluators |
| **SymbolicReasoner** (`brain_ai/reasoning/symbolic.py`) | Constraint loss, truth values | `constraint_loss` scalar for training, `truth` `(B, Q)` for queries |
| **DualProcessReasoner** (`brain_ai/reasoning/system2.py`) | Symbolic features | System 2 receives rule violation vectors and entity embeddings as input for iterative deliberation |
| **Active Inference** (`brain_ai/decision/`) | Symbolic beliefs | Predicate truth values serve as belief distributions for expected free energy computation |
| **Meta-Learning** (`brain_ai/meta/`) | Per-symbol parameters | MAML inner loop can adapt predicate/relation parameters for few-shot symbolic tasks |

**Gradient flow path**: Task loss and constraint loss both backpropagate through the grounding pipeline into the workspace and ultimately into upstream encoders. This is the mechanism by which symbolic rule satisfaction shapes perceptual representations. Verify this path is intact by checking that encoder parameter gradients are non-zero after a backward pass that includes the constraint loss term.
