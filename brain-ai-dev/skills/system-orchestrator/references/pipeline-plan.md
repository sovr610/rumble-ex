# PipelinePlan Architecture

## Overview

The PipelinePlan replaces scattered `if config.use_X` conditionals with an explicit,
ordered execution graph built once at `__init__`. This makes the forward pass a simple
loop over stages, each with a guaranteed bypass when disabled.

## Core Classes

### PipelineContext

The context object carries all tensors, state, and details collectors through the pipeline:

```python
@dataclass
class PipelineContext:
    """Mutable context passed through all pipeline stages."""
    # Input
    batch: ModalityBatch
    config: BrainAIConfig

    # Accumulated results (set by stages)
    encoder_outputs: Dict[str, EncoderOutput] = field(default_factory=dict)
    repr: Optional[Tensor] = None              # Main representation (B,T,D) or (B,K,D)
    anomaly_score: Optional[Tensor] = None
    reasoning: Optional[ReasoningOutput] = None
    modulators: Optional[Dict[str, Tensor]] = None
    action: Optional[Tensor] = None
    decision: Optional[DecisionOutput] = None

    # State (for streaming)
    state: Optional[BrainAIState] = None
    new_state: Optional[BrainAIState] = None

    # Details collection
    details: Dict[str, Any] = field(default_factory=dict)
    return_details: bool = False
    deterministic: bool = False

    # Telemetry
    telemetry: Optional['TelemetrySink'] = None

    # Device / dtype
    device: torch.device = torch.device("cpu")
    compute_dtype: torch.dtype = torch.float32
```

### Stage

Each pipeline stage wraps a module with its contract and bypass:

```python
@dataclass
class Stage:
    """A single step in the pipeline plan."""
    name: str                                  # "encode", "snn", "workspace", etc.
    enabled: bool                              # From config feature flag
    run: Callable[[PipelineContext], PipelineContext]
    bypass: Callable[[PipelineContext], PipelineContext]
    contract_in: Optional[Type] = None         # Expected input type
    contract_out: Optional[Type] = None        # Expected output type
    requires: List[str] = field(default_factory=list)  # Stage names that must run first
```

### PipelinePlan

```python
class PipelinePlan:
    """Ordered execution plan built from config."""

    def __init__(self, config: BrainAIConfig, modules: Dict[str, nn.Module]):
        self.stages: List[Stage] = []
        self._build(config, modules)

    def _build(self, config, modules):
        """Construct stage list from config flags and available modules."""
        # Stage 1: Encoding (always enabled)
        self.stages.append(Stage(
            name="encode",
            enabled=True,
            run=self._make_encode_fn(modules),
            bypass=lambda ctx: ctx,  # Never bypassed
        ))

        # Stage 2: SNN transform (optional)
        self.stages.append(Stage(
            name="snn",
            enabled=config.use_snn and "snn" in modules,
            run=self._make_snn_fn(modules.get("snn")),
            bypass=lambda ctx: ctx,  # Identity: raw encoder output
            requires=["encode"],
        ))

        # Stage 3: Workspace competition (with bypass)
        self.stages.append(Stage(
            name="workspace",
            enabled=config.use_workspace and "workspace" in modules,
            run=self._make_workspace_fn(modules.get("workspace")),
            bypass=self._make_workspace_bypass(),
            requires=["encode"],
        ))

        # Stage 4: HTM temporal (with bypass)
        self.stages.append(Stage(
            name="htm",
            enabled=config.use_htm and "htm" in modules,
            run=self._make_htm_fn(modules.get("htm")),
            bypass=self._make_htm_bypass(),
            requires=["workspace"],
        ))

        # Stage 5: Symbolic reasoning (with bypass)
        self.stages.append(Stage(
            name="reasoning",
            enabled=config.use_symbolic and "reasoner" in modules,
            run=self._make_reasoning_fn(modules.get("reasoner")),
            bypass=self._make_reasoning_bypass(),
            requires=["workspace"],
        ))

        # Stage 6: Meta-learning / neuromodulation (with bypass)
        self.stages.append(Stage(
            name="meta",
            enabled=config.use_meta and "neuromodulation" in modules,
            run=self._make_meta_fn(modules.get("neuromodulation")),
            bypass=self._make_meta_bypass(),
            requires=["workspace"],
        ))

        # Stage 7: Active inference decision (with bypass)
        self.stages.append(Stage(
            name="decision",
            enabled="active_inference" in modules,
            run=self._make_decision_fn(modules.get("active_inference")),
            bypass=lambda ctx: ctx,
            requires=["workspace"],
        ))

        # Stage 8: Output heads (always enabled)
        self.stages.append(Stage(
            name="output",
            enabled=True,
            run=self._make_output_fn(modules.get("decision_heads")),
            bypass=lambda ctx: ctx,
            requires=["workspace"],
        ))

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute all stages in order."""
        for stage in self.stages:
            if ctx.telemetry:
                ctx.telemetry.on_module_start(stage.name)

            if stage.enabled:
                ctx = stage.run(ctx)
            else:
                ctx = stage.bypass(ctx)

            if ctx.telemetry:
                ctx.telemetry.on_module_end(stage.name)

        return ctx
```

## Stage Implementation Patterns

### Encode Stage

```python
def _make_encode_fn(self, modules):
    encoders = modules.get("encoders", {})
    sorted_modalities = sorted(encoders.keys())  # Deterministic order

    def run(ctx):
        for name in sorted_modalities:
            if name in ctx.batch.modalities_present:
                raw = getattr(ctx.batch, name, None)
                if raw is not None:
                    enc_out = encoders[name](raw)
                    # Normalize to EncoderOutput contract
                    ctx.encoder_outputs[name] = normalize_encoder_output(enc_out, name)
        return ctx
    return run
```

### Workspace Bypass

When workspace is disabled, produce a workspace-like representation:

```python
def _make_workspace_bypass(self):
    def bypass(ctx):
        # Concatenate encoder features and pool to (B, T, D)
        all_feats = [eo.feats for eo in ctx.encoder_outputs.values()]
        if all_feats:
            concat = torch.cat(all_feats, dim=1)  # (B, T_total, D)
            ctx.repr = self.fallback_proj(concat)  # Project to workspace_dim
        else:
            B = 1  # Minimum batch
            ctx.repr = torch.zeros(B, 1, self.workspace_dim, device=ctx.device)
        return ctx
    return bypass
```

### HTM Bypass

```python
def _make_htm_bypass(self):
    def bypass(ctx):
        B = ctx.repr.shape[0]
        ctx.anomaly_score = torch.zeros(B, device=ctx.device)
        ctx.details["htm"] = {"anomaly_score": ctx.anomaly_score,
                               "pred_sdr_stats": None,
                               "promoted_patterns_count": 0}
        return ctx
    return bypass
```

## How BrainAI.__init__ Uses PipelinePlan

```python
class BrainAI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Build all modules
        self._build_modules()

        # Build pipeline plan from config + modules
        self.plan = PipelinePlan(config, self._get_module_dict())

    def forward(self, inputs, state=None, return_details=False, deterministic=False):
        ctx = PipelineContext(
            batch=normalize_inputs(inputs, self.config),
            config=self.config,
            state=state,
            return_details=return_details,
            deterministic=deterministic,
            device=next(self.parameters()).device,
        )
        ctx = self.plan.execute(ctx)
        return self._assemble_output(ctx)
```

## Adding a New Stage

To add a new module to the pipeline:

1. Define its output contract in `types.py`
2. Add a feature flag to `BrainAIConfig`
3. Implement the module in its own directory
4. Add a `Stage` to `PipelinePlan._build()` with:
   - `run` function wrapping the module call
   - `bypass` function producing valid neutral output
   - `requires` listing upstream dependencies
5. Update `SystemDetails` to include the new module's introspection fields
6. Add contract assertions
7. Add tests (see `references/testing-matrix.md`)
