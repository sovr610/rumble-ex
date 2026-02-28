# State Management

## Problem

Several modules are stateful:
- SNN membrane potentials (if streaming frame-by-frame)
- Workspace working memory (CfC/LTC/GRU hidden state)
- HTM temporal memory and spatial pooler state
- Active inference belief state
- Meta-learning eligibility traces
- Per-module RNG state (for reproducibility)

Without consolidation, state lives inside modules where it becomes impossible to
checkpoint, reproduce, or transfer between contexts.

## BrainAIState

All state is consolidated into one serializable container:

```python
@dataclass
class BrainAIState:
    """Consolidated state for all stateful modules."""
    wm_state: Optional[Any] = None            # Working memory hidden state
    htm_state: Optional[Tuple] = None         # (tm_state, sp_state)
    snn_state: Optional[Dict[str, Tensor]] = None  # {layer_name: membrane_potential}
    belief_state: Optional[Tensor] = None     # (B, S) active inference belief
    eligibility_state: Optional[Tensor] = None  # Eligibility trace tensors
    rng_state: Optional[Dict[str, Any]] = None  # Per-module torch RNG states
    step_count: int = 0                       # Steps since last reset
```

## Two Forward Modes

### Stateless Mode

Standard training forward pass. No state carried between calls:

```python
output = brain(inputs)  # Implicitly: state=None
# or explicitly:
output, new_state = brain(inputs, state=None, return_state=True)
```

When `state=None`, modules initialize their own internal state (zeros, random, etc.)
and discard it after the forward pass.

### Streaming Mode

For real-time inference, RL rollouts, or video processing:

```python
state = BrainAIState()  # Initial empty state
for frame in video_stream:
    output, state = brain.step({"vision": frame}, state)
    # state carries forward to next frame
```

The `step()` method:
1. Unpacks `state` into per-module states
2. Passes state to each module
3. Collects updated states
4. Returns new consolidated `BrainAIState`

```python
def step(self, inputs, state: BrainAIState) -> Tuple[SystemOutput, BrainAIState]:
    """Single timestep forward with state management."""
    ctx = PipelineContext(
        batch=normalize_inputs(inputs, self.config),
        config=self.config,
        state=state,
        return_details=False,
        device=self.device,
    )
    ctx = self.plan.execute(ctx)
    output = self._assemble_output(ctx)

    new_state = BrainAIState(
        wm_state=ctx.new_state.get("wm"),
        htm_state=ctx.new_state.get("htm"),
        snn_state=ctx.new_state.get("snn"),
        belief_state=ctx.new_state.get("belief"),
        eligibility_state=ctx.new_state.get("eligibility"),
        rng_state=ctx.new_state.get("rng"),
        step_count=state.step_count + 1,
    )
    return output, new_state
```

## Checkpointing

### Save/Load State

```python
# Save
state_dict = {
    "model": brain.state_dict(),
    "brain_state": brain_state,  # BrainAIState
    "config": brain.config,
    "step": global_step,
    "deps_report": brain.deps_report(),
}
torch.save(state_dict, "checkpoint.pt")

# Load
ckpt = torch.load("checkpoint.pt")
brain = BrainAI(ckpt["config"])
brain.load_state_dict(ckpt["model"])
brain_state = ckpt["brain_state"]
```

### State Serialization

`BrainAIState` must be serializable by `torch.save`:
- Tensor fields: saved as-is
- `rng_state`: use `torch.get_rng_state()` / `torch.set_rng_state()`
- Module-specific opaque state (HTM, CfC): must implement `state_dict()`/`load_state_dict()`

### Partial State Reset

Sometimes only specific module states need resetting (e.g., reset HTM on new episode
but keep workspace working memory):

```python
def reset_state(self, state: BrainAIState, modules: List[str] = None) -> BrainAIState:
    """Reset specific module states. If modules is None, reset all."""
    if modules is None:
        return BrainAIState()

    new_state = copy(state)
    if "htm" in modules:
        new_state.htm_state = None
    if "wm" in modules:
        new_state.wm_state = None
    if "snn" in modules:
        new_state.snn_state = None
    if "belief" in modules:
        new_state.belief_state = None
    if "eligibility" in modules:
        new_state.eligibility_state = None
    new_state.step_count = 0
    return new_state
```

## State in PipelineContext

Each stage reads its input state from `ctx.state` and writes updated state to
`ctx.new_state`:

```python
# Inside workspace stage run function:
def workspace_run(ctx):
    wm_state = ctx.state.wm_state if ctx.state else None
    ws_output = workspace(ctx.encoder_outputs, wm_state=wm_state)

    ctx.repr = ws_output.slots
    ctx.new_state["wm"] = ws_output.wm_state  # Updated working memory
    return ctx
```

This ensures:
- Each stage only reads/writes its own state slice
- No stage can accidentally corrupt another's state
- The orchestrator collects all state updates after execution

## RNG State Management

For reproducibility, each module gets a keyed RNG:

```python
def init_module_rngs(self, global_seed: int):
    """Create per-module RNG streams."""
    self._rngs = {}
    for i, stage in enumerate(self.plan.stages):
        module_seed = global_seed + hash(stage.name) % (2**32)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(module_seed)
        self._rngs[stage.name] = gen
```

Pass the generator to dropout, sampling, etc. within each stage. This ensures:
- Same global seed → same module behavior
- Reordering or adding modules doesn't change other modules' RNG streams
- State is saveable/loadable via `generator.get_state()` / `set_state()`
