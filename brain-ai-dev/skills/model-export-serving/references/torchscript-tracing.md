# TorchScript Tracing and Scripting for BrainAI

## Overview

TorchScript is PyTorch's mechanism for serializing models into a format that can be loaded and run in C++ (libtorch) or optimized Python environments without the Python runtime. For BrainAI, TorchScript is the preferred export format because it handles the full range of BrainAI's dynamic behavior: conditional dual-process routing, iterative System 2 reasoning, SNN temporal loops, and feature flag branching. This reference covers when to use tracing vs. scripting, handling each BrainAI module, serialization best practices, and common pitfalls.

## Tracing vs. Scripting: Fundamental Differences

### Tracing (`torch.jit.trace`)

Tracing records the operations that actually run during a single forward pass with concrete inputs. It produces a static computation graph.

**Strengths**:
- Simple to use -- just provide sample inputs
- Captures the exact computation path taken
- Works with any PyTorch code, including third-party libraries
- Produces optimized graphs (dead code elimination, constant folding)

**Weaknesses**:
- Cannot capture data-dependent control flow (if/else on tensor values)
- Cannot capture dynamic loops (for loops with variable iteration count)
- Only records the single path taken during tracing
- Silent failures: produces wrong results instead of errors when control flow is hit

### Scripting (`torch.jit.script`)

Scripting compiles Python code into TorchScript IR by analyzing the source code AST. It can represent control flow, loops, and conditionals.

**Strengths**:
- Faithfully captures all control flow (if/else, while loops, for loops)
- No silent failures -- compilation errors are explicit
- Supports type annotations for validation
- Can handle data-dependent branching

**Weaknesses**:
- More restrictive Python subset (no arbitrary Python objects, limited stdlib)
- Requires type annotations on function signatures
- Cannot handle some Python patterns (generators, decorators, complex comprehensions)
- Third-party library calls may not be scriptable

### Decision Matrix for BrainAI Modules

| Module | Recommended | Reason |
|--------|-------------|--------|
| Vision Encoder | Trace | Pure feedforward CNN/ViT, no control flow |
| Text Encoder | Trace | Standard transformer, deterministic path |
| Audio Encoder | Trace | Feedforward mel-spectrogram + CNN |
| Sensor Encoder | Trace | Simple MLP |
| SNN Core | Script | Temporal loop over timesteps with state |
| HTM Layer (LSTM fallback) | Trace | Standard LSTM, deterministic |
| HTM Layer (native) | Script | Sparse operations with conditionals |
| Global Workspace | Script | Competition mechanism with dynamic selection |
| Dual-Process Reasoner | Script | Conditional System 1/2 routing |
| Active Inference Agent | Script | Planning loop with variable steps |
| Neuromodulatory Gate | Trace | Simple feedforward gating |
| Decision Heads | Trace | Linear projection layers |
| Engram Memory | Script | Hash computation with dynamic indexing |

## Module-by-Module Export Guide

### Vision Encoder (Trace)

The vision encoder is a feedforward network (CNN or ViT) with no data-dependent control flow. Tracing is ideal.

```python
vision_encoder = model.encoders['vision']
vision_encoder.set_mode_inference()

sample_image = torch.randn(1, 1, 28, 28)
traced_vision = torch.jit.trace(vision_encoder, sample_image)

# Verify
with torch.no_grad():
    original_out = vision_encoder(sample_image)
    traced_out = traced_vision(sample_image)
    assert torch.allclose(original_out, traced_out, atol=1e-6)
```

**Caveat**: If the vision encoder uses `nn.AdaptiveAvgPool2d` with output size computed dynamically, tracing captures the specific output size used during tracing. Ensure the sample input has representative spatial dimensions.

### Text Encoder (Trace with Padding Mask Handling)

The text encoder uses transformer layers with optional padding masks. Tracing captures the mask handling for the specific mask pattern provided.

```python
text_encoder = model.encoders['text']
text_encoder.set_mode_inference()

# Provide representative inputs including masks
sample_tokens = torch.randint(0, 1000, (1, 128))
traced_text = torch.jit.trace(text_encoder, sample_tokens)
```

**Warning**: If the text encoder has different behavior for different sequence lengths (e.g., positional encoding clipping), ensure the sample input covers the expected range or use scripting for that component.

### SNN Core (Script)

The SNN core contains temporal loops and state updates that require scripting.

```python
class SNNCoreScriptable(nn.Module):
    """SNN Core adapted for TorchScript scripting."""

    def __init__(self, original_snn):
        super().__init__()
        self.layers = original_snn.layers
        self.beta = original_snn.beta
        self.threshold = 1.0
        self.num_timesteps: int = original_snn.num_timesteps

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        hidden_size = self.layers[0].out_features

        mem: torch.Tensor = torch.zeros(batch_size, hidden_size, device=x.device)
        spk_sum: torch.Tensor = torch.zeros(batch_size, hidden_size, device=x.device)

        for t in range(self.num_timesteps):
            cur = self.layers[0](x)
            mem = self.beta * mem + cur
            spk = (mem > self.threshold).float()
            mem = mem * (1.0 - spk)  # Reset on spike
            spk_sum = spk_sum + spk

        return spk_sum / float(self.num_timesteps)

scripted_snn = torch.jit.script(SNNCoreScriptable(model.snn_core))
```

**Key points**:
- Type annotations are required for all function arguments and local variables that TorchScript cannot infer
- The `self.num_timesteps: int` annotation ensures the for loop range is known
- In-place operations (`mem += cur`) should be replaced with functional equivalents (`mem = mem + cur`) for cleaner graph representation
- Surrogate gradient functions must be replaced with simple threshold comparisons

### Dual-Process Reasoner (Script -- Critical)

The dual-process reasoner is the most complex module to export because it contains data-dependent branching: System 1 runs always, and System 2 engages only when System 1 confidence is below a threshold.

```python
class DualProcessReasonerScriptable(nn.Module):
    """Scriptable dual-process reasoner with explicit control flow."""

    def __init__(self, original_reasoner):
        super().__init__()
        self.system1 = original_reasoner.system1
        self.system2 = original_reasoner.system2
        self.confidence_threshold: float = original_reasoner.confidence_threshold
        self.max_iterations: int = original_reasoner.max_iterations
        self.blend_gate = original_reasoner.blend_gate

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # System 1: fast, always runs
        s1_out = self.system1(x)
        confidence = self.compute_confidence(s1_out)

        # System 2: slow, conditional
        if confidence.mean() < self.confidence_threshold:
            s2_out = x
            for i in range(self.max_iterations):
                s2_out = self.system2(s2_out)

            # Blend System 1 and System 2
            gate = self.blend_gate(torch.cat([s1_out, s2_out], dim=-1))
            output = gate * s1_out + (1.0 - gate) * s2_out
        else:
            output = s1_out

        return output, confidence

    def compute_confidence(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x.norm(dim=-1, keepdim=True))

scripted_reasoner = torch.jit.script(DualProcessReasonerScriptable(model.reasoner))
```

**Critical considerations**:
- The `if confidence.mean() < self.confidence_threshold` is a data-dependent branch. Tracing would only capture one path. Scripting correctly handles both paths.
- The for loop in System 2 has a fixed iteration count (`self.max_iterations`), which scripting can unroll. If the original implementation uses a while loop with early stopping (confidence exceeds threshold mid-iteration), that pattern is also scriptable but produces a different graph.
- The `@torch.jit.export` decorator ensures the method is available in the serialized module.

### Global Workspace (Script)

The workspace competition mechanism selects which modality representations gain access to the global workspace. This involves:
- Computing attention scores across modalities
- Top-k selection (capacity limit)
- Broadcasting selected representations

```python
class GlobalWorkspaceScriptable(nn.Module):
    """Scriptable global workspace with competition."""

    def __init__(self, original_workspace):
        super().__init__()
        self.projection = original_workspace.projection
        self.attention = original_workspace.attention
        self.capacity_limit: int = original_workspace.capacity_limit

    @torch.jit.export
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        projected: List[torch.Tensor] = []
        for key in sorted(modality_features.keys()):
            feat = modality_features[key]
            proj = self.projection(feat)
            projected.append(proj)

        stacked = torch.stack(projected, dim=1)  # [B, num_modalities, D]

        # Competition via attention
        scores = self.attention(stacked)  # [B, num_modalities]

        # Select top-k (capacity limit)
        k = min(self.capacity_limit, stacked.shape[1])
        _, indices = torch.topk(scores, k, dim=1)

        # Gather selected
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, stacked.shape[-1])
        selected = torch.gather(stacked, 1, indices_expanded)

        # Integrate (mean pooling of selected)
        workspace = selected.mean(dim=1)
        return workspace
```

**Note**: The `Dict[str, torch.Tensor]` input type is supported in TorchScript. The keys must be sorted for deterministic behavior during tracing validation.

### Active Inference Agent (Script)

The active inference agent contains a planning loop that assesses policies over a planning horizon:

```python
class ActiveInferenceScriptable(nn.Module):
    def __init__(self, original_agent):
        super().__init__()
        self.transition_model = original_agent.transition_model
        self.policy_net = original_agent.policy_net
        self.planning_horizon: int = original_agent.planning_horizon

    @torch.jit.export
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        state = observation
        total_efe: torch.Tensor = torch.zeros(observation.shape[0], 1, device=observation.device)

        for t in range(self.planning_horizon):
            action = self.policy_net(state)
            next_state = self.transition_model(torch.cat([state, action], dim=-1))
            efe = self.compute_efe(state, next_state, action)
            total_efe = total_efe + efe
            state = next_state

        return self.policy_net(observation)

    def compute_efe(self, state: torch.Tensor, next_state: torch.Tensor,
                    action: torch.Tensor) -> torch.Tensor:
        return (next_state - state).pow(2).sum(dim=-1, keepdim=True)
```

## Handling the Complete BrainAI System

### Hybrid Approach: Trace + Script

The recommended approach for the full `BrainAI` system is a hybrid strategy:

1. **Trace** individual encoder modules (vision, text, audio, sensor)
2. **Script** modules with control flow (SNN, dual-process, workspace, active inference)
3. **Compose** the traced and scripted modules in a scriptable wrapper

```python
class BrainAIExportWrapper(nn.Module):
    """Wrapper that combines traced encoders with scripted core modules."""

    def __init__(self, brain_ai):
        super().__init__()
        # Trace encoders
        self.vision_encoder = torch.jit.trace(
            brain_ai.encoders['vision'],
            torch.randn(1, 1, 28, 28)
        )

        # Script complex modules
        self.workspace = torch.jit.script(
            GlobalWorkspaceScriptable(brain_ai.workspace)
        )
        self.reasoner = torch.jit.script(
            DualProcessReasonerScriptable(brain_ai.reasoner)
        )
        self.decision = brain_ai.decision_heads

    def forward(self, vision_input: torch.Tensor) -> torch.Tensor:
        encoded = self.vision_encoder(vision_input)
        features = {'vision': encoded}
        workspace = self.workspace(features)
        output, confidence = self.reasoner(workspace)
        return self.decision.classify(output)['logits']
```

### Full System Scripting

Alternatively, make the entire `BrainAI.forward` scriptable by adding type annotations and replacing unsupported patterns:

```python
@torch.jit.export
def forward_scriptable(
    self,
    vision_input: torch.Tensor,
    text_input: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scriptable forward pass with explicit typed inputs."""
    # ... (full implementation with type annotations)
```

This requires more refactoring but produces a single cohesive module.

## Serialization Format

### Saving

```python
# Save as TorchScript archive
scripted_model.save("brain_ai_scripted.pt")

# Save with extra files (metadata)
extra_files = {
    'config.json': json.dumps(config_dict),
    'version.txt': '1.0.0',
}
scripted_model.save("brain_ai_scripted.pt", _extra_files=extra_files)
```

### Loading

```python
# Python
loaded = torch.jit.load("brain_ai_scripted.pt")
output = loaded(sample_input)

# C++ (libtorch)
# torch::jit::script::Module module = torch::jit::load("brain_ai_scripted.pt");
# std::vector<torch::jit::IValue> inputs;
# inputs.push_back(torch::randn({1, 1, 28, 28}));
# auto output = module.forward(inputs);
```

### Archive Contents

A `.pt` TorchScript archive contains:
- `code/` -- Python-like source code for scripted modules
- `data/` -- Serialized tensor data (weights, buffers)
- `constants/` -- Compile-time constants
- `extra/` -- User-provided extra files

### File Size Considerations

BrainAI model sizes by configuration:
- Minimal (~1M params): ~5 MB
- 1B params: ~4 GB
- 3B params: ~12 GB
- 7B params: ~28 GB

For large models, consider:
- `torch.jit.save` uses ZIP compression by default
- External weight storage is not natively supported (unlike ONNX)
- Use `torch.package` for more flexible packaging of large models

## Control Flow Patterns in BrainAI

### Pattern 1: Feature Flag Branching

BrainAI uses feature flags (`use_snn`, `use_htm`, etc.) that determine which modules are active. These are static at export time.

```python
# Good: Branch on attribute that is constant at script time
if self.use_htm:
    x = self.htm(x)

# This works because self.use_htm is a bool attribute set in __init__
# TorchScript assesses it at compile time and includes only the active branch
```

### Pattern 2: Confidence-Based Routing

The dual-process system routes based on runtime confidence values:

```python
# This requires scripting (not tracing)
confidence = self.system1_confidence(x)
if confidence.item() > self.threshold:
    return self.fast_path(x)
else:
    return self.slow_path(x)
```

**Important**: `.item()` converts a tensor to a Python scalar, which TorchScript supports. For batch processing where different items may take different paths, use `torch.where()`:

```python
# Batch-compatible conditional
s1 = self.fast_path(x)
s2 = self.slow_path(x)
mask = (confidence > self.threshold).unsqueeze(-1)
return torch.where(mask, s1, s2)
```

### Pattern 3: Iterative Refinement

System 2 reasoning uses iterative GRU steps:

```python
# Fixed iteration count (works with both tracing and scripting)
for i in range(self.max_iterations):
    x = self.gru_step(x)

# Variable iteration count (scripting only)
iteration = 0
while confidence < self.threshold and iteration < self.max_iterations:
    x = self.gru_step(x)
    confidence = self.compute_confidence(x)
    iteration += 1
```

### Pattern 4: Dictionary Inputs

BrainAI takes `Dict[str, Tensor]` inputs. TorchScript supports `Dict[str, Tensor]` natively:

```python
@torch.jit.export
def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    features: List[torch.Tensor] = []
    if 'vision' in inputs:
        features.append(self.vision_encoder(inputs['vision']))
    if 'text' in inputs:
        features.append(self.text_encoder(inputs['text']))
    # ...
```

## Verification and Testing

### Round-Trip Verification

```python
def verify_torchscript(original_model, scripted_model, sample_inputs, n_tests=100):
    """Verify TorchScript model matches original across random inputs."""
    original_model.requires_grad_(False)
    results = []

    for i in range(n_tests):
        # Generate random input with same shape
        test_input = {k: torch.randn_like(v) for k, v in sample_inputs.items()}

        with torch.no_grad():
            original_out = original_model(test_input)
            scripted_out = scripted_model(test_input)

        max_diff = (original_out - scripted_out).abs().max().item()
        results.append(max_diff)

    return {
        'max_diff': max(results),
        'mean_diff': sum(results) / len(results),
        'all_pass': all(r < 1e-4 for r in results),
    }
```

### Graph Inspection

```python
# Print the TorchScript graph for debugging
print(scripted_model.graph)

# Print readable code
print(scripted_model.code)

# Check for unsupported operations
for node in scripted_model.graph.nodes():
    if 'prim::PythonOp' in str(node):
        print(f"WARNING: Python fallback detected: {node}")
```

### Performance Comparison

```python
import time

def benchmark(model, sample_input, n_warmup=10, n_iterations=100):
    """Benchmark model inference speed."""
    model.requires_grad_(False)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample_input)

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iterations):
            model(sample_input)
    elapsed = time.perf_counter() - start

    return {
        'total_time': elapsed,
        'avg_latency': elapsed / n_iterations,
        'throughput': n_iterations / elapsed,
    }
```

## Common Pitfalls and Solutions

**Pitfall: Tracing a module with data-dependent control flow**
Symptom: Model silently produces wrong results for some inputs.
Solution: Use `torch.jit.script` for modules with conditionals. Enable tracing warnings: `torch.jit.trace(..., check_trace=True, check_inputs=[input1, input2])`.

**Pitfall: Unsupported Python operations in scripting**
Symptom: `torch.jit.frontend.UnsupportedNodeError`.
Solution: Refactor the unsupported pattern. Common issues: `zip()` (use manual indexing), `enumerate()` (use range + indexing), complex dict comprehensions (use for loops), lambda functions (use named functions).

**Pitfall: Type inference failures**
Symptom: `RuntimeError: Expected a value of type 'Tensor' but found 'NoneType'`.
Solution: Add explicit type annotations. Use `Optional[Tensor]` where values might be None. Initialize variables with explicit types: `result: List[Tensor] = []`.

**Pitfall: Mutable default arguments**
Symptom: `ScriptModule has no attribute 'xxx'`.
Solution: Avoid mutable defaults in `__init__`. Use `field(default_factory=...)` or initialize in the constructor body.

**Pitfall: Non-deterministic operations**
Symptom: Traced model gives different results on re-run.
Solution: Set `torch.manual_seed()` before tracing. Avoid `torch.rand` in the forward pass (pass random values as inputs instead). Use deterministic algorithms: `torch.use_deterministic_algorithms(True)`.

**Pitfall: Large graph size with SNN unrolling**
Symptom: Serialized model file is very large; loading is slow.
Solution: Use scripting (which keeps the loop structure) instead of tracing (which unrolls loops). For SNN with 50 timesteps, scripting produces a compact loop while tracing produces 50x the operations.

## Optimization After Export

TorchScript provides built-in optimization passes:

```python
# Freeze the model (inline constants, remove training-only code)
frozen = torch.jit.freeze(scripted_model)

# Optimize for inference
optimized = torch.jit.optimize_for_inference(frozen)

# Save the optimized model
optimized.save("brain_ai_optimized.pt")
```

The `optimize_for_inference` pass:
- Fuses BatchNorm into preceding Conv/Linear layers
- Fuses activation functions with preceding operations
- Removes dropout (already handled by turning off training mode)
- Constant propagation
- Dead code elimination

For BrainAI specifically, these optimizations can reduce inference latency by 10-30% depending on the model configuration.
