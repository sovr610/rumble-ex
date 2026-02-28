# ONNX Export for BrainAI

## Overview

ONNX (Open Neural Network Exchange) provides an interoperable format for deploying BrainAI models across different runtimes (ONNX Runtime, TensorRT, OpenVINO, CoreML). However, the brain-inspired architecture presents unique challenges: SNN temporal dynamics with membrane potential state, HTM sparse distributed representations, conditional dual-process routing, and engram memory hash lookups. This reference covers operator coverage, dynamic axes, opset requirements, custom op registration, validation strategies, and stateless vs. stateful export modes.

## Operator Coverage for BrainAI Modules

### Standard Operators (Full Support)

The following BrainAI components export cleanly to ONNX using standard operators available in opset 17+:

**Linear layers and MLPs**: All projection layers in encoders, workspace, decision heads, and reasoning modules use `nn.Linear`, which maps directly to ONNX `Gemm` or `MatMul` + `Add` operators. No issues arise here.

**Convolution layers**: Vision encoder CNN stages (`nn.Conv2d`) and audio encoder mel-spectrogram processing export as ONNX `Conv` operators. Batch normalization layers export as `BatchNormalization`. These are among the most mature ONNX operators and cause no problems.

**Attention mechanisms**: Multi-head attention in the Global Workspace and text encoder uses `torch.nn.MultiheadAttention`. Starting with opset 14, scaled dot-product attention is supported. For opset 17, the newer `torch.nn.functional.scaled_dot_product_attention` can be exported directly. The recommendation is to use opset 17 for best attention support.

**Activation functions**: ReLU, GELU, SiLU (Swish), Sigmoid, Tanh -- all have direct ONNX operator equivalents and export without modification.

**Normalization**: LayerNorm, BatchNorm, GroupNorm all have opset 17 support. RMSNorm (if used in text encoder) can be decomposed into standard ops by the exporter.

**Pooling**: AdaptiveAvgPool2d, MaxPool2d, and global pooling operations export as standard ONNX pool operators.

### Partially Supported Operators

**Softmax with temperature**: The dual-process routing and workspace competition use softmax with learned temperature scaling. The softmax itself exports fine, but if temperature is a learnable parameter that changes per-forward-pass, it must be captured as a constant at export time or passed as an input.

**Einsum**: Some attention variants use `torch.einsum`. ONNX supports Einsum from opset 12+, but complex einsum expressions may need decomposition for certain runtimes.

**Complex indexing**: Advanced tensor indexing (gather with computed indices) in engram memory and HTM sparse lookups may require careful handling. ONNX `Gather`, `GatherElements`, and `ScatterND` cover most cases but can be slow on certain backends.

### Unsupported / Problematic Operators

**SNN Surrogate Gradients**: The surrogate gradient functions (`ATanSurrogate`, `FastSigmoidSurrogate`, `StraightThroughEstimator`) use custom autograd functions with separate forward/backward behavior. ONNX only captures the forward graph. The solution is to replace surrogate functions with their forward-only equivalents before export:

```python
# Pre-export: replace surrogate with Heaviside step
def replace_surrogates_for_export(model):
    """Replace surrogate gradient functions with forward-only step functions."""
    for module in model.modules():
        if hasattr(module, 'surrogate'):
            module.surrogate = HeavisideStep()  # Standard step function
    return model
```

**HTM Sparse Distributed Representations (SDR)**: htm.core's native SDR operations (spatial pooler, temporal memory) are C++ extensions not representable in ONNX. When using the LSTM fallback (which is the default when htm.core is unavailable), export works fine. For native HTM, the solution is to use the LSTM fallback path during export.

**Dynamic control flow in System 2 reasoning**: The `DualProcessReasoner` uses conditional execution -- System 1 (fast path) runs always, but System 2 (slow path with iterative GRU) only engages when confidence falls below a threshold. ONNX does not support dynamic `if` statements well. Solutions:
1. Export only System 1 path (fast inference)
2. Always execute both paths and blend with a mask
3. Use a fixed number of System 2 iterations (unroll the loop)

**Engram hash table lookups**: The engram memory uses hash-based N-gram lookups (`torch.remainder`, computed indices into large embedding tables). While the individual operations are ONNX-compatible, the pattern of dynamic index computation followed by gather can be inefficient and sometimes fails validation.

**In-place operations**: SNN membrane potential updates often use in-place operations (`mem += input`). ONNX requires functional (non-in-place) computation. The exporter usually handles this automatically, but in rare cases it may fail. Pre-converting to functional style resolves this.

## Dynamic Axes Configuration

Dynamic axes are critical for serving BrainAI models with variable batch sizes and, for text/audio modalities, variable sequence lengths.

### Recommended Dynamic Axes

```python
dynamic_axes = {
    # Vision input: variable batch, fixed spatial
    'vision_input': {0: 'batch_size'},

    # Text input: variable batch and sequence length
    'text_input': {0: 'batch_size', 1: 'seq_length'},

    # Audio input: variable batch and time
    'audio_input': {0: 'batch_size', 2: 'audio_length'},

    # Sensor input: variable batch
    'sensor_input': {0: 'batch_size'},

    # Output: variable batch
    'output': {0: 'batch_size'},
    'confidence': {0: 'batch_size'},
}
```

### SNN Timestep Handling

The SNN core processes inputs over T timesteps. For ONNX export, there are two strategies:

**Static unrolling (recommended for ONNX)**: Unroll all T timesteps into the graph. The timestep dimension becomes a fixed part of the model. This produces a larger graph but is fully compatible with all ONNX runtimes.

```python
# In SNNCore, replace the timestep loop with static unrolling
class SNNCoreForExport(nn.Module):
    def __init__(self, original_snn, num_timesteps):
        super().__init__()
        self.snn = original_snn
        self.T = num_timesteps

    def forward(self, x):
        # x: [batch, features]
        batch_size = x.shape[0]
        mem = torch.zeros(batch_size, self.snn.hidden_size, device=x.device)
        spk_rec = []
        for t in range(self.T):  # Unrolled at trace time
            spk, mem = self.snn.lif_step(x, mem)
            spk_rec.append(spk)
        return torch.stack(spk_rec, dim=1).mean(dim=1)
```

**Dynamic timesteps as input**: Pass T as an input tensor. This requires ONNX Loop operator (opset 13+) which has limited runtime support and is generally not recommended.

## Opset Requirements

### Minimum: Opset 14

Opset 14 provides:
- `Trilu` for attention masks
- Improved `Reshape` semantics
- `HardSwish` activation

### Recommended: Opset 17

Opset 17 adds:
- `LayerNormalization` as a native op (better performance)
- `GroupNormalization`
- Improved `Pad` operator
- Better support for `scaled_dot_product_attention`

### Future: Opset 18+

Opset 18 introduces:
- `BitwiseAnd`, `BitwiseOr` (useful for SDR operations if HTM is exported)
- `CenterCropPad`
- Improved shape inference

**Recommendation**: Use opset 17 for all BrainAI exports. This provides the best balance of operator coverage, runtime support, and performance.

## Custom Op Registration

For BrainAI components that cannot be expressed with standard ONNX operators, custom op registration provides an escape hatch.

### SNN LIF Neuron Custom Op

```python
from torch.onnx import register_custom_op_symbolic

def lif_neuron_symbolic(g, input, mem, beta, threshold):
    """Custom ONNX symbolic for LIF neuron step."""
    # Compute new membrane potential
    decay = g.op("Mul", mem, beta)
    new_mem = g.op("Add", decay, input)

    # Spike generation (Heaviside step)
    spike = g.op("Greater", new_mem, threshold)
    spike_float = g.op("Cast", spike, to_i=1)  # FLOAT

    # Reset: subtract threshold on spike
    reset = g.op("Mul", spike_float, threshold)
    final_mem = g.op("Sub", new_mem, reset)

    return spike_float, final_mem

register_custom_op_symbolic('brain_ai::lif_step', lif_neuron_symbolic, opset_version=17)
```

### HTM Spatial Pooler Approximation

For the HTM spatial pooler, register a custom op that approximates the top-k sparse activation:

```python
def spatial_pooler_symbolic(g, input, k):
    """Approximate HTM spatial pooler as top-k activation."""
    values, indices = g.op("TopK", input, k, outputs=2)
    # Create sparse output
    zeros = g.op("ConstantOfShape", g.op("Shape", input))
    output = g.op("ScatterElements", zeros, indices, values, axis_i=1)
    return output
```

### Registration Strategy

Register all custom ops before calling `torch.onnx.export`:

```python
def register_brain_ai_custom_ops():
    """Register all BrainAI custom ONNX operators."""
    register_custom_op_symbolic('brain_ai::lif_step', lif_neuron_symbolic, 17)
    register_custom_op_symbolic('brain_ai::spatial_pooler', spatial_pooler_symbolic, 17)
    # Add more as needed
```

## Validation with ONNX Runtime

After export, rigorous validation ensures the ONNX model produces outputs matching the original PyTorch model.

### Three-Stage Validation Pipeline

**Stage 1: Structural validation** -- Use `onnx.checker.check_model()` to verify the ONNX graph is well-formed, all operators are defined, and tensor shapes are consistent.

```python
import onnx

model = onnx.load("brain_ai.onnx")
onnx.checker.check_model(model)  # Raises on structural errors
print(f"Graph has {len(model.graph.node)} nodes")
print(f"Opset version: {model.opset_import[0].version}")
```

**Stage 2: Shape inference** -- Run shape inference to verify all intermediate tensor shapes can be determined.

```python
from onnx import shape_inference
inferred_model = shape_inference.infer_shapes(model)
# Check that all value_info have shapes
for vi in inferred_model.graph.value_info:
    shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    assert all(s > 0 or s == 0 for s in shape), f"Unknown shape for {vi.name}"
```

**Stage 3: Numerical validation** -- Compare ONNX Runtime outputs against PyTorch outputs on multiple random inputs.

```python
import onnxruntime as ort
import numpy as np

def validate_numerical(pytorch_model, onnx_path, sample_inputs, tolerance=1e-4):
    """Compare PyTorch and ONNX Runtime outputs."""
    session = ort.InferenceSession(onnx_path)

    pytorch_model.set_to_inference_mode()
    with torch.no_grad():
        pt_output = pytorch_model(sample_inputs)

    # Convert inputs for ORT
    ort_inputs = {}
    for name, tensor in sample_inputs.items():
        ort_inputs[name + '_input'] = tensor.cpu().numpy()

    ort_output = session.run(None, ort_inputs)

    # Compare
    max_diff = np.max(np.abs(pt_output.cpu().numpy() - ort_output[0]))
    mean_diff = np.mean(np.abs(pt_output.cpu().numpy() - ort_output[0]))

    return {
        'max_diff': float(max_diff),
        'mean_diff': float(mean_diff),
        'within_tolerance': max_diff < tolerance,
    }
```

### Validation Test Matrix

Run validation across multiple configurations:

| Configuration | Batch Size | Sequence Length | Expected Tolerance |
|--------------|------------|-----------------|-------------------|
| Vision only | 1 | N/A | 1e-5 |
| Vision only | 32 | N/A | 1e-4 |
| Text only | 1 | 128 | 1e-4 |
| Text only | 16 | 512 | 1e-4 |
| Multi-modal | 1 | 128 | 1e-3 |
| Multi-modal | 8 | 256 | 1e-3 |

Note: Multi-modal configurations may have slightly larger numerical differences due to the workspace competition mechanism.

## Stateless vs. Stateful Export Modes

### Stateless Mode

In stateless mode, the SNN processes inputs without carrying state between calls. This is appropriate for:
- Classification tasks where each input is independent
- Batch processing where temporal context is not needed
- Simpler deployment with no state management

```python
class BrainAIStatelessWrapper(nn.Module):
    """Wraps BrainAI for stateless ONNX export."""

    def __init__(self, brain_ai):
        super().__init__()
        self.brain_ai = brain_ai

    def forward(self, vision_input):
        self.brain_ai.reset_state()  # Reset all temporal state
        return self.brain_ai({'vision': vision_input})
```

Export command:
```python
torch.onnx.export(
    stateless_wrapper,
    (sample_vision_input,),
    "brain_ai_stateless.onnx",
    input_names=['vision_input'],
    output_names=['output'],
    dynamic_axes={'vision_input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=17,
)
```

### Stateful Mode

In stateful mode, SNN membrane potentials and HTM temporal state are passed as additional inputs and outputs. This enables:
- Streaming inference (audio, video)
- Temporal reasoning that builds over time
- Recurrent processing of sequential data

```python
class BrainAIStatefulWrapper(nn.Module):
    """Wraps BrainAI for stateful ONNX export with explicit state I/O."""

    def __init__(self, brain_ai):
        super().__init__()
        self.brain_ai = brain_ai

    def forward(self, vision_input, snn_state, htm_state):
        # Inject state
        self.brain_ai.set_snn_state(snn_state)
        self.brain_ai.set_htm_state(htm_state)

        output = self.brain_ai({'vision': vision_input})

        # Extract updated state
        new_snn_state = self.brain_ai.get_snn_state()
        new_htm_state = self.brain_ai.get_htm_state()

        return output, new_snn_state, new_htm_state
```

### Choosing Between Modes

| Criterion | Stateless | Stateful |
|-----------|-----------|----------|
| Deployment simplicity | Better | More complex |
| Temporal reasoning | No | Yes |
| Batch processing | Better throughput | Per-sequence tracking |
| ONNX compatibility | Higher | Requires careful design |
| Memory usage | Lower | Higher (state buffers) |

**Recommendation**: Start with stateless mode for initial deployment. Only move to stateful when temporal continuity is required by the application.

## Export Pipeline Summary

The complete ONNX export pipeline for BrainAI:

1. **Prepare the model**: Set to inference mode, replace surrogate gradients, force LSTM fallback for HTM, optionally disable System 2 reasoning.

2. **Configure export**: Select opset version (17), define dynamic axes, choose stateless or stateful mode.

3. **Register custom ops**: If any non-standard operations remain, register their ONNX symbolic functions.

4. **Export**: Call `torch.onnx.export` with appropriate parameters.

5. **Validate structure**: Run `onnx.checker.check_model()` and shape inference.

6. **Validate numerics**: Compare against PyTorch outputs on 100+ random inputs with tolerance checks.

7. **Optimize**: Run `onnxruntime.transformers.optimizer` for graph optimizations (constant folding, operator fusion).

8. **Benchmark**: Measure inference latency and throughput with ONNX Runtime.

## Common Issues and Solutions

**Issue: "Exporting the operator X is not supported"**
Solution: Check the operator mapping table at the PyTorch ONNX supported ops documentation. If not supported, decompose into supported ops or register a custom symbolic.

**Issue: "TracerWarning: Converting a tensor to a Python boolean"**
Solution: This indicates dynamic control flow (if/else on tensor values). Refactor to use `torch.where()` for conditional operations, or export only the path that does not use dynamic control flow.

**Issue: Large model size after export**
Solution: Use external data format for models > 2GB: `torch.onnx.export(..., use_external_data_format=True)`. This stores weights in separate files alongside the `.onnx` protobuf.

**Issue: "ShapeInferenceError" during validation**
Solution: Ensure all tensor shapes are deterministic at export time. Dynamic shapes should only vary along explicitly declared dynamic axes. Check for operations that produce shape-dependent outputs (like `nonzero()`).

**Issue: Numerical drift > tolerance**
Solution: Check for operations with non-deterministic ordering (like `set` operations in workspace competition). Force deterministic behavior during export validation. Consider increasing tolerance for modules with inherent numerical sensitivity (attention softmax with small values).

## Performance Considerations

ONNX Runtime provides several execution providers for optimized inference:

- **CPUExecutionProvider**: Default, good baseline performance
- **CUDAExecutionProvider**: GPU acceleration, best for large models
- **TensorrtExecutionProvider**: NVIDIA TensorRT optimization, best GPU latency
- **OpenVINOExecutionProvider**: Intel hardware optimization

For BrainAI specifically:
- Vision encoder benefits most from GPU/TensorRT acceleration
- SNN core with unrolled timesteps creates a deep sequential graph -- TensorRT may struggle with very deep graphs
- Workspace attention is compute-bound and benefits from GPU
- Decision heads are small and run efficiently on CPU

The recommendation is to use `CUDAExecutionProvider` for GPU deployments and consider TensorRT only for the encoder subgraphs where it excels.
