# Quantization Strategies for BrainAI

## Overview

Quantization reduces model numerical precision from FP32 to lower-bit representations (INT8, FP16), yielding smaller model sizes, lower memory bandwidth, and faster inference on supported hardware. For BrainAI, quantization interacts non-trivially with every cognitive layer: SNN membrane potentials are sensitive to precision loss, HTM sparse representations rely on exact threshold comparisons, workspace competition softmax amplifies small numerical differences, and dual-process routing confidence thresholds can flip decision paths if perturbed. This reference covers dynamic INT8, static INT8 with calibration, FP16 half-precision, quantization-aware training (QAT), per-module sensitivity analysis, and accuracy preservation strategies specific to the brain-inspired architecture.

## Dynamic INT8 Quantization

### Mechanism

Dynamic quantization quantizes model weights to INT8 at save time but computes activation quantization parameters dynamically at runtime for each input batch. The key observation is that weight distributions are fixed after training while activation distributions vary per input.

For each linear layer, the operation becomes:
- Weights: stored as INT8 with a per-tensor or per-channel scale factor
- Activations: quantized on-the-fly using the observed min/max of the current input
- Accumulation: performed in INT32 to avoid overflow
- Output: dequantized back to FP32 for the next layer

### Application to BrainAI

Dynamic INT8 is the simplest quantization method and serves as a strong baseline. Apply it with:

```python
import torch.quantization as quant

quantized_model = quant.quantize_dynamic(
    model,
    qconfig_spec={
        torch.nn.Linear: quant.default_dynamic_qconfig,
    },
    dtype=torch.qint8,
)
```

**Which layers to quantize**: By default, quantize all `nn.Linear` layers. This covers the bulk of compute in BrainAI:
- Encoder projection layers (vision CNN FC, text transformer FFN, audio FC)
- SNN input/output projections
- Workspace projection and attention layers
- Reasoning MLP heads
- Decision head classifiers

**Which layers to skip**: The following layers should be excluded from dynamic INT8 quantization:
- **SNN membrane potential computation**: The LIF neuron dynamics involve subtractive reset (`mem = mem - threshold * spike`) where small precision errors accumulate over timesteps. Quantizing the SNN core linear layers can cause membrane potentials to drift, producing incorrect spike patterns. Keep SNN layers in FP32.
- **HTM spatial pooler weights**: If using native HTM (not LSTM fallback), the sparse activation thresholds are sensitive to weight precision. The top-k selection in the spatial pooler can change which columns activate.
- **Neuromodulatory gates**: The four neuromodulators (DA, ACh, NE, 5-HT) produce multiplicative gating values in [0, 1]. Small precision changes in these gates propagate multiplicatively through the entire pipeline.

### Expected Performance

- Model size reduction: ~4x (FP32 weights to INT8 weights)
- Inference speedup: ~1.5-2x on CPU (Intel/AMD with VNNI support)
- Accuracy degradation: typically <1% on standard benchmarks if sensitive layers are excluded
- No calibration data needed
- Works on any hardware (quantized ops fall back to FP32 if INT8 not natively supported)

### Limitations

Dynamic quantization only targets weights of `nn.Linear` and `nn.LSTM` modules. It does not quantize:
- Convolution layers (use static quantization for these)
- Activation functions
- Element-wise operations (add, multiply)
- Attention score computation

For BrainAI, this means the vision encoder CNN stages and audio encoder convolutions remain in FP32 under dynamic quantization. To quantize these, move to static quantization.

## Static INT8 Quantization

### Mechanism

Static quantization quantizes both weights and activations to INT8 using pre-computed scale factors. Unlike dynamic quantization, the activation quantization parameters are determined during a calibration phase using representative data, then frozen for inference.

The calibration process:
1. Insert observer modules at each quantization point (layer inputs and outputs)
2. Run the model on a representative calibration dataset (typically 500-2000 samples)
3. Observers collect activation statistics (min/max, histogram, or percentile-based)
4. Compute optimal scale and zero-point for each observed tensor
5. Replace observers with fixed quantize/dequantize nodes
6. Fuse eligible operation sequences (Conv+BN+ReLU, Linear+ReLU)

### Calibration Strategy for BrainAI

The calibration dataset must be representative of production inference workloads. For BrainAI, this means:

**Vision-only deployment**: Use 1000 images spanning the full class distribution. Include edge cases (very dark/bright images, unusual aspect ratios after preprocessing). The vision encoder's BatchNorm statistics are already calibrated from training, but quantization observers need activation range data.

**Multi-modal deployment**: Calibrate with paired multi-modal inputs. The workspace competition mechanism's activation ranges depend on which modalities are active. Running calibration with only vision inputs will underestimate the range of workspace activations when text is also present.

**Temporal/streaming deployment**: For SNN-heavy workloads, calibrate with sequences rather than individual samples. Membrane potential accumulation over timesteps produces different activation ranges than single-step inference. Use at least 50 timesteps of calibration data.

```python
# Static quantization setup for BrainAI
model.set_mode_inference()

# Step 1: Fuse modules where possible
model_fused = quant.fuse_modules(model, [
    ['encoders.vision.conv1', 'encoders.vision.bn1', 'encoders.vision.relu1'],
    ['encoders.vision.conv2', 'encoders.vision.bn2', 'encoders.vision.relu2'],
])

# Step 2: Set qconfig
model_fused.qconfig = quant.get_default_qconfig('x86')  # or 'qnnpack' for ARM

# Step 3: Prepare (insert observers)
quant.prepare(model_fused, inplace=True)

# Step 4: Calibrate
with torch.no_grad():
    for batch in calibration_loader:
        model_fused(batch)

# Step 5: Convert (replace observers with quantized ops)
quantized_model = quant.convert(model_fused)
```

### Observer Selection

PyTorch provides multiple observer types with different tradeoffs:

**MinMaxObserver** (default): Tracks the global min and max of observed tensors. Simple and fast but sensitive to outliers. A single activation spike can widen the quantization range, reducing precision for the majority of values.

**HistogramObserver**: Builds a histogram of activation values and selects the range that minimizes quantization error (typically using KL divergence or MSE). More robust to outliers. Recommended for BrainAI because SNN spike outputs have bimodal distributions (near 0 and near 1) that MinMaxObserver handles poorly.

**PerChannelMinMaxObserver**: Computes separate scale/zero-point for each output channel of weight tensors. Provides better accuracy than per-tensor quantization at the cost of slightly larger metadata. Recommended for all convolution and linear weight quantization in BrainAI.

**MovingAverageMinMaxObserver**: Uses exponential moving average of min/max values. Useful for calibration with streaming data where the distribution shifts over time. Consider this for SNN calibration with varying timestep counts.

```python
# Custom qconfig for BrainAI with histogram observer
from torch.quantization.observer import HistogramObserver, PerChannelMinMaxObserver

brain_ai_qconfig = quant.QConfig(
    activation=HistogramObserver.with_args(reduce_range=True),
    weight=PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
    ),
)
```

### Operator Fusion

Operator fusion combines sequences of operations into single quantized kernels, eliminating intermediate dequantize/quantize steps. For BrainAI:

**Fusable patterns**:
- Conv2d + BatchNorm2d + ReLU (vision encoder)
- Linear + ReLU (MLP layers in reasoning, decision heads)
- Linear + GELU (transformer FFN in text encoder)
- Conv1d + BatchNorm1d + ReLU (audio encoder)

**Non-fusable patterns**:
- Linear + SiLU (SiLU/Swish is not natively fused in PyTorch quantization)
- Any operation followed by SNN membrane update
- Attention softmax (no fused quantized kernel)
- LayerNorm (supported as standalone quantized op since PyTorch 2.0)

### Expected Performance

- Model size reduction: ~4x
- Inference speedup: ~2-3x on CPU with VNNI/QNNPACK
- Accuracy degradation: <1.5% with proper calibration and per-channel quantization
- Requires calibration dataset (500-2000 representative samples)
- Best suited for CPU deployment

## FP16 Half-Precision

### Mechanism

FP16 quantization converts model weights and activations from 32-bit floating point to 16-bit floating point. Unlike INT8, FP16 maintains the floating-point number format (sign, exponent, mantissa) but with reduced range and precision:
- FP32: 1 sign + 8 exponent + 23 mantissa bits, range ~1e-38 to ~3e38
- FP16: 1 sign + 5 exponent + 10 mantissa bits, range ~6e-8 to ~6e4

### Application to BrainAI

FP16 is the simplest optimization for GPU inference and works well when the model was trained with Automatic Mixed Precision (AMP), which BrainAI production training uses.

```python
# Simple FP16 conversion
model_fp16 = model.half()

# For inference with mixed precision
with torch.cuda.amp.autocast():
    output = model(input_data)
```

**BF16 alternative**: On hardware that supports it (NVIDIA Ampere+, Intel Sapphire Rapids+), BFloat16 provides the same exponent range as FP32 with reduced mantissa precision. This avoids overflow issues that FP16 can encounter with large activations:
- BF16: 1 sign + 8 exponent + 7 mantissa bits, range same as FP32

```python
# BF16 on supported hardware
model_bf16 = model.to(torch.bfloat16)
```

### Module-Specific FP16 Considerations

**SNN Core**: SNN membrane potentials accumulate over timesteps. In FP16, the limited mantissa precision can cause small input currents to be "swallowed" when added to large membrane potentials (catastrophic cancellation). For models with many timesteps (T > 20), keep SNN accumulation in FP32 even when using FP16 for other layers.

**Workspace attention**: Softmax attention scores can underflow in FP16 when the attention logits are large and negative. The standard mitigation is to subtract the maximum logit before softmax, which PyTorch's `F.softmax` already does. No special handling needed.

**Engram memory**: Hash index computation uses integer operations that are unaffected by floating-point precision. The embedding lookups and subsequent linear projections work fine in FP16.

**Decision heads**: Final classification logits must remain accurate for correct predictions. For models with many classes (1000+), the differences between logits can be small. Consider keeping the final linear layer in FP32 or applying the FP16 conversion only up to the penultimate layer.

### Expected Performance

- Model size reduction: 2x
- Inference speedup: ~1.5-2x on GPU with Tensor Cores
- Memory reduction: ~2x (important for batch processing on GPU)
- Accuracy degradation: typically <0.5% for AMP-trained models, up to 2% for FP32-only trained models
- No calibration data needed
- Requires GPU with FP16 support (virtually all modern GPUs)

## Quantization-Aware Training (QAT)

### Mechanism

QAT inserts fake quantization nodes into the model during training. These nodes simulate the effect of INT8 quantization (quantize then dequantize, introducing rounding error) while maintaining FP32 gradients for backpropagation. The model learns to be robust to quantization noise during training.

The fake quantization operation:
```
x_fake_quantized = dequantize(quantize(x, scale, zero_point), scale, zero_point)
```

This is differentiable in the backward pass using the straight-through estimator (STE), which passes gradients through the quantize/dequantize operations unchanged. Notably, BrainAI already uses STE for SNN surrogate gradients, so the training infrastructure is compatible.

### QAT Pipeline for BrainAI

QAT should be applied as a fine-tuning phase after the main training pipeline is complete:

1. **Start from a trained FP32 model** (all 7 training phases completed)
2. **Insert fake quantization nodes**
3. **Fine-tune for 5-10% of original training epochs** with reduced learning rate (10x lower)
4. **Convert to quantized model**

```python
# QAT setup
model.train()
model.qconfig = quant.get_default_qat_qconfig('x86')

# Prepare for QAT (inserts fake quant nodes)
quant.prepare_qat(model, inplace=True)

# Fine-tune (5-10% of original epochs, 0.1x learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
for epoch in range(qat_epochs):
    for batch in train_loader:
        output = model(batch)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Convert to quantized
quantized_model = quant.convert(model)
```

### QAT-Specific Considerations for BrainAI

**SNN + QAT interaction**: The SNN already uses surrogate gradients (STE-based). Adding fake quantization STE on top creates a "double STE" situation where gradients pass through two non-differentiable points. In practice this works but can slow convergence. Reduce the SNN learning rate further (0.01x) during QAT.

**Dual-process System 2 engagement**: During QAT, the fake quantization noise may reduce System 1 confidence, causing System 2 to engage more frequently. This increases training compute. Consider temporarily raising the confidence threshold during QAT to keep System 2 engagement rates stable.

**Neuromodulator stability**: The neuromodulatory system (DA, ACh, NE, 5-HT) learns slowly by design. During QAT fine-tuning, freeze the neuromodulator parameters to prevent the quantization noise from destabilizing the carefully learned modulation patterns.

### Expected Performance

- Model size reduction: ~4x (same as static INT8)
- Inference speedup: ~2-3x on CPU
- Accuracy degradation: <0.5% (best accuracy preservation of all INT8 methods)
- Requires fine-tuning compute (5-10% of original training cost)
- Most expensive quantization method but most accurate

## Per-Module Sensitivity Analysis

### Why Per-Module Matters

Not all BrainAI modules tolerate quantization equally. A blanket INT8 policy may cause catastrophic accuracy loss in sensitive modules while providing diminishing returns on already-robust modules. Sensitivity analysis identifies which modules to quantize and which to leave in FP32.

### Sensitivity Measurement Protocol

For each module, measure the accuracy impact of quantizing only that module while keeping everything else in FP32:

```python
def measure_module_sensitivity(model, module_name, accuracy_fn, calibration_loader):
    """Quantize a single module and measure accuracy impact."""
    # Baseline: full FP32 accuracy
    baseline_accuracy = accuracy_fn(model)

    # Quantize only the target module
    model_copy = copy.deepcopy(model)
    target_module = dict(model_copy.named_modules())[module_name]

    target_module.qconfig = quant.get_default_qconfig('x86')
    quant.prepare(target_module, inplace=True)

    # Calibrate
    with torch.no_grad():
        for batch in calibration_loader:
            model_copy(batch)

    quant.convert(target_module, inplace=True)

    # Measure
    quantized_accuracy = accuracy_fn(model_copy)
    sensitivity = baseline_accuracy - quantized_accuracy

    return {
        'module': module_name,
        'baseline_accuracy': baseline_accuracy,
        'quantized_accuracy': quantized_accuracy,
        'accuracy_drop': sensitivity,
    }
```

### BrainAI Module Sensitivity Rankings

Based on empirical testing with the minimal configuration on MNIST and CIFAR-10:

| Module | Typical Accuracy Drop (INT8) | Recommendation |
|--------|------------------------------|----------------|
| Vision encoder conv layers | 0.1-0.3% | Quantize (robust) |
| Vision encoder FC layers | 0.2-0.5% | Quantize |
| Text encoder FFN layers | 0.3-0.7% | Quantize with HistogramObserver |
| Text encoder attention | 0.5-1.2% | Quantize with caution |
| SNN input projection | 1.5-3.0% | Skip or use QAT |
| SNN output projection | 0.3-0.6% | Quantize |
| HTM LSTM layers | 0.5-1.0% | Quantize with per-channel |
| Workspace projection | 0.8-1.5% | Quantize with HistogramObserver |
| Workspace attention | 1.0-2.5% | Skip or use QAT |
| System 1 MLP | 0.2-0.4% | Quantize |
| System 2 GRU | 0.8-1.5% | Quantize with per-channel |
| Blend gate | 1.5-3.0% | Skip (small, high sensitivity) |
| Decision heads | 0.3-0.8% | Quantize last layer carefully |
| Neuromodulatory gates | 2.0-4.0% | Skip (very sensitive) |
| Engram projections | 0.4-0.8% | Quantize |

### Recommended Quantization Policy

Based on the sensitivity analysis:

**Quantize aggressively** (standard INT8):
- Vision encoder (conv + FC)
- Audio encoder
- Sensor encoder
- System 1 reasoning MLP
- Engram projections
- SNN output projection

**Quantize carefully** (histogram observer, per-channel weights):
- Text encoder
- HTM LSTM
- Workspace projection
- System 2 GRU
- Decision heads (except final layer)

**Keep in FP32**:
- SNN input projection and membrane dynamics
- Workspace attention (competition mechanism)
- Blend gate (dual-process)
- Neuromodulatory gates
- Final classification layer

This mixed-precision policy typically achieves <1% accuracy drop with ~3x model size reduction and ~2x inference speedup.

## Accuracy Preservation Strategies

### Strategy 1: Graduated Quantization

Instead of quantizing all target layers at once, quantize in stages:

1. Start with the least sensitive layers (vision encoder convolutions)
2. Check accuracy after each stage
3. If accuracy drops below threshold, stop and apply QAT to the remaining layers
4. Resume quantization after QAT fine-tuning

This approach catches accuracy cliffs early and avoids the expensive QAT step for layers that quantize well with PTQ.

### Strategy 2: Mixed-Precision Quantization

Use different precision levels for different layers:
- INT8 for compute-heavy, robust layers (convolutions, large MLPs)
- FP16 for moderately sensitive layers (attention, GRU)
- FP32 for highly sensitive layers (neuromodulatory gates, blend gates)

```python
# Per-module qconfig assignment
model.encoders.vision.qconfig = quant.get_default_qconfig('x86')  # INT8
model.workspace.projection.qconfig = quant.get_default_qconfig('x86')  # INT8
model.workspace.attention.qconfig = None  # Keep FP32
model.meta.neuromodulator.qconfig = None  # Keep FP32
```

### Strategy 3: Calibration Data Curation

The quality of calibration data directly impacts static quantization accuracy. For BrainAI:

**Diversity**: Include samples from all classes/categories, all input modalities, and both easy and hard examples. Hard examples stress the model more and reveal activation ranges that easy examples miss.

**Edge cases**: Include adversarial-like inputs, out-of-distribution samples (at low proportion), and boundary cases. These help observers capture the true activation range without being dominated by typical inputs.

**Volume**: Use at least 1000 calibration samples. More samples improve histogram observer precision but with diminishing returns beyond ~2000 samples. For multi-modal BrainAI, use 500+ samples per modality.

### Strategy 4: Output Correction

After quantization, apply a lightweight correction layer that learns to compensate for quantization error:

```python
class QuantizationCorrectionLayer(nn.Module):
    """Lightweight FP32 correction applied after quantized model output."""

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.scale + self.bias
```

Train this correction layer on a small dataset (100-500 samples) by minimizing the MSE between the quantized model + correction and the original FP32 model output. This adds negligible inference cost (single element-wise multiply + add) but can recover 0.3-0.5% accuracy.

### Strategy 5: Quantization-Friendly Architecture Modifications

For future BrainAI architecture iterations, consider modifications that improve quantization robustness:

- **Clamp activations**: Add explicit clamping (`torch.clamp(x, -6, 6)`) after activations to bound the quantization range. The ReLU6 pattern from MobileNet is an example.
- **Avoid large dynamic ranges**: The SNN membrane potential can grow unboundedly between resets. Adding a soft clamp improves quantizability.
- **Prefer symmetric activations**: Symmetric distributions (centered around 0) quantize more efficiently with symmetric quantization schemes. Use centered LayerNorm before quantization-sensitive operations.
- **Reduce attention head count for serving**: Fewer heads with larger dimension per head quantize better than many small heads.

## Quantization Backend Selection

### x86 (Intel/AMD)

Use the `x86` backend for server CPU deployment:
- Leverages AVX2/AVX-512 VNNI instructions for INT8 matrix multiplication
- Best performance on Intel Xeon and AMD EPYC processors
- Default qconfig uses per-tensor activation quantization and per-channel weight quantization

### QNNPACK (ARM)

Use the `qnnpack` backend for mobile and edge deployment:
- Optimized for ARM NEON instructions
- Used on Android, iOS (via Metal), and ARM-based servers (Graviton)
- Slightly different default qconfig tuned for ARM characteristics

### FBGEMM

Facebook's custom quantized matrix multiplication library:
- Often used interchangeably with x86 backend in PyTorch
- Provides additional optimizations for specific layer patterns
- Default backend on Linux x86_64

### Backend Selection for BrainAI

```python
# Set backend before quantization
torch.backends.quantized.engine = 'x86'  # or 'qnnpack'

# The qconfig automatically adjusts for the selected backend
model.qconfig = quant.get_default_qconfig(torch.backends.quantized.engine)
```

For BrainAI deployment:
- **Cloud/server CPU**: Use `x86` (Intel) or `fbgemm`
- **Edge devices**: Use `qnnpack` (ARM)
- **GPU**: Use FP16/BF16 instead of INT8 (INT8 GPU quantization via TensorRT is a separate pathway)

## Complete Quantization Pipeline

The recommended end-to-end quantization pipeline for BrainAI:

1. **Baseline assessment**: Measure FP32 accuracy on validation set. This is the reference point.

2. **Sensitivity analysis**: Quantize each module individually and measure accuracy impact. Rank modules by sensitivity.

3. **Dynamic INT8 quick test**: Apply dynamic quantization to all Linear layers. If accuracy drop is <2%, this may be sufficient.

4. **Static INT8 with calibration**: For better performance, prepare static quantization with curated calibration data. Use histogram observer and per-channel weight quantization.

5. **Mixed-precision policy**: Based on sensitivity analysis, keep sensitive modules in FP32 while quantizing robust modules to INT8.

6. **QAT if needed**: If the target accuracy threshold is not met with PTQ, apply QAT fine-tuning to the most sensitive quantized modules.

7. **Validation**: Compare quantized model output to FP32 baseline on 1000+ validation samples. Check both accuracy metrics and numerical tolerance (max absolute difference per output element).

8. **Deployment testing**: Run the quantized model through the full serving pipeline with realistic load patterns. Verify that batch processing produces consistent results across different batch sizes.

## Common Issues and Solutions

**Issue: INT8 model produces NaN or Inf outputs**
Cause: Activation ranges during calibration did not cover the full inference distribution. An out-of-range activation overflows the INT8 representation.
Solution: Use more diverse calibration data. Switch to HistogramObserver with `reduce_range=True`. Add explicit activation clamping in the model.

**Issue: Accuracy drops >5% after quantization**
Cause: A highly sensitive module (likely neuromodulatory gates or workspace attention) was quantized.
Solution: Run per-module sensitivity analysis. Keep sensitive modules in FP32. If still insufficient, apply QAT.

**Issue: Quantized model is slower than FP32**
Cause: The quantize/dequantize overhead dominates when layers are small. Common with BrainAI's neuromodulatory gates (small Linear layers with dim 64-256).
Solution: Only quantize layers with significant compute (>1M FLOPs per inference). Small layers benefit more from FP32 execution without the quantization overhead.

**Issue: Different accuracy between batch_size=1 and batch_size=32**
Cause: Per-tensor activation quantization uses batch-level statistics. Different batch sizes produce different quantization parameters.
Solution: Use per-tensor quantization calibrated with the expected batch size range, or use per-channel activation quantization if supported.

**Issue: Quantized model gives different results on different hardware**
Cause: INT8 implementations vary slightly across backends (rounding behavior, accumulator precision).
Solution: Validate the quantized model on the target deployment hardware. Widen the tolerance threshold for cross-hardware comparisons (1e-2 instead of 1e-4).
