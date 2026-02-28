---
name: Model Export & Serving
description: >
  This skill should be used when the user asks to "export the model",
  "convert to ONNX", "trace with TorchScript", "quantize the model",
  "deploy the model", "create API server", "serve predictions",
  "containerize the model", "optimize for inference", "export to INT8",
  "create FastAPI endpoint", "set up model serving", "prune the model",
  "distill knowledge", "create Docker image", or needs guidance on
  model export formats, quantization strategies, serving infrastructure,
  or deployment pipelines for the brain_ai system.
version: 0.1.0
---

# Model Export & Serving

## Overview

Guide implementation of model export, optimization, and serving infrastructure. The brain_ai system has unique export challenges: SNN dynamics with temporal state, HTM's sparse representations, conditional dual-process routing, and engram memory lookups. Cover ONNX export, TorchScript tracing, quantization (PTQ and QAT), pruning, knowledge distillation, FastAPI serving, and containerization.

## Public Contract

### ONNXExporter

Export BrainAI models to ONNX format with dynamic axes and operator validation.

```python
class ONNXExporter:
    def __init__(self, model: BrainAI, config: ExportConfig): ...
    def export(self, output_path: str, sample_input: Dict[str, Tensor]) -> ONNXExportResult: ...
    def validate(self, output_path: str, sample_input: Dict[str, Tensor]) -> ValidationResult: ...
    def get_unsupported_ops(self) -> List[str]: ...
```

### TorchScriptExporter

Trace or script BrainAI for C++ deployment.

```python
class TorchScriptExporter:
    def __init__(self, model: BrainAI, config: ExportConfig): ...
    def trace(self, sample_input: Dict[str, Tensor]) -> torch.jit.ScriptModule: ...
    def script(self) -> torch.jit.ScriptModule: ...
    def save(self, module: torch.jit.ScriptModule, path: str) -> None: ...
```

### ModelQuantizer

Post-training and quantization-aware training for INT8/FP16 inference.

```python
class ModelQuantizer:
    def __init__(self, model: BrainAI, config: QuantizationConfig): ...
    def quantize_dynamic(self) -> nn.Module: ...       # Dynamic INT8
    def quantize_static(self, calibration_loader: DataLoader) -> nn.Module: ...
    def quantize_aware_training(self, train_fn: Callable) -> nn.Module: ...
    def measure_accuracy_delta(self, eval_fn: Callable) -> Dict[str, float]: ...
```

### ModelPruner

Structured and unstructured pruning with iterative magnitude pruning.

```python
class ModelPruner:
    def __init__(self, model: BrainAI, config: PruningConfig): ...
    def prune_unstructured(self, sparsity: float) -> nn.Module: ...
    def prune_structured(self, sparsity: float) -> nn.Module: ...
    def measure_sparsity(self) -> Dict[str, float]: ...
```

### ServingEngine

FastAPI-based model serving with batching and health checks.

```python
class ServingEngine:
    def __init__(self, model_path: str, config: ServingConfig): ...
    def create_app(self) -> FastAPI: ...
    def predict(self, request: PredictRequest) -> PredictResponse: ...
    def batch_predict(self, requests: List[PredictRequest]) -> List[PredictResponse]: ...
    def health_check(self) -> HealthStatus: ...
```

## Key Concepts

### Export Compatibility Matrix

| Feature | ONNX | TorchScript | Notes |
|---------|------|-------------|-------|
| SNN dynamics | Partial | Full | ONNX: unroll time steps statically |
| HTM sparse ops | No | Partial | Custom ops needed for SDR |
| Dual-process routing | No | Full | Dynamic control flow requires scripting |
| Engram memory | No | Partial | Hash lookups need custom handling |
| Feature flags | N/A | Full | Export per-configuration |

### Quantization Strategy

- **Dynamic INT8**: Weights quantized statically, activations dynamically. Good baseline, ~1.5-2× speedup.
- **Static INT8**: Both quantized with calibration data. Best CPU speedup (~2-3×), requires representative calibration set.
- **FP16**: Half precision on GPU. ~2× memory reduction, ~1.5× speedup. Use with AMP-trained models.
- **QAT**: Fine-tune with fake quantization nodes. Best accuracy preservation.

### SNN Export Challenges

SNN neurons maintain temporal state (membrane potential). For export:
- **Stateless mode**: Unroll T time steps into a single forward, reset state per sample
- **Stateful mode**: Accept and return state tensors as additional inputs/outputs
- Surrogate gradient functions must be replaced with forward-only equivalents

### Serving Architecture

```
Client → FastAPI → Request Queue → Batch Assembler → Model → Response
                        ↓
                  Health Monitor → Metrics (Prometheus)
```

## Configuration Surface

```python
@dataclass
class ExportConfig:
    format: str = "torchscript"          # onnx | torchscript | both
    opset_version: int = 17              # ONNX opset
    dynamic_axes: bool = True            # Dynamic batch dimension
    snn_mode: str = "stateless"          # stateless | stateful
    validate_export: bool = True
    tolerance: float = 1e-4

@dataclass
class QuantizationConfig:
    method: str = "dynamic"              # dynamic | static | qat
    dtype: str = "int8"                  # int8 | fp16
    calibration_samples: int = 1000
    accuracy_threshold: float = 0.02     # Max acceptable accuracy drop

@dataclass
class ServingConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 32
    batch_timeout_ms: int = 50
    num_workers: int = 1
    device: str = "auto"
    model_path: str = "model.pt"
```

## Done-When Gates

1. **Export Round-Trip** — TorchScript traced model produces outputs matching original within tolerance (1e-4) on 100 random inputs. ONNX validates with `onnx.checker.check_model()`.
2. **Quantization Accuracy** — INT8 dynamic quantized model accuracy drops <2% on dev benchmark relative to FP32 baseline.
3. **Serving Latency** — FastAPI server handles single requests in <100ms (CPU) or <20ms (GPU) for minimal-config model; health check returns within 1s.

## Failure Modes

| Mode | Symptom | Fix |
|------|---------|-----|
| Unsupported ONNX op | Export fails | Check get_unsupported_ops(), use TorchScript instead |
| Tracing misses control flow | Wrong results | Use torch.jit.script for conditional modules |
| Quantization accuracy collapse | >5% drop | Use QAT, or quantize fewer layers |
| Serve OOM | Crash under load | Reduce max_batch_size, enable model streaming |
| State mismatch in SNN export | Temporal artifacts | Verify snn_mode matches inference pattern |

## Anti-Patterns

- Exporting without validation — always compare outputs pre/post export
- Quantizing all layers uniformly — skip sensitive layers (first/last, attention)
- Hardcoding input shapes — use dynamic axes for variable batch/sequence
- Serving without health checks — always implement /health endpoint
- Ignoring warmup — first inference is slow; run warmup requests at startup

## Resources

### Reference Files
- **`references/onnx-export.md`** — ONNX export guide, operator coverage, dynamic axes
- **`references/torchscript-tracing.md`** — Tracing vs scripting, control flow handling
- **`references/quantization-strategies.md`** — PTQ, QAT, calibration, accuracy preservation
- **`references/serving-architecture.md`** — FastAPI design, batching, health checks, Docker
- **`references/testing-matrix.md`** — Test scenarios for export and serving infrastructure

### Asset Files
- **`assets/onnx_exporter_template.py`** — ONNXExporter with validation, self-tests
- **`assets/torchscript_exporter_template.py`** — TorchScriptExporter with trace/script modes
- **`assets/quantizer_template.py`** — ModelQuantizer with PTQ, QAT, accuracy measurement
- **`assets/pruner_template.py`** — ModelPruner with structured/unstructured pruning
- **`assets/serving_template.py`** — ServingEngine with FastAPI, batching, health checks
- **`assets/export_config_template.py`** — All config dataclasses + validation

### Scripts
- **`scripts/validate_export.py`** — Validates export infrastructure against done-when gates
- **`scripts/gen_export_tests.py`** — Generates 100+ pytest test cases for export modules
- **`scripts/export_benchmark.py`** — Latency and throughput benchmarks for exported models
