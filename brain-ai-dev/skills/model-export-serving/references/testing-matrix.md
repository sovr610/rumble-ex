# Testing Matrix for Model Export and Serving

## Overview

Testing the export and serving infrastructure for BrainAI requires covering a wide matrix of model configurations, export formats, quantization strategies, input modalities, hardware targets, and runtime conditions. Unlike testing the model's training accuracy, export testing focuses on functional correctness (does the exported model produce the same outputs as the original?), format compliance (does the exported artifact pass structural validation?), performance (does the optimized model meet latency/throughput targets?), and operational reliability (does the serving infrastructure handle edge cases gracefully?). This reference defines the complete test matrix and the rationale behind each test dimension.

## Test Dimensions

### Dimension 1: Model Configuration

BrainAI supports multiple configurations via feature flags and scale presets. Each configuration produces a different computational graph that must be tested independently.

| Configuration | Feature Flags | Scale | Test Priority |
|--------------|---------------|-------|---------------|
| Minimal | All flags on | ~1M params | P0 (always test) |
| Vision-only | use_snn, use_htm off | ~1M params | P0 |
| Full cognitive | All flags on | ~1M params | P0 |
| SNN-disabled | use_snn=False | ~1M params | P1 |
| HTM-disabled | use_htm=False | ~1M params | P1 |
| Workspace-disabled | use_workspace=False | ~1M params | P1 |
| Symbolic-disabled | use_symbolic=False | ~1M params | P1 |
| Meta-disabled | use_meta=False | ~1M params | P1 |
| Engram-disabled | use_engram=False | ~1M params | P1 |
| Production 1B | All flags on | ~1B params | P2 (resource-gated) |
| Production 3B | All flags on | ~3B params | P2 |
| Production 7B | All flags on | ~7B params | P3 (nightly only) |

**Rationale**: Feature flags change the computational graph topology. A module being disabled means its subgraph is absent, which affects export tracing paths, TorchScript compilation, and quantization target enumeration. Each combination must be verified to not cause export errors.

### Dimension 2: Export Format

| Format | Variants | Test Focus |
|--------|----------|------------|
| TorchScript Trace | Single modality, multi-modal | Output equivalence, graph completeness |
| TorchScript Script | Full system, per-module | Control flow preservation, type correctness |
| TorchScript Hybrid | Traced encoders + scripted core | Composition correctness |
| ONNX Stateless | Opset 14, 17 | Structural validity, numerical accuracy |
| ONNX Stateful | Opset 17 | State I/O correctness, multi-step consistency |
| ONNX with custom ops | LIF neuron, spatial pooler | Custom op registration, runtime support |

### Dimension 3: Quantization Method

| Method | Dtype | Calibration | Test Focus |
|--------|-------|-------------|------------|
| None (FP32 baseline) | float32 | N/A | Reference accuracy |
| Dynamic INT8 | qint8 | N/A | Accuracy delta, speedup |
| Static INT8 | qint8 | MinMax observer | Accuracy delta, calibration coverage |
| Static INT8 | qint8 | Histogram observer | Accuracy delta, outlier handling |
| FP16 | float16 | N/A | GPU accuracy, overflow |
| BF16 | bfloat16 | N/A | Accuracy on supported hardware |
| QAT INT8 | qint8 | Training-integrated | Post-QAT accuracy, convergence |
| Mixed precision | mixed | Per-module | Composition correctness |

### Dimension 4: Input Modality

| Modality Combination | Input Shapes | Notes |
|---------------------|-------------|-------|
| Vision only | [B, 1, 28, 28] (MNIST) | Simplest, fastest test |
| Vision only | [B, 3, 224, 224] (ImageNet) | Production resolution |
| Text only | [B, 128] (token IDs) | Variable sequence length |
| Audio only | [B, 1, 16000] (1s @ 16kHz) | Time-series input |
| Sensor only | [B, 64] (flat features) | Simplest input shape |
| Vision + Text | Both shapes | Multi-modal workspace test |
| Vision + Text + Audio | Three shapes | Full multi-modal |
| All modalities | All shapes | Comprehensive test |

### Dimension 5: Batch Size

| Batch Size | Test Focus |
|-----------|------------|
| 1 | Minimum batch, no batch dimension issues |
| 2 | Smallest non-trivial batch |
| 8 | Typical serving batch |
| 32 | Large serving batch |
| 64 | Maximum typical batch |
| 128 | Stress test |

**Rationale**: Batch size affects dynamic axis handling, quantization statistics (per-tensor vs. per-batch), and memory usage. Edge cases at batch_size=1 are common sources of export bugs (squeeze/unsqueeze issues, batch norm behavior).

### Dimension 6: Hardware Target

| Target | Runtime | Test Focus |
|--------|---------|------------|
| CPU x86 | PyTorch, ONNX Runtime | Baseline correctness, quantized INT8 |
| CPU ARM | PyTorch (QNNPACK) | Mobile/edge deployment |
| GPU CUDA | PyTorch, ONNX Runtime CUDA | FP16/BF16 accuracy, GPU memory |
| GPU TensorRT | ONNX + TensorRT | Operator support, optimization |

## Test Categories

### Category 1: Export Correctness Tests

These tests verify that the exported model produces outputs numerically equivalent to the original PyTorch model.

**Test EC-001: TorchScript trace round-trip (vision)**
- Input: Minimal config, vision-only, batch_size=1
- Action: Trace model, run inference on 100 random inputs
- Assert: Max absolute difference < 1e-5 for all outputs

**Test EC-002: TorchScript trace round-trip (multi-modal)**
- Input: Minimal config, vision+text, batch_size=4
- Action: Trace model, run inference on 50 random inputs
- Assert: Max absolute difference < 1e-4 for all outputs

**Test EC-003: TorchScript script round-trip (full system)**
- Input: Minimal config, all flags on, batch_size=1
- Action: Script model, run inference on 100 random inputs
- Assert: Max absolute difference < 1e-4 for all outputs

**Test EC-004: TorchScript hybrid round-trip**
- Input: Minimal config, vision-only, batch_size=8
- Action: Trace encoders, script core, compose, run inference
- Assert: Max absolute difference < 1e-4

**Test EC-005: ONNX export structural validity**
- Input: Minimal config, vision-only
- Action: Export to ONNX, run onnx.checker.check_model()
- Assert: No structural errors (or graceful mock if onnx not installed)

**Test EC-006: ONNX numerical validation**
- Input: Minimal config, vision-only, batch_size=1
- Action: Export to ONNX, compare ORT output vs. PyTorch on 100 inputs
- Assert: Max absolute difference < 1e-4 (or skip if ORT not installed)

**Test EC-007: ONNX dynamic axes validation**
- Input: Export with dynamic batch axis
- Action: Run inference with batch_size=1, 4, 16, 32
- Assert: All batch sizes produce correct shapes and values

**Test EC-008: ONNX stateful mode**
- Input: Minimal config with SNN, stateful mode
- Action: Export with state inputs/outputs, run multi-step inference
- Assert: State propagation produces consistent results across steps

**Test EC-009: Feature flag export combinations**
- Input: Each feature flag disabled individually (6 configs)
- Action: Export each config via TorchScript trace
- Assert: All exports succeed, output shapes match expected

**Test EC-010: Save and reload**
- Input: TorchScript model
- Action: Save to file, reload, run inference
- Assert: Reloaded model produces identical outputs

### Category 2: Quantization Accuracy Tests

These tests verify that quantized models maintain acceptable accuracy.

**Test QA-001: Dynamic INT8 accuracy baseline**
- Input: Minimal config, FP32 trained model
- Action: Apply dynamic INT8 quantization
- Assert: Accuracy drop < 2% on dev benchmark

**Test QA-002: Static INT8 with MinMax observer**
- Input: Minimal config, 1000 calibration samples
- Action: Calibrate and convert to static INT8
- Assert: Accuracy drop < 2%

**Test QA-003: Static INT8 with Histogram observer**
- Input: Same as QA-002 but with HistogramObserver
- Assert: Accuracy drop < 1.5% (histogram should be more accurate)

**Test QA-004: FP16 conversion**
- Input: Minimal config, GPU required
- Action: Convert to FP16, run inference
- Assert: Accuracy drop < 0.5%

**Test QA-005: Per-module sensitivity measurement**
- Input: Minimal config
- Action: Quantize each module individually, measure accuracy
- Assert: Sensitivity rankings match expected order (neuromod > SNN > workspace > encoders)

**Test QA-006: Mixed-precision policy**
- Input: Minimal config with recommended skip list
- Action: Apply mixed INT8/FP32 quantization
- Assert: Accuracy drop < 1%

**Test QA-007: Quantization + export pipeline**
- Input: Minimal config
- Action: Quantize dynamic INT8, then export to TorchScript
- Assert: Exported quantized model runs correctly

**Test QA-008: Batch size invariance**
- Input: Quantized model (static INT8)
- Action: Run inference with batch_size=1, 8, 32
- Assert: Per-sample outputs are identical (or within tolerance) regardless of batch size

**Test QA-009: Calibration sample count sensitivity**
- Input: Static INT8 with 100, 500, 1000, 2000 calibration samples
- Assert: Accuracy stabilizes (diminishing returns beyond 500)

**Test QA-010: QAT convergence**
- Input: Minimal config, 5 QAT epochs
- Action: Run QAT fine-tuning, convert, measure accuracy
- Assert: Accuracy drop < 0.5%

### Category 3: Pruning Tests

**Test PR-001: Unstructured L1 pruning at 30% sparsity**
- Action: Apply unstructured pruning
- Assert: Model sparsity matches target, accuracy drop < 3%

**Test PR-002: Unstructured pruning at 50% sparsity**
- Assert: Model runs correctly, accuracy measured

**Test PR-003: Structured pruning (channel pruning)**
- Action: Remove 20% of channels from conv layers
- Assert: Model architecture is smaller, inference works

**Test PR-004: Pruning + quantization composition**
- Action: Prune 30%, then quantize to INT8
- Assert: Pipeline completes, model runs

**Test PR-005: Sparsity measurement accuracy**
- Action: Prune at known sparsity, measure
- Assert: Measured sparsity matches target within 1%

**Test PR-006: Iterative pruning**
- Action: Prune 10% over 3 rounds
- Assert: Final sparsity ~27% (cumulative), accuracy better than single 30% prune

### Category 4: Serving Infrastructure Tests

**Test SI-001: Health check endpoint**
- Action: Start engine (no actual server), call health_check()
- Assert: Returns healthy status with all fields populated

**Test SI-002: Single prediction**
- Action: Submit single PredictRequest
- Assert: Returns PredictResponse with correct fields, processing_time > 0

**Test SI-003: Batch prediction**
- Action: Submit 8 requests via batch_predict()
- Assert: Returns 8 responses, all valid

**Test SI-004: Empty request handling**
- Action: Submit request with no modality inputs
- Assert: Returns appropriate error (validation error or empty prediction)

**Test SI-005: Invalid input handling**
- Action: Submit request with wrong tensor shapes
- Assert: Returns validation error, does not crash

**Test SI-006: Concurrent request handling**
- Action: Submit 100 requests concurrently (simulated)
- Assert: All requests get responses, no deadlocks

**Test SI-007: Model reload**
- Action: Save model, reload via engine
- Assert: Predictions unchanged after reload

**Test SI-008: Metrics collection**
- Action: Run 10 predictions, check metrics
- Assert: Request count matches, latency histogram populated

**Test SI-009: Warmup verification**
- Action: Run warmup, measure latency before and after
- Assert: Post-warmup latency is lower than first inference

**Test SI-010: Request timeout**
- Action: Configure short timeout, submit request requiring System 2 reasoning
- Assert: Timeout handled gracefully

### Category 5: Performance Benchmark Tests

**Test PB-001: Baseline FP32 latency**
- Input: Minimal config, batch_size=1, CPU
- Assert: Latency < 100ms

**Test PB-002: INT8 speedup**
- Input: Dynamic INT8, batch_size=1, CPU
- Assert: Latency < 70ms (>1.3x speedup)

**Test PB-003: FP16 speedup**
- Input: FP16, batch_size=1, GPU
- Assert: Latency < 20ms

**Test PB-004: Throughput scaling**
- Input: Batch_size=1, 8, 32, 64
- Assert: Throughput increases with batch size (not necessarily linearly)

**Test PB-005: TorchScript optimization speedup**
- Input: Scripted + frozen + optimize_for_inference
- Assert: Latency < unoptimized scripted model

**Test PB-006: Model size measurements**
- Input: FP32, INT8, FP16 models
- Assert: INT8 ~4x smaller, FP16 ~2x smaller than FP32

**Test PB-007: Memory usage**
- Input: FP32, INT8, FP16 models
- Assert: INT8 uses less peak memory than FP32

### Category 6: Edge Case and Regression Tests

**Test RE-001: Zero input**
- Action: Pass all-zero tensors through exported model
- Assert: No NaN/Inf in output

**Test RE-002: Large input values**
- Action: Pass tensors with values in range [-100, 100]
- Assert: No NaN/Inf, output is reasonable

**Test RE-003: Negative input values**
- Action: Pass tensors with all-negative values
- Assert: Model handles correctly (ReLU zeroing, etc.)

**Test RE-004: Single-element batch**
- Action: batch_size=1 through all export formats
- Assert: Correct shapes and values

**Test RE-005: Maximum batch**
- Action: batch_size=128 through exported model
- Assert: No OOM on test hardware, correct values

**Test RE-006: Repeated inference determinism**
- Action: Run same input 10 times through exported model
- Assert: All outputs identical (no non-deterministic ops)

**Test RE-007: NaN input handling**
- Action: Pass input containing NaN values
- Assert: No crash (output may contain NaN but should not hang)

**Test RE-008: Empty sequence**
- Action: Text input with sequence_length=0
- Assert: Graceful error or empty output, no crash

## Test Execution Strategy

### Priority Levels

**P0 tests** (always run, < 2 minutes total):
- EC-001 through EC-004 (TorchScript round-trip)
- QA-001 (dynamic INT8 baseline)
- SI-001 through SI-005 (serving basics)
- RE-001 through RE-004 (critical edge cases)

**P1 tests** (run on PR merge, < 10 minutes):
- All P0 tests plus
- EC-005 through EC-010 (ONNX tests, feature flags)
- QA-002 through QA-008 (quantization variants)
- PR-001 through PR-006 (pruning)
- SI-006 through SI-010 (serving advanced)
- PB-001 through PB-003 (latency benchmarks)

**P2 tests** (nightly, < 60 minutes):
- All P1 tests plus
- QA-009, QA-010 (calibration sensitivity, QAT)
- PB-004 through PB-007 (full benchmarks)
- RE-005 through RE-008 (stress tests)
- Production-scale configs (1B, 3B)

**P3 tests** (weekly, resource-gated):
- Full 7B model export and serving
- Multi-GPU deployment testing
- Long-running stability tests (1 hour continuous serving)

### Test Infrastructure Requirements

| Requirement | P0 | P1 | P2 | P3 |
|------------|------|------|------|------|
| CPU | 2 cores | 4 cores | 8 cores | 32 cores |
| RAM | 2 GB | 4 GB | 16 GB | 64 GB |
| GPU | Not required | Optional | 1x GPU | 4x GPU |
| Disk | 1 GB | 2 GB | 10 GB | 100 GB |
| Time | 2 min | 10 min | 60 min | 4 hours |

### Test Data Requirements

**Random inputs**: Most tests use `torch.randn()` for input generation. Seed the random number generator with a fixed seed for reproducibility.

**Calibration data**: Static quantization tests require calibration data. Use a synthetic calibration set generated from a known distribution (Gaussian with class-conditional means).

**Benchmark data**: Performance tests should use a fixed dataset for consistent measurements. A subset of 1000 MNIST images serves as the standard dev benchmark.

### Mocking Strategy

Several components may not be available in all test environments:

- **ONNX Runtime**: Mock `onnxruntime.InferenceSession` when ORT is not installed. The mock returns pre-computed outputs for known inputs.
- **ONNX checker**: Mock `onnx.checker.check_model` to return success by default, allowing structural validation tests to verify the export code path without the onnx package.
- **GPU/CUDA**: Skip GPU-specific tests when `torch.cuda.is_available()` returns False.
- **FastAPI/uvicorn**: Test the engine logic directly without starting an HTTP server. Mock the ASGI framework for route testing.
- **Large models**: Use the minimal configuration for all correctness tests. Only test large configurations when explicitly enabled via environment variable.

## Test Reporting

### Metrics to Track Per Test Run

- Total tests executed
- Pass/fail/skip counts
- Per-test execution time
- Accuracy measurements (FP32 baseline, quantized accuracy, delta)
- Latency measurements (mean, p50, p95, p99)
- Model size measurements (FP32, quantized, pruned)
- Memory peak usage

### Regression Detection

Track key metrics across test runs to detect regressions:
- If accuracy delta for dynamic INT8 increases by >0.5% compared to the previous run, flag as regression
- If P50 latency increases by >20% compared to the previous run, flag as regression
- If model export fails for a configuration that previously succeeded, flag as regression

These metrics should be stored in a structured format (JSON or CSV) and compared against the historical baseline.
