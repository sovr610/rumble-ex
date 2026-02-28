# Serving Architecture for BrainAI

## Overview

Serving BrainAI models in production requires a purpose-built infrastructure that handles the unique characteristics of the brain-inspired architecture: variable-latency inference from dual-process routing (System 1 fast path vs. System 2 slow path), temporal state management for SNN streaming inference, multi-modal input preprocessing, and the need for health monitoring that captures cognitive-layer-specific metrics. This reference covers the FastAPI-based serving design, dynamic batching, health checks, Prometheus metrics, Docker containerization, and deployment patterns.

## FastAPI Application Design

### Core Application Structure

The serving application is built on FastAPI for its async support, automatic OpenAPI documentation, and Pydantic request validation. The application follows a layered architecture:

```
HTTP Layer (FastAPI routes)
    -> Validation Layer (Pydantic models)
        -> Preprocessing Layer (input normalization, tokenization)
            -> Batching Layer (dynamic batch assembly)
                -> Inference Layer (model forward pass)
                    -> Postprocessing Layer (output formatting)
```

### Request and Response Models

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class PredictRequest(BaseModel):
    """Single prediction request."""
    vision_input: Optional[List[List[List[float]]]] = None   # [C, H, W]
    text_input: Optional[str] = None
    audio_input: Optional[List[List[float]]] = None           # [C, T]
    sensor_input: Optional[List[float]] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    return_confidence: bool = True
    return_workspace_state: bool = False

class PredictResponse(BaseModel):
    """Single prediction response."""
    request_id: str
    predictions: List[float]
    predicted_class: Optional[int] = None
    confidence: Optional[float] = None
    processing_time_ms: float
    system_path: Optional[str] = None      # "system1" or "system2"
    workspace_state: Optional[Dict[str, float]] = None

class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    requests: List[PredictRequest]

class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    responses: List[PredictResponse]
    total_time_ms: float
    batch_size: int

class HealthStatus(BaseModel):
    """Health check response."""
    status: str                    # "healthy", "degraded", "unhealthy"
    model_loaded: bool
    device: str
    uptime_seconds: float
    total_requests: int
    avg_latency_ms: float
    last_error: Optional[str] = None
    components: Dict[str, str] = {}
```

### Route Definitions

```python
app = FastAPI(title="BrainAI Serving", version="1.0.0")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Single prediction endpoint."""
    ...

@app.post("/batch_predict", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest):
    """Batch prediction endpoint."""
    ...

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health and readiness check."""
    ...

@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    ...

@app.post("/reload")
async def reload_model(model_path: str):
    """Hot-reload model without downtime."""
    ...
```

### Startup and Shutdown Lifecycle

```python
@app.on_event("startup")
async def startup():
    """Initialize model and resources on server start."""
    # Load model
    engine.load_model(config.model_path)

    # Warmup: run dummy inference to trigger JIT compilation
    warmup_input = engine.create_warmup_input()
    for _ in range(config.warmup_iterations):
        engine.predict(warmup_input)

    # Initialize metrics collector
    metrics_collector.reset()

    # Start batch processing background task
    asyncio.create_task(engine.batch_processor_loop())

@app.on_event("shutdown")
async def shutdown():
    """Clean shutdown with request draining."""
    # Signal no new requests
    engine.accepting_requests = False

    # Wait for pending requests to complete (max 30s)
    await engine.drain(timeout=30.0)

    # Release model resources
    engine.unload_model()
```

## Dynamic Batching

### Why Dynamic Batching

Individual inference requests are inefficient on modern hardware (especially GPUs) because they underutilize parallel compute units. Dynamic batching collects individual requests over a short time window and processes them together, improving throughput without requiring clients to implement batching logic.

For BrainAI, batching is particularly important because:
- The workspace attention mechanism is more efficient with larger batch sizes (better GPU utilization)
- SNN timestep processing amortizes loop overhead across batch items
- The dual-process router can make batch-level routing decisions

### Batching Implementation

The batch assembler uses an async queue with a timeout-based flushing strategy:

```python
class DynamicBatcher:
    def __init__(self, max_batch_size: int, timeout_ms: int, process_fn):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.process_fn = process_fn
        self.queue = asyncio.Queue()

    async def submit(self, request):
        """Submit a request and wait for the result."""
        future = asyncio.Future()
        await self.queue.put((request, future))
        return await future

    async def batch_processor_loop(self):
        """Background loop that assembles and processes batches."""
        while True:
            batch = []
            futures = []

            # Collect first item (blocking wait)
            request, future = await self.queue.get()
            batch.append(request)
            futures.append(future)

            # Collect more items up to max_batch_size or timeout
            deadline = time.monotonic() + self.timeout_ms / 1000.0
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    request, future = await asyncio.wait_for(
                        self.queue.get(), timeout=remaining
                    )
                    batch.append(request)
                    futures.append(future)
                except asyncio.TimeoutError:
                    break

            # Process the batch
            try:
                results = await asyncio.get_event_loop().run_in_executor(
                    None, self.process_fn, batch
                )
                for future, result in zip(futures, results):
                    future.set_result(result)
            except Exception as e:
                for future in futures:
                    future.set_exception(e)
```

### Batching Tradeoffs

**max_batch_size**: Larger batches improve throughput but increase latency for early arrivals (they wait for the batch to fill). For BrainAI:
- CPU serving: max_batch_size=8-16 (limited parallelism)
- GPU serving: max_batch_size=32-64 (good GPU utilization)
- Minimal config: max_batch_size=64-128 (small model, GPU memory permits)

**timeout_ms**: The maximum time to wait before processing an incomplete batch. This is the upper bound on added latency from batching:
- Low latency requirement (<50ms): timeout_ms=10
- Balanced: timeout_ms=50
- Throughput optimized: timeout_ms=200

### Handling Variable-Size Inputs

BrainAI accepts multi-modal inputs where different requests may have different modalities active. The batcher must handle this:

**Strategy 1: Pad and mask**. Pad all inputs to the maximum size in the batch and use attention masks to ignore padding. This is the standard approach for text (variable sequence length) and works well for vision (if images have different sizes after preprocessing).

**Strategy 2: Group by modality**. Separate requests into groups based on which modalities are present, and batch within each group. This avoids padding overhead but reduces batch sizes. Useful when the modality distribution is skewed (e.g., 90% vision-only requests).

**Strategy 3: Fixed preprocessing**. Require all inputs to be preprocessed to fixed sizes (224x224 for vision, 512 tokens for text). This simplifies batching at the cost of client-side preprocessing requirements.

For BrainAI serving, Strategy 1 is recommended as the default, with Strategy 3 as an optimization for production deployments with known input characteristics.

## Health Checks

### Three-Level Health Model

**Level 1: Liveness** (`/health/live`)
- Is the process running and responding?
- Checks: HTTP response within 1 second
- Used by: container orchestrator (Kubernetes liveness probe)
- Failure action: restart container

**Level 2: Readiness** (`/health/ready`)
- Is the model loaded and ready to serve?
- Checks: model loaded, warmup complete, no OOM errors
- Used by: load balancer (Kubernetes readiness probe)
- Failure action: remove from load balancer pool

**Level 3: Deep health** (`/health/deep`)
- Is the model producing correct results?
- Checks: run a known input through the model and verify the output matches expected value (canary inference)
- Used by: monitoring systems, periodic health assessment
- Failure action: alert, investigate, potentially reload model

### BrainAI-Specific Health Metrics

Beyond standard health checks, BrainAI's cognitive architecture provides unique health signals:

**System 2 engagement rate**: If System 2 is engaging on >50% of production requests (up from a baseline of ~20%), the model may be encountering distribution shift or the confidence threshold may need recalibration. This metric is tracked per time window (1 minute, 5 minutes, 1 hour).

**Workspace competition entropy**: The global workspace selects modality representations via attention-based competition. If the competition entropy drops (one modality always wins), it may indicate an input pipeline issue (e.g., one modality always producing zeros due to a preprocessing bug).

**SNN spike rate**: The average spike rate across SNN neurons should be in a healthy range (5-30% for typical inputs). Spike rates near 0% (silent network) or near 100% (saturated) indicate model corruption or extreme input distribution shift.

**Neuromodulator levels**: The four neuromodulators (DA, ACh, NE, 5-HT) should remain within their typical operating ranges. Extreme values may indicate model instability.

```python
class BrainAIHealthMonitor:
    def __init__(self, engine):
        self.engine = engine
        self.canary_input = self._create_canary()
        self.canary_expected = None  # Set during warmup

    def check_deep_health(self):
        result = self.engine.predict(self.canary_input)
        output_diff = abs(result.confidence - self.canary_expected)

        status = "healthy"
        issues = []

        if output_diff > 0.01:
            status = "degraded"
            issues.append(f"Canary output drift: {output_diff:.4f}")

        if self.engine.metrics.system2_rate > 0.5:
            status = "degraded"
            issues.append(f"High System 2 rate: {self.engine.metrics.system2_rate:.2f}")

        return HealthStatus(
            status=status,
            model_loaded=True,
            device=str(self.engine.device),
            components={
                'canary_check': 'pass' if output_diff < 0.01 else 'fail',
                'system2_rate': f'{self.engine.metrics.system2_rate:.2f}',
            }
        )
```

## Prometheus Metrics

### Metric Types

The serving infrastructure exposes metrics in Prometheus format for monitoring and alerting:

**Counters** (monotonically increasing):
- `brainai_requests_total{endpoint, status}` -- total requests by endpoint and HTTP status
- `brainai_predictions_total{system_path}` -- predictions by System 1/2 path
- `brainai_errors_total{type}` -- errors by type (validation, inference, timeout)

**Histograms** (distribution tracking):
- `brainai_request_duration_seconds{endpoint}` -- request latency distribution
- `brainai_batch_size` -- actual batch sizes used
- `brainai_inference_duration_seconds` -- pure model inference time (excluding pre/post processing)
- `brainai_preprocessing_duration_seconds` -- input preprocessing time

**Gauges** (current value):
- `brainai_model_loaded` -- 1 if model is loaded, 0 otherwise
- `brainai_queue_depth` -- current number of requests waiting in batch queue
- `brainai_gpu_memory_used_bytes` -- GPU memory usage
- `brainai_system2_engagement_rate` -- rolling System 2 engagement rate
- `brainai_snn_spike_rate` -- rolling average SNN spike rate

### Metric Implementation

```python
class MetricsCollector:
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.latency_sum = 0.0
        self.latency_count = 0
        self.latency_buckets = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        self.latency_histogram = [0] * (len(self.latency_buckets) + 1)

    def record_request(self, latency_seconds, status, system_path=None):
        self.request_count += 1
        self.latency_sum += latency_seconds
        self.latency_count += 1

        # Update histogram
        for i, bucket in enumerate(self.latency_buckets):
            if latency_seconds <= bucket:
                self.latency_histogram[i] += 1
                break
        else:
            self.latency_histogram[-1] += 1

    def format_prometheus(self):
        """Format metrics in Prometheus exposition format."""
        lines = []
        lines.append(f'# HELP brainai_requests_total Total requests')
        lines.append(f'# TYPE brainai_requests_total counter')
        lines.append(f'brainai_requests_total {self.request_count}')

        lines.append(f'# HELP brainai_request_duration_seconds Request latency')
        lines.append(f'# TYPE brainai_request_duration_seconds histogram')
        cumulative = 0
        for i, bucket in enumerate(self.latency_buckets):
            cumulative += self.latency_histogram[i]
            lines.append(
                f'brainai_request_duration_seconds_bucket{{le="{bucket}"}} {cumulative}'
            )
        cumulative += self.latency_histogram[-1]
        lines.append(f'brainai_request_duration_seconds_bucket{{le="+Inf"}} {cumulative}')
        lines.append(f'brainai_request_duration_seconds_sum {self.latency_sum}')
        lines.append(f'brainai_request_duration_seconds_count {self.latency_count}')

        return '\n'.join(lines)
```

### Alerting Rules

Recommended Prometheus alerting rules for BrainAI serving:

- **High latency**: `histogram_quantile(0.99, brainai_request_duration_seconds) > 1.0` -- 99th percentile latency exceeds 1 second
- **Error rate**: `rate(brainai_errors_total[5m]) / rate(brainai_requests_total[5m]) > 0.01` -- error rate exceeds 1%
- **Queue buildup**: `brainai_queue_depth > 100` -- batch queue depth exceeds 100 (requests backing up)
- **System 2 spike**: `brainai_system2_engagement_rate > 0.5` -- System 2 engaging on majority of requests (possible distribution shift)
- **Model not loaded**: `brainai_model_loaded == 0` -- model failed to load or was unloaded

## Docker Containerization

### Dockerfile Structure

```dockerfile
# Multi-stage build for minimal production image
# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Production image
FROM python:3.11-slim

# Non-root user for security
RUN useradd -m -s /bin/bash brainai
WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code
COPY brain_ai/ brain_ai/
COPY scripts/serve.py .

# Copy model (or mount at runtime)
# COPY model.pt /models/model.pt

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"

# Environment
ENV BRAINAI_MODEL_PATH=/models/model.pt
ENV BRAINAI_DEVICE=auto
ENV BRAINAI_PORT=8000
ENV BRAINAI_WORKERS=1

USER brainai
EXPOSE 8000

CMD ["python", "serve.py"]
```

### GPU Support

For GPU-enabled containers, use the NVIDIA CUDA base image:

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and PyTorch with CUDA support
RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Set the NVIDIA runtime in docker-compose:

```yaml
services:
  brainai:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Docker Compose for Full Stack

```yaml
version: "3.8"

services:
  brainai:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models:ro
    environment:
      - BRAINAI_MODEL_PATH=/models/model.pt
      - BRAINAI_DEVICE=cpu
      - BRAINAI_MAX_BATCH_SIZE=32
    healthcheck:
      test: ["CMD", "python", "-c",
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"]
      interval: 30s
      timeout: 5s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards:ro
```

### Resource Limits

Set appropriate resource limits based on the model configuration:

| Model Size | CPU Limit | Memory Limit | GPU Memory |
|-----------|-----------|-------------|------------|
| Minimal (~1M) | 2 cores | 1 GB | 512 MB |
| 1B params | 8 cores | 8 GB | 6 GB |
| 3B params | 16 cores | 24 GB | 16 GB |
| 7B params | 32 cores | 56 GB | 40 GB |

## Deployment Patterns

### Single-Model Serving

The simplest pattern: one container runs one model. Suitable for development and low-traffic production deployments.

### Model Versioning with Blue-Green Deployment

Run two model versions simultaneously (blue and green). Route traffic to the active version. When deploying a new model:
1. Load new model into the inactive (green) deployment
2. Run health checks and canary inference on green
3. Switch traffic from blue to green
4. Keep blue running for rollback if needed

### A/B Testing

Route a percentage of traffic to different model versions for comparison:
- Version A: current production model (e.g., FP32)
- Version B: optimized model (e.g., INT8 quantized)

Compare accuracy, latency, and user-facing metrics to validate the optimization.

### Scaling Patterns

**Horizontal scaling**: Run multiple serving containers behind a load balancer. Each container loads the full model. Works well for CPU serving where models fit in RAM.

**GPU sharing**: For GPU serving, multiple containers can share a GPU using NVIDIA MPS (Multi-Process Service) or time-slicing. This is useful when the model does not fully utilize the GPU.

**Model parallelism**: For large models (3B+) that do not fit on a single GPU, split the model across multiple GPUs. The serving framework routes requests to the appropriate GPU for each pipeline stage. This requires custom routing logic in the serving layer.

## Performance Optimization

### Warmup Strategy

The first inference after model loading is significantly slower due to JIT compilation, CUDA kernel caching, and memory allocation. Implement a warmup phase:

```python
async def warmup(engine, config):
    """Run warmup inferences before accepting traffic."""
    warmup_input = engine.create_warmup_input()

    for i in range(config.warmup_iterations):
        engine.predict(warmup_input)

    # For GPU: also warmup with different batch sizes
    if engine.device.type == 'cuda':
        for batch_size in [1, 4, 16, config.max_batch_size]:
            batch_input = engine.create_warmup_input(batch_size=batch_size)
            engine.predict(batch_input)
```

### Request Timeout and Circuit Breaker

Protect the server from hanging requests and cascading failures:

```python
@app.middleware("http")
async def timeout_middleware(request, call_next):
    try:
        response = await asyncio.wait_for(
            call_next(request),
            timeout=config.request_timeout_seconds,
        )
        return response
    except asyncio.TimeoutError:
        metrics.record_error("timeout")
        return JSONResponse(
            status_code=504,
            content={"error": "Request timed out"},
        )
```

### Memory Management

For long-running serving processes, memory management is critical:

- **Periodic garbage collection**: Force `gc.collect()` and `torch.cuda.empty_cache()` during low-traffic periods
- **Request-scoped tensors**: Ensure intermediate tensors from inference are freed after each request
- **Model pinning**: Pin model parameters in memory to prevent paging
- **Input validation**: Reject oversized inputs (e.g., images >10MB, sequences >10K tokens) before they consume memory
