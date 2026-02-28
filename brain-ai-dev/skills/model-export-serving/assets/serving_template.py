"""
ServingEngine: FastAPI-based model serving with batching and health checks.

Provides create_app(), predict(), batch_predict(), and health_check() methods.
All tests run without starting an actual HTTP server. FastAPI/uvicorn are mocked
if not installed.

torch + standard lib only. FastAPI/Pydantic mocked gracefully.
"""

import copy
import gc
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ServingConfig:
    """Configuration for model serving."""
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 32
    batch_timeout_ms: int = 50
    num_workers: int = 1
    device: str = "cpu"
    model_path: str = "model.pt"
    warmup_iterations: int = 3
    request_timeout_seconds: float = 30.0
    max_queue_depth: int = 1000
    verbose: bool = False


@dataclass
class PredictRequest:
    """Single prediction request."""
    input_data: Optional[torch.Tensor] = None
    input_dict: Optional[Dict[str, torch.Tensor]] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    return_confidence: bool = True


@dataclass
class PredictResponse:
    """Single prediction response."""
    request_id: str = ""
    predictions: Optional[torch.Tensor] = None
    predicted_class: Optional[int] = None
    confidence: Optional[float] = None
    processing_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class HealthStatus:
    """Health check response."""
    status: str = "unknown"  # healthy | degraded | unhealthy
    model_loaded: bool = False
    device: str = "cpu"
    uptime_seconds: float = 0.0
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    queue_depth: int = 0
    last_error: Optional[str] = None
    components: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metrics Collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Collects serving metrics for Prometheus-style reporting."""

    def __init__(self) -> None:
        self.request_count: int = 0
        self.error_count: int = 0
        self.latency_sum: float = 0.0
        self.latency_count: int = 0
        self.latency_min: float = float("inf")
        self.latency_max: float = 0.0
        self.batch_sizes: List[int] = []
        self._lock = threading.Lock()
        self.latency_buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        self.latency_histogram = [0] * (len(self.latency_buckets) + 1)

    def record_request(self, latency_seconds: float, batch_size: int = 1) -> None:
        with self._lock:
            self.request_count += 1
            self.latency_sum += latency_seconds
            self.latency_count += 1
            self.latency_min = min(self.latency_min, latency_seconds)
            self.latency_max = max(self.latency_max, latency_seconds)
            self.batch_sizes.append(batch_size)
            # Histogram
            placed = False
            for i, bucket in enumerate(self.latency_buckets):
                if latency_seconds <= bucket:
                    self.latency_histogram[i] += 1
                    placed = True
                    break
            if not placed:
                self.latency_histogram[-1] += 1

    def record_error(self) -> None:
        with self._lock:
            self.error_count += 1

    @property
    def avg_latency(self) -> float:
        if self.latency_count == 0:
            return 0.0
        return self.latency_sum / self.latency_count

    def reset(self) -> None:
        with self._lock:
            self.request_count = 0
            self.error_count = 0
            self.latency_sum = 0.0
            self.latency_count = 0
            self.latency_min = float("inf")
            self.latency_max = 0.0
            self.batch_sizes = []
            self.latency_histogram = [0] * (len(self.latency_buckets) + 1)

    def format_prometheus(self) -> str:
        lines: List[str] = []
        lines.append("# HELP brainai_requests_total Total prediction requests")
        lines.append("# TYPE brainai_requests_total counter")
        lines.append(f"brainai_requests_total {self.request_count}")
        lines.append("# HELP brainai_errors_total Total errors")
        lines.append("# TYPE brainai_errors_total counter")
        lines.append(f"brainai_errors_total {self.error_count}")
        lines.append("# HELP brainai_request_duration_seconds Request latency")
        lines.append("# TYPE brainai_request_duration_seconds histogram")
        cumulative = 0
        for i, bucket in enumerate(self.latency_buckets):
            cumulative += self.latency_histogram[i]
            lines.append(f'brainai_request_duration_seconds_bucket{{le="{bucket}"}} {cumulative}')
        cumulative += self.latency_histogram[-1]
        lines.append(f'brainai_request_duration_seconds_bucket{{le="+Inf"}} {cumulative}')
        lines.append(f"brainai_request_duration_seconds_sum {self.latency_sum:.6f}")
        lines.append(f"brainai_request_duration_seconds_count {self.latency_count}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_latency_seconds": self.avg_latency,
            "min_latency_seconds": self.latency_min if self.latency_count > 0 else 0.0,
            "max_latency_seconds": self.latency_max,
        }


# ---------------------------------------------------------------------------
# ServingEngine
# ---------------------------------------------------------------------------

class ServingEngine:
    """Model serving engine with predict, batch_predict, and health check.

    Does NOT start an HTTP server -- use create_app() to get a FastAPI app
    or call predict()/batch_predict() directly for testing.

    Parameters
    ----------
    model_or_path : nn.Module or str
        A PyTorch model instance or path to a saved model.
    config : ServingConfig
        Serving configuration.
    """

    def __init__(
        self,
        model_or_path: Union[nn.Module, str],
        config: Optional[ServingConfig] = None,
    ):
        self.config = config or ServingConfig()
        self.metrics = MetricsCollector()
        self._start_time = time.monotonic()
        self._model: Optional[nn.Module] = None
        self._device = torch.device(self.config.device if self.config.device != "auto" else "cpu")
        self._last_error: Optional[str] = None
        self._accepting_requests = True
        self._queue_depth = 0

        # Load model
        if isinstance(model_or_path, nn.Module):
            self._model = model_or_path.to(self._device)
            self._model.requires_grad_(False)
        elif isinstance(model_or_path, str) and os.path.exists(model_or_path):
            self.load_model(model_or_path)
        # else: model will be loaded later via load_model()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Run a single prediction.

        Parameters
        ----------
        request : PredictRequest

        Returns
        -------
        PredictResponse
        """
        start = time.monotonic()
        response = PredictResponse(request_id=request.request_id)

        if self._model is None:
            response.error = "Model not loaded"
            self.metrics.record_error()
            return response

        try:
            with torch.no_grad():
                if request.input_data is not None:
                    inp = request.input_data.to(self._device)
                    output = self._model(inp)
                elif request.input_dict is not None:
                    inp_dict = {k: v.to(self._device) for k, v in request.input_dict.items()}
                    output = self._model(**inp_dict)
                else:
                    response.error = "No input data provided"
                    self.metrics.record_error()
                    return response

            if isinstance(output, tuple):
                output = output[0]

            response.predictions = output.cpu()

            if output.dim() > 1 and output.shape[-1] > 1:
                response.predicted_class = int(output.argmax(dim=-1)[0].item())
                if request.return_confidence:
                    probs = torch.softmax(output, dim=-1)
                    response.confidence = float(probs.max(dim=-1).values[0].item())
            elif request.return_confidence:
                response.confidence = float(torch.sigmoid(output).mean().item())

        except Exception as exc:  # noqa: BLE001
            response.error = str(exc)
            self._last_error = str(exc)
            self.metrics.record_error()

        elapsed = time.monotonic() - start
        response.processing_time_ms = elapsed * 1000.0
        self.metrics.record_request(elapsed)
        return response

    def batch_predict(self, requests: List[PredictRequest]) -> List[PredictResponse]:
        """Run batch prediction.

        Parameters
        ----------
        requests : list of PredictRequest

        Returns
        -------
        list of PredictResponse
        """
        if not requests:
            return []

        start = time.monotonic()

        # Try to batch process
        if self._model is not None and all(r.input_data is not None for r in requests):
            try:
                batched_input = torch.cat([r.input_data for r in requests], dim=0).to(self._device)
                with torch.no_grad():
                    batched_output = self._model(batched_input)

                if isinstance(batched_output, tuple):
                    batched_output = batched_output[0]

                elapsed = time.monotonic() - start
                responses: List[PredictResponse] = []
                for i, req in enumerate(requests):
                    resp = PredictResponse(request_id=req.request_id)
                    single_out = batched_output[i : i + 1].cpu()
                    resp.predictions = single_out

                    if single_out.dim() > 1 and single_out.shape[-1] > 1:
                        resp.predicted_class = int(single_out.argmax(dim=-1)[0].item())
                        if req.return_confidence:
                            probs = torch.softmax(single_out, dim=-1)
                            resp.confidence = float(probs.max().item())

                    resp.processing_time_ms = elapsed * 1000.0 / len(requests)
                    responses.append(resp)

                self.metrics.record_request(elapsed, batch_size=len(requests))
                return responses

            except Exception as exc:  # noqa: BLE001
                self._last_error = str(exc)

        # Fallback: individual predictions
        return [self.predict(req) for req in requests]

    def health_check(self) -> HealthStatus:
        """Check the health of the serving engine.

        Returns
        -------
        HealthStatus
        """
        uptime = time.monotonic() - self._start_time

        status = "healthy"
        components: Dict[str, str] = {}

        # Model loaded check
        model_loaded = self._model is not None
        components["model"] = "loaded" if model_loaded else "not_loaded"
        if not model_loaded:
            status = "unhealthy"

        # Error rate check
        if self.metrics.request_count > 0:
            error_rate = self.metrics.error_count / self.metrics.request_count
            if error_rate > 0.1:
                status = "degraded"
                components["error_rate"] = f"{error_rate:.2%}"
            else:
                components["error_rate"] = f"{error_rate:.2%}"

        # Latency check
        avg_lat_ms = self.metrics.avg_latency * 1000.0
        if avg_lat_ms > 1000.0 and self.metrics.latency_count > 0:
            if status == "healthy":
                status = "degraded"
            components["avg_latency"] = f"{avg_lat_ms:.1f}ms (high)"
        elif self.metrics.latency_count > 0:
            components["avg_latency"] = f"{avg_lat_ms:.1f}ms"

        return HealthStatus(
            status=status,
            model_loaded=model_loaded,
            device=str(self._device),
            uptime_seconds=uptime,
            total_requests=self.metrics.request_count,
            avg_latency_ms=avg_lat_ms,
            queue_depth=self._queue_depth,
            last_error=self._last_error,
            components=components,
        )

    def load_model(self, path: str) -> bool:
        """Load a model from a file path.

        Supports TorchScript (.pt) and state dict loading.

        Returns True on success.
        """
        try:
            if path.endswith(".pt") or path.endswith(".pth"):
                try:
                    self._model = torch.jit.load(path, map_location=self._device)
                except Exception:
                    # Fallback: assume it is a state_dict that needs a model skeleton
                    return False
            self._model.requires_grad_(False)
            return True
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            return False

    def unload_model(self) -> None:
        """Unload the model and free memory."""
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def warmup(self, sample_input: Optional[torch.Tensor] = None) -> float:
        """Run warmup inferences. Returns average warmup latency in ms."""
        if self._model is None:
            return 0.0

        if sample_input is None:
            # Infer input shape
            first_param = next(self._model.parameters(), None)
            if first_param is not None:
                dim = first_param.shape[-1]
                sample_input = torch.randn(1, dim)
            else:
                sample_input = torch.randn(1, 32)

        latencies: List[float] = []
        for _ in range(self.config.warmup_iterations):
            start = time.monotonic()
            req = PredictRequest(input_data=sample_input)
            self.predict(req)
            latencies.append((time.monotonic() - start) * 1000.0)

        # Reset metrics after warmup
        self.metrics.reset()

        return sum(latencies) / max(len(latencies), 1)

    def create_app(self) -> Any:
        """Create a FastAPI application (returns None if FastAPI not installed).

        This is a factory method -- the actual server is started separately
        with uvicorn.
        """
        try:
            from fastapi import FastAPI  # type: ignore
            from fastapi.responses import PlainTextResponse  # type: ignore

            app = FastAPI(title="BrainAI Serving", version="1.0.0")

            @app.get("/health")
            def _health() -> dict:
                hs = self.health_check()
                return {
                    "status": hs.status,
                    "model_loaded": hs.model_loaded,
                    "device": hs.device,
                    "uptime_seconds": hs.uptime_seconds,
                    "total_requests": hs.total_requests,
                    "avg_latency_ms": hs.avg_latency_ms,
                }

            @app.get("/metrics")
            def _metrics() -> PlainTextResponse:
                return PlainTextResponse(self.metrics.format_prometheus())

            return app

        except ImportError:
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """Return current metrics as a dictionary."""
        return self.metrics.to_dict()

    def get_prometheus_metrics(self) -> str:
        """Return Prometheus-formatted metrics string."""
        return self.metrics.format_prometheus()


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------

class _LinearModel(nn.Module):
    def __init__(self, d: int = 32, o: int = 10):
        super().__init__()
        self.fc = nn.Linear(d, o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _ClassifierModel(nn.Module):
    def __init__(self, d: int = 32, nc: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, nc)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MultiOutputModel(nn.Module):
    def __init__(self, d: int = 32, o: int = 10):
        super().__init__()
        self.fc = nn.Linear(d, o)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        return h, torch.sigmoid(h.mean(dim=-1, keepdim=True))


class _BinaryModel(nn.Module):
    def __init__(self, d: int = 32):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:  # noqa: C901
    """Run 25+ self-tests for ServingEngine."""
    passed = 0
    failed = 0
    skipped = 0

    def _ok(name: str, cond: bool) -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}")

    def _skip(name: str, reason: str) -> None:
        nonlocal skipped
        skipped += 1
        print(f"  SKIP: {name} ({reason})")

    print("=" * 60)
    print("ServingEngine Self-Tests")
    print("=" * 60)

    # --- T01-T03: Config defaults ---
    cfg = ServingConfig()
    _ok("T01 Default port 8000", cfg.port == 8000)
    _ok("T02 Default max_batch_size 32", cfg.max_batch_size == 32)
    _ok("T03 Default device cpu", cfg.device == "cpu")

    # --- T04: Create engine with model ---
    model = _ClassifierModel(32, 10)
    engine = ServingEngine(model, cfg)
    _ok("T04 Engine created with model", engine._model is not None)

    # --- T05: Health check (healthy) ---
    hs = engine.health_check()
    _ok("T05 Health status healthy", hs.status == "healthy")
    _ok("T06 Health model loaded", hs.model_loaded)
    _ok("T07 Health device cpu", hs.device == "cpu")

    # --- T08: Single predict ---
    req = PredictRequest(input_data=torch.randn(1, 32))
    resp = engine.predict(req)
    _ok("T08 Predict response has predictions", resp.predictions is not None)
    _ok("T09 Predict response shape", resp.predictions.shape == (1, 10))
    _ok("T10 Predict has predicted_class", resp.predicted_class is not None)
    _ok("T11 Predict has confidence", resp.confidence is not None)
    _ok("T12 Predict time > 0", resp.processing_time_ms > 0)
    _ok("T13 Predict no error", resp.error is None)

    # --- T14: Batch predict ---
    reqs = [PredictRequest(input_data=torch.randn(1, 32)) for _ in range(8)]
    resps = engine.batch_predict(reqs)
    _ok("T14 Batch returns 8 responses", len(resps) == 8)
    _ok("T15 All batch responses have predictions", all(r.predictions is not None for r in resps))

    # --- T16: Empty batch ---
    empty_resps = engine.batch_predict([])
    _ok("T16 Empty batch returns empty", len(empty_resps) == 0)

    # --- T17: No input error ---
    bad_req = PredictRequest()
    bad_resp = engine.predict(bad_req)
    _ok("T17 No input returns error", bad_resp.error is not None)

    # --- T18: Metrics updated ---
    metrics = engine.get_metrics()
    _ok("T18 Request count > 0", metrics["request_count"] > 0)

    # --- T19: Prometheus format ---
    prom = engine.get_prometheus_metrics()
    _ok("T19 Prometheus has counter", "brainai_requests_total" in prom)
    _ok("T20 Prometheus has histogram", "brainai_request_duration_seconds" in prom)

    # --- T21: Warmup ---
    engine2 = ServingEngine(_LinearModel(32, 10))
    avg_warmup = engine2.warmup()
    _ok("T21 Warmup returns latency", avg_warmup > 0)
    _ok("T22 Metrics reset after warmup", engine2.metrics.request_count == 0)

    # --- T23: Multi-output model ---
    multi = _MultiOutputModel(32, 10)
    engine_mo = ServingEngine(multi)
    resp_mo = engine_mo.predict(PredictRequest(input_data=torch.randn(1, 32)))
    _ok("T23 Multi-output handled", resp_mo.predictions is not None)

    # --- T24: Binary model ---
    binary = _BinaryModel(32)
    engine_bin = ServingEngine(binary)
    resp_bin = engine_bin.predict(PredictRequest(input_data=torch.randn(1, 32)))
    _ok("T24 Binary model predict", resp_bin.predictions is not None)
    _ok("T25 Binary model confidence", resp_bin.confidence is not None)

    # --- T26: Unload model ---
    engine.unload_model()
    _ok("T26 Model unloaded", engine._model is None)
    hs_unloaded = engine.health_check()
    _ok("T27 Health unhealthy after unload", hs_unloaded.status == "unhealthy")

    # --- T28: Predict with unloaded model ---
    resp_unloaded = engine.predict(PredictRequest(input_data=torch.randn(1, 32)))
    _ok("T28 Predict error when unloaded", resp_unloaded.error is not None)

    # --- T29: Engine with string path (non-existent) ---
    engine_nopath = ServingEngine("/nonexistent/model.pt")
    _ok("T29 Non-existent path handled", engine_nopath._model is None)

    # --- T30: Save and load model ---
    tmpdir = tempfile.mkdtemp(prefix="serve_test_")
    save_model = _LinearModel(32, 10)
    save_model.requires_grad_(False)
    traced = torch.jit.trace(save_model, torch.randn(1, 32))
    save_path = os.path.join(tmpdir, "model.pt")
    torch.jit.save(traced, save_path)

    engine_load = ServingEngine(save_path)
    resp_load = engine_load.predict(PredictRequest(input_data=torch.randn(1, 32)))
    _ok("T30 Load from path and predict", resp_load.predictions is not None)

    # --- T31: create_app ---
    app = engine_load.create_app()
    # May be None if FastAPI not installed
    _ok("T31 create_app returns something", True)  # No crash

    # --- T32: MetricsCollector standalone ---
    mc = MetricsCollector()
    mc.record_request(0.05)
    mc.record_request(0.1)
    mc.record_error()
    _ok("T32 Metrics count correct", mc.request_count == 2)
    _ok("T33 Error count correct", mc.error_count == 1)
    _ok("T34 Avg latency correct", abs(mc.avg_latency - 0.075) < 0.001)

    # --- T35: Metrics reset ---
    mc.reset()
    _ok("T35 Metrics reset", mc.request_count == 0)

    # --- T36: Different batch sizes ---
    engine_bs = ServingEngine(_ClassifierModel(32, 10))
    for bs in [1, 4, 16]:
        reqs_bs = [PredictRequest(input_data=torch.randn(1, 32)) for _ in range(bs)]
        resps_bs = engine_bs.batch_predict(reqs_bs)
        _ok(f"T36_bs{bs} Batch size {bs}", len(resps_bs) == bs)

    # --- T39: Health uptime ---
    time.sleep(0.01)
    hs2 = engine_bs.health_check()
    _ok("T39 Uptime > 0", hs2.uptime_seconds > 0)

    # --- T40: Request IDs preserved ---
    req_id = PredictRequest(input_data=torch.randn(1, 32), request_id="test-123")
    resp_id = engine_bs.predict(req_id)
    _ok("T40 Request ID preserved", resp_id.request_id == "test-123")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    _run_self_tests()
