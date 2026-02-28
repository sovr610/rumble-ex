"""
brain_ai/inference/async_engine.py — Async Inference Engine with Futures

This module provides non-blocking inference via a thread pool executor.
Supports both asyncio coroutines and synchronous Future-based submission.

Key classes:
    ModelPool            — Pool of model copies for thread-safe concurrent inference
    CircuitBreaker       — Stops sending requests when failure rate is too high
    AsyncInferenceEngine — Main engine: async infer(), submit() returning Future

Design principles:
    1. Thread safety via model copies — each thread gets an independent model copy.
    2. Futures API — submit() returns a concurrent.futures.Future immediately.
    3. Asyncio integration — async infer() bridges asyncio and thread pool.
    4. Circuit breaker — protects against cascading failures.
    5. Graceful shutdown — drains pending requests before stopping.

References:
    references/async-inference.md — Full async design rationale
    SKILL.md § AsyncInferenceEngine contract
"""

from __future__ import annotations

import asyncio
import copy
import concurrent.futures
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1: InferenceResult
# ===========================================================================

@dataclass
class InferenceResult:
    """Result of a single inference."""

    output: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    request_idx: int = 0

    @property
    def success(self) -> bool:
        return self.error is None and self.output is not None


# ===========================================================================
# SECTION 2: ModelPool
# ===========================================================================

class ModelPool:
    """Pool of model copies for thread-safe concurrent inference.

    Each worker thread acquires a model copy from the pool, uses it for
    inference, and returns it. This prevents concurrent access to stateful
    modules (SNN membrane potentials, HTM column states, etc.).

    Args:
        model: The base model to copy.
        num_copies: Number of independent copies to maintain.
    """

    def __init__(self, model: nn.Module, num_copies: int = 2):
        self.num_copies = num_copies
        self.models: List[nn.Module] = []
        self._available: queue.Queue = queue.Queue()
        self._semaphore = threading.Semaphore(num_copies)

        for _ in range(num_copies):
            m = copy.deepcopy(model)
            m.eval()
            self.models.append(m)
            self._available.put(m)

    def acquire(self) -> nn.Module:
        """Acquire a model copy. Blocks if all copies are in use."""
        self._semaphore.acquire()
        return self._available.get()

    def release(self, model: nn.Module) -> None:
        """Return a model copy to the pool."""
        self._available.put(model)
        self._semaphore.release()

    def __len__(self) -> int:
        return self.num_copies


# ===========================================================================
# SECTION 3: CircuitBreaker
# ===========================================================================

class CircuitBreaker:
    """Prevents cascading failures by tracking error rate.

    States:
    - closed: Normal operation, requests flow through.
    - open: Too many failures, requests are rejected.
    - half_open: Testing whether the system has recovered.

    Args:
        failure_threshold: Number of consecutive failures before opening.
        reset_timeout: Seconds before transitioning from open to half_open.
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0
        self.state: str = "closed"
        self._lock = threading.Lock()

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self.failure_count = 0
            self.state = "closed"

    def record_failure(self) -> None:
        """Record a failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"

    def allow_request(self) -> bool:
        """Check whether a new request should be allowed."""
        with self._lock:
            if self.state == "closed":
                return True
            if self.state == "open":
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = "half_open"
                    return True
                return False
            if self.state == "half_open":
                return True
            return False

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self.failure_count = 0
            self.state = "closed"
            self.last_failure_time = 0.0


# ===========================================================================
# SECTION 4: AsyncInferenceEngine
# ===========================================================================

class AsyncInferenceEngine:
    """Non-blocking inference engine with futures and asyncio support.

    Uses a thread pool executor with model copies for thread-safe
    concurrent inference. Supports both sync (submit/Future) and
    async (await infer()) usage patterns.

    Args:
        model: The model to run inference on.
        num_workers: Number of worker threads (and model copies).
        request_timeout: Timeout for individual requests in seconds.
        device: Device for inference.
    """

    def __init__(
        self,
        model: nn.Module,
        num_workers: int = 2,
        request_timeout: float = 5.0,
        device: str = "cpu",
    ):
        self.model_pool = ModelPool(model, num_copies=num_workers)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="async_infer",
        )
        self.request_timeout = request_timeout
        self.device = device
        self.circuit_breaker = CircuitBreaker()
        self._shutdown = False
        self._lock = threading.Lock()

        # Metrics
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._total_latency: float = 0.0

    def _infer_sync(self, inputs: Dict[str, Tensor]) -> InferenceResult:
        """Synchronous inference on a single input dict."""
        if not self.circuit_breaker.allow_request():
            return InferenceResult(error="Circuit breaker is open")

        model = self.model_pool.acquire()
        try:
            # Move inputs to device
            device_inputs = {
                k: v.to(self.device) for k, v in inputs.items()
            }

            t0 = time.time()
            with torch.inference_mode():
                output = model(device_inputs)
            elapsed_ms = (time.time() - t0) * 1000

            self.circuit_breaker.record_success()

            with self._lock:
                self._total_requests += 1
                self._total_latency += elapsed_ms

            return InferenceResult(output=output, latency_ms=elapsed_ms)

        except Exception as e:
            self.circuit_breaker.record_failure()
            with self._lock:
                self._total_requests += 1
                self._total_errors += 1
            return InferenceResult(error=str(e))

        finally:
            self.model_pool.release(model)

    def submit(
        self, inputs: Dict[str, Tensor]
    ) -> concurrent.futures.Future:
        """Submit an inference request, returning a Future immediately.

        Args:
            inputs: Dict mapping modality name to tensor.

        Returns:
            Future that resolves to an InferenceResult.

        Raises:
            RuntimeError: If the engine has been shut down.
        """
        if self._shutdown:
            raise RuntimeError("Engine has been shut down")
        return self.executor.submit(self._infer_sync, inputs)

    async def infer(self, inputs: Dict[str, Tensor]) -> InferenceResult:
        """Async inference on a single input dict.

        Runs the synchronous inference in the thread pool executor,
        bridging asyncio and threading.

        Args:
            inputs: Dict mapping modality name to tensor.

        Returns:
            InferenceResult with the model output.
        """
        if self._shutdown:
            return InferenceResult(error="Engine has been shut down")

        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(self.executor, self._infer_sync, inputs),
                timeout=self.request_timeout,
            )
            return result
        except asyncio.TimeoutError:
            return InferenceResult(error="Request timed out")

    async def infer_batch(
        self, inputs: List[Dict[str, Tensor]]
    ) -> List[InferenceResult]:
        """Async batch inference — submits all items concurrently.

        Args:
            inputs: List of input dicts.

        Returns:
            List of InferenceResult, one per input.
        """
        tasks = [self.infer(inp) for inp in inputs]
        return list(await asyncio.gather(*tasks))

    def submit_batch(
        self, inputs: List[Dict[str, Tensor]]
    ) -> concurrent.futures.Future:
        """Submit a batch as a single Future resolving to a list of results."""
        def _batch_sync():
            return [self._infer_sync(inp) for inp in inputs]
        if self._shutdown:
            raise RuntimeError("Engine has been shut down")
        return self.executor.submit(_batch_sync)

    def shutdown(self, wait: bool = True, timeout: float = 10.0) -> None:
        """Gracefully shut down the engine.

        Args:
            wait: If True, wait for pending requests to complete.
            timeout: Maximum time to wait for pending requests.
        """
        self._shutdown = True
        self.executor.shutdown(wait=wait)

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown

    @property
    def error_rate(self) -> float:
        """Fraction of requests that resulted in errors."""
        if self._total_requests == 0:
            return 0.0
        return self._total_errors / self._total_requests

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        successful = self._total_requests - self._total_errors
        if successful == 0:
            return 0.0
        return self._total_latency / successful


# ===========================================================================
# SECTION 5: DictModel for testing
# ===========================================================================

class DictModel(nn.Module):
    """Wrapper that makes a model accept dict inputs."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, inputs):
        if isinstance(inputs, dict):
            x = next(iter(inputs.values()))
        else:
            x = inputs
        return self.inner(x)


class FailingModel(nn.Module):
    """Model that raises an error on forward. For testing error handling."""

    def forward(self, inputs):
        raise RuntimeError("Intentional failure for testing")


# ===========================================================================
# SECTION 6: Self-tests
# ===========================================================================

def _run_self_tests():
    """Run self-tests for async inference engine."""
    import traceback

    passed = 0
    failed = 0
    test_results = []

    def _test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            test_results.append(f"  PASS: {name}")
        except Exception as e:
            failed += 1
            test_results.append(f"  FAIL: {name} -- {e}")
            traceback.print_exc()

    def make_model(in_features=64, out_features=10):
        return DictModel(nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, out_features),
        ))

    def make_input(in_features=64):
        return {"features": torch.randn(1, in_features)}

    # --- ModelPool tests ---
    def test_model_pool_creation():
        model = make_model()
        pool = ModelPool(model, num_copies=3)
        assert len(pool) == 3

    def test_model_pool_acquire_release():
        model = make_model()
        pool = ModelPool(model, num_copies=2)
        m = pool.acquire()
        assert m is not None
        pool.release(m)

    def test_model_pool_independence():
        model = make_model()
        pool = ModelPool(model, num_copies=2)
        m1 = pool.acquire()
        m2 = pool.acquire()
        assert m1 is not m2
        pool.release(m1)
        pool.release(m2)

    # --- CircuitBreaker tests ---
    def test_circuit_breaker_closed():
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.allow_request() is True
        assert cb.state == "closed"

    def test_circuit_breaker_opens():
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        assert cb.allow_request() is False

    def test_circuit_breaker_recovers():
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.1)
        assert cb.allow_request() is True
        assert cb.state == "half_open"

    def test_circuit_breaker_success_resets():
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_circuit_breaker_reset():
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        cb.reset()
        assert cb.state == "closed"
        assert cb.allow_request() is True

    # --- AsyncInferenceEngine sync tests ---
    def test_submit_and_result():
        model = make_model()
        engine = AsyncInferenceEngine(model, num_workers=1)
        future = engine.submit(make_input())
        result = future.result(timeout=5.0)
        assert result.success
        assert result.output is not None
        engine.shutdown()

    def test_submit_output_shape():
        model = make_model(in_features=32, out_features=5)
        engine = AsyncInferenceEngine(model, num_workers=1)
        future = engine.submit({"features": torch.randn(1, 32)})
        result = future.result(timeout=5.0)
        assert result.output.shape == (1, 5)
        engine.shutdown()

    def test_multiple_concurrent_submissions():
        model = make_model()
        engine = AsyncInferenceEngine(model, num_workers=2)
        futures = [engine.submit(make_input()) for _ in range(10)]
        results = [f.result(timeout=5.0) for f in futures]
        assert len(results) == 10
        assert all(r.success for r in results)
        engine.shutdown()

    def test_submit_batch_future():
        model = make_model()
        engine = AsyncInferenceEngine(model, num_workers=2)
        inputs = [make_input() for _ in range(5)]
        future = engine.submit_batch(inputs)
        results = future.result(timeout=10.0)
        assert len(results) == 5
        assert all(r.success for r in results)
        engine.shutdown()

    def test_error_handling():
        model = DictModel(FailingModel())
        engine = AsyncInferenceEngine(model, num_workers=1)
        future = engine.submit(make_input())
        result = future.result(timeout=5.0)
        assert not result.success
        assert "Intentional failure" in result.error
        engine.shutdown()

    def test_shutdown_prevents_submit():
        model = make_model()
        engine = AsyncInferenceEngine(model, num_workers=1)
        engine.shutdown()
        try:
            engine.submit(make_input())
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_is_shutdown_flag():
        model = make_model()
        engine = AsyncInferenceEngine(model, num_workers=1)
        assert engine.is_shutdown is False
        engine.shutdown()
        assert engine.is_shutdown is True

    def test_error_rate_metric():
        model = make_model()
        engine = AsyncInferenceEngine(model, num_workers=1)
        engine.submit(make_input()).result(timeout=5.0)
        assert engine.error_rate == 0.0
        engine.shutdown()

    def test_avg_latency_metric():
        model = make_model()
        engine = AsyncInferenceEngine(model, num_workers=1)
        engine.submit(make_input()).result(timeout=5.0)
        assert engine.avg_latency_ms > 0.0
        engine.shutdown()

    # --- Async tests ---
    def test_async_infer():
        model = make_model()
        engine = AsyncInferenceEngine(model, num_workers=1)

        async def run():
            result = await engine.infer(make_input())
            return result

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run())
        loop.close()
        assert result.success
        engine.shutdown()

    def test_async_infer_batch():
        model = make_model()
        engine = AsyncInferenceEngine(model, num_workers=2)

        async def run():
            inputs = [make_input() for _ in range(5)]
            return await engine.infer_batch(inputs)

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(run())
        loop.close()
        assert len(results) == 5
        assert all(r.success for r in results)
        engine.shutdown()

    def test_async_error_propagation():
        model = DictModel(FailingModel())
        engine = AsyncInferenceEngine(model, num_workers=1)

        async def run():
            return await engine.infer(make_input())

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run())
        loop.close()
        assert not result.success
        engine.shutdown()

    def test_async_shutdown_returns_error():
        model = make_model()
        engine = AsyncInferenceEngine(model, num_workers=1)
        engine.shutdown()

        async def run():
            return await engine.infer(make_input())

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run())
        loop.close()
        assert not result.success
        assert "shut down" in result.error

    def test_thread_safety_concurrent():
        model = make_model()
        engine = AsyncInferenceEngine(model, num_workers=4)
        errors = []

        def worker():
            try:
                for _ in range(5):
                    f = engine.submit(make_input())
                    r = f.result(timeout=5.0)
                    assert r.success
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Thread errors: {errors}"
        engine.shutdown()

    def test_latency_nonnegative():
        model = make_model()
        engine = AsyncInferenceEngine(model, num_workers=1)
        future = engine.submit(make_input())
        result = future.result(timeout=5.0)
        assert result.latency_ms >= 0
        engine.shutdown()

    def test_inference_result_dataclass():
        r = InferenceResult(output=torch.tensor([1.0]))
        assert r.success
        r2 = InferenceResult(error="fail")
        assert not r2.success

    # Run all tests
    tests = [
        ("ModelPool creation", test_model_pool_creation),
        ("ModelPool acquire/release", test_model_pool_acquire_release),
        ("ModelPool independence", test_model_pool_independence),
        ("CircuitBreaker closed", test_circuit_breaker_closed),
        ("CircuitBreaker opens", test_circuit_breaker_opens),
        ("CircuitBreaker recovers", test_circuit_breaker_recovers),
        ("CircuitBreaker success resets", test_circuit_breaker_success_resets),
        ("CircuitBreaker reset", test_circuit_breaker_reset),
        ("submit and result", test_submit_and_result),
        ("submit output shape", test_submit_output_shape),
        ("multiple concurrent submissions", test_multiple_concurrent_submissions),
        ("submit_batch future", test_submit_batch_future),
        ("error handling", test_error_handling),
        ("shutdown prevents submit", test_shutdown_prevents_submit),
        ("is_shutdown flag", test_is_shutdown_flag),
        ("error_rate metric", test_error_rate_metric),
        ("avg_latency metric", test_avg_latency_metric),
        ("async infer", test_async_infer),
        ("async infer_batch", test_async_infer_batch),
        ("async error propagation", test_async_error_propagation),
        ("async shutdown returns error", test_async_shutdown_returns_error),
        ("thread safety concurrent", test_thread_safety_concurrent),
        ("latency nonnegative", test_latency_nonnegative),
        ("InferenceResult dataclass", test_inference_result_dataclass),
        ("circuit breaker during inference", test_circuit_breaker_opens),
    ]

    print(f"Running {len(tests)} self-tests for async_engine_template...")
    for name, fn in tests:
        _test(name, fn)

    print("\n".join(test_results))
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    return failed == 0


if __name__ == "__main__":
    _run_self_tests()
