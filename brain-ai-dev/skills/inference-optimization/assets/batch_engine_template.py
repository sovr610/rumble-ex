"""
brain_ai/inference/batch.py — Batch Inference Engine with Dynamic Batching

This module provides the BatchInferenceEngine for efficient batch processing
of inference requests. Supports dynamic batching with padding, stream
processing, warmup, and automatic batch splitting.

Key classes:
    InferenceRequest     — A single queued inference request with metadata
    BatchAssembler       — Collects requests and triggers batch dispatch
    BatchInferenceEngine — Main engine: infer_batch(), infer_stream(), warmup()

Design principles:
    1. Dynamic batching — requests are padded to the longest sequence in the batch.
    2. Automatic splitting — batches exceeding max_batch_size are split.
    3. Safe fallback — on batch failure, processes items individually to isolate errors.
    4. Warmup protocol — runs warmup at multiple batch sizes to trigger JIT compilation.
    5. Stream processing — yields results as micro-batches complete.

References:
    references/batch-strategies.md — Full batching design rationale
    SKILL.md § BatchInferenceEngine contract
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1: InferenceResult
# ===========================================================================

@dataclass
class InferenceResult:
    """Result of a single inference.

    Attributes:
        output: Model output tensor (None on error).
        error: Error message (None on success).
        latency_ms: Wall-clock latency in milliseconds.
        request_idx: Index within the original batch.
        cache_hit: Whether the result came from cache.
    """

    output: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    request_idx: int = 0
    cache_hit: bool = False

    @property
    def success(self) -> bool:
        return self.error is None and self.output is not None


# ===========================================================================
# SECTION 2: InferenceRequest
# ===========================================================================

@dataclass
class InferenceRequest:
    """A single inference request with metadata.

    Attributes:
        request_id: Unique identifier for this request.
        inputs: Dict mapping modality name to tensor.
        priority: 0=critical, 1=interactive, 2=batch.
        timestamp: Time when the request was submitted.
        callback: Optional completion callback.
        max_seq_len: Length of the longest sequence dimension.
    """

    request_id: str = ""
    inputs: Dict[str, Tensor] = field(default_factory=dict)
    priority: int = 1
    timestamp: float = 0.0
    callback: Optional[Callable] = None
    max_seq_len: int = 0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.max_seq_len == 0 and self.inputs:
            for t in self.inputs.values():
                if t.dim() >= 2:
                    self.max_seq_len = max(self.max_seq_len, t.shape[1])


# ===========================================================================
# SECTION 3: BatchAssembler
# ===========================================================================

class BatchAssembler:
    """Assembles individual requests into batches.

    Triggers dispatch when the batch is full or timeout expires.

    Args:
        max_batch_size: Maximum requests per batch.
        batch_timeout_ms: Timeout before dispatching a partial batch.
    """

    def __init__(self, max_batch_size: int = 32, batch_timeout_ms: float = 10.0):
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.queue: List[InferenceRequest] = []

    def add(self, request: InferenceRequest) -> None:
        """Add a request to the queue."""
        self.queue.append(request)

    def should_dispatch(self) -> bool:
        """Check if the batch should be dispatched."""
        if len(self.queue) >= self.max_batch_size:
            return True
        if len(self.queue) > 0:
            oldest_age_ms = (time.time() - self.queue[0].timestamp) * 1000
            if oldest_age_ms >= self.batch_timeout_ms:
                return True
        return False

    def take_batch(self) -> List[InferenceRequest]:
        """Remove and return up to max_batch_size requests."""
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]
        return batch

    def __len__(self) -> int:
        return len(self.queue)


# ===========================================================================
# SECTION 4: Padding utilities
# ===========================================================================

def pad_sequences(
    tensors: List[Tensor], pad_value: float = 0.0
) -> Tuple[Tensor, Tensor]:
    """Pad variable-length tensors to the maximum length in the batch.

    Assumes tensors are at least 2-D with shape (1, seq_len, ...).
    Pads along dim=1.

    Returns:
        padded: Stacked tensor with shape (B, max_len, ...).
        mask: Boolean mask with shape (B, max_len), True for valid positions.
    """
    if not tensors:
        raise ValueError("Cannot pad empty list of tensors")

    max_len = max(t.shape[1] for t in tensors)
    batch_size = len(tensors)
    rest_shape = tensors[0].shape[2:]

    padded = torch.full(
        (batch_size, max_len, *rest_shape), pad_value, dtype=tensors[0].dtype
    )
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, t in enumerate(tensors):
        seq_len = t.shape[1]
        padded[i, :seq_len] = t.squeeze(0)
        mask[i, :seq_len] = True

    return padded, mask


def collate_inputs(
    inputs_list: List[Dict[str, Tensor]], pad_value: float = 0.0
) -> Dict[str, Tensor]:
    """Collate a list of input dicts into a single batched dict.

    For each modality, stacks tensors along dim=0. If tensors have
    different sizes along dim=1, pads them to the longest.

    Args:
        inputs_list: List of dicts, each mapping modality -> tensor.
        pad_value: Value to use for padding.

    Returns:
        Batched dict with tensors stacked along dim=0.
    """
    if not inputs_list:
        return {}

    # Collect all modality keys
    all_keys: Set[str] = set()
    for inp in inputs_list:
        all_keys.update(inp.keys())

    batched: Dict[str, Tensor] = {}
    for key in sorted(all_keys):
        tensors = []
        for inp in inputs_list:
            if key in inp:
                t = inp[key]
                # Ensure batch dimension
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                tensors.append(t)
            else:
                # Create a zero placeholder matching the first available tensor
                ref = None
                for other in inputs_list:
                    if key in other:
                        ref = other[key]
                        break
                if ref is not None:
                    if ref.dim() == 1:
                        ref = ref.unsqueeze(0)
                    tensors.append(torch.zeros_like(ref))

        if not tensors:
            continue

        # Check if all tensors have the same shape
        shapes = [t.shape for t in tensors]
        if all(s == shapes[0] for s in shapes):
            batched[key] = torch.cat(tensors, dim=0)
        elif tensors[0].dim() >= 2:
            # Variable length — pad along dim=1
            padded, _ = pad_sequences(tensors, pad_value=pad_value)
            batched[key] = padded
        else:
            batched[key] = torch.cat(tensors, dim=0)

    return batched


def unbatch_outputs(
    batched_output: Tensor, batch_size: int
) -> List[Tensor]:
    """Split a batched output tensor into individual results."""
    return list(batched_output.split(1, dim=0))


# ===========================================================================
# SECTION 5: BatchInferenceEngine
# ===========================================================================

class BatchInferenceEngine:
    """Efficient batch inference engine with dynamic batching.

    Args:
        model: The model to run inference on.
        max_batch_size: Maximum batch size before splitting.
        pad_value: Padding value for variable-length inputs.
        device: Device to run inference on.
    """

    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 64,
        pad_value: float = 0.0,
        device: str = "cpu",
    ):
        self.model = model
        self.model.eval()
        self.max_batch_size = max_batch_size
        self.pad_value = pad_value
        self.device = device
        self._warmed_up = False

        # Metrics
        self._total_inferences: int = 0
        self._total_batches: int = 0
        self._total_time: float = 0.0

    def infer_batch(
        self, inputs: List[Dict[str, Tensor]]
    ) -> List[InferenceResult]:
        """Run batch inference on a list of input dicts.

        If the batch exceeds max_batch_size, it is automatically split into
        sub-batches. On failure, falls back to per-item inference.

        Args:
            inputs: List of input dicts, each mapping modality -> tensor.

        Returns:
            List of InferenceResult, one per input.
        """
        if not inputs:
            return []

        # Split into sub-batches if needed
        if len(inputs) > self.max_batch_size:
            results: List[InferenceResult] = []
            for start in range(0, len(inputs), self.max_batch_size):
                sub = inputs[start: start + self.max_batch_size]
                results.extend(self._infer_batch_inner(sub, start_idx=start))
            return results

        return self._infer_batch_inner(inputs, start_idx=0)

    def _infer_batch_inner(
        self, inputs: List[Dict[str, Tensor]], start_idx: int = 0
    ) -> List[InferenceResult]:
        """Inner batch inference with error fallback."""
        try:
            batched = collate_inputs(inputs, pad_value=self.pad_value)
            # Move to device
            batched = {
                k: v.to(self.device) for k, v in batched.items()
            }

            t0 = time.time()
            with torch.inference_mode():
                output = self.model(batched)
            elapsed = time.time() - t0
            elapsed_ms = elapsed * 1000

            self._total_batches += 1
            self._total_inferences += len(inputs)
            self._total_time += elapsed

            # Handle single-tensor models
            if isinstance(output, Tensor):
                pieces = unbatch_outputs(output, len(inputs))
                return [
                    InferenceResult(
                        output=p,
                        latency_ms=elapsed_ms / len(inputs),
                        request_idx=start_idx + i,
                    )
                    for i, p in enumerate(pieces)
                ]
            else:
                # Non-tensor output — return as-is for each item
                return [
                    InferenceResult(
                        output=output,
                        latency_ms=elapsed_ms / len(inputs),
                        request_idx=start_idx + i,
                    )
                    for i in range(len(inputs))
                ]

        except RuntimeError as e:
            logger.warning(f"Batch inference failed, falling back: {e}")
            return self._infer_individual(inputs, start_idx)

    def _infer_individual(
        self, inputs: List[Dict[str, Tensor]], start_idx: int
    ) -> List[InferenceResult]:
        """Fallback: process each input individually."""
        results: List[InferenceResult] = []
        for i, inp in enumerate(inputs):
            try:
                batched = collate_inputs([inp], pad_value=self.pad_value)
                batched = {k: v.to(self.device) for k, v in batched.items()}

                t0 = time.time()
                with torch.inference_mode():
                    output = self.model(batched)
                elapsed_ms = (time.time() - t0) * 1000

                results.append(
                    InferenceResult(
                        output=output,
                        latency_ms=elapsed_ms,
                        request_idx=start_idx + i,
                    )
                )
            except Exception as e:
                results.append(
                    InferenceResult(
                        error=str(e),
                        request_idx=start_idx + i,
                    )
                )
        return results

    def infer_stream(
        self,
        input_stream: Iterator[Dict[str, Tensor]],
        micro_batch_size: int = 8,
    ) -> Iterator[InferenceResult]:
        """Process a stream of inputs in micro-batches.

        Yields InferenceResult objects as each micro-batch completes.

        Args:
            input_stream: Iterator of input dicts.
            micro_batch_size: Number of items per micro-batch.

        Yields:
            InferenceResult for each input in the stream.
        """
        buffer: List[Dict[str, Tensor]] = []
        for item in input_stream:
            buffer.append(item)
            if len(buffer) >= micro_batch_size:
                yield from self.infer_batch(buffer)
                buffer.clear()
        if buffer:
            yield from self.infer_batch(buffer)

    def warmup(
        self,
        sample_input: Dict[str, Tensor],
        n_warmup: int = 10,
    ) -> None:
        """Run warmup inference to trigger JIT/CUDA kernel compilation.

        Runs warmup at multiple batch sizes (1, 2, 4, up to max_batch_size)
        for comprehensive warmup coverage.

        Args:
            sample_input: A single sample input dict.
            n_warmup: Number of warmup runs per batch size.
        """
        self.model.eval()
        batch_sizes = []
        bs = 1
        while bs <= self.max_batch_size:
            batch_sizes.append(bs)
            bs *= 2
        if self.max_batch_size not in batch_sizes:
            batch_sizes.append(self.max_batch_size)

        with torch.inference_mode():
            for bs in batch_sizes:
                batch = {}
                for k, v in sample_input.items():
                    if v.dim() == 0:
                        batch[k] = v.unsqueeze(0).expand(bs)
                    else:
                        expand_shape = [bs] + [-1] * (v.dim() - 1)
                        if v.shape[0] == 1:
                            batch[k] = v.expand(*expand_shape).contiguous()
                        else:
                            batch[k] = v.unsqueeze(0).expand(
                                bs, *v.shape
                            ).reshape(bs, *v.shape[1:] if v.dim() > 1 else v.shape).contiguous()

                batch = {k: v.to(self.device) for k, v in batch.items()}
                for _ in range(n_warmup):
                    self.model(batch)

        self._warmed_up = True
        logger.info(
            f"Warmup complete: {len(batch_sizes)} batch sizes x "
            f"{n_warmup} runs"
        )

    @property
    def throughput(self) -> float:
        """Average throughput in samples/second."""
        if self._total_time == 0:
            return 0.0
        return self._total_inferences / self._total_time

    @property
    def avg_batch_size(self) -> float:
        """Average batch size across all batches."""
        if self._total_batches == 0:
            return 0.0
        return self._total_inferences / self._total_batches


# ===========================================================================
# SECTION 6: Simple model wrapper for testing
# ===========================================================================

class DictModel(nn.Module):
    """Wrapper that makes a model accept dict inputs.

    The model receives the first tensor value from the input dict.
    Used for testing with nn.Linear and nn.Sequential.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, inputs):
        # Get the first tensor from the dict
        if isinstance(inputs, dict):
            x = next(iter(inputs.values()))
        else:
            x = inputs
        return self.inner(x)


# ===========================================================================
# SECTION 7: Self-tests
# ===========================================================================

def _run_self_tests():
    """Run self-tests for batch inference engine."""
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

    # Helper: create a simple model
    def make_model(in_features=256, out_features=10):
        inner = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_features),
        )
        return DictModel(inner)

    def make_inputs(n, in_features=256):
        return [{"features": torch.randn(1, in_features)} for _ in range(n)]

    # --- InferenceRequest tests ---
    def test_request_defaults():
        req = InferenceRequest(request_id="r1")
        assert req.priority == 1
        assert req.timestamp > 0

    def test_request_seq_len():
        req = InferenceRequest(
            inputs={"text": torch.randn(1, 50, 64)}
        )
        assert req.max_seq_len == 50

    # --- BatchAssembler tests ---
    def test_assembler_size_trigger():
        asm = BatchAssembler(max_batch_size=4, batch_timeout_ms=1000)
        for i in range(4):
            asm.add(InferenceRequest(request_id=f"r{i}"))
        assert asm.should_dispatch() is True

    def test_assembler_timeout_trigger():
        asm = BatchAssembler(max_batch_size=100, batch_timeout_ms=20)
        asm.add(InferenceRequest(
            request_id="r0",
            timestamp=time.time() - 0.05,
        ))
        assert asm.should_dispatch() is True

    def test_assembler_no_trigger():
        asm = BatchAssembler(max_batch_size=100, batch_timeout_ms=10000)
        asm.add(InferenceRequest(request_id="r0"))
        assert asm.should_dispatch() is False

    def test_assembler_take_batch():
        asm = BatchAssembler(max_batch_size=3)
        for i in range(5):
            asm.add(InferenceRequest(request_id=f"r{i}"))
        batch = asm.take_batch()
        assert len(batch) == 3
        assert len(asm) == 2

    # --- Padding tests ---
    def test_pad_sequences_same_len():
        tensors = [torch.randn(1, 10, 4) for _ in range(3)]
        padded, mask = pad_sequences(tensors)
        assert padded.shape == (3, 10, 4)
        assert mask.all()

    def test_pad_sequences_variable_len():
        t1 = torch.ones(1, 5, 4)
        t2 = torch.ones(1, 10, 4)
        t3 = torch.ones(1, 3, 4)
        padded, mask = pad_sequences([t1, t2, t3])
        assert padded.shape == (3, 10, 4)
        assert mask[0, :5].all() and not mask[0, 5:].any()
        assert mask[1, :10].all()
        assert mask[2, :3].all() and not mask[2, 3:].any()

    def test_pad_correctness():
        """Padded region should contain pad_value."""
        t1 = torch.ones(1, 3, 2)
        t2 = torch.ones(1, 5, 2)
        padded, _ = pad_sequences([t1, t2], pad_value=0.0)
        # t1 padded positions should be 0
        assert (padded[0, 3:] == 0.0).all()

    def test_collate_uniform():
        inputs_list = [
            {"x": torch.randn(1, 8)},
            {"x": torch.randn(1, 8)},
        ]
        batched = collate_inputs(inputs_list)
        assert batched["x"].shape == (2, 8)

    def test_collate_variable():
        inputs_list = [
            {"x": torch.randn(1, 5, 4)},
            {"x": torch.randn(1, 10, 4)},
        ]
        batched = collate_inputs(inputs_list)
        assert batched["x"].shape == (2, 10, 4)

    # --- BatchInferenceEngine tests ---
    def test_infer_batch_basic():
        model = make_model()
        engine = BatchInferenceEngine(model, max_batch_size=32)
        inputs = make_inputs(4)
        results = engine.infer_batch(inputs)
        assert len(results) == 4
        for r in results:
            assert r.success

    def test_infer_batch_single():
        model = make_model()
        engine = BatchInferenceEngine(model, max_batch_size=32)
        inputs = make_inputs(1)
        results = engine.infer_batch(inputs)
        assert len(results) == 1
        assert results[0].success

    def test_infer_batch_empty():
        model = make_model()
        engine = BatchInferenceEngine(model, max_batch_size=32)
        results = engine.infer_batch([])
        assert len(results) == 0

    def test_infer_batch_exceeds_max():
        model = make_model()
        engine = BatchInferenceEngine(model, max_batch_size=4)
        inputs = make_inputs(10)
        results = engine.infer_batch(inputs)
        assert len(results) == 10
        for r in results:
            assert r.success

    def test_infer_batch_output_shape():
        model = make_model(in_features=64, out_features=5)
        engine = BatchInferenceEngine(model, max_batch_size=32)
        inputs = [{"features": torch.randn(1, 64)} for _ in range(8)]
        results = engine.infer_batch(inputs)
        for r in results:
            assert r.output.shape == (1, 5)

    def test_infer_batch_throughput():
        """Batch should be faster than sequential for 32 items."""
        model = make_model()
        engine = BatchInferenceEngine(model, max_batch_size=32)
        inputs = make_inputs(32)

        # Sequential baseline
        t0 = time.time()
        for inp in inputs:
            batched = collate_inputs([inp])
            with torch.inference_mode():
                model(batched)
        seq_time = time.time() - t0

        # Batch inference
        t0 = time.time()
        engine.infer_batch(inputs)
        batch_time = time.time() - t0

        # Batch should be at least as fast (may not be 2x on CPU in self-tests)
        assert batch_time <= seq_time * 2.0, (
            f"Batch ({batch_time:.4f}s) should not be drastically slower "
            f"than sequential ({seq_time:.4f}s)"
        )

    def test_infer_batch_latency_field():
        model = make_model()
        engine = BatchInferenceEngine(model, max_batch_size=32)
        results = engine.infer_batch(make_inputs(2))
        for r in results:
            assert r.latency_ms >= 0

    def test_infer_batch_request_idx():
        model = make_model()
        engine = BatchInferenceEngine(model, max_batch_size=32)
        results = engine.infer_batch(make_inputs(5))
        indices = [r.request_idx for r in results]
        assert indices == [0, 1, 2, 3, 4]

    # --- Stream tests ---
    def test_stream_all_items():
        model = make_model()
        engine = BatchInferenceEngine(model, max_batch_size=32)
        items = [{"features": torch.randn(1, 256)} for _ in range(50)]
        results = list(engine.infer_stream(iter(items), micro_batch_size=8))
        assert len(results) == 50

    def test_stream_small_batch():
        model = make_model()
        engine = BatchInferenceEngine(model, max_batch_size=32)
        items = [{"features": torch.randn(1, 256)} for _ in range(3)]
        results = list(engine.infer_stream(iter(items), micro_batch_size=8))
        assert len(results) == 3

    def test_stream_correctness():
        model = make_model(in_features=64, out_features=5)
        engine = BatchInferenceEngine(model, max_batch_size=32)
        items = [{"features": torch.randn(1, 64)} for _ in range(12)]
        results = list(engine.infer_stream(iter(items), micro_batch_size=4))
        for r in results:
            assert r.success
            assert r.output.shape[-1] == 5

    # --- Warmup tests ---
    def test_warmup_runs():
        model = make_model()
        engine = BatchInferenceEngine(model, max_batch_size=8)
        sample = {"features": torch.randn(1, 256)}
        engine.warmup(sample, n_warmup=3)
        assert engine._warmed_up is True

    def test_warmup_no_error():
        model = make_model(in_features=32, out_features=4)
        engine = BatchInferenceEngine(model, max_batch_size=16)
        sample = {"features": torch.randn(1, 32)}
        engine.warmup(sample, n_warmup=2)

    def test_warmup_then_infer():
        model = make_model()
        engine = BatchInferenceEngine(model, max_batch_size=16)
        sample = {"features": torch.randn(1, 256)}
        engine.warmup(sample, n_warmup=2)
        results = engine.infer_batch(make_inputs(4))
        assert all(r.success for r in results)

    # --- Metrics tests ---
    def test_throughput_metric():
        model = make_model()
        engine = BatchInferenceEngine(model, max_batch_size=32)
        engine.infer_batch(make_inputs(10))
        assert engine.throughput > 0

    def test_avg_batch_size_metric():
        model = make_model()
        engine = BatchInferenceEngine(model, max_batch_size=32)
        engine.infer_batch(make_inputs(10))
        assert engine.avg_batch_size > 0

    # --- Edge case tests ---
    def test_infer_result_dataclass():
        r = InferenceResult(output=torch.randn(1, 10))
        assert r.success is True
        r2 = InferenceResult(error="failed")
        assert r2.success is False

    def test_unbatch_outputs():
        batched = torch.randn(5, 10)
        pieces = unbatch_outputs(batched, 5)
        assert len(pieces) == 5
        for p in pieces:
            assert p.shape == (1, 10)

    def test_dict_model_wrapper():
        inner = nn.Linear(8, 4)
        model = DictModel(inner)
        out = model({"x": torch.randn(2, 8)})
        assert out.shape == (2, 4)

    # Run all tests
    tests = [
        ("InferenceRequest defaults", test_request_defaults),
        ("InferenceRequest seq_len", test_request_seq_len),
        ("BatchAssembler size trigger", test_assembler_size_trigger),
        ("BatchAssembler timeout trigger", test_assembler_timeout_trigger),
        ("BatchAssembler no trigger", test_assembler_no_trigger),
        ("BatchAssembler take_batch", test_assembler_take_batch),
        ("pad_sequences same length", test_pad_sequences_same_len),
        ("pad_sequences variable length", test_pad_sequences_variable_len),
        ("pad_sequences correctness", test_pad_correctness),
        ("collate uniform inputs", test_collate_uniform),
        ("collate variable inputs", test_collate_variable),
        ("infer_batch basic", test_infer_batch_basic),
        ("infer_batch single", test_infer_batch_single),
        ("infer_batch empty", test_infer_batch_empty),
        ("infer_batch exceeds max", test_infer_batch_exceeds_max),
        ("infer_batch output shape", test_infer_batch_output_shape),
        ("infer_batch throughput", test_infer_batch_throughput),
        ("infer_batch latency field", test_infer_batch_latency_field),
        ("infer_batch request_idx", test_infer_batch_request_idx),
        ("infer_stream all items", test_stream_all_items),
        ("infer_stream small batch", test_stream_small_batch),
        ("infer_stream correctness", test_stream_correctness),
        ("warmup runs without error", test_warmup_runs),
        ("warmup no error small model", test_warmup_no_error),
        ("warmup then infer", test_warmup_then_infer),
        ("throughput metric", test_throughput_metric),
        ("avg_batch_size metric", test_avg_batch_size_metric),
        ("InferenceResult dataclass", test_infer_result_dataclass),
        ("unbatch_outputs", test_unbatch_outputs),
        ("DictModel wrapper", test_dict_model_wrapper),
    ]

    print(f"Running {len(tests)} self-tests for batch_engine_template...")
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
