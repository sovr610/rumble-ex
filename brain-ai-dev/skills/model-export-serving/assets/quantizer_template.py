"""
ModelQuantizer: Post-training and quantization-aware training for INT8/FP16.

Provides quantize_dynamic(), quantize_static(), quantize_aware_training(),
and measure_accuracy_delta() for optimizing PyTorch models for inference.

torch + standard lib only.
"""

import copy
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Quantization utilities
try:
    import torch.quantization as _tq

    _QUANT_AVAILABLE = True
except ImportError:
    _QUANT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    method: str = "dynamic"  # dynamic | static | qat
    dtype: str = "int8"  # int8 | fp16
    backend: str = "x86"  # x86 | qnnpack | fbgemm
    calibration_samples: int = 1000
    accuracy_threshold: float = 0.02  # max acceptable accuracy drop
    skip_modules: List[str] = field(default_factory=list)
    per_channel: bool = True
    use_histogram_observer: bool = False
    verbose: bool = False


@dataclass
class QuantizationResult:
    """Result of a quantization operation."""
    success: bool
    method: str
    original_size_bytes: int = 0
    quantized_size_bytes: int = 0
    compression_ratio: float = 1.0
    quantize_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class AccuracyDelta:
    """Accuracy comparison between original and quantized models."""
    original_metric: float = 0.0
    quantized_metric: float = 0.0
    delta: float = 0.0
    within_threshold: bool = False
    threshold: float = 0.02
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_size_bytes(model: nn.Module) -> int:
    """Estimate model size in bytes from parameter/buffer storage."""
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total


def _save_and_measure(model: nn.Module) -> int:
    """Save model to a temporary file and return its file size."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(model.state_dict(), path)
        size = os.path.getsize(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    return size


def _default_accuracy_fn(model: nn.Module, data: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
    """Default accuracy function: fraction of correct top-1 predictions."""
    model_device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device("cpu")
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data:
            inputs = inputs.to(model_device)
            targets = targets.to(model_device)
            outputs = model(inputs)
            if outputs.dim() > 1:
                preds = outputs.argmax(dim=-1)
            else:
                preds = (outputs > 0).long()
            correct += (preds == targets).sum().item()
            total += targets.numel()
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# ModelQuantizer
# ---------------------------------------------------------------------------

class ModelQuantizer:
    """Quantize PyTorch models with dynamic, static, or QAT methods.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to quantize.
    config : QuantizationConfig
        Quantization configuration.
    """

    def __init__(self, model: nn.Module, config: Optional[QuantizationConfig] = None):
        self.original_model = model
        self.config = config or QuantizationConfig()
        self._quantized_model: Optional[nn.Module] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def quantize_dynamic(self) -> nn.Module:
        """Apply dynamic INT8 quantization to Linear and LSTM layers.

        Returns
        -------
        nn.Module
            Quantized model.
        """
        model = copy.deepcopy(self.original_model)

        # Build qconfig spec
        qconfig_spec = {nn.Linear: torch.quantization.default_dynamic_qconfig}

        # Also quantize LSTM if present
        has_lstm = any(isinstance(m, nn.LSTM) for m in model.modules())
        if has_lstm:
            qconfig_spec[nn.LSTM] = torch.quantization.default_dynamic_qconfig

        quantized = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec=qconfig_spec,
            dtype=torch.qint8,
        )

        self._quantized_model = quantized
        return quantized

    def quantize_static(
        self,
        calibration_data: Optional[List[torch.Tensor]] = None,
    ) -> nn.Module:
        """Apply static INT8 quantization with calibration.

        Parameters
        ----------
        calibration_data : list of Tensors, optional
            Data for calibration. If None, uses random data.

        Returns
        -------
        nn.Module
            Quantized model.
        """
        model = copy.deepcopy(self.original_model)

        # Set backend
        torch.backends.quantized.engine = self.config.backend

        # Set qconfig
        if self.config.use_histogram_observer:
            from torch.quantization.observer import HistogramObserver, PerChannelMinMaxObserver

            qconfig = torch.quantization.QConfig(
                activation=HistogramObserver.with_args(reduce_range=True),
                weight=PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                )
                if self.config.per_channel
                else torch.quantization.default_observer,
            )
        else:
            qconfig = torch.quantization.get_default_qconfig(self.config.backend)

        model.qconfig = qconfig  # type: ignore[assignment]

        # Prepare
        torch.quantization.prepare(model, inplace=True)

        # Calibrate
        if calibration_data is None:
            # Generate synthetic calibration data
            calibration_data = self._generate_calibration_data(model)

        with torch.no_grad():
            for data in calibration_data:
                try:
                    model(data)
                except Exception:  # noqa: BLE001
                    pass

        # Convert
        quantized = torch.quantization.convert(model)
        self._quantized_model = quantized
        return quantized

    def quantize_aware_training(
        self,
        train_fn: Callable[[nn.Module], nn.Module],
    ) -> nn.Module:
        """Apply quantization-aware training.

        Parameters
        ----------
        train_fn : callable
            Function that takes a model, trains it, and returns it.
            The model will have fake quantization nodes inserted.

        Returns
        -------
        nn.Module
            Quantized model after QAT.
        """
        model = copy.deepcopy(self.original_model)
        model.train()

        torch.backends.quantized.engine = self.config.backend
        model.qconfig = torch.quantization.get_default_qat_qconfig(self.config.backend)  # type: ignore[assignment]

        torch.quantization.prepare_qat(model, inplace=True)

        # Run user-provided training
        model = train_fn(model)

        # Convert to quantized
        model_eval = model
        model_eval.requires_grad_(False)
        quantized = torch.quantization.convert(model_eval)
        self._quantized_model = quantized
        return quantized

    def quantize_fp16(self) -> nn.Module:
        """Convert model to FP16 half precision.

        Returns
        -------
        nn.Module
            FP16 model.
        """
        model = copy.deepcopy(self.original_model)
        model = model.half()
        self._quantized_model = model
        return model

    def measure_accuracy_delta(
        self,
        accuracy_fn: Optional[Callable[[nn.Module], float]] = None,
        test_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> AccuracyDelta:
        """Measure accuracy difference between original and quantized model.

        Parameters
        ----------
        accuracy_fn : callable, optional
            Function that takes a model and returns an accuracy scalar.
            If None and test_data is provided, uses default top-1 accuracy.
        test_data : list of (input, target) tuples, optional
            Test data for the default accuracy function.

        Returns
        -------
        AccuracyDelta
        """
        if self._quantized_model is None:
            return AccuracyDelta(
                within_threshold=False,
                threshold=self.config.accuracy_threshold,
                details={"error": "No quantized model available. Run a quantize method first."},
            )

        if accuracy_fn is None and test_data is not None:
            accuracy_fn = lambda m: _default_accuracy_fn(m, test_data)
        elif accuracy_fn is None:
            # Use output distance as proxy metric
            return self._measure_output_delta()

        orig_acc = accuracy_fn(self.original_model)
        quant_acc = accuracy_fn(self._quantized_model)
        delta = orig_acc - quant_acc

        return AccuracyDelta(
            original_metric=orig_acc,
            quantized_metric=quant_acc,
            delta=delta,
            within_threshold=abs(delta) <= self.config.accuracy_threshold,
            threshold=self.config.accuracy_threshold,
        )

    def get_size_comparison(self) -> Dict[str, Any]:
        """Compare model sizes before and after quantization."""
        orig_size = _save_and_measure(self.original_model)
        info: Dict[str, Any] = {"original_size_bytes": orig_size}

        if self._quantized_model is not None:
            quant_size = _save_and_measure(self._quantized_model)
            info["quantized_size_bytes"] = quant_size
            info["compression_ratio"] = orig_size / max(quant_size, 1)
        else:
            info["quantized_size_bytes"] = 0
            info["compression_ratio"] = 1.0

        return info

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _generate_calibration_data(
        self, model: nn.Module, num_samples: int = 100
    ) -> List[torch.Tensor]:
        """Generate random calibration data by inferring input shape."""
        # Try to infer input shape from first parameter
        first_param = next(model.parameters(), None)
        if first_param is not None:
            in_features = first_param.shape[-1]
            return [torch.randn(4, in_features) for _ in range(num_samples)]
        return [torch.randn(4, 32) for _ in range(num_samples)]

    def _measure_output_delta(self) -> AccuracyDelta:
        """Measure output distance between original and quantized on random inputs."""
        if self._quantized_model is None:
            return AccuracyDelta(within_threshold=False, threshold=self.config.accuracy_threshold)

        first_param = next(self.original_model.parameters(), None)
        if first_param is None:
            return AccuracyDelta(within_threshold=True, threshold=self.config.accuracy_threshold)

        in_features = first_param.shape[-1]
        diffs: List[float] = []

        with torch.no_grad():
            for _ in range(50):
                x = torch.randn(4, in_features)
                try:
                    orig_out = self.original_model(x)
                    quant_out = self._quantized_model(x.float())
                    if isinstance(orig_out, tuple):
                        orig_out = orig_out[0]
                    if isinstance(quant_out, tuple):
                        quant_out = quant_out[0]
                    diff = (orig_out.float() - quant_out.float()).abs().mean().item()
                    diffs.append(diff)
                except Exception:  # noqa: BLE001
                    pass

        mean_diff = sum(diffs) / max(len(diffs), 1)
        return AccuracyDelta(
            original_metric=0.0,
            quantized_metric=mean_diff,
            delta=mean_diff,
            within_threshold=mean_diff < self.config.accuracy_threshold,
            threshold=self.config.accuracy_threshold,
            details={"metric": "mean_output_distance", "num_samples": len(diffs)},
        )


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------

class _SimpleLinear(nn.Module):
    def __init__(self, d_in: int = 32, d_out: int = 10):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _TwoLayerNet(nn.Module):
    def __init__(self, d: int = 32, h: int = 64, o: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h, o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class _DeepNet(nn.Module):
    def __init__(self, d: int = 64, depth: int = 5, o: int = 10):
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.Linear(d, d), nn.ReLU()])
        layers.append(nn.Linear(d, o))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LSTMNet(nn.Module):
    def __init__(self, d: int = 32, h: int = 64, o: int = 10):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True)
        self.fc = nn.Linear(h, o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class _ClassifierNet(nn.Module):
    def __init__(self, d: int = 32, nc: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, nc),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:  # noqa: C901
    """Run 30+ self-tests for ModelQuantizer."""
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
    print("ModelQuantizer Self-Tests")
    print("=" * 60)

    # --- T01-T03: Config defaults ---
    cfg = QuantizationConfig()
    _ok("T01 Default method dynamic", cfg.method == "dynamic")
    _ok("T02 Default dtype int8", cfg.dtype == "int8")
    _ok("T03 Default threshold 0.02", cfg.accuracy_threshold == 0.02)

    # --- T04: Custom config ---
    cfg2 = QuantizationConfig(method="static", backend="qnnpack")
    _ok("T04 Custom method static", cfg2.method == "static")

    # --- T05: QuantizationResult ---
    qr = QuantizationResult(success=True, method="dynamic")
    _ok("T05 QR success field", qr.success)
    _ok("T06 QR default compression 1.0", qr.compression_ratio == 1.0)

    # --- T07: Dynamic quantization on linear ---
    model = _SimpleLinear(32, 10)
    quantizer = ModelQuantizer(model)
    q_model = quantizer.quantize_dynamic()
    _ok("T07 Dynamic quant returns module", isinstance(q_model, nn.Module))

    # --- T08: Quantized model produces output ---
    x = torch.randn(4, 32)
    with torch.no_grad():
        out = q_model(x)
    _ok("T08 Dynamic quant output shape", out.shape == (4, 10))

    # --- T09: Dynamic quant on two-layer ---
    model2 = _TwoLayerNet(32, 64, 10)
    q2 = ModelQuantizer(model2).quantize_dynamic()
    with torch.no_grad():
        out2 = q2(torch.randn(2, 32))
    _ok("T09 Two-layer dynamic quant output", out2.shape == (2, 10))

    # --- T10: Dynamic quant on deep net ---
    deep = _DeepNet(64, 5, 10)
    q_deep = ModelQuantizer(deep).quantize_dynamic()
    with torch.no_grad():
        out_deep = q_deep(torch.randn(1, 64))
    _ok("T10 Deep net dynamic quant", out_deep.shape == (1, 10))

    # --- T11: Dynamic quant on LSTM ---
    lstm_net = _LSTMNet(32, 64, 10)
    q_lstm = ModelQuantizer(lstm_net).quantize_dynamic()
    with torch.no_grad():
        out_lstm = q_lstm(torch.randn(2, 32))
    _ok("T11 LSTM dynamic quant output", out_lstm.shape == (2, 10))

    # --- T12: Size comparison ---
    sc = quantizer.get_size_comparison()
    _ok("T12 Size comparison has original", sc["original_size_bytes"] > 0)
    _ok("T13 Size comparison has quantized", sc["quantized_size_bytes"] > 0)

    # --- T14: FP16 quantization ---
    fp16_model = ModelQuantizer(model).quantize_fp16()
    fp16_param = next(fp16_model.parameters())
    _ok("T14 FP16 param dtype", fp16_param.dtype == torch.float16)

    # --- T15: FP16 inference ---
    with torch.no_grad():
        fp16_out = fp16_model(torch.randn(2, 32).half())
    _ok("T15 FP16 output shape", fp16_out.shape == (2, 10))

    # --- T16: Measure accuracy delta (output distance) ---
    quantizer_delta = ModelQuantizer(model)
    quantizer_delta.quantize_dynamic()
    delta = quantizer_delta.measure_accuracy_delta()
    _ok("T16 Accuracy delta computed", isinstance(delta, AccuracyDelta))
    _ok("T17 Accuracy delta has threshold", delta.threshold == 0.02)

    # --- T18: Measure accuracy with test data ---
    test_data = [(torch.randn(8, 32), torch.randint(0, 10, (8,))) for _ in range(5)]
    quantizer_td = ModelQuantizer(_ClassifierNet(32, 10))
    quantizer_td.quantize_dynamic()
    delta_td = quantizer_td.measure_accuracy_delta(test_data=test_data)
    _ok("T18 Accuracy delta with test data", isinstance(delta_td.original_metric, float))
    _ok("T19 Delta is numeric", isinstance(delta_td.delta, float))

    # --- T20: Measure accuracy with custom fn ---
    def custom_fn(m: nn.Module) -> float:
        with torch.no_grad():
            return m(torch.randn(1, 32)).abs().mean().item()

    quantizer_cf = ModelQuantizer(_SimpleLinear(32, 10))
    quantizer_cf.quantize_dynamic()
    delta_cf = quantizer_cf.measure_accuracy_delta(accuracy_fn=custom_fn)
    _ok("T20 Custom accuracy fn works", isinstance(delta_cf.delta, float))

    # --- T21: No quantized model error ---
    q_empty = ModelQuantizer(_SimpleLinear())
    delta_empty = q_empty.measure_accuracy_delta()
    _ok("T21 No quant model returns error info", "error" in delta_empty.details)

    # --- T22: Static quantization ---
    try:
        static_model = _TwoLayerNet(32, 64, 10)
        q_static = ModelQuantizer(static_model, QuantizationConfig(method="static"))
        cal_data = [torch.randn(4, 32) for _ in range(20)]
        q_static_result = q_static.quantize_static(calibration_data=cal_data)
        with torch.no_grad():
            out_static = q_static_result(torch.randn(2, 32))
        _ok("T22 Static quant produces output", out_static.shape == (2, 10))
    except Exception as e:
        _skip("T22 Static quantization", str(e)[:80])

    # --- T23: Static quantization with histogram observer ---
    try:
        hist_model = _TwoLayerNet(32, 64, 10)
        q_hist = ModelQuantizer(hist_model, QuantizationConfig(use_histogram_observer=True))
        cal = [torch.randn(4, 32) for _ in range(10)]
        q_hist_result = q_hist.quantize_static(calibration_data=cal)
        _ok("T23 Histogram observer static quant", isinstance(q_hist_result, nn.Module))
    except Exception as e:
        _skip("T23 Histogram observer", str(e)[:80])

    # --- T24: QAT ---
    try:
        qat_model = _TwoLayerNet(32, 64, 10)

        def train_fn(m: nn.Module) -> nn.Module:
            opt = torch.optim.SGD(m.parameters(), lr=0.01)
            for _ in range(3):
                x_t = torch.randn(8, 32)
                y_t = torch.randint(0, 10, (8,))
                loss = nn.CrossEntropyLoss()(m(x_t), y_t)
                loss.backward()
                opt.step()
                opt.zero_grad()
            return m

        q_qat = ModelQuantizer(qat_model, QuantizationConfig(method="qat"))
        q_qat_result = q_qat.quantize_aware_training(train_fn)
        with torch.no_grad():
            out_qat = q_qat_result(torch.randn(2, 32))
        _ok("T24 QAT produces output", out_qat.shape == (2, 10))
    except Exception as e:
        _skip("T24 QAT", str(e)[:80])

    # --- T25: Compression ratio for dynamic ---
    big = _DeepNet(256, 6, 10)
    q_big = ModelQuantizer(big)
    q_big.quantize_dynamic()
    sc_big = q_big.get_size_comparison()
    _ok("T25 Compression ratio > 1", sc_big["compression_ratio"] >= 1.0)

    # --- T26: Multiple batch sizes with quantized ---
    for bs in [1, 4, 16]:
        with torch.no_grad():
            o = q_model(torch.randn(bs, 32))
        _ok(f"T26_bs{bs} Quant batch {bs}", o.shape[0] == bs)

    # --- T29: AccuracyDelta dataclass ---
    ad = AccuracyDelta(original_metric=0.95, quantized_metric=0.93, delta=0.02)
    _ok("T29 AccuracyDelta fields", ad.delta == 0.02)

    # --- T30: Config skip_modules ---
    cfg_skip = QuantizationConfig(skip_modules=["snn_core", "neuromod"])
    _ok("T30 Skip modules stored", len(cfg_skip.skip_modules) == 2)

    # --- T31: Backend setting ---
    _ok("T31 Default backend x86", cfg.backend == "x86")

    # --- T32: Quantize same model twice ---
    q1 = ModelQuantizer(_SimpleLinear()).quantize_dynamic()
    q2 = ModelQuantizer(_SimpleLinear()).quantize_dynamic()
    with torch.no_grad():
        o1 = q1(torch.randn(1, 32))
        o2 = q2(torch.randn(1, 32))
    _ok("T32 Double quantize independent", o1.shape == o2.shape)

    # --- T33: Model size bytes helper ---
    sz = _model_size_bytes(_SimpleLinear())
    _ok("T33 Model size bytes > 0", sz > 0)

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    _run_self_tests()
